"""Overlapping whole-slide prediction with bounded accumulators."""

from __future__ import annotations

import ctypes
import gc
import multiprocessing as mp
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from spagc.histology.dataset import CTransPathPatchDataset, _validate_h5_metadata
from spagc.histology.features import (
    extract_ctranspath_to_memory,
    load_ctranspath,
    tissue_mask_from_he,
)
from spagc.histology.memory import format_bytes, get_effective_available_memory_bytes
from spagc.histology.metadata import (
    _label_shape_from_level0_slide,
    counts_with_background,
)
from spagc.histology.model import load_segmentation_model
from spagc.histology.slides import open_slide, validate_slide_level

_GIB = 1024**3
TILE_SIZE = 224


def _resolve_output_geometry(
    he_tif_path: str,
    configured_label_shape: tuple[int, int] | None,
    region_size: int = 28,
    *,
    slide_name: str | None = None,
    verbose: bool = True,
) -> tuple[tuple[int, int], tuple[int, int]]:
    configured = None
    if configured_label_shape is not None:
        configured = tuple(map(int, configured_label_shape))
        if configured[0] <= 0 or configured[1] <= 0:
            raise ValueError(f"Invalid configured label_shape: {configured}")
    image_shape = None
    image_error = None
    try:
        (image_resolved, image_shape) = _label_shape_from_level0_slide(
            he_tif_path, region_size=region_size, tile_size=TILE_SIZE
        )
    except Exception as exc:
        image_resolved = None
        image_error = exc
    if configured is not None:
        resolved = configured
        he_shape = image_shape or tuple(n * region_size for n in resolved)
    elif image_resolved is not None:
        resolved = image_resolved
        he_shape = image_shape
    else:
        raise image_error
    if verbose:
        prefix = f"{slide_name}: " if slide_name else ""
        expected_bytes = resolved[0] * resolved[1] * np.dtype(np.uint16).itemsize
        if (
            configured is not None
            and image_resolved is not None
            and (configured != image_resolved)
        ):
            print(
                f"[shape check] {prefix}using configured label_shape={configured}; image-derived={image_resolved}; level0_HE={he_shape}; expected_each={format_bytes(expected_bytes)}"
            )
        elif configured is not None and image_resolved is None:
            print(
                f"[shape check] {prefix}using configured label_shape={configured}; image_shape_unavailable={image_error}; expected_each={format_bytes(expected_bytes)}"
            )
        else:
            print(
                f"[shape check] {prefix}label_shape={resolved}; level0_HE={he_shape}; expected_each={format_bytes(expected_bytes)}"
            )
    return (resolved, he_shape)


def gaussian_weight_np(size: int, sigma_scale: float = 0.125) -> np.ndarray:
    """Generate center-heavy overlap weights."""
    sigma = size * sigma_scale
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2
    g = np.exp(-(ax**2) / (2 * sigma**2))
    w = np.outer(g, g)
    w = np.maximum(w / w.max(), 0.05)
    return w.astype(np.float32)


def gaussian_weight_region_np(
    tile_size: int = 224, region_size: int = 28, sigma_scale: float = 0.125
) -> np.ndarray:
    if tile_size % region_size != 0:
        raise ValueError("tile size must be divisible by region size")
    grid = tile_size // region_size
    w = gaussian_weight_np(size=tile_size, sigma_scale=sigma_scale)
    w = w.reshape(grid, region_size, grid, region_size).mean(axis=(1, 3))
    return w.astype(np.float32)


def _scatter_prob(
    acc: np.ndarray,
    prob_np: np.ndarray,
    meta_np: np.ndarray,
    weight: np.ndarray,
    region_size: int,
    H: int,
    W: int,
    level_downsample: float = 1.0,
) -> None:
    """Accumulate one batch of region probabilities into a full-slide map."""
    if level_downsample != 1.0:
        d = float(level_downsample)
        if not np.isfinite(d) or d < 1:
            raise ValueError("level_downsample must be finite and >= 1")
        for p, m in zip(prob_np, meta_np):
            (x, y, h, w) = map(int, m)
            gx0 = max(0, int(np.ceil(x * d / region_size - 0.5)))
            gy0 = max(0, int(np.ceil(y * d / region_size - 0.5)))
            gx1 = min(W, int(np.ceil((x + w) * d / region_size - 0.5)))
            gy1 = min(H, int(np.ceil((y + h) * d / region_size - 0.5)))
            if gx1 <= gx0 or gy1 <= gy0:
                continue
            ix = np.floor((np.arange(gx0, gx1) + 0.5) / d - x / region_size).astype(int)
            iy = np.floor((np.arange(gy0, gy1) + 0.5) / d - y / region_size).astype(int)
            ix = np.clip(ix, 0, p.shape[2] - 1)
            iy = np.clip(iy, 0, p.shape[1] - 1)
            weighted = p * weight[None, :, :]
            for row in range(0, len(iy), 64):
                rows = iy[row : row + 64]
                acc[:, gy0 + row : gy0 + row + len(rows), gx0:gx1] += weighted[
                    :, rows[:, None], ix[None, :]
                ]
        return
    for p, m in zip(prob_np, meta_np):
        (x, y, h, w) = map(int, m)
        if x % region_size != 0 or y % region_size != 0:
            raise ValueError(f"Invalid meta coordinates: {m}")
        gx = x // region_size
        gy = y // region_size
        region_h = (h + region_size - 1) // region_size
        region_w = (w + region_size - 1) // region_size
        gx1 = min(gx + region_w, W)
        gy1 = min(gy + region_h, H)
        valid_h = gy1 - gy
        valid_w = gx1 - gx
        if valid_h <= 0 or valid_w <= 0:
            continue
        acc[:, gy:gy1, gx:gx1] += (
            p[:, :valid_h, :valid_w] * weight[:valid_h, :valid_w][None, :, :]
        )


def _write_argmax_dat(
    acc: np.ndarray,
    out_path: str,
    H: int,
    W: int,
    tissue_mask: np.ndarray | None = None,
    chunk_rows: int = 1024,
) -> str:
    """
    Write uint16 argmax in row chunks and enforce the exact output contract.

    This preserves the original result semantics while avoiding a full-slide int64
    argmax temporary.  The final file must be exactly H * W * 2 bytes.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if acc.shape[1:] != (H, W):
        raise RuntimeError(
            f"Accumulator shape mismatch for {out_path}: {acc.shape[1:]} != {(H, W)}"
        )
    if tissue_mask is not None and tissue_mask.shape != (H, W):
        raise RuntimeError(
            f"Tissue-mask shape mismatch for {out_path}: {tissue_mask.shape} != {(H, W)}"
        )
    tmp_path = out_path + ".tmp"
    expected_size = H * W * np.dtype(np.uint16).itemsize
    try:
        pred = np.memmap(tmp_path, dtype=np.uint16, mode="w+", shape=(H, W))
        for y0 in range(0, H, max(1, int(chunk_rows))):
            y1 = min(y0 + max(1, int(chunk_rows)), H)
            block = acc[:, y0:y1, :].argmax(axis=0).astype(np.uint16, copy=False)
            if tissue_mask is not None:
                block[~tissue_mask[y0:y1, :]] = 0
            pred[y0:y1, :] = block
            del block
        pred.flush()
        del pred
        actual_tmp_size = os.path.getsize(tmp_path)
        if actual_tmp_size != expected_size:
            raise RuntimeError(
                f"Prediction tmp size mismatch for {tmp_path}: got {actual_tmp_size}, expected {expected_size} bytes"
            )
        os.replace(tmp_path, out_path)
        actual_size = os.path.getsize(out_path)
        if actual_size != expected_size:
            try:
                os.remove(out_path)
            except OSError:
                pass
            raise RuntimeError(
                f"Prediction size mismatch for {out_path}: got {actual_size}, expected {expected_size} bytes"
            )
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return out_path


def _make_accumulator(
    shape: tuple[int, int, int], *, use_memmap: bool, work_dir: str, prefix: str
) -> tuple[np.ndarray, str | None]:
    """Allocate a zeroed float32 accumulator in RAM or as a disk-backed memmap."""
    if not use_memmap:
        return (np.zeros(shape, dtype=np.float32), None)
    os.makedirs(work_dir or ".", exist_ok=True)
    (fd, path) = tempfile.mkstemp(prefix=prefix, suffix=".acc.tmp", dir=work_dir or ".")
    os.close(fd)
    acc = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
    return (acc, path)


def _close_accumulator(acc: np.ndarray | None, path: str | None) -> None:
    if acc is not None and isinstance(acc, np.memmap):
        acc.flush()
        mmap_obj = getattr(acc, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _prediction_file_is_complete(path: str, label_shape: tuple[int, int]) -> bool:
    """Check whether a uint16 prediction file has exactly the expected byte size."""
    (H, W) = label_shape
    expected_size = H * W * np.dtype(np.uint16).itemsize
    try:
        return os.path.isfile(path) and os.path.getsize(path) == expected_size
    except OSError:
        return False


def _release_host_memory() -> None:
    """Release Python objects and ask glibc to return free heap pages to Linux."""
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except (OSError, AttributeError):
            pass


def _loader_process_settings(
    dataset: CTransPathPatchDataset, num_workers: int, prefetch_factor: int
) -> tuple[int, int]:
    """Choose a bounded-memory DataLoader configuration."""
    workers = max(0, int(num_workers))
    prefetch = max(1, int(prefetch_factor))
    if dataset.cache_mode == "full" and workers > 0:
        try:
            start_method = mp.get_start_method(allow_none=True)
        except RuntimeError:
            start_method = None
        if start_method == "spawn" or sys.platform in ("win32", "darwin"):
            print(
                "[DataLoader] full RAM cache + spawn detected; forcing num_workers=0 to avoid duplicating the cache."
            )
            workers = 0
    if dataset.cache_mode != "full":
        workers = min(workers, 4)
        prefetch = min(prefetch, 2)
    return (workers, prefetch)


@torch.inference_mode()
def infer_from_h5_overlap(
    h5_path: str | None,
    he_tif_path: str,
    seg_model: torch.nn.Module,
    domain_out_path: str,
    niche_out_path: str,
    label_shape: tuple[int, int],
    num_domain_classes: int,
    num_niche_classes: int,
    batch_size: int = 64,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    device: str | torch.device = "cuda",
    use_amp: bool = True,
    load_h5_RAM: str | bool = "Auto",
    use_rgb_region: bool = True,
    region_size: int = 28,
    pin_memory: bool = True,
    standardize_features: bool = True,
    mask_background: bool = True,
    smooth_sigma: float = 0.0,
    memory_headroom_gb: float = 16.0,
    feature_stats_chunk_tiles: int = 256,
    accumulator_memmap: bool | None = None,
    background_gray_thresh: float = 200.0,
    background_sat_thresh: float = 15.0,
    feature_arrays: dict[str, np.ndarray] | None = None,
    level_downsample: float = 1.0,
) -> tuple[str, str]:
    """
    Dual-head overlap inference with the original output geometry and bounded memory.

    Correctness invariants:
        - output geometry matches label generation: OpenSlide level-0 H/W are
          rounded down by 224 and mapped to 28-pixel regions;
        - H5 coverage never determines output shape (include_edge=False may omit edge);
        - overlap accumulation stays float32;
        - feature standardization uses all tiles, matching the old load_h5_RAM=Yes path;
        - output stays uint16 with exactly H * W * 2 bytes.

    Memory controls:
        - H5 features are normally read in contiguous DataLoader batches, while
          extract-infer mode passes identically shaped arrays directly from RAM;
        - argmax is written row-by-row instead of materializing a full int64 map;
        - very large full-slide accumulators automatically become disk-backed memmaps.
    """
    if level_downsample != 1.0 and feature_arrays is None:
        raise ValueError(
            "Multi-level inference requires in-memory features with RGB regions"
        )
    os.makedirs(os.path.dirname(domain_out_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(niche_out_path) or ".", exist_ok=True)
    device = torch.device(device)
    seg_model = seg_model.to(device)
    seg_model.eval()
    if device.type == "cuda":
        seg_model = seg_model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
    (resolved_shape, _level0_he_shape) = _resolve_output_geometry(
        he_tif_path=he_tif_path,
        configured_label_shape=label_shape,
        region_size=region_size,
        verbose=False,
    )
    (H, W) = resolved_shape
    label_shape = resolved_shape
    if feature_arrays is None:
        if h5_path is None:
            raise ValueError("h5_path is required when feature_arrays is not provided")
        _validate_h5_metadata(h5_path=h5_path, region_size=region_size)
    acc_bytes = (
        (num_domain_classes + num_niche_classes)
        * int(H)
        * int(W)
        * np.dtype(np.float32).itemsize
    )
    available = get_effective_available_memory_bytes()
    requested_headroom = int(float(memory_headroom_gb) * _GIB)
    if available is None:
        runtime_headroom = requested_headroom
    else:
        runtime_headroom = max(
            int(available * 0.08), min(requested_headroom, int(available * 0.25))
        )
    cache_memory_reserve = acc_bytes + runtime_headroom
    print(
        f"[memory plan] accumulators={format_bytes(acc_bytes)}, headroom={format_bytes(runtime_headroom)}, available={format_bytes(available)}"
    )
    dataset = CTransPathPatchDataset(
        h5_path=h5_path,
        label_path=None,
        label_shape=label_shape,
        num_clusters=None,
        he_tif_path=he_tif_path,
        ignore_idx=0,
        normalize_rgb=True,
        load_h5_RAM=load_h5_RAM,
        region_size=region_size,
        use_rgb_region=use_rgb_region,
        return_meta=True,
        standardize_features=standardize_features,
        return_labels=False,
        defer_feature_standardization=False,
        cache_memory_reserve_bytes=cache_memory_reserve,
        feature_stats_chunk_tiles=feature_stats_chunk_tiles,
        feature_stats_mode="exact_chunked",
        feature_stats_cache=False,
        feature_stats_cache_path=None,
        feature_arrays=feature_arrays,
    )
    (effective_workers, effective_prefetch) = _loader_process_settings(
        dataset, num_workers=num_workers, prefetch_factor=prefetch_factor
    )
    print(
        f"[DataLoader] cache_mode={dataset.cache_mode}, batch_size={batch_size}, workers={effective_workers}, prefetch={effective_prefetch}, pin_memory={pin_memory}"
    )
    load_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if effective_workers > 0:
        load_kwargs.update(persistent_workers=False, prefetch_factor=effective_prefetch)
    loader = DataLoader(**load_kwargs)
    if accumulator_memmap is None:
        if available is None:
            use_acc_memmap = acc_bytes >= 8 * _GIB
        else:
            use_acc_memmap = acc_bytes > int(
                available * 0.45
            ) or acc_bytes + runtime_headroom > int(available * 0.8)
    else:
        use_acc_memmap = bool(accumulator_memmap)
    acc_dir = os.path.dirname(domain_out_path) or "."
    if use_acc_memmap:
        free_disk = shutil.disk_usage(acc_dir).free
        needed_disk = int(acc_bytes * 1.05)
        if free_disk < needed_disk:
            raise RuntimeError(
                f"Not enough free disk space for memory-safe accumulators: need about {format_bytes(needed_disk)}, free={format_bytes(free_disk)}, dir={acc_dir}"
            )
        print(
            f"[memory plan] using disk-backed float32 accumulators; temporary_disk={format_bytes(acc_bytes)}"
        )
    domain_acc = niche_acc = None
    domain_acc_path = niche_acc_path = None
    weight = tissue_mask = None
    pbar = None
    try:
        (domain_acc, domain_acc_path) = _make_accumulator(
            (num_domain_classes, H, W),
            use_memmap=use_acc_memmap,
            work_dir=acc_dir,
            prefix="domain_",
        )
        (niche_acc, niche_acc_path) = _make_accumulator(
            (num_niche_classes, H, W),
            use_memmap=use_acc_memmap,
            work_dir=acc_dir,
            prefix="niche_",
        )
        weight = gaussian_weight_region_np(tile_size=TILE_SIZE, region_size=region_size)
        amp_enabled = bool(use_amp and device.type == "cuda")
        pbar = tqdm(loader, desc="infering...")
        saw_batch = False
        for batch in pbar:
            saw_batch = True
            (feat, rgb, meta) = batch
            feat = feat.to(device, non_blocking=True)
            if not amp_enabled:
                feat = feat.float()
            rgb = rgb.to(device, non_blocking=True, memory_format=torch.channels_last)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                (domain_logits, niche_logits) = seg_model(feat, rgb)
                domain_prob = domain_logits.softmax(dim=1)
                niche_prob = niche_logits.softmax(dim=1)
            meta_np = meta.numpy()
            domain_np = (
                domain_prob.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            niche_np = niche_prob.detach().cpu().numpy().astype(np.float32, copy=False)
            _scatter_prob(
                domain_acc,
                domain_np,
                meta_np,
                weight,
                region_size,
                H,
                W,
                level_downsample=level_downsample,
            )
            _scatter_prob(
                niche_acc,
                niche_np,
                meta_np,
                weight,
                region_size,
                H,
                W,
                level_downsample=level_downsample,
            )
        pbar.close()
        pbar = None
        if saw_batch:
            del batch, feat, rgb, meta
            del domain_logits, niche_logits, domain_prob, niche_prob
            del meta_np, domain_np, niche_np
        loader = None
        if smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter

            for acc in (domain_acc, niche_acc):
                for ci in range(acc.shape[0]):
                    gaussian_filter(acc[ci], sigma=smooth_sigma, output=acc[ci])
        if mask_background:
            tissue_mask = tissue_mask_from_he(
                he_tif_path,
                (H, W),
                gray_thresh=background_gray_thresh,
                sat_thresh=background_sat_thresh,
            )
            print(
                f"[background] gray={background_gray_thresh}, RGB-range={background_sat_thresh}, tissue_fraction={tissue_mask.mean():.4f}"
            )
        expected_bytes = H * W * np.dtype(np.uint16).itemsize
        print(
            f"[output contract] shape={(H, W)}, dtype=uint16, expected_each={format_bytes(expected_bytes)}"
        )
        _write_argmax_dat(domain_acc, domain_out_path, H, W, tissue_mask)
        _write_argmax_dat(niche_acc, niche_out_path, H, W, tissue_mask)
    finally:
        if pbar is not None:
            pbar.close()
        dataset.close()
        _close_accumulator(domain_acc, domain_acc_path)
        _close_accumulator(niche_acc, niche_acc_path)
        domain_acc = niche_acc = None
        weight = tissue_mask = None
        loader = None
        load_kwargs.clear()
        feature_arrays = None
        _release_host_memory()
    return (domain_out_path, niche_out_path)


def infer_from_memory_overlap(
    feature_arrays: dict[str, np.ndarray],
    he_tif_path: str,
    seg_model: torch.nn.Module,
    domain_out_path: str,
    niche_out_path: str,
    label_shape: tuple[int, int],
    num_domain_classes: int,
    num_niche_classes: int,
    **kwargs,
) -> tuple[str, str]:
    """Run the existing overlap-fusion path without an intermediate H5 file."""
    kwargs.pop("load_h5_RAM", None)
    return infer_from_h5_overlap(
        h5_path=None,
        he_tif_path=he_tif_path,
        seg_model=seg_model,
        domain_out_path=domain_out_path,
        niche_out_path=niche_out_path,
        label_shape=label_shape,
        num_domain_classes=num_domain_classes,
        num_niche_classes=num_niche_classes,
        feature_arrays=feature_arrays,
        load_h5_RAM=True,
        **kwargs,
    )


def infer_multi_slices(
    model_path: str,
    pred_dat_dir: str,
    slides: list[dict],
    num_domain_classes: int = 5,
    num_niche_classes: int = 21,
    device: str | torch.device = "cuda",
    batch_size: int = 64,
    mask_background: bool = True,
    smooth_sigma: float = 0.0,
    num_workers: int = 0,
    load_h5_RAM: str | bool = "Auto",
    standardize_features: bool | None = None,
    model_config: dict | None = None,
    skip_existing: bool = False,
    background_gray_thresh: float = 200.0,
    background_sat_thresh: float = 15.0,
    on_slide_complete=None,
):
    """Run dual-head inference; optional explicit reuse of existing outputs."""
    os.makedirs(pred_dat_dir, exist_ok=True)
    pending_slides = []
    for slide in slides:
        name = slide["name"]
        (resolved_shape, level0_he_shape) = _resolve_output_geometry(
            he_tif_path=slide["he_tif_path"],
            configured_label_shape=slide.get("label_shape"),
            region_size=28,
            slide_name=name,
            verbose=True,
        )
        slide["label_shape"] = resolved_shape
        slide["he_shape"] = tuple(n * 28 for n in resolved_shape)
        domain_path = os.path.join(pred_dat_dir, name + "_domain.dat")
        niche_path = os.path.join(pred_dat_dir, name + "_niche.dat")
        slide["domain_pred_dat_path"] = domain_path
        slide["niche_pred_dat_path"] = niche_path
        domain_ok = _prediction_file_is_complete(domain_path, slide["label_shape"])
        niche_ok = _prediction_file_is_complete(niche_path, slide["label_shape"])
        if skip_existing and domain_ok and niche_ok:
            print(f"[skip] {name}: prediction files already complete")
            if on_slide_complete is not None:
                on_slide_complete(slide)
            continue
        pending_slides.append(slide)
    if not pending_slides:
        print("All slides already have complete predictions; nothing to infer.")
        return
    print(
        f"Need inference: {len(pending_slides)}/{len(slides)} slides; skip: {len(slides) - len(pending_slides)}"
    )
    (num_domain_output_classes, num_niche_output_classes) = counts_with_background(
        num_domain_classes, num_niche_classes
    )
    fallback = dict(model_config or {})
    fallback.update(
        num_domain_classes=num_domain_output_classes,
        num_niche_classes=num_niche_output_classes,
    )
    (seg, checkpoint) = load_segmentation_model(model_path, fallback)
    if (
        seg.domain_head.out_channels != num_domain_output_classes
        or seg.niche_head.out_channels != num_niche_output_classes
    ):
        print(
            f"[checkpoint] configured class counts disagree with checkpoint; using checkpoint heads domain={seg.domain_head.out_channels}, niche={seg.niche_head.out_channels}"
        )
        num_domain_output_classes = int(seg.domain_head.out_channels)
        num_niche_output_classes = int(seg.niche_head.out_channels)
    if standardize_features is None:
        standardize_features = checkpoint.get("standardize_features", True)
    del checkpoint
    device_obj = torch.device(device)
    seg = seg.to(device_obj)
    seg.eval()
    for index, slide in enumerate(pending_slides, start=1):
        print(f"[infer {index}/{len(pending_slides)}] {slide['name']}")
        infer_from_h5_overlap(
            h5_path=slide["h5_path"],
            he_tif_path=slide["he_tif_path"],
            seg_model=seg,
            domain_out_path=slide["domain_pred_dat_path"],
            niche_out_path=slide["niche_pred_dat_path"],
            label_shape=slide["label_shape"],
            num_domain_classes=num_domain_output_classes,
            num_niche_classes=num_niche_output_classes,
            batch_size=batch_size,
            load_h5_RAM=load_h5_RAM,
            num_workers=num_workers,
            standardize_features=standardize_features,
            device=device_obj,
            pin_memory=device_obj.type == "cuda",
            mask_background=mask_background,
            background_gray_thresh=background_gray_thresh,
            background_sat_thresh=background_sat_thresh,
            smooth_sigma=smooth_sigma,
        )
        if on_slide_complete is not None:
            on_slide_complete(slide)


def extract_and_infer_multi_slices(
    model_path: str,
    pred_dat_dir: str,
    slides: list[dict],
    ctranspath_checkpoint: str | None = None,
    num_domain_classes: int = 5,
    num_niche_classes: int = 21,
    device: str | torch.device = "cuda",
    feature_batch_size: int = 256,
    inference_batch_size: int = 128,
    stride: int = 112,
    include_edge: bool = False,
    mask_background: bool = True,
    smooth_sigma: float = 0.0,
    num_workers: int = 0,
    standardize_features: bool | None = None,
    model_config: dict | None = None,
    skip_existing: bool = False,
    background_gray_thresh: float = 200.0,
    background_sat_thresh: float = 15.0,
    on_slide_complete=None,
    slide_level: int = 0,
) -> None:
    """Extract each slide to RAM and immediately run the normal inference path."""
    os.makedirs(pred_dat_dir, exist_ok=True)
    pending_slides = []
    for slide in slides:
        name = slide["name"]
        with open_slide(slide["he_tif_path"]) as image:
            (dimensions, downsample) = validate_slide_level(image, slide_level)
        slide["slide_level"] = slide_level
        slide["level_downsample"] = downsample
        output_name = name if slide_level == 0 else f"{name}_level{slide_level}"
        print(
            f"[level] {name}: level={slide_level}, dimensions={dimensions}, downsample={downsample}"
        )
        (resolved_shape, _) = _resolve_output_geometry(
            he_tif_path=slide["he_tif_path"],
            configured_label_shape=slide.get("label_shape"),
            region_size=28,
            slide_name=name,
            verbose=True,
        )
        slide["label_shape"] = resolved_shape
        slide["he_shape"] = tuple(n * 28 for n in resolved_shape)
        slide["domain_pred_dat_path"] = os.path.join(
            pred_dat_dir, output_name + "_domain.dat"
        )
        slide["niche_pred_dat_path"] = os.path.join(
            pred_dat_dir, output_name + "_niche.dat"
        )
        outputs_complete = all(
            _prediction_file_is_complete(slide[f"{head}_pred_dat_path"], resolved_shape)
            for head in ("domain", "niche")
        )
        if skip_existing and outputs_complete:
            print(f"[skip] {name}: prediction files already complete")
            if on_slide_complete is not None:
                on_slide_complete(slide)
            continue
        pending_slides.append(slide)
    if not pending_slides:
        print("All slides already have complete predictions; nothing to infer.")
        return
    (domain_out, niche_out) = counts_with_background(
        num_domain_classes, num_niche_classes
    )
    fallback = dict(model_config or {})
    fallback.update(num_domain_classes=domain_out, num_niche_classes=niche_out)
    (seg_model, checkpoint) = load_segmentation_model(model_path, fallback)
    if (
        seg_model.domain_head.out_channels != domain_out
        or seg_model.niche_head.out_channels != niche_out
    ):
        print(
            f"[checkpoint] configured class counts disagree with checkpoint; using checkpoint heads domain={seg_model.domain_head.out_channels}, niche={seg_model.niche_head.out_channels}"
        )
        domain_out = int(seg_model.domain_head.out_channels)
        niche_out = int(seg_model.niche_head.out_channels)
    if standardize_features is None:
        standardize_features = checkpoint.get("standardize_features", True)
    del checkpoint
    device_obj = torch.device(device)
    seg_model = seg_model.to(device_obj).eval()
    feature_model = load_ctranspath(
        checkpoint_path=ctranspath_checkpoint, device=device_obj
    ).eval()
    try:
        for index, slide in enumerate(pending_slides, start=1):
            print(f"[extract-infer {index}/{len(pending_slides)}] {slide['name']}")
            (H, W) = slide["label_shape"]
            acc_bytes = (domain_out + niche_out) * H * W * np.dtype(np.float32).itemsize
            available = get_effective_available_memory_bytes()
            if available is None:
                runtime_headroom = 2 * _GIB
            else:
                runtime_headroom = max(
                    int(available * 0.08), min(16 * _GIB, int(available * 0.25))
                )
            features = extract_ctranspath_to_memory(
                image_path=slide["he_tif_path"],
                checkpoint_path=ctranspath_checkpoint,
                batch_size=feature_batch_size,
                device=device_obj,
                stride=stride,
                include_edge=include_edge,
                model=feature_model,
                memory_reserve_bytes=acc_bytes + runtime_headroom,
                slide_level=slide_level,
            )
            try:
                infer_from_memory_overlap(
                    feature_arrays=features.as_dataset_arrays(),
                    level_downsample=features.level_downsample,
                    he_tif_path=slide["he_tif_path"],
                    seg_model=seg_model,
                    domain_out_path=slide["domain_pred_dat_path"],
                    niche_out_path=slide["niche_pred_dat_path"],
                    label_shape=slide["label_shape"],
                    num_domain_classes=domain_out,
                    num_niche_classes=niche_out,
                    batch_size=inference_batch_size,
                    num_workers=num_workers,
                    standardize_features=bool(standardize_features),
                    device=device_obj,
                    pin_memory=device_obj.type == "cuda",
                    mask_background=mask_background,
                    background_gray_thresh=background_gray_thresh,
                    background_sat_thresh=background_sat_thresh,
                    smooth_sigma=smooth_sigma,
                )
            finally:
                released_bytes = features.release()
                del features
                _release_host_memory()
                print(
                    f"[RAM release] {slide['name']}: released feature buffers={format_bytes(released_bytes)}, available={format_bytes(get_effective_available_memory_bytes())}"
                )
            if on_slide_complete is not None:
                on_slide_complete(slide)
    finally:
        del feature_model, seg_model
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()
        _release_host_memory()

"""Frozen CTransPath encoding and streamed feature extraction."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import timm
import torch
import torch.nn as nn
from timm.layers import to_2tuple
from timm.models.swin_transformer import checkpoint_filter_fn
from tqdm import tqdm

from spagc.histology.memory import format_bytes, get_effective_available_memory_bytes
from spagc.histology.slides import (
    SlideReader,
    open_slide,
    read_region_resized_rgb,
    validate_slide_level,
)

FEAT_DIM = 768
REGION_SIZE = 28
TILE_SIZE = 224
REGION_GRID = 8
TOKEN_NUM = 49
REQUIRED_H5_DATASETS = {
    "global_features": ((FEAT_DIM,), np.dtype("float16")),
    "patch_features": ((TOKEN_NUM, FEAT_DIM), np.dtype("float16")),
    "coords": ((2,), np.dtype("int32")),
    "valid_size": ((2,), np.dtype("int32")),
}


class ConvStem(nn.Module):
    """CTransPath convolutional stem that replaces the original Swin PatchEmbed layer."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer=None,
        flatten: bool = True,
        output_fmt=None,
        strict_img_size: bool = True,
    ):
        super().__init__()
        assert patch_size == 4
        assert embed_dim % 8 == 0
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = to_2tuple(img_size[0] // patch_size[0])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten and output_fmt is None
        self.output_fmt = output_fmt
        self.strict_img_size = strict_img_size
        stem = []
        (in_dim, out_dim) = (in_chans, embed_dim // 8)
        for _ in range(2):
            stem.append(
                nn.Conv2d(
                    in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False
                )
            )
            stem.append(nn.BatchNorm2d(out_dim))
            stem.append(nn.ReLU(inplace=True))
            in_dim = out_dim
            out_dim *= 2
        stem.append(nn.Conv2d(in_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        (_, _, h, w) = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()
        if self.output_fmt == "NHWC":
            x = x.permute(0, 2, 3, 1)
        return self.norm(x)


def ctranspath():
    """Build the CTransPath Swin-Tiny backbone with convolutional patch embedding."""
    return timm.create_model(
        "swin_tiny_patch4_window7_224", embed_layer=ConvStem, pretrained=False
    )


def load_ctranspath(
    checkpoint_path: str | Path, device: torch.device | str = "cuda"
) -> nn.Module:
    """Restore and freeze the explicitly supplied pretrained CTransPath weights."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CTransPath checkpoint not found: {checkpoint_path}")
    model = ctranspath()
    model.head = nn.Identity()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = {k.removeprefix("module."): v for (k, v) in state_dict.items()}
    state_dict = checkpoint_filter_fn(state_dict, model)
    model.load_state_dict(state_dict, strict=True)
    for module in model.modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = False
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def make_norm_tensors(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move normalization constants to the device once and return mean and standard-deviation tensors."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(
        1, 3, 1, 1
    )
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(
        1, 3, 1, 1
    )
    return (mean, std)


def preprocess_np(
    batch_np: np.ndarray,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    use_amp: bool = True,
) -> torch.Tensor:
    x = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous()
    if device.type == "cuda":
        x = x.pin_memory()
        target_dtype = torch.float16 if use_amp else torch.float32
    else:
        target_dtype = torch.float32
    x = x.to(device=device, dtype=target_dtype, non_blocking=True).div_(255.0)
    return (x - mean) / std


def compute_rgb_regions_on_gpu(image: torch.Tensor, device: torch.device) -> np.ndarray:
    """Calculate regional RGB summaries on the GPU from normalized image tensors."""
    x = image.unfold(2, REGION_SIZE, REGION_SIZE).unfold(3, REGION_SIZE, REGION_SIZE)
    x = x.mean(dim=(4, 5))
    return x.detach().cpu().numpy()


def _to_spatial_tokens(features: torch.Tensor) -> torch.Tensor:
    """Normalize outputs from different timm versions to `[batch, 49, 768]` spatial tokens."""
    if features.ndim == 3:
        tokens = features
    elif features.ndim == 4 and features.shape[-1] == 768:
        tokens = features.reshape(features.shape[0], -1, features.shape[-1])
    elif features.ndim == 4 and features.shape[1] == 768:
        tokens = features.flatten(2).transpose(1, 2).contiguous()
    else:
        raise ValueError(
            f"Unsupported Swin feature shape: {tuple(features.shape)}. Expected [B,49,768], [B,7,7,768], or [B,768,7,7]."
        )
    if tokens.shape[-1] != 768:
        raise ValueError(f"Expected feature dimension 768, got {tokens.shape[-1]}")
    if tokens.shape[1] != 49:
        raise ValueError(
            f"Expected 49 spatial tokens, got {tokens.shape[1]}; full shape={tuple(tokens.shape)}"
        )
    return tokens


def _forward_spatial_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run the backbone without discarding the final 7-by-7 spatial grid."""
    backbone = getattr(model, "_orig_mod", model)
    x = backbone.patch_embed(x)
    absolute_pos_embed = getattr(backbone, "absolute_pos_embed", None)
    if absolute_pos_embed is not None:
        x = x + absolute_pos_embed
    pos_drop = getattr(backbone, "pos_drop", None)
    if pos_drop is not None:
        x = pos_drop(x)
    patch_drop = getattr(backbone, "patch_drop", None)
    if patch_drop is not None:
        x = patch_drop(x)
    norm_pre = getattr(backbone, "norm_pre", None)
    if norm_pre is not None:
        x = norm_pre(x)
    x = backbone.layers(x)
    x = backbone.norm(x)
    return x


@torch.no_grad()
def extract_ctranspath_features(
    model: nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global and spatial CTransPath features for normalized RGB images."""
    features = _forward_spatial_features(model=model, x=x)
    spatial_tokens = _to_spatial_tokens(features)
    global_features = spatial_tokens.mean(dim=1)
    return (global_features, spatial_tokens)


def window_starts(
    total: int, win: int, stride: int, include_edge: bool = False
) -> list[int]:
    """Generate overlapping window starts while ensuring coverage of the final image boundary."""
    if stride <= 0 or stride % REGION_SIZE or stride > win:
        raise ValueError("stride must be a positive multiple of 28, at most 224")
    if include_edge:
        total = total // TILE_SIZE * TILE_SIZE
    if total <= win:
        return [0]
    starts = list(range(0, total - win + 1, stride))
    if include_edge and starts[-1] != total - win:
        starts.append(total - win)
    return starts


def _expected_h5_layout(
    image_path: str | os.PathLike[str],
    stride: int,
    include_edge: bool,
    save_rgb_regions: bool,
) -> tuple[int, tuple[int, int], dict[str, tuple[tuple[int, ...], np.dtype]]]:
    with open_slide(image_path) as image:
        dimensions = image.dimensions
        xs = window_starts(dimensions[0], TILE_SIZE, stride, include_edge)
        ys = window_starts(dimensions[1], TILE_SIZE, stride, include_edge)
    n_tiles = len(xs) * len(ys)
    layout = {
        name: ((n_tiles, *tail_shape), dtype)
        for (name, (tail_shape, dtype)) in REQUIRED_H5_DATASETS.items()
    }
    if save_rgb_regions:
        layout["rgb_regions"] = (
            (n_tiles, 3, REGION_GRID, REGION_GRID),
            np.dtype("float32"),
        )
    return (n_tiles, dimensions, layout)


def h5_feature_file_is_complete(
    h5_path: str | os.PathLike[str],
    image_path: str | os.PathLike[str],
    stride: int = 112,
    include_edge: bool = False,
    save_rgb_regions: bool = True,
) -> bool:
    """Check that an existing feature H5 satisfies the current output contract."""
    path = Path(h5_path)
    if not path.is_file():
        return False
    try:
        (n_tiles, dimensions, layout) = _expected_h5_layout(
            image_path,
            stride=stride,
            include_edge=include_edge,
            save_rgb_regions=save_rgb_regions,
        )
        expected_payload_bytes = sum(
            int(np.prod(shape)) * dtype.itemsize for (shape, dtype) in layout.values()
        )
        if path.stat().st_size < expected_payload_bytes:
            return False
        with h5py.File(path, "r") as h5_file:
            if "extraction_complete" in h5_file.attrs and (
                not bool(h5_file.attrs["extraction_complete"])
            ):
                return False
            expected_attrs = {
                "feature_extractor": "CTransPath",
                "tile_size": TILE_SIZE,
                "stride": stride,
                "feature_dim": FEAT_DIM,
                "slide_level": 0,
            }
            for name, expected in expected_attrs.items():
                if name not in h5_file.attrs or h5_file.attrs[name] != expected:
                    return False
            if (
                "tile_count" in h5_file.attrs
                and int(h5_file.attrs["tile_count"]) != n_tiles
            ):
                return False
            if (
                "source_dimensions" in h5_file.attrs
                and tuple(map(int, h5_file.attrs["source_dimensions"])) != dimensions
            ):
                return False
            for name, (shape, dtype) in layout.items():
                if name not in h5_file:
                    return False
                dataset = h5_file[name]
                if dataset.shape != shape or dataset.dtype != dtype:
                    return False
                if dataset.id.get_storage_size() < dataset.nbytes:
                    return False
    except (OSError, RuntimeError, ValueError, KeyError):
        return False
    return True


def read_batch_tiles_into(
    img: SlideReader,
    coords: list[tuple[int, int]],
    img_w: int,
    img_h: int,
    batch_arr: np.ndarray,
    meta_arr: np.ndarray,
    slide_level: int = 0,
    level_downsample: float = 1.0,
) -> None:
    """Read grouped image regions and split them into tiles in memory to reduce I/O calls."""
    rows = {}
    for i, (x, y) in enumerate(coords):
        if y not in rows:
            rows[y] = []
        rows[y].append((x, i))
    for y, x_list in rows.items():
        xs = [item[0] for item in x_list]
        indices = [item[1] for item in x_list]
        min_x = min(xs)
        max_x = max(xs)
        region_w = min(max_x + TILE_SIZE, img_w) - min_x
        region_h = min(TILE_SIZE, img_h - y)
        if hasattr(img, "read_region_rgb"):
            region_arr = img.read_region_rgb(
                (round(min_x * level_downsample), round(y * level_downsample)),
                slide_level,
                (region_w, region_h),
            )
        else:
            region = img.extract_area(min_x, y, region_w, region_h)
            region_arr = np.ndarray(
                buffer=region.write_to_memory(),
                dtype=np.uint8,
                shape=(region_h, region_w, region.bands),
            )[..., :3]
            if region_arr.shape[-1] == 1:
                region_arr = np.repeat(region_arr, 3, axis=-1)
        for x, idx in zip(xs, indices):
            tile = batch_arr[idx]
            tile.fill(255)
            w = min(TILE_SIZE, img_w - x)
            h = min(TILE_SIZE, img_h - y)
            offset = x - min_x
            tile[:h, :w, :] = region_arr[:h, offset : offset + w, :]
            meta_arr[idx] = (x, y, h, w)


@torch.inference_mode()
def run_batch(
    model: nn.Module,
    batch_np: np.ndarray,
    meta_np: np.ndarray,
    feature_store: MutableMapping[str, object],
    idx: int,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    use_amp: bool = True,
    save_rgb_regions: bool = True,
) -> int:
    b = batch_np.shape[0]
    image = preprocess_np(
        batch_np=batch_np, device=device, mean=mean, std=std, use_amp=use_amp
    )
    amp_enabled = bool(use_amp and device.type == "cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
        (global_features, spatial_tokens) = extract_ctranspath_features(
            model=model, x=image
        )
    end = idx + b
    if "global_features" in feature_store:
        feature_store["global_features"][idx:end] = (
            global_features.detach().to(torch.float16).cpu().numpy()
        )
    feature_store["patch_features"][idx:end] = (
        spatial_tokens.detach().to(torch.float16).cpu().numpy()
    )
    feature_store["coords"][idx:end] = meta_np[:, 0:2].astype(np.int32, copy=False)
    feature_store["valid_size"][idx:end] = meta_np[:, 2:4].astype(np.int32, copy=False)
    if save_rgb_regions:
        rgb = compute_rgb_regions_on_gpu(image=image, device=device)
        feature_store["rgb_regions"][idx:end] = rgb
    return end


def _extract_batches(
    img: SlideReader,
    xs: list[int],
    ys: list[int],
    batch_size: int,
    model: nn.Module,
    feature_store: MutableMapping[str, object],
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    use_amp: bool,
    save_rgb_regions: bool,
    slide_level: int = 0,
    level_downsample: float = 1.0,
) -> None:
    """Shared tile loop for persistent H5 and ephemeral RAM outputs."""
    (img_w, img_h) = (
        (img.width, img.height)
        if slide_level == 0
        else img.level_dimensions[slide_level]
    )
    n_tiles = len(xs) * len(ys)
    batch_arr = np.empty((batch_size, TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    meta_arr = np.empty((batch_size, 4), dtype=np.int32)
    idx = 0
    with tqdm(total=n_tiles, desc="Extracting features...") as pbar:
        for start in range(0, n_tiles, batch_size):
            end = min(start + batch_size, n_tiles)
            b = end - start
            batch_coords = [
                (xs[i % len(xs)], ys[i // len(xs)]) for i in range(start, end)
            ]
            read_batch_tiles_into(
                img=img,
                coords=batch_coords,
                img_w=img_w,
                img_h=img_h,
                batch_arr=batch_arr,
                meta_arr=meta_arr,
                slide_level=slide_level,
                level_downsample=level_downsample,
            )
            idx = run_batch(
                model=model,
                batch_np=batch_arr[:b],
                meta_np=meta_arr[:b],
                feature_store=feature_store,
                idx=idx,
                device=device,
                mean=mean,
                std=std,
                use_amp=use_amp,
                save_rgb_regions=save_rgb_regions,
            )
            pbar.update(b)
    if idx != n_tiles:
        raise RuntimeError(
            f"Incomplete feature extraction: expected {n_tiles} tiles, wrote {idx}"
        )


def extract_ctranspath_from_svs(
    svs_path: str,
    out_h5: str,
    checkpoint_path: str | None = None,
    batch_size: int = 256,
    device: torch.device | str = "cuda",
    stride: int = 112,
    include_edge: bool = False,
    save_rgb_regions: bool = True,
    model: nn.Module | None = None,
    use_amp: bool = True,
    h5_chunk_tiles: int = 128,
) -> str:
    """Extract CTransPath and RGB-region features from a whole-slide image with batched I/O."""
    svs_path = os.fspath(svs_path)
    out_h5 = os.fspath(out_h5)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    out_dir = os.path.dirname(out_h5)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    temporary_h5 = out_h5 + ".tmp"
    if os.path.exists(temporary_h5):
        os.unlink(temporary_h5)
    device = torch.device(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except RuntimeError:
            pass
    ext = svs_path.rsplit(".", 1)[-1].lower()
    img = open_slide(svs_path)
    xs = window_starts(img.width, TILE_SIZE, stride, include_edge=include_edge)
    ys = window_starts(img.height, TILE_SIZE, stride, include_edge=include_edge)
    n_tiles = len(xs) * len(ys)
    own_model = model is None
    if model is None:
        model = load_ctranspath(checkpoint_path=checkpoint_path, device=device)
    model.eval()
    norm_dtype = torch.float16 if use_amp and device.type == "cuda" else torch.float32
    (mean, std) = make_norm_tensors(device=device, dtype=norm_dtype)
    chunk_tiles = max(1, min(int(h5_chunk_tiles), int(batch_size), 128, n_tiles))
    with h5py.File(temporary_h5, "w") as h5_file:
        h5_file.attrs["feature_extractor"] = "CTransPath"
        h5_file.attrs["tile_size"] = TILE_SIZE
        h5_file.attrs["stride"] = stride
        h5_file.attrs["token_grid"] = "7x7"
        h5_file.attrs["feature_dim"] = FEAT_DIM
        h5_file.attrs["source_format"] = ext.upper()
        h5_file.attrs["slide_level"] = 0
        h5_file.attrs["source_dimensions"] = (img.width, img.height)
        h5_file.attrs["tile_count"] = n_tiles
        h5_file.attrs["extraction_complete"] = False
        h5_file.create_dataset(
            "global_features",
            shape=(n_tiles, FEAT_DIM),
            dtype="float16",
            chunks=(chunk_tiles, FEAT_DIM),
        )
        h5_file.create_dataset(
            "patch_features",
            shape=(n_tiles, TOKEN_NUM, FEAT_DIM),
            dtype="float16",
            chunks=(chunk_tiles, TOKEN_NUM, FEAT_DIM),
        )
        h5_file.create_dataset(
            "coords", shape=(n_tiles, 2), dtype="int32", chunks=(chunk_tiles, 2)
        )
        h5_file.create_dataset(
            "valid_size", shape=(n_tiles, 2), dtype="int32", chunks=(chunk_tiles, 2)
        )
        if save_rgb_regions:
            h5_file.create_dataset(
                "rgb_regions",
                shape=(n_tiles, 3, REGION_GRID, REGION_GRID),
                dtype="float32",
                chunks=(min(4096, n_tiles), 3, REGION_GRID, REGION_GRID),
            )
            h5_file.attrs["rgb_region"] = (
                "precomputed normalized RGB pooled to [3, 8, 8]"
            )
            h5_file.attrs["rgb_region_size"] = REGION_SIZE
            h5_file.attrs["rgb_region_grid"] = REGION_GRID
            h5_file.attrs["rgb_region_source_format"] = ext.upper()
            h5_file.attrs["rgb_region_slide_level"] = 0
        _extract_batches(
            img=img,
            xs=xs,
            ys=ys,
            batch_size=batch_size,
            model=model,
            feature_store=h5_file,
            device=device,
            mean=mean,
            std=std,
            use_amp=use_amp,
            save_rgb_regions=save_rgb_regions,
        )
        h5_file.attrs["extraction_complete"] = True
        h5_file.flush()
    os.replace(temporary_h5, out_h5)
    img.close()
    if own_model and device.type == "cuda":
        torch.cuda.empty_cache()
    return out_h5


@dataclass
class InMemoryFeatures:
    """Feature-extraction result kept in host RAM for immediate inference."""

    patch_features: np.ndarray | None
    coords: np.ndarray | None
    valid_size: np.ndarray | None
    rgb_regions: np.ndarray | None
    source_dimensions: tuple[int, int]
    stride: int
    slide_level: int = 0
    level_downsample: float = 1.0

    @property
    def nbytes(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self.patch_features,
                self.coords,
                self.valid_size,
                self.rgb_regions,
            )
            if array is not None
        )

    def as_dataset_arrays(self) -> dict[str, np.ndarray]:
        if any(
            array is None
            for array in (
                self.patch_features,
                self.coords,
                self.valid_size,
                self.rgb_regions,
            )
        ):
            raise RuntimeError("In-memory features have already been released")
        return {
            "patch_features": self.patch_features,
            "coords": self.coords,
            "valid_size": self.valid_size,
            "rgb_regions": self.rgb_regions,
        }

    def release(self) -> int:
        """Drop all references to the large arrays and return released bytes."""
        released = self.nbytes
        self.patch_features = None
        self.coords = None
        self.valid_size = None
        self.rgb_regions = None
        return released


def extract_ctranspath_to_memory(
    image_path: str,
    checkpoint_path: str | None = None,
    batch_size: int = 256,
    device: torch.device | str = "cuda",
    stride: int = 112,
    include_edge: bool = False,
    model: nn.Module | None = None,
    use_amp: bool = True,
    memory_reserve_bytes: int = 0,
    slide_level: int = 0,
) -> InMemoryFeatures:
    """Extract the segmentation inputs into RAM without creating an H5 file."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except RuntimeError:
            pass
    img = open_slide(image_path)
    try:
        (dimensions, downsample) = validate_slide_level(img, slide_level)
        xs = window_starts(dimensions[0], TILE_SIZE, stride, include_edge=include_edge)
        ys = window_starts(dimensions[1], TILE_SIZE, stride, include_edge=include_edge)
        n_tiles = len(xs) * len(ys)
        feature_bytes = (
            n_tiles * TOKEN_NUM * FEAT_DIM * np.dtype(np.float16).itemsize
            + n_tiles * 2 * np.dtype(np.int32).itemsize * 2
            + n_tiles * 3 * REGION_GRID * REGION_GRID * np.dtype(np.float32).itemsize
        )
        available = get_effective_available_memory_bytes()
        reserve = max(0, int(memory_reserve_bytes))
        print(
            f"[RAM extraction] tiles={n_tiles}, features={format_bytes(feature_bytes)}, reserve={format_bytes(reserve)}, available={format_bytes(available)}"
        )
        if available is not None and feature_bytes + reserve > int(available * 0.9):
            raise MemoryError(
                f"Insufficient RAM for extract-infer mode: features + reserve need {format_bytes(feature_bytes + reserve)}, available={format_bytes(available)}. Use the separate extract/infer commands or reduce concurrent memory use."
            )
        arrays = InMemoryFeatures(
            patch_features=np.empty((n_tiles, TOKEN_NUM, FEAT_DIM), dtype=np.float16),
            coords=np.empty((n_tiles, 2), dtype=np.int32),
            valid_size=np.empty((n_tiles, 2), dtype=np.int32),
            rgb_regions=np.empty(
                (n_tiles, 3, REGION_GRID, REGION_GRID), dtype=np.float32
            ),
            source_dimensions=dimensions,
            stride=stride,
            slide_level=slide_level,
            level_downsample=downsample,
        )
        own_model = model is None
        if model is None:
            model = load_ctranspath(checkpoint_path=checkpoint_path, device=device)
        model.eval()
        norm_dtype = (
            torch.float16 if use_amp and device.type == "cuda" else torch.float32
        )
        (mean, std) = make_norm_tensors(device=device, dtype=norm_dtype)
        _extract_batches(
            img=img,
            xs=xs,
            ys=ys,
            batch_size=batch_size,
            model=model,
            feature_store=arrays.as_dataset_arrays(),
            device=device,
            mean=mean,
            std=std,
            use_amp=use_amp,
            save_rgb_regions=True,
            slide_level=slide_level,
            level_downsample=downsample,
        )
    finally:
        img.close()
    if own_model:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"[RAM extraction] complete: {format_bytes(arrays.nbytes)}")
    return arrays


def tissue_mask_from_he(
    he_tif_path: str,
    label_shape: tuple[int, int],
    gray_thresh: float = 200.0,
    sat_thresh: float = 15.0,
) -> np.ndarray:
    """Build a region-resolution tissue mask from H&E intensity and saturation."""
    (H, W) = label_shape
    with open_slide(he_tif_path) as slide:
        arr = read_region_resized_rgb(slide, (0, 0), (W * 28, H * 28), (W, H)).astype(
            np.float32
        )
    if arr.shape[:2] != (H, W):
        padded = np.full((H, W, 3), 255.0, dtype=np.float32)
        h = min(H, arr.shape[0])
        w = min(W, arr.shape[1])
        padded[:h, :w] = arr[:h, :w]
        arr = padded
    gray = arr.mean(axis=2)
    sat = arr.max(axis=2) - arr.min(axis=2)
    return (gray < gray_thresh) | (sat > sat_thresh)

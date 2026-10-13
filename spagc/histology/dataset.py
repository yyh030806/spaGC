"""Patch features, label rasters and training-set feature statistics."""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from spagc.histology.memory import (
    _dataset_nbytes,
    format_bytes,
    get_effective_available_memory_bytes,
)
from spagc.histology.slides import open_slide

TILE_SIZE = 224
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def _validate_h5_metadata(
    h5_path: str, region_size: int = 28, scan_chunk: int = 262144
) -> None:
    """Validate metadata invariants without using H5 coverage to choose shape.

    H5 coordinates are level-0 pixel coordinates.  Because extraction may omit the
    final right/bottom edge, max(coords + valid_size) is deliberately *not* compared
    with, nor used to reconstruct, the output .dat shape.
    """
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f or "valid_size" not in f:
            raise KeyError(f"{h5_path} must contain 'coords' and 'valid_size'")
        coords = f["coords"]
        valid = f["valid_size"]
        n = len(coords)
        if coords.shape != (n, 2) or valid.shape != (n, 2):
            raise ValueError("coords and valid_size must have shape [N,2]")
        if "patch_features" not in f or f["patch_features"].shape != (n, 49, 768):
            raise ValueError("patch_features must have shape [N,49,768]")
        if "rgb_regions" in f and f["rgb_regions"].shape != (n, 3, 8, 8):
            raise ValueError("rgb_regions must have shape [N,3,8,8]")
        if n == 0:
            raise ValueError("Empty H5 feature dataset")
        if len(coords) != len(valid):
            raise ValueError(
                f"coords/valid_size length mismatch in {h5_path}: {len(coords)} vs {len(valid)}"
            )
        for start in range(0, len(coords), scan_chunk):
            end = min(start + scan_chunk, len(coords))
            c = np.asarray(coords[start:end], dtype=np.int64)
            v = np.asarray(valid[start:end], dtype=np.int64)
            if c.size == 0:
                continue
            if np.any(c < 0) or np.any(v <= 0) or np.any(v > TILE_SIZE):
                raise ValueError(
                    f"Negative coordinate or non-positive valid_size in {h5_path}"
                )
            if np.any(c[:, 0] % region_size) or np.any(c[:, 1] % region_size):
                raise ValueError(
                    f"H5 coordinates must be divisible by region_size={region_size}: {h5_path}"
                )


class CTransPathPatchDataset(Dataset):
    """
    Dataset with a speed-first, memory-safe H5 cache strategy.

    load_h5_RAM="Auto":
        Keep H5 datasets on disk and read on demand.

    load_h5_RAM="Yes":
        Prefer a full RAM cache, preserving H5 dtypes (notably float16 features).
        If the full cache would violate the configured memory reserve, fall back to
        a hybrid mode: cache small metadata / RGB regions when safe, while reading
        patch_features in contiguous batches through __getitems__.

    defer_feature_standardization=True:
        Intended for inference. It returns raw H5 features (typically float16) and
        lets infer.py standardize in float32 on the GPU. This halves feature host
        memory / PCIe traffic compared with converting every feature tile to float32
        in DataLoader workers.
    """

    def __getitems__(self, indices: Sequence[int]):
        """
        PyTorch DataLoader batch fast-path.

        With shuffle=False (inference), indices are contiguous, so hybrid mode reads
        one H5 slice per batch instead of 128 individual H5 reads. Full-cache mode
        also vectorizes feature dtype conversion / standardization during training.
        """
        idx = np.asarray(indices, dtype=np.int64)
        if idx.ndim != 1:
            idx = idx.reshape(-1)
        if idx.size == 0:
            return []
        feat_batch, coords_batch, valid_batch, rgb_batch = self._read_batch_arrays(idx)
        feat_batch = self._prepare_feature_batch(feat_batch, idx)
        samples = []
        for j, raw_idx in enumerate(idx.tolist()):
            x, y = coords_batch[j]
            h, w = valid_batch[j]
            rgb = rgb_batch[j] if self.use_rgb_region else None
            samples.append(
                self._assemble_sample(
                    feat=feat_batch[j],
                    rgb=rgb,
                    x=int(x),
                    y=int(y),
                    h=int(h),
                    w=int(w),
                    raw_idx=int(raw_idx),
                    feature_already_prepared=True,
                )
            )
        return samples

    def __getitem__(self, idx):
        """Scalar fallback; DataLoader normally uses __getitems__ for batches."""
        raw_idx = int(idx)
        if self.cache_mode == "full":
            feat = self.features[raw_idx]
            x, y = self.coords[raw_idx]
            h, w = self.valid_size[raw_idx]
            rgb = self.rgb_regions[raw_idx] if self.use_rgb_region else None
        else:
            f = self._open_h5()
            feat = f["patch_features"][raw_idx]
            if self.coords is not None:
                x, y = self.coords[raw_idx]
            else:
                x, y = f["coords"][raw_idx]
            if self.valid_size is not None:
                h, w = self.valid_size[raw_idx]
            else:
                h, w = f["valid_size"][raw_idx]
            if self.use_rgb_region:
                if self.rgb_regions is not None:
                    rgb = self.rgb_regions[raw_idx]
                else:
                    rgb = f["rgb_regions"][raw_idx]
            else:
                rgb = None
        return self._assemble_sample(
            feat=feat, rgb=rgb, x=int(x), y=int(y), h=int(h), w=int(w), raw_idx=raw_idx
        )

    def __init__(
        self,
        h5_path: str | None,
        label_path: str | None,
        label_shape: tuple,
        num_clusters: int | None,
        he_tif_path: str,
        ignore_idx: int = 0,
        normalize_rgb: bool = True,
        load_h5_RAM: str | bool = "Auto",
        cache_h5: bool | None = None,
        region_size: int = 28,
        use_rgb_region: bool = True,
        return_meta: bool = False,
        domain_label_path: str | None = None,
        standardize_features: bool = False,
        augment: bool = False,
        feature_noise_std: float = 0.05,
        rgb_jitter: float = 0.1,
        return_labels: bool = True,
        defer_feature_standardization: bool = False,
        cache_memory_reserve_bytes: int = 0,
        feature_stats_chunk_tiles: int = 2048,
        feature_stats_mode: str = "exact_chunked",
        feature_stats_cache: bool = False,
        feature_stats_cache_path: str | None = None,
        feature_arrays: dict[str, np.ndarray] | None = None,
    ):
        self.h5_path = h5_path
        self.he_tif_path = he_tif_path
        self.standardize_features = standardize_features
        self.defer_feature_standardization = defer_feature_standardization
        self.augment = augment
        self.feature_noise_std = feature_noise_std
        self.rgb_jitter = rgb_jitter
        self.return_labels = return_labels
        self.feat_mean: np.ndarray | None = None
        self.feat_std: np.ndarray | None = None
        self.feature_stats_chunk_tiles = max(1, int(feature_stats_chunk_tiles))
        self.feature_stats_mode = str(feature_stats_mode).lower()
        if self.feature_stats_mode not in {"legacy", "exact_chunked"}:
            raise ValueError(
                f"feature_stats_mode must be 'legacy' or 'exact_chunked', got {feature_stats_mode!r}"
            )
        self.feature_stats_cache = bool(feature_stats_cache)
        self.feature_stats_cache_path = (
            feature_stats_cache_path
            if feature_stats_cache_path is not None
            else self.h5_path + ".feature_stats_v2.npz"
            if self.h5_path is not None
            else None
        )
        self.augment_index_set = None
        self.label = (
            np.memmap(label_path, dtype=np.uint16, mode="r", shape=label_shape)
            if label_path is not None
            else None
        )
        self.domain_label = (
            np.memmap(domain_label_path, dtype=np.uint16, mode="r", shape=label_shape)
            if domain_label_path is not None
            else None
        )
        self.num_clusters = num_clusters
        self.ignore_idx = ignore_idx
        self.normalize_rgb = normalize_rgb
        if cache_h5 is not None:
            load_h5_RAM = "Yes" if bool(cache_h5) else "No"
        mode = str(load_h5_RAM).strip().lower()
        if mode in {"yes", "y", "true", "1"}:
            mode = "yes"
        elif mode in {"no", "n", "false", "0"}:
            mode = "no"
        elif mode == "auto":
            mode = "auto"
        else:
            raise ValueError("load_h5_RAM must be 'Yes', 'No' or 'Auto'")
        self.load_h5_RAM = mode
        self.cache_h5 = False
        self.region_size = region_size
        self.use_rgb_region = use_rgb_region
        self.return_meta = return_meta
        self.region_grid = TILE_SIZE // region_size
        if TILE_SIZE % region_size != 0:
            raise ValueError("tile size must be divisible by region size")
        self.features: np.ndarray | None = None
        self.coords: np.ndarray | None = None
        self.valid_size: np.ndarray | None = None
        self.rgb_regions: np.ndarray | None = None
        self.h5: h5py.File | None = None
        self.he_img = None
        self.cache_mode = "none"
        self.full_cache_bytes = 0
        self.actual_cached_bytes = 0
        self.feature_dtype = np.dtype(np.float32)
        if feature_arrays is not None:
            required = {"patch_features", "coords", "valid_size"}
            missing = required - set(feature_arrays)
            if missing:
                raise KeyError(f"In-memory features missing arrays: {sorted(missing)}")
            if use_rgb_region and "rgb_regions" not in feature_arrays:
                raise KeyError("In-memory features missing rgb_regions")
            self.features = np.asarray(feature_arrays["patch_features"])
            self.coords = np.asarray(feature_arrays["coords"])
            self.valid_size = np.asarray(feature_arrays["valid_size"])
            self.rgb_regions = (
                np.asarray(feature_arrays["rgb_regions"]) if use_rgb_region else None
            )
            self.length = len(self.features)
            expected_shapes = {
                "patch_features": (self.length, 49, 768),
                "coords": (self.length, 2),
                "valid_size": (self.length, 2),
            }
            if use_rgb_region:
                expected_shapes["rgb_regions"] = (self.length, 3, 8, 8)
            actual_arrays = {
                "patch_features": self.features,
                "coords": self.coords,
                "valid_size": self.valid_size,
                "rgb_regions": self.rgb_regions,
            }
            for name, expected in expected_shapes.items():
                if actual_arrays[name].shape != expected:
                    raise ValueError(
                        f"In-memory {name} must have shape {expected}, got {actual_arrays[name].shape}"
                    )
            if self.length == 0:
                raise ValueError("Empty in-memory feature dataset")
            if np.any(self.coords < 0) or np.any(self.valid_size <= 0):
                raise ValueError("Invalid in-memory coords or valid_size")
            if np.any(self.valid_size > TILE_SIZE):
                raise ValueError("In-memory valid_size exceeds tile size")
            if np.any(self.coords % region_size):
                raise ValueError(
                    f"In-memory coordinates must be divisible by {region_size}"
                )
            self.feature_dtype = self.features.dtype
            self.full_cache_bytes = sum(
                array.nbytes
                for array in (
                    self.features,
                    self.coords,
                    self.valid_size,
                    self.rgb_regions,
                )
                if array is not None
            )
            self.actual_cached_bytes = self.full_cache_bytes
            self.cache_mode = "full"
            self.cache_h5 = True
            self.feature_stats_cache = False
            print(
                f"[feature source] RAM, tiles={self.length}, size={self.full_cache_bytes / 1024**3:.2f} GiB"
            )
            if self.standardize_features:
                self._compute_feature_stats_from_array(self.features)
            return
        if self.h5_path is None:
            raise ValueError("h5_path is required when feature_arrays is not provided")
        _validate_h5_metadata(self.h5_path, region_size=region_size)
        with h5py.File(self.h5_path, "r") as f:
            if use_rgb_region and "rgb_regions" not in f:
                raise ValueError(
                    "rgb_regions not found in h5, but use_rgb_region is True"
                )
            feature_ds = f["patch_features"]
            self.length = feature_ds.shape[0]
            self.feature_dtype = np.dtype(feature_ds.dtype)
            feature_shape = tuple(feature_ds.shape[1:])
            if feature_shape != (49, 768):
                raise ValueError(
                    f"CTransPath H5 must contain patch_features shaped [N, 49, 768], but received {feature_ds.shape}."
                )
            self.full_cache_bytes = (
                _dataset_nbytes(feature_ds)
                + _dataset_nbytes(f["coords"])
                + _dataset_nbytes(f["valid_size"])
                + (_dataset_nbytes(f["rgb_regions"]) if self.use_rgb_region else 0)
            )
            reserve = max(0, int(cache_memory_reserve_bytes))
            available = get_effective_available_memory_bytes()
            if self.load_h5_RAM == "yes":
                use_ram = True
            elif self.load_h5_RAM == "no":
                use_ram = False
            else:
                use_ram = available is None or self.full_cache_bytes + reserve <= int(
                    available * 0.6
                )
            available_text = (
                f"{available / 1024**3:.2f} GiB" if available is not None else "unknown"
            )
            print(
                f"[H5 RAM] requested={load_h5_RAM}, use={('Yes' if use_ram else 'No')}, file={os.path.basename(self.h5_path)}, size={self.full_cache_bytes / 1024**3:.2f} GiB, available={available_text}"
            )
            self.cache_h5 = use_ram
            if use_ram:
                self._initialize_smart_cache(f=f, memory_reserve_bytes=reserve)
            if self.standardize_features:
                if not self._load_feature_stats_cache():
                    self._compute_feature_stats(f)
                    self._save_feature_stats_cache()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        if self.he_img is not None:
            self.he_img.close()
            self.he_img = None
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None
        self.features = None
        self.coords = None
        self.valid_size = None
        self.rgb_regions = None
        self.label = None
        self.domain_label = None

    def __len__(self):
        return self.length

    def _open_h5(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")
        return self.h5

    def _open_he(self):
        if self.he_img is None:
            self.he_img = open_slide(self.he_tif_path)
        return self.he_img

    def _read_rgb_tile(self, x: int, y: int, h: int, w: int):
        """Read an HE tile when precomputed rgb_regions are not used."""
        img = self._open_he()
        region = img.read_region((x, y), 0, (w, h))
        arr = np.array(region.convert("RGB"))
        tile = np.full((TILE_SIZE, TILE_SIZE, 3), 255, dtype=np.uint8)
        tile[: int(h), : int(w)] = arr
        tile = tile.astype(np.float32) / 255.0
        if self.normalize_rgb:
            tile = (tile - RGB_MEAN) / RGB_STD
        tile = np.transpose(tile, (2, 0, 1)).copy()
        return torch.from_numpy(tile)

    @staticmethod
    def _contiguous_slice(indices: np.ndarray) -> slice | None:
        if indices.size == 0:
            return slice(0, 0)
        if indices.size == 1:
            i = int(indices[0])
            return slice(i, i + 1)
        if np.all(indices[1:] == indices[:-1] + 1):
            return slice(int(indices[0]), int(indices[-1]) + 1)
        return None

    @staticmethod
    def _take_h5(ds: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
        """Fast contiguous read; robust sorted fancy read for shuffled batches."""
        sl = CTransPathPatchDataset._contiguous_slice(indices)
        if sl is not None:
            return ds[sl]
        order = np.argsort(indices)
        sorted_idx = indices[order]
        if np.any(sorted_idx[1:] == sorted_idx[:-1]):
            return np.stack([ds[int(i)] for i in indices], axis=0)
        sorted_values = ds[sorted_idx]
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        return sorted_values[inverse]

    def _read_batch_arrays(self, indices: np.ndarray):
        sl = self._contiguous_slice(indices)
        if self.cache_mode == "full":
            feat = self.features[sl] if sl is not None else self.features[indices]
            coords = self.coords[sl] if sl is not None else self.coords[indices]
            valid = self.valid_size[sl] if sl is not None else self.valid_size[indices]
            if self.use_rgb_region:
                rgb = (
                    self.rgb_regions[sl]
                    if sl is not None
                    else self.rgb_regions[indices]
                )
            else:
                rgb = None
            return (feat, coords, valid, rgb)
        f = self._open_h5()
        feat = self._take_h5(f["patch_features"], indices)
        if self.coords is not None:
            coords = self.coords[sl] if sl is not None else self.coords[indices]
        else:
            coords = self._take_h5(f["coords"], indices)
        if self.valid_size is not None:
            valid = self.valid_size[sl] if sl is not None else self.valid_size[indices]
        else:
            valid = self._take_h5(f["valid_size"], indices)
        if self.use_rgb_region:
            if self.rgb_regions is not None:
                rgb = (
                    self.rgb_regions[sl]
                    if sl is not None
                    else self.rgb_regions[indices]
                )
            else:
                rgb = self._take_h5(f["rgb_regions"], indices)
        else:
            rgb = None
        return (feat, coords, valid, rgb)

    def _prepare_feature_batch(
        self, feat_batch: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        if self.defer_feature_standardization:
            feat_batch = np.ascontiguousarray(feat_batch)
        else:
            feat_batch = np.array(feat_batch, dtype=np.float32, order="C", copy=True)
            if self.standardize_features and self.feat_mean is not None:
                feat_batch -= self.feat_mean.reshape(1, 1, -1)
                feat_batch /= self.feat_std.reshape(1, 1, -1)
        if self.augment and self.feature_noise_std > 0:
            if feat_batch.dtype != np.float32:
                feat_batch = feat_batch.astype(np.float32)
            for j, raw_idx in enumerate(indices.tolist()):
                do_aug = (
                    self.augment_index_set is None or raw_idx in self.augment_index_set
                )
                if do_aug:
                    feat_batch[j] += (
                        np.random.randn(*feat_batch[j].shape).astype(np.float32)
                        * self.feature_noise_std
                    )
        return feat_batch

    def _make_labels_and_meta(self, x: int, y: int, h: int, w: int):
        if x % self.region_size != 0 or y % self.region_size != 0:
            raise ValueError("Tile coordinate must be divisible by region size")
        meta = torch.tensor([x, y, h, w], dtype=torch.long)
        if not self.return_labels:
            return (None, None, meta)
        gx = x // self.region_size
        gy = y // self.region_size
        region_h = (h + self.region_size - 1) // self.region_size
        region_w = (w + self.region_size - 1) // self.region_size
        niche_label = np.full(
            (self.region_grid, self.region_grid), self.ignore_idx, dtype=np.int64
        )
        if self.label is not None:
            patch_label = self.label[gy : gy + region_h, gx : gx + region_w]
            ph, pw = patch_label.shape[:2]
            niche_label[:ph, :pw] = patch_label
            if self.num_clusters is not None:
                invalid = (niche_label < 0) | (niche_label >= self.num_clusters)
                if invalid.any():
                    raise ValueError(
                        "Niche label ID exceeds model class count (including background)"
                    )
        domain_label = np.full(
            (self.region_grid, self.region_grid), self.ignore_idx, dtype=np.int64
        )
        if self.domain_label is not None:
            patch_domain = self.domain_label[gy : gy + region_h, gx : gx + region_w]
            ph, pw = patch_domain.shape[:2]
            domain_label[:ph, :pw] = patch_domain
        return (torch.from_numpy(niche_label), torch.from_numpy(domain_label), meta)

    def _assemble_sample(
        self,
        feat: np.ndarray,
        rgb: np.ndarray | None,
        x: int,
        y: int,
        h: int,
        w: int,
        raw_idx: int,
        feature_already_prepared: bool = False,
    ):
        if not feature_already_prepared:
            feat_batch = self._prepare_feature_batch(
                np.expand_dims(feat, axis=0), np.asarray([raw_idx], dtype=np.int64)
            )
            feat = feat_batch[0]
        if self.use_rgb_region:
            rgb_t = torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32))
        else:
            rgb_t = self._read_rgb_tile(x=x, y=y, h=h, w=w)
        do_aug = self.augment and (
            self.augment_index_set is None or raw_idx in self.augment_index_set
        )
        if do_aug and self.rgb_jitter > 0:
            scale = 1.0 + (torch.rand(3, 1, 1) * 2 - 1) * self.rgb_jitter
            bias = (torch.rand(3, 1, 1) * 2 - 1) * self.rgb_jitter
            rgb_t = rgb_t * scale + bias
        niche_label, domain_label, meta = self._make_labels_and_meta(x, y, h, w)
        feat_t = torch.from_numpy(feat)
        if self.return_labels:
            if self.return_meta:
                return (feat_t, rgb_t, niche_label, domain_label, meta)
            return (feat_t, rgb_t, niche_label, domain_label)
        if self.return_meta:
            return (feat_t, rgb_t, meta)
        return (feat_t, rgb_t)

    def _initialize_smart_cache(self, f: h5py.File, memory_reserve_bytes: int) -> None:
        available = get_effective_available_memory_bytes()
        full_cache_safe = (
            available is None
            or self.full_cache_bytes + memory_reserve_bytes <= available
        )
        print(
            f"[H5 cache plan] file={os.path.basename(self.h5_path)}, full_cache={format_bytes(self.full_cache_bytes)}, available={format_bytes(available)}, reserved={format_bytes(memory_reserve_bytes)}"
        )
        if full_cache_safe:
            self.features = f["patch_features"][:]
            self.coords = f["coords"][:]
            self.valid_size = f["valid_size"][:]
            if self.use_rgb_region:
                self.rgb_regions = f["rgb_regions"][:]
            self.cache_mode = "full"
            self.actual_cached_bytes = sum(
                arr.nbytes
                for arr in (
                    self.features,
                    self.coords,
                    self.valid_size,
                    self.rgb_regions,
                )
                if arr is not None
            )
            print(
                f"[H5 cache] mode=full, cached={format_bytes(self.actual_cached_bytes)}, feature_dtype={self.features.dtype}"
            )
            return
        self.cache_mode = "hybrid"
        self.features = None
        self.coords = f["coords"][:]
        self.valid_size = f["valid_size"][:]
        cached = self.coords.nbytes + self.valid_size.nbytes
        if self.use_rgb_region:
            rgb_bytes = _dataset_nbytes(f["rgb_regions"])
            current_available = get_effective_available_memory_bytes()
            rgb_safe = (
                current_available is None
                or rgb_bytes + memory_reserve_bytes <= current_available
            )
            if rgb_safe:
                self.rgb_regions = f["rgb_regions"][:]
                cached += self.rgb_regions.nbytes
        self.actual_cached_bytes = cached
        print(
            f"[H5 cache] mode=hybrid, cached={format_bytes(self.actual_cached_bytes)}; patch_features will be read by contiguous DataLoader batches."
        )
        warnings.warn(
            "Full H5 RAM cache was skipped to avoid an OOM kill. Hybrid batched H5 reading is enabled instead.",
            RuntimeWarning,
        )

    def _compute_feature_stats_from_array(self, features: np.ndarray) -> None:
        """All-tile population mean/std for an in-memory float16 feature array."""
        if len(features) <= 0:
            raise ValueError("Empty in-memory patch_features dataset")
        print(
            f"[feature stats] computing all-tile mean/std from RAM (chunk_tiles={self.feature_stats_chunk_tiles})..."
        )
        sum_x = np.zeros(768, dtype=np.float64)
        sum_x2 = np.zeros(768, dtype=np.float64)
        count = 0
        for start in range(0, len(features), self.feature_stats_chunk_tiles):
            end = min(start + self.feature_stats_chunk_tiles, len(features))
            block = np.asarray(features[start:end], dtype=np.float32)
            sum_x += block.sum(axis=(0, 1), dtype=np.float64)
            sum_x2 += np.einsum(
                "ntd,ntd->d", block, block, dtype=np.float64, optimize=True
            )
            count += block.shape[0] * block.shape[1]
        mean64 = sum_x / count
        var64 = sum_x2 / count - mean64 * mean64
        np.maximum(var64, 0.0, out=var64)
        self.feat_mean = mean64.astype(np.float32)
        self.feat_std = np.sqrt(var64).astype(np.float32)
        self.feat_std += np.float32(1e-06)

    def _stats_source_signature(self) -> tuple[int, int, tuple[int, ...], str]:
        st = os.stat(self.h5_path)
        return (
            int(st.st_size),
            int(st.st_mtime_ns),
            (self.length, 49, 768),
            self.feature_dtype.str,
        )

    def _load_feature_stats_cache(self) -> bool:
        if not self.feature_stats_cache:
            return False
        path = self.feature_stats_cache_path
        if not path or not os.path.isfile(path):
            return False
        try:
            file_size, mtime_ns, shape, dtype_str = self._stats_source_signature()
            with np.load(path, allow_pickle=False) as z:
                if str(z["stats_mode"].item()) != self.feature_stats_mode:
                    return False
                if int(z["file_size"]) != file_size:
                    return False
                if int(z["mtime_ns"]) != mtime_ns:
                    return False
                if tuple(z["feature_shape"].tolist()) != shape:
                    return False
                if str(z["feature_dtype"].item()) != dtype_str:
                    return False
                mean = z["mean"].astype(np.float32, copy=False)
                std = z["std"].astype(np.float32, copy=False)
                if mean.shape != (768,) or std.shape != (768,):
                    return False
                self.feat_mean = mean
                self.feat_std = std
            print(f"[feature stats] loaded: {path}")
            return True
        except (OSError, ValueError, KeyError, EOFError):
            return False

    def _save_feature_stats_cache(self) -> None:
        if (
            not self.feature_stats_cache
            or self.feat_mean is None
            or self.feat_std is None
        ):
            return
        path = self.feature_stats_cache_path
        if not path:
            return
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            file_size, mtime_ns, shape, dtype_str = self._stats_source_signature()
            tmp = f"{path}.tmp.{os.getpid()}.npz"
            np.savez(
                tmp,
                stats_mode=np.asarray(self.feature_stats_mode),
                file_size=np.int64(file_size),
                mtime_ns=np.int64(mtime_ns),
                feature_shape=np.asarray(shape, dtype=np.int64),
                feature_dtype=np.asarray(dtype_str),
                mean=self.feat_mean,
                std=self.feat_std,
            )
            os.replace(tmp, path)
            print(f"[feature stats] saved: {path}")
        except OSError as e:
            warnings.warn(
                f"Could not save feature-stat cache {path}: {e}", RuntimeWarning
            )

    def _compute_feature_stats(self, f: h5py.File) -> None:
        """
        Compute per-channel feature statistics without materializing the whole H5.

        Legacy statistics use the historical channel-wise normalization:
          * load_h5_RAM=No -> deterministic sample of at most 2000 tiles;
          * load_h5_RAM=Yes  -> statistics over all tiles.

        exact_chunked is used by inference: infer_prev.py requested load_h5_RAM=Yes, so
        the correct semantic baseline is statistics over every tile, but we compute
        them in bounded chunks instead of holding the entire float32 tensor in RAM.
        """
        if self.feature_stats_mode == "legacy" and (not self.cache_h5):
            n = self.length
            if n <= 0:
                raise ValueError(f"Empty patch_features dataset: {self.h5_path}")
            sample_idx = np.sort(
                np.random.default_rng(0).choice(n, size=min(n, 2000), replace=False)
            )
            feats = f["patch_features"][sample_idx].astype(np.float32)
            flat = feats.reshape(-1, feats.shape[-1])
            self.feat_mean = flat.mean(axis=0).astype(np.float32)
            self.feat_std = flat.std(axis=0).astype(np.float32) + np.float32(1e-06)
            del flat, feats
            return
        self._compute_feature_stats_exact_chunked(f)

    def _compute_feature_stats_exact_chunked(self, f: h5py.File) -> None:
        """All-tile population mean/std with bounded temporary memory."""
        if self.length <= 0:
            raise ValueError(f"Empty patch_features dataset: {self.h5_path}")
        print(
            f"[feature stats] computing all-tile mean/std in chunks (chunk_tiles={self.feature_stats_chunk_tiles})..."
        )
        sum_x = np.zeros(768, dtype=np.float64)
        sum_x2 = np.zeros(768, dtype=np.float64)
        count = 0
        source = self.features if self.cache_mode == "full" else f["patch_features"]
        n = self.length
        for start in range(0, n, self.feature_stats_chunk_tiles):
            end = min(start + self.feature_stats_chunk_tiles, n)
            block = np.asarray(source[start:end], dtype=np.float32)
            sum_x += block.sum(axis=(0, 1), dtype=np.float64)
            sum_x2 += np.einsum(
                "ntd,ntd->d", block, block, dtype=np.float64, optimize=True
            )
            count += block.shape[0] * block.shape[1]
            del block
        mean64 = sum_x / count
        var64 = sum_x2 / count - mean64 * mean64
        np.maximum(var64, 0.0, out=var64)
        self.feat_mean = mean64.astype(np.float32)
        self.feat_std = np.sqrt(var64).astype(np.float32)
        self.feat_std += np.float32(1e-06)


def build_slide_dataset(
    slides,
    num_niche_classes: int = 23,
    ignore_idx: int = 0,
    region_size: int = 28,
    use_rgb_region: bool = True,
    load_h5_RAM: str | bool = "Auto",
    standardize_features: bool = False,
    augment: bool = False,
    feature_noise_std: float = 0.05,
    rgb_jitter: float = 0.1,
):
    datasets = []
    for slide in slides:
        dataset = CTransPathPatchDataset(
            h5_path=slide["h5_path"],
            label_path=slide.get("niche_label_path", slide.get("label_path")),
            label_shape=slide["label_shape"],
            num_clusters=num_niche_classes,
            he_tif_path=slide["he_tif_path"],
            ignore_idx=ignore_idx,
            load_h5_RAM=load_h5_RAM,
            region_size=region_size,
            use_rgb_region=use_rgb_region,
            return_meta=False,
            domain_label_path=slide.get("domain_label_path"),
            standardize_features=standardize_features,
            augment=augment,
            feature_noise_std=feature_noise_std,
            rgb_jitter=rgb_jitter,
        )
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)

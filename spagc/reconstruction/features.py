"""Frozen UNI2-h features on registered physical grids."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
from torchvision import transforms

from spagc._io import require_new
from spagc.reconstruction.coordinates import (
    NativeCropReader,
    source_to_registered_points,
)


class FeatureStore:
    """Store and query masked morphology grids in physical coordinates."""

    def __init__(self, path, mode="r", cache_mb=512):
        if mode not in ("r", "a", "x"):
            raise ValueError("Use r/a/x; destructive overwrite is not supported")
        self.path = str(path)
        self.cache_bytes = int(cache_mb * 1024**2)
        self._cache = OrderedDict()
        if mode != "r":
            with h5py.File(self.path, mode) as f:
                f.require_group("sections")
                if "revision" not in f.attrs:
                    f.attrs["revision"] = str(uuid.uuid4())
                f.attrs["format"] = "deepspatial-uni2-grid-v1"
        self.refresh()

    def refresh(self):
        self._cache.clear()
        self._spatial_indices = {}
        with h5py.File(self.path, "r") as f:
            self.revision = str(f.attrs["revision"])
            self.metadata = {
                str(g.attrs["section_id"]): json.loads(g.attrs["metadata"])
                for key, g in f["sections"].items()
                if not key.startswith("_pending_")
            }
        self.sections = sorted(self.metadata, key=lambda s: self.metadata[s]["z_um"])
        self.zs = np.array([self.metadata[s]["z_um"] for s in self.sections])
        self.feature_dim = (
            self.metadata[self.sections[0]]["shape"][-1] if self.sections else None
        )

    @staticmethod
    def _key(sid):
        return hashlib.sha256(str(sid).encode()).hexdigest()

    def add_section(
        self,
        section_id,
        features,
        *,
        z_um,
        origin_um,
        spacing_um,
        patch_size_um,
        mpp,
        coordinate_frame,
        valid_mask=None,
        provenance=None,
    ):
        f = np.asarray(features)
        if f.ndim != 3 or min(f.shape) < 1 or (not np.isfinite(f).all()):
            raise ValueError("Features must be finite [H,W,D]")
        for key, val in [("origin", origin_um), ("spacing", spacing_um), ("mpp", mpp)]:
            if (
                len(val) != 2
                or not np.isfinite(val).all()
                or (key != "origin" and min(val) <= 0)
            ):
                raise ValueError(f"Invalid {key}")
        if (
            not np.isfinite(z_um)
            or not np.isfinite(patch_size_um)
            or patch_size_um <= 0
            or (not coordinate_frame)
        ):
            raise ValueError(
                "Finite physical Z, positive FOV and explicit frame required"
            )
        sid = str(section_id)
        if sid in self.metadata or z_um in self.zs:
            raise ValueError(
                "Duplicate section ID or physical Z; use a separate series store"
            )
        if self.sections:
            ref = self.metadata[self.sections[0]]
            if (
                ref["coordinate_frame"] != coordinate_frame
                or self.feature_dim != f.shape[-1]
            ):
                raise ValueError("Feature dimensions and coordinate frame must agree")
            if ref["patch_size_um"] != float(patch_size_um):
                raise ValueError("Physical patch FOV must be constant across sections")
        mask = (
            np.ones(f.shape[:2], bool)
            if valid_mask is None
            else np.asarray(valid_mask, bool)
        )
        if mask.shape != f.shape[:2]:
            raise ValueError("valid_mask must have shape [H,W]")
        meta = dict(
            z_um=float(z_um),
            origin_um=list(map(float, origin_um)),
            spacing_um=list(map(float, spacing_um)),
            patch_size_um=float(patch_size_um),
            mpp=list(map(float, mpp)),
            coordinate_frame=coordinate_frame,
            shape=list(f.shape),
            provenance=provenance or {},
        )
        with h5py.File(self.path, "a") as out:
            pending = "_pending_" + str(uuid.uuid4())
            g = out["sections"].create_group(pending)
            g.attrs["section_id"] = sid
            g.attrs["metadata"] = json.dumps(meta)
            g.create_dataset(
                "features",
                data=f.astype("float32"),
                chunks=(min(16, f.shape[0]), min(16, f.shape[1]), f.shape[2]),
                compression="lzf",
            )
            g.create_dataset("valid", data=mask, chunks=True, compression="lzf")
            out.flush()
            out["sections"].move(pending, self._key(sid))
            out.attrs["revision"] = str(uuid.uuid4())
        self.refresh()

    def _corners(self, sid, iy, ix):
        """Lazy gather, caching small planes; large planes are read by row spans."""
        if sid in self._cache:
            f, m = self._cache.pop(sid)
            self._cache[sid] = (f, m)
            return (f[iy, ix], m[iy, ix])
        with h5py.File(self.path, "r") as file:
            g = file["sections"][self._key(sid)]
            d = g["features"]
            valid = g["valid"]
            size = int(np.prod(d.shape)) * 4 + int(np.prod(d.shape[:2]))
            if size <= self.cache_bytes:
                while (
                    self._cache
                    and sum((a.nbytes + b.nbytes for a, b in self._cache.values()))
                    + size
                    > self.cache_bytes
                ):
                    self._cache.popitem(last=False)
                f, m = (d[:], valid[:])
                self._cache[sid] = (f, m)
                return (f[iy, ix], m[iy, ix])
            out = np.empty((len(iy), d.shape[-1]), np.float32)
            ok = np.empty(len(iy), bool)
            for row in np.unique(iy):
                idx = np.flatnonzero(iy == row)
                cols, inverse = np.unique(ix[idx], return_inverse=True)
                out[idx] = d[int(row), cols, :][inverse]
                ok[idx] = valid[int(row), cols][inverse]
            return (out, ok)

    def _nearest_valid(self, sid, xy, maximum_distance_um):
        """Return nearest valid grid features and a bounded acceptance mask."""
        from scipy.spatial import cKDTree

        if sid not in self._spatial_indices:
            metadata = self.metadata[sid]
            with h5py.File(self.path, "r") as file:
                valid = file["sections"][self._key(sid)]["valid"][:]
            iy, ix = np.nonzero(valid)
            coordinates = np.column_stack(
                [
                    metadata["origin_um"][0] + ix * metadata["spacing_um"][0],
                    metadata["origin_um"][1] + iy * metadata["spacing_um"][1],
                ]
            )
            self._spatial_indices[sid] = (cKDTree(coordinates), iy, ix)
        tree, iy, ix = self._spatial_indices[sid]
        distance, nearest = tree.query(np.asarray(xy, dtype=float), k=1)
        accepted = np.asarray(distance) <= float(maximum_distance_um)
        values = np.zeros((len(xy), self.feature_dim), dtype=np.float32)
        if accepted.any():
            selected = np.asarray(nearest)[accepted]
            gathered, _ = self._corners(sid, iy[selected], ix[selected])
            values[accepted] = gathered
        return (values, accepted)

    def get_feature(
        self,
        section_id,
        xy,
        *,
        return_valid=False,
        return_quality=False,
        nearest_max_distance_um=None,
        minimum_valid_weight=1e-06,
    ):
        """Query masked bilinear morphology features in physical XY.

        Quality codes are 0=unavailable, 1=full bilinear, 2=partial masked
        bilinear and 3=bounded nearest-valid fallback.
        """
        xy = torch.as_tensor(xy)
        if not xy.is_floating_point():
            xy = xy.float()
        if xy.ndim != 2 or xy.shape[1] != 2 or (not torch.isfinite(xy).all()):
            raise ValueError("xy must be finite [N,2]")
        sid = str(section_id)
        m = self.metadata[sid]
        h, w, _ = m["shape"]
        uv = (xy.detach().cpu().numpy() - m["origin_um"]) / m["spacing_um"]
        inside = (
            (uv[:, 0] >= -1e-06)
            & (uv[:, 0] <= w - 1 + 1e-06)
            & (uv[:, 1] >= -1e-06)
            & (uv[:, 1] <= h - 1 + 1e-06)
        )
        uv = np.clip(uv, [0, 0], [w - 1, h - 1])
        lo = np.floor(uv).astype(int)
        hi = np.minimum(lo + 1, [w - 1, h - 1])
        frac = uv - lo
        values = np.zeros((len(xy), self.feature_dim), np.float32)
        valid_weight = np.zeros(len(xy), dtype=np.float64)
        all_active_corners_valid = np.ones(len(xy), dtype=bool)
        for a, b in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            ix = hi[:, 0] if a else lo[:, 0]
            iy = hi[:, 1] if b else lo[:, 1]
            weight = (frac[:, 0] if a else 1 - frac[:, 0]) * (
                frac[:, 1] if b else 1 - frac[:, 1]
            )
            v, valid = self._corners(sid, iy, ix)
            effective = weight * valid
            values += v * effective[:, None]
            valid_weight += effective
            all_active_corners_valid &= valid | (weight < 1e-08)
        ok = (
            inside
            & (valid_weight >= float(minimum_valid_weight))
            & np.isfinite(valid_weight)
        )
        values[ok] /= valid_weight[ok, None]
        ok &= np.linalg.norm(values, axis=1) > 1e-10
        quality = np.zeros(len(xy), dtype=np.uint8)
        quality[ok & all_active_corners_valid] = 1
        quality[ok & ~all_active_corners_valid] = 2
        maximum = (
            1.5 * max(m["spacing_um"])
            if nearest_max_distance_um is None
            else float(nearest_max_distance_um)
        )
        missing = ~ok
        if maximum > 0 and missing.any():
            nearest_values, nearest_ok = self._nearest_valid(
                sid, xy.detach().cpu().numpy()[missing], maximum
            )
            recovered = np.flatnonzero(missing)[nearest_ok]
            values[recovered] = nearest_values[nearest_ok]
            ok[recovered] = True
            quality[recovered] = 3
        values[~ok] = 0
        result = torch.as_tensor(values, device=xy.device, dtype=xy.dtype)
        mask = torch.as_tensor(ok, device=xy.device)
        methods = torch.as_tensor(quality, device=xy.device)
        if not return_valid and (not return_quality) and (not ok.all()):
            raise ValueError(
                f"{sid}: {int((~ok).sum())} feature queries outside valid tissue/grid"
            )
        if return_quality:
            return (result, mask, methods)
        return (result, mask) if return_valid else result

    def query(self, xy, z_um, *, return_valid=False):
        xy = torch.as_tensor(xy)
        if not xy.is_floating_point():
            xy = xy.float()
        z = np.broadcast_to(
            torch.as_tensor(z_um).detach().cpu().numpy().reshape(-1), (len(xy),)
        )
        if (
            not len(self.zs)
            or not np.isfinite(z).all()
            or (z < self.zs[0] - 1e-05).any()
            or (z > self.zs[-1] + 1e-05).any()
        ):
            raise ValueError("Physical Z outside feature coverage")
        z = np.clip(z, self.zs[0], self.zs[-1])
        hi = np.searchsorted(self.zs, z, side="left").clip(0, len(self.zs) - 1)
        lo = np.maximum(hi - 1, 0)
        lo[np.isclose(z, self.zs[hi], atol=1e-07, rtol=0)] = hi[
            np.isclose(z, self.zs[hi], atol=1e-07, rtol=0)
        ]
        out = torch.empty((len(xy), self.feature_dim), device=xy.device, dtype=xy.dtype)
        valid = torch.ones(len(xy), device=xy.device, dtype=torch.bool)
        for lower_section, upper_section in np.unique(np.stack([lo, hi], 1), axis=0):
            ids = np.flatnonzero((lo == lower_section) & (hi == upper_section))
            idx = torch.as_tensor(ids, device=xy.device)
            f0, m0 = self.get_feature(
                self.sections[lower_section], xy[idx], return_valid=True
            )
            if lower_section == upper_section:
                out[idx] = f0
                valid[idx] = m0
                continue
            f1, m1 = self.get_feature(
                self.sections[upper_section], xy[idx], return_valid=True
            )
            a = torch.as_tensor(
                (z[ids] - self.zs[lower_section])
                / (self.zs[upper_section] - self.zs[lower_section]),
                device=xy.device,
                dtype=xy.dtype,
            )[:, None]
            out[idx] = (1 - a) * f0 + a * f1
            valid[idx] = m0 & m1
        if not return_valid and (not valid.all()):
            raise ValueError("Invalid XY feature query at interpolated Z")
        return (out, valid) if return_valid else out


class UNI2Encoder:
    """Frozen UNI2-h encoder restored from an explicitly supplied local checkpoint."""

    feature_dim = 1536

    def __init__(self, checkpoint, device="cuda"):
        import timm

        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"UNI2-h checkpoint not found: {checkpoint}")
        kwargs = dict(
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-05,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        )
        self.model = timm.create_model(
            "vit_giant_patch14_224", pretrained=False, **kwargs
        )
        self.model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.device = torch.device(device)
        self.model.requires_grad_(False).eval().to(self.device)

    @torch.inference_mode()
    def encode(self, patches, batch_size=32):
        """PIL RGB patches -> CPU float32 [N,1536]; no autograd or fine-tuning."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model.eval()
        results = []
        for start in range(0, len(patches), batch_size):
            batch = torch.stack(
                [
                    self.transform(p.convert("RGB"))
                    for p in patches[start : start + batch_size]
                ]
            )
            out = self.model(batch.to(self.device))
            if out.ndim != 2 or out.shape[1] != self.feature_dim:
                raise ValueError("Expected UNI2-h CLS embeddings [B,1536]")
            results.append(out.float().cpu())
        return torch.cat(results) if results else torch.empty((0, self.feature_dim))


def precompute_native_section(
    row,
    encoder,
    store,
    *,
    spacing_um=56.0,
    patch_size_um=112.0,
    batch_size=8,
    checkpoint_sha256=None,
    roi_um=None,
    feature_halo_um=112.0,
    coordinate_frame=None,
):
    """Address features in registered XY while reading only native crop pixels."""
    coordinate_frame = (
        row.coordinate_frame if coordinate_frame is None else coordinate_frame
    )
    if spacing_um <= 0 or patch_size_um <= 0 or batch_size <= 0:
        raise ValueError(
            "Grid spacing, physical FOV and extraction batch size must be positive."
        )
    reader = NativeCropReader(row)
    if roi_um is None:
        low, high = reader.registered_tissue_bounds()
        low = np.floor(low / spacing_um) * spacing_um
        high = np.ceil(high / spacing_um) * spacing_um
    else:
        xmin, ymin, xmax, ymax = map(float, roi_um)
        if (
            not np.isfinite(roi_um).all()
            or xmin >= xmax
            or ymin >= ymax
            or (feature_halo_um < 0)
        ):
            raise ValueError("Invalid physical ROI or halo.")
        low = np.array([xmin, ymin]) - feature_halo_um
        high = np.array([xmax, ymax]) + feature_halo_um
    nx, ny = np.floor((high - low) / spacing_um).astype(int) + 1
    iy, ix = np.mgrid[:ny, :nx]
    registered = np.column_stack([ix.ravel(), iy.ravel()]) * spacing_um + low
    source = reader.registered_to_source(registered)
    roundtrip = source_to_registered_points(source, reader.section, reader.transforms)
    inverse_error = float(np.max(np.linalg.norm(roundtrip - registered, axis=1)))
    if inverse_error > 0.02:
        raise ValueError(
            f"Inverse registration residual too large: {inverse_error:.6g} um"
        )
    valid = reader.center_supported(source)
    if not valid.any():
        raise ValueError(
            f"Section {row.section_id} has no native tissue-supported grid centers."
        )
    features = np.zeros((len(registered), encoder.feature_dim), np.float32)
    ids = np.flatnonzero(valid)
    for start in range(0, len(ids), int(batch_size)):
        chosen = ids[start : start + batch_size]
        patches = [
            reader.extract_patch(point, patch_size_um=patch_size_um)
            for point in source[chosen]
        ]
        values = encoder.encode(patches, batch_size=batch_size).numpy()
        if values.shape != (len(chosen), 1536) or not np.isfinite(values).all():
            raise ValueError(
                "Official UNI2 must produce finite 1536-dimensional embeddings."
            )
        features[chosen] = values
    details = {
        "encoder": "UNI2-h_frozen",
        "checkpoint_sha256": checkpoint_sha256,
        "source_kind": "native_level0_ome_tiff",
        "image": Path(row.he_file).name,
        "native_mpp_um": reader.mpp,
        "effective_encoder_mpp_um": patch_size_um / 224,
        "center_mapping": "saved_nonlinear_chain_numerically_inverted",
        "inverse_roundtrip_max_um": inverse_error,
        "patch_geometry": "source_orientation_no_local_warp",
        "mask_policy": "native_tissue_at_patch_center",
        "coordinate_frame": coordinate_frame,
        "data_release": "00029_g0_reproducibility_bundle_v1",
    }
    store.add_section(
        str(row.section_id),
        features.reshape(ny, nx, -1),
        z_um=float(row.z_um),
        origin_um=low,
        spacing_um=(spacing_um,) * 2,
        patch_size_um=float(patch_size_um),
        mpp=(patch_size_um / 224,) * 2,
        coordinate_frame=coordinate_frame,
        valid_mask=valid.reshape(ny, nx),
        provenance=details,
    )
    reader.transforms.clear()
    return {
        "section_id": int(row.section_id),
        "patches": int(valid.sum()),
        "grid_points": len(valid),
        "inverse_roundtrip_max_um": inverse_error,
        "native_mpp_um": reader.mpp,
    }


def align_nucleus_grids(source_path, output_path, morphology):
    """Resample supplied seven-channel anatomy descriptors onto the current feature grid.

    This uses registered physical XY and supplied descriptors, not inferred cell
    labels or newly segmented nuclei. The original cache is never modified.
    """
    original = FeatureStore(source_path)
    aligned = FeatureStore(require_new(output_path), mode="x")
    for section in morphology.sections:
        target = morphology.metadata[section]
        reference = original.metadata[section]
        if reference["coordinate_frame"] != target[
            "coordinate_frame"
        ] or not np.isclose(reference["z_um"], target["z_um"]):
            raise ValueError(
                "Nuclear anatomy and morphology do not share registered coordinates."
            )
        (ny, nx) = target["shape"][:2]
        (iy, ix) = np.mgrid[:ny, :nx]
        xy = (
            np.column_stack([ix.ravel(), iy.ravel()]) * target["spacing_um"]
            + target["origin_um"]
        )
        (values, valid) = original.get_feature(section, xy, return_valid=True)
        aligned.add_section(
            section,
            values.numpy().reshape(ny, nx, -1),
            z_um=target["z_um"],
            origin_um=target["origin_um"],
            spacing_um=target["spacing_um"],
            patch_size_um=target["patch_size_um"],
            mpp=target["mpp"],
            coordinate_frame=target["coordinate_frame"],
            valid_mask=valid.numpy().reshape(ny, nx),
            provenance={
                "source_revision": original.revision,
                "semantics": "supplied local nuclear anatomy, not cell lineage",
            },
        )
    return aligned

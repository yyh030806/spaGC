"""Morphology-guided continuous spatial paths between anchors."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import OrderedDict

import h5py
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator


class MorphologyPathCache:
    """Cache morphology-guided paths between matched anchor cells."""

    def __init__(
        self,
        path,
        feature_store,
        *,
        coordinate_frame=None,
        candidate_radius_um=100.0,
        candidate_spacing_um=25.0,
        move_weight=0.001,
        morphology_weight=1.0,
        nucleus_store=None,
        nucleus_weight=0.0,
        max_candidates=1024,
        allow_linear_fallback=False,
    ):
        self.path = str(path)
        self.store = feature_store
        self._memory = OrderedDict()
        store_frames = {
            str(feature_store.metadata[section_id]["coordinate_frame"])
            for section_id in feature_store.sections
        }
        if len(store_frames) != 1:
            raise ValueError(
                f"Morphology feature store must use one coordinate frame, got {sorted(store_frames)}"
            )
        store_frame = next(iter(store_frames))
        if coordinate_frame is not None and str(coordinate_frame) != store_frame:
            raise ValueError(
                f"Path coordinate frame does not match morphology feature store: requested={coordinate_frame!r}, store={store_frame!r}"
            )
        self.coordinate_frame = store_frame
        self.params = dict(
            candidate_radius_um=float(candidate_radius_um),
            candidate_spacing_um=float(candidate_spacing_um),
            move_weight=float(move_weight),
            morphology_weight=float(morphology_weight),
            nucleus_weight=float(nucleus_weight),
            max_candidates=int(max_candidates),
            allow_linear_fallback=bool(allow_linear_fallback),
        )
        if (
            candidate_radius_um < 0
            or candidate_spacing_um <= 0
            or min(move_weight, morphology_weight, nucleus_weight) < 0
        ):
            raise ValueError("Invalid path search parameters")
        self.nucleus_store = nucleus_store
        if nucleus_weight > 0 and nucleus_store is None:
            raise ValueError("Positive nucleus_weight requires nucleus_store")
        if nucleus_store is not None:
            if nucleus_store.sections != feature_store.sections:
                raise ValueError(
                    "Nucleus and morphology stores must contain the same sections"
                )
            for sid in feature_store.sections:
                left = feature_store.metadata[sid]
                right = nucleus_store.metadata[sid]
                if (
                    left["coordinate_frame"] != right["coordinate_frame"]
                    or left["shape"][:2] != right["shape"][:2]
                    or (not np.allclose(left["origin_um"], right["origin_um"]))
                    or (not np.allclose(left["spacing_um"], right["spacing_um"]))
                    or (not np.isclose(left["z_um"], right["z_um"]))
                ):
                    raise ValueError(f"Nucleus feature grid mismatch for section {sid}")
        n = int(np.floor(candidate_radius_um / candidate_spacing_um))
        if (2 * n + 1) ** 2 > max_candidates:
            raise ValueError("Local candidate grid exceeds max_candidates")
        a = np.arange(-n, n + 1) * candidate_spacing_um
        xx, yy = np.meshgrid(a, a)
        self.offsets = np.stack([xx.ravel(), yy.ravel()], 1)
        path_preexisted = h5py.is_hdf5(self.path)
        with h5py.File(self.path, "a") as f:
            f.require_group("paths")
            f.attrs["format"] = "deepspatial-pchip-t-v1"
            existing_frame = f.attrs.get("coordinate_frame")
            if (
                existing_frame is not None
                and str(existing_frame) != self.coordinate_frame
            ):
                raise ValueError(
                    f"Path cache coordinate frame does not match feature store: cache={existing_frame!r}, store={self.coordinate_frame!r}"
                )
            if existing_frame is None and (not path_preexisted):
                f.attrs["coordinate_frame"] = self.coordinate_frame

    def prepare(self, s0, s1, x0, x1, pair_ids):
        """Precompute unique selected pairs; persistent content-addressed IDs."""
        x0 = np.asarray(x0)
        x1 = np.asarray(x1)
        if (
            x0.shape != x1.shape
            or x0.shape != (len(pair_ids), 2)
            or (not np.isfinite([x0, x1]).all())
        ):
            raise ValueError("Endpoints must be finite [N,2] with N pair IDs")
        ids = []
        with h5py.File(self.path, "a") as f:
            for a, b, pair in zip(x0, x1, pair_ids):
                info = dict(
                    version=2 if self.nucleus_store is not None else 1,
                    feature_revision=self.store.revision,
                    nucleus_feature_revision=self.nucleus_store.revision
                    if self.nucleus_store is not None
                    else None,
                    s0=s0,
                    s1=s1,
                    x0=a.tolist(),
                    x1=b.tolist(),
                    pair_id=str(pair),
                    coordinate_frame=self.coordinate_frame,
                    params=self.params,
                )
                key = hashlib.sha256(
                    json.dumps(info, sort_keys=True).encode()
                ).hexdigest()
                ids.append(key)
                if key in f["paths"]:
                    continue
                ts, nodes, coeff, fallback_sections, nucleus_fallback_sections = (
                    self._construct(s0, s1, a, b)
                )
                pending = "_pending_" + str(uuid.uuid4())
                g = f["paths"].create_group(pending)
                g.attrs["metadata"] = json.dumps(info)
                g.attrs["morphology_fallback_sections"] = json.dumps(fallback_sections)
                g.attrs["nucleus_fallback_sections"] = json.dumps(
                    nucleus_fallback_sections
                )
                g.create_dataset("t", data=ts)
                g.create_dataset("xy_um", data=nodes)
                g.create_dataset("coefficients", data=coeff)
                f.flush()
                f["paths"].move(pending, key)
        return ids

    def evaluate(self, ids, t):
        """Cached coefficients -> position and dx/dt, both [B,2], on t.device."""
        t = torch.as_tensor(t).reshape(-1)
        if (
            len(ids) != len(t)
            or not torch.isfinite(t).all()
            or (t < 0).any()
            or (t > 1).any()
        ):
            raise ValueError("One t in [0,1] per cached endpoint pair required")
        groups = {}
        with h5py.File(self.path, "r") as f:
            for i, key in enumerate(ids):
                if key in self._memory:
                    knots, coeff = self._memory.pop(key)
                else:
                    g = f["paths"][key]
                    knots, coeff = (g["t"][:], g["coefficients"][:])
                self._memory[key] = (knots, coeff)
                if len(self._memory) > 4096:
                    self._memory.popitem(last=False)
                groups.setdefault(tuple(knots), []).append((i, coeff))
        x = torch.empty((len(t), 2), device=t.device, dtype=t.dtype)
        v = torch.empty_like(x)
        for knots, entries in groups.items():
            idx = torch.tensor([i for i, _ in entries], device=t.device)
            k = torch.tensor(knots, device=t.device, dtype=t.dtype)
            c = torch.as_tensor(
                np.stack([c for _, c in entries]), device=t.device, dtype=t.dtype
            )
            seg = (
                torch.searchsorted(k, t[idx], right=True)
                .sub(1)
                .clamp(0, len(knots) - 2)
            )
            c = c[torch.arange(len(idx), device=t.device), :, seg, :]
            dt = (t[idx] - k[seg])[:, None]
            x[idx] = ((c[:, 0] * dt + c[:, 1]) * dt + c[:, 2]) * dt + c[:, 3]
            v[idx] = (3 * c[:, 0] * dt + 2 * c[:, 1]) * dt + c[:, 2]
        return (x, v)

    def _construct(self, s0, s1, x0, x1):
        z0 = self.store.metadata[s0]["z_um"]
        z1 = self.store.metadata[s1]["z_um"]
        if z1 <= z0:
            raise ValueError("Anchor sections must have increasing physical Z")
        sections = (
            [s0]
            + [
                s
                for s in self.store.sections
                if z0 < self.store.metadata[s]["z_um"] < z1
            ]
            + [s1]
        )
        ts = np.array(
            [(self.store.metadata[s]["z_um"] - z0) / (z1 - z0) for s in sections]
        )
        candidates = []
        features = []
        feature_validity = []
        nucleus_features = []
        nucleus_validity = []
        fallback_sections = []
        nucleus_fallback_sections = []

        def add_unique(values, section):
            section = str(section)
            if section not in values:
                values.append(section)

        for k, (s, t) in enumerate(zip(sections, ts)):
            p = (
                x0[None]
                if k == 0
                else x1[None]
                if k == len(sections) - 1
                else ((1 - t) * x0 + t * x1)[None] + self.offsets
            )
            h, valid = self.store.get_feature(s, p, return_valid=True)
            valid = valid.numpy()
            h = h.numpy()
            if self.nucleus_store is not None:
                nh, nvalid = self.nucleus_store.get_feature(s, p, return_valid=True)
                nvalid = nvalid.numpy()
                nh = nh.numpy()
                if not nvalid.any():
                    add_unique(nucleus_fallback_sections, s)
            if k in (0, len(sections) - 1) and (not valid.all()):
                add_unique(fallback_sections, s)
            if k not in (0, len(sections) - 1):
                if valid.any():
                    keep = valid.copy()
                    p = p[keep]
                    h = h[keep]
                    valid = valid[keep]
                    if self.nucleus_store is not None:
                        nh = nh[keep]
                        nvalid = nvalid[keep]
                else:
                    center = ((1 - t) * x0 + t * x1)[None]
                    metadata = self.store.metadata[s]
                    uv = (center[0] - np.asarray(metadata["origin_um"])) / np.asarray(
                        metadata["spacing_um"]
                    )
                    height, width, _ = metadata["shape"]
                    inside_grid = (
                        -1e-06 <= uv[0] <= width - 1 + 1e-06
                        and -1e-06 <= uv[1] <= height - 1 + 1e-06
                    )
                    if not inside_grid:
                        if not self.params["allow_linear_fallback"]:
                            raise ValueError(
                                f"No valid morphology candidates in {s}; inspect mask/coverage/radius"
                            )
                    fallback_sections.append(str(s))
                    p = center
                    h = np.zeros((1, self.store.feature_dim), dtype=np.float32)
                    valid = np.zeros(1, dtype=bool)
                    if self.nucleus_store is not None:
                        nh, nvalid = self.nucleus_store.get_feature(
                            s, p, return_valid=True
                        )
                        nh = nh.numpy()
                        nvalid = nvalid.numpy()
                        if not nvalid.any():
                            add_unique(nucleus_fallback_sections, s)
            candidates.append(p)
            features.append(
                h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
            )
            feature_validity.append(valid)
            if self.nucleus_store is not None:
                nucleus_features.append(
                    nh / np.maximum(np.linalg.norm(nh, axis=1, keepdims=True), 1e-12)
                )
                nucleus_validity.append(nvalid)
        score = np.zeros(1)
        back = []
        for k in range(1, len(sections)):
            move = (
                (candidates[k - 1][:, None, :] - candidates[k][None, :, :]) ** 2
            ).sum(-1)
            morph = np.clip(1 - features[k - 1] @ features[k].T, 0, 2)
            both_valid = feature_validity[k - 1][:, None] & feature_validity[k][None, :]
            morph = np.where(both_valid, morph, 0.0)
            nucleus = np.zeros_like(morph)
            if self.nucleus_store is not None and self.params["nucleus_weight"] > 0:
                nucleus = np.clip(
                    1 - nucleus_features[k - 1] @ nucleus_features[k].T, 0, 2
                )
                both_nucleus_valid = (
                    nucleus_validity[k - 1][:, None] & nucleus_validity[k][None, :]
                )
                nucleus = np.where(both_nucleus_valid, nucleus, 0.0)
            cost = (
                score[:, None]
                + self.params["move_weight"] * move
                + self.params["morphology_weight"] * morph
                + self.params["nucleus_weight"] * nucleus
            )
            prev = cost.argmin(0)
            back.append(prev)
            score = cost[prev, np.arange(cost.shape[1])]
        route = [0]
        for prev in reversed(back):
            route.append(int(prev[route[-1]]))
        nodes = np.array([p[i] for p, i in zip(candidates, reversed(route))])
        spline = PchipInterpolator(ts, nodes, axis=0)
        return (ts, nodes, spline.c, fallback_sections, nucleus_fallback_sections)

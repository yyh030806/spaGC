"""Morphology supervision configuration and physical-coordinate queries."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from spagc.reconstruction.features import FeatureStore
from spagc.reconstruction.paths import MorphologyPathCache


@dataclass
class HistologyConfig:
    """One registered tissue/g series per store; all physical values in um."""

    use_histology: bool = False
    use_morphology_uot: bool = False
    use_morphology_path: bool = False
    use_nucleus_path: bool = False
    feature_path: str | None = None
    nucleus_feature_path: str | None = None
    path_cache: str | None = None
    section_key: str = "section_id"
    coordinate_frame: str | None = None
    spatial_weight: float | None = None
    gene_weight: float | None = None
    histology_weight: float = 1.0
    class_weight: float = 10.0
    spatial_metric: str = "legacy_normalized"
    candidate_radius_um: float = 100.0
    candidate_spacing_um: float = 25.0
    move_weight: float = 0.001
    morphology_weight: float = 1.0
    nucleus_weight: float = 0.0
    max_candidates: int = 1024
    allow_linear_fallback: bool = False
    max_uot_entries: int = 25000000
    feature_cache_mb: int = 512
    seed: int = 0

    def __post_init__(self):
        if self.spatial_metric not in ("legacy_normalized", "physical_um"):
            raise ValueError("spatial_metric must be legacy_normalized or physical_um")
        if not self.use_histology and any(
            (self.use_morphology_uot, self.use_morphology_path, self.use_nucleus_path)
        ):
            raise ValueError("Histology subcomponents require use_histology=True")
        if self.use_histology and (not self.feature_path or not self.coordinate_frame):
            raise ValueError(
                "Histology requires feature_path and explicit coordinate_frame"
            )
        if self.use_morphology_path and (not self.path_cache):
            raise ValueError("Morphology path requires a persistent path_cache")
        if self.use_nucleus_path and (not self.use_morphology_path):
            raise ValueError("Nucleus path requires use_morphology_path=True")
        if self.use_nucleus_path and (not self.nucleus_feature_path):
            raise ValueError("Nucleus path requires nucleus_feature_path")
        for key in (
            "histology_weight",
            "class_weight",
            "move_weight",
            "morphology_weight",
            "nucleus_weight",
            "candidate_radius_um",
        ):
            v = getattr(self, key)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in ("spatial_weight", "gene_weight"):
            v = getattr(self, key)
            if v is not None and (not math.isfinite(v) or v < 0):
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in (
            "candidate_spacing_um",
            "max_candidates",
            "max_uot_entries",
            "feature_cache_mb",
        ):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")

    def to_dict(self):
        return asdict(self)


class HistologyRuntime:
    def __init__(self, config, spatial_stats):
        self.config = config
        self.stats = spatial_stats
        self.path_ids = []
        self.store = FeatureStore(config.feature_path, cache_mb=config.feature_cache_mb)
        self.nucleus_store = None
        if config.use_nucleus_path:
            self.nucleus_store = FeatureStore(
                config.nucleus_feature_path, cache_mb=config.feature_cache_mb
            )
            if self.nucleus_store.sections != self.store.sections:
                raise ValueError(
                    "Nucleus and UNI2 stores must contain the same sections"
                )
            for sid in self.store.sections:
                left = self.store.metadata[sid]
                right = self.nucleus_store.metadata[sid]
                if (
                    left["coordinate_frame"] != right["coordinate_frame"]
                    or left["shape"][:2] != right["shape"][:2]
                    or (not np.allclose(left["origin_um"], right["origin_um"]))
                    or (not np.allclose(left["spacing_um"], right["spacing_um"]))
                    or (not np.isclose(left["z_um"], right["z_um"]))
                ):
                    raise ValueError(f"Nucleus store grid mismatch for section {sid}")
        if not self.store.sections:
            raise ValueError("Empty histology feature store")
        if any(
            m["coordinate_frame"] != config.coordinate_frame
            for m in self.store.metadata.values()
        ):
            raise ValueError(
                "Feature coordinate_frame differs from requested registered series"
            )
        self.paths = (
            MorphologyPathCache(
                config.path_cache,
                self.store,
                candidate_radius_um=config.candidate_radius_um,
                candidate_spacing_um=config.candidate_spacing_um,
                move_weight=config.move_weight,
                morphology_weight=config.morphology_weight,
                nucleus_store=self.nucleus_store,
                nucleus_weight=config.nucleus_weight,
                max_candidates=config.max_candidates,
                allow_linear_fallback=config.allow_linear_fallback,
            )
            if config.use_morphology_path
            else None
        )

    def xy_um(self, x):
        offset = [self.stats["x_min"], self.stats["y_min"]]
        scale = [self.stats["x_range"], self.stats["y_range"]]
        if torch.is_tensor(x):
            return x * x.new_tensor(scale) + x.new_tensor(offset)
        return x * np.array(scale) + np.array(offset)

    def features(self, x, z):
        z_um = z * self.stats["z_range"] + self.stats["z_min"]
        (features, _) = self.store.query(self.xy_um(x), z_um, return_valid=True)
        return features

    def plan(self, indices, t):
        ids = [self.path_ids[i] for i in indices.detach().cpu().reshape(-1).tolist()]
        (x, v) = self.paths.evaluate(ids, t)
        scale = x.new_tensor([self.stats["x_range"], self.stats["y_range"]])
        origin = x.new_tensor([self.stats["x_min"], self.stats["y_min"]])
        return ((x - origin) / scale, v / scale)

    def section(self, adata):
        key = self.config.section_key
        if key not in adata.obs or adata.obs[key].nunique() != 1:
            raise ValueError(f"Each anchor needs exactly one obs[{key!r}]")
        sid = str(adata.obs[key].iloc[0])
        if sid not in self.store.metadata:
            raise ValueError(f"Missing anchor H&E feature plane: {sid}")
        z = (
            float(adata.obs["z_norm"].iloc[0]) * self.stats["z_range"]
            + self.stats["z_min"]
        )
        if not np.isclose(z, self.store.metadata[sid]["z_um"], rtol=0, atol=0.0001):
            raise ValueError(f"Physical Z mismatch for {sid}")
        return sid

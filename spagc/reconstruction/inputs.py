"""Measured anchors and native-image manifests with explicit coordinate keys."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_anchors(
    directory,
    *,
    require_labels=False,
    labels_path=None,
    max_cells=None,
    max_genes=None,
    section_range=None,
    roi_um=None,
    max_cells_per_anchor=None,
    seed=29,
    label_key="cell_class",
    spatial_key="spatial_st_corrected",
    z_key="z_um",
):
    """Load processed anchors in physical-Z order without rescaling expression."""
    paths = sorted(Path(directory).glob("section-*.h5ad"))
    if section_range is not None:
        low, high = section_range
        paths = [p for p in paths if low <= int(p.stem.split("-")[-1]) <= high]
    if len(paths) < 2:
        raise FileNotFoundError(f"At least two anchor H5ADs are required: {directory}")
    label_table = None
    if labels_path is not None and Path(labels_path).is_file():
        label_table = pd.read_csv(labels_path, dtype={"cell_id": str}).set_index(
            "cell_id"
        )
        if label_table.index.has_duplicates or label_key not in label_table:
            raise ValueError(
                "Cell-type CSV requires unique cell_id and the declared label column."
            )
    anchors = []
    for path in paths:
        obj = ad.read_h5ad(path)
        if max_cells is not None:
            obj = obj[:max_cells, :max_genes].copy()
        if (
            spatial_key not in obj.obsm
            or z_key not in obj.obs
            or "section_id" not in obj.obs
        ):
            raise KeyError(f"Missing spatial/depth/section metadata: {path}")
        xy = np.asarray(obj.obsm[spatial_key])
        zs = np.asarray(obj.obs[z_key], dtype=float)
        if (
            xy.shape != (obj.n_obs, 2)
            or not np.isfinite(xy).all()
            or (not np.isfinite(zs).all())
        ):
            raise ValueError(f"Invalid physical coordinates: {path}")
        if not len(zs) or len(np.unique(zs)) != 1 or obj.obs.section_id.nunique() != 1:
            raise ValueError(
                "Each nonempty anchor must contain exactly one physical depth and section."
            )
        if obj.obs_names.has_duplicates or obj.var_names.has_duplicates:
            raise ValueError(
                "Cell and gene identifiers must be unique within each anchor."
            )
        values = obj.X.data if sp.issparse(obj.X) else np.asarray(obj.X)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Expression must be finite and nonnegative.")
        if roi_um is not None:
            xmin, ymin, xmax, ymax = roi_um
            selected = (
                (xy[:, 0] >= xmin)
                & (xy[:, 0] <= xmax)
                & (xy[:, 1] >= ymin)
                & (xy[:, 1] <= ymax)
            )
            obj = obj[selected].copy()
            if obj.n_obs == 0:
                raise ValueError(f"No anchor cells in the declared ROI: {path.name}")
        if max_cells_per_anchor is not None and obj.n_obs > max_cells_per_anchor:
            rng = np.random.default_rng(seed + int(obj.obs.section_id.iloc[0]))
            selected = np.sort(
                rng.choice(obj.n_obs, int(max_cells_per_anchor), replace=False)
            )
            obj = obj[selected].copy()
        if label_table is not None:
            obj.obs[label_key] = pd.Categorical(
                label_table[label_key].reindex(obj.obs_names).to_numpy()
            )
        if require_labels and (
            label_key not in obj.obs or obj.obs[label_key].isna().any()
        ):
            raise ValueError(
                f"{path.name} has no complete {label_key!r} labels. Supply cell_type_labels.csv keyed by exact obs_names, or deliberately set USE_CELLTYPE=False for an ablation."
            )
        anchors.append(obj)
    anchors.sort(key=lambda a: float(a.obs[z_key].iloc[0]))
    if not all(a.var_names.equals(anchors[0].var_names) for a in anchors):
        raise ValueError(
            "Anchor gene identities and order differ; do not silently intersect genes."
        )
    if not (np.diff([float(a.obs[z_key].iloc[0]) for a in anchors]) > 0).all():
        raise ValueError("Anchor depths must increase strictly.")
    return anchors


def load_he_manifest(root, *, section_range=None, coordinate_frame=None):
    """Resolve only portable members of the native-resolution bundle."""
    root = Path(root).resolve()
    table = pd.read_csv(root / "manifest.csv")
    if section_range is not None:
        table = table[table.section_id.between(*section_range)].copy()
    table = table.sort_values("z_um").reset_index(drop=True)
    if table.empty or table.section_id.duplicated().any():
        raise ValueError("Select nonempty H&E sections with unique identifiers.")
    if not np.isfinite(table.z_um).all() or not (np.diff(table.z_um) > 0).all():
        raise ValueError("H&E depths must be finite and increase strictly.")
    members = []
    for row in table.itertuples(index=False):
        sid = int(row.section_id)
        metadata_file = root / f"metadata/section-{sid:03d}.json"
        metadata = json.loads(metadata_file.read_text())
        if coordinate_frame is None:
            coordinate_frame = metadata["coordinate_frame"]
        if metadata["coordinate_frame"] != coordinate_frame or metadata["image_warped"]:
            raise ValueError(
                "Expected native pixels and the declared registered frame."
            )
        resolved = {}
        for name, field in [
            ("image", "he_file"),
            ("mask", "mask_file"),
            ("section_transform", "section_transform"),
        ]:
            value = (metadata_file.parent / metadata["bundle_paths"][name]).resolve()
            if not value.is_relative_to(root) or not value.is_file():
                raise FileNotFoundError(f"Invalid bundled {name}: {value}")
            resolved[field] = value
        if not np.isclose(metadata["raw_mpp_um"], row.raw_mpp_um):
            raise ValueError("Manifest and metadata native pixel sizes disagree.")
        members.append(
            {
                **resolved,
                "metadata_file": metadata_file,
                "coordinate_frame": coordinate_frame,
                "native_pixel_size_um": float(row.raw_mpp_um),
            }
        )
    for field in members[0]:
        table[field] = [value[field] for value in members]
    return table

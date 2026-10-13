"""Reference annotation on independent physical-depth slabs."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.spatial import Delaunay, QhullError


def _decode(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind not in "SO":
        return values
    return np.asarray(
        [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in values
        ],
        dtype=object,
    )


def read_dataframe_column(group: h5py.Group, key: str) -> np.ndarray:
    obj = group[key]
    if isinstance(obj, h5py.Dataset):
        return _decode(obj[:])
    categories = _decode(obj["categories"][:])
    codes = obj["codes"][:]
    result = np.full(codes.shape[0], None, dtype=object)
    valid = codes >= 0
    result[valid] = categories[codes[valid]]
    return result


def read_csr_rows(
    group: h5py.Group, indptr: np.ndarray, start: int, end: int, n_vars: int
) -> sp.csr_matrix:
    pointer_start = int(indptr[start])
    pointer_end = int(indptr[end])
    data = group["data"][pointer_start:pointer_end].astype(np.float32, copy=False)
    indices = group["indices"][pointer_start:pointer_end]
    row_pointer = indptr[start : end + 1] - pointer_start
    return sp.csr_matrix((data, indices, row_pointer), shape=(end - start, n_vars))


def log_normalize(counts: sp.csr_matrix, target_sum: float = 10000.0) -> sp.csr_matrix:
    counts = counts.astype(np.float32, copy=True)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    counts = counts.multiply((target_sum / totals)[:, None]).tocsr()
    counts.data = np.log1p(counts.data)
    return counts


def build_delaunay_graph(coords: np.ndarray, percentile: float) -> sp.csr_matrix:
    """Construct the undirected, percentile-pruned 2D Delaunay graph."""
    n_obs = coords.shape[0]
    if n_obs < 3:
        return sp.csr_matrix((n_obs, n_obs), dtype=np.bool_)
    try:
        triangles = Delaunay(coords).simplices
    except QhullError:
        triangles = Delaunay(coords, qhull_options="QJ").simplices
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [0, 2]]], axis=0
    )
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    distances = np.linalg.norm(coords[rows] - coords[cols], axis=1)
    keep = distances <= np.percentile(distances, percentile)
    graph = sp.coo_matrix(
        (np.ones(int(keep.sum()), dtype=np.bool_), (rows[keep], cols[keep])),
        shape=(n_obs, n_obs),
    ).tocsr()
    graph.setdiag(False)
    graph.eliminate_zeros()
    return graph


def _without_diagonal(matrix: sp.csr_matrix) -> sp.csr_matrix:
    matrix = matrix.tolil(copy=True)
    matrix.setdiag(False)
    matrix = matrix.tocsr()
    matrix.eliminate_zeros()
    return matrix


def aggregate_neighbors(
    adjacency: sp.csr_matrix, features: np.ndarray, n_layers: int
) -> np.ndarray:
    """Reproduce CellCharter mean aggregation for layers 0..n_layers."""
    adjacency = _without_diagonal(adjacency.astype(np.bool_))
    hop = adjacency.copy()
    visited = adjacency.copy().tolil()
    visited.setdiag(True)
    visited = visited.tocsr()
    outputs = [np.asarray(features, dtype=np.float32)]
    for layer in range(1, n_layers + 1):
        if layer > 1:
            hop = hop @ adjacency
            hop = (hop > visited).tocsr()
            visited = (visited + hop).astype(np.bool_).tocsr()
        degrees = np.asarray(hop.sum(axis=1)).ravel().astype(np.float32)
        inverse = np.zeros_like(degrees)
        nonzero = degrees > 0
        inverse[nonzero] = 1.0 / degrees[nonzero]
        normalized = sp.diags(inverse) @ hop
        outputs.append(np.asarray(normalized @ features, dtype=np.float32))
    return np.hstack(outputs).astype(np.float32, copy=False)


def prediction_codes(model: dict, embedding: np.ndarray, threshold: float):
    probabilities = model["pipeline"].predict_proba(embedding)
    codes = probabilities.argmax(axis=1).astype(np.int16)
    confidence = probabilities.max(axis=1).astype(np.float32)
    codes[confidence < threshold] = len(model["classes"])
    categories = [str(value) for value in model["classes"]] + ["Unknown"]
    return (codes, confidence, categories)


def one_hot_from_codes(
    codes: np.ndarray, predicted_categories: list[str], reference_categories: list[str]
) -> np.ndarray:
    result = np.zeros((codes.size, len(reference_categories)), dtype=np.float32)
    category_to_reference = {
        name: index for (index, name) in enumerate(reference_categories)
    }
    for predicted_code, name in enumerate(predicted_categories):
        reference_code = category_to_reference.get(name)
        if reference_code is not None:
            result[codes == predicted_code, reference_code] = 1.0
    return result


def write_categorical(
    group: h5py.Group, key: str, codes: np.ndarray, categories: list[str]
):
    if key in group:
        del group[key]
    categorical = group.create_group(key)
    categorical.attrs["encoding-type"] = "categorical"
    categorical.attrs["encoding-version"] = "0.2.0"
    categorical.attrs["ordered"] = np.bool_(False)
    category_dtype = h5py.string_dtype(encoding="utf-8")
    category_ds = categorical.create_dataset(
        "categories", data=np.asarray(categories, dtype=object), dtype=category_dtype
    )
    category_ds.attrs["encoding-type"] = "string-array"
    category_ds.attrs["encoding-version"] = "0.2.0"
    code_dtype = np.int8 if len(categories) < 128 else np.int16
    code_ds = categorical.create_dataset("codes", data=codes.astype(code_dtype))
    code_ds.attrs["encoding-type"] = "array"
    code_ds.attrs["encoding-version"] = "0.2.0"


def write_array(group: h5py.Group, key: str, values: np.ndarray):
    if key in group:
        del group[key]
    dataset = group.create_dataset(
        key, data=values, compression="gzip", compression_opts=1
    )
    dataset.attrs["encoding-type"] = "array"
    dataset.attrs["encoding-version"] = "0.2.0"


def append_predictions_to_h5ad(
    input_path: Path,
    output_path: Path,
    niche_codes: np.ndarray,
    niche_categories: list[str],
    niche_confidence: np.ndarray,
    domain_codes: np.ndarray,
    domain_categories: list[str],
    domain_confidence: np.ndarray,
    slab_centres: np.ndarray,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    with h5py.File(output_path, "r+") as handle:
        obs = handle["obs"]
        values = {
            "pred_niche": ("categorical", niche_codes, niche_categories),
            "pred_niche_conf": ("array", niche_confidence, None),
            "pred_domain": ("categorical", domain_codes, domain_categories),
            "pred_domain_conf": ("array", domain_confidence, None),
            "gcsa_z_slab_um": ("array", slab_centres.astype(np.float32), None),
        }
        for key, (kind, data, categories) in values.items():
            if kind == "categorical":
                write_categorical(obs, key, data, categories)
            else:
                write_array(obs, key, data)
        columns = [str(value) for value in obs.attrs["column-order"]]
        for key in values:
            if key not in columns:
                columns.append(key)
        if "column-order" in obs.attrs:
            del obs.attrs["column-order"]
        obs.attrs.create(
            "column-order",
            np.asarray(columns, dtype=h5py.string_dtype(encoding="utf-8")),
        )


def predict_volume(
    input_path,
    model_dir,
    sidecar_output,
    full_output=None,
    summary_output=None,
    spatial_key="spatial_st_corrected",
    z_key="z_um",
    group_key=None,
    slab_thickness=10.0,
    confidence_threshold=0.5,
    delaunay_percentile=99.0,
    read_batch_size=50000,
    max_slabs=None,
) -> None:
    """Annotate a reconstructed count matrix with the trusted PCA/MLP transfer model."""
    if slab_thickness <= 0:
        raise ValueError("--slab-thickness must be positive")
    sidecar_output.parent.mkdir(parents=True, exist_ok=True)
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
    if full_output is not None:
        full_output.parent.mkdir(parents=True, exist_ok=True)
    pca_data = np.load(model_dir / "pca.npz", allow_pickle=True)
    pca_mean = pca_data["mean"].astype(np.float32)
    pca_components = pca_data["components"].astype(np.float32)
    model_genes = [str(value) for value in pca_data["genes"]]
    mean_projection = pca_mean @ pca_components.T
    niche_model = joblib.load(model_dir / "mlp_niche.joblib")
    domain_model = joblib.load(model_dir / "mlp_domain.joblib")
    reference_niches = json.loads((model_dir / "niche_categories.json").read_text())
    with h5py.File(input_path, "r") as handle:
        matrix_group = handle["X"]
        matrix_shape = tuple(int(value) for value in matrix_group.attrs["shape"])
        (n_obs, n_vars) = matrix_shape
        input_genes = [str(value) for value in _decode(handle["var"]["gene"][:])]
        gene_to_column = {gene: index for (index, gene) in enumerate(input_genes)}
        missing = [gene for gene in model_genes if gene not in gene_to_column]
        if missing:
            raise ValueError(
                f"Input is missing {len(missing)} model genes: {missing[:20]}"
            )
        model_columns = np.asarray([gene_to_column[gene] for gene in model_genes])
        coordinates = np.asarray(handle["obsm"][spatial_key][:], dtype=np.float64)
        z_um = np.asarray(read_dataframe_column(handle["obs"], z_key), dtype=np.float64)
        obs_group = handle["obs"]
        obs_index_key = obs_group.attrs.get("_index", "_index")
        if isinstance(obs_index_key, bytes):
            obs_index_key = obs_index_key.decode("utf-8")
        obs_names = [str(value) for value in _decode(obs_group[str(obs_index_key)][:])]
        indptr = matrix_group["indptr"][:]
        slab_ids = np.rint(z_um / slab_thickness).astype(np.int32)
        slab_centres = slab_ids.astype(np.float32) * slab_thickness
        if group_key is None:
            group_values = np.full(n_obs, "all", dtype=object)
        else:
            group_values = read_dataframe_column(handle["obs"], group_key)
            if any(value is None for value in group_values):
                raise ValueError(f"obs[{group_key!r}] contains missing values")
            group_values = np.asarray(
                [str(value) for value in group_values], dtype=object
            )
        (group_names, group_codes) = np.unique(group_values, return_inverse=True)
        selected_slabs = sorted(
            {
                (int(group_code), int(slab_id))
                for (group_code, slab_id) in zip(group_codes, slab_ids)
            }
        )
        if max_slabs is not None:
            selected_slabs = selected_slabs[:max_slabs]
        selected_mask = np.zeros(n_obs, dtype=bool)
        for group_code, slab_id in selected_slabs:
            selected_mask |= (group_codes == group_code) & (slab_ids == slab_id)
        print(
            f"Input: {n_obs:,} cells x {n_vars:,} genes; model overlap {len(model_genes):,}/{len(model_genes):,}; processing {len(group_names)} groups and {len(selected_slabs)} group-by-depth slabs",
            flush=True,
        )
        with tempfile.TemporaryDirectory(prefix="gcsa_3d_", dir="/tmp") as temporary:
            pca_path = Path(temporary) / "pca_scores.dat"
            pca_scores = np.memmap(
                pca_path,
                dtype=np.float32,
                mode="w+",
                shape=(n_obs, pca_components.shape[0]),
            )
            pca_scores[:] = np.nan
            for start in range(0, n_obs, read_batch_size):
                end = min(start + read_batch_size, n_obs)
                local_mask = selected_mask[start:end]
                if not local_mask.any():
                    continue
                counts = read_csr_rows(matrix_group, indptr, start, end, n_vars)
                counts = counts[local_mask][:, model_columns]
                normalized = log_normalize(counts)
                projected = np.asarray(normalized @ pca_components.T, dtype=np.float32)
                projected -= mean_projection[None, :]
                target_rows = np.flatnonzero(local_mask) + start
                pca_scores[target_rows] = projected
                print(f"PCA: {end:,}/{n_obs:,}", flush=True)
            pca_scores.flush()
            niche_codes = np.full(n_obs, len(niche_model["classes"]), dtype=np.int16)
            niche_confidence = np.zeros(n_obs, dtype=np.float32)
            domain_codes = np.full(n_obs, len(domain_model["classes"]), dtype=np.int16)
            domain_confidence = np.zeros(n_obs, dtype=np.float32)
            niche_categories = [str(value) for value in niche_model["classes"]] + [
                "Unknown"
            ]
            domain_categories = [str(value) for value in domain_model["classes"]] + [
                "Unknown"
            ]
            summary_rows: list[dict[str, object]] = []
            for slab_number, (group_code, slab_id) in enumerate(
                selected_slabs, start=1
            ):
                group_name = str(group_names[group_code])
                indices = np.flatnonzero(
                    (group_codes == group_code) & (slab_ids == slab_id)
                )
                slab_coords = coordinates[indices, :2]
                adjacency = build_delaunay_graph(slab_coords, delaunay_percentile)
                niche_embedding = aggregate_neighbors(
                    adjacency, pca_scores[indices], n_layers=3
                )
                (n_codes, n_conf, predicted_niche_categories) = prediction_codes(
                    niche_model, niche_embedding, confidence_threshold
                )
                niche_one_hot = one_hot_from_codes(
                    n_codes, predicted_niche_categories, reference_niches
                )
                domain_embedding = aggregate_neighbors(
                    adjacency, niche_one_hot, n_layers=7
                )
                (d_codes, d_conf, predicted_domain_categories) = prediction_codes(
                    domain_model, domain_embedding, confidence_threshold
                )
                if predicted_domain_categories != domain_categories:
                    raise RuntimeError("Domain category ordering changed between slabs")
                niche_codes[indices] = n_codes
                niche_confidence[indices] = n_conf
                domain_codes[indices] = d_codes
                domain_confidence[indices] = d_conf
                summary_rows.append(
                    {
                        "group_key": group_key or "all",
                        "group": group_name,
                        "slab_um": float(slab_id * slab_thickness),
                        "n_cells": int(indices.size),
                        "n_edges": int(adjacency.nnz // 2),
                        "niche_unknown_fraction": float(
                            np.mean(n_codes == len(niche_categories) - 1)
                        ),
                        "domain_unknown_fraction": float(
                            np.mean(d_codes == len(domain_categories) - 1)
                        ),
                        "mean_niche_confidence": float(n_conf.mean()),
                        "mean_domain_confidence": float(d_conf.mean()),
                    }
                )
                print(
                    f"Slab {slab_number}/{len(selected_slabs)} group={group_name} z={slab_id * slab_thickness:g} um: {indices.size:,} cells",
                    flush=True,
                )
    sidecar_obs = pd.DataFrame(index=pd.Index(obs_names, name="cell_id"))
    sidecar_obs["pred_niche"] = pd.Categorical.from_codes(niche_codes, niche_categories)
    sidecar_obs["pred_niche_conf"] = niche_confidence
    sidecar_obs["pred_domain"] = pd.Categorical.from_codes(
        domain_codes, domain_categories
    )
    sidecar_obs["pred_domain_conf"] = domain_confidence
    sidecar_obs["gcsa_z_slab_um"] = slab_centres
    if group_key is not None:
        sidecar_obs[group_key] = pd.Categorical(group_values)
    sidecar = AnnData(obs=sidecar_obs)
    sidecar.uns["gc_spatial_annotator"] = {
        "model_dir": str(model_dir.resolve()),
        "spatial_key": spatial_key,
        "z_key": z_key,
        "group_key": group_key or "",
        "slab_thickness_um": float(slab_thickness),
        "confidence_threshold": float(confidence_threshold),
        "delaunay_percentile": float(delaunay_percentile),
        "graph_strategy": "2D Delaunay graph within rounded z-depth slabs",
    }
    sidecar.write_h5ad(sidecar_output, compression="gzip")
    if summary_output is not None:
        pd.DataFrame(summary_rows).to_csv(summary_output, index=False)
    if full_output is not None:
        append_predictions_to_h5ad(
            input_path,
            full_output,
            niche_codes,
            niche_categories,
            niche_confidence,
            domain_codes,
            domain_categories,
            domain_confidence,
            slab_centres,
        )
    print(f"Saved sidecar: {sidecar_output}", flush=True)
    if full_output is not None:
        print(f"Saved annotated h5ad: {full_output}", flush=True)


def decode_scalar(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def obs_index_dataset(obs: h5py.Group) -> h5py.Dataset:
    key = decode_scalar(obs.attrs.get("_index", "_index"))
    return obs[key]


def assert_matching_obs(
    base_obs: h5py.Group, annotation_obs: h5py.Group, chunk_size: int
) -> None:
    base_index = obs_index_dataset(base_obs)
    annotation_index = obs_index_dataset(annotation_obs)
    if base_index.shape != annotation_index.shape:
        raise ValueError(
            f"Observation counts differ: {base_index.shape} != {annotation_index.shape}"
        )
    for start in range(0, base_index.shape[0], chunk_size):
        end = min(start + chunk_size, base_index.shape[0])
        if not np.array_equal(base_index[start:end], annotation_index[start:end]):
            raise ValueError(
                f"Observation identifiers/order differ at rows {start}:{end}"
            )


def write_compact(
    base_path, annotations_path, output_path, columns, comparison_chunk_size=100000
) -> None:
    """Copy selected hard annotations after exact shape and observation-order checks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(output_path.name + ".tmp")
    if temporary_output.exists():
        temporary_output.unlink()
    with h5py.File(base_path, "r") as base, h5py.File(annotations_path, "r") as source:
        base_shape = tuple(int(value) for value in base["X"].attrs["shape"])
        source_shape = tuple(int(value) for value in source["X"].attrs["shape"])
        if base_shape != source_shape:
            raise ValueError(f"AnnData shapes differ: {base_shape} != {source_shape}")
        assert_matching_obs(base["obs"], source["obs"], comparison_chunk_size)
        missing = [key for key in columns if key not in source["obs"]]
        if missing:
            raise KeyError(f"Annotation source is missing obs columns: {missing}")
    shutil.copy2(base_path, temporary_output)
    try:
        with (
            h5py.File(annotations_path, "r") as source,
            h5py.File(temporary_output, "r+") as destination,
        ):
            source_obs = source["obs"]
            destination_obs = destination["obs"]
            for key in columns:
                if key in destination_obs:
                    del destination_obs[key]
                source_obs.copy(key, destination_obs, name=key)
            column_order = [
                decode_scalar(value) for value in destination_obs.attrs["column-order"]
            ]
            for key in columns:
                if key not in column_order:
                    column_order.append(key)
            del destination_obs.attrs["column-order"]
            destination_obs.attrs.create(
                "column-order",
                np.asarray(column_order, dtype=h5py.string_dtype(encoding="utf-8")),
            )
        os.replace(temporary_output, output_path)
    except Exception:
        if temporary_output.exists():
            temporary_output.unlink()
        raise
    print(f"Wrote compact AnnData: {output_path}")
    print(f"Copied obs columns: {', '.join(columns)}")

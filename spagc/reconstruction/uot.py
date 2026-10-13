"""Hybrid costs and dense or candidate-restricted unbalanced transport."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ot
import scipy.sparse as sp
from scipy.spatial import cKDTree

_EPS = 1e-12


def _as_numpy(value):
    """Convert CPU tensors/array-likes without changing the public API."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _validate_weights(
    alpha_spatial, spatial_weight, gene_weight, histology_weight, class_weight
):
    for value in (alpha_spatial, histology_weight, class_weight):
        if not np.isfinite(value) or value < 0:
            raise ValueError("Invalid UOT weight")
    if alpha_spatial > 1:
        raise ValueError("alpha_spatial must be in [0,1]")
    for value in (spatial_weight, gene_weight):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError("Invalid UOT weight")


def _normalize_cost(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=1.0, posinf=1.0, neginf=0.0)
    maximum = values.max(initial=0.0)
    if maximum > 0:
        values = values / (maximum + 1e-09)
    return values


def _cosine_edge_cost(a, b, row, col, chunk_size=4096):
    """Cosine distance for selected pairs without materialising an N-by-M matrix."""
    a = _as_numpy(a)
    b = _as_numpy(b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Pairwise cosine inputs must be [N,D] and [M,D] with equal D")
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    result = np.empty(len(row), dtype=np.float64)
    for start in range(0, len(row), chunk_size):
        stop = min(start + chunk_size, len(row))
        aa = a[row[start:stop]]
        bb = b[col[start:stop]]
        numerator = np.einsum("ij,ij->i", aa, bb, optimize=True)
        denominator = norm_a[row[start:stop]] * norm_b[col[start:stop]]
        similarity = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=denominator > _EPS,
        )
        result[start:stop] = 1.0 - similarity
    return np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=0.0)


def _dot_edge_cost(a, b, row, col, chunk_size=16384):
    """Return 1 - dot product for selected pairs."""
    a = _as_numpy(a)
    b = _as_numpy(b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Pairwise dot inputs must be [N,D] and [M,D] with equal D")
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    result = np.empty(len(row), dtype=np.float64)
    for start in range(0, len(row), chunk_size):
        stop = min(start + chunk_size, len(row))
        result[start:stop] = 1.0 - np.einsum(
            "ij,ij->i", a[row[start:stop]], b[col[start:stop]], optimize=True
        )
    return np.clip(np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _query_candidates(source, target, top_k, radius):
    tree = cKDTree(target)
    k = min(int(top_k), len(target))
    upper = np.inf if radius is None else float(radius)
    (distances, indices) = tree.query(source, k=k, distance_upper_bound=upper)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    valid = np.isfinite(distances) & (indices < len(target))
    source_indices = np.repeat(np.arange(len(source), dtype=np.int64), k)
    rows = source_indices[valid.reshape(-1)]
    cols = indices.reshape(-1)[valid.reshape(-1)].astype(np.int64)
    missing = np.flatnonzero(~valid.any(axis=1))
    if len(missing):
        (_, nearest) = tree.query(source[missing], k=1)
        rows = np.concatenate([rows, missing.astype(np.int64)])
        cols = np.concatenate([cols, np.asarray(nearest, dtype=np.int64).reshape(-1)])
    return (rows, cols)


def build_topk_candidates(x0, x1, top_k, *, bidirectional=True, candidate_radius=None):
    """Build a sparse candidate graph from registered XY coordinates.

    The forward graph contains up to ``top_k`` target cells per source cell.
    With ``bidirectional=True`` the reverse kNN graph is added as well, which
    ensures that every target cell has support even when cell densities differ.
    ``candidate_radius`` is expressed in the same coordinate units as x0/x1;
    nearest-neighbour fallback keeps every marginal supported when a radius is
    too restrictive.
    """
    x0 = _as_numpy(x0).astype(np.float64, copy=False)
    x1 = _as_numpy(x1).astype(np.float64, copy=False)
    if x0.ndim != 2 or x1.ndim != 2 or x0.shape[1] != 2 or (x1.shape[1] != 2):
        raise ValueError("x0 and x1 must have shape [N,2] and [M,2]")
    if not len(x0) or not len(x1):
        raise ValueError("Cannot build candidates for an empty slice")
    if not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if candidate_radius is not None and (
        not np.isfinite(candidate_radius) or candidate_radius < 0
    ):
        raise ValueError("candidate_radius must be finite and nonnegative")
    (forward_rows, forward_cols) = _query_candidates(x0, x1, top_k, candidate_radius)
    all_rows = [forward_rows]
    all_cols = [forward_cols]
    if bidirectional:
        (reverse_rows, reverse_cols) = _query_candidates(
            x1, x0, top_k, candidate_radius
        )
        all_rows.append(reverse_cols)
        all_cols.append(reverse_rows)
    keys = np.concatenate(all_rows).astype(np.int64) * len(x1) + np.concatenate(
        all_cols
    )
    keys = np.unique(keys)
    rows = keys // len(x1)
    cols = keys % len(x1)
    distances = np.linalg.norm(x0[rows] - x1[cols], axis=1)
    return (rows.astype(np.int64), cols.astype(np.int64), distances.astype(np.float64))


def compute_cost_matrix(
    x0,
    g0,
    c0,
    x1,
    g1,
    c1,
    alpha_spatial=0.5,
    *,
    spatial_weight=None,
    gene_weight=None,
    histology_weight=0.0,
    class_weight=10.0,
    h0=None,
    h1=None,
    h0_valid=None,
    h1_valid=None,
    use_celltype=True,
    max_uot_entries=None,
):
    """Compute the original dense hybrid UOT cost matrix."""
    eps = 1e-09
    x0 = _as_numpy(x0)
    x1 = _as_numpy(x1)
    _validate_weights(
        alpha_spatial, spatial_weight, gene_weight, histology_weight, class_weight
    )
    if max_uot_entries is not None and len(x0) * len(x1) > max_uot_entries:
        raise MemoryError(
            "Dense UOT exceeds max_uot_entries; subset/aggregate anchors explicitly. This solver does not silently change to sparse or minibatch OT."
        )
    cost_spatial = ot.dist(x0, x1, metric="euclidean")
    s_max = cost_spatial.max()
    cost_spatial = cost_spatial / (s_max + eps) if s_max > 0 else cost_spatial
    cost_gene = ot.dist(_as_numpy(g0), _as_numpy(g1), metric="cosine")
    cost_gene = np.nan_to_num(cost_gene, nan=1.0, posinf=1.0, neginf=0.0)
    g_max = cost_gene.max()
    cost_gene = cost_gene / (g_max + eps) if g_max > 0 else cost_gene
    c0_np = _as_numpy(c0)
    c1_np = _as_numpy(c1)
    cost_class = np.clip(1.0 - np.dot(c0_np, c1_np.T), 0, 1)
    ws = alpha_spatial if spatial_weight is None else spatial_weight
    wg = 1 - alpha_spatial if gene_weight is None else gene_weight
    C = ws * cost_spatial + wg * cost_gene
    if use_celltype:
        C = C + class_weight * cost_class
    if histology_weight:
        if h0 is None or h1 is None:
            raise ValueError("Nonzero histology weight requires cached h0 and h1")
        a = _as_numpy(h0)
        b = _as_numpy(h1)
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.shape[1] != b.shape[1]
            or (len(a) != len(x0))
            or (len(b) != len(x1))
        ):
            raise ValueError("Expected histology arrays [N,D], [N1,D]")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Invalid morphology features")
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        if h0_valid is None and (na <= eps).any():
            raise ValueError("Zero morphology feature: invalid tissue query")
        if h1_valid is None and (nb <= eps).any():
            raise ValueError("Zero morphology feature: invalid tissue query")
        valid0 = (
            np.ones(len(a), dtype=bool)
            if h0_valid is None
            else np.asarray(h0_valid, dtype=bool).copy()
        )
        valid1 = (
            np.ones(len(b), dtype=bool)
            if h1_valid is None
            else np.asarray(h1_valid, dtype=bool).copy()
        )
        if len(valid0) != len(a) or len(valid1) != len(b):
            raise ValueError("Morphology validity masks must match feature rows")
        valid0 &= na > eps
        valid1 &= nb > eps
        safe_na = np.maximum(na, eps)[:, None]
        safe_nb = np.maximum(nb, eps)[:, None]
        dh = np.clip(1 - a / safe_na @ (b / safe_nb).T, 0, 2)
        dh[~(valid0[:, None] & valid1[None, :])] = 0
        dh_max = dh.max()
        if dh_max > 0:
            C = C + histology_weight * dh / (dh_max + eps)
    del cost_spatial, cost_gene, cost_class
    gc.collect()
    return C


def compute_sparse_cost(
    x0,
    g0,
    c0,
    x1,
    g1,
    c1,
    row,
    col,
    alpha_spatial=0.5,
    *,
    spatial_weight=None,
    gene_weight=None,
    histology_weight=0.0,
    class_weight=10.0,
    h0=None,
    h1=None,
    h0_valid=None,
    h1_valid=None,
    use_celltype=True,
):
    """Compute hybrid cost only on the supplied sparse candidate edges."""
    eps = 1e-09
    x0 = _as_numpy(x0)
    x1 = _as_numpy(x1)
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    if len(row) != len(col):
        raise ValueError("row and col candidate indices must have equal length")
    if len(row) == 0:
        raise ValueError("At least one candidate edge is required")
    if row.min() < 0 or row.max() >= len(x0) or col.min() < 0 or (col.max() >= len(x1)):
        raise IndexError("Candidate edge index is outside slice bounds")
    _validate_weights(
        alpha_spatial, spatial_weight, gene_weight, histology_weight, class_weight
    )
    cost_spatial = _normalize_cost(np.linalg.norm(x0[row] - x1[col], axis=1))
    cost_gene = _normalize_cost(_cosine_edge_cost(g0, g1, row, col))
    ws = alpha_spatial if spatial_weight is None else spatial_weight
    wg = 1 - alpha_spatial if gene_weight is None else gene_weight
    result = ws * cost_spatial + wg * cost_gene
    if use_celltype:
        result = result + class_weight * _dot_edge_cost(c0, c1, row, col)
    if histology_weight:
        if h0 is None or h1 is None:
            raise ValueError("Nonzero histology weight requires cached h0 and h1")
        a = _as_numpy(h0)
        b = _as_numpy(h1)
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.shape[1] != b.shape[1]
            or (len(a) != len(x0))
            or (len(b) != len(x1))
        ):
            raise ValueError("Expected histology arrays [N,D], [N1,D]")
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        if h0_valid is None and (na <= eps).any():
            raise ValueError("Zero morphology feature: invalid tissue query")
        if h1_valid is None and (nb <= eps).any():
            raise ValueError("Zero morphology feature: invalid tissue query")
        valid0 = (
            np.ones(len(a), dtype=bool)
            if h0_valid is None
            else np.asarray(h0_valid, dtype=bool).copy()
        )
        valid1 = (
            np.ones(len(b), dtype=bool)
            if h1_valid is None
            else np.asarray(h1_valid, dtype=bool).copy()
        )
        if len(valid0) != len(a) or len(valid1) != len(b):
            raise ValueError("Morphology validity masks must match feature rows")
        valid0 &= na > eps
        valid1 &= nb > eps
        edge_valid = valid0[row] & valid1[col]
        morphology = np.zeros(len(row), dtype=np.float64)
        if edge_valid.any():
            valid_edges = np.flatnonzero(edge_valid)
            morphology[valid_edges] = np.clip(
                _cosine_edge_cost(
                    a, b, row[valid_edges], col[valid_edges], chunk_size=4096
                ),
                0.0,
                2.0,
            )
            morphology_max = morphology[edge_valid].max()
            if morphology_max > 0:
                result = result + histology_weight * morphology / (morphology_max + eps)
    return np.nan_to_num(
        result, nan=np.finfo(np.float64).max, posinf=np.finfo(np.float64).max
    )


@dataclass
class SparseCoupling:
    """Edge-list representation of a UOT coupling."""

    row: np.ndarray
    col: np.ndarray
    mass: np.ndarray
    shape: tuple
    cost: np.ndarray | None = None

    def __post_init__(self):
        self.row = np.asarray(self.row, dtype=np.int64)
        self.col = np.asarray(self.col, dtype=np.int64)
        self.mass = np.asarray(self.mass, dtype=np.float64)
        if not len(self.row) == len(self.col) == len(self.mass):
            raise ValueError("Sparse coupling edge arrays must have equal length")
        if len(self.row) and (
            self.row.min() < 0
            or self.col.min() < 0
            or self.row.max() >= self.shape[0]
            or (self.col.max() >= self.shape[1])
        ):
            raise ValueError("Sparse coupling edge index outside declared shape")

    def __len__(self):
        return len(self.mass)

    @property
    def total_mass(self):
        return float(self.mass.sum())

    def sample(self, n_samples, rng=None):
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        if self.total_mass <= 0 or not np.isfinite(self.total_mass):
            raise ValueError("Cannot sample from an empty sparse coupling")
        rng = np.random.default_rng() if rng is None else rng
        probabilities = self.mass / self.total_mass
        indices = rng.choice(len(self), size=n_samples, replace=True, p=probabilities)
        return (self.row[indices], self.col[indices])

    def to_dense(self, max_entries=None):
        entries = int(self.shape[0]) * int(self.shape[1])
        if max_entries is not None and entries > max_entries:
            raise MemoryError("Refusing to materialize a large sparse coupling")
        dense = np.zeros(self.shape, dtype=np.float64)
        dense[self.row, self.col] = self.mass
        return dense


def _sparse_unbalanced_sinkhorn(
    row, col, cost, shape, a, b, reg, reg_m, num_iter_max=100, stop_thr=0.001
):
    """KL-relaxed Sinkhorn using sparse K-v and K.T-u products."""
    if reg <= 0 or reg_m <= 0:
        raise ValueError("reg and reg_m must be positive")
    reg = max(float(reg), 0.01)
    exponent = reg_m / (reg_m + reg)
    kernel_values = np.exp(-np.clip(np.asarray(cost, dtype=np.float64) / reg, 0, 700))
    kernel_values = kernel_values * a[row] * b[col]
    kernel = sp.csr_matrix((kernel_values, (row, col)), shape=shape)
    u = np.ones(shape[0], dtype=np.float64)
    v = np.ones(shape[1], dtype=np.float64)
    for _ in range(int(num_iter_max)):
        kv = np.asarray(kernel.dot(v)).reshape(-1)
        u_new = np.power(a / np.maximum(kv, _EPS), exponent)
        ktu = np.asarray(kernel.T.dot(u_new)).reshape(-1)
        v_new = np.power(b / np.maximum(ktu, _EPS), exponent)
        if not np.isfinite(u_new).all() or not np.isfinite(v_new).all():
            raise FloatingPointError("Sparse UOT Sinkhorn produced non-finite scalings")
        error = max(
            np.max(
                np.abs(np.log(np.maximum(u_new, _EPS)) - np.log(np.maximum(u, _EPS)))
            ),
            np.max(
                np.abs(np.log(np.maximum(v_new, _EPS)) - np.log(np.maximum(v, _EPS)))
            ),
        )
        (u, v) = (u_new, v_new)
        if error < stop_thr:
            break
    mass = kernel_values * u[row] * v[col]
    if not np.isfinite(mass).all() or mass.sum() <= 0:
        raise FloatingPointError("Sparse UOT coupling has no finite positive mass")
    return mass


def compute_uot_coupling(
    x0,
    g0,
    c0,
    x1,
    g1,
    c1,
    alpha_spatial=0.5,
    uot_reg=0.8,
    uot_tau=0.05,
    *,
    solver="dense",
    top_k=None,
    bidirectional=True,
    candidate_radius=None,
    candidate_x0=None,
    candidate_x1=None,
    **cost_kwargs,
):
    """Compute a dense or spatial top-k sparse unbalanced coupling.

    ``solver='dense'`` preserves the release behavior and returns an N0-by-N1
    NumPy matrix. ``solver='sparse_topk'`` returns :class:`SparseCoupling` and
    only evaluates the candidate graph generated from registered XY positions.
    """
    if solver not in ("dense", "sparse_topk"):
        raise ValueError("solver must be 'dense' or 'sparse_topk'")
    if solver == "dense":
        C = compute_cost_matrix(x0, g0, c0, x1, g1, c1, alpha_spatial, **cost_kwargs)
        (n0, n1) = (len(_as_numpy(x0)), len(_as_numpy(x1)))
        a = np.ones(n0) / n0
        b = np.ones(n1) / n1
        pi = ot.unbalanced.sinkhorn_unbalanced(
            a,
            b,
            C,
            reg=max(uot_reg, 0.01),
            reg_m=uot_tau,
            numItermax=100,
            stopThr=0.001,
            verbose=False,
        )
        del C, a, b
        gc.collect()
        return pi
    if top_k is None:
        raise ValueError("top_k is required for solver='sparse_topk'")
    cost_kwargs.pop("max_uot_entries", None)
    (rows, cols, _) = build_topk_candidates(
        x0 if candidate_x0 is None else candidate_x0,
        x1 if candidate_x1 is None else candidate_x1,
        top_k,
        bidirectional=bidirectional,
        candidate_radius=candidate_radius,
    )
    edge_cost = compute_sparse_cost(
        x0, g0, c0, x1, g1, c1, rows, cols, alpha_spatial, **cost_kwargs
    )
    (n0, n1) = (len(_as_numpy(x0)), len(_as_numpy(x1)))
    a = np.ones(n0, dtype=np.float64) / n0
    b = np.ones(n1, dtype=np.float64) / n1
    mass = _sparse_unbalanced_sinkhorn(
        rows,
        cols,
        edge_cost,
        (n0, n1),
        a,
        b,
        reg=max(uot_reg, 0.01),
        reg_m=uot_tau,
        num_iter_max=100,
        stop_thr=0.001,
    )
    del edge_cost, a, b
    gc.collect()
    return SparseCoupling(rows, cols, mass, (n0, n1))


def save_sparse_coupling(path, coupling, metadata=None):
    """Atomically save an edge-list coupling as a compressed NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    payload = {
        "row": coupling.row,
        "col": coupling.col,
        "mass": coupling.mass,
        "shape": np.asarray(coupling.shape, dtype=np.int64),
        "metadata": np.asarray(json.dumps(metadata or {}, sort_keys=True, default=str)),
    }
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def load_sparse_coupling(path):
    """Load a coupling cache and its JSON metadata."""
    with np.load(path, allow_pickle=False) as data:
        coupling = SparseCoupling(
            row=data["row"],
            col=data["col"],
            mass=data["mass"],
            shape=tuple(int(x) for x in data["shape"]),
        )
        metadata = json.loads(str(data["metadata"].item()))
    return (coupling, metadata)


def _weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(values) == 0 or weights.sum() <= 0:
        return float("nan")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    target = float(quantile) * cumulative[-1]
    index = np.searchsorted(cumulative, target, side="left")
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def summarize_sparse_coupling(coupling, source_xy, target_xy):
    """Return JSON-safe support, mass and distance statistics for QC."""
    source_xy = np.asarray(source_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    if source_xy.shape != (coupling.shape[0], 2) or target_xy.shape != (
        coupling.shape[1],
        2,
    ):
        raise ValueError("XY arrays do not match coupling shape")
    edge_distances = np.linalg.norm(
        source_xy[coupling.row] - target_xy[coupling.col], axis=1
    )
    total_mass = coupling.total_mass
    row_mass = np.bincount(
        coupling.row, weights=coupling.mass, minlength=coupling.shape[0]
    )
    col_mass = np.bincount(
        coupling.col, weights=coupling.mass, minlength=coupling.shape[1]
    )
    dense_pair_count = int(coupling.shape[0]) * int(coupling.shape[1])
    return {
        "source_count": int(coupling.shape[0]),
        "target_count": int(coupling.shape[1]),
        "edge_count": int(len(coupling)),
        "dense_pair_count": dense_pair_count,
        "edge_fraction_of_dense": float(len(coupling) / dense_pair_count),
        "dense_to_sparse_reduction": float(dense_pair_count / max(len(coupling), 1)),
        "total_mass": float(total_mass),
        "source_support_fraction": float(np.mean(row_mass > 0)),
        "target_support_fraction": float(np.mean(col_mass > 0)),
        "positive_mass_edge_fraction": float(np.mean(coupling.mass > 0)),
        "weighted_distance_mean_um": float(
            np.dot(coupling.mass, edge_distances) / total_mass
            if total_mass > 0
            else float("nan")
        ),
        "weighted_distance_p50_um": _weighted_quantile(
            edge_distances, coupling.mass, 0.5
        ),
        "weighted_distance_p95_um": _weighted_quantile(
            edge_distances, coupling.mass, 0.95
        ),
        "unweighted_distance_p95_um": float(np.quantile(edge_distances, 0.95)),
    }

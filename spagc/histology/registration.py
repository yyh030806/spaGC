"""Matched-section GALA registration in native H&E coordinates."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from spagc._io import require_files
from spagc.histology.slides import open_slide, read_region_resized_rgb


def _backend_call(name, *args, **kwargs):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        if name == "gala":
            from GALA.gala import gala as function
        else:
            from GALA import utils

            function = getattr(utils, name)
        return function(*args, **kwargs)
    finally:
        torch.set_default_dtype(previous)


def normalize(*args, **kwargs):
    return _backend_call("normalize", *args, **kwargs)


def rasterize(*args, **kwargs):
    return _backend_call("rasterize", *args, **kwargs)


def _matrix_is_usable(M):
    if M is None:
        return False
    try:
        return M.shape[0] > 0 and M.shape[1] > 0
    except (AttributeError, IndexError, TypeError):
        return False


def get_expression_matrix(adata, layer=None, verbose=True):
    """Find a usable expression matrix by priority and raise ValueError when none is available."""
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(
                f"Count layer {layer!r} is missing; available layers: {list(adata.layers)}"
            )
        M = adata.layers[layer]
        if not _matrix_is_usable(M):
            raise ValueError(f"Expression layer {layer!r} is empty.")
        if verbose:
            print(f"Using adata.layers[{layer!r}]")
        return (M, f"layers['{layer}']")
    if _matrix_is_usable(adata.X):
        if verbose:
            print("Using adata.X")
        return (adata.X, "X")
    for cand in ("counts", "raw_counts", "raw", "spliced", "matrix"):
        if cand in adata.layers and _matrix_is_usable(adata.layers[cand]):
            if verbose:
                print(f"Empty adata.X; using adata.layers[{cand!r}]")
            return (adata.layers[cand], f"layers['{cand}']")
    for k, M in adata.layers.items():
        if _matrix_is_usable(M):
            if verbose:
                print(f"Empty adata.X; using adata.layers[{k!r}]")
            return (M, f"layers['{k}']")
    if adata.raw is not None and _matrix_is_usable(adata.raw.X):
        if verbose:
            print("Using adata.raw.X")
        return (adata.raw.X, "raw.X")
    raise ValueError("no expression matrix")


def to_signal_2d(M):
    """Sum matrix rows into a two-dimensional `(n_cells, 1)` signal for GALA rasterization."""
    s = M.sum(axis=1)
    s = np.asarray(s).reshape(-1, 1).astype(np.float64)
    return s


def as_col(arr):
    """Convert a one-dimensional array or Series to an `(n, 1)` NumPy array."""
    a = np.asarray(arr).astype(np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def preprocess_he(he_path, scale=1.0):
    if not os.path.exists(he_path):
        raise FileNotFoundError(he_path)
    if not 0 < scale <= 1:
        raise ValueError("he scale must lie in (0, 1]")
    with open_slide(he_path) as slide:
        (width, height) = slide.dimensions
        target_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        img = read_region_resized_rgb(slide, (0, 0), (width, height), target_size)
    V = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(V, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(V, cv2.CV_64F, 0, 1, ksize=3)
    image_signal = normalize(cv2.magnitude(sobelx, sobely))
    image_signal = image_signal[None, ...]
    del V, sobelx, sobely
    YI = np.arange(image_signal.shape[1]) * 1.0
    XI = np.arange(image_signal.shape[2]) * 1.0
    return (img, image_signal, [YI, XI], scale)


def preprocess_st(h5ad_path, dx=30, counts_layer=None, signal_obs=None):
    """Prepare spatial-transcriptomics coordinates and signal values for GALA alignment."""
    print(f"Reading h5ad: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  shape = {adata.shape}, obsm keys = {list(adata.obsm.keys())}")
    if "spatial" not in adata.obsm:
        raise KeyError(f"Missing obsm['spatial']; available keys: {list(adata.obsm)}")
    xj = adata.obsm["spatial"][:, 0].astype(np.float64)
    yj = adata.obsm["spatial"][:, 1].astype(np.float64)
    n_cells = adata.shape[0]
    if signal_obs is not None:
        if signal_obs not in adata.obs.columns:
            raise KeyError(
                f"Signal column {signal_obs!r} is missing; available columns: {list(adata.obs)}"
            )
        col = adata.obs[signal_obs]
        if not pd.api.types.is_numeric_dtype(col):
            raise TypeError(f"obs[{signal_obs!r}] must be numeric; got {col.dtype}.")
        g = as_col(col.to_numpy())
        gj = np.log1p(g)
        signal_src = f"obs['{signal_obs}']"
        print(f"  signal source: {signal_src}")
    else:
        try:
            (M, src) = get_expression_matrix(adata, layer=counts_layer, verbose=True)
            g = to_signal_2d(M)
            adata_proc = adata.copy()
            adata_proc.X = M.copy() if hasattr(M, "copy") else np.array(M)
            sc.pp.normalize_total(adata_proc, inplace=True)
            sc.pp.log1p(adata_proc)
            gj = to_signal_2d(adata_proc.X)
            signal_src = f"adata.{src} (total counts)"
            print(f"  signal source: {signal_src}")
        except (ValueError, KeyError) as e:
            if counts_layer is not None:
                raise
            print(
                f"No expression signal ({e}); registering cell density on a {dx} µm grid."
            )
            g = np.ones((n_cells, 1), dtype=np.float64)
            gj = np.ones((n_cells, 1), dtype=np.float64)
            signal_src = "cell density (uniform g=1)"
    (XJ, YJ, spatial_signal) = rasterize(xj, yj, g=gj, dx=dx)
    return (adata, xj, yj, g, spatial_signal, [YJ, XJ], signal_src)


@dataclass
class PreparedRegistration:
    """Registration inputs with explicit native-pixel and physical-ST conventions."""

    cells: ad.AnnData
    preview: np.ndarray
    image_signal: np.ndarray
    image_axes: list
    st_signal: np.ndarray
    st_axes: list
    x: np.ndarray
    y: np.ndarray
    signal: np.ndarray
    image_scale: float
    signal_source: str


def prepare_registration(
    image_path: Path, cells_path: Path, *, image_scale: float, raster_spacing: float
) -> PreparedRegistration:
    """Read inputs and rasterize them without running the expensive optimizer."""
    require_files(image_path, cells_path)
    previous_dtype = torch.get_default_dtype()
    (preview, image_signal, image_axes, scale) = preprocess_he(
        str(image_path), image_scale
    )
    (cells, x, y, signal, st_signal, st_axes, source) = preprocess_st(
        str(cells_path), dx=raster_spacing
    )
    if signal.shape != (cells.n_obs, 1) or not np.isfinite(signal).all():
        raise ValueError("Registration signal must be a finite N x 1 column.")
    if torch.get_default_dtype() != previous_dtype:
        raise RuntimeError("Registration preprocessing changed the global torch dtype.")
    return PreparedRegistration(
        cells,
        preview,
        image_signal,
        image_axes,
        st_signal,
        st_axes,
        x,
        y,
        signal,
        scale,
        source,
    )


def to_cpu(*args, **kwargs):
    return _backend_call("to_cpu", *args, **kwargs)


def gala(*args, **kwargs):
    return _backend_call("gala", *args, **kwargs)


def run_gala(
    image_axes,
    image_signal,
    spatial_axes,
    spatial_signal,
    angle_range=None,
    num_generations=1000,
    num_iterations=2000,
    num_repeats=2,
    random_seed=42,
):
    angle_dict = angle_range if angle_range is not None else {}
    gene_space = [
        angle_dict,
        {"low": 3.0, "high": 7.0},
        {"low": -7.0, "high": -3.0},
        {"low": -6000, "high": 6000},
        {"low": -1000, "high": 1000},
    ]
    params = dict(
        num_generations=num_generations,
        num_iterations=num_iterations,
        num_repeats=num_repeats,
        epV=10.0,
        a=20.0,
        sigmaR=50000.0,
        sigmaM=image_signal.std(),
        sigmaB=spatial_signal.std(),
        random_seed=random_seed,
    )
    t0 = time.time()
    resu = gala(
        image_axes,
        image_signal,
        spatial_axes,
        spatial_signal,
        gene_space=gene_space,
        **params,
    )
    print(f"Alignment runtime: {time.time() - t0:.2f} s")
    if torch.cuda.is_available():
        resu = to_cpu(resu)
    return resu


def optimize_registration(
    prepared: PreparedRegistration,
    *,
    generations: int,
    iterations: int,
    repeats: int,
    seed: int,
) -> dict:
    """Call the preserved GALA solver only after preprocessing is inspected."""
    if min(generations, iterations, repeats) <= 0:
        raise ValueError("Registration budgets must be positive.")
    return run_gala(
        prepared.image_axes,
        prepared.image_signal,
        prepared.st_axes,
        prepared.st_signal,
        num_generations=generations,
        num_iterations=iterations,
        num_repeats=repeats,
        random_seed=seed,
    )


def transform_points_target_to_source(*args, **kwargs):
    return _backend_call("transform_points_target_to_source", *args, **kwargs)


def project_st_to_he(resu, xj, yj, device="cpu"):
    (v, xv, A) = (resu["v"], resu["xv"], resu["A"])
    xIj = torch.tensor(np.stack([yj, xj], axis=1), device=device, dtype=torch.float64)
    tpointsI = transform_points_target_to_source(xv, v, A, xIj)
    x_aligned = tpointsI[:, 1].detach().cpu().numpy()
    y_aligned = tpointsI[:, 0].detach().cpu().numpy()
    return (x_aligned, y_aligned)


def plot_alignment(
    img_small, xj, yj, g, x_aligned_small, y_aligned_small, he_scale, save_path=None
):
    g_c = np.asarray(g).reshape(-1)
    (fig, axs) = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].scatter(xj, yj, s=1, c=g_c, cmap="viridis", marker=".", linewidths=0)
    axs[0].set_title("Transcriptomic profiles")
    axs[0].set_aspect("equal")
    axs[0].axis("off")
    axs[1].imshow(img_small)
    axs[1].invert_yaxis()
    axs[1].set_title(f"H&E (scale={he_scale})")
    axs[1].axis("off")
    axs[2].imshow(img_small, zorder=1)
    axs[2].scatter(
        xj, yj, s=1, c=g_c, cmap="viridis", marker=".", linewidths=0, zorder=0
    )
    axs[2].set_title("ST vs H&E (before)")
    axs[2].invert_yaxis()
    axs[3].imshow(img_small)
    axs[3].scatter(
        x_aligned_small,
        y_aligned_small,
        s=1,
        c=g_c,
        cmap="viridis",
        marker=".",
        linewidths=0,
    )
    axs[3].set_title("Aligned by GALA")
    axs[3].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.close(fig)


def export_registration(
    prepared: PreparedRegistration,
    transform: dict,
    output_path: Path,
    *,
    image_path: Path,
    cells_path: Path,
    raster_spacing: float,
    seed: int,
) -> ad.AnnData:
    """Project to native level-zero pixels, save an overlay and reject overwrites."""
    if output_path.exists() or output_path.with_suffix(".png").exists():
        raise FileExistsError(f"Alignment output already exists: {output_path}")
    (x, y) = project_st_to_he(transform, prepared.x, prepared.y)
    native_xy = np.stack([x, y], axis=1) / prepared.image_scale
    if native_xy.shape != (prepared.cells.n_obs, 2) or not np.isfinite(native_xy).all():
        raise ValueError("Alignment produced invalid native-pixel coordinates.")
    aligned = prepared.cells.copy()
    aligned.obsm["spatial_aligned"] = native_xy.astype(np.float64)
    aligned.uns["gala_alignment"] = {
        "he_image_path": str(image_path.resolve()),
        "source_h5ad": str(cells_path.resolve()),
        "dx": float(raster_spacing),
        "he_scale_used": prepared.image_scale,
        "he_image_shape_small": np.array(prepared.preview.shape[:2]),
        "random_seed": int(seed),
        "counts_layer": "auto",
        "signal_source": prepared.signal_source,
        "coord_space": "original H&E pixel coordinates",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.write_h5ad(output_path)
    plot_alignment(
        prepared.preview,
        prepared.x,
        prepared.y,
        prepared.signal,
        x,
        y,
        he_scale=prepared.image_scale,
        save_path=str(output_path.with_suffix(".png")),
    )
    return aligned

"""Tissue-supported niche propagation and domain-label conversion."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from spagc.histology.slides import open_slide

UNLABELED = ""


def make_tissue_mask(
    HE_rgb,
    min_od=0.12,
    min_saturation=0.045,
    max_gray=0.96,
    weak_od_factor=0.6,
    debug=False,
):
    img = np.asarray(HE_rgb)
    rgb_raw = img[..., :3]
    if np.issubdtype(rgb_raw.dtype, np.integer):
        denom = float(np.iinfo(rgb_raw.dtype).max)
        rgb = rgb_raw.astype(np.float32) / denom
    else:
        rgb = rgb_raw.astype(np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    delta = maxc - minc
    saturation = delta / (maxc + 1e-06)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    rgb_safe = np.clip(rgb, 1.0 / 255.0, 1.0)
    od = -np.log(rgb_safe)
    od_sum = od[..., 0] + od[..., 1] + od[..., 2]
    stained = (od_sum > min_od) & (saturation > min_saturation) & (gray < max_gray)
    pink_like = (
        (od_sum > min_od * weak_od_factor)
        & (gray < max_gray)
        & (r > g + 0.01)
        & (b > g - 0.1)
        & (saturation > min_saturation * 0.5)
    )
    purple_like = (
        (od_sum > min_od * weak_od_factor)
        & (gray < max_gray)
        & (b > g + 0.01)
        & (r > g - 0.12)
        & (saturation > min_saturation * 0.5)
    )
    mask = stained | pink_like | purple_like
    if img.shape[2] >= 4:
        alpha = img[..., 3]
        mask &= alpha > 0
    mask = mask.astype(bool)
    if debug:
        debug_info = {
            "rgb": rgb,
            "gray": gray,
            "saturation": saturation,
            "od_sum": od_sum,
            "stained": stained,
            "pink_like": pink_like,
            "purple_like": purple_like,
        }
        return (mask, debug_info)
    return mask


def encode_domains(values, unlabeled="", categories=None):
    values = np.asarray(values, dtype=object)
    valid = ~pd.isna(values) & (values != unlabeled)
    names = values[valid].astype(str)
    categories = (
        sorted(set(names)) if categories is None else list(map(str, categories))
    )
    if len(categories) != len(set(categories)) or len(categories) > 65535:
        raise ValueError("Categories must be unique and fit uint16")
    mapping = {name: i + 1 for (i, name) in enumerate(categories)}
    unknown = set(names) - mapping.keys()
    if unknown:
        raise ValueError(f"Labels missing from shared vocabulary: {sorted(unknown)}")
    result = np.zeros(values.shape, dtype=np.uint16)
    result[valid] = [mapping[name] for name in names]
    return (result, mapping)


def generate_labels(
    h5ad_path,
    image,
    output,
    column="niche",
    spatial_key="spatial_aligned",
    categories=None,
    **options,
):
    import anndata as ad

    data = ad.read_h5ad(h5ad_path)
    result = st_domains_to_he_wsi_region(
        image,
        data.obsm[spatial_key],
        data.obs[column],
        output,
        categories=categories,
        **options,
    )
    (_, mapping, shape, he_shape) = result
    Path(str(output) + ".json").write_text(
        json.dumps(
            {
                "dtype": "uint16",
                "shape": shape,
                "he_shape": he_shape,
                "region_size": 28,
                "column": column,
                "classes": mapping,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


def gaussian_kernel_1d(sigma: int = 60, truncate: float = 3.0, device="cuda"):
    if sigma <= 0 or truncate <= 0:
        raise ValueError("sigma and truncate must be positive")
    radius = int(truncate * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_filter_torch(x, sigma: int = 60, truncate: float = 3.0):
    device = x.device
    kernel = gaussian_kernel_1d(sigma=sigma, truncate=truncate, device=device)
    r = kernel.numel() // 2
    kx = kernel.view(1, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1)
    x = F.conv2d(x, kx, padding=(0, r))
    x = F.conv2d(x, ky, padding=(r, 0))
    return x


def st_domains_to_he_batched(
    HE,
    coords_st_aligned,
    st_domains,
    tissue_mask=None,
    sigma: int = 60,
    truncate: float = 3.0,
    min_total_weight: float = 0.0001,
    min_winner_frac: float = 0.45,
    min_margin: float = 0.0001,
    device="cuda",
):
    """Apply Gaussian filtering to all classes in parallel for faster label propagation."""
    (H, W) = HE
    coords_st_aligned = np.asarray(coords_st_aligned, dtype=np.float32)
    st_domains = np.asarray(st_domains, dtype=np.uint16)
    x = np.rint(coords_st_aligned[:, 0]).astype(np.int64)
    y = np.rint(coords_st_aligned[:, 1]).astype(np.int64)
    valid = np.isfinite(coords_st_aligned).all(axis=1) & (st_domains > 0)
    inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    valid &= inside
    (x, y, st_domains) = (x[valid], y[valid], st_domains[valid])
    present_domains = np.unique(st_domains)
    num_domains = len(present_domains)
    if num_domains == 0:
        return np.zeros((H, W), dtype=np.uint16)
    impulses = torch.zeros((num_domains, 1, H, W), dtype=torch.float32, device=device)
    for i, label in enumerate(present_domains):
        idx = st_domains == int(label)
        yy = np.asarray(y[idx]).astype(np.int64)
        xx = np.asarray(x[idx]).astype(np.int64)
        impulses[i, 0, yy, xx] = 1.0
    scores = gaussian_filter_torch(impulses, sigma=sigma, truncate=truncate)
    scores = scores[:, 0]
    total = scores.sum(dim=0)
    (sorted_scores, sorted_idx) = torch.sort(scores, dim=0, descending=True)
    best = sorted_scores[0]
    second = sorted_scores[1] if num_domains > 1 else torch.zeros_like(best)
    winner_idx = sorted_idx[0]
    winner = torch.zeros_like(winner_idx, dtype=torch.int32)
    for i, label in enumerate(present_domains):
        winner[winner_idx == i] = int(label)
    winner_frac = best / (total + 1e-08)
    margin = best - second
    spot_any = np.zeros((H, W), dtype=bool)
    spot_any[y, x] = True
    dist_to_spot = distance_transform_edt(~spot_any)
    max_dist = sigma * truncate
    if tissue_mask is None:
        assign_np = dist_to_spot < max_dist
    else:
        tissue_mask = np.asarray(tissue_mask).astype(bool)
        if tissue_mask.shape != (H, W):
            raise ValueError(
                f"tissue_mask shape {tissue_mask.shape} does not match HE shape {(H, W)}"
            )
        assign_np = tissue_mask & (dist_to_spot < max_dist)
    assign = torch.from_numpy(assign_np).to(device)
    assign = (
        assign
        & (total > min_total_weight)
        & (winner_frac > min_winner_frac)
        & (margin > min_margin)
    )
    label_id_map = torch.where(assign, winner, torch.zeros_like(winner))
    label_id_map = label_id_map.cpu().numpy().astype(np.uint16)
    del impulses, scores, best, second, total, winner, assign
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return label_id_map


def pix_to_region(label_crop: np.ndarray, region_size: int = 28) -> np.ndarray:
    """Compress pixel labels to region labels by majority vote within each region."""
    (h, w) = label_crop.shape
    out_h = (h + region_size - 1) // region_size
    out_w = (w + region_size - 1) // region_size
    region_label = np.zeros((out_h, out_w), dtype=np.uint16)
    for oy in range(out_h):
        y0 = oy * region_size
        y1 = min(y0 + region_size, h)
        for ox in range(out_w):
            x0 = ox * region_size
            x1 = min(x0 + region_size, w)
            block = label_crop[y0:y1, x0:x1].reshape(-1)
            if block.size == 0:
                continue
            block = block.astype(np.int64, copy=False)
            region_label[oy, ox] = np.bincount(block).argmax()
    return region_label


def st_domains_to_he_wsi_region(
    tif_path,
    coords_st_aligned,
    st_domains,
    out_path: str = "./label_region_uint16.dat",
    region_size: int = 28,
    coarse_tile_size: int = 128,
    sigma: int = 60,
    truncate: float = 3.0,
    min_total_weight: float = 0.0001,
    min_winner_frac: float = 0.45,
    min_margin: float = 0.0001,
    device="cuda",
    categories=None,
):
    """Generate region-level supervision labels for a whole-slide image."""
    if region_size != 28 or coarse_tile_size <= 0:
        raise ValueError("region_size must be 28 and coarse_tile_size positive")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    slide = open_slide(tif_path)
    (W, H) = slide.dimensions
    region_H = H // 224 * (224 // region_size)
    region_W = W // 224 * (224 // region_size)
    coords = np.asarray(coords_st_aligned)
    (domain_ids, domain_to_id) = encode_domains(
        st_domains, unlabeled=UNLABELED, categories=categories
    )
    x_all = coords[:, 0]
    y_all = coords[:, 1]
    halo = int(sigma * truncate + 0.5)
    label_region_mm = np.memmap(
        out_path, dtype=np.uint16, mode="w+", shape=(region_H, region_W)
    )
    label_region_mm[:] = 0
    label_region_mm.flush()
    ys = list(range(0, region_H, coarse_tile_size))
    xs = list(range(0, region_W, coarse_tile_size))
    tile_count = len(ys) * len(xs)
    tiles = ((gy, gx) for gy in ys for gx in xs)
    for gy, gx in tqdm(tiles, total=tile_count, desc="Generating labels", unit="tile"):
        gy1 = min(gy + coarse_tile_size, region_H)
        gx1 = min(gx + coarse_tile_size, region_W)
        x = gx * region_size
        y = gy * region_size
        x_end = min(gx1 * region_size, W)
        y_end = min(gy1 * region_size, H)
        w = x_end - x
        h = y_end - y
        if w < 0 or h < 0:
            continue
        hx0 = max(0, x - halo)
        hy0 = max(0, y - halo)
        hx1 = min(W, x_end + halo)
        hy1 = min(H, y_end + halo)
        hw = hx1 - hx0
        hh = hy1 - hy0
        HE_halo = np.array(slide.read_region((hx0, hy0), 0, (hw, hh)))[..., :3]
        in_halo = (
            (x_all >= hx0)
            & (x_all < hx1)
            & (y_all >= hy0)
            & (y_all < hy1)
            & (domain_ids > 0)
        )
        if not np.any(in_halo):
            label_region_mm[gy:gy1, gx:gx1] = 0
            label_region_mm.flush()
            continue
        coords_halo = coords[in_halo].copy()
        coords_halo[:, 0] -= hx0
        coords_halo[:, 1] -= hy0
        domain_ids_halo = domain_ids[in_halo]
        tissue_mask_halo = make_tissue_mask(HE_rgb=HE_halo)
        label_halo = st_domains_to_he_batched(
            HE=(hh, hw),
            coords_st_aligned=coords_halo,
            st_domains=domain_ids_halo,
            tissue_mask=tissue_mask_halo,
            sigma=sigma,
            truncate=truncate,
            min_total_weight=min_total_weight,
            min_winner_frac=min_winner_frac,
            min_margin=min_margin,
            device=device,
        )
        cy0 = y - hy0
        cx0 = x - hx0
        label_center = label_halo[cy0 : cy0 + h, cx0 : cx0 + w]
        region_label = pix_to_region(label_crop=label_center, region_size=region_size)
        label_region_mm[
            gy : gy + region_label.shape[0], gx : gx + region_label.shape[1]
        ] = region_label
        label_region_mm.flush()
    slide.close()
    return (out_path, domain_to_id, (region_H, region_W), (H, W))


def convert_niche_to_domain(
    niche_path, output, h5ad_path, niche_categories, domain_categories
):
    import anndata as ad

    data = ad.read_h5ad(h5ad_path)
    mapping = {}
    for niche, domain in zip(data.obs["niche"], data.obs["domain"]):
        if pd.isna(niche) or pd.isna(domain):
            continue
        (niche, domain) = (str(niche), str(domain))
        if niche in mapping and mapping[niche] != domain:
            raise ValueError(f"Niche {niche!r} maps to multiple domains")
        mapping[niche] = domain
    (_, domain_ids) = encode_domains(domain_categories, categories=domain_categories)
    lookup = np.zeros(len(niche_categories) + 1, dtype=np.uint16)
    for i, niche in enumerate(niche_categories, 1):
        lookup[i] = domain_ids[mapping[str(niche)]]
    source = np.memmap(niche_path, mode="r", dtype=np.uint16)
    if source.size and source.max() >= len(lookup):
        raise ValueError("Niche IDs do not match supplied vocabulary")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    dest = np.memmap(output, mode="w+", dtype=np.uint16, shape=source.shape)
    for start in range(0, len(source), 1000000):
        dest[start : start + 1000000] = lookup[source[start : start + 1000000]]
    dest.flush()
    return str(output)

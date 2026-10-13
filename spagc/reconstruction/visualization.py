"""Stratified sampling and interactive annotated-volume export."""

from __future__ import annotations

import h5py
import numpy as np
import plotly.graph_objects as go


def decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def read_categorical(obs: h5py.Group, key: str) -> tuple[np.ndarray, list[str]]:
    group = obs[key]
    return (group["codes"][:].astype(np.int16), decode_strings(group["categories"][:]))


def read_obs_column(obs: h5py.Group, key: str) -> np.ndarray:
    obj = obs[key]
    if isinstance(obj, h5py.Group):
        (codes, categories) = read_categorical(obs, key)
        values = np.full(codes.size, None, dtype=object)
        valid = codes >= 0
        values[valid] = np.asarray(categories, dtype=object)[codes[valid]]
        return values
    values = obj[:]
    if values.dtype.kind in "SO":
        return np.asarray(decode_strings(values), dtype=object)
    return values


def add_stratified_indices(
    selected: np.ndarray,
    label_codes: np.ndarray,
    z_codes: np.ndarray,
    quota: int,
    rng: np.random.Generator,
) -> None:
    """Select up to ``quota`` cells for every label-by-depth combination."""
    n_z = int(z_codes.max()) + 1
    group_codes = label_codes.astype(np.int64) * n_z + z_codes
    order = np.argsort(group_codes, kind="stable")
    ordered_groups = group_codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_groups)) + 1]
    ends = np.r_[starts[1:], order.size]
    for start, end in zip(starts, ends):
        candidates = order[start:end]
        if candidates.size > quota:
            candidates = rng.choice(candidates, size=quota, replace=False)
        selected[candidates] = True


def select_display_sample(
    niche_codes: np.ndarray,
    domain_codes: np.ndarray,
    z: np.ndarray,
    max_points: int,
    points_per_stratum: int,
    seed: int,
) -> np.ndarray:
    """Create a deterministic sample that retains labels throughout the volume."""
    n_obs = z.size
    if max_points >= n_obs:
        return np.arange(n_obs)
    rng = np.random.default_rng(seed)
    (_, z_codes) = np.unique(np.round(z, decimals=4), return_inverse=True)
    selected = np.zeros(n_obs, dtype=bool)
    add_stratified_indices(selected, niche_codes, z_codes, points_per_stratum, rng)
    add_stratified_indices(selected, domain_codes, z_codes, points_per_stratum, rng)
    for codes in (niche_codes, domain_codes):
        (_, counts) = np.unique(codes, return_counts=True)
        for code, count in enumerate(counts):
            if count <= 1500:
                selected[codes == code] = True
    chosen = np.flatnonzero(selected)
    if chosen.size > max_points:
        chosen = rng.choice(chosen, size=max_points, replace=False)
    elif chosen.size < max_points:
        remaining = np.flatnonzero(~selected)
        fill = rng.choice(remaining, size=max_points - chosen.size, replace=False)
        chosen = np.r_[chosen, fill]
    return np.sort(chosen)


def add_label_traces(
    fig: go.Figure,
    coords: np.ndarray,
    z: np.ndarray,
    codes: np.ndarray,
    confidence: np.ndarray,
    categories: list[str],
    colors: dict[str, str],
    visible: bool,
    trace_group: str,
) -> int:
    n_traces = 0
    for code, category in enumerate(categories):
        mask = codes == code
        if not np.any(mask):
            continue
        fig.add_trace(
            go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=z[mask],
                mode="markers",
                name=category,
                legendgroup=trace_group,
                visible=visible,
                marker={
                    "size": 1.8,
                    "color": colors.get(category, "#808080"),
                    "opacity": 0.72,
                },
                customdata=confidence[mask, None],
                hovertemplate="<b>%{fullData.name}</b><br>x: %{x:.1f} µm<br>y: %{y:.1f} µm<br>z: %{z:.1f} µm<br>confidence: %{customdata[0]:.3f}<extra></extra>",
            )
        )
        n_traces += 1
    return n_traces


def render_volume(
    input_path,
    output_path,
    spatial_key="spatial_st_corrected",
    z_key="z_um",
    group_key=None,
    group_value=None,
    max_points=100000,
    seed=29,
    points_per_stratum=60,
    *,
    domain_colors=None,
    niche_colors=None,
) -> None:
    """Export a self-contained interactive 3D domain/niche HTML without changing predictions."""
    domain_colors = {} if domain_colors is None else domain_colors
    niche_colors = {} if niche_colors is None else niche_colors
    if (group_key is None) != (group_value is None):
        raise ValueError("--group-key and --group-value must be provided together")
    with h5py.File(input_path, "r") as handle:
        obs = handle["obs"]
        coords = handle[f"obsm/{spatial_key}"][:]
        z = np.asarray(read_obs_column(obs, z_key), dtype=float)
        niche_codes, niche_categories = read_categorical(obs, "pred_niche")
        domain_codes, domain_categories = read_categorical(obs, "pred_domain")
        niche_confidence = obs["pred_niche_conf"][:]
        domain_confidence = obs["pred_domain_conf"][:]
        if group_key is not None:
            group_values = read_obs_column(obs, group_key)
            group_mask = np.asarray(group_values, dtype=str) == group_value
            if not group_mask.any():
                raise ValueError(f"No cells found for {group_key}={group_value!r}")
            coords = coords[group_mask]
            z = z[group_mask]
            niche_codes = niche_codes[group_mask]
            domain_codes = domain_codes[group_mask]
            niche_confidence = niche_confidence[group_mask]
            domain_confidence = domain_confidence[group_mask]
    finite = np.isfinite(coords).all(axis=1) & np.isfinite(z)
    if not finite.all():
        coords = coords[finite]
        z = z[finite]
        niche_codes = niche_codes[finite]
        domain_codes = domain_codes[finite]
        niche_confidence = niche_confidence[finite]
        domain_confidence = domain_confidence[finite]
    display_indices = select_display_sample(
        niche_codes,
        domain_codes,
        z,
        max_points=max_points,
        points_per_stratum=points_per_stratum,
        seed=seed,
    )
    coords_display = coords[display_indices]
    z_display = z[display_indices]
    fig = go.Figure()
    n_domain = add_label_traces(
        fig,
        coords_display,
        z_display,
        domain_codes[display_indices],
        domain_confidence[display_indices],
        domain_categories,
        domain_colors,
        visible=True,
        trace_group="domain",
    )
    n_niche = add_label_traces(
        fig,
        coords_display,
        z_display,
        niche_codes[display_indices],
        niche_confidence[display_indices],
        niche_categories,
        niche_colors,
        visible=False,
        trace_group="niche",
    )
    domain_visibility = [True] * n_domain + [False] * n_niche
    niche_visibility = [False] * n_domain + [True] * n_niche
    sample_note = f"Interactive display: {display_indices.size:,} stratified cells from {z.size:,} predictions; display sampling does not alter the full result"
    group_suffix = f" — {group_value}" if group_value is not None else ""
    domain_title = f"3D gastric cancer spatial domains{group_suffix}"
    niche_title = f"3D gastric cancer spatial niches{group_suffix}"
    fig.update_layout(
        title={
            "text": f"{domain_title}<br><sup>{sample_note}</sup>",
            "x": 0.01,
            "xanchor": "left",
        },
        template="plotly_white",
        margin={"l": 0, "r": 250, "t": 95, "b": 0},
        legend={
            "title": {"text": "Domain"},
            "itemsizing": "constant",
            "x": 1.01,
            "y": 0.98,
            "xanchor": "left",
            "yanchor": "top",
        },
        scene={
            "xaxis": {"title": "x (µm)", "gridcolor": "#E3E3E3"},
            "yaxis": {"title": "y (µm)", "gridcolor": "#E3E3E3"},
            "zaxis": {"title": "z depth (µm)", "gridcolor": "#E3E3E3"},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.0, "y": 0.85, "z": 0.45},
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.1}},
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.01,
                "y": 1.03,
                "xanchor": "left",
                "yanchor": "bottom",
                "showactive": True,
                "buttons": [
                    {
                        "label": "Domains",
                        "method": "update",
                        "args": [
                            {"visible": domain_visibility},
                            {
                                "title.text": f"{domain_title}<br><sup>{sample_note}</sup>",
                                "legend.title.text": "Domain",
                            },
                        ],
                    },
                    {
                        "label": "Niches",
                        "method": "update",
                        "args": [
                            {"visible": niche_visibility},
                            {
                                "title.text": f"{niche_title}<br><sup>{sample_note}</sup>",
                                "legend.title.text": "Niche",
                            },
                        ],
                    },
                ],
            }
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "adata_00029_3d_spatial_predictions",
                "scale": 2,
            },
        },
    )
    print(f"Wrote {output_path}")
    print(f"Displayed {display_indices.size:,} / {z.size:,} cells")

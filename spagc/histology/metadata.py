"""Slide geometry, label vocabularies and patient partitions."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from spagc.histology.slides import open_slide


def attach_sample_metadata(slides: list[dict], metadata_path: str | None) -> list[dict]:
    """Optionally attach patient/centre IDs; never invent identity from slide names."""
    records = [dict(slide) for slide in slides]
    if not metadata_path:
        return records
    table = pd.read_csv(metadata_path, dtype=str)
    if not {"name", "patient_id"}.issubset(table.columns):
        raise ValueError("Sample metadata needs name and patient_id columns.")
    if (
        table["name"].duplicated().any()
        or table[["name", "patient_id"]].isna().any().any()
    ):
        raise ValueError(
            "Metadata must have unique slide names and nonmissing patient IDs."
        )
    indexed = table.set_index("name")
    for slide in records:
        if slide["name"] not in indexed.index:
            raise ValueError(f"Missing patient metadata for {slide['name']}")
        slide.update(indexed.loc[slide["name"]].dropna().to_dict())
    return records


def validate_patient_partition(train: list[dict], validation: list[dict]) -> bool:
    """Validate patient separation when identities are supplied; otherwise report unknown."""
    records = train + validation
    supplied = [bool(record.get("patient_id")) for record in records]
    if any(supplied) and (not all(supplied)):
        raise ValueError("Patient IDs must cover every training and validation slide.")
    if not all(supplied):
        return False
    overlap = {s["patient_id"] for s in train} & {s["patient_id"] for s in validation}
    if overlap:
        raise ValueError(
            f"Patients occur in both training and validation: {sorted(overlap)}"
        )
    return True


def with_background(num_real_classes: int) -> int:
    value = int(num_real_classes)
    if value <= 0:
        raise ValueError("class count must be a positive number of real classes")
    return value + 1


def counts_with_background(
    num_domain_classes: int, num_niche_classes: int
) -> tuple[int, int]:
    return (with_background(num_domain_classes), with_background(num_niche_classes))


def _level0_shape_from_image(path: str) -> tuple[int, int]:
    """Return level-0 image shape as (H, W) for tif/kfb/svs/mrxs-like slides."""
    try:
        with open_slide(path) as slide:
            (level0_w, level0_h) = map(int, slide.dimensions)
            return (level0_h, level0_w)
    except Exception as exc:
        raise ValueError(f"Failed to read image dimensions for {path}: {exc}") from exc


def _label_shape_from_level0_slide(
    he_tif_path: str, region_size: int = 28, tile_size: int = 224
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the baseline output shape and level-0 HE shape.

    Match the raster geometry used during label propagation:

        region_H = H // 224 * (224 // region_size)
        region_W = W // 224 * (224 // region_size)

    The full-resolution slide geometry is authoritative.  H5 coverage is not:
    Extraction defaults to include_edge=False, so the last H5 tile can stop before
    the right/bottom image boundary.
    """
    if tile_size % region_size != 0:
        raise ValueError(
            f"tile_size={tile_size} must be divisible by region_size={region_size}"
        )
    (level0_h, level0_w) = _level0_shape_from_image(he_tif_path)
    grid_per_tile = tile_size // region_size
    out_h = level0_h // tile_size * grid_per_tile
    out_w = level0_w // tile_size * grid_per_tile
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"Invalid level-0 slide dimensions {(level0_h, level0_w)} for tile_size={tile_size}: {he_tif_path}"
        )
    return ((out_h, out_w), (level0_h, level0_w))


def _shape_from_slide_metadata(slide):
    if "label_shape" in slide:
        shape = slide["label_shape"]
    elif "st_shape" in slide:
        shape = slide["st_shape"]
    elif "H" in slide and "W" in slide:
        shape = (slide["H"], slide["W"])
    else:
        return None
    if len(shape) != 2:
        raise ValueError(f"{slide['name']}: label_shape must have two values")
    shape = tuple(map(int, shape))
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"{slide['name']}: invalid label_shape={shape}")
    return shape


def setup_slide_shapes(slides):
    for slide in slides:
        shape = _shape_from_slide_metadata(slide)
        if shape is None:
            (shape, full) = _label_shape_from_level0_slide(slide["he_tif_path"])
        else:
            full = tuple(
                map(
                    int,
                    slide.get(
                        "level0_shape",
                        slide.get("he_level0_shape", (shape[0] * 28, shape[1] * 28)),
                    ),
                )
            )
        slide["label_shape"] = shape
        slide["he_shape"] = tuple(n * 28 for n in shape)
        slide["level0_shape"] = full
        (slide["H"], slide["W"]) = shape


def validate_training_slides(
    train, val, domain_classes, niche_classes, spatial_split=False
):
    validate_patient_partition(train, val)
    (domain_output_classes, niche_output_classes) = counts_with_background(
        domain_classes, niche_classes
    )
    if not spatial_split:
        a = {str(Path(s["h5_path"]).resolve()) for s in train}
        b = {str(Path(s["h5_path"]).resolve()) for s in val}
        if a & b or {s["name"] for s in train} & {s["name"] for s in val}:
            raise ValueError(
                "Train/validation contain the same slide; use disjoint manifests"
            )
    vocabularies = {}
    for slide in train + val:
        niche_path = slide.get("niche_label_path", slide.get("label_path"))
        for key, classes in [
            ("niche_label_path", niche_output_classes),
            ("domain_label_path", domain_output_classes),
        ]:
            if key == "niche_label_path" and niche_path is not None:
                path = niche_path
            else:
                path = slide.get(key)
            if path is None:
                continue
            sidecar = Path(str(path) + ".json")
            if sidecar.exists():
                vocabulary = json.loads(sidecar.read_text()).get("classes")
                if vocabulary is not None:
                    if key in vocabularies and vocabularies[key] != vocabulary:
                        raise ValueError(
                            f"{key}: inconsistent class vocabularies across slides"
                        )
                    vocabularies[key] = vocabulary
            expected = int(np.prod(slide["label_shape"])) * 2
            if Path(path).stat().st_size != expected:
                raise ValueError(f"{path}: expected {expected} bytes for label_shape")
            labels = np.memmap(path, dtype=np.uint16, mode="r")
            if labels.max() >= classes:
                raise ValueError(
                    f"{path}: max ID {labels.max()} requires real class count >= {int(labels.max())}"
                )


def prepare_slides(records, *, root=None):
    """Normalize in-notebook slide records using the exported validation rules."""
    root = Path.cwd() if root is None else Path(root)
    slides = deepcopy(records)
    if not isinstance(slides, list) or not slides:
        raise ValueError("Slide records must be a nonempty list")
    names = set()
    for slide in slides:
        name = slide["name"]
        if name in names or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Duplicate or unsafe slide name: {name}")
        names.add(name)
        if "niche_label_path" in slide:
            if (
                slide.get("label_path", slide["niche_label_path"])
                != slide["niche_label_path"]
            ):
                raise ValueError("Conflicting niche label paths")
        elif "label_path" in slide:
            slide["niche_label_path"] = slide["label_path"]
        for key, value in list(slide.items()):
            if key.endswith("_path") and value:
                path = Path(value).expanduser()
                slide[key] = str(path if path.is_absolute() else root / path)
    return slides

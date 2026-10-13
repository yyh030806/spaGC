"""Saved registration chains and calibrated native-image patch access."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvips
from PIL import Image
from scipy.ndimage import binary_erosion, map_coordinates


def _validate_points(points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    if value.ndim != 2 or value.shape[1] != 2 or (not np.isfinite(value).all()):
        raise ValueError("points must be a finite N x 2 array")
    return value


def _apply_homogeneous(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("matrix must be a finite 3 x 3 homogeneous transform")
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=float)])
    transformed = homogeneous @ matrix.T
    if np.any(np.isclose(transformed[:, 2], 0.0)):
        raise ValueError("homogeneous transform produced zero scale")
    return transformed[:, :2] / transformed[:, 2, None]


def source_to_registered_points(points, section, store, *, batch_size=32768):
    """Apply preorientation and an existing nonlinear edge chain."""
    value = _validate_points(points)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    result = np.empty_like(value)
    for start in range(0, len(value), int(batch_size)):
        stop = min(start + int(batch_size), len(value))
        batch = _apply_homogeneous(value[start:stop], section.preorientation_matrix)
        for edge_key, direction in section.edge_chain:
            batch = store.transform(batch, store.load_edge(edge_key), direction)
        result[start:stop] = batch
    if not np.isfinite(result).all():
        raise ValueError("registration produced non-finite coordinates")
    return result


def registered_to_source_points(points, section, store, *, batch_size=32768):
    """Invert a saved registration chain back to its source H&E frame."""
    value = _validate_points(points)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    inverse_preorientation = np.linalg.inv(
        np.asarray(section.preorientation_matrix, dtype=float)
    )
    result = np.empty_like(value)
    opposite = {"forward": "reverse", "reverse": "forward"}
    for start in range(0, len(value), int(batch_size)):
        stop = min(start + int(batch_size), len(value))
        batch = value[start:stop].copy()
        for edge_key, direction in reversed(section.edge_chain):
            if direction not in opposite:
                raise ValueError(f"unknown edge direction: {direction}")
            batch = store.transform(
                batch, store.load_edge(edge_key), opposite[direction]
            )
        result[start:stop] = _apply_homogeneous(batch, inverse_preorientation)
    if not np.isfinite(result).all():
        raise ValueError("inverse registration produced non-finite coordinates")
    return result


@dataclass
class SavedSectionTransform:
    preorientation_matrix: np.ndarray
    edge_chain: list
    matrix: np.ndarray


def load_section_transform(path):
    """Read the release's XY preorientation and ordered nonlinear edge chain."""
    with np.load(path, allow_pickle=False) as data:
        return SavedSectionTransform(
            data["preorientation_matrix"].copy(),
            json.loads(data["edge_chain_json"].item()),
            data["matrix"].copy(),
        )


class SavedTransformStore:
    """Evaluate saved STalign fields; never re-estimate a registration.

    Saved A/velocity arrays use row-column order, unlike public XY interfaces.
    Forward Euler evaluation matches STalign. Reverse evaluation solves each
    saved forward Euler step numerically in reverse order. This corrects the
    accumulated error of the upstream approximate backward traversal without
    fitting a new transform or changing the supplied physical frame.
    """

    def __init__(self, directory, *, tolerance_um=1e-07, max_iterations=100):
        self.directory = Path(directory).resolve()
        self.tolerance_um = float(tolerance_um)
        self.max_iterations = int(max_iterations)
        self._edges = {}

    def load_edge(self, key):
        if key not in self._edges:
            path = (self.directory / f"forward-{key}.npz").resolve()
            if not path.is_relative_to(self.directory):
                raise ValueError("Invalid saved edge key.")
            with np.load(path, allow_pickle=False) as saved:
                edge = {name: saved[name].copy() for name in ["A", "v", "xv_0", "xv_1"]}
            axes = [edge["xv_0"], edge["xv_1"]]
            if any(
                len(a) < 2
                or not np.allclose(np.diff(a), np.diff(a)[0])
                or np.diff(a)[0] <= 0
                for a in axes
            ):
                raise ValueError(
                    "Saved velocity axes must be regular increasing grids."
                )
            if not all(np.isfinite(v).all() for v in edge.values()):
                raise ValueError("Saved transform contains nonfinite values.")
            self._edges[key] = edge
        return self._edges[key]

    @staticmethod
    def _sample(field, points_yx, origin, spacing):
        indices = ((points_yx - origin) / spacing).T
        return np.column_stack(
            [
                map_coordinates(
                    field[..., component],
                    indices,
                    order=1,
                    mode="grid-constant",
                    cval=0.0,
                    prefilter=False,
                )
                for component in range(2)
            ]
        )

    def transform(self, points_xy, edge, direction):
        if direction not in {"forward", "reverse"}:
            raise ValueError(f"Unknown saved edge direction: {direction}")
        points = np.asarray(points_xy, dtype=float)[:, ::-1].copy()
        (affine, velocity) = (edge["A"], edge["v"])
        origin = np.array([edge["xv_0"][0], edge["xv_1"][0]])
        spacing = np.array([np.diff(edge["xv_0"])[0], np.diff(edge["xv_1"])[0]])
        if direction == "forward":
            for field in velocity:
                points += self._sample(field, points, origin, spacing) / len(velocity)
            points = points @ affine[:2, :2].T + affine[:2, 2]
        else:
            inverse = np.linalg.inv(affine)
            points = points @ inverse[:2, :2].T + inverse[:2, 2]
            for field in velocity[::-1]:
                target = points.copy()
                for _ in range(self.max_iterations):
                    updated = target - self._sample(
                        field, points, origin, spacing
                    ) / len(velocity)
                    error = np.max(np.abs(updated - points), initial=0.0)
                    points = updated
                    if error <= self.tolerance_um:
                        break
                else:
                    raise ValueError(
                        "Saved nonlinear inverse did not converge; no affine fallback is permitted."
                    )
        if not np.isfinite(points).all():
            raise ValueError("Saved transform produced nonfinite coordinates.")
        return points[:, ::-1]

    def clear(self):
        self._edges.clear()


class NativeCropReader:
    """Lazy tiled OME-TIFF access without loading a gigabyte-scale RGB image."""

    def __init__(self, row):
        self.row = row
        self.metadata = json.loads(Path(row.metadata_file).read_text())
        self.mpp = float(self.metadata["raw_mpp_um"])
        self.source_origin = np.asarray(
            self.metadata["source_frame_crop_origin_um"], float
        )
        pyvips.cache_set_max_mem(128 * 1024**2)
        pyvips.cache_set_max(64)
        self.image = pyvips.Image.new_from_file(str(row.he_file), access="random")
        self.mask = pyvips.Image.new_from_file(str(row.mask_file), access="random")
        expected = tuple(self.metadata["raw_crop_shape_yx"])
        if (self.image.height, self.image.width) != expected or self.image.bands != 3:
            raise ValueError("Native RGB dimensions differ from the bundle contract.")
        if (self.mask.height, self.mask.width) != expected:
            raise ValueError("Native tissue mask and image dimensions differ.")
        self.section = load_section_transform(row.section_transform)
        self.transforms = SavedTransformStore(Path(row.section_transform).parent)

    def source_to_pixels(self, source_um):
        """Companion sections use local zero; serial sections use raw crop origin."""
        return (np.asarray(source_um, float) - self.source_origin) / self.mpp

    def registered_to_source(self, xy):
        return registered_to_source_points(xy, self.section, self.transforms)

    def center_supported(self, source_um):
        pixels = np.rint(self.source_to_pixels(source_um)).astype(int)
        valid = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 0] < self.mask.width)
            & (pixels[:, 1] < self.mask.height)
        )
        for i in np.flatnonzero(valid):
            valid[i] = self.mask.getpoint(int(pixels[i, 0]), int(pixels[i, 1]))[0] > 0
        return valid

    def registered_tissue_bounds(self):
        """Use a coarse mask only for bounding the grid, never for RGB extraction."""
        step = 16
        small = self.mask.subsample(step, step).cast("uchar")
        array = (
            np.frombuffer(small.write_to_memory(), np.uint8).reshape(
                small.height, small.width
            )
            > 0
        )
        boundary = array & ~binary_erosion(array)
        (iy, ix) = np.nonzero(boundary)
        if not len(ix):
            raise ValueError("Native tissue mask is empty.")
        selected = np.linspace(0, len(ix) - 1, min(len(ix), 20000), dtype=int)
        source = (
            np.column_stack([ix[selected], iy[selected]]) * step * self.mpp
            + self.source_origin
        )
        xy = source_to_registered_points(source, self.section, self.transforms)
        return (xy.min(0), xy.max(0))

    def extract_patch(self, source_um, *, patch_size_um=112.0, output_size=224):
        """Read native pixels at fixed source-oriented FOV, then resize once."""
        size = max(1, int(round(float(patch_size_um) / self.mpp)))
        start = np.rint(self.source_to_pixels(source_um) - size / 2).astype(int)
        end = start + size
        lo = np.maximum(start, 0)
        hi = np.minimum(end, [self.image.width, self.image.height])
        canvas = Image.new("RGB", (size, size), "white")
        if (hi > lo).all():
            crop = self.image.crop(
                int(lo[0]), int(lo[1]), int(hi[0] - lo[0]), int(hi[1] - lo[1])
            ).cast("uchar")
            values = np.frombuffer(crop.write_to_memory(), np.uint8).reshape(
                crop.height, crop.width, 3
            )
            canvas.paste(Image.fromarray(values), tuple((lo - start).astype(int)))
        return canvas.resize(
            (int(output_size), int(output_size)), Image.Resampling.BILINEAR
        )

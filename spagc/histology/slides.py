"""Bounded-memory readers for native whole-slide images."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_MAX_EAGER_LEVEL_PIXELS = 64 * 1024 * 1024
_FALLBACK_SOURCE_TILE_SIDE = 4096
KFB_SUFFIXES = {".kfb"}


def is_mirax_directory(path: str | os.PathLike[str]) -> bool:
    """Recognize a MIRAX data directory without scanning its large DAT files."""
    folder = Path(path)
    return folder.is_dir() and (folder / "Slidedat.ini").is_file()


def resolve_slide_path(path: str | os.PathLike[str]) -> Path:
    """Accept an image file or MIRAX data directory and return its entry file.

    MIRAX needs both ``sample.mrxs`` and the adjacent ``sample/`` directory.
    Never guess another slide or attempt to interpret individual DAT files.
    """
    source = Path(path).expanduser()
    if source.is_dir():
        if not is_mirax_directory(source):
            raise ValueError(f"MIRAX directory is missing Slidedat.ini: {source}")
        candidates = sorted(
            p
            for p in source.parent.iterdir()
            if p.is_file() and p.stem == source.name and (p.suffix.lower() == ".mrxs")
        )
        if not candidates:
            raise FileNotFoundError(
                f"MIRAX entry file is missing: {source.parent / (source.name + '.mrxs')}. Keep the .mrxs file alongside its matching data directory."
            )
        if len(candidates) != 1:
            raise ValueError(f"Multiple MIRAX entry files found: {candidates}")
        source = candidates[0]
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def _is_kfb(path: str | os.PathLike[str]) -> bool:
    return Path(path).suffix.lower() in KFB_SUFFIXES


def _open_kfb(path: str):
    try:
        import kfbslide
    except ImportError as exc:
        raise ImportError(
            "KFB images require the optional kfbslide reader (tested with version 0.3.3)."
        ) from exc
    try:
        return kfbslide.OpenSlide(path)
    except Exception as exc:
        raise ValueError(f"kfbslide could not open {path}: {exc}") from exc


def _rgb_on_white(image: Image.Image) -> Image.Image:
    """Composite unscanned/partially covered WSI pixels onto white for RGB."""
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    if alpha.getextrema() == (255, 255):
        return rgba.convert("RGB")
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=alpha)
    return background


class SlideReader:
    """OpenSlide-like reader used throughout the project.

    Coordinates passed to :meth:`read_region` are always level-0 coordinates,
    matching the OpenSlide convention.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = os.fspath(resolve_slide_path(path))
        self.backend = ""
        self._slide: Any = None
        self._vips: Any = None
        errors: list[str] = []
        if _is_kfb(self.path):
            self._slide = _open_kfb(self.path)
            self.backend = "kfbslide"
            return
        try:
            import openslide

            self._slide = openslide.OpenSlide(self.path)
            self.backend = "openslide"
            return
        except Exception as exc:
            errors.append(f"OpenSlide: {exc}")
        try:
            import pyvips

            self._vips = pyvips.Image.new_from_file(self.path, access="random")
            self.backend = "pyvips"
        except Exception as exc:
            errors.append(f"libvips: {exc}")
            raise ValueError(
                f"Could not read image {self.path}: {'; '.join(errors)}"
            ) from exc

    @property
    def dimensions(self) -> tuple[int, int]:
        if self._slide is not None:
            (width, height) = self._slide.dimensions
        else:
            (width, height) = (self._vips.width, self._vips.height)
        return (int(width), int(height))

    @property
    def width(self) -> int:
        return self.dimensions[0]

    @property
    def height(self) -> int:
        return self.dimensions[1]

    @property
    def level_count(self) -> int:
        if self._slide is not None:
            return int(self._slide.level_count)
        return 1

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        if self._slide is not None:
            return tuple(
                (int(width), int(height))
                for (width, height) in self._slide.level_dimensions
            )
        return (self.dimensions,)

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        if self._slide is not None:
            return tuple(map(float, self._slide.level_downsamples))
        return (1.0,)

    @property
    def properties(self):
        if self._slide is not None:
            return self._slide.properties
        return {}

    def get_best_level_for_downsample(self, downsample: float) -> int:
        if self._slide is not None:
            return int(self._slide.get_best_level_for_downsample(float(downsample)))
        return 0

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> Image.Image:
        """Read a region and return an RGBA PIL image of exactly ``size``."""
        (x, y) = map(int, location)
        (width, height) = map(int, size)
        level = int(level)
        if width <= 0 or height <= 0:
            raise ValueError(f"region size must be positive, got {(width, height)}")
        if level < 0 or level >= self.level_count:
            raise ValueError(f"invalid level {level}; level_count={self.level_count}")
        if self._slide is not None:
            image = self._slide.read_region((x, y), level, (width, height))
            return image.convert("RGBA")
        canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        (image_width, image_height) = self.dimensions
        src_x0 = max(0, x)
        src_y0 = max(0, y)
        src_x1 = min(image_width, x + width)
        src_y1 = min(image_height, y + height)
        if src_x1 <= src_x0 or src_y1 <= src_y0:
            return canvas
        crop = self._vips.crop(src_x0, src_y0, src_x1 - src_x0, src_y1 - src_y0)
        if crop.format != "uchar":
            crop = crop.cast("uchar")
        array = np.ndarray(
            buffer=crop.write_to_memory(),
            dtype=np.uint8,
            shape=(crop.height, crop.width, crop.bands),
        )
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        mode = "RGBA" if array.shape[2] >= 4 else "RGB"
        patch = Image.fromarray(array[..., : 4 if mode == "RGBA" else 3], mode=mode)
        canvas.paste(patch, (src_x0 - x, src_y0 - y))
        return canvas

    def read_region_rgb(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> np.ndarray:
        return np.asarray(
            _rgb_on_white(self.read_region(location, level, size)), dtype=np.uint8
        )

    def close(self) -> None:
        if self._slide is not None:
            self._slide.close()
            self._slide = None
        self._vips = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> SlideReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def open_slide(path: str | os.PathLike[str]) -> SlideReader:
    """Open KFB, OpenSlide-compatible WSI, or a libvips-readable image."""
    return SlideReader(path)


def read_region_resized_rgb(
    slide: SlideReader,
    location: tuple[int, int],
    source_size: tuple[int, int],
    output_size: tuple[int, int],
    *,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> np.ndarray:
    """Read a level-0 rectangle efficiently and resize it to an exact size."""
    (source_width, source_height) = map(int, source_size)
    (output_width, output_height) = map(int, output_size)
    if min(source_width, source_height, output_width, output_height) <= 0:
        raise ValueError(
            f"source_size and output_size must be positive: {source_size}, {output_size}"
        )
    requested_downsample = min(
        source_width / output_width, source_height / output_height
    )
    level = slide.get_best_level_for_downsample(max(1.0, requested_downsample))
    downsample = slide.level_downsamples[level]
    level_size = (
        max(1, int(math.ceil(source_width / downsample))),
        max(1, int(math.ceil(source_height / downsample))),
    )
    if level_size[0] * level_size[1] > _MAX_EAGER_LEVEL_PIXELS:
        image = _read_region_resized_vips_rgb(slide, location, source_size, output_size)
        if image is not None:
            return image
        return _read_region_resized_tiled_rgb(
            slide, location, level, level_size, output_size, downsample, resample
        )
    image = _rgb_on_white(slide.read_region(location, level, level_size))
    if image.size != (output_width, output_height):
        image = image.resize((output_width, output_height), resample)
    return np.asarray(image, dtype=np.uint8)


def _read_region_resized_vips_rgb(
    slide: SlideReader,
    location: tuple[int, int],
    source_size: tuple[int, int],
    output_size: tuple[int, int],
) -> np.ndarray | None:
    """Stream a large level-0 rectangle through libvips when possible.

    This is the important BigTIFF path: single-level slides can contain tens of
    gigabytes, so asking OpenSlide/Pillow for the complete source rectangle before
    resizing is not viable.
    """
    (x, y) = map(int, location)
    (source_width, source_height) = map(int, source_size)
    (output_width, output_height) = map(int, output_size)
    if x < 0 or y < 0:
        return None
    if x + source_width > slide.width or y + source_height > slide.height:
        return None
    path = getattr(slide, "path", None)
    if not path or not os.path.isfile(path):
        return None
    try:
        import pyvips

        image = pyvips.Image.new_from_file(path, access="sequential")
        image = image.crop(x, y, source_width, source_height)
        if image.bands in (2, 4):
            image = image.flatten(background=[255, 255, 255])
        if image.bands == 1:
            image = image.bandjoin([image, image])
        elif image.bands > 3:
            image = image.extract_band(0, n=3)
        image = image.thumbnail_image(output_width, height=output_height, size="force")
        if image.format != "uchar":
            image = image.cast("uchar")
        array = np.ndarray(
            buffer=image.write_to_memory(),
            dtype=np.uint8,
            shape=(image.height, image.width, image.bands),
        )
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        if array.shape[:2] != (output_height, output_width):
            return None
        return np.ascontiguousarray(array[..., :3])
    except Exception:
        return None


def _read_region_resized_tiled_rgb(
    slide: SlideReader,
    location: tuple[int, int],
    level: int,
    level_size: tuple[int, int],
    output_size: tuple[int, int],
    downsample: float,
    resample: Image.Resampling,
) -> np.ndarray:
    """Bound memory while resizing a huge region without a usable pyramid."""
    (level_width, level_height) = level_size
    (output_width, output_height) = output_size
    output = Image.new("RGB", output_size, "white")
    block_w = max(
        1, min(output_width, _FALLBACK_SOURCE_TILE_SIDE * output_width // level_width)
    )
    block_h = max(
        1,
        min(output_height, _FALLBACK_SOURCE_TILE_SIDE * output_height // level_height),
    )
    (base_x, base_y) = map(int, location)
    for out_y0 in range(0, output_height, block_h):
        out_y1 = min(output_height, out_y0 + block_h)
        src_y0 = round(out_y0 * level_height / output_height)
        src_y1 = round(out_y1 * level_height / output_height)
        src_y1 = max(src_y0 + 1, src_y1)
        for out_x0 in range(0, output_width, block_w):
            out_x1 = min(output_width, out_x0 + block_w)
            src_x0 = round(out_x0 * level_width / output_width)
            src_x1 = round(out_x1 * level_width / output_width)
            src_x1 = max(src_x0 + 1, src_x1)
            patch = _rgb_on_white(
                slide.read_region(
                    (
                        base_x + round(src_x0 * downsample),
                        base_y + round(src_y0 * downsample),
                    ),
                    level,
                    (src_x1 - src_x0, src_y1 - src_y0),
                )
            )
            target_size = (out_x1 - out_x0, out_y1 - out_y0)
            if patch.size != target_size:
                patch = patch.resize(target_size, resample)
            output.paste(patch, (out_x0, out_y0))
    return np.asarray(output, dtype=np.uint8)


def validate_slide_level(slide, level: int) -> tuple[tuple[int, int], float]:
    """Validate a requested pyramid level before allocating feature buffers."""
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ValueError("slide_level must be a non-negative integer")
    if level >= slide.level_count:
        raise ValueError(
            f"Requested slide_level={level}, but backend={slide.backend} provides {slide.level_count} level(s) for {slide.path}"
        )
    downsample = float(slide.level_downsamples[level])
    if not math.isfinite(downsample) or downsample < 1:
        raise ValueError(f"Invalid level downsample: {downsample}")
    return (tuple(slide.level_dimensions[level]), downsample)

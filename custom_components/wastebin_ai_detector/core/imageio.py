"""Image loading and the single geometry pipeline.

:func:`extract_working_roi` is the ONLY entry point that turns a source
image into the working array (EXIF transpose → RGB → ROI crop → resize).
Calibration sampling, threshold learning and detection all go through
it: learned statistics are only valid for images processed identically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import ImageLoadError, RoiError
from .profile import KNOWN_RESAMPLE, REL_EPS as _REL_EPS, Roi

_RESAMPLE_FILTERS = {
    "bilinear": Image.Resampling.BILINEAR,
    "nearest": Image.Resampling.NEAREST,
}
assert set(_RESAMPLE_FILTERS) == set(KNOWN_RESAMPLE)


def load_image_rgb(path: str | Path) -> Image.Image:
    """Open an image file, apply EXIF orientation, return an RGB image.

    EXIF transpose comes first so that relative ROI coordinates always
    refer to the upright image, no matter how the camera stored it.
    """
    path = Path(path)
    try:
        img = Image.open(path)
        img.load()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageLoadError(f"cannot load image {path}: {exc}") from exc
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def load_image_rgb_bytes(data: bytes) -> Image.Image:
    """Same normalization as :func:`load_image_rgb`, for in-memory bytes.

    Camera APIs deliver raw bytes; truncated or empty payloads must
    surface as :class:`ImageLoadError` exactly like broken files do.
    """
    import io

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageLoadError(f"cannot decode image bytes: {exc}") from exc
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def roi_to_pixels(roi: Roi, width: int, height: int) -> tuple[int, int, int, int]:
    """Map a relative ROI to half-open pixel bounds ``(x0, y0, x1, y1)``.

    Deterministic rounding on both edges; raises :class:`RoiError` for
    out-of-range coordinates or a crop that degenerates to zero pixels.
    """
    for name, lo, span in (("x", roi.x, roi.w), ("y", roi.y, roi.h)):
        if lo < -_REL_EPS or span <= 0.0 or lo + span > 1.0 + _REL_EPS:
            raise RoiError(
                f"ROI {name}-range [{lo}, {lo + span}] outside [0, 1] or empty"
            )
    x0 = max(round(roi.x * width), 0)
    x1 = min(round((roi.x + roi.w) * width), width)
    y0 = max(round(roi.y * height), 0)
    y1 = min(round((roi.y + roi.h) * height), height)
    if x1 <= x0 or y1 <= y0:
        raise RoiError(f"ROI {roi} degenerates to zero pixels on {width}x{height}")
    return x0, y0, x1, y1


def extract_working_roi(
    img: Image.Image,
    roi: Roi,
    working_width: int | None,
    resample: str,
) -> np.ndarray:
    """Crop the ROI and resize to the working width.

    Returns a float32 ``(height, width, 3)`` array in [0, 1]. With
    ``working_width=None`` the native crop size is kept (identity - the
    absence of a resize step, not a chosen number).
    """
    if resample not in _RESAMPLE_FILTERS:
        raise RoiError(f"unknown resample {resample!r}; known: {KNOWN_RESAMPLE}")
    x0, y0, x1, y1 = roi_to_pixels(roi, img.width, img.height)
    crop = img.crop((x0, y0, x1, y1))
    if working_width is not None and crop.width != working_width:
        if working_width <= 0:
            raise RoiError(f"working_width must be positive, got {working_width}")
        target_h = round(crop.height * working_width / crop.width)
        if target_h <= 0:
            raise RoiError(
                f"working_width {working_width} degenerates the "
                f"{crop.width}x{crop.height} ROI crop to zero height"
            )
        crop = crop.resize((working_width, target_h), _RESAMPLE_FILTERS[resample])
    return np.asarray(crop, dtype=np.float32) / 255.0


def rect_to_pixels(rect_x: float, rect_y: float, rect_w: float, rect_h: float,
                   width: int, height: int) -> tuple[int, int, int, int]:
    """Map a ROI-relative rectangle onto working-array pixel bounds.

    Same deterministic rounding rules as :func:`roi_to_pixels`, applied
    to the working array - sample pixels are cut from exactly the grid
    the detector will later see.
    """
    if rect_x < -_REL_EPS or rect_w <= 0.0 or rect_x + rect_w > 1.0 + _REL_EPS:
        raise RoiError(f"rect x-range [{rect_x}, {rect_x + rect_w}] outside [0, 1]")
    if rect_y < -_REL_EPS or rect_h <= 0.0 or rect_y + rect_h > 1.0 + _REL_EPS:
        raise RoiError(f"rect y-range [{rect_y}, {rect_y + rect_h}] outside [0, 1]")
    x0 = max(round(rect_x * width), 0)
    x1 = min(round((rect_x + rect_w) * width), width)
    y0 = max(round(rect_y * height), 0)
    y1 = min(round((rect_y + rect_h) * height), height)
    if x1 <= x0 or y1 <= y0:
        raise RoiError(
            f"sample rect ({rect_x}, {rect_y}, {rect_w}, {rect_h}) degenerates "
            f"to zero pixels on the {width}x{height} working image"
        )
    return x0, y0, x1, y1

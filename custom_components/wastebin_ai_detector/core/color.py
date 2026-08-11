"""Color math: vectorized float RGB→HSV and circular hue statistics.

PIL's built-in ``convert("HSV")`` is 8-bit quantized (≈1.41°/step) and
assigns hue 0 (red) to grey pixels. Both properties poison learned
color statistics, so this module implements the standard float
conversion itself and marks undefined hue as NaN.
"""

from __future__ import annotations

import numpy as np

from .errors import CalibrationError

# Guard against division by zero in the HSV formulas. Purely a
# floating-point epsilon: far below one 8-bit quantum (1/255 ≈ 3.9e-3),
# so it can never reclassify a real color — it only protects 0/0.
_EPS = 1e-12


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an (..., 3) float array in [0, 1] to HSV.

    Returns ``(hue_deg, sat, val)`` where ``hue_deg`` is in [0, 360) and
    NaN wherever chroma is 0: hue is mathematically undefined for grey
    pixels, and they must not enter circular hue statistics.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim < 1 or rgb.shape[-1] != 3:
        raise ValueError(f"expected (..., 3) RGB array, got shape {rgb.shape}")
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    val = maxc
    chroma = maxc - minc
    sat = np.where(maxc > _EPS, chroma / np.where(maxc > _EPS, maxc, 1.0), 0.0)
    safe_chroma = np.where(chroma > _EPS, chroma, 1.0)
    h_r = ((g - b) / safe_chroma) % 6.0
    h_g = (b - r) / safe_chroma + 2.0
    h_b = (r - g) / safe_chroma + 4.0
    hue = np.select([maxc == r, maxc == g], [h_r, h_g], default=h_b) * 60.0
    hue = np.where(chroma > _EPS, hue % 360.0, np.nan)
    return hue, sat, val


def circular_mean_deg(hues_deg: np.ndarray) -> tuple[float, float]:
    """Circular mean of hue angles in degrees, ignoring NaN entries.

    Returns ``(mean_deg, resultant_length)``. The resultant length R in
    [0, 1] measures concentration: R→1 means tightly clustered hues,
    R→0 means the angles cancel out (no meaningful mean direction).
    Raises :class:`CalibrationError` if no valid (non-NaN) hue exists.
    """
    hues = np.asarray(hues_deg, dtype=np.float64).ravel()
    valid = hues[~np.isnan(hues)]
    if valid.size == 0:
        raise CalibrationError(
            "no valid hue pixels in sample (all grey/unsaturated) — the "
            "sample rectangle must cover a colored surface. Grey/black "
            "lids cannot be color-calibrated; attach a small colored "
            "marker to the lid and sample that instead"
        )
    rad = np.deg2rad(valid)
    x = float(np.cos(rad).mean())
    y = float(np.sin(rad).mean())
    resultant = float(np.hypot(x, y))
    mean = float(np.rad2deg(np.arctan2(y, x))) % 360.0
    # Float-modulo edge: a tiny negative angle (hues symmetric around
    # the 0°/360° red wraparound) yields 360 − tiny, which rounds UP to
    # exactly 360.0 because ulp(360) ≈ 5.7e-14 exceeds the remainder.
    # Fold it back onto the documented [0, 360) interval.
    if mean >= 360.0:
        mean = 0.0
    return mean, resultant


def circular_dist_deg(hues_deg: np.ndarray, center_deg: float) -> np.ndarray:
    """Absolute circular distance in degrees, in [0, 180]; NaN propagates.

    The ``((h − c + 180) mod 360) − 180`` form handles the 0°/360° red
    wraparound without special cases.
    """
    hues = np.asarray(hues_deg, dtype=np.float64)
    return np.abs(((hues - center_deg + 180.0) % 360.0) - 180.0)

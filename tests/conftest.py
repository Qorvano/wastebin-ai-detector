"""Shared test fixtures: deterministic synthetic scenes.

Scenes are grey noisy backgrounds with colored rectangles ("lids").
All numbers in here are test-fixture data (the scene being drawn), not
detector tuning - the detector learns its thresholds from these scenes
through the real calibration code path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "custom_components"))
sys.path.insert(0, str(_REPO_ROOT / "tools"))

# Saturated, well-separated lid colors (hues ≈ 49°, 223°, 25°, 358°).
YELLOW = (0.95, 0.80, 0.10)
BLUE = (0.15, 0.35, 0.85)
BROWN = (0.55, 0.30, 0.12)
RED = (0.90, 0.10, 0.12)  # hue ≈ 358.5° - exercises the 0°/360° wraparound


def make_scene(
    size: tuple[int, int] = (320, 200),
    rects: list[tuple[tuple[float, float, float], float, float, float, float]] = (),
    bg_grey: float = 0.5,
    noise: float = 0.02,
    seed: int = 0,
) -> Image.Image:
    """Render a synthetic scene: grey noise + colored rectangles.

    ``rects`` entries are ``(rgb, x, y, w, h)`` with image-relative
    coordinates in 0..1.
    """
    rng = np.random.default_rng(seed)
    width, height = size
    base = np.full((height, width, 3), bg_grey)
    for rgb, rx, ry, rw, rh in rects:
        x0, x1 = round(rx * width), round((rx + rw) * width)
        y0, y1 = round(ry * height), round((ry + rh) * height)
        base[y0:y1, x0:x1] = rgb
    # Noise goes over the WHOLE image, lids included: real camera pixels
    # always vary, and learned percentile floors only leave downward
    # slack if the calibration distribution has spread. Uniform lids
    # would make the floors knife-edge exact and any JPEG re-encode
    # would push every lid pixel below them.
    arr = np.clip(base + rng.normal(0.0, noise, base.shape), 0.0, 1.0)
    return Image.fromarray((arr * 255.0).round().astype(np.uint8), "RGB")


def inner_rect(
    rect: tuple[float, float, float, float], margin_frac: float = 0.25
) -> tuple[float, float, float, float]:
    """Shrink a rectangle so a sample stays fully inside the drawn lid."""
    x, y, w, h = rect
    return (
        x + w * margin_frac,
        y + h * margin_frac,
        w * (1.0 - 2.0 * margin_frac),
        h * (1.0 - 2.0 * margin_frac),
    )

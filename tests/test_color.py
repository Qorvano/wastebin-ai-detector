"""Color math against the colorsys reference plus circular statistics."""

from __future__ import annotations

import colorsys

import numpy as np
import pytest

from wastebin_ai_detector.core import (
    CalibrationError,
    circular_dist_deg,
    circular_mean_deg,
    rgb_to_hsv,
)


def test_hsv_matches_colorsys_reference():
    rng = np.random.default_rng(42)
    rgb = rng.uniform(0.0, 1.0, (400, 3))
    hue, sat, val = rgb_to_hsv(rgb)
    for i, (r, g, b) in enumerate(rgb):
        ref_h, ref_s, ref_v = colorsys.rgb_to_hsv(r, g, b)
        assert sat[i] == pytest.approx(ref_s, abs=1e-9)
        assert val[i] == pytest.approx(ref_v, abs=1e-9)
        if max(r, g, b) - min(r, g, b) > 1e-6:
            diff = abs(((hue[i] - ref_h * 360.0 + 180.0) % 360.0) - 180.0)
            assert diff < 1e-6


def test_grey_pixels_have_nan_hue_and_zero_sat():
    grey = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    hue, sat, val = rgb_to_hsv(grey)
    assert np.all(np.isnan(hue))
    assert np.all(sat == 0.0)
    assert val == pytest.approx([0.0, 0.5, 1.0])


def test_rgb_shape_rejected():
    with pytest.raises(ValueError):
        rgb_to_hsv(np.zeros((4, 4)))


def test_circular_mean_red_wraparound():
    mean, resultant = circular_mean_deg(np.array([358.0, 359.0, 1.0, 2.0]))
    assert min(mean, 360.0 - mean) == pytest.approx(0.0, abs=1e-9)
    assert resultant > 0.99


def test_circular_mean_never_returns_360():
    # Symmetric hue pairs around 0°/360° cancel to a tiny negative
    # angle; float modulo rounds 360 − tiny up to exactly 360.0, which
    # must be folded back into the documented [0, 360) interval.
    for hues in ([350.0, 10.0], [359.0, 1.0], [355.0, 5.0, 0.0]):
        mean, _ = circular_mean_deg(np.array(hues))
        assert 0.0 <= mean < 360.0
        assert mean == pytest.approx(0.0, abs=1e-9)


def test_circular_mean_ignores_nan():
    mean, _ = circular_mean_deg(np.array([90.0, np.nan, 90.0]))
    assert mean == pytest.approx(90.0)


def test_circular_mean_all_nan_raises():
    with pytest.raises(CalibrationError):
        circular_mean_deg(np.array([np.nan, np.nan]))


def test_circular_dist():
    assert circular_dist_deg(np.array([350.0]), 10.0)[0] == pytest.approx(20.0)
    assert circular_dist_deg(np.array([10.0]), 350.0)[0] == pytest.approx(20.0)
    assert circular_dist_deg(np.array([0.0]), 180.0)[0] == pytest.approx(180.0)
    assert np.isnan(circular_dist_deg(np.array([np.nan]), 10.0)[0])

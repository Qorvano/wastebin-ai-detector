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


def test_weighted_circular_mean():
    from wastebin_ai_detector.core import circular_mean_deg as cm

    hues = np.array([10.0, 200.0])
    mean, _ = cm(hues, weights=np.array([1.0, 0.0]))
    assert mean == pytest.approx(10.0)


def test_kappa_from_resultant_monotone():
    from wastebin_ai_detector.core import vonmises_kappa_from_resultant as k

    assert k(0.0) == 0.0
    assert 0.0 < k(0.2) < k(0.5) < k(0.9)
    assert k(1.0) == float("inf")


def test_mixture_single_pixel_is_trivially_coherent():
    from wastebin_ai_detector.core import fit_vonmises_uniform_mixture

    fit = fit_vonmises_uniform_mixture(
        np.array([120.0]),
        init_center_deg=0.0,
        init_weight=0.5,
        init_kappa=1.0,
        kappa_max=100.0,
    )
    assert fit.center_deg == pytest.approx(120.0)
    assert fit.weight == 1.0
    assert fit.posterior[0] == 1.0


def test_mixture_iterations_bounded_by_n():
    from wastebin_ai_detector.core import fit_vonmises_uniform_mixture

    rng = np.random.default_rng(31)
    hues = rng.uniform(0.0, 360.0, 300)
    # Adversarial start (grossly data-inconsistent weight) must still
    # terminate within n iterations.
    fit = fit_vonmises_uniform_mixture(
        hues,
        init_center_deg=10.0,
        init_weight=0.99,
        init_kappa=5.0,
        kappa_max=1e4,
    )
    assert fit.n_iter <= 300


def test_mixture_separates_coherent_from_uniform():
    from wastebin_ai_detector.core import fit_vonmises_uniform_mixture

    rng = np.random.default_rng(21)
    coherent = rng.normal(120.0, 5.0, 800) % 360.0
    junk = rng.uniform(0.0, 360.0, 200)
    hues = np.concatenate([coherent, junk])
    fit = fit_vonmises_uniform_mixture(
        hues,
        init_center_deg=120.0,
        init_weight=0.5,
        init_kappa=10.0,
        kappa_max=1e6,
    )
    assert fit.center_deg == pytest.approx(120.0, abs=2.0)
    assert 0.7 < fit.weight < 0.95
    # The known-coherent pixels overwhelmingly classify as coherent.
    assert (fit.posterior[:800] >= 0.5).mean() > 0.95

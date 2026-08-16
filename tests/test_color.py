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


def test_weighted_percentile_midpoint_convention():
    from wastebin_ai_detector.core.color import weighted_percentile

    values = np.array([0.0, 10.0])
    ones = np.ones_like(values)
    # Uniform two-point median interpolates to the midpoint; edge
    # queries clamp to the extreme values.
    assert weighted_percentile(values, ones, 50.0) == pytest.approx(5.0)
    assert weighted_percentile(values, ones, 0.0) == pytest.approx(0.0)
    assert weighted_percentile(values, ones, 100.0) == pytest.approx(10.0)
    # An observation holding almost all mass IS the median.
    heavy = np.array([1.0, 98.0, 1.0])
    assert weighted_percentile(
        np.array([0.0, 10.0, 20.0]), heavy, 50.0
    ) == pytest.approx(10.0)


def test_weighted_percentile_mass_shifts_result():
    from wastebin_ai_detector.core.color import weighted_percentile

    values = np.array([0.0, 10.0])
    # Equal mass: the median interpolates to the midpoint (type 7).
    assert weighted_percentile(values, np.array([1.0, 1.0]), 50.0) == (
        pytest.approx(5.0)
    )
    # Nearly all mass on the right observation pulls the median there
    # (midpoint positions 0.005 and 0.505: q=0.5 sits just inside).
    assert weighted_percentile(values, np.array([1.0, 99.0]), 50.0) == (
        pytest.approx(9.9, abs=0.01)
    )


def test_learn_color_model_one_image_one_vote():
    """The field regression behind per-image weighting: one huge
    washed-out sample rectangle must not outvote two small clean ones
    from other images. Unweighted, the pixel majority wins (the old
    failure mode); with 1/(image pixels) weights, the image majority
    wins and the big rectangle is classified as junk."""
    from wastebin_ai_detector.core.learn import learn_color_model

    big = np.linspace(19.5, 20.5, 1000)  # one huge washed-out patch
    small_a = np.linspace(198.0, 202.0, 100)  # two clean lid samples
    small_b = np.linspace(199.0, 201.0, 100)
    hue = np.concatenate([big, small_a, small_b])
    sat = np.full(hue.size, 0.6)
    val = np.full(hue.size, 0.7)
    weights = np.concatenate(
        [
            np.full(big.size, 1.0 / big.size),
            np.full(small_a.size, 1.0 / small_a.size),
            np.full(small_b.size, 1.0 / small_b.size),
        ]
    )
    unweighted = learn_color_model(hue, sat, val, bin_id="braune_tonne")
    weighted = learn_color_model(
        hue, sat, val, bin_id="braune_tonne", weights=weights
    )
    assert abs(unweighted.hue_center_deg - 20.0) < 5.0
    assert abs(weighted.hue_center_deg - 200.0) < 5.0
    # The big rectangle is one of three image votes: junk mass 1/3.
    assert weighted.stats["junk_fraction"] == pytest.approx(1.0 / 3.0, abs=0.05)

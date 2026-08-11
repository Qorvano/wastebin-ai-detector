"""Unit tests for the learning functions and their edge cases."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wastebin_ai_detector.core import CalibrationError, learn_area_threshold, learn_color_model


def _pixels(hue_deg: float, n: int = 500, sat: float = 0.8, val: float = 0.7):
    hue = np.full(n, hue_deg)
    return hue, np.full(n, sat), np.full(n, val)


class TestAreaThreshold:
    def test_separable_geometric_mean(self):
        result = learn_area_threshold([0.03, 0.05], [0.01], bin_id="t")
        assert result.min_area_frac == pytest.approx(math.sqrt(0.03 * 0.01))
        assert result.stats["separable"] is True
        assert result.stats["provisional"] is False
        assert result.warnings == []

    def test_no_negatives_is_provisional(self):
        result = learn_area_threshold([0.04], [], bin_id="t")
        assert result.min_area_frac == pytest.approx(0.02)
        assert result.stats["provisional"] is True
        # No observed negative: the stat must be None, not a claimed
        # 0.0, or the ambiguity interval would swallow (0, min_pos).
        assert result.stats["max_neg_area_frac"] is None
        assert any("provisional" in w for w in result.warnings)

    def test_zero_negatives_not_provisional(self):
        # Negatives exist and show zero color response - best possible case.
        result = learn_area_threshold([0.04], [0.0, 0.0], bin_id="t")
        assert result.min_area_frac == pytest.approx(0.02)
        assert result.stats["provisional"] is False
        assert result.stats["separable"] is True
        assert result.warnings == []

    def test_overlap_warns_not_separable(self):
        result = learn_area_threshold([0.02], [0.03], bin_id="t")
        assert result.stats["separable"] is False
        assert any("NOT separable" in w for w in result.warnings)

    def test_zero_area_positive_raises(self):
        with pytest.raises(CalibrationError):
            learn_area_threshold([0.0, 0.02], [], bin_id="t")

    def test_no_positives_raises(self):
        with pytest.raises(CalibrationError):
            learn_area_threshold([], [0.01], bin_id="t")


class TestColorModel:
    def test_consistent_color_learns_tight_band(self):
        rng = np.random.default_rng(7)
        hue = 220.0 + rng.normal(0.0, 3.0, 2000)
        result = learn_color_model(
            hue % 360.0, np.full(2000, 0.8), np.full(2000, 0.6), bin_id="t"
        )
        assert result.hue_center_deg == pytest.approx(220.0, abs=1.0)
        assert 0.0 < result.hue_tol_deg < 15.0
        assert result.stats["resultant_r"] > 0.99

    def test_uniform_hue_gets_nonzero_tolerance(self):
        hue, sat, val = _pixels(49.0, sat=0.8, val=0.7)
        result = learn_color_model(hue, sat, val, bin_id="t")
        # Data-derived floor: half the finest 8-bit hue quantum of the
        # samples, 60 / (255 · chroma) / 2 with chroma = sat · val.
        assert result.hue_tol_deg == pytest.approx(
            60.0 / (255.0 * 0.8 * 0.7) / 2.0
        )

    def test_red_wraparound_model_is_profile_valid(self):
        # Hues symmetric around 0°/360° must not learn a center of
        # exactly 360.0 (which profile validation rejects).
        hue = np.array([0.01, 359.99] * 100, dtype=float)
        result = learn_color_model(
            hue, np.full(hue.size, 0.9), np.full(hue.size, 0.8), bin_id="t"
        )
        assert 0.0 <= result.hue_center_deg < 360.0

    def test_wraparound_center(self):
        hue = np.array([357.0, 359.0, 1.0, 3.0] * 100, dtype=float)
        result = learn_color_model(
            hue, np.full(hue.size, 0.8), np.full(hue.size, 0.7), bin_id="t"
        )
        center = min(result.hue_center_deg, 360.0 - result.hue_center_deg)
        assert center == pytest.approx(0.0, abs=1e-6)
        assert result.hue_tol_deg < 10.0

    def test_opposing_hues_degenerate(self):
        hue = np.array([0.0] * 100 + [180.0] * 100, dtype=float)
        with pytest.raises(CalibrationError):
            learn_color_model(
                hue, np.full(200, 0.8), np.full(200, 0.7), bin_id="t"
            )

    def test_all_grey_raises(self):
        hue = np.full(100, np.nan)
        with pytest.raises(CalibrationError):
            learn_color_model(hue, np.zeros(100), np.full(100, 0.5), bin_id="t")

    def test_small_sample_warns(self):
        hue, sat, val = _pixels(49.0, n=10)
        result = learn_color_model(hue, sat, val, bin_id="t")
        assert any("sample pixels" in w for w in result.warnings)

    def test_empty_sample_raises(self):
        with pytest.raises(CalibrationError):
            learn_color_model(np.array([]), np.array([]), np.array([]), bin_id="t")

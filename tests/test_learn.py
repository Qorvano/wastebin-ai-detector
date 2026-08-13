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


class TestJunkRobustness:
    """Field regressions from the 2026-08-11 midday calibration failures."""

    @staticmethod
    def _mixed_sample(n_coherent, n_junk, seed):
        rng = np.random.default_rng(seed)
        hue = np.concatenate(
            [
                rng.normal(50.0, 3.0, n_coherent) % 360.0,
                rng.uniform(0.0, 360.0, n_junk),
            ]
        )
        # Junk pixels: overexposed (low saturation, high value).
        sat = np.concatenate(
            [rng.uniform(0.5, 0.7, n_coherent), rng.uniform(0.02, 0.15, n_junk)]
        )
        val = np.concatenate(
            [rng.uniform(0.5, 0.8, n_coherent), rng.uniform(0.85, 1.0, n_junk)]
        )
        return hue, sat, val

    def test_majority_junk_still_fails(self):
        # Attempt 1 in the field: ~25% coherent yellow, R was 0.232.
        hue, sat, val = self._mixed_sample(500, 1500, seed=11)
        with pytest.raises(CalibrationError):
            learn_color_model(hue, sat, val, bin_id="gelb")

    def test_recoverable_junk_now_succeeds(self):
        # Attempt 2 in the field: ~60% coherent yellow, R was 0.605 and
        # the old percentile learner failed; the mixture must recover.
        hue, sat, val = self._mixed_sample(1200, 800, seed=12)
        result = learn_color_model(hue, sat, val, bin_id="gelb")
        assert result.hue_center_deg == pytest.approx(50.0, abs=3.0)
        assert result.hue_tol_deg < 30.0
        assert 0.25 < result.stats["junk_fraction"] < 0.5
        # Floors decontaminated: junk saturation (max 0.15) must not
        # drag sat_min below the coherent population.
        assert result.sat_min > 0.4
        assert any("junk" in w for w in result.warnings)

    def test_two_colors_dominant_mode_wins(self):
        # 70/30 split of two tight colors: dominant is learned, the
        # minority is reported as junk share.
        rng = np.random.default_rng(13)
        hue = np.concatenate(
            [
                rng.normal(50.0, 3.0, 700) % 360.0,
                rng.normal(200.0, 3.0, 300) % 360.0,
            ]
        )
        sat = np.full(1000, 0.7)
        val = np.full(1000, 0.6)
        result = learn_color_model(hue, sat, val, bin_id="t")
        assert result.hue_center_deg == pytest.approx(50.0, abs=3.0)
        assert result.hue_tol_deg < 30.0
        assert 0.2 < result.stats["junk_fraction"] < 0.4

    def test_pure_uniform_junk_fails(self):
        rng = np.random.default_rng(14)
        hue = rng.uniform(0.0, 360.0, 2000)
        sat = rng.uniform(0.3, 0.8, 2000)
        val = rng.uniform(0.3, 0.8, 2000)
        with pytest.raises(CalibrationError):
            learn_color_model(hue, sat, val, bin_id="t")


class TestQualityGates:
    def test_empty_returns_none(self):
        from wastebin_ai_detector.core import derive_quality_gates

        assert derive_quality_gates([]) is None

    def test_single_sample_has_zero_slack(self):
        from wastebin_ai_detector.core import derive_quality_gates

        gates = derive_quality_gates([[0.3, 0.6, 0.02, 0.01]])
        assert gates == {
            "daylight_sat_min": 0.3,
            "daylight_val_max": 0.6,
            "overexposure_clip_max": 0.02,
            "row_dup_max": 0.01,
        }

    def test_slack_extends_extrema(self):
        from wastebin_ai_detector.core import derive_quality_gates

        samples = [
            [0.30, 0.60, 0.020, 0.00],
            [0.32, 0.62, 0.025, 0.01],
            [0.28, 0.64, 0.030, 0.02],
        ]
        gates = derive_quality_gates(samples)
        # sat diffs: 0.02, 0.04 -> slack 0.03; min 0.28 - 0.03 = 0.25
        assert gates["daylight_sat_min"] == pytest.approx(0.25)
        # val diffs: 0.02, 0.02 -> slack 0.02; max 0.64 + 0.02 = 0.66
        assert gates["daylight_val_max"] == pytest.approx(0.66)
        # clip diffs: 0.005, 0.005 -> slack 0.005; max 0.03 + 0.005
        assert gates["overexposure_clip_max"] == pytest.approx(0.035)
        # row-dup diffs: 0.01, 0.01 -> slack 0.01; max 0.02 + 0.01
        assert gates["row_dup_max"] == pytest.approx(0.03)

    def test_bounds_clamped(self):
        from wastebin_ai_detector.core import derive_quality_gates

        gates = derive_quality_gates(
            [[0.001, 0.999, 0.999, 0.999], [0.05, 0.5, 0.5, 0.5]]
        )
        assert gates["daylight_sat_min"] >= 0.0
        assert gates["daylight_val_max"] <= 1.0
        assert gates["overexposure_clip_max"] <= 1.0
        assert gates["row_dup_max"] <= 1.0

    def test_bad_shape_rejected(self):
        from wastebin_ai_detector.core import derive_quality_gates

        with pytest.raises(CalibrationError):
            derive_quality_gates([[0.1, 0.2]])

"""Detection result invariants and the learned ambiguity interval."""

from __future__ import annotations

from wastebin_ai_detector.core import BinResult, is_uncertain


def test_to_dict_never_contradicts_present():
    # area_frac just below the threshold: present is False and the
    # serialized values must show it (no display rounding that would
    # claim margin == 1.0).
    result = BinResult(
        id="t",
        name="T",
        present=False,
        area_frac=0.0009996,
        min_area_frac=0.001,
        margin=0.0009996 / 0.001,
    )
    data = result.to_dict()
    assert data["present"] is False
    assert data["margin"] < 1.0
    assert data["area_frac"] < data["min_area_frac"]
    assert (data["margin"] >= 1.0) == data["present"]


class TestIsUncertain:
    STATS = {"min_pos_area_frac": 0.03, "max_neg_area_frac": 0.01}

    def test_inside_interval(self):
        assert is_uncertain(0.02, self.STATS) is True

    def test_outside_interval(self):
        assert is_uncertain(0.035, self.STATS) is False
        assert is_uncertain(0.005, self.STATS) is False

    def test_bounds_are_exclusive(self):
        # A frame exactly at an observed extreme matches calibration
        # evidence and is not ambiguous.
        assert is_uncertain(0.03, self.STATS) is False
        assert is_uncertain(0.01, self.STATS) is False

    def test_provisional_no_negatives(self):
        stats = {"min_pos_area_frac": 0.03, "max_neg_area_frac": 0.0}
        assert is_uncertain(0.015, stats) is True
        assert is_uncertain(0.0, stats) is False

    def test_inverted_non_separable(self):
        stats = {"min_pos_area_frac": 0.01, "max_neg_area_frac": 0.03}
        assert is_uncertain(0.02, stats) is True
        assert is_uncertain(0.04, stats) is False

    def test_missing_stats(self):
        assert is_uncertain(0.02, {}) is False

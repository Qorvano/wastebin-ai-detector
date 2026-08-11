"""Detection result serialization invariants."""

from __future__ import annotations

from wastebin_ai_detector.core import BinResult


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

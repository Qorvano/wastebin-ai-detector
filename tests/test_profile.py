"""Profile and store (de)serialization, validation, versioning."""

from __future__ import annotations

import pytest

from wastebin_ai_detector.core import (
    BinDecl,
    BinModel,
    CalibrationError,
    CalibrationStore,
    Profile,
    ProfileError,
    Rect,
    Roi,
    load_profile,
    load_store,
    profile_from_dict,
    profile_to_dict,
    save_profile,
    save_store,
    validate_store,
)


def _valid_profile() -> Profile:
    return Profile(
        roi=Roi(0.2, 0.3, 0.5, 0.4),
        working_width=480,
        resample="bilinear",
        daylight_sat_min=0.15,
        bins=[
            BinModel("gelb", "Gelbe Tonne", 52.0, 12.0, 0.4, 0.2, 0.01),
            BinModel("blau", "Blaue Tonne", 220.0, 10.0, 0.35, 0.15, 0.02),
        ],
    )


def test_profile_roundtrip(tmp_path):
    path = tmp_path / "profile.json"
    original = _valid_profile()
    save_profile(original, path)
    loaded = load_profile(path)
    assert loaded == original


def test_unsupported_version_rejected():
    data = profile_to_dict(_valid_profile())
    data["schema_version"] = 2
    with pytest.raises(ProfileError):
        profile_from_dict(data)


def test_duplicate_bin_ids_rejected():
    data = profile_to_dict(_valid_profile())
    data["bins"][1]["id"] = "gelb"
    with pytest.raises(ProfileError):
        profile_from_dict(data)


def test_degenerate_hue_tol_rejected():
    data = profile_to_dict(_valid_profile())
    data["bins"][0]["hue_tol_deg"] = 90.0
    with pytest.raises(ProfileError):
        profile_from_dict(data)


def test_bad_area_frac_rejected():
    data = profile_to_dict(_valid_profile())
    data["bins"][0]["min_area_frac"] = 0.0
    with pytest.raises(ProfileError):
        profile_from_dict(data)


def test_bad_learning_stats_rejected():
    for bad in ("kaputt", 1.5, True):
        data = profile_to_dict(_valid_profile())
        data["bins"][0]["learning_stats"] = {"min_pos_area_frac": bad}
        with pytest.raises(ProfileError):
            profile_from_dict(data)


def test_missing_field_rejected():
    data = profile_to_dict(_valid_profile())
    del data["roi"]
    with pytest.raises(ProfileError):
        profile_from_dict(data)


def _valid_store() -> CalibrationStore:
    return CalibrationStore(
        roi=Roi(0.2, 0.3, 0.5, 0.4),
        working_width=480,
        resample="bilinear",
        bins=[BinDecl("gelb", "Gelbe Tonne"), BinDecl("blau", "Blaue Tonne")],
    )


def test_store_roundtrip(tmp_path):
    path = tmp_path / "store.json"
    store = _valid_store()
    store.add_sample("img1.png", "gelb", Rect(0.1, 0.1, 0.2, 0.2))
    store.set_labels("img1.png", present=["gelb"], absent=["blau"])
    save_store(store, path)
    loaded = load_store(path)
    assert loaded == store


def test_store_unknown_bin_sample_rejected():
    with pytest.raises(CalibrationError):
        _valid_store().add_sample("x.png", "nope", Rect(0.1, 0.1, 0.1, 0.1))


def test_store_conflicting_labels_rejected():
    store = _valid_store()
    with pytest.raises(CalibrationError):
        store.set_labels("x.png", present=["gelb"], absent=["gelb"])


def test_store_forget_image():
    store = _valid_store()
    store.add_sample("img1.png", "gelb", Rect(0.1, 0.1, 0.2, 0.2))
    store.set_labels("img1.png", present=["gelb"])
    assert store.forget_image("img1.png") is True
    assert store.get_image("img1.png") is None
    assert store.forget_image("img1.png") is False


def test_store_relabel_moves_bin():
    store = _valid_store()
    store.set_labels("x.png", present=["gelb"])
    store.set_labels("x.png", absent=["gelb"])
    entry = store.get_image("x.png")
    assert entry.present == [] and entry.absent == ["gelb"]
    validate_store(store)


def test_image_rect_conversion():
    store = _valid_store()
    rect = store.image_rect_to_roi_rect(Rect(0.2, 0.3, 0.25, 0.2))
    assert rect.x == pytest.approx(0.0)
    assert rect.y == pytest.approx(0.0)
    assert rect.w == pytest.approx(0.5)
    assert rect.h == pytest.approx(0.5)
    with pytest.raises(CalibrationError):
        store.image_rect_to_roi_rect(Rect(0.0, 0.0, 0.1, 0.1))  # outside ROI


def test_image_rect_flush_with_roi_edge_accepted():
    # FP drift: (0.02−0.01)/0.15 + 0.14/0.15 == 1.0000000000000002,
    # although the rect ends exactly on the ROI edge. Must be accepted
    # (same REL_EPS policy as the pixel mapping) and clamped to [0, 1].
    store = CalibrationStore(
        roi=Roi(0.01, 0.1, 0.15, 0.5),
        working_width=None,
        resample="bilinear",
        bins=[BinDecl("gelb", "Gelbe Tonne")],
    )
    rect = store.image_rect_to_roi_rect(Rect(0.02, 0.2, 0.14, 0.3))
    assert 0.0 <= rect.x and rect.x + rect.w <= 1.0
    assert 0.0 <= rect.y and rect.y + rect.h <= 1.0
    assert rect.x + rect.w == pytest.approx(1.0)


def _store_dict_with(entry_overrides: dict) -> dict:
    from wastebin_ai_detector.core import store_to_dict

    store = _valid_store()
    data = store_to_dict(store)
    data["images"] = [
        {"path": "img.png", "samples": {}, "present": [], "absent": []}
        | entry_overrides
    ]
    return data


def test_store_rejects_undeclared_bin_references():
    from wastebin_ai_detector.core import store_from_dict

    for overrides in (
        {"samples": {"yelow": [{"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}]}},
        {"present": ["yellwo"]},
        {"absent": ["lila"]},
    ):
        with pytest.raises(ProfileError):
            store_from_dict(_store_dict_with(overrides))


def test_store_rejects_bad_setup_values():
    from wastebin_ai_detector.core import store_from_dict, store_to_dict

    data = store_to_dict(_valid_store())
    data["working_width"] = 0
    with pytest.raises(ProfileError):
        store_from_dict(data)

    data = store_to_dict(_valid_store())
    data["roi"]["w"] = 0.9  # x + w = 1.1 > 1
    with pytest.raises(ProfileError):
        store_from_dict(data)

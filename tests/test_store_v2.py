"""Schema v2: migration, learning view, epochs, dynamic reconfiguration.

Every rule here is exercised against MULTIPLE geometries (grown,
shrunken, shifted, mirrored-asymmetric ROIs) - a rule that only holds
for one configuration is a bug by project convention.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from scenes import BLUE, BROWN, YELLOW, inner_rect, make_scene
from wastebin_ai_detector.core import (
    BinDecl,
    CalibrationError,
    CalibrationStore,
    ImageEntry,
    ProfileError,
    Rect,
    Roi,
    SampleRect,
    learn_profile,
    learning_view,
    roi_rect_to_image_rect,
    store_from_dict,
    store_to_dict,
    validate_store,
)

FIELD_ROI = {"x": 0.24, "y": 0.18, "w": 0.42, "h": 0.78}


def _v1_store_dict() -> dict:
    """A v1 store dict shaped like the real field store."""
    return {
        "schema_version": 1,
        "roi": dict(FIELD_ROI),
        "working_width": 640,
        "resample": "bilinear",
        "bins": [
            {"id": "gelbe_tonne", "name": "Gelbe Tonne"},
            {"id": "blaue_tonne", "name": "Blaue Tonne"},
        ],
        "images": [
            {
                "path": "a.jpg",
                "samples": {
                    "gelbe_tonne": [
                        {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1},
                        # Flush with the ROI edge: x + w == 1.0 exactly.
                        {"x": 0.9, "y": 0.0, "w": 0.1, "h": 0.2},
                    ]
                },
                "present": ["gelbe_tonne"],
                "absent": ["blaue_tonne"],
            },
            {"path": "b.jpg", "samples": {}, "present": [], "absent": []},
        ],
    }


class TestMigrationV1V2:
    def test_rects_move_to_exact_image_space(self):
        store = store_from_dict(_v1_store_dict())
        roi = Roi(**FIELD_ROI)
        sample = store.get_image("a.jpg").samples["gelbe_tonne"][0]
        # Affine: x_img = roi.x + 0.5 * roi.w, etc.
        assert sample.rect.x == pytest.approx(0.24 + 0.5 * 0.42, abs=1e-12)
        assert sample.rect.y == pytest.approx(0.18 + 0.5 * 0.78, abs=1e-12)
        assert sample.rect.w == pytest.approx(0.5 * 0.42 * 0.2, abs=1e-12)
        assert sample.roi == roi
        assert sample.epoch == 0

    def test_edge_flush_rect_survives_and_validates(self):
        store = store_from_dict(_v1_store_dict())
        edge = store.get_image("a.jpg").samples["gelbe_tonne"][1]
        assert edge.rect.x + edge.rect.w <= 1.0
        validate_store(store)

    def test_labels_get_roi_and_epoch_stamps(self):
        store = store_from_dict(_v1_store_dict())
        entry = store.get_image("a.jpg")
        assert entry.label_roi == Roi(**FIELD_ROI)
        assert entry.label_epoch == {"gelbe_tonne": 0, "blaue_tonne": 0}
        assert store.get_image("b.jpg").label_roi is None

    def test_migration_is_idempotent(self):
        once = store_from_dict(_v1_store_dict())
        twice = store_from_dict(store_to_dict(once))
        assert once == twice

    def test_v2_roundtrip_preserves_everything(self):
        store = store_from_dict(_v1_store_dict())
        store.forget_image("b.jpg")
        store.mark_bin_appearance_changed("gelbe_tonne")
        loaded = store_from_dict(store_to_dict(store))
        assert loaded == store

    def test_unknown_version_still_rejected(self):
        bad = _v1_store_dict() | {"schema_version": 3}
        with pytest.raises(ProfileError):
            store_from_dict(bad)


def _store(roi: Roi, view_epoch: int = 0) -> CalibrationStore:
    return CalibrationStore(
        roi=roi,
        working_width=320,
        resample="bilinear",
        bins=[BinDecl("gelb", "Gelbe Tonne"), BinDecl("blau", "Blaue Tonne")],
        view_epoch=view_epoch,
    )


# Several base geometries: the rules must hold for every one of them.
GEOMETRIES = [
    Roi(0.24, 0.18, 0.42, 0.78),
    Roi(0.0, 0.0, 1.0, 1.0),
    Roi(0.5, 0.05, 0.45, 0.4),
]


class TestLearningViewContainment:
    @pytest.mark.parametrize(
        "base", [g for g in GEOMETRIES if g != Roi(0.0, 0.0, 1.0, 1.0)]
    )
    def test_growing_roi_keeps_present_drops_absent(self, base):
        store = _store(base)
        store.add_sample("img.jpg", "gelb", Rect(base.x, base.y, 0.05, 0.05))
        store.set_labels("img.jpg", present=["gelb"], absent=["blau"])
        # Grow to the full frame: strictly contains every base ROI.
        store.roi = Roi(0.0, 0.0, 1.0, 1.0)
        view, warnings = learning_view(store)
        entry = view.get_image("img.jpg")
        assert entry.present == ["gelb"]
        # The newly covered strip may contain the blue bin: absent
        # claims must not extend beyond the labeled crop.
        assert entry.absent == []
        assert any("absent labels set aside" in w for w in warnings)

    @pytest.mark.parametrize("base", GEOMETRIES)
    def test_shrinking_roi_keeps_absent_gates_present_by_rects(self, base):
        store = _store(base)
        # Sample rect in the upper-left corner of the base ROI.
        inner = Rect(
            base.x + 0.02 * base.w,
            base.y + 0.02 * base.h,
            0.1 * base.w,
            0.1 * base.h,
        )
        store.add_sample("img.jpg", "gelb", inner)
        store.set_labels("img.jpg", present=["gelb"], absent=["blau"])
        # Shrink to the upper-left quadrant: still contains the rect.
        store.roi = Roi(base.x, base.y, base.w / 2, base.h / 2)
        view, _ = learning_view(store)
        entry = view.get_image("img.jpg")
        assert entry.present == ["gelb"]  # rect proves the lid location
        assert entry.absent == ["blau"]  # subset of a bin-free region
        # Shrink to the LOWER-right quadrant: rect now outside.
        store.roi = Roi(
            base.x + base.w / 2, base.y + base.h / 2, base.w / 2, base.h / 2
        )
        view, warnings = learning_view(store)
        entry = view.get_image("img.jpg")
        assert entry.present == []
        assert entry.absent == ["blau"]
        assert any("present labels set aside" in w for w in warnings)

    def test_present_without_rects_needs_containment(self):
        store = _store(Roi(0.2, 0.2, 0.5, 0.5))
        store.set_labels("img.jpg", present=["gelb"])
        store.roi = Roi(0.25, 0.25, 0.3, 0.3)  # shrunk: no proof left
        view, _ = learning_view(store)
        assert view.get_image("img.jpg").present == []

    def test_view_epoch_mismatch_excludes_area_evidence(self):
        store = _store(Roi(0.2, 0.2, 0.5, 0.5))
        store.add_sample("img.jpg", "gelb", Rect(0.3, 0.3, 0.1, 0.1))
        store.set_labels("img.jpg", present=["gelb"], absent=["blau"])
        store.bump_view_epoch([])
        view, warnings = learning_view(store)
        entry = view.get_image("img.jpg")
        assert entry.present == [] and entry.absent == []
        # Color samples survive the view change (hue is geometry-free).
        assert entry.samples["gelb"]
        assert store.confirm_image_view("img.jpg") is True
        view, _ = learning_view(store)
        entry = view.get_image("img.jpg")
        assert entry.present == ["gelb"] and entry.absent == ["blau"]


class TestAppearanceEpochs:
    def test_bump_dormants_samples_and_present_not_absent(self):
        store = _store(Roi(0.1, 0.1, 0.8, 0.8))
        store.add_sample("img.jpg", "gelb", Rect(0.3, 0.3, 0.1, 0.1))
        store.set_labels("img.jpg", present=["gelb"])
        store.set_labels("neg.jpg", absent=["gelb"])
        assert store.mark_bin_appearance_changed("gelb") == 1
        view, warnings = learning_view(store)
        assert "gelb" not in view.get_image("img.jpg").samples
        assert view.get_image("img.jpg").present == []
        # Background stays background under any lid color.
        assert view.get_image("neg.jpg").absent == ["gelb"]
        assert any("appearance epoch" in w for w in warnings)

    def test_epoch_revert_reactivates_old_evidence(self):
        store = _store(Roi(0.1, 0.1, 0.8, 0.8))
        store.add_sample("img.jpg", "gelb", Rect(0.3, 0.3, 0.1, 0.1))
        store.set_labels("img.jpg", present=["gelb"])
        store.mark_bin_appearance_changed("gelb")
        decl = store.get_bin("gelb")
        store.bins[store.bins.index(decl)] = replace(decl, appearance_epoch=0)
        view, _ = learning_view(store)
        assert view.get_image("img.jpg").samples["gelb"]
        assert view.get_image("img.jpg").present == ["gelb"]

    def test_new_samples_after_bump_are_current(self):
        store = _store(Roi(0.1, 0.1, 0.8, 0.8))
        store.add_sample("img.jpg", "gelb", Rect(0.3, 0.3, 0.1, 0.1))
        store.mark_bin_appearance_changed("gelb")
        store.add_sample("img2.jpg", "gelb", Rect(0.4, 0.4, 0.1, 0.1))
        view, _ = learning_view(store)
        assert "gelb" not in view.get_image("img.jpg").samples
        assert view.get_image("img2.jpg").samples["gelb"][0].epoch == 1


class TestBinLifecycle:
    def test_retired_bin_contributes_nothing_but_keeps_data(self):
        store = _store(Roi(0.1, 0.1, 0.8, 0.8))
        store.add_sample("img.jpg", "gelb", Rect(0.3, 0.3, 0.1, 0.1))
        store.set_labels("img.jpg", present=["gelb"], absent=["blau"])
        gelb = store.get_bin("gelb")
        store.bins[store.bins.index(gelb)] = replace(gelb, active=False)
        validate_store(store)  # retired still counts as declared
        view, _ = learning_view(store)
        assert view.bin_ids() == ["blau"]
        assert "gelb" not in view.get_image("img.jpg").samples
        assert view.get_image("img.jpg").present == []
        # The persistent store still has everything.
        assert store.get_image("img.jpg").samples["gelb"]
        assert store.get_image("img.jpg").present == ["gelb"]

    def test_sampling_or_labeling_retired_bin_rejected(self):
        store = _store(Roi(0.1, 0.1, 0.8, 0.8))
        gelb = store.get_bin("gelb")
        store.bins[store.bins.index(gelb)] = replace(gelb, active=False)
        with pytest.raises(CalibrationError):
            store.add_sample("img.jpg", "gelb", Rect(0.3, 0.3, 0.1, 0.1))
        with pytest.raises(CalibrationError):
            store.set_labels("img.jpg", present=["gelb"])


class TestCaptureEpochs:
    def test_unmaterialized_files_keep_their_capture_epoch(self):
        store = _store(Roi(0.1, 0.1, 0.8, 0.8))
        store.bump_view_epoch(["old_capture.jpg"])
        # Materialized AFTER the bump, but captured before it.
        store.set_labels("old_capture.jpg", present=["gelb"])
        assert store.get_image("old_capture.jpg").view_epoch == 0
        assert "old_capture.jpg" not in store.capture_epochs
        # A genuinely new capture gets the current epoch.
        store.set_labels("new_capture.jpg", present=["gelb"])
        assert store.get_image("new_capture.jpg").view_epoch == 1

    def test_capture_epoch_of_existing_entry_untouched(self):
        store = _store(Roi(0.1, 0.1, 0.8, 0.8))
        store.set_labels("img.jpg", present=["gelb"])
        store.bump_view_epoch(["img.jpg"])
        assert store.get_image("img.jpg").view_epoch == 0
        assert "img.jpg" not in store.capture_epochs


class TestSampleOutsideRoi:
    def test_out_of_roi_sample_gets_full_frame_grid(self):
        store = _store(Roi(0.4, 0.4, 0.3, 0.3))
        store.add_sample("img.jpg", "gelb", Rect(0.05, 0.05, 0.1, 0.1))
        sample = store.get_image("img.jpg").samples["gelb"][0]
        assert sample.roi == Roi(0.0, 0.0, 1.0, 1.0)
        validate_store(store)

    def test_in_roi_sample_gets_current_roi_grid(self):
        store = _store(Roi(0.4, 0.4, 0.3, 0.3))
        store.add_sample("img.jpg", "gelb", Rect(0.45, 0.45, 0.1, 0.1))
        assert store.get_image("img.jpg").samples["gelb"][0].roi == Roi(
            0.4, 0.4, 0.3, 0.3
        )


ROI_LEARN = Roi(0.25, 0.25, 0.5, 0.5)
RECT_YELLOW = (0.30, 0.30, 0.10, 0.10)
RECT_BLUE = (0.45, 0.35, 0.10, 0.10)


def _write_scene(path: Path, with_yellow: bool = True, seed: int = 0) -> None:
    rects = [(BLUE, *RECT_BLUE)]
    if with_yellow:
        rects.append((YELLOW, *RECT_YELLOW))
    make_scene(size=(320, 200), rects=rects, seed=seed).save(path, format="PNG")


class TestLearnProfileDynamic:
    def _base_store(self, tmp_path: Path) -> CalibrationStore:
        store = CalibrationStore(
            roi=ROI_LEARN,
            working_width=160,
            resample="bilinear",
            bins=[
                BinDecl("gelb", "Gelbe Tonne"),
                BinDecl("blau", "Blaue Tonne"),
            ],
        )
        for name, seed in (("a.png", 0), ("b.png", 3)):
            _write_scene(tmp_path / name, seed=seed)
            for bin_id, rect in (("gelb", RECT_YELLOW), ("blau", RECT_BLUE)):
                store.add_sample(name, bin_id, Rect(*inner_rect(rect)))
            store.set_labels(name, present=["gelb", "blau"])
        _write_scene(tmp_path / "neg.png", with_yellow=False, seed=7)
        store.add_sample("neg.png", "blau", Rect(*inner_rect(RECT_BLUE)))
        store.set_labels("neg.png", present=["blau"], absent=["gelb"])
        return store

    def test_bin_without_samples_degrades_not_fails(self, tmp_path):
        store = self._base_store(tmp_path)
        store.bins.append(BinDecl("schwarz", "Schwarze Tonne"))
        profile, warnings = learn_profile(store, tmp_path / "store.json")
        assert {b.id for b in profile.bins} == {"gelb", "blau"}
        assert any(
            "schwarz" in w and "untrained" in w for w in warnings
        )

    def test_all_bins_untrained_raises(self, tmp_path):
        store = CalibrationStore(
            roi=ROI_LEARN,
            working_width=160,
            resample="bilinear",
            bins=[BinDecl("gelb", "Gelbe Tonne")],
        )
        _write_scene(tmp_path / "a.png")
        store.set_labels("a.png", present=["gelb"])  # labels but no samples
        with pytest.raises(CalibrationError):
            learn_profile(store, tmp_path / "store.json")

    def test_roi_change_recomputes_thresholds_consistently(self, tmp_path):
        store = self._base_store(tmp_path)
        anchor = tmp_path / "store.json"
        before, _ = learn_profile(store, anchor)
        # Grow the ROI to the full frame: all present labels stay
        # usable (containment), areas are recomputed under the larger
        # denominator, detection profile stays coherent.
        store.roi = Roi(0.0, 0.0, 1.0, 1.0)
        after, warnings = learn_profile(store, anchor)
        assert {b.id for b in after.bins} == {"gelb", "blau"}
        gelb_before = next(b for b in before.bins if b.id == "gelb")
        gelb_after = next(b for b in after.bins if b.id == "gelb")
        # Same blob, four times the reference area: fractions shrink.
        assert (
            gelb_after.learning_stats["min_pos_area_frac"]
            < gelb_before.learning_stats["min_pos_area_frac"]
        )
        assert after.roi == Roi(0.0, 0.0, 1.0, 1.0)

    def test_out_of_roi_sample_feeds_color_model(self, tmp_path):
        # Yellow lid OUTSIDE the configured ROI: the sample must still
        # produce a color model (full-frame grid) even though area
        # learning for that bin has no usable positive inside the ROI.
        store = CalibrationStore(
            # Contains the blue lid (0.45-0.55, 0.35-0.45) but not the
            # yellow one (0.30-0.40, 0.30-0.40).
            roi=Roi(0.42, 0.30, 0.5, 0.5),
            working_width=160,
            resample="bilinear",
            bins=[
                BinDecl("gelb", "Gelbe Tonne"),
                BinDecl("blau", "Blaue Tonne"),
            ],
        )
        for name, seed in (("a.png", 0), ("b.png", 3)):
            _write_scene(tmp_path / name, seed=seed)
            store.add_sample(name, "gelb", Rect(*inner_rect(RECT_YELLOW)))
            store.add_sample(name, "blau", Rect(*inner_rect(RECT_BLUE)))
            store.set_labels(name, present=["blau"])
        profile, warnings = learn_profile(store, tmp_path / "store.json")
        # Blue is trainable (inside ROI); yellow degrades to untrained
        # on the AREA side but its color samples were extractable.
        assert {b.id for b in profile.bins} == {"blau"}
        assert any("gelb" in w and "untrained" in w for w in warnings)

    def test_differing_aspect_ratios_warn(self, tmp_path):
        store = self._base_store(tmp_path)
        _write_scene(tmp_path / "c.png", seed=9)
        # Same scene content, different frame aspect (320x200 vs 320x240).
        make_scene(
            size=(320, 240),
            rects=[(YELLOW, *RECT_YELLOW), (BLUE, *RECT_BLUE)],
            seed=9,
        ).save(tmp_path / "c.png", format="PNG")
        store.add_sample("c.png", "gelb", Rect(*inner_rect(RECT_YELLOW)))
        store.set_labels("c.png", present=["gelb"])
        _profile, warnings = learn_profile(store, tmp_path / "store.json")
        assert any("aspect" in w for w in warnings)


class TestReviewFindings:
    """Regressions for the confirmed adversarial-review findings."""

    def test_migration_drops_zero_area_rects_only(self):
        data = _v1_store_dict()
        data["images"][0]["samples"]["gelbe_tonne"].append(
            {"x": 1.0, "y": 0.2, "w": 0.0, "h": 0.1}  # v1 edge-clamp artifact
        )
        store = store_from_dict(data)  # must not raise
        rects = store.get_image("a.jpg").samples["gelbe_tonne"]
        assert len(rects) == 2  # the two real rects survive
        assert all(r.rect.w > 0 and r.rect.h > 0 for r in rects)

    def test_confirm_image_view_keeps_appearance_epochs_dormant(self):
        store = _store(Roi(0.1, 0.1, 0.8, 0.8))
        store.add_sample("img.jpg", "gelb", Rect(0.3, 0.3, 0.1, 0.1))
        store.set_labels("img.jpg", present=["gelb"])
        store.mark_bin_appearance_changed("gelb")
        store.bump_view_epoch([])
        assert store.confirm_image_view("img.jpg") is True
        entry = store.get_image("img.jpg")
        # View and ROI are re-asserted, the appearance epoch is NOT:
        # the archived pixels still show the old lid.
        assert entry.view_epoch == store.view_epoch
        assert entry.label_epoch["gelb"] == 0
        view, _ = learning_view(store)
        assert view.get_image("img.jpg").present == []

    def test_stale_epoch_rect_outside_roi_vetoes_present_label(self):
        store = _store(Roi(0.0, 0.0, 1.0, 1.0))
        # Old-color rect on the lid's left edge, then a recolor bump
        # and a fresh rect on the lid center plus a fresh label.
        store.add_sample("img.jpg", "gelb", Rect(0.05, 0.4, 0.1, 0.1))
        store.mark_bin_appearance_changed("gelb")
        store.add_sample("img.jpg", "gelb", Rect(0.4, 0.4, 0.1, 0.1))
        store.set_labels("img.jpg", present=["gelb"])
        # Shrink so the old rect (proving lid extent) falls outside.
        store.roi = Roi(0.3, 0.3, 0.5, 0.5)
        view, _ = learning_view(store)
        assert view.get_image("img.jpg").present == []
        # Growing back over both rects revives the label.
        store.roi = Roi(0.0, 0.0, 1.0, 1.0)
        view, _ = learning_view(store)
        assert view.get_image("img.jpg").present == ["gelb"]


class TestViewScopedStatistics:
    def test_gates_and_aspect_ignore_stale_view_frames(self, tmp_path):
        store = CalibrationStore(
            roi=ROI_LEARN,
            working_width=160,
            resample="bilinear",
            bins=[BinDecl("gelb", "Gelbe Tonne")],
        )
        # Old-view frame with a very different aspect and content.
        make_scene(
            size=(320, 240), rects=[(YELLOW, *RECT_YELLOW)], seed=1
        ).save(tmp_path / "old.png", format="PNG")
        store.add_sample("old.png", "gelb", Rect(*inner_rect(RECT_YELLOW)))
        store.bump_view_epoch([])
        for name, seed in (("a.png", 0), ("b.png", 3)):
            _write_scene(tmp_path / name, seed=seed)
            store.add_sample(name, "gelb", Rect(*inner_rect(RECT_YELLOW)))
            store.set_labels(name, present=["gelb"])
        profile, warnings = learn_profile(store, tmp_path / "store.json")
        # No cross-format accusation: the old aspect belongs to a
        # correctly marked older view.
        assert not any("aspect" in w for w in warnings)
        # Gate statistics come from the two current-view frames only.
        assert profile.daylight_stats["n_images"] == 2


class TestFrameIntegrity:
    """v0.3.4: smeared/truncated keyframes must be caught by the
    learned row-duplication gate (field incident 2026-08-12)."""

    @staticmethod
    def _smear(path: Path, source: Path, keep_top: float) -> None:
        """Replicate the last kept row downward, like ffmpeg error
        concealment on a truncated keyframe."""
        import numpy as np
        from PIL import Image

        arr = np.array(Image.open(source).convert("RGB"))
        cut = max(int(arr.shape[0] * keep_top), 1)
        arr[cut:] = arr[cut - 1]
        Image.fromarray(arr).save(path, format="PNG")

    def test_smear_frame_flagged_clean_frame_not(self, tmp_path):
        from wastebin_ai_detector.core import detect_file

        store = CalibrationStore(
            roi=ROI_LEARN,
            working_width=160,
            resample="bilinear",
            bins=[BinDecl("gelb", "Gelbe Tonne")],
        )
        for name, seed in (("a.png", 0), ("b.png", 3)):
            _write_scene(tmp_path / name, seed=seed)
            store.add_sample(name, "gelb", Rect(*inner_rect(RECT_YELLOW)))
            store.set_labels(name, present=["gelb"])
        profile, _ = learn_profile(store, tmp_path / "store.json")
        # Noisy synthetic scenes have (near) zero duplicated rows.
        assert profile.row_dup_max < 0.5
        assert profile.daylight_stats["max_row_dup_frac"] == profile.row_dup_max

        clean = detect_file(tmp_path / "a.png", profile)
        assert clean.frame_integrity_suspect is False

        self._smear(tmp_path / "smear.png", tmp_path / "a.png", keep_top=0.15)
        smeared = detect_file(tmp_path / "smear.png", profile)
        assert smeared.row_dup_frac > profile.row_dup_max
        assert smeared.frame_integrity_suspect is True

    def test_row_duplicate_fraction_bounds(self):
        import numpy as np

        from wastebin_ai_detector.core import row_duplicate_fraction

        rng = np.random.default_rng(11)
        noisy = rng.uniform(0.0, 1.0, (50, 40, 3))
        assert row_duplicate_fraction(noisy) == 0.0
        flat = np.tile(noisy[0:1], (50, 1, 1))
        assert row_duplicate_fraction(flat) == 1.0
        assert row_duplicate_fraction(noisy[:1]) == 0.0  # single row


class TestBlobLocalization:
    """Phase 2.4a: bounding box and centroid of the detected lid, in
    full-image relative coordinates - the calibration card overlay."""

    def test_bbox_and_centroid_match_the_drawn_lid(self, tmp_path):
        from wastebin_ai_detector.core import detect_file

        store = CalibrationStore(
            roi=ROI_LEARN,
            working_width=160,
            resample="bilinear",
            bins=[
                BinDecl("gelb", "Gelbe Tonne"),
                BinDecl("blau", "Blaue Tonne"),
            ],
        )
        for name, seed in (("a.png", 0), ("b.png", 3)):
            _write_scene(tmp_path / name, seed=seed)
            store.add_sample(name, "gelb", Rect(*inner_rect(RECT_YELLOW)))
            store.add_sample(name, "blau", Rect(*inner_rect(RECT_BLUE)))
            store.set_labels(name, present=["gelb", "blau"])
        profile, _ = learn_profile(store, tmp_path / "store.json")
        result = detect_file(tmp_path / "a.png", profile)
        for bin_id, rect in (("gelb", RECT_YELLOW), ("blau", RECT_BLUE)):
            r = next(b for b in result.bins if b.id == bin_id)
            assert r.present and r.bbox is not None
            x, y, w, h = r.bbox
            # The detected box must tightly cover the drawn lid: the
            # tolerance is one working pixel in image fractions, plus
            # the blob may be a pixel short of the exact edge.
            px = ROI_LEARN.w / 160.0
            assert abs(x - rect[0]) <= 2 * px
            assert abs(y - rect[1]) <= 2 * px
            assert abs(w - rect[2]) <= 4 * px
            assert abs(h - rect[3]) <= 4 * px
            cx, cy = r.centroid
            assert rect[0] <= cx <= rect[0] + rect[2]
            assert rect[1] <= cy <= rect[1] + rect[3]
        # Absent lid: no location claim.
        _write_scene(tmp_path / "no_yellow.png", with_yellow=False, seed=9)
        result = detect_file(tmp_path / "no_yellow.png", profile)
        gelb = next(b for b in result.bins if b.id == "gelb")
        assert gelb.area_frac == 0.0 and gelb.bbox is None

    def test_region_matches_area_on_random_masks(self):
        import numpy as np

        from wastebin_ai_detector.core import (
            largest_component_area,
            largest_component_region,
        )

        rng = np.random.default_rng(4)
        for density in (0.2, 0.5, 0.8):
            mask = rng.uniform(0, 1, (40, 60)) < density
            region = largest_component_region(mask)
            expected = largest_component_area(mask)
            if expected == 0:
                assert region is None
                continue
            area, (x0, y0, x1, y1), (cx, cy) = region
            assert area == expected
            assert 0 <= x0 < x1 <= 60 and 0 <= y0 < y1 <= 40
            assert x0 <= cx < x1 and y0 <= cy < y1


class TestPolygonRegion:
    """v0.5.0: polygon region end-to-end through learn and detect."""

    def _region_store(self, tmp_path: Path) -> CalibrationStore:
        from wastebin_ai_detector.core import rect_as_rings

        store = CalibrationStore(
            roi=ROI_LEARN,
            working_width=160,
            resample="bilinear",
            bins=[BinDecl("gelb", "Gelbe Tonne")],
        )
        for name, seed in (("a.png", 0), ("b.png", 3)):
            _write_scene(tmp_path / name, seed=seed)
            store.add_sample(name, "gelb", Rect(*inner_rect(RECT_YELLOW)))
            store.set_labels(name, present=["gelb"])
        return store

    def test_rect_ring_region_is_bit_identical_to_rect(self, tmp_path):
        from wastebin_ai_detector.core import detect_file, rect_as_rings

        store = self._region_store(tmp_path)
        plain, _ = learn_profile(store, tmp_path / "store.json")
        store.set_region(rect_as_rings(ROI_LEARN))
        ringed, _ = learn_profile(store, tmp_path / "store.json")
        gelb_a = next(b for b in plain.bins if b.id == "gelb")
        gelb_b = next(b for b in ringed.bins if b.id == "gelb")
        # Same denominator (full bbox), same thresholds.
        assert gelb_a.min_area_frac == gelb_b.min_area_frac
        ra = detect_file(tmp_path / "a.png", plain)
        rb = detect_file(tmp_path / "a.png", ringed)
        assert ra.bins[0].area_frac == rb.bins[0].area_frac

    def test_polygon_excludes_background_region(self, tmp_path):
        """A distractor patch of lid color outside the polygon must not
        count - the hedge-fringe field case."""
        from scenes import make_scene

        store = CalibrationStore(
            roi=Roi(0.0, 0.0, 1.0, 1.0),
            working_width=160,
            resample="bilinear",
            bins=[BinDecl("gelb", "Gelbe Tonne")],
        )
        # Scene: real lid at RECT_YELLOW, decoy of the same color at
        # the far right (outside the polygon drawn below).
        DECOY = (0.75, 0.30, 0.10, 0.10)
        for name, seed in (("a.png", 0), ("b.png", 3)):
            make_scene(
                size=(320, 200),
                rects=[(YELLOW, *RECT_YELLOW), (YELLOW, *DECOY)],
                seed=seed,
            ).save(tmp_path / name, format="PNG")
            store.add_sample(name, "gelb", Rect(*inner_rect(RECT_YELLOW)))
            store.set_labels(name, present=["gelb"])
        # Polygon around the lid area only (generous, decoy outside).
        store.set_region([[(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]])
        profile, _ = learn_profile(store, tmp_path / "store.json")
        from wastebin_ai_detector.core import detect_file

        # Lid present: detected from inside the region.
        res = detect_file(tmp_path / "a.png", profile)
        assert res.bins[0].present is True
        # Lid gone, decoy stays: nothing inside the region -> absent.
        make_scene(
            size=(320, 200), rects=[(YELLOW, *DECOY)], seed=7
        ).save(tmp_path / "gone.png", format="PNG")
        res = detect_file(tmp_path / "gone.png", profile)
        assert res.bins[0].area_frac == 0.0
        assert res.bins[0].present is False

    def test_older_store_dicts_migrate_to_the_current_schema(self):
        from wastebin_ai_detector.core.store import STORE_SCHEMA_VERSION

        for version in (2, 3):
            data = store_to_dict(_store(Roi(0.1, 0.1, 0.8, 0.8)))
            data["schema_version"] = version
            for e in data["images"]:
                e.pop("auto", None)
                if version < 3:
                    e.pop("label_polygons", None)
            if version < 3:
                data.pop("roi_polygons", None)
            loaded = store_from_dict(data)
            assert loaded.schema_version == STORE_SCHEMA_VERSION
            assert loaded.roi_polygons is None
            # Every pre-v4 entry means "made by a human".
            assert all(e.auto is None for e in loaded.images)

    def test_polygon_region_containment_in_learning_view(self):
        store = _store(Roi(0.0, 0.0, 1.0, 1.0))
        store.set_region([[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]])
        store.add_sample("img.jpg", "gelb", Rect(0.3, 0.3, 0.1, 0.1))
        store.set_labels("img.jpg", present=["gelb"], absent=["blau"])
        view, _ = learning_view(store)
        entry = view.get_image("img.jpg")
        assert entry.present == ["gelb"] and entry.absent == ["blau"]
        # Shrink the polygon so the rect falls outside: label region no
        # longer contained, rect outside -> present set aside; current
        # region still inside label region -> absent stays.
        store.set_region([[(0.5, 0.5), (0.9, 0.5), (0.9, 0.9), (0.5, 0.9)]])
        view, _ = learning_view(store)
        entry = view.get_image("img.jpg")
        assert entry.present == []
        assert entry.absent == ["blau"]


class TestShapePlausibility:
    def test_hedge_fringe_and_streak_rejected_lid_accepted(self, tmp_path):
        """Field regression: ragged low-fill fringe and a tall vertical
        streak of lid color must not stand in for the lid."""
        import numpy as np

        from scenes import make_scene
        from wastebin_ai_detector.core import detect_file

        store = CalibrationStore(
            roi=Roi(0.0, 0.0, 1.0, 1.0),
            working_width=160,
            resample="bilinear",
            bins=[BinDecl("gelb", "Gelbe Tonne")],
        )
        for name, seed in (("a.png", 0), ("b.png", 3)):
            _write_scene(tmp_path / name, seed=seed)
            store.add_sample(name, "gelb", Rect(*inner_rect(RECT_YELLOW)))
            store.set_labels(name, present=["gelb"])
        profile, _ = learn_profile(store, tmp_path / "store.json")
        gelb = profile.bins[0]
        assert gelb.learning_stats["shape_n"] == 2

        # Tall narrow streak (like the sunlit black-bin body edge).
        make_scene(
            size=(320, 200), rects=[(YELLOW, 0.40, 0.10, 0.025, 0.70)], seed=9
        ).save(tmp_path / "streak.png", format="PNG")
        res = detect_file(tmp_path / "streak.png", profile)
        assert res.bins[0].area_frac == 0.0, "streak must be implausible"

        # Ragged fringe: scattered small patches of lid color (leaves).
        rng_rects = [
            (YELLOW, 0.30 + 0.05 * i, 0.25 + 0.04 * (i % 3), 0.012, 0.012)
            for i in range(8)
        ]
        make_scene(size=(320, 200), rects=rng_rects, seed=11).save(
            tmp_path / "fringe.png", format="PNG"
        )
        res = detect_file(tmp_path / "fringe.png", profile)
        # Each patch is a tiny component; even the largest is far from
        # lid fill/aspect... but tiny squares are compact, so the guard
        # here is that their AREA is below the threshold - the fill
        # criterion targets connected ragged fringes instead:
        assert res.bins[0].present is False

        # The real lid still passes.
        res = detect_file(tmp_path / "a.png", profile)
        assert res.bins[0].present is True

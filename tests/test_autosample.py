"""Unattended colour sampling: the bounded reservoir, the reference
rectangles it re-applies, and the adoption test that decides whether
what it collected may enter the models."""

from __future__ import annotations

import math

import pytest

from wastebin_ai_detector.core import (
    AUTO_RESERVOIR_CAPACITY,
    AutoStamp,
    adoption_verdict,
    describe_reservoir,
    has_auto_evidence,
    learn_profile,
    reference_rects,
    regime_distance,
    reservoir_decision,
    reservoir_trim,
    without_auto_evidence,
)
from wastebin_ai_detector.core.autosample import REGIME_QUANTUM
from wastebin_ai_detector.core.errors import ProfileError
from wastebin_ai_detector.core.learn import SV_MIN_PERCENTILE
from wastebin_ai_detector.core.profile import BinModel, Profile
from wastebin_ai_detector.core.profile import Roi as PRoi
from wastebin_ai_detector.core.store import (
    BinDecl,
    CalibrationStore,
    Rect,
    Roi,
    store_from_dict,
    store_to_dict,
    validate_store,
)


class TestCapacity:
    def test_derived_from_the_percentile_convention(self):
        # Never the literal: the number IS the smallest vote count at
        # which the learner's floor percentile stops being an extremum.
        assert AUTO_RESERVOIR_CAPACITY == int(
            math.ceil(1.0 + 100.0 / SV_MIN_PERCENTILE)
        )


class TestReservoir:
    def test_fills_up_to_capacity(self):
        retained = []
        for i in range(AUTO_RESERVOIR_CAPACITY):
            coord = (0.1 + i * 0.01, 0.5)
            decision = reservoir_decision(retained, coord)
            assert decision.accept and decision.reason == "fill"
            retained.append((f"f{i}.jpg", coord))
        assert len(retained) == AUTO_RESERVOIR_CAPACITY

    def test_duplicate_regime_is_declined(self):
        retained = [("a.jpg", (0.4, 0.6))]
        decision = reservoir_decision(
            retained, (0.4 + REGIME_QUANTUM / 2, 0.6)
        )
        assert not decision.accept
        assert decision.reason == "duplicate_regime"

    def test_full_reservoir_takes_only_more_dispersed_candidates(self):
        # A tight cluster plus one outlier: a candidate far away
        # improves the spread, a candidate inside the cluster does not.
        retained = [
            (f"c{i}.jpg", (0.30 + i * 0.005, 0.50))
            for i in range(AUTO_RESERVOIR_CAPACITY)
        ]
        far = reservoir_decision(retained, (0.90, 0.90))
        assert far.accept and far.reason == "improves_dispersion"
        assert far.evict is not None
        near = reservoir_decision(retained, (0.3025, 0.50))
        assert not near.accept

    def test_capacity_is_never_exceeded_and_spread_never_shrinks(self):
        import random

        rng = random.Random(7)
        retained: list[tuple[str, tuple[float, float]]] = []
        previous = -math.inf
        for i in range(400):
            coord = (rng.random(), rng.random())
            decision = reservoir_decision(retained, coord)
            if decision.accept:
                if decision.evict:
                    retained = [r for r in retained if r[0] != decision.evict]
                retained.append((f"f{i}.jpg", coord))
            assert len(retained) <= AUTO_RESERVOIR_CAPACITY
            if len(retained) == AUTO_RESERVOIR_CAPACITY:
                coords = [c for _p, c in retained]
                spread = min(
                    regime_distance(coords[a], coords[b])
                    for a in range(len(coords))
                    for b in range(a + 1, len(coords))
                )
                assert spread >= previous - 1e-12
                previous = spread

    def test_trim_drops_the_most_redundant(self):
        retained = [
            ("far.jpg", (0.9, 0.9)),
            ("a.jpg", (0.10, 0.10)),
            ("b.jpg", (0.11, 0.10)),
        ]
        dropped = reservoir_trim(retained, capacity=2)
        assert dropped in (["a.jpg"], ["b.jpg"])


def _store_with_marks() -> CalibrationStore:
    store = CalibrationStore(
        roi=Roi(0.0, 0.0, 1.0, 1.0),
        working_width=160,
        resample="bilinear",
        bins=[BinDecl("gelb", "Gelb"), BinDecl("blau", "Blau")],
    )
    store.add_sample("manual.jpg", "gelb", Rect(0.30, 0.30, 0.05, 0.05))
    store.add_sample("manual.jpg", "blau", Rect(0.60, 0.30, 0.05, 0.05))
    store.set_labels("manual.jpg", present=["gelb", "blau"])
    return store


class TestReferenceRects:
    def test_uses_the_newest_manual_marks(self):
        store = _store_with_marks()
        store.add_sample("newer.jpg", "gelb", Rect(0.31, 0.31, 0.04, 0.04))
        refs = reference_rects(store)
        assert set(refs) == {"gelb", "blau"}
        assert refs["gelb"][0].rect.x == pytest.approx(0.31)

    def test_auto_and_excluded_entries_are_no_reference(self):
        store = _store_with_marks()
        store.record_auto_frame(
            "auto.jpg",
            AutoStamp(0.4, 0.5),
            samples={"gelb": [Rect(0.9, 0.9, 0.02, 0.02)]},
            present=["gelb"],
            absent=[],
        )
        store.forget_image("manual.jpg")
        assert reference_rects(store) == {}

    def test_a_bumped_camera_stops_collection(self):
        store = _store_with_marks()
        store.bump_view_epoch([])
        assert reference_rects(store) == {}

    def test_a_recoloured_lid_invalidates_only_its_own_reference(self):
        store = _store_with_marks()
        store.mark_bin_appearance_changed("gelb")
        refs = reference_rects(store)
        assert set(refs) == {"blau"}


class TestAutoEvidenceStore:
    def test_auto_entries_carry_the_declared_situation(self):
        store = _store_with_marks()
        store.record_auto_frame(
            "auto.jpg",
            AutoStamp(0.4, 0.5),
            samples={"gelb": [Rect(0.3, 0.3, 0.05, 0.05)]},
            present=["gelb"],
            absent=["blau"],
        )
        validate_store(store)
        entry = store.get_image("auto.jpg")
        assert entry.auto == AutoStamp(0.4, 0.5)
        # The labels are the user's declaration for the run, applied
        # to a frame captured during it.
        assert entry.present == ["gelb"] and entry.absent == ["blau"]
        assert has_auto_evidence(store)

    def test_a_broken_stamp_is_rejected_by_validation(self):
        store = _store_with_marks()
        store.record_auto_frame(
            "auto.jpg",
            AutoStamp(0.4, 0.5),
            samples={"gelb": [Rect(0.3, 0.3, 0.05, 0.05)]},
            present=["gelb"],
            absent=["blau"],
        )
        store.get_image("auto.jpg").auto = AutoStamp(1.5, 0.5)
        with pytest.raises(ProfileError):
            validate_store(store)

    def test_human_attention_promotes_an_auto_entry(self):
        store = _store_with_marks()
        store.record_auto_frame(
            "auto.jpg",
            AutoStamp(0.4, 0.5),
            samples={"gelb": [Rect(0.3, 0.3, 0.05, 0.05)]},
            present=["gelb"],
            absent=["blau"],
        )
        store.set_labels("auto.jpg", present=["gelb"])
        assert store.get_image("auto.jpg").auto is None
        assert store.auto_reservoir() == []
        validate_store(store)

    def test_the_collector_never_writes_into_existing_entries(self):
        store = _store_with_marks()
        with pytest.raises(Exception):
            store.record_auto_frame(
                "manual.jpg",
                AutoStamp(0.4, 0.5),
                samples={"gelb": [Rect(0.3, 0.3, 0.05, 0.05)]},
                present=["gelb"],
                absent=[],
            )

    def test_bulk_discard_and_restore_delete_nothing(self):
        store = _store_with_marks()
        store.record_auto_frame(
            "auto.jpg",
            AutoStamp(0.4, 0.5),
            samples={"gelb": [Rect(0.3, 0.3, 0.05, 0.05)]},
            present=["gelb"],
            absent=["blau"],
        )
        before = store_to_dict(store)
        assert store.discard_auto_evidence() == ["auto.jpg"]
        assert not has_auto_evidence(store)
        assert store.get_image("auto.jpg").samples["gelb"]
        assert store.restore_auto_evidence() == ["auto.jpg"]
        assert store_to_dict(store) == before

    def test_round_trip_keeps_the_stamp(self):
        store = _store_with_marks()
        store.record_auto_frame(
            "auto.jpg",
            AutoStamp(0.4, 0.5),
            samples={"gelb": [Rect(0.3, 0.3, 0.05, 0.05)]},
            present=["gelb"],
            absent=["blau"],
        )
        again = store_from_dict(store_to_dict(store))
        assert again.get_image("auto.jpg").auto == AutoStamp(0.4, 0.5)

    def test_baseline_copy_switches_auto_evidence_off(self):
        store = _store_with_marks()
        store.record_auto_frame(
            "auto.jpg",
            AutoStamp(0.4, 0.5),
            samples={"gelb": [Rect(0.3, 0.3, 0.05, 0.05)]},
            present=["gelb"],
            absent=["blau"],
        )
        baseline = without_auto_evidence(store)
        assert baseline.get_image("auto.jpg").excluded is True
        # The original is untouched.
        assert store.get_image("auto.jpg").excluded is False

    def test_progress_description(self):
        store = _store_with_marks()
        for i in range(3):
            store.record_auto_frame(
                f"auto{i}.jpg",
                AutoStamp(0.3 + i * 0.1, 0.5),
                samples={"gelb": [Rect(0.3, 0.3, 0.05, 0.05)]},
                present=["gelb"],
                absent=[],
            )
        info = describe_reservoir(store)
        assert info["total_retained"] == 3
        assert info["capacity_per_situation"] == AUTO_RESERVOIR_CAPACITY
        # One declared situation, so one group.
        assert len(info["situations"]) == 1
        group = info["situations"][0]
        assert group["present"] == ["gelb"] and group["absent"] == []
        assert group["retained"] == 3
        assert group["sat_range"] == [pytest.approx(0.3), pytest.approx(0.5)]

    def test_each_declared_situation_gets_its_own_capacity(self):
        from wastebin_ai_detector.core import (
            over_capacity_paths,
            reservoir_by_situation,
        )

        store = _store_with_marks()
        for i in range(3):
            store.record_auto_frame(
                f"present{i}.jpg",
                AutoStamp(0.3 + i * 0.1, 0.5),
                samples={"gelb": [Rect(0.3, 0.3, 0.05, 0.05)]},
                present=["gelb"],
                absent=[],
            )
        for i in range(2):
            store.record_auto_frame(
                f"absent{i}.jpg",
                AutoStamp(0.2 + i * 0.1, 0.4),
                samples={},
                present=[],
                absent=["gelb"],
            )
        grouped = reservoir_by_situation(store)
        assert len(grouped) == 2
        assert {len(v) for v in grouped.values()} == {3, 2}
        # Both groups are within capacity, so nothing has to give way.
        assert over_capacity_paths(store) == []


def _profile(**per_bin) -> Profile:
    bins = [
        BinModel(
            id=bin_id,
            name=bin_id,
            hue_center_deg=stats.pop("hue", 60.0),
            hue_tol_deg=stats.pop("tol", 8.0),
            sat_min=0.2,
            val_min=0.2,
            min_area_frac=0.01,
            learning_stats=stats,
        )
        for bin_id, stats in per_bin.items()
    ]
    return Profile(
        roi=PRoi(0.0, 0.0, 1.0, 1.0),
        working_width=160,
        resample="bilinear",
        daylight_sat_min=0.1,
        bins=bins,
    )


class TestAdoption:
    def test_unchanged_measurements_are_adopted(self):
        stats = dict(separable=True, min_pos_area_frac=0.05, max_neg_area_frac=0.01)
        verdict = adoption_verdict(
            _profile(gelb=dict(stats)), _profile(gelb=dict(stats))
        )
        assert verdict.adopt and verdict.regressions == []

    def test_a_lost_bin_blocks(self):
        base = _profile(
            gelb=dict(separable=True, min_pos_area_frac=0.05, max_neg_area_frac=0.01),
            blau=dict(hue=210.0, separable=True,
                      min_pos_area_frac=0.05, max_neg_area_frac=0.01),
        )
        cand = _profile(
            gelb=dict(separable=True, min_pos_area_frac=0.05, max_neg_area_frac=0.01)
        )
        verdict = adoption_verdict(base, cand)
        assert not verdict.adopt
        assert any("no longer trainable" in r for r in verdict.regressions)

    def test_losing_separability_blocks(self):
        base = _profile(
            gelb=dict(separable=True, min_pos_area_frac=0.05, max_neg_area_frac=0.01)
        )
        cand = _profile(
            gelb=dict(separable=False, min_pos_area_frac=0.005, max_neg_area_frac=0.01)
        )
        verdict = adoption_verdict(base, cand)
        assert not verdict.adopt
        assert any("stopped separating" in r for r in verdict.regressions)

    def test_an_already_overlapping_bin_may_not_lose_more_ground(self):
        """The field's brown bin: no slack left, so any further loss is
        unacceptable - while a separable bin absorbs the same noise."""
        base = _profile(
            braun=dict(
                separable=False, min_pos_area_frac=0.017, max_neg_area_frac=0.025
            )
        )
        worse = _profile(
            braun=dict(
                separable=False, min_pos_area_frac=0.010, max_neg_area_frac=0.025
            )
        )
        better = _profile(
            braun=dict(
                separable=False, min_pos_area_frac=0.020, max_neg_area_frac=0.025
            )
        )
        assert not adoption_verdict(base, worse).adopt
        assert adoption_verdict(base, better).adopt

    def test_a_separable_bin_tolerates_single_frame_noise(self):
        base = _profile(
            gelb=dict(separable=True, min_pos_area_frac=0.05, max_neg_area_frac=0.01)
        )
        noisier = _profile(
            gelb=dict(separable=True, min_pos_area_frac=0.04, max_neg_area_frac=0.012)
        )
        assert adoption_verdict(base, noisier).adopt

    def test_a_new_hue_overlap_blocks_but_a_pre_existing_one_does_not(self):
        base = _profile(
            gelb=dict(hue=60.0, tol=8.0, separable=True,
                      min_pos_area_frac=0.05, max_neg_area_frac=0.01),
            braun=dict(hue=20.0, tol=40.0, separable=True,
                       min_pos_area_frac=0.05, max_neg_area_frac=0.01),
        )
        # Same overlapping pair: status quo, not a regression.
        assert adoption_verdict(base, base).adopt
        widened = _profile(
            gelb=dict(hue=60.0, tol=8.0, separable=True,
                      min_pos_area_frac=0.05, max_neg_area_frac=0.01),
            braun=dict(hue=20.0, tol=40.0, separable=True,
                       min_pos_area_frac=0.05, max_neg_area_frac=0.01),
            blau=dict(hue=200.0, tol=200.0 - 60.0 - 8.0 + 1.0, separable=True,
                      min_pos_area_frac=0.05, max_neg_area_frac=0.01),
        )
        base3 = _profile(
            gelb=dict(hue=60.0, tol=8.0, separable=True,
                      min_pos_area_frac=0.05, max_neg_area_frac=0.01),
            braun=dict(hue=20.0, tol=40.0, separable=True,
                       min_pos_area_frac=0.05, max_neg_area_frac=0.01),
            blau=dict(hue=200.0, tol=8.0, separable=True,
                      min_pos_area_frac=0.05, max_neg_area_frac=0.01),
        )
        verdict = adoption_verdict(base3, widened)
        assert not verdict.adopt
        assert any("hue bands now overlap" in r for r in verdict.regressions)

    def test_rescuing_an_untrained_bin_never_blocks(self):
        base = _profile(
            gelb=dict(separable=True, min_pos_area_frac=0.05, max_neg_area_frac=0.01)
        )
        cand = _profile(
            gelb=dict(separable=True, min_pos_area_frac=0.05, max_neg_area_frac=0.01),
            blau=dict(hue=210.0, separable=True,
                      min_pos_area_frac=0.05, max_neg_area_frac=0.01),
        )
        assert adoption_verdict(base, cand).adopt


class TestEvidenceSemantics:
    """What a learning run may and may not teach.

    A declaration says the bin STANDS there. It does not say that its
    lid is measurable in a given frame, and the two differ exactly when
    a shadow, a passing van or heavy rain covers most of the lid. The
    thresholds that decide "present" rest on extrema, so a truthful
    declaration plus one covered lid would otherwise collapse them.
    """

    def _store(self, tmp_path, auto_area_scale: float):
        from scenes import YELLOW, make_scene

        store = CalibrationStore(
            roi=Roi(0.0, 0.0, 1.0, 1.0),
            working_width=160,
            resample="bilinear",
            bins=[BinDecl("gelb", "Gelb")],
        )
        lid = (0.30, 0.30, 0.20, 0.20)
        for i in range(2):
            name = f"m{i}.png"
            make_scene(rects=[(YELLOW, *lid)], seed=i).save(
                tmp_path / name, format="PNG"
            )
            store.add_sample(name, "gelb", Rect(0.34, 0.34, 0.06, 0.06))
            store.set_labels(name, present=["gelb"])
        make_scene(rects=[], seed=9).save(tmp_path / "empty.png", format="PNG")
        store.set_labels("empty.png", absent=["gelb"])
        # One frame from a declared-present run in which the lid is
        # mostly covered: the declaration is TRUE, the measurement tiny.
        covered = (lid[0], lid[1], lid[2] * auto_area_scale, lid[3])
        make_scene(rects=[(YELLOW, *covered)], seed=21).save(
            tmp_path / "run.png", format="PNG"
        )
        store.record_auto_frame(
            "run.png",
            AutoStamp(0.5, 0.5),
            samples={"gelb": [Rect(0.32, 0.34, 0.03, 0.06)]},
            present=["gelb"],
            absent=[],
        )
        return store

    def test_a_run_frame_is_no_positive_and_no_shape_observation(
        self, tmp_path
    ):
        """The covered lid must not become the smallest positive: the
        threshold would follow it down and every speck would read
        present afterwards (measured before this rule: 305-fold)."""
        profile, _warnings = learn_profile(
            self._store(tmp_path, 0.1), tmp_path / "s.json"
        )
        stats = profile.bins[0].learning_stats
        # Two manual present images, one manual absent - the run's own
        # frame is in neither count.
        assert stats["n_pos"] == 2
        assert stats["n_neg"] == 1
        assert stats["shape_n"] == 2

    def test_a_declared_absent_run_frame_does_count_as_negative(
        self, tmp_path
    ):
        """The other direction is exactly what runs are for: an empty
        yard is a complete statement, no visibility judgement needed."""
        from scenes import YELLOW, make_scene

        store = self._store(tmp_path, 0.1)
        make_scene(rects=[], seed=33).save(
            tmp_path / "away.png", format="PNG"
        )
        store.record_auto_frame(
            "away.png",
            AutoStamp(0.45, 0.45),
            samples={},
            present=[],
            absent=["gelb"],
        )
        profile, _warnings = learn_profile(store, tmp_path / "s.json")
        stats = profile.bins[0].learning_stats
        assert stats["n_pos"] == 2
        assert stats["n_neg"] == 2

    def test_but_a_run_frame_does_teach_colour(self, tmp_path):
        with_run, _w = learn_profile(
            self._store(tmp_path, 0.1), tmp_path / "s.json"
        )
        # Three sample images: two manual plus the run's frame.
        assert with_run.bins[0].learning_stats["n_sample_images"] == 3

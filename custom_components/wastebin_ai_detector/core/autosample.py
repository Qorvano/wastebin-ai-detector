"""Unattended color sampling: which frames to keep, and when to trust
what they taught.

While learning mode runs, the collector can re-apply the sample
rectangles the user drew once to every new snapshot. Two mechanisms
keep that from turning into a slow poisoning of the models:

- a bounded RESERVOIR that keeps the frames spread as widely as
  possible over the measured light space, so weeks of collecting
  neither grow the store without limit nor drown the user's own
  samples in near-duplicates of one afternoon;
- an ADOPTION test that relearns twice (with and without the collected
  evidence) and keeps the collected half only if it does not regress
  the measurements taken on human-labeled images - evidence the
  collector can never touch, which is what makes the test honest
  instead of circular.

Nothing here generates presence labels. ``min_area_frac`` is anchored
by the smallest positive and the largest negative blob a HUMAN
confirmed; the only machine able to produce such a label is the
detector itself, so a machine-made label would be the detector grading
its own homework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .color import RGB_8BIT_LEVELS
from .learn import SV_MIN_PERCENTILE, hue_bands_overlap
from .profile import Profile
from .store import CalibrationStore, SampleRect

# How many unattended frames are worth keeping. Derived, not chosen:
# color evidence votes once per IMAGE, and the floors are learned as
# the SV_MIN_PERCENTILE-th percentile of the pooled votes. The q-th
# percentile of n observations still equals the sample minimum while
# q/100 * (n - 1) < 1 (the same order-statistics fact the learner
# already uses for its "too few coherent pixels" warning), so
# n = 1 + 100/q is the smallest count at which that percentile becomes
# a genuine percentile rather than an extremum. Beyond it, more
# unattended votes buy a better-estimated same statistic, not a
# qualitatively different one.
AUTO_RESERVOIR_CAPACITY = int(math.ceil(1.0 + 100.0 / SV_MIN_PERCENTILE))

# Two light regimes closer than one 8-bit quantum on both axes are the
# same regime as far as the camera can express it.
REGIME_QUANTUM = 1.0 / RGB_8BIT_LEVELS


def regime_distance(
    a: tuple[float, float], b: tuple[float, float]
) -> float:
    """Chebyshev distance between two light-regime coordinates.

    "Distinguishable on at least one axis" is the property that
    matters, and the two axes are independent measurements in the same
    unit - so the maximum, not the sum or the euclidean mix.
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


@dataclass(frozen=True)
class ReservoirDecision:
    accept: bool
    reason: str
    evict: str | None = None


def _min_separation(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return math.inf
    return min(
        regime_distance(points[i], points[j])
        for i in range(len(points))
        for j in range(i + 1, len(points))
    )


def reservoir_decision(
    retained: list[tuple[str, tuple[float, float]]],
    candidate: tuple[float, float],
    capacity: int = AUTO_RESERVOIR_CAPACITY,
) -> ReservoirDecision:
    """Keep the retained set as spread out over the light space as the
    weather allows.

    Below capacity every distinguishable regime is taken. At capacity a
    candidate is admitted only if replacing the most redundant member
    with it increases the minimum distance between retained regimes -
    so the set converges towards a maximally dispersed sample of the
    conditions this yard actually produces, and acceptances become
    rarer over time all by themselves. There is no time constant in
    this feature: dispersion IS the rate limit.
    """
    coords = [coord for _path, coord in retained]
    if any(regime_distance(candidate, c) < REGIME_QUANTUM for c in coords):
        return ReservoirDecision(False, "duplicate_regime")
    if len(retained) < capacity:
        return ReservoirDecision(True, "fill")
    # Candidate first, so a tie keeps the status quo.
    universe = [("", candidate), *retained]
    best_index, best_separation = 0, -math.inf
    for i in range(len(universe)):
        rest = [c for j, (_p, c) in enumerate(universe) if j != i]
        separation = _min_separation(rest)
        if separation > best_separation:
            best_index, best_separation = i, separation
    if best_index == 0:
        return ReservoirDecision(False, "less_dispersed")
    return ReservoirDecision(
        True, "improves_dispersion", evict=universe[best_index][0]
    )


def reservoir_trim(
    retained: list[tuple[str, tuple[float, float]]],
    capacity: int = AUTO_RESERVOIR_CAPACITY,
) -> list[str]:
    """Paths to drop when the reservoir is over capacity (after a
    restore), most redundant first - the same dispersion rule."""
    working = list(retained)
    dropped: list[str] = []
    while len(working) > capacity:
        best_index, best_separation = 0, -math.inf
        for i in range(len(working)):
            rest = [c for j, (_p, c) in enumerate(working) if j != i]
            separation = _min_separation(rest)
            if separation > best_separation:
                best_index, best_separation = i, separation
        dropped.append(working.pop(best_index)[0])
    return dropped


def reference_rects(
    store: CalibrationStore,
) -> dict[str, list[SampleRect]]:
    """Per bin, the rectangles the auto sampler re-applies.

    The reference is the NEWEST manual entry (store order) that carries
    rectangles for that bin at the bin's current appearance epoch,
    under the current view. Each clause earns its place:

    - manual only, so drift cannot compound over generations;
    - current appearance epoch, so a recolored lid invalidates its own
      reference through machinery that already exists;
    - current view epoch, so a bumped camera stops collection until the
      user marks a lid once on a current frame;
    - newest wins, so "mark it again" is the whole redefinition UX.

    Bins without such an entry are absent from the result and are not
    auto-sampled: no reference, no guessing.
    """
    epoch_of = {b.id: b.appearance_epoch for b in store.bins if b.active}
    result: dict[str, list[SampleRect]] = {}
    for entry in store.images:
        if entry.auto is not None or entry.excluded:
            continue
        if entry.view_epoch != store.view_epoch:
            continue
        for bin_id, rects in entry.samples.items():
            if bin_id not in epoch_of:
                continue
            current = [r for r in rects if r.epoch == epoch_of[bin_id]]
            if current:
                result[bin_id] = list(current)
    return result


@dataclass(frozen=True)
class AdoptionVerdict:
    adopt: bool
    regressions: list[str]
    gaps: dict[str, float | None]


def _bin_stats(profile: Profile) -> dict[str, dict[str, Any]]:
    return {b.id: b.learning_stats for b in profile.bins}


def _gap(stats: dict[str, Any]) -> float | None:
    """How far the two measured extremes are apart, as a ratio.

    ``min_pos / max_neg`` above 1 means the calibration separates; the
    ratio is scale free, so it can be compared across relearns.
    """
    min_pos = stats.get("min_pos_area_frac")
    max_neg = stats.get("max_neg_area_frac")
    if min_pos is None or max_neg is None:
        return None
    if max_neg <= 0.0:
        return math.inf
    return float(min_pos) / float(max_neg)


def overlapping_hue_pairs(profile: Profile) -> set[tuple[str, str]]:
    pairs = set()
    for i, a in enumerate(profile.bins):
        for b in profile.bins[i + 1 :]:
            if hue_bands_overlap(a, b):
                pairs.add(tuple(sorted((a.id, b.id))))
    return pairs


def adoption_verdict(
    baseline: Profile, candidate: Profile
) -> AdoptionVerdict:
    """May the automatically collected evidence be adopted?

    Both profiles are learned from the SAME human labels (auto entries
    carry none), so only the color models differ and the comparison is
    like for like on evidence the collector cannot influence.

    Rejected when, for any bin the baseline could train:
    - it can no longer be trained at all;
    - its calibration stopped separating;
    - a new pair of overlapping hue bands appeared (a pre-existing pair
      is the status quo, only growth counts);
    - a bin that ALREADY did not separate lost further ground. Its
      min/max are single-frame extrema with no robustness, so a
      ratchet on them would let one unlucky frame block the mechanism
      forever - but a bin with no slack left cannot afford any loss.
    A bin the baseline could not train and the candidate can is an
    improvement and never blocks.
    """
    base_stats = _bin_stats(baseline)
    cand_stats = _bin_stats(candidate)
    regressions: list[str] = []
    gaps: dict[str, float | None] = {}
    for bin_id, base in base_stats.items():
        cand = cand_stats.get(bin_id)
        if cand is None:
            regressions.append(f"{bin_id}: no longer trainable")
            gaps[bin_id] = None
            continue
        base_gap, cand_gap = _gap(base), _gap(cand)
        gaps[bin_id] = cand_gap
        if base.get("separable") and not cand.get("separable"):
            regressions.append(f"{bin_id}: stopped separating")
            continue
        if not base.get("separable", True):
            if (
                base_gap is not None
                and cand_gap is not None
                and cand_gap < base_gap
            ):
                regressions.append(
                    f"{bin_id}: already overlapping, and the overlap grew"
                )
    new_pairs = overlapping_hue_pairs(candidate) - overlapping_hue_pairs(
        baseline
    )
    for pair in sorted(new_pairs):
        regressions.append(f"{pair[0]}/{pair[1]}: hue bands now overlap")
    return AdoptionVerdict(not regressions, regressions, gaps)


def without_auto_evidence(store: CalibrationStore) -> CalibrationStore:
    """A copy of the store in which auto evidence is switched off.

    Uses the existing soft-exclusion semantics rather than deleting
    anything, so the baseline pass is exactly "what the user's own
    evidence alone would have produced".
    """
    from .store import store_from_dict, store_to_dict

    baseline = store_from_dict(store_to_dict(store))
    for entry in baseline.images:
        if entry.auto is not None:
            entry.excluded = True
    return baseline


def has_auto_evidence(store: CalibrationStore) -> bool:
    return any(
        e.auto is not None and not e.excluded for e in store.images
    )


def situation_of(entry) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The declared situation an entry belongs to.

    The capacity is enforced per situation - a run that recorded an
    empty yard must not be evicted by one that recorded a full one,
    they are different statements - so every count and every trim is
    per situation too.
    """
    return (tuple(sorted(entry.present)), tuple(sorted(entry.absent)))


def reservoir_by_situation(
    store: CalibrationStore,
) -> dict[tuple, list[tuple[str, tuple[float, float]]]]:
    grouped: dict[tuple, list[tuple[str, tuple[float, float]]]] = {}
    for entry in store.images:
        if entry.auto is None or entry.excluded:
            continue
        grouped.setdefault(situation_of(entry), []).append(
            (entry.path, (entry.auto.median_sat, entry.auto.median_val))
        )
    return grouped


def over_capacity_paths(store: CalibrationStore) -> list[str]:
    """Paths to set aside so every situation is back within capacity.

    Needed after a bulk restore, which knows nothing about which
    frames the reservoir had displaced over the run's history.
    """
    dropped: list[str] = []
    for retained in reservoir_by_situation(store).values():
        dropped.extend(reservoir_trim(retained))
    return dropped


def describe_reservoir(store: CalibrationStore) -> dict[str, Any]:
    """Progress numbers for the UI: how much of the light space the
    collection has covered, per declared situation."""
    grouped = reservoir_by_situation(store)
    situations = []
    for (present, absent), retained in sorted(grouped.items()):
        coords = [c for _p, c in retained]
        situations.append(
            {
                "present": list(present),
                "absent": list(absent),
                "retained": len(retained),
                "min_separation": (
                    None if len(coords) < 2 else _min_separation(coords)
                ),
                "sat_range": [
                    min(c[0] for c in coords),
                    max(c[0] for c in coords),
                ],
                "val_range": [
                    min(c[1] for c in coords),
                    max(c[1] for c in coords),
                ],
            }
        )
    return {
        "capacity_per_situation": AUTO_RESERVOIR_CAPACITY,
        "total_retained": sum(s["retained"] for s in situations),
        "situations": situations,
    }

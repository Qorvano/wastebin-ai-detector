"""Calibration store: the single source of truth for learning.

Holds the user-declared setup (ROI, bin list, working width, resample)
plus, per calibration image, the drawn lid-sample rectangles and the
presence labels. Profiles are always recomputed from this store in
full, never patched incrementally - changing a sample later can never
leave stale thresholds behind.

Schema v2 invariants (dynamic reconfiguration without data loss):

- Evidence is append-only. Nothing here ever deletes samples, labels or
  image entries; "removing" is always a flag (``excluded``, ``active``)
  or a learn-time exclusion, so any reconfiguration can be reverted
  without loss.
- Sample rectangles are stored relative to the upright FULL frame (the
  only coordinate system that survives ROI edits), together with the
  ROI they were drawn under (``SampleRect.roi``, the extraction grid
  that keeps their pixel content reproducible forever) and the bin's
  appearance epoch at draw time.
- Whether a stored datum counts for the CURRENT configuration is a pure
  geometry/stamp decision made by :func:`learning_view` at learn time -
  never by area thresholds, and never at storage time.

Image paths are stored relative to the store file, so the calibration
folder can be moved between machines.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .errors import CalibrationError, ProfileError
from .profile import KNOWN_RESAMPLE, REL_EPS, Rect, Roi

STORE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class BinDecl:
    """User-declared bin: identity plus lifecycle state.

    ``active``: retired bins keep their declaration (all stored evidence
    keeps referencing a declared id) but produce no model and no entity.
    ``appearance_epoch``: bumped when the physical lid changes color
    (municipality swap). Evidence carries the epoch it was recorded
    under; only current-epoch evidence feeds the color/area models,
    older epochs stay stored and become active again if the epoch is
    ever set back.
    """

    id: str
    name: str
    active: bool = True
    appearance_epoch: int = 0


@dataclass(frozen=True)
class SampleRect:
    """One drawn lid-sample rectangle.

    ``rect`` is relative to the upright full frame (0..1). ``roi`` is
    the region of interest that was configured when the rectangle was
    drawn: color learning re-extracts the sample pixels through exactly
    this grid, so the sample stays valid under any later ROI change.
    """

    rect: Rect
    roi: Roi
    epoch: int = 0


@dataclass
class ImageEntry:
    path: str
    samples: dict[str, list[SampleRect]] = field(default_factory=dict)
    present: list[str] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)
    # ROI configured at the last set_labels call: a label asserts
    # "visible / not visible inside THIS crop of THIS frame".
    label_roi: Roi | None = None
    # bin id -> bin.appearance_epoch at the last labeling of that bin.
    label_epoch: dict[str, int] = field(default_factory=dict)
    # Scene->frame mapping generation this file was captured under.
    view_epoch: int = 0
    # Soft forget: excluded entries keep all their data but contribute
    # nothing to learning until restored.
    excluded: bool = False


@dataclass
class CalibrationStore:
    roi: Roi
    working_width: int | None
    resample: str
    bins: list[BinDecl]
    images: list[ImageEntry] = field(default_factory=list)
    schema_version: int = STORE_SCHEMA_VERSION
    # Current scene->frame mapping generation. Bumped when the camera
    # was swapped/re-aimed with a changed field of view.
    view_epoch: int = 0
    # filename -> view epoch for archive files captured before a view
    # bump but not yet materialized as ImageEntry (entries only come
    # into existence at the first sample/label). Popped on ensure_image.
    capture_epochs: dict[str, int] = field(default_factory=dict)

    # -- queries ---------------------------------------------------------

    def bin_ids(self) -> list[str]:
        return [b.id for b in self.bins]

    def active_bin_ids(self) -> list[str]:
        return [b.id for b in self.bins if b.active]

    def get_bin(self, bin_id: str) -> BinDecl | None:
        for b in self.bins:
            if b.id == bin_id:
                return b
        return None

    def get_image(self, path: str) -> ImageEntry | None:
        for entry in self.images:
            if entry.path == path:
                return entry
        return None

    # -- mutations -------------------------------------------------------

    def ensure_image(self, path: str) -> ImageEntry:
        entry = self.get_image(path)
        if entry is None:
            entry = ImageEntry(
                path=path,
                view_epoch=self.capture_epochs.pop(path, self.view_epoch),
            )
            self.images.append(entry)
        return entry

    def add_sample(self, path: str, bin_id: str, rect: Rect) -> None:
        """Record a lid-sample rectangle in FULL-FRAME coordinates."""
        decl = self.get_bin(bin_id)
        if decl is None:
            raise CalibrationError(
                f"unknown bin id {bin_id!r}; declared: {self.bin_ids()}"
            )
        if not decl.active:
            raise CalibrationError(
                f"bin {bin_id!r} is retired - reactivate it before adding samples"
            )
        rect = clamp_unit_rect(rect)
        # The stamped ROI is the extraction grid, so it must contain the
        # rect. A rect outside the current ROI is legal input (it feeds
        # color learning immediately and area learning once the ROI
        # covers it); its grid is then the full frame - the only other
        # crop that is guaranteed to contain it (pure geometry).
        grid = (
            self.roi
            if image_rect_in_roi(rect, self.roi) is not None
            else Roi(x=0.0, y=0.0, w=1.0, h=1.0)
        )
        entry = self.ensure_image(path)
        entry.samples.setdefault(bin_id, []).append(
            SampleRect(rect=rect, roi=grid, epoch=decl.appearance_epoch)
        )

    def forget_image(self, path: str) -> bool:
        """Exclude an image from learning WITHOUT deleting anything.

        The entry, its samples and labels all stay in the store (and the
        archived file stays on disk); the image simply stops feeding the
        models until :meth:`restore_image`. The recovery path for
        samples drawn on the wrong spot or on an overexposed lid.
        Returns True if an entry existed.
        """
        entry = self.get_image(path)
        if entry is None:
            return False
        entry.excluded = True
        return True

    def restore_image(self, path: str) -> bool:
        """Undo :meth:`forget_image`. Returns True if an entry existed."""
        entry = self.get_image(path)
        if entry is None:
            return False
        entry.excluded = False
        return True

    def set_labels(
        self,
        path: str,
        present: list[str] | None = None,
        absent: list[str] | None = None,
    ) -> None:
        present = list(present or [])
        absent = list(absent or [])
        unknown = [b for b in present + absent if b not in self.bin_ids()]
        if unknown:
            raise CalibrationError(
                f"unknown bin ids {unknown}; declared: {self.bin_ids()}"
            )
        retired = [
            b for b in present + absent
            if (decl := self.get_bin(b)) is not None and not decl.active
        ]
        if retired:
            raise CalibrationError(
                f"bins {retired} are retired - reactivate them before labeling"
            )
        conflict = set(present) & set(absent)
        if conflict:
            raise CalibrationError(
                f"bins labeled both present and absent for {path}: {sorted(conflict)}"
            )
        entry = self.ensure_image(path)
        for bin_id in present:
            if bin_id not in entry.present:
                entry.present.append(bin_id)
            if bin_id in entry.absent:
                entry.absent.remove(bin_id)
        for bin_id in absent:
            if bin_id not in entry.absent:
                entry.absent.append(bin_id)
            if bin_id in entry.present:
                entry.present.remove(bin_id)
        # Labeling (re-)asserts the claim under the CURRENT view: stamp
        # the crop it was made under and, per touched bin, the bin's
        # current appearance epoch.
        entry.label_roi = self.roi
        for bin_id in present + absent:
            decl = self.get_bin(bin_id)
            assert decl is not None  # validated above
            entry.label_epoch[bin_id] = decl.appearance_epoch

    def confirm_image_view(self, path: str) -> bool:
        """Re-assert an entry's labels under the CURRENT view and ROI.

        For images the user has visually re-checked after a ROI edit or
        camera change: restamps ``label_roi`` and ``view_epoch`` without
        touching the labels themselves. Deliberately does NOT touch the
        per-bin appearance epochs: the archived pixels still show the
        lid as it looked back then, so appearance-stale evidence must
        stay dormant (re-labeling the image is the explicit way to
        re-assert it). Returns True if the entry exists and carries
        labels.
        """
        entry = self.get_image(path)
        if entry is None or not (entry.present or entry.absent):
            return False
        entry.label_roi = self.roi
        entry.view_epoch = self.view_epoch
        return True

    def mark_bin_appearance_changed(self, bin_id: str) -> int:
        """Bump a bin's appearance epoch (lid color swap). Returns it."""
        decl = self.get_bin(bin_id)
        if decl is None:
            raise CalibrationError(
                f"unknown bin id {bin_id!r}; declared: {self.bin_ids()}"
            )
        new = replace(decl, appearance_epoch=decl.appearance_epoch + 1)
        self.bins[self.bins.index(decl)] = new
        return new.appearance_epoch

    def bump_view_epoch(self, unmaterialized: list[str]) -> int:
        """Advance the view epoch (camera swapped/re-aimed).

        ``unmaterialized`` lists archive filenames without an ImageEntry
        yet; they were captured under the OLD view and are stamped so a
        later label attaches the correct epoch.
        """
        for name in unmaterialized:
            if self.get_image(name) is None:
                self.capture_epochs.setdefault(name, self.view_epoch)
        self.view_epoch += 1
        return self.view_epoch

    # -- geometry --------------------------------------------------------

    def image_rect_to_roi_rect(self, rect: Rect) -> Rect:
        """Convert an image-relative rectangle into ROI-relative coordinates.

        Pure geometry (no thresholds). Raises when the rectangle is not
        inside the current ROI.
        """
        converted = image_rect_in_roi(rect, self.roi)
        if converted is None:
            raise CalibrationError(
                f"sample rect {rect} lies outside the ROI {self.roi} - "
                "samples must be drawn inside the region of interest"
            )
        return converted


# -- pure geometry helpers (shared by migration, services, learning) ----


def clamp_unit_rect(rect: Rect) -> Rect:
    """Validate a full-frame rectangle and clamp FP drift into [0, 1]."""
    if not (
        -REL_EPS <= rect.x
        and rect.x + rect.w <= 1.0 + REL_EPS
        and -REL_EPS <= rect.y
        and rect.y + rect.h <= 1.0 + REL_EPS
        and rect.w > 0.0
        and rect.h > 0.0
    ):
        raise CalibrationError(
            f"rect {rect} lies outside the image frame [0, 1] or is empty"
        )
    x = min(max(rect.x, 0.0), 1.0)
    y = min(max(rect.y, 0.0), 1.0)
    return Rect(x=x, y=y, w=min(rect.w, 1.0 - x), h=min(rect.h, 1.0 - y))


def roi_rect_to_image_rect(rect: Rect, roi: Roi) -> Rect:
    """Affine ROI-relative -> image-relative (exact inverse of the
    image->ROI conversion; same REL_EPS clamp policy)."""
    return clamp_unit_rect(
        Rect(
            x=roi.x + rect.x * roi.w,
            y=roi.y + rect.y * roi.h,
            w=rect.w * roi.w,
            h=rect.h * roi.h,
        )
    )


def image_rect_in_roi(rect: Rect, roi: Roi) -> Rect | None:
    """Image-relative rect -> ROI-relative, or None if not fully inside.

    Containment uses the shared REL_EPS policy; the result is clamped
    so stored/derived rects are exactly in [0, 1].
    """
    x = (rect.x - roi.x) / roi.w
    y = (rect.y - roi.y) / roi.h
    w = rect.w / roi.w
    h = rect.h / roi.h
    if not (
        -REL_EPS <= x
        and x + w <= 1.0 + REL_EPS
        and -REL_EPS <= y
        and y + h <= 1.0 + REL_EPS
    ):
        return None
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    return Rect(x=x, y=y, w=min(w, 1.0 - x), h=min(h, 1.0 - y))


def roi_contains(outer: Roi, inner: Roi) -> bool:
    """Is ``inner`` fully inside ``outer`` (within REL_EPS)?"""
    return (
        inner.x >= outer.x - REL_EPS
        and inner.y >= outer.y - REL_EPS
        and inner.x + inner.w <= outer.x + outer.w + REL_EPS
        and inner.y + inner.h <= outer.y + outer.h + REL_EPS
    )


def roi_equal(a: Roi, b: Roi) -> bool:
    return (
        abs(a.x - b.x) <= REL_EPS
        and abs(a.y - b.y) <= REL_EPS
        and abs(a.w - b.w) <= REL_EPS
        and abs(a.h - b.h) <= REL_EPS
    )


# -- learning view ------------------------------------------------------


def learning_view(
    store: CalibrationStore,
) -> tuple[CalibrationStore, list[str]]:
    """The learn-time filter: which stored evidence counts RIGHT NOW.

    Returns a deep, filtered copy shaped for ``learn_profile`` plus
    warnings describing everything that was set aside. All decisions
    are pure geometry or exact id/epoch comparisons - never an area
    threshold (that would be circular: excluding evidence based on the
    measurement it is supposed to anchor).

    Rules per active bin:
    - color samples count iff their epoch matches the bin's current
      appearance epoch (extraction later runs through their own stored
      draw-time ROI, so ROI and view changes never invalidate them);
    - a PRESENT label counts iff its epoch matches, the image's view
      epoch is current, and either the labeling-time ROI is contained
      in the current ROI (growing the ROI keeps the lid inside) or
      every stored sample rect of that bin in that image lies fully
      inside the current ROI (the rects are the recorded proof of where
      the lid is in that frame);
    - an ABSENT label counts iff the image's view epoch is current and
      the current ROI is contained in the labeling-time ROI (a subset
      of a bin-free region is bin-free). Absent labels are appearance-
      epoch independent: background stays background under any lid
      color, so recoloring a bin keeps its negative evidence.
    - excluded images and retired bins contribute nothing.
    """
    warnings: list[str] = []
    active = [b for b in store.bins if b.active]
    epoch_of = {b.id: b.appearance_epoch for b in active}
    n_excluded_images = 0
    n_stale_samples = 0
    stale_present: list[str] = []
    stale_absent: list[str] = []

    images: list[ImageEntry] = []
    for entry in store.images:
        if entry.excluded:
            n_excluded_images += 1
            continue
        view_current = entry.view_epoch == store.view_epoch
        samples: dict[str, list[SampleRect]] = {}
        for bin_id, rects in entry.samples.items():
            if bin_id not in epoch_of:
                continue
            kept = [r for r in rects if r.epoch == epoch_of[bin_id]]
            n_stale_samples += len(rects) - len(kept)
            if kept:
                samples[bin_id] = list(kept)
        present: list[str] = []
        for bin_id in entry.present:
            if bin_id not in epoch_of:
                continue
            # Rect fallback over ALL stored rects of the bin in this
            # image, regardless of appearance epoch: lid POSITION is
            # epoch-independent, so an old-color rect outside the
            # current ROI still proves part of the lid lies outside
            # (veto), and an old-color rect inside still proves where
            # the lid is (usable).
            all_rects = entry.samples.get(bin_id, [])
            usable = (
                entry.label_epoch.get(bin_id) == epoch_of[bin_id]
                and view_current
                and (
                    (
                        entry.label_roi is not None
                        and roi_contains(store.roi, entry.label_roi)
                    )
                    or (
                        bool(all_rects)
                        and all(
                            image_rect_in_roi(r.rect, store.roi) is not None
                            for r in all_rects
                        )
                    )
                )
            )
            if usable:
                present.append(bin_id)
            else:
                stale_present.append(f"{entry.path}:{bin_id}")
        absent: list[str] = []
        for bin_id in entry.absent:
            if bin_id not in epoch_of:
                continue
            usable = (
                view_current
                and entry.label_roi is not None
                and roi_contains(entry.label_roi, store.roi)
            )
            if usable:
                absent.append(bin_id)
            else:
                stale_absent.append(f"{entry.path}:{bin_id}")
        images.append(
            ImageEntry(
                path=entry.path,
                samples=samples,
                present=present,
                absent=absent,
                label_roi=entry.label_roi,
                label_epoch=dict(entry.label_epoch),
                view_epoch=entry.view_epoch,
            )
        )

    if n_excluded_images:
        warnings.append(
            f"{n_excluded_images} calibration image(s) are excluded from "
            "training (restore_image reverses this)"
        )
    if n_stale_samples:
        warnings.append(
            f"{n_stale_samples} sample rectangle(s) belong to an older "
            "appearance epoch and are dormant (kept in the store)"
        )
    if stale_present:
        warnings.append(
            "present labels set aside (labeled under a different view/"
            f"crop/appearance; re-confirm or re-label to reuse): "
            f"{', '.join(sorted(stale_present))}"
        )
    if stale_absent:
        warnings.append(
            "absent labels set aside (current region extends beyond the "
            f"labeled crop or view changed): {', '.join(sorted(stale_absent))}"
        )
    view = CalibrationStore(
        roi=store.roi,
        working_width=store.working_width,
        resample=store.resample,
        bins=[replace(b) for b in active],
        images=images,
        schema_version=store.schema_version,
        view_epoch=store.view_epoch,
    )
    return view, warnings


# -- validation ---------------------------------------------------------


def _validate_unit_span(what: str, lo: float, span: float) -> None:
    if lo < -REL_EPS or span <= 0.0 or lo + span > 1.0 + REL_EPS:
        raise ProfileError(
            f"{what}-range [{lo}, {lo + span}] outside [0, 1] or empty"
        )


def _validate_roi(what: str, roi: Roi) -> None:
    _validate_unit_span(f"{what} x", roi.x, roi.w)
    _validate_unit_span(f"{what} y", roi.y, roi.h)


def validate_store(store: CalibrationStore) -> None:
    if store.schema_version != STORE_SCHEMA_VERSION:
        raise ProfileError(
            f"unsupported store schema_version {store.schema_version} "
            f"(supported: {STORE_SCHEMA_VERSION})"
        )
    if store.resample not in KNOWN_RESAMPLE:
        raise ProfileError(f"unknown resample {store.resample!r}")
    if store.working_width is not None and store.working_width <= 0:
        raise ProfileError(f"working_width must be positive, got {store.working_width}")
    _validate_roi("ROI", store.roi)
    if store.view_epoch < 0:
        raise ProfileError(f"negative view_epoch {store.view_epoch}")
    for name, epoch in store.capture_epochs.items():
        if not 0 <= epoch < store.view_epoch:
            raise ProfileError(
                f"capture epoch {epoch} for {name!r} outside "
                f"[0, {store.view_epoch})"
            )
    if not store.bins:
        raise ProfileError("store declares no bins")
    epoch_of: dict[str, int] = {}
    for b in store.bins:
        if not b.id:
            raise ProfileError("bin with empty id")
        if b.id in epoch_of:
            raise ProfileError(f"duplicate bin id {b.id!r}")
        if b.appearance_epoch < 0:
            raise ProfileError(f"bin {b.id}: negative appearance_epoch")
        epoch_of[b.id] = b.appearance_epoch
    declared = set(epoch_of)
    paths: set[str] = set()
    for entry in store.images:
        if entry.path in paths:
            raise ProfileError(f"duplicate image entry {entry.path!r}")
        paths.add(entry.path)
        # Dangling bin references (hand-edited/foreign store files) must
        # fail loudly here - learn silently skips unmatched ids, which
        # would quietly weaken the learned profile.
        unknown = sorted(
            (
                set(entry.samples)
                | set(entry.present)
                | set(entry.absent)
                | set(entry.label_epoch)
            )
            - declared
        )
        if unknown:
            raise ProfileError(
                f"image {entry.path}: references undeclared bin ids "
                f"{unknown}; declared: {sorted(declared)}"
            )
        conflict = set(entry.present) & set(entry.absent)
        if conflict:
            raise ProfileError(
                f"image {entry.path}: bins labeled present AND absent: {sorted(conflict)}"
            )
        if not 0 <= entry.view_epoch <= store.view_epoch:
            raise ProfileError(
                f"image {entry.path}: view_epoch {entry.view_epoch} outside "
                f"[0, {store.view_epoch}]"
            )
        if entry.label_roi is not None:
            _validate_roi(f"image {entry.path} label_roi", entry.label_roi)
        for bin_id, epoch in entry.label_epoch.items():
            if not 0 <= epoch <= epoch_of[bin_id]:
                raise ProfileError(
                    f"image {entry.path}: label epoch {epoch} for {bin_id} "
                    f"outside [0, {epoch_of[bin_id]}]"
                )
        for bin_id, rects in entry.samples.items():
            for r in rects:
                _validate_unit_span(
                    f"image {entry.path} sample({bin_id}) x", r.rect.x, r.rect.w
                )
                _validate_unit_span(
                    f"image {entry.path} sample({bin_id}) y", r.rect.y, r.rect.h
                )
                _validate_roi(f"image {entry.path} sample({bin_id}) roi", r.roi)
                if not 0 <= r.epoch <= epoch_of[bin_id]:
                    raise ProfileError(
                        f"image {entry.path}: sample epoch {r.epoch} for "
                        f"{bin_id} outside [0, {epoch_of[bin_id]}]"
                    )
                # The rect must lie inside its own draw-time ROI - that
                # is the grid its pixels are extracted through.
                if image_rect_in_roi(r.rect, r.roi) is None:
                    raise ProfileError(
                        f"image {entry.path}: sample rect {r.rect} for "
                        f"{bin_id} lies outside its draw-time ROI {r.roi}"
                    )


# -- migration ----------------------------------------------------------


def migrate_store_dict_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Lossless, deterministic v1 -> v2 store-dict migration.

    In v1 the ROI was immutable, so every stored ROI-relative sample
    rect was drawn under exactly ``data["roi"]`` - which makes the
    image-space conversion exact (pure affine, no information created
    or lost). Idempotent by construction: only runs on
    ``schema_version == 1`` input.
    """
    try:
        roi = Roi(**{k: float(data["roi"][k]) for k in ("x", "y", "w", "h")})
        out = {
            "schema_version": STORE_SCHEMA_VERSION,
            "roi": dict(data["roi"]),
            "working_width": data["working_width"],
            "resample": data["resample"],
            "view_epoch": 0,
            "capture_epochs": {},
            "bins": [
                {
                    "id": b["id"],
                    "name": b["name"],
                    "active": True,
                    "appearance_epoch": 0,
                }
                for b in data["bins"]
            ],
            "images": [],
        }
        for e in data.get("images", []):
            present = [str(b) for b in e.get("present", [])]
            absent = [str(b) for b in e.get("absent", [])]
            out["images"].append(
                {
                    "path": e["path"],
                    "samples": {
                        bin_id: [
                            {
                                "rect": asdict(
                                    roi_rect_to_image_rect(
                                        Rect(
                                            **{
                                                k: float(r[k])
                                                for k in ("x", "y", "w", "h")
                                            }
                                        ),
                                        roi,
                                    )
                                ),
                                "roi": dict(data["roi"]),
                                "epoch": 0,
                            }
                            for r in rects
                            # A zero-area rect contains no pixels and
                            # therefore no evidence: dropping it is
                            # lossless, while keeping it would brick
                            # the whole store on validation (v1 never
                            # validated stored rects, so such rects
                            # exist in the wild).
                            if float(r["w"]) > 0.0 and float(r["h"]) > 0.0
                        ]
                        for bin_id, rects in e.get("samples", {}).items()
                    },
                    "present": present,
                    "absent": absent,
                    "label_roi": (
                        dict(data["roi"]) if (present or absent) else None
                    ),
                    "label_epoch": {b: 0 for b in present + absent},
                    "view_epoch": 0,
                    "excluded": False,
                }
            )
    except (KeyError, TypeError, ValueError, CalibrationError) as exc:
        raise ProfileError(
            f"cannot migrate v1 calibration store: {exc!r}"
        ) from exc
    return out


# -- serialization ------------------------------------------------------


def store_to_dict(store: CalibrationStore) -> dict[str, Any]:
    data = asdict(store)
    return {
        "schema_version": data["schema_version"],
        "roi": data["roi"],
        "working_width": data["working_width"],
        "resample": data["resample"],
        "view_epoch": data["view_epoch"],
        "capture_epochs": data["capture_epochs"],
        "bins": data["bins"],
        "images": data["images"],
    }


def _roi_from(data: dict[str, Any]) -> Roi:
    return Roi(**{k: float(data[k]) for k in ("x", "y", "w", "h")})


def _rect_from(data: dict[str, Any]) -> Rect:
    return Rect(**{k: float(data[k]) for k in ("x", "y", "w", "h")})


def store_from_dict(data: dict[str, Any]) -> CalibrationStore:
    if int(data.get("schema_version", 0)) == 1:
        data = migrate_store_dict_v1_to_v2(data)
    try:
        store = CalibrationStore(
            schema_version=int(data["schema_version"]),
            roi=_roi_from(data["roi"]),
            working_width=(
                None if data["working_width"] is None else int(data["working_width"])
            ),
            resample=str(data["resample"]),
            view_epoch=int(data.get("view_epoch", 0)),
            capture_epochs={
                str(k): int(v)
                for k, v in data.get("capture_epochs", {}).items()
            },
            bins=[
                BinDecl(
                    id=str(b["id"]),
                    name=str(b["name"]),
                    active=bool(b.get("active", True)),
                    appearance_epoch=int(b.get("appearance_epoch", 0)),
                )
                for b in data["bins"]
            ],
            images=[
                ImageEntry(
                    path=str(e["path"]),
                    samples={
                        str(bin_id): [
                            SampleRect(
                                rect=_rect_from(r["rect"]),
                                roi=_roi_from(r["roi"]),
                                epoch=int(r.get("epoch", 0)),
                            )
                            for r in rects
                        ]
                        for bin_id, rects in e.get("samples", {}).items()
                    },
                    present=[str(b) for b in e.get("present", [])],
                    absent=[str(b) for b in e.get("absent", [])],
                    label_roi=(
                        None
                        if e.get("label_roi") is None
                        else _roi_from(e["label_roi"])
                    ),
                    label_epoch={
                        str(k): int(v)
                        for k, v in e.get("label_epoch", {}).items()
                    },
                    view_epoch=int(e.get("view_epoch", 0)),
                    excluded=bool(e.get("excluded", False)),
                )
                for e in data.get("images", [])
            ],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProfileError(f"malformed calibration store: {exc!r}") from exc
    validate_store(store)
    return store


def load_store(path: str | Path) -> CalibrationStore:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot read calibration store {path}: {exc}") from exc
    return store_from_dict(data)


def save_store(store: CalibrationStore, path: str | Path) -> None:
    validate_store(store)
    Path(path).write_text(
        json.dumps(store_to_dict(store), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def resolve_image_path(store_path: str | Path, entry_path: str) -> Path:
    """Resolve an image path stored relative to the store file."""
    entry = Path(entry_path)
    if entry.is_absolute():
        return entry
    return Path(store_path).resolve().parent / entry

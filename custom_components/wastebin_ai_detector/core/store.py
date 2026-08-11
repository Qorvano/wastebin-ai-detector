"""Calibration store: the single source of truth for learning.

Holds the user-declared setup (ROI, bin list, working width, resample)
plus, per calibration image, the drawn lid-sample rectangles and the
presence labels. Profiles are always recomputed from this store in
full, never patched incrementally - changing a sample later can never
leave stale thresholds behind.

Image paths are stored relative to the store file, so the calibration
folder can be moved between machines.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import CalibrationError, ProfileError
from .profile import KNOWN_RESAMPLE, REL_EPS, Rect, Roi

STORE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BinDecl:
    """User-declared bin: identity only - everything else is learned."""

    id: str
    name: str


@dataclass
class ImageEntry:
    path: str
    samples: dict[str, list[Rect]] = field(default_factory=dict)
    present: list[str] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)


@dataclass
class CalibrationStore:
    roi: Roi
    working_width: int | None
    resample: str
    bins: list[BinDecl]
    images: list[ImageEntry] = field(default_factory=list)
    schema_version: int = STORE_SCHEMA_VERSION

    # -- queries ---------------------------------------------------------

    def bin_ids(self) -> list[str]:
        return [b.id for b in self.bins]

    def get_image(self, path: str) -> ImageEntry | None:
        for entry in self.images:
            if entry.path == path:
                return entry
        return None

    # -- mutations -------------------------------------------------------

    def ensure_image(self, path: str) -> ImageEntry:
        entry = self.get_image(path)
        if entry is None:
            entry = ImageEntry(path=path)
            self.images.append(entry)
        return entry

    def add_sample(self, path: str, bin_id: str, rect: Rect) -> None:
        if bin_id not in self.bin_ids():
            raise CalibrationError(
                f"unknown bin id {bin_id!r}; declared: {self.bin_ids()}"
            )
        entry = self.ensure_image(path)
        entry.samples.setdefault(bin_id, []).append(rect)

    def forget_image(self, path: str) -> bool:
        """Remove an image entry (all its samples and labels).

        The archived file itself stays on disk; only its calibration
        contribution is dropped. The recovery path for samples drawn on
        the wrong spot or on an overexposed lid. Returns True if an
        entry existed.
        """
        entry = self.get_image(path)
        if entry is None:
            return False
        self.images.remove(entry)
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

    # -- geometry --------------------------------------------------------

    def image_rect_to_roi_rect(self, rect: Rect) -> Rect:
        """Convert an image-relative rectangle into ROI-relative coordinates.

        Pure geometry (no thresholds): the CLI lets users give rectangles
        in full-image coordinates, storage is always ROI-relative.
        """
        roi = self.roi
        x = (rect.x - roi.x) / roi.w
        y = (rect.y - roi.y) / roi.h
        w = rect.w / roi.w
        h = rect.h / roi.h
        # Same FP tolerance as every other relative-coordinate check in
        # the pipeline (REL_EPS): a rect drawn flush with the ROI edge
        # arrives as e.g. 1.0000000000000002 after the division above.
        if not (
            -REL_EPS <= x
            and x + w <= 1.0 + REL_EPS
            and -REL_EPS <= y
            and y + h <= 1.0 + REL_EPS
        ):
            raise CalibrationError(
                f"sample rect {rect} lies outside the ROI {roi} - "
                "samples must be drawn inside the region of interest"
            )
        # Clamp the FP drift away so stored rects are exactly in [0, 1].
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        w = min(w, 1.0 - x)
        h = min(h, 1.0 - y)
        return Rect(x=x, y=y, w=w, h=h)


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
    for name, lo, span in (
        ("x", store.roi.x, store.roi.w),
        ("y", store.roi.y, store.roi.h),
    ):
        if lo < -REL_EPS or span <= 0.0 or lo + span > 1.0 + REL_EPS:
            raise ProfileError(
                f"ROI {name}-range [{lo}, {lo + span}] outside [0, 1] or empty"
            )
    if not store.bins:
        raise ProfileError("store declares no bins")
    seen: set[str] = set()
    for b in store.bins:
        if not b.id:
            raise ProfileError("bin with empty id")
        if b.id in seen:
            raise ProfileError(f"duplicate bin id {b.id!r}")
        seen.add(b.id)
    paths: set[str] = set()
    for entry in store.images:
        if entry.path in paths:
            raise ProfileError(f"duplicate image entry {entry.path!r}")
        paths.add(entry.path)
        # Dangling bin references (hand-edited/foreign store files) must
        # fail loudly here - learn silently skips unmatched ids, which
        # would quietly weaken the learned profile.
        unknown = sorted(
            (set(entry.samples) | set(entry.present) | set(entry.absent)) - seen
        )
        if unknown:
            raise ProfileError(
                f"image {entry.path}: references undeclared bin ids "
                f"{unknown}; declared: {sorted(seen)}"
            )
        conflict = set(entry.present) & set(entry.absent)
        if conflict:
            raise ProfileError(
                f"image {entry.path}: bins labeled present AND absent: {sorted(conflict)}"
            )


def store_to_dict(store: CalibrationStore) -> dict[str, Any]:
    data = asdict(store)
    return {
        "schema_version": data["schema_version"],
        "roi": data["roi"],
        "working_width": data["working_width"],
        "resample": data["resample"],
        "bins": data["bins"],
        "images": data["images"],
    }


def store_from_dict(data: dict[str, Any]) -> CalibrationStore:
    try:
        store = CalibrationStore(
            schema_version=int(data["schema_version"]),
            roi=Roi(**{k: float(data["roi"][k]) for k in ("x", "y", "w", "h")}),
            working_width=(
                None if data["working_width"] is None else int(data["working_width"])
            ),
            resample=str(data["resample"]),
            bins=[BinDecl(id=str(b["id"]), name=str(b["name"])) for b in data["bins"]],
            images=[
                ImageEntry(
                    path=str(e["path"]),
                    samples={
                        str(bin_id): [
                            Rect(**{k: float(r[k]) for k in ("x", "y", "w", "h")})
                            for r in rects
                        ]
                        for bin_id, rects in e.get("samples", {}).items()
                    },
                    present=[str(b) for b in e.get("present", [])],
                    absent=[str(b) for b in e.get("absent", [])],
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

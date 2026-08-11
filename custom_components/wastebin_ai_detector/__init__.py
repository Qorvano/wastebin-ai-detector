"""Wastebin AI Detector — camera-based waste-bin presence detection.

Phase 1 ships the self-contained detection core (``core/``) plus the
offline calibration CLI (``tools/calibrate.py`` in the repository). The
Home Assistant wiring — config flow, update coordinator and one
``binary_sensor`` per bin — lands in phase 2; this setup is
intentionally a no-op so early installs are safe.
"""

from __future__ import annotations

from .const import DOMAIN

__all__ = ["DOMAIN"]


async def async_setup(hass, config) -> bool:  # type: ignore[no-untyped-def]
    """Phase 1: nothing to set up yet."""
    return True

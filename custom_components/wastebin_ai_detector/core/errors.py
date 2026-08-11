"""Exception hierarchy for the Wastebin AI Detector core.

The core never guesses: invalid input raises one of these instead of
returning a fabricated result. The Home Assistant layer (phase 2)
decides how to surface them (keep last state, mark unavailable, ...).
"""


class WastebinError(Exception):
    """Base class for all core errors."""


class ImageLoadError(WastebinError):
    """An image file could not be opened or decoded."""


class ProfileError(WastebinError):
    """A profile file is structurally invalid or has an unsupported version."""


class RoiError(WastebinError):
    """A region of interest is out of range or degenerates to zero pixels."""


class CalibrationError(WastebinError):
    """Calibration data is missing, inconsistent or has no discriminative power."""

"""Wastebin AI Detector - self-contained detection core.

Pure numpy + Pillow, no Home Assistant imports: everything in this
package can be exercised offline (calibration CLI, tests) and is reused
unchanged by the Home Assistant integration layer.
"""

from .ccl import largest_component_area
from .color import circular_dist_deg, circular_mean_deg, rgb_to_hsv
from .detect import BinResult, DetectionResult, bin_mask, detect, detect_file
from .errors import (
    CalibrationError,
    ImageLoadError,
    ProfileError,
    RoiError,
    WastebinError,
)
from .imageio import (
    extract_working_roi,
    load_image_rgb,
    load_image_rgb_bytes,
    roi_to_pixels,
)
from .learn import learn_area_threshold, learn_color_model, learn_profile
from .profile import (
    SCHEMA_VERSION,
    BinModel,
    Profile,
    Rect,
    Roi,
    load_profile,
    profile_from_dict,
    profile_to_dict,
    save_profile,
    validate_profile,
)
from .store import (
    BinDecl,
    CalibrationStore,
    ImageEntry,
    load_store,
    resolve_image_path,
    save_store,
    store_from_dict,
    store_to_dict,
    validate_store,
)

__all__ = [
    "BinDecl",
    "BinModel",
    "BinResult",
    "CalibrationError",
    "CalibrationStore",
    "DetectionResult",
    "ImageEntry",
    "ImageLoadError",
    "Profile",
    "ProfileError",
    "Rect",
    "Roi",
    "RoiError",
    "SCHEMA_VERSION",
    "WastebinError",
    "bin_mask",
    "circular_dist_deg",
    "circular_mean_deg",
    "detect",
    "detect_file",
    "extract_working_roi",
    "largest_component_area",
    "learn_area_threshold",
    "learn_color_model",
    "learn_profile",
    "load_image_rgb",
    "load_image_rgb_bytes",
    "load_profile",
    "load_store",
    "profile_from_dict",
    "profile_to_dict",
    "resolve_image_path",
    "rgb_to_hsv",
    "roi_to_pixels",
    "save_profile",
    "save_store",
    "store_from_dict",
    "store_to_dict",
    "validate_profile",
    "validate_store",
]

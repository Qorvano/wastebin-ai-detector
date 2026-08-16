"""Constants for the Wastebin AI Detector integration."""

from __future__ import annotations

DOMAIN = "wastebin_ai_detector"

CONF_CAMERA = "camera_entity"
CONF_ROI_X = "roi_x"
CONF_ROI_Y = "roi_y"
CONF_ROI_W = "roi_w"
CONF_ROI_H = "roi_h"
CONF_WORKING_WIDTH = "working_width"
CONF_BINS = "bins"
CONF_BIN_ID = "id"
CONF_BIN_NAME = "name"
CONF_SCAN_INTERVAL = "scan_interval_minutes"
CONF_CAPTURE_INTERVAL = "capture_interval_minutes"
CONF_CONFIRM_SCANS = "confirm_scans"
# Monotone counter in entry.data: bumped when the user declares the
# scene->frame mapping changed (camera swapped/re-aimed); reconciled
# into the store's view_epoch on setup.
CONF_VIEW_GENERATION = "view_generation"
CONF_BIN_ACTIVE = "active"
# Optional polygon region (list of rings, image-relative [x, y] pairs)
# refining the roi bbox; absent/None = plain rectangle region.
CONF_ROI_POLYGONS = "roi_polygons"

# Config DEFAULTS. These are user-changeable schema defaults (a
# legitimate value source), not hidden in-code thresholds:
# - 640 px analysis width: a lid spans tens of pixels at typical yard
#   camera fields of view while keeping the numpy work cheap on a Pi.
# - 15 min detection cadence: bins move on human timescales.
# - 30 min learning-capture cadence: collects lighting variance across
#   a day without flooding the archive (roughly 28 images/day).
DEFAULT_WORKING_WIDTH = 640
DEFAULT_SCAN_INTERVAL_MIN = 15
DEFAULT_CAPTURE_INTERVAL_MIN = 30
# 1 = clear evidence switches immediately (uncertain frames never
# switch, they hold). Raise to demand that many consecutive confident
# analyses before a state flip is accepted.
DEFAULT_CONFIRM_SCANS = 1
# UI upper bound: at the default 15-minute cadence, 20 confirmations
# already mean a five-hour worst-case flip latency; anything beyond
# would mask same-day events entirely instead of stabilizing them.
MAX_CONFIRM_SCANS = 20

SERVICE_RELEARN = "relearn"
SERVICE_CAPTURE = "capture_snapshot"
SERVICE_ADD_SAMPLE = "add_sample"
SERVICE_LABEL_IMAGE = "label_image"
SERVICE_FORGET_IMAGE = "forget_image"
SERVICE_RESTORE_IMAGE = "restore_image"
SERVICE_RECONFIRM_IMAGES = "reconfirm_images"
SERVICE_START_LEARNING = "start_learning"
SERVICE_STOP_LEARNING = "stop_learning"
SERVICE_DISCARD_AUTO = "discard_auto_evidence"
SERVICE_RESTORE_AUTO = "restore_auto_evidence"
SERVICE_MARK_BIN_CHANGED = "mark_bin_appearance_changed"
SERVICE_SET_ROI = "set_roi"

ATTR_ENTRY_ID = "entry_id"
ATTR_FILENAME = "filename"
ATTR_FILENAMES = "filenames"
ATTR_BIN = "bin"
ATTR_RECT = "rect"
ATTR_SPACE = "space"
ATTR_PRESENT = "present"
ATTR_ABSENT = "absent"
ATTR_AUTO_RELEARN = "auto_relearn"

STORAGE_VERSION = 1

# Closed set of analysis outcomes, shown by the enum status sensor.
STATUS_OUTCOMES = (
    "no_run_yet",
    "ok",
    "hold_greyscale",
    "hold_overexposure",
    "hold_greyscale_and_overexposure",
    "hold_frame_integrity",
    "ambiguous_cold_start",
    "camera_error",
    "detect_error",
    "not_calibrated",
    "watchdog_timeout",
    "previous_run_still_busy",
)

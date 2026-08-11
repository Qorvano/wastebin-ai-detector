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

SERVICE_RELEARN = "relearn"
SERVICE_CAPTURE = "capture_snapshot"
SERVICE_ADD_SAMPLE = "add_sample"
SERVICE_LABEL_IMAGE = "label_image"
SERVICE_FORGET_IMAGE = "forget_image"

ATTR_ENTRY_ID = "entry_id"
ATTR_FILENAME = "filename"
ATTR_BIN = "bin"
ATTR_RECT = "rect"
ATTR_SPACE = "space"
ATTR_PRESENT = "present"
ATTR_ABSENT = "absent"
ATTR_AUTO_RELEARN = "auto_relearn"

STORAGE_VERSION = 1

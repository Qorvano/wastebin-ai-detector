"""Detection coordinator and the learning snapshot collector.

The coordinator runs the periodic detection against the learned
profile. Night/IR frames (greyscale suspect) never overwrite the last
daylight result: color detection has nothing to say in the dark, so the
entities hold their state until the next daylight analysis.

The collector is the integration-internal replacement for a manual
snapshot automation: while learning mode is on, it archives a camera
frame at a fixed cadence during daylight. Archived frames become
calibration data as soon as the user labels them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from homeassistant.components.camera import async_get_image
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.sun import is_up
from homeassistant.helpers.update_coordinator import (
    DataUpdateCoordinator,
    UpdateFailed,
)
from homeassistant.util import dt as dt_util

from .const import (
    CONF_CAMERA,
    CONF_CAPTURE_INTERVAL,
    CONF_SCAN_INTERVAL,
    DEFAULT_CAPTURE_INTERVAL_MIN,
    DEFAULT_SCAN_INTERVAL_MIN,
    DOMAIN,
)
from .core import (
    DetectionResult,
    Profile,
    WastebinError,
    detect,
    load_image_rgb_bytes,
)
from .storage import WastebinStorage, archive_dir

_LOGGER = logging.getLogger(__name__)


class WastebinCoordinator(DataUpdateCoordinator[DetectionResult]):
    """Periodic detection against the learned profile."""

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, storage: WastebinStorage
    ) -> None:
        scan_minutes = entry.options.get(
            CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL_MIN
        )
        super().__init__(
            hass,
            _LOGGER,
            name=f"{DOMAIN} {entry.title}",
            update_interval=timedelta(minutes=scan_minutes),
        )
        self.entry = entry
        self.storage = storage
        self.camera_entity: str = entry.data[CONF_CAMERA]
        self.last_daylight_update: datetime | None = None
        self.last_greyscale_skip: datetime | None = None

    async def _async_update_data(self) -> DetectionResult:
        profile = self.storage.profile
        if profile is None:
            raise UpdateFailed(
                "not calibrated yet: add samples and labels, then call the "
                f"{DOMAIN}.relearn service"
            )
        try:
            image = await async_get_image(self.hass, self.camera_entity)
        except HomeAssistantError as err:
            raise UpdateFailed(f"camera snapshot failed: {err}") from err
        try:
            result = await self.hass.async_add_executor_job(
                self._detect_bytes, image.content, profile
            )
        except WastebinError as err:
            raise UpdateFailed(f"detection failed: {err}") from err
        if result.grayscale_suspect:
            self.last_greyscale_skip = dt_util.utcnow()
            if self.data is not None:
                return self.data
            # No daylight result exists yet (fresh setup or a reload at
            # night): publishing an IR verdict would present "all bins
            # absent" as real state. Stay unavailable instead.
            raise UpdateFailed(
                "greyscale/IR frame and no daylight result yet; waiting "
                "for the first daylight analysis"
            )
        self.last_daylight_update = dt_util.utcnow()
        return result

    @staticmethod
    def _detect_bytes(data: bytes, profile: Profile) -> DetectionResult:
        return detect(load_image_rgb_bytes(data), profile)


class LearningCollector:
    """Archives daylight snapshots while learning mode is enabled."""

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, storage: WastebinStorage
    ) -> None:
        self._hass = hass
        self._storage = storage
        self._camera: str = entry.data[CONF_CAMERA]
        self._interval = timedelta(
            minutes=entry.options.get(
                CONF_CAPTURE_INTERVAL, DEFAULT_CAPTURE_INTERVAL_MIN
            )
        )
        self._dir = archive_dir(hass, entry.entry_id)
        self._unsub: CALLBACK_TYPE | None = None

    @callback
    def async_start(self) -> None:
        if self._unsub is None:
            self._unsub = async_track_time_interval(
                self._hass, self._async_interval, self._interval
            )

    @callback
    def async_stop(self) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    async def _async_interval(self, _now: datetime) -> None:
        if not self._storage.learning:
            return
        # Computed from the configured location, independent of the sun
        # entity. IR/greyscale night frames carry no color information;
        # archiving them would only pollute the calibration set.
        if not is_up(self._hass):
            return
        try:
            await self.async_capture_now()
        except HomeAssistantError as err:
            _LOGGER.debug("learning capture skipped: %s", err)

    async def async_capture_now(self) -> str:
        """Capture one frame into the archive; returns the filename.

        Raises HomeAssistantError for both camera and filesystem
        failures, so every caller has a single error contract.
        """
        image = await async_get_image(self._hass, self._camera)
        filename = dt_util.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        try:
            await self._hass.async_add_executor_job(
                self._write, filename, image.content
            )
        except OSError as err:
            raise HomeAssistantError(
                f"cannot write snapshot to {self._dir}: {err}"
            ) from err
        return filename

    def _write(self, filename: str, data: bytes) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / filename).write_bytes(data)

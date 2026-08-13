"""Config and options flow for the Wastebin AI Detector."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers import selector
from homeassistant.util import slugify

from .const import (
    CONF_BIN_ACTIVE,
    CONF_ROI_POLYGONS,
    CONF_BIN_NAME,
    CONF_BINS,
    CONF_CAMERA,
    CONF_CAPTURE_INTERVAL,
    CONF_CONFIRM_SCANS,
    CONF_ROI_H,
    CONF_ROI_W,
    CONF_ROI_X,
    CONF_ROI_Y,
    CONF_SCAN_INTERVAL,
    CONF_VIEW_GENERATION,
    CONF_WORKING_WIDTH,
    DEFAULT_CAPTURE_INTERVAL_MIN,
    DEFAULT_CONFIRM_SCANS,
    DEFAULT_SCAN_INTERVAL_MIN,
    DEFAULT_WORKING_WIDTH,
    DOMAIN,
    MAX_CONFIRM_SCANS,
)

CONF_ADD_ANOTHER = "add_another"
CONF_VIEW_UNCHANGED = "view_unchanged"
ATTR_BIN_SELECT = "bin"

_REL_COORD = selector.NumberSelector(
    selector.NumberSelectorConfig(
        min=0.0, max=1.0, step=0.001, mode=selector.NumberSelectorMode.BOX
    )
)


def _roi_error(x: float, y: float, w: float, h: float) -> str | None:
    """Shared ROI validation for setup and reconfigure (pure geometry)."""
    if w <= 0 or h <= 0 or x + w > 1.0 or y + h > 1.0:
        return "invalid_roi"
    return None


def _active_bins(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        b for b in data[CONF_BINS] if b.get(CONF_BIN_ACTIVE, True)
    ]


def _retired_bins(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        b for b in data[CONF_BINS] if not b.get(CONF_BIN_ACTIVE, True)
    ]


class WastebinConfigFlow(ConfigFlow, domain=DOMAIN):
    """Guided setup: camera, region of interest, bins.

    Reconfiguration (camera swap, ROI edit, bin lifecycle) runs through
    ``async_step_reconfigure``; the fresh-install steps below stay
    unchanged. Evidence in the calibration store is never touched here:
    entry.data only declares the ACTIVE configuration, the reconcile on
    setup translates it into store flags without deleting anything.
    """

    VERSION = 1
    MINOR_VERSION = 2

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._bins: list[dict[str, str]] = []

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_area()
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_CAMERA): selector.EntitySelector(
                        selector.EntitySelectorConfig(domain="camera")
                    )
                }
            ),
        )

    async def async_step_area(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            roi_error = _roi_error(
                user_input[CONF_ROI_X],
                user_input[CONF_ROI_Y],
                user_input[CONF_ROI_W],
                user_input[CONF_ROI_H],
            )
            if roi_error:
                errors["base"] = roi_error
            else:
                self._data.update(user_input)
                return await self.async_step_bin()
        return self.async_show_form(
            step_id="area",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_ROI_X, default=0.0): _REL_COORD,
                    vol.Required(CONF_ROI_Y, default=0.0): _REL_COORD,
                    vol.Required(CONF_ROI_W, default=1.0): _REL_COORD,
                    vol.Required(CONF_ROI_H, default=1.0): _REL_COORD,
                    vol.Required(
                        CONF_WORKING_WIDTH, default=DEFAULT_WORKING_WIDTH
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement="px",
                        )
                    ),
                }
            ),
            errors=errors,
        )

    async def async_step_bin(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            name = user_input[CONF_BIN_NAME].strip()
            bin_id = slugify(name)
            if not bin_id:
                errors["base"] = "invalid_name"
            elif any(b["id"] == bin_id for b in self._bins):
                errors["base"] = "duplicate_bin"
            else:
                self._bins.append({"id": bin_id, "name": name})
                if user_input.get(CONF_ADD_ANOTHER):
                    return await self.async_step_bin()
                # Same camera + same region = same detector; a second
                # ROI on the same camera stays allowed.
                self._async_abort_entries_match(
                    {
                        CONF_CAMERA: self._data[CONF_CAMERA],
                        CONF_ROI_X: self._data[CONF_ROI_X],
                        CONF_ROI_Y: self._data[CONF_ROI_Y],
                        CONF_ROI_W: self._data[CONF_ROI_W],
                        CONF_ROI_H: self._data[CONF_ROI_H],
                    }
                )
                self._data[CONF_BINS] = self._bins
                self._data[CONF_WORKING_WIDTH] = int(
                    self._data[CONF_WORKING_WIDTH]
                )
                camera_state = self.hass.states.get(self._data[CONF_CAMERA])
                title = (
                    camera_state.name if camera_state else "Wastebin AI Detector"
                )
                return self.async_create_entry(title=title, data=self._data)
        return self.async_show_form(
            step_id="bin",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_BIN_NAME): selector.TextSelector(),
                    vol.Optional(
                        CONF_ADD_ANOTHER, default=False
                    ): selector.BooleanSelector(),
                }
            ),
            errors=errors,
            description_placeholders={
                "bins": ", ".join(b["name"] for b in self._bins) or "-"
            },
        )

    # -- reconfiguration -------------------------------------------------

    def _reconf_duplicate(
        self, camera: str, x: float, y: float, w: float, h: float
    ) -> bool:
        """Same camera + same region as ANOTHER entry = same detector.

        The fresh-install guard (_async_abort_entries_match) would match
        the entry being reconfigured itself, so the self-exclusion is
        explicit here.
        """
        own = self._get_reconfigure_entry().entry_id
        for other in self._async_current_entries():
            if other.entry_id == own:
                continue
            d = other.data
            if (
                d.get(CONF_CAMERA) == camera
                and d.get(CONF_ROI_POLYGONS) is None
                and d.get(CONF_ROI_X) == x
                and d.get(CONF_ROI_Y) == y
                and d.get(CONF_ROI_W) == w
                and d.get(CONF_ROI_H) == h
            ):
                return True
        return False

    def _finish_reconfigure(
        self, entry: ConfigEntry, data_updates: dict[str, Any]
    ) -> ConfigFlowResult:
        """Persist the change and finish; the entry's update listener
        schedules the single reload (the combined update-reload-abort
        helper would reload a second time and is deprecated alongside
        an update listener from HA 2026.12)."""
        self.hass.config_entries.async_update_entry(
            entry, data={**entry.data, **data_updates}
        )
        return self.async_abort(reason="reconfigure_successful")

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return self.async_show_menu(
            step_id="reconfigure",
            menu_options=[
                "reconf_camera",
                "reconf_area",
                "reconf_view_changed",
                "reconf_add_bin",
                "reconf_retire_bin",
                "reconf_reactivate_bin",
            ],
        )

    async def async_step_reconf_camera(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}
        if user_input is not None:
            camera = user_input[CONF_CAMERA]
            if self._reconf_duplicate(
                camera,
                entry.data[CONF_ROI_X],
                entry.data[CONF_ROI_Y],
                entry.data[CONF_ROI_W],
                entry.data[CONF_ROI_H],
            ):
                errors["base"] = "duplicate_detector"
            else:
                updates: dict[str, Any] = {CONF_CAMERA: camera}
                if not user_input[CONF_VIEW_UNCHANGED]:
                    updates[CONF_VIEW_GENERATION] = (
                        int(entry.data.get(CONF_VIEW_GENERATION, 0)) + 1
                    )
                return self._finish_reconfigure(
                    entry, data_updates=updates
                )
        return self.async_show_form(
            step_id="reconf_camera",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_CAMERA, default=entry.data[CONF_CAMERA]
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(domain="camera")
                    ),
                    vol.Required(
                        CONF_VIEW_UNCHANGED, default=True
                    ): selector.BooleanSelector(),
                }
            ),
            errors=errors,
        )

    async def async_step_reconf_view_changed(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Camera was physically re-aimed (bumped/re-mounted): the old
        area evidence no longer describes the new scene mapping."""
        entry = self._get_reconfigure_entry()
        if user_input is not None:
            return self._finish_reconfigure(
                entry,
                data_updates={
                    CONF_VIEW_GENERATION: (
                        int(entry.data.get(CONF_VIEW_GENERATION, 0)) + 1
                    )
                },
            )
        return self.async_show_form(
            step_id="reconf_view_changed", data_schema=vol.Schema({})
        )

    async def async_step_reconf_area(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}
        if user_input is not None:
            roi_error = _roi_error(
                user_input[CONF_ROI_X],
                user_input[CONF_ROI_Y],
                user_input[CONF_ROI_W],
                user_input[CONF_ROI_H],
            )
            if roi_error:
                errors["base"] = roi_error
            elif self._reconf_duplicate(
                entry.data[CONF_CAMERA],
                user_input[CONF_ROI_X],
                user_input[CONF_ROI_Y],
                user_input[CONF_ROI_W],
                user_input[CONF_ROI_H],
            ):
                errors["base"] = "duplicate_detector"
            else:
                unchanged = (
                    entry.data[CONF_ROI_X] == user_input[CONF_ROI_X]
                    and entry.data[CONF_ROI_Y] == user_input[CONF_ROI_Y]
                    and entry.data[CONF_ROI_W] == user_input[CONF_ROI_W]
                    and entry.data[CONF_ROI_H] == user_input[CONF_ROI_H]
                    and entry.data[CONF_WORKING_WIDTH]
                    == int(user_input[CONF_WORKING_WIDTH])
                )
                if unchanged:
                    # No-op guard: submitting the unchanged numbers must
                    # not clear a drawn polygon region.
                    return self.async_abort(reason="reconfigure_successful")
                rect_changed = not (
                    entry.data[CONF_ROI_X] == user_input[CONF_ROI_X]
                    and entry.data[CONF_ROI_Y] == user_input[CONF_ROI_Y]
                    and entry.data[CONF_ROI_W] == user_input[CONF_ROI_W]
                    and entry.data[CONF_ROI_H] == user_input[CONF_ROI_H]
                )
                updates = {
                    CONF_ROI_X: user_input[CONF_ROI_X],
                    CONF_ROI_Y: user_input[CONF_ROI_Y],
                    CONF_ROI_W: user_input[CONF_ROI_W],
                    CONF_ROI_H: user_input[CONF_ROI_H],
                    CONF_WORKING_WIDTH: int(user_input[CONF_WORKING_WIDTH]),
                }
                if rect_changed:
                    # Editing the RECT numbers deliberately sets a plain
                    # rectangle region (documented in the step text);
                    # a working-width-only change keeps a drawn polygon.
                    updates[CONF_ROI_POLYGONS] = None
                return self._finish_reconfigure(entry, data_updates=updates)
        return self.async_show_form(
            step_id="reconf_area",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_ROI_X, default=entry.data[CONF_ROI_X]
                    ): _REL_COORD,
                    vol.Required(
                        CONF_ROI_Y, default=entry.data[CONF_ROI_Y]
                    ): _REL_COORD,
                    vol.Required(
                        CONF_ROI_W, default=entry.data[CONF_ROI_W]
                    ): _REL_COORD,
                    vol.Required(
                        CONF_ROI_H, default=entry.data[CONF_ROI_H]
                    ): _REL_COORD,
                    vol.Required(
                        CONF_WORKING_WIDTH,
                        default=entry.data[CONF_WORKING_WIDTH],
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                            unit_of_measurement="px",
                        )
                    ),
                }
            ),
            errors=errors,
        )

    async def async_step_reconf_add_bin(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}
        if user_input is not None:
            name = user_input[CONF_BIN_NAME].strip()
            bin_id = slugify(name)
            bins = [dict(b) for b in entry.data[CONF_BINS]]
            existing = next((b for b in bins if b["id"] == bin_id), None)
            if not bin_id:
                errors["base"] = "invalid_name"
            elif existing is not None and existing.get(CONF_BIN_ACTIVE, True):
                errors["base"] = "duplicate_bin"
            elif existing is not None:
                # Same slug as a retired bin: adding it again is the
                # reactivation path (use the dedicated step so history
                # handling stays an explicit choice).
                errors["base"] = "bin_retired"
            else:
                bins.append(
                    {"id": bin_id, "name": name, CONF_BIN_ACTIVE: True}
                )
                return self._finish_reconfigure(
                    entry, data_updates={CONF_BINS: bins}
                )
        return self.async_show_form(
            step_id="reconf_add_bin",
            data_schema=vol.Schema(
                {vol.Required(CONF_BIN_NAME): selector.TextSelector()}
            ),
            errors=errors,
        )

    async def async_step_reconf_retire_bin(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        entry = self._get_reconfigure_entry()
        active = _active_bins(entry.data)
        errors: dict[str, str] = {}
        if user_input is not None:
            if len(active) <= 1:
                errors["base"] = "last_active_bin"
            else:
                bins = [dict(b) for b in entry.data[CONF_BINS]]
                for b in bins:
                    if b["id"] == user_input[ATTR_BIN_SELECT]:
                        b[CONF_BIN_ACTIVE] = False
                return self._finish_reconfigure(
                    entry, data_updates={CONF_BINS: bins}
                )
        if not active:
            return self.async_abort(reason="no_active_bins")
        return self.async_show_form(
            step_id="reconf_retire_bin",
            data_schema=vol.Schema(
                {
                    vol.Required(ATTR_BIN_SELECT): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                selector.SelectOptionDict(
                                    value=b["id"], label=b["name"]
                                )
                                for b in active
                            ],
                            mode=selector.SelectSelectorMode.LIST,
                        )
                    )
                }
            ),
            errors=errors,
        )

    async def async_step_reconf_reactivate_bin(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        entry = self._get_reconfigure_entry()
        retired = _retired_bins(entry.data)
        if not retired:
            return self.async_abort(reason="no_retired_bins")
        if user_input is not None:
            bins = [dict(b) for b in entry.data[CONF_BINS]]
            for b in bins:
                if b["id"] == user_input[ATTR_BIN_SELECT]:
                    b[CONF_BIN_ACTIVE] = True
            return self._finish_reconfigure(
                entry, data_updates={CONF_BINS: bins}
            )
        return self.async_show_form(
            step_id="reconf_reactivate_bin",
            data_schema=vol.Schema(
                {
                    vol.Required(ATTR_BIN_SELECT): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                selector.SelectOptionDict(
                                    value=b["id"], label=b["name"]
                                )
                                for b in retired
                            ],
                            mode=selector.SelectSelectorMode.LIST,
                        )
                    )
                }
            ),
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        return WastebinOptionsFlow()


class WastebinOptionsFlow(OptionsFlow):
    """Runtime tuning: detection and learning-capture cadence."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(data=user_input)
        options = self.config_entry.options
        minutes = selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=1,
                max=1440,
                step=1,
                mode=selector.NumberSelectorMode.BOX,
                unit_of_measurement="min",
            )
        )
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_SCAN_INTERVAL,
                        default=options.get(
                            CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL_MIN
                        ),
                    ): minutes,
                    vol.Required(
                        CONF_CAPTURE_INTERVAL,
                        default=options.get(
                            CONF_CAPTURE_INTERVAL, DEFAULT_CAPTURE_INTERVAL_MIN
                        ),
                    ): minutes,
                    vol.Required(
                        CONF_CONFIRM_SCANS,
                        default=options.get(
                            CONF_CONFIRM_SCANS, DEFAULT_CONFIRM_SCANS
                        ),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            max=MAX_CONFIRM_SCANS,
                            step=1,
                            mode=selector.NumberSelectorMode.BOX,
                        )
                    ),
                }
            ),
        )

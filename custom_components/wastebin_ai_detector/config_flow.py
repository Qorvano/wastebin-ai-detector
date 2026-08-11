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
    CONF_BIN_NAME,
    CONF_BINS,
    CONF_CAMERA,
    CONF_CAPTURE_INTERVAL,
    CONF_ROI_H,
    CONF_ROI_W,
    CONF_ROI_X,
    CONF_ROI_Y,
    CONF_SCAN_INTERVAL,
    CONF_WORKING_WIDTH,
    DEFAULT_CAPTURE_INTERVAL_MIN,
    DEFAULT_SCAN_INTERVAL_MIN,
    DEFAULT_WORKING_WIDTH,
    DOMAIN,
)

CONF_ADD_ANOTHER = "add_another"

_REL_COORD = selector.NumberSelector(
    selector.NumberSelectorConfig(
        min=0.0, max=1.0, step=0.001, mode=selector.NumberSelectorMode.BOX
    )
)


class WastebinConfigFlow(ConfigFlow, domain=DOMAIN):
    """Guided setup: camera, region of interest, bins."""

    VERSION = 1

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
            x = user_input[CONF_ROI_X]
            y = user_input[CONF_ROI_Y]
            w = user_input[CONF_ROI_W]
            h = user_input[CONF_ROI_H]
            if w <= 0 or h <= 0 or x + w > 1.0 or y + h > 1.0:
                errors["base"] = "invalid_roi"
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
                }
            ),
        )

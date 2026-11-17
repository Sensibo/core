"""DataUpdateCoordinator for the Sensibo integration."""

import asyncio
from datetime import timedelta
import random
from typing import TYPE_CHECKING, override

from aiohttp import ClientError
from pysensibo import (
    EVENT_CHANGED,
    EVENT_HELLO,
    EVENT_MEASUREMENTS,
    SensiboClient,
    update_measurements,
)
from pysensibo.exceptions import (
    AuthenticationError,
    SensiboError,
    StreamUnavailableError,
)
from pysensibo.model import SensiboData, SensiboEvent

from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.debounce import Debouncer
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import DEFAULT_SCAN_INTERVAL, DOMAIN, LOGGER, STREAM_SCAN_INTERVAL, TIMEOUT

if TYPE_CHECKING:
    from . import SensiboConfigEntry

REQUEST_REFRESH_DELAY = 0.35
STREAM_RETRY_DELAY = 5
STREAM_MAX_RETRY_DELAY = 300


class SensiboDataUpdateCoordinator(DataUpdateCoordinator[SensiboData]):
    """A Sensibo Data Update Coordinator."""

    config_entry: SensiboConfigEntry

    def __init__(self, hass: HomeAssistant, config_entry: SensiboConfigEntry) -> None:
        """Initialize the Sensibo coordinator."""
        super().__init__(
            hass,
            LOGGER,
            config_entry=config_entry,
            name=DOMAIN,
            update_interval=timedelta(seconds=DEFAULT_SCAN_INTERVAL),
            # We don't want an immediate refresh since the device
            # takes a moment to reflect the state change
            request_refresh_debouncer=Debouncer(
                hass, LOGGER, cooldown=REQUEST_REFRESH_DELAY, immediate=False
            ),
        )
        self.client = SensiboClient(
            self.config_entry.data[CONF_API_KEY],
            session=async_get_clientsession(hass),
            timeout=TIMEOUT,
        )
        self.previous_devices: set[str] = set()
        self._stream_open = False

    async def async_listen_events(self) -> None:
        """Keep the event stream open.

        While the stream is open, the API tells when a device changes and the
        devices are polled at a slow interval. Without it they are polled as before.
        """
        failures = 0
        while True:
            retry_after = 0.0
            try:
                async for event in self.client.async_stream_events():
                    failures = 0
                    await self._async_handle_event(event)
            except AuthenticationError:
                # The next poll starts the reauthentication flow
                await self._async_stream_closed()
                return
            except StreamUnavailableError as err:
                retry_after = err.retry_after or 0.0
                LOGGER.debug("Event stream is not available: %s", err)
            except (SensiboError, ClientError, TimeoutError) as err:
                LOGGER.debug("Event stream failed: %s", err)
            await self._async_stream_closed()
            failures += 1
            delay = max(
                min(
                    STREAM_RETRY_DELAY * 2 ** min(failures - 1, 6),
                    STREAM_MAX_RETRY_DELAY,
                ),
                retry_after,
            )
            # The random part keeps the clients apart after a restart of the API
            await asyncio.sleep(delay * (1 + random.random()))

    async def _async_handle_event(self, event: SensiboEvent) -> None:
        """Handle an event of the event stream."""
        if event.name == EVENT_HELLO:
            self._stream_open = True
            self.update_interval = timedelta(seconds=STREAM_SCAN_INTERVAL)
        elif event.name == EVENT_MEASUREMENTS:
            if update_measurements(self.data, event):
                self.async_update_listeners()
                return
        elif event.name != EVENT_CHANGED:
            return
        await self.async_request_refresh()

    async def _async_stream_closed(self) -> None:
        """Poll at the default interval again."""
        if not self._stream_open:
            return
        self._stream_open = False
        self.update_interval = timedelta(seconds=DEFAULT_SCAN_INTERVAL)
        await self.async_request_refresh()

    def get_devices(
        self, added_devices: set[str]
    ) -> tuple[set[str], set[str], set[str]]:
        """Addition and removal of devices."""
        data = self.data
        current_motion_sensors = {
            sensor_id
            for device_data in data.parsed.values()
            if device_data.motion_sensors
            for sensor_id in device_data.motion_sensors
        }
        current_devices: set[str] = set(data.parsed)
        LOGGER.debug(
            "Current devices: %s, moption sensors: %s",
            current_devices,
            current_motion_sensors,
        )
        new_devices: set[str] = (
            current_motion_sensors | current_devices
        ) - added_devices
        remove_devices = added_devices - current_devices - current_motion_sensors
        new_added_devices = (added_devices - remove_devices) | new_devices

        LOGGER.debug(
            "New devices: %s, Removed devices: %s, Added devices: %s",
            new_devices,
            remove_devices,
            new_added_devices,
        )
        return (new_devices, remove_devices, new_added_devices)

    @override
    async def _async_update_data(self) -> SensiboData:
        """Fetch data from Sensibo."""
        try:
            data = await self.client.async_get_devices_data()
        except AuthenticationError as error:
            raise ConfigEntryAuthFailed(
                translation_domain=DOMAIN,
                translation_key="auth_error",
            ) from error
        except SensiboError as error:
            raise UpdateFailed(
                translation_domain=DOMAIN,
                translation_key="update_error",
                translation_placeholders={"error": str(error)},
            ) from error

        if not data.raw:
            raise UpdateFailed(translation_domain=DOMAIN, translation_key="no_data")

        current_devices = set(data.parsed)
        for device_data in data.parsed.values():
            if device_data.motion_sensors:
                for motion_sensor_id in device_data.motion_sensors:
                    current_devices.add(motion_sensor_id)

        if stale_devices := self.previous_devices - current_devices:
            LOGGER.debug("Removing stale devices: %s", stale_devices)
            device_registry = dr.async_get(self.hass)
            for _id in stale_devices:
                device = device_registry.async_get_device_by_identifier(
                    (DOMAIN, _id), self.config_entry.entry_id
                )
                if device:
                    device_registry.async_remove_device(device.id)
        self.previous_devices = current_devices

        return data

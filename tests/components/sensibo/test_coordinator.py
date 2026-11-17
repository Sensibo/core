"""The test for the sensibo coordinator."""

import asyncio
from collections.abc import AsyncGenerator
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock, patch

from aiohttp import ClientError
from freezegun.api import FrozenDateTimeFactory
from pysensibo import EVENT_CHANGED, EVENT_HELLO, EVENT_MEASUREMENTS, EVENT_RESYNC
from pysensibo.exceptions import (
    AuthenticationError,
    SensiboError,
    StreamUnavailableError,
)
from pysensibo.model import SensiboData, SensiboEvent
import pytest
from pytest_homeassistant.common import MockConfigEntry, async_fire_time_changed

from homeassistant.components.climate import ATTR_CURRENT_TEMPERATURE, HVACMode
from homeassistant.components.sensibo.const import (
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
    STREAM_SCAN_INTERVAL,
)
from homeassistant.components.sensibo.coordinator import (
    REQUEST_REFRESH_DELAY,
    STREAM_MAX_RETRY_DELAY,
    STREAM_RETRY_DELAY,
)
from homeassistant.const import STATE_OFF, STATE_ON, STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant

from . import ENTRY_CONFIG

type EventQueue = asyncio.Queue[SensiboEvent | Exception]


@pytest.fixture(name="event_queue")
def mock_event_stream(mock_client: MagicMock) -> EventQueue:
    """Mock the event stream: an event arrives, an exception closes the stream."""
    queue: EventQueue = asyncio.Queue()

    async def _stream() -> AsyncGenerator[SensiboEvent]:
        while True:
            item = await queue.get()
            if isinstance(item, Exception):
                queue.task_done()
                raise item
            yield item
            queue.task_done()

    mock_client.async_stream_events.side_effect = _stream
    return queue


async def _async_send(
    hass: HomeAssistant, queue: EventQueue, item: SensiboEvent | Exception
) -> None:
    """Send an item on the event stream and wait until it is handled."""
    queue.put_nowait(item)
    await queue.join()
    await hass.async_block_till_done()


async def _async_tick(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory, seconds: float
) -> None:
    """Move the time."""
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass)
    await hass.async_block_till_done()


async def test_coordinator(
    hass: HomeAssistant,
    mock_client: MagicMock,
    get_data: tuple[SensiboData, dict[str, Any], dict[str, Any]],
    freezer: FrozenDateTimeFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the Sensibo coordinator with errors."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        data=ENTRY_CONFIG,
        entry_id="1",
        unique_id="firstnamelastname",
        version=2,
    )

    config_entry.add_to_hass(hass)

    mock_client.async_get_devices_data.return_value.parsed[
        "ABC999111"
    ].hvac_mode = "heat"
    mock_client.async_get_devices_data.return_value.parsed["ABC999111"].device_on = True

    mock_data = mock_client.async_get_devices_data
    mock_data.return_value = get_data[0]
    await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    mock_data.assert_called_once()
    state = hass.states.get("climate.hallway_hallway")
    assert state.state == HVACMode.HEAT
    mock_data.reset_mock()

    mock_data.side_effect = SensiboError("info")
    freezer.tick(timedelta(minutes=1))
    async_fire_time_changed(hass)
    await hass.async_block_till_done()
    mock_data.assert_called_once()
    state = hass.states.get("climate.hallway_hallway")
    assert state.state == STATE_UNAVAILABLE
    mock_data.reset_mock()

    mock_data.return_value = SensiboData(raw={}, parsed={})
    mock_data.side_effect = None
    freezer.tick(timedelta(minutes=1))
    async_fire_time_changed(hass)
    await hass.async_block_till_done()
    mock_data.assert_called_once()
    state = hass.states.get("climate.hallway_hallway")
    assert state.state == STATE_UNAVAILABLE
    mock_data.reset_mock()

    mock_data.return_value = get_data[0]
    mock_data.side_effect = None
    freezer.tick(timedelta(minutes=1))
    async_fire_time_changed(hass)
    await hass.async_block_till_done()
    mock_data.assert_called_once()
    state = hass.states.get("climate.hallway_hallway")
    assert state.state == HVACMode.HEAT
    mock_data.reset_mock()

    mock_data.side_effect = AuthenticationError("info")
    freezer.tick(timedelta(minutes=1))
    async_fire_time_changed(hass)
    await hass.async_block_till_done()
    mock_data.assert_called_once()
    state = hass.states.get("climate.hallway_hallway")
    assert state.state == STATE_UNAVAILABLE

    assert "Platform sensibo does not generate unique IDs" not in caplog.text


@pytest.mark.usefixtures("freezer", "event_queue", "load_int")
async def test_event_stream(
    hass: HomeAssistant,
    mock_client: MagicMock,
    event_queue: EventQueue,
    freezer: FrozenDateTimeFactory,
) -> None:
    """Test that events update the devices and replace the frequent poll."""
    mock_data = mock_client.async_get_devices_data
    mock_data.assert_called_once()
    mock_data.reset_mock()

    # The open stream causes a refresh and a slow poll interval
    await _async_send(hass, event_queue, SensiboEvent(EVENT_HELLO, {"version": 1}))
    await _async_tick(hass, freezer, REQUEST_REFRESH_DELAY)
    mock_data.assert_called_once()
    mock_data.reset_mock()
    await _async_tick(hass, freezer, DEFAULT_SCAN_INTERVAL)
    mock_data.assert_not_called()

    # Measurements arrive in the events, with no request
    await _async_send(
        hass,
        event_queue,
        SensiboEvent(
            EVENT_MEASUREMENTS, {"id": "AAZZAAZZ", "measurements": {"pm25": 2}}
        ),
    )
    state = hass.states.get("sensor.kitchen_kitchen_pure_aqi")
    assert state.state == "moderate"
    await _async_send(
        hass,
        event_queue,
        SensiboEvent(
            EVENT_MEASUREMENTS,
            {
                "id": "AABBCC",
                "podId": "ABC999111",
                "measurements": {"motion": False, "temperature": 19.5},
            },
        ),
    )
    state = hass.states.get("binary_sensor.hallway_motion_sensor_motion")
    assert state.state == STATE_OFF
    state = hass.states.get("climate.hallway_hallway")
    assert state.attributes[ATTR_CURRENT_TEMPERATURE] == 19.5
    await _async_send(hass, event_queue, SensiboEvent(EVENT_RESYNC, {}))
    await _async_tick(hass, freezer, REQUEST_REFRESH_DELAY)
    mock_data.assert_not_called()

    # Other changes and unknown devices cause a refresh
    for event in (
        SensiboEvent(EVENT_CHANGED, {"id": "ABC999111", "fields": ["acState"]}),
        SensiboEvent(
            EVENT_MEASUREMENTS, {"id": "NEW", "measurements": {"temperature": 20}}
        ),
    ):
        await _async_send(hass, event_queue, event)
        await _async_tick(hass, freezer, REQUEST_REFRESH_DELAY)
        mock_data.assert_called_once()
        mock_data.reset_mock()

    # The slow poll continues while events arrive
    await _async_tick(hass, freezer, STREAM_SCAN_INTERVAL - DEFAULT_SCAN_INTERVAL)
    mock_data.assert_not_called()
    await _async_send(
        hass,
        event_queue,
        SensiboEvent(
            EVENT_MEASUREMENTS,
            {"id": "AABBCC", "podId": "ABC999111", "measurements": {"motion": True}},
        ),
    )
    state = hass.states.get("binary_sensor.hallway_motion_sensor_motion")
    assert state.state == STATE_ON
    await _async_tick(hass, freezer, DEFAULT_SCAN_INTERVAL)
    mock_data.assert_called_once()
    mock_data.reset_mock()

    # Without the stream the default poll interval is back
    await _async_send(hass, event_queue, ClientError("closed"))
    await _async_tick(hass, freezer, REQUEST_REFRESH_DELAY)
    mock_data.assert_called_once()
    mock_data.reset_mock()
    await _async_tick(hass, freezer, DEFAULT_SCAN_INTERVAL)
    mock_data.assert_called_once()


@pytest.mark.parametrize(
    ("error", "delay"),
    [
        (StreamUnavailableError("off", retry_after=300), 300),
        (StreamUnavailableError("busy"), STREAM_RETRY_DELAY),
        (SensiboError("error"), STREAM_RETRY_DELAY),
        (ClientError("closed"), STREAM_RETRY_DELAY),
        (TimeoutError(), STREAM_RETRY_DELAY),
    ],
)
@pytest.mark.usefixtures("freezer", "event_queue", "load_int")
async def test_event_stream_opens_again_after_a_delay(
    hass: HomeAssistant,
    mock_client: MagicMock,
    event_queue: EventQueue,
    freezer: FrozenDateTimeFactory,
    error: Exception,
    delay: int,
) -> None:
    """Test that a stream that cannot open is tried again, each time later."""
    mock_data = mock_client.async_get_devices_data
    mock_stream = mock_client.async_stream_events
    assert mock_stream.call_count == 1

    with patch(
        "homeassistant.components.sensibo.coordinator.random.random", return_value=0
    ):
        await _async_send(hass, event_queue, error)
        await _async_tick(hass, freezer, delay - 1)
        assert mock_stream.call_count == 1
        await _async_tick(hass, freezer, 2)
        assert mock_stream.call_count == 2

        for _ in range(8):
            await _async_send(hass, event_queue, ClientError("closed"))
            await _async_tick(hass, freezer, STREAM_MAX_RETRY_DELAY + 1)
        assert mock_stream.call_count == 10
        await _async_send(hass, event_queue, ClientError("closed"))
        await _async_tick(hass, freezer, STREAM_MAX_RETRY_DELAY - 1)
        assert mock_stream.call_count == 10

    # The devices are polled as before
    mock_data.reset_mock()
    await _async_tick(hass, freezer, DEFAULT_SCAN_INTERVAL)
    mock_data.assert_called_once()


@pytest.mark.usefixtures("freezer", "event_queue", "load_int")
async def test_event_stream_authentication_error(
    hass: HomeAssistant,
    mock_client: MagicMock,
    event_queue: EventQueue,
    freezer: FrozenDateTimeFactory,
) -> None:
    """Test that the stream is not tried again with a wrong API key."""
    mock_stream = mock_client.async_stream_events

    await _async_send(hass, event_queue, AuthenticationError("Invalid API key"))
    await _async_tick(hass, freezer, STREAM_MAX_RETRY_DELAY * 2)

    assert mock_stream.call_count == 1

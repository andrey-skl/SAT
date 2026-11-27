"""Tests focused on heating control behavior."""

import asyncio
import logging
from typing import Optional
from unittest.mock import AsyncMock

import pytest
from homeassistant.components.climate import HVACMode

from custom_components.sat.const import (
    COLD_SETPOINT,
    CONF_FLAME_OFF_SETPOINT_OFFSET_CELSIUS,
    CONF_FLOW_SETPOINT_OFFSET_CELSIUS,
    CONF_FORCE_PULSE_WIDTH_MODULATION,
    CONF_HEATING_SYSTEM,
    CONF_MINIMUM_SETPOINT,
    CONF_MODE,
    CONF_MODULATION_SUPPRESSION_DELAY_SECONDS,
    CONF_OVERSHOOT_PROTECTION,
    DHW_FLAME_HOLD_SECONDS,
    DHW_IGNITION_GRACE_SECONDS,
    DHW_SUPPRESSION_MAX_SECONDS,
    HEATER_STARTUP_TIMEFRAME,
    SATURATION_SUSTAIN_SECONDS,
    TARGET_TEMPERATURE_CHANGE_WINDOW_SECONDS,
    UNDERHEAT_SUSTAIN_SECONDS,
    HeatingSystem,
    MINIMUM_RELATIVE_MODULATION,
    MINIMUM_SETPOINT,
    OPTIONS_DEFAULTS,
)
from custom_components.sat.const import OVERSHOOT_REFERENCE_WINDOW_SECONDS, OVERSHOOT_SUSTAIN_SECONDS
from custom_components.sat.cycles.const import OVERSHOOT_MARGIN_CELSIUS
from custom_components.sat.entry_data import SatConfig, SatMode
from custom_components.sat.heating_control import (
    HeatingDemand,
    SatHeatingControl,
)
from custom_components.sat.helpers import timestamp
from custom_components.sat.manufacturer import ManufacturerFactory
from custom_components.sat.types import HeaterState, PWMStatus, RelativeModulationState

pytestmark = pytest.mark.parametrize(
    ("domains", "data", "options", "config"),
    [
        (
            [],
            {
                CONF_HEATING_SYSTEM: HeatingSystem.RADIATORS,
            },
            {},
            {},
        ),
    ],
)


def _update_coordinator_config(coordinator) -> None:
    options = {
        **OPTIONS_DEFAULTS,
    }
    config = SatConfig(
        entry_id="test",
        data={
            CONF_MODE: SatMode.FAKE.value,
            CONF_MINIMUM_SETPOINT: 40.0,
            CONF_OVERSHOOT_PROTECTION: True,
            CONF_HEATING_SYSTEM: HeatingSystem.RADIATORS,
        },
        options=options,
    )
    coordinator._config = config
    coordinator._manufacturer = ManufacturerFactory.resolve_by_name(config.manufacturer)


def _enable_pwm(heating_control: SatHeatingControl, status: PWMStatus) -> None:
    heating_control._pwm._enabled = True
    heating_control._pwm._status = status


def _make_demand(
    requested_setpoint: float,
    hvac_mode: HVACMode = HVACMode.HEAT,
    outside_temperature: float = 10.0,
    valves_open: bool = True,
) -> HeatingDemand:
    return HeatingDemand(
        valves_open=valves_open,
        hvac_mode=hvac_mode,
        timestamp=timestamp(),
        requested_setpoint=requested_setpoint,
        outside_temperature=outside_temperature,
    )


async def test_hvac_off_forces_minimum(heating_control):
    heating_control._control_setpoint = 45.0

    await heating_control.update(_make_demand(45.0, hvac_mode=HVACMode.OFF))

    assert heating_control.control_setpoint == MINIMUM_SETPOINT
    assert heating_control._coordinator.active is False


async def test_setpoint_below_cold_resets_pwm_cycle_and_forces_minimum(heating_control, coordinator):
    _enable_pwm(heating_control, PWMStatus.ON)

    await heating_control.update(_make_demand(COLD_SETPOINT - 0.1))

    assert heating_control.control_setpoint == MINIMUM_SETPOINT
    assert heating_control.relative_modulation_value == heating_control._config.pwm.maximum_relative_modulation
    assert heating_control.pwm_state.enabled is True
    assert heating_control.pwm_state.status is PWMStatus.OFF
    assert coordinator.active is False


async def test_closed_valves_force_minimum_and_keep_pwm_enabled(heating_control, coordinator):
    _enable_pwm(heating_control, PWMStatus.ON)

    await heating_control.update(_make_demand(45.0, valves_open=False))

    assert heating_control.control_setpoint == MINIMUM_SETPOINT
    assert heating_control.relative_modulation_value == heating_control._config.pwm.maximum_relative_modulation
    assert heating_control.pwm_state.enabled is True
    assert heating_control.pwm_state.status is PWMStatus.OFF
    assert coordinator.active is False


async def test_continuous_uses_requested_without_boiler_temperature(heating_control):
    await heating_control.update(_make_demand(42.0))

    assert heating_control.control_setpoint == 42.0


async def test_continuous_follows_requested_at_or_above_boiler_temperature(heating_control, coordinator):
    await coordinator.async_set_boiler_temperature(35.0)
    await coordinator.async_set_heater_state(HeaterState.ON)

    await heating_control.update(_make_demand(40.0))

    assert heating_control.control_setpoint == 40.0


async def test_continuous_clamps_below_boiler_temperature(heating_control, coordinator):
    await coordinator.async_set_boiler_temperature(55.0)
    await coordinator.async_set_heater_state(HeaterState.ON)

    await heating_control.update(_make_demand(40.0))

    expected = 55.0 - OPTIONS_DEFAULTS[CONF_FLOW_SETPOINT_OFFSET_CELSIUS]
    assert heating_control.control_setpoint == expected


async def test_continuous_allows_requested_above_offset(heating_control, coordinator):
    await coordinator.async_set_boiler_temperature(55.0)
    await coordinator.async_set_heater_state(HeaterState.ON)

    requested = 55.0 - OPTIONS_DEFAULTS[CONF_FLOW_SETPOINT_OFFSET_CELSIUS] + 0.5
    await heating_control.update(_make_demand(requested))

    assert heating_control.control_setpoint == requested


async def test_continuous_holds_previous_when_offset_rises(heating_control, coordinator):
    heating_control._control_setpoint = 45.0
    await coordinator.async_set_boiler_temperature(50.0)
    await coordinator.async_set_heater_state(HeaterState.ON)

    await heating_control.update(_make_demand(42.0))

    assert heating_control.control_setpoint == 45.0


async def test_continuous_allows_change_when_flame_off(heating_control, coordinator):
    heating_control._control_setpoint = 45.0
    await coordinator.async_set_boiler_temperature(50.0)
    await coordinator.async_set_heater_state(HeaterState.OFF)

    await heating_control.update(_make_demand(42.0))

    assert heating_control.control_setpoint == 42.0


async def test_continuous_allows_increase_when_requested_increases(heating_control, coordinator):
    heating_control._control_setpoint = 40.0
    await coordinator.async_set_boiler_temperature(50.0)
    await coordinator.async_set_heater_state(HeaterState.ON)

    await heating_control.update(_make_demand(41.0))

    expected = 50.0 - OPTIONS_DEFAULTS[CONF_FLOW_SETPOINT_OFFSET_CELSIUS]
    assert heating_control.control_setpoint == expected


async def test_pwm_suppression_applied(hass, coordinator, monkeypatch):
    _update_coordinator_config(coordinator)

    heating_control = SatHeatingControl(hass=hass, coordinator=coordinator, config=coordinator._config)
    _enable_pwm(heating_control, PWMStatus.ON)
    monkeypatch.setattr(heating_control._pwm, "update", lambda *args, **kwargs: None)

    await coordinator.async_set_heater_state(HeaterState.ON)
    await coordinator.async_set_boiler_temperature(50.0)
    heating_control._device_tracker._last_flame_on_at = timestamp() - (
            OPTIONS_DEFAULTS[CONF_MODULATION_SUPPRESSION_DELAY_SECONDS] + 1
    )

    await heating_control.update(_make_demand(40.0))

    assert heating_control.control_setpoint == 49.0


async def test_pwm_flame_off_return_offset(hass, monkeypatch, coordinator):
    _update_coordinator_config(coordinator)
    monkeypatch.setattr(type(coordinator), "return_temperature", property(lambda self: 30.0))

    heating_control = SatHeatingControl(hass=hass, coordinator=coordinator, config=coordinator._config)
    _enable_pwm(heating_control, PWMStatus.ON)
    monkeypatch.setattr(heating_control._pwm, "update", lambda *args, **kwargs: None)

    await coordinator.async_set_heater_state(HeaterState.OFF)

    await heating_control.update(_make_demand(40.0))

    assert heating_control.control_setpoint == 30.0 + OPTIONS_DEFAULTS[CONF_FLAME_OFF_SETPOINT_OFFSET_CELSIUS]


async def test_flame_off_setpoint_held_until_suppression_delay(hass, monkeypatch, coordinator):
    _update_coordinator_config(coordinator)
    monkeypatch.setattr(type(coordinator), "return_temperature", property(lambda self: 30.0))

    heating_control = SatHeatingControl(hass=hass, coordinator=coordinator, config=coordinator._config)
    _enable_pwm(heating_control, PWMStatus.ON)
    monkeypatch.setattr(heating_control._pwm, "update", lambda *args, **kwargs: None)

    await coordinator.async_set_heater_state(HeaterState.OFF)

    await heating_control.update(_make_demand(40.0))

    assert heating_control.control_setpoint == 30.0 + OPTIONS_DEFAULTS[CONF_FLAME_OFF_SETPOINT_OFFSET_CELSIUS]

    await coordinator.async_set_heater_state(HeaterState.ON)
    await coordinator.async_set_boiler_temperature(50.0)
    heating_control._device_tracker._last_flame_on_at = timestamp() - (
            OPTIONS_DEFAULTS[CONF_MODULATION_SUPPRESSION_DELAY_SECONDS] - 1
    )

    await heating_control.update(_make_demand(40.0))

    assert heating_control.control_setpoint == 30.0 + OPTIONS_DEFAULTS[CONF_FLAME_OFF_SETPOINT_OFFSET_CELSIUS]


async def test_enables_pwm_on_sustained_overshoot(heating_control, coordinator):
    await coordinator.async_set_heater_state(HeaterState.ON)
    await coordinator.async_set_boiler_temperature(40.0)

    requested = 30.0
    start_time = timestamp()

    await heating_control.update(HeatingDemand(
        timestamp=start_time,
        valves_open=True,
        hvac_mode=HVACMode.HEAT,
        outside_temperature=10.0,
        requested_setpoint=requested,
    ))

    assert heating_control.pwm_state.enabled is False

    await heating_control.update(HeatingDemand(
        valves_open=True,
        hvac_mode=HVACMode.HEAT,
        outside_temperature=10.0,
        requested_setpoint=requested,
        timestamp=start_time + OVERSHOOT_SUSTAIN_SECONDS + 1,
    ))

    assert heating_control.pwm_state.enabled is True


async def test_no_overshoot_while_the_flame_is_off(heating_control, coordinator, monkeypatch):
    monkeypatch.setattr(type(coordinator), "flame_active", property(lambda self: False))
    await coordinator.async_set_heater_state(HeaterState.ON)
    await coordinator.async_set_boiler_temperature(40.0)
    start_time = timestamp()

    for offset in (0.0, OVERSHOOT_SUSTAIN_SECONDS + 1):
        await heating_control.update(HeatingDemand(
            valves_open=True,
            hvac_mode=HVACMode.HEAT,
            requested_setpoint=30.0,
            outside_temperature=10.0,
            timestamp=start_time + offset,
        ))

    assert heating_control.pwm_state.enabled is False


async def test_modulation_is_minimum_when_pwm_active_and_supported(heating_control, monkeypatch):
    heating_control._coordinator.config.supports_relative_modulation_management = True
    _enable_pwm(heating_control, PWMStatus.ON)
    monkeypatch.setattr(heating_control._pwm, "update", lambda *args, **kwargs: None)

    await heating_control.update(_make_demand(40.0))

    assert heating_control.relative_modulation_value == MINIMUM_RELATIVE_MODULATION


async def test_modulation_is_minimum_when_pwm_enabled_and_idle(heating_control, monkeypatch):
    heating_control._coordinator.config.supports_relative_modulation_management = True
    _enable_pwm(heating_control, PWMStatus.IDLE)
    monkeypatch.setattr(heating_control._pwm, "update", lambda *args, **kwargs: None)

    await heating_control.update(_make_demand(40.0))

    assert heating_control.relative_modulation_value == MINIMUM_RELATIVE_MODULATION


async def test_modulation_stays_maximum_during_hot_water_when_supported(heating_control, monkeypatch):
    heating_control._coordinator.config.supports_relative_modulation_management = True
    _enable_pwm(heating_control, PWMStatus.ON)
    monkeypatch.setattr(type(heating_control._coordinator), "hot_water_active", property(lambda self: True))

    await heating_control.update(_make_demand(40.0))

    assert heating_control.relative_modulation_value == heating_control._config.pwm.maximum_relative_modulation


async def test_disables_pwm_on_sustained_underheat(heating_control, coordinator, monkeypatch):
    _enable_pwm(heating_control, PWMStatus.ON)
    monkeypatch.setattr(heating_control._pwm, "update", lambda *args, **kwargs: None)

    requested_setpoint = 40.0
    start_time = timestamp()

    await coordinator.async_set_heater_state(HeaterState.ON)
    await coordinator.async_set_boiler_temperature(requested_setpoint - 4.0)
    heating_control._device_tracker._last_flame_on_at = start_time - HEATER_STARTUP_TIMEFRAME

    await heating_control.update(HeatingDemand(
        timestamp=start_time,
        valves_open=True,
        hvac_mode=HVACMode.HEAT,
        requested_setpoint=requested_setpoint,
        outside_temperature=10.0,
    ))
    assert heating_control.pwm_state.enabled is True

    await heating_control.update(HeatingDemand(
        valves_open=True,
        hvac_mode=HVACMode.HEAT,
        requested_setpoint=requested_setpoint,
        outside_temperature=10.0,
        timestamp=start_time + UNDERHEAT_SUSTAIN_SECONDS + 1,
    ))
    assert heating_control.pwm_state.enabled is False


async def test_disables_pwm_on_sustained_saturation(heating_control, coordinator, monkeypatch):
    _enable_pwm(heating_control, PWMStatus.ON)
    monkeypatch.setattr(heating_control._pwm, "update", lambda *args, **kwargs: None)

    requested_setpoint = 40.0
    start_time = timestamp()

    await coordinator.async_set_heater_state(HeaterState.ON)
    await coordinator.async_set_boiler_temperature(requested_setpoint + 20.0)
    heating_control._pwm._duty_cycle = (300, 0)
    heating_control._pwm._effective_on_temperature = 44.0
    heating_control._pwm._floor_observed_at = start_time

    await heating_control.update(HeatingDemand(
        timestamp=start_time,
        valves_open=True,
        hvac_mode=HVACMode.HEAT,
        requested_setpoint=requested_setpoint,
        outside_temperature=10.0,
    ))
    assert heating_control.pwm_state.enabled is True

    await heating_control.update(HeatingDemand(
        valves_open=True,
        hvac_mode=HVACMode.HEAT,
        requested_setpoint=requested_setpoint,
        outside_temperature=10.0,
        timestamp=start_time + SATURATION_SUSTAIN_SECONDS + 1,
    ))
    assert heating_control.pwm_state.enabled is False


async def test_async_added_to_hass_replaces_existing_listeners(heating_control, monkeypatch):
    removed = {"coordinator": 0, "pwm_cycle": 0}
    calls = {"coordinator_add": 0, "pwm_listen": 0}

    heating_control._coordinator_listener_remove = lambda: removed.__setitem__("coordinator", removed["coordinator"] + 1)
    heating_control._pwm_cycle_listener_remove = lambda: removed.__setitem__("pwm_cycle", removed["pwm_cycle"] + 1)

    def _add_listener(callback):
        calls["coordinator_add"] += 1
        return lambda: None

    def _listen(_self, event_type, callback):
        calls["pwm_listen"] += 1
        return lambda: None

    heating_control._coordinator.async_add_listener = _add_listener
    monkeypatch.setattr(type(heating_control._hass.bus), "async_listen", _listen)

    await heating_control.async_added_to_hass()

    assert removed["coordinator"] == 1
    assert removed["pwm_cycle"] == 1
    assert calls["coordinator_add"] == 1
    assert calls["pwm_listen"] == 1


def _config_without_overshoot_protection() -> SatConfig:
    return SatConfig(
        entry_id="test",
        data={
            CONF_MODE: SatMode.FAKE.value,
            CONF_MINIMUM_SETPOINT: 40.0,
            CONF_OVERSHOOT_PROTECTION: False,
            CONF_HEATING_SYSTEM: HeatingSystem.RADIATORS,
        },
        options={**OPTIONS_DEFAULTS},
    )


async def test_no_pwm_enable_on_sustained_overshoot_without_protection(heating_control, coordinator):
    heating_control._config = _config_without_overshoot_protection()

    await coordinator.async_set_heater_state(HeaterState.ON)
    await coordinator.async_set_boiler_temperature(40.0)

    requested = 30.0
    start_time = timestamp()

    await heating_control.update(HeatingDemand(
        timestamp=start_time,
        valves_open=True,
        hvac_mode=HVACMode.HEAT,
        outside_temperature=10.0,
        requested_setpoint=requested,
    ))

    await heating_control.update(HeatingDemand(
        valves_open=True,
        hvac_mode=HVACMode.HEAT,
        outside_temperature=10.0,
        requested_setpoint=requested,
        timestamp=start_time + OVERSHOOT_SUSTAIN_SECONDS + 1,
    ))

    assert heating_control.pwm_state.enabled is False


async def test_pwm_off_phase_forces_minimum_without_protection(heating_control, coordinator):
    heating_control._config = _config_without_overshoot_protection()

    await coordinator.async_set_boiler_temperature(50.0)

    _enable_pwm(heating_control, PWMStatus.OFF)
    heating_control._control_setpoint = 45.0

    await heating_control.update(_make_demand(45.0))

    assert heating_control.control_setpoint == MINIMUM_SETPOINT


async def test_force_pulse_width_modulation_keeps_pwm_enabled(heating_control, coordinator):
    heating_control._config = SatConfig(
        entry_id="test",
        data={
            CONF_MODE: SatMode.FAKE.value,
            CONF_MINIMUM_SETPOINT: 40.0,
            CONF_OVERSHOOT_PROTECTION: True,
            CONF_HEATING_SYSTEM: HeatingSystem.RADIATORS,
        },
        options={**OPTIONS_DEFAULTS, CONF_FORCE_PULSE_WIDTH_MODULATION: True},
    )

    await coordinator.async_set_boiler_temperature(50.0)
    await heating_control.update(_make_demand(45.0))

    assert heating_control.pwm_state.enabled is True

    await heating_control.update(_make_demand(heating_control._config.limits.maximum_setpoint + 1))

    assert heating_control.pwm_state.enabled is True


async def test_underheat_guard_ignores_flow_while_burner_is_off(heating_control, coordinator, monkeypatch):
    _enable_pwm(heating_control, PWMStatus.ON)
    monkeypatch.setattr(heating_control._pwm, "update", lambda *args, **kwargs: None)
    monkeypatch.setattr(type(coordinator), "flame_active", property(lambda self: False))

    requested_setpoint = 40.0
    start_time = timestamp()

    await coordinator.async_set_heater_state(HeaterState.ON)
    await coordinator.async_set_boiler_temperature(requested_setpoint - 4.0)

    for offset in (0.0, UNDERHEAT_SUSTAIN_SECONDS + 1, 2 * UNDERHEAT_SUSTAIN_SECONDS + 2):
        await heating_control.update(HeatingDemand(
            valves_open=True,
            hvac_mode=HVACMode.HEAT,
            requested_setpoint=requested_setpoint,
            outside_temperature=10.0,
            timestamp=start_time + offset,
        ))

    assert heating_control.pwm_state.enabled is True


async def test_underheat_guard_waits_for_burner_to_establish(heating_control, coordinator, monkeypatch):
    _enable_pwm(heating_control, PWMStatus.ON)
    monkeypatch.setattr(heating_control._pwm, "update", lambda *args, **kwargs: None)

    requested_setpoint = 40.0
    start_time = timestamp()

    await coordinator.async_set_heater_state(HeaterState.ON)
    await coordinator.async_set_boiler_temperature(requested_setpoint - 4.0)
    heating_control._device_tracker._last_flame_on_at = start_time
    heating_control._control_setpoint = requested_setpoint

    await heating_control.update(HeatingDemand(
        timestamp=start_time,
        valves_open=True,
        hvac_mode=HVACMode.HEAT,
        requested_setpoint=requested_setpoint,
        outside_temperature=10.0,
    ))

    await heating_control.update(HeatingDemand(
        valves_open=True,
        hvac_mode=HVACMode.HEAT,
        requested_setpoint=requested_setpoint,
        outside_temperature=10.0,
        timestamp=start_time + UNDERHEAT_SUSTAIN_SECONDS + 1,
    ))

    assert heating_control.pwm_state.enabled is True

    await heating_control.update(HeatingDemand(
        valves_open=True,
        hvac_mode=HVACMode.HEAT,
        requested_setpoint=requested_setpoint,
        outside_temperature=10.0,
        timestamp=start_time + HEATER_STARTUP_TIMEFRAME + UNDERHEAT_SUSTAIN_SECONDS + 2,
    ))

    assert heating_control.pwm_state.enabled is False


async def test_falling_request_does_not_count_as_overshoot(heating_control, coordinator):
    await coordinator.async_set_heater_state(HeaterState.ON)
    await coordinator.async_set_boiler_temperature(62.0)
    start_time = timestamp()

    await heating_control.update(HeatingDemand(
        timestamp=start_time,
        valves_open=True,
        hvac_mode=HVACMode.HEAT,
        outside_temperature=10.0,
        requested_setpoint=65.0,
    ))

    for offset in (60.0, 120.0, 180.0):
        await heating_control.update(HeatingDemand(
            valves_open=True,
            hvac_mode=HVACMode.HEAT,
            outside_temperature=10.0,
            requested_setpoint=55.0,
            timestamp=start_time + offset,
        ))

    assert heating_control.pwm_state.enabled is False

    await heating_control.update(HeatingDemand(
        valves_open=True,
        hvac_mode=HVACMode.HEAT,
        outside_temperature=10.0,
        requested_setpoint=55.0,
        timestamp=start_time + OVERSHOOT_REFERENCE_WINDOW_SECONDS + 1,
    ))
    await heating_control.update(HeatingDemand(
        valves_open=True,
        hvac_mode=HVACMode.HEAT,
        outside_temperature=10.0,
        requested_setpoint=55.0,
        timestamp=start_time + OVERSHOOT_REFERENCE_WINDOW_SECONDS + OVERSHOOT_SUSTAIN_SECONDS + 2,
    ))

    assert heating_control.pwm_state.enabled is True


async def test_reopened_valves_wait_out_the_off_phase(heating_control, coordinator):
    await coordinator.async_set_boiler_temperature(45.0)
    pwm = heating_control._pwm
    pwm._enabled = True
    pwm._effective_on_temperature = 50.0
    base_offset = pwm._heating_system.base_offset
    setpoint = base_offset + 0.5 * (50.0 - base_offset)

    now = timestamp()
    pwm._last_update = now - 400.0
    pwm._status = PWMStatus.ON
    pwm._cycle_starts = [now - 400.0]

    await heating_control.update(_make_demand(setpoint, valves_open=False))

    assert heating_control.pwm_state.status is PWMStatus.OFF
    assert pwm.cycles_this_hour == 1

    await heating_control.update(_make_demand(setpoint))

    assert heating_control.pwm_state.status is PWMStatus.OFF
    assert pwm.cycles_this_hour == 1


async def test_fresh_ignition_keeps_flame_off_hold_when_tracker_lags(hass, monkeypatch, coordinator):
    _update_coordinator_config(coordinator)
    monkeypatch.setattr(type(coordinator), "return_temperature", property(lambda self: 30.0))

    heating_control = SatHeatingControl(hass=hass, coordinator=coordinator, config=coordinator._config)
    _enable_pwm(heating_control, PWMStatus.ON)
    monkeypatch.setattr(heating_control._pwm, "update", lambda *args, **kwargs: None)

    await coordinator.async_set_heater_state(HeaterState.OFF)
    await heating_control.update(_make_demand(40.0))

    hold = 30.0 + OPTIONS_DEFAULTS[CONF_FLAME_OFF_SETPOINT_OFFSET_CELSIUS]
    assert heating_control.control_setpoint == hold

    await coordinator.async_set_heater_state(HeaterState.ON)
    await coordinator.async_set_boiler_temperature(41.0)
    heating_control._device_tracker._last_flame_on_at = timestamp() - 1500.0
    heating_control._device_tracker._last_flame_off_at = timestamp() - 200.0

    await heating_control.update(_make_demand(40.0))

    assert heating_control.control_setpoint == hold


async def test_fresh_ignition_without_hold_derives_it_from_return_temperature(hass, monkeypatch, coordinator):
    _update_coordinator_config(coordinator)
    monkeypatch.setattr(type(coordinator), "return_temperature", property(lambda self: 30.0))

    heating_control = SatHeatingControl(hass=hass, coordinator=coordinator, config=coordinator._config)
    _enable_pwm(heating_control, PWMStatus.ON)
    monkeypatch.setattr(heating_control._pwm, "update", lambda *args, **kwargs: None)

    await coordinator.async_set_heater_state(HeaterState.ON)
    await coordinator.async_set_boiler_temperature(41.0)
    heating_control._control_setpoint = MINIMUM_SETPOINT

    await heating_control.update(_make_demand(40.0))

    assert heating_control.control_setpoint == 30.0 + OPTIONS_DEFAULTS[CONF_FLAME_OFF_SETPOINT_OFFSET_CELSIUS]


async def test_pwm_flame_off_without_return_temperature_offsets_the_flow(hass, monkeypatch, coordinator):
    _update_coordinator_config(coordinator)
    monkeypatch.setattr(type(coordinator), "return_temperature", property(lambda self: None))

    heating_control = SatHeatingControl(hass=hass, coordinator=coordinator, config=coordinator._config)
    _enable_pwm(heating_control, PWMStatus.ON)
    monkeypatch.setattr(heating_control._pwm, "update", lambda *args, **kwargs: None)

    await coordinator.async_set_heater_state(HeaterState.OFF)
    await coordinator.async_set_boiler_temperature(35.0)
    heating_control._control_setpoint = MINIMUM_SETPOINT

    await heating_control.update(_make_demand(40.0))

    assert heating_control.control_setpoint == 35.0 + OPTIONS_DEFAULTS[CONF_FLAME_OFF_SETPOINT_OFFSET_CELSIUS]


async def test_fresh_ignition_without_return_temperature_holds_from_the_flow(hass, monkeypatch, coordinator):
    _update_coordinator_config(coordinator)
    monkeypatch.setattr(type(coordinator), "return_temperature", property(lambda self: None))

    heating_control = SatHeatingControl(hass=hass, coordinator=coordinator, config=coordinator._config)
    _enable_pwm(heating_control, PWMStatus.ON)
    monkeypatch.setattr(heating_control._pwm, "update", lambda *args, **kwargs: None)

    await coordinator.async_set_heater_state(HeaterState.ON)
    await coordinator.async_set_boiler_temperature(37.0)
    heating_control._control_setpoint = MINIMUM_SETPOINT

    await heating_control.update(_make_demand(40.0))

    assert heating_control.control_setpoint == 37.0 + OPTIONS_DEFAULTS[CONF_FLAME_OFF_SETPOINT_OFFSET_CELSIUS]


async def _saturated_demand(heating_control, coordinator, monkeypatch, target_changed_seconds_ago: float) -> None:
    _enable_pwm(heating_control, PWMStatus.ON)
    monkeypatch.setattr(heating_control._pwm, "update", lambda *args, **kwargs: None)

    await coordinator.async_set_heater_state(HeaterState.ON)
    await coordinator.async_set_boiler_temperature(60.0)
    heating_control._pwm._duty_cycle = (300, 0)

    now = timestamp()
    heating_control._pwm._effective_on_temperature = 44.0
    heating_control._pwm._floor_observed_at = now
    await heating_control.update(HeatingDemand(
        timestamp=now,
        valves_open=True,
        hvac_mode=HVACMode.HEAT,
        requested_setpoint=40.0,
        outside_temperature=10.0,
        target_temperature_changed_at=now - target_changed_seconds_ago,
    ))


async def test_saturation_right_after_a_target_change_leaves_pwm_at_once(heating_control, coordinator, monkeypatch):
    await _saturated_demand(heating_control, coordinator, monkeypatch, target_changed_seconds_ago=20.0)

    assert heating_control.pwm_state.enabled is False


async def test_saturation_long_after_a_target_change_still_waits(heating_control, coordinator, monkeypatch):
    await _saturated_demand(heating_control, coordinator, monkeypatch, target_changed_seconds_ago=TARGET_TEMPERATURE_CHANGE_WINDOW_SECONDS + 1)

    assert heating_control.pwm_state.enabled is True


async def test_burner_counts_as_recent_for_one_pwm_period(heating_control, coordinator, monkeypatch):
    monkeypatch.setattr(type(coordinator), "flame_active", property(lambda self: False))
    now = timestamp()
    period = heating_control._pwm.maximum_cycle_seconds

    assert heating_control.burned_within_pwm_period(now) is False

    heating_control._device_tracker._last_flame_off_at = now - period + 1.0
    assert heating_control.burned_within_pwm_period(now) is True

    heating_control._device_tracker._last_flame_off_at = now - period - 1.0
    assert heating_control.burned_within_pwm_period(now) is False

    monkeypatch.setattr(type(coordinator), "flame_active", property(lambda self: True))
    assert heating_control.burned_within_pwm_period(now) is True


def _demand_at(requested_setpoint: float, at: float, target_temperature_changed_at: Optional[float] = None) -> HeatingDemand:
    return HeatingDemand(
        timestamp=at,
        valves_open=True,
        hvac_mode=HVACMode.HEAT,
        outside_temperature=10.0,
        requested_setpoint=requested_setpoint,
        target_temperature_changed_at=target_temperature_changed_at,
    )


async def _prime_saturated_pwm(heating_control, coordinator, monkeypatch, floor: Optional[float], observed_at: Optional[float]) -> None:
    _enable_pwm(heating_control, PWMStatus.ON)
    monkeypatch.setattr(heating_control._pwm, "update", lambda *args, **kwargs: None)

    await coordinator.async_set_heater_state(HeaterState.ON)
    await coordinator.async_set_boiler_temperature(21.0)

    heating_control._pwm._duty_cycle = (2400, 0)
    heating_control._pwm._effective_on_temperature = floor
    heating_control._pwm._floor_observed_at = observed_at


async def test_saturation_without_learned_floor_keeps_pwm(heating_control, coordinator, monkeypatch):
    await _prime_saturated_pwm(heating_control, coordinator, monkeypatch, floor=None, observed_at=None)
    now = timestamp()

    await heating_control.update(_demand_at(32.0, now))
    await heating_control.update(_demand_at(32.0, now + SATURATION_SUSTAIN_SECONDS + 1))

    assert heating_control.pwm_state.enabled is True


async def test_saturation_after_target_change_without_floor_keeps_pwm(heating_control, coordinator, monkeypatch):
    await _prime_saturated_pwm(heating_control, coordinator, monkeypatch, floor=None, observed_at=None)
    now = timestamp()

    await heating_control.update(_demand_at(32.0, now, target_temperature_changed_at=now - 20.0))

    assert heating_control.pwm_state.enabled is True


async def test_saturation_on_a_prior_floor_needs_a_clear_margin(heating_control, coordinator, monkeypatch):
    await _prime_saturated_pwm(heating_control, coordinator, monkeypatch, floor=44.0, observed_at=None)
    now = timestamp()

    await heating_control.update(_demand_at(46.0, now, target_temperature_changed_at=now - 20.0))
    assert heating_control.pwm_state.enabled is True

    await heating_control.update(_demand_at(44.0 + OVERSHOOT_MARGIN_CELSIUS, now + 1.0, target_temperature_changed_at=now - 20.0))
    assert heating_control.pwm_state.enabled is False


async def test_saturation_on_an_observed_floor_uses_plain_duty(heating_control, coordinator, monkeypatch):
    now = timestamp()
    await _prime_saturated_pwm(heating_control, coordinator, monkeypatch, floor=44.0, observed_at=now)

    await heating_control.update(_demand_at(45.0, now))
    assert heating_control.pwm_state.enabled is True

    await heating_control.update(_demand_at(45.0, now + SATURATION_SUSTAIN_SECONDS + 1))
    assert heating_control.pwm_state.enabled is False


async def test_cold_ignition_keeps_pwm_with_a_known_floor(heating_control, coordinator, monkeypatch):
    flame = {"active": False}
    monkeypatch.setattr(type(coordinator), "flame_active", property(lambda self: flame["active"]))

    await coordinator.async_set_heater_state(HeaterState.ON)
    await coordinator.async_set_boiler_temperature(21.0)

    now = timestamp()
    pwm = heating_control._pwm
    pwm._enabled = True
    pwm._last_update = now
    pwm._effective_on_temperature = 44.0

    await heating_control.update(_demand_at(32.0, now, target_temperature_changed_at=now))
    assert heating_control.pwm_state.status is PWMStatus.ON

    await heating_control.update(_demand_at(32.0, now + SATURATION_SUSTAIN_SECONDS + 1, target_temperature_changed_at=now))
    assert heating_control.pwm_state.enabled is True

    flame["active"] = True
    await coordinator.async_set_boiler_temperature(27.0)

    await heating_control.update(_demand_at(32.0, now + SATURATION_SUSTAIN_SECONDS + 31, target_temperature_changed_at=now))
    assert heating_control.pwm_state.enabled is True
    assert heating_control.pwm_state.effective_on_temperature == 44.0

    target_changed_at = now + SATURATION_SUSTAIN_SECONDS + 41
    await heating_control.update(_demand_at(61.0, now + SATURATION_SUSTAIN_SECONDS + 61, target_temperature_changed_at=target_changed_at))
    assert heating_control.pwm_state.enabled is False


async def test_coordinator_release_switches_heating_off_once(coordinator, monkeypatch):
    await coordinator.async_set_heater_state(HeaterState.ON)
    await coordinator.async_set_control_setpoint(50.0)

    await coordinator.async_release()

    assert coordinator.released is True
    assert coordinator.active is False
    assert coordinator.setpoint == MINIMUM_SETPOINT

    release = AsyncMock()
    monkeypatch.setattr(coordinator, "_async_release", release)
    await coordinator.async_release()

    release.assert_not_awaited()


async def test_released_boiler_is_kept_released(heating_control, coordinator, monkeypatch):
    await heating_control.async_release()

    keep_released = AsyncMock()
    monkeypatch.setattr(coordinator, "_async_keep_released", keep_released)
    await heating_control.async_release()

    keep_released.assert_awaited_once()
    assert coordinator.released is True


async def test_commanding_again_ends_the_release(coordinator):
    await coordinator.async_release()
    await coordinator.async_set_control_setpoint(45.0)

    assert coordinator.released is False


async def test_undelivered_release_is_retried(coordinator, monkeypatch):
    monkeypatch.setattr(type(coordinator), "available", property(lambda self: False))

    await coordinator.async_release()

    assert coordinator.released is False


async def test_release_waits_for_an_update_in_flight(heating_control, coordinator, monkeypatch):
    sending, finish = asyncio.Event(), asyncio.Event()

    async def _slow_modulation(value):
        sending.set()
        await finish.wait()

    monkeypatch.setattr(coordinator, "async_set_control_max_relative_modulation", _slow_modulation)
    update = asyncio.create_task(heating_control.update(_make_demand(45.0)))
    await sending.wait()
    release = asyncio.create_task(heating_control.async_release())
    await asyncio.sleep(0)

    assert coordinator.released is False

    finish.set()
    await asyncio.gather(update, release)

    assert coordinator.released is True
    assert coordinator.active is False


async def test_backend_that_never_returns_does_not_block_the_next_command(heating_control, coordinator, monkeypatch):
    monkeypatch.setattr("custom_components.sat.heating_control.COMMAND_TIMEOUT_SECONDS", 0.01)

    async def _never_returns(value):
        await asyncio.Event().wait()

    with monkeypatch.context() as hanging:
        hanging.setattr(coordinator, "async_set_control_max_relative_modulation", _never_returns)

        with pytest.raises(TimeoutError):
            await heating_control.update(_make_demand(45.0))

    await heating_control.update(_make_demand(45.0))
    await heating_control.async_release()

    assert coordinator.released is True


OFFSET = OPTIONS_DEFAULTS[CONF_FLAME_OFF_SETPOINT_OFFSET_CELSIUS]


def _report_hot_water(heating_control, monkeypatch, *, flame: bool = False, persisted: bool = True) -> dict:
    """Report hot water, and return the burner flag so a test can light or drop the flame mid-episode.

    Also declares flame detection as supported, because the base coordinator derives flame activity from the
    heating state instead of measuring it and therefore reports no corroboration capability by default. A lit
    flame is recorded as a running ignition in the device tracker, since corroboration requires the flame to
    have persisted through the ignition grace; pass ``persisted=False`` to model one that has only just lit.
    """
    burner = {"flame": flame}

    monkeypatch.setattr(type(heating_control._coordinator), "hot_water_active", property(lambda self: True))
    monkeypatch.setattr(type(heating_control._coordinator), "flame_active", property(lambda self: burner["flame"]))
    monkeypatch.setattr(type(heating_control._coordinator), "supports_flame_detection", property(lambda self: True))
    monkeypatch.setattr(heating_control._pwm, "update", lambda *args, **kwargs: None)

    if flame:
        elapsed = (DHW_IGNITION_GRACE_SECONDS + 2.0) if persisted else 0.0
        heating_control._device_tracker._last_flame_on_at = timestamp() - elapsed

    return burner


async def test_hot_water_within_cap_forces_minimum_over_stale_setpoint(heating_control, monkeypatch):
    _enable_pwm(heating_control, PWMStatus.ON)
    _report_hot_water(heating_control, monkeypatch, flame=True)
    heating_control._device_tracker._last_hot_water_on_at = timestamp() - 60.0
    heating_control._control_setpoint = 46.2
    heating_control._flame_off_hold_setpoint = 35.0

    await heating_control.update(_make_demand(46.2))

    assert heating_control.control_setpoint == MINIMUM_SETPOINT
    assert heating_control._flame_off_hold_setpoint is None


async def test_stale_hot_water_beyond_cap_resumes_heating_control(heating_control, coordinator, monkeypatch):
    _enable_pwm(heating_control, PWMStatus.ON)
    _report_hot_water(heating_control, monkeypatch)
    monkeypatch.setattr(type(coordinator), "return_temperature", property(lambda self: 30.0))

    heating_control._device_tracker._last_hot_water_on_at = timestamp() - (DHW_SUPPRESSION_MAX_SECONDS + 1.0)
    heating_control._control_setpoint = MINIMUM_SETPOINT

    await coordinator.async_set_heater_state(HeaterState.OFF)
    await heating_control.update(_make_demand(46.2))

    assert heating_control.control_setpoint == 30.0 + OFFSET


async def test_hot_water_without_tracker_timestamp_counts_as_fresh(heating_control, monkeypatch):
    _enable_pwm(heating_control, PWMStatus.ON)
    _report_hot_water(heating_control, monkeypatch)
    heating_control._device_tracker._last_hot_water_on_at = None
    heating_control._control_setpoint = 46.2

    await heating_control.update(_make_demand(46.2))

    assert heating_control.control_setpoint == MINIMUM_SETPOINT


async def test_hot_water_with_lagging_tracker_still_suppresses(heating_control, monkeypatch):
    """A new episode reported before the tracker catches up must not read as stale.

    The tracker keeps the previous episode's ON timestamp until it observes the transition, so an ON
    timestamp that predates the last OFF belongs to an earlier episode and counts as just started.
    """
    _enable_pwm(heating_control, PWMStatus.ON)
    _report_hot_water(heating_control, monkeypatch)
    heating_control._device_tracker._last_hot_water_on_at = timestamp() - (DHW_SUPPRESSION_MAX_SECONDS + 100.0)
    heating_control._device_tracker._last_hot_water_off_at = timestamp() - 10.0
    heating_control._control_setpoint = 46.2

    await heating_control.update(_make_demand(46.2))

    assert heating_control.control_setpoint == MINIMUM_SETPOINT


async def test_hot_water_without_tracker_timestamp_still_hits_the_cap(heating_control, coordinator, monkeypatch):
    """A report the tracker never timestamped must still be bounded by the backstop.

    _elapsed_since_hot_water_on used to return 0.0 when the tracker had no ON timestamp, which reads as "just
    started" forever and disables the only guard a stale report has.
    """
    _enable_pwm(heating_control, PWMStatus.ON)
    _report_hot_water(heating_control, monkeypatch, flame=False)
    monkeypatch.setattr(type(coordinator), "supports_flame_detection", property(lambda self: False))
    monkeypatch.setattr(type(coordinator), "return_temperature", property(lambda self: 30.0))

    heating_control._device_tracker._last_hot_water_on_at = None
    heating_control._hot_water_episode_started_at = timestamp() - (DHW_SUPPRESSION_MAX_SECONDS + 1.0)

    await heating_control.update(_make_demand(46.2))

    assert heating_control.control_setpoint == 30.0 + OFFSET


async def test_elapsed_since_hot_water_on_clamps_a_backwards_clock(heating_control):
    """Wall-clock timestamps can move backwards (NTP or DST); elapsed must not go negative."""
    heating_control._device_tracker._last_hot_water_on_at = timestamp() + 3600.0

    assert heating_control._elapsed_since_hot_water_on() == 0.0


async def test_reset_forgets_the_hot_water_corroboration(heating_control):
    """A reset must not leave stale corroboration behind for the next episode."""
    heating_control._dhw_flame_seen_at = timestamp()
    heating_control._hot_water_episode_started_at = timestamp()
    heating_control._stale_hot_water_logged = True
    heating_control._unconfirmed_hot_water_logged = True

    heating_control.reset()

    assert heating_control._dhw_flame_seen_at is None
    assert heating_control._hot_water_episode_started_at is None
    assert heating_control._stale_hot_water_logged is False
    assert heating_control._unconfirmed_hot_water_logged is False


async def test_hot_water_without_flame_does_not_suppress_heating(heating_control, coordinator, monkeypatch, caplog):
    """The DHW bit alone is not a draw.

    The OpenTherm status bit reports "central heating is not being served", so it turns on as a side effect of
    the PWM OFF phase dropping the CH enable. A boiler that claims hot water without firing must not keep
    central heating suppressed, and the release must be logged once per episode.
    """
    _enable_pwm(heating_control, PWMStatus.ON)
    burner = _report_hot_water(heating_control, monkeypatch, flame=False)
    monkeypatch.setattr(type(coordinator), "return_temperature", property(lambda self: 30.0))

    heating_control._device_tracker._last_hot_water_on_at = timestamp() - 60.0
    heating_control._control_setpoint = MINIMUM_SETPOINT

    with caplog.at_level(logging.DEBUG, logger="custom_components.sat.heating_control"):
        await heating_control.update(_make_demand(46.2))
        await heating_control.update(_make_demand(46.2))

    assert burner["flame"] is False
    assert heating_control.control_setpoint == 30.0 + OFFSET

    releases = [record.getMessage() for record in caplog.records if "not firing" in record.getMessage()]
    assert len(releases) == 1
    # Per-cycle noise: the message must stay at DEBUG, not warn the user on every PWM off-phase.
    assert not [record for record in caplog.records if "not firing" in record.getMessage() and record.levelno >= logging.INFO]


async def test_hot_water_suppression_survives_flame_drop_inside_hold_window(heating_control, monkeypatch):
    """A long draw modulates the burner; the drop must not switch central heating back on within the hold."""
    _enable_pwm(heating_control, PWMStatus.ON)
    burner = _report_hot_water(heating_control, monkeypatch, flame=True)

    heating_control._device_tracker._last_hot_water_on_at = timestamp() - 30.0

    await heating_control.update(_make_demand(46.2))
    assert heating_control.control_setpoint == MINIMUM_SETPOINT
    assert heating_control._dhw_flame_seen_at is not None  # the corroboration is remembered for the hold

    burner["flame"] = False
    await heating_control.update(_make_demand(46.2))

    assert heating_control.control_setpoint == MINIMUM_SETPOINT


async def test_hot_water_suppression_released_after_flame_hold_expires(heating_control, coordinator, monkeypatch, caplog):
    _enable_pwm(heating_control, PWMStatus.ON)
    burner = _report_hot_water(heating_control, monkeypatch, flame=True)
    monkeypatch.setattr(type(coordinator), "return_temperature", property(lambda self: 30.0))

    heating_control._device_tracker._last_hot_water_on_at = timestamp() - 120.0

    await heating_control.update(_make_demand(46.2))
    assert heating_control.control_setpoint == MINIMUM_SETPOINT

    burner["flame"] = False
    heating_control._dhw_flame_seen_at = timestamp() - (DHW_FLAME_HOLD_SECONDS + 1.0)

    with caplog.at_level(logging.DEBUG, logger="custom_components.sat.heating_control"):
        await heating_control.update(_make_demand(46.2))

    assert heating_control.control_setpoint == 30.0 + OFFSET
    assert any("not firing" in record.getMessage() for record in caplog.records)


async def test_hot_water_ignition_grace_still_suppresses_before_the_burner_lights(heating_control, monkeypatch):
    _enable_pwm(heating_control, PWMStatus.ON)
    _report_hot_water(heating_control, monkeypatch, flame=False)

    heating_control._device_tracker._last_hot_water_on_at = timestamp() - (DHW_IGNITION_GRACE_SECONDS - 5.0)
    heating_control._control_setpoint = 46.2

    await heating_control.update(_make_demand(46.2))

    assert heating_control.control_setpoint == MINIMUM_SETPOINT
    assert heating_control._unconfirmed_hot_water_logged is False  # still inside the grace, nothing released


async def test_fresh_flame_provoked_by_the_released_setpoint_does_not_latch(heating_control, coordinator, monkeypatch):
    """A CH relight provoked by our own setpoint drop must not be counted as a draw.

    Sequence behind the loop this guards against: the spurious DHW bit outlives the ignition grace, the
    predicate releases and publishes the heating setpoint, and the burner relights for central heating while
    the bit is still set. That flame is seconds old, so it must neither confirm the report nor arm the hold.
    """
    _enable_pwm(heating_control, PWMStatus.ON)
    burner = _report_hot_water(heating_control, monkeypatch, flame=True, persisted=False)
    monkeypatch.setattr(type(coordinator), "return_temperature", property(lambda self: 30.0))

    heating_control._device_tracker._last_hot_water_on_at = timestamp() - 60.0

    await heating_control.update(_make_demand(46.2))

    assert burner["flame"] is True
    assert heating_control.hot_water_suppresses_heating is False
    assert heating_control._dhw_flame_seen_at is None
    assert heating_control.control_setpoint == 30.0 + OFFSET


async def test_hot_water_cap_still_releases_heating_with_a_burning_boiler(heating_control, coordinator, monkeypatch, caplog):
    """The 1800 s cap stays a backstop: it releases heating even while the flame corroborates the report."""
    _enable_pwm(heating_control, PWMStatus.ON)
    _report_hot_water(heating_control, monkeypatch, flame=True)
    monkeypatch.setattr(type(coordinator), "return_temperature", property(lambda self: 30.0))

    heating_control._device_tracker._last_hot_water_on_at = timestamp() - (DHW_SUPPRESSION_MAX_SECONDS + 1.0)

    with caplog.at_level(logging.DEBUG, logger="custom_components.sat.heating_control"):
        await heating_control.update(_make_demand(46.2))

    assert heating_control.control_setpoint == 30.0 + OFFSET
    assert any("treating as stale" in record.getMessage() for record in caplog.records)


async def test_hot_water_flame_memory_is_cleared_between_episodes(heating_control, monkeypatch):
    """A new episode must not inherit the hold-off recorded for the previous draw."""
    _enable_pwm(heating_control, PWMStatus.ON)
    burner = _report_hot_water(heating_control, monkeypatch, flame=True)

    heating_control._device_tracker._last_hot_water_on_at = timestamp() - 30.0
    await heating_control.update(_make_demand(46.2))

    monkeypatch.setattr(type(heating_control._coordinator), "hot_water_active", property(lambda self: False))
    burner["flame"] = False
    await heating_control.update(_make_demand(46.2))

    assert heating_control._dhw_flame_seen_at is None


async def test_hot_water_flame_memory_cleared_when_episode_ends_while_heating_off(heating_control, coordinator, monkeypatch):
    """An episode ending while heating is off must not leak the flame memory into the next episode.

    The reset cannot live in the hot-water branch of _compute_pwm_control_setpoint, because that branch is only
    reached with heating running under PWM. Here the episode ends with heating off, so no demand is pushed at
    all. Without a reset on the always-run path, the next unlit episode inherits a flame timestamp from the
    previous draw and stays suppressed for the whole flame hold instead of releasing after its own grace.
    """
    _enable_pwm(heating_control, PWMStatus.ON)
    report = {"hot_water": True}
    burner = _report_hot_water(heating_control, monkeypatch, flame=True)
    # Patch after the helper, which sets hot_water_active to a constant True and would otherwise win.
    monkeypatch.setattr(type(coordinator), "hot_water_active", property(lambda self: report["hot_water"]))
    monkeypatch.setattr(type(coordinator), "return_temperature", property(lambda self: 30.0))

    heating_control._device_tracker._last_hot_water_on_at = timestamp() - 30.0
    await heating_control.update(_make_demand(46.2))

    assert heating_control.control_setpoint == MINIMUM_SETPOINT
    assert heating_control._dhw_flame_seen_at is not None

    # The report ends while heating is off: no demand is pushed, only the always-run coordinator path ticks.
    report["hot_water"] = False
    burner["flame"] = False
    heating_control._handle_coordinator_update()

    assert heating_control._dhw_flame_seen_at is None
    assert heating_control._unconfirmed_hot_water_logged is False
    assert heating_control._stale_hot_water_logged is False

    # A new unlit episode must release after its own ignition grace, not the previous hold window.
    report["hot_water"] = True
    heating_control._device_tracker._last_hot_water_on_at = timestamp() - (DHW_IGNITION_GRACE_SECONDS + 5.0)

    await heating_control.update(_make_demand(46.2))

    assert heating_control.control_setpoint == 30.0 + OFFSET


async def test_hot_water_flame_observed_while_heating_off_survives_into_pwm(heating_control, coordinator, monkeypatch):
    """A draw that burns while heating is off must leave corroboration for the hold window.

    The flame timestamp used to be recorded only inside _compute_pwm_control_setpoint, which does not run while
    heating is off. A genuine draw burning during that time then left no corroboration, and when PWM heating
    resumed during a burner pause past the ignition grace the heating restarted although the flame had been
    seen seconds earlier.
    """
    _enable_pwm(heating_control, PWMStatus.ON)
    report = {"hot_water": True}
    burner = _report_hot_water(heating_control, monkeypatch, flame=True)
    monkeypatch.setattr(type(coordinator), "hot_water_active", property(lambda self: report["hot_water"]))
    monkeypatch.setattr(type(coordinator), "return_temperature", property(lambda self: 30.0))

    # The draw burns while no demand is pushed, i.e. only the always-run coordinator path ticks.
    heating_control._device_tracker._last_hot_water_on_at = timestamp() - 30.0
    heating_control._handle_coordinator_update()

    assert heating_control._dhw_flame_seen_at is not None

    # PWM heating resumes during a burner pause, past the ignition grace: the hold must still suppress.
    burner["flame"] = False
    heating_control._device_tracker._last_hot_water_on_at = timestamp() - (DHW_IGNITION_GRACE_SECONDS + 5.0)

    await heating_control.update(_make_demand(46.2))

    assert heating_control.control_setpoint == MINIMUM_SETPOINT


async def test_hot_water_without_flame_telemetry_keeps_capped_suppression(heating_control, coordinator, monkeypatch):
    """A device that cannot report the flame must not be read as "the burner is off".

    Backends such as ESPHome return None for a configured but missing flame sensor and collapse that to False.
    Treating that as proof of no draw would release central heating after the ignition grace during a genuine
    hot-water episode. Without corroboration the previous capped behaviour has to stand.
    """
    _enable_pwm(heating_control, PWMStatus.ON)
    burner = _report_hot_water(heating_control, monkeypatch, flame=False)
    monkeypatch.setattr(type(coordinator), "supports_flame_detection", property(lambda self: False))

    heating_control._device_tracker._last_hot_water_on_at = timestamp() - 60.0
    heating_control._control_setpoint = 46.2

    await heating_control.update(_make_demand(46.2))

    assert burner["flame"] is False
    assert heating_control.control_setpoint == MINIMUM_SETPOINT


async def test_hot_water_without_flame_telemetry_is_still_capped(heating_control, coordinator, monkeypatch):
    """Capped has to mean the cap fires: an uncorroborated backend still releases at the backstop."""
    _enable_pwm(heating_control, PWMStatus.ON)
    _report_hot_water(heating_control, monkeypatch, flame=False)
    monkeypatch.setattr(type(coordinator), "supports_flame_detection", property(lambda self: False))
    monkeypatch.setattr(type(coordinator), "return_temperature", property(lambda self: 30.0))

    heating_control._device_tracker._last_hot_water_on_at = timestamp() - (DHW_SUPPRESSION_MAX_SECONDS + 1.0)

    await heating_control.update(_make_demand(46.2))

    assert heating_control.control_setpoint == 30.0 + OFFSET


async def test_uncorroborated_hot_water_does_not_command_maximum_modulation(heating_control, coordinator, monkeypatch):
    """The reported state stays raw, but the command must follow the corroborated predicate.

    The DHW bit is asserted by our own PWM off-phase, so commanding maximum modulation off it while the
    entity reports that no draw is happening both fights the boiler and contradicts the reported reason.
    """
    _enable_pwm(heating_control, PWMStatus.ON)
    _report_hot_water(heating_control, monkeypatch, flame=False)
    coordinator.config.supports_relative_modulation_management = True

    heating_control._device_tracker._last_hot_water_on_at = timestamp() - 60.0

    sent = []
    original = coordinator.async_set_control_max_relative_modulation

    async def _record(value):
        sent.append(value)
        await original(value)

    monkeypatch.setattr(coordinator, "async_set_control_max_relative_modulation", _record)

    await heating_control.update(_make_demand(46.2))

    assert heating_control.relative_modulation_state is RelativeModulationState.HOT_WATER
    assert heating_control.hot_water_suppresses_heating is False
    assert sent[-1] == MINIMUM_RELATIVE_MODULATION


async def test_hot_water_episode_does_not_hold_the_boiler_while_released(heating_control, coordinator, monkeypatch):
    """Episode tracking must not command the boiler, so a release cannot be held by a reported draw.

    The tracking path runs on every coordinator update, including while the boiler is released. If it
    commanded, a reported draw would take control back from the release on the next tick.
    """
    _enable_pwm(heating_control, PWMStatus.ON)
    _report_hot_water(heating_control, monkeypatch, flame=True)

    setpoint_command = AsyncMock()
    heater_command = AsyncMock()
    modulation_command = AsyncMock()
    monkeypatch.setattr(coordinator, "async_set_control_setpoint", setpoint_command)
    monkeypatch.setattr(coordinator, "async_set_heater_state", heater_command)
    monkeypatch.setattr(coordinator, "async_set_control_max_relative_modulation", modulation_command)

    heating_control._device_tracker._last_hot_water_on_at = timestamp() - 30.0
    heating_control._handle_coordinator_update()

    setpoint_command.assert_not_awaited()
    heater_command.assert_not_awaited()
    modulation_command.assert_not_awaited()

    await heating_control.async_release()

    assert coordinator.released is True
    assert coordinator.active is False


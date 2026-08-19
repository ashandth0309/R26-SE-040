"""Tests for the BUDDY hardware abstraction layer."""

from __future__ import annotations

import builtins

import pytest

from buddy_core.hardware import (
    BuddyHardwareError,
    MockGPIOBackend,
    get_gpio_backend,
)


def make_config(
    *,
    enabled: bool = False,
    simulation: bool = True,
    numbering_mode: str = "BCM",
):
    return {
        "hardware": {
            "enabled": enabled,
            "simulation": simulation,
            "gpio": {
                "numbering_mode": numbering_mode,
            },
        }
    }


def test_mock_backend_initialization():
    backend = MockGPIOBackend()

    assert backend.pin_modes == {}
    assert backend.pin_values == {}
    assert backend.pwm_channels == {}
    assert backend.cleaned_up is False


def test_configure_digital_output():
    backend = MockGPIOBackend()

    backend.configure_output(
        17,
        initial=False,
    )

    assert backend.pin_modes[17] == "output"
    assert backend.pin_values[17] is False


def test_digital_output_state_change():
    backend = MockGPIOBackend()

    backend.configure_output(
        17,
        initial=False,
    )

    backend.write(
        17,
        True,
    )

    assert backend.pin_values[17] is True


def test_write_requires_output_configuration():
    backend = MockGPIOBackend()

    with pytest.raises(BuddyHardwareError):
        backend.write(
            17,
            True,
        )


def test_configure_and_read_input():
    backend = MockGPIOBackend()

    backend.configure_input(
        22,
        pull="down",
    )

    backend.set_mock_input(
        22,
        True,
    )

    assert backend.read(22) is True
    assert backend.pin_modes[22] == "input"
    assert backend.pull_modes[22] == "down"


def test_invalid_input_pull_mode():
    backend = MockGPIOBackend()

    with pytest.raises(BuddyHardwareError):
        backend.configure_input(
            22,
            pull="invalid",
        )


def test_pwm_setup():
    backend = MockGPIOBackend()

    state = backend.setup_pwm(
        18,
        frequency_hz=1000,
        duty_cycle=25,
    )

    assert backend.pin_modes[18] == "pwm"
    assert state.frequency_hz == 1000.0
    assert state.duty_cycle == 25.0


def test_pwm_update():
    backend = MockGPIOBackend()

    backend.setup_pwm(
        18,
        frequency_hz=1000,
        duty_cycle=0,
    )

    backend.update_pwm(
        18,
        60,
    )

    assert (
        backend.pwm_channels[18].duty_cycle
        == 60.0
    )


def test_invalid_pwm_duty_cycle():
    backend = MockGPIOBackend()

    with pytest.raises(BuddyHardwareError):
        backend.setup_pwm(
            18,
            frequency_hz=1000,
            duty_cycle=101,
        )


def test_cleanup_returns_mock_outputs_safe():
    backend = MockGPIOBackend()

    backend.configure_output(
        17,
        initial=True,
    )

    backend.setup_pwm(
        18,
        frequency_hz=1000,
        duty_cycle=75,
    )

    backend.cleanup()

    assert backend.pin_values[17] is False
    assert (
        backend.pwm_channels[18].duty_cycle
        == 0.0
    )
    assert backend.cleaned_up is True


def test_hardware_disabled_uses_mock_backend():
    backend = get_gpio_backend(
        make_config(
            enabled=False,
            simulation=False,
        )
    )

    assert isinstance(
        backend,
        MockGPIOBackend,
    )


def test_simulation_mode_uses_mock_backend():
    backend = get_gpio_backend(
        make_config(
            enabled=True,
            simulation=True,
        )
    )

    assert isinstance(
        backend,
        MockGPIOBackend,
    )


def test_missing_hardware_config_fails_safely():
    with pytest.raises(BuddyHardwareError):
        get_gpio_backend({})


def test_invalid_hardware_enabled_fails_safely():
    config = make_config()

    config["hardware"]["enabled"] = "yes"

    with pytest.raises(BuddyHardwareError):
        get_gpio_backend(config)


def test_invalid_gpio_numbering_mode_fails():
    config = make_config(
        numbering_mode="INVALID",
    )

    with pytest.raises(BuddyHardwareError):
        get_gpio_backend(config)


def test_missing_raspberry_pi_dependency_fails_safely(
    monkeypatch,
):
    config = make_config(
        enabled=True,
        simulation=False,
    )

    original_import = builtins.__import__

    def blocked_import(
        name,
        globals=None,
        locals=None,
        fromlist=(),
        level=0,
    ):
        if name == "RPi.GPIO":
            raise ImportError(
                "RPi.GPIO intentionally unavailable"
            )

        return original_import(
            name,
            globals,
            locals,
            fromlist,
            level,
        )

    monkeypatch.setattr(
        builtins,
        "__import__",
        blocked_import,
    )

    with pytest.raises(
        BuddyHardwareError,
    ):
        get_gpio_backend(config)

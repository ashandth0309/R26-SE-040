"""Safe configuration-driven motor-system factory for BUDDY."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from buddy_core.hardware import (
    BuddyHardwareError,
    GPIOBackend,
    get_gpio_backend,
)
from buddy_core.logging_utils import get_buddy_logger
from buddy_core.motors.drive import BuddyDrive
from buddy_core.motors.tb6612 import (
    TB6612Motor,
    TB6612Standby,
)


@dataclass
class BuddyMotorSystem:
    """Complete safe motor subsystem for BUDDY's 4WD base."""

    gpio: GPIOBackend
    drive: BuddyDrive
    driver_1_standby: TB6612Standby
    driver_2_standby: TB6612Standby
    initialized: bool = False

    def initialize(self) -> None:
        """Initialize motor control with both drivers disabled."""
        self.driver_1_standby.initialize()
        self.driver_2_standby.initialize()

        try:
            self.drive.initialize()
        except Exception:
            self.driver_1_standby.disable()
            self.driver_2_standby.disable()
            raise

        self.initialized = True

    def enable_drivers(self) -> None:
        """Explicitly enable both TB6612 driver boards."""
        if not self.initialized:
            raise RuntimeError(
                "Motor system must be initialized before "
                "enabling the drivers."
            )

        self.driver_1_standby.enable()
        self.driver_2_standby.enable()

    def disable_drivers(self) -> None:
        """Stop all motors and disable both TB6612 boards."""
        self.drive.stop()
        self.driver_1_standby.disable()
        self.driver_2_standby.disable()

    def cleanup(self) -> None:
        """Return the complete motor subsystem to a safe state."""
        self.disable_drivers()
        self.drive.cleanup()
        self.driver_1_standby.cleanup()
        self.driver_2_standby.cleanup()
        self.gpio.cleanup()
        self.initialized = False


def _require_mapping(
    parent: Mapping[str, Any],
    key: str,
    path: str,
) -> Mapping[str, Any]:
    value = parent.get(key)

    if not isinstance(value, Mapping):
        raise BuddyHardwareError(
            f"{path} must be a mapping."
        )

    return value


def _require_pin(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
) -> int:
    value = mapping.get(key)

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise BuddyHardwareError(
            f"{path} must be a configured "
            "non-negative GPIO integer."
        )

    return value


def create_motor_system(
    config: Mapping[str, Any],
) -> BuddyMotorSystem:
    """Create BUDDY's safe four-motor system from configuration."""

    hardware = _require_mapping(
        config,
        "hardware",
        "hardware",
    )

    motors = _require_mapping(
        hardware,
        "motors",
        "hardware.motors",
    )

    motors_enabled = motors.get("enabled")

    if not isinstance(motors_enabled, bool):
        raise BuddyHardwareError(
            "hardware.motors.enabled must be boolean."
        )

    if not motors_enabled:
        raise BuddyHardwareError(
            "Motor control is disabled by configuration."
        )

    pwm_frequency_hz = motors.get(
        "pwm_frequency_hz"
    )

    if (
        isinstance(pwm_frequency_hz, bool)
        or not isinstance(
            pwm_frequency_hz,
            (int, float),
        )
        or pwm_frequency_hz <= 0
    ):
        raise BuddyHardwareError(
            "hardware.motors.pwm_frequency_hz "
            "must be a positive number."
        )

    driver_1 = _require_mapping(
        motors,
        "driver_1",
        "hardware.motors.driver_1",
    )

    driver_2 = _require_mapping(
        motors,
        "driver_2",
        "hardware.motors.driver_2",
    )

    front_left = _require_mapping(
        driver_1,
        "front_left",
        "hardware.motors.driver_1.front_left",
    )

    front_right = _require_mapping(
        driver_1,
        "front_right",
        "hardware.motors.driver_1.front_right",
    )

    rear_left = _require_mapping(
        driver_2,
        "rear_left",
        "hardware.motors.driver_2.rear_left",
    )

    rear_right = _require_mapping(
        driver_2,
        "rear_right",
        "hardware.motors.driver_2.rear_right",
    )

    pins = {
        "driver_1_standby": _require_pin(
            driver_1,
            "standby",
            "hardware.motors.driver_1.standby",
        ),
        "front_left_in1": _require_pin(
            front_left,
            "in1",
            "hardware.motors.driver_1.front_left.in1",
        ),
        "front_left_in2": _require_pin(
            front_left,
            "in2",
            "hardware.motors.driver_1.front_left.in2",
        ),
        "front_left_pwm": _require_pin(
            front_left,
            "pwm",
            "hardware.motors.driver_1.front_left.pwm",
        ),
        "front_right_in1": _require_pin(
            front_right,
            "in1",
            "hardware.motors.driver_1.front_right.in1",
        ),
        "front_right_in2": _require_pin(
            front_right,
            "in2",
            "hardware.motors.driver_1.front_right.in2",
        ),
        "front_right_pwm": _require_pin(
            front_right,
            "pwm",
            "hardware.motors.driver_1.front_right.pwm",
        ),
        "driver_2_standby": _require_pin(
            driver_2,
            "standby",
            "hardware.motors.driver_2.standby",
        ),
        "rear_left_in1": _require_pin(
            rear_left,
            "in1",
            "hardware.motors.driver_2.rear_left.in1",
        ),
        "rear_left_in2": _require_pin(
            rear_left,
            "in2",
            "hardware.motors.driver_2.rear_left.in2",
        ),
        "rear_left_pwm": _require_pin(
            rear_left,
            "pwm",
            "hardware.motors.driver_2.rear_left.pwm",
        ),
        "rear_right_in1": _require_pin(
            rear_right,
            "in1",
            "hardware.motors.driver_2.rear_right.in1",
        ),
        "rear_right_in2": _require_pin(
            rear_right,
            "in2",
            "hardware.motors.driver_2.rear_right.in2",
        ),
        "rear_right_pwm": _require_pin(
            rear_right,
            "pwm",
            "hardware.motors.driver_2.rear_right.pwm",
        ),
    }

    pin_values = list(pins.values())

    if len(pin_values) != len(set(pin_values)):
        raise BuddyHardwareError(
            "Motor GPIO configuration contains duplicate pins."
        )

    gpio = get_gpio_backend(config)

    logger = get_buddy_logger(
        "motors.factory"
    )

    logger.info(
        "Creating BUDDY 4WD motor system."
    )

    drive = BuddyDrive(
        front_left=TB6612Motor(
            gpio,
            pins["front_left_in1"],
            pins["front_left_in2"],
            pins["front_left_pwm"],
            pwm_frequency_hz,
        ),
        front_right=TB6612Motor(
            gpio,
            pins["front_right_in1"],
            pins["front_right_in2"],
            pins["front_right_pwm"],
            pwm_frequency_hz,
        ),
        rear_left=TB6612Motor(
            gpio,
            pins["rear_left_in1"],
            pins["rear_left_in2"],
            pins["rear_left_pwm"],
            pwm_frequency_hz,
        ),
        rear_right=TB6612Motor(
            gpio,
            pins["rear_right_in1"],
            pins["rear_right_in2"],
            pins["rear_right_pwm"],
            pwm_frequency_hz,
        ),
    )

    return BuddyMotorSystem(
        gpio=gpio,
        drive=drive,
        driver_1_standby=TB6612Standby(
            gpio,
            pins["driver_1_standby"],
        ),
        driver_2_standby=TB6612Standby(
            gpio,
            pins["driver_2_standby"],
        ),
    )

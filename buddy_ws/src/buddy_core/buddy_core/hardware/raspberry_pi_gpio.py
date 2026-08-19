"""Raspberry Pi GPIO backend for BUDDY.

The Raspberry Pi-specific dependency is loaded lazily so that importing
buddy_core on a development machine does not require Raspberry Pi GPIO
packages to be installed.
"""

from __future__ import annotations

from typing import Any

from buddy_core.hardware.base import (
    BuddyHardwareError,
    GPIOBackend,
)


class RaspberryPiGPIOBackend(GPIOBackend):
    """Physical Raspberry Pi GPIO backend."""

    def __init__(
        self,
        numbering_mode: str = "BCM",
    ) -> None:
        if numbering_mode not in {"BCM", "BOARD"}:
            raise BuddyHardwareError(
                "GPIO numbering mode must be 'BCM' or 'BOARD'."
            )

        try:
            import RPi.GPIO as gpio
        except (ImportError, RuntimeError) as exc:
            raise BuddyHardwareError(
                "Physical Raspberry Pi GPIO mode was requested, "
                "but the RPi.GPIO dependency is unavailable or "
                "cannot be initialized on this system."
            ) from exc

        self._gpio = gpio
        self._numbering_mode = numbering_mode
        self._pwm_channels: dict[int, Any] = {}
        self._configured_outputs: set[int] = set()
        self._cleaned_up = False

        if numbering_mode == "BCM":
            self._gpio.setmode(self._gpio.BCM)
        else:
            self._gpio.setmode(self._gpio.BOARD)

        self._gpio.setwarnings(False)

    def configure_input(
        self,
        pin: int,
        pull: str | None = None,
    ) -> None:
        self._validate_pin(pin)

        pull_map = {
            None: self._gpio.PUD_OFF,
            "up": self._gpio.PUD_UP,
            "down": self._gpio.PUD_DOWN,
        }

        if pull not in pull_map:
            raise BuddyHardwareError(
                "pull must be one of: None, 'up', 'down'."
            )

        self._gpio.setup(
            pin,
            self._gpio.IN,
            pull_up_down=pull_map[pull],
        )

    def configure_output(
        self,
        pin: int,
        initial: bool = False,
    ) -> None:
        self._validate_pin(pin)

        if not isinstance(initial, bool):
            raise BuddyHardwareError(
                "initial output value must be boolean."
            )

        initial_value = (
            self._gpio.HIGH
            if initial
            else self._gpio.LOW
        )

        self._gpio.setup(
            pin,
            self._gpio.OUT,
            initial=initial_value,
        )

        self._configured_outputs.add(pin)

    def write(
        self,
        pin: int,
        value: bool,
    ) -> None:
        self._validate_pin(pin)

        if pin not in self._configured_outputs:
            raise BuddyHardwareError(
                f"GPIO {pin} is not configured as an output."
            )

        if not isinstance(value, bool):
            raise BuddyHardwareError(
                "GPIO output value must be boolean."
            )

        self._gpio.output(
            pin,
            self._gpio.HIGH
            if value
            else self._gpio.LOW,
        )

    def read(
        self,
        pin: int,
    ) -> bool:
        self._validate_pin(pin)

        return bool(
            self._gpio.input(pin)
        )

    def setup_pwm(
        self,
        pin: int,
        frequency_hz: float,
        duty_cycle: float = 0.0,
    ) -> Any:
        self._validate_pin(pin)
        self._validate_frequency(frequency_hz)
        self._validate_duty_cycle(duty_cycle)

        if pin not in self._configured_outputs:
            self.configure_output(
                pin,
                initial=False,
            )

        pwm = self._gpio.PWM(
            pin,
            float(frequency_hz),
        )

        pwm.start(
            float(duty_cycle)
        )

        self._pwm_channels[pin] = pwm

        return pwm

    def update_pwm(
        self,
        pin: int,
        duty_cycle: float,
    ) -> None:
        self._validate_pin(pin)
        self._validate_duty_cycle(duty_cycle)

        pwm = self._pwm_channels.get(pin)

        if pwm is None:
            raise BuddyHardwareError(
                f"GPIO {pin} has no configured PWM channel."
            )

        pwm.ChangeDutyCycle(
            float(duty_cycle)
        )

    def cleanup(self) -> None:
        """Stop PWM and return outputs to a safe LOW state."""

        if self._cleaned_up:
            return

        for pwm in self._pwm_channels.values():
            try:
                pwm.ChangeDutyCycle(0.0)
                pwm.stop()
            except Exception:
                pass

        for pin in self._configured_outputs:
            try:
                self._gpio.output(
                    pin,
                    self._gpio.LOW,
                )
            except Exception:
                pass

        try:
            self._gpio.cleanup()
        finally:
            self._pwm_channels.clear()
            self._configured_outputs.clear()
            self._cleaned_up = True

    @staticmethod
    def _validate_pin(pin: int) -> None:
        if (
            isinstance(pin, bool)
            or not isinstance(pin, int)
            or pin < 0
        ):
            raise BuddyHardwareError(
                "GPIO pin must be a non-negative integer."
            )

    @staticmethod
    def _validate_frequency(
        frequency_hz: float,
    ) -> None:
        if (
            isinstance(frequency_hz, bool)
            or not isinstance(
                frequency_hz,
                (int, float),
            )
            or frequency_hz <= 0
        ):
            raise BuddyHardwareError(
                "PWM frequency must be a positive number."
            )

    @staticmethod
    def _validate_duty_cycle(
        duty_cycle: float,
    ) -> None:
        if (
            isinstance(duty_cycle, bool)
            or not isinstance(
                duty_cycle,
                (int, float),
            )
            or duty_cycle < 0
            or duty_cycle > 100
        ):
            raise BuddyHardwareError(
                "PWM duty cycle must be between 0 and 100."
            )

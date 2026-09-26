"""In-memory GPIO backend used for BUDDY simulation and testing."""

from __future__ import annotations

from dataclasses import dataclass

from buddy_core.hardware.base import (
    BuddyHardwareError,
    GPIOBackend,
)


@dataclass
class MockPWMState:
    """Recorded state for a simulated PWM channel."""

    frequency_hz: float
    duty_cycle: float


class MockGPIOBackend(GPIOBackend):
    """GPIO backend that never accesses physical hardware."""

    def __init__(self) -> None:
        self.pin_modes: dict[int, str] = {}
        self.pin_values: dict[int, bool] = {}
        self.pull_modes: dict[int, str | None] = {}
        self.pwm_channels: dict[int, MockPWMState] = {}
        self.cleaned_up = False

    def configure_input(
        self,
        pin: int,
        pull: str | None = None,
    ) -> None:
        self._validate_pin(pin)

        if pull not in {None, "up", "down"}:
            raise BuddyHardwareError(
                "pull must be one of: None, 'up', 'down'."
            )

        self.pin_modes[pin] = "input"
        self.pull_modes[pin] = pull
        self.pin_values.setdefault(pin, False)

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

        self.pin_modes[pin] = "output"
        self.pin_values[pin] = initial

    def write(
        self,
        pin: int,
        value: bool,
    ) -> None:
        self._validate_pin(pin)

        if self.pin_modes.get(pin) != "output":
            raise BuddyHardwareError(
                f"GPIO {pin} is not configured as an output."
            )

        if not isinstance(value, bool):
            raise BuddyHardwareError(
                "GPIO output value must be boolean."
            )

        self.pin_values[pin] = value

    def read(
        self,
        pin: int,
    ) -> bool:
        self._validate_pin(pin)

        if self.pin_modes.get(pin) != "input":
            raise BuddyHardwareError(
                f"GPIO {pin} is not configured as an input."
            )

        return self.pin_values.get(pin, False)

    def set_mock_input(
        self,
        pin: int,
        value: bool,
    ) -> None:
        """Set a simulated input value for tests."""

        self._validate_pin(pin)

        if self.pin_modes.get(pin) != "input":
            raise BuddyHardwareError(
                f"GPIO {pin} is not configured as an input."
            )

        if not isinstance(value, bool):
            raise BuddyHardwareError(
                "Mock input value must be boolean."
            )

        self.pin_values[pin] = value

    def setup_pwm(
        self,
        pin: int,
        frequency_hz: float,
        duty_cycle: float = 0.0,
    ) -> MockPWMState:
        self._validate_pin(pin)
        self._validate_frequency(frequency_hz)
        self._validate_duty_cycle(duty_cycle)

        self.pin_modes[pin] = "pwm"

        state = MockPWMState(
            frequency_hz=float(frequency_hz),
            duty_cycle=float(duty_cycle),
        )

        self.pwm_channels[pin] = state

        return state

    def update_pwm(
        self,
        pin: int,
        duty_cycle: float,
    ) -> None:
        self._validate_pin(pin)
        self._validate_duty_cycle(duty_cycle)

        if pin not in self.pwm_channels:
            raise BuddyHardwareError(
                f"GPIO {pin} has no configured PWM channel."
            )

        self.pwm_channels[pin].duty_cycle = float(duty_cycle)

    def cleanup(self) -> None:
        for pin in list(self.pin_values):
            self.pin_values[pin] = False

        for state in self.pwm_channels.values():
            state.duty_cycle = 0.0

        self.cleaned_up = True

    @staticmethod
    def _validate_pin(pin: int) -> None:
        if isinstance(pin, bool) or not isinstance(pin, int) or pin < 0:
            raise BuddyHardwareError(
                "GPIO pin must be a non-negative integer."
            )

    @staticmethod
    def _validate_frequency(frequency_hz: float) -> None:
        if (
            isinstance(frequency_hz, bool)
            or not isinstance(frequency_hz, (int, float))
            or frequency_hz <= 0
        ):
            raise BuddyHardwareError(
                "PWM frequency must be a positive number."
            )

    @staticmethod
    def _validate_duty_cycle(duty_cycle: float) -> None:
        if (
            isinstance(duty_cycle, bool)
            or not isinstance(duty_cycle, (int, float))
            or duty_cycle < 0
            or duty_cycle > 100
        ):
            raise BuddyHardwareError(
                "PWM duty cycle must be between 0 and 100."
            )

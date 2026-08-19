"""Primitive hardware contracts for BUDDY."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BuddyHardwareError(RuntimeError):
    """Raised when BUDDY hardware cannot be initialized or used safely."""


class GPIOBackend(ABC):
    """Hardware-independent GPIO API for BUDDY."""

    @abstractmethod
    def configure_input(
        self,
        pin: int,
        pull: str | None = None,
    ) -> None:
        """Configure a GPIO pin as an input."""

    @abstractmethod
    def configure_output(
        self,
        pin: int,
        initial: bool = False,
    ) -> None:
        """Configure a GPIO pin as an output."""

    @abstractmethod
    def write(
        self,
        pin: int,
        value: bool,
    ) -> None:
        """Write a digital value to an output pin."""

    @abstractmethod
    def read(
        self,
        pin: int,
    ) -> bool:
        """Read a digital value from an input pin."""

    @abstractmethod
    def setup_pwm(
        self,
        pin: int,
        frequency_hz: float,
        duty_cycle: float = 0.0,
    ) -> Any:
        """Create/configure PWM on a pin."""

    @abstractmethod
    def update_pwm(
        self,
        pin: int,
        duty_cycle: float,
    ) -> None:
        """Update PWM duty cycle."""

    @abstractmethod
    def cleanup(self) -> None:
        """Return backend resources to a safe state."""

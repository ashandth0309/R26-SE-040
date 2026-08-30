"""Hardware-independent neck backend contracts for BUDDY."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BuddyNeckError(RuntimeError):
    """Raised when BUDDY neck control cannot operate safely."""


class NeckBackend(ABC):
    """Abstract 2-axis neck backend."""

    @abstractmethod
    def set_pan(self, angle_deg: float) -> None:
        """Set PAN angle in degrees."""

    @abstractmethod
    def set_tilt(self, angle_deg: float) -> None:
        """Set TILT angle in degrees."""

    @abstractmethod
    def release(self) -> None:
        """Release servo outputs without commanding a new position."""

    @abstractmethod
    def cleanup(self) -> None:
        """Return backend resources to a safe state."""

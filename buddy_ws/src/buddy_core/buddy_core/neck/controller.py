"""Safe high-level 2-axis neck controller for BUDDY."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from buddy_core.neck.base import BuddyNeckError, NeckBackend


class NeckController:
    """Safely control BUDDY PAN and TILT within calibrated limits."""

    def __init__(
        self,
        backend: NeckBackend,
        config: Mapping[str, Any],
    ) -> None:
        hardware = config.get("hardware")

        if not isinstance(hardware, Mapping):
            raise BuddyNeckError(
                "Missing or invalid hardware configuration."
            )

        neck = hardware.get("neck")

        if not isinstance(neck, Mapping):
            raise BuddyNeckError(
                "Missing or invalid hardware.neck configuration."
            )

        pan = neck.get("pan")
        tilt = neck.get("tilt")

        if not isinstance(pan, Mapping):
            raise BuddyNeckError(
                "Missing or invalid PAN configuration."
            )

        if not isinstance(tilt, Mapping):
            raise BuddyNeckError(
                "Missing or invalid TILT configuration."
            )

        self._backend = backend

        self.pan_min = float(pan["min_angle_deg"])
        self.pan_center = float(pan["center_angle_deg"])
        self.pan_max = float(pan["max_angle_deg"])

        self.tilt_min = float(tilt["min_angle_deg"])
        self.tilt_center = float(tilt["center_angle_deg"])
        self.tilt_max = float(tilt["max_angle_deg"])

        self.current_pan: float | None = None
        self.current_tilt: float | None = None

    @staticmethod
    def _validate_angle(angle_deg: float) -> float:
        if (
            isinstance(angle_deg, bool)
            or not isinstance(angle_deg, (int, float))
        ):
            raise BuddyNeckError(
                "Neck angle must be a number."
            )

        return float(angle_deg)

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        return max(minimum, min(maximum, value))

    def set_pan(self, angle_deg: float) -> float:
        """Set PAN, clamped to the calibrated safe range."""

        requested = self._validate_angle(angle_deg)

        safe_angle = self._clamp(
            requested,
            self.pan_min,
            self.pan_max,
        )

        self._backend.set_pan(safe_angle)
        self.current_pan = safe_angle

        return safe_angle

    def set_tilt(self, angle_deg: float) -> float:
        """Set TILT, clamped to the calibrated safe range."""

        requested = self._validate_angle(angle_deg)

        safe_angle = self._clamp(
            requested,
            self.tilt_min,
            self.tilt_max,
        )

        self._backend.set_tilt(safe_angle)
        self.current_tilt = safe_angle

        return safe_angle

    def look(
        self,
        pan_deg: float,
        tilt_deg: float,
    ) -> tuple[float, float]:
        """Set PAN and TILT to safe target positions."""

        safe_pan = self.set_pan(pan_deg)
        safe_tilt = self.set_tilt(tilt_deg)

        return safe_pan, safe_tilt

    def center(self) -> tuple[float, float]:
        """Move neck to the calibrated front/level position."""

        return self.look(
            self.pan_center,
            self.tilt_center,
        )

    def release(self) -> None:
        """Release servo outputs."""

        self._backend.release()

    def cleanup(self) -> None:
        """Safely clean up the neck backend."""

        self._backend.cleanup()

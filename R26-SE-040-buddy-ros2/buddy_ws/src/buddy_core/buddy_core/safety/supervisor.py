"""Deterministic obstacle-aware movement safety for BUDDY."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyDecision:
    """Result of validating one requested movement."""

    allowed: bool
    reason: str


class SafetySupervisor:
    """Validate movement requests against BUDDY's safety state."""

    COMMAND_STOP = 0
    COMMAND_FORWARD = 1
    COMMAND_BACKWARD = 2
    COMMAND_TURN_LEFT = 3
    COMMAND_TURN_RIGHT = 4
    COMMAND_ROTATE_LEFT = 5
    COMMAND_ROTATE_RIGHT = 6

    VALID_COMMANDS = {
        COMMAND_STOP,
        COMMAND_FORWARD,
        COMMAND_BACKWARD,
        COMMAND_TURN_LEFT,
        COMMAND_TURN_RIGHT,
        COMMAND_ROTATE_LEFT,
        COMMAND_ROTATE_RIGHT,
    }

    def __init__(self) -> None:
        self._emergency_stop_active = False

    @property
    def emergency_stop_active(self) -> bool:
        """Return whether emergency stop is currently latched."""
        return self._emergency_stop_active

    def activate_emergency_stop(self) -> None:
        """Latch emergency stop."""
        self._emergency_stop_active = True

    def reset_emergency_stop(self) -> None:
        """Clear the emergency-stop latch."""
        self._emergency_stop_active = False

    def evaluate(
        self,
        command: int,
        *,
        front_blocked: bool = False,
        rear_blocked: bool = False,
        left_blocked: bool = False,
        right_blocked: bool = False,
        obstacle_data_available: bool = True,
    ) -> SafetyDecision:
        """Return whether the requested movement is currently safe."""

        if command not in self.VALID_COMMANDS:
            return SafetyDecision(
                False,
                "Unknown movement command.",
            )

        # STOP must always remain available.
        if command == self.COMMAND_STOP:
            return SafetyDecision(
                True,
                "Stop command accepted.",
            )

        if self._emergency_stop_active:
            return SafetyDecision(
                False,
                "Emergency stop is active.",
            )

        # Fail safe when no obstacle status has been received.
        if not obstacle_data_available:
            return SafetyDecision(
                False,
                "Obstacle data is unavailable.",
            )

        if command == self.COMMAND_FORWARD and front_blocked:
            return SafetyDecision(
                False,
                "Front obstacle blocks forward movement.",
            )

        if command == self.COMMAND_BACKWARD and rear_blocked:
            return SafetyDecision(
                False,
                "Rear obstacle blocks backward movement.",
            )

        if command in (
            self.COMMAND_TURN_LEFT,
            self.COMMAND_ROTATE_LEFT,
        ) and left_blocked:
            return SafetyDecision(
                False,
                "Left obstacle blocks leftward movement.",
            )

        if command in (
            self.COMMAND_TURN_RIGHT,
            self.COMMAND_ROTATE_RIGHT,
        ) and right_blocked:
            return SafetyDecision(
                False,
                "Right obstacle blocks rightward movement.",
            )

        return SafetyDecision(
            True,
            "Movement request accepted.",
        )

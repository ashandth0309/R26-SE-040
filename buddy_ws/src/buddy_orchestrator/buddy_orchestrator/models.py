"""Plain-data views of the messages the coordinator reads and writes.

The ROS node converts messages to and from these, so the coordinator logic
is testable without ROS and every field it relies on is named in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from buddy_orchestrator.contracts import Err

# Navigation command kinds (NavigationCommand.COMMAND_*)
NAV_KINDS = {
    0: "stop", 1: "move_forward", 2: "turn_left", 3: "turn_right",
    4: "start_patrol", 5: "go_to_location", 6: "cancel", 7: "pause",
    8: "resume",
}
NAV_CODES = {name: code for code, name in NAV_KINDS.items()}

# NavigationStatus.STATE_*
NAV_STATES = {0: "idle", 1: "running", 2: "paused", 3: "succeeded",
              4: "failed", 5: "cancelled"}

# MovementCommand.COMMAND_*
MOVE_STOP = 0
MOVE_FORWARD = 1
MOVE_BACKWARD = 2
MOVE_TURN_LEFT = 3
MOVE_TURN_RIGHT = 4
MOVE_ROTATE_LEFT = 5
MOVE_ROTATE_RIGHT = 6


@dataclass(frozen=True)
class NavRequest:
    """A NavigationCommand, as plain data."""

    kind: str
    request_id: str = ""
    source: str = ""
    duration_s: float = 0.0
    route_id: str = ""
    repeat: bool = False
    location_name: str = ""

    @property
    def is_stop(self) -> bool:
        return self.kind == "stop"


#: Requests that are always allowed through: they only reduce motion.
SAFE_NAV_KINDS = frozenset({"stop", "cancel", "pause", "resume"})
#: Requests that start something that moves the robot.
MOTION_NAV_KINDS = frozenset({"move_forward", "turn_left", "turn_right",
                              "start_patrol", "go_to_location"})


@dataclass(frozen=True)
class MoveIn:
    """A MovementCommand relayed from the voice node."""

    command: int
    source: str = ""
    sequence_id: int = 0
    linear_speed: float = 0.0
    angular_speed: float = 0.0
    valid_for_s: float = 0.3
    emergency: bool = False


@dataclass(frozen=True)
class NavStatusIn:
    """A NavigationStatus."""

    state: str
    request_id: str = ""
    active_command: str = ""
    reason: str = ""
    motion_enabled: bool = False
    pose_trusted: bool = False
    sensors_healthy: bool = False
    emergency_stop_active: bool = False
    progress: float = 0.0


@dataclass(frozen=True)
class SafetyIn:
    """A SafetyState."""

    emergency_stop_active: bool = False
    movement_allowed: bool = False
    degraded: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ObstacleIn:
    """The parts of ObstacleStatus the coordinator uses."""

    front_valid: bool = False
    rear_valid: bool = False
    left_valid: bool = False
    right_valid: bool = False
    degraded: bool = True


@dataclass(frozen=True)
class VisionEventIn:
    """A VisionEvent."""

    kind: str
    person_key: str = ""
    severity: str = "info"
    suggested_text: str = ""
    needs_voice_check: bool = False
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PersonIn:
    """One VisionPerson."""

    person_key: str
    known: bool = False
    display_name: str = ""
    identity_state: str = ""
    posture: str = ""
    motion: str = ""
    fall_state: str = ""
    bbox_x: float = 0.0
    bbox_width: float = 0.0


@dataclass(frozen=True)
class RouteAnswer:
    """What the coordinator tells the voice node about an utterance."""

    claimed: bool
    reply: str = ""
    intent: str = ""
    error: Err = Err.OK

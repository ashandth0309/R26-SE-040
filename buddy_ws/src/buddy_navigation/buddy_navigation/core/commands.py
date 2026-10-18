"""ROS-free description of movement requests.

The numeric codes mirror ``buddy_interfaces/msg/MovementCommand`` so the ROS
node can translate one-to-one.  Keeping them here lets the navigation logic
and simulator run (and be tested) without ROS.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class MoveCode(IntEnum):
    """Same values as ``MovementCommand.COMMAND_*``."""

    STOP = 0
    FORWARD = 1
    BACKWARD = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4
    ROTATE_LEFT = 5
    ROTATE_RIGHT = 6


@dataclass(frozen=True)
class MoveRequest:
    """One movement request.

    ``speed`` is the existing normalised motor speed (0..1).  A request is
    only valid for ``valid_for_s`` seconds; the issuer must keep re-sending
    it, which is what makes a crashed or stalled issuer stop the robot.
    """

    code: MoveCode
    speed: float = 0.0
    valid_for_s: float = 0.3
    source: str = "buddy_navigation"
    emergency: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.speed <= 1.0:
            raise ValueError("speed must be within 0..1.")

        if self.valid_for_s <= 0.0:
            raise ValueError("valid_for_s must be positive.")


STOP_REQUEST = MoveRequest(MoveCode.STOP, 0.0, 0.3)

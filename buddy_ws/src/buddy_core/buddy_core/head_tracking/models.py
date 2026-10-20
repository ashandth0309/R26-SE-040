"""Plain-data types shared by the BUDDY head-tracking modules.

Nothing here imports ROS, OpenCV, numpy or any other subsystem.  The
vision and voice subsystems live on other branches, so head tracking
consumes these small dataclasses through adapters instead of importing
their code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


class Priority(IntEnum):
    """Command priority.  A LOWER number always wins."""

    EMERGENCY_STOP = 1      # emergency stop / hardware fault (latched flags)
    SAFETY_LIMITS = 2       # servo safety limits (always-on clamps)
    USER_STOP = 3           # explicit user stop (latched flag)
    HAND = 4                # active hand-following command
    SPEAKER_FACE = 5        # active speaker (caller) face tracking
    PASSIVE_FACE = 6        # passive face tracking
    IDLE = 7                # idle behaviour / neutral


#: Priorities that may submit motion targets.  1-3 are latched flags or
#: clamps inside the coordinator, never target sources.
TARGET_SOURCES = (
    Priority.HAND,
    Priority.SPEAKER_FACE,
    Priority.PASSIVE_FACE,
    Priority.IDLE,
)


class HeadState(str, Enum):
    """Behaviour states.  The first five are the specified cycle."""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    SOUND_ORIENTING = "SOUND_ORIENTING"
    FACE_TRACKING = "FACE_TRACKING"
    HAND_TRACKING = "HAND_TRACKING"
    STOPPED = "STOPPED"                  # explicit user stop
    EMERGENCY_STOP = "EMERGENCY_STOP"    # safety latch
    FAULT = "FAULT"                      # servo/hardware fault latch


class EventKind(str, Enum):
    """Events published by the head-tracking system."""

    WAKE_WORD_DETECTED = "WAKE_WORD_DETECTED"
    SOUND_DIRECTION_ESTIMATED = "SOUND_DIRECTION_ESTIMATED"
    SOUND_DIRECTION_UNCERTAIN = "SOUND_DIRECTION_UNCERTAIN"
    HEAD_TURN_REQUESTED = "HEAD_TURN_REQUESTED"
    STATE_CHANGED = "STATE_CHANGED"
    TARGET_ACQUIRED = "TARGET_ACQUIRED"
    TARGET_LOST = "TARGET_LOST"
    HAND_ACTIVATED = "HAND_ACTIVATED"
    HAND_DEACTIVATED = "HAND_DEACTIVATED"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    EMERGENCY_STOP_CLEARED = "EMERGENCY_STOP_CLEARED"
    USER_STOP = "USER_STOP"
    USER_RESUME = "USER_RESUME"
    FAULT = "FAULT"
    COMMAND_REJECTED = "COMMAND_REJECTED"


@dataclass(frozen=True)
class HeadEvent:
    kind: EventKind
    t: float
    data: dict[str, Any] = field(default_factory=dict)


def is_number(value: Any) -> bool:
    """True for a finite real number that is not a bool."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


@dataclass(frozen=True)
class FaceObservation:
    """One face box in frame pixels (top-left x, y plus width, height).

    ``track_id`` is the vision tracker's opaque id (Sobiya's
    ``person_key``).  No name or identity is read or stored here.
    """

    track_id: str
    x: float
    y: float
    w: float
    h: float
    confidence: float = 1.0
    known: bool = False


@dataclass(frozen=True)
class FaceFrame:
    """All faces seen in one camera frame, stamped with capture time."""

    t: float
    width: int
    height: int
    faces: tuple[FaceObservation, ...] = ()


@dataclass(frozen=True)
class HandPoint:
    x: float
    y: float
    confidence: float = 1.0


@dataclass(frozen=True)
class GestureEvent:
    """A recognised gesture (Sobiya: wave, raised_hand, both_hands_up,
    pointing).  ``side`` is the person's own left/right."""

    name: str
    track_id: str = ""
    side: str | None = None


@dataclass(frozen=True)
class HandObservation:
    """Wrist and (when a model exists) index-fingertip points per side.

    ``fingertips`` stays empty until a real hand-landmark model is
    integrated; today only ``wrists`` can be filled from body pose.
    """

    t: float
    width: int
    height: int
    track_id: str = ""
    wrists: dict[str, HandPoint] = field(default_factory=dict)
    fingertips: dict[str, HandPoint] = field(default_factory=dict)
    gestures: tuple[GestureEvent, ...] = ()


@dataclass(frozen=True)
class WakeWordEvent:
    t: float
    confidence: float = 1.0
    speaker_track_id: str | None = None   # only if voice/vision can supply it

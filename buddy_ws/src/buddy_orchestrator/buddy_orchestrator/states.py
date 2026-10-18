"""The coordinator's two state machines.

Buddy can listen while it moves, so one flat state list cannot describe it.
The integration spec's fourteen states are split into two regions that run
at the same time:

* ``RobotState``  - what the body is doing (motion axis).
* ``ConversationState`` - what the voice interaction is doing.

``EMERGENCY_STOP`` is a *robot* state and wins over everything: it can be
entered from any state except ``SHUTTING_DOWN`` and can only be left to
``IDLE`` or ``DEGRADED_MODE`` after the supervisor clears the latch, never
straight back into motion.

Camera, microphone and internet losses do not change ``RobotState``; they
are capability flags (see ``readiness.Capabilities``) that switch off the
features that need them.  ``DEGRADED_MODE`` means the *motion-safety* inputs
(obstacle sensing, supervisor, navigation) cannot be trusted, so no new
motion is accepted until they recover.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from enum import Enum


class RobotState(str, Enum):
    STARTING = "STARTING"
    INITIALIZING = "INITIALIZING"
    IDLE = "IDLE"
    NAVIGATION_ACTIVE = "NAVIGATION_ACTIVE"
    PATROL_ACTIVE = "PATROL_ACTIVE"
    PAUSED = "PAUSED"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    DEGRADED_MODE = "DEGRADED_MODE"
    SHUTTING_DOWN = "SHUTTING_DOWN"


class ConversationState(str, Enum):
    LISTENING = "LISTENING"
    USER_SPEAKING = "USER_SPEAKING"
    PROCESSING = "PROCESSING"
    VISION_REQUEST = "VISION_REQUEST"
    BUDDY_SPEAKING = "BUDDY_SPEAKING"


R = RobotState
C = ConversationState

_EVERYWHERE = (R.EMERGENCY_STOP, R.SHUTTING_DOWN)

ROBOT_TRANSITIONS: dict[RobotState, frozenset[RobotState]] = {
    R.STARTING: frozenset({R.INITIALIZING, *_EVERYWHERE}),
    R.INITIALIZING: frozenset({R.IDLE, R.DEGRADED_MODE, *_EVERYWHERE}),
    R.IDLE: frozenset({R.NAVIGATION_ACTIVE, R.PATROL_ACTIVE,
                       R.DEGRADED_MODE, *_EVERYWHERE}),
    R.NAVIGATION_ACTIVE: frozenset({R.IDLE, R.PAUSED, R.DEGRADED_MODE,
                                    *_EVERYWHERE}),
    R.PATROL_ACTIVE: frozenset({R.IDLE, R.PAUSED, R.DEGRADED_MODE,
                                *_EVERYWHERE}),
    R.PAUSED: frozenset({R.NAVIGATION_ACTIVE, R.PATROL_ACTIVE, R.IDLE,
                         R.DEGRADED_MODE, *_EVERYWHERE}),
    R.DEGRADED_MODE: frozenset({R.IDLE, *_EVERYWHERE}),
    R.EMERGENCY_STOP: frozenset({R.IDLE, R.DEGRADED_MODE,
                                 R.SHUTTING_DOWN}),
    R.SHUTTING_DOWN: frozenset(),
}

CONVERSATION_TRANSITIONS: dict[
        ConversationState, frozenset[ConversationState]] = {
    C.LISTENING: frozenset({C.USER_SPEAKING, C.PROCESSING,
                            C.VISION_REQUEST, C.BUDDY_SPEAKING}),
    C.USER_SPEAKING: frozenset({C.LISTENING, C.PROCESSING}),
    C.PROCESSING: frozenset({C.LISTENING, C.USER_SPEAKING,
                             C.VISION_REQUEST, C.BUDDY_SPEAKING}),
    C.VISION_REQUEST: frozenset({C.LISTENING, C.USER_SPEAKING,
                                 C.BUDDY_SPEAKING}),
    C.BUDDY_SPEAKING: frozenset({C.LISTENING, C.USER_SPEAKING,
                                 C.PROCESSING}),
}

# Robot states in which Buddy may start new motion.
MOTION_READY = frozenset({R.IDLE, R.PAUSED})
# Robot states that mean the body is (or may be) moving.
MOVING = frozenset({R.NAVIGATION_ACTIVE, R.PATROL_ACTIVE})


class StateMachine:
    """A table-driven state machine that records what it refused."""

    def __init__(self, initial, table, *,
                 clock: Callable[[], float] = time.monotonic,
                 history: int = 100) -> None:
        self.state = initial
        self._table = table
        self._clock = clock
        self.history: deque = deque(maxlen=history)
        self.violations: deque = deque(maxlen=history)

    def can(self, target) -> bool:
        """True if ``target`` is reachable from the current state."""

        return target == self.state or target in self._table[self.state]

    def go(self, target, reason: str = "") -> bool:
        """Move to ``target``.  Returns False (and records it) if illegal.

        A move to the current state is a no-op that succeeds.
        """

        if target == self.state:
            return True

        if target not in self._table[self.state]:
            self.violations.append(
                (self._clock(), self.state, target, reason))

            return False

        self.history.append((self._clock(), self.state, target, reason))
        self.state = target

        return True


def robot_machine(clock=time.monotonic) -> StateMachine:
    return StateMachine(R.STARTING, ROBOT_TRANSITIONS, clock=clock)


def conversation_machine(clock=time.monotonic) -> StateMachine:
    return StateMachine(C.LISTENING, CONVERSATION_TRANSITIONS, clock=clock)

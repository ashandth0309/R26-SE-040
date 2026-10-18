"""Route validated voice commands to the robot, never around its safety.

The router is the only voice-side publisher.  It sends:

* ``/movement/request`` (default): the existing path, which still goes
  through ``safety_supervisor`` before the motors.
* ``/navigation/command``: only when navigation is enabled and the
  navigation package from the ``Yohan_Mapping`` branch is present.

The router never talks to the motors, never publishes to
``/movement/safe`` and never clears an emergency stop.  It owns exactly
one active motion at a time, so there are no duplicate or conflicting
publishers from the voice side.

Command and message codes are mirrored here as constants.  The movement
codes are checked against ``MovementCommand.msg`` by a test.  The
navigation codes mirror ``NavigationCommand.msg`` on ``Yohan_Mapping``
and are UNVERIFIED against it until the branches are combined.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from buddy_core.voice.voice_commands import (
    MOTION_COMMANDS,
    NAVIGATION_COMMANDS,
    ParsedCommand,
    VoiceCommand,
)

# MovementCommand.msg
MOVE_STOP = 0
MOVE_FORWARD = 1
MOVE_BACKWARD = 2
MOVE_TURN_LEFT = 3
MOVE_TURN_RIGHT = 4
MOVE_ROTATE_LEFT = 5
MOVE_ROTATE_RIGHT = 6

# NavigationCommand.msg (Yohan_Mapping)
NAV_STOP = 0
NAV_MOVE_FORWARD = 1
NAV_TURN_LEFT = 2
NAV_TURN_RIGHT = 3
NAV_START_PATROL = 4
NAV_GO_TO_LOCATION = 5
NAV_CANCEL = 6
NAV_PAUSE = 7
NAV_RESUME = 8

SOURCE = "buddy_voice"


class MovementSink(Protocol):
    """Publishes one MovementCommand."""

    def send_movement(
        self,
        command: int,
        linear: float,
        angular: float,
        valid_for_s: float,
        emergency: bool,
        sequence_id: int,
        source: str,
    ) -> None:
        """Publish to /movement/request."""


class NavigationSink(Protocol):
    """Publishes one NavigationCommand."""

    def send_navigation(
        self,
        command: int,
        request_id: str,
        source: str,
        duration_s: float,
        location_name: str,
    ) -> None:
        """Publish to /navigation/command."""


class RouteStatus(str, Enum):
    """Outcome of a routing attempt."""

    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    LOW_CONFIDENCE = "low_confidence"
    SLEEPING = "sleeping"
    MOTION_DISABLED = "motion_disabled"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"
    FAILED = "failed"


@dataclass(frozen=True)
class RouteResult:
    """What happened to a command, and what to say about it."""

    status: RouteStatus
    command_id: str | None = None
    ack: str = ""
    reason: str = ""

    @property
    def accepted(self) -> bool:
        """True if something was sent."""

        return self.status is RouteStatus.ACCEPTED


class CommandState(str, Enum):
    """Tracked lifecycle of a command id."""

    SENT = "sent"
    ACTIVE = "active"
    DONE = "done"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True)
class RouterConfig:
    """Router limits."""

    use_navigation: bool = False
    move_speed: float = 0.20
    turn_speed: float = 0.60
    valid_for_s: float = 0.30
    max_motion_s: float = 8.0
    min_motion_confidence: float = 0.60
    nav_step_s: float = 2.0
    republish_s: float = 0.10
    motion_enabled: bool = True


@dataclass
class _Active:
    command: VoiceCommand
    started: float
    command_id: str


@dataclass
class TrackedCommand:
    """Record for status feedback."""

    command_id: str
    kind: VoiceCommand
    state: CommandState = CommandState.SENT
    detail: str = ""


_ACKS = {
    VoiceCommand.STOP: "Stopping.",
    VoiceCommand.FORWARD: "Going forward.",
    VoiceCommand.BACKWARD: "Coming back.",
    VoiceCommand.TURN_LEFT: "Turning left.",
    VoiceCommand.TURN_RIGHT: "Turning right.",
    VoiceCommand.PATROL: "Starting patrol.",
    VoiceCommand.GO_TO: "On my way.",
    VoiceCommand.CANCEL: "Cancelled.",
    VoiceCommand.PAUSE: "Pausing.",
    VoiceCommand.RESUME: "Resuming.",
}

_MOVEMENT_MAP = {
    VoiceCommand.FORWARD: (MOVE_FORWARD, "linear"),
    VoiceCommand.BACKWARD: (MOVE_BACKWARD, "linear"),
    VoiceCommand.TURN_LEFT: (MOVE_ROTATE_LEFT, "angular"),
    VoiceCommand.TURN_RIGHT: (MOVE_ROTATE_RIGHT, "angular"),
}

_NAV_MAP = {
    VoiceCommand.FORWARD: NAV_MOVE_FORWARD,
    VoiceCommand.TURN_LEFT: NAV_TURN_LEFT,
    VoiceCommand.TURN_RIGHT: NAV_TURN_RIGHT,
    VoiceCommand.PATROL: NAV_START_PATROL,
    VoiceCommand.GO_TO: NAV_GO_TO_LOCATION,
    VoiceCommand.CANCEL: NAV_CANCEL,
    VoiceCommand.PAUSE: NAV_PAUSE,
    VoiceCommand.RESUME: NAV_RESUME,
}


class CommandRouter:
    """Validate, deduplicate and publish voice commands."""

    def __init__(
        self,
        movement: MovementSink,
        navigation: NavigationSink | None = None,
        *,
        config: RouterConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or RouterConfig()
        self._movement = movement
        self._navigation = navigation
        self._clock = clock
        self._seq = itertools.count(1)
        self._active: _Active | None = None
        self._nav_goal_active = False
        self.tracked: dict[str, TrackedCommand] = {}
        self.last_stop_time: float | None = None
        self.publish_errors = 0

    # ----- state -----

    @property
    def navigation_available(self) -> bool:
        """True if the navigation sink exists and is enabled."""

        return self.config.use_navigation and self._navigation is not None

    @property
    def moving(self) -> bool:
        """True while the router is publishing a motion command."""

        return self._active is not None

    @property
    def active_command(self) -> VoiceCommand | None:
        """The motion currently being published."""

        return self._active.command if self._active else None

    def set_motion_enabled(self, enabled: bool) -> None:
        """Turn voice motion on or off (stop is always possible)."""

        self.config = RouterConfig(
            **{**self.config.__dict__, "motion_enabled": enabled}
        )

        if not enabled:
            self.stop("motion disabled")

    # ----- helpers -----

    def _new_id(self, kind: VoiceCommand) -> str:
        command_id = f"voice-{next(self._seq)}"
        self.tracked[command_id] = TrackedCommand(command_id, kind)

        if len(self.tracked) > 50:
            oldest = next(iter(self.tracked))
            del self.tracked[oldest]

        return command_id

    def _send_movement(
        self,
        code: int,
        linear: float = 0.0,
        angular: float = 0.0,
        emergency: bool = False,
    ) -> bool:
        try:
            self._movement.send_movement(
                code, linear, angular, self.config.valid_for_s, emergency,
                next(self._seq), SOURCE,
            )
        except Exception:
            self.publish_errors += 1

            return False

        return True

    def _send_nav(self, code: int, command_id: str, duration: float = 0.0,
                  location: str = "") -> bool:
        if self._navigation is None:
            return False

        try:
            self._navigation.send_navigation(
                code, command_id, SOURCE, duration, location,
            )
        except Exception:
            self.publish_errors += 1

            return False

        return True

    # ----- stop -----

    def stop(self, reason: str = "") -> RouteResult:
        """Stop everything.  Highest priority, no preconditions."""

        now = self._clock()
        command_id = self._new_id(VoiceCommand.STOP)
        self._active = None
        self.last_stop_time = now
        sent = self._send_movement(MOVE_STOP)

        if self.navigation_available:
            sent = self._send_nav(NAV_STOP, command_id) or sent

        self._nav_goal_active = False

        for tracked in self.tracked.values():
            if tracked.state in (CommandState.SENT, CommandState.ACTIVE) \
                    and tracked.command_id != command_id:
                tracked.state = CommandState.DONE
                tracked.detail = "stopped"

        self.tracked[command_id].state = (
            CommandState.DONE if sent else CommandState.REJECTED
        )

        return RouteResult(
            RouteStatus.ACCEPTED if sent else RouteStatus.FAILED,
            command_id, _ACKS[VoiceCommand.STOP], reason,
        )

    # ----- general entry point -----

    def handle(
        self,
        parsed: ParsedCommand,
        *,
        awake: bool,
        confidence: float | None = None,
    ) -> RouteResult:
        """Route a parsed command."""

        kind = parsed.kind

        if kind is VoiceCommand.STOP:
            return self.stop("spoken")

        if not awake:
            return RouteResult(RouteStatus.SLEEPING, reason="asleep")

        if (
            confidence is not None
            and confidence < self.config.min_motion_confidence
        ):
            return RouteResult(
                RouteStatus.LOW_CONFIDENCE,
                reason=f"confidence {confidence:.2f}",
            )

        if kind in MOTION_COMMANDS:
            return self._motion(kind)

        if kind in NAVIGATION_COMMANDS:
            return self._navigation_command(parsed)

        return RouteResult(RouteStatus.NOT_APPLICABLE)

    # ----- motion -----

    def _motion(self, kind: VoiceCommand) -> RouteResult:
        if not self.config.motion_enabled:
            return RouteResult(
                RouteStatus.MOTION_DISABLED,
                ack="Movement is turned off.",
                reason="motion disabled",
            )

        now = self._clock()

        if self._active and self._active.command is kind:
            self._active.started = now

            return RouteResult(
                RouteStatus.DUPLICATE, self._active.command_id, "",
                "already active",
            )

        command_id = self._new_id(kind)
        self._active = None

        if self.navigation_available and kind in _NAV_MAP:
            ok = self._send_nav(
                _NAV_MAP[kind], command_id, self.config.nav_step_s,
            )
            self._nav_goal_active = ok
        else:
            self._active = _Active(kind, now, command_id)
            ok = self._publish_active()

        if not ok:
            self._active = None
            self.tracked[command_id].state = CommandState.REJECTED

            return RouteResult(
                RouteStatus.FAILED, command_id,
                "Sorry, I could not do that.", "publish failed",
            )

        self.tracked[command_id].state = CommandState.ACTIVE

        return RouteResult(
            RouteStatus.ACCEPTED, command_id, _ACKS[kind],
        )

    def _publish_active(self) -> bool:
        active = self._active

        if active is None:
            return False

        code, axis = _MOVEMENT_MAP[active.command]
        cfg = self.config

        if axis == "linear":
            return self._send_movement(code, linear=cfg.move_speed)

        return self._send_movement(code, angular=cfg.turn_speed)

    # ----- navigation -----

    def _navigation_command(self, parsed: ParsedCommand) -> RouteResult:
        kind = parsed.kind

        if not self.navigation_available:
            return RouteResult(
                RouteStatus.UNAVAILABLE,
                ack="Navigation is not available yet.",
                reason="navigation disabled",
            )

        if not self.config.motion_enabled and kind in (
            VoiceCommand.PATROL, VoiceCommand.GO_TO, VoiceCommand.RESUME,
        ):
            return RouteResult(
                RouteStatus.MOTION_DISABLED,
                ack="Movement is turned off.",
            )

        if kind in (
            VoiceCommand.CANCEL, VoiceCommand.PAUSE, VoiceCommand.RESUME,
        ) and not self._nav_goal_active:
            return RouteResult(RouteStatus.NOT_APPLICABLE)

        if self._active is not None:
            self._active = None
            self._send_movement(MOVE_STOP)

        command_id = self._new_id(kind)
        ok = self._send_nav(
            _NAV_MAP[kind], command_id, 0.0, parsed.location or "",
        )

        if not ok:
            self.tracked[command_id].state = CommandState.REJECTED

            return RouteResult(
                RouteStatus.FAILED, command_id,
                "Sorry, I could not do that.", "publish failed",
            )

        if kind in (VoiceCommand.PATROL, VoiceCommand.GO_TO,
                    VoiceCommand.RESUME):
            self._nav_goal_active = True
        elif kind is VoiceCommand.CANCEL:
            self._nav_goal_active = False

        self.tracked[command_id].state = CommandState.ACTIVE

        return RouteResult(
            RouteStatus.ACCEPTED, command_id, _ACKS[kind],
        )

    def on_navigation_status(self, request_id: str, state: str,
                             detail: str = "") -> None:
        """Feed NavigationStatus back in so spoken feedback is honest."""

        tracked = self.tracked.get(request_id)

        if tracked is None:
            return

        mapping = {
            "active": CommandState.ACTIVE,
            "done": CommandState.DONE,
            "rejected": CommandState.REJECTED,
            "failed": CommandState.REJECTED,
        }
        tracked.state = mapping.get(state, tracked.state)
        tracked.detail = detail

        if state in ("done", "rejected", "failed"):
            self._nav_goal_active = False

    # ----- periodic -----

    def tick(self) -> None:
        """Republish the active motion; expire it after ``max_motion_s``.

        The motor side expires commands quickly, so a live motion must be
        refreshed.  If the voice node stops ticking, the robot stops by
        itself.
        """

        active = self._active

        if active is None:
            return

        now = self._clock()

        if now - active.started >= self.config.max_motion_s:
            self.tracked[active.command_id].state = CommandState.EXPIRED
            self.stop("motion time limit")

            return

        if not self._publish_active():
            self.stop("publish failure")


@dataclass
class RecordingSink:
    """Test double that records everything it is given."""

    movement: list = field(default_factory=list)
    navigation: list = field(default_factory=list)
    fail: bool = False

    def send_movement(self, command, linear, angular, valid_for_s,
                      emergency, sequence_id, source):
        if self.fail:
            raise RuntimeError("publish failed")

        self.movement.append((command, linear, angular, valid_for_s,
                              emergency, sequence_id, source))

    def send_navigation(self, command, request_id, source, duration_s,
                        location_name):
        if self.fail:
            raise RuntimeError("publish failed")

        self.navigation.append((command, request_id, source, duration_s,
                                location_name))

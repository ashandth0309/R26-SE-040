"""Navigation coordinator: turns goals into guarded movement requests.

The coordinator is pure Python and tick driven.  It never talks to motors.
Each tick it returns exactly one ``MoveRequest`` for the existing
``/movement/request`` interface, which the safety supervisor and motor
executor still validate afterwards.  This layer adds its own, stricter
checks and never replaces them.

Safety rules enforced here, in priority order:

1. physical motion is off unless ``allow_motion`` is set explicitly;
2. an emergency stop or a manual override produces STOP and fails the goal;
3. missing, stale or faulty sensors, or insufficient coverage, produce STOP;
4. path following needs a trusted pose, otherwise the goal is refused;
5. speeds are clamped to configured limits, and SLOW scales them down.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from buddy_navigation.core.commands import (
    STOP_REQUEST,
    MoveCode,
    MoveRequest,
)
from buddy_navigation.core.geometry import Pose2D, angle_difference
from buddy_navigation.core.obstacle_assessment import (
    Motion,
    ObstacleAssessor,
    Recommendation,
)
from buddy_navigation.core.sensor_model import SensorState

MOTION_FOR_CODE = {
    MoveCode.FORWARD: Motion.FORWARD,
    MoveCode.BACKWARD: Motion.BACKWARD,
    MoveCode.ROTATE_LEFT: Motion.ROTATE_LEFT,
    MoveCode.TURN_LEFT: Motion.ROTATE_LEFT,
    MoveCode.ROTATE_RIGHT: Motion.ROTATE_RIGHT,
    MoveCode.TURN_RIGHT: Motion.ROTATE_RIGHT,
}


class NavState(str, Enum):
    """Coordinator state."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class NavigationConfig:
    """Limits for guarded motion.

    ``allow_motion`` defaults to False: nothing physical happens until it
    is enabled on purpose.  Speeds are the existing normalised motor
    speeds; the defaults stay at the previously calibrated demo values.
    """

    allow_motion: bool = False
    max_linear_speed: float = 0.2
    max_rotation_speed: float = 0.3
    request_valid_for_s: float = 0.3
    goal_timeout_s: float = 60.0
    blocked_timeout_s: float = 3.0
    waypoint_tolerance_m: float = 0.10
    align_tolerance_rad: float = 0.20
    realign_tolerance_rad: float = 0.45
    no_progress_timeout_s: float = 8.0
    min_progress_m: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.max_linear_speed <= 1.0:
            raise ValueError("max_linear_speed must be within (0, 1].")

        if not 0.0 < self.max_rotation_speed <= 1.0:
            raise ValueError("max_rotation_speed must be within (0, 1].")


@dataclass(frozen=True)
class PrimitiveGoal:
    """Open-loop timed motion (demo only, no pose needed)."""

    code: MoveCode
    duration_s: float
    speed: float = 0.2

    def __post_init__(self) -> None:
        if self.code is MoveCode.STOP:
            raise ValueError("A primitive goal cannot be STOP.")

        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive.")


@dataclass(frozen=True)
class PathGoal:
    """Follow world-frame waypoints; requires a trusted pose."""

    waypoints: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.waypoints:
            raise ValueError("A path goal needs at least one waypoint.")


@dataclass(frozen=True)
class Feedback:
    """Per-tick progress report."""

    state: NavState
    reason: str
    waypoint_index: int = 0
    distance_remaining_m: float | None = None
    blocked_for_s: float = 0.0
    speed_scale: float = 1.0


@dataclass
class _Run:
    goal: PrimitiveGoal | PathGoal
    started: float
    index: int = 0
    blocked_since: float | None = None
    best_distance: float = math.inf
    progress_time: float = 0.0
    primitive_elapsed: float = 0.0
    last_tick: float | None = None
    rotating: bool = True
    distance_remaining: float | None = None


@dataclass
class NavigationCoordinator:
    """Single-goal navigation state machine."""

    assessor: ObstacleAssessor
    config: NavigationConfig = field(default_factory=NavigationConfig)
    state: NavState = NavState.IDLE
    reason: str = "No goal."
    _run: _Run | None = None

    # -- goal control -------------------------------------------------

    def start(self, goal: PrimitiveGoal | PathGoal, now: float) -> bool:
        """Begin a goal; returns False (and fails) if it is refused."""

        if self.state in (NavState.RUNNING, NavState.PAUSED):
            return False

        if not self.config.allow_motion:
            return self._finish(
                NavState.FAILED, "Physical motion is disabled in config."
            )

        self._run = _Run(goal=goal, started=now, progress_time=now)
        self.state = NavState.RUNNING
        self.reason = "Goal accepted."

        return True

    def cancel(self, reason: str = "Cancelled by request.") -> None:
        """Cancel the current goal; the next tick returns STOP."""

        if self.state in (NavState.RUNNING, NavState.PAUSED):
            self._finish(NavState.CANCELLED, reason)

    def pause(self) -> None:
        """Pause: STOP is returned until ``resume``."""

        if self.state is NavState.RUNNING:
            self.state = NavState.PAUSED
            self.reason = "Paused."

    def resume(self) -> None:
        """Resume a paused goal."""

        if self.state is NavState.PAUSED:
            self.state = NavState.RUNNING
            self.reason = "Resumed."

            if self._run:
                self._run.last_tick = None

    def _finish(self, state: NavState, reason: str) -> bool:
        self.state = state
        self.reason = reason
        self._run = None

        return state is NavState.SUCCEEDED

    # -- main tick ----------------------------------------------------

    def tick(
        self,
        now: float,
        states: Mapping[str, SensorState],
        pose: Pose2D | None = None,
        pose_trusted: bool = False,
        estop: bool = False,
        manual_override: bool = False,
    ) -> tuple[MoveRequest, Feedback]:
        """Advance one control step and return the request to publish."""

        run = self._run

        if self.state in (NavState.RUNNING, NavState.PAUSED) and (
            estop or manual_override
        ):
            self._finish(
                NavState.FAILED,
                "Emergency stop active."
                if estop
                else "Manual override took control.",
            )
            run = None

        if self.state is not NavState.RUNNING or run is None:
            return STOP_REQUEST, self._feedback(run, 0.0)

        goal = run.goal

        if now - run.started > self.config.goal_timeout_s:
            self._finish(NavState.FAILED, "Goal timed out.")

            return STOP_REQUEST, self._feedback(None, 0.0)

        if isinstance(goal, PathGoal):
            wanted = self._path_command(run, now, pose, pose_trusted)
        else:
            wanted = self._primitive_command(run, now)

        if self.state is not NavState.RUNNING or wanted is None:
            return STOP_REQUEST, self._feedback(self._run, 0.0)

        return self._guard(run, wanted, now, states)

    # -- command generation ------------------------------------------

    def _primitive_command(
        self,
        run: _Run,
        now: float,
    ) -> MoveRequest | None:
        goal = run.goal
        assert isinstance(goal, PrimitiveGoal)

        if run.last_tick is not None:
            run.primitive_elapsed += min(now - run.last_tick, 0.2)

        run.last_tick = now

        if run.primitive_elapsed >= goal.duration_s:
            self._finish(NavState.SUCCEEDED, "Timed motion completed.")

            return None

        return self._request(goal.code, goal.speed)

    def _path_command(
        self,
        run: _Run,
        now: float,
        pose: Pose2D | None,
        pose_trusted: bool,
    ) -> MoveRequest | None:
        goal = run.goal
        assert isinstance(goal, PathGoal)

        if pose is None or not pose_trusted:
            self._finish(
                NavState.FAILED,
                "No reliable pose: refusing to follow a path.",
            )

            return None

        cfg = self.config

        while run.index < len(goal.waypoints):
            wx, wy = goal.waypoints[run.index]

            if (
                math.hypot(wx - pose.x, wy - pose.y)
                > cfg.waypoint_tolerance_m
            ):
                break

            run.index += 1
            run.best_distance = math.inf
            run.progress_time = now

        if run.index >= len(goal.waypoints):
            self._finish(NavState.SUCCEEDED, "Reached final waypoint.")

            return None

        wx, wy = goal.waypoints[run.index]
        distance = math.hypot(wx - pose.x, wy - pose.y)

        if distance < run.best_distance - cfg.min_progress_m:
            run.best_distance = distance
            run.progress_time = now
        elif now - run.progress_time > cfg.no_progress_timeout_s:
            self._finish(NavState.FAILED, "No progress towards waypoint.")

            return None

        error = angle_difference(
            math.atan2(wy - pose.y, wx - pose.x),
            pose.yaw,
        )
        # Hysteresis: once rotating, keep rotating until well aligned;
        # while driving, only start rotating after a larger error.
        threshold = (
            cfg.align_tolerance_rad
            if run.rotating
            else cfg.realign_tolerance_rad
        )
        run.rotating = abs(error) > threshold
        run.distance_remaining = distance

        if run.rotating:
            code = (
                MoveCode.ROTATE_LEFT
                if error > 0
                else MoveCode.ROTATE_RIGHT
            )

            return self._request(code, cfg.max_rotation_speed)

        return self._request(MoveCode.FORWARD, cfg.max_linear_speed)

    def _request(self, code: MoveCode, speed: float) -> MoveRequest:
        cfg = self.config
        rotating = code in (
            MoveCode.ROTATE_LEFT,
            MoveCode.ROTATE_RIGHT,
            MoveCode.TURN_LEFT,
            MoveCode.TURN_RIGHT,
        )
        limit = cfg.max_rotation_speed if rotating else cfg.max_linear_speed

        return MoveRequest(
            code,
            min(speed, limit),
            cfg.request_valid_for_s,
        )

    # -- safety gate --------------------------------------------------

    def _guard(
        self,
        run: _Run,
        wanted: MoveRequest,
        now: float,
        states: Mapping[str, SensorState],
    ) -> tuple[MoveRequest, Feedback]:
        motion = MOTION_FOR_CODE[wanted.code]
        verdict = self.assessor.assess_motion(motion, states)

        if verdict.recommendation is Recommendation.PROCEED or (
            verdict.recommendation is Recommendation.SLOW
        ):
            run.blocked_since = None
            speed = wanted.speed * verdict.speed_scale
            request = MoveRequest(
                wanted.code,
                min(speed, wanted.speed),
                wanted.valid_for_s,
                wanted.source,
            )

            return request, self._feedback(run, 0.0, verdict.speed_scale)

        if run.blocked_since is None:
            run.blocked_since = now

        blocked_for = now - run.blocked_since

        if blocked_for > self.config.blocked_timeout_s:
            self._finish(
                NavState.FAILED,
                f"Blocked: {verdict.reason}",
            )

            return STOP_REQUEST, self._feedback(None, blocked_for)

        self.reason = f"Waiting: {verdict.reason}"

        # Do not let blocked time count as lack of progress or timed motion.
        run.progress_time = now
        run.last_tick = now

        return STOP_REQUEST, self._feedback(run, blocked_for)

    def _feedback(
        self,
        run: _Run | None,
        blocked_for: float,
        speed_scale: float = 1.0,
    ) -> Feedback:
        return Feedback(
            state=self.state,
            reason=self.reason,
            waypoint_index=run.index if run else 0,
            distance_remaining_m=run.distance_remaining if run else None,
            blocked_for_s=blocked_for,
            speed_scale=speed_scale,
        )


def waypoints_from_cells(
    cells: Sequence[tuple[int, int]],
    to_world,
) -> tuple[tuple[float, float], ...]:
    """Convert planner cells to world waypoints, skipping the start cell."""

    return tuple(to_world(cell) for cell in cells[1:])

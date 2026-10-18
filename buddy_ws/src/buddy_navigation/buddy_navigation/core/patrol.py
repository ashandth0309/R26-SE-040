"""Security patrol manager.

Runs a user-defined, ordered route as a sequence of guarded legs.  Each leg
is planned with Dijkstra on a grid (when one is provided) and followed by the
``NavigationCoordinator``, so every safety rule of the coordinator applies.

A patrol only starts when localisation is trusted and the ultrasonic array
is healthy.  Anything else is refused with a reason, because a patrol that
cannot know where it is, or cannot sense obstacles, must not move.

The state names match the strings used as ``PatrolRoute.action`` feedback.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum

from buddy_navigation.core.commands import STOP_REQUEST, MoveRequest
from buddy_navigation.core.coordinator import (
    NavigationCoordinator,
    NavState,
    PathGoal,
)
from buddy_navigation.core.geometry import Pose2D
from buddy_navigation.core.grid import PlanningGrid
from buddy_navigation.core.planner import (
    PlannerConfig,
    dijkstra,
)
from buddy_navigation.core.sensor_model import SensorState


class PatrolState(str, Enum):
    """Patrol lifecycle."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class PatrolPoint:
    """One named patrol point in the world frame."""

    name: str
    x: float
    y: float


@dataclass(frozen=True)
class PatrolRoute:
    """User-defined ordered route."""

    route_id: str
    points: tuple[PatrolPoint, ...]
    repeat: bool = False

    def __post_init__(self) -> None:
        if not self.route_id:
            raise ValueError("A route needs an id.")

        if not self.points:
            raise ValueError("A route needs at least one point.")


@dataclass(frozen=True)
class PatrolFeedback:
    """Mirrors the ``PatrolRoute.action`` feedback fields."""

    state: PatrolState
    current_waypoint: int
    total_waypoints: int
    progress: float
    reason: str


@dataclass
class PatrolManager:
    """Sequence of guarded legs with pause, resume and cancel."""

    coordinator: NavigationCoordinator
    grid: PlanningGrid | None = None
    planner_config: PlannerConfig = field(default_factory=PlannerConfig)
    log: Callable[[str], None] | None = None
    state: PatrolState = PatrolState.IDLE
    reason: str = "No patrol."
    events: list[dict] = field(default_factory=list)
    laps: int = 0
    _route: PatrolRoute | None = None
    _leg: int = 0
    _leg_active: bool = False

    # -- logging ------------------------------------------------------

    def _event(self, now: float, event: str, **details) -> None:
        record = {
            "t": round(now, 3),
            "event": event,
            "route": self._route.route_id if self._route else None,
            "state": self.state.value,
            **details,
        }
        self.events.append(record)

        if self.log:
            self.log(json.dumps(record, sort_keys=True))

    # -- control ------------------------------------------------------

    def start(
        self,
        route: PatrolRoute,
        now: float,
        pose: Pose2D | None,
        pose_trusted: bool,
        states: Mapping[str, SensorState],
    ) -> bool:
        """Begin a patrol if every precondition is met."""

        if self.state in (PatrolState.RUNNING, PatrolState.PAUSED):
            return False

        self._route = route
        self._leg = 0
        self._leg_active = False
        self.laps = 0

        refusal = self._preconditions(pose, pose_trusted, states)

        if refusal:
            return self._fail(now, refusal)

        self.state = PatrolState.RUNNING
        self.reason = "Patrol started."
        self._event(now, "patrol_started", points=len(route.points))

        return True

    def _preconditions(
        self,
        pose: Pose2D | None,
        pose_trusted: bool,
        states: Mapping[str, SensorState],
    ) -> str:
        if pose is None or not pose_trusted:
            return "Localisation is not reliable; patrol refused."

        assessment = self.coordinator.assessor.assess(states)

        if assessment.degraded:
            return (
                "Ultrasonic sensors are not healthy "
                f"(missing {list(assessment.missing_sensors)}, "
                f"unreliable {list(assessment.unreliable_sensors)}); "
                "patrol refused."
            )

        return ""

    def pause(self, now: float) -> None:
        """Pause in place."""

        if self.state is PatrolState.RUNNING:
            self.coordinator.pause()
            self.state = PatrolState.PAUSED
            self.reason = "Paused."
            self._event(now, "patrol_paused")

    def resume(self, now: float) -> None:
        """Continue after a pause."""

        if self.state is PatrolState.PAUSED:
            self.coordinator.resume()
            self.state = PatrolState.RUNNING
            self.reason = "Resumed."
            self._event(now, "patrol_resumed")

    def cancel(self, now: float, reason: str = "Cancelled.") -> None:
        """Abort the patrol; the robot stops."""

        if self.state in (PatrolState.RUNNING, PatrolState.PAUSED):
            self.coordinator.cancel(reason)
            self.state = PatrolState.CANCELLED
            self.reason = reason
            self._event(now, "patrol_cancelled", reason=reason)

    def _fail(self, now: float, reason: str) -> bool:
        self.state = PatrolState.FAILED
        self.reason = reason
        self._event(now, "patrol_failed", reason=reason)

        return False

    # -- main tick ----------------------------------------------------

    def tick(
        self,
        now: float,
        states: Mapping[str, SensorState],
        pose: Pose2D | None,
        pose_trusted: bool,
        estop: bool = False,
        manual_override: bool = False,
    ) -> tuple[MoveRequest, PatrolFeedback]:
        """Advance the patrol and return the request to publish."""

        if self.state is PatrolState.RUNNING and not self._leg_active:
            if not self._begin_leg(now, pose, pose_trusted):
                return STOP_REQUEST, self.feedback()

        request, nav = self.coordinator.tick(
            now, states, pose, pose_trusted, estop, manual_override
        )

        if self.state is PatrolState.RUNNING and self._leg_active:
            self._after_leg_tick(now, nav.state)

        return request, self.feedback()

    def _begin_leg(
        self,
        now: float,
        pose: Pose2D | None,
        pose_trusted: bool,
    ) -> bool:
        route = self._route
        target = route.points[self._leg]

        if pose is None or not pose_trusted:
            self._fail(now, "Localisation lost; patrol stopped.")

            return False

        waypoints = [(target.x, target.y)]

        if self.grid is not None:
            start = self.grid.world_to_cell(pose.x, pose.y)
            goal = self.grid.world_to_cell(target.x, target.y)
            plan = dijkstra(self.grid, start, goal, self.planner_config)

            if not plan.success:
                self._fail(
                    now, f"Cannot plan to '{target.name}': {plan.reason}"
                )

                return False

            # Keep one waypoint per cell so heading is re-aimed often;
            # long straight legs let open-loop heading drift build up.
            cells = plan.cells[1:]
            waypoints = [self.grid.cell_to_world(c) for c in cells[:-1]]
            waypoints.append((target.x, target.y))

        if not self.coordinator.start(PathGoal(tuple(waypoints)), now):
            self._fail(now, self.coordinator.reason)

            return False

        self._leg_active = True
        self._event(now, "leg_started", target=target.name, leg=self._leg)

        return True

    def _after_leg_tick(self, now: float, nav_state: NavState) -> None:
        route = self._route
        target = route.points[self._leg]

        if nav_state is NavState.SUCCEEDED:
            self._event(now, "waypoint_reached", target=target.name)
            self._leg_active = False
            self._leg += 1

            if self._leg >= len(route.points):
                self.laps += 1

                if route.repeat:
                    self._leg = 0
                    self._event(now, "lap_completed", laps=self.laps)
                else:
                    self.state = PatrolState.COMPLETED
                    self.reason = "Patrol completed."
                    self._event(now, "patrol_completed")

        elif nav_state in (NavState.FAILED, NavState.CANCELLED):
            self._leg_active = False
            self._fail(
                now,
                f"Leg to '{target.name}' failed: {self.coordinator.reason}",
            )

    def feedback(self) -> PatrolFeedback:
        """Current feedback in ``PatrolRoute.action`` terms."""

        total = len(self._route.points) if self._route else 0
        done = min(self._leg, total)

        if self.state is PatrolState.COMPLETED:
            done = total

        return PatrolFeedback(
            state=self.state,
            current_waypoint=done,
            total_waypoints=total,
            progress=(done / total) if total else 0.0,
            reason=self.reason,
        )


def now_s() -> float:
    """Monotonic seconds (for the ROS layer)."""

    return time.monotonic()

"""ROS-free navigation service: the logic behind the navigation node.

The ROS node only translates messages to and from this class, so every rule
below is covered by ordinary unit tests.

Publishing policy.  The service returns a movement request only while a
goal is running, plus a short burst of STOP requests after a goal ends.  It
stays silent otherwise so it can never fight the voice module's own
``/movement/request`` traffic.  If anyone else publishes a non-STOP movement
request while a goal is running, that is treated as a manual override: the
goal is aborted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from buddy_navigation.core.commands import STOP_REQUEST, MoveCode, MoveRequest
from buddy_navigation.core.config import NavigationSetup
from buddy_navigation.core.coordinator import (
    NavigationCoordinator,
    NavState,
    PathGoal,
    PrimitiveGoal,
)
from buddy_navigation.core.geometry import Pose2D
from buddy_navigation.core.obstacle_assessment import ObstacleAssessor
from buddy_navigation.core.patrol import PatrolManager, PatrolState
from buddy_navigation.core.planner import dijkstra
from buddy_navigation.core.sensor_model import (
    RangeSample,
    SensorHealthMonitor,
)

OWN_SOURCE = "buddy_navigation"
MAX_TIMED_MOTION_S = 5.0
STOP_BURST_TICKS = 3
OVERRIDE_HOLD_S = 1.0

KINDS = (
    "stop", "move_forward", "turn_left", "turn_right", "start_patrol",
    "go_to_location", "cancel", "pause", "resume",
)

TIMED_CODES = {
    "move_forward": (MoveCode.FORWARD, "linear"),
    "turn_left": (MoveCode.ROTATE_LEFT, "rotation"),
    "turn_right": (MoveCode.ROTATE_RIGHT, "rotation"),
}


@dataclass(frozen=True)
class CommandRequest:
    """Transport-independent navigation request."""

    kind: str
    request_id: str = ""
    duration_s: float = 0.0
    route_id: str = ""
    repeat: bool = False
    location_name: str = ""


@dataclass(frozen=True)
class Ack:
    """Immediate answer to a command."""

    accepted: bool
    reason: str


@dataclass(frozen=True)
class StatusReport:
    """Snapshot for ``NavigationStatus``."""

    state: str
    request_id: str
    active_command: str
    reason: str
    motion_enabled: bool
    pose_trusted: bool
    sensors_healthy: bool
    emergency_stop_active: bool
    progress: float


@dataclass
class NavigationService:
    """Owns the monitor, coordinator and patrol manager."""

    setup: NavigationSetup
    log: object = None
    monitor: SensorHealthMonitor = field(init=False)
    coordinator: NavigationCoordinator = field(init=False)
    patrol: PatrolManager = field(init=False)
    pose: Pose2D | None = None
    pose_trusted: bool = False
    estop: bool = False
    request_id: str = ""
    active_command: str = ""
    _override_until: float = -math.inf
    _stop_ticks: int = 0
    _mode: str = "idle"

    def __post_init__(self) -> None:
        self.monitor = SensorHealthMonitor(self.setup.sensors)
        assessor = ObstacleAssessor(
            self.setup.sensors, self.setup.footprint, self.setup.assessment
        )
        self.coordinator = NavigationCoordinator(
            assessor, self.setup.navigation
        )
        self.patrol = PatrolManager(
            self.coordinator, self.setup.planning_grid, log=self.log
        )

    # -- inputs -------------------------------------------------------

    def update_sample(self, sample: RangeSample) -> None:
        """Feed one ultrasonic reading."""

        self.monitor.update(sample)

    def set_pose(self, pose: Pose2D | None, trusted: bool) -> None:
        """Set the latest pose and whether it may be relied on."""

        self.pose = pose
        self.pose_trusted = bool(trusted and pose is not None)

    def set_estop(self, active: bool) -> None:
        """Mirror the safety supervisor's emergency-stop latch."""

        self.estop = bool(active)

    def note_external_movement(
        self,
        now: float,
        source: str,
        code: int,
    ) -> None:
        """Record a movement request somebody else published."""

        if source != OWN_SOURCE and code != int(MoveCode.STOP):
            self._override_until = now + OVERRIDE_HOLD_S

    # -- commands -----------------------------------------------------

    def handle(self, cmd: CommandRequest, now: float) -> Ack:
        """Execute a navigation command."""

        if cmd.kind not in KINDS:
            return Ack(False, f"Unknown command '{cmd.kind}'.")

        if cmd.kind == "stop" or cmd.kind == "cancel":
            self._cancel_everything(now)

            return Ack(True, "Stopped.")

        if cmd.kind == "pause":
            return self._pause(now)

        if cmd.kind == "resume":
            return self._resume(now)

        if self._busy():
            return Ack(False, "A goal is already running; cancel it first.")

        self.request_id = cmd.request_id
        self.active_command = cmd.kind

        if cmd.kind in TIMED_CODES:
            return self._timed(cmd, now)

        if cmd.kind == "start_patrol":
            return self._start_patrol(cmd, now)

        return self._go_to(cmd, now)

    def _busy(self) -> bool:
        return self._mode != "idle" and self._active_state() in (
            NavState.RUNNING.value, NavState.PAUSED.value,
        )

    def _active_state(self) -> str:
        if self._mode == "patrol":
            return {
                PatrolState.RUNNING: NavState.RUNNING.value,
                PatrolState.PAUSED: NavState.PAUSED.value,
                PatrolState.COMPLETED: NavState.SUCCEEDED.value,
                PatrolState.FAILED: NavState.FAILED.value,
                PatrolState.CANCELLED: NavState.CANCELLED.value,
                PatrolState.IDLE: NavState.IDLE.value,
            }[self.patrol.state]

        return self.coordinator.state.value

    def _reason(self) -> str:
        if self._mode == "patrol":
            return self.patrol.reason

        return self.coordinator.reason

    def _cancel_everything(self, now: float) -> None:
        if self._mode == "patrol":
            self.patrol.cancel(now)
        else:
            self.coordinator.cancel()

        self._stop_ticks = STOP_BURST_TICKS

    def _pause(self, now: float) -> Ack:
        if not self._busy():
            return Ack(False, "Nothing to pause.")

        if self._mode == "patrol":
            self.patrol.pause(now)
        else:
            self.coordinator.pause()

        self._stop_ticks = STOP_BURST_TICKS

        return Ack(True, "Paused.")

    def _resume(self, now: float) -> Ack:
        if not self._busy():
            return Ack(False, "Nothing to resume.")

        if self._mode == "patrol":
            self.patrol.resume(now)
        else:
            self.coordinator.resume()

        return Ack(True, "Resumed.")

    def _timed(self, cmd: CommandRequest, now: float) -> Ack:
        code, kind = TIMED_CODES[cmd.kind]
        duration = cmd.duration_s or self.setup.default_timed_motion_s
        duration = min(max(duration, 0.1), MAX_TIMED_MOTION_S)
        nav = self.setup.navigation
        speed = (
            nav.max_linear_speed if kind == "linear"
            else nav.max_rotation_speed
        )
        self._mode = "primitive"

        if self.coordinator.start(PrimitiveGoal(code, duration, speed), now):
            return Ack(True, "Timed motion started.")

        return Ack(False, self.coordinator.reason)

    def _start_patrol(self, cmd: CommandRequest, now: float) -> Ack:
        route = self.setup.routes.get(cmd.route_id)

        if route is None:
            return Ack(False, f"Unknown route '{cmd.route_id}'.")

        if cmd.repeat and not route.repeat:
            from dataclasses import replace

            route = replace(route, repeat=True)

        self._mode = "patrol"
        states = self.monitor.snapshot(now)

        if self.patrol.start(
            route, now, self.pose, self.pose_trusted, states
        ):
            return Ack(True, "Patrol started.")

        return Ack(False, self.patrol.reason)

    def _go_to(self, cmd: CommandRequest, now: float) -> Ack:
        target = self.setup.locations.get(cmd.location_name)

        if target is None:
            return Ack(False, f"Unknown location '{cmd.location_name}'.")

        if not self.pose_trusted or self.pose is None:
            return Ack(
                False,
                "Localisation is not reliable; cannot go to a location.",
            )

        waypoints = [target]
        grid = self.setup.planning_grid

        if grid is not None:
            plan = dijkstra(
                grid,
                grid.world_to_cell(self.pose.x, self.pose.y),
                grid.world_to_cell(*target),
            )

            if not plan.success:
                return Ack(False, f"Cannot plan: {plan.reason}")

            waypoints = [grid.cell_to_world(c) for c in plan.cells[1:-1]]
            waypoints.append(target)

        self._mode = "goto"

        if self.coordinator.start(PathGoal(tuple(waypoints)), now):
            return Ack(True, "Navigating to location.")

        return Ack(False, self.coordinator.reason)

    # -- periodic -----------------------------------------------------

    def tick(self, now: float) -> tuple[MoveRequest | None, StatusReport]:
        """Run one control step.

        Returns the request to publish (or None for "publish nothing") and
        the current status.
        """

        states = self.monitor.snapshot(now)
        override = now < self._override_until
        request = None

        if self._mode == "patrol":
            request, _ = self.patrol.tick(
                now, states, self.pose, self.pose_trusted,
                self.estop, override,
            )
            running = self.patrol.state is PatrolState.RUNNING
        elif self._mode in ("primitive", "goto"):
            request, _ = self.coordinator.tick(
                now, states, self.pose, self.pose_trusted,
                self.estop, override,
            )
            running = self.coordinator.state is NavState.RUNNING
        else:
            running = False

        if running:
            self._stop_ticks = STOP_BURST_TICKS
        elif self._stop_ticks > 0:
            self._stop_ticks -= 1
            request = STOP_REQUEST
        else:
            request = None

        return request, self.status(now, states)

    def status(self, now: float, states=None) -> StatusReport:
        """Current status snapshot."""

        if states is None:
            states = self.monitor.snapshot(now)

        assessment = self.coordinator.assessor.assess(states)
        progress = 0.0

        if self._mode == "patrol":
            progress = self.patrol.feedback().progress
        elif self._active_state() == NavState.SUCCEEDED.value:
            progress = 1.0

        return StatusReport(
            state=self._active_state(),
            request_id=self.request_id,
            active_command=self.active_command,
            reason=self._reason(),
            motion_enabled=self.setup.navigation.allow_motion,
            pose_trusted=self.pose_trusted,
            sensors_healthy=not assessment.degraded,
            emergency_stop_active=self.estop,
            progress=progress,
        )

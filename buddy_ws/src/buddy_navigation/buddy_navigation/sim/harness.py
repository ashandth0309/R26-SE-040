"""Closed-loop simulation harness (simulation only).

Wires together the same pieces the robot uses:

    mock sensors -> health monitor -> coordinator -> safety gate -> robot

The safety gate is the real ``SafetySupervisor`` class from ``buddy_core``
fed with ``/obstacle_status``-style flags computed with the same pair
minimum rule as ``obstacle_node``, plus the supervisor node's one-second
freshness rule.  Nothing here drives hardware.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from buddy_core.safety.supervisor import SafetySupervisor

from buddy_navigation.core.commands import STOP_REQUEST, MoveCode, MoveRequest
from buddy_navigation.core.coordinator import (
    Feedback,
    NavigationConfig,
    NavigationCoordinator,
)
from buddy_navigation.core.geometry import Pose2D
from buddy_navigation.core.local_map import LocalObstacleMap
from buddy_navigation.core.obstacle_assessment import ObstacleAssessor
from buddy_navigation.core.presets import (
    simulation_assessment_config,
    simulation_footprint,
    simulation_sensor_specs,
)
from buddy_navigation.core.sensor_model import (
    SensorHealthMonitor,
    SensorStatus,
)
from buddy_navigation.sim.mock_robot import MockRobot
from buddy_navigation.sim.mock_ultrasonic import MockUltrasonicArray
from buddy_navigation.sim.world import World2D

SUPERVISOR_CODE = {code: int(code) for code in MoveCode}


@dataclass
class SimHarness:
    """Run navigation goals against a simulated world."""

    world: World2D
    start: Pose2D
    nav_config: NavigationConfig = field(
        default_factory=lambda: NavigationConfig(allow_motion=True)
    )
    seed: int = 1
    noise_std_m: float = 0.01
    pose_trusted: bool = True
    obstacle_stop_distance_m: float = 0.25
    control_period_s: float = 0.1
    dt: float = 0.01

    def __post_init__(self) -> None:
        specs = simulation_sensor_specs()
        footprint = simulation_footprint()
        self.specs = specs
        self.array = MockUltrasonicArray(
            self.world, specs, noise_std_m=self.noise_std_m, seed=self.seed
        )
        self.monitor = SensorHealthMonitor(specs)
        self.assessor = ObstacleAssessor(
            specs, footprint, simulation_assessment_config()
        )
        self.coordinator = NavigationCoordinator(
            self.assessor, self.nav_config
        )
        self.robot = MockRobot(self.world, footprint, self.start)
        self.supervisor = SafetySupervisor()
        self.local_map = LocalObstacleMap()
        self.now = 0.0
        self._next_control = 0.0
        self.last_obstacle_msg = -math.inf
        self.latest_distance: dict[str, float | None] = {}
        self.feedback: Feedback | None = None
        self.manual_override = False
        self.published: list[tuple[float, MoveRequest]] = []
        self.rejected: list[tuple[float, str]] = []
        self.min_clearance_m = math.inf

    # -- helpers ------------------------------------------------------

    def _legacy_flags(self) -> dict:
        d = self.latest_distance

        def pair(a, b):
            values = [v for v in (d.get(a), d.get(b)) if v is not None]
            valid = bool(values)

            return valid, (min(values) if valid else math.nan)

        stop = self.obstacle_stop_distance_m
        fv, f = pair("front_left", "front_right")
        rv, r = pair("rear_left", "rear_right")
        lv, left = pair("front_left", "rear_left")
        rtv, right = pair("front_right", "rear_right")

        return dict(
            front_blocked=fv and f <= stop,
            rear_blocked=rv and r <= stop,
            left_blocked=lv and left <= stop,
            right_blocked=rtv and right <= stop,
        )

    def _gate(self, request: MoveRequest) -> MoveRequest:
        available = self.now - self.last_obstacle_msg <= 1.0
        decision = self.supervisor.evaluate(
            SUPERVISOR_CODE[request.code],
            obstacle_data_available=available,
            **self._legacy_flags(),
        )

        if not decision.allowed:
            self.rejected.append((self.now, decision.reason))

            return STOP_REQUEST

        return request

    def emergency_stop(self) -> None:
        """Latch the e-stop exactly like the /emergency_stop service."""

        self.supervisor.activate_emergency_stop()
        self.robot.apply(STOP_REQUEST, self.now)

    def reset_emergency_stop(self) -> None:
        """Clear the latch."""

        self.supervisor.reset_emergency_stop()

    # -- simulation loop ---------------------------------------------

    def step(self) -> None:
        """Advance by ``dt`` seconds."""

        sample = self.array.step(self.robot.pose, self.now)

        if sample is not None:
            self.monitor.update(sample)
            spec = self.array.specs[sample.name]
            self.local_map.update(
                spec, sample.status, sample.distance_m, self.now,
                self.robot.pose,
            )
            self.latest_distance[sample.name] = (
                sample.distance_m
                if sample.status is SensorStatus.OK
                else None
            )

            if len(self.latest_distance) == len(self.specs):
                self.last_obstacle_msg = self.now

        if self.now >= self._next_control:
            self._next_control += self.control_period_s
            self._control()

        self.robot.step(self.now, self.dt)
        self.local_map.expire(self.now)
        self._track_clearance()
        self.now += self.dt

    def _control(self) -> None:
        states = self.monitor.snapshot(self.now)
        request, self.feedback = self.coordinator.tick(
            self.now,
            states,
            pose=self.robot.pose,
            pose_trusted=self.pose_trusted,
            estop=self.supervisor.emergency_stop_active,
            manual_override=self.manual_override,
        )
        self.published.append((self.now, request))
        self.robot.apply(self._gate(request), self.now)

    def _track_clearance(self) -> None:
        pose = self.robot.pose

        for box in self.world.obstacles:
            dx = max(box.xmin - pose.x, 0.0, pose.x - box.xmax)
            dy = max(box.ymin - pose.y, 0.0, pose.y - box.ymax)
            self.min_clearance_m = min(
                self.min_clearance_m, math.hypot(dx, dy)
            )

    def run(
        self,
        seconds: float,
        until: Callable[["SimHarness"], bool] | None = None,
    ) -> None:
        """Run for up to ``seconds`` or until ``until`` is true."""

        end = self.now + seconds

        while self.now < end:
            self.step()

            if until is not None and until(self):
                break

"""Kinematic mock robot that obeys the real motor_executor rules."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from buddy_navigation.core.commands import MoveCode, MoveRequest
from buddy_navigation.core.geometry import Pose2D, normalize_angle
from buddy_navigation.core.sensor_model import Footprint
from buddy_navigation.sim.world import World2D


@dataclass(frozen=True)
class MotionModel:
    """SIMULATION ASSUMPTIONS, not measured Buddy characteristics.

    Metres per second and radians per second produced by speed 1.0.  The
    real values must be characterised on the chassis before any open-loop
    motion is enabled.
    """

    metres_per_s_per_unit: float = 0.35
    rad_per_s_per_unit: float = 2.0
    command_timeout_s: float = 1.0


@dataclass
class MockRobot:
    """Moves only while a fresh, non-STOP command is active.

    Mirrors ``motor_executor``: a command stops being applied after
    ``min(valid_for, command_timeout)``, so a silent controller means a
    stopped robot.  A collision halts the robot and is recorded; tests
    should assert it never happens.
    """

    world: World2D
    footprint: Footprint
    pose: Pose2D
    model: MotionModel = field(default_factory=MotionModel)
    collided: bool = False
    odometer_m: float = 0.0
    history: list[tuple[float, float, float, float]] = field(
        default_factory=list
    )
    _command: MoveRequest | None = None
    _command_time: float = -math.inf

    def apply(self, request: MoveRequest, now: float) -> None:
        """Receive an already safety-approved command."""

        self._command = request
        self._command_time = now

    def active_command(self, now: float) -> MoveRequest | None:
        """The command currently being executed, if still valid."""

        if self._command is None or self._command.code is MoveCode.STOP:
            return None

        limit = min(self._command.valid_for_s, self.model.command_timeout_s)

        if now - self._command_time > limit:
            return None

        return self._command

    @property
    def stopped(self) -> bool:
        """Whether the robot has no active motion (as of the last step)."""

        return self._last_active is None

    _last_active: MoveRequest | None = None

    def step(self, now: float, dt: float) -> None:
        """Advance the simulation by ``dt`` seconds."""

        active = self.active_command(now)
        self._last_active = active
        self.history.append((now, self.pose.x, self.pose.y, self.pose.yaw))

        if active is None or self.collided:
            return

        speed = active.speed
        code = active.code
        v = 0.0
        w = 0.0

        if code is MoveCode.FORWARD:
            v = speed * self.model.metres_per_s_per_unit
        elif code is MoveCode.BACKWARD:
            v = -speed * self.model.metres_per_s_per_unit
        elif code in (MoveCode.ROTATE_LEFT, MoveCode.TURN_LEFT):
            w = speed * self.model.rad_per_s_per_unit
        elif code in (MoveCode.ROTATE_RIGHT, MoveCode.TURN_RIGHT):
            w = -speed * self.model.rad_per_s_per_unit

        yaw = normalize_angle(self.pose.yaw + w * dt)
        x = self.pose.x + v * dt * math.cos(self.pose.yaw)
        y = self.pose.y + v * dt * math.sin(self.pose.yaw)

        if self.world.collides(x, y, self.footprint.circumradius_m):
            self.collided = True

            return

        self.odometer_m += abs(v) * dt
        self.pose = Pose2D(x, y, yaw)

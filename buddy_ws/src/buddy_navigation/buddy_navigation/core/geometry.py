"""Small 2D geometry helpers following ROS REP-103 conventions.

Frames are right handed: x forward, y left, yaw counter-clockwise positive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def normalize_angle(angle: float) -> float:
    """Wrap an angle in radians into [-pi, pi)."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_difference(target: float, source: float) -> float:
    """Return the signed shortest rotation from ``source`` to ``target``."""

    return normalize_angle(target - source)


@dataclass(frozen=True)
class Pose2D:
    """A planar pose: position in metres and heading in radians."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def transform_point(
        self,
        local_x: float,
        local_y: float,
    ) -> tuple[float, float]:
        """Express a point given in this pose's frame in the parent frame."""

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        return (
            self.x + cos_yaw * local_x - sin_yaw * local_y,
            self.y + sin_yaw * local_x + cos_yaw * local_y,
        )

    def compose(self, other: "Pose2D") -> "Pose2D":
        """Return ``other`` (given in this frame) in the parent frame."""

        x, y = self.transform_point(other.x, other.y)

        return Pose2D(x, y, normalize_angle(self.yaw + other.yaw))

    def inverse_transform_point(
        self,
        parent_x: float,
        parent_y: float,
    ) -> tuple[float, float]:
        """Express a parent-frame point in this pose's frame."""

        dx = parent_x - self.x
        dy = parent_y - self.y
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        return (
            cos_yaw * dx + sin_yaw * dy,
            -sin_yaw * dx + cos_yaw * dy,
        )


def distance(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    """Euclidean distance between two points."""

    return math.hypot(second[0] - first[0], second[1] - first[1])

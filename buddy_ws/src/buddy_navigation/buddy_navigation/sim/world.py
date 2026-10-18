"""Configurable 2D world with analytic ray casting (simulation only)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from buddy_navigation.core.geometry import Pose2D, normalize_angle
from buddy_navigation.core.sensor_model import SensorSpec


@dataclass(frozen=True)
class Box:
    """Axis-aligned rectangular obstacle."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        if self.xmax <= self.xmin or self.ymax <= self.ymin:
            raise ValueError("Box needs positive width and height.")

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Whether a point lies inside the box grown by ``margin``."""

        return (
            self.xmin - margin <= x <= self.xmax + margin
            and self.ymin - margin <= y <= self.ymax + margin
        )

    def ray(self, ox: float, oy: float, dx: float, dy: float) -> float | None:
        """Distance along a ray to the box (slab method), or None."""

        near, far = -math.inf, math.inf

        for origin, direction, low, high in (
            (ox, dx, self.xmin, self.xmax),
            (oy, dy, self.ymin, self.ymax),
        ):
            if abs(direction) < 1e-12:
                if origin < low or origin > high:
                    return None

                continue

            t1 = (low - origin) / direction
            t2 = (high - origin) / direction
            near = max(near, min(t1, t2))
            far = min(far, max(t1, t2))

        if near > far or far < 0.0:
            return None

        return max(near, 0.0)


@dataclass
class World2D:
    """Rectangular room [0,width]x[0,height] with box obstacles."""

    width_m: float
    height_m: float
    obstacles: list[Box] = field(default_factory=list)

    def _walls(self) -> list[Box]:
        t = 0.05
        w, h = self.width_m, self.height_m

        return [
            Box(-t, -t, w + t, 0.0),
            Box(-t, h, w + t, h + t),
            Box(-t, -t, 0.0, h + t),
            Box(w, -t, w + t, h + t),
        ]

    def add_obstacle(self, box: Box) -> None:
        """Add an obstacle at runtime (e.g. a person stepping in)."""

        self.obstacles.append(box)

    def ray_distance(
        self,
        x: float,
        y: float,
        yaw: float,
        max_range: float,
    ) -> float:
        """Distance to the nearest surface along a ray (``inf`` if none)."""

        dx, dy = math.cos(yaw), math.sin(yaw)
        best = math.inf

        for box in self.obstacles + self._walls():
            hit = box.ray(x, y, dx, dy)

            if hit is not None and hit < best:
                best = hit

        return best if best <= max_range else math.inf

    def beam_distance(
        self,
        robot: Pose2D,
        spec: SensorSpec,
        rays: int = 15,
    ) -> float:
        """Nearest-reflector distance inside a sensor cone.

        Models a wide ultrasonic beam as the minimum over several rays,
        which is why the true object bearing is ambiguous.
        """

        sx, sy = robot.transform_point(spec.x, spec.y)
        base = normalize_angle(robot.yaw + spec.yaw)
        best = math.inf

        for i in range(rays):
            frac = i / (rays - 1) - 0.5 if rays > 1 else 0.0
            yaw = base + frac * spec.fov_rad
            best = min(
                best,
                self.ray_distance(sx, sy, yaw, spec.max_range_m),
            )

        return best

    def collides(self, x: float, y: float, radius: float) -> bool:
        """Whether a disc of ``radius`` at (x, y) touches anything."""

        if (
            x - radius < 0.0
            or y - radius < 0.0
            or x + radius > self.width_m
            or y + radius > self.height_m
        ):
            return True

        return any(box.contains(x, y, radius) for box in self.obstacles)

    @classmethod
    def from_rows(cls, rows: list[str], cell_m: float) -> "World2D":
        """Build a world from text rows (top row first, ``#`` = blocked)."""

        height = len(rows)
        width = len(rows[0])
        world = cls(width * cell_m, height * cell_m)

        for printed, text in enumerate(rows):
            row = height - 1 - printed

            for column, char in enumerate(text):
                if char == "#":
                    world.obstacles.append(
                        Box(
                            column * cell_m,
                            row * cell_m,
                            (column + 1) * cell_m,
                            (row + 1) * cell_m,
                        )
                    )

        return world


SIX_BY_SIX_ROWS = [
    "......",
    ".##...",
    ".#..#.",
    ".#.##.",
    "...#..",
    "......",
]


def six_by_six_world(cell_m: float = 0.5) -> World2D:
    """Simple 6x6 teaching scenario (NOT a map of Buddy's real space)."""

    return World2D.from_rows(SIX_BY_SIX_ROWS, cell_m)

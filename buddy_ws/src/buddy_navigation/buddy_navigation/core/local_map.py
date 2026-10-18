"""Conservative local obstacle map built from wide-beam ultrasonic readings.

An HC-SR04 echo only says "the nearest reflector inside a ~30 degree cone is
at range r".  The object could be anywhere on the arc at radius r.  This map
therefore:

* raises a *moderate* occupancy belief on the whole arc (never a precise
  point), and treats the cone in front of it as free evidence only;
* leaves cells never observed as UNKNOWN, and does not treat them as free;
* lets every cell expire back to UNKNOWN so stale readings cannot linger;
* ignores faulty, stale or missing sensors rather than reading them as free.

Frames.  Without a trustworthy pose (the current robot has no encoders or
IMU) the map is ``robot-relative``: the caller feeds it with the identity
pose, a short ``ttl_s`` keeps it honest, and it must not be used for global
planning.  With a pose from a trusted source (simulation ground truth, or a
future odometry source) the same class works as a ``tracked`` world map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from buddy_navigation.core.geometry import Pose2D, normalize_angle
from buddy_navigation.core.grid import UNKNOWN, OccupancyGrid2D
from buddy_navigation.core.sensor_model import SensorSpec, SensorStatus


@dataclass(frozen=True)
class LocalMapConfig:
    """Local map parameters (metres, seconds, log-odds)."""

    size_m: float = 4.0
    resolution_m: float = 0.05
    ttl_s: float = 3.0
    arc_tolerance_m: float = 0.04
    arc_log_odds: float = 0.36
    too_close_log_odds: float = 1.5
    free_log_odds: float = -0.40
    clamp_log_odds: float = 3.0
    free_margin_m: float = 0.08
    clear_means_free: bool = True
    occupied_threshold: int = 55


class LocalObstacleMap:
    """Log-odds occupancy grid with expiry and unknown-space handling."""

    def __init__(self, config: LocalMapConfig | None = None) -> None:
        self.config = config or LocalMapConfig()
        cfg = self.config
        cells = int(round(cfg.size_m / cfg.resolution_m))
        self.log_odds = np.zeros((cells, cells), dtype=np.float32)
        self.observed = np.zeros((cells, cells), dtype=bool)
        self.updated_at = np.full((cells, cells), -math.inf)
        self.origin = -cfg.size_m / 2.0

        centres = self.origin + (np.arange(cells) + 0.5) * cfg.resolution_m
        self._cx, self._cy = np.meshgrid(centres, centres)

    @property
    def cells(self) -> int:
        """Cells along each side."""

        return int(self.log_odds.shape[0])

    def clear(self) -> None:
        """Forget everything (all cells become UNKNOWN)."""

        self.log_odds[:] = 0.0
        self.observed[:] = False
        self.updated_at[:] = -math.inf

    def expire(self, now: float) -> int:
        """Reset cells older than ``ttl_s`` to unknown; return the count."""

        stale = self.observed & (now - self.updated_at > self.config.ttl_s)
        count = int(stale.sum())

        if count:
            self.log_odds[stale] = 0.0
            self.observed[stale] = False
            self.updated_at[stale] = -math.inf

        return count

    def update(
        self,
        spec: SensorSpec,
        status: SensorStatus,
        distance_m: float | None,
        now: float,
        pose: Pose2D | None = None,
    ) -> None:
        """Integrate one sensor reading taken while the robot was at ``pose``.

        ``pose`` is the robot pose in the map frame (identity when omitted).
        Only OK, CLEAR and TOO_CLOSE readings change the map.
        """

        if status not in (
            SensorStatus.OK,
            SensorStatus.CLEAR,
            SensorStatus.TOO_CLOSE,
        ):
            return

        pose = pose or Pose2D(0.0, 0.0, 0.0)
        cfg = self.config

        sx, sy = pose.transform_point(spec.x, spec.y)
        syaw = normalize_angle(pose.yaw + spec.yaw)

        dx = self._cx - sx
        dy = self._cy - sy
        radius = np.hypot(dx, dy)
        bearing = np.arctan2(dy, dx)
        off_axis = np.abs(
            (bearing - syaw + math.pi) % (2 * math.pi) - math.pi
        )
        in_cone = (off_axis <= spec.fov_rad / 2.0) & (
            radius >= spec.min_range_m
        )

        if status is SensorStatus.OK and distance_m is not None:
            free = in_cone & (
                radius < distance_m - cfg.arc_tolerance_m - cfg.free_margin_m
            )
            arc = in_cone & (
                np.abs(radius - distance_m) <= cfg.arc_tolerance_m
            )
            self._add(free, cfg.free_log_odds, now)
            self._add(arc, cfg.arc_log_odds, now)

        elif status is SensorStatus.CLEAR and cfg.clear_means_free:
            free = in_cone & (radius < spec.max_range_m - cfg.free_margin_m)
            self._add(free, cfg.free_log_odds, now)

        elif status is SensorStatus.TOO_CLOSE:
            # An object closer than the minimum range cannot be localised:
            # it may sit anywhere from the sensor face outwards.  Mark the
            # whole forward half-disc next to the sensor, at least 1.5
            # cells wide so it is never smaller than the grid can show.
            reach = max(
                spec.min_range_m + cfg.arc_tolerance_m + 0.02,
                1.5 * cfg.resolution_m,
            )
            forward = (dx * math.cos(syaw) + dy * math.sin(syaw)) >= 0.0
            self._add((radius <= reach) & forward, cfg.too_close_log_odds, now)

    def _add(self, mask: np.ndarray, delta: float, now: float) -> None:
        cfg = self.config
        self.log_odds[mask] = np.clip(
            self.log_odds[mask] + delta,
            -cfg.clamp_log_odds,
            cfg.clamp_log_odds,
        )
        self.observed[mask] = True
        self.updated_at[mask] = now

    def to_occupancy_grid(
        self,
        frame_id: str = "base_link",
    ) -> OccupancyGrid2D:
        """Export as percent occupancy; unobserved cells are -1 (unknown)."""

        probability = 1.0 / (1.0 + np.exp(-self.log_odds.astype(np.float64)))
        data = np.where(
            self.observed,
            np.rint(probability * 100.0),
            UNKNOWN,
        ).astype(np.int8)

        return OccupancyGrid2D(
            data=data,
            resolution_m=self.config.resolution_m,
            origin_x=self.origin,
            origin_y=self.origin,
            frame_id=frame_id,
        )

    def occupancy_at(self, x: float, y: float) -> int:
        """Percent occupancy at a point, or -1 for unknown/outside."""

        cfg = self.config
        column = int(math.floor((x - self.origin) / cfg.resolution_m))
        row = int(math.floor((y - self.origin) / cfg.resolution_m))

        if not (0 <= column < self.cells and 0 <= row < self.cells):
            return UNKNOWN

        if not self.observed[row, column]:
            return UNKNOWN

        return int(
            round(100.0 / (1.0 + math.exp(-float(self.log_odds[row, column]))))
        )

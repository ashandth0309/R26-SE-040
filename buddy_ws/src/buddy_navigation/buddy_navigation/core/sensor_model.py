"""Ultrasonic sensor geometry and health tracking.

Nothing here depends on ROS.  The ROS nodes convert ``sensor_msgs/Range``
messages into ``RangeSample`` objects and feed them to the
``SensorHealthMonitor``.

A raw ultrasonic reading is not trustworthy on its own.  The monitor turns
the stream of samples into a ``SensorState`` per sensor that distinguishes:

* ``OK``: a fresh measured distance;
* ``CLEAR``: fresh, but nothing was detected within range;
* ``TOO_CLOSE``: an object nearer than the sensor can resolve;
* ``FAULT``: the module reports no response, a stuck line or a read error;
* ``STALE``: no sample has arrived recently enough;
* ``NO_DATA``: no sample has ever arrived.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum


class SensorStatus(str, Enum):
    """Navigation-level interpretation of one sensor."""

    OK = "ok"
    CLEAR = "clear"
    TOO_CLOSE = "too_close"
    FAULT = "fault"
    STALE = "stale"
    NO_DATA = "no_data"


USABLE_STATUSES = frozenset(
    {
        SensorStatus.OK,
        SensorStatus.CLEAR,
        SensorStatus.TOO_CLOSE,
    }
)


@dataclass(frozen=True)
class Footprint:
    """Robot body rectangle centred on the base frame origin.

    ``margin_m`` is extra clearance added around the body for every safety
    decision.  Dimensions are configurable and must be measured.
    """

    length_m: float = 0.30
    width_m: float = 0.20
    margin_m: float = 0.05

    def __post_init__(self) -> None:
        if self.length_m <= 0 or self.width_m <= 0:
            raise ValueError("Footprint dimensions must be positive.")

        if self.margin_m < 0:
            raise ValueError("Footprint margin must not be negative.")

    @property
    def circumradius_m(self) -> float:
        """Radius of the circle swept by the body while pivoting."""

        return 0.5 * math.hypot(self.length_m, self.width_m)

    @property
    def inflation_radius_m(self) -> float:
        """Radius used to inflate obstacles for planning."""

        return self.circumradius_m + self.margin_m


@dataclass(frozen=True)
class SensorSpec:
    """Mounting pose and beam model for one ultrasonic sensor."""

    name: str
    x: float
    y: float
    yaw: float
    fov_rad: float = math.radians(30.0)
    min_range_m: float = 0.02
    max_range_m: float = 4.0

    def __post_init__(self) -> None:
        if not 0.0 < self.fov_rad < math.pi:
            raise ValueError(
                f"Sensor '{self.name}' fov_rad must be in (0, pi)."
            )

        if not 0.0 < self.min_range_m < self.max_range_m:
            raise ValueError(
                f"Sensor '{self.name}' needs 0 < min_range < max_range."
            )

    def covers_point(self, px: float, py: float) -> bool:
        """Return whether a base-frame point lies inside the beam."""

        dx = px - self.x
        dy = py - self.y
        rho = math.hypot(dx, dy)

        if rho < self.min_range_m or rho > self.max_range_m:
            return False

        bearing = math.atan2(dy, dx) - self.yaw
        bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi

        return abs(bearing) <= 0.5 * self.fov_rad


@dataclass(frozen=True)
class RangeSample:
    """One measurement as received from the sensor layer."""

    name: str
    status: SensorStatus
    distance_m: float | None
    timestamp_s: float

    def __post_init__(self) -> None:
        if self.status is SensorStatus.OK and (
            self.distance_m is None or not math.isfinite(self.distance_m)
        ):
            raise ValueError("An OK sample needs a finite distance.")


@dataclass(frozen=True)
class SensorState:
    """Health-checked view of one sensor at a point in time."""

    name: str
    status: SensorStatus
    distance_m: float | None
    age_s: float | None
    consecutive_faults: int
    recent_near: bool

    @property
    def usable(self) -> bool:
        """Whether the state may be used for a movement decision."""

        return self.status in USABLE_STATUSES


@dataclass
class _SensorTrack:
    samples: deque = field(default_factory=lambda: deque(maxlen=3))
    last_status: SensorStatus = SensorStatus.NO_DATA
    last_timestamp: float | None = None
    consecutive_faults: int = 0
    clear_streak: int = 0
    last_near_timestamp: float | None = None


class SensorHealthMonitor:
    """Track per-sensor health, filtering and stale-data detection."""

    def __init__(
        self,
        specs: Iterable[SensorSpec],
        *,
        stale_after_s: float = 0.8,
        near_distance_m: float = 0.45,
        clear_confirmations: int = 3,
        near_memory_s: float = 1.5,
    ) -> None:
        self._specs = {spec.name: spec for spec in specs}

        if not self._specs:
            raise ValueError("At least one sensor spec is required.")

        if stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive.")

        if clear_confirmations < 1:
            raise ValueError("clear_confirmations must be at least 1.")

        self._stale_after = float(stale_after_s)
        self._near_distance = float(near_distance_m)
        self._clear_confirmations = int(clear_confirmations)
        self._near_memory = float(near_memory_s)
        self._tracks = {name: _SensorTrack() for name in self._specs}

    @property
    def names(self) -> tuple[str, ...]:
        """Configured sensor names."""

        return tuple(self._specs)

    def update(self, sample: RangeSample) -> None:
        """Record one sample.  Unknown sensor names are rejected."""

        track = self._tracks.get(sample.name)

        if track is None:
            raise KeyError(f"Unknown sensor '{sample.name}'.")

        if (
            track.last_timestamp is not None
            and sample.timestamp_s < track.last_timestamp
        ):
            # Out-of-order sample: ignore rather than rewind history.
            return

        track.last_status = sample.status
        track.last_timestamp = sample.timestamp_s

        if sample.status is SensorStatus.FAULT:
            track.consecutive_faults += 1
            track.clear_streak = 0
            return

        track.consecutive_faults = 0

        if sample.status is SensorStatus.OK:
            track.samples.append(float(sample.distance_m))
            track.clear_streak = 0

            if sample.distance_m <= self._near_distance:
                track.last_near_timestamp = sample.timestamp_s

        elif sample.status is SensorStatus.TOO_CLOSE:
            track.samples.clear()
            track.clear_streak = 0
            track.last_near_timestamp = sample.timestamp_s

        elif sample.status is SensorStatus.CLEAR:
            track.clear_streak += 1

    def _filtered_distance(self, track: _SensorTrack) -> float | None:
        if not track.samples:
            return None

        # The nearer of the newest reading and the short-window median:
        # an approaching obstacle is reported immediately, while a single
        # spurious far reading cannot hide a nearer obstacle.
        return min(track.samples[-1], statistics.median(track.samples))

    def state(self, name: str, now_s: float) -> SensorState:
        """Return the health-checked state of one sensor."""

        track = self._tracks[name]

        if track.last_timestamp is None:
            return SensorState(
                name,
                SensorStatus.NO_DATA,
                None,
                None,
                track.consecutive_faults,
                False,
            )

        age = max(0.0, now_s - track.last_timestamp)
        recent_near = (
            track.last_near_timestamp is not None
            and now_s - track.last_near_timestamp <= self._near_memory
        )

        if age > self._stale_after:
            return SensorState(
                name,
                SensorStatus.STALE,
                None,
                age,
                track.consecutive_faults,
                recent_near,
            )

        status = track.last_status
        distance = None

        if status is SensorStatus.OK:
            distance = self._filtered_distance(track)

        elif status is SensorStatus.CLEAR and recent_near:
            # An obstacle was seen very recently.  Until enough consecutive
            # CLEAR results confirm it has gone, keep the last near
            # distance: an angled or soft surface can drop out of the echo.
            if track.clear_streak < self._clear_confirmations:
                status = SensorStatus.OK
                distance = self._filtered_distance(track)

        elif status is SensorStatus.TOO_CLOSE:
            distance = 0.0

        return SensorState(
            name,
            status,
            distance,
            age,
            track.consecutive_faults,
            recent_near,
        )

    def snapshot(self, now_s: float) -> Mapping[str, SensorState]:
        """Return every sensor's state at ``now_s``."""

        return {name: self.state(name, now_s) for name in self._specs}

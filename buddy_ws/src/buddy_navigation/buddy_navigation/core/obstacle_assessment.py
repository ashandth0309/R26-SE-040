"""Configurable, coverage-aware obstacle assessment.

This replaces rules of the form "front distance < X, so turn right".  For
each intended motion the assessor samples the area the robot body would
sweep through, using the robot footprint, and asks of every sample point:

* is it *proven free* (a healthy beam covers it and the reading is farther);
* is it *possibly occupied* (it lies on the arc where a beam saw an echo);
* or is it *unobserved* (no healthy beam covers it, or it is hidden behind
  a nearer echo)?

An ultrasonic echo only says "the nearest surface inside this cone is at
distance d".  An obstacle could be anywhere on that arc, so the whole arc
inside the corridor is treated as possibly occupied, and nothing beyond it
is claimed to be free.

Unobserved area in the stop zone is never assumed to be free.  If too much
of the stop zone is unobserved, because a sensor is missing, stale or
faulty or because the beams simply do not cover it, the decision is STOP.

An obstacle narrower than ``min_obstacle_width_m`` can slip between beams;
coverage is therefore judged against an obstacle of that minimum width, not
against an infinitely thin one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from buddy_navigation.core.sensor_model import (
    Footprint,
    SensorSpec,
    SensorState,
    SensorStatus,
)


class Motion(str, Enum):
    """Intended body motions the assessor can judge."""

    FORWARD = "forward"
    BACKWARD = "backward"
    ROTATE_LEFT = "rotate_left"
    ROTATE_RIGHT = "rotate_right"


class Recommendation(str, Enum):
    """What the caller should do about the intended motion."""

    PROCEED = "proceed"
    SLOW = "slow"
    REPLAN = "replan"
    STOP = "stop"


class Cause(str, Enum):
    """Why a recommendation other than PROCEED was made."""

    NONE = "none"
    OBSTACLE = "obstacle"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    SENSOR_FAULT = "sensor_fault"
    PARTIAL_COVERAGE = "partial_coverage"


class _Evidence(Enum):
    FREE = 1
    OCCUPIED = 2
    UNKNOWN = 3
    TOO_CLOSE = 4


@dataclass(frozen=True)
class AssessmentConfig:
    """Tunable assessment thresholds.  All distances are metres.

    The coverage requirements default to the strictest value (every sample
    point of the swept area must be observed).  Four corner-mounted
    ultrasonic beams cannot achieve that, so with these defaults the
    assessor will STOP until the real sensor geometry has been measured and
    the team has consciously chosen lower, documented thresholds.
    """

    stop_distance_m: float = 0.25
    slow_distance_m: float = 0.60
    min_obstacle_width_m: float = 0.12
    arc_tolerance_m: float = 0.06
    sample_step_m: float = 0.05
    required_coverage_linear: float = 1.0
    required_coverage_rotation: float = 1.0
    slow_speed_scale: float = 0.5
    rotation_lookahead_rad: float = 0.6
    side_equal_margin_m: float = 0.10
    clear_means_free: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.stop_distance_m < self.slow_distance_m:
            raise ValueError("Need 0 < stop_distance_m < slow_distance_m.")

        if self.sample_step_m <= 0 or self.min_obstacle_width_m <= 0:
            raise ValueError("Step and obstacle width must be positive.")

        if not 0.0 < self.slow_speed_scale <= 1.0:
            raise ValueError("slow_speed_scale must be in (0, 1].")

        for name in (
            "required_coverage_linear",
            "required_coverage_rotation",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")


@dataclass(frozen=True)
class MotionAssessment:
    """Assessment of one intended motion."""

    motion: Motion
    recommendation: Recommendation
    cause: Cause
    reason: str
    speed_scale: float
    clearance_m: float | None
    coverage_fraction: float
    missing_sensors: tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        """Whether the motion may start or continue."""

        return self.recommendation in (
            Recommendation.PROCEED,
            Recommendation.SLOW,
        )


@dataclass(frozen=True)
class ObstacleAssessment:
    """Assessment of every motion plus overall sensor health."""

    motions: Mapping[Motion, MotionAssessment]
    sensors: Mapping[str, SensorState]
    left_clearance_m: float | None
    right_clearance_m: float | None
    more_clearance_side: str
    missing_sensors: tuple[str, ...] = field(default_factory=tuple)
    unreliable_sensors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def front_blocked(self) -> bool:
        """Whether moving forward is currently not allowed."""

        return not self.motions[Motion.FORWARD].allowed

    @property
    def reverse_safe(self) -> bool:
        """Whether reversing is currently allowed."""

        return self.motions[Motion.BACKWARD].allowed

    @property
    def degraded(self) -> bool:
        """Whether any sensor is missing or unreliable."""

        return bool(self.missing_sensors or self.unreliable_sensors)

    def for_motion(self, motion: Motion) -> MotionAssessment:
        """Return the assessment for one motion."""

        return self.motions[motion]


@dataclass(frozen=True)
class _Point:
    zone_depth: float
    entries: tuple[tuple[str, float], ...]
    in_stop_zone: bool


class ObstacleAssessor:
    """Judge intended motions against health-checked sensor states."""

    def __init__(
        self,
        sensors: tuple[SensorSpec, ...],
        footprint: Footprint,
        config: AssessmentConfig | None = None,
    ) -> None:
        if not sensors:
            raise ValueError("At least one sensor spec is required.")

        self._specs = {spec.name: spec for spec in sensors}
        self._footprint = footprint
        self._config = config or AssessmentConfig()
        self._points = {
            Motion.FORWARD: self._linear_points(+1.0),
            Motion.BACKWARD: self._linear_points(-1.0),
            Motion.ROTATE_LEFT: self._rotation_points(+1.0),
            Motion.ROTATE_RIGHT: self._rotation_points(-1.0),
        }

    @property
    def config(self) -> AssessmentConfig:
        """The thresholds in use."""

        return self._config

    @property
    def sensor_names(self) -> tuple[str, ...]:
        """Configured sensor names."""

        return tuple(self._specs)

    # ------------------------------------------------------------------
    # Geometry, computed once because it only depends on configuration
    # ------------------------------------------------------------------

    def _probe_offsets(self) -> tuple[tuple[float, float], ...]:
        half = 0.5 * self._config.min_obstacle_width_m

        return (
            (0.0, 0.0),
            (half, 0.0),
            (-half, 0.0),
            (0.0, half),
            (0.0, -half),
        )

    def _entries_for(self, px: float, py: float) -> tuple:
        entries = []

        for name, spec in self._specs.items():
            for dx, dy in self._probe_offsets():
                qx, qy = px + dx, py + dy

                if spec.covers_point(qx, qy):
                    rho = math.hypot(qx - spec.x, qy - spec.y)
                    entries.append((name, rho))

        return tuple(entries)

    def _linear_points(self, direction: float) -> tuple[_Point, ...]:
        cfg = self._config
        fp = self._footprint
        edge = 0.5 * fp.length_m
        half_width = 0.5 * fp.width_m + fp.margin_m
        step = cfg.sample_step_m
        depth_count = int(round(cfg.slow_distance_m / step))
        lateral_count = int(math.ceil(half_width / step))

        points = []

        for i in range(depth_count + 1):
            depth = i * step

            for j in range(-lateral_count, lateral_count + 1):
                lateral = max(-half_width, min(half_width, j * step))
                px = direction * (edge + depth)
                entries = self._entries_for(px, lateral)

                points.append(
                    _Point(
                        zone_depth=depth,
                        entries=entries,
                        in_stop_zone=depth <= cfg.stop_distance_m,
                    )
                )

        return tuple(points)

    def _rotation_points(self, direction: float) -> tuple[_Point, ...]:
        cfg = self._config
        fp = self._footprint
        half_length = 0.5 * fp.length_m
        half_width = 0.5 * fp.width_m
        expanded_length = half_length + fp.margin_m
        expanded_width = half_width + fp.margin_m
        radius = math.hypot(expanded_length, expanded_width)
        step = cfg.sample_step_m
        count = int(math.ceil(radius / step))
        angles = [
            direction * cfg.rotation_lookahead_rad * k / 6.0
            for k in range(1, 7)
        ]

        points = []

        for i in range(-count, count + 1):
            for j in range(-count, count + 1):
                px, py = i * step, j * step

                if abs(px) <= half_length and abs(py) <= half_width:
                    continue

                swept = False

                for phi in angles:
                    bx = math.cos(phi) * px + math.sin(phi) * py
                    by = -math.sin(phi) * px + math.cos(phi) * py

                    if abs(bx) <= expanded_length and abs(by) <= expanded_width:
                        swept = True
                        break

                if swept:
                    points.append(
                        _Point(
                            zone_depth=0.0,
                            entries=self._entries_for(px, py),
                            in_stop_zone=True,
                        )
                    )

        return tuple(points)

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------

    def _evidence(
        self,
        point: _Point,
        states: Mapping[str, SensorState],
    ) -> tuple[_Evidence, float | None]:
        tolerance = self._config.arc_tolerance_m
        free = False
        occupied_at: float | None = None

        for name, rho in point.entries:
            state = states.get(name)

            if state is None or not state.usable:
                continue

            if state.status is SensorStatus.TOO_CLOSE:
                return _Evidence.TOO_CLOSE, 0.0

            if state.status is SensorStatus.CLEAR:
                if self._config.clear_means_free:
                    free = True

                continue

            distance = state.distance_m

            if distance is None:
                continue

            if rho < distance - tolerance:
                free = True
            elif rho <= distance + tolerance:
                occupied_at = distance if occupied_at is None else min(
                    occupied_at, distance
                )

        if occupied_at is not None:
            return _Evidence.OCCUPIED, occupied_at

        if free:
            return _Evidence.FREE, None

        return _Evidence.UNKNOWN, None

    def _missing_for(
        self,
        motion: Motion,
        states: Mapping[str, SensorState],
    ) -> tuple[str, ...]:
        relevant = {
            name
            for point in self._points[motion]
            for name, _ in point.entries
        }

        return tuple(
            sorted(
                name
                for name in relevant
                if name not in states or not states[name].usable
            )
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_motion(
        self,
        motion: Motion,
        states: Mapping[str, SensorState],
    ) -> MotionAssessment:
        """Assess one intended motion."""

        cfg = self._config
        points = self._points[motion]
        linear = motion in (Motion.FORWARD, Motion.BACKWARD)
        required = (
            cfg.required_coverage_linear
            if linear
            else cfg.required_coverage_rotation
        )

        stop_total = 0
        stop_observed = 0
        slow_total = 0
        slow_observed = 0
        stop_obstacle_depth: float | None = None
        slow_obstacle_depth: float | None = None
        too_close = False

        for point in points:
            evidence, _ = self._evidence(point, states)
            observed = evidence in (_Evidence.FREE, _Evidence.OCCUPIED)

            if evidence is _Evidence.TOO_CLOSE:
                too_close = True
                observed = True

            if point.in_stop_zone:
                stop_total += 1
                stop_observed += 1 if observed else 0

                if evidence is _Evidence.OCCUPIED:
                    depth = point.zone_depth

                    if (
                        stop_obstacle_depth is None
                        or depth < stop_obstacle_depth
                    ):
                        stop_obstacle_depth = depth
            else:
                slow_total += 1
                slow_observed += 1 if observed else 0

                if evidence is _Evidence.OCCUPIED:
                    depth = point.zone_depth

                    if (
                        slow_obstacle_depth is None
                        or depth < slow_obstacle_depth
                    ):
                        slow_obstacle_depth = depth

        coverage = stop_observed / stop_total if stop_total else 0.0
        missing = self._missing_for(motion, states)

        if too_close:
            return MotionAssessment(
                motion,
                Recommendation.REPLAN,
                Cause.OBSTACLE,
                "Obstacle closer than the sensor minimum range.",
                0.0,
                0.0,
                coverage,
                missing,
            )

        if stop_obstacle_depth is not None:
            return MotionAssessment(
                motion,
                Recommendation.REPLAN,
                Cause.OBSTACLE,
                f"Obstacle within {stop_obstacle_depth + 0.0:.2f} m "
                "of the body in the stop zone.",
                0.0,
                stop_obstacle_depth,
                coverage,
                missing,
            )

        if coverage < required:
            cause = Cause.SENSOR_FAULT if missing else (
                Cause.INSUFFICIENT_COVERAGE
            )
            detail = (
                f"sensors missing or unreliable: {', '.join(missing)}"
                if missing
                else "sensor beams do not cover the swept area"
            )

            return MotionAssessment(
                motion,
                Recommendation.STOP,
                cause,
                f"Insufficient sensor coverage ({coverage:.0%} of the "
                f"stop zone, {required:.0%} required): {detail}.",
                0.0,
                None,
                coverage,
                missing,
            )

        if slow_obstacle_depth is not None:
            return MotionAssessment(
                motion,
                Recommendation.SLOW,
                Cause.OBSTACLE,
                f"Obstacle {slow_obstacle_depth:.2f} m ahead of the body; "
                "slowing.",
                cfg.slow_speed_scale,
                slow_obstacle_depth,
                coverage,
                missing,
            )

        slow_coverage = slow_observed / slow_total if slow_total else 1.0

        # Blind spots remain even though coverage meets the requirement.
        # Anything that could hide there is unseen, so reduce speed.
        if coverage < 1.0 - 1e-9 or (linear and slow_coverage < required):
            return MotionAssessment(
                motion,
                Recommendation.SLOW,
                Cause.PARTIAL_COVERAGE,
                f"Blind spots remain ({coverage:.0%} of the stop zone "
                "observed); slowing.",
                cfg.slow_speed_scale,
                None,
                coverage,
                missing,
            )

        return MotionAssessment(
            motion,
            Recommendation.PROCEED,
            Cause.NONE,
            "Path observed clear.",
            1.0,
            None,
            coverage,
            missing,
        )

    def _side_clearance(
        self,
        states: Mapping[str, SensorState],
        side_sign: float,
    ) -> float | None:
        values = []

        for name, spec in self._specs.items():
            if spec.y * side_sign <= 0:
                continue

            state = states.get(name)

            if state is None or not state.usable:
                return None

            if state.status is SensorStatus.TOO_CLOSE:
                values.append(0.0)
            elif state.status is SensorStatus.CLEAR:
                values.append(spec.max_range_m)
            elif state.distance_m is not None:
                values.append(state.distance_m)

        return min(values) if values else None

    def assess(
        self,
        states: Mapping[str, SensorState],
    ) -> ObstacleAssessment:
        """Assess every motion and summarise sensor health."""

        motions = {
            motion: self.assess_motion(motion, states)
            for motion in Motion
        }

        left = self._side_clearance(states, +1.0)
        right = self._side_clearance(states, -1.0)

        if left is None or right is None:
            side = "unknown"
        elif abs(left - right) < self._config.side_equal_margin_m:
            side = "equal"
        else:
            side = "left" if left > right else "right"

        missing = tuple(
            sorted(
                name
                for name in self._specs
                if name not in states or not states[name].usable
            )
        )
        unreliable = tuple(
            sorted(
                name
                for name in self._specs
                if name in states
                and (
                    states[name].consecutive_faults > 0
                    or states[name].status is SensorStatus.FAULT
                )
            )
        )

        return ObstacleAssessment(
            motions=motions,
            sensors=dict(states),
            left_clearance_m=left,
            right_clearance_m=right,
            more_clearance_side=side,
            missing_sensors=missing,
            unreliable_sensors=unreliable,
        )

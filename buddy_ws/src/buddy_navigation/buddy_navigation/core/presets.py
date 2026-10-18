"""SIMULATION-ONLY default geometry and thresholds.

Nothing in this module describes the real BUDDY chassis.  The footprint,
sensor mounting poses and relaxed coverage thresholds below exist so the
navigation stack can be exercised on a laptop.  The real values must be
measured on the robot and supplied through configuration.
"""

from __future__ import annotations

import math

from buddy_navigation.core.obstacle_assessment import AssessmentConfig
from buddy_navigation.core.sensor_model import Footprint, SensorSpec

SENSOR_NAMES = ("front_left", "front_right", "rear_left", "rear_right")


def simulation_footprint() -> Footprint:
    """Assumed body size (SIMULATION ONLY): 30 cm x 20 cm."""

    return Footprint(length_m=0.30, width_m=0.20, margin_m=0.05)


def simulation_sensor_specs(
    fov_deg: float = 30.0,
    max_range_m: float = 4.0,
) -> tuple[SensorSpec, ...]:
    """Assumed corner mounting (SIMULATION ONLY).

    Front sensors toe out by 20 degrees, rear sensors likewise, each pair
    sitting 8 cm either side of the centre line.
    """

    fov = math.radians(fov_deg)
    toe = math.radians(20.0)

    def spec(name, x, y, yaw):
        return SensorSpec(
            name=name,
            x=x,
            y=y,
            yaw=yaw,
            fov_rad=fov,
            min_range_m=0.02,
            max_range_m=max_range_m,
        )

    return (
        spec("front_left", 0.13, 0.08, toe),
        spec("front_right", 0.13, -0.08, -toe),
        spec("rear_left", -0.13, 0.08, math.pi - toe),
        spec("rear_right", -0.13, -0.08, -math.pi + toe),
    )


def simulation_assessment_config() -> AssessmentConfig:
    """Relaxed coverage thresholds (SIMULATION ONLY).

    With the assumed geometry the beams observe about 85 % of the forward
    stop zone and about 40 % of a pivot sweep.  These thresholds accept
    that, and the assessor then slows the robot because blind spots remain.
    """

    return AssessmentConfig(
        required_coverage_linear=0.80,
        required_coverage_rotation=0.35,
    )

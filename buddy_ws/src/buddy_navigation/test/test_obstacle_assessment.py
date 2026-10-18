"""Tests for coverage-aware obstacle assessment."""

import math

import pytest

from buddy_navigation.core.obstacle_assessment import (
    AssessmentConfig,
    Cause,
    Motion,
    ObstacleAssessor,
    Recommendation,
)
from buddy_navigation.core.presets import (
    simulation_assessment_config,
    simulation_footprint,
    simulation_sensor_specs,
)
from buddy_navigation.core.sensor_model import (
    SensorSpec,
    SensorState,
    SensorStatus,
)

NAMES = ("front_left", "front_right", "rear_left", "rear_right")


def state(name, status=SensorStatus.OK, distance=None, faults=0):
    return SensorState(name, status, distance, 0.05, faults, False)


def states(**kwargs):
    """Build sensor states.  Unspecified sensors are CLEAR."""

    result = {}

    for name in NAMES:
        value = kwargs.get(name, "clear")

        if value == "clear":
            result[name] = state(name, SensorStatus.CLEAR)
        elif value == "fault":
            result[name] = state(name, SensorStatus.FAULT, faults=3)
        elif value == "stale":
            result[name] = state(name, SensorStatus.STALE)
        elif value == "too_close":
            result[name] = state(name, SensorStatus.TOO_CLOSE, 0.0)
        else:
            result[name] = state(name, SensorStatus.OK, float(value))

    return result


@pytest.fixture
def assessor():
    return ObstacleAssessor(
        simulation_sensor_specs(),
        simulation_footprint(),
        simulation_assessment_config(),
    )


def forward(assessor, **kwargs):
    return assessor.assess_motion(Motion.FORWARD, states(**kwargs))


def test_open_space_allows_forward_but_flags_blind_spots(assessor):
    result = forward(assessor)

    assert result.allowed
    assert result.recommendation is Recommendation.SLOW
    assert result.cause is Cause.PARTIAL_COVERAGE
    assert 0.8 <= result.coverage_fraction < 1.0
    assert result.speed_scale == pytest.approx(0.5)


def test_obstacle_directly_ahead_in_stop_zone_blocks_forward(assessor):
    result = forward(assessor, front_left=0.30, front_right=0.30)

    assert not result.allowed
    assert result.recommendation is Recommendation.REPLAN
    assert result.cause is Cause.OBSTACLE


def test_obstacle_in_slow_zone_slows_forward(assessor):
    result = forward(assessor, front_left=0.55, front_right=0.55)

    assert result.recommendation is Recommendation.SLOW
    assert result.cause is Cause.OBSTACLE
    assert result.allowed


def test_distant_obstacle_does_not_affect_forward(assessor):
    result = forward(assessor, front_left=2.0, front_right=2.0)

    assert result.allowed
    assert result.cause is not Cause.OBSTACLE


def test_front_blocked_is_not_a_blanket_turn_rule(assessor):
    snapshot = assessor.assess(
        states(
            front_left=0.30,
            front_right=0.30,
            rear_left=2.0,
            rear_right=2.0,
        )
    )

    assert snapshot.front_blocked
    assert snapshot.reverse_safe


def test_more_clearance_side_follows_the_readings(assessor):
    left_open = assessor.assess(
        states(
            front_left=3.0,
            rear_left=3.0,
            front_right=0.7,
            rear_right=0.8,
        )
    )
    right_open = assessor.assess(
        states(
            front_right=3.0,
            rear_right=3.0,
            front_left=0.7,
            rear_left=0.8,
        )
    )

    assert left_open.more_clearance_side == "left"
    assert right_open.more_clearance_side == "right"


def test_equal_clearance_is_reported_as_equal(assessor):
    snapshot = assessor.assess(
        states(
            front_left=1.0,
            rear_left=1.0,
            front_right=1.02,
            rear_right=1.04,
        )
    )

    assert snapshot.more_clearance_side == "equal"


def test_clearance_side_is_unknown_when_a_side_sensor_is_missing(assessor):
    snapshot = assessor.assess(states(front_left="fault"))

    assert snapshot.more_clearance_side == "unknown"
    assert snapshot.left_clearance_m is None


def test_obstacle_at_rear_blocks_reverse_but_not_forward(assessor):
    snapshot = assessor.assess(
        states(rear_left=0.30, rear_right=0.30),
    )

    assert not snapshot.reverse_safe
    assert not snapshot.front_blocked


def test_too_close_reading_blocks_motion_in_that_direction(assessor):
    result = forward(assessor, front_left="too_close")

    assert result.recommendation is Recommendation.REPLAN
    assert result.cause is Cause.OBSTACLE


@pytest.mark.parametrize("bad", ["fault", "stale"])
def test_missing_front_sensor_stops_forward(assessor, bad):
    result = forward(assessor, front_left=bad, front_right=bad)

    assert result.recommendation is Recommendation.STOP
    assert result.cause is Cause.SENSOR_FAULT
    assert set(result.missing_sensors) >= {"front_left", "front_right"}
    assert result.coverage_fraction < 0.8


def test_one_front_sensor_down_stops_forward_when_coverage_drops(assessor):
    result = forward(assessor, front_left="fault")

    assert result.recommendation is Recommendation.STOP
    assert "front_left" in result.missing_sensors


def test_no_sensor_data_at_all_stops_every_motion(assessor):
    empty = {}

    for motion in Motion:
        result = assessor.assess_motion(motion, empty)

        assert result.recommendation is Recommendation.STOP, motion
        assert result.coverage_fraction == 0.0


def test_strict_defaults_stop_even_with_all_sensors_healthy():
    strict = ObstacleAssessor(
        simulation_sensor_specs(),
        simulation_footprint(),
        AssessmentConfig(),
    )

    result = strict.assess_motion(Motion.FORWARD, states())

    assert result.recommendation is Recommendation.STOP
    assert result.cause is Cause.INSUFFICIENT_COVERAGE


def test_narrow_beams_that_miss_the_corridor_centre_stop_forward():
    narrow = tuple(
        SensorSpec(
            spec.name,
            spec.x,
            spec.y,
            math.copysign(math.radians(60), spec.y) if spec.x > 0
            else spec.yaw,
            fov_rad=math.radians(8),
        )
        for spec in simulation_sensor_specs()
    )
    assessor = ObstacleAssessor(
        narrow,
        simulation_footprint(),
        simulation_assessment_config(),
    )

    result = assessor.assess_motion(Motion.FORWARD, states())

    assert result.recommendation is Recommendation.STOP
    assert result.cause is Cause.INSUFFICIENT_COVERAGE


def test_clear_is_not_free_when_configured_conservatively():
    cautious = ObstacleAssessor(
        simulation_sensor_specs(),
        simulation_footprint(),
        AssessmentConfig(
            required_coverage_linear=0.8,
            required_coverage_rotation=0.35,
            clear_means_free=False,
        ),
    )

    result = cautious.assess_motion(Motion.FORWARD, states())

    assert result.recommendation is Recommendation.STOP


def test_rotation_blocked_by_an_obstacle_beside_the_body(assessor):
    result = assessor.assess_motion(
        Motion.ROTATE_LEFT,
        states(front_left=0.12, rear_left=0.12),
    )

    assert not result.allowed


def test_rotation_allowed_in_open_space_with_blind_spot_slowdown(assessor):
    result = assessor.assess_motion(Motion.ROTATE_LEFT, states())

    assert result.allowed
    assert result.speed_scale <= 0.5


def test_rotation_stops_when_most_sensors_are_down(assessor):
    result = assessor.assess_motion(
        Motion.ROTATE_RIGHT,
        states(
            front_left="fault",
            front_right="fault",
            rear_left="fault",
        ),
    )

    assert result.recommendation is Recommendation.STOP


def test_assessment_reports_missing_and_unreliable_sensors(assessor):
    snapshot = assessor.assess(states(rear_right="fault", front_left="stale"))

    assert snapshot.degraded
    assert set(snapshot.missing_sensors) == {"rear_right", "front_left"}
    assert "rear_right" in snapshot.unreliable_sensors


def test_healthy_system_is_not_degraded(assessor):
    assert not assessor.assess(states()).degraded


def test_all_motions_are_assessed(assessor):
    snapshot = assessor.assess(states())

    assert set(snapshot.motions) == set(Motion)


def test_config_validation():
    with pytest.raises(ValueError):
        AssessmentConfig(stop_distance_m=0.8, slow_distance_m=0.5)

    with pytest.raises(ValueError):
        AssessmentConfig(slow_speed_scale=0.0)

    with pytest.raises(ValueError):
        AssessmentConfig(required_coverage_linear=1.5)

    with pytest.raises(ValueError):
        ObstacleAssessor((), simulation_footprint())

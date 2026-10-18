"""Tests for sensor geometry and the sensor health monitor."""

import math

import pytest

from buddy_navigation.core.presets import (
    simulation_footprint,
    simulation_sensor_specs,
)
from buddy_navigation.core.sensor_model import (
    Footprint,
    RangeSample,
    SensorHealthMonitor,
    SensorSpec,
    SensorStatus,
)


def sample(name, status, distance, t):
    return RangeSample(name, status, distance, t)


@pytest.fixture
def monitor():
    return SensorHealthMonitor(
        simulation_sensor_specs(),
        stale_after_s=0.8,
        near_distance_m=0.45,
        clear_confirmations=3,
        near_memory_s=1.5,
    )


def test_footprint_circumradius_and_inflation():
    fp = Footprint(0.30, 0.20, 0.05)

    assert fp.circumradius_m == pytest.approx(math.hypot(0.15, 0.10))
    assert fp.inflation_radius_m == pytest.approx(
        math.hypot(0.15, 0.10) + 0.05
    )


def test_footprint_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        Footprint(0.0, 0.2)

    with pytest.raises(ValueError):
        Footprint(0.3, 0.2, margin_m=-0.1)


def test_sensor_beam_covers_points_inside_the_cone_only():
    spec = SensorSpec("s", 0.0, 0.0, 0.0, fov_rad=math.radians(30))

    assert spec.covers_point(1.0, 0.0)
    assert spec.covers_point(1.0, 0.2)
    assert not spec.covers_point(1.0, 0.4)
    assert not spec.covers_point(-1.0, 0.0)
    assert not spec.covers_point(5.0, 0.0)
    assert not spec.covers_point(0.01, 0.0)


def test_sensor_spec_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        SensorSpec("s", 0, 0, 0, fov_rad=0.0)

    with pytest.raises(ValueError):
        SensorSpec("s", 0, 0, 0, min_range_m=1.0, max_range_m=0.5)


def test_ok_sample_requires_a_finite_distance():
    with pytest.raises(ValueError):
        RangeSample("front_left", SensorStatus.OK, None, 0.0)

    with pytest.raises(ValueError):
        RangeSample("front_left", SensorStatus.OK, math.nan, 0.0)


def test_no_data_before_the_first_sample(monitor):
    state = monitor.state("front_left", 0.0)

    assert state.status is SensorStatus.NO_DATA
    assert not state.usable


def test_fresh_ok_sample_is_usable(monitor):
    monitor.update(sample("front_left", SensorStatus.OK, 1.0, 10.0))

    state = monitor.state("front_left", 10.1)

    assert state.status is SensorStatus.OK
    assert state.distance_m == pytest.approx(1.0)
    assert state.usable


def test_stale_data_is_not_usable(monitor):
    monitor.update(sample("front_left", SensorStatus.OK, 1.0, 10.0))

    state = monitor.state("front_left", 11.0)

    assert state.status is SensorStatus.STALE
    assert state.distance_m is None
    assert not state.usable


def test_fault_samples_count_and_are_unusable(monitor):
    for t in (1.0, 1.1, 1.2):
        monitor.update(sample("front_left", SensorStatus.FAULT, None, t))

    state = monitor.state("front_left", 1.3)

    assert state.status is SensorStatus.FAULT
    assert state.consecutive_faults == 3
    assert not state.usable


def test_good_sample_resets_the_fault_counter(monitor):
    monitor.update(sample("front_left", SensorStatus.FAULT, None, 1.0))
    monitor.update(sample("front_left", SensorStatus.OK, 1.0, 1.1))

    assert monitor.state("front_left", 1.2).consecutive_faults == 0


def test_approaching_obstacle_is_reported_without_lag(monitor):
    for t, d in ((1.0, 2.0), (1.1, 2.0), (1.2, 0.3)):
        monitor.update(sample("front_left", SensorStatus.OK, d, t))

    assert monitor.state("front_left", 1.25).distance_m == pytest.approx(0.3)


def test_single_far_outlier_cannot_hide_a_near_obstacle(monitor):
    for t, d in ((1.0, 0.3), (1.1, 0.3), (1.2, 3.0)):
        monitor.update(sample("front_left", SensorStatus.OK, d, t))

    assert monitor.state("front_left", 1.25).distance_m == pytest.approx(0.3)


def test_clear_is_not_trusted_immediately_after_a_near_reading(monitor):
    monitor.update(sample("front_left", SensorStatus.OK, 0.3, 1.0))
    monitor.update(sample("front_left", SensorStatus.CLEAR, None, 1.1))

    state = monitor.state("front_left", 1.15)

    assert state.status is SensorStatus.OK
    assert state.distance_m == pytest.approx(0.3)


def test_clear_is_accepted_after_enough_confirmations(monitor):
    monitor.update(sample("front_left", SensorStatus.OK, 0.3, 1.0))

    for t in (1.1, 1.2, 1.3):
        monitor.update(sample("front_left", SensorStatus.CLEAR, None, t))

    assert monitor.state("front_left", 1.35).status is SensorStatus.CLEAR


def test_clear_is_accepted_when_no_recent_obstacle_was_seen(monitor):
    monitor.update(sample("front_left", SensorStatus.CLEAR, None, 1.0))

    assert monitor.state("front_left", 1.1).status is SensorStatus.CLEAR


def test_too_close_reports_zero_distance_and_is_usable(monitor):
    monitor.update(sample("front_left", SensorStatus.TOO_CLOSE, None, 1.0))

    state = monitor.state("front_left", 1.1)

    assert state.status is SensorStatus.TOO_CLOSE
    assert state.distance_m == 0.0
    assert state.usable


def test_out_of_order_sample_is_ignored(monitor):
    monitor.update(sample("front_left", SensorStatus.OK, 1.0, 5.0))
    monitor.update(sample("front_left", SensorStatus.OK, 0.1, 4.0))

    assert monitor.state("front_left", 5.1).distance_m == pytest.approx(1.0)


def test_unknown_sensor_name_is_rejected(monitor):
    with pytest.raises(KeyError):
        monitor.update(sample("top", SensorStatus.OK, 1.0, 1.0))


def test_snapshot_covers_every_sensor(monitor):
    assert set(monitor.snapshot(0.0)) == set(monitor.names)


def test_default_simulation_geometry_is_labelled_and_valid():
    fp = simulation_footprint()
    specs = simulation_sensor_specs()

    assert len(specs) == 4
    assert {s.name for s in specs} == {
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
    }
    assert fp.length_m > 0

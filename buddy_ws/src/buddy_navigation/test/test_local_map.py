"""Tests for the conservative local obstacle map."""

import math

import numpy as np
import pytest

from buddy_navigation.core.geometry import Pose2D
from buddy_navigation.core.grid import UNKNOWN
from buddy_navigation.core.local_map import LocalMapConfig, LocalObstacleMap
from buddy_navigation.core.sensor_model import SensorSpec, SensorStatus

FRONT = SensorSpec("front", 0.1, 0.0, 0.0)


@pytest.fixture
def local_map():
    return LocalObstacleMap(LocalMapConfig(size_m=3.0, ttl_s=2.0))


def test_starts_entirely_unknown(local_map):
    grid = local_map.to_occupancy_grid()

    assert (grid.data == UNKNOWN).all()
    assert local_map.occupancy_at(0.5, 0.0) == UNKNOWN


def test_echo_marks_free_cone_and_arc_not_a_point(local_map):
    local_map.update(FRONT, SensorStatus.OK, 0.8, now=0.0)
    cone_half = FRONT.fov_rad / 2

    # Free in front of the echo, on the axis.
    assert local_map.occupancy_at(0.5, 0.0) <= 40
    # The arc is occupied at the axis and at the cone edges.
    assert local_map.occupancy_at(0.1 + 0.8, 0.0) >= 55
    edge_x = 0.1 + 0.8 * math.cos(cone_half * 0.9)
    edge_y = 0.8 * math.sin(cone_half * 0.9)
    assert local_map.occupancy_at(edge_x, edge_y) >= 55
    # Beyond the arc nothing is claimed.
    assert local_map.occupancy_at(1.3, 0.0) == UNKNOWN
    # Outside the cone nothing is claimed.
    assert local_map.occupancy_at(0.5, 0.6) == UNKNOWN


def test_arc_belief_is_moderate_after_one_reading(local_map):
    local_map.update(FRONT, SensorStatus.OK, 0.8, now=0.0)

    assert local_map.occupancy_at(0.9, 0.0) < 70


def test_repeated_readings_raise_confidence(local_map):
    for step in range(5):
        local_map.update(FRONT, SensorStatus.OK, 0.8, now=step * 0.1)

    assert local_map.occupancy_at(0.9, 0.0) > 80


def test_clear_reading_frees_the_cone_and_leaves_the_rest_unknown(local_map):
    local_map.update(FRONT, SensorStatus.CLEAR, None, now=0.0)

    assert 0 <= local_map.occupancy_at(1.0, 0.0) <= 40
    assert local_map.occupancy_at(-0.5, 0.0) == UNKNOWN


def test_clear_can_be_disabled_for_untrusted_clear(local_map):
    cautious = LocalObstacleMap(
        LocalMapConfig(size_m=3.0, clear_means_free=False)
    )
    cautious.update(FRONT, SensorStatus.CLEAR, None, now=0.0)

    assert cautious.occupancy_at(1.0, 0.0) == UNKNOWN


def test_too_close_marks_the_zone_in_front_of_the_sensor_occupied(local_map):
    local_map.update(FRONT, SensorStatus.TOO_CLOSE, None, now=0.0)

    assert local_map.occupancy_at(0.13, 0.0) >= 55


@pytest.mark.parametrize(
    "status",
    [SensorStatus.FAULT, SensorStatus.STALE, SensorStatus.NO_DATA],
)
def test_faulty_sensors_never_change_the_map(local_map, status):
    local_map.update(FRONT, status, 0.5, now=0.0)

    assert not local_map.observed.any()


def test_cells_expire_back_to_unknown(local_map):
    local_map.update(FRONT, SensorStatus.OK, 0.8, now=0.0)

    assert local_map.expire(1.0) == 0
    assert local_map.occupancy_at(0.9, 0.0) >= 55

    assert local_map.expire(2.5) > 0
    assert local_map.occupancy_at(0.9, 0.0) == UNKNOWN
    assert not local_map.observed.any()


def test_fresh_readings_survive_while_old_ones_expire(local_map):
    local_map.update(FRONT, SensorStatus.OK, 0.8, now=0.0)
    rear = SensorSpec("rear", -0.1, 0.0, math.pi)
    local_map.update(rear, SensorStatus.OK, 0.8, now=2.0)
    local_map.expire(3.0)

    assert local_map.occupancy_at(0.9, 0.0) == UNKNOWN
    assert local_map.occupancy_at(-0.9, 0.0) >= 55


def test_tracked_mode_places_obstacles_in_the_world_frame(local_map):
    pose = Pose2D(0.0, 0.0, math.pi / 2)  # facing +y
    local_map.update(FRONT, SensorStatus.OK, 0.8, now=0.0, pose=pose)

    assert local_map.occupancy_at(0.0, 0.9) >= 55
    assert local_map.occupancy_at(0.9, 0.0) == UNKNOWN


def test_values_stay_bounded_under_many_updates(local_map):
    for step in range(200):
        local_map.update(FRONT, SensorStatus.OK, 0.8, now=step * 0.01)

    limit = local_map.config.clamp_log_odds
    assert np.abs(local_map.log_odds).max() <= limit + 1e-6


def test_export_uses_occupancy_grid_conventions(local_map):
    local_map.update(FRONT, SensorStatus.OK, 0.8, now=0.0)
    grid = local_map.to_occupancy_grid()

    assert grid.data.dtype == np.int8
    assert grid.data.max() <= 100
    assert grid.data.min() == UNKNOWN
    assert grid.origin_x == pytest.approx(-1.5)
    assert grid.width == local_map.cells

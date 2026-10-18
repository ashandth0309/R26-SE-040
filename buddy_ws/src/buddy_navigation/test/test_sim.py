"""Tests for the simulation building blocks."""

import math

import pytest

from buddy_navigation.core.commands import MoveCode, MoveRequest
from buddy_navigation.core.geometry import Pose2D
from buddy_navigation.core.presets import (
    simulation_footprint,
    simulation_sensor_specs,
)
from buddy_navigation.core.sensor_model import SensorSpec, SensorStatus
from buddy_navigation.sim.mock_robot import MockRobot
from buddy_navigation.sim.mock_ultrasonic import MockUltrasonicArray
from buddy_navigation.sim.world import Box, World2D, six_by_six_world


def open_room():
    return World2D(4.0, 4.0)


def test_ray_hits_wall_and_box():
    world = open_room()

    assert world.ray_distance(1.0, 2.0, 0.0, 10.0) == pytest.approx(3.0)

    world.add_obstacle(Box(2.0, 1.5, 2.5, 2.5))

    assert world.ray_distance(1.0, 2.0, 0.0, 10.0) == pytest.approx(1.0)
    assert world.ray_distance(1.0, 2.0, math.pi, 10.0) == pytest.approx(1.0)
    assert math.isinf(world.ray_distance(1.0, 2.0, 0.0, 0.5))


def test_ray_parallel_to_box_edge_misses():
    world = open_room()
    world.add_obstacle(Box(2.0, 3.0, 3.0, 3.5))

    assert world.ray_distance(1.0, 2.0, 0.0, 10.0) == pytest.approx(3.0)


def test_wide_beam_sees_off_axis_object_at_its_own_range():
    world = open_room()
    # Object well off the axis but inside the 30 degree cone.
    world.add_obstacle(Box(2.0, 2.3, 2.2, 2.6))
    spec = SensorSpec("s", 0.0, 0.0, 0.0)
    pose = Pose2D(1.0, 2.0, 0.0)

    assert world.beam_distance(pose, spec) < 1.2
    # The same object is outside a narrow cone.
    narrow = SensorSpec("n", 0.0, 0.0, 0.0, fov_rad=math.radians(5))

    assert world.beam_distance(pose, narrow) > 2.5


def test_collision_checks_walls_and_boxes():
    world = open_room()
    world.add_obstacle(Box(2.0, 2.0, 2.5, 2.5))

    assert world.collides(0.05, 1.0, 0.1)
    assert world.collides(1.95, 2.2, 0.1)
    assert not world.collides(1.0, 1.0, 0.1)


def test_six_by_six_world_matches_the_grid():
    world = six_by_six_world(0.5)

    assert world.width_m == pytest.approx(3.0)
    assert len(world.obstacles) == 8


def make_array(world, **kwargs):
    return MockUltrasonicArray(
        world, simulation_sensor_specs(), **kwargs
    )


def test_mock_array_reports_distance_clear_and_too_close():
    world = open_room()
    array = make_array(world)
    pose = Pose2D(1.0, 2.0, 0.0)

    sample = array.measure("front_left", pose, 0.0)

    assert sample.status is SensorStatus.OK
    assert sample.distance_m == pytest.approx(3.0 - 0.13, abs=0.4)

    world.add_obstacle(Box(1.14, 1.9, 1.3, 2.3))

    assert (
        array.measure("front_left", pose, 0.1).status
        is SensorStatus.TOO_CLOSE
    )

    far = Pose2D(0.2, 2.0, math.pi)
    assert (
        array.measure("rear_left", far, 0.2).status
        is SensorStatus.OK
    )


def test_mock_array_triggers_one_sensor_at_a_time_in_order():
    array = make_array(open_room(), trigger_interval_s=0.06)
    pose = Pose2D(1.0, 2.0, 0.0)
    names = []

    for k in range(40):
        sample = array.step(pose, k * 0.01)

        if sample:
            names.append(sample.name)

    assert names[:4] == [
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
    ]
    assert array.min_trigger_gap() >= 0.06 - 1e-9


@pytest.mark.parametrize("mode", ["no_response", "stuck_high"])
def test_faulted_sensor_reports_fault(mode):
    array = make_array(open_room())
    array.inject_fault("front_left", mode)

    sample = array.measure("front_left", Pose2D(1, 2, 0), 0.0)

    assert sample.status is SensorStatus.FAULT


def test_specular_fault_hides_a_real_obstacle():
    world = open_room()
    world.add_obstacle(Box(1.4, 1.5, 1.6, 2.5))
    array = make_array(world)
    pose = Pose2D(1.0, 2.0, 0.0)

    assert array.measure("front_left", pose, 0.0).status is SensorStatus.OK

    array.inject_fault("front_left", "specular")

    assert array.measure("front_left", pose, 0.1).status is SensorStatus.CLEAR

    array.clear_fault("front_left")

    assert array.measure("front_left", pose, 0.2).status is SensorStatus.OK


def test_unknown_fault_requests_are_rejected():
    array = make_array(open_room())

    with pytest.raises(ValueError):
        array.inject_fault("front_left", "gremlins")

    with pytest.raises(ValueError):
        array.inject_fault("nope", "frozen")


def make_robot(world=None, pose=None):
    return MockRobot(
        world or open_room(),
        simulation_footprint(),
        pose or Pose2D(1.0, 2.0, 0.0),
    )


def run(robot, seconds, request=None, dt=0.02, resend=True):
    t = 0.0

    while t < seconds:
        if request is not None and (resend or t == 0.0):
            robot.apply(request, t)

        robot.step(t, dt)
        t += dt


def test_robot_moves_forward_while_commanded():
    robot = make_robot()
    run(robot, 1.0, MoveRequest(MoveCode.FORWARD, 0.5))

    assert robot.pose.x == pytest.approx(1.0 + 0.5 * 0.35, abs=0.02)
    assert robot.pose.y == pytest.approx(2.0)
    assert not robot.collided


def test_robot_stops_when_commands_stop_arriving():
    robot = make_robot()
    request = MoveRequest(MoveCode.FORWARD, 0.5, valid_for_s=0.3)
    run(robot, 2.0, request, resend=False)

    # Command expired after 0.3 s, so travel is about 0.3 s * 0.175 m/s.
    assert robot.pose.x - 1.0 == pytest.approx(0.0525, abs=0.01)
    assert robot.stopped


def test_valid_for_is_capped_by_the_executor_timeout():
    robot = make_robot()
    request = MoveRequest(MoveCode.FORWARD, 1.0, valid_for_s=30.0)
    run(robot, 3.0, request, resend=False)

    assert robot.pose.x - 1.0 == pytest.approx(0.35, abs=0.03)


def test_stop_command_halts_immediately():
    robot = make_robot()
    robot.apply(MoveRequest(MoveCode.FORWARD, 0.5), 0.0)
    robot.step(0.0, 0.02)
    robot.apply(MoveRequest(MoveCode.STOP), 0.02)
    before = robot.pose

    robot.step(0.02, 0.02)

    assert robot.pose == before


def test_rotation_and_reverse():
    robot = make_robot()
    run(robot, 0.5, MoveRequest(MoveCode.ROTATE_LEFT, 0.5))

    assert robot.pose.yaw == pytest.approx(0.5, abs=0.05)

    robot = make_robot()
    run(robot, 1.0, MoveRequest(MoveCode.BACKWARD, 0.5))

    assert robot.pose.x < 1.0


def test_collision_is_detected_and_halts_the_robot():
    world = open_room()
    world.add_obstacle(Box(1.5, 1.0, 1.7, 3.0))
    robot = make_robot(world)
    run(robot, 3.0, MoveRequest(MoveCode.FORWARD, 1.0))

    assert robot.collided
    assert robot.pose.x < 1.5

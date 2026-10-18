"""Closed-loop tests: coordinator + safety gate + mock world."""

import math

import pytest

from buddy_navigation.core.commands import MoveCode
from buddy_navigation.core.coordinator import (
    NavigationConfig,
    NavState,
    PathGoal,
    PrimitiveGoal,
)
from buddy_navigation.core.geometry import Pose2D
from buddy_navigation.sim.harness import SimHarness
from buddy_navigation.sim.world import Box, World2D


def room(**kw):
    return World2D(4.0, 4.0, **kw)


def done(h):
    return h.coordinator.state not in (NavState.RUNNING, NavState.PAUSED)


def warm_up(h, seconds=0.6):
    """Let every sensor report once before a goal starts."""

    h.run(seconds)


def test_motion_is_refused_unless_explicitly_enabled():
    h = SimHarness(room(), Pose2D(1, 2, 0), NavigationConfig())
    warm_up(h)

    assert not h.coordinator.start(PrimitiveGoal(MoveCode.FORWARD, 1.0), h.now)
    assert h.coordinator.state is NavState.FAILED

    h.run(1.0)

    assert h.robot.odometer_m == 0.0


def test_no_motion_before_sensors_have_reported():
    h = SimHarness(room(), Pose2D(1, 2, 0))
    h.coordinator.start(PrimitiveGoal(MoveCode.FORWARD, 1.0), 0.0)
    h.run(0.15)

    assert h.robot.odometer_m == 0.0


def test_timed_forward_motion_completes_in_free_space():
    h = SimHarness(room(), Pose2D(1, 2, 0))
    warm_up(h)
    h.coordinator.start(PrimitiveGoal(MoveCode.FORWARD, 1.0, 0.2), h.now)
    h.run(4.0, until=done)

    assert h.coordinator.state is NavState.SUCCEEDED
    assert 0.03 < h.robot.odometer_m < 0.15
    assert not h.robot.collided


def test_forward_stops_before_a_wall_and_never_collides():
    h = SimHarness(room(), Pose2D(3.0, 2.0, 0), NavigationConfig(
        allow_motion=True, blocked_timeout_s=1.0))
    warm_up(h)
    h.coordinator.start(PrimitiveGoal(MoveCode.FORWARD, 30.0, 0.2), h.now)
    h.run(60.0, until=done)

    assert not h.robot.collided
    assert h.coordinator.state is NavState.FAILED
    assert "Blocked" in h.coordinator.reason
    assert h.min_clearance_m > 0.0


def test_obstacle_appearing_midway_stops_the_robot():
    world = room()
    h = SimHarness(world, Pose2D(0.6, 2.0, 0))
    warm_up(h)
    goal = PathGoal(((3.5, 2.0),))
    h.coordinator.start(goal, h.now)
    h.run(2.0)
    assert h.robot.odometer_m > 0.05

    x = h.robot.pose.x
    world.add_obstacle(Box(x + 0.55, 1.6, x + 0.8, 2.4))
    h.run(10.0)

    assert not h.robot.collided
    assert h.robot.pose.x < x + 0.55


def test_path_following_reaches_waypoints_in_free_space():
    h = SimHarness(room(), Pose2D(1, 1, 0))
    warm_up(h)
    h.coordinator.start(PathGoal(((2.0, 1.0), (2.0, 2.0))), h.now)
    h.run(60.0, until=done)

    assert h.coordinator.state is NavState.SUCCEEDED
    assert math.hypot(h.robot.pose.x - 2.0, h.robot.pose.y - 2.0) < 0.15
    assert not h.robot.collided


def test_path_following_refuses_without_a_trusted_pose():
    h = SimHarness(room(), Pose2D(1, 1, 0), pose_trusted=False)
    warm_up(h)
    h.coordinator.start(PathGoal(((2.0, 1.0),)), h.now)
    h.run(1.0)

    assert h.coordinator.state is NavState.FAILED
    assert "pose" in h.coordinator.reason
    assert h.robot.odometer_m == 0.0


def test_sensor_timeout_stops_the_robot():
    h = SimHarness(room(), Pose2D(0.6, 2.0, 0))
    warm_up(h)
    h.coordinator.start(PathGoal(((3.5, 2.0),)), h.now)
    h.run(2.0)
    moved = h.robot.odometer_m
    assert moved > 0.05

    h.array.inject_fault("front_left", "no_response")
    h.run(3.0)
    stopped_at = h.robot.pose

    h.run(2.0)

    assert h.robot.pose == stopped_at
    assert not h.robot.collided


def test_silent_specular_failure_is_a_known_limit_not_hidden():
    """A sensor reporting CLEAR for a real obstacle is undetectable here.

    The other forward sensor still sees it, and the robot stops with both
    ultrasonic beams overlapping the obstacle; with a single covering
    sensor this would be unsafe.  Documented in the limitations.
    """

    world = room()
    world.add_obstacle(Box(2.0, 1.0, 2.2, 3.0))
    h = SimHarness(world, Pose2D(0.8, 2.0, 0))
    warm_up(h)
    h.array.inject_fault("front_left", "specular")
    h.coordinator.start(PathGoal(((3.5, 2.0),)), h.now)
    h.run(40.0, until=done)

    assert not h.robot.collided


def test_emergency_stop_latches_and_blocks_further_motion():
    h = SimHarness(room(), Pose2D(0.6, 2.0, 0))
    warm_up(h)
    h.coordinator.start(PathGoal(((3.5, 2.0),)), h.now)
    h.run(1.5)
    h.emergency_stop()
    h.run(0.5)
    parked = h.robot.pose
    h.run(2.0)

    assert h.robot.pose == parked
    assert h.coordinator.state is NavState.FAILED
    assert "mergency" in h.coordinator.reason

    # Even a fresh goal cannot move the robot while the latch is set.
    h.coordinator.start(PrimitiveGoal(MoveCode.FORWARD, 2.0), h.now)
    h.run(2.0)

    assert h.robot.pose == parked

    # The supervisor layer refuses on its own, independent of the
    # coordinator (defence in depth).
    from buddy_navigation.core.commands import MoveRequest

    gated = h._gate(MoveRequest(MoveCode.FORWARD, 0.2))

    assert gated.code is MoveCode.STOP
    assert h.rejected


def test_manual_override_takes_priority():
    h = SimHarness(room(), Pose2D(0.6, 2.0, 0))
    warm_up(h)
    h.coordinator.start(PathGoal(((3.5, 2.0),)), h.now)
    h.run(1.0)
    h.manual_override = True
    h.run(0.3)

    assert h.coordinator.state is NavState.FAILED
    assert "anual" in h.coordinator.reason


def test_cancel_pause_and_resume():
    h = SimHarness(room(), Pose2D(0.6, 2.0, 0))
    warm_up(h)
    h.coordinator.start(PathGoal(((3.5, 2.0),)), h.now)
    h.run(1.0)
    h.coordinator.pause()
    h.run(0.3)
    paused_at = h.robot.pose
    h.run(1.0)

    assert h.robot.pose == paused_at

    h.coordinator.resume()
    h.run(1.0)

    assert h.robot.pose.x > paused_at.x

    h.coordinator.cancel()
    h.run(0.3)
    parked = h.robot.pose
    h.run(1.0)

    assert h.coordinator.state is NavState.CANCELLED
    assert h.robot.pose == parked


def test_commands_never_exceed_configured_speed_limits():
    cfg = NavigationConfig(
        allow_motion=True, max_linear_speed=0.2, max_rotation_speed=0.3
    )
    h = SimHarness(room(), Pose2D(1, 1, 0), cfg)
    warm_up(h)
    h.coordinator.start(PathGoal(((2.5, 2.5),)), h.now)
    h.run(30.0, until=done)

    for _, request in h.published:
        rotating = request.code in (
            MoveCode.ROTATE_LEFT, MoveCode.ROTATE_RIGHT)
        assert request.speed <= (0.3 if rotating else 0.2) + 1e-9


def test_invalid_goals_and_config_are_rejected():
    with pytest.raises(ValueError):
        PathGoal(())

    with pytest.raises(ValueError):
        PrimitiveGoal(MoveCode.STOP, 1.0)

    with pytest.raises(ValueError):
        NavigationConfig(max_linear_speed=0.0)


def test_speed_is_reduced_when_an_obstacle_is_in_the_slow_zone():
    world = room()
    world.add_obstacle(Box(1.95, 1.0, 2.2, 3.0))
    h = SimHarness(world, Pose2D(1.2, 2.0, 0))
    warm_up(h)
    h.coordinator.start(PrimitiveGoal(MoveCode.FORWARD, 5.0, 0.2), h.now)
    h.run(1.0)

    forward = [
        r.speed for _, r in h.published
        if r.code is MoveCode.FORWARD
    ]

    assert forward
    assert max(forward) < 0.2
    assert not h.robot.collided

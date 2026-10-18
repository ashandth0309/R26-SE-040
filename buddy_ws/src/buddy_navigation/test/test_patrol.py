"""Patrol manager tests, run closed loop in the simulator."""

import json

import pytest

from buddy_navigation.core.coordinator import NavigationConfig
from buddy_navigation.core.geometry import Pose2D
from buddy_navigation.core.grid import PlanningGrid
from buddy_navigation.core.patrol import (
    PatrolManager,
    PatrolPoint,
    PatrolRoute,
    PatrolState,
)
from buddy_navigation.sim.harness import SimHarness
from buddy_navigation.sim.world import SIX_BY_SIX_ROWS, Box, World2D

CELL = 1.0


def cell_centre(column, row):
    return ((column + 0.5) * CELL, (row + 0.5) * CELL)


def six_world():
    return World2D.from_rows(SIX_BY_SIX_ROWS, CELL)


def six_grid():
    blocked = set()

    for printed, text in enumerate(SIX_BY_SIX_ROWS):
        row = 5 - printed

        for column, char in enumerate(text):
            if char == "#":
                blocked.add((column, row))

    return PlanningGrid.from_blocked_cells(6, 6, blocked, CELL)


def route(*cells, repeat=False, route_id="r1"):
    points = tuple(
        PatrolPoint(f"p{i}", *cell_centre(*c)) for i, c in enumerate(cells)
    )

    return PatrolRoute(route_id, points, repeat)


def make(world=None, start_cell=(0, 0), grid=True, **kw):
    start = cell_centre(*start_cell)
    h = SimHarness(
        world or six_world(),
        Pose2D(start[0], start[1], 0.0),
        NavigationConfig(allow_motion=True, goal_timeout_s=400.0),
        **kw,
    )
    logs = []
    manager = PatrolManager(
        h.coordinator, six_grid() if grid else None, log=logs.append
    )
    h.controller = manager
    h.run(0.6)

    return h, manager, logs


def begin(h, manager, r):
    return manager.start(
        r, h.now, h.robot.pose, h.pose_trusted,
        h.monitor.snapshot(h.now),
    )


def finished(manager):
    return manager.state not in (PatrolState.RUNNING, PatrolState.PAUSED)


def test_patrol_visits_points_in_order_around_obstacles():
    h, manager, logs = make()
    r = route((0, 0), (5, 0), (5, 5))
    assert begin(h, manager, r)
    h.run(900.0, until=lambda _: finished(manager))

    assert manager.state is PatrolState.COMPLETED
    assert not h.robot.collided
    reached = [
        e["target"] for e in manager.events if e["event"] == "waypoint_reached"
    ]
    assert reached == ["p0", "p1", "p2"]
    assert manager.feedback().progress == pytest.approx(1.0)
    assert all(json.loads(line)["route"] == "r1" for line in logs)


def test_feedback_reports_progress():
    h, manager, _ = make()
    begin(h, manager, route((0, 0), (5, 0)))
    h.run(2.0)
    fb = manager.feedback()

    assert fb.state is PatrolState.RUNNING
    assert fb.total_waypoints == 2


def test_patrol_refused_without_trusted_pose():
    h, manager, _ = make(pose_trusted=False)

    assert not begin(h, manager, route((0, 0), (5, 0)))
    assert manager.state is PatrolState.FAILED
    assert "Localisation" in manager.reason
    h.run(1.0)

    assert h.robot.odometer_m == 0.0


def test_patrol_refused_when_a_sensor_is_unhealthy():
    h, manager, _ = make()
    h.array.inject_fault("rear_right", "no_response")
    h.run(1.0)

    assert not begin(h, manager, route((0, 0), (5, 0)))
    assert "not healthy" in manager.reason


def test_unplannable_waypoint_fails_before_moving():
    h, manager, _ = make()
    # Cell (1, 2) is a wall cell in the teaching grid.
    assert begin(h, manager, route((0, 0), (1, 2)))
    h.run(30.0, until=lambda _: finished(manager))

    assert manager.state is PatrolState.FAILED
    assert "Cannot plan" in manager.reason
    assert not h.robot.collided


def test_obstacle_on_the_route_stops_and_reports_failure():
    world = six_world()
    h, manager, _ = make(world)
    begin(h, manager, route((0, 0), (5, 0)))
    h.run(3.0)
    x, y = h.robot.pose.x, h.robot.pose.y
    world.add_obstacle(Box(x + 0.5, y - 0.3, x + 0.75, y + 0.3))
    h.run(40.0, until=lambda _: finished(manager))

    assert manager.state is PatrolState.FAILED
    assert not h.robot.collided
    assert any(e["event"] == "patrol_failed" for e in manager.events)


def test_pause_resume_and_cancel():
    h, manager, _ = make()
    begin(h, manager, route((0, 0), (5, 0)))
    h.run(2.0)
    manager.pause(h.now)
    h.run(0.4)
    parked = h.robot.pose
    h.run(1.0)

    assert h.robot.pose == parked
    assert manager.state is PatrolState.PAUSED

    manager.resume(h.now)
    h.run(1.0)

    assert h.robot.pose != parked

    manager.cancel(h.now)
    h.run(0.4)
    parked = h.robot.pose
    h.run(1.0)

    assert manager.state is PatrolState.CANCELLED
    assert h.robot.pose == parked


def test_estop_during_patrol_stops_and_fails_the_patrol():
    h, manager, _ = make()
    begin(h, manager, route((0, 0), (5, 0)))
    h.run(2.0)
    h.emergency_stop()
    h.run(0.5)
    parked = h.robot.pose
    h.run(1.0)

    assert h.robot.pose == parked
    assert manager.state is PatrolState.FAILED


def test_repeat_routes_complete_laps():
    h, manager, _ = make()
    begin(h, manager, route((0, 0), (1, 0), repeat=True))
    h.run(200.0, until=lambda _: manager.laps >= 2)

    assert manager.laps >= 2
    assert manager.state is PatrolState.RUNNING


def test_route_validation():
    with pytest.raises(ValueError):
        PatrolRoute("", (PatrolPoint("a", 0, 0),))

    with pytest.raises(ValueError):
        PatrolRoute("x", ())

"""NavigationService tests: command handling and publishing policy."""

from pathlib import Path

import pytest

from buddy_navigation.core.commands import MoveCode
from buddy_navigation.core.config import load_navigation_config
from buddy_navigation.core.geometry import Pose2D
from buddy_navigation.core.service import (
    Ack,
    CommandRequest,
    NavigationService,
)
from buddy_navigation.sim.mock_robot import MockRobot
from buddy_navigation.sim.mock_ultrasonic import MockUltrasonicArray
from buddy_navigation.sim.world import World2D, six_by_six_world

CONFIG = Path(__file__).resolve().parents[1] / "config"


class Rig:
    """Service + mock sensors + mock robot on a fixed time step."""

    def __init__(self, config="navigation_sim.yaml", world=None,
                 start=(0.5, 0.5), trusted=True, patch=None):
        self.setup = load_navigation_config(CONFIG / config)

        if patch:
            patch(self.setup)

        self.world = world or six_by_six_world(1.0)
        self.service = NavigationService(self.setup)
        self.robot = MockRobot(
            self.world, self.setup.footprint, Pose2D(*start, 0.0)
        )
        self.array = MockUltrasonicArray(self.world, self.setup.sensors)
        self.trusted = trusted
        self.now = 0.0
        self.published = []
        self.step_until(0.6)

    def step_until(self, end):
        while self.now < end:
            sample = self.array.step(self.robot.pose, self.now)

            if sample:
                self.service.update_sample(sample)

            self.service.set_pose(self.robot.pose, self.trusted)
            request, _ = self.service.tick(self.now)

            if request is not None:
                self.published.append((self.now, request))
                self.robot.apply(request, self.now)

            self.robot.step(self.now, 0.01)
            self.now += 0.01

    def run(self, seconds):
        self.step_until(self.now + seconds)

    def send(self, kind, **kw):
        return self.service.handle(CommandRequest(kind, **kw), self.now)


def test_service_is_silent_when_idle():
    rig = Rig()
    rig.run(2.0)

    assert rig.published == []


def test_timed_forward_then_stop_burst_then_silence():
    rig = Rig(start=(2.5, 5.5), world=World2D(6, 6))
    assert rig.send("move_forward", duration_s=1.0, request_id="a1").accepted
    rig.run(3.0)
    codes = [r.code for _, r in rig.published]

    assert MoveCode.FORWARD in codes
    assert codes[-1] is MoveCode.STOP
    last_time = rig.published[-1][0]
    rig.run(2.0)

    assert rig.published[-1][0] == last_time
    assert rig.service.status(rig.now).state == "succeeded"
    assert rig.service.status(rig.now).request_id == "a1"


def test_motion_commands_are_refused_on_the_physical_default():
    rig = Rig(config="navigation.yaml")
    ack = rig.send("move_forward")

    assert not ack.accepted
    assert "disabled" in ack.reason
    rig.run(1.0)

    assert rig.robot.odometer_m == 0.0
    status = rig.service.status(rig.now)

    assert not status.motion_enabled
    assert status.state == "failed"


def test_go_to_location_reaches_target():
    rig = Rig(start=(0.5, 0.5))
    ack = rig.send("go_to_location", location_name="corner")

    assert ack.accepted

    for _ in range(400):
        rig.run(1.0)

        if rig.service.status(rig.now).state != "running":
            break

    assert not rig.robot.collided
    assert rig.service.status(rig.now).state == "succeeded"
    assert rig.robot.pose.x > 5.0


def test_go_to_location_needs_a_trusted_pose_and_a_known_name():
    rig = Rig(trusted=False)
    ack = rig.send("go_to_location", location_name="corner")

    assert not ack.accepted and "Localisation" in ack.reason

    rig = Rig()

    assert not rig.send("go_to_location", location_name="nowhere").accepted


def test_patrol_by_route_id_with_progress_and_cancel():
    rig = Rig()
    assert rig.send("start_patrol", route_id="demo_loop").accepted
    rig.run(20.0)
    status = rig.service.status(rig.now)

    assert status.state == "running"

    assert rig.send("cancel").accepted
    rig.run(1.0)
    parked = rig.robot.pose
    rig.run(1.0)

    assert rig.robot.pose == parked
    assert rig.service.status(rig.now).state == "cancelled"
    assert not rig.send("start_patrol", route_id="missing").accepted


def test_second_goal_is_rejected_while_one_is_running():
    rig = Rig(start=(2.5, 5.5), world=World2D(6, 6))
    assert rig.send("move_forward", duration_s=2.0).accepted
    ack = rig.send("turn_left")

    assert ack == Ack(False, "A goal is already running; cancel it first.")


def test_external_movement_request_aborts_the_goal():
    rig = Rig(start=(2.5, 5.5), world=World2D(6, 6))
    rig.send("move_forward", duration_s=3.0)
    rig.run(0.5)
    rig.service.note_external_movement(rig.now, "buddy_voice", 1)
    rig.run(0.3)

    assert rig.service.status(rig.now).state == "failed"
    assert "anual" in rig.service.status(rig.now).reason


def test_own_and_stop_requests_do_not_count_as_override():
    rig = Rig(start=(2.5, 5.5), world=World2D(6, 6))
    rig.send("move_forward", duration_s=3.0)
    rig.run(0.5)
    rig.service.note_external_movement(rig.now, "buddy_navigation", 1)
    rig.service.note_external_movement(rig.now, "buddy_voice", 0)
    rig.run(0.3)

    assert rig.service.status(rig.now).state == "running"


def test_estop_flag_is_reported_and_aborts():
    rig = Rig(start=(2.5, 5.5), world=World2D(6, 6))
    rig.send("move_forward", duration_s=3.0)
    rig.run(0.5)
    rig.service.set_estop(True)
    rig.run(0.3)
    status = rig.service.status(rig.now)

    assert status.emergency_stop_active
    assert status.state == "failed"


def test_unknown_command_and_duration_clamping():
    rig = Rig(start=(2.5, 5.5), world=World2D(6, 6))

    assert not rig.send("dance").accepted
    assert rig.send("turn_left", duration_s=999.0).accepted

    goal = rig.service.coordinator._run.goal

    assert goal.duration_s == pytest.approx(5.0)


def test_sensor_loss_is_reported_unhealthy_and_blocks_motion():
    rig = Rig(start=(2.5, 5.5), world=World2D(6, 6))
    rig.array.inject_fault("front_left", "no_response")
    rig.run(1.5)
    rig.send("move_forward", duration_s=2.0)
    rig.run(3.0)

    assert not rig.service.status(rig.now).sensors_healthy
    assert rig.robot.odometer_m == 0.0

"""Logic tests for the ROS node wrappers using strict ROS stubs.

These check topic names, message fields, command translation and the
publishing policy.  They do NOT exercise DDS, QoS, executors or the patrol
action server; those need a real ROS 2 machine.
"""

import importlib
import math
from pathlib import Path

import pytest

import ros_stubs

pytestmark = pytest.mark.skipif(
    not ros_stubs.stubs_available(),
    reason="Real ROS 2 is installed; stub-based tests are not used.",
)

ros_stubs.install(
    extra_messages=[
        "buddy_nav_interfaces/NavigationCommand",
        "buddy_nav_interfaces/NavigationStatus",
    ]
)

CONFIG = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture
def navnode(monkeypatch):
    module = importlib.import_module(
        "buddy_navigation.nodes.navigation_node"
    )
    monkeypatch.setattr(
        module,
        "default_config_path",
        lambda: str(CONFIG / "navigation_sim.yaml"),
    )

    return module, module.NavigationNode()


def range_msg(module, value, node=None):
    msg = module.Range()
    msg.range = value
    msg.min_range = 0.02
    msg.max_range = 4.0

    return msg


def test_node_exposes_the_documented_topics(navnode):
    module, node = navnode

    assert {"/movement/request", "/navigation/status",
            "/navigation/local_map"} <= set(node.publishers)
    topics = set(node.subscriptions)

    for name in ("front_left", "front_right", "rear_left", "rear_right"):
        assert f"/ultrasonic/{name}" in topics

    assert {"/navigation/command", "/safety_state",
            "/movement/request", "/sim/pose"} <= topics


def test_node_publishes_nothing_while_idle(navnode):
    module, node = navnode
    node.clock.advance(1.0)

    for _ in range(20):
        node.clock.advance(0.1)
        node.fire_timers(0.1)

    assert node.publishers["/movement/request"].messages == []


def feed_clear_ranges(module, node):
    for name in ("front_left", "front_right", "rear_left", "rear_right"):
        node.deliver(f"/ultrasonic/{name}", range_msg(module, math.inf))


def test_command_translation_and_movement_request_fields(navnode):
    module, node = navnode
    nav_msgs = importlib.import_module("buddy_nav_interfaces.msg")

    for _ in range(10):
        node.clock.advance(0.1)
        feed_clear_ranges(module, node)

    cmd = nav_msgs.NavigationCommand()
    cmd.command = nav_msgs.NavigationCommand.COMMAND_MOVE_FORWARD
    cmd.duration_s = 1.0
    cmd.request_id = "r-1"
    cmd.source = "test"
    node.deliver("/navigation/command", cmd)

    for _ in range(5):
        node.clock.advance(0.1)
        feed_clear_ranges(module, node)
        node.fire_timers(0.1)

    sent = node.publishers["/movement/request"].messages

    assert sent, "a running goal must publish movement requests"
    first = sent[0]

    assert first.source == "buddy_navigation"
    assert first.command in (
        module.MovementCommand.COMMAND_FORWARD,
        module.MovementCommand.COMMAND_STOP,
    )
    assert 0.0 <= first.linear_speed <= 0.2 + 1e-6
    assert first.valid_for.nanosec == pytest.approx(0.3e9, rel=0.01)
    assert [m.sequence_id for m in sent] == sorted(
        m.sequence_id for m in sent
    )


def test_unknown_command_is_ignored_with_a_warning(navnode):
    module, node = navnode
    nav_msgs = importlib.import_module("buddy_nav_interfaces.msg")
    cmd = nav_msgs.NavigationCommand()
    cmd.command = 99
    node.deliver("/navigation/command", cmd)

    assert any("unknown" in m.lower() for m in node.logger.messages("warning"))


def test_status_message_reflects_the_service(navnode):
    module, node = navnode
    node.clock.advance(1.0)
    node.fire_timers(1.0)
    status = node.publishers["/navigation/status"].messages[-1]

    assert status.state == 0
    assert status.motion_enabled
    assert not status.pose_trusted
    assert not status.sensors_healthy


def test_safety_state_updates_emergency_flag(navnode):
    module, node = navnode
    msg = module.SafetyState()
    msg.emergency_stop_active = True
    node.deliver("/safety_state", msg)
    node.fire_timers(1.0)

    assert node.publishers["/navigation/status"].messages[-1] \
        .emergency_stop_active


def test_local_map_message_layout(navnode):
    module, node = navnode
    feed_clear_ranges(module, node)
    node.fire_timers(0.5)
    grid = node.publishers["/navigation/local_map"].messages[-1]

    assert grid.header.frame_id == "base_link"
    assert grid.info.width * grid.info.height == len(grid.data)
    assert grid.info.resolution == pytest.approx(0.05)
    assert any(v == 0 or v < 50 for v in grid.data if v >= 0)


def test_camera_node_publishes_image_and_diagnostics():
    module = importlib.import_module("buddy_navigation.nodes.camera_node")
    node = module.CameraNode()
    node.clock.advance(1.0)
    node.fire_timers(0.1)
    image = node.publishers["/camera/image_raw"].messages[-1]

    assert (image.width, image.height) == (640, 480)
    assert image.encoding == "bgr8"
    assert image.step == 640 * 3
    assert len(image.data) == 640 * 480 * 3
    assert image.header.frame_id == "camera_link"

    node.fire_timers(1.0)
    array = node.publishers["/diagnostics"].messages[-1]

    assert array.status[0].name == "buddy_camera"
    assert array.status[0].level == 0

    node.destroy_node()

    assert not node.camera.opened


def test_sim_node_matches_the_real_sensor_topic_contract():
    module = importlib.import_module("buddy_navigation.nodes.sim_node")
    node = module.SimNode()

    for _ in range(40):
        node.clock.advance(0.01)
        node.fire_timers(module.DT)

    ranges = {
        n: node.publishers[f"/ultrasonic/{n}"].messages
        for n in ("front_left", "front_right", "rear_left", "rear_right")
    }

    assert all(ranges.values())
    first = ranges["front_left"][-1]

    assert first.header.frame_id == "ultrasonic_front_left_link"
    assert first.range > 0.0

    status = node.publishers["/obstacle_status"].messages[-1]

    assert status.front_valid and status.rear_valid
    assert status.degraded is False

    node.fire_timers(0.05)
    pose = node.publishers["/sim/pose"].messages[-1]

    assert pose.pose.position.x == pytest.approx(0.5)
    assert pose.pose.orientation.w == pytest.approx(1.0)


def test_sim_node_obeys_safe_movement_commands_only():
    module = importlib.import_module("buddy_navigation.nodes.sim_node")
    node = module.SimNode()
    command = module.MovementCommand()
    command.command = module.MovementCommand.COMMAND_FORWARD
    command.linear_speed = 0.5
    command.valid_for.sec = 1
    node.deliver("/movement/safe", command)

    for _ in range(50):
        node.clock.advance(0.01)
        node.fire_timers(module.DT)

    assert node.robot.pose.x > 0.5

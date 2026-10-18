"""Logic tests for the extended ``obstacle_node`` using ROS stubs.

These run the real node class against strict message stubs and the fake echo
GPIO.  They check topic names, message fields, the exact ``/obstacle_status``
semantics the safety supervisor depends on, and the new per-sensor ``Range``
and ``/diagnostics`` outputs.

They do NOT exercise DDS, QoS or the rclpy executor.  Those belong to the
ROS integration tests, which need a ROS 2 machine.
"""

import importlib
import math

import pytest

import ros_stubs

pytestmark = pytest.mark.skipif(
    not ros_stubs.stubs_available(),
    reason="Real ROS 2 is installed; stub-based tests are not used.",
)

ros_stubs.install()

from buddy_core.sensors import ultrasonic as ultrasonic_module  # noqa: E402
from buddy_core.sensors.ultrasonic_array import (  # noqa: E402
    UltrasonicArrayScanner,
)
from ultrasonic_fakes import (  # noqa: E402
    WIRING,
    FakeClock,
    FakeEchoGPIO,
    FakeTimeModule,
)

ECHO = {name: echo for name, (_, echo) in WIRING.items()}


@pytest.fixture
def node_module():
    return importlib.import_module("buddy_core.sensors.obstacle_node")


@pytest.fixture
def rig(monkeypatch, node_module):
    clock = FakeClock()
    monkeypatch.setattr(
        ultrasonic_module,
        "time",
        FakeTimeModule(clock),
    )
    gpio = FakeEchoGPIO(clock, WIRING)
    monkeypatch.setattr(
        node_module,
        "get_gpio_backend",
        lambda config: gpio,
    )

    node = node_module.ObstacleSensorNode()

    # Use the virtual clock for spacing so tests need no real waiting.
    node.scanner = UltrasonicArrayScanner(
        node.sensors,
        order=node_module.SENSOR_ORDER,
        min_trigger_interval_seconds=node.inter_sensor_delay,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )

    class Rig:
        pass

    rig = Rig()
    rig.node = node
    rig.gpio = gpio
    rig.clock = clock
    rig.status = node.publishers["/obstacle_status"]

    def scan(**scenarios):
        for name in node_module.SENSOR_ORDER:
            value = scenarios.get(name, "clear")

            if value in ("clear", "none", "stuck"):
                gpio.set_scenario(ECHO[name], value)
            else:
                gpio.set_scenario(ECHO[name], "distance", value)

        for _ in node_module.SENSOR_ORDER:
            clock.now += node.inter_sensor_delay + 0.001
            node.fire_timers(0.01)

        return rig.status.messages[-1]

    rig.scan = scan

    return rig


def test_node_publishes_expected_topics(rig):
    topics = set(rig.node.publishers)

    assert "/obstacle_status" in topics
    assert "/diagnostics" in topics

    for name in ("front_left", "front_right", "rear_left", "rear_right"):
        assert f"/ultrasonic/{name}" in topics


def test_status_is_only_published_after_a_full_scan(rig, node_module):
    for name in node_module.SENSOR_ORDER[:3]:
        rig.gpio.set_scenario(ECHO[name], "distance", 1.0)

    for _ in range(3):
        rig.clock.now += 0.07
        rig.node.fire_timers(0.01)

    assert rig.status.messages == []


def test_status_uses_pair_minimums_like_the_original_node(rig):
    msg = rig.scan(
        front_left=0.5,
        front_right=0.8,
        rear_left=1.2,
        rear_right=0.6,
    )

    assert msg.front_distance_m == pytest.approx(0.5, abs=0.03)
    assert msg.rear_distance_m == pytest.approx(0.6, abs=0.03)
    assert msg.left_distance_m == pytest.approx(0.5, abs=0.03)
    assert msg.right_distance_m == pytest.approx(0.6, abs=0.03)
    assert msg.front_valid and msg.rear_valid
    assert msg.left_valid and msg.right_valid
    assert not msg.degraded
    assert not (msg.front_blocked or msg.rear_blocked)


def test_blocked_flags_use_the_stop_distance(rig):
    msg = rig.scan(front_left=0.2, front_right=1.0)

    assert msg.front_blocked
    assert msg.left_blocked
    assert not msg.right_blocked


def test_failed_sensor_marks_the_status_degraded(rig):
    msg = rig.scan(front_left="none", front_right=0.8)

    assert msg.degraded
    assert msg.front_valid
    assert msg.front_distance_m == pytest.approx(0.8, abs=0.03)


def test_known_legacy_gap_unmeasured_pair_is_not_reported_blocked(rig):
    """Characterises existing behaviour that navigation must not rely on.

    When both front sensors return no distance (nothing in range, a wiring
    fault, or an obstacle closer than 2 cm) the legacy status is simply
    "invalid" and NOT blocked.  The safety supervisor therefore cannot stop
    forward motion for it.  The navigation stack handles this case itself.
    """

    msg = rig.scan(front_left="none", front_right="none")

    assert not msg.front_valid
    assert not msg.front_blocked
    assert math.isnan(msg.front_distance_m)
    assert msg.degraded


def test_range_messages_follow_rep_117(rig):
    rig.scan(
        front_left=0.7,
        front_right="clear",
        rear_left="none",
        rear_right=0.004,
    )

    ranges = {
        name: rig.node.publishers[f"/ultrasonic/{name}"].messages[-1]
        for name in ("front_left", "front_right", "rear_left", "rear_right")
    }

    assert ranges["front_left"].range == pytest.approx(0.7, abs=0.03)
    assert ranges["front_right"].range == math.inf
    assert math.isnan(ranges["rear_left"].range)
    assert ranges["rear_right"].range == -math.inf

    first = ranges["front_left"]

    assert first.header.frame_id == "ultrasonic_front_left_link"
    assert first.radiation_type == 0
    assert first.min_range == pytest.approx(0.02)
    assert first.max_range == pytest.approx(4.0)
    assert first.field_of_view == pytest.approx(0.2618)


def test_diagnostics_report_faults_and_recovery(rig):
    rig.scan(front_left="none")
    rig.node.fire_timers(1.0)

    array = rig.node.publishers["/diagnostics"].messages[-1]
    by_name = {s.name: s for s in array.status}
    front_left = by_name["buddy_ultrasonic/front_left"]
    front_right = by_name["buddy_ultrasonic/front_right"]

    assert front_left.level == 2
    assert "no_response" in front_left.message
    assert front_right.level == 0
    assert {kv.key for kv in front_left.values} == {
        "last_status",
        "consecutive_faults",
    }

    rig.scan(front_left=1.0)

    assert any(
        "recovered" in line
        for line in rig.node.logger.messages("info")
    )


def test_fault_is_logged_once_not_every_scan(rig):
    rig.scan(front_left="none")
    rig.scan(front_left="none")
    rig.scan(front_left="none")

    faults = [
        line
        for line in rig.node.logger.messages("warning")
        if "front_left" in line and "fault" in line
    ]

    assert len(faults) == 1


def test_destroy_returns_triggers_low(rig):
    rig.node.destroy_node()

    assert rig.node.destroyed

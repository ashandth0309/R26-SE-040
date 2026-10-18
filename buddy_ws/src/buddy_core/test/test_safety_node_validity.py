"""The safety node must pass sensor validity to the supervisor.

Exercises the real ``SafetySupervisorNode`` with strict ROS stand-ins
(no ROS needed) so the regression covers the wiring, not only the policy.
"""

import importlib
import sys

import pytest

import ros_stub_vision as stubs

SRVS = ("EmergencyStop", "ResetEmergencyStop")


class _Moment:
    def __init__(self, ns):
        self.ns = ns

    def to_msg(self):
        return stubs.base.Time(self.ns // 10**9, self.ns % 10**9)

    def __sub__(self, other):
        return _Duration(self.ns - other.ns)


class _Duration:
    def __init__(self, ns):
        self.nanoseconds = ns


class _Clock:
    def __init__(self):
        self.ns = 10**9

    def now(self):
        return _Moment(self.ns)


@pytest.fixture
def node_env():
    msgs = stubs.install(extra_msgs=("ObstacleStatus",), srvs=SRVS)
    sys.modules.pop("buddy_core.safety.safety_node", None)
    module = importlib.import_module("buddy_core.safety.safety_node")
    node = module.SafetySupervisorNode()
    clock = _Clock()
    node.get_clock = lambda: clock
    node.clock = clock
    yield node, msgs
    sys.modules.pop("buddy_core.safety.safety_node", None)
    stubs.uninstall()


def _status(msgs, **fields):
    status = msgs.ObstacleStatus()
    status.front_valid = status.rear_valid = True
    status.left_valid = status.right_valid = True

    for key, value in fields.items():
        setattr(status, key, value)

    return status


def _request(msgs, command):
    request = msgs.MovementCommand()
    request.command = command
    request.source = "test"

    return request


def _safe_commands(node):
    return [m.command for m in node.safe_command_publisher.sent]


def test_forward_is_forwarded_when_front_is_measured_and_clear(node_env):
    node, msgs = node_env
    node._obstacle_callback(_status(msgs))
    node._movement_callback(
        _request(msgs, msgs.MovementCommand.COMMAND_FORWARD))

    assert _safe_commands(node) == [msgs.MovementCommand.COMMAND_FORWARD]


def test_forward_is_refused_when_front_sensors_gave_no_reading(node_env):
    node, msgs = node_env
    node._obstacle_callback(_status(msgs, front_valid=False))
    node._movement_callback(
        _request(msgs, msgs.MovementCommand.COMMAND_FORWARD))

    # Only a STOP may leave the safety node; the movement is not forwarded.
    assert _safe_commands(node) == [msgs.MovementCommand.COMMAND_STOP]


def test_active_motion_stops_when_its_side_becomes_unmeasured(node_env):
    node, msgs = node_env
    node._obstacle_callback(_status(msgs))
    node._movement_callback(
        _request(msgs, msgs.MovementCommand.COMMAND_FORWARD))
    node._obstacle_callback(_status(msgs, front_valid=False))

    assert _safe_commands(node)[-1] == msgs.MovementCommand.COMMAND_STOP

"""Tests for BUDDY's deterministic movement-safety policy."""

import pytest

from buddy_core.safety.supervisor import SafetySupervisor


@pytest.fixture
def supervisor():
    return SafetySupervisor()


def test_stop_is_always_allowed(supervisor):
    decision = supervisor.evaluate(
        SafetySupervisor.COMMAND_STOP,
        obstacle_data_available=False,
    )

    assert decision.allowed is True


def test_forward_allowed_when_front_clear(supervisor):
    decision = supervisor.evaluate(
        SafetySupervisor.COMMAND_FORWARD,
    )

    assert decision.allowed is True


def test_forward_blocked_by_front_obstacle(supervisor):
    decision = supervisor.evaluate(
        SafetySupervisor.COMMAND_FORWARD,
        front_blocked=True,
    )

    assert decision.allowed is False


def test_backward_blocked_by_rear_obstacle(supervisor):
    decision = supervisor.evaluate(
        SafetySupervisor.COMMAND_BACKWARD,
        rear_blocked=True,
    )

    assert decision.allowed is False


@pytest.mark.parametrize(
    "command",
    [
        SafetySupervisor.COMMAND_TURN_LEFT,
        SafetySupervisor.COMMAND_ROTATE_LEFT,
    ],
)
def test_left_movement_blocked_by_left_obstacle(
    supervisor,
    command,
):
    decision = supervisor.evaluate(
        command,
        left_blocked=True,
    )

    assert decision.allowed is False


@pytest.mark.parametrize(
    "command",
    [
        SafetySupervisor.COMMAND_TURN_RIGHT,
        SafetySupervisor.COMMAND_ROTATE_RIGHT,
    ],
)
def test_right_movement_blocked_by_right_obstacle(
    supervisor,
    command,
):
    decision = supervisor.evaluate(
        command,
        right_blocked=True,
    )

    assert decision.allowed is False


def test_missing_obstacle_data_blocks_movement(supervisor):
    decision = supervisor.evaluate(
        SafetySupervisor.COMMAND_FORWARD,
        obstacle_data_available=False,
    )

    assert decision.allowed is False


def test_emergency_stop_blocks_movement(supervisor):
    supervisor.activate_emergency_stop()

    decision = supervisor.evaluate(
        SafetySupervisor.COMMAND_FORWARD,
    )

    assert decision.allowed is False


def test_emergency_stop_does_not_block_stop_command(supervisor):
    supervisor.activate_emergency_stop()

    decision = supervisor.evaluate(
        SafetySupervisor.COMMAND_STOP,
    )

    assert decision.allowed is True


def test_reset_emergency_stop_restores_movement(supervisor):
    supervisor.activate_emergency_stop()
    supervisor.reset_emergency_stop()

    decision = supervisor.evaluate(
        SafetySupervisor.COMMAND_FORWARD,
    )

    assert decision.allowed is True


def test_unknown_command_is_rejected(supervisor):
    decision = supervisor.evaluate(255)

    assert decision.allowed is False

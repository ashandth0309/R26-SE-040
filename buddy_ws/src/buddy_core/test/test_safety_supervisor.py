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


# ---- unknown (unmeasured) sides must not be treated as clear ----------
#
# Regression for the legacy permissive behaviour: the obstacle node sets
# ``*_blocked`` only when the side has a valid reading, so a side whose
# sensors all failed used to look "not blocked" and motion was allowed.


@pytest.mark.parametrize(
    ("command", "kwargs"),
    [
        (SafetySupervisor.COMMAND_FORWARD, {"front_valid": False}),
        (SafetySupervisor.COMMAND_BACKWARD, {"rear_valid": False}),
        (SafetySupervisor.COMMAND_TURN_LEFT, {"left_valid": False}),
        (SafetySupervisor.COMMAND_ROTATE_LEFT, {"left_valid": False}),
        (SafetySupervisor.COMMAND_TURN_RIGHT, {"right_valid": False}),
        (SafetySupervisor.COMMAND_ROTATE_RIGHT, {"right_valid": False}),
    ],
)
def test_motion_toward_unmeasured_side_is_refused(
    supervisor,
    command,
    kwargs,
):
    decision = supervisor.evaluate(command, **kwargs)

    assert decision.allowed is False
    assert "no valid reading" in decision.reason


def test_unmeasured_side_behind_does_not_block_forward(supervisor):
    decision = supervisor.evaluate(
        SafetySupervisor.COMMAND_FORWARD,
        rear_valid=False,
        left_valid=False,
        right_valid=False,
    )

    assert decision.allowed is True


def test_stop_is_allowed_even_with_every_side_unmeasured(supervisor):
    decision = supervisor.evaluate(
        SafetySupervisor.COMMAND_STOP,
        front_valid=False,
        rear_valid=False,
        left_valid=False,
        right_valid=False,
    )

    assert decision.allowed is True


def test_callers_that_omit_validity_keep_the_old_behaviour(supervisor):
    decision = supervisor.evaluate(SafetySupervisor.COMMAND_FORWARD)

    assert decision.allowed is True

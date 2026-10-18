"""The two state machines."""

import pytest

from buddy_orchestrator.states import (
    MOTION_READY,
    MOVING,
    ConversationState,
    RobotState,
    conversation_machine,
    robot_machine,
)


def test_robot_starts_in_starting():
    assert robot_machine().state is RobotState.STARTING


@pytest.mark.parametrize("state", [s for s in RobotState
                                   if s is not RobotState.SHUTTING_DOWN])
def test_emergency_stop_is_reachable_from_every_live_state(state):
    m = robot_machine()
    m.state = state

    if state is RobotState.EMERGENCY_STOP:
        return

    assert m.go(RobotState.EMERGENCY_STOP, "test")


def test_emergency_stop_never_goes_straight_to_motion():
    m = robot_machine()
    m.state = RobotState.EMERGENCY_STOP

    for target in (RobotState.NAVIGATION_ACTIVE, RobotState.PATROL_ACTIVE,
                   RobotState.PAUSED):
        assert not m.go(target, "test")

    assert len(m.violations) == 3


def test_emergency_stop_can_only_clear_to_idle_or_degraded():
    for ok in (RobotState.IDLE, RobotState.DEGRADED_MODE):
        m = robot_machine()
        m.state = RobotState.EMERGENCY_STOP
        assert m.go(ok, "latch cleared")


def test_shutting_down_is_final():
    m = robot_machine()
    m.go(RobotState.SHUTTING_DOWN, "bye")

    for s in RobotState:
        if s is not RobotState.SHUTTING_DOWN:
            assert not m.can(s)


def test_degraded_only_recovers_through_idle():
    m = robot_machine()
    m.state = RobotState.DEGRADED_MODE
    assert not m.can(RobotState.NAVIGATION_ACTIVE)
    assert m.can(RobotState.IDLE)


def test_illegal_transition_is_recorded_not_applied():
    m = robot_machine()
    assert not m.go(RobotState.PATROL_ACTIVE, "skipping ahead")
    assert m.state is RobotState.STARTING
    assert m.violations[-1][1:3] == (RobotState.STARTING,
                                     RobotState.PATROL_ACTIVE)


def test_same_state_is_a_noop():
    m = robot_machine()
    assert m.go(RobotState.STARTING) and not m.violations


def test_motion_sets():
    assert RobotState.IDLE in MOTION_READY
    assert RobotState.EMERGENCY_STOP not in MOTION_READY
    assert RobotState.DEGRADED_MODE not in MOTION_READY
    assert MOVING == {RobotState.NAVIGATION_ACTIVE,
                      RobotState.PATROL_ACTIVE}


def test_conversation_machine_every_state_can_return_to_listening():
    for s in ConversationState:
        m = conversation_machine()
        m.state = s
        assert m.can(ConversationState.LISTENING)


def test_vision_request_cannot_jump_to_processing():
    m = conversation_machine()
    m.state = ConversationState.VISION_REQUEST
    assert not m.can(ConversationState.PROCESSING)

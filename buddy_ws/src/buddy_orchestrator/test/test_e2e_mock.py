"""Acceptance scenarios A-P and the spec's cross-component flows.

Every test here runs the in-process mock system (``mock_system.System``):
real coordinator, voice engine, vision logic and navigation service, with
fake audio, camera, models and motors, and direct calls in place of ROS
topics.  A pass is a MOCK result.  It says nothing about the real
microphones, speaker, camera, ultrasonic sensors, motors, ROS 2 transport
or the Raspberry Pi.  See docs/buddy_testing_matrix.md.
"""

import numpy as np
import pytest
from mock_system import (
    System,
    fall_frames,
    person_detection,
)

from buddy_core.vision.objects import ObjectDetection
from buddy_navigation.core.commands import MoveCode
from buddy_orchestrator.models import VisionEventIn
from buddy_orchestrator.states import RobotState

CUP = ObjectDetection("cup", .9, (200, 150, 400, 350), "other")


def motion_requests(s):
    """Everything that reached the robot except STOP."""

    return [r for _, r in s.moves if r.code is not MoveCode.STOP]


def not_moved(s):
    return s.robot.pose.x == pytest.approx(0.5) and \
        s.robot.pose.y == pytest.approx(0.5) and not motion_requests(s)


# ===== A, B: conversation does not touch the robot ==========================

@pytest.mark.parametrize("text", [
    "hey buddy how are you", "I'm having a terrible day"])
def test_A_B_conversation_gets_a_spoken_reply_and_no_motion(text):
    s = System()
    s.say(text)
    assert s.texts, "Buddy said nothing"
    assert not_moved(s)
    assert s.coord.counters.nav_forwarded == 0
    assert s.crashes == []


# ===== spec section 6: ambiguous language must never move the robot ===========

AMBIGUOUS = [
    "I was thinking about moving forward in life",
    "he said we should go forward with the plan",
    "don't go forward",
    "please do not turn left",
    "the road turns left at the church",
    "my forward progress at work is slow",
    "I might go forward with it later",
    "go forward in time",
]


@pytest.mark.parametrize("text", AMBIGUOUS)
def test_ambiguous_language_does_not_move(text):
    s = System()
    s.say(text)
    s.run(3.0)
    assert not_moved(s), s.moves[:2]
    assert s.coord.counters.nav_forwarded == 0
    assert s.coord.counters.movement_relayed == 0


def test_low_confidence_command_does_not_move():
    s = System()
    s.say("go forward", confidence=0.2)
    s.run(3.0)
    assert not_moved(s)


def test_clear_command_moves_a_bounded_distance():
    s = System()
    s.say("go forward")
    s.run(6.0)
    assert s.robot.pose.x > 0.5
    assert s.robot.pose.x < 2.0
    assert not s.collided
    assert s.nav.status(s.t).state in ("succeeded", "idle")


def test_words_that_contain_stop_always_stop_rather_than_move():
    s = System()
    s.say("I could stop by the shop later")
    s.run(2.0)
    assert not_moved(s)


# ===== D, E: person and object questions =====================================

def test_D_known_person_is_recognised():
    s = System(person="known", objects=[person_detection()])
    s.run(2.0)
    s.say("who is this")
    assert s.spoke("Ann")


def test_D_unknown_person_is_not_named():
    s = System(person="unknown", objects=[person_detection()])
    s.run(2.0)
    s.say("who is this")
    assert s.spoke("don't know who")
    assert not s.spoke("Ann")


def test_D_nobody_in_view():
    s = System()
    s.run(2.0)
    s.say("who is this")
    assert s.spoke("don't see anyone")


def test_E_what_is_this_names_the_detected_object():
    s = System(objects=[CUP])
    s.run(2.0)
    s.say("what is this")
    assert s.spoke("cup")


def test_E_no_vision_module_never_invents_an_object():
    s = System(with_vision=False)
    s.say("what is this")
    assert s.spoke("can't see")
    assert not any(w in " ".join(s.texts).lower()
                   for w in ("cup", "pen", "phone", "bottle"))


def test_E_no_object_in_view_is_honest():
    s = System(objects=[])
    s.run(2.0)
    s.say("what is this")
    text = " ".join(s.texts).lower()
    assert "can't see" in text or "not sure" in text or "hold it" in text


def test_E_camera_off_is_spoken_and_voice_keeps_working():
    s = System(objects=[CUP])
    s.run(2.0)
    s.vision_core.set_camera_enabled(False)
    s.run(2.5)
    s.say("what is this")
    assert s.spoke("camera is off") or s.spoke("can't")
    n = len(s.spoken)
    s.say("hello there")
    assert len(s.spoken) > n, "voice must keep working without the camera"


# ===== F: teaching =========================================================

def teach(s, name="red pen"):
    s.say(f"this is my {name}")
    rng = np.random.default_rng(3)
    frames = []

    for _ in range(12):
        img = rng.integers(90, 200, (480, 640, 3), dtype=np.uint8)
        frames.append(img)
        s.image = img
        s.run(1.0)

    s.say("yes")

    return frames


def test_F_teach_then_recognise_with_the_fake_embedder():
    s = System(objects=[CUP])
    frames = teach(s)
    assert s.spoke("I will remember red pen")
    s.image = frames[0]
    s.run(1.0)
    n = len(s.spoken)
    s.say("what is this")
    assert "red pen" in " ".join(t for _, t, _ in s.spoken[n:])
    s.image = np.full((480, 640, 3), 150, np.uint8)
    s.run(1.0)
    n = len(s.spoken)
    s.say("what is this")
    assert "red pen" not in " ".join(t for _, t, _ in s.spoken[n:])


def test_F_teach_without_a_name_asks_for_one():
    s = System(objects=[CUP])
    s.say("remember this")
    assert s.spoke("call it")


def test_F_teaching_can_be_declined():
    s = System(objects=[CUP])
    s.say("remember this")
    s.say("never mind")
    s.run(2.0)
    assert not s.spoke("I will remember")


# ===== H, O: proactive speech obeys cooldown, quiet mode and conversation ==

def test_H_offer_is_spoken_once_within_the_cooldown():
    s = System()
    for _ in range(3):
        s.coord.on_vision_event(VisionEventIn("proactive_offer", "p1"))
        s.run(1.0)

    s.wait_quiet()
    offers = [t for _, t, _ in s.spoken if "company" in t]
    assert len(offers) == 1
    assert all(p == 3 for _, t, p in s.spoken if "company" in t)


def test_H_no_offer_in_quiet_mode():
    s = System()
    s.say("be quiet")
    s.run(1.5)
    n = len(s.spoken)
    s.coord.on_vision_event(VisionEventIn("proactive_offer", "p1"))
    s.run(2.0)
    assert not [t for _, t, _ in s.spoken[n:] if "company" in t]


def test_O_the_same_person_is_not_greeted_repeatedly():
    s = System()

    for _ in range(4):
        s.coord.on_vision_event(VisionEventIn("greeting_due", "p1"))
        s.run(1.0)

    s.wait_quiet()
    assert len([t for _, t, _ in s.spoken if "Nice to see" in t]) == 1


# ===== I: obstacle avoidance ===============================================

def test_I_obstacle_ahead_is_not_hit():
    s = System()
    s.add_obstacle(1.0, 0.0, 1.3, 1.0)
    s.run(1.0)
    s.say("go forward")
    s.run(8.0)
    assert not s.collided
    assert s.robot.pose.x < 0.95


def test_I_obstacle_appearing_during_a_move_stops_it():
    s = System()
    s.say("go to the corner")
    s.run(3.0)
    x = s.robot.pose.x
    s.add_obstacle(x + 0.45, 0.0, x + 0.8, 1.0)
    s.run(10.0)
    assert not s.collided


def test_I_unsafe_motion_is_spoken_not_silent():
    s = System()
    s.array.inject_fault("front_left", "no_response")
    s.array.inject_fault("front_right", "no_response")
    s.run(3.0)
    s.say("go forward")
    s.run(4.0)
    assert not_moved(s)
    assert s.spoke("sensors") or s.spoke("not moving") or \
        s.spoke("can't")


# ===== J, K: navigation and patrol ==========================================

def test_J_go_to_a_named_destination():
    s = System()
    s.say("go to the corner")
    assert s.spoke("on my way")
    s.run(150.0)
    assert not s.collided
    assert s.robot.pose.x > 5.0
    assert s.nav.status(s.t).state == "succeeded"
    assert s.spoke("I'm here")


def test_J_unknown_destination_is_never_a_command():
    s = System()
    s.say("go to the kitchen")
    s.run(3.0)
    assert not_moved(s)


def test_K_patrol_runs_the_configured_route_and_reports():
    s = System()
    s.say("start patrol")
    s.run(3.0)
    assert s.nav.status(s.t).state == "running"
    assert s.coord.robot.state is RobotState.PATROL_ACTIVE
    s.run(400.0)
    assert not s.collided
    assert s.nav.status(s.t).state == "succeeded"
    assert s.spoke("Patrol finished")
    assert s.coord.robot.state is RobotState.IDLE


def test_K_patrol_without_a_route_is_refused_honestly():
    s = System(profile="offline")
    s.coord.cfg.default_patrol_route = ""
    s.say("start patrol")
    s.run(3.0)
    assert s.spoke("route")
    assert s.nav.status(s.t).state == "idle"


def test_K_patrol_can_be_cancelled():
    s = System()
    s.say("start patrol")
    s.run(5.0)
    s.say("cancel")
    s.run(3.0)
    parked = s.robot.pose
    s.run(3.0)
    assert s.robot.pose == parked
    assert s.nav.status(s.t).state == "cancelled"


def test_untrusted_pose_refuses_goal_navigation_in_words():
    s = System(trusted=False)
    s.run(2.0)
    s.say("go to the corner")
    s.run(3.0)
    assert not_moved(s)
    assert s.spoke("where I am") or s.spoke("couldn't")


# ===== L: stop and emergency stop ==========================================

def test_L_spoken_stop_halts_motion_quickly():
    s = System()
    s.say("go to the corner")
    s.run(4.0)
    assert s.robot.pose.x > 0.6
    s.say("stop")
    s.run(0.6)
    x = s.robot.pose.x
    s.run(2.0)
    assert s.robot.pose.x == pytest.approx(x, abs=1e-6)
    assert s.nav.status(s.t).state in ("cancelled", "idle")


def test_L_stop_works_with_the_coordinator_dead():
    s = System()
    s.say("go to the corner")
    s.run(4.0)
    # A dead coordinator is a topic nobody listens to: messages vanish.
    s.coord.on_voice_navigation = lambda r: False
    s.coord.on_voice_movement = lambda m: False
    s.say("stop")
    s.run(0.8)
    x = s.robot.pose.x
    s.run(2.0)
    assert s.robot.pose.x == pytest.approx(x, abs=1e-6)


def test_L_stop_clears_queued_speech():
    s = System()

    for _ in range(4):
        s.coord._say(__import__("buddy_orchestrator.speech",
                                fromlist=["Kind"]).Kind.REPLY,
                     f"long answer number {_} " * 6)

    s.run(0.2)
    s.say("stop", wait=False)
    s.run(0.2)
    assert s.comp.speaker.queued == 0


def test_L_emergency_stop_latches_announces_and_blocks_new_motion():
    s = System()
    s.say("start patrol")
    s.run(5.0)
    s.press_estop()
    s.run(1.5)
    assert s.coord.robot.state is RobotState.EMERGENCY_STOP
    assert s.spoke("Emergency stop")
    x = s.robot.pose.x
    s.say("go forward")
    s.run(3.0)
    assert s.robot.pose.x == pytest.approx(x, abs=1e-6)
    assert not s.collided
    s.release_estop()
    s.run(2.0)
    assert s.coord.robot.state is RobotState.IDLE
    assert s.nav.status(s.t).state != "running"


# ===== M: no internet =====================================================

def test_M_essential_functions_work_offline():
    s = System(profile="offline", objects=[CUP])
    s.run(2.0)
    assert not s.coord.capabilities.internet
    s.say("hello there")
    assert s.texts
    s.say("what is this")
    assert s.spoke("cup")
    s.say("go forward")
    s.run(4.0)
    assert s.robot.pose.x > 0.5


# ===== N: hardware failure =================================================

def test_N_one_ultrasonic_sensor_failing_blocks_motion_not_voice():
    s = System()
    s.array.inject_fault("front_left", "no_response")
    s.run(4.0)
    s.say("go forward")
    s.run(3.0)
    assert not_moved(s)
    n = len(s.spoken)
    s.say("hello")
    assert len(s.spoken) > n


def test_N_navigation_status_loss_degrades_and_stops():
    s = System()
    s.say("start patrol")
    s.run(5.0)
    assert s.coord.robot.state is RobotState.PATROL_ACTIVE
    s.coord.on_nav_status = lambda *_: None          # node died
    s.run(5.0)
    assert s.coord.robot.state is RobotState.DEGRADED_MODE
    assert s.coord.counters.stops >= 1
    assert s.spoke("lost contact")


def test_N_camera_stops_delivering_frames():
    s = System(objects=[CUP])
    s.run(2.0)
    s.camera_dead = True
    s.run(8.0)
    assert not s.coord.capabilities.camera
    assert any(e.kind == "camera_lost" for e in s.events)
    s.say("what is this")
    assert s.spoke("camera is off") or s.spoke("can't see")
    assert not s.spoke("cup")
    n = len(s.spoken)
    s.say("hello")
    assert len(s.spoken) > n                      # voice still works
    s.say("go forward")
    s.run(4.0)
    assert s.robot.pose.x > 0.5                   # navigation still works


# ===== P: possible fall ====================================================

def run_to_fall(s, limit=120.0):
    end = s.t + limit

    while s.t < end:
        s.run(1.0)

        if any(e.kind == "fall_suspected" for e in s.events):
            return True

    return False


def test_P_possible_fall_asks_and_never_moves_the_robot():
    s = System(person="unknown", objects=[person_detection()],
               pose_frames=fall_frames())
    assert run_to_fall(s), "mock fall never detected"
    s.wait_quiet()
    assert s.spoke("Are you okay")
    assert not_moved(s)
    assert s.coord.counters.nav_forwarded == 0


def test_P_unanswered_fall_does_not_start_any_movement():
    s = System(person="unknown", objects=[person_detection()],
               pose_frames=fall_frames())
    run_to_fall(s)
    s.run(60.0)
    assert not_moved(s)
    assert s.coord.counters.nav_forwarded == 0
    assert s.coord.counters.movement_relayed == 0
    assert s.coord.robot.state is RobotState.IDLE


# ===== cross-component contracts (level 3) =================================

def test_navigation_refusal_returns_spoken_status():
    s = System()
    s.press_estop()
    s.run(1.5)
    s.say("go to the corner")
    assert s.spoke("Emergency stop") or s.spoke("can't move")


def test_voice_request_reaches_vision_and_result_is_spoken():
    s = System(objects=[CUP])
    s.run(2.0)
    s.say("what is this")
    assert s.comp.engine.gateway_claims == 1
    assert s.coord.counters.utterances_claimed == 1


def test_ordinary_chat_is_not_claimed_by_the_coordinator():
    s = System()
    s.say("tell me about your day")
    assert s.comp.engine.gateway_claims == 0


def test_only_the_coordinator_reaches_the_robot_besides_stop():
    s = System()
    s.say("go forward")
    s.run(3.0)
    assert s.coord.counters.nav_forwarded == 1


def test_no_state_machine_violations_in_a_busy_session():
    s = System(objects=[CUP])
    s.say("start patrol")
    s.run(5.0)
    s.press_estop()
    s.run(2.0)
    s.release_estop()
    s.run(2.0)
    s.say("what is this")
    s.say("go forward")
    s.run(5.0)
    assert s.coord.status()["state_violations"] == 0
    assert s.crashes == []

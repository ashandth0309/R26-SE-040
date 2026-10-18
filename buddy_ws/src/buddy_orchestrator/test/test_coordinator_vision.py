"""Utterance routing, vision events, approach refusal."""

import pytest
from conftest import Rig, known

from buddy_orchestrator.contracts import Err
from buddy_orchestrator.models import PersonIn, VisionEventIn
from buddy_orchestrator.ports import VisionCommandResult


def test_what_is_this_uses_vision_result(rig):
    a = rig.c.route_utterance("what is this?")
    assert a.claimed and a.reply == "I think that's a cup."
    assert rig.vision.calls == [("identify",)]


def test_unknown_object_offer_text_is_passed_through(rig):
    rig.vision.identify_result = {
        "success": True,
        "text": "I'm not sure what this is. Would you like to teach me?"}
    a = rig.c.route_utterance("what's this")
    assert "teach me" in a.reply


def test_camera_question_never_falls_to_the_model_when_vision_down():
    rig = Rig(vision=False)
    rig.healthy()
    a = rig.c.route_utterance("what is this?")
    assert a.claimed and a.error is Err.UNAVAILABLE
    assert a.reply == "I can't see right now."


def test_camera_off_is_said_honestly(rig):
    rig.vision_up(camera=False)
    a = rig.c.route_utterance("what is this?")
    assert a.claimed and "camera is off" in a.reply
    assert rig.vision.calls == []


def test_vision_service_failure_is_claimed_not_invented(rig):
    rig.vision.raises = ConnectionError("gone")
    a = rig.c.route_utterance("what is this")
    assert a.claimed and a.error is Err.UNAVAILABLE
    assert "can't see" in a.reply


def test_vision_timeout_asks_to_retry(rig):
    rig.vision.raises = TimeoutError()
    a = rig.c.route_utterance("what is this")
    assert a.claimed and a.error is Err.TIMEOUT
    assert "try again" in a.reply


def test_identify_unsuccessful_is_honest(rig):
    rig.vision.identify_result = {"success": False, "text": ""}
    a = rig.c.route_utterance("what is this")
    assert a.claimed and a.reply == "I can't see right now."


def test_teach_with_name(rig):
    a = rig.c.route_utterance("this is my red pen")
    assert a.claimed and a.reply == "Got it."
    assert rig.vision.calls[0][0:2] == ("teach", "start")
    assert "red pen" in rig.vision.calls[0][2]


def test_teach_without_name_asks_then_uses_next_utterance(rig):
    a = rig.c.route_utterance("remember this")
    assert a.claimed and "call it" in a.reply
    assert rig.vision.calls == []
    a = rig.c.route_utterance("blue mug")
    assert a.claimed and a.reply == "Got it."
    assert rig.vision.calls[-1] == ("teach", "start", "blue mug")


def test_teach_name_can_be_cancelled(rig):
    rig.c.route_utterance("remember this")
    a = rig.c.route_utterance("never mind")
    assert a.claimed and rig.vision.calls == []


def test_teach_name_expires(rig):
    rig.c.route_utterance("remember this")
    rig.t += 60
    a = rig.c.route_utterance("tell me a joke")
    assert not a.claimed


def test_teach_refused_by_vision_is_reported(rig):
    rig.vision.teach_result = {"ok": False, "reply": "I can't save that.",
                               "phase": "", "saved": False}
    a = rig.c.route_utterance("this is my key")
    assert a.reply == "I can't save that."
    assert a.error is Err.REFUSED_CONSENT


def test_where_is(rig):
    a = rig.c.route_utterance("where did you last see my book")
    assert a.claimed and "desk" in a.reply
    assert rig.vision.calls[-1][0] == "where_is"


@pytest.mark.parametrize("text", [
    "I was thinking about moving forward in life",
    "tell me a joke",
    "how are you today",
    "what is the capital of France",
    "I love this song",
    "the weather is nice",
])
def test_ordinary_talk_is_not_claimed(rig, text):
    assert not rig.c.route_utterance(text).claimed
    assert rig.vision.calls == [("command", text)] or \
        rig.vision.calls == []


def test_vision_settings_command_reply_is_claimed(rig):
    rig.vision.command_result = VisionCommandResult(
        True, "Okay, camera off.", False)
    a = rig.c.route_utterance("turn off the camera")
    assert a.claimed and a.reply == "Okay, camera off."


def test_vision_command_pass_to_dialogue_is_not_claimed(rig):
    rig.vision.command_result = VisionCommandResult(True, "Sure.", True)
    assert not rig.c.route_utterance("do you like cats").claimed


def test_route_error_fails_open(rig):
    rig.c._route = lambda *_: 1 / 0
    a = rig.c.route_utterance("hello")
    assert not a.claimed


def test_who_is_this_known_single(rig):
    rig.people(known("Yohan"))
    a = rig.c.route_utterance("who is this")
    assert a.reply == "I think that's Yohan."


def test_who_is_this_unknown(rig):
    rig.people(known(known=False, identity_state="unknown"))
    a = rig.c.route_utterance("who is this")
    assert "don't know who" in a.reply


def test_who_is_this_does_not_name_among_several(rig):
    rig.people(known("A"), known("B", person_key="p2"))
    a = rig.c.route_utterance("who is this")
    assert "A" not in a.reply.replace("a", "") or "won't name" in a.reply
    assert "won't name" in a.reply


def test_who_is_this_with_stale_people_does_not_guess(rig):
    rig.people(known("Yohan"))
    rig.t += 30
    rig.healthy()
    a = rig.c.route_utterance("who is this")
    assert a.error is Err.STALE and "Yohan" not in a.reply


def test_check_floor_reports_lying_person_as_question_not_alarm(rig):
    rig.people(known(posture="lying"))
    a = rig.c.route_utterance("is anyone on the floor")
    assert "Are you okay" in a.reply
    assert rig.stops == [] and rig.nav == []


def test_intro_asks_for_consent(rig):
    a = rig.c.route_utterance("this is my friend Sam")
    assert a.claimed
    assert rig.vision.calls == []


# ----- approach -----

def test_come_here_refused_on_hardware_even_if_everything_else_ok(rig):
    rig.people(known())
    a = rig.c.route_utterance("come here")
    assert a.claimed and a.error is Err.REFUSED_SAFETY
    assert rig.nav == [] and rig.stops == []
    assert rig.c.approach.active is None


def test_come_here_unknown_person_refused(sim_rig):
    sim_rig.people(known(known=False, identity_state="unknown"))
    a = sim_rig.c.route_utterance("come here")
    assert "not sure who" in a.reply


def test_come_here_nobody_visible(sim_rig):
    a = sim_rig.c.route_utterance("come here")
    assert a.error is Err.UNAVAILABLE


def test_come_here_simulated_succeeds_without_any_command(sim_rig):
    sim_rig.people(known())
    a = sim_rig.c.route_utterance("come here")
    assert a.reply == "Okay, coming over."
    assert sim_rig.c.approach.active is not None
    for _ in range(40):
        sim_rig.advance(0.2)
        sim_rig.healthy()
        sim_rig.people(known())
    assert sim_rig.c.approach.active is None
    assert any("simulated" in t for t in sim_rig.texts)
    assert sim_rig.nav == [] and sim_rig.stops == []


def test_estop_cancels_simulated_approach(sim_rig):
    sim_rig.people(known())
    sim_rig.c.route_utterance("come here")
    sim_rig.healthy(estop=True)
    assert sim_rig.c.approach.active is None


def test_quiet_mode_refuses_approach(sim_rig):
    sim_rig.c.on_voice_status({"silence_mode": "quiet"})
    sim_rig.people(known())
    assert sim_rig.c.route_utterance("come here").error is \
        Err.REFUSED_SAFETY


# ----- events -----

def test_fall_suspected_asks_and_never_moves(rig):
    rig.c.on_vision_event(VisionEventIn(
        "fall_suspected", "p1", "warning", "", True))
    assert any("Are you okay" in t for t in rig.texts)
    assert rig.nav == [] and rig.stops == [] and rig.relayed == []


def test_fall_alert_without_voice_check_is_still_a_question(rig):
    rig.c.on_vision_event(VisionEventIn("fall_alert", "p1", "critical",
                                        "", False))
    assert rig.nav == [] and rig.relayed == []


def test_greeting_is_proactive_and_gated_by_quiet(rig):
    rig.c.on_voice_status({"silence_mode": "quiet"})
    rig.c.on_vision_event(VisionEventIn("greeting_due", "p1"))
    assert rig.said == []


def test_proactive_not_while_user_speaking(rig):
    rig.c.on_voice_status({"state": "USER_SPEAKING"})
    rig.c.on_vision_event(VisionEventIn("proactive_offer", "p1"))
    assert rig.said == []


def test_proactive_suppressed_under_estop(rig):
    rig.healthy(estop=True)
    rig.said.clear()
    rig.c.on_vision_event(VisionEventIn("greeting_due", "p1"))
    assert rig.said == []


def test_camera_lost_event_disables_camera_features(rig):
    rig.c.on_vision_event(VisionEventIn("camera_lost"))
    assert not rig.c.capabilities.camera
    a = rig.c.route_utterance("what is this")
    assert "camera is off" in a.reply
    rig.c.on_vision_event(VisionEventIn("camera_restored"))
    assert rig.c.capabilities.camera


def test_unknown_event_kind_ignored(rig):
    rig.c.on_vision_event(VisionEventIn("something_new"))
    assert rig.said == []


def test_dataclass_person_fields():
    p = PersonIn("k")
    assert p.bbox_width == 0.0

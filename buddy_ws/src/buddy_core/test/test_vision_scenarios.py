"""Phase 22: companion scenarios A to H, driven through VisionNodeCore.

Real data flow with fakes at the perception edge (faces, expression,
objects, pose are scripted).  These prove the policy and wiring.  They say
nothing about camera accuracy or how a real person looks to the models.
"""

import math

import numpy as np
from test_vision_context import (
    IMG, FakeExpr, FakeFaces, FakeObjects, FakePose, person_store)
from test_vision_gestures import arms
from test_vision_node_core import core_for, fall_core, frame, run_until
from test_vision_object_teach import det as obj_det
from test_vision_object_teach import scene

from buddy_core.vision.proactive import (
    OFFERS, ProactiveConfig, ProactiveInteractionManager)
from buddy_core.vision.vision_settings import VisionSettings

FEELING_WORDS = ("sad", "angry", "upset", "depressed", "aggressive", "mad",
                 "furious", "unhappy")


def opted_in_core(label, conf=0.85, **kw):
    store, pid = person_store()
    store.set_setting(pid, "proactive", True)
    pm = ProactiveInteractionManager(ProactiveConfig(sustain_s=5.0))
    core, c = core_for(FakeFaces(pid), store, expr=FakeExpr(label, conf),
                       proactive=pm, **kw)

    return core, c, store, pid, pm


def all_event_text(events):
    return " ".join(e["suggested_text"] for e in events).lower()


# ----- A: someone appears sad -----

def test_a_sustained_subdued_face_leads_to_a_question_then_conversation():
    core, c, _s, _pid, pm = opted_in_core("sad")
    out, offer = run_until(core, c, "proactive_offer", n=120)
    assert offer, "an opted-in person looking subdued for a while"
    assert offer["suggested_text"] in OFFERS
    assert offer["suggested_text"].endswith("?")
    assert not any(w in offer["suggested_text"].lower().split()
                   for w in FEELING_WORDS)
    assert core.awaiting_reply()
    r = core.command("Yes, today was difficult")
    # the voice engine must also answer what they said, naturally
    assert r.handled and r.pass_to_dialogue and r.reply == ""
    assert pm.state.value == "conversation_active"
    kinds = []

    for _ in range(60):
        kinds += [e["kind"] for e in frame(core, c).events]

    assert "proactive_offer" not in kinds       # no second offer mid-chat


def test_a_bare_yes_gets_a_short_acknowledgement():
    core, c, *_ = opted_in_core("sad")
    run_until(core, c, "proactive_offer", n=120)
    r = core.command("yes")
    assert r.handled and r.reply and not r.pass_to_dialogue


def test_a_no_consent_no_offer():
    store, pid = person_store()                  # never opted in
    pm = ProactiveInteractionManager(ProactiveConfig(sustain_s=5.0))
    core, c = core_for(FakeFaces(pid), store, expr=FakeExpr("sad", 0.9),
                       proactive=pm)

    for _ in range(120):
        assert not [e for e in frame(core, c).events
                    if e["kind"] == "proactive_offer"]


# ----- B: someone appears angry -----

def test_b_frustration_never_labels_the_person_and_asks_gently():
    core, c, *_ = opted_in_core("angry", 0.9)
    texts, events = [], []

    for _ in range(120):
        out = frame(core, c)
        events += out.events

        if any(e["kind"] == "proactive_offer" for e in out.events):
            break

    offer = [e for e in events if e["kind"] == "proactive_offer"]
    assert offer, "negative group with agreement may lead to a question"
    texts = all_event_text(events)
    assert not any(w in texts for w in FEELING_WORDS)
    assert offer[0]["suggested_text"].endswith("?")
    # what the user then says is not a yes/no: vision steps aside
    r = core.command("I'm just frustrated")
    assert not r.handled and r.pass_to_dialogue


def test_b_the_voice_hint_is_a_coarse_group_never_an_insult():
    store, pid = person_store()
    store.set_setting(pid, "expression", True)
    core, c = core_for(FakeFaces(pid), store, expr=FakeExpr("angry", 0.9))
    words = set()

    for _ in range(40):
        words |= {e["expression"] for e in frame(core, c).emotions}

    assert words <= {"negative", "unknown"}, words


# ----- C: a happy person waves -----

def test_c_known_person_smiles_and_waves_one_greeting_not_one_per_frame():
    store, pid = person_store()
    store.set_setting(pid, "expression", True)
    wave = [arms((260, 290), (330 + 45 * math.sin(i * 1.2), 120))
            for i in range(400)]
    person = obj_det("person", (100, 50, 560, 470))
    core, c = core_for(
        FakeFaces(pid), store, expr=FakeExpr("happy", 0.9),
        objects=FakeObjects(person), pose=FakePose(wave),
        settings=VisionSettings(profile="enhanced",
                                overrides={"pose": 0.1, "objects": 0.1}))
    greets, waves, happy = [], [], False

    for _ in range(150):
        out = frame(core, c, dt=0.1)
        greets += [e for e in out.events if e["kind"] == "greeting_due"]
        waves += [e for e in out.events if e["kind"] == "gesture"]
        happy |= any(m["expression"] == "happy" for m in out.emotions)

    assert len(greets) == 1 and "Ann" in greets[0]["suggested_text"]
    assert waves and json_gesture(waves[0]) == "wave"
    assert happy


def json_gesture(event):
    import json

    return json.loads(event["data_json"])["gesture"]


# ----- D: learning an object -----

def teach_pen(core, c):
    first = core.teach_object("start", "my pen")
    assert first["ok"]
    colour = (30, 160, 220)
    reply = None

    for i in range(120):
        c.t += 1.0
        out = core.on_frame(scene(colour, seed=i))

        for e in (out.events if out else ()):
            if e["kind"] == "teach_prompt":
                reply = e

        if reply and "save" in reply["suggested_text"].lower():
            break

    return reply


def test_d_teach_pen_stores_only_after_yes_then_recognises_with_confidence():
    core, c = core_for(FakeFaces(), objects=FakeObjects(obj_det("cup")))
    prompt = teach_pen(core, c)
    assert prompt, "the session should ask before saving"
    assert core.b.objects.list() == [], "nothing stored before yes"
    done = core.teach_object("answer", "yes")
    assert done["saved"] and [o["name"] for o in core.b.objects.list()] == \
        ["pen"]
    # later: a fresh view of the same pen
    c.t += 5
    core.on_frame(scene((30, 160, 220), seed=999))
    r = core.identify_object()
    assert r["success"] and r["kind"] == "taught" and r["name"] == "pen"
    assert 0.5 < r["confidence"] <= 1.0
    assert "percent" in r["text"] and r["text"].startswith("I think")
    # a different looking object is not called the pen
    c.t += 5
    core.on_frame(scene((200, 20, 30), seed=5))
    other = core.identify_object()
    assert other["name"] != "pen"


def test_d_declining_the_save_stores_nothing():
    core, c = core_for(FakeFaces(), objects=FakeObjects(obj_det("cup")))
    assert teach_pen(core, c)
    core.teach_object("answer", "no")
    assert core.b.objects.list() == []


# ----- E: possible fall -----

def test_e_fall_asks_first_and_only_alerts_without_an_answer():
    core, c = fall_core()
    out, ev = run_until(core, c, "fall_suspected")
    assert ev["suggested_text"].lower().startswith("are you okay")
    assert out.security == []                    # asking is not alerting
    r = core.command("I'm fine, thanks")
    assert r.handled and not r.pass_to_dialogue
    kinds = []

    for _ in range(80):
        kinds += [e["kind"] for e in frame(core, c).events]

    assert "fall_alert" not in kinds


def test_e_silence_escalates_through_the_configured_security_event():
    core, c = fall_core()
    run_until(core, c, "fall_suspected")
    out, ev = run_until(core, c, "fall_alert", n=100)
    assert ev and len(out.security) == 1
    assert out.security[0]["event_type"] == 4


# ----- F: the user declines -----

def test_f_decline_is_respected_for_the_whole_cooldown():
    core, c, _s, _pid, pm = opted_in_core("sad")
    run_until(core, c, "proactive_offer", n=120)
    r = core.command("No.")
    assert r.handled and not r.pass_to_dialogue
    later = 0

    for _ in range(540):                         # 90 minutes at 10 s
        later += sum(1 for e in frame(core, c, dt=10.0).events
                     if e["kind"] == "proactive_offer")

    assert later == 0, "declined: no asking within the 2 h cooldown"


def test_f_stop_asking_is_remembered_for_a_long_time():
    core, c, _s, _pid, pm = opted_in_core("sad")
    run_until(core, c, "proactive_offer", n=120)
    r = core.command("Please stop asking me that")
    assert r.handled and "won't" in r.reply.lower()
    n = 0

    for _ in range(600):                         # about 4 hours at 25 s
        n += sum(1 for e in frame(core, c, dt=25.0).events
                 if e["kind"] == "proactive_offer")

    assert n == 0


# ----- G: unknown visitor -----

def test_g_unknown_visitor_is_not_named_enrolled_or_read(tmp_path,
                                                         monkeypatch):
    monkeypatch.chdir(tmp_path)
    store, pid = person_store()
    before = {p["person_id"] for p in store.list_people()}
    expr = FakeExpr("sad", 0.95)
    pm = ProactiveInteractionManager(ProactiveConfig(sustain_s=5.0))
    core, c = core_for(FakeFaces(None), store, expr=expr, proactive=pm)
    kinds, names = [], []

    for _ in range(150):
        out = frame(core, c)
        kinds += [e["kind"] for e in out.events]
        names += [p["display_name"] for p in out.people]

        assert out.emotions == []                # no expression read
        assert all(not p["known"] for p in out.people)

    assert not set(names) - {""}
    assert "proactive_offer" not in kinds and "greeting_due" not in kinds
    assert {p["person_id"] for p in store.list_people()} == before
    assert expr.calls == 0, "expression model never ran on a stranger"
    assert list(tmp_path.iterdir()) == [], "nothing written to disk"


# ----- H: camera failure -----

def test_h_camera_failure_clears_state_and_vision_actions_stop():
    store, pid = person_store()
    store.set_setting(pid, "expression", True)
    core, c = core_for(FakeFaces(pid), store, expr=FakeExpr("happy", 0.9))

    for _ in range(12):
        frame(core, c)

    assert core.status()["people_in_view"] == 1 and core.camera_ok
    c.t += 5.0                                    # no frames arrive
    core.check_camera()
    d = core.take_deferred()
    kinds = [e["kind"] for e in d.events]
    assert kinds == ["camera_lost"] and not core.camera_ok
    assert [m["expression"] for m in d.emotions] == ["unknown"]
    assert d.events[0]["severity"] == "warn"
    core.check_camera()                           # reported once only
    assert core.take_deferred().events == []
    assert core.identify_object()["kind"] == "unavailable"
    assert not core.awaiting_reply()
    assert core.status()["camera_ok"] is False


def test_h_other_functions_keep_working_and_camera_can_recover():
    core, c = core_for(FakeFaces(None))
    frame(core, c)
    c.t += 5.0
    core.check_camera()
    core.take_deferred()
    # spoken settings and replies do not need the camera
    r = core.command("what do you know about me")
    assert r.handled
    assert core.command("hello there") == (False, "", True)
    out = frame(core, c)
    assert out is not None and core.camera_ok
    assert [e["kind"] for e in core.take_deferred().events] == \
        ["camera_restored"]


def test_h_user_switching_the_camera_off_is_not_reported_as_failure():
    core, c = core_for(FakeFaces(None))
    frame(core, c)
    core.set_camera_enabled(False)
    c.t += 30.0
    core.check_camera()
    assert core.camera_ok and core.take_deferred().events == []


def test_h_vision_node_has_no_voice_or_motion_dependency():
    import buddy_core.vision.vision_node as vn

    src = open(vn.__file__).read()
    assert "buddy_core.voice" not in src
    assert "MovementCommand" not in src and "cmd_vel" not in src
    assert IMG is not None and np is not None

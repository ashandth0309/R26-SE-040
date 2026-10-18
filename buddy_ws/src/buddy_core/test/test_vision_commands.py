"""Phases 18-19: spoken settings and privacy commands."""

import time

import pytest

from buddy_core.vision.face_store import Consent, FaceProfileStore
from buddy_core.vision.object_memory import ObjectMemory
from buddy_core.vision.proactive import ProactiveInteractionManager
from buddy_core.vision.vision_commands import (
    Command, VisionController, parse_command)
from buddy_core.vision.vision_settings import VisionSettings


def setup():
    store = FaceProfileStore()
    pid = store.add_person("Ann", Consent(time.time(), "spoken_yes", True))
    applied = []
    mem = ObjectMemory()
    ctl = VisionController(VisionSettings(), applied.append, store, mem,
                           ProactiveInteractionManager())

    return ctl, store, pid, applied, mem


@pytest.mark.parametrize("text,kind,value", [
    ("don't greet me", "greet", False),
    ("please stop greeting me", "greet", False),
    ("greet me", "greet", True),
    ("stop checking on me", "proactive", False),
    ("you can check on me", "proactive", True),
    ("stop reading my expression", "expression", False),
    ("you can read my facial expression", "expression", True),
    ("don't recognise me", "recognition", False),
    ("what do you know about me", "about_me", None),
    ("forget my face", "forget_me", None),
    ("delete everything about me", "forget_me", None),
    ("I'd like some quiet", "quiet", True),
    ("turn off the camera", "camera", False),
    ("close your eyes", "camera", False),
    ("turn on the camera", "camera", True),
    ("remember where things are", "object_memory", True),
    ("forget where things are", "object_memory", False),
    ("stop remembering where things are", "object_memory", False),
    ("turn off fall detection", "fall_detection", False),
    ("switch to lightweight", "profile", "lightweight"),
    ("enhanced mode", "profile", "enhanced"),
    ("stop recognising faces", "global_recognition", False),
])
def test_parse(text, kind, value):
    c = parse_command(text)
    assert c is not None and c.kind == kind and c.value == value


@pytest.mark.parametrize("text", [
    "what time is it", "tell me a joke", "move forward", "", "stop",
    "I like the camera", "greetings from the kitchen"])
def test_unrelated_speech_is_ignored(text):
    assert parse_command(text) is None


def test_greeting_text_is_cleaned_and_limited():
    c = parse_command("greet me with Hello Captain!")
    assert c == Command("greet_me".replace("greet_me", "greeting_text"),
                        "hello captain", "person")
    assert parse_command("greet me with http://x.y/z").kind == \
        "greeting_text_bad"
    assert parse_command("greet me with " + "a" * 60).kind == \
        "greeting_text_bad"


def test_max_checkins_parsing():
    assert parse_command("limit check-ins to two a day").value == 2
    assert parse_command("only 3 times a day").value == 3
    assert parse_command("at most none a day").value == 0


def test_person_settings_apply_to_the_recognised_person():
    ctl, store, pid, applied, mem = setup()
    assert "won't greet" in ctl.handle("don't greet me", pid, 0.0)
    assert store.get_person(pid)["settings"]["greet"] is False
    assert "check in" in ctl.handle("you can check on me", pid, 1.0)
    st = store.get_person(pid)["settings"]
    assert st["proactive"] is True and st["expression"] is True
    ctl.handle("stop reading my expression", pid, 2.0)
    assert store.get_person(pid)["settings"]["expression"] is False
    ctl.handle("limit check-ins to one a day", pid, 3.0)
    assert store.get_person(pid)["settings"]["max_checkins_per_day"] == 1
    ctl.handle("greet me with Hello Captain", pid, 4.0)
    assert store.get_person(pid)["settings"]["preferred_greeting"] == \
        "hello captain"


def test_person_commands_need_a_recognised_person():
    ctl, store, pid, applied, mem = setup()
    r = ctl.handle("don't greet me", None, 0.0)
    assert "don't recognise you" in r
    assert store.get_person(pid)["settings"]["greet"] is True
    assert "don't recognise you" in ctl.handle("forget my face", None, 0.0)
    assert "don't recognise you" in ctl.handle("forget my face", "p_ghost",
                                               0.0)


def test_forget_me_needs_a_yes_then_deletes_everything():
    ctl, store, pid, applied, mem = setup()
    store.kv_set(pid, "cooldown_until", 5.0)
    q = ctl.handle("forget my face", pid, 0.0)
    assert "go ahead" in q and ctl.awaiting_confirmation
    assert store.get_person(pid) is not None
    r = ctl.handle("yes", pid, 2.0)
    assert "deleted" in r and store.get_person(pid) is None
    assert store.kv_get(pid, "cooldown_until", None) is None


def test_forget_me_declined_or_timed_out_keeps_data():
    ctl, store, pid, applied, mem = setup()
    ctl.handle("forget my face", pid, 0.0)
    assert "won't" in ctl.handle("no", pid, 1.0)
    assert store.get_person(pid) is not None
    ctl.handle("forget my face", pid, 10.0)
    assert ctl.handle("yes", pid, 100.0) is None      # timed out
    assert store.get_person(pid) is not None


def test_camera_global_settings_and_apply_callback():
    ctl, store, pid, applied, mem = setup()
    assert "camera is off" in ctl.handle("turn off the camera", None, 0.0)
    assert applied[-1].camera_enabled is False
    assert "camera is on" in ctl.handle("turn on the camera", None, 1.0)
    assert applied[-1].camera_enabled is True
    ctl.handle("switch to lightweight", None, 2.0)
    assert applied[-1].profile == "lightweight"


def test_object_memory_toggle_enables_and_wipes():
    ctl, store, pid, applied, mem = setup()
    ctl.handle("remember where things are", None, 0.0)
    assert mem.enabled and applied[-1].object_memory
    mem.observe("o", "taught", "cup", "mug", "sofa", (0, 0, 10, 10),
                (100, 100, 3), 0.0)
    assert mem.export() != "[]"
    ctl.handle("forget where things are", None, 1.0)
    assert not mem.enabled and mem.export() == "[]"


def test_turning_off_fall_detection_asks_first():
    ctl, store, pid, applied, mem = setup()
    q = ctl.handle("turn off fall detection", None, 0.0)
    assert "Are you sure" in q and applied == []
    ctl.handle("yes", None, 1.0)
    assert applied[-1].fall_detection is False
    ctl.handle("turn on fall detection", None, 2.0)
    assert applied[-1].fall_detection is True


def test_about_me_reports_counts_not_vectors():
    import numpy as np
    ctl, store, pid, applied, mem = setup()
    store.add_embedding(pid, np.ones(128, np.float32) / 11.3)
    r = ctl.handle("what do you know about me", pid, 0.0)
    assert "1 face templates" in r and "no photos" in r
    assert "forget you" in r

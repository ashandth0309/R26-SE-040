"""Phase 21: VisionNodeCore logic (no ROS)."""

import json

import numpy as np
from skeletons import hold, lying, standing, transition
from test_vision_context import (
    IMG, FakeExpr, FakeFaces, FakeObjects, FakePose, build,
    person_store)

from buddy_core.vision.face_store import FaceProfileStore
from buddy_core.vision.object_embedding import FakeObjectEmbedder
from buddy_core.vision.object_memory import ObjectMemory
from buddy_core.vision.object_teach import ObjectStore
from buddy_core.vision.objects import ObjectDetection
from buddy_core.vision.proactive import (
    ProactiveConfig, ProactiveInteractionManager)
from buddy_core.vision.vision_commands import VisionController
from buddy_core.vision.vision_factory import BuiltVision
from buddy_core.vision.vision_node import (
    VisionNodeCore, image_to_bgr)
from buddy_core.vision.vision_settings import VisionSettings


def core_for(faces=None, store=None, hz=0.0, embedder=True, **kw):
    mgr, clock = build(faces or FakeFaces(), store=store, **kw)
    ctl = VisionController(mgr.settings, mgr.apply_settings,
                           store or FaceProfileStore(), None,
                           mgr.c.proactive, mgr.c.greeting)
    built = BuiltVision(
        mgr, ctl, store or FaceProfileStore(), ObjectStore(), ObjectMemory(),
        mgr.settings, {}, None,
        FakeObjectEmbedder() if embedder else None)
    core = VisionNodeCore(built, max_process_hz=hz, clock=clock)

    return core, clock


def frame(core, clock, dt=0.5, image=IMG):
    clock.t += dt

    return core.on_frame(image)


def test_people_and_recognized_are_published_for_a_known_person():
    store, pid = person_store()
    core, c = core_for(FakeFaces(pid), store)
    out = frame(core, c)
    assert out.people[0]["known"] and out.people[0]["display_name"] == "Ann"
    assert out.recognized[0]["person_id"] == pid
    assert out.recognized[0]["bbox"][2] == 80.0     # width, not x2


def test_unknown_person_is_not_named_or_marked_known():
    core, c = core_for(FakeFaces(None))
    out = frame(core, c)
    assert not out.people[0]["known"] and out.people[0]["display_name"] == ""
    assert not out.recognized[0]["known"]
    assert out.recognized[0]["confidence"] == 0.0


def test_objects_are_published_only_when_the_stage_ran():
    det = ObjectDetection("cup", .9, (10, 10, 100, 100), "other")
    core, c = core_for(FakeFaces(), objects=FakeObjects([det]))
    seen = []

    for _ in range(6):
        out = frame(core, c, dt=1.0)

        if out.objects is not None:
            seen.append(out.objects)

    assert seen and seen[-1]["objects"][0]["label"] == "cup"


def test_greeting_is_marked_so_it_is_not_repeated():
    store, pid = person_store()
    core, c = core_for(FakeFaces(pid), store)
    kinds = []

    for _ in range(30):
        out = frame(core, c)
        kinds += [e["kind"] for e in out.events]

    assert kinds.count("greeting_due") == 1
    ev = []
    assert ev == []


def test_greeting_event_carries_text_for_the_voice_side():
    store, pid = person_store()
    core, c = core_for(FakeFaces(pid), store)
    texts = []

    for _ in range(10):
        texts += [e["suggested_text"] for e in frame(core, c).events
                  if e["kind"] == "greeting_due"]

    assert texts and "Ann" in texts[0]


def test_emotion_only_for_opted_in_people_and_cleared_when_they_leave():
    store, pid = person_store()
    store.set_setting(pid, "expression", True)
    faces = FakeFaces(pid)
    core, c = core_for(faces, store, expr=FakeExpr("happy", 0.9))
    words = []

    for _ in range(12):
        words += [e["expression"] for e in frame(core, c).emotions]

    assert "happy" in words
    faces.present = False
    cleared = []

    for _ in range(4):
        cleared += [e for e in frame(core, c).emotions]

    assert cleared and cleared[0]["expression"] == "unknown"
    assert cleared[0]["confidence"] == 0.0


def test_no_emotion_message_without_opt_in():
    store, pid = person_store()
    core, c = core_for(FakeFaces(pid), store, expr=FakeExpr("sad", 0.9))

    for _ in range(12):
        assert frame(core, c).emotions == []


def test_camera_off_publishes_nothing_and_clears_voice_hint():
    store, pid = person_store()
    store.set_setting(pid, "expression", True)
    core, c = core_for(FakeFaces(pid), store, expr=FakeExpr("happy", 0.9))

    for _ in range(12):
        frame(core, c)

    ok, _msg = core.set_camera_enabled(False)
    out = frame(core, c)
    assert ok and out.people == [] and out.objects is None
    assert [e["expression"] for e in out.emotions] == ["unknown"]
    assert core.status()["camera_enabled"] is False


def test_rate_limit_drops_extra_frames():
    core, c = core_for(hz=2.0)
    assert core.on_frame(IMG) is not None
    assert core.on_frame(IMG) is None
    assert core.frames_dropped == 1
    c.t += 1.0
    assert core.on_frame(IMG) is not None


def test_voice_status_gates_proactive_flags():
    core, _c = core_for()
    core.on_voice_status({"state": "BUDDY_SPEAKING",
                          "silence_mode": "quiet", "moving": True})
    m = core.b.manager
    assert m.voice_speaking and m.quiet_mode and m.robot_busy
    core.on_voice_status({"state": "SLEEPING", "silence_mode": "normal",
                          "moving": False, "muted": False})
    assert not (m.voice_speaking or m.quiet_mode or m.robot_busy)
    core.on_voice_status({"state": "AWAKE", "muted": True})
    assert m.quiet_mode


def fall_core():
    person = ObjectDetection("person", .9, (100, 50, 560, 470), "other")
    frames = [p for _, p in hold(standing(), 0.0, 3.0, fps=2)]
    frames += [p for _, p in transition(standing(), lying(), 0.0, 0.5,
                                        fps=2)]
    frames += [p for _, p in hold(lying(), 0.0, 60.0, fps=2)]
    core, c = core_for(
        FakeFaces(), objects=FakeObjects([person]), pose=FakePose(frames),
        settings=VisionSettings(profile="enhanced",
                                overrides={"pose": 0.5, "objects": 0.5}))

    return core, c


def run_until(core, c, kind, n=80, dt=0.5):
    for _ in range(n):
        out = frame(core, c, dt=dt)

        for e in out.events:
            if e["kind"] == kind:
                return out, e

    return None, None


def test_fall_question_sets_awaiting_reply_and_okay_clears_it():
    core, c = fall_core()
    _out, ev = run_until(core, c, "fall_suspected")
    assert ev and ev["needs_voice_check"]
    assert "okay" in ev["suggested_text"].lower()
    assert core.awaiting_reply() and core.status()["awaiting_reply"]
    handled, text, _p = core.command("I'm okay")
    assert handled and text
    assert not core.awaiting_reply()
    assert core.take_deferred().events[0]["kind"] == "fall_recovered"


def test_fall_alert_publishes_one_security_event_without_an_image():
    core, c = fall_core()
    run_until(core, c, "fall_suspected")
    out, ev = run_until(core, c, "fall_alert", n=100, dt=0.5)
    assert ev, "no answer should lead to an alert"
    assert len(out.security) == 1
    s = out.security[0]
    assert s["event_type"] == 4 and s["severity"] == 3
    assert "image" not in json.dumps(s).replace("No image was saved", "")


def test_help_reply_alerts_immediately_through_deferred_events():
    core, c = fall_core()
    run_until(core, c, "fall_suspected")
    handled, _t, _p = core.command("help")
    d = core.take_deferred()
    assert handled
    assert any(e["kind"] == "fall_alert" for e in d.events)
    assert len(d.security) == 1
    assert core.take_deferred().events == []


def test_proactive_offer_is_a_question_and_marks_awaiting_reply():
    store, pid = person_store()
    store.set_setting(pid, "proactive", True)
    pm = ProactiveInteractionManager(ProactiveConfig(sustain_s=5.0))
    core, c = core_for(FakeFaces(pid), store,
                       expr=FakeExpr("sad", 0.85), proactive=pm)
    _out, ev = run_until(core, c, "proactive_offer", n=80)
    assert ev and ev["suggested_text"].endswith("?")
    assert core.awaiting_reply()


def test_settings_command_goes_through_the_controller():
    core, _c = core_for()
    handled, text, _p = core.command("turn off the camera")
    assert handled and text


def test_unrelated_speech_is_not_handled():
    core, _c = core_for()
    assert core.command("what is the weather") == (False, "", True)


def test_identify_without_objects_stage_is_honest():
    core, _c = core_for()
    r = core.identify_object()
    assert not r["success"] and r["kind"] == "unavailable"


def test_identify_names_taught_object_from_last_frame():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    det = ObjectDetection("cup", .9, (100, 100, 300, 300), "other")
    core, c = core_for(FakeFaces(), objects=FakeObjects([det]))
    emb = core.b.object_embedder
    from buddy_core.vision.object_teach import crop
    core.b.objects.add("blue mug", [emb.embed(crop(img, det.box))], "cup")
    frame(core, c, image=img)
    r = core.identify_object()
    assert r["success"] and r["kind"] == "taught" and r["name"] == "blue mug"
    assert r["text"].startswith("I think")


def test_teach_list_forget_and_unavailable():
    core, _c = core_for(embedder=False)
    assert core.teach_object("start", "mug")["ok"] is False
    assert "haven't taught" in core.teach_object("list", "")["reply"]
    core2, _c2 = core_for()
    core2.b.objects.add("mug", [np.ones(192, np.float32)], None)
    assert "mug" in core2.teach_object("list", "")["reply"]
    assert core2.teach_object("forget", "mug")["ok"]
    assert core2.b.objects.list() == []
    assert core2.teach_object("bogus", "")["ok"] is False


def test_teach_start_marks_awaiting_reply_and_cancel_ends_it():
    core, _c = core_for()
    r = core.teach_object("start", "mug")
    assert r["ok"] and r["phase"] != "idle"
    assert core.awaiting_reply()
    core.teach_object("cancel", "")
    assert not core.awaiting_reply()


def test_where_is_without_memory_is_honest():
    core, _c = core_for()
    core.b.memory = None
    r = core.where_is("keys")
    assert not r["found"] and "not keeping track" in r["text"]


def test_status_has_no_names_and_declares_no_motion_interface():
    store, pid = person_store()
    core, c = core_for(FakeFaces(pid), store)
    frame(core, c)
    text = json.dumps(core.status())
    assert "Ann" not in text and pid not in text
    assert core.status()["motion_interface"] == "none"


def test_image_conversion():
    class M:
        height, width, step, encoding = 2, 3, 9, "bgr8"
        data = bytes(range(18))

    a = image_to_bgr(M)
    assert a.shape == (2, 3, 3) and a[0, 0, 0] == 0
    M.encoding = "rgb8"
    assert image_to_bgr(M)[0, 0, 0] == 2
    M.encoding = "mono8"
    assert image_to_bgr(M) is None
    M.encoding, M.data = "bgr8", b"123"
    assert image_to_bgr(M) is None
    M.height, M.data = 0, bytes(18)
    assert image_to_bgr(M) is None

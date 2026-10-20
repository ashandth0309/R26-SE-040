"""Adapter and (stubbed) ROS node tests.

The node tests use strict stand-ins for rclpy / std_msgs / buddy_interfaces.
They check wiring and message handling only.  Nothing here has been built
or run under a real ROS 2 installation.
"""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace as NS

import pytest
import ros_stub_head as stub
from head_fakes import make_stack

from buddy_core.head_tracking.models import EventKind, HeadState, HeadEvent
from buddy_core.head_tracking.ros_adapter import (
    HeadTrackingBridge,
    event_json,
    faces_from_people,
    gestures_from_people,
    hand_from_json,
    sound_from_json,
)
from buddy_core.head_tracking.sound_direction import Reason

NOW = 100.0


def person(key="session_person_1", x=280, y=200, w=80, h=80, **kw):
    base = dict(person_key=key, display_name="Alice Secret", known=True,
                bbox_x=x, bbox_y=y, bbox_width=w, bbox_height=h,
                gestures=[], expression_group="happy")
    base.update(kw)
    return NS(**base)


def bridge():
    _b, co, mgr, _ = make_stack()
    assert co.start(NOW)
    return HeadTrackingBridge(mgr), mgr, co


# ------------------------------------------------------ pure adapters
def test_people_become_faces_without_reading_identity():
    fr = faces_from_people(NS(people=[person()]), 640, 480, NOW)
    f = fr.faces[0]
    assert (f.track_id, f.x, f.w) == ("session_person_1", 280.0, 80.0)
    assert "Alice" not in repr(fr)               # display_name not carried


@pytest.mark.parametrize("bad", [
    person(key=""), person(w=0), person(h=-3), person(x=float("nan")),
    person(x=None), person(w="80"),
])
def test_malformed_people_are_skipped(bad):
    assert faces_from_people(NS(people=[bad]), 640, 480, NOW).faces == ()


def test_missing_people_field_is_tolerated():
    assert faces_from_people(NS(), 640, 480, NOW).faces == ()


def test_gestures_are_extracted_per_person():
    g = gestures_from_people(NS(people=[
        person(gestures=["raised_hand", "wave"]), person(key="k2")]))
    assert [(x.name, x.track_id) for x in g] == [
        ("raised_hand", "session_person_1"), ("wave", "session_person_1")]


def test_hand_json_roundtrip_and_validation():
    text = json.dumps({
        "width": 640, "height": 480, "track_id": "k",
        "wrists": {"left": [10, 20, 0.9], "up": [1, 2, 3]},
        "fingertips": {"right": [100, 200]},
        "gestures": [{"name": "raised_hand", "track_id": "k",
                      "side": "right"}, {"nope": 1}]})
    obs = hand_from_json(text, NOW)
    assert set(obs.wrists) == {"left"} and set(obs.fingertips) == {"right"}
    assert obs.fingertips["right"].confidence == 1.0
    assert [g.name for g in obs.gestures] == ["raised_hand"]
    for bad in ("not json", "[]", "{}", '{"width":0,"height":5}',
                '{"width":5}', ""):
        assert hand_from_json(bad, NOW) is None
    nan = hand_from_json('{"width":5,"height":5,"wrists":'
                         '{"left":[NaN,1,1]}}', NOW)
    assert nan.wrists == {}


def test_sound_json_requires_synchronization():
    ok = sound_from_json(json.dumps(
        {"ok": True, "azimuth_deg": -20, "confidence": 0.7,
         "synchronized": True}), NOW)
    assert ok.ok and ok.azimuth_deg == -20.0
    unsync = sound_from_json(json.dumps(
        {"ok": True, "azimuth_deg": -20, "confidence": 0.7}), NOW)
    assert not unsync.ok and unsync.reason == Reason.UNSYNCHRONIZED
    for bad in ('{"ok":true,"synchronized":true,"azimuth_deg":500}',
                '{"ok":true,"synchronized":true,"azimuth_deg":"x"}',
                'garbage', '[]'):
        assert not sound_from_json(bad, NOW).ok
    notok = sound_from_json(json.dumps(
        {"ok": False, "synchronized": True, "reason": "noise"}), NOW)
    assert not notok.ok and notok.reason == "noise"


def test_event_json_is_machine_readable():
    out = json.loads(event_json(HeadEvent(EventKind.HEAD_TURN_REQUESTED,
                                          1.234, {"pan_deg": 120.0})))
    assert out["event"] == "HEAD_TURN_REQUESTED"
    assert out["data"]["pan_deg"] == 120.0


# ------------------------------------------------------------- bridge
def test_bridge_drives_face_tracking_from_people_messages():
    b, mgr, co = bridge()
    t = NOW
    for _ in range(10):
        t += 0.05
        b.on_people(NS(people=[person(x=480, y=200)]), t)
        mgr.tick(t)
    assert mgr.state is HeadState.FACE_TRACKING and co.pan > 90.0


def test_bridge_wake_word_only_when_the_flag_is_true():
    b, mgr, _ = bridge()
    b.on_voice_command(NS(wake_word_detected=False), NOW)
    assert mgr.state is HeadState.IDLE
    b.on_voice_command(NS(wake_word_detected="yes"), NOW)   # not a bool
    assert mgr.state is HeadState.IDLE
    b.on_voice_command(NS(wake_word_detected=True), NOW)
    assert mgr.state is HeadState.LISTENING


def test_bridge_reads_speaking_and_mic_topology_from_voice_status():
    b, mgr, _ = bridge()
    b.on_voice_status(json.dumps({"state": "BUDDY_SPEAKING",
                                  "mic_topology": "stereo"}), NOW)
    assert mgr.speaking_blocked and b.mics_synchronized
    b.on_voice_status(json.dumps({"state": "AWAKE",
                                  "mic_topology": "two_devices"}), NOW + 1)
    assert not mgr.speaking_blocked and not b.mics_synchronized
    b.on_voice_status("not json", NOW)                    # ignored


def test_bridge_safety_state_latches_the_emergency_stop():
    b, mgr, co = bridge()
    b.on_safety(NS(emergency_stop_active=True, safety_level=3,
                   reason="bumper"), NOW)
    assert mgr.state is HeadState.EMERGENCY_STOP and co.estop
    b.on_safety(NS(emergency_stop_active=False, safety_level=0,
                   reason=""), NOW + 1)
    assert mgr.state is HeadState.IDLE
    b.on_safety(NS(emergency_stop_active=False, safety_level=3,
                   reason=""), NOW + 2)                   # level only
    assert mgr.state is HeadState.EMERGENCY_STOP


def test_bridge_commands():
    b, mgr, _ = bridge()
    assert b.on_command("stop", NOW) and mgr.state is HeadState.STOPPED
    assert b.on_command("RESUME", NOW) and mgr.state is HeadState.IDLE
    assert b.on_command("hand_on left", NOW)
    assert mgr.state is HeadState.HAND_TRACKING
    assert b.on_command("hand_off", NOW) and mgr.state is HeadState.IDLE
    for junk in ("", "dance", "hand_on up", "stop now please? no"):
        before = mgr.state
        assert b.on_command(junk, NOW) is (junk == "stop now please? no")
        if junk != "stop now please? no":
            assert mgr.state is before
        else:
            b.on_command("resume", NOW)


def test_bridge_hand_json_activation_and_following():
    b, mgr, co = bridge()
    t = NOW
    b.on_command("hand_on right", t)
    for _ in range(30):
        t += 0.05
        b.on_hand_json(json.dumps({
            "width": 640, "height": 480,
            "wrists": {"right": [560, 240, 0.9]}}), t)
        mgr.tick(t)
    assert co.pan > 90.0
    assert b.on_hand_json("junk", t) is False


def test_status_json_has_no_track_ids_or_names():
    b, mgr, _ = bridge()
    for i in range(8):
        b.on_people(NS(people=[person(key="private-key-1")]),
                    NOW + 0.05 * i)
    s = b.status_json(NOW + 1)
    assert "private-key-1" not in s and "Alice" not in s
    assert json.loads(s)["state"] in {h.value for h in HeadState}


# --------------------------------------------------------- stubbed node
@pytest.fixture
def node_env():
    types_ = stub.install(with_people=True)
    yield types_
    stub.uninstall()


def run_node(**overrides):
    stub.Node.overrides = overrides
    mod = importlib.import_module("buddy_core.head_tracking.ros_node")
    mod = importlib.reload(mod)
    return mod.main([]), stub.Node.instances[-1]


def test_node_starts_in_mock_mode_and_wires_expected_topics(node_env):
    rc, node = run_node()
    assert rc == 0 and stub.Rclpy.state["spun"] == 1
    assert {"/safety_state", "/vision/people", "/voice/command",
            "/voice/status", "/head/command", "/head/hand_observation",
            "/head/sound_direction"} <= set(node.subs)
    assert {"/head/events", "/head/status"} <= set(node.pubs)
    assert any("mock" in t for lvl, t in node._logger.lines
               if lvl == "info")
    assert node.destroyed


def test_node_never_subscribes_to_raw_camera_or_audio(node_env):
    _rc, node = run_node()
    for topic in node.subs:
        assert "image" not in topic and "audio" not in topic
    for _mt, _cb in node.subs.values():
        assert _mt is not None


def test_node_routes_messages_and_publishes_status_and_events(node_env):
    types_ = node_env
    mod = importlib.import_module("buddy_core.head_tracking.ros_node")
    importlib.reload(mod)
    stub.Rclpy.spin = classmethod(lambda cls, n: None)
    holder = {}
    orig_destroy = stub.Node.destroy_node

    def keep(self):
        holder["node"] = self
        orig_destroy(self)

    stub.Node.destroy_node = keep
    try:
        mod.main([])
    finally:
        stub.Node.destroy_node = orig_destroy
        stub.Rclpy.spin = classmethod(lambda cls, n: None)
    node = holder["node"]
    # destroyed already: wiring is still inspectable
    cmd = types_["VoiceCommand"](wake_word_detected=True)
    node.subs["/voice/command"][1](cmd)
    sent = [json.loads(m.data) for m in node.pubs["/head/events"].sent]
    assert any(e["event"] == "WAKE_WORD_DETECTED" for e in sent)
    node.timers[1.0]()
    status = json.loads(node.pubs["/head/status"].sent[-1].data)
    assert status["state"] in ("LISTENING", "IDLE")


def test_node_without_vision_branch_warns_and_still_starts():
    stub.install(with_people=False)
    try:
        rc, node = run_node()
        assert rc == 0 and "/vision/people" not in node.subs
        assert any(lvl == "warn" and "VisionPeople" in t
                   for lvl, t in node._logger.lines)
    finally:
        stub.uninstall()


def test_node_refuses_hardware_mode_without_all_gates(node_env, tmp_path):
    cfg = tmp_path / "ht.yaml"
    cfg.write_text("head_tracking:\n  hardware:\n    mode: hardware\n")
    stub.Node.overrides = {"head_config_path": str(cfg)}
    mod = importlib.import_module("buddy_core.head_tracking.ros_node")
    mod = importlib.reload(mod)
    from buddy_core.head_tracking.config import HeadTrackingConfigError
    with pytest.raises(HeadTrackingConfigError):
        mod.main([])
    node = stub.Node.instances[-1]
    assert any(lvl == "error" for lvl, _ in node._logger.lines)
    assert stub.Rclpy.state["shutdowns"] >= 1

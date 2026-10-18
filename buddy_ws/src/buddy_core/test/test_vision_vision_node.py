"""vision_node under strict ROS stubs generated from the repo .msg/.srv."""

import importlib
import json

import numpy as np
import pytest

import ros_stub_vision as stub
from test_vision_node_core import core_for, fall_core, run_until

MSGS = ("EmotionObservation", "RecognizedPerson", "SecurityEvent",
        "VisionObject", "VisionObjects", "VisionPerson", "VisionPeople",
        "VisionEvent")
SRVS = ("VisionCommand", "IdentifyObject", "TeachObject", "WhereIs")
msgs = None


@pytest.fixture(scope="module", autouse=True)
def ros_stubs():
    global msgs
    msgs = stub.install(MSGS, SRVS)
    yield
    stub.uninstall()


@pytest.fixture
def node(monkeypatch):
    stub.base.Node.overrides = {}
    stub.base.Node.existing_nodes = []
    stub.base.Node.instances = []
    module = importlib.reload(
        importlib.import_module("buddy_core.vision.vision_node"))
    core, clock = core_for()

    def fake_build(**kw):
        return core.b

    monkeypatch.setattr("buddy_core.vision.vision_factory.build_vision",
                        fake_build)
    cls = module.build_node_class()
    n = cls()
    n.clock = clock

    return n


def image_msg(w=64, h=48, enc="bgr8"):
    m = stub.Image()
    m.height, m.width, m.encoding, m.step = h, w, enc, w * 3
    m.data = np.zeros((h, w, 3), np.uint8).tobytes()

    return m


def test_interfaces_match_the_documented_names(node):
    assert set(node.pubs) == {
        "/vision/people", "/vision/objects", "/vision/events",
        "/vision/status", "/person/recognized", "/emotion/observation",
        "/security/event"}
    assert set(node.subs) == {"/camera/image_raw", "/voice/status"}
    assert set(node.services) == {
        "/vision/command", "/vision/identify_object",
        "/vision/teach_object", "/vision/where_is", "/vision/enable"}


def test_node_has_no_motion_or_navigation_interface(node):
    for topic in list(node.pubs) + list(node.subs):
        assert not topic.startswith(("/movement", "/navigation", "/cmd_vel"))

    assert "/safety_state" not in node.subs


def test_frame_in_people_out(node):
    node._on_image(image_msg())
    node.timers[min(node.timers)]()
    sent = node.pubs["/vision/people"].sent
    assert len(sent) == 1 and len(sent[0].people) == 1
    person = sent[0].people[0]
    assert not person.known and person.bbox_width == 80.0
    assert node.pubs["/person/recognized"].sent[0].known is False


def test_same_frame_is_not_processed_twice_and_bad_image_ignored(node):
    node._on_image(image_msg(enc="mono8"))
    node.timers[min(node.timers)]()
    assert node.pubs["/vision/people"].sent == []
    node._on_image(image_msg())
    tick = node.timers[min(node.timers)]
    tick()
    tick()
    assert len(node.pubs["/vision/people"].sent) == 1


def test_stale_frame_is_dropped(node):
    node._on_image(image_msg())
    node._latest_t -= 5.0
    node.timers[min(node.timers)]()
    assert node.pubs["/vision/people"].sent == []
    assert node.core.frames_dropped == 1


def test_voice_status_json_reaches_core_and_garbage_is_ignored(node):
    s = stub.base.StringMsg()
    s.data = json.dumps({"state": "BUDDY_SPEAKING"})
    node._on_voice(s)
    assert node.core.b.manager.voice_speaking
    s.data = "not json"
    node._on_voice(s)


def test_status_topic_is_json_without_names(node):
    node._publish_status()
    data = json.loads(node.pubs["/vision/status"].sent[0].data)
    assert data["motion_interface"] == "none" and "awaiting_reply" in data


def test_services_fill_strict_responses(node):
    SRV = importlib.import_module("buddy_interfaces.srv")
    req = SRV.VisionCommand.Request(text="turn off the camera")
    res = node._srv_command(req, SRV.VisionCommand.Response())
    assert res.handled and res.reply
    res = node._srv_identify(SRV.IdentifyObject.Request(),
                             SRV.IdentifyObject.Response())
    assert not res.success and res.kind == "unavailable"
    res = node._srv_teach(SRV.TeachObject.Request(action="list"),
                          SRV.TeachObject.Response())
    assert res.ok
    res = node._srv_where(SRV.WhereIs.Request(name="keys"),
                          SRV.WhereIs.Response())
    assert isinstance(res.found, bool) and res.text
    sb = importlib.import_module("std_srvs.srv").SetBool
    res = node._srv_enable(sb.Request(data=False), sb.Response())
    assert res.success and not node.core.b.manager.settings.camera_enabled


def test_fall_alert_reaches_security_topic_and_events(node):
    core, c = fall_core()
    node.core = core
    run_until(core, c, "fall_suspected")
    handled, _t, _p = core.command("help")
    node._publish(core.take_deferred())
    sec = node.pubs["/security/event"].sent
    assert handled and len(sec) == 1
    assert sec[0].event_type == 4 and sec[0].severity == 3
    assert sec[0].image_path == "" and sec[0].video_path == ""
    kinds = [e.kind for e in node.pubs["/vision/events"].sent]
    assert "fall_alert" in kinds

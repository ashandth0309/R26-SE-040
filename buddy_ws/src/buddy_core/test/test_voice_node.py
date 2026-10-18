"""voice_node under strict ROS stubs (no real ROS 2 involved)."""

import importlib
import json

import pytest

import ros_stub_voice as stub
from buddy_core.voice import command_router as cr
from buddy_core.voice.voice_commands import parse_command

msgs = None


@pytest.fixture(scope="module", autouse=True)
def ros_stubs():
    """Install the stubs for this module only, then restore."""

    global msgs
    msgs = stub.install()
    yield
    stub.uninstall()


@pytest.fixture
def vn():
    stub.Node.overrides = {"dry_run": True, "autostart_runner": False,
                           "memory_db_path": ":memory:"}
    stub.Node.existing_nodes = []
    stub.Node.instances = []
    stub.Rclpy.state.update(ok=False, spun=0, shutdowns=0)
    module = importlib.import_module("buddy_core.voice.voice_node")
    module = importlib.reload(module)

    return module


def drain(node):
    node.runner.step()


def test_node_builds_with_expected_interfaces(vn):
    node = vn.VoiceCompanionNode()
    assert set(node.pubs) >= {"/movement/request", "/voice/command",
                              "/voice/status"}
    assert set(node.subs) == {"/safety_state", "/emotion/observation",
                              "/voice/say"}
    assert "/voice/mute" in node.services
    assert "/movement/safe" not in node.pubs
    assert "/navigation/command" not in node.pubs


def test_refuses_to_run_beside_the_old_voice_node(vn):
    stub.Node.existing_nodes = ["buddy_voice"]
    with pytest.raises(RuntimeError, match="buddy_voice"):
        vn.VoiceCompanionNode()
    stub.Node.overrides["allow_coexist"] = True
    vn.VoiceCompanionNode()


def test_movement_message_matches_the_real_definition(vn):
    node = vn.VoiceCompanionNode()
    node.companion.router.handle(parse_command("go forward"), awake=True)
    sent = node.pubs["/movement/request"].sent[-1]
    assert isinstance(sent, msgs.MovementCommand)
    assert sent.command == msgs.MovementCommand.COMMAND_FORWARD
    assert sent.source == "buddy_voice" and sent.linear_speed == 0.2
    assert sent.valid_for.sec == 0 and sent.valid_for.nanosec == 300000000
    assert sent.emergency is False and sent.sequence_id >= 1
    assert cr.MOVE_FORWARD == msgs.MovementCommand.COMMAND_FORWARD
    assert cr.MOVE_ROTATE_LEFT == msgs.MovementCommand.COMMAND_ROTATE_LEFT


def test_emergency_state_vetoes_motion_until_cleared(vn):
    node = vn.VoiceCompanionNode()
    router = node.companion.router
    router.handle(parse_command("go forward"), awake=True)
    _, cb = node.subs["/safety_state"]
    cb(msgs.SafetyState(emergency_stop_active=True, movement_allowed=False))
    drain(node)
    assert not router.moving
    sent = node.pubs["/movement/request"].sent
    assert sent[-1].command == msgs.MovementCommand.COMMAND_STOP
    res = router.handle(parse_command("go forward"), awake=True)
    assert res.ack == "I can't move right now." and not router.moving
    assert router.handle(parse_command("stop"), awake=True).accepted
    cb(msgs.SafetyState(emergency_stop_active=False,
                        movement_allowed=True))
    drain(node)
    assert router.handle(parse_command("go forward"),
                         awake=True).accepted


def test_emotion_observation_becomes_a_vision_hint(vn):
    node = vn.VoiceCompanionNode()
    _, cb = node.subs["/emotion/observation"]
    cb(msgs.EmotionObservation(expression="happy", confidence=0.8))
    drain(node)
    assert node.companion.emotion.vision_expression == "happy"
    assert node.companion.emotion.vision_confidence == pytest.approx(0.8)


def test_mute_service_mutes_stops_and_only_service_unmutes(vn):
    node = vn.VoiceCompanionNode()
    node.companion.router.handle(parse_command("go forward"), awake=True)
    _, cb = node.services["/voice/mute"]
    resp = cb(stub.SetBool.Request(True), stub.SetBool.Response())
    assert resp.success and resp.message == "muted"
    drain(node)
    assert node.companion.engine.muted
    assert node.pubs["/movement/request"].sent[-1].command == 0
    assert node.companion.status()["muted"] is True
    cb(stub.SetBool.Request(False), stub.SetBool.Response())
    drain(node)
    assert not node.companion.engine.muted


def test_status_topic_is_json_without_speech(vn):
    node = vn.VoiceCompanionNode()
    node.timers[1.0]()
    data = json.loads(node.pubs["/voice/status"].sent[-1].data)
    assert data["state"] == "LISTENING_FOR_WAKE"
    assert data["cloud"] == "disabled" and data["muted"] is False
    assert not any(k in data for k in ("text", "transcript", "audio"))


def test_voice_command_message_published_for_commands_only(vn):
    from buddy_core.voice.stt import Transcript

    node = vn.VoiceCompanionNode()
    engine = node.companion.engine
    engine.turn.wake()
    engine._handle_awake_text(Transcript("go forward", 0.9),
                              Transcript("go forward", 0.9))
    sent = node.pubs["/voice/command"].sent
    assert len(sent) == 1 and isinstance(sent[0], msgs.VoiceCommand)
    assert sent[0].intent == "forward" and sent[0].request_id == "voice-1"
    assert sent[0].confidence == pytest.approx(0.9)
    engine.turn.user_started()
    engine.turn.user_ended()
    engine._handle_awake_text(None, Transcript("what time is it", 0.9))
    assert len(node.pubs["/voice/command"].sent) == 1


def test_voice_command_publication_can_be_disabled(vn):
    from buddy_core.voice.stt import Transcript

    stub.Node.overrides["publish_voice_command"] = False
    node = vn.VoiceCompanionNode()
    node.companion.engine.turn.wake()
    node.companion.engine._handle_awake_text(
        Transcript("go forward", 0.9), Transcript("go forward", 0.9))
    assert node.pubs["/voice/command"].sent == []


def test_navigation_requested_but_package_missing_stays_off(vn):
    stub.Node.overrides["enable_navigation"] = True
    node = vn.VoiceCompanionNode()
    assert "/navigation/command" not in node.pubs
    assert not node.companion.router.navigation_available
    assert any(level == "warn" for level, _ in node.get_logger().lines)


def test_navigation_wired_when_package_present(vn):
    import sys
    import types

    nav = types.ModuleType("buddy_nav_interfaces.msg")
    nav.NavigationCommand = type("NavigationCommand", (), {
        "__slots__": ("stamp", "request_id", "source", "command",
                      "duration_s", "route_id", "repeat", "location_name"),
        "__init__": lambda self: None})
    nav.NavigationStatus = type("NavigationStatus", (), {
        "__slots__": ("request_id", "state")})
    pkg = types.ModuleType("buddy_nav_interfaces")
    pkg.msg = nav
    sys.modules.update({"buddy_nav_interfaces": pkg,
                        "buddy_nav_interfaces.msg": nav})
    try:
        stub.Node.overrides["enable_navigation"] = True
        node = vn.VoiceCompanionNode()
        assert "/navigation/command" in node.pubs
        assert "/navigation/status" in node.subs
        assert node.companion.router.navigation_available
        node.companion.router.handle(parse_command("start patrol"),
                                     awake=True)
        sent = node.pubs["/navigation/command"].sent[-1]
        assert sent.command == cr.NAV_START_PATROL
        assert sent.request_id == "voice-1" and sent.source == "buddy_voice"
        status = nav.NavigationStatus()
        status.request_id, status.state = "voice-1", 4
        node.subs["/navigation/status"][1](status)
        drain(node)
        assert node.companion.router.tracked["voice-1"].state.value == \
            "rejected"
    finally:
        sys.modules.pop("buddy_nav_interfaces", None)
        sys.modules.pop("buddy_nav_interfaces.msg", None)


def test_destroy_stops_the_robot(vn):
    node = vn.VoiceCompanionNode()
    node.companion.router.handle(parse_command("go forward"), awake=True)
    node.destroy_node()
    assert node.destroyed
    assert node.pubs["/movement/request"].sent[-1].command == 0


def test_main_exit_codes(vn, capsys):
    stub.Node.overrides.pop("dry_run")
    stub.Node.overrides["vosk_model_path"] = ""
    assert vn.main() == 2
    assert "cannot start" in capsys.readouterr().err
    assert stub.Rclpy.state["shutdowns"] == 1
    stub.Node.overrides["dry_run"] = True
    assert vn.main() == 0
    assert stub.Rclpy.state["spun"] == 1


def test_main_reports_other_failures_with_code_1(vn, capsys):
    stub.Node.existing_nodes = ["buddy_voice"]
    assert vn.main() == 1
    assert "buddy_voice" in capsys.readouterr().err

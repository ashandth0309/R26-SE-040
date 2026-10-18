"""Voice-node hooks used by the integrated system.

Strict ROS stand-ins only; nothing here is a real ROS 2 run.
"""

import importlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import ros_stub_voice as stub
from buddy_core.voice import command_router as cr
from buddy_core.voice.dialogue_manager import DialogueReply
from buddy_core.voice.stt import Transcript
from buddy_core.voice.voice_commands import parse_command
from buddy_core.voice.voice_engine import GatewayAnswer

msgs = None
NAV_MSG = (Path(__file__).resolve().parents[2] / "buddy_nav_interfaces"
           / "msg" / "NavigationStatus.msg")


@pytest.fixture(scope="module", autouse=True)
def ros_stubs():
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
    module = importlib.import_module("buddy_core.voice.voice_node")

    return importlib.reload(module)


def drain(node):
    node.runner.step()


def record_speech(node):
    """Wrap the speaker so tests see (text, priority) of what is queued."""

    log = []
    speaker = node.companion.speaker
    original = speaker.say

    def recording(text, samples, rate, priority=None, **kw):
        log.append((text, int(priority) if priority is not None else None))

        return original(text, samples, rate, priority=priority, **kw)

    speaker.say = recording

    return log


def say(msgs_, text, priority=2, ttl=0.0, interrupt=False):
    request = msgs_.SpeechRequest()
    request.text, request.priority = text, priority
    request.ttl_s, request.interrupt = ttl, interrupt

    return request


# ---- the nav state table must match the real NavigationStatus.msg --------


def test_nav_state_names_match_navigation_status_msg(vn):
    constants = dict(
        re.findall(r"^uint8 STATE_(\w+)=(\d+)", NAV_MSG.read_text(),
                   re.MULTILINE))
    expected = {"IDLE": "idle", "RUNNING": "active", "PAUSED": "paused",
                "SUCCEEDED": "done", "FAILED": "failed",
                "CANCELLED": "cancelled"}

    assert {int(v): expected[k] for k, v in constants.items()} == \
        vn.NAV_STATE_NAMES


def test_succeeded_navigation_is_reported_as_done_not_rejected(vn):
    node = vn.VoiceCompanionNode()
    router = node.companion.router
    router.tracked["voice-9"] = cr.TrackedCommand(
        "voice-9", cr.VoiceCommand.PATROL, cr.CommandState.ACTIVE)
    status = type("S", (), {"request_id": "voice-9", "state": 3,
                            "reason": "route finished"})()
    node._on_nav_status(status)
    drain(node)

    assert router.tracked["voice-9"].state is cr.CommandState.DONE
    assert router.tracked["voice-9"].detail == "route finished"


def test_cancelled_navigation_clears_the_active_goal(vn):
    node = vn.VoiceCompanionNode()
    router = node.companion.router
    router._nav_goal_active = True
    router.tracked["voice-4"] = cr.TrackedCommand(
        "voice-4", cr.VoiceCommand.PATROL, cr.CommandState.ACTIVE)
    router.on_navigation_status("voice-4", "cancelled")

    assert router._nav_goal_active is False
    assert router.tracked["voice-4"].state is cr.CommandState.DONE


# ---- STOP always has a direct route ------------------------------------


def test_default_wiring_publishes_stop_once(vn):
    node = vn.VoiceCompanionNode()
    node.companion.router.stop("test")

    assert len(node.pubs["/movement/request"].sent) == 1


def test_stop_also_goes_direct_when_movement_is_relayed(vn):
    stub.Node.overrides["movement_topic"] = "/coordinator/voice_movement"
    node = vn.VoiceCompanionNode()
    router = node.companion.router
    router.handle(parse_command("go forward"), awake=True)

    # motion goes only to the relay topic, never straight to the supervisor
    assert node.pubs["/coordinator/voice_movement"].sent[-1].command == 1
    assert "/movement/request" in node.pubs
    assert node.pubs["/movement/request"].sent == []

    router.stop("test")

    assert node.pubs["/coordinator/voice_movement"].sent[-1].command == 0
    assert node.pubs["/movement/request"].sent[-1].command == 0


def test_navigation_stop_also_goes_direct_when_navigation_is_relayed(vn):
    import sys
    import types

    nav = types.ModuleType("buddy_nav_interfaces.msg")
    nav.NavigationCommand = type("NavigationCommand", (), {
        "__slots__": ("stamp", "request_id", "source", "command",
                      "duration_s", "route_id", "repeat", "location_name"),
        "__init__": lambda self: None})
    nav.NavigationStatus = type("NavigationStatus", (), {
        "__slots__": ("request_id", "state", "reason")})
    pkg = types.ModuleType("buddy_nav_interfaces")
    pkg.msg = nav
    sys.modules.update({"buddy_nav_interfaces": pkg,
                        "buddy_nav_interfaces.msg": nav})
    try:
        stub.Node.overrides.update(
            enable_navigation=True,
            navigation_topic="/coordinator/voice_navigation")
        node = vn.VoiceCompanionNode()
        router = node.companion.router
        router.handle(parse_command("start patrol"), awake=True)

        assert node.pubs["/coordinator/voice_navigation"].sent[-1] \
            .command == cr.NAV_START_PATROL
        assert node.pubs["/navigation/command"].sent == []

        router.stop("test")

        assert node.pubs["/navigation/command"].sent[-1].command == \
            cr.NAV_STOP
    finally:
        sys.modules.pop("buddy_nav_interfaces", None)
        sys.modules.pop("buddy_nav_interfaces.msg", None)


# ---- /voice/say: one speaker, one queue ----------------------------------


def test_say_request_is_queued_with_its_priority(vn):
    node = vn.VoiceCompanionNode()
    log = record_speech(node)
    node.subs["/voice/say"][1](say(msgs, "It is a pen.", priority=2))
    drain(node)

    assert log == [("It is a pen.", 2)]


def test_say_priority_is_clamped_to_the_valid_range(vn):
    node = vn.VoiceCompanionNode()
    log = record_speech(node)
    node.subs["/voice/say"][1](say(msgs, "odd", priority=200))
    drain(node)

    assert log == [("odd", 3)]


def test_stale_say_request_is_dropped(vn):
    node = vn.VoiceCompanionNode()
    request = say(msgs, "too late", ttl=0.05)
    node.subs["/voice/say"][1](request)
    time.sleep(0.1)
    drain(node)

    assert node.say_dropped["stale"] == 1


def test_quiet_mode_silences_everything_but_safety_speech(vn):
    node = vn.VoiceCompanionNode()
    log = record_speech(node)
    node.companion.engine.quiet_mode = True
    cb = node.subs["/voice/say"][1]
    cb(say(msgs, "proactive hello", priority=3))
    cb(say(msgs, "here is your answer", priority=2))
    cb(say(msgs, "Warning, obstacle.", priority=0))
    drain(node)

    assert node.say_dropped["quiet"] == 2
    assert log == [("Warning, obstacle.", 0)]


def test_empty_say_request_is_ignored(vn):
    node = vn.VoiceCompanionNode()
    node.subs["/voice/say"][1](say(msgs, "   "))
    drain(node)

    assert node.say_dropped == {"stale": 0, "quiet": 0}


# ---- utterance gateway ---------------------------------------------------


def srv_stub():
    """Minimal RouteUtterance type, fields as in the real .srv file."""

    class Request:
        __slots__ = ("request_id", "text", "confidence", "speaker_id")

    class Response:
        __slots__ = ("claimed", "reply", "intent", "error_code")

        def __init__(self, claimed=False, reply="", intent=""):
            self.claimed, self.reply, self.intent = claimed, reply, intent
            self.error_code = 0

    return type("RouteUtterance", (), {"Request": Request,
                                       "Response": Response})


def test_gateway_returns_the_coordinators_answer(vn):
    srv = srv_stub()
    client = stub.StubClient(srv, "/coordinator/route_utterance")
    client.responder = lambda req: srv.Response(True, "That is a pen.",
                                                "identify")
    gateway = vn.RosUtteranceGateway(client, srv, 1.0)
    answer = gateway("what is this", 0.9)

    assert answer == GatewayAnswer(True, "That is a pen.", "identify")
    assert client.requests[0].text == "what is this"
    assert client.requests[0].confidence == pytest.approx(0.9)


def test_gateway_fails_open_when_service_is_not_up(vn):
    srv = srv_stub()
    client = stub.StubClient(srv, "x")
    client.ready = False
    gateway = vn.RosUtteranceGateway(client, srv, 1.0)

    assert gateway("what is this", 0.9) is None
    assert gateway.unavailable == 1 and client.requests == []


def test_gateway_fails_open_on_timeout(vn):
    srv = srv_stub()
    client = stub.StubClient(srv, "x")      # never answers
    gateway = vn.RosUtteranceGateway(client, srv, 0.05)
    started = time.monotonic()

    assert gateway("what is this", 0.9) is None
    assert time.monotonic() - started < 0.5
    assert gateway.timeouts == 1


def test_gateway_is_off_unless_enabled(vn):
    node = vn.VoiceCompanionNode()

    assert node.gateway is None and node.clients == {}


def test_gateway_enabled_creates_client_when_interface_exists(vn):
    import sys
    import types

    srv_mod = types.ModuleType("buddy_interfaces.srv")
    srv_mod.RouteUtterance = srv_stub()
    sys.modules["buddy_interfaces"].srv = srv_mod
    sys.modules["buddy_interfaces.srv"] = srv_mod
    try:
        stub.Node.overrides["enable_gateway"] = True
        node = vn.VoiceCompanionNode()

        assert "/coordinator/route_utterance" in node.clients
        assert node.gateway is not None
    finally:
        sys.modules.pop("buddy_interfaces.srv", None)


def test_gateway_requested_but_interface_missing_stays_off(vn):
    stub.Node.overrides["enable_gateway"] = True
    node = vn.VoiceCompanionNode()

    assert node.gateway is None
    assert any(level == "warn" for level, _ in node.get_logger().lines)


# ---- engine behaviour with a gateway -------------------------------------


def awake_engine(node):
    engine = node.companion.engine
    engine.turn.wake()
    engine.turn.user_started()
    engine.turn.user_ended()

    return engine


def test_claimed_utterance_is_answered_by_the_gateway(vn):
    node = vn.VoiceCompanionNode()
    log = record_speech(node)
    engine = awake_engine(node)
    engine.utterance_gateway = lambda text, conf: GatewayAnswer(
        True, "That looks like a pen.", "identify")
    engine._handle_awake_text(None, Transcript("what is this", 0.9))

    assert log == [("That looks like a pen.", 2)]
    assert engine.gateway_claims == 1


def test_unclaimed_utterance_gets_a_normal_reply(vn):
    node = vn.VoiceCompanionNode()
    log = record_speech(node)
    engine = awake_engine(node)
    engine.utterance_gateway = lambda text, conf: GatewayAnswer(False)
    engine._handle_awake_text(None, Transcript("how are you", 0.9))

    assert engine.gateway_claims == 0
    assert log, "dialogue should still answer"


def test_gateway_crash_falls_back_to_dialogue(vn):
    node = vn.VoiceCompanionNode()
    log = record_speech(node)
    engine = awake_engine(node)

    def boom(text, conf):
        raise RuntimeError("coordinator exploded")

    engine.utterance_gateway = boom
    engine._handle_awake_text(None, Transcript("how are you", 0.9))

    assert engine.errors >= 1 and log


def test_slow_gateway_never_blocks_the_audio_path_and_stop_cancels_it(vn):
    node = vn.VoiceCompanionNode()
    log = record_speech(node)
    engine = awake_engine(node)
    release = threading.Event()
    engine.executor = ThreadPoolExecutor(max_workers=1)
    engine.utterance_gateway = lambda text, conf: (
        release.wait(5) and GatewayAnswer(True, "late answer"))
    started = time.monotonic()
    engine._handle_awake_text(None, Transcript("what is this", 0.9))

    assert time.monotonic() - started < 0.5, "reply must be asynchronous"
    assert engine._inflight is not None

    engine._do_stop("test stop")          # hearing "stop" still works
    release.set()
    time.sleep(0.05)
    engine.tick()
    engine.executor.shutdown(wait=False, cancel_futures=True)

    assert all("late answer" not in text for text, _ in log)
    assert node.pubs["/movement/request"].sent[-1].command == 0

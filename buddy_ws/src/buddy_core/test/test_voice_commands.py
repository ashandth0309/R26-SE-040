"""Phase 3 tests: command parsing, routing, recogniser and TTS adapters."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from buddy_core.voice import command_router as cr
from buddy_core.voice.command_router import (
    CommandRouter,
    CommandState,
    RecordingSink,
    RouterConfig,
    RouteStatus,
)
from buddy_core.voice.stt import ScriptedRecognizer, Transcript, VoskRecognizer
from buddy_core.voice.tts_backends import (
    AplayPlayback,
    FakeSynthesizer,
    PiperSynthesizer,
    SpeechStyle,
)
from buddy_core.voice.vision_interface import MockVision, fresh
from buddy_core.voice.voice_commands import (
    VoiceCommand,
    command_grammar,
    is_sleep,
    is_stop,
    is_wake,
    parse_command,
    wake_grammar,
)

REPO = Path(__file__).resolve().parents[2]


class Clock:
    def __init__(self):
        self.t = 10.0

    def __call__(self):
        return self.t


def kind(text, locs=()):
    p = parse_command(text, locs)
    return p.kind if p else None


# ----- parsing -----

@pytest.mark.parametrize("text,expected", [
    ("stop", VoiceCommand.STOP),
    ("buddy stop", VoiceCommand.STOP),
    ("halt", VoiceCommand.STOP),
    ("Freeze!", VoiceCommand.STOP),
    ("please stop now", VoiceCommand.STOP),
    ("buddy go forward", VoiceCommand.FORWARD),
    ("Move forward", VoiceCommand.FORWARD),
    ("buddy please go back", VoiceCommand.BACKWARD),
    ("reverse", VoiceCommand.BACKWARD),
    ("turn left", VoiceCommand.TURN_LEFT),
    ("buddy rotate right", VoiceCommand.TURN_RIGHT),
    ("start patrol", VoiceCommand.PATROL),
    ("cancel navigation", VoiceCommand.CANCEL),
])
def test_valid_commands(text, expected):
    assert kind(text) is expected


@pytest.mark.parametrize("text", [
    "", "go", "left", "back", "buddy", "i went forward yesterday",
    "he said go forward to the car", "don't go forward",
    "never turn left", "if you go forward", "should i turn right",
    "what does go forward mean", "forward me the email",
    "how are you", "tell me a joke",
])
def test_ambiguous_or_conversational_never_a_command(text):
    assert kind(text) is None


def test_stop_is_loose_by_design():
    assert is_stop("don't stop the music")
    assert kind("don't stop") is VoiceCommand.STOP
    assert not is_stop("go forward")


def test_go_to_requires_known_location():
    locs = ("kitchen", "front door")
    p = parse_command("buddy go to the kitchen", locs)
    assert p and p.kind is VoiceCommand.GO_TO and p.location == "kitchen"
    assert parse_command("go to front door", locs).location == "front door"
    assert parse_command("go to the moon", locs) is None
    assert parse_command("go to the kitchen") is None


def test_wake_and_sleep_phrases():
    assert is_wake("Hey Buddy")
    assert is_wake("buddy")
    assert not is_wake("buddy go forward")
    assert is_sleep("go to sleep")
    assert not is_sleep("i could sleep")


def test_grammars_contain_stop_and_unk():
    assert "stop" in wake_grammar() and "[unk]" in wake_grammar()
    g = command_grammar(("kitchen",))
    assert "go to the kitchen" in g and "buddy stop" in g


# ----- constants match the real message definitions -----

def test_movement_constants_match_msg():
    text = (REPO / "buddy_interfaces" / "msg" /
            "MovementCommand.msg").read_text()
    found = dict(re.findall(r"uint8 COMMAND_(\w+)=(\d+)", text))
    assert int(found["STOP"]) == cr.MOVE_STOP
    assert int(found["FORWARD"]) == cr.MOVE_FORWARD
    assert int(found["BACKWARD"]) == cr.MOVE_BACKWARD
    assert int(found["ROTATE_LEFT"]) == cr.MOVE_ROTATE_LEFT
    assert int(found["ROTATE_RIGHT"]) == cr.MOVE_ROTATE_RIGHT


# ----- router -----

def router(**cfg):
    sink = RecordingSink()
    clock = Clock()
    nav = sink if cfg.get("use_navigation") else None
    return CommandRouter(sink, nav, config=RouterConfig(**cfg),
                         clock=clock), sink, clock


def cmd(text, locs=()):
    return parse_command(text, locs)


def test_stop_works_while_asleep_and_low_confidence():
    r, sink, _ = router()
    res = r.handle(cmd("stop"), awake=False, confidence=0.01)
    assert res.accepted and res.ack == "Stopping."
    assert sink.movement[-1][0] == cr.MOVE_STOP


def test_motion_requires_awake_and_confidence():
    r, sink, _ = router()
    assert r.handle(cmd("go forward"), awake=False).status \
        is RouteStatus.SLEEPING
    assert r.handle(cmd("go forward"), awake=True,
                    confidence=0.2).status is RouteStatus.LOW_CONFIDENCE
    assert sink.movement == []


def test_forward_publishes_with_speed_and_id():
    r, sink, _ = router()
    res = r.handle(cmd("go forward"), awake=True, confidence=0.9)
    assert res.accepted and res.command_id == "voice-1"
    code, lin, ang, valid, emerg, seq, src = sink.movement[0]
    assert (code, lin, ang) == (cr.MOVE_FORWARD, 0.20, 0.0)
    assert valid == 0.30 and not emerg and src == "buddy_voice"
    assert r.tracked[res.command_id].state is CommandState.ACTIVE


def test_turn_uses_rotate_codes():
    r, sink, _ = router()
    r.handle(cmd("turn left"), awake=True)
    assert sink.movement[0][0] == cr.MOVE_ROTATE_LEFT
    assert sink.movement[0][2] == 0.60


def test_tick_republishes_and_expires():
    r, sink, clock = router(max_motion_s=2.0)
    r.handle(cmd("go forward"), awake=True)
    for _ in range(5):
        clock.t += 0.1
        r.tick()
    assert len(sink.movement) == 6
    clock.t += 2.0
    r.tick()
    assert sink.movement[-1][0] == cr.MOVE_STOP
    assert not r.moving
    n = len(sink.movement)
    r.tick()
    assert len(sink.movement) == n


def test_duplicate_motion_not_republished_as_new():
    r, sink, _ = router()
    first = r.handle(cmd("go forward"), awake=True)
    second = r.handle(cmd("go forward"), awake=True)
    assert second.status is RouteStatus.DUPLICATE
    assert second.command_id == first.command_id
    assert len(sink.movement) == 1


def test_new_motion_replaces_old_single_publisher():
    r, sink, _ = router()
    r.handle(cmd("go forward"), awake=True)
    r.handle(cmd("turn left"), awake=True)
    assert r.active_command is VoiceCommand.TURN_LEFT
    sink.movement.clear()
    r.tick()
    assert [m[0] for m in sink.movement] == [cr.MOVE_ROTATE_LEFT]


def test_stop_cancels_active_and_marks_tracking():
    r, sink, _ = router()
    first = r.handle(cmd("go forward"), awake=True)
    r.handle(cmd("stop"), awake=True)
    assert not r.moving
    assert r.tracked[first.command_id].state is CommandState.DONE
    sink.movement.clear()
    r.tick()
    assert sink.movement == []


def test_publish_failure_is_reported_not_raised():
    r, sink, _ = router()
    sink.fail = True
    res = r.handle(cmd("go forward"), awake=True)
    assert res.status is RouteStatus.FAILED
    assert not r.moving
    assert r.publish_errors >= 1


def test_stop_reports_failure_when_publish_fails():
    r, sink, _ = router()
    sink.fail = True
    assert r.stop().status is RouteStatus.FAILED


def test_motion_disabled_blocks_motion_but_not_stop():
    r, sink, _ = router(motion_enabled=False)
    res = r.handle(cmd("go forward"), awake=True)
    assert res.status is RouteStatus.MOTION_DISABLED
    assert r.handle(cmd("stop"), awake=True).accepted


def test_disabling_motion_stops_robot():
    r, sink, _ = router()
    r.handle(cmd("go forward"), awake=True)
    r.set_motion_enabled(False)
    assert sink.movement[-1][0] == cr.MOVE_STOP
    assert not r.moving


def test_navigation_unavailable_by_default():
    r, sink, _ = router()
    res = r.handle(cmd("start patrol"), awake=True)
    assert res.status is RouteStatus.UNAVAILABLE
    assert sink.navigation == [] and sink.movement == []


def test_navigation_mode_routes_patrol_and_goto():
    r, sink, _ = router(use_navigation=True)
    res = r.handle(cmd("start patrol"), awake=True)
    assert res.accepted
    assert sink.navigation[0][0] == cr.NAV_START_PATROL
    res = r.handle(cmd("go to kitchen", ("kitchen",)), awake=True)
    assert sink.navigation[-1][0] == cr.NAV_GO_TO_LOCATION
    assert sink.navigation[-1][4] == "kitchen"
    assert res.command_id == sink.navigation[-1][1]


def test_navigation_mode_stop_goes_to_both_topics():
    r, sink, _ = router(use_navigation=True)
    r.handle(cmd("start patrol"), awake=True)
    r.stop()
    assert sink.movement[-1][0] == cr.MOVE_STOP
    assert sink.navigation[-1][0] == cr.NAV_STOP


def test_navigation_mode_forward_uses_nav_not_movement():
    r, sink, _ = router(use_navigation=True)
    r.handle(cmd("go forward"), awake=True)
    assert sink.navigation[0][0] == cr.NAV_MOVE_FORWARD
    assert sink.movement == []
    r.handle(cmd("go back"), awake=True)
    assert sink.movement[0][0] == cr.MOVE_BACKWARD


def test_pause_without_goal_falls_through_to_chat():
    r, sink, _ = router(use_navigation=True)
    res = r.handle(cmd("pause"), awake=True)
    assert res.status is RouteStatus.NOT_APPLICABLE
    assert sink.navigation == []


def test_nav_status_feedback_updates_tracking():
    r, sink, _ = router(use_navigation=True)
    res = r.handle(cmd("start patrol"), awake=True)
    r.on_navigation_status(res.command_id, "failed", "no trusted pose")
    t = r.tracked[res.command_id]
    assert t.state is CommandState.REJECTED and "pose" in t.detail
    assert r.handle(cmd("cancel"), awake=True).status \
        is RouteStatus.NOT_APPLICABLE


def test_router_never_publishes_to_safe_topic_or_emergency_flag():
    r, sink, _ = router()
    r.handle(cmd("go forward"), awake=True)
    r.stop()
    assert all(m[4] is False for m in sink.movement)


# ----- recogniser adapter -----

class FakeKaldi:
    def __init__(self, final=True, result=None):
        self.final = final
        self.result = result or {}

    def AcceptWaveform(self, data):
        return self.final

    def Result(self):
        return json.dumps(self.result)

    def PartialResult(self):
        return json.dumps({"partial": "go for"})


def make_vosk(result, final=True):
    return VoskRecognizer(
        None, recognizer_factory=lambda m, r, g: FakeKaldi(final, result),
    )


def test_vosk_adapter_extracts_text_and_confidence():
    rec = make_vosk({"text": "go forward", "result": [
        {"word": "go", "conf": 0.9}, {"word": "forward", "conf": 0.7}]})
    out = rec.accept(b"\0\0")
    assert out.text == "go forward"
    assert out.confidence == pytest.approx(0.8)


def test_vosk_adapter_ignores_unk_and_empty_and_partial():
    assert make_vosk({"text": "[unk]"}).accept(b"") is None
    assert make_vosk({"text": ""}).accept(b"") is None
    assert make_vosk({"text": "x"}, final=False).accept(b"") is None
    assert make_vosk({}, final=False).partial() == "go for"


def test_vosk_without_word_confidence():
    assert make_vosk({"text": "hello"}).accept(b"").confidence is None


def test_scripted_recognizer():
    rec = ScriptedRecognizer([Transcript("hi", 0.9)])
    assert rec.accept(b"").text == "hi"
    assert rec.accept(b"") is None


# ----- tts adapters -----

def test_fake_synth_length_scales_with_text_and_style():
    s = FakeSynthesizer()
    a, rate = s.synthesize("hi", SpeechStyle())
    b, _ = s.synthesize("hello there friend", SpeechStyle())
    c, _ = s.synthesize("hi", SpeechStyle(length_scale=2.0))
    assert b.size > a.size and c.size > a.size and rate == 22050


def test_piper_adapter_reads_wav_from_voice():
    class Voice:
        def synthesize_wav(self, text, wav, **kw):
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(22050)
            wav.writeframes(np.arange(100, dtype=np.int16).tobytes())

    samples, rate = PiperSynthesizer("x.onnx", voice=Voice()).synthesize(
        "hi", SpeechStyle())
    assert rate == 22050 and samples.size == 100


class FakeProc:
    def __init__(self):
        self.code = None
        self.terminated = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = -15

    def wait(self, timeout=None):
        return self.code

    def kill(self):
        self.code = -9


def test_aplay_playback_lifecycle_and_gain(tmp_path):
    procs = []

    def popen(cmd, **kw):
        p = FakeProc()
        p.cmd = cmd
        procs.append(p)
        return p

    pb = AplayPlayback(device="plughw:9,9", popen=popen)
    pb.start(np.full(100, 10000, np.int16), 22050, 0.5)
    assert pb.is_playing()
    assert procs[0].cmd[:4] == ["aplay", "-q", "-D", "plughw:9,9"]
    path = procs[0].cmd[-1]
    import wave
    with wave.open(path) as w:
        data = np.frombuffer(w.readframes(100), np.int16)
    assert data[0] == 5000
    pb.stop()
    assert procs[0].terminated and not pb.is_playing()
    assert not Path(path).exists()


def test_aplay_default_device_has_no_hardcoded_card():
    procs = []
    pb = AplayPlayback(popen=lambda cmd, **kw: procs.append(cmd) or
                       FakeProc())
    pb.start(np.zeros(10, np.int16), 16000, 1.0)
    assert "-D" not in procs[0]
    pb.stop()


# ----- vision -----

def test_mock_vision_and_freshness():
    clock = Clock()
    v = MockVision(clock)
    assert v.latest() is None
    v.set(person_visible=True, facial_expression="happy")
    snap = v.latest()
    assert fresh(snap, clock.t + 1) is snap
    assert fresh(snap, clock.t + 10) is None
    assert fresh(None, 0) is None

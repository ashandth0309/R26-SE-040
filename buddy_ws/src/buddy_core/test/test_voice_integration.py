"""Phase 8 (pure-Python half): factory, runner, profiles, mic loss."""

import random
from pathlib import Path

import numpy as np
import pytest

from buddy_core.voice import command_router as cr
from buddy_core.voice.audio_devices import MicTopology
from buddy_core.voice.audio_manager import FrameChunker, SimulatedSource
from buddy_core.voice.command_router import RecordingSink
from buddy_core.voice.config import (
    VoiceSettings,
    apply_profile,
    load_settings,
)
from buddy_core.voice.engine_factory import (
    ModelMissing,
    VoiceBackends,
    alsa_name_from_device,
    build_companion,
    real_backends,
)
from buddy_core.voice.llm_router import RouterMode
from buddy_core.voice.runner import EngineRunner
from buddy_core.voice.stt import ScriptedRecognizer, Transcript
from buddy_core.voice.tts_backends import FakeSynthesizer

RATE = 16000
BLOCK = 320


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class Speaker:
    def __init__(self, clock):
        self.clock = clock
        self.until = 0.0
        self.played = 0

    def start(self, samples, rate, gain):
        self.until = self.clock() + samples.size / rate
        self.played += 1

    def is_playing(self):
        return self.clock() < self.until

    def stop(self):
        self.until = 0.0


def tone(seconds, amp=0.3):
    n = int(seconds * RATE)
    t = np.arange(n) / RATE

    return (amp * 32767 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def hush(seconds, seed=0):
    rng = np.random.default_rng(seed)

    return (rng.standard_normal(int(seconds * RATE)) * 0.002 * 32767
            ).astype(np.int16)


def make(settings=None, **kw):
    clock = Clock()
    settings = settings or VoiceSettings()
    src = SimulatedSource("mic", clock=clock)
    backends = VoiceBackends(
        mic_sources=[src], topology=MicTopology.SINGLE,
        playback=Speaker(clock), synthesizer=FakeSynthesizer(),
        wake_rec=ScriptedRecognizer(), command_rec=ScriptedRecognizer(),
        general_rec=ScriptedRecognizer())
    sink = RecordingSink()
    comp = build_companion(settings, backends, sink, clock=clock,
                           rng=random.Random(1), **kw)
    runner = EngineRunner(comp, clock=clock, max_frames_per_step=1,
                          sleep=lambda s: None)

    return comp, runner, src, sink, backends, clock


def drive(runner, clock, src, audio):
    src.feed(audio)

    for _ in range(audio.size // BLOCK):
        clock.advance(0.02)
        runner.step()


def phrase(runner, clock, src, seconds=1.0, tail=1.2):
    drive(runner, clock, src, tone(seconds))
    drive(runner, clock, src, hush(tail, seed=3))


def started(settings=None, **kw):
    out = make(settings, **kw)
    comp, runner, src, sink, backends, clock = out
    comp.mics.start()
    drive(runner, clock, src, hush(1.0))

    return out


def T(text, conf=0.9):
    return Transcript(text, conf)


# ----- chunker -----

def test_chunker_reblocks_without_loss():
    ch = FrameChunker(320)
    data = np.arange(1000, dtype=np.int16)
    out = ch.push(data[:150]) + ch.push(data[150:700]) + ch.push(data[700:])
    assert all(f.size == 320 for f in out) and len(out) == 3
    assert np.array_equal(np.concatenate(out), data[:960])
    assert ch.push(np.zeros(280, np.int16))[0][:40].tolist() == \
        data[960:1000].tolist()


# ----- factory / profiles -----

def test_default_build_is_offline_and_consent_free():
    comp, *_ = make()
    status = comp.status()
    assert status["cloud"] == "disabled" and status["llm_mode"] == "offline"
    assert not status["memory_consent"] and status["state"] == \
        "LISTENING_FOR_WAKE"
    assert "mic_status" in status
    assert not any(isinstance(v, str) and "hey" in v
                   for v in status.values())


def test_lightweight_profile_turns_heavy_features_off():
    s = VoiceSettings(profile="lightweight")
    s.online.local_llm = True
    s.online.cloud_enabled = True
    apply_profile(s)
    comp, *_ = make(s)
    assert comp.engine.utterance_observer is None
    assert not comp.engine.backchannel.config.enabled
    assert comp.suggester is None
    assert comp.llm.mode is RouterMode.OFFLINE


def test_enhanced_profile_uses_local_llm_when_configured():
    s = VoiceSettings(profile="enhanced")
    s.online.local_llm = True
    apply_profile(s)
    comp, *_ = make(s)
    assert comp.llm.mode is RouterMode.LOCAL
    assert comp.engine.utterance_observer is not None


def test_unknown_profile_falls_back_to_balanced():
    assert apply_profile(VoiceSettings(profile="turbo")).profile == \
        "balanced"


def test_cloud_needs_endpoint_and_user_confirmation():
    s = VoiceSettings()
    s.online.cloud_enabled = True
    comp, *_ = make(s)
    assert not comp.cloud_state["on"]
    s.online.cloud_base_url = "https://example.invalid/v1"
    s.online.cloud_model = "m"
    comp2, *_ = make(s)
    assert comp2.cloud_state["on"] and comp2.llm.mode is RouterMode.AUTO
    comp2.controls.set_cloud(False)
    assert comp2.llm.mode is RouterMode.OFFLINE
    assert comp2.connectivity.link.value == "disabled"


def test_navigation_needs_both_setting_and_sink():
    s = VoiceSettings()
    s.navigation.enabled = True
    comp, *_ = make(s)
    assert not comp.router.navigation_available
    comp2, *_ = make(s, navigation_sink=RecordingSink())
    assert comp2.router.navigation_available
    comp3, *_ = make(VoiceSettings(), navigation_sink=RecordingSink())
    assert not comp3.router.navigation_available


def test_shipped_yaml_sections_load():
    path = Path(__file__).resolve().parents[1] / "config" / \
        "voice_companion.yaml"
    s = load_settings(path)
    assert not s.online.cloud_enabled and not s.navigation.enabled
    assert s.router.max_motion_s == 8.0 and s.models.vosk_model_path == ""


def test_real_backends_fail_clearly_without_models():
    with pytest.raises(ModelMissing) as exc:
        real_backends(VoiceSettings())
    assert "vosk_model_path" in str(exc.value)


def test_alsa_name_parsing():
    assert alsa_name_from_device("USB Audio: - (hw:2,0)") == "plughw:2,0"
    assert alsa_name_from_device("default") is None


# ----- end to end with fakes -----

def test_full_conversation_path_wake_move_stop():
    comp, runner, src, sink, be, clock = started()
    be.wake_rec.finals = [T("hey buddy")]
    phrase(runner, clock, src)
    assert comp.turn.awake
    drive(runner, clock, src, hush(1.0, seed=9))
    be.command_rec.finals = [T("go forward")]
    be.general_rec.finals = [T("go forward")]
    phrase(runner, clock, src)
    assert cr.MOVE_FORWARD in [m[0] for m in sink.movement]
    be.command_rec.finals = [T("stop")]
    phrase(runner, clock, src)
    assert sink.movement[-1][0] == cr.MOVE_STOP
    assert not comp.router.moving


def test_command_observer_receives_parsed_command_and_result():
    seen = []
    comp, runner, src, sink, be, clock = started(
        command_observer=lambda cmd, res, tr: seen.append(
            (cmd.kind.value, res.status.value, tr.text)))
    comp.turn.wake()
    be.command_rec.finals = [T("turn left")]
    be.general_rec.finals = [T("turn left")]
    phrase(runner, clock, src)
    assert seen == [("turn_left", "accepted", "turn left")]


def test_memory_stays_empty_without_consent_through_the_pipeline():
    comp, runner, src, sink, be, clock = started()
    comp.turn.wake()
    be.general_rec.finals = [T("my name is Yohan")]
    phrase(runner, clock, src)
    assert comp.memory.all_facts() == []
    assert comp.engine.context.user_name == "Yohan"


def test_stored_name_is_loaded_at_startup(tmp_path):
    from buddy_core.voice.memory_store import CompanionMemoryManager

    mem = CompanionMemoryManager()
    mem.grant_consent()
    mem.remember("name", "asha", "name")
    comp, *_ = make(memory=mem)
    assert comp.engine.context.user_name == "Asha"


# ----- safety around the loop -----

def test_losing_every_microphone_stops_a_moving_robot():
    comp, runner, src, sink, be, clock = started()
    comp.turn.wake()
    be.command_rec.finals = [T("go forward")]
    be.general_rec.finals = [T("go forward")]
    phrase(runner, clock, src)
    assert comp.router.moving
    src.break_device()

    for _ in range(100):
        clock.advance(0.02)
        runner.step()

    assert not comp.router.moving
    assert sink.movement[-1][0] == cr.MOVE_STOP
    assert comp.status()["mic_active"] is None


def test_loop_thread_survives_component_exceptions():
    import threading
    import time as _t

    comp, runner, src, sink, be, clock = started()

    def boom(frame):
        raise RuntimeError("engine bug")

    comp.engine.process_frame = boom
    src.feed(tone(0.5))
    thread = threading.Thread(target=runner._loop, daemon=True)
    thread.start()

    for _ in range(200):
        if runner.loop_errors >= 3:
            break

        _t.sleep(0.005)

    assert thread.is_alive() and runner.loop_errors >= 3
    runner._stop.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_inbox_work_runs_on_engine_step():
    comp, runner, src, sink, be, clock = started()
    seen = []
    runner.submit(lambda: seen.append("ran"))
    runner.step()
    assert seen == ["ran"]


def test_inbox_exception_is_contained():
    comp, runner, src, sink, be, clock = started()
    runner.submit(lambda: 1 / 0)
    runner.step()
    assert runner.loop_errors == 1


def test_shutdown_stops_robot_and_devices():
    comp, runner, src, sink, be, clock = started()
    comp.turn.wake()
    be.command_rec.finals = [T("go forward")]
    be.general_rec.finals = [T("go forward")]
    phrase(runner, clock, src)
    runner.shutdown()
    assert sink.movement[-1][0] == cr.MOVE_STOP
    assert src.status.value == "stopped"


def test_background_thread_starts_warms_cache_and_stops():
    comp, runner, src, sink, be, clock = make()
    real_runner = EngineRunner(comp, clock=clock, sleep=lambda s: None)
    real_runner.start()
    assert comp.engine.synth.calls
    real_runner.submit(lambda: sink.movement.append("marker"))

    for _ in range(200):
        if "marker" in sink.movement:
            break

        import time as _t
        _t.sleep(0.005)

    real_runner.shutdown()
    assert "marker" in sink.movement
    assert not real_runner._thread.is_alive()


def test_proactive_suggestion_spoken_when_idle_and_long_awake():
    comp, runner, src, sink, be, clock = started()
    comp.turn.wake()
    runner._awake_since = clock() - 100 * 60
    runner._hour = lambda: 15
    drive(runner, clock, src, hush(30.0, seed=5))
    spoken = [t for _, k, t in comp.engine.events if k == "speak"]
    assert any("water" in s or "stretch" in s for s in spoken)


def test_proactive_never_when_muted_or_quiet():
    comp, runner, src, sink, be, clock = started()
    comp.turn.wake()
    comp.engine.set_quiet(True)
    runner._awake_since = clock() - 100 * 60
    runner._proactive(clock())
    assert not [1 for _, k, _ in comp.engine.events if k == "speak"]


def test_status_has_no_speech_text_or_audio():
    comp, runner, src, sink, be, clock = started()
    comp.turn.wake()
    be.general_rec.finals = [T("my secret thought")]
    phrase(runner, clock, src)
    assert "secret" not in str(comp.status())

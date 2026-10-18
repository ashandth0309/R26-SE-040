"""Phase 2 tests: discovery, microphones, VAD, speaker, echo.

All audio is synthetic.  No hardware is touched.
"""

import numpy as np
import pytest

from buddy_core.voice.audio_devices import (
    DeviceInfo,
    MicTopology,
    discover_inputs,
    discover_output,
)
from buddy_core.voice.audio_manager import (
    AudioFrame,
    ManagerConfig,
    MicrophoneManager,
    SimulatedSource,
    SoundDeviceSource,
    SourceQuality,
    SourceStatus,
    build_sources,
)
from buddy_core.voice.config import load_settings
from buddy_core.voice.dsp import dbfs, resample, rms
from buddy_core.voice.echo import EchoConfig, EchoGuard
from buddy_core.voice.speaker_manager import (
    Priority,
    SpeakerConfig,
    SpeakerManager,
)
from buddy_core.voice.vad import VadConfig, VoiceActivityDetector

RATE = 16000
BLOCK = 320


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def dev(index, name, ins=1, outs=0, rate=48000, hostapi="ALSA"):
    return DeviceInfo(
        index=index, name=name, max_input_channels=ins,
        max_output_channels=outs, default_samplerate=rate,
        default_low_input_latency=0.01, default_low_output_latency=0.01,
        hostapi=hostapi,
    )


class FakeBackend:
    def __init__(self, devices, bad=()):
        self.devices = devices
        self.bad = set(bad)

    def query_devices(self):
        return self.devices

    def supports_input(self, index, samplerate, channels):
        return index not in self.bad

    def supports_output(self, index, samplerate, channels):
        return index not in self.bad


def tone(seconds, amp=0.3, freq=220.0, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def noise(seconds, amp=0.003, seed=0, rate=RATE):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * rate)) * amp * 32767
            ).astype(np.int16)


def blocks(samples, size=BLOCK):
    for i in range(0, samples.size - size + 1, size):
        yield samples[i:i + size]


# ----- discovery -----

def test_two_distinct_devices_are_two_topology():
    backend = FakeBackend([
        dev(0, "USB Mic A"), dev(1, "USB Mic B"), dev(2, "Spk", 0, 2),
    ])
    sel = discover_inputs(backend)
    assert sel.topology is MicTopology.TWO_DEVICES
    assert len(sel.inputs) == 2


def test_stereo_single_device():
    backend = FakeBackend([dev(0, "AB13X USB Audio", ins=2)])
    sel = discover_inputs(backend)
    assert sel.topology is MicTopology.STEREO


def test_single_mic_fallback():
    backend = FakeBackend([dev(0, "Only Mic", ins=1)])
    sel = discover_inputs(backend)
    assert sel.topology is MicTopology.SINGLE
    assert sel.usable


def test_no_inputs():
    sel = discover_inputs(FakeBackend([dev(0, "Spk", 0, 2)]))
    assert sel.topology is MicTopology.NONE
    assert not sel.usable


def test_virtual_devices_excluded_by_default():
    backend = FakeBackend([dev(0, "Loopback Monitor"), dev(1, "Real Mic")])
    sel = discover_inputs(backend)
    assert [c.device.name for c in sel.inputs] == ["Real Mic"]


def test_preferred_override_and_missing_note():
    backend = FakeBackend([dev(0, "Mic A"), dev(1, "Mic B")])
    sel = discover_inputs(backend, preferred=["Mic B", "Ghost"],
                          max_microphones=1)
    assert sel.inputs[0].device.name == "Mic B"
    assert any("Ghost" in n for n in sel.notes)


def test_unsupported_device_is_skipped():
    backend = FakeBackend([dev(0, "Bad Mic"), dev(1, "Good Mic")], bad={0})
    sel = discover_inputs(backend)
    assert [c.device.name for c in sel.inputs] == ["Good Mic"]


def test_output_discovery_and_invalid_preference():
    backend = FakeBackend([dev(0, "Mic"), dev(1, "Speaker", 0, 2)])
    out = discover_output(backend, preferred="Nope")
    assert out is not None
    assert out.device.name == "Speaker"
    assert discover_output(FakeBackend([dev(0, "Mic")])) is None


# ----- dsp -----

def test_resample_length_and_antialias():
    high = tone(0.5, amp=0.5, freq=20000, rate=48000)
    out = resample(high.astype(np.float32) / 32768.0, 48000, 16000)
    assert abs(out.size - 8000) <= 2
    naive = high[::3].astype(np.float32) / 32768.0
    # 20 kHz would alias into the band with naive decimation.
    assert rms(out) < 0.2 * rms(naive)


# ----- VAD -----

def run_vad(vad, samples):
    return [vad.process(b) for b in blocks(samples)]


def test_vad_detects_speech_and_silence():
    vad = VoiceActivityDetector(VadConfig())
    audio = np.concatenate([noise(1.0), tone(1.0), noise(1.5)])
    events = run_vad(vad, audio)
    assert any(e.speech_started for e in events)
    assert any(e.speech_ended for e in events)


def test_vad_quiet_noise_is_not_speech():
    vad = VoiceActivityDetector(VadConfig())
    events = run_vad(vad, noise(3.0))
    assert not any(e.speech_started for e in events)


def test_vad_ignores_short_click():
    vad = VoiceActivityDetector(VadConfig())
    audio = np.concatenate([noise(1.0), tone(0.04), noise(1.0)])
    events = run_vad(vad, audio)
    assert not any(e.speech_ended and e.utterance_ms > 120 for e in events)


def test_vad_adapts_to_louder_noise_floor():
    vad = VoiceActivityDetector(VadConfig())
    events = run_vad(vad, noise(14.0, amp=0.02))
    assert vad.noise_floor_dbfs > -45
    # once adapted, the same noise is no longer called speech
    late = run_vad(vad, noise(3.0, amp=0.02, seed=5))
    assert not any(e.speech_started for e in late)
    assert events


# ----- microphone manager -----

def make_sources(clock, a, b):
    sa = SimulatedSource("A", a, clock=clock)
    sb = SimulatedSource("B", b, clock=clock)
    return sa, sb


def drive(mgr, clock, steps):
    out = []
    for _ in range(steps):
        clock.advance(0.02)
        out.append(mgr.pump())
    return out


def test_manager_requires_a_source():
    with pytest.raises(ValueError):
        MicrophoneManager([], topology=MicTopology.NONE)


def speech_like(seconds, floor, loud, seed):
    n = int(seconds * RATE)
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n) * floor * 32767
    gate = (np.arange(n) // 3200) % 2
    burst = tone(seconds, amp=loud)[:n] * gate
    return (base + burst).astype(np.int16)


def test_manager_selects_better_source_and_never_sums():
    clock = Clock()
    good = speech_like(10, 0.002, 0.3, 1)
    bad = speech_like(10, 0.05, 0.1, 2)
    sa, sb = make_sources(clock, bad, good)
    mgr = MicrophoneManager(
        [sa, sb], topology=MicTopology.TWO_DEVICES,
        config=ManagerConfig(min_dwell_s=0.5), clock=clock,
    )
    mgr.start()
    frames = [f for f in drive(mgr, clock, 400) if f is not None]
    assert mgr.active == "B"
    assert not mgr.fusion_active
    # output frames are exactly one source's samples, not a mixture
    assert all(f.source in ("A", "B") for f in frames)


def test_fusion_refused_for_independent_devices():
    clock = Clock()
    sa, sb = make_sources(clock, tone(1), tone(1))
    mgr = MicrophoneManager(
        [sa, sb], topology=MicTopology.TWO_DEVICES,
        config=ManagerConfig(fusion="sum", fusion_geometry_validated=True),
        clock=clock,
    )
    assert not mgr.fusion_active


def test_no_flapping_with_equal_sources():
    clock = Clock()
    a = speech_like(10, 0.002, 0.3, 1)
    b = speech_like(10, 0.002, 0.3, 1)
    sa, sb = make_sources(clock, a, b)
    mgr = MicrophoneManager(
        [sa, sb], topology=MicTopology.TWO_DEVICES, clock=clock,
    )
    mgr.start()
    seen = set()
    for _ in range(400):
        clock.advance(0.02)
        mgr.pump()
        seen.add(mgr.active)
    assert len(seen) == 1


def test_failover_when_active_source_dies():
    clock = Clock()
    sa, sb = make_sources(clock, tone(10), tone(10))
    mgr = MicrophoneManager(
        [sa, sb], topology=MicTopology.TWO_DEVICES, clock=clock,
    )
    mgr.start()
    drive(mgr, clock, 10)
    dead = mgr.active
    mgr.sources[dead].break_device()
    frames = drive(mgr, clock, 10)
    assert mgr.active != dead
    assert any(f is not None for f in frames)


def test_single_mic_fallback_runs():
    clock = Clock()
    sa = SimulatedSource("only", tone(2), clock=clock)
    mgr = MicrophoneManager(
        [sa], topology=MicTopology.SINGLE, clock=clock,
    )
    mgr.start()
    frames = drive(mgr, clock, 20)
    assert mgr.active == "only"
    assert any(f is not None for f in frames)


def test_all_sources_dead_gives_none():
    clock = Clock()
    sa = SimulatedSource("only", tone(2), clock=clock)
    mgr = MicrophoneManager(
        [sa], topology=MicTopology.SINGLE, clock=clock,
    )
    mgr.start()
    sa.break_device()
    assert mgr.pump() is None
    assert mgr.active is None


def test_quality_penalises_clipping():
    q_ok, q_clip = SourceQuality(), SourceQuality()
    for block in blocks(speech_like(3, 0.002, 0.3, 3)):
        q_ok.update(AudioFrame(block, 0.0, "x"))
    for block in blocks(speech_like(3, 0.002, 0.3, 3)):
        q_clip.update(AudioFrame(np.clip(block.astype(np.int32) * 40,
                                         -32768, 32767).astype(np.int16),
                                 0.0, "x"))
    assert q_ok.score() > q_clip.score()


def test_drift_estimate_detects_rate_error():
    q = SourceQuality()
    t = 0.0
    for _ in range(300):
        # 1 percent too slow
        t += BLOCK / RATE * 1.01
        q.update(AudioFrame(np.zeros(BLOCK, np.int16), t, "x"))
    ppm = q.rate_error_ppm()
    assert ppm is not None and -12000 < ppm < -8000


# ----- sound device source -----

class FakeStream:
    def __init__(self, fail_start=False):
        self.fail_start = fail_start
        self.started = False
        self.closed = False

    def start(self):
        if self.fail_start:
            raise OSError("device gone")
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


def make_choice(rate=48000, channels=2, channel=0):
    from buddy_core.voice.audio_devices import InputChoice

    return InputChoice(dev(0, "Mic", ins=channels), channel, rate,
                       channels, 0.01)


def test_sounddevice_source_converts_and_resamples():
    clock = Clock()
    holder = {}

    def factory(choice, cb):
        holder["cb"] = cb
        return FakeStream()

    src = SoundDeviceSource(make_choice(), stream_factory=factory,
                            clock=clock)
    assert src.start()
    data = tone(0.02, rate=48000)
    stereo = np.stack([data, data], axis=1)
    holder["cb"](stereo, len(data), None, None)
    frame = src.read()
    assert frame is not None
    assert frame.samples.dtype == np.int16
    assert abs(frame.samples.size - 320) <= 2


def test_sounddevice_source_recovers_with_backoff():
    clock = Clock()
    streams = []

    def factory(choice, cb):
        s = FakeStream(fail_start=len(streams) in (1, 2))
        streams.append(s)
        return s

    src = SoundDeviceSource(make_choice(), stream_factory=factory,
                            clock=clock, stall_timeout_s=1.0)
    assert src.start()
    clock.advance(2.0)
    src.maintain(clock())            # stall -> restart fails
    assert src.status is SourceStatus.RECOVERING
    src.maintain(clock())            # too early to retry
    assert len(streams) == 2
    clock.advance(0.6)
    src.maintain(clock())            # second failure, backoff doubles
    clock.advance(1.1)
    src.maintain(clock())            # now succeeds
    assert src.status is SourceStatus.OK
    assert src.restarts >= 2


def test_start_failure_marks_failed():
    src = SoundDeviceSource(
        make_choice(), stream_factory=lambda c, cb: FakeStream(True),
    )
    assert not src.start()
    assert src.status is SourceStatus.FAILED


def test_queue_overflow_is_counted_not_fatal():
    holder = {}

    def factory(choice, cb):
        holder["cb"] = cb
        return FakeStream()

    src = SoundDeviceSource(make_choice(1, 1, None), stream_factory=factory,
                            queue_blocks=2)
    src.start()
    for _ in range(5):
        holder["cb"](np.zeros((16, 1), np.int16), 16, None, None)
    assert src.overflows == 3


def test_build_sources_names_unique():
    backend = FakeBackend([dev(0, "Mic A"), dev(1, "Mic B")])
    sel = discover_inputs(backend)
    names = [s.name for s in build_sources(sel,
                                           stream_factory=lambda c, cb: None)]
    assert len(set(names)) == 2


# ----- speaker -----

class FakeSpeaker:
    def __init__(self, clock, fail=0):
        self.clock = clock
        self.fail = fail
        self.until = 0.0
        self.log = []
        self.max_concurrent = 0

    def start(self, samples, rate, gain):
        if self.fail > 0:
            self.fail -= 1
            raise OSError("no device")
        if self.is_playing():
            self.max_concurrent = 2
        else:
            self.max_concurrent = max(self.max_concurrent, 1)
        self.until = self.clock() + samples.size / rate
        self.log.append(("start", samples.size, gain))

    def is_playing(self):
        return self.clock() < self.until

    def stop(self):
        self.until = 0.0
        self.log.append(("stop",))


def pcm(seconds):
    return np.ones(int(RATE * seconds), np.int16)


def make_speaker(fail=0, fallback=None, **cfg):
    clock = Clock()
    be = FakeSpeaker(clock, fail)
    sm = SpeakerManager(be, config=SpeakerConfig(**cfg), clock=clock,
                        fallback=fallback)
    return sm, be, clock


def test_speaker_plays_in_order_without_overlap():
    sm, be, clock = make_speaker()
    sm.say("one", pcm(1), RATE)
    sm.say("two", pcm(1), RATE)
    for _ in range(50):
        sm.tick()
        clock.advance(0.1)
    assert [e for e in be.log if e[0] == "start"].__len__() == 2
    assert be.max_concurrent == 1


def test_speaker_priority_order():
    sm, be, clock = make_speaker()
    sm.say("first", pcm(1), RATE, priority=Priority.NORMAL)
    sm.tick()
    sm.say("low-prio chat", pcm(0.5), RATE, priority=Priority.NORMAL)
    sm.say("answer", pcm(0.3), RATE, priority=Priority.HIGH)
    order = []
    sm._on_start = lambda u: order.append(u.text)
    for _ in range(60):
        clock.advance(0.1)
        sm.tick()
    assert order == ["answer", "low-prio chat"]


def test_critical_interrupts_and_clears_queue():
    sm, be, clock = make_speaker()
    sm.say("long", pcm(5), RATE)
    sm.say("queued", pcm(1), RATE)
    sm.tick()
    sm.say("Stopping.", pcm(0.5), RATE, priority=Priority.CRITICAL)
    assert not sm.speaking
    sm.tick()
    assert sm.current.text == "Stopping."
    assert sm.queued == 0
    assert be.max_concurrent == 1


def test_low_priority_dropped_when_busy():
    sm, be, clock = make_speaker()
    sm.say("talk", pcm(1), RATE)
    sm.tick()
    assert sm.say("mm-hm", pcm(0.3), RATE, priority=Priority.LOW) is None
    assert sm.dropped == 1


def test_cancel_all_stops_immediately():
    sm, be, clock = make_speaker()
    sm.say("talk", pcm(3), RATE)
    sm.tick()
    sm.cancel_all()
    assert not sm.speaking
    assert ("stop",) in be.log


def test_volume_cap_applied():
    sm, be, clock = make_speaker(max_volume=0.5)
    sm.say("loud", pcm(0.2), RATE, volume=1.0)
    sm.tick()
    assert be.log[0][2] == 0.5


def test_muted_speaker_accepts_nothing():
    sm, be, clock = make_speaker()
    sm.say("talk", pcm(2), RATE)
    sm.tick()
    sm.mute()
    assert not sm.speaking
    assert sm.say("more", pcm(1), RATE) is None


def test_queue_limit_drops_lowest():
    sm, be, clock = make_speaker(max_queue=2)
    sm.say("busy", pcm(1), RATE)
    sm.tick()
    sm.say("a", pcm(1), RATE, priority=Priority.NORMAL)
    sm.say("b", pcm(1), RATE, priority=Priority.NORMAL)
    assert sm.say("c", pcm(1), RATE, priority=Priority.NORMAL) is None
    assert sm.say("urgent", pcm(1), RATE, priority=Priority.HIGH) is not None
    assert sm.queued == 2


def test_backend_failure_recovers_and_falls_back():
    clock = Clock()
    bad = FakeSpeaker(clock, fail=99)
    good = FakeSpeaker(clock)
    sm = SpeakerManager(bad, config=SpeakerConfig(max_start_failures=1),
                        clock=clock, fallback=good)
    for text in ("a", "b", "c", "d"):
        sm.say(text, pcm(0.2), RATE)
    for _ in range(40):
        clock.advance(0.1)
        sm.tick()
    assert any(e[0] == "start" for e in good.log)


def test_total_backend_failure_does_not_crash_or_hang():
    clock = Clock()
    bad = FakeSpeaker(clock, fail=99)
    sm = SpeakerManager(bad, config=SpeakerConfig(max_start_failures=1),
                        clock=clock)
    for _ in range(5):
        sm.say("x", pcm(0.2), RATE)
        sm.tick()
        clock.advance(0.1)
    assert sm.backend_failed
    assert not sm.speaking


# ----- echo -----

def feed_playback(guard, level_amp):
    guard.playback_level(tone(0.02, amp=level_amp))


def test_guard_blocks_during_and_after_speech():
    clock = Clock()
    g = EchoGuard(clock=clock)
    assert not g.blocked
    g.playback_started()
    assert g.blocked
    g.playback_stopped()
    assert g.blocked
    clock.advance(0.5)
    assert not g.blocked


def test_own_speech_leakage_never_triggers_barge_in():
    clock = Clock()
    g = EchoGuard(clock=clock)
    g.playback_started()
    for _ in range(200):
        feed_playback(g, 0.3)
        # mic hears Buddy at about the default leakage level
        leaked = tone(0.02, amp=0.3 * 10 ** (-12 / 20))
        assert not g.observe(leaked)
    assert g.barge_in_count == 0


def test_learned_leakage_adapts_to_louder_room():
    clock = Clock()
    g = EchoGuard(clock=clock)
    g.playback_started()
    for _ in range(300):
        feed_playback(g, 0.3)
        g.observe(tone(0.02, amp=0.3 * 10 ** (-4 / 20)))
    assert g.leakage_db > -8


def test_user_talking_over_buddy_is_barge_in():
    clock = Clock()
    g = EchoGuard(clock=clock)
    g.playback_started()
    hit = False
    for _ in range(30):
        feed_playback(g, 0.1)
        hit = g.observe(tone(0.02, amp=0.6)) or hit
    assert hit


def test_barge_in_can_be_disabled():
    g = EchoGuard(EchoConfig(barge_in_enabled=False), clock=Clock())
    g.playback_started()
    for _ in range(30):
        feed_playback(g, 0.1)
        assert not g.observe(tone(0.02, amp=0.6))


def test_observe_ignored_when_not_speaking():
    g = EchoGuard(clock=Clock())
    assert not g.observe(tone(0.02, amp=0.9))


def test_buddy_interrupted_during_tts_end_to_end():
    clock = Clock()
    be = FakeSpeaker(clock)
    guard = EchoGuard(clock=clock)
    sm = SpeakerManager(
        be, clock=clock,
        on_start=lambda u: guard.playback_started(),
        on_finish=lambda u, ok: guard.playback_stopped(),
    )
    sm.say("long answer", pcm(5), RATE)
    sm.tick()
    assert guard.blocked
    stopped = False
    for _ in range(30):
        feed_playback(guard, 0.1)
        if guard.observe(tone(0.02, amp=0.6)):
            sm.cancel_all()
            stopped = True
            break
    assert stopped and not sm.speaking
    assert dbfs(rms(tone(0.02))) > -40


# ----- config -----

def test_config_defaults_without_file():
    s = load_settings(None)
    assert s.mics.fusion == "off"
    assert s.speaker.max_volume <= 1.0


def test_config_overrides_and_ignores_unknown(tmp_path):
    f = tmp_path / "v.yaml"
    f.write_text("speaker:\n  max_volume: 0.4\n  bogus: 1\n"
                 "microphones:\n  min_dwell_s: 9\n")
    s = load_settings(f)
    assert s.speaker.max_volume == 0.4
    assert s.mics.min_dwell_s == 9


def test_config_bad_file_falls_back(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text(": : :\n\t- [")
    assert load_settings(f).speaker.max_volume == 0.8


def test_shipped_config_loads():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "config" / \
        "voice_companion.yaml"
    s = load_settings(path)
    assert s.profile == "balanced"

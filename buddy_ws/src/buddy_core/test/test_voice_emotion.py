"""Phase 5 tests: audio features, arousal, fusion and responses.

Audio is synthetic.  These tests check the mechanics (pitch tracking,
baselines, caps, overrides, wording rules).  They say nothing about
accuracy on real speech, which has not been measured.
"""

import builtins
import random

import numpy as np
import pytest

from buddy_core.voice.dialogue_manager import (
    ConversationContext,
    DialogueManager,
)
from buddy_core.voice.emotion_audio import (
    MAX_AUDIO_CONFIDENCE,
    AudioEmotion,
    AudioEmotionEstimator,
    evaluate,
    extract_features,
    mfcc,
    track_pitch,
)
from buddy_core.voice.emotion_fusion import (
    EmotionFusion,
    FusionConfig,
    text_emotion,
)
from buddy_core.voice.emotional_response import (
    CORRECTIONS,
    CRISIS_REPLY,
    REPLIES,
    EmotionalResponseManager,
    make_utterance_observer,
)
from buddy_core.voice.tts_backends import SpeechStyle

RATE = 16000


def voice(f0=150.0, amp=0.1, syl=3.0, seconds=2.0, noise=0.0, seed=0):
    """Harmonic source with syllable-like amplitude bursts."""

    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    env = 0.5 * (1 + np.sin(2 * np.pi * syl * t - np.pi / 2)) ** 1.5
    sig = sig / np.abs(sig).max() * env
    rng = np.random.default_rng(seed)
    sig = sig * amp + rng.standard_normal(n) * noise
    return (np.clip(sig, -1, 1) * 32767).astype(np.int16)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# ----- features -----

def test_short_or_quiet_audio_is_unusable():
    assert extract_features(voice(seconds=0.3)) is None
    assert extract_features(voice(amp=0.0005)) is None
    assert extract_features(voice(), rate=8000) is None


@pytest.mark.parametrize("f0", [110.0, 150.0, 220.0, 300.0])
def test_pitch_tracking_accuracy(f0):
    feats = extract_features(voice(f0=f0))
    assert feats.pitch_median_hz == pytest.approx(f0, rel=0.04)
    assert feats.voiced_ratio > 0.3


def test_noise_has_no_pitch():
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(RATE * 2) * 0.1 * 32767).astype(np.int16)
    feats = extract_features(noise)
    assert feats.pitch_median_hz is None
    assert feats.voiced_ratio < 0.2
    pitch, voiced = track_pitch(noise.astype(np.float32) / 32768)
    assert voiced.mean() < 0.2


def test_loudness_orders_correctly():
    quiet = extract_features(voice(amp=0.05))
    loud = extract_features(voice(amp=0.4))
    assert loud.rms_db > quiet.rms_db + 10


def test_rate_proxy_orders_correctly():
    slow = extract_features(voice(syl=2.0, seconds=3.0))
    fast = extract_features(voice(syl=5.0, seconds=3.0))
    assert fast.rate_per_s > slow.rate_per_s
    assert slow.rate_per_s == pytest.approx(2.0, abs=1.0)


def test_mfcc_shape_finite_and_discriminative():
    a = mfcc(voice(f0=120).astype(np.float32) / 32768)
    b = mfcc(voice(f0=300).astype(np.float32) / 32768)
    assert a.shape[1] == 13 and np.isfinite(a).all()
    assert not np.allclose(a.mean(0), b.mean(0), atol=0.5)
    assert len(extract_features(voice()).mfcc_mean) == 13


def test_features_never_touch_files(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("audio must not be written")

    monkeypatch.setattr(builtins, "open", boom)
    assert extract_features(voice()) is not None


# ----- arousal -----

def trained(n=10):
    est = AudioEmotionEstimator()

    for i in range(n):
        est.estimate(extract_features(
            voice(f0=150 + (i % 3), amp=0.1 + 0.005 * (i % 3),
                  syl=3.0 + 0.1 * (i % 2), seed=i)))

    return est


def test_unknown_until_baseline_exists():
    est = AudioEmotionEstimator()

    for i in range(4):
        r = est.estimate(extract_features(voice(seed=i)))
        assert r.label == "unknown" and r.confidence == 0.0


def test_high_and_low_arousal_relative_to_own_baseline():
    est = trained()
    high = est.estimate(extract_features(
        voice(f0=210, amp=0.45, syl=5.0)), learn=False)
    low = est.estimate(extract_features(
        voice(f0=120, amp=0.03, syl=1.5)), learn=False)
    same = est.estimate(extract_features(voice(f0=151, amp=0.1)),
                        learn=False)
    assert high.label == "high" and high.arousal > 0.3
    assert low.label == "low" and low.arousal < -0.3
    assert same.label == "normal"


def test_same_voice_is_calm_for_a_low_pitched_speaker():
    est = AudioEmotionEstimator()

    for i in range(10):
        est.estimate(extract_features(voice(f0=100, seed=i)))

    r = est.estimate(extract_features(voice(f0=101)), learn=False)
    assert r.label == "normal"


def test_confidence_is_capped_and_never_certain():
    est = trained(40)
    r = est.estimate(extract_features(voice(f0=250, amp=0.5, syl=6)),
                     learn=False)
    assert 0 < r.confidence <= 0.6
    assert MAX_AUDIO_CONFIDENCE == 0.6


def test_noise_lowers_confidence():
    est = trained(25)
    clean = est.estimate(extract_features(voice(f0=210, amp=0.4, syl=5)),
                         learn=False)
    noisy = est.estimate(extract_features(
        voice(f0=210, amp=0.4, syl=5, noise=0.5)), learn=False)
    assert noisy.confidence < clean.confidence


def test_unusable_audio_gives_unknown():
    r = AudioEmotionEstimator().estimate(None)
    assert r.label == "unknown" and r.confidence == 0


def test_evaluate_reports_counts_and_no_invented_accuracy():
    clips = [(voice(seed=i), "normal") for i in range(3)]
    out = evaluate(AudioEmotionEstimator, clips)
    assert out["clips"] == 3 and out["committed"] == 0
    assert out["accuracy"] is None
    many = [(voice(seed=i), "normal") for i in range(12)]
    many.append((voice(f0=220, amp=0.5, syl=6), "high"))
    out = evaluate(AudioEmotionEstimator, many)
    assert out["committed"] + out["unknown"] == 13
    assert out["accuracy"] is None or 0 <= out["accuracy"] <= 1


# ----- text emotion -----

@pytest.mark.parametrize("text,label,explicit", [
    ("I'm feeling really tired", "tired", True),
    ("i am so happy today", "happy", True),
    ("I feel lonely", "lonely", True),
    ("I'm stressed", "stressed", True),
    ("what a long day", "tired", False),
    ("I passed my exam", "happy", False),
    ("hello there", "unknown", False),
    ("I'm down for pizza", "unknown", False),
    ("I feel like going for a walk", "unknown", False),
])
def test_text_emotion(text, label, explicit):
    te = text_emotion(text)
    assert te.label == label and te.explicit == explicit


def test_text_negation_is_a_correction_not_the_emotion():
    te = text_emotion("I'm not sad")
    assert te.label == "unknown" and te.correction_of == "sad"
    assert text_emotion("that's not how i feel").generic_correction


# ----- fusion -----

def fusion(**cfg):
    clock = Clock()

    return EmotionFusion(FusionConfig(**cfg), clock=clock), clock


def audio(label, conf=0.5):
    return AudioEmotion(0.6 if label == "high" else -0.6, label, conf, 4)


def test_explicit_words_beat_voice():
    f, _ = fusion()
    state = f.update(text_emotion("I'm happy"), audio("low"))
    assert state.label == "happy" and state.explicit
    assert not state.uncertain


def test_voice_alone_gives_only_a_weak_energy_hint():
    f, _ = fusion()
    state = f.update(None, audio("low", 0.6))
    assert state.label == "low_energy"
    assert state.confidence <= 0.45 and state.uncertain
    assert f.update(None, audio("high", 0.6)).label == "high_energy"


def test_voice_supports_matching_text_only():
    f, _ = fusion()
    base = f.update(text_emotion("what a long day")).confidence
    f2, _ = fusion()
    boosted = f2.update(text_emotion("what a long day"),
                        audio("low")).confidence
    f3, _ = fusion()
    unchanged = f3.update(text_emotion("what a long day"),
                          audio("high")).confidence
    assert boosted > base and unchanged == base


def test_correction_blocks_label_for_a_while_then_expires():
    f, clock = fusion(denial_s=100)
    f.update(text_emotion("I'm sad"))
    f.update(text_emotion("I'm not sad"))
    assert f.current().label == "unknown"
    assert f.update(text_emotion("feeling down")).label == "unknown"
    clock.t += 101
    assert f.update(text_emotion("feeling down")).label == "sad"


def test_generic_correction_clears_current_estimate():
    f, _ = fusion()
    f.update(text_emotion("what a long day"))
    f.update(text_emotion("that's not how i feel"))
    assert f.current().label == "unknown"


def test_state_decays_to_unknown_unless_explicit():
    f, clock = fusion(half_life_s=60)
    f.update(text_emotion("what a long day"))
    assert f.current().label == "tired"
    clock.t += 200
    assert f.current().label == "unknown"
    f.update(text_emotion("I'm exhausted"))
    clock.t += 400
    assert f.current().label == "tired"


def test_vision_never_overrides_words():
    f, _ = fusion()
    state = f.update(text_emotion("I'm sad"), None, "happy", 0.95)
    assert state.label == "sad"
    f2, _ = fusion()
    assert f2.update(None, None, "happy", 0.95).label == "unknown"
    assert f2.update(None, None, "happy", 0.3).label == "unknown"


# ----- responses -----

FORBIDDEN = ("depress", "disorder", "diagnos", "therap", "symptom",
             "mental", "illness", "clinical", "medicat", "anxiety",
             "psych", "condition")


def mgr(seed=1):
    return EmotionalResponseManager(
        EmotionFusion(clock=Clock()), rng=random.Random(seed))


def test_no_clinical_vocabulary_in_any_reply():
    texts = [t for v in REPLIES.values() for t in v]
    texts += list(CORRECTIONS) + [CRISIS_REPLY]
    for text in texts:
        assert not any(w in text.lower() for w in FORBIDDEN), text


def test_explicit_feeling_gets_supportive_reply():
    m = mgr()
    reply = m.hook("I'm feeling sad", ConversationContext())
    assert reply.text in REPLIES["sad"] and reply.intent == "feeling_sad"


def test_neutral_and_unrelated_text_pass_through():
    m = mgr()
    ctx = ConversationContext()
    assert m.hook("what time is it", ctx) is None
    assert m.hook("I'm fine", ctx) is None


def test_voice_hint_alone_never_produces_an_emotion_statement():
    m = mgr()
    m.observe_audio(audio("low", 0.6))
    assert m.hook("what time is it", ConversationContext()) is None
    assert m.hint() == "low_energy"


def test_wording_varies_on_repeat():
    m = mgr()
    lines = [m.hook("I'm tired", ConversationContext()).text
             for _ in range(8)]
    assert all(a != b for a, b in zip(lines, lines[1:]))


def test_correction_is_accepted_immediately():
    m = mgr()
    m.hook("I'm sad", ConversationContext())
    reply = m.hook("I'm not sad", ConversationContext())
    assert reply.text in CORRECTIONS
    assert m.hint() is None


@pytest.mark.parametrize("text", [
    "I want to die", "i'm going to kill myself", "I want to hurt myself",
])
def test_crisis_gets_fixed_safety_reply_before_anything_else(text):
    m = mgr()
    reply = m.hook(text, ConversationContext())
    assert reply.text == CRISIS_REPLY and reply.source == "safety"
    assert not any(ch.isdigit() for ch in reply.text)


def test_crisis_beats_other_hooks_and_llm():
    class Llm:
        def complete(self, *a):
            raise AssertionError("LLM must not answer a crisis")

    m = mgr()
    d = DialogueManager(hooks=[m.hook], llm=Llm(), rng=random.Random(1))
    reply = d.respond("I want to die", ConversationContext())
    assert reply.source == "safety"


def test_style_is_gentler_for_low_mood_and_default_when_unknown():
    m = mgr()
    default = m.style()
    assert default == SpeechStyle()
    m.hook("I'm exhausted", ConversationContext())
    gentle = m.style()
    assert gentle.length_scale > default.length_scale
    assert gentle.volume < default.volume


def test_observer_hands_only_a_hint_to_the_manager():
    m = mgr()
    est = AudioEmotionEstimator()
    obs = make_utterance_observer(m, est)
    obs(voice())
    assert est.baseline.count == 1
    assert m._audio is not None
    obs(np.zeros(100, np.int16))
    assert m._audio.label == "unknown"


# ----- engine integration -----

def test_engine_passes_audio_to_observer_then_discards_it():
    from test_voice_engine import Rig, T

    rig = Rig(awake=True)
    got = []
    rig.engine.utterance_observer = lambda a: got.append(a.size)
    rig.gen.finals = [T("what time is it")]
    rig.say_phrase(seconds=1.0)
    assert len(got) == 1 and got[0] >= RATE * 0.9
    assert rig.engine._utt_audio == []


def test_engine_uses_emotion_style_for_replies():
    from test_voice_engine import Rig, T

    rig = Rig(awake=True)
    m = mgr()
    rig.dialogue.hooks.append(m.hook)
    rig.engine.style_provider = m.style
    rig.gen.finals = [T("I'm feeling exhausted")]
    rig.say_phrase()
    text, style = rig.synth.calls[-1]
    assert text in REPLIES["tired"]
    assert style.length_scale > 1.0

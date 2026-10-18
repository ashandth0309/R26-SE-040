"""Vision/voice fusion, including contradictory evidence."""

import pytest

from buddy_core.vision.expression_tracker import ExprState, ExpressionEstimate
from buddy_core.vision.multimodal import (
    Agreement,
    FusionConfig,
    VoiceAffect,
    fuse,
    observation_fields,
)

NOW = 1000.0


def vis(group="negative", conf=0.7, state=ExprState.CONFIRMED):
    return ExpressionEstimate(state, group, group, conf, 10, "")


def voice(label="sad", conf=0.6, explicit=False, age=5.0, **kw):
    return VoiceAffect(label, conf, explicit, NOW - age, **kw)


def test_person_says_sad_while_smiling_their_words_win():
    r = fuse(vis("positive", 0.9), NOW, voice("sad", 0.9, True), NOW)
    assert r.agreement is Agreement.USER_STATED
    assert r.group == "negative" and "face_disagrees_with_words" in r.notes
    assert r.needs_confirmation is False


def test_person_says_happy_while_face_looks_negative_words_still_win():
    r = fuse(vis("negative", 0.9), NOW, voice("happy", 0.9, True), NOW)
    assert r.group == "positive" and r.agreement is Agreement.USER_STATED


def test_face_and_voice_agree_gives_consistent_but_capped_confidence():
    r = fuse(vis("negative", 0.95), NOW, voice("tired", 0.95), NOW)
    assert r.agreement is Agreement.CONSISTENT and r.group == "negative"
    assert r.confidence <= FusionConfig().max_consistent
    assert r.needs_confirmation and set(r.sources) == {"face", "voice"}


def test_face_and_voice_disagree_gives_conflict_with_no_label():
    r = fuse(vis("positive", 0.9), NOW, voice("sad", 0.8), NOW)
    assert r.agreement is Agreement.CONFLICT
    assert r.group is None and r.confidence <= 0.25 and r.needs_confirmation
    assert r.visual_group == "positive" and r.voice_label == "sad"


def test_face_alone_is_weak_and_needs_confirmation():
    r = fuse(vis("negative", 0.99), NOW, None, NOW)
    assert r.agreement is Agreement.VISUAL_ONLY
    assert r.confidence <= FusionConfig().max_visual_only
    assert r.needs_confirmation and r.sources == ("face",)


def test_recent_denial_suppresses_that_interpretation():
    v = voice("neutral", 0.5, denied_groups=("negative",),
              denied_until=NOW + 300)
    r = fuse(vis("negative", 0.9), NOW, v, NOW)
    assert r.agreement is Agreement.SUPPRESSED and r.group is None
    expired = voice("unknown", 0.0, denied_groups=("negative",),
                    denied_until=NOW - 1)
    again = fuse(vis("negative", 0.9), NOW, expired, NOW)
    assert again.agreement is Agreement.VISUAL_ONLY
    assert again.group == "negative"


def test_stale_or_unconfirmed_vision_is_ignored():
    assert fuse(vis(), NOW - 10, None, NOW).agreement is Agreement.NONE
    tentative = vis(state=ExprState.TENTATIVE)
    assert fuse(tentative, NOW, None, NOW).agreement is Agreement.NONE
    weak = vis(conf=0.3)
    assert fuse(weak, NOW, None, NOW).agreement is Agreement.NONE
    unknown = ExpressionEstimate(ExprState.UNKNOWN, None, None, None, 0, "")
    assert fuse(unknown, NOW, None, NOW).agreement is Agreement.NONE


def test_old_voice_state_decays_out():
    old = voice("sad", 0.9, True, age=500)
    r = fuse(vis("positive", 0.9), NOW, old, NOW)
    assert r.agreement is Agreement.VISUAL_ONLY and r.group == "positive"


def test_high_energy_voice_cannot_confirm_a_face():
    r = fuse(vis("negative", 0.9), NOW, voice("high_energy", 0.5), NOW)
    assert r.agreement is Agreement.VISUAL_ONLY


def test_voice_only_does_not_make_a_visual_claim():
    r = fuse(None, 0.0, voice("sad", 0.6), NOW)
    assert r.agreement is Agreement.NONE and r.group is None


def test_nothing_at_all_is_nothing():
    r = fuse(None, 0.0, None, NOW)
    assert r.agreement is Agreement.NONE and r.sources == ()


def test_no_combination_exceeds_its_cap():
    cfg = FusionConfig()
    for g in ("positive", "neutral", "negative", "surprise"):
        for lab in ("happy", "sad", "neutral", "tired", "excited"):
            r = fuse(vis(g, 1.0), NOW, voice(lab, 1.0), NOW, cfg)
            if r.agreement is Agreement.CONSISTENT:
                assert r.confidence <= cfg.max_consistent
            if r.agreement is Agreement.VISUAL_ONLY:
                assert r.confidence <= cfg.max_visual_only
            if r.agreement is Agreement.CONFLICT:
                assert r.confidence <= cfg.conflict_confidence


# ---------- message fields and the real voice engine ----------

def test_observation_fields_clear_when_nothing_is_confirmed():
    tentative = vis(state=ExprState.TENTATIVE)
    f = observation_fields(tentative, "session_person_1")
    assert f["expression"] == "unknown" and f["confidence"] == 0.0
    ok = observation_fields(vis("positive", 0.8), "p")
    assert ok["expression"] == "happy" and ok["confidence"] == 0.8
    neg = observation_fields(vis("negative", 0.8), "p")
    assert neg["expression"] == "negative"


def test_message_fields_match_the_real_message_definition():
    from pathlib import Path

    msg = (Path(__file__).resolve().parents[2] / "buddy_interfaces" /
           "msg" / "EmotionObservation.msg").read_text()
    declared = {line.split()[1] for line in msg.splitlines()
                if line.strip() and not line.startswith("#")}
    assert set(observation_fields(vis(), "p")) <= declared


def test_voice_engine_does_not_act_on_a_face_hint_alone():
    from buddy_core.voice.emotion_fusion import EmotionFusion

    for word in ("happy", "negative", "surprised", "unknown"):
        fusion = EmotionFusion()
        state = fusion.update(vision_expression=word,
                              vision_confidence=0.95)
        assert state.label == "unknown", word


def test_voice_engine_uses_face_only_to_support_a_stated_feeling():
    from buddy_core.voice.emotion_fusion import EmotionFusion, text_emotion

    fusion = EmotionFusion()
    plain = fusion.update(text=text_emotion("I'm feeling happy"))
    fusion2 = EmotionFusion()
    boosted = fusion2.update(text=text_emotion("I'm feeling happy"),
                             vision_expression="happy",
                             vision_confidence=0.9)
    assert boosted.confidence >= plain.confidence
    assert pytest.approx(boosted.confidence, abs=0.3) == plain.confidence

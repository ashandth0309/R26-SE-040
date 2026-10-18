"""Fuse what the camera sees with what the person says and sounds like.

The voice companion already keeps a fused emotional hint (words, acoustics
and a face hint).  This module is the vision side of that handshake: it
decides how much a *visible expression* should count given the voice
side's latest state, and it never lets the face outvote the person.

Rules, strongest first:

1. The person's own words win.  If they said how they feel (``explicit``),
   the face is recorded as agreeing or disagreeing and is otherwise ignored.
2. A correction ("I'm fine") suppresses that interpretation for a while.
3. Voice-only hints (tone, implied wording) and a face that point the same
   way make a *consistent* but still tentative observation.
4. When they point different ways the result is a *conflict*: no label, low
   confidence, and a note saying so.
5. A face alone is weak: capped confidence, ``visual_only``.

The confidence caps below are design limits, not measured probabilities.
They exist so a camera hint can never look more certain than it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from buddy_core.vision.expression_tracker import ExprState, ExpressionEstimate

# Voice-side labels (buddy_core.voice.emotion_fusion.LABELS) -> group
VOICE_GROUP = {
    "happy": "positive", "excited": "positive",
    "neutral": "neutral",
    "sad": "negative", "stressed": "negative", "tired": "negative",
    "angry": "negative", "anxious": "negative", "lonely": "negative",
    "low_energy": "negative",
}
# "high_energy" can be excitement or anger: it supports nothing by itself.
AMBIGUOUS_VOICE = {"high_energy"}


class Agreement(str, Enum):
    NONE = "none"                 # nothing usable
    VISUAL_ONLY = "visual_only"
    CONSISTENT = "consistent"
    CONFLICT = "conflict"
    USER_STATED = "user_stated"
    SUPPRESSED = "suppressed"     # the person denied this recently


@dataclass(frozen=True)
class VoiceAffect:
    """Latest state from the voice engine (mirror of its EmotionState)."""

    label: str = "unknown"
    confidence: float = 0.0
    explicit: bool = False        # the person said it themselves
    stamp: float = 0.0
    denied_groups: tuple[str, ...] = ()   # groups the person rejected
    denied_until: float = 0.0


@dataclass
class FusionConfig:
    max_visual_only: float = 0.45
    max_consistent: float = 0.70
    conflict_confidence: float = 0.20
    vision_max_age_s: float = 3.0
    voice_max_age_s: float = 120.0
    consistent_bonus: float = 0.10
    min_visual_confidence: float = 0.55


@dataclass(frozen=True)
class FusedObservation:
    agreement: Agreement
    group: str | None             # None unless there is something to say
    confidence: float
    visual_group: str | None
    voice_label: str | None
    sources: tuple[str, ...]
    explanation: str
    needs_confirmation: bool      # must ask before acting on it
    notes: tuple[str, ...] = field(default_factory=tuple)


def fuse(vision: ExpressionEstimate | None, vision_stamp: float,
         voice: VoiceAffect | None, now: float,
         cfg: FusionConfig | None = None) -> FusedObservation:
    cfg = cfg or FusionConfig()
    notes: list[str] = []

    visual_group = None

    if vision is not None and vision.state is ExprState.CONFIRMED \
            and vision.group and (vision.confidence or 0) >= \
            cfg.min_visual_confidence \
            and now - vision_stamp <= cfg.vision_max_age_s:
        visual_group = vision.group

    voice_ok = voice is not None and voice.label != "unknown" and \
        now - voice.stamp <= cfg.voice_max_age_s

    if voice is not None and visual_group and \
            visual_group in voice.denied_groups and \
            now < voice.denied_until:
        return FusedObservation(
            Agreement.SUPPRESSED, None, 0.0, visual_group,
            voice.label if voice_ok else None, ("correction",),
            "the person recently said this isn't how they feel", False)

    if voice_ok and voice.explicit:
        group = VOICE_GROUP.get(voice.label)

        if visual_group and group and visual_group != group:
            notes.append("face_disagrees_with_words")
        elif visual_group and group:
            notes.append("face_agrees_with_words")

        return FusedObservation(
            Agreement.USER_STATED, group, voice.confidence, visual_group,
            voice.label, ("words",),
            "the person said how they feel; that is what counts", False,
            tuple(notes))

    voice_group = None

    if voice_ok and voice.label not in AMBIGUOUS_VOICE:
        voice_group = VOICE_GROUP.get(voice.label)

    if visual_group and voice_group:
        if visual_group == voice_group:
            conf = min(cfg.max_consistent,
                       max(vision.confidence or 0.0, voice.confidence)
                       * 0.8 + cfg.consistent_bonus)

            return FusedObservation(
                Agreement.CONSISTENT, visual_group, conf, visual_group,
                voice.label, ("face", "voice"),
                "face and voice point the same way; still only a hint",
                True)

        return FusedObservation(
            Agreement.CONFLICT, None, cfg.conflict_confidence,
            visual_group, voice.label, ("face", "voice"),
            "face and voice disagree, so I won't guess", True,
            ("conflict",))

    if visual_group:
        conf = min(cfg.max_visual_only, (vision.confidence or 0.0) * 0.6)

        return FusedObservation(
            Agreement.VISUAL_ONLY, visual_group, conf, visual_group, None,
            ("face",), "a visible expression only; weak evidence", True)

    if voice_group:
        return FusedObservation(
            Agreement.NONE, None, 0.0, None, voice.label, ("voice",),
            "voice hint only; vision has nothing reliable", True)

    return FusedObservation(Agreement.NONE, None, 0.0, None, None, (),
                            "nothing reliable to report", False)


# ----- EmotionObservation (voice's existing input) -----

# The voice engine only understands these expression words and ignores the
# rest, so "negative" is published as-is and does no harm there.
TO_VOICE_WORD = {"positive": "happy", "neutral": "neutral",
                 "surprise": "surprised", "negative": "negative"}


def observation_fields(estimate: ExpressionEstimate, person_id: str,
                       source: str = "vision_expression") -> dict:
    """Fields for ``buddy_interfaces/EmotionObservation``.

    Unconfirmed or unknown estimates are published as ``unknown`` with zero
    confidence, which the voice side ignores.  That also clears an old hint:
    the voice engine keeps the last value it received.
    """

    if estimate.state is not ExprState.CONFIRMED or not estimate.group:
        return {"person_id": person_id, "expression": "unknown",
                "confidence": 0.0, "source": source}

    return {"person_id": person_id,
            "expression": TO_VOICE_WORD.get(estimate.group, "unknown"),
            "confidence": float(estimate.confidence or 0.0),
            "source": source}

"""Combine what the user said, how they sounded and what vision saw.

Priority, strongest first:

1. A correction from the user ("I'm not sad") always wins and blocks that
   label for a while.
2. An explicit statement ("I'm feeling tired") beats any prediction.
3. Implicit wording (e.g. "ugh, what a day") is moderate evidence.
4. Voice arousal and a facial-expression hint can *support* a text-based
   label and, on their own, can only produce a vague "low_energy" or
   "high_energy" hint with low confidence.

Below ``min_confidence`` the state is ``unknown``.  The result is a hint
for gentler wording and pacing, never a fact about the person.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from buddy_core.voice.emotion_audio import AudioEmotion

LABELS = (
    "neutral", "happy", "sad", "stressed", "tired", "angry", "anxious",
    "lonely", "excited", "low_energy", "high_energy", "unknown",
)

LEXICON: dict[str, str] = {
    "happy": "happy", "glad": "happy", "great": "happy", "good": "happy",
    "wonderful": "happy", "fine": "neutral", "okay": "neutral",
    "excited": "excited", "thrilled": "excited",
    "sad": "sad", "unhappy": "sad", "upset": "sad",
    "depressed": "sad", "miserable": "sad",
    "tired": "tired", "exhausted": "tired", "sleepy": "tired",
    "drained": "tired",
    "stressed": "stressed", "overwhelmed": "stressed",
    "pressured": "stressed",
    "angry": "angry", "furious": "angry", "annoyed": "angry",
    "mad": "angry", "frustrated": "angry",
    "anxious": "anxious", "worried": "anxious", "nervous": "anxious",
    "scared": "anxious", "afraid": "anxious",
    "lonely": "lonely",
}

_ADVERB = r"(?:very |so |really |a bit |quite |kind of |pretty |super )?"
_LEAD = (
    r"(?:i am|i'm|im|i feel|i've been|i have been|feeling)\s+"
    r"(?:feeling\s+)?"
)
_EXPLICIT = re.compile(r"\b" + _LEAD + _ADVERB + r"([a-z]+)\b")
_NEGATED = re.compile(
    r"\b(?:i am not|i'm not|im not|i don't feel|i do not feel|"
    r"i wasn't|i was not)\s+(?:feeling\s+)?" + _ADVERB + r"([a-z]+)\b"
)
_GENERIC_CORRECTION = re.compile(
    r"\b(?:that's not how i feel|that is not how i feel|you're wrong|"
    r"you are wrong|no i'm fine|no i am fine|i didn't say that)\b"
)
_IMPLICIT = {
    "tired": ("what a long day", "long day", "so sleepy", "can't sleep"),
    "stressed": ("so much to do", "too much work", "deadline"),
    "sad": ("i miss", "nothing is going right", "feel like crying",
            "feeling down", "feel down"),
    "happy": ("best day", "so proud", "i passed", "i won"),
}


@dataclass(frozen=True)
class TextEmotion:
    """What the words say."""

    label: str
    confidence: float
    explicit: bool = False
    correction_of: str | None = None
    generic_correction: bool = False


def text_emotion(text: str) -> TextEmotion:
    """Read emotion from wording only."""

    lower = text.lower().strip()

    if _GENERIC_CORRECTION.search(lower):
        return TextEmotion("unknown", 0.0, generic_correction=True)

    neg = _NEGATED.search(lower)

    if neg and neg.group(1) in LEXICON:
        return TextEmotion("unknown", 0.0, correction_of=LEXICON[
            neg.group(1)])

    match = _EXPLICIT.search(lower)

    if match and match.group(1) in LEXICON:
        return TextEmotion(LEXICON[match.group(1)], 0.9, explicit=True)

    for label, phrases in _IMPLICIT.items():
        if any(p in lower for p in phrases):
            return TextEmotion(label, 0.55)

    return TextEmotion("unknown", 0.0)


@dataclass(frozen=True)
class EmotionState:
    """Fused hint with the evidence behind it."""

    label: str = "unknown"
    confidence: float = 0.0
    sources: tuple[str, ...] = ()
    explicit: bool = False
    stamp: float = 0.0

    @property
    def uncertain(self) -> bool:
        """True unless the user said it themselves."""

        return not self.explicit


@dataclass(frozen=True)
class FusionConfig:
    """Fusion settings."""

    min_confidence: float = 0.4
    half_life_s: float = 120.0
    denial_s: float = 600.0
    vision_weight: float = 0.15
    audio_weight: float = 0.25


@dataclass
class EmotionFusion:
    """Maintains the current emotional hint."""

    config: FusionConfig = field(default_factory=FusionConfig)
    clock: Callable[[], float] = time.monotonic
    state: EmotionState = field(default_factory=EmotionState)
    denied: dict[str, float] = field(default_factory=dict)

    def _is_denied(self, label: str, now: float) -> bool:
        until = self.denied.get(label)

        return until is not None and now < until

    def current(self) -> EmotionState:
        """State with time decay applied."""

        now = self.clock()
        s = self.state

        if s.label == "unknown":
            return s

        age = max(0.0, now - s.stamp)
        decayed = s.confidence * 0.5 ** (age / self.config.half_life_s)

        if decayed < self.config.min_confidence and not s.explicit:
            return EmotionState("unknown", decayed, s.sources, False,
                                s.stamp)

        return EmotionState(s.label, decayed, s.sources, s.explicit,
                            s.stamp)

    def correct(self, label: str | None = None) -> None:
        """The user says the estimate is wrong."""

        now = self.clock()
        target = label or self.state.label

        if target and target != "unknown":
            self.denied[target] = now + self.config.denial_s

        self.state = EmotionState("unknown", 0.0, ("correction",), False,
                                  now)

    def update(
        self,
        text: TextEmotion | None = None,
        audio: AudioEmotion | None = None,
        vision_expression: str | None = None,
        vision_confidence: float | None = None,
    ) -> EmotionState:
        """Fuse new evidence and return the new state."""

        now = self.clock()
        cfg = self.config

        if text is not None and (text.generic_correction
                                 or text.correction_of):
            self.correct(text.correction_of)

            return self.state

        scores: dict[str, float] = {}
        sources: list[str] = []
        explicit = False

        if text and text.label != "unknown":
            scores[text.label] = text.confidence
            sources.append("words")
            explicit = text.explicit

        if audio and audio.label in ("low", "high") \
                and audio.confidence > 0:
            hint = "low_energy" if audio.label == "low" else "high_energy"
            support = {"low": ("sad", "tired", "lonely", "low_energy"),
                       "high": ("stressed", "angry", "anxious", "excited",
                                "happy", "high_energy")}[audio.label]
            gain = cfg.audio_weight * audio.confidence / 0.6

            if scores:
                top = max(scores, key=scores.get)

                if top in support:
                    scores[top] = min(0.95, scores[top] + gain)
                    sources.append("voice")
            else:
                scores[hint] = min(0.45, gain + 0.2)
                sources.append("voice")

        if vision_expression and (vision_confidence or 0) >= 0.6:
            mapped = {"happy": "happy", "sad": "sad", "angry": "angry",
                      "neutral": "neutral"}.get(vision_expression)

            if mapped and scores and mapped in scores:
                scores[mapped] = min(0.95, scores[mapped]
                                     + cfg.vision_weight)
                sources.append("face")
            elif mapped and not scores:
                scores[mapped] = 0.3
                sources.append("face")

        for label in list(scores):
            if self._is_denied(label, now):
                del scores[label]

        if not scores:
            return self.current()

        label = max(scores, key=scores.get)
        confidence = scores[label]

        if confidence < cfg.min_confidence and not explicit:
            return self.current()

        self.state = EmotionState(label, confidence, tuple(sources),
                                  explicit, now)

        return self.state

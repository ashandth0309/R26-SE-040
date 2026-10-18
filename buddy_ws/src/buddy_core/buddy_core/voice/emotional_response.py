"""Supportive, varied replies to what the user says about how they feel.

Rules:

* Never name a clinical condition or suggest a diagnosis.  A test checks
  every line for clinical vocabulary.
* Speak to what the person *said*.  Voice-only hints change pacing and
  volume but never produce a statement like "you sound sad".
* Do not comment on feelings every turn; repeated wording is avoided.
* A correction from the user is accepted at once.
* Statements about wanting to die or harming oneself get a fixed, caring
  safety reply that is never produced by a language model and is not
  stored.  It points to people nearby and local emergency services without
  inventing a phone number.
"""

from __future__ import annotations

import random
import re

from buddy_core.voice.dialogue_manager import (
    ConversationContext,
    DialogueReply,
)
from buddy_core.voice.emotion_audio import AudioEmotion
from buddy_core.voice.emotion_fusion import (
    EmotionFusion,
    EmotionState,
    text_emotion,
)
from buddy_core.voice.tts_backends import SpeechStyle
from buddy_core.voice.variation import VariantPicker

CRISIS = re.compile(
    r"\b(?:kill myself|end my life|want to die|wanna die|"
    r"don't want to live|do not want to live|hurt myself|harm myself|"
    r"suicid\w*|self[- ]harm|better off dead)\b"
)

CRISIS_REPLY = (
    "I'm really sorry you're hurting this much. You matter, and you "
    "don't have to carry it alone. Please tell someone near you right "
    "now, or contact your local emergency number if you might be in "
    "danger. I'm here with you."
)

REPLIES: dict[str, tuple[str, ...]] = {
    "sad": (
        "I'm sorry you're feeling low. Do you want to tell me what "
        "happened?",
        "That sounds heavy. I'm here if you want to talk about it.",
        "Thanks for telling me. Want some company for a while?",
    ),
    "tired": (
        "Sounds like you could use some rest. Has it been a long day?",
        "That sounds draining. Do you want things quiet for a bit?",
        "Take it easy. I can keep it short tonight.",
    ),
    "stressed": (
        "That sounds like a lot. Want to talk it through?",
        "I'm sorry it's so much right now. What's weighing on you most?",
        "One thing at a time. I'm here if you want to say it out loud.",
    ),
    "angry": (
        "That sounds frustrating. What happened?",
        "I can hear why that would be annoying. Want to vent a little?",
        "Okay, I'm listening. Tell me what went wrong.",
    ),
    "anxious": (
        "That sounds unsettling. I'm right here. What's on your mind?",
        "It's okay. Want to tell me what's worrying you?",
        "Let's slow down together. What's the biggest worry right now?",
    ),
    "lonely": (
        "I'm glad you told me. I'm here with you.",
        "You're not on your own right now. Want to chat for a while?",
        "Thanks for sharing that. I'm happy to keep you company.",
    ),
    "happy": (
        "That's lovely to hear! What made it good?",
        "Nice! Tell me more.",
        "I'm glad. What's been going well?",
    ),
    "excited": (
        "Ooh, tell me everything!",
        "That sounds exciting! What's happening?",
        "I love that energy. What's the news?",
    ),
}

CORRECTIONS = (
    "Okay, thanks for telling me. I'll take your word for it.",
    "Got it, sorry for assuming.",
    "Understood. You know best how you feel.",
)

STYLES = {
    "sad": (1.12, 0.9), "tired": (1.12, 0.85), "lonely": (1.08, 0.9),
    "anxious": (1.08, 0.9), "stressed": (1.05, 0.9),
    "angry": (1.05, 0.85), "excited": (0.95, 1.0),
    "happy": (0.97, 1.0), "high_energy": (0.97, 1.0),
    "low_energy": (1.08, 0.9),
}


class EmotionalResponseManager:
    """Hook for the dialogue manager plus delivery style."""

    def __init__(
        self,
        fusion: EmotionFusion | None = None,
        *,
        rng: random.Random | None = None,
        base_style: SpeechStyle | None = None,
    ) -> None:
        self.fusion = fusion or EmotionFusion()
        self._picker = VariantPicker(rng, memory=2)
        self.base_style = base_style or SpeechStyle()
        self._audio: AudioEmotion | None = None
        self.vision_expression: str | None = None
        self.vision_confidence: float | None = None

    def observe_audio(self, audio: AudioEmotion) -> None:
        """Voice arousal for the utterance now being handled."""

        self._audio = audio

    def observe_vision(self, expression: str | None,
                       confidence: float | None) -> None:
        """Latest facial-expression hint from the vision module."""

        self.vision_expression = expression
        self.vision_confidence = confidence

    def hint(self) -> str | None:
        """Current label for other components, or None if unknown."""

        label = self.fusion.current().label

        return None if label == "unknown" else label

    def style(self) -> SpeechStyle:
        """Delivery style for the current hint."""

        state: EmotionState = self.fusion.current()
        scale, vol = STYLES.get(state.label, (1.0, 1.0))
        base = self.base_style

        return SpeechStyle(
            length_scale=round(base.length_scale * scale, 3),
            pause_scale=base.pause_scale,
            volume=round(base.volume * vol, 3),
        )

    def hook(self, text: str, context: ConversationContext
             ) -> DialogueReply | None:
        """Dialogue hook: answer feelings, corrections and crises."""

        lower = text.lower()

        if CRISIS.search(lower):
            return DialogueReply(CRISIS_REPLY, "safety", "crisis")

        reading = text_emotion(text)
        audio, self._audio = self._audio, None

        if reading.correction_of or reading.generic_correction:
            self.fusion.update(reading)

            return DialogueReply(
                self._picker.pick("correction", CORRECTIONS),
                "emotion", "correction")

        self.fusion.update(reading, audio, self.vision_expression,
                           self.vision_confidence)

        if reading.label in REPLIES and (reading.explicit
                                         or reading.confidence >= 0.5):
            return DialogueReply(
                self._picker.pick(reading.label, REPLIES[reading.label]),
                "emotion", f"feeling_{reading.label}")

        return None


def make_utterance_observer(manager: EmotionalResponseManager,
                            estimator):
    """Build the engine's ``utterance_observer``.

    Extracts features in memory and hands only the resulting arousal hint
    to the manager; the audio itself is not kept.
    """

    from buddy_core.voice.emotion_audio import extract_features

    def observe(samples):
        manager.observe_audio(estimator.estimate(extract_features(samples)))

    return observe

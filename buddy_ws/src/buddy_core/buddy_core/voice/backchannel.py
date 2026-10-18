"""Short listening sounds ("mm-hm") while the user talks.

Backchannels make a long statement feel heard, but done badly they are
annoying or unsafe.  The rules here are conservative:

* Only during a pause inside a long user utterance, never while Buddy is
  speaking, and never right after another backchannel.
* Never while the user is giving a command (command words in the partial
  transcript), when silence mode or privacy mute is on, or when disabled.
* Capped per minute, with a minimum gap, and wording is not repeated.
* They are LOW priority, so they are dropped if anything else is queued,
  and cancelled immediately if the user resumes.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from enum import Enum

from buddy_core.voice.variation import VariantPicker
from buddy_core.voice.voice_commands import (
    PHRASES,
    STOP_WORDS,
    tokenize,
)


class BackchannelKind(str, Enum):
    """Why we are making the sound."""

    ACKNOWLEDGE = "acknowledge"
    EMPATHY = "empathy"
    ENCOURAGE = "encourage"
    SURPRISE = "surprise"


LINES: dict[BackchannelKind, tuple[str, ...]] = {
    BackchannelKind.ACKNOWLEDGE: ("Mm-hm.", "I see.", "Right.", "Okay."),
    BackchannelKind.EMPATHY: (
        "Oh.", "I hear you.", "That sounds hard.",
    ),
    BackchannelKind.ENCOURAGE: ("Go on.", "Mm, tell me more.", "And?"),
    BackchannelKind.SURPRISE: ("Oh!", "Really?", "Wow."),
}

NEGATIVE_CUES = frozenset({
    "sad", "tired", "stressed", "lonely", "worried", "anxious", "upset",
    "bad", "awful", "terrible", "hurt", "angry", "scared", "exhausted",
})
POSITIVE_CUES = frozenset({
    "won", "passed", "excited", "amazing", "great", "finally", "promoted",
})


@dataclass(frozen=True)
class BackchannelConfig:
    """Timing and limits."""

    enabled: bool = True
    min_utterance_ms: int = 4000
    min_pause_ms: int = 600
    max_pause_ms: int = 1500
    min_gap_s: float = 8.0
    max_per_minute: int = 4


class BackchannelManager:
    """Decide whether and what to backchannel."""

    def __init__(
        self,
        config: BackchannelConfig | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or BackchannelConfig()
        self._picker = VariantPicker(rng)
        self._last_time = -1e9
        self._times: deque[float] = deque()
        self._fired_this_pause = False
        self.suppressed_reason = ""

    def set_enabled(self, enabled: bool) -> None:
        """Turn backchannels on or off at runtime."""

        self.config = BackchannelConfig(
            **{**self.config.__dict__, "enabled": enabled}
        )

    def on_user_resumed(self) -> None:
        """The user started talking again after a pause."""

        self._fired_this_pause = False

    @staticmethod
    def looks_like_command(partial_text: str) -> bool:
        """True if the partial transcript is, or may become, a command."""

        tokens = tokenize(partial_text)

        if any(t in STOP_WORDS for t in tokens):
            return True

        text = " ".join(tokens)

        if not text or len(tokens) > 4:
            return False

        # A prefix of a command phrase ("go for") may still become one.
        return any(
            phrase in text or (len(text) > 2 and phrase.startswith(text))
            for phrase in PHRASES
        )

    def choose_kind(self, partial_text: str, emotion: str | None
                    ) -> BackchannelKind:
        """Pick a category from the words and the emotion hint."""

        tokens = set(tokenize(partial_text))

        if emotion in ("sad", "stressed", "tired", "angry") \
                or tokens & NEGATIVE_CUES:
            return BackchannelKind.EMPATHY

        if tokens & POSITIVE_CUES:
            return BackchannelKind.SURPRISE

        if len(tokens) > 25:
            return BackchannelKind.ENCOURAGE

        return BackchannelKind.ACKNOWLEDGE

    def consider(
        self,
        now: float,
        *,
        user_speaking_ms: int,
        pause_ms: int,
        partial_text: str = "",
        emotion: str | None = None,
        buddy_speaking: bool = False,
        silenced: bool = False,
        command_active: bool = False,
    ) -> str | None:
        """Return a line to say now, or None."""

        reason = self._blocked(
            now, user_speaking_ms, pause_ms, partial_text,
            buddy_speaking, silenced, command_active,
        )
        self.suppressed_reason = reason

        if reason:
            return None

        kind = self.choose_kind(partial_text, emotion)
        line = self._picker.pick(kind.value, LINES[kind])
        self._last_time = now
        self._times.append(now)
        self._fired_this_pause = True

        return line

    def _blocked(self, now, speaking_ms, pause_ms, text, buddy, silenced,
                 command) -> str:
        cfg = self.config

        if not cfg.enabled:
            return "disabled"

        if silenced:
            return "silenced"

        if buddy:
            return "buddy speaking"

        if command or self.looks_like_command(text):
            return "command"

        if speaking_ms < cfg.min_utterance_ms:
            return "utterance too short"

        if not cfg.min_pause_ms <= pause_ms <= cfg.max_pause_ms:
            return "pause timing"

        if self._fired_this_pause:
            return "already fired this pause"

        if now - self._last_time < cfg.min_gap_s:
            return "too soon"

        while self._times and now - self._times[0] > 60.0:
            self._times.popleft()

        if len(self._times) >= cfg.max_per_minute:
            return "rate limit"

        return ""

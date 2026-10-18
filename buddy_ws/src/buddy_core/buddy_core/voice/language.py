"""Language detection and the English/Tamil support policy.

What is and is not supported, stated plainly:

* Offline recognition uses the English Vosk model.  It cannot recognise
  Tamil; Tamil speech through it comes out as wrong English words, and
  nothing here can detect that.  A separate Tamil recogniser would have to
  be supplied and verified.
* Offline speech uses an English Piper voice.  It cannot read Tamil script.
  A Tamil voice would have to be supplied and verified.
* Tamil *text* (typed, or from an online recogniser) can be detected by
  script, and romanised Tamil by a small word list that is only a hint.

``LanguagePolicy`` decides what to do rather than pretending.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from buddy_core.voice.dialogue_manager import (
    ConversationContext,
    DialogueReply,
)

TAMIL_RANGE = re.compile(r"[஀-௿]")
LATIN = re.compile(r"[A-Za-z]")

ROMAN_TAMIL = frozenset({
    "vanakkam", "nandri", "epdi", "eppadi", "irukinga", "irukkeenga",
    "irukka", "enna", "sapteengala", "saptiya", "romba", "illa",
    "seri", "amma", "appa", "thambi", "thangachi", "podhum", "va",
    "poda", "poi", "vaa", "inge", "ange", "nalla", "naan", "neenga",
})


def detect_language(text: str) -> str:
    """Return "en", "ta", "ta_latn" (romanised hint), "mixed" or
    "unknown"."""

    tamil = len(TAMIL_RANGE.findall(text))
    latin = len(LATIN.findall(text))

    if tamil and latin:
        return "mixed"

    if tamil:
        return "ta"

    words = re.findall(r"[a-z']+", text.lower())

    if not words:
        return "unknown"

    hits = sum(1 for w in words if w in ROMAN_TAMIL)

    if hits >= 2 or (hits == 1 and len(words) <= 2):
        return "ta_latn"

    return "en"


@dataclass(frozen=True)
class LanguagePolicy:
    """Which languages can be understood and spoken right now."""

    understood: frozenset = frozenset({"en"})
    spoken: frozenset = frozenset({"en"})

    def with_online(self, online: bool,
                    tamil_voice: bool = False) -> "LanguagePolicy":
        """Policy for the current mode.

        Online mode may understand Tamil text via a cloud model; speaking
        Tamil still needs a Tamil voice.
        """

        understood = {"en"} | ({"ta"} if online else set())
        spoken = {"en"} | ({"ta"} if tamil_voice else set())

        return LanguagePolicy(frozenset(understood), frozenset(spoken))


UNSUPPORTED_UNDERSTOOD = (
    "I think you spoke Tamil, but I can only understand English "
    "offline right now."
)
UNSUPPORTED_SPOKEN = (
    "I understood that, but I can only speak English with my current "
    "voice, so I'll answer in English."
)


@dataclass
class LanguageGuard:
    """Dialogue hook that handles languages Buddy cannot serve."""

    policy: LanguagePolicy = field(default_factory=LanguagePolicy)
    answered_in_english: bool = False

    def hook(self, text: str, context: ConversationContext
             ) -> DialogueReply | None:
        """Reply honestly when the language is not supported."""

        lang = detect_language(text)

        if lang in ("ta", "ta_latn", "mixed"):
            base = "ta"
        else:
            return None

        if base not in self.policy.understood:
            return DialogueReply(UNSUPPORTED_UNDERSTOOD, "language",
                                 "unsupported_language")

        if base not in self.policy.spoken:
            self.answered_in_english = True

            return None

        return None


def supported(languages: Iterable[str], lang: str) -> bool:
    """Convenience membership test."""

    return lang in set(languages)

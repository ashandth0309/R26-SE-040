"""Turn recognised speech into a command, strictly.

Rules (movement is safety critical):

* The whole utterance must match a known phrase after removing polite
  filler words.  Substring matching is never used for motion.
* Negated or conditional speech ("don't go forward", "if you go forward")
  is rejected, not executed.
* STOP is the exception: it is deliberately easy to trigger, because a
  false stop is harmless and a missed stop is not.
* Single ambiguous words such as "go", "left" or "back" are not commands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class VoiceCommand(str, Enum):
    """Commands the voice layer can issue."""

    STOP = "stop"
    FORWARD = "forward"
    BACKWARD = "backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    PATROL = "patrol"
    GO_TO = "go_to"
    CANCEL = "cancel"
    PAUSE = "pause"
    RESUME = "resume"


MOTION_COMMANDS = frozenset({
    VoiceCommand.FORWARD,
    VoiceCommand.BACKWARD,
    VoiceCommand.TURN_LEFT,
    VoiceCommand.TURN_RIGHT,
})

NAVIGATION_COMMANDS = frozenset({
    VoiceCommand.PATROL,
    VoiceCommand.GO_TO,
    VoiceCommand.CANCEL,
    VoiceCommand.PAUSE,
    VoiceCommand.RESUME,
})

FILLER = frozenset({
    "buddy", "please", "hey", "now", "can", "you", "could", "just",
    "would", "kindly", "the", "a", "ok", "okay",
})

NEGATIONS = frozenset({
    "don't", "dont", "do", "not", "never", "no", "didn't", "didnt",
    "if", "when", "whenever", "unless", "should", "shouldn't",
})

STOP_WORDS = frozenset({"stop", "halt", "freeze"})

PHRASES: dict[str, VoiceCommand] = {
    "go forward": VoiceCommand.FORWARD,
    "move forward": VoiceCommand.FORWARD,
    "walk forward": VoiceCommand.FORWARD,
    "forward": VoiceCommand.FORWARD,
    "come back": VoiceCommand.BACKWARD,
    "go back": VoiceCommand.BACKWARD,
    "go backward": VoiceCommand.BACKWARD,
    "go backwards": VoiceCommand.BACKWARD,
    "move back": VoiceCommand.BACKWARD,
    "move backward": VoiceCommand.BACKWARD,
    "back up": VoiceCommand.BACKWARD,
    "reverse": VoiceCommand.BACKWARD,
    "go reverse": VoiceCommand.BACKWARD,
    "turn left": VoiceCommand.TURN_LEFT,
    "rotate left": VoiceCommand.TURN_LEFT,
    "turn right": VoiceCommand.TURN_RIGHT,
    "rotate right": VoiceCommand.TURN_RIGHT,
    "start patrol": VoiceCommand.PATROL,
    "begin patrol": VoiceCommand.PATROL,
    "go on patrol": VoiceCommand.PATROL,
    "patrol": VoiceCommand.PATROL,
    "cancel": VoiceCommand.CANCEL,
    "cancel navigation": VoiceCommand.CANCEL,
    "abort": VoiceCommand.CANCEL,
    "pause": VoiceCommand.PAUSE,
    "pause navigation": VoiceCommand.PAUSE,
    "resume": VoiceCommand.RESUME,
    "resume navigation": VoiceCommand.RESUME,
}

WAKE_PHRASES = frozenset({
    "buddy", "hey buddy", "hello buddy", "buddy wake up",
    "wake up buddy", "wake up",
})

SLEEP_PHRASES = frozenset({
    "go to sleep", "buddy go to sleep", "sleep", "good night buddy",
    "go to sleep buddy",
})

_TOKEN = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class ParsedCommand:
    """A command recognised in speech."""

    kind: VoiceCommand
    text: str
    location: str | None = None


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens."""

    return _TOKEN.findall(text.lower())


def is_wake(text: str) -> bool:
    """True for an explicit wake phrase."""

    return " ".join(tokenize(text)) in WAKE_PHRASES


def is_sleep(text: str) -> bool:
    """True for an explicit sleep request."""

    return " ".join(tokenize(text)) in SLEEP_PHRASES


def is_stop(text: str) -> bool:
    """True if the speech contains a stop word.

    Intentionally loose: stopping by mistake is safe.
    """

    return any(t in STOP_WORDS for t in tokenize(text))


def parse_command(
    text: str,
    known_locations: tuple[str, ...] = (),
) -> ParsedCommand | None:
    """Return the command for ``text``, or None if it is not one."""

    tokens = tokenize(text)

    if not tokens:
        return None

    if any(t in STOP_WORDS for t in tokens):
        return ParsedCommand(VoiceCommand.STOP, text)

    if any(t in NEGATIONS for t in tokens):
        return None

    core = [t for t in tokens if t not in FILLER]
    phrase = " ".join(core)

    if phrase in PHRASES:
        return ParsedCommand(PHRASES[phrase], text)

    if len(core) >= 3 and core[0] == "go" and core[1] == "to":
        wanted = " ".join(core[2:])

        for name in known_locations:
            if " ".join(tokenize(name)) == wanted:
                return ParsedCommand(VoiceCommand.GO_TO, text, name)

    return None


def wake_grammar() -> list[str]:
    """Phrases for a restricted recogniser while sleeping."""

    return sorted(WAKE_PHRASES | {"stop", "buddy stop"}) + ["[unk]"]


def command_grammar(known_locations: tuple[str, ...] = ()) -> list[str]:
    """Phrases for a restricted recogniser while awake."""

    phrases = set(PHRASES)
    phrases |= {f"buddy {p}" for p in PHRASES}
    phrases |= {"stop", "buddy stop", "halt", "freeze", "emergency stop"}
    phrases |= SLEEP_PHRASES

    for name in known_locations:
        phrases.add(f"go to {name}")
        phrases.add(f"go to the {name}")

    return sorted(phrases) + ["[unk]"]

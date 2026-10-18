"""Turn a model's suggested action into a confirmed, validated command.

The language model may *suggest* ("ACTION: go forward").  It never acts.
This gate:

1. Accepts only suggestions that parse as a known motion command (never
   stop; stop does not need suggesting).
2. Turns the suggestion into a question to the user.
3. Releases the command only if the user's very next utterance is a clear
   yes within a short window.  Anything else cancels it.

The released command still goes through ``CommandRouter`` (awake check,
confidence, motion enabled) and then the safety supervisor.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from buddy_core.voice.voice_commands import (
    MOTION_COMMANDS,
    ParsedCommand,
    parse_command,
    tokenize,
)

YES = frozenset({"yes", "yeah", "yep", "sure", "ok", "okay", "please",
                 "do it", "go ahead", "yes please"})
SPOKEN = {
    "forward": "go forward", "backward": "go back",
    "turn_left": "turn left", "turn_right": "turn right",
}


@dataclass
class ActionGate:
    """Holds one suggested action awaiting confirmation."""

    window_s: float = 15.0
    clock: Callable[[], float] = time.monotonic
    pending: ParsedCommand | None = None
    expires: float = 0.0

    def propose(self, suggestion: str | None) -> str | None:
        """Register a suggestion; return the question to append, or None.

        Unknown, empty or non-motion suggestions are dropped.
        """

        self.pending = None

        if not suggestion:
            return None

        parsed = parse_command(suggestion)

        if parsed is None or parsed.kind not in MOTION_COMMANDS:
            return None

        self.pending = parsed
        self.expires = self.clock() + self.window_s

        return f"Do you want me to {SPOKEN[parsed.kind.value]}?"

    def confirm(self, text: str) -> ParsedCommand | None:
        """Called with the next utterance; returns the command on yes."""

        pending, self.pending = self.pending, None

        if pending is None or self.clock() > self.expires:
            return None

        if " ".join(tokenize(text)) in YES:
            return pending

        return None

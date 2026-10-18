"""Rare, polite suggestions that remember being told no.

Suggestions are about the *person* (drink water, stretch).  They never
start robot motion.  Rules:

* Never while muted, silenced, busy or right after the user spoke.
* A minimum gap and a daily cap.
* "Not now" snoozes that suggestion for a day; "don't suggest that again"
  blocks it for good.  Permanent blocks are kept with the memory manager
  when the user has consented, otherwise for the session.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from buddy_core.voice.dialogue_manager import (
    ConversationContext,
    DialogueReply,
)
from buddy_core.voice.memory_store import CompanionMemoryManager


@dataclass(frozen=True)
class Suggestion:
    """One thing Buddy might suggest."""

    sid: str
    text: str


CATALOG = (
    Suggestion("water", "You've been up a while. Maybe have a glass of "
                        "water?"),
    Suggestion("stretch", "Want to stretch your legs for a minute?"),
    Suggestion("rest_eyes", "It might help to rest your eyes for a "
                            "moment."),
    Suggestion("bedtime", "It's pretty late. Maybe wind down for the "
                          "night?"),
)


@dataclass(frozen=True)
class SuggestContext:
    """What Buddy knows right now."""

    hour: int
    awake_minutes: float
    emotion: str | None = None
    silenced: bool = False
    muted: bool = False
    busy: bool = False
    seconds_since_user_spoke: float = 1e9


@dataclass(frozen=True)
class ProactiveConfig:
    """Pacing."""

    min_gap_s: float = 1800.0
    max_per_day: int = 3
    snooze_s: float = 86400.0
    quiet_after_user_s: float = 20.0
    answer_window_s: float = 60.0


_YES = re.compile(r"^(?:yes|yeah|sure|okay|ok|good idea|will do)\b")
_NO = re.compile(r"^(?:no|nope|not now|maybe later|no thanks)\b")
_NEVER = re.compile(
    r"\b(?:don't suggest that again|stop suggesting|never suggest|"
    r"don't ask again|do not ask again)\b")


@dataclass
class ProactiveSuggester:
    """Decides when to speak up, and learns from the answer."""

    memory: CompanionMemoryManager | None = None
    config: ProactiveConfig = field(default_factory=ProactiveConfig)
    rng: random.Random = field(default_factory=random.Random)
    clock: Callable[[], float] = time.monotonic
    _last: float = -1e12
    _day_count: dict = field(default_factory=dict)
    _snooze: dict = field(default_factory=dict)
    _blocked: set = field(default_factory=set)
    _pending: tuple | None = None
    accepted: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.memory is not None:
            for s in CATALOG:
                if self.memory.recall(f"no_suggest_{s.sid}"):
                    self._blocked.add(s.sid)

    # ----- deciding -----

    def _eligible(self, ctx: SuggestContext) -> list[Suggestion]:
        wanted = []

        if ctx.awake_minutes >= 90:
            wanted.append("water")

        if ctx.awake_minutes >= 60:
            wanted.append("stretch")

        if ctx.emotion in ("tired", "stressed", "low_energy"):
            wanted.append("rest_eyes")

        if (ctx.hour >= 23 or ctx.hour < 5) and ctx.emotion in (
                "tired", "low_energy", None):
            wanted.append("bedtime")

        now = self.clock()

        return [
            s for s in CATALOG
            if s.sid in wanted and s.sid not in self._blocked
            and self._snooze.get(s.sid, 0) <= now
        ]

    def maybe_suggest(self, ctx: SuggestContext, day: int = 0
                      ) -> Suggestion | None:
        """A suggestion to speak now, or None."""

        cfg = self.config
        now = self.clock()

        if ctx.silenced or ctx.muted or ctx.busy:
            return None

        if ctx.seconds_since_user_spoke < cfg.quiet_after_user_s:
            return None

        if now - self._last < cfg.min_gap_s:
            return None

        if self._day_count.get(day, 0) >= cfg.max_per_day:
            return None

        options = self._eligible(ctx)

        if not options:
            return None

        choice = self.rng.choice(options)
        self._last = now
        self._day_count[day] = self._day_count.get(day, 0) + 1
        self._pending = (choice.sid, now + cfg.answer_window_s)

        return choice

    # ----- learning -----

    def hook(self, text: str, context: ConversationContext
             ) -> DialogueReply | None:
        """Interpret the user's answer to a suggestion."""

        lower = text.lower().strip(" .!?")

        if _NEVER.search(lower):
            sid = self._pending[0] if self._pending else None
            self._pending = None

            if sid is None:
                return None

            self._blocked.add(sid)

            if self.memory is not None:
                self.memory.remember(f"no_suggest_{sid}", "blocked",
                                     "preference")

            return DialogueReply("Okay, I won't suggest that again.",
                                 "proactive", "blocked")

        if self._pending is None:
            return None

        sid, expires = self._pending

        if self.clock() > expires:
            self._pending = None

            return None

        if _YES.match(lower):
            self._pending = None
            self.accepted.append(sid)

            return DialogueReply("Great.", "proactive", "accepted")

        if _NO.match(lower):
            self._pending = None
            self._snooze[sid] = self.clock() + self.config.snooze_s

            return DialogueReply("No problem.", "proactive", "snoozed")

        self._pending = None

        return None

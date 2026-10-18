"""One door to the speaker.

The voice node owns the speaker and its queue.  Everything else that wants
Buddy to speak (navigation results, vision suggestions, safety warnings)
goes through this arbiter, which publishes to ``/voice/say`` and is the
only publisher on that topic.  It decides *whether* something may be said
now; the voice node decides *when* it plays.

Gates, in order:

1. Shutting down: nothing.
2. Emergency speech always passes (and may interrupt).
3. Quiet mode ("let's be quiet"): only emergency and important status.
4. Emergency stop active: no proactive speech.
5. Proactive speech: not while the user is talking, Buddy is already
   speaking, a conversation is active, or within ``proactive_gap_s`` of the
   previous proactive line.
6. The same words are not said twice within ``dedupe_s``.
"""

from __future__ import annotations

import itertools
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from buddy_orchestrator.contracts import SpeechPriority


class Kind(str, Enum):
    """Why Buddy wants to speak."""

    EMERGENCY = "emergency"     # safety warning, may interrupt
    STATUS = "status"           # important robot status or a refusal
    REPLY = "reply"             # answer to something the user asked
    PROACTIVE = "proactive"     # unprompted companion speech


KIND_PRIORITY = {
    Kind.EMERGENCY: SpeechPriority.CRITICAL,
    Kind.STATUS: SpeechPriority.HIGH,
    Kind.REPLY: SpeechPriority.NORMAL,
    Kind.PROACTIVE: SpeechPriority.LOW,
}

DEFAULT_TTL_S = {
    Kind.EMERGENCY: 10.0,
    Kind.STATUS: 8.0,
    Kind.REPLY: 6.0,
    Kind.PROACTIVE: 5.0,
}


@dataclass(frozen=True)
class Utterance:
    """What is published on ``/voice/say``."""

    request_id: str
    source: str
    text: str
    kind: Kind
    priority: SpeechPriority
    ttl_s: float
    interrupt: bool


@dataclass
class SpeechPolicy:
    dedupe_s: float = 5.0
    proactive_gap_s: float = 20.0
    max_text_chars: int = 300


@dataclass
class SpeechContext:
    """What the arbiter needs to know about the moment."""

    shutting_down: bool = False
    emergency_stop: bool = False
    quiet_mode: bool = False
    user_speaking: bool = False
    buddy_speaking: bool = False
    conversation_active: bool = False


class SpeechArbiter:
    """Decides whether speech may be requested, then requests it."""

    def __init__(self, sink: Callable[[Utterance], None], *,
                 policy: SpeechPolicy | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._sink = sink
        self.policy = policy or SpeechPolicy()
        self._clock = clock
        self._ids = itertools.count(1)
        self._recent: deque[tuple[float, str]] = deque(maxlen=32)
        self._last_proactive = -1e12
        self.context = SpeechContext()
        self.sent = 0
        self.dropped: dict[str, int] = {}

    def _drop(self, why: str) -> bool:
        self.dropped[why] = self.dropped.get(why, 0) + 1

        return False

    def say(self, kind: Kind, text: str, source: str) -> bool:
        """Ask for speech.  Returns True if it was published."""

        text = " ".join((text or "").split())[: self.policy.max_text_chars]
        ctx = self.context
        now = self._clock()

        if not text:
            return self._drop("empty")

        if ctx.shutting_down:
            return self._drop("shutting_down")

        if kind is not Kind.EMERGENCY:
            if ctx.quiet_mode and kind in (Kind.REPLY, Kind.PROACTIVE):
                return self._drop("quiet_mode")

            if kind is Kind.PROACTIVE:
                if ctx.emergency_stop:
                    return self._drop("emergency_stop")

                if ctx.user_speaking or ctx.buddy_speaking \
                        or ctx.conversation_active:
                    return self._drop("conversation_busy")

                if now - self._last_proactive < self.policy.proactive_gap_s:
                    return self._drop("proactive_gap")

            if any(t == text and now - ts < self.policy.dedupe_s
                   for ts, t in self._recent):
                return self._drop("duplicate")

        if kind is Kind.PROACTIVE:
            self._last_proactive = now

        self._recent.append((now, text))
        utterance = Utterance(
            request_id=f"coord-{next(self._ids)}", source=source,
            text=text, kind=kind, priority=KIND_PRIORITY[kind],
            ttl_s=DEFAULT_TTL_S[kind],
            interrupt=kind is Kind.EMERGENCY)
        self._sink(utterance)
        self.sent += 1

        return True

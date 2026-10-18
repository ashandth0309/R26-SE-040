"""Single-speaker manager: one voice, never overlapping.

All speech goes through ``SpeakerManager``.  It owns a priority queue and a
playback backend, and guarantees that at most one utterance plays at a
time.  It is driven by ``tick()`` (called from a ROS timer or a test) so
that cancelling never has to wait on a blocking ``aplay`` call.

Priorities (lower number plays first):

* ``CRITICAL``  safety acknowledgements such as "Stopping."  These cancel
  whatever is playing.
* ``HIGH``      direct answers.
* ``NORMAL``    ordinary chat.
* ``LOW``       backchannels and fillers.  Dropped if anything else is
  playing or queued, never queued behind real speech.
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol

import numpy as np


class Priority(IntEnum):
    """Utterance priority, lower plays first."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class PlaybackBackend(Protocol):
    """Non-blocking audio output."""

    def start(self, samples: np.ndarray, rate: int, gain: float) -> None:
        """Begin playing; raises on device failure."""

    def is_playing(self) -> bool:
        """True while audio is still being played."""

    def stop(self) -> None:
        """Stop immediately."""


@dataclass(order=True)
class _Item:
    sort_key: tuple
    utterance: "Utterance" = field(compare=False)


@dataclass(frozen=True)
class Utterance:
    """One thing to say."""

    uid: int
    text: str
    samples: np.ndarray
    rate: int
    priority: Priority
    volume: float
    tag: str = ""


@dataclass(frozen=True)
class SpeakerConfig:
    """Limits for the speaker."""

    max_volume: float = 0.8
    default_volume: float = 0.6
    max_queue: int = 6
    max_start_failures: int = 2
    failure_cooldown_s: float = 5.0
    max_utterance_s: float = 30.0


class SpeakerManager:
    """Queue, play, cancel; never overlap."""

    def __init__(
        self,
        backend: PlaybackBackend,
        *,
        config: SpeakerConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_start: Callable[[Utterance], None] | None = None,
        on_finish: Callable[[Utterance, bool], None] | None = None,
        fallback: PlaybackBackend | None = None,
    ) -> None:
        self.backend = backend
        self.fallback = fallback
        self.config = config or SpeakerConfig()
        self._clock = clock
        self._heap: list[_Item] = []
        self._counter = itertools.count(1)
        self._current: Utterance | None = None
        self._current_end = 0.0
        self._on_start = on_start
        self._on_finish = on_finish
        self._failures = 0
        self._blocked_until = 0.0
        self.muted = False
        self.backend_failed = False
        self.dropped = 0

    # ----- state -----

    @property
    def speaking(self) -> bool:
        """True while an utterance is playing."""

        return self._current is not None

    @property
    def current(self) -> Utterance | None:
        """The utterance now playing, if any."""

        return self._current

    @property
    def queued(self) -> int:
        """Number of utterances waiting."""

        return len(self._heap)

    def effective_gain(self, volume: float) -> float:
        """Requested volume limited by the configured cap."""

        return max(0.0, min(volume, self.config.max_volume))

    # ----- input -----

    def say(
        self,
        text: str,
        samples: np.ndarray,
        rate: int,
        *,
        priority: Priority = Priority.NORMAL,
        volume: float | None = None,
        tag: str = "",
    ) -> int | None:
        """Queue speech.  Returns an id, or None if it was not accepted."""

        if self.muted or samples.size == 0:
            return None

        if samples.size / rate > self.config.max_utterance_s:
            samples = samples[: int(rate * self.config.max_utterance_s)]

        busy = self.speaking or bool(self._heap)

        if priority is Priority.LOW and busy:
            self.dropped += 1

            return None

        uid = next(self._counter)
        item = Utterance(
            uid,
            text,
            samples,
            rate,
            priority,
            self.config.default_volume if volume is None else volume,
            tag,
        )

        if priority is Priority.CRITICAL:
            self._drop_queue()
            self._interrupt_current()
        elif len(self._heap) >= self.config.max_queue:
            if not self._drop_lowest(priority):
                self.dropped += 1

                return None

        heapq.heappush(self._heap, _Item((int(priority), uid), item))

        return uid

    def _drop_queue(self) -> None:
        self.dropped += len(self._heap)
        self._heap.clear()

    def _drop_lowest(self, incoming: Priority) -> bool:
        worst = max(self._heap, key=lambda i: i.sort_key)

        if worst.utterance.priority <= incoming:
            return False

        self._heap.remove(worst)
        heapq.heapify(self._heap)
        self.dropped += 1

        return True

    # ----- cancel -----

    def _interrupt_current(self) -> None:
        if self._current is None:
            return

        item, self._current = self._current, None

        try:
            self.backend.stop()
        except Exception:
            pass

        if self._on_finish is not None:
            self._on_finish(item, False)

    def cancel_all(self) -> None:
        """Stop talking now and clear the queue."""

        self._drop_queue()
        self._interrupt_current()

    def cancel_low_priority(self) -> None:
        """Drop queued and playing backchannels / fillers."""

        self._heap = [
            i for i in self._heap if i.utterance.priority < Priority.LOW
        ]
        heapq.heapify(self._heap)

        if self._current and self._current.priority is Priority.LOW:
            self._interrupt_current()

    def mute(self, muted: bool = True) -> None:
        """Mute speech output entirely (privacy / silence mode)."""

        self.muted = muted

        if muted:
            self.cancel_all()

    # ----- driving -----

    def tick(self) -> None:
        """Advance: finish the current item, start the next."""

        now = self._clock()

        if self._current is not None:
            try:
                playing = self.backend.is_playing()
            except Exception:
                playing = False

            if playing and now < self._current_end:
                return

            if playing:
                self._interrupt_current()
            else:
                done, self._current = self._current, None

                if self._on_finish is not None:
                    self._on_finish(done, True)

        if now < self._blocked_until or not self._heap:
            return

        item = heapq.heappop(self._heap).utterance
        self._start(item, now)

    def _start(self, item: Utterance, now: float) -> None:
        gain = self.effective_gain(item.volume)
        backend = self.fallback if self.backend_failed and self.fallback \
            else self.backend

        try:
            backend.start(item.samples, item.rate, gain)
        except Exception:
            self._failures += 1

            if self._failures > self.config.max_start_failures:
                self._handle_backend_failure(now)

            if self._on_finish is not None:
                self._on_finish(item, False)

            return

        self._failures = 0
        self._current = item
        self._current_end = now + item.samples.size / item.rate + 1.0

        if self._on_start is not None:
            self._on_start(item)

    def _handle_backend_failure(self, now: float) -> None:
        self._failures = 0

        if self.fallback is not None and not self.backend_failed:
            self.backend, self.fallback = self.fallback, self.backend
            self.backend_failed = False
        else:
            self.backend_failed = True
            self._blocked_until = now + self.config.failure_cooldown_s
            self._drop_queue()

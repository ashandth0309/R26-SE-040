"""When is it appropriate to greet a recognised person?"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class GreetingConfig:
    min_absence_s: float = 120.0     # must have been out of view this long
    cooldown_s: float = 4 * 3600.0   # at most one greeting per person
    confirm_s: float = 1.0           # in view this long before greeting


class GreetingPolicy:
    """Greets each consenting, recognised person rarely, never every frame.

    It only decides; the voice engine chooses the words and may still stay
    silent (quiet mode, someone talking, ...).
    """

    def __init__(self, cfg: GreetingConfig | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg, self._clock = cfg or GreetingConfig(), clock
        self._first_in_view: dict[str, float] = {}
        self._last_seen: dict[str, float] = {}
        self._last_greeted: dict[str, float] = {}
        self._was_absent: dict[str, bool] = {}

    def observe(self, person_id: str, settings: dict) -> bool:
        """Call once per perception update while ``person_id`` is in view.
        Returns True when a greeting is appropriate right now."""

        now = self._clock()
        last = self._last_seen.get(person_id)
        gap = None if last is None else now - last
        self._last_seen[person_id] = now

        if gap is None or gap >= self.cfg.min_absence_s:
            self._first_in_view[person_id] = now
            self._was_absent[person_id] = True
        elif person_id not in self._first_in_view:
            self._first_in_view[person_id] = now

        if not settings.get("greet", True) or \
                not settings.get("recognition", True):
            return False

        if not self._was_absent.get(person_id, True):
            return False

        if now - self._first_in_view[person_id] < self.cfg.confirm_s:
            return False

        greeted = self._last_greeted.get(person_id)

        if greeted is not None and now - greeted < self.cfg.cooldown_s:
            return False

        return True

    def mark_greeted(self, person_id: str) -> None:
        self._last_greeted[person_id] = self._clock()
        self._was_absent[person_id] = False

    def forget(self, person_id: str) -> None:
        for table in (self._first_in_view, self._last_seen,
                      self._last_greeted, self._was_absent):
            table.pop(person_id, None)

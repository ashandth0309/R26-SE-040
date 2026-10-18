"""Buddy's personality settings and repetition control.

Settings come from YAML so they can be changed without code.  The engine
adds small, honest touches (a friendly opener now and then, a shorter
reply when verbosity is low) and refuses to say nearly the same thing
twice in a row.  It does not invent a backstory or claim feelings.
"""

from __future__ import annotations

import random
import re
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path

from buddy_core.voice.variation import VariantPicker


@dataclass(frozen=True)
class PersonalityConfig:
    """Tunable traits, each 0 to 1 unless noted."""

    name: str = "Buddy"
    warmth: float = 0.7
    playfulness: float = 0.4
    verbosity: float = 0.5
    opener_chance: float = 0.25
    max_sentences: int = 3
    repeat_threshold: float = 0.8
    history: int = 6


def load_personality(path: str | Path | None = None) -> PersonalityConfig:
    """Read YAML; bad or missing file gives defaults."""

    base = PersonalityConfig()

    if path is None:
        return base

    try:
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return base

    if not isinstance(data, dict):
        return base

    values = {}

    for f in fields(PersonalityConfig):
        if f.name not in data:
            continue

        want = type(getattr(base, f.name))

        try:
            value = want(data[f.name])
        except (TypeError, ValueError):
            continue

        if want is float:
            value = min(1.0, max(0.0, value))

        values[f.name] = value

    return PersonalityConfig(**{**base.__dict__, **values})


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def similarity(a: str, b: str) -> float:
    """Word-set overlap between two replies, 0 to 1."""

    ta, tb = _tokens(a), _tokens(b)

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


OPENERS = ("Sure.", "Okay.", "Alright.", "Right.", "Of course.")


class BuddyPersonalityEngine:
    """Applies personality and checks for repetition."""

    def __init__(self, config: PersonalityConfig | None = None, *,
                 rng: random.Random | None = None) -> None:
        self.config = config or PersonalityConfig()
        self._rng = rng or random.Random()
        self._picker = VariantPicker(self._rng, memory=3)
        self._recent: deque[str] = deque(maxlen=self.config.history)

    def is_repetitive(self, text: str) -> bool:
        """True if ``text`` is nearly the same as a recent reply."""

        limit = self.config.repeat_threshold

        return any(similarity(text, old) >= limit for old in self._recent)

    def shorten(self, text: str) -> str:
        """Trim to the sentence limit, higher verbosity keeps more."""

        limit = max(1, round(self.config.max_sentences
                             * (0.5 + self.config.verbosity)))
        parts = re.split(r"(?<=[.!?])\s+", text.strip())

        return " ".join(parts[:limit])

    def finish(self, text: str, *, allow_opener: bool = True,
               alternatives: list[str] | None = None) -> str:
        """Final polish: trim, avoid repeats, maybe a light opener."""

        text = self.shorten(text)

        if self.is_repetitive(text) and alternatives:
            for alt in alternatives:
                alt = self.shorten(alt)

                if not self.is_repetitive(alt):
                    text = alt
                    break

        if (
            allow_opener
            and not self.is_repetitive(text)
            and len(text.split()) > 6
            and self._rng.random() < self.config.opener_chance
            and not text.startswith(OPENERS)
        ):
            text = self._picker.pick("opener", OPENERS) + " " + text

        self._recent.append(text)

        return text

    def forget_history(self) -> None:
        """Clear the recent-reply memory."""

        self._recent.clear()
        self._picker.forget()

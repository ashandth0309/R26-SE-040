"""Pick wording without repeating yourself."""

from __future__ import annotations

import random
from collections import deque


class VariantPicker:
    """Choose among variants, avoiding recent repeats per key."""

    def __init__(self, rng: random.Random | None = None, memory: int = 3):
        self._rng = rng or random.Random()
        self._memory = memory
        self._recent: dict[str, deque] = {}

    def pick(self, key: str, variants: list[str] | tuple[str, ...]) -> str:
        """Return a variant not used in the last ``memory`` picks."""

        if not variants:
            return ""

        recent = self._recent.setdefault(key, deque(maxlen=self._memory))
        fresh = [v for v in variants if v not in recent]
        pool = fresh or [v for v in variants if v != (
            recent[-1] if recent else None)] or list(variants)
        choice = self._rng.choice(pool)
        recent.append(choice)

        return choice

    def forget(self) -> None:
        """Clear history."""

        self._recent.clear()

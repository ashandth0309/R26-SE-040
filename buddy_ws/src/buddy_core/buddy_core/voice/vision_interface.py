"""Interface to the vision module (Sobiya's component).

The voice engine only *consumes* vision information.  It never opens the
camera and loads no vision models, so there is exactly one camera owner
(the vision node).  Until that module is wired in, ``MockVision`` supplies
scripted values.  The ROS-side provider is added in Phase 8 and is
UNVERIFIED until the vision branch is merged.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VisionSnapshot:
    """What vision currently reports.  Everything is optional."""

    stamp: float
    person_visible: bool | None = None
    known_person: str | None = None
    facial_expression: str | None = None
    expression_confidence: float | None = None


class VisionProvider(Protocol):
    """Source of vision hints."""

    def latest(self) -> VisionSnapshot | None:
        """Most recent snapshot, or None if vision is unavailable."""


class MockVision:
    """Scripted provider for tests and for running without a camera."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._snapshot: VisionSnapshot | None = None

    def set(self, **fields) -> None:
        """Replace the snapshot."""

        self._snapshot = VisionSnapshot(self._clock(), **fields)

    def clear(self) -> None:
        """Simulate vision being unavailable."""

        self._snapshot = None

    def latest(self) -> VisionSnapshot | None:
        return self._snapshot


def fresh(snapshot: VisionSnapshot | None, now: float,
          max_age_s: float = 3.0) -> VisionSnapshot | None:
    """Return ``snapshot`` only if it is recent enough to trust."""

    if snapshot is None or now - snapshot.stamp > max_age_s:
        return None

    return snapshot

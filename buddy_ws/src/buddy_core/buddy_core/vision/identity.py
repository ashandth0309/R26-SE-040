"""Turning embeddings into an identity decision, with honest "unknown".

States, from weakest to strongest claim:

* ``NO_FACE``    nothing to look at (the caller sets this, not the matcher)
* ``UNKNOWN``    a face, but no enrolled person is close enough
* ``UNCERTAIN``  close to someone, not close enough to say; never named
* ``RECOGNIZED`` close enough and clearly better than anyone else

A high similarity is evidence, not proof.  Defaults are deliberately
conservative starting values that have NOT been calibrated on the people
who will use this robot; run ``identity_eval`` on consented captures first.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from buddy_core.vision.embedding import l2_normalize


class IdentityState(str, Enum):
    NO_FACE = "no_face"
    UNKNOWN = "unknown"
    UNCERTAIN = "uncertain"
    RECOGNIZED = "recognized"


@dataclass(frozen=True)
class MatchConfig:
    accept: float = 0.50     # cosine; zoo suggests 0.363 for 1:1 checks
    floor: float = 0.363     # below this nobody is even a candidate
    margin: float = 0.08     # best must beat the runner-up person by this
    top_k: int = 3           # templates averaged per person


@dataclass(frozen=True)
class MatchResult:
    state: IdentityState
    person_id: str | None      # set only for RECOGNIZED
    candidate: str | None      # best guess for UNCERTAIN (never spoken)
    score: float | None
    margin: float | None


UNKNOWN_RESULT = MatchResult(IdentityState.UNKNOWN, None, None, None, None)


def person_score(templates: np.ndarray, probe: np.ndarray,
                 top_k: int) -> float:
    sims = np.sort(templates @ probe)[::-1]

    return float(np.mean(sims[:max(1, min(top_k, len(sims)))]))


def match(probe: np.ndarray, gallery: dict[str, np.ndarray],
          cfg: MatchConfig = MatchConfig()) -> MatchResult:
    """Compare one embedding with every enrolled person."""

    if not gallery:
        return UNKNOWN_RESULT

    probe = l2_normalize(probe)
    scores = {pid: person_score(np.asarray(t, np.float32).reshape(
        -1, probe.size), probe, cfg.top_k) for pid, t in gallery.items()
        if len(t)}

    if not scores:
        return UNKNOWN_RESULT

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best_id, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else None
    margin = None if second is None else best - second

    if best < cfg.floor:
        return MatchResult(IdentityState.UNKNOWN, None, None, best, margin)

    if best >= cfg.accept and (margin is None or margin >= cfg.margin):
        return MatchResult(IdentityState.RECOGNIZED, best_id, best_id,
                           best, margin)

    return MatchResult(IdentityState.UNCERTAIN, None, best_id, best,
                       margin)


@dataclass
class TrackIdentityConfig:
    confirm_hits: int = 3        # agreeing results before naming someone
    release_hits: int = 4        # contrary results before un-naming
    retry_s: float = 0.5         # re-embed interval while undecided
    unknown_retry_s: float = 3.0  # slower retries once looks unknown
    verify_s: float = 5.0        # periodic re-check of a named track
    max_window: int = 8


class TrackIdentity:
    """Per-track voting so one noisy frame cannot name or un-name anyone."""

    def __init__(self, cfg: TrackIdentityConfig | None = None) -> None:
        self.cfg = cfg or TrackIdentityConfig()
        self.state = IdentityState.UNKNOWN
        self.person_id: str | None = None
        self.score: float | None = None
        self.evaluated = False
        self._recent: list[MatchResult] = []
        self._last_attempt = -1e9
        self.attempts = 0

    def due(self, now: float) -> bool:
        gap = now - self._last_attempt

        if self.state == IdentityState.RECOGNIZED:
            return gap >= self.cfg.verify_s

        if self.state == IdentityState.UNKNOWN and \
                len(self._recent) >= self.cfg.confirm_hits:
            return gap >= self.cfg.unknown_retry_s

        return gap >= self.cfg.retry_s

    def add(self, result: MatchResult, now: float) -> None:
        self._last_attempt = now
        self.attempts += 1
        self.evaluated = True
        self._recent.append(result)
        self._recent = self._recent[-self.cfg.max_window:]
        self._update()

    def _update(self) -> None:
        recent = self._recent

        if self.person_id is not None:
            contrary = [r for r in recent[-self.cfg.release_hits:]
                        if r.person_id != self.person_id]

            if len(contrary) >= self.cfg.release_hits:
                self.person_id = None
                self.state = IdentityState.UNCERTAIN
            else:
                self.score = recent[-1].score

                return

        last = recent[-self.cfg.confirm_hits:]
        named = [r.person_id for r in last if r.person_id]

        if len(last) >= self.cfg.confirm_hits and len(named) == len(last) \
                and len(set(named)) == 1:
            self.person_id = named[0]
            self.state = IdentityState.RECOGNIZED
            self.score = float(np.mean([r.score for r in last]))

            return

        newest = recent[-1]
        self.score = newest.score
        self.person_id = None
        leaning = any(r.state in (IdentityState.UNCERTAIN,
                                  IdentityState.RECOGNIZED) for r in last)
        self.state = IdentityState.UNCERTAIN if leaning \
            else IdentityState.UNKNOWN

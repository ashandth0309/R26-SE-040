"""Multi-object tracker for faces (or people) on a Raspberry Pi budget.

Time-based, so it behaves the same at any frame rate: association by box
overlap plus centre distance, a constant-velocity guess while a target is
briefly hidden, and ids that are never reused within a run.  A tracker id
(``session_person_N``) is a *transient* label for one visible stretch; it is
not an identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - optional on the Pi
    linear_sum_assignment = None

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + \
        (b[2] - b[0]) * (b[3] - b[1]) - inter

    return inter / union if union > 0 else 0.0


def centre(b: Box) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def nms(boxes: list[Box], scores: list[float], thr: float) -> list[int]:
    """Indices kept after dropping overlapping duplicates."""

    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep: list[int] = []

    for i in order:
        if all(iou(boxes[i], boxes[j]) < thr for j in keep):
            keep.append(i)

    return keep


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"          # hidden; position is a guess


@dataclass
class TrackerConfig:
    n_init: int = 2                # detections before a track is confirmed
    lost_after_s: float = 0.35     # unseen this long -> LOST (hidden)
    max_age_s: float = 1.5         # keep a hidden track this long
    tentative_age_s: float = 0.6   # drop an unconfirmed track sooner
    min_iou: float = 0.1
    max_centre_dist: float = 0.6   # in units of the larger box side
    dedup_iou: float = 0.6
    velocity_smoothing: float = 0.5


@dataclass
class Track:
    track_id: int
    box: Box
    created: float
    last_seen: float
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    velocity: tuple[float, float] = (0.0, 0.0)  # px/s of the centre
    score: float = 0.0
    data: dict = field(default_factory=dict)

    @property
    def session_id(self) -> str:
        return f"session_person_{self.track_id}"

    def predicted(self, now: float, horizon: float = 0.5) -> Box:
        dt = max(0.0, min(horizon, now - self.last_seen))
        dx, dy = self.velocity[0] * dt, self.velocity[1] * dt
        x1, y1, x2, y2 = self.box

        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


class MultiTracker:
    """Tracks boxes across calls to ``update``."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.cfg = config or TrackerConfig()
        self.tracks: list[Track] = []
        self._next_id = 1
        self.created_total = 0

    # ----- association -----

    def _cost(self, track: Track, box: Box, now: float) -> float | None:
        pred = track.predicted(now)
        overlap = iou(pred, box)
        (pcx, pcy), (dcx, dcy) = centre(pred), centre(box)
        side = max(pred[2] - pred[0], pred[3] - pred[1],
                   box[2] - box[0], box[3] - box[1], 1.0)
        dist = float(np.hypot(pcx - dcx, pcy - dcy)) / side

        if overlap < self.cfg.min_iou and dist > self.cfg.max_centre_dist:
            return None

        return 0.7 * (1.0 - overlap) + 0.3 * min(dist, 1.0)

    def _assign(self, boxes: list[Box], now: float
                ) -> list[tuple[int, int]]:
        costs = np.full((len(self.tracks), len(boxes)), 1e6)

        for ti, track in enumerate(self.tracks):
            for di, box in enumerate(boxes):
                c = self._cost(track, box, now)

                if c is not None:
                    costs[ti, di] = c

        if not self.tracks or not boxes:
            return []

        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(costs)
            pairs = list(zip(rows.tolist(), cols.tolist()))
        else:
            pairs, used_t, used_d = [], set(), set()

            for flat in np.argsort(costs, axis=None):
                ti, di = divmod(int(flat), costs.shape[1])

                if ti in used_t or di in used_d:
                    continue

                pairs.append((ti, di))
                used_t.add(ti)
                used_d.add(di)

        return [(t, d) for t, d in pairs if costs[t, d] < 1e5]

    # ----- public -----

    def update(self, boxes: list[Box], now: float,
               scores: list[float] | None = None) -> list[Track]:
        """Feed one detection pass.  Returns tracks seen *this* pass."""

        scores = scores if scores is not None else [1.0] * len(boxes)
        keep = nms(boxes, scores, self.cfg.dedup_iou)
        boxes = [boxes[i] for i in keep]
        scores = [scores[i] for i in keep]
        matches = self._assign(boxes, now)
        matched_t = {t for t, _ in matches}
        matched_d = {d for _, d in matches}
        seen: list[Track] = []

        for ti, di in matches:
            track, box = self.tracks[ti], boxes[di]
            dt = max(1e-3, now - track.last_seen)
            (ocx, ocy), (ncx, ncy) = centre(track.box), centre(box)
            a = self.cfg.velocity_smoothing
            vx = a * (ncx - ocx) / dt + (1 - a) * track.velocity[0]
            vy = a * (ncy - ocy) / dt + (1 - a) * track.velocity[1]
            track.velocity = (vx, vy)
            track.box, track.last_seen = box, now
            track.hits += 1
            track.score = scores[di]
            track.state = TrackState.CONFIRMED \
                if track.hits >= self.cfg.n_init else track.state
            seen.append(track)

        for di, box in enumerate(boxes):
            if di in matched_d:
                continue

            track = Track(self._next_id, box, now, now, score=scores[di])
            track.state = TrackState.CONFIRMED \
                if self.cfg.n_init <= 1 else TrackState.TENTATIVE
            self._next_id += 1
            self.created_total += 1
            self.tracks.append(track)
            seen.append(track)

        for ti, track in enumerate(list(self.tracks)):
            if ti not in matched_t and track not in seen:
                self._age(track, now)

        self._prune(now)

        return seen

    def coast(self, now: float) -> None:
        """No detection ran this frame: age tracks without new evidence."""

        for track in self.tracks:
            self._age(track, now)

        self._prune(now)

    def _age(self, track: Track, now: float) -> None:
        if track.state == TrackState.CONFIRMED and \
                now - track.last_seen > self.cfg.lost_after_s:
            track.state = TrackState.LOST

    def _prune(self, now: float) -> None:
        keep = []

        for t in self.tracks:
            age = now - t.last_seen
            limit = self.cfg.tentative_age_s \
                if t.state == TrackState.TENTATIVE else self.cfg.max_age_s

            if age <= limit:
                keep.append(t)

        self.tracks = keep

    def visible(self) -> list[Track]:
        return [t for t in self.tracks
                if t.state == TrackState.CONFIRMED]

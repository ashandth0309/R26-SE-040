"""From per-frame expression scores to a calm, honest signal.

Single frames flicker.  This module averages over a short window, refuses
to claim anything on poor frames or too few frames, requires a lead to
persist before calling it, holds a call through brief changes, and emits an
*event* only when the confirmed state actually changes (never once per
frame).  The default decision is made on four coarse groups (positive,
neutral, negative, surprise) because the underlying model is clearly
better at some seven-way labels (happy) than others (sad, fearful); see
``docs/evaluation``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from buddy_core.vision.expression import (
    GROUP_NAMES,
    GROUPS,
    LABELS,
    ExpressionScores,
)
from buddy_core.vision.faces import FaceDetection, FaceQuality


class ExprState(str, Enum):
    UNKNOWN = "unknown"       # nothing reliable to say
    TENTATIVE = "tentative"   # leading, not yet persistent
    CONFIRMED = "confirmed"   # persistent across frames and time


@dataclass
class SmoothingConfig:
    window_s: float = 4.0
    min_frames: int = 4
    min_confidence: float = 0.55
    min_margin: float = 0.15
    confirm_s: float = 1.5
    change_hold_s: float = 1.0
    max_gap_s: float = 3.0        # no usable frame this long -> UNKNOWN
    use_groups: bool = True
    min_detection_score: float = 0.80
    # A frame whose seven-way scores are nearly flat says nothing, and
    # group totals would still look decisive (four labels are negative).
    min_frame_top_prob: float = 0.35
    max_frame_entropy: float = 0.90


@dataclass(frozen=True)
class ExpressionEstimate:
    state: ExprState
    label: str | None             # group name or model label; None if unknown
    group: str | None
    confidence: float | None      # smoothed probability of that label
    frames: int                   # usable frames in the window
    reason: str                   # why UNKNOWN / TENTATIVE
    note: str = ("visible expression only; says nothing certain about "
                 "how the person feels")


def conditions(det: FaceDetection, quality: FaceQuality | None,
               cfg: SmoothingConfig) -> tuple[str, ...]:
    """Reasons this frame should not be used for expression."""

    reasons: list[str] = []

    if quality is None:
        return ("no_quality",)

    for r in quality.reasons:
        if r in ("too_dark", "too_bright", "low_contrast", "blurry",
                 "too_small", "head_turned", "partly_out_of_frame",
                 "empty_crop"):
            reasons.append(r)

    if det.score < cfg.min_detection_score:
        reasons.append("possible_occlusion")
    elif len(det.landmarks) == 5:
        (rx, ry), (lx, ly), (nx, ny), (rmx, rmy), (lmx, lmy) = \
            det.landmarks
        x1, y1, x2, y2 = det.box
        eyes_y, mouth_y = (ry + ly) / 2.0, (rmy + lmy) / 2.0
        inside = all(x1 - 0.1 * det.w <= px <= x2 + 0.1 * det.w and
                     y1 - 0.1 * det.h <= py <= y2 + 0.1 * det.h
                     for px, py in det.landmarks)

        if not (eyes_y < ny < mouth_y) or not inside:
            reasons.append("possible_occlusion")

    return tuple(dict.fromkeys(reasons))


class TrackExpression:
    """Smoother for one tracked face."""

    def __init__(self, cfg: SmoothingConfig | None = None) -> None:
        self.cfg = cfg or SmoothingConfig()
        self._win: deque[tuple[float, np.ndarray]] = deque()
        self._last_usable: float | None = None
        self._lead: str | None = None
        self._lead_since = 0.0
        self._confirmed: str | None = None
        self._rival: str | None = None
        self._rival_since = 0.0

    def _vector(self, scores: ExpressionScores) -> np.ndarray:
        if self.cfg.use_groups:
            g = scores.group_probs()

            return np.array([g[n] for n in GROUP_NAMES])

        return np.array(scores.probs)

    def _names(self) -> tuple[str, ...]:
        return GROUP_NAMES if self.cfg.use_groups else LABELS

    def observe(self, scores: ExpressionScores | None,
                bad_conditions: tuple[str, ...], now: float
                ) -> ExpressionEstimate:
        cfg = self.cfg

        if scores is not None and not bad_conditions and (
                scores.top_prob < cfg.min_frame_top_prob or
                scores.entropy > cfg.max_frame_entropy):
            bad_conditions = ("uninformative",)

        if scores is not None and not bad_conditions:
            self._win.append((now, self._vector(scores)))
            self._last_usable = now

        while self._win and now - self._win[0][0] > cfg.window_s:
            self._win.popleft()

        if self._last_usable is None or \
                now - self._last_usable > cfg.max_gap_s:
            self._reset()
            reason = bad_conditions[0] if bad_conditions else "no_face_data"

            return ExpressionEstimate(ExprState.UNKNOWN, None, None, None,
                                      0, reason)

        n = len(self._win)

        if n < cfg.min_frames:
            return ExpressionEstimate(ExprState.UNKNOWN, None, None, None,
                                      n, "too_few_frames")

        mean = np.mean([v for _, v in self._win], axis=0)
        order = np.argsort(-mean)
        names = self._names()
        top, second = names[order[0]], mean[order[1]]
        conf, margin = float(mean[order[0]]), float(mean[order[0]] - second)

        if conf < cfg.min_confidence or margin < cfg.min_margin:
            self._lead = None

            return self._hold(now, conf, n, "ambiguous")

        if top != self._lead:
            self._lead, self._lead_since = top, now

        if self._confirmed is None:
            if now - self._lead_since >= cfg.confirm_s:
                self._confirmed = top

                return self._estimate(ExprState.CONFIRMED, top, conf, n)

            return self._estimate(ExprState.TENTATIVE, top, conf, n,
                                  "not_persistent_yet")

        if top == self._confirmed:
            self._rival = None

            return self._estimate(ExprState.CONFIRMED, top, conf, n)

        if self._rival != top:
            self._rival, self._rival_since = top, now

        if now - self._rival_since >= cfg.change_hold_s:
            self._confirmed, self._rival = top, None

            return self._estimate(ExprState.CONFIRMED, top, conf, n)

        return self._estimate(ExprState.CONFIRMED, self._confirmed,
                              float(mean[names.index(self._confirmed)]), n)

    def _hold(self, now: float, conf: float, n: int, reason: str
              ) -> ExpressionEstimate:
        """Ambiguous window: keep an earlier confirmation, else unknown."""

        if self._confirmed is not None:
            names = self._names()
            mean = np.mean([v for _, v in self._win], axis=0)

            return self._estimate(
                ExprState.TENTATIVE, self._confirmed,
                float(mean[names.index(self._confirmed)]), n, reason)

        return ExpressionEstimate(ExprState.UNKNOWN, None, None, conf, n,
                                  reason)

    def _estimate(self, state: ExprState, label: str, conf: float, n: int,
                  reason: str = "") -> ExpressionEstimate:
        group = label if self.cfg.use_groups else GROUPS[label]

        return ExpressionEstimate(state, label, group, conf, n, reason)

    def _reset(self) -> None:
        self._win.clear()
        self._lead = self._confirmed = self._rival = None


@dataclass(frozen=True)
class ExpressionEvent:
    track_id: int
    kind: str                 # "expression_changed"
    previous: str | None
    label: str
    confidence: float
    stamp: float


@dataclass
class EventGateConfig:
    min_event_gap_s: float = 20.0
    emit_neutral: bool = False   # neutral is the baseline, not news


class ExpressionService:
    """Per-track smoothing plus a change-only event gate."""

    def __init__(self, smoothing: SmoothingConfig | None = None,
                 gate: EventGateConfig | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        self.smoothing = smoothing or SmoothingConfig()
        self.gate = gate or EventGateConfig()
        self._tracks: dict[int, TrackExpression] = {}
        self._announced: dict[int, str | None] = {}
        self._last_event: dict[int, float] = {}
        self.events_emitted = 0
        self.frames = 0

    def update(self, track_id: int, scores: ExpressionScores | None,
               det: FaceDetection, quality: FaceQuality | None,
               now: float) -> tuple[ExpressionEstimate,
                                    ExpressionEvent | None]:
        self.frames += 1
        track = self._tracks.setdefault(track_id,
                                        TrackExpression(self.smoothing))
        est = track.observe(scores, conditions(det, quality,
                                               self.smoothing), now)
        event = None

        if est.state is ExprState.CONFIRMED and est.label is not None:
            prev = self._announced.get(track_id)

            if est.label != prev:
                self._announced[track_id] = est.label
                quiet = est.label == "neutral" and not self.gate.emit_neutral
                last = self._last_event.get(track_id, -1e9)

                if not quiet and now - last >= self.gate.min_event_gap_s:
                    event = ExpressionEvent(track_id, "expression_changed",
                                            prev, est.label,
                                            est.confidence or 0.0, now)
                    self._last_event[track_id] = now
                    self.events_emitted += 1

        return est, event

    def forget(self, track_id: int) -> None:
        for table in (self._tracks, self._announced, self._last_event):
            table.pop(track_id, None)

    def prune(self, live_ids: set[int]) -> None:
        for tid in list(self._tracks):
            if tid not in live_ids:
                self.forget(tid)


__all__ = ["ExprState", "SmoothingConfig", "ExpressionEstimate",
           "TrackExpression", "ExpressionService", "ExpressionEvent",
           "EventGateConfig", "conditions"]

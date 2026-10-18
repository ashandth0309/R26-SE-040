"""Face detection + tracking + identity, scheduled for a Pi 4.

Per frame the pipeline does the cheapest thing that keeps the output
correct: detection runs on a downscaled image at a limited rate; between
detections the tracker coasts; the expensive embedding runs only for tracks
that need it, and only when the face is good enough.  Every timing it
reports was measured with ``perf_counter`` on the machine it ran on.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from buddy_core.vision.embedding import FaceEmbedder
from buddy_core.vision.faces import (
    FaceDetection,
    FaceDetector,
    FaceQuality,
    QualityConfig,
    assess_quality,
)
from buddy_core.vision.identity import (
    IdentityState,
    MatchConfig,
    MatchResult,
    TrackIdentity,
    TrackIdentityConfig,
    match,
)
from buddy_core.vision.tracker import MultiTracker, Track, TrackerConfig


@dataclass
class PipelineConfig:
    detect_width: int = 320          # detection runs at this width
    detect_min_interval_s: float = 0.1
    max_embeds_per_frame: int = 1
    recognition_enabled: bool = True
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    identity: TrackIdentityConfig = field(
        default_factory=TrackIdentityConfig)


@dataclass(frozen=True)
class TrackView:
    track_id: int
    session_id: str
    box: tuple[float, float, float, float]   # full-frame pixels
    visible: bool                            # False while only predicted
    score: float
    identity_state: IdentityState
    person_id: str | None
    identity_score: float | None
    quality: FaceQuality | None
    detection: FaceDetection | None = None


@dataclass(frozen=True)
class PipelineOutput:
    stamp: float
    tracks: tuple[TrackView, ...]
    detected: bool          # a detection pass ran for this frame
    faces_in_view: int      # confirmed, currently visible tracks
    no_face: bool           # detector ran and nothing is in view
    timings_ms: dict


class FaceTrackingPipeline:
    def __init__(self, detector: FaceDetector,
                 embedder: FaceEmbedder | None = None,
                 gallery: Callable[[], dict[str, np.ndarray]] | None = None,
                 config: PipelineConfig | None = None,
                 negatives: Callable[[], dict[str, np.ndarray]] | None
                 = None) -> None:
        self.cfg = config or PipelineConfig()
        self.detector, self.embedder = detector, embedder
        self.gallery = gallery or (lambda: {})
        self.negatives = negatives or (lambda: {})
        # A face is only "hidden" after it has been missed for clearly
        # longer than the gap between detection passes.
        self.cfg.tracker.lost_after_s = max(
            self.cfg.tracker.lost_after_s,
            2.5 * self.cfg.detect_min_interval_s)
        self.tracker = MultiTracker(self.cfg.tracker)
        self._ident: dict[int, TrackIdentity] = {}
        self._last_det: dict[int, FaceDetection] = {}
        self._last_detect_time = -1e9
        self.embeds = 0
        self.frames = 0
        self.detect_passes = 0

    # ----- helpers -----

    def _detect(self, image: np.ndarray) -> list[FaceDetection]:
        h, w = image.shape[:2]

        if w <= self.cfg.detect_width:
            return self.detector.detect(image)

        factor = w / float(self.cfg.detect_width)
        small = cv2.resize(image, (self.cfg.detect_width,
                                   int(round(h / factor))),
                           interpolation=cv2.INTER_AREA)

        return [d.scaled(factor) for d in self.detector.detect(small)]

    def _identity_for(self, track: Track) -> TrackIdentity:
        if track.track_id not in self._ident:
            self._ident[track.track_id] = TrackIdentity(self.cfg.identity)

        return self._ident[track.track_id]

    def _recognise(self, image: np.ndarray, seen: list[Track],
                   quality: dict[int, FaceQuality], now: float) -> None:
        if not (self.cfg.recognition_enabled and self.embedder):
            return

        gallery = self.gallery()
        budget = self.cfg.max_embeds_per_frame

        for track in seen:
            if budget <= 0:
                break

            ident = self._identity_for(track)
            det = self._last_det.get(track.track_id)
            q = quality.get(track.track_id)

            if det is None or q is None or not q.usable or not ident.due(now):
                continue

            emb = self.embedder.embed(image, det)
            budget -= 1
            self.embeds += 1

            if emb is None:
                continue

            ident.add(match(emb, gallery, self.cfg.match,
                            self.negatives())
                      if gallery else MatchResult(
                          IdentityState.UNKNOWN, None, None, None, None),
                      now)

    # ----- main entry -----

    def process(self, image: np.ndarray, now: float | None = None
                ) -> PipelineOutput:
        now = time.monotonic() if now is None else now
        self.frames += 1
        timings: dict[str, float] = {}
        detected = (now - self._last_detect_time) >= \
            self.cfg.detect_min_interval_s
        quality: dict[int, FaceQuality] = {}
        seen: list[Track] = []

        if detected:
            began = time.perf_counter()
            dets = self._detect(image)
            timings["detect"] = (time.perf_counter() - began) * 1000.0
            self._last_detect_time = now
            self.detect_passes += 1
            seen = self.tracker.update([d.box for d in dets], now,
                                       [d.score for d in dets])

            for track in seen:
                # Attach the detection that produced this track update.
                best = max(dets, key=lambda d: _iou(d.box, track.box),
                           default=None)

                if best is not None:
                    self._last_det[track.track_id] = best
                    quality[track.track_id] = assess_quality(
                        image, best, self.cfg.quality)
                    track.data["quality"] = quality[track.track_id]
        else:
            self.tracker.coast(now)

        began = time.perf_counter()
        self._recognise(image, [t for t in seen if t.state.value ==
                                "confirmed"], quality, now)
        timings["recognise"] = (time.perf_counter() - began) * 1000.0
        live = {t.track_id for t in self.tracker.tracks}

        for tid in list(self._ident):
            if tid not in live:
                del self._ident[tid]
                self._last_det.pop(tid, None)

        views = []

        for track in self.tracker.tracks:
            if track.state.value == "tentative":
                continue

            ident = self._identity_for(track)
            visible = track.state.value == "confirmed"
            box = track.box if visible else track.predicted(now)
            views.append(TrackView(
                track.track_id, track.session_id, box, visible,
                track.score, ident.state, ident.person_id, ident.score,
                track.data.get("quality"),
                self._last_det.get(track.track_id)))

        faces = sum(1 for v in views if v.visible)

        return PipelineOutput(now, tuple(views), detected, faces,
                              detected and faces == 0, timings)


def _iou(a, b) -> float:
    from buddy_core.vision.tracker import iou

    return iou(a, b)

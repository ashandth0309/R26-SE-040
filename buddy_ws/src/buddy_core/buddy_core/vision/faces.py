"""Face detection and per-face quality checks.

``YuNetDetector`` (OpenCV's YuNet, MIT) is the main detector and also
returns five landmarks.  ``HaarDetector`` is a weaker frontal-only fallback
that needs no model file.  Detection says nothing about identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    """One face in pixel coordinates of the image that was searched."""

    x: float
    y: float
    w: float
    h: float
    score: float
    # right eye, left eye, nose, right mouth corner, left mouth corner
    # ("right"/"left" are the person's, as YuNet names them).
    landmarks: tuple[tuple[float, float], ...] = ()
    raw: np.ndarray | None = field(default=None, compare=False, repr=False)

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def scaled(self, factor: float) -> "FaceDetection":
        raw = None if self.raw is None else self.raw * np.float32(factor)

        if raw is not None:
            raw[14] = self.raw[14]

        return FaceDetection(
            self.x * factor, self.y * factor, self.w * factor,
            self.h * factor, self.score,
            tuple((px * factor, py * factor) for px, py in self.landmarks),
            raw)


class FaceDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[FaceDetection]: ...


class YuNetDetector:
    """YuNet through ``cv2.FaceDetectorYN``."""

    def __init__(self, model_path: str, score_threshold: float = 0.7,
                 nms_threshold: float = 0.3, top_k: int = 20) -> None:
        self._model = str(model_path)
        self._args = (score_threshold, nms_threshold, top_k)
        self._det = cv2.FaceDetectorYN.create(
            self._model, "", (320, 320), *self._args)
        self._size = (320, 320)

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        h, w = image.shape[:2]

        if (w, h) != self._size:
            self._det.setInputSize((w, h))
            self._size = (w, h)

        _, rows = self._det.detect(image)

        if rows is None:
            return []

        out = []

        for row in rows:
            lm = tuple((float(row[4 + 2 * i]), float(row[5 + 2 * i]))
                       for i in range(5))
            out.append(FaceDetection(
                float(row[0]), float(row[1]), float(row[2]), float(row[3]),
                float(row[14]), lm, row.astype(np.float32)))

        return out


class HaarDetector:
    """OpenCV Haar cascade: frontal faces only, no landmarks."""

    def __init__(self, min_size: int = 40) -> None:
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                "this OpenCV build has no CascadeClassifier (OpenCV 5 "
                "removed it); use YuNetDetector")

        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)

        if self._cascade.empty():
            raise RuntimeError("Haar cascade could not be loaded")

        self.min_size = min_size

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        found = self._cascade.detectMultiScale(
            gray, 1.15, 5, minSize=(self.min_size, self.min_size))

        # The cascade has no confidence; 1.0 would overstate it.
        return [FaceDetection(float(x), float(y), float(w), float(h), 0.5)
                for x, y, w, h in found]


class ScriptedDetector:
    """Test double: returns pre-set detections per call."""

    def __init__(self, script: list[list[FaceDetection]]) -> None:
        self.script, self.calls = list(script), 0

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        out = self.script[min(self.calls, len(self.script) - 1)] \
            if self.script else []
        self.calls += 1

        return list(out)


# ----- quality -----

@dataclass(frozen=True)
class QualityConfig:
    min_face_px: int = 60
    min_brightness: float = 40.0
    max_brightness: float = 220.0
    min_contrast: float = 18.0
    min_sharpness: float = 25.0       # variance of Laplacian at 112 px
    max_yaw_hint: float = 0.55
    max_cut_fraction: float = 0.12    # part of the box outside the frame


@dataclass(frozen=True)
class FaceQuality:
    brightness: float
    contrast: float
    sharpness: float
    yaw_hint: float | None
    roll_deg: float | None
    cut_fraction: float
    reasons: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return not self.reasons


def yaw_and_roll(landmarks) -> tuple[float | None, float | None]:
    """Rough head turn and tilt from the five YuNet landmarks.

    ``yaw_hint`` is the nose offset from the eye midpoint in units of eye
    distance: about 0 for a frontal face, growing as the head turns.  It is
    a geometric heuristic, not a measured angle.
    """

    if len(landmarks) < 3:
        return None, None

    (rx, ry), (lx, ly), (nx, _) = landmarks[0], landmarks[1], landmarks[2]
    eye_dist = float(np.hypot(lx - rx, ly - ry))

    if eye_dist < 1e-3:
        return None, None

    mid_x = (rx + lx) / 2.0
    yaw = (nx - mid_x) / eye_dist
    roll = float(np.degrees(np.arctan2(ly - ry, lx - rx)))

    return float(yaw), roll


def assess_quality(image: np.ndarray, det: FaceDetection,
                   cfg: QualityConfig = QualityConfig()) -> FaceQuality:
    """Decide whether a face crop is good enough to use."""

    h, w = image.shape[:2]
    x1, y1, x2, y2 = det.box
    area = max(1e-6, (x2 - x1) * (y2 - y1))
    ix1, iy1 = max(0.0, x1), max(0.0, y1)
    ix2, iy2 = min(float(w), x2), min(float(h), y2)
    inside = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    cut = 1.0 - inside / area
    reasons: list[str] = []

    if min(det.w, det.h) < cfg.min_face_px:
        reasons.append("too_small")

    crop = image[int(iy1):int(iy2), int(ix1):int(ix2)]

    if crop.size == 0 or min(crop.shape[:2]) < 4:
        return FaceQuality(0.0, 0.0, 0.0, None, None, cut,
                           tuple(reasons + ["empty_crop"]))

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 \
        else crop
    small = cv2.resize(gray, (112, 112), interpolation=cv2.INTER_AREA)
    brightness = float(small.mean())
    contrast = float(small.std())
    sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
    yaw, roll = yaw_and_roll(det.landmarks)

    if brightness < cfg.min_brightness:
        reasons.append("too_dark")
    elif brightness > cfg.max_brightness:
        reasons.append("too_bright")

    if contrast < cfg.min_contrast:
        reasons.append("low_contrast")

    if sharp < cfg.min_sharpness:
        reasons.append("blurry")

    if yaw is not None and abs(yaw) > cfg.max_yaw_hint:
        reasons.append("head_turned")

    if cut > cfg.max_cut_fraction:
        reasons.append("partly_out_of_frame")

    return FaceQuality(brightness, contrast, sharp, yaw, roll, cut,
                       tuple(reasons))

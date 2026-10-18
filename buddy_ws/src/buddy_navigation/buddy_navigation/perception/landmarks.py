"""Basic visual features and landmarks (experimental).

ORB keypoints are extracted with OpenCV when it is installed.  They are a
building block for a future visual landmark map.  They are NOT used for
safety, distance sensing or localisation in this phase: a monocular camera
without calibration cannot give metric distance, and no claim of SLAM is
made.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole intrinsics; ``None`` means the camera is not calibrated."""

    fx: float
    fy: float
    cx: float
    cy: float
    reprojection_error_px: float | None = None

    def bearing_rad(self, u: float) -> float:
        """Horizontal bearing of pixel column ``u`` (left positive)."""

        import math

        return -math.atan2(u - self.cx, self.fx)


@dataclass(frozen=True)
class Landmark:
    """One visual feature."""

    u: float
    v: float
    response: float
    descriptor: bytes


class LandmarkExtractor:
    """ORB feature extraction with a clear failure mode."""

    def __init__(self, max_features: int = 200) -> None:
        self.max_features = max_features
        self.available = False
        self._orb = None

        try:
            import cv2

            self._cv2 = cv2
            self._orb = cv2.ORB_create(nfeatures=max_features)
            self.available = True
        except Exception:
            self.available = False

    def extract(self, bgr: np.ndarray) -> list[Landmark]:
        """Return strongest features, or [] if OpenCV is unavailable."""

        if not self.available or bgr.size == 0:
            return []

        gray = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)

        if descriptors is None:
            return []

        found = [
            Landmark(kp.pt[0], kp.pt[1], kp.response, bytes(desc))
            for kp, desc in zip(keypoints, descriptors)
        ]

        return sorted(found, key=lambda lm: -lm.response)[
            : self.max_features
        ]

    def match_count(self, a: list[Landmark], b: list[Landmark]) -> int:
        """Number of cross-checked descriptor matches between two sets."""

        if not self.available or not a or not b:
            return 0

        da = np.array([np.frombuffer(m.descriptor, np.uint8) for m in a])
        db = np.array([np.frombuffer(m.descriptor, np.uint8) for m in b])
        matcher = self._cv2.BFMatcher(self._cv2.NORM_HAMMING, crossCheck=True)

        return len(matcher.match(da, db))

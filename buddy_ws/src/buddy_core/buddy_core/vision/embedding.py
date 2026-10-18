"""Face embeddings (SFace, Apache-2.0) and cosine helpers."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from buddy_core.vision.faces import FaceDetection


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))

    return v / n if n > 0 else v


class FaceEmbedder(Protocol):
    dim: int

    def embed(self, image: np.ndarray, det: FaceDetection
              ) -> np.ndarray | None: ...


class SFaceEmbedder:
    """128-d embeddings through ``cv2.FaceRecognizerSF``.

    Needs a detection that carries YuNet's raw row so the face can be
    aligned from its landmarks.  Returns None when alignment is impossible
    (for example a Haar detection, which has no landmarks).
    """

    dim = 128

    def __init__(self, model_path: str) -> None:
        self._rec = cv2.FaceRecognizerSF.create(str(model_path), "")

    def embed(self, image: np.ndarray, det: FaceDetection
              ) -> np.ndarray | None:
        if det.raw is None:
            return None

        try:
            aligned = self._rec.alignCrop(image, det.raw)
            feat = self._rec.feature(aligned)
        except cv2.error:
            return None

        return l2_normalize(feat)


class HashEmbedder:
    """Test double: deterministic embedding from a label drawn on the
    detection (``det.raw`` is not used).  Faces with the same label match."""

    dim = 16

    def __init__(self, labels: dict[int, str] | None = None) -> None:
        self.labels = labels or {}
        self.calls = 0

    def embed(self, image, det):
        self.calls += 1
        label = self.labels.get(int(round(det.x)), f"anon{int(det.x)}")
        rng = np.random.default_rng(abs(hash(label)) % (2 ** 32))

        return l2_normalize(rng.standard_normal(self.dim))

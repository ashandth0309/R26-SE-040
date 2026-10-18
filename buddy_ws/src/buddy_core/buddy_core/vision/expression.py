"""Facial *expression* classification (not emotion detection).

The model reports which of seven visible expression categories a face most
resembles.  A smile is not proof of happiness and a neutral face is not
proof that someone is fine; downstream code must treat these as weak,
short-lived hints and let the person's own words override them.

Model: MobileFaceNet "progressive teacher" from the OpenCV model zoo
(Apache-2.0).  The upstream project reports 88.27% accuracy on RAF-DB, a
dataset of posed and in-the-wild web photos.  That number says nothing
about this robot's camera, rooms, or users; see ``expression_eval``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from buddy_core.vision.faces import FaceDetection
from buddy_core.vision.onnx_runtime import OnnxSession

LABELS = ("angry", "disgust", "fearful", "happy", "neutral", "sad",
          "surprised")

GROUPS = {
    "happy": "positive",
    "neutral": "neutral",
    "surprised": "surprise",
    "angry": "negative",
    "disgust": "negative",
    "fearful": "negative",
    "sad": "negative",
}
GROUP_NAMES = ("positive", "neutral", "negative", "surprise")

# Template for the 112x112 aligned face used by the model.
_STD_POINTS = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                        [56.0252, 71.7366], [41.5493, 92.3655],
                        [70.7299, 92.2041]], dtype=np.float64)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares rotation+scale+translation mapping src to dst (2x3)."""

    n = src.shape[0]
    a = np.zeros((2 * n, 4))
    a[:n, 0], a[:n, 1], a[:n, 2] = src[:, 0], -src[:, 1], 1.0
    a[n:, 0], a[n:, 1], a[n:, 3] = src[:, 1], src[:, 0], 1.0
    b = np.concatenate([dst[:, 0], dst[:, 1]])
    p, *_ = np.linalg.lstsq(a, b, rcond=None)
    s, r, tx, ty = p

    return np.array([[s, -r, tx], [r, s, ty]])


def align_face(image: np.ndarray, landmarks) -> np.ndarray:
    """112x112 aligned crop from the five YuNet landmarks."""

    src = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    m = similarity_transform(src, _STD_POINTS)

    return cv2.warpAffine(image, m, (112, 112))


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / max(1e-6, temperature)
    z -= z.max()
    e = np.exp(z)

    return e / e.sum()


@dataclass(frozen=True)
class ExpressionScores:
    probs: tuple[float, ...]            # aligned with LABELS
    top: str
    top_prob: float
    margin: float                       # top minus runner-up
    entropy: float                      # 0 certain .. 1 uniform

    def group_probs(self) -> dict[str, float]:
        out = dict.fromkeys(GROUP_NAMES, 0.0)

        for label, p in zip(LABELS, self.probs, strict=True):
            out[GROUPS[label]] += p

        return out


def scores_from_probs(probs: np.ndarray) -> ExpressionScores:
    p = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-p)
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))

    return ExpressionScores(tuple(float(x) for x in p), LABELS[order[0]],
                            float(p[order[0]]),
                            float(p[order[0]] - p[order[1]]), ent)


class ExpressionClassifier:
    """Runs the ONNX model on an aligned face."""

    def __init__(self, model_path: str, temperature: float = 1.0,
                 threads: int = 2) -> None:
        self._sess = OnnxSession(model_path, threads)
        self.temperature = float(temperature)

    def logits(self, image: np.ndarray, det: FaceDetection
               ) -> np.ndarray | None:
        if len(det.landmarks) != 5:
            return None

        face = align_face(image, det.landmarks)

        return self.logits_aligned(face)

    def logits_aligned(self, face_bgr_112: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_bgr_112, cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
        blob = np.transpose(x, (2, 0, 1))[None]

        return np.asarray(self._sess.run(blob)[0]).reshape(-1)

    def predict(self, image: np.ndarray, det: FaceDetection
                ) -> ExpressionScores | None:
        lg = self.logits(image, det)

        if lg is None:
            return None

        return scores_from_probs(softmax(lg, self.temperature))

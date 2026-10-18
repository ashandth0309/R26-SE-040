"""Body pose (33 MediaPipe BlazePose landmarks) for one person at a time.

Pipeline: a person box (from the COCO detector) -> square crop -> rotate by
0, 90 or 270 degrees if needed -> pose network.  Trying the sideways
rotations is what lets a *lying* person be read at all; the network is
trained on upright bodies.  The extra passes run only when the upright
pass is unconvincing or the box is wide, so the usual cost is one pass.

Output coordinates are frame pixels.  Nothing here is a diagnosis: it is a
skeleton estimate that can be wrong, especially when the legs are hidden
behind furniture, in low light, or when two people overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from buddy_core.vision.models import model_path
from buddy_core.vision.onnx_runtime import OnnxSession

NOSE, L_EYE, R_EYE = 0, 2, 5
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
N_KEYPOINTS = 33


@dataclass(frozen=True)
class Pose:
    xy: np.ndarray            # (33, 2) frame pixels
    visibility: np.ndarray    # (33,) 0..1
    confidence: float         # network's pose-present score
    rotation: int             # 0 upright, 1 = 90 deg, 3 = 270 deg crop
    box: tuple[float, float, float, float]   # tight box over visible points

    def point(self, idx: int, min_vis: float = 0.4):
        return self.xy[idx] if self.visibility[idx] >= min_vis else None


def _rotate_k(img: np.ndarray, k: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(img, k % 4))


def _unrotate_points(p: np.ndarray, k: int, size: float) -> np.ndarray:
    """Undo np.rot90(img, k) for points in a square crop of ``size``."""

    q = p.copy()

    for _ in range(k % 4):
        # one np.rot90 step maps (x, y) -> (y, size - x); invert it
        q = np.column_stack([size - q[:, 1], q[:, 0]])

    return q


class PoseEstimator:
    SIZE = 256

    def __init__(self, path: str | None = None, threads: int = 2,
                 min_confidence: float = 0.5, enlarge: float = 1.25
                 ) -> None:
        self.session = OnnxSession(path or model_path("pose"), threads)
        self.min_confidence = float(min_confidence)
        self.enlarge = float(enlarge)
        self.last_ms = 0.0
        self.passes = 0

    def _run(self, crop_bgr: np.ndarray):
        rgb = cv2.cvtColor(cv2.resize(crop_bgr, (self.SIZE, self.SIZE),
                                      interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0)[None]
        outs = self.session.run(x)
        lm = next(o for o in outs if o.size == 195).reshape(39, 5)
        conf = float(next(o for o in outs if o.size == 1).reshape(-1)[0])
        vis = 1.0 / (1.0 + np.exp(-lm[:33, 3]))

        return lm[:33, :2].copy(), vis, conf

    def estimate(self, bgr: np.ndarray, person_box,
                 try_sideways: bool | None = None) -> Pose | None:
        import time
        t0 = time.perf_counter()
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = person_box
        side = max(x2 - x1, y2 - y1) * self.enlarge
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
        s = max(8, int(round(side)))
        pad = np.zeros((s, s, 3), np.uint8)
        sx1, sy1 = max(0, left), max(0, top)
        sx2, sy2 = min(w, left + s), min(h, top + s)

        if sx2 <= sx1 or sy2 <= sy1:
            return None

        pad[sy1 - top:sy2 - top, sx1 - left:sx2 - left] = \
            bgr[sy1:sy2, sx1:sx2]
        wide = (x2 - x1) > 1.2 * (y2 - y1)
        best = None
        self.passes = 0
        ks = [0]

        for round_ in range(2):
            for k in ks:
                self.passes += 1
                xy, vis, conf = self._run(_rotate_k(pad, k))

                if best is None or conf > best[0]:
                    best = (conf, k, xy, vis)

            if round_ == 0:
                need = try_sideways if try_sideways is not None else (
                    wide or best[0] < self.min_confidence + 0.2)

                if not need:
                    break

                ks = [1, 3]

        conf, k, xy, vis = best
        self.last_ms = (time.perf_counter() - t0) * 1000.0

        if conf < self.min_confidence:
            return None

        pts = xy * (s / self.SIZE)
        pts = _unrotate_points(pts, k, float(s))
        pts += np.array([left, top], np.float32)
        ok = vis >= 0.3
        bx = pts[ok] if ok.sum() >= 3 else pts

        return Pose(pts.astype(np.float32), vis.astype(np.float32), conf, k,
                    (float(bx[:, 0].min()), float(bx[:, 1].min()),
                     float(bx[:, 0].max()), float(bx[:, 1].max())))

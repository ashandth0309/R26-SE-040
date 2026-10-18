"""Everyday-object detection with NanoDet-Plus (COCO, 80 classes).

What this is and is not:

* It labels objects that belong to the 80 COCO classes.  It does not know
  *your* mug versus another mug (see ``object_teach`` for that) and it
  does not know any class outside the list.
* Scores are detector scores, not calibrated probabilities.  Accuracy on a
  living-room Pi camera is unknown until someone measures it with
  ``object_eval``.
* People are removed from the object list on purpose; people go through
  the face and person pipelines with their own consent rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from buddy_core.vision.models import model_path
from buddy_core.vision.onnx_runtime import OnnxSession

COCO = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush')

# Classes that matter in a home, grouped for "what is on the table" style
# answers.  Anything else is still reported, as "other".
CATEGORIES = {
    "drink": {"bottle", "cup", "wine glass"},
    "food": {"banana", "apple", "sandwich", "orange", "broccoli", "carrot",
             "hot dog", "pizza", "donut", "cake", "bowl"},
    "cutlery": {"fork", "knife", "spoon"},
    "furniture": {"chair", "couch", "bed", "dining table", "toilet",
                  "potted plant", "bench"},
    "electronics": {"tv", "laptop", "mouse", "remote", "keyboard",
                    "cell phone", "microwave", "oven", "toaster",
                    "refrigerator", "clock"},
    "personal": {"backpack", "handbag", "umbrella", "suitcase", "book",
                 "tie", "toothbrush", "hair drier", "scissors"},
    "pet": {"cat", "dog", "bird"},
    "toy": {"teddy bear", "sports ball", "frisbee", "kite"},
    "household": {"vase", "sink"},
}
HAZARDS = {"knife", "scissors", "oven", "toaster"}   # only informational


def category_of(name: str) -> str:
    for cat, names in CATEGORIES.items():
        if name in names:
            return cat

    return "other"


@dataclass(frozen=True)
class ObjectDetection:
    label: str
    score: float
    box: tuple[float, float, float, float]      # x1, y1, x2, y2 in pixels
    category: str = "other"

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box

        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def letterbox(img: np.ndarray, size: int = 416):
    """Resize keeping aspect ratio, pad with zeros (centre)."""

    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    out = np.zeros((size, size, 3), np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_AREA)

    return out, (top, left, nh, nw)


class NanoDetDetector:
    """NanoDet-Plus 416x416 via onnxruntime (OpenCV DNN fallback)."""

    STRIDES = (8, 16, 32)       # this ONNX has three heads, not four
    REG_MAX = 7
    SIZE = 416
    MEAN = np.array([103.53, 116.28, 123.675], np.float32)
    STD = np.array([57.375, 57.12, 58.395], np.float32)

    def __init__(self, path: str | None = None, score_threshold=0.40,
                 nms_iou=0.5, threads: int = 2,
                 exclude_person: bool = True) -> None:
        self.session = OnnxSession(path or model_path("object_detector"),
                                   threads)
        self.score_threshold = float(score_threshold)
        self.nms_iou = float(nms_iou)
        self.exclude_person = exclude_person
        self.last_ms = 0.0
        self._anchors = []

        for s in self.STRIDES:
            n = self.SIZE // s
            xs, ys = np.meshgrid(np.arange(n) * s, np.arange(n) * s)
            self._anchors.append(np.column_stack(
                [xs.ravel() + 0.5 * (s - 1), ys.ravel() + 0.5 * (s - 1)]))

    def detect(self, bgr: np.ndarray) -> list[ObjectDetection]:
        import time
        t0 = time.perf_counter()
        img, (top, left, nh, nw) = letterbox(bgr, self.SIZE)
        x = (img.astype(np.float32) - self.MEAN) / self.STD
        blob = np.transpose(x, (2, 0, 1))[None].astype(np.float32)
        outs = self.session.run(blob)
        dets = self._post(outs)
        h, w = bgr.shape[:2]
        sx, sy = w / nw, h / nh
        res = []

        for x1, y1, x2, y2, sc, cid in dets:
            label = COCO[int(cid)]

            if self.exclude_person and label == "person":
                continue

            box = (float(np.clip((x1 - left) * sx, 0, w)),
                   float(np.clip((y1 - top) * sy, 0, h)),
                   float(np.clip((x2 - left) * sx, 0, w)),
                   float(np.clip((y2 - top) * sy, 0, h)))
            res.append(ObjectDetection(label, float(sc), box,
                                       category_of(label)))

        self.last_ms = (time.perf_counter() - t0) * 1000.0

        return res

    def _post(self, outs):
        # Pair heads by shape: class maps have 80 channels, boxes 4*(reg+1).
        cls = sorted((o for o in outs if o.shape[-1] == len(COCO)),
                     key=lambda o: -o.shape[-2])
        reg = sorted((o for o in outs
                      if o.shape[-1] == 4 * (self.REG_MAX + 1)),
                     key=lambda o: -o.shape[-2])

        if len(cls) != len(self.STRIDES) or len(reg) != len(self.STRIDES):
            raise RuntimeError(
                "unexpected NanoDet outputs: "
                + str([o.shape for o in outs]))

        boxes, scores = [], []

        for stride, c, r, anchors in zip(self.STRIDES, cls, reg,
                                         self._anchors, strict=True):
            c = np.squeeze(c, 0) if c.ndim == 3 else c
            r = np.squeeze(r, 0) if r.ndim == 3 else r
            e = np.exp(r.reshape(-1, self.REG_MAX + 1)
                       - r.reshape(-1, self.REG_MAX + 1).max(1,
                                                             keepdims=True))
            p = e / e.sum(1, keepdims=True)
            d = (p @ np.arange(self.REG_MAX + 1)).reshape(-1, 4) * stride
            boxes.append(np.column_stack([
                np.clip(anchors[:, 0] - d[:, 0], 0, self.SIZE),
                np.clip(anchors[:, 1] - d[:, 1], 0, self.SIZE),
                np.clip(anchors[:, 0] + d[:, 2], 0, self.SIZE),
                np.clip(anchors[:, 1] + d[:, 3], 0, self.SIZE)]))
            scores.append(c)

        b = np.concatenate(boxes)
        s = np.concatenate(scores)
        cid = s.argmax(1)
        conf = s.max(1)
        keep = conf >= self.score_threshold

        if not keep.any():
            return []

        b, conf, cid = b[keep], conf[keep], cid[keep]
        wh = b.copy()
        wh[:, 2:] -= wh[:, :2]
        idx = cv2.dnn.NMSBoxes(wh.tolist(), conf.tolist(),
                               self.score_threshold, self.nms_iou)
        idx = np.array(idx).reshape(-1)

        return [(*b[i], conf[i], cid[i]) for i in idx]


class ScriptedObjectDetector:
    """Test double returning pre-set detections per call."""

    def __init__(self, script) -> None:
        self.script = list(script)
        self.i = 0
        self.last_ms = 0.0

    def detect(self, bgr) -> list[ObjectDetection]:
        out = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1

        return list(out)

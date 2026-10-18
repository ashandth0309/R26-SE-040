"""Benchmark the object detector on a labelled folder you supply.

Layout::

    root/images/<name>.jpg
    root/labels.json   {"<name>.jpg": [{"label": "cup",
                                        "box": [x1, y1, x2, y2]}, ...]}

For each class: true positives (right label, IoU >= 0.5), false
positives, false negatives, precision and recall.  Detections that
overlap a ground-truth box but carry another label are counted as
confusions.  Nothing here is a result until you run it on your own
images from the Pi camera; the sandbox has no such set.

    python -m buddy_core.vision.object_eval --root DIR --out report.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2

from buddy_core.vision.tracker import iou


def score_image(dets, truth, iou_thr=0.5):
    """Greedy one-to-one matching.  Returns tp, fp, fn, confusions."""

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    conf = defaultdict(int)
    used = set()

    for d in sorted(dets, key=lambda d: -d.score):
        best, best_i = None, 0.0

        for i, t in enumerate(truth):
            if i in used:
                continue

            v = iou(d.box, tuple(t["box"]))

            if v > best_i:
                best, best_i = i, v

        if best is not None and best_i >= iou_thr:
            used.add(best)
            t = truth[best]

            if t["label"] == d.label:
                tp[d.label] += 1
            else:
                fp[d.label] += 1
                fn[t["label"]] += 1
                conf[(t["label"], d.label)] += 1
        else:
            fp[d.label] += 1

    for i, t in enumerate(truth):
        if i not in used:
            fn[t["label"]] += 1

    return tp, fp, fn, conf


def summarise(tp, fp, fn, conf):
    labels = sorted(set(tp) | set(fp) | set(fn))
    per = {}

    for k in labels:
        p = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else None
        r = tp[k] / (tp[k] + fn[k]) if tp[k] + fn[k] else None
        per[k] = {"tp": tp[k], "fp": fp[k], "fn": fn[k],
                  "precision": p, "recall": r}

    return {"per_class": per,
            "confusions": {f"{a}->{b}": n for (a, b), n in conf.items()}}


def evaluate(detector, root: str, iou_thr: float = 0.5) -> dict:
    root_p = Path(root)
    labels = json.loads((root_p / "labels.json").read_text())
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    conf = defaultdict(int)
    times = []
    n = 0

    for name, truth in labels.items():
        img = cv2.imread(str(root_p / "images" / name))

        if img is None:
            continue

        t0 = time.perf_counter()
        dets = detector.detect(img)
        times.append((time.perf_counter() - t0) * 1000.0)
        a, b, c, d = score_image(dets, truth, iou_thr)

        for src, dst in ((a, tp), (b, fp), (c, fn), (d, conf)):
            for k, v in src.items():
                dst[k] += v

        n += 1

    out = summarise(tp, fp, fn, conf)
    out["images"] = n
    out["iou_threshold"] = iou_thr

    if times:
        s = sorted(times)
        out["latency_ms"] = {"mean": sum(s) / len(s),
                             "p95": s[int(0.95 * (len(s) - 1))],
                             "note": "this machine only; not the Pi"}

    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out")
    ap.add_argument("--score", type=float, default=0.4)
    a = ap.parse_args(argv)
    from buddy_core.vision.objects import NanoDetDetector
    rep = evaluate(NanoDetDetector(score_threshold=a.score), a.root)
    text = json.dumps(rep, indent=2)

    if a.out:
        Path(a.out).write_text(text)

    print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

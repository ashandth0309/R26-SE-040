"""Measure expression models on a labelled folder of face images.

    python3 -m buddy_core.vision.expression_eval zoo --root DIR [--out F]
    python3 -m buddy_core.vision.expression_eval keras --root DIR --model M

``DIR`` holds one sub-folder per class (angry, disgust, fear, happy,
neutral, sad, surprise).  Reported: confusion matrix, per-class precision
and recall, macro-F1, group-level results, calibration (ECE, and a
temperature fitted on one half and scored on the other), how often a
confident wrong "negative" is produced for neutral faces, and the share of
images where no face could be found.

What the numbers mean: they describe THIS dataset only.  FER-style sets are
48x48 grayscale web images with noisy labels; they say little about a Pi
camera in a living room.  Cross-dataset results are expected to be lower.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from buddy_core.vision.evaluation import confusion_matrix, per_class_metrics
from buddy_core.vision.expression import GROUPS, LABELS, softmax

# FER folder names -> the model's label names
FER_TO_MODEL = {"fear": "fearful", "surprise": "surprised"}
MODEL_TO_FER = {v: k for k, v in FER_TO_MODEL.items()}
FER_CLASSES = tuple(MODEL_TO_FER.get(label, label) for label in LABELS)


def list_images(root: Path, per_class: int | None = None,
                seed: int = 0) -> list[tuple[Path, str]]:
    rng = random.Random(seed)
    out = []

    for cls in FER_CLASSES:
        files = sorted(p for p in (root / cls).glob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        rng.shuffle(files)
        out += [(p, cls) for p in files[:per_class]]

    return out


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               bins: int = 10) -> float:
    """Gap between stated confidence and actual accuracy, averaged."""

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (conf > lo) & (conf <= hi)

        if sel.any():
            ece += sel.mean() * abs(
                float((pred[sel] == labels[sel]).mean()) -
                float(conf[sel].mean()))

    return float(ece)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature minimising negative log-likelihood (grid search)."""

    best_t, best_nll = 1.0, np.inf

    for t in np.arange(0.5, 5.01, 0.05):
        p = np.array([softmax(row, t) for row in logits])
        nll = -float(np.mean(np.log(p[np.arange(len(labels)), labels]
                                    + 1e-12)))

        if nll < best_nll:
            best_t, best_nll = float(t), nll

    return round(best_t, 2)


def summarise(y_true: list[str], y_pred: list[str],
              no_face: dict[str, int] | None = None) -> dict:
    """Metrics from true and predicted FER class names."""

    labels = list(FER_CLASSES)
    mat = confusion_matrix(y_true, y_pred, labels)
    per = per_class_metrics(mat, labels)
    total = int(mat.sum())
    acc = float(np.trace(mat) / total) if total else None
    f1s = [v["f1"] for v in per.values() if v["f1"] is not None]
    group_of = {FER_TO_MODEL.get(c, c): GROUPS[FER_TO_MODEL.get(c, c)]
                for c in labels}
    glabels = ["positive", "neutral", "negative", "surprise"]
    gmat = confusion_matrix([group_of[FER_TO_MODEL.get(t, t)]
                             for t in y_true],
                            [group_of[FER_TO_MODEL.get(p, p)]
                             for p in y_pred], glabels)
    neutral_total = int(mat[labels.index("neutral")].sum())
    neutral_neg = int(gmat[glabels.index("neutral"),
                           glabels.index("negative")])

    return {
        "labels": labels, "confusion": mat.tolist(),
        "per_class": per, "accuracy": acc,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "group_labels": glabels, "group_confusion": gmat.tolist(),
        "group_per_class": per_class_metrics(gmat, glabels),
        "neutral_faces_called_negative": {
            "count": neutral_neg, "of": neutral_total,
            "rate": neutral_neg / neutral_total if neutral_total
            else None},
        "images_scored": total,
        "no_face_found": no_face or {},
    }


def calibration_report(logits: np.ndarray, y: np.ndarray) -> dict:
    """Fit temperature on a random half, score both halves."""

    n = len(y)

    if n < 40:
        return {"warning": f"only {n} scored images; calibration skipped"}

    idx = np.random.default_rng(0).permutation(n)
    fit, held = idx[: n // 2], idx[n // 2:]
    t = fit_temperature(logits[fit], y[fit])

    def ece(sel, temp):
        p = np.array([softmax(r, temp) for r in logits[sel]])

        return expected_calibration_error(p, y[sel])

    p_all = np.array([softmax(r, 1.0) for r in logits])
    conf = p_all.max(axis=1)
    correct = p_all.argmax(axis=1) == y
    thresholds = {}

    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        sel = conf >= thr
        thresholds[str(thr)] = {
            "share_of_images": float(sel.mean()),
            "accuracy_when_above": float(correct[sel].mean())
            if sel.any() else None}

    return {"temperature_fit_on_half": t,
            "ece_held_out_before": ece(held, 1.0),
            "ece_held_out_after": ece(held, t),
            "confidence_vs_accuracy": thresholds}


# ----- runners -----

def run_zoo(root: Path, per_class: int | None, detector_threshold: float
            ) -> dict:
    import cv2

    from buddy_core.vision import models
    from buddy_core.vision.expression import ExpressionClassifier
    from buddy_core.vision.faces import YuNetDetector

    det = YuNetDetector(str(models.model_path("face_detector")),
                        score_threshold=detector_threshold)
    clf = ExpressionClassifier(str(models.model_path("expression")))
    y_true, y_pred, logits, ys = [], [], [], []
    no_face: dict[str, int] = {}

    for path, cls in list_images(root, per_class):
        img = cv2.imread(str(path))

        if img is None:
            continue

        big = cv2.resize(img, (192, 192), interpolation=cv2.INTER_CUBIC)
        pad = cv2.copyMakeBorder(big, 64, 64, 64, 64, cv2.BORDER_REPLICATE)
        faces = det.detect(pad)

        if not faces:
            no_face[cls] = no_face.get(cls, 0) + 1

            continue

        lg = clf.logits(pad, max(faces, key=lambda f: f.score))
        pred = MODEL_TO_FER.get(LABELS[int(np.argmax(lg))],
                                LABELS[int(np.argmax(lg))])
        y_true.append(cls)
        y_pred.append(pred)
        logits.append(lg)
        ys.append(FER_CLASSES.index(cls))

    report = summarise(y_true, y_pred, no_face)
    report["calibration"] = calibration_report(np.array(logits),
                                               np.array(ys))
    report["setup"] = {
        "model": "opencv_zoo facial_expression_recognition_mobilefacenet_"
                 "2022july", "detector": "YuNet 2023mar",
        "preprocessing": "48x48 gray -> 192x192 bicubic, replicate-pad "
                         "64px, YuNet landmarks, 112x112 alignment",
        "detector_threshold": detector_threshold}

    return report


def run_keras(root: Path, per_class: int | None, model_path: str,
              rescale: float) -> dict:
    import cv2
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    y_true, y_pred, logits, ys = [], [], [], []

    for path, cls in list_images(root, per_class):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        x = cv2.resize(img, (48, 48)).astype("float32")[None, :, :, None]
        out = model.predict(x * rescale, verbose=0)[0]
        y_true.append(cls)
        y_pred.append(FER_CLASSES[int(np.argmax(out))])
        logits.append(np.log(np.clip(out, 1e-9, 1.0)))
        ys.append(FER_CLASSES.index(cls))

    report = summarise(y_true, y_pred)
    report["calibration"] = calibration_report(np.array(logits),
                                               np.array(ys))
    report["setup"] = {
        "model": Path(model_path).name,
        "preprocessing": f"48x48 grayscale, x{rescale}",
        "class_order_assumed": list(FER_CLASSES),
        "note": "class order assumed alphabetical (Keras directory "
                "order); the repo does not record it for this file"}

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["zoo", "keras"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--model", default="")
    parser.add_argument("--rescale", type=float, default=1 / 255)
    parser.add_argument("--detector-threshold", type=float, default=0.5)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    root = Path(args.root)

    if args.kind == "zoo":
        report = run_zoo(root, args.per_class, args.detector_threshold)
    else:
        report = run_keras(root, args.per_class, args.model, args.rescale)

    text = json.dumps(report, indent=2)

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")

    print(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())

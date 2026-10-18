"""Evaluation maths and a command-line identity evaluation.

Everything here reports only what the supplied data shows, and says so when
the data is too small to trust.  Results on synthetic or augmented images
must be labelled as such by the caller; they do not describe real use.

Identity evaluation on consented captures:

    python3 -m buddy_core.vision.evaluation identity --root DIR

where DIR holds one sub-folder of face photos per consenting person.  Each
photo is compared with every other person's photos (impostors) and with the
same person's other photos (genuine).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

MIN_TRIALS = 30


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str],
                     labels: Sequence[str]) -> np.ndarray:
    """Rows are the true class, columns the predicted class."""

    index = {label: i for i, label in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)

    for t, p in zip(y_true, y_pred, strict=True):
        if t in index and p in index:
            mat[index[t], index[p]] += 1

    return mat


def per_class_metrics(mat: np.ndarray, labels: Sequence[str]) -> dict:
    """Precision, recall, F1 and support per class (None when undefined)."""

    out = {}

    for i, label in enumerate(labels):
        tp = int(mat[i, i])
        support = int(mat[i].sum())
        predicted = int(mat[:, i].sum())
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else
              (0.0 if precision is not None and recall is not None
               else None))
        out[label] = {"precision": precision, "recall": recall, "f1": f1,
                      "support": support}

    return out


def verification_metrics(genuine: Sequence[float],
                         impostor: Sequence[float],
                         thresholds: Sequence[float] | None = None) -> dict:
    """False accept / false reject rates at each threshold, and the
    approximate equal-error rate (None if either list is empty)."""

    g = np.asarray(genuine, dtype=float)
    i = np.asarray(impostor, dtype=float)
    warnings = []

    if len(g) < MIN_TRIALS:
        warnings.append(f"only {len(g)} genuine trials (< {MIN_TRIALS})")

    if len(i) < MIN_TRIALS:
        warnings.append(f"only {len(i)} impostor trials (< {MIN_TRIALS})")

    thresholds = list(thresholds) if thresholds is not None \
        else [round(t, 2) for t in np.arange(0.20, 0.81, 0.05)]
    rows = []

    for t in thresholds:
        far = float((i >= t).mean()) if len(i) else None
        frr = float((g < t).mean()) if len(g) else None
        rows.append({"threshold": float(t), "far": far, "frr": frr})

    eer = None

    if len(g) and len(i):
        best = min(rows, key=lambda r: abs(r["far"] - r["frr"]))
        eer = (best["far"] + best["frr"]) / 2.0

    return {"genuine": int(len(g)), "impostor": int(len(i)),
            "rows": rows, "eer_approx": eer, "warnings": warnings}


def identification_summary(trials: Sequence[tuple[str | None, str | None]]
                           ) -> dict:
    """``trials`` are (true person or None for a stranger, decided person
    or None for unknown).  Counts the outcomes that matter for safety."""

    c = {"correct_named": 0, "wrong_person_named": 0,
         "stranger_named": 0, "enrolled_not_recognised": 0,
         "stranger_rejected": 0}

    for truth, said in trials:
        if truth is None and said is None:
            c["stranger_rejected"] += 1
        elif truth is None:
            c["stranger_named"] += 1
        elif said is None:
            c["enrolled_not_recognised"] += 1
        elif said == truth:
            c["correct_named"] += 1
        else:
            c["wrong_person_named"] += 1

    strangers = c["stranger_named"] + c["stranger_rejected"]
    enrolled = sum(c[k] for k in ("correct_named", "wrong_person_named",
                                  "enrolled_not_recognised"))
    c["false_name_rate_strangers"] = \
        c["stranger_named"] / strangers if strangers else None
    c["wrong_name_rate_enrolled"] = \
        c["wrong_person_named"] / enrolled if enrolled else None
    c["recognition_rate_enrolled"] = \
        c["correct_named"] / enrolled if enrolled else None
    c["warnings"] = [f"only {n} {w} trials (< {MIN_TRIALS})"
                     for w, n in (("stranger", strangers),
                                  ("enrolled", enrolled))
                     if n < MIN_TRIALS]

    return c


# ----- command line -----

def _image_paths(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def identity_from_folder(root: Path) -> dict:
    """Embed every photo under ``root/<person>/`` and report genuine and
    impostor similarity.  Needs the YuNet and SFace models."""

    import cv2

    from buddy_core.vision import models
    from buddy_core.vision.embedding import SFaceEmbedder
    from buddy_core.vision.faces import YuNetDetector

    det = YuNetDetector(str(models.model_path("face_detector")))
    emb = SFaceEmbedder(str(models.model_path("face_embedder")))
    vecs: dict[str, list[np.ndarray]] = {}
    skipped = 0

    for person in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in _image_paths(person):
            img = cv2.imread(str(path))
            faces = det.detect(img) if img is not None else []

            if not faces:
                skipped += 1

                continue

            best = max(faces, key=lambda f: f.w * f.h)
            v = emb.embed(img, best)

            if v is None:
                skipped += 1
            else:
                vecs.setdefault(person.name, []).append(v)

    genuine, impostor = [], []
    people = list(vecs)

    for a, pa in enumerate(people):
        for i in range(len(vecs[pa])):
            for j in range(i + 1, len(vecs[pa])):
                genuine.append(float(vecs[pa][i] @ vecs[pa][j]))

        for pb in people[a + 1:]:
            for x in vecs[pa]:
                for y in vecs[pb]:
                    impostor.append(float(x @ y))

    result = verification_metrics(genuine, impostor)
    result["people"] = {p: len(v) for p, v in vecs.items()}
    result["images_without_usable_face"] = skipped
    result["note"] = ("Cosine similarity between SFace embeddings. Valid "
                      "only for the people, cameras and lighting in this "
                      "folder.")

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    ident = sub.add_parser("identity")
    ident.add_argument("--root", required=True)
    ident.add_argument("--out", default="")
    args = parser.parse_args(argv)

    if args.cmd == "identity":
        report = identity_from_folder(Path(args.root))
        text = json.dumps(report, indent=2)
        print(text)

        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())

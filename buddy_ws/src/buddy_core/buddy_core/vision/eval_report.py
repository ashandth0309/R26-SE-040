"""Builds docs/evaluation/test_matrix.md from the matrix and result JSONs.

    python -m buddy_core.vision.eval_report \
        --out docs/evaluation/test_matrix.md

Only numbers found in the JSON result files are printed.  Nothing here
is a real-camera result; the report says so at the top.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
EVAL = REPO / "docs" / "evaluation"

RESULTS = (
    ("Zoo MobileFaceNet (the model Buddy uses)",
     "expression_zoo_fer2013_test.json"),
    ("Team emotion_model.h5 (not used by Buddy; for comparison)",
     "expression_team_h5_fer2013_test.json"),
)


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _num(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def confusion_table(labels: list[str], mat: list[list[int]]) -> str:
    head = "| true \\ predicted | " + " | ".join(labels) + " |"
    sep = "|---|" + "---|" * len(labels)
    rows = [f"| **{lab}** | " + " | ".join(str(v) for v in row) + " |"
            for lab, row in zip(labels, mat)]

    return "\n".join([head, sep, *rows])


def per_class_table(per: dict) -> str:
    rows = ["| class | precision | recall | F1 | support |",
            "|---|---|---|---|---|"]

    for name, m in per.items():
        rows.append(f"| {name} | {_num(m['precision'])} | "
                    f"{_num(m['recall'])} | {_num(m['f1'])} | "
                    f"{m['support']} |")

    return "\n".join(rows)


def expression_section(title: str, data: dict) -> str:
    cal = data.get("calibration", {})
    neu = data.get("neutral_faces_called_negative", {})
    out = [f"### {title}", "",
           f"Images scored: {data['images_scored']}. "
           f"Accuracy {_pct(data['accuracy'])}, "
           f"macro-F1 {_num(data['macro_f1'])}. "
           f"Faces not found: {sum(data['no_face_found'].values())}.", "",
           "Seven classes:", "",
           confusion_table(data["labels"], data["confusion"]), "",
           per_class_table(data["per_class"]), "",
           "Coarse groups (what Buddy actually uses):", "",
           confusion_table(data["group_labels"], data["group_confusion"]),
           "", per_class_table(data["group_per_class"]), ""]

    if neu:
        out.append(f"Neutral faces called negative: {neu['count']} of "
                   f"{neu['of']} ({_pct(neu['rate'])}).")
        out.append("")

    if cal:
        out.append(
            f"Calibration: expected calibration error "
            f"{_num(cal.get('ece_held_out_before'))} before and "
            f"{_num(cal.get('ece_held_out_after'))} after temperature "
            f"scaling (T={cal.get('temperature_fit_on_half')}, fitted on "
            f"half the data, evaluated on the other half).")
        out.append("")

    return "\n".join(out)


def matrix_section(matrix: dict) -> str:
    rows = ["| # | Area | Evidence | Tests | Not verified |",
            "|---|---|---|---|---|"]

    for a in matrix["areas"]:
        rows.append(f"| {a['id']} | {a['name']} | `{a['evidence']}` | "
                    f"{len(a['tests'])} | {a['not_verified']} |")

    return "\n".join(rows)


def build_report(eval_dir: Path = EVAL) -> str:
    matrix = json.loads((eval_dir / "test_matrix.json").read_text())
    parts = [
        "# Vision evaluation report", "",
        "**Real-camera results: none.** Every automated result below is a "
        "mock, a synthetic scripted case, a source scan, or a measurement "
        "on a public dataset. None of it was produced with the Raspberry "
        "Pi Camera Module 2, so none of it tells you how well Buddy sees "
        "in your home.", "",
        matrix["note"], "",
        "## The 30 test areas", "", matrix_section(matrix), "",
        "Per-area detail (tests, what was measured, manual procedure) is "
        "in `docs/evaluation/test_matrix.json`.", "",
        "## Expression recognition on FER2013 (cross-dataset)", "",
        "FER2013 test set, 48x48 grayscale faces. It is a different "
        "camera, different people and different conditions from the "
        "robot. A facial expression is a visible pattern; it does not "
        "reveal how a person feels, and these numbers must not be read "
        "as the ability to detect emotions.", ""]

    for title, name in RESULTS:
        path = eval_dir / name

        if path.exists():
            parts.append(expression_section(title, json.loads(
                path.read_text())))

    parts += [
        "### Coverage that is missing", "",
        "FER2013 carries no labels for skin tone, age group, glasses, "
        "occlusion, lighting or head angle, so none of those slices could "
        "be evaluated. Disgust has only 55 test images and fear is rarely "
        "recognised, so those per-class figures are unreliable. Before any "
        "claim about those conditions, collect consented, labelled "
        "captures from the robot itself and run "
        "`python -m buddy_core.vision.expression_eval`.", ""]

    return "\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(EVAL / "test_matrix.md"))
    args = ap.parse_args(argv)
    Path(args.out).write_text(build_report())
    print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

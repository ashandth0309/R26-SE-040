"""Evaluation maths (hand-checked values) and the identity CLI."""

import json

import cv2
import numpy as np
import pytest

from buddy_core.vision import evaluation as ev
from buddy_core.vision import models


def test_confusion_matrix_and_per_class_values_by_hand():
    y_true = ["a", "a", "a", "b", "b", "c"]
    y_pred = ["a", "a", "b", "b", "b", "a"]
    labels = ["a", "b", "c"]
    mat = ev.confusion_matrix(y_true, y_pred, labels)
    assert mat.tolist() == [[2, 1, 0], [0, 2, 0], [1, 0, 0]]
    m = ev.per_class_metrics(mat, labels)
    assert m["a"]["precision"] == pytest.approx(2 / 3)
    assert m["a"]["recall"] == pytest.approx(2 / 3)
    assert m["b"]["precision"] == pytest.approx(2 / 3)
    assert m["b"]["recall"] == 1.0
    assert m["c"]["precision"] is None   # never predicted
    assert m["c"]["recall"] == 0.0 and m["c"]["f1"] is None
    assert m["a"]["support"] == 3


def test_unknown_labels_are_ignored_not_miscounted():
    mat = ev.confusion_matrix(["a", "zzz"], ["a", "a"], ["a"])
    assert mat.tolist() == [[1]]


def test_far_frr_and_eer_by_hand():
    genuine = [0.9, 0.8, 0.7, 0.4]
    impostor = [0.1, 0.2, 0.3, 0.6]
    r = ev.verification_metrics(genuine, impostor, [0.35, 0.5, 0.75])
    rows = {row["threshold"]: row for row in r["rows"]}
    assert rows[0.35]["far"] == 0.25 and rows[0.35]["frr"] == 0.0
    assert rows[0.5]["far"] == 0.25 and rows[0.5]["frr"] == 0.25
    assert rows[0.75]["far"] == 0.0 and rows[0.75]["frr"] == 0.5
    assert r["eer_approx"] == 0.25
    assert any("trials" in w for w in r["warnings"])


def test_empty_inputs_give_none_not_zero():
    r = ev.verification_metrics([], [])
    assert r["eer_approx"] is None
    assert all(row["far"] is None and row["frr"] is None
               for row in r["rows"])


def test_identification_counts_the_dangerous_cases_separately():
    trials = [("alice", "alice")] * 3 + [("alice", "bob"),
                                         ("alice", None),
                                         (None, None), (None, "alice")]
    s = ev.identification_summary(trials)
    assert s["correct_named"] == 3 and s["wrong_person_named"] == 1
    assert s["enrolled_not_recognised"] == 1
    assert s["stranger_rejected"] == 1 and s["stranger_named"] == 1
    assert s["false_name_rate_strangers"] == 0.5
    assert s["recognition_rate_enrolled"] == pytest.approx(3 / 5)
    assert len(s["warnings"]) == 2


need_models = pytest.mark.skipif(
    not (models.available("face_detector") and
         models.available("face_embedder")),
    reason="vision models not installed")


@need_models
def test_identity_cli_runs_on_a_folder_and_flags_small_samples(tmp_path):
    skimage_data = pytest.importorskip("skimage.data")
    a = cv2.cvtColor(skimage_data.astronaut(), cv2.COLOR_RGB2BGR)
    b = cv2.cvtColor(skimage_data.camera(), cv2.COLOR_GRAY2BGR)
    for name, img in (("p_one", a), ("p_two", b)):
        folder = tmp_path / name
        folder.mkdir()
        cv2.imwrite(str(folder / "1.png"), img)
        cv2.imwrite(str(folder / "2.png"), cv2.GaussianBlur(img, (0, 0), 1))
    (tmp_path / "p_one" / "bad.png").write_bytes(b"not an image")
    out = tmp_path / "report.json"
    assert ev.main(["identity", "--root", str(tmp_path),
                    "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["people"] == {"p_one": 2, "p_two": 2}
    assert report["genuine"] == 2 and report["impostor"] == 4
    assert report["images_without_usable_face"] == 1
    assert any("trials" in w for w in report["warnings"])
    gen = np.mean([r["frr"] for r in report["rows"]])
    assert gen is not None

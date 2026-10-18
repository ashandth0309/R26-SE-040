"""Phase 10: object detection plumbing, stabilisation and benchmark maths."""

import json

import cv2
import numpy as np
import pytest

from buddy_core.vision import models
from buddy_core.vision.object_eval import evaluate, score_image
from buddy_core.vision.object_stabilizer import (
    ObjectStabilizer, StabilizerConfig)
from buddy_core.vision.objects import (
    COCO, ObjectDetection, ScriptedObjectDetector, category_of, letterbox)


def det(label="cup", score=0.8, box=(100, 100, 200, 200)):
    return ObjectDetection(label, score, box, category_of(label))


def test_coco_has_80_classes_and_categories():
    assert len(COCO) == 80 and len(set(COCO)) == 80
    assert category_of("cup") == "drink"
    assert category_of("dog") == "pet"
    assert category_of("hair drier") == "personal"
    assert category_of("zebra") == "other"


def test_letterbox_keeps_aspect_and_centres():
    img = np.full((100, 200, 3), 255, np.uint8)
    out, (top, left, nh, nw) = letterbox(img, 416)
    assert out.shape == (416, 416, 3)
    assert nw == 416 and nh == 208 and left == 0 and top == 104
    assert out[:top].sum() == 0 and out[top:top + nh].min() == 255


def test_stabilizer_needs_repeated_sightings():
    s = ObjectStabilizer(StabilizerConfig(min_hits=3))
    area = 640 * 480
    assert s.update([det()], 0.0, area).appeared == []
    assert s.update([det()], 0.2, area).appeared == []
    ev = s.update([det()], 0.4, area)
    assert [o.label for o in ev.appeared] == ["cup"]
    assert len(s.objects()) == 1
    assert s.update([det()], 0.6, area).appeared == []     # no repeat event


def test_flicker_never_confirms():
    s = ObjectStabilizer(StabilizerConfig(min_hits=3, max_gap_s=1.0))
    area = 640 * 480
    t = 0.0

    for i in range(10):
        s.update([det()] if i % 5 == 0 else [], t, area)
        t += 1.2

    assert s.objects() == []


def test_disappears_after_gap_and_low_score_or_tiny_ignored():
    s = ObjectStabilizer(StabilizerConfig(min_hits=2, max_gap_s=1.0))
    area = 640 * 480
    s.update([det()], 0.0, area)
    s.update([det()], 0.2, area)
    assert s.update([], 0.8, area).disappeared == []
    ev = s.update([], 1.5, area)
    assert [o.label for o in ev.disappeared] == ["cup"]
    s2 = ObjectStabilizer()
    s2.update([det(score=0.2), det(box=(0, 0, 3, 3))], 0.0, area)
    assert s2._objs == {}


def test_same_label_two_instances_stay_separate():
    s = ObjectStabilizer(StabilizerConfig(min_hits=2))
    area = 640 * 480
    a, b = det(box=(10, 10, 110, 110)), det(box=(300, 300, 400, 400))

    for t in (0.0, 0.1, 0.2):
        s.update([a, b], t, area)

    assert len(s.objects()) == 2


def test_score_image_counts_tp_fp_fn_and_confusion():
    truth = [{"label": "cup", "box": [100, 100, 200, 200]},
             {"label": "bottle", "box": [300, 300, 400, 400]},
             {"label": "book", "box": [10, 10, 60, 60]}]
    dets = [det("cup"), det("cup", 0.6, (305, 305, 405, 405)),
            det("laptop", 0.5, (500, 500, 600, 600))]
    tp, fp, fn, conf = score_image(dets, truth)
    assert tp["cup"] == 1
    assert fp["cup"] == 1 and fn["bottle"] == 1
    assert conf[("bottle", "cup")] == 1
    assert fp["laptop"] == 1 and fn["book"] == 1


def test_evaluate_folder_with_scripted_detector(tmp_path):
    (tmp_path / "images").mkdir()

    for n in ("a.jpg", "b.jpg"):
        cv2.imwrite(str(tmp_path / "images" / n),
                    np.zeros((50, 50, 3), np.uint8))

    labels = {"a.jpg": [{"label": "cup", "box": [100, 100, 200, 200]}],
              "b.jpg": [{"label": "cup", "box": [100, 100, 200, 200]}]}
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    d = ScriptedObjectDetector([[det("cup")], []])
    rep = evaluate(d, str(tmp_path))
    c = rep["per_class"]["cup"]
    assert rep["images"] == 2
    assert (c["tp"], c["fn"], c["fp"]) == (1, 1, 0)
    assert c["precision"] == 1.0 and c["recall"] == 0.5
    assert "not the Pi" in rep["latency_ms"]["note"]


needs_model = pytest.mark.skipif(
    not models.available("object_detector"), reason="model not fetched")


@needs_model
def test_real_detector_finds_cup_on_sample_image():
    from skimage import data

    from buddy_core.vision.objects import NanoDetDetector
    d = NanoDetDetector()
    bgr = cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR)
    labels = {x.label for x in d.detect(bgr)}
    assert "cup" in labels


@needs_model
def test_real_detector_removes_people_by_default():
    from skimage import data

    from buddy_core.vision.objects import NanoDetDetector
    bgr = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    assert all(x.label != "person" for x in NanoDetDetector().detect(bgr))

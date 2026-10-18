"""Phase 20: the benchmark harness itself (not Pi numbers)."""

import json

import numpy as np
import pytest

from buddy_core.vision import benchmark as B
from buddy_core.vision import models


def test_summary_and_timeit():
    s = B.summary([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["n"] == 5 and s["p50_ms"] == 3.0 and s["max_ms"] == 100.0
    assert s["p95_ms"] == 100.0
    t = B.timeit(lambda: sum(range(1000)), repeats=5, warmup=1)
    assert t["n"] == 5 and t["mean_ms"] >= 0


def test_machine_info_never_claims_a_pi_here():
    m = B.machine_info()
    assert {"machine", "cpu_count", "is_raspberry_pi", "opencv"} <= set(m)
    assert m["is_raspberry_pi"] is False or "Raspberry" in (
        m["device_tree_model"] or "")


def test_targets_are_flagged_when_missed():
    rep = {"stages": {"face_detect_320": {"p95_ms": 500.0}},
           "pipelines": [{"profile": "balanced", "frame": {"p95_ms": 999.0}}],
           "peak_rss_mb": 5000.0}
    misses = B.check_targets(rep)
    assert len(misses) == 3
    ok = {"stages": {"face_detect_320": {"p95_ms": 5.0}},
          "pipelines": [{"profile": "balanced", "frame": {"p95_ms": 9.0}}],
          "peak_rss_mb": 100.0}
    assert B.check_targets(ok) == []


def test_test_image_fallbacks(tmp_path):
    img, what = B.test_image(None)
    assert img.shape[2] == 3 and what

    with pytest.raises(SystemExit):
        B.test_image(str(tmp_path / "missing.jpg"))


@pytest.mark.skipif(
    not all(models.available(k) for k in
            ("face_detector", "face_embedder", "expression",
             "object_detector", "pose", "object_embedder_source")),
    reason="models not fetched")
def test_short_end_to_end_run_produces_a_labelled_report(tmp_path):
    out = tmp_path / "r.json"
    assert B.main(["--repeats", "2", "--seconds", "1", "--profiles",
                   "lightweight", "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["machine"]["is_raspberry_pi"] is False
    assert "describes the robot" in rep["note"]
    assert rep["stages"]["face_detect_320"]["n"] == 2
    assert rep["pipelines"][0]["frames"] > 0
    assert isinstance(rep["targets"], dict)
    assert np.isfinite(rep["peak_rss_mb"])

"""Tests for the bring-up and microphone-check tools."""

from __future__ import annotations

import numpy as np
from head_fakes import FS, SPACING, stereo_from_azimuth

from buddy_core.head_tracking import bringup
from buddy_core.head_tracking.mic_check import analyze_capture


def test_bringup_limits_in_mock_mode(capsys):
    out = []
    assert bringup.run(["limits"], out=out.append) == 0
    text = "\n".join(out)
    assert "mode: mock" in text and "pan  range" in text


def test_bringup_nudge_moves_the_mock_neck_only_a_little():
    out = []
    rc = bringup.run(["nudge", "--axis", "pan", "--deg", "5"],
                     sleep=lambda s: None, out=out.append)
    assert rc == 0
    assert any("pan_sign" in line for line in out)


def test_bringup_refuses_large_nudges():
    out = []
    assert bringup.run(["nudge", "--deg", "40"], out=out.append) == 2
    assert any("REFUSED" in line for line in out)


def test_bringup_refuses_hardware_without_gates(tmp_path):
    cfg = tmp_path / "ht.yaml"
    cfg.write_text("head_tracking:\n  hardware:\n    mode: hardware\n")
    out = []
    rc = bringup.run(["center", "--config", str(cfg)], out=out.append)
    assert rc == 2 and any("REFUSED" in line for line in out)


def capture(az, seconds=2.0, seed=0):
    left, right = stereo_from_azimuth(az, seconds=seconds, seed=seed)
    return np.column_stack([left, right]).astype("float32")


def test_mic_check_reports_a_stable_direction_for_synthetic_stereo():
    rep = analyze_capture(capture(30), FS, SPACING)
    assert rep["usable"] and rep["accepted"] >= 6
    assert abs(rep["azimuth_median_deg"] - 30) <= 4
    assert rep["verdict"].startswith("plausible")


def test_mic_check_rejects_single_channel_and_noise():
    mono = np.zeros((FS, 1), dtype="float32")
    assert analyze_capture(mono, FS, SPACING)["usable"] is False
    rng = np.random.default_rng(0)
    noise = 0.05 * rng.standard_normal((FS * 2, 2)).astype("float32")
    rep = analyze_capture(noise, FS, SPACING)
    assert rep["accepted"] == 0 and "do not enable" in rep["verdict"]

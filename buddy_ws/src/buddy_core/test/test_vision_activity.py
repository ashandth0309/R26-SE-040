"""Phases 13: posture rules and activity smoothing (geometry tests only)."""

import cv2
import numpy as np
import pytest
from skeletons import (
    bending, lying, make, sitting, standing, upper_body_only)

from buddy_core.vision import models
from buddy_core.vision import pose as P
from buddy_core.vision.activity import (
    ActivityConfig, ActivityTracker, classify_posture)


@pytest.mark.parametrize("pose,label", [
    (standing(), "standing"), (sitting(), "sitting"),
    (lying(), "lying"), (bending(), "bending"),
    (upper_body_only(), "upright_upper_body")])
def test_posture_labels(pose, label):
    assert classify_posture(pose).label == label


def test_missing_torso_points_is_unknown():
    p = make({P.NOSE: (10, 10), P.L_WRIST: (20, 90)})
    r = classify_posture(p)
    assert r.label == "unknown" and r.confidence == 0.0


def test_tiny_torso_is_unknown():
    p = standing(scale=0.05)
    assert classify_posture(p).reason == "torso_too_small"


def test_sideways_torso_with_tall_body_is_not_lying():
    # torso horizontal but legs extend downward: crouched, not lying
    p = make({
        P.NOSE: (100, 300), P.L_SHOULDER: (160, 300), P.R_SHOULDER: (160, 330),
        P.L_HIP: (300, 300), P.R_HIP: (300, 330),
        P.L_KNEE: (300, 450), P.R_KNEE: (300, 450),
        P.L_ANKLE: (300, 600), P.R_ANKLE: (300, 600)})
    assert classify_posture(p).label == "bending"


def test_lying_either_direction_and_flipped_head():
    a = classify_posture(lying(left=100.0))
    b = classify_posture(lying(left=100.0))
    assert a.label == b.label == "lying"
    flipped = make({i: (700 - x, y) for i, (x, y) in {
        P.NOSE: (100, 380), P.L_SHOULDER: (160, 355),
        P.R_SHOULDER: (160, 405), P.L_HIP: (300, 362), P.R_HIP: (300, 398),
        P.L_KNEE: (420, 362), P.R_KNEE: (420, 398)}.items()})
    assert classify_posture(flipped).label == "lying"


def feed(tr, pose, t0, seconds, fps=10, tid="p"):
    out = None

    for i in range(int(seconds * fps)):
        out = tr.update(tid, pose, t0 + i / fps)

    return out


def test_smoothing_needs_dwell_before_changing():
    tr = ActivityTracker()
    a = feed(tr, standing(), 0.0, 3.0)
    assert a.posture == "standing"
    b = tr.update("p", sitting(), 3.0)
    assert b.posture == "standing" and b.raw_posture == "sitting"
    c = feed(tr, sitting(), 3.1, 3.0)
    assert c.posture == "sitting" and c.posture_s < 3.0


def test_single_noisy_frame_does_not_change_posture():
    tr = ActivityTracker()
    feed(tr, standing(), 0.0, 3.0)
    tr.update("p", lying(), 3.0)
    a = feed(tr, standing(), 3.1, 1.0)
    assert a.posture == "standing"


def test_low_confidence_frames_do_not_vote():
    tr = ActivityTracker()
    feed(tr, standing(), 0.0, 3.0)
    weak = make({k: v for k, v in {
        P.L_SHOULDER: (270, 160), P.R_SHOULDER: (330, 160),
        P.L_HIP: (280, 300), P.R_HIP: (320, 300)}.items()}, vis=0.45)
    a = feed(tr, weak, 3.0, 2.0)
    assert a.posture == "standing"


def test_motion_still_and_moving_and_still_duration():
    tr = ActivityTracker(ActivityConfig(window_s=1.0))
    a = feed(tr, standing(), 0.0, 3.0)
    assert a.motion == "still" and a.still_s >= 1.5

    t, out = 3.0, None

    for i in range(20):                 # walks sideways fast
        out = tr.update("p", standing(x=300 + i * 25), t + i * 0.1)

    assert out.motion == "moving" and out.still_s == 0.0


def test_hand_raise_requires_hold():
    tr = ActivityTracker()
    raised = make({
        P.NOSE: (300, 100), P.L_SHOULDER: (270, 160), P.R_SHOULDER: (330, 160),
        P.L_WRIST: (250, 60), P.R_WRIST: (330, 290),
        P.L_HIP: (280, 300), P.R_HIP: (320, 300)})
    assert not tr.update("p", raised, 0.0).hand_raised
    assert not tr.update("p", raised, 0.3).hand_raised
    assert tr.update("p", raised, 0.8).hand_raised
    assert not tr.update("p", standing(), 1.0).hand_raised


def test_no_pose_returns_unknown_and_tracks_are_independent():
    tr = ActivityTracker()
    feed(tr, standing(), 0.0, 3.0, tid="a")
    feed(tr, sitting(), 0.0, 3.0, tid="b")
    assert tr.update("a", None, 3.1).raw_posture == "unknown"
    assert tr.update("a", standing(), 3.2).posture == "standing"
    assert tr.update("b", sitting(), 3.2).posture == "sitting"
    tr.forget("a")
    assert "a" not in tr._tr


def test_labels_never_include_sleeping_or_medical_words():
    from buddy_core.vision.activity import LABELS
    assert not {"sleeping", "ill", "fainted", "eating"} & set(LABELS)


needs = pytest.mark.skipif(
    not (models.available("pose") and models.available("object_detector")),
    reason="models not fetched")


@needs
def test_real_pose_and_sideways_rotation_recovers_the_same_person():
    from skimage import data

    from buddy_core.vision.objects import NanoDetDetector
    im = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    pe = P.PoseEstimator()
    up = pe.estimate(im, (0, 0, 512, 512))
    assert up is not None and up.rotation == 0
    cw = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    side = pe.estimate(cw, (0, 0, 512, 512), try_sideways=True)
    assert side is not None and side.rotation in (1, 3)
    # nose of the rotated image = rotated nose of the original
    ox, oy = up.xy[P.NOSE]
    expect = np.array([511 - oy, ox])
    assert np.linalg.norm(side.xy[P.NOSE] - expect) < 40
    assert NanoDetDetector  # import check only

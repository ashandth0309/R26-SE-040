"""Expression scores, smoothing, confirmation and the event gate."""

import cv2
import numpy as np
import pytest

from buddy_core.vision import models
from buddy_core.vision.expression import (
    GROUP_NAMES,
    GROUPS,
    LABELS,
    _STD_POINTS,
    align_face,
    scores_from_probs,
    similarity_transform,
    softmax,
)
from buddy_core.vision.expression_eval import (
    expected_calibration_error,
    fit_temperature,
    summarise,
)
from buddy_core.vision.expression_tracker import (
    EventGateConfig,
    ExprState,
    ExpressionService,
    SmoothingConfig,
    TrackExpression,
    conditions,
)
from buddy_core.vision.faces import FaceDetection, FaceQuality

GOOD_LM = ((45, 50), (75, 50), (60, 70), (48, 90), (72, 90))


def face(score=0.95, lm=GOOD_LM):
    return FaceDetection(20, 20, 80, 90, score, lm)


GOOD_Q = FaceQuality(140, 40, 300, 0.0, 0.0, 0.0, ())


def probs_for(label, strength=0.8):
    p = np.full(len(LABELS), (1 - strength) / (len(LABELS) - 1))
    p[LABELS.index(label)] = strength

    return scores_from_probs(p)


def run(track, seq, start=0.0, dt=0.2, bad=()):
    """Feed (label|None) items; return the last estimate."""

    est = None
    for i, label in enumerate(seq):
        scores = probs_for(label) if label else None
        est = track.observe(scores, bad if label is None else (),
                            start + i * dt)

    return est


# ---------- scores ----------

def test_softmax_and_scores_are_consistent():
    p = softmax(np.array([2.0, 1.0, 0.1, 0, 0, 0, 0]))
    assert p.sum() == pytest.approx(1.0)
    s = scores_from_probs(p)
    assert s.top == "angry" and s.top_prob == pytest.approx(p[0])
    assert s.margin == pytest.approx(p[0] - p[1])
    assert 0 <= s.entropy <= 1
    assert sum(s.group_probs().values()) == pytest.approx(1.0)
    assert set(s.group_probs()) == set(GROUP_NAMES)
    flat = scores_from_probs(np.full(7, 1 / 7))
    assert flat.entropy == pytest.approx(1.0) and flat.margin == 0.0


def test_temperature_softens_confidence():
    z = np.array([4.0, 0, 0, 0, 0, 0, 0])
    assert softmax(z, 3.0).max() < softmax(z, 1.0).max()


def test_every_label_has_exactly_one_group():
    assert set(GROUPS) == set(LABELS)
    assert set(GROUPS.values()) == set(GROUP_NAMES)


def test_alignment_maps_landmarks_onto_the_template():
    rng = np.random.default_rng(0)
    theta, scale, shift = np.radians(20), 1.7, np.array([300.0, 120.0])
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    src = (_STD_POINTS @ rot.T) * scale + shift
    m = similarity_transform(src, _STD_POINTS)
    mapped = src @ m[:, :2].T + m[:, 2]
    assert np.abs(mapped - _STD_POINTS).max() < 1e-6
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    assert align_face(img, src).shape == (112, 112, 3)


# ---------- smoothing ----------

def test_nothing_is_claimed_from_too_few_frames():
    t = TrackExpression()
    est = run(t, ["happy"] * 3)
    assert est.state is ExprState.UNKNOWN and est.reason == "too_few_frames"
    assert est.label is None


def test_steady_smile_becomes_confirmed_positive_after_time():
    t = TrackExpression()
    mid = run(t, ["happy"] * 5)
    assert mid.state is ExprState.TENTATIVE and mid.label == "positive"
    done = run(t, ["happy"] * 10, start=1.0)
    assert done.state is ExprState.CONFIRMED and done.group == "positive"
    assert "visible expression" in done.note


def test_one_odd_frame_does_not_change_the_call():
    t = TrackExpression()
    run(t, ["happy"] * 15)
    est = run(t, ["sad"] + ["happy"] * 3, start=3.0)
    assert est.state is ExprState.CONFIRMED and est.label == "positive"


def test_sustained_change_replaces_the_call_only_after_a_hold():
    t = TrackExpression(SmoothingConfig(window_s=1.0, change_hold_s=1.0))
    run(t, ["happy"] * 15)
    first = run(t, ["sad"] * 3, start=3.0)
    assert first.label == "positive"          # window still mostly happy
    later = run(t, ["sad"] * 20, start=3.6)
    assert later.label == "negative" and later.state is ExprState.CONFIRMED


def test_competing_strong_expressions_are_unknown_not_a_guess():
    t = TrackExpression()
    est = run(t, ["happy", "surprised"] * 10)
    assert est.state is ExprState.UNKNOWN and est.reason == "ambiguous"
    assert est.label is None


def test_flat_scores_never_become_a_negative_call():
    """Four of seven labels are negative, so a uniform output would sum to
    57% 'negative'.  It must stay unknown."""

    t = TrackExpression()
    flat = scores_from_probs(np.full(7, 1 / 7))
    assert flat.group_probs()["negative"] > 0.55
    est = None
    for i in range(40):
        est = t.observe(flat, (), i * 0.2)
    assert est.state is ExprState.UNKNOWN and est.reason == "uninformative"
    near = np.array([0.20, 0.15, 0.15, 0.10, 0.20, 0.10, 0.10])
    est2 = None
    for i in range(40):
        est2 = t.observe(scores_from_probs(near), (), 10 + i * 0.2)
    assert est2.state is not ExprState.CONFIRMED


def test_poor_conditions_never_feed_the_smoother():
    t = TrackExpression()
    est = run(t, ["happy"] * 10)
    assert est.frames >= 4
    for i in range(10):
        est = t.observe(probs_for("angry"), ("too_dark",), 3.0 + i * 0.2)
    assert est.label != "negative"
    gone = t.observe(probs_for("angry"), ("too_dark",), 20.0)
    assert gone.state is ExprState.UNKNOWN and gone.reason == "too_dark"


def test_stale_track_resets_instead_of_remembering_an_old_mood():
    t = TrackExpression(SmoothingConfig(max_gap_s=2.0))
    run(t, ["happy"] * 15)
    late = t.observe(None, (), 30.0)
    assert late.state is ExprState.UNKNOWN
    again = run(t, ["sad"] * 4, start=30.2)
    assert again.state is ExprState.TENTATIVE   # nothing carried over
    assert again.label == "negative"


def test_specific_label_mode_works_too():
    t = TrackExpression(SmoothingConfig(use_groups=False))
    est = run(t, ["surprised"] * 15)
    assert est.label == "surprised" and est.group == "surprise"


# ---------- conditions ----------

def test_conditions_flag_lighting_pose_and_possible_occlusion():
    cfg = SmoothingConfig()
    assert conditions(face(), GOOD_Q, cfg) == ()
    assert conditions(face(), None, cfg) == ("no_quality",)
    dark = FaceQuality(10, 5, 5, 0.0, 0.0, 0.0, ("too_dark", "blurry"))
    assert conditions(face(), dark, cfg) == ("too_dark", "blurry")
    assert "possible_occlusion" in conditions(face(score=0.6), GOOD_Q, cfg)
    upside = ((45, 50), (75, 50), (60, 95), (48, 90), (72, 90))
    assert "possible_occlusion" in conditions(face(lm=upside), GOOD_Q, cfg)
    outside = ((45, 50), (75, 50), (60, 70), (48, 400), (72, 400))
    assert "possible_occlusion" in conditions(face(lm=outside), GOOD_Q, cfg)
    turned = FaceQuality(140, 40, 300, 0.7, 0.0, 0.0, ("head_turned",))
    assert conditions(face(), turned, cfg) == ("head_turned",)


# ---------- events ----------

def drive(svc, labels, dt=0.2, start=0.0, tid=1):
    events = []
    for i, label in enumerate(labels):
        s = probs_for(label) if label else None
        _, ev = svc.update(tid, s, face(), GOOD_Q, start + i * dt)
        if ev:
            events.append(ev)

    return events


def test_constant_negative_expression_gives_one_event_not_hundreds():
    svc = ExpressionService()
    events = drive(svc, ["sad"] * 1000)
    assert len(events) == 1
    assert events[0].label == "negative" and events[0].previous is None
    assert svc.frames == 1000 and svc.events_emitted == 1


def test_neutral_baseline_is_not_news():
    svc = ExpressionService()
    assert drive(svc, ["neutral"] * 200) == []


def test_flickering_between_expressions_is_rate_limited():
    svc = ExpressionService(gate=EventGateConfig(min_event_gap_s=20))
    seq = (["happy"] * 12 + ["sad"] * 12) * 10
    events = drive(svc, seq)
    # ~48 s of video: far fewer events than expression flips.
    assert 1 <= len(events) <= 3


def test_events_are_per_track_and_forgotten_with_the_track():
    svc = ExpressionService()
    a = drive(svc, ["happy"] * 30, tid=1)
    b = drive(svc, ["happy"] * 30, tid=2)
    assert len(a) == 1 and len(b) == 1
    svc.prune({2})
    assert 1 not in svc._tracks and 2 in svc._tracks
    assert len(drive(svc, ["happy"] * 30, tid=1)) == 1  # new session


def test_bad_frames_produce_unknown_and_no_event():
    svc = ExpressionService()
    dark = FaceQuality(5, 5, 5, 0.0, 0.0, 0.0, ("too_dark",))
    events = []
    for i in range(60):
        est, ev = svc.update(1, probs_for("angry"), face(), dark, i * 0.2)
        if ev:
            events.append(ev)
    assert events == [] and est.state is ExprState.UNKNOWN


# ---------- evaluation maths ----------

def test_ece_is_zero_when_confidence_matches_accuracy():
    probs = np.array([[0.8, 0.2]] * 10)
    labels = np.array([0] * 8 + [1] * 2)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)
    over = np.array([[0.99, 0.01]] * 10)
    assert expected_calibration_error(over, labels) == pytest.approx(0.19)


def test_temperature_fit_softens_an_overconfident_model():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 600)
    logits = rng.normal(0, 1, (600, 3))
    logits[np.arange(600), labels] += 1.0
    logits *= 4.0                      # overconfident on purpose
    assert fit_temperature(logits, labels) > 2.0


def test_summarise_counts_neutral_called_negative():
    y_true = ["neutral"] * 4 + ["sad"] * 2 + ["happy"] * 2
    y_pred = ["neutral", "sad", "angry", "neutral", "sad", "neutral",
              "happy", "happy"]
    r = summarise(y_true, y_pred)
    assert r["images_scored"] == 8
    assert r["neutral_faces_called_negative"] == {"count": 2, "of": 4,
                                                  "rate": 0.5}
    assert r["accuracy"] == pytest.approx(5 / 8)
    assert r["per_class"]["sad"]["recall"] == 0.5
    assert r["group_per_class"]["negative"]["support"] == 2


# ---------- real model (skipped if not installed) ----------

@pytest.mark.skipif(not (models.available("expression") and
                         models.available("face_detector")),
                    reason="vision models not installed")
def test_real_model_returns_valid_distribution_on_a_portrait():
    skimage_data = pytest.importorskip("skimage.data")
    from buddy_core.vision.expression import ExpressionClassifier
    from buddy_core.vision.faces import YuNetDetector

    img = cv2.cvtColor(skimage_data.astronaut(), cv2.COLOR_RGB2BGR)
    det = YuNetDetector(str(models.model_path("face_detector"))
                        ).detect(img)[0]
    clf = ExpressionClassifier(str(models.model_path("expression")))
    scores = clf.predict(img, det)
    assert sum(scores.probs) == pytest.approx(1.0, abs=1e-5)
    assert scores.top in LABELS
    assert clf.predict(img, FaceDetection(0, 0, 50, 50, 0.9)) is None

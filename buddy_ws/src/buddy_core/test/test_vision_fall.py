"""Phase 14: fall detection logic on scripted skeleton sequences.

These tests check the *logic* (timing, context, asking before alerting).
They say nothing about detection accuracy on real people.
"""

import numpy as np
from skeletons import (
    bending, hold, lying, sitting, standing, transition)

from buddy_core.vision.fall import FallConfig, FallDetector
from buddy_core.vision.pose import Pose


def run(det, frames, tid="p", furniture=()):
    out = []

    for t, pose in frames:
        out += det.update(tid, pose, t, furniture)

    return out


def kinds(events):
    return [e.kind for e in events]


def fall_frames(rapid=0.6, lie_s=40.0, start=3.0):
    f = hold(standing(), 0.0, start)
    f += transition(standing(), lying(), start, rapid)
    t = start + rapid
    f += hold(lying(), t, lie_s)

    return f


def test_rapid_fall_asks_then_alerts_when_nobody_answers():
    d = FallDetector()
    ev = run(d, fall_frames())
    assert kinds(ev) == ["fall_suspected", "fall_alert"]
    sus, alert = ev
    assert sus.needs_voice_check and sus.severity == "check"
    assert alert.severity == "alert" and "no_response" in alert.reasons
    assert alert.t - sus.t >= 19.5          # waited for an answer first
    assert sus.t - 3.0 < 10.0               # asked a few seconds in


def test_person_says_ok_cancels_and_no_alert_follows():
    d = FallDetector()
    frames = fall_frames()
    before = [fr for fr in frames if fr[0] <= 10.5]
    ev = run(d, before)
    assert kinds(ev) == ["fall_suspected"]
    ok = d.on_voice("I'm okay, don't worry", 10.6)
    assert kinds(ok) == ["fall_recovered"]
    after = [fr for fr in frames if fr[0] > 10.6]
    assert "fall_alert" not in kinds(run(d, after))


def test_help_request_alerts_immediately():
    d = FallDetector()
    frames = fall_frames()
    run(d, [fr for fr in frames if fr[0] <= 10.5])
    ev = d.on_voice("help me, I can't get up", 11.0)
    assert kinds(ev) == ["fall_alert"]
    assert "asked_for_help" in ev[0].reasons


def test_getting_up_before_the_check_ends_it():
    d = FallDetector()
    f = hold(standing(), 0.0, 3.0)
    f += transition(standing(), lying(), 3.0, 0.6)
    f += hold(lying(), 3.6, 3.0)
    f += transition(lying(), standing(), 6.6, 1.5)
    f += hold(standing(), 8.1, 4.0)
    ev = run(d, f)
    assert "fall_alert" not in kinds(ev)
    assert "fall_suspected" not in kinds(ev)


def test_getting_up_after_the_question_cancels_the_alert():
    d = FallDetector()
    f = fall_frames(lie_s=9.0)
    t = f[-1][0] + 0.1
    f += transition(lying(), standing(), t, 1.5)
    f += hold(standing(), t + 1.5, 5.0)
    ev = run(d, f)
    assert kinds(ev) == ["fall_suspected", "fall_recovered"]


def test_slow_lowering_is_resting_not_a_fall():
    d = FallDetector()
    f = hold(standing(), 0.0, 3.0)
    f += transition(standing(), lying(), 3.0, 7.0)
    f += hold(lying(), 10.0, 40.0)
    ev = run(d, f)
    assert kinds(ev) == ["resting_lying"]
    assert "lowered_slowly" in ev[0].reasons


def test_quickly_sitting_down_is_not_a_fall():
    d = FallDetector()
    f = hold(standing(), 0.0, 3.0)
    f += transition(standing(), sitting(), 3.0, 0.5)
    f += hold(sitting(), 3.5, 30.0)
    assert run(d, f) == []


def test_bending_over_and_back_is_not_a_fall():
    d = FallDetector()
    f = hold(standing(), 0.0, 3.0)
    f += transition(standing(), bending(), 3.0, 1.0)
    f += hold(bending(), 4.0, 5.0)
    f += transition(bending(), standing(), 9.0, 1.0)
    f += hold(standing(), 10.0, 5.0)
    assert run(d, f) == []


def test_person_first_seen_lying_is_not_called_a_fall():
    d = FallDetector()
    ev = run(d, hold(lying(), 0.0, 40.0))
    assert kinds(ev) == ["resting_lying"]


def test_small_drop_is_not_enough():
    d = FallDetector()
    high_lying = lying(y=330.0)           # hips barely lower than standing
    f = hold(standing(), 0.0, 3.0)
    f += transition(standing(), high_lying, 3.0, 0.6)
    f += hold(high_lying, 3.6, 40.0)
    assert "fall_suspected" not in kinds(run(d, f))


def test_rapid_arrival_on_furniture_waits_longer():
    d = FallDetector()
    sofa = [(0, 300, 800, 480)]
    frames = fall_frames(lie_s=30.0)
    early = [fr for fr in frames if fr[0] <= 3.6 + 9.0]
    assert "fall_suspected" not in kinds(run(d, early, furniture=sofa))
    late = [fr for fr in frames if 3.6 + 9.0 < fr[0] <= 3.6 + 16.0]
    ev = run(d, late, furniture=sofa)
    assert "fall_suspected" in kinds(ev)
    assert "on_furniture_longer_wait" in ev[0].reasons


def test_view_dropout_while_lying_keeps_the_clock_running():
    d = FallDetector()
    frames = fall_frames(lie_s=7.0)
    ev = run(d, frames)
    t = frames[-1][0]
    ev += run(d, [(t + 0.1 * i, None) for i in range(1, 400)])
    assert kinds(ev)[:1] == ["fall_suspected"]
    assert "fall_alert" not in kinds(ev)         # person left view: no alert
    assert "lost_after_descent" in kinds(ev)


def test_short_dropout_still_reaches_alert():
    d = FallDetector()
    frames = fall_frames(lie_s=7.0)
    ev = run(d, frames)
    t = frames[-1][0]
    ev += run(d, [(t + 0.5 * i, None) for i in range(1, 60)])
    assert "fall_alert" in kinds(ev) or "lost_after_descent" in kinds(ev)


def test_alert_cooldown_prevents_repeats():
    d = FallDetector(FallConfig(alert_cooldown_s=600.0))
    frames = fall_frames(lie_s=40.0)
    ev = run(d, frames)
    t = frames[-1][0] + 0.1
    f = transition(lying(), standing(), t, 1.0)
    f += hold(standing(), t + 1.0, 4.0)
    f += transition(standing(), lying(), t + 5.0, 0.6)
    f += hold(lying(), t + 5.6, 40.0)
    ev += run(d, f)
    assert kinds(ev).count("fall_alert") == 1
    assert kinds(ev).count("fall_suspected") == 2


def test_tracks_are_independent_and_low_confidence_is_ignored():
    d = FallDetector()
    a = fall_frames()
    ev = run(d, a, tid="a")
    ev += run(d, hold(standing(), 0.0, 60.0), tid="b")
    assert all(e.track_id == "a" for e in ev)
    weak = Pose(lying().xy, lying().visibility, 0.2, 0, lying().box)
    d2 = FallDetector()
    assert run(d2, [(0.0, standing())] * 1 + [
        (0.1 * i, weak) for i in range(1, 100)]) == []


def test_vanishing_during_fast_descent_is_flagged_but_never_alerts():
    d = FallDetector()
    f = hold(standing(), 0.0, 3.0)
    f += transition(standing(), lying(), 3.0, 0.3)[:2]
    ev = run(d, f)
    ev += run(d, [(3.2 + 0.2 * i, None) for i in range(1, 40)])
    assert "fall_alert" not in kinds(ev)


def test_no_images_or_identity_in_events():
    d = FallDetector()
    ev = run(d, fall_frames())

    for e in ev:
        assert not any(isinstance(v, np.ndarray) for v in e.details.values())
        assert set(e.__dict__) == {"kind", "track_id", "t", "severity",
                                   "reasons", "needs_voice_check",
                                   "details"}


def test_slow_or_jittery_pose_rate_still_confirms_lying():
    """Pose at 0.55 s, 1 s or 1.5 s must not silently never confirm."""

    for step in (0.55, 1.0, 1.5):
        frames, last = [], -9.0

        for t, pose in fall_frames():
            if t - last >= step - 1e-9:
                frames.append((t, pose))
                last = t

        assert kinds(run(FallDetector(), frames)) == [
            "fall_suspected", "fall_alert"], step

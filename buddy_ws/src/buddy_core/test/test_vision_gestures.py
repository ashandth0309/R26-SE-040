"""Phase 15: arm gestures from scripted skeleton motion."""

import math

from skeletons import make

from buddy_core.vision import pose as P
from buddy_core.vision.gestures import (
    GestureConfig, GestureRecognizer, pointed_object)


def arms(lw, rw, lel=None, rel=None, x=300.0):
    pts = {P.NOSE: (x, 100), P.L_SHOULDER: (x - 30, 160),
           P.R_SHOULDER: (x + 30, 160), P.L_HIP: (x - 20, 300),
           P.R_HIP: (x + 20, 300), P.L_WRIST: lw, P.R_WRIST: rw}
    pts[P.L_ELBOW] = lel or ((lw[0] + x - 30) / 2, (lw[1] + 160) / 2)
    pts[P.R_ELBOW] = rel or ((rw[0] + x + 30) / 2, (rw[1] + 160) / 2)

    return make(pts)


REST = ((260, 290), (340, 290))


def run(gr, poses_at, tid="p"):
    out = []

    for t, p in poses_at:
        out += gr.update(tid, p, t)

    return out


def names(ev):
    return [g.name for g in ev]


def test_raised_hand_needs_hold_and_reports_side():
    gr = GestureRecognizer()
    f = [(i / 10, arms((250, 80), REST[1])) for i in range(15)]
    ev = run(gr, f)
    assert names(ev) == ["raised_hand"] and ev[0].side == "left"
    assert ev[0].t >= 0.8


def test_brief_arm_lift_is_ignored():
    gr = GestureRecognizer()
    f = [(i / 10, arms((250, 80), REST[1])) for i in range(4)]
    f += [(0.4 + i / 10, arms(*REST)) for i in range(10)]
    assert run(gr, f) == []


def test_wave_detected_by_side_to_side_motion():
    gr = GestureRecognizer()
    f = []

    for i in range(30):
        x = 330 + 45 * math.sin(i * 1.2)
        f.append((i / 10, arms(REST[0], (x, 120))))

    ev = run(gr, f)
    assert "wave" in names(ev)
    assert "raised_hand" not in names(ev)
    assert next(g for g in ev if g.name == "wave").side == "right"


def test_slow_drift_is_not_a_wave():
    gr = GestureRecognizer()
    f = [(i / 10, arms(REST[0], (330 + i * 2, 120))) for i in range(30)]
    assert "wave" not in names(run(gr, f))


def test_low_hand_swinging_is_not_a_wave():
    gr = GestureRecognizer()
    f = [(i / 10, arms(REST[0], (330 + 45 * math.sin(i * 1.2), 330)))
         for i in range(30)]
    assert "wave" not in names(run(gr, f))


def test_both_hands_up():
    gr = GestureRecognizer()
    f = [(i / 10, arms((250, 80), (350, 80))) for i in range(15)]
    assert "both_hands_up" in names(run(gr, f))


def test_pointing_needs_straight_steady_extended_arm():
    gr = GestureRecognizer()
    point = arms(REST[0], (470, 170), rel=(400, 165))
    ev = run(gr, [(i / 10, point) for i in range(15)])
    g = [e for e in ev if e.name == "pointing"]
    assert len(g) == 1 and g[0].side == "right"
    assert g[0].direction[0] > 0.9
    bent = arms(REST[0], (400, 120), rel=(420, 200))
    gr2 = GestureRecognizer()
    assert "pointing" not in names(run(gr2, [(i / 10, bent)
                                             for i in range(15)]))
    crooked = arms(REST[0], (470, 170), rel=(380, 250))   # far but bent
    gr4 = GestureRecognizer()
    assert "pointing" not in names(run(gr4, [(i / 10, crooked)
                                             for i in range(15)]))
    short = arms(REST[0], (350, 170), rel=(340, 168))
    gr3 = GestureRecognizer()
    assert "pointing" not in names(run(gr3, [(i / 10, short)
                                             for i in range(15)]))


def test_cooldown_prevents_repeat_events():
    gr = GestureRecognizer(GestureConfig(cooldown_s=2.0))
    f = [(i / 10, arms((250, 80), REST[1])) for i in range(30)]
    assert names(run(gr, f)).count("raised_hand") == 2


def test_missing_pose_or_shoulders_resets_and_is_safe():
    gr = GestureRecognizer()
    assert gr.update("p", None, 0.0) == []
    only_wrist = make({P.L_WRIST: (10, 10)})
    assert gr.update("p", only_wrist, 0.1) == []


def test_tracks_are_independent():
    gr = GestureRecognizer()
    f = [(i / 10, arms((250, 80), REST[1])) for i in range(15)]
    a = run(gr, f, tid="a")
    b = run(gr, [(i / 10, arms(*REST)) for i in range(15)], tid="b")
    assert names(a) == ["raised_hand"] and b == []


def test_pointed_object_picks_nearest_on_ray():
    boxes = [("cup", (500, 150, 560, 200)), ("tv", (800, 100, 900, 250)),
             ("chair", (0, 0, 50, 50))]
    assert pointed_object((330, 170), (1.0, 0.0), boxes) == "cup"
    assert pointed_object((330, 170), (-1.0, 0.0), boxes) is None
    assert pointed_object((330, 170), (0.0, 1.0), boxes) is None
    assert pointed_object((330, 170), (1.0, 0.0), []) is None


def test_gestures_have_no_motion_fields():
    from buddy_core.vision.gestures import Gesture
    fields = set(Gesture.__dataclass_fields__)
    assert not {"velocity", "linear", "angular", "command"} & fields

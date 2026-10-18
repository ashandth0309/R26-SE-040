"""Arm gestures from the pose skeleton: wave, raised hand, both hands up,
pointing.

Scope, stated plainly:

* Only whole-arm gestures are recognised.  Hand-shape gestures (thumbs up,
  fist, open palm, finger counts) are NOT implemented: that needs the
  palm and hand-landmark models, which are staged in the model manifest
  but have not been integrated or validated on any real hand images.
* A gesture is an *event*.  It never moves the robot.  A "stop" style
  gesture (both hands up) is only a hint for the voice and safety layers
  to ask or to stop; the vision subsystem does not command motors.
* Thresholds are starting values, untested on real people.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from buddy_core.vision.pose import (
    L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, R_ELBOW, R_HIP, R_SHOULDER,
    R_WRIST, Pose)


@dataclass
class GestureConfig:
    min_vis: float = 0.5
    window_s: float = 2.0
    wave_reversals: int = 3
    wave_amplitude_torsos: float = 0.30
    wave_max_wrist_below_shoulder: float = 0.30   # torso lengths
    raise_margin_torsos: float = 0.15
    hold_s: float = 0.8
    point_min_reach_torsos: float = 0.9
    point_straight_deg: float = 150.0
    point_hold_s: float = 1.0
    point_max_speed: float = 0.5                  # torso/s, must be steady
    cooldown_s: float = 2.0


@dataclass(frozen=True)
class Gesture:
    name: str               # wave | raised_hand | both_hands_up | pointing
    track_id: str
    t: float
    side: str | None = None             # left | right (person's side)
    direction: tuple[float, float] | None = None   # unit vector in image
    details: dict = field(default_factory=dict)


@dataclass
class _G:
    wrist: dict = field(default_factory=lambda: {
        "left": deque(), "right": deque()})
    since: dict = field(default_factory=dict)
    last_fired: dict = field(default_factory=dict)


def _angle(a, b, c) -> float:
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)

    if n < 1e-6:
        return 0.0

    return math.degrees(math.acos(float(np.clip(v1 @ v2 / n, -1, 1))))


class GestureRecognizer:
    SIDES = {"left": (L_SHOULDER, L_ELBOW, L_WRIST),
             "right": (R_SHOULDER, R_ELBOW, R_WRIST)}

    def __init__(self, cfg: GestureConfig | None = None) -> None:
        self.cfg = cfg or GestureConfig()
        self._g: dict[str, _G] = {}

    def forget(self, track_id: str) -> None:
        self._g.pop(track_id, None)

    def _fire(self, st: _G, name: str, now: float) -> bool:
        if now - st.last_fired.get(name, -1e9) < self.cfg.cooldown_s:
            return False

        st.last_fired[name] = now

        return True

    def _held(self, st: _G, key: str, cond: bool, now: float,
              hold: float) -> bool:
        if not cond:
            st.since.pop(key, None)

            return False

        st.since.setdefault(key, now)

        return now - st.since[key] >= hold

    def update(self, track_id: str, pose: Pose | None, now: float
               ) -> list[Gesture]:
        cfg = self.cfg
        st = self._g.setdefault(track_id, _G())

        if pose is None:
            for dq in st.wrist.values():
                dq.clear()

            st.since.clear()

            return []

        mv = cfg.min_vis
        ls, rs = pose.point(L_SHOULDER, mv), pose.point(R_SHOULDER, mv)
        lh, rh = pose.point(L_HIP, 0.3), pose.point(R_HIP, 0.3)

        if ls is None or rs is None:
            st.since.clear()

            return []

        sh_mid = (ls + rs) / 2
        torso = None

        if lh is not None and rh is not None:
            torso = float(np.hypot(*((lh + rh) / 2 - sh_mid)))

        if torso is None or torso < 20:
            torso = float(np.hypot(*(ls - rs))) * 1.4     # shoulder width

        if torso < 20:
            return []

        out: list[Gesture] = []
        up = {}

        for side, (si, ei, wi) in self.SIDES.items():
            sh, el, wr = (pose.point(si, mv), pose.point(ei, mv),
                          pose.point(wi, mv))
            dq = st.wrist[side]

            if sh is None or wr is None:
                dq.clear()
                st.since.pop(f"raise_{side}", None)
                st.since.pop(f"point_{side}", None)
                up[side] = False

                continue

            dq.append((now, float(wr[0] - sh[0]), float(wr[1])))

            while dq and now - dq[0][0] > cfg.window_s:
                dq.popleft()

            above = wr[1] < sh[1] - cfg.raise_margin_torsos * torso
            up[side] = above

            if self._held(st, f"raise_{side}", above, now, cfg.hold_s) \
                    and not self._waving(dq, torso):
                if self._fire(st, f"raised_{side}", now):
                    out.append(Gesture("raised_hand", track_id, now, side))

            if self._waving(dq, torso) and \
                    wr[1] < sh[1] + cfg.wave_max_wrist_below_shoulder * \
                    torso:
                if self._fire(st, "wave", now):
                    out.append(Gesture("wave", track_id, now, side))

            out += self._pointing(st, track_id, side, sh, el, wr, torso,
                                  dq, now)

        if self._held(st, "both_up", up.get("left", False) and
                      up.get("right", False), now, cfg.hold_s):
            if self._fire(st, "both_hands_up", now):
                out.append(Gesture("both_hands_up", track_id, now))

        return out

    def _waving(self, dq, torso: float) -> bool:
        cfg = self.cfg

        if len(dq) < 6:
            return False

        xs = np.array([x for _, x, _ in dq]) / torso

        if xs.max() - xs.min() < cfg.wave_amplitude_torsos:
            return False

        d = np.sign(np.diff(xs))
        d = d[d != 0]

        return int(np.sum(d[1:] != d[:-1])) >= cfg.wave_reversals

    def _pointing(self, st, track_id, side, sh, el, wr, torso, dq, now
                  ) -> list[Gesture]:
        cfg = self.cfg
        key = f"point_{side}"

        if el is None:
            st.since.pop(key, None)

            return []

        reach = float(np.hypot(*(wr - sh))) / torso
        straight = _angle(sh, el, wr) >= cfg.point_straight_deg
        level = abs(wr[1] - sh[1]) < 0.6 * torso
        speed = 0.0

        if len(dq) >= 2:
            (t0, x0, y0), (t1, x1, y1) = dq[0], dq[-1]
            speed = math.hypot(x1 - x0, y1 - y0) / torso / max(t1 - t0, 1e-3)

        ok = (reach >= cfg.point_min_reach_torsos and straight and level
              and speed <= cfg.point_max_speed)

        if self._held(st, key, ok, now, cfg.point_hold_s) and \
                self._fire(st, f"point_{side}", now):
            v = wr - sh
            v = v / (np.hypot(*v) or 1.0)

            return [Gesture("pointing", track_id, now, side,
                            (float(v[0]), float(v[1])),
                            {"reach_torsos": reach})]

        return []


def pointed_object(pose_origin, direction, boxes, max_dist: float = 4000.0):
    """The first box hit by the ray from the shoulder through the wrist.

    ``boxes`` is a list of (label, (x1, y1, x2, y2)).  Returns the label
    of the nearest box along the ray, or None.  Image-space only: it
    cannot tell a near object from a far one on the same line.
    """

    ox, oy = pose_origin
    dx, dy = direction
    best, best_t = None, max_dist

    for label, (x1, y1, x2, y2) in boxes:
        tmin, tmax = 0.0, max_dist

        for o, d, lo, hi in ((ox, dx, x1, x2), (oy, dy, y1, y2)):
            if abs(d) < 1e-9:
                if not lo <= o <= hi:
                    tmin, tmax = 1.0, 0.0

                continue

            t1, t2 = (lo - o) / d, (hi - o) / d
            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))

        if tmin <= tmax and tmin < best_t:
            best, best_t = label, tmin

    return best

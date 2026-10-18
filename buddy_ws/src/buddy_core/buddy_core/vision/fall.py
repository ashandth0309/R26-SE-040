"""Fall detection (rebuild).

The old module thresholded dark pixels.  This one reads the *sequence* of
body postures from the pose skeleton:

    upright  ->  quickly lying  ->  stays lying  ->  Buddy asks  ->  alert

and tells a fall apart from lying down on purpose by how the change
happened: a controlled lowering takes several seconds and often ends on a
bed or sofa; a fall is fast.  After a suspected fall Buddy first asks out
loud ("Are you okay?").  An alert is raised only if there is no answer and
the person has not got up, or if they ask for help.

Limits (read before trusting it):

* One camera, no depth, no floor plane.  All distances are relative to
  the person's own torso length.
* Thresholds are starting values.  No real fall recordings exist here, so
  there are no sensitivity or false-alarm numbers.  The tests use scripted
  skeletons and show the logic, not detection accuracy.
* A fall out of view, behind furniture, or in the dark is missed.  Floor
  exercise or yoga looks like lying; a person lowered quickly onto a
  sofa can look like a fall.
* It is not a medical device and must not be the only safety measure.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

from buddy_core.vision.activity import PostureConfig, classify_posture
from buddy_core.vision.pose import Pose

UPRIGHT = {"standing", "sitting", "upright_upper_body", "bending"}


@dataclass
class FallConfig:
    rapid_s: float = 1.5               # last stretch of the drop takes this
    history_s: float = 10.0            # how far back to look for the drop
    settle_s: float = 1.0              # wait for the body to come to rest
    min_drop_torsos: float = 0.8       # hips must fall this many torsos
    lying_confirm_s: float = 1.0
    lying_ratio: float = 0.7
    confirm_still_s: float = 6.0       # lying this long before asking
    response_s: float = 20.0           # time to answer "are you okay"
    recover_upright_s: float = 2.0
    alert_cooldown_s: float = 600.0
    min_pose_conf: float = 0.5
    furniture_slow_extra_s: float = 6.0   # on a bed/sofa, wait longer
    lost_after_s: float = 3.0
    lost_grace_s: float = 10.0         # keep timing if the view drops out


@dataclass(frozen=True)
class FallEvent:
    kind: str            # fall_suspected | fall_alert | fall_recovered |
    #                      resting_lying | lost_after_descent
    track_id: str
    t: float
    severity: str        # info | check | alert
    reasons: tuple[str, ...]
    needs_voice_check: bool = False
    details: dict = field(default_factory=dict)


@dataclass
class _T:
    hist: deque = field(default_factory=deque)       # (t, label, hip_y)
    ref_len: float | None = None
    last_upright_t: float | None = None
    last_upright_hip: float | None = None
    last_seen: float = 0.0
    lying_since: float | None = None
    rapid: bool = False
    on_furniture: bool = False
    drop: float = 0.0
    transition_s: float | None = None
    state: str = "normal"       # normal | resting | suspect | asked | alerted
    suspect_at: float = 0.0
    asked_at: float = 0.0
    upright_since: float | None = None
    alerted_at: float = -1e12
    pending_lost: bool = False
    classified: bool = False


def _inside(point, box) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


class FallDetector:
    def __init__(self, cfg: FallConfig | None = None,
                 pcfg: PostureConfig | None = None) -> None:
        self.cfg = cfg or FallConfig()
        self.pcfg = pcfg or PostureConfig()
        self._t: dict[str, _T] = {}

    def forget(self, track_id: str) -> None:
        self._t.pop(track_id, None)

    def states(self) -> dict[str, str]:
        return {k: v.state for k, v in self._t.items()}

    def _on_furniture(self, pose: Pose, furniture) -> bool:
        cx = (pose.box[0] + pose.box[2]) / 2
        cy = (pose.box[1] + pose.box[3]) / 2

        return any(_inside((cx, cy), f) for f in furniture or ())

    def update(self, track_id: str, pose: Pose | None, now: float,
               furniture=()) -> list[FallEvent]:
        cfg = self.cfg
        tr = self._t.setdefault(track_id, _T())
        events: list[FallEvent] = []

        if pose is None or pose.confidence < cfg.min_pose_conf:
            return self._no_pose(track_id, tr, now)

        tr.last_seen = now
        tr.pending_lost = False
        p = classify_posture(pose, self.pcfg)
        hip_y = p.hip_y
        tr.hist.append((now, p.label, hip_y))

        while tr.hist and now - tr.hist[0][0] > 30.0:
            tr.hist.popleft()

        torso = self._torso_len(pose)

        if p.label in UPRIGHT:
            if torso:
                tr.ref_len = torso if tr.ref_len is None else \
                    0.8 * tr.ref_len + 0.2 * torso

            tr.last_upright_t = now
            tr.last_upright_hip = hip_y
            tr.upright_since = tr.upright_since or now
        else:
            tr.upright_since = None

        lying = self._lying_confirmed(tr, now)

        if lying and tr.lying_since is None:
            tr.lying_since = self._lying_start(tr, now)
            tr.classified = False
            tr.on_furniture = self._on_furniture(pose, furniture)

        if not lying and p.label in UPRIGHT:
            events += self._recovery(track_id, tr, now)

            if tr.upright_since and now - tr.upright_since >= \
                    cfg.recover_upright_s:
                tr.lying_since = None
                tr.rapid = False
                tr.classified = False

                if tr.state in ("resting", "suspect", "asked", "alerted"):
                    tr.state = "normal"

        if lying:
            events += self._while_lying(track_id, tr, now)

        return events

    # ----- pieces -----

    @staticmethod
    def _torso_len(pose: Pose) -> float | None:
        from buddy_core.vision.pose import (
            L_HIP, L_SHOULDER, R_HIP, R_SHOULDER)
        import numpy as np
        sh = [pose.point(i) for i in (L_SHOULDER, R_SHOULDER)]
        hp = [pose.point(i) for i in (L_HIP, R_HIP)]
        sh = [s for s in sh if s is not None]
        hp = [h for h in hp if h is not None]

        if not sh or not hp:
            return None

        return float(np.hypot(*(np.mean(hp, 0) - np.mean(sh, 0))))

    def _lying_confirmed(self, tr: _T, now: float) -> bool:
        recent = [lab for t, lab, _ in tr.hist
                  if now - t <= self.cfg.lying_confirm_s]

        if len(recent) < 3:
            return False

        return sum(1 for lab in recent if lab == "lying") / len(recent) >= \
            self.cfg.lying_ratio

    @staticmethod
    def _lying_start(tr: _T, now: float) -> float:
        start = now

        for t, lab, _ in reversed(tr.hist):
            if lab == "lying":
                start = t
            elif lab in UPRIGHT:
                break

        return start

    def _classify_descent(self, tr: _T) -> None:
        """Was the last stretch of the drop fast?  (hips, in torso units)"""

        cfg = self.cfg
        tr.rapid, tr.drop, tr.transition_s = False, 0.0, None
        tr.classified = True
        ref = tr.ref_len

        if not ref:
            return

        series = [(t, h) for t, _, h in tr.hist
                  if h is not None and tr.lying_since is not None
                  and t >= tr.lying_since - cfg.history_s]
        end = [h for t, h in series if t >= tr.lying_since + 0.2]

        if not series or not end:
            return

        h_end = sorted(end)[len(end) // 2]
        tr.drop = (h_end - min(h for _, h in series)) / ref
        hi = h_end - cfg.min_drop_torsos * ref
        t_hi = None

        for t, h in series:
            if h <= hi:
                t_hi = t

        if t_hi is None:
            return

        t_floor = next((t for t, h in series
                        if t > t_hi and h >= h_end - 0.15 * ref), None)

        if t_floor is None:
            return

        tr.transition_s = t_floor - t_hi
        tr.rapid = tr.transition_s <= cfg.rapid_s

    def _while_lying(self, track_id: str, tr: _T, now: float
                     ) -> list[FallEvent]:
        cfg = self.cfg
        ev: list[FallEvent] = []
        lying_for = now - tr.lying_since

        if tr.state == "normal" and not tr.classified:
            if lying_for < cfg.settle_s:
                return ev

            self._classify_descent(tr)

        if tr.state == "normal":
            slow = tr.transition_s is not None and \
                tr.transition_s > cfg.rapid_s

            if not tr.rapid:
                tr.state = "resting"
                why = ["lowered_slowly" if slow else
                       "no_fast_drop_seen"]

                if tr.on_furniture:
                    why.append("on_furniture")

                ev.append(FallEvent("resting_lying", track_id, now, "info",
                                    tuple(why), details=self._d(tr)))

                return ev

            tr.state = "suspect"
            tr.suspect_at = now

        if tr.state == "suspect":
            wait = cfg.confirm_still_s + (
                cfg.furniture_slow_extra_s if tr.on_furniture else 0.0)

            if lying_for >= wait:
                tr.state = "asked"
                tr.asked_at = now
                why = ["rapid_descent", "still_lying"]

                if tr.on_furniture:
                    why.append("on_furniture_longer_wait")

                ev.append(FallEvent("fall_suspected", track_id, now, "check",
                                    tuple(why), True, self._d(tr)))
        elif tr.state == "asked":
            if now - tr.asked_at >= cfg.response_s:
                ev += self._alert(track_id, tr, now, "no_response")

        return ev

    def _alert(self, track_id: str, tr: _T, now: float, why: str
               ) -> list[FallEvent]:
        if now - tr.alerted_at < self.cfg.alert_cooldown_s:
            tr.state = "alerted"

            return []

        tr.state = "alerted"
        tr.alerted_at = now

        return [FallEvent("fall_alert", track_id, now, "alert",
                          ("rapid_descent", "still_lying", why), False,
                          self._d(tr))]

    def _recovery(self, track_id: str, tr: _T, now: float
                  ) -> list[FallEvent]:
        if tr.state in ("suspect", "asked", "alerted") and \
                tr.upright_since is not None and \
                now - tr.upright_since >= self.cfg.recover_upright_s:
            tr.state = "normal"

            return [FallEvent("fall_recovered", track_id, now, "info",
                              ("person_upright_again",))]

        return []

    def _no_pose(self, track_id: str, tr: _T, now: float
                 ) -> list[FallEvent]:
        """Person not (reliably) visible this frame."""

        cfg = self.cfg
        gone = now - tr.last_seen

        if tr.state in ("suspect", "asked") and tr.lying_since is not None:
            if gone <= cfg.lost_grace_s:
                return self._while_lying(track_id, tr, now)

            tr.state = "normal"
            tr.lying_since = None

            return [FallEvent("lost_after_descent", track_id, now, "info",
                              ("person_left_view_after_suspected_fall",),
                              details=self._d(tr))]

        if tr.state != "normal" or tr.pending_lost or tr.last_seen <= 0:
            return []

        if gone >= cfg.lost_after_s and self._descending(tr):
            tr.pending_lost = True

            return [FallEvent("lost_after_descent", track_id, now, "info",
                              ("person_vanished_while_going_down",))]

        return []

    @staticmethod
    def _descending(tr: _T) -> bool:
        pts = [(t, h) for t, _, h in tr.hist if h is not None]

        if len(pts) < 3 or not tr.ref_len:
            return False

        (t0, h0), (t1, h1) = pts[-3], pts[-1]

        return t1 > t0 and (h1 - h0) / tr.ref_len >= 0.5 and t1 - t0 <= 1.5

    def _d(self, tr: _T) -> dict:
        return {"transition_s": tr.transition_s, "drop_torsos": tr.drop,
                "on_furniture": tr.on_furniture, "rapid": tr.rapid}

    # ----- voice -----

    def on_voice(self, text: str, now: float) -> list[FallEvent]:
        """The person answered 'are you okay?' or called for help."""

        t = (text or "").lower()
        out: list[FallEvent] = []

        for tid, tr in self._t.items():
            if tr.state not in ("suspect", "asked"):
                continue

            if re.search(r"\b(help|call|ambulance|hurt|can'?t get up|"
                         r"cannot get up|fell)\b", t):
                out += self._alert(tid, tr, now, "asked_for_help")
            elif re.search(r"\b(i'?m (ok|okay|fine|alright|all right)|"
                           r"i am (ok|okay|fine)|no (need|problem)|"
                           r"don'?t (worry|call)|false alarm)\b", t):
                tr.state = "resting"     # stay quiet until they get up
                out.append(FallEvent("fall_recovered", tid, now, "info",
                                     ("person_said_they_are_ok",)))

        return out

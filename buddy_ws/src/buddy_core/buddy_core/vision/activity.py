"""Posture and simple activity from a pose skeleton.

Rule based and geometric on purpose: every decision can be explained and
checked against the skeleton.  The thresholds are starting values that
were not tuned on real recordings from the robot's camera.

Labels:  standing, sitting, lying, bending, upright_upper_body (legs not
visible, so standing and sitting cannot be told apart), unknown.

What it does not do: it does not say a person is *sleeping*, *eating*,
*ill* or *in danger*.  Lying is only a body orientation; whether it is a
fall depends on how it started (see ``fall``).
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from buddy_core.vision.pose import (
    L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, NOSE, R_ANKLE, R_HIP,
    R_KNEE, R_SHOULDER, R_WRIST, Pose)

LABELS = ("standing", "sitting", "lying", "bending",
          "upright_upper_body", "unknown")


@dataclass
class PostureConfig:
    min_vis: float = 0.4
    lying_torso_deg: float = 62.0          # torso this far from vertical
    lying_min_aspect: float = 1.0          # body box width / height
    bending_torso_deg: float = 30.0
    stand_thigh_drop: float = 0.60         # (knee_y - hip_y) / torso_len
    sit_thigh_drop: float = 0.35
    min_torso_px: float = 20.0


@dataclass(frozen=True)
class Posture:
    label: str
    confidence: float
    torso_deg: float | None
    aspect: float | None
    thigh_drop: float | None
    reason: str
    head_y: float | None = None
    hip_y: float | None = None


def _mid(pose: Pose, a: int, b: int, min_vis: float):
    pa, pb = pose.point(a, min_vis), pose.point(b, min_vis)

    if pa is not None and pb is not None:
        return (pa + pb) / 2

    return pa if pa is not None else pb


def classify_posture(pose: Pose, cfg: PostureConfig | None = None
                     ) -> Posture:
    cfg = cfg or PostureConfig()
    mv = cfg.min_vis
    sh = _mid(pose, L_SHOULDER, R_SHOULDER, mv)
    hip = _mid(pose, L_HIP, R_HIP, mv)
    bw = pose.box[2] - pose.box[0]
    bh = pose.box[3] - pose.box[1]
    aspect = bw / bh if bh > 1 else None

    if sh is None or hip is None:
        return Posture("unknown", 0.0, None, aspect, None,
                       "shoulders_or_hips_not_visible")

    t = hip - sh
    torso_len = float(np.hypot(*t))

    if torso_len < cfg.min_torso_px:
        return Posture("unknown", 0.0, None, aspect, None, "torso_too_small",
                       head_y=None, hip_y=float(hip[1]))

    # angle from straight down in the image (0 = upright, 90 = sideways)
    deg = math.degrees(math.atan2(abs(t[0]), t[1])) if t[1] >= 0 else \
        180.0 - math.degrees(math.atan2(abs(t[0]), -t[1]))
    nose = pose.point(NOSE, mv)
    head_y = float(nose[1]) if nose is not None else float(sh[1])
    vis = float(np.mean([pose.visibility[i] for i in
                         (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)]))

    if deg >= cfg.lying_torso_deg and (aspect or 0) >= cfg.lying_min_aspect:
        return Posture("lying", min(1.0, vis), deg, aspect, None,
                       "torso_horizontal_and_body_wide", head_y,
                       float(hip[1]))

    if deg >= cfg.lying_torso_deg:
        return Posture("bending", 0.5 * vis, deg, aspect, None,
                       "torso_sideways_but_body_tall", head_y, float(hip[1]))

    knees = [pose.point(i, mv) for i in (L_KNEE, R_KNEE)]
    ankles = [pose.point(i, mv) for i in (L_ANKLE, R_ANKLE)]
    knees = [k for k in knees if k is not None]
    n_ankles = sum(a is not None for a in ankles)

    if deg >= cfg.bending_torso_deg:
        return Posture("bending", 0.7 * vis, deg, aspect, None,
                       "torso_leaning", head_y, float(hip[1]))

    if not knees:
        return Posture("upright_upper_body", 0.6 * vis, deg, aspect, None,
                       "legs_not_visible", head_y, float(hip[1]))

    drop = float(np.mean([k[1] for k in knees]) - hip[1]) / torso_len

    if drop >= cfg.stand_thigh_drop and n_ankles >= 1:
        return Posture("standing", vis, deg, aspect, drop,
                       "thigh_long_and_down", head_y, float(hip[1]))

    if drop <= cfg.sit_thigh_drop:
        return Posture("sitting", 0.8 * vis, deg, aspect, drop,
                       "knees_near_hip_height", head_y, float(hip[1]))

    return Posture("unknown", 0.3 * vis, deg, aspect, drop,
                   "between_sitting_and_standing", head_y, float(hip[1]))


@dataclass
class ActivityConfig:
    window_s: float = 1.5
    min_votes: int = 3
    min_confidence: float = 0.4
    min_dwell_s: float = 1.0
    still_speed: float = 0.05        # body-lengths per second
    moving_speed: float = 0.25
    hand_raise_hold_s: float = 0.6


@dataclass(frozen=True)
class Activity:
    posture: str                 # smoothed
    raw_posture: str
    motion: str                  # still | moving | unknown
    speed: float | None          # torso lengths per second
    hand_raised: bool
    posture_s: float             # time in the current posture
    still_s: float               # time without notable movement
    confidence: float


@dataclass
class _Track:
    votes: deque = field(default_factory=deque)
    posture: str = "unknown"
    since: float = 0.0
    pending: str | None = None
    pending_since: float = 0.0
    last_centre: np.ndarray | None = None
    last_t: float | None = None
    speed_hist: deque = field(default_factory=deque)
    still_since: float | None = None
    hand_since: float | None = None


class ActivityTracker:
    def __init__(self, cfg: ActivityConfig | None = None,
                 pcfg: PostureConfig | None = None) -> None:
        self.cfg = cfg or ActivityConfig()
        self.pcfg = pcfg or PostureConfig()
        self._tr: dict[str, _Track] = {}

    def forget(self, track_id: str) -> None:
        self._tr.pop(track_id, None)

    def update(self, track_id: str, pose: Pose | None, now: float
               ) -> Activity:
        cfg = self.cfg
        tr = self._tr.setdefault(track_id, _Track(since=now))

        if pose is None:
            return Activity(tr.posture, "unknown", "unknown", None, False,
                            now - tr.since, 0.0, 0.0)

        p = classify_posture(pose, self.pcfg)
        tr.votes.append((now, p.label, p.confidence))

        while tr.votes and now - tr.votes[0][0] > cfg.window_s:
            tr.votes.popleft()

        good = [lab for _, lab, c in tr.votes
                if lab != "unknown" and c >= cfg.min_confidence]
        stable = tr.posture

        if len(good) >= cfg.min_votes:
            top, n = Counter(good).most_common(1)[0]

            if n > len(good) / 2:
                if top == tr.posture:
                    tr.pending = None
                else:
                    if tr.pending != top:
                        tr.pending, tr.pending_since = top, now

                    if now - tr.pending_since >= cfg.min_dwell_s:
                        tr.posture, tr.since, tr.pending = top, now, None

                stable = tr.posture
        elif not good and now - tr.votes[0][0] >= cfg.window_s * .9:
            stable = tr.posture

        # motion in torso lengths per second (scale free)
        sh = _mid(pose, L_SHOULDER, R_SHOULDER, self.pcfg.min_vis)
        hip = _mid(pose, L_HIP, R_HIP, self.pcfg.min_vis)
        speed = None

        if sh is not None and hip is not None:
            tl = max(float(np.hypot(*(hip - sh))), self.pcfg.min_torso_px)
            c = (sh + hip) / 2

            if tr.last_centre is not None and tr.last_t is not None and \
                    now > tr.last_t:
                speed = float(np.hypot(*(c - tr.last_centre))) / tl / \
                    (now - tr.last_t)

            tr.last_centre, tr.last_t = c, now

        if speed is not None:
            tr.speed_hist.append((now, speed))

        while tr.speed_hist and now - tr.speed_hist[0][0] > cfg.window_s:
            tr.speed_hist.popleft()

        avg = float(np.mean([s for _, s in tr.speed_hist])) \
            if tr.speed_hist else None

        if avg is None:
            motion = "unknown"
        elif avg >= cfg.moving_speed:
            motion = "moving"
            tr.still_since = None
        elif avg <= cfg.still_speed:
            motion = "still"
            tr.still_since = tr.still_since if tr.still_since is not None \
                else now
        else:
            motion = "moving"
            tr.still_since = None

        raised = False
        lw, rw = pose.point(L_WRIST, self.pcfg.min_vis), \
            pose.point(R_WRIST, self.pcfg.min_vis)

        if sh is not None and ((lw is not None and lw[1] < sh[1] - 10) or
                               (rw is not None and rw[1] < sh[1] - 10)):
            tr.hand_since = tr.hand_since if tr.hand_since is not None \
                else now
            raised = now - tr.hand_since >= cfg.hand_raise_hold_s
        else:
            tr.hand_since = None

        return Activity(stable, p.label, motion, avg, raised,
                        now - tr.since,
                        (now - tr.still_since) if tr.still_since is not None
                        else 0.0, p.confidence)

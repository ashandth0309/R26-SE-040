"""Hand following with an explicit activation gate.

HONEST SCOPE.  Sobiya's vision code provides BODY pose (wrists) and
whole-arm gestures only.  It has no hand-landmark or index-fingertip
detection; the palm model is staged but not integrated.  So:

* ``index_fingertip`` points are used when a real fingertip model fills
  ``HandObservation.fingertips`` (none exists today);
* otherwise the WRIST is followed and the result is labelled
  ``wrist_fallback``.  A wrist is not a fingertip: it can be 20-30 cm
  from the pointing finger, and this fallback must not be described as
  finger tracking.

The follower is inactive until explicitly activated (a command, or a
configured gesture), follows ONE hand of ONE person, and deactivates on
a stop gesture, a command, hand loss, or ``max_active_s``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from buddy_core.head_tracking.config import HandConfig
from buddy_core.head_tracking.face_follow import PoseAt
from buddy_core.head_tracking.servo import VisualServo2D
from buddy_core.head_tracking.models import HandObservation, HandPoint

KIND_FINGERTIP = "index_fingertip"
KIND_WRIST = "wrist_fallback"


class HandStatus(str, Enum):
    INACTIVE = "INACTIVE"
    TRACKING = "TRACKING"
    HOLDING = "HOLDING"
    LOST = "LOST"
    STALE = "STALE"


@dataclass
class HandResult:
    status: HandStatus
    kind: str | None = None        # KIND_FINGERTIP | KIND_WRIST
    side: str | None = None
    pan: float | None = None
    tilt: float | None = None
    reason: str = ""


class HandFollower:
    def __init__(self, cfg: HandConfig | None = None, pan_sign: int = 1,
                 tilt_sign: int = 1) -> None:
        self.cfg = cfg or HandConfig()
        self.servo = VisualServo2D(self.cfg.servo, pan_sign, tilt_sign)
        self.active = False
        self.side: str | None = None
        self.owner: str | None = None
        self.kind: str | None = None
        self._activated_t = 0.0
        self._last_seen: float | None = None
        self._last_frame_t: float | None = None

    def activate(self, t: float, side: str | None = None,
                 owner: str | None = None) -> bool:
        """Explicit activation.  ``side`` overrides config ``hand_side``
        only when config is ``auto``."""

        cfg_side = self.cfg.hand_side
        chosen = cfg_side if cfg_side != "auto" else side
        if chosen not in (None, "left", "right"):
            return False
        self.active = True
        self.side = chosen
        self.owner = owner
        self.kind = None
        self._activated_t = t
        self._last_seen = t
        self._last_frame_t = None
        self.servo.reset()
        return True

    def deactivate(self) -> None:
        self.active = False
        self.side = self.owner = self.kind = None
        self._last_seen = None
        self.servo.reset()

    def _pick(self, obs: HandObservation):
        c = self.cfg
        sides = [self.side] if self.side else ["left", "right"]
        best: tuple[float, str, str, HandPoint] | None = None
        for s in sides:
            tip = obs.fingertips.get(s)
            if c.prefer_fingertip and tip and tip.confidence >= \
                    c.min_confidence:
                cand = (tip.confidence + 1.0, s, KIND_FINGERTIP, tip)
            else:
                wr = obs.wrists.get(s)
                if (c.allow_wrist_fallback and wr
                        and wr.confidence >= c.min_confidence):
                    cand = (wr.confidence, s, KIND_WRIST, wr)
                else:
                    continue
            if best is None or cand[0] > best[0]:
                best = cand
        return best

    def update(self, obs: HandObservation, now: float,
               pose_at: PoseAt) -> HandResult:
        c = self.cfg
        if not self.active:
            return HandResult(HandStatus.INACTIVE)
        if now - self._activated_t > c.max_active_s:
            self.deactivate()
            return HandResult(HandStatus.LOST, reason="max_active_s")
        if (now - obs.t > c.max_obs_age_s
                or (self._last_frame_t is not None
                    and obs.t <= self._last_frame_t)):
            return HandResult(HandStatus.STALE)
        if self.owner and obs.track_id and obs.track_id != self.owner:
            return HandResult(HandStatus.STALE, reason="other person")
        dt = 0.05 if self._last_frame_t is None else obs.t - \
            self._last_frame_t
        self._last_frame_t = obs.t

        pick = self._pick(obs)
        if pick is None:
            return self._missing(now)
        _, side, kind, pt = pick
        if self.side is None:
            self.side = side            # auto: stay on this hand
        self.kind = kind
        self._last_seen = obs.t
        out = self.servo.step(pt.x, pt.y, obs.width, obs.height, dt)
        pan0, tilt0 = pose_at(obs.t)
        return HandResult(HandStatus.TRACKING, kind, side,
                          pan0 + out.d_pan_deg, tilt0 + out.d_tilt_deg)

    def _missing(self, now: float) -> HandResult:
        if self._last_seen is not None and (
                now - self._last_seen <= self.cfg.lost_hold_s):
            return HandResult(HandStatus.HOLDING, self.kind, self.side)
        self.deactivate()
        return HandResult(HandStatus.LOST, reason="hand lost")

    def check_timeout(self, now: float) -> HandResult | None:
        if not self.active:
            return None
        if now - self._activated_t > self.cfg.max_active_s:
            self.deactivate()
            return HandResult(HandStatus.LOST, reason="max_active_s")
        if self._last_seen is not None and (
                now - self._last_seen > self.cfg.lost_hold_s):
            self.deactivate()
            return HandResult(HandStatus.LOST, reason="no observations")
        return None

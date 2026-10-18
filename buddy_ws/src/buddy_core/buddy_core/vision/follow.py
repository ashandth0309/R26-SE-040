"""Following a person (Phase 16).  DISABLED on hardware by default.

Following means a moving robot keeps close to a walking person.  With one
monocular camera there is no reliable distance, so this module can only
supply *who* to follow and *which direction* they are in.  It never sends
velocities.  A request goes to the navigation adapter, and only if

* the configuration enables it (``hardware_enabled``, default False),
* navigation reports that following has been validated on this robot
  (``follow_validated``) together with reliable distance, localisation,
  obstacle avoidance and permission from the safety supervisor,
* the person asking is a recognised, consenting person.

Until all of that is true every request is refused with a reason.  The
logic below (identity lock, lost-target stop, wrong-person guard) is
tested against the mock adapter only.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from buddy_core.vision.approach import (
    ApproachStatus, NavigationAdapter)

PI_CAMERA_V2_HFOV_DEG = 62.2


def bearing_from_box(box, frame_width: float,
                     hfov_deg: float = PI_CAMERA_V2_HFOV_DEG) -> float:
    """Direction of the box centre: 0 = straight ahead, + = to the right."""

    cx = (box[0] + box[2]) / 2
    return (cx / frame_width - 0.5) * hfov_deg


@dataclass(frozen=True)
class FollowGoal:
    request_id: str
    person_key: str
    desired_distance_m: float
    timeout_s: float


@dataclass(frozen=True)
class TargetObs:
    track_id: str
    person_key: str | None        # persistent id if currently recognised
    box: tuple[float, float, float, float]
    confidence: float


@dataclass
class FollowConfig:
    hardware_enabled: bool = False
    min_desired_distance_m: float = 1.0
    default_desired_distance_m: float = 1.5
    max_duration_s: float = 600.0
    lost_stop_s: float = 3.0
    min_confidence: float = 0.5
    too_close_frac: float = 0.55        # box height / frame height
    min_gap_s: float = 10.0


@dataclass(frozen=True)
class FollowFeedback:
    state: str
    target_visible: bool
    confidence: float
    bearing_deg: float | None
    too_close: bool
    reason: str = ""


@dataclass
class FollowRequest:
    now: float
    requester_key: str | None
    requester_track: str | None       # the track that said "follow me"
    recognized: bool
    consent: bool                     # person agreed to being followed
    desired_distance_m: float | None = None
    quiet_mode: bool = False
    emergency: bool = False


@dataclass(frozen=True)
class FollowDecision:
    allowed: bool
    reason: str
    goal: FollowGoal | None = None


class FollowController:
    def __init__(self, adapter: NavigationAdapter,
                 cfg: FollowConfig | None = None) -> None:
        self.adapter = adapter
        self.cfg = cfg or FollowConfig()
        self._ids = itertools.count(1)
        self.goal: FollowGoal | None = None
        self.target_track: str | None = None
        self._started = 0.0
        self._last_seen = 0.0
        self._last_request = -1e12
        self.log: list[tuple[float, str, str]] = []

    def _note(self, now, what, why) -> None:
        self.log.append((now, what, why))
        del self.log[:-200]

    def request(self, r: FollowRequest) -> FollowDecision:
        why = self._refusal(r)

        if why:
            self._note(r.now, "refused", why)

            return FollowDecision(False, why)

        goal = FollowGoal(
            f"follow_{next(self._ids)}", r.requester_key,
            max(self.cfg.min_desired_distance_m,
                r.desired_distance_m or self.cfg.default_desired_distance_m),
            self.cfg.max_duration_s)

        if not self.adapter.send(goal):          # type: ignore[arg-type]
            self._note(r.now, "refused", "navigation_rejected")

            return FollowDecision(False, "navigation_rejected")

        self.goal, self.target_track = goal, r.requester_track
        self._started = self._last_seen = self._last_request = r.now
        self._note(r.now, "sent", goal.request_id)

        return FollowDecision(True, "sent", goal)

    def _refusal(self, r: FollowRequest) -> str | None:
        cfg = self.cfg

        if not cfg.hardware_enabled:
            return "follow_disabled_on_hardware"

        if self.goal is not None:
            return "already_following"

        if r.emergency:
            return "emergency_in_progress"

        if r.requester_key is None or not r.recognized or \
                r.requester_track is None:
            return "person_not_recognised"

        if not r.consent:
            return "no_consent_to_be_followed"

        if r.quiet_mode:
            return "quiet_mode"

        if r.now - self._last_request < cfg.min_gap_s:
            return "too_soon"

        c = self.adapter.capabilities()

        if not c.available:
            return "navigation_unavailable"

        if c.emergency_stop_active or not c.movement_allowed:
            return "safety_not_allowing_movement"

        for ok, why in ((c.follow_validated, "follow_not_validated"),
                        (c.distance_reliable, "distance_not_reliable"),
                        (c.localization_ok, "localization_not_reliable"),
                        (c.obstacle_avoidance_ok,
                         "obstacle_avoidance_not_ready")):
            if not ok:
                return why

        return None

    def update(self, now: float, observations: list[TargetObs],
               frame_shape, voice_stop: bool = False,
               quiet_mode: bool = False, emergency: bool = False
               ) -> FollowFeedback:
        g = self.goal

        if g is None:
            return FollowFeedback("idle", False, 0.0, None, False)

        cfg = self.cfg
        h, w = frame_shape[:2]
        target = self._pick(g, observations)
        too_close = False
        bearing = None
        conf = 0.0

        if target is not None:
            self._last_seen = now
            self.target_track = target.track_id
            bearing = bearing_from_box(target.box, w)
            conf = target.confidence
            too_close = (target.box[3] - target.box[1]) / h >= \
                cfg.too_close_frac

        reason = None

        if voice_stop:
            reason = "voice_stop"
        elif emergency:
            reason = "emergency_in_progress"
        elif quiet_mode:
            reason = "quiet_mode"
        elif now - self._started > g.timeout_s:
            reason = "max_duration"
        elif now - self._last_seen > cfg.lost_stop_s:
            reason = "target_lost"
        elif self.adapter.status(g.request_id) in (
                ApproachStatus.FAILED, ApproachStatus.CANCELLED,
                ApproachStatus.SUCCEEDED):
            reason = "navigation_ended"
        else:
            c = self.adapter.capabilities()

            if c.emergency_stop_active or not c.movement_allowed:
                reason = "safety_not_allowing_movement"
            elif not (c.available and c.follow_validated and
                      c.distance_reliable and c.localization_ok):
                reason = "navigation_degraded"

        if reason:
            self.stop(now, reason)

            return FollowFeedback("stopped", target is not None, conf,
                                  bearing, too_close, reason)

        return FollowFeedback("following", target is not None, conf,
                              bearing, too_close)

    def _pick(self, g: FollowGoal, obs: list[TargetObs]):
        """Keep the same person.  Never switch to someone else."""

        cfg = self.cfg
        same_track = [o for o in obs if o.track_id == self.target_track]
        same_person = [o for o in obs if o.person_key == g.person_key]
        pool = same_person or same_track

        if not pool:
            return None

        best = max(pool, key=lambda o: o.confidence)

        if best.person_key not in (None, g.person_key):
            return None

        return best if best.confidence >= cfg.min_confidence else None

    def stop(self, now: float, reason: str) -> None:
        if self.goal is not None:
            self.adapter.cancel(self.goal.request_id, reason)
            self._note(now, "stopped", reason)

        self.goal = None
        self.target_track = None

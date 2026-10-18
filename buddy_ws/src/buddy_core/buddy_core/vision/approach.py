"""Asking navigation to approach a person (Phase 9).

Vision never drives motors.  This module only decides whether a request
may be *sent* to the navigation stack and watches the request afterwards.
The goal it sends mirrors the existing ``ApproachCaller`` action (a
bearing and a confidence, no velocities, no path), and navigation plus the
safety supervisor stay in charge of how, and whether, the robot moves.

Only a mock adapter exists.  A single Pi camera cannot measure distance
reliably, so with the current hardware the real adapter would report
``distance_reliable=False`` and every request is refused.  That is the
intended behaviour until navigation is validated.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class ApproachStatus(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"
    FAILED = "failed"
    REFUSED = "refused"


@dataclass(frozen=True)
class NavCapabilities:
    """What the navigation stack itself says it can do right now."""

    available: bool = False
    distance_reliable: bool = False
    localization_ok: bool = False
    obstacle_avoidance_ok: bool = False
    movement_allowed: bool = False       # safety supervisor
    emergency_stop_active: bool = True
    follow_validated: bool = False       # following tested on this robot


@dataclass(frozen=True)
class ApproachGoal:
    request_id: str
    person_key: str
    bearing_deg: float
    bearing_confidence: float
    stop_distance_m: float
    timeout_s: float


class NavigationAdapter(Protocol):
    def capabilities(self) -> NavCapabilities: ...

    def send(self, goal: ApproachGoal) -> bool: ...

    def cancel(self, request_id: str, reason: str) -> None: ...

    def status(self, request_id: str) -> ApproachStatus: ...


class MockNavigationAdapter:
    """Records goals and lets a test script the outcome."""

    def __init__(self, caps: NavCapabilities | None = None) -> None:
        self.caps = caps or NavCapabilities()
        self.sent: list[ApproachGoal] = []
        self.cancelled: list[tuple[str, str]] = []
        self._status: dict[str, ApproachStatus] = {}
        self.accept = True

    def capabilities(self) -> NavCapabilities:
        return self.caps

    def send(self, goal: ApproachGoal) -> bool:
        if not self.accept:
            return False

        self.sent.append(goal)
        self._status[goal.request_id] = ApproachStatus.ACTIVE

        return True

    def cancel(self, request_id: str, reason: str) -> None:
        self.cancelled.append((request_id, reason))
        self._status[request_id] = ApproachStatus.CANCELLED

    def status(self, request_id: str) -> ApproachStatus:
        return self._status.get(request_id, ApproachStatus.IDLE)

    def finish(self, request_id: str, ok: bool = True) -> None:
        self._status[request_id] = (ApproachStatus.SUCCEEDED if ok
                                    else ApproachStatus.FAILED)


@dataclass
class ApproachConfig:
    min_stop_distance_m: float = 1.0       # never ask to go closer
    default_stop_distance_m: float = 1.2
    min_bearing_confidence: float = 0.6
    max_target_age_s: float = 1.0
    lost_cancel_s: float = 2.0
    timeout_s: float = 30.0
    min_gap_s: float = 60.0                # between requests


@dataclass
class ApproachRequest:
    """What the caller knows when asking."""

    now: float
    person_key: str | None
    recognized: bool
    invited: bool                    # the person asked Buddy to come
    bearing_deg: float | None
    bearing_confidence: float
    target_stamp: float
    stop_distance_m: float | None = None
    quiet_mode: bool = False
    person_settings: dict = field(default_factory=dict)
    emergency: bool = False          # fall response etc. (not an approach)


@dataclass(frozen=True)
class ApproachDecision:
    allowed: bool
    reason: str
    goal: ApproachGoal | None = None


class ApproachController:
    def __init__(self, adapter: NavigationAdapter,
                 cfg: ApproachConfig | None = None) -> None:
        self.adapter = adapter
        self.cfg = cfg or ApproachConfig()
        self._ids = itertools.count(1)
        self.active: ApproachGoal | None = None
        self._started = 0.0
        self._last_seen = 0.0
        self._last_request = -1e12
        self.log: list[tuple[float, str, str]] = []

    def _note(self, now: float, what: str, why: str) -> None:
        self.log.append((now, what, why))
        del self.log[:-200]

    def request(self, r: ApproachRequest) -> ApproachDecision:
        cfg = self.cfg
        why = self._refusal(r)

        if why:
            self._note(r.now, "refused", why)

            return ApproachDecision(False, why)

        stop = max(cfg.min_stop_distance_m,
                   r.stop_distance_m or cfg.default_stop_distance_m)
        goal = ApproachGoal(f"approach_{next(self._ids)}", r.person_key,
                            float(r.bearing_deg), r.bearing_confidence,
                            stop, cfg.timeout_s)

        if not self.adapter.send(goal):
            self._note(r.now, "refused", "navigation_rejected")

            return ApproachDecision(False, "navigation_rejected")

        self.active = goal
        self._started = self._last_seen = self._last_request = r.now
        self._note(r.now, "sent", goal.request_id)

        return ApproachDecision(True, "sent", goal)

    def _refusal(self, r: ApproachRequest) -> str | None:
        cfg = self.cfg

        if self.active is not None:
            return "already_approaching"

        if r.emergency:
            return "emergency_in_progress"

        if r.person_key is None or not r.recognized:
            return "person_not_recognised"

        if not r.invited:
            return "no_invitation"

        if r.quiet_mode or r.person_settings.get("quiet"):
            return "quiet_mode"

        if r.person_settings.get("approach", True) is False:
            return "person_disabled_approach"

        if r.now - self._last_request < cfg.min_gap_s:
            return "too_soon"

        if r.bearing_deg is None or \
                r.bearing_confidence < cfg.min_bearing_confidence:
            return "bearing_unreliable"

        if r.now - r.target_stamp > cfg.max_target_age_s:
            return "target_stale"

        c = self.adapter.capabilities()

        if not c.available:
            return "navigation_unavailable"

        if c.emergency_stop_active or not c.movement_allowed:
            return "safety_not_allowing_movement"

        if not c.distance_reliable:
            return "distance_not_reliable"

        if not c.localization_ok:
            return "localization_not_reliable"

        if not c.obstacle_avoidance_ok:
            return "obstacle_avoidance_not_ready"

        return None

    def update(self, now: float, person_visible: bool,
               quiet_mode: bool = False, person_declined: bool = False,
               voice_stop: bool = False, emergency: bool = False) -> str:
        """Call regularly while a request is active.  Returns the state."""

        g = self.active

        if g is None:
            return ApproachStatus.IDLE.value

        st = self.adapter.status(g.request_id)

        if st in (ApproachStatus.SUCCEEDED, ApproachStatus.FAILED,
                  ApproachStatus.CANCELLED):
            self._finish(now, st.value)

            return st.value

        if person_visible:
            self._last_seen = now

        reason = None

        if voice_stop:
            reason = "voice_stop"
        elif person_declined:
            reason = "person_declined"
        elif emergency:
            reason = "emergency_in_progress"
        elif quiet_mode:
            reason = "quiet_mode"
        elif now - self._started > g.timeout_s:
            reason = "timeout"
        elif now - self._last_seen > self.cfg.lost_cancel_s:
            reason = "person_lost"
        else:
            c = self.adapter.capabilities()

            if c.emergency_stop_active or not c.movement_allowed:
                reason = "safety_not_allowing_movement"
            elif not (c.available and c.distance_reliable
                      and c.localization_ok):
                reason = "navigation_degraded"

        if reason:
            self.cancel(now, reason)

            return ApproachStatus.CANCELLED.value

        return ApproachStatus.ACTIVE.value

    def cancel(self, now: float, reason: str) -> None:
        g = self.active

        if g is not None:
            self.adapter.cancel(g.request_id, reason)
            self._note(now, "cancelled", reason)

        self.active = None

    def _finish(self, now: float, what: str) -> None:
        self._note(now, what, self.active.request_id if self.active else "")
        self.active = None

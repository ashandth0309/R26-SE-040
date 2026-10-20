"""Safe pan/tilt motion coordinator.

The only code in head tracking that talks to the neck.  It sits on top of
the existing ``NeckController`` (which keeps clamping to the calibrated
limits) and adds:

* priority arbitration - several sources may submit targets, but only the
  highest-priority unexpired one is followed, so the neck never receives
  conflicting commands;
* velocity and acceleration (slew) limiting with a braking profile, so a
  target is approached without overshoot;
* dead zone and minimum write step, so tiny jitter never reaches a servo;
* an inward limit margin on top of the calibrated limits;
* validation of every command (NaN, inf, bool, strings are rejected and
  cause no motion);
* latched emergency stop, user stop and hardware-fault states;
* target expiry, a neutral-pose watchdog, and safe shutdown.

It never commands locomotion and does not import anything from
``buddy_core.motors``.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from buddy_core.head_tracking.config import MotionConfig
from buddy_core.head_tracking.models import (
    TARGET_SOURCES,
    EventKind,
    HeadEvent,
    Priority,
    is_number,
)
from buddy_core.neck.base import BuddyNeckError
from buddy_core.neck.controller import NeckController

EventSink = Callable[[HeadEvent], None]


@dataclass
class _Request:
    pan: float
    tilt: float
    t: float
    expires: float
    speed_scale: float


def _move_toward(value: float, goal: float, step: float) -> float:
    if abs(goal - value) <= step:
        return goal
    return value + math.copysign(step, goal - value)


class HeadMotionCoordinator:
    """Single owner of the neck.  Not thread-safe; call from one loop."""

    def __init__(
        self,
        neck: NeckController,
        cfg: MotionConfig | None = None,
        on_event: EventSink | None = None,
        pan_sign: int = 1,
        tilt_sign: int = 1,
    ) -> None:
        self.neck = neck
        self.cfg = cfg or MotionConfig()
        self.pan_sign = pan_sign
        self.tilt_sign = tilt_sign
        self._on_event = on_event

        m = self.cfg.limit_margin_deg
        self.pan_lo, self.pan_hi = self._range(
            neck.pan_min, neck.pan_max, m)
        self.tilt_lo, self.tilt_hi = self._range(
            neck.tilt_min, neck.tilt_max, m)
        self.pan_neutral = min(max(neck.pan_center, self.pan_lo),
                               self.pan_hi)
        self.tilt_neutral = min(max(neck.tilt_center, self.tilt_lo),
                                self.tilt_hi)

        self.pan = self.pan_neutral
        self.tilt = self.tilt_neutral
        self._vp = 0.0
        self._vt = 0.0
        self._sent: tuple[float, float] | None = None
        self._last_t: float | None = None
        self._last_request_t: float | None = None
        self._requests: dict[Priority, _Request] = {}
        self._history: deque[tuple[float, float, float]] = deque()

        self.started = False
        self.estop = False
        self.user_stopped = False
        self.faulted = False
        self.shut_down = False
        self.fault_reason = ""

        self.rejected_commands = 0
        self.suppressed_conflicts = 0
        self.limit_hits = 0
        self.writes = 0

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _range(lo: float, hi: float, margin: float) -> tuple[float, float]:
        if hi - lo > 2 * margin:
            return lo + margin, hi - margin
        mid = (lo + hi) / 2.0       # travel too small for a margin
        return mid, mid

    def _emit(self, kind: EventKind, t: float, /, **data) -> None:
        if self._on_event is not None:
            self._on_event(HeadEvent(kind, t, data))

    @property
    def blocked(self) -> bool:
        return (self.estop or self.user_stopped or self.faulted
                or self.shut_down)

    def _clamp(self, pan: float, tilt: float) -> tuple[float, float]:
        cp = min(max(pan, self.pan_lo), self.pan_hi)
        ct = min(max(tilt, self.tilt_lo), self.tilt_hi)
        if cp != pan or ct != tilt:
            self.limit_hits += 1
        return cp, ct

    # ------------------------------------------------------------ startup
    def start(self, t: float | None = None) -> bool:
        """Move to the neutral pose.  Servos have no position feedback, so
        this first command is an unramped step from wherever the head is.
        """

        t = time.monotonic() if t is None else t
        if self.blocked:
            return False
        try:
            self.pan, self.tilt = self.neck.look(
                self.pan_neutral, self.tilt_neutral)
        except BuddyNeckError as exc:
            self._fault(t, f"start failed: {exc}")
            return False
        self._sent = (self.pan, self.tilt)
        self._last_t = t
        self._history.clear()
        self._history.append((t, self.pan, self.tilt))
        self.started = True
        return True

    # ----------------------------------------------------------- commands
    def submit(
        self,
        source: Priority,
        pan: float,
        tilt: float,
        t: float,
        speed_scale: float = 1.0,
    ) -> bool:
        """Offer a target.  True if accepted into the arbiter."""

        if source not in TARGET_SOURCES:
            return self._reject(t, source, "not a target source")
        if not (is_number(pan) and is_number(tilt) and is_number(t)):
            return self._reject(t if is_number(t) else 0.0, source,
                                "non-finite or non-numeric target")
        if not is_number(speed_scale) or not 0 < speed_scale <= 1:
            return self._reject(t, source, "speed_scale must be in (0, 1]")
        if self.blocked or not self.started:
            return self._reject(t, source, "head motion is blocked")

        active = self._active_source(t)
        if active is not None and source > active:
            self.suppressed_conflicts += 1      # a higher priority is live
        cp, ct = self._clamp(float(pan), float(tilt))
        self._requests[source] = _Request(
            cp, ct, t, t + self.cfg.request_ttl_s, float(speed_scale))
        self._last_request_t = t
        return True

    def _reject(self, t: float, source, reason: str) -> bool:
        self.rejected_commands += 1
        self._emit(EventKind.COMMAND_REJECTED, t,
                   source=str(source), reason=reason)
        return False

    def release_source(self, source: Priority) -> None:
        self._requests.pop(source, None)

    def active_source(self, t: float) -> Priority | None:
        return self._active_source(t)

    def _active_source(self, t: float) -> Priority | None:
        live = [p for p, r in self._requests.items() if r.expires > t]
        return min(live) if live else None

    # ------------------------------------------------------ safety latches
    def emergency_stop(self, reason: str = "", t: float | None = None):
        t = time.monotonic() if t is None else t
        if not self.estop:
            self.estop = True
            self._emit(EventKind.EMERGENCY_STOP, t, reason=reason)
        self._requests.clear()
        self._vp = self._vt = 0.0
        if self.cfg.release_on_emergency_stop:
            self._safe_release()

    def reset_emergency(self, t: float | None = None) -> None:
        t = time.monotonic() if t is None else t
        if self.estop:
            self.estop = False
            self._last_t = t
            self._emit(EventKind.EMERGENCY_STOP_CLEARED, t)

    def user_stop(self, t: float | None = None) -> None:
        t = time.monotonic() if t is None else t
        if not self.user_stopped:
            self.user_stopped = True
            self._emit(EventKind.USER_STOP, t)
        self._requests.clear()
        self._vp = self._vt = 0.0

    def resume(self, t: float | None = None) -> None:
        t = time.monotonic() if t is None else t
        if self.user_stopped:
            self.user_stopped = False
            self._last_t = t
            self._emit(EventKind.USER_RESUME, t)

    def reset_fault(self, t: float | None = None) -> bool:
        """Operator action after checking the hardware; re-centres."""

        t = time.monotonic() if t is None else t
        if not self.faulted:
            return True
        self.faulted = False
        self.fault_reason = ""
        self.started = False
        return self.start(t)

    def _fault(self, t: float, reason: str) -> None:
        self.faulted = True
        self.fault_reason = reason
        self._requests.clear()
        self._vp = self._vt = 0.0
        self._emit(EventKind.FAULT, t, reason=reason)
        if self.cfg.release_on_fault:
            self._safe_release()

    def _safe_release(self) -> None:
        try:
            self.neck.release()
        except Exception:       # a dead bus must not raise from a safety path
            pass

    # ---------------------------------------------------------- main step
    def update(self, t: float) -> tuple[float, float]:
        """Advance the motion profile to time ``t``; returns the pose."""

        if not is_number(t):
            return self.pan, self.tilt
        if self.blocked or not self.started:
            self._last_t = t
            return self.pan, self.tilt
        if self._last_t is None:
            self._last_t = t
        dt = min(max(t - self._last_t, 0.0), self.cfg.max_dt_s)
        self._last_t = t
        if dt <= 0:
            return self.pan, self.tilt

        goal_pan, goal_tilt, scale = self._goal(t)
        new_pan, self._vp = self._axis(
            self.pan, goal_pan, self._vp, dt,
            self.cfg.max_velocity_pan_dps * scale)
        new_tilt, self._vt = self._axis(
            self.tilt, goal_tilt, self._vt, dt,
            self.cfg.max_velocity_tilt_dps * scale)
        new_pan, new_tilt = self._clamp(new_pan, new_tilt)
        self._write(t, new_pan, new_tilt, goal_pan, goal_tilt)
        return self.pan, self.tilt

    def _goal(self, t: float) -> tuple[float, float, float]:
        src = self._active_source(t)
        if src is not None:
            r = self._requests[src]
            return r.pan, r.tilt, r.speed_scale
        last = self._last_request_t
        if last is not None and t - last > self.cfg.neutral_after_s:
            return self.pan_neutral, self.tilt_neutral, 0.5
        return self.pan, self.tilt, 1.0      # hold

    def _axis(self, pos: float, goal: float, vel: float, dt: float,
              vmax: float) -> tuple[float, float]:
        err = goal - pos
        # The dead zone only stops a move from STARTING; a move already in
        # progress keeps its braking profile and ends exactly on the goal.
        if abs(err) < self.cfg.dead_zone_deg and abs(vel) < 1e-6:
            err = 0.0
        a = self.cfg.max_accel_dps2
        # brake with half the acceleration limit so the discrete-time
        # velocity can actually follow the braking curve
        want = math.copysign(
            min(vmax, math.sqrt(2 * 0.5 * a * abs(err))), err)
        vel = _move_toward(vel, want, a * dt)
        if err == 0.0 and abs(vel) < 1e-9:
            return pos, 0.0
        new = pos + vel * dt
        if (goal - pos) * (goal - new) < 0:       # would cross the goal
            new, vel = goal, 0.0                  # stop on it, no overshoot
        return new, vel

    def _write(self, t, new_pan, new_tilt, goal_pan, goal_tilt) -> None:
        sent = self._sent or (self.pan, self.tilt)
        step = self.cfg.min_step_deg
        settled = (abs(goal_pan - new_pan) < 1e-6
                   and abs(goal_tilt - new_tilt) < 1e-6)
        try:
            if abs(new_pan - sent[0]) >= step or (
                    settled and new_pan != sent[0]):
                new_pan = self.neck.set_pan(new_pan)
                sent = (new_pan, sent[1])
                self.writes += 1
            if abs(new_tilt - sent[1]) >= step or (
                    settled and new_tilt != sent[1]):
                new_tilt = self.neck.set_tilt(new_tilt)
                sent = (sent[0], new_tilt)
                self.writes += 1
        except BuddyNeckError as exc:
            self._fault(t, f"neck write failed: {exc}")
            return
        self._sent = sent
        self.pan, self.tilt = new_pan, new_tilt
        self._history.append((t, self.pan, self.tilt))
        while self._history and t - self._history[0][0] > self.cfg.history_s:
            self._history.popleft()

    # ------------------------------------------------------------ queries
    def position_at(self, t: float) -> tuple[float, float]:
        """Commanded pose at an earlier time (camera latency compensation)."""

        if not self._history:
            return self.pan, self.tilt
        best = self._history[0]
        for item in self._history:
            if item[0] <= t:
                best = item
            else:
                break
        return best[1], best[2]

    def at_target(self, t: float, tol_deg: float) -> bool:
        src = self._active_source(t)
        if src is None:
            return True
        r = self._requests[src]
        return (abs(r.pan - self.pan) <= tol_deg
                and abs(r.tilt - self.tilt) <= tol_deg)

    # ----------------------------------------------------------- shutdown
    def shutdown(self, slew_to_neutral: bool = True,
                 sleep: Callable[[float], None] = time.sleep,
                 step_s: float = 0.02) -> None:
        """Return to neutral under the slew limits, release and clean up.

        Safe to call more than once and from any state.  Nothing is moved
        when an emergency stop, user stop or fault is latched.
        """

        if self.shut_down:
            return
        if slew_to_neutral and self.started and not self.blocked:
            now = self._last_t or 0.0
            end = now + self.cfg.shutdown_timeout_s
            self._requests.clear()
            self._requests[Priority.IDLE] = _Request(
                self.pan_neutral, self.tilt_neutral, now, end, 1.0)
            while now < end and not self.faulted:
                now += step_s
                self.update(now)
                if (abs(self.pan - self.pan_neutral) < 1e-3
                        and abs(self.tilt - self.tilt_neutral) < 1e-3):
                    break
                sleep(step_s)
        self.shut_down = True
        self._requests.clear()
        self._safe_release()
        try:
            self.neck.cleanup()
        except Exception:
            pass

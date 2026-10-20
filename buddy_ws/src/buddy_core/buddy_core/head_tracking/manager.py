"""Head-tracking behaviour manager (the priority state machine).

States: IDLE -> LISTENING -> SOUND_ORIENTING -> FACE_TRACKING ->
HAND_TRACKING -> IDLE, plus latched STOPPED / EMERGENCY_STOP / FAULT.

Only the manager's CURRENT owner submits motion targets, and every target
goes through ``HeadMotionCoordinator`` which arbitrates by priority again
as a second line of defence.  Nothing here, or anywhere in head tracking,
can command wheels or legs.

Inputs are plain method calls (wake word, sound estimate, face frame,
hand observation, speaking flag, safety state, user commands, ``tick``);
ROS or any other transport sits outside, in ``ros_adapter``.  Track ids
are used internally and are NOT put into published events or status.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable

from buddy_core.head_tracking.config import HeadTrackingConfig
from buddy_core.head_tracking.face_follow import (
    FaceFollower,
    TrackStatus,
)
from buddy_core.head_tracking.hand_follow import HandFollower, HandStatus
from buddy_core.head_tracking.models import (
    EventKind,
    FaceFrame,
    HandObservation,
    HeadEvent,
    HeadState,
    Priority,
    WakeWordEvent,
)
from buddy_core.head_tracking.motion import HeadMotionCoordinator
from buddy_core.head_tracking.sound_direction import (
    Reason,
    SoundDirectionFilter,
    SoundDirectionResult,
)

S = HeadState
_LATCHED = (S.STOPPED, S.EMERGENCY_STOP, S.FAULT)

#: Allowed transitions.  Entering a latched state is allowed from anywhere
#: and is handled separately.
TRANSITIONS: dict[HeadState, frozenset[HeadState]] = {
    S.IDLE: frozenset({S.LISTENING, S.FACE_TRACKING, S.HAND_TRACKING}),
    S.LISTENING: frozenset({S.IDLE, S.SOUND_ORIENTING, S.FACE_TRACKING,
                            S.HAND_TRACKING}),
    S.SOUND_ORIENTING: frozenset({S.IDLE, S.FACE_TRACKING, S.LISTENING,
                                  S.HAND_TRACKING}),
    S.FACE_TRACKING: frozenset({S.IDLE, S.LISTENING, S.SOUND_ORIENTING,
                                S.HAND_TRACKING}),
    S.HAND_TRACKING: frozenset({S.FACE_TRACKING, S.IDLE}),
    S.STOPPED: frozenset({S.IDLE}),
    S.EMERGENCY_STOP: frozenset({S.IDLE}),
    S.FAULT: frozenset({S.IDLE}),
}


class HeadTrackingManager:
    def __init__(
        self,
        coordinator: HeadMotionCoordinator,
        cfg: HeadTrackingConfig | None = None,
        on_event: Callable[[HeadEvent], None] | None = None,
        max_events: int = 200,
    ) -> None:
        self.co = coordinator
        self.cfg = cfg or HeadTrackingConfig()
        ps, ts = coordinator.pan_sign, coordinator.tilt_sign
        self.face = FaceFollower(self.cfg.face, ps, ts)
        self.hand = HandFollower(self.cfg.hand, ps, ts)
        self.sound_filter = SoundDirectionFilter(self.cfg.sound)
        self.events: deque[HeadEvent] = deque(maxlen=max_events)
        self._sink = on_event
        self._prev_co_sink = coordinator._on_event
        coordinator._on_event = self._record

        self.state = S.IDLE
        self._listen_until = 0.0
        self._speaker_until = 0.0
        self._caller_associated = False
        self._orient: tuple[float, float] | None = None
        self._orient_deadline = 0.0
        self._last_turn_t = -1e9
        self._speaking = False
        self._speaking_ended = -1e9
        self._last_activity = 0.0
        self._hand_kind: str | None = None
        self._lost_neutral_sent = False
        self._announced = False

    # ----------------------------------------------------------- plumbing
    def _record(self, ev: HeadEvent) -> None:
        self.events.append(ev)
        if self._prev_co_sink is not None:
            self._prev_co_sink(ev)
        if self._sink is not None:
            self._sink(ev)

    def _emit(self, kind: EventKind, t: float, /, **data) -> None:
        self._record(HeadEvent(kind, t, data))

    def _go(self, new: HeadState, t: float, reason: str = "") -> bool:
        old = self.state
        if new is old:
            return True
        if new not in _LATCHED and new not in TRANSITIONS[old]:
            self._emit(EventKind.COMMAND_REJECTED, t,
                       reason=f"illegal transition {old.value}->"
                              f"{new.value}")
            return False
        self.state = new
        self._emit(EventKind.STATE_CHANGED, t, old=old.value,
                   new=new.value, reason=reason)
        return True

    def _latched(self, t: float) -> bool:
        """Mirror coordinator latches into the state; True if latched."""

        co = self.co
        want = (S.EMERGENCY_STOP if co.estop else S.FAULT if co.faulted
                else S.STOPPED if co.user_stopped else None)
        if want is not None:
            if self.state is not want:
                self.face.reset()
                self.hand.deactivate()
                self._orient = None
                self._announced = False
                self._go(want, t, "latched")
            return True
        if self.state in _LATCHED:
            self._go(S.IDLE, t, "latch cleared")
        return False

    def _pose_at(self, t: float):
        return self.co.position_at(t)

    @property
    def speaking_blocked(self) -> bool:
        return self._speaking

    def _speech_gate(self, now: float) -> bool:
        return self._speaking or (
            now - self._speaking_ended < self.cfg.state.speaking_cooldown_s)

    def _neutral(self, now: float) -> None:
        if self.cfg.state.return_to_neutral_on_loss:
            self.co.submit(Priority.IDLE, self.co.pan_neutral,
                           self.co.tilt_neutral, now, speed_scale=0.5)

    # ------------------------------------------------------------- inputs
    def set_speaking(self, speaking: bool, now: float) -> None:
        if self._speaking and not speaking:
            self._speaking_ended = now
        self._speaking = bool(speaking)

    def on_wake_word(self, ev: WakeWordEvent, now: float) -> None:
        self._emit(EventKind.WAKE_WORD_DETECTED, now,
                   confidence=float(ev.confidence))
        if self._latched(now):
            return
        self._last_activity = now
        self.sound_filter.reset()
        self._caller_associated = False
        self._speaker_until = now + self.cfg.state.speaker_priority_s
        if ev.speaker_track_id:
            self.face.set_preferred(ev.speaker_track_id)
            self._caller_associated = True
        if self.state is not S.HAND_TRACKING:
            self._listen_until = now + self.cfg.state.listening_timeout_s
            self._go(S.LISTENING, now, "wake word")

    def on_sound_estimate(self, result: SoundDirectionResult,
                          now: float) -> None:
        if self._latched(now):
            return
        if self._speech_gate(now):
            self._emit(EventKind.SOUND_DIRECTION_UNCERTAIN, now,
                       reason=Reason.SPEAKING)
            return
        r = self.sound_filter.add(result)
        if not r.ok:
            self._emit(EventKind.SOUND_DIRECTION_UNCERTAIN, now,
                       reason=r.reason, confidence=r.confidence)
            return
        self._emit(EventKind.SOUND_DIRECTION_ESTIMATED, now,
                   azimuth_deg=r.azimuth_deg, confidence=r.confidence,
                   resolution_deg=r.resolution_deg,
                   front_back_ambiguous=True)
        if self.state is not S.LISTENING:
            return      # only a wake word opens a window for turning
        if now - self._last_turn_t < self.cfg.state.turn_cooldown_s:
            self._emit(EventKind.COMMAND_REJECTED, now,
                       reason="turn_cooldown")
            return
        st = self.cfg.state
        if (self.face.has_target
                and abs(self.face.last_ex) <= st.confirm_center_norm
                and abs(r.azimuth_deg) <= st.confirm_azimuth_deg):
            self.face.set_preferred(self.face.locked)   # visual confirm
            self._caller_associated = True
            self._go(S.FACE_TRACKING, now, "voice confirmed by face")
            return
        self._request_turn(r, now)

    def _request_turn(self, r: SoundDirectionResult, now: float) -> None:
        sc = self.cfg.sound
        base = self.co.pan if sc.mics_mounted_on == "head" \
            else self.co.pan_neutral
        pan = base + self.co.pan_sign * r.azimuth_deg * sc.pan_gain
        pan = min(max(pan, self.co.pan_lo), self.co.pan_hi)
        tilt = self.co.tilt
        self._orient = (pan, tilt)
        self._last_turn_t = now
        self._orient_deadline = now + self.cfg.state.orient_timeout_s
        self._emit(EventKind.HEAD_TURN_REQUESTED, now, pan_deg=pan,
                   tilt_deg=tilt, azimuth_deg=r.azimuth_deg,
                   confidence=r.confidence)
        self.co.submit(Priority.SPEAKER_FACE, pan, tilt, now)
        self._go(S.SOUND_ORIENTING, now, "sound direction")

    def on_face_frame(self, frame: FaceFrame, now: float) -> None:
        if self._latched(now):
            return
        if self.state is S.SOUND_ORIENTING:
            self._orienting_frame(frame, now)
            return
        res = self.face.update(frame, now, self._pose_at)
        self._announce_target(now)
        if self.state is S.HAND_TRACKING:
            return                      # lock kept, nothing submitted
        if res.status is TrackStatus.STALE:
            return
        if (self.state is S.LISTENING and not self._caller_associated
                and self.face.has_target):
            who = self.face.associate_caller(frame)
            if who is not None:
                self.face.set_preferred(who)
                self._caller_associated = True
        if res.status is TrackStatus.ACQUIRING and self.face.has_target:
            if self.state is S.IDLE and self.cfg.state.passive_face_tracking:
                self._go(S.FACE_TRACKING, now, "face acquired")
        elif res.status is TrackStatus.TRACKING:
            self._last_activity = now
            self._lost_neutral_sent = False
            if self.state is S.IDLE:
                if self.cfg.state.passive_face_tracking:
                    self._go(S.FACE_TRACKING, now, "face tracking")
                else:
                    return
            self.co.submit(self._face_priority(now), res.pan, res.tilt,
                           frame.t if frame.t <= now else now)
        elif res.status is TrackStatus.LOST:
            self._face_lost(now)

    def _announce_target(self, now: float) -> None:
        """Emit TARGET_ACQUIRED once per lock, however it was obtained."""

        if self.face.has_target and not self._announced:
            self._announced = True
            self._emit(EventKind.TARGET_ACQUIRED, now)
        elif not self.face.has_target:
            self._announced = False

    def _face_priority(self, now: float) -> Priority:
        if self._caller_associated and now < self._speaker_until:
            return Priority.SPEAKER_FACE
        return Priority.PASSIVE_FACE

    def _face_lost(self, now: float) -> None:
        self._emit(EventKind.TARGET_LOST, now, kind="face")
        self._announced = False
        self.co.release_source(Priority.SPEAKER_FACE)
        self.co.release_source(Priority.PASSIVE_FACE)
        self._caller_associated = False
        if self.state in (S.FACE_TRACKING, S.LISTENING):
            self._go(S.IDLE, now, "face lost")
        self._neutral(now)

    def _orienting_frame(self, frame: FaceFrame, now: float) -> None:
        tol = self.cfg.state.orient_tolerance_deg
        if not self.co.at_target(now, tol) or not frame.faces:
            return
        who = self.face.associate_caller(frame)
        if who is not None:
            self.face.set_preferred(who)
            self._caller_associated = True
        self._orient = None
        self.face.reset()
        self._go(S.FACE_TRACKING, now, "face found after turn")

    def on_hand_observation(self, obs: HandObservation, now: float) -> None:
        if self._latched(now):
            return
        c = self.cfg.hand
        for g in obs.gestures:
            if not self.hand.active and g.name in c.activation_gestures:
                if (c.require_face_target_match and self.face.locked
                        and g.track_id != self.face.locked):
                    continue            # someone else's hand
                self._activate_hand(now, g.side, g.track_id or obs.track_id)
            elif self.hand.active and g.name in c.deactivation_gestures:
                self.deactivate_hand(now, "stop gesture")
                return
        if not self.hand.active:
            return
        res = self.hand.update(obs, now, self._pose_at)
        if res.status is HandStatus.TRACKING:
            self._last_activity = now
            if res.kind != self._hand_kind:
                self._hand_kind = res.kind
                self._emit(EventKind.HAND_ACTIVATED, now, kind=res.kind,
                           side=res.side,
                           accurate_fingertip=res.kind == "index_fingertip")
            self.co.submit(Priority.HAND, res.pan, res.tilt,
                           obs.t if obs.t <= now else now)
        elif res.status is HandStatus.LOST:
            self._hand_ended(now, res.reason)

    # --------------------------------------------------------- hand logic
    def _activate_hand(self, now, side, owner) -> bool:
        if self._latched(now):
            return False
        if not self.hand.activate(now, side, owner or None):
            return False
        self._hand_kind = None
        self._emit(EventKind.HAND_ACTIVATED, now, side=self.hand.side,
                   kind="pending")
        self.co.release_source(Priority.SPEAKER_FACE)
        self.co.release_source(Priority.PASSIVE_FACE)
        self._orient = None
        return self._go(S.HAND_TRACKING, now, "hand activated")

    def activate_hand(self, now: float, side: str | None = None) -> bool:
        """Explicit command (voice/UI) to start following a hand."""

        return self._activate_hand(now, side, self.face.locked)

    def deactivate_hand(self, now: float, reason: str = "command") -> None:
        if self.hand.active:
            self.hand.deactivate()
            self._hand_ended(now, reason)

    def _hand_ended(self, now: float, reason: str) -> None:
        self.co.release_source(Priority.HAND)
        self._hand_kind = None
        self._emit(EventKind.HAND_DEACTIVATED, now, reason=reason)
        if self.state is S.HAND_TRACKING:
            self._go(S.FACE_TRACKING if self.face.has_target else S.IDLE,
                     now, "hand ended")

    # ----------------------------------------------------------- commands
    def set_emergency_stop(self, active: bool, now: float,
                           reason: str = "") -> None:
        if active:
            self.co.emergency_stop(reason, now)
        else:
            self.co.reset_emergency(now)
        self._latched(now)

    def user_stop(self, now: float) -> None:
        self.co.user_stop(now)
        self._latched(now)

    def resume(self, now: float) -> None:
        self.co.resume(now)
        self._latched(now)

    def reset_fault(self, now: float) -> bool:
        ok = self.co.reset_fault(now)
        self._latched(now)
        return ok

    # --------------------------------------------------------------- tick
    def tick(self, now: float) -> dict:
        """Run timeouts, resubmit held targets, advance the servos."""

        if self._latched(now):
            self.co.update(now)
            return self.status(now)

        if self.state is S.LISTENING and now >= self._listen_until:
            nxt = S.FACE_TRACKING if self.face.has_target else S.IDLE
            self._go(nxt, now, "listening timeout")
        if self.state is S.SOUND_ORIENTING and self._orient is not None:
            self.co.submit(Priority.SPEAKER_FACE, *self._orient, now)
            if now >= self._orient_deadline:
                self._orient = None
                self._go(S.IDLE, now, "no face after turn")
                self._neutral(now)
        elif self.state is S.FACE_TRACKING or self.state is S.LISTENING:
            res = self.face.check_timeout(now)
            if res is not None and res.status is TrackStatus.LOST:
                self._face_lost(now)
        if self.hand.active:
            res = self.hand.check_timeout(now)
            if res is not None:
                self._hand_ended(now, res.reason)

        if (self.state is S.IDLE and self.cfg.idle.enabled
                and now - self._last_activity >= self.cfg.idle.quiet_before_s
                and not self.co.blocked):
            amp = min(self.cfg.idle.amplitude_deg,
                      (self.co.pan_hi - self.co.pan_lo) / 2.0)
            phase = math.sin(2 * math.pi * now / self.cfg.idle.period_s)
            self.co.submit(Priority.IDLE,
                           self.co.pan_neutral + amp * phase,
                           self.co.tilt_neutral, now, speed_scale=0.3)
        self.co.update(now)
        return self.status(now)

    def status(self, now: float | None = None) -> dict:
        co = self.co
        t = 0.0 if now is None else now
        src = co.active_source(t) if now is not None else None
        return {
            "state": self.state.value,
            "pan_deg": round(co.pan, 2),
            "tilt_deg": round(co.tilt, 2),
            "active_source": src.name if src is not None else None,
            "face_target": self.face.has_target,
            "hand_active": self.hand.active,
            "hand_kind": self._hand_kind,
            "emergency_stop": co.estop,
            "user_stopped": co.user_stopped,
            "fault": co.fault_reason if co.faulted else "",
            "rejected_commands": co.rejected_commands,
            "suppressed_conflicts": co.suppressed_conflicts,
        }

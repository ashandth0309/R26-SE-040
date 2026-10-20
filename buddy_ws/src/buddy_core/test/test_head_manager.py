"""State machine, priority and safety tests for the head manager."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
from head_fakes import (
    FS,
    SPACING,
    face,
    frame,
    gesture,
    hand_obs,
    kinds,
    make_stack,
    stereo_from_azimuth,
)

from buddy_core.head_tracking import config_from_dict
from buddy_core.head_tracking.manager import TRANSITIONS
from buddy_core.head_tracking.models import (
    EventKind,
    HeadState,
    Priority,
    WakeWordEvent,
)
from buddy_core.head_tracking.sound_direction import (
    Reason,
    SoundDirectionEstimator,
)

S = HeadState
SOUND = {"sound": {"enabled": True, "mic_spacing_m": SPACING}}


class Rig:
    """A started manager plus a clock and small input helpers."""

    def __init__(self, overrides=None, **kw):
        merged = {"sound": dict(SOUND["sound"])}
        for k, v in (overrides or {}).items():
            merged[k] = {**merged.get(k, {}), **v} if isinstance(
                v, dict) else v
        self.backend, self.co, self.mgr, self.cfg = make_stack(
            merged, **kw)
        self.est = SoundDirectionEstimator(self.cfg.sound)
        assert self.co.start(0.0)
        self.t = 0.0

    def run(self, seconds, dt=0.02):
        for _ in range(int(round(seconds / dt))):
            self.t += dt
            self.mgr.tick(self.t)
        return self

    def wake(self, **kw):
        self.mgr.on_wake_word(WakeWordEvent(self.t, **kw), self.t)

    def sound(self, az, seed=0, sync=True):
        r = self.est.estimate(*stereo_from_azimuth(az, seed=seed), FS, sync,
                              t=self.t)
        self.mgr.on_sound_estimate(r, self.t)
        return r

    def two_sounds(self, az):
        self.sound(az, seed=0)
        self.t += 0.2
        self.sound(az + 1, seed=1)

    def faces(self, *faces, n=1, dt=0.05):
        for _ in range(n):
            self.t += dt
            self.mgr.on_face_frame(frame(self.t, *faces), self.t)
            self.mgr.tick(self.t)

    def hand(self, **kw):
        self.t += 0.05
        self.mgr.on_hand_observation(hand_obs(self.t, **kw), self.t)
        self.mgr.tick(self.t)

    @property
    def state(self):
        return self.mgr.state

    @property
    def events(self):
        return self.mgr.events_list

    def count(self, kind):
        return sum(1 for e in self.events if e.kind is kind)


# ------------------------------------------------------ wake + listening
def test_starts_idle():
    assert Rig().state is S.IDLE


def test_wake_word_enters_listening_and_emits_event():
    r = Rig()
    r.wake()
    assert r.state is S.LISTENING
    assert r.count(EventKind.WAKE_WORD_DETECTED) == 1


def test_listening_times_out_to_idle():
    r = Rig()
    r.wake()
    r.run(r.cfg.state.listening_timeout_s + 0.5)
    assert r.state is S.IDLE


# ------------------------------------------------------- voice direction
def test_call_from_the_left_turns_the_head_left():
    r = Rig()
    r.wake()
    r.two_sounds(-40)
    assert r.state is S.SOUND_ORIENTING
    assert r.count(EventKind.HEAD_TURN_REQUESTED) == 1
    r.run(2.0)
    assert r.co.pan < 90.0 - 25.0


def test_call_from_the_right_turns_the_head_right():
    r = Rig()
    r.wake()
    r.two_sounds(40)
    r.run(2.0)
    assert r.co.pan > 90.0 + 25.0


def test_servo_sign_inverts_the_voice_turn():
    r = Rig(pan_sign=-1)
    r.wake()
    r.two_sounds(40)
    r.run(2.0)
    assert r.co.pan < 90.0 - 25.0


def test_single_estimate_is_not_enough_to_turn():
    r = Rig()
    r.wake()
    r.sound(40)
    r.run(1.0)
    assert r.count(EventKind.HEAD_TURN_REQUESTED) == 0
    assert r.count(EventKind.SOUND_DIRECTION_UNCERTAIN) == 1
    assert r.co.pan == 90.0


def test_sound_without_wake_word_never_turns_the_head():
    r = Rig()
    r.two_sounds(40)
    r.run(1.0)
    assert r.count(EventKind.SOUND_DIRECTION_ESTIMATED) >= 1
    assert r.count(EventKind.HEAD_TURN_REQUESTED) == 0
    assert r.co.pan == 90.0 and r.state is S.IDLE


def test_unsynchronized_microphones_never_move_the_head():
    r = Rig()
    r.wake()
    for _ in range(4):
        r.sound(40, sync=False)
        r.t += 0.2
    r.run(1.0)
    assert r.count(EventKind.SOUND_DIRECTION_UNCERTAIN) == 4
    assert r.count(EventKind.HEAD_TURN_REQUESTED) == 0
    assert r.co.pan == 90.0


def test_uncertain_result_reason_is_reported():
    r = Rig()
    r.wake()
    r.sound(40, sync=False)
    ev = [e for e in r.events
          if e.kind is EventKind.SOUND_DIRECTION_UNCERTAIN][0]
    assert ev.data["reason"] == Reason.UNSYNCHRONIZED


def test_no_turn_while_buddy_is_speaking_and_during_cooldown():
    r = Rig()
    r.mgr.set_speaking(True, r.t)
    r.wake()
    r.two_sounds(40)
    assert r.count(EventKind.HEAD_TURN_REQUESTED) == 0
    reasons = [e.data["reason"] for e in r.events
               if e.kind is EventKind.SOUND_DIRECTION_UNCERTAIN]
    assert reasons and set(reasons) == {Reason.SPEAKING}
    r.t += 0.5
    r.mgr.set_speaking(False, r.t)
    r.t += 0.2
    r.sound(40)                                  # echo tail still gated
    assert r.count(EventKind.HEAD_TURN_REQUESTED) == 0
    r.t += 1.0
    r.wake()
    r.two_sounds(40)
    assert r.count(EventKind.HEAD_TURN_REQUESTED) == 1
    assert r.co.pan == 90.0 or r.state is S.SOUND_ORIENTING


def test_repeated_estimates_do_not_cause_repeated_turns():
    r = Rig()
    r.wake()
    r.two_sounds(40)
    for i in range(6):
        r.t += 0.1
        r.sound(40, seed=10 + i)
    r.run(1.0)
    r.wake()
    r.two_sounds(-40)                  # inside the cooldown window
    assert r.count(EventKind.HEAD_TURN_REQUESTED) == 1


def test_turn_target_is_clamped_to_the_safe_range():
    r = Rig({"sound": {"pan_gain": 10.0}})
    r.wake()
    r.two_sounds(60)
    ev = [e for e in r.events
          if e.kind is EventKind.HEAD_TURN_REQUESTED][0]
    assert ev.data["pan_deg"] <= r.co.pan_hi
    r.run(6.0)
    assert max(r.backend.writes("pan")) <= 177.0


def test_orienting_then_visual_acquisition_prefers_the_caller():
    r = Rig()
    r.wake()
    r.two_sounds(30)
    r.run(2.0)
    r.faces(face("caller", 0.5, 0.5), face("bystander", 0.9, 0.5))
    assert r.state is S.FACE_TRACKING
    r.faces(face("caller", 0.5, 0.5), face("bystander", 0.9, 0.5), n=4)
    assert r.mgr.face.locked == "caller"
    assert r.co.active_source(r.t) in (Priority.SPEAKER_FACE, None)


def test_ambiguous_faces_after_turn_do_not_force_a_caller():
    r = Rig()
    r.wake()
    r.two_sounds(30)
    r.run(2.0)
    r.faces(face("l", 0.35, 0.5), face("r", 0.65, 0.5))
    assert r.state is S.FACE_TRACKING
    assert r.mgr._caller_associated is False


def test_no_face_after_turn_returns_to_idle_and_neutral():
    r = Rig()
    r.wake()
    r.two_sounds(40)
    r.run(r.cfg.state.orient_timeout_s + 1.0)
    assert r.state is S.IDLE
    r.run(10.0)
    assert abs(r.co.pan - 90.0) <= 0.6


def test_voice_confirmed_by_a_centered_face_does_not_turn():
    r = Rig()
    r.faces(face("a", 0.5, 0.5), n=6)
    assert r.state is S.FACE_TRACKING
    r.wake()
    r.two_sounds(8)
    assert r.count(EventKind.HEAD_TURN_REQUESTED) == 0
    assert r.state is S.FACE_TRACKING
    assert r.mgr._caller_associated


# ----------------------------------------------------------- face states
def test_passive_face_tracking_starts_from_idle():
    r = Rig()
    r.faces(face("a", 0.75, 0.5), n=6)
    assert r.state is S.FACE_TRACKING
    assert r.co.active_source(r.t) is Priority.PASSIVE_FACE
    r.faces(face("a", 0.75, 0.5), n=10)
    assert r.co.pan > 90.0


def test_passive_tracking_can_be_disabled():
    r = Rig({"state": {"passive_face_tracking": False}})
    r.faces(face("a", 0.8, 0.5), n=10)
    assert r.state is S.IDLE and r.co.pan == 90.0


def test_losing_the_face_returns_to_idle_and_neutral():
    r = Rig()
    r.faces(face("a", 0.7, 0.5), n=12)
    moved = r.co.pan
    assert moved > 90.0
    r.faces(n=40)                                 # empty frames
    assert r.state is S.IDLE
    assert r.count(EventKind.TARGET_LOST) == 1
    r.run(10.0)
    assert abs(r.co.pan - 90.0) <= 0.6


def test_head_holds_still_while_the_target_is_briefly_missing():
    r = Rig()
    r.faces(face("a", 0.7, 0.5), n=12)
    r.run(0.7)                                    # let it settle
    held = r.co.pan
    r.faces(n=10)                                 # 0.5 s without the face
    assert r.state is S.FACE_TRACKING
    assert r.co.pan == pytest.approx(held, abs=0.7)


def test_missing_camera_frames_stop_tracking_safely():
    r = Rig()
    r.faces(face("a", 0.7, 0.5), n=12)
    r.run(r.cfg.face.lost_hold_s + 1.0)           # no frames at all
    assert r.state is S.IDLE
    assert r.count(EventKind.TARGET_LOST) == 1
    r.run(10.0)
    assert abs(r.co.pan - 90.0) <= 0.6


def test_stale_frames_are_ignored_by_the_manager():
    r = Rig()
    r.t = 10.0
    r.mgr.on_face_frame(frame(1.0, face("a", 0.8, 0.5)), r.t)
    r.mgr.tick(r.t + 0.1)
    assert r.state is S.IDLE


# ------------------------------------------------------------- hand states
def face_tracked_rig(**kw):
    r = Rig(**kw)
    r.faces(face("p1", 0.5, 0.5), n=8)
    assert r.mgr.face.locked == "p1"
    return r


def test_hand_gesture_activates_following_and_wrist_moves_the_head():
    r = face_tracked_rig()
    r.hand(gestures=[gesture("raised_hand")])
    assert r.state is S.HAND_TRACKING
    for _ in range(40):
        r.hand(wrist=(0.85, 0.5))
    assert r.co.pan > 90.0 + 5.0
    ev = [e for e in r.events if e.kind is EventKind.HAND_ACTIVATED][-1]
    assert ev.data["kind"] == "wrist_fallback"
    assert ev.data["accurate_fingertip"] is False


def test_hand_vertical_motion_moves_the_tilt():
    r = face_tracked_rig()
    r.hand(gestures=[gesture("raised_hand")])
    for _ in range(40):
        r.hand(wrist=(0.5, 0.15))
    up = r.co.tilt
    assert up < 90.0 - 3.0
    for _ in range(80):
        r.hand(wrist=(0.5, 0.9))
    assert r.co.tilt > up + 5.0


def test_hand_following_never_starts_without_an_activation():
    r = face_tracked_rig()
    for _ in range(40):
        r.faces(face("p1", 0.5, 0.5))
        r.hand(wrist=(0.9, 0.5))
    assert r.state is S.FACE_TRACKING and not r.mgr.hand.active
    assert abs(r.co.pan - 90.0) <= 1.0


def test_gesture_from_someone_else_is_ignored():
    r = face_tracked_rig()
    r.hand(track="stranger", gestures=[gesture("raised_hand", "stranger")])
    assert r.state is S.FACE_TRACKING and not r.mgr.hand.active


def test_unconfigured_gesture_does_not_activate():
    r = face_tracked_rig()
    r.hand(gestures=[gesture("wave")])
    assert r.state is S.FACE_TRACKING


def test_stop_gesture_returns_to_face_tracking():
    r = face_tracked_rig()
    r.hand(gestures=[gesture("raised_hand")])
    r.hand(wrist=(0.8, 0.5), gestures=[gesture("both_hands_up")])
    assert r.state is S.FACE_TRACKING and not r.mgr.hand.active
    assert r.count(EventKind.HAND_DEACTIVATED) == 1


def test_lost_hand_returns_to_face_tracking():
    r = face_tracked_rig()
    r.hand(gestures=[gesture("raised_hand")])
    for _ in range(5):
        r.hand(wrist=(0.8, 0.5))
    for _ in range(80):
        r.faces(face("p1", 0.5, 0.5))
    assert r.state is S.FACE_TRACKING and not r.mgr.hand.active


def test_explicit_commands_activate_and_deactivate_hand_following():
    r = Rig()
    assert r.mgr.activate_hand(r.t, "right") is True
    assert r.state is S.HAND_TRACKING
    r.mgr.deactivate_hand(r.t)
    assert r.state is S.IDLE


def test_face_and_hand_do_not_send_conflicting_commands():
    r = face_tracked_rig()
    r.hand(gestures=[gesture("raised_hand")])
    for _ in range(60):
        # the face wants LEFT, the hand wants RIGHT, at the same time
        r.t += 0.05
        r.mgr.on_face_frame(frame(r.t, face("p1", 0.1, 0.5)), r.t)
        r.mgr.on_hand_observation(hand_obs(r.t, wrist=(0.9, 0.5)), r.t)
        r.mgr.tick(r.t)
    assert r.co.pan > 90.0 + 5.0
    assert r.co.suppressed_conflicts == 0     # face never even submitted
    pans = r.backend.writes("pan")
    assert all(b >= a - 1e-9 for a, b in zip(pans, pans[1:]))


def test_wake_word_during_hand_tracking_does_not_interrupt_it():
    r = face_tracked_rig()
    r.hand(gestures=[gesture("raised_hand")])
    r.wake()
    r.two_sounds(-40)
    assert r.state is S.HAND_TRACKING
    assert r.count(EventKind.HEAD_TURN_REQUESTED) == 0


# ---------------------------------------------------- stop / e-stop / fault
def test_emergency_stop_during_hand_tracking_halts_everything():
    r = face_tracked_rig()
    r.hand(gestures=[gesture("raised_hand")])
    for _ in range(10):
        r.hand(wrist=(0.9, 0.5))
    r.mgr.set_emergency_stop(True, r.t, "button")
    assert r.state is S.EMERGENCY_STOP
    n, pan = len(r.backend.calls), r.co.pan
    for _ in range(40):
        r.hand(wrist=(0.1, 0.5))
        r.faces(face("p1", 0.1, 0.5))
    r.wake()
    r.two_sounds(-40)
    assert len(r.backend.calls) == n and r.co.pan == pan
    assert r.state is S.EMERGENCY_STOP
    assert not r.mgr.hand.active


def test_clearing_the_emergency_stop_returns_to_idle_without_motion():
    r = face_tracked_rig()
    r.mgr.set_emergency_stop(True, r.t)
    r.mgr.set_emergency_stop(False, r.t + 1)
    assert r.state is S.IDLE
    pan = r.co.pan
    r.run(1.0)
    assert r.co.pan == pan
    r.wake()
    assert r.state is S.LISTENING


def test_user_stop_blocks_all_inputs_until_resumed():
    r = face_tracked_rig()
    r.mgr.user_stop(r.t)
    assert r.state is S.STOPPED
    n = len(r.backend.calls)
    r.faces(face("p1", 0.9, 0.5), n=20)
    r.wake()
    assert len(r.backend.calls) == n and r.state is S.STOPPED
    r.mgr.resume(r.t)
    assert r.state is S.IDLE


def test_emergency_stop_outranks_user_stop():
    r = Rig()
    r.mgr.user_stop(r.t)
    r.mgr.set_emergency_stop(True, r.t)
    assert r.state is S.EMERGENCY_STOP
    r.mgr.set_emergency_stop(False, r.t)
    assert r.state is S.STOPPED            # the user stop is still latched
    r.mgr.resume(r.t)
    assert r.state is S.IDLE


def test_hardware_fault_enters_fault_state_and_needs_a_reset():
    r = face_tracked_rig()
    r.backend.fail_after = len(r.backend.calls) + 1
    r.faces(face("p1", 0.9, 0.5), n=20)
    assert r.state is S.FAULT
    assert r.status_has_fault()
    r.faces(face("p1", 0.9, 0.5), n=10)
    assert r.state is S.FAULT
    r.backend.fail_after = None
    assert r.mgr.reset_fault(r.t) is True
    assert r.state is S.IDLE


Rig.status_has_fault = lambda self: bool(self.mgr.status(self.t)["fault"])


# -------------------------------------------------------- state machine
def test_full_nominal_cycle_passes_through_all_five_states():
    r = Rig()
    seen = [r.state]

    def note():
        if r.state is not seen[-1]:
            seen.append(r.state)

    r.wake()
    note()
    r.two_sounds(30)
    note()
    r.run(2.0)
    r.faces(face("p1", 0.5, 0.5), n=8)
    note()
    r.hand(gestures=[gesture("raised_hand")])
    note()
    r.hand(gestures=[gesture("both_hands_up")])
    r.faces(n=60)
    note()
    assert seen[0] is S.IDLE and S.LISTENING in seen
    for needed in (S.SOUND_ORIENTING, S.FACE_TRACKING, S.HAND_TRACKING):
        assert needed in seen
    assert r.state is S.IDLE


def test_every_state_has_a_transition_entry():
    assert set(TRANSITIONS) == set(HeadState)


def test_illegal_transition_is_refused_and_reported():
    r = Rig()
    assert r.mgr._go(S.HAND_TRACKING, 0.0) is True
    assert r.mgr._go(S.LISTENING, 0.0) is False
    assert "COMMAND_REJECTED" in kinds(r.events)
    assert r.state is S.HAND_TRACKING


def test_state_changes_are_emitted():
    r = Rig()
    r.wake()
    changes = [e for e in r.events if e.kind is EventKind.STATE_CHANGED]
    assert changes[-1].data["new"] == "LISTENING"


# ------------------------------------------------------------------ idle
def test_idle_movement_is_off_by_default():
    r = Rig()
    r.run(60.0)
    assert r.backend.writes("pan") == [90.0]


def test_idle_movement_is_small_slow_and_only_when_safe():
    r = Rig({"idle": {"enabled": True, "amplitude_deg": 6.0,
                      "quiet_before_s": 2.0}})
    r.run(1.0)
    assert r.co.pan == 90.0                       # not yet quiet long enough
    r.run(40.0)
    pans = r.backend.writes("pan")
    assert len(pans) > 5
    assert max(pans) <= 96.7 and min(pans) >= 83.3
    r.mgr.set_emergency_stop(True, r.t)
    n = len(r.backend.calls)
    r.run(10.0)
    assert len(r.backend.calls) == n


def test_idle_movement_yields_to_face_tracking():
    r = Rig({"idle": {"enabled": True, "amplitude_deg": 6.0,
                      "quiet_before_s": 1.0}})
    r.run(20.0)
    r.faces(face("a", 0.8, 0.5), n=12)
    assert r.co.active_source(r.t) is Priority.PASSIVE_FACE


# ---------------------------------------------------- privacy / isolation
def test_published_status_and_events_do_not_contain_track_ids():
    r = Rig()
    r.wake(speaker_track_id="secret-person-key")
    r.faces(face("secret-person-key", 0.6, 0.5), n=10)
    r.hand(gestures=[gesture("raised_hand", "secret-person-key")])
    blob = json.dumps(r.mgr.status(r.t)) + json.dumps(
        [e.data for e in r.events], default=str)
    assert "secret-person-key" not in blob


def _code_strings(tree):
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    getattr(body[0], "value", None), ast.Constant):
                docs.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docs:
                yield node.value


def test_head_tracking_cannot_command_locomotion():
    """Static check: no import of, name of, or topic for locomotion."""

    root = Path(__import__("buddy_core").__file__).parent / "head_tracking"
    banned_modules = ("buddy_core.motors", "buddy_navigation",
                      "buddy_nav_interfaces")
    banned_names = {"MovementCommand", "NavigationCommand", "MotorExecutor"}
    banned_topics = re.compile(r"/movement|cmd_vel|/navigation", re.I)
    files = list(root.glob("*.py"))
    assert len(files) >= 8
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
                mods += [f"{node.module}.{a.name}" for a in node.names]
            else:
                mods = []
            for m in mods:
                assert not m.startswith(banned_modules), (path.name, m)
                assert m.rsplit(".", 1)[-1] not in banned_names
            if isinstance(node, ast.Name):
                assert node.id not in banned_names, (path.name, node.id)
        for text in _code_strings(tree):
            assert not banned_topics.search(text), (path.name, text)


def test_required_event_kinds_exist():
    names = {k.value for k in EventKind}
    assert {"WAKE_WORD_DETECTED", "SOUND_DIRECTION_ESTIMATED",
            "SOUND_DIRECTION_UNCERTAIN", "HEAD_TURN_REQUESTED"} <= names


def test_config_has_no_surprise_hardware_defaults():
    cfg = config_from_dict({})
    assert cfg.hardware.mode == "mock"
    assert cfg.hardware.hardware_arm is False
    assert cfg.sound.enabled is False and cfg.sound.mic_spacing_m is None
    assert cfg.idle.enabled is False

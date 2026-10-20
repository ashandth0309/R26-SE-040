"""Closed-loop mock integration tests.

A virtual camera turns "where people are" and "where the head points"
into face boxes, so the whole chain (vision input -> follower -> manager
-> coordinator -> mock servos) runs without hardware.  Passing here proves
the control logic is stable in simulation; it does NOT prove anything
about the real camera, servos, latency or microphones.
"""

from __future__ import annotations

import numpy as np
from head_fakes import (
    FS,
    H,
    SPACING,
    W,
    gesture,
    hand_obs,
    make_stack,
    stereo_from_azimuth,
)

from buddy_core.head_tracking.models import (
    EventKind,
    FaceFrame,
    FaceObservation,
    HeadState,
    WakeWordEvent,
)
from buddy_core.head_tracking.sound_direction import SoundDirectionEstimator

S = HeadState
HFOV, VFOV = 60.0, 45.0
SOUND = {"sound": {"enabled": True, "mic_spacing_m": SPACING}}


class Person:
    def __init__(self, tid, pan, tilt, vel_pan=0.0, size=90.0):
        self.tid, self.pan, self.tilt = tid, pan, tilt
        self.vel_pan, self.size = vel_pan, size
        self.present = True


class Sim:
    """30 Hz loop with a pinhole-style camera and optional latency."""

    def __init__(self, overrides=None, latency_s=0.0, pan_sign=1,
                 tilt_sign=1):
        merged = {"sound": SOUND["sound"]}
        merged.update(overrides or {})
        self.backend, self.co, self.mgr, self.cfg = make_stack(
            merged, pan_sign=pan_sign, tilt_sign=tilt_sign)
        self.pan_sign, self.tilt_sign = pan_sign, tilt_sign
        self.latency = latency_s
        self.people: list[Person] = []
        self.est = SoundDirectionEstimator(self.cfg.sound)
        self.t = 0.0
        self.dt = 1 / 30
        assert self.co.start(0.0)
        self.pan_trace: list[float] = []

    def frame_at(self, t_capture):
        pan, tilt = self.co.position_at(t_capture)
        faces = []
        for p in self.people:
            if not p.present:
                continue
            ex = self.pan_sign * (p.pan - pan) / (HFOV / 2)
            ey = self.tilt_sign * (p.tilt - tilt) / (VFOV / 2)
            if abs(ex) >= 1 or abs(ey) >= 1:
                continue                      # outside the field of view
            cx, cy = (ex / 2 + 0.5) * W, (ey / 2 + 0.5) * H
            faces.append(FaceObservation(
                p.tid, cx - p.size / 2, cy - p.size / 2, p.size, p.size,
                0.9))
        return FaceFrame(t_capture, W, H, tuple(faces))

    def step(self, hand=None):
        self.t += self.dt
        for p in self.people:
            p.pan += p.vel_pan * self.dt
        fr = self.frame_at(max(self.t - self.latency, 0.0))
        self.mgr.on_face_frame(fr, self.t)
        if hand is not None:
            self.mgr.on_hand_observation(hand(self.t), self.t)
        self.mgr.tick(self.t)
        self.pan_trace.append(self.co.pan)

    def run(self, seconds, **kw):
        for _ in range(int(seconds / self.dt)):
            self.step(**kw)

    def error(self, person):
        return (person.pan - self.co.pan) * self.pan_sign


def reversals(trace, tol=0.15, skip=0):
    """Direction changes of motion larger than ``tol`` degrees."""

    d = np.diff(np.asarray(trace[skip:]))
    d = d[np.abs(d) > 1e-9]
    signs = np.sign(d)
    return int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0


def test_centers_a_static_face_without_oscillating():
    sim = Sim()
    sim.people = [Person("a", 90 + 22, 90 - 8)]
    sim.run(8.0)
    assert abs(sim.error(sim.people[0])) <= 3.0
    settled = sim.pan_trace[int(5 / sim.dt):]
    assert max(settled) - min(settled) <= 1.0
    assert reversals(sim.pan_trace) <= 2


def test_stable_with_camera_latency():
    for latency in (0.0, 0.1, 0.2):
        sim = Sim(latency_s=latency)
        sim.people = [Person("a", 90 + 24, 90)]
        sim.run(12.0)
        assert abs(sim.error(sim.people[0])) <= 4.0, latency
        tail = sim.pan_trace[int(8 / sim.dt):]
        assert max(tail) - min(tail) <= 2.0, latency


def test_works_with_inverted_servo_directions():
    sim = Sim(pan_sign=-1, tilt_sign=-1)
    sim.people = [Person("a", 90 - 20, 90 + 6)]
    sim.run(8.0)
    assert abs(sim.error(sim.people[0])) <= 3.0
    assert sim.co.pan < 90.0


def test_follows_a_moving_person_with_bounded_error():
    sim = Sim()
    sim.people = [Person("a", 90 - 20, 90, vel_pan=6.0)]
    sim.run(3.0)
    errs = []
    for _ in range(int(5 / sim.dt)):
        sim.step()
        errs.append(abs(sim.error(sim.people[0])))
    assert max(errs) <= 10.0
    assert sim.mgr.state is S.FACE_TRACKING


def test_moving_target_horizontal_and_vertical_directions():
    sim = Sim()
    p = Person("a", 90, 90)
    sim.people = [p]
    sim.run(3.0)
    base_pan, base_tilt = sim.co.pan, sim.co.tilt
    p.pan, p.tilt = 90 + 15, 90 - 8
    sim.run(4.0)
    assert sim.co.pan > base_pan + 8 and sim.co.tilt < base_tilt - 3
    p.pan, p.tilt = 90 - 15, 90 + 8
    sim.run(5.0)
    assert sim.co.pan < base_pan - 8 and sim.co.tilt > base_tilt + 3


def test_second_person_crossing_does_not_steal_the_target():
    sim = Sim()
    a = Person("a", 90 + 4, 90)
    b = Person("b", 90 - 25, 90, vel_pan=12.0, size=170)
    sim.people = [a, b]
    sim.run(2.0)
    assert sim.mgr.face.locked is not None
    first = sim.mgr.face.locked
    ids = set()
    for _ in range(int(6 / sim.dt)):
        sim.step()
        ids.add(sim.mgr.face.locked)
    assert ids == {first}


def test_target_leaving_sends_the_head_back_to_neutral():
    sim = Sim()
    p = Person("a", 90 + 25, 90)
    sim.people = [p]
    sim.run(6.0)
    assert sim.co.pan > 105
    p.present = False
    sim.run(4.0)
    assert sim.mgr.state is S.IDLE
    sim.run(12.0)
    assert abs(sim.co.pan - 90.0) <= 0.6


def test_every_servo_write_stays_inside_limits_during_chaotic_input():
    sim = Sim()
    rng = np.random.default_rng(2)
    p = Person("a", 90, 90)
    sim.people = [p]
    for _ in range(20):
        p.pan = 90 + rng.uniform(-80, 80)
        p.tilt = 90 + rng.uniform(-40, 40)
        sim.run(1.0)
    for a in sim.backend.writes("pan"):
        assert 0.0 <= a <= 180.0
    for a in sim.backend.writes("tilt"):
        assert 45.0 <= a <= 135.0


def test_end_to_end_call_acquire_follow_hand_and_recover():
    """The whole story from the definition of done, in simulation."""

    sim = Sim()
    est = sim.est
    caller = Person("caller", 90 + 35, 90 - 5)     # 35 deg to the right
    bystander = Person("bystander", 90 - 30, 90)
    sim.people = [caller, bystander]

    # 1) the caller says "Buddy": wake word + two audio windows
    sim.mgr.on_wake_word(WakeWordEvent(sim.t), sim.t)
    assert sim.mgr.state is S.LISTENING
    for seed in (0, 1):
        r = est.estimate(*stereo_from_azimuth(35 + seed, seed=seed), FS,
                         True, t=sim.t)
        sim.mgr.on_sound_estimate(r, sim.t)
        sim.run(0.1)
    assert sim.mgr.state in (S.SOUND_ORIENTING, S.FACE_TRACKING)

    # 2) the head turns right and finds faces; it keeps the caller
    sim.run(5.0)
    assert sim.mgr.state is S.FACE_TRACKING
    assert abs(sim.error(caller)) <= 4.0
    assert sim.co.pan > 90 + 25

    # 3) the caller raises a hand and moves it; head follows the wrist
    def held(t):
        """A wrist 15 deg left of and 12 deg above the caller, in the
        world; its image position follows from where the head points."""

        pan, tilt = sim.co.position_at(t)
        ex = (caller.pan - 15 - pan) / (HFOV / 2)
        ey = (caller.tilt - 12 - tilt) / (VFOV / 2)
        ex, ey = max(-0.99, min(0.99, ex)), max(-0.99, min(0.99, ey))
        return hand_obs(t, wrist=(ex / 2 + 0.5, ey / 2 + 0.5),
                        track="caller")

    sim.mgr.on_hand_observation(
        hand_obs(sim.t, track="caller",
                 gestures=[gesture("raised_hand", "caller")]), sim.t)
    assert sim.mgr.state is S.HAND_TRACKING
    before = sim.co.pan
    sim.run(3.0, hand=held)
    assert sim.co.pan < before - 8.0               # hand is to the left
    assert sim.co.tilt < 90 - 5 - 3.0              # and above the caller

    # 4) stop gesture: back to face tracking of the same caller
    sim.mgr.on_hand_observation(
        hand_obs(sim.t, track="caller",
                 gestures=[gesture("both_hands_up", "caller")]), sim.t)
    assert sim.mgr.state is S.FACE_TRACKING
    sim.run(6.0)
    assert abs(sim.error(caller)) <= 4.0

    # 5) the caller walks away: safe neutral, nothing out of limits
    caller.present = bystander.present = False
    sim.run(5.0)
    sim.run(12.0)
    assert sim.mgr.state is S.IDLE
    assert abs(sim.co.pan - 90.0) <= 0.6
    assert sim.co.suppressed_conflicts == 0

    # 6) shutdown parks and releases the servos
    sim.co.shutdown(sleep=lambda s: None)
    assert sim.backend.released and sim.backend.cleaned_up
    kinds = {e.kind for e in sim.mgr.events}
    assert {EventKind.WAKE_WORD_DETECTED, EventKind.HEAD_TURN_REQUESTED,
            EventKind.SOUND_DIRECTION_ESTIMATED,
            EventKind.TARGET_ACQUIRED, EventKind.TARGET_LOST,
            EventKind.HAND_ACTIVATED,
            EventKind.HAND_DEACTIVATED} <= kinds

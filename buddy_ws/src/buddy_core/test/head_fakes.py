"""Shared fakes and builders for the head-tracking tests."""

from __future__ import annotations

import copy

import numpy as np

from buddy_core.head_tracking import (
    HeadMotionCoordinator,
    HeadTrackingManager,
    config_from_dict,
)
from buddy_core.head_tracking.models import (
    FaceFrame,
    FaceObservation,
    GestureEvent,
    HandObservation,
    HandPoint,
)
from buddy_core.neck.base import BuddyNeckError
from buddy_core.neck.controller import NeckController
from buddy_core.neck.mock import MockNeckBackend

W, H = 640, 480
FS = 16000
SPACING = 0.15


def neck_config(pan=(0.0, 90.0, 180.0), tilt=(45.0, 90.0, 135.0)):
    """A symmetric test neck.  NOT the robot's calibration."""

    def axis(v):
        return {"min_angle_deg": v[0], "center_angle_deg": v[1],
                "max_angle_deg": v[2]}

    return {"hardware": {"neck": {"pan": axis(pan), "tilt": axis(tilt)}}}


class RecordingBackend(MockNeckBackend):
    """Mock backend that logs every call and can simulate a dead bus."""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[str, float | None]] = []
        self.fail_after: int | None = None
        self.fail_release = False

    def _maybe_fail(self):
        n = sum(1 for c in self.calls if c[0] in ("pan", "tilt"))
        if self.fail_after is not None and n >= self.fail_after:
            raise BuddyNeckError("I2C bus disconnected")

    def set_pan(self, angle_deg):
        self._maybe_fail()
        self.calls.append(("pan", angle_deg))
        super().set_pan(angle_deg)

    def set_tilt(self, angle_deg):
        self._maybe_fail()
        self.calls.append(("tilt", angle_deg))
        super().set_tilt(angle_deg)

    def release(self):
        self.calls.append(("release", None))
        if self.fail_release:
            raise BuddyNeckError("release failed")
        super().release()

    def cleanup(self):
        self.calls.append(("cleanup", None))
        super().cleanup()

    def writes(self, axis):
        return [a for k, a in self.calls if k == axis]


def make_stack(overrides=None, neck_cfg=None, pan_sign=1, tilt_sign=1,
               backend=None):
    cfg = config_from_dict(overrides or {})
    backend = backend or RecordingBackend()
    neck = NeckController(backend, neck_cfg or neck_config())
    events = []
    co = HeadMotionCoordinator(neck, cfg.motion, events.append,
                               pan_sign, tilt_sign)
    mgr = HeadTrackingManager(co, cfg, on_event=None)
    mgr.events_list = events
    return backend, co, mgr, cfg


def run(co_or_mgr, t0, seconds, dt=0.02):
    """Advance time; returns the final time."""

    t = t0
    steps = int(round(seconds / dt))
    for _ in range(steps):
        t += dt
        if hasattr(co_or_mgr, "tick"):
            co_or_mgr.tick(t)
        else:
            co_or_mgr.update(t)
    return t


def face(track, cx, cy, size=80.0, conf=0.9):
    """Face at normalised image position (0..1, 0..1)."""

    return FaceObservation(track, cx * W - size / 2, cy * H - size / 2,
                           size, size, conf)


def frame(t, *faces):
    return FaceFrame(t, W, H, tuple(faces))


def hand_obs(t, wrist=None, tip=None, side="right", track="p1",
             gestures=()):
    wr = {side: HandPoint(wrist[0] * W, wrist[1] * H, 0.9)} if wrist else {}
    tp = {side: HandPoint(tip[0] * W, tip[1] * H, 0.9)} if tip else {}
    return HandObservation(t, W, H, track, wr, tp, tuple(gestures))


def gesture(name, track="p1", side="right"):
    return GestureEvent(name, track, side)


def kinds(events):
    return [e.kind.value for e in events]


# ------------------------------------------------------------- audio
def frac_delay(x, delay_s, fs=FS):
    n = len(x)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return np.fft.irfft(spec * np.exp(-2j * np.pi * f * delay_s), n)


def stereo_from_azimuth(az_deg, seconds=0.3, snr_db=20.0, seed=0,
                        spacing=SPACING, level=0.05):
    """Synthetic far-field source.  az > 0 = robot's right.

    The nearer microphone hears the sound first, so for a source on the
    right the right channel (ch1) leads: ch1 = ch0 delayed by a
    NEGATIVE amount.
    """

    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    s = rng.standard_normal(n)
    tau = -spacing * np.sin(np.radians(az_deg)) / 343.0
    sigma = 10 ** (-snr_db / 20.0)
    ch0 = s + sigma * rng.standard_normal(n)
    ch1 = frac_delay(s, tau) + sigma * rng.standard_normal(n)
    return level * ch0, level * ch1


def deep(cfg):
    return copy.deepcopy(cfg)

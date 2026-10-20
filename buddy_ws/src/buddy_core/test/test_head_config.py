"""Configuration, hardware gating, mock PCA9685 and disconnection tests."""

from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import pytest
from head_fakes import face, frame

from buddy_core.config_loader import load_buddy_config
from buddy_core.head_tracking import (
    HeadTrackingConfigError,
    build_head_system,
    config_from_dict,
    hardware_gate_problems,
    load_head_tracking_config,
)
from buddy_core.head_tracking.factory import ARM_ENV
from buddy_core.head_tracking.models import HeadState, Priority
from buddy_core.neck.mock import MockNeckBackend
from buddy_core.neck.pca9685 import PCA9685NeckBackend

ARMED = {ARM_ENV: "1"}
TEMPLATE = Path(__file__).resolve().parents[4] / "config" / \
    "head_tracking.yaml"


def physical_buddy():
    cfg = copy.deepcopy(load_buddy_config())
    cfg["hardware"]["enabled"] = True
    cfg["hardware"]["simulation"] = False
    cfg["hardware"]["neck"]["enabled"] = True
    return cfg


def armed_ht(**over):
    hw = {"mode": "hardware", "hardware_arm": True,
          "calibration_confirmed": True, "pan_sign": 1, "tilt_sign": 1}
    hw.update(over)
    return config_from_dict({"hardware": hw})


class FakeServo:
    def __init__(self, log, ch, fail):
        self._log, self._ch, self._fail = log, ch, fail
        self._angle = None

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        if self._fail["on"] and value is not None:
            raise OSError("[Errno 121] Remote I/O error")
        self._angle = value
        self._log.append((self._ch, value))


class FakeServoKit:
    log: list = []
    fail = {"on": False}
    deinit_called = False

    def __init__(self, channels, address, frequency):
        FakeServoKit.init_args = (channels, address, frequency)
        self.servo = [FakeServo(FakeServoKit.log, c, FakeServoKit.fail)
                      for c in range(channels)]
        outer = FakeServoKit

        class _Pca:
            def deinit(self_inner):
                outer.deinit_called = True

        self._pca = _Pca()


@pytest.fixture
def fake_kit(monkeypatch):
    FakeServoKit.log = []
    FakeServoKit.fail = {"on": False}
    FakeServoKit.deinit_called = False
    mod = types.ModuleType("adafruit_servokit")
    mod.ServoKit = FakeServoKit
    monkeypatch.setitem(sys.modules, "adafruit_servokit", mod)
    return FakeServoKit


# ----------------------------------------------------------- config
def test_template_loads_validates_and_is_safe():
    cfg = load_head_tracking_config(TEMPLATE)
    assert cfg.hardware.mode == "mock"
    assert cfg.hardware.hardware_arm is False
    assert cfg.hardware.calibration_confirmed is False
    assert cfg.sound.enabled is False
    assert cfg.idle.enabled is False
    assert cfg.hand.allow_wrist_fallback is True


def test_missing_file_falls_back_to_safe_defaults(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_head_tracking_config(tmp_path / "nope.yaml")
    assert config_from_dict(None).hardware.mode == "mock"


@pytest.mark.parametrize("bad", [
    {"motion": {"max_velocity_pan_dps": 0}},
    {"motion": {"max_velocity_pan_dps": -5}},
    {"motion": {"max_accel_dps2": float("nan")}},
    {"motion": {"limit_margin_deg": -1}},
    {"motion": {"typo_key": 1}},
    {"hardware": {"mode": "real"}},
    {"hardware": {"pan_sign": 2}},
    {"hardware": {"hardware_arm": "yes"}},
    {"face": {"servo": {"smoothing_alpha": 0}}},
    {"face": {"servo": {"dead_zone": 1.5}}},
    {"face": {"servo": {"horizontal_fov_deg": 200}}},
    {"hand": {"hand_side": "middle"}},
    {"sound": {"mic_spacing_m": 0}},
    {"sound": {"band_hz": [3000, 300]}},
    {"sound": {"mics_mounted_on": "tail"}},
    {"tick_rate_hz": 0},
    {"unknown_section": {}},
    {"motion": []},
])
def test_invalid_config_is_rejected(bad):
    with pytest.raises(HeadTrackingConfigError):
        config_from_dict(bad)


# ----------------------------------------------------- hardware gating
def test_default_config_never_selects_real_hardware(fake_kit):
    # even when buddy.yaml itself says "physical neck enabled"
    system = build_head_system(physical_buddy(), config_from_dict({}),
                               env=ARMED)
    assert system.mode == "mock"
    assert isinstance(system.backend, MockNeckBackend)
    assert not hasattr(fake_kit, "init_args")


def test_hardware_mode_lists_every_missing_gate():
    with pytest.raises(HeadTrackingConfigError) as err:
        build_head_system(load_buddy_config(),
                          config_from_dict({"hardware": {
                              "mode": "hardware"}}), env={})
    text = str(err.value)
    for needle in ("hardware_arm", "calibration_confirmed", "pan_sign",
                   "hardware.enabled", "simulation", "neck.enabled",
                   ARM_ENV):
        assert needle in text


@pytest.mark.parametrize("missing", [
    "arm", "calibration", "pan_sign", "tilt_sign", "buddy_enabled",
    "buddy_simulation", "buddy_neck", "env",
])
def test_each_single_missing_gate_blocks_hardware(missing, fake_kit):
    buddy, env = physical_buddy(), dict(ARMED)
    over = {}
    if missing == "arm":
        over["hardware_arm"] = False
    elif missing == "calibration":
        over["calibration_confirmed"] = False
    elif missing == "pan_sign":
        over["pan_sign"] = None
    elif missing == "tilt_sign":
        over["tilt_sign"] = None
    elif missing == "buddy_enabled":
        buddy["hardware"]["enabled"] = False
    elif missing == "buddy_simulation":
        buddy["hardware"]["simulation"] = True
    elif missing == "buddy_neck":
        buddy["hardware"]["neck"]["enabled"] = False
    elif missing == "env":
        env = {}
    with pytest.raises(HeadTrackingConfigError):
        build_head_system(buddy, armed_ht(**over), env=env)
    assert not hasattr(fake_kit, "init_args")        # never even opened


def test_environment_value_must_be_exactly_one():
    for value in ("0", "true", "", "yes"):
        assert hardware_gate_problems(physical_buddy(), armed_ht(),
                                      {ARM_ENV: value})


def test_all_gates_open_selects_the_pca9685_backend(fake_kit):
    system = build_head_system(physical_buddy(), armed_ht(), env=ARMED)
    assert system.mode == "hardware"
    assert isinstance(system.backend, PCA9685NeckBackend)
    assert fake_kit.init_args == (16, 0x40, 50.0)
    assert hardware_gate_problems(physical_buddy(), armed_ht(), ARMED) == []


# ------------------------------------------------- mock PCA9685 commands
def test_pca9685_receives_calibrated_channel_angle_commands(fake_kit):
    buddy = physical_buddy()
    system = build_head_system(buddy, armed_ht(), env=ARMED)
    assert system.start(0.0)
    pan_ch = buddy["hardware"]["neck"]["pan"]["channel"]
    tilt_ch = buddy["hardware"]["neck"]["tilt"]["channel"]
    assert (pan_ch, 135.0) in fake_kit.log
    assert (tilt_ch, 177.0) in fake_kit.log           # inside the margin
    co = system.coordinator
    t = 0.0
    for _ in range(100):
        co.submit(Priority.HAND, 170.0, 100.0, t)
        t += 0.02
        co.update(t)
    pan_cmds = [a for ch, a in fake_kit.log if ch == pan_ch and a]
    tilt_cmds = [a for ch, a in fake_kit.log if ch == tilt_ch and a]
    n = buddy["hardware"]["neck"]
    assert max(pan_cmds) <= n["pan"]["max_angle_deg"] - 3.0
    assert min(pan_cmds) >= n["pan"]["min_angle_deg"] + 3.0
    assert max(tilt_cmds) <= n["tilt"]["max_angle_deg"] - 3.0
    assert min(tilt_cmds) >= n["tilt"]["min_angle_deg"] + 3.0
    assert max(pan_cmds) > 150.0 and min(tilt_cmds) < 177.0


def test_hardware_shutdown_parks_releases_and_deinits(fake_kit):
    buddy = physical_buddy()
    system = build_head_system(buddy, armed_ht(), env=ARMED)
    system.start(0.0)
    system.coordinator.submit(Priority.HAND, 170.0, 177.0, 0.0)
    for i in range(1, 40):
        system.coordinator.update(i * 0.02)
    system.shutdown(sleep=lambda s: None)
    pan_ch = buddy["hardware"]["neck"]["pan"]["channel"]
    pan = [a for ch, a in fake_kit.log if ch == pan_ch]
    assert pan[-1] is None                      # released (no PWM)
    parked = [a for a in pan if a is not None][-1]
    assert parked == pytest.approx(135.0, abs=0.7)
    assert fake_kit.deinit_called


def test_hardware_disconnection_during_tracking_latches_a_fault(fake_kit):
    system = build_head_system(physical_buddy(), armed_ht(), env=ARMED)
    system.start(0.0)
    mgr, co = system.manager, system.coordinator
    t = 0.0
    for _ in range(12):
        t += 0.05
        mgr.on_face_frame(frame(t, face("a", 0.7, 0.5)), t)
        mgr.tick(t)
    assert mgr.state is HeadState.FACE_TRACKING
    fake_kit.fail["on"] = True                  # cable pulled
    for _ in range(20):
        t += 0.05
        mgr.on_face_frame(frame(t, face("a", 0.9, 0.5)), t)
        mgr.tick(t)
    assert mgr.state is HeadState.FAULT
    assert co.faulted and "failed" in co.fault_reason.lower()
    log_len = len(fake_kit.log)
    for _ in range(20):
        t += 0.05
        mgr.on_face_frame(frame(t, face("a", 0.9, 0.5)), t)
        mgr.tick(t)
    assert len(fake_kit.log) == log_len         # no retry storm
    system.shutdown(sleep=lambda s: None)       # must not raise


def test_missing_servokit_library_is_a_clean_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "adafruit_servokit", None)
    from buddy_core.neck.base import BuddyNeckError
    with pytest.raises(BuddyNeckError):
        build_head_system(physical_buddy(), armed_ht(), env=ARMED)

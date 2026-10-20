"""Validated configuration for BUDDY head tracking.

Every number below is a conservative STARTING VALUE, not a measured one.
Mechanical travel, servo direction, camera field of view and microphone
spacing must be measured on the real robot (see
``docs/head_tracking_physical_validation.md``).  Unknown keys are
rejected so a typo in a safety setting cannot silently fall back to a
default.
"""

from __future__ import annotations

import dataclasses
import math
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class HeadTrackingConfigError(ValueError):
    """Invalid or unsafe head-tracking configuration."""


@dataclass
class HardwareGate:
    """All of these must be satisfied before real servos can move."""

    mode: str = "mock"                   # "mock" | "hardware"
    hardware_arm: bool = False           # operator deliberately arms it
    calibration_confirmed: bool = False  # limits + directions measured
    pan_sign: int | None = None          # +1 / -1, REQUIRED in hardware
    tilt_sign: int | None = None


@dataclass
class MotionConfig:
    max_velocity_pan_dps: float = 45.0
    max_velocity_tilt_dps: float = 30.0
    max_accel_dps2: float = 150.0
    dead_zone_deg: float = 0.6           # ignore target changes below this
    min_step_deg: float = 0.3            # do not write smaller steps
    limit_margin_deg: float = 3.0        # stay inside calibrated limits
    request_ttl_s: float = 0.6           # a target expires after this
    neutral_after_s: float = 6.0         # no requests -> slew to neutral
    max_dt_s: float = 0.25               # clamp stalled-loop time steps
    release_on_emergency_stop: bool = False
    release_on_fault: bool = True
    shutdown_timeout_s: float = 4.0
    history_s: float = 2.0


@dataclass
class ServoConfig:
    """Visual-servo settings shared by face and hand following."""

    kp: float = 0.5
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = 0.3
    dead_zone: float = 0.08              # normalised error, -1..1
    dead_zone_exit_ratio: float = 0.6    # hysteresis
    smoothing_alpha: float = 0.5         # 1.0 = no smoothing
    max_step_deg: float = 6.0            # per observation
    horizontal_fov_deg: float = 60.0     # ASSUMED, measure the camera
    vertical_fov_deg: float = 45.0       # ASSUMED, measure the camera


@dataclass
class FaceConfig:
    servo: ServoConfig = field(default_factory=ServoConfig)
    min_face_px: float = 24.0
    min_confidence: float = 0.4
    acquire_frames: int = 3              # frames before locking a target
    lost_hold_s: float = 1.5             # hold pose this long, then lost
    max_obs_age_s: float = 0.5           # ignore older frames
    vertical_aim: float = 0.5            # 0 = top of box, 1 = bottom
    assoc_margin: float = 1.5            # caller-association ratio


@dataclass
class HandConfig:
    servo: ServoConfig = field(
        default_factory=lambda: ServoConfig(
            kp=0.6, dead_zone=0.06, max_step_deg=8.0))
    hand_side: str = "auto"              # "left" | "right" | "auto"
    prefer_fingertip: bool = True
    allow_wrist_fallback: bool = True
    min_confidence: float = 0.5
    lost_hold_s: float = 1.0
    max_active_s: float = 30.0
    max_obs_age_s: float = 0.5
    activation_gestures: list[str] = field(
        default_factory=lambda: ["raised_hand"])
    deactivation_gestures: list[str] = field(
        default_factory=lambda: ["both_hands_up"])
    require_face_target_match: bool = True


@dataclass
class SoundConfig:
    enabled: bool = False                # needs verified stereo hardware
    mic_spacing_m: float | None = None   # MUST be measured, no default
    speed_of_sound_mps: float = 343.0
    delay_offset_s: float = 0.0          # calibrated inter-mic offset
    swap_channels: bool = False
    mics_mounted_on: str = "head"        # "head" | "body"
    band_hz: list[float] = field(default_factory=lambda: [300.0, 3400.0])
    interp: int = 8
    min_window_s: float = 0.12
    min_rms: float = 0.005               # full scale = 1.0
    max_clip_fraction: float = 0.01
    min_snr_ratio: float = 2.0           # used when noise_rms is given
    min_peak: float = 0.12
    min_psr: float = 6.0                 # peak-to-background ratio
    min_peak_ratio: float = 1.5          # best vs. second source peak
    good_peak: float = 0.5               # confidence saturation points
    good_psr: float = 15.0
    good_ratio: float = 3.0
    min_confidence: float = 0.5
    geometry_tolerance: float = 0.15     # allowed excess over d/c
    max_abs_azimuth_deg: float = 75.0
    agree_deg: float = 15.0              # consistency filter
    min_agree: int = 2
    filter_window_s: float = 1.5
    pan_gain: float = 1.0                # head degrees per azimuth degree


@dataclass
class StateConfig:
    listening_timeout_s: float = 4.0
    orient_timeout_s: float = 3.0
    orient_tolerance_deg: float = 4.0
    turn_cooldown_s: float = 2.5
    speaking_cooldown_s: float = 0.8
    speaker_priority_s: float = 8.0
    confirm_center_norm: float = 0.25
    confirm_azimuth_deg: float = 20.0
    passive_face_tracking: bool = True
    return_to_neutral_on_loss: bool = True


@dataclass
class IdleConfig:
    enabled: bool = False
    amplitude_deg: float = 8.0
    period_s: float = 12.0
    quiet_before_s: float = 5.0          # only after this much calm


@dataclass
class HeadTrackingConfig:
    hardware: HardwareGate = field(default_factory=HardwareGate)
    motion: MotionConfig = field(default_factory=MotionConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    sound: SoundConfig = field(default_factory=SoundConfig)
    state: StateConfig = field(default_factory=StateConfig)
    idle: IdleConfig = field(default_factory=IdleConfig)
    tick_rate_hz: float = 30.0


def _is_dc(tp: Any) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _check(value: Any, tp: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is typing.Union or str(origin) == "<class 'types.UnionType'>":
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        return _check(value, inner[0], where)
    if tp is bool:
        if not isinstance(value, bool):
            raise HeadTrackingConfigError(f"{where}: expected bool")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise HeadTrackingConfigError(f"{where}: expected int")
        return value
    if tp is float:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise HeadTrackingConfigError(
                f"{where}: expected a finite number")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise HeadTrackingConfigError(f"{where}: expected string")
        return value
    if origin is list:
        if not isinstance(value, list):
            raise HeadTrackingConfigError(f"{where}: expected list")
        return [_check(v, args[0], where) for v in value]
    return value


def _build(cls: Any, data: Any, where: str) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise HeadTrackingConfigError(f"{where}: expected a mapping")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise HeadTrackingConfigError(
            f"{where}: unknown key(s) {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        tp = hints[key]
        path = f"{where}.{key}"
        kwargs[key] = (_build(tp, value, path) if _is_dc(tp)
                       else _check(value, tp, path))
    return cls(**kwargs)


def _need(ok: bool, msg: str) -> None:
    if not ok:
        raise HeadTrackingConfigError(msg)


def validate(cfg: HeadTrackingConfig) -> HeadTrackingConfig:
    """Range checks.  Raises HeadTrackingConfigError."""

    g, m = cfg.hardware, cfg.motion
    _need(g.mode in ("mock", "hardware"), "hardware.mode: mock|hardware")
    for name in ("pan_sign", "tilt_sign"):
        v = getattr(g, name)
        _need(v in (None, 1, -1), f"hardware.{name} must be 1 or -1")
    for name in ("max_velocity_pan_dps", "max_velocity_tilt_dps",
                 "max_accel_dps2", "request_ttl_s", "neutral_after_s",
                 "max_dt_s", "shutdown_timeout_s", "history_s"):
        _need(getattr(m, name) > 0, f"motion.{name} must be > 0")
    for name in ("dead_zone_deg", "min_step_deg", "limit_margin_deg"):
        _need(getattr(m, name) >= 0, f"motion.{name} must be >= 0")
    for label, sv in (("face.servo", cfg.face.servo),
                      ("hand.servo", cfg.hand.servo)):
        _need(0 < sv.smoothing_alpha <= 1, f"{label}.smoothing_alpha")
        _need(0 <= sv.dead_zone < 1, f"{label}.dead_zone in [0,1)")
        _need(0 < sv.dead_zone_exit_ratio <= 1, f"{label}.exit_ratio")
        _need(sv.kp >= 0 and sv.ki >= 0 and sv.kd >= 0, f"{label} gains")
        _need(sv.max_step_deg > 0, f"{label}.max_step_deg > 0")
        _need(0 < sv.horizontal_fov_deg < 180, f"{label} hfov")
        _need(0 < sv.vertical_fov_deg < 180, f"{label} vfov")
    _need(cfg.hand.hand_side in ("left", "right", "auto"),
          "hand.hand_side: left|right|auto")
    s = cfg.sound
    _need(s.mics_mounted_on in ("head", "body"), "sound.mics_mounted_on")
    _need(len(s.band_hz) == 2 and 0 < s.band_hz[0] < s.band_hz[1],
          "sound.band_hz must be [low, high]")
    _need(s.interp >= 1, "sound.interp >= 1")
    _need(s.mic_spacing_m is None or s.mic_spacing_m > 0,
          "sound.mic_spacing_m must be > 0")
    _need(0 < s.max_abs_azimuth_deg < 90, "sound.max_abs_azimuth_deg")
    _need(cfg.tick_rate_hz > 0, "tick_rate_hz > 0")
    return cfg


def config_from_dict(data: dict[str, Any] | None) -> HeadTrackingConfig:
    """Build and validate a config from a plain mapping."""

    return validate(_build(HeadTrackingConfig, data, "head_tracking"))


def load_head_tracking_config(
        path: str | Path | None = None) -> HeadTrackingConfig:
    """Load ``config/head_tracking.yaml`` (or ``path``)."""

    if path is None:
        here = Path(__file__).resolve()
        for parent in here.parents:
            cand = parent / "config" / "head_tracking.yaml"
            if cand.is_file():
                path = cand
                break
        else:
            return config_from_dict({})
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return config_from_dict(raw.get("head_tracking", raw))

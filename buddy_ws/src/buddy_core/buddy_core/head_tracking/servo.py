"""Visual-servo controller: image error -> head angle correction.

Error convention (matches the image):  ``ex > 0`` means the target is to
the RIGHT of the frame centre, ``ey > 0`` means BELOW the centre; both in
[-1, 1].  Converting that to a servo direction needs ``pan_sign`` and
``tilt_sign`` (+1: increasing servo angle moves the view toward image
right / down).  Those depend on how the servos were mounted and are
measured, never assumed, in hardware mode.
"""

from __future__ import annotations

from dataclasses import dataclass

from buddy_core.head_tracking.config import ServoConfig


class AxisServo:
    """One axis: smoothing, dead zone with hysteresis, PID, step limit."""

    def __init__(self, cfg: ServoConfig, fov_deg: float) -> None:
        self.cfg = cfg
        self.half_fov = fov_deg / 2.0
        self.reset()

    def reset(self) -> None:
        self._smooth: float | None = None
        self._integral = 0.0
        self._prev: float | None = None
        self._engaged = False

    def step(self, error: float, dt: float) -> float:
        """Return the correction in degrees (positive = toward +error)."""

        c = self.cfg
        if self._smooth is None:
            self._smooth = error
        else:
            a = c.smoothing_alpha
            self._smooth = a * error + (1.0 - a) * self._smooth
        e = self._smooth

        limit = c.dead_zone if not self._engaged else (
            c.dead_zone * c.dead_zone_exit_ratio)
        self._engaged = abs(e) > limit
        if not self._engaged:
            self._integral = 0.0
            self._prev = e
            return 0.0

        dt = max(dt, 1e-3)
        self._integral = max(-c.integral_limit, min(
            c.integral_limit, self._integral + e * dt))
        deriv = 0.0 if self._prev is None else (e - self._prev) / dt
        self._prev = e
        out = (c.kp * e + c.ki * self._integral + c.kd * deriv)
        out *= self.half_fov
        return max(-c.max_step_deg, min(c.max_step_deg, out))


@dataclass
class ServoOutput:
    d_pan_deg: float
    d_tilt_deg: float
    ex: float
    ey: float


class VisualServo2D:
    """Pan and tilt visual servo around a target point in the image."""

    def __init__(self, cfg: ServoConfig, pan_sign: int = 1,
                 tilt_sign: int = 1) -> None:
        self.pan_sign = pan_sign
        self.tilt_sign = tilt_sign
        self.pan_axis = AxisServo(cfg, cfg.horizontal_fov_deg)
        self.tilt_axis = AxisServo(cfg, cfg.vertical_fov_deg)

    def reset(self) -> None:
        self.pan_axis.reset()
        self.tilt_axis.reset()

    def step(self, cx_px: float, cy_px: float, width: int, height: int,
             dt: float) -> ServoOutput:
        ex = (cx_px / width - 0.5) * 2.0
        ey = (cy_px / height - 0.5) * 2.0
        ex = max(-1.0, min(1.0, ex))
        ey = max(-1.0, min(1.0, ey))
        dp = self.pan_sign * self.pan_axis.step(ex, dt)
        dt_ = self.tilt_sign * self.tilt_axis.step(ey, dt)
        return ServoOutput(dp, dt_, ex, ey)

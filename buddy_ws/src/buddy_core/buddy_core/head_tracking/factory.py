"""Build a head-tracking system with explicit hardware gating.

Mock mode ALWAYS uses ``MockNeckBackend``, whatever ``buddy.yaml`` says.
Real servos move only when every gate below is satisfied; a missing gate
raises ``HeadTrackingConfigError`` listing what is missing.  There is no
partial or silent fallback to hardware.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from buddy_core.head_tracking.config import (
    HeadTrackingConfig,
    HeadTrackingConfigError,
)
from buddy_core.head_tracking.manager import HeadTrackingManager
from buddy_core.head_tracking.models import HeadEvent
from buddy_core.head_tracking.motion import HeadMotionCoordinator
from buddy_core.neck.controller import NeckController
from buddy_core.neck.factory import get_neck_backend
from buddy_core.neck.mock import MockNeckBackend

ARM_ENV = "BUDDY_HEAD_HARDWARE_ARMED"


def hardware_gate_problems(buddy_cfg: dict[str, Any],
                           ht: HeadTrackingConfig,
                           env: dict[str, str] | None = None) -> list[str]:
    """Everything still missing before real servos may move."""

    env = os.environ if env is None else env
    hw = buddy_cfg.get("hardware", {})
    neck = hw.get("neck", {})
    g = ht.hardware
    problems = []
    if g.mode != "hardware":
        problems.append("head_tracking.hardware.mode is not 'hardware'")
    if not g.hardware_arm:
        problems.append("head_tracking.hardware.hardware_arm is false")
    if not g.calibration_confirmed:
        problems.append("head_tracking.hardware.calibration_confirmed is "
                        "false (limits and directions not measured)")
    if g.pan_sign is None or g.tilt_sign is None:
        problems.append("pan_sign / tilt_sign not set (measure them)")
    if hw.get("enabled") is not True:
        problems.append("buddy.yaml hardware.enabled is not true")
    if hw.get("simulation") is not False:
        problems.append("buddy.yaml hardware.simulation is not false")
    if neck.get("enabled") is not True:
        problems.append("buddy.yaml hardware.neck.enabled is not true")
    if env.get(ARM_ENV) != "1":
        problems.append(f"environment variable {ARM_ENV}=1 is not set")
    return problems


@dataclass
class HeadTrackingSystem:
    mode: str
    backend: Any
    neck: NeckController
    coordinator: HeadMotionCoordinator
    manager: HeadTrackingManager

    def start(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return self.coordinator.start(now)

    def shutdown(self, **kw) -> None:
        self.coordinator.shutdown(**kw)


def build_head_system(
    buddy_cfg: dict[str, Any],
    ht: HeadTrackingConfig,
    on_event: Callable[[HeadEvent], None] | None = None,
    env: dict[str, str] | None = None,
) -> HeadTrackingSystem:
    if ht.hardware.mode == "hardware":
        problems = hardware_gate_problems(buddy_cfg, ht, env)
        if problems:
            raise HeadTrackingConfigError(
                "refusing to enable physical head movement: "
                + "; ".join(problems))
        backend = get_neck_backend(buddy_cfg)
        if isinstance(backend, MockNeckBackend):
            raise HeadTrackingConfigError(
                "hardware mode requested but the neck factory returned "
                "the mock backend")
        mode = "hardware"
    else:
        backend = MockNeckBackend()
        mode = "mock"

    neck = NeckController(backend, buddy_cfg)
    ps = ht.hardware.pan_sign if ht.hardware.pan_sign else 1
    ts = ht.hardware.tilt_sign if ht.hardware.tilt_sign else 1
    co = HeadMotionCoordinator(neck, ht.motion, pan_sign=ps, tilt_sign=ts)
    mgr = HeadTrackingManager(co, ht, on_event=on_event)
    return HeadTrackingSystem(mode, backend, neck, co, mgr)

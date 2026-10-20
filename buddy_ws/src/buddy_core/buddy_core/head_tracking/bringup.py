"""Manual bring-up tool for the physical head (safe, small steps).

    python -m buddy_core.head_tracking.bringup limits
    python -m buddy_core.head_tracking.bringup center
    python -m buddy_core.head_tracking.bringup nudge --axis pan --deg 5

Used to MEASURE what this code must not assume: which way each servo
turns the camera (``pan_sign`` / ``tilt_sign``) and where the mechanical
stops are.  Goes through the same gates and coordinator as the real
system; ``--deg`` is capped at 10 per call, and motion is slew-limited.
Mock mode unless every hardware gate is satisfied.
"""

from __future__ import annotations

import argparse
import sys
import time

from buddy_core.head_tracking.config import (
    HeadTrackingConfigError,
    load_head_tracking_config,
)
from buddy_core.head_tracking.factory import build_head_system
from buddy_core.head_tracking.models import Priority

MAX_NUDGE_DEG = 10.0


def run(argv: list[str], buddy_cfg: dict | None = None,
        sleep=time.sleep, out=print) -> int:
    ap = argparse.ArgumentParser(prog="head-bringup")
    ap.add_argument("action", choices=["limits", "center", "nudge"])
    ap.add_argument("--axis", choices=["pan", "tilt"], default="pan")
    ap.add_argument("--deg", type=float, default=0.0)
    ap.add_argument("--config", default=None,
                    help="head_tracking.yaml (default: repo template)")
    ns = ap.parse_args(argv)

    if buddy_cfg is None:
        from buddy_core.config_loader import load_buddy_config
        buddy_cfg = load_buddy_config()
    try:
        ht = load_head_tracking_config(ns.config)
        system = build_head_system(buddy_cfg, ht)
    except HeadTrackingConfigError as exc:
        out(f"REFUSED: {exc}")
        return 2

    co = system.coordinator
    out(f"mode: {system.mode}")
    out(f"pan  range {co.pan_lo:.1f}..{co.pan_hi:.1f} "
        f"neutral {co.pan_neutral:.1f}")
    out(f"tilt range {co.tilt_lo:.1f}..{co.tilt_hi:.1f} "
        f"neutral {co.tilt_neutral:.1f}")
    if ns.action == "limits":
        out("(limits come from buddy.yaml hardware.neck; they are only "
            "as good as the last measurement)")
        return 0
    if abs(ns.deg) > MAX_NUDGE_DEG:
        out(f"REFUSED: --deg is capped at {MAX_NUDGE_DEG}")
        return 2

    ok = system.start(0.0)
    t = 0.0
    try:
        if not ok:
            out(f"FAULT: {co.fault_reason}")
            return 3
        if ns.action == "nudge":
            pan = co.pan + (ns.deg if ns.axis == "pan" else 0.0)
            tilt = co.tilt + (ns.deg if ns.axis == "tilt" else 0.0)
            end = 0.0
            while end < 3.0 and not co.faulted:
                t += 0.02
                end += 0.02
                co.submit(Priority.IDLE, pan, tilt, t)
                co.update(t)
                sleep(0.02)
            out(f"commanded {ns.axis} {ns.deg:+.1f} deg -> pose "
                f"pan {co.pan:.1f} tilt {co.tilt:.1f}")
            out("Watch the CAMERA VIEW: did it move toward image "
                "right/down for a positive step? record pan_sign / "
                "tilt_sign (+1 yes, -1 no)")
        return 0 if not co.faulted else 3
    finally:
        system.shutdown(sleep=sleep)


def main() -> int:
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

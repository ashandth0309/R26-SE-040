"""Microphone check: can these microphones localise sound at all?

    python -m buddy_core.head_tracking.mic_check --list
    python -m buddy_core.head_tracking.mic_check --device 2 --spacing 0.15

Records a few seconds from ONE input device with TWO channels, then
reports per-channel level, clipping and a series of direction estimates.
It never writes audio to disk.  Two separate USB microphones appear as two
devices with independent clocks and CANNOT be used this way; only one
device that exposes two channels shares a clock.

Read the numbers honestly: a stable estimate while someone talks from a
KNOWN position is evidence; a number printed with no ground truth is not.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from buddy_core.head_tracking.config import SoundConfig
from buddy_core.head_tracking.sound_direction import SoundDirectionEstimator


def analyze_capture(data: np.ndarray, fs: int, spacing_m: float,
                    window_s: float = 0.25) -> dict:
    """Pure analysis of a (samples, channels) float array in [-1, 1]."""

    if data.ndim != 2 or data.shape[1] < 2:
        return {"usable": False,
                "verdict": "need a device with 2 input channels; two "
                           "separate devices are not synchronised"}
    x, y = data[:, 0], data[:, 1]
    cfg = SoundConfig(enabled=True, mic_spacing_m=spacing_m)
    est = SoundDirectionEstimator(cfg)
    step = int(window_s * fs)
    results = []
    for i in range(0, len(x) - step + 1, step):
        r = est.estimate(x[i:i + step], y[i:i + step], fs, True)
        results.append(r)
    ok = [r for r in results if r.ok]
    az = [r.azimuth_deg for r in ok]
    reasons: dict[str, int] = {}
    for r in results:
        if not r.ok:
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
    report = {
        "usable": True,
        "windows": len(results),
        "accepted": len(ok),
        "rms_ch0": float(np.sqrt(np.mean(x * x))),
        "rms_ch1": float(np.sqrt(np.mean(y * y))),
        "clipped": float(np.mean(np.abs(data[:, :2]) >= 0.999)),
        "azimuth_median_deg": float(np.median(az)) if az else None,
        "azimuth_spread_deg": float(np.std(az)) if len(az) > 1 else None,
        "rejected_reasons": reasons,
        "resolution_deg": ok[0].resolution_deg if ok else None,
    }
    if not ok:
        report["verdict"] = ("no usable direction estimate; do not "
                             "enable sound localisation")
    elif report["azimuth_spread_deg"] is not None and \
            report["azimuth_spread_deg"] > 15:
        report["verdict"] = ("estimates are unstable (spread > 15 deg); "
                             "do not trust voice direction yet")
    else:
        report["verdict"] = ("plausible; now repeat from known positions "
                             "(left/centre/right) and compare")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mic-check")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--spacing", type=float, default=None,
                    help="measured centre-to-centre distance in metres")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--rate", type=int, default=16000)
    ns = ap.parse_args(argv)
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        print(f"sounddevice unavailable: {exc}")
        return 2
    if ns.list:
        for i, d in enumerate(sd.query_devices()):
            print(i, d["name"], "in:", d["max_input_channels"])
        return 0
    if ns.device is None or ns.spacing is None:
        print("--device and --spacing (measured!) are required")
        return 2
    data = sd.rec(int(ns.seconds * ns.rate), samplerate=ns.rate,
                  channels=2, device=ns.device, dtype="float32")
    sd.wait()
    for k, v in analyze_capture(np.asarray(data), ns.rate,
                                ns.spacing).items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

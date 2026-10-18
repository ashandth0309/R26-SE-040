"""Measure the voice engine on the machine it runs on.

Run on the Raspberry Pi to get real numbers:

    python3 -m buddy_core.voice.benchmark --out benchmark_results/pi.json

Nothing is assumed or invented: every figure printed was measured by this
run, on this machine, and the report says which machine that was.  Numbers
from a development PC say nothing about the Pi.  The software-only parts
(VAD, features, parsing, the stop path with fake backends) need no models;
with ``--vosk-model`` and ``--piper-model`` it also times the real
recogniser and synthesiser.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

RATE = 16000
FRAME = 320

# Starting targets for judging Pi results.  They are NOT measurements and
# are not met or missed until the benchmark runs on the Pi.
TARGETS = {
    "vad_realtime_factor_max": 0.05,
    "stop_path_ms_max": 50.0,
    "feature_realtime_factor_max": 0.5,
}


def _timeit(fn: Callable[[], object], repeats: int) -> list[float]:
    out = []

    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        out.append((time.perf_counter() - start) * 1000.0)

    return out


def _summary(ms: list[float]) -> dict:
    ordered = sorted(ms)

    return {
        "n": len(ms),
        "mean_ms": statistics.fmean(ms),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max_ms": ordered[-1],
    }


def machine_info() -> dict:
    """Describe the machine so results are never misattributed."""

    temp = None

    try:
        raw = Path("/sys/class/thermal/thermal_zone0/temp").read_text()
        temp = int(raw.strip()) / 1000.0
    except (OSError, ValueError):
        pass

    model = None

    try:
        model = Path("/proc/device-tree/model").read_text().strip("\0\n ")
    except OSError:
        pass

    import os

    return {
        "machine": platform.machine(),
        "system": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "device_tree_model": model,
        "cpu_temp_c": temp,
        "is_raspberry_pi": bool(model and "Raspberry" in model),
    }


def _speech(seconds: float) -> np.ndarray:
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    sig = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 6))
    env = 0.5 * (1 + np.sin(2 * np.pi * 3 * t - np.pi / 2))
    rng = np.random.default_rng(0)
    out = 0.15 * sig / np.abs(sig).max() * env + \
        rng.standard_normal(n) * 0.002

    return (np.clip(out, -1, 1) * 32767).astype(np.int16)


def bench_vad(seconds: float = 10.0) -> dict:
    """Per-frame VAD cost and its real-time factor."""

    from buddy_core.voice.vad import VoiceActivityDetector

    audio = _speech(seconds)
    frames = [audio[i:i + FRAME] for i in range(0, audio.size - FRAME,
                                                FRAME)]
    vad = VoiceActivityDetector()
    times = []

    for frame in frames:
        start = time.perf_counter()
        vad.process(frame)
        times.append((time.perf_counter() - start) * 1000.0)

    result = _summary(times)
    result["realtime_factor"] = result["mean_ms"] / 20.0

    return result


def bench_features(repeats: int = 20) -> dict:
    """Cost of acoustic features for a 3 s utterance."""

    from buddy_core.voice.emotion_audio import extract_features

    audio = _speech(3.0)
    result = _summary(_timeit(lambda: extract_features(audio), repeats))
    result["realtime_factor"] = result["mean_ms"] / 3000.0

    return result


def bench_parse(repeats: int = 2000) -> dict:
    """Command parsing cost."""

    from buddy_core.voice.voice_commands import parse_command

    return _summary(_timeit(lambda: parse_command("buddy go forward"),
                            repeats))


def bench_dialogue(repeats: int = 500) -> dict:
    """Rule-based reply cost (no model)."""

    from buddy_core.voice.dialogue_manager import (
        ConversationContext,
        DialogueManager,
    )

    dm = DialogueManager()
    ctx = ConversationContext()

    return _summary(_timeit(lambda: dm.respond("how are you", ctx),
                            repeats))


def bench_stop_path(repeats: int = 200) -> dict:
    """Software latency from the stop decision to the STOP publish.

    Uses fake audio backends, so it excludes recognition and hardware.
    """

    from buddy_core.voice.engine_factory import (
        build_companion,
        simulated_backends,
    )
    from buddy_core.voice.command_router import RecordingSink
    from buddy_core.voice.config import VoiceSettings

    settings = VoiceSettings()
    settings.memory.enabled = False
    sink = RecordingSink()
    comp = build_companion(settings, simulated_backends(), sink)
    times = []

    for _ in range(repeats):
        before = len(sink.movement)
        start = time.perf_counter()
        comp.engine._do_stop("benchmark")
        elapsed = (time.perf_counter() - start) * 1000.0

        if len(sink.movement) > before:
            times.append(elapsed)

    return _summary(times) if times else {"n": 0}


def bench_real(vosk_model: str, piper_model: str) -> dict:
    """Time the real recogniser and synthesiser (needs the models)."""

    from vosk import Model

    from buddy_core.voice.stt import VoskRecognizer
    from buddy_core.voice.tts_backends import PiperSynthesizer, SpeechStyle

    out: dict = {}
    start = time.perf_counter()
    model = Model(vosk_model)
    out["vosk_load_s"] = time.perf_counter() - start
    rec = VoskRecognizer(model)
    audio = _speech(3.0)
    frames = [audio[i:i + FRAME] for i in range(0, audio.size - FRAME,
                                                FRAME)]
    start = time.perf_counter()

    for frame in frames:
        rec.accept(frame.tobytes())

    rec.finalize()
    out["vosk_realtime_factor"] = (time.perf_counter() - start) / 3.0
    start = time.perf_counter()
    synth = PiperSynthesizer(piper_model)
    synth.synthesize("warm up", SpeechStyle())
    out["piper_load_s"] = time.perf_counter() - start
    times = _timeit(lambda: synth.synthesize("Going forward.",
                                             SpeechStyle()), 5)
    out["piper_short_phrase_ms"] = _summary(times)

    return out


def check_targets(report: dict) -> list[str]:
    """Targets the report misses.  Meaningful only on the Pi."""

    misses = []
    vad = report["vad"]["realtime_factor"]
    feat = report["features"]["realtime_factor"]
    stop = report["stop_path"].get("p95_ms")

    if vad > TARGETS["vad_realtime_factor_max"]:
        misses.append(f"VAD real-time factor {vad:.3f}")

    if feat > TARGETS["feature_realtime_factor_max"]:
        misses.append(f"feature real-time factor {feat:.3f}")

    if stop is None or stop > TARGETS["stop_path_ms_max"]:
        misses.append(f"stop path p95 {stop}")

    return misses


def run(vosk_model: str = "", piper_model: str = "") -> dict:
    """Run every benchmark and return the report."""

    report = {
        "machine": machine_info(),
        "vad": bench_vad(),
        "features": bench_features(),
        "parse": bench_parse(),
        "dialogue_rules": bench_dialogue(),
        "stop_path": bench_stop_path(),
    }

    if vosk_model and piper_model:
        report["real_models"] = bench_real(vosk_model, piper_model)

    report["peak_rss_mb"] = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss / 1024.0
    report["targets"] = TARGETS
    report["targets_missed"] = check_targets(report)
    report["note"] = (
        "Measured on the machine above. Only a run where "
        "machine.is_raspberry_pi is true describes the robot.")

    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vosk-model", default="")
    parser.add_argument("--piper-model", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    report = run(args.vosk_model, args.piper_model)
    text = json.dumps(report, indent=2)
    print(text)

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

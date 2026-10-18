"""Measure the vision stack on the machine it runs on.

    python3 -m buddy_core.vision.benchmark --out benchmark_results/pi.json
    python3 -m buddy_core.vision.benchmark --image photo.jpg --seconds 20

Everything reported was measured with ``perf_counter`` on this machine.
Only a report where ``machine.is_raspberry_pi`` is true describes the
robot.  The ``targets`` block holds goals to aim for, not results.  The
default test picture is a public-domain portrait that ships with
scikit-image, or random noise if that is missing (noise measures speed
only, not detection quality).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from buddy_core.vision import models
from buddy_core.vision.resources import read_temperature_c

# goals, not results
TARGETS = {
    "face_detect_p95_ms_max": 60.0,
    "pipeline_frame_p95_ms_max": 150.0,
    "pose_pass_p95_ms_max": 250.0,
    "peak_rss_mb_max": 900.0,
    "cpu_temp_c_max": 75.0,
}


def machine_info() -> dict:
    model = None

    try:
        model = Path("/proc/device-tree/model").read_text().strip("\0\n ")
    except OSError:
        pass

    return {
        "machine": platform.machine(), "system": platform.platform(),
        "python": platform.python_version(), "cpu_count": os.cpu_count(),
        "device_tree_model": model, "cpu_temp_c": read_temperature_c(),
        "is_raspberry_pi": bool(model and "Raspberry" in model),
        "opencv": cv2.__version__,
    }


def summary(ms: list[float]) -> dict:
    s = sorted(ms)

    return {"n": len(ms), "mean_ms": statistics.fmean(ms),
            "p50_ms": s[len(s) // 2],
            "p95_ms": s[min(len(s) - 1, int(0.95 * len(s)))],
            "max_ms": s[-1]}


def timeit(fn, repeats: int, warmup: int = 2) -> dict:
    for _ in range(warmup):
        fn()

    out = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)

    return summary(out)


def test_image(path: str | None) -> tuple[np.ndarray, str]:
    if path:
        img = cv2.imread(path)

        if img is None:
            raise SystemExit(f"cannot read {path}")

        return img, f"file:{Path(path).name}"

    try:
        from skimage import data
        im = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)

        return cv2.resize(im, (640, 480)), "skimage astronaut (portrait)"
    except Exception:
        rng = np.random.default_rng(0)

        return rng.integers(0, 255, (480, 640, 3), np.uint8), \
            "random noise (speed only)"


def stage_benchmarks(img: np.ndarray, repeats: int, threads: int) -> dict:
    from buddy_core.vision.embedding import SFaceEmbedder
    from buddy_core.vision.expression import ExpressionClassifier
    from buddy_core.vision.faces import YuNetDetector
    from buddy_core.vision.object_embedding import ObjectEmbedder
    from buddy_core.vision.objects import NanoDetDetector
    from buddy_core.vision.pose import PoseEstimator
    out: dict = {}
    h, w = img.shape[:2]
    small = cv2.resize(img, (320, int(h * 320 / w)))

    def guard(name, fn):
        try:
            out[name] = fn()
        except models.ModelMissing as exc:
            out[name] = {"skipped": str(exc)}
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}

    det = {}

    def face():
        d = YuNetDetector(str(models.model_path("face_detector")))
        det["faces"] = d.detect(small)

        return timeit(lambda: d.detect(small), repeats)

    guard("face_detect_320", face)

    def embed():
        e = SFaceEmbedder(str(models.model_path("face_embedder")))
        d = [x.scaled(w / 320) for x in det["faces"]]

        if not d:
            return {"skipped": "no face found in the test image"}

        return timeit(lambda: e.embed(img, d[0]), repeats)

    guard("face_embed", embed)

    def expression():
        c = ExpressionClassifier(str(models.model_path("expression")),
                                 threads=threads)
        d = [x.scaled(w / 320) for x in det["faces"]]

        if not d:
            return {"skipped": "no face found in the test image"}

        return timeit(lambda: c.predict(img, d[0]), repeats)

    guard("expression", expression)

    def objects():
        o = NanoDetDetector(threads=threads, exclude_person=False)

        return timeit(lambda: o.detect(img), repeats)

    guard("object_detect", objects)

    def pose():
        p = PoseEstimator(threads=threads)
        box = (0, 0, w, h)
        up = timeit(lambda: p.estimate(img, box, try_sideways=False),
                    repeats)
        side = timeit(lambda: p.estimate(img, box, try_sideways=True),
                      max(2, repeats // 3))

        return {"upright_one_pass": up, "with_sideways_passes": side}

    guard("pose", pose)

    def obj_embed():
        e = ObjectEmbedder(threads=threads)

        return timeit(lambda: e.embed(img), repeats)

    guard("object_embed", obj_embed)

    return out


def pipeline_benchmark(img: np.ndarray, profile: str, seconds: float,
                       threads: int, fps: float = 15.0) -> dict:
    from buddy_core.vision.vision_factory import build_vision
    from buddy_core.vision.vision_settings import VisionSettings

    v = build_vision(settings=VisionSettings(profile=profile),
                     in_memory=True, threads=threads)
    frame_ms: list[float] = []
    ran: dict[str, int] = {}
    t_wall = time.perf_counter()
    cpu0 = time.process_time()
    sim = 0.0
    temp0 = read_temperature_c()

    while time.perf_counter() - t_wall < seconds:
        t0 = time.perf_counter()
        ctx = v.manager.process(img, sim)
        frame_ms.append((time.perf_counter() - t0) * 1000.0)

        for s in ctx.stages_run:
            ran[s] = ran.get(s, 0) + 1

        sim += 1.0 / fps

    wall = time.perf_counter() - t_wall
    cpu = time.process_time() - cpu0

    return {"profile": profile, "frames": len(frame_ms),
            "frames_per_second_processed": len(frame_ms) / wall,
            "frame": summary(frame_ms), "stage_runs": ran,
            "cpu_seconds_per_wall_second": cpu / wall,
            "cpu_temp_c_before": temp0,
            "cpu_temp_c_after": read_temperature_c(),
            "unavailable": v.unavailable,
            "note": "frames are fed as fast as they can be processed; "
                    "the stack schedules stages by simulated time at "
                    f"{fps:g} fps"}


def check_targets(report: dict) -> list[str]:
    misses = []
    f = report.get("stages", {}).get("face_detect_320", {})

    if "p95_ms" in f and f["p95_ms"] > TARGETS["face_detect_p95_ms_max"]:
        misses.append(f"face detect p95 {f['p95_ms']:.1f} ms")

    for r in report.get("pipelines", []):
        if r["frame"]["p95_ms"] > TARGETS["pipeline_frame_p95_ms_max"]:
            misses.append(f"{r['profile']} frame p95 "
                          f"{r['frame']['p95_ms']:.1f} ms")

    if report["peak_rss_mb"] > TARGETS["peak_rss_mb_max"]:
        misses.append(f"peak RSS {report['peak_rss_mb']:.0f} MB")

    return misses


def run(image: str | None = None, repeats: int = 15, seconds: float = 10.0,
        threads: int = 2, profiles=("lightweight", "balanced")) -> dict:
    img, what = test_image(image)
    report = {"machine": machine_info(), "test_image": what,
              "threads": threads, "stages": stage_benchmarks(
                  img, repeats, threads), "pipelines": []}

    for prof in profiles:
        report["pipelines"].append(
            pipeline_benchmark(img, prof, seconds, threads))

    report["peak_rss_mb"] = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss / 1024.0
    report["targets"] = TARGETS
    report["targets_missed"] = check_targets(report)
    report["note"] = (
        "Measured on the machine above. Only a run where "
        "machine.is_raspberry_pi is true describes the robot. Detection "
        "quality is not measured here.")

    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default=None)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--profiles", default="lightweight,balanced")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    rep = run(a.image, a.repeats, a.seconds, a.threads,
              tuple(a.profiles.split(",")))
    text = json.dumps(rep, indent=2)
    print(text)

    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

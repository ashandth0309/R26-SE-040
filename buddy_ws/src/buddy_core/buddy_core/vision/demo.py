"""Offline demonstration: run the vision stack on images or a video file.

    python -m buddy_core.vision.demo --image portrait.jpg --repeat 20
    python -m buddy_core.vision.demo --video clip.mp4 --profile balanced

No camera, no ROS, no network, and nothing is saved: the face store and
object store live in memory and are discarded on exit.  Prints one JSON
line per processed frame (people, objects, events) and a summary.  A
missing model leaves its stage out and is reported, never hidden.

Frame time is simulated from ``--fps`` so schedules behave as on a robot;
the reported timings are real wall-clock on whatever machine this runs on.
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2

from buddy_core.vision.camera_sources import CameraError, VideoFileSource
from buddy_core.vision.vision_factory import build_vision
from buddy_core.vision.vision_node import VisionNodeCore
from buddy_core.vision.vision_settings import VisionSettings


class SimClock:
    t = 1000.0

    def __call__(self) -> float:
        return self.t


def frames_from(args):
    for path in args.image or ():
        img = cv2.imread(path)

        if img is None:
            raise SystemExit(f"cannot read image: {path}")

        for _ in range(args.repeat):
            yield img

    if args.video:
        src = VideoFileSource(args.video, loop=False)

        try:
            src.open()
        except CameraError as exc:
            raise SystemExit(str(exc))

        try:
            while True:
                try:
                    yield src.read()
                except CameraError:
                    break
        finally:
            src.close()


def run(args, out=sys.stdout) -> dict:
    clock = SimClock()
    built = build_vision(
        settings=VisionSettings(profile=args.profile),
        models_dir=args.models_dir or None, clock=clock, in_memory=True,
        threads=args.threads)
    core = VisionNodeCore(built, max_process_hz=0.0, clock=clock)
    n = events = 0

    for img in frames_from(args):
        clock.t += 1.0 / args.fps
        o = core.on_frame(img)
        n += 1

        if o is None:
            continue

        events += len(o.events)

        if args.max_frames and n > args.max_frames:
            break

        out.write(json.dumps({
            "frame": n, "people": len(o.people),
            "objects": [x["label"] for x in (o.objects or {}).get(
                "objects", [])],
            "events": [e["kind"] for e in o.events]}) + "\n")

    summary = {"frames": n, "events": events,
               "unavailable_stages": dict(built.unavailable),
               "status": core.status(),
               "note": "nothing was saved; this is not a Raspberry Pi "
                       "measurement unless run on the Pi"}
    out.write(json.dumps(summary) + "\n")

    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", action="append")
    ap.add_argument("--video")
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--profile", default="balanced",
                    choices=("lightweight", "balanced", "enhanced"))
    ap.add_argument("--models-dir", default="")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args(argv)

    if not args.image and not args.video:
        ap.error("give --image or --video")

    run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

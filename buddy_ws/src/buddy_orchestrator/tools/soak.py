"""Long-running mock soak (Test Q) - SANDBOX ONLY.

Runs the in-process mock system (see ``test/mock_system.py``) for a chosen
amount of SIMULATED time with a repeating mix of conversation, vision
questions, navigation, patrol, stop and emergency stop, and reports what
was measured: wall time, CPU time, peak Python memory and queue sizes.

These numbers describe this process on the machine that ran it.  They are
NOT Raspberry Pi 4 figures and say nothing about real audio, camera or
motor load.

    python3 tools/soak.py --minutes 10
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent

for p in (ROOT / "test", ROOT, SRC / "buddy_core",
          SRC / "buddy_core" / "test", SRC / "buddy_navigation"):
    sys.path.insert(0, str(p))

from mock_system import CUP_DETECTION, System  # noqa: E402

SCRIPT = [
    ("say", "hey buddy how are you"),
    ("say", "what is this"),
    ("say", "go forward"),
    ("run", 6.0),
    ("say", "go to the corner"),
    ("run", 20.0),
    ("say", "stop"),
    ("say", "start patrol"),
    ("run", 30.0),
    ("estop", 3.0),
    ("say", "cancel"),
    ("run", 10.0),
]


def soak(minutes: float) -> dict:
    s = System(objects=[CUP_DETECTION])
    tracemalloc.start()
    wall0, cpu0 = time.perf_counter(), time.process_time()
    end = s.t + minutes * 60.0
    cycles = 0
    peak_queue = 0
    peak_pending = 0
    heap_by_cycle = []
    harness_rows = 0

    while s.t < end:
        for kind, arg in SCRIPT:
            if kind == "say":
                s.say(arg)
            elif kind == "run":
                s.run(arg)
            elif kind == "estop":
                s.press_estop()
                s.run(arg)
                s.release_estop()

            peak_queue = max(peak_queue, s.comp.speaker.queued)
            peak_pending = max(peak_pending, len(s.coord.pending))

            if s.t >= end:
                break

        # keep the world clear of leftovers between cycles
        s.nav.handle(__import__(
            "buddy_navigation.core.service",
            fromlist=["CommandRequest"]).CommandRequest("cancel"), s.t)
        cycles += 1
        # The harness records every robot request, utterance and event for
        # the tests.  Count them, then drop them, so what is measured below
        # is the product code and not the recording.
        harness_rows += len(s.moves) + len(s.robot.history) \
            + len(s.spoken) + len(s.events)
        s.moves.clear()
        s.robot.history.clear()
        s.spoken.clear()
        s.events.clear()
        heap_by_cycle.append(tracemalloc.get_traced_memory()[0])

    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    _, peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    status = s.coord.status()

    warm = heap_by_cycle[min(2, len(heap_by_cycle) - 1)]
    growth = (heap_by_cycle[-1] - warm) / max(
        1, len(heap_by_cycle) - 1 - min(2, len(heap_by_cycle) - 1))

    return {
        "label": "MOCK / SANDBOX - not Raspberry Pi",
        "heap_mb_after_cycle_3": round(warm / 1e6, 2),
        "heap_mb_after_last_cycle": round(heap_by_cycle[-1] / 1e6, 2),
        "heap_growth_kb_per_cycle": round(growth / 1e3, 1),
        "harness_rows_recorded_and_dropped": harness_rows,
        "simulated_minutes": round((s.t - 1.5) / 60.0, 2),
        "script_cycles": cycles,
        "wall_seconds": round(wall, 1),
        "cpu_seconds": round(cpu, 1),
        "peak_python_heap_mb": round(peak_py / 1e6, 1),
        "max_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "peak_speaker_queue": peak_queue,
        "peak_pending_navigation": peak_pending,
        "final_pending_navigation": len(s.coord.pending),
        "crashes": len(s.crashes),
        "collisions": int(s.collided),
        "state_violations": status["state_violations"],
        "robot_state": status["robot_state"],
        "speech_sent": status["speech"]["sent"],
        "speech_dropped": status["speech"]["dropped"],
        "counters": status["counters"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--minutes", type=float, default=2.0)
    args = ap.parse_args()
    print(json.dumps(soak(args.minutes), indent=2))


if __name__ == "__main__":
    main()

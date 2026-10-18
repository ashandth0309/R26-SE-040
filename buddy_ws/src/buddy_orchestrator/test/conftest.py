"""Shared fixtures: a coordinator with recorders and a manual clock."""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
SRC = HERE.parents[2]

for p in (SRC / "buddy_orchestrator", SRC / "buddy_core"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from buddy_orchestrator.coordinator import BuddyCoordinator  # noqa: E402
from buddy_orchestrator.models import (  # noqa: E402
    NavStatusIn,
    ObstacleIn,
    PersonIn,
    SafetyIn,
)
from buddy_orchestrator.ports import Ports, VisionCommandResult  # noqa: E402
from buddy_orchestrator.profiles import get_profile  # noqa: E402


class FakeVision:
    """A scriptable vision port that records what it was asked."""

    def __init__(self):
        self.calls = []
        self.identify_result = {"success": True,
                                "text": "I think that's a cup."}
        self.teach_result = {"ok": True, "reply": "Got it.",
                             "phase": "done", "saved": True}
        self.where_result = {"found": True,
                             "text": "I last saw it on the desk."}
        self.command_result = VisionCommandResult(False)
        self.raises = None

    def _go(self, name, *args):
        self.calls.append((name, *args))

        if self.raises:
            raise self.raises

    def command(self, text, person_id=""):
        self._go("command", text)

        return self.command_result

    def identify(self):
        self._go("identify")

        return self.identify_result

    def teach(self, action, text):
        self._go("teach", action, text)

        return self.teach_result

    def where_is(self, name):
        self._go("where_is", name)

        return self.where_result


class Rig:
    def __init__(self, profile="offline", vision=True):
        self.t = 100.0
        self.said, self.nav, self.stops, self.relayed = [], [], [], []
        self.vision = FakeVision() if vision else None
        self.c = BuddyCoordinator(
            get_profile(profile),
            Ports(self.said.append, self.nav.append, self.stops.append,
                  self.relayed.append, self.vision),
            clock=lambda: self.t)

    @property
    def texts(self):
        return [u.text for u in self.said]

    def advance(self, dt):
        self.t += dt
        self.c.tick()

    def healthy(self, *, estop=False, allowed=True, motion=True,
                pose=True, sensors=True, nav_state="idle", rid="",
                cmd="", obstacle_degraded=False):
        c = self.c
        c.on_safety(SafetyIn(estop, allowed and not estop))
        c.on_obstacle(ObstacleIn(True, True, True, True,
                                 obstacle_degraded))
        c.on_nav_status(NavStatusIn(
            nav_state, request_id=rid, active_command=cmd,
            motion_enabled=motion, pose_trusted=pose,
            sensors_healthy=sensors))
        c.tick()

    def vision_up(self, camera=True):
        self.c.on_vision_status({"camera_ok": camera,
                                 "camera_enabled": True})

    def people(self, *people):
        self.c.on_people(list(people))


def known(name="Yohan", **kw):
    base = dict(person_key="p1", known=True, display_name=name,
                identity_state="recognized", posture="standing",
                bbox_x=270.0, bbox_width=100.0)
    base.update(kw)

    return PersonIn(**base)


@pytest.fixture
def rig():
    r = Rig()
    r.healthy()
    r.vision_up()

    return r


@pytest.fixture
def sim_rig():
    r = Rig("simulation")
    r.healthy()
    r.vision_up()

    return r

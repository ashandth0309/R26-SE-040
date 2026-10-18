"""Is Buddy ready?  Answered from evidence, not from processes existing.

``evaluate`` turns what the coordinator has actually *received* (fresh
safety state, fresh obstacle data, a navigation status that says motion is
enabled, no ownership conflicts) into a report.  A node being listed on the
graph proves nothing; ``ready_for_motion`` needs messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from buddy_orchestrator.hardware import Conflict
from buddy_orchestrator.models import NavStatusIn, ObstacleIn, SafetyIn
from buddy_orchestrator.profiles import Profile


@dataclass
class Capabilities:
    """Which features can be used right now."""

    microphone: bool = False
    speaker: bool = False
    camera: bool = False
    vision: bool = False
    obstacle_sensors: bool = False
    navigation: bool = False
    safety_supervisor: bool = False
    internet: bool = False

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    critical: bool = True


@dataclass
class ReadinessReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str,
            critical: bool = True) -> None:
        self.checks.append(Check(name, ok, detail, critical))

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    @property
    def ready_for_motion(self) -> bool:
        return all(c.ok for c in self.checks if c.name.startswith("motion:"))

    @property
    def summary(self) -> str:
        bad = [c for c in self.failures if c.critical]

        if not bad:
            return "ready"

        return "; ".join(f"{c.name}: {c.detail}" for c in bad)


def evaluate(*, profile: Profile,
             safety: SafetyIn | None, safety_fresh: bool,
             obstacle: ObstacleIn | None, obstacle_fresh: bool,
             nav: NavStatusIn | None, nav_fresh: bool,
             conflicts: list[Conflict],
             capabilities: Capabilities) -> ReadinessReport:
    """Build the readiness report.  All inputs are what was received."""

    r = ReadinessReport()
    r.add("motion:ownership", not conflicts,
          "; ".join(c.message for c in conflicts) or "one owner per device")
    r.add("motion:safety_supervisor", safety is not None and safety_fresh,
          "safety state is arriving" if safety_fresh
          else "no fresh /safety_state")
    r.add("motion:estop_clear",
          bool(safety_fresh and safety and not safety.emergency_stop_active),
          "emergency stop clear" if safety_fresh and safety
          and not safety.emergency_stop_active
          else "emergency stop active or unknown")
    r.add("motion:movement_allowed",
          bool(safety_fresh and safety and safety.movement_allowed),
          "supervisor allows movement" if safety_fresh and safety
          and safety.movement_allowed else "supervisor does not allow "
          "movement")
    r.add("motion:obstacle_sensors",
          bool(obstacle_fresh and obstacle and not obstacle.degraded),
          "all four sensors reporting" if obstacle_fresh and obstacle
          and not obstacle.degraded else "obstacle data stale or degraded")
    r.add("motion:navigation", nav is not None and nav_fresh,
          "navigation status is arriving" if nav_fresh
          else "no fresh /navigation/status")
    r.add("motion:navigation_motion_enabled",
          bool(nav_fresh and nav and nav.motion_enabled),
          "navigation motion enabled" if nav_fresh and nav
          and nav.motion_enabled else "navigation motion is disabled "
          "(allow_motion false)")
    r.add("motion:sensors_healthy",
          bool(nav_fresh and nav and nav.sensors_healthy),
          "navigation sees healthy sensors" if nav_fresh and nav
          and nav.sensors_healthy else "navigation reports degraded sensing")
    r.add("motion:profile",
          profile.physical_motion_allowed or profile.mock_hardware,
          f"profile '{profile.name}'")
    r.add("microphone", capabilities.microphone, "microphone active"
          if capabilities.microphone else "no active microphone",
          critical=False)
    r.add("speaker", capabilities.speaker, "speaker ok"
          if capabilities.speaker else "speaker failed or unknown",
          critical=False)
    r.add("camera", capabilities.camera, "camera streaming"
          if capabilities.camera else "camera off or failed",
          critical=False)

    return r

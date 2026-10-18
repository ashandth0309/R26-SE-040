"""Which nodes each BUDDY launch file starts, as plain data.

The six launch files turn these specs into ``launch_ros`` actions.  Keeping
the choice here means the rules (one camera node, one voice node, one
sensor driver, voice talking through the coordinator) are checked by unit
tests that need no ROS install.

A spec is a dict: package, executable, name, parameters.  Parameter values
may be ``("share", package, "relative/path")`` tuples, which the launch
file resolves to an installed path.
"""

from __future__ import annotations

from buddy_orchestrator.contracts import Topic

PROFILES = ("offline", "hybrid", "simulation")


def share(package: str, relative: str) -> tuple:
    return ("share", package, relative)


def _spec(package, executable, name, **parameters) -> dict:
    return {"package": package, "executable": executable, "name": name,
            "parameters": parameters}


def voice_node(profile: str, *, interposed: bool) -> dict:
    """The one node that owns the microphones and the speaker."""

    params = {
        "profile": "balanced",
        "enable_cloud": profile == "hybrid",
        "dry_run": profile == "simulation",
        "enable_navigation": interposed,
        "enable_gateway": interposed,
    }

    if interposed:
        # Voice motion requests go to the coordinator; only STOP also has a
        # direct route, so stopping never depends on the coordinator.
        params.update({
            "movement_topic": Topic.VOICE_MOVEMENT_INTAKE,
            "navigation_topic": Topic.VOICE_NAVIGATION_INTAKE,
            "direct_stop_topic": Topic.MOVEMENT_REQUEST,
            "direct_nav_stop_topic": Topic.NAV_COMMAND,
        })

    return _spec("buddy_core", "voice_companion",
                 "buddy_voice_companion", **params)


def vision_nodes(profile: str) -> list[dict]:
    """The one camera owner and the vision node that reads its images."""

    simulated = profile == "simulation"

    return [
        _spec("buddy_core", "vision_camera", "buddy_camera",
              backend="mock" if simulated else "picamera2"),
        _spec("buddy_core", "vision_node", "buddy_vision",
              profile="lightweight" if simulated else "balanced",
              in_memory=simulated),
    ]


def navigation_nodes(profile: str) -> list[dict]:
    """Sensing, safety path and navigation.  Exactly one sensor driver."""

    if profile == "simulation":
        return [
            _spec("buddy_navigation", "sim_node", "buddy_navigation_sim"),
            _spec("buddy_core", "safety_supervisor", "safety_supervisor"),
            _spec("buddy_navigation", "navigation_node",
                  "buddy_navigation",
                  config_file=share("buddy_navigation",
                                    "config/navigation_sim.yaml")),
        ]

    return [
        _spec("buddy_core", "obstacle_sensor", "buddy_obstacle_sensor"),
        _spec("buddy_core", "safety_supervisor", "safety_supervisor"),
        _spec("buddy_core", "motor_executor", "motor_executor"),
        _spec("buddy_navigation", "navigation_node", "buddy_navigation",
              config_file=share("buddy_navigation",
                                "config/navigation.yaml")),
    ]


def coordinator_node(profile: str, *, vision: bool) -> dict:
    return _spec("buddy_orchestrator", "buddy_coordinator",
                 "buddy_coordinator", profile=profile,
                 enable_vision=vision,
                 config_file=share("buddy_orchestrator",
                                   "config/orchestrator.yaml"))


def full_system(profile: str) -> list[dict]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile '{profile}'")

    return [*navigation_nodes(profile), *vision_nodes(profile),
            voice_node(profile, interposed=True),
            coordinator_node(profile, vision=True)]


def voice_only(profile: str = "offline") -> list[dict]:
    return [voice_node(profile, interposed=False)]


def vision_only(profile: str = "offline") -> list[dict]:
    return vision_nodes(profile)


def navigation_only(profile: str = "simulation") -> list[dict]:
    return navigation_nodes(profile)


def duplicate_names(specs: list[dict]) -> list[str]:
    seen, dupes = set(), []

    for s in specs:
        if s["name"] in seen:
            dupes.append(s["name"])

        seen.add(s["name"])

    return dupes

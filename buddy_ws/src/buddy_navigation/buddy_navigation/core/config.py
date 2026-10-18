"""Navigation configuration loading with strict validation.

Two profiles exist:

* ``physical``: values come from measuring the real robot.  Until
  ``geometry_confirmed`` is true, motion stays disabled no matter what else
  is configured.
* ``simulation``: the labelled assumptions in ``presets``.  Never use on the
  robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from buddy_navigation.core.coordinator import NavigationConfig
from buddy_navigation.core.grid import PlanningGrid
from buddy_navigation.core.obstacle_assessment import AssessmentConfig
from buddy_navigation.core.patrol import PatrolPoint, PatrolRoute
from buddy_navigation.core.presets import (
    simulation_assessment_config,
    simulation_footprint,
    simulation_sensor_specs,
)
from buddy_navigation.core.sensor_model import Footprint, SensorSpec

REQUIRED_SENSORS = ("front_left", "front_right", "rear_left", "rear_right")


class ConfigError(ValueError):
    """Raised for an invalid or unsafe configuration."""


@dataclass(frozen=True)
class NavigationSetup:
    """Everything the navigation node needs, validated."""

    profile: str
    footprint: Footprint
    sensors: tuple[SensorSpec, ...]
    assessment: AssessmentConfig
    navigation: NavigationConfig
    geometry_confirmed: bool
    pose_source: str
    default_timed_motion_s: float
    locations: dict[str, tuple[float, float]] = field(default_factory=dict)
    routes: dict[str, PatrolRoute] = field(default_factory=dict)
    planning_grid: PlanningGrid | None = None
    warnings: tuple[str, ...] = ()


def _require(mapping: dict, key: str, where: str):
    if key not in mapping:
        raise ConfigError(f"Missing '{key}' in {where}.")

    return mapping[key]


def _grid_from_config(cfg: dict | None) -> PlanningGrid | None:
    if not cfg:
        return None

    rows = _require(cfg, "rows", "grid")
    resolution = float(_require(cfg, "resolution_m", "grid"))

    if not rows or len({len(r) for r in rows}) != 1:
        raise ConfigError("grid.rows must be non-empty and rectangular.")

    height = len(rows)
    blocked = {
        (column, height - 1 - printed)
        for printed, text in enumerate(rows)
        for column, char in enumerate(text)
        if char == "#"
    }

    return PlanningGrid.from_blocked_cells(
        len(rows[0]),
        height,
        blocked,
        resolution,
        float(cfg.get("origin_x", 0.0)),
        float(cfg.get("origin_y", 0.0)),
    )


def parse_navigation_config(data: dict) -> NavigationSetup:
    """Validate a parsed YAML mapping."""

    root = data.get("buddy_navigation", data)
    profile = root.get("profile", "physical")
    warnings: list[str] = []

    if profile not in ("physical", "simulation"):
        raise ConfigError("profile must be 'physical' or 'simulation'.")

    nav = root.get("navigation", {})

    if profile == "simulation":
        footprint = simulation_footprint()
        sensors = simulation_sensor_specs()
        assessment = simulation_assessment_config()
        confirmed = True
    else:
        fp = _require(root, "footprint", "config")
        footprint = Footprint(
            float(_require(fp, "length_m", "footprint")),
            float(_require(fp, "width_m", "footprint")),
            float(fp.get("margin_m", 0.05)),
        )
        sensors = _sensors(_require(root, "sensors", "config"))
        assessment = AssessmentConfig(**root.get("assessment", {}))
        confirmed = bool(root.get("geometry_confirmed", False))

    if not confirmed:
        warnings.append(
            "geometry_confirmed is false: autonomous motion is disabled "
            "until sensor and body geometry have been measured."
        )

    wants_motion = bool(nav.get("allow_motion", False))
    allow_motion = wants_motion and confirmed

    if wants_motion and not confirmed:
        warnings.append("allow_motion was requested but is being ignored.")

    navigation = NavigationConfig(
        allow_motion=allow_motion,
        **{
            k: nav[k]
            for k in (
                "max_linear_speed",
                "max_rotation_speed",
                "request_valid_for_s",
                "goal_timeout_s",
                "blocked_timeout_s",
                "waypoint_tolerance_m",
                "no_progress_timeout_s",
            )
            if k in nav
        },
    )

    pose_source = str(root.get("pose_source", "none"))

    if pose_source not in ("none", "sim", "odom"):
        raise ConfigError("pose_source must be none, sim or odom.")

    if profile == "physical" and pose_source == "sim":
        raise ConfigError("pose_source 'sim' is not allowed on the robot.")

    locations = {
        str(name): (float(p["x"]), float(p["y"]))
        for name, p in (root.get("locations") or {}).items()
    }

    routes = {}

    for route_id, spec in (root.get("routes") or {}).items():
        points = tuple(
            PatrolPoint(str(p["name"]), float(p["x"]), float(p["y"]))
            for p in _require(spec, "points", f"route '{route_id}'")
        )
        routes[str(route_id)] = PatrolRoute(
            str(route_id), points, bool(spec.get("repeat", False))
        )

    return NavigationSetup(
        profile=profile,
        footprint=footprint,
        sensors=sensors,
        assessment=assessment,
        navigation=navigation,
        geometry_confirmed=confirmed,
        pose_source=pose_source,
        default_timed_motion_s=float(
            nav.get("default_timed_motion_s", 1.0)
        ),
        locations=locations,
        routes=routes,
        planning_grid=_grid_from_config(root.get("grid")),
        warnings=tuple(warnings),
    )


def _sensors(items: list[dict]) -> tuple[SensorSpec, ...]:
    specs = tuple(
        SensorSpec(
            name=str(_require(item, "name", "sensor")),
            x=float(_require(item, "x_m", "sensor")),
            y=float(_require(item, "y_m", "sensor")),
            yaw=math.radians(float(_require(item, "yaw_deg", "sensor"))),
            fov_rad=math.radians(float(item.get("fov_deg", 30.0))),
            min_range_m=float(item.get("min_range_m", 0.02)),
            max_range_m=float(item.get("max_range_m", 4.0)),
        )
        for item in items
    )
    names = [s.name for s in specs]

    if sorted(names) != sorted(REQUIRED_SENSORS):
        raise ConfigError(
            f"sensors must be exactly {list(REQUIRED_SENSORS)}, got {names}."
        )

    return specs


def load_navigation_config(path: str | Path) -> NavigationSetup:
    """Load and validate a YAML file."""

    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    return parse_navigation_config(data)

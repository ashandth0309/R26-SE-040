"""Tests for navigation configuration loading."""

from pathlib import Path

import pytest

from buddy_navigation.core.config import (
    ConfigError,
    load_navigation_config,
    parse_navigation_config,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_shipped_physical_config_keeps_motion_disabled():
    setup = load_navigation_config(CONFIG_DIR / "navigation.yaml")

    assert setup.profile == "physical"
    assert not setup.geometry_confirmed
    assert not setup.navigation.allow_motion
    assert setup.pose_source == "none"
    assert setup.assessment.required_coverage_linear == 1.0
    assert any("geometry_confirmed" in w for w in setup.warnings)
    assert {s.name for s in setup.sensors} == {
        "front_left", "front_right", "rear_left", "rear_right"}


def test_allow_motion_is_ignored_until_geometry_is_confirmed():
    data = {"buddy_navigation": {
        "profile": "physical",
        "geometry_confirmed": False,
        "navigation": {"allow_motion": True},
        "footprint": {"length_m": 0.3, "width_m": 0.2},
        "sensors": [
            {"name": n, "x_m": 0.1, "y_m": 0.0, "yaw_deg": 0.0}
            for n in ("front_left", "front_right",
                      "rear_left", "rear_right")],
    }}
    setup = parse_navigation_config(data)

    assert not setup.navigation.allow_motion
    assert any("ignored" in w for w in setup.warnings)

    data["buddy_navigation"]["geometry_confirmed"] = True

    assert parse_navigation_config(data).navigation.allow_motion


def test_simulation_config_loads_routes_locations_and_grid():
    setup = load_navigation_config(CONFIG_DIR / "navigation_sim.yaml")

    assert setup.profile == "simulation"
    assert setup.navigation.allow_motion
    assert setup.locations["far"] == (5.5, 5.5)
    assert [p.name for p in setup.routes["demo_loop"].points] == [
        "start", "corner", "far"]
    assert setup.planning_grid.width == 6
    assert not setup.planning_grid.is_free((1, 4))


def test_sim_pose_source_is_rejected_on_the_physical_profile():
    with pytest.raises(ConfigError):
        parse_navigation_config({"buddy_navigation": {
            "profile": "physical", "pose_source": "sim",
            "footprint": {"length_m": 0.3, "width_m": 0.2},
            "sensors": []}})


@pytest.mark.parametrize("data", [
    {"profile": "robot"},
    {"profile": "physical"},
    {"profile": "physical", "footprint": {"length_m": 0.3},
     "sensors": []},
    {"profile": "physical", "pose_source": "gps",
     "footprint": {"length_m": 0.3, "width_m": 0.2}, "sensors": []},
    {"profile": "simulation",
     "grid": {"resolution_m": 1.0, "rows": ["..", "."]}},
])
def test_invalid_configs_raise(data):
    with pytest.raises(ConfigError):
        parse_navigation_config({"buddy_navigation": data})


def test_sensor_names_must_match_the_four_corner_array():
    with pytest.raises(ConfigError):
        parse_navigation_config({"buddy_navigation": {
            "profile": "physical",
            "footprint": {"length_m": 0.3, "width_m": 0.2},
            "sensors": [{"name": "front_left", "x_m": 0, "y_m": 0,
                         "yaw_deg": 0}]}})

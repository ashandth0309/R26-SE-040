"""Launch composition rules, checked without ROS.

These verify the files are well formed and the node sets obey the
single-owner rules.  They do not start any node.
"""

import ast
import re
from pathlib import Path

import pytest

from buddy_orchestrator import launch_support as ls
from buddy_orchestrator.contracts import HARDWARE_OWNERS, Topic

PKG = Path(__file__).resolve().parents[1]
SRC = PKG.parent
LAUNCH = sorted((PKG / "launch").glob("*.launch.py"))
NAMES = {"full_offline", "full_hybrid", "full_simulation", "voice_only",
         "vision_only", "navigation_only"}


def entry_points(package):
    text = (SRC / package / "setup.py").read_text()

    return set(re.findall(r"'(\w+) = ", text))


def node_names(package):
    out = set()

    for f in (SRC / package).rglob("*.py"):
        if "test" in f.parts or "build" in f.parts:
            continue

        out.update(re.findall(r'super\(\).__init__\(\s*"(\w+)"',
                              f.read_text()))

    return out


ALL = {
    "full_offline": ls.full_system("offline"),
    "full_hybrid": ls.full_system("hybrid"),
    "full_simulation": ls.full_system("simulation"),
    "voice_only": ls.voice_only(),
    "vision_only": ls.vision_only(),
    "navigation_only": ls.navigation_only(),
}


def test_six_launch_files_exist():
    assert {p.name.split(".")[0] for p in LAUNCH} == NAMES


@pytest.mark.parametrize("path", LAUNCH, ids=lambda p: p.stem)
def test_launch_file_parses_and_defines_entry(path):
    tree = ast.parse(path.read_text())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert [f.name for f in fn] == ["generate_launch_description"]


@pytest.mark.parametrize("name", sorted(ALL))
def test_no_duplicate_node_names(name):
    assert ls.duplicate_names(ALL[name]) == []


@pytest.mark.parametrize("name", sorted(ALL))
def test_every_executable_is_a_real_entry_point(name):
    for spec in ALL[name]:
        assert spec["executable"] in entry_points(spec["package"]), spec


@pytest.mark.parametrize("name", sorted(ALL))
def test_node_names_match_the_code(name):
    for spec in ALL[name]:
        assert spec["name"] in node_names(spec["package"]), spec


@pytest.mark.parametrize("name", sorted(ALL))
def test_at_most_one_owner_per_resource(name):
    names = [s["name"] for s in ALL[name]]

    for resource, owners in HARDWARE_OWNERS.items():
        assert sum(names.count(o) for o in owners) <= 1, resource


@pytest.mark.parametrize("name", ["full_offline", "full_hybrid",
                                  "full_simulation"])
def test_full_system_has_every_part(name):
    names = {s["name"] for s in ALL[name]}
    assert {"buddy_voice_companion", "buddy_vision", "buddy_camera",
            "buddy_navigation", "safety_supervisor",
            "buddy_coordinator"} <= names


@pytest.mark.parametrize("name", ["full_offline", "full_hybrid"])
def test_hardware_profiles_use_real_sensor_and_motor_nodes(name):
    names = {s["name"] for s in ALL[name]}
    assert {"buddy_obstacle_sensor", "motor_executor"} <= names
    assert "buddy_navigation_sim" not in names


def test_simulation_has_no_hardware_nodes():
    names = {s["name"] for s in ALL["full_simulation"]}
    assert "buddy_obstacle_sensor" not in names
    assert "motor_executor" not in names
    assert "buddy_navigation_sim" in names
    voice = next(s for s in ALL["full_simulation"]
                 if s["name"] == "buddy_voice_companion")
    assert voice["parameters"]["dry_run"] is True


def test_exactly_one_sensor_driver_per_profile():
    for name, specs in ALL.items():
        drivers = [s for s in specs
                   if s["name"] in ("buddy_obstacle_sensor",
                                    "buddy_navigation_sim")]
        assert len(drivers) <= 1, name


def test_cloud_only_in_hybrid():
    for name in ("full_offline", "full_simulation"):
        v = next(s for s in ALL[name]
                 if s["name"] == "buddy_voice_companion")
        assert v["parameters"]["enable_cloud"] is False

    v = next(s for s in ALL["full_hybrid"]
             if s["name"] == "buddy_voice_companion")
    assert v["parameters"]["enable_cloud"] is True


def test_full_voice_is_interposed_but_stop_is_direct():
    v = next(s for s in ALL["full_offline"]
             if s["name"] == "buddy_voice_companion")["parameters"]
    assert v["movement_topic"] == Topic.VOICE_MOVEMENT_INTAKE
    assert v["navigation_topic"] == Topic.VOICE_NAVIGATION_INTAKE
    assert v["direct_stop_topic"] == Topic.MOVEMENT_REQUEST
    assert v["direct_nav_stop_topic"] == Topic.NAV_COMMAND
    assert v["enable_gateway"] is True


def test_voice_only_is_not_interposed():
    v = ALL["voice_only"][0]["parameters"]
    assert "movement_topic" not in v
    assert v["enable_gateway"] is False and v["enable_navigation"] is False


def test_voice_parameters_exist_in_the_voice_node():
    text = (SRC / "buddy_core/buddy_core/voice/voice_node.py").read_text()

    params = next(s for s in ALL["full_offline"]
                  if s["name"] == "buddy_voice_companion")["parameters"]

    for key in params:
        assert f'"{key}"' in text, key


def test_vision_camera_parameters_exist():
    cam = (SRC / "buddy_core/buddy_core/vision/camera_node.py").read_text()
    vis = (SRC / "buddy_core/buddy_core/vision/vision_node.py").read_text()

    for spec, text in zip(ls.vision_nodes("offline"), (cam, vis)):
        for key in spec["parameters"]:
            assert f'"{key}"' in text, key


def test_coordinator_parameters_exist_in_node():
    text = (PKG / "buddy_orchestrator/nodes/coordinator_node.py"
            ).read_text()

    for key in ls.coordinator_node("offline", vision=True)["parameters"]:
        assert f'"{key}"' in text, key


def test_config_files_referenced_by_launch_exist():
    for specs in ALL.values():
        for s in specs:
            for v in s["parameters"].values():
                if isinstance(v, tuple):
                    pkg = SRC / v[1]
                    assert (pkg / v[2]).exists(), v


def test_yaml_keys_are_known_parameters():
    import yaml

    data = yaml.safe_load((PKG / "config/orchestrator.yaml").read_text())
    text = (PKG / "buddy_orchestrator/nodes/coordinator_node.py"
            ).read_text()

    for key in data:
        assert f'"{key}"' in text, key


def test_launch_files_installed_by_setup():
    setup = (PKG / "setup.py").read_text()
    assert "launch/*.py" in setup and "config/*.yaml" in setup


def test_package_xml_lists_run_dependencies():
    xml = (PKG / "package.xml").read_text()

    for dep in ("buddy_core", "buddy_interfaces", "buddy_nav_interfaces",
                "buddy_navigation", "launch_ros"):
        assert dep in xml


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        ls.full_system("turbo")

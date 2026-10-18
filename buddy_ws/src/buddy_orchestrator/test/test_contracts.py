"""The contracts agree with the real messages, services and code."""

import re
from pathlib import Path

import pytest

from buddy_orchestrator import contracts as c
from buddy_orchestrator import models as m

SRC = Path(__file__).resolve().parents[2]
DOCS = SRC.parents[1] / "docs"
IFACE = SRC / "buddy_interfaces"
NAV = SRC / "buddy_nav_interfaces"


def constants(path, prefix):
    return {k: int(v) for k, v in re.findall(
        rf"^uint8 {prefix}(\w+)=(\d+)", path.read_text(), re.MULTILINE)}


def all_source():
    out = []

    for pkg in ("buddy_core", "buddy_navigation", "buddy_orchestrator"):
        for f in (SRC / pkg).rglob("*.py"):
            if "test" not in f.parts and "build" not in f.parts:
                out.append(f.read_text())

    return "\n".join(out)


def test_navigation_command_kinds_match_the_message():
    got = constants(NAV / "msg/NavigationCommand.msg", "COMMAND_")
    assert {v: k.lower() for k, v in got.items()} == {
        0: "stop", 1: "move_forward", 2: "turn_left", 3: "turn_right",
        4: "start_patrol", 5: "go_to_location", 6: "cancel", 7: "pause",
        8: "resume"} == m.NAV_KINDS


def test_navigation_states_match_the_message():
    got = constants(NAV / "msg/NavigationStatus.msg", "STATE_")
    assert {v: k.lower() for k, v in got.items()} == m.NAV_STATES


def test_movement_codes_match_the_message():
    got = constants(IFACE / "msg/MovementCommand.msg", "COMMAND_")
    assert got["STOP"] == m.MOVE_STOP
    assert got["FORWARD"] == m.MOVE_FORWARD
    assert got["ROTATE_RIGHT"] == m.MOVE_ROTATE_RIGHT


def test_speech_priorities_match_the_message():
    got = constants(IFACE / "msg/SpeechRequest.msg", "PRIORITY_")
    assert {k: int(v) for k, v in c.SpeechPriority.__members__.items()} \
        == got


def test_every_contract_topic_is_used_by_real_code():
    text = all_source()
    unused = [v for k, v in vars(c.Topic).items()
              if not k.startswith("_") and f'"{v}"' not in text]
    assert unused == []


def test_every_contract_service_is_used_by_real_code():
    text = all_source()
    unused = [v for k, v in vars(c.Service).items()
              if not k.startswith("_") and f'"{v}"' not in text]
    assert unused == []


def test_vision_services_match_srv_files():
    names = {p.stem for p in (IFACE / "srv").glob("*.srv")}
    assert {"VisionCommand", "IdentifyObject", "TeachObject", "WhereIs",
            "RouteUtterance"} <= names


def test_error_codes_are_unique_and_start_at_ok():
    values = [e.value for e in c.Err]
    assert len(values) == len(set(values)) and c.Err.OK == 0


def test_stale_limits_are_positive_and_obstacle_is_strictest():
    assert all(v > 0 for v in c.STALE_AFTER_S.values())
    assert c.STALE_AFTER_S["obstacle_status"] == min(
        c.STALE_AFTER_S.values())


def test_single_speaker_publisher_and_one_safe_motion_publisher():
    assert c.PUBLISHER_LIMITS[c.Topic.VOICE_SAY] == 1
    assert c.PUBLISHER_LIMITS[c.Topic.MOVEMENT_SAFE] == 1
    assert c.PUBLISHER_LIMITS[c.Topic.CAMERA_IMAGE] == 1


def test_interface_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", c.INTERFACE_VERSION)


@pytest.mark.skipif(not (DOCS / "buddy_integration_interfaces.md").exists(),
                    reason="docs not written yet")
def test_interface_document_lists_every_topic_service_and_the_version():
    text = (DOCS / "buddy_integration_interfaces.md").read_text()

    for group in (c.Topic, c.Service):
        for k, v in vars(group).items():
            if not k.startswith("_"):
                assert v in text, f"{v} missing from the interface doc"

    assert c.INTERFACE_VERSION in text


def test_route_utterance_srv_matches_the_wrapper_fields():
    srv = (IFACE / "srv/RouteUtterance.srv").read_text()

    for field in ("request_id", "text", "confidence", "claimed", "reply",
                  "intent", "error_code"):
        assert field in srv

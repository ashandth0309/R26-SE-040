"""One owner per device, checked from what is visible on the graph."""

from buddy_orchestrator import hardware as hw
from buddy_orchestrator.contracts import Topic


def test_clean_graph_has_no_conflicts():
    nodes = ["buddy_voice_companion", "buddy_camera", "buddy_vision",
             "buddy_obstacle_sensor", "motor_executor"]
    assert hw.audit(nodes, {Topic.VOICE_SAY: 1}) == []


def test_second_camera_node_is_a_conflict():
    found = hw.audit(["buddy_camera", "buddy_camera"])
    assert [f.resource for f in found] == ["camera"]


def test_legacy_voice_node_conflicts_with_the_companion():
    found = hw.audit(["buddy_voice", "buddy_voice_companion"])
    assert {f.resource for f in found} == {"microphones", "speaker"}


def test_legacy_voice_alone_is_not_a_conflict():
    assert hw.audit(["buddy_voice"]) == []


def test_two_speech_publishers_is_a_conflict():
    found = hw.audit([], {Topic.VOICE_SAY: 2})
    assert found and Topic.VOICE_SAY in found[0].message


def test_two_publishers_on_the_safe_motion_topic():
    found = hw.audit([], {Topic.MOVEMENT_SAFE: 2})
    assert found


def test_missing_owners_are_listed():
    gone = hw.missing_owners(["buddy_camera"])
    assert "microphones" in gone and "camera" not in gone


def test_two_ultrasonic_drivers_conflict():
    found = hw.audit(["buddy_obstacle_sensor", "buddy_obstacle_sensor"])
    assert [f.resource for f in found] == ["ultrasonic"]

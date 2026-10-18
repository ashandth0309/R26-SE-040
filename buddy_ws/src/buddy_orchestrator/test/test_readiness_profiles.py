"""Profiles and the motion readiness report."""

import pytest
from conftest import Rig

from buddy_orchestrator.profiles import PROFILES, get_profile
from buddy_orchestrator.readiness import Capabilities


def test_three_profiles_exist():
    assert set(PROFILES) == {"offline", "hybrid", "simulation"}


def test_only_hybrid_allows_cloud():
    assert [p.name for p in PROFILES.values() if p.cloud_allowed] == [
        "hybrid"]


def test_simulation_is_mock_and_cannot_move_hardware():
    p = get_profile("simulation")
    assert p.mock_hardware and p.mock_person_approach
    assert not p.physical_motion_allowed


def test_hardware_profiles_never_claim_person_approach():
    assert not get_profile("offline").mock_person_approach
    assert not get_profile("hybrid").mock_person_approach


def test_unknown_profile_is_an_error():
    with pytest.raises(ValueError):
        get_profile("production")


def test_ready_when_everything_is_healthy():
    r = Rig()
    r.healthy()
    assert r.c.readiness().ready_for_motion
    assert r.c.readiness().summary == "ready"


@pytest.mark.parametrize("kwargs,fragment", [
    ({"estop": True}, "emergency stop"),
    ({"allowed": False}, "does not allow"),
    ({"motion": False}, "disabled"),
    ({"sensors": False}, "degraded"),
    ({"obstacle_degraded": True}, "degraded"),
])
def test_each_motion_check_can_fail_alone(kwargs, fragment):
    r = Rig()
    r.healthy(**kwargs)
    report = r.c.readiness()
    assert not report.ready_for_motion
    assert fragment in report.summary.lower()


def test_nothing_received_is_not_ready():
    r = Rig()
    assert not r.c.readiness().ready_for_motion
    assert "safety" in r.c.readiness().summary


def test_hardware_conflict_blocks_readiness():
    r = Rig()
    r.healthy()
    r.c.graph_changed(["buddy_camera", "buddy_camera"], {})
    assert not r.c.readiness().ready_for_motion
    assert "camera" in r.c.status()["hardware_conflicts"][0]


def test_camera_microphone_speaker_do_not_gate_motion():
    r = Rig()
    r.healthy()
    r.c.capabilities = Capabilities()          # all off
    assert r.c.readiness().ready_for_motion


def test_status_reports_version_and_profile():
    r = Rig("hybrid")
    st = r.c.status()
    assert st["profile"] == "hybrid" and st["interface_version"]

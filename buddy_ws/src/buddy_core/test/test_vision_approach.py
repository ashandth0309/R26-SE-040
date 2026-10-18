"""Phase 9: approach requests go through a mock navigation adapter only."""

import pytest

from buddy_core.vision.approach import (
    ApproachConfig, ApproachController, ApproachRequest, ApproachStatus,
    MockNavigationAdapter, NavCapabilities)

GOOD = NavCapabilities(True, True, True, True, True, False)


def req(now=100.0, **kw):
    base = dict(now=now, person_key="p1", recognized=True, invited=True,
                bearing_deg=10.0, bearing_confidence=0.9,
                target_stamp=now - 0.2)
    base.update(kw)

    return ApproachRequest(**base)


def make(caps=GOOD, **cfg):
    a = MockNavigationAdapter(caps)

    return a, ApproachController(a, ApproachConfig(**cfg))


def test_default_hardware_capabilities_refuse_everything():
    a, c = make(NavCapabilities())
    d = c.request(req())
    assert not d.allowed and d.reason == "navigation_unavailable"
    assert a.sent == []


@pytest.mark.parametrize("field,reason", [
    ("distance_reliable", "distance_not_reliable"),
    ("localization_ok", "localization_not_reliable"),
    ("obstacle_avoidance_ok", "obstacle_avoidance_not_ready"),
    ("movement_allowed", "safety_not_allowing_movement"),
])
def test_each_missing_capability_blocks(field, reason):
    caps = NavCapabilities(**{**GOOD.__dict__, field: False})
    a, c = make(caps)
    d = c.request(req())
    assert not d.allowed and d.reason == reason and a.sent == []


def test_emergency_stop_blocks():
    caps = NavCapabilities(**{**GOOD.__dict__, "emergency_stop_active": True})
    a, c = make(caps)
    assert c.request(req()).reason == "safety_not_allowing_movement"


@pytest.mark.parametrize("kw,reason", [
    (dict(recognized=False), "person_not_recognised"),
    (dict(person_key=None), "person_not_recognised"),
    (dict(invited=False), "no_invitation"),
    (dict(quiet_mode=True), "quiet_mode"),
    (dict(person_settings={"approach": False}), "person_disabled_approach"),
    (dict(bearing_deg=None), "bearing_unreliable"),
    (dict(bearing_confidence=0.3), "bearing_unreliable"),
    (dict(target_stamp=90.0), "target_stale"),
    (dict(emergency=True), "emergency_in_progress"),
])
def test_request_refusals(kw, reason):
    a, c = make()
    d = c.request(req(**kw))
    assert not d.allowed and d.reason == reason and a.sent == []


def test_valid_request_sends_bearing_goal_only():
    a, c = make()
    d = c.request(req())
    assert d.allowed and len(a.sent) == 1
    g = a.sent[0]
    assert g.person_key == "p1" and g.bearing_deg == 10.0
    assert not hasattr(g, "linear") and not hasattr(g, "velocity")


def test_stop_distance_never_below_minimum():
    a, c = make()
    c.request(req(stop_distance_m=0.2))
    assert a.sent[0].stop_distance_m >= 1.0


def test_one_request_at_a_time_and_rate_limit():
    a, c = make()
    assert c.request(req()).allowed
    assert c.request(req(now=101.0)).reason == "already_approaching"
    c.cancel(102.0, "test")
    assert c.request(req(now=103.0)).reason == "too_soon"
    assert c.request(req(now=200.0)).allowed


def test_navigation_can_reject():
    a, c = make()
    a.accept = False
    d = c.request(req())
    assert not d.allowed and d.reason == "navigation_rejected"
    assert c.active is None


@pytest.mark.parametrize("kw,reason", [
    (dict(voice_stop=True), "voice_stop"),
    (dict(person_declined=True), "person_declined"),
    (dict(quiet_mode=True), "quiet_mode"),
    (dict(emergency=True), "emergency_in_progress"),
])
def test_cancel_conditions(kw, reason):
    a, c = make()
    c.request(req())
    assert c.update(101.0, True, **kw) == "cancelled"
    assert a.cancelled[-1][1] == reason and c.active is None


def test_person_lost_timeout_and_degraded_cancel():
    a, c = make()
    c.request(req())
    assert c.update(101.0, True) == "active"
    assert c.update(102.5, False) == "active"
    assert c.update(103.5, False) == "cancelled"
    assert a.cancelled[-1][1] == "person_lost"

    a, c = make(timeout_s=5.0)
    c.request(req())
    assert c.update(104.0, True) == "active"
    assert c.update(106.0, True) == "cancelled"
    assert a.cancelled[-1][1] == "timeout"

    a, c = make()
    c.request(req())
    a.caps = NavCapabilities(**{**GOOD.__dict__, "localization_ok": False})
    assert c.update(101.0, True) == "cancelled"
    assert a.cancelled[-1][1] == "navigation_degraded"

    a, c = make()
    c.request(req())
    a.caps = NavCapabilities(**{**GOOD.__dict__, "movement_allowed": False})
    c.update(101.0, True)
    assert a.cancelled[-1][1] == "safety_not_allowing_movement"


def test_success_clears_active_and_idle_update_is_safe():
    a, c = make()
    assert c.update(1.0, True) == "idle"
    g = c.request(req()).goal
    a.finish(g.request_id)
    assert c.update(101.0, True) == "succeeded" and c.active is None


def test_decisions_are_logged():
    a, c = make()
    c.request(req(invited=False))
    c.request(req())
    kinds = [k for _, k, _ in c.log]
    assert kinds == ["refused", "sent"]
    assert ApproachStatus.IDLE.value == "idle"

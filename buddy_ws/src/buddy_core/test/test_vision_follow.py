"""Phase 16: following policy (mock navigation only; off on hardware)."""

import pytest

from buddy_core.vision.approach import MockNavigationAdapter, NavCapabilities
from buddy_core.vision.follow import (
    FollowConfig, FollowController, FollowRequest, TargetObs,
    bearing_from_box)

GOOD = NavCapabilities(True, True, True, True, True, False, True)
SHAPE = (480, 640, 3)


def req(now=100.0, **kw):
    base = dict(now=now, requester_key="p1", requester_track="t1",
                recognized=True, consent=True)
    base.update(kw)

    return FollowRequest(**base)


def make(caps=GOOD, enabled=True, **cfg):
    a = MockNavigationAdapter(caps)

    return a, FollowController(a, FollowConfig(hardware_enabled=enabled,
                                               **cfg))


def obs(track="t1", key="p1", box=(280, 100, 360, 400), conf=0.9):
    return TargetObs(track, key, box, conf)


def test_default_config_refuses_on_hardware():
    a = MockNavigationAdapter(GOOD)
    c = FollowController(a)
    d = c.request(req())
    assert not d.allowed and d.reason == "follow_disabled_on_hardware"
    assert a.sent == []


def test_default_navigation_capabilities_refuse_even_if_enabled():
    a, c = make(NavCapabilities())
    assert c.request(req()).reason == "navigation_unavailable"


@pytest.mark.parametrize("field,reason", [
    ("follow_validated", "follow_not_validated"),
    ("distance_reliable", "distance_not_reliable"),
    ("localization_ok", "localization_not_reliable"),
    ("obstacle_avoidance_ok", "obstacle_avoidance_not_ready"),
    ("movement_allowed", "safety_not_allowing_movement"),
])
def test_each_capability_is_required(field, reason):
    a, c = make(NavCapabilities(**{**GOOD.__dict__, field: False}))
    assert c.request(req()).reason == reason and a.sent == []


@pytest.mark.parametrize("kw,reason", [
    (dict(recognized=False), "person_not_recognised"),
    (dict(requester_key=None), "person_not_recognised"),
    (dict(requester_track=None), "person_not_recognised"),
    (dict(consent=False), "no_consent_to_be_followed"),
    (dict(quiet_mode=True), "quiet_mode"),
    (dict(emergency=True), "emergency_in_progress"),
])
def test_request_refusals(kw, reason):
    a, c = make()
    assert c.request(req(**kw)).reason == reason


def test_valid_request_and_distance_floor():
    a, c = make()
    d = c.request(req(desired_distance_m=0.3))
    assert d.allowed and a.sent[0].desired_distance_m >= 1.0
    assert c.request(req(now=101.0)).reason == "already_following"


def test_follows_same_person_and_reports_bearing():
    a, c = make()
    c.request(req())
    fb = c.update(101.0, [obs()], SHAPE)
    assert fb.state == "following" and fb.target_visible
    assert abs(fb.bearing_deg) < 1.0
    right = c.update(102.0, [obs(box=(500, 100, 580, 400))], SHAPE)
    assert right.bearing_deg > 10


def test_never_switches_to_another_person():
    a, c = make()
    c.request(req())
    other = obs(track="t9", key="p2", conf=0.99)
    fb = c.update(101.0, [other], SHAPE)
    assert not fb.target_visible
    fb = c.update(105.0, [other], SHAPE)
    assert fb.state == "stopped" and fb.reason == "target_lost"
    assert a.cancelled[-1][1] == "target_lost"


def test_track_id_switch_is_followed_only_by_matching_identity():
    a, c = make()
    c.request(req())
    fb = c.update(101.0, [obs(track="t7", key="p1")], SHAPE)
    assert fb.target_visible and c.target_track == "t7"


def test_unrecognised_same_track_is_still_the_target():
    a, c = make()
    c.request(req())
    fb = c.update(101.0, [obs(track="t1", key=None)], SHAPE)
    assert fb.target_visible


def test_low_confidence_target_counts_as_lost():
    a, c = make()
    c.request(req())
    c.update(101.0, [obs(conf=0.2)], SHAPE)
    fb = c.update(105.0, [obs(conf=0.2)], SHAPE)
    assert fb.reason == "target_lost"


@pytest.mark.parametrize("kw,reason", [
    (dict(voice_stop=True), "voice_stop"),
    (dict(quiet_mode=True), "quiet_mode"),
    (dict(emergency=True), "emergency_in_progress")])
def test_stop_conditions(kw, reason):
    a, c = make()
    c.request(req())
    fb = c.update(101.0, [obs()], SHAPE, **kw)
    assert fb.state == "stopped" and fb.reason == reason and c.goal is None


def test_max_duration_and_navigation_degradation_and_end():
    a, c = make(max_duration_s=5.0)
    c.request(req())
    c.update(103.0, [obs()], SHAPE)
    assert c.update(106.0, [obs()], SHAPE).reason == "max_duration"
    a, c = make()
    c.request(req())
    a.caps = NavCapabilities(**{**GOOD.__dict__, "distance_reliable": False})
    assert c.update(101.0, [obs()], SHAPE).reason == "navigation_degraded"
    a, c = make()
    g = c.request(req()).goal
    a.finish(g.request_id, ok=False)
    assert c.update(101.0, [obs()], SHAPE).reason == "navigation_ended"


def test_too_close_is_reported_not_acted_on():
    a, c = make()
    c.request(req())
    fb = c.update(101.0, [obs(box=(100, 10, 540, 470))], SHAPE)
    assert fb.too_close and fb.state == "following"


def test_bearing_geometry():
    assert bearing_from_box((0, 0, 100, 100), 640) < -25
    assert bearing_from_box((540, 0, 640, 100), 640) > 25
    assert abs(bearing_from_box((270, 0, 370, 100), 640)) < 1.0


def test_idle_update_is_safe_and_goal_has_no_velocity():
    a, c = make()
    assert c.update(1.0, [], SHAPE).state == "idle"
    g = c.request(req()).goal
    assert not hasattr(g, "linear") and not hasattr(g, "velocity")

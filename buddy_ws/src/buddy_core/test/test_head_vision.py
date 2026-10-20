"""Tests for visual servoing, face following and hand following."""

from __future__ import annotations

import numpy as np
import pytest
from head_fakes import face, frame, gesture, hand_obs

from buddy_core.head_tracking.config import (
    FaceConfig,
    HandConfig,
    ServoConfig,
)
from buddy_core.head_tracking.face_follow import FaceFollower, TrackStatus
from buddy_core.head_tracking.hand_follow import (
    KIND_FINGERTIP,
    KIND_WRIST,
    HandFollower,
    HandStatus,
)
from buddy_core.head_tracking.servo import VisualServo2D

POSE = (90.0, 90.0)


def pose_at(_t):
    return POSE


def locked_follower(cfg=None, pan_sign=1, tilt_sign=1, track="a"):
    ff = FaceFollower(cfg or FaceConfig(), pan_sign, tilt_sign)
    t = 0.0
    for _ in range(3):
        t += 0.05
        ff.update(frame(t, face(track, 0.5, 0.5)), t, pose_at)
    assert ff.locked == track
    return ff, t


# ------------------------------------------------------------- servo
def test_servo_error_sign_convention():
    s = VisualServo2D(ServoConfig())
    out = s.step(0.9 * 640, 0.5 * 480, 640, 480, 0.05)
    assert out.ex > 0 and out.d_pan_deg > 0       # right of centre
    s.reset()
    out = s.step(0.5 * 640, 0.9 * 480, 640, 480, 0.05)
    assert out.ey > 0 and out.d_tilt_deg > 0      # below centre


def test_servo_step_is_limited_and_error_clamped():
    cfg = ServoConfig(kp=5.0, max_step_deg=4.0, smoothing_alpha=1.0)
    s = VisualServo2D(cfg)
    out = s.step(10_000, -10_000, 640, 480, 0.05)
    assert abs(out.d_pan_deg) <= 4.0 and abs(out.d_tilt_deg) <= 4.0
    assert out.ex == 1.0 and out.ey == -1.0


def test_servo_integral_is_bounded():
    cfg = ServoConfig(kp=0.0, ki=10.0, smoothing_alpha=1.0, max_step_deg=99,
                      integral_limit=0.2)
    s = VisualServo2D(cfg)
    outs = [s.step(0.9 * 640, 240, 640, 480, 0.1).d_pan_deg
            for _ in range(200)]
    assert max(outs) <= 10.0 * 0.2 * cfg.horizontal_fov_deg / 2 + 1e-6


# -------------------------------------------------------------- face
def test_horizontal_direction_right_then_left():
    ff, t = locked_follower()
    t += 0.05
    r = ff.update(frame(t, face("a", 0.85, 0.5)), t, pose_at)
    assert r.status is TrackStatus.TRACKING and r.pan > POSE[0]
    ff, t = locked_follower()
    t += 0.05
    r = ff.update(frame(t, face("a", 0.15, 0.5)), t, pose_at)
    assert r.pan < POSE[0]


def test_vertical_direction_below_then_above():
    ff, t = locked_follower()
    t += 0.05
    r = ff.update(frame(t, face("a", 0.5, 0.85)), t, pose_at)
    assert r.tilt > POSE[1]
    ff, t = locked_follower()
    t += 0.05
    r = ff.update(frame(t, face("a", 0.5, 0.15)), t, pose_at)
    assert r.tilt < POSE[1]


def test_servo_signs_invert_the_motion():
    ff, t = locked_follower(pan_sign=-1, tilt_sign=-1)
    t += 0.05
    r = ff.update(frame(t, face("a", 0.85, 0.85)), t, pose_at)
    assert r.pan < POSE[0] and r.tilt < POSE[1]


def test_centered_face_produces_no_correction():
    ff, t = locked_follower()
    t += 0.05
    r = ff.update(frame(t, face("a", 0.52, 0.49)), t, pose_at)
    assert (r.pan, r.tilt) == POSE


def test_dead_zone_and_hysteresis():
    cfg = FaceConfig(servo=ServoConfig(smoothing_alpha=1.0, dead_zone=0.1,
                                       dead_zone_exit_ratio=0.5))
    ff, t = locked_follower(cfg)
    # 0.06 normalised error (cx = 0.53) is inside the dead zone
    t += 0.05
    r = ff.update(frame(t, face("a", 0.53, 0.5)), t, pose_at)
    assert r.pan == POSE[0]
    # large error engages; 0.08 then stays engaged (above exit 0.05)
    t += 0.05
    ff.update(frame(t, face("a", 0.7, 0.5)), t, pose_at)
    t += 0.05
    r = ff.update(frame(t, face("a", 0.54, 0.5)), t, pose_at)
    assert r.pan > POSE[0]
    # falling under the exit threshold stops the correction
    t += 0.05
    r = ff.update(frame(t, face("a", 0.51, 0.5)), t, pose_at)
    assert r.pan == POSE[0]


def test_jitter_inside_dead_zone_causes_no_command():
    ff, t = locked_follower()
    rng = np.random.default_rng(0)
    corrections = []
    for _ in range(200):
        t += 0.05
        cx = 0.5 + rng.uniform(-0.03, 0.03)
        cy = 0.5 + rng.uniform(-0.03, 0.03)
        r = ff.update(frame(t, face("a", cx, cy)), t, pose_at)
        corrections.append((r.pan - POSE[0], r.tilt - POSE[1]))
    assert max(abs(c[0]) for c in corrections) == 0.0
    assert max(abs(c[1]) for c in corrections) == 0.0


def test_smoothing_reduces_noise_on_an_offset_target():
    rng = np.random.default_rng(1)
    noisy = [0.75 + rng.uniform(-0.05, 0.05) for _ in range(300)]

    def spread(alpha):
        cfg = FaceConfig(servo=ServoConfig(smoothing_alpha=alpha,
                                           max_step_deg=99.0))
        ff, t = locked_follower(cfg)
        out = []
        for cx in noisy:
            t += 0.05
            out.append(ff.update(frame(t, face("a", cx, 0.5)), t,
                                 pose_at).pan)
        return float(np.std(out))

    assert spread(0.3) < 0.6 * spread(1.0)


def test_correction_per_frame_is_limited():
    ff, t = locked_follower(FaceConfig(servo=ServoConfig(max_step_deg=3.0)))
    t += 0.05
    r = ff.update(frame(t, face("a", 0.99, 0.01)), t, pose_at)
    assert abs(r.pan - POSE[0]) <= 3.0 and abs(r.tilt - POSE[1]) <= 3.0


def test_target_is_computed_from_pose_at_capture_time():
    seen = []

    def spy(t):
        seen.append(t)
        return POSE

    ff, t = locked_follower()
    t += 0.05
    ff.update(frame(t - 0.02, face("a", 0.8, 0.5)), t, spy)
    assert seen == [t - 0.02]


def test_face_must_be_seen_for_several_frames_before_locking():
    ff = FaceFollower(FaceConfig(acquire_frames=3))
    t = 0.0
    for i in range(2):
        t += 0.05
        r = ff.update(frame(t, face("a", 0.5, 0.5)), t, pose_at)
        assert not ff.has_target and r.pan is None
    t += 0.05
    ff.update(frame(t, face("a", 0.5, 0.5)), t, pose_at)
    assert ff.has_target


def test_flickering_candidates_never_lock():
    ff = FaceFollower(FaceConfig(acquire_frames=3))
    t = 0.0
    for i in range(10):
        t += 0.05
        ff.update(frame(t, face("ab"[i % 2], 0.5, 0.5)), t, pose_at)
    assert not ff.has_target


def test_small_or_low_confidence_faces_are_ignored():
    ff = FaceFollower()
    t = 0.0
    for _ in range(6):
        t += 0.05
        ff.update(frame(t, face("tiny", 0.5, 0.5, size=10),
                        face("weak", 0.4, 0.5, conf=0.1)), t, pose_at)
    assert not ff.has_target


def test_multiple_faces_do_not_cause_target_jumping():
    ff, t = locked_follower()                       # locked on "a"
    picked = set()
    for i in range(60):
        t += 0.05
        # "b" is larger and closer to centre than "a" from now on
        r = ff.update(frame(t, face("a", 0.75, 0.5, size=60),
                            face("b", 0.5, 0.5, size=200)), t, pose_at)
        picked.add(r.track_id)
    assert picked == {"a"}


def test_multiple_faces_initial_choice_is_the_most_central_large_face():
    ff = FaceFollower()
    t = 0.0
    for _ in range(3):
        t += 0.05
        ff.update(frame(t, face("far", 0.1, 0.5), face("mid", 0.55, 0.5),
                        face("edge", 0.95, 0.5)), t, pose_at)
    assert ff.locked == "mid"


def test_target_missing_briefly_is_held_then_resumes():
    ff, t = locked_follower()
    t += 0.05
    r = ff.update(frame(t), t, pose_at)
    assert r.status is TrackStatus.HOLDING and r.pan is None
    t += 0.3
    r = ff.update(frame(t, face("a", 0.7, 0.5)), t, pose_at)
    assert r.status is TrackStatus.TRACKING


def test_target_lost_after_hold_time():
    cfg = FaceConfig(lost_hold_s=1.0)
    ff, t = locked_follower(cfg)
    status = None
    for _ in range(40):
        t += 0.05
        status = ff.update(frame(t), t, pose_at).status
        if status is TrackStatus.LOST:
            break
    assert status is TrackStatus.LOST
    assert not ff.has_target


def test_lost_target_is_replaced_only_after_acquisition():
    cfg = FaceConfig(lost_hold_s=0.5, acquire_frames=3)
    ff, t = locked_follower(cfg)
    ids = []
    for _ in range(30):
        t += 0.05
        r = ff.update(frame(t, face("other", 0.4, 0.5)), t, pose_at)
        ids.append(r.track_id)
    assert ff.locked == "other"


def test_frames_stopping_entirely_is_detected_by_timeout():
    cfg = FaceConfig(lost_hold_s=1.0, max_obs_age_s=0.3)
    ff, t = locked_follower(cfg)
    assert ff.check_timeout(t + 0.1) is None
    assert ff.check_timeout(t + 0.5).status is TrackStatus.HOLDING
    assert ff.check_timeout(t + 2.0).status is TrackStatus.LOST
    assert not ff.has_target


def test_stale_and_duplicate_frames_are_ignored():
    ff, t = locked_follower()
    r = ff.update(frame(t - 5.0, face("a", 0.9, 0.5)), t, pose_at)
    assert r.status is TrackStatus.STALE
    r = ff.update(frame(t, face("a", 0.9, 0.5)), t, pose_at)
    assert r.status is TrackStatus.STALE            # same stamp as last


def test_preferred_caller_wins_once_and_is_not_forced_when_absent():
    ff, t = locked_follower()                       # locked on "a"
    ff.set_preferred("caller")
    t += 0.05
    r = ff.update(frame(t, face("a", 0.5, 0.5)), t, pose_at)
    assert r.track_id == "a"                        # caller not visible
    t += 0.05
    r = ff.update(frame(t, face("a", 0.5, 0.5), face("caller", 0.8, 0.5)),
                  t, pose_at)
    assert r.track_id == "caller"
    t += 0.05
    r = ff.update(frame(t, face("a", 0.5, 0.5), face("caller", 0.8, 0.5)),
                  t, pose_at)
    assert r.track_id == "caller"                   # and stays


def test_caller_association_is_only_made_when_reliable():
    ff = FaceFollower()
    assert ff.associate_caller(frame(0.0)) is None
    assert ff.associate_caller(frame(0.0, face("x", 0.9, 0.5))) == "x"
    clear = frame(0.0, face("near", 0.52, 0.5), face("far", 0.9, 0.5))
    assert ff.associate_caller(clear) == "near"
    ambiguous = frame(0.0, face("l", 0.35, 0.5), face("r", 0.65, 0.5))
    assert ff.associate_caller(ambiguous) is None


# -------------------------------------------------------------- hand
def active_hand(cfg=None, side="right", owner="p1", **kw):
    hf = HandFollower(cfg or HandConfig(), **kw)
    assert hf.activate(0.0, side, owner)
    return hf


def test_inactive_hand_follower_never_commands():
    hf = HandFollower()
    r = hf.update(hand_obs(0.1, wrist=(0.9, 0.5)), 0.1, pose_at)
    assert r.status is HandStatus.INACTIVE and r.pan is None


def test_hand_moves_right_left_up_down():
    t = 0.1
    hf = active_hand()
    r = hf.update(hand_obs(t, wrist=(0.9, 0.5)), t, pose_at)
    assert r.pan > POSE[0]
    hf = active_hand()
    r = hf.update(hand_obs(t, wrist=(0.1, 0.5)), t, pose_at)
    assert r.pan < POSE[0]
    hf = active_hand()
    r = hf.update(hand_obs(t, wrist=(0.5, 0.1)), t, pose_at)
    assert r.tilt < POSE[1]                         # hand rises: tilt up
    hf = active_hand()
    r = hf.update(hand_obs(t, wrist=(0.5, 0.9)), t, pose_at)
    assert r.tilt > POSE[1]


def test_wrist_is_labelled_as_a_fallback_not_a_fingertip():
    hf = active_hand()
    r = hf.update(hand_obs(0.1, wrist=(0.8, 0.5)), 0.1, pose_at)
    assert r.kind == KIND_WRIST
    assert KIND_WRIST != KIND_FINGERTIP
    assert "fallback" in KIND_WRIST


def test_fingertip_is_used_when_a_model_provides_one():
    hf = active_hand()
    r = hf.update(hand_obs(0.1, wrist=(0.2, 0.5), tip=(0.8, 0.5)), 0.1,
                  pose_at)
    assert r.kind == KIND_FINGERTIP and r.pan > POSE[0]


def test_wrist_fallback_can_be_disabled():
    hf = active_hand(HandConfig(allow_wrist_fallback=False))
    r = hf.update(hand_obs(0.1, wrist=(0.9, 0.5)), 0.1, pose_at)
    assert r.status is HandStatus.HOLDING and r.pan is None


def test_fingertip_can_be_ignored_by_config():
    hf = active_hand(HandConfig(prefer_fingertip=False))
    r = hf.update(hand_obs(0.1, wrist=(0.2, 0.5), tip=(0.8, 0.5)), 0.1,
                  pose_at)
    assert r.kind == KIND_WRIST and r.pan < POSE[0]


def test_configured_hand_side_is_followed_exclusively():
    hf = active_hand(HandConfig(hand_side="left"), side="right")
    obs = hand_obs(0.1, wrist=(0.9, 0.5), side="right")
    assert hf.update(obs, 0.1, pose_at).status is HandStatus.HOLDING
    obs = hand_obs(0.2, wrist=(0.9, 0.5), side="left")
    r = hf.update(obs, 0.2, pose_at)
    assert r.status is HandStatus.TRACKING and r.side == "left"


def test_auto_side_sticks_to_the_hand_it_started_with():
    hf = HandFollower(HandConfig(hand_side="auto"))
    hf.activate(0.0, None, "p1")
    from head_fakes import W, H
    from buddy_core.head_tracking.models import HandObservation, HandPoint
    obs = HandObservation(0.1, W, H, "p1", {
        "left": HandPoint(100, 100, 0.6), "right": HandPoint(500, 100, 0.9)})
    r = hf.update(obs, 0.1, pose_at)
    assert r.side == "right"
    obs = HandObservation(0.2, W, H, "p1", {
        "left": HandPoint(100, 100, 0.99), "right": HandPoint(500, 100, 0.8)})
    assert hf.update(obs, 0.2, pose_at).side == "right"


def test_another_persons_hand_is_ignored():
    hf = active_hand(owner="p1")
    r = hf.update(hand_obs(0.1, wrist=(0.9, 0.5), track="stranger"), 0.1,
                  pose_at)
    assert r.pan is None


def test_low_confidence_hand_points_are_ignored():
    from head_fakes import W, H
    from buddy_core.head_tracking.models import HandObservation, HandPoint
    hf = active_hand()
    obs = HandObservation(0.1, W, H, "p1", {"right": HandPoint(500, 100,
                                                               0.1)})
    assert hf.update(obs, 0.1, pose_at).pan is None


def test_hand_loss_holds_then_deactivates():
    hf = active_hand(HandConfig(lost_hold_s=0.5))
    hf.update(hand_obs(0.1, wrist=(0.8, 0.5)), 0.1, pose_at)
    r = hf.update(hand_obs(0.2), 0.2, pose_at)
    assert r.status is HandStatus.HOLDING
    r = hf.update(hand_obs(0.9), 0.9, pose_at)
    assert r.status is HandStatus.LOST and not hf.active


def test_hand_following_expires_after_max_active_time():
    hf = active_hand(HandConfig(max_active_s=2.0))
    r = hf.update(hand_obs(2.5, wrist=(0.8, 0.5)), 2.5, pose_at)
    assert r.status is HandStatus.LOST and not hf.active


def test_hand_timeout_without_any_observations():
    hf = active_hand(HandConfig(lost_hold_s=0.5))
    assert hf.check_timeout(0.3) is None
    assert hf.check_timeout(1.0).status is HandStatus.LOST


def test_hand_side_value_is_validated():
    hf = HandFollower(HandConfig(hand_side="auto"))
    assert hf.activate(0.0, "middle", None) is False


def test_stale_hand_observation_is_ignored():
    hf = active_hand()
    r = hf.update(hand_obs(0.0, wrist=(0.9, 0.5)), 5.0, pose_at)
    assert r.status is HandStatus.STALE


@pytest.mark.parametrize("gname", ["wave", "raised_hand", "pointing"])
def test_gesture_events_are_plain_data(gname):
    assert gesture(gname).name == gname

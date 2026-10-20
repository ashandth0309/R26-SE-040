"""Tests for the safe pan/tilt motion coordinator."""

from __future__ import annotations

import math

import pytest
from head_fakes import RecordingBackend, make_stack, neck_config, run

from buddy_core.head_tracking.models import EventKind, Priority
from buddy_core.head_tracking.motion import HeadMotionCoordinator
from buddy_core.neck.controller import NeckController


def started(overrides=None, **kw):
    backend, co, _mgr, cfg = make_stack(overrides, **kw)
    assert co.start(0.0)
    return backend, co, cfg


def test_start_moves_to_neutral_inside_margin():
    backend, co, _ = started()
    assert (co.pan, co.tilt) == (90.0, 90.0)
    assert backend.writes("pan") == [90.0]


def test_repo_calibration_neutral_is_pulled_inside_limit_margin():
    # The repo's tilt calibration has center == max (180).  The margin
    # keeps Buddy 3 degrees away from the hard end.
    from buddy_core.config_loader import load_buddy_config
    neck = NeckController(RecordingBackend(), load_buddy_config())
    co = HeadMotionCoordinator(neck)
    assert co.tilt_neutral == 177.0
    assert co.pan_neutral == 135.0


def test_targets_are_clamped_to_limits_with_margin():
    backend, co, _ = started()
    co.submit(Priority.PASSIVE_FACE, 10_000, -10_000, 0.0)
    run(co, 0.0, 0.5)
    for _ in range(40):
        co.submit(Priority.PASSIVE_FACE, 10_000, -10_000, co._last_t)
        run(co, co._last_t, 0.5)
    assert max(backend.writes("pan")) <= 180.0 - 3.0
    assert min(backend.writes("tilt")) >= 45.0 + 3.0
    assert co.limit_hits > 0


def test_every_written_angle_is_inside_calibrated_limits():
    backend, co, _ = started()
    t = 0.0
    for target in (-500, 500, 90, -50, 1e9):
        for _ in range(60):
            co.submit(Priority.PASSIVE_FACE, target, target, t)
            t = run(co, t, 0.05)
    for a in backend.writes("pan"):
        assert 0.0 <= a <= 180.0
    for a in backend.writes("tilt"):
        assert 45.0 <= a <= 135.0


def test_velocity_is_limited():
    backend, co, cfg = started()
    t, last, worst = 0.0, co.pan, 0.0
    for _ in range(100):
        co.submit(Priority.PASSIVE_FACE, 170.0, 90.0, t)
        t += 0.02
        co.update(t)
        worst = max(worst, abs(co.pan - last) / 0.02)
        last = co.pan
    assert worst <= cfg.motion.max_velocity_pan_dps + 1e-6
    assert worst > 0.5 * cfg.motion.max_velocity_pan_dps


def test_acceleration_is_limited():
    _, co, cfg = started()
    t, vel_prev = 0.0, 0.0
    last = co.pan
    for _ in range(120):
        co.submit(Priority.PASSIVE_FACE, 170.0, 90.0, t)
        t += 0.02
        co.update(t)
        vel = (co.pan - last) / 0.02
        last = co.pan
        landed = co.pan == 170.0 and abs(vel) < 8.0
        if not landed:
            assert abs(vel - vel_prev) / 0.02 <= (
                cfg.motion.max_accel_dps2 + 1)
        else:       # the final step may drop at most 2 steps of speed
            assert abs(vel_prev) <= 2 * cfg.motion.max_accel_dps2 * 0.02
        vel_prev = vel


def test_approach_has_no_overshoot():
    _, co, _ = started()
    t, peak = 0.0, 0.0
    for _ in range(300):
        co.submit(Priority.PASSIVE_FACE, 120.0, 90.0, t)
        t += 0.02
        co.update(t)
        peak = max(peak, co.pan)
    assert peak <= 120.0 + 1e-9
    assert abs(co.pan - 120.0) <= 0.6


def test_dead_zone_ignores_tiny_target_changes():
    backend, co, _ = started()
    n = len(backend.calls)
    t = 0.0
    for _ in range(50):
        co.submit(Priority.PASSIVE_FACE, 90.3, 90.2, t)
        t = run(co, t, 0.02)
    assert len(backend.calls) == n          # nothing written
    co.submit(Priority.PASSIVE_FACE, 93.0, 90.0, t)
    run(co, t, 0.3)
    assert len(backend.calls) > n


def test_min_step_limits_servo_writes():
    backend, co, cfg = started()
    t = 0.0
    for _ in range(100):
        co.submit(Priority.PASSIVE_FACE, 100.0, 90.0, t)
        t = run(co, t, 0.02)
    pans = backend.writes("pan")
    diffs = [abs(b - a) for a, b in zip(pans, pans[1:])]
    # all but the final settling write respect the minimum step
    assert all(d >= cfg.motion.min_step_deg - 1e-9 for d in diffs[:-1])


@pytest.mark.parametrize("bad", [
    float("nan"), float("inf"), -float("inf"), True, "90", None, [1],
])
def test_invalid_commands_cause_no_motion(bad):
    backend, co, _ = started()
    n = len(backend.calls)
    assert co.submit(Priority.HAND, bad, 90.0, 0.1) is False
    assert co.submit(Priority.HAND, 90.0, bad, 0.1) is False
    run(co, 0.0, 0.5)
    assert len(backend.calls) == n
    assert co.rejected_commands == 2
    assert (co.pan, co.tilt) == (90.0, 90.0)


@pytest.mark.parametrize("scale", [0, -1, 1.5, float("nan"), True])
def test_invalid_speed_scale_is_rejected(scale):
    _, co, _ = started()
    assert co.submit(Priority.HAND, 100, 90, 0.1, speed_scale=scale) is False


def test_non_target_priorities_cannot_submit():
    _, co, _ = started()
    for p in (Priority.EMERGENCY_STOP, Priority.SAFETY_LIMITS,
              Priority.USER_STOP):
        assert co.submit(p, 100, 90, 0.0) is False


def test_submit_before_start_is_rejected():
    backend, co, _mgr, _ = make_stack()
    assert co.submit(Priority.HAND, 100, 90, 0.0) is False
    assert backend.calls == []


def test_highest_priority_source_wins_and_conflicts_are_counted():
    _, co, _ = started()
    t = 0.0
    for _ in range(100):
        co.submit(Priority.PASSIVE_FACE, 150.0, 90.0, t)
        co.submit(Priority.HAND, 30.0, 90.0, t)
        co.submit(Priority.SPEAKER_FACE, 160.0, 90.0, t)
        t = run(co, t, 0.02)
    assert co.active_source(t) is Priority.HAND
    assert co.pan < 90.0                       # followed HAND, not faces
    assert co.suppressed_conflicts > 0


def test_lower_priority_resumes_when_higher_is_released():
    _, co, _ = started()
    t = 0.0
    co.submit(Priority.PASSIVE_FACE, 130.0, 90.0, t)
    co.submit(Priority.HAND, 50.0, 90.0, t)
    co.release_source(Priority.HAND)
    assert co.active_source(t) is Priority.PASSIVE_FACE


def test_no_conflicting_commands_reach_the_backend():
    # With two live sources the pan write sequence is monotonic toward
    # the winner; it never ping-pongs between the two targets.
    backend, co, _ = started()
    t = 0.0
    for _ in range(150):
        co.submit(Priority.PASSIVE_FACE, 160.0, 90.0, t)
        co.submit(Priority.SPEAKER_FACE, 20.0, 90.0, t)
        t = run(co, t, 0.02)
    pans = backend.writes("pan")
    assert all(b <= a for a, b in zip(pans, pans[1:]))


def test_expired_target_holds_then_returns_to_neutral():
    _, co, cfg = started()
    co.submit(Priority.PASSIVE_FACE, 120.0, 90.0, 0.0)
    t = run(co, 0.0, 1.0)
    held = co.pan
    assert 90.0 < held < 120.0                 # expired mid-move: held
    t = run(co, t, 2.0)
    assert co.pan == held                      # still holding
    run(co, t, cfg.motion.neutral_after_s + 6.0)
    assert abs(co.pan - 90.0) <= cfg.motion.dead_zone_deg


def test_stalled_loop_cannot_cause_a_jump():
    _, co, cfg = started()
    co.submit(Priority.PASSIVE_FACE, 170.0, 90.0, 29.99)   # fresh target
    co.update(30.0)                            # after a 30 s stall
    assert co.pan - 90.0 <= (cfg.motion.max_velocity_pan_dps
                             * cfg.motion.max_dt_s) + 1e-6


def test_time_going_backwards_does_not_move():
    backend, co, _ = started()
    co.update(1.0)
    n = len(backend.calls)
    co.update(0.5)
    co.update(float("nan"))
    assert len(backend.calls) == n


# ----------------------------------------------------- emergency stop
def test_emergency_stop_halts_motion_and_blocks_commands():
    backend, co, _ = started()
    t = 0.0
    for _ in range(20):
        co.submit(Priority.HAND, 170.0, 90.0, t)
        t = run(co, t, 0.02)
    co.emergency_stop("test", t)
    n, pan = len(backend.calls), co.pan
    assert co.submit(Priority.HAND, 170.0, 90.0, t) is False
    run(co, t, 2.0)
    assert len(backend.calls) == n and co.pan == pan
    assert co.estop


def test_emergency_stop_clears_pending_targets_and_needs_reset():
    _, co, _ = started()
    co.submit(Priority.HAND, 170.0, 90.0, 0.0)
    co.emergency_stop("x", 0.0)
    co.reset_emergency(0.1)
    # old target must not survive the stop
    assert co.active_source(0.2) is None
    assert co.submit(Priority.HAND, 100.0, 90.0, 0.2) is True


def test_emergency_stop_can_release_servos_when_configured():
    backend, co, _ = started({"motion": {"release_on_emergency_stop": True}})
    co.emergency_stop("x", 0.0)
    assert ("release", None) in backend.calls


def test_emergency_stop_events():
    _, co, _mgr, _ = make_stack()
    co.start(0.0)
    co.emergency_stop("button", 1.0)
    co.reset_emergency(2.0)
    from head_fakes import kinds
    ks = kinds(_mgr.events_list)
    assert "EMERGENCY_STOP" in ks and "EMERGENCY_STOP_CLEARED" in ks


def test_user_stop_and_resume():
    backend, co, _ = started()
    co.submit(Priority.HAND, 150.0, 90.0, 0.0)
    co.user_stop(0.0)
    n = len(backend.calls)
    run(co, 0.0, 1.0)
    assert len(backend.calls) == n
    assert co.submit(Priority.HAND, 150.0, 90.0, 1.0) is False
    co.resume(1.0)
    assert co.submit(Priority.HAND, 150.0, 90.0, 1.1) is True


# ----------------------------------------------- hardware disconnection
def test_write_failure_latches_fault_and_stops_writing():
    backend, co, _mgr, _ = make_stack()
    co.start(0.0)
    backend.fail_after = len(backend.calls) + 2
    t = 0.0
    for _ in range(100):
        co.submit(Priority.HAND, 170.0, 90.0, t)
        t = run(co, t, 0.02)
    assert co.faulted
    assert "disconnected" in co.fault_reason
    n = len(backend.calls)
    run(co, t, 1.0)
    assert len(backend.calls) == n
    assert co.submit(Priority.HAND, 100.0, 90.0, t) is False
    assert EventKind.FAULT in [e.kind for e in _mgr.events_list]


def test_fault_attempts_release_and_survives_a_dead_release():
    backend, co, _ = started()
    backend.fail_release = True
    backend.fail_after = len(backend.calls)
    co.submit(Priority.HAND, 170.0, 90.0, 0.0)
    run(co, 0.0, 0.2)                           # must not raise
    assert co.faulted
    assert ("release", None) in backend.calls


def test_start_failure_is_a_fault_not_an_exception():
    backend = RecordingBackend()
    backend.fail_after = 0
    _b, co, _m, _ = make_stack(backend=backend)
    assert co.start(0.0) is False
    assert co.faulted


def test_reset_fault_recentres_when_bus_is_back():
    backend, co, _ = started()
    backend.fail_after = len(backend.calls)
    co.submit(Priority.HAND, 170.0, 90.0, 0.0)
    run(co, 0.0, 0.2)
    assert co.faulted
    backend.fail_after = None
    assert co.reset_fault(5.0) is True
    assert not co.faulted and co.started


# --------------------------------------------------------- shutdown
def test_shutdown_returns_to_neutral_then_releases_and_cleans_up():
    backend, co, _ = started()
    t = 0.0
    for _ in range(40):
        co.submit(Priority.HAND, 140.0, 90.0, t)
        t = run(co, t, 0.02)
    assert co.pan > 100
    co.shutdown(sleep=lambda s: None)
    assert abs(co.pan - 90.0) <= 0.6
    kinds = [k for k, _ in backend.calls]
    assert kinds.index("release") < kinds.index("cleanup")
    assert backend.released and backend.cleaned_up
    assert co.shut_down


def test_shutdown_is_idempotent_and_blocks_further_motion():
    backend, co, _ = started()
    co.shutdown(sleep=lambda s: None)
    n = len(backend.calls)
    co.shutdown(sleep=lambda s: None)
    assert len(backend.calls) == n
    assert co.submit(Priority.HAND, 100.0, 90.0, 5.0) is False


def test_shutdown_does_not_move_when_estopped():
    backend, co, _ = started()
    co.submit(Priority.HAND, 140.0, 90.0, 0.0)
    run(co, 0.0, 0.3)
    co.emergency_stop("x", 0.3)
    moves = len(backend.writes("pan"))
    co.shutdown(sleep=lambda s: None)
    assert len(backend.writes("pan")) == moves
    assert ("release", None) in backend.calls


def test_shutdown_survives_dead_hardware():
    backend, co, _ = started()
    backend.fail_after = len(backend.calls)
    backend.fail_release = True
    co.shutdown(sleep=lambda s: None)           # must not raise
    assert co.shut_down


def test_shutdown_before_start_is_safe():
    backend, co, _m, _ = make_stack()
    co.shutdown(sleep=lambda s: None)
    assert ("cleanup", None) in backend.calls


def test_position_at_returns_pose_when_frame_was_captured():
    _, co, _ = started()
    t = 0.0
    for _ in range(30):
        co.submit(Priority.HAND, 150.0, 90.0, t)
        t = run(co, t, 0.02)
    early, _ = co.position_at(0.2)
    late, _ = co.position_at(t)
    assert early < late
    assert math.isclose(late, co.pan)
    assert co.position_at(-5.0)[0] == 90.0


def test_tiny_travel_collapses_to_midpoint():
    cfg = neck_config(pan=(88.0, 90.0, 92.0), tilt=(88.0, 90.0, 92.0))
    _b, co, _m, _ = make_stack(neck_cfg=cfg)
    assert co.pan_lo == co.pan_hi == 90.0

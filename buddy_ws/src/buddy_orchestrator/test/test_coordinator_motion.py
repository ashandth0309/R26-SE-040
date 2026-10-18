"""Motion intake: nothing moves unless every safety input says it may."""

from conftest import Rig

from buddy_orchestrator.models import MoveIn, NavRequest, NavStatusIn
from buddy_orchestrator.states import RobotState


def test_healthy_forward_is_forwarded(rig):
    assert rig.c.robot.state is RobotState.IDLE
    assert rig.c.on_voice_navigation(NavRequest("move_forward",
                                                request_id="r1"))
    assert [n.kind for n in rig.nav] == ["move_forward"]


def test_estop_refuses_and_announces_once(rig):
    rig.healthy(estop=True)
    rig.healthy(estop=True)
    assert rig.c.robot.state is RobotState.EMERGENCY_STOP
    assert not rig.c.on_voice_navigation(NavRequest("start_patrol",
                                                    request_id="r1"))
    assert rig.nav[-1].kind == "stop"
    assert rig.texts.count("Emergency stop is active, so I can't move.") \
        >= 1
    assert "move_forward" not in [n.kind for n in rig.nav]


def test_estop_cannot_be_left_into_motion(rig):
    rig.healthy(estop=True)
    rig.healthy()
    assert rig.c.robot.state in (RobotState.IDLE, RobotState.DEGRADED_MODE)
    assert rig.c.robot.state is not RobotState.NAVIGATION_ACTIVE


def test_resume_refused_under_estop(rig):
    rig.healthy(estop=True)
    before = len(rig.nav)
    assert not rig.c.on_voice_navigation(NavRequest("resume"))
    assert all(n.kind != "resume" for n in rig.nav[before:])


def test_stop_always_goes_through_even_without_inputs():
    rig = Rig()                              # no status ever received
    assert rig.c.on_voice_navigation(NavRequest("stop"))
    assert rig.stops and rig.nav[-1].kind == "stop"


def test_stop_reaches_movement_even_if_nav_port_dies():
    rig = Rig()

    def boom(_):
        raise RuntimeError("nav port dead")

    rig.c.ports.nav_send = boom
    rig.c.stop_all("test")
    assert rig.stops == ["test"]


def test_stale_inputs_refuse_motion(rig):
    rig.t += 10.0                            # nothing refreshed
    assert not rig.c.on_voice_navigation(NavRequest("move_forward",
                                                    request_id="x"))
    assert "move_forward" not in [n.kind for n in rig.nav]


def test_losing_status_while_moving_stops_and_degrades(rig):
    rig.c.on_voice_navigation(NavRequest("start_patrol", request_id="p"))
    rig.healthy(nav_state="running", rid="p", cmd="start_patrol")
    assert rig.c.robot.state is RobotState.PATROL_ACTIVE
    rig.advance(5.0)                         # silence from every node
    assert rig.c.robot.state is RobotState.DEGRADED_MODE
    assert rig.stops
    assert any("lost contact" in t for t in rig.texts)


def test_degraded_recovers_to_idle(rig):
    rig.advance(5.0)
    assert rig.c.robot.state is RobotState.DEGRADED_MODE
    rig.healthy()
    assert rig.c.robot.state is RobotState.IDLE


def test_unmeasured_sensors_refuse_motion(rig):
    rig.healthy(sensors=False)
    assert not rig.c.on_voice_navigation(NavRequest("move_forward",
                                                    request_id="a"))
    assert any("sensors" in t for t in rig.texts)


def test_motion_disabled_refused(rig):
    rig.healthy(motion=False)
    assert not rig.c.on_voice_navigation(NavRequest("move_forward",
                                                    request_id="a"))
    assert any("switched off" in t for t in rig.texts)


def test_patrol_needs_trusted_pose(rig):
    rig.healthy(pose=False)
    assert not rig.c.on_voice_navigation(NavRequest("start_patrol",
                                                    request_id="a"))
    assert any("where I am" in t for t in rig.texts)


def test_busy_while_navigating(rig):
    rig.c.on_voice_navigation(NavRequest("start_patrol", request_id="p"))
    rig.healthy(nav_state="running", rid="p", cmd="start_patrol")
    assert not rig.c.on_voice_navigation(NavRequest("go_to_location",
                                                    request_id="g"))
    assert any("already doing something" in t for t in rig.texts)


def test_safety_not_allowing_movement_refuses(rig):
    rig.healthy(allowed=False)
    assert not rig.c.on_voice_navigation(NavRequest("turn_left",
                                                    request_id="a"))


def test_patrol_result_is_announced(rig):
    rig.c.on_voice_navigation(NavRequest("start_patrol", request_id="p"))
    rig.healthy(nav_state="running", rid="p", cmd="start_patrol")
    rig.c.on_nav_status(NavStatusIn(
        "succeeded", request_id="p", motion_enabled=True,
        pose_trusted=True, sensors_healthy=True))
    assert "Patrol finished." in rig.texts
    assert "p" not in rig.c.pending


def test_failed_nav_reason_is_spoken(rig):
    rig.c.on_voice_navigation(NavRequest("go_to_location",
                                         request_id="g"))
    rig.c.on_nav_status(NavStatusIn(
        "failed", request_id="g", reason="Path is blocked.",
        motion_enabled=True, pose_trusted=True, sensors_healthy=True))
    assert any("Path is blocked" in t for t in rig.texts)


def test_raw_movement_relayed_and_blocks_navigation_start(rig):
    m = MoveIn(1, source="voice", valid_for_s=0.5)
    assert rig.c.on_voice_movement(m)
    assert rig.c.on_voice_movement(m)        # a repeat keeps it alive
    assert rig.relayed == [m, m]
    assert not rig.c.on_voice_navigation(NavRequest("start_patrol",
                                                    request_id="p"))
    rig.advance(2.0)
    rig.healthy()
    assert rig.c.on_voice_navigation(NavRequest("start_patrol",
                                                request_id="p"))


def test_raw_movement_refused_under_estop_publishes_stop(rig):
    rig.healthy(estop=True)
    n = len(rig.stops)
    assert not rig.c.on_voice_movement(MoveIn(1, source="voice"))
    assert len(rig.stops) > n
    assert rig.relayed == []


def test_raw_movement_refused_while_navigating(rig):
    rig.c.on_voice_navigation(NavRequest("start_patrol", request_id="p"))
    rig.healthy(nav_state="running", rid="p", cmd="start_patrol")
    assert not rig.c.on_voice_movement(MoveIn(1, source="voice"))


def test_raw_stop_clears_everything(rig):
    rig.c.on_voice_movement(MoveIn(1, source="voice", valid_for_s=2.0))
    assert rig.c.on_voice_movement(MoveIn(0, source="voice"))
    assert rig.stops
    assert rig.c._raw_motion_until < 0
    assert rig.c.on_voice_navigation(NavRequest("move_forward",
                                                request_id="after"))


def test_relay_failure_is_refusal_not_crash(rig):
    def boom(_):
        raise OSError("topic gone")

    rig.c.ports.movement_relay = boom
    assert not rig.c.on_voice_movement(MoveIn(1, source="voice"))


def test_shutdown_stops_and_silences(rig):
    rig.c.shutdown()
    assert rig.stops
    rig.said.clear()
    rig.c._say(__import__("buddy_orchestrator.speech",
                          fromlist=["Kind"]).Kind.STATUS, "hello")
    assert rig.said == []


def test_state_machine_never_violated(rig):
    for _ in range(20):
        rig.healthy(estop=True)
        rig.healthy()
        rig.advance(5.0)
        rig.healthy()
    assert rig.c.status()["state_violations"] == 0

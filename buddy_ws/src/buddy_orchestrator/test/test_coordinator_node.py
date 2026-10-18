"""The ROS wrapper, against strict message stubs (not a real ROS run)."""

import importlib

import pytest
import ros_stub_orch as stub

ns = None
mod = None


@pytest.fixture(scope="module", autouse=True)
def ros_stubs():
    global ns, mod
    ns = stub.install()
    mod = importlib.import_module(
        "buddy_orchestrator.nodes.coordinator_node")
    yield
    stub.uninstall()


@pytest.fixture
def node(ros_stubs):
    stub.base.Node.instances.clear()
    stub.base.Node.overrides = {"profile": "simulation"}
    stub.base.Node.existing_nodes = []
    n = mod.CoordinatorNode()
    yield n
    stub.base.Node.overrides = {}


def feed_healthy(node, estop=False, now_ok=True):
    S, O, N = ns.msgs["SafetyState"], ns.msgs["ObstacleStatus"], \
        ns.nav["NavigationStatus"]
    s = S()
    s.emergency_stop_active, s.movement_allowed = estop, not estop
    o = O()
    o.front_valid = o.rear_valid = o.left_valid = o.right_valid = True
    n = N()
    n.motion_enabled = n.pose_trusted = n.sensors_healthy = True
    node.subs["/safety_state"][1](s)
    node.subs["/obstacle_status"][1](o)
    node.subs["/navigation/status"][1](n)
    node.core.tick()


def nav_cmd(kind, rid="r1", source="voice"):
    m = ns.nav["NavigationCommand"]()
    m.command = {"stop": 0, "move_forward": 1, "start_patrol": 4}[kind]
    m.request_id, m.source = rid, source

    return m


def test_publishers_subscribers_services_exist(node):
    assert set(node.pubs) >= {"/voice/say", "/movement/request",
                              "/navigation/command", "/robot_status",
                              "/coordinator/status"}
    assert set(node.subs) >= {
        "/safety_state", "/obstacle_status", "/navigation/status",
        "/voice/status", "/vision/status", "/vision/people",
        "/vision/events", "/coordinator/voice_movement",
        "/coordinator/voice_navigation"}
    assert "/coordinator/route_utterance" in node.services
    assert set(node.clients) == {
        "/vision/command", "/vision/identify_object",
        "/vision/teach_object", "/vision/where_is"}


def test_forwarded_navigation_keeps_every_field(node):
    feed_healthy(node)
    m = nav_cmd("start_patrol", "abc")
    m.route_id, m.repeat = "loop", True
    node.subs["/coordinator/voice_navigation"][1](m)
    out = node.pubs["/navigation/command"].sent[-1]
    assert out.command == 4 and out.request_id == "abc"
    assert out.route_id == "loop" and out.repeat is True


def test_stop_on_navigation_intake_stops_movement_and_navigation(node):
    node.subs["/coordinator/voice_navigation"][1](nav_cmd("stop"))
    stop = node.pubs["/movement/request"].sent[-1]
    assert stop.command == 0 and stop.emergency
    assert node.pubs["/navigation/command"].sent[-1].command == 0


def test_motion_refused_without_status(node):
    node.subs["/coordinator/voice_navigation"][1](nav_cmd("move_forward"))
    assert all(m.command != 1
               for m in node.pubs["/navigation/command"].sent)


def test_voice_movement_relayed_with_duration(node):
    feed_healthy(node)
    M = ns.msgs["MovementCommand"]
    m = M()
    m.command, m.source, m.sequence_id = 1, "voice", 7
    m.linear_speed = 0.2
    m.valid_for = stub.base.Duration(1, 500_000_000)
    node.subs["/coordinator/voice_movement"][1](m)
    out = node.pubs["/movement/request"].sent[-1]
    assert out.command == 1 and out.linear_speed == pytest.approx(0.2)
    assert out.valid_for.sec == 1
    assert out.valid_for.nanosec == pytest.approx(500_000_000, abs=10)


def test_speech_goes_out_as_speech_request(node):
    feed_healthy(node, estop=True)
    sent = node.pubs["/voice/say"].sent
    assert sent and sent[-1].priority == 0 and sent[-1].interrupt
    assert sent[-1].source == "buddy_coordinator"


def test_route_service_fills_response(node):
    srv, cb = node.services["/coordinator/route_utterance"]
    req = srv.Request()
    req.text = "what is this?"
    res = cb(req, srv.Response())
    assert res.claimed is True
    assert res.error_code == 1                   # UNAVAILABLE: no vision
    assert res.reply == "I can't see right now."


def test_route_service_passes_ordinary_talk(node):
    srv, cb = node.services["/coordinator/route_utterance"]
    req = srv.Request()
    req.text = "tell me a joke"
    assert cb(req, srv.Response()).claimed is False


def test_vision_client_not_ready_means_unavailable(node):
    node.subs["/vision/status"][1](stub.StringMsg(
        '{"camera_ok": true, "camera_enabled": true}'))
    node.clients["/vision/identify_object"].ready = False
    srv, cb = node.services["/coordinator/route_utterance"]
    req = srv.Request()
    req.text = "what is this"
    res = cb(req, srv.Response())
    assert res.claimed and "can't see" in res.reply


def test_vision_client_answer_is_used(node):
    node.subs["/vision/status"][1](stub.StringMsg(
        '{"camera_ok": true, "camera_enabled": true}'))
    ident = ns.srvs["IdentifyObject"]
    client = node.clients["/vision/identify_object"]

    def respond(_req):
        r = ident.Response()
        r.success, r.text = True, "I think that's a mug."

        return r

    client.responder = respond
    srv, cb = node.services["/coordinator/route_utterance"]
    req = srv.Request()
    req.text = "what is this"
    assert cb(req, srv.Response()).reply == "I think that's a mug."


def test_status_json_and_robot_status(node):
    feed_healthy(node)
    node.timers[1.0]()
    assert node.pubs["/coordinator/status"].sent[-1].data.startswith("{")
    rs = node.pubs["/robot_status"].sent[-1]
    assert rs.mode in (ns.msgs["RobotStatus"].MODE_IDLE,
                       ns.msgs["RobotStatus"].MODE_STARTING)


def test_vision_event_conversion_tolerates_bad_json():
    E = ns.msgs["VisionEvent"]
    e = E()
    e.kind, e.data_json = "fall_suspected", "{not json"
    assert mod.event_in(e).data == {}


def test_person_conversion():
    P = ns.msgs["VisionPerson"]
    p = P()
    p.person_key, p.known, p.bbox_x, p.bbox_width = "k", True, 10.0, 50.0
    got = mod.person_in(p)
    assert got.known and got.bbox_width == 50.0


def test_duplicate_coordinator_refused(ros_stubs):
    stub.base.Node.existing_nodes = ["buddy_coordinator"]
    try:
        with pytest.raises(RuntimeError):
            mod.CoordinatorNode()
    finally:
        stub.base.Node.existing_nodes = []


def test_audit_reports_second_camera(node):
    stub.base.Node.existing_nodes = ["buddy_camera", "buddy_camera"]
    node.timers[2.0]()
    assert any("camera" in c.message for c in node.core.conflicts)
    stub.base.Node.existing_nodes = []


def test_main_spins_and_shuts_down(ros_stubs):
    stub.base.Node.overrides = {"profile": "simulation"}
    stub.base.Rclpy.state.update(shutdowns=0)
    mod.main([])
    assert stub.Executor.spun >= 1
    assert stub.base.Rclpy.state["shutdowns"] == 1
    stub.base.Node.overrides = {}

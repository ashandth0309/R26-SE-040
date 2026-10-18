"""ROS 2 wrapper for the BUDDY coordinator.

All decisions live in ``buddy_orchestrator.coordinator``; this file only
converts messages to plain data and back, and owns the ROS plumbing.

* subscribes   /safety_state, /obstacle_status, /navigation/status,
               /voice/status, /vision/status, /vision/people,
               /vision/events, /coordinator/voice_movement,
               /coordinator/voice_navigation
* publishes    /voice/say, /movement/request (relay and STOP only),
               /navigation/command, /robot_status, /coordinator/status
* serves       /coordinator/route_utterance
* calls        /vision/command, /vision/identify_object,
               /vision/teach_object, /vision/where_is

It drives no motor and opens no device.
"""

from __future__ import annotations

import json
import threading

import rclpy
from builtin_interfaces.msg import Duration
from buddy_interfaces.msg import (
    MovementCommand,
    ObstacleStatus,
    RobotStatus,
    SafetyState,
    SpeechRequest,
    VisionEvent,
    VisionPeople,
)
from buddy_interfaces.srv import (
    IdentifyObject,
    RouteUtterance,
    TeachObject,
    VisionCommand,
    WhereIs,
)
from buddy_nav_interfaces.msg import NavigationCommand, NavigationStatus
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import String

from buddy_orchestrator.contracts import PUBLISHER_LIMITS, Service, Topic
from buddy_orchestrator.coordinator import (
    BuddyCoordinator,
    CoordinatorConfig,
)
from buddy_orchestrator.models import (
    NAV_CODES,
    NAV_KINDS,
    NAV_STATES,
    MoveIn,
    NavRequest,
    NavStatusIn,
    ObstacleIn,
    PersonIn,
    SafetyIn,
    VisionEventIn,
)
from buddy_orchestrator.ports import Ports, VisionCommandResult
from buddy_orchestrator.profiles import get_profile
from buddy_orchestrator.states import RobotState

ROBOT_MODE = {
    RobotState.STARTING: RobotStatus.MODE_STARTING,
    RobotState.INITIALIZING: RobotStatus.MODE_STARTING,
    RobotState.IDLE: RobotStatus.MODE_IDLE,
    RobotState.NAVIGATION_ACTIVE: RobotStatus.MODE_AUTONOMOUS,
    RobotState.PATROL_ACTIVE: RobotStatus.MODE_PATROL,
    RobotState.PAUSED: RobotStatus.MODE_AUTONOMOUS,
    RobotState.DEGRADED_MODE: RobotStatus.MODE_ERROR,
    RobotState.EMERGENCY_STOP: RobotStatus.MODE_EMERGENCY_STOP,
    RobotState.SHUTTING_DOWN: RobotStatus.MODE_SHUTTING_DOWN,
}


# ----- message <-> plain data (module level so tests can call them) -----

def nav_status_in(m) -> NavStatusIn:
    return NavStatusIn(
        state=NAV_STATES.get(int(m.state), "failed"),
        request_id=m.request_id,
        active_command=str(m.active_command),
        reason=m.reason, motion_enabled=bool(m.motion_enabled),
        pose_trusted=bool(m.pose_trusted),
        sensors_healthy=bool(m.sensors_healthy),
        emergency_stop_active=bool(m.emergency_stop_active),
        progress=float(m.progress))


def safety_in(m) -> SafetyIn:
    return SafetyIn(bool(m.emergency_stop_active),
                    bool(m.movement_allowed), bool(m.degraded_mode),
                    m.reason)


def obstacle_in(m) -> ObstacleIn:
    return ObstacleIn(bool(m.front_valid), bool(m.rear_valid),
                      bool(m.left_valid), bool(m.right_valid),
                      bool(m.degraded))


def person_in(p) -> PersonIn:
    return PersonIn(
        person_key=p.person_key, known=bool(p.known),
        display_name=p.display_name, identity_state=p.identity_state,
        posture=p.posture, motion=p.motion, fall_state=p.fall_state,
        bbox_x=float(p.bbox_x), bbox_width=float(p.bbox_width))


def event_in(m) -> VisionEventIn:
    try:
        data = json.loads(m.data_json) if m.data_json else {}
    except ValueError:
        data = {}

    return VisionEventIn(
        kind=m.kind, person_key=m.person_key, severity=m.severity,
        suggested_text=m.suggested_text,
        needs_voice_check=bool(m.needs_voice_check),
        data=data if isinstance(data, dict) else {})


def move_in(m) -> MoveIn:
    return MoveIn(
        command=int(m.command), source=m.source,
        sequence_id=int(m.sequence_id),
        linear_speed=float(m.linear_speed),
        angular_speed=float(m.angular_speed),
        valid_for_s=float(m.valid_for.sec) + m.valid_for.nanosec / 1e9,
        emergency=bool(m.emergency))


def nav_request_in(m) -> NavRequest:
    return NavRequest(
        kind=NAV_KINDS.get(int(m.command), "unknown"),
        request_id=m.request_id, source=m.source,
        duration_s=float(m.duration_s), route_id=m.route_id,
        repeat=bool(m.repeat), location_name=m.location_name)


class RosVision:
    """Typed, bounded calls to the vision node's services."""

    def __init__(self, node, timeout_s: float) -> None:
        group = ReentrantCallbackGroup()
        self._node, self._timeout = node, timeout_s
        mk = node.create_client
        self.c_command = mk(VisionCommand, Service.VISION_COMMAND,
                            callback_group=group)
        self.c_identify = mk(IdentifyObject, Service.VISION_IDENTIFY,
                             callback_group=group)
        self.c_teach = mk(TeachObject, Service.VISION_TEACH,
                          callback_group=group)
        self.c_where = mk(WhereIs, Service.VISION_WHERE_IS,
                          callback_group=group)

    def _call(self, client, request):
        if not client.service_is_ready():
            raise ConnectionError("vision service not available")

        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _f: done.set())

        if not done.wait(self._timeout):
            raise TimeoutError("vision service timed out")

        result = future.result()

        if result is None:
            raise ConnectionError("vision service returned nothing")

        return result

    def command(self, text, person_id=""):
        r = self._call(self.c_command, VisionCommand.Request(
            text=text, person_id=person_id))

        return VisionCommandResult(bool(r.handled), r.reply,
                                   bool(r.pass_to_dialogue))

    def identify(self):
        r = self._call(self.c_identify, IdentifyObject.Request())

        return {"success": bool(r.success), "text": r.text,
                "kind": r.kind, "name": r.name,
                "offer_teach": bool(r.offer_teach)}

    def teach(self, action, text):
        r = self._call(self.c_teach, TeachObject.Request(
            action=action, text=text))

        return {"ok": bool(r.ok), "reply": r.reply, "phase": r.phase,
                "saved": bool(r.saved)}

    def where_is(self, name):
        r = self._call(self.c_where, WhereIs.Request(name=name))

        return {"found": bool(r.found), "text": r.text}


class CoordinatorNode(Node):
    def __init__(self) -> None:
        super().__init__("buddy_coordinator")
        self._file_defaults = self._load_defaults()
        p = self._param
        profile = get_profile(str(p("profile", "offline")))
        cfg = CoordinatorConfig(
            person_distance_validated=bool(
                p("person_distance_validated", False)),
            init_timeout_s=float(p("init_timeout_s", 10.0)))

        if self._names().count("buddy_coordinator") > 1:
            raise RuntimeError("another buddy_coordinator is running")

        self.say_pub = self.create_publisher(
            SpeechRequest, Topic.VOICE_SAY, 10)
        self.move_pub = self.create_publisher(
            MovementCommand, Topic.MOVEMENT_REQUEST, 10)
        self.nav_pub = self.create_publisher(
            NavigationCommand, Topic.NAV_COMMAND, 10)
        self.robot_pub = self.create_publisher(
            RobotStatus, Topic.ROBOT_STATUS, 10)
        self.status_pub = self.create_publisher(
            String, Topic.COORDINATOR_STATUS, 10)
        self._seq = 0

        vision = RosVision(self, float(p("vision_timeout_s", 2.5))) \
            if p("enable_vision", True) else None
        self.core = BuddyCoordinator(
            profile,
            Ports(self._speak, self._nav_send, self._movement_stop,
                  self._movement_relay, vision),
            config=cfg)

        sub = self.create_subscription
        sub(SafetyState, Topic.SAFETY_STATE,
            lambda m: self.core.on_safety(safety_in(m)), 10)
        sub(ObstacleStatus, Topic.OBSTACLE_STATUS,
            lambda m: self.core.on_obstacle(obstacle_in(m)), 10)
        sub(NavigationStatus, Topic.NAV_STATUS,
            lambda m: self.core.on_nav_status(nav_status_in(m)), 10)
        sub(String, Topic.VOICE_STATUS, self._on_voice_status, 10)
        sub(String, Topic.VISION_STATUS, self._on_vision_status, 10)
        sub(VisionPeople, Topic.VISION_PEOPLE,
            lambda m: self.core.on_people(
                [person_in(x) for x in m.people]), 10)
        sub(VisionEvent, Topic.VISION_EVENTS,
            lambda m: self.core.on_vision_event(event_in(m)), 10)
        sub(MovementCommand, Topic.VOICE_MOVEMENT_INTAKE,
            lambda m: self.core.on_voice_movement(move_in(m)), 10)
        sub(NavigationCommand, Topic.VOICE_NAVIGATION_INTAKE,
            lambda m: self.core.on_voice_navigation(
                nav_request_in(m)), 10)

        self.create_service(
            RouteUtterance, Service.ROUTE_UTTERANCE, self._on_route,
            callback_group=ReentrantCallbackGroup())
        self.create_timer(0.1, self.core.tick)
        self.create_timer(1.0, self._publish_status)
        self.create_timer(2.0, self._audit_graph)
        self.get_logger().info(
            f"coordinator ready (profile={profile.name}, "
            f"vision={'on' if vision else 'off'})")

    # ----- ports -----

    def _param(self, name, default):
        default = self._file_defaults.get(name, default)
        self.declare_parameter(name, default)

        return self.get_parameter(name).value

    def _load_defaults(self) -> dict:
        """Defaults from ``config_file``; launch parameters still win."""

        self.declare_parameter("config_file", "")
        path = str(self.get_parameter("config_file").value or "")

        if not path:
            return {}

        try:
            import yaml

            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except (OSError, ImportError, ValueError) as exc:
            self.get_logger().warning(
                f"could not read config_file '{path}': {exc}; using "
                "built-in defaults")

            return {}

        return data if isinstance(data, dict) else {}

    def _names(self) -> list[str]:
        try:
            return list(self.get_node_names())
        except Exception:
            return []

    def _stamp(self):
        return self.get_clock().now().to_msg()

    def _speak(self, u) -> None:
        m = SpeechRequest()
        m.stamp = self._stamp()
        m.request_id, m.source, m.text = u.request_id, u.source, u.text
        m.priority, m.ttl_s, m.interrupt = int(u.priority), \
            float(u.ttl_s), bool(u.interrupt)
        self.say_pub.publish(m)

    def _next_move(self, command: int, source: str) -> MovementCommand:
        self._seq += 1
        m = MovementCommand()
        m.stamp, m.sequence_id = self._stamp(), self._seq
        m.source, m.command = source, command

        return m

    def _movement_stop(self, reason: str) -> None:
        m = self._next_move(MovementCommand.COMMAND_STOP,
                            "buddy_coordinator")
        m.emergency = True
        self.move_pub.publish(m)

    def _movement_relay(self, mv: MoveIn) -> None:
        m = self._next_move(mv.command, mv.source or "voice")
        m.linear_speed, m.angular_speed = mv.linear_speed, mv.angular_speed
        d = Duration()
        d.sec = int(mv.valid_for_s)
        d.nanosec = int((mv.valid_for_s - d.sec) * 1e9)
        m.valid_for = d
        self.move_pub.publish(m)

    def _nav_send(self, r: NavRequest) -> None:
        m = NavigationCommand()
        m.stamp = self._stamp()
        m.request_id, m.source = r.request_id, r.source or "buddy_coordinator"
        m.command = NAV_CODES[r.kind]
        m.duration_s, m.route_id = float(r.duration_s), r.route_id
        m.repeat, m.location_name = bool(r.repeat), r.location_name
        self.nav_pub.publish(m)

    # ----- callbacks -----

    @staticmethod
    def _json(msg) -> dict:
        try:
            data = json.loads(msg.data)
        except ValueError:
            return {}

        return data if isinstance(data, dict) else {}

    def _on_voice_status(self, msg) -> None:
        self.core.on_voice_status(self._json(msg))

    def _on_vision_status(self, msg) -> None:
        self.core.on_vision_status(self._json(msg))

    def _on_route(self, request, response):
        answer = self.core.route_utterance(
            request.text, float(request.confidence), request.request_id)
        response.claimed = bool(answer.claimed)
        response.reply = answer.reply
        response.intent = answer.intent
        response.error_code = int(answer.error)

        return response

    def _audit_graph(self) -> None:
        counts = {}

        for topic in PUBLISHER_LIMITS:
            try:
                counts[topic] = self.count_publishers(topic)
            except Exception:
                counts[topic] = 0

        self.core.graph_changed(self._names(), counts)

    def _publish_status(self) -> None:
        st = self.core.status()
        self.status_pub.publish(String(data=json.dumps(st)))
        m = RobotStatus()
        m.stamp = self._stamp()
        m.robot_id = "buddy"
        m.mode = int(ROBOT_MODE.get(self.core.robot.state,
                                    RobotStatus.MODE_ERROR))
        caps = self.core.capabilities
        m.online = True
        m.camera_available = bool(caps.camera)
        m.microphone_available = bool(caps.microphone)
        m.movement_available = bool(st["ready_for_motion"])
        m.obstacle_sensors_available = bool(caps.obstacle_sensors)
        m.internet_available = bool(caps.internet)
        m.activity = self.core.robot.state.value
        m.health_summary = st["readiness"][:200]
        self.robot_pub.publish(m)

    def destroy_node(self):
        try:
            self.core.shutdown()
        finally:
            super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CoordinatorNode()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

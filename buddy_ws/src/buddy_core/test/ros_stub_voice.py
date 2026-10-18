"""Strict ROS 2 stand-ins for testing voice_node without ROS.

Message classes are generated from the repo's real ``.msg`` files and use
``__slots__``, so a misspelt field name fails like it would on a robot.
"""

import sys
import types
from pathlib import Path

MSG_DIR = Path(__file__).resolve().parents[2] / "buddy_interfaces" / "msg"


class Time:
    __slots__ = ("sec", "nanosec")

    def __init__(self, sec=0, nanosec=0):
        self.sec, self.nanosec = sec, nanosec


class Duration(Time):
    __slots__ = ()


_DEFAULTS = {"bool": False, "string": "", "float32": 0.0, "float64": 0.0,
             "uint8": 0, "uint32": 0, "int32": 0, "int64": 0,
             "builtin_interfaces/Time": Time,
             "builtin_interfaces/Duration": Duration}


def make_msg(name):
    constants, fields = {}, []

    for line in (MSG_DIR / f"{name}.msg").read_text().splitlines():
        line = line.split("#")[0].strip()

        if not line:
            continue

        left, right = line.split(None, 1)

        if "=" in right:
            const, value = right.split("=")
            constants[const.strip()] = int(value)
        else:
            fields.append((right.strip(), left))

    names = [f for f, _ in fields]

    def init(self, **kw):
        for fname, ftype in fields:
            default = _DEFAULTS[ftype]
            setattr(self, fname, default() if callable(default)
                    else default)

        for key, value in kw.items():
            setattr(self, key, value)

    return type(name, (), {"__slots__": tuple(names), "__init__": init,
                           **constants})


class StubPublisher:
    def __init__(self, msg_type, topic):
        self.msg_type, self.topic, self.sent = msg_type, topic, []

    def publish(self, msg):
        assert isinstance(msg, self.msg_type), (
            f"{self.topic}: expected {self.msg_type.__name__}")
        self.sent.append(msg)


class StubLogger:
    def __init__(self):
        self.lines = []

    def info(self, text):
        self.lines.append(("info", text))

    def warn(self, text):
        self.lines.append(("warn", text))

    warning = warn

    def error(self, text):
        self.lines.append(("error", text))

    debug = info


class StubClock:
    class _Now:
        def to_msg(self):
            return Time(1, 2)

    def now(self):
        return self._Now()


class StubParam:
    def __init__(self, value):
        self.value = value


class Node:
    overrides = {}
    existing_nodes = []
    existing_publishers = {}
    instances = []

    def __init__(self, name):
        self.name = name
        self.params = {}
        self.pubs, self.subs, self.timers, self.services = {}, {}, {}, {}
        self._logger = StubLogger()
        self.destroyed = False
        Node.instances.append(self)

    def declare_parameter(self, name, default=None):
        self.params[name] = Node.overrides.get(name, default)

    def get_parameter(self, name):
        return StubParam(self.params[name])

    def create_publisher(self, msg_type, topic, qos):
        pub = StubPublisher(msg_type, topic)
        self.pubs[topic] = pub

        return pub

    def create_subscription(self, msg_type, topic, cb, qos):
        self.subs[topic] = (msg_type, cb)

    def create_timer(self, period, cb):
        self.timers[period] = cb

    def create_service(self, srv_type, name, cb):
        self.services[name] = (srv_type, cb)

    def get_logger(self):
        return self._logger

    def get_clock(self):
        return StubClock()

    def get_node_names(self):
        # Like rclpy, the graph includes this node itself.
        return [*Node.existing_nodes, self.name]

    def count_publishers(self, topic):
        return Node.existing_publishers.get(topic, 0) + sum(
            1 for n in Node.instances
            if n is not self and topic in n.pubs)

    def destroy_node(self):
        self.destroyed = True


class SetBool:
    class Request:
        __slots__ = ("data",)

        def __init__(self, data=False):
            self.data = data

    class Response:
        __slots__ = ("success", "message")

        def __init__(self):
            self.success, self.message = False, ""


class StringMsg:
    __slots__ = ("data",)

    def __init__(self):
        self.data = ""


class Rclpy:
    state = {"ok": False, "spun": 0, "shutdowns": 0}

    @classmethod
    def init(cls, args=None):
        cls.state["ok"] = True

    @classmethod
    def ok(cls):
        return cls.state["ok"]

    @classmethod
    def shutdown(cls):
        cls.state["ok"] = False
        cls.state["shutdowns"] += 1

    @classmethod
    def spin(cls, node):
        cls.state["spun"] += 1


_NAMES = ("buddy_interfaces", "buddy_interfaces.msg", "rclpy",
          "rclpy.node", "std_msgs", "std_msgs.msg", "std_srvs",
          "std_srvs.srv", "buddy_nav_interfaces",
          "buddy_nav_interfaces.msg", "buddy_core.voice.voice_node")
_saved = {}


def uninstall():
    """Put ``sys.modules`` back as it was before ``install``."""

    for name in _NAMES:
        sys.modules.pop(name, None)

    for name, module in _saved.items():
        sys.modules[name] = module

    _saved.clear()


def install():
    """Insert the stubs into ``sys.modules``; ``uninstall`` undoes it."""

    for name in _NAMES:
        if name in sys.modules and name not in _saved:
            _saved[name] = sys.modules[name]

    msgs = types.ModuleType("buddy_interfaces.msg")

    for name in ("MovementCommand", "SafetyState", "EmotionObservation",
                 "VoiceCommand"):
        setattr(msgs, name, make_msg(name))

    pkg = types.ModuleType("buddy_interfaces")
    pkg.msg = msgs
    rclpy = types.ModuleType("rclpy")
    for attr in ("init", "ok", "shutdown", "spin"):
        setattr(rclpy, attr, getattr(Rclpy, attr))
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = Node
    rclpy.node = rclpy_node
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.String = StringMsg
    std_msgs.msg = std_msgs_msg
    std_srvs = types.ModuleType("std_srvs")
    std_srvs_srv = types.ModuleType("std_srvs.srv")
    std_srvs_srv.SetBool = SetBool
    std_srvs.srv = std_srvs_srv
    sys.modules.update({
        "buddy_interfaces": pkg, "buddy_interfaces.msg": msgs,
        "rclpy": rclpy, "rclpy.node": rclpy_node,
        "std_msgs": std_msgs, "std_msgs.msg": std_msgs_msg,
        "std_srvs": std_srvs, "std_srvs.srv": std_srvs_srv,
    })
    sys.modules.pop("buddy_nav_interfaces", None)
    sys.modules.pop("buddy_nav_interfaces.msg", None)

    return msgs

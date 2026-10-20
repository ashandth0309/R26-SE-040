"""Strict ROS 2 stand-ins for testing the head-tracking node without ROS.

Message classes for SafetyState and VoiceCommand are generated from the
repo's real ``.msg`` files with ``__slots__`` so a misspelt field fails the
way it would on a robot.  Nothing here proves behaviour under real rclpy
or DDS.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

MSG_DIR = Path(__file__).resolve().parents[2] / "buddy_interfaces" / "msg"


class Time:
    __slots__ = ("sec", "nanosec")

    def __init__(self, sec=0, nanosec=0):
        self.sec, self.nanosec = sec, nanosec


_DEFAULTS = {"bool": False, "string": "", "float32": 0.0, "uint8": 0,
             "builtin_interfaces/Time": Time}


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


class StringMsg:
    __slots__ = ("data",)

    def __init__(self, data=""):
        self.data = data


class Logger:
    def __init__(self):
        self.lines = []

    def info(self, t):
        self.lines.append(("info", t))

    def warn(self, t):
        self.lines.append(("warn", t))

    def error(self, t):
        self.lines.append(("error", t))


class Clock:
    now_ns = 1_000_000_000

    class _Now:
        def __init__(self, ns):
            self.nanoseconds = ns

    def now(self):
        return self._Now(Clock.now_ns)


class Pub:
    def __init__(self, msg_type, topic):
        self.msg_type, self.topic, self.sent = msg_type, topic, []

    def publish(self, msg):
        assert isinstance(msg, self.msg_type), self.topic
        self.sent.append(msg)


class Param:
    def __init__(self, value):
        self.value = value


class Node:
    overrides: dict = {}
    instances: list = []

    def __init__(self, name):
        self.name = name
        self.params = {}
        self.pubs, self.subs, self.timers = {}, {}, {}
        self._logger = Logger()
        self.destroyed = False
        Node.instances.append(self)

    def declare_parameter(self, name, default=None):
        self.params[name] = Node.overrides.get(name, default)

    def get_parameter(self, name):
        return Param(self.params[name])

    def create_publisher(self, msg_type, topic, qos):
        pub = Pub(msg_type, topic)
        self.pubs[topic] = pub
        return pub

    def create_subscription(self, msg_type, topic, cb, qos):
        self.subs[topic] = (msg_type, cb)

    def create_timer(self, period, cb):
        self.timers[period] = cb

    def get_logger(self):
        return self._logger

    def get_clock(self):
        return Clock()

    def destroy_node(self):
        self.destroyed = True


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


_NAMES = ("rclpy", "rclpy.node", "std_msgs", "std_msgs.msg",
          "buddy_interfaces", "buddy_interfaces.msg",
          "buddy_core.head_tracking.ros_node")
_saved: dict = {}


def install(with_people: bool = False):
    """Put stubs into ``sys.modules``; returns a dict of the stub types."""

    for n in _NAMES:
        _saved[n] = sys.modules.get(n)
    Node.instances.clear()
    Node.overrides = {}
    Rclpy.state.update(ok=False, spun=0, shutdowns=0)

    rclpy = types.ModuleType("rclpy")
    rclpy.init, rclpy.ok = Rclpy.init, Rclpy.ok
    rclpy.shutdown, rclpy.spin = Rclpy.shutdown, Rclpy.spin
    rnode = types.ModuleType("rclpy.node")
    rnode.Node = Node
    std = types.ModuleType("std_msgs")
    stdm = types.ModuleType("std_msgs.msg")
    stdm.String = StringMsg
    bi = types.ModuleType("buddy_interfaces")
    bim = types.ModuleType("buddy_interfaces.msg")
    bim.SafetyState = make_msg("SafetyState")
    bim.VoiceCommand = make_msg("VoiceCommand")
    if with_people:
        bim.VisionPeople = type("VisionPeople", (), {"__slots__": (
            "stamp", "people")})
    for name, mod in (("rclpy", rclpy), ("rclpy.node", rnode),
                      ("std_msgs", std), ("std_msgs.msg", stdm),
                      ("buddy_interfaces", bi),
                      ("buddy_interfaces.msg", bim)):
        sys.modules[name] = mod
    sys.modules.pop("buddy_core.head_tracking.ros_node", None)
    return {"String": StringMsg, "SafetyState": bim.SafetyState,
            "VoiceCommand": bim.VoiceCommand,
            "VisionPeople": getattr(bim, "VisionPeople", None)}


def uninstall():
    for n in _NAMES:
        if _saved.get(n) is None:
            sys.modules.pop(n, None)
        else:
            sys.modules[n] = _saved[n]
    _saved.clear()

"""Strict ROS 2 stand-ins for the coordinator node.

Messages and services are generated from the repo's real ``.msg`` and
``.srv`` files with ``__slots__``, so a wrong field name fails here the
way it would on a robot.  This is not a real ROS 2 run.
"""

import sys
import types
from pathlib import Path

import ros_stub_voice as base

SRC = Path(__file__).resolve().parents[2]
IFACE = SRC / "buddy_interfaces"
NAV = SRC / "buddy_nav_interfaces"

_DEFAULTS = dict(base._DEFAULTS)
_DEFAULTS.update({"string[]": list, "VisionPerson[]": list,
                  "int32": 0, "float64": 0.0})


def _parse(path):
    consts, fields = {}, []

    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()

        if not line:
            continue

        left, right = line.split(None, 1)

        if "=" in right:
            k, v = right.split("=")
            consts[k.strip()] = int(v)
        else:
            fields.append((right.strip(), left))

    return consts, fields


def _cls(name, consts, fields):
    names = [f for f, _ in fields]

    def init(self, **kw):
        for fname, ftype in fields:
            d = _DEFAULTS[ftype]
            setattr(self, fname, d() if callable(d) else d)

        for k, v in kw.items():
            setattr(self, k, v)

    return type(name, (), {"__slots__": tuple(names), "__init__": init,
                           **consts})


def make_msg(path):
    consts, fields = _parse(path)

    return _cls(path.stem, consts, fields)


def make_srv(path):
    text = path.read_text().split("\n---")
    req, res = text[0], text[1]
    parts = []

    for part, label in ((req, "Request"), (res, "Response")):
        consts, fields = {}, []

        for line in part.splitlines():
            line = line.split("#")[0].strip()

            if not line or line == "---":
                continue

            left, right = line.split(None, 1)
            fields.append((right.strip(), left))

        parts.append(_cls(label, consts, fields))

    return type(path.stem, (), {"Request": parts[0],
                                "Response": parts[1]})


class StringMsg:
    __slots__ = ("data",)

    def __init__(self, data=""):
        self.data = data


class OrchNode(base.Node):
    def create_service(self, srv_type, name, cb, callback_group=None):
        self.services[name] = (srv_type, cb)

    def create_client(self, srv_type, name, callback_group=None):
        client = base.StubClient(srv_type, name)
        self.clients[name] = client

        return client


class Executor:
    spun = 0

    def __init__(self, num_threads=1):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def spin(self):
        Executor.spun += 1


_NAMES = ("buddy_interfaces", "buddy_interfaces.msg",
          "buddy_interfaces.srv", "buddy_nav_interfaces",
          "buddy_nav_interfaces.msg", "builtin_interfaces",
          "builtin_interfaces.msg", "rclpy", "rclpy.node",
          "rclpy.callback_groups", "rclpy.executors", "std_msgs",
          "std_msgs.msg", "yaml")


def install():
    mods = {}

    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        mods[name] = m

        return m

    bi_msgs = {p.stem: make_msg(p) for p in (IFACE / "msg").glob("*.msg")}
    bi_srvs = {p.stem: make_srv(p) for p in (IFACE / "srv").glob("*.srv")}
    nav_msgs = {p.stem: make_msg(p) for p in (NAV / "msg").glob("*.msg")}
    module("buddy_interfaces").msg = module("buddy_interfaces.msg",
                                            **bi_msgs)
    mods["buddy_interfaces"].srv = module("buddy_interfaces.srv",
                                          **bi_srvs)
    module("buddy_nav_interfaces").msg = module(
        "buddy_nav_interfaces.msg", **nav_msgs)
    module("builtin_interfaces").msg = module(
        "builtin_interfaces.msg", Duration=base.Duration, Time=base.Time)
    rclpy = module("rclpy", init=base.Rclpy.init, ok=base.Rclpy.ok,
                   shutdown=base.Rclpy.shutdown, spin=base.Rclpy.spin)
    rclpy.node = module("rclpy.node", Node=OrchNode)
    rclpy.callback_groups = module("rclpy.callback_groups",
                                   ReentrantCallbackGroup=object)
    rclpy.executors = module("rclpy.executors",
                             MultiThreadedExecutor=Executor)
    module("std_msgs").msg = module("std_msgs.msg", String=StringMsg)
    sys.modules.update(mods)
    sys.modules.pop("buddy_orchestrator.nodes.coordinator_node", None)

    return types.SimpleNamespace(msgs=bi_msgs, srvs=bi_srvs,
                                 nav=nav_msgs)


def uninstall():
    sys.modules.pop("buddy_orchestrator.nodes.coordinator_node", None)

    for name in _NAMES:
        sys.modules.pop(name, None)

"""Strict, minimal ROS 2 stand-ins so node logic can run without ROS.

This is NOT ROS.  It exists because the development laptop and CI have no
ROS 2 installation.  It catches logic errors in node wrappers (wrong field
names, wrong topics, wrong call order) but it cannot prove DDS, QoS or
executor behaviour.  Tests that use it must say so, and the real ROS
integration tests under ``test/ros`` remain the authority on a ROS machine.

Message classes are generated from the repository's real ``.msg`` files for
the BUDDY packages, and from inline definitions for the standard ROS
messages the nodes use, so assigning to a field that does not exist raises
``AttributeError`` just as it would on a real message.
"""

from __future__ import annotations

import sys
import types
from collections import defaultdict
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[2]

PRIMITIVES = {
    "bool": False,
    "byte": 0,
    "char": 0,
    "int8": 0,
    "uint8": 0,
    "int16": 0,
    "uint16": 0,
    "int32": 0,
    "uint32": 0,
    "int64": 0,
    "uint64": 0,
    "float32": 0.0,
    "float64": 0.0,
    "string": "",
}

STANDARD_DEFINITIONS = {
    "builtin_interfaces/Time": "int32 sec\nuint32 nanosec\n",
    "builtin_interfaces/Duration": "int32 sec\nuint32 nanosec\n",
    "std_msgs/Header": "builtin_interfaces/Time stamp\nstring frame_id\n",
    "std_msgs/String": "string data\n",
    "std_msgs/Bool": "bool data\n",
    "sensor_msgs/Range": (
        "uint8 ULTRASOUND=0\nuint8 INFRARED=1\n"
        "std_msgs/Header header\nuint8 radiation_type\n"
        "float32 field_of_view\nfloat32 min_range\n"
        "float32 max_range\nfloat32 range\n"
    ),
    "sensor_msgs/Image": (
        "std_msgs/Header header\nuint32 height\nuint32 width\n"
        "string encoding\nuint8 is_bigendian\nuint32 step\n"
        "uint8[] data\n"
    ),
    "sensor_msgs/CameraInfo": (
        "std_msgs/Header header\nuint32 height\nuint32 width\n"
        "string distortion_model\nfloat64[] d\nfloat64[9] k\n"
        "float64[9] r\nfloat64[12] p\n"
    ),
    "diagnostic_msgs/KeyValue": "string key\nstring value\n",
    "diagnostic_msgs/DiagnosticStatus": (
        "byte OK=0\nbyte WARN=1\nbyte ERROR=2\nbyte STALE=3\n"
        "byte level\nstring name\nstring message\nstring hardware_id\n"
        "diagnostic_msgs/KeyValue[] values\n"
    ),
    "diagnostic_msgs/DiagnosticArray": (
        "std_msgs/Header header\n"
        "diagnostic_msgs/DiagnosticStatus[] status\n"
    ),
    "geometry_msgs/Point": "float64 x\nfloat64 y\nfloat64 z\n",
    "geometry_msgs/Quaternion": (
        "float64 x\nfloat64 y\nfloat64 z\nfloat64 w\n"
    ),
    "geometry_msgs/Pose": (
        "geometry_msgs/Point position\n"
        "geometry_msgs/Quaternion orientation\n"
    ),
    "geometry_msgs/PoseStamped": (
        "std_msgs/Header header\ngeometry_msgs/Pose pose\n"
    ),
    "geometry_msgs/PoseWithCovarianceStamped": (
        "std_msgs/Header header\ngeometry_msgs/Pose pose\n"
        "float64[36] covariance\n"
    ),
    "nav_msgs/MapMetaData": (
        "builtin_interfaces/Time map_load_time\nfloat32 resolution\n"
        "uint32 width\nuint32 height\ngeometry_msgs/Pose origin\n"
    ),
    "nav_msgs/OccupancyGrid": (
        "std_msgs/Header header\nnav_msgs/MapMetaData info\n"
        "int8[] data\n"
    ),
    "nav_msgs/Path": (
        "std_msgs/Header header\ngeometry_msgs/PoseStamped[] poses\n"
    ),
}

_REGISTRY: dict[str, type] = {}


def _parse(definition: str):
    constants = {}
    fields = []

    for raw in definition.splitlines():
        line = raw.split("#", 1)[0].strip()

        if not line:
            continue

        type_name, rest = line.split(None, 1)

        if "=" in rest:
            name, value = (part.strip() for part in rest.split("=", 1))
            constants[name] = (
                float(value) if "." in value else int(value)
                if value.lstrip("-").isdigit() else value
            )
            continue

        fields.append((type_name, rest.strip()))

    return constants, fields


def _default_for(type_name: str):
    if type_name.endswith("]"):
        return list

    base = type_name

    if base in PRIMITIVES:
        value = PRIMITIVES[base]
        return lambda: value

    return lambda: message_class(base)()


def message_class(full_name: str) -> type:
    """Return (building if needed) the stub class for ``pkg/Name``."""

    if full_name in _REGISTRY:
        return _REGISTRY[full_name]

    if full_name in STANDARD_DEFINITIONS:
        definition = STANDARD_DEFINITIONS[full_name]
    else:
        package, name = full_name.split("/")
        kind = "srv" if name.endswith("Request") else "msg"
        path = REPO_SRC / package / kind / f"{name}.msg"

        if not path.is_file():
            raise KeyError(f"No definition for message {full_name}")

        definition = path.read_text()

    constants, fields = _parse(definition)
    factories = {name: _default_for(type_name) for type_name, name in fields}

    def __init__(self, **kwargs):
        for field_name, factory in factories.items():
            object.__setattr__(self, field_name, factory())

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, key, value):
        if key not in factories:
            raise AttributeError(
                f"{full_name} has no field '{key}'"
            )

        object.__setattr__(self, key, value)

    namespace = {
        "__init__": __init__,
        "__setattr__": __setattr__,
        "_FIELDS": tuple(factories),
        "_FULL_NAME": full_name,
        **constants,
    }

    cls = type(full_name.split("/")[1], (), namespace)
    _REGISTRY[full_name] = cls

    return cls


class StubTime:
    """A ROS time value."""

    def __init__(self, nanoseconds: int):
        self.nanoseconds = nanoseconds

    def to_msg(self):
        message = message_class("builtin_interfaces/Time")()
        message.sec = self.nanoseconds // 1_000_000_000
        message.nanosec = self.nanoseconds % 1_000_000_000
        return message

    def __sub__(self, other):
        return StubDuration(self.nanoseconds - other.nanoseconds)


class StubDuration:
    """Difference between two times."""

    def __init__(self, nanoseconds: int):
        self.nanoseconds = nanoseconds


class StubClock:
    """Manually advanced clock."""

    def __init__(self):
        self.ns = 1_000_000_000

    def now(self):
        return StubTime(self.ns)

    def advance(self, seconds: float):
        self.ns += int(seconds * 1_000_000_000)


class StubLogger:
    """Collects log lines for assertions."""

    def __init__(self):
        self.lines = []

    def _log(self, level, text):
        self.lines.append((level, str(text)))

    def info(self, text):
        self._log("info", text)

    def warning(self, text):
        self._log("warning", text)

    warn = warning

    def error(self, text):
        self._log("error", text)

    def debug(self, text):
        self._log("debug", text)

    def messages(self, level=None):
        return [t for lv, t in self.lines if level in (None, lv)]


class StubPublisher:
    """Records everything published."""

    def __init__(self, topic, msg_type):
        self.topic = topic
        self.msg_type = msg_type
        self.messages = []

    def publish(self, message):
        expected = self.msg_type
        actual = type(message)

        if actual is not expected:
            raise TypeError(
                f"Publishing {actual.__name__} on {self.topic}, "
                f"expected {expected.__name__}"
            )

        self.messages.append(message)


class StubParameter:
    """A parameter value holder."""

    def __init__(self, value):
        self.value = value


class StubNode:
    """Subset of ``rclpy.node.Node`` used by the BUDDY nodes."""

    def __init__(self, node_name, **kwargs):
        self.node_name = node_name
        self._params = {}
        self._overrides = {}
        self.clock = StubClock()
        self.logger = StubLogger()
        self.publishers = {}
        self.subscriptions = defaultdict(list)
        self.services = {}
        self.clients = {}
        self.timers = []
        self.destroyed = False

    def declare_parameter(self, name, default=None, *args, **kwargs):
        value = self._overrides.get(name, default)
        self._params[name] = StubParameter(value)

        return self._params[name]

    def get_parameter(self, name):
        return self._params[name]

    def get_clock(self):
        return self.clock

    def get_logger(self):
        return self.logger

    def create_publisher(self, msg_type, topic, qos):
        publisher = StubPublisher(topic, msg_type)
        self.publishers[topic] = publisher

        return publisher

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscriptions[topic].append((msg_type, callback))

        return object()

    def create_service(self, srv_type, name, callback):
        self.services[name] = (srv_type, callback)

        return object()

    def create_timer(self, period, callback):
        self.timers.append((period, callback))

        return object()

    def destroy_node(self):
        self.destroyed = True

        return True

    # Test helpers ------------------------------------------------------

    def deliver(self, topic, message):
        """Invoke subscriber callbacks as the middleware would."""

        for msg_type, callback in self.subscriptions[topic]:
            if type(message) is not msg_type:
                raise TypeError(
                    f"Delivering {type(message).__name__} to {topic}, "
                    f"expected {msg_type.__name__}"
                )

            callback(message)

    def fire_timers(self, period=None):
        for timer_period, callback in list(self.timers):
            if period is None or timer_period == period:
                callback()


def _module(name, **attributes):
    module = types.ModuleType(name)

    for key, value in attributes.items():
        setattr(module, key, value)

    sys.modules[name] = module

    return module


def stubs_available() -> bool:
    """Return whether stubs may be used (no real ROS 2 is installed)."""

    existing = sys.modules.get("rclpy")

    if existing is not None:
        return bool(getattr(existing, "_buddy_stub", False))

    try:
        import importlib.util

        return importlib.util.find_spec("rclpy") is None
    except (ImportError, ValueError):
        return True


def install(extra_messages=()) -> None:
    """Register stub ROS modules in ``sys.modules`` (idempotent)."""

    if not stubs_available():
        raise RuntimeError("Real rclpy is installed; refusing to stub it.")

    already = getattr(sys.modules.get("rclpy"), "_buddy_stub", False)

    if already:
        _add_extra_messages(extra_messages)

        return

    rclpy = _module(
        "rclpy",
        init=lambda *a, **k: None,
        ok=lambda: True,
        spin=lambda *a, **k: None,
        shutdown=lambda *a, **k: None,
        _buddy_stub=True,
    )
    rclpy.node = _module("rclpy.node", Node=StubNode)
    sys.modules["rclpy.node"] = rclpy.node

    packages = {
        "builtin_interfaces": ["Time", "Duration"],
        "std_msgs": ["Header", "String", "Bool"],
        "sensor_msgs": ["Range", "Image", "CameraInfo"],
        "diagnostic_msgs": [
            "KeyValue",
            "DiagnosticStatus",
            "DiagnosticArray",
        ],
        "geometry_msgs": [
            "Point",
            "Quaternion",
            "Pose",
            "PoseStamped",
            "PoseWithCovarianceStamped",
        ],
        "nav_msgs": ["MapMetaData", "OccupancyGrid", "Path"],
        "buddy_interfaces": [
            "ObstacleStatus",
            "MovementCommand",
            "SafetyState",
            "SecurityEvent",
            "RobotStatus",
            "VoiceCommand",
        ],
    }

    for package, names in packages.items():
        messages = {
            name: message_class(f"{package}/{name}") for name in names
        }
        _module(package)
        module = _module(f"{package}.msg", **messages)
        setattr(sys.modules[package], "msg", module)

    _add_extra_messages(extra_messages)


def _add_extra_messages(extra_messages) -> None:
    for full_name in extra_messages:
        package, name = full_name.split("/")
        # Create modules only when missing: ``setdefault(key, _module(..))``
        # would build (and register) a fresh module on every iteration and
        # drop the messages added before it.
        module = sys.modules.get(f"{package}.msg") or _module(
            f"{package}.msg"
        )
        setattr(module, name, message_class(full_name))
        parent = sys.modules.get(package) or _module(package)
        setattr(parent, "msg", module)

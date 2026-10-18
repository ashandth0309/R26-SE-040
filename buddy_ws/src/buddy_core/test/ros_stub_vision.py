"""ROS stand-ins for the vision nodes, built on the voice stubs.

Adds ``sensor_msgs``, ``diagnostic_msgs`` and extra ``buddy_interfaces``
messages generated from the repo's real ``.msg`` files.
"""

import sys
import types

import ros_stub_voice as base


class _Header:
    def __init__(self):
        self.stamp = base.Time()
        self.frame_id = ""


class Image:
    __slots__ = ("header", "height", "width", "encoding", "is_bigendian",
                 "step", "data")

    def __init__(self):
        self.header = _Header()
        self.height = self.width = self.is_bigendian = self.step = 0
        self.encoding = ""
        self.data = b""


class KeyValue:
    __slots__ = ("key", "value")

    def __init__(self, key="", value=""):
        self.key, self.value = key, value


class DiagnosticStatus:
    OK, WARN, ERROR, STALE = 0, 1, 2, 3
    __slots__ = ("level", "name", "message", "hardware_id", "values")

    def __init__(self):
        self.level, self.name, self.message = 0, "", ""
        self.hardware_id, self.values = "", []


class DiagnosticArray:
    __slots__ = ("header", "status")

    def __init__(self):
        self.header = _Header()
        self.status = []


_EXTRA = ("sensor_msgs", "sensor_msgs.msg", "diagnostic_msgs",
          "diagnostic_msgs.msg")


def install(extra_msgs=()):
    msgs = base.install()

    for name in extra_msgs:
        setattr(msgs, name, base.make_msg(name))

    sm = types.ModuleType("sensor_msgs")
    smm = types.ModuleType("sensor_msgs.msg")
    smm.Image = Image
    sm.msg = smm
    dm = types.ModuleType("diagnostic_msgs")
    dmm = types.ModuleType("diagnostic_msgs.msg")
    dmm.DiagnosticArray = DiagnosticArray
    dmm.DiagnosticStatus = DiagnosticStatus
    dmm.KeyValue = KeyValue
    dm.msg = dmm
    sys.modules.update({"sensor_msgs": sm, "sensor_msgs.msg": smm,
                        "diagnostic_msgs": dm, "diagnostic_msgs.msg": dmm})

    return msgs


def uninstall():
    for name in _EXTRA:
        sys.modules.pop(name, None)

    base.uninstall()

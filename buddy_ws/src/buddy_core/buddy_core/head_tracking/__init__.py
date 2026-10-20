"""BUDDY intelligent head tracking (pan/tilt).

See ``docs/head_tracking_architecture.md``.  Pure-Python cores; ROS 2 is
only used by ``ros_node`` through ``ros_adapter``.
"""

from buddy_core.head_tracking.config import (
    HeadTrackingConfig,
    HeadTrackingConfigError,
    config_from_dict,
    load_head_tracking_config,
)
from buddy_core.head_tracking.factory import (
    HeadTrackingSystem,
    build_head_system,
    hardware_gate_problems,
)
from buddy_core.head_tracking.manager import HeadTrackingManager
from buddy_core.head_tracking.models import EventKind, HeadState, Priority
from buddy_core.head_tracking.motion import HeadMotionCoordinator

__all__ = [
    "EventKind",
    "HeadMotionCoordinator",
    "HeadState",
    "HeadTrackingConfig",
    "HeadTrackingConfigError",
    "HeadTrackingManager",
    "HeadTrackingSystem",
    "Priority",
    "build_head_system",
    "config_from_dict",
    "hardware_gate_problems",
    "load_head_tracking_config",
]

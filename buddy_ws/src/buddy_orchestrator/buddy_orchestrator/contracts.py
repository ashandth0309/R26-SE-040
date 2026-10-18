"""Names and rules shared by every BUDDY component.

This module is the machine-readable half of
``docs/buddy_integration_interfaces.md``.  Nothing here imports ROS, so the
tests can check the contracts against the real ``.msg`` and ``.srv`` files
and against the nodes' own sources.

Version rules: ``INTERFACE_VERSION`` follows semantic versioning.  Removing
or renaming a topic, service or field is a major change; adding one is
minor.
"""

from __future__ import annotations

from enum import IntEnum

INTERFACE_VERSION = "1.0.0"


class Topic:
    """Topic names.  A comment says who may publish."""

    # Motion path (existing, unchanged)
    MOVEMENT_REQUEST = "/movement/request"    # nav; coordinator+voice STOP
    MOVEMENT_SAFE = "/movement/safe"          # safety_supervisor ONLY
    OBSTACLE_STATUS = "/obstacle_status"      # buddy_obstacle_sensor
    SAFETY_STATE = "/safety_state"            # safety_supervisor
    ROBOT_STATUS = "/robot_status"            # coordinator

    # Navigation (Yohan_Mapping)
    NAV_COMMAND = "/navigation/command"       # coordinator; voice STOP
    NAV_STATUS = "/navigation/status"         # buddy_navigation
    NAV_LOCAL_MAP = "/navigation/local_map"   # buddy_navigation

    # Voice (Ashandth_Voice)
    VOICE_STATUS = "/voice/status"            # buddy_voice_companion
    VOICE_COMMAND = "/voice/command"          # buddy_voice_companion
    VOICE_SAY = "/voice/say"                  # coordinator ONLY

    # Vision (Sobiya_AdvancedVision)
    CAMERA_IMAGE = "/camera/image_raw"        # shared camera node ONLY
    VISION_EVENTS = "/vision/events"
    VISION_PEOPLE = "/vision/people"
    VISION_OBJECTS = "/vision/objects"
    VISION_STATUS = "/vision/status"
    PERSON_RECOGNIZED = "/person/recognized"
    EMOTION_OBSERVATION = "/emotion/observation"
    SECURITY_EVENT = "/security/event"

    # Coordinator
    COORDINATOR_STATUS = "/coordinator/status"
    # Intake topics: the voice node publishes its motion requests here when
    # the coordinator runs, instead of straight to the motion path.
    VOICE_MOVEMENT_INTAKE = "/coordinator/voice_movement"
    VOICE_NAVIGATION_INTAKE = "/coordinator/voice_navigation"


class Service:
    """Service names."""

    ROUTE_UTTERANCE = "/coordinator/route_utterance"
    EMERGENCY_STOP = "/emergency_stop"
    RESET_EMERGENCY_STOP = "/reset_emergency_stop"
    VISION_COMMAND = "/vision/command"
    VISION_IDENTIFY = "/vision/identify_object"
    VISION_TEACH = "/vision/teach_object"
    VISION_WHERE_IS = "/vision/where_is"
    VISION_ENABLE = "/vision/enable"
    CAMERA_ENABLE = "/camera/enable"
    VOICE_MUTE = "/voice/mute"


class Err(IntEnum):
    """``error_code`` values used by coordinator services."""

    OK = 0
    UNAVAILABLE = 1        # a module needed for this is not running
    TIMEOUT = 2            # a module did not answer in time
    REFUSED_SAFETY = 3     # robot state, sensors or the supervisor said no
    REFUSED_CONSENT = 4    # needs a consent or invitation that is missing
    NOT_UNDERSTOOD = 5
    STALE = 6              # the data needed is too old to act on
    BUSY = 7               # another task owns the resource
    CANCELLED = 8
    HARDWARE_CONFLICT = 9  # two owners for one device


class SpeechPriority(IntEnum):
    """Same values as ``SpeechRequest.PRIORITY_*`` and the voice queue.

    The integration spec lists six speech classes; the speaker queue has
    four levels.  They map as follows (order inside a level is arrival
    order):

    * emergency and safety warnings         -> CRITICAL
    * user stop acknowledgements            -> CRITICAL (sent by voice)
    * important robot status                -> HIGH
    * direct answers to the user            -> NORMAL
    * proactive companion speech            -> LOW
    * listening backchannels                -> LOW (voice-internal)
    """

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


# How old a message may be before it is not acted on (seconds).
STALE_AFTER_S = {
    "safety_state": 2.0,        # published every 0.5 s
    "navigation_status": 3.0,   # about once per second
    "obstacle_status": 1.0,     # the supervisor's own limit
    "vision_people": 2.0,
    "vision_status": 3.0,
    "voice_status": 3.0,
}

# Hardware each node may own.  One owner per resource.
HARDWARE_OWNERS = {
    "microphones": ("buddy_voice_companion",),
    "speaker": ("buddy_voice_companion",),
    "camera": ("buddy_camera",),
    "ultrasonic": ("buddy_obstacle_sensor",),
    "motors": ("motor_executor",),
}

# Nodes that would fight an owner for the same hardware or topics.
# name -> (resources it would also claim, who it conflicts with)
LEGACY_NODES = {
    "buddy_voice": (("microphones", "speaker"), "buddy_voice_companion"),
}

# Topics that must have at most this many publishers in a running system.
PUBLISHER_LIMITS = {
    Topic.MOVEMENT_SAFE: 1,
    Topic.CAMERA_IMAGE: 1,
    Topic.OBSTACLE_STATUS: 1,
    Topic.VOICE_SAY: 1,
}

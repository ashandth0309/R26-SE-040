"""Transport adapter: ROS messages / JSON <-> head-tracking calls.

Pure Python (no rclpy import) so it is unit-tested without ROS.  Message
objects are read by attribute name only (duck typing), which lets the same
code accept Sobiya's ``VisionPeople`` and Ashandth's ``VoiceCommand`` even
though those message types are not in the base branch.

Interfaces actually present in the repository (base branch):
    /safety_state        buddy_interfaces/SafetyState     (present)
    /camera/image_raw    sensor_msgs/Image                (NOT used here)
Interfaces that exist only on other branches (guarded, optional):
    /vision/people       buddy_interfaces/VisionPeople    (Sobiya)
    /voice/command       buddy_interfaces/VoiceCommand    (Ashandth; its
                         wake_word_detected is currently always False)
    /voice/status        std_msgs/String JSON             (Ashandth)
Interfaces this package defines (std_msgs/String JSON, no new .msg files):
    /head/command, /head/hand_observation, /head/sound_direction (in)
    /head/status, /head/events (out)

Time: observations are stamped with RECEIPT time.  Sobiya's
``VisionPeople.stamp`` is the publish time, not the camera capture time,
so true camera latency is unknown and not compensated.
"""

from __future__ import annotations

import json
import math
from typing import Any

from buddy_core.head_tracking.manager import HeadTrackingManager
from buddy_core.head_tracking.models import (
    FaceFrame,
    FaceObservation,
    GestureEvent,
    HandObservation,
    HandPoint,
    HeadEvent,
    WakeWordEvent,
    is_number,
)
from buddy_core.head_tracking.sound_direction import (
    Reason,
    SoundDirectionResult,
)

SAFETY_EMERGENCY = 3          # SafetyState.SAFETY_EMERGENCY


def faces_from_people(msg: Any, width: int, height: int,
                      now: float) -> FaceFrame:
    """VisionPeople -> FaceFrame.  Only person_key and the box are read;
    display_name, expression and every other field are ignored."""

    faces = []
    for p in getattr(msg, "people", []) or []:
        key = str(getattr(p, "person_key", "") or "")
        vals = [getattr(p, n, None) for n in
                ("bbox_x", "bbox_y", "bbox_width", "bbox_height")]
        if not key or not all(is_number(v) for v in vals):
            continue
        x, y, w, h = (float(v) for v in vals)
        if w <= 0 or h <= 0:
            continue
        faces.append(FaceObservation(key, x, y, w, h, 1.0,
                                     bool(getattr(p, "known", False))))
    return FaceFrame(now, int(width), int(height), tuple(faces))


def gestures_from_people(msg: Any) -> tuple[GestureEvent, ...]:
    """Gesture names Sobiya attaches to each person (no side info)."""

    out = []
    for p in getattr(msg, "people", []) or []:
        key = str(getattr(p, "person_key", "") or "")
        for name in getattr(p, "gestures", []) or []:
            out.append(GestureEvent(str(name), key, None))
    return tuple(out)


def _point(raw: Any) -> HandPoint | None:
    try:
        x, y = float(raw[0]), float(raw[1])
        c = float(raw[2]) if len(raw) > 2 else 1.0
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if not all(math.isfinite(v) for v in (x, y, c)):
        return None
    return HandPoint(x, y, c)


def hand_from_json(text: str, now: float) -> HandObservation | None:
    """Parse the documented /head/hand_observation JSON, or None.

    {"width":640,"height":480,"track_id":"k","wrists":{"left":[x,y,c]},
     "fingertips":{"right":[x,y,c]},
     "gestures":[{"name":"raised_hand","track_id":"k","side":"right"}]}
    Pixel coordinates.  ``fingertips`` stays empty until a real model.
    """

    try:
        d = json.loads(text)
        width, height = int(d["width"]), int(d["height"])
    except (TypeError, ValueError, KeyError):
        return None
    if width <= 0 or height <= 0 or not isinstance(d, dict):
        return None

    def points(key):
        raw = d.get(key) or {}
        if not isinstance(raw, dict):
            return {}
        out = {}
        for side, val in raw.items():
            pt = _point(val)
            if side in ("left", "right") and pt is not None:
                out[side] = pt
        return out

    gestures = []
    for g in d.get("gestures") or []:
        if isinstance(g, dict) and isinstance(g.get("name"), str):
            side = g.get("side")
            gestures.append(GestureEvent(
                g["name"], str(g.get("track_id", "")),
                side if side in ("left", "right") else None))
    return HandObservation(now, width, height, str(d.get("track_id", "")),
                           points("wrists"), points("fingertips"),
                           tuple(gestures))


def sound_from_json(text: str, now: float) -> SoundDirectionResult:
    """Parse an upstream (mic-owning node's) direction estimate.

    {"ok":true,"azimuth_deg":-32.5,"confidence":0.8,"synchronized":true}
    ``synchronized`` must be true or the estimate is refused.
    """

    def bad(reason):
        return SoundDirectionResult(False, reason, t=now)

    try:
        d = json.loads(text)
        ok = d.get("ok") is True
        sync = d.get("synchronized") is True
        az = d.get("azimuth_deg")
        conf = d.get("confidence", 0.0)
    except (TypeError, ValueError, AttributeError):
        return bad(Reason.BAD_INPUT)
    if not sync:
        return bad(Reason.UNSYNCHRONIZED)
    if not ok:
        return bad(str(d.get("reason", Reason.LOW_CONFIDENCE))[:60])
    if not (is_number(az) and is_number(conf)) or abs(az) > 90:
        return bad(Reason.BAD_INPUT)
    return SoundDirectionResult(True, Reason.OK, t=now,
                                azimuth_deg=float(az),
                                confidence=float(conf))


def event_json(ev: HeadEvent) -> str:
    return json.dumps({"event": ev.kind.value, "t": round(ev.t, 3),
                       "data": ev.data}, default=str, sort_keys=True)


class HeadTrackingBridge:
    """Routes decoded inputs into the manager; builds outputs."""

    def __init__(self, manager: HeadTrackingManager,
                 frame_width: int = 640, frame_height: int = 480) -> None:
        self.mgr = manager
        self.width, self.height = int(frame_width), int(frame_height)
        self.mics_synchronized = False   # learned from /voice/status

    # ---- inputs
    def on_people(self, msg: Any, now: float) -> None:
        self.mgr.on_face_frame(
            faces_from_people(msg, self.width, self.height, now), now)
        gestures = gestures_from_people(msg)
        if gestures:
            self.mgr.on_hand_observation(HandObservation(
                now, self.width, self.height, "", {}, {}, gestures), now)

    def on_voice_command(self, msg: Any, now: float) -> None:
        if getattr(msg, "wake_word_detected", False) is True:
            self.mgr.on_wake_word(WakeWordEvent(now), now)

    def on_voice_status(self, text: str, now: float) -> None:
        try:
            d = json.loads(text)
            state = str(d.get("state", ""))
            topo = str(d.get("mic_topology", ""))
        except (TypeError, ValueError, AttributeError):
            return
        self.mgr.set_speaking(state == "BUDDY_SPEAKING", now)
        self.mics_synchronized = topo == "stereo"

    def on_safety(self, msg: Any, now: float) -> None:
        active = bool(getattr(msg, "emergency_stop_active", False)) or (
            getattr(msg, "safety_level", 0) == SAFETY_EMERGENCY)
        self.mgr.set_emergency_stop(
            active, now, str(getattr(msg, "reason", "") or "")[:80])

    def on_hand_json(self, text: str, now: float) -> bool:
        obs = hand_from_json(text, now)
        if obs is None:
            return False
        self.mgr.on_hand_observation(obs, now)
        return True

    def on_sound_json(self, text: str, now: float) -> None:
        self.mgr.on_sound_estimate(sound_from_json(text, now), now)

    def on_command(self, text: str, now: float) -> bool:
        parts = str(text).strip().lower().split()
        if not parts:
            return False
        cmd, arg = parts[0], (parts[1] if len(parts) > 1 else None)
        if cmd == "stop":
            self.mgr.user_stop(now)
        elif cmd == "resume":
            self.mgr.resume(now)
        elif cmd == "reset_fault":
            self.mgr.reset_fault(now)
        elif cmd == "hand_on" and arg in (None, "left", "right"):
            self.mgr.activate_hand(now, arg)
        elif cmd == "hand_off":
            self.mgr.deactivate_hand(now, "command")
        elif cmd == "wake":
            self.mgr.on_wake_word(WakeWordEvent(now), now)
        else:
            return False
        return True

    # ---- outputs
    def status_json(self, now: float) -> str:
        st = self.mgr.status(now)
        st["mics_synchronized"] = self.mics_synchronized
        return json.dumps(st, sort_keys=True)

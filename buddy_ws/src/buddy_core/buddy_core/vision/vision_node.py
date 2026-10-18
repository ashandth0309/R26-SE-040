"""ROS 2 vision node: turns camera images into people, objects and events.

Consumes   /camera/image_raw       sensor_msgs/Image   (never opens the camera)
           /voice/status           std_msgs/String     (JSON; only to avoid
                                                        interrupting speech)
Publishes  /vision/people          buddy_interfaces/VisionPeople
           /vision/objects         buddy_interfaces/VisionObjects
           /vision/events          buddy_interfaces/VisionEvent
           /vision/status          std_msgs/String     (JSON; no names)
           /person/recognized      buddy_interfaces/RecognizedPerson
           /emotion/observation    buddy_interfaces/EmotionObservation
           /security/event         buddy_interfaces/SecurityEvent (falls)
Services   /vision/command         VisionCommand   settings, privacy, replies
           /vision/identify_object IdentifyObject  "what is this?"
           /vision/teach_object    TeachObject     teach, correct, forget
           /vision/where_is        WhereIs         last seen place
           /vision/enable          std_srvs/SetBool  camera privacy switch

It has no motion interface of any kind.  Approach and follow policies
exist (see approach.py, follow.py) but are NOT connected to ROS: they use
a mock adapter until navigation is validated.

Topic names marked (assumed) are not defined by any existing doc and must
be confirmed with their consumers.  UNVERIFIED under real ROS 2.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from buddy_core.vision.context import ContextEvent, VisualContext
from buddy_core.vision.multimodal import observation_fields
from buddy_core.vision.object_teach import (
    ObjectRecognizer, TeachSession, correct, forget, forget_everything,
    identify, select_target)
from buddy_core.vision.objects import ObjectDetection
from buddy_core.vision.vision_factory import BuiltVision

SPOKEN = {
    "fall_suspected": "Are you okay? I think you may have fallen.",
    "fall_alert": "I didn't hear back, so I'm sending an alert.",
    "fall_recovered": "Good, I'm glad you're okay.",
}


class CommandResult(NamedTuple):
    handled: bool
    reply: str
    pass_to_dialogue: bool


@dataclass
class Outputs:
    frame: bool = False          # True only for output of a processed frame
    people: list[dict] = field(default_factory=list)
    objects: dict | None = None
    events: list[dict] = field(default_factory=list)
    emotions: list[dict] = field(default_factory=list)
    recognized: list[dict] = field(default_factory=list)
    security: list[dict] = field(default_factory=list)


def _ev(kind: str, severity: str = "info", person: str = "", text: str = "",
        needs_check: bool = False, data: dict | None = None) -> dict:
    return {"kind": kind, "person_key": person, "severity": severity,
            "suggested_text": text, "needs_voice_check": needs_check,
            "data_json": json.dumps(data or {}, default=str)}


class VisionNodeCore:
    """All the logic of the node, testable without ROS."""

    def __init__(self, built: BuiltVision, max_process_hz: float = 10.0,
                 clock=time.monotonic, stale_after_s: float = 2.0) -> None:
        self.b = built
        self._clock = clock
        self._min_dt = 1.0 / max_process_hz if max_process_hz > 0 else 0.0
        self._lock = threading.RLock()
        self._last_proc = -1e9
        self._last_image: np.ndarray | None = None
        self._ctx: VisualContext | None = None
        self._emitted_emotion: dict[str, tuple[str, float]] = {}
        self._emotion_time: dict[str, float] = {}
        self._teach: TeachSession | None = None
        self._deferred = Outputs()
        self._stale_after = stale_after_s
        self._last_frame_t = clock()
        self.camera_ok = True
        self.frames_in = 0
        self.frames_dropped = 0
        self.safety = {"emergency": False, "movement_allowed": True}

        if built.object_embedder is not None:
            self._teach = TeachSession(built.objects,
                                       built.object_embedder)

    # ----- inputs -----

    def on_voice_status(self, status: dict) -> None:
        with self._lock:
            m = self.b.manager
            m.voice_speaking = str(status.get("state", "")).upper() in (
                "USER_SPEAKING", "BUDDY_SPEAKING", "PROCESSING")
            m.quiet_mode = str(status.get("silence_mode", "normal")) != \
                "normal" or bool(status.get("muted", False))
            m.robot_busy = bool(status.get("moving", False))

    def on_safety(self, emergency: bool, movement_allowed: bool) -> None:
        with self._lock:
            self.safety = {"emergency": emergency,
                           "movement_allowed": movement_allowed}

    def on_frame(self, image: np.ndarray, day: int = 0) -> Outputs | None:
        with self._lock:
            now = self._clock()
            self.frames_in += 1
            self._last_frame_t = now

            if not self.camera_ok:
                self.camera_ok = True
                self._deferred.events.append(_ev(
                    "camera_restored", "info", text=""))

            if now - self._last_proc < self._min_dt:
                self.frames_dropped += 1

                return None

            self._last_proc = now
            self._last_image = image
            ctx = self.b.manager.process(image, now, day=day)
            self._ctx = ctx

            return self._outputs(ctx, now)

    def check_camera(self, now: float | None = None) -> None:
        """Call about once a second.  If frames stop arriving, drop all
        perception state, clear the voice side's last emotion hint and
        report it.  Vision-dependent actions then have nothing to act on.
        A user who switched the camera off is not a failure.
        """

        with self._lock:
            now = self._clock() if now is None else now

            if not self.b.manager.settings.camera_enabled or \
                    not self.camera_ok:
                return

            if now - self._last_frame_t <= self._stale_after:
                return

            self.camera_ok = False
            self.b.manager.reset()
            self._ctx, self._last_image = None, None

            if self._teach is not None and self._teach.active:
                self._teach.cancel("Okay, never mind.")

            self._clear_emotions(self._deferred, "camera_lost")
            self._deferred.events.append(_ev(
                "camera_lost", "warn",
                text="My camera isn't working right now, so I can't see.",
                data={"silent_for_s": round(now - self._last_frame_t, 1)}))

    # ----- output assembly -----

    def _outputs(self, ctx: VisualContext, now: float) -> Outputs:
        out = Outputs(frame=True)

        if ctx.paused:
            self._clear_emotions(out, "camera_off")

            return out

        for p in ctx.people:
            out.people.append({
                "person_key": p.key,
                "display_name": p.name or "", "known": p.person_id
                is not None, "identity_state": p.identity_state or "",
                "posture": p.posture or "", "motion": p.motion or "",
                "expression_group": p.expression_group or "",
                "expression_confidence": float(p.expression_confidence
                                               or 0.0),
                "expression_agreement": p.agreement or "",
                "fall_state": p.fall_state or "",
                "gestures": list(p.gestures),
                "box": p.box or (0, 0, 0, 0)})
            x1, y1, x2, y2 = p.box or (0, 0, 0, 0)
            out.recognized.append({
                "person_id": p.person_id or p.key,
                "display_name": p.name or "", "known": p.person_id
                is not None, "confidence": 1.0 if p.person_id else 0.0,
                "bbox": (x1, y1, x2 - x1, y2 - y1)})

        if "objects" in ctx.stages_run:
            out.objects = {"place": ctx.place or "", "objects": [
                {"object_id": str(o["id"]), "label": o["label"],
                 "name": o["name"] or "", "category": o["category"],
                 "confidence": float(o["score"]), "box": o["box"]}
                for o in ctx.objects]}

        self._emotions(ctx, now, out)

        for e in ctx.events:
            self._event(e, ctx, out)

        if self._teach is not None and self._teach.active:
            self._feed_teach(ctx, now, out)

        return out

    def _emotions(self, ctx, now, out) -> None:
        live = set()

        for p in ctx.people:
            if p.estimate is None:
                continue

            live.add(p.key)
            msg = observation_fields(p.estimate, p.key)
            prev = self._emitted_emotion.get(p.key)
            due = prev is None or prev[0] != msg["expression"] or \
                now - self._emotion_time.get(p.key, 0.0) >= 2.0

            if due:
                self._emitted_emotion[p.key] = (msg["expression"],
                                                msg["confidence"])
                self._emotion_time[p.key] = now
                out.emotions.append(msg)

        self._clear_emotions(out, "no_longer_visible", keep=live)

    def _clear_emotions(self, out, why, keep=frozenset()) -> None:
        """The voice side never expires a vision hint, so say 'unknown'."""

        for key, (word, _c) in list(self._emitted_emotion.items()):
            if key in keep:
                continue

            if word != "unknown":
                out.emotions.append({"person_id": key,
                                     "expression": "unknown",
                                     "confidence": 0.0,
                                     "source": "vision_expression"})

            del self._emitted_emotion[key]
            self._emotion_time.pop(key, None)

    def _event(self, e: ContextEvent, ctx, out: Outputs) -> None:
        d = e.data
        person = d.get("person") or d.get("key") or d.get("body") or ""
        text = SPOKEN.get(e.kind, "")
        sev = str(d.get("severity", "info"))
        needs = bool(d.get("needs_voice_check", False))

        if e.kind == "greeting_due":
            text = d.get("preferred") or f"Hello {d.get('name') or ''}!"
            text = text.replace("  ", " ").strip()
            self.b.manager.c.greeting.mark_greeted(d["person"])
        elif e.kind in ("proactive_offer", "proactive_ack"):
            text = d.get("text", "")
        elif e.kind == "gesture" and d.get("gesture") == "wave":
            text = "Hi!"

        out.events.append(_ev(e.kind, sev, person, text, needs, d))

        if e.kind == "fall_alert":
            out.security.append({
                "event_id": f"fall_{int(e.t * 1000)}", "event_type": 4,
                "severity": 3, "person_id": "",
                "description": "Possible fall: no response after Buddy "
                               "asked. No image was saved."})

    # ----- teaching -----

    def _detections(self, ctx) -> list[ObjectDetection]:
        return [ObjectDetection(o["label"], o["score"], tuple(o["box"]),
                                o["category"]) for o in ctx.objects]

    def _feed_teach(self, ctx, now, out) -> None:
        faces = [p.box for p in ctx.people if p.box]
        r = self._teach.feed(self._last_image, self._detections(ctx), now,
                             faces)

        if r is not None:
            out.events.append(_ev("teach_prompt", text=r.text,
                                  data={"phase": r.phase.value}))

    # ----- services -----

    def _me(self) -> str | None:
        if self._ctx is None:
            return None

        rec = [p for p in self._ctx.people if p.person_id]
        rec.sort(key=lambda p: -((p.box[2] - p.box[0]) *
                                 (p.box[3] - p.box[1])) if p.box else 0)

        return rec[0].person_id if rec else None

    def awaiting_reply(self) -> bool:
        with self._lock:
            pm = self.b.manager.c.proactive
            fall = self.b.manager.c.fall
            asked = fall is not None and any(
                s in ("suspect", "asked") for s in fall.states().values())
            offered = pm is not None and pm.state.value in (
                "interaction_offered", "conversation_active")

            return bool(asked or offered or
                        self.b.controller.awaiting_confirmation or
                        (self._teach is not None and self._teach.active))

    def command(self, text: str, person_id: str = "") -> CommandResult:
        """A spoken utterance that may be addressed to vision.

        ``handled`` means vision produced (or consumed) something;
        ``pass_to_dialogue`` tells the voice engine to also answer the
        utterance normally, e.g. "Yes, today was difficult" after an offer.
        """

        with self._lock:
            now = self._clock()
            reply = self.b.controller.handle(text, person_id or self._me(),
                                             now)

            if reply is not None:
                return CommandResult(True, reply, False)

            events = self.b.manager.on_voice_reply(text, now)

            if self._teach is not None and self._teach.active:
                return CommandResult(True, self._teach.answer(text).text,
                                     False)

            events = [e for e in events if e.kind != "proactive_none"]
            pass_on = False
            texts = []

            for e in events:
                self._event(e, None, self._deferred)
                accepted = e.kind == "proactive_ack" and \
                    e.data.get("state") == "conversation_active"

                if accepted and len(text.split()) > 3:
                    pass_on = True      # they said more: let voice answer
                    continue

                texts.append(SPOKEN.get(e.kind, "") or e.data.get("text", ""))

            if not events:
                return CommandResult(False, "", True)

            return CommandResult(True, " ".join(t for t in texts if t),
                                 pass_on)

    def take_deferred(self) -> Outputs:
        """Events caused by a spoken reply, to publish without a frame."""

        with self._lock:
            out, self._deferred = self._deferred, Outputs()

            return out

    def identify_object(self) -> dict:
        with self._lock:
            b = self.b

            if self._last_image is None or b.object_embedder is None or \
                    b.manager.c.objects is None:
                return {"success": False, "kind": "unavailable",
                        "text": "I can't look at objects right now.",
                        "name": "", "label": "", "offer_teach": False,
                        "confidence": 0.0}

            dets = [d for d in b.manager.c.objects.detect(self._last_image)
                    if d.label != "person"]
            faces = [p.box for p in (self._ctx.people if self._ctx else ())
                     if p.box]
            r = identify(self._last_image, dets, b.object_embedder,
                         ObjectRecognizer(b.objects), faces=faces)

            return {"success": True, "kind": r.kind, "text": r.text,
                    "name": r.name or "", "label": r.label or "",
                    "offer_teach": r.offer_teach,
                    "confidence": float(r.confidence)}

    def teach_object(self, action: str, text: str) -> dict:
        with self._lock:
            b, t = self.b, self._teach
            act = (action or "").lower()

            def res(reply, phase="idle", saved=False, ok=True):
                return {"ok": ok, "reply": reply, "phase": phase,
                        "saved": saved}

            if act == "list":
                names = [o["name"] for o in b.objects.list()]

                return res("I know: " + ", ".join(names) + "." if names
                           else "You haven't taught me anything yet.")

            if act == "forget":
                return res(forget(b.objects, text))

            if act == "forget_all":
                return res(forget_everything(b.objects))

            if t is None:
                return res("I can't learn objects right now.", ok=False)

            if act == "start":
                r = t.start(text, self._clock())
            elif act == "answer":
                r = t.answer(text)
            elif act == "cancel":
                r = t.cancel()
            elif act == "correct":
                return self._correct(text, res)
            else:
                return res("Unknown action.", ok=False)

            return res(r.text, r.phase.value, r.saved)

    def _correct(self, text: str, res) -> dict:
        wrong, _, right = (text or "").partition("->")

        if self._last_image is None or self.b.object_embedder is None:
            return res("I can't see it right now.", ok=False)

        dets = self._detections(self._ctx) if self._ctx else []
        tgt = select_target(dets, self._last_image.shape)
        from buddy_core.vision.object_teach import crop
        vec = self.b.object_embedder.embed(crop(self._last_image, tgt.box))

        return res(correct(self.b.objects, wrong.strip(), vec,
                           right.strip() or None))

    def where_is(self, name: str) -> dict:
        with self._lock:
            if self.b.memory is None:
                return {"found": False,
                        "text": "I'm not keeping track of where things "
                                "are."}

            a = self.b.memory.where_is(name, self._clock())

            return {"found": a.found, "text": a.text}

    def set_camera_enabled(self, enabled: bool) -> tuple[bool, str]:
        with self._lock:
            from dataclasses import replace
            s = replace(self.b.manager.settings, camera_enabled=enabled)
            (self.b.apply_settings or self.b.manager.apply_settings)(s)
            self.b.controller.settings = s

            return True, "camera on" if enabled else "camera off"

    def set_profile(self, profile: str) -> None:
        with self._lock:
            from dataclasses import replace
            s = replace(self.b.manager.settings, profile=profile)
            s.validate()
            (self.b.apply_settings or self.b.manager.apply_settings)(s)
            self.b.controller.settings = s

    # ----- status -----

    def status(self) -> dict:
        with self._lock:
            m = self.b.manager
            ctx = self._ctx

            return {
                "camera_enabled": m.settings.camera_enabled,
                "camera_ok": self.camera_ok,
                "profile": m.settings.profile,
                "object_memory": m.settings.object_memory,
                "frames_in": self.frames_in,
                "frames_dropped": self.frames_dropped,
                "paused_frames": m.paused_frames,
                "people_in_view": len(ctx.people) if ctx else 0,
                "stages_run": list(ctx.stages_run) if ctx else [],
                "stages_skipped": dict(ctx.stages_skipped) if ctx else {},
                "timings_ms": {k: round(v, 1) for k, v in
                               (ctx.timings_ms if ctx else {}).items()},
                "unavailable": dict(self.b.unavailable),
                "awaiting_reply": self.awaiting_reply(),
                "teaching": bool(self._teach and self._teach.active),
                "motion_interface": "none",
                "approach_follow": "mock adapter only; not wired to ROS",
            }


# ---------------------------------------------------------------------------
# ROS wrapper.  Everything above is plain Python and is tested directly.
# ---------------------------------------------------------------------------

def image_to_bgr(msg) -> np.ndarray | None:
    """sensor_msgs/Image to a BGR array; None if the format is unsupported."""

    h, w = int(msg.height), int(msg.width)
    enc = str(msg.encoding).lower()

    if h <= 0 or w <= 0 or enc not in ("bgr8", "rgb8"):
        return None

    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    if data.size != h * w * 3 or int(msg.step) not in (0, w * 3):
        return None

    img = data.reshape(h, w, 3)

    return img[:, :, ::-1].copy() if enc == "rgb8" else img.copy()


def build_node_class():
    """Defer the ROS imports so the module imports (and tests) without ROS."""

    import rclpy  # noqa: F401
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from std_srvs.srv import SetBool

    from buddy_interfaces.msg import (
        EmotionObservation, RecognizedPerson, SecurityEvent, VisionEvent,
        VisionObject, VisionObjects, VisionPeople, VisionPerson)
    from buddy_interfaces.srv import (
        IdentifyObject, TeachObject, VisionCommand, WhereIs)

    class VisionNode(Node):
        def __init__(self) -> None:
            super().__init__("buddy_vision")
            p = self._param
            from buddy_core.vision.vision_factory import build_vision

            built = build_vision(
                data_dir=str(p("data_dir", "")) or None, settings=None,
                models_dir=str(p("models_dir", "")) or None,
                in_memory=bool(p("in_memory", False)))
            self.core = VisionNodeCore(
                built, max_process_hz=float(p("max_process_hz", 10.0)))
            profile = str(p("profile", ""))

            if profile:                 # empty keeps the saved choice
                self.core.set_profile(profile)
            self._latest = None
            self._latest_t = 0.0
            self._max_age = float(p("max_frame_age_s", 1.0))
            self._lock = threading.Lock()
            self._seq = 0
            self._done = 0

            self.p_people = self.create_publisher(
                VisionPeople, p("people_topic", "/vision/people"), 5)
            self.p_objects = self.create_publisher(
                VisionObjects, p("objects_topic", "/vision/objects"), 5)
            self.p_events = self.create_publisher(
                VisionEvent, p("events_topic", "/vision/events"), 20)
            self.p_status = self.create_publisher(
                String, p("status_topic", "/vision/status"), 5)
            self.p_person = self.create_publisher(
                RecognizedPerson, p("person_topic", "/person/recognized"), 5)
            self.p_emotion = self.create_publisher(
                EmotionObservation,
                p("emotion_topic", "/emotion/observation"), 10)
            self.p_security = self.create_publisher(
                SecurityEvent, p("security_topic", "/security/event"), 10)

            self.create_subscription(
                Image, p("image_topic", "/camera/image_raw"),
                self._on_image, 1)
            self.create_subscription(
                String, p("voice_status_topic", "/voice/status"),
                self._on_voice, 5)

            self.create_service(VisionCommand, "/vision/command",
                                self._srv_command)
            self.create_service(IdentifyObject, "/vision/identify_object",
                                self._srv_identify)
            self.create_service(TeachObject, "/vision/teach_object",
                                self._srv_teach)
            self.create_service(WhereIs, "/vision/where_is", self._srv_where)
            self.create_service(SetBool, "/vision/enable", self._srv_enable)
            self.create_timer(
                1.0 / max(1.0, float(p("max_process_hz", 10.0))), self._tick)
            self.create_timer(0.5, self._watchdog)
            self.create_timer(1.0, self._publish_status)
            self._classes = (VisionObject, VisionPerson)
            self.get_logger().info(
                "vision node ready; perception only, no motion interface")

        def _param(self, name, default):
            self.declare_parameter(name, default)

            return self.get_parameter(name).value

        # ----- inputs -----

        def _on_image(self, msg) -> None:
            img = image_to_bgr(msg)

            if img is None:
                return

            with self._lock:
                self._latest, self._latest_t = img, time.monotonic()
                self._seq += 1

        def _on_voice(self, msg) -> None:
            try:
                self.core.on_voice_status(json.loads(msg.data))
            except (ValueError, TypeError):
                pass

        def _tick(self) -> None:
            with self._lock:
                img, t, seq = self._latest, self._latest_t, self._seq

            if img is None or seq == self._done:
                return

            self._done = seq

            if time.monotonic() - t > self._max_age:
                self.core.frames_dropped += 1

                return

            out = self.core.on_frame(img)
            self._publish(self.core.take_deferred())

            if out is not None:
                self._publish(out)

        # ----- outputs -----

        def _stamp(self):
            return self.get_clock().now().to_msg()

        def _publish(self, out: Outputs) -> None:
            VObj, VPer = self._classes
            people = VisionPeople()
            people.stamp = self._stamp()

            for d in out.people:
                m = VPer()
                x1, y1, x2, y2 = d["box"]

                for k in ("person_key", "display_name", "known",
                          "identity_state", "posture", "motion",
                          "expression_group", "expression_confidence",
                          "expression_agreement", "fall_state"):
                    setattr(m, k, d[k])

                m.gestures = list(d["gestures"])
                m.bbox_x, m.bbox_y = float(x1), float(y1)
                m.bbox_width, m.bbox_height = float(x2 - x1), float(y2 - y1)
                people.people.append(m)

            if out.frame:
                self.p_people.publish(people)

            for d in out.recognized:
                m = RecognizedPerson()
                m.stamp = self._stamp()
                m.person_id, m.display_name = d["person_id"], d["display_name"]
                m.known, m.confidence = d["known"], float(d["confidence"])
                m.bbox_x, m.bbox_y, m.bbox_width, m.bbox_height = (
                    float(v) for v in d["bbox"])
                self.p_person.publish(m)

            if out.objects is not None:
                m = VisionObjects()
                m.stamp, m.place = self._stamp(), out.objects["place"]

                for o in out.objects["objects"]:
                    v = VObj()
                    v.object_id, v.label = o["object_id"], o["label"]
                    v.name, v.category = o["name"], o["category"]
                    v.confidence = o["confidence"]
                    v.x1, v.y1, v.x2, v.y2 = (float(c) for c in o["box"])
                    m.objects.append(v)

                self.p_objects.publish(m)

            for d in out.emotions:
                m = EmotionObservation()
                m.stamp = self._stamp()
                m.person_id, m.expression = d["person_id"], d["expression"]
                m.confidence, m.source = float(d["confidence"]), d["source"]
                self.p_emotion.publish(m)

            for d in out.events:
                m = VisionEvent()
                m.stamp = self._stamp()

                for k in ("kind", "person_key", "severity", "suggested_text",
                          "needs_voice_check", "data_json"):
                    setattr(m, k, d[k])

                self.p_events.publish(m)

            for d in out.security:
                m = SecurityEvent()
                m.stamp = self._stamp()
                m.event_id, m.event_type = d["event_id"], d["event_type"]
                m.severity, m.person_id = d["severity"], d["person_id"]
                m.description = d["description"]
                self.p_security.publish(m)

        def _watchdog(self) -> None:
            self.core.check_camera()
            self._publish(self.core.take_deferred())

        def _publish_status(self) -> None:
            s = String()
            s.data = json.dumps(self.core.status())
            self.p_status.publish(s)

        # ----- services -----

        def _srv_command(self, req, res):
            r = self.core.command(req.text, req.person_id)
            res.handled, res.reply = r.handled, r.reply
            res.pass_to_dialogue = r.pass_to_dialogue
            self._publish(self.core.take_deferred())

            return res

        def _srv_identify(self, req, res):
            r = self.core.identify_object()
            res.success, res.kind, res.text = r["success"], r["kind"], \
                r["text"]
            res.name, res.label = r["name"], r["label"]
            res.offer_teach = r["offer_teach"]
            res.confidence = float(r["confidence"])

            return res

        def _srv_teach(self, req, res):
            r = self.core.teach_object(req.action, req.text)
            res.ok, res.reply, res.phase, res.saved = (
                r["ok"], r["reply"], r["phase"], r["saved"])

            return res

        def _srv_where(self, req, res):
            r = self.core.where_is(req.name)
            res.found, res.text = r["found"], r["text"]

            return res

        def _srv_enable(self, req, res):
            res.success, res.message = self.core.set_camera_enabled(
                bool(req.data))

            return res

        def destroy_node(self) -> bool:
            self.core.b.store.close()

            return super().destroy_node()

    return VisionNode


def main(args=None) -> None:
    import rclpy

    rclpy.init(args=args)
    node = build_node_class()()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

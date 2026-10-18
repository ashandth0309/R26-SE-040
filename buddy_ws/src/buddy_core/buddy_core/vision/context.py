"""VisualContextManager: one place that runs the vision stages and hands
the rest of Buddy a single, small, honest picture of what the camera sees.

What it does
* Runs each stage (faces, expression, objects, pose/activity/fall/gesture,
  scene) on its own schedule from the chosen profile, in priority order,
  and skips optional stages when a frame goes over its time budget.
* Applies privacy rules before any analysis: camera off means nothing is
  processed; expression is read only for recognised people who opted in;
  object memory is off unless switched on.
* Turns stage outputs into events (person appeared, greeting due, fall
  check, proactive offer, ...) for the voice engine to use or ignore.

What it never does: move the robot, speak, save images, or bypass the
safety supervisor.  Everything it knows is passed in; nothing is imported
that opens a camera or a microphone.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from buddy_core.vision.activity import ActivityTracker
from buddy_core.vision.fall import FallDetector
from buddy_core.vision.gestures import GestureRecognizer
from buddy_core.vision.multimodal import fuse
from buddy_core.vision.object_stabilizer import ObjectStabilizer
from buddy_core.vision.proactive import (
    ProactiveContext, ProactiveInteractionManager)
from buddy_core.vision.tracker import MultiTracker, TrackerConfig
from buddy_core.vision.vision_settings import VisionSettings

FURNITURE = {"bed", "couch"}
REMEMBER_CLASSES = {"remote", "cell phone", "book", "laptop", "cup",
                    "bottle", "backpack", "handbag", "umbrella", "scissors"}


@dataclass
class Components:
    faces: object | None = None             # FaceTrackingPipeline
    expression: object | None = None        # .predict(image, det)
    expr_service: object | None = None      # ExpressionService
    objects: object | None = None           # .detect(bgr) incl. persons
    stabilizer: ObjectStabilizer | None = None
    namer: Callable | None = None           # (image, box, label) -> name|None
    pose: object | None = None              # .estimate(bgr, box)
    activity: ActivityTracker | None = None
    fall: FallDetector | None = None
    gestures: GestureRecognizer | None = None
    memory: object | None = None            # ObjectMemory
    scene: Callable | None = None           # (image) -> place name | None
    proactive: ProactiveInteractionManager | None = None
    greeting: object | None = None          # GreetingPolicy
    store: object | None = None             # FaceProfileStore


@dataclass(frozen=True)
class PersonContext:
    key: str
    session_id: str | None
    person_id: str | None
    name: str | None
    identity_state: str | None
    box: tuple[float, float, float, float] | None
    expression_group: str | None = None
    expression_confidence: float | None = None
    agreement: str | None = None
    posture: str | None = None
    motion: str | None = None
    fall_state: str | None = None
    gestures: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextEvent:
    kind: str
    t: float
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VisualContext:
    stamp: float
    paused: bool
    camera_fresh: bool
    people: tuple[PersonContext, ...]
    objects: tuple[dict, ...]
    place: str | None
    events: tuple[ContextEvent, ...]
    stages_run: tuple[str, ...]
    stages_skipped: dict
    timings_ms: dict


STAGE_ORDER = ("faces", "objects", "pose", "expression", "scene")


def _new_bodies() -> MultiTracker:
    # person boxes arrive about once a second, so be patient with gaps
    return MultiTracker(TrackerConfig(n_init=1, lost_after_s=3.0,
                                      max_age_s=6.0))


@dataclass
class ManagerConfig:
    frame_budget_ms: float | None = None   # None = no budget
    max_frame_age_s: float = 1.0
    max_pose_people: int = 2
    body_lost_s: float = 5.0
    mark_seen_every_s: float = 60.0
    ema: float = 0.3


class VisualContextManager:
    def __init__(self, comps: Components,
                 settings: VisionSettings | None = None,
                 cfg: ManagerConfig | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.c, self.cfg, self._clock = comps, cfg or ManagerConfig(), clock
        self.settings = settings or VisionSettings()
        self.bodies = _new_bodies()
        self._last_run: dict[str, float] = {}
        self._cost: dict[str, float] = {}
        self._expr: dict[int, object] = {}
        self._expr_stamp: dict[int, float] = {}
        self._known_people: dict[str, float] = {}
        self._body_seen: dict[int, float] = {}
        self._body_act: dict[int, object] = {}
        self._body_gest: dict[int, tuple] = {}
        self._body_face: dict[int, int] = {}
        self._names: dict[str, str | None] = {}
        self._invited: str | None = None
        self.place: str | None = None
        self.voice_speaking = False
        self.quiet_mode = False
        self.robot_busy = False
        self.voice_affect: Callable[[str], object] | None = None
        self._day = 0
        self._last_mark: dict[str, float] = {}
        self.frames = 0
        self.paused_frames = 0
        self._pending: list[ContextEvent] = []

    # ----- controls from other components -----

    def apply_settings(self, settings: VisionSettings) -> None:
        settings.validate()
        was = self.settings.camera_enabled
        self.settings = settings

        if was and not settings.camera_enabled:
            self.reset()

    def reset(self) -> None:
        """Forget all transient perception state (e.g. camera switched off)."""

        self.bodies = _new_bodies()
        self._expr.clear()
        self._expr_stamp.clear()
        self._body_seen.clear()
        self._body_act.clear()
        self._body_gest.clear()
        self._body_face.clear()
        self._known_people.clear()
        self._last_run.clear()
        self.place = None

        if self.c.stabilizer:
            self.c.stabilizer = ObjectStabilizer(self.c.stabilizer.cfg)

        if self.c.activity:
            self.c.activity = ActivityTracker(self.c.activity.cfg,
                                              self.c.activity.pcfg)

        if self.c.fall:
            self.c.fall = FallDetector(self.c.fall.cfg, self.c.fall.pcfg)

        if self.c.gestures:
            self.c.gestures = GestureRecognizer(self.c.gestures.cfg)

        if self.c.expr_service:
            self.c.expr_service.prune(set())

    def invite(self, person_key: str) -> None:
        """The person asked for company; lets a proactive offer through."""

        self._invited = person_key

    def on_voice_reply(self, text: str, now: float) -> list[ContextEvent]:
        out: list[ContextEvent] = []

        if self.c.fall:
            for e in self.c.fall.on_voice(text, now):
                out.append(ContextEvent(e.kind, now, {
                    "severity": e.severity, "reasons": e.reasons,
                    "track": e.track_id}))

        if self.c.proactive:
            s = self.c.proactive.on_reply(text, now, self._day)

            if s.kind != "none":
                out.append(ContextEvent("proactive_" + s.kind, now, {
                    "text": s.text, "person": s.person_key,
                    "state": s.state.value}))

        return out

    # ----- scheduling -----

    def _due(self, stage: str, now: float) -> bool:
        interval = self.settings.plan().get(stage)

        if interval is None:
            return False

        return now - self._last_run.get(stage, -1e9) >= interval

    def _affordable(self, stage: str, spent_ms: float) -> bool:
        b = self.cfg.frame_budget_ms

        return b is None or spent_ms + self._cost.get(stage, 0.0) <= b

    def _timed(self, stage: str, fn, timings: dict, now: float):
        t0 = time.perf_counter()
        out = fn()
        ms = (time.perf_counter() - t0) * 1000.0
        timings[stage] = ms
        a = self.cfg.ema
        self._cost[stage] = ms if stage not in self._cost else \
            a * ms + (1 - a) * self._cost[stage]
        self._last_run[stage] = now

        return out

    # ----- main entry -----

    def process(self, image: np.ndarray, now: float | None = None,
                frame_stamp: float | None = None, day: int = 0
                ) -> VisualContext:
        now = self._clock() if now is None else now
        self._day = day
        self.frames += 1
        events: list[ContextEvent] = []

        if not self.settings.camera_enabled or image is None:
            self.paused_frames += 1

            return VisualContext(now, True, False, (), (), None, (), (),
                                 {"all": "camera_off"}, {})

        fresh = (frame_stamp is None or
                 now - frame_stamp <= self.cfg.max_frame_age_s)
        timings: dict[str, float] = {}
        ran: list[str] = []
        skipped: dict[str, str] = {}

        if not fresh:
            return VisualContext(now, False, False, (), (), self.place, (),
                                 (), {"all": "stale_frame"}, {})

        spent = 0.0
        out = None
        persons_det: list = []
        objs_det: list = []
        pose_by_body: dict[int, object] = {}
        faces_out = None

        for stage in STAGE_ORDER:
            if stage == "faces":
                if self.c.faces is None:
                    continue

                faces_out = self._timed("faces", lambda: self.c.faces
                                        .process(image, now), timings, now)
                ran.append("faces")
                spent += timings["faces"]

                continue

            if self.c.faces is None and stage == "expression":
                continue

            if not self._due(stage, now):
                continue

            if not self._affordable(stage, spent):
                skipped[stage] = "over_frame_budget"

                continue

            if stage == "objects" and self.c.objects is not None:
                objs_det = self._timed(
                    "objects", lambda: list(self.c.objects.detect(image)),
                    timings, now)
                persons_det = [d for d in objs_det if d.label == "person"]
                objs_det = [d for d in objs_det if d.label != "person"]

                if persons_det:
                    self.bodies.update([d.box for d in persons_det], now,
                                       [d.score for d in persons_det])
                else:
                    self.bodies.coast(now)

                self._after_objects(image, objs_det, now, events)
                ran.append("objects")
                spent += timings["objects"]
            elif stage == "pose" and self.c.pose is not None:
                pose_by_body = self._timed(
                    "pose", lambda: self._run_pose(image, now), timings,
                    now)
                ran.append("pose")
                spent += timings["pose"]
            elif stage == "expression" and self.c.expression is not None:
                self._timed("expression", lambda: self._run_expression(
                    image, faces_out, now, events), timings, now)
                ran.append("expression")
                spent += timings["expression"]
            elif stage == "scene" and self.c.scene is not None and \
                    self.settings.object_memory:
                self.place = self._timed(
                    "scene", lambda: self.c.scene(image), timings, now)
                ran.append("scene")
                spent += timings["scene"]

        if "pose" in ran:
            self._after_pose(pose_by_body, now, events)

        events.extend(self._pending)
        self._pending.clear()
        people = self._people(faces_out, now, events)
        self._proactive(people, now, events)
        out = VisualContext(
            now, False, True, tuple(people),
            self._object_snapshot(), self.place, tuple(events), tuple(ran),
            skipped, timings)

        return out

    # ----- stage helpers -----

    def _object_snapshot(self) -> tuple[dict, ...]:
        if not self.c.stabilizer or not self.settings.objects:
            return ()

        return tuple({"id": o.object_id, "label": o.label,
                      "name": self._names.get(o.object_id),
                      "category": o.category, "score": round(o.score, 3),
                      "box": tuple(round(v, 1) for v in o.box)}
                     for o in self.c.stabilizer.objects())

    def _after_objects(self, image, dets, now, events) -> None:
        st = self.c.stabilizer

        if st is None:
            return

        h, w = image.shape[:2]
        ev = st.update(dets, now, float(h * w))

        for o in ev.appeared:
            name = None

            if self.c.namer is not None:
                try:
                    name = self.c.namer(image, o.box, o.label)
                except Exception:           # naming must never break vision
                    name = None

            self._names[o.object_id] = name
            events.append(ContextEvent("object_appeared", now, {
                "id": o.object_id, "label": o.label, "name": name}))

        for o in ev.disappeared:
            events.append(ContextEvent("object_disappeared", now, {
                "id": o.object_id, "label": o.label,
                "name": self._names.pop(o.object_id, None)}))

        mem = self.c.memory

        if mem is not None and self.settings.object_memory:
            for o in st.objects():
                name = self._names.get(o.object_id)

                if name:
                    mem.observe(o.object_id, "taught", o.label, name,
                                self.place, o.box, image.shape, now,
                                o.score)
                elif o.label in REMEMBER_CLASSES:
                    mem.observe(f"generic:{o.label}", "generic", o.label,
                                None, self.place, o.box, image.shape, now,
                                o.score)

    def _run_pose(self, image, now) -> dict:
        res = {}
        live = sorted(self.bodies.visible(),
                      key=lambda t: -((t.box[2] - t.box[0]) *
                                      (t.box[3] - t.box[1])))
        live = live[:self.cfg.max_pose_people]
        ids = {t.track_id for t in live}

        for t in live:
            res[t.track_id] = (self.c.pose.estimate(image, t.box), t.box)
            self._body_seen[t.track_id] = now

        for tid in list(self._body_seen):
            if tid in ids:
                continue

            if self.c.fall is not None:
                for e in self.c.fall.update(f"body_{tid}", None, now):
                    self._pending.append(ContextEvent(e.kind, now, {
                        "severity": e.severity, "reasons": e.reasons,
                        "body": f"body_{tid}", "details": e.details}))

            if now - self._body_seen[tid] > self.cfg.body_lost_s:
                self._drop_body(tid)

        return res

    def _drop_body(self, tid: int) -> None:
        key = f"body_{tid}"

        for comp in (self.c.activity, self.c.gestures):
            if comp is not None:
                comp.forget(key)

        if self.c.fall is not None:
            self.c.fall.forget(key)

        self._body_seen.pop(tid, None)
        self._body_act.pop(tid, None)
        self._body_gest.pop(tid, None)
        self._body_face.pop(tid, None)

    def _after_pose(self, by_body, now, events) -> None:
        s = self.settings
        furniture = []

        if self.c.stabilizer:
            furniture = [o.box for o in self.c.stabilizer.objects()
                         if o.label in FURNITURE]

        for tid, (pose, _box) in by_body.items():
            key = f"body_{tid}"

            if self.c.activity is not None:
                self._body_act[tid] = self.c.activity.update(key, pose, now)

            if self.c.gestures is not None and s.gestures:
                g = self.c.gestures.update(key, pose, now)
                self._body_gest[tid] = tuple(x.name for x in g)

                for x in g:
                    events.append(ContextEvent("gesture", now, {
                        "gesture": x.name, "body": key, "side": x.side,
                        "direction": x.direction}))

            if self.c.fall is not None and s.fall_detection:
                for e in self.c.fall.update(key, pose, now, furniture):
                    events.append(ContextEvent(e.kind, now, {
                        "severity": e.severity, "reasons": e.reasons,
                        "needs_voice_check": e.needs_voice_check,
                        "body": key, "details": e.details}))

    def _settings_for(self, person_id: str | None) -> dict:
        if not person_id or self.c.store is None:
            return {}

        p = self.c.store.get_person(person_id)

        return dict(p["settings"]) if p else {}

    def _run_expression(self, image, faces_out, now, events) -> None:
        if faces_out is None or self.c.expr_service is None:
            return

        live = set()

        for v in faces_out.tracks:
            if not v.visible or v.detection is None:
                continue

            st = self._settings_for(v.person_id)
            allowed = (self.settings.expression and v.person_id is not None
                       and (st.get("expression") or st.get("proactive")))
            live.add(v.track_id)

            if not allowed:
                self.c.expr_service.forget(v.track_id)
                self._expr.pop(v.track_id, None)

                continue

            scores = self.c.expression.predict(image, v.detection)
            est, ev = self.c.expr_service.update(
                v.track_id, scores, v.detection, v.quality, now)
            self._expr[v.track_id] = est
            self._expr_stamp[v.track_id] = now

            if ev is not None:
                events.append(ContextEvent("expression_changed", now, {
                    "person": v.person_id, "label": ev.label,
                    "confidence": ev.confidence}))

        self.c.expr_service.prune(live)

    def _people(self, faces_out, now, events) -> list[PersonContext]:
        people: list[PersonContext] = []
        present_keys: set[str] = set()

        if faces_out is not None:
            for v in faces_out.tracks:
                if not v.visible:
                    continue

                recognized = v.person_id is not None
                name = None
                st = {}

                if recognized and self.c.store is not None:
                    p = self.c.store.get_person(v.person_id)
                    name = p["display_name"] if p else None
                    st = dict(p["settings"]) if p else {}

                key = v.person_id or v.session_id
                present_keys.add(key)

                if key not in self._known_people:
                    events.append(ContextEvent("person_appeared", now, {
                        "key": key, "recognized": recognized}))

                self._known_people[key] = now

                if recognized:
                    self._touch(v.person_id, now)

                    if self.c.greeting is not None and \
                            self.settings.greeting and \
                            self.c.greeting.observe(v.person_id, st):
                        events.append(ContextEvent("greeting_due", now, {
                            "person": v.person_id, "name": name,
                            "preferred": st.get("preferred_greeting", "")}))

                est = self._expr.get(v.track_id)
                fused = self._fuse(key, est, v.track_id, now)
                body = self._body_for_face(v.box)
                act = self._body_act.get(body) if body is not None else None
                fall = None

                if body is not None and self.c.fall is not None:
                    fall = self.c.fall.states().get(f"body_{body}")

                people.append(PersonContext(
                    key, v.session_id, v.person_id, name,
                    v.identity_state.value, tuple(v.box),
                    fused.group if fused else None,
                    fused.confidence if fused else None,
                    fused.agreement.value if fused else None,
                    getattr(act, "posture", None),
                    getattr(act, "motion", None), fall,
                    self._body_gest.get(body, ()) if body is not None
                    else ()))

        for key in [k for k, t in self._known_people.items()
                    if k not in present_keys and now - t > 5.0]:
            self._known_people.pop(key)
            events.append(ContextEvent("person_left", now, {"key": key}))

        return people

    def _body_for_face(self, face_box) -> int | None:
        cx = (face_box[0] + face_box[2]) / 2
        cy = (face_box[1] + face_box[3]) / 2

        for t in self.bodies.visible():
            if t.track_id in self._body_seen and \
                    t.box[0] <= cx <= t.box[2] and t.box[1] <= cy <= t.box[3]:
                return t.track_id

        return None

    def _fuse(self, key, est, track_id, now):
        if est is None:
            return None

        affect = self.voice_affect(key) if self.voice_affect else None

        return fuse(est, self._expr_stamp.get(track_id, now), affect, now)

    def _touch(self, person_id: str, now: float) -> None:
        if self.c.store is None:
            return

        if now - self._last_mark.get(person_id, -1e9) >= \
                self.cfg.mark_seen_every_s:
            self._last_mark[person_id] = now

            try:
                self.c.store.mark_seen(person_id)
            except Exception:
                pass

    def _proactive(self, people, now, events) -> None:
        pm = self.c.proactive

        if pm is None:
            return

        emergency = False

        if self.c.fall is not None:
            emergency = any(s in ("suspect", "asked", "alerted")
                            for s in self.c.fall.states().values())

        cands = [p for p in people if p.person_id]
        cands.sort(key=lambda p: -((p.box[2] - p.box[0]) *
                                   (p.box[3] - p.box[1])) if p.box else 0)
        top = cands[0] if cands else None
        st = self._settings_for(top.person_id) if top else {}
        obs = None

        if top is not None and top.expression_group is not None:
            est = self._expr.get(self._track_of(top))
            obs = self._fuse(top.key, est, self._track_of(top), now)

        ctx = ProactiveContext(
            now=now, day=self._day,
            person_key=top.person_id if top else None,
            person_settings=st if self.settings.proactive else {},
            recognized=top is not None,
            person_present=bool(people),
            observation=obs,
            observation_stamp=self._expr_stamp.get(
                self._track_of(top), now) if top else now,
            camera_fresh=True, quiet_mode=self.quiet_mode,
            someone_speaking=self.voice_speaking,
            robot_busy=self.robot_busy,
            person_busy_hint=bool(top and top.motion == "moving"),
            invited=bool(top and self._invited == top.person_id),
            emergency=emergency)
        s = pm.tick(ctx)

        if s.kind == "offer":
            self._invited = None
            events.append(ContextEvent("proactive_offer", now, {
                "text": s.text, "person": s.person_key,
                "state": s.state.value, "reason": s.reason}))

    def _track_of(self, p: PersonContext) -> int:
        try:
            return int(p.session_id.rsplit("_", 1)[1]) if p.session_id \
                else -1
        except (ValueError, IndexError):
            return -1

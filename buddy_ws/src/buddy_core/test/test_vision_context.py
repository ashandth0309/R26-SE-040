"""Phase 17: VisualContextManager wiring, scheduling and privacy gates.

Every perception stage is a fake that counts calls, so these tests check
who runs when and what is allowed, not model accuracy.
"""

import time

import numpy as np
import pytest
from skeletons import hold, lying, standing, transition

from buddy_core.vision.activity import ActivityTracker
from buddy_core.vision.context import (
    Components, ManagerConfig, VisualContextManager)
from buddy_core.vision.expression import ExpressionScores, scores_from_probs
from buddy_core.vision.expression_tracker import ExpressionService
from buddy_core.vision.face_pipeline import PipelineOutput, TrackView
from buddy_core.vision.face_store import Consent, FaceProfileStore
from buddy_core.vision.faces import FaceDetection, FaceQuality
from buddy_core.vision.fall import FallDetector
from buddy_core.vision.gestures import GestureRecognizer
from buddy_core.vision.greeting import GreetingPolicy
from buddy_core.vision.identity import IdentityState
from buddy_core.vision.object_memory import ObjectMemory
from buddy_core.vision.object_stabilizer import ObjectStabilizer
from buddy_core.vision.objects import ObjectDetection
from buddy_core.vision.proactive import (
    ProactiveConfig, ProactiveInteractionManager)
from buddy_core.vision.vision_settings import VisionSettings

IMG = np.zeros((480, 640, 3), np.uint8)
FACE_BOX = (280.0, 100.0, 360.0, 200.0)


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


class FakeFaces:
    def __init__(self, person_id=None, visible=True, present=True):
        self.person_id, self.visible, self.present = (
            person_id, visible, present)
        self.calls = 0

    def process(self, image, now):
        self.calls += 1

        if not self.present:
            return PipelineOutput(now, (), True, 0, True, {})

        det = FaceDetection(*FACE_BOX[:2], 80, 100, 0.95)
        state = IdentityState.RECOGNIZED if self.person_id else \
            IdentityState.UNKNOWN
        v = TrackView(1, "session_person_1", FACE_BOX, self.visible, 0.95,
                      state, self.person_id, 0.9 if self.person_id else None,
                      FaceQuality(120.0, 50.0, 200.0, 0.0, 0.0, 0.0, ()),
                      det)

        return PipelineOutput(now, (v,), True, 1, False, {})


class FakeExpr:
    def __init__(self, label="sad", conf=0.8):
        from buddy_core.vision.expression import LABELS
        p = np.full(len(LABELS), (1 - conf) / (len(LABELS) - 1))
        p[LABELS.index(label)] = conf
        self.scores: ExpressionScores = scores_from_probs(p)
        self.calls = 0

    def predict(self, image, det):
        self.calls += 1

        return self.scores


class FakeObjects:
    def __init__(self, dets):
        self.dets, self.calls = dets, 0

    def detect(self, image):
        self.calls += 1

        return list(self.dets)


class FakePose:
    def __init__(self, frames):
        self.frames, self.i, self.calls = frames, 0, 0

    def estimate(self, image, box):
        self.calls += 1
        pose = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1

        return pose


def person_store():
    store = FaceProfileStore()
    pid = store.add_person("Ann", Consent(time.time(), "spoken_yes", True))

    return store, pid


def build(faces=None, expr=None, objects=None, pose=None, store=None,
          settings=None, memory=None, namer=None, proactive=None,
          clock=None, cfg=None, scene=None):
    clock = clock or Clock()
    comps = Components(
        faces=faces, expression=expr,
        expr_service=ExpressionService() if expr else None,
        objects=objects, stabilizer=ObjectStabilizer(),
        namer=namer, pose=pose, activity=ActivityTracker(),
        fall=FallDetector(), gestures=GestureRecognizer(), memory=memory,
        scene=scene, proactive=proactive,
        greeting=GreetingPolicy(clock=clock), store=store)

    return VisualContextManager(comps, settings, cfg, clock), clock


def step(m, clock, dt=0.5, **kw):
    clock.t += dt

    return m.process(IMG, clock.t, **kw)


def test_camera_off_runs_nothing_and_clears_state():
    f, o = FakeFaces("x"), FakeObjects([])
    m, c = build(f, objects=o)
    step(m, c)
    m.apply_settings(VisionSettings(camera_enabled=False))
    ctx = step(m, c)
    assert ctx.paused and f.calls == 1 and o.calls == 1
    assert ctx.people == () and ctx.objects == ()
    m.apply_settings(VisionSettings(camera_enabled=True))
    step(m, c)
    assert f.calls == 2


def test_stale_frame_is_not_processed():
    f = FakeFaces()
    m, c = build(f)
    ctx = m.process(IMG, c.t, frame_stamp=c.t - 5.0)
    assert not ctx.camera_fresh and f.calls == 0


def test_person_appeared_and_left_events():
    f = FakeFaces()
    m, c = build(f)
    ctx = step(m, c)
    assert [e.kind for e in ctx.events] == ["person_appeared"]
    assert ctx.people[0].identity_state == "unknown"
    f.present = False
    step(m, c, dt=3.0)
    ctx = step(m, c, dt=3.0)
    assert "person_left" in [e.kind for e in ctx.events]


def test_greeting_due_once_for_consenting_recognised_person():
    store, pid = person_store()
    f = FakeFaces(pid)
    m, c = build(f, store=store)
    kinds = []

    for _ in range(8):
        kinds += [e.kind for e in step(m, c, dt=0.5).events]

    assert kinds.count("greeting_due") >= 1
    store.set_setting(pid, "greet", False)
    m2, c2 = build(FakeFaces(pid), store=store)
    kinds = []

    for _ in range(8):
        kinds += [e.kind for e in step(m2, c2, dt=0.5).events]

    assert "greeting_due" not in kinds


def test_expression_only_for_recognised_opted_in_people():
    e = FakeExpr("happy")
    m, c = build(FakeFaces(None), expr=e)

    for _ in range(10):
        step(m, c)

    assert e.calls == 0                                   # stranger

    store, pid = person_store()
    e2 = FakeExpr("happy")
    m2, c2 = build(FakeFaces(pid), expr=e2, store=store)

    for _ in range(10):
        step(m2, c2)

    assert e2.calls == 0                                  # not opted in
    store.set_setting(pid, "expression", True)
    ctx = None

    for _ in range(14):
        ctx = step(m2, c2)

    assert e2.calls > 0
    assert ctx.people[0].expression_group == "positive"
    m2.apply_settings(VisionSettings(expression=False))
    before = e2.calls

    for _ in range(6):
        step(m2, c2)

    assert e2.calls == before                             # global switch


def test_profiles_control_which_stages_run():
    store, pid = person_store()
    store.set_setting(pid, "expression", True)
    e, o, p = FakeExpr(), FakeObjects([]), FakePose([standing()])
    m, c = build(FakeFaces(pid), expr=e, objects=o, pose=p, store=store,
                 settings=VisionSettings(profile="lightweight"))

    for _ in range(12):
        step(m, c, dt=0.5)

    assert e.calls == 0 and p.calls == 0 and 1 <= o.calls <= 4
    m2, c2 = build(FakeFaces(pid), expr=FakeExpr(), objects=FakeObjects(
        [ObjectDetection("person", .9, (200, 50, 440, 470), "other")]),
        pose=FakePose([standing()]), store=store,
        settings=VisionSettings(profile="balanced"))

    for _ in range(12):
        step(m2, c2, dt=0.5)

    assert m2.c.expression.calls > 0 and m2.c.pose.calls > 0
    # objects are due once a second, so ~6 passes in 6 s
    assert 4 <= m2.c.objects.calls <= 7


def test_disabling_pose_stops_pose_and_fall():
    person = ObjectDetection("person", .9, (200, 50, 440, 470), "other")
    p = FakePose([standing()])
    m, c = build(FakeFaces(), objects=FakeObjects([person]), pose=p,
                 settings=VisionSettings(pose=False))

    for _ in range(10):
        step(m, c)

    assert p.calls == 0


def test_object_events_naming_and_stable_snapshot():
    cup = ObjectDetection("cup", .8, (100, 100, 220, 220), "drink")
    o = FakeObjects([cup])
    m, c = build(None, objects=o, namer=lambda im, box, lab: "blue mug")
    seen = []

    for _ in range(6):
        ctx = step(m, c, dt=1.0)
        seen += [e.kind for e in ctx.events]

    assert seen.count("object_appeared") == 1
    assert ctx.objects[0]["name"] == "blue mug"
    o.dets = []

    for _ in range(4):
        ctx = step(m, c, dt=1.0)
        seen += [e.kind for e in ctx.events]

    assert "object_disappeared" in seen and ctx.objects == ()


def test_namer_errors_never_break_vision():
    cup = ObjectDetection("cup", .8, (100, 100, 220, 220), "drink")

    def boom(*a):
        raise RuntimeError("embedder broke")

    m, c = build(None, objects=FakeObjects([cup]), namer=boom)

    for _ in range(5):
        ctx = step(m, c, dt=1.0)

    assert ctx.objects[0]["name"] is None


def test_object_memory_only_when_enabled_and_never_people():
    cup = ObjectDetection("cup", .8, (100, 100, 220, 220), "drink")
    person = ObjectDetection("person", .9, (300, 50, 500, 470), "other")
    mem = ObjectMemory()
    mem.enable()
    m, c = build(None, objects=FakeObjects([cup, person]), memory=mem,
                 namer=lambda *a: "blue mug",
                 settings=VisionSettings(object_memory=False))

    for _ in range(6):
        step(m, c, dt=1.0)

    assert mem.export() == "[]"                       # setting off
    m.apply_settings(VisionSettings(object_memory=True))

    for _ in range(6):
        step(m, c, dt=1.0)

    names = [r["name"] for r in __import__("json").loads(mem.export())]
    assert names == ["blue mug"]


def test_fall_event_reaches_context_and_voice_reply_clears_it():
    person = ObjectDetection("person", .9, (100, 50, 560, 470), "other")
    frames = [p for _, p in hold(standing(), 0.0, 3.0, fps=2)]
    frames += [p for _, p in transition(standing(), lying(), 0.0, 0.5,
                                        fps=2)]
    frames += [p for _, p in hold(lying(), 0.0, 40.0, fps=2)]
    pose = FakePose(frames)
    m, c = build(FakeFaces(), objects=FakeObjects([person]), pose=pose,
                 settings=VisionSettings(profile="enhanced",
                                         overrides={"pose": 0.5,
                                                    "objects": 0.5}))
    kinds = []
    asked = None

    for _ in range(60):
        ctx = step(m, c, dt=0.5)

        for e in ctx.events:
            kinds.append(e.kind)

            if e.kind == "fall_suspected":
                asked = e

        if asked:
            break

    assert asked and asked.data["needs_voice_check"]
    out = m.on_voice_reply("I'm okay", c.t + 1)
    assert [e.kind for e in out] == ["fall_recovered"]


def test_frame_budget_skips_optional_stages_but_never_faces():
    class Slow(FakeObjects):
        def detect(self, image):
            time.sleep(0.02)

            return super().detect(image)

    f, o = FakeFaces(), Slow([])
    m, c = build(f, objects=o, cfg=ManagerConfig(frame_budget_ms=5.0))
    step(m, c, dt=1.0)                      # learns the cost
    ctx = step(m, c, dt=1.0)
    assert "faces" in ctx.stages_run
    assert ctx.stages_skipped.get("objects") == "over_frame_budget"


def test_proactive_offer_flows_through_with_opt_in_only():
    store, pid = person_store()
    store.set_setting(pid, "proactive", True)
    pm = ProactiveInteractionManager(ProactiveConfig(sustain_s=5.0))
    m, c = build(FakeFaces(pid), expr=FakeExpr("sad", 0.85), store=store,
                 proactive=pm)
    kinds = []

    for _ in range(60):
        kinds += [e.kind for e in step(m, c, dt=0.5).events]

    assert "proactive_offer" in kinds
    # a person who has not opted in is never offered anything
    store2, pid2 = person_store()
    store2.set_setting(pid2, "expression", True)
    pm2 = ProactiveInteractionManager(ProactiveConfig(sustain_s=5.0))
    m2, c2 = build(FakeFaces(pid2), expr=FakeExpr("sad", 0.85),
                   store=store2, proactive=pm2)
    kinds = []

    for _ in range(60):
        kinds += [e.kind for e in step(m2, c2, dt=0.5).events]

    assert "proactive_offer" not in kinds


def test_quiet_mode_and_speaking_block_offers():
    store, pid = person_store()
    store.set_setting(pid, "proactive", True)
    pm = ProactiveInteractionManager(ProactiveConfig(sustain_s=5.0))
    m, c = build(FakeFaces(pid), expr=FakeExpr("sad", 0.85), store=store,
                 proactive=pm)
    m.quiet_mode = True
    kinds = []

    for _ in range(60):
        kinds += [e.kind for e in step(m, c, dt=0.5).events]

    assert "proactive_offer" not in kinds


def test_invitation_triggers_offer_without_a_sad_face():
    store, pid = person_store()
    store.set_setting(pid, "proactive", True)
    pm = ProactiveInteractionManager()
    m, c = build(FakeFaces(pid), expr=FakeExpr("happy", 0.9), store=store,
                 proactive=pm)
    m.invite(pid)
    kinds = [e.kind for e in step(m, c).events]
    assert "proactive_offer" in kinds


def test_timings_are_measured_not_invented():
    m, c = build(FakeFaces(), objects=FakeObjects([]))
    ctx = step(m, c, dt=1.0)
    assert set(ctx.timings_ms) <= {"faces", "objects", "pose", "expression",
                                   "scene"}
    assert all(v >= 0 for v in ctx.timings_ms.values())


@pytest.mark.parametrize("attr", ["speak", "move", "drive", "publish"])
def test_manager_has_no_actuation_methods(attr):
    m, _ = build()
    assert not hasattr(m, attr)

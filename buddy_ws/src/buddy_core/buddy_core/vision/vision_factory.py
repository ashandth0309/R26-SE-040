"""Build the whole vision stack from the pinned models and a data folder.

A missing model never stops startup: the stage that needs it is left out
and the reason is recorded in ``unavailable`` so the node can publish it
and the doctor can show it.  Nothing here opens a camera.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from buddy_core.vision import models
from buddy_core.vision.activity import ActivityTracker
from buddy_core.vision.context import Components, VisualContextManager
from buddy_core.vision.expression_tracker import ExpressionService
from buddy_core.vision.face_pipeline import (
    FaceTrackingPipeline, PipelineConfig)
from buddy_core.vision.face_store import FaceProfileStore
from buddy_core.vision.fall import FallDetector
from buddy_core.vision.gestures import GestureRecognizer
from buddy_core.vision.greeting import GreetingPolicy
from buddy_core.vision.object_memory import ObjectMemory
from buddy_core.vision.object_stabilizer import ObjectStabilizer
from buddy_core.vision.object_teach import (
    ObjectRecognizer, ObjectStore, crop)
from buddy_core.vision.proactive import (
    ProactiveInteractionManager, StoreState)
from buddy_core.vision.vision_commands import VisionController
from buddy_core.vision.vision_settings import (
    PLANS, VisionSettings, load as load_settings)


def default_data_dir() -> Path:
    env = os.environ.get("BUDDY_VISION_DATA")

    return Path(env).expanduser() if env else \
        Path("~/.buddy/vision").expanduser()


@dataclass
class BuiltVision:
    manager: VisualContextManager
    controller: VisionController
    store: FaceProfileStore
    objects: ObjectStore
    memory: ObjectMemory
    settings: VisionSettings
    unavailable: dict = field(default_factory=dict)
    embedder: object | None = None
    object_embedder: object | None = None


def make_namer(embedder, recognizer: ObjectRecognizer) -> Callable:
    def namer(image, box, label):
        c = crop(image, box)

        if c.size == 0:
            return None

        m = recognizer.match(embedder.embed(c), label)

        return m.name if m.state == "match" else None

    return namer


def make_scene(embedder, memory: ObjectMemory) -> Callable:
    def scene(image):
        return memory.recognise_place(embedder.embed(image))[0]

    return scene


def build_vision(data_dir: str | os.PathLike | None = None,
                 settings: VisionSettings | None = None,
                 models_dir: str | os.PathLike | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 in_memory: bool = False, threads: int = 2
                 ) -> BuiltVision:
    d = Path(data_dir) if data_dir else default_data_dir()
    mdir = Path(models_dir) if models_dir else None
    unavailable: dict[str, str] = {}

    if not in_memory:
        d.mkdir(parents=True, exist_ok=True, mode=0o700)

    settings = settings or (VisionSettings() if in_memory
                            else load_settings(d / "settings.json"))
    p = (lambda n: ":memory:" if in_memory else str(d / n))
    store = FaceProfileStore(p("faces.db"))
    objstore = ObjectStore(p("objects.db"))
    memory = ObjectMemory(p("memory.db"))

    if settings.object_memory:
        memory.enable()

    def need(stage: str, key: str, make):
        try:
            return make(models.model_path(key, mdir))
        except models.ModelMissing as exc:
            unavailable[stage] = str(exc)
        except Exception as exc:                # a broken model file
            unavailable[stage] = f"{type(exc).__name__}: {exc}"

        return None

    from buddy_core.vision.embedding import SFaceEmbedder
    from buddy_core.vision.expression import ExpressionClassifier
    from buddy_core.vision.faces import YuNetDetector
    from buddy_core.vision.objects import NanoDetDetector
    from buddy_core.vision.pose import PoseEstimator

    detector = need("faces", "face_detector", lambda m: YuNetDetector(m))
    embedder = need("recognition", "face_embedder",
                    lambda m: SFaceEmbedder(m))
    pipeline = None

    if detector is not None:
        pcfg = PipelineConfig(recognition_enabled=embedder is not None
                              and settings.recognition)
        pcfg.detect_min_interval_s = PLANS[settings.profile]["faces"] or 0.1
        pipeline = FaceTrackingPipeline(
            detector, embedder, store.gallery, pcfg, store.negatives)

    expr = need("expression", "expression",
                lambda m: ExpressionClassifier(m, threads=threads))
    objects = need("objects", "object_detector",
                   lambda m: NanoDetDetector(m, threads=threads,
                                             exclude_person=False))
    pose = need("pose", "pose", lambda m: PoseEstimator(m, threads=threads))
    object_embedder = None
    namer = scene = None

    try:
        from buddy_core.vision.object_embedding import ObjectEmbedder
        object_embedder = ObjectEmbedder(threads=threads)
        namer = make_namer(object_embedder,
                           ObjectRecognizer(objstore))
        scene = make_scene(object_embedder, memory)
    except models.ModelMissing as exc:
        unavailable["object_naming"] = str(exc)
    except Exception as exc:
        unavailable["object_naming"] = f"{type(exc).__name__}: {exc}"

    greeting = GreetingPolicy(clock=clock)
    proactive = ProactiveInteractionManager(backend=StoreState(store))
    comps = Components(
        faces=pipeline, expression=expr,
        expr_service=ExpressionService() if expr else None,
        objects=objects, stabilizer=ObjectStabilizer(), namer=namer,
        pose=pose, activity=ActivityTracker(), fall=FallDetector(),
        gestures=GestureRecognizer(), memory=memory, scene=scene,
        proactive=proactive, greeting=greeting, store=store)
    manager = VisualContextManager(comps, settings, clock=clock)
    controller = VisionController(
        settings, manager.apply_settings, store, memory, proactive,
        greeting)

    return BuiltVision(manager, controller, store, objstore, memory,
                       settings, unavailable, embedder, object_embedder)

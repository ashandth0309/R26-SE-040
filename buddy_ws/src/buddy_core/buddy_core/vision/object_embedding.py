"""Image embeddings for taught objects (MobileNetV2 features, 1280-d).

The zoo ships MobileNetV2 as a classifier.  We cut the graph at global
average pooling to get a general-purpose feature vector.  Building that
cut model needs the ``onnx`` package once; running it needs only
onnxruntime.  These are ImageNet features, not an object-instance model:
they separate "my blue mug" from "a red bottle" well and "my mug" from
"your identical mug" badly.  Thresholds are uncalibrated starting values.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from buddy_core.vision.models import (
    ModelMissing, model_dir, model_path, sha256_of)
from buddy_core.vision.onnx_runtime import OnnxSession

FEATURE_FILE = "object_embedder_mobilenetv2_features.onnx"
SIDECAR = FEATURE_FILE + ".json"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DIM = 1280


def build_feature_model(src: Path, dst: Path) -> None:
    try:
        import onnx
        from onnx import utils
    except ImportError as exc:        # pragma: no cover
        raise ModelMissing(
            "building the object embedder needs the 'onnx' package "
            "(pip install onnx); it is only needed once") from exc

    m = onnx.load(str(src))
    pool = [n for n in m.graph.node if n.op_type == "GlobalAveragePool"]

    if len(pool) != 1:
        raise ModelMissing("unexpected MobileNetV2 graph layout")

    utils.extract_model(str(src), str(dst), [m.graph.input[0].name],
                        [pool[0].output[0]])
    (dst.parent / SIDECAR).write_text(json.dumps({
        "source_sha256": sha256_of(src), "cut_at": pool[0].name,
        "dim": DIM}))


def ensure_feature_model(directory: Path | None = None) -> Path:
    d = directory or model_dir()
    dst = d / FEATURE_FILE
    src = model_path("object_embedder_source", d)
    side = d / SIDECAR

    if dst.is_file() and side.is_file():
        try:
            if json.loads(side.read_text())["source_sha256"] == \
                    sha256_of(src):
                return dst
        except (ValueError, KeyError):
            pass

    build_feature_model(src, dst)

    return dst


class ObjectEmbedder:
    def __init__(self, path: str | None = None, threads: int = 2) -> None:
        self.session = OnnxSession(path or str(ensure_feature_model()),
                                   threads)

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        out = self.session.run(x.transpose(2, 0, 1)[None])[0]
        v = np.asarray(out, np.float32).reshape(-1)
        n = float(np.linalg.norm(v))

        return v / n if n > 0 else v


class FakeObjectEmbedder:
    """Test double: 8x8 colour layout, L2 normalised."""

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        small = cv2.resize(bgr, (8, 8), interpolation=cv2.INTER_AREA)
        v = small.astype(np.float32).reshape(-1) + 1.0
        v -= v.mean()
        n = float(np.linalg.norm(v))

        return v / n if n > 0 else v

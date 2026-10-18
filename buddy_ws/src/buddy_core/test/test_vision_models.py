"""Model manifest, verification and safe fetching."""

import hashlib
import io
import json

import pytest

from buddy_core.vision import models


def make_manifest(tmp_path, payloads):
    entries = {}

    for key, data in payloads.items():
        entries[key] = {"file": f"{key}.onnx", "url": f"mem://{key}",
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "licence": "MIT"}

    return {"models": entries}


def opener_for(payloads, calls):
    def opener(url, timeout=0):
        calls.append(url)
        key = url.removeprefix("mem://")

        return io.BytesIO(payloads[key])

    return opener


def test_real_manifest_is_pinned_and_licensed():
    man = models.load_manifest()
    assert len(man["source_commit"]) == 40
    assert {"face_detector", "face_embedder", "expression",
            "object_detector", "pose"} <= set(man["models"])

    for key, entry in man["models"].items():
        assert len(entry["sha256"]) == 64, key
        assert entry["url"].startswith("https://"), key
        assert man["source_commit"] in entry["url"], key
        assert entry["licence"] in ("MIT", "Apache-2.0"), key
        assert entry["file"].endswith(".onnx")


def test_missing_and_unknown_models_raise(tmp_path):
    man = make_manifest(tmp_path, {"a": b"x"})
    with pytest.raises(models.ModelMissing, match="fetch"):
        models.model_path("a", tmp_path, man)
    with pytest.raises(models.ModelMissing, match="unknown"):
        models.model_path("zzz", tmp_path, man)
    assert not models.available("a", tmp_path)


def test_tampered_file_is_rejected(tmp_path):
    man = make_manifest(tmp_path, {"a": b"good"})
    (tmp_path / "a.onnx").write_bytes(b"evil")
    with pytest.raises(models.ModelMissing, match="hash"):
        models.model_path("a", tmp_path, man)
    assert models.model_path("a", tmp_path, man, verify=False)


def test_fetch_downloads_verifies_and_is_idempotent(tmp_path):
    data = {"a": b"alpha", "b": b"beta"}
    man, calls = make_manifest(tmp_path, data), []
    status = models.fetch(directory=tmp_path, manifest=man,
                          opener=opener_for(data, calls))
    assert status == {"a": "downloaded", "b": "downloaded"}
    assert models.model_path("a", tmp_path, man).read_bytes() == b"alpha"
    again = models.fetch(directory=tmp_path, manifest=man,
                         opener=opener_for(data, calls))
    assert again == {"a": "present", "b": "present"} and len(calls) == 2


def test_fetch_discards_corrupt_download(tmp_path):
    good = {"a": b"alpha"}
    man = make_manifest(tmp_path, good)
    status = models.fetch(directory=tmp_path, manifest=man,
                          opener=opener_for({"a": b"corrupt"}, []))
    assert "mismatch" in status["a"]
    assert not (tmp_path / "a.onnx").exists()
    assert not list(tmp_path.glob("*.part"))


def test_fetch_reports_network_failure_and_cleans_up(tmp_path):
    man = make_manifest(tmp_path, {"a": b"alpha"})

    def broken(url, timeout=0):
        raise OSError("offline")

    status = models.fetch(directory=tmp_path, manifest=man, opener=broken)
    assert "offline" in status["a"]
    assert not list(tmp_path.iterdir())


def test_model_dir_honours_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("BUDDY_VISION_MODELS", str(tmp_path))
    assert models.model_dir() == tmp_path


def test_manifest_json_is_valid_json():
    json.loads(models.MANIFEST_PATH.read_text())

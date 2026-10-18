"""Privacy choices survive a restart, and a bad file fails closed."""

import json
import os
import stat
from dataclasses import replace

import numpy as np
import pytest

from buddy_core.vision.vision_factory import build_vision
from buddy_core.vision.vision_node import VisionNodeCore

IMG = np.zeros((120, 160, 3), np.uint8)


def build(tmp_path, **kw):
    return build_vision(data_dir=tmp_path / "data",
                        models_dir=tmp_path / "no_models", threads=1, **kw)


def test_spoken_camera_off_is_saved_and_still_off_after_restart(tmp_path):
    b = build(tmp_path)
    reply = b.controller.handle("turn off the camera", None, 1.0)
    assert reply
    path = tmp_path / "data" / "settings.json"
    assert json.loads(path.read_text())["camera_enabled"] is False
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    b.store.close()
    again = build(tmp_path)
    assert again.manager.settings.camera_enabled is False
    core = VisionNodeCore(again, max_process_hz=0.0)
    out = core.on_frame(IMG)
    assert out.people == [] and core.status()["camera_enabled"] is False


def test_unreadable_settings_fail_closed_with_camera_off(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "settings.json").write_text("{not json")
    b = build(tmp_path)
    assert b.manager.settings.camera_enabled is False
    assert "camera left OFF" in b.unavailable["settings"]


def test_unknown_keys_in_settings_also_fail_closed(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "settings.json").write_text('{"camera": true}')
    assert build(tmp_path).manager.settings.camera_enabled is False


def test_in_memory_build_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    b = build_vision(in_memory=True, models_dir=tmp_path / "x", threads=1)
    b.controller.handle("turn off the camera", None, 1.0)
    assert list(tmp_path.iterdir()) == []


def test_node_core_camera_switch_and_profile_are_persisted(tmp_path):
    b = build(tmp_path)
    core = VisionNodeCore(b, max_process_hz=0.0)
    core.set_camera_enabled(False)
    core.set_profile("lightweight")
    saved = json.loads((tmp_path / "data" / "settings.json").read_text())
    assert saved["camera_enabled"] is False
    assert saved["profile"] == "lightweight"

    with pytest.raises(ValueError):
        core.set_profile("turbo")

    assert b.manager.settings.profile == "lightweight"


def test_object_memory_choice_survives_restart(tmp_path):
    b = build(tmp_path)
    b.apply_settings(replace(b.manager.settings, object_memory=True))
    b.store.close()
    again = build(tmp_path)
    assert again.manager.settings.object_memory is True
    assert again.memory.enabled

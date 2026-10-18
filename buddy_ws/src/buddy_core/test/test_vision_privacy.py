"""Phase 19: privacy rules checked against the code itself."""

from pathlib import Path

import buddy_core.vision as vision
from buddy_core.vision import privacy

PKG = Path(vision.__file__).parent


def test_no_code_writes_images_to_disk():
    assert privacy.scan_for_image_writes(PKG) == []


def test_only_known_modules_write_files():
    extra = privacy.files_that_open_for_writing(PKG) - \
        privacy.allowed_writers()
    assert extra == set(), f"new file writers need review: {sorted(extra)}"


def test_inventory_declares_no_photos_and_no_unlisted_claims():
    assert all(item["photos"] is False for item in privacy.DATA_INVENTORY)
    assert all({"data", "where", "created_when", "retention", "delete"}
               <= set(i) for i in privacy.DATA_INVENTORY)
    # honesty: at-rest encryption is not claimed anywhere
    assert not any(i["encrypted_at_rest"] for i in privacy.DATA_INVENTORY)


def test_vision_package_never_imports_motion_or_audio_output():
    banned = ("sounddevice", "geometry_msgs", "MovementCommand",
              "cmd_vel", "subprocess", "aplay")

    for p in PKG.glob("*.py"):
        if p.name in ("privacy.py", "camera_node.py"):
            continue

        text = p.read_text()

        for word in banned:
            assert word not in text, f"{p.name} mentions {word}"


def test_camera_is_opened_only_by_the_camera_modules():
    openers = {"camera_sources.py", "camera_service.py", "camera_node.py"}

    for p in PKG.glob("*.py"):
        if p.name in openers or p.name == "privacy.py":
            continue

        text = p.read_text()
        assert "VideoCapture" not in text, p.name
        assert "Picamera2" not in text or p.name == "camera_sources.py"

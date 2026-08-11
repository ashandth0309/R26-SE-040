from copy import deepcopy

import pytest
import yaml

from buddy_core.config_loader import (
    BuddyConfigError,
    deep_merge,
    load_buddy_config,
    load_yaml_file,
    validate_config,
)


VALID_CONFIG = {
    "robot": {
        "id": "buddy-test",
        "name": "BUDDY",
    },
    "runtime": {
        "environment": "test",
        "log_level": "INFO",
        "startup_mode": "IDLE",
    },
    "timing": {
        "heartbeat_seconds": 2.0,
        "command_default_timeout_seconds": 1.0,
    },
    "movement": {
        "max_linear_speed_mps": None,
        "max_angular_speed_radps": None,
        "acceleration_limit_mps2": None,
    },
    "safety": {
        "obstacle_stop_distance_m": None,
        "obstacle_caution_distance_m": None,
        "command_timeout_seconds": 1.0,
        "connection_loss_stop_seconds": 1.0,
    },
    "camera": {
        "enabled": False,
        "width": 640,
        "height": 480,
        "fps": 15,
    },
    "audio": {
        "enabled": False,
        "wake_word": "Buddy",
        "sample_rate_hz": 16000,
    },
    "vision": {
        "face_recognition_enabled": False,
        "emotion_recognition_enabled": False,
    },
    "network": {
        "local_api_enabled": False,
        "remote_access_enabled": False,
    },
    "storage": {
        "data_directory": "runtime_data",
        "media_directory": "runtime_media",
        "database_path": "runtime_data/buddy.db",
    },
    "features": {
        "voice": False,
        "vision": False,
        "movement": False,
        "navigation": False,
        "patrol": False,
        "security": False,
        "mobile_control": False,
    },
}


def test_load_valid_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(VALID_CONFIG),
        encoding="utf-8",
    )

    loaded = load_yaml_file(path)

    assert loaded["robot"]["id"] == "buddy-test"


def test_recursive_override_merge():
    base = {
        "camera": {
            "enabled": True,
            "width": 640,
            "height": 480,
        }
    }

    override = {
        "camera": {
            "enabled": False,
        }
    }

    merged = deep_merge(base, override)

    assert merged["camera"]["enabled"] is False
    assert merged["camera"]["width"] == 640
    assert merged["camera"]["height"] == 480


def test_deep_merge_does_not_mutate_base():
    base = {
        "camera": {
            "enabled": True,
            "width": 640,
        }
    }

    original = deepcopy(base)

    deep_merge(
        base,
        {
            "camera": {
                "enabled": False,
            }
        },
    )

    assert base == original


def test_invalid_yaml(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "robot: [invalid",
        encoding="utf-8",
    )

    with pytest.raises(BuddyConfigError):
        load_yaml_file(path)


def test_missing_required_section():
    config = deepcopy(VALID_CONFIG)
    del config["robot"]

    with pytest.raises(BuddyConfigError):
        validate_config(config)


def test_invalid_log_level():
    config = deepcopy(VALID_CONFIG)
    config["runtime"]["log_level"] = "BANANA"

    with pytest.raises(BuddyConfigError):
        validate_config(config)


def test_negative_timing_value():
    config = deepcopy(VALID_CONFIG)
    config["timing"]["heartbeat_seconds"] = -1.0

    with pytest.raises(BuddyConfigError):
        validate_config(config)


def test_camera_enabled_invalid_resolution():
    config = deepcopy(VALID_CONFIG)
    config["camera"]["enabled"] = True
    config["camera"]["width"] = 0

    with pytest.raises(BuddyConfigError):
        validate_config(config)


def test_base_and_override_files(tmp_path):
    base_path = tmp_path / "base.yaml"
    override_path = tmp_path / "override.yaml"

    base_path.write_text(
        yaml.safe_dump(VALID_CONFIG),
        encoding="utf-8",
    )

    override_path.write_text(
        yaml.safe_dump(
            {
                "runtime": {
                    "environment": "development",
                },
                "camera": {
                    "enabled": False,
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_buddy_config(
        base_path=base_path,
        override_path=override_path,
    )

    assert config["runtime"]["environment"] == "development"
    assert config["camera"]["enabled"] is False
    assert config["camera"]["width"] == 640


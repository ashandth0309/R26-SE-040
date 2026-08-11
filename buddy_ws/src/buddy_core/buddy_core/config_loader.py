from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import yaml


class BuddyConfigError(RuntimeError):
    """Raised when BUDDY configuration is missing or invalid."""


REQUIRED_SECTIONS = {
    "robot",
    "runtime",
    "timing",
    "movement",
    "safety",
    "camera",
    "audio",
    "vision",
    "network",
    "storage",
    "features",
}

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

FEATURE_KEYS = {
    "voice",
    "vision",
    "movement",
    "navigation",
    "patrol",
    "security",
    "mobile_control",
}


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()

    if not config_path.is_file():
        raise BuddyConfigError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise BuddyConfigError(
            f"Invalid YAML in configuration file {config_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise BuddyConfigError(
            f"Unable to read configuration file {config_path}: {exc}"
        ) from exc

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise BuddyConfigError(
            f"Top-level YAML content must be a mapping: {config_path}"
        )

    return data


def deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    result = deepcopy(base)

    for key, override_value in override.items():
        base_value = result.get(key)

        if isinstance(base_value, dict) and isinstance(override_value, dict):
            result[key] = deep_merge(base_value, override_value)
        else:
            result[key] = deepcopy(override_value)

    return result


def _require_mapping(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section)

    if not isinstance(value, dict):
        raise BuddyConfigError(
            f"Configuration section '{section}' must be a mapping."
        )

    return value


def _require_non_empty_string(
    mapping: dict[str, Any],
    key: str,
    field_name: str,
) -> None:
    value = mapping.get(key)

    if not isinstance(value, str) or not value.strip():
        raise BuddyConfigError(
            f"'{field_name}' must be a non-empty string."
        )


def _require_positive_number(value: Any, field_name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        raise BuddyConfigError(
            f"'{field_name}' must be a positive number."
        )


def _require_non_negative_or_none(value: Any, field_name: str) -> None:
    if value is None:
        return

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 0
    ):
        raise BuddyConfigError(
            f"'{field_name}' must be non-negative or null."
        )


def validate_config(config: dict[str, Any]) -> None:
    missing_sections = REQUIRED_SECTIONS - config.keys()

    if missing_sections:
        raise BuddyConfigError(
            "Missing required configuration section(s): "
            + ", ".join(sorted(missing_sections))
        )

    robot = _require_mapping(config, "robot")
    runtime = _require_mapping(config, "runtime")
    timing = _require_mapping(config, "timing")
    movement = _require_mapping(config, "movement")
    safety = _require_mapping(config, "safety")
    camera = _require_mapping(config, "camera")
    audio = _require_mapping(config, "audio")
    _require_mapping(config, "vision")
    _require_mapping(config, "network")
    storage = _require_mapping(config, "storage")
    features = _require_mapping(config, "features")

    _require_non_empty_string(robot, "id", "robot.id")
    _require_non_empty_string(robot, "name", "robot.name")

    log_level = runtime.get("log_level")

    if not isinstance(log_level, str) or log_level.upper() not in VALID_LOG_LEVELS:
        raise BuddyConfigError(
            "runtime.log_level must be one of: "
            + ", ".join(sorted(VALID_LOG_LEVELS))
        )

    _require_positive_number(
        timing.get("heartbeat_seconds"),
        "timing.heartbeat_seconds",
    )

    _require_positive_number(
        timing.get("command_default_timeout_seconds"),
        "timing.command_default_timeout_seconds",
    )

    for key in (
        "max_linear_speed_mps",
        "max_angular_speed_radps",
        "acceleration_limit_mps2",
    ):
        _require_non_negative_or_none(
            movement.get(key),
            f"movement.{key}",
        )

    for key in (
        "obstacle_stop_distance_m",
        "obstacle_caution_distance_m",
    ):
        _require_non_negative_or_none(
            safety.get(key),
            f"safety.{key}",
        )

    _require_positive_number(
        safety.get("command_timeout_seconds"),
        "safety.command_timeout_seconds",
    )

    _require_positive_number(
        safety.get("connection_loss_stop_seconds"),
        "safety.connection_loss_stop_seconds",
    )

    camera_enabled = camera.get("enabled")

    if not isinstance(camera_enabled, bool):
        raise BuddyConfigError("camera.enabled must be boolean.")

    if camera_enabled:
        width = camera.get("width")
        height = camera.get("height")
        fps = camera.get("fps")

        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise BuddyConfigError(
                "camera.width must be a positive integer when camera is enabled."
            )

        if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            raise BuddyConfigError(
                "camera.height must be a positive integer when camera is enabled."
            )

        _require_positive_number(
            fps,
            "camera.fps",
        )

    audio_enabled = audio.get("enabled")

    if not isinstance(audio_enabled, bool):
        raise BuddyConfigError("audio.enabled must be boolean.")

    if audio_enabled:
        sample_rate = audio.get("sample_rate_hz")

        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate <= 0
        ):
            raise BuddyConfigError(
                "audio.sample_rate_hz must be a positive integer "
                "when audio is enabled."
            )

    for key in FEATURE_KEYS:
        if key not in features:
            raise BuddyConfigError(
                f"Missing required feature flag: features.{key}"
            )

        if not isinstance(features[key], bool):
            raise BuddyConfigError(
                f"features.{key} must be boolean."
            )

    for key in (
        "data_directory",
        "media_directory",
        "database_path",
    ):
        value = storage.get(key)

        if not isinstance(value, str) or not value.strip():
            raise BuddyConfigError(
                f"storage.{key} must be a non-empty path string."
            )


def find_project_root() -> Path:
    env_root = os.getenv("BUDDY_PROJECT_ROOT")

    if env_root:
        root = Path(env_root).expanduser().resolve()

        if (root / "config" / "buddy.yaml").is_file():
            return root

        raise BuddyConfigError(
            f"BUDDY_PROJECT_ROOT does not contain config/buddy.yaml: {root}"
        )

    current_file = Path(__file__).resolve()

    for parent in current_file.parents:
        if (parent / "config" / "buddy.yaml").is_file():
            return parent

    raise BuddyConfigError(
        "Unable to locate BUDDY project root. "
        "Set BUDDY_PROJECT_ROOT or BUDDY_CONFIG_PATH."
    )


def load_buddy_config(
    base_path: str | Path | None = None,
    override_path: str | Path | None = None,
) -> dict[str, Any]:
    config_path_env = os.getenv("BUDDY_CONFIG_PATH")

    if base_path is None and config_path_env:
        base_path = config_path_env

    if base_path is None:
        project_root = find_project_root()
        base_path = project_root / "config" / "buddy.yaml"

    base_config = load_yaml_file(base_path)

    if override_path is None:
        override_env = os.getenv("BUDDY_CONFIG_OVERRIDE")

        if override_env:
            override_path = override_env

    if override_path is not None:
        override_config = load_yaml_file(override_path)
        config = deep_merge(base_config, override_config)
    else:
        config = base_config

    validate_config(config)

    return config

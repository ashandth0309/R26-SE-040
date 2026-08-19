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
    "hardware",
    "camera",
    "audio",
    "vision",
    "network",
    "storage",
    "features",
    "logging",
}

VALID_LOG_LEVELS = {
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
}

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
    """Recursively merge override into a copy of base."""

    result = deepcopy(base)

    for key, override_value in override.items():
        base_value = result.get(key)

        if isinstance(base_value, dict) and isinstance(override_value, dict):
            result[key] = deep_merge(
                base_value,
                override_value,
            )
        else:
            result[key] = deepcopy(override_value)

    return result


def _require_mapping(
    config: dict[str, Any],
    section: str,
) -> dict[str, Any]:
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


def _require_positive_number(
    value: Any,
    field_name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        raise BuddyConfigError(
            f"'{field_name}' must be a positive number."
        )


def _require_non_negative_or_none(
    value: Any,
    field_name: str,
) -> None:
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


def _require_boolean(
    mapping: dict[str, Any],
    key: str,
    field_name: str,
) -> None:
    value = mapping.get(key)

    if not isinstance(value, bool):
        raise BuddyConfigError(
            f"'{field_name}' must be boolean."
        )


def _require_positive_integer(
    value: Any,
    field_name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise BuddyConfigError(
            f"'{field_name}' must be a positive integer."
        )


def _require_non_negative_integer(
    value: Any,
    field_name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise BuddyConfigError(
            f"'{field_name}' must be a non-negative integer."
        )


def validate_config(config: dict[str, Any]) -> None:
    """Validate merged BUDDY configuration."""

    if not isinstance(config, dict):
        raise BuddyConfigError(
            "Configuration must be a mapping."
        )

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
    hardware = _require_mapping(config, "hardware")
    camera = _require_mapping(config, "camera")
    audio = _require_mapping(config, "audio")
    _require_mapping(config, "vision")
    _require_mapping(config, "network")
    storage = _require_mapping(config, "storage")
    features = _require_mapping(config, "features")
    logging_config = _require_mapping(config, "logging")

    # Robot
    _require_non_empty_string(
        robot,
        "id",
        "robot.id",
    )

    _require_non_empty_string(
        robot,
        "name",
        "robot.name",
    )

    # Runtime
    environment = runtime.get("environment")

    if not isinstance(environment, str) or not environment.strip():
        raise BuddyConfigError(
            "runtime.environment must be a non-empty string."
        )

    log_level = runtime.get("log_level")

    if (
        not isinstance(log_level, str)
        or log_level.upper() not in VALID_LOG_LEVELS
    ):
        raise BuddyConfigError(
            "runtime.log_level must be one of: "
            + ", ".join(sorted(VALID_LOG_LEVELS))
        )

    startup_mode = runtime.get("startup_mode")

    if not isinstance(startup_mode, str) or not startup_mode.strip():
        raise BuddyConfigError(
            "runtime.startup_mode must be a non-empty string."
        )

    # Timing
    _require_positive_number(
        timing.get("heartbeat_seconds"),
        "timing.heartbeat_seconds",
    )

    _require_positive_number(
        timing.get("command_default_timeout_seconds"),
        "timing.command_default_timeout_seconds",
    )

    # Movement
    for key in (
        "max_linear_speed_mps",
        "max_angular_speed_radps",
        "acceleration_limit_mps2",
    ):
        _require_non_negative_or_none(
            movement.get(key),
            f"movement.{key}",
        )

    # Safety
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

        # Hardware
    _require_boolean(
        hardware,
        "enabled",
        "hardware.enabled",
    )

    _require_boolean(
        hardware,
        "simulation",
        "hardware.simulation",
    )

    gpio_config = hardware.get("gpio")

    if not isinstance(gpio_config, dict):
        raise BuddyConfigError(
            "hardware.gpio must be a mapping."
        )

    numbering_mode = gpio_config.get(
        "numbering_mode"
    )

    if numbering_mode not in {
        "BCM",
        "BOARD",
    }:
        raise BuddyConfigError(
            "hardware.gpio.numbering_mode must "
            "be 'BCM' or 'BOARD'."
        )

    # Camera
    _require_boolean(
        camera,
        "enabled",
        "camera.enabled",
    )

    if camera["enabled"]:
        _require_positive_integer(
            camera.get("width"),
            "camera.width",
        )

        _require_positive_integer(
            camera.get("height"),
            "camera.height",
        )

        _require_positive_number(
            camera.get("fps"),
            "camera.fps",
        )

    # Audio
    _require_boolean(
        audio,
        "enabled",
        "audio.enabled",
    )

    if audio["enabled"]:
        _require_non_empty_string(
            audio,
            "wake_word",
            "audio.wake_word",
        )

        _require_positive_integer(
            audio.get("sample_rate_hz"),
            "audio.sample_rate_hz",
        )

    # Feature flags
    for key in FEATURE_KEYS:
        if key not in features:
            raise BuddyConfigError(
                f"Missing required feature flag: features.{key}"
            )

        if not isinstance(features[key], bool):
            raise BuddyConfigError(
                f"features.{key} must be boolean."
            )

    # Storage paths
    for key in (
        "data_directory",
        "media_directory",
        "database_path",
        "log_directory",
    ):
        value = storage.get(key)

        if not isinstance(value, str) or not value.strip():
            raise BuddyConfigError(
                f"storage.{key} must be a non-empty path string."
            )

    # Logging
    for key in (
        "enabled",
        "console_enabled",
        "file_enabled",
        "include_context",
    ):
        _require_boolean(
            logging_config,
            key,
            f"logging.{key}",
        )

    if logging_config["file_enabled"]:
        _require_non_empty_string(
            logging_config,
            "file_name",
            "logging.file_name",
        )

        _require_positive_integer(
            logging_config.get("max_bytes"),
            "logging.max_bytes",
        )

    _require_non_negative_integer(
        logging_config.get("backup_count"),
        "logging.backup_count",
    )


def find_project_root() -> Path:
    """
    Locate the BUDDY project root without relying on the shell working
    directory or a machine-specific absolute path.
    """

    env_root = os.getenv("BUDDY_PROJECT_ROOT")

    if env_root:
        root = Path(env_root).expanduser().resolve()

        if (root / "config" / "buddy.yaml").is_file():
            return root

        raise BuddyConfigError(
            "BUDDY_PROJECT_ROOT does not contain "
            f"config/buddy.yaml: {root}"
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
    """
    Load, optionally merge, and validate BUDDY configuration.
    """

    config_path_env = os.getenv("BUDDY_CONFIG_PATH")

    if base_path is None and config_path_env:
        base_path = config_path_env

    if base_path is None:
        project_root = find_project_root()
        base_path = (
            project_root
            / "config"
            / "buddy.yaml"
        )

    base_config = load_yaml_file(base_path)

    if override_path is None:
        override_env = os.getenv(
            "BUDDY_CONFIG_OVERRIDE"
        )

        if override_env:
            override_path = override_env

    if override_path is not None:
        override_config = load_yaml_file(
            override_path
        )

        config = deep_merge(
            base_config,
            override_config,
        )
    else:
        config = base_config

    validate_config(config)

    return config

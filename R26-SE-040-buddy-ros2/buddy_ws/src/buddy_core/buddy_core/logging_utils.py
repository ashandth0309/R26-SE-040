"""Centralized logging utilities for BUDDY."""

from __future__ import annotations

from copy import deepcopy
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Mapping


class BuddyLoggingError(RuntimeError):
    """Raised when BUDDY logging cannot be configured."""


_LOGGER_ROOT = "buddy"

_SENSITIVE_KEYS = {
    "token",
    "api_key",
    "api_secret",
    "password",
    "authorization",
    "secret",
    "bearer",
}

_VALID_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_configured = False
_log_file_path: Path | None = None


class _UTCFormatter(logging.Formatter):
    """Logging formatter that renders timestamps in UTC."""

    converter = __import__("time").gmtime


def _get_log_level(level_name: str) -> int:
    """Convert a configured level name into a Python logging level."""

    if not isinstance(level_name, str):
        raise BuddyLoggingError(
            "Logging level must be a string."
        )

    normalized = level_name.upper()

    if normalized not in _VALID_LEVELS:
        raise BuddyLoggingError(
            "Invalid logging level. Expected one of: "
            + ", ".join(_VALID_LEVELS)
        )

    return _VALID_LEVELS[normalized]


def _validate_logging_config(config: Mapping[str, Any]) -> None:
    """Perform defensive validation of logging configuration."""

    logging_config = config.get("logging")
    storage = config.get("storage")
    runtime = config.get("runtime")

    if not isinstance(logging_config, Mapping):
        raise BuddyLoggingError(
            "Missing or invalid logging configuration."
        )

    if not isinstance(storage, Mapping):
        raise BuddyLoggingError(
            "Missing or invalid storage configuration."
        )

    if not isinstance(runtime, Mapping):
        raise BuddyLoggingError(
            "Missing or invalid runtime configuration."
        )

    for key in (
        "enabled",
        "console_enabled",
        "file_enabled",
        "include_context",
    ):
        if not isinstance(logging_config.get(key), bool):
            raise BuddyLoggingError(
                f"logging.{key} must be boolean."
            )

    _get_log_level(runtime.get("log_level"))

    if logging_config["file_enabled"]:
        file_name = logging_config.get("file_name")

        if not isinstance(file_name, str) or not file_name.strip():
            raise BuddyLoggingError(
                "logging.file_name must be a non-empty string."
            )

        max_bytes = logging_config.get("max_bytes")

        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise BuddyLoggingError(
                "logging.max_bytes must be a positive integer."
            )

        log_directory = storage.get("log_directory")

        if (
            not isinstance(log_directory, str)
            or not log_directory.strip()
        ):
            raise BuddyLoggingError(
                "storage.log_directory must be a non-empty string."
            )

    backup_count = logging_config.get("backup_count")

    if (
        isinstance(backup_count, bool)
        or not isinstance(backup_count, int)
        or backup_count < 0
    ):
        raise BuddyLoggingError(
            "logging.backup_count must be a non-negative integer."
        )


def _remove_buddy_handlers(logger: logging.Logger) -> None:
    """Remove and close handlers managed by the BUDDY logger."""

    for handler in list(logger.handlers):
        if getattr(handler, "_buddy_handler", False):
            logger.removeHandler(handler)

            try:
                handler.close()
            except Exception:
                pass


def configure_logging(
    config: Mapping[str, Any],
    project_root: str | Path | None = None,
) -> Path | None:
    """
    Configure the central BUDDY logger hierarchy.

    Returns the active log-file path when file logging is enabled,
    otherwise returns None.
    """

    global _configured
    global _log_file_path

    _validate_logging_config(config)

    logging_config = config["logging"]
    runtime = config["runtime"]
    storage = config["storage"]

    logger = logging.getLogger(_LOGGER_ROOT)

    # Reconfiguration must never accumulate duplicate handlers.
    _remove_buddy_handlers(logger)

    level = _get_log_level(runtime["log_level"])

    logger.setLevel(level)
    logger.propagate = False

    formatter = _UTCFormatter(
        fmt="%(asctime)sZ | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if not logging_config["enabled"]:
        logger.disabled = True
        _configured = True
        _log_file_path = None
        return None

    logger.disabled = False

    if logging_config["console_enabled"]:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler._buddy_handler = True
        logger.addHandler(console_handler)

    if logging_config["file_enabled"]:
        root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else Path.cwd().resolve()
        )

        configured_directory = Path(storage["log_directory"]).expanduser()

        if configured_directory.is_absolute():
            log_directory = configured_directory
        else:
            log_directory = root / configured_directory

        try:
            log_directory.mkdir(
                parents=True,
                exist_ok=True,
            )
        except OSError as exc:
            raise BuddyLoggingError(
                f"Unable to create log directory "
                f"{log_directory}: {exc}"
            ) from exc

        log_file = log_directory / logging_config["file_name"]

        try:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=logging_config["max_bytes"],
                backupCount=logging_config["backup_count"],
                encoding="utf-8",
            )
        except OSError as exc:
            raise BuddyLoggingError(
                f"Unable to create BUDDY log file "
                f"{log_file}: {exc}"
            ) from exc

        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler._buddy_handler = True
        logger.addHandler(file_handler)

        _log_file_path = log_file.resolve()
    else:
        _log_file_path = None

    _configured = True

    return _log_file_path


def get_buddy_logger(
    name: str,
    context: Mapping[str, Any] | None = None,
) -> logging.Logger | logging.LoggerAdapter:
    """Return a logger inside the BUDDY logger hierarchy."""

    if not isinstance(name, str) or not name.strip():
        raise BuddyLoggingError(
            "Logger name must be a non-empty string."
        )

    normalized_name = name.strip()

    if normalized_name == _LOGGER_ROOT:
        logger_name = _LOGGER_ROOT
    elif normalized_name.startswith(f"{_LOGGER_ROOT}."):
        logger_name = normalized_name
    else:
        logger_name = f"{_LOGGER_ROOT}.{normalized_name}"

    logger = logging.getLogger(logger_name)

    if context is None:
        return logger

    if not isinstance(context, Mapping):
        raise BuddyLoggingError(
            "Logger context must be a mapping."
        )

    safe_context = redact_mapping(context)

    context_text = " ".join(
        f"{key}={value}"
        for key, value in safe_context.items()
    )

    return logging.LoggerAdapter(
        logger,
        {
            "buddy_context": context_text,
        },
    )


def redact_mapping(
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Return a redacted deep copy of a mapping.

    Known sensitive key names are replaced with '[REDACTED]'.
    """

    if not isinstance(mapping, Mapping):
        raise BuddyLoggingError(
            "Value to redact must be a mapping."
        )

    copied = deepcopy(dict(mapping))

    def redact_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            result = {}

            for key, nested_value in value.items():
                normalized_key = str(key).lower()

                if any(
                    sensitive in normalized_key
                    for sensitive in _SENSITIVE_KEYS
                ):
                    result[key] = "[REDACTED]"
                else:
                    result[key] = redact_value(nested_value)

            return result

        if isinstance(value, list):
            return [
                redact_value(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return tuple(
                redact_value(item)
                for item in value
            )

        return value

    return redact_value(copied)


def log_exception(
    logger: logging.Logger | logging.LoggerAdapter,
    message: str,
    exc: BaseException,
) -> None:
    """Log an exception with traceback information."""

    if not isinstance(message, str) or not message.strip():
        raise BuddyLoggingError(
            "Exception log message must be a non-empty string."
        )

    logger.error(
        "%s: %s",
        message,
        type(exc).__name__,
        exc_info=(
            type(exc),
            exc,
            exc.__traceback__,
        ),
    )


def get_log_file_path() -> Path | None:
    """Return the currently configured BUDDY log-file path."""

    return _log_file_path


def is_logging_configured() -> bool:
    """Return whether configure_logging has been called."""

    return _configured

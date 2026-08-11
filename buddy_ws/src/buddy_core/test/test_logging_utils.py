"""Tests for BUDDY centralized logging utilities."""

from copy import deepcopy
import logging
from logging.handlers import RotatingFileHandler

import pytest

from buddy_core.logging_utils import (
    BuddyLoggingError,
    configure_logging,
    get_buddy_logger,
    redact_mapping,
)


def make_config():
    """Create a valid logging configuration for tests."""

    return {
        "runtime": {
            "log_level": "DEBUG",
        },
        "storage": {
            "log_directory": "logs",
        },
        "logging": {
            "enabled": True,
            "console_enabled": True,
            "file_enabled": True,
            "file_name": "buddy.log",
            "max_bytes": 1024,
            "backup_count": 2,
            "include_context": True,
        },
    }


def test_log_level_configuration(tmp_path):
    config = make_config()

    configure_logging(
        config,
        project_root=tmp_path,
    )

    logger = logging.getLogger("buddy")

    assert logger.level == logging.DEBUG


def test_console_handler_created(tmp_path):
    config = make_config()

    configure_logging(
        config,
        project_root=tmp_path,
    )

    logger = logging.getLogger("buddy")

    stream_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, RotatingFileHandler)
    ]

    assert len(stream_handlers) == 1


def test_file_handler_created(tmp_path):
    config = make_config()

    configure_logging(
        config,
        project_root=tmp_path,
    )

    logger = logging.getLogger("buddy")

    file_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, RotatingFileHandler)
    ]

    assert len(file_handlers) == 1


def test_log_directory_created(tmp_path):
    config = make_config()

    configure_logging(
        config,
        project_root=tmp_path,
    )

    assert (tmp_path / "logs").is_dir()


def test_log_file_created(tmp_path):
    config = make_config()

    configure_logging(
        config,
        project_root=tmp_path,
    )

    logger = logging.getLogger("buddy")
    logger.info("BUDDY test log message.")

    for handler in logger.handlers:
        handler.flush()

    assert (
        tmp_path
        / "logs"
        / "buddy.log"
    ).is_file()


def test_duplicate_handler_protection(tmp_path):
    config = make_config()

    configure_logging(
        config,
        project_root=tmp_path,
    )

    configure_logging(
        config,
        project_root=tmp_path,
    )

    logger = logging.getLogger("buddy")

    buddy_handlers = [
        handler
        for handler in logger.handlers
        if getattr(
            handler,
            "_buddy_handler",
            False,
        )
    ]

    assert len(buddy_handlers) == 2


def test_redact_token():
    result = redact_mapping(
        {
            "username": "demo",
            "token": "fake-token",
        }
    )

    assert result["username"] == "demo"
    assert result["token"] == "[REDACTED]"


def test_redact_password():
    result = redact_mapping(
        {
            "password": "fake-password",
        }
    )

    assert result["password"] == "[REDACTED]"


def test_nested_redaction():
    result = redact_mapping(
        {
            "username": "demo",
            "nested": {
                "api_key": "fake-key",
                "normal_value": "visible",
            },
        }
    )

    assert result["username"] == "demo"
    assert (
        result["nested"]["api_key"]
        == "[REDACTED]"
    )
    assert (
        result["nested"]["normal_value"]
        == "visible"
    )


def test_redaction_does_not_mutate_input():
    source = {
        "username": "demo",
        "nested": {
            "secret": "fake-secret",
        },
    }

    original = deepcopy(source)

    result = redact_mapping(source)

    assert source == original
    assert (
        result["nested"]["secret"]
        == "[REDACTED]"
    )


def test_invalid_max_bytes(tmp_path):
    config = make_config()

    config["logging"]["max_bytes"] = 0

    with pytest.raises(BuddyLoggingError):
        configure_logging(
            config,
            project_root=tmp_path,
        )


def test_invalid_backup_count(tmp_path):
    config = make_config()

    config["logging"]["backup_count"] = -1

    with pytest.raises(BuddyLoggingError):
        configure_logging(
            config,
            project_root=tmp_path,
        )


def test_file_logging_disabled(tmp_path):
    config = make_config()

    config["logging"]["file_enabled"] = False

    log_path = configure_logging(
        config,
        project_root=tmp_path,
    )

    assert log_path is None

    assert not (
        tmp_path
        / "logs"
        / "buddy.log"
    ).exists()


def test_project_relative_log_path(tmp_path):
    config = make_config()

    log_path = configure_logging(
        config,
        project_root=tmp_path,
    )

    expected_path = (
        tmp_path
        / "logs"
        / "buddy.log"
    ).resolve()

    assert log_path == expected_path


def test_log_rotation(tmp_path):
    config = make_config()

    config["logging"]["max_bytes"] = 256
    config["logging"]["backup_count"] = 2

    configure_logging(
        config,
        project_root=tmp_path,
    )

    logger = logging.getLogger("buddy")

    file_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(
            handler,
            RotatingFileHandler,
        )
    ]

    assert len(file_handlers) == 1

    file_handler = file_handlers[0]

    assert file_handler.maxBytes == 256
    assert file_handler.backupCount == 2

    for index in range(200):
        logger.info(
            "rotation-test-message-%03d-%s",
            index,
            "x" * 200,
        )

    file_handler.flush()

    log_directory = tmp_path / "logs"

    assert (
        log_directory
        / "buddy.log"
    ).is_file()

    rotated_files = list(
        log_directory.glob(
            "buddy.log.*"
        )
    )

    assert rotated_files, (
        "Expected RotatingFileHandler to "
        "create at least one rotated log backup."
    )


def test_context_logger(tmp_path):
    config = make_config()

    configure_logging(
        config,
        project_root=tmp_path,
    )

    logger = get_buddy_logger(
        "context",
        context={
            "robot_id": "buddy-test",
            "event_id": "event-1",
        },
    )

    assert isinstance(
        logger,
        logging.LoggerAdapter,
    )

"""Smoke test for the BUDDY centralized logging architecture."""

from __future__ import annotations

import sys

from buddy_core.config_loader import (
    BuddyConfigError,
    find_project_root,
    load_buddy_config,
)
from buddy_core.logging_utils import (
    BuddyLoggingError,
    configure_logging,
    get_buddy_logger,
    get_log_file_path,
)


def main() -> int:
    """Run a safe smoke test of BUDDY centralized logging."""

    try:
        project_root = find_project_root()

        base_config = project_root / "config" / "buddy.yaml"
        development_config = (
            project_root / "config" / "development.yaml"
        )

        config = load_buddy_config(
            base_path=base_config,
            override_path=development_config,
        )

        configured_log_path = configure_logging(
            config,
            project_root=project_root,
        )

        robot_id = config["robot"]["id"]

        logger = get_buddy_logger(
            "core.logging_smoke_test",
            context={
                "robot_id": robot_id,
                "module": "logging_smoke_test",
            },
        )

        logger.debug("BUDDY logging DEBUG smoke-test message.")
        logger.info("BUDDY logging configured successfully.")
        logger.warning("BUDDY logging WARNING smoke-test message.")

        log_level = config["runtime"]["log_level"]
        file_enabled = config["logging"]["file_enabled"]

        log_file_path = get_log_file_path()

        if file_enabled:
            if configured_log_path is None:
                raise BuddyLoggingError(
                    "File logging is enabled but no log path was returned."
                )

            if log_file_path is None:
                raise BuddyLoggingError(
                    "File logging is enabled but no active log path exists."
                )

            if not log_file_path.is_file():
                raise BuddyLoggingError(
                    f"Expected log file was not created: {log_file_path}"
                )

        print("BUDDY logging smoke test PASS")
        print(f"Log level: {log_level}")
        print(f"File logging: {file_enabled}")
        print(
            "Log file exists: "
            f"{bool(log_file_path and log_file_path.is_file())}"
        )

        return 0

    except (BuddyConfigError, BuddyLoggingError, OSError) as exc:
        print(
            f"BUDDY logging smoke test FAIL: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

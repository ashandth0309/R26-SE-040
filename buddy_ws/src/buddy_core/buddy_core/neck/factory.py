"""Neck backend selection for BUDDY."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from buddy_core.logging_utils import get_buddy_logger
from buddy_core.neck.base import BuddyNeckError, NeckBackend
from buddy_core.neck.mock import MockNeckBackend


def get_neck_backend(
    config: Mapping[str, Any],
) -> NeckBackend:
    """Select mock or physical neck backend safely."""

    hardware = config.get("hardware")

    if not isinstance(hardware, Mapping):
        raise BuddyNeckError(
            "Missing or invalid hardware configuration."
        )

    enabled = hardware.get("enabled")
    simulation = hardware.get("simulation")
    neck = hardware.get("neck")

    if not isinstance(neck, Mapping):
        raise BuddyNeckError(
            "Missing or invalid hardware.neck configuration."
        )

    neck_enabled = neck.get("enabled")

    if not isinstance(enabled, bool):
        raise BuddyNeckError(
            "hardware.enabled must be boolean."
        )

    if not isinstance(simulation, bool):
        raise BuddyNeckError(
            "hardware.simulation must be boolean."
        )

    if not isinstance(neck_enabled, bool):
        raise BuddyNeckError(
            "hardware.neck.enabled must be boolean."
        )

    logger = get_buddy_logger("neck.factory")

    if not enabled or simulation or not neck_enabled:
        logger.info(
            "Physical neck disabled; using mock neck backend."
        )
        return MockNeckBackend()

    pan = neck.get("pan")
    tilt = neck.get("tilt")

    if not isinstance(pan, Mapping) or not isinstance(tilt, Mapping):
        raise BuddyNeckError(
            "Invalid PAN or TILT neck configuration."
        )

    logger.warning(
        "Physical PCA9685 neck backend requested."
    )

    from buddy_core.neck.pca9685 import PCA9685NeckBackend

    return PCA9685NeckBackend(
        i2c_address=int(neck["i2c_address"]),
        frequency_hz=float(neck["frequency_hz"]),
        pan_channel=int(pan["channel"]),
        tilt_channel=int(tilt["channel"]),
    )

"""GPIO backend selection for BUDDY."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from buddy_core.hardware.base import (
    BuddyHardwareError,
    GPIOBackend,
)
from buddy_core.hardware.mock_gpio import (
    MockGPIOBackend,
)
from buddy_core.logging_utils import (
    get_buddy_logger,
)


def _validate_hardware_config(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the Task 09 hardware configuration."""

    hardware = config.get("hardware")

    if not isinstance(hardware, Mapping):
        raise BuddyHardwareError(
            "Missing or invalid 'hardware' configuration section."
        )

    enabled = hardware.get("enabled")
    simulation = hardware.get("simulation")

    if not isinstance(enabled, bool):
        raise BuddyHardwareError(
            "hardware.enabled must be boolean."
        )

    if not isinstance(simulation, bool):
        raise BuddyHardwareError(
            "hardware.simulation must be boolean."
        )

    gpio_config = hardware.get("gpio")

    if not isinstance(gpio_config, Mapping):
        raise BuddyHardwareError(
            "hardware.gpio must be a mapping."
        )

    numbering_mode = gpio_config.get(
        "numbering_mode"
    )

    if numbering_mode not in {"BCM", "BOARD"}:
        raise BuddyHardwareError(
            "hardware.gpio.numbering_mode must "
            "be 'BCM' or 'BOARD'."
        )

    return hardware


def get_gpio_backend(
    config: Mapping[str, Any],
) -> GPIOBackend:
    """Select a safe GPIO backend from BUDDY configuration."""

    hardware = _validate_hardware_config(
        config
    )

    enabled = hardware["enabled"]
    simulation = hardware["simulation"]

    logger = get_buddy_logger(
        "hardware.factory"
    )

    if not enabled:
        logger.info(
            "Physical hardware disabled; "
            "using mock GPIO backend."
        )

        return MockGPIOBackend()

    if simulation:
        logger.info(
            "Hardware simulation enabled; "
            "using mock GPIO backend."
        )

        return MockGPIOBackend()

    numbering_mode = hardware["gpio"][
        "numbering_mode"
    ]

    logger.warning(
        "Physical GPIO backend requested."
    )

    # Import only when physical hardware was explicitly requested.
    from buddy_core.hardware.raspberry_pi_gpio import (
        RaspberryPiGPIOBackend,
    )

    return RaspberryPiGPIOBackend(
        numbering_mode=numbering_mode,
    )

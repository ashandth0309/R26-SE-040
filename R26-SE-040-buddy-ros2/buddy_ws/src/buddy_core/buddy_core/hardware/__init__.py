"""BUDDY hardware abstraction package."""

from buddy_core.hardware.base import (
    BuddyHardwareError,
    GPIOBackend,
)
from buddy_core.hardware.factory import (
    get_gpio_backend,
)
from buddy_core.hardware.mock_gpio import (
    MockGPIOBackend,
)

__all__ = [
    "BuddyHardwareError",
    "GPIOBackend",
    "MockGPIOBackend",
    "get_gpio_backend",
]

"""Motor-control abstractions for BUDDY."""

from .drive import BuddyDrive
from .factory import (
    BuddyMotorSystem,
    create_motor_system,
)
from .tb6612 import (
    TB6612Motor,
    TB6612Standby,
)

__all__ = [
    "BuddyDrive",
    "BuddyMotorSystem",
    "TB6612Motor",
    "TB6612Standby",
    "create_motor_system",
]

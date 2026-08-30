"""BUDDY neck control package."""

from buddy_core.neck.base import (
    BuddyNeckError,
    NeckBackend,
)
from buddy_core.neck.factory import (
    get_neck_backend,
)
from buddy_core.neck.mock import (
    MockNeckBackend,
)

__all__ = [
    "BuddyNeckError",
    "NeckBackend",
    "MockNeckBackend",
    "get_neck_backend",
]

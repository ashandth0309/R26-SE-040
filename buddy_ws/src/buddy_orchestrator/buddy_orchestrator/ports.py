"""What the coordinator needs from the outside world, as callables.

The ROS node supplies ROS-backed ports; tests supply recorders or the real
in-process components.  Port calls may raise; the coordinator treats any
exception as "that module is unavailable" and degrades, never crashes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple, Protocol

from buddy_orchestrator.models import MoveIn, NavRequest
from buddy_orchestrator.speech import Utterance


class VisionCommandResult(NamedTuple):
    handled: bool
    reply: str = ""
    pass_to_dialogue: bool = False


class VisionPort(Protocol):
    """Typed requests to the vision node (its four services)."""

    def command(self, text: str, person_id: str = "") -> VisionCommandResult:
        ...

    def identify(self) -> dict:
        """``/vision/identify_object`` response as a dict."""

    def teach(self, action: str, text: str) -> dict:
        """``/vision/teach_object`` response as a dict."""

    def where_is(self, name: str) -> dict:
        """``/vision/where_is`` response as a dict."""


@dataclass
class Ports:
    """Everything the coordinator can do to the world."""

    #: Publish a speech request on /voice/say.
    speak: Callable[[Utterance], None]
    #: Forward a NavigationCommand to /navigation/command.
    nav_send: Callable[[NavRequest], None]
    #: Publish a STOP straight to /movement/request (no navigation needed).
    movement_stop: Callable[[str], None]
    #: Relay a voice MovementCommand to /movement/request.
    movement_relay: Callable[[MoveIn], None]
    vision: VisionPort | None = None

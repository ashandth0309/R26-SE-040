"""Hardware ownership audit.

Every device has exactly one owning node (``contracts.HARDWARE_OWNERS``).
The coordinator calls ``audit`` with the nodes and publisher counts it can
see on the ROS graph and reports every violation, so two nodes fighting
over the microphone, speaker or camera is a startup diagnostic rather than
a mysterious audio or camera failure later.

The coordinator itself owns no hardware.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from buddy_orchestrator.contracts import (
    HARDWARE_OWNERS,
    LEGACY_NODES,
    PUBLISHER_LIMITS,
)


@dataclass(frozen=True)
class Conflict:
    """One ownership violation."""

    resource: str
    nodes: tuple[str, ...]
    message: str


def audit(nodes: Iterable[str],
          publisher_counts: Mapping[str, int] | None = None
          ) -> list[Conflict]:
    """Return every ownership conflict visible on the graph.

    ``nodes`` is the list of node names (a name appears once per running
    node, so a duplicate is a second instance).  ``publisher_counts`` maps a
    topic to its current publisher count.
    """

    counts = Counter(nodes)
    found: list[Conflict] = []

    for resource, owners in HARDWARE_OWNERS.items():
        for owner in owners:
            if counts[owner] > 1:
                found.append(Conflict(
                    resource, (owner,) * counts[owner],
                    f"{counts[owner]} nodes named '{owner}' are running; "
                    f"only one may own the {resource}."))

    for legacy, (resources, owner) in LEGACY_NODES.items():
        if counts[legacy] and counts[owner]:
            for resource in resources:
                found.append(Conflict(
                    resource, (legacy, owner),
                    f"'{legacy}' and '{owner}' both claim the {resource}; "
                    f"stop '{legacy}'."))

    for topic, limit in PUBLISHER_LIMITS.items():
        seen = (publisher_counts or {}).get(topic, 0)

        if seen > limit:
            found.append(Conflict(
                topic, (),
                f"{seen} publishers on {topic}; at most {limit} allowed."))

    return found


def missing_owners(nodes: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Resources whose owning node is not running."""

    present = set(nodes)

    return {resource: owners for resource, owners in HARDWARE_OWNERS.items()
            if not present.intersection(owners)}

"""The three deployment profiles.

One coordinator runs in all of them; only these switches differ.

* ``offline``    Essential operation with no network at all.
* ``hybrid``     Local models for anything time-critical; online services
                 only where the user has enabled them, and only for
                 enhancements.  Losing the network falls back to offline
                 behaviour without touching safety.
* ``simulation`` Mock microphones, speaker, camera, ultrasonic sensors,
                 pose and motors.  Physical motion is impossible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    cloud_allowed: bool
    physical_motion_allowed: bool
    mock_hardware: bool
    #: Person approach can only be demonstrated against the mock.
    mock_person_approach: bool


PROFILES = {
    "offline": Profile("offline", False, True, False, False),
    "hybrid": Profile("hybrid", True, True, False, False),
    "simulation": Profile("simulation", False, False, True, True),
}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown profile '{name}'; choose one of "
            f"{', '.join(sorted(PROFILES))}") from None

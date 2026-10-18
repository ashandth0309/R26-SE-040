"""Global vision switches and performance profiles (Phases 18 to 20).

Per-person choices (greet, proactive check-ins, expression, quiet) live
with each person's consented profile in the face store.  This file holds
the choices that apply to the whole robot.  Defaults lean private:
object memory is off, and expression reading needs both this switch and
the person's own opt-in.

The profile intervals are starting values.  Their CPU cost on the
Raspberry Pi 4 has not been measured; run the benchmark harness.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROFILES = ("lightweight", "balanced", "enhanced")

# seconds between runs of each stage (None = stage off in this profile)
PLANS: dict[str, dict[str, float | None]] = {
    "lightweight": {"faces": 0.2, "expression": None, "objects": 2.0,
                    "pose": None, "scene": None},
    "balanced": {"faces": 0.1, "expression": 0.5, "objects": 1.0,
                 "pose": 0.5, "scene": 10.0},
    "enhanced": {"faces": 0.1, "expression": 0.25, "objects": 0.5,
                 "pose": 0.25, "scene": 5.0},
}

BOOLS = ("camera_enabled", "recognition", "expression", "objects", "pose",
         "fall_detection", "gestures", "object_memory", "proactive",
         "greeting")


class SettingsError(ValueError):
    pass


@dataclass
class VisionSettings:
    profile: str = "balanced"
    camera_enabled: bool = True       # False releases the camera entirely
    recognition: bool = True
    expression: bool = True           # also needs the person's opt-in
    objects: bool = True
    pose: bool = True                 # posture, gestures, fall detection
    fall_detection: bool = True
    gestures: bool = True
    object_memory: bool = False       # off until the user asks for it
    proactive: bool = True            # also needs the person's opt-in
    greeting: bool = True
    overrides: dict = field(default_factory=dict)   # stage -> seconds

    def plan(self) -> dict[str, float | None]:
        plan = dict(PLANS[self.profile])

        for k, v in self.overrides.items():
            if k in plan:
                plan[k] = v

        for key, flag in (("expression", self.expression),
                          ("objects", self.objects), ("pose", self.pose)):
            if not flag:
                plan[key] = None

        return plan

    def validate(self) -> None:
        if self.profile not in PROFILES:
            raise SettingsError(f"unknown profile {self.profile!r}")

        for k in BOOLS:
            if not isinstance(getattr(self, k), bool):
                raise SettingsError(f"{k} must be true or false")

        for k, v in self.overrides.items():
            if k not in PLANS["balanced"]:
                raise SettingsError(f"unknown stage {k!r}")

            if v is not None and not (isinstance(v, (int, float))
                                      and 0.05 <= v <= 600):
                raise SettingsError(f"{k} interval out of range")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> VisionSettings:
        known = set(cls.__dataclass_fields__)
        extra = set(d) - known

        if extra:
            raise SettingsError(f"unknown settings: {sorted(extra)}")

        s = cls(**d)
        s.validate()

        return s


def load(path: str | os.PathLike) -> VisionSettings:
    p = Path(path)

    if not p.is_file():
        return VisionSettings()

    try:
        return VisionSettings.from_dict(json.loads(p.read_text()))
    except (ValueError, TypeError) as exc:
        raise SettingsError(f"bad settings file: {exc}") from exc


def save(settings: VisionSettings, path: str | os.PathLike) -> None:
    settings.validate()
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)

    with os.fdopen(fd, "w") as f:
        json.dump(settings.to_dict(), f, indent=2)

    os.replace(tmp, p)

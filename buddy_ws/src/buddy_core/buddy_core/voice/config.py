"""Load the voice companion settings from YAML with safe defaults."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from buddy_core.voice.audio_manager import ManagerConfig
from buddy_core.voice.echo import EchoConfig
from buddy_core.voice.speaker_manager import SpeakerConfig
from buddy_core.voice.vad import VadConfig


@dataclass
class AudioSettings:
    """Device preferences.  Empty means auto-discover."""

    preferred_inputs: list[str] = field(default_factory=list)
    preferred_output: str | None = None
    max_microphones: int = 2
    allow_virtual_devices: bool = False


@dataclass
class VoiceSettings:
    """All Phase 2 settings in one place."""

    profile: str = "balanced"
    audio: AudioSettings = field(default_factory=AudioSettings)
    vad: VadConfig = field(default_factory=VadConfig)
    mics: ManagerConfig = field(default_factory=ManagerConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    echo: EchoConfig = field(default_factory=EchoConfig)


def _apply(target: Any, values: dict) -> Any:
    """Return ``target`` with known keys from ``values`` applied."""

    if not isinstance(values, dict):
        return target

    names = {f.name for f in fields(target)}
    updates = {}

    for key, value in values.items():
        if key not in names:
            continue

        current = getattr(target, key)

        if is_dataclass(current):
            updates[key] = _apply(current, value)
        else:
            updates[key] = value

    if target.__class__.__dataclass_params__.frozen:
        from dataclasses import replace

        return replace(target, **updates)

    for key, value in updates.items():
        setattr(target, key, value)

    return target


def load_settings(path: str | Path | None = None) -> VoiceSettings:
    """Read settings; missing file or bad YAML gives the defaults."""

    settings = VoiceSettings()

    if path is None:
        return settings

    try:
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return settings

    if not isinstance(data, dict):
        return settings

    if "profile" in data:
        settings.profile = str(data["profile"])

    settings.audio = _apply(settings.audio, data.get("audio", {}))
    settings.vad = _apply(settings.vad, data.get("vad", {}))
    settings.mics = _apply(settings.mics, data.get("microphones", {}))
    settings.speaker = _apply(settings.speaker, data.get("speaker", {}))
    settings.echo = _apply(settings.echo, data.get("echo", {}))

    return settings

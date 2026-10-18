"""Load the voice companion settings from YAML with safe defaults."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from buddy_core.voice.audio_manager import ManagerConfig
from buddy_core.voice.backchannel import BackchannelConfig
from buddy_core.voice.command_router import RouterConfig
from buddy_core.voice.echo import EchoConfig
from buddy_core.voice.speaker_manager import SpeakerConfig
from buddy_core.voice.turn_manager import TurnConfig
from buddy_core.voice.vad import VadConfig
from buddy_core.voice.voice_engine import EngineConfig


@dataclass
class AudioSettings:
    """Device preferences.  Empty means auto-discover."""

    preferred_inputs: list[str] = field(default_factory=list)
    preferred_output: str | None = None
    max_microphones: int = 2
    allow_virtual_devices: bool = False


@dataclass
class ModelSettings:
    """Model locations.  Binaries are never committed; set on the robot."""

    vosk_model_path: str = ""
    piper_model_path: str = ""
    aplay_device: str | None = None


@dataclass
class MemorySettings:
    """Where companion memory lives."""

    enabled: bool = True
    db_path: str = "~/.buddy/companion.db"
    export_dir: str = "~/.buddy/exports"


@dataclass
class OnlineSettings:
    """Online features.  Everything is off by default."""

    cloud_enabled: bool = False
    local_llm: bool = False
    local_llm_model: str = "llama3.2:1b"
    local_llm_url: str = "http://127.0.0.1:11434"
    cloud_base_url: str = ""
    cloud_model: str = ""
    cloud_api_key_env: str = "BUDDY_LLM_API_KEY"


@dataclass
class FeatureSettings:
    """Feature switches the profiles set."""

    emotion_audio: bool = True
    backchannel: bool = True
    proactive: bool = True


@dataclass
class NavigationSettings:
    """Navigation hand-off.  Off until the navigation package is merged."""

    enabled: bool = False
    locations: list[str] = field(default_factory=list)


@dataclass
class VoiceSettings:
    """All voice companion settings in one place."""

    profile: str = "balanced"
    audio: AudioSettings = field(default_factory=AudioSettings)
    vad: VadConfig = field(default_factory=VadConfig)
    mics: ManagerConfig = field(default_factory=ManagerConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    echo: EchoConfig = field(default_factory=EchoConfig)
    models: ModelSettings = field(default_factory=ModelSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    online: OnlineSettings = field(default_factory=OnlineSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    navigation: NavigationSettings = field(
        default_factory=NavigationSettings)
    router: RouterConfig = field(default_factory=RouterConfig)
    turn: TurnConfig = field(default_factory=TurnConfig)
    backchannel: BackchannelConfig = field(default_factory=BackchannelConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    personality_path: str = ""


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

    for name in ("models", "memory", "online", "features", "navigation",
                 "router", "turn", "backchannel", "engine"):
        setattr(settings, name,
                _apply(getattr(settings, name), data.get(name, {})))

    settings.personality_path = str(data.get("personality_path", ""))

    return apply_profile(settings)


PROFILES = ("lightweight", "balanced", "enhanced")


def apply_profile(settings: VoiceSettings) -> VoiceSettings:
    """Set feature switches for the chosen profile.

    * lightweight: fewest CPU-heavy features.
    * balanced: local features, no language model.
    * enhanced: balanced plus the local language model if configured.

    CPU and memory cost of each has NOT been measured on the Pi.
    """

    profile = settings.profile

    if profile == "lightweight":
        settings.features.emotion_audio = False
        settings.features.backchannel = False
        settings.features.proactive = False
        settings.online.local_llm = False
        settings.online.cloud_enabled = False
    elif profile == "enhanced":
        settings.features.emotion_audio = True
        settings.features.backchannel = True
        settings.features.proactive = True
    elif profile != "balanced":
        settings.profile = "balanced"

    return settings

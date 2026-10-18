"""Assemble the whole voice companion from settings and backends.

Everything that touches hardware or heavy models lives behind
``VoiceBackends``.  ``build_companion`` is pure Python and is exercised in
tests with fakes.  ``real_backends`` is the only code that opens devices
and loads Vosk or Piper, and it is UNVERIFIED until run on the Pi.
"""

from __future__ import annotations

import datetime
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from buddy_core.voice.action_gate import ActionGate
from buddy_core.voice.audio_devices import (
    MicTopology,
    SoundDeviceBackend,
    discover_inputs,
    discover_output,
)
from buddy_core.voice.audio_manager import (
    AudioSource,
    MicrophoneManager,
    build_sources,
)
from buddy_core.voice.backchannel import BackchannelManager
from buddy_core.voice.command_router import (
    CommandRouter,
    MovementSink,
    NavigationSink,
)
from buddy_core.voice.companion_commands import (
    CompanionControls,
    bind_engine,
)
from buddy_core.voice.config import VoiceSettings
from buddy_core.voice.connectivity import ConnectivityManager
from buddy_core.voice.dialogue_manager import (
    ConversationContext,
    DialogueManager,
    IntentClassifier,
    MlIntentClassifier,
)
from buddy_core.voice.echo import EchoGuard
from buddy_core.voice.emotion_audio import AudioEmotionEstimator
from buddy_core.voice.emotion_fusion import EmotionFusion
from buddy_core.voice.emotional_response import (
    EmotionalResponseManager,
    make_utterance_observer,
)
from buddy_core.voice.language import LanguageGuard, LanguagePolicy
from buddy_core.voice.llm_router import (
    LlmRouter,
    OllamaProvider,
    OpenAICompatProvider,
    ProviderEntry,
    RouterMode,
)
from buddy_core.voice.memory_store import CompanionMemoryManager, MemoryConfig
from buddy_core.voice.personality import (
    BuddyPersonalityEngine,
    load_personality,
)
from buddy_core.voice.proactive import ProactiveSuggester
from buddy_core.voice.speaker_manager import PlaybackBackend, SpeakerManager
from buddy_core.voice.stt import Recognizer, VoskRecognizer
from buddy_core.voice.tts_backends import (
    AplayPlayback,
    PiperSynthesizer,
    Synthesizer,
)
from buddy_core.voice.turn_manager import TurnManager
from buddy_core.voice.vad import VoiceActivityDetector
from buddy_core.voice.voice_commands import command_grammar, wake_grammar
from buddy_core.voice.voice_engine import VoiceEngine

WARM_PHRASES = [
    "Stopping.", "Yes?", "Going forward.", "Coming back.",
    "Turning left.", "Turning right.", "Okay, going to sleep.",
    "Sorry, I didn't catch that.",
]


class ModelMissing(RuntimeError):
    """A required model file is not configured or not present."""


@dataclass
class VoiceBackends:
    """Everything hardware- or model-specific."""

    mic_sources: list[AudioSource]
    topology: MicTopology
    playback: PlaybackBackend
    synthesizer: Synthesizer
    wake_rec: Recognizer
    command_rec: Recognizer
    general_rec: Recognizer
    classifier: IntentClassifier | None = None
    cloud_probe: Callable[[], bool] | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class Companion:
    """The assembled system."""

    settings: VoiceSettings
    engine: VoiceEngine
    mics: MicrophoneManager
    speaker: SpeakerManager
    router: CommandRouter
    memory: CompanionMemoryManager
    controls: CompanionControls
    emotion: EmotionalResponseManager
    suggester: ProactiveSuggester | None
    connectivity: ConnectivityManager
    llm: LlmRouter
    personality: BuddyPersonalityEngine
    guard: EchoGuard
    turn: TurnManager
    cloud_state: dict
    notes: list[str] = field(default_factory=list)

    def status(self) -> dict:
        """Compact, privacy-safe status (no text, no audio)."""

        health = self.mics.health()

        return {
            "state": self.turn.state.value,
            "awake": self.turn.awake,
            "muted": self.engine.muted,
            "silence_mode": self.controls.silence_mode.value,
            "moving": self.router.moving,
            "mic_topology": health.topology.value,
            "mic_active": health.active,
            "mic_status": {k: v.value for k, v in health.statuses.items()},
            "speaker_failed": self.speaker.backend_failed,
            "cloud": self.connectivity.link.value,
            "llm_mode": self.llm.mode.value,
            "memory_consent": self.memory.consent,
            "errors": self.engine.errors,
            "profile": self.settings.profile,
        }


def alsa_name_from_device(name: str) -> str | None:
    """Turn a PortAudio name like '... (hw:1,0)' into 'plughw:1,0'."""

    match = re.search(r"hw:(\d+),(\d+)", name)

    return f"plughw:{match.group(1)},{match.group(2)}" if match else None


def build_companion(
    settings: VoiceSettings,
    backends: VoiceBackends,
    movement_sink: MovementSink,
    navigation_sink: NavigationSink | None = None,
    *,
    clock: Callable[[], float] = time.monotonic,
    executor=None,
    rng: random.Random | None = None,
    memory: CompanionMemoryManager | None = None,
    command_observer: Callable | None = None,
) -> Companion:
    """Wire every component together."""

    rng = rng or random.Random()
    features = settings.features
    notes = list(backends.notes)

    mem_cfg = MemoryConfig(enabled=settings.memory.enabled)

    if memory is None:
        db = ":memory:"

        if settings.memory.enabled:
            path = Path(settings.memory.db_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            db = str(path)

        memory = CompanionMemoryManager(db, config=mem_cfg)

    cloud_available = bool(settings.online.cloud_base_url
                           and settings.online.cloud_model)
    cloud_state = {"on": bool(settings.online.cloud_enabled
                              and cloud_available)}
    connectivity = ConnectivityManager(
        enabled=lambda: cloud_state["on"], probe=backends.cloud_probe,
        clock=clock)
    entries: list[ProviderEntry] = []

    if settings.online.local_llm:
        entries.append(ProviderEntry("ollama", "local", OllamaProvider(
            settings.online.local_llm_model, settings.online.local_llm_url)))

    if cloud_available:
        entries.append(ProviderEntry("cloud", "cloud", OpenAICompatProvider(
            settings.online.cloud_base_url, settings.online.cloud_model,
            api_key_env=settings.online.cloud_api_key_env)))

    llm = LlmRouter(entries, RouterMode.OFFLINE, connectivity,
                    cloud_enabled=lambda: cloud_state["on"], clock=clock)

    def refresh_llm_mode() -> None:
        if cloud_state["on"] and cloud_available:
            llm.mode = RouterMode.AUTO
        elif settings.online.local_llm:
            llm.mode = RouterMode.LOCAL
        else:
            llm.mode = RouterMode.OFFLINE

    refresh_llm_mode()

    nav_enabled = settings.navigation.enabled and navigation_sink is not None
    router = CommandRouter(
        movement_sink, navigation_sink if nav_enabled else None,
        config=replace(settings.router, use_navigation=nav_enabled),
        clock=clock)

    speaker = SpeakerManager(backends.playback, config=settings.speaker,
                             clock=clock)
    guard = EchoGuard(settings.echo, clock=clock)
    turn = TurnManager(settings.turn, clock=clock)
    fusion = EmotionFusion(clock=clock)
    emotion = EmotionalResponseManager(fusion, rng=rng)
    personality = BuddyPersonalityEngine(
        load_personality(settings.personality_path or None), rng=rng)
    controls = CompanionControls(
        memory,
        export_dir=Path(settings.memory.export_dir).expanduser(),
        cloud_available=lambda: cloud_available,
        cloud_enabled=lambda: cloud_state["on"],
        clock=clock)

    def set_cloud(enabled: bool) -> None:
        cloud_state["on"] = bool(enabled and cloud_available)
        refresh_llm_mode()

    controls.set_cloud = set_cloud
    policy = LanguagePolicy().with_online(cloud_state["on"])
    language = LanguageGuard(policy)

    suggester = ProactiveSuggester(memory, rng=rng, clock=clock) \
        if features.proactive else None
    hooks = [controls.hook, language.hook, emotion.hook]

    if suggester is not None:
        hooks.append(suggester.hook)

    dialogue = DialogueManager(
        classifier=backends.classifier or None, llm=llm,
        llm_allowed=lambda: llm.mode is not RouterMode.OFFLINE,
        hooks=hooks, rng=rng)

    bc_cfg = replace(settings.backchannel,
                     enabled=settings.backchannel.enabled
                     and features.backchannel)
    estimator = AudioEmotionEstimator()
    observer = make_utterance_observer(emotion, estimator) \
        if features.emotion_audio else None

    engine = VoiceEngine(
        vad=VoiceActivityDetector(settings.vad),
        wake_rec=backends.wake_rec, command_rec=backends.command_rec,
        general_rec=backends.general_rec, router=router, speaker=speaker,
        synth=backends.synthesizer, guard=guard, turn=turn,
        dialogue=dialogue,
        backchannel=BackchannelManager(bc_cfg, rng=rng),
        context=ConversationContext(),
        config=settings.engine,
        known_locations=tuple(settings.navigation.locations),
        clock=clock, emotion_hint=emotion.hint,
        utterance_observer=observer, style_provider=emotion.style,
        executor=executor, action_gate=ActionGate(clock=clock),
        reply_filter=personality.finish,
        command_observer=command_observer)
    bind_engine(controls, engine)
    engine.tick_hooks.append(connectivity.tick)

    saved = memory.recall("name")

    if saved:
        engine.context.user_name = saved.title()

    mics = MicrophoneManager(backends.mic_sources,
                             topology=backends.topology,
                             config=settings.mics, clock=clock)

    return Companion(settings, engine, mics, speaker, router, memory,
                     controls, emotion, suggester, connectivity, llm,
                     personality, guard, turn, cloud_state, notes)


# ----- real hardware / models (UNVERIFIED off the Pi) -----

def real_backends(settings: VoiceSettings) -> VoiceBackends:
    """Open devices and load models.  Raises ``ModelMissing`` clearly."""

    models = settings.models

    for label, value in (("vosk_model_path", models.vosk_model_path),
                         ("piper_model_path", models.piper_model_path)):
        if not value or not Path(value).expanduser().exists():
            raise ModelMissing(
                f"{label} is not set or does not exist: '{value}'. Set "
                f"it in the voice config or ROS parameters.")

    notes: list[str] = []
    audio = SoundDeviceBackend()
    selection = discover_inputs(
        audio, preferred=settings.audio.preferred_inputs,
        max_microphones=settings.audio.max_microphones,
        allow_virtual=settings.audio.allow_virtual_devices)
    notes.extend(selection.notes)

    if not selection.usable:
        raise RuntimeError("No usable microphone was found.")

    sources = build_sources(selection)
    out = discover_output(audio, preferred=settings.audio.preferred_output,
                          allow_virtual=settings.audio.allow_virtual_devices)
    device = models.aplay_device or (
        alsa_name_from_device(out.device.name) if out else None)
    notes.append(f"playback device: {device or 'ALSA default'}")

    from vosk import Model

    vosk_model = Model(str(Path(models.vosk_model_path).expanduser()))
    known = ()

    return VoiceBackends(
        mic_sources=sources, topology=selection.topology,
        playback=AplayPlayback(device),
        synthesizer=PiperSynthesizer(
            str(Path(models.piper_model_path).expanduser())),
        wake_rec=VoskRecognizer(vosk_model, grammar=wake_grammar()),
        command_rec=VoskRecognizer(vosk_model,
                                   grammar=command_grammar(known)),
        general_rec=VoskRecognizer(vosk_model),
        classifier=MlIntentClassifier(), notes=notes)


def current_hour() -> int:
    """Local hour of day (for proactive suggestions)."""

    return datetime.datetime.now().hour


class NullPlayback:
    """Playback that plays nothing (dry runs)."""

    def __init__(self) -> None:
        self._until = 0.0

    def start(self, samples, rate, gain) -> None:
        self._until = time.monotonic() + samples.size / rate

    def is_playing(self) -> bool:
        return time.monotonic() < self._until

    def stop(self) -> None:
        self._until = 0.0


def simulated_backends() -> VoiceBackends:
    """Silent, scripted backends for desktop dry runs (no hardware)."""

    from buddy_core.voice.audio_manager import SimulatedSource
    from buddy_core.voice.stt import ScriptedRecognizer
    from buddy_core.voice.tts_backends import FakeSynthesizer

    return VoiceBackends(
        mic_sources=[SimulatedSource("simulated")],
        topology=MicTopology.SIMULATED,
        playback=NullPlayback(), synthesizer=FakeSynthesizer(),
        wake_rec=ScriptedRecognizer(), command_rec=ScriptedRecognizer(),
        general_rec=ScriptedRecognizer(),
        notes=["simulated backends: no hardware, no models"])

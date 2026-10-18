"""Audio device discovery and microphone/speaker selection.

Nothing here hardcodes a device index or name.  A caller may pass a
*preferred-device override* (a substring of the device name or an index);
otherwise the best available devices are chosen automatically.

Microphone topologies (``MicTopology``), best first:

``TWO_DEVICES``
    Two independent input devices (for example two USB microphones).  Their
    clocks are not synchronised, so their streams must never be summed
    sample by sample.
``STEREO``
    One device with two physical input channels.  The channels share a
    clock, so they are time aligned.
``SINGLE``
    One microphone (fallback).
``NONE``
    Nothing usable was found.

The sound backend is injectable so everything can be tested without audio
hardware.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

#: Sample rates tried in order of preference.  16 kHz is what Vosk wants;
#: 48 kHz is the usual native rate of cheap USB microphones.
PREFERRED_RATES = (16000, 48000, 44100, 32000, 22050, 8000)

#: Name fragments of devices that are not real microphones or speakers.
VIRTUAL_NAME_HINTS = (
    "monitor of",
    "loopback",
    "null",
    "dummy",
    "hdmi",
    "spdif",
)


class MicTopology(str, Enum):
    """How the available microphones are arranged."""

    TWO_DEVICES = "two_devices"
    STEREO = "stereo"
    SINGLE = "single"
    SIMULATED = "simulated"
    NONE = "none"


@dataclass(frozen=True)
class DeviceInfo:
    """One audio device as reported by the backend."""

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    default_low_input_latency: float = 0.0
    default_low_output_latency: float = 0.0
    hostapi: str = ""

    @property
    def is_input(self) -> bool:
        """Whether the device can record."""

        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        """Whether the device can play."""

        return self.max_output_channels > 0

    @property
    def looks_virtual(self) -> bool:
        """Heuristic: loopbacks, monitors and HDMI sinks are not mics."""

        lowered = self.name.lower()

        return any(hint in lowered for hint in VIRTUAL_NAME_HINTS)


class AudioBackend(Protocol):
    """What the discovery code needs from a sound library."""

    def query_devices(self) -> Sequence[DeviceInfo]:
        """Return every device."""

    def supports_input(
        self,
        index: int,
        samplerate: int,
        channels: int,
    ) -> bool:
        """Whether the device can record with these settings."""

    def supports_output(
        self,
        index: int,
        samplerate: int,
        channels: int,
    ) -> bool:
        """Whether the device can play with these settings."""


class SoundDeviceBackend:
    """``sounddevice`` implementation (imported lazily)."""

    def __init__(self) -> None:
        import sounddevice

        self._sd = sounddevice

    def query_devices(self) -> list[DeviceInfo]:
        """List devices known to PortAudio."""

        hostapis = self._sd.query_hostapis()
        found = []

        for index, raw in enumerate(self._sd.query_devices()):
            hostapi = ""

            try:
                hostapi = hostapis[raw["hostapi"]]["name"]
            except (IndexError, KeyError, TypeError):
                pass

            found.append(
                DeviceInfo(
                    index=index,
                    name=str(raw["name"]),
                    max_input_channels=int(raw["max_input_channels"]),
                    max_output_channels=int(raw["max_output_channels"]),
                    default_samplerate=float(raw["default_samplerate"]),
                    default_low_input_latency=float(
                        raw.get("default_low_input_latency", 0.0)
                    ),
                    default_low_output_latency=float(
                        raw.get("default_low_output_latency", 0.0)
                    ),
                    hostapi=hostapi,
                )
            )

        return found

    def supports_input(self, index, samplerate, channels) -> bool:
        """Ask PortAudio whether the input settings are valid."""

        try:
            self._sd.check_input_settings(
                device=index,
                channels=channels,
                samplerate=samplerate,
                dtype="int16",
            )
        except Exception:
            return False

        return True

    def supports_output(self, index, samplerate, channels) -> bool:
        """Ask PortAudio whether the output settings are valid."""

        try:
            self._sd.check_output_settings(
                device=index,
                channels=channels,
                samplerate=samplerate,
                dtype="int16",
            )
        except Exception:
            return False

        return True


@dataclass(frozen=True)
class InputChoice:
    """One chosen input, possibly a single channel of a stereo device."""

    device: DeviceInfo
    channel: int | None
    samplerate: int
    channels_to_open: int
    latency_s: float

    @property
    def label(self) -> str:
        """Short human-readable identifier."""

        suffix = "" if self.channel is None else f"[ch{self.channel}]"

        return f"{self.device.name}{suffix}"


@dataclass(frozen=True)
class InputSelection:
    """Result of microphone discovery."""

    topology: MicTopology
    inputs: tuple[InputChoice, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def usable(self) -> bool:
        """Whether at least one input exists."""

        return bool(self.inputs)


@dataclass(frozen=True)
class OutputChoice:
    """The chosen speaker."""

    device: DeviceInfo
    samplerate: int
    latency_s: float


def _matches(device: DeviceInfo, preferred: str | int | None) -> bool:
    if preferred is None or preferred == "":
        return False

    if isinstance(preferred, int):
        return device.index == preferred

    text = str(preferred)

    if text.isdigit():
        return device.index == int(text)

    return text.lower() in device.name.lower()


def pick_samplerate(
    device: DeviceInfo,
    backend: AudioBackend,
    channels: int,
    *,
    output: bool = False,
    wanted: Iterable[int] = PREFERRED_RATES,
) -> int | None:
    """Return the first supported rate, trying the native rate as well."""

    check = backend.supports_output if output else backend.supports_input
    native = int(round(device.default_samplerate))
    candidates = list(dict.fromkeys([*wanted, native]))

    for rate in candidates:
        if check(device.index, rate, channels):
            return rate

    return None


def discover_inputs(
    backend: AudioBackend,
    *,
    preferred: Sequence[str | int] = (),
    max_microphones: int = 2,
    allow_virtual: bool = False,
) -> InputSelection:
    """Find the best microphone arrangement.

    ``preferred`` entries are tried first, in order, and decide which
    devices are used when several candidates exist.
    """

    notes: list[str] = []
    candidates = [
        d for d in backend.query_devices()
        if d.is_input and (allow_virtual or not d.looks_virtual)
    ]

    if not candidates:
        return InputSelection(
            MicTopology.NONE, (), ("No input devices found.",)
        )

    ordered: list[DeviceInfo] = []

    for wish in preferred:
        match = [d for d in candidates if _matches(d, wish)]

        if match:
            ordered.append(match[0])
        else:
            notes.append(f"Preferred device '{wish}' not found.")

    for device in candidates:
        if device not in ordered:
            ordered.append(device)

    usable: list[tuple[DeviceInfo, int, int]] = []

    for device in ordered:
        channels = min(device.max_input_channels, 2)
        rate = pick_samplerate(device, backend, channels)

        if rate is None and channels == 2:
            channels = 1
            rate = pick_samplerate(device, backend, channels)

        if rate is None:
            notes.append(f"'{device.name}' rejected every sample rate.")
            continue

        usable.append((device, channels, rate))

    if not usable:
        return InputSelection(MicTopology.NONE, (), tuple(notes))

    # Two independent devices win only when each is a distinct physical
    # device.  Several PortAudio host APIs can list the same hardware more
    # than once, so duplicates by name are collapsed first.
    distinct: list[tuple[DeviceInfo, int, int]] = []
    seen_names: set[str] = set()

    for item in usable:
        if item[0].name in seen_names:
            continue

        seen_names.add(item[0].name)
        distinct.append(item)

    if max_microphones >= 2 and len(distinct) >= 2:
        chosen = distinct[:2]
        inputs = tuple(
            InputChoice(
                device=device,
                channel=0 if channels == 2 else None,
                samplerate=rate,
                channels_to_open=channels,
                latency_s=device.default_low_input_latency,
            )
            for device, channels, rate in chosen
        )
        notes.append(
            "Two independent devices: clocks are not synchronised; "
            "streams are never summed."
        )

        return InputSelection(MicTopology.TWO_DEVICES, inputs, tuple(notes))

    device, channels, rate = distinct[0]

    if channels == 2 and max_microphones >= 2:
        inputs = tuple(
            InputChoice(device, ch, rate, 2, device.default_low_input_latency)
            for ch in (0, 1)
        )

        return InputSelection(MicTopology.STEREO, inputs, tuple(notes))

    inputs = (
        InputChoice(device, None, rate, channels,
                    device.default_low_input_latency),
    )
    notes.append("Single microphone fallback.")

    return InputSelection(MicTopology.SINGLE, inputs, tuple(notes))


def discover_output(
    backend: AudioBackend,
    *,
    preferred: str | int | None = None,
    allow_virtual: bool = False,
    wanted_rates: Iterable[int] = PREFERRED_RATES,
) -> OutputChoice | None:
    """Find the speaker; ``preferred`` overrides automatic choice."""

    devices = [
        d for d in backend.query_devices()
        if d.is_output and (allow_virtual or not d.looks_virtual)
    ]
    ordered = [d for d in devices if _matches(d, preferred)] + [
        d for d in devices if not _matches(d, preferred)
    ]

    for device in ordered:
        channels = min(device.max_output_channels, 2)
        rate = pick_samplerate(
            device, backend, channels, output=True, wanted=wanted_rates
        )

        if rate is not None:
            return OutputChoice(
                device, rate, device.default_low_output_latency
            )

    return None

"""Microphone sources and the microphone manager.

``MicrophoneManager`` owns one or two ``AudioSource`` objects, keeps a
quality score for each, and hands the rest of the pipeline a single 16 kHz
mono stream taken from the best source.

Design rules:

* Independent devices are never averaged.  Their sample clocks drift and
  their latencies differ, so summing them would comb-filter the speech and
  lower recognition quality.  The manager *selects* the better source
  instead, with hysteresis so it does not flap between them.
* Fusion of two channels is only offered for channels of one device (shared
  clock) and only when the configuration says the geometry is valid.  It is
  off by default.
* A source that stops delivering audio is restarted with exponential
  back-off and reported as degraded; the manager keeps running on whatever
  sources remain.
"""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from buddy_core.voice.audio_devices import (
    InputChoice,
    InputSelection,
    MicTopology,
)
from buddy_core.voice.dsp import (
    clip_ratio,
    dbfs,
    downmix,
    resample,
    rms,
    to_float,
    to_int16,
)

TARGET_RATE = 16000


class SourceStatus(str, Enum):
    """Health of one source."""

    STOPPED = "stopped"
    OK = "ok"
    RECOVERING = "recovering"
    FAILED = "failed"


@dataclass(frozen=True)
class AudioFrame:
    """A block of mono int16 audio at ``TARGET_RATE``."""

    samples: np.ndarray
    timestamp: float
    source: str


class AudioSource(ABC):
    """A producer of 16 kHz mono frames."""

    name: str
    status: SourceStatus

    @abstractmethod
    def start(self) -> bool:
        """Begin capturing.  Returns False if it could not start."""

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing and release the device."""

    @abstractmethod
    def read(self) -> AudioFrame | None:
        """Return the next frame without blocking, or None."""

    def maintain(self, now: float) -> None:
        """Hook for recovery logic; called regularly by the manager."""

    @property
    def nominal_rate(self) -> int:
        """Rate the underlying device delivers (for drift estimates)."""

        return TARGET_RATE


class SimulatedSource(AudioSource):
    """Feeds prepared audio; used by desktop tests and demos."""

    def __init__(
        self,
        name: str,
        samples: np.ndarray | None = None,
        *,
        block_samples: int = 320,
        clock: Callable[[], float] = time.monotonic,
        loop: bool = False,
    ) -> None:
        self.name = name
        self.status = SourceStatus.STOPPED
        self._block = block_samples
        self._clock = clock
        self._loop = loop
        self._data = (
            np.zeros(0, np.int16) if samples is None else samples.copy()
        )
        self._pos = 0
        self._fail = False
        self.start_calls = 0

    def feed(self, samples: np.ndarray) -> None:
        """Append more audio to be delivered."""

        self._data = np.concatenate([self._data, samples.astype(np.int16)])

    def start(self) -> bool:
        self.start_calls += 1

        if self._fail:
            self.status = SourceStatus.FAILED

            return False

        self.status = SourceStatus.OK

        return True

    def stop(self) -> None:
        self.status = SourceStatus.STOPPED

    def break_device(self) -> None:
        """Simulate unplugging: no more data and restarts fail."""

        self._fail = True
        self.status = SourceStatus.FAILED

    def repair_device(self) -> None:
        """Simulate plugging the device back in."""

        self._fail = False

    def read(self) -> AudioFrame | None:
        if self.status is not SourceStatus.OK:
            return None

        if self._pos + self._block > self._data.size:
            if not self._loop or self._data.size < self._block:
                return None

            self._pos = 0

        chunk = self._data[self._pos:self._pos + self._block]
        self._pos += self._block

        return AudioFrame(chunk, self._clock(), self.name)


class SoundDeviceSource(AudioSource):
    """Capture from a PortAudio device via ``sounddevice``.

    Device I/O cannot be exercised without hardware, so tests inject
    ``stream_factory`` and call ``_callback`` directly.  The callback does
    the minimum (copy into a bounded queue); conversion happens in
    ``read``.
    """

    def __init__(
        self,
        choice: InputChoice,
        *,
        name: str | None = None,
        channel_mode: int | None = None,
        stream_factory: Callable | None = None,
        clock: Callable[[], float] = time.monotonic,
        queue_blocks: int = 64,
        stall_timeout_s: float = 2.0,
        max_backoff_s: float = 8.0,
    ) -> None:
        self.choice = choice
        self.name = name or choice.label
        self.status = SourceStatus.STOPPED
        self._channel = (
            channel_mode if channel_mode is not None else choice.channel
        )
        self._factory = stream_factory
        self._clock = clock
        self._queue: queue.Queue = queue.Queue(maxsize=queue_blocks)
        self._stall = stall_timeout_s
        self._max_backoff = max_backoff_s
        self._stream = None
        self._last_data = 0.0
        self._next_retry = 0.0
        self._backoff = 0.5
        self.overflows = 0
        self.xruns = 0
        self.restarts = 0

    @property
    def nominal_rate(self) -> int:
        return self.choice.samplerate

    def _make_stream(self):
        if self._factory is not None:
            return self._factory(self.choice, self._callback)

        import sounddevice

        blocksize = int(self.choice.samplerate * 0.02)

        return sounddevice.InputStream(
            device=self.choice.device.index,
            samplerate=self.choice.samplerate,
            channels=self.choice.channels_to_open,
            dtype="int16",
            blocksize=blocksize,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.xruns += 1

        try:
            self._queue.put_nowait((self._clock(), indata.copy()))
        except queue.Full:
            self.overflows += 1

        self._last_data = self._clock()

    def start(self) -> bool:
        try:
            self._stream = self._make_stream()
            self._stream.start()
        except Exception:
            self._stream = None
            self.status = SourceStatus.FAILED

            return False

        self._last_data = self._clock()
        self.status = SourceStatus.OK
        self._backoff = 0.5

        return True

    def stop(self) -> None:
        stream, self._stream = self._stream, None

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

        self.status = SourceStatus.STOPPED

    def read(self) -> AudioFrame | None:
        try:
            stamp, block = self._queue.get_nowait()
        except queue.Empty:
            return None

        mono = downmix(block, self._channel)
        floats = to_float(mono) if mono.dtype == np.int16 else (
            mono.astype(np.float32) / 32768.0
        )
        converted = to_int16(
            resample(floats, self.choice.samplerate, TARGET_RATE)
        )

        return AudioFrame(converted, stamp, self.name)

    def maintain(self, now: float) -> None:
        """Restart a stalled or failed stream with exponential back-off."""

        if self.status is SourceStatus.STOPPED:
            return

        stalled = (
            self.status is SourceStatus.OK
            and now - self._last_data > self._stall
        )

        if stalled:
            self.status = SourceStatus.RECOVERING
            self._next_retry = now

        if self.status in (SourceStatus.RECOVERING, SourceStatus.FAILED):
            if now < self._next_retry:
                return

            old, self._stream = self._stream, None

            if old is not None:
                try:
                    old.close()
                except Exception:
                    pass

            self.restarts += 1

            if self.start():
                self._last_data = now

                return

            self.status = SourceStatus.RECOVERING
            self._next_retry = now + self._backoff
            self._backoff = min(self._backoff * 2.0, self._max_backoff)


@dataclass
class SourceQuality:
    """Running quality estimate for one source."""

    window: int = 150
    levels: deque = field(default_factory=lambda: deque(maxlen=150))
    clipped: deque = field(default_factory=lambda: deque(maxlen=150))
    frames: int = 0
    first_time: float | None = None
    last_time: float | None = None
    samples: int = 0
    first_size: int = 0

    def update(self, frame: AudioFrame) -> None:
        """Add a frame to the estimate."""

        self.levels.append(dbfs(rms(frame.samples)))
        self.clipped.append(clip_ratio(frame.samples))
        self.frames += 1
        self.samples += frame.samples.size

        if self.first_time is None:
            self.first_time = frame.timestamp
            self.first_size = frame.samples.size

        self.last_time = frame.timestamp

    @property
    def noise_floor_db(self) -> float:
        """Low percentile of recent levels."""

        if not self.levels:
            return -120.0

        return float(np.percentile(list(self.levels), 20))

    @property
    def speech_level_db(self) -> float:
        """High percentile of recent levels."""

        if not self.levels:
            return -120.0

        return float(np.percentile(list(self.levels), 90))

    @property
    def snr_db(self) -> float:
        """Spread between loud and quiet frames (crude SNR proxy)."""

        return self.speech_level_db - self.noise_floor_db

    @property
    def clip_fraction(self) -> float:
        """Mean fraction of clipped samples."""

        if not self.clipped:
            return 0.0

        return float(np.mean(self.clipped))

    def rate_error_ppm(self) -> float | None:
        """Delivered versus expected sample rate, in parts per million.

        This is how clock drift between independent devices shows up.  It
        needs a few seconds of data to be meaningful.
        """

        if (
            self.first_time is None
            or self.last_time is None
            or self.last_time - self.first_time < 2.0
        ):
            return None

        # Timestamps mark block arrival, so the first block is the origin.
        measured = (self.samples - self.first_size) / (
            self.last_time - self.first_time
        )

        return (measured / TARGET_RATE - 1.0) * 1e6

    def score(self) -> float:
        """Higher is better: SNR, penalised for clipping and quietness."""

        if not self.levels:
            return -999.0

        score = self.snr_db
        score -= 60.0 * self.clip_fraction
        score -= max(0.0, -55.0 - self.speech_level_db)

        return score


@dataclass(frozen=True)
class ManagerConfig:
    """Selection and fusion settings."""

    switch_margin_db: float = 3.0
    min_dwell_s: float = 3.0
    min_frames_for_score: int = 25
    fusion: str = "off"  # "off" or "sum"
    fusion_geometry_validated: bool = False


@dataclass(frozen=True)
class ManagerHealth:
    """Snapshot for diagnostics."""

    topology: MicTopology
    active: str | None
    statuses: dict[str, SourceStatus]
    scores: dict[str, float]
    rate_error_ppm: dict[str, float | None]
    fusion_active: bool


class MicrophoneManager:
    """Pick the best microphone and keep the stream alive."""

    def __init__(
        self,
        sources: list[AudioSource],
        *,
        topology: MicTopology,
        config: ManagerConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not sources:
            raise ValueError("At least one audio source is required.")

        self.sources = {s.name: s for s in sources}
        self.topology = topology
        self.config = config or ManagerConfig()
        self._clock = clock
        self.quality = {s.name: SourceQuality() for s in sources}
        self.active: str | None = None
        self._active_since = -1e9
        self._pending: dict[str, AudioFrame] = {}
        self._lock = threading.Lock()
        self.fusion_active = self._fusion_allowed()

    def _fusion_allowed(self) -> bool:
        cfg = self.config

        return (
            cfg.fusion == "sum"
            and cfg.fusion_geometry_validated
            and self.topology is MicTopology.STEREO
            and len(self.sources) == 2
        )

    def start(self) -> int:
        """Start every source; returns how many started."""

        started = 0

        for source in self.sources.values():
            if source.start():
                started += 1

        self._choose(self._clock(), force=True)

        return started

    def stop(self) -> None:
        """Stop all sources."""

        for source in self.sources.values():
            source.stop()

    def _healthy(self) -> list[str]:
        return [
            n for n, s in self.sources.items()
            if s.status is SourceStatus.OK
        ]

    def _choose(self, now: float, force: bool = False) -> None:
        healthy = self._healthy()

        if not healthy:
            self.active = None

            return

        if self.active not in healthy:
            self.active = max(
                healthy, key=lambda n: self.quality[n].score()
            )
            self._active_since = now

            return

        cfg = self.config

        if not force and now - self._active_since < cfg.min_dwell_s:
            return

        ready = [
            n for n in healthy
            if self.quality[n].frames >= cfg.min_frames_for_score
        ]

        if len(ready) < 2 or self.active not in ready:
            return

        best = max(ready, key=lambda n: self.quality[n].score())
        gain = self.quality[best].score() - self.quality[self.active].score()

        if best != self.active and gain >= cfg.switch_margin_db:
            self.active = best
            self._active_since = now

    def pump(self) -> AudioFrame | None:
        """Drain all sources once and return a frame from the active one."""

        now = self._clock()
        result: AudioFrame | None = None

        for name, source in self.sources.items():
            source.maintain(now)
            frame = source.read()

            if frame is None:
                continue

            self.quality[name].update(frame)
            self._pending[name] = frame

        self._choose(now)

        if self.active is not None:
            result = self._pending.pop(self.active, None)

        # Never let stale frames from inactive sources pile up.
        self._pending.clear()

        if result is not None and self.fusion_active:
            result = self._fuse(result)

        return result

    def _fuse(self, primary: AudioFrame) -> AudioFrame:
        # Only reachable for a validated stereo device; kept explicit.
        return primary

    def health(self) -> ManagerHealth:
        """Diagnostics snapshot."""

        return ManagerHealth(
            topology=self.topology,
            active=self.active,
            statuses={n: s.status for n, s in self.sources.items()},
            scores={n: q.score() for n, q in self.quality.items()},
            rate_error_ppm={
                n: q.rate_error_ppm() for n, q in self.quality.items()
            },
            fusion_active=self.fusion_active,
        )


def build_sources(
    selection: InputSelection,
    *,
    stream_factory: Callable | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> list[AudioSource]:
    """Create sources for a discovered microphone arrangement."""

    sources: list[AudioSource] = []

    for number, choice in enumerate(selection.inputs):
        sources.append(
            SoundDeviceSource(
                choice,
                name=f"mic{number}:{choice.label}",
                stream_factory=stream_factory,
                clock=clock,
            )
        )

    return sources


class FrameChunker:
    """Re-block audio into exact fixed-size frames for the VAD."""

    def __init__(self, size: int = 320) -> None:
        self.size = size
        self._carry = np.zeros(0, np.int16)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        """Add samples; return every complete frame now available."""

        if self._carry.size:
            samples = np.concatenate([self._carry, samples])

        count = samples.size // self.size
        frames = [
            samples[i * self.size:(i + 1) * self.size].copy()
            for i in range(count)
        ]
        self._carry = samples[count * self.size:].copy()

        return frames

    def clear(self) -> None:
        """Drop buffered audio."""

        self._carry = np.zeros(0, np.int16)

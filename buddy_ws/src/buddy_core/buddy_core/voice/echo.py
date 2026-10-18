"""Keep Buddy from hearing itself, and detect real interruptions.

Strategy, in order of how much it relies on extra machinery:

1. **Guard**: while Buddy speaks, and for a short tail afterwards, the
   microphone stream is not given to the recogniser.
2. **Leakage estimate**: the guard learns how loud Buddy's own voice
   appears at the microphone (ratio of mic level to playback level) while
   nobody else is talking.
3. **Barge-in**: if the microphone is clearly louder than that learned
   leakage for several consecutive frames, someone is talking over Buddy,
   and the guard reports an interruption so the speaker can be cut off.
4. **AEC** (optional): an ``EchoCanceller`` adapter can be plugged in.
   None is bundled as verified; ``SpeexAec`` is a thin, untested-on-device
   wrapper.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from buddy_core.voice.dsp import dbfs, rms


class EchoCanceller(Protocol):
    """Removes the known playback signal from a microphone frame."""

    def process(self, mic: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Return the cleaned microphone frame."""


class SpeexAec:
    """Adapter for the optional ``speexdsp`` package.

    UNVERIFIED: written against the documented API but not run on the Pi.
    Falls back to returning the input untouched if the package is absent.
    """

    def __init__(self, frame: int = 320, tail: int = 3200,
                 rate: int = 16000) -> None:
        self._aec = None

        try:
            from speexdsp import EchoCanceller as _Aec

            self._aec = _Aec.create(frame, tail, rate)
        except Exception:
            self._aec = None

    @property
    def available(self) -> bool:
        """True when the native canceller loaded."""

        return self._aec is not None

    def process(self, mic: np.ndarray, reference: np.ndarray) -> np.ndarray:
        if self._aec is None:
            return mic

        out = self._aec.process(mic.tobytes(), reference.tobytes())

        return np.frombuffer(out, dtype=np.int16)


@dataclass(frozen=True)
class EchoConfig:
    """Tunable guard settings."""

    guard_after_s: float = 0.35
    barge_in_margin_db: float = 9.0
    barge_in_frames: int = 6
    default_leak_db: float = -12.0
    min_playback_dbfs: float = -60.0
    learn_rate: float = 0.05
    barge_in_enabled: bool = True
    # Mic floor the guard never reports as an interruption.
    min_barge_dbfs: float = -45.0


class EchoGuard:
    """Tracks when Buddy is speaking and what to ignore."""

    def __init__(
        self,
        config: EchoConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        canceller: EchoCanceller | None = None,
    ) -> None:
        self.config = config or EchoConfig()
        self._clock = clock
        self.canceller = canceller
        self._speaking = False
        self._guard_until = 0.0
        self._leak_db = self.config.default_leak_db
        self._run = 0
        self._playback_db = -120.0
        self.barge_in_count = 0

    @property
    def leakage_db(self) -> float:
        """Learned mic-minus-playback level in dB."""

        return self._leak_db

    def playback_started(self) -> None:
        """Call when the speaker starts."""

        self._speaking = True
        self._run = 0

    def playback_stopped(self) -> None:
        """Call when the speaker stops, for any reason."""

        self._speaking = False
        self._guard_until = self._clock() + self.config.guard_after_s
        self._run = 0

    def playback_level(self, frame: np.ndarray) -> None:
        """Feed the level of the audio being played (per frame)."""

        self._playback_db = dbfs(rms(frame))

    def playback_level_db(self, level_db: float) -> None:
        """Feed the playback level when it is already known in dB."""

        self._playback_db = level_db

    @property
    def blocked(self) -> bool:
        """True when the recogniser must not receive microphone audio."""

        return self._speaking or self._clock() < self._guard_until

    def clean(self, mic: np.ndarray, reference: np.ndarray | None = None
              ) -> np.ndarray:
        """Apply AEC if one is configured."""

        if self.canceller is None or reference is None:
            return mic

        return self.canceller.process(mic, reference)

    def observe(self, mic: np.ndarray) -> bool:
        """Feed a mic frame while speaking.  Returns True on barge-in.

        Only meaningful while ``speaking``; otherwise returns False.
        """

        if not self._speaking:
            return False

        cfg = self.config
        mic_db = dbfs(rms(mic))

        if self._playback_db < cfg.min_playback_dbfs:
            return False

        expected = self._playback_db + self._leak_db
        loud = (
            mic_db > expected + cfg.barge_in_margin_db
            and mic_db > cfg.min_barge_dbfs
        )

        if loud and cfg.barge_in_enabled:
            self._run += 1

            if self._run >= cfg.barge_in_frames:
                self._run = 0
                self.barge_in_count += 1

                return True

            return False

        self._run = 0
        observed = mic_db - self._playback_db
        self._leak_db += cfg.learn_rate * (observed - self._leak_db)

        return False

"""Voice activity detection with endpointing.

The default detector is an adaptive-energy VAD: it tracks the background
noise floor and flags speech when frame energy rises clearly above it, with
hysteresis, a minimum speech duration and a hangover so that short pauses do
not end an utterance.  It is deliberately simple so it runs anywhere.

If ``webrtcvad`` is installed it can additionally vote on each frame
(``use_webrtc=True``), which reduces false positives from steady noise.

The energy detector is not a trained model.  Its thresholds are
configuration, not validated numbers; tune them on the real microphones.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from buddy_core.voice.dsp import dbfs, rms


class VadState(str, Enum):
    """Detector state."""

    SILENCE = "silence"
    SPEECH = "speech"


@dataclass(frozen=True)
class VadConfig:
    """Detector parameters.  All times are in milliseconds."""

    sample_rate: int = 16000
    frame_ms: int = 20
    start_margin_db: float = 9.0
    stop_margin_db: float = 5.0
    min_noise_floor_dbfs: float = -75.0
    min_speech_dbfs: float = -50.0
    min_speech_ms: int = 120
    hangover_ms: int = 450
    max_utterance_ms: int = 20000
    noise_floor_window: int = 100
    # Frames (10 s at 20 ms) used to follow a sudden, sustained rise in
    # background noise that the fast estimate would mistake for speech.
    slow_floor_window: int = 500
    use_webrtc: bool = False
    webrtc_aggressiveness: int = 2

    @property
    def frame_samples(self) -> int:
        """Samples per analysis frame."""

        return self.sample_rate * self.frame_ms // 1000


@dataclass(frozen=True)
class VadEvent:
    """Result of processing one frame."""

    state: VadState
    is_speech_frame: bool
    speech_started: bool
    speech_ended: bool
    level_dbfs: float
    noise_floor_dbfs: float
    utterance_ms: int
    silence_ms: int


class VoiceActivityDetector:
    """Frame-by-frame speech detector with endpointing."""

    def __init__(self, config: VadConfig | None = None) -> None:
        self.config = config or VadConfig()
        self.state = VadState.SILENCE
        self._noise: deque[float] = deque(
            maxlen=self.config.noise_floor_window
        )
        self._noise_floor = self.config.min_noise_floor_dbfs
        self._fast_floor = self._noise_floor
        self._all: deque[float] = deque(
            maxlen=self.config.slow_floor_window
        )
        self._speech_run = 0
        self._silence_run = 0
        self._utterance_frames = 0
        self._webrtc = None

        if self.config.use_webrtc:
            try:
                import webrtcvad

                self._webrtc = webrtcvad.Vad(
                    self.config.webrtc_aggressiveness
                )
            except ImportError:
                self._webrtc = None

    @property
    def noise_floor_dbfs(self) -> float:
        """Current estimate of the background level."""

        return self._noise_floor

    def reset(self) -> None:
        """Forget the current utterance (not the noise estimate)."""

        self.state = VadState.SILENCE
        self._speech_run = 0
        self._silence_run = 0
        self._utterance_frames = 0

    def _update_noise(self, level: float) -> None:
        self._noise.append(level)
        ordered = sorted(self._noise)
        # A low percentile follows the floor even if speech is frequent.
        index = max(0, int(0.2 * (len(ordered) - 1)))
        self._fast_floor = max(
            self.config.min_noise_floor_dbfs, ordered[index]
        )

    def _refresh_floor(self) -> None:
        """Combine the fast estimate with the slow, all-frames one.

        Speech has pauses, so its 10th percentile stays near the real
        floor.  Constant loud noise has no pauses, so after one full
        window the slow estimate lifts the floor and the VAD stops
        calling it speech.  A monologue with no pauses at all for the
        whole window would also lift it; that is accepted.
        """

        floor = self._fast_floor

        if len(self._all) == self._all.maxlen:
            slow = sorted(self._all)[int(0.1 * (len(self._all) - 1))]
            floor = max(floor, slow)

        self._noise_floor = max(self.config.min_noise_floor_dbfs, floor)

    def _webrtc_vote(self, frame: np.ndarray) -> bool:
        if self._webrtc is None:
            return True

        try:
            return bool(
                self._webrtc.is_speech(
                    frame.astype(np.int16).tobytes(),
                    self.config.sample_rate,
                )
            )
        except Exception:
            return True

    def process(self, frame: np.ndarray) -> VadEvent:
        """Process one frame of int16 audio at ``sample_rate``."""

        cfg = self.config
        level = dbfs(rms(frame))
        floor = self._noise_floor
        in_speech = self.state is VadState.SPEECH
        margin = cfg.stop_margin_db if in_speech else cfg.start_margin_db
        loud_enough = level >= max(floor + margin, cfg.min_speech_dbfs)
        is_speech = loud_enough and self._webrtc_vote(frame)

        self._all.append(level)

        if not is_speech and not in_speech:
            self._update_noise(level)

        self._refresh_floor()

        started = False
        ended = False

        if is_speech:
            self._speech_run += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            self._speech_run = 0

        frame_ms = cfg.frame_ms

        if self.state is VadState.SILENCE:
            if self._speech_run * frame_ms >= cfg.min_speech_ms:
                self.state = VadState.SPEECH
                self._utterance_frames = self._speech_run
                started = True
        else:
            self._utterance_frames += 1
            too_long = (
                self._utterance_frames * frame_ms >= cfg.max_utterance_ms
            )

            if self._silence_run * frame_ms >= cfg.hangover_ms or too_long:
                self.state = VadState.SILENCE
                ended = True

        return VadEvent(
            state=self.state,
            is_speech_frame=is_speech,
            speech_started=started,
            speech_ended=ended,
            level_dbfs=level,
            noise_floor_dbfs=self._noise_floor,
            utterance_ms=self._utterance_frames * frame_ms
            if self.state is VadState.SPEECH or ended
            else 0,
            silence_ms=self._silence_run * frame_ms,
        )

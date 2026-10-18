"""Speech synthesis and playback adapters.

* ``PiperSynthesizer`` wraps ``piper.PiperVoice`` (UNVERIFIED off-Pi).
* ``AplayPlayback`` plays audio without blocking, so cancel is immediate.
* ``FakeSynthesizer`` produces a deterministic tone for tests.

Piper has no native emotion control.  Expressiveness comes from the text,
the pauses, the speaking rate and the volume chosen by the response layer,
not from the model.
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
import wave
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SpeechStyle:
    """Honest, model-independent delivery controls."""

    length_scale: float = 1.0   # >1 slower
    pause_scale: float = 1.0    # scales inserted pauses
    volume: float = 0.6


class Synthesizer(Protocol):
    """Turns text into PCM."""

    def synthesize(
        self, text: str, style: SpeechStyle
    ) -> tuple[np.ndarray, int]:
        """Return (int16 samples, sample rate)."""


class PiperSynthesizer:
    """Piper TTS.  The model path is configuration, never hardcoded."""

    def __init__(self, model_path: str, voice=None) -> None:
        self._path = model_path
        self._voice = voice

    def _load(self):
        if self._voice is None:
            from piper import PiperVoice

            self._voice = PiperVoice.load(self._path)

        return self._voice

    def synthesize(self, text: str, style: SpeechStyle):
        voice = self._load()
        buf = io.BytesIO()

        with wave.open(buf, "wb") as wav:
            try:
                from piper import SynthesisConfig

                voice.synthesize_wav(
                    text, wav,
                    syn_config=SynthesisConfig(
                        length_scale=style.length_scale
                    ),
                )
            except ImportError:
                voice.synthesize_wav(text, wav)

        buf.seek(0)

        with wave.open(buf, "rb") as wav:
            rate = wav.getframerate()
            data = wav.readframes(wav.getnframes())

        return np.frombuffer(data, dtype=np.int16).copy(), rate


class FakeSynthesizer:
    """Deterministic stand-in: duration scales with text length."""

    def __init__(self, rate: int = 22050, seconds_per_char: float = 0.05):
        self.rate = rate
        self.spc = seconds_per_char
        self.calls: list[tuple[str, SpeechStyle]] = []
        self.fail = False

    def synthesize(self, text: str, style: SpeechStyle):
        if self.fail:
            raise RuntimeError("synthesis failed")

        self.calls.append((text, style))
        n = int(self.rate * self.spc * len(text) * style.length_scale)
        t = np.arange(max(n, 1)) / self.rate

        return (0.3 * 32767 * np.sin(2 * np.pi * 200 * t)).astype(
            np.int16), self.rate


class AplayPlayback:
    """Non-blocking playback through ``aplay``.

    ``device`` comes from configuration or discovery; there is no
    hardcoded card number.  Volume is applied to the samples because
    ``aplay`` has no gain control.
    """

    def __init__(
        self,
        device: str | None = None,
        popen: Callable = subprocess.Popen,
    ) -> None:
        self.device = device
        self._popen = popen
        self._proc = None
        self._path: str | None = None

    def start(self, samples: np.ndarray, rate: int, gain: float) -> None:
        self.stop()
        scaled = np.clip(
            samples.astype(np.float32) * gain, -32768, 32767
        ).astype(np.int16)
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="buddy_tts_")
        os.close(fd)

        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(scaled.tobytes())

        cmd = ["aplay", "-q"]

        if self.device:
            cmd += ["-D", self.device]

        cmd.append(path)
        self._path = path
        self._proc = self._popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def is_playing(self) -> bool:
        if self._proc is None:
            return False

        if self._proc.poll() is None:
            return True

        self._cleanup()

        return False

    def stop(self) -> None:
        proc, self._proc = self._proc, None

        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=0.5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        self._cleanup()

    def _cleanup(self) -> None:
        path, self._path = self._path, None

        if path:
            try:
                os.remove(path)
            except OSError:
                pass

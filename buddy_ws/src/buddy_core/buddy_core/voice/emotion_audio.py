"""Voice-based arousal estimation from simple acoustic features.

HONEST SCOPE.  This module measures how a speaker's voice differs from
*their own recent baseline* in loudness, pitch, pitch variation and speaking
rate.  That is a crude indicator of arousal (calm versus activated), not of
specific emotions, and it says nothing reliable about valence (pleasant
versus unpleasant).  It has not been evaluated on real speakers: no accuracy
figure is claimed anywhere, and the confidence it reports is capped well
below certainty.  ``evaluate`` exists so the team can measure it on
consented, labelled recordings; until that is run, treat every output as a
hint that the user's words can override.

Audio is processed in memory and discarded.  Nothing here writes audio to
disk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from buddy_core.voice.dsp import dbfs, rms, to_float

RATE = 16000
MIN_DURATION_S = 0.5
MIN_LEVEL_DBFS = -55.0
MAX_AUDIO_CONFIDENCE = 0.6


@dataclass(frozen=True)
class AudioFeatures:
    """Summary of one utterance."""

    duration_s: float
    rms_db: float
    pitch_median_hz: float | None
    pitch_var_semitones: float | None
    voiced_ratio: float
    rate_per_s: float | None
    centroid_hz: float
    zcr: float
    mfcc_mean: tuple[float, ...] = field(default_factory=tuple)


def _frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    if x.size < size:
        return np.zeros((0, size), np.float32)

    count = 1 + (x.size - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(count)[:, None]

    return x[idx]


def track_pitch(x: np.ndarray, rate: int = RATE
                ) -> tuple[np.ndarray, np.ndarray]:
    """Return (pitch per frame in Hz or 0, voiced mask).

    Autocorrelation with a normalised-peak voicing decision.
    """

    size = int(0.04 * rate)
    hop = int(0.02 * rate)
    frames = _frames(x, size, hop)
    lo = int(rate / 400)
    hi = int(rate / 70)
    pitch = np.zeros(frames.shape[0])
    voiced = np.zeros(frames.shape[0], dtype=bool)

    for i, frame in enumerate(frames):
        frame = frame - frame.mean()
        energy = float(np.dot(frame, frame))

        if energy <= 1e-8 or dbfs(math.sqrt(energy / size)) < -50.0:
            continue

        spec = np.fft.rfft(frame, 2 * size)
        ac = np.fft.irfft(spec * np.conj(spec))[:size]
        ac = ac / ac[0]
        seg = ac[lo:hi]
        k = int(np.argmax(seg))

        if seg[k] < 0.45:
            continue

        lag = lo + k

        if 0 < lag < size - 1:
            a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
            denom = a - 2 * b + c

            if abs(denom) > 1e-9:
                lag = lag + 0.5 * (a - c) / denom

        pitch[i] = rate / lag
        voiced[i] = True

    return pitch, voiced


def _mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _inv_mel(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mfcc(x: np.ndarray, rate: int = RATE, n_mfcc: int = 13,
         n_mels: int = 26) -> np.ndarray:
    """MFCC matrix (frames x n_mfcc), numpy only."""

    size, hop, nfft = int(0.025 * rate), int(0.01 * rate), 512
    frames = _frames(x, size, hop)

    if frames.shape[0] == 0:
        return np.zeros((0, n_mfcc))

    frames = frames * np.hamming(size)
    power = np.abs(np.fft.rfft(frames, nfft)) ** 2 / nfft
    mels = np.linspace(_mel(100.0), _mel(min(7600.0, rate / 2 - 100)),
                       n_mels + 2)
    bins = np.floor((nfft + 1) * _inv_mel(mels) / rate).astype(int)
    bank = np.zeros((n_mels, nfft // 2 + 1))

    for m in range(1, n_mels + 1):
        left, centre, right = bins[m - 1], bins[m], bins[m + 1]

        for k in range(left, centre):
            bank[m - 1, k] = (k - left) / max(centre - left, 1)

        for k in range(centre, right):
            bank[m - 1, k] = (right - k) / max(right - centre, 1)

    logmel = np.log(power @ bank.T + 1e-10)
    n = np.arange(n_mels)
    dct = np.cos(np.pi / n_mels * (n + 0.5)[None, :]
                 * np.arange(n_mfcc)[:, None])

    return logmel @ dct.T


def extract_features(samples: np.ndarray, rate: int = RATE
                     ) -> AudioFeatures | None:
    """Features for one utterance, or None if it is unusable."""

    if rate != RATE:
        return None

    x = to_float(samples) if samples.dtype == np.int16 else \
        samples.astype(np.float32)
    duration = x.size / rate

    if duration < MIN_DURATION_S:
        return None

    level = dbfs(rms(x))

    if level < MIN_LEVEL_DBFS:
        return None

    pitch, voiced = track_pitch(x, rate)
    voiced_ratio = float(voiced.mean()) if voiced.size else 0.0
    values = pitch[voiced]
    median = float(np.median(values)) if values.size >= 5 else None
    var = None

    if median:
        semis = 12.0 * np.log2(values / median)
        var = float(np.std(semis))

    hop = int(0.02 * rate)
    env = np.array([rms(f) for f in _frames(x, hop, hop)])
    rate_per_s = None

    if env.size > 10 and env.max() > 0:
        smooth = np.convolve(env, np.ones(3) / 3, mode="same")
        thr = 0.5 * smooth.max()
        peaks, last = 0, -10

        for i in range(1, smooth.size - 1):
            if (smooth[i] > thr and smooth[i] >= smooth[i - 1]
                    and smooth[i] > smooth[i + 1] and i - last >= 6):
                peaks += 1
                last = i

        rate_per_s = peaks / duration

    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, 1.0 / rate)
    centroid = float((spec * freqs).sum() / max(spec.sum(), 1e-9))
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    coeffs = mfcc(x, rate)
    mean = tuple(float(v) for v in coeffs.mean(axis=0)) if coeffs.size \
        else ()

    return AudioFeatures(duration, level, median, var, voiced_ratio,
                         rate_per_s, centroid, zcr, mean)


_FIELDS = {
    "rms_db": (0.35, 3.0),
    "pitch_median_hz": (0.30, 15.0),
    "pitch_var_semitones": (0.20, 1.0),
    "rate_per_s": (0.15, 0.7),
}


class SpeakerBaseline:
    """Running mean and spread of one speaker's own features."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.count = 0
        self.mean: dict[str, float] = {}
        self.var: dict[str, float] = {}

    def update(self, f: AudioFeatures) -> None:
        """Fold an utterance into the baseline."""

        self.count += 1
        a = max(1.0 / self.count, self.alpha)

        for name in _FIELDS:
            value = getattr(f, name)

            if value is None:
                continue

            if name not in self.mean:
                self.mean[name] = value
                self.var[name] = 0.0
                continue

            diff = value - self.mean[name]
            self.mean[name] += a * diff
            self.var[name] = (1 - a) * (self.var[name] + a * diff * diff)

    def reset(self) -> None:
        """Forget the baseline."""

        self.__init__(self.alpha)

    def z(self, name: str, value: float) -> float | None:
        """Standardised distance from this speaker's usual."""

        if name not in self.mean:
            return None

        floor = _FIELDS[name][1]
        std = max(math.sqrt(self.var[name]), floor)

        return float(np.clip((value - self.mean[name]) / std, -3.0, 3.0))


@dataclass(frozen=True)
class AudioEmotion:
    """Arousal hint.  Never a diagnosis."""

    arousal: float | None
    label: str            # "low", "normal", "high" or "unknown"
    confidence: float
    cues: int
    reason: str = ""


class AudioEmotionEstimator:
    """Compare an utterance with the speaker's baseline."""

    def __init__(self, baseline: SpeakerBaseline | None = None,
                 min_baseline: int = 5) -> None:
        self.baseline = baseline or SpeakerBaseline()
        self.min_baseline = min_baseline

    def estimate(self, f: AudioFeatures | None, *, learn: bool = True
                 ) -> AudioEmotion:
        """Estimate arousal for ``f`` and (optionally) learn from it."""

        if f is None:
            return AudioEmotion(None, "unknown", 0.0, 0, "unusable audio")

        base = self.baseline
        zs = []

        for name, (weight, _) in _FIELDS.items():
            value = getattr(f, name)
            z = None if value is None else base.z(name, value)

            if z is not None:
                zs.append((weight, z))

        if learn:
            base.update(f)

        if base.count - (1 if learn else 0) < self.min_baseline or not zs:
            return AudioEmotion(None, "unknown", 0.0, len(zs),
                                "not enough baseline yet")

        total = sum(w for w, _ in zs)
        arousal = sum(w * z for w, z in zs) / total / 3.0
        arousal = float(np.clip(arousal * 2.0, -1.0, 1.0))
        coverage = len(zs) / len(_FIELDS)
        known = min(1.0, (base.count - 1) / 20.0)
        voicing = min(1.0, f.voiced_ratio / 0.5)
        length = min(1.0, f.duration_s / 2.0)
        confidence = MAX_AUDIO_CONFIDENCE * coverage * known * voicing \
            * length

        if arousal > 0.3:
            label = "high"
        elif arousal < -0.3:
            label = "low"
        else:
            label = "normal"

        return AudioEmotion(arousal, label, float(confidence), len(zs),
                            "relative to this speaker's baseline")


def evaluate(estimator_factory, clips: list[tuple[np.ndarray, str]]
             ) -> dict:
    """Measure arousal-label accuracy on consented labelled clips.

    ``clips`` is ``[(int16 samples, "low"|"normal"|"high"), ...]`` in
    recording order for ONE speaker.  Returns counts and accuracy over
    the clips the estimator committed to.  Run this before ever quoting
    a number.
    """

    est = estimator_factory()
    correct = committed = unknown = 0

    for samples, truth in clips:
        result = est.estimate(extract_features(samples))

        if result.label == "unknown":
            unknown += 1
            continue

        committed += 1
        correct += result.label == truth

    return {
        "clips": len(clips),
        "unknown": unknown,
        "committed": committed,
        "correct": correct,
        "accuracy": (correct / committed) if committed else None,
    }

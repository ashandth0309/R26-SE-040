"""Small signal-processing helpers shared by the voice modules.

Only numpy is required.  ``scipy`` is used for resampling when present.
"""

from __future__ import annotations

import math

import numpy as np

INT16_MAX = 32767.0


def to_float(samples: np.ndarray) -> np.ndarray:
    """int16 (or float) samples as float32 in [-1, 1]."""

    if samples.dtype == np.int16:
        return samples.astype(np.float32) / 32768.0

    return samples.astype(np.float32)


def to_int16(samples: np.ndarray) -> np.ndarray:
    """float samples in [-1, 1] as clipped int16."""

    return np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)


def rms(samples: np.ndarray) -> float:
    """Root mean square in [0, 1] full scale."""

    if samples.size == 0:
        return 0.0

    x = to_float(samples)

    return float(np.sqrt(np.mean(x * x)))


def dbfs(value: float, floor: float = -120.0) -> float:
    """Convert a linear amplitude to dB full scale."""

    if value <= 0.0:
        return floor

    return max(floor, 20.0 * math.log10(value))


def clip_ratio(samples: np.ndarray) -> float:
    """Fraction of samples at or beyond 99 percent of full scale."""

    if samples.size == 0:
        return 0.0

    limit = 0.99 * INT16_MAX
    return float(np.mean(np.abs(samples.astype(np.int32)) >= limit))


def _fir_lowpass(cutoff: float, taps: int) -> np.ndarray:
    """Windowed-sinc low-pass; ``cutoff`` as a fraction of Nyquist."""

    n = np.arange(taps) - (taps - 1) / 2.0
    h = np.sinc(cutoff * n) * cutoff
    h *= np.hamming(taps)

    return (h / h.sum()).astype(np.float32)


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample float32 audio with proper anti-alias filtering.

    The previous voice node simply kept every third sample of 48 kHz audio,
    which folds energy above 8 kHz back into the speech band.  This
    function low-pass filters first.
    """

    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32)

    try:
        from scipy.signal import resample_poly

        divisor = math.gcd(src_rate, dst_rate)

        return resample_poly(
            samples.astype(np.float32),
            dst_rate // divisor,
            src_rate // divisor,
        ).astype(np.float32)
    except ImportError:
        pass

    x = samples.astype(np.float32)

    if dst_rate < src_rate:
        cutoff = 0.9 * dst_rate / src_rate
        x = np.convolve(x, _fir_lowpass(cutoff, 63), mode="same")

    positions = np.arange(0, x.size, src_rate / dst_rate)

    return np.interp(positions, np.arange(x.size), x).astype(np.float32)


def downmix(block: np.ndarray, channel: int | None) -> np.ndarray:
    """Pick one channel of an (n, channels) block or average them.

    Averaging is only valid for channels that share a clock, which is the
    case inside one device.  Never use it across independent devices.
    """

    if block.ndim == 1:
        return block

    if channel is None:
        return block.astype(np.float32).mean(axis=1)

    return block[:, min(channel, block.shape[1] - 1)]


def envelope(samples: np.ndarray, frame: int) -> np.ndarray:
    """RMS envelope with ``frame`` samples per value."""

    n = samples.size // frame

    if n == 0:
        return np.zeros(0, dtype=np.float32)

    x = to_float(samples[: n * frame]).reshape(n, frame)

    return np.sqrt(np.mean(x * x, axis=1))

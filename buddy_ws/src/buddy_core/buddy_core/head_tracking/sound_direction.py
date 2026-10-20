"""Two-microphone sound direction (TDOA via GCC-PHAT).

WHAT THIS CAN AND CANNOT DO.
* Two microphones give one time difference, hence one angle measured from
  the line perpendicular to the microphone pair.  Front and back cannot be
  told apart (``front_back_ambiguous`` is always True); the camera has to
  settle that.
* The timing is only meaningful if both channels are sampled by ONE clock
  (one stereo device).  Two separate USB microphones have independent
  clocks; their cross-correlation delay is meaningless.  The caller must
  state ``synchronized`` explicitly, and ``False`` is always rejected.
* Resolution is limited by the sample rate and spacing: one raw sample of
  delay equals ``c / (fs * d)`` radians near broadside (about 8 degrees
  at 16 kHz and 15 cm).  ``resolution_deg`` reports it.  The accuracy on
  the real robot is NOT known; it must be measured
  (``docs/head_tracking_physical_validation.md``).
* Thresholds are starting values checked on synthetic audio only.

Azimuth sign: positive = source on the robot's RIGHT, negative = LEFT.
Channel 0 is the left microphone unless ``swap_channels`` is set.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from buddy_core.head_tracking.config import SoundConfig


class Reason:
    OK = "ok"
    UNSYNCHRONIZED = "microphones_not_synchronized"
    GEOMETRY_UNSET = "mic_spacing_not_configured"
    BAD_INPUT = "invalid_audio"
    TOO_SHORT = "window_too_short"
    SILENT = "signal_too_quiet"
    CLIPPING = "signal_clipping"
    LOW_SNR = "low_snr"
    LOW_PEAK = "weak_correlation_peak"
    AMBIGUOUS = "ambiguous_multiple_peaks"
    OUT_OF_RANGE = "delay_outside_microphone_geometry"
    ENDFIRE = "direction_too_close_to_microphone_axis"
    LOW_CONFIDENCE = "low_confidence"
    DISABLED = "sound_localization_disabled"
    SPEAKING = "buddy_speaking"
    INCONSISTENT = "estimates_disagree"


@dataclass(frozen=True)
class SoundDirectionResult:
    ok: bool
    reason: str
    t: float = 0.0
    azimuth_deg: float | None = None
    delay_s: float | None = None
    confidence: float = 0.0
    peak: float = 0.0
    peak_ratio: float = 0.0
    resolution_deg: float | None = None
    front_back_ambiguous: bool = True
    details: dict = field(default_factory=dict)


def gcc_phat(x: np.ndarray, y: np.ndarray, fs: float, max_tau_s: float,
             band_hz: tuple[float, float], interp: int = 8,
             background_s: float = 0.008):
    """Delay of ``y`` relative to ``x`` (positive = y later).

    Returns ``(tau_s, peak, psr, ratio)``:

    * ``peak``  - PHAT correlation at the best lag inside the geometric
      search window (about 1 for a clean coherent source, small for
      noise);
    * ``psr``   - peak over the RMS of the correlation far from the peak
      (``background_s`` either side), a peak-to-background ratio;
    * ``ratio`` - peak over the best value inside the search window but
      outside the main lobe (about 1 when two sources compete; ``inf``
      when the window holds only the main lobe).
    """

    n = len(x) + len(y)
    nfft = 1 << (n - 1).bit_length()
    X = np.fft.rfft(x, nfft)
    Y = np.fft.rfft(y, nfft)
    R = Y * np.conj(X)
    mag = np.abs(R)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    keep = (freqs >= band_hz[0]) & (freqs <= band_hz[1]) & (mag > 1e-12)
    R = np.where(keep, R / np.where(mag > 1e-12, mag, 1.0), 0.0)
    nbins = max(int(np.count_nonzero(keep)), 1)
    size = nfft * interp
    cc = np.fft.irfft(R, size) * size / (2.0 * nbins)

    wide = min(int(math.ceil(background_s * fs * interp)), size // 2 - 1)
    search = min(int(math.ceil(max_tau_s * fs * interp)), wide)
    lags = np.arange(-wide, wide + 1)
    full = cc[lags % size]                       # lag k <-> full[k + wide]
    win = full[wide - search:wide + search + 1]
    i = int(np.argmax(win))
    peak = float(win[i])
    centre = wide - search + i                   # index inside ``full``

    lobe = int(1.5 / (band_hz[1] - band_hz[0]) * fs * interp)
    far = np.ones(len(full), bool)
    far[max(0, centre - lobe):centre + lobe + 1] = False
    bg = float(np.sqrt(np.mean(full[far] ** 2))) if far.any() else 0.0
    psr = peak / bg if bg > 1e-12 else float("inf")

    wmask = np.ones(len(win), bool)
    wmask[max(0, i - lobe):i + lobe + 1] = False
    second = float(np.max(win[wmask])) if wmask.any() else 0.0
    ratio = peak / second if second > 1e-9 else float("inf")

    if 0 < i < len(win) - 1:
        a, b, c = win[i - 1], win[i], win[i + 1]
        den = a - 2 * b + c
        shift = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
    else:
        shift = 0.0
    tau = (i - search + shift) / (fs * interp)
    return float(tau), peak, float(psr), float(ratio)


class SoundDirectionEstimator:
    """Validate a stereo window and estimate azimuth, or say why not."""

    def __init__(self, cfg: SoundConfig | None = None) -> None:
        self.cfg = cfg or SoundConfig()

    def estimate(
        self,
        ch0: np.ndarray,
        ch1: np.ndarray,
        fs: float,
        synchronized: bool,
        t: float = 0.0,
        noise_rms: float | None = None,
    ) -> SoundDirectionResult:
        c = self.cfg

        def fail(reason: str, **kw) -> SoundDirectionResult:
            return SoundDirectionResult(False, reason, t=t, **kw)

        if not c.enabled:
            return fail(Reason.DISABLED)
        if not synchronized:
            return fail(Reason.UNSYNCHRONIZED)
        if c.mic_spacing_m is None:
            return fail(Reason.GEOMETRY_UNSET)
        try:
            a = np.asarray(ch0, dtype=np.float64)
            b = np.asarray(ch1, dtype=np.float64)
        except (TypeError, ValueError):
            return fail(Reason.BAD_INPUT)
        if (a.ndim != 1 or b.ndim != 1 or len(a) != len(b)
                or not fs or fs <= 0 or not math.isfinite(fs)
                or not (np.isfinite(a).all() and np.isfinite(b).all())):
            return fail(Reason.BAD_INPUT)
        if len(a) < int(c.min_window_s * fs):
            return fail(Reason.TOO_SHORT)
        if c.swap_channels:
            a, b = b, a

        clip = max(float(np.mean(np.abs(a) >= 0.999)),
                   float(np.mean(np.abs(b) >= 0.999)))
        if clip > c.max_clip_fraction:
            return fail(Reason.CLIPPING)
        a = a - a.mean()
        b = b - b.mean()
        rms = min(float(np.sqrt(np.mean(a * a))),
                  float(np.sqrt(np.mean(b * b))))
        if rms < c.min_rms:
            return fail(Reason.SILENT)
        if noise_rms is not None and noise_rms > 0 and \
                rms / noise_rms < c.min_snr_ratio:
            return fail(Reason.LOW_SNR)

        d, v = c.mic_spacing_m, c.speed_of_sound_mps
        max_tau = d / v
        search = max_tau * (1.0 + c.geometry_tolerance) + abs(
            c.delay_offset_s)
        tau, peak, psr, ratio = gcc_phat(
            a, b, fs, search, (c.band_hz[0], c.band_hz[1]), c.interp)
        tau -= c.delay_offset_s           # remove calibrated offset
        base = dict(delay_s=tau, peak=peak, peak_ratio=min(ratio, 99.0),
                    details={"psr": round(min(psr, 999.0), 2)})

        if peak < c.min_peak or psr < c.min_psr:
            return fail(Reason.LOW_PEAK, **base)
        if ratio < c.min_peak_ratio:
            return fail(Reason.AMBIGUOUS, **base)
        if abs(tau) > max_tau * (1.0 + c.geometry_tolerance):
            return fail(Reason.OUT_OF_RANGE, **base)

        s = max(-1.0, min(1.0, tau * v / d))
        # y later than x  => source nearer to x (left)  => negative azimuth
        az = -math.degrees(math.asin(s))
        if abs(az) > c.max_abs_azimuth_deg:
            return fail(Reason.ENDFIRE, azimuth_deg=az, **base)

        conf = min(1.0, peak / c.good_peak) * min(
            1.0, psr / c.good_psr)
        if math.isfinite(ratio):
            conf *= min(1.0, (ratio - 1.0) / (c.good_ratio - 1.0))
        res = math.degrees(v / (fs * d * max(math.cos(math.radians(az)),
                                             0.2)))
        if conf < c.min_confidence:
            return fail(Reason.LOW_CONFIDENCE, azimuth_deg=az,
                        confidence=conf, resolution_deg=res, **base)
        return SoundDirectionResult(
            True, Reason.OK, t=t, azimuth_deg=az, confidence=conf,
            resolution_deg=res, **base)


class SoundDirectionFilter:
    """Accept a direction only when several recent estimates agree.

    Single windows are easily fooled by noise, reverberation or speech
    from elsewhere; requiring ``min_agree`` estimates within ``agree_deg``
    inside ``filter_window_s`` rejects most of that.
    """

    def __init__(self, cfg: SoundConfig | None = None) -> None:
        self.cfg = cfg or SoundConfig()
        self._recent: deque[SoundDirectionResult] = deque()

    def reset(self) -> None:
        self._recent.clear()

    def add(self, r: SoundDirectionResult) -> SoundDirectionResult:
        c = self.cfg
        if not r.ok:
            return r
        self._recent.append(r)
        while self._recent and r.t - self._recent[0].t > c.filter_window_s:
            self._recent.popleft()
        vals = [x.azimuth_deg for x in self._recent]
        near = [v for v in vals if abs(v - r.azimuth_deg) <= c.agree_deg]
        if len(near) < c.min_agree:
            return SoundDirectionResult(
                False, Reason.INCONSISTENT, t=r.t,
                azimuth_deg=r.azimuth_deg, confidence=r.confidence,
                details={"agreeing": len(near)})
        med = float(np.median(near))
        return SoundDirectionResult(
            True, Reason.OK, t=r.t, azimuth_deg=med,
            delay_s=r.delay_s, confidence=r.confidence, peak=r.peak,
            peak_ratio=r.peak_ratio, resolution_deg=r.resolution_deg,
            details={"agreeing": len(near)})

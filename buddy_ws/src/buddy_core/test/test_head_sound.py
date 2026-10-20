"""Tests for two-microphone direction estimation (synthetic audio only).

These prove the algorithm and its rejection logic on simulated signals.
They say nothing about the accuracy on the real microphones.
"""

from __future__ import annotations

import numpy as np
import pytest
from head_fakes import FS, SPACING, frac_delay, stereo_from_azimuth

from buddy_core.head_tracking import config_from_dict
from buddy_core.head_tracking.sound_direction import (
    Reason,
    SoundDirectionEstimator,
    SoundDirectionFilter,
)


def estimator(**kw):
    sound = {"enabled": True, "mic_spacing_m": SPACING}
    sound.update(kw)
    return SoundDirectionEstimator(config_from_dict({"sound": sound}).sound)


@pytest.mark.parametrize("az", [-60, -45, -20, 0, 20, 45, 60])
def test_synthetic_source_is_localised_within_resolution(az):
    r = estimator().estimate(*stereo_from_azimuth(az), FS, True)
    assert r.ok, r.reason
    assert abs(r.azimuth_deg - az) <= 3.0
    assert r.confidence >= 0.5


def test_left_call_is_negative_and_right_call_is_positive():
    est = estimator()
    left = est.estimate(*stereo_from_azimuth(-40), FS, True)
    right = est.estimate(*stereo_from_azimuth(+40), FS, True)
    assert left.azimuth_deg < -30 and right.azimuth_deg > 30


def test_swap_channels_mirrors_the_result():
    a = estimator().estimate(*stereo_from_azimuth(35), FS, True)
    b = estimator(swap_channels=True).estimate(*stereo_from_azimuth(35),
                                               FS, True)
    assert a.azimuth_deg > 0 > b.azimuth_deg


def test_works_with_moderate_noise():
    r = estimator().estimate(*stereo_from_azimuth(30, snr_db=5, seed=4),
                             FS, True)
    assert r.ok and abs(r.azimuth_deg - 30) <= 4


def test_confidence_falls_as_noise_rises():
    est = estimator(min_confidence=0.0, min_psr=0.0, min_peak=0.0)
    hi = est.estimate(*stereo_from_azimuth(20, snr_db=20), FS, True)
    lo = est.estimate(*stereo_from_azimuth(20, snr_db=-3), FS, True)
    assert hi.confidence > lo.confidence


def test_result_states_its_limits_honestly():
    r = estimator().estimate(*stereo_from_azimuth(0), FS, True)
    assert r.front_back_ambiguous is True
    # one raw sample at 16 kHz / 15 cm is roughly 8 degrees
    assert 6.0 < r.resolution_deg < 10.0


# --------------------------------------------------------- rejection
def test_unsynchronized_microphones_are_always_rejected():
    r = estimator().estimate(*stereo_from_azimuth(30), FS, False)
    assert not r.ok and r.reason == Reason.UNSYNCHRONIZED
    assert r.azimuth_deg is None


def test_a_clock_offset_larger_than_the_geometry_is_not_misread():
    # two free-running devices: channel 1 is 5 ms late.  Even if the
    # caller wrongly claims they are synchronized, the true peak lies
    # outside the physical window and nothing credible remains.
    x, y = stereo_from_azimuth(30, seconds=0.5)
    y = np.roll(y, int(0.005 * FS))
    r = estimator().estimate(x, y, FS, True)
    assert not r.ok
    assert r.reason in (Reason.LOW_PEAK, Reason.AMBIGUOUS,
                        Reason.LOW_CONFIDENCE)


def test_independent_noise_on_each_channel_is_uncertain():
    rng = np.random.default_rng(5)
    x = 0.05 * rng.standard_normal(4800)
    y = 0.05 * rng.standard_normal(4800)
    r = estimator().estimate(x, y, FS, True)
    assert not r.ok and r.reason == Reason.LOW_PEAK


def test_two_equal_sources_are_ambiguous():
    rng = np.random.default_rng(6)
    n = int(0.3 * FS)
    s1, s2 = rng.standard_normal(n), rng.standard_normal(n)
    d1 = -SPACING * np.sin(np.radians(-45)) / 343.0
    d2 = -SPACING * np.sin(np.radians(45)) / 343.0
    x = 0.05 * (s1 + s2)
    y = 0.05 * (frac_delay(s1, d1) + frac_delay(s2, d2))
    r = estimator().estimate(x, y, FS, True)
    assert not r.ok and r.reason == Reason.AMBIGUOUS


def test_geometry_unset_disabled_and_bad_input():
    x, y = stereo_from_azimuth(10)
    assert estimator(mic_spacing_m=None).estimate(
        x, y, FS, True).reason == Reason.GEOMETRY_UNSET
    assert estimator(enabled=False).estimate(
        x, y, FS, True).reason == Reason.DISABLED
    est = estimator()
    assert est.estimate(x, y[:-5], FS, True).reason == Reason.BAD_INPUT
    bad = x.copy()
    bad[10] = np.nan
    assert est.estimate(bad, y, FS, True).reason == Reason.BAD_INPUT
    assert est.estimate(x, y, 0, True).reason == Reason.BAD_INPUT
    assert est.estimate("abc", y, FS, True).reason == Reason.BAD_INPUT
    assert est.estimate(np.zeros((2, 3)), np.zeros((2, 3)), FS,
                        True).reason == Reason.BAD_INPUT


def test_short_silent_and_clipped_windows_are_rejected():
    est = estimator()
    x, y = stereo_from_azimuth(10)
    assert est.estimate(x[:200], y[:200], FS, True).reason \
        == Reason.TOO_SHORT
    z = np.zeros_like(x)
    assert est.estimate(z, z, FS, True).reason == Reason.SILENT
    clipped = np.sign(x)
    assert est.estimate(clipped, np.sign(y), FS, True).reason \
        == Reason.CLIPPING


def test_low_snr_against_a_known_noise_floor_is_rejected():
    x, y = stereo_from_azimuth(10)
    rms = float(np.sqrt(np.mean(x * x)))
    r = estimator().estimate(x, y, FS, True, noise_rms=rms * 0.9)
    assert r.reason == Reason.LOW_SNR


def test_direction_near_the_microphone_axis_is_rejected():
    r = estimator().estimate(*stereo_from_azimuth(86), FS, True)
    assert not r.ok and r.reason in (Reason.ENDFIRE, Reason.OUT_OF_RANGE)


def test_delay_outside_geometry_is_rejected():
    # a source claimed to be 1 m apart on a 15 cm pair
    x, y = stereo_from_azimuth(0)
    y = frac_delay(y, 0.0012)
    r = estimator().estimate(x, y, FS, True)
    assert not r.ok


# ------------------------------------------------------- calibration
def test_calibrated_delay_offset_is_removed():
    x, y = stereo_from_azimuth(0)
    offset = 3 / FS                       # 3 samples of channel skew
    y = frac_delay(y, offset)
    raw = estimator().estimate(x, y, FS, True)
    fixed = estimator(delay_offset_s=offset).estimate(x, y, FS, True)
    assert abs(fixed.azimuth_deg) <= 3.0
    assert raw.azimuth_deg is None or abs(raw.azimuth_deg) > 10.0


def test_wrong_spacing_changes_the_angle_so_it_must_be_measured():
    x, y = stereo_from_azimuth(30)
    right = estimator().estimate(x, y, FS, True).azimuth_deg
    wrong = estimator(mic_spacing_m=SPACING * 0.7).estimate(
        x, y, FS, True)
    assert not wrong.ok or abs(wrong.azimuth_deg - right) > 5.0


# ------------------------------------------------------------ filter
def _ok(est, az, t, seed=0):
    r = est.estimate(*stereo_from_azimuth(az, seed=seed), FS, True, t=t)
    assert r.ok
    return r


def test_filter_needs_agreement_before_accepting():
    cfg = config_from_dict({"sound": {"enabled": True,
                                      "mic_spacing_m": SPACING}}).sound
    est, flt = SoundDirectionEstimator(cfg), SoundDirectionFilter(cfg)
    first = flt.add(_ok(est, 30, 0.0))
    assert not first.ok and first.reason == Reason.INCONSISTENT
    second = flt.add(_ok(est, 31, 0.2, seed=1))
    assert second.ok and abs(second.azimuth_deg - 30) <= 3


def test_filter_rejects_disagreeing_estimates():
    cfg = config_from_dict({"sound": {"enabled": True,
                                      "mic_spacing_m": SPACING}}).sound
    est, flt = SoundDirectionEstimator(cfg), SoundDirectionFilter(cfg)
    flt.add(_ok(est, -40, 0.0))
    assert not flt.add(_ok(est, 40, 0.2, seed=1)).ok


def test_filter_forgets_old_estimates():
    cfg = config_from_dict({"sound": {"enabled": True,
                                      "mic_spacing_m": SPACING}}).sound
    est, flt = SoundDirectionEstimator(cfg), SoundDirectionFilter(cfg)
    flt.add(_ok(est, 30, 0.0))
    assert not flt.add(_ok(est, 30, 10.0, seed=1)).ok


def test_filter_passes_failures_through_unchanged():
    cfg = config_from_dict({"sound": {"enabled": True,
                                      "mic_spacing_m": SPACING}}).sound
    est, flt = SoundDirectionEstimator(cfg), SoundDirectionFilter(cfg)
    bad = est.estimate(*stereo_from_azimuth(0), FS, False)
    assert flt.add(bad) is bad

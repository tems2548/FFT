"""Duty Cycle Analyzer: oscilloscope-style threshold-crossing pulse
measurements (period/frequency/duty/HIGH-LOW/rise-fall time)."""
import numpy as np
import pytest

import FFT

FS = 100000.0


def make_square_wave(freq, duty, fs, duration_s, smooth_samples=1):
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    phase = (t * freq) % 1.0
    signal = np.where(phase < duty, 1.0, -1.0)
    if smooth_samples > 1:
        kernel = np.ones(smooth_samples) / smooth_samples
        signal = np.convolve(signal, kernel, mode="same")
    return signal


class TestComputePulseMetrics:
    def test_known_square_wave_100hz_30_percent_duty(self):
        # Verified by hand against these exact expected values when the
        # function was written; smoothing gives the edges a small, finite
        # rise/fall time (an ideal step has zero, which is a degenerate
        # case for a threshold-crossing measurement).
        signal = make_square_wave(freq=100.0, duty=0.3, fs=FS, duration_s=0.05, smooth_samples=5)
        result = FFT.compute_pulse_metrics(signal, FS)
        assert result is not None
        assert result["period"] * 1000 == pytest.approx(10.0, abs=0.01)
        assert result["frequency"] == pytest.approx(100.0, abs=0.1)
        assert result["duty_percent"] == pytest.approx(30.0, abs=0.5)
        assert result["high_time"] * 1000 == pytest.approx(3.0, abs=0.05)
        assert result["low_time"] * 1000 == pytest.approx(7.0, abs=0.05)
        assert result["pulse_width"] == result["high_time"]
        assert result["rise_time"] > 0
        assert result["fall_time"] > 0

    def test_known_square_wave_25hz_50_percent_duty(self):
        signal = make_square_wave(freq=25.0, duty=0.5, fs=FS, duration_s=0.2, smooth_samples=3)
        result = FFT.compute_pulse_metrics(signal, FS)
        assert result is not None
        assert result["frequency"] == pytest.approx(25.0, abs=0.1)
        assert result["duty_percent"] == pytest.approx(50.0, abs=0.5)

    def test_flat_signal_returns_none(self):
        assert FFT.compute_pulse_metrics(np.ones(1000), FS) is None
        assert FFT.compute_pulse_metrics(np.zeros(1000), FS) is None

    def test_too_few_edges_returns_none(self):
        # A single transition -- not even one full period.
        signal = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        assert FFT.compute_pulse_metrics(signal, FS) is None

    def test_higher_duty_cycle_gives_longer_high_time(self):
        low_duty = FFT.compute_pulse_metrics(make_square_wave(100.0, 0.2, FS, 0.05), FS)
        high_duty = FFT.compute_pulse_metrics(make_square_wave(100.0, 0.8, FS, 0.05), FS)
        assert high_duty["high_time"] > low_duty["high_time"]
        assert high_duty["duty_percent"] > low_duty["duty_percent"]


class TestThresholdCrossings:
    def test_finds_rising_and_falling_crossings_on_a_ramp(self):
        # A triangle-ish ramp 0 -> 2 -> 0 over 100 samples, threshold at 1.0
        # (the midpoint) should show exactly one rising and one falling
        # crossing.
        n = 100
        up = np.linspace(0, 2, n // 2)
        down = np.linspace(2, 0, n // 2)
        signal = np.concatenate([up, down])
        times, directions = FFT._threshold_crossings(signal, fs=1000.0, level=1.0)
        assert len(times) == 2
        assert list(directions) == [1, -1]

    def test_sub_sample_interpolation_is_accurate(self):
        # A linear ramp 0, 0.1, 0.2, ..., 1.0 at fs=10 (1 sample = 0.1s).
        # level=0.55 sits exactly halfway between sample 5 (0.5) and
        # sample 6 (0.6), so the interpolated crossing must land exactly
        # halfway between their times too: t=0.55s.
        signal = np.linspace(0, 1, 11)
        times, directions = FFT._threshold_crossings(signal, fs=10.0, level=0.55)
        assert len(times) == 1
        assert times[0] == pytest.approx(0.55, abs=1e-9)
        assert directions[0] == 1

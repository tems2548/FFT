"""Goertzel algorithm: single-bin DFT correctness and arbitrary-frequency
selectivity."""
import numpy as np
import pytest

import FFT
from dsp_helpers import bin_aligned_freq, make_tone

FS = 1000.0
N = 1024


class TestGoertzelMatchesFft:
    def test_matches_fft_bin_exactly_when_on_grid(self):
        bin_idx = 37
        freq = bin_aligned_freq(bin_idx, FS, N)
        window_func = np.hanning(N)
        mag_scale = 2.0 / np.sum(window_func)
        signal = make_tone(freq, FS, N, amplitude=0.7)
        windowed = signal * window_func

        fft_mag = np.abs(np.fft.rfft(windowed)[bin_idx]) * mag_scale
        goertzel_mag = FFT.compute_goertzel(windowed, freq, FS, mag_scale)
        assert goertzel_mag == pytest.approx(fft_mag, abs=1e-9)
        assert goertzel_mag == pytest.approx(0.7, rel=1e-6)

    def test_recovers_amplitude_of_off_grid_tone(self):
        # The whole point of Goertzel over reading an FFT bin: a target
        # frequency doesn't need to land on the fs/N grid.
        off_grid_freq = 123.456
        window_func = np.hanning(N)
        mag_scale = 2.0 / np.sum(window_func)
        signal = make_tone(off_grid_freq, FS, N, amplitude=0.5)
        windowed = signal * window_func
        mag = FFT.compute_goertzel(windowed, off_grid_freq, FS, mag_scale)
        assert mag == pytest.approx(0.5, rel=1e-3)

    def test_selective_around_target_frequency(self):
        # Evaluating Goertzel at frequencies away from an injected tone
        # should read much lower than evaluating it right at the tone.
        window_func = np.hanning(N)
        mag_scale = 2.0 / np.sum(window_func)
        tone_freq = 150.0
        signal = make_tone(tone_freq, FS, N, amplitude=1.0)
        windowed = signal * window_func
        on_target = FFT.compute_goertzel(windowed, tone_freq, FS, mag_scale)
        off_target = FFT.compute_goertzel(windowed, tone_freq + 50.0, FS, mag_scale)
        assert on_target > 10 * off_target

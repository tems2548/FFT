"""Noise Analysis metrics (against known white noise) and Cepstrum Analysis
(against a signal with a known, evenly-spaced harmonic series)."""
import numpy as np
import pytest

import FFT
from dsp_helpers import run_pipeline

FS = 2000.0
N = 4096


class TestNoiseMetrics:
    def test_integrated_rms_matches_injected_white_noise(self):
        rng = np.random.default_rng(seed=7)
        noise_std = 0.05
        noise = rng.normal(0.0, noise_std, N)
        result = run_pipeline(noise, FS, "Hann")
        mask = np.ones(len(result["mag"]), dtype=bool)
        mask[0] = False  # DC
        _rms_noise, _density, integrated = FFT.compute_noise_metrics(
            result["mag"] ** 2, result["mag_scale"], result["window_func"], N, FS, mask
        )
        # A true-RMS meter looking at just this noise should read close to
        # the value that generated it -- verified empirically within ~1%
        # for this seed; tolerance here is deliberately looser (5%) so the
        # test isn't sensitive to the exact seed/window used.
        assert integrated == pytest.approx(noise_std, rel=0.05)

    def test_rms_noise_matches_noise_floor_db_exactly(self):
        # compute_noise_metrics.rms_noise is documented as the linear-volts
        # version of compute_snr's noise_floor_db -- 20*log10(rms_noise)
        # must equal noise_floor_db exactly, not just approximately.
        rng = np.random.default_rng(seed=3)
        noise = rng.normal(0.0, 0.02, N)
        result = run_pipeline(noise, FS, "Hann")
        mask = np.ones(len(result["mag"]), dtype=bool)
        mask[0] = False
        rms_noise, _density, _integrated = FFT.compute_noise_metrics(
            result["mag"] ** 2, result["mag_scale"], result["window_func"], N, FS, mask
        )
        noise_floor_db = 10 * np.log10(np.mean((result["mag"][mask]) ** 2))
        assert 20 * np.log10(rms_noise) == pytest.approx(noise_floor_db, abs=1e-9)

    def test_empty_mask_returns_zeros(self):
        mag = np.ones(10)
        mask = np.zeros(10, dtype=bool)
        window_func = np.ones(20)
        result = FFT.compute_noise_metrics(mag**2, 1.0, window_func, 20, FS, mask)
        assert result == (0.0, 0.0, 0.0)


class TestFormatDensity:
    def test_auto_scales_to_nv(self):
        assert "nV" in FFT.format_density(5e-9)

    def test_auto_scales_to_uv(self):
        assert "uV" in FFT.format_density(5e-6)

    def test_auto_scales_to_mv(self):
        assert "mV" in FFT.format_density(5e-3)


class TestCepstrum:
    def test_peak_quefrency_matches_fundamental_period(self):
        # A signal built from 5 harmonics of f0 has strong periodic
        # structure in its log-magnitude spectrum with period f0 -- the
        # cepstrum's peak quefrency should land at 1/f0.
        f0 = 50.0
        t = np.arange(N) / FS
        signal = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
        result = run_pipeline(signal, FS, "Hann")
        cepstrum = FFT.compute_real_cepstrum(result["spectrum"], N)
        peak = FFT.find_cepstrum_peak(cepstrum, FS)
        assert peak is not None
        quefrency_s, equiv_freq_hz, _amplitude, _idx = peak
        assert quefrency_s == pytest.approx(1.0 / f0, abs=1e-4)
        assert equiv_freq_hz == pytest.approx(f0, abs=0.5)

    def test_matches_direct_fft_based_cepstrum_for_even_window(self):
        # compute_real_cepstrum() is documented to reuse the rfft half-
        # spectrum (mirrored) instead of running a second full FFT, purely
        # for performance -- must be numerically identical to the textbook
        # IFFT(log|FFT(x)|) definition computed the naive way.
        rng = np.random.default_rng(11)
        signal = rng.normal(0.0, 1.0, 512)
        spectrum_half = np.fft.rfft(signal)
        via_helper = FFT.compute_real_cepstrum(spectrum_half, 512)

        full_spectrum = np.fft.fft(signal)
        naive = np.real(np.fft.ifft(np.log(np.abs(full_spectrum) + 1e-12)))
        assert via_helper == pytest.approx(naive, abs=1e-9)

    def test_returns_none_when_window_too_small(self):
        cepstrum = np.array([1.0, 2.0, 3.0])
        assert FFT.find_cepstrum_peak(cepstrum, FS, min_quefrency_samples=8) is None

"""Time-domain stats, waveform classification, SINAD/ENOB/THD.

Expected values for the waveform-shape tests were captured by running the
actual pipeline once and reading off the result (see the numbers in each
test's comment) -- not derived from a textbook formula, since e.g. THD
depends on exactly which harmonics find_harmonics() locates.
"""
import numpy as np
import pytest

import FFT
from dsp_helpers import bin_aligned_freq, run_pipeline

FS = 2000.0
N = 4096


def generate(wave, freq, t):
    if wave == "sine":
        return np.sin(2 * np.pi * freq * t)
    if wave == "square":
        return np.sign(np.sin(2 * np.pi * freq * t))
    if wave == "sawtooth":
        return 2 * (t * freq - np.floor(0.5 + t * freq))
    raise ValueError(wave)


class TestTimeDomainStats:
    def test_sine_amplitude_rms_and_crest_factor(self):
        t = np.arange(N) / FS
        signal = 2.0 * np.sin(2 * np.pi * 30 * t)
        amplitude_pp, rms, crest_factor = FFT.compute_time_domain_stats(signal)
        # abs=1e-3, not tighter: N/FS isn't an exact integer number of
        # cycles of a 30Hz tone, so the discrete RMS over this finite
        # window isn't exactly the continuous-signal ideal.
        assert amplitude_pp == pytest.approx(4.0, abs=1e-6)
        assert rms == pytest.approx(2.0 / np.sqrt(2), abs=1e-3)
        assert crest_factor == pytest.approx(np.sqrt(2), abs=1e-3)

    def test_square_wave_crest_factor_is_one(self):
        t = np.arange(N) / FS
        signal = np.sign(np.sin(2 * np.pi * 30 * t))
        _pp, _rms, crest_factor = FFT.compute_time_domain_stats(signal)
        assert crest_factor == pytest.approx(1.0, abs=1e-3)

    def test_dc_signal_has_zero_rms_gives_infinite_crest_factor(self):
        signal = np.zeros(100)
        _pp, rms, crest_factor = FFT.compute_time_domain_stats(signal)
        assert rms == 0.0
        assert crest_factor == np.inf


class TestDutyCycle:
    def test_symmetric_square_is_50_percent(self):
        signal = np.array([1.0, 1.0, -1.0, -1.0] * 100)
        assert FFT.compute_duty_cycle(signal) == pytest.approx(50.0)

    def test_mostly_high_signal(self):
        signal = np.array([1.0] * 90 + [-1.0] * 10)
        assert FFT.compute_duty_cycle(signal) == pytest.approx(90.0)

    def test_empty_buffer_returns_zero(self):
        assert FFT.compute_duty_cycle(np.array([])) == 0.0


class TestEnob:
    def test_enob_is_the_standard_sinad_conversion(self):
        # Definitional -- (SINAD - 1.76) / 6.02 -- but worth pinning down
        # as a regression guard against a typo'd constant.
        assert FFT.compute_enob(1.76) == pytest.approx(0.0)
        assert FFT.compute_enob(1.76 + 6.02) == pytest.approx(1.0)
        assert FFT.compute_enob(20.0) == pytest.approx((20.0 - 1.76) / 6.02)


class TestThd:
    def test_no_harmonics_returns_none(self):
        assert FFT.compute_thd(np.array([1.0, 0.0, 0.0]), 0, {}) == (None, None)

    def test_pure_sine_has_near_zero_thd(self):
        t = np.arange(N) / FS
        freq = bin_aligned_freq(20, FS, N)
        signal = np.sin(2 * np.pi * freq * t)
        result = run_pipeline(signal, FS, "Hann")
        _f, _d, peak_idx = FFT.find_peak(result["db"], N, FS)
        harmonics = FFT.find_harmonics(result["db"], peak_idx, N, FS, max_harmonic=5)
        thd_percent, _thd_db = FFT.compute_thd(result["mag"], peak_idx, harmonics)
        # Real find_harmonics() will pick up small numerical-noise-floor
        # "harmonics" even for a clean tone; the ceiling here is generous.
        assert thd_percent < 0.01

    def test_square_wave_thd_in_expected_range(self):
        # Empirically ~38.9% with harmonics 2-5 (odd harmonics only found
        # -- 3rd and 5th); ideal square wave with *all* harmonics is
        # ~48.3%, so this is expected to be somewhat lower, not higher.
        t = np.arange(N) / FS
        freq = bin_aligned_freq(20, FS, N)
        signal = np.sign(np.sin(2 * np.pi * freq * t))
        result = run_pipeline(signal, FS, "Hann")
        _f, _d, peak_idx = FFT.find_peak(result["db"], N, FS)
        harmonics = FFT.find_harmonics(result["db"], peak_idx, N, FS, max_harmonic=5)
        thd_percent, _thd_db = FFT.compute_thd(result["mag"], peak_idx, harmonics)
        assert 30.0 < thd_percent < 45.0


class TestClassifyWaveform:
    @pytest.mark.parametrize(
        "wave,expected_label",
        [
            ("sine", "Sine wave"),
            ("square", "Square wave"),
            ("sawtooth", "Sawtooth wave"),
        ],
    )
    def test_recognizes_known_waveform_shapes(self, wave, expected_label):
        t = np.arange(N) / FS
        freq = bin_aligned_freq(20, FS, N)
        signal = generate(wave, freq, t)
        result = run_pipeline(signal, FS, "Hann")
        mag, db = result["mag"], result["db"]
        _f, _d, peak_idx = FFT.find_peak(db, N, FS)
        harmonics = FFT.find_harmonics(db, peak_idx, N, FS, max_harmonic=5)
        _snr_db, noise_floor_db = FFT.compute_snr(mag, peak_idx)
        _pp, _rms, crest_factor = FFT.compute_time_domain_stats(signal)
        shape = FFT.classify_waveform(mag, peak_idx, harmonics, noise_floor_db, crest_factor)
        assert shape == expected_label

    def test_below_noise_floor_is_no_clear_signal(self):
        mag = np.full(100, 1e-6)
        harmonics = {}
        shape = FFT.classify_waveform(mag, 5, harmonics, noise_floor_db=0.0, crest_factor=1.4)
        assert shape == "No clear signal"

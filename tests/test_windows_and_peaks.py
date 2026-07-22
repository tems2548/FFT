"""Window functions, peak/harmonic detection, SNR, and SFDR."""
import numpy as np
import pytest

import FFT
from dsp_helpers import bin_aligned_freq, make_tone, run_pipeline

FS = 1000.0
N = 2048


class TestWindowFunctions:
    def test_rectangular_is_all_ones(self):
        assert np.array_equal(FFT.WINDOW_FUNCTIONS["Rectangular"](256), np.ones(256))

    @pytest.mark.parametrize("window_name", list(FFT.WINDOW_FUNCTIONS.keys()))
    def test_bin_aligned_tone_recovers_exact_amplitude(self, window_name):
        # A tone placed exactly on an FFT bin has zero spectral leakage, so
        # mag_scale's coherent-gain compensation should recover the
        # injected amplitude essentially exactly, for every window --
        # verified numerically for all 6 windows before writing this
        # assertion (see the session notes / commit history).
        bin_idx = 50
        freq = bin_aligned_freq(bin_idx, FS, N)
        signal = make_tone(freq, FS, N, amplitude=0.7)
        result = run_pipeline(signal, FS, window_name)
        assert result["mag"][bin_idx] == pytest.approx(0.7, rel=1e-6)

    def test_narrower_windows_have_a_narrower_main_lobe(self):
        # Rectangular's main lobe should be visibly narrower than
        # Flat-top's -- the whole reason to offer a choice of window.
        # Measured as -3dB half-width in bins.
        bin_idx = 50
        freq = bin_aligned_freq(bin_idx, FS, N)
        signal = make_tone(freq, FS, N, amplitude=1.0)

        def half_width_bins(window_name):
            db = run_pipeline(signal, FS, window_name)["db"]
            peak_db = db[bin_idx]
            i = bin_idx
            while db[i] > peak_db - 3.0:
                i += 1
            return i - bin_idx

        assert half_width_bins("Rectangular") < half_width_bins("Flat-top")


class TestParabolicInterpolation:
    def test_symmetric_peak_returns_same_bin(self):
        mag_db = np.array([-50.0, -10.0, 0.0, -10.0, -50.0])
        peak_bin, peak_val = FFT.parabolic_interpolation(mag_db, 2)
        assert peak_bin == pytest.approx(2.0)
        assert peak_val == pytest.approx(0.0)

    def test_edge_index_returns_unmodified(self):
        mag_db = np.array([0.0, -10.0, -50.0])
        peak_bin, peak_val = FFT.parabolic_interpolation(mag_db, 0)
        assert peak_bin == 0.0
        assert peak_val == mag_db[0]

    def test_recovers_off_bin_frequency_within_a_fraction_of_a_bin(self):
        # A tone placed deliberately *between* two bins (bin 50.37) --
        # find_peak's parabolic sub-bin fit should land much closer to the
        # true frequency than naively rounding to the nearest bin would.
        true_bin = 50.37
        freq = true_bin * FS / N
        signal = make_tone(freq, FS, N, amplitude=1.0)
        db = run_pipeline(signal, FS, "Hann")["db"]
        peak_freq, _peak_db, _idx = FFT.find_peak(db, N, FS)
        bin_width_hz = FS / N
        assert abs(peak_freq - freq) < 0.05 * bin_width_hz


class TestHarmonics:
    def test_finds_fundamental_and_two_harmonics_at_expected_frequencies(self):
        fund_bin = 40
        fund_freq = bin_aligned_freq(fund_bin, FS, N)
        h2_freq = bin_aligned_freq(fund_bin * 2, FS, N)
        h3_freq = bin_aligned_freq(fund_bin * 3, FS, N)
        signal = (
            1.0 * make_tone(fund_freq, FS, N)
            + 0.1 * make_tone(h2_freq, FS, N)
            + 0.05 * make_tone(h3_freq, FS, N)
        )
        db = run_pipeline(signal, FS, "Hann")["db"]
        peak_freq, _peak_db, peak_idx = FFT.find_peak(db, N, FS)
        assert peak_freq == pytest.approx(fund_freq, abs=0.5)

        harmonics = FFT.find_harmonics(db, peak_idx, N, FS, max_harmonic=3)
        assert set(harmonics) == {2, 3}
        h2_result_freq, h2_db, _ = harmonics[2]
        h3_result_freq, h3_db, _ = harmonics[3]
        assert h2_result_freq == pytest.approx(h2_freq, abs=0.5)
        assert h3_result_freq == pytest.approx(h3_freq, abs=0.5)
        # 0.1/1.0 amplitude ratio -> -20dB; 0.05/1.0 -> -26dB (approximately,
        # windowing/leakage aside since everything here is bin-aligned).
        assert h2_db - _peak_db == pytest.approx(-20.0, abs=0.5)
        assert h3_db - _peak_db == pytest.approx(-26.0, abs=0.5)

    def test_missing_harmonic_above_nyquist_is_absent(self):
        # A fundamental above fs/3 has no room for a 3rd harmonic below
        # Nyquist -- find_harmonics must simply omit it, not error.
        fund_freq = FS * 0.4
        signal = make_tone(fund_freq, FS, N)
        db = run_pipeline(signal, FS, "Hann")["db"]
        _peak_freq, _peak_db, peak_idx = FFT.find_peak(db, N, FS)
        harmonics = FFT.find_harmonics(db, peak_idx, N, FS, max_harmonic=5)
        assert 3 not in harmonics


class TestSecondPeakAndSfdr:
    def test_second_peak_locates_unrelated_tone(self):
        bin1, bin2 = 40, 137  # not a harmonic relationship
        f1 = bin_aligned_freq(bin1, FS, N)
        f2 = bin_aligned_freq(bin2, FS, N)
        signal = 1.0 * make_tone(f1, FS, N) + 0.2 * make_tone(f2, FS, N)
        db = run_pipeline(signal, FS, "Hann")["db"]
        _peak_freq, peak_db, peak_idx = FFT.find_peak(db, N, FS)
        second = FFT.find_second_peak(db, peak_idx, N, FS)
        assert second is not None
        second_freq, second_db, _idx = second
        assert second_freq == pytest.approx(f2, abs=0.5)
        assert second_db == pytest.approx(peak_db - 20 * np.log10(5), abs=0.5)  # 0.2 -> -13.98dB

    def test_sfdr_is_the_gap_to_the_second_peak(self):
        bin1, bin2 = 40, 137
        f1 = bin_aligned_freq(bin1, FS, N)
        f2 = bin_aligned_freq(bin2, FS, N)
        signal = 1.0 * make_tone(f1, FS, N) + 0.2 * make_tone(f2, FS, N)
        db = run_pipeline(signal, FS, "Hann")["db"]
        _peak_freq, _peak_db, peak_idx = FFT.find_peak(db, N, FS)
        sfdr_db = FFT.compute_sfdr(db, peak_idx, N, FS)
        assert sfdr_db == pytest.approx(20 * np.log10(5), abs=0.5)  # ~13.98dB

    def test_no_added_noise_leaves_only_floating_point_residual(self):
        # A pure bin-aligned tone has no *signal* content anywhere else,
        # but every other bin still isn't exactly zero -- floating-point
        # FFT round-off leaves a residual around -240dB, verified
        # empirically. find_second_peak correctly reports that residual
        # (there's no way to distinguish "no spur" from "a spur 240dB
        # down" from the spectrum alone); the meaningful assertion is that
        # it's nowhere near a real signal level.
        f1 = bin_aligned_freq(40, FS, N)
        signal = make_tone(f1, FS, N)
        db = run_pipeline(signal, FS, "Rectangular")["db"]
        _peak_freq, peak_db, peak_idx = FFT.find_peak(db, N, FS)
        second = FFT.find_second_peak(db, peak_idx, N, FS)
        assert second is not None
        _second_freq, second_db, _idx = second
        assert second_db < peak_db - 200


class TestSnr:
    def test_snr_drops_20db_per_decade_of_noise(self):
        # compute_snr's noise floor is a *per-bin average* power, not
        # total broadband noise power, so it doesn't equal the textbook
        # RMS-ratio SNR formula in absolute terms (that constant offset
        # was verified empirically rather than assumed). But a 10x
        # increase in noise standard deviation must still show up as
        # exactly a 20dB drop in measured SNR, since that per-bin-vs-
        # total normalization constant cancels out in the difference --
        # this is the robust, formula-independent thing to assert.
        freq = bin_aligned_freq(40, FS, N)

        def measure_snr(noise_std, seed):
            rng = np.random.default_rng(seed)
            signal = make_tone(freq, FS, N, amplitude=1.0) + rng.normal(0.0, noise_std, N)
            result = run_pipeline(signal, FS, "Hann")
            _f, _d, peak_idx = FFT.find_peak(result["db"], N, FS)
            snr_db, _noise_floor_db = FFT.compute_snr(result["mag"], peak_idx)
            return snr_db

        snr_low_noise = measure_snr(0.01, seed=1)
        snr_high_noise = measure_snr(0.1, seed=2)
        assert snr_low_noise - snr_high_noise == pytest.approx(20.0, abs=2.0)
        assert snr_low_noise > snr_high_noise > 0

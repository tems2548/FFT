"""Shared synthetic-signal helpers for the DSP test modules -- mirrors the
windowing/FFT/scaling math in FFTBenchWindow.update_frame() closely enough
to be a faithful stand-in, without needing a real Qt window."""
import numpy as np

import FFT


def make_tone(freq, fs, n, amplitude=1.0, phase=0.0):
    t = np.arange(n) / fs
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def bin_aligned_freq(bin_idx, fs, n):
    """The exact frequency that lands precisely on FFT bin `bin_idx` for
    an n-sample window at sample rate fs -- avoids spectral leakage
    entirely, so tests using it isolate the thing under test (windowing/
    scaling/detection logic) from leakage, which is a separate concern."""
    return bin_idx * fs / n


def run_pipeline(signal, fs, window_name="Hann"):
    """The same windowing -> FFT -> magnitude/dB steps update_frame() runs
    every frame, as a reusable pure-function pipeline for tests."""
    n = len(signal)
    window_func = FFT.WINDOW_FUNCTIONS[window_name](n)
    mag_scale = 2.0 / np.sum(window_func)
    spectrum = np.fft.rfft(signal * window_func)
    mag = np.abs(spectrum) * mag_scale
    db = 20 * np.log10(mag + 1e-12)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    return {
        "freqs": freqs,
        "spectrum": spectrum,
        "mag": mag,
        "db": db,
        "window_func": window_func,
        "mag_scale": mag_scale,
        "window_size": n,
    }

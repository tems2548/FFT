"""
Real-time FFT test bench: scrolling time-domain waveform next to its
live frequency-domain spectrum, driven by a synthetic dynamic test signal.

Run:
    python FFT.py
    python FFT.py --wave chirp --freq 20 --freq2 80
    python FFT.py --wave sine --freq 440 --samplerate 44100

Use the waveform dropdown to switch signal live, and the sliders to change
the base frequency / noise level while it's running.

Or analyze a live ESP32 ADC feed instead of a synthetic signal (see
main.c, which streams decimated ADC samples over the same USB-UART used
for flashing/logging):
    python FFT.py --serial              # pick the port from a GUI list
    python FFT.py --serial COM5
    python FFT.py --serial /dev/ttyUSB0 --baud 3000000
"""
import argparse
import csv
import datetime
import itertools
import queue
import struct
import sys
import threading
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtSvg, QtWidgets

try:
    import serial
except ImportError:
    serial = None

try:
    import pyqtgraph.opengl as gl
except ImportError:
    gl = None

try:
    import psutil
except ImportError:
    psutil = None

WAVE_TYPES = ["demo", "sine", "chirp", "square", "sawtooth", "noise"]
SPECTROGRAM_HISTORY = 120

# 3D FFT waterfall: caps how many (decimated) frequency bins feed the GL
# surface mesh, independent of the FFT window size (which can be up to
# several thousand bins) -- keeps the vertex count, and so the per-frame
# mesh/normal rebuild cost, bounded regardless of --window.
GL3D_MAX_FREQ_BINS = 150

# Drift Analysis panel: selectable metrics, each tracked as a rolling
# history over time, with its own unit (they don't share a common scale,
# so only one is plotted at a time rather than overlaid).
DRIFT_METRICS = ["Frequency", "DC Bias", "Noise Floor", "THD", "SINAD", "Die Temperature"]
DRIFT_UNITS = {
    "Frequency": "Hz",
    "DC Bias": "V",
    "Noise Floor": "dB",
    "THD": "%",
    "SINAD": "dB",
    "Die Temperature": "°C",
}

# Performance Benchmark panel: named wall-clock stages of update_frame(),
# timed in the order they actually run (see the perf_t_* markers there).
# "Other" covers the small remaining housekeeping (history/snapshot
# bookkeeping) not worth its own named stage.
PERF_STAGE_NAMES = ["Total", "Acquire+FFT", "Detection", "Cepstrum", "Goertzel", "Duty Cycle", "Spectrogram/3D", "Other"]

# Graphs section: (group caption, [checkbox labels]) -- purely a sidebar
# organization aid so 12 checkboxes read as 3 clusters instead of one
# undifferentiated list. Labels must match the keys used when building the
# checkboxes in _build_sidebar().
GRAPH_GROUPS = [
    ("Core", ["Time domain", "Frequency domain", "Phase spectrum", "Bode plot", "Spectrogram"]),
    ("Trends", ["Noise floor & SNR trend", "Drift Analysis", "Performance Benchmark", "CPU / RAM Usage"]),
    ("Advanced", ["Cepstrum Analysis", "Goertzel Analyzer", "3D FFT (waterfall)"]),
]

# One-line tooltip per Graphs checkbox, shown on hover instead of a
# permanent hint label -- with 12 graphs, a fixed 2-3 line hint under each
# one would roughly double the sidebar's length.
GRAPH_TOOLTIPS = {
    "Time domain": "Scrolling waveform, AC-coupled (DC bias removed) in live mode.",
    "Frequency domain": "Live spectrum: peak/2nd-peak markers, peak-hold, delta cursors, hover readout.",
    "Phase spectrum": "Instantaneous phase per bin; blanked out below the noise floor (meaningless there).",
    "Bode plot": "Magnitude and phase together on one shared frequency axis (dual Y axis).",
    "Spectrogram": "Scrolling time x frequency x magnitude waterfall (2D, color-mapped).",
    "Noise floor & SNR trend": "Rolling history of the noise floor and SNR over time.",
    "Drift Analysis": "Rolling history of a selectable metric: frequency, DC bias, noise floor, THD, SINAD, or die temperature.",
    "Performance Benchmark": "Wall-clock time per pipeline stage, each frame -- this app's own cost, not the signal's.",
    "CPU / RAM Usage": "This process's CPU% and RAM (MB) over time. Requires psutil.",
    "Cepstrum Analysis": "IFFT(log|FFT|) -- reveals periodic structure in the spectrum itself (pitch / echo delay).",
    "Goertzel Analyzer": "Exact magnitude at specific target frequencies, not snapped to the FFT's bin grid.",
    "3D FFT (waterfall)": "Rotatable OpenGL view of the same spectrogram data. Requires PyOpenGL.",
}

MAGIC_META = b"META"
MAGIC_DATA = b"DATA"


def crc16_ccitt(data, crc=0xFFFF):
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF). Must match main.c's
    crc16_ccitt_update() bit-for-bit -- verified against the standard test
    vector (CRC of b"123456789" == 0x29B1) when this was written."""
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def _cosine_sum_window(n, coeffs):
    """Generic cosine-sum window: coeffs[0] - coeffs[1]*cos(2*pi*k/(n-1)) +
    coeffs[2]*cos(4*pi*k/(n-1)) - ... Covers Hann/Hamming/Blackman/
    Blackman-Harris/flat-top, which are all members of this family."""
    k = np.arange(n)
    w = np.zeros(n)
    for i, c in enumerate(coeffs):
        sign = -1.0 if i % 2 else 1.0
        w += sign * c * np.cos(2 * np.pi * i * k / (n - 1))
    return w


# Each window trades frequency resolution (narrow main lobe) against
# amplitude/SFDR accuracy (low sidelobes, wide main lobe) differently:
# Rectangular has the narrowest main lobe but leaks badly; flat-top has a
# very wide main lobe but the flattest passband, so it's the standard
# choice when you need an accurate amplitude reading rather than to
# resolve closely-spaced tones. Coefficients match scipy.signal.windows
# so results are the same as elsewhere without adding a scipy dependency.
WINDOW_FUNCTIONS = {
    "Hann": lambda n: np.hanning(n),
    "Hamming": lambda n: np.hamming(n),
    "Blackman": lambda n: np.blackman(n),
    "Blackman-Harris": lambda n: _cosine_sum_window(n, [0.35875, 0.48829, 0.14128, 0.01168]),
    "Flat-top": lambda n: _cosine_sum_window(n, [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]),
    "Rectangular": lambda n: np.ones(n),
}


class SerialReader:
    """Reads main.c's framed ADC stream off a serial port in a background
    thread and feeds decoded voltage samples into a queue.

    Packets are framed with an ASCII magic word (META/DATA) so the parser
    can resync past any ESP_LOG text that lands in the same UART stream.
    """

    def __init__(self, port, baud):
        if serial is None:
            raise RuntimeError("--serial requires pyserial: pip install pyserial")
        self.ser = serial.Serial(port, baud, timeout=0.5)
        self.sample_queue = queue.Queue()
        self.sample_rate = None
        self.temp_c = None  # ESP32-S3 die temperature; NaN if the board has no working sensor
        self.packets_ok = 0
        self.packets_bad = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1)
        self.ser.close()

    def _run(self):
        buf = bytearray()
        while not self._stop.is_set():
            chunk = self.ser.read(4096)
            if chunk:
                buf.extend(chunk)
                self._parse(buf)

    def _parse(self, buf):
        """Consumes complete packets from buf in place, leaving any
        trailing partial packet for the next read."""
        while True:
            idx_meta = buf.find(MAGIC_META)
            idx_data = buf.find(MAGIC_DATA)
            candidates = [i for i in (idx_meta, idx_data) if i != -1]
            if not candidates:
                # Keep a short tail in case a magic word is split across reads.
                del buf[: max(0, len(buf) - 3)]
                return
            idx = min(candidates)
            if idx > 0:
                del buf[:idx]

            if buf.startswith(MAGIC_META):
                if len(buf) < 14:
                    return
                payload = bytes(buf[4:12])  # rate(4) + temp_c float32(4)
                (crc_received,) = struct.unpack_from("<H", buf, 12)
                if crc16_ccitt(payload) == crc_received:
                    self.sample_rate, self.temp_c = struct.unpack_from("<If", payload, 0)
                    self.packets_ok += 1
                else:
                    self.packets_bad += 1
                del buf[:14]
            else:  # MAGIC_DATA
                if len(buf) < 6:
                    return
                (count,) = struct.unpack_from("<H", buf, 4)
                needed = 8 + count * 2  # magic(4) + count(2) + samples + crc(2)
                if len(buf) < needed:
                    return
                payload = bytes(buf[4 : 6 + count * 2])  # count field + samples
                (crc_received,) = struct.unpack_from("<H", buf, 6 + count * 2)
                if crc16_ccitt(payload) == crc_received:
                    samples_mv = struct.unpack_from(f"<{count}h", payload, 2)
                    for mv in samples_mv:
                        self.sample_queue.put(mv / 1000.0)  # convert mV to V
                    self.packets_ok += 1
                else:
                    self.packets_bad += 1
                del buf[:needed]


CONNECT_TIMEOUT_S = 6.0  # ESP32 reboots on port-open + runs a 1s rate
                          # measurement before its first META packet.


class SerialPortDialog(QtWidgets.QDialog):
    """Qt dialog to pick a serial port and connect to the ESP32.

    exec() returns Accepted once a live SerialReader (with sample_rate
    already populated) is available as self.reader / self.port, or
    Rejected if the user closes the window.
    """

    def __init__(self, default_baud, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect to ESP32")
        self.setMinimumWidth(420)
        self.reader = None
        self.port = None
        self._pending_reader = None
        self._pending_port = None
        self._deadline = 0.0

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll)

        layout = QtWidgets.QGridLayout(self)

        layout.addWidget(QtWidgets.QLabel("Serial port:"), 0, 0)
        self.port_combo = QtWidgets.QComboBox()
        layout.addWidget(self.port_combo, 0, 1, 1, 2)
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_ports)
        layout.addWidget(refresh_btn, 0, 3)

        layout.addWidget(QtWidgets.QLabel("Baud rate:"), 1, 0)
        self.baud_edit = QtWidgets.QLineEdit(str(default_baud))
        layout.addWidget(self.baud_edit, 1, 1)

        self.status_label = QtWidgets.QLabel("Select a port and click Connect.")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("dialogStatus")
        layout.addWidget(self.status_label, 2, 0, 1, 4)

        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.setDefault(True)
        self.connect_btn.clicked.connect(self._on_connect)
        layout.addWidget(self.connect_btn, 3, 0, 1, 4)

        self._refresh_ports()

    def _refresh_ports(self):
        import serial.tools.list_ports as list_ports

        current = self.port_combo.currentData()
        self.port_combo.clear()
        ports = list(list_ports.comports())
        for p in ports:
            self.port_combo.addItem(f"{p.device} — {p.description}", p.device)
        if current is not None:
            i = self.port_combo.findData(current)
            if i >= 0:
                self.port_combo.setCurrentIndex(i)

    def _on_connect(self):
        if self.port_combo.count() == 0:
            self.status_label.setText("No serial ports found. Plug in the board and Refresh.")
            return
        port = self.port_combo.currentData()
        try:
            baud = int(self.baud_edit.text())
        except ValueError:
            self.status_label.setText("Baud rate must be an integer.")
            return

        self.connect_btn.setEnabled(False)
        self.status_label.setText(f"Connecting to {port} @ {baud}...")

        try:
            reader = SerialReader(port, baud)
            reader.start()
        except Exception as exc:
            self.status_label.setText(f"Failed to open {port}: {exc}")
            self.connect_btn.setEnabled(True)
            return

        self._pending_reader = reader
        self._pending_port = port
        self._deadline = time.time() + CONNECT_TIMEOUT_S
        self._poll_timer.start()

    def _poll(self):
        reader = self._pending_reader
        if reader.sample_rate is not None:
            self._poll_timer.stop()
            self.reader = reader
            self.port = self._pending_port
            self.accept()
            return
        if time.time() > self._deadline:
            self._poll_timer.stop()
            reader.stop()
            self.status_label.setText(
                f"No data from {self._pending_port} within {CONNECT_TIMEOUT_S:.0f}s — "
                "check the port/baud, or the board may still be booting. Try again."
            )
            self.connect_btn.setEnabled(True)


def sweep_phase(t, f_lo, f_hi, period):
    """Closed-form phase for a frequency that sweeps sinusoidally between
    f_lo and f_hi with the given period, evaluated at absolute time t so it
    stays continuous across chunk boundaries."""
    fc = (f_hi + f_lo) / 2.0
    fd = (f_hi - f_lo) / 2.0
    w = 2 * np.pi / period
    return 2 * np.pi * fc * t + fd * period * (1 - np.cos(w * t))


def generate_chunk(wave, start_n, n, fs, freq, freq2, sweep_period, noise_level):
    t = (start_n + np.arange(n)) / fs

    if wave == "sine":
        s = np.sin(2 * np.pi * freq * t)
    elif wave == "chirp":
        s = np.sin(sweep_phase(t, freq, freq2, sweep_period))
    elif wave == "square":
        s = np.sign(np.sin(2 * np.pi * freq * t))
    elif wave == "sawtooth":
        s = 2 * (t * freq - np.floor(0.5 + t * freq))
    elif wave == "noise":
        s = np.zeros(n)
    elif wave == "demo":
        s = (
            0.6 * np.sin(2 * np.pi * 6 * t)
            + 0.4 * np.sin(2 * np.pi * 14 * t)
            + 0.5 * np.sin(sweep_phase(t, 20, 80, 8.0))
        )
    else:
        raise ValueError(f"unknown wave type {wave!r}")

    if noise_level > 0:
        s = s + np.random.normal(0.0, noise_level, n)
    return s


def parabolic_interpolation(mag_db, idx):
    """Quadratic fit through the bin at idx and its two neighbors.

    FFT bin spacing is coarse, so the raw argmax lands on whichever bin
    happens to be closest to the true tone frequency, off by up to half a
    bin. Fitting a parabola through the (log-magnitude) peak and its
    neighbors gives a sub-bin estimate of the true peak location and
    amplitude without needing a larger FFT.

    Returns (interpolated_bin, interpolated_value_db).
    """
    n = len(mag_db)
    if idx <= 0 or idx >= n - 1:
        return float(idx), mag_db[idx]
    alpha, beta, gamma = mag_db[idx - 1], mag_db[idx], mag_db[idx + 1]
    denom = alpha - 2 * beta + gamma
    if denom == 0:
        return float(idx), beta
    p = 0.5 * (alpha - gamma) / denom
    peak_bin = idx + p
    peak_val = beta - 0.25 * (alpha - gamma) * p
    return peak_bin, peak_val


def find_peak(mag_db, window_size, fs):
    """Locate the dominant tone (skipping DC) with sub-bin accuracy."""
    idx = int(np.argmax(mag_db[1:]) + 1)
    peak_bin, peak_val = parabolic_interpolation(mag_db, idx)
    freq = peak_bin * fs / window_size
    return freq, peak_val, idx


def find_harmonic(mag_db, fundamental_bin, window_size, fs, harmonic_number=2, search_radius=3):
    """Look for a harmonic near harmonic_number * fundamental_bin.

    Real tones rarely land on an exact integer multiple of the fundamental
    bin, so this searches a small window around the expected location for
    the local max, then refines it the same way as find_peak.
    """
    target_bin = fundamental_bin * harmonic_number
    n = len(mag_db)
    if target_bin < 1 or target_bin >= n - 1:
        return None
    lo = max(1, int(round(target_bin)) - search_radius)
    hi = min(n - 1, int(round(target_bin)) + search_radius + 1)
    local_idx = lo + int(np.argmax(mag_db[lo:hi]))
    peak_bin, peak_val = parabolic_interpolation(mag_db, local_idx)
    freq = peak_bin * fs / window_size
    return freq, peak_val, local_idx


def compute_snr(mag, peak_idx, harmonic_idx=None, exclude_radius=2):
    """Fundamental peak power vs. average noise-floor power.

    The bins around the fundamental (and, if given, the harmonic) are
    excluded from the noise floor estimate since they carry signal, not
    noise; DC is excluded too.

    Returns (snr_db, noise_floor_db).
    """
    power = mag ** 2
    n = len(power)
    mask = np.ones(n, dtype=bool)
    mask[0] = False
    for idx in (peak_idx, harmonic_idx):
        if idx is None:
            continue
        lo, hi = max(0, idx - exclude_radius), min(n, idx + exclude_radius + 1)
        mask[lo:hi] = False
    noise_floor = power[mask].mean() if mask.any() else 1e-20
    signal_power = power[peak_idx]
    noise_floor_db = 10 * np.log10(max(noise_floor, 1e-20))
    snr_db = 10 * np.log10(signal_power / max(noise_floor, 1e-20))
    return snr_db, noise_floor_db


def find_harmonics(mag_db, fundamental_bin, window_size, fs, max_harmonic=5, search_radius=3):
    """Locate harmonics 2..max_harmonic in a single pass.

    The on-screen 2nd-harmonic reading and the THD calculation both need
    harmonic locations; computing them once here and sharing the result
    avoids running find_harmonic for h=2 twice per frame.

    Returns {harmonic_number: (freq, db, idx)} for whichever harmonics were
    found (some may be missing near/above Nyquist).
    """
    results = {}
    for h in range(2, max_harmonic + 1):
        result = find_harmonic(mag_db, fundamental_bin, window_size, fs, harmonic_number=h, search_radius=search_radius)
        if result is not None:
            results[h] = result
    return results


def find_second_peak(mag_db, primary_idx, window_size, fs, exclude_radius=2):
    """Locate the second-strongest independent spectral component, with
    sub-bin accuracy — for multi-tone signals where a second tone isn't
    harmonically related to the first (an arbitrary frequency, unlike
    find_harmonic which only searches near integer multiples of the
    fundamental). DC and the primary peak's own bins are excluded.

    Returns (freq, db, idx), or None if nothing else is present.
    """
    n = len(mag_db)
    mask = np.ones(n, dtype=bool)
    mask[0] = False
    lo, hi = max(0, primary_idx - exclude_radius), min(n, primary_idx + exclude_radius + 1)
    mask[lo:hi] = False
    if not mask.any():
        return None
    masked_positions = np.flatnonzero(mask)
    idx = int(masked_positions[np.argmax(mag_db[mask])])
    peak_bin, peak_val = parabolic_interpolation(mag_db, idx)
    freq = peak_bin * fs / window_size
    return freq, peak_val, idx


def compute_sfdr(mag_db, peak_idx, window_size, fs, exclude_radius=2):
    """Spurious-Free Dynamic Range: dB gap between the fundamental peak and
    the next-largest spectral component (DC and the fundamental's own bins
    excluded) — how far the signal sits above its worst spur, whether
    that spur is a harmonic, an unrelated second tone, or noise.
    """
    second = find_second_peak(mag_db, peak_idx, window_size, fs, exclude_radius)
    if second is None:
        return np.inf
    _freq, spur_db, _idx = second
    return mag_db[peak_idx] - spur_db


def compute_time_domain_stats(buffer):
    """Peak-to-peak amplitude, RMS, and crest factor (zero-to-peak / RMS —
    how "peaky" the waveform is; ~1.41 for a sine, higher for impulsive
    signals) of the current time-domain window."""
    amplitude_pp = buffer.max() - buffer.min()
    rms = float(np.sqrt(np.mean(buffer ** 2)))
    peak = float(np.max(np.abs(buffer)))
    crest_factor = peak / rms if rms > 0 else np.inf
    return amplitude_pp, rms, crest_factor


def compute_duty_cycle(buffer):
    """Percentage of samples above the buffer's own midlevel (0V in live
    mode, since it's already AC-coupled by the caller; synthetic
    waveforms are zero-mean by construction). Most meaningful for
    square/PWM-like signals -- a symmetric sine naturally comes out near
    50%."""
    if len(buffer) == 0:
        return 0.0
    return float(np.mean(buffer > 0.0) * 100.0)


def classify_waveform(mag, peak_idx, harmonics, noise_floor_db, crest_factor):
    """Best-effort waveform-shape hint from crest factor and which
    harmonics carry real energy above the noise floor. Heuristic, not
    authoritative: real-world noise, asymmetry, and bandwidth limiting can
    all fool it. Square/sawtooth/triangle waves have distinct, well-known
    harmonic signatures (odd-only vs. odd+even, and how fast the harmonic
    amplitudes fall off), which is what this keys off rather than crest
    factor alone (crest factor alone can't tell a triangle from a sine).
    """
    peak_db = 20 * np.log10(mag[peak_idx] + 1e-12)
    if peak_db < noise_floor_db + 6:
        return "No clear signal"

    def rel_db(h):
        result = harmonics.get(h)
        if result is None:
            return None
        _freq, h_db, _idx = result
        return h_db - peak_db

    h2, h3, h4, h5 = rel_db(2), rel_db(3), rel_db(4), rel_db(5)
    margin = noise_floor_db - peak_db + 6  # "at least 6dB above the noise floor", relative to peak

    def present(x):
        return x is not None and x > margin

    odd_present = present(h3) or present(h5)
    even_present = present(h2) or present(h4)

    if not odd_present and not even_present:
        # No measurable harmonics could also just mean "noisy signal with
        # a random peak, no real periodic content" -- a true sine's crest
        # factor should be close to sqrt(2) ~= 1.41.
        return "Sine wave" if 1.0 <= crest_factor <= 2.2 else "Sine-like / noisy"
    if even_present and odd_present:
        return "Sawtooth wave"
    if odd_present:
        # Square's odd harmonics fall off as 1/n (3rd ~ -9.5dB); triangle's
        # fall off much faster, as 1/n^2 (3rd ~ -19dB) -- a clear gap to
        # threshold on.
        if h3 is not None and h3 < -15:
            return "Triangle wave"
        return "Square wave"
    return "Complex / harmonic-rich"


def compute_noise_metrics(power_avg, mag_scale, window_func, window_size, fs, mask):
    """Three related but distinct noise readings, over the given (already
    signal-excluding) bin mask:

    - rms_noise: representative per-bin RMS amplitude (V) -- the
      linear-volts version of noise_floor_db (verified:
      20*log10(rms_noise) == noise_floor_db exactly).
    - density: amplitude spectral density (V/sqrt(Hz)), noise normalized
      to a 1 Hz bandwidth, for comparing against datasheet noise specs.
      Needs the window's *noise* gain (sum(w^2)), not the *coherent*
      (tone) gain mag_scale uses for amplitude accuracy -- conflating the
      two overstated the integrated RMS below by ~40% in testing before
      this conversion was added.
    - integrated: total broadband RMS noise (V) across all masked bins --
      what a true-RMS meter would read looking at just the noise.

    Verified against synthetic white noise of a known RMS level (frame-
    averaged like real usage): integrated RMS converged to within 0.1% of
    the injected value, and density to within 1%.
    """
    if not mask.any():
        return 0.0, 0.0, 0.0

    rms_noise = float(np.sqrt(np.mean(power_avg[mask])))

    win_power = np.sum(window_func ** 2)
    psd = power_avg / (mag_scale ** 2) * 2.0 / (fs * win_power)
    delta_f = fs / window_size
    noise_psd = psd[mask]
    density = float(np.sqrt(np.mean(noise_psd)))
    integrated = float(np.sqrt(np.sum(noise_psd) * delta_f))
    return rms_noise, density, integrated


def format_density(density_v):
    """Auto-scaled V/sqrt(Hz) -> the most readable of nV, uV, or mV per
    sqrt(Hz), matching how real noise-analysis tools present this."""
    if density_v < 1e-6:
        return f"{density_v * 1e9:7.2f} nV/rtHz"
    if density_v < 1e-3:
        return f"{density_v * 1e6:7.2f} uV/rtHz"
    return f"{density_v * 1e3:7.2f} mV/rtHz"


def compute_sinad(mag, peak_idx, exclude_radius=2):
    """Signal-to-Noise-and-Distortion: fundamental power vs. everything else
    in the spectrum (noise floor AND harmonics together), unlike SNR which
    excludes harmonics. This is what ENOB is derived from.
    """
    power = mag ** 2
    n = len(power)
    mask = np.ones(n, dtype=bool)
    mask[0] = False  # DC
    lo, hi = max(0, peak_idx - exclude_radius), min(n, peak_idx + exclude_radius + 1)
    mask[lo:hi] = False  # fundamental itself
    signal_power = power[peak_idx]
    noise_and_distortion_power = power[mask].sum()
    return 10 * np.log10(signal_power / max(noise_and_distortion_power, 1e-20))


def compute_enob(sinad_db):
    """Effective Number of Bits: the standard SINAD -> ENOB conversion used
    to characterize real-world ADC resolution."""
    return (sinad_db - 1.76) / 6.02


def compute_thd(mag, peak_idx, harmonics):
    """Total Harmonic Distortion: RMS of the given harmonics relative to
    the fundamental amplitude, as a percentage and in dB.

    Takes the harmonics dict from find_harmonics rather than searching
    again, so THD stays consistent with the on-screen readings for free.

    Returns (None, None) if harmonics is empty. This happens once the
    fundamental is high enough that even the 2nd harmonic falls above
    Nyquist (fs/2) -- there's no harmonic content left in the sampled band
    to measure at all, a hard physical limit of FFT analysis, not a bug.
    Reporting 0% in that case would misleadingly claim a verified-clean
    signal instead of "couldn't be measured".
    """
    if not harmonics:
        return None, None
    fundamental_mag = mag[peak_idx]
    if fundamental_mag <= 0:
        return 0.0, -np.inf
    harmonic_power_sum = sum(mag[idx] ** 2 for (_, _, idx) in harmonics.values())
    thd_ratio = np.sqrt(harmonic_power_sum) / fundamental_mag
    thd_percent = thd_ratio * 100
    thd_db = 20 * np.log10(thd_ratio) if thd_ratio > 0 else -np.inf
    return thd_percent, thd_db


def compute_real_cepstrum(spectrum, window_size):
    """Real cepstrum: IFFT(log|FFT(x)|), i.e. the "spectrum of a spectrum".

    A strong peak at quefrency tau means the log-magnitude spectrum itself
    has periodic structure with period 1/tau -- e.g. evenly-spaced
    harmonics (pitch/fundamental period detection) or a delayed copy of the
    signal added to itself (echo detection). This is information a plain
    magnitude spectrum can't separate from the overall spectral envelope
    shape.

    Reuses the rfft spectrum already computed for the main display rather
    than running a second forward FFT: the full-length complex spectrum a
    real signal would produce is exactly recoverable from the rfft half via
    conjugate symmetry, so only the log-magnitude needs mirroring before
    the (unavoidable) inverse FFT.
    """
    log_mag_half = np.log(np.abs(spectrum) + 1e-12)
    if window_size % 2 == 0:
        log_mag_full = np.concatenate([log_mag_half, log_mag_half[-2:0:-1]])
    else:
        log_mag_full = np.concatenate([log_mag_half, log_mag_half[:0:-1]])
    return np.real(np.fft.ifft(log_mag_full))


def find_cepstrum_peak(cepstrum, fs, min_quefrency_samples=8):
    """Locate the dominant quefrency peak, skipping the first few samples.

    Low-quefrency bins reflect the slowly-varying overall spectral
    envelope shape (not periodicity), so they'd otherwise always dominate
    and mask any genuine periodic structure -- min_quefrency_samples
    excludes that region the same way find_peak excludes DC.

    Returns (quefrency_s, equivalent_freq_hz, amplitude, idx), or None if
    the search range is empty (pathologically small window).
    """
    n = len(cepstrum) // 2
    if n <= min_quefrency_samples:
        return None
    idx = int(np.argmax(cepstrum[min_quefrency_samples:n])) + min_quefrency_samples
    quefrency_s = idx / fs
    equivalent_freq_hz = fs / idx
    return quefrency_s, equivalent_freq_hz, float(cepstrum[idx]), idx


def compute_goertzel(windowed_signal, target_freq, fs, mag_scale):
    """Goertzel algorithm: the DFT coefficient at one specific target
    frequency, without computing (or needing the fixed bin grid of) a full
    FFT.

    Classic hardware Goertzel is a 2nd-order IIR recursion evaluated one
    sample at a time, valued on embedded/fixed-point targets because it
    avoids a full FFT's log(N) passes when only a handful of frequencies
    matter. A per-sample Python loop over a multi-thousand-sample window
    every UI frame would be far slower than numpy here, so this computes
    the mathematically identical result (both are the exact DFT value
    X(k) at the same target bin -- verified: matches np.fft.rfft's own bin
    to within float error when target_freq lands exactly on an FFT bin)
    via one vectorized dot product instead of the recursive loop.

    Unlike reading an FFT bin, target_freq isn't restricted to the FFT's
    fixed fs/N grid spacing -- k = target_freq * n / fs is evaluated as a
    real (non-integer) number, so a known tone (mains hum, a DTMF digit, a
    specific frequency of interest) can be measured exactly rather than
    via the nearest quantized bin.

    windowed_signal must already have the same window function applied as
    the main spectrum, for a consistent (comparable) amplitude scale via
    mag_scale.
    """
    n = len(windowed_signal)
    k = target_freq * n / fs
    sample_idx = np.arange(n)
    w = 2 * np.pi * k / n
    real = np.dot(windowed_signal, np.cos(w * sample_idx))
    imag = -np.dot(windowed_signal, np.sin(w * sample_idx))
    magnitude = np.sqrt(real ** 2 + imag ** 2) * mag_scale
    return magnitude


def parse_goertzel_targets(text, fs):
    """Comma-separated frequency list -> sorted list of valid targets
    (0 < f < Nyquist). Silently drops unparseable tokens and out-of-range
    values rather than erroring, so a user mid-edit (trailing comma,
    partial number) doesn't get an exception on every keystroke."""
    targets = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            f = float(token)
        except ValueError:
            continue
        if 0 < f < fs / 2:
            targets.append(f)
    return sorted(targets)


def _threshold_crossings(buffer, fs, level):
    """Sample indices where buffer crosses `level`, refined to a sub-sample
    time via linear interpolation between the two straddling samples.

    Returns (times_s, directions) where directions[i] is +1 for a rising
    crossing (below to above) and -1 for falling -- used both for the 50%
    crossings (period/duty/pulse-width) and the 10%/90% crossings
    (rise/fall time) by compute_pulse_metrics().
    """
    above = buffer > level
    edge_idx = np.flatnonzero(np.diff(above.astype(np.int8)))
    times, directions = [], []
    for i in edge_idx:
        y0, y1 = buffer[i], buffer[i + 1]
        if y1 == y0:
            continue
        frac = (level - y0) / (y1 - y0)
        times.append((i + frac) / fs)
        directions.append(1 if y1 > y0 else -1)
    return np.array(times), np.array(directions)


def compute_pulse_metrics(buffer, fs):
    """Oscilloscope-style pulse measurements via threshold crossings
    (50% for period/duty/pulse-width, 10%/90% for rise/fall time),
    sub-sample-interpolated for accuracy well below one sample period.

    Meaningful for square/pulse/PWM-like signals; a smooth sine will still
    produce a period/frequency/~50% duty reading (its "edges" are just the
    steepest part of the curve), but rise/fall time there reflects the
    sine's slope near the midpoint rather than a true edge speed.

    Returns None if the window is flat (no amplitude range to threshold)
    or doesn't contain enough edges to measure a full period.
    """
    lo, hi = float(buffer.min()), float(buffer.max())
    span = hi - lo
    if span <= 0:
        return None

    t50, dir50 = _threshold_crossings(buffer, fs, lo + 0.5 * span)
    rising = t50[dir50 == 1]
    falling = t50[dir50 == -1]
    if len(rising) < 2 and len(falling) < 2:
        return None

    # Period from consecutive same-direction edges (falls back to falling
    # edges if the window happens to catch only one rising edge).
    periods = np.diff(rising) if len(rising) >= 2 else np.diff(falling)
    period = float(np.mean(periods))
    frequency = 1.0 / period if period > 0 else np.nan

    def paired_gaps(starts, ends):
        gaps = []
        for s in starts:
            later = ends[ends > s]
            if len(later):
                gaps.append(later[0] - s)
        return float(np.mean(gaps)) if gaps else np.nan

    high_time = paired_gaps(rising, falling)
    low_time = paired_gaps(falling, rising)
    duty_percent = (high_time / period * 100.0) if (period > 0 and not np.isnan(high_time)) else np.nan

    t10, dir10 = _threshold_crossings(buffer, fs, lo + 0.1 * span)
    t90, dir90 = _threshold_crossings(buffer, fs, lo + 0.9 * span)

    def edge_speed(cross_first, cross_second):
        # cross_second times matched to the nearest preceding cross_first
        # time -- i.e. the 10%->90% (or 90%->10%) transition of one edge.
        deltas = []
        for t2 in cross_second:
            earlier = cross_first[cross_first < t2]
            if len(earlier):
                deltas.append(t2 - earlier[-1])
        return float(np.mean(deltas)) if deltas else np.nan

    rise_time = edge_speed(t10[dir10 == 1], t90[dir90 == 1])
    fall_time = edge_speed(t90[dir90 == -1], t10[dir10 == -1])

    return {
        "duty_percent": duty_percent,
        "high_time": high_time,
        "low_time": low_time,
        "rise_time": rise_time,
        "fall_time": fall_time,
        "pulse_width": high_time,  # positive pulse width -- same quantity as HIGH time, standard scope terminology
        "period": period,
        "frequency": frequency,
    }


def save_snapshot_csv(
    t_axis,
    buffer,
    freqs,
    mag,
    db,
    peak_freq,
    peak_db,
    harmonics,
    second_peak,
    snr_db,
    noise_floor_db,
    thd_percent,
    sfdr_db,
    amplitude_pp,
    rms,
    crest_factor,
    dc_bias,
    sinad_db,
    enob,
    duty_cycle,
    waveform_shape,
    rms_noise,
    noise_density,
    integrated_noise,
):
    """Write the current time/frequency snapshot plus summary stats to CSV."""
    fname = f"fft_snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# peak_freq_hz", f"{peak_freq:.4f}"])
        writer.writerow(["# peak_db", f"{peak_db:.2f}"])
        if second_peak is not None:
            second_freq, second_db, _idx = second_peak
            writer.writerow(["# second_peak_freq_hz", f"{second_freq:.4f}"])
            writer.writerow(["# second_peak_db", f"{second_db:.2f}"])
        else:
            writer.writerow(["# second_peak_freq_hz", ""])
            writer.writerow(["# second_peak_db", ""])
        for h, (h_freq, h_db, _idx) in sorted(harmonics.items()):
            writer.writerow([f"# harmonic{h}_freq_hz", f"{h_freq:.4f}"])
            writer.writerow([f"# harmonic{h}_db", f"{h_db:.2f}"])
        writer.writerow(["# dc_bias", f"{dc_bias:.4f}"])
        writer.writerow(["# amplitude_pp", f"{amplitude_pp:.4f}"])
        writer.writerow(["# rms", f"{rms:.4f}"])
        writer.writerow(["# crest_factor", f"{crest_factor:.3f}"])
        writer.writerow(["# snr_db", f"{snr_db:.2f}"])
        writer.writerow(["# sinad_db", f"{sinad_db:.2f}"])
        writer.writerow(["# enob_bits", f"{enob:.2f}"])
        writer.writerow(["# noise_floor_db", f"{noise_floor_db:.2f}"])
        writer.writerow(["# thd_percent", f"{thd_percent:.3f}" if thd_percent is not None else ""])
        writer.writerow(["# sfdr_db", f"{sfdr_db:.2f}"])
        writer.writerow(["# duty_cycle_percent", f"{duty_cycle:.3f}"])
        writer.writerow(["# waveform_shape", waveform_shape])
        writer.writerow(["# rms_noise_v", f"{rms_noise:.6f}"])
        writer.writerow(["# noise_density_v_per_sqrt_hz", f"{noise_density:.9f}"])
        writer.writerow(["# integrated_noise_v", f"{integrated_noise:.6f}"])
        writer.writerow([])
        writer.writerow(["time_s", "amplitude", "freq_hz", "magnitude", "magnitude_db"])
        for row in itertools.zip_longest(t_axis, buffer, freqs, mag, db, fillvalue=""):
            writer.writerow(row)
    return fname


# --- Look & feel -----------------------------------------------------------
# One cohesive dark theme for both the Qt chrome and the plots, instead of
# matplotlib's default light-gray-on-white. Bigger fonts and higher-contrast
# accent colors so the four panels stay readable side by side.
BG = "#11141a"
PANEL_BG = "#171b23"
GRID_FG = "#8b93a7"
TEXT_FG = "#e5e7eb"
ACCENT_TIME = "#3b82f6"
ACCENT_FREQ = "#f97316"
ACCENT_NOISE_FLOOR = "#94a3b8"
ACCENT_SNR = "#22c55e"
ACCENT_OK = "#22c55e"
ACCENT_PEAK_HOLD = "#fbbf24"
ACCENT_PHASE = "#a78bfa"
ACCENT_CURSOR_A = "#38bdf8"
ACCENT_CURSOR_B = "#fb7185"
ACCENT_DRIFT = "#2dd4bf"
ACCENT_CEPSTRUM = "#f472b6"
ACCENT_GOERTZEL = "#fb923c"
ACCENT_DUTY = "#a3e635"
ACCENT_CPU = "#fbbf24"
ACCENT_RAM = "#60a5fa"

# One color per Performance Benchmark stage -- reuses each feature's own
# accent color where there's a natural match (Cepstrum/Goertzel), so the
# benchmark plot's legend visually ties back to the panel it's timing.
PERF_STAGE_COLORS = {
    "Total": "#f8fafc",
    "Acquire+FFT": ACCENT_FREQ,
    "Detection": ACCENT_SNR,
    "Cepstrum": ACCENT_CEPSTRUM,
    "Goertzel": ACCENT_GOERTZEL,
    "Duty Cycle": ACCENT_DUTY,
    "Spectrogram/3D": ACCENT_DRIFT,
    "Other": GRID_FG,
}

# 1.5px reads thin/faint once antialiased on most displays; the primary
# data curves (the actual signal, not overlays like peak-hold or cursors)
# get a heavier weight so they stand out clearly against the dark panels.
CURVE_WIDTH = 2.5

STYLESHEET = f"""
QWidget {{
    background: {BG};
    color: {TEXT_FG};
    font-size: 13px;
}}
QMainWindow, QDialog {{ background: {BG}; }}
QLabel#sectionTitle {{ font-size: 14px; font-weight: 600; color: {TEXT_FG}; }}
QLabel#statsLabel {{
    font-family: Consolas, monospace;
    font-size: 13px;
    background: {PANEL_BG};
    border: 1px solid #262b36;
    border-radius: 6px;
    padding: 10px;
}}
QLabel#modeLabel {{ color: #9ca3af; font-size: 12px; }}
QLabel#dialogStatus {{ color: #9ca3af; }}
QComboBox, QLineEdit {{
    background: {PANEL_BG};
    border: 1px solid #2f3542;
    border-radius: 4px;
    padding: 4px 6px;
}}
QPushButton {{
    background: #1f2430;
    border: 1px solid #2f3542;
    border-radius: 5px;
    padding: 7px 10px;
}}
QPushButton:hover {{ background: #262c3a; }}
QPushButton:disabled {{ color: #5b6472; }}
QPushButton#saveButton {{ background: #16321f; border-color: #1f5c34; }}
QPushButton#saveButton:hover {{ background: #1b3f27; }}
QSlider::groove:horizontal {{ height: 4px; background: #2a2f3b; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {ACCENT_TIME};
    width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;
}}
"""


def apply_plot_theme(plot_item, title, xlabel, ylabel):
    plot_item.setTitle(title, color=TEXT_FG, size="12pt")
    plot_item.setLabel("bottom", xlabel, color=GRID_FG)
    plot_item.setLabel("left", ylabel, color=GRID_FG)
    plot_item.showGrid(x=True, y=True, alpha=0.25)
    plot_item.getAxis("bottom").setTextPen(GRID_FG)
    plot_item.getAxis("left").setTextPen(GRID_FG)


class FFTBenchWindow(QtWidgets.QMainWindow):
    def __init__(self, args, live, reader, port_label, fs):
        super().__init__()
        self.args = args
        self.live = live
        self.reader = reader
        self.fs = fs

        self.window_size = args.window
        self.samples_per_frame = max(1, int(fs / args.fps))
        self.buffer = np.zeros(self.window_size)
        self.window_name = args.fft_window
        self.window_func = None
        self.mag_scale = None
        self._recompute_window()
        self.t_axis = np.linspace(0, self.window_size / fs, self.window_size, endpoint=False)
        self.freqs = np.fft.rfftfreq(self.window_size, d=1 / fs)

        self.spec_history = np.full((SPECTROGRAM_HISTORY, len(self.freqs)), -100.0)
        self.noise_history_len = SPECTROGRAM_HISTORY
        self.noise_time_axis = np.linspace(-self.noise_history_len / args.fps, 0, self.noise_history_len)
        self.noise_floor_history = np.full(self.noise_history_len, -100.0)
        self.snr_history = np.full(self.noise_history_len, 0.0)

        # 3D FFT waterfall: same rolling time x frequency x magnitude data
        # as the 2D spectrogram (self.spec_history), just decimated along
        # the frequency axis to bound the GL mesh's vertex count.
        self.gl3d_freq_stride = max(1, len(self.freqs) // GL3D_MAX_FREQ_BINS)

        # Rolling history for the Drift Analysis panel -- one array per
        # selectable metric, all sharing the same time base as the Noise
        # floor/SNR trend above. Kept as plain floats (not dB-scaled or
        # otherwise transformed) since each metric has its own natural unit.
        self.drift_histories = {name: np.full(self.noise_history_len, 0.0) for name in DRIFT_METRICS}
        self.drift_metric = "Frequency"

        # Cepstrum Analysis: only the first half is meaningful (the real
        # cepstrum of a real signal is symmetric about window_size/2).
        self.quefrency_axis = np.arange(self.window_size // 2) / fs

        # Goertzel Analyzer: user-editable list of specific target
        # frequencies to measure exactly (not snapped to the FFT bin
        # grid). Defaults to mains hum (50/60 Hz) and its 2nd harmonic --
        # an arbitrary but broadly useful starting point.
        self.goertzel_targets_text = "50, 60, 120"
        self.goertzel_targets = parse_goertzel_targets(self.goertzel_targets_text, fs)

        # Performance Benchmark: rolling per-stage frame time (ms), sharing
        # the same history length/time base as the Noise floor/Drift panels.
        # Seeded at a small positive epsilon, not 0.0 -- the plot's log Y
        # axis can't represent log10(0), and real perf_counter() deltas are
        # always > 0 anyway.
        self.perf_histories = {name: np.full(self.noise_history_len, 1e-3) for name in PERF_STAGE_NAMES}
        self.perf_stage_ms = {name: 0.0 for name in PERF_STAGE_NAMES}

        # CPU/RAM usage (this process, not system-wide) -- psutil.Process
        # is stateful for cpu_percent(): the first call just primes the
        # baseline (result is meaningless/0), and every later call returns
        # usage *since the previous call*, which happens to be exactly the
        # non-blocking, per-frame sampling this app needs.
        self.psutil_process = None
        if psutil is not None:
            try:
                self.psutil_process = psutil.Process()
                self.psutil_process.cpu_percent(None)
            except Exception:
                self.psutil_process = None
        self.cpu_history = np.full(self.noise_history_len, 0.0)
        self.ram_history = np.full(self.noise_history_len, 0.0)

        # Duty Cycle Analyzer Mode: off by default -- the threshold-
        # crossing measurements it computes are meaningless for a
        # continuous tone/noise signal and only worth the per-frame cost
        # for square/pulse/PWM-like signals.
        self.duty_cycle_mode = False

        self.n = 0
        self.wave = args.wave
        self.freq = args.freq
        self.freq2 = args.freq2
        self.noise = args.noise
        self.fps = float(args.fps)
        self.last_frame_time = None
        self.last_snapshot = None
        self.paused = False

        # Power-domain exponential averaging of the spectrum (like a real
        # analyzer's "trace averaging"), so the displayed curve and all the
        # readings derived from it settle down instead of jumping every
        # frame -- especially needed at high sample rates, where each
        # window covers a short, disjoint (non-overlapping) slice of
        # signal. Averaging happens in power (mag**2), not dB, since
        # averaging logs is statistically biased.
        self.spectrum_alpha = 1.0 - args.averaging / 100.0
        self.power_avg = None

        self.log_freq_axis = False

        # Max-hold overlay: the highest dB seen at each bin since the last
        # reset, like a real analyzer's peak-hold trace -- catches
        # intermittent spurs that come and go between frames, which the
        # live/averaged trace alone can miss.
        self.show_peak_hold = True
        self.peak_hold = np.full(len(self.freqs), -100.0)

        self.show_cursors = False

        self.dsp_lab_mode = False

        title = f"FFT Test Bench — Live ESP32 ADC ({port_label} @ {fs:.0f} Hz)" if live else "FFT Test Bench"
        self.setWindowTitle(title)
        self._build_ui(title)
        self._setup_shortcuts()

        self.settings = QtCore.QSettings("FFTBench", "ESP32FFTVisualizer")
        self._load_settings()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / args.fps))

    def _setup_shortcuts(self):
        shortcuts = (
            ("Space", self._on_pause_click),
            ("Ctrl+S", self._on_save_click),
            ("Ctrl+E", self._on_export_click),
            ("Ctrl+R", self._on_peak_hold_reset),
        )
        self._shortcuts = []  # keep strong refs -- QShortcut is parented but Python GC can still race it
        for key, slot in shortcuts:
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.activated.connect(slot)
            self._shortcuts.append(sc)

    def _settings_checkbox_key(self, label):
        return "graphs/" + label.replace(" ", "_").replace("/", "-")

    def _load_settings(self):
        """Best-effort restore of last session's window size/position and
        control state. Every read has a fallback to the current (just-
        built) default, so a first run, a settings file from an older
        version with missing keys, or a corrupted value all just fall
        back to defaults rather than raising."""
        s = self.settings
        geometry = s.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        for label, checkbox in self._graph_checkboxes.items():
            checked = s.value(self._settings_checkbox_key(label), None)
            if checked is not None:
                checkbox.setChecked(str(checked).lower() == "true")

        window_name = s.value("spectrum/fft_window")
        if window_name in WINDOW_FUNCTIONS:
            self.window_combo.setCurrentText(window_name)
        averaging = s.value("spectrum/averaging", type=int, defaultValue=None)
        if averaging is not None:
            self.averaging_slider.setValue(averaging)
        log_axis = s.value("spectrum/log_axis")
        if log_axis is not None:
            self.log_axis_checkbox.setChecked(str(log_axis).lower() == "true")
        peak_hold = s.value("spectrum/peak_hold")
        if peak_hold is not None:
            self.peak_hold_checkbox.setChecked(str(peak_hold).lower() == "true")
        drift_metric = s.value("spectrum/drift_metric")
        if drift_metric in DRIFT_METRICS:
            self.drift_metric_combo.setCurrentText(drift_metric)
        goertzel_targets_text = s.value("spectrum/goertzel_targets")
        if goertzel_targets_text:
            self.goertzel_freqs_edit.setText(goertzel_targets_text)
            self._on_goertzel_targets_change()

    def _save_settings(self):
        s = self.settings
        s.setValue("window/geometry", self.saveGeometry())
        for label, checkbox in self._graph_checkboxes.items():
            s.setValue(self._settings_checkbox_key(label), checkbox.isChecked())
        s.setValue("spectrum/fft_window", self.window_name)
        s.setValue("spectrum/averaging", self.averaging_slider.value())
        s.setValue("spectrum/log_axis", self.log_freq_axis)
        s.setValue("spectrum/peak_hold", self.show_peak_hold)
        s.setValue("spectrum/drift_metric", self.drift_metric)
        s.setValue("spectrum/goertzel_targets", self.goertzel_targets_text)

    def _recompute_window(self):
        self.window_func = WINDOW_FUNCTIONS[self.window_name](self.window_size)
        self.mag_scale = 2.0 / np.sum(self.window_func)
        # A different window changes the spectrum's scale/shape, so blending
        # it into the old running average would produce a misleading
        # transient; just restart averaging from the next frame instead.
        self.power_avg = None

    def _apply_freq_axis_mode(self):
        """(Re)configure the frequency plot's X axis for linear or log
        display. Only PlotDataItem curves (self.freq_curve) get pyqtgraph's
        automatic log10 transform when setLogMode() is called on the
        PlotItem -- ScatterPlotItem markers and the crosshair's
        InfiniteLines/TextItem are plain GraphicsItems that don't
        participate in that, so their positions are converted manually via
        _freq_to_axis_x() wherever they're set. Applies to both the
        magnitude and phase plots, which share the same frequency axis."""
        low = max(self.freqs[1], 1e-6)  # log10(0) is undefined; nothing
        # below the FFT's own resolution is resolvable anyway.
        for plot_widget in (self.freq_plot, self.phase_plot):
            plot_item = plot_widget.getPlotItem()
            plot_item.setLogMode(x=self.log_freq_axis, y=False)
            if self.log_freq_axis:
                plot_item.setXRange(np.log10(low), np.log10(self.fs / 2), padding=0)
            else:
                plot_item.setXRange(0, self.fs / 2, padding=0)

    def _freq_to_axis_x(self, freq):
        return np.log10(freq) if self.log_freq_axis else freq

    def _axis_x_to_freq(self, axis_x):
        if not self.log_freq_axis:
            return axis_x
        # Guard against a pathological cursor/mouse position (e.g. dragged
        # or left in a stale coordinate space) overflowing float range.
        return 10 ** min(axis_x, 300)

    # -- UI construction -----------------------------------------------

    def _build_ui(self, title):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Built before the sidebar (even though the sidebar appears on the
        # left) so the sidebar's per-graph show/hide checkboxes can wire up
        # directly to the already-existing plot widgets.
        plots_widget = self._build_plots()

        # More sections (Signal / Harmonics / Quality) than fit in a fixed
        # 260px-tall column at some window sizes, so scroll rather than
        # clip or squeeze the plots to make room.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(280)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self._build_sidebar(title))
        root.addWidget(scroll, 0)

        # Each plot has a minimum readable height (set in _build_plots), so
        # once there are enough panels to not all fit at a comfortable size,
        # scroll the column instead of silently squeezing every plot thinner.
        plots_scroll = QtWidgets.QScrollArea()
        plots_scroll.setWidgetResizable(True)
        plots_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        plots_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        plots_scroll.setWidget(plots_widget)
        root.addWidget(plots_scroll, 1)

        self.resize(1320, 900)

    def _build_sidebar(self, title):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(10)

        mode_label = QtWidgets.QLabel(title)
        mode_label.setObjectName("modeLabel")
        mode_label.setWordWrap(True)
        layout.addWidget(mode_label)

        # -- Graphs ----------------------------------------------------
        graphs_layout = self._make_collapsible_section(layout, "Graphs", start_expanded=True)

        show_hide_row = QtWidgets.QHBoxLayout()
        show_all_btn = QtWidgets.QPushButton(" Show All")
        show_all_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogYesButton))
        hide_all_btn = QtWidgets.QPushButton(" Hide All")
        hide_all_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogNoButton))
        show_hide_row.addWidget(show_all_btn)
        show_hide_row.addWidget(hide_all_btn)
        graphs_layout.addLayout(show_hide_row)

        plot_widgets_by_label = {
            "Time domain": self.time_plot,
            "Frequency domain": self.freq_plot,
            "Phase spectrum": self.phase_plot,
            "Bode plot": self.bode_plot,
            "Spectrogram": self.spec_plot,
            "Noise floor & SNR trend": self.noise_plot,
            "Drift Analysis": self.drift_plot,
            "Performance Benchmark": self.perf_plot,
            "CPU / RAM Usage": self.sysres_plot,
            "Cepstrum Analysis": self.cepstrum_plot,
            "Goertzel Analyzer": self.goertzel_plot,
            "3D FFT (waterfall)": self.fft3d_plot,
        }
        self._graph_checkboxes = {}
        for group_caption, group_labels in GRAPH_GROUPS:
            group_title = QtWidgets.QLabel(group_caption)
            group_title.setStyleSheet("color: #6b7280; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;")
            graphs_layout.addWidget(group_title)
            for label in group_labels:
                plot_widget = plot_widgets_by_label[label]
                checkbox = QtWidgets.QCheckBox(label)
                checkbox.setToolTip(GRAPH_TOOLTIPS[label])
                # All graphs start hidden -- explicitly set both the
                # widget's visibility and the checkbox's checked state to
                # False *before* connecting toggled, so they start in
                # agreement and the first click (which only fires toggled
                # on an actual change) behaves correctly in either
                # direction.
                plot_widget.setVisible(False)
                checkbox.setChecked(False)
                checkbox.toggled.connect(plot_widget.setVisible)
                graphs_layout.addWidget(checkbox)
                self._graph_checkboxes[label] = checkbox

        show_all_btn.clicked.connect(lambda: [cb.setChecked(True) for cb in self._graph_checkboxes.values()])
        hide_all_btn.clicked.connect(lambda: [cb.setChecked(False) for cb in self._graph_checkboxes.values()])

        drift_metric_row = QtWidgets.QHBoxLayout()
        drift_metric_row.addWidget(QtWidgets.QLabel("Drift metric:"))
        self.drift_metric_combo = QtWidgets.QComboBox()
        self.drift_metric_combo.addItems(DRIFT_METRICS)
        self.drift_metric_combo.setCurrentText(self.drift_metric)
        self.drift_metric_combo.currentTextChanged.connect(self._on_drift_metric_change)
        drift_metric_row.addWidget(self.drift_metric_combo)
        graphs_layout.addLayout(drift_metric_row)

        # -- Spectrum Controls -------------------------------------------
        spectrum_layout = self._make_collapsible_section(layout, "Spectrum Controls", start_expanded=True)

        window_title = QtWidgets.QLabel("FFT Window")
        window_title.setObjectName("sectionTitle")
        spectrum_layout.addWidget(window_title)
        self.window_combo = QtWidgets.QComboBox()
        self.window_combo.addItems(list(WINDOW_FUNCTIONS.keys()))
        self.window_combo.setCurrentText(self.window_name)
        self.window_combo.currentTextChanged.connect(self._on_window_change)
        self.window_combo.setToolTip(
            "Narrow lobe (Rectangular/Hann) resolves close tones; wide lobe\n"
            "(Flat-top) gives the most accurate amplitude/SFDR reading."
        )
        spectrum_layout.addWidget(self.window_combo)

        self.averaging_label = QtWidgets.QLabel(f"Averaging: {self.args.averaging:.0f}%")
        spectrum_layout.addWidget(self.averaging_label)
        self.averaging_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.averaging_slider.setRange(0, 99)
        self.averaging_slider.setValue(int(self.args.averaging))
        self.averaging_slider.valueChanged.connect(self._on_averaging_change)
        self.averaging_slider.setToolTip(
            "Smooths the spectrum trace and readings across frames -- raise\n"
            "this if the display looks jumpy (common at high sample rates)."
        )
        spectrum_layout.addWidget(self.averaging_slider)

        self.log_axis_checkbox = QtWidgets.QCheckBox("Log frequency axis")
        self.log_axis_checkbox.setChecked(self.log_freq_axis)
        self.log_axis_checkbox.toggled.connect(self._on_log_axis_toggle)
        self.log_axis_checkbox.setToolTip(
            "Spreads low frequencies out instead of squeezing them into a\n"
            "sliver of a wide linear range -- closer to how you'd read an\n"
            "octave/decade-spaced signal."
        )
        spectrum_layout.addWidget(self.log_axis_checkbox)

        peak_hold_row = QtWidgets.QHBoxLayout()
        self.peak_hold_checkbox = QtWidgets.QCheckBox("Peak hold")
        self.peak_hold_checkbox.setChecked(self.show_peak_hold)
        self.peak_hold_checkbox.toggled.connect(self._on_peak_hold_toggle)
        self.peak_hold_checkbox.setToolTip(
            "Dashed trace tracks the highest level ever seen per bin --\n"
            "catches intermittent spurs the live trace alone can miss."
        )
        peak_hold_row.addWidget(self.peak_hold_checkbox)
        self.peak_hold_reset_button = QtWidgets.QPushButton("Reset")
        self.peak_hold_reset_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload))
        self.peak_hold_reset_button.clicked.connect(self._on_peak_hold_reset)
        peak_hold_row.addWidget(self.peak_hold_reset_button)
        spectrum_layout.addLayout(peak_hold_row)

        self.cursors_checkbox = QtWidgets.QCheckBox("Delta cursors")
        self.cursors_checkbox.setChecked(self.show_cursors)
        self.cursors_checkbox.toggled.connect(self._on_cursors_toggle)
        self.cursors_checkbox.setToolTip(
            "Drag the two lines on the time/frequency plots to measure\n"
            "the difference between any two points."
        )
        spectrum_layout.addWidget(self.cursors_checkbox)
        self.cursor_label = QtWidgets.QLabel("—")
        self.cursor_label.setObjectName("statsLabel")
        self.cursor_label.setWordWrap(True)
        spectrum_layout.addWidget(self.cursor_label)

        # -- Waveform (synthetic mode only) -------------------------------
        waveform_layout = self._make_collapsible_section(layout, "Waveform", start_expanded=True)

        self.wave_combo = QtWidgets.QComboBox()
        self.wave_combo.addItems(WAVE_TYPES)
        self.wave_combo.setCurrentText(self.wave)
        self.wave_combo.currentTextChanged.connect(self._on_wave_change)
        waveform_layout.addWidget(self.wave_combo)

        self.freq_label = QtWidgets.QLabel(f"Frequency: {self.freq:.0f} Hz")
        waveform_layout.addWidget(self.freq_label)
        self.freq_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        freq_max = max(2, int(min(self.fs / 2, 500)))
        self.freq_slider.setRange(1, freq_max)
        self.freq_slider.setValue(int(min(max(self.freq, 1), freq_max)))
        self.freq_slider.valueChanged.connect(self._on_freq_change)
        waveform_layout.addWidget(self.freq_slider)

        self.noise_label = QtWidgets.QLabel(f"Noise: {self.noise:.2f}")
        waveform_layout.addWidget(self.noise_label)
        self.noise_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.noise_slider.setRange(0, 50)
        self.noise_slider.setValue(int(self.noise * 100))
        self.noise_slider.valueChanged.connect(self._on_noise_change)
        waveform_layout.addWidget(self.noise_slider)

        if self.live:
            # These controls only affect the synthetic generator; disable
            # rather than leave them present-but-inert on live ADC data.
            for w in (self.wave_combo, self.freq_slider, self.noise_slider):
                w.setEnabled(False)

        # -- Snapshot & Export ---------------------------------------------
        snapshot_layout = self._make_collapsible_section(layout, "Snapshot & Export", start_expanded=True)

        self.pause_button = QtWidgets.QPushButton("Pause  (Space)")
        self.pause_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPause))
        self.pause_button.clicked.connect(self._on_pause_click)
        snapshot_layout.addWidget(self.pause_button)
        self.save_button = QtWidgets.QPushButton("Save CSV  (Ctrl+S)")
        self.save_button.setObjectName("saveButton")
        self.save_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton))
        self.save_button.clicked.connect(self._on_save_click)
        snapshot_layout.addWidget(self.save_button)
        self.save_status_label = QtWidgets.QLabel("")
        self.save_status_label.setStyleSheet(f"color: {ACCENT_OK};")
        snapshot_layout.addWidget(self.save_status_label)

        export_row = QtWidgets.QHBoxLayout()
        self.export_format_combo = QtWidgets.QComboBox()
        self.export_format_combo.addItems(["PNG", "SVG", "PDF report"])
        export_row.addWidget(self.export_format_combo)
        self.export_button = QtWidgets.QPushButton("Export  (Ctrl+E)")
        self.export_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowDown))
        self.export_button.clicked.connect(self._on_export_click)
        export_row.addWidget(self.export_button)
        snapshot_layout.addLayout(export_row)
        self.export_status_label = QtWidgets.QLabel("")
        self.export_status_label.setStyleSheet(f"color: {ACCENT_OK};")
        self.export_status_label.setWordWrap(True)
        snapshot_layout.addWidget(self.export_status_label)

        self.fps_label = QtWidgets.QLabel("")
        self.fps_label.setStyleSheet("color: #6b7280; font-size: 11px;")
        snapshot_layout.addWidget(self.fps_label)

        if self.live:
            self.link_label = QtWidgets.QLabel("")
            self.link_label.setStyleSheet("color: #6b7280; font-size: 11px;")
            snapshot_layout.addWidget(self.link_label)
        else:
            self.link_label = None

        # -- Readings --------------------------------------------------
        readings_layout = self._make_collapsible_section(layout, "Readings", start_expanded=True)

        _, self.signal_label = self._add_stats_section(readings_layout, "Signal")
        _, self.harmonics_label = self._add_stats_section(readings_layout, "Harmonics")
        _, self.quality_label = self._add_stats_section(readings_layout, "Signal Quality")
        noise_container, self.noise_analysis_label = self._add_stats_section(readings_layout, "Noise Analysis")
        noise_container.setToolTip(
            "Most accurate for sinusoidal or noise-like signals. Square/\n"
            "sawtooth/PWM waves read an inflated noise level here -- their\n"
            "fast edges leak real broadband energy into every bin, which\n"
            "is physical, not a bug."
        )

        # -- Advanced Analysis (collapsed by default -- specialized tools,
        # not needed for every session) ------------------------------------
        advanced_layout = self._make_collapsible_section(layout, "Advanced Analysis", start_expanded=False)

        self.cepstrum_section, self.cepstrum_label = self._add_stats_section(advanced_layout, "Cepstrum")
        self.cepstrum_section.setToolTip(GRAPH_TOOLTIPS["Cepstrum Analysis"])
        self.cepstrum_section.setVisible(False)

        goertzel_title = QtWidgets.QLabel("Goertzel Analyzer")
        goertzel_title.setObjectName("sectionTitle")
        advanced_layout.addWidget(goertzel_title)
        self.goertzel_freqs_edit = QtWidgets.QLineEdit(self.goertzel_targets_text)
        self.goertzel_freqs_edit.editingFinished.connect(self._on_goertzel_targets_change)
        self.goertzel_freqs_edit.setToolTip(
            "Comma-separated target frequencies (Hz) to measure exactly,\n"
            "not snapped to the FFT's bin grid -- press Enter to apply."
        )
        advanced_layout.addWidget(self.goertzel_freqs_edit)
        self.goertzel_section, self.goertzel_label = self._add_stats_section(advanced_layout, "Goertzel")
        self.goertzel_section.setVisible(False)

        self.perf_section, self.perf_label = self._add_stats_section(advanced_layout, "Performance Benchmark")
        self.perf_section.setToolTip(GRAPH_TOOLTIPS["Performance Benchmark"])
        self.perf_section.setVisible(False)

        self.system_section, self.system_label = self._add_stats_section(advanced_layout, "System (this process)")
        self.system_section.setToolTip(GRAPH_TOOLTIPS["CPU / RAM Usage"])
        self.system_section.setVisible(False)

        # These 4 stats sections show/hide together with the same Graphs
        # checkbox that already gates the matching (expensive) per-frame
        # computation in update_frame() -- one flag, two effects, so a
        # hidden section is never left showing frozen/stale numbers.
        for graph_label, section in (
            ("Cepstrum Analysis", self.cepstrum_section),
            ("Goertzel Analyzer", self.goertzel_section),
            ("Performance Benchmark", self.perf_section),
            ("CPU / RAM Usage", self.system_section),
        ):
            self._graph_checkboxes[graph_label].toggled.connect(section.setVisible)

        self.dsp_lab_checkbox = QtWidgets.QCheckBox("DSP Laboratory Mode")
        self.dsp_lab_checkbox.setChecked(self.dsp_lab_mode)
        self.dsp_lab_checkbox.toggled.connect(self._on_dsp_lab_toggle)
        self.dsp_lab_checkbox.setToolTip(
            "Shows every stage of the pipeline as its own plot/reading:\n"
            "Raw Signal -> DC removal -> Windowing -> FFT -> Power\n"
            "Spectrum -> Peak/Harmonic Detection -> SINAD/THD/ENOB."
        )
        advanced_layout.addWidget(self.dsp_lab_checkbox)

        self.pipeline_section, self.pipeline_label = self._add_stats_section(advanced_layout, "DSP Pipeline")
        self.pipeline_section.setVisible(self.dsp_lab_mode)

        self.duty_cycle_checkbox = QtWidgets.QCheckBox("Duty Cycle Analyzer Mode")
        self.duty_cycle_checkbox.setChecked(self.duty_cycle_mode)
        self.duty_cycle_checkbox.toggled.connect(self._on_duty_cycle_toggle)
        self.duty_cycle_checkbox.setToolTip(
            "Oscilloscope-style pulse measurements via threshold crossings --\n"
            "most meaningful for square/pulse/PWM signals, not a continuous tone."
        )
        advanced_layout.addWidget(self.duty_cycle_checkbox)

        self.duty_cycle_section, self.duty_cycle_label = self._add_stats_section(advanced_layout, "Duty Cycle Analyzer")
        self.duty_cycle_section.setVisible(self.duty_cycle_mode)

        layout.addStretch(1)
        return panel

    def _add_stats_section(self, layout, title):
        """A titled readout box, wrapped in its own container so callers
        that need to show/hide the whole section (title + value together,
        e.g. tied to a feature's enable checkbox) can toggle one widget
        instead of two. Callers that don't need that just ignore the
        returned container -- it has no margins of its own, so it's
        visually identical to adding the title/label directly."""
        container = QtWidgets.QWidget()
        inner = QtWidgets.QVBoxLayout(container)
        inner.setContentsMargins(0, 0, 0, 0)
        section_title = QtWidgets.QLabel(title)
        section_title.setObjectName("sectionTitle")
        inner.addWidget(section_title)
        label = QtWidgets.QLabel("—")
        label.setObjectName("statsLabel")
        label.setWordWrap(True)
        inner.addWidget(label)
        layout.addWidget(container)
        return container, label

    def _make_collapsible_section(self, parent_layout, title, start_expanded=True):
        """A clickable header (arrow + title) that shows/hides a content
        area below it. With ~20 total controls, one flat column reads as a
        wall of undifferentiated widgets; collapsing sections the user
        isn't actively using turns them into a single line instead of a
        full column, without losing anything (nothing is destroyed, just
        hidden).

        Returns the QVBoxLayout to populate with this section's content.
        """
        header = QtWidgets.QToolButton()
        header.setText(title)
        header.setCheckable(True)
        header.setChecked(start_expanded)
        header.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        header.setArrowType(QtCore.Qt.ArrowType.DownArrow if start_expanded else QtCore.Qt.ArrowType.RightArrow)
        header.setStyleSheet(
            "QToolButton { border: none; font-weight: 600; font-size: 14px; padding: 2px 0; }"
        )

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(4, 2, 0, 0)
        content_layout.setSpacing(10)
        content.setVisible(start_expanded)

        def on_toggled(checked):
            content.setVisible(checked)
            header.setArrowType(QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow)

        header.toggled.connect(on_toggled)

        parent_layout.addWidget(header)
        parent_layout.addWidget(content)
        return content_layout

    def _build_plots(self):
        pg.setConfigOptions(antialias=True)
        container = QtWidgets.QWidget()
        self.plots_container = container  # kept for screenshot/report export
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setSpacing(10)

        # DSP Laboratory Mode, stage 1: the untouched input, before DC
        # removal or windowing -- only shown when that mode is on.
        self.raw_signal_plot = pg.PlotWidget(background=PANEL_BG)
        apply_plot_theme(
            self.raw_signal_plot.getPlotItem(),
            "1. Raw Signal (before DC removal)",
            "Time in window (s)",
            "Voltage (V)" if self.live else "Amplitude",
        )
        self.raw_signal_curve = self.raw_signal_plot.plot(self.t_axis, self.buffer, pen=pg.mkPen(ACCENT_TIME, width=CURVE_WIDTH))
        self.raw_signal_plot.setXRange(0, self.window_size / self.fs, padding=0)
        self.raw_signal_plot.setYRange(*((-0.2, 3.5) if self.live else (-2.2, 2.2)))
        self.raw_signal_plot.setMinimumHeight(180)
        self.raw_signal_plot.setVisible(self.dsp_lab_mode)
        vbox.addWidget(self.raw_signal_plot, 1)

        # Time domain (this is, in pipeline terms, stage 2: after DC
        # removal -- the DSP Lab plots on either side of it show stage 1,
        # the untouched raw input, and stage 3/4, after windowing)
        self.time_plot = pg.PlotWidget(background=PANEL_BG)
        apply_plot_theme(
            self.time_plot.getPlotItem(),
            "2. Time domain (DC removed)" if self.dsp_lab_mode else "Time domain",
            "Time in window (s)",
            "AC amplitude (V, bias removed)" if self.live else "Amplitude",
        )
        self.time_curve = self.time_plot.plot(self.t_axis, self.buffer, pen=pg.mkPen(ACCENT_TIME, width=CURVE_WIDTH))
        self.time_plot.setXRange(0, self.window_size / self.fs, padding=0)
        self.time_plot.setYRange(*((-1.8, 1.8) if self.live else (-2.2, 2.2)))
        self.time_plot.setMinimumHeight(180)
        self.time_cursor_a, self.time_cursor_b = self._add_delta_cursor_pair(
            self.time_plot, 0, self.window_size / self.fs
        )
        vbox.addWidget(self.time_plot, 1)

        # DSP Laboratory Mode, stages 3-4: the signal after the window
        # function is applied -- what actually goes into the FFT. Shows
        # the characteristic taper-to-zero-at-the-edges shape.
        self.windowed_signal_plot = pg.PlotWidget(background=PANEL_BG)
        apply_plot_theme(
            self.windowed_signal_plot.getPlotItem(),
            "3-4. Windowed Signal (fed to the FFT)",
            "Time in window (s)",
            "AC amplitude (V, bias removed)" if self.live else "Amplitude",
        )
        self.windowed_signal_curve = self.windowed_signal_plot.plot(
            self.t_axis, self.buffer, pen=pg.mkPen(ACCENT_TIME, width=CURVE_WIDTH)
        )
        self.windowed_signal_plot.setXRange(0, self.window_size / self.fs, padding=0)
        self.windowed_signal_plot.setYRange(*((-1.8, 1.8) if self.live else (-2.2, 2.2)))
        self.windowed_signal_plot.setMinimumHeight(180)
        self.windowed_signal_plot.setVisible(self.dsp_lab_mode)
        vbox.addWidget(self.windowed_signal_plot, 1)

        # Frequency domain, with a hover crosshair for reading values off
        # the curve (a plain static line is hard to read precisely).
        self.freq_plot = pg.PlotWidget(background=PANEL_BG)
        apply_plot_theme(self.freq_plot.getPlotItem(), "Frequency domain", "Frequency (Hz)", "Magnitude (dB)")
        self.freq_curve = self.freq_plot.plot(self.freqs, np.full_like(self.freqs, -100.0), pen=pg.mkPen(ACCENT_FREQ, width=CURVE_WIDTH))
        self.peak_hold_curve = self.freq_plot.plot(
            self.freqs, self.peak_hold, pen=pg.mkPen(ACCENT_PEAK_HOLD, width=1, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.peak_hold_curve.setVisible(self.show_peak_hold)
        self.freq_plot.setYRange(-100, 20, padding=0)
        self.freq_plot.setMinimumHeight(180)
        self.peak_marker = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(ACCENT_FREQ), pen=pg.mkPen(TEXT_FG, width=1))
        self.freq_plot.addItem(self.peak_marker)
        self.second_peak_marker = pg.ScatterPlotItem(
            size=9, symbol="t", brush=pg.mkBrush(ACCENT_SNR), pen=pg.mkPen(TEXT_FG, width=1)
        )
        self.freq_plot.addItem(self.second_peak_marker)
        self._add_crosshair(self.freq_plot, "Hz", "dB")
        self.freq_cursor_a, self.freq_cursor_b = self._add_delta_cursor_pair(self.freq_plot, 0, self.fs / 2)
        vbox.addWidget(self.freq_plot, 1)

        # Phase spectrum, from the raw (unaveraged) FFT bins -- unlike the
        # magnitude trace above, phase can't be power-averaged frame to
        # frame without special handling (see the comment on self.power_avg
        # for why vector-averaging isn't used), so this always shows the
        # instantaneous phase. Bins that aren't meaningfully above the
        # noise floor are blanked out (NaN) rather than plotted, since raw
        # phase there is essentially random and would just be visual noise.
        self.phase_plot = pg.PlotWidget(background=PANEL_BG)
        apply_plot_theme(self.phase_plot.getPlotItem(), "Phase spectrum", "Frequency (Hz)", "Phase (deg)")
        self.phase_curve = self.phase_plot.plot(self.freqs, np.zeros_like(self.freqs), pen=pg.mkPen(ACCENT_PHASE, width=CURVE_WIDTH))
        self.phase_plot.setYRange(-180, 180, padding=0)
        self.phase_plot.setMinimumHeight(180)
        self._add_crosshair(self.phase_plot, "Hz", "deg")
        vbox.addWidget(self.phase_plot, 1)

        # Bode plot: magnitude and phase together on one shared frequency
        # axis, classic dual-Y-axis style. This is just a different layout
        # of the same magnitude/phase data already shown above -- not a
        # true DUT transfer-function Bode plot (that needs a known
        # stimulus and a 2nd channel/reference, which this single-ADC-
        # channel hardware doesn't have). Linear frequency axis only (not
        # linked to the log-axis toggle) to keep the dual-viewbox scope
        # contained.
        self.bode_plot = pg.PlotWidget(background=PANEL_BG)
        bode_item = self.bode_plot.getPlotItem()
        apply_plot_theme(bode_item, "Bode Plot (Magnitude + Phase)", "Frequency (Hz)", "Magnitude (dB)")
        bode_item.getAxis("left").setPen(pg.mkPen(ACCENT_FREQ))
        bode_item.getAxis("left").setTextPen(ACCENT_FREQ)
        self.bode_mag_curve = bode_item.plot(self.freqs, np.full_like(self.freqs, -100.0), pen=pg.mkPen(ACCENT_FREQ, width=CURVE_WIDTH))
        bode_item.setYRange(-100, 20, padding=0)
        bode_item.setXRange(0, self.fs / 2, padding=0)
        self.bode_plot.setMinimumHeight(180)

        self.bode_phase_vb = pg.ViewBox()
        bode_item.showAxis("right")
        bode_item.scene().addItem(self.bode_phase_vb)
        bode_item.getAxis("right").linkToView(self.bode_phase_vb)
        self.bode_phase_vb.setXLink(bode_item)
        bode_item.getAxis("right").setLabel("Phase (deg)", color=ACCENT_PHASE)
        bode_item.getAxis("right").setPen(pg.mkPen(ACCENT_PHASE))
        bode_item.getAxis("right").setTextPen(ACCENT_PHASE)
        self.bode_phase_vb.setYRange(-180, 180, padding=0)
        self.bode_phase_curve = pg.PlotDataItem(self.freqs, np.zeros_like(self.freqs), pen=pg.mkPen(ACCENT_PHASE, width=CURVE_WIDTH))
        self.bode_phase_vb.addItem(self.bode_phase_curve)

        def sync_bode_views():
            self.bode_phase_vb.setGeometry(bode_item.vb.sceneBoundingRect())
            self.bode_phase_vb.linkedViewChanged(bode_item.vb, self.bode_phase_vb.XAxis)

        self._sync_bode_views = sync_bode_views  # keep a strong ref alongside the Qt signal connection
        bode_item.vb.sigResized.connect(sync_bode_views)
        sync_bode_views()
        vbox.addWidget(self.bode_plot, 1)

        self._apply_freq_axis_mode()  # needs both freq_plot and phase_plot to exist

        # Spectrogram
        self.spec_plot = pg.PlotWidget(background=PANEL_BG)
        apply_plot_theme(self.spec_plot.getPlotItem(), "Spectrogram", "Time (s ago)", "Frequency (Hz)")
        self.spec_image = pg.ImageItem()
        # setRect() computes its transform from the image's current
        # width/height, so an image must be assigned before calling it —
        # otherwise it silently scales against a 1x1 placeholder and the
        # image ends up rendered far outside the plot's view range.
        self.spec_image.setImage(self.spec_history, autoLevels=False, levels=(-100, 20))
        self.spec_image.setColorMap(pg.colormap.get("magma"))
        x0 = -SPECTROGRAM_HISTORY / self.args.fps
        self.spec_image.setRect(QtCore.QRectF(x0, 0, -x0, self.fs / 2))
        self.spec_plot.addItem(self.spec_image)
        self.spec_plot.setXRange(x0, 0, padding=0)
        self.spec_plot.setYRange(0, self.fs / 2, padding=0)
        cbar = pg.ColorBarItem(values=(-100, 20), colorMap=pg.colormap.get("magma"), label="dB")
        cbar.setImageItem(self.spec_image, insert_in=self.spec_plot.getPlotItem())
        self.spec_plot.setMinimumHeight(180)
        vbox.addWidget(self.spec_plot, 1)

        # Noise floor / SNR trend
        self.noise_plot = pg.PlotWidget(background=PANEL_BG)
        apply_plot_theme(self.noise_plot.getPlotItem(), "Noise floor & SNR trend", "Time (s ago)", "dB")
        self.noise_floor_curve = self.noise_plot.plot(
            self.noise_time_axis, self.noise_floor_history, pen=pg.mkPen(ACCENT_NOISE_FLOOR, width=CURVE_WIDTH), name="Noise floor (dB)"
        )
        self.snr_curve = self.noise_plot.plot(
            self.noise_time_axis, self.snr_history, pen=pg.mkPen(ACCENT_SNR, width=CURVE_WIDTH), name="SNR (dB)"
        )
        self.noise_plot.addLegend(offset=(10, 10))
        self.noise_plot.setXRange(self.noise_time_axis[0], self.noise_time_axis[-1], padding=0)
        self.noise_plot.setYRange(-100, 60, padding=0)
        self.noise_plot.setMinimumHeight(180)
        vbox.addWidget(self.noise_plot, 1)

        # Drift Analysis: rolling history of one selectable metric at a
        # time (frequency/DC bias/noise floor/THD/SINAD/die temperature).
        # Different units per metric, so auto-range the Y axis rather than
        # a fixed one -- it's recomputed on every setData call.
        self.drift_plot = pg.PlotWidget(background=PANEL_BG)
        apply_plot_theme(self.drift_plot.getPlotItem(), f"Drift Analysis: {self.drift_metric}", "Time (s ago)", DRIFT_UNITS[self.drift_metric])
        self.drift_curve = self.drift_plot.plot(
            self.noise_time_axis, self.drift_histories[self.drift_metric], pen=pg.mkPen(ACCENT_DRIFT, width=CURVE_WIDTH)
        )
        self.drift_plot.getPlotItem().enableAutoRange(axis=pg.ViewBox.YAxis)
        self.drift_plot.setXRange(self.noise_time_axis[0], self.noise_time_axis[-1], padding=0)
        self.drift_plot.setMinimumHeight(180)
        vbox.addWidget(self.drift_plot, 1)

        # Cepstrum Analysis: IFFT(log|FFT|) of the windowed signal, for
        # spotting periodic structure in the spectrum itself (pitch/
        # fundamental-period and echo-delay detection) that a plain
        # magnitude spectrum conflates with the spectral envelope shape.
        self.cepstrum_plot = pg.PlotWidget(background=PANEL_BG)
        apply_plot_theme(self.cepstrum_plot.getPlotItem(), "Cepstrum Analysis", "Quefrency (s)", "Amplitude")
        self.cepstrum_curve = self.cepstrum_plot.plot(
            self.quefrency_axis, np.zeros_like(self.quefrency_axis), pen=pg.mkPen(ACCENT_CEPSTRUM, width=CURVE_WIDTH)
        )
        self.cepstrum_peak_marker = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(ACCENT_CEPSTRUM), pen=pg.mkPen(TEXT_FG, width=1))
        self.cepstrum_plot.addItem(self.cepstrum_peak_marker)
        self.cepstrum_plot.getPlotItem().enableAutoRange(axis=pg.ViewBox.YAxis)
        self.cepstrum_plot.setXRange(self.quefrency_axis[0], self.quefrency_axis[-1], padding=0)
        self.cepstrum_plot.setMinimumHeight(180)
        vbox.addWidget(self.cepstrum_plot, 1)

        # Goertzel Analyzer: exact magnitude at a handful of user-chosen
        # target frequencies, as a small bar chart (a full spectrum trace
        # would be pointless here -- the whole point is looking at just
        # the few frequencies that matter, not the ones between them).
        self.goertzel_plot = pg.PlotWidget(background=PANEL_BG)
        apply_plot_theme(self.goertzel_plot.getPlotItem(), "Goertzel Analyzer (target frequencies)", "Frequency (Hz)", "Magnitude (dB)")
        self.goertzel_plot.setYRange(-100, 20, padding=0)
        self.goertzel_plot.setMinimumHeight(180)
        self.goertzel_bar_item = None
        self._rebuild_goertzel_bars()
        vbox.addWidget(self.goertzel_plot, 1)

        # 3D FFT waterfall: a rotatable OpenGL perspective on the same
        # rolling time x frequency x magnitude history as the 2D
        # spectrogram above -- a different view of the same data, not a
        # separate pipeline. Optional (requires PyOpenGL); degrades to an
        # explanatory placeholder rather than crashing if it's missing or
        # GL context creation fails on this machine.
        self.fft3d_plot = self._build_3d_fft_panel()
        self.fft3d_plot.setMinimumHeight(260)
        vbox.addWidget(self.fft3d_plot, 1)

        # Performance Benchmark: rolling per-stage wall-clock time (ms) for
        # update_frame() itself -- diagnostic for the app's own cost, not
        # the signal. Log Y axis since "Total" and a cheap stage like
        # Goertzel can differ by 1-2 orders of magnitude; a shared linear
        # axis would flatten the cheaper stages to invisible.
        self.perf_plot = pg.PlotWidget(background=PANEL_BG)
        apply_plot_theme(self.perf_plot.getPlotItem(), "Performance Benchmark", "Time (s ago)", "Frame time (ms)")
        self.perf_plot.getPlotItem().setLogMode(x=False, y=True)
        self.perf_curves = {}
        for name in PERF_STAGE_NAMES:
            width = CURVE_WIDTH if name == "Total" else 1.5
            self.perf_curves[name] = self.perf_plot.plot(
                self.noise_time_axis, self.perf_histories[name], pen=pg.mkPen(PERF_STAGE_COLORS[name], width=width), name=name
            )
        self.perf_plot.addLegend(offset=(10, 10))
        self.perf_plot.getPlotItem().enableAutoRange(axis=pg.ViewBox.YAxis)
        self.perf_plot.setXRange(self.noise_time_axis[0], self.noise_time_axis[-1], padding=0)
        self.perf_plot.setMinimumHeight(180)
        vbox.addWidget(self.perf_plot, 1)

        # CPU / RAM usage (this process): dual-axis like the Bode plot
        # above, since the two share no common scale (0-100% vs however
        # many MB this process happens to be using).
        self.sysres_plot = pg.PlotWidget(background=PANEL_BG)
        sysres_item = self.sysres_plot.getPlotItem()
        apply_plot_theme(sysres_item, "CPU / RAM Usage (this process)", "Time (s ago)", "CPU (%)")
        sysres_item.getAxis("left").setPen(pg.mkPen(ACCENT_CPU))
        sysres_item.getAxis("left").setTextPen(ACCENT_CPU)
        self.cpu_curve = sysres_item.plot(self.noise_time_axis, self.cpu_history, pen=pg.mkPen(ACCENT_CPU, width=CURVE_WIDTH))
        sysres_item.setYRange(0, 100, padding=0)
        sysres_item.setXRange(self.noise_time_axis[0], self.noise_time_axis[-1], padding=0)
        self.sysres_plot.setMinimumHeight(180)

        self.ram_vb = pg.ViewBox()
        sysres_item.showAxis("right")
        sysres_item.scene().addItem(self.ram_vb)
        sysres_item.getAxis("right").linkToView(self.ram_vb)
        self.ram_vb.setXLink(sysres_item)
        sysres_item.getAxis("right").setLabel("RAM (MB)", color=ACCENT_RAM)
        sysres_item.getAxis("right").setPen(pg.mkPen(ACCENT_RAM))
        sysres_item.getAxis("right").setTextPen(ACCENT_RAM)
        self.ram_vb.enableAutoRange(axis=pg.ViewBox.YAxis)
        self.ram_curve = pg.PlotDataItem(self.noise_time_axis, self.ram_history, pen=pg.mkPen(ACCENT_RAM, width=CURVE_WIDTH))
        self.ram_vb.addItem(self.ram_curve)

        def sync_sysres_views():
            self.ram_vb.setGeometry(sysres_item.vb.sceneBoundingRect())
            self.ram_vb.linkedViewChanged(sysres_item.vb, self.ram_vb.XAxis)

        self._sync_sysres_views = sync_sysres_views  # keep a strong ref alongside the Qt signal connection
        sysres_item.vb.sigResized.connect(sync_sysres_views)
        sync_sysres_views()
        vbox.addWidget(self.sysres_plot, 1)

        return container

    def _build_3d_fft_panel(self):
        self.fft3d_view = None
        self.fft3d_surface = None
        self.fft3d_colormap = pg.colormap.get("magma")
        if gl is not None:
            try:
                view = gl.GLViewWidget()
                view.setBackgroundColor(PANEL_BG)
                view.setCameraPosition(distance=22, elevation=25, azimuth=-60)

                floor = gl.GLGridItem()
                floor.setSize(10, 10)
                floor.setSpacing(1, 1)
                view.addItem(floor)

                surface = gl.GLSurfacePlotItem(shader="shaded", smooth=True, computeNormals=True)
                view.addItem(surface)

                self.fft3d_view = view
                self.fft3d_surface = surface

                container = QtWidgets.QWidget()
                layout = QtWidgets.QVBoxLayout(container)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(2)
                title = QtWidgets.QLabel("3D FFT (waterfall)")
                title.setStyleSheet(f"color: {TEXT_FG}; font-size: 12pt; font-weight: 600; padding: 4px 2px;")
                layout.addWidget(title)
                layout.addWidget(view, 1)
                return container
            except Exception as exc:
                print(f"3D FFT view unavailable ({exc}); showing fallback message instead.")

        placeholder = QtWidgets.QLabel(
            "3D FFT (waterfall) view requires PyOpenGL, which isn't installed\n"
            "(or a usable OpenGL context isn't available on this machine).\n\n"
            "pip install PyOpenGL"
        )
        placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet(f"color: #6b7280; background: {PANEL_BG}; border-radius: 6px;")
        return placeholder

    def _rebuild_goertzel_bars(self):
        """(Re)creates the bar graph item to match self.goertzel_targets --
        called at startup and whenever the target-frequency list itself
        changes (add/remove/edit), not every frame (per-frame updates just
        change bar heights via setOpts)."""
        if self.goertzel_bar_item is not None:
            self.goertzel_plot.removeItem(self.goertzel_bar_item)
            self.goertzel_bar_item = None
        if not self.goertzel_targets:
            return
        targets = np.array(self.goertzel_targets)
        # Bar width: a fraction of the smallest gap between adjacent
        # targets (or, with only one target, a fraction of Nyquist) so
        # closely-spaced targets don't visually merge into one block.
        if len(targets) > 1:
            min_gap = float(np.min(np.diff(targets)))
        else:
            min_gap = self.fs / 2 * 0.1
        width = max(min_gap * 0.5, self.fs / 2 * 0.005)
        self.goertzel_bar_item = pg.BarGraphItem(
            x=targets, height=np.full(len(targets), -100.0), width=width,
            brush=pg.mkBrush(ACCENT_GOERTZEL), pen=pg.mkPen(TEXT_FG, width=1),
        )
        self.goertzel_plot.addItem(self.goertzel_bar_item)

    def _add_delta_cursor_pair(self, plot_widget, x_lo, x_hi):
        """Two draggable vertical lines (pyqtgraph's built-in movable
        InfiniteLine, so dragging needs no custom mouse-event code) at 25%
        and 75% across the given range. Hidden until "Delta cursors" is
        checked. Readout is recomputed on every drag (sigPositionChanged)
        as well as every animation frame, so it tracks both cursor moves
        and incoming live data."""
        span = x_hi - x_lo
        line_a = pg.InfiniteLine(pos=x_lo + span * 0.25, angle=90, movable=True, pen=pg.mkPen(ACCENT_CURSOR_A, width=1.5))
        line_b = pg.InfiniteLine(pos=x_lo + span * 0.75, angle=90, movable=True, pen=pg.mkPen(ACCENT_CURSOR_B, width=1.5))
        for line in (line_a, line_b):
            plot_widget.addItem(line)
            line.setVisible(self.show_cursors)
            line.sigPositionChanged.connect(self._update_cursor_readouts)
        return line_a, line_b

    def _add_crosshair(self, plot_widget, x_unit, y_unit):
        vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("#4b5563", width=1))
        hline = pg.InfiniteLine(angle=0, pen=pg.mkPen("#4b5563", width=1))
        label = pg.TextItem(color=TEXT_FG, anchor=(0, 1))
        plot_widget.addItem(vline, ignoreBounds=True)
        plot_widget.addItem(hline, ignoreBounds=True)
        plot_widget.addItem(label)
        vline.hide()
        hline.hide()
        label.hide()

        def on_move(pos):
            plot_item = plot_widget.getPlotItem()
            if not plot_item.sceneBoundingRect().contains(pos):
                vline.hide()
                hline.hide()
                label.hide()
                return
            mouse_pt = plot_item.vb.mapSceneToView(pos)
            x, y = mouse_pt.x(), mouse_pt.y()
            vline.setPos(x)
            hline.setPos(y)
            label.setPos(x, y)
            # x is in the ViewBox's own coordinate system, which is log10(Hz)
            # when the log axis is on -- convert back to Hz for the readout.
            display_x = (10 ** x) if getattr(self, "log_freq_axis", False) else x
            label.setText(f"{display_x:.1f} {x_unit}, {y:.1f} {y_unit}")
            vline.show()
            hline.show()
            label.show()

        plot_widget.scene().sigMouseMoved.connect(on_move)

    # -- Control callbacks ------------------------------------------------

    def _on_window_change(self, text):
        self.window_name = text
        self._recompute_window()

    def _on_averaging_change(self, value):
        self.spectrum_alpha = 1.0 - value / 100.0
        self.averaging_label.setText(f"Averaging: {value}%")

    def _on_log_axis_toggle(self, checked):
        # A cursor's raw .value() lives in the ViewBox's own coordinate
        # system, which flips between Hz and log10(Hz) here -- read the
        # physical frequencies under the OLD mode first, then reposition
        # the lines in the new mode so they keep pointing at the same spot.
        #
        # Signals are blocked while repositioning: setValue() on cursor A
        # fires sigPositionChanged immediately, which would otherwise run
        # _update_cursor_readouts() while cursor B still holds its OLD
        # (now-wrong-interpretation) coordinate -- e.g. reading a leftover
        # linear value like 1500 as if it were log10(Hz) tries 10**1500.
        fa = self._axis_x_to_freq(self.freq_cursor_a.value())
        fb = self._axis_x_to_freq(self.freq_cursor_b.value())
        self.log_freq_axis = checked
        self._apply_freq_axis_mode()
        self.freq_cursor_a.blockSignals(True)
        self.freq_cursor_b.blockSignals(True)
        self.freq_cursor_a.setValue(self._freq_to_axis_x(fa))
        self.freq_cursor_b.setValue(self._freq_to_axis_x(fb))
        self.freq_cursor_a.blockSignals(False)
        self.freq_cursor_b.blockSignals(False)
        self._update_cursor_readouts()

    def _on_peak_hold_toggle(self, checked):
        self.show_peak_hold = checked
        self.peak_hold_curve.setVisible(checked)

    def _on_peak_hold_reset(self):
        self.peak_hold[:] = -100.0

    def _on_dsp_lab_toggle(self, checked):
        self.dsp_lab_mode = checked
        self.raw_signal_plot.setVisible(checked)
        self.windowed_signal_plot.setVisible(checked)
        self.pipeline_section.setVisible(checked)
        self.time_plot.getPlotItem().setTitle(
            "2. Time domain (DC removed)" if checked else "Time domain", color=TEXT_FG, size="12pt"
        )

    def _on_duty_cycle_toggle(self, checked):
        self.duty_cycle_mode = checked
        self.duty_cycle_section.setVisible(checked)

    def _on_drift_metric_change(self, metric):
        self.drift_metric = metric
        self.drift_plot.getPlotItem().setTitle(f"Drift Analysis: {metric}", color=TEXT_FG, size="12pt")
        self.drift_plot.getPlotItem().setLabel("left", DRIFT_UNITS[metric], color=GRID_FG)
        self.drift_curve.setData(self.noise_time_axis, self.drift_histories[metric])

    def _on_goertzel_targets_change(self):
        self.goertzel_targets_text = self.goertzel_freqs_edit.text()
        new_targets = parse_goertzel_targets(self.goertzel_targets_text, self.fs)
        if new_targets != self.goertzel_targets:
            self.goertzel_targets = new_targets
            self._rebuild_goertzel_bars()

    def _on_cursors_toggle(self, checked):
        self.show_cursors = checked
        for line in (self.time_cursor_a, self.time_cursor_b, self.freq_cursor_a, self.freq_cursor_b):
            line.setVisible(checked)
        if checked:
            self._update_cursor_readouts()
        else:
            self.cursor_label.setText("—")

    def _update_cursor_readouts(self):
        if not self.show_cursors:
            return

        ta = float(self.time_cursor_a.value())
        tb = float(self.time_cursor_b.value())
        va = float(np.interp(ta, self.time_curve.xData, self.time_curve.yData))
        vb = float(np.interp(tb, self.time_curve.xData, self.time_curve.yData))
        dt = tb - ta
        dv = vb - va
        dt_freq = (1.0 / abs(dt)) if dt != 0 else float("inf")
        unit = "V" if self.live else ""

        # freq_curve.xData is always plain Hz (pyqtgraph's log transform is
        # applied lazily at render time, not stored back into the curve),
        # so interpolation needs the converted-back linear frequency here
        # even though the cursor's own .value() may be in log10(Hz).
        fa = self._axis_x_to_freq(self.freq_cursor_a.value())
        fb = self._axis_x_to_freq(self.freq_cursor_b.value())
        dba = float(np.interp(fa, self.freq_curve.xData, self.freq_curve.yData))
        dbb = float(np.interp(fb, self.freq_curve.xData, self.freq_curve.yData))

        self.cursor_label.setText(
            "\n".join([
                "Time domain",
                f"A:  {ta * 1000:9.3f} ms  {va:7.3f} {unit}",
                f"B:  {tb * 1000:9.3f} ms  {vb:7.3f} {unit}",
                f"dT: {dt * 1000:9.3f} ms  (~{dt_freq:.1f} Hz)",
                f"dV: {dv:7.3f} {unit}",
                "",
                "Frequency domain",
                f"A:  {fa:9.2f} Hz  {dba:6.1f} dB",
                f"B:  {fb:9.2f} Hz  {dbb:6.1f} dB",
                f"df: {fb - fa:9.2f} Hz",
                f"dDB: {dbb - dba:6.1f} dB",
            ])
        )

    def _on_wave_change(self, text):
        self.wave = text

    def _on_freq_change(self, value):
        self.freq = float(value)
        self.freq2 = self.freq * 3
        self.freq_label.setText(f"Frequency: {value} Hz")

    def _on_noise_change(self, value):
        self.noise = value / 100.0
        self.noise_label.setText(f"Noise: {self.noise:.2f}")

    def _on_save_click(self):
        if self.last_snapshot is None:
            return
        fname = save_snapshot_csv(**self.last_snapshot)
        self.save_status_label.setText(f"Saved {fname}")

    def _on_export_click(self):
        fmt = self.export_format_combo.currentText()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            if fmt == "PNG":
                fname = f"fft_export_{timestamp}.png"
                self.plots_container.grab().save(fname, "PNG")
            elif fmt == "SVG":
                fname = f"fft_export_{timestamp}.svg"
                size = self.plots_container.size()
                generator = QtSvg.QSvgGenerator()
                generator.setFileName(fname)
                generator.setSize(size)
                generator.setViewBox(QtCore.QRect(0, 0, size.width(), size.height()))
                generator.setTitle("FFT Test Bench Export")
                painter = QtGui.QPainter(generator)
                self.plots_container.render(painter)
                painter.end()
            else:  # "PDF report"
                fname = f"fft_report_{timestamp}.pdf"
                self._export_pdf_report(fname)
        except Exception as exc:
            self.export_status_label.setStyleSheet("color: #ef4444;")
            self.export_status_label.setText(f"Export failed: {exc}")
            return
        self.export_status_label.setStyleSheet(f"color: {ACCENT_OK};")
        self.export_status_label.setText(f"Saved {fname}")

    def _export_pdf_report(self, fname):
        """Page 1: a text summary of every readings section (a genuine
        report, not just a picture of the numbers). Page 2: the plots,
        scaled to fill the page width."""
        writer = QtGui.QPdfWriter(fname)
        writer.setPageSize(QtGui.QPageSize(QtGui.QPageSize.PageSizeId.A4))
        writer.setResolution(150)
        painter = QtGui.QPainter(writer)
        page_rect = writer.pageLayout().paintRectPixels(writer.resolution())
        margin = 40
        x = page_rect.left() + margin
        y = page_rect.top() + margin

        painter.setFont(QtGui.QFont("Sans Serif", 16, QtGui.QFont.Weight.Bold))
        painter.drawText(x, y, "FFT Test Bench Report")
        y += 24
        painter.setFont(QtGui.QFont("Sans Serif", 9))
        mode = f"Live ADC ({self.reader is not None and 'connected' or 'n/a'})" if self.live else "Synthetic signal"
        painter.drawText(x, y, f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  --  {mode}  --  fs={self.fs:.0f} Hz")
        y += 30

        body_font = QtGui.QFont("Consolas", 10)
        header_font = QtGui.QFont("Sans Serif", 12, QtGui.QFont.Weight.Bold)
        for title, label in (
            ("Signal", self.signal_label),
            ("Harmonics", self.harmonics_label),
            ("Signal Quality", self.quality_label),
            ("Noise Analysis", self.noise_analysis_label),
        ):
            painter.setFont(header_font)
            painter.drawText(x, y, title)
            y += 20
            painter.setFont(body_font)
            for line in label.text().splitlines():
                painter.drawText(x, y, line)
                y += 16
            y += 14

        writer.newPage()
        pixmap = self.plots_container.grab()
        scaled = pixmap.scaledToWidth(page_rect.width(), QtCore.Qt.TransformationMode.SmoothTransformation)
        painter.drawPixmap(page_rect.left(), page_rect.top(), scaled)
        painter.end()

    def _on_pause_click(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_button.setText("Resume  (Space)")
            self.pause_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay))
            self.pause_button.setStyleSheet(f"background: {ACCENT_FREQ}; border-color: {ACCENT_FREQ}; color: #111;")
            self.fps_label.setText("PAUSED")
        else:
            self.pause_button.setText("Pause  (Space)")
            self.pause_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPause))
            self.pause_button.setStyleSheet("")
            self.last_frame_time = None  # don't count the paused interval as a slow frame

    # -- Frame update -------------------------------------------------------

    def update_frame(self):
        if self.paused:
            if self.live:
                # Keep draining the queue so it doesn't grow unbounded while
                # frozen -- the samples are just discarded, not displayed.
                try:
                    while True:
                        self.reader.sample_queue.get_nowait()
                except queue.Empty:
                    pass
            return

        now = time.perf_counter()
        if self.last_frame_time is not None:
            dt = now - self.last_frame_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_frame_time = now

        perf_t0 = time.perf_counter()

        if self.live:
            new_samples = []
            try:
                while True:
                    new_samples.append(self.reader.sample_queue.get_nowait())
            except queue.Empty:
                pass
            n_new = min(len(new_samples), self.window_size)
            new = np.array(new_samples[-n_new:]) if n_new else None
        else:
            n_new = self.samples_per_frame
            new = generate_chunk(
                self.wave, self.n, self.samples_per_frame, self.fs,
                self.freq, self.freq2, self.args.sweep_period, self.noise,
            )

        if new is not None and n_new > 0:
            self.buffer[:-n_new] = self.buffer[n_new:]
            self.buffer[-n_new:] = new
            self.n += n_new

        # Live signals ride on a DC bias (e.g. a 1.65 V mid-supply front
        # end), which would otherwise dominate the display and leak into
        # nearby FFT bins through the Hann window's sidelobes. Removing the
        # window's mean AC-couples it in software.
        dc_bias = self.buffer.mean()
        display_buffer = self.buffer - dc_bias if self.live else self.buffer
        self.time_curve.setData(self.t_axis, display_buffer)

        windowed_signal = display_buffer * self.window_func
        if self.dsp_lab_mode:
            self.raw_signal_curve.setData(self.t_axis, self.buffer)
            self.windowed_signal_curve.setData(self.t_axis, windowed_signal)

        spectrum = np.fft.rfft(windowed_signal)
        mag_instant = np.abs(spectrum) * self.mag_scale
        power_instant = mag_instant ** 2
        if self.power_avg is None:
            self.power_avg = power_instant
        else:
            self.power_avg += self.spectrum_alpha * (power_instant - self.power_avg)
        mag = np.sqrt(self.power_avg)
        db = 20 * np.log10(mag + 1e-12)
        if self.log_freq_axis:
            # log10(0) is undefined -- the DC bin can't be plotted on a log
            # axis, so drop it here (it was never meaningful to look at
            # anyway once the DC bias is already removed above).
            self.freq_curve.setData(self.freqs[1:], db[1:])
        else:
            self.freq_curve.setData(self.freqs, db)

        np.maximum(self.peak_hold, db, out=self.peak_hold)
        if self.show_peak_hold:
            if self.log_freq_axis:
                self.peak_hold_curve.setData(self.freqs[1:], self.peak_hold[1:])
            else:
                self.peak_hold_curve.setData(self.freqs, self.peak_hold)

        self._update_cursor_readouts()

        perf_t_fft = time.perf_counter()

        peak_freq, peak_db, peak_idx = find_peak(db, self.window_size, self.fs)
        harmonics = find_harmonics(db, peak_idx, self.window_size, self.fs, max_harmonic=5)
        harmonic2 = harmonics.get(2)
        harmonic_idx = harmonic2[2] if harmonic2 else None
        snr_db, noise_floor_db = compute_snr(mag, peak_idx, harmonic_idx)

        # Raw phase is essentially random noise for bins that aren't
        # meaningfully above the noise floor; blank those out (NaN, which
        # pyqtgraph renders as a gap) rather than plotting a scribble.
        phase_deg = np.degrees(np.angle(spectrum))
        phase_deg = np.where(db >= noise_floor_db + 10.0, phase_deg, np.nan)
        if self.log_freq_axis:
            self.phase_curve.setData(self.freqs[1:], phase_deg[1:])
        else:
            self.phase_curve.setData(self.freqs, phase_deg)

        self.bode_mag_curve.setData(self.freqs, db)
        self.bode_phase_curve.setData(self.freqs, phase_deg)

        thd_percent, _thd_db = compute_thd(mag, peak_idx, harmonics)
        second_peak = find_second_peak(db, peak_idx, self.window_size, self.fs)
        sfdr_db = compute_sfdr(db, peak_idx, self.window_size, self.fs)
        amplitude_pp, rms, crest_factor = compute_time_domain_stats(display_buffer)
        sinad_db = compute_sinad(mag, peak_idx)
        enob = compute_enob(sinad_db)
        duty_cycle = compute_duty_cycle(display_buffer)
        waveform_shape = classify_waveform(mag, peak_idx, harmonics, noise_floor_db, crest_factor)

        # "Pure noise" mask for the Noise Analysis panel: DC, the
        # fundamental, and every harmonic up to Nyquist excluded (not just
        # the 5 shown in the Harmonics panel -- verified this matters: a
        # square wave's real harmonics extend well past the 5th and were
        # otherwise counted as "noise"). Even so, a harmonic-rich signal
        # (square/sawtooth/PWM) will still read a genuinely elevated noise
        # level here: its fast edges leak real broadband energy into every
        # bin, not just at the harmonics -- verified this is real signal
        # leakage, not a bug (a pure sine + known noise recovers the
        # correct value; window-function choice doesn't reduce it, ruling
        # out ordinary sidelobe leakage as the cause). These readings are
        # most trustworthy for sinusoidal or genuinely noise-like signals.
        noise_exclusion_harmonics = find_harmonics(db, peak_idx, self.window_size, self.fs, max_harmonic=60)
        noise_mask = np.ones(len(mag), dtype=bool)
        noise_mask[0] = False
        lo, hi = max(0, peak_idx - 2), min(len(mag), peak_idx + 2 + 1)
        noise_mask[lo:hi] = False
        for _h, (_h_freq, _h_db, h_idx) in noise_exclusion_harmonics.items():
            lo, hi = max(0, h_idx - 2), min(len(mag), h_idx + 2 + 1)
            noise_mask[lo:hi] = False
        rms_noise, noise_density, integrated_noise = compute_noise_metrics(
            self.power_avg, self.mag_scale, self.window_func, self.window_size, self.fs, noise_mask
        )
        self.peak_marker.setData([self._freq_to_axis_x(peak_freq)], [peak_db])
        if second_peak is not None:
            self.second_peak_marker.setData([self._freq_to_axis_x(second_peak[0])], [second_peak[1]])
        else:
            self.second_peak_marker.setData([], [])

        unit = "V" if self.live else ""
        self.fps_label.setText(f"{self.fps:.1f} FPS")
        if self.link_label is not None:
            bad = self.reader.packets_bad
            if bad > 0:
                self.link_label.setStyleSheet("color: #ef4444; font-size: 11px;")
                self.link_label.setText(f"Link: {self.reader.packets_ok} ok / {bad} CRC errors")
            else:
                self.link_label.setStyleSheet("color: #6b7280; font-size: 11px;")
                self.link_label.setText(f"Link: {self.reader.packets_ok} packets ok")

        if second_peak is not None:
            second_freq, second_db, _idx = second_peak
            second_line = f"2nd peak:  {second_freq:8.2f} Hz ({second_db:6.1f} dB)"
        else:
            second_line = "2nd peak:  n/a"

        self.signal_label.setText(
            "\n".join([
                f"Shape:     {waveform_shape}",
                f"Peak:      {peak_freq:8.2f} Hz ({peak_db:6.1f} dB)",
                second_line,
                f"DC bias:   {dc_bias:7.3f} {unit}",
                f"Amplitude: {amplitude_pp:7.3f} {unit}pp",
                f"RMS:       {rms:7.3f} {unit}",
                f"Crest fac: {crest_factor:7.2f}",
                f"Duty cyc:  {duty_cycle:6.2f} %",
            ])
        )

        harmonic_lines = []
        for h in range(2, 6):
            result = harmonics.get(h)
            ordinal = {2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}[h]
            if result is not None:
                h_freq, h_db, _idx = result
                harmonic_lines.append(f"{ordinal}: {h_freq:8.2f} Hz ({h_db:6.1f} dB)")
            else:
                harmonic_lines.append(f"{ordinal}: n/a")
        self.harmonics_label.setText("\n".join(harmonic_lines))

        thd_line = f"THD:   {thd_percent:6.2f} %" if thd_percent is not None else "THD:   n/a (fundamental too high -- harmonics exceed Nyquist)"
        self.quality_label.setText(
            "\n".join([
                f"SNR:   {snr_db:6.1f} dB",
                f"SINAD: {sinad_db:6.1f} dB",
                f"ENOB:  {enob:6.2f} bits",
                thd_line,
                f"SFDR:  {sfdr_db:6.1f} dB",
            ])
        )

        self.noise_analysis_label.setText(
            "\n".join([
                f"RMS noise:  {rms_noise:9.5f} {unit}",
                f"Density:    {format_density(noise_density)}",
                f"Integrated: {integrated_noise:9.5f} {unit}",
            ])
        )

        if self.dsp_lab_mode:
            raw_vpp = float(self.buffer.max() - self.buffer.min())
            windowed_vpp = float(windowed_signal.max() - windowed_signal.min())
            coherent_gain = float(np.sum(self.window_func) / self.window_size)
            pipeline_thd = f"{thd_percent:.2f} %" if thd_percent is not None else "n/a (see Signal Quality)"
            self.pipeline_label.setText(
                "\n".join([
                    f" 1. Raw Signal:      {raw_vpp:7.3f} {unit}pp",
                    f" 2. Remove DC offset: -{dc_bias:7.4f} {unit}",
                    f" 3. Window Function:  {self.window_name} (gain={coherent_gain:.3f})",
                    f" 4. Windowed Signal:  {windowed_vpp:7.3f} {unit}pp",
                    f" 5. FFT:              {self.window_size} samples -> {len(self.freqs)} bins",
                    f" 6. Power Spectrum:   peak {peak_db:6.1f} dB @ {peak_freq:8.2f} Hz",
                    f" 7. Peak Detection:   {peak_freq:8.2f} Hz ({peak_db:6.1f} dB)",
                    f" 8. Harmonic Detect:  {len(harmonics)} found (2nd-5th searched)",
                    f" 9. SINAD:            {sinad_db:6.1f} dB",
                    f"10. THD:              {pipeline_thd}",
                    f"11. ENOB:             {enob:6.2f} bits",
                ])
            )

        perf_t_detect = time.perf_counter()

        # Cepstrum/Goertzel/3D-FFT/2D-spectrogram below are each skipped
        # entirely while their panel is hidden -- all four defaulted off
        # (see the Graphs section), so paying their per-frame cost for a
        # panel nobody is looking at would be pure waste. Each one's own
        # Performance Benchmark stage correctly reads ~0 while hidden as a
        # direct, visible consequence of this.
        if self.cepstrum_plot.isVisible():
            cepstrum = compute_real_cepstrum(spectrum, self.window_size)
            half = self.window_size // 2
            self.cepstrum_curve.setData(self.quefrency_axis, cepstrum[:half])
            cepstrum_peak = find_cepstrum_peak(cepstrum, self.fs)
            if cepstrum_peak is not None:
                quefrency_s, equiv_freq_hz, quef_amp, _quef_idx = cepstrum_peak
                self.cepstrum_peak_marker.setData([quefrency_s], [quef_amp])
                self.cepstrum_label.setText(
                    "\n".join([
                        f"Peak quefrency: {quefrency_s * 1000:8.3f} ms",
                        f"-> periodicity: {equiv_freq_hz:8.2f} Hz",
                        f"Amplitude:      {quef_amp:8.3f}",
                    ])
                )
            else:
                self.cepstrum_peak_marker.setData([], [])
                self.cepstrum_label.setText("n/a (window too small)")

        perf_t_cepstrum = time.perf_counter()

        if self.goertzel_plot.isVisible():
            if self.goertzel_targets:
                goertzel_lines = []
                goertzel_db = np.empty(len(self.goertzel_targets))
                for i, target_freq in enumerate(self.goertzel_targets):
                    mag_g = compute_goertzel(windowed_signal, target_freq, self.fs, self.mag_scale)
                    db_g = 20 * np.log10(mag_g + 1e-12)
                    goertzel_db[i] = db_g
                    goertzel_lines.append(f"{target_freq:8.2f} Hz: {db_g:6.1f} dB")
                if self.goertzel_bar_item is not None:
                    self.goertzel_bar_item.setOpts(height=goertzel_db)
                self.goertzel_label.setText("\n".join(goertzel_lines))
            else:
                self.goertzel_label.setText("n/a (no valid target frequencies)")

        perf_t_goertzel = time.perf_counter()

        if self.duty_cycle_mode:
            pulse = compute_pulse_metrics(display_buffer, self.fs)
            if pulse is not None:
                self.duty_cycle_label.setText(
                    "\n".join([
                        f"Duty cycle:  {pulse['duty_percent']:7.2f} %",
                        f"HIGH time:   {pulse['high_time'] * 1000:9.4f} ms",
                        f"LOW time:    {pulse['low_time'] * 1000:9.4f} ms",
                        f"Rise time:   {pulse['rise_time'] * 1e6:9.2f} us",
                        f"Fall time:   {pulse['fall_time'] * 1e6:9.2f} us",
                        f"Pulse width: {pulse['pulse_width'] * 1000:9.4f} ms",
                        f"Period:      {pulse['period'] * 1000:9.4f} ms",
                        f"Frequency:   {pulse['frequency']:9.2f} Hz",
                    ])
                )
            else:
                self.duty_cycle_label.setText("n/a (signal is flat, or window has too few edges)")

        perf_t_duty_cycle = time.perf_counter()

        # spec_history feeds both the 2D spectrogram image and the 3D
        # waterfall, so it's only worth rolling forward if at least one of
        # them is actually visible; each consumer is then further gated
        # individually below.
        spec_plot_visible = self.spec_plot.isVisible()
        fft3d_visible = self.fft3d_surface is not None and self.fft3d_plot.isVisible()
        if spec_plot_visible or fft3d_visible:
            self.spec_history[:-1] = self.spec_history[1:]
            self.spec_history[-1] = db

            if spec_plot_visible:
                self.spec_image.setImage(self.spec_history, autoLevels=False)

            if fft3d_visible:
                # Frequency axis decimated (see GL3D_MAX_FREQ_BINS); time
                # axis (SPECTROGRAM_HISTORY=120) is already small, kept as-is.
                z = self.spec_history[:, :: self.gl3d_freq_stride].T  # (n_freq_ds, SPECTROGRAM_HISTORY)
                z_norm = np.clip((z + 100.0) / 120.0, 0.0, 1.0)  # -100..20 dB -> 0..1
                colors = self.fft3d_colormap.map(z_norm, mode="float")
                # Axes rescaled to a common ~0-10 visual range purely so the
                # surface reads as a legible 3D shape -- GLViewWidget has no
                # tick labels, so this view is a qualitative companion to the
                # calibrated 2D spectrogram above, not a substitute for it.
                x_display = self.freqs[:: self.gl3d_freq_stride] / max(self.fs / 2, 1e-9) * 10.0
                y_display = (self.noise_time_axis - self.noise_time_axis[0]) / max(-self.noise_time_axis[0], 1e-9) * 10.0
                z_display = z_norm * 4.0
                self.fft3d_surface.setData(x=x_display, y=y_display, z=z_display, colors=colors)

        perf_t_spectrogram = time.perf_counter()

        self.noise_floor_history[:-1] = self.noise_floor_history[1:]
        self.noise_floor_history[-1] = noise_floor_db
        self.snr_history[:-1] = self.snr_history[1:]
        self.snr_history[-1] = snr_db
        self.noise_floor_curve.setData(self.noise_time_axis, self.noise_floor_history)
        self.snr_curve.setData(self.noise_time_axis, self.snr_history)

        temp_c = self.reader.temp_c if (self.live and self.reader is not None and self.reader.temp_c is not None) else float("nan")
        drift_values = {
            "Frequency": peak_freq,
            "DC Bias": dc_bias,
            "Noise Floor": noise_floor_db,
            "THD": thd_percent if thd_percent is not None else np.nan,
            "SINAD": sinad_db,
            "Die Temperature": temp_c,
        }
        for name, value in drift_values.items():
            hist = self.drift_histories[name]
            hist[:-1] = hist[1:]
            hist[-1] = value
        self.drift_curve.setData(self.noise_time_axis, self.drift_histories[self.drift_metric])

        self.last_snapshot = dict(
            t_axis=self.t_axis,
            buffer=display_buffer.copy(),
            freqs=self.freqs,
            mag=mag,
            db=db,
            peak_freq=peak_freq,
            peak_db=peak_db,
            harmonics=harmonics,
            second_peak=second_peak,
            snr_db=snr_db,
            noise_floor_db=noise_floor_db,
            thd_percent=thd_percent,
            sfdr_db=sfdr_db,
            amplitude_pp=amplitude_pp,
            rms=rms,
            crest_factor=crest_factor,
            dc_bias=dc_bias,
            sinad_db=sinad_db,
            enob=enob,
            duty_cycle=duty_cycle,
            waveform_shape=waveform_shape,
            rms_noise=rms_noise,
            noise_density=noise_density,
            integrated_noise=integrated_noise,
        )

        perf_t_end = time.perf_counter()
        self.perf_stage_ms = {
            "Total": (perf_t_end - perf_t0) * 1000.0,
            "Acquire+FFT": (perf_t_fft - perf_t0) * 1000.0,
            "Detection": (perf_t_detect - perf_t_fft) * 1000.0,
            "Cepstrum": (perf_t_cepstrum - perf_t_detect) * 1000.0,
            "Goertzel": (perf_t_goertzel - perf_t_cepstrum) * 1000.0,
            "Duty Cycle": (perf_t_duty_cycle - perf_t_goertzel) * 1000.0,
            "Spectrogram/3D": (perf_t_spectrogram - perf_t_duty_cycle) * 1000.0,
            "Other": (perf_t_end - perf_t_spectrogram) * 1000.0,
        }
        # Both blocks below are skipped while their own panel is hidden --
        # updating a chart/readout nobody can see is wasted work, and (for
        # CPU/RAM) skips a real psutil syscall every frame too.
        if self.perf_plot.isVisible():
            for name, ms in self.perf_stage_ms.items():
                hist = self.perf_histories[name]
                hist[:-1] = hist[1:]
                hist[-1] = ms
                self.perf_curves[name].setData(self.noise_time_axis, hist)

            budget_ms = 1000.0 / self.args.fps
            total_ms = self.perf_stage_ms["Total"]
            perf_lines = [f"Total:          {total_ms:7.2f} ms ({total_ms / budget_ms * 100:5.1f}% of {budget_ms:.1f} ms budget)"]
            for name in PERF_STAGE_NAMES[1:]:
                perf_lines.append(f"{name:<15} {self.perf_stage_ms[name]:7.2f} ms")
            perf_lines.append(f"Effective FPS:  {self.fps:5.1f}")
            self.perf_label.setText("\n".join(perf_lines))

        if self.sysres_plot.isVisible():
            if self.psutil_process is not None:
                try:
                    cpu_percent = self.psutil_process.cpu_percent(None)
                    ram_mb = self.psutil_process.memory_info().rss / (1024.0 * 1024.0)
                except Exception:
                    cpu_percent, ram_mb = float("nan"), float("nan")
                self.system_label.setText(f"CPU usage: {cpu_percent:6.1f} %\nRAM usage: {ram_mb:8.1f} MB")
            else:
                cpu_percent, ram_mb = float("nan"), float("nan")
                self.system_label.setText("n/a (psutil not installed)")

            self.cpu_history[:-1] = self.cpu_history[1:]
            self.cpu_history[-1] = cpu_percent
            self.ram_history[:-1] = self.ram_history[1:]
            self.ram_history[-1] = ram_mb
            self.cpu_curve.setData(self.noise_time_axis, self.cpu_history)
            self.ram_curve.setData(self.noise_time_axis, self.ram_history)

    def closeEvent(self, event):
        self._save_settings()
        if self.live and self.reader is not None:
            self.reader.stop()
        super().closeEvent(event)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wave", choices=WAVE_TYPES, default="demo")
    p.add_argument("--freq", type=float, default=10.0, help="base frequency (Hz)")
    p.add_argument("--freq2", type=float, default=60.0, help="chirp target frequency (Hz)")
    p.add_argument("--sweep-period", type=float, default=8.0, help="chirp sweep period (s)")
    p.add_argument("--samplerate", type=float, default=2000.0, help="sample rate (Hz)")
    p.add_argument("--window", type=int, default=2048, help="FFT / display window size (samples)")
    p.add_argument(
        "--fft-window",
        dest="fft_window",
        choices=list(WINDOW_FUNCTIONS.keys()),
        default="Hann",
        help="FFT window function (also switchable live from the sidebar)",
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--noise", type=float, default=0.03, help="initial noise std dev")
    p.add_argument(
        "--averaging",
        type=float,
        default=70.0,
        help="spectrum trace averaging, 0 (raw/instant) to ~99 (heavy smoothing); "
        "also live-adjustable from the sidebar",
    )
    p.add_argument(
        "--serial",
        nargs="?",
        const="__PICK__",
        default=None,
        metavar="PORT",
        help="read live ADC samples from main.c instead of generating a synthetic signal. "
        "Give a port directly (e.g. COM5, /dev/ttyUSB0), or pass --serial with no value "
        "to pick one from a GUI list.",
    )
    p.add_argument("--baud", type=int, default=3000000, help="serial baud rate for --serial (must match CONFIG_ESP_CONSOLE_UART_BAUDRATE in main.c's sdkconfig)")
    args = p.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    live = args.serial is not None
    reader = None
    port_label = args.serial
    fs = args.samplerate

    if live:
        if args.serial == "__PICK__":
            dialog = SerialPortDialog(args.baud)
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                sys.exit(0)
            reader, port_label = dialog.reader, dialog.port
            fs = float(reader.sample_rate)
        else:
            reader = SerialReader(args.serial, args.baud)
            reader.start()
            wait_start = time.time()
            while reader.sample_rate is None and time.time() - wait_start < CONNECT_TIMEOUT_S:
                time.sleep(0.05)
            if reader.sample_rate is None:
                reader.stop()
                raise SystemExit(
                    f"No META packet received from {args.serial} within {CONNECT_TIMEOUT_S:.0f}s "
                    "-- check the port/baud, or the board may still be booting. "
                    "(Opening the port resets the ESP32; it then takes ~1s to measure its "
                    "sample rate before it sends anything.)"
                )
            fs = float(reader.sample_rate)

    window = FFTBenchWindow(args, live, reader, port_label, fs)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

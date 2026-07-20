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
from pyqtgraph.Qt import QtCore, QtWidgets

try:
    import serial
except ImportError:
    serial = None

WAVE_TYPES = ["demo", "sine", "chirp", "square", "sawtooth", "noise"]
SPECTROGRAM_HISTORY = 120

MAGIC_META = b"META"
MAGIC_DATA = b"DATA"


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
                if len(buf) < 8:
                    return
                (self.sample_rate,) = struct.unpack_from("<I", buf, 4)
                del buf[:8]
            else:  # MAGIC_DATA
                if len(buf) < 6:
                    return
                (count,) = struct.unpack_from("<H", buf, 4)
                needed = 6 + count * 2
                if len(buf) < needed:
                    return
                samples_mv = struct.unpack_from(f"<{count}h", buf, 6)
                for mv in samples_mv:
                    self.sample_queue.put(mv / 1000.0)  # convert mV to V
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

        self.n = 0
        self.wave = args.wave
        self.freq = args.freq
        self.freq2 = args.freq2
        self.noise = args.noise
        self.fps = float(args.fps)
        self.last_frame_time = None
        self.last_snapshot = None

        # Power-domain exponential averaging of the spectrum (like a real
        # analyzer's "trace averaging"), so the displayed curve and all the
        # readings derived from it settle down instead of jumping every
        # frame -- especially needed at high sample rates, where each
        # window covers a short, disjoint (non-overlapping) slice of
        # signal. Averaging happens in power (mag**2), not dB, since
        # averaging logs is statistically biased.
        self.spectrum_alpha = 1.0 - args.averaging / 100.0
        self.power_avg = None

        title = f"FFT Test Bench — Live ESP32 ADC ({port_label} @ {fs:.0f} Hz)" if live else "FFT Test Bench"
        self.setWindowTitle(title)
        self._build_ui(title)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / args.fps))

    def _recompute_window(self):
        self.window_func = WINDOW_FUNCTIONS[self.window_name](self.window_size)
        self.mag_scale = 2.0 / np.sum(self.window_func)
        # A different window changes the spectrum's scale/shape, so blending
        # it into the old running average would produce a misleading
        # transient; just restart averaging from the next frame instead.
        self.power_avg = None

    # -- UI construction -----------------------------------------------

    def _build_ui(self, title):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

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
        root.addWidget(self._build_plots(), 1)

        self.resize(1320, 900)

    def _build_sidebar(self, title):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(14)

        mode_label = QtWidgets.QLabel(title)
        mode_label.setObjectName("modeLabel")
        mode_label.setWordWrap(True)
        layout.addWidget(mode_label)

        window_title = QtWidgets.QLabel("FFT Window")
        window_title.setObjectName("sectionTitle")
        layout.addWidget(window_title)
        self.window_combo = QtWidgets.QComboBox()
        self.window_combo.addItems(list(WINDOW_FUNCTIONS.keys()))
        self.window_combo.setCurrentText(self.window_name)
        self.window_combo.currentTextChanged.connect(self._on_window_change)
        layout.addWidget(self.window_combo)
        window_hint = QtWidgets.QLabel(
            "Narrow lobe (Rectangular/Hann) resolves close tones; wide lobe\n"
            "(Flat-top) gives the most accurate amplitude/SFDR reading."
        )
        window_hint.setStyleSheet("color: #6b7280; font-size: 11px;")
        window_hint.setWordWrap(True)
        layout.addWidget(window_hint)

        self.averaging_label = QtWidgets.QLabel(f"Averaging: {self.args.averaging:.0f}%")
        layout.addWidget(self.averaging_label)
        self.averaging_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.averaging_slider.setRange(0, 99)
        self.averaging_slider.setValue(int(self.args.averaging))
        self.averaging_slider.valueChanged.connect(self._on_averaging_change)
        layout.addWidget(self.averaging_slider)
        averaging_hint = QtWidgets.QLabel(
            "Smooths the spectrum trace and readings across frames — raise\n"
            "this if the display looks jumpy (common at high sample rates)."
        )
        averaging_hint.setStyleSheet("color: #6b7280; font-size: 11px;")
        averaging_hint.setWordWrap(True)
        layout.addWidget(averaging_hint)

        wave_title = QtWidgets.QLabel("Waveform")
        wave_title.setObjectName("sectionTitle")
        layout.addWidget(wave_title)
        self.wave_combo = QtWidgets.QComboBox()
        self.wave_combo.addItems(WAVE_TYPES)
        self.wave_combo.setCurrentText(self.wave)
        self.wave_combo.currentTextChanged.connect(self._on_wave_change)
        layout.addWidget(self.wave_combo)

        self.freq_label = QtWidgets.QLabel(f"Frequency: {self.freq:.0f} Hz")
        layout.addWidget(self.freq_label)
        self.freq_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        freq_max = max(2, int(min(self.fs / 2, 500)))
        self.freq_slider.setRange(1, freq_max)
        self.freq_slider.setValue(int(min(max(self.freq, 1), freq_max)))
        self.freq_slider.valueChanged.connect(self._on_freq_change)
        layout.addWidget(self.freq_slider)

        self.noise_label = QtWidgets.QLabel(f"Noise: {self.noise:.2f}")
        layout.addWidget(self.noise_label)
        self.noise_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.noise_slider.setRange(0, 50)
        self.noise_slider.setValue(int(self.noise * 100))
        self.noise_slider.valueChanged.connect(self._on_noise_change)
        layout.addWidget(self.noise_slider)

        if self.live:
            # These controls only affect the synthetic generator; disable
            # rather than leave them present-but-inert on live ADC data.
            for w in (self.wave_combo, self.freq_slider, self.noise_slider):
                w.setEnabled(False)

        layout.addSpacing(6)
        save_title = QtWidgets.QLabel("Snapshot")
        save_title.setObjectName("sectionTitle")
        layout.addWidget(save_title)
        self.save_button = QtWidgets.QPushButton("Save CSV")
        self.save_button.setObjectName("saveButton")
        self.save_button.clicked.connect(self._on_save_click)
        layout.addWidget(self.save_button)
        self.save_status_label = QtWidgets.QLabel("")
        self.save_status_label.setStyleSheet(f"color: {ACCENT_OK};")
        layout.addWidget(self.save_status_label)

        self.fps_label = QtWidgets.QLabel("")
        self.fps_label.setStyleSheet("color: #6b7280; font-size: 11px;")
        layout.addWidget(self.fps_label)

        self.signal_label = self._add_stats_section(layout, "Signal")
        self.harmonics_label = self._add_stats_section(layout, "Harmonics")
        self.quality_label = self._add_stats_section(layout, "Signal Quality")

        layout.addStretch(1)
        return panel

    def _add_stats_section(self, layout, title):
        section_title = QtWidgets.QLabel(title)
        section_title.setObjectName("sectionTitle")
        layout.addWidget(section_title)
        label = QtWidgets.QLabel("—")
        label.setObjectName("statsLabel")
        label.setWordWrap(True)
        layout.addWidget(label)
        return label

    def _build_plots(self):
        pg.setConfigOptions(antialias=True)
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setSpacing(10)

        # Time domain
        self.time_plot = pg.PlotWidget(background=PANEL_BG)
        apply_plot_theme(
            self.time_plot.getPlotItem(),
            "Time domain",
            "Time in window (s)",
            "AC amplitude (V, bias removed)" if self.live else "Amplitude",
        )
        self.time_curve = self.time_plot.plot(self.t_axis, self.buffer, pen=pg.mkPen(ACCENT_TIME, width=1.5))
        self.time_plot.setXRange(0, self.window_size / self.fs, padding=0)
        self.time_plot.setYRange(*((-1.8, 1.8) if self.live else (-2.2, 2.2)))
        vbox.addWidget(self.time_plot, 1)

        # Frequency domain, with a hover crosshair for reading values off
        # the curve (a plain static line is hard to read precisely).
        self.freq_plot = pg.PlotWidget(background=PANEL_BG)
        apply_plot_theme(self.freq_plot.getPlotItem(), "Frequency domain", "Frequency (Hz)", "Magnitude (dB)")
        self.freq_curve = self.freq_plot.plot(self.freqs, np.full_like(self.freqs, -100.0), pen=pg.mkPen(ACCENT_FREQ, width=1.5))
        self.freq_plot.setXRange(0, self.fs / 2, padding=0)
        self.freq_plot.setYRange(-100, 20, padding=0)
        self.peak_marker = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(ACCENT_FREQ), pen=pg.mkPen(TEXT_FG, width=1))
        self.freq_plot.addItem(self.peak_marker)
        self.second_peak_marker = pg.ScatterPlotItem(
            size=9, symbol="t", brush=pg.mkBrush(ACCENT_SNR), pen=pg.mkPen(TEXT_FG, width=1)
        )
        self.freq_plot.addItem(self.second_peak_marker)
        self._add_crosshair(self.freq_plot, "Hz", "dB")
        vbox.addWidget(self.freq_plot, 1)

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
        vbox.addWidget(self.spec_plot, 1)

        # Noise floor / SNR trend
        self.noise_plot = pg.PlotWidget(background=PANEL_BG)
        apply_plot_theme(self.noise_plot.getPlotItem(), "Noise floor & SNR trend", "Time (s ago)", "dB")
        self.noise_floor_curve = self.noise_plot.plot(
            self.noise_time_axis, self.noise_floor_history, pen=pg.mkPen(ACCENT_NOISE_FLOOR, width=1.5), name="Noise floor (dB)"
        )
        self.snr_curve = self.noise_plot.plot(
            self.noise_time_axis, self.snr_history, pen=pg.mkPen(ACCENT_SNR, width=1.5), name="SNR (dB)"
        )
        self.noise_plot.addLegend(offset=(10, 10))
        self.noise_plot.setXRange(self.noise_time_axis[0], self.noise_time_axis[-1], padding=0)
        self.noise_plot.setYRange(-100, 60, padding=0)
        vbox.addWidget(self.noise_plot, 1)

        return container

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
            label.setText(f"{x:.1f} {x_unit}, {y:.1f} {y_unit}")
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

    # -- Frame update -------------------------------------------------------

    def update_frame(self):
        now = time.perf_counter()
        if self.last_frame_time is not None:
            dt = now - self.last_frame_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_frame_time = now

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

        spectrum = np.fft.rfft(display_buffer * self.window_func)
        mag_instant = np.abs(spectrum) * self.mag_scale
        power_instant = mag_instant ** 2
        if self.power_avg is None:
            self.power_avg = power_instant
        else:
            self.power_avg += self.spectrum_alpha * (power_instant - self.power_avg)
        mag = np.sqrt(self.power_avg)
        db = 20 * np.log10(mag + 1e-12)
        self.freq_curve.setData(self.freqs, db)

        peak_freq, peak_db, peak_idx = find_peak(db, self.window_size, self.fs)
        harmonics = find_harmonics(db, peak_idx, self.window_size, self.fs, max_harmonic=5)
        harmonic2 = harmonics.get(2)
        harmonic_idx = harmonic2[2] if harmonic2 else None
        snr_db, noise_floor_db = compute_snr(mag, peak_idx, harmonic_idx)
        thd_percent, _thd_db = compute_thd(mag, peak_idx, harmonics)
        second_peak = find_second_peak(db, peak_idx, self.window_size, self.fs)
        sfdr_db = compute_sfdr(db, peak_idx, self.window_size, self.fs)
        amplitude_pp, rms, crest_factor = compute_time_domain_stats(display_buffer)
        sinad_db = compute_sinad(mag, peak_idx)
        enob = compute_enob(sinad_db)
        self.peak_marker.setData([peak_freq], [peak_db])
        if second_peak is not None:
            self.second_peak_marker.setData([second_peak[0]], [second_peak[1]])
        else:
            self.second_peak_marker.setData([], [])

        unit = "V" if self.live else ""
        self.fps_label.setText(f"{self.fps:.1f} FPS")

        if second_peak is not None:
            second_freq, second_db, _idx = second_peak
            second_line = f"2nd peak:  {second_freq:8.2f} Hz ({second_db:6.1f} dB)"
        else:
            second_line = "2nd peak:  n/a"

        self.signal_label.setText(
            "\n".join([
                f"Peak:      {peak_freq:8.2f} Hz ({peak_db:6.1f} dB)",
                second_line,
                f"DC bias:   {dc_bias:7.3f} {unit}",
                f"Amplitude: {amplitude_pp:7.3f} {unit}pp",
                f"RMS:       {rms:7.3f} {unit}",
                f"Crest fac: {crest_factor:7.2f}",
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

        self.spec_history[:-1] = self.spec_history[1:]
        self.spec_history[-1] = db
        self.spec_image.setImage(self.spec_history, autoLevels=False)

        self.noise_floor_history[:-1] = self.noise_floor_history[1:]
        self.noise_floor_history[-1] = noise_floor_db
        self.snr_history[:-1] = self.snr_history[1:]
        self.snr_history[-1] = snr_db
        self.noise_floor_curve.setData(self.noise_time_axis, self.noise_floor_history)
        self.snr_curve.setData(self.noise_time_axis, self.snr_history)

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
        )

    def closeEvent(self, event):
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

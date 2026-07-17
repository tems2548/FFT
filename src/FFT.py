"""
Real-time FFT test bench: scrolling time-domain waveform next to its
live frequency-domain spectrum, driven by a synthetic dynamic test signal.

Run:
    python fft_visualizer.py
    python fft_visualizer.py --wave chirp --freq 20 --freq2 80
    python fft_visualizer.py --wave sine --freq 440 --samplerate 44100

Use the radio buttons to switch waveform live, and the sliders to change
the base frequency / sweep target / noise level while it's running.

Or analyze a live ESP32 ADC feed instead of a synthetic signal (see
main.c, which streams decimated ADC samples over the same USB-UART used
for flashing/logging):
    python FFT.py --serial              # pick the port from a GUI list
    python FFT.py --serial COM5
    python fft_visualizer.py --serial /dev/ttyUSB0 --baud 115200
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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons, Slider

try:
    import serial
except ImportError:
    serial = None

WAVE_TYPES = ["demo", "sine", "chirp", "square", "sawtooth", "noise"]
SPECTROGRAM_HISTORY = 120

MAGIC_META = b"META"
MAGIC_DATA = b"DATA"


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
                    self.sample_queue.put(mv / 1000.0)
                del buf[:needed]


CONNECT_TIMEOUT_S = 6.0  # ESP32 reboots on port-open + runs a 1s rate
                          # measurement before its first META packet.


def pick_serial_port_gui(default_baud):
    """Tkinter dialog to pick a serial port and connect to the ESP32.

    Blocks until the user connects successfully (returning the live
    SerialReader) or closes the window (exiting the process) — Tkinter is
    stdlib so this adds no new dependency.
    """
    import tkinter as tk
    from tkinter import ttk

    import serial.tools.list_ports as list_ports

    result = {}
    root = tk.Tk()
    root.title("Connect to ESP32")
    root.resizable(False, False)

    tk.Label(root, text="Serial port:").grid(row=0, column=0, sticky="w", padx=8, pady=(10, 2))
    port_var = tk.StringVar()
    port_combo = ttk.Combobox(root, textvariable=port_var, width=42, state="readonly")
    port_combo.grid(row=0, column=1, columnspan=2, padx=8, pady=(10, 2))

    def refresh_ports():
        ports = [f"{p.device} — {p.description}" for p in list_ports.comports()]
        port_combo["values"] = ports
        if ports and port_var.get() not in ports:
            port_combo.current(0)

    tk.Button(root, text="Refresh", command=refresh_ports).grid(row=0, column=3, padx=8)
    refresh_ports()

    tk.Label(root, text="Baud rate:").grid(row=1, column=0, sticky="w", padx=8, pady=2)
    baud_var = tk.StringVar(value=str(default_baud))
    tk.Entry(root, textvariable=baud_var, width=14).grid(row=1, column=1, sticky="w", padx=8, pady=2)

    status_var = tk.StringVar(value="Select a port and click Connect.")
    tk.Label(root, textvariable=status_var, fg="#555", wraplength=380, justify="left").grid(
        row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(6, 8)
    )

    def on_connect():
        selection = port_var.get()
        if not selection:
            status_var.set("No port selected.")
            return
        port = selection.split(" — ")[0]
        try:
            baud = int(baud_var.get())
        except ValueError:
            status_var.set("Baud rate must be an integer.")
            return

        connect_btn.config(state="disabled")
        status_var.set(f"Connecting to {port} @ {baud}...")

        try:
            reader = SerialReader(port, baud)
            reader.start()
        except Exception as exc:
            status_var.set(f"Failed to open {port}: {exc}")
            connect_btn.config(state="normal")
            return

        deadline = time.time() + CONNECT_TIMEOUT_S

        def poll():
            if reader.sample_rate is not None:
                result["reader"] = reader
                result["port"] = port
                root.destroy()
                return
            if time.time() > deadline:
                reader.stop()
                status_var.set(
                    f"No data from {port} within {CONNECT_TIMEOUT_S:.0f}s — "
                    "check the port/baud, or the board may still be booting. Try again."
                )
                connect_btn.config(state="normal")
                return
            root.after(100, poll)

        poll()

    connect_btn = tk.Button(root, text="Connect", command=on_connect, width=14)
    connect_btn.grid(row=3, column=0, columnspan=4, pady=(0, 10))

    root.protocol("WM_DELETE_WINDOW", lambda: (root.destroy(), sys.exit(0)))
    root.mainloop()

    if "reader" not in result:
        sys.exit(0)
    return result["reader"], result["port"]


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


def compute_thd(mag, peak_idx, harmonics):
    """Total Harmonic Distortion: RMS of the given harmonics relative to
    the fundamental amplitude, as a percentage and in dB.

    Takes the harmonics dict from find_harmonics rather than searching
    again, so THD stays consistent with the on-screen readings for free.
    """
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
    harmonic_freq,
    harmonic_db,
    snr_db,
    noise_floor_db,
    thd_percent,
):
    """Write the current time/frequency snapshot plus summary stats to CSV."""
    fname = f"fft_snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# peak_freq_hz", f"{peak_freq:.4f}"])
        writer.writerow(["# peak_db", f"{peak_db:.2f}"])
        writer.writerow(["# harmonic2_freq_hz", f"{harmonic_freq:.4f}" if harmonic_freq is not None else ""])
        writer.writerow(["# harmonic2_db", f"{harmonic_db:.2f}" if harmonic_db is not None else ""])
        writer.writerow(["# snr_db", f"{snr_db:.2f}"])
        writer.writerow(["# noise_floor_db", f"{noise_floor_db:.2f}"])
        writer.writerow(["# thd_percent", f"{thd_percent:.3f}"])
        writer.writerow([])
        writer.writerow(["time_s", "amplitude", "freq_hz", "magnitude", "magnitude_db"])
        for row in itertools.zip_longest(t_axis, buffer, freqs, mag, db, fillvalue=""):
            writer.writerow(row)
    return fname


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wave", choices=WAVE_TYPES, default="demo")
    p.add_argument("--freq", type=float, default=10.0, help="base frequency (Hz)")
    p.add_argument("--freq2", type=float, default=60.0, help="chirp target frequency (Hz)")
    p.add_argument("--sweep-period", type=float, default=8.0, help="chirp sweep period (s)")
    p.add_argument("--samplerate", type=float, default=2000.0, help="sample rate (Hz)")
    p.add_argument("--window", type=int, default=2048, help="FFT / display window size (samples)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--noise", type=float, default=0.03, help="initial noise std dev")
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
    p.add_argument("--baud", type=int, default=115200, help="serial baud rate for --serial")
    args = p.parse_args()

    live = args.serial is not None
    reader = None
    port_label = args.serial
    fs = args.samplerate
    if live:
        if args.serial == "__PICK__":
            reader, port_label = pick_serial_port_gui(args.baud)
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

    window_size = args.window
    samples_per_frame = max(1, int(fs / args.fps))

    buffer = np.zeros(window_size)
    hann = np.hanning(window_size)
    mag_scale = 2.0 / np.sum(hann)
    state = {
        "n": 0,
        "wave": args.wave,
        "freq": args.freq,
        "freq2": args.freq2,
        "noise": args.noise,
        "fps": float(args.fps),
        "last_frame_time": None,
        "live": live,
        "reader": reader,
    }

    fig = plt.figure(figsize=(10, 11))
    fig.suptitle(
        f"FFT Test Bench — Live ESP32 ADC ({port_label} @ {fs:.0f} Hz)"
        if live
        else "FFT Test Bench — Time / Frequency / Spectrogram / Noise"
    )
    ax_time = fig.add_axes([0.32, 0.80, 0.63, 0.16])
    ax_freq = fig.add_axes([0.32, 0.59, 0.63, 0.16])
    ax_spec = fig.add_axes([0.32, 0.38, 0.63, 0.16])
    ax_noise = fig.add_axes([0.32, 0.17, 0.63, 0.16])

    t_axis = np.linspace(0, window_size / fs, window_size, endpoint=False)
    (line_time,) = ax_time.plot(t_axis, buffer, color="#3b82f6", lw=1)
    ax_time.set_xlim(0, window_size / fs)
    if live:
        # DC bias (e.g. the 1.65 V mid-supply bias of an AC-coupled front
        # end) is subtracted per-frame before display/FFT, so this is a
        # fixed range around 0 V sized for the ADC's ~3.3 V full scale.
        ax_time.set_ylim(-1.8, 1.8)
        ax_time.set_ylabel("AC amplitude (V, bias removed)")
    else:
        ax_time.set_ylim(-2.2, 2.2)
        ax_time.set_ylabel("Amplitude")
    ax_time.set_xlabel("Time in window (s)")
    ax_time.set_title("Time domain")
    ax_time.grid(alpha=0.3)
    fps_text = ax_time.text(
        0.99, 0.95, "", transform=ax_time.transAxes, ha="right", va="top", fontsize=9, family="monospace"
    )

    freqs = np.fft.rfftfreq(window_size, d=1 / fs)
    (line_freq,) = ax_freq.plot(freqs, np.full_like(freqs, -100.0), color="#f97316", lw=1)
    ax_freq.set_xlim(0, fs / 2)
    ax_freq.set_ylim(-100, 20)
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Magnitude (dB)")
    ax_freq.set_title("Frequency domain")
    ax_freq.grid(alpha=0.3)
    peak_text = ax_freq.text(
        0.99, 0.95, "", transform=ax_freq.transAxes, ha="right", va="top", fontsize=9, family="monospace"
    )

    spec_history = np.full((len(freqs), SPECTROGRAM_HISTORY), -100.0)
    im_spec = ax_spec.imshow(
        spec_history,
        aspect="auto",
        origin="lower",
        extent=[-SPECTROGRAM_HISTORY / args.fps, 0, 0, fs / 2],
        cmap="magma",
        vmin=-100,
        vmax=20,
        interpolation="nearest",
    )
    ax_spec.set_xlabel("Time (s ago)")
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_title("Spectrogram")

    NOISE_HISTORY = SPECTROGRAM_HISTORY
    noise_time_axis = np.linspace(-NOISE_HISTORY / args.fps, 0, NOISE_HISTORY)
    noise_floor_history = np.full(NOISE_HISTORY, -100.0)
    snr_history = np.full(NOISE_HISTORY, 0.0)
    (line_noise_floor,) = ax_noise.plot(
        noise_time_axis, noise_floor_history, color="#64748b", lw=1.2, label="Noise floor (dB)"
    )
    (line_snr,) = ax_noise.plot(noise_time_axis, snr_history, color="#22c55e", lw=1.2, label="SNR (dB)")
    ax_noise.set_xlim(noise_time_axis[0], noise_time_axis[-1])
    ax_noise.set_ylim(-100, 60)
    ax_noise.set_xlabel("Time (s ago)")
    ax_noise.set_ylabel("dB")
    ax_noise.set_title("Noise floor & SNR trend")
    ax_noise.grid(alpha=0.3)
    ax_noise.legend(loc="upper left", fontsize=8)

    ax_radio = fig.add_axes([0.03, 0.68, 0.18, 0.24])
    ax_radio.set_title("Waveform", fontsize=10)
    radio = RadioButtons(ax_radio, WAVE_TYPES, active=WAVE_TYPES.index(args.wave))

    ax_freq_slider = fig.add_axes([0.32, 0.04, 0.63, 0.02])
    freq_slider = Slider(ax_freq_slider, "freq (Hz)", 1, min(fs / 2, 500), valinit=args.freq)

    ax_noise_slider = fig.add_axes([0.03, 0.60, 0.18, 0.02])
    noise_slider = Slider(ax_noise_slider, "noise", 0.0, 0.5, valinit=args.noise)

    ax_save_button = fig.add_axes([0.03, 0.45, 0.18, 0.05])
    save_button = Button(ax_save_button, "Save CSV")
    save_status = fig.text(0.03, 0.42, "", fontsize=8, color="#16a34a")

    def on_wave_select(label):
        state["wave"] = label

    def on_freq_change(val):
        state["freq"] = val
        state["freq2"] = val * 3

    def on_noise_change(val):
        state["noise"] = val

    def on_save_click(_event):
        last = state.get("last")
        if last is None:
            return
        fname = save_snapshot_csv(**last)
        save_status.set_text(f"Saved {fname}")
        fig.canvas.draw_idle()

    radio.on_clicked(on_wave_select)
    freq_slider.on_changed(on_freq_change)
    noise_slider.on_changed(on_noise_change)
    save_button.on_clicked(on_save_click)

    def update(_frame):
        nonlocal spec_history
        now = time.perf_counter()
        if state["last_frame_time"] is not None:
            dt = now - state["last_frame_time"]
            if dt > 0:
                inst_fps = 1.0 / dt
                state["fps"] = 0.9 * state["fps"] + 0.1 * inst_fps
        state["last_frame_time"] = now
        fps_text.set_text(f"{state['fps']:.1f} FPS")

        if state["live"]:
            new_samples = []
            try:
                while True:
                    new_samples.append(state["reader"].sample_queue.get_nowait())
            except queue.Empty:
                pass
            n_new = min(len(new_samples), window_size)
            new = np.array(new_samples[-n_new:]) if n_new else None
        else:
            n_new = samples_per_frame
            new = generate_chunk(
                state["wave"],
                state["n"],
                samples_per_frame,
                fs,
                state["freq"],
                state["freq2"],
                args.sweep_period,
                state["noise"],
            )

        if new is not None and n_new > 0:
            # In-place shift (no new array allocation, unlike np.roll) then
            # drop the new chunk into the vacated tail.
            buffer[:-n_new] = buffer[n_new:]
            buffer[-n_new:] = new
            state["n"] += n_new

        # Live signals ride on a DC bias (e.g. a 1.65 V mid-supply front
        # end), which would otherwise dominate the display and leak into
        # nearby FFT bins through the Hann window's sidelobes. Removing the
        # window's mean AC-couples it in software.
        display_buffer = buffer - buffer.mean() if state["live"] else buffer

        line_time.set_ydata(display_buffer)

        spectrum = np.fft.rfft(display_buffer * hann)
        mag = np.abs(spectrum) * mag_scale
        db = 20 * np.log10(mag + 1e-12)
        line_freq.set_ydata(db)

        peak_freq, peak_db, peak_idx = find_peak(db, window_size, fs)
        harmonics = find_harmonics(db, peak_idx, window_size, fs, max_harmonic=5)
        harmonic2 = harmonics.get(2)
        harmonic_freq, harmonic_db, harmonic_idx = harmonic2 if harmonic2 else (None, None, None)
        snr_db, noise_floor_db = compute_snr(mag, peak_idx, harmonic_idx)
        thd_percent, _thd_db = compute_thd(mag, peak_idx, harmonics)

        lines = [f"peak: {peak_freq:.2f} Hz  ({peak_db:.1f} dB)"]
        if harmonic_freq is not None:
            lines.append(f"2nd harmonic: {harmonic_freq:.2f} Hz  ({harmonic_db:.1f} dB)")
        else:
            lines.append("2nd harmonic: n/a")
        lines.append(f"SNR: {snr_db:.1f} dB")
        lines.append(f"THD: {thd_percent:.2f} %")
        peak_text.set_text("\n".join(lines))

        # Measured faster than an in-place slice-shift for this 2D shape
        # (numpy's overlap-safe copy path costs more here than a realloc).
        spec_history = np.roll(spec_history, -1, axis=1)
        spec_history[:, -1] = db
        im_spec.set_data(spec_history)

        noise_floor_history[:-1] = noise_floor_history[1:]
        noise_floor_history[-1] = noise_floor_db
        snr_history[:-1] = snr_history[1:]
        snr_history[-1] = snr_db
        line_noise_floor.set_ydata(noise_floor_history)
        line_snr.set_ydata(snr_history)

        state["last"] = dict(
            t_axis=t_axis,
            buffer=display_buffer.copy(),
            freqs=freqs,
            mag=mag,
            db=db,
            peak_freq=peak_freq,
            peak_db=peak_db,
            harmonic_freq=harmonic_freq,
            harmonic_db=harmonic_db,
            snr_db=snr_db,
            noise_floor_db=noise_floor_db,
            thd_percent=thd_percent,
        )

        return line_time, line_freq, peak_text, im_spec, fps_text, line_noise_floor, line_snr

    # All axis limits are fixed up front (the noise panel no longer
    # autoscales per frame), so blitting is safe here: matplotlib redraws
    # only the artists update() returns instead of the whole figure.
    anim = FuncAnimation(fig, update, interval=1000 / args.fps, blit=True, cache_frame_data=False)
    if live:
        fig.canvas.mpl_connect("close_event", lambda _evt: reader.stop())
    plt.show()


if __name__ == "__main__":
    main()

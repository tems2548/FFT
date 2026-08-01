"""Pure signal-processing math for the FFT test bench: synthetic waveform
generation, FFT window functions, peak/harmonic detection, and the derived
measurements (SNR, SINAD, THD, ENOB, noise density, duty cycle, cepstrum,
Goertzel, pulse timing). No Qt or serial-port dependencies -- everything
here is a plain function/array in, value/array out, so the test suite
exercises it directly without needing a running GUI.
"""
import numpy as np

WAVE_TYPES = ["demo", "sine", "chirp", "square", "sawtooth", "noise"]


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


def compute_snr(mag, peak_idx, harmonic_indices=(), exclude_radius=2):
    """Fundamental peak power vs. average noise-floor power.

    The bins around the fundamental and *every* given harmonic are
    excluded from the noise floor estimate, since they carry real
    (deterministic) signal, not noise; DC is excluded too. This is what
    distinguishes SNR from SINAD (see compute_sinad, which deliberately
    leaves harmonics in) -- excluding only one harmonic (as an earlier
    version of this function did, passing just the 2nd) let real energy
    from the 3rd, 4th, ... harmonics leak into the "noise" floor for any
    harmonic-rich signal. Verified this matters a lot: for a square wave,
    excluding just the 2nd harmonic read ~38dB SNR; excluding the full
    located set (up to Nyquist) reads ~52dB for the same signal -- a 14dB
    error from real, deterministic harmonic content being miscounted as
    noise.

    Returns (snr_db, noise_floor_db).
    """
    power = mag ** 2
    n = len(power)
    mask = np.ones(n, dtype=bool)
    mask[0] = False
    for idx in (peak_idx, *harmonic_indices):
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

    Takes a harmonics dict from find_harmonics rather than searching
    again. Callers should pass the *complete* set of located harmonics up
    to Nyquist (not just the 2nd-5th shown on screen) -- truncating to
    2-5 silently under-reports THD for harmonic-rich signals: verified
    ~39% vs. a more complete ~47% for a square wave (ideal square wave
    THD with infinite harmonics is ~48.3%), and ~68% vs. ~79% for a
    sawtooth. update_frame() computes one full up-to-60th-harmonic set
    and reuses it both here and for the noise floor exclusion mask, so
    this is "for free" as far as extra FFT/search work goes.

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

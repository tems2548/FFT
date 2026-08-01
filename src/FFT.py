"""Backward-compatible launcher/shim.

FFT.py used to contain the whole application; it has since been split into:
  - dsp.py      -- pure signal-processing math (no Qt/serial dependencies)
  - protocol.py -- the ESP32 serial wire format (CRC framing, SerialReader)
  - ui.py       -- the PyQt6/pyqtgraph application (FFTBenchWindow, main())

This module re-exports everything from all three so `import FFT` and
`python FFT.py` keep working exactly as before the split -- existing
tooling, tests, and the `python FFT.py [--wave ...|--serial ...]` command
line need no changes.

The three modules live together in the FFT_Visualize/ folder next to this
file (kept out of it so this launcher is the one plain, run-anywhere
script); that folder is added to sys.path below so their own plain
`from dsp import ...`-style sibling imports keep resolving unchanged.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "FFT_Visualize"))

from dsp import (
    WAVE_TYPES,
    WINDOW_FUNCTIONS,
    _cosine_sum_window,
    _threshold_crossings,
    classify_waveform,
    compute_duty_cycle,
    compute_enob,
    compute_goertzel,
    compute_noise_metrics,
    compute_pulse_metrics,
    compute_real_cepstrum,
    compute_sfdr,
    compute_sinad,
    compute_snr,
    compute_thd,
    compute_time_domain_stats,
    find_cepstrum_peak,
    find_harmonic,
    find_harmonics,
    find_peak,
    find_second_peak,
    format_density,
    generate_chunk,
    parabolic_interpolation,
    parse_goertzel_targets,
    sweep_phase,
)
from protocol import (
    CONNECT_TIMEOUT_S,
    MAGIC_DATA,
    MAGIC_META,
    SerialReader,
    crc16_ccitt,
    serial,
)
from ui import (
    ACCENT_CEPSTRUM,
    ACCENT_CPU,
    ACCENT_CURSOR_A,
    ACCENT_CURSOR_B,
    ACCENT_DRIFT,
    ACCENT_DUTY,
    ACCENT_FREQ,
    ACCENT_GOERTZEL,
    ACCENT_NOISE_FLOOR,
    ACCENT_OK,
    ACCENT_PEAK_HOLD,
    ACCENT_PHASE,
    ACCENT_RAM,
    ACCENT_SNR,
    ACCENT_TIME,
    ACCENT_TRIGGER,
    BG,
    CURVE_WIDTH,
    DRIFT_METRICS,
    DRIFT_UNITS,
    GL3D_MAX_FREQ_BINS,
    GRAPH_GROUPS,
    GRAPH_TOOLTIPS,
    GRID_FG,
    PANEL_BG,
    PERF_STAGE_COLORS,
    PERF_STAGE_NAMES,
    SPECTROGRAM_HISTORY,
    STYLESHEET,
    TEXT_FG,
    FFTBenchWindow,
    QtCore,
    QtGui,
    QtSvg,
    QtWidgets,
    SerialPortDialog,
    apply_plot_theme,
    build_app_icon,
    gl,
    main,
    psutil,
    pg,
    save_snapshot_csv,
)
from ui import _window_refs  # noqa: F401 -- mutated in place by ui.py's own code; re-exported by name, not copied

if __name__ == "__main__":
    main()

"""GUI-level regression tests: constructs a real FFTBenchWindow on the Qt
offscreen platform (set in conftest.py) -- no real display needed.

These are slower and more environment-sensitive than the pure-function
tests elsewhere, but they're what actually caught several real bugs this
project ran into during development (a checkbox-initial-state bug, a
sidebar/plot build-order bug) that no pure-function test could have. Every
QSettings read/write is redirected to a throwaway tmp_path ini file --
never the real registry-backed FFTBench/ESP32FFTVisualizer location a
normal run uses.
"""
import pytest

import FFT
import ui
from PyQt6 import QtWidgets

pytestmark = pytest.mark.gui


class Args:
    wave = "demo"
    freq = 10.0
    freq2 = 60.0
    sweep_period = 8.0
    samplerate = 2000.0
    window = 2048
    fft_window = "Hann"
    fps = 10
    noise = 0.03
    averaging = 70.0
    baud = 3000000


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture
def make_window(qapp, tmp_path, monkeypatch):
    """Factory fixture: make_window() -> a shown FFTBenchWindow whose
    QSettings is redirected to an isolated ini file under tmp_path (shared
    across every window built by the same test, so settings round-trip
    tests can build a 2nd window and see the 1st one's saved state)."""
    ini_path = str(tmp_path / "settings.ini")
    real_qsettings = FFT.QtCore.QSettings

    def fake_qsettings(*_args, **_kwargs):
        return real_qsettings(ini_path, real_qsettings.Format.IniFormat)

    monkeypatch.setattr(FFT.QtCore, "QSettings", fake_qsettings)

    windows = []

    def _make(args=None):
        win = FFT.FFTBenchWindow(args or Args(), False, None, None, Args.samplerate)
        FFT._window_refs.append(win)  # matches main()/_on_connect_click's real bookkeeping
        win.show()
        windows.append(win)
        return win

    yield _make

    for win in windows:
        win.close()


class TestGraphsStartHidden:
    def test_every_graph_checkbox_starts_unchecked(self, make_window):
        win = make_window()
        assert len(win._graph_checkboxes) == 12
        assert all(not cb.isChecked() for cb in win._graph_checkboxes.values())

    def test_every_plot_widget_starts_hidden(self, make_window):
        win = make_window()
        plot_widgets = [
            win.time_plot, win.freq_plot, win.phase_plot, win.bode_plot, win.spec_plot,
            win.noise_plot, win.drift_plot, win.cepstrum_plot, win.goertzel_plot,
            win.fft3d_plot, win.perf_plot, win.sysres_plot,
        ]
        assert all(not p.isVisible() for p in plot_widgets)


class TestCheckboxWiring:
    def test_toggling_checkbox_shows_and_hides_its_plot(self, make_window):
        win = make_window()
        checkbox = win._graph_checkboxes["Cepstrum Analysis"]
        checkbox.setChecked(True)
        assert win.cepstrum_plot.isVisible()
        checkbox.setChecked(False)
        assert not win.cepstrum_plot.isVisible()

    @pytest.mark.parametrize(
        "graph_label,section_attr",
        [
            ("Cepstrum Analysis", "cepstrum_section"),
            ("Goertzel Analyzer", "goertzel_section"),
            ("Performance Benchmark", "perf_section"),
            ("CPU / RAM Usage", "system_section"),
        ],
    )
    def test_toggling_checkbox_also_shows_its_sidebar_section(self, make_window, graph_label, section_attr):
        win = make_window()
        # Expand "Advanced Analysis" first -- these sections live inside
        # it, so their own visible flag is real but isVisible() would
        # read False while an ancestor is collapsed (that's Qt's normal
        # nested-visibility behavior, not a bug to route around here).
        for header in win.findChildren(QtWidgets.QToolButton):
            if header.text() == "Advanced Analysis":
                header.setChecked(True)
        section = getattr(win, section_attr)
        win._graph_checkboxes[graph_label].setChecked(True)
        assert section.isVisible()
        win._graph_checkboxes[graph_label].setChecked(False)
        assert not section.isVisible()

    def test_show_all_and_hide_all_buttons(self, make_window):
        win = make_window()
        show_all = next(b for b in win.findChildren(QtWidgets.QPushButton) if "Show All" in b.text())
        hide_all = next(b for b in win.findChildren(QtWidgets.QPushButton) if "Hide All" in b.text())
        show_all.click()
        assert all(cb.isChecked() for cb in win._graph_checkboxes.values())
        hide_all.click()
        assert all(not cb.isChecked() for cb in win._graph_checkboxes.values())


class TestPerformanceGating:
    def test_hidden_panels_cost_almost_nothing(self, make_window):
        win = make_window()
        for _ in range(3):
            win.update_frame()
        # All 4 of these panels default to hidden -- their stages should
        # be near-zero, not just "smaller".
        for stage in ("Cepstrum", "Goertzel", "Spectrogram/3D"):
            assert win.perf_stage_ms[stage] < 0.05

    def test_enabling_a_panel_makes_its_stage_measurably_nonzero(self, make_window):
        win = make_window()
        win._graph_checkboxes["Cepstrum Analysis"].setChecked(True)
        for _ in range(3):
            win.update_frame()
        assert win.perf_stage_ms["Cepstrum"] > 0.0
        # And the readout it feeds shouldn't be stuck on the placeholder.
        assert win.cepstrum_label.text() != "—"


class FakeReader:
    """Stands in for SerialReader without opening a real port -- just the
    attributes FFTBenchWindow reads from a live reader."""

    def __init__(self, sample_rate=80000.0):
        import queue as _queue
        self.sample_rate = sample_rate
        self.temp_c = None
        self.packets_ok = 0
        self.packets_bad = 0
        self.sample_queue = _queue.Queue()
        self.stopped = False

    def stop(self):
        self.stopped = True


class FakeSerialPortDialog:
    """Stands in for ui.SerialPortDialog -- .exec() returns Accepted
    immediately instead of showing a real (modal, blocking) dialog."""

    def __init__(self, _baud, parent=None):
        pass

    def exec(self):
        return QtWidgets.QDialog.DialogCode.Accepted

    reader = FakeReader()
    port = "COM7"


class TestConnectToSerialPort:
    def test_synthetic_window_shows_connect_button(self, make_window):
        win = make_window()
        assert not win.live
        assert win.connection_label.text() == "Synthetic mode (no hardware)"
        assert win.connect_button.text() == "Connect to serial port..."

    def test_connect_click_replaces_window_with_a_live_one(self, make_window, monkeypatch):
        win = make_window()
        # Patched on ui, not FFT: _on_connect_click (defined in ui.py) looks
        # up the bare name SerialPortDialog in ui.py's own module globals,
        # not FFT.py's re-exported copy -- patching the shim wouldn't reach it.
        monkeypatch.setattr(ui, "SerialPortDialog", FakeSerialPortDialog)

        assert win in FFT._window_refs
        win._on_connect_click()

        assert win not in FFT._window_refs  # old window closed and unregistered
        new_win = FFT._window_refs[-1]
        try:
            assert new_win.live
            assert new_win.port_label == "COM7"
            assert new_win.fs == 80000.0
            assert new_win.connection_label.text() == "Connected: COM7 @ 80000 Hz"
            assert new_win.connect_button.text() == "Change port..."
        finally:
            new_win.close()  # not tracked by make_window's own cleanup

        assert new_win not in FFT._window_refs

    def test_declining_the_dialog_leaves_the_window_untouched(self, make_window, monkeypatch):
        class DecliningDialog(FakeSerialPortDialog):
            def exec(self):
                return QtWidgets.QDialog.DialogCode.Rejected

        win = make_window()
        monkeypatch.setattr(ui, "SerialPortDialog", DecliningDialog)
        win._on_connect_click()
        assert not win.live
        assert win in FFT._window_refs


class TestModes:
    def test_dsp_lab_mode_reveals_pipeline_plots(self, make_window):
        win = make_window()
        assert not win.raw_signal_plot.isVisible()
        win.dsp_lab_checkbox.setChecked(True)
        assert win.raw_signal_plot.isVisible()
        assert win.windowed_signal_plot.isVisible()
        win.dsp_lab_checkbox.setChecked(False)
        assert not win.raw_signal_plot.isVisible()

    def test_duty_cycle_mode_computes_pulse_metrics(self, make_window):
        win = make_window(Args())
        win.duty_cycle_checkbox.setChecked(True)
        win.update_frame()
        assert win.duty_cycle_label.text() != "—"


class SquareWaveArgs(Args):
    wave = "square"
    freq = 20.0 * 2000.0 / 4096  # bin-aligned at this window/samplerate
    samplerate = 2000.0
    window = 4096
    noise = 0.0


class TestThdUsesCompleteHarmonicSet:
    def test_square_wave_thd_reflects_all_harmonics_not_just_2nd_5th(self, make_window):
        # Regression guard: THD must be computed from the complete
        # up-to-Nyquist harmonic set update_frame() locates (verified
        # ~47% for this signal), not the truncated 2nd-5th-only set the
        # Harmonics panel displays (which reads a misleadingly low ~39%
        # for the same signal) -- see compute_thd's docstring.
        win = make_window(SquareWaveArgs())
        for _ in range(5):
            win.update_frame()
        thd_percent = win.last_snapshot["thd_percent"]
        assert thd_percent is not None
        assert 42.0 < thd_percent < 52.0  # complete-set range; truncated range is ~35-42%


class NoisySquareWaveArgs(Args):
    wave = "square"
    freq = 20.0 * 2000.0 / 4096  # bin-aligned at this window/samplerate
    samplerate = 2000.0
    window = 4096
    noise = 0.01


class TestSnrExcludesAllHarmonics:
    def test_square_wave_snr_reflects_full_harmonic_exclusion(self, make_window):
        # Regression guard: compute_snr's noise-floor average must exclude
        # every located harmonic, not just the 2nd -- excluding only one
        # leaves the 3rd/4th/5th/... harmonics' real energy counted as
        # "noise", understating SNR by ~14dB for this exact signal
        # (verified: ~38dB with only the 2nd excluded vs. ~52dB with the
        # full set). fps=10/window=4096 needs the sliding buffer several
        # frames to fill with real (non-zero-padded) signal before this
        # settles -- 30 is comfortably past the ~21 needed.
        win = make_window(NoisySquareWaveArgs())
        for _ in range(30):
            win.update_frame()
        snr_db = win.last_snapshot["snr_db"]
        assert 45.0 < snr_db < 58.0  # full-exclusion range; single-harmonic range is ~30-42dB


class TestSettingsRoundTrip:
    def test_graph_visibility_and_controls_survive_a_restart(self, make_window):
        win1 = make_window()
        win1._graph_checkboxes["Cepstrum Analysis"].setChecked(True)
        win1._graph_checkboxes["3D FFT (waterfall)"].setChecked(True)
        win1.window_combo.setCurrentText("Blackman-Harris")
        win1.averaging_slider.setValue(42)
        win1.log_axis_checkbox.setChecked(True)
        win1.drift_metric_combo.setCurrentText("THD")
        win1.goertzel_freqs_edit.setText("100, 200, 300")
        win1.goertzel_freqs_edit.editingFinished.emit()
        win1._save_settings()

        win2 = make_window()  # fresh instance, same tmp_path ini file
        assert win2._graph_checkboxes["Cepstrum Analysis"].isChecked()
        assert win2._graph_checkboxes["3D FFT (waterfall)"].isChecked()
        assert win2.window_name == "Blackman-Harris"
        assert win2.averaging_slider.value() == 42
        assert win2.log_freq_axis is True
        assert win2.drift_metric == "THD"
        assert win2.goertzel_targets == [100.0, 200.0, 300.0]

import sys
import os
import logging
import re
from collections import deque

import yaml
from pathlib import Path
from datetime import datetime

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSplitter,
    QGroupBox,
    QPushButton,
    QCheckBox,
    QLabel,
    QSpinBox,
    QGridLayout,
    QFrame,
    QComboBox,
    QApplication,
    QSizePolicy,
    QMainWindow,
    QMessageBox,
)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure


class SafeFigureCanvas(FigureCanvas):
    """FigureCanvasQTAgg, but ignores resize events with a non-positive
    width/height instead of crashing.

    Qt can legitimately deliver a transient invalid size during layout
    renegotiation, for example right after toggling a splitter panel's
    visibility while the window is near its minimum width. Matplotlib's
    default resizeEvent raises outright on that rather than ignoring it.
    """

    def resizeEvent(self, event):
        size = event.size()

        if size.width() <= 0 or size.height() <= 0:
            return

        super().resizeEvent(event)


# --- local project paths ---------------------------------------------------
# Adjust these to match wherever cred_controller.py / pyCRED live in your
# environment.
script_dir = Path(__file__).parent.absolute()

if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from cred_controller import CredOneController, CredOneError, CredOneBusyError


# Optional GUI modules.
try:
    from guis.widgets.logger_widget import LoggerWidget
    from kaotools.ao_logging.ao_logging import setup_logger

except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Some functionality may be limited.")


# Status tokens -> display colors.
STATUS_COLORS = {
    "operational": "#2ecc71",
    "isbeingcooled": "#f1c40f",
    "ready": "#3498db",
}

STATUS_DEFAULT_COLOR = "#e74c3c"


class SessionLogHandler(logging.Handler):
    """Keep a complete in-memory log for one monitor-GUI session.

    LoggerWidget may limit the number of lines visible in the GUI, so this
    handler independently retains every record emitted after the window is
    opened. A snapshot can then be written to disk at any time.
    """

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    def emit(self, record):
        try:
            self.records.append(self.format(record))
        except Exception:
            self.handleError(record)

    def snapshot(self):
        self.acquire()
        try:
            return list(self.records)
        finally:
            self.release()


class CredMonitorWidget(QWidget):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cam = config.get("cam")
        if self.cam is None:
            self.cam = CredOneController(
                skip_serial_while_taking=True,
            )
        self.log = config.get("logger")
        if self.log is None:
            self.log = setup_logger()
        self.cam.log = self.log

        self.auto_refresh_enabled = True
        monitor_config = config.get("monitor", {})

        self.refresh_interval = monitor_config.get(
            "refresh_interval_ms",
            2000,
        )

        self.max_history_points = monitor_config.get(
            "max_history_points",
            600,
        )

        # Event-detection settings. These defaults require a sustained rise
        # of at least 1 K across three monitor samples before a warming event
        # is marked. They can be overridden in cred_monitor_gui_config.yaml.
        self.temp_rise_alert_k = float(
            monitor_config.get("temperature_rise_alert_k", 1.0)
        )
        self.temp_rise_samples = max(
            2,
            int(monitor_config.get("temperature_rise_samples", 3)),
        )
        self.temp_rise_alert_cooldown_s = float(
            monitor_config.get("temperature_rise_alert_cooldown_s", 300)
        )

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(
            self.refresh_status
        )

        # Guard against overlapping polls.
        self._polling = False
        self._paused_for_take = False

        # Full history for the life of this session.
        self.temp_history = {}
        self.pressure_history = []
        self.time_history = []

        # State used to mark noteworthy events in the saved session log.
        self._temperature_samples = {}
        self._last_temperature_alert = {}
        self._last_cooling_command = None
        self._last_safe_state = None

        self.setupUI()

        if self.auto_refresh_enabled:
            self.refresh_timer.start(
                self.refresh_interval
            )
            self.refresh_status()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def setupUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(
            5,
            5,
            5,
            5,
        )
        main_layout.setSpacing(5)

        main_splitter = QSplitter(
            QtCore.Qt.Vertical
        )
        main_layout.addWidget(main_splitter)

        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        top_splitter = QSplitter(
            QtCore.Qt.Horizontal
        )
        top_layout.addWidget(top_splitter)

        # ---------------- Left panel: controls + readouts ----------------
        control_panel = QWidget()
        control_layout = QVBoxLayout(
            control_panel
        )
        control_layout.setContentsMargins(
            5,
            5,
            5,
            5,
        )

        # Status indicator
        status_box = QGroupBox(
            "Camera Status / Health"
        )
        status_layout = QVBoxLayout(status_box)

        self.status_indicator = QFrame()
        self.status_indicator.setFixedHeight(28)
        self.status_indicator.setStyleSheet(
            f"background-color: "
            f"{STATUS_DEFAULT_COLOR}; "
            f"border-radius: 4px;"
        )

        self.status_label = QLabel("Unknown")
        self.status_label.setAlignment(
            QtCore.Qt.AlignCenter
        )
        self.status_label.setStyleSheet(
            "font-weight: bold; "
            "font-size: 13px;"
        )

        self.status_raw_label = QLabel("")
        self.status_raw_label.setWordWrap(True)
        self.status_raw_label.setStyleSheet(
            "color: gray; "
            "font-size: 10px;"
        )

        status_layout.addWidget(
            self.status_indicator
        )
        status_layout.addWidget(
            self.status_label
        )
        status_layout.addWidget(
            self.status_raw_label
        )

        control_layout.addWidget(status_box)

        # Cooling controls
        cooling_box = QGroupBox("Cooling")
        cooling_layout = QHBoxLayout(
            cooling_box
        )

        cooling_on_btn = QPushButton(
            "Cooling ON"
        )
        cooling_on_btn.clicked.connect(
            lambda: self.set_cooling(True)
        )

        cooling_off_btn = QPushButton(
            "Cooling OFF"
        )
        cooling_off_btn.clicked.connect(
            lambda: self.set_cooling(False)
        )

        cooling_layout.addWidget(
            cooling_on_btn
        )
        cooling_layout.addWidget(
            cooling_off_btn
        )

        control_layout.addWidget(cooling_box)

        # Temperature readout table
        temp_box = QGroupBox(
            "Temperature (K)"
        )
        self.temp_grid = QGridLayout(temp_box)

        # Populated dynamically as sensors appear.
        self.temp_value_labels = {}

        control_layout.addWidget(temp_box)
        self.temp_box = temp_box

        # Pressure readout
        pressure_box = QGroupBox("Pressure")
        pressure_layout = QVBoxLayout(
            pressure_box
        )

        self.pressure_label = QLabel("--")
        self.pressure_label.setStyleSheet(
            "font-size: 14px; "
            "font-weight: bold;"
        )

        self.pressure_raw_label = QLabel("")
        self.pressure_raw_label.setWordWrap(True)
        self.pressure_raw_label.setStyleSheet(
            "color: gray; "
            "font-size: 10px;"
        )

        pressure_layout.addWidget(
            self.pressure_label
        )
        pressure_layout.addWidget(
            self.pressure_raw_label
        )

        control_layout.addWidget(
            pressure_box
        )

        # Refresh controls
        refresh_box = QGroupBox(
            "Auto Refresh"
        )
        refresh_layout = QGridLayout(
            refresh_box
        )

        self.auto_refresh_checkbox = QCheckBox(
            "Enabled"
        )
        self.auto_refresh_checkbox.setChecked(
            self.auto_refresh_enabled
        )
        self.auto_refresh_checkbox.stateChanged.connect(
            self.toggle_auto_refresh
        )

        interval_label = QLabel(
            "Interval (s):"
        )

        self.interval_spinbox = QSpinBox()
        self.interval_spinbox.setRange(
            1,
            300,
        )
        self.interval_spinbox.setValue(
            self.refresh_interval // 1000
        )
        self.interval_spinbox.valueChanged.connect(
            self.update_refresh_interval
        )

        manual_refresh_btn = QPushButton(
            "Refresh Now"
        )
        manual_refresh_btn.clicked.connect(
            self.refresh_status
        )

        refresh_layout.addWidget(
            self.auto_refresh_checkbox,
            0,
            0,
            1,
            2,
        )
        refresh_layout.addWidget(
            interval_label,
            1,
            0,
        )
        refresh_layout.addWidget(
            self.interval_spinbox,
            1,
            1,
        )
        refresh_layout.addWidget(
            manual_refresh_btn,
            2,
            0,
            1,
            2,
        )

        control_layout.addWidget(refresh_box)

        # Optional history-plot display
        display_box = QGroupBox("Display")
        display_box_layout = QVBoxLayout(
            display_box
        )

        self.show_plots_checkbox = QCheckBox(
            "Show History Plots"
        )
        self.show_plots_checkbox.setChecked(
            False
        )
        self.show_plots_checkbox.stateChanged.connect(
            self.toggle_show_plots
        )

        display_box_layout.addWidget(
            self.show_plots_checkbox
        )
        control_layout.addWidget(display_box)

        # Session-log controls. Saving writes every log record emitted
        # since this GUI was opened and exports the complete temperature /
        # pressure history plot, even when the plot panel is hidden.
        session_log_box = QGroupBox("Session Log")
        session_log_layout = QVBoxLayout(session_log_box)

        save_log_btn = QPushButton("Save Session Log + Plots")
        save_log_btn.clicked.connect(
            self.request_save_session_log
        )

        mark_continue_btn = QPushButton(
            "Mark Continue Pressed"
        )
        mark_continue_btn.clicked.connect(
            self.mark_continue_pressed
        )

        session_log_layout.addWidget(save_log_btn)
        session_log_layout.addWidget(
            mark_continue_btn
        )
        control_layout.addWidget(session_log_box)

        control_layout.addStretch()

        # ---------------- Right panel: history plots ---------------------
        plot_background = "#1e1e1e"

        self.display_widget = QWidget()
        self.display_widget.setMinimumWidth(0)
        self.display_widget.setMinimumHeight(
            500
        )
        self.display_widget.setVisible(False)

        display_layout = QVBoxLayout(
            self.display_widget
        )

        self.figure = Figure(
            figsize=(6, 6),
            dpi=100,
            facecolor=plot_background,
        )

        self.canvas = SafeFigureCanvas(
            self.figure
        )

        self.axes_temp = self.figure.add_subplot(
            211
        )
        self.axes_pressure = (
            self.figure.add_subplot(
                212,
                sharex=self.axes_temp,
            )
        )

        self._style_dark_axes(
            self.axes_temp,
            "Temperature (K)",
            "Temperature History",
        )

        self._style_dark_axes(
            self.axes_pressure,
            "Pressure",
            "Pressure History",
            xlabel="Time",
        )

        self.figure.tight_layout()
        display_layout.addWidget(self.canvas)

        top_splitter.addWidget(control_panel)
        top_splitter.addWidget(
            self.display_widget
        )

        top_splitter.setSizes(
            self.config.get(
                "gui",
                {},
            ).get(
                "top_splitter_sizes",
                [350, 850],
            )
        )

        self.top_splitter = top_splitter

        top_splitter.setStretchFactor(
            0,
            0,
        )
        top_splitter.setStretchFactor(
            1,
            1,
        )

        # ---------------- Bottom: logger ---------------------------------
        logger_config = self.config.get(
            "logging",
            {},
        )

        self.logger_widget = LoggerWidget(
            name="CRED Monitor Log",
            max_lines=logger_config.get(
                "max_lines",
                300,
            ),
            min_height=100,
            font_size=logger_config.get(
                "font_size",
                8,
            ),
        )

        main_splitter.addWidget(top_widget)
        main_splitter.addWidget(
            self.logger_widget
        )

        main_splitter.setStretchFactor(
            0,
            4,
        )
        main_splitter.setStretchFactor(
            1,
            1,
        )
        main_splitter.setCollapsible(
            1,
            True,
        )

        main_splitter.setSizes(
            self.config.get(
                "gui",
                {},
            ).get(
                "main_splitter_sizes",
                [650, 150],
            )
        )

    def _style_dark_axes(
        self,
        ax,
        ylabel,
        title,
        xlabel=None,
    ):
        """Apply consistent dark styling to a plot axis."""

        ax.set_facecolor("#1e1e1e")
        ax.set_ylabel(
            ylabel,
            color="white",
        )
        ax.set_title(
            title,
            color="white",
        )

        if xlabel:
            ax.set_xlabel(
                xlabel,
                color="white",
            )

        ax.tick_params(
            colors="white",
            labelsize=8,
        )

        for spine in ax.spines.values():
            spine.set_color("#666666")

        ax.grid(
            True,
            color="#444444",
            linewidth=0.5,
            alpha=0.6,
        )

    def toggle_show_plots(self, state):
        show = state == QtCore.Qt.Checked

        self.display_widget.setVisible(show)

        if show:
            self.display_widget.setMinimumWidth(
                600
            )

            # Widen the containing window before showing the plot panel.
            top_window = self.window()
            top_window.setMinimumWidth(1000)

            if top_window.width() < 1000:
                top_window.resize(
                    1000,
                    top_window.height(),
                )

            self.top_splitter.setSizes(
                [350, 850]
            )

            # Draw immediately when the panel appears.
            self._redraw_history_plots()

        else:
            self.display_widget.setMinimumWidth(
                0
            )
            self.window().setMinimumWidth(
                380
            )
            self.top_splitter.setSizes(
                [1, 0]
            )

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------
    def refresh_status(self):
        if self._polling:
            self.log.debug(
                "Skipping refresh tick, "
                "previous poll still in flight"
            )
            return

        self._polling = True

        try:
            # Hold one nonblocking shared lock for the complete monitor
            # cycle. If ./take owns the exclusive lock, skip the whole
            # cycle so the GUI stays responsive and no partial history
            # sample is recorded.
            with self.cam.monitor_poll():
                now = datetime.now()

                self.time_history.append(now)
                self.refresh_temperature(now)
                self.refresh_pressure(now)
                self.refresh_camera_status()

                if self.display_widget.isVisible():
                    self._redraw_history_plots()

            if self._paused_for_take:
                self.log.info(
                    "Image acquisition finished; monitor polling resumed"
                )
                self._paused_for_take = False

        except CredOneBusyError:
            if not self._paused_for_take:
                self.log.info(
                    "./take is running; monitor polling paused"
                )

            self._paused_for_take = True
            self.status_label.setText(
                "Paused for image acquisition"
            )
            self.status_indicator.setStyleSheet(
                "background-color: #f1c40f; border-radius: 4px;"
            )
            self.status_raw_label.setText(
                "The control GUI is running ./take. Temperature, pressure, "
                "and status polling will resume automatically when the "
                "acquisition finishes."
            )

        finally:
            self._polling = False

    @staticmethod
    def _safe_state_from_status(raw):
        """Return SAFE/PREV SAFE when either state appears in status text."""
        normalized = re.sub(
            r"[_-]+",
            " ",
            raw.lower(),
        )
        normalized = " ".join(normalized.split())

        if (
            "prev safe" in normalized
            or "previous safe" in normalized
            or "prevsafe" in raw.lower()
        ):
            return "PREV SAFE"

        if re.search(r"\bsafe\b", normalized):
            return "SAFE"

        return None

    def _record_status_events(self, raw):
        safe_state = self._safe_state_from_status(
            raw
        )

        if safe_state is not None:
            if safe_state != self._last_safe_state:
                self.log.critical(
                    f"[EVENT SAFE_STATE] Camera status entered "
                    f"{safe_state}. Raw status: {raw!r}"
                )

        elif self._last_safe_state is not None:
            self.log.warning(
                f"[EVENT SAFE_CLEARED] Camera left "
                f"{self._last_safe_state}. Continue or another "
                f"recovery action may have occurred. "
                f"Raw status: {raw!r}"
            )

        self._last_safe_state = safe_state
        return safe_state

    def _record_temperature_event(
        self,
        sensor_name,
        value,
        now,
    ):
        """Mark a sustained cryogenic-temperature rise in the session log."""
        sensor_key = sensor_name.lower()

        if not any(
            plotted_name in sensor_key
            for plotted_name in self.TEMP_PLOT_SENSORS
        ):
            return

        samples = self._temperature_samples.setdefault(
            sensor_name,
            deque(maxlen=self.temp_rise_samples),
        )
        samples.append((now, value))

        if len(samples) < self.temp_rise_samples:
            return

        values = [
            sample_value
            for _, sample_value in samples
        ]

        steadily_rising = all(
            later > earlier
            for earlier, later in zip(
                values,
                values[1:],
            )
        )
        total_rise = values[-1] - values[0]

        if (
            not steadily_rising
            or total_rise < self.temp_rise_alert_k
        ):
            return

        last_alert = self._last_temperature_alert.get(
            sensor_name
        )
        if (
            last_alert is not None
            and (now - last_alert).total_seconds()
            < self.temp_rise_alert_cooldown_s
        ):
            return

        if self._last_cooling_command is True:
            cooling_context = (
                "the last cooling command sent from this GUI was ON"
            )
        elif self._last_cooling_command is False:
            cooling_context = (
                "the last cooling command sent from this GUI was OFF"
            )
        else:
            cooling_context = (
                "no cooling command has been sent from this GUI "
                "during this session"
            )

        event_name = (
            "UNEXPECTED_WARMING"
            if self._last_cooling_command is not False
            else "WARMING_AFTER_COOLING_OFF"
        )
        log_level = (
            logging.CRITICAL
            if self._last_cooling_command is not False
            else logging.WARNING
        )

        self.log.log(
            log_level,
            f"[EVENT {event_name}] {sensor_name} rose "
            f"{total_rise:.2f} K, from {values[0]:.2f} K "
            f"at {samples[0][0].isoformat(timespec='seconds')} "
            f"to {values[-1]:.2f} K at "
            f"{samples[-1][0].isoformat(timespec='seconds')}; "
            f"{cooling_context}.",
        )
        self._last_temperature_alert[
            sensor_name
        ] = now

    def refresh_camera_status(self):
        try:
            raw = self.cam.get_status()

            self.status_raw_label.setText(raw)
            safe_state = self._record_status_events(
                raw
            )

            token = next(
                (
                    status_token
                    for status_token in STATUS_COLORS
                    if status_token in raw.lower()
                ),
                None,
            )

            if safe_state is not None:
                color = STATUS_DEFAULT_COLOR
                label = safe_state
            else:
                color = STATUS_COLORS.get(
                    token,
                    STATUS_DEFAULT_COLOR,
                )

                if token:
                    label = token
                else:
                    label = "Unknown / check log"

            self.status_label.setText(
                label.capitalize()
            )

            self.status_indicator.setStyleSheet(
                f"background-color: {color}; "
                f"border-radius: 4px;"
            )

        except CredOneError as e:
            self.status_label.setText("Error")

            self.status_indicator.setStyleSheet(
                f"background-color: "
                f"{STATUS_DEFAULT_COLOR}; "
                f"border-radius: 4px;"
            )

            self.status_raw_label.setText(
                str(e)
            )

            self.log.error(
                f"Failed to get camera status: {e}"
            )

    def refresh_temperature(self, now):
        try:
            raw, parsed = (
                self.cam.get_temperature()
            )

            self._ensure_temp_rows(
                parsed.keys()
            )

            for name, value in parsed.items():
                self.temp_value_labels[
                    name
                ].setText(
                    f"{value:.1f}"
                )

                self.temp_history.setdefault(
                    name,
                    [],
                ).append(value)

                self._record_temperature_event(
                    name,
                    value,
                    now,
                )

            if not parsed:
                self.log.warning(
                    "Could not parse temperature "
                    f"output, raw: {raw!r}"
                )

        except CredOneError as e:
            self.log.error(
                f"Failed to get temperature: {e}"
            )

    def refresh_pressure(self, now):
        try:
            raw, value = self.cam.get_pressure()

            if value is not None:
                self.pressure_label.setText(
                    f"{value:.3g}"
                )
            else:
                self.pressure_label.setText("--")

            self.pressure_raw_label.setText(raw)
            self.pressure_history.append(value)

        except CredOneError as e:
            self.pressure_label.setText("Error")
            self.pressure_raw_label.setText(
                str(e)
            )
            self.pressure_history.append(None)

            self.log.error(
                f"Failed to get pressure: {e}"
            )

    def _ensure_temp_rows(
        self,
        sensor_names,
    ):
        """Add a row for each newly seen sensor."""

        for name in sensor_names:
            if name in self.temp_value_labels:
                continue

            row = self.temp_grid.rowCount()

            name_label = QLabel(name)
            value_label = QLabel("--")
            value_label.setStyleSheet(
                "font-weight: bold;"
            )

            self.temp_grid.addWidget(
                name_label,
                row,
                0,
            )
            self.temp_grid.addWidget(
                value_label,
                row,
                1,
            )

            self.temp_value_labels[
                name
            ] = value_label

    # Only these two temperature sensors are plotted.
    TEMP_PLOT_SENSORS = {
        "cryopt": "#00d4ff",
        "cryod": "#ff6ec7",
    }

    def _redraw_history_plots(self):
        times = list(self.time_history)

        self.axes_temp.clear()

        self._style_dark_axes(
            self.axes_temp,
            "Temperature (K)",
            "Temperature History",
        )

        plotted_any = False

        for name, values in (
            self.temp_history.items()
        ):
            match = next(
                (
                    key
                    for key in self.TEMP_PLOT_SENSORS
                    if key in name.lower()
                ),
                None,
            )

            if match is None:
                continue

            values = list(values)
            nvalues = min(
                len(times),
                len(values),
            )

            if nvalues == 0:
                continue

            self.axes_temp.plot(
                times[-nvalues:],
                values[-nvalues:],
                marker="o",
                markersize=2,
                label=name,
                color=self.TEMP_PLOT_SENSORS[
                    match
                ],
            )

            plotted_any = True

        if plotted_any:
            legend = self.axes_temp.legend(
                loc="upper right",
                fontsize=8,
                facecolor="#1e1e1e",
            )

            for text in legend.get_texts():
                text.set_color("white")

        self.axes_pressure.clear()

        self._style_dark_axes(
            self.axes_pressure,
            "Pressure",
            "Pressure History",
            xlabel="Time",
        )

        pressure_values = list(
            self.pressure_history
        )

        nvalues = min(
            len(times),
            len(pressure_values),
        )

        if nvalues > 0:
            plot_times = [
                time_value
                for time_value, pressure_value
                in zip(
                    times[-nvalues:],
                    pressure_values[-nvalues:],
                )
                if pressure_value is not None
            ]

            plot_values = [
                pressure_value
                for pressure_value
                in pressure_values[-nvalues:]
                if pressure_value is not None
            ]

            if plot_values:
                self.axes_pressure.plot(
                    plot_times,
                    plot_values,
                    marker="o",
                    markersize=2,
                    color="#f1c40f",
                )

        time_formatter = mdates.DateFormatter(
            "%H:%M:%S"
        )

        self.axes_temp.xaxis.set_major_formatter(
            time_formatter
        )
        self.axes_pressure.xaxis.set_major_formatter(
            time_formatter
        )

        self.figure.autofmt_xdate()
        self.canvas.draw()

    def save_history_plot(self, path, dpi=150):
        """Save the complete session temperature/pressure history.

        A separate Figure is built from the stored history, so this works
        even when the on-screen history panel is hidden.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        plot_background = "#1e1e1e"
        figure = Figure(
            figsize=(10, 8),
            dpi=dpi,
            facecolor=plot_background,
        )
        axes_temp = figure.add_subplot(211)
        axes_pressure = figure.add_subplot(212, sharex=axes_temp)

        self._style_dark_axes(
            axes_temp,
            "Temperature (K)",
            "C-RED ONE Temperature History",
        )
        self._style_dark_axes(
            axes_pressure,
            "Pressure",
            "C-RED ONE Pressure History",
            xlabel="Time",
        )

        times = list(self.time_history)
        plotted_temperature = False

        for name, values in self.temp_history.items():
            match = next(
                (
                    key
                    for key in self.TEMP_PLOT_SENSORS
                    if key in name.lower()
                ),
                None,
            )
            if match is None:
                continue

            values = list(values)
            nvalues = min(len(times), len(values))
            if nvalues == 0:
                continue

            axes_temp.plot(
                times[-nvalues:],
                values[-nvalues:],
                marker="o",
                markersize=2,
                label=name,
                color=self.TEMP_PLOT_SENSORS[match],
            )
            plotted_temperature = True

        if plotted_temperature:
            legend = axes_temp.legend(
                loc="upper right",
                fontsize=8,
                facecolor=plot_background,
            )
            for legend_text in legend.get_texts():
                legend_text.set_color("white")
        else:
            axes_temp.text(
                0.5,
                0.5,
                "No temperature samples recorded",
                ha="center",
                va="center",
                color="white",
                transform=axes_temp.transAxes,
            )

        pressure_values = list(self.pressure_history)
        nvalues = min(len(times), len(pressure_values))
        plotted_pressure = False

        if nvalues > 0:
            plot_times = [
                time_value
                for time_value, pressure_value in zip(
                    times[-nvalues:],
                    pressure_values[-nvalues:],
                )
                if pressure_value is not None
            ]
            plot_values = [
                pressure_value
                for pressure_value in pressure_values[-nvalues:]
                if pressure_value is not None
            ]

            if plot_values:
                axes_pressure.plot(
                    plot_times,
                    plot_values,
                    marker="o",
                    markersize=2,
                    color="#f1c40f",
                )
                plotted_pressure = True

        if not plotted_pressure:
            axes_pressure.text(
                0.5,
                0.5,
                "No pressure samples recorded",
                ha="center",
                va="center",
                color="white",
                transform=axes_pressure.transAxes,
            )

        time_formatter = mdates.DateFormatter("%H:%M:%S")
        axes_temp.xaxis.set_major_formatter(time_formatter)
        axes_pressure.xaxis.set_major_formatter(time_formatter)
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(
            path,
            dpi=dpi,
            facecolor=figure.get_facecolor(),
            bbox_inches="tight",
        )
        figure.clear()
        return path

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def request_save_session_log(self):
        main_window = self.window()

        if hasattr(main_window, "save_session_log"):
            main_window.save_session_log(
                show_confirmation=True,
                closing=False,
            )
        else:
            QMessageBox.critical(
                self,
                "Session Log Error",
                "Could not locate the main window's log saver.",
            )

    def mark_continue_pressed(self):
        self.log.warning(
            "[EVENT CONTINUE_PRESSED] Operator marked that "
            "Continue was pressed."
        )

        main_window = self.window()
        if hasattr(main_window, "statusBar"):
            main_window.statusBar().showMessage(
                "Continue press marked in session log",
                5000,
            )

    def set_cooling(self, on):
        if on:
            title = "Confirm Cooling"
            message = (
                "Are you sure you want to turn "
                "camera cooling ON?\n\n"
                "This will begin the camera "
                "cooldown process."
            )
        else:
            title = "Confirm Warm Up"
            message = (
                "Are you sure you want to turn "
                "camera cooling OFF?\n\n"
                "This will begin the camera "
                "warm-up process."
            )

        response = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if response != QMessageBox.Yes:
            self.log.info(
                f"{'Cooling' if on else 'Warm-up'} "
                f"command cancelled by user"
            )
            return

        try:
            self.cam.set_cooling(on)
            self._last_cooling_command = on

            self.log.warning(
                f"[EVENT COOLING_COMMAND] Cooling "
                f"{'ON' if on else 'OFF'} command sent "
                f"from the monitor GUI."
            )

            self.refresh_camera_status()

        except CredOneError as e:
            self.log.error(
                f"Failed to set cooling: {e}"
            )

            QMessageBox.critical(
                self,
                "Error",
                f"Failed to set cooling:\n{e}",
            )

    def toggle_auto_refresh(self, state):
        self.auto_refresh_enabled = (
            state == QtCore.Qt.Checked
        )

        if self.auto_refresh_enabled:
            self.refresh_timer.start(
                self.refresh_interval
            )

            self.log.info(
                "Auto-refresh enabled "
                f"({self.refresh_interval / 1000:.0f}s "
                "interval)"
            )

        else:
            self.refresh_timer.stop()
            self.log.info(
                "Auto-refresh disabled"
            )

    def update_refresh_interval(
        self,
        seconds,
    ):
        self.refresh_interval = seconds * 1000

        if self.auto_refresh_enabled:
            self.refresh_timer.stop()
            self.refresh_timer.start(
                self.refresh_interval
            )

            self.log.info(
                "Auto-refresh interval updated "
                f"to {seconds}s"
            )


class CredMonitorMainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.session_started_at = datetime.now()

        self.log = config.get("logger")
        if self.log is None:
            self.log = setup_logger()

        # Capture the complete monitor session independently of the
        # on-screen LoggerWidget, which may retain only a limited number
        # of visible lines.
        self.session_log_handler = SessionLogHandler()
        self.log.addHandler(
            self.session_log_handler
        )
        if self.log.level == logging.NOTSET or self.log.level > logging.DEBUG:
            self.log.setLevel(logging.DEBUG)

        self.config["logger"] = self.log
        self.data_root = Path(
            config.get(
                "data_root",
                "/usr/local/aodev/CRED-One/Data",
            )
        )

        date_string = self.session_started_at.strftime(
            "%Y%m%d"
        )
        filename = (
            "cred_monitor_"
            f"{self.session_started_at:%Y%m%d_%H%M%S}.log"
        )
        self.session_log_path = (
            self.data_root
            / date_string
            / "log"
            / filename
        )

        session_log_config = config.get("session_log", {})
        self.save_plot_with_log = bool(
            session_log_config.get("save_plot_with_log", True)
        )
        self.session_plot_dpi = max(
            72,
            int(session_log_config.get("plot_dpi", 150)),
        )
        self.session_plot_format = str(
            session_log_config.get("plot_format", "png")
        ).lower().lstrip(".") or "png"
        self.session_plot_suffix = str(
            session_log_config.get("plot_filename_suffix", "_plots")
        )
        self.session_plot_path = self.session_log_path.with_name(
            self.session_log_path.stem
            + self.session_plot_suffix
            + "."
            + self.session_plot_format
        )

        self.log.info(
            "[SESSION START] C-RED ONE monitor GUI opened "
            f"at {self.session_started_at.isoformat(timespec='seconds')}"
        )

        self.setWindowTitle(
            config.get(
                "gui",
                {},
            ).get(
                "window_title",
                "C-RED ONE Monitor",
            )
        )

        geometry = config.get(
            "gui",
            {},
        ).get(
            "window_geometry",
            [100, 100, 1200, 800],
        )

        self.setGeometry(*geometry)
        self.setMinimumSize(380, 500)

        use_keck_theme = self.config.get(
            "styling",
            {},
        ).get(
            "use_keck_theme",
            True,
        )

        if use_keck_theme:
            try:
                self.apply_keck_theme()

            except Exception as e:
                self.log.warning(
                    "Could not apply Keck theme: "
                    f"{e}"
                )

        self.statusBar().showMessage("Ready")

        self.widget = CredMonitorWidget(config)
        self.setCentralWidget(self.widget)

    def save_session_log(
        self,
        show_confirmation=True,
        closing=False,
    ):
        """Write every record captured since this window was opened."""
        saved_at = datetime.now()

        try:
            self.session_log_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.log.info(
                "[SESSION LOG SAVE] Writing complete session log "
                f"to {self.session_log_path}"
            )

            records = (
                self.session_log_handler.snapshot()
            )

            header = [
                "C-RED ONE MONITOR SESSION LOG",
                "=" * 72,
                (
                    "Session opened: "
                    f"{self.session_started_at.isoformat(timespec='seconds')}"
                ),
                (
                    "Log saved:      "
                    f"{saved_at.isoformat(timespec='seconds')}"
                ),
                (
                    "Save reason:    "
                    f"{'GUI close' if closing else 'manual save'}"
                ),
                f"Data root:       {self.data_root}",
                (
                    "History plot:    "
                    + (
                        str(self.session_plot_path)
                        if self.save_plot_with_log
                        else "disabled in configuration"
                    )
                ),
                "",
                "Important automatic/manual event markers:",
                "  [EVENT SAFE_STATE]",
                "  [EVENT SAFE_CLEARED]",
                "  [EVENT UNEXPECTED_WARMING]",
                "  [EVENT WARMING_AFTER_COOLING_OFF]",
                "  [EVENT COOLING_COMMAND]",
                "  [EVENT CONTINUE_PRESSED]",
                "",
                "-" * 72,
            ]

            payload = "\n".join(
                header + records
            ) + "\n"

            with open(
                self.session_log_path,
                "w",
                encoding="utf-8",
            ) as log_file:
                log_file.write(payload)
                log_file.flush()
                os.fsync(log_file.fileno())

            if self.save_plot_with_log:
                self.widget.save_history_plot(
                    self.session_plot_path,
                    dpi=self.session_plot_dpi,
                )

        except (OSError, ValueError, RuntimeError) as e:
            self.log.error(
                f"Failed to save session log/plot: {e}"
            )
            QMessageBox.critical(
                self,
                "Session Save Failed",
                (
                    "Could not save the monitor session log and plot:\n\n"
                    f"{e}\n\n"
                    f"Requested path:\n{self.session_log_path}"
                ),
            )
            return None

        self.statusBar().showMessage(
            f"Session log and plot saved under {self.session_log_path.parent}",
            10000,
        )

        if show_confirmation:
            QMessageBox.information(
                self,
                "Session Log Saved",
                (
                    "The complete monitor session was saved to:\n\n"
                    f"Log:  {self.session_log_path}\n"
                    + (
                        f"Plot: {self.session_plot_path}"
                        if self.save_plot_with_log
                        else "Plot saving is disabled in the YAML configuration."
                    )
                ),
            )

        return self.session_log_path

    def closeEvent(self, event):
        timer_was_active = (
            hasattr(self, "widget")
            and self.widget.refresh_timer.isActive()
        )

        if hasattr(self, "widget"):
            self.widget.refresh_timer.stop()

        reply = QMessageBox.question(
            self,
            "Save Monitor Session?",
            (
                "Do you want to save the complete monitor log and "
                "history plots before closing?\n\n"
                "They will be written under today's data directory "
                "in the log subfolder."
            ),
            (
                QMessageBox.Yes
                | QMessageBox.No
                | QMessageBox.Cancel
            ),
            QMessageBox.Yes,
        )

        if reply == QMessageBox.Cancel:
            if timer_was_active:
                self.widget.refresh_timer.start(
                    self.widget.refresh_interval
                )
            event.ignore()
            return

        if reply == QMessageBox.Yes:
            self.log.info(
                "[SESSION END] Monitor GUI close requested "
                "with log save."
            )

            saved_path = self.save_session_log(
                show_confirmation=False,
                closing=True,
            )
            if saved_path is None:
                if timer_was_active:
                    self.widget.refresh_timer.start(
                        self.widget.refresh_interval
                    )
                event.ignore()
                return
        else:
            self.log.info(
                "[SESSION END] Monitor GUI closed without "
                "saving a final log snapshot."
            )

        event.accept()

    def apply_keck_theme(self):
        """Load the shared Keck theme when available."""

        try:
            stylesheet_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "keck_theme",
                "keck_dark_purple.qss",
            )

            if os.path.exists(stylesheet_path):
                with open(
                    stylesheet_path,
                    "r",
                ) as file_handle:
                    self.setStyleSheet(
                        file_handle.read()
                    )

                self.log.info(
                    "Full Keck theme applied"
                )

            else:
                self.log.warning(
                    "Keck theme file not found, "
                    "using compatibility theme"
                )

                self._apply_compatibility_theme()

        except Exception as e:
            self.log.error(
                f"Error applying Keck theme: {e}, "
                "using compatibility theme"
            )

            self._apply_compatibility_theme()

    def _apply_compatibility_theme(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #2b2b2b;
                color: white;
            }

            QGroupBox {
                color: white;
                border: 2px solid #483D8B;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
                padding-top: 15px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                left: 10px;
                padding: 0 5px 0 5px;
            }

            QPushButton {
                background-color: #483D8B;
                color: white;
                border: 2px solid #357ABD;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #357ABD;
            }

            QPushButton:pressed {
                background-color: #2A5A8B;
            }

            QLabel {
                color: white;
                background: transparent;
            }
            """
        )


def load_config(config_file=None):
    try:
        if config_file is None:
            config_path = (
                Path(__file__).parent
                / "cred_monitor_gui_config.yaml"
            ).resolve()
        else:
            config_path = Path(
                config_file
            ).resolve()

        if not config_path.exists():
            print(
                "Configuration file not found at: "
                f"{config_path}. Using defaults."
            )
            return {}

        with open(config_path, "r") as file_handle:
            return (
                yaml.safe_load(file_handle)
                or {}
            )

    except yaml.YAMLError as e:
        print(
            "Error parsing YAML configuration: "
            f"{e}. Using defaults."
        )
        return {}


if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setApplicationName(
        "CRED Monitor"
    )
    app.setApplicationDisplayName(
        "C-RED ONE Monitor GUI"
    )

    config = load_config()
    cam_config = config.get("cam", {})

    # Match the control GUI's data root. Monitor session logs are saved as:
    # <data_root>/<YYYYMMDD>/log/cred_monitor_<session start>.log
    config["data_root"] = cam_config.get(
        "data_root",
        config.get(
            "data_root",
            "/usr/local/aodev/CRED-One/Data",
        ),
    )

    try:
        config["cam"] = CredOneController(
            edt_dir=cam_config.get(
                "edt_dir",
                "/opt/EDTpdv",
            ),
            tmp_frame_path=cam_config.get(
                "tmp_frame_path",
                "/usr/local/aodev/CRED-One/Data/"
                "tmp/CRED_frame.raw",
            ),
            lock_path=cam_config.get(
                "lock_path",
                "/tmp/pycred_camera_io.lock",
            ),
            skip_serial_while_taking=True,
        )
    except CredOneError as e:
        QMessageBox.critical(
            None,
            "C-RED ONE Initialization Error",
            str(e),
        )
        sys.exit(1)

    window = CredMonitorMainWindow(config)
    window.show()

    sys.exit(app.exec_())

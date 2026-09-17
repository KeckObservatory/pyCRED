import sys
import os
import threading
import time

import numpy as np
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
    QDoubleSpinBox,
    QGridLayout,
    QComboBox,
    QApplication,
    QSizePolicy,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QInputDialog,
)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)

# --- local project paths ---------------------------------------------------
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from cred_controller import CredOneController, CredOneError

# Optional GUI modules (same pattern as the DM template)
try:
    from guis.widgets.logger_widget import LoggerWidget
    from kaotools.ao_logging.ao_logging import setup_logger
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Some functionality may be limited.")


# Readout-mode descriptions and full-frame FPS ceilings.
READOUT_MODE_INFO = {
    "globalresetsingle": {
        "display_name": "Global Reset / Single Read",
        "max_fps": 3500,
        "description": (
            "The detector is globally reset and read once per frame. "
            "This is the fastest full-frame readout mode."
        ),
    },
    "globalresetcds": {
        "display_name": "Global Reset / CDS",
        "max_fps": 1750,
        "description": (
            "The detector is read immediately after reset and again after "
            "the integration. The two reads are subtracted to perform "
            "correlated double sampling (CDS), which reduces reset noise. "
            "Because each frame requires two reads, the maximum frame rate "
            "is approximately half that of single-read mode."
        ),
    },
    "globalresetbursts": {
        "display_name": "Global Reset / NDR Burst",
        "max_fps": 3500,
        "description": (
            "The detector is globally reset once and then read multiple "
            "times without another reset. This produces a non-destructive "
            "read, or up-the-ramp, sequence. The frame rate and number of "
            "reads per reset must be selected together because too many "
            "reads at a low frame rate can saturate the detector before "
            "the next reset."
        ),
    },
}

READOUT_MODES = list(READOUT_MODE_INFO.keys())

# Combined type+format choice for the save prompt: display string ->
# (subfolder, save_type). Folder is still just "dark"/"flat" -- the
# format is folded into a single choice so it can't be forgotten
# separately from the type.
SAVE_OPTIONS = {
    "dark (.npy)": ("dark", ".npy"),
    "dark (.npz)": ("dark", ".npz"),
    "flat (.npy)": ("flat", ".npy"),
    "flat (.npz)": ("flat", ".npz"),
}


class CredControlWidget(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cam = config.get("cam")
        if self.cam is None:
            self.cam = CredOneController()
        self.log = config.get("logger", setup_logger())
        self.cam.log = self.log

        fps_cfg = config.get("fps", {})
        self.fps_range = fps_cfg.get("range", [1, 3500])
        self.fps_default = fps_cfg.get("default", 3500)

        gain_cfg = config.get("gain", {})
        self.gain_range = gain_cfg.get("range", [0, 1000])
        self.gain_default = gain_cfg.get("default", 1)

        ndr_cfg = config.get("ndr", {})
        self.ndr_range = ndr_cfg.get("range", [1, 1000])
        self.ndr_default = ndr_cfg.get("default", 1)

        burst_cfg = config.get("burst", {})
        self.burst_nframes_default = burst_cfg.get(
            "nframes_default",
            50,
        )

        # Where "Save Current Image" writes to: <data_root>/<YYYYMMDD>/
        self.data_root = config.get(
            "data_root",
            "/usr/local/aodev/CRED-One/Data",
        )

        self.operation_in_progress = False
        self.current_image_array = None
        self.current_capture_meta = {}
        
        ### Pupil selection

        # Default pupil mask values, in detector pixels
        self.default_pupil_radius = 26.5
        self.default_pupil_separation = 76.0
        self.default_mask_center_x = 160.0
        self.default_mask_center_y = 128.0

        saved_pupil = config.get("pupil_selection",{},)

        self.pupil_radius = saved_pupil.get("radius",self.default_pupil_radius,)

        self.pupil_separation = saved_pupil.get("separation",self.default_pupil_separation,)

        self.mask_center_x = saved_pupil.get("center_x",self.default_mask_center_x,)

        self.mask_center_y = saved_pupil.get("center_y",self.default_mask_center_y,)


        # Last frame acquired by live view
        self.last_live_frame = None

        # Whether the GUI is currently in pupil-selection mode
        self.pupil_selection_active = False

        # Matplotlib Circle objects.
        self.pupil_circles = []

        self.live_view_enabled = False
        self.live_view_interval = config.get(
            "live_view",
            {},
        ).get("interval_ms", 500)

        self.live_view_timer = QTimer()
        self.live_view_timer.timeout.connect(self.live_view_tick)

        self.setupUI()
        self.display_placeholder_image()

        try:
            self.get_fps()
            self.get_gain()
        except Exception as e:
            self.log.warning(
                f"Could not populate initial FPS/gain readback: {e}"
            )

    # ------------------------------------------------------------------
    def setupUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        main_splitter = QSplitter(QtCore.Qt.Vertical)
        main_layout.addWidget(main_splitter)

        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        top_splitter = QSplitter(QtCore.Qt.Horizontal)
        top_layout.addWidget(top_splitter)

        # ---------------- Left panel: controls -----------------------------
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)

        # FPS
        self.fps_box = QGroupBox(
            f"Frame Rate "
            f"({self.fps_range[0]}-{self.fps_range[1]} Hz)"
        )
        fps_layout = QGridLayout(self.fps_box)

        self.fps_input = QDoubleSpinBox()
        self.fps_input.setRange(*self.fps_range)
        self.fps_input.setValue(self.fps_default)

        fps_set_btn = QPushButton("Set")
        fps_set_btn.clicked.connect(self.set_fps)

        self.fps_current_label = QLabel("--")
        self.fps_current_label.setStyleSheet(
            "font-weight: bold;"
        )

        fps_layout.addWidget(self.fps_input, 0, 0)
        fps_layout.addWidget(fps_set_btn, 0, 1)
        fps_layout.addWidget(QLabel("Current:"), 1, 0)
        fps_layout.addWidget(self.fps_current_label, 1, 1)

        control_layout.addWidget(self.fps_box)

        # Gain
        gain_box = QGroupBox(
            f"Gain "
            f"({self.gain_range[0]}-{self.gain_range[1]})"
        )
        gain_layout = QGridLayout(gain_box)

        self.gain_input = QDoubleSpinBox()
        self.gain_input.setRange(*self.gain_range)
        self.gain_input.setValue(self.gain_default)

        gain_set_btn = QPushButton("Set")
        gain_set_btn.clicked.connect(self.set_gain)

        self.gain_current_label = QLabel("--")
        self.gain_current_label.setStyleSheet(
            "font-weight: bold;"
        )

        gain_layout.addWidget(self.gain_input, 0, 0)
        gain_layout.addWidget(gain_set_btn, 0, 1)
        gain_layout.addWidget(QLabel("Current:"), 1, 0)
        gain_layout.addWidget(self.gain_current_label, 1, 1)

        control_layout.addWidget(gain_box)

        # Readout mode
        mode_box = QGroupBox("Readout Mode")
        mode_layout = QHBoxLayout(mode_box)

        self.mode_dropdown = QComboBox()
        self.mode_dropdown.addItems(READOUT_MODES)

        # The camera mode is not queried at startup, so this remains None
        # until a mode has been successfully applied from this GUI.
        self.current_readout_mode = None

        # Show user-friendly definitions as tooltips while retaining the
        # exact command tokens required by the camera controller.
        for index, mode in enumerate(READOUT_MODES):
            info = READOUT_MODE_INFO[mode]

            self.mode_dropdown.setItemData(
                index,
                (
                    f"{info['display_name']} — maximum full-frame "
                    f"rate: {info['max_fps']} FPS"
                ),
                QtCore.Qt.ToolTipRole,
            )

        mode_set_btn = QPushButton("Set")
        mode_set_btn.clicked.connect(
            self.set_readout_mode
        )

        mode_layout.addWidget(self.mode_dropdown)
        mode_layout.addWidget(mode_set_btn)

        control_layout.addWidget(mode_box)

        # NDR + raw images
        ndr_box = QGroupBox("NDR / Raw Images")
        ndr_layout = QGridLayout(ndr_box)

        self.ndr_input = QSpinBox()
        self.ndr_input.setRange(*self.ndr_range)
        self.ndr_input.setValue(self.ndr_default)

        ndr_set_btn = QPushButton("Set NDR")
        ndr_set_btn.clicked.connect(self.set_ndr)

        self.rawimages_dropdown = QComboBox()
        self.rawimages_dropdown.addItems(["Off", "On"])

        rawimages_set_btn = QPushButton("Set Raw Images")
        rawimages_set_btn.clicked.connect(
            self.set_rawimages
        )

        ndr_layout.addWidget(QLabel("NDR:"), 0, 0)
        ndr_layout.addWidget(self.ndr_input, 0, 1)
        ndr_layout.addWidget(ndr_set_btn, 0, 2)
        ndr_layout.addWidget(QLabel("Raw images:"), 1, 0)
        ndr_layout.addWidget(
            self.rawimages_dropdown,
            1,
            1,
        )
        ndr_layout.addWidget(
            rawimages_set_btn,
            1,
            2,
        )

        control_layout.addWidget(ndr_box)

        # Imaging
        image_box = QGroupBox("Imaging")
        image_layout = QVBoxLayout(image_box)

        single_btn = QPushButton("Take Single Image")
        single_btn.clicked.connect(
            self.take_single_image
        )
        image_layout.addWidget(single_btn)

        burst_layout = QHBoxLayout()

        self.burst_nframes_input = QSpinBox()
        self.burst_nframes_input.setRange(1, 5000)
        self.burst_nframes_input.setValue(
            self.burst_nframes_default
        )

        burst_btn = QPushButton("Take Burst")
        burst_btn.clicked.connect(self.take_burst)

        burst_layout.addWidget(QLabel("N frames:"))
        burst_layout.addWidget(
            self.burst_nframes_input
        )
        burst_layout.addWidget(burst_btn)

        image_layout.addLayout(burst_layout)

        self.live_view_checkbox = QCheckBox("Live View")
        self.live_view_checkbox.stateChanged.connect(
            self.toggle_live_view
        )
        image_layout.addWidget(
            self.live_view_checkbox
        )

        save_btn = QPushButton("Save Current Image")
        save_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #0ABAB5;
                color: white;
                border: 2px solid #7FFFD4;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #40E0D0;
            }

            QPushButton:pressed {
                background-color: #008B8B;
            }
            """
        )
        save_btn.clicked.connect(
            self.save_current_image
        )
        image_layout.addWidget(save_btn)

        control_layout.addWidget(image_box)

        ### Pupil Selection
        pupil_box = QGroupBox("Pupil Selection")
        pupil_layout = QGridLayout(pupil_box)

        # Buttons
        self.select_pupil_btn = QPushButton("Select Pupil")
        self.select_pupil_btn.clicked.connect(self.select_pupil)

        self.auto_pupil_btn = QPushButton("Auto")
        self.auto_pupil_btn.clicked.connect(self.auto_pupil)

        self.save_pupil_btn = QPushButton("Save Pupil")
        self.save_pupil_btn.clicked.connect(self.save_pupil)

        pupil_layout.addWidget(self.select_pupil_btn,0,0,)

        pupil_layout.addWidget(self.auto_pupil_btn,0,1,)

        pupil_layout.addWidget(self.save_pupil_btn,0,2,)

        # Pupil Radius
        pupil_layout.addWidget(QLabel("Pupil Radius"),1,0,)

        self.pupil_radius_input = QDoubleSpinBox()
        self.pupil_radius_input.setRange(1.0, 320.0)
        self.pupil_radius_input.setDecimals(1)
        self.pupil_radius_input.setSingleStep(1.0)
        self.pupil_radius_input.setValue(self.pupil_radius)

        radius_minus_btn = QPushButton("<")
        radius_plus_btn = QPushButton(">")

        radius_minus_btn.clicked.connect(
            lambda: self.adjust_pupil_value(self.pupil_radius_input,-1,))

        radius_plus_btn.clicked.connect(
            lambda: self.adjust_pupil_value(self.pupil_radius_input,+1,))

        pupil_layout.addWidget(radius_minus_btn,1,1,)

        pupil_layout.addWidget(self.pupil_radius_input,1,2,)

        pupil_layout.addWidget(radius_plus_btn,1,3,)

        # Pupil Separation
        pupil_layout.addWidget(QLabel("Pupil Separation"),2,0,)

        self.pupil_separation_input = QDoubleSpinBox()
        self.pupil_separation_input.setRange(1.0,320.0,)
        self.pupil_separation_input.setDecimals(1)
        self.pupil_separation_input.setSingleStep(1.0)
        self.pupil_separation_input.setValue(self.pupil_separation)

        separation_minus_btn = QPushButton("<")
        separation_plus_btn = QPushButton(">")

        separation_minus_btn.clicked.connect(
            lambda: self.adjust_pupil_value(self.pupil_separation_input,-1,))

        separation_plus_btn.clicked.connect(
            lambda: self.adjust_pupil_value(self.pupil_separation_input,+1,))

        pupil_layout.addWidget(separation_minus_btn,2,1,)

        pupil_layout.addWidget(self.pupil_separation_input,2,2,)

        pupil_layout.addWidget(separation_plus_btn,2,3,)

        # Mask Center x
        pupil_layout.addWidget(QLabel("Mask Center x"),3,0,)

        self.mask_center_x_input = QDoubleSpinBox()
        self.mask_center_x_input.setRange(1.0,320.0,)
        self.mask_center_x_input.setDecimals(1)
        self.mask_center_x_input.setSingleStep(1.0)
        self.mask_center_x_input.setValue(self.mask_center_x)

        center_x_minus_btn = QPushButton("<")
        center_x_plus_btn = QPushButton(">")

        center_x_minus_btn.clicked.connect(
            lambda: self.adjust_pupil_value(self.mask_center_x_input,-1,))

        center_x_plus_btn.clicked.connect(
            lambda: self.adjust_pupil_value(self.mask_center_x_input,+1,))

        pupil_layout.addWidget(center_x_minus_btn,3,1,)

        pupil_layout.addWidget(self.mask_center_x_input,3,2,)

        pupil_layout.addWidget(center_x_plus_btn,3,3,)

        # Mask Center y
        pupil_layout.addWidget(QLabel("Mask Center y"),4,0,)

        self.mask_center_y_input = QDoubleSpinBox()
        self.mask_center_y_input.setRange(1.0,320.0,)
        self.mask_center_y_input.setDecimals(1)
        self.mask_center_y_input.setSingleStep(1.0)
        self.mask_center_y_input.setValue(self.mask_center_y)

        center_y_minus_btn = QPushButton("<")
        center_y_plus_btn = QPushButton(">")

        center_y_minus_btn.clicked.connect(
            lambda: self.adjust_pupil_value(self.mask_center_y_input,-1,))

        center_y_plus_btn.clicked.connect(
            lambda: self.adjust_pupil_value(self.mask_center_y_input,+1,))

        pupil_layout.addWidget(center_y_minus_btn,4,1,)

        pupil_layout.addWidget(self.mask_center_y_input,4,2,)

        pupil_layout.addWidget(center_y_plus_btn,4,3,)

        # Update mask whenever a value is typed/changed
        self.pupil_radius_input.valueChanged.connect(self.pupil_parameters_changed)

        self.pupil_separation_input.valueChanged.connect(self.pupil_parameters_changed)

        self.mask_center_x_input.valueChanged.connect(self.pupil_parameters_changed)

        self.mask_center_y_input.valueChanged.connect(self.pupil_parameters_changed)

        control_layout.addWidget(pupil_box)
        control_layout.addStretch()
        ###


        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)

        # ---------------- Right panel: image display ----------------------
        self.display_widget = QWidget()
        self.display_widget.setMinimumWidth(600)
        self.display_widget.setMinimumHeight(500)
        self.display_widget.setStyleSheet(
            "background-color: #f0f0f0; "
            "border: 1px solid #ccc;"
        )

        display_layout = QVBoxLayout(
            self.display_widget
        )

        self.figure = Figure(
            figsize=(6, 6),
            dpi=100,
            facecolor="black",
        )

        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)

        self.toolbar = NavigationToolbar(
            self.canvas,
            self.display_widget,
        )

        display_layout.addWidget(self.toolbar)
        display_layout.addWidget(self.canvas)

        top_splitter.addWidget(control_panel)
        top_splitter.addWidget(self.display_widget)
        top_splitter.setSizes(
            self.config.get(
                "gui",
                {},
            ).get(
                "top_splitter_sizes",
                [350, 850],
            )
        )
        top_splitter.setStretchFactor(0, 0)
        top_splitter.setStretchFactor(1, 1)

        # ---------------- Bottom: logger ----------------------------------
        logger_config = self.config.get(
            "logging",
            {},
        )

        self.logger_widget = LoggerWidget(
            name="CRED Control Log",
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
        main_splitter.setStretchFactor(0, 4)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setCollapsible(1, True)
        main_splitter.setSizes(
            self.config.get(
                "gui",
                {},
            ).get(
                "main_splitter_sizes",
                [650, 150],
            )
        )

    # ------------------------------------------------------------------
    # Settings actions
    # ------------------------------------------------------------------
    def get_fps(self):
        """Refresh the read-only current FPS label.

        This does not modify the FPS input, which remains the user's
        proposed next setpoint.
        """
        try:
            value, raw = self.cam.get_fps()

            if value is not None:
                self.fps_current_label.setText(
                    f"{value:g}"
                )
            else:
                self.fps_current_label.setText(raw)

            self.log.info(f"FPS: {raw}")

        except CredOneError as e:
            self.fps_current_label.setText("Error")
            self.log.error(
                f"Failed to get FPS: {e}"
            )

    def set_fps(self):
        try:
            requested_fps = self.fps_input.value()
            raw = self.cam.set_fps(requested_fps)

            self.log.info(
                f"Set FPS to {requested_fps}: {raw}"
            )

            # Refresh with the value the camera actually accepted.
            self.get_fps()

        except CredOneError as e:
            self.log.error(
                f"Failed to set FPS: {e}"
            )
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to set FPS:\n{e}",
            )

    def get_gain(self):
        try:
            value, raw = self.cam.get_gain()

            if value is not None:
                self.gain_current_label.setText(
                    f"{value:g}"
                )
            else:
                self.gain_current_label.setText(raw)

            self.log.info(f"Gain: {raw}")

        except CredOneError as e:
            self.gain_current_label.setText("Error")
            self.log.error(
                f"Failed to get gain: {e}"
            )

    def set_gain(self):
        try:
            requested_gain = self.gain_input.value()
            raw = self.cam.set_gain(requested_gain)

            self.log.info(
                f"Set gain to {requested_gain}: {raw}"
            )

            # Refresh with the value the camera actually accepted.
            self.get_gain()

        except CredOneError as e:
            self.log.error(
                f"Failed to set gain: {e}"
            )
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to set gain:\n{e}",
            )

    def set_readout_mode(self):
        mode = self.mode_dropdown.currentText()
        mode_info = READOUT_MODE_INFO[mode]

        display_name = mode_info["display_name"]
        max_fps = mode_info["max_fps"]
        description = mode_info["description"]

        confirmation = QMessageBox(self)
        confirmation.setIcon(QMessageBox.Warning)
        confirmation.setWindowTitle(
            "Confirm Readout Mode Change"
        )
        confirmation.setText(
            f"Change the readout mode to {display_name}?"
        )
        confirmation.setInformativeText(
            f"{description}\n\n"
            f"Maximum full-frame rate in this mode: "
            f"{max_fps} FPS.\n\n"
            "Changing the readout mode changes the maximum "
            "frame rate available to the camera. The GUI "
            "frame-rate limit will be updated after the "
            "mode is applied.\n\n"
            "Do you want to continue?"
        )
        confirmation.setStandardButtons(
            QMessageBox.Yes | QMessageBox.No
        )
        confirmation.setDefaultButton(
            QMessageBox.No
        )

        response = confirmation.exec_()

        if response != QMessageBox.Yes:
            self.log.info(
                f"Readout mode change to {mode} "
                f"cancelled by user"
            )

            # If this GUI has previously applied a mode, return the
            # dropdown to that mode. On the first cancellation, leave
            # the selection alone because the startup mode is unknown.
            if self.current_readout_mode is not None:
                self.mode_dropdown.setCurrentText(
                    self.current_readout_mode
                )

            return

        try:
            raw = self.cam.set_readout_mode(mode)

            self.log.info(
                f"Set readout mode to {mode}: {raw}"
            )

            self.current_readout_mode = mode

            # Restrict future FPS commands to this mode's full-frame
            # ceiling. Do not command a new FPS here because the camera
            # may change it internally when the mode is applied.
            min_fps = self.fps_range[0]

            self.fps_input.setRange(
                min_fps,
                max_fps,
            )

            self.fps_box.setTitle(
                f"Frame Rate "
                f"({min_fps:g}-{max_fps:g} Hz)"
            )

            # Read back whatever FPS the camera selected after the mode
            # change.
            self.get_fps()

        except CredOneError as e:
            self.log.error(
                f"Failed to set readout mode: {e}"
            )

            if self.current_readout_mode is not None:
                self.mode_dropdown.setCurrentText(
                    self.current_readout_mode
                )

            QMessageBox.critical(
                self,
                "Error",
                f"Failed to set readout mode:\n{e}",
            )

    def set_ndr(self):
        try:
            requested_ndr = self.ndr_input.value()
            raw = self.cam.set_ndr(requested_ndr)

            self.log.info(
                f"Set NDR to {requested_ndr}: {raw}"
            )

        except CredOneError as e:
            self.log.error(
                f"Failed to set NDR: {e}"
            )
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to set NDR:\n{e}",
            )

    def set_rawimages(self):
        on = (
            self.rawimages_dropdown.currentText()
            == "On"
        )

        try:
            raw = self.cam.set_rawimages(on)

            self.log.info(
                f"Set raw images "
                f"{'On' if on else 'Off'}: {raw}"
            )

        except CredOneError as e:
            self.log.error(
                f"Failed to set raw images: {e}"
            )
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to set raw images:\n{e}",
            )

    # ------------------------------------------------------------------
    # Imaging actions
    # ------------------------------------------------------------------
    def set_buttons_enabled(self, enabled):
        for widget in self.findChildren(QPushButton):
            if widget.text() == "Save Current Image":
                continue

            widget.setEnabled(enabled)

    def take_single_image(self):
        if self.operation_in_progress:
            QMessageBox.warning(
                self,
                "Operation in Progress",
                "Another operation is already running.",
            )
            return

        try:
            frame, time_str = self.cam.get_image()

            self.display_image(
                frame,
                title=f"Single frame ({time_str})",
            )

            self.current_capture_meta = {
                "nframes": 1,
            }

            self.log.info(
                f"Captured single image at {time_str}"
            )

            self._prompt_save_then_type(
                frame,
                extra_meta=self.current_capture_meta,
            )

        except Exception as e:
            self.log.error(
                f"Failed to capture image: {e}"
            )
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to capture image:\n{e}",
            )

    def take_burst(self):
        if self.operation_in_progress:
            QMessageBox.warning(
                self,
                "Operation in Progress",
                "Another operation is already running.",
            )
            return

        nframes = self.burst_nframes_input.value()

        reply = QMessageBox.question(
            self,
            "Take Burst",
            (
                f"Capture {nframes} sequential frames? "
                f"This may take a while since each frame "
                f"is grabbed individually.\n\n"
                f"Continue?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            self.log.info(
                "Burst capture cancelled by user"
            )
            return

        try:
            self.operation_in_progress = True
            self.set_buttons_enabled(False)

            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)

            self.log.info(
                f"Starting burst capture of "
                f"{nframes} frames..."
            )

            QApplication.processEvents()

            cube = self.run_with_gui_updates(
                self.cam.get_image_multiframe,
                nframes,
            )

            median_frame = np.median(cube, axis=0)

            self.display_image(
                median_frame,
                title=(
                    f"Burst median "
                    f"({nframes} frames)"
                ),
            )

            self.current_capture_meta = {
                "nframes": nframes,
            }

            self.log.info(
                f"Burst capture complete. "
                f"Cube shape: {cube.shape}"
            )

            self._prompt_save_then_type(
                cube,
                extra_meta=self.current_capture_meta,
            )

        except Exception as e:
            self.log.error(
                f"Burst capture failed: {e}"
            )
            QMessageBox.critical(
                self,
                "Error",
                f"Burst capture failed:\n{e}",
            )

        finally:
            self.operation_in_progress = False
            self.progress_bar.setVisible(False)
            self.set_buttons_enabled(True)

    def run_with_gui_updates(
        self,
        func,
        *args,
        **kwargs,
    ):
        """Run a function in a background thread.

        Qt events continue to be processed while the function runs so
        the GUI remains responsive.
        """
        self.function_complete = False
        self.function_error = None
        self.function_result = None

        last_log_time = time.time()

        def run_function():
            try:
                self.function_result = func(
                    *args,
                    **kwargs,
                )
            except Exception as e:
                self.function_error = e
            finally:
                self.function_complete = True

        thread = threading.Thread(
            target=run_function
        )
        thread.daemon = True
        thread.start()

        while not self.function_complete:
            QApplication.processEvents()

            if time.time() - last_log_time > 30:
                self.log.info(
                    "Operation in progress..."
                )
                last_log_time = time.time()

            time.sleep(0.1)

        thread.join()

        if self.function_error:
            raise self.function_error

        return self.function_result

    def toggle_live_view(self, state):
        self.live_view_enabled = (
            state == QtCore.Qt.Checked
        )

        if self.live_view_enabled:
            self.live_view_timer.start(
                self.live_view_interval
            )
            self.log.info("Live view started")
        else:
            self.live_view_timer.stop()
            self.log.info("Live view stopped")


    def live_view_tick(self):
    if self.operation_in_progress:
        return

    try:
        frame, time_str = self.cam.get_image()

        # Keep the most recently acquired live-view frame
        self.last_live_frame = frame.copy()

        self.display_image(frame,title=f"Live view ({time_str})",)

        self.current_capture_meta = {"nframes": 1,}

    except Exception as e:
        self.log.warning(f"Live view frame failed: {e}")

    # ------------------------------------------------------------------
    # Display / save
    # ------------------------------------------------------------------
    def display_placeholder_image(self):
        self.axes.clear()
        self.axes.set_facecolor("black")
        self.figure.patch.set_facecolor("black")

        self.axes.text(
            0.5,
            0.5,
            "C-RED ONE Display",
            ha="center",
            va="center",
            transform=self.axes.transAxes,
            fontsize=14,
            color="white",
            fontweight="bold",
        )

        self.axes.set_xlim(0, 1)
        self.axes.set_ylim(0, 1)
        self.axes.set_xticks([])
        self.axes.set_yticks([])

        self.canvas.draw()

    def display_image(
        self,
        image_array,
        title="",
    ):
        try:
            self.current_image_array = (
                image_array.copy()
            )

            self.figure.clear()
            self.axes = self.figure.add_subplot(111)

            self.figure.patch.set_facecolor("black")
            self.axes.set_facecolor("black")

            for spine in self.axes.spines.values():
                spine.set_color("white")

            im = self.axes.imshow(
                image_array,
                cmap="magma",
                aspect="equal",
            )

            self.colorbar = self.figure.colorbar(
                im,
                ax=self.axes,
                shrink=0.8,
            )

            self.colorbar.ax.tick_params(
                colors="white",
                labelsize=10,
            )

            self.axes.set_xlabel(
                "X (pixels)",
                color="white",
                fontsize=12,
                fontweight="bold",
            )

            self.axes.set_ylabel(
                "Y (pixels)",
                color="white",
                fontsize=12,
                fontweight="bold",
            )

            self.axes.tick_params(
                colors="white",
                labelsize=10,
            )

            stats = (
                f"min: {image_array.min():.0f}  "
                f"max: {image_array.max():.0f}  "
                f"mean: {image_array.mean():.1f}"
            )

            if title:
                full_title = f"{title}\n{stats}"
            else:
                full_title = stats

            self.axes.set_title(
                full_title,
                color="white",
                fontsize=11,
                fontweight="bold",
            )

            self.figure.subplots_adjust(
                left=0.1,
                right=0.94,
                top=0.9,
                bottom=0.12,
            )

            self.canvas.draw()

        except Exception as e:
            self.log.error(
                f"Failed to display image: {e}"
            )
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to display image:\n{e}",
            )

    def _choose_type_and_save(
        self,
        array,
        extra_meta=None,
    ):
        """Ask for dark/flat and file format, then save.

        Returns the saved path, or None if the operation was cancelled
        or failed.
        """
        choice, ok = QInputDialog.getItem(
            self,
            "Save Image",
            "Save as:",
            list(SAVE_OPTIONS.keys()),
            0,
            False,
        )

        if not ok:
            self.log.info(
                "Save cancelled at type selection"
            )
            return None

        image_type, save_type = SAVE_OPTIONS[choice]

        try:
            path = self.cam.save_image(
                array,
                self.data_root,
                save_type=save_type,
                extra_meta=extra_meta,
                subfolder=image_type,
            )

        except OSError as e:
            self.log.error(
                f"Failed to save: {e}"
            )
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save:\n{e}",
            )
            return None

        self.log.info(f"Saved to {path}")

        QMessageBox.information(
            self,
            "Saved",
            f"Saved to:\n{path}",
        )

        return path

    def _prompt_save_then_type(
        self,
        array,
        extra_meta=None,
    ):
        """Ask whether to save, then request the type and format."""
        reply = QMessageBox.question(
            self,
            "Save Image",
            "Save this capture?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            self.log.info(
                "Save skipped by user"
            )
            return None

        return self._choose_type_and_save(
            array,
            extra_meta=extra_meta,
        )

    def save_current_image(self):
        """Save the image currently displayed in the GUI.

        For a burst, this saves the displayed median frame. The complete
        burst cube is offered immediately after the burst is captured.
        """
        if self.current_image_array is None:
            QMessageBox.warning(
                self,
                "No Image",
                "No image is currently displayed to save.",
            )

            self.log.warning(
                "Attempted to save image but no image "
                "is currently displayed"
            )
            return

        self._choose_type_and_save(
            self.current_image_array,
            extra_meta=self.current_capture_meta,
        )
        

    ### Pupil functions
    def adjust_pupil_value(self, spinbox, amount):
        """Increase or decrease a pupil parameter by one pixel"""

        new_value = spinbox.value() + amount

        new_value = max(spinbox.minimum(),min(spinbox.maximum(), new_value),)

        spinbox.setValue(new_value)

            
    def get_pupil_centers(self):
        """Return the four pupil centers defining the square mask."""

        half_separation = (self.pupil_separation / 2.0)

        centers = [
            (self.mask_center_x - half_separation,self.mask_center_y - half_separation,),
            (self.mask_center_x + half_separation,self.mask_center_y - half_separation,),
            (self.mask_center_x - half_separation,self.mask_center_y + half_separation,),
            (self.mask_center_x + half_separation,self.mask_center_y + half_separation,),]

        return centers
    def select_pupil(self):
        """Enter pupil-selection mode using the last live frame."""

        if self.last_live_frame is None:
            QMessageBox.warning(
                self,
                "No Live Frame",
                "No live frame is available.\n\n"
                "Start Live View and acquire at least "
                "one frame first.",)
            return

        # Stop live view
        if self.live_view_enabled:
            self.live_view_checkbox.setChecked(False)

        self.pupil_selection_active = True

        self.log.info("Entering pupil selection mode" )

        # Display the last live frame.
        self.display_pupil_selection()

    def display_pupil_selection(self):
        """Display last live frame with the four-pupil mask."""

        if self.last_live_frame is None:
            return

        image_array = self.last_live_frame

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        self.figure.patch.set_facecolor("black")
        self.axes.set_facecolor("black")

        for spine in self.axes.spines.values():
            spine.set_color("white")

        im = self.axes.imshow(image_array,cmap="magma",origin="lower",aspect="equal",)

        self.colorbar = self.figure.colorbar(im,ax=self.axes,shrink=0.8,)

        self.colorbar.ax.tick_params(colors="white",labelsize=10,)

        self.axes.set_xlabel("X (pixels)",color="white",fontsize=12,fontweight="bold",)

        self.axes.set_ylabel("Y (pixels)",color="white",fontsize=12,fontweight="bold",)

        self.axes.tick_params(colors="white",labelsize=10,)

        self.axes.set_title("Pupil Selection",color="white",fontsize=11,fontweight="bold",)

        # Remove any old circles
        self.pupil_circles = []

        # Get the four pupil centers
        centers = self.get_pupil_centers()

        # Draw the four pupils
        for center_x, center_y in centers:
            circle = Circle((center_x,center_y,),
                            self.pupil_radius,
                            fill=False,
                            edgecolor="cyan",
                            linewidth=2,)

            self.axes.add_patch(circle)

            self.pupil_circles.append(circle)

        self.figure.subplots_adjust(left=0.1,right=0.94,top=0.9,bottom=0.12,)

        self.canvas.draw()

    def pupil_parameters_changed(self):
        """Update pupil mask values and redraw the mask."""

        self.pupil_radius = (self.pupil_radius_input.value())

        self.pupil_separation = (self.pupil_separation_input.value())

        self.mask_center_x = (self.mask_center_x_input.value())

        self.mask_center_y = (self.mask_center_y_input.value())

        if self.pupil_selection_active:
            self.update_pupil_circles()



    def update_pupil_circles(self):
        """Update the existing four circles without reacquiring an image"""

        if not self.pupil_selection_active:
            return

        if len(self.pupil_circles) != 4:
            self.display_pupil_selection()
            return

        centers = self.get_pupil_centers()

        for circle, (x, y) in zip(self.pupil_circles,centers,):
            circle.center = (x, y)
            circle.radius = self.pupil_radius

        self.canvas.draw_idle()



    def auto_pupil(self):
        """Reset pupil mask parameters to the default values."""

        self.pupil_radius_input.setValue(self.default_pupil_radius)

        self.pupil_separation_input.setValue(self.default_pupil_separation)

        self.mask_center_x_input.setValue(self.default_mask_center_x)

        self.mask_center_y_input.setValue(self.default_mask_center_y)

        self.log.info("Pupil mask reset to default values")


    def save_pupil(self):
        """Save the current pupil mask and exit pupil selection mode"""

        if not self.pupil_selection_active:
            self.log.info(
                "Save Pupil pressed while not in "
                "pupil selection mode")
            return

        # Read the current values.
        self.pupil_radius = (self.pupil_radius_input.value())

        self.pupil_separation = (self.pupil_separation_input.value())

        self.mask_center_x = (self.mask_center_x_input.value())

        self.mask_center_y = (self.mask_center_y_input.value())

        # Store them in the configuration dictionary
        self.config["pupil_selection"] = {
            "radius": self.pupil_radius,
            "separation": self.pupil_separation,
            "center_x": self.mask_center_x,
            "center_y": self.mask_center_y,}

        # Save configuration to YAML
        config_path = (Path(__file__).parent/ "cred_control_gui_config.yaml")

        try:
            with open(config_path, "w") as f:
                yaml.safe_dump(self.config,f,sort_keys=False,)

            self.log.info(
                "Pupil mask saved: "
                f"radius={self.pupil_radius}, "
                f"separation={self.pupil_separation}, "
                f"center_x={self.mask_center_x}, "
                f"center_y={self.mask_center_y}")

        except Exception as e:
            self.log.error(f"Failed to save pupil configuration: {e}")

            QMessageBox.critical(self,"Save Error",f"Failed to save pupil configuration:\n{e}",)
            return

        # Exit pupil-selection mode
        self.pupil_selection_active = False

        # Remove circles from the display
        self.pupil_circles = []

        # Return to normal display
        self.display_image(self.last_live_frame,title="Last Live Frame",)

        self.log.info("Pupil selection complete")



    ###








class CredControlMainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.log = setup_logger()

        self.setWindowTitle(
            config.get(
                "gui",
                {},
            ).get(
                "window_title",
                "C-RED ONE Control",
            )
        )

        geom = config.get(
            "gui",
            {},
        ).get(
            "window_geometry",
            [100, 100, 1200, 800],
        )

        self.setGeometry(*geom)
        self.setMinimumSize(1000, 700)

        if self.config.get(
            "styling",
            {},
        ).get(
            "use_keck_theme",
            True,
        ):
            try:
                self.apply_keck_theme()
            except Exception as e:
                self.log.warning(
                    f"Could not apply Keck theme: {e}"
                )

        self.statusBar().showMessage("Ready")

        self.widget = CredControlWidget(config)
        self.setCentralWidget(self.widget)

    def apply_keck_theme(self):
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
                ) as fh:
                    self.setStyleSheet(fh.read())

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
                f"using compatibility theme"
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

            QLineEdit,
            QSpinBox,
            QDoubleSpinBox,
            QComboBox {
                background-color: #404040;
                color: white;
                border: 1px solid #483D8B;
                border-radius: 3px;
                padding: 4px;
            }
            """
        )


def load_config(config_file=None):
    try:
        if config_file is None:
            config_path = (
                Path(__file__).parent
                / "cred_control_gui_config.yaml"
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

        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}

    except yaml.YAMLError as e:
        print(
            f"Error parsing YAML configuration: {e}. "
            f"Using defaults."
        )
        return {}


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("CRED Control")
    app.setApplicationDisplayName(
        "C-RED ONE Control GUI"
    )

    config = load_config()
    cam_cfg = config.get("cam", {})

    # Promote to a top-level key before config["cam"] gets overwritten
    # with the live controller instance below.
    config["data_root"] = cam_cfg.get(
        "data_root",
        "/usr/local/aodev/CRED-One/Data",
    )

    try:
        config["cam"] = CredOneController(
            edt_dir=cam_cfg.get(
                "edt_dir",
                "/opt/EDTpdv",
            ),
            tmp_frame_path=cam_cfg.get(
                "tmp_frame_path",
                "/usr/local/aodev/CRED-One/Data/tmp/"
                "CRED_frame.raw",
            ),
            lock_path=cam_cfg.get(
                "lock_path",
                "/tmp/pycred_camera_io.lock",
            ),
            skip_serial_while_taking=False,
        )
    except CredOneError as e:
        QMessageBox.critical(
            None,
            "C-RED ONE Initialization Error",
            str(e),
        )
        sys.exit(1)

    window = CredControlMainWindow(config)
    window.show()

    sys.exit(app.exec_())

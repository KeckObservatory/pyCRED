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
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
                              QGroupBox, QPushButton, QCheckBox, QLabel,
                              QSpinBox, QDoubleSpinBox, QGridLayout,
                              QComboBox, QApplication, QSizePolicy,
                              QMainWindow, QMessageBox, QLineEdit, QFileDialog,
                              QProgressBar)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# --- local project paths ---------------------------------------------------
# Adjust to match wherever cred_controller.py / pyCRED live in your
# environment (mirrors the sys.path.insert block in the DM template).
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

# TODO: confirm the full/authoritative list of readout modes against the
# C-RED ONE manual. globalresetsingle is the camera's actual default mode
# (confirmed on hardware); globalresetbursts is the only other one
# exercised in cred_test_prelim.py.
READOUT_MODES = [
    "globalresetsingle",
    "globalresetbursts",
    "globalresetcds",
    "rollingresetsingle",
    "rollingresetcds",
]


class CredControlWidget(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cam = config.get("cam", CredOneController())
        self.log = config.get("logger", setup_logger())
        self.cam.log = self.log

        fps_cfg = config.get("fps", {})
        self.fps_range = fps_cfg.get("range", [1, 3500])
        self.fps_default = fps_cfg.get("default", 3500)

        gain_cfg = config.get("gain", {})
        self.gain_range = gain_cfg.get("range", [0, 1000])
        self.gain_default = gain_cfg.get("default", 1)

        ndr_cfg = config.get("ndr", {})
        self.ndr_range = ndr_cfg.get("range", [1, 200])
        self.ndr_default = ndr_cfg.get("default", 1)

        burst_cfg = config.get("burst", {})
        self.burst_nframes_default = burst_cfg.get("nframes_default", 50)

        self.cred_cfg_path = config.get("cred_cfg_path",
                                         "~/../../usr/local/aodev/CRED-One/cred_default.cfg")

        self.operation_in_progress = False
        self.current_image_array = None
        self.current_burst_cube = None
        self.live_view_enabled = False
        self.live_view_interval = config.get("live_view", {}).get("interval_ms", 500)
        self.live_view_timer = QTimer()
        self.live_view_timer.timeout.connect(self.live_view_tick)

        self.setupUI()
        self.display_placeholder_image()

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
        fps_box = QGroupBox(f"Frame Rate ({self.fps_range[0]}-{self.fps_range[1]} Hz)")
        fps_layout = QHBoxLayout(fps_box)
        self.fps_input = QDoubleSpinBox()
        self.fps_input.setRange(*self.fps_range)
        self.fps_input.setValue(self.fps_default)
        fps_get_btn = QPushButton("Get")
        fps_get_btn.clicked.connect(self.get_fps)
        fps_set_btn = QPushButton("Set")
        fps_set_btn.clicked.connect(self.set_fps)
        fps_layout.addWidget(self.fps_input)
        fps_layout.addWidget(fps_get_btn)
        fps_layout.addWidget(fps_set_btn)
        control_layout.addWidget(fps_box)

        # Gain
        gain_box = QGroupBox(f"Gain ({self.gain_range[0]}-{self.gain_range[1]})")
        gain_layout = QHBoxLayout(gain_box)
        self.gain_input = QDoubleSpinBox()
        self.gain_input.setRange(*self.gain_range)
        self.gain_input.setValue(self.gain_default)
        gain_get_btn = QPushButton("Get")
        gain_get_btn.clicked.connect(self.get_gain)
        gain_set_btn = QPushButton("Set")
        gain_set_btn.clicked.connect(self.set_gain)
        gain_layout.addWidget(self.gain_input)
        gain_layout.addWidget(gain_get_btn)
        gain_layout.addWidget(gain_set_btn)
        control_layout.addWidget(gain_box)

        # Readout mode
        mode_box = QGroupBox("Readout Mode")
        mode_layout = QHBoxLayout(mode_box)
        self.mode_dropdown = QComboBox()
        self.mode_dropdown.addItems(READOUT_MODES)
        mode_set_btn = QPushButton("Set")
        mode_set_btn.clicked.connect(self.set_readout_mode)
        mode_layout.addWidget(self.mode_dropdown)
        mode_layout.addWidget(mode_set_btn)
        control_layout.addWidget(mode_box)

        # Frame grabber re-init (the README's `initcam` startup step).
        # Worth trying if `take` starts failing with a pdv_multibuf /
        # ring-buffer error after changing readout mode -- the board's
        # ROI/tap config can end up stale relative to the camera's
        # current mode.
        reinit_box = QGroupBox("Frame Grabber")
        reinit_layout = QVBoxLayout(reinit_box)
        reinit_btn = QPushButton("Re-init Frame Grabber (initcam)")
        reinit_btn.clicked.connect(self.reinit_framegrabber)
        reinit_layout.addWidget(reinit_btn)
        control_layout.addWidget(reinit_box)

        # NDR + raw images
        ndr_box = QGroupBox(f"NDR / Raw Images")
        ndr_layout = QGridLayout(ndr_box)
        self.ndr_input = QSpinBox()
        self.ndr_input.setRange(*self.ndr_range)
        self.ndr_input.setValue(self.ndr_default)
        ndr_set_btn = QPushButton("Set NDR")
        ndr_set_btn.clicked.connect(self.set_ndr)
        self.rawimages_dropdown = QComboBox()
        self.rawimages_dropdown.addItems(["Off", "On"])
        rawimages_set_btn = QPushButton("Set Raw Images")
        rawimages_set_btn.clicked.connect(self.set_rawimages)
        ndr_layout.addWidget(QLabel("NDR:"), 0, 0)
        ndr_layout.addWidget(self.ndr_input, 0, 1)
        ndr_layout.addWidget(ndr_set_btn, 0, 2)
        ndr_layout.addWidget(QLabel("Raw images:"), 1, 0)
        ndr_layout.addWidget(self.rawimages_dropdown, 1, 1)
        ndr_layout.addWidget(rawimages_set_btn, 1, 2)
        control_layout.addWidget(ndr_box)

        # Imaging
        image_box = QGroupBox("Imaging")
        image_layout = QVBoxLayout(image_box)

        single_btn = QPushButton("Take Single Image")
        single_btn.clicked.connect(self.take_single_image)
        image_layout.addWidget(single_btn)

        burst_layout = QHBoxLayout()
        self.burst_nframes_input = QSpinBox()
        self.burst_nframes_input.setRange(1, 5000)
        self.burst_nframes_input.setValue(self.burst_nframes_default)
        burst_btn = QPushButton("Take Burst")
        burst_btn.clicked.connect(self.take_burst)
        burst_layout.addWidget(QLabel("N frames:"))
        burst_layout.addWidget(self.burst_nframes_input)
        burst_layout.addWidget(burst_btn)
        image_layout.addLayout(burst_layout)

        self.live_view_checkbox = QCheckBox("Live View")
        self.live_view_checkbox.stateChanged.connect(self.toggle_live_view)
        image_layout.addWidget(self.live_view_checkbox)

        save_btn = QPushButton("Save Current Image")
        save_btn.setStyleSheet("""
            QPushButton { background-color: #0ABAB5; color: white; border: 2px solid #7FFFD4;
                          border-radius: 5px; padding: 8px 16px; font-weight: bold; }
            QPushButton:hover { background-color: #40E0D0; }
            QPushButton:pressed { background-color: #008B8B; }
        """)
        save_btn.clicked.connect(self.save_current_image)
        image_layout.addWidget(save_btn)

        control_layout.addWidget(image_box)
        control_layout.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)

        # ---------------- Right panel: image display ------------------------
        self.display_widget = QWidget()
        self.display_widget.setMinimumWidth(600)
        self.display_widget.setMinimumHeight(500)
        self.display_widget.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        display_layout = QVBoxLayout(self.display_widget)

        self.figure = Figure(figsize=(6, 6), dpi=100, facecolor="black")
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self.display_widget)
        display_layout.addWidget(self.toolbar)
        display_layout.addWidget(self.canvas)

        top_splitter.addWidget(control_panel)
        top_splitter.addWidget(self.display_widget)
        top_splitter.setSizes(self.config.get("gui", {}).get("top_splitter_sizes", [350, 850]))
        top_splitter.setStretchFactor(0, 0)
        top_splitter.setStretchFactor(1, 1)

        # ---------------- Bottom: logger -------------------------------------
        logger_config = self.config.get("logging", {})
        self.logger_widget = LoggerWidget(
            name="CRED Control Log",
            max_lines=logger_config.get("max_lines", 300),
            min_height=100,
            font_size=logger_config.get("font_size", 8),
        )

        main_splitter.addWidget(top_widget)
        main_splitter.addWidget(self.logger_widget)
        main_splitter.setStretchFactor(0, 4)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setCollapsible(1, True)
        main_splitter.setSizes(self.config.get("gui", {}).get("main_splitter_sizes", [650, 150]))

    # ------------------------------------------------------------------
    # Readback / settings actions
    # ------------------------------------------------------------------
    def get_fps(self):
        try:
            value, raw = self.cam.get_fps()
            if value is not None:
                self.fps_input.setValue(value)
            self.log.info(f"FPS: {raw}")
        except CredOneError as e:
            self.log.error(f"Failed to get FPS: {e}")
            QMessageBox.critical(self, "Error", f"Failed to get FPS:\n{e}")

    def set_fps(self):
        try:
            raw = self.cam.set_fps(self.fps_input.value())
            self.log.info(f"Set FPS to {self.fps_input.value()}: {raw}")
        except CredOneError as e:
            self.log.error(f"Failed to set FPS: {e}")
            QMessageBox.critical(self, "Error", f"Failed to set FPS:\n{e}")

    def get_gain(self):
        try:
            value, raw = self.cam.get_gain()
            if value is not None:
                self.gain_input.setValue(value)
            self.log.info(f"Gain: {raw}")
        except CredOneError as e:
            self.log.error(f"Failed to get gain: {e}")
            QMessageBox.critical(self, "Error", f"Failed to get gain:\n{e}")

    def set_gain(self):
        try:
            raw = self.cam.set_gain(self.gain_input.value())
            self.log.info(f"Set gain to {self.gain_input.value()}: {raw}")
        except CredOneError as e:
            self.log.error(f"Failed to set gain: {e}")
            QMessageBox.critical(self, "Error", f"Failed to set gain:\n{e}")

    def set_readout_mode(self):
        mode = self.mode_dropdown.currentText()
        try:
            raw = self.cam.set_readout_mode(mode)
            self.log.info(f"Set readout mode to {mode}: {raw}")
        except CredOneError as e:
            self.log.error(f"Failed to set readout mode: {e}")
            QMessageBox.critical(self, "Error", f"Failed to set readout mode:\n{e}")

    def reinit_framegrabber(self):
        reply = QMessageBox.question(
            self, "Re-init Frame Grabber",
            "This re-runs the frame grabber's initcam startup step and will "
            "briefly interrupt acquisition. Only needed if 'take' is failing "
            "(e.g. a pdv_multibuf / ring-buffer error) after changing readout "
            "mode.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            self.log.info("Frame grabber re-init cancelled by user")
            return
        try:
            raw = self.cam.reinit_framegrabber(self.cred_cfg_path)
            self.log.info(f"Frame grabber re-initialized: {raw}")
        except CredOneError as e:
            self.log.error(f"Failed to re-init frame grabber: {e}")
            QMessageBox.critical(self, "Error", f"Failed to re-init frame grabber:\n{e}")

    def set_ndr(self):
        try:
            raw = self.cam.set_ndr(self.ndr_input.value())
            self.log.info(f"Set NDR to {self.ndr_input.value()}: {raw}")
        except CredOneError as e:
            self.log.error(f"Failed to set NDR: {e}")
            QMessageBox.critical(self, "Error", f"Failed to set NDR:\n{e}")

    def set_rawimages(self):
        on = self.rawimages_dropdown.currentText() == "On"
        try:
            raw = self.cam.set_rawimages(on)
            self.log.info(f"Set raw images {'On' if on else 'Off'}: {raw}")
        except CredOneError as e:
            self.log.error(f"Failed to set raw images: {e}")
            QMessageBox.critical(self, "Error", f"Failed to set raw images:\n{e}")

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
            QMessageBox.warning(self, "Operation in Progress", "Another operation is already running.")
            return
        try:
            frame, time_str = self.cam.get_image()
            self.display_image(frame, title=f"Single frame ({time_str})")
            self.log.info(f"Captured single image at {time_str}")
        except Exception as e:
            self.log.error(f"Failed to capture image: {e}")
            QMessageBox.critical(self, "Error", f"Failed to capture image:\n{e}")

    def take_burst(self):
        if self.operation_in_progress:
            QMessageBox.warning(self, "Operation in Progress", "Another operation is already running.")
            return

        nframes = self.burst_nframes_input.value()
        reply = QMessageBox.question(
            self, "Take Burst",
            f"Capture {nframes} sequential frames? This may take a while "
            f"since each frame is grabbed individually.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            self.log.info("Burst capture cancelled by user")
            return

        try:
            self.operation_in_progress = True
            self.set_buttons_enabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.log.info(f"Starting burst capture of {nframes} frames...")
            QApplication.processEvents()

            cube = self.run_with_gui_updates(self.cam.get_image_multiframe, nframes, clear_directory=True)

            median_frame = np.median(cube, axis=0)
            self.display_image(median_frame, title=f"Burst median ({nframes} frames)")
            self.current_burst_cube = cube
            self.log.info(f"Burst capture complete. Cube shape: {cube.shape}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"cred_burst_{timestamp}.npy"
            reply = QMessageBox.question(
                self, "Save Burst", f"Save the {nframes}-frame cube as {default_filename}?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                file_path, _ = QFileDialog.getSaveFileName(self, "Save Burst Cube", default_filename,
                                                             "NumPy files (*.npy);;All files (*.*)")
                if file_path:
                    if not file_path.lower().endswith(".npy"):
                        file_path += ".npy"
                    np.save(file_path, cube)
                    self.log.info(f"Burst cube saved as {file_path}")

        except Exception as e:
            self.log.error(f"Burst capture failed: {e}")
            QMessageBox.critical(self, "Error", f"Burst capture failed:\n{e}")
        finally:
            self.operation_in_progress = False
            self.progress_bar.setVisible(False)
            self.set_buttons_enabled(True)

    def run_with_gui_updates(self, func, *args, **kwargs):
        """Run func in a background thread while keeping the GUI responsive,
        same pattern as DMReceptionWidget.run_with_gui_updates."""
        self.function_complete = False
        self.function_error = None
        self.function_result = None
        last_log_time = time.time()

        def run_function():
            try:
                self.function_result = func(*args, **kwargs)
            except Exception as e:
                self.function_error = e
            finally:
                self.function_complete = True

        thread = threading.Thread(target=run_function)
        thread.daemon = True
        thread.start()

        while not self.function_complete:
            QApplication.processEvents()
            if time.time() - last_log_time > 30:
                self.log.info("Operation in progress...")
                last_log_time = time.time()
            time.sleep(0.1)

        thread.join()
        if self.function_error:
            raise self.function_error
        return self.function_result

    def toggle_live_view(self, state):
        self.live_view_enabled = state == 2  # Qt.Checked
        if self.live_view_enabled:
            self.live_view_timer.start(self.live_view_interval)
            self.log.info("Live view started")
        else:
            self.live_view_timer.stop()
            self.log.info("Live view stopped")

    def live_view_tick(self):
        if self.operation_in_progress:
            return
        try:
            frame, time_str = self.cam.get_image()
            self.display_image(frame, title=f"Live view ({time_str})")
        except Exception as e:
            self.log.warning(f"Live view frame failed: {e}")

    # ------------------------------------------------------------------
    # Display / save
    # ------------------------------------------------------------------
    def display_placeholder_image(self):
        self.axes.clear()
        self.axes.set_facecolor("black")
        self.figure.patch.set_facecolor("black")
        self.axes.text(0.5, 0.5, "C-RED ONE Display", ha="center", va="center",
                        transform=self.axes.transAxes, fontsize=14, color="white",
                        fontweight="bold")
        self.axes.set_xlim(0, 1)
        self.axes.set_ylim(0, 1)
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.canvas.draw()

    def display_image(self, image_array, title=""):
        try:
            self.current_image_array = image_array.copy()

            self.figure.clear()
            self.axes = self.figure.add_subplot(111)
            self.figure.patch.set_facecolor("black")
            self.axes.set_facecolor("black")
            for spine in self.axes.spines.values():
                spine.set_color("white")

            im = self.axes.imshow(image_array, cmap="magma", aspect="equal")
            self.colorbar = self.figure.colorbar(im, ax=self.axes, shrink=0.8)
            self.colorbar.ax.tick_params(colors="white", labelsize=10)

            self.axes.set_xlabel("X (pixels)", color="white", fontsize=12, fontweight="bold")
            self.axes.set_ylabel("Y (pixels)", color="white", fontsize=12, fontweight="bold")
            self.axes.tick_params(colors="white", labelsize=10)

            stats = f"min: {image_array.min():.0f}  max: {image_array.max():.0f}  mean: {image_array.mean():.1f}"
            full_title = f"{title}\n{stats}" if title else stats
            self.axes.set_title(full_title, color="white", fontsize=11, fontweight="bold")

            self.figure.subplots_adjust(left=0.1, right=0.94, top=0.9, bottom=0.12)
            self.canvas.draw()
        except Exception as e:
            self.log.error(f"Failed to display image: {e}")
            QMessageBox.critical(self, "Error", f"Failed to display image:\n{e}")

    def save_current_image(self):
        if self.current_image_array is None:
            QMessageBox.warning(self, "No Image", "No image is currently displayed to save.")
            self.log.warning("Attempted to save image but no image is currently displayed")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"cred_image_{timestamp}.npy"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save CRED Image", default_filename,
                                                     "NumPy files (*.npy);;All files (*.*)")
        if not file_path:
            self.log.info("Save operation cancelled by user")
            return
        if not file_path.lower().endswith(".npy"):
            file_path += ".npy"
        np.save(file_path, self.current_image_array)
        self.log.info(f"Current image saved as {file_path}")
        QMessageBox.information(self, "Image Saved", f"Image saved successfully as:\n{file_path}")


class CredControlMainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.log = setup_logger()
        self.setWindowTitle(config.get("gui", {}).get("window_title", "C-RED ONE Control"))

        geom = config.get("gui", {}).get("window_geometry", [100, 100, 1200, 800])
        self.setGeometry(*geom)
        self.setMinimumSize(1000, 700)

        if self.config.get("styling", {}).get("use_keck_theme", True):
            try:
                self.apply_keck_theme()
            except Exception as e:
                self.log.warning(f"Could not apply Keck theme: {e}")

        self.statusBar().showMessage("Ready")
        self.widget = CredControlWidget(config)
        self.setCentralWidget(self.widget)

    def apply_keck_theme(self):
        try:
            stylesheet_path = os.path.join(os.path.dirname(__file__), "..", "keck_theme", "keck_dark_purple.qss")
            if os.path.exists(stylesheet_path):
                with open(stylesheet_path, "r") as fh:
                    self.setStyleSheet(fh.read())
                self.log.info("Full Keck theme applied")
            else:
                self.log.warning("Keck theme file not found, using compatibility theme")
                self._apply_compatibility_theme()
        except Exception as e:
            self.log.error(f"Error applying Keck theme: {e}, using compatibility theme")
            self._apply_compatibility_theme()

    def _apply_compatibility_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #2b2b2b; color: white; }
            QGroupBox { color: white; border: 2px solid #483D8B; border-radius: 5px;
                        margin-top: 1ex; font-weight: bold; padding-top: 15px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center;
                                left: 10px; padding: 0 5px 0 5px; }
            QPushButton { background-color: #483D8B; color: white; border: 2px solid #357ABD;
                          border-radius: 5px; padding: 8px 16px; font-weight: bold; }
            QPushButton:hover { background-color: #357ABD; }
            QPushButton:pressed { background-color: #2A5A8B; }
            QLabel { color: white; background: transparent; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #404040; color: white; border: 1px solid #483D8B;
                border-radius: 3px; padding: 4px;
            }
        """)


def load_config(config_file=None):
    try:
        if config_file is None:
            config_path = (Path(__file__).parent / "cred_control_gui_config.yaml").resolve()
        else:
            config_path = Path(config_file).resolve()
        if not config_path.exists():
            print(f"Configuration file not found at: {config_path}. Using defaults.")
            return {}
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}. Using defaults.")
        return {}


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("CRED Control")
    app.setApplicationDisplayName("C-RED ONE Control GUI")

    config = load_config()
    cam_cfg = config.get("cam", {})
    config["cred_cfg_path"] = cam_cfg.get("cred_cfg_path",
                                           "~/../../usr/local/aodev/CRED-One/cred_default.cfg")
    config["cam"] = CredOneController(
        edt_dir=cam_cfg.get("edt_dir", "/opt/EDTpdv"),
        tmp_frame_path=cam_cfg.get("tmp_frame_path",
                                    "/usr/local/aodev/CRED-One/Data/tmp/CRED_frame.raw"),
        take_nbuffers=cam_cfg.get("take_nbuffers", 200),
    )

    window = CredControlMainWindow(config)
    window.show()
    sys.exit(app.exec_())

"""
cred_controller.py

Thin Python wrapper around the FLI C-RED ONE / EDT frame-grabber
command-line tools (`serial_cmd`, `take`) that already exist in
cred_test_prelim.py and CREDOne_Disp.py.

This plays the same role that `asdk.AlpaoDeformableMirror()` and
`zygo.Zygo()` play in the DM acceptance-test GUI: a single object that
GUIs pass around and call methods on, so the GUI code never has to know
about subprocess/serial_cmd details directly. Both cred_monitor_gui.py
and cred_control_gui.py import and use this class -- do not duplicate
subprocess calls in either GUI file.

Nothing here talks to a Python SDK. As confirmed, the only interface to
the camera is the EDT `serial_cmd` (settings/status) and `take`
(frame grabbing) command-line tools, run from /opt/EDTpdv, exactly as in
cred_test_prelim.py.
"""
import os
import re
import subprocess
from datetime import datetime
from glob import glob

import numpy as np


class CredOneError(RuntimeError):
    """Raised when a serial_cmd / take call fails or returns unexpected output."""
    pass


class CredOneController:
    # (rows, cols) -- matches cred_default.cfg (320 wide x 256 high) and
    # the reshape used in cred_test_prelim.get_image()
    FRAME_SHAPE = (256, 320)

    def __init__(self, edt_dir="/opt/EDTpdv",
                 tmp_frame_path="/home/aodev/CRED-One/Data/tmp/CRED_frame.raw",
                 logger=None):
        self.edt_dir = edt_dir
        self.tmp_frame_path = tmp_frame_path
        self.log = logger

    # ------------------------------------------------------------------
    # low-level helpers
    # ------------------------------------------------------------------
    def _run(self, cmd, timeout=30):
        """Run a shell command from self.edt_dir, return (stdout, stderr, rc)."""
        full_cmd = f"cd {self.edt_dir}; {cmd}"
        try:
            result = subprocess.run(full_cmd, shell=True, capture_output=True,
                                     text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise CredOneError(f"Command timed out after {timeout}s: {cmd}") from e
        if self.log:
            self.log.debug(f"CMD: {cmd} -> rc={result.returncode} "
                            f"stdout={result.stdout.strip()!r} "
                            f"stderr={result.stderr.strip()!r}")
        return result.stdout.strip(), result.stderr.strip(), result.returncode

    def _serial(self, arg, timeout=30):
        """Run `./serial_cmd '<arg>'` in self.edt_dir, return stdout text.

        NOTE: cred_test_prelim.py never checks serial_cmd's return code --
        it just prints stdout/stderr and moves on, since return-code
        semantics for this tool weren't verified. This mirrors that: a
        nonzero rc is logged as a warning (visible in the GUI logger) but
        does NOT raise, so it won't block a call that actually succeeded.
        CredOneError is reserved for cases we're sure indicate a real
        failure (e.g. the subprocess timing out).
        """
        stdout, stderr, rc = self._run(f"./serial_cmd '{arg}'", timeout=timeout)
        if rc != 0:
            msg = f"serial_cmd '{arg}' returned rc={rc}: {stderr or stdout}"
            if self.log:
                self.log.warning(msg)
            else:
                print(f"WARNING: {msg}")
        return stdout

    @staticmethod
    def _first_float(text):
        match = re.search(r"(-?\d+\.?\d*(?:[eE]-?\d+)?)", text)
        return float(match.group(1)) if match else None

    # ------------------------------------------------------------------
    # status / health / cooling
    # ------------------------------------------------------------------
    def get_status(self):
        """Raw text from `serial_cmd status`. Per the README this contains
        tokens such as 'ready', 'isbeingcooled', or 'operational' -- see the
        C-RED ONE user manual for the complete state list and meanings."""
        return self._serial("status")

    def get_temperature(self):
        """
        Returns (raw_text, parsed_dict). raw_text is exactly what
        `serial_cmd temperature` prints. parsed_dict is a best-effort parse
        of "<name>: <value>" style tokens (e.g. the cryopt(pulseTube) field
        the README references) into {name: float}. If a field you need
        isn't showing up in parsed_dict, read it out of raw_text instead --
        the parser is intentionally permissive rather than exhaustive since
        the exact output format wasn't available to verify against hardware.
        """
        raw = self._serial("temperature")
        parsed = {}
        for match in re.finditer(r"([A-Za-z0-9_\(\)]+)\s*[:=]\s*(-?\d+\.?\d*)", raw):
            name, val = match.groups()
            try:
                parsed[name] = float(val)
            except ValueError:
                pass
        return raw, parsed

    def get_pressure(self):
        """
        Returns (raw_text, value_or_None) from `serial_cmd pressure`.
        TODO: confirm the exact serial_cmd syntax/units for pressure
        (mbar vs Torr, single value vs multiple sensors) against the
        C-RED ONE manual or live camera -- this wasn't exercised against
        real hardware output in the source repo, so treat raw_text as the
        source of truth until this is validated.
        """
        raw = self._serial("pressure")
        return raw, self._first_float(raw)

    def set_cooling(self, on=True):
        return self._serial(f"set cooling {'on' if on else 'off'}")

    # ------------------------------------------------------------------
    # readout settings
    # ------------------------------------------------------------------
    def get_fps(self):
        raw = self._serial("fps")
        return self._first_float(raw), raw

    def set_fps(self, fps):
        return self._serial(f"set fps {fps}")

    def get_gain(self):
        raw = self._serial("gain")
        return self._first_float(raw), raw

    def set_gain(self, gain):
        return self._serial(f"set gain {gain}")

    def set_readout_mode(self, mode):
        """
        mode: a C-RED ONE acquisition mode string, e.g. 'globalresetbursts'
        (the only one exercised in cred_test_prelim.py). Other candidate
        modes documented for this camera family include
        'globalresetsingle', 'globalresetcds', 'rollingresetsingle', and
        'rollingresetcds' -- TODO confirm the authoritative list (e.g. via
        `serial_cmd modes` or the manual) before exposing all of them in a
        GUI dropdown.
        """
        return self._serial(f"set mode {mode}")

    def set_rawimages(self, on=True):
        return self._serial(f"set rawimages {'on' if on else 'off'}")

    def set_ndr(self, ndr):
        """Set number of non-destructive reads before reset (nbreadworeset)."""
        return self._serial(f"set nbreadworeset {ndr}")

    # ------------------------------------------------------------------
    # image acquisition (mirrors cred_test_prelim.get_image /
    # get_image_multiframe exactly, wrapped as methods)
    # ------------------------------------------------------------------
    def get_image(self):
        """Grab a single frame via `./take -N 200 -f <tmp_frame_path>`.
        Returns (frame_array, timestamp_str).

        Same forgiving-rc note as _serial(): cred_test_prelim.get_image()
        never checks `take`'s return code either -- it just reads the raw
        file afterward. We do the same, and only raise CredOneError if the
        file that comes back doesn't actually contain a full frame, since
        that's the one failure mode we can detect for certain.
        """
        os.makedirs(os.path.dirname(self.tmp_frame_path), exist_ok=True)
        cmd = f"./take -N 200 -f '{self.tmp_frame_path}'"
        stdout, stderr, rc = self._run(cmd, timeout=60)
        if rc != 0:
            msg = f"take returned rc={rc}: {stderr or stdout}"
            if self.log:
                self.log.warning(msg)
            else:
                print(f"WARNING: {msg}")

        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        im = np.fromfile(self.tmp_frame_path, dtype=np.uint16)
        expected = self.FRAME_SHAPE[0] * self.FRAME_SHAPE[1]
        if im.size != expected:
            raise CredOneError(
                f"Unexpected frame size {im.size} (expected {expected}); "
                f"check camera is streaming and tmp_frame_path is correct.")
        return im.reshape(self.FRAME_SHAPE), time_str

    def get_image_multiframe(self, nframes, clear_directory=False):
        """Capture nframes sequential single frames into one
        (nframes, 256, 320) cube. Slow -- calls get_image() in a loop,
        same as cred_test_prelim.get_image_multiframe()."""
        frame_dir = os.path.dirname(self.tmp_frame_path)
        if clear_directory:
            for f in glob(os.path.join(frame_dir, "CRED_frame*")):
                os.remove(f)
        cube = np.zeros((nframes,) + self.FRAME_SHAPE)
        for i in range(nframes):
            frame, _ = self.get_image()
            cube[i] = frame
        return cube

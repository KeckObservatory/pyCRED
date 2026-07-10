"""
cred_controller.py

Thin wrapper around the C-RED ONE's serial_cmd / take command-line tools,
matching the same calls cred_test_prelim.py already uses. Kept simple on
purpose: no retries, no polling, no auto-recovery, and it never deletes
any files -- if something needs to change here, change it deliberately,
don't let this module do it as a side effect.

The only thing added beyond a direct port of cred_test_prelim.py's
subprocess calls is running commands in their own process group so a
timeout kills the whole command tree (including 'take') instead of
orphaning it -- that's what let a hung 'take' hold /dev/edt0 for over a
day in practice. Everything else here is a straight match to the
existing scripts.
"""
import os
import re
import signal
import subprocess
from datetime import datetime

import numpy as np


class CredOneError(RuntimeError):
    """Raised when a serial_cmd / take call fails in a way we can detect
    for certain (a timeout, or a frame file that can't be read)."""
    pass


class CredOneController:
    FRAME_SHAPE = (256, 320)  # (rows, cols)

    def __init__(self, edt_dir="/opt/EDTpdv",
                 tmp_frame_path="/usr/local/aodev/CRED-One/Data/tmp/CRED_frame.raw",
                 logger=None):
        self.edt_dir = edt_dir
        self.tmp_frame_path = tmp_frame_path
        self.log = logger

    # ------------------------------------------------------------------
    def _run(self, cmd, timeout=60):
        """Run a shell command from self.edt_dir. Own process group so a
        timeout kills 'take' too, not just the shell wrapper around it."""
        full_cmd = f"cd {self.edt_dir}; {cmd}"
        proc = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True,
                                 start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            stdout, stderr = proc.communicate()
            raise CredOneError(f"Command timed out after {timeout}s: {cmd}")

        if self.log:
            self.log.debug(f"CMD: {cmd} -> rc={proc.returncode} "
                            f"stdout={stdout.strip()!r} stderr={stderr.strip()!r}")
        return stdout.strip(), stderr.strip(), proc.returncode

    def _serial(self, arg):
        """Run `./serial_cmd '<arg>'`. Doesn't check rc -- serial_cmd's
        return code isn't reliable on this hardware, same as in
        cred_test_prelim.py, which never checks it either."""
        stdout, stderr, rc = self._run(f"./serial_cmd '{arg}'")
        if rc != 0 and self.log:
            self.log.warning(f"serial_cmd '{arg}' rc={rc}: {stderr or stdout}")
        return stdout

    @staticmethod
    def _first_float(text):
        match = re.search(r"(-?\d+\.?\d*(?:[eE]-?\d+)?)", text)
        return float(match.group(1)) if match else None

    # ---- status ----
    def get_status(self):
        return self._serial("status")

    def get_temperature(self):
        """Returns (raw_text, parsed_dict). parsed_dict is a best-effort
        parse of "<name>: <value>" tokens -- if a field you need isn't in
        it, read raw_text instead."""
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
        raw = self._serial("pressure")
        return raw, self._first_float(raw)

    def set_cooling(self, on=True):
        return self._serial(f"set cooling {'on' if on else 'off'}")

    # ---- readout settings ----
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

    def get_mode(self):
        """e.g. raw = 'Mode: globalresetsingle'."""
        raw = self._serial("mode")
        match = re.search(r":\s*(\S+)", raw)
        return (match.group(1) if match else raw), raw

    def set_readout_mode(self, mode):
        return self._serial(f"set mode {mode}")

    def get_ndr(self):
        """NOTE: inferred by pattern (bare command name = getter, same as
        fps/gain/mode) -- not confirmed against hardware output."""
        raw = self._serial("nbreadworeset")
        return self._first_float(raw), raw

    def set_ndr(self, ndr):
        return self._serial(f"set nbreadworeset {ndr}")

    def set_rawimages(self, on=True):
        return self._serial(f"set rawimages {'on' if on else 'off'}")

    # ---- imaging ----
    def get_image(self):
        """Grab a single frame via `take`, same as
        cred_test_prelim.get_image(): run take, then read the raw file.
        No retries, no polling, no deleting anything."""
        os.makedirs(os.path.dirname(self.tmp_frame_path), exist_ok=True)
        cmd = f"./take -N 200 -f '{self.tmp_frame_path}'"
        stdout, stderr, rc = self._run(cmd)
        if rc != 0 and self.log:
            self.log.warning(f"take rc={rc}: {stderr or stdout}")

        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            im = np.fromfile(self.tmp_frame_path, dtype=np.uint16)
        except OSError as e:
            raise CredOneError(f"Could not read frame file {self.tmp_frame_path}: {e}") from e

        expected = self.FRAME_SHAPE[0] * self.FRAME_SHAPE[1]
        if im.size != expected:
            raise CredOneError(f"Unexpected frame size {im.size} (expected {expected})")
        return im.reshape(self.FRAME_SHAPE), time_str

    def get_image_multiframe(self, nframes):
        """Capture nframes sequential frames into one cube. Never deletes
        anything -- each get_image() call just overwrites tmp_frame_path,
        same as calling it in a loop by hand."""
        cube = np.zeros((nframes,) + self.FRAME_SHAPE)
        for i in range(nframes):
            frame, _ = self.get_image()
            cube[i] = frame
        return cube

    # ---- metadata, for saving alongside an image ----
    def get_metadata(self):
        """Collect current settings/environment to save alongside an
        image (mode, gain, fps, ndr, temperature, pressure) so it can be
        converted to a FITS header later if needed. Best-effort: any
        field that fails to read is None rather than raising."""
        meta = {}
        try:
            meta["mode"], _ = self.get_mode()
        except CredOneError:
            meta["mode"] = None
        try:
            meta["gain"], _ = self.get_gain()
        except CredOneError:
            meta["gain"] = None
        try:
            meta["fps"], _ = self.get_fps()
        except CredOneError:
            meta["fps"] = None
        try:
            meta["ndr"], _ = self.get_ndr()
        except CredOneError:
            meta["ndr"] = None
        try:
            temp_raw, temp_parsed = self.get_temperature()
            meta["temperature_raw"] = temp_raw
            for name, val in temp_parsed.items():
                key = re.sub(r"[^0-9a-zA-Z_]", "_", f"temp_{name}")
                meta[key] = val
        except CredOneError:
            meta["temperature_raw"] = None
        try:
            pressure_raw, pressure_val = self.get_pressure()
            meta["pressure_raw"] = pressure_raw
            meta["pressure"] = pressure_val
        except CredOneError:
            meta["pressure_raw"] = None
            meta["pressure"] = None
        return meta

    # ---- saving ----
    def save_image(self, image_array, data_root, save_type=".npy", extra_meta=None, subfolder=None):
        """
        Save image_array under data_root/<YYYYMMDD>/[<subfolder>/]<YYYYMMDD>_<HHMMSS>.ext
        (folders created if needed). subfolder is meant for a shot
        classification like "dark" or "flat" -- pass None to skip it.
        This is the only place that builds the save path or writes the
        file -- the GUI just calls this and shows the result.

        save_type: ".npy" saves the array only (plain, no metadata --
        that's what .npy is for). ".npz" additionally collects
        mode/gain/fps/ndr/temperature/pressure via get_metadata() and
        saves them as separate top-level keys alongside the image, so
        they map directly to FITS header cards later if needed.
        extra_meta: optional dict of additional fields to include in the
        .npz (e.g. {"nframes": 50} for a burst cube) -- ignored for .npy
        since plain arrays can't carry metadata.

        Returns the full path saved to.
        """
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")
        save_dir = os.path.join(data_root, date_str)
        if subfolder:
            save_dir = os.path.join(save_dir, subfolder)
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{date_str}_{time_str}{save_type}"
        path = os.path.join(save_dir, filename)

        if save_type == ".npz":
            meta = self.get_metadata()
            if extra_meta:
                meta.update(extra_meta)
            if subfolder:
                meta["image_type"] = subfolder
            np.savez(path, image=image_array, timestamp=now.isoformat(), **meta)
        else:
            np.save(path, image_array)

        return path

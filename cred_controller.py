"""
cred_controller.py

Thin wrapper around the C-RED ONE's serial_cmd / take command-line tools,
matching the same calls cred_test_prelim.py already uses. Kept simple on
purpose: no retries, no polling, no auto-recovery, and it never deletes
any files.

Camera-I/O locking
------------------
serial_cmd operations take a shared advisory flock. The ./take operation
takes an exclusive advisory flock. Therefore an image acquisition waits
for any monitor command already in flight, then prevents new monitor
commands from starting until acquisition is complete.

The monitor GUI uses a nonblocking shared lock for each complete refresh
cycle. If ./take owns the exclusive lock, the refresh is skipped and the
monitor remains responsive. The control GUI's normal commands wait for the
exclusive acquisition lock when necessary.

For a multiframe capture, the exclusive lock is held for the whole burst,
so monitor polling cannot slip between individual ./take calls. Linux
releases flock locks automatically if a process exits or crashes.
"""

import errno
import fcntl
import os
import re
import signal
import subprocess
import threading
from contextlib import contextmanager
from datetime import datetime

import numpy as np


class CredOneError(RuntimeError):
    """Raised when a C-RED ONE command or file operation fails."""
    pass


class CredOneBusyError(CredOneError):
    """Raised when nonblocking monitor access is denied during ./take."""
    pass


class CredOneController:
    FRAME_SHAPE = (256, 320)  # (rows, cols)
    DEFAULT_LOCK_PATH = "/tmp/pycred_camera_io.lock"

    def __init__(
        self,
        edt_dir="/opt/EDTpdv",
        tmp_frame_path="/usr/local/aodev/CRED-One/Data/tmp/CRED_frame.raw",
        logger=None,
        lock_path=None,
        skip_serial_while_taking=False,
        # Retained only for compatibility with older GUI calls.
        client_name=None,
        acquire_lock=None,
    ):
        self.edt_dir = edt_dir
        self.tmp_frame_path = tmp_frame_path
        self.log = logger
        self.lock_path = lock_path or self.DEFAULT_LOCK_PATH
        self.skip_serial_while_taking = skip_serial_while_taking
        self.client_name = client_name

        # Each thread tracks whether it already owns this controller's
        # lock context. This allows a monitor refresh to hold one shared
        # lock around status + temperature + pressure while the individual
        # getter methods call _serial() inside that same context.
        self._lock_state = threading.local()

    # ------------------------------------------------------------------
    # Camera I/O locking
    # ------------------------------------------------------------------
    @contextmanager
    def camera_io_lock(self, exclusive=False, blocking=True):
        """Hold the cross-process camera-I/O lock.

        Shared locks are used by serial_cmd operations. Exclusive locks are
        used by ./take. A nonblocking shared request raises
        CredOneBusyError instead of freezing the caller.

        The context is re-entrant within one thread. A shared lock cannot be
        upgraded to exclusive inside the same context, but shared operations
        may be nested inside an existing exclusive context.
        """
        depth = getattr(self._lock_state, "depth", 0)

        if depth:
            held_exclusive = self._lock_state.exclusive
            if exclusive and not held_exclusive:
                raise CredOneError(
                    "Cannot upgrade an active shared camera lock to an "
                    "exclusive acquisition lock."
                )

            self._lock_state.depth = depth + 1
            try:
                yield
            finally:
                self._lock_state.depth -= 1
            return

        lock_directory = os.path.dirname(self.lock_path)
        if lock_directory:
            os.makedirs(lock_directory, exist_ok=True)

        try:
            lock_file = open(self.lock_path, "a+", encoding="utf-8")
        except OSError as e:
            raise CredOneError(
                f"Could not open camera I/O lock file {self.lock_path}: {e}"
            ) from e

        lock_flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if not blocking:
            lock_flags |= fcntl.LOCK_NB

        try:
            fcntl.flock(lock_file.fileno(), lock_flags)
        except OSError as e:
            lock_file.close()

            if e.errno in (errno.EACCES, errno.EAGAIN):
                raise CredOneBusyError(
                    "Image acquisition is in progress. Monitor polling is "
                    "temporarily paused until ./take finishes."
                ) from e

            raise CredOneError(
                f"Could not acquire camera I/O lock {self.lock_path}: {e}"
            ) from e

        self._lock_state.depth = 1
        self._lock_state.exclusive = exclusive
        self._lock_state.lock_file = lock_file

        try:
            yield
        finally:
            self._lock_state.depth = 0
            self._lock_state.exclusive = False
            self._lock_state.lock_file = None

            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                lock_file.close()

    def monitor_poll(self):
        """Return a nonblocking shared-lock context for one monitor cycle."""
        return self.camera_io_lock(exclusive=False, blocking=False)

    # ------------------------------------------------------------------
    def _run(self, cmd, timeout=60):
        """Run a shell command from self.edt_dir.

        Each command gets its own process group so a timeout kills the
        complete command tree, including take.
        """
        full_cmd = f"cd {self.edt_dir}; {cmd}"
        proc = subprocess.Popen(
            full_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            stdout, stderr = proc.communicate()
            raise CredOneError(
                f"Command timed out after {timeout}s: {cmd}"
            )

        if self.log:
            self.log.debug(
                f"CMD: {cmd} -> rc={proc.returncode} "
                f"stdout={stdout.strip()!r} stderr={stderr.strip()!r}"
            )

        return stdout.strip(), stderr.strip(), proc.returncode

    def _serial(self, arg):
        """Run ./serial_cmd under a shared camera-I/O lock."""
        blocking = not self.skip_serial_while_taking

        with self.camera_io_lock(exclusive=False, blocking=blocking):
            stdout, stderr, rc = self._run(f"./serial_cmd '{arg}'")

        if rc != 0 and self.log:
            self.log.warning(
                f"serial_cmd '{arg}' rc={rc}: {stderr or stdout}"
            )
        return stdout

    @staticmethod
    def _first_float(text):
        match = re.search(r"(-?\d+\.?\d*(?:[eE]-?\d+)?)", text)
        return float(match.group(1)) if match else None

    # ---- status ----
    def get_status(self):
        return self._serial("status")

    def get_temperature(self):
        """Return (raw_text, parsed_dict)."""
        raw = self._serial("temperature")
        parsed = {}
        for match in re.finditer(
            r"([A-Za-z0-9_\(\)]+)\s*[:=]\s*(-?\d+\.?\d*)", raw
        ):
            name, value = match.groups()
            try:
                parsed[name] = float(value)
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
        """Return the parsed mode and raw response."""
        raw = self._serial("mode")
        match = re.search(r":\s*(\S+)", raw)
        return (match.group(1) if match else raw), raw

    def set_readout_mode(self, mode):
        return self._serial(f"set mode {mode}")

    def get_ndr(self):
        raw = self._serial("nbreadworeset")
        return self._first_float(raw), raw

    def set_ndr(self, ndr):
        return self._serial(f"set nbreadworeset {ndr}")

    def set_rawimages(self, on=True):
        return self._serial(f"set rawimages {'on' if on else 'off'}")

    # ---- imaging ----
    def _get_image_unlocked(self):
        """Run one ./take command. Caller must hold the exclusive lock."""
        os.makedirs(os.path.dirname(self.tmp_frame_path), exist_ok=True)
        cmd = f"./take -N 200 -f '{self.tmp_frame_path}'"
        stdout, stderr, rc = self._run(cmd)

        if rc != 0 and self.log:
            self.log.warning(f"take rc={rc}: {stderr or stdout}")

        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            image = np.fromfile(self.tmp_frame_path, dtype=np.uint16)
        except OSError as e:
            raise CredOneError(
                f"Could not read frame file {self.tmp_frame_path}: {e}"
            ) from e

        expected = self.FRAME_SHAPE[0] * self.FRAME_SHAPE[1]
        if image.size != expected:
            raise CredOneError(
                f"Unexpected frame size {image.size} (expected {expected})"
            )

        return image.reshape(self.FRAME_SHAPE), time_str

    def get_image(self):
        """Acquire one frame while blocking monitor serial commands."""
        if self.log:
            self.log.debug("Waiting for exclusive camera lock for ./take")

        with self.camera_io_lock(exclusive=True, blocking=True):
            if self.log:
                self.log.debug("Exclusive camera lock acquired for ./take")
            return self._get_image_unlocked()

    def get_image_multiframe(self, nframes):
        """Capture a burst while keeping monitor polling paused throughout."""
        cube = np.zeros((nframes,) + self.FRAME_SHAPE)

        if self.log:
            self.log.debug(
                f"Waiting for exclusive camera lock for {nframes}-frame burst"
            )

        with self.camera_io_lock(exclusive=True, blocking=True):
            if self.log:
                self.log.debug(
                    f"Exclusive camera lock acquired for {nframes}-frame burst"
                )

            for index in range(nframes):
                frame, _ = self._get_image_unlocked()
                cube[index] = frame

        return cube

    # ---- metadata ----
    def get_metadata(self):
        """Collect current camera settings on a best-effort basis."""
        metadata = {}

        try:
            metadata["mode"], _ = self.get_mode()
        except CredOneError:
            metadata["mode"] = None
        try:
            metadata["gain"], _ = self.get_gain()
        except CredOneError:
            metadata["gain"] = None
        try:
            metadata["fps"], _ = self.get_fps()
        except CredOneError:
            metadata["fps"] = None
        try:
            metadata["ndr"], _ = self.get_ndr()
        except CredOneError:
            metadata["ndr"] = None
        try:
            temperature_raw, temperature_parsed = self.get_temperature()
            metadata["temperature_raw"] = temperature_raw
            for name, value in temperature_parsed.items():
                key = re.sub(r"[^0-9a-zA-Z_]", "_", f"temp_{name}")
                metadata[key] = value
        except CredOneError:
            metadata["temperature_raw"] = None
        try:
            pressure_raw, pressure_value = self.get_pressure()
            metadata["pressure_raw"] = pressure_raw
            metadata["pressure"] = pressure_value
        except CredOneError:
            metadata["pressure_raw"] = None
            metadata["pressure"] = None

        return metadata

    # ---- saving ----
    def save_image(
        self,
        image_array,
        data_root,
        save_type=".npy",
        extra_meta=None,
        subfolder=None,
    ):
        """Save an image or cube under the dated data directory."""
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")
        save_directory = os.path.join(data_root, date_str)

        if subfolder:
            save_directory = os.path.join(save_directory, subfolder)
        os.makedirs(save_directory, exist_ok=True)

        filename = f"{date_str}_{time_str}{save_type}"
        path = os.path.join(save_directory, filename)

        if save_type == ".npz":
            metadata = self.get_metadata()
            if extra_meta:
                metadata.update(extra_meta)
            if subfolder:
                metadata["image_type"] = subfolder
            np.savez(
                path,
                image=image_array,
                timestamp=now.isoformat(),
                **metadata,
            )
        else:
            np.save(path, image_array)

        return path

"""Scratch folder: finished results of the scans open in the editor.

A result of a 600 dpi scan takes 100-200 MB in memory, so only the last few
stay there. The rest are kept here, as 16-bit data (exact for 8- and 16-bit
saving, half the size of the float32 image), so going back to any scan in
the list - or saving them all - does not need it to be processed again.

Each editor session has its own folder (``session-<pid>-<time>``) inside the
base folder (Options > Processing > Scratch folder, default: the system's
temporary folder). It is deleted when the editor closes; folders of sessions
that did not close normally are removed at the next start.
"""
from __future__ import annotations

import itertools
import os
import shutil
import tempfile
import threading
import time

import numpy as np

_count = itertools.count()


def base_dir(setting: str = "") -> str:
    return setting.strip() or os.path.join(tempfile.gettempdir(), "CoverCleaner-scratch")


def save_result(path: str, img: np.ndarray) -> str:
    """Write a float image (0..1) as 16-bit data; returns `path`."""
    data = (np.clip(img, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    tmp = path + ".part.npy"
    np.save(tmp, data)
    os.replace(tmp, path)
    return path


def load_result(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32) * np.float32(1.0 / 65535.0)


def _alive(pid: int) -> bool:
    try:
        from .update_helper import _alive as alive
        return alive(pid)
    except Exception:
        return True                                  # unsure: leave it alone


def clean_stale(base: str) -> None:
    """Remove folders left by editor sessions that are no longer running."""
    try:
        names = os.listdir(base)
    except OSError:
        return
    for name in names:
        if not name.startswith("session-"):
            continue
        try:
            pid = int(name.split("-")[1])
        except (IndexError, ValueError):
            continue
        if pid != os.getpid() and not _alive(pid):
            shutil.rmtree(os.path.join(base, name), ignore_errors=True)


class Scratch:
    """This editor session's scratch folder (created on first use)."""

    def __init__(self, setting: str = ""):
        self.base = base_dir(setting)
        self.dir = os.path.join(self.base, f"session-{os.getpid()}-{time.strftime('%Y%m%d%H%M%S')}")
        self._made = False
        self._lock = threading.Lock()
        threading.Thread(target=clean_stale, args=(self.base,), daemon=True).start()

    def new_path(self) -> str:
        with self._lock:
            if not self._made:
                os.makedirs(self.dir, exist_ok=True)
                self._made = True
        return os.path.join(self.dir, f"r{next(_count):05d}.npy")

    def free_bytes(self) -> int | None:
        try:
            return shutil.disk_usage(self.dir if self._made else self.base
                                     if os.path.isdir(self.base) else tempfile.gettempdir()).free
        except OSError:
            return None

    @staticmethod
    def remove(path: str | None) -> None:
        if path:
            try:
                os.remove(path)
            except OSError:
                pass

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

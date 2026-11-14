"""Crash reports for the editor.

The GUI usually runs windowless (pythonw), where an error message or a hard
crash leaves no trace at all. This module records:

* a session log (``session.log``): what the program was doing - files
  opened, Auto runs, renders, background jobs, free memory - so a report
  shows what led up to a problem;
* Python errors that nothing else catches, in any thread: written to a
  ``crash-<time>.txt`` report straight away (and shown by the GUI);
* hard crashes - the process dying inside native code such as OpenCV, Qt,
  onnxruntime or a GPU driver - via faulthandler, which writes the Python
  stack of every thread to ``fault.log`` as the process dies.

A ``running`` marker is created at start and removed on a normal exit. If
it is still there at the next start, the previous session ended abnormally
and a report is assembled from ``fault.log`` and the end of the previous
session's log.

Everything stays in the logs folder next to settings.json; nothing is sent
anywhere. Reports contain file names and paths of the scans opened.
"""
from __future__ import annotations

import faulthandler
import logging
import os
import platform
import sys
import threading
import time
import traceback

from . import __version__

log = logging.getLogger("covercleaner")
ISSUES_URL = "https://github.com/biopanda24/Cover-Cleaner/issues/new"

_state = {"dir": None, "fault": None, "on_error": None}


def log_dir() -> str:
    from .settings import settings_path
    return os.path.join(os.path.dirname(settings_path()), "logs")


def _stamp() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")


def environment() -> str:
    """Versions and system facts for the top of a report."""
    lines = [f"Cover Cleaner {__version__}",
             f"Python {platform.python_version()} ({sys.executable})",
             f"System {platform.platform()} {platform.machine()}"]
    for name in ("numpy", "cv2", "PySide6", "onnxruntime"):
        mod = sys.modules.get(name)
        if mod is not None:
            lines.append(f"{name} {getattr(mod, '__version__', '?')}")
    try:
        from .cli import available_memory
        mem = available_memory()
        if mem:
            lines.append(f"Free memory {mem / 1e9:.1f} GB")
    except Exception:
        pass
    return "\n".join(lines)


def _tail(path: str, lines: int = 400) -> str:
    try:
        with open(path, encoding="utf8", errors="replace") as fh:
            return "".join(fh.readlines()[-lines:])
    except OSError:
        return ""


def write_report(title: str, body: str, session_log: str | None = None) -> str:
    """Write a crash report; returns its path."""
    d = _state["dir"] or log_dir()
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"crash-{_stamp()}.txt")
    log_text = _tail(session_log or os.path.join(d, "session.log"))
    with open(path, "w", encoding="utf8") as fh:
        fh.write(f"{title}\n{'=' * len(title)}\n\n{environment()}\n\n{body.rstrip()}\n\n"
                 f"---- last lines of the session log ----\n{log_text}")
    _prune(d)
    return path


def _prune(d: str, keep: int = 20) -> None:
    reports = sorted(f for f in os.listdir(d) if f.startswith("crash-"))
    for f in reports[:-keep]:
        try:
            os.remove(os.path.join(d, f))
        except OSError:
            pass


def _previous_session(d: str) -> str | None:
    """A report on the last session if it did not end normally."""
    marker = os.path.join(d, "running")
    if not os.path.exists(marker):
        return None
    fault = _tail(os.path.join(d, "fault.log"), 2000).strip()
    prev_log = os.path.join(d, "session.prev.log")
    if fault:
        body = ("The program stopped inside native code (OpenCV, Qt, onnxruntime, a GPU "
                "driver...). Python stack of every thread at that moment:\n\n" + fault +
                "\n\n(Lines like 'Windows fatal exception: code 0xe06d7363' before the last one "
                "are C++ exceptions that were handled; the last exception is what stopped it.)")
        title = "Cover Cleaner crashed"
    else:
        body = ("The previous session did not close normally, but no error was recorded. It "
                "may have been ended from Task Manager, by shutting the computer down, or by "
                "the system when memory ran out (see free memory in the log below).")
        title = "Cover Cleaner did not close normally"
    try:
        os.replace(os.path.join(d, "session.log"), prev_log)
    except OSError:                         # still open (or gone): read it where it is
        prev_log = os.path.join(d, "session.log")
    return write_report(title, body, prev_log)


def install(on_error=None) -> str | None:
    """Start logging and crash capture for this session. `on_error(path)` is
    called (on whatever thread) after an uncaught Python error was written
    to a report. Returns the path of a report on the previous session if
    that ended abnormally."""
    d = log_dir()
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return None
    _state["dir"], _state["on_error"] = d, on_error
    previous = _previous_session(d)
    session = os.path.join(d, "session.log")
    if previous is None and os.path.exists(session):
        try:
            os.replace(session, os.path.join(d, "session.prev.log"))
        except OSError:
            pass
    handler = logging.FileHandler(session, mode="w", encoding="utf8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(logging.INFO)
    log.propagate = False
    log.info("session start\n%s", environment())
    with open(os.path.join(d, "running"), "w") as fh:
        fh.write(f"{os.getpid()} {_stamp()}\n")
    try:
        _state["fault"] = open(os.path.join(d, "fault.log"), "w", encoding="utf8")
        faulthandler.enable(_state["fault"], all_threads=True)
    except OSError:
        pass
    sys.excepthook = _excepthook
    threading.excepthook = lambda a: _excepthook(a.exc_type, a.exc_value, a.exc_traceback,
                                                 a.thread.name if a.thread else "thread")
    return previous


def _excepthook(etype, value, tb, where: str = "main thread") -> None:
    text = "".join(traceback.format_exception(etype, value, tb))
    log.error("uncaught error in %s:\n%s", where, text)
    try:
        path = write_report("Cover Cleaner error", f"Uncaught error in the {where}:\n\n{text}")
    except OSError:
        path = None
    cb = _state["on_error"]
    if cb is not None and path:
        try:
            cb(path)
        except Exception:
            pass
    if sys.stderr is not None:
        sys.__excepthook__(etype, value, tb)


def qt_message_handler(mode, context, message) -> None:
    """Qt's own warnings and errors, into the session log."""
    level = {0: logging.DEBUG, 1: logging.WARNING, 2: logging.ERROR,
             3: logging.CRITICAL, 4: logging.INFO}.get(int(getattr(mode, "value", mode)),
                                                       logging.INFO)
    if level >= logging.WARNING:
        log.log(level, "Qt: %s", message)


def clean_exit() -> None:
    """Call when the program closes normally."""
    log.info("session end")
    d = _state["dir"]
    if not d:
        return
    try:
        faulthandler.disable()
        if _state["fault"] is not None:
            _state["fault"].close()
    except Exception:
        pass
    try:
        os.remove(os.path.join(d, "running"))
    except OSError:
        pass

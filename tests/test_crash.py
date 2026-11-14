"""Crash reports: normal and abnormal exits, uncaught errors."""
import faulthandler
import os
import sys
import threading

import pytest

from covercleaner import crash


@pytest.fixture
def logs(tmp_path, monkeypatch):
    monkeypatch.setenv("COVERCLEANER_SETTINGS", str(tmp_path / "settings.json"))
    hooks = sys.excepthook, threading.excepthook
    yield tmp_path / "logs"
    crash.clean_exit()
    sys.excepthook, threading.excepthook = hooks
    if not faulthandler.is_enabled():
        faulthandler.enable()               # pytest's own


def _reports(d):
    return sorted(p for p in os.listdir(d) if p.startswith("crash-"))


def test_normal_exit_leaves_no_report(logs):
    assert crash.install() is None
    crash.log.info("hello")
    crash.clean_exit()
    assert not (logs / "running").exists()
    assert crash.install() is None and _reports(logs) == []


def test_abnormal_exit_is_reported_next_time(logs):
    crash.install()
    crash.log.info("opened something.png")
    # no clean_exit: the process "died"
    report = crash.install()
    text = open(report, encoding="utf8").read()
    assert "did not close normally" in text and "opened something.png" in text


def test_native_crash_stack_is_in_the_report(logs):
    crash.install()
    with open(logs / "fault.log", "w", encoding="utf8") as fh:
        fh.write("Fatal Python error: Segmentation fault\n  File \"x.py\", line 1 in boom\n")
    report = crash.install()
    text = open(report, encoding="utf8").read()
    assert text.startswith("Cover Cleaner crashed") and "in boom" in text


def test_uncaught_error_writes_a_report_and_calls_back(logs):
    seen = []
    crash.install(on_error=seen.append)
    try:
        raise ValueError("bad value 42")
    except ValueError:
        crash._excepthook(*sys.exc_info())
    assert len(seen) == 1 and "bad value 42" in open(seen[0], encoding="utf8").read()

"""Install a downloaded update once Cover Cleaner has closed.

Started by updater.start_update() as a separate process (a copy of this
file outside the app folder, standard library only):

  1. wait for the program (--pid) to exit;
  2. move the app folder aside, copy the new one in, keep models/;
  3. install the new requirements (plus GPU packages if they were set up);
  4. on any failure put the old app folder back;
  5. start Cover Cleaner again.

Everything is logged to <root>/update.log.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

KEEP = ("models",)        # folders inside the app that belong to the user


def _log(root: str, msg: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    try:
        with open(os.path.join(root, "update.log"), "a", encoding="utf8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        SYNCHRONIZE = 0x00100000
        h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not h:
            return False
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(h, 0) == 0x102   # WAIT_TIMEOUT
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _retry(fn, tries: int = 20, delay: float = 0.5):
    """File operations on Windows can fail briefly while handles close."""
    for k in range(tries):
        try:
            return fn()
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(delay)


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _install_requirements(a, target: str) -> None:
    req = os.path.join(target, "requirements.txt")
    if a.uv and os.path.isfile(a.uv):
        base = [a.uv, "pip", "install", "--python", a.python]
    else:
        base = [a.python, "-m", "pip", "install"]
    for cmd in (base + ["-r", req], base + list(a.extra) if a.extra else None):
        if cmd is None:
            continue
        r = subprocess.run(cmd, capture_output=True, text=True, creationflags=_no_window())
        _log(a.root, f"$ {' '.join(cmd)}\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
        if r.returncode != 0:
            raise RuntimeError(f"installing requirements failed (exit {r.returncode})")


def _relaunch(a, target: str) -> None:
    py = a.python
    if sys.platform == "win32":
        w = os.path.join(os.path.dirname(py), "pythonw.exe")
        py = w if os.path.isfile(w) else py
    env = dict(os.environ, PYTHONPATH=target)
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([py, "-m", "covercleaner", "gui"], cwd=target, env=env,
                     creationflags=flags, close_fds=True,
                     start_new_session=(sys.platform != "win32"))


def run(a) -> int:
    target = os.path.abspath(a.target)
    backup = target + ".old"
    _log(a.root, f"update to {a.version}: waiting for the program to close")
    t0 = time.time()
    while _alive(a.pid) and time.time() - t0 < 120:
        time.sleep(0.5)
    if _alive(a.pid):
        _log(a.root, "the program did not close; update cancelled")
        if a.cleanup:
            shutil.rmtree(a.cleanup, ignore_errors=True)
        return 1
    swapped = False
    try:
        if os.path.exists(backup):
            _retry(lambda: shutil.rmtree(backup))
        _retry(lambda: os.replace(target, backup))
        swapped = True
        shutil.copytree(a.source, target)
        for name in KEEP:
            old = os.path.join(backup, name)
            new = os.path.join(target, name)
            if os.path.isdir(old) and not os.path.exists(new):
                _retry(lambda: os.replace(old, new))
        if not a.no_install:
            _install_requirements(a, target)
        mf = os.path.join(a.root, "install.json")
        try:
            with open(mf, encoding="utf8") as fh:
                info = json.load(fh)
        except (OSError, ValueError):
            info = {}
        info["version"] = a.version
        with open(mf, "w", encoding="utf8") as fh:
            json.dump(info, fh, indent=2)
        shutil.rmtree(backup, ignore_errors=True)
        _log(a.root, f"updated to {a.version}")
        ok = 0
    except Exception as exc:                            # roll back
        _log(a.root, f"update failed: {exc!r}; restoring the previous version")
        if swapped:
            try:
                if os.path.exists(target):
                    for name in KEEP:                   # models may already be moved
                        moved = os.path.join(target, name)
                        if os.path.isdir(moved) and not os.path.exists(os.path.join(backup, name)):
                            os.replace(moved, os.path.join(backup, name))
                    _retry(lambda: shutil.rmtree(target))
                _retry(lambda: os.replace(backup, target))
            except Exception as exc2:
                _log(a.root, f"ROLLBACK FAILED: {exc2!r}; the previous version is in {backup}")
        ok = 1
    if a.cleanup:
        shutil.rmtree(a.cleanup, ignore_errors=True)    # the downloaded files
    if not a.no_relaunch:
        _relaunch(a, target)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--uv", default="")
    ap.add_argument("--cleanup", default="")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--no-install", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-relaunch", action="store_true", help=argparse.SUPPRESS)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

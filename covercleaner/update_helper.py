"""Install a downloaded update once Cover Cleaner has closed.

Started by updater.start_update() as a separate process (a copy of this
file outside the app folder, standard library only):

  1. wait for the program (--pid) to exit;
  2. move the app folder aside, copy the new one in, keep models/;
  3. install the new locked requirements (plus GPU support if it was set
     up); if the release pins a different Python, build a new environment
     with it next to the old one and swap it in;
  4. on any failure put the old app folder (and environment) back;
  5. start Cover Cleaner again.

It runs from the base Python, not from inside the environment, so the
environment can be replaced.

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


def _uv_env(root: str) -> dict:
    """Keep uv's Python inside the install folder (as the installers do)."""
    return dict(os.environ, UV_PYTHON_INSTALL_DIR=os.path.join(root, "python"),
                UV_MANAGED_PYTHON="1", UV_PYTHON_INSTALL_BIN="0",
                UV_PYTHON_INSTALL_REGISTRY="0", UV_NO_CACHE="1")


def _run(a, cmd) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, creationflags=_no_window(),
                       env=_uv_env(a.root))
    _log(a.root, f"$ {' '.join(cmd)}\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    if r.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed (exit {r.returncode})")


def _has_uv(a) -> bool:
    return bool(a.uv) and os.path.isfile(a.uv)


def _install_requirements(a, target: str, python: str) -> None:
    if _has_uv(a):
        base = [a.uv, "pip", "install", "--python", python]
    else:
        base = [python, "-m", "pip", "install"]
    lock = os.path.join(target, "requirements.lock")
    if os.path.isfile(lock):
        _run(a, base + ["--require-hashes", "-r", lock])
    else:                                               # older release
        _run(a, base + ["-r", os.path.join(target, "requirements.txt")])
    if a.gpu or a.extra:
        gpu_lock = os.path.join(target, "requirements-gpu.lock")
        if a.gpu and os.path.isfile(gpu_lock):
            _run(a, base + ["--require-hashes", "-r", gpu_lock])
        elif a.extra:
            _run(a, base + list(a.extra))


def _python_version(python: str) -> str:
    try:
        r = subprocess.run([python, "-c", "import platform; print(platform.python_version())"],
                           capture_output=True, text=True, timeout=60,
                           creationflags=_no_window())
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _wanted_python(target: str) -> str:
    try:
        with open(os.path.join(target, ".python-version"), encoding="utf8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _prune_pythons(root: str, keep: str) -> None:
    """Remove private Pythons other than `keep` (and the one running this)."""
    pdir = os.path.join(root, "python")
    if not os.path.isdir(pdir):
        return
    running = os.path.realpath(sys.executable)
    for name in os.listdir(pdir):
        path = os.path.join(pdir, name)
        if (name.startswith("cpython-") and not name.startswith(f"cpython-{keep}-")
                and not running.startswith(os.path.realpath(path) + os.sep)):
            shutil.rmtree(path, ignore_errors=True)


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


def _read_manifest(root: str) -> dict:
    """install.json; utf-8-sig because Windows PowerShell 5.1 writes a BOM."""
    try:
        with open(os.path.join(root, "install.json"), encoding="utf-8-sig") as fh:
            info = json.load(fh)
        return info if isinstance(info, dict) else {}
    except (OSError, ValueError):
        return {}


def _console_python(py: str) -> str:
    """python.exe rather than pythonw.exe (which has no output to read)."""
    d, name = os.path.split(py)
    if name.lower() == "pythonw.exe" and os.path.isfile(os.path.join(d, "python.exe")):
        return os.path.join(d, "python.exe")
    return py


def _fill_in(a) -> None:
    """Work out what the program that started this helper may not have
    passed: older versions read install.json wrongly (BOM) and so sent no
    uv and no GPU flag, and passed pythonw.exe."""
    info = _read_manifest(a.root)
    a.python = _console_python(a.python)
    if not _has_uv(a):
        names = [info.get("uv") or "", os.path.join("bin", "uv.exe"), os.path.join("bin", "uv")]
        for n in names:
            path = os.path.join(a.root, n) if n else ""
            if path and os.path.isfile(path):
                a.uv = path
                break
    if info.get("gpu"):
        a.gpu = True


def run(a) -> int:
    _fill_in(a)
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
    venv = os.path.dirname(os.path.dirname(os.path.abspath(a.python)))   # venv/Scripts/python
    venv_new, venv_old = venv + ".new", venv + ".old"
    swapped = venv_swapped = installing_in_place = False
    want = ""
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
        want = _wanted_python(target)
        have = _python_version(a.python)
        if not a.no_install:
            if want and have and want != have and _has_uv(a):
                _log(a.root, f"new Python {want} (was {have}): building a new environment")
                shutil.rmtree(venv_new, ignore_errors=True)
                _run(a, [a.uv, "venv", "--python", want, venv_new])
                new_py = os.path.join(venv_new, os.path.relpath(os.path.abspath(a.python), venv))
                _install_requirements(a, target, new_py)
                if os.path.exists(venv_old):
                    _retry(lambda: shutil.rmtree(venv_old))
                _retry(lambda: os.replace(venv, venv_old))
                venv_swapped = True
                _retry(lambda: os.replace(venv_new, venv))
            else:
                if want and have and want != have:
                    _log(a.root, f"release wants Python {want} but uv is missing; keeping {have}")
                    want = have
                installing_in_place = True
                _install_requirements(a, target, a.python)
        mf = os.path.join(a.root, "install.json")
        info = _read_manifest(a.root)
        info["version"] = a.version
        if want and not a.no_install:
            info["python_version"] = want
        with open(mf, "w", encoding="utf8") as fh:
            json.dump(info, fh, indent=2)
        shutil.rmtree(backup, ignore_errors=True)
        if venv_swapped:
            shutil.rmtree(venv_old, ignore_errors=True)
            _prune_pythons(a.root, want)
        _log(a.root, f"updated to {a.version}")
        ok = 0
    except Exception as exc:                            # roll back
        _log(a.root, f"update failed: {exc!r}; restoring the previous version")
        shutil.rmtree(venv_new, ignore_errors=True)
        if venv_swapped:
            try:
                if os.path.exists(venv):
                    _retry(lambda: shutil.rmtree(venv))
                _retry(lambda: os.replace(venv_old, venv))
            except Exception as exc2:
                _log(a.root, f"ROLLBACK FAILED: {exc2!r}; the previous environment is in {venv_old}")
        if swapped:
            try:
                if os.path.exists(target):
                    for name in KEEP:                   # models may already be moved
                        moved = os.path.join(target, name)
                        if os.path.isdir(moved) and not os.path.exists(os.path.join(backup, name)):
                            os.replace(moved, os.path.join(backup, name))
                    _retry(lambda: shutil.rmtree(target))
                _retry(lambda: os.replace(backup, target))
                if installing_in_place:             # put the old exact packages back
                    try:
                        _install_requirements(a, target, a.python)
                    except Exception as exc3:
                        _log(a.root, f"could not restore the previous packages: {exc3!r}")
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
    ap.add_argument("--gpu", action="store_true", help="GPU support was set up")
    ap.add_argument("--no-install", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-relaunch", action="store_true", help=argparse.SUPPRESS)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

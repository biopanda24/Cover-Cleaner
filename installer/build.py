"""Build the Cover Cleaner installers.

    python installer/build.py                       all targets
    python installer/build.py --targets linux,macos
    python installer/build.py --out dist

Produces, in dist/:
  CoverCleaner-<v>-windows-setup.exe   Windows setup (needs Inno Setup 6's ISCC;
                                       without it: CoverCleaner-<v>-windows.zip
                                       with an "Install Cover Cleaner.cmd")
  CoverCleaner-<v>-linux.run           Linux self-extracting installer (bash)
  CoverCleaner-<v>-macos.zip           macOS installer (.command inside, keeps +x)
  CoverCleaner-<v>-app.zip (+ .sha256) the program alone - used by in-app updates

The Linux and macOS installers and the app archive are plain archives and
scripts, so they can be built on any system; the Windows .exe needs Inno
Setup (free; preinstalled on GitHub's Windows runners - see
.github/workflows/installers.yml, which builds everything on a tag).

Every installer sets up Python and the packages with uv, and offers -
ticked by default - the ML models and NVIDIA GPU support.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "installer")
# What goes into an installed copy (the program, not tests or dev tools).
INCLUDE = ("covercleaner/", "tools/__init__.py", "tools/lama_float.py",
           "requirements.txt", "requirements.lock", "requirements-gpu.lock",
           ".python-version", "README.md", "CHANGELOG.md", "LICENSE")
# uv, which installs Python and the packages, is pinned too (a release that
# has been out for a while and knows the Python version in .python-version).
UV_VERSION = "0.13.6"          # also in .github/workflows/installers.yml


def version() -> str:
    src = open(os.path.join(ROOT, "covercleaner", "__init__.py"), encoding="utf8").read()
    return re.search(r'__version__\s*=\s*"([^"]+)"', src).group(1)


def app_files() -> list[str]:
    """Tracked files of the program (git ls-files), else a walk."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.split("\n")
    except (OSError, subprocess.CalledProcessError):
        out = []
        for dp, dn, fn in os.walk(ROOT):
            dn[:] = [d for d in dn if d not in (".git", ".venv", "__pycache__", "dist")]
            out += [os.path.relpath(os.path.join(dp, f), ROOT).replace(os.sep, "/") for f in fn]
    files = [f for f in out if f and any(f == i or f.startswith(i) for i in INCLUDE)
             and "__pycache__" not in f and os.path.isfile(os.path.join(ROOT, f))]
    if "covercleaner/__init__.py" not in files:
        raise SystemExit("covercleaner/ not found - run from the repository")
    for need in ("requirements.lock", "requirements-gpu.lock", ".python-version"):
        if need not in files:
            raise SystemExit(f"{need} is missing - run installer/lock.py")
    return sorted(files)


def python_version() -> str:
    return open(os.path.join(ROOT, ".python-version"), encoding="utf8").read().strip()


REPO_URL = "https://github.com/biopanda24/Cover-Cleaner"


def release_notes(v: str) -> str:
    """The CHANGELOG.md section for version `v` (shown in the update dialog
    and on the GitHub release page)."""
    text = open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf8").read().replace("\r\n", "\n")
    m = re.search(rf"^## {re.escape(v)}\b[^\n]*\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    if not m or not m.group(1).strip():
        raise SystemExit(f"CHANGELOG.md has no section for {v}: add one before building")
    return (m.group(1).strip() + f"\n\n[All changes]({REPO_URL}/blob/main/CHANGELOG.md)\n")


def _template(name: str, v: str, **extra) -> str:
    text = open(os.path.join(HERE, name), encoding="utf8").read()
    for k, val in dict(VERSION=v, PYTHON=python_version(), UV=UV_VERSION, **extra).items():
        text = text.replace(f"@{k}@", val)
    left = re.findall(r"@[A-Z_]+@", text)
    if left:
        raise SystemExit(f"{name}: unfilled placeholders {sorted(set(left))}")
    return text


def _tar_app(files: list[str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in files:
            info = tar.gettarinfo(os.path.join(ROOT, f), arcname="app/" + f)
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with open(os.path.join(ROOT, f), "rb") as fh:
                tar.addfile(info, fh)
    return buf.getvalue()


def _sha256(path: str) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    with open(path + ".sha256", "w", encoding="utf8", newline="\n") as fh:
        fh.write(f"{h.hexdigest()}  {os.path.basename(path)}\n")


def build_app_zip(files, v, out) -> str:
    path = os.path.join(out, f"CoverCleaner-{v}-app.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(ROOT, f), "app/" + f)
    _sha256(path)
    return path


def _unix_installer(files, v, os_kind, os_name) -> bytes:
    script = _template("install.sh", v, OS=os_kind, OS_NAME=os_name)
    script = script.replace("\r\n", "\n")
    if not script.endswith("\n"):
        script += "\n"
    return script.encode("utf8") + _tar_app(files)


def build_linux(files, v, out) -> str:
    path = os.path.join(out, f"CoverCleaner-{v}-linux.run")
    with open(path, "wb") as fh:
        fh.write(_unix_installer(files, v, "linux", "Linux"))
    try:
        os.chmod(path, 0o755)
    except OSError:
        pass
    return path


def build_macos(files, v, out) -> str:
    """A .command (double-clickable in Finder) inside a zip, so the
    executable bit survives the download."""
    data = _unix_installer(files, v, "macos", "macOS")
    path = os.path.join(out, f"CoverCleaner-{v}-macos.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo(f"Install Cover Cleaner {v}.command", time.localtime()[:6])
        info.create_system = 3                                  # unix: keep permissions
        info.external_attr = (stat.S_IFREG | 0o755) << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, data)
    return path


def _find_iscc() -> str | None:
    for c in (shutil.which("ISCC"), shutil.which("iscc"),
              r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
              r"C:\Program Files\Inno Setup 6\ISCC.exe",
              os.path.expandvars(r"%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe")):
        if c and os.path.isfile(c):
            return c
    return None


def build_windows(files, v, out, force_zip=False) -> str:
    stage = os.path.join(out, "stage-windows")
    shutil.rmtree(stage, ignore_errors=True)
    for f in files:
        dst = os.path.join(stage, "app", *f.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, f), dst)
    with open(os.path.join(stage, "install.ps1"), "w", encoding="utf-8-sig", newline="\r\n") as fh:
        fh.write(_template("install.ps1", v))
    iscc = None if force_zip else _find_iscc()
    try:
        if iscc:
            iss = os.path.join(stage, "CoverCleaner.iss")
            with open(iss, "w", encoding="utf-8-sig", newline="\r\n") as fh:
                fh.write(_template("CoverCleaner.iss", v, PAYLOAD=stage,
                                   OUTDIR=os.path.abspath(out)))
            subprocess.run([iscc, "/Q", iss], check=True)
            return os.path.join(out, f"CoverCleaner-{v}-windows-setup.exe")
        # No Inno Setup: a zip with a double-clickable installer
        cmd = ('@echo off\r\n'
               'echo Installing Cover Cleaner %s ...\r\n'
               'powershell -NoProfile -ExecutionPolicy Bypass -File "%%~dp0install.ps1" '
               '-AppSource "%%~dp0." -Models -Gpu -Shortcuts\r\n'
               'pause\r\n' % v)
        with open(os.path.join(stage, "Install Cover Cleaner.cmd"), "w", newline="") as fh:
            fh.write(cmd)
        path = os.path.join(out, f"CoverCleaner-{v}-windows.zip")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for dp, dn, fn in os.walk(stage):
                for f in fn:
                    full = os.path.join(dp, f)
                    z.write(full, os.path.join(f"CoverCleaner-{v}",
                                               os.path.relpath(full, stage)))
        return path
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default="windows,linux,macos",
                    help="comma list of windows, linux, macos (the app archive is always built)")
    ap.add_argument("--out", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--windows-zip", action="store_true",
                    help="build the Windows zip installer even if Inno Setup is available")
    a = ap.parse_args(argv)
    v = version()
    a.out = os.path.abspath(a.out)                 # Inno Setup resolves relative paths from the .iss
    os.makedirs(a.out, exist_ok=True)
    files = app_files()
    print(f"Cover Cleaner {v}: {len(files)} files")
    with open(os.path.join(a.out, "release-notes.md"), "w", encoding="utf8", newline="\n") as fh:
        fh.write(release_notes(v))
    made = [build_app_zip(files, v, a.out)]
    targets = {t.strip() for t in a.targets.split(",") if t.strip()}
    if "linux" in targets:
        made.append(build_linux(files, v, a.out))
    if "macos" in targets:
        made.append(build_macos(files, v, a.out))
    if "windows" in targets:
        w = build_windows(files, v, a.out, a.windows_zip)
        made.append(w)
        if w.endswith(".zip") and not a.windows_zip:
            print("  (Inno Setup not found: built the zip installer instead of a setup .exe)")
    for p in made:
        print(f"  {os.path.relpath(p, ROOT)}  {os.path.getsize(p) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

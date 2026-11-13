"""New-version check and in-program updates from GitHub releases.

An installed copy looks like this (see installer/):

    <root>/app/            the program (this package, tools/, requirements.txt)
    <root>/app/models/     downloaded model files (kept across updates)
    <root>/venv/           its Python environment
    <root>/bin/uv          the package installer used by the installers
    <root>/install.json    what was installed (version, models, gpu)

An update downloads the release's app archive, checks its SHA-256 when the
release publishes one, unpacks it next to the install, and starts
update_helper.py, which waits for the program to close, swaps the app
folder (keeping models/), installs the new requirements, rolls back if
anything fails, and starts the program again.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import zipfile
from dataclasses import dataclass, field

from . import __version__, net

REPO = "biopanda24/Cover-Cleaner"
REPO_URL = f"https://github.com/{REPO}"
API = f"https://api.github.com/repos/{REPO}"


class UpdateError(Exception):
    pass


@dataclass
class Release:
    version: str
    tag: str
    url: str                        # release page
    notes: str = ""
    archive_url: str = ""           # app archive to install
    sha256_url: str = ""            # its checksum file, when published
    assets: dict = field(default_factory=dict)


def parse_version(v: str) -> tuple:
    """'v1.2.3' -> (1, 2, 3); pre-release suffixes sort before the release."""
    m = re.match(r"^\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(.*)$", v or "")
    if not m:
        return (0, 0, 0, 0)
    major, minor, patch, rest = m.groups()
    return (int(major), int(minor or 0), int(patch or 0), 0 if rest.strip() else 1)


def is_newer(candidate: str, current: str = __version__) -> bool:
    return parse_version(candidate) > parse_version(current)


def _get_json(url: str, timeout: float):
    with net.urlopen(url, timeout, {"Accept": "application/vnd.github+json"}) as r:
        return json.loads(r.read().decode("utf8"))


def latest_release(timeout: float = 10.0) -> Release | None:
    """The newest published release, or None if there is none yet."""
    try:
        data = _get_json(f"{API}/releases/latest", timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None                               # no releases (or no public repo)
        raise UpdateError(f"GitHub answered {exc.code}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise UpdateError(f"could not reach GitHub ({exc})") from exc
    tag = data.get("tag_name") or ""
    version = tag.lstrip("vV")
    assets = {a.get("name", ""): a.get("browser_download_url", "") for a in data.get("assets", [])}
    rel = Release(version=version, tag=tag, url=data.get("html_url") or REPO_URL + "/releases",
                  notes=(data.get("body") or "").strip(), assets=assets)
    name = f"CoverCleaner-{version}-app.zip"
    if name in assets:
        rel.archive_url = assets[name]
        rel.sha256_url = assets.get(name + ".sha256", "")
    else:
        rel.archive_url = data.get("zipball_url") or ""   # the release's source
    return rel


def check(timeout: float = 10.0) -> Release | None:
    """The latest release if it is newer than this version, else None."""
    rel = latest_release(timeout)
    return rel if rel and is_newer(rel.version) else None


# ---- install layout ----------------------------------------------------------
def app_dir() -> str:
    """Folder holding the covercleaner package."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def install_root() -> str | None:
    root = os.path.dirname(app_dir())
    return root if os.path.isfile(os.path.join(root, "install.json")) else None


def manifest() -> dict:
    root = install_root()
    if not root:
        return {}
    try:
        with open(os.path.join(root, "install.json"), encoding="utf8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def can_update() -> tuple[bool, str]:
    """Whether this copy can update itself, and if not, why."""
    if os.path.isdir(os.path.join(app_dir(), ".git")):
        return False, ("This is a development copy (a git checkout); update it with "
                       "'git pull' instead.")
    if not install_root():
        return False, ("This copy was not set up by the Cover Cleaner installer, so it "
                       "cannot update itself. Download the new version from GitHub.")
    root = install_root()
    if not os.access(root, os.W_OK):
        return False, f"No permission to write to {root}."
    return True, ""


# ---- download and apply ------------------------------------------------------
def _download(url: str, dest: str, progress=None) -> None:
    try:
        with net.urlopen(url, 60) as r, open(dest, "wb") as fh:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    except (urllib.error.URLError, OSError) as exc:
        raise UpdateError(f"download failed ({exc})") from exc


def download(rel: Release, progress=None) -> str:
    """Download and unpack `rel`; returns the folder of the new app (staged
    next to the install, so swapping it in is a rename on the same disk)."""
    if not rel.archive_url:
        raise UpdateError("the release has no downloadable archive")
    root = install_root() or tempfile.gettempdir()
    stage = os.path.join(root, "update", rel.version)
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage, exist_ok=True)
    archive = os.path.join(stage, "app.zip")
    _download(rel.archive_url, archive, progress)
    if rel.sha256_url:
        sums = os.path.join(stage, "app.zip.sha256")
        _download(rel.sha256_url, sums)
        with open(sums, encoding="utf8") as fh:
            want = fh.read().split()[0].strip().lower()
        h = hashlib.sha256()
        with open(archive, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            raise UpdateError("the download is damaged (checksum mismatch); nothing was changed")
    out = os.path.join(stage, "unpacked")
    try:
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():                      # no paths outside the stage
                target = os.path.realpath(os.path.join(out, n))
                if not target.startswith(os.path.realpath(out) + os.sep) and target != os.path.realpath(out):
                    raise UpdateError("unsafe path in the update archive")
            z.extractall(out)
    except zipfile.BadZipFile as exc:
        raise UpdateError("the download is not a valid archive") from exc
    for dirpath, dirnames, filenames in os.walk(out):
        if "__init__.py" in filenames and os.path.basename(dirpath) == "covercleaner":
            new_app = os.path.dirname(dirpath)
            if os.path.isfile(os.path.join(new_app, "requirements.txt")):
                return new_app
    raise UpdateError("the download does not contain Cover Cleaner")


def _gui_python(py: str) -> str:
    if sys.platform == "win32":
        w = os.path.join(os.path.dirname(py), "pythonw.exe")
        if os.path.isfile(w):
            return w
    return py


def base_python() -> str:
    """The Python the environment was made from (the helper runs from it, so
    the environment itself can be replaced)."""
    cfg = os.path.join(sys.prefix, "pyvenv.cfg")
    try:
        with open(cfg, encoding="utf8") as fh:
            home = next((ln.split("=", 1)[1].strip() for ln in fh
                         if ln.split("=", 1)[0].strip().lower() == "home"), "")
    except OSError:
        home = ""
    names = ("python.exe",) if sys.platform == "win32" else ("python3", "python")
    for n in names:
        if home and os.path.isfile(os.path.join(home, n)):
            return os.path.join(home, n)
    return sys.executable


def _stage_of(new_app: str) -> str:
    """The update/<version> folder that download() unpacked `new_app` into."""
    d = os.path.abspath(new_app)
    while os.path.basename(os.path.dirname(d)) != "update" and os.path.dirname(d) != d:
        d = os.path.dirname(d)
    return d


def start_update(new_app: str, version: str) -> None:
    """Start the helper that installs `new_app` once this program has closed.
    The caller should quit right after."""
    root = install_root()
    if not root:
        raise UpdateError("not an installed copy")
    info = manifest()
    py = sys.executable
    helper_src = os.path.join(new_app, "covercleaner", "update_helper.py")
    if not os.path.isfile(helper_src):                  # older archive: use ours
        helper_src = os.path.join(app_dir(), "covercleaner", "update_helper.py")
    stage = _stage_of(new_app)
    helper = os.path.join(stage, "update_helper.py")
    shutil.copy2(helper_src, helper)                    # runs from outside the app folder
    uv = os.path.join(root, info.get("uv", "")) if info.get("uv") else ""
    args = [_gui_python(base_python()), helper, "--pid", str(os.getpid()), "--source", new_app,
            "--target", app_dir(), "--root", root, "--python", py, "--version", version,
            "--cleanup", stage]
    if uv and os.path.isfile(uv):
        args += ["--uv", uv]
    if info.get("gpu"):
        from .setup_models import GPU_PACKAGES
        args += ["--gpu", "--extra"] + GPU_PACKAGES
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(args, cwd=root, creationflags=flags, close_fds=True,
                     start_new_session=(sys.platform != "win32"))

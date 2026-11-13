"""Write requirements.lock and requirements-gpu.lock from the tested environment.

    .venv/Scripts/python installer/lock.py          (run with the dev venv)

Every package is pinned to the version installed in the environment that
ran the tests, with the SHA-256 of each of its files on PyPI, so every
install gets exactly what was tested and nothing that was tampered with.
Dependencies are worked out separately for each platform the installers
support, so a Windows-only dependency gets a Windows-only marker. The script
stops if a dependency another platform needs is not installed here, or if a
pinned version has no build for a supported platform and the pinned Python.

To upgrade a package: upgrade it in .venv, run the tests, run this script.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import re
import sys
import urllib.request

try:
    from packaging.markers import Marker
    from packaging.requirements import Requirement
except ImportError:                                     # pip's copy
    from pip._vendor.packaging.markers import Marker  # noqa: F401
    from pip._vendor.packaging.requirements import Requirement

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PY = open(os.path.join(ROOT, ".python-version"), encoding="utf8").read().strip()
PY_TAG = "cp" + "".join(PY.split(".")[:2])              # cp310


def _env(sys_platform, platform_system, machine):
    return {"python_version": ".".join(PY.split(".")[:2]), "python_full_version": PY,
            "sys_platform": sys_platform, "platform_system": platform_system,
            "platform_machine": machine, "os_name": "nt" if sys_platform == "win32" else "posix",
            "implementation_name": "cpython", "platform_python_implementation": "CPython",
            "implementation_version": PY, "platform_release": "", "platform_version": "",
            "extra": ""}


TARGETS = {
    "windows": (_env("win32", "Windows", "AMD64"), ("win_amd64",)),
    "linux": (_env("linux", "Linux", "x86_64"), ("manylinux", "x86_64")),
    "macos-arm64": (_env("darwin", "Darwin", "arm64"), ("macosx", "arm64|universal2")),
    "macos-x86_64": (_env("darwin", "Darwin", "x86_64"), ("macosx", "x86_64|universal2")),
}
GPU_TARGETS = ("windows", "linux")
PLATFORM_MARKER = {"windows": 'sys_platform == "win32"', "linux": 'sys_platform == "linux"',
                   "macos-arm64": 'sys_platform == "darwin"', "macos-x86_64": 'sys_platform == "darwin"'}


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def closure(tops: list[str], env: dict) -> dict[str, str]:
    """name -> installed version for everything `tops` needs under `env`."""
    out: dict[str, str] = {}
    todo = [Requirement(t) for t in tops]
    done: set = set()
    while todo:
        req = todo.pop()
        key = (norm(req.name), tuple(sorted(req.extras)))
        if key in done:
            continue
        done.add(key)
        try:
            dist = md.distribution(req.name)
        except md.PackageNotFoundError:
            raise SystemExit(f"{req.name} is needed (on some platform) but not installed in "
                             f"this environment - install the tested version and rerun")
        if not req.specifier.contains(dist.version, prereleases=True):
            raise SystemExit(f"installed {req.name} {dist.version} does not satisfy {req}")
        out[norm(req.name)] = dist.version
        for r in dist.requires or []:
            dep = Requirement(r)
            if dep.marker is None:
                todo.append(dep)
                continue
            extras = req.extras or {""}
            if any(dep.marker.evaluate(dict(env, extra=e)) for e in extras):
                todo.append(dep)
    return out


def pypi_files(name: str, version: str) -> list[dict]:
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "lock.py"}),
                                timeout=30) as r:
        return json.load(r)["urls"]


def _fits(filename: str, platform_parts) -> bool:
    if not filename.endswith(".whl"):
        return False
    tags = filename[:-4].split("-")
    pytag, abitag, plat = tags[-3], tags[-2], tags[-1]
    py_ok = (PY_TAG in pytag.split(".") or "py3" in pytag.split(".")
             or (abitag == "abi3" and any(int(t[3:] or 0) <= int(PY_TAG[3:])
                                          for t in pytag.split(".") if t.startswith("cp3"))))
    if not py_ok:
        return False
    if plat == "any":
        return True
    os_part, arch_part = platform_parts if len(platform_parts) == 2 else (platform_parts[0], "")
    return os_part in plat and (not arch_part or any(a in plat for a in arch_part.split("|")))


def write_lock(path: str, tops: list[str], targets, title: str) -> None:
    per_target = {t: closure(tops, TARGETS[t][0]) for t in targets}
    names = sorted({n for c in per_target.values() for n in c})
    lines = [f"# {title}",
             f"# Pinned by installer/lock.py from the tested development environment, for Python {PY}.",
             "# Do not edit by hand: upgrade in .venv, run the tests, run installer/lock.py.",
             ""]
    problems = []
    for name in names:
        where = [t for t in targets if name in per_target[t]]
        versions = {per_target[t][name] for t in where}
        version = versions.pop()
        files = pypi_files(name, version)
        for t in where:
            if not any(_fits(f["filename"], TARGETS[t][1]) for f in files):
                problems.append(f"{name} {version} has no wheel for {t} / Python {PY}")
        marker = ""
        if len(where) < len(targets):
            ms = sorted({PLATFORM_MARKER[t] for t in where})
            marker = " ; " + " or ".join(ms)
        hashes = sorted({f["digests"]["sha256"] for f in files})
        lines.append(f"{name}=={version}{marker} \\")
        lines += [f"    --hash=sha256:{h} \\" for h in hashes[:-1]]
        lines.append(f"    --hash=sha256:{hashes[-1]}")
        print(f"  {name}=={version}{marker}  ({len(hashes)} files)")
    if problems:
        raise SystemExit("\n".join(["cannot lock:"] + problems))
    with open(path, "w", encoding="utf8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    if not sys.version.startswith(".".join(PY.split(".")[:2]) + "."):
        print(f"warning: running on Python {sys.version.split()[0]}, the installers use {PY}")
    tops = [ln.split("#")[0].strip() for ln in
            open(os.path.join(ROOT, "requirements.txt"), encoding="utf8")]
    tops = [t for t in tops if t]
    from covercleaner.setup_models import GPU_PACKAGES
    print("requirements.lock")
    write_lock(os.path.join(ROOT, "requirements.lock"), tops, list(TARGETS),
               "Cover Cleaner: exact packages for every platform")
    print("requirements-gpu.lock")
    write_lock(os.path.join(ROOT, "requirements-gpu.lock"), list(GPU_PACKAGES), list(GPU_TARGETS),
               "Cover Cleaner: NVIDIA GPU support (Windows and Linux x86-64)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

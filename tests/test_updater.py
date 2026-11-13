"""Updater and update helper, offline (file:// archives, a fake install)."""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import zipfile

import pytest

from covercleaner import update_helper, updater


def test_versions():
    assert updater.parse_version("v1.2.3") == (1, 2, 3, 1)
    assert updater.parse_version("0.4") == (0, 4, 0, 1)
    assert updater.is_newer("0.10.0", "0.9.9")
    assert updater.is_newer("v1.0.0", "1.0.0-rc1")
    assert not updater.is_newer("1.0.0-rc1", "1.0.0")
    assert not updater.is_newer("0.3.1", "0.3.1")
    assert not updater.is_newer("garbage", "0.1.0")


def _app_zip(path, extra=None):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("app/covercleaner/__init__.py", '__version__ = "9.9.9"\n')
        z.writestr("app/requirements.txt", "numpy\n")
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return path


def _release(archive, sha=None, tmp=None):
    rel = updater.Release("9.9.9", "v9.9.9", "https://example.invalid",
                          archive_url=pathlib.Path(archive).as_uri())
    if sha is not None:
        p = tmp / "sums.sha256"
        p.write_text(f"{sha}  app.zip\n")
        rel.sha256_url = p.as_uri()
    return rel


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "install"
    r.mkdir()
    monkeypatch.setattr(updater, "install_root", lambda: str(r))
    return r


def test_download_ok_and_checksum(tmp_path, root):
    z = _app_zip(tmp_path / "a.zip")
    sha = hashlib.sha256(z.read_bytes()).hexdigest()
    new_app = updater.download(_release(z, sha, tmp_path))
    assert os.path.isfile(os.path.join(new_app, "covercleaner", "__init__.py"))
    assert updater._stage_of(new_app) == str(root / "update" / "9.9.9")
    with pytest.raises(updater.UpdateError, match="checksum"):
        updater.download(_release(z, "0" * 64, tmp_path))


def test_download_rejects_unsafe_paths(tmp_path, root):
    z = _app_zip(tmp_path / "evil.zip", {"../../escape.txt": "x"})
    with pytest.raises(updater.UpdateError, match="unsafe"):
        updater.download(_release(z))
    assert not (tmp_path / "escape.txt").exists()


def test_download_needs_the_program(tmp_path, root):
    p = tmp_path / "other.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("readme.txt", "hi")
    with pytest.raises(updater.UpdateError, match="does not contain"):
        updater.download(_release(p))


def _fake_install(tmp_path):
    root = tmp_path / "inst"
    (root / "app" / "covercleaner").mkdir(parents=True)
    (root / "app" / "covercleaner" / "__init__.py").write_text('__version__ = "0.1.0"\n')
    (root / "app" / "models").mkdir()
    (root / "app" / "models" / "m.onnx").write_text("model")
    (root / "install.json").write_text(json.dumps({"version": "0.1.0", "gpu": False}))
    new = tmp_path / "stage" / "unpacked" / "app"
    (new / "covercleaner").mkdir(parents=True)
    (new / "covercleaner" / "__init__.py").write_text('__version__ = "0.2.0"\n')
    (new / "requirements.txt").write_text("")
    return root, new


def _finished_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _args(root, source, **kw):
    a = ["--pid", str(_finished_pid()), "--source", str(source), "--target", str(root / "app"),
         "--root", str(root), "--python", sys.executable, "--version", "0.2.0",
         "--no-install", "--no-relaunch"]
    for k, v in kw.items():
        a += [f"--{k}", str(v)]
    return a


def test_helper_swaps_and_keeps_models(tmp_path):
    root, new = _fake_install(tmp_path)
    assert update_helper.main(_args(root, new, cleanup=tmp_path / "stage")) == 0
    assert '"0.2.0"' in (root / "app" / "covercleaner" / "__init__.py").read_text()
    assert (root / "app" / "models" / "m.onnx").read_text() == "model"
    assert json.loads((root / "install.json").read_text())["version"] == "0.2.0"
    assert not (root / "app.old").exists()
    assert not (tmp_path / "stage").exists()


def test_helper_rolls_back(tmp_path):
    root, _new = _fake_install(tmp_path)
    missing = tmp_path / "nothing-here"
    assert update_helper.main(_args(root, missing)) == 1
    assert '"0.1.0"' in (root / "app" / "covercleaner" / "__init__.py").read_text()
    assert (root / "app" / "models" / "m.onnx").read_text() == "model"
    assert json.loads((root / "install.json").read_text())["version"] == "0.1.0"
    assert "update failed" in (root / "update.log").read_text()


def test_dev_checkout_cannot_self_update():
    ok, why = updater.can_update()
    assert not ok and why


# ---- Python / environment handling (external commands simulated) ----------
def _layout_with_venv(tmp_path, new_python="3.10.22", lock=True):
    root, new = _fake_install(tmp_path)
    (root / "app" / "requirements.txt").write_text("")
    py = root / "venv" / "Scripts" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_text("")
    (root / "venv" / "OLD").write_text("")
    uv = root / "bin" / "uv.exe"
    uv.parent.mkdir()
    uv.write_text("")
    (new / ".python-version").write_text(new_python + "\n")
    if lock:
        (new / "requirements.lock").write_text("")
    return root, new, py, uv


def _simulate(monkeypatch, have, fail_on=None):
    cmds = []

    def run(a, cmd):
        cmds.append(cmd)
        if cmd[1] == "venv":
            d = pathlib.Path(cmd[-1])
            (d / "Scripts").mkdir(parents=True)
            (d / "Scripts" / "python.exe").write_text("")
            (d / "NEW").write_text("")
        if fail_on and fail_on(cmd):
            raise RuntimeError("simulated failure")

    monkeypatch.setattr(update_helper, "_run", run)
    monkeypatch.setattr(update_helper, "_python_version", lambda py: have)
    return cmds


def _helper_args(root, new, py, uv):
    return ["--pid", str(_finished_pid()), "--source", str(new), "--target", str(root / "app"),
            "--root", str(root), "--python", str(py), "--version", "0.2.0", "--uv", str(uv),
            "--no-relaunch"]


def test_helper_new_python_builds_and_swaps_env(tmp_path, monkeypatch):
    root, new, py, uv = _layout_with_venv(tmp_path)
    cmds = _simulate(monkeypatch, have="3.10.6")
    assert update_helper.main(_helper_args(root, new, py, uv)) == 0
    assert (root / "venv" / "NEW").exists() and not (root / "venv.old").exists()
    assert json.loads((root / "install.json").read_text())["python_version"] == "3.10.22"
    assert cmds[0][1:4] == ["venv", "--python", "3.10.22"]
    assert "--require-hashes" in cmds[1] and str(root / "venv.new") in cmds[1][4]


def test_helper_failed_new_env_keeps_old_one(tmp_path, monkeypatch):
    root, new, py, uv = _layout_with_venv(tmp_path)
    _simulate(monkeypatch, have="3.10.6", fail_on=lambda c: c[1] == "pip")
    assert update_helper.main(_helper_args(root, new, py, uv)) == 1
    assert (root / "venv" / "OLD").exists() and not (root / "venv.new").exists()
    assert '"0.1.0"' in (root / "app" / "covercleaner" / "__init__.py").read_text()
    assert "python_version" not in json.loads((root / "install.json").read_text())


def test_helper_same_python_failure_restores_old_packages(tmp_path, monkeypatch):
    root, new, py, uv = _layout_with_venv(tmp_path, new_python="3.10.6")
    fails = iter([True])
    cmds = _simulate(monkeypatch, have="3.10.6", fail_on=lambda c: next(fails, False))
    assert update_helper.main(_helper_args(root, new, py, uv)) == 1
    assert not any(c[1] == "venv" for c in cmds)
    assert cmds[0][-1].endswith("requirements.lock")         # the new release's lock failed
    assert cmds[1][-1].endswith("requirements.txt")          # the old app's packages put back
    assert '"0.1.0"' in (root / "app" / "covercleaner" / "__init__.py").read_text()


def test_prune_pythons(tmp_path):
    for name in ("cpython-3.10.6-windows-x86_64-none", "cpython-3.10.22-windows-x86_64-none"):
        (tmp_path / "python" / name).mkdir(parents=True)
    update_helper._prune_pythons(str(tmp_path), "3.10.22")
    assert os.listdir(tmp_path / "python") == ["cpython-3.10.22-windows-x86_64-none"]


def test_lock_covers_requirements():
    root = pathlib.Path(__file__).resolve().parents[1]
    lock = (root / "requirements.lock").read_text()
    pinned = {ln.split("==")[0] for ln in lock.splitlines() if "==" in ln}
    for ln in (root / "requirements.txt").read_text().splitlines():
        name = ln.split("#")[0].strip()
        if name:
            base = name.split(">")[0].split("<")[0].split("=")[0].strip().lower().replace("_", "-")
            assert base in pinned, f"{base} is not in requirements.lock - run installer/lock.py"
    assert (root / ".python-version").read_text().strip() in lock


def test_release_notes_from_changelog():
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "installer"))
    try:
        import build
    finally:
        sys.path.pop(0)
    from covercleaner import __version__
    notes = build.release_notes(__version__)
    assert notes.strip() and "\n## " not in notes and "CHANGELOG.md" in notes
    with pytest.raises(SystemExit):
        build.release_notes("999.0.0")


def test_helper_works_out_uv_gpu_and_python_from_a_bom_manifest(tmp_path, monkeypatch):
    """What 0.5.0-0.5.2 send: no --uv, no --gpu, pythonw.exe; install.json with a BOM."""
    root, new, py, uv = _layout_with_venv(tmp_path, new_python="3.10.22")
    (py.parent / "pythonw.exe").write_text("")
    (new / "requirements-gpu.lock").write_text("")
    manifest = {"version": "0.1.0", "gpu": True, "uv": os.path.join("bin", "uv.exe" if os.name == "nt" else "uv"),
                "python_version": "3.10.22"}
    (root / "install.json").write_bytes(b"\xef\xbb\xbf" + json.dumps(manifest).encode())
    if os.name != "nt":
        uv.rename(uv.with_name("uv"))
        uv = uv.with_name("uv")
    cmds = _simulate(monkeypatch, have="3.10.22")
    args = ["--pid", str(_finished_pid()), "--source", str(new), "--target", str(root / "app"),
            "--root", str(root), "--python", str(py.parent / "pythonw.exe"), "--version", "0.2.0",
            "--no-relaunch"]
    assert update_helper.main(args) == 0
    assert [c[0] for c in cmds] == [str(uv), str(uv)]                 # uv, not pip
    assert cmds[0][4] == str(py)                                      # python.exe
    assert cmds[1][-1].endswith("requirements-gpu.lock")              # GPU kept
    info = json.loads((root / "install.json").read_text(encoding="utf-8"))
    assert info["version"] == "0.2.0" and info["gpu"] and info["uv"]  # nothing lost


def test_manifest_reads_bom(tmp_path, monkeypatch):
    (tmp_path / "install.json").write_bytes(b"\xef\xbb\xbf" + b'{"gpu": true}')
    monkeypatch.setattr(updater, "install_root", lambda: str(tmp_path))
    assert updater.manifest() == {"gpu": True}

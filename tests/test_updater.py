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

"""Download the optional ML models and set up NVIDIA GPU support.

    python -m covercleaner.setup_models --models      ML models (LaMa, disc text)
    python -m covercleaner.setup_models --gpu         GPU packages + GPU model copy
    python -m covercleaner.setup_models --models --gpu

Used by the installers; can also be run by hand at any time.  GPU support
is only set up on Windows/Linux x86-64 with an NVIDIA GPU (nvidia-smi
present); elsewhere ML runs on the CPU.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys

from . import __version__, net

HF = "https://huggingface.co/opencv"
MODELS = [
    ("inpainting_lama_2025jan.onnx", f"{HF}/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx",
     "LaMa inpainting (ML pipeline)"),
    ("text_detection_en_ppocrv3_2023may.onnx",
     f"{HF}/text_detection_ppocr/resolve/main/text_detection_en_ppocrv3_2023may.onnx",
     "text detection (turning discs upright)"),
    ("text_recognition_CRNN_CH_2021sep.onnx",
     f"{HF}/text_recognition_crnn/resolve/main/text_recognition_CRNN_CH_2021sep.onnx",
     "text recognition (turning discs upright)"),
]
# Tested together: cuDNN 9.2x fails to initialise with onnxruntime-gpu 1.23.
GPU_PACKAGES = ["onnxruntime-gpu[cuda]==1.23.2", "nvidia-cudnn-cu12==9.10.2.21", "onnx"]


def models_dir() -> str:
    from . import ml
    return os.path.dirname(ml.model_path())


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def gpu_platform() -> bool:
    """CUDA wheels exist for Windows and Linux on x86-64 only."""
    return sys.platform in ("win32", "linux") and platform.machine().lower() in ("amd64", "x86_64")


def has_nvidia_gpu() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=15,
                           creationflags=_no_window())
        return r.returncode == 0 and "GPU" in r.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def download(url: str, dest: str, label: str = "", progress=None) -> None:
    """Fetch `url` to `dest` (via a .part file, so a broken download never
    looks complete).  `progress(done_bytes, total_bytes)` is optional."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with net.urlopen(url, 60) as r, open(tmp, "wb") as fh:
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
    if total and os.path.getsize(tmp) != total:
        os.remove(tmp)
        raise IOError(f"incomplete download of {label or url}")
    os.replace(tmp, dest)


def _print_progress(label):
    last = [-1]

    def cb(done, total):
        if total:
            pct = int(done * 100 / total)
            if pct != last[0] and pct % 5 == 0:
                last[0] = pct
                print(f"\r  {label}: {pct:3d}% of {total / 1e6:.0f} MB", end="", flush=True)
    return cb


def setup_models(force: bool = False) -> bool:
    ok = True
    for name, url, what in MODELS:
        dest = os.path.join(models_dir(), name)
        if os.path.isfile(dest) and not force:
            print(f"  {name}: already installed")
            continue
        print(f"  Downloading {what}…")
        try:
            download(url, dest, name, _print_progress(name))
            print()
        except Exception as exc:                      # keep going with the others
            print(f"\n  FAILED: {name}: {exc}")
            ok = False
    return ok


def _installer_cmd() -> list[str]:
    """uv (as used by the installers) if available, else pip of this Python."""
    uv = os.environ.get("COVERCLEANER_UV") or shutil.which("uv")
    if uv and (os.path.isfile(uv) or shutil.which(uv)):
        return [uv, "pip", "install", "--python", sys.executable]
    try:
        import pip  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)
    return [sys.executable, "-m", "pip", "install"]


def setup_gpu() -> bool:
    if not gpu_platform():
        print("  GPU support: not available on this platform (ML runs on the CPU).")
        return True
    if not has_nvidia_gpu():
        print("  GPU support: no NVIDIA GPU found (nvidia-smi); skipped. ML runs on the CPU.")
        return True
    print("  Installing NVIDIA GPU support (onnxruntime-gpu, CUDA and cuDNN; about 1.5 GB)…")
    r = subprocess.run(_installer_cmd() + GPU_PACKAGES)
    if r.returncode != 0:
        print("  FAILED to install the GPU packages; ML will run on the CPU.")
        return False
    from . import ml
    if not ml.available():
        print("  GPU packages installed; the GPU model copy is made when the LaMa model is present.")
        return True
    print("  Preparing the GPU copy of the LaMa model…")
    if ml.prepare_gpu_model():
        print("  GPU support ready.")
        return True
    print("  Could not prepare the GPU model copy; it is retried on first use.")
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="covercleaner.setup_models", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", action="store_true", help="download the ML model files")
    ap.add_argument("--gpu", action="store_true", help="set up NVIDIA GPU support")
    ap.add_argument("--force", action="store_true", help="download models again")
    a = ap.parse_args(argv)
    if not (a.models or a.gpu):
        ap.print_help()
        return 0
    ok = True
    if a.models:
        print("ML models:")
        ok &= setup_models(a.force)
    if a.gpu:
        print("GPU:")
        ok &= setup_gpu()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

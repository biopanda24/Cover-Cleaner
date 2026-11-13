"""Command line / batch interface.

    python -m covercleaner auto  Scans/ -o output/   [-j 6]
    python -m covercleaner apply scan.png -p scan.json -o fixed.png
    python -m covercleaner gui [image]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import ml
from .auto import auto_params
from .imageio import SUPPORTED, load_image, read_dpi, save_image
from .params import Params
from .pipeline import Pipeline


def _collect(inputs: list[str]) -> list[str]:
    files: list[str] = []
    for item in inputs:
        if os.path.isdir(item):
            for name in sorted(os.listdir(item)):
                if name.lower().endswith(SUPPORTED):
                    files.append(os.path.join(item, name))
        else:
            hits = sorted(glob.glob(item))
            files.extend(hits if hits else [item])
    return files


def _out_path(src: str, out: str | None, many: bool, fmt: str) -> str:
    base = os.path.splitext(os.path.basename(src))[0]
    if out and not many and os.path.splitext(out)[1]:
        return out
    folder = out or os.path.join(os.path.dirname(os.path.abspath(src)), "restored")
    return os.path.join(folder, f"{base}.{fmt}")


def default_jobs(n_files: int, gpu: bool = False) -> int:
    """Scans processed at once: about one per four CPU threads (each scan
    still uses a few threads inside OpenCV), at most one per ~3 GB of
    memory, which a large scan with its cached stages can use, and - when
    ML inpainting runs on the GPU - one per ~1.9 GB of free GPU memory."""
    cpus = os.cpu_count() or 1
    jobs = max(1, cpus // 4)
    try:
        import ctypes

        class _Mem(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                        ("pt", ctypes.c_ulonglong), ("pa", ctypes.c_ulonglong),
                        ("vt", ctypes.c_ulonglong), ("va", ctypes.c_ulonglong),
                        ("ve", ctypes.c_ulonglong)]
        m = _Mem(); m.len = ctypes.sizeof(_Mem)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            jobs = min(jobs, max(1, int(m.avail / 3e9)))
    except Exception:
        pass
    if gpu:
        free = ml.gpu_free_mb()
        if free:
            jobs = min(jobs, max(1, int((free - 500) / 1900)))
    return max(1, min(jobs, n_files))


def _worker_init(threads: int) -> None:
    import cv2
    cv2.setNumThreads(threads)


def _auto_one(path: str, opts: dict) -> tuple[str, bool]:
    """Analyse and restore one scan. Returns (report text, ok)."""
    t0 = time.time()
    lines = [f"== {path}"]
    try:
        src = load_image(path)
        pipe = Pipeline(src)
        mode = opts["pipeline"] or ("ml" if ml.available() else "traditional")
        params, report = auto_params(src, pipeline=pipe, mode=mode, dpi=read_dpi(path))
        params.output.scale = opts["scale"]
        lines += ["   " + line for line in report]
        if opts["analyze_only"]:
            return "\n".join(lines), True
        result = pipe.run(params)
        dst = _out_path(path, opts["output"], opts["many"], opts["format"])
        save_image(dst, result, opts["bits"])
        if opts["save_params"]:
            with open(os.path.splitext(dst)[0] + ".json", "w", encoding="utf8") as fh:
                fh.write(params.to_json())
        lines.append(f"   -> {dst}  ({result.shape[1]} x {result.shape[0]}, "
                     f"{time.time() - t0:.1f}s)")
        return "\n".join(lines), True
    except Exception as exc:  # keep going with the rest of the batch
        lines.append(f"   FAILED: {exc}")
        return "\n".join(lines), False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="covercleaner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    a = sub.add_parser("auto", help="analyse and restore scans automatically")
    a.add_argument("inputs", nargs="+", help="image files, folders or wildcards")
    a.add_argument("-o", "--output", help="output folder (or file for a single input)")
    a.add_argument("--format", default="png", choices=["png", "tif", "jpg"])
    a.add_argument("--bits", type=int, default=8, choices=[8, 16])
    a.add_argument("--scale", type=float, default=1.0, help="resize factor for the result")
    a.add_argument("--save-params", action="store_true",
                   help="write the chosen settings next to each result (.json)")
    a.add_argument("--analyze-only", action="store_true", help="report only, write nothing")
    a.add_argument("--pipeline", choices=["traditional", "ml"], default=None,
                   help="ml uses neural inpainting for repairs (default: ml when the model "
                        "file is present, else traditional)")
    a.add_argument("-j", "--jobs", type=int, default=0,
                   help="scans processed at once (default: from CPU threads and memory)")

    b = sub.add_parser("apply", help="apply saved settings (.json) to a scan")
    b.add_argument("input")
    b.add_argument("-p", "--params", required=True)
    b.add_argument("-o", "--output", required=True)
    b.add_argument("--bits", type=int, default=8, choices=[8, 16])

    g = sub.add_parser("gui", help="open the interactive editor (default)")
    g.add_argument("image", nargs="?")

    args = ap.parse_args(argv)
    if args.cmd in (None, "gui"):
        from .gui.app import run_gui
        return run_gui(getattr(args, "image", None))

    if args.cmd == "apply":
        with open(args.params, encoding="utf8") as fh:
            params = Params.from_json(fh.read())
        result = Pipeline(load_image(args.input)).run(params)
        save_image(args.output, result, args.bits)
        print("saved", args.output)
        return 0

    files = _collect(args.inputs)
    if not files:
        print("No input images found.", file=sys.stderr)
        return 1
    opts = dict(pipeline=args.pipeline, scale=args.scale, analyze_only=args.analyze_only,
                output=args.output, many=len(files) > 1, format=args.format, bits=args.bits,
                save_params=args.save_params)
    mode = args.pipeline or ("ml" if ml.available() else "traditional")
    jobs = args.jobs if args.jobs > 0 else default_jobs(len(files), mode == "ml" and ml.gpu_usable())
    jobs = max(1, min(jobs, len(files)))
    t_start = time.time()
    failed = 0
    if jobs == 1:
        for path in files:
            text, ok = _auto_one(path, opts)
            print(text, flush=True)
            failed += not ok
    else:
        threads = max(1, (os.cpu_count() or 1) // jobs)
        print(f"Processing {len(files)} scans, {jobs} at a time.", flush=True)
        with ProcessPoolExecutor(jobs, initializer=_worker_init, initargs=(threads,)) as ex:
            futures = {ex.submit(_auto_one, path, opts): path for path in files}
            for done, fut in enumerate(as_completed(futures), 1):
                try:
                    text, ok = fut.result()
                except Exception as exc:          # a worker died (e.g. out of memory)
                    text, ok = f"== {futures[fut]}\n   FAILED: {exc}", False
                print(f"[{done}/{len(files)}] " + text, flush=True)
                failed += not ok
    print(f"Done: {len(files) - failed} of {len(files)} scans in {time.time() - t_start:.0f}s"
          + (f", {failed} failed." if failed else "."), flush=True)
    return 1 if failed else 0

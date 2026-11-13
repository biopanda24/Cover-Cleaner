"""Command line / batch interface.

    python -m covercleaner auto  Scans/ -o output/   [-j 6] [--flat]
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


IMAGE_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


def _auto_one(path: str, dst: str | None, opts: dict) -> tuple[str, bool]:
    """Analyse and restore one scan into `dst`. Returns (report text, ok)."""
    from . import settings as _settings
    t0 = time.time()
    st = _settings.AppSettings.from_dict(opts["settings"])
    ml.set_use_gpu(st.use_gpu)
    lines = [f"== {path}"]
    try:
        src = load_image(path)
        pipe = Pipeline(src)
        params, report = auto_params(src, pipeline=pipe, mode=st.mode(ml.available()),
                                     dpi=st.dpi_for(read_dpi(path)), flat_cleaner=st.flat_cleaner,
                                     rules=st.rules, scale=st.scale)
        lines += ["   " + line for line in report]
        if opts.get("analyze_only") or not dst:
            return "\n".join(lines), True
        result = pipe.run(params)
        save_image(dst, result, st.bits, st.jpeg_quality)
        if st.save_params:
            with open(os.path.splitext(dst)[0] + ".json", "w", encoding="utf8") as fh:
                fh.write(params.to_json())
        lines.append(f"   -> {dst}  ({result.shape[1]} x {result.shape[0]}, "
                     f"{time.time() - t0:.1f}s)")
        return "\n".join(lines), True
    except Exception as exc:  # keep going with the rest of the batch
        lines.append(f"   FAILED: {exc}")
        return "\n".join(lines), False


def plan_outputs(files: list[str], st, output: str | None = None,
                 output_is_dir: bool = False) -> dict:
    """Output file for each scan, all different and none over an original.

    `output` is a folder - or, for a single scan on the command line, a file
    name when it ends in an image extension and is not an existing folder.
    """
    if (output and len(files) == 1 and not output_is_dir and not os.path.isdir(output)
            and os.path.splitext(output)[1].lower() in IMAGE_EXTS):
        dst = os.path.abspath(output)
        if os.path.normcase(dst) == os.path.normcase(os.path.abspath(files[0])):
            raise ValueError(f"refusing to overwrite the scan itself: {dst}")
        return {files[0]: dst}
    taken: set = set()
    return {f: st.output_path(f, output, taken) for f in files}


def run_batch(files: list[str], st, output: str | None = None, analyze_only: bool = False,
              emit=print, progress=None, output_is_dir: bool = False) -> int:
    """Run Auto on `files` with options `st` (AppSettings); returns the number
    that failed.  `emit` receives each scan's report as it finishes,
    `progress(done, total)` the count."""
    from concurrent.futures.process import BrokenProcessPool
    ml.set_use_gpu(st.use_gpu)
    opts = dict(settings=st.to_dict(), analyze_only=analyze_only)
    dsts = {} if analyze_only else plan_outputs(files, st, output, output_is_dir)
    mode = st.mode(ml.available())
    use_gpu = mode == "ml" and ml.gpu_usable()
    jobs = st.jobs if st.jobs > 0 else default_jobs(len(files), use_gpu)
    jobs = max(1, min(jobs, len(files)))
    if use_gpu and jobs > 1:
        ml.prepare_gpu_model()            # convert once here, not in every worker
    t_start = time.time()
    failed = 0
    done = 0
    retry: list[str] = []

    def report(text, ok):
        nonlocal failed, done
        done += 1
        emit(f"[{done}/{len(files)}] " + text if len(files) > 1 else text)
        failed += not ok
        if progress:
            progress(done, len(files))

    if jobs == 1:
        for path in files:
            report(*_auto_one(path, dsts.get(path), opts))
    else:
        threads = max(1, (os.cpu_count() or 1) // jobs)
        emit(f"Processing {len(files)} scans, {jobs} at a time.")
        with ProcessPoolExecutor(jobs, initializer=_worker_init, initargs=(threads,)) as ex:
            futures = {ex.submit(_auto_one, path, dsts.get(path), opts): path for path in files}
            for fut in as_completed(futures):
                try:
                    report(*fut.result())
                except BrokenProcessPool:         # a worker died (e.g. out of memory)
                    retry.append(futures[fut])
                except Exception as exc:
                    report(f"== {futures[fut]}\n   FAILED: {exc}", False)
        if retry:
            emit(f"A worker stopped unexpectedly; retrying {len(retry)} scans one at a time.")
            for path in retry:
                report(*_auto_one(path, dsts.get(path), opts))
    emit(f"Done: {len(files) - failed} of {len(files)} scans in {time.time() - t_start:.0f}s"
         + (f", {failed} failed." if failed else "."))
    return failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="covercleaner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    from . import __version__
    ap.add_argument("--version", action="version", version=f"Cover Cleaner {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    a = sub.add_parser("auto", help="analyse and restore scans automatically",
                       description="Defaults come from Options (the editor's Settings > "
                                   "Options...); these flags override them for this run.")
    a.add_argument("inputs", nargs="+", help="image files, folders or wildcards")
    a.add_argument("-o", "--output", help="output folder (or file for a single input)")
    a.add_argument("--format", default=None, choices=["png", "tif", "jpg"])
    a.add_argument("--bits", type=int, default=None, choices=[8, 16])
    a.add_argument("--scale", type=float, default=None, help="resize factor for the result")
    a.add_argument("--suffix", default=None, help="added to each file name, e.g. _clean")
    a.add_argument("--save-params", action=argparse.BooleanOptionalAction, default=None,
                   help="write the chosen settings next to each result (.json)")
    a.add_argument("--analyze-only", action="store_true", help="report only, write nothing")
    a.add_argument("--pipeline", choices=["traditional", "ml"], default=None,
                   help="ml uses neural inpainting for repairs (default from Options: ml when "
                        "the model file is present)")
    a.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=None,
                   help="run ML inpainting on the GPU when available")
    a.add_argument("--flat", action=argparse.BooleanOptionalAction, default=None,
                   help="(Experimental) flat colour cleanup when flat sections are detected")
    a.add_argument("--dpi", type=float, default=None,
                   help="DPI to assume for scans that store none")
    a.add_argument("-j", "--jobs", type=int, default=None,
                   help="scans processed at once (0: from CPU threads and memory)")

    b = sub.add_parser("apply", help="apply saved settings (.json) to a scan")
    b.add_argument("input")
    b.add_argument("-p", "--params", required=True)
    b.add_argument("-o", "--output", required=True)
    b.add_argument("--bits", type=int, default=8, choices=[8, 16])

    g = sub.add_parser("gui", help="open the interactive editor (default)")
    g.add_argument("paths", nargs="*", help="scans and/or folders to open")

    args = ap.parse_args(argv)
    if args.cmd in (None, "gui"):
        from .gui.app import run_gui
        return run_gui(getattr(args, "paths", None))

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
    from . import settings as _settings
    st = _settings.load()
    for name, attr in (("format", "format"), ("bits", "bits"), ("scale", "scale"),
                       ("suffix", "suffix"), ("save_params", "save_params"),
                       ("pipeline", "pipeline"), ("gpu", "use_gpu"), ("flat", "flat_cleaner"),
                       ("dpi", "fallback_dpi"), ("jobs", "jobs")):
        val = getattr(args, name)
        if val is not None:
            setattr(st, attr, val)
    failed = run_batch(files, st, args.output, args.analyze_only,
                       emit=lambda t: print(t, flush=True))
    return 1 if failed else 0

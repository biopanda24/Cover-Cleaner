"""Command line / batch interface.

    python -m covercleaner auto  Scans/ -o output/
    python -m covercleaner apply scan.png -p scan.json -o fixed.png
    python -m covercleaner gui [image]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

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
    failed = 0
    for path in files:
        t0 = time.time()
        print(f"== {path}")
        try:
            src = load_image(path)
            pipe = Pipeline(src)
            mode = args.pipeline or ("ml" if ml.available() else "traditional")
            params, report = auto_params(src, pipeline=pipe, mode=mode, dpi=read_dpi(path))
            params.output.scale = args.scale
            for line in report:
                print("   " + line)
            if args.analyze_only:
                continue
            result = pipe.run(params)
            dst = _out_path(path, args.output, len(files) > 1, args.format)
            save_image(dst, result, args.bits)
            if args.save_params:
                with open(os.path.splitext(dst)[0] + ".json", "w", encoding="utf8") as fh:
                    fh.write(params.to_json())
            print(f"   -> {dst}  ({result.shape[1]} x {result.shape[0]}, {time.time() - t0:.1f}s)")
        except Exception as exc:  # keep going with the rest of the batch
            failed += 1
            print(f"   FAILED: {exc}", file=sys.stderr)
    return 1 if failed else 0

"""Run Auto on a folder in parallel and collect material for review.

    python tools/batch_eval.py Scans output/eval/run1 [--workers 4] [--pipeline traditional]
        [--only name1,name2]

For each scan it writes into the output folder:
    full/<name>.jpg     restored result, full resolution (JPEG q92)
    thumb/<name>.jpg    restored result, 600 px
    outline/<name>.jpg  source scan with the detected outline, 600 px
    meta/<name>.json    chosen settings, Auto report, timings
and summary.json with one line per scan.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def _thumb(img_u8, size=600):
    s = size / max(img_u8.shape[:2])
    return cv2.resize(img_u8, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def process(src: str, out: str, mode: str) -> dict:
    from covercleaner import geometry
    from covercleaner.auto import auto_params
    from covercleaner.imageio import load_image, read_dpi, to_uint8
    from covercleaner.pipeline import Pipeline

    name = os.path.splitext(os.path.basename(src))[0]
    t0 = time.time()
    try:
        img = load_image(src)
        pipe = Pipeline(img)
        params, report = auto_params(img, pipeline=pipe, mode=mode, dpi=read_dpi(src))
        t1 = time.time()
        result = pipe.run(params)
        t2 = time.time()
        res8 = to_uint8(result)
        cv2.imwrite(os.path.join(out, "full", name + ".jpg"), res8, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(out, "thumb", name + ".jpg"), _thumb(res8))
        vis = to_uint8(img)
        g = params.geometry
        if g.corners is not None:
            pts = geometry.outline_points(np.array(g.corners), g.bow if g.use_bow else None)
            cv2.polylines(vis, [pts.astype(np.int32)], True, (0, 0, 255), max(3, img.shape[1] // 600))
        cv2.imwrite(os.path.join(out, "outline", name + ".jpg"), _thumb(vis))
        meta = dict(name=name, ok=True, report=report, params=params.to_dict(),
                    src_size=[img.shape[1], img.shape[0]],
                    out_size=[result.shape[1], result.shape[0]],
                    t_auto=round(t1 - t0, 1), t_render=round(t2 - t1, 1))
    except Exception:
        meta = dict(name=name, ok=False, error=traceback.format_exc())
    with open(os.path.join(out, "meta", name + ".json"), "w", encoding="utf8") as fh:
        json.dump(meta, fh, indent=1)
    return {k: meta[k] for k in ("name", "ok") } | (
        {"out_size": meta["out_size"], "t": meta["t_auto"] + meta["t_render"]} if meta["ok"]
        else {"error": meta["error"].strip().splitlines()[-1]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pipeline", default="traditional", choices=["traditional", "ml"])
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    for sub in ("full", "thumb", "outline", "meta"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    names = sorted(n for n in os.listdir(a.src) if n.lower().endswith((".png", ".tif", ".tiff", ".jpg")))
    if a.only:
        keep = set(a.only.split(","))
        names = [n for n in names if os.path.splitext(n)[0] in keep]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process, os.path.join(a.src, n), a.out, a.pipeline): n for n in names}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            print(f"[{i}/{len(names)}] {r['name']} {'ok' if r['ok'] else 'FAILED ' + r['error']}",
                  flush=True)
    rows.sort(key=lambda r: r["name"])
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf8") as fh:
        json.dump(rows, fh, indent=1)
    print(f"done in {time.time() - t0:.0f}s, failed: {sum(not r['ok'] for r in rows)}")


if __name__ == "__main__":
    main()

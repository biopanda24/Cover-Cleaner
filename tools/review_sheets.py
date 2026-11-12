"""Contact sheets (outline on scan | result) and a decision table for a batch_eval run.

    python tools/review_sheets.py output/eval/run1 [--per 12] [--only a,b]
"""
from __future__ import annotations

import argparse
import json
import os
import re

import cv2
import numpy as np


def load(run):
    metas = []
    for f in sorted(os.listdir(os.path.join(run, "meta"))):
        metas.append(json.load(open(os.path.join(run, "meta", f), encoding="utf8")))
    return metas


def row_for(m):
    if not m["ok"]:
        return dict(name=m["name"], error=m["error"].strip().splitlines()[-1])
    p = m["params"]
    rep = " | ".join(m["report"])
    g = p["geometry"]
    w, h = m["out_size"]
    flags = []
    for key, pat in (("lowconf", "Low confidence"), ("notvis", "not visible"), ("uneven", "unevenly"),
                     ("inside", "inside the rough"), ("nobg", "not a uniform"), ("worn", "Worn"),
                     ("nocover", "No cover"), ("small", "too small"), ("fullframe", "keeping the full frame")):
        if pat in rep:
            flags.append(key)
    return dict(
        name=m["name"], src=m["src_size"], out=[w, h], ratio=round(w / h, 3),
        crop=g["corners"] is not None, trim=g["trim"], bow=g["use_bow"],
        descreen=round(p["descreen"]["period"], 2) if p["descreen"]["enabled"] else 0,
        shading=p["shading"]["enabled"], dust=p["defects"]["enabled"], flat=p["flat"]["enabled"],
        denoise=round(p["denoise"]["strength"], 2) if p["denoise"]["enabled"] else 0,
        black=[round(v * 255) for v in p["tone"]["black"][::-1]] if p["tone"]["enabled"] else None,
        white=[round(v * 255) for v in p["tone"]["white"][::-1]] if p["tone"]["enabled"] else None,
        flags=flags, t=round(m["t_auto"] + m["t_render"], 1))


def sheet(run, metas, idx, per, tag):
    tiles = []
    for i, m in zip(idx, metas):
        a = cv2.imread(os.path.join(run, "outline", m["name"] + ".jpg"))
        b = cv2.imread(os.path.join(run, "thumb", m["name"] + ".jpg")) if m["ok"] else None
        cell = np.full((420, 820, 3), 50, np.uint8)
        if a is not None:
            s = 400 / max(a.shape[:2])
            a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cell[:a.shape[0], :a.shape[1]] = a
        if b is not None:
            s = 400 / max(b.shape[:2])
            b = cv2.resize(b, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cell[:b.shape[0], 415:415 + b.shape[1]] = b
        label = f"{i} {m['name'][-16:]}"
        cv2.putText(cell, label, (4, 414), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(cell)
    while len(tiles) % 3:
        tiles.append(np.full((420, 820, 3), 50, np.uint8))
    rows = [np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)]
    cv2.imwrite(os.path.join(run, f"review_{tag}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--per", type=int, default=12)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    metas = load(a.run)
    idx = list(range(len(metas)))
    if a.only:
        keep = a.only.split(",")
        sel = [(i, m) for i, m in enumerate(metas) if m["name"] in keep or str(i) in keep]
        idx, metas = [i for i, _ in sel], [m for _, m in sel]
    rows = [row_for(m) for m in metas]
    with open(os.path.join(a.run, "table.json"), "w") as fh:
        json.dump(rows, fh, indent=0)
    for k in range(0, len(metas), a.per):
        sheet(a.run, metas[k:k + a.per], idx[k:k + a.per], a.per, f"{k // a.per:02d}")
    for i, r in zip(idx, rows):
        print(i, json.dumps(r, separators=(",", ":")))


if __name__ == "__main__":
    main()

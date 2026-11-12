"""Detect covers in selected scans and draw the outlines (fast; no restoration).

    python tools/outline_check.py Scans out.jpg name1 name2 ...
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402
import numpy as np  # noqa: E402


def one(path):
    from covercleaner import geometry as G
    from covercleaner.imageio import load_image, read_dpi, to_uint8
    img = load_image(path)
    d = G.detect_cover(img, dpi=read_dpi(path))
    vis = to_uint8(img)
    pts = G.outline_points(d.corners, d.bow)
    cv2.polylines(vis, [pts.astype(np.int32)], True, (0, 0, 255), 8)
    s = min(480 / vis.shape[1], 310 / vis.shape[0])
    vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    w, h = G.output_size(d.corners)
    label = f"{os.path.basename(path)[-14:-4]} {w}x{h} {w / h:.3f}"
    cell = np.full((330, 490, 3), 50, np.uint8)
    cell[:vis.shape[0], :vis.shape[1]] = vis
    cv2.putText(cell, label, (4, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return cell, label, [round(c, 2) for c in d.side_confidence], d.notes


if __name__ == "__main__":
    src, out, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(one, [os.path.join(src, n + ".png") for n in names]))
    cells = [r[0] for r in res]
    for r in res:
        print(r[1], r[2], [n[:70] for n in r[3]])
    while len(cells) % 4:
        cells.append(np.full((330, 490, 3), 50, np.uint8))
    cv2.imwrite(out, np.vstack([np.hstack(cells[k:k + 4]) for k in range(0, len(cells), 4)]))

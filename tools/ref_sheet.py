"""Overview sheet: reference crop (green) vs Auto crop (red) on the scan.

    python tools/ref_sheet.py Scans output/ref_truth.json out.jpg name-fragment ...
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tools.ref_compare import auto_crop  # noqa: E402


def one(args):
    scans, name, ref = args
    auto, det = auto_crop(os.path.join(scans, name))
    img = cv2.imdecode(np.fromfile(os.path.join(scans, name), np.uint8), cv2.IMREAD_COLOR)
    cv2.polylines(img, [np.array(ref["corners"], np.int32)], True, (0, 200, 0), 9)
    cv2.polylines(img, [auto.astype(np.int32)], True, (0, 0, 255), 5)
    s = min(480 / img.shape[1], 330 / img.shape[0])
    t = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cell = np.full((350, 490, 3), 50, np.uint8)
    cell[:t.shape[0], :t.shape[1]] = t
    cv2.putText(cell, name[8:-4], (4, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return cell


if __name__ == "__main__":
    scans, truth_p, out = sys.argv[1:4]
    frags = sys.argv[4:]
    truth = json.load(open(truth_p))
    names = [n for n in sorted(truth) if not frags or any(f in n for f in frags)]
    with ProcessPoolExecutor(8) as ex:
        cells = list(ex.map(one, [(scans, n, truth[n]) for n in names]))
    while len(cells) % 4:
        cells.append(np.full((350, 490, 3), 50, np.uint8))
    cv2.imwrite(out, np.vstack([np.hstack(cells[k:k + 4]) for k in range(0, len(cells), 4)]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])

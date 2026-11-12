"""Zoom on one side: reference edge (green) and Auto's crop (red).

    python tools/ref_zoom.py Scans output/ref_truth.json out.jpg name side [half_width]
side: top/right/bottom/left.  Shows a strip centred on the reference edge.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tools.ref_compare import SIDES, auto_crop  # noqa: E402

scans, truth_p, out, frag, side = sys.argv[1:6]
half = int(sys.argv[6]) if len(sys.argv) > 6 else 300
truth = json.load(open(truth_p))
name = [n for n in truth if frag in n][0]
ref = np.array(truth[name]["corners"])
auto, det = auto_crop(os.path.join(scans, name))
img = cv2.imdecode(np.fromfile(os.path.join(scans, name), np.uint8), cv2.IMREAD_COLOR)
cv2.polylines(img, [ref.astype(np.int32)], True, (0, 200, 0), 1)
cv2.polylines(img, [auto.astype(np.int32)], True, (0, 0, 255), 1)
i = SIDES.index(side)
tiles = []
for f in (0.2, 0.5, 0.8):
    p = ref[i] + f * (ref[(i + 1) % 4] - ref[i])
    x, y = int(p[0]), int(p[1])
    if side in ("left", "right"):
        crop = img[max(0, y - 60):y + 60, max(0, x - half):x + half]
    else:
        crop = img[max(0, y - half):y + half, max(0, x - 60):x + 60]
        crop = np.ascontiguousarray(np.rot90(crop))
    tiles.append(cv2.copyMakeBorder(crop, 0, 4, 0, 2 * half - crop.shape[1], cv2.BORDER_CONSTANT))
cv2.imwrite(out, np.vstack(tiles))
print(name, side, "notes:", det.notes, "conf", [round(c, 2) for c in det.side_confidence])

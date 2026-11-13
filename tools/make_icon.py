"""Generate the program icon (design B, "wiped cover") in every format.

    .venv/Scripts/python tools/make_icon.py

Writes covercleaner/assets/:
  icon.svg            the full design (plain shapes: no clip paths or
                      patterns, so Qt's SVG renderer draws it correctly)
  icon_<size>.png     16 ... 1024 px
  icon.ico            Windows (16-256 px)
  icon.icns           macOS (16-1024 px)

Small sizes use simplified drawings: no halftone dots or scratches below
64 px, and no sparkle below 32 px, so the icon stays crisp in title bars,
taskbars and file lists.
"""
from __future__ import annotations

import math
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "covercleaner", "assets")

TILE = "#0E6E6A"
PAPER = "#FFF4DC"
SUN = "#F59E0B"
HILL = "#2A9D8F"
GRIME = "#5C6670"
DOTS = "#2F3A44"
SPARKLE = "#FFD84D"

# detail level -> drawing parameters (on a 256 x 256 canvas)
LEVELS = {
    "full":   dict(tile_rx=56, x0=56, x1=200, rx=8, sun_r=0.153, grime=0.45, dots=True,
                   scratches=True, wipe=9, sparkle=(200, 54, 32)),
    "medium": dict(tile_rx=52, x0=44, x1=212, rx=10, sun_r=0.16, grime=0.62, dots=False,
                   scratches=False, wipe=15, sparkle=(206, 50, 30)),
    "small":  dict(tile_rx=48, x0=34, x1=222, rx=12, sun_r=0.17, grime=0.66, dots=False,
                   scratches=False, wipe=20, sparkle=None),
}
SIZES = {16: "small", 24: "small", 32: "medium", 48: "medium", 64: "full", 128: "full",
         256: "full", 512: "full", 1024: "full"}


def _f(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _sparkle(cx, cy, s) -> str:
    return (f'<path d="M{_f(cx)},{_f(cy - s)} Q{_f(cx)},{_f(cy)} {_f(cx + s)},{_f(cy)} '
            f'Q{_f(cx)},{_f(cy)} {_f(cx)},{_f(cy + s)} Q{_f(cx)},{_f(cy)} {_f(cx - s)},{_f(cy)} '
            f'Q{_f(cx)},{_f(cy)} {_f(cx)},{_f(cy - s)}Z" fill="{SPARKLE}"/>')


def _dots(x0, x1, rx) -> list[str]:
    """Halftone dots on a 30-degree grid, inside the grimy lower-left triangle."""
    step, r = 9.0, 2.4
    y0, y1 = x0, x1
    ux, uy = step * math.cos(math.radians(30)), step * math.sin(math.radians(30))
    vx, vy = -uy, ux
    out = []
    n = int((x1 - x0) / step) * 3
    for i in range(-n, n):
        for j in range(-n, n):
            x, y = x0 + i * ux + j * vx, y0 + i * uy + j * vy
            if not (x0 + r + 1 <= x and y <= y1 - r - 1 and (y - y0) - (x - x0) >= r * 2.2):
                continue
            if x < x0 + rx and y > y1 - rx and math.hypot(x - (x0 + rx), y - (y1 - rx)) > rx - r:
                continue
            out.append(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{r}"/>')
    return out


def svg(level: str) -> str:
    p = LEVELS[level]
    x0, x1, rx = p["x0"], p["x1"], p["rx"]
    y0, y1, w = x0, x1, x1 - x0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">',
             f'<rect width="256" height="256" rx="{p["tile_rx"]}" fill="{TILE}"/>',
             f'<rect x="{x0}" y="{y0}" width="{w}" height="{w}" rx="{rx}" fill="{PAPER}"/>',
             f'<circle cx="{_f(x0 + .708 * w)}" cy="{_f(y0 + .292 * w)}" r="{_f(p["sun_r"] * w)}" '
             f'fill="{SUN}"/>',
             f'<path d="M{x0},{_f(y0 + .778 * w)} Q{_f(x0 + .278 * w)},{_f(y0 + .5 * w)} '
             f'{_f(x0 + .556 * w)},{_f(y0 + .694 * w)} T{x1},{_f(y0 + .625 * w)} '
             f'L{x1},{_f(y1 - rx)} Q{x1},{y1} {_f(x1 - rx)},{y1} L{_f(x0 + rx)},{y1} '
             f'Q{x0},{y1} {x0},{_f(y1 - rx)}Z" fill="{HILL}"/>',
             # the grimy, halftoned half that has not been cleaned yet
             f'<path d="M{x0},{y0} L{x0},{_f(y1 - rx)} Q{x0},{y1} {_f(x0 + rx)},{y1} L{x1},{y1}Z" '
             f'fill="{GRIME}" fill-opacity="{p["grime"]}"/>']
    if p["dots"]:
        parts.append(f'<g fill="{DOTS}" fill-opacity=".55">' + "".join(_dots(x0, x1, rx)) + "</g>")
    if p["scratches"]:
        s = w / 144
        parts.append(f'<path d="M{_f(x0 + 14 * s)},{_f(y0 + 64 * s)} L{_f(x0 + 40 * s)},{_f(y0 + 94 * s)} '
                     f'M{_f(x0 + 28 * s)},{_f(y0 + 120 * s)} L{_f(x0 + 56 * s)},{_f(y0 + 130 * s)}" '
                     f'stroke="#E9EEF2" stroke-width="2.5" stroke-linecap="round"/>')
    parts.append(f'<line x1="{x0 - 6}" y1="{y0 - 6}" x2="{x1 + 6}" y2="{y1 + 6}" stroke="#FFFFFF" '
                 f'stroke-width="{p["wipe"]}" stroke-linecap="round"/>')
    if p["sparkle"]:
        parts.append(_sparkle(*p["sparkle"]))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render(svg_text: str, size: int) -> bytes:
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    r = QSvgRenderer(QByteArray(svg_text.encode()))
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    r.render(p, QRectF(0, 0, size, size))
    p.end()
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


def _dib(png: bytes) -> bytes:
    """A classic 32-bit bitmap icon image (what Windows and .NET read most
    reliably below 256 px): BITMAPINFOHEADER, BGRA rows bottom-up, AND mask."""
    from PySide6.QtGui import QImage
    img = QImage.fromData(png).convertToFormat(QImage.Format_ARGB32)
    w, h = img.width(), img.height()
    rows = [bytes(img.constBits())[y * img.bytesPerLine(): y * img.bytesPerLine() + 4 * w]
            for y in range(h)]
    mask_row = ((w + 31) // 32) * 4
    header = struct.pack("<IiiHHIIiiII", 40, w, 2 * h, 1, 32, 0, 4 * w * h + mask_row * h,
                         0, 0, 0, 0)
    return header + b"".join(reversed(rows)) + b"\0" * (mask_row * h)


def write_ico(path: str, pngs: dict[int, bytes]) -> None:
    """PNG for 256 px, classic bitmaps for the smaller sizes."""
    sizes = [s for s in (16, 24, 32, 48, 64, 128, 256) if s in pngs]
    head = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    entries, data = b"", b""
    for s in sizes:
        d = pngs[s] if s >= 256 else _dib(pngs[s])
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(d), offset)
        offset += len(d)
        data += d
    with open(path, "wb") as fh:
        fh.write(head + entries + data)


def write_icns(path: str, pngs: dict[int, bytes]) -> None:
    types = [(b"icp4", 16), (b"icp5", 32), (b"icp6", 64), (b"ic07", 128), (b"ic08", 256),
             (b"ic09", 512), (b"ic10", 1024), (b"ic11", 32), (b"ic12", 64), (b"ic13", 256),
             (b"ic14", 512)]
    body = b"".join(t + struct.pack(">I", 8 + len(pngs[s])) + pngs[s] for t, s in types)
    with open(path, "wb") as fh:
        fh.write(b"icns" + struct.pack(">I", 8 + len(body)) + body)


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "icon.svg"), "w", encoding="utf8", newline="\n") as fh:
        fh.write(svg("full"))
    pngs = {}
    for size, level in SIZES.items():
        pngs[size] = render(svg(level), size)
        with open(os.path.join(OUT, f"icon_{size}.png"), "wb") as fh:
            fh.write(pngs[size])
    write_ico(os.path.join(OUT, "icon.ico"), pngs)
    write_icns(os.path.join(OUT, "icon.icns"), pngs)
    print(f"wrote {OUT}: icon.svg, {len(pngs)} PNGs, icon.ico, icon.icns")
    return 0


if __name__ == "__main__":
    sys.exit(main())

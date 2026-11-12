"""End-to-end checks on synthetic scans where the ground truth is known."""
import cv2
import numpy as np
import pytest

from covercleaner import defects, descreen, geometry, shading, tone
from covercleaner.auto import auto_params
from covercleaner.params import Params
from covercleaner.pipeline import Pipeline


def _text(img, text, org, scale, colour, thick):
    """cv2.putText only draws on 8-bit images; composite it onto a float one."""
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.putText(m, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, 255, thick, cv2.LINE_AA)
    a = (m.astype(np.float32) / 255.0)[:, :, None]
    img[:] = img * (1 - a) + np.float32(colour) * a


def make_cover(size=900, seed=0):
    """A clean 'digital original': gradients, shapes, lettering."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.dstack([0.25 + 0.5 * x, 0.35 + 0.3 * y, 0.75 - 0.4 * x * y]).astype(np.float32)
    img = np.ascontiguousarray(img)
    for _ in range(6):
        c = tuple(float(v) for v in rng.uniform(0.1, 0.9, 3))
        p = rng.integers(60, size - 60, 2)
        cv2.circle(img, (int(p[0]), int(p[1])), int(rng.integers(30, 90)), c, -1, cv2.LINE_AA)
    cv2.rectangle(img, (40, size - 200), (size - 40, size - 120), (0.03, 0.03, 0.03), -1)
    _text(img, "COVER", (60, size - 135), 2.2, 0.97, 4)
    return np.clip(img, 0, 1)


def place_on_scanner(cover, angle=1.7, margin=140, shadow=True):
    """Rotate the cover and put it on a white scanner lid with a soft shadow."""
    h, w = cover.shape[:2]
    H, W = h + 2 * margin, w + 2 * margin
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[:, 2] += margin
    canvas = np.full((H, W, 3), 0.975, np.float32)
    alpha = cv2.warpAffine(np.ones((h, w), np.float32), m, (W, H), flags=cv2.INTER_LINEAR)
    warped = cv2.warpAffine(cover, m, (W, H), flags=cv2.INTER_CUBIC)
    if shadow:
        sh = cv2.GaussianBlur(np.roll(alpha, (-6, -4), axis=(0, 1)), (0, 0), 4)
        canvas *= (1 - 0.45 * sh)[:, :, None]
    scan = canvas * (1 - alpha[:, :, None]) + warped * alpha[:, :, None]
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    truth = (m[:, :2] @ corners.T).T + m[:, 2]
    noise = np.random.default_rng(1).normal(0, 0.004, scan.shape).astype(np.float32)
    return np.clip(scan + noise, 0, 1).astype(np.float32), truth


def halftone(img, period=3.6):
    """Crude AM halftone: per-channel rotated dot screens."""
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.empty_like(img)
    for c, ang in enumerate((15.0, 75.0, 45.0)):
        a = np.deg2rad(ang)
        u = (x * np.cos(a) + y * np.sin(a)) * 2 * np.pi / period
        v = (-x * np.sin(a) + y * np.cos(a)) * 2 * np.pi / period
        thr = 0.5 + 0.25 * (np.cos(u) + np.cos(v))
        out[:, :, c] = np.clip((img[:, :, c] - thr) * 4 + 0.5, 0, 1)
    return (cv2.GaussianBlur(out, (0, 0), 0.7) * 0.9 + 0.05).astype(np.float32)


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12)))


def test_cover_detection_finds_corners():
    scan, truth = place_on_scanner(make_cover(), angle=1.7)
    det = geometry.detect_cover(scan)
    assert det.has_background
    err = np.abs(det.corners - truth).max()
    assert err < 2.5, f"corner error {err:.2f}px"
    assert abs(abs(det.angle_deg) - 1.7) < 0.05
    out = geometry.apply_geometry(scan, det.corners, det.bow, det.trim)
    assert abs(out.shape[0] - 900) <= 14 and abs(out.shape[1] - 900) <= 14


def test_already_cropped_scan_is_left_alone():
    det = geometry.detect_cover(make_cover())
    assert not det.has_background


def test_curved_edge_is_straightened():
    scan, _ = place_on_scanner(make_cover(), angle=0.0, shadow=True)
    # bend the top edge downwards in the middle by 6 px
    h, w = scan.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.clip((xx - 140) / 900, 0, 1)
    fall = np.clip(1 - (yy - 140) / 450, 0, 1)
    bent = cv2.remap(scan, xx, yy - 6 * 4 * t * (1 - t) * fall, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    det = geometry.detect_cover(bent)
    assert 3.5 < det.max_bow < 9, det.max_bow
    assert np.abs(det.bow[0]).max() == pytest.approx(det.max_bow)


def test_screen_detection_and_descreen():
    clean = make_cover()
    dotted = halftone(clean, period=3.6)
    scr = descreen.analyze_screen(dotted)
    assert scr.found
    assert abs(scr.period - 3.6) < 0.25, scr.period
    out = descreen.descreen(dotted, scr.peaks, scr.period, 1.0, 0.75, 0.7)
    ref = cv2.GaussianBlur(clean, (0, 0), 1.5) * 0.9 + 0.05
    assert psnr(out, ref) > psnr(dotted, ref) + 6
    assert not descreen.analyze_screen(clean).found       # no false alarm


def test_dust_on_flat_colour_is_removed_and_detail_kept():
    img = np.full((700, 700, 3), (0.1, 0.25, 0.9), np.float32)
    _text(img, "small print 123", (60, 350), 0.6, 1.0, 1)
    clean = img.copy()
    spots = [(120, 110), (500, 150), (300, 560), (600, 600), (80, 620)]
    for i, (x, y) in enumerate(spots):
        cv2.circle(img, (x, y), 2 + i % 2, (1, 1, 1) if i % 2 else (0, 0, 0), -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    mask = defects.detect_dust(img, 0.5, 1.0)
    for x, y in spots:
        assert mask[y, x] == 255, f"missed speck at {(x, y)}"
    assert mask[330:360, 50:260].sum() == 0, "lettering mistaken for dust"
    fixed = defects.inpaint(img, mask)
    ref = cv2.GaussianBlur(clean, (0, 0), 0.8)
    assert np.abs(fixed - ref).max() < 0.03


def test_edge_shadow_is_detected_and_corrected():
    clean = make_cover(seed=3)
    ramp = np.ones(900, np.float32)
    ramp[:36] = 0.7 + 0.3 * np.linspace(0, 1, 36) ** 0.7
    shaded = clean * ramp[None, :, None]
    res = shading.analyze_shading(shaded, 4.0)
    assert res.is_shadow[0] and not any(res.is_shadow[1:]), (res.is_shadow, res.falloff)
    assert not any(shading.analyze_shading(clean, 4.0).is_shadow)
    fixed = shading.correct_shading(shaded, 4.0, 1.0, res.is_shadow)
    band = (slice(None), slice(0, 36))
    assert np.abs(fixed[band] - clean[band]).mean() < 0.35 * np.abs(shaded[band] - clean[band]).mean()


def test_tone_references():
    img = (make_cover() * 0.86 + 0.06).astype(np.float32)   # ink black ~0.09, white ~0.9
    ta = tone.analyze_tone(img)
    assert ta.has_black and ta.has_white
    out = tone.apply_tone(img, ta.black, ta.white)
    assert out.min() < 0.02 and out.max() > 0.98
    coloured = np.full((400, 400, 3), (0.8, 0.5, 0.9), np.float32)   # no black, no white
    tb = tone.analyze_tone(coloured)
    assert not tb.has_black and not tb.has_white


def test_auto_end_to_end_improves_on_plain_crop():
    clean = make_cover()
    scan, _ = place_on_scanner(halftone(clean), angle=-1.1)
    pipe = Pipeline(scan)
    params, report = auto_params(scan, pipeline=pipe)
    assert params.geometry.corners is not None and params.descreen.enabled
    assert any(line.startswith("Descreen") for line in report)
    result = pipe.run(params)
    plain = pipe.stage_output("geometry")
    assert result.shape == plain.shape
    # dot pattern energy (difference from a smoothed copy) must drop sharply
    def ripple(im):
        return float(np.abs(im - cv2.GaussianBlur(im, (0, 0), 2.5))[60:-260, 60:-60].mean())
    assert ripple(result) < 0.35 * ripple(plain)


def test_params_roundtrip_and_cache():
    p = Params()
    p.tone.enabled = True
    p.tone.saturation = 1.2
    q = Params.from_json(p.to_json())
    assert q.to_dict() == p.to_dict()
    pipe = Pipeline(make_cover(300))
    p.geometry.corners = None
    a = pipe.run(p)
    geo = pipe.stage_output("geometry")
    p.tone.saturation = 0.8
    b = pipe.run(p)
    assert pipe.stage_output("geometry") is geo           # early stages reused
    assert not np.allclose(a, b)

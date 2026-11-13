"""End-to-end checks on synthetic scans where the ground truth is known."""
import cv2
import numpy as np
import pytest

from covercleaner import defects, descreen, flat, geometry, ml, ocr, shading, tone
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


def place_on_scanner(cover, angle=1.7, margin=140, shadow=True, lid_gain=None):
    """Rotate the cover and put it on a white scanner lid with a soft shadow.

    `lid_gain` (H x W) darkens parts of the lid, e.g. where it did not lie flat.
    """
    h, w = cover.shape[:2]
    H, W = h + 2 * margin, w + 2 * margin
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[:, 2] += margin
    canvas = np.full((H, W, 3), 0.975, np.float32)
    if lid_gain is not None:
        canvas *= lid_gain[:, :, None]
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
    det = geometry.detect_cover(bent, fit_bow=True)
    assert 3.5 < det.max_bow < 9, det.max_bow
    assert np.abs(det.bow[0]).max() == pytest.approx(det.max_bow)


def _dark_left_lid(H, W, width=140):
    """Lid gain for a scan whose lid is darker along the left side (as in 4.png):
    about 4% darker at the image edge, 6% next to the cover, darkest mid-height."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    depth = (0.035 + 0.02 * np.clip(x / width, 0, 1)) * (0.6 + 0.4 * np.sin(np.pi * y / H))
    depth[x > width] = 0
    return cv2.GaussianBlur(1.0 - depth, (0, 0), 6)


def test_darker_lid_strip_is_not_mistaken_for_cover():
    cover = make_cover()
    H = W = 900 + 2 * 140
    scan, truth = place_on_scanner(cover, angle=0.6, shadow=False, lid_gain=_dark_left_lid(H, W))
    det = geometry.detect_cover(scan)
    err = np.abs(det.corners - truth).max()
    assert err < 2.5, f"corner error {err:.2f}px, corners {np.round(det.corners, 1).tolist()}"


def test_edge_far_inside_rough_outline_is_found(monkeypatch):
    """If the rough pass is off by more than the normal search band, the
    wider inward search still finds the paper edge."""
    scan, truth = place_on_scanner(make_cover(), angle=0.0, shadow=False)
    real = geometry._coarse_rect

    def too_wide(img, det):
        box = real(img, det)
        box[0, 0] -= 100            # left side 100 px out into the lid
        box[3, 0] -= 100
        return box

    monkeypatch.setattr(geometry, "_coarse_rect", too_wide)
    det = geometry.detect_cover(scan)
    err = np.abs(det.corners - truth).max()
    assert err < 2.5, f"corner error {err:.2f}px"
    assert any("inside the rough outline" in n for n in det.notes)


def test_worn_edge_is_trimmed_but_printed_edge_stripe_is_kept():
    cover = make_cover(size=1800)
    # left: worn, whitened paper fading over 10 px
    x = np.arange(10, dtype=np.float32)
    wear = (0.85 * (1 - x / 10) ** 1.2)[None, :, None]
    cream = np.float32([0.80, 0.90, 0.97])
    cover[:, :10] = cover[:, :10] * (1 - wear) + cream * wear
    # right: a printed stripe of even colour running along the edge
    cover[:, -10:] = (0.6, 0.2, 0.1)
    scan, truth = place_on_scanner(cover, angle=0.0, shadow=False)
    det = geometry.detect_cover(scan)
    left, top, right, bottom = det.trim
    assert 9 <= left <= 13, det.trim
    assert right <= 4 and top <= 4 and bottom <= 4, det.trim
    assert any("Worn" in n and "left" in n for n in det.notes)


def test_disc_is_found_and_outside_made_white():
    H, W = 1300, 1800
    scan = np.full((H, W, 3), 0.975, np.float32)
    cx, cy, r = 820.0, 640.0, 520.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.hypot(xx - cx, yy - cy)
    disc = make_cover(size=2 * int(r) + 2)[:H, :W]
    art = np.zeros_like(scan)
    y0, x0 = int(cy - r) - 1, int(cx - r) - 1
    art[y0:y0 + disc.shape[0], x0:x0 + disc.shape[1]] = disc[:H - y0, :W - x0]
    a = np.clip(r - d + 0.5, 0, 1)[:, :, None]
    hole = np.clip(d - 60 + 0.5, 0, 1)[:, :, None]          # centre hole shows the lid
    scan = scan * (1 - a * hole) + art * a * hole
    scan += np.random.default_rng(3).normal(0, 0.004, scan.shape).astype(np.float32)
    scan = np.clip(scan, 0, 1).astype(np.float32)
    det = geometry.detect_cover(scan)
    assert det.disc is not None, det.notes
    fx, fy, fr = det.disc
    assert abs(fx - cx) < 1.5 and abs(fy - cy) < 1.5 and abs(fr - r) < 1.5, det.disc
    out = geometry.apply_geometry(scan, det.corners, None, det.trim, 0, disc=True)
    assert out[2, 2].min() > 0.99 and out[-3, -3].min() > 0.99      # outside: white
    m = out.shape[0] // 2
    assert np.allclose(out[m, m], scan[int(cy), int(cx)], atol=0.05)


def test_text_angle_levels_straight_text():
    img = np.full((900, 900, 3), 0.95, np.float32)
    for k, word in enumerate(("CRESCENT", "MAY 2009", "LIMITED", "ANGCD-000")):
        _text(img, word, (180, 250 + 110 * k), 1.8, 0.1, 3)
    for true in (5.0, -12.0):
        m = cv2.getRotationMatrix2D((450, 450), true, 1.0)
        rot = cv2.warpAffine(img, m, (900, 900), borderValue=(0.95, 0.95, 0.95))
        ang, strength = geometry.text_angle(rot)
        assert abs(ang + true) < 0.5 and strength > 1.6, (true, ang, strength)
        level = geometry.apply_geometry(rot, None, angle=ang)
        assert abs(geometry.text_angle(level)[0]) < 0.5


@pytest.mark.skipif(not ocr.available(), reason="text models not downloaded")
def test_upright_turn_is_found_from_text():
    img = np.full((900, 900, 3), 0.95, np.float32)
    for k, word in enumerate(("Crescent", "May 2009", "ANGCD-000", "Limited article")):
        _text(img, word, (120, 220 + 140 * k), 2.0, 0.1, 4)
    for k in range(4):
        turns, margin, scores = ocr.upright_turns(np.ascontiguousarray(np.rot90(img, k)))
        assert turns == (4 - k) % 4 and margin >= 2.0, (k, turns, scores)


def test_cover_running_off_the_scan_keeps_that_edge():
    scan, truth = place_on_scanner(make_cover(), angle=0.0, shadow=False)
    cut = np.ascontiguousarray(scan[:900])                 # bottom of the cover cut off
    det = geometry.detect_cover(cut)
    assert det.has_background and "bottom" in det.off_sides, det.notes
    assert np.abs(det.corners[0] - truth[0]).max() < 2.5
    assert np.abs(det.corners[1] - truth[1]).max() < 2.5
    assert abs(det.corners[2, 1] - 899) < 1 and abs(det.corners[3, 1] - 899) < 1
    assert det.trim[3] == 0


def test_invisible_edge_is_placed_from_cd_booklet_size():
    # a white cover (120 mm at 600 dpi = 2835 px) whose left edge is invisible
    side = 2835
    cover = np.full((side, side, 3), 0.975, np.float32)       # same white as the lid
    cover[:, 300:] = make_cover(size=side)[:, 300:]          # picture, white left margin
    scan, truth = place_on_scanner(cover, angle=0.0, margin=150, shadow=False)
    # a thin lid shadow along the top edge, across the white margin too, is
    # what lets the rough outline reach the true corner (as on the real scan)
    shade = np.ones(scan.shape[:2], np.float32)
    shade[150 - 7:150, 150:150 + side] = 0.75
    scan = (scan * cv2.GaussianBlur(shade, (0, 0), 1.5)[:, :, None]).astype(np.float32)
    det = geometry.detect_cover(scan, dpi=600.0)
    w, h = geometry.output_size(det.corners)
    assert abs(w - side) <= 6 and abs(h - side) <= 6, (w, h, det.notes)
    assert any("CD booklet" in n for n in det.notes)
    assert np.abs(det.corners - truth).max() < 8
    # without a resolution the margin is lost, but nothing else breaks
    det2 = geometry.detect_cover(scan)
    assert geometry.output_size(det2.corners)[0] < side - 200


def test_cover_cut_by_image_border_is_not_cropped_into():
    """A cover running off the image on one side must keep that side at the
    image border, not jump to some edge inside the picture."""
    cover = np.ascontiguousarray(make_cover()[:450])   # wide, short (VHS-like)
    scan, truth = place_on_scanner(cover, angle=0.0, shadow=False)
    cut = np.ascontiguousarray(scan[:, 200:])          # cover now starts before x = 0
    det = geometry.detect_cover(cut)
    assert det.has_background
    assert max(det.corners[0, 0], det.corners[3, 0]) < 4, np.round(det.corners, 1).tolist()
    assert np.abs(det.corners[1:3] - (truth[1:3] - [200, 0])).max() < 2.5
    assert not any("inside the rough outline" in n for n in det.notes)


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


def _poster(size=700):
    """Flat artwork: two colours, a dark logo, plus a smooth photo-like gradient."""
    img = np.empty((size, size, 3), np.float32)
    img[:] = (0.88, 0.74, 0.97)
    cv2.rectangle(img, (120, 120), (480, 480), (0.93, 0.86, 0.98), -1)
    _text(img, "LOGO", (135, 345), 3.6, 0.0, 26)
    y = np.linspace(0, 1, 160, dtype=np.float32)[:, None]
    img[520:680, 40:660] = 0.2 + 0.6 * y[:, :, None]            # gradient strip
    img[:] = cv2.GaussianBlur(img, (0, 0), 0.8)
    return img


def test_flat_colour_cleanup_restores_solid_areas_and_keeps_gradients():
    clean = _poster()
    rng = np.random.default_rng(5)
    blotch = cv2.GaussianBlur(rng.normal(0, 1, clean.shape[:2]).astype(np.float32), (0, 0), 25)
    blotch = blotch / np.abs(blotch).max() * 0.035                # uneven scan / ink mottle
    scan = clean + blotch[:, :, None] + rng.normal(0, 0.008, clean.shape).astype(np.float32)
    scan = scan * 0.93 + 0.05                                     # grey blacks
    scan = np.clip(scan, 0, 1).astype(np.float32)
    res = flat.find_flat_regions(scan)
    assert res.coverage > 0.5
    out, mask = flat.flatten(scan, result=res)
    bg = out[20:100, 20:680].reshape(-1, 3)
    assert np.ptp(bg, axis=0).max() < 1e-4, "background is not one even colour"
    assert np.ptp(out[150, 140:460], axis=0).max() < 1e-4         # inner panel too
    letter = out[250:350, 150:470]
    assert (letter.max(axis=2) < 0.01).mean() > 0.15, "logo not snapped to pure black"
    grad = (slice(540, 660), slice(60, 640))
    assert mask[grad].mean() < 5, "smooth gradient was flattened"
    assert np.abs(out[grad] - scan[grad]).max() < 1e-6
    # protected pixels are never changed
    prot = np.zeros(scan.shape[:2], np.uint8)
    prot[40:60, 40:60] = 255
    out2, _ = flat.flatten(scan, result=res, protect=prot)
    assert np.array_equal(out2[40:60, 40:60], scan[40:60, 40:60])


def test_flat_cleanup_leaves_photos_alone():
    rng = np.random.default_rng(2)
    photo = cv2.GaussianBlur(rng.uniform(0, 1, (600, 600, 3)).astype(np.float32), (0, 0), 12)
    photo = (photo - photo.min()) / (photo.max() - photo.min())
    assert flat.find_flat_regions(photo).coverage < 0.05


@pytest.mark.skipif(not ml.available(), reason="LaMa model file not downloaded")
def test_ml_inpainting_beats_diffusion_on_structure_and_matches_on_flat():
    # structured picture: diagonal stripes with a hole across them
    y, x = np.mgrid[0:600, 0:600].astype(np.float32)
    stripes = (np.sin((x + y) / 9.0) > 0).astype(np.float32)
    img = np.dstack([0.2 + 0.6 * stripes, 0.3 + 0.4 * stripes, 0.8 - 0.5 * stripes])
    img = cv2.GaussianBlur(img, (0, 0), 1.0).astype(np.float32)
    mask = np.zeros((600, 600), np.uint8)
    cv2.circle(mask, (300, 300), 22, 255, -1)
    sel = mask > 0
    err_ml = np.abs(ml.inpaint(img, mask) - img)[sel].mean()
    err_td = np.abs(defects.inpaint(img, mask) - img)[sel].mean()
    assert err_ml < 0.6 * err_td, (err_ml, err_td)
    # flat colour: routed to plain diffusion, which is exact
    flat_img = np.full((300, 300, 3), (0.2, 0.5, 0.9), np.float32)
    m2 = np.zeros((300, 300), np.uint8)
    cv2.circle(m2, (150, 150), 10, 255, -1)
    assert np.abs(ml.inpaint(flat_img, m2) - flat_img).max() < 2e-3
    # pipeline switch
    p = Params()
    pipe = Pipeline(img)
    pipe.ensure_masks(img.shape)
    pipe.heal_mask[:] = mask
    pipe.masks_changed()
    p.geometry.corners = None
    a = pipe.run(p, upto="defects")
    p.mode = "ml"
    b = pipe.run(p, upto="defects")
    assert np.abs(b - img)[sel].mean() < np.abs(a - img)[sel].mean()
    assert Params.from_json(p.to_json()).mode == "ml"


def _stapled(picture: bool):
    """A 300 dpi cream booklet with two grey staples at the left (spine) edge."""
    from covercleaner import staples  # noqa: F401
    n = 900
    img = np.full((n, n, 3), (0.86, 0.95, 0.97), np.float32)
    if picture:                              # colour changing along the spine
        y = np.arange(n, dtype=np.float32)[:, None, None]
        img = img * (0.88 + 0.12 * np.sin(y / 30.0)) * np.ones((1, n, 1), np.float32)
    for y in (int(0.2 * n), int(0.78 * n)):
        img[y:y + 142, 0:4] = 0.45          # 12 mm of silver-grey wire
    return img


def test_staples_found_and_filled_on_solid_colour():
    from covercleaner import staples
    img = _stapled(picture=False)
    r = staples.detect(img, dpi=300)
    assert r.side == 3 and len(r.staples) == 2 and all(s.solid for s in r.staples)
    p = Params()
    p.geometry.enabled = False
    p.defects.patches = [list(s.box) for s in r.staples]
    out = Pipeline(img).run(p, upto="defects")
    assert np.abs(out[:, :8] - img[400:500, 40:48].mean(axis=(0, 1))).max() < 0.05
    clean = np.full_like(img, (0.86, 0.95, 0.97))
    assert not staples.detect(clean, dpi=300).staples


def test_staples_over_picture_are_not_solid():
    from covercleaner import staples
    r = staples.detect(_stapled(picture=True), dpi=300)
    assert r.side == 3 and r.staples and not all(s.solid for s in r.staples)

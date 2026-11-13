# Cover Cleaner

Restores scanned media covers (CD, DVD, VHS, cassette) towards what the
original artwork looked like before it was printed and scanned.

## Run

```
CoverCleaner.bat                 open the editor
CoverCleaner.bat Scans           open the editor on a folder
.venv\Scripts\python -m covercleaner auto Scans -o output --save-params
.venv\Scripts\python -m covercleaner apply scan.png -p scan.json -o fixed.png
```

First-time setup: `python -m venv .venv` then
`.venv\Scripts\python -m pip install -r requirements.txt`.

## What it fixes

Stages run in this order; each works on the output of the one before.

| Stage | Problem | Method |
|---|---|---|
| Crop / deskew / flatten | scanner background, crooked scan, curved edges, unevenly lit lid | Background colour modelled per side from the image border (a lid that is darker along one side, e.g. lifted by a booklet spine, is not mistaken for cover), then each paper edge is located at full resolution (outermost sharp transition, stepping over the lid shadow) and fitted with a robust line. Covers are flat rectangles, so the outline is squared up (one rotation, no perspective); curved-edge flattening is available but off in Auto. |
| Descreen | halftone dot pattern, rosette/moire | The averaged power spectrum reveals the screen frequencies; those are notched out, plus a low-pass just below the screen frequency. Solid lettering falls back to notch-only so it stays crisp. |
| Edge shading | darker edges where paper lifted | Median of pixel-to-pixel log differences towards the edge measures the shadow independent of picture content. |
| Dust / specks | dust, print voids | Small blobs that stand out from a median-filtered copy, sit on a uniform surrounding and are isolated; inpainted. |
| Glare | reflections on glossy / creased paper | Soft neutral bumps in the darkest channel are subtracted rather than painted over. |
| Flat colour cleanup | mottled, blotchy or tinted areas that were one solid colour in the artwork | Locally even areas with a consistent colour and artwork-like outlines become regions; matching colours are unified into one palette colour across the cover, near-black / near-white snap to pure values, the gap between two flat colours (ink fringe, mottle and scanner blur where a region stops short of its outline) is redrawn as a crisp, evenly anti-aliased mix of the two clean colours - only where the pixels really are such a mix, so photo shading and third colours such as a grey shadow line are left alone - and faint blemishes inside a region are absorbed. |
| Denoise | paper / scanner grain | Bilateral filter, stronger on colour than on brightness. |
| Tone | grey blacks, tinted paper white | Ink black and paper white are measured and mapped to 0 / 255. |
| Sharpen | softness left by the print screen | Unsharp mask on brightness with halo limiting. |

## Traditional and ML pipelines

The **Pipeline** selector in the toolbar (or `--pipeline traditional|ml`)
switches between:

* **Traditional** - classical image processing only.
* **ML** - the same pipeline, but dust and Heal-brush repairs inside textured
  or structured areas are filled by the LaMa inpainting network instead of
  colour diffusion. Holes surrounded by flat colour still use diffusion,
  which is exact there.

Measured on these scans by hiding real picture content under artificial
defects and comparing the fill with the original (mean error, 0-255): city
photo 11.9 -> 4.3, tape photo 8.1 -> 4.2, leaf collage 6.6 -> 2.6. ML is
slower: about 2 s per 512 px window on the CPU.

ML needs one model file, `models/inpainting_lama_2025jan.onnx` (92.6 MB,
Apache-2.0), from OpenCV's model zoo:
<https://huggingface.co/opencv/inpainting_lama>. It runs with OpenCV's DNN
module (OpenCV 5), so no other framework is required. Without the file the
ML option is greyed out and everything falls back to Traditional. ML is the
default whenever the file is present.

Descreening is traditional in both modes: no descreening network from a
trustworthy source was found.

Discs are turned upright with two more models from OpenCV's model zoo
(Apache-2.0), used in both pipelines when present in `models/`:
`text_detection_en_ppocrv3_2023may.onnx` (2.4 MB, from
<https://huggingface.co/opencv/text_detection_ppocr>) finds lines of text and
`text_recognition_CRNN_CH_2021sep.onnx` (65 MB, from
<https://huggingface.co/opencv/text_recognition_crnn>) reads them. The
quarter turn at which the most characters read with 80 % confidence wins,
if it has at least 8 such characters and twice as many as any other turn.
The recogniser reads Latin letters, digits and symbols; a disc with only
Japanese text is left as it is.

## Auto mode

Auto analyses the scan and only switches a fix on when it finds evidence for
it; the report panel lists every decision. It is deliberately conservative
where a mistake would damage the picture:

* Dust is removed automatically only on flat colour. Dots inside photos
  (catch lights, reflections) are left alone - raise *Dust sensitivity* above
  0.75 or use the Heal brush.
* Flat colour cleanup is switched on only when at least half the cover is
  solid-colour artwork; on photo covers it would posterise even patches.
  Thin lettering (strokes under about 15 px) is too narrow to be detected
  as a flat area and is left as scanned. Use the Protect brush to exclude a
  spot, and *Tolerance* to make it more or less aggressive.
* Glare is reported but never applied automatically, because soft highlights
  in a photo look the same.
* Black / white points are only set from neutral ink black and bare paper
  white, so coloured covers keep their palette.
* A paper edge is the outermost *sharp* transition: soft lid shadows and a
  lid lifted by a booklet spine (a gradient up to ~1000 px wide) are
  stepped over, as is a smooth, neutral shadow line just outside the paper.
* If an edge is not found near the rough outline, it is searched for up to
  45% of the side further inward, and accepted only when it is a strong
  straight edge with nothing but plain lid outside it (no print texture).
  A side without a trustworthy edge is kept parallel/perpendicular to the
  reliable sides (a cover is a rectangle).
* Edges that cannot be seen at all (white cover on a white lid) are placed
  from the standard sizes - CD booklet 120 x 120 mm, slim-case insert
  135.5 x 120 mm - using the DPI stored in the scan, then moved onto any
  faint edge found near that guess. A reliable edge 2-6% off a standard
  size is also checked against it. Faint tint changes (a pale cover on a
  neutral lid) count as edges. The report says when this happened.
* A thin rim at the edge (lid shadow, a slightly larger sheet, the disc
  tray or another item under the jacket) is cropped off up to the straight
  edge of the print: white rims up to 15 px, dark ones up to 20 px, neutral
  or not-parallel ones up to 45 px, and items under the jacket up to 85 px.
* Staples at a booklet's spine (thin silver lines 9-16 mm long right at the
  edge, in a symmetric pair) are filled in when they sit on solid colour;
  over a picture, that edge is trimmed just past them instead.
* A cover running past one or two edges of the scan keeps those edges.
* Discs (CD/DVD, recognised by shape and 120/80 mm size) are cropped square,
  everything outside the disc becomes white, and the disc is turned so its
  straight text is level. It is then turned upright by reading its text at
  all four quarter turns (see below); without the text models, use the
  rotate buttons. *Fine rotate* in the editor adjusts any cover by degrees.
  Covers are not turned this way: many have deliberately sideways titles.
* A paper edge that is white on a white scanner lid with no shadow cannot be
  seen; the report flags it so the corner can be dragged by hand.
* Worn or discoloured paper along an edge (a whitened spine fold, the white
  core of a cut edge, lid light glowing onto a dark cover) is trimmed off,
  up to 0.8% of the side (18 px at 600 dpi). It must shift the colour the
  same way along at least 75% of the side, be clearly visible at the edge
  and fade inward; picture or a printed stripe running to the edge does
  not, and is kept. The report lists what was trimmed; the Trim boxes
  override it.

## Manual control

* **Original + outline** view: drag the four green corners.
* Every fix has an on/off box and sliders; changes re-render from the first
  affected stage only.
* **Heal / Protect / Eraser** brushes paint repairs or exclusions directly on
  the result. *Show repairs* overlays what was changed.
* **Pick black / Pick white** set the tone references from a click.
* Hold **Space** (or *Hold: before*) to compare with the unprocessed crop.
* Settings can be saved as JSON and re-applied from the command line.

## Layout

```
covercleaner/
  geometry.py   cover detection and rectification
  descreen.py   halftone analysis and removal
  shading.py    edge shadow correction
  defects.py    dust detection, inpainting, glare
  flat.py       flat colour cleanup
  ml.py         LaMa inpainting backend (ML pipeline)
  ocr.py        text detection/recognition: which way up a disc goes
  tone.py       levels, colour, denoise, sharpen, resize
  pipeline.py   staged, cached pipeline
  auto.py       Auto mode analysis
  params.py     all settings (JSON serialisable)
  cli.py        command line / batch
  gui/          PySide6 editor
tests/          synthetic ground-truth tests (pytest)
tools/          batch_eval.py (Auto on a folder, in parallel, for review),
                review_sheets.py (before/after sheets + decision table),
                outline_check.py / edge_zoom.py (inspect detected outlines)
```

`Scans/`, `output/` and `models/` are not tracked by git.

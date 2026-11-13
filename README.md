# Cover Cleaner

Restores scanned media covers (CD, DVD, VHS, cassette) towards what the
original artwork looked like before it was printed and scanned.

## Install

Download the installer for your system from the
[releases page](https://github.com/biopanda24/Cover-Cleaner/releases):

- **Windows:** `CoverCleaner-<version>-windows-setup.exe`. Installs for the
  current user (no administrator rights needed) under
  `%LOCALAPPDATA%\Programs\Cover Cleaner`, with a Start menu entry.
- **Linux:** `bash CoverCleaner-<version>-linux.run` (`--help` for the
  options, e.g. `--yes` to accept the recommended answers). Installs to
  `~/.local/share/cover-cleaner` with an applications-menu entry and a
  `cover-cleaner` command. The Qt window needs the `libxcb-cursor0`
  system library.
- **macOS:** unzip `CoverCleaner-<version>-macos.zip` and double-click the
  `.command` file (if macOS blocks it: right-click > Open). Installs
  *Cover Cleaner* in `~/Applications`.

Every installer is self-contained: it puts its own copy of Python (the
exact version in `.python-version`) and the exact, checksum-verified
package versions from `requirements.lock` inside the install folder, using
a pinned version of [uv](https://docs.astral.sh/uv/). It never uses or
changes a Python already on the computer, adds nothing to PATH, and the
uninstaller removes everything. It asks whether to also download - both
recommended and on by default:

- the **ML models** (LaMa inpainting and the disc text models, about
  160 MB);
- **NVIDIA GPU acceleration** for ML (about 1.5 GB; Windows and Linux,
  only offered when an NVIDIA GPU is found).

Either can be added later from an installed copy with
`python -m covercleaner.setup_models --models --gpu`.

## Updating

Cover Cleaner checks GitHub for a new version when it starts (turn this off
in *Options > Processing > Updates*) and from *Help > About > Check for
updates*. *Update now* downloads the new version, checks it against its
published SHA-256, closes Cover Cleaner, installs it (keeping the
downloaded models and your options), and starts it again; if anything goes
wrong the previous version is put back (see `update.log` in the install
folder). Copies run from a git checkout update with `git pull` instead.

## Building the installers

```
python installer/build.py                      all targets, into dist/
python installer/build.py --targets linux,macos
```

What gets installed is pinned in three places:

- `.python-version` - the Python version;
- `requirements.lock` and `requirements-gpu.lock` - every package, with
  its SHA-256, for Windows, Linux and macOS. They are written by
  `.venv\Scripts\python installer/lock.py` from the development
  environment, so to upgrade a package: upgrade it in `.venv`, run the
  tests, run `lock.py`. `requirements.txt` only lists what is needed;
- `UV_VERSION` in `installer/build.py` (and the workflow) - uv.

The workflow runs the tests on Windows, Linux and macOS with exactly that
Python and those packages on every push, and builds the installers only
if they pass.

The Linux and macOS installers and the app archive used by updates can be
built on any system. The Windows setup .exe needs
[Inno Setup 6](https://jrsoftware.org/isinfo.php) (without it a zip with
an `Install Cover Cleaner.cmd` is built instead). Pushing a version tag
(`git tag v0.4.0 && git push origin v0.4.0`) runs
`.github/workflows/installers.yml`, which builds everything on GitHub and
publishes the release that the in-app updater looks for.

## Run

```
CoverCleaner.bat                 open the editor
CoverCleaner.bat Scans           open the editor on a folder (or drop
                                 scans / folders onto the window or the .bat)
.venv\Scripts\python -m covercleaner auto Scans -o output --save-params
.venv\Scripts\python -m covercleaner apply scan.png -p scan.json -o fixed.png
```

`auto` processes several scans at once (about one per four CPU threads,
limited by free memory); set the number with `-j`, e.g. `-j 1` for one at
a time.

From a source checkout, first-time setup: `python -m venv .venv` then
`.venv\Scripts\python -m pip install --require-hashes -r requirements.lock`
(the tested versions) and `pip install pytest` for the tests.

## What it fixes

Stages run in this order; each works on the output of the one before.

| Stage | Problem | Method |
|---|---|---|
| Crop / deskew / flatten | scanner background, crooked scan, curved edges, unevenly lit lid | Background colour modelled per side from the image border (a lid that is darker along one side, e.g. lifted by a booklet spine, is not mistaken for cover), then each paper edge is located at full resolution (outermost sharp transition, stepping over the lid shadow) and fitted with a robust line. Covers are flat rectangles, so the outline is squared up (one rotation, no perspective); curved-edge flattening is available but off in Auto. |
| Descreen | halftone dot pattern, rosette/moire | The averaged power spectrum reveals the screen frequencies; those are notched out, plus a low-pass just below the screen frequency. Solid lettering falls back to notch-only so it stays crisp. |
| Edge shading | darker edges where paper lifted | Median of pixel-to-pixel log differences towards the edge measures the shadow independent of picture content. |
| Dust / specks | dust, print voids | Small blobs that stand out from a median-filtered copy, sit on a uniform surrounding and are isolated; inpainted. |
| Glare | reflections on glossy / creased paper | Soft neutral bumps in the darkest channel are subtracted rather than painted over. |
| Flat colour cleanup | mottled, blotchy or tinted areas that were one solid colour in the artwork | Locally even areas with a consistent colour and artwork-like outlines become regions; matching colours are unified into one palette colour across the cover, near-black / near-white snap to pure values, each section is grown out to its true edge and painted with its single colour (grain, mottle, flare and small specks included; stains and soft shading fade out instead of ending in a hard edge; dots and hairlines that clearly stand out are kept), pockets of a section's colour cut off by lettering (the gaps between the strokes of a character on a coloured band) join that colour when flat sections enclose them, they are smooth and not thin streaks - even where blur from the strokes has darkened them - and only the thin boundary between two sections is redrawn as a crisp, evenly anti-aliased mix of their two clean colours - only where the pixels really are such a mix, so photo shading and third colours such as a grey shadow line are left alone - and faint blemishes inside a region are absorbed. |
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
slower on the CPU (about 3 s per 512 px window); on an NVIDIA GPU it takes
about 0.16 s (see below).

ML needs one model file, `models/inpainting_lama_2025jan.onnx` (92.6 MB,
Apache-2.0), from OpenCV's model zoo:
<https://huggingface.co/opencv/inpainting_lama>. It runs with OpenCV's DNN
module (OpenCV 5), so no other framework is required. Without the file the
ML option is greyed out and everything falls back to Traditional. ML is the
default whenever the file is present.

**GPU (optional, NVIDIA):** with ONNX Runtime's CUDA build installed,

```
.venv\Scripts\python -m pip install "onnxruntime-gpu[cuda]==1.23.2" "nvidia-cudnn-cu12==9.10.2.21" onnx
```

LaMa runs on the GPU, about 20 times faster than on the CPU. On first use
a float32 copy of the model is written next to it
(`inpainting_lama_2025jan.float.onnx`, 210 MB): the original stores
block-quantized weights the CUDA provider cannot run, and its shape
arithmetic is folded to constants so the whole network stays on the GPU
(`tools/lama_float.py`). cuDNN is pinned to 9.10: the 9.2x builds fail to
initialise with onnxruntime-gpu 1.23. Results match the CPU path to within
1% in the repaired pixels. Batch runs then use about 1.9 GB of GPU memory
per scan processed at once, which caps the default `-j`. Set
`COVERCLEANER_DEVICE=cpu` to force the CPU.

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

## Options

*Settings > Options...* (Ctrl+,) in the editor. They are stored in
`%APPDATA%\CoverCleaner\settings.json` and are also the defaults of the
command line, whose flags override them for one run (`--format`, `--bits`,
`--scale`, `--suffix`, `--[no-]save-params`, `--pipeline`, `--[no-]gpu`,
`--[no-]flat`, `--dpi`, `-j`).

* **Output:** default save location (a `restored` folder next to the
  scans unless set; results are never written into a scans folder itself), format, bit depth, JPEG quality, file name suffix,
  overwrite or number new files (`name (2).png`), save the chosen settings
  (.json) with each result, output scale.
* **Processing:** default pipeline (automatic / ML / Traditional), use the
  GPU for ML inpainting, scans processed at once (automatic or a number),
  fallback DPI for scans that store none.
* **Auto:** *use flat colour cleanup when flat sections are detected*
  (**Experimental**, off by default - Auto then only reports flat artwork;
  the fix can still be switched on per scan in the Fixes panel), and
  switches for the crop rules: standard-size placement, white-border
  restore, cropping things showing past one edge, staples.
* **Models:** which model files are installed, and whether the GPU can be
  used.
* **About:** version and credits. *Help > About* shows the same.

## Versions

Cover Cleaner uses semantic versioning (see `CHANGELOG.md`): the minor
number goes up for new features, the patch number for fixes. The version is
shown in the window title, in *Help > About* and by
`python -m covercleaner --version`.

## Auto mode

Auto analyses the scan and only switches a fix on when it finds evidence for
it; the report panel lists every decision. It is deliberately conservative
where a mistake would damage the picture:

* Dust is removed automatically only on flat colour. Dots inside photos
  (catch lights, reflections) are left alone - raise *Dust sensitivity* above
  0.75 or use the Heal brush.
* Flat colour cleanup (Experimental) is switched on only when allowed in
  Options and at least half the cover is solid-colour artwork; on photo covers it would posterise even patches.
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
* A white border that the crop cut off (white paper on a white lid, the
  paper edge invisible) is given back: when the other sides show a clear
  border and the scan outside this side is plain white, the side moves out
  by the same border width (onto a faint paper edge if one shows), as long
  as the cover then comes out within 1.5% of a standard size. The reverse
  also holds: a thin plain-white strip along one side only, while the other
  sides are printed to the edge, is cropped off.
* Something lying under the jacket and showing past one edge (a booklet,
  tray card, shadow line) is cropped off when that edge alone has a band
  unlike the artwork along its whole length, ending in a sharp step; bands
  over 1 mm must look like paper, and the same colour at the opposite edge
  marks a printed frame, which is kept.
* An edge that cannot be seen at all is placed from the standard size up
  to 6% away (normally 4%).
* Staples on a side whose edge was placed from the size, not seen, are
  also looked for up to 1.5 mm inside the crop.
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
  updater.py    new-version check and in-program updates (GitHub releases)
  update_helper.py  installs a downloaded update after the program closes
  setup_models.py   downloads the ML models / sets up GPU support
  net.py        HTTPS via the system certificate store
  gui/          PySide6 editor
  assets/       program icon (PNG sizes, .ico, .icns; from tools/make_icon.py)
installer/      build.py, lock.py and the installer sources
tests/          synthetic ground-truth and updater tests (pytest)
tools/          batch_eval.py (Auto on a folder, in parallel, for review),
                review_sheets.py (before/after sheets + decision table),
                outline_check.py / edge_zoom.py (inspect detected outlines),
                make_icon.py (draws the program icon in every format)
```

`Scans/`, `output/` and `models/` are not tracked by git.

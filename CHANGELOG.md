# Changelog

Version numbers follow semantic versioning: the minor number goes up for
new features, the patch number for fixes.

## 0.7.0 - 2026-11-13

- Crash reports. The editor keeps a session log (files opened, Auto runs,
  renders, background jobs, free memory) and writes a report to the logs
  folder when something goes wrong: an uncaught Python error in any thread
  or a failed background job (shown straight away, the program keeps
  running), or a hard crash inside native code (OpenCV, Qt, onnxruntime, a
  GPU driver), captured with the stack of every thread as the program dies
  and offered at the next start. A session that ended without an error
  (Task Manager, out of memory) is reported too. *Help* has *Open crash
  reports and logs folder* and *Report a problem on GitHub*. Nothing is sent
  automatically.
- Dropping a single image (or choosing one with *File > Open*) opens just
  that image instead of every scan in its folder. *File > Open* can now
  pick several images; *Open folder* opens a whole folder.

## 0.6.0 - 2026-11-13

- Switching between scans in the list no longer starts each one over.
  Every scan you have opened keeps its settings, your edits (sliders,
  corners, staple fills), its painted repairs and the analysis report for
  the session, so going back to it skips Auto. The last few scans - up to
  four, while at least 2.5 GB of memory stays free - also keep the loaded
  image and finished result, so going back is instant. An Auto run or
  render that finishes after you moved on is kept for that scan instead
  of being thrown away. *Auto* still re-analyses a scan from scratch.

## 0.5.4 - 2026-11-13

- *Auto-restore all*: choosing the suggested `restored` folder no longer
  fails with "path does not exist" when the folder is new. It is created
  before the folder picker opens, and removed again if you cancel or pick
  a different folder.
- *Save*: cancelling no longer leaves an empty `restored` folder behind.

## 0.5.3 - 2026-11-13

- *Update now* failed on Windows installs: the program closed and started
  again without updating. The Windows installer writes `install.json` with
  a byte-order mark, which the updater could not read, so it did not find
  uv (or that GPU support was installed) and the package install failed.
  The update helper now reads it correctly and works out uv, GPU support
  and the right Python itself, so installed 0.5.0-0.5.2 copies can update
  to this version. New installs write `install.json` without the mark.

## 0.5.2 - 2026-11-13

- The new-version window shows what changed: the release notes appear
  formatted, with working links, above *Update now*, *View on GitHub*,
  *Skip this version* and *Later*. Choosing *Update now* starts the update
  without asking a second time.
- *Help > About > Check for updates* opens the same window.
- Each GitHub release now uses that version's CHANGELOG section as its
  notes (they were only a link to the list of commits).

## 0.5.1 - 2026-11-13

- The release build on GitHub failed: `installer/build.py --out dist` gave
  Inno Setup relative paths, which it reads from the .iss file's folder.
  The output folder is now always made absolute.
- GitHub Actions updated to versions that run on Node.js 24.

## 0.5.0 - 2026-11-13

- Program icon: a cover being wiped clean, halftoned and grimy on one side
  and clean on the other. It is drawn by `tools/make_icon.py` with simpler
  versions for small sizes, and used for the window, the Windows taskbar
  (Cover Cleaner gets its own taskbar entry instead of Python's), the
  setup program, Start menu and desktop shortcuts, Add/Remove Programs,
  the Linux applications menu and the macOS app.

## 0.4.1 - 2026-11-13

Installs are self-contained and reproducible:

- Each install has its own Python 3.10.22 in its folder: never the
  system's, not on PATH or in the registry, removed when uninstalling.
  The uv version that sets it up is pinned too.
- Packages are installed from `requirements.lock` /
  `requirements-gpu.lock`: exact versions with SHA-256 checksums, the
  versions the tests ran with (`installer/lock.py` writes them).
- Only the Qt parts Cover Cleaner uses are installed (PySide6-Essentials):
  about 460 MB less than the full PySide6.
- Updates install the new release's locked packages; if a release moves
  to a different Python, a new environment is built beside the old one
  and swapped in. A failed update also puts the previous package versions
  back. The update helper runs from the base Python so the environment
  can be replaced.
- GitHub Actions runs the tests on Windows, Linux and macOS with exactly
  the shipped Python and packages, on every push; installers are only
  built when they pass.

## 0.4.0 - 2026-11-13

Installers and updates:

- Installers for Windows (setup .exe), Linux (.run) and macOS (.command),
  built by `installer/build.py` and, on a version tag, by a GitHub Actions
  workflow that publishes the release. Each sets up its own Python and
  packages and offers - on by default - the ML models and NVIDIA GPU
  support (`python -m covercleaner.setup_models` does the same later).
- *Help > About* links to the GitHub project and has *Check for updates*.
- Cover Cleaner checks for a new version when it starts (can be turned off
  in Options; a version can be skipped) and can update itself: it
  downloads the release, checks its SHA-256, installs it after closing,
  keeps the models, rolls back on failure and starts again.
- Downloads use the system certificate store (truststore), so HTTPS checks
  no longer fail at random on Windows.
- Requires OpenCV 5.

## 0.3.1 - 2026-11-13

Fixes from a code review:

- Results are never written into a scans folder: a `restored` folder
  inside it is used instead (command line too), so originals can't be
  overwritten.
- Output names are reserved for the whole batch up front: two scans with
  the same name (a.png and a.tif, or front.png from two folders) no longer
  overwrite each other in a parallel batch.
- A one-scan batch into a folder with a dot in its name works (the editor's
  target is always a folder; the command line treats `-o` as a file only
  when it ends in an image extension).
- If a batch worker crashes, the remaining scans are retried one at a time
  instead of all being reported as failed.
- The GPU copy of the model is made once before a parallel batch starts,
  not by every worker at once.
- Editor: rotating, trimming, moving corners, Detect outline and Whole
  image clear staple fills and painted repairs (they no longer land on the
  wrong spot), with a note in the status bar.
- Editor: browsing through files while one renders no longer queues
  several Auto runs that overwrite your edits.
- Editor: Detect outline uses the fallback DPI and the Auto crop rules from
  Options, and runs in the background, as does Detect screen.
- Editor: a pending Pick white/black ends when another scan is opened.
- Editor: files dropped from different drives together open correctly.
- Scan resolution is read from TIFF files and from JPEG Exif data, and
  JFIF resolutions in dots per cm are converted.
- Flat colour cleanup: much faster and lighter with many colours (22
  colours: 21 s -> 4 s on a test image); results unchanged.
- Options file: an unreadable value only resets that one option, and
  text like "false" is read correctly.
- No console window flashes when checking GPU memory; tiny images no
  longer crash edge shading.
- Removed the "Straighten curved edges" checkbox (it had no effect) and
  other unused code.

## 0.3.0 - 2026-11-13

- Drag scans and/or folders onto the editor window to open them: a
  folder lists its scans, a single scan opens with the rest of its folder,
  several files and folders are listed together.
- `python -m covercleaner gui` (and `CoverCleaner.bat`) accept several
  scans and folders, e.g. when files are dropped onto the .bat.
- Options: bit depth is greyed out while JPEG is the selected format.

## 0.2.1 - 2026-11-13

- Options: the About tab is gone (Help > About shows the same); the
  options file location is shown at the bottom of the dialog.
- Options: JPEG quality is only shown when JPEG is the selected format.

## 0.2.0 - 2026-11-13

- Options menu (Settings > Options...): default save location, format, bit
  depth, JPEG quality, file name suffix and overwrite, saving the settings
  (.json) with results, output scale; default pipeline, GPU on/off, scans
  processed at once, fallback DPI; switches for the Auto crop rules; model
  file status; credits.
- Flat colour cleanup is marked Experimental and is off by default in
  Auto; switch it on in Options ("Use flat colour cleanup when flat sections
  are detected"), or for one scan in the editor.
- Help > About shows the version and credits.
- The editor's batch runs several scans at once.
- Everything before this release: crop, deskew and rectangle fit; white
  borders, one-sided strips, items under the jacket, standard sizes;
  staples; descreen, shading, dust and glare; flat colour cleanup with
  sections, pockets and clean edges; tone and sharpening; discs levelled
  and turned upright by their text; ML inpainting (LaMa) on the GPU;
  parallel command-line batch.

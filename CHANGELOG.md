# Changelog

Version numbers follow semantic versioning: the minor number goes up for
new features, the patch number for fixes.

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

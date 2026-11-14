"""Interactive editor: Auto analysis plus manual control over every fix."""
from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDockWidget,
                               QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
                               QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget)

from .. import __version__
from .. import cli as _cli
from .. import crash as _crash
from .. import descreen as _descreen
from .. import settings as _settings
from .. import updater as _updater
from .. import geometry as _geometry
from .. import ml as _ml
from ..auto import auto_params
from ..imageio import SUPPORTED, load_image, read_dpi, save_image
from ..params import Params
from ..pipeline import Pipeline
from .options import APP_NAME, EXPERIMENTAL, AboutDialog, OptionsDialog, UpdateDialog
from .view import ImageView, mask_pixmap, to_pixmap


# --------------------------------------------------------------------------
# background jobs
# --------------------------------------------------------------------------

class _Signals(QObject):
    progress = Signal(str, float)
    done = Signal(object)
    error = Signal(str)


class Job(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = _Signals()

    def run(self):
        try:
            self.signals.done.emit(self.fn(self.signals.progress.emit))
        except Exception:
            self.signals.error.emit(traceback.format_exc())


# --------------------------------------------------------------------------
# small widgets
# --------------------------------------------------------------------------

class FloatSlider(QWidget):
    changed = Signal(float)

    def __init__(self, label: str, lo: float, hi: float, decimals: int = 2, tip: str = ""):
        super().__init__()
        self.lo, self.hi = lo, hi
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lab = QLabel(label)
        lab.setMinimumWidth(96)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep((hi - lo) / 100.0)
        self.spin.setFixedWidth(92)
        lay.addWidget(lab)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        if tip:
            self.setToolTip(tip)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _from_slider(self, v):
        val = self.lo + (self.hi - self.lo) * v / 1000.0
        self.spin.blockSignals(True)
        self.spin.setValue(val)
        self.spin.blockSignals(False)
        self.changed.emit(val)

    def _from_spin(self, val):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round((val - self.lo) / (self.hi - self.lo) * 1000)))
        self.slider.blockSignals(False)
        self.changed.emit(val)

    def set_value(self, val: float):
        for wdg in (self.slider, self.spin):
            wdg.blockSignals(True)
        self.spin.setValue(val)
        self.slider.setValue(int(round((val - self.lo) / (self.hi - self.lo) * 1000)))
        for wdg in (self.slider, self.spin):
            wdg.blockSignals(False)


# (group attr, title, tooltip, [(attr, label, lo, hi, decimals, tip)], [(attr, label)])
GROUPS = [
    ("descreen", "Descreen (remove print dots)",
     "Removes the halftone dot pattern of offset printing.",
     [("strength", "Strength", 0, 1, 2, "Blend between scan and descreened image."),
      ("smoothing", "Smoothing", 0, 1, 2, "0 = remove only the dot frequencies; higher also "
       "smooths remaining dot structure."),
      ("protect_edges", "Keep text crisp", 0, 1, 2, "Less smoothing on solid lettering and line art.")],
     []),
    ("shading", "Edge shading (paper not flat)",
     "Brightens edges that are darker because the paper lifted off the glass.",
     [("width_pct", "Band width %", 1, 15, 1, "How far from the edge the shadow reaches."),
      ("strength", "Strength", 0, 1.5, 2, "")],
     []),
    ("defects", "Dust, specks and glare",
     "Finds isolated specks and inpaints them. Paint with the Heal brush to fix anything missed.",
     [("sensitivity", "Dust sensitivity", 0, 1, 2, "Up to 0.75 only specks on flat colour are "
       "removed; above that also isolated dots inside pictures."),
      ("max_size", "Max speck size %", 0.1, 2, 2, "Largest speck, as % of the cover's short side."),
      ("glare_sensitivity", "Glare sensitivity", 0, 1, 2, ""),
      ("glare_strength", "Glare strength", 0, 1.5, 2, "")],
     [("dust_dark", "Dark specks"), ("dust_bright", "Light specks"), ("glare", "Glare")]),
    ("flat", f"Flat colour cleanup {EXPERIMENTAL}",
     "Experimental. Auto only switches it on when Options > Auto allows it. "
     "Paints areas that were one solid colour in the artwork (backgrounds, stripes, "
     "lettering) with a single even colour. Use the Protect brush to exclude a spot.",
     [("tolerance", "Tolerance", 0.01, 0.2, 3, "How uneven an area may be and still count as "
       "one flat colour. Raise it if blotchy areas are not evened out; lower it if "
       "gradients or photos get flattened."),
      ("min_area", "Min area %", 0.005, 1, 3, "Smallest area that can start a flat colour."),
      ("strength", "Strength", 0, 1, 2, ""),
      ("fill_marks", "Absorb marks %", 0, 2, 2, "Also paint over enclosed marks up to this "
       "size (% of the short side). Beware: small print inside a flat area is a 'mark' too.")],
     [("unify", "Unify colours"), ("crisp_only", "Artwork only"),
      ("pure_bw", "Pure black/white"), ("clean_edges", "Clean edges")]),
    ("denoise", "Denoise", "Edge-preserving smoothing of scanner / paper grain.",
     [("strength", "Strength", 0, 1, 2, ""), ("chroma", "Colour noise", 0, 1, 2, "")], []),
    ("tone", "Tone and colour", "Restores ink black, paper white and colour.",
     [("gamma", "Midtones", 0.5, 2, 2, ""), ("contrast", "Contrast", -1, 1, 2, ""),
      ("saturation", "Saturation", 0, 2, 2, ""), ("temperature", "Temperature", -1, 1, 2, ""),
      ("tint", "Tint", -1, 1, 2, "")], []),
    ("sharpen", "Sharpen", "Unsharp mask on brightness only.",
     [("amount", "Amount", 0, 2, 2, ""), ("radius", "Radius px", 0.3, 5, 1, "")], []),
]


# --------------------------------------------------------------------------
# main window
# --------------------------------------------------------------------------

@dataclass
class FileSession:
    """What a file in the list had when you switched away from it.

    Every visited file keeps its settings (Auto's results and your edits),
    the analysis report and the painted masks (PNG-compressed: a few KB), so
    switching back skips Auto. The most recently used files also keep the
    loaded image and the finished result, so switching back is instant."""
    params: Params
    report: str = ""
    masks: tuple | None = None          # (heal PNG, protect PNG)
    needs_render: bool = True           # no up-to-date result was kept
    source: np.ndarray | None = None    # "heavy" part, recent files only
    result: np.ndarray | None = None
    src_pixmap: object = None
    res_pixmap: object = None

    def heavy_bytes(self) -> int:
        return sum(a.nbytes for a in (self.source, self.result) if a is not None)

    def drop_heavy(self) -> None:
        self.source = self.result = self.src_pixmap = self.res_pixmap = None
        self.needs_render = True


KEEP_HEAVY = 4                          # at most this many recent files in full
KEEP_FREE = 2.5e9                       # ...while leaving this much memory free


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = _settings.load()
        _ml.set_use_gpu(self.settings.use_gpu)
        self.setWindowTitle(self._title())
        self.resize(1500, 950)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)          # the pipeline is not re-entrant
        self.path: str | None = None
        self.source: np.ndarray | None = None
        self.pipe: Pipeline | None = None
        self.params = Params()
        self.result: np.ndarray | None = None
        self.busy = False
        self.dirty = False
        self.view_mode = "result"
        self.pick_target: str | None = None
        self._widgets: dict[tuple[str, str], object] = {}
        self._jobs: list[Job] = []
        self._auto_pending = False
        self.net_pool = QThreadPool(self)          # update checks/downloads, apart from renders
        self.net_pool.setMaxThreadCount(1)
        self._auto_path: str | None = None     # file waiting for its Auto run
        self._sessions: dict[str, FileSession] = {}   # files switched away from

        self.view = ImageView()
        self.setCentralWidget(self.view)
        self.view.cornerMoved.connect(self._corner_moved)
        self.view.cornerReleased.connect(self.request_render)
        self.view.strokeFinished.connect(self._stroke_finished)
        self.view.picked.connect(self._picked)

        self.setAcceptDrops(True)                 # drop scans or folders anywhere
        self.view.setAcceptDrops(False)
        self.view.viewport().setAcceptDrops(False)
        self._build_toolbar()
        self._build_menus()
        self._build_files_dock()
        self._build_controls_dock()
        self._build_report_dock()
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setRange(0, 100)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._start_render)
        self._set_enabled(False)
        self.statusBar().showMessage("Open a scan or a folder of scans to begin.")
        if self.settings.check_updates:
            QTimer.singleShot(2000, self._startup_update_check)

    # ---- construction ----------------------------------------------------
    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)

        def act(text, slot, shortcut=None, tip=None):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.setToolTip((tip or text) + (f"  ({shortcut})" if shortcut else ""))
            tb.addAction(a)
            return a

        self.act_open = act("Open…", self.open_dialog, "Ctrl+O")
        self.act_open_folder = act("Open folder…", self.open_folder_dialog, "Ctrl+Shift+O")
        tb.addSeparator()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Traditional", "ML"])
        self.mode_combo.setToolTip(
            "Traditional: classical image processing only.\n"
            "ML: also uses a neural network (LaMa) to repair dust and Heal-brush strokes "
            "inside textured areas - better fills, but slower."
            + ("" if _ml.available() else "\n\nML model file not found:\n" + _ml.model_path()))
        if not _ml.available():
            self.mode_combo.model().item(1).setEnabled(False)
        self.mode_combo.setCurrentIndex(1 if self.settings.mode(_ml.available()) == "ml" else 0)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        tb.addWidget(QLabel(" Pipeline: "))
        tb.addWidget(self.mode_combo)
        self.act_auto = act("Auto", self.run_auto, "Ctrl+A",
                            "Analyse the scan and choose the best fixes")
        self.act_reset = act("Reset", self.reset_params, None, "Switch every fix off")
        tb.addSeparator()
        self.act_save = act("Save result…", self.save_dialog, "Ctrl+S")
        self.act_batch = act("Auto-restore all…", self.batch_dialog, None,
                             "Run Auto on every file in the list and save the results")
        self.act_load_settings = act("Load settings…", self.load_settings)
        self.act_save_settings = act("Save settings…", self.save_settings)
        tb.addSeparator()
        self.btn_original = QToolButton(); self.btn_original.setText("Original + outline")
        self.btn_result = QToolButton(); self.btn_result.setText("Result")
        grp = QButtonGroup(self)
        for b, mode in ((self.btn_original, "original"), (self.btn_result, "result")):
            b.setCheckable(True)
            grp.addButton(b)
            tb.addWidget(b)
            b.clicked.connect(lambda _=False, m=mode: self.set_view_mode(m))
        self.btn_result.setChecked(True)
        self.btn_compare = QToolButton(); self.btn_compare.setText("Hold: before")
        self.btn_compare.setToolTip("Hold to see the cropped scan before the fixes (Space)")
        self.btn_compare.pressed.connect(lambda: self._show_before(True))
        self.btn_compare.released.connect(lambda: self._show_before(False))
        tb.addWidget(self.btn_compare)
        act("Fit", self.view.fit, "F")
        act("100%", self.view.actual_size, "1")

    def _build_files_dock(self):
        dock = QDockWidget("Scans", self)
        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self._file_selected)
        dock.setWidget(self.file_list)
        dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.files_dock = dock
        self.folder = ""

    def _build_report_dock(self):
        dock = QDockWidget("Auto analysis report", self)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setAcceptDrops(False)        # dropped scans go to the window
        self.report.setMaximumHeight(170)
        dock.setWidget(self.report)
        dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

    def _build_controls_dock(self):
        panel = QWidget()
        col = QVBoxLayout(panel)
        col.setSpacing(8)

        # --- geometry
        box = QGroupBox("Crop, deskew and flatten")
        box.setCheckable(True)
        box.toggled.connect(lambda on: self._set("geometry", "enabled", on))
        self._widgets[("geometry", "enabled")] = box
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        b = QPushButton("Detect outline"); b.clicked.connect(self.detect_outline)
        b.setToolTip("Find the cover edges again")
        row.addWidget(b)
        b = QPushButton("Whole image"); b.clicked.connect(self.outline_full)
        b.setToolTip("Use the whole scan; then drag the corners in 'Original + outline'")
        row.addWidget(b)
        lay.addLayout(row)
        grid = QGridLayout()
        self.trim_spins = []
        for i, name in enumerate(("Trim left", "top", "right", "bottom")):
            sp = QSpinBox(); sp.setRange(0, 2000); sp.setSuffix(" px")
            sp.valueChanged.connect(lambda v, k=i: self._set_trim(k, v))
            grid.addWidget(QLabel(name), i // 2, (i % 2) * 2)
            grid.addWidget(sp, i // 2, (i % 2) * 2 + 1)
            self.trim_spins.append(sp)
        lay.addLayout(grid)
        row = QHBoxLayout()
        for text, step in (("⟲ Rotate left", -1), ("Rotate right ⟳", 1)):
            b = QPushButton(text); b.clicked.connect(lambda _=False, s=step: self._rotate(s))
            row.addWidget(b)
        lay.addLayout(row)
        sl = FloatSlider("Fine rotate °", -45, 45, 1,
                         "Rotate by a few degrees, e.g. to level the text on a disc. "
                         "Positive = counter-clockwise.")
        sl.changed.connect(lambda v: self._set("geometry", "angle", round(float(v), 1)))
        self._widgets[("geometry", "angle")] = sl
        lay.addWidget(sl)
        col.addWidget(box)

        # --- generic groups
        for attr, title, tip, sliders, checks in GROUPS:
            box = QGroupBox(title)
            box.setCheckable(True)
            box.setToolTip(tip)
            box.toggled.connect(lambda on, a=attr: self._group_toggled(a, on))
            self._widgets[(attr, "enabled")] = box
            lay = QVBoxLayout(box)
            if checks:
                row = QHBoxLayout()
                for name, label in checks:
                    cb = QCheckBox(label)
                    cb.toggled.connect(lambda on, a=attr, n=name: self._set(a, n, on))
                    self._widgets[(attr, name)] = cb
                    row.addWidget(cb)
                lay.addLayout(row)
            if attr == "flat":
                self.show_flat = QCheckBox("Show flat areas (green)")
                self.show_flat.toggled.connect(self._update_overlay)
                lay.addWidget(self.show_flat)
            if attr == "shading":
                row = QHBoxLayout()
                self.side_checks = []
                for i, label in enumerate(("Left", "Top", "Right", "Bottom")):
                    cb = QCheckBox(label)
                    cb.toggled.connect(lambda on, k=i: self._set_side(k, on))
                    self.side_checks.append(cb)
                    row.addWidget(cb)
                lay.addLayout(row)
            if attr == "tone":
                self.black_slider = FloatSlider("Black point", 0, 0.5, 3,
                                                "Scan level that becomes pure black.")
                self.white_slider = FloatSlider("White point", 0.5, 1, 3,
                                                "Scan level that becomes pure white.")
                self.black_slider.changed.connect(lambda v: self._set_level("black", v))
                self.white_slider.changed.connect(lambda v: self._set_level("white", v))
                lay.addWidget(self.black_slider)
                lay.addWidget(self.white_slider)
                row = QHBoxLayout()
                for text, target in (("Pick black", "black"), ("Pick white / grey", "white")):
                    b = QPushButton(text)
                    b.setToolTip("Then click a spot in the result that should be neutral "
                                 + ("black" if target == "black" else "white (removes colour cast)"))
                    b.clicked.connect(lambda _=False, t=target: self._start_pick(t))
                    row.addWidget(b)
                lay.addLayout(row)
            for name, label, lo, hi, dec, stip in sliders:
                sl = FloatSlider(label, lo, hi, dec, stip)
                sl.changed.connect(lambda v, a=attr, n=name: self._set(a, n, float(v)))
                self._widgets[(attr, name)] = sl
                lay.addWidget(sl)
            if attr == "descreen":
                self.screen_label = QLabel("No screen analysed yet")
                b = QPushButton("Detect screen"); b.clicked.connect(self.detect_screen)
                row = QHBoxLayout(); row.addWidget(self.screen_label, 1); row.addWidget(b)
                lay.addLayout(row)
            if attr == "defects":
                row = QHBoxLayout()
                self.brush_combo = QComboBox()
                self.brush_combo.addItems(["Brush off (pan)", "Heal brush", "Protect brush",
                                           "Eraser"])
                self.brush_combo.setToolTip("Heal: repair what you paint.  Protect: never "
                                            "repair here.  Eraser: remove painted marks.")
                self.brush_combo.currentIndexChanged.connect(self._brush_changed)
                self.brush_size = QSpinBox(); self.brush_size.setRange(2, 400)
                self.brush_size.setValue(24); self.brush_size.setSuffix(" px")
                self.brush_size.valueChanged.connect(self._brush_changed)
                row.addWidget(self.brush_combo, 1); row.addWidget(self.brush_size)
                lay.addLayout(row)
                row = QHBoxLayout()
                self.show_mask = QCheckBox("Show repairs")
                self.show_mask.setToolTip("Red = repaired, blue = protected")
                self.show_mask.toggled.connect(self._update_overlay)
                b = QPushButton("Clear painted"); b.clicked.connect(self._clear_masks)
                row.addWidget(self.show_mask); row.addWidget(b)
                lay.addLayout(row)
            col.addWidget(box)

        # --- output
        box = QGroupBox("Output")
        lay = QVBoxLayout(box)
        sl = FloatSlider("Resize", 0.25, 2.0, 2, "Final resampling. 0.5 turns a 600 dpi scan "
                         "into a 300 dpi image.")
        sl.changed.connect(lambda v: self._set("output", "scale", float(v)))
        self._widgets[("output", "scale")] = sl
        lay.addWidget(sl)
        row = QHBoxLayout()
        row.addWidget(QLabel("Bit depth"))
        self.bits = QComboBox(); self.bits.addItems(["8", "16"])
        self.bits.setCurrentIndex(1 if self.settings.bits == 16 else 0)
        row.addWidget(self.bits); row.addStretch(1)
        lay.addLayout(row)
        col.addWidget(box)
        col.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(440)
        dock = QDockWidget("Fixes", self)
        dock.setWidget(scroll)
        dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.controls = panel

    # ---- parameter plumbing ----------------------------------------------
    def _set(self, group: str, name: str, value):
        if getattr(getattr(self.params, group), name) == value:
            return
        setattr(getattr(self.params, group), name, value)
        if group == "geometry" and name in ("enabled", "angle"):
            self._geometry_edited()
        self.request_render()

    def _geometry_edited(self):
        """The crop or rotation changed: staple fills and painted repairs are
        in the old crop's pixel positions and would land on the wrong spot."""
        had = bool(self.params.defects.patches) or (
            self.pipe is not None and self.pipe.heal_mask is not None
            and (self.pipe.heal_mask.any() or self.pipe.protect_mask.any()))
        self.params.defects.patches = []
        if self.pipe:
            self.pipe.clear_masks()
        if had:
            self.statusBar().showMessage("The crop changed: staple fills and painted repairs "
                                         "were cleared (run Auto or paint them again).")

    def _mode(self) -> str:
        return "ml" if self.mode_combo.currentIndex() == 1 else "traditional"

    def _mode_changed(self, *_):
        self.params.mode = self._mode()
        self.request_render()

    def _group_toggled(self, group: str, on: bool):
        if group == "descreen" and on and not self.params.descreen.peaks:
            self.params.descreen.enabled = True
            self.detect_screen()
            return
        self._set(group, "enabled", on)

    def _set_trim(self, k: int, v: int):
        if self.params.geometry.trim[k] != v:
            self.params.geometry.trim[k] = int(v)
            self._geometry_edited()
            self.request_render()

    def _set_side(self, k: int, on: bool):
        if self.params.shading.sides[k] != on:
            self.params.shading.sides[k] = bool(on)
            self.request_render()

    def _set_level(self, which: str, value: float):
        cur = getattr(self.params.tone, which)
        shift = value - float(np.mean(cur))
        setattr(self.params.tone, which, [float(np.clip(c + shift, 0, 1)) for c in cur])
        self.request_render()

    def _rotate(self, step: int):
        self.params.geometry.rotate90 = (self.params.geometry.rotate90 + step) % 4
        self._geometry_edited()
        self.request_render()

    def sync_controls(self):
        """Push self.params into every widget without triggering renders."""
        p = self.params
        for (group, name), wdg in self._widgets.items():
            val = getattr(getattr(p, group), name)
            wdg.blockSignals(True)
            if isinstance(wdg, FloatSlider):
                wdg.set_value(float(val))
            else:
                wdg.setChecked(bool(val))
            wdg.blockSignals(False)
        for sp, v in zip(self.trim_spins, p.geometry.trim):
            sp.blockSignals(True); sp.setValue(int(v)); sp.blockSignals(False)
        for cb, v in zip(self.side_checks, p.shading.sides):
            cb.blockSignals(True); cb.setChecked(bool(v)); cb.blockSignals(False)
        self.black_slider.set_value(float(np.mean(p.tone.black)))
        self.white_slider.set_value(float(np.mean(p.tone.white)))
        d = p.descreen
        self.screen_label.setText(f"Dot pitch {d.period:.2f} px, {len(d.peaks)} frequencies"
                                  if d.peaks else "No halftone screen found")

    def _set_enabled(self, on: bool):
        self.controls.setEnabled(on)
        for a in (self.act_auto, self.act_reset, self.act_save, self.act_save_settings):
            a.setEnabled(on)

    # ---- files -----------------------------------------------------------
    def open_dialog(self):
        exts = " ".join("*" + e for e in SUPPORTED)
        paths, _ = QFileDialog.getOpenFileNames(self, "Open scans", self.folder,
                                                f"Images ({exts})")
        if paths:
            self.open_paths(paths)

    def open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Open folder of scans", self.folder)
        if folder:
            self.set_folder(folder)

    def set_folder(self, folder: str, select: str | None = None):
        names = sorted(n for n in os.listdir(folder) if n.lower().endswith(SUPPORTED))
        self.set_files([os.path.join(folder, n) for n in names],
                       os.path.join(folder, select) if select else None, folder)

    def set_files(self, paths: list[str], select: str | None = None,
                  folder: str | None = None):
        """Show `paths` in the scan list and open `select` (or the first)."""
        paths = [os.path.abspath(p) for p in paths]
        dirs = {os.path.dirname(p) for p in paths}
        if folder:
            self.folder = folder
        elif len(dirs) == 1:
            self.folder = dirs.pop()
        else:
            try:
                self.folder = os.path.commonpath(paths) if paths else self.folder
            except ValueError:              # files on different drives
                self.folder = os.path.dirname(paths[0])
        keep = set(paths)
        for p in [p for p in self._sessions if p not in keep]:
            del self._sessions[p]
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for p in paths:
            if len(dirs) <= 1:
                label = os.path.basename(p)
            else:
                try:
                    label = os.path.relpath(p, self.folder)
                except ValueError:          # another drive
                    label = p
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, p)
            item.setToolTip(p)
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)
        if paths:
            sel = os.path.abspath(select) if select else None
            idx = paths.index(sel) if sel in paths else 0
            self.file_list.setCurrentRow(idx)

    def _file_selected(self, item, _previous=None):
        if item is not None:
            self.load(item.data(Qt.UserRole))

    def _list_paths(self) -> list[str]:
        return [self.file_list.item(i).data(Qt.UserRole) for i in range(self.file_list.count())]

    # ---- drag and drop -----------------------------------------------------
    @staticmethod
    def _dropped_paths(event) -> list[str]:
        md = event.mimeData()
        if not md.hasUrls():
            return []
        return [u.toLocalFile() for u in md.urls() if u.isLocalFile() and u.toLocalFile()]

    def dragEnterEvent(self, event):
        if any(os.path.isdir(p) or p.lower().endswith(SUPPORTED)
               for p in self._dropped_paths(event)):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        paths = self._dropped_paths(event)
        if not paths:
            return
        event.acceptProposedAction()
        self.open_paths(paths)

    def open_paths(self, paths: list[str]):
        """Open what was dropped, chosen or given on the command line: a folder
        shows its scans; files are listed on their own (just the one, for a
        single file); several files and/or folders are listed together."""
        paths = [os.path.abspath(p) for p in paths if os.path.exists(p)]
        if not paths:
            return
        if len(paths) == 1 and os.path.isdir(paths[0]):
            self.set_folder(paths[0])
            return
        files: list[str] = []
        for p in paths:
            if os.path.isdir(p):
                files += sorted(os.path.join(p, n) for n in os.listdir(p)
                                if n.lower().endswith(SUPPORTED))
            elif p.lower().endswith(SUPPORTED):
                files.append(p)
        seen = set()
        files = [f for f in files if not (f in seen or seen.add(f))]
        if files:
            self.set_files(files)
        else:
            self.statusBar().showMessage("No scans (images) among the dropped items.")

    # ---- per-file sessions --------------------------------------------------
    def _stash_current(self) -> None:
        """Remember the open file's settings (and, memory allowing, its image
        and result) before another file is opened."""
        path = self.path
        if path is None or self.source is None or self._auto_path == path:
            return                          # nothing yet (its Auto has not finished)
        s = FileSession(self.params.copy(), self.report.toPlainText())
        p = self.pipe
        if p is not None and p.heal_mask is not None and (p.heal_mask.any()
                                                          or p.protect_mask.any()):
            s.masks = (cv2.imencode(".png", p.heal_mask)[1].tobytes(),
                       cv2.imencode(".png", p.protect_mask)[1].tobytes())
        s.needs_render = self.result is None or self.dirty or self.busy
        s.source = self.source
        s.src_pixmap = self._src_pixmap
        if not s.needs_render:
            s.result, s.res_pixmap = self.result, self._res_pixmap
        self._sessions.pop(path, None)
        self._sessions[path] = s            # most recent last
        self._trim_sessions()

    def _trim_sessions(self) -> None:
        """Keep the images of only the most recent files, within memory."""
        heavy = [k for k, s in self._sessions.items() if s.source is not None]
        avail = _cli.available_memory()
        while heavy and (len(heavy) > KEEP_HEAVY
                         or (avail is not None and avail < KEEP_FREE)):
            s = self._sessions[heavy.pop(0)]
            if avail is not None:
                avail += s.heavy_bytes()
            s.drop_heavy()

    def _restore(self, s: FileSession) -> None:
        """Bring back a remembered file's state (the file itself is loaded)."""
        self.params = s.params.copy()
        self.report.setPlainText(s.report)
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(1 if self.params.mode == "ml" else 0)
        self.mode_combo.blockSignals(False)
        if s.masks is not None:
            heal = cv2.imdecode(np.frombuffer(s.masks[0], np.uint8), cv2.IMREAD_UNCHANGED)
            protect = cv2.imdecode(np.frombuffer(s.masks[1], np.uint8), cv2.IMREAD_UNCHANGED)
            self.pipe.heal_mask, self.pipe.protect_mask = heal, protect
            self.pipe.masks_changed()
        self.sync_controls()
        name = os.path.basename(self.path)
        if not s.needs_render and s.result is not None:
            self.result, self._res_pixmap = s.result, s.res_pixmap
            self.set_view_mode("result", keep_view=False)
            self.statusBar().showMessage(f"{name}: your settings and result from earlier "
                                         "(Auto re-analyses it)")
        else:
            self.set_view_mode("result", keep_view=False)
            self.request_render()
            self.statusBar().showMessage(f"{name}: your settings from earlier "
                                         "(Auto re-analyses it)")

    def _file_done_elsewhere(self, path: str, **kw) -> None:
        """An Auto run or render for a file you have since left finished."""
        if not any(p == path for p in self._list_paths()):
            return
        s = self._sessions.get(path)
        if "params" in kw:                  # Auto: its settings, to render later
            if s is None:
                self._sessions[path] = FileSession(kw["params"], kw["report"])
        elif s is not None and "result" in kw:  # a render of exactly what is kept
            if kw["rendered"].to_json() == s.params.to_json() and s.source is not None:
                s.result, s.res_pixmap = kw["result"], to_pixmap(kw["result"])
                s.needs_render = False

    def load(self, path: str, auto: bool = True):
        if path != self.path:
            self._stash_current()
        s = self._sessions.pop(path, None) if auto else None
        try:
            self.source = s.source if (s is not None and s.source is not None) \
                else load_image(path)
        except Exception as exc:
            QMessageBox.warning(self, "Cover Cleaner", str(exc))
            return
        self.path = path
        mem = _cli.available_memory()
        _crash.log.info("open %s (%d x %d)%s, free memory %s", path, self.source.shape[1],
                        self.source.shape[0], ", remembered" if s is not None else "",
                        f"{mem / 1e9:.1f} GB" if mem else "?")
        self.pipe = Pipeline(self.source)
        self.params = Params()
        self.params.mode = self._mode()
        self.result = None
        self.report.setPlainText("")
        self.setWindowTitle(self._title(os.path.basename(path)))
        self._src_pixmap = (s.src_pixmap if (s is not None and s.src_pixmap is not None)
                            else to_pixmap(self.source))
        self._set_enabled(True)
        self.sync_controls()
        self.view.set_overlay(None)
        self.pick_target = None             # a pending "Pick white/black" ends here
        self.brush_combo.blockSignals(True)
        self.brush_combo.setCurrentIndex(0)
        self.brush_combo.blockSignals(False)
        self.view.set_mode("pan")
        self.set_view_mode("original", keep_view=False)
        self._auto_path = None              # a pending Auto for the old file is void
        if s is not None:
            self._restore(s)
        elif auto:
            self.run_auto()

    # ---- jobs ------------------------------------------------------------
    def _run_job(self, fn, done, label: str):
        job = Job(fn)
        self._jobs.append(job)
        job.setAutoDelete(False)

        def finish(*_):
            if job in self._jobs:
                self._jobs.remove(job)

        job.signals.progress.connect(self._progress)
        job.signals.done.connect(lambda res: (finish(), done(res)))
        job.signals.error.connect(lambda msg: (finish(), self._job_error(msg)))
        self.progress.show()
        self.progress.setValue(0)
        self.statusBar().showMessage(label)
        self.pool.start(job)

    def _progress(self, name: str, frac: float):
        self.progress.setValue(int(frac * 100))
        if name != "done":
            self.statusBar().showMessage(name[:1].upper() + name[1:] + "…")

    def _job_error(self, msg: str):
        self.busy = False
        self.progress.hide()
        self.statusBar().showMessage("Error")
        _crash.log.error("background job failed:\n%s", msg)
        try:
            report = _crash.write_report("Cover Cleaner error", "A background job failed:\n\n" + msg)
        except OSError:
            report = None
        self.show_crash_report(report, detail=msg.strip().splitlines()[-1] if msg.strip() else "")

    # ---- crash reports ---------------------------------------------------------
    def show_crash_report(self, path: str | None, after_crash: bool = False, detail: str = ""):
        """Tell the user about an error or an earlier crash, with its report."""
        now = time.time()
        if not after_crash and now - getattr(self, "_last_error_box", 0) < 5:
            return                              # an error repeating: once is enough
        self._last_error_box = now
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        box = QMessageBox(self)
        box.setWindowTitle(APP_NAME)
        box.setIcon(QMessageBox.Warning if after_crash else QMessageBox.Critical)
        if after_crash:
            box.setText("<b>Cover Cleaner closed unexpectedly last time.</b>")
        else:
            box.setText("<b>Something went wrong.</b>" + (f"<br>{detail}" if detail else ""))
        if path:
            box.setInformativeText(
                "A report was saved. If you report the problem on GitHub, please attach it "
                "and say what you were doing. It contains the names of the scans you had "
                "open, and nothing is sent automatically.")
        b_open = box.addButton("Open report", QMessageBox.AcceptRole) if path else None
        b_folder = box.addButton("Open folder", QMessageBox.ActionRole)
        b_issue = box.addButton("Report on GitHub", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Close)
        box.exec()
        clicked = box.clickedButton()
        if b_open is not None and clicked is b_open:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        elif clicked is b_folder:
            self.open_logs_folder()
        elif clicked is b_issue:
            self.report_problem()

    def report_problem(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl(_crash.ISSUES_URL))

    def open_logs_folder(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        d = _crash.log_dir()
        os.makedirs(d, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    def run_auto(self):
        if self.source is None:
            return
        self._auto_path = self.path         # the file this Auto run is for
        if self.busy:                       # e.g. the previous file is still rendering
            if not self._auto_pending:      # one retry, for whatever file is open then
                self._auto_pending = True
                QTimer.singleShot(300, self._retry_auto)
            return
        self.busy = True
        src, pipe, path, mode = self.source, self.pipe, self.path, self._mode()
        _crash.log.info("Auto (%s): %s", mode, os.path.basename(path))
        st = self.settings

        def work(progress):
            return path, auto_params(src, progress, pipe, mode, st.dpi_for(read_dpi(path)),
                                     flat_cleaner=st.flat_cleaner, rules=st.rules,
                                     scale=st.scale)

        def done(res):
            self.busy = False
            rpath, (params, report) = res
            _crash.log.info("Auto done: %s", os.path.basename(rpath))
            if rpath != self.path:          # another file was opened meanwhile
                self._file_done_elsewhere(rpath, params=params, report="\n".join(report))
                return
            self._auto_path = None
            self.params = params
            self.report.setPlainText("\n".join(report))
            self.sync_controls()
            self.set_view_mode("result")
            self._refresh_outline()
            self.request_render()

        self._run_job(work, done, "Analysing…")

    def _retry_auto(self):
        self._auto_pending = False
        if self._auto_path is not None and self._auto_path == self.path:
            self.run_auto()

    def request_render(self):
        if self.pipe is None:
            return
        self.dirty = True
        self.timer.start(250)

    def _start_render(self):
        if self.pipe is None or not self.dirty:
            return
        if self.busy:
            self.timer.start(250)
            return
        self.busy = True
        self.dirty = False
        pipe, params, path = self.pipe, self.params.copy(), self.path
        t0 = time.time()
        _crash.log.info("render (%s): %s", params.mode, os.path.basename(path))

        def work(progress):
            return path, pipe.run(params, progress=progress)

        def done(res):
            self.busy = False
            rpath, img = res
            self.progress.hide()
            _crash.log.info("render done in %.1f s: %s", time.time() - t0, os.path.basename(rpath))
            if rpath != self.path:
                self._file_done_elsewhere(rpath, result=img, rendered=params)
                return
            self.result = img
            self._res_pixmap = to_pixmap(img)
            if self.view_mode == "result":
                self.view.set_image(self._res_pixmap)
                self._update_overlay()
            self.statusBar().showMessage(f"Result {img.shape[1]} x {img.shape[0]} px")
            if self.dirty:
                self.timer.start(10)

        self._run_job(work, done, "Processing…")

    # ---- view ------------------------------------------------------------
    def set_view_mode(self, mode: str, keep_view: bool = True):
        self.view_mode = mode
        (self.btn_original if mode == "original" else self.btn_result).setChecked(True)
        if self.source is None:
            return
        if mode == "original":
            self.view.set_overlay(None)
            self.view.set_image(self._src_pixmap, keep_view)
            self._refresh_outline()
            self.brush_combo.setCurrentIndex(0)
        else:
            self.view.set_outline(None, None)
            if self.result is not None:
                self.view.set_image(self._res_pixmap, keep_view)
                self._update_overlay()

    def _corners(self) -> np.ndarray:
        c = self.params.geometry.corners
        return _geometry.full_corners(self.source.shape) if c is None else np.asarray(c, float)

    def _bow(self):
        g = self.params.geometry
        return g.bow if (g.use_bow and g.bow) else None

    def _refresh_outline(self):
        if self.view_mode != "original" or self.source is None:
            return
        c = self._corners()
        self.view.set_outline(_geometry.outline_points(c, self._bow()), c)

    def _corner_moved(self, index: int, x: float, y: float):
        c = self._corners()
        c[index] = (x, y)
        self.params.geometry.corners = [[float(a), float(b)] for a, b in c]
        self.params.geometry.disc = False   # hand-placed corners: plain crop
        self.view.update_outline_only(_geometry.outline_points(c, self._bow()))
        self._geometry_edited()             # painted marks no longer line up
        self.dirty = True                   # render when the handle is released

    def detect_outline(self):
        if self.source is None:
            return
        if self.busy:
            self.statusBar().showMessage("Still processing - try again in a moment.")
            return
        self.busy = True
        src, path, st = self.source, self.path, self.settings

        def work(progress):
            progress("Finding cover outline", 0.1)
            return path, _geometry.detect_cover(src, dpi=st.dpi_for(read_dpi(path)),
                                                rules=st.rules)

        def done(res):
            self.busy = False
            self.progress.hide()
            rpath, det = res
            if rpath != self.path:
                return
            g = self.params.geometry
            if det.has_background:
                g.corners = [[float(x), float(y)] for x, y in det.corners]
                g.bow = [[float(v) for v in row] for row in det.bow]
                g.trim = [int(t) for t in det.trim]
                g.disc = det.disc is not None
                self.statusBar().showMessage("Outline detected.")
            else:
                g.corners = None
                self.statusBar().showMessage("No scanner background found around the cover.")
            g.enabled = True
            self._geometry_edited()
            self.sync_controls()
            self._refresh_outline()
            self.request_render()

        self._run_job(work, done, "Finding cover outline…")

    def outline_full(self):
        g = self.params.geometry
        g.corners = [[float(x), float(y)] for x, y in _geometry.full_corners(self.source.shape)]
        g.bow = None
        g.trim = [0, 0, 0, 0]
        g.disc = False
        self._geometry_edited()
        self.sync_controls()
        self.set_view_mode("original")
        self.request_render()

    def detect_screen(self):
        if self.pipe is None:
            return
        if self.busy:
            QTimer.singleShot(300, self.detect_screen)
            return
        self.busy = True
        pipe, params, path = self.pipe, self.params.copy(), self.path

        def work(progress):
            progress("Analysing print screen", 0.2)
            return path, _descreen.analyze_screen(pipe.run(params, upto="geometry"))

        def done(res):
            self.busy = False
            self.progress.hide()
            rpath, scr = res
            if rpath != self.path:
                return
            d = self.params.descreen
            if scr.found:
                d.peaks, d.period, d.enabled = scr.peaks, scr.period, True
            else:
                d.peaks, d.period, d.enabled = [], 0.0, False
                self.statusBar().showMessage("No halftone screen found in this scan.")
            self.sync_controls()
            self.request_render()

        self._run_job(work, done, "Analysing print screen…")

    def _show_before(self, on: bool):
        if self.view_mode != "result" or self.result is None or self.pipe is None:
            return
        if on:
            before = self.pipe.stage_output("geometry")
            if before is None:
                return
            if before.shape[:2] != self.result.shape[:2]:
                before = cv2.resize(before, self.result.shape[1::-1], interpolation=cv2.INTER_AREA)
            self.view.set_overlay(None)
            self.view.set_image(to_pixmap(before))
        else:
            self.view.set_image(self._res_pixmap)
            self._update_overlay()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._show_before(True)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._show_before(False)
        else:
            super().keyReleaseEvent(event)

    # ---- masks / brushes -------------------------------------------------
    def _update_overlay(self, *_):
        if self.view_mode != "result" or self.pipe is None or self.result is None:
            return
        brushing = self.brush_combo.currentIndex() > 0
        layers = []
        if self.show_flat.isChecked() and self.pipe.flat_mask is not None:
            layers.append((self.pipe.flat_mask, (40, 220, 90)))
        if (self.show_mask.isChecked() or brushing) and self.pipe.defect_mask is not None:
            layers.append((self.pipe.defect_mask, (255, 40, 40)))
            layers.append((self.pipe.protect_mask, (60, 120, 255)))
        if not layers:
            self.view.set_overlay(None)
            return
        self.view.set_overlay(mask_pixmap(layers, 120), self.params.output.scale)

    def _brush_changed(self, *_):
        idx = self.brush_combo.currentIndex()
        if idx > 0 and self.view_mode != "result":
            self.set_view_mode("result")
        self.view.brush_size = self.brush_size.value()
        self.view.brush_color = [QColor(0, 0, 0, 0), QColor(255, 60, 60, 150),
                                 QColor(60, 120, 255, 150), QColor(255, 255, 255, 150)][idx]
        self.pick_target = None
        self.view.set_mode("brush" if idx > 0 else "pan")
        self._update_overlay()

    def _stroke_finished(self, pts):
        if self.pipe is None or self.pipe.heal_mask is None:
            return
        s = 1.0 / max(self.params.output.scale, 1e-6)
        poly = np.array([[x * s, y * s] for x, y in pts], np.float32).round().astype(np.int32)
        thick = max(1, int(round(self.brush_size.value() * s)))
        stroke = np.zeros_like(self.pipe.heal_mask)
        cv2.polylines(stroke, [poly], False, 255, thick, cv2.LINE_8)
        for x, y in (poly[0], poly[-1]):
            cv2.circle(stroke, (int(x), int(y)), thick // 2, 255, -1)
        sel = stroke > 0
        idx = self.brush_combo.currentIndex()
        if idx == 1:
            self.pipe.heal_mask[sel] = 255; self.pipe.protect_mask[sel] = 0
        elif idx == 2:
            self.pipe.protect_mask[sel] = 255; self.pipe.heal_mask[sel] = 0
        else:
            self.pipe.heal_mask[sel] = 0; self.pipe.protect_mask[sel] = 0
        self.pipe.masks_changed()
        self.request_render()

    def _clear_masks(self):
        if self.pipe:
            self.pipe.clear_masks()
            self.request_render()

    def _start_pick(self, target: str):
        if self.result is None:
            return
        self.brush_combo.setCurrentIndex(0)
        self.set_view_mode("result")
        self.pick_target = target
        self.view.set_mode("pick")
        self.statusBar().showMessage(f"Click a spot that should be neutral {target}.")

    def _picked(self, x: float, y: float):
        target, self.pick_target = self.pick_target, None
        self.view.set_mode("pan")
        base = self.pipe.stage_output("denoise") if self.pipe else None
        if target is None or base is None:
            return
        s = 1.0 / max(self.params.output.scale, 1e-6)
        cx, cy = int(x * s), int(y * s)
        h, w = base.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return
        patch = base[max(0, cy - 4): cy + 5, max(0, cx - 4): cx + 5].reshape(-1, 3)
        val = [float(v) for v in np.median(patch, axis=0)]
        t = self.params.tone
        t.enabled = True
        if target == "black":
            t.black = val
        else:
            # Keep overall brightness when the spot is grey rather than white.
            peak = max(val)
            scale = 1.0 if peak > 0.8 else max(float(np.mean(t.white)), peak) / max(peak, 1e-3)
            t.white = [float(min(1.0, v * scale)) for v in val]
        self.sync_controls()
        self.request_render()

    # ---- settings / saving -----------------------------------------------
    def reset_params(self):
        self.params = Params()
        self.params.mode = self._mode()
        if self.pipe:
            self.pipe.clear_masks()
        self.report.setPlainText("All fixes switched off.")
        self.sync_controls()
        self._refresh_outline()
        self.request_render()

    def save_settings(self):
        base = os.path.splitext(self.path)[0] + ".json" if self.path else ""
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", base, "Settings (*.json)")
        if path:
            with open(path, "w", encoding="utf8") as fh:
                fh.write(self.params.to_json())

    def load_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", self.folder,
                                              "Settings (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf8") as fh:
                self.params = Params.from_json(fh.read())
            if self.params.mode == "ml" and not _ml.available():
                self.params.mode = "traditional"
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(1 if self.params.mode == "ml" else 0)
            self.mode_combo.blockSignals(False)
        except Exception as exc:
            QMessageBox.warning(self, "Cover Cleaner", f"Could not read settings:\n{exc}")
            return
        self.sync_controls()
        self._refresh_outline()
        self.request_render()

    @staticmethod
    def _prepare_dir(path: str) -> bool:
        """Create a default output folder before a file dialog opens on it
        (Windows' folder picker rejects a folder that does not exist yet).
        True if it was created here."""
        if os.path.isdir(path):
            return False
        try:
            os.makedirs(path)
            return True
        except OSError:
            return False

    @staticmethod
    def _drop_if_unused(path: str, created: bool, chosen: str) -> None:
        """Remove a folder _prepare_dir made if the user did not use it."""
        if not created:
            return
        if chosen and os.path.normcase(os.path.abspath(chosen)).startswith(
                os.path.normcase(os.path.abspath(path))):
            return
        try:
            os.rmdir(path)                        # only if still empty
        except OSError:
            pass

    def save_dialog(self):
        if self.result is None or self.busy or self.dirty:
            self.statusBar().showMessage("Still processing - try again in a moment.")
            return
        st = self.settings
        base = st.output_path(self.path)
        created = self._prepare_dir(os.path.dirname(base))
        filters = {"png": "PNG (*.png)", "tif": "TIFF (*.tif)", "jpg": "JPEG (*.jpg)"}
        order = [st.format] + [k for k in filters if k != st.format]
        path, _ = QFileDialog.getSaveFileName(self, "Save result", base,
                                              ";;".join(filters[k] for k in order))
        self._drop_if_unused(os.path.dirname(base), created, os.path.dirname(path) if path else "")
        if not path:
            return
        try:
            save_image(path, self.result, int(self.bits.currentText()), st.jpeg_quality)
            if st.save_params:
                with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf8") as fh:
                    fh.write(self.params.to_json())
            self.statusBar().showMessage(f"Saved {path}")
        except Exception as exc:
            QMessageBox.critical(self, APP_NAME, str(exc))

    def batch_dialog(self):
        files = self._list_paths()
        if not files or self.busy:
            return
        start = self.settings.save_dir or os.path.join(self.folder, "restored")
        created = self._prepare_dir(start)
        out = QFileDialog.getExistingDirectory(self, "Folder for the restored images", start)
        if out:
            src_dirs = {os.path.normcase(os.path.dirname(f)) for f in files}
            if os.path.normcase(os.path.abspath(out)) in src_dirs:
                out = os.path.join(out, "restored")   # never overwrite the scans
        self._drop_if_unused(start, created, out)
        if not out:
            return
        st = _settings.AppSettings.from_dict(self.settings.to_dict())
        st.pipeline = self._mode()                # what the toolbar shows now
        st.bits = int(self.bits.currentText())
        self.busy = True

        def work(progress):
            log = []

            def emit(text):
                log.append(text)
                _crash.log.info("batch: %s", text)

            _cli.run_batch(files, st, out, emit=emit, output_is_dir=True,
                           progress=lambda k, n: progress(f"Restoring {k}/{n}", k / n))
            return log

        def done(log):
            self.busy = False
            self.progress.hide()
            self.report.setPlainText("\n".join(log))
            self.statusBar().showMessage(f"Batch finished: results in {out}")

        self._run_job(work, done, "Batch…")

    # ---- menus, options, about -------------------------------------------
    def _title(self, name: str | None = None) -> str:
        t = f"{APP_NAME} {__version__}"
        return f"{t} - {name}" if name else t

    def _build_menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&File")
        for a in (self.act_open, self.act_open_folder):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.act_save)
        m.addAction(self.act_batch)
        m.addSeparator()
        m.addAction(self.act_load_settings)
        m.addAction(self.act_save_settings)
        m.addSeparator()
        quit_ = QAction("E&xit", self)
        quit_.triggered.connect(self.close)
        m.addAction(quit_)
        m = mb.addMenu("&Settings")
        opt = QAction("&Options…", self)
        opt.setShortcut(QKeySequence("Ctrl+,"))
        opt.triggered.connect(self.open_options)
        m.addAction(opt)
        m = mb.addMenu("&Help")
        logs = QAction("Open crash reports and &logs folder", self)
        logs.triggered.connect(self.open_logs_folder)
        m.addAction(logs)
        issue = QAction("&Report a problem on GitHub…", self)
        issue.triggered.connect(self.report_problem)
        m.addAction(issue)
        m.addSeparator()
        about = QAction(f"&About {APP_NAME}", self)
        about.triggered.connect(self.show_about)
        m.addAction(about)

    # ---- updates -----------------------------------------------------------
    def _net_job(self, fn, done, error=None):
        job = Job(fn)
        job.setAutoDelete(False)
        self._jobs.append(job)

        def finish():
            if job in self._jobs:
                self._jobs.remove(job)

        job.signals.done.connect(lambda res: (finish(), done(res)))
        job.signals.error.connect(lambda msg: (finish(), error(msg) if error else None))
        self.net_pool.start(job)
        return job

    def check_for_updates(self, on_result):
        """Background check; on_result(release_or_None, error_or_None)."""
        def work(_progress):
            try:
                return _updater.check(), None
            except _updater.UpdateError as exc:
                return None, str(exc)

        self._net_job(work, lambda res: on_result(*res),
                      lambda msg: on_result(None, msg.strip().splitlines()[-1]))

    def _startup_update_check(self):
        def done(release, error):
            if release is not None and release.version != self.settings.skip_version:
                self._offer_update(release, startup=True)
        self.check_for_updates(done)

    def show_about(self):
        AboutDialog(self, check=self.check_for_updates, update=self._offer_update).exec()

    def _offer_update(self, release, startup: bool = False):
        dlg = UpdateDialog(release, self, allow_skip=startup)
        dlg.exec()
        if dlg.choice == "update":
            self.update_to(release, confirm=False)      # the dialog was the confirmation
        elif dlg.choice == "view":
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl(release.url))
        elif dlg.choice == "skip":
            self.settings.skip_version = release.version
            try:
                _settings.save(self.settings)
            except OSError:
                pass

    def update_to(self, release, confirm: bool = True):
        ok, why = _updater.can_update()
        if not ok:
            QMessageBox.information(
                self, APP_NAME, f"{why}<br><br>Version {release.version}: "
                f'<a href="{release.url}">{release.url}</a>')
            return
        if self.busy:
            QMessageBox.information(self, APP_NAME, "Please wait until the current work "
                                    "has finished, then update.")
            return
        if confirm and QMessageBox.question(
                self, APP_NAME,
                f"Download and install {APP_NAME} {release.version}?\n\n"
                "Cover Cleaner closes while it installs the update and then starts again. "
                "Unsaved results are lost.") != QMessageBox.Yes:
            return
        from PySide6.QtWidgets import QProgressDialog
        dlg = QProgressDialog(f"Downloading {APP_NAME} {release.version}…", None, 0, 100, self)
        dlg.setWindowTitle(APP_NAME)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setValue(0)
        dlg.show()

        def work(progress):
            def cb(done, total):
                if total:
                    progress("downloading", done / total)
            return _updater.download(release, cb)

        def done(new_app):
            dlg.setLabelText("Starting the update…")
            try:
                _updater.start_update(new_app, release.version)
            except Exception as exc:
                dlg.close()
                QMessageBox.critical(self, APP_NAME, f"The update could not start:\n{exc}")
                return
            dlg.close()
            QApplication.instance().quit()

        def failed(msg):
            dlg.close()
            QMessageBox.critical(self, APP_NAME, "The update failed; nothing was changed.\n\n"
                                 + msg.strip().splitlines()[-1])

        job = self._net_job(work, done, failed)
        job.signals.progress.connect(lambda _n, f: dlg.setValue(int(f * 100)))

    def open_options(self):
        dlg = OptionsDialog(self.settings, self)
        if not dlg.exec():
            return
        new = dlg.result_settings()
        try:
            _settings.save(new)
        except OSError as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not save the options:\n{exc}")
        old, self.settings = self.settings, new
        _ml.set_use_gpu(new.use_gpu)
        if new.pipeline != old.pipeline:
            self.mode_combo.setCurrentIndex(1 if new.mode(_ml.available()) == "ml" else 0)
        if new.bits != old.bits:
            self.bits.setCurrentIndex(1 if new.bits == 16 else 0)
        self.statusBar().showMessage("Options saved. Auto uses them from the next run.")


ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def app_icon():
    """The program icon (tools/make_icon.py), in the sizes Qt picks from."""
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QIcon
    icon = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        path = os.path.join(ASSETS, f"icon_{s}.png")
        if os.path.isfile(path):
            icon.addFile(path, QSize(s, s))
    return icon


class _ErrorBridge(QObject):
    """Carries 'an error report was written' from any thread to the window."""
    error = Signal(str)


def run_gui(paths=None) -> int:
    if sys.platform == "win32":
        # Its own taskbar entry and icon, instead of being grouped under Python.
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Biopanda.CoverCleaner")
        except (AttributeError, OSError):
            pass
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setDesktopFileName("cover-cleaner")        # matches the Linux .desktop file
    app.setWindowIcon(app_icon())
    bridge = _ErrorBridge()
    previous = _crash.install(on_error=bridge.error.emit)
    from PySide6.QtCore import qInstallMessageHandler
    qInstallMessageHandler(_crash.qt_message_handler)
    win = MainWindow()
    bridge.error.connect(lambda path: win.show_crash_report(path))
    win.show()
    if previous:
        QTimer.singleShot(800, lambda: win.show_crash_report(previous, after_crash=True))
    if isinstance(paths, str):
        paths = [paths]
    if paths:
        win.open_paths(list(paths))
    try:
        return app.exec()
    finally:
        _crash.clean_exit()

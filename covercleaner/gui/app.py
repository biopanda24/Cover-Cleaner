"""Interactive editor: Auto analysis plus manual control over every fix."""
from __future__ import annotations

import os
import sys
import traceback

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDockWidget,
                               QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QListWidget, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
                               QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget)

from .. import __version__
from .. import cli as _cli
from .. import descreen as _descreen
from .. import settings as _settings
from .. import geometry as _geometry
from .. import ml as _ml
from ..auto import auto_params
from ..imageio import SUPPORTED, load_image, read_dpi, save_image
from ..params import Params
from ..pipeline import Pipeline
from .options import APP_NAME, EXPERIMENTAL, AboutDialog, OptionsDialog
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

        self.view = ImageView()
        self.setCentralWidget(self.view)
        self.view.cornerMoved.connect(self._corner_moved)
        self.view.cornerReleased.connect(self.request_render)
        self.view.strokeFinished.connect(self._stroke_finished)
        self.view.picked.connect(self._picked)

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
        self.file_list.currentTextChanged.connect(self._file_selected)
        dock.setWidget(self.file_list)
        dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.files_dock = dock
        self.folder = ""

    def _build_report_dock(self):
        dock = QDockWidget("Auto analysis report", self)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
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
        cb = QCheckBox("Straighten curved edges (paper not flat)")
        cb.toggled.connect(lambda on: self._set("geometry", "use_bow", on))
        self._widgets[("geometry", "use_bow")] = cb
        lay.addWidget(cb)
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
        self.request_render()

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
        if self.pipe:
            self.pipe.clear_masks()
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
        path, _ = QFileDialog.getOpenFileName(self, "Open scan", self.folder, f"Images ({exts})")
        if path:
            self.set_folder(os.path.dirname(path), select=os.path.basename(path))

    def open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Open folder of scans", self.folder)
        if folder:
            self.set_folder(folder)

    def set_folder(self, folder: str, select: str | None = None):
        self.folder = folder
        names = sorted(n for n in os.listdir(folder) if n.lower().endswith(SUPPORTED))
        self.file_list.blockSignals(True)
        self.file_list.clear()
        self.file_list.addItems(names)
        self.file_list.blockSignals(False)
        if names:
            idx = names.index(select) if select in names else 0
            self.file_list.setCurrentRow(idx)

    def _file_selected(self, name: str):
        if name:
            self.load(os.path.join(self.folder, name))

    def load(self, path: str, auto: bool = True):
        try:
            self.source = load_image(path)
        except Exception as exc:
            QMessageBox.warning(self, "Cover Cleaner", str(exc))
            return
        self.path = path
        self.pipe = Pipeline(self.source)
        self.params = Params()
        self.params.mode = self._mode()
        self.result = None
        self.report.setPlainText("")
        self.setWindowTitle(self._title(os.path.basename(path)))
        self._src_pixmap = to_pixmap(self.source)
        self._set_enabled(True)
        self.sync_controls()
        self.view.set_overlay(None)
        self.set_view_mode("original", keep_view=False)
        if auto:
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
        QMessageBox.critical(self, "Cover Cleaner", msg[-1500:])

    def run_auto(self):
        if self.source is None:
            return
        if self.busy:                       # e.g. the previous file is still rendering
            QTimer.singleShot(300, self.run_auto)
            return
        self.busy = True
        src, pipe, path, mode = self.source, self.pipe, self.path, self._mode()
        st = self.settings

        def work(progress):
            return path, auto_params(src, progress, pipe, mode, st.dpi_for(read_dpi(path)),
                                     flat_cleaner=st.flat_cleaner, rules=st.rules,
                                     scale=st.scale)

        def done(res):
            self.busy = False
            rpath, (params, report) = res
            if rpath != self.path:
                return                      # another file was opened meanwhile
            self.params = params
            self.report.setPlainText("\n".join(report))
            self.sync_controls()
            self.set_view_mode("result")
            self._refresh_outline()
            self.request_render()

        self._run_job(work, done, "Analysing…")

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

        def work(progress):
            return path, pipe.run(params, progress=progress)

        def done(res):
            self.busy = False
            rpath, img = res
            self.progress.hide()
            if rpath != self.path:
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
        if self.pipe:
            self.pipe.clear_masks()         # painted marks no longer line up
        self.dirty = True                   # render when the handle is released

    def detect_outline(self):
        if self.source is None:
            return
        det = _geometry.detect_cover(self.source, dpi=read_dpi(self.path))
        g = self.params.geometry
        if det.has_background:
            g.corners = [[float(x), float(y)] for x, y in det.corners]
            g.bow = [[float(v) for v in row] for row in det.bow]
            g.trim = [int(t) for t in det.trim]
            g.disc = det.disc is not None
        else:
            g.corners = None
            self.statusBar().showMessage("No scanner background found around the cover.")
        g.enabled = True
        self.sync_controls()
        self._refresh_outline()
        self.request_render()

    def outline_full(self):
        g = self.params.geometry
        g.corners = [[float(x), float(y)] for x, y in _geometry.full_corners(self.source.shape)]
        g.bow = None
        g.trim = [0, 0, 0, 0]
        g.disc = False
        self.sync_controls()
        self.set_view_mode("original")
        self.request_render()

    def detect_screen(self):
        if self.pipe is None:
            return
        img = self.pipe.run(self.params, upto="geometry")
        scr = _descreen.analyze_screen(img)
        d = self.params.descreen
        if scr.found:
            d.peaks, d.period, d.enabled = scr.peaks, scr.period, True
        else:
            d.peaks, d.period, d.enabled = [], 0.0, False
            self.statusBar().showMessage("No halftone screen found in this scan.")
        self.sync_controls()
        self.request_render()

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

    def save_dialog(self):
        if self.result is None or self.busy or self.dirty:
            self.statusBar().showMessage("Still processing - try again in a moment.")
            return
        st = self.settings
        base = st.output_path(self.path)
        os.makedirs(os.path.dirname(base), exist_ok=True)
        filters = {"png": "PNG (*.png)", "tif": "TIFF (*.tif)", "jpg": "JPEG (*.jpg)"}
        order = [st.format] + [k for k in filters if k != st.format]
        path, _ = QFileDialog.getSaveFileName(self, "Save result", base,
                                              ";;".join(filters[k] for k in order))
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
        names = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not names or self.busy:
            return
        start = self.settings.save_dir or os.path.join(self.folder, "restored")
        out = QFileDialog.getExistingDirectory(self, "Folder for the restored images", start)
        if not out:
            return
        if os.path.normcase(os.path.abspath(out)) == os.path.normcase(os.path.abspath(self.folder)):
            out = os.path.join(out, "restored")   # never overwrite the scans
        st = _settings.AppSettings.from_dict(self.settings.to_dict())
        st.pipeline = self._mode()                # what the toolbar shows now
        st.bits = int(self.bits.currentText())
        files = [os.path.join(self.folder, n) for n in names]
        self.busy = True

        def work(progress):
            log = []

            def emit(text):
                log.append(text)

            _cli.run_batch(files, st, out, emit=emit,
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
        about = QAction(f"&About {APP_NAME}", self)
        about.triggered.connect(lambda: AboutDialog(self).exec())
        m.addAction(about)

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


def run_gui(image: str | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    if image:
        if os.path.isdir(image):
            win.set_folder(image)
        else:
            win.set_folder(os.path.dirname(os.path.abspath(image)), os.path.basename(image))
    return app.exec()

"""Options and About dialogs."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QSpinBox, QTabWidget, QVBoxLayout,
                               QWidget)

from .. import __version__
from .. import ml as _ml
from .. import ocr as _ocr
from ..settings import AppSettings, AutoRules, settings_path

APP_NAME = "Cover Cleaner"
CREDITS = "Biopanda"
EXPERIMENTAL = "(Experimental)"


def _form(parent: QTabWidget, title: str) -> QFormLayout:
    page = QWidget()
    form = QFormLayout(page)
    form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
    parent.addTab(page, title)
    return form


def _note(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet("color: gray;")
    return lab


class OptionsDialog(QDialog):
    """Edit AppSettings. `result_settings()` returns the edited copy."""

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Options")
        self.setMinimumWidth(560)
        self._s = AppSettings.from_dict(settings.to_dict())
        tabs = QTabWidget()

        # ---- Output ------------------------------------------------------
        f = _form(tabs, "Output")
        row = QHBoxLayout()
        self.save_dir = QLineEdit(self._s.save_dir)
        self.save_dir.setPlaceholderText("a 'restored' folder next to the scans")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        clear = QPushButton("Next to scans")
        clear.setToolTip("Save into a 'restored' folder next to each scan")
        clear.clicked.connect(lambda: self.save_dir.setText(""))
        row.addWidget(self.save_dir, 1)
        row.addWidget(browse)
        row.addWidget(clear)
        f.addRow("Default save location", row)
        self.format = QComboBox()
        self.format.addItems(["PNG", "TIFF", "JPEG"])
        self.format.setCurrentIndex(("png", "tif", "jpg").index(self._s.format))
        f.addRow("Format", self.format)
        self.bits = QComboBox()
        self.bits.addItems(["8-bit", "16-bit"])
        self.bits.setCurrentIndex(1 if self._s.bits == 16 else 0)
        self.bits.setToolTip("16-bit is kept for PNG and TIFF; JPEG is always 8-bit.")
        f.addRow("Bit depth", self.bits)
        self.jpeg_q = QSpinBox()
        self.jpeg_q.setRange(50, 100)
        self.jpeg_q.setValue(self._s.jpeg_quality)
        f.addRow("JPEG quality", self.jpeg_q)
        out_form = f
        # JPEG quality only matters for JPEG; bit depth only for PNG and TIFF
        def format_changed(i):
            jpeg = i == 2
            out_form.setRowVisible(self.jpeg_q, jpeg)
            self.bits.setEnabled(not jpeg)
            self.bits.setToolTip("JPEG is always 8-bit." if jpeg else
                                 "16-bit is kept for PNG and TIFF.")

        self.format.currentIndexChanged.connect(format_changed)
        format_changed(self.format.currentIndex())
        self.suffix = QLineEdit(self._s.suffix)
        self.suffix.setPlaceholderText("none, e.g. _clean")
        f.addRow("File name suffix", self.suffix)
        self.overwrite = QCheckBox("Overwrite existing files")
        self.overwrite.setChecked(self._s.overwrite)
        self.overwrite.setToolTip("Otherwise a number is added: 'name (2).png'.")
        f.addRow("", self.overwrite)
        self.save_params = QCheckBox("Save the chosen settings (.json) with each result")
        self.save_params.setChecked(self._s.save_params)
        self.save_params.setToolTip("They can be loaded again (File > Load settings…) to "
                                    "adjust a result later.")
        f.addRow("", self.save_params)
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.1, 2.0)
        self.scale.setSingleStep(0.05)
        self.scale.setDecimals(2)
        self.scale.setValue(self._s.scale)
        self.scale.setToolTip("Resize factor for results of Auto; 0.5 turns a 600 dpi scan "
                              "into 300 dpi.")
        f.addRow("Output scale", self.scale)

        # ---- Processing --------------------------------------------------
        f = _form(tabs, "Processing")
        self.pipeline = QComboBox()
        self.pipeline.addItems(["Automatic (ML when the model is installed)", "ML",
                                "Traditional"])
        self.pipeline.setCurrentIndex(("auto", "ml", "traditional").index(self._s.pipeline))
        f.addRow("Default pipeline", self.pipeline)
        self.gpu = QCheckBox("Use the GPU for ML inpainting when available")
        self.gpu.setChecked(self._s.use_gpu)
        f.addRow("", self.gpu)
        self.jobs = QSpinBox()
        self.jobs.setRange(0, 64)
        self.jobs.setSpecialValueText("Automatic")
        self.jobs.setValue(self._s.jobs)
        self.jobs.setToolTip("For 'Auto-restore all' and the command line. Automatic picks "
                             "a number from the CPU threads, free memory and, with the GPU, "
                             "free GPU memory (about 1.9 GB per scan).")
        f.addRow("Scans processed at once", self.jobs)
        self.dpi = QDoubleSpinBox()
        self.dpi.setRange(0, 4800)
        self.dpi.setDecimals(0)
        self.dpi.setSingleStep(100)
        self.dpi.setSpecialValueText("None")
        self.dpi.setValue(self._s.fallback_dpi)
        self.dpi.setToolTip("Assumed for scans that store no resolution. The standard-size, "
                            "white-border and staple rules need it.")
        f.addRow("Fallback DPI", self.dpi)

        # ---- Auto --------------------------------------------------------
        f = _form(tabs, "Auto")
        self.flat = QCheckBox(f"Use flat colour cleanup when flat sections are detected "
                              f"{EXPERIMENTAL}")
        self.flat.setChecked(self._s.flat_cleaner)
        self.flat.setToolTip("Covers that are mostly solid-colour artwork get every flat "
                             "section painted with its one colour and clean edges between "
                             "them. Still experimental: check the results.")
        f.addRow(self.flat)
        f.addRow(_note("Off: Auto reports flat sections but leaves them as scanned; "
                       "flat colour cleanup can still be switched on for a single scan in "
                       "the Fixes panel."))
        f.addRow(QLabel("<b>Crop rules</b>"))
        r = self._s.rules
        self.r_size = QCheckBox("Place edges that cannot be seen from standard jacket sizes")
        self.r_size.setChecked(r.standard_size)
        self.r_white = QCheckBox("Restore a white border cut off on a white scanner lid")
        self.r_white.setChecked(r.white_border)
        self.r_strip = QCheckBox("Crop off things showing past one edge "
                                 "(booklet, tray card, thin white strip)")
        self.r_strip.setChecked(r.strip_crop)
        self.r_staples = QCheckBox("Fill in or crop out staples at a booklet's spine")
        self.r_staples.setChecked(r.staples)
        for cb in (self.r_size, self.r_white, self.r_strip, self.r_staples):
            f.addRow(cb)

        # ---- Models ------------------------------------------------------
        f = _form(tabs, "Models")
        self._model_rows(f)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel
                                   | QDialogButtonBox.RestoreDefaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._defaults)
        lay = QVBoxLayout(self)
        lay.addWidget(tabs)
        lay.addWidget(_note(f"Options are stored in {settings_path()}"))
        lay.addWidget(buttons)

    def _model_rows(self, f: QFormLayout):
        def status(path: str) -> str:
            if os.path.isfile(path):
                return f"installed ({os.path.getsize(path) / 1e6:.0f} MB)"
            return "missing"

        f.addRow(QLabel("<b>Inpainting (ML pipeline)</b>"))
        f.addRow("LaMa model", QLabel(status(_ml.model_path())))
        path_lab = _note(_ml.model_path())
        path_lab.setWordWrap(False)
        path_lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        f.addRow("", path_lab)
        f.addRow("GPU copy", QLabel(status(_ml.float_model_path())
                                    + ("" if os.path.isfile(_ml.float_model_path()) else
                                       " - written on first GPU use")))
        try:
            import onnxruntime as ort
            gpu = ("CUDA available" if "CUDAExecutionProvider" in ort.get_available_providers()
                   else "no CUDA provider")
            gpu = f"onnxruntime {ort.__version__}: {gpu}"
        except ImportError:
            gpu = "onnxruntime-gpu not installed - ML runs on the CPU"
        f.addRow("GPU support", QLabel(gpu))
        f.addRow(QLabel("<b>Disc text (turning discs upright)</b>"))
        for path in getattr(_ocr, "model_paths", lambda: [])():
            f.addRow(os.path.basename(path), QLabel(status(path)))

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Default save location",
                                             self.save_dir.text() or os.getcwd())
        if d:
            self.save_dir.setText(d)

    def _defaults(self):
        d = AppSettings()
        self.save_dir.setText(d.save_dir)
        self.format.setCurrentIndex(("png", "tif", "jpg").index(d.format))
        self.bits.setCurrentIndex(1 if d.bits == 16 else 0)
        self.jpeg_q.setValue(d.jpeg_quality)
        self.suffix.setText(d.suffix)
        self.overwrite.setChecked(d.overwrite)
        self.save_params.setChecked(d.save_params)
        self.scale.setValue(d.scale)
        self.pipeline.setCurrentIndex(("auto", "ml", "traditional").index(d.pipeline))
        self.gpu.setChecked(d.use_gpu)
        self.jobs.setValue(d.jobs)
        self.dpi.setValue(d.fallback_dpi)
        self.flat.setChecked(d.flat_cleaner)
        for cb, v in ((self.r_size, d.rules.standard_size), (self.r_white, d.rules.white_border),
                      (self.r_strip, d.rules.strip_crop), (self.r_staples, d.rules.staples)):
            cb.setChecked(v)

    def result_settings(self) -> AppSettings:
        s = AppSettings()
        s.save_dir = self.save_dir.text().strip()
        s.format = ("png", "tif", "jpg")[self.format.currentIndex()]
        s.bits = 16 if self.bits.currentIndex() == 1 else 8
        s.jpeg_quality = self.jpeg_q.value()
        s.suffix = self.suffix.text().strip()
        s.overwrite = self.overwrite.isChecked()
        s.save_params = self.save_params.isChecked()
        s.scale = float(self.scale.value())
        s.pipeline = ("auto", "ml", "traditional")[self.pipeline.currentIndex()]
        s.use_gpu = self.gpu.isChecked()
        s.jobs = self.jobs.value()
        s.fallback_dpi = float(self.dpi.value())
        s.flat_cleaner = self.flat.isChecked()
        s.rules = AutoRules(standard_size=self.r_size.isChecked(),
                            white_border=self.r_white.isChecked(),
                            strip_crop=self.r_strip.isChecked(),
                            staples=self.r_staples.isChecked())
        return s


def about_label() -> QLabel:
    lab = QLabel(
        f"<h2>{APP_NAME}</h2>"
        f"<p>Version {__version__}</p>"
        "<p>Restores scanned CD, DVD, VHS and cassette covers to look like the clean "
        "original artwork.</p>"
        f"<p><b>Credits</b><br>{CREDITS}</p>"
        "<p>LaMa inpainting model and text models from the OpenCV model zoo "
        "(Apache-2.0).</p>")
    lab.setTextFormat(Qt.RichText)
    lab.setWordWrap(True)
    return lab


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {APP_NAME}")
        lay = QVBoxLayout(self)
        lay.addWidget(about_label())
        b = QDialogButtonBox(QDialogButtonBox.Ok)
        b.accepted.connect(self.accept)
        lay.addWidget(b)

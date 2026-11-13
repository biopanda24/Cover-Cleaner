"""Application options, shared by the editor and the command line.

Stored as JSON in the user's settings folder (%APPDATA%\\CoverCleaner on
Windows, ~/.config/covercleaner elsewhere).  Command-line flags override
them for a single run.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields

FORMATS = ("png", "tif", "jpg")
PIPELINES = ("auto", "ml", "traditional")


@dataclass
class AutoRules:
    """Individual Auto crop rules that can be switched off."""
    standard_size: bool = True    # place unseen edges from standard jacket sizes
    white_border: bool = True     # restore a white border cut off on a white lid
    strip_crop: bool = True       # crop things showing past one edge / one-sided white strip
    staples: bool = True          # fill in or crop out staples at the spine


@dataclass
class AppSettings:
    # output
    save_dir: str = ""            # "" = a 'restored' folder next to the scans
    format: str = "png"
    bits: int = 8
    jpeg_quality: int = 95
    suffix: str = ""              # added to the file name, e.g. "_clean"
    overwrite: bool = False       # otherwise "name (2).png" etc.
    save_params: bool = True      # write the chosen settings (.json) next to each result
    scale: float = 1.0            # resize factor for saved results
    # processing
    pipeline: str = "auto"        # auto = ML when the model is installed
    use_gpu: bool = True
    jobs: int = 0                 # scans processed at once in a batch; 0 = automatic
    fallback_dpi: float = 600.0   # assumed when a scan stores none; 0 = none
    # Auto
    flat_cleaner: bool = False    # Experimental: flat colour cleanup when flat areas are found
    rules: AutoRules = field(default_factory=AutoRules)

    # ---- helpers ---------------------------------------------------------
    def mode(self, ml_available: bool) -> str:
        if self.pipeline == "ml" and ml_available:
            return "ml"
        if self.pipeline == "traditional":
            return "traditional"
        return "ml" if ml_available else "traditional"

    def dpi_for(self, stored: float | None) -> float | None:
        if stored:
            return stored
        return self.fallback_dpi if self.fallback_dpi and self.fallback_dpi > 0 else None

    def output_path(self, src: str, folder: str | None = None) -> str:
        """Where the result of `src` is saved under these options."""
        folder = folder or self.save_dir or os.path.join(os.path.dirname(os.path.abspath(src)),
                                                         "restored")
        base = os.path.splitext(os.path.basename(src))[0] + self.suffix
        path = os.path.join(folder, f"{base}.{self.format}")
        if self.overwrite:
            return path
        k = 2
        while os.path.exists(path):
            path = os.path.join(folder, f"{base} ({k}).{self.format}")
            k += 1
        return path

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AppSettings":
        s = cls()
        for f in fields(cls):
            if f.name not in d:
                continue
            if f.name == "rules":
                r = AutoRules()
                for rf in fields(AutoRules):
                    if rf.name in (d["rules"] or {}):
                        setattr(r, rf.name, bool(d["rules"][rf.name]))
                s.rules = r
            else:
                setattr(s, f.name, type(getattr(s, f.name))(d[f.name]))
        if s.format not in FORMATS:
            s.format = "png"
        if s.pipeline not in PIPELINES:
            s.pipeline = "auto"
        s.bits = 16 if s.bits == 16 else 8
        return s


def settings_path() -> str:
    env = os.environ.get("COVERCLEANER_SETTINGS")
    if env:
        return env
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "CoverCleaner" if os.environ.get("APPDATA") else "covercleaner",
                        "settings.json")


def load() -> AppSettings:
    try:
        with open(settings_path(), encoding="utf8") as fh:
            return AppSettings.from_dict(json.load(fh))
    except (OSError, ValueError, TypeError):
        return AppSettings()


def save(s: AppSettings) -> None:
    path = settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as fh:
        json.dump(s.to_dict(), fh, indent=2)
    os.replace(tmp, path)

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
    # updates
    check_updates: bool = True    # look for a new version when the program starts
    skip_version: str = ""        # a version the user chose to skip

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

    def output_path(self, src: str, folder: str | None = None, taken: set | None = None) -> str:
        """Where the result of `src` is saved under these options.

        Never in the scan's own folder (a 'restored' folder inside it is used
        instead), so an original can never be overwritten.  `taken` collects
        the names already given out in this batch: two scans with the same
        name (a.png and a.tif, or front.png from two folders) get different
        files even when they are written at the same time.
        """
        folder = os.path.abspath(folder or self.save_dir or
                                 os.path.join(os.path.dirname(os.path.abspath(src)), "restored"))
        if os.path.normcase(folder) == os.path.normcase(os.path.dirname(os.path.abspath(src))):
            folder = os.path.join(folder, "restored")
        base = os.path.splitext(os.path.basename(src))[0] + self.suffix
        taken = taken if taken is not None else set()
        path = os.path.join(folder, f"{base}.{self.format}")
        k = 2
        while (os.path.normcase(path) in taken
               or (not self.overwrite and os.path.exists(path))):
            path = os.path.join(folder, f"{base} ({k}).{self.format}")
            k += 1
        taken.add(os.path.normcase(path))
        return path

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AppSettings":
        """Read stored options field by field: a missing or unreadable value
        (hand-edited, from an older version) keeps its default instead of
        resetting everything."""
        s = cls()
        if not isinstance(d, dict):
            return s
        for f in fields(cls):
            if f.name not in d or d[f.name] is None:
                continue
            if f.name == "rules":
                r = AutoRules()
                if isinstance(d["rules"], dict):
                    for rf in fields(AutoRules):
                        v = _as_bool(d["rules"].get(rf.name))
                        if v is not None:
                            setattr(r, rf.name, v)
                s.rules = r
                continue
            v = _coerce(d[f.name], getattr(s, f.name))
            if v is not None:
                setattr(s, f.name, v)
        if s.format not in FORMATS:
            s.format = "png"
        if s.pipeline not in PIPELINES:
            s.pipeline = "auto"
        s.bits = 16 if s.bits == 16 else 8
        s.jpeg_quality = int(min(100, max(1, s.jpeg_quality)))
        s.scale = float(min(4.0, max(0.05, s.scale)))
        s.jobs = max(0, s.jobs)
        s.fallback_dpi = max(0.0, s.fallback_dpi)
        return s


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "yes", "on", "1", "false", "no", "off", "0"):
        return v.strip().lower() in ("true", "yes", "on", "1")
    return None


def _coerce(v, default):
    """`v` converted to the type of `default`, or None if it cannot be."""
    try:
        if isinstance(default, bool):
            return _as_bool(v)
        if isinstance(default, int):
            return int(float(v))
        if isinstance(default, float):
            x = float(v)
            return x if x == x else None          # not NaN
        if isinstance(default, str):
            return v if isinstance(v, str) else None
    except (TypeError, ValueError):
        return None
    return None


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
    except (OSError, ValueError, TypeError, AttributeError):
        return AppSettings()


def save(s: AppSettings) -> None:
    path = settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as fh:
        json.dump(s.to_dict(), fh, indent=2)
    os.replace(tmp, path)

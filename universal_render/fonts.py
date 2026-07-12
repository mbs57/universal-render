# universal_render/fonts.py
"""
Script-aware font management: per-script candidate tables, glyph-coverage
validation, and automatic font fallback.

Resolution order for a piece of text:

1. Explicit ``font_path``  (registered, its family used directly)
2. Explicit ``font_family`` (validated for coverage; falls back if it
   cannot draw the text and fallback is allowed)
3. Per-script default set via set_script_font()
4. First installed candidate from SCRIPT_FONT_CANDIDATES that covers
   the text
5. The package default font
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence

from .backend import ensure_qt_application
from .scripts import COMMON, INHERITED, detect_char_script, detect_script, segment_runs

try:
    from PySide6.QtGui import (
        QFont,
        QFontDatabase,
        QFontMetrics,
        QRawFont,
    )
    QT_FONT_AVAILABLE = True
    QT_FONT_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    QFont = None
    QFontDatabase = None
    QFontMetrics = None
    QRawFont = None
    QT_FONT_AVAILABLE = False
    QT_FONT_IMPORT_ERROR = e


# ---------------------------------------------------------------------
# Candidate tables
# ---------------------------------------------------------------------
# Ordered per script: fonts shipped with Windows first (guaranteed on the
# primary target platform), then Noto families (common on Linux/CI and
# easy to install anywhere), then legacy/system alternatives.

SCRIPT_FONT_CANDIDATES: Dict[str, List[str]] = {
    "Bengali": [
        "Nirmala UI", "Noto Sans Bengali", "Noto Serif Bengali",
        "SolaimanLipi", "Kalpurush", "Nikosh", "Vrinda", "Siyam Rupali",
        "Hind Siliguri", "Mukti", "Akaash", "Lohit Bengali",
    ],
    "Devanagari": [
        "Nirmala UI", "Noto Sans Devanagari", "Noto Serif Devanagari",
        "Mangal", "Hind", "Lohit Devanagari", "Aparajita",
    ],
    "Tamil": [
        "Nirmala UI", "Noto Sans Tamil", "Latha", "Vijaya", "Lohit Tamil",
    ],
    "Telugu": [
        "Nirmala UI", "Noto Sans Telugu", "Gautami", "Vani", "Lohit Telugu",
    ],
    "Kannada": [
        "Nirmala UI", "Noto Sans Kannada", "Tunga", "Lohit Kannada",
    ],
    "Malayalam": [
        "Nirmala UI", "Noto Sans Malayalam", "Kartika", "Lohit Malayalam",
    ],
    "Gujarati": [
        "Nirmala UI", "Noto Sans Gujarati", "Shruti", "Lohit Gujarati",
    ],
    "Gurmukhi": [
        "Nirmala UI", "Noto Sans Gurmukhi", "Raavi", "Lohit Gurmukhi",
    ],
    "Odia": [
        "Nirmala UI", "Noto Sans Oriya", "Kalinga", "Lohit Odia",
    ],
    "Sinhala": [
        "Nirmala UI", "Noto Sans Sinhala", "Iskoola Pota",
    ],
    "Arabic": [
        "Segoe UI", "Noto Naskh Arabic", "Noto Sans Arabic",
        "Tahoma", "Traditional Arabic", "Arial",
    ],
    "Hebrew": [
        "Segoe UI", "Noto Sans Hebrew", "David", "Tahoma", "Arial",
    ],
    "Thai": [
        "Leelawadee UI", "Noto Sans Thai", "Tahoma", "Angsana New",
    ],
    "Lao": [
        "Leelawadee UI", "Noto Sans Lao", "Lao UI", "DokChampa",
    ],
    "Khmer": [
        "Leelawadee UI", "Noto Sans Khmer", "Khmer UI", "DaunPenh",
    ],
    "Myanmar": [
        "Myanmar Text", "Noto Sans Myanmar", "Padauk",
    ],
    "Tibetan": [
        "Microsoft Himalaya", "Noto Serif Tibetan", "Noto Sans Tibetan",
    ],
    "Ethiopic": [
        "Ebrima", "Noto Sans Ethiopic", "Nyala",
    ],
    "Georgian": [
        "Sylfaen", "Noto Sans Georgian", "Segoe UI",
    ],
    "Armenian": [
        "Sylfaen", "Noto Sans Armenian", "Segoe UI",
    ],
    "Han": [
        "Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans SC", "SimSun",
    ],
    "Hiragana": [
        "Yu Gothic", "Meiryo", "Noto Sans CJK JP", "Noto Sans JP", "MS Gothic",
    ],
    "Katakana": [
        "Yu Gothic", "Meiryo", "Noto Sans CJK JP", "Noto Sans JP", "MS Gothic",
    ],
    "Hangul": [
        "Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR", "Gulim",
    ],
    "Greek": [
        "Segoe UI", "Noto Sans", "Arial", "DejaVu Sans",
    ],
    "Cyrillic": [
        "Segoe UI", "Noto Sans", "Arial", "DejaVu Sans",
    ],
    "Latin": [
        "Segoe UI", "Noto Sans", "Arial", "DejaVu Sans",
    ],
    COMMON: [
        "Segoe UI", "Noto Sans", "Arial", "DejaVu Sans",
    ],
}


# ---------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------

_DEFAULT_FONT_FAMILY = "Segoe UI"
_DEFAULT_FONT_PATH: Optional[str] = None

# User-pinned per-script overrides (script -> family)
_SCRIPT_FONT_OVERRIDES: Dict[str, str] = {}

_REGISTERED_FONT_FILES: List[str] = []
_REGISTERED_FONT_FAMILIES: Dict[str, List[str]] = {}  # font_path -> families

# Cache: (script, text-key) -> resolved family
_RESOLVE_CACHE: Dict[Any, str] = {}
_RESOLVE_CACHE_MAX = 1024


@dataclass
class FontValidationResult:
    ok: bool
    requested_family: Optional[str] = None
    requested_path: Optional[str] = None
    resolved_family: Optional[str] = None
    path_registered: bool = False
    family_exists: bool = False
    covers_sample: bool = False
    sample_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _ensure_font_runtime() -> None:
    ensure_qt_application()
    if not QT_FONT_AVAILABLE:
        raise RuntimeError(
            "Qt font classes are not available. "
            f"Original import error: {QT_FONT_IMPORT_ERROR}"
        )


def _normalize_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _cache_put(key: Any, family: str) -> None:
    if len(_RESOLVE_CACHE) >= _RESOLVE_CACHE_MAX:
        _RESOLVE_CACHE.clear()
    _RESOLVE_CACHE[key] = family


# ---------------------------------------------------------------------
# Font registration / listing
# ---------------------------------------------------------------------

def register_font(font_path: str) -> List[str]:
    """
    Register a font file (.ttf/.otf) with Qt and return its family names.
    """
    _ensure_font_runtime()
    path = _normalize_path(font_path)

    if path in _REGISTERED_FONT_FAMILIES:
        return list(_REGISTERED_FONT_FAMILIES[path])

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Font file not found: {path}")

    font_id = QFontDatabase.addApplicationFont(path)
    if font_id < 0:
        raise RuntimeError(f"Qt failed to load font file: {path}")

    families = list(QFontDatabase.applicationFontFamilies(font_id))
    _REGISTERED_FONT_FILES.append(path)
    _REGISTERED_FONT_FAMILIES[path] = families
    _RESOLVE_CACHE.clear()
    return families


def register_fonts(font_paths: Sequence[str]) -> Dict[str, List[str]]:
    """Register multiple font files; returns {path: families}."""
    return {p: register_font(p) for p in font_paths}


def list_available_fonts() -> List[str]:
    """All font families Qt can see on this system."""
    _ensure_font_runtime()
    return [str(f) for f in QFontDatabase.families()]


def list_registered_fonts() -> Dict[str, List[str]]:
    """Font files registered through register_font()."""
    return {p: list(f) for p, f in _REGISTERED_FONT_FAMILIES.items()}


def family_exists(family: str) -> bool:
    """True if a family with this (case-insensitive) name is installed."""
    _ensure_font_runtime()
    target = family.strip().lower()
    return any(str(f).strip().lower() == target for f in QFontDatabase.families())


# ---------------------------------------------------------------------
# Glyph coverage
# ---------------------------------------------------------------------

def font_covers_text(family: str, text: str, min_ratio: float = 1.0) -> bool:
    """
    True if ``family`` itself (font merging disabled) has glyphs for the
    script characters in ``text``.

    Common characters (spaces, digits, punctuation) are ignored so that
    a Bengali font is not rejected for missing a rarely used symbol.
    ``min_ratio`` allows tolerance, e.g. 0.9 accepts fonts covering 90%
    of the sampled characters.
    """
    _ensure_font_runtime()

    font = QFont(family)
    font.setStyleStrategy(QFont.StyleStrategy.NoFontMerging)
    raw = QRawFont.fromFont(font)
    if not raw.isValid():
        return False

    checked = 0
    covered = 0
    seen = set()
    for ch in str(text):
        if ch in seen:
            continue
        seen.add(ch)
        script = detect_char_script(ch)
        if script in (COMMON, INHERITED):
            continue
        checked += 1
        # Pass the integer codepoint: PySide6's QChar overload misresolves
        # single-character strings and reports False for supported glyphs.
        if raw.supportsCharacter(ord(ch)):
            covered += 1
        if checked >= 64:  # sampling cap for very long strings
            break

    if checked == 0:
        return True
    return (covered / checked) >= min_ratio


# ---------------------------------------------------------------------
# Script-level resolution
# ---------------------------------------------------------------------

def set_script_font(script: str, family: str) -> None:
    """
    Pin a font family for a script, taking priority over the candidate
    table. Example: set_script_font("Bengali", "Kalpurush").
    """
    _SCRIPT_FONT_OVERRIDES[script] = family
    _RESOLVE_CACHE.clear()


def get_script_font(script: str) -> Optional[str]:
    """Return the pinned family for a script, if any."""
    return _SCRIPT_FONT_OVERRIDES.get(script)


def find_font_for_script(
    script: str,
    sample_text: Optional[str] = None,
    preferred: Optional[str] = None,
) -> str:
    """
    Return the best installed font family for ``script``.

    Checks, in order: ``preferred``, the pinned override, each entry of
    SCRIPT_FONT_CANDIDATES. A candidate must be installed and (when
    ``sample_text`` is given) cover its script characters. Falls back to
    the package default family when nothing matches.
    """
    _ensure_font_runtime()

    key = ("script", script, sample_text, preferred)
    cached = _RESOLVE_CACHE.get(key)
    if cached is not None:
        return cached

    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)
    override = _SCRIPT_FONT_OVERRIDES.get(script)
    if override:
        candidates.append(override)
    candidates.extend(SCRIPT_FONT_CANDIDATES.get(script, []))
    candidates.append(_DEFAULT_FONT_FAMILY)

    for family in candidates:
        if not family_exists(family):
            continue
        if sample_text and not font_covers_text(family, sample_text):
            continue
        _cache_put(key, family)
        return family

    # Nothing installed covers it; return the default and let Qt's own
    # font merging find *some* glyph source at draw time.
    _cache_put(key, _DEFAULT_FONT_FAMILY)
    return _DEFAULT_FONT_FAMILY


def auto_font_fallback(text: str, preferred: Optional[str] = None) -> str:
    """
    Pick a font family for ``text`` automatically.

    The dominant script is detected, then resolved through
    find_font_for_script() with ``text`` itself as the coverage sample.
    """
    script = detect_script(text)
    return find_font_for_script(script, sample_text=text, preferred=preferred)


def resolve_fonts_for_runs(text: str) -> List[Dict[str, str]]:
    """
    For mixed-script text, return one entry per script run with the font
    that would be used: [{"text", "script", "family"}, ...].

    Rendering itself draws the whole string with the dominant-script
    font and lets Qt shape each run (with font merging covering the
    rest); this function exists for diagnostics and evaluation tables.
    """
    out = []
    for run in segment_runs(text):
        out.append({
            "text": run.text,
            "script": run.script,
            "family": find_font_for_script(run.script, sample_text=run.text),
        })
    return out


# ---------------------------------------------------------------------
# Default font / general resolution
# ---------------------------------------------------------------------

def set_default_font(font_family: Optional[str] = None, font_path: Optional[str] = None) -> str:
    """
    Set the package-wide default family (used for Latin/Common text and
    as the last-resort fallback). A ``font_path`` is registered first.
    """
    global _DEFAULT_FONT_FAMILY, _DEFAULT_FONT_PATH

    if font_path:
        families = register_font(font_path)
        _DEFAULT_FONT_PATH = _normalize_path(font_path)
        if not font_family and families:
            font_family = families[0]

    if font_family:
        _DEFAULT_FONT_FAMILY = str(font_family)

    _RESOLVE_CACHE.clear()
    return _DEFAULT_FONT_FAMILY


def get_default_font() -> str:
    return _DEFAULT_FONT_FAMILY


def resolve_font(
    font_family: Optional[str] = None,
    font_path: Optional[str] = None,
    text: Optional[str] = None,
) -> str:
    """
    Resolve the family to render with.

    - ``font_path`` wins: it is registered and its first family used.
    - ``font_family`` is honored when installed (no silent override).
    - Otherwise, when ``text`` is given, the family is chosen per the
      text's dominant script via auto_font_fallback().
    - Otherwise the package default family is returned.
    """
    _ensure_font_runtime()

    if font_path:
        families = register_font(font_path)
        if families:
            return families[0]

    if font_family:
        if family_exists(font_family):
            return str(font_family)
        # Requested family missing: fall through to script-aware choice.

    if text is not None:
        return auto_font_fallback(str(text), preferred=font_family)

    return _DEFAULT_FONT_FAMILY


def validate_font(
    font_family: Optional[str] = None,
    font_path: Optional[str] = None,
    sample_text: str = "Aআক্ষর",
) -> FontValidationResult:
    """
    Check that a font can actually draw ``sample_text``.
    """
    result = FontValidationResult(
        ok=False,
        requested_family=font_family,
        requested_path=font_path,
        sample_text=sample_text,
    )
    try:
        _ensure_font_runtime()

        if font_path:
            families = register_font(font_path)
            result.path_registered = True
            if families:
                font_family = font_family or families[0]

        if not font_family:
            result.error = "No font family to validate."
            return result

        result.family_exists = family_exists(font_family)
        if not result.family_exists:
            result.warnings.append(f"Family not installed: {font_family}")
            return result

        result.resolved_family = font_family
        result.covers_sample = font_covers_text(font_family, sample_text)
        if not result.covers_sample:
            result.warnings.append(
                f"'{font_family}' lacks glyphs for sample: {sample_text!r}"
            )
        result.ok = result.covers_sample
        return result
    except Exception as e:
        result.error = str(e)
        return result


def font_info(family: str) -> Dict[str, Any]:
    """Basic information about an installed family."""
    _ensure_font_runtime()
    exists = family_exists(family)
    info: Dict[str, Any] = {"family": family, "exists": exists}
    if exists:
        try:
            info["styles"] = [str(s) for s in QFontDatabase.styles(family)]
            info["scalable"] = bool(QFontDatabase.isScalable(family))
        except Exception:
            pass
    return info


def coverage_report(text: str) -> Dict[str, Any]:
    """
    Per-run font resolution for ``text`` plus the whole-string choice.
    Handy for the comparative-evaluation section of the paper.
    """
    return {
        "text": str(text),
        "dominant_script": detect_script(text),
        "whole_string_family": auto_font_fallback(text),
        "runs": resolve_fonts_for_runs(text),
    }

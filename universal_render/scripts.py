# universal_render/scripts.py
"""
Unicode script detection, mixed-text run segmentation, and script metadata.

This module is the script-awareness core of universal_render:

- detect_script(text)      -> dominant script name for a string
- segment_runs(text)       -> list of TextRun(text, script) for mixed content
- script_direction(script) -> "ltr" or "rtl"
- to_native_numerals(...)  -> digit conversion for scripts with native digits
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------
# Script range table
# ---------------------------------------------------------------------
# (start, end_inclusive, script). Kept to the major blocks needed for
# scientific figure text; unknown codepoints fall back to "Common".

_SCRIPT_RANGES: List[Tuple[int, int, str]] = [
    # Basic Latin letters (digits/punct handled as Common below)
    (0x0041, 0x005A, "Latin"),
    (0x0061, 0x007A, "Latin"),
    (0x00C0, 0x024F, "Latin"),
    (0x1E00, 0x1EFF, "Latin"),
    (0x2C60, 0x2C7F, "Latin"),
    (0xA720, 0xA7FF, "Latin"),

    (0x0370, 0x03FF, "Greek"),
    (0x1F00, 0x1FFF, "Greek"),

    (0x0400, 0x052F, "Cyrillic"),
    (0x2DE0, 0x2DFF, "Cyrillic"),
    (0xA640, 0xA69F, "Cyrillic"),

    (0x0530, 0x058F, "Armenian"),

    (0x0590, 0x05FF, "Hebrew"),
    (0xFB1D, 0xFB4F, "Hebrew"),

    (0x0600, 0x06FF, "Arabic"),
    (0x0750, 0x077F, "Arabic"),
    (0x08A0, 0x08FF, "Arabic"),
    (0xFB50, 0xFDFF, "Arabic"),
    (0xFE70, 0xFEFF, "Arabic"),

    (0x0900, 0x097F, "Devanagari"),
    (0xA8E0, 0xA8FF, "Devanagari"),

    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Odia"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),

    (0x0E00, 0x0E7F, "Thai"),
    (0x0E80, 0x0EFF, "Lao"),
    (0x0F00, 0x0FFF, "Tibetan"),

    (0x1000, 0x109F, "Myanmar"),
    (0xA9E0, 0xA9FF, "Myanmar"),
    (0xAA60, 0xAA7F, "Myanmar"),

    (0x10A0, 0x10FF, "Georgian"),
    (0x1200, 0x139F, "Ethiopic"),
    (0x2D80, 0x2DDF, "Ethiopic"),
    (0xAB00, 0xAB2F, "Ethiopic"),

    (0x1780, 0x17FF, "Khmer"),
    (0x19E0, 0x19FF, "Khmer"),

    (0x3040, 0x309F, "Hiragana"),
    (0x30A0, 0x30FF, "Katakana"),
    (0x31F0, 0x31FF, "Katakana"),

    (0x3400, 0x4DBF, "Han"),
    (0x4E00, 0x9FFF, "Han"),
    (0xF900, 0xFAFF, "Han"),

    (0x1100, 0x11FF, "Hangul"),
    (0x3130, 0x318F, "Hangul"),
    (0xA960, 0xA97F, "Hangul"),
    (0xAC00, 0xD7FF, "Hangul"),

    # Combining marks that inherit the script of their base character
    (0x0300, 0x036F, "Inherited"),
    (0x1AB0, 0x1AFF, "Inherited"),
    (0x20D0, 0x20FF, "Inherited"),
    (0xFE00, 0xFE0F, "Inherited"),
]

_SCRIPT_RANGES.sort(key=lambda r: r[0])
_RANGE_STARTS = [r[0] for r in _SCRIPT_RANGES]


COMMON = "Common"
INHERITED = "Inherited"

#: Scripts written right-to-left.
RTL_SCRIPTS = frozenset({"Arabic", "Hebrew"})

#: All script names this package knows about.
KNOWN_SCRIPTS = sorted({r[2] for r in _SCRIPT_RANGES} | {COMMON})


# ---------------------------------------------------------------------
# Native digit tables
# ---------------------------------------------------------------------

_DIGIT_SETS: Dict[str, str] = {
    "Bengali":    "০১২৩৪৫৬৭৮৯",
    "Devanagari": "०१२३४५६७८९",
    "Gurmukhi":   "੦੧੨੩੪੫੬੭੮੯",
    "Gujarati":   "૦૧૨૩૪૫૬૭૮૯",
    "Odia":       "୦୧୨୩୪୫୬୭୮୯",
    "Tamil":      "௦௧௨௩௪௫௬௭௮௯",
    "Telugu":     "౦౧౨౩౪౫౬౭౮౯",
    "Kannada":    "೦೧೨೩೪೫೬೭೮೯",
    "Malayalam":  "൦൧൨൩൪൫൬൭൮൯",
    "Arabic":     "٠١٢٣٤٥٦٧٨٩",
    "Persian":    "۰۱۲۳۴۵۶۷۸۹",
    "Thai":       "๐๑๒๓๔๕๖๗๘๙",
    "Lao":        "໐໑໒໓໔໕໖໗໘໙",
    "Myanmar":    "၀၁၂၃၄၅၆၇၈၉",
    "Khmer":      "០១២៣៤៥៦៧៨៩",
    "Tibetan":    "༠༡༢༣༤༥༦༧༨༩",
}

_DIGIT_TRANSLATIONS: Dict[str, Dict[int, str]] = {
    script: {ord(str(i)): digits[i] for i in range(10)}
    for script, digits in _DIGIT_SETS.items()
}


def scripts_with_native_digits() -> List[str]:
    """Scripts for which to_native_numerals() can convert 0-9."""
    return sorted(_DIGIT_TRANSLATIONS)


def to_native_numerals(value, script: str) -> str:
    """
    Convert ASCII digits in ``value`` to the native digits of ``script``.

    Scripts without a native digit table (e.g. Latin, Han) return the
    string unchanged.

        >>> to_native_numerals(2026, "Bengali")
        '২০২৬'
    """
    table = _DIGIT_TRANSLATIONS.get(script)
    s = str(value)
    if table is None:
        return s
    return s.translate(table)


# ---------------------------------------------------------------------
# Per-character detection
# ---------------------------------------------------------------------

def detect_char_script(ch: str) -> str:
    """
    Return the script name for a single character.

    Digits, punctuation, whitespace, and symbols return "Common";
    combining marks return "Inherited".
    """
    cp = ord(ch[0])
    idx = bisect_right(_RANGE_STARTS, cp) - 1
    if idx >= 0:
        start, end, script = _SCRIPT_RANGES[idx]
        if start <= cp <= end:
            return script
    return COMMON


# ---------------------------------------------------------------------
# String-level detection
# ---------------------------------------------------------------------

def detect_scripts(text: str) -> Dict[str, int]:
    """
    Count characters per script in ``text``, excluding Common/Inherited.
    """
    counts: Counter = Counter()
    for ch in str(text):
        script = detect_char_script(ch)
        if script not in (COMMON, INHERITED):
            counts[script] += 1
    return dict(counts)


def detect_script(text: str, default: str = "Latin") -> str:
    """
    Return the dominant script of ``text``.

    The dominant script is the one with the most characters, ignoring
    Common (digits, punctuation, spaces) and Inherited (combining marks)
    characters. If no script characters are present at all — e.g. the
    text is purely numeric — ``default`` is returned.
    """
    counts = detect_scripts(text)
    if not counts:
        return default
    return max(counts.items(), key=lambda kv: kv[1])[0]


def script_direction(script: str) -> str:
    """Return "rtl" for right-to-left scripts, else "ltr"."""
    return "rtl" if script in RTL_SCRIPTS else "ltr"


def is_rtl_text(text: str) -> bool:
    """True when the dominant script of ``text`` is right-to-left."""
    return script_direction(detect_script(text)) == "rtl"


def is_mixed_script(text: str) -> bool:
    """True when ``text`` contains characters from 2+ scripts."""
    return len(detect_scripts(text)) >= 2


# ---------------------------------------------------------------------
# Run segmentation for mixed text
# ---------------------------------------------------------------------

@dataclass
class TextRun:
    """A maximal substring whose characters share one script."""
    text: str
    script: str
    start: int  # index into the original string
    end: int    # exclusive

    def __len__(self) -> int:
        return self.end - self.start


def segment_runs(text: str, default: str = "Latin") -> List[TextRun]:
    """
    Split ``text`` into maximal same-script runs.

    Common characters (spaces, digits, punctuation) and Inherited marks
    are merged into the preceding run so that "বাংলা 2026 data" yields
    runs ["বাংলা 2026 ", "data"] rather than five fragments. A leading
    Common run attaches to the first real script that follows; a string
    with no script characters at all becomes a single ``default`` run.
    """
    text = str(text)
    if not text:
        return []

    # First pass: raw per-character scripts with Common/Inherited resolved
    # to the previous concrete script when possible.
    raw: List[str] = []
    prev: Optional[str] = None
    for ch in text:
        script = detect_char_script(ch)
        if script in (COMMON, INHERITED):
            raw.append(prev if prev is not None else COMMON)
        else:
            raw.append(script)
            prev = script

    # Resolve any leading Common region to the first concrete script.
    first_concrete = next((s for s in raw if s != COMMON), default)
    resolved = [first_concrete if s == COMMON else s for s in raw]

    # Second pass: group consecutive equal scripts into runs.
    runs: List[TextRun] = []
    run_start = 0
    for i in range(1, len(resolved) + 1):
        if i == len(resolved) or resolved[i] != resolved[run_start]:
            runs.append(TextRun(
                text=text[run_start:i],
                script=resolved[run_start],
                start=run_start,
                end=i,
            ))
            run_start = i
    return runs


def describe_text(text: str) -> Dict[str, object]:
    """
    Diagnostic summary of a string: dominant script, direction,
    mixed-script flag, and the segmented runs. Useful for debugging
    and for reporting in evaluation experiments.
    """
    dominant = detect_script(text)
    return {
        "text": str(text),
        "dominant_script": dominant,
        "direction": script_direction(dominant),
        "is_mixed": is_mixed_script(text),
        "script_counts": detect_scripts(text),
        "runs": [
            {"text": r.text, "script": r.script, "start": r.start, "end": r.end}
            for r in segment_runs(text)
        ],
    }

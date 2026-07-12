# universal_render/diagnostics.py
"""
Self-testing, benchmarking, and evaluation figure generation.

- self_test()                : does each script actually render here?
- benchmark_render()         : cold/warm render latency statistics
- save_comparison_figure()   : side-by-side default-Matplotlib vs
                               universal_render figure (paper-ready)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .fonts import auto_font_fallback, font_covers_text
from .renderer import (
    clear_render_cache,
    render_text_array,
)
from .scripts import detect_script, script_direction


#: Known-tricky sample words per script: conjuncts, reordering vowels,
#: contextual joining, stacked marks. Used by self_test() and the
#: comparison figure.
SCRIPT_TEST_SAMPLES: Dict[str, str] = {
    "Bengali": "বিশ্ববিদ্যালয়ের শিক্ষার্থী",
    "Devanagari": "विश्वविद्यालय क्षेत्र",
    "Tamil": "பல்கலைக்கழகம்",
    "Telugu": "విశ్వవిద్యాలయం",
    "Kannada": "ವಿಶ್ವವಿದ್ಯಾಲಯ",
    "Malayalam": "സർവ്വകലാശാല",
    "Gujarati": "યુનિવર્સિટી",
    "Gurmukhi": "ਯੂਨੀਵਰਸਿਟੀ",
    "Odia": "ବିଶ୍ୱବିଦ୍ୟାଳୟ",
    "Sinhala": "විශ්වවිද්‍යාලය",
    "Arabic": "جامعة القاهرة",
    "Hebrew": "אוניברסיטה",
    "Thai": "มหาวิทยาลัย",
    "Lao": "ມະຫາວິທະຍາໄລ",
    "Khmer": "សាកលវិទ្យាល័យ",
    "Myanmar": "တက္ကသိုလ်",
    "Han": "大学研究",
    "Hangul": "대학교 연구",
    "Hiragana": "だいがく",
    "Greek": "Πανεπιστήμιο",
    "Cyrillic": "Университет",
    "Latin": "University",
}


# ---------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------

def _ink_pixels(arr: np.ndarray) -> int:
    """Count non-transparent pixels in an RGBA uint8 array."""
    return int((arr[:, :, 3] > 0).sum())


def self_test(
    scripts: Optional[Sequence[str]] = None,
    font_size: int = 32,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Render a known-tricky sample for each script and report whether this
    environment can display it.

    For each script: the resolved font, whether that font actually has
    the glyphs (coverage), whether the render produced visible ink, and
    a pass/fail flag. Returns {script: report} and prints a table when
    ``verbose``.
    """
    chosen = list(scripts) if scripts else list(SCRIPT_TEST_SAMPLES)
    results: Dict[str, Dict[str, Any]] = {}

    for script in chosen:
        sample = SCRIPT_TEST_SAMPLES.get(script)
        if sample is None:
            results[script] = {"ok": False, "error": "no test sample defined"}
            continue

        report: Dict[str, Any] = {"sample": sample}
        try:
            report["detected_script"] = detect_script(sample)
            report["direction"] = script_direction(script)
            family = auto_font_fallback(sample)
            report["font_family"] = family
            report["glyph_coverage"] = font_covers_text(family, sample)
            arr = render_text_array(sample, font_size=font_size)
            ink = _ink_pixels(arr)
            report["image_size"] = (arr.shape[1], arr.shape[0])
            report["ink_pixels"] = ink
            report["rendered_nonempty"] = ink > 0
            report["ok"] = bool(
                report["detected_script"] == script
                and report["glyph_coverage"]
                and report["rendered_nonempty"]
            )
        except Exception as e:
            report["ok"] = False
            report["error"] = str(e)
        results[script] = report

    if verbose:
        print(f"{'script':<12} {'ok':<4} {'font':<24} {'coverage':<9} {'ink_px':>8}")
        print("-" * 62)
        for script, r in results.items():
            print(
                f"{script:<12} "
                f"{'yes' if r.get('ok') else 'NO ':<4} "
                f"{str(r.get('font_family', '-')):<24} "
                f"{str(r.get('glyph_coverage', '-')):<9} "
                f"{r.get('ink_pixels', 0):>8}"
            )
        n_ok = sum(1 for r in results.values() if r.get("ok"))
        print("-" * 62)
        print(f"{n_ok}/{len(results)} scripts render correctly in this environment")

    return results


# ---------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------

def benchmark_render(
    texts: Optional[Sequence[str]] = None,
    font_size: int = 32,
    repeats: int = 20,
) -> Dict[str, Any]:
    """
    Measure render latency, cold (cache cleared each call) and warm
    (cache hits). Returns per-text and aggregate statistics in
    milliseconds — the raw material for the paper's performance table.
    """
    if texts is None:
        texts = list(SCRIPT_TEST_SAMPLES.values())

    per_text: List[Dict[str, Any]] = []

    for text in texts:
        cold: List[float] = []
        for _ in range(repeats):
            clear_render_cache()
            t0 = time.perf_counter()
            render_text_array(text, font_size=font_size)
            cold.append((time.perf_counter() - t0) * 1000.0)

        # Prime the cache once, then measure hits
        render_text_array(text, font_size=font_size)
        warm: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            render_text_array(text, font_size=font_size)
            warm.append((time.perf_counter() - t0) * 1000.0)

        per_text.append({
            "text": text,
            "script": detect_script(text),
            "cold_ms_mean": float(np.mean(cold)),
            "cold_ms_std": float(np.std(cold)),
            "warm_ms_mean": float(np.mean(warm)),
            "warm_ms_std": float(np.std(warm)),
        })

    return {
        "font_size": font_size,
        "repeats": repeats,
        "per_text": per_text,
        "cold_ms_mean_overall": float(np.mean([r["cold_ms_mean"] for r in per_text])),
        "warm_ms_mean_overall": float(np.mean([r["warm_ms_mean"] for r in per_text])),
    }


# ---------------------------------------------------------------------
# Before/after comparison figure
# ---------------------------------------------------------------------

def save_comparison_figure(
    output_path: str,
    samples: Optional[Dict[str, str]] = None,
    font_size: int = 30,
    dpi: int = 200,
    left_title: str = "Matplotlib default",
    right_title: str = "universal_render",
):
    """
    Produce a two-column figure: each sample rendered by Matplotlib's
    own text engine (left) and by universal_render (right). This is the
    before/after evidence figure for each script family.

    ``samples`` maps a row label to the text, defaulting to
    SCRIPT_TEST_SAMPLES. Returns the Matplotlib figure.

    Matplotlib emits "glyph missing" warnings for the left panel; that
    is the defect being demonstrated, so those warnings are suppressed.
    """
    import warnings

    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    if samples is None:
        samples = dict(SCRIPT_TEST_SAMPLES)

    n = len(samples)
    fig_h = max(2.0, 0.62 * n + 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, fig_h))

    for ax, title in zip(axes, (left_title, right_title)):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, n)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    for i, (label, text) in enumerate(samples.items()):
        y = n - i - 0.5

        # Row label on the left margin of the left panel
        axes[0].text(
            -0.02, y, label, fontsize=10, ha="right", va="center",
            transform=axes[0].transData, clip_on=False,
        )

        # Left: Matplotlib's own text engine (typically broken shaping)
        axes[0].text(
            0.5, y, text, fontsize=font_size * 0.55,
            ha="center", va="center",
        )

        # Right: Qt-shaped image artist
        arr = render_text_array(text, font_size=font_size) / 255.0
        oi = OffsetImage(arr, zoom=0.5)
        ab = AnnotationBbox(
            oi, (0.5, y), frameon=False, box_alignment=(0.5, 0.5),
            annotation_clip=False,
        )
        axes[1].add_artist(ab)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Gg]lyph.*missing.*")
        warnings.filterwarnings("ignore", message=".*does not support.*")
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return fig

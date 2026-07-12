"""
Comparative evaluation: shaping correctness, missing glyphs, and latency
across text-rendering backends (Phase 3 of the paper).

Design
------
For each script sample, every backend renders the SAME text with the
SAME font file. The font is therefore held constant and the only free
variable is the backend's text-shaping engine:

- universal_render (Qt/HarfBuzz)      <- also serves as the shaping reference
- matplotlib-agg, default font        <- what users get out of the box
- matplotlib-agg, correct font        <- font installed, no shaping engine
- mplcairo, correct font              <- strongest baseline (Raqm shaping)

Metrics per (script, backend):
- warnings   : Matplotlib "missing glyph" warnings emitted during draw
- ink_iou    : ink-mask IoU vs the HarfBuzz reference (size-normalized)
- ssim       : structural similarity vs the reference (size-normalized)
- latency_ms : mean wall-clock time to produce the text image

Outputs
-------
- results_backend_comparison.csv          (one row per script x backend)
- output/cmp_<script>_<backend>.png       (individual renders)
Run:  py evaluation/compare_backends.py
"""
import io
import os
import sys
import time
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import universal_render as ur

try:
    from mplcairo.base import FigureCanvasCairo
    MPLCAIRO_OK = True
except Exception as e:
    FigureCanvasCairo = None
    MPLCAIRO_OK = False
    print(f"note: mplcairo unavailable ({e})")

from skimage.metrics import structural_similarity
from skimage.transform import resize

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

FONT_SIZE_PT = 36
REPEATS = 10

SAMPLES = {
    "Bengali": "বিশ্ববিদ্যালয়ের শিক্ষার্থী",
    "Devanagari": "विश्वविद्यालय क्षेत्र",
    "Tamil": "பல்கலைக்கழகம்",
    "Arabic": "جامعة القاهرة",
    "Thai": "มหาวิทยาลัย",
    "Latin": "University",
}


# ---------------------------------------------------------------------
# Rendering helpers — each returns (rgba uint8 array, n_glyph_warnings)
# ---------------------------------------------------------------------

def _grab_canvas(fig, canvas) -> np.ndarray:
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    import matplotlib.image as mpimg
    img = mpimg.imread(buf)  # float 0..1 RGBA
    return (img * 255).astype(np.uint8)


def render_mpl(text, font_prop, canvas_cls) -> tuple:
    fig = Figure(figsize=(6, 1.6), dpi=100)
    canvas = canvas_cls(fig)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.5, text, ha="center", va="center",
             fontsize=FONT_SIZE_PT, fontproperties=font_prop)
    n_warn = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        arr = _grab_canvas(fig, canvas)
        for w in caught:
            msg = str(w.message).lower()
            if "glyph" in msg and "missing" in msg:
                n_warn += 1
            elif "does not support" in msg:
                n_warn += 1
    return arr, n_warn


def render_ur(text, family) -> tuple:
    arr = ur.render_text_array(text, font_family=family,
                               font_size=int(FONT_SIZE_PT * 100 / 72))
    # composite onto white so all backends compare on equal ground
    rgb = arr[:, :, :3].astype(np.float32)
    a = (arr[:, :, 3:4].astype(np.float32)) / 255.0
    white = np.full_like(rgb, 255.0)
    out = (rgb * a + white * (1 - a)).astype(np.uint8)
    rgba = np.dstack([out, np.full(out.shape[:2], 255, np.uint8)])
    return rgba, 0


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def _ink_mask(rgba: np.ndarray) -> np.ndarray:
    """Boolean ink mask: anything darker than near-white."""
    gray = rgba[:, :, :3].astype(np.float32).mean(axis=2)
    return gray < 240.0


def _trim(mask: np.ndarray, img: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return img[y0:y1, x0:x1]


def _norm_gray(rgba: np.ndarray, shape) -> np.ndarray:
    gray = rgba[:, :, :3].astype(np.float32).mean(axis=2) / 255.0
    trimmed = _trim(_ink_mask(rgba), gray)
    if trimmed is None or trimmed.size == 0:
        return np.ones(shape, dtype=np.float32)
    return resize(trimmed, shape, anti_aliasing=True).astype(np.float32)


def compare_to_reference(ref_rgba, test_rgba, shape=(64, 512)):
    ref = _norm_gray(ref_rgba, shape)
    test = _norm_gray(test_rgba, shape)
    ssim = float(structural_similarity(ref, test, data_range=1.0))
    ref_ink, test_ink = ref < 0.94, test < 0.94
    inter = np.logical_and(ref_ink, test_ink).sum()
    union = np.logical_or(ref_ink, test_ink).sum()
    iou = float(inter / union) if union else 0.0
    return ssim, iou


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ur.init_renderer()
    rows = []

    for script, text in SAMPLES.items():
        family = ur.auto_font_fallback(text)
        try:
            font_file = findfont(FontProperties(family=family),
                                 fallback_to_default=False)
        except Exception:
            font_file = None
        correct_prop = (FontProperties(fname=font_file, size=FONT_SIZE_PT)
                        if font_file else FontProperties(size=FONT_SIZE_PT))

        backends = {}

        # Reference + our framework (same thing here, by construction)
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            ur.clear_render_cache()
            ref_rgba, _ = render_ur(text, family)
        ur_ms = (time.perf_counter() - t0) * 1000 / REPEATS
        backends["universal_render"] = (ref_rgba, 0, ur_ms)

        # Matplotlib default font
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            arr, nw = render_mpl(text, FontProperties(size=FONT_SIZE_PT),
                                 FigureCanvasAgg)
        ms = (time.perf_counter() - t0) * 1000 / REPEATS
        backends["mpl_default"] = (arr, nw, ms)

        # Matplotlib with the CORRECT font file (no shaping engine)
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            arr, nw = render_mpl(text, correct_prop, FigureCanvasAgg)
        ms = (time.perf_counter() - t0) * 1000 / REPEATS
        backends["mpl_correct_font"] = (arr, nw, ms)

        # mplcairo with the correct font (Raqm shaping)
        if MPLCAIRO_OK:
            t0 = time.perf_counter()
            for _ in range(REPEATS):
                arr, nw = render_mpl(text, correct_prop, FigureCanvasCairo)
            ms = (time.perf_counter() - t0) * 1000 / REPEATS
            backends["mplcairo"] = (arr, nw, ms)

        ref_rgba = backends["universal_render"][0]
        for name, (arr, nw, ms) in backends.items():
            ssim, iou = compare_to_reference(ref_rgba, arr)
            rows.append({
                "script": script, "backend": name, "font": family,
                "glyph_warnings": nw,
                "ssim_vs_ref": round(ssim, 3),
                "ink_iou_vs_ref": round(iou, 3),
                "latency_ms": round(ms, 2),
            })
            import matplotlib.image as mpimg
            mpimg.imsave(
                os.path.join(OUT_DIR, f"cmp_{script}_{name}.png"), arr)

    # ---- report ------------------------------------------------------
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results_backend_comparison.csv")
    import csv as _csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'script':<12} {'backend':<18} {'warn':>4} {'ssim':>6} "
          f"{'iou':>6} {'ms':>8}")
    print("-" * 60)
    for r in rows:
        print(f"{r['script']:<12} {r['backend']:<18} "
              f"{r['glyph_warnings']:>4} {r['ssim_vs_ref']:>6} "
              f"{r['ink_iou_vs_ref']:>6} {r['latency_ms']:>8}")
    print(f"\nCSV: {csv_path}\nPNGs: {OUT_DIR}")


if __name__ == "__main__":
    main()

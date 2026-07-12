"""
Six-script verification demo for universal_render.

Renders the roadmap's minimal evaluation set — Bengali, Hindi, Tamil,
Arabic, Thai, English — as (1) standalone word renders, (2) a line plot
with multilingual labels, and (3) a heatmap with in-cell text per script.
Outputs go to examples/output/.
"""
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import universal_render as ur

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

# text: (label, sample word/phrase)
SAMPLES = {
    "Bengali": "বিশ্ববিদ্যালয়",       # conjuncts + vowel signs
    "Devanagari": "विश्वविद्यालय",     # Hindi
    "Tamil": "பல்கலைக்கழகம்",
    "Arabic": "جامعة القاهرة",          # RTL + contextual joining
    "Thai": "มหาวิทยาลัย",              # combining marks, no spaces
    "Latin": "University 2026",
}


def demo_word_renders():
    print("--- word renders + script detection ---")
    for script, word in SAMPLES.items():
        detected = ur.detect_script(word)
        family = ur.auto_font_fallback(word)
        path = os.path.join(OUT, f"word_{script}.png")
        ur.render_text(word, output_path=path, font_size=48)
        status = "OK " if detected == script else "?? "
        print(f"{status} {script:<12} detected={detected:<12} font={family:<22} -> {os.path.basename(path)}")


def demo_line_plot():
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(1, 7)
    ax.plot(x, [3, 5, 4, 6, 5, 7], marker="o")

    ur.set_multilingual_title(ax, "বহুভাষিক প্লট — многоязычный", font_size=34)
    ur.set_multilingual_xlabel(ax, "أشهر السنة", font_size=26)          # Arabic
    ur.set_multilingual_ylabel(ax, "மாத விற்பனை", font_size=26)         # Tamil
    ur.set_multilingual_xticks(
        ax, list(x),
        ["এক", "दो", "மூன்று", "أربعة", "ห้า", "six"],                    # 6 scripts
        font_size=18,
    )
    ur.apply_multilingual_layout(fig, auto=True)

    path = os.path.join(OUT, "line_plot_six_scripts.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"OK  line plot -> {os.path.basename(path)}")


def demo_heatmap():
    words = np.array([
        ["খুশি", "खुशी", "மகிழ்ச்சி"],
        ["فرح", "ความสุข", "joy"],
        ["দুঃখ", "दुख", "சோகம்"],
    ])
    data = np.random.default_rng(7).random(words.shape)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(data, cmap="viridis")
    rows, cols = data.shape
    for i in range(rows):
        for j in range(cols):
            ur.add_multilingual_cell_text(
                ax, i, j, words[i, j], rows=rows, cols=cols,
                font_size=30, color="white",
            )
    ax.set_xticks(range(cols)); ax.set_xticklabels([])
    ax.set_yticks(range(rows)); ax.set_yticklabels([])
    ur.set_multilingual_title(ax, "Emotion — আবেগ — عاطفة", font_size=30)
    fig.colorbar(im)

    path = os.path.join(OUT, "heatmap_six_scripts.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"OK  heatmap -> {os.path.basename(path)}")


def demo_mixed_and_rtl():
    print("--- mixed-script diagnostics ---")
    mixed = "ঢাকা Dhaka ২০২৬ (2026)"
    info = ur.describe_text(mixed)
    print(f"text={mixed!r}")
    print(f"dominant={info['dominant_script']} mixed={info['is_mixed']} runs={len(info['runs'])}")
    for r in info["runs"]:
        print(f"   run {r['script']:<10} {r['text']!r}")
    ur.render_text(mixed, output_path=os.path.join(OUT, "mixed_bengali_english.png"), font_size=40)

    rtl = "القاهرة Cairo 2026"
    print(f"rtl check: {rtl!r} -> is_rtl={ur.is_rtl_text(rtl)}")
    ur.render_text(rtl, output_path=os.path.join(OUT, "mixed_arabic_english.png"), font_size=40)

    print("native numerals:",
          ur.to_native_numerals(2026, "Bengali"),
          ur.to_native_numerals(2026, "Devanagari"),
          ur.to_native_numerals(2026, "Arabic"),
          ur.to_native_numerals(2026, "Thai"))


if __name__ == "__main__":
    ur.init_renderer()
    demo_word_renders()
    demo_mixed_and_rtl()
    demo_line_plot()
    demo_heatmap()
    print(f"\nAll outputs in: {OUT}")

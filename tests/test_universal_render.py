"""
Test suite for universal_render.

Runs under pytest, or standalone:  py tests/test_universal_render.py
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


# ---------------------------------------------------------------------
# scripts.py
# ---------------------------------------------------------------------

def test_detect_script_per_family():
    cases = {
        "বাংলা": "Bengali",
        "हिन्दी": "Devanagari",
        "தமிழ்": "Tamil",
        "العربية": "Arabic",
        "עברית": "Hebrew",
        "ไทย": "Thai",
        "မြန်မာ": "Myanmar",
        "ខ្មែរ": "Khmer",
        "中文": "Han",
        "한국어": "Hangul",
        "Ελληνικά": "Greek",
        "Русский": "Cyrillic",
        "English": "Latin",
    }
    for text, expected in cases.items():
        got = ur.detect_script(text)
        assert got == expected, f"{text!r}: expected {expected}, got {got}"


def test_detect_script_numeric_fallback():
    assert ur.detect_script("123 + 456") == "Latin"
    assert ur.detect_script("", default="Bengali") == "Bengali"


def test_segment_runs_mixed():
    runs = ur.segment_runs("ঢাকা Dhaka ২০২৬")
    scripts = [r.script for r in runs]
    assert scripts == ["Bengali", "Latin", "Bengali"], scripts
    # Reassembling runs must reproduce the original string
    assert "".join(r.text for r in runs) == "ঢাকা Dhaka ২০২৬"


def test_segment_runs_leading_common():
    runs = ur.segment_runs("2026 সাল")
    assert runs[0].script == "Bengali"


def test_native_numerals():
    assert ur.to_native_numerals(2026, "Bengali") == "২০২৬"
    assert ur.to_native_numerals(2026, "Devanagari") == "२०२६"
    assert ur.to_native_numerals(2026, "Arabic") == "٢٠٢٦"
    assert ur.to_native_numerals("3.14", "Thai") == "๓.๑๔"
    assert ur.to_native_numerals(42, "Latin") == "42"  # unchanged


def test_rtl_detection():
    assert ur.is_rtl_text("العربية")
    assert ur.is_rtl_text("עברית")
    assert not ur.is_rtl_text("বাংলা")
    assert ur.script_direction("Arabic") == "rtl"
    assert ur.script_direction("Tamil") == "ltr"


def test_describe_text():
    # Bengali run is longer than the Latin run, so Bengali dominates
    info = ur.describe_text("ঢাকা বিভাগের Dhaka")
    assert info["dominant_script"] == "Bengali"
    assert info["is_mixed"] is True
    assert len(info["runs"]) == 2


# ---------------------------------------------------------------------
# highlevel.py: language names
# ---------------------------------------------------------------------

def test_language_to_script():
    assert ur.language_to_script("bangla") == "Bengali"
    assert ur.language_to_script("Hindi") == "Devanagari"
    assert ur.language_to_script("urdu") == "Arabic"
    assert ur.language_to_script("japanese") == "Hiragana"
    try:
        ur.language_to_script("klingon")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_localized_numerals():
    assert ur.localized_numerals(7, "bangla") == "৭"
    assert ur.localized_numerals(7, "hindi") == "७"


# ---------------------------------------------------------------------
# fonts.py
# ---------------------------------------------------------------------

def test_auto_font_fallback_covers_text():
    for text in ["বাংলা", "हिन्दी", "العربية", "ไทย"]:
        family = ur.auto_font_fallback(text)
        assert ur.family_exists(family), family
        assert ur.font_covers_text(family, text), (family, text)


def test_explicit_family_honored():
    assert ur.resolve_font(font_family="Arial") == "Arial"


def test_script_font_override():
    original = ur.get_script_font("Bengali")
    try:
        ur.set_script_font("Bengali", "Nirmala UI")
        assert ur.find_font_for_script("Bengali") == "Nirmala UI"
    finally:
        if original is None:
            ur.set_script_font("Bengali", "Nirmala UI")  # harmless reset
        else:
            ur.set_script_font("Bengali", original)


def test_register_font_file():
    font_file = r"C:\Users\mrina\Desktop\bangla-render\NotoSansBengali-Regular.ttf"
    if not os.path.isfile(font_file):
        return  # environment-specific; skip silently
    families = ur.register_font(font_file)
    assert any("Bengali" in f for f in families), families


def test_coverage_report():
    report = ur.coverage_report("ঢাকা বিভাগের Dhaka")
    assert report["dominant_script"] == "Bengali"
    assert len(report["runs"]) == 2
    for run in report["runs"]:
        assert run["family"]


# ---------------------------------------------------------------------
# renderer.py
# ---------------------------------------------------------------------

def _ink(arr):
    return int((arr[:, :, 3] > 0).sum())


def test_render_produces_ink():
    from universal_render.renderer import render_text_array
    for text in ["বিশ্ববিদ্যালয়", "جامعة", "มหาวิทยาลัย", "University"]:
        arr = render_text_array(text, font_size=32)
        assert _ink(arr) > 50, text


def test_render_cache():
    from universal_render.renderer import render_text_array
    ur.clear_render_cache()
    render_text_array("ক্যাশ পরীক্ষা", font_size=24)
    size_after_first = ur.get_render_cache_info()["size"]
    render_text_array("ক্যাশ পরীক্ষা", font_size=24)
    assert ur.get_render_cache_info()["size"] == size_after_first >= 1


def test_measure_text():
    m = ur.measure_text("পরিমাপ", font_size=30)
    assert m["text_width_px"] > 0
    assert m["text_height_px"] > 0
    assert m["font_family"]


def test_render_mixed_text_and_paragraph():
    qimg = ur.render_mixed_text("ঢাকা Dhaka ২০২৬ (2026)", font_size=28)
    assert qimg.width() > 0 and qimg.height() > 0
    # trim=False keeps the requested canvas width exactly
    qimg2 = ur.render_paragraph("م" * 5 + " " + "test " * 10, width=300,
                                font_size=20, trim=False)
    assert qimg2.width() == 300


# ---------------------------------------------------------------------
# Matplotlib API
# ---------------------------------------------------------------------

def test_mpl_label_functions():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [3, 1, 4])
    assert ur.set_multilingual_title(ax, "শিরোনাম") is not None
    assert ur.set_multilingual_xlabel(ax, "أشهر") is not None
    assert ur.set_multilingual_ylabel(ax, "விற்பனை") is not None
    assert ur.set_multilingual_xticks(ax, [1, 2, 3], ["এক", "दो", "மூன்று"]) is not None
    t = ur.multilingual_text(ax, 0.5, 0.5, "মাঝে", coord="axes")
    assert t is not None
    plt.close(fig)


def test_localize_axes_one_call():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3], label="a")
    artists = ur.localize_axes(
        ax,
        title="বিক্রয়",
        xlabel="महीना",
        ylabel="மதிப்பு",
        xticklabels=["এক", "দুই", "তিন"],
        legend_labels=["ধারা"],
        auto_layout=False,
    )
    for key in ("title", "xlabel", "ylabel", "xticks", "legend"):
        assert key in artists, key
    plt.close(fig)


def test_multilingual_heatmap_one_call():
    fig, ax = plt.subplots()
    data = np.arange(9, dtype=float).reshape(3, 3)
    labels = np.array([["ক", "ख", "க"], ["ا", "ก", "a"], ["গ", "ग", "௧"]])
    im, artists = ur.multilingual_heatmap(
        ax, data, labels=labels,
        row_labels=["সারি১", "সারি২", "সারি৩"],
        col_labels=["স্তম্ভ১", "স্তম্ভ২", "স্তম্ভ৩"],
        title="হিটম্যাপ",
        colorbar=False,
    )
    assert len(artists["cells"]) == 9
    plt.close(fig)


def test_multilingual_heatmap_value_format():
    fig, ax = plt.subplots()
    data = np.array([[0.1, 0.9], [0.5, 0.3]])
    im, artists = ur.multilingual_heatmap(
        ax, data, value_format=".1f", value_script="Bengali", colorbar=False,
    )
    assert len(artists["cells"]) == 4
    plt.close(fig)


def test_bar_labels():
    fig, ax = plt.subplots()
    ax.bar([1, 2, 3], [3, 5, 2])
    artists = ur.multilingual_bar_labels(ax, ["তিন", "पाँच", "இரண்டு"])
    assert len(artists) == 3
    plt.close(fig)


def test_numeric_ticks_scripts():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    g = ur.set_multilingual_numeric_ticks(ax, script="Devanagari", axis="x",
                                          positions=[1, 2, 3])
    assert g is not None
    plt.close(fig)


# ---------------------------------------------------------------------
# matplotlib-parity styling
# ---------------------------------------------------------------------

def test_bold_and_italic_render_differently():
    from universal_render.renderer import render_text_array
    normal = render_text_array("গুরুত্বপূর্ণ", font_size=32)
    bold = render_text_array("গুরুত্বপূর্ণ", font_size=32, weight="bold")
    italic = render_text_array("Important", font_size=32, italic=True)
    plain = render_text_array("Important", font_size=32)
    assert _ink(bold) > _ink(normal)          # bold has more ink
    assert italic.shape != plain.shape or (italic != plain).any()


def test_alpha_reduces_opacity():
    from universal_render.renderer import render_text_array
    solid = render_text_array("আলফা", font_size=32)
    faint = render_text_array("আলফা", font_size=32, alpha=0.3)
    assert faint[:, :, 3].max() < solid[:, :, 3].max()


def test_rotation_changes_geometry():
    from universal_render.renderer import render_text_array
    flat = render_text_array("ঘোরানো লেখা", font_size=28)
    rot45 = render_text_array("ঘোরানো লেখা", font_size=28, rotation=45)
    rot90 = render_text_array("ঘোরানো লেখা", font_size=28, rotation=90)
    # 45° is taller than flat; 90° swaps width/height (± trim margin)
    assert rot45.shape[0] > flat.shape[0]
    assert abs(rot90.shape[0] - flat.shape[1]) <= 4
    assert abs(rot90.shape[1] - flat.shape[0]) <= 4


def test_multiline_text():
    from universal_render.renderer import render_text_array
    one = render_text_array("প্রথম লাইন", font_size=24)
    two = render_text_array("প্রথম লাইন\nদ্বিতীয় লাইন", font_size=24)
    assert two.shape[0] > one.shape[0] * 1.5


def test_suptitle_and_title_loc():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [3, 1, 4])
    t = ur.set_multilingual_title(ax, "বাম শিরোনাম", loc="left", weight="bold")
    s = ur.set_multilingual_suptitle(fig, "চিত্রের সুপারটাইটেল")
    assert t is not None and t.artist is not None
    assert s is not None and s.artist is not None
    # suptitle must sit above the title
    r = fig.canvas.get_renderer()
    assert s.artist.get_window_extent(r).y0 >= t.artist.get_window_extent(r).y0
    plt.close(fig)


def test_title_gap_above_axes():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [3, 1, 4])
    t = ur.set_multilingual_title(ax, "ফাঁক পরীক্ষা")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    title_bottom = t.artist.get_window_extent(r).y0
    axes_top = ax.get_window_extent(r).y1
    assert title_bottom - axes_top >= 10, (title_bottom, axes_top)
    plt.close(fig)


def test_rotated_ticks():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    g = ur.set_multilingual_xticks(
        ax, [1, 2, 3], ["দীর্ঘ লেবেল এক", "लंबा लेबल दो", "நீண்ட லேபிள்"],
        rotation=45,
    )
    assert g is not None and len(g.artists) > 0
    plt.close(fig)


# ---------------------------------------------------------------------
# compat layer
# ---------------------------------------------------------------------

def test_compat_layer():
    import universal_render.compat as br
    assert br.to_bangla_numerals(2026) == "২০২৬"
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    assert br.set_bangla_title(ax, "বাংলা শিরোনাম") is not None
    assert br.bangla_text(ax, 0.5, 0.5, "লেখা", coord="axes") is not None
    plt.close(fig)


# ---------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------

def test_self_test_minimal_set():
    results = ur.self_test(
        scripts=["Bengali", "Devanagari", "Tamil", "Arabic", "Thai", "Latin"],
        verbose=False,
    )
    failures = {s: r for s, r in results.items() if not r.get("ok")}
    assert not failures, failures


def test_benchmark_shape():
    bench = ur.benchmark_render(texts=["বাংলা", "English"], repeats=3)
    assert bench["cold_ms_mean_overall"] > 0
    assert bench["warm_ms_mean_overall"] <= bench["cold_ms_mean_overall"]
    assert len(bench["per_text"]) == 2


# ---------------------------------------------------------------------
# standalone runner
# ---------------------------------------------------------------------

if __name__ == "__main__":
    ur.init_renderer()
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL {name}: {e!r}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    sys.exit(1 if failed else 0)

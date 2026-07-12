# universal_render/__init__.py

"""
universal_render: universal complex-script text rendering for Matplotlib.

Text is shaped through Qt's text stack (HarfBuzz), so scripts that
Matplotlib's own text engine cannot shape — Bengali, Hindi, Tamil,
Arabic, Thai, and other complex scripts — render correctly in titles,
axis labels, tick labels, annotations, legends, and heatmap cells.
Fonts are chosen automatically from each string's dominant Unicode
script, with per-script fallback tables and glyph-coverage validation.

Typical usage:

    import universal_render as ur
    import matplotlib.pyplot as plt

    ur.init_renderer()

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [3, 1, 4])

    ur.set_multilingual_title(ax, "বাংলা শিরোনাম")     # Bengali
    ur.set_multilingual_xlabel(ax, "أشهر السنة")        # Arabic (RTL)
    ur.set_multilingual_ylabel(ax, "மாத விற்பனை")       # Tamil

    ur.set_multilingual_xticks(ax, [1, 2, 3], ["एक", "दो", "तीन"])  # Hindi
    ur.text(ax, 0.5, 0.5, "ข้อความ Mixed 2026", coord="axes")       # Thai + Latin

    ur.apply_multilingual_layout(fig, auto=True)
"""

__version__ = "0.1.0"

# ---------------------------------------------------------------------
# Backend / environment
# ---------------------------------------------------------------------

from .backend import (
    init_renderer,
    ensure_qt_application,
    get_renderer_status,
    check_environment,
    is_headless_environment,
    is_notebook_environment,
    is_colab_environment,
    is_kaggle_environment,
    reset_renderer_state,
)

# ---------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------

from .scripts import (
    KNOWN_SCRIPTS,
    RTL_SCRIPTS,
    TextRun,
    detect_char_script,
    detect_script,
    detect_scripts,
    describe_text,
    is_mixed_script,
    is_rtl_text,
    script_direction,
    scripts_with_native_digits,
    segment_runs,
    to_native_numerals,
)

# Roadmap-compatible alias
script_detection_auto = detect_script

# ---------------------------------------------------------------------
# Font management
# ---------------------------------------------------------------------

from .fonts import (
    SCRIPT_FONT_CANDIDATES,
    auto_font_fallback,
    coverage_report,
    family_exists,
    find_font_for_script,
    font_covers_text,
    font_info,
    get_default_font,
    get_script_font,
    list_available_fonts,
    list_registered_fonts,
    register_font,
    register_fonts,
    resolve_font,
    resolve_fonts_for_runs,
    set_default_font,
    set_script_font,
    validate_font,
)

# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------

from .renderer import (
    render_text,
    render_text_array,
    render_mixed_text,
    render_paragraph,
    render_text_qimage,
    render_paragraph_qimage,
    measure_text,
    clear_render_cache,
    get_render_cache_info,
    set_render_cache_maxsize,
    set_render_defaults,
    get_render_defaults,
)

# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------

from .layout import (
    get_layout_manager,
    clear_layout_manager,
)

# ---------------------------------------------------------------------
# Matplotlib-facing APIs
# ---------------------------------------------------------------------

from .mpl_support import (
    set_multilingual_legend,
    set_multilingual_numeric_ticks,
    set_multilingual_title,
    set_multilingual_xlabel,
    set_multilingual_ylabel,
    set_multilingual_suptitle,
    set_multilingual_xticks,
    set_multilingual_yticks,
    add_multilingual_cell_text,
    multilingual_text,
    annotate_multilingual,
    multilingual_paragraph,
    apply_multilingual_layout,
)

# ---------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------

from .highlevel import (
    LANGUAGE_TO_SCRIPT,
    font_for_language,
    language_to_script,
    localize_axes,
    localized_numerals,
    multilingual_bar_labels,
    multilingual_heatmap,
    set_language_font,
    supported_languages,
)

# ---------------------------------------------------------------------
# Diagnostics / evaluation
# ---------------------------------------------------------------------

from .diagnostics import (
    SCRIPT_TEST_SAMPLES,
    benchmark_render,
    save_comparison_figure,
    self_test,
)


def text(ax, *args, **kwargs):
    """
    Short alias for multilingual_text(), so you can write:

        import universal_render as ur
        ur.text(ax, 0.5, 0.5, "বাংলা / العربية / ไทย", coord="axes")
    """
    return multilingual_text(ax, *args, **kwargs)


__all__ = [
    "__version__",
    # backend
    "init_renderer",
    "ensure_qt_application",
    "get_renderer_status",
    "check_environment",
    "is_headless_environment",
    "is_notebook_environment",
    "is_colab_environment",
    "is_kaggle_environment",
    "reset_renderer_state",
    # scripts
    "KNOWN_SCRIPTS",
    "RTL_SCRIPTS",
    "TextRun",
    "detect_char_script",
    "detect_script",
    "detect_scripts",
    "describe_text",
    "is_mixed_script",
    "is_rtl_text",
    "script_direction",
    "script_detection_auto",
    "scripts_with_native_digits",
    "segment_runs",
    "to_native_numerals",
    # fonts
    "SCRIPT_FONT_CANDIDATES",
    "auto_font_fallback",
    "coverage_report",
    "family_exists",
    "find_font_for_script",
    "font_covers_text",
    "font_info",
    "get_default_font",
    "get_script_font",
    "list_available_fonts",
    "list_registered_fonts",
    "register_font",
    "register_fonts",
    "resolve_font",
    "resolve_fonts_for_runs",
    "set_default_font",
    "set_script_font",
    "validate_font",
    # renderer
    "render_text",
    "render_text_array",
    "render_mixed_text",
    "render_paragraph",
    "render_text_qimage",
    "render_paragraph_qimage",
    "measure_text",
    "clear_render_cache",
    "get_render_cache_info",
    "set_render_cache_maxsize",
    "set_render_defaults",
    "get_render_defaults",
    # layout
    "get_layout_manager",
    "clear_layout_manager",
    # mpl support
    "set_multilingual_legend",
    "set_multilingual_numeric_ticks",
    "set_multilingual_title",
    "set_multilingual_xlabel",
    "set_multilingual_ylabel",
    "set_multilingual_suptitle",
    "set_multilingual_xticks",
    "set_multilingual_yticks",
    "add_multilingual_cell_text",
    "multilingual_text",
    "annotate_multilingual",
    "multilingual_paragraph",
    "apply_multilingual_layout",
    "text",
    # high-level
    "LANGUAGE_TO_SCRIPT",
    "font_for_language",
    "language_to_script",
    "localize_axes",
    "localized_numerals",
    "multilingual_bar_labels",
    "multilingual_heatmap",
    "set_language_font",
    "supported_languages",
    # diagnostics
    "SCRIPT_TEST_SAMPLES",
    "benchmark_render",
    "save_comparison_figure",
    "self_test",
]

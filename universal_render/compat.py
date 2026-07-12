# universal_render/compat.py
"""
bangla_render compatibility layer.

Existing bangla-render code keeps working after switching the import:

    # old:  import bangla_render as br
    import universal_render.compat as br

    br.set_bangla_title(ax, "বাংলা শিরোনাম")
    br.set_bangla_xticks(ax, [1, 2], ["এক", "দুই"])

Every set_bangla_* name maps to its set_multilingual_* equivalent; the
behavior is identical for Bengali text (same Qt pipeline, same layout
manager), with automatic script-aware font fallback as a bonus.
"""
from __future__ import annotations

from .backend import (  # noqa: F401
    check_environment,
    ensure_qt_application,
    get_renderer_status,
    init_renderer,
    is_colab_environment,
    is_headless_environment,
    is_kaggle_environment,
    is_notebook_environment,
    reset_renderer_state,
)
from .fonts import (  # noqa: F401
    get_default_font,
    list_available_fonts,
    list_registered_fonts,
    register_font,
    register_fonts,
    resolve_font,
    set_default_font,
    validate_font,
    font_info,
    SCRIPT_FONT_CANDIDATES,
)
from .renderer import (  # noqa: F401
    clear_render_cache,
    get_render_cache_info,
    get_render_defaults,
    measure_text,
    render_paragraph,
    render_paragraph_qimage,
    render_text,
    render_text_qimage,
    set_render_cache_maxsize,
    set_render_defaults,
)
from .layout import (  # noqa: F401
    clear_layout_manager,
    get_layout_manager,
)
from .scripts import to_native_numerals
from .mpl_support import (
    add_multilingual_cell_text,
    annotate_multilingual,
    apply_multilingual_layout,
    multilingual_paragraph,
    multilingual_text,
    set_multilingual_legend,
    set_multilingual_numeric_ticks,
    set_multilingual_suptitle,
    set_multilingual_title,
    set_multilingual_xlabel,
    set_multilingual_xticks,
    set_multilingual_ylabel,
    set_multilingual_yticks,
)

# Bengali-candidate list under its historical name
BANGLA_FONT_CANDIDATES = list(SCRIPT_FONT_CANDIDATES["Bengali"])


def to_bangla_numerals(value):
    """Convert ASCII digits to Bengali digits (bangla_render API)."""
    return to_native_numerals(value, "Bengali")


# Drop-in aliases -----------------------------------------------------

set_bangla_title = set_multilingual_title
set_bangla_xlabel = set_multilingual_xlabel
set_bangla_ylabel = set_multilingual_ylabel
set_bangla_suptitle = set_multilingual_suptitle
set_bangla_xticks = set_multilingual_xticks
set_bangla_yticks = set_multilingual_yticks
set_bangla_legend = set_multilingual_legend
bangla_text = multilingual_text
annotate_bangla = annotate_multilingual
add_bangla_in_cell = add_multilingual_cell_text
bangla_paragraph = multilingual_paragraph
apply_bangla_layout = apply_multilingual_layout


def set_bangla_numeric_ticks(ax, *args, **kwargs):
    """bangla_render numeric ticks: always Bengali digits."""
    kwargs.setdefault("script", "Bengali")
    return set_multilingual_numeric_ticks(ax, *args, **kwargs)


def text(ax, *args, **kwargs):
    """Alias for bangla_text()/multilingual_text()."""
    return multilingual_text(ax, *args, **kwargs)


__all__ = [
    "BANGLA_FONT_CANDIDATES",
    "to_bangla_numerals",
    "set_bangla_title",
    "set_bangla_xlabel",
    "set_bangla_ylabel",
    "set_bangla_suptitle",
    "set_bangla_xticks",
    "set_bangla_yticks",
    "set_bangla_legend",
    "set_bangla_numeric_ticks",
    "bangla_text",
    "annotate_bangla",
    "add_bangla_in_cell",
    "bangla_paragraph",
    "apply_bangla_layout",
    "text",
]

# universal_render/highlevel.py
"""
High-level convenience API.

Users think in languages ("hindi", "bangla"), not Unicode script names,
and usually want to label a whole plot in one call rather than six.
This module provides:

- language_to_script("hindi")          -> "Devanagari"
- set_language_font("bangla", family)  -> pin a font by language name
- localize_axes(ax, title=..., ...)    -> label an entire axes in one call
- multilingual_heatmap(ax, data, labels=...) -> annotated heatmap in one call
- multilingual_bar_labels(ax, labels)  -> value/category labels on bar charts
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .fonts import find_font_for_script, set_script_font
from .scripts import detect_script, to_native_numerals
from .mpl_support import (
    add_multilingual_cell_text,
    apply_multilingual_layout,
    multilingual_text,
    set_multilingual_legend,
    set_multilingual_suptitle,
    set_multilingual_title,
    set_multilingual_xlabel,
    set_multilingual_xticks,
    set_multilingual_ylabel,
    set_multilingual_yticks,
)


# ---------------------------------------------------------------------
# Language names -> script
# ---------------------------------------------------------------------

LANGUAGE_TO_SCRIPT: Dict[str, str] = {
    # Indic
    "bengali": "Bengali", "bangla": "Bengali", "assamese": "Bengali",
    "hindi": "Devanagari", "marathi": "Devanagari", "nepali": "Devanagari",
    "sanskrit": "Devanagari", "konkani": "Devanagari",
    "tamil": "Tamil",
    "telugu": "Telugu",
    "kannada": "Kannada",
    "malayalam": "Malayalam",
    "gujarati": "Gujarati",
    "punjabi": "Gurmukhi", "panjabi": "Gurmukhi",
    "odia": "Odia", "oriya": "Odia",
    "sinhala": "Sinhala", "sinhalese": "Sinhala",
    # RTL
    "arabic": "Arabic", "urdu": "Arabic", "persian": "Arabic", "farsi": "Arabic",
    "pashto": "Arabic", "kurdish": "Arabic",
    "hebrew": "Hebrew", "yiddish": "Hebrew",
    # Southeast Asian
    "thai": "Thai",
    "lao": "Lao",
    "khmer": "Khmer", "cambodian": "Khmer",
    "burmese": "Myanmar", "myanmar": "Myanmar",
    "tibetan": "Tibetan", "dzongkha": "Tibetan",
    # East Asian
    "chinese": "Han", "mandarin": "Han", "cantonese": "Han",
    "japanese": "Hiragana",
    "korean": "Hangul",
    # European / other
    "english": "Latin", "french": "Latin", "german": "Latin",
    "spanish": "Latin", "portuguese": "Latin", "italian": "Latin",
    "vietnamese": "Latin", "turkish": "Latin", "indonesian": "Latin",
    "russian": "Cyrillic", "ukrainian": "Cyrillic", "bulgarian": "Cyrillic",
    "serbian": "Cyrillic", "kazakh": "Cyrillic",
    "greek": "Greek",
    "georgian": "Georgian",
    "armenian": "Armenian",
    "amharic": "Ethiopic", "tigrinya": "Ethiopic",
}


def supported_languages() -> List[str]:
    """Language names accepted by language_to_script()."""
    return sorted(LANGUAGE_TO_SCRIPT)


def language_to_script(language: str) -> str:
    """
    Map a language name to its Unicode script.

        >>> language_to_script("hindi")
        'Devanagari'
        >>> language_to_script("urdu")
        'Arabic'
    """
    key = str(language).strip().lower()
    script = LANGUAGE_TO_SCRIPT.get(key)
    if script is None:
        raise KeyError(
            f"Unknown language: {language!r}. "
            f"Known languages: {', '.join(supported_languages())}"
        )
    return script


def set_language_font(language: str, family: str) -> None:
    """Pin a font family by language name, e.g. ("bangla", "Kalpurush")."""
    set_script_font(language_to_script(language), family)


def font_for_language(language: str, sample_text: Optional[str] = None) -> str:
    """Best installed font family for a language name."""
    return find_font_for_script(language_to_script(language), sample_text=sample_text)


def localized_numerals(value, language: str) -> str:
    """to_native_numerals() addressed by language name."""
    return to_native_numerals(value, language_to_script(language))


# ---------------------------------------------------------------------
# One-call axes localization
# ---------------------------------------------------------------------

def localize_axes(
    ax,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    suptitle: Optional[str] = None,
    xticklabels: Optional[Sequence[str]] = None,
    xtick_positions: Optional[Sequence[float]] = None,
    yticklabels: Optional[Sequence[str]] = None,
    ytick_positions: Optional[Sequence[float]] = None,
    legend_labels: Optional[Sequence[str]] = None,
    legend_loc: str = "upper right",
    title_size: int = 32,
    label_size: int = 24,
    tick_size: int = 16,
    legend_size: int = 16,
    color: str = "black",
    font_family: Optional[str] = None,
    auto_layout: bool = True,
) -> Dict[str, Any]:
    """
    Label an entire axes with script-aware text in a single call.

    Every argument is optional; only what you pass gets applied. Each
    string picks its own font from its dominant script, so the title can
    be Bengali while the xlabel is Arabic.

        ur.localize_axes(
            ax,
            title="বিক্রয় প্রতিবেদন",
            xlabel="महीना",
            ylabel="மதிப்பு",
            xticklabels=["এক", "दो", "மூன்று"],
        )

    Returns a dict of the created artists, keyed by role.
    """
    artists: Dict[str, Any] = {}

    if title is not None:
        artists["title"] = set_multilingual_title(
            ax, title, font_size=title_size, color=color, font_family=font_family,
        )
    if xlabel is not None:
        artists["xlabel"] = set_multilingual_xlabel(
            ax, xlabel, font_size=label_size, color=color, font_family=font_family,
        )
    if ylabel is not None:
        artists["ylabel"] = set_multilingual_ylabel(
            ax, ylabel, font_size=label_size, color=color, font_family=font_family,
        )
    if suptitle is not None:
        artists["suptitle"] = set_multilingual_suptitle(
            ax.figure, suptitle, font_size=title_size + 2, color=color,
            font_family=font_family,
        )

    if xticklabels is not None:
        positions = (
            list(xtick_positions) if xtick_positions is not None
            else list(ax.get_xticks())[: len(xticklabels)]
        )
        if len(positions) != len(xticklabels):
            positions = list(range(1, len(xticklabels) + 1))
        artists["xticks"] = set_multilingual_xticks(
            ax, positions, list(xticklabels),
            font_size=tick_size, color=color, font_family=font_family,
        )

    if yticklabels is not None:
        positions = (
            list(ytick_positions) if ytick_positions is not None
            else list(ax.get_yticks())[: len(yticklabels)]
        )
        if len(positions) != len(yticklabels):
            positions = list(range(1, len(yticklabels) + 1))
        artists["yticks"] = set_multilingual_yticks(
            ax, positions, list(yticklabels),
            font_size=tick_size, color=color, font_family=font_family,
        )

    if legend_labels is not None:
        artists["legend"] = set_multilingual_legend(
            ax, list(legend_labels), loc=legend_loc,
            font_size=legend_size, color=color, font_family=font_family,
        )

    if auto_layout:
        apply_multilingual_layout(ax.figure, auto=True)

    return artists


# ---------------------------------------------------------------------
# One-call heatmap
# ---------------------------------------------------------------------

def multilingual_heatmap(
    ax,
    data,
    labels=None,
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    cell_font_size: int = 24,
    tick_font_size: int = 16,
    title_font_size: int = 30,
    cell_color: str = "auto",
    colorbar: bool = True,
    value_format: Optional[str] = None,
    value_script: Optional[str] = None,
    imshow_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Draw a fully script-aware annotated heatmap in one call.

    Parameters
    ----------
    data:
        2D array of cell values.
    labels:
        Optional 2D array of cell label strings (any script). When None
        and ``value_format`` is given, numeric values are formatted (and
        converted to ``value_script`` digits if provided).
    row_labels / col_labels:
        Script-aware tick labels for the axes.
    cell_color:
        "auto" flips each cell's text between black/white based on the
        cell's luminance; any other value is used as-is.

    Returns (im, artists_dict).
    """
    import matplotlib.pyplot as plt  # local import; backend already chosen

    data = np.asarray(data, dtype=float)
    rows, cols = data.shape

    kwargs = dict(imshow_kwargs or {})
    kwargs.setdefault("cmap", cmap)
    im = ax.imshow(data, **kwargs)

    # Build labels from values when not supplied
    if labels is None and value_format is not None:
        labels = np.empty(data.shape, dtype=object)
        for i in range(rows):
            for j in range(cols):
                s = format(data[i, j], value_format)
                if value_script:
                    s = to_native_numerals(s, value_script)
                labels[i, j] = s

    artists: Dict[str, Any] = {"cells": []}

    if labels is not None:
        labels = np.asarray(labels, dtype=object)
        if labels.shape != data.shape:
            raise ValueError(
                f"labels shape {labels.shape} != data shape {data.shape}"
            )
        cmap_obj = im.get_cmap()
        vmin, vmax = im.get_clim()
        span = (vmax - vmin) or 1.0
        for i in range(rows):
            for j in range(cols):
                if cell_color == "auto":
                    r, g, b, _ = cmap_obj((data[i, j] - vmin) / span)
                    luminance = 0.299 * r + 0.587 * g + 0.114 * b
                    color = "black" if luminance > 0.55 else "white"
                else:
                    color = cell_color
                artists["cells"].append(add_multilingual_cell_text(
                    ax, i, j, str(labels[i, j]),
                    rows=rows, cols=cols,
                    font_size=cell_font_size, color=color,
                ))

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if col_labels is not None:
        artists["xticks"] = set_multilingual_xticks(
            ax, list(range(cols)), list(col_labels), font_size=tick_font_size,
        )
    if row_labels is not None:
        artists["yticks"] = set_multilingual_yticks(
            ax, list(range(rows)), list(row_labels), font_size=tick_font_size,
        )
    if title is not None:
        artists["title"] = set_multilingual_title(
            ax, title, font_size=title_font_size,
        )
    if colorbar:
        artists["colorbar"] = ax.figure.colorbar(im, ax=ax)

    return im, artists


# ---------------------------------------------------------------------
# Bar-chart labels
# ---------------------------------------------------------------------

def multilingual_bar_labels(
    ax,
    labels: Sequence[str],
    bars=None,
    orientation: str = "vertical",
    font_size: int = 16,
    color: str = "black",
    offset_frac: float = 0.02,
):
    """
    Put a script-aware label at the end of each bar.

    ``bars`` defaults to the first BarContainer on the axes (the result
    of ax.bar()/ax.barh()). ``orientation`` is "vertical" for ax.bar()
    or "horizontal" for ax.barh().
    """
    if bars is None:
        containers = getattr(ax, "containers", [])
        if not containers:
            raise ValueError("No bar containers found on axes; pass bars=")
        bars = containers[0]

    patches = list(bars)
    labels = list(labels)
    if len(patches) != len(labels):
        raise ValueError("labels and bars must have the same length")

    vertical = str(orientation).lower().startswith("v")
    if vertical:
        span = ax.get_ylim()
        pad = (span[1] - span[0]) * offset_frac
    else:
        span = ax.get_xlim()
        pad = (span[1] - span[0]) * offset_frac

    artists = []
    for patch, label in zip(patches, labels):
        if vertical:
            x = patch.get_x() + patch.get_width() / 2.0
            y = patch.get_y() + patch.get_height() + pad
            ha, va = "center", "bottom"
        else:
            x = patch.get_x() + patch.get_width() + pad
            y = patch.get_y() + patch.get_height() / 2.0
            ha, va = "left", "center"
        artists.append(multilingual_text(
            ax, x, y, str(label),
            coord="data", ha=ha, va=va,
            font_size=font_size, color=color,
        ))
    return artists

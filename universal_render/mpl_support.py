# universal_render/mpl_support.py
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (
    AnnotationBbox,
    AnchoredOffsetbox,
    DrawingArea,
    HPacker,
    OffsetImage,
    VPacker,
)
from matplotlib.patches import Patch, Rectangle

from .layout import get_layout_manager
from .renderer import render_paragraph_qimage, render_text_qimage
from .scripts import to_native_numerals


try:
    from PySide6.QtGui import QImage
except Exception:  # pragma: no cover
    QImage = None


def _qimage_to_rgba_array(qimg: QImage) -> np.ndarray:
    """
    Convert a QImage to an RGBA numpy array in [0, 1].
    """
    qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = qimg.width(), qimg.height()

    ptr = qimg.bits()
    buf = ptr.tobytes()
    arr = np.frombuffer(buf, np.uint8).reshape((h, w, 4))
    return arr / 255.0


def _resolve_font_size(fontsize, font_size, default):
    """
    Allow both Matplotlib-style `fontsize` and internal `font_size`.
    Priority: explicit font_size > fontsize > default.
    """
    if font_size is not None:
        return font_size
    if fontsize is not None:
        return fontsize
    return default


def _resolve_xycoords(ax, coord: str):
    """
    Map a user-facing coord string to Matplotlib coordinates.
    """
    coord = (coord or "data").lower()
    if coord == "axes":
        return ax.transAxes
    if coord == "figure":
        return ax.figure.transFigure
    return ax.transData


def _alignment_to_box_alignment(
    ha: str = "center",
    va: str = "center",
) -> Tuple[float, float]:
    ha_map = {
        "center": 0.5,
        "left": 0.0,
        "right": 1.0,
    }
    va_map = {
        "center": 0.5,
        "middle": 0.5,
        "bottom": 0.0,
        "baseline": 0.0,
        "top": 1.0,
    }
    return (
        ha_map.get((ha or "center").lower(), 0.5),
        va_map.get((va or "center").lower(), 0.5),
    )


def _default_zoom_for_fontsize(font_size: float) -> float:
    # Rendered image height already scales with font_size, so zoom must
    # stay constant: scaling it by font_size again squares the effect
    # and makes small labels (ticks, legends) unreadably tiny.
    return 0.40


def _build_offset_image(
    text: str,
    font_size: int,
    font_family: Optional[str] = None,
    font_path: Optional[str] = None,
    color: str = "black",
    bg: str = "transparent",
    padding: int = 10,
    scale: float = 1.0,
    zoom: Optional[float] = None,
    rotate_90: bool = False,
    weight=None,
    italic: bool = False,
    alpha: float = 1.0,
    rotation: float = 0.0,
):
    qimg = render_text_qimage(
        text=text,
        font_family=font_family,
        font_path=font_path,
        font_size=font_size,
        color=color,
        bg=bg,
        padding=padding,
        scale=scale,
        weight=weight,
        italic=italic,
        alpha=alpha,
        rotation=rotation,
    )
    img = _qimage_to_rgba_array(qimg)

    if rotate_90:
        img = np.rot90(img, k=1)

    if zoom is None:
        zoom = _default_zoom_for_fontsize(font_size)

    oi = OffsetImage(img, zoom=zoom)
    return qimg, img, oi


def _build_paragraph_offset_image(
    text: str,
    width: int,
    height: Optional[int] = None,
    font_size: int = 18,
    font_family: Optional[str] = None,
    font_path: Optional[str] = None,
    color: str = "black",
    bg: str = "transparent",
    margin: int = 12,
    scale: float = 1.0,
    zoom: Optional[float] = None,
):
    qimg = render_paragraph_qimage(
        text=text,
        width=width,
        height=height,
        font_family=font_family,
        font_path=font_path,
        font_size=font_size,
        color=color,
        bg=bg,
        margin=margin,
        scale=scale,
    )
    img = _qimage_to_rgba_array(qimg)

    if zoom is None:
        zoom = _default_zoom_for_fontsize(font_size)

    oi = OffsetImage(img, zoom=zoom)
    return qimg, img, oi


# ---------------------------------------------------------------------
# Renderer-based legend helpers
# ---------------------------------------------------------------------

def _normalize_legend_loc(loc):
    """
    Convert matplotlib legend loc strings/ints to AnchoredOffsetbox loc values.
    """
    mapping = {
        "best": "upper right",
        "upper right": "upper right",
        "upper left": "upper left",
        "lower left": "lower left",
        "lower right": "lower right",
        "right": "center right",
        "center left": "center left",
        "center right": "center right",
        "lower center": "lower center",
        "upper center": "upper center",
        "center": "center",
    }

    if isinstance(loc, str):
        return mapping.get(loc.lower(), "upper right")

    int_map = {
        0: "upper right",
        1: "upper right",
        2: "upper left",
        3: "lower left",
        4: "lower right",
        5: "right",
        6: "center left",
        7: "center right",
        8: "lower center",
        9: "upper center",
        10: "center",
    }
    return int_map.get(loc, "upper right")


def _build_legend_text_box(
    text: str,
    font_size: int,
    font_family: Optional[str],
    font_path: Optional[str],
    color: str,
    bg: str,
    padding: int,
    scale: float,
    zoom: Optional[float],
):
    _, _, oi = _build_offset_image(
        text=text,
        font_size=font_size,
        font_family=font_family,
        font_path=font_path,
        color=color,
        bg=bg,
        padding=padding,
        scale=scale,
        zoom=zoom,
        rotate_90=False,
    )
    return oi


def _build_line_handle_box(handle, width=34, height=18):
    """
    Create a simple legend handle sample for Line2D.
    """
    da = DrawingArea(width, height, 0, 0)

    color = getattr(handle, "get_color", lambda: "black")()
    linewidth = getattr(handle, "get_linewidth", lambda: 2.0)()
    linestyle = getattr(handle, "get_linestyle", lambda: "-")()
    marker = getattr(handle, "get_marker", lambda: None)()
    markersize = getattr(handle, "get_markersize", lambda: 6.0)()
    markerfacecolor = getattr(handle, "get_markerfacecolor", lambda: color)()
    markeredgecolor = getattr(handle, "get_markeredgecolor", lambda: color)()

    line = Line2D(
        [2, width - 2],
        [height / 2.0, height / 2.0],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    da.add_artist(line)

    if marker not in (None, "", "None", " "):
        mark = Line2D(
            [width / 2.0],
            [height / 2.0],
            color=color,
            marker=marker,
            markersize=markersize,
            linestyle="None",
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
        )
        da.add_artist(mark)

    return da


def _build_patch_handle_box(handle, width=18, height=12):
    """
    Create a simple legend handle sample for Patch-like artists.
    """
    da = DrawingArea(width, height, 0, 0)

    facecolor = getattr(handle, "get_facecolor", lambda: (0.7, 0.7, 0.7, 1.0))()
    edgecolor = getattr(handle, "get_edgecolor", lambda: (0.0, 0.0, 0.0, 1.0))()
    linewidth = getattr(handle, "get_linewidth", lambda: 1.0)()

    if isinstance(facecolor, (list, tuple)) and len(facecolor) and isinstance(facecolor[0], (list, tuple, np.ndarray)):
        facecolor = facecolor[0]
    if isinstance(edgecolor, (list, tuple)) and len(edgecolor) and isinstance(edgecolor[0], (list, tuple, np.ndarray)):
        edgecolor = edgecolor[0]

    rect = Rectangle(
        (1, 1),
        width - 2,
        height - 2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    da.add_artist(rect)
    return da


def _build_generic_handle_box(width=18, height=12):
    da = DrawingArea(width, height, 0, 0)
    rect = Rectangle(
        (1, 1),
        width - 2,
        height - 2,
        facecolor=(0.75, 0.75, 0.75, 1.0),
        edgecolor=(0.0, 0.0, 0.0, 1.0),
        linewidth=1.0,
    )
    da.add_artist(rect)
    return da


def _build_collection_handle_box(handle, width=18, height=12):
    """
    Legend handle sample for scatter plots (PathCollection): a marker
    dot in the collection's face colour.
    """
    da = DrawingArea(width, height, 0, 0)
    color = (0.3, 0.3, 0.3, 1.0)
    try:
        fc = handle.get_facecolor()
        if len(fc):
            color = tuple(fc[0])
    except Exception:
        pass
    mark = Line2D(
        [width / 2.0], [height / 2.0],
        marker="o", markersize=6, linestyle="None",
        markerfacecolor=color, markeredgecolor=color,
    )
    da.add_artist(mark)
    return da


def _build_legend_handle_box(handle):
    from matplotlib.collections import Collection
    if isinstance(handle, Line2D):
        return _build_line_handle_box(handle)
    if isinstance(handle, Patch):
        return _build_patch_handle_box(handle)
    if isinstance(handle, Collection):
        return _build_collection_handle_box(handle)
    return _build_generic_handle_box()


def set_multilingual_legend(
    ax,
    labels,
    handles=None,
    loc="upper right",
    frameon=True,
    fontsize=None,
    font_size=None,
    font_family=None,
    font_path=None,
    color="black",
    bg="transparent",
    padding=8,
    scale=1.0,
    zoom=None,
    title=None,
    title_fontsize=None,
    title_font_size=None,
    title_color=None,
    title_zoom=None,
    borderpad=0.5,
    labelspacing=0.35,
    handletextpad=0.6,
    borderaxespad=0.5,
    facecolor="white",
    edgecolor="0.8",
    framealpha=0.95,
    zorder=8,
):
    """
    Renderer-based multilingual legend.

    This version does not rely on native Matplotlib legend text rendering.
    multilingual labels (and optional title) are rendered through render_text_qimage()
    and packed into an AnchoredOffsetbox.

    Parameters
    ----------
    labels:
        multilingual legend label strings.
    handles:
        Optional artist handles. If None, uses ax.get_legend_handles_labels().
    loc:
        Legend location, similar to matplotlib legend loc.
    """
    if handles is None:
        handles, _ = ax.get_legend_handles_labels()

    handles = list(handles)
    labels = list(labels)

    if len(handles) != len(labels):
        raise ValueError("handles and labels must have the same length")

    # Remove any native legend already present
    if getattr(ax, "legend_", None) is not None:
        try:
            ax.legend_.remove()
        except Exception:
            pass
        ax.legend_ = None

    fs = _resolve_font_size(fontsize, font_size, default=16)
    title_fs = _resolve_font_size(title_fontsize, title_font_size, default=max(fs, 16))
    title_color = color if title_color is None else title_color

    row_boxes = []
    sep_px = max(2, int(round(8 * handletextpad)))

    for handle, label in zip(handles, labels):
        handle_box = _build_legend_handle_box(handle)
        text_box = _build_legend_text_box(
            text=label,
            font_size=fs,
            font_family=font_family,
            font_path=font_path,
            color=color,
            bg=bg,
            padding=padding,
            scale=scale,
            zoom=zoom,
        )

        row = HPacker(
            children=[handle_box, text_box],
            align="center",
            pad=0,
            sep=sep_px,
        )
        row_boxes.append(row)

    children = []

    if title:
        title_box = _build_legend_text_box(
            text=title,
            font_size=title_fs,
            font_family=font_family,
            font_path=font_path,
            color=title_color,
            bg=bg,
            padding=padding,
            scale=scale,
            zoom=title_zoom if title_zoom is not None else zoom,
        )
        children.append(title_box)

    children.extend(row_boxes)

    row_sep = max(2, int(round(10 * labelspacing)))
    legend_box = VPacker(
        children=children,
        align="left",
        pad=0,
        sep=row_sep,
    )

    anchored = AnchoredOffsetbox(
        loc=_normalize_legend_loc(loc),
        child=legend_box,
        frameon=frameon,
        pad=borderpad,
        borderpad=borderaxespad,
        bbox_to_anchor=None,
        bbox_transform=ax.transAxes,
    )

    anchored.set_zorder(zorder)

    if frameon and getattr(anchored, "patch", None) is not None:
        try:
            anchored.patch.set_facecolor(facecolor)
            anchored.patch.set_edgecolor(edgecolor)
            anchored.patch.set_alpha(framealpha)
        except Exception:
            pass

    ax.add_artist(anchored)
    return anchored


def set_multilingual_numeric_ticks(
    ax,
    script: str = "Bengali",
    axis: str = "x",
    positions=None,
    values=None,
    fontsize=None,
    font_size=None,
    font_family=None,
    font_path=None,
    color="black",
    bg="transparent",
    padding=8,
    scale=1.0,
    zoom=None,
    extra_pad_axes: float = 0.02,
    ha=None,
    va=None,
    zorder=5,
    hide_native: bool = True,
    collision_avoidance: bool = True,
    formatter=None,
):
    """
    Render numeric tick labels using the native digits of ``script``
    (e.g. "Bengali" -> ২০২৬, "Devanagari" -> २०२६, "Arabic" -> ٢٠٢٦).
    Scripts without native digits keep ASCII digits.
    """
    axis = (axis or "x").lower()
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")

    if positions is None:
        positions = ax.get_xticks() if axis == "x" else ax.get_yticks()

    positions = list(positions)

    if values is None:
        values = positions
    else:
        values = list(values)

    if len(positions) != len(values):
        raise ValueError("positions and values must have the same length")

    labels = []
    for v in values:
        s = formatter(v) if formatter is not None else str(v)
        labels.append(to_native_numerals(s, script))

    if axis == "x":
        return set_multilingual_xticks(
            ax=ax,
            positions=positions,
            labels=labels,
            fontsize=fontsize,
            font_size=font_size,
            font_family=font_family,
            font_path=font_path,
            color=color,
            bg=bg,
            padding=padding,
            scale=scale,
            zoom=zoom,
            extra_pad_axes=extra_pad_axes,
            ha="center" if ha is None else ha,
            va="top" if va is None else va,
            zorder=zorder,
            hide_native=hide_native,
            collision_avoidance=collision_avoidance,
        )

    return set_multilingual_yticks(
        ax=ax,
        positions=positions,
        labels=labels,
        fontsize=fontsize,
        font_size=font_size,
        font_family=font_family,
        font_path=font_path,
        color=color,
        bg=bg,
        padding=padding,
        scale=scale,
        zoom=zoom,
        extra_pad_axes=extra_pad_axes,
        ha="right" if ha is None else ha,
        va="center" if va is None else va,
        zorder=zorder,
        hide_native=hide_native,
        collision_avoidance=collision_avoidance,
    )


# ---------------------------------------------------------------------
# Core managed labels via layout manager
# ---------------------------------------------------------------------

def set_multilingual_title(
    ax,
    text,
    fontsize=None,
    font_size=None,
    font_family=None,
    font_path=None,
    color="black",
    padding=10,
    scale=1.0,
    zoom=0.40,
    extra_pad_px=10.0,
    zorder=5,
    loc="center",
    weight=None,
    italic=False,
    alpha=1.0,
):
    """
    Set a multilingual title above the axes using the layout manager.

    ``loc`` matches Matplotlib's set_title(loc=): "left", "center",
    or "right". ``weight``/``italic``/``alpha`` match Matplotlib text
    styling ("bold", True, 0..1).
    """
    manager = get_layout_manager(ax.figure)
    return manager.add_title(
        ax=ax,
        text=text,
        fontsize=fontsize,
        font_size=font_size,
        font_family=font_family,
        font_path=font_path,
        color=color,
        padding=padding,
        scale=scale,
        zoom=zoom,
        extra_pad_px=extra_pad_px,
        zorder=zorder,
        loc=loc,
        weight=weight,
        italic=italic,
        alpha=alpha,
    )


def set_multilingual_xlabel(
    ax,
    text,
    fontsize=None,
    font_size=None,
    font_family=None,
    font_path=None,
    color="black",
    padding=10,
    scale=1.0,
    zoom=0.40,
    extra_pad_px=8.0,
    zorder=5,
    weight=None,
    italic=False,
    alpha=1.0,
):
    """
    Set a multilingual x-axis label using the layout manager.
    """
    manager = get_layout_manager(ax.figure)
    return manager.add_xlabel(
        ax=ax,
        text=text,
        fontsize=fontsize,
        font_size=font_size,
        font_family=font_family,
        font_path=font_path,
        color=color,
        padding=padding,
        scale=scale,
        zoom=zoom,
        extra_pad_px=extra_pad_px,
        zorder=zorder,
        weight=weight,
        italic=italic,
        alpha=alpha,
    )


def set_multilingual_ylabel(
    ax,
    text,
    fontsize=None,
    font_size=None,
    font_family=None,
    font_path=None,
    color="black",
    padding=10,
    scale=1.0,
    zoom=0.40,
    extra_pad_px=8.0,
    zorder=5,
    weight=None,
    italic=False,
    alpha=1.0,
):
    """
    Set a multilingual y-axis label using the layout manager.
    """
    manager = get_layout_manager(ax.figure)
    return manager.add_ylabel(
        ax=ax,
        text=text,
        fontsize=fontsize,
        font_size=font_size,
        font_family=font_family,
        font_path=font_path,
        color=color,
        padding=padding,
        scale=scale,
        zoom=zoom,
        extra_pad_px=extra_pad_px,
        zorder=zorder,
        weight=weight,
        italic=italic,
        alpha=alpha,
    )


def set_multilingual_suptitle(
    fig,
    text,
    fontsize=None,
    font_size=None,
    font_family=None,
    font_path=None,
    color="black",
    padding=10,
    scale=1.0,
    zoom=0.40,
    extra_pad_px=18.0,
    zorder=6,
    weight=None,
    italic=False,
    alpha=1.0,
):
    """
    Set a multilingual suptitle for the whole figure, stacked above any
    per-axes titles.
    """
    manager = get_layout_manager(fig)
    return manager.add_suptitle(
        fig=fig,
        text=text,
        fontsize=fontsize,
        font_size=font_size,
        font_family=font_family,
        font_path=font_path,
        color=color,
        padding=padding,
        scale=scale,
        zoom=zoom,
        extra_pad_px=extra_pad_px,
        zorder=zorder,
        weight=weight,
        italic=italic,
        alpha=alpha,
    )


# ---------------------------------------------------------------------
# General multilingual text / annotation helpers
# ---------------------------------------------------------------------

def multilingual_text(
    ax,
    x,
    y,
    text,
    fontsize=None,
    font_size=None,
    font_family=None,
    font_path=None,
    color="black",
    bg="transparent",
    padding=10,
    scale=1.0,
    coord="data",
    ha="center",
    va="center",
    zoom=None,
    zorder=5,
    weight=None,
    italic=False,
    alpha=1.0,
    rotation=0.0,
):
    """
    General multilingual text helper, similar to ax.text() but using Qt
    rendering. Supports Matplotlib-style ``weight`` ("bold"), ``italic``,
    ``alpha`` (0..1), ``rotation`` (CCW degrees, any angle), and
    multiline strings containing ``\\n``.
    """
    fs = _resolve_font_size(fontsize, font_size, default=18)

    _, _, oi = _build_offset_image(
        text=text,
        font_size=fs,
        font_family=font_family,
        font_path=font_path,
        color=color,
        bg=bg,
        padding=padding,
        scale=scale,
        zoom=zoom,
        rotate_90=False,
        weight=weight,
        italic=italic,
        alpha=alpha,
        rotation=rotation,
    )

    box_alignment = _alignment_to_box_alignment(ha=ha, va=va)
    xycoords = _resolve_xycoords(ax, coord)

    ab = AnnotationBbox(
        oi,
        (x, y),
        xycoords=xycoords,
        frameon=False,
        box_alignment=box_alignment,
        zorder=zorder,
        annotation_clip=False,
    )

    if (coord or "data").lower() == "figure":
        ax.figure.add_artist(ab)
    else:
        ax.add_artist(ab)

    return ab


def text(ax, *args, **kwargs):
    """
    Alias for multilingual_text().
    """
    return multilingual_text(ax, *args, **kwargs)


def annotate_multilingual(
    ax,
    text,
    xy,
    xytext=None,
    fontsize=None,
    font_size=None,
    font_family=None,
    font_path=None,
    color="black",
    bg="transparent",
    padding=10,
    scale=1.0,
    coord="data",
    textcoord=None,
    ha="center",
    va="center",
    zoom=None,
    zorder=5,
    arrowprops=None,
):
    """
    Add multilingual annotation text, with an optional Matplotlib arrow.
    """
    if xytext is None:
        xytext = xy
    if textcoord is None:
        textcoord = coord

    arrow_artist = None
    if arrowprops is not None:
        arrow_artist = ax.annotate(
            "",
            xy=xy,
            xytext=xytext,
            xycoords=_resolve_xycoords(ax, coord),
            textcoords=_resolve_xycoords(ax, textcoord),
            arrowprops=arrowprops,
            annotation_clip=False,
        )

    text_artist = multilingual_text(
        ax=ax,
        x=xytext[0],
        y=xytext[1],
        text=text,
        fontsize=fontsize,
        font_size=font_size,
        font_family=font_family,
        font_path=font_path,
        color=color,
        bg=bg,
        padding=padding,
        scale=scale,
        coord=textcoord,
        ha=ha,
        va=va,
        zoom=zoom,
        zorder=zorder,
    )

    return {
        "text_artist": text_artist,
        "arrow_artist": arrow_artist,
    }


# ---------------------------------------------------------------------
# Heatmap / cell helpers
# ---------------------------------------------------------------------

def add_multilingual_cell_text(
    ax,
    row,
    col,
    text,
    rows,
    cols,
    fontsize=None,
    font_size=None,
    font_family=None,
    font_path=None,
    color="black",
    bg="transparent",
    padding=10,
    scale=1.0,
    origin="upper",
    zoom=None,
    zorder=6,
):
    """
    Draw multilingual text at the center of a heatmap cell using axes coordinates.
    """
    fs = _resolve_font_size(fontsize, font_size, default=22)

    x_axes = (col + 0.5) / cols
    if origin == "upper":
        y_axes = 1.0 - (row + 0.5) / rows
    else:
        y_axes = (row + 0.5) / rows

    return multilingual_text(
        ax=ax,
        x=x_axes,
        y=y_axes,
        text=text,
        coord="axes",
        ha="center",
        va="center",
        fontsize=fs,
        font_family=font_family,
        font_path=font_path,
        color=color,
        bg=bg,
        padding=padding,
        scale=scale,
        zoom=zoom,
        zorder=zorder,
    )


def multilingual_paragraph(
    ax,
    x,
    y,
    text,
    width=400,
    height=None,
    fontsize=None,
    font_size=None,
    font_family=None,
    font_path=None,
    color="black",
    bg="transparent",
    margin=12,
    scale=1.0,
    coord="axes",
    ha="left",
    va="top",
    zoom=None,
    zorder=5,
):
    """
    Place wrapped multilingual paragraph text as an image artist.
    Note: this is a free/manual artist, not layout-managed.
    """
    fs = _resolve_font_size(fontsize, font_size, default=18)

    _, _, oi = _build_paragraph_offset_image(
        text=text,
        width=width,
        height=height,
        font_size=fs,
        font_family=font_family,
        font_path=font_path,
        color=color,
        bg=bg,
        margin=margin,
        scale=scale,
        zoom=zoom,
    )

    box_alignment = _alignment_to_box_alignment(ha=ha, va=va)
    xycoords = _resolve_xycoords(ax, coord)

    ab = AnnotationBbox(
        oi,
        (x, y),
        xycoords=xycoords,
        frameon=False,
        box_alignment=box_alignment,
        zorder=zorder,
        annotation_clip=False,
    )

    if (coord or "data").lower() == "figure":
        ax.figure.add_artist(ab)
    else:
        ax.add_artist(ab)

    return ab


# ---------------------------------------------------------------------
# Tick helpers via layout manager
# ---------------------------------------------------------------------

def set_multilingual_xticks(
    ax,
    positions: Sequence[float],
    labels: Sequence[str],
    fontsize=None,
    font_size=None,
    font_family=None,
    font_path=None,
    color="black",
    bg="transparent",
    padding=8,
    scale=1.0,
    zoom=None,
    extra_pad_axes: float = 0.02,
    ha="center",
    va="top",
    zorder=5,
    hide_native: bool = True,
    collision_avoidance: bool = True,
    rotation: float = 0.0,
):
    """
    Replace x tick labels with script-aware rendered image labels using the layout manager.
    ``rotation`` rotates each label counter-clockwise (e.g. 45 for
    slanted long labels, matching Matplotlib's tick rotation).
    """
    if len(positions) != len(labels):
        raise ValueError("positions and labels must have the same length")

    manager = get_layout_manager(ax.figure)
    return manager.add_xticks(
        ax=ax,
        positions=list(positions),
        labels=list(labels),
        fontsize=fontsize,
        font_size=font_size,
        font_family=font_family,
        font_path=font_path,
        color=color,
        bg=bg,
        padding=padding,
        scale=scale,
        zoom=zoom,
        extra_pad_axes=extra_pad_axes,
        ha=ha,
        va=va,
        zorder=zorder,
        hide_native=hide_native,
        collision_avoidance=collision_avoidance,
        rotation=rotation,
    )


def set_multilingual_yticks(
    ax,
    positions: Sequence[float],
    labels: Sequence[str],
    fontsize=None,
    font_size=None,
    font_family=None,
    font_path=None,
    color="black",
    bg="transparent",
    padding=8,
    scale=1.0,
    zoom=None,
    extra_pad_axes: float = 0.02,
    ha="right",
    va="center",
    zorder=5,
    hide_native: bool = True,
    collision_avoidance: bool = True,
    rotation: float = 0.0,
):
    """
    Replace y tick labels with script-aware rendered image labels using the layout manager.
    ``rotation`` rotates each label counter-clockwise (degrees).
    """
    if len(positions) != len(labels):
        raise ValueError("positions and labels must have the same length")

    manager = get_layout_manager(ax.figure)
    return manager.add_yticks(
        ax=ax,
        positions=list(positions),
        labels=list(labels),
        fontsize=fontsize,
        font_size=font_size,
        font_family=font_family,
        font_path=font_path,
        color=color,
        bg=bg,
        padding=padding,
        scale=scale,
        zoom=zoom,
        extra_pad_axes=extra_pad_axes,
        ha=ha,
        va=va,
        zorder=zorder,
        hide_native=hide_native,
        collision_avoidance=collision_avoidance,
        rotation=rotation,
    )


# ---------------------------------------------------------------------
# Layout helper
# ---------------------------------------------------------------------

def apply_multilingual_layout(
    fig,
    left=0.18,
    right=0.88,
    bottom=0.22,
    top=0.84,
    auto=False,
    min_left=0.12,
    min_right=0.90,
    min_bottom=0.12,
    min_top=0.88,
):
    """
    Adjust figure margins to leave enough space for multilingual titles,
    axis labels, and managed multilingual tick labels.
    """
    manager = get_layout_manager(fig)

    if auto:
        manager.update_layout()
        manager.auto_adjust_margins(
            min_left=min_left,
            min_right=min_right,
            min_bottom=min_bottom,
            min_top=min_top,
        )
        manager.update_layout()
        return manager

    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
    )
    manager.update_layout()
    return manager
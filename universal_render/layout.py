# universal_render/layout.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.transforms import Bbox

from .renderer import render_text_qimage

try:
    from PySide6.QtGui import QImage
except Exception:  # pragma: no cover
    QImage = None


_MANAGER_REGISTRY: Dict[int, "UniversalLayoutManager"] = {}


# ─────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────

def _qimage_to_rgba_array(qimg: QImage) -> np.ndarray:
    qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = qimg.width(), qimg.height()
    buf  = qimg.bits().tobytes()
    return np.frombuffer(buf, np.uint8).reshape((h, w, 4)) / 255.0


def _renderer_px_size(fig, r):
    """
    Canvas size in the RENDERER's pixels. During savefig(dpi=...) the
    renderer runs at a different dpi than fig.dpi, so renderer-measured
    extents must be normalized by this size — not by fig.dpi — or every
    placement drifts by the dpi ratio.
    """
    try:
        w, h = float(r.width), float(r.height)
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    win, hin = fig.get_size_inches()
    return win * fig.dpi, hin * fig.dpi


def _pixels_to_fig_dx(fig, px: float) -> float:
    fw = fig.get_size_inches()[0] * fig.dpi
    return px / fw if fw > 0 else 0.0


def _pixels_to_fig_dy(fig, px: float) -> float:
    fh = fig.get_size_inches()[1] * fig.dpi
    return px / fh if fh > 0 else 0.0


def _pixels_to_axes_dx(ax, px: float) -> float:
    bp = ax.get_position()
    fw = ax.figure.get_size_inches()[0] * ax.figure.dpi
    return px / max(1.0, bp.width * fw)


def _pixels_to_axes_dy(ax, px: float) -> float:
    bp = ax.get_position()
    fh = ax.figure.get_size_inches()[1] * ax.figure.dpi
    return px / max(1.0, bp.height * fh)


def _remove_artist_safe(artist) -> None:
    if artist is None:
        return
    try:
        artist.remove()
    except Exception:
        pass


def _resolve_font_size(fontsize, font_size, default):
    if font_size is not None:  return font_size
    if fontsize   is not None: return fontsize
    return default


def _ensure_list(value) -> List[Any]:
    if value is None:           return []
    if isinstance(value, list): return value
    return list(value)


def _union_bboxes(bboxes: List[Optional[Bbox]]) -> Optional[Bbox]:
    valid = [b for b in bboxes
             if b is not None and np.isfinite([b.x0, b.y0, b.x1, b.y1]).all()]
    if not valid:
        return None
    return Bbox.from_extents(
        min(b.x0 for b in valid), min(b.y0 for b in valid),
        max(b.x1 for b in valid), max(b.y1 for b in valid),
    )


# ─────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ManagedTextItem:
    kind: str
    text: str
    ax:   Any = None
    fig:  Any = None

    fontsize:    Optional[float] = None
    font_size:   Optional[float] = None
    font_family: Optional[str]   = None
    font_path:   Optional[str]   = None
    color:        str   = "black"
    padding:      int   = 10
    scale:        float = 1.0
    zoom:         float = 0.40
    extra_pad_px: float = 8.0
    zorder:       int   = 5
    weight:       Any   = None
    italic:       bool  = False
    alpha:        float = 1.0
    loc:          str   = "center"

    artist: Any = None
    last_image_size_px:  Tuple[int, int]     = field(default_factory=lambda: (0, 0))
    last_render_size_px: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))


@dataclass
class ManagedTickGroup:
    kind: str
    ax:   Any
    positions: Sequence[float]
    labels:    Sequence[str]

    fontsize:    Optional[float] = None
    font_size:   Optional[float] = None
    font_family: Optional[str]   = None
    font_path:   Optional[str]   = None
    color:  str   = "black"
    bg:     str   = "transparent"
    padding: int  = 8
    scale:  float = 1.0
    zoom:   Optional[float] = None
    extra_pad_axes: float = 0.02
    zorder: int   = 5
    rotation: float = 0.0
    ha: str = "center"
    va: str = "top"
    hide_native:         bool = True
    collision_avoidance: bool = True

    artists:       List[Any]                 = field(default_factory=list)
    last_sizes_px: List[Tuple[float, float]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Layout manager
# ─────────────────────────────────────────────────────────────────────

class UniversalLayoutManager:
    """
    ylabel placement — colorbar-aware, correct for any subplot position
    ─────────────────────────────────────────────────────────────────────
    Position formula:
        cx_px = strip_right - GAP - screen_width / 2

    where strip_right = ytick labels left edge (renderer-measured,
    post-colorbar accurate).

    The x position is clamped to stay left of the axes left edge
    (NOT to a fixed figure fraction like 0.48 — that was wrong for
    right-side subplots where the correct x fraction can be > 0.48).

    Colorbar detection: if any other axes has its right edge within
    COLORBAR_OVERLAP_TOL px of the ytick left edge, AND the remaining
    space is < MIN_YLABEL_STRIP_PX, the ylabel is skipped.

    User zoom is used exactly — no auto-zoom manipulation.
    """

    X_TICK_GAP_PX           = 20.0
    Y_TICK_GAP_PX           = 20.0
    X_LABEL_BASE_GAP_PX     = 10.0
    Y_LABEL_BASE_GAP_PX     = 14.0
    TITLE_TO_AXES_GAP_PX    = 10.0
    SUPTITLE_GAP_PX         = 12.0
    LEFT_MARGIN_SAFETY_PX   = 10.0
    BOTTOM_MARGIN_SAFETY_PX = 10.0
    TOP_MARGIN_SAFETY_PX    = 12.0

    # Any other axes whose right edge is within this px of ytick_left
    # is treated as potentially blocking the ylabel strip.
    COLORBAR_OVERLAP_TOL = 60.0

    # Skip ylabel if clear space left of ytick labels is < this.
    MIN_YLABEL_STRIP_PX  = 20.0

    def __init__(self, fig):
        self.fig = fig
        self.items:       List[ManagedTextItem] = []
        self.tick_groups: List[ManagedTickGroup]  = []
        self._draw_cid   = None
        self._resize_cid = None
        self._is_updating = False
        self._orig_savefig = None
        self._connect_events()
        self._wrap_savefig()

    # ── events ──────────────────────────────────────────────────────

    def _connect_events(self):
        c = self.fig.canvas
        if c is None: return
        self._draw_cid   = c.mpl_connect("draw_event",   self._on_draw_event)
        self._resize_cid = c.mpl_connect("resize_event", self._on_resize_event)

    def _wrap_savefig(self):
        """
        Refresh managed placements immediately before every save.

        Aspect-locked axes (imshow) and colorbars change the axes box at
        draw time, after labels were placed. Layout cannot run DURING
        the save (see _on_draw_event), so it runs here: update_layout()
        draws once on the normal-dpi canvas, measures the final axes
        geometry, and re-places everything; the save then renders those
        placements untouched.
        """
        if getattr(self.fig, "_ur_savefig_wrapped", False):
            return
        self._orig_savefig = self.fig.savefig
        manager = self

        def savefig_with_layout(*args, **kwargs):
            try:
                manager.update_layout()
            except Exception:
                pass
            return manager._orig_savefig(*args, **kwargs)

        self.fig.savefig = savefig_with_layout
        self.fig._ur_savefig_wrapped = True

    def _unwrap_savefig(self):
        if self._orig_savefig is not None:
            try:
                self.fig.savefig = self._orig_savefig
                self.fig._ur_savefig_wrapped = False
            except Exception:
                pass
            self._orig_savefig = None

    def disconnect(self):
        self._unwrap_savefig()
        c = self.fig.canvas
        if c is None: return
        for attr in ("_draw_cid", "_resize_cid"):
            cid = getattr(self, attr, None)
            if cid is not None:
                try: c.mpl_disconnect(cid)
                except Exception: pass
                setattr(self, attr, None)

    def _is_saving(self) -> bool:
        """True while the canvas is inside savefig/print_figure."""
        c = self.fig.canvas
        checker = getattr(c, "is_saving", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                pass
        return bool(getattr(c, "_is_saving", False))

    def _on_draw_event(self, event):
        # Never re-place during savefig: with bbox_inches="tight" the
        # crop window is computed from one draw and applied to another,
        # and fig.dpi is temporarily the save dpi — any placement change
        # in between shifts artists out of the crop. The pre-save layout
        # refresh happens in the savefig wrapper instead.
        if self._is_saving(): return
        if not self._is_updating: self.update_layout()

    def _on_resize_event(self, _):
        if self._is_saving(): return
        if not self._is_updating: self.update_layout()

    # ── registration ────────────────────────────────────────────────

    def _make_item(self, kind, text, ax=None, fig=None, **kw):
        return ManagedTextItem(
            kind=kind, text=text, ax=ax, fig=fig,
            fontsize=kw.get("fontsize"),
            font_size=kw.get("font_size"),
            font_family=kw.get("font_family"),
            font_path=kw.get("font_path"),
            color=kw.get("color", "black"),
            padding=kw.get("padding", 10),
            scale=kw.get("scale", 1.0),
            zoom=kw.get("zoom", 0.40),
            extra_pad_px=kw.get("extra_pad_px", 8.0),
            zorder=kw.get("zorder", 5),
            weight=kw.get("weight"),
            italic=kw.get("italic", False),
            alpha=kw.get("alpha", 1.0),
            loc=kw.get("loc", "center"),
        )

    def add_title(self, ax, text, **kw):
        self._rm_item("title", ax=ax)
        it = self._make_item("title", text, ax=ax, fig=ax.figure, **kw)
        self.items.append(it); ax.set_title("")
        self.update_layout(); return it

    def add_xlabel(self, ax, text, **kw):
        self._rm_item("xlabel", ax=ax)
        it = self._make_item("xlabel", text, ax=ax, fig=ax.figure, **kw)
        self.items.append(it); ax.set_xlabel("")
        self.update_layout(); return it

    def add_ylabel(self, ax, text, **kw):
        self._rm_item("ylabel", ax=ax)
        it = self._make_item("ylabel", text, ax=ax, fig=ax.figure, **kw)
        self.items.append(it); ax.set_ylabel("")
        self.update_layout(); return it

    def add_suptitle(self, fig, text, **kw):
        self._rm_item("suptitle", fig=fig)
        it = self._make_item("suptitle", text, ax=None, fig=fig, **kw)
        self.items.append(it)
        try: fig.suptitle("")
        except Exception: pass
        self.update_layout(); return it

    def add_xticks(self, ax, positions, labels, **kw):
        self._rm_tick("xticks", ax)
        g = ManagedTickGroup(
            kind="xticks", ax=ax,
            positions=list(positions), labels=list(labels),
            fontsize=kw.get("fontsize"),       font_size=kw.get("font_size"),
            font_family=kw.get("font_family"), font_path=kw.get("font_path"),
            color=kw.get("color", "black"),    bg=kw.get("bg", "transparent"),
            padding=kw.get("padding", 8),      scale=kw.get("scale", 1.0),
            zoom=kw.get("zoom"),
            extra_pad_axes=kw.get("extra_pad_axes", 0.02),
            zorder=kw.get("zorder", 5),
            rotation=kw.get("rotation", 0.0),
            ha=kw.get("ha", "center"),         va=kw.get("va", "top"),
            hide_native=kw.get("hide_native", True),
            collision_avoidance=kw.get("collision_avoidance", True),
        )
        self.tick_groups.append(g)
        ax.set_xticks(list(positions))
        if g.hide_native: ax.set_xticklabels([])
        self.update_layout(); return g

    def add_yticks(self, ax, positions, labels, **kw):
        self._rm_tick("yticks", ax)
        g = ManagedTickGroup(
            kind="yticks", ax=ax,
            positions=list(positions), labels=list(labels),
            fontsize=kw.get("fontsize"),       font_size=kw.get("font_size"),
            font_family=kw.get("font_family"), font_path=kw.get("font_path"),
            color=kw.get("color", "black"),    bg=kw.get("bg", "transparent"),
            padding=kw.get("padding", 8),      scale=kw.get("scale", 1.0),
            zoom=kw.get("zoom"),
            extra_pad_axes=kw.get("extra_pad_axes", 0.02),
            zorder=kw.get("zorder", 5),
            rotation=kw.get("rotation", 0.0),
            ha=kw.get("ha", "right"),          va=kw.get("va", "center"),
            hide_native=kw.get("hide_native", True),
            collision_avoidance=kw.get("collision_avoidance", True),
        )
        self.tick_groups.append(g)
        ax.set_yticks(list(positions))
        if g.hide_native: ax.set_yticklabels([])
        self.update_layout(); return g

    # ── removal ─────────────────────────────────────────────────────

    def _rm_item(self, kind, ax=None, fig=None):
        for it in [x for x in self.items if x.kind == kind and (
                (ax  is not None and x.ax  is ax) or
                (fig is not None and x.fig is fig))]:
            self.remove_item(it)

    def _rm_tick(self, kind, ax):
        for g in [x for x in self.tick_groups if x.kind == kind and x.ax is ax]:
            self.remove_tick_group(g)

    def remove_item(self, it):
        _remove_artist_safe(it.artist); it.artist = None
        try: self.items.remove(it)
        except ValueError: pass

    def remove_tick_group(self, g):
        for a in _ensure_list(g.artists): _remove_artist_safe(a)
        g.artists = []; g.last_sizes_px = []
        try: self.tick_groups.remove(g)
        except ValueError: pass

    def clear(self):
        for it in self.items:
            _remove_artist_safe(it.artist); it.artist = None
        self.items.clear()
        for g in self.tick_groups:
            for a in _ensure_list(g.artists): _remove_artist_safe(a)
            g.artists = []; g.last_sizes_px = []
        self.tick_groups.clear()

    # ── rendering ────────────────────────────────────────────────────

    def _default_fs(self, kind):
        return {"title": 32, "xlabel": 26, "ylabel": 26,
                "suptitle": 34, "xticks": 16, "yticks": 16}.get(kind, 24)

    def _render_item_image(self, item):
        fs = _resolve_font_size(item.fontsize, item.font_size,
                                self._default_fs(item.kind))
        q  = render_text_qimage(
            item.text, font_family=item.font_family, font_path=item.font_path,
            font_size=fs, color=item.color, bg="transparent",
            padding=item.padding, scale=item.scale,
            weight=item.weight, italic=item.italic, alpha=item.alpha,
        )
        img = _qimage_to_rgba_array(q)
        rw  = q.width()  * item.zoom
        rh  = q.height() * item.zoom
        item.last_image_size_px  = (q.width(), q.height())
        item.last_render_size_px = (rw, rh)
        return fs, q, img, rw, rh

    def _render_tick_image(self, g, label):
        fs = _resolve_font_size(g.fontsize, g.font_size, self._default_fs(g.kind))
        q  = render_text_qimage(
            label, font_family=g.font_family, font_path=g.font_path,
            font_size=fs, color=g.color, bg=g.bg,
            padding=g.padding, scale=g.scale,
            rotation=g.rotation,
        )
        img = _qimage_to_rgba_array(q)
        z   = g.zoom if g.zoom is not None else 0.40
        return fs, q, img, q.width() * z, q.height() * z

    # ── lookup ───────────────────────────────────────────────────────

    def _get_item(self, kind, ax=None, fig=None):
        for it in self.items:
            if it.kind != kind: continue
            if ax  is not None and it.ax  is ax:  return it
            if fig is not None and it.fig is fig:  return it
        return None

    def _get_tick_group(self, kind, ax):
        for g in self.tick_groups:
            if g.kind == kind and g.ax is ax: return g
        return None

    def _get_managed_axes(self):
        axes = []
        for it in self.items:
            if it.ax is not None and it.ax not in axes: axes.append(it.ax)
        for g in self.tick_groups:
            if g.ax is not None and g.ax not in axes: axes.append(g.ax)
        return axes

    # ── renderer ────────────────────────────────────────────────────

    def _get_renderer(self):
        """
        Draw first so renderer reflects post-colorbar positions.
        Without canvas.draw(), colorbar axes positions are stale.
        """
        c = self.fig.canvas
        if c is None: return None
        try: c.draw()
        except Exception: pass
        try: return c.get_renderer()
        except Exception: return None

    # ── bbox helpers ─────────────────────────────────────────────────

    def _artist_bbox(self, artist, r):
        if artist is None or r is None: return None
        try:
            b = artist.get_window_extent(r)
            return b if (b and b.width > 0 and b.height > 0) else None
        except Exception: return None

    def _group_bbox(self, g, r):
        if g is None: return None
        return _union_bboxes([self._artist_bbox(a, r) for a in _ensure_list(g.artists)])

    def _native_tick_bbox(self, ax, axis, r):
        if r is None: return None
        lbls = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
        bbs  = []
        for lbl in lbls:
            try:
                if not lbl.get_visible() or not lbl.get_text(): continue
                b = lbl.get_window_extent(r)
                if b and b.width > 0 and b.height > 0: bbs.append(b)
            except Exception: pass
        return _union_bboxes(bbs)

    def _total_x_tick_bbox(self, ax, r):
        return _union_bboxes([
            self._group_bbox(self._get_tick_group("xticks", ax), r),
            self._native_tick_bbox(ax, "x", r),
        ])

    def _total_y_tick_bbox(self, ax, r):
        return _union_bboxes([
            self._group_bbox(self._get_tick_group("yticks", ax), r),
            self._native_tick_bbox(ax, "y", r),
        ])

    def _x_tick_outward_px(self, ax, r):
        b = self._total_x_tick_bbox(ax, r)
        if b is None: return 0.0
        try: return max(0.0, ax.get_window_extent(r).y0 - b.y0)
        except Exception: return 0.0

    # ── colorbar detection ───────────────────────────────────────────

    def _ylabel_blocked(self, ax, r, cx_px: float, screen_w: float) -> bool:
        """
        Return True if placing the ylabel at cx_px would overlap any
        other axes (including colorbar axes with their tick labels).

        Uses two tests:
        1. Direct image bbox collision with any other axes bbox.
        2. Whether cx_px falls inside the gap between any other axes
           and the ytick labels — i.e. cx_px is to the RIGHT of any
           other axes left edge and LEFT of the ytick left edge.
           This catches colorbars whose axes bbox (including tick labels)
           extends past cx_px even if the colorbar bar itself does not.
        """
        label_left  = cx_px - screen_w / 2.0 - 2.0
        label_right = cx_px + screen_w / 2.0 + 2.0

        for other in self.fig.axes:
            if other is ax: continue
            try:
                ob = other.get_window_extent(r)
            except Exception: continue
            if ob.width <= 0 or ob.height <= 0: continue

            # Test 1: direct bbox overlap
            if label_right > ob.x0 and label_left < ob.x1:
                return True

            # Test 2: cx_px is inside the gap between this axes and ours.
            # If another axes has its LEFT edge to the LEFT of cx_px,
            # it means cx_px is in the territory of or to the right of
            # that other axes — which means we are in the gap region
            # between that axes and the ytick labels.
            # This catches colorbars that end before cx_px but whose
            # visual presence occupies the gap.
            if ob.x0 < cx_px and ob.x1 > label_left:
                return True

        return False

    # ── tick placement ───────────────────────────────────────────────

    def _tick_ba(self, ha, va):
        hm = {"left": 0.0, "center": 0.5, "right": 1.0}
        vm = {"bottom": 0.0, "baseline": 0.0,
              "center": 0.5, "middle": 0.5, "top": 1.0}
        return (hm.get((ha or "center").lower(), 0.5),
                vm.get((va or "center").lower(), 0.5))

    def _xpx(self, ax, pos):
        pts = ax.transData.transform(
            np.column_stack([pos, np.zeros(len(pos))]))
        return [float(p[0]) for p in pts]

    def _ypx(self, ax, pos):
        pts = ax.transData.transform(
            np.column_stack([np.zeros(len(pos)), pos]))
        return [float(p[1]) for p in pts]

    @staticmethod
    def _filt_extents(centers, extents, gap=6.0):
        """
        Greedy 1-D overlap filter. Processes labels in ascending center
        order (positions may arrive descending, e.g. imshow with
        origin="upper") and hides any label whose span would overlap the
        previously kept one.
        """
        vis = [True] * len(centers)
        last = None
        for i in sorted(range(len(centers)), key=lambda k: centers[k]):
            lo = centers[i] - extents[i] / 2
            hi = centers[i] + extents[i] / 2
            if last is None or lo >= last + gap:
                last = hi
            else:
                vis[i] = False
        return vis

    def _filt_x(self, cx, wx, gap=6.0):
        return self._filt_extents(cx, wx, gap)

    def _filt_y(self, cy, hy, gap=6.0):
        return self._filt_extents(cy, hy, gap)

    def _place_xticks(self, g):
        ax = g.ax
        for a in _ensure_list(g.artists): _remove_artist_safe(a)
        g.artists = []; g.last_sizes_px = []
        if g.hide_native: ax.set_xticklabels([])
        ax.set_xticks(list(g.positions))
        trans = ax.get_xaxis_transform()
        ba    = self._tick_ba(g.ha, g.va)
        gap   = _pixels_to_axes_dy(ax, self.X_TICK_GAP_PX)
        pays  = [self._render_tick_image(g, lbl) for lbl in g.labels]
        vis   = ([True] * len(g.positions) if not g.collision_avoidance
                 else self._filt_x(self._xpx(ax, g.positions), [p[3] for p in pays]))
        for i, pos in enumerate(g.positions):
            if not vis[i]: continue
            fs, _, img, dw, dh = pays[i]
            z  = g.zoom if g.zoom is not None else 0.40
            ab = AnnotationBbox(OffsetImage(img, zoom=z), (pos, -gap),
                                xycoords=trans, frameon=False,
                                box_alignment=ba, zorder=g.zorder,
                                annotation_clip=False)
            ax.add_artist(ab)
            g.artists.append(ab); g.last_sizes_px.append((dw, dh))

    def _place_yticks(self, g):
        ax = g.ax
        for a in _ensure_list(g.artists): _remove_artist_safe(a)
        g.artists = []; g.last_sizes_px = []
        if g.hide_native: ax.set_yticklabels([])
        ax.set_yticks(list(g.positions))
        trans = ax.get_yaxis_transform()
        ba    = self._tick_ba(g.ha, g.va)
        gap   = _pixels_to_axes_dx(ax, self.Y_TICK_GAP_PX)
        pays  = [self._render_tick_image(g, lbl) for lbl in g.labels]
        vis   = ([True] * len(g.positions) if not g.collision_avoidance
                 else self._filt_y(self._ypx(ax, g.positions), [p[4] for p in pays]))
        for i, pos in enumerate(g.positions):
            if not vis[i]: continue
            fs, _, img, dw, dh = pays[i]
            z  = g.zoom if g.zoom is not None else 0.40
            ab = AnnotationBbox(OffsetImage(img, zoom=z), (-gap, pos),
                                xycoords=trans, frameon=False,
                                box_alignment=ba, zorder=g.zorder,
                                annotation_clip=False)
            ax.add_artist(ab)
            g.artists.append(ab); g.last_sizes_px.append((dw, dh))

    # ── label / title placement ──────────────────────────────────────

    def _place_title(self, item):
        ax, fig = item.ax, item.ax.figure
        bp = ax.get_position()
        _, _, img, _, rh = self._render_item_image(item)
        gap_px = self.TITLE_TO_AXES_GAP_PX + item.extra_pad_px
        # Anchor the title's BOTTOM edge at a fixed gap above the axes,
        # so tall glyphs (conjunct stacks, ascenders) grow upward instead
        # of pushing the title into the axes frame.
        y = min(0.985, bp.y1 + _pixels_to_fig_dy(fig, gap_px))

        loc = (item.loc or "center").lower()
        if loc == "left":
            x, ba_x = bp.x0, 0.0
        elif loc == "right":
            x, ba_x = bp.x1, 1.0
        else:
            x, ba_x = bp.x0 + bp.width / 2.0, 0.5

        ab = AnnotationBbox(
            OffsetImage(img, zoom=item.zoom),
            (x, y),
            xycoords=fig.transFigure, frameon=False,
            box_alignment=(ba_x, 0.0), zorder=item.zorder,
        )
        fig.add_artist(ab); return ab

    def _place_suptitle(self, item, r):
        """
        Place the figure suptitle above everything: axes tops, per-axes
        titles, and any artists already measured by the renderer.
        """
        fig = item.fig
        _, fh_r = _renderer_px_size(fig, r)

        # Highest occupied point, as a figure fraction (renderer units
        # normalized by the renderer canvas so savefig dpi cancels out).
        top_frac = 0.0
        for ax in fig.axes:
            try:
                top_frac = max(top_frac, ax.get_window_extent(r).y1 / fh_r)
            except Exception:
                top_frac = max(top_frac, ax.get_position().y1)
        for other in self.items:
            if other.kind == "title" and other.artist is not None:
                bb = self._artist_bbox(other.artist, r)
                if bb is not None:
                    top_frac = max(top_frac, bb.y1 / fh_r)

        _, _, img, _, rh = self._render_item_image(item)
        gap_frac = _pixels_to_fig_dy(fig, self.SUPTITLE_GAP_PX + item.extra_pad_px)
        y = min(0.995, top_frac + gap_frac)

        ab = AnnotationBbox(
            OffsetImage(img, zoom=item.zoom),
            (0.5, y),
            xycoords=fig.transFigure, frameon=False,
            box_alignment=(0.5, 0.0), zorder=item.zorder,
        )
        fig.add_artist(ab); return ab

    def _place_xlabel(self, item, r):
        ax, fig = item.ax, item.ax.figure
        bp = ax.get_position()
        _, _, img, _, rh = self._render_item_image(item)
        # tick_px is renderer-measured: normalize by the renderer canvas.
        # Constant gaps and rendered sizes are in fig.dpi pixels.
        _, fh_r = _renderer_px_size(fig, r)
        tick_frac = self._x_tick_outward_px(ax, r) / max(1.0, fh_r)
        const_frac = _pixels_to_fig_dy(
            fig, self.X_LABEL_BASE_GAP_PX + item.extra_pad_px + rh / 2.0
        )
        y = max(0.0, bp.y0 - tick_frac - const_frac)
        ab = AnnotationBbox(
            OffsetImage(img, zoom=item.zoom),
            (bp.x0 + bp.width / 2.0, y),
            xycoords=fig.transFigure, frameon=False,
            box_alignment=(0.5, 0.5), zorder=item.zorder,
        )
        fig.add_artist(ab); return ab

    def _place_ylabel(self, item, r):
        """
        Place the ylabel just left of the ytick labels.

        Formula:
            screen_width = img_height * zoom  (image is rotated 90°)
            cx_px = strip_right - GAP - screen_width / 2

        where strip_right = ytick labels left edge (renderer-measured).

        THE KEY FIX:
            x_fig is clamped to [0.01, axes_left - small_margin]
            NOT to a fixed 0.48 maximum.
            Fixed 0.48 was wrong for right-side subplots where the
            correct x fraction can be > 0.48.

        Skips ylabel silently if a colorbar blocks the strip.
        User zoom is used exactly — no manipulation.
        """
        ax, fig = item.ax, item.ax.figure
        fw_r, _ = _renderer_px_size(fig, r)

        # ytick labels left edge as a figure fraction (renderer units
        # over renderer canvas, so savefig dpi cancels out)
        ytick_bb = self._total_y_tick_bbox(ax, r)
        if ytick_bb is not None:
            strip_frac = ytick_bb.x0 / fw_r
        else:
            try:    strip_frac = ax.get_window_extent(r).x0 / fw_r
            except Exception: strip_frac = ax.get_position().x0

        # Render (user zoom unchanged)
        _, _, img, _, _ = self._render_item_image(item)
        rotated  = np.rot90(img, k=1)

        # After 90° rotation: screen_width = original_img_height * zoom
        # (fig.dpi pixels, so normalize with the fig.dpi canvas)
        img_h    = item.last_image_size_px[1]
        screen_w_frac = _pixels_to_fig_dx(fig, img_h * item.zoom)

        # Anchor the label's RIGHT edge a fixed gap left of the ytick
        # strip. Right-edge anchoring means a cramped left margin makes
        # the label encroach on the ticks (visible, recoverable via
        # auto margins) instead of being pushed off the figure edge.
        gap_frac = _pixels_to_fig_dx(fig, self.Y_LABEL_BASE_GAP_PX + item.extra_pad_px)
        right_frac = strip_frac - gap_frac
        # Keep the label fully inside the figure.
        right_frac = max(screen_w_frac + 0.004, right_frac)

        # Skip if the computed position overlaps any other axes (e.g. a
        # colorbar sitting between subplots). Blocked-test coordinates
        # are in renderer pixels.
        cx_px = (right_frac - screen_w_frac / 2.0) * fw_r
        if self._ylabel_blocked(ax, r, cx_px, screen_w_frac * fw_r):
            return None

        x_fig = right_frac

        bp    = ax.get_position()
        y_fig = bp.y0 + bp.height / 2.0

        ab = AnnotationBbox(
            OffsetImage(rotated, zoom=item.zoom),
            (x_fig, y_fig),
            xycoords=fig.transFigure, frameon=False,
            box_alignment=(1.0, 0.5), zorder=item.zorder,
        )
        fig.add_artist(ab)
        item.last_render_size_px = (
            img_h * item.zoom,
            item.last_image_size_px[0] * item.zoom,
        )
        return ab

    # ── full placement pass ──────────────────────────────────────────

    def _place_all_artists(self):
        for g in self.tick_groups:
            if   g.kind == "xticks": self._place_xticks(g)
            elif g.kind == "yticks": self._place_yticks(g)

        r = self._get_renderer()

        for it in self.items:
            if it.kind in ("title", "suptitle"): continue
            _remove_artist_safe(it.artist); it.artist = None
            if   it.kind == "xlabel": it.artist = self._place_xlabel(it, r)
            elif it.kind == "ylabel": it.artist = self._place_ylabel(it, r)

        for it in self.items:
            if it.kind != "title": continue
            _remove_artist_safe(it.artist)
            it.artist = self._place_title(it)

        # Suptitles go last: they stack above the freshly placed titles.
        suptitles = [it for it in self.items if it.kind == "suptitle"]
        if suptitles:
            r2 = self._get_renderer()
            for it in suptitles:
                _remove_artist_safe(it.artist)
                it.artist = self._place_suptitle(it, r2)

    def update_layout(self):
        if self._is_updating: return
        self._is_updating = True
        try: self._place_all_artists()
        finally: self._is_updating = False

    # ── auto_adjust_margins ──────────────────────────────────────────

    def auto_adjust_margins(
        self,
        min_left=0.12, min_right=0.90,
        min_bottom=0.12, min_top=0.88,
        extra_left_px=10.0, extra_bottom_px=10.0,
        extra_top_px=18.0, extra_right_px=6.0,
    ):
        fig = self.fig
        self._place_all_artists()
        try: fig.canvas.draw()
        except Exception: pass
        r = self._get_renderer()
        if r is None: return

        fw = fig.get_size_inches()[0] * fig.dpi
        fh = fig.get_size_inches()[1] * fig.dpi
        axes_seen = self._get_managed_axes()

        left_px = 0.0; bot_px = 0.0
        title_bbs = []

        for ax in axes_seen:
            axb = ax.get_window_extent(r)
            xi  = self._get_item("xlabel", ax=ax)
            yi  = self._get_item("ylabel", ax=ax)
            ti  = self._get_item("title",  ax=ax)
            xb  = self._total_x_tick_bbox(ax, r)
            yb  = self._total_y_tick_bbox(ax, r)
            xbb = self._artist_bbox(xi.artist, r) if xi else None
            ybb = self._artist_bbox(yi.artist, r) if yi else None
            title_bbs.append(self._artist_bbox(ti.artist, r) if ti else None)
            si = self._get_item("suptitle", fig=self.fig)
            if si is not None:
                title_bbs.append(self._artist_bbox(si.artist, r))

            lu = _union_bboxes([yb, ybb])
            if lu:
                left_px = max(left_px,
                    max(0.0, axb.x0 - lu.x0)
                    + self.LEFT_MARGIN_SAFETY_PX + extra_left_px)

            bu = _union_bboxes([xb, xbb])
            if bu:
                bot_px = max(bot_px,
                    max(0.0, axb.y0 - bu.y0)
                    + self.BOTTOM_MARGIN_SAFETY_PX + extra_bottom_px)

        left   = max(min_left,   left_px / max(1.0, fw) + 0.01)
        bottom = max(min_bottom, bot_px  / max(1.0, fh) + 0.01)
        right  = min_right
        top    = min_top

        tu = _union_bboxes(title_bbs)
        if tu:
            prot = max(0.0, tu.y1 - fh)
            if prot > 0:
                top = min(top,
                    1.0 - (prot + self.TOP_MARGIN_SAFETY_PX + extra_top_px)
                    / max(1.0, fh))

        # Reserve headroom for the title/suptitle stack. Without this,
        # a suptitle hits the figure-edge clamp in _place_suptitle and
        # gets squashed against the title instead of sitting above it.
        stack_px = 0.0
        for ax in axes_seen:
            ti = self._get_item("title", ax=ax)
            if ti is not None:
                _, _, _, _, rh = self._render_item_image(ti)
                stack_px = max(
                    stack_px,
                    rh + self.TITLE_TO_AXES_GAP_PX + ti.extra_pad_px,
                )
        si = self._get_item("suptitle", fig=self.fig)
        if si is not None:
            _, _, _, _, rh_s = self._render_item_image(si)
            stack_px += rh_s + self.SUPTITLE_GAP_PX + si.extra_pad_px
        if stack_px > 0.0:
            top = min(top, 1.0 - (stack_px + self.TOP_MARGIN_SAFETY_PX) / max(1.0, fh))

        left   = min(max(left,   0.0), 0.95)   # wide upper bound for multi-subplot
        right  = min(max(right,  0.05), 1.0)
        bottom = min(max(bottom, 0.0), 0.45)
        top    = min(max(top,    0.45), 1.0)

        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        self._place_all_artists()
        try: fig.canvas.draw_idle()
        except Exception: pass

    def summary(self):
        return {
            "figure_id": id(self.fig),
            "managed_items":      [it.kind for it in self.items],
            "managed_tick_groups":[g.kind  for g  in self.tick_groups],
            "num_items":       len(self.items),
            "num_tick_groups": len(self.tick_groups),
        }


# ─────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────

def get_layout_manager(fig) -> UniversalLayoutManager:
    key = id(fig)
    if key not in _MANAGER_REGISTRY:
        _MANAGER_REGISTRY[key] = UniversalLayoutManager(fig)
    return _MANAGER_REGISTRY[key]


def clear_layout_manager(fig) -> None:
    key = id(fig)
    m   = _MANAGER_REGISTRY.pop(key, None)
    if m is not None:
        m.clear(); m.disconnect()
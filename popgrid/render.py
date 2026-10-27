"""
Matplotlib rendering for popgrid visualisations.

Layout strategy
---------------
Secondary landmasses are classified into two groups:

  inline   — Cell bbox overlaps with (or is within INLINE_MARGIN_CELLS of)
             the main map bbox.  Plotted directly on the main axes at their
             correct geographic position.
             Example: Balearic Islands (east of Valencia), Ceuta & Melilla
             (south of Andalucía).

  panel    — Far outside the main map extent.  Each gets its own subplot
             panel in a row beneath the main map, enclosed in a visible
             border box with a label.
             Example: Canary Islands, Guyane française.

Panel heights are computed at the SAME visual scale as the main map
(same inches-per-metre ratio), so territories appear correctly sized
relative to the mainland.  Guyane française (15% of France's land area)
therefore occupies roughly 15% of the height of metropolitan France.

Non-distinct micro-islands (tiny islets whose region already appears in
the main landmass) are silently filtered out.
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patheffects import withStroke

logger = logging.getLogger(__name__)

# How many cells of tolerance to use when deciding inline vs panel.
# 5 cells ≈ 112 km for Spain, enough to include Ceuta & Melilla inline.
INLINE_MARGIN_CELLS: int = 20


# ── Colour palettes ───────────────────────────────────────────────────────────

_PALETTE_DATABITES: list[str] = [
    "#2D9B4E", "#1A3829", "#38B86E", "#5E7D6A", "#A8DDC4",
    "#4CAF7D", "#3D8B5E", "#7BC99A", "#246840", "#66BB90",
    "#29744A", "#9AE0AD", "#2E8054", "#9DD4B3", "#52A870",
    "#1F5C36", "#6BBF85", "#84D09A", "#3D8B5E", "#4CAF7D",
]
_PALETTE_QUAL: list[str] = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#499894", "#86BCB6", "#D37295", "#FABFD2", "#B6992D",
    "#F1CE63", "#79706E", "#D4A6C8", "#A0CBE8", "#FFBE7D",
]
_PALETTE_BOLD: list[str] = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#a9a9a9", "#ffffff", "#000000",
]
PALETTES: dict[str, list[str]] = {
    "databites": _PALETTE_DATABITES,
    "qual": _PALETTE_QUAL,
    "bold": _PALETTE_BOLD,
}


# ── Style ─────────────────────────────────────────────────────────────────────

#: Every visual knob that ``draw`` / ``plot`` accept through ``style=``.
#: ``None`` means "auto": popgrid derives the value from the figure size,
#: the background colour, or the region colour, exactly as it did before
#: these options existed.
DEFAULT_STYLE: dict = {
    "edge_color": "#000000",   # region outline colour
    "edge_width": None,        # outline width in points (None → 8% of the cell)
    "grid_color": None,        # background grid lines / tiles (None → auto)
    "grid_width": 0.35,        # background grid line width in points
    "cell_gap": None,          # gap between blocks, fraction of a cell (None → auto)
    "label_color": None,       # None → black or white, by region luminance
    "label_size": None,        # fixed label size in points (None → fit to region)
    "label_min_size": 7.0,     # smallest auto label, in points
    "label_max_size": 18.0,    # largest auto label, in points
    "label_weight": "bold",
    "label_font": None,        # font family (None → matplotlib default)
}

LABEL_MODES = ("auto", "full", "short", "none")


def resolve_style(style: dict | None) -> dict:
    """Merge a user ``style`` dict over :data:`DEFAULT_STYLE`, rejecting typos."""
    if style is None:
        return dict(DEFAULT_STYLE)
    unknown = sorted(set(style) - set(DEFAULT_STYLE))
    if unknown:
        raise ValueError(
            f"Unknown style key(s): {unknown}. "
            f"Valid keys: {sorted(DEFAULT_STYLE)}"
        )
    merged = dict(DEFAULT_STYLE)
    merged.update({k: v for k, v in style.items() if v is not None})
    return merged


# ── Colour utilities ──────────────────────────────────────────────────────────

def _luminance(hex_color: str) -> float:
    r, g, b = mcolors.to_rgb(hex_color)
    return 0.299 * r + 0.587 * g + 0.114 * b

def _text_color(bg: str) -> str:
    return "#ffffff" if _luminance(bg) < 0.48 else "#111111"

def build_color_map(
    region_names: list[str],
    palette: str | list[str] = "qual",
) -> dict[str, str]:
    unique = sorted(r for r in set(region_names) if r is not None)
    colors = PALETTES.get(palette, palette) if isinstance(palette, str) else palette  # type: ignore[arg-type]
    if not isinstance(colors, list):
        colors = _PALETTE_QUAL
    return {name: colors[i % len(colors)] for i, name in enumerate(unique)}


def _color_distance(a: str, b: str) -> float:
    """Perceptual-ish RGB distance between two hex colours (0–~1.7)."""
    ar, ag, ab = mcolors.to_rgb(a)
    br, bg, bb = mcolors.to_rgb(b)
    # Weighted Euclidean (eye is most sensitive to green, least to blue).
    return (0.30 * (ar - br) ** 2 + 0.59 * (ag - bg) ** 2 + 0.11 * (ab - bb) ** 2) ** 0.5


def build_adjacency_color_map(
    region_names: list[str],
    adjacency: dict[str, set],
    palette: str | list[str] = "qual",
    sizes: dict[str, int] | None = None,
) -> dict[str, str]:
    """
    Assign a palette colour to every region so that **adjacent regions are
    always visually distinct** and, additionally, **the largest regions are
    distinct from one another** even when they are not adjacent.

    Regions are coloured largest-first (by ``sizes``, falling back to adjacency
    degree).  Each region takes the palette colour that maximises the minimum
    colour distance to its already-coloured neighbours; among the biggest
    regions an extra reward keeps them far apart in colour space, so two large
    blocks never share a hue (e.g. Japan's Kanto and Shikoku, China's Henan and
    Sichuan).  Ties break toward the least-used colour to keep the map balanced.
    """
    from collections import Counter

    unique = sorted(r for r in set(region_names) if r is not None)
    colors = PALETTES.get(palette, palette) if isinstance(palette, str) else palette  # type: ignore[arg-type]
    if not isinstance(colors, list) or not colors:
        colors = _PALETTE_QUAL

    sizes = sizes or {}
    # Largest first (then most-connected), so the dominant regions claim
    # well-separated colours before the small ones fill in around them.
    order = sorted(
        unique,
        key=lambda r: (-sizes.get(r, 0), -len(adjacency.get(r, set()))),
    )
    # The "big" regions that must be mutually distinct: the top regions by
    # size, capped by how many palette colours exist.
    n_big = min(len(colors), max(1, len(order) // 2))
    big_set = set(order[:n_big])

    assigned: dict[str, str] = {}
    big_colors: list[str] = []
    usage: Counter = Counter()

    for region in order:
        neigh_colors = [
            assigned[n] for n in adjacency.get(region, set()) if n in assigned
        ]
        best, best_score = colors[0], -1e9
        for c in colors:
            d_neigh = min((_color_distance(c, nc) for nc in neigh_colors), default=10.0)
            score = d_neigh - 0.05 * usage[c]
            if region in big_set and big_colors:
                # Reward distance from the colours already taken by big regions.
                score += 0.6 * min(_color_distance(c, bc) for bc in big_colors)
            if score > best_score:
                best_score, best = score, c
        assigned[region] = best
        usage[best] += 1
        if region in big_set:
            big_colors.append(best)

    return assigned


# ── Cell rendering ────────────────────────────────────────────────────────────

def _grid_color(background_color: str) -> str:
    """
    Compute a slightly darker shade of background_color for the grid.

    The grid squares are full-size with this fill and no border.
    The 10% gap between colored cells exposes the grid fill as a single
    clean grid line — no double-line artifacts.
    """
    r, g, b = mcolors.to_rgb(background_color)
    # Darken by ~8% — subtle but clearly visible as a grid
    factor = 0.88
    return mcolors.to_hex((r * factor, g * factor, b * factor))


def _full_grid_background(
    ax: plt.Axes,
    cx_vals: np.ndarray,
    cy_vals: np.ndarray,
    cell_size: float,
    background_color: str,
    style: str = "grid",
    grid_color: Optional[str] = None,
    grid_width: float = 0.35,
) -> None:
    """
    Draw a background grid or solid fill over the cell bounding box.

    ``grid_color`` overrides the automatic colour (light grey lines for
    ``'grid'``, a darkened background for ``'solid'``); ``grid_width`` is the
    line width in points for the ``'grid'`` style.

    Alignment guarantee
    -------------------
    The grid is anchored to ``cx_vals.min()`` and ``cy_vals.min()`` — the
    actual positions of the coloured cells — and extended outward by integer
    multiples of ``cell_size``.  This ensures every grid line or square
    boundary falls exactly on a coloured-cell boundary (no sub-pixel offset).

    The previous approach (``floor(x / cell_size) * cell_size + half``)
    produced an arbitrary fractional offset that made the background grid
    visibly misaligned with the coloured squares.

    style='grid'
        Light-grey lines drawn at every cell boundary across the full
        bounding box.  No fill, no squares — just the lines.  Gaps between
        coloured cells expose the lines underneath.

    style='solid'
        Full-size filled squares (100%), no gaps.  Tiles into a uniform
        solid-coloured rectangle.
    """
    from matplotlib.collections import LineCollection

    half = cell_size / 2.0
    n_margin = 3          # cells of margin around the country extent
    grid_fill = grid_color or _grid_color(background_color)
    line_color = grid_color or "#b8b8b8"

    # ── Anchor to actual cell positions ──────────────────────────────────────
    # Extend outward by integer multiples of cell_size so the grid and
    # the coloured cells share the same slot positions.
    x_min_cell = float(cx_vals.min())
    x_max_cell = float(cx_vals.max())
    y_min_cell = float(cy_vals.min())
    y_max_cell = float(cy_vals.max())

    x_start = x_min_cell - n_margin * cell_size   # centre of first grid column
    x_end   = x_max_cell + n_margin * cell_size
    y_start = y_min_cell - n_margin * cell_size
    y_end   = y_max_cell + n_margin * cell_size

    if style == "grid":
        # ── Lines at every cell boundary ─────────────────────────────────────
        # Boundaries are at centre ± half, so the first vertical line is at
        # x_start - half, then every cell_size after that.
        x_lines = np.arange(x_start - half, x_end + half + cell_size * 0.5, cell_size)
        y_lines = np.arange(y_start - half, y_end + half + cell_size * 0.5, cell_size)

        # Extent of the line field (slightly beyond outermost cells)
        left   = x_start - half
        right  = x_end   + half
        bottom = y_start - half
        top    = y_end   + half

        vsegs = [[(x, bottom), (x, top)]   for x in x_lines]
        hsegs = [[(left, y),   (right, y)] for y in y_lines]

        lc = LineCollection(
            vsegs + hsegs,
            colors=[line_color],
            linewidths=grid_width,
            zorder=1,
        )
        ax.add_collection(lc)

    else:  # style == "solid"
        # ── Full-size squares, no gaps ────────────────────────────────────────
        xs = np.arange(x_start, x_end + cell_size * 0.5, cell_size)
        ys = np.arange(y_start, y_end + cell_size * 0.5, cell_size)
        xx, yy = np.meshgrid(xs, ys)
        gx = xx.ravel()
        gy = yy.ravel()

        rects = [
            mpatches.Rectangle(xy=(cx - half, cy - half),
                               width=cell_size, height=cell_size)
            for cx, cy in zip(gx, gy)
        ]
        coll = PatchCollection(
            rects,
            facecolors=[grid_fill] * len(rects),
            linewidths=0,
            zorder=1,
        )
        ax.add_collection(coll)


def _plot_cells(
    ax: plt.Axes,
    cells_gdf,
    color_map: dict[str, str],
    cell_size: float,
    show_labels: bool,
    label_min_cells: int,
    gap_frac: float = 0.06,
    set_limits: bool = True,
    background_style: str = "grid",
    show_region_borders: bool = True,
    border_lw: float = 0.0,
    style: Optional[dict] = None,
    label_mode: str = "auto",
    labels: Optional[dict[str, str]] = None,
    cell_pts: float = 0.0,
) -> None:
    """
    Draw colored cells and optional region borders.

    background_style (default 'grid')
        Controls the cell gap size to match the background.

    show_region_borders=True (default)
        An outline is drawn around each region's total combined cell area,
        in ``style["edge_color"]``.

    border_lw
        Region border linewidth in points.  When 0 (default), falls back
        to the legacy formula ``cell_size / 18_000``.  create_figure always
        passes a pre-computed value proportional to the visual cell size so
        borders look consistent across countries of all sizes.
        ``style["edge_width"]`` wins over both when set.

    style, label_mode, labels, cell_pts
        See :func:`draw_map`.  ``cell_pts`` is the visual size of one cell in
        points; labels are sized to fit their region from it.
    """
    if cells_gdf is None or cells_gdf.empty:
        return

    st = resolve_style(style)
    cx_vals = cells_gdf["cx"].to_numpy()
    cy_vals = cells_gdf["cy"].to_numpy()
    half = cell_size / 2.0

    # ── 1. Cell gap ───────────────────────────────────────────────────────────
    # With background ('grid' or 'solid'): 10% gap exposes the background.
    # Without background ('none'): 4% cosmetic gap only.
    if st["cell_gap"] is not None:
        gap = float(np.clip(st["cell_gap"], 0.0, 0.5))
        inner = half * (1.0 - 2.0 * gap)
    else:
        inner = half * 0.90 if background_style != "none" else half * 0.96

    # ── 2. Colored cells ──────────────────────────────────────────────────────
    names: list[str] = cells_gdf["region_name"].fillna("Unknown").tolist()
    face_colors = [color_map.get(n, "#cccccc") for n in names]

    rects = [
        mpatches.Rectangle(xy=(cx - inner, cy - inner),
                           width=2 * inner, height=2 * inner)
        for cx, cy in zip(cx_vals, cy_vals)
    ]
    coll = PatchCollection(rects, facecolors=face_colors, linewidths=0, zorder=2)
    ax.add_collection(coll)

    # ── 3. Region borders ─────────────────────────────────────────────────────
    if show_region_borders:
        import shapely
        from shapely.ops import unary_union

        # Explicit style wins; then the pre-computed visual-scale linewidth;
        # then the legacy formula for callers outside create_figure / draw_map.
        if st["edge_width"] is not None:
            lw = float(st["edge_width"])
        elif border_lw > 0:
            lw = border_lw
        else:
            lw = float(np.clip(cell_size / 18_000, 1.2, 3.0))
        edge_color = st["edge_color"]

        # Buffer epsilon: bridges diagonal-adjacency gaps so cells that only
        # touch at a corner still merge into one continuous region outline.
        # Without this, regions like Castilla-La Mancha produce 32 separate
        # polygons and their individual outlines draw lines INSIDE the region.
        eps = half * 0.08  # 8% of half-cell — closes diagonal gaps, keeps shape

        for region_name, grp in cells_gdf.groupby("region_name"):
            boxes = shapely.box(
                grp["cx"].to_numpy() - half,
                grp["cy"].to_numpy() - half,
                grp["cx"].to_numpy() + half,
                grp["cy"].to_numpy() + half,
            )
            # Buffer out slightly → merge diagonal neighbours → buffer back in
            buffered = shapely.buffer(
                boxes, eps, cap_style="square", join_style="mitre"
            )
            outline = unary_union(buffered)
            outline = shapely.buffer(
                outline, -eps * 0.6, cap_style="square", join_style="mitre"
            )

            geoms = (list(outline.geoms)
                     if outline.geom_type == "MultiPolygon"
                     else [outline])

            for geom in geoms:
                if geom.is_empty:
                    continue
                xs, ys = geom.exterior.xy
                ax.plot(xs, ys, color=edge_color, linewidth=lw,
                        zorder=5, solid_capstyle="round",
                        solid_joinstyle="round")
                for interior in geom.interiors:
                    xs, ys = interior.xy
                    ax.plot(xs, ys, color=edge_color, linewidth=lw,
                            zorder=5, solid_capstyle="round",
                            solid_joinstyle="round")

    if set_limits:
        margin = cell_size * 1.2
        ax.set_xlim(cx_vals.min() - margin, cx_vals.max() + margin)
        ax.set_ylim(cy_vals.min() - margin, cy_vals.max() + margin)

    ax.set_aspect("equal")
    ax.axis("off")

    if show_labels and label_mode != "none":
        _label_regions(
            ax, cells_gdf, color_map, cell_size,
            label_min_cells=label_min_cells, style=st,
            label_mode=label_mode, labels=labels, cell_pts=cell_pts,
        )


# ── Region labels ─────────────────────────────────────────────────────────────

# Lower-case connectives dropped when abbreviating a multi-word region name.
_STOP_WORDS = {
    "de", "del", "la", "las", "los", "el", "y", "i", "e", "da", "do", "das",
    "dos", "of", "the", "and", "le", "les", "du", "des", "et", "van", "von",
    "der", "den", "di", "a", "o",
}

# Connectives that mark an "X of Y" name, where the last part can stand alone
# (``Foral de Navarra`` → ``Navarra``). Articles like ``la`` do not qualify:
# ``Castilla-La Mancha`` is one name, not Castilla of Mancha.
_GENITIVES = {"de", "del", "of", "du", "des", "di", "da", "do", "van", "von", "der"}

# Average glyph width as a fraction of the font size, for a bold sans face.
_GLYPH_EM = 0.62
_LINE_EM = 1.15


def _abbreviate(name: str) -> str:
    """
    Short form of a region name for cramped labels.

    Multi-word names become initials (``Castilla y León`` → ``CL``,
    ``New South Wales`` → ``NSW``); single words keep their first three
    letters (``Madrid`` → ``MAD``).  Lower-case connectives are dropped.
    """
    import re

    words = [w for w in re.split(r"[\s\-–/]+", name.strip()) if w]
    if not words:
        return name[:3].upper()
    kept = [w for w in words if w.lower() not in _STOP_WORDS] or words
    if len(kept) >= 2:
        return "".join(w[0].upper() for w in kept[:4])
    return kept[0][:3].upper()


def _key_word(name: str) -> Optional[str]:
    """
    The distinctive word of a name built around a connective, e.g.
    ``Foral de Navarra`` → ``Navarra``, ``Region of Murcia`` → ``Murcia``.
    Returns ``None`` for names without a connective, where no single word
    can stand in for the whole.
    """
    import re

    words = [w for w in re.split(r"[\s\-–/]+", name.strip()) if w]
    gen_idx = [k for k, w in enumerate(words) if w.lower() in _GENITIVES]
    if not gen_idx:
        return None
    # The part after the last connective is the name proper.
    after = [w for w in words[gen_idx[-1] + 1:] if w.lower() not in _STOP_WORDS]
    kept = after or [w for w in words if w.lower() not in _STOP_WORDS]
    return max(kept, key=len) if kept else None


def _wrap_name(name: str) -> Optional[list[str]]:
    """
    Split a name into two lines at the space or hyphen closest to its middle.
    Returns ``None`` when the name has no break point.
    """
    import re

    breaks = [m.end() for m in re.finditer(r"[\s]|(?<=-)", name)]
    breaks = [b for b in breaks if 0 < b < len(name)]
    if not breaks:
        return None
    best = min(breaks, key=lambda b: max(len(name[:b].rstrip()), len(name[b:].lstrip())))
    return [name[:best].rstrip(), name[best:].lstrip()]


def _largest_rectangle(comp: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    """
    Largest axis-aligned rectangle of cells fully inside ``comp``.

    Returns ``(i0, j0, cols, rows)`` with ``(i0, j0)`` the lower-left cell.
    Classic maximal-rectangle-in-binary-matrix via per-row histograms; the
    grids here are at most a few thousand cells so the cost is negligible.
    """
    i_min = min(i for i, _ in comp)
    i_max = max(i for i, _ in comp)
    j_min = min(j for _, j in comp)
    j_max = max(j for _, j in comp)
    width = i_max - i_min + 1
    heights = [0] * width
    best = (i_min, j_min, 1, 1)
    best_area = 0
    for j in range(j_min, j_max + 1):
        for k in range(width):
            heights[k] = heights[k] + 1 if (i_min + k, j) in comp else 0
        # Largest rectangle under the histogram ending at this row.
        stack: list[int] = []
        for k in range(width + 1):
            h = heights[k] if k < width else 0
            while stack and heights[stack[-1]] >= h:
                top = stack.pop()
                top_h = heights[top]
                left = stack[-1] + 1 if stack else 0
                area = top_h * (k - left)
                if area > best_area:
                    best_area = area
                    best = (i_min + left, j - top_h + 1, k - left, top_h)
            if k < width:
                stack.append(k)
    return best


def _largest_component(grp, cell_size: float) -> tuple[float, float, int, int]:
    """
    Centre (x, y) and size in cells (width, height) of the largest
    rectangle inscribed in the region's largest 4-connected block of cells.

    Labels sit on this rectangle rather than on the mean of all cells: for
    interleaved or L-shaped regions the plain mean can land on a one-cell
    strip or outside the region entirely; the inscribed rectangle is always
    solid region, and its size says how much text actually fits.
    """
    gcx = grp["cx"].to_numpy()
    gcy = grp["cy"].to_numpy()
    gminx, gminy = gcx.min(), gcy.min()
    keys = {
        (round((x - gminx) / cell_size), round((y - gminy) / cell_size)): (x, y)
        for x, y in zip(gcx, gcy)
    }
    seen: set = set()
    best: list = []
    for start in keys:
        if start in seen:
            continue
        comp = [start]
        stack = [start]
        seen.add(start)
        while stack:
            ci, cj = stack.pop()
            for nb in ((ci + 1, cj), (ci - 1, cj), (ci, cj + 1), (ci, cj - 1)):
                if nb in keys and nb not in seen:
                    seen.add(nb)
                    comp.append(nb)
                    stack.append(nb)
        if len(comp) > len(best):
            best = comp
    i0, j0, cols, rows = _largest_rectangle(set(best))
    # Rectangle centre in data coordinates
    x = gminx + (i0 + (cols - 1) / 2.0) * cell_size
    y = gminy + (j0 + (rows - 1) / 2.0) * cell_size
    return float(x), float(y), cols, rows


def _text_em(text: str, st: dict) -> float:
    """Rendered width of ``text`` in em units for the label font."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    try:
        prop = FontProperties(family=st.get("label_font"), weight=st.get("label_weight", "bold"))
        path = TextPath((0, 0), text, size=10, prop=prop)
        x0, _, x1, _ = path.get_extents().extents
        return max(float(x1 - x0), 1.0) / 10.0
    except Exception:  # noqa: BLE001  (missing font, headless quirks)
        return _GLYPH_EM * max(len(text), 1)


def _fit_size(lines: list[str], w_pts: float, h_pts: float, st: dict) -> float:
    """Largest font size (points) at which ``lines`` fit inside a w × h box."""
    widest_em = max(_text_em(line, st) for line in lines)
    by_width = (w_pts * 0.94) / widest_em
    by_height = (h_pts * 0.90) / (_LINE_EM * len(lines))
    return min(by_width, by_height)


def _choose_label(
    name: str,
    cols: int,
    rows: int,
    cell_pts: float,
    st: dict,
    label_mode: str,
    override: Optional[str],
) -> tuple[str, float]:
    """
    Pick the text and font size for one region label.

    ``auto`` tries the full name, then a two-line wrap, then the abbreviation,
    taking the first that fits at or above ``label_min_size``.  The
    abbreviation is used even when it does not fit, so every region above
    ``label_min_cells`` still gets a readable mark.
    """
    lo, hi = float(st["label_min_size"]), float(st["label_max_size"])
    fixed = st["label_size"]
    w_pts = cols * cell_pts
    h_pts = rows * cell_pts

    def size_for(lines: list[str]) -> float:
        if fixed is not None:
            return float(fixed)
        if cell_pts <= 0:  # no scale information: legacy heuristic
            return float(np.clip(5.0 + np.sqrt(cols * rows) * 0.9, lo, hi))
        return _fit_size(lines, w_pts, h_pts, st)

    def fits(lines: list[str]) -> bool:
        if cell_pts <= 0:
            return True
        if fixed is not None:
            return _fit_size(lines, w_pts, h_pts, st) >= float(fixed)
        return _fit_size(lines, w_pts, h_pts, st) >= lo

    if override is not None:
        text = override
        return text, float(np.clip(size_for(text.split("\n")), lo, hi))

    if label_mode == "short":
        text = _abbreviate(name)
        return text, float(np.clip(size_for([text]), lo, hi))

    wrapped = _wrap_name(name)
    if label_mode == "full":
        lines = [name]
        if not fits(lines) and wrapped and fits(wrapped):
            lines = wrapped
        return "\n".join(lines), float(np.clip(size_for(lines), lo, hi))

    # auto
    if fits([name]):
        return name, float(np.clip(size_for([name]), lo, hi))
    if wrapped and fits(wrapped):
        return "\n".join(wrapped), float(np.clip(size_for(wrapped), lo, hi))
    key_word = _key_word(name)
    if key_word and fits([key_word]):
        return key_word, float(np.clip(size_for([key_word]), lo, hi))
    # Abbreviations are a fallback, not a headline: keep them modest so a
    # three-letter code never outshouts a neighbour's full name.
    short = _abbreviate(name)
    return short, float(np.clip(size_for([short]), lo, min(hi, lo * 1.5)))


def _label_regions(
    ax: plt.Axes,
    cells_gdf,
    color_map: dict[str, str],
    cell_size: float,
    label_min_cells: int,
    style: dict,
    label_mode: str = "auto",
    labels: Optional[dict[str, str]] = None,
    cell_pts: float = 0.0,
) -> None:
    """Write one label per region with at least ``label_min_cells`` cells."""
    if label_mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be one of {LABEL_MODES}; got {label_mode!r}")
    labels = labels or {}
    for region_name, grp in cells_gdf.groupby("region_name"):
        if len(grp) < label_min_cells:
            continue
        name = str(region_name)
        if name in labels and labels[name] in ("", None):
            continue
        x, y, cols, rows = _largest_component(grp, cell_size)
        text, fontsize = _choose_label(
            name, cols, rows, cell_pts, style, label_mode, labels.get(name)
        )
        bg = color_map.get(region_name, "#cccccc")  # type: ignore[arg-type]
        fg = style["label_color"] or _text_color(bg)
        outline = "#00000066" if _luminance(fg) > 0.5 else "#ffffff99"
        ax.text(
            x, y, text,
            fontsize=fontsize, ha="center", va="center",
            fontweight=style["label_weight"], fontfamily=style["label_font"],
            color=fg, zorder=4, linespacing=1.05,
            path_effects=[withStroke(linewidth=max(1.5, fontsize * 0.16),
                                     foreground=outline)],
        )


def _apply_limits(ax: plt.Axes, cell_groups: list, cell_size: float) -> None:
    """Set axes limits to encompass all cell groups with a uniform margin."""
    all_cx = np.concatenate([g["cx"].to_numpy() for g in cell_groups if not g.empty])
    all_cy = np.concatenate([g["cy"].to_numpy() for g in cell_groups if not g.empty])
    margin = cell_size * 1.5
    ax.set_xlim(all_cx.min() - margin, all_cx.max() + margin)
    ax.set_ylim(all_cy.min() - margin, all_cy.max() + margin)


def _label_small_inline(
    ax: plt.Axes,
    cells_gdf,
    color_map: dict[str, str],
    cell_size: float,
    label_min_cells: int,
    style: Optional[dict] = None,
    labels: Optional[dict[str, str]] = None,
) -> None:
    """
    Add name annotations for inline regions that are too small to be
    labelled by the standard threshold (e.g. Ceuta, Melilla: 1 cell each).

    These get a small italic label placed just above the cell with a
    pointing offset so it doesn't overlap the coloured square.
    """
    st = resolve_style(style)
    labels = labels or {}
    fontsize = max(6.5, float(st["label_size"] or 0) * 0.75, float(st["label_min_size"]) * 0.9)
    for region_name, grp in cells_gdf.groupby("region_name"):
        if len(grp) >= label_min_cells:
            continue  # already handled by _plot_cells
        text = labels.get(str(region_name), str(region_name))
        if not text:
            continue
        cx = float(grp["cx"].mean())
        cy = float(grp["cy"].mean())
        ax.annotate(
            text,
            xy=(cx, cy),
            xytext=(cx + cell_size * 0.7, cy + cell_size * 1.1),
            fontsize=fontsize,
            ha="left",
            va="bottom",
            fontstyle="italic",
            fontfamily=st["label_font"],
            color=st["label_color"] or "#1A3829",
            zorder=5,
            arrowprops=dict(
                arrowstyle="-",
                color="#888888",
                lw=0.5,
            ),
        )


# ── Landmass classification ───────────────────────────────────────────────────

def _classify_secondary(
    main_lm: dict,
    secondary_lms: list[dict],
    cell_size: float,
    inline_margin_cells: int = INLINE_MARGIN_CELLS,
) -> tuple[list[dict], list[dict]]:
    """
    Split secondary landmasses into ``inline`` and ``panel`` groups.

    inline
        Cell bbox overlaps (within ``inline_margin_cells * cell_size``) with
        the main map bbox.  Plotted at correct geographic position on main axes.

    panel
        Everything else — rendered as a bordered subplot panel.

    Non-distinct micro-islands are silently filtered.
    """
    main_cells = main_lm["cells"]
    main_x0 = main_cells["cx"].min()
    main_x1 = main_cells["cx"].max()
    main_y0 = main_cells["cy"].min()
    main_y1 = main_cells["cy"].max()
    margin = inline_margin_cells * cell_size

    main_region_names = set(main_cells["region_name"].dropna())

    inline: list[dict] = []
    panel: list[dict] = []

    for lm in secondary_lms:
        cells = lm["cells"]
        if cells.empty:
            continue

        # Drop non-distinct micro-islands (region already in main map)
        lm_names = set(cells["region_name"].dropna())
        if not (lm_names - main_region_names):
            logger.debug("Landmass %d filtered (non-distinct: %s)", lm["id"], lm_names)
            continue

        lm_x0, lm_x1 = cells["cx"].min(), cells["cx"].max()
        lm_y0, lm_y1 = cells["cy"].min(), cells["cy"].max()

        x_ok = lm_x0 <= main_x1 + margin and lm_x1 >= main_x0 - margin
        y_ok = lm_y0 <= main_y1 + margin and lm_y1 >= main_y0 - margin

        if x_ok and y_ok:
            inline.append(lm)
            logger.debug("Landmass %d (%s) → inline (margin %.0fkm)",
                         lm["id"], lm_names, margin / 1000)
        else:
            panel.append(lm)
            logger.debug("Landmass %d (%s) → panel inset", lm["id"], lm_names)

    return inline, panel


# ── Panel helpers ─────────────────────────────────────────────────────────────

def _panel_label(lm: dict) -> str:
    names = lm["cells"]["region_name"].dropna().unique().tolist()
    return names[0] if len(names) == 1 else ", ".join(sorted(names))


def _panel_data_dims(lm: dict, cell_size: float) -> tuple[float, float]:
    """
    Data-space width and height (metres) of a panel, including margin.

    Single-cell territories use 1.5× cell_size so the cell fills the
    panel with a small breathing margin — not 3× (which made them look
    oversized relative to the main map).
    """
    cells = lm["cells"]
    margin = cell_size * 1.5
    if len(cells) == 1:
        w = h = cell_size * 1.5
    else:
        w = (cells["cx"].max() - cells["cx"].min()) + 2 * margin
        h = (cells["cy"].max() - cells["cy"].min()) + 2 * margin
    return w, h


# ── Panel grouping ───────────────────────────────────────────────────────────


def _group_panels_by_region(panel_lms: list[dict]) -> list[dict]:
    """
    Merge panel landmass dicts that share the same primary region name.

    Countries like the USA have Alaska and Hawaii each split into many
    separate island clusters by ``detect_landmasses``.  Without grouping,
    every Aleutian island becomes its own tiny panel.  This function merges
    all clusters that belong to the same region (e.g. all "Alaska" clusters
    → one combined "Alaska" panel) so the panel row stays readable.

    Returns panels sorted by total cell count, largest first.
    """
    import pandas as pd
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for lm in panel_lms:
        cells = lm["cells"]
        if cells.empty:
            continue
        primary = cells["region_name"].mode().iloc[0] if len(cells) > 0 else "Unknown"
        groups[primary].append(lm)

    result: list[dict] = []
    for name, lms in groups.items():
        if len(lms) == 1:
            result.append(lms[0])
        else:
            # Merge all cell DataFrames; keep largest geom as representative
            merged_cells = pd.concat(
                [lm["cells"] for lm in lms], ignore_index=True
            )
            base = max(lms, key=lambda x: len(x["cells"]))
            result.append({
                "id": base["id"],
                "geom": base["geom"],
                "region_idx": base["region_idx"],
                "cells": merged_cells,
            })

    result.sort(key=lambda lm: len(lm["cells"]), reverse=True)
    logger.debug(
        "Panel grouping: %d clusters → %d panels",
        len(panel_lms),
        len(result),
    )
    return result


# ── Draw into an existing Axes ────────────────────────────────────────────────

def _axes_size_in(ax: plt.Axes) -> tuple[float, float]:
    """Width and height of ``ax`` in inches, from its figure-relative box."""
    fig = ax.get_figure()
    pos = ax.get_position()
    return pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()


def _cell_points(ax: plt.Axes, data_w: float, data_h: float, cell_size: float) -> float:
    """Visual size of one cell in points once ``data_w × data_h`` fills ``ax``."""
    ax_w_in, ax_h_in = _axes_size_in(ax)
    scale = min(ax_w_in / data_w, ax_h_in / data_h)   # inches per metre
    return cell_size * scale * 72.0


def draw_map(
    ax: plt.Axes,
    landmasses: list[dict],
    color_map: dict[str, str],
    cell_size: float,
    scope: str = "islands",
    show_labels: bool = True,
    label_min_cells: int = 3,
    background_style: str = "grid",
    background_color: Optional[str] = None,
    show_region_borders: bool = True,
    style: Optional[dict] = None,
    label_mode: str = "auto",
    labels: Optional[dict[str, str]] = None,
    set_limits: bool = True,
) -> plt.Axes:
    """
    Draw the block map into a caller-supplied Matplotlib Axes.

    This is what :meth:`AreaGrid.draw` and :meth:`PopGrid.draw` call. It
    renders the main landmass plus any inline islands (``scope='islands'``),
    or the main landmass alone (``scope='mainland'``). Distant territories
    that :func:`create_figure` puts in panel insets are not drawn here, since a
    single Axes has nowhere to put them; ``scope='all'`` is accepted and
    behaves like ``'islands'``.

    Parameters
    ----------
    ax:
        Target Axes. Its aspect is set to ``'equal'`` and its axis turned off.
    background_color:
        Fill for the Axes patch. ``None`` leaves the Axes background untouched,
        so the map sits on whatever the caller's figure already has.
    style:
        Visual overrides; see :data:`DEFAULT_STYLE` for the keys.
    label_mode:
        ``'auto'`` fits the full name, wraps it, or abbreviates it as space
        allows; ``'full'`` never abbreviates; ``'short'`` always abbreviates;
        ``'none'`` draws no labels.
    labels:
        ``{region_name: text}`` overrides. An empty string suppresses that
        region's label.
    set_limits:
        Fit the Axes limits to the drawn cells. Pass ``False`` to manage them
        yourself.

    Returns
    -------
    The same Axes, for chaining.
    """
    if scope not in ("all", "islands", "mainland"):
        raise ValueError(f"scope must be 'all', 'islands', or 'mainland'; got {scope!r}")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be one of {LABEL_MODES}; got {label_mode!r}")
    st = resolve_style(style)

    main = landmasses[0]
    secondary_raw = [lm for lm in landmasses[1:] if not lm["cells"].empty]
    inline_lms, _panel_lms = _classify_secondary(main, secondary_raw, cell_size)
    if scope == "mainland":
        inline_lms = []

    main_cells = main["cells"]
    all_inline_cells = [main_cells] + [lm["cells"] for lm in inline_lms]
    all_cx = np.concatenate([g["cx"].to_numpy() for g in all_inline_cells])
    all_cy = np.concatenate([g["cy"].to_numpy() for g in all_inline_cells])

    margin = cell_size * 1.2
    data_w = (all_cx.max() - all_cx.min()) + 2 * margin
    data_h = (all_cy.max() - all_cy.min()) + 2 * margin
    cell_pts = _cell_points(ax, data_w, data_h, cell_size)
    border_lw = float(np.clip(cell_pts * 0.08, 0.6, 2.0))

    if background_color is not None:
        ax.set_facecolor(background_color)
    bg_for_grid = background_color or mcolors.to_hex(ax.get_facecolor())
    ax.axis("off")

    if background_style != "none":
        _full_grid_background(
            ax, all_cx, all_cy, cell_size, bg_for_grid,
            style=background_style,
            grid_color=st["grid_color"], grid_width=st["grid_width"],
        )

    common = dict(
        show_labels=show_labels, set_limits=False,
        background_style=background_style,
        show_region_borders=show_region_borders, border_lw=border_lw,
        style=st, label_mode=label_mode, labels=labels, cell_pts=cell_pts,
    )
    _plot_cells(ax, main_cells, color_map, cell_size,
                label_min_cells=label_min_cells, **common)

    inline_min = max(1, label_min_cells)
    for lm in inline_lms:
        _plot_cells(ax, lm["cells"], color_map, cell_size,
                    label_min_cells=inline_min, **common)
        # Leader-line labels for inline territories below the in-cell threshold
        if show_labels and label_mode != "none":
            _label_small_inline(
                ax, lm["cells"], color_map, cell_size, inline_min,
                style=st, labels=labels,
            )

    if set_limits:
        _apply_limits(ax, all_inline_cells, cell_size)
    ax.set_aspect("equal")
    return ax


# ── Public figure builder ─────────────────────────────────────────────────────

def create_figure(
    landmasses: list[dict],
    color_map: dict[str, str],
    cell_size: float,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    show_labels: bool = True,
    label_min_cells: int = 3,
    figsize: Optional[tuple[float, float]] = None,
    background_color: str = "#FAF6F0",
    scope: str = "all",
    background_style: str = "grid",
    show_region_borders: bool = True,
    data_source: Optional[str] = None,
    style: Optional[dict] = None,
    label_mode: str = "auto",
    labels: Optional[dict[str, str]] = None,
) -> plt.Figure:
    """
    Build the complete matplotlib Figure.

    Parameters
    ----------
    scope:
        ``'all'`` — render everything (default).
        ``'islands'`` — main + inline nearby islands, no distant panel insets.
        ``'mainland'`` — main landmass only.
    background_style:
        Background style for the map area:
        ``'grid'``   — visible individual squares (graph-paper, default).
        ``'solid'``  — solid colour rectangle, no grid lines.
        ``'none'``   — no background; plain figure background colour shows.
    show_region_borders:
        Outline around each region boundary.  Default True.
    data_source:
        Optional attribution string shown bottom-left.
    style, label_mode, labels:
        See :func:`draw_map`.
    """
    st = resolve_style(style)
    main = landmasses[0]
    secondary_raw = [lm for lm in landmasses[1:] if not lm["cells"].empty]

    inline_lms, panel_lms = _classify_secondary(main, secondary_raw, cell_size)
    panel_lms = _group_panels_by_region(panel_lms)

    # Apply scope filter
    if scope == "mainland":
        inline_lms = []
        panel_lms = []
    elif scope == "islands":
        panel_lms = []

    main_cells = main["cells"]

    # ── Figure width and scale ────────────────────────────────────────────────
    fig_w = 12.0  # inches

    # Main map data extent (including inline secondary landmasses)
    all_inline_cells = [main_cells] + [lm["cells"] for lm in inline_lms]
    all_cx = np.concatenate([g["cx"].to_numpy() for g in all_inline_cells])
    all_cy = np.concatenate([g["cy"].to_numpy() for g in all_inline_cells])
    data_w = (all_cx.max() - all_cx.min()) + 2 * cell_size * 1.5
    data_h = (all_cy.max() - all_cy.min()) + 2 * cell_size * 1.5

    # Visual scale: inches per metre, anchored to figure width
    scale_in_per_m = fig_w / data_w
    main_h_in = data_h * scale_in_per_m

    # Border linewidth: 8% of the visual cell size in points.
    # Using visual cell size (not data-space cell_size) ensures borders look
    # consistent regardless of country size — USA and Spain cells appear
    # roughly the same physical size on the 12-inch figure.
    _cell_visual_pts = cell_size * scale_in_per_m * 72.0
    _border_lw = float(np.clip(_cell_visual_pts * 0.08, 0.6, 2.0))

    # ── Panel row height: same visual scale as main map ───────────────────────
    panel_dims: list[tuple[float, float]] = []
    panel_h_in = 0.0

    if panel_lms:
        for lm in panel_lms:
            panel_dims.append(_panel_data_dims(lm, cell_size))

        # Natural height at main-map scale for each panel
        natural_heights = [h * scale_in_per_m for _, h in panel_dims]

        # Clamp: at least 1.0 inch (single-cell panels visible), at most
        # 80% of main height (prevents extreme figure sizes for tiny countries
        # with one huge overseas territory)
        panel_h_in = float(np.clip(
            max(natural_heights),
            1.0,
            main_h_in * 0.80,
        ))

    # ── Figure height ─────────────────────────────────────────────────────────
    title_h_in = (0.65 if title else 0.0) + (0.45 if subtitle else 0.0)
    sep_h_in = 0.55 if panel_lms else 0.0
    # Footer: always present — white space that hosts source attribution,
    # clear of any panel insets regardless of figure height.
    footer_h_in = 0.35

    total_h = main_h_in + panel_h_in + title_h_in + sep_h_in + footer_h_in

    if figsize is not None:
        fig_w, total_h = figsize

    fig = plt.figure(figsize=(fig_w, total_h), facecolor=background_color)

    # ── GridSpec: title | main | [sep | panels] | footer ─────────────────────
    if panel_lms:
        n_rows = 5
        height_ratios = [title_h_in, main_h_in, sep_h_in, panel_h_in, footer_h_in]
    else:
        n_rows = 3
        height_ratios = [title_h_in, main_h_in, footer_h_in]

    outer = gridspec.GridSpec(
        n_rows, 1,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.0,
        top=1.0, bottom=0.0, left=0.0, right=1.0,
    )

    # ── Title area ────────────────────────────────────────────────────────────
    title_ax = fig.add_subplot(outer[0])
    title_ax.axis("off")
    title_ax.set_facecolor(background_color)

    y = 0.88
    if title:
        title_ax.text(
            0.5, y, title,
            ha="center", va="top",
            fontsize=26, fontweight="bold",
            color="#1A3829",
            transform=title_ax.transAxes,
        )
        y -= 0.52
    if subtitle:
        title_ax.text(
            0.5, y, subtitle,
            ha="center", va="top",
            fontsize=13,
            color="#5E7D6A",
            fontstyle="italic",
            transform=title_ax.transAxes,
        )

    # ── Main axes ─────────────────────────────────────────────────────────────
    main_ax = fig.add_subplot(outer[1])
    draw_map(
        main_ax, landmasses, color_map, cell_size,
        scope="mainland" if scope == "mainland" else "islands",
        show_labels=show_labels, label_min_cells=label_min_cells,
        background_style=background_style, background_color=background_color,
        show_region_borders=show_region_borders,
        style=st, label_mode=label_mode, labels=labels,
    )

    # ── Panel row ─────────────────────────────────────────────────────────────
    if panel_lms:
        n = len(panel_lms)
        # Width ratios proportional to data width, minimum for single cells
        raw_widths = [max(w, cell_size * 1.5) for w, _ in panel_dims]
        total_raw_w = sum(raw_widths)

        inner = gridspec.GridSpecFromSubplotSpec(
            1, n,
            subplot_spec=outer[3],
            wspace=0.08,
            width_ratios=raw_widths,
        )

        # Standard data height shared by ALL panels.  Derived from the panel
        # row height at the main-map visual scale so every cell in every panel
        # is the same physical size as cells in the main map.
        panel_std_data_h = panel_h_in / scale_in_per_m  # metres

        for i, (lm, (data_w_p, data_h_p)) in enumerate(zip(panel_lms, panel_dims)):
            ax = fig.add_subplot(inner[i])
            ax.set_facecolor(background_color)

            n_cells = len(lm["cells"])

            # Plot cells WITHOUT auto limits — set manually below
            if background_style != "none":
                cells = lm["cells"]
                panel_cx = cells["cx"].to_numpy()
                panel_cy = cells["cy"].to_numpy()
                _full_grid_background(
                    ax, panel_cx, panel_cy, cell_size, background_color,
                    style=background_style,
                    grid_color=st["grid_color"], grid_width=st["grid_width"],
                )

            # Labels go ABOVE the box — no need to repeat inside cells
            _plot_cells(
                ax, lm["cells"], color_map, cell_size,
                show_labels=False,
                label_min_cells=1,
                set_limits=False,
                background_style=background_style,
                show_region_borders=show_region_borders,
                border_lw=_border_lw,
                style=st,
            )

            # ax.axis("off") inside _plot_cells sets an internal _axis_off flag
            # that prevents spines from rendering even when set_visible(True).
            # set_axis_on() clears that flag; then we hide ticks manually.
            ax.set_axis_on()
            ax.tick_params(
                left=False, bottom=False,
                labelleft=False, labelbottom=False,
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#1A3829")
                spine.set_linewidth(1.5)

            # ── Consistent scale limits ───────────────────────────────────────
            # All panels use the same metres-per-inch ratio as the main map.
            # Estimated panel physical width determines its data x-extent.
            panel_w_in_est = fig_w * raw_widths[i] / total_raw_w
            panel_std_data_w = panel_w_in_est / scale_in_per_m  # metres

            cells = lm["cells"]
            cx_center = float((cells["cx"].min() + cells["cx"].max()) / 2)
            cy_center = float((cells["cy"].min() + cells["cy"].max()) / 2)

            ax.set_xlim(cx_center - panel_std_data_w / 2,
                        cx_center + panel_std_data_w / 2)
            ax.set_ylim(cy_center - panel_std_data_h / 2,
                        cy_center + panel_std_data_h / 2)
            # Do NOT call set_aspect("equal") — xlim/ylim are already at the
            # correct scale so cells render as squares without extra adjustment.

            # Panel name label is placed in sep_ax (see below)
            # so it doesn't overlap with "Insets not to geographic scale".

    # ── Separator row: dashed line + "Insets" heading + per-panel names ───────
    if panel_lms:
        sep_ax = fig.add_subplot(outer[2])
        sep_ax.axis("off")
        sep_ax.set_facecolor(background_color)

        # Dashed separator line — sits near the top of the sep row,
        # leaving breathing space below it for the two text lines.
        from matplotlib.lines import Line2D
        sep_ax.add_artist(Line2D(
            [0.01, 0.99], [0.96, 0.96],
            transform=sep_ax.transAxes,
            color="#1A3829", linewidth=0.8, linestyle="--",
            zorder=20, clip_on=False,
        ))

        # Line 1: section heading — centered, italic
        sep_ax.text(
            0.5, 0.62,
            "Geographically Detached Territories",
            transform=sep_ax.transAxes,
            ha="center", va="center",
            fontsize=12, color="#5E7D6A",
            fontstyle="italic",
        )

        # Line 2: individual panel names, each centered exactly above its box.
        # SubplotSpec.get_position(fig) gives the real panel x-ranges including
        # wspace — more accurate than proportional raw_widths estimates.
        sep_pos = outer[2].get_position(fig)
        for i, lm in enumerate(panel_lms):
            panel_pos = inner[i].get_position(fig)
            x_fig_center = (panel_pos.x0 + panel_pos.x1) / 2
            x_sep = (x_fig_center - sep_pos.x0) / (sep_pos.x1 - sep_pos.x0)
            sep_ax.text(
                x_sep, 0.12,
                _panel_label(lm),
                transform=sep_ax.transAxes,
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="#1A3829",
            )

    # ── Footer row: source attribution ───────────────────────────────────────
    # Dedicated axes row so source text is always below all content —
    # never overlaps inset panels.
    footer_idx = 4 if panel_lms else 2
    footer_ax = fig.add_subplot(outer[footer_idx])
    footer_ax.axis("off")
    footer_ax.set_facecolor(background_color)

    # An explicit blank/whitespace data_source means "no footer" (used by the
    # compare view, which draws one shared source line). None falls back to the
    # default attribution.
    if data_source is not None and not str(data_source).strip():
        _source_text = None
    else:
        _source_text = data_source or "Natural Earth (ne_10m_admin_1_states_provinces) · popgrid"
    if _source_text:
        footer_ax.text(
            0.014, 0.65,
            f"Source: {_source_text}",
            transform=footer_ax.transAxes,
            ha="left", va="center",
            fontsize=9,
            color="#5E7D6A",
            fontstyle="italic",
        )

    fig.patch.set_facecolor(background_color)
    return fig

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
INLINE_MARGIN_CELLS: int = 5


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
) -> None:
    """
    Draw a background grid or solid fill over the cell bounding box.

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
    grid_fill = _grid_color(background_color)

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
            colors=["#b8b8b8"],
            linewidths=0.35,
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
) -> None:
    """
    Draw colored cells and optional region borders.

    background_style (default 'grid')
        Controls the cell gap size to match the background.

    show_region_borders=True (default)
        A thick black outline is drawn around each region's total combined
        cell area.

    border_lw
        Region border linewidth in points.  When 0 (default), falls back
        to the legacy formula ``cell_size / 18_000``.  create_figure always
        passes a pre-computed value proportional to the visual cell size so
        borders look consistent across countries of all sizes.
    """
    if cells_gdf is None or cells_gdf.empty:
        return

    cx_vals = cells_gdf["cx"].to_numpy()
    cy_vals = cells_gdf["cy"].to_numpy()
    half = cell_size / 2.0

    # ── 1. Cell gap ───────────────────────────────────────────────────────────
    # With background ('grid' or 'solid'): 10% gap exposes the background.
    # Without background ('none'): 4% cosmetic gap only.
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

        # Use the pre-computed visual-scale linewidth if provided;
        # fall back to legacy formula only when called outside create_figure.
        if border_lw > 0:
            lw = border_lw
        else:
            lw = float(np.clip(cell_size / 18_000, 1.2, 3.0))

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
                ax.plot(xs, ys, color="#000000", linewidth=lw,
                        zorder=5, solid_capstyle="round",
                        solid_joinstyle="round")
                for interior in geom.interiors:
                    xs, ys = interior.xy
                    ax.plot(xs, ys, color="#000000", linewidth=lw,
                            zorder=5, solid_capstyle="round",
                            solid_joinstyle="round")

    if set_limits:
        margin = cell_size * 1.2
        ax.set_xlim(cx_vals.min() - margin, cx_vals.max() + margin)
        ax.set_ylim(cy_vals.min() - margin, cy_vals.max() + margin)

    ax.set_aspect("equal")
    ax.axis("off")

    if not show_labels:
        return

    for region_name, grp in cells_gdf.groupby("region_name"):
        n_cells = len(grp)
        if n_cells < label_min_cells:
            continue
        label_cx = grp["cx"].mean()
        label_cy = grp["cy"].mean()
        bg = color_map.get(region_name, "#cccccc")  # type: ignore[arg-type]
        fg = _text_color(bg)
        outline = "#00000066" if fg == "#ffffff" else "#ffffff99"
        fontsize = float(np.clip(5.0 + np.sqrt(n_cells) * 0.9, 5.0, 11.0))
        ax.text(
            label_cx, label_cy, region_name,
            fontsize=fontsize, ha="center", va="center",
            fontweight="bold", color=fg, zorder=4,
            path_effects=[withStroke(linewidth=1.5, foreground=outline)],
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
) -> None:
    """
    Add name annotations for inline regions that are too small to be
    labelled by the standard threshold (e.g. Ceuta, Melilla: 1 cell each).

    These get a small italic label placed just above the cell with a
    pointing offset so it doesn't overlap the coloured square.
    """
    for region_name, grp in cells_gdf.groupby("region_name"):
        if len(grp) >= label_min_cells:
            continue  # already handled by _plot_cells
        cx = float(grp["cx"].mean())
        cy = float(grp["cy"].mean())
        bg = color_map.get(region_name, "#cccccc")
        fg = _text_color(bg)
        ax.annotate(
            region_name,
            xy=(cx, cy),
            xytext=(cx + cell_size * 0.7, cy + cell_size * 1.1),
            fontsize=5.5,
            ha="left",
            va="bottom",
            fontstyle="italic",
            color="#1A3829",
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
        Thick black outline around each region boundary.  Default True.
    data_source:
        Optional attribution string shown bottom-left.
    """
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
    main_ax.set_facecolor(background_color)
    main_ax.axis("off")

    # Draw background ONCE for the full combined extent (main + inline islands)
    if background_style != "none":
        _full_grid_background(
            main_ax, all_cx, all_cy, cell_size, background_color,
            style=background_style,
        )

    _plot_cells(main_ax, main_cells, color_map, cell_size,
                show_labels=show_labels, label_min_cells=label_min_cells,
                set_limits=False, background_style=background_style,
                show_region_borders=show_region_borders,
                border_lw=_border_lw)

    for lm in inline_lms:
        _plot_cells(main_ax, lm["cells"], color_map, cell_size,
                    show_labels=show_labels,
                    label_min_cells=max(1, label_min_cells - 1),
                    set_limits=False, background_style=background_style,
                    show_region_borders=show_region_borders,
                    border_lw=_border_lw)
        # Add small annotated labels for single-cell inline territories
        if show_labels:
            _label_small_inline(
                main_ax, lm["cells"], color_map, cell_size, label_min_cells
            )

    _apply_limits(main_ax, all_inline_cells, cell_size)

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

    _source_text = data_source or "Natural Earth (ne_10m_admin_1_states_provinces) · popgrid"
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

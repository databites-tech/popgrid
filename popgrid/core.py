"""
AreaGrid – land-area tiled map class.

Each cell represents an equal share of the country's land area.

Typical usage::

    from popgrid import AreaGrid

    ag = AreaGrid("ESP", n=1000)
    fig = ag.plot()
    fig.savefig("spain.png", dpi=200, bbox_inches="tight")
"""

from __future__ import annotations

import logging
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely

from .data import (
    load_country_regions,
    load_region_populations,
    admin1_parent_map,
    RECOMMENDED_SETTINGS,
    REGION_LABELS,
)
from .geometry import detect_landmasses, project_regions
from .grid import cell_size_for_n, generate_cells
from .render import build_adjacency_color_map, build_color_map, create_figure, draw_map

logger = logging.getLogger(__name__)

ENGINES = ("auto", "dougenik", "diffusion")


def _country_title(iso3: str, region_label: Optional[str] = None) -> str:
    """
    Return a polished figure title: "{Country} by {RegionLabel} in a square-based grid"

    Falls back to just the country name if no region label is known.
    """
    try:
        import pycountry
        country = pycountry.countries.get(alpha_3=iso3.upper())
        name = country.name if country else iso3
    except ImportError:
        name = iso3

    label = region_label or REGION_LABELS.get(iso3.upper())
    if label:
        return f"A Tiled Map of {name} by {label}"
    return f"A Tiled Map of {name}"


def _prepare_custom_gdf(
    gdf: gpd.GeoDataFrame,
    region_col: str,
    weight_col: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Validate and normalise a user-supplied GeoDataFrame into the standard
    ``region_id | region_name | geometry`` schema used internally.

    Parameters
    ----------
    gdf:
        Any GeoDataFrame with polygon geometries and at least one name column.
        CRS can be anything — it will be re-projected to LAEA in ``build()``.
    region_col:
        Name of the column to use as the region label (e.g. ``'NAME_1'``,
        ``'district'``, ``'state'``).
    weight_col:
        Optional numeric column to carry through as ``region_weight`` (used by
        ``PopGrid`` to size regions by, e.g., population). Ignored by AreaGrid.

    Returns
    -------
    GeoDataFrame with columns ``region_id``, ``region_name``, ``geometry``
    (and ``region_weight`` if *weight_col* given) in the original CRS
    (re-projection happens later in the pipeline).
    """
    import shapely as _shapely

    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(
            f"gdf must be a GeoDataFrame, got {type(gdf).__name__!r}."
        )
    if region_col not in gdf.columns:
        available = [c for c in gdf.columns if c != "geometry"]
        raise ValueError(
            f"Column {region_col!r} not found in GeoDataFrame. "
            f"Available columns: {available}"
        )
    if weight_col is not None and weight_col not in gdf.columns:
        available = [c for c in gdf.columns if c != "geometry"]
        raise ValueError(
            f"weight_col {weight_col!r} not found in GeoDataFrame. "
            f"Available columns: {available}"
        )
    if "geometry" not in gdf.columns or gdf.geometry.is_empty.all():
        raise ValueError("GeoDataFrame has no usable geometry column.")

    # Drop rows with null geometry or null region name
    cols = [region_col, "geometry"] + ([weight_col] if weight_col else [])
    out = gdf[cols].copy()
    out = out[out["geometry"].notna() & out[region_col].notna()].copy()
    out = out[~out["geometry"].is_empty].copy()

    if out.empty:
        raise ValueError(
            "GeoDataFrame has no rows with valid geometry and non-null "
            f"values in column {region_col!r}."
        )

    # Repair geometries
    out["geometry"] = _shapely.make_valid(out["geometry"].values)

    # Rename to standard schema
    rename = {region_col: "region_name"}
    if weight_col:
        rename[weight_col] = "region_weight"
    out = out.rename(columns=rename)
    out["region_name"] = out["region_name"].astype(str).str.strip()
    out["region_id"] = range(len(out))
    keep = ["region_id", "region_name", "geometry"]
    if weight_col:
        out["region_weight"] = pd.to_numeric(out["region_weight"], errors="coerce")
        keep.append("region_weight")
    out = out[keep].reset_index(drop=True)

    # project_regions expects WGS84 geographic coordinates for its median
    # centroid calculation — reproject if necessary.
    result = gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)
    if gdf.crs is not None and not gdf.crs.is_geographic:
        result = result.to_crs("EPSG:4326")
    elif gdf.crs is None:
        result = result.set_crs("EPSG:4326", allow_override=True)
    return result


def _hamilton_quotas(raw: dict[str, float], n: int) -> dict[str, int]:
    """
    Allocate *n* integer seats using the Hamilton / largest-remainder method.

    Guarantees ``sum(result.values()) == n``.
    """
    floor_q = {k: int(v) for k, v in raw.items()}
    remainders = {k: raw[k] - floor_q[k] for k in raw}
    deficit = n - sum(floor_q.values())
    for k in sorted(remainders, key=lambda k: -remainders[k])[:deficit]:
        floor_q[k] += 1
    return floor_q


def _subsample_to_quota(
    cells: gpd.GeoDataFrame,
    quotas: dict[str, int],
) -> gpd.GeoDataFrame:
    """
    For each region keep at most ``quotas[region_name]`` cells.

    Uses uniform random sampling (random_state=42 for reproducibility) to
    preserve the region's geographic shape — cells are sampled uniformly
    across the region's extent, maintaining its boundary silhouette rather
    than collapsing to a centroid blob.
    """
    kept = []
    for region_name, group in cells.groupby("region_name"):
        quota = quotas.get(str(region_name), len(group))
        if len(group) <= quota:
            kept.append(group)
        else:
            kept.append(group.sample(n=quota, random_state=42))
    if not kept:
        return cells.iloc[:0].copy()
    return pd.concat(kept).reset_index(drop=True)


def _weighted_voronoi_cells(
    landmass_geom: "shapely.Geometry",
    regions_proj: gpd.GeoDataFrame,
    cell_size: float,
    landmass_id: int,
    quotas: dict[str, int],
) -> gpd.GeoDataFrame:
    """
    Partition a landmass into grid cells using an iteratively balanced
    weighted Voronoi.

    **Algorithm**

    1. Generate a regular grid inside the landmass boundary.
    2. Compute ``score[cell, region] = dist²(cell, centroid) / w[region]``.
    3. Assign each cell to the region with the lowest score.
    4. Check actual counts vs target quotas.
    5. Multiplicatively update weights: ``w ← w × (target / actual)``
       — over-quota regions shrink, under-quota regions grow.
    6. Repeat steps 2-5 until counts match quotas (≤ 1 cell off per region)
       or 30 iterations elapse.

    The result is connected, population-proportional, and preserves the
    country's overall geographic shape (cells are still inside the boundary,
    and region centroids keep their relative positions).

    Parameters
    ----------
    landmass_geom:
        Projected LAEA geometry for the entire landmass.
    regions_proj:
        Projected GeoDataFrame for regions in this landmass.
    cell_size:
        Grid cell side length in metres — same as AreaGrid cell_size.
    landmass_id:
        Integer tag written into every output row.
    quotas:
        Global ``{region_name: n_cells}`` mapping from Hamilton allocation.
    """
    if regions_proj.empty:
        return gpd.GeoDataFrame()

    minx, miny, maxx, maxy = landmass_geom.bounds
    half = cell_size / 2.0
    xs = np.arange(minx + half, maxx, cell_size)
    ys = np.arange(miny + half, maxy, cell_size)

    if xs.size == 0 or ys.size == 0:
        return generate_cells(landmass_geom, regions_proj, cell_size, landmass_id)

    xx, yy = np.meshgrid(xs, ys)
    cx_all = xx.ravel()
    cy_all = yy.ravel()

    pts = shapely.points(cx_all, cy_all)
    inside = shapely.contains_properly(landmass_geom, pts)

    if not inside.any():
        return generate_cells(landmass_geom, regions_proj, cell_size, landmass_id)

    cx_in = cx_all[inside]
    cy_in = cy_all[inside]
    n_cells = int(inside.sum())

    # ── Region metadata ───────────────────────────────────────────────────────
    centroids_x = np.array([row.geometry.centroid.x for _, row in regions_proj.iterrows()])
    centroids_y = np.array([row.geometry.centroid.y for _, row in regions_proj.iterrows()])
    q_raw = np.array(
        [max(float(quotas.get(str(row["region_name"]), 1)), 1.0)
         for _, row in regions_proj.iterrows()]
    )
    names = regions_proj["region_name"].tolist()
    ids   = regions_proj["region_id"].tolist()

    # Scale quotas to this landmass's actual cell count so they sum to n_cells
    q_target = q_raw * (n_cells / q_raw.sum())

    # ── Iterative balanced Voronoi ────────────────────────────────────────────
    # Precompute distance matrix (n_cells × n_regions) — done once, reused.
    dx = cx_in[:, np.newaxis] - centroids_x[np.newaxis, :]
    dy = cy_in[:, np.newaxis] - centroids_y[np.newaxis, :]
    dist2 = dx ** 2 + dy ** 2

    w = q_target.copy()
    winner = np.zeros(n_cells, dtype=int)

    for _iter in range(50):
        score  = dist2 / np.maximum(w[np.newaxis, :], 1e-9)
        winner = np.argmin(score, axis=1)
        counts = np.bincount(winner, minlength=len(w)).astype(float)

        if np.max(np.abs(counts - np.round(q_target))) <= 1:
            break

        # Damped multiplicative update (exponent 0.5 prevents oscillation):
        # over-quota → w decreases → region wins fewer cells next round
        ratio = q_target / np.maximum(counts, 0.5)
        w = w * np.power(ratio, 0.5)
        w = np.clip(w, q_target * 0.001, q_target * 10_000)

    # ── Build result GeoDataFrame ─────────────────────────────────────────────
    boxes = shapely.box(cx_in - half, cy_in - half, cx_in + half, cy_in + half)

    cells = gpd.GeoDataFrame(
        {
            "cx":          cx_in,
            "cy":          cy_in,
            "region_id":   [ids[i]   for i in winner],
            "region_name": [names[i] for i in winner],
            "landmass_id": landmass_id,
            "geometry":    boxes,
        },
        crs=regions_proj.crs,
    )
    logger.info(
        "Landmass %d: %d cells via balanced Voronoi (%d regions, %d iters).",
        landmass_id, len(cells), len(regions_proj), _iter + 1,
    )
    return cells.reset_index(drop=True)


class AreaGrid:
    """
    Population grid map for any country.

    Each of the *n* cells represents an equal share of the country's total
    land area.  Regions are coloured distinctly and labelled.

    Parameters
    ----------
    country_iso3:
        ISO 3166-1 alpha-3 code, e.g. ``'ESP'``, ``'DEU'``, ``'GBR'``.
    n:
        Target number of grid cells.  Default 1000 (each cell = 0.1 % of
        land area).
    cluster_distance_km:
        Polygon centroids within this distance are merged into the same
        landmass panel.  Default 200 km.  Increase for countries with
        remote overseas territories.
    palette:
        Colour palette.  Accepts a named string — ``'qual'`` (default),
        ``'databites'``, ``'bold'`` — or a custom list of hex colours.
    """

    def __init__(
        self,
        country_iso3: str,
        n: int = 1000,
        cluster_distance_km: float = 200.0,
        palette: str | list[str] = "qual",
        dissolve_by: Optional[str] = "auto",
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        data_source: Optional[str] = None,
        color_map: Optional[dict[str, str]] = None,
        engine: str = "auto",
        exclude: Optional[list[str]] = None,
    ) -> None:
        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}; got {engine!r}")
        self.country_iso3 = country_iso3.strip().upper()
        self.n = n
        self.cluster_distance_km = cluster_distance_km
        self.palette = palette
        self.dissolve_by = dissolve_by
        self.title = title
        self.subtitle = subtitle
        self.data_source = data_source
        # Explicit {region_name: hex} colours; applied on top of the palette
        # assignment so regions you don't name keep their automatic colour.
        self.color_overrides: dict[str, str] = dict(color_map or {})
        # Cartogram engine used by PopGrid: 'dougenik' (rubber-sheet),
        # 'diffusion' (Gastner-Newman) or 'auto' (rubber-sheet, falling back to
        # diffusion when it fails to converge, e.g. China, Russia, Canada).
        self.engine = engine
        # Regions removed before the grid is built, so they take no blocks and
        # the remaining regions' shares are computed without them. Merged with
        # the country's RECOMMENDED_SETTINGS["exclude"] (uninhabited specks
        # such as Macquarie Island).
        self.exclude: set[str] = set(exclude or ())

        # Custom GeoDataFrame path (from_geodataframe)
        self._custom_gdf: Optional[gpd.GeoDataFrame] = None
        self._custom_region_col: Optional[str] = None
        self._custom_weight_col: Optional[str] = None

        # Internal state – populated by build()
        self._regions_wgs84: Optional[gpd.GeoDataFrame] = None
        self._regions_proj: Optional[gpd.GeoDataFrame] = None
        self._landmasses: Optional[list[dict]] = None
        self._cell_size: Optional[float] = None
        self._color_map: Optional[dict[str, str]] = None
        self._built: bool = False

    # ── Build pipeline ────────────────────────────────────────────────────────

    def _assign_stable_colors(self) -> None:
        """
        Assign a region → colour map from *mode-invariant* geographic inputs:
        the real border-adjacency graph and each region's land area. Because
        neither depends on whether cells are weighted by area or population,
        an ``AreaGrid`` and a ``PopGrid`` of the same country get **identical
        colours**, so the two maps can be compared region by region.
        """
        from .cartogram import real_adjacency  # local import avoids cycles

        names = self._regions_proj["region_name"].dropna().tolist()
        try:
            adjacency = real_adjacency(self._regions_proj)
        except Exception:  # noqa: BLE001
            adjacency = {n: set() for n in names}
        # Land area is a fixed property of each region, identical across grid
        # modes — used as the "dominant region" key so the same large regions
        # claim well-separated colours in both maps.
        sizes = {
            str(row["region_name"]): float(row.geometry.area)
            for _, row in self._regions_proj.iterrows()
            if row["region_name"] is not None and row.geometry is not None
        }
        self._color_map = build_adjacency_color_map(
            names, adjacency, self.palette, sizes=sizes
        )
        if self.color_overrides:
            unknown = sorted(set(self.color_overrides) - set(self._color_map))
            if unknown:
                logger.warning(
                    "color_map names not found among regions (ignored): %s", unknown
                )
            self._color_map.update(
                {k: v for k, v in self.color_overrides.items() if k in self._color_map}
            )

    def build(self) -> "AreaGrid":
        """
        Run the full data → geometry → grid pipeline.

        Called automatically on the first call to :meth:`plot` or any
        property accessor.  Safe to call explicitly for eager loading.
        """
        # 1. Load regions — from user GeoDataFrame or Natural Earth
        if self._custom_gdf is not None:
            logger.info("[1/5] Preparing custom GeoDataFrame …")
            self._regions_wgs84 = _prepare_custom_gdf(
                self._custom_gdf, self._custom_region_col,
                weight_col=self._custom_weight_col,
            )
        else:
            logger.info("[1/5] Loading regions for %s …", self.country_iso3)
            self._regions_wgs84 = load_country_regions(
                self.country_iso3,
                dissolve_by=self.dissolve_by,
            )

        self._regions_wgs84 = self._apply_exclude(self._regions_wgs84)

        # 2. Project to country-centred LAEA (equal-area, metres)
        logger.info("[2/5] Projecting to equal-area CRS …")
        self._regions_proj, self._crs = project_regions(self._regions_wgs84)

        # 3. Compute cell size
        total_area_m2: float = self._regions_proj.geometry.union_all().area
        self._cell_size = cell_size_for_n(total_area_m2, self.n)
        logger.info(
            "[3/5] Total land area: %.0f km²  →  cell_size = %.1f m  "
            "(target n=%d cells)",
            total_area_m2 / 1e6,
            self._cell_size,
            self.n,
        )

        # 4. Detect landmass clusters
        logger.info("[4/5] Detecting landmasses …")
        raw_lms = detect_landmasses(
            self._regions_proj,
            cluster_distance_m=self.cluster_distance_km * 1_000,
        )

        # 5. Generate grid cells for each landmass
        logger.info("[5/5] Generating grid cells …")
        all_region_names: list[str] = []
        landmasses: list[dict] = []

        for lm in raw_lms:
            regions_sub = self._regions_proj.iloc[lm.region_idx].copy()
            cells = generate_cells(
                lm.geom,
                regions_sub,
                self._cell_size,
                lm.id,
            )
            lm_dict = {
                "id": lm.id,
                "geom": lm.geom,
                "region_idx": lm.region_idx,
                "cells": cells,
            }
            landmasses.append(lm_dict)

            if not cells.empty:
                all_region_names.extend(cells["region_name"].dropna().tolist())

        self._landmasses = landmasses

        # 6. Assign colours — use ALL loaded region names so every region
        #    gets a stable colour even if it received 0 cells (e.g. Hamburg
        #    at low N, or tiny territories at country-wide cell size).
        all_region_names = self._regions_proj["region_name"].dropna().tolist()
        self._assign_stable_colors()

        self._built = True
        logger.info(
            "Build complete — %d landmass(es), %d total cells.",
            len(self._landmasses),
            self.total_cells,
        )
        return self

    def _ensure_built(self) -> None:
        if not self._built:
            self.build()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def total_cells(self) -> int:
        """Actual number of grid cells generated (may differ slightly from *n*)."""
        self._ensure_built()
        return sum(len(lm["cells"]) for lm in self._landmasses)  # type: ignore[union-attr]

    @property
    def cell_size_km(self) -> float:
        """Cell side length in kilometres."""
        self._ensure_built()
        return self._cell_size / 1_000  # type: ignore[operator]

    @property
    def n_landmasses(self) -> int:
        """Number of landmass clusters detected."""
        self._ensure_built()
        return len(self._landmasses)  # type: ignore[arg-type]

    @property
    def regions(self) -> gpd.GeoDataFrame:
        """WGS84 GeoDataFrame of the loaded admin-1 regions."""
        self._ensure_built()
        return self._regions_wgs84  # type: ignore[return-value]

    @property
    def color_map(self) -> dict[str, str]:
        """``{region_name: hex_color}`` mapping used in the plot."""
        self._ensure_built()
        return self._color_map  # type: ignore[return-value]

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot(
        self,
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
        Render the grid map as a complete, self-contained Figure.

        For drawing into your own Axes (subplots, custom layouts) use
        :meth:`draw` instead.

        Parameters
        ----------
        title:
            Figure title.  Defaults to ``"<ISO3> · <n> cells"`` if *None*.
        subtitle:
            Optional subtitle, e.g.
            ``"Every block = 0.1% of the national land area"``.
        show_labels:
            Annotate each region with its name.  Default ``True``.
        label_min_cells:
            Minimum number of cells a region must have to receive a label.
        figsize:
            ``(width_in, height_in)``.  Auto-computed if ``None``.
        background_color:
            Hex background colour.  Defaults to DataBites cream ``#FAF6F0``.
        scope:
            Controls which landmasses are rendered:

            - ``'all'`` (default) — main landmass + all secondary landmasses.
            - ``'islands'`` — main + geographically nearby inline islands
              (e.g. Balearics for Spain), **no** distant panel insets.
            - ``'mainland'`` — main landmass only; all islands and overseas
              territories are hidden.
        background_style:
            Background style for the map area:
            ``'grid'`` (default) — visible graph-paper squares.
            ``'solid'``          — solid colour background rectangle.
            ``'none'``           — plain background colour, no grid.
        show_region_borders:
            Thin dark outline around each region boundary.  Default ``True``.
        data_source:
            Attribution string shown bottom-left of the figure, e.g.
            ``'Natural Earth · popgrid'``.
        style:
            Visual overrides as a dict. Keys: ``edge_color``, ``edge_width``,
            ``grid_color``, ``grid_width``, ``cell_gap``, ``label_color``,
            ``label_size``, ``label_min_size``, ``label_max_size``,
            ``label_weight``, ``label_font``. Any key left out keeps its
            automatic value. See :data:`popgrid.render.DEFAULT_STYLE`.
        label_mode:
            ``'auto'`` (default) sizes each label to its region: the full name
            if it fits, otherwise wrapped on two lines, otherwise an
            abbreviation (``Castilla y León`` → ``CL``). ``'full'`` never
            abbreviates, ``'short'`` always does, ``'none'`` draws no labels.
        labels:
            ``{region_name: text}`` to override individual labels. An empty
            string hides that region's label.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._ensure_built()
        if scope not in ("all", "islands", "mainland"):
            raise ValueError(
                f"scope must be 'all', 'islands', or 'mainland'; got {scope!r}"
            )

        # Resolve title: plot() arg > instance attr > auto-generated
        if title is None:
            title = self.title
        if title is None:
            if self._custom_gdf is not None:
                title = "A Tiled Map"
            else:
                title = _country_title(self.country_iso3)

        # Resolve subtitle: plot() arg > instance attr > auto-generated
        if subtitle is None:
            subtitle = self.subtitle
        if subtitle is None:
            pct = round(100 / self.n, 2) if self.n > 0 else 0
            subtitle = (
                f"Every block = {pct}% of the national land area"
                f"  ({self.total_cells:,} blocks total)"
            )

        return create_figure(
            landmasses=self._landmasses,  # type: ignore[arg-type]
            color_map=self._color_map,  # type: ignore[arg-type]
            cell_size=self._cell_size,  # type: ignore[arg-type]
            title=title,
            subtitle=subtitle,
            show_labels=show_labels,
            label_min_cells=label_min_cells,
            figsize=figsize,
            background_color=background_color,
            scope=scope,
            background_style=background_style,
            show_region_borders=show_region_borders,
            data_source=data_source if data_source is not None else self.data_source,
            style=style,
            label_mode=label_mode,
            labels=labels,
            panel_regions=self._panel_regions(),
            display_names=self._display_names(),
        )

    def _excluded(self) -> set[str]:
        """Regions to drop before building: constructor ``exclude`` plus the country default."""
        out = set(self.exclude)
        if self._custom_gdf is None:
            out |= set(RECOMMENDED_SETTINGS.get(self.country_iso3, {}).get("exclude", ()))
        return out

    def _apply_exclude(self, regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        excluded = self._excluded()
        if not excluded:
            return regions
        names = regions["region_name"].astype(str)
        hit = names.isin(excluded)
        missing = sorted(excluded - set(names))
        if missing:
            logger.warning("exclude: region(s) not found and ignored: %s", missing)
        if hit.any():
            logger.info("Excluding %d region(s): %s", int(hit.sum()), sorted(names[hit]))
            regions = regions[~hit].reset_index(drop=True)
        return regions

    def _display_names(self) -> dict[str, str]:
        """Label renames for this country (per RECOMMENDED_SETTINGS); not for custom data."""
        if self._custom_gdf is not None:
            return {}
        return dict(RECOMMENDED_SETTINGS.get(self.country_iso3, {}).get("display_names", {}))

    def _panel_regions(self) -> set[str]:
        """Regions this country always shows in a panel inset (per RECOMMENDED_SETTINGS)."""
        if self._custom_gdf is not None:
            return set()
        return set(RECOMMENDED_SETTINGS.get(self.country_iso3, {}).get("panel_regions", ()))

    def draw(
        self,
        ax: plt.Axes,
        *,
        scope: str = "islands",
        show_labels: bool = True,
        label_min_cells: int = 3,
        background_style: str = "grid",
        background_color: Optional[str] = None,
        show_region_borders: bool = True,
        style: Optional[dict] = None,
        label_mode: str = "auto",
        labels: Optional[dict[str, str]] = None,
    ) -> plt.Axes:
        """
        Draw the grid map into an existing Matplotlib Axes.

        Use this to put a popgrid map inside your own figure: a row of
        subplots comparing several countries, a map next to a chart, a
        custom title block. Nothing but the map is drawn: no title, no
        subtitle, no source line, no panel insets. Those stay with
        :meth:`plot`, which builds a complete figure.

        ::

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            AreaGrid("ESP").draw(ax1)
            PopGrid("ESP").draw(ax2)

        Parameters
        ----------
        ax:
            Target Axes. The map is drawn with an equal aspect ratio and the
            axis is turned off.
        scope:
            ``'islands'`` (default) draws the main landmass plus nearby inline
            islands; ``'mainland'`` the main landmass only. Distant territories
            (Canary Islands, Hawaii, overseas France) are never drawn into a
            single Axes; call :meth:`draw` on a separate Axes with a filtered
            grid if you need them, or use :meth:`plot`. ``'all'`` is accepted
            and behaves like ``'islands'``.
        background_color:
            Fill colour for the Axes. ``None`` (default) leaves the Axes as it
            is, so the map takes on your figure's background.
        show_labels, label_min_cells, background_style, show_region_borders,
        style, label_mode, labels:
            Same as :meth:`plot`.

        Returns
        -------
        matplotlib.axes.Axes
            The same ``ax``, so calls can be chained.
        """
        self._ensure_built()
        return draw_map(
            ax,
            landmasses=self._landmasses,  # type: ignore[arg-type]
            color_map=self._color_map,  # type: ignore[arg-type]
            cell_size=self._cell_size,  # type: ignore[arg-type]
            scope=scope,
            show_labels=show_labels,
            label_min_cells=label_min_cells,
            background_style=background_style,
            background_color=background_color,
            show_region_borders=show_region_borders,
            style=style,
            label_mode=label_mode,
            labels=labels,
            panel_regions=self._panel_regions(),
            display_names=self._display_names(),
        )

    def save(
        self,
        path: str,
        dpi: int = 200,
        **plot_kwargs,
    ) -> None:
        """
        Build (if not already) and save the figure to *path*.

        Parameters
        ----------
        path:
            Output file path.  Extension determines format
            (``'.png'``, ``'.svg'``, ``'.pdf'`` …).
        dpi:
            Resolution for raster formats.
        **plot_kwargs:
            Forwarded to :meth:`plot`.
        """
        fig = self.plot(**plot_kwargs)
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        logger.info("Saved → %s", path)

    # ── Alternative constructor ──────────────────────────────────────────────

    @classmethod
    def from_geodataframe(
        cls,
        gdf: gpd.GeoDataFrame,
        region_col: str,
        weight_col: Optional[str] = None,
        n: int = 1000,
        cluster_distance_km: float = 200.0,
        palette: str | list[str] = "qual",
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        data_source: Optional[str] = None,
        color_map: Optional[dict[str, str]] = None,
        engine: str = "auto",
        exclude: Optional[list[str]] = None,
    ) -> "AreaGrid":
        """
        Build an AreaGrid or PopGrid from any GeoDataFrame — no Natural Earth
        download.

        This is the entry point for custom shapefiles, GeoJSON files,
        PostGIS queries, or any other polygon data source. It works on both
        classes:

        * ``AreaGrid.from_geodataframe(gdf, region_col=...)`` sizes each region
          by its land area.
        * ``PopGrid.from_geodataframe(gdf, region_col=..., weight_col=...)``
          sizes each region by the numeric ``weight_col`` (e.g. population).

        Parameters
        ----------
        gdf:
            GeoDataFrame with polygon geometries and at least one name column.
            Any CRS is accepted — it will be re-projected to LAEA internally.
        region_col:
            Column to use as the region label, e.g. ``'NAME_1'``,
            ``'district'``, ``'state_name'``.
        weight_col:
            Numeric column to weight regions by (PopGrid only — required when
            calling ``PopGrid.from_geodataframe``). Ignored by AreaGrid, which
            always weights by land area.
        n:
            Target number of cells.  Default 1000.
        cluster_distance_km:
            Max distance (km) to group landmasses together.  Default 200.
        palette:
            Named palette or list of hex colours.
        title:
            Figure title.  If ``None``, defaults to ``"A Tiled Map"``.
            Pass a custom string to override, e.g. ``"My Region Map"``.
        color_map:
            ``{region_name: hex}`` colours to pin. Unlisted regions keep
            their automatic palette colour.
        exclude:
            Region names to remove before building. They take no blocks and
            do not count toward the total.

        Returns
        -------
        AreaGrid or PopGrid (not yet built — call ``.build()`` or ``.plot()``)

        Examples
        --------
        ::

            import geopandas as gpd
            from popgrid import AreaGrid, PopGrid

            gdf = gpd.read_file("barcelona_districts.geojson")

            # Land-area cartogram
            ag = AreaGrid.from_geodataframe(gdf, region_col="district")

            # Population cartogram from your own column
            pg = PopGrid.from_geodataframe(
                gdf, region_col="district", weight_col="population"
            )
            fig = pg.plot(title="Population of Barcelona")
        """
        # __init__ does no I/O, so the normal constructor is safe here;
        # "CUSTOM" is never looked up because _custom_gdf short-circuits build().
        instance = cls(
            "CUSTOM", n=n, cluster_distance_km=cluster_distance_km,
            palette=palette, dissolve_by=None, title=title, subtitle=subtitle,
            data_source=data_source, color_map=color_map, engine=engine,
            exclude=exclude,
        )
        instance._custom_gdf = gdf
        instance._custom_region_col = region_col
        instance._custom_weight_col = weight_col
        return instance

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = (
            f"built  |  {self.total_cells} cells  |  "
            f"{self.n_landmasses} landmass(es)"
            if self._built
            else "not built"
        )
        if self._custom_gdf is not None:
            source = f"custom GeoDataFrame ({len(self._custom_gdf)} rows)"
        else:
            source = f"country='{self.country_iso3}', dissolve_by={self.dissolve_by!r}"
        return f"AreaGrid({source}, n={self.n})  [{status}]"


def _fill_outline_holes(geom):
    """
    Return ``geom`` with all interior rings removed (only exterior boundaries
    kept).  The deformed region polygons leave thin gaps between one another;
    their union therefore has holes, and cells falling in those holes are
    dropped during rasterisation.  Filling the holes lets every interior cell
    be rasterised and assigned to its nearest region — no seams.
    """
    from shapely.geometry import Polygon, MultiPolygon

    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior.coords)
    if geom.geom_type == "MultiPolygon":
        return shapely.union_all(
            [Polygon(p.exterior.coords) for p in geom.geoms]
        )
    return geom


def _reanchor_secondary(
    all_cells: gpd.GeoDataFrame,
    mainland_id: int,
    geo_bbox: tuple,
    def_bbox: tuple,
    cell_size: float,
) -> gpd.GeoDataFrame:
    """
    Reposition detached-territory cells so they keep the same position relative
    to the *deformed* mainland that they had relative to the *geographic*
    mainland.  Without this, deforming the mainland shifts its extent and an
    enclave such as Melilla drifts outside the inline margin and is wrongly
    promoted to a detached panel.  The island-classification logic is left
    untouched; this only restores the enclave's frame.
    """
    gx0, gy0, gx1, gy1 = geo_bbox
    dx0, dy0, dx1, dy1 = def_bbox
    gw = (gx1 - gx0) or 1.0
    gh = (gy1 - gy0) or 1.0

    def remap(cx, cy):
        nx = dx0 + (cx - gx0) / gw * (dx1 - dx0)
        ny = dy0 + (cy - gy0) / gh * (dy1 - dy0)
        return nx, ny

    if (all_cells["landmass_id"] != mainland_id).sum() == 0:
        return all_cells
    half = cell_size / 2.0
    out = all_cells.copy()
    # Translate each detached territory as a rigid block by the offset its
    # centroid needs in the deformed frame — this keeps the territory's bearing
    # relative to the cartogram WITHOUT rescaling its internal cell spacing
    # (per-cell scaling pushed enclaves like Ceuta off the grid lattice).
    for lm_id, grp in all_cells.groupby("landmass_id"):
        if lm_id == mainland_id:
            continue
        cx0 = float(grp["cx"].mean())
        cy0 = float(grp["cy"].mean())
        rx, ry = remap(cx0, cy0)
        off_x, off_y = rx - cx0, ry - cy0
        m = out["landmass_id"] == lm_id
        nx = out.loc[m, "cx"].to_numpy() + off_x
        ny = out.loc[m, "cy"].to_numpy() + off_y
        out.loc[m, "cx"] = nx
        out.loc[m, "cy"] = ny
        out.loc[m, "geometry"] = shapely.box(nx - half, ny - half, nx + half, ny + half)
    return out


def _is_folded(geom) -> bool:
    """
    True when a deformed region polygon has collapsed on itself: ``make_valid``
    turned it into a GeometryCollection with line or point fragments, which is
    what a torn rubber sheet looks like (China's western provinces).
    """
    if geom is None or geom.is_empty:
        return False
    if geom.geom_type == "GeometryCollection":
        return True
    return False


class PopGrid(AreaGrid):
    """
    Population tiled map — each cell represents an equal share of the
    country's population rather than its land area.

    Uses the same cell size as :class:`AreaGrid` but allocates cells to
    regions proportionally to population via the Hamilton / largest-remainder
    quota method.  Population data is read from the bundled
    ``resources/population.csv`` covering 24 countries.

    Parameters
    ----------
    country_iso3:
        ISO 3166-1 alpha-3 code, e.g. ``'ESP'``, ``'DEU'``, ``'JPN'``.
    n:
        Target number of cells.  Default 1000.
    dissolve_by, palette, cluster_distance_km, title, subtitle, data_source:
        Same as :class:`AreaGrid`.
    """

    def _deform(self, shaped: gpd.GeoDataFrame, sub_pops: dict[str, float]) -> gpd.GeoDataFrame:
        """
        Run the configured cartogram engine on one landmass.

        ``auto`` tries the rubber-sheet engine first and switches to diffusion
        when it did not converge (its divergence guard tripped) or left a
        region more than 2× off its target area: that is the China case, where
        the sheet tears. A per-country ``engine`` in RECOMMENDED_SETTINGS
        overrides ``auto``.
        """
        from .cartogram import diffusion_cartogram, dougenik_cartogram  # avoids cycles

        engine = self.engine
        if engine == "auto":
            engine = RECOMMENDED_SETTINGS.get(self.country_iso3, {}).get("engine", "auto")
        if engine == "diffusion":
            return diffusion_cartogram(shaped, sub_pops)
        diag: dict = {}
        out = dougenik_cartogram(
            shaped, sub_pops, iterations=60, max_step_fraction=0.5, diagnostics=diag
        )
        if engine == "auto":
            folded = int(sum(_is_folded(g) for g in out.geometry))
            if folded or not diag.get("converged", True) or diag.get("max_error", 1) > 2.0:
                logger.info(
                    "%s: rubber-sheet cartogram failed (%d folded region(s), converged=%s, "
                    "max area error %.2f); using diffusion engine",
                    self.country_iso3, folded, diag.get("converged"), diag.get("max_error", 0),
                )
                return diffusion_cartogram(shaped, sub_pops)
        return out

    def build(self) -> "PopGrid":
        """
        Population cartogram pipeline.

        Instead of clamping cells to the geographic outline (which locks the
        country shape to geography), every landmass is deformed so that each
        region's **area becomes proportional to its population**, then the
        deformed shapes are rasterised into the grid.  The country outline is
        not imposed — it emerges from the population distribution.  Dense
        regions bulge, sparse regions pinch, relative position and silhouette
        are preserved.

        Steps
        -----
        1. Load regions, project to equal-area CRS, compute the global cell
           size (each cell = 100/n % of national population).
        2. Hamilton-allocate an integer cell quota per region.
        3. For each landmass: size it to its population quota, then — if it
           holds more than one region — run a Dougenik contiguous-area
           cartogram so internal areas track population.  Single-region
           landmasses (islands, enclaves) keep their literal silhouette,
           uniformly scaled to their population quota.
        4. Rasterise the deformed shapes with the standard grid generator.
        5. Fit every region to its exact quota (trim ragged edges / grow
           into concavities) so the totals are precise.

        Rendering then reuses the AreaGrid figure builder, so the population
        cartogram inherits the full DataBites styling (grid, palette, region
        borders, title, detached-territory inset panels).
        """
        from .cartogram import (
            dougenik_cartogram, scale_to_area, fit_quota,
            real_adjacency, enforce_topology, fill_gaps,
        )

        # ── 1. Load regions ────────────────────────────────────────────────
        if self._custom_gdf is not None:
            logger.info("[1/6] Preparing custom GeoDataFrame …")
            self._regions_wgs84 = _prepare_custom_gdf(
                self._custom_gdf, self._custom_region_col,
                weight_col=self._custom_weight_col,
            )
        else:
            logger.info("[1/6] Loading regions for %s …", self.country_iso3)
            self._regions_wgs84 = load_country_regions(
                self.country_iso3, dissolve_by=self.dissolve_by
            )

        self._regions_wgs84 = self._apply_exclude(self._regions_wgs84)

        # ── 2. Project + global cell size ──────────────────────────────────
        logger.info("[2/6] Projecting to equal-area CRS …")
        self._regions_proj, self._crs = project_regions(self._regions_wgs84)
        total_area_m2: float = self._regions_proj.geometry.union_all().area
        self._cell_size = cell_size_for_n(total_area_m2, self.n)
        cell_area = self._cell_size ** 2

        # Colour NOW, from the raw projected geometry, before anything
        # population-specific touches it. Colours derive from the real border
        # graph plus land area, and AreaGrid computes them from exactly this
        # geometry; the coastline simplification and the zero-block drop below
        # both change the inputs (simplified borders stop touching, dropped
        # regions shift the greedy order), so colouring after them would give
        # the same state a different colour in the two maps.
        self._assign_stable_colors()

        # Simplify only *pathologically* high-vertex outlines (e.g. UK
        # 'Highlands and Islands' at 4,200+ vertices), whose coastline detail
        # is wasted on a coarse grid but makes the cartogram's per-iteration
        # make_valid hang. Per-polygon simplification does not preserve borders
        # shared between regions, so the threshold is set high enough that
        # normal adjacent regions (e.g. Spain's Castillas around Madrid) are
        # never touched — simplifying them would break Madrid's shared border
        # and wreck its quota.
        try:
            simp_tol = self._cell_size * 0.5
            geoms = list(self._regions_proj.geometry)
            new_geoms = []
            for g in geoms:
                if g is not None and shapely.get_num_coordinates(g) > 2500:
                    s = shapely.make_valid(g.simplify(simp_tol, preserve_topology=True))
                    new_geoms.append(s if not s.is_empty else g)
                else:
                    new_geoms.append(g)
            self._regions_proj["geometry"] = gpd.GeoSeries(
                new_geoms, index=self._regions_proj.index, crs=self._crs
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Geometry simplification skipped: %s", exc)

        # ── 3. Population weights → integer quotas ─────────────────────────
        region_names: list[str] = self._regions_proj["region_name"].dropna().tolist()
        if self._custom_gdf is not None:
            # Custom GeoDataFrame: weights come from the user's weight_col,
            # carried through as the 'region_weight' column.
            if "region_weight" not in self._regions_proj.columns:
                raise ValueError(
                    "PopGrid.from_geodataframe requires weight_col=... (a numeric "
                    "column such as population). For an area-weighted map use "
                    "AreaGrid.from_geodataframe instead."
                )
            logger.info("[3/6] Reading custom weights …")
            pop_weights = {
                str(row["region_name"]): float(row["region_weight"])
                for _, row in self._regions_proj.iterrows()
                if row["region_name"] is not None and pd.notna(row["region_weight"])
            }
            if not pop_weights or sum(pop_weights.values()) <= 0:
                raise ValueError(
                    f"weight_col contained no usable positive values "
                    f"(got {pop_weights!r})."
                )
        else:
            logger.info("[3/6] Loading population weights …")
            # When geometries were dissolved to a coarser level, aggregate the
            # fine-grained populations up to those parent regions.
            resolved = self.dissolve_by
            if resolved == "auto":
                resolved = RECOMMENDED_SETTINGS.get(
                    self.country_iso3, {}
                ).get("dissolve_by", None)
            name_to_parent = (
                admin1_parent_map(self.country_iso3, resolved) if resolved else None
            )
            pop_weights = load_region_populations(
                self.country_iso3, region_names, name_to_parent=name_to_parent
            )
        total_pop = sum(pop_weights.values()) or len(region_names)
        raw_quotas = {
            name: self.n * pop_weights.get(name, 0) / total_pop
            for name in region_names
        }
        quotas = _hamilton_quotas(raw_quotas, self.n)

        # Drop regions that round to zero blocks (e.g. tiny disputed reefs like
        # Paracel Islands at 1,000 people).  They can never be drawn, and their
        # extreme population ratio destabilises the cartogram force field.
        keep = {name for name, q in quotas.items() if q >= 1}
        if len(keep) < len(region_names):
            dropped = [n for n in region_names if n not in keep]
            logger.info("Dropping %d zero-block region(s): %s",
                        len(dropped), ", ".join(dropped))
            self._regions_proj = self._regions_proj[
                self._regions_proj["region_name"].isin(keep)
            ].reset_index(drop=True)
            region_names = [n for n in region_names if n in keep]
            quotas = {n: q for n, q in quotas.items() if n in keep}

        # ── 4. Detect landmasses ───────────────────────────────────────────
        logger.info("[4/6] Detecting landmasses …")
        # Allow a per-country cluster-distance override (e.g. Japan needs a
        # tighter 100km so Okinawa separates into its own panel instead of
        # being bundled with Kyushu and dragged to the far south).
        cluster_km = self.cluster_distance_km
        if abs(cluster_km - 200.0) < 1e-6:  # still the default → use recommendation
            rec_km = RECOMMENDED_SETTINGS.get(
                self.country_iso3, {}
            ).get("cluster_distance_km")
            if rec_km:
                cluster_km = rec_km
        raw_lms = detect_landmasses(
            self._regions_proj,
            cluster_distance_m=cluster_km * 1_000,
        )

        # A multi-part region (mainland body + offshore islets) is reported by
        # detect_landmasses in every cluster it touches.  Rasterising it once
        # per landmass would multiply its cells.  Assign each region to the
        # single landmass that overlaps it most, so every region is built once.
        owner: dict[int, int] = {}
        for iloc in range(len(self._regions_proj)):
            g = self._regions_proj.geometry.iloc[iloc]
            if g is None or g.is_empty:
                continue
            best_id, best_area = None, -1.0
            for lm in raw_lms:
                if lm.geom.intersects(g):
                    a = lm.geom.intersection(g).area
                    if a > best_area:
                        best_area, best_id = a, lm.id
            if best_id is not None:
                owner[iloc] = best_id

        # ── 5. Deform each landmass to population, then rasterise ───────────
        logger.info("[5/6] Building population cartogram cells …")
        # Oversize factor counters the perimeter cells lost when a jagged
        # deformed outline is rasterised; fit_quota trims back to exact counts.
        OVERSIZE = 1.18
        cell_frames: list[gpd.GeoDataFrame] = []
        meta: list[dict] = []
        mainland_id = raw_lms[0].id if raw_lms else None
        geo_bbox = None   # mainland bounds before deformation
        def_bbox = None   # mainland bounds after deformation

        for lm in raw_lms:
            owned_ilocs = [i for i in lm.region_idx if owner.get(i) == lm.id]
            regions_sub = (
                self._regions_proj.iloc[owned_ilocs].copy().reset_index(drop=True)
            )
            lm_quota = int(sum(
                quotas.get(str(n), 0) for n in regions_sub["region_name"]
            ))
            entry = {"id": lm.id, "geom": lm.geom, "region_idx": lm.region_idx,
                     "cells": gpd.GeoDataFrame()}
            if lm_quota <= 0 or regions_sub.empty:
                meta.append(entry)
                continue

            if lm.id == mainland_id:
                geo_bbox = tuple(regions_sub.total_bounds)

            target_area = lm_quota * cell_area * OVERSIZE
            shaped = scale_to_area(regions_sub, target_area)

            if len(shaped) > 1:
                sub_pops = {
                    str(n): max(float(pop_weights.get(str(n), 1)), 1.0)
                    for n in shaped["region_name"]
                }
                shaped = self._deform(shaped, sub_pops)

            if lm.id == mainland_id:
                def_bbox = tuple(shaped.total_bounds)

            outline = shapely.make_valid(shaped.geometry.union_all())
            # Fill the holes the deformed region polygons leave between each
            # other, so every interior cell is rasterised (and assigned to its
            # nearest region) instead of dropping through the seams.
            outline = _fill_outline_holes(outline)
            cells = generate_cells(outline, shaped, self._cell_size, lm.id)
            if not cells.empty:
                cell_frames.append(cells)
            meta.append(entry)

        # ── 6. Fit to exact per-region quotas, then enforce real borders ───
        logger.info("[6/6] Fitting quotas and enforcing real-border topology …")
        if cell_frames:
            all_cells = gpd.GeoDataFrame(
                pd.concat(cell_frames, ignore_index=True), crs=cell_frames[0].crs
            )
            # Gap-free rasterisation (filled outline) feeds quota fitting and
            # border enforcement; a final fill_gaps mops up the few seams that
            # quota-trimming re-opens at internal region borders.
            all_cells = fit_quota(all_cells, quotas, self._cell_size)
            radj = real_adjacency(self._regions_proj)
            all_cells = enforce_topology(all_cells, radj, quotas, self._cell_size)
            all_cells = fill_gaps(all_cells, self._cell_size, quotas=quotas)
            # Re-anchor detached territories to the *deformed* mainland frame so
            # they keep their geographic bearing relative to the cartogram (this
            # is why an enclave like Melilla must not drift into a panel).
            if mainland_id is not None and geo_bbox and def_bbox:
                all_cells = _reanchor_secondary(
                    all_cells, mainland_id, geo_bbox, def_bbox, self._cell_size
                )
            # Snap every cell to one global lattice (the mainland's grid phase)
            # so detached/ inline territories line up with the main grid instead
            # of sitting half a cell off. Each landmass is internally cell-spaced,
            # so this is a uniform per-landmass shift — no collisions.
            if not all_cells.empty:
                cs = self._cell_size
                main_cells = all_cells[all_cells["landmass_id"] == mainland_id]
                ref = main_cells if not main_cells.empty else all_cells
                ox = float(ref["cx"].min())
                oy = float(ref["cy"].min())
                half = cs / 2.0
                sx = ox + np.round((all_cells["cx"].to_numpy() - ox) / cs) * cs
                sy = oy + np.round((all_cells["cy"].to_numpy() - oy) / cs) * cs
                all_cells["cx"] = sx
                all_cells["cy"] = sy
                all_cells["geometry"] = shapely.box(
                    sx - half, sy - half, sx + half, sy + half
                )
        else:
            all_cells = gpd.GeoDataFrame()

        for entry in meta:
            if not all_cells.empty:
                entry["cells"] = (
                    all_cells[all_cells["landmass_id"] == entry["id"]]
                    .reset_index(drop=True)
                )
        self._landmasses = meta

        all_region_names = self._regions_proj["region_name"].dropna().tolist()
        self._built = True

        if self.subtitle is None:
            pct = round(100 / self.n, 2) if self.n > 0 else 0
            self.subtitle = (
                f"Every block = {pct}% of the national population"
                f"  ({self.total_cells:,} blocks total)"
            )

        logger.info(
            "PopGrid build complete — %d landmass(es), %d total cells.",
            len(self._landmasses), self.total_cells,
        )
        return self

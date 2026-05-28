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

from .data import load_country_regions, REGION_LABELS
from .geometry import detect_landmasses, project_regions
from .grid import cell_size_for_n, generate_cells
from .render import build_color_map, create_figure

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.country_iso3 = country_iso3.strip().upper()
        self.n = n
        self.cluster_distance_km = cluster_distance_km
        self.palette = palette
        self.dissolve_by = dissolve_by

        # Internal state – populated by build()
        self._regions_wgs84: Optional[gpd.GeoDataFrame] = None
        self._regions_proj: Optional[gpd.GeoDataFrame] = None
        self._landmasses: Optional[list[dict]] = None
        self._cell_size: Optional[float] = None
        self._color_map: Optional[dict[str, str]] = None
        self._built: bool = False

    # ── Build pipeline ────────────────────────────────────────────────────────

    def build(self) -> "AreaGrid":
        """
        Run the full data → geometry → grid pipeline.

        Called automatically on the first call to :meth:`plot` or any
        property accessor.  Safe to call explicitly for eager loading.
        """
        # 1. Load Natural Earth admin-1 data
        logger.info("[1/5] Loading regions for %s …", self.country_iso3)
        self._regions_wgs84 = load_country_regions(
            self.country_iso3,
            dissolve_by=self.dissolve_by,
        )

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
        self._color_map = build_color_map(all_region_names, self.palette)

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
    ) -> plt.Figure:
        """
        Render the population grid map.

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

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._ensure_built()
        if scope not in ("all", "islands", "mainland"):
            raise ValueError(
                f"scope must be 'all', 'islands', or 'mainland'; got {scope!r}"
            )

        if title is None:
            title = _country_title(self.country_iso3)

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
            data_source=data_source,
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

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = (
            f"built  |  {self.total_cells} cells  |  "
            f"{self.n_landmasses} landmass(es)"
            if self._built
            else "not built"
        )
        return (
            f"AreaGrid(country='{self.country_iso3}', n={self.n}, "
            f"dissolve_by={self.dissolve_by!r})  [{status}]"
        )

class PopGrid(AreaGrid):
    """
    Population tiled map — each cell represents an equal share of the
    country's population rather than its land area.

    .. note::
        Not yet implemented. Will be available in v0.4.0.
        Use :class:`AreaGrid` for land-area tiled maps.
    """

    def build(self) -> "PopGrid":
        raise NotImplementedError(
            "PopGrid (population-weighted cells) is not yet implemented. "
            "Use AreaGrid for land-area tiled maps instead:\n\n"
            "    from popgrid import AreaGrid\n"
            "    ag = AreaGrid('ESP', dissolve_by='auto')\n"
            "    ag.plot()"
        )

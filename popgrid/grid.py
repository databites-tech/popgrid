"""
Grid generation: rasterise a landmass into N equal-area square cells.

Algorithm
---------
1. Compute candidate cell centres that tile the landmass bounding box
   at the given *cell_size*.
2. Vectorised ``shapely.contains_properly`` filters centres to those
   strictly inside the landmass geometry (fast even for 10 k+ candidates).
3. ``geopandas.sjoin`` (predicate=``'within'``) assigns each centre to its
   admin-1 region.  A ``sjoin_nearest`` fallback handles the rare boundary
   cells that slip through.
4. Cell box geometries (squares centred on each kept point) are built
   vectorially with ``shapely.box``.

Sub-cell-size territories
--------------------------
Some territories (e.g. Ceuta, Melilla) are smaller than one cell at the
country-wide cell_size.  Skipping them silently is wrong — they are real,
named regions.  Instead we force exactly 1 cell placed at the territory
centroid.  The cell visually represents "this place exists" and is
correctly labelled.  The cell proportions are not accurate at this scale,
but that is a fundamental limit of any grid at N=1000 for a large country.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


# ── Cell-size helper ──────────────────────────────────────────────────────────


def cell_size_for_n(total_area_m2: float, n: int) -> float:
    """
    Return the cell side length (metres) such that a perfect rasterisation
    would yield exactly *n* cells over *total_area_m2*.

    ``cell_size = sqrt(total_area / n)``
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if total_area_m2 <= 0:
        raise ValueError(f"total_area_m2 must be positive, got {total_area_m2}")
    return float(np.sqrt(total_area_m2 / n))


# ── Forced single-cell fallback ───────────────────────────────────────────────


def _single_centroid_cell(
    landmass_geom: Polygon,
    regions_proj: gpd.GeoDataFrame,
    cell_size: float,
    landmass_id: int,
) -> gpd.GeoDataFrame:
    """
    Create exactly one cell at the landmass centroid.

    Used when the landmass bounding box is smaller than *cell_size* — i.e.
    the territory is too small to contain even one standard grid cell.
    The territory is still shown as a labelled inset cell.
    """
    cx = float(landmass_geom.centroid.x)
    cy = float(landmass_geom.centroid.y)
    half = cell_size / 2.0

    pt = shapely.points([cx], [cy])
    centre_gdf = gpd.GeoDataFrame(
        {"cx": [cx], "cy": [cy]},
        geometry=gpd.GeoSeries(pt, crs=regions_proj.crs),
        crs=regions_proj.crs,
    )
    reg_sub = regions_proj[["region_id", "region_name", "geometry"]].copy()
    nearest = gpd.sjoin_nearest(centre_gdf, reg_sub, how="left")
    nearest = nearest[~nearest.index.duplicated(keep="first")]
    nearest = nearest.drop(columns=["index_right"], errors="ignore")

    nearest["geometry"] = shapely.box(
        np.array([cx - half]),
        np.array([cy - half]),
        np.array([cx + half]),
        np.array([cy + half]),
    )
    nearest["landmass_id"] = landmass_id
    nearest = nearest.set_geometry("geometry")

    logger.info(
        "Landmass %d: territory smaller than one cell → forced 1 centroid cell "
        "(region: %s).",
        landmass_id,
        nearest["region_name"].iloc[0] if not nearest.empty else "?",
    )
    return nearest.reset_index(drop=True)


# ── Core grid generator ───────────────────────────────────────────────────────


def generate_cells(
    landmass_geom: Polygon,
    regions_proj: gpd.GeoDataFrame,
    cell_size: float,
    landmass_id: int,
) -> gpd.GeoDataFrame:
    """
    Rasterise a single landmass into grid cells and assign each to a region.

    Parameters
    ----------
    landmass_geom:
        Projected (LAEA) Shapely geometry for the landmass.
    regions_proj:
        Projected GeoDataFrame for the admin-1 regions belonging to this
        landmass.  Must contain columns ``region_id`` (int) and
        ``region_name`` (str).
    cell_size:
        Square cell side length in projected units (metres).
    landmass_id:
        Integer tag written into every output row.

    Returns
    -------
    GeoDataFrame with one row per cell:
        ``cx``, ``cy``      – cell centre (metres, projected CRS)
        ``geometry``        – square box polygon for the cell
        ``region_id``       – integer ID of the owning region
        ``region_name``     – string name of the owning region
        ``landmass_id``     – integer landmass tag
    """
    if regions_proj.empty:
        logger.warning("Landmass %d: no regions supplied — skipping.", landmass_id)
        return gpd.GeoDataFrame()

    minx, miny, maxx, maxy = landmass_geom.bounds
    half = cell_size / 2.0

    # ── Step 1: candidate cell centres tiling the bounding box ───────────────
    xs = np.arange(minx + half, maxx, cell_size)
    ys = np.arange(miny + half, maxy, cell_size)

    # Territory smaller than one cell — force a single representative cell
    if xs.size == 0 or ys.size == 0:
        return _single_centroid_cell(landmass_geom, regions_proj, cell_size, landmass_id)

    xx, yy = np.meshgrid(xs, ys)
    cx_all = xx.ravel()
    cy_all = yy.ravel()

    logger.debug(
        "Landmass %d: %d candidate cells in bounding box",
        landmass_id,
        cx_all.size,
    )

    # ── Step 2: vectorised point-in-landmass filter ───────────────────────────
    pts = shapely.points(cx_all, cy_all)
    inside = shapely.contains_properly(landmass_geom, pts)

    # No centres landed inside — landmass is small but bounding box > cell_size
    # (thin/irregular shape).  Fall back to centroid cell.
    if not inside.any():
        logger.warning(
            "Landmass %d: no cell centres fell inside landmass geometry "
            "(thin or irregular shape) — forcing 1 centroid cell.",
            landmass_id,
        )
        return _single_centroid_cell(landmass_geom, regions_proj, cell_size, landmass_id)

    cx_in = cx_all[inside]
    cy_in = cy_all[inside]
    pts_in = pts[inside]

    logger.debug(
        "Landmass %d: %d cells kept after landmass filter", landmass_id, cx_in.size
    )

    # ── Step 3: spatial join → assign each cell centre to a region ───────────
    centres_gdf = gpd.GeoDataFrame(
        {"cx": cx_in, "cy": cy_in},
        geometry=gpd.GeoSeries(pts_in, crs=regions_proj.crs),
        crs=regions_proj.crs,
    )

    reg_sub = regions_proj[["region_id", "region_name", "geometry"]].copy()

    # First pass: exact containment
    joined = gpd.sjoin(centres_gdf, reg_sub, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    # Second pass: nearest-region fallback for unassigned boundary points
    miss_mask = joined["region_id"].isna()
    if miss_mask.any():
        n_miss = miss_mask.sum()
        logger.debug(
            "Landmass %d: %d boundary cell(s) → nearest-region fallback.",
            landmass_id,
            n_miss,
        )
        missing_centres = centres_gdf.loc[miss_mask.index].copy()
        nearest = gpd.sjoin_nearest(missing_centres, reg_sub, how="left")
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        joined.loc[miss_mask, ["region_id", "region_name"]] = (
            nearest[["region_id", "region_name"]].values
        )

    joined = joined.drop(columns=["index_right"], errors="ignore")

    # ── Step 4: replace point geometry with square box geometry ──────────────
    cx_final = joined["cx"].to_numpy()
    cy_final = joined["cy"].to_numpy()

    joined["geometry"] = shapely.box(
        cx_final - half,
        cy_final - half,
        cx_final + half,
        cy_final + half,
    )
    joined["landmass_id"] = landmass_id
    joined = joined.set_geometry("geometry")

    logger.info("Landmass %d: %d cells generated.", landmass_id, len(joined))
    return joined.reset_index(drop=True)

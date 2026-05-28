"""
Geometry utilities: equal-area projection and landmass detection.

Landmass detection algorithm
-----------------------------
1. Dissolve all admin-1 regions → country outline geometry
2. Explode any MultiPolygon → list of individual Polygon objects
3. Cluster polygons whose centroids lie within *cluster_distance_m* of each
   other using a Union-Find / path-compression algorithm (O(n²) centroid
   comparisons; n is rarely > 200 for a country)
4. For each cluster, record which admin-1 regions intersect it
5. Sort clusters by area (largest = main landmass)
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import geopandas as gpd
import numpy as np
import pyproj
import shapely
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union  # used for list-of-geoms only

from .exceptions import GeometryError

logger = logging.getLogger(__name__)


# ── CRS helpers ───────────────────────────────────────────────────────────────


def laea_crs(lon_0: float, lat_0: float) -> CRS:
    """
    Lambert Azimuthal Equal-Area CRS centred on *(lon_0, lat_0)*, units metres.

    Equal-area is essential so that ``cell_size = sqrt(total_area / N)``
    produces the correct number of cells regardless of the country's latitude.
    """
    return CRS.from_proj4(
        f"+proj=laea +lat_0={lat_0:.6f} +lon_0={lon_0:.6f} "
        f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )


def project_regions(regions_wgs84: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, CRS]:
    """
    Reproject an admin-1 GeoDataFrame to a country-centred LAEA CRS.

    Uses the **median** of all polygon centroids as the projection origin,
    not the area-weighted union centroid.  This prevents a single large
    territory (e.g. Alaska at 1.7 M km²) from pulling the centre far away
    from the main inhabited area and causing it to be classified as inline
    instead of a panel inset.
    """
    # Use median of each polygon's bbox centre — valid in geographic CRS
    # (avoids "geometry is in a geographic CRS" warning from .centroid)
    bounds = regions_wgs84.geometry.bounds
    lon_0 = float(((bounds["minx"] + bounds["maxx"]) / 2).median())
    lat_0 = float(((bounds["miny"] + bounds["maxy"]) / 2).median())
    crs = laea_crs(lon_0, lat_0)
    projected = regions_wgs84.to_crs(crs)
    logger.debug("Projected to LAEA centred at (%.3f, %.3f) [median centroid]", lon_0, lat_0)
    return projected, crs


# ── Landmass detection ────────────────────────────────────────────────────────


def _explode_to_polygons(geom) -> list[Polygon]:
    """Flatten any geometry type to a list of Polygon objects."""
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    # GeometryCollection or other
    result: list[Polygon] = []
    if hasattr(geom, "geoms"):
        for g in geom.geoms:
            result.extend(_explode_to_polygons(g))
    return result


def _union_find_cluster(polygons: list[Polygon], max_dist_m: float) -> dict[int, list[int]]:
    """
    Cluster *polygons* so that any two whose centroids are ≤ *max_dist_m*
    apart end up in the same cluster.  Uses Union-Find with path compression.

    Returns
    -------
    dict mapping root_index → [member_indices]
    """
    n = len(polygons)
    parent = list(range(n))

    def find(x: int) -> int:
        # Path compression
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    cx = np.array([p.centroid.x for p in polygons])
    cy = np.array([p.centroid.y for p in polygons])

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if dist <= max_dist_m:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    return clusters


class Landmass(NamedTuple):
    """
    A geographically connected group of admin-1 polygons.

    Attributes
    ----------
    id          : unique integer
    geom        : dissolved Shapely geometry in projected (LAEA) CRS
    region_idx  : iloc-positions into the projected GeoDataFrame
    """

    id: int
    geom: Polygon | MultiPolygon
    region_idx: list[int]


def detect_landmasses(
    regions_proj: gpd.GeoDataFrame,
    cluster_distance_m: float = 200_000.0,
) -> list[Landmass]:
    """
    Identify separate landmass clusters from a projected admin-1 GeoDataFrame.

    Parameters
    ----------
    regions_proj:
        Projected (LAEA) GeoDataFrame with a ``geometry`` column.
    cluster_distance_m:
        Polygon centroids within this distance (metres) are merged into the
        same landmass.  Default 200 km works for most countries; increase for
        countries with very remote overseas territories.

    Returns
    -------
    List of :class:`Landmass` objects, sorted by area descending
    (index 0 is always the main landmass).
    """
    country_union = regions_proj.geometry.union_all()

    if country_union is None or country_union.is_empty:
        raise GeometryError("Country geometry union is empty — check input data.")

    raw_polygons = _explode_to_polygons(country_union)
    logger.debug("Country dissolved to %d individual polygon(s)", len(raw_polygons))

    if len(raw_polygons) == 1:
        return [
            Landmass(
                id=0,
                geom=raw_polygons[0],
                region_idx=list(range(len(regions_proj))),
            )
        ]

    poly_clusters = _union_find_cluster(raw_polygons, max_dist_m=cluster_distance_m)
    logger.debug(
        "Clustered %d polygons into %d landmass group(s) "
        "(cluster_distance_m=%.0f)",
        len(raw_polygons),
        len(poly_clusters),
        cluster_distance_m,
    )

    landmasses: list[Landmass] = []
    for cluster_id, poly_indices in poly_clusters.items():
        cluster_geom = unary_union([raw_polygons[i] for i in poly_indices])

        # Which admin-1 regions touch this cluster?
        region_idx = [
            iloc
            for iloc in range(len(regions_proj))
            if (
                regions_proj.geometry.iloc[iloc] is not None
                and not regions_proj.geometry.iloc[iloc].is_empty
                and cluster_geom.intersects(regions_proj.geometry.iloc[iloc])
            )
        ]

        landmasses.append(
            Landmass(id=cluster_id, geom=cluster_geom, region_idx=region_idx)
        )

    # Sort largest-first so index-0 is always the main landmass
    landmasses.sort(key=lambda lm: lm.geom.area, reverse=True)

    for i, lm in enumerate(landmasses):
        logger.info(
            "  Landmass %d: area=%.0f km²  regions=%d",
            lm.id,
            lm.geom.area / 1e6,
            len(lm.region_idx),
        )

    return landmasses

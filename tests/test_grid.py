"""
Tests for popgrid.grid — cell rasterisation and sub-cell fallback.
"""

import pytest
import geopandas as gpd
import numpy as np
from shapely.geometry import box, Point

from popgrid.data import load_country_regions
from popgrid.geometry import detect_landmasses, project_regions
from popgrid.grid import cell_size_for_n, generate_cells


# ── cell_size_for_n ───────────────────────────────────────────────────────────

def test_cell_size_basic():
    # 1 000 000 m² / 100 cells → 100 m side
    assert cell_size_for_n(1_000_000, 100) == pytest.approx(100.0)


def test_cell_size_one_cell():
    area = 500_000.0
    cs = cell_size_for_n(area, 1)
    assert cs == pytest.approx(np.sqrt(area))


def test_cell_size_raises_on_zero_n():
    with pytest.raises(ValueError):
        cell_size_for_n(1_000_000, 0)


def test_cell_size_raises_on_negative_n():
    with pytest.raises(ValueError):
        cell_size_for_n(1_000_000, -5)


def test_cell_size_raises_on_zero_area():
    with pytest.raises(ValueError):
        cell_size_for_n(0, 100)


# ── generate_cells — normal landmass ─────────────────────────────────────────

@pytest.fixture(scope="module")
def germany_main_landmass():
    regions = load_country_regions("DEU")
    projected, _ = project_regions(regions)
    lms = detect_landmasses(projected)
    total_area = projected.geometry.union_all().area
    cell_size = cell_size_for_n(total_area, 1000)
    main = lms[0]
    regions_sub = projected.iloc[main.region_idx].copy()
    cells = generate_cells(main.geom, regions_sub, cell_size, landmass_id=0)
    return cells, cell_size


def test_germany_cells_is_geodataframe(germany_main_landmass):
    cells, _ = germany_main_landmass
    assert isinstance(cells, gpd.GeoDataFrame)


def test_germany_cells_not_empty(germany_main_landmass):
    cells, _ = germany_main_landmass
    assert len(cells) > 0


def test_germany_cells_required_columns(germany_main_landmass):
    cells, _ = germany_main_landmass
    for col in ("cx", "cy", "geometry", "region_id", "region_name", "landmass_id"):
        assert col in cells.columns, f"Missing column: {col}"


def test_germany_cells_no_null_region(germany_main_landmass):
    cells, _ = germany_main_landmass
    assert cells["region_name"].notna().all()
    assert cells["region_id"].notna().all()


def test_germany_cells_correct_landmass_id(germany_main_landmass):
    cells, _ = germany_main_landmass
    assert (cells["landmass_id"] == 0).all()


def test_germany_cells_all_16_bundeslaender_present(germany_main_landmass):
    cells, _ = germany_main_landmass
    assert cells["region_name"].nunique() == 16


def test_germany_cells_geometry_is_square(germany_main_landmass):
    cells, cell_size = germany_main_landmass
    # Every cell geometry should have 4 sides (5 coords incl. closing point)
    sample = cells["geometry"].iloc[:20]
    for geom in sample:
        coords = list(geom.exterior.coords)
        assert len(coords) == 5  # closed ring


def test_germany_cells_geometry_correct_size(germany_main_landmass):
    cells, cell_size = germany_main_landmass
    sample = cells["geometry"].iloc[:20]
    for geom in sample:
        minx, miny, maxx, maxy = geom.bounds
        assert (maxx - minx) == pytest.approx(cell_size, rel=1e-3)
        assert (maxy - miny) == pytest.approx(cell_size, rel=1e-3)


def test_germany_cells_count_near_target(germany_main_landmass):
    cells, _ = germany_main_landmass
    # Allow ±10% of 1000 (grid discretisation always causes some deviation)
    assert 900 <= len(cells) <= 1100


# ── generate_cells — sub-cell-size fallback ───────────────────────────────────

@pytest.fixture(scope="module")
def tiny_landmass_data():
    """A tiny polygon (1 km²) that is smaller than the cell_size (10 km)."""
    from pyproj import CRS
    tiny_poly = box(0, 0, 1_000, 1_000)  # 1 km × 1 km
    cell_size = 10_000.0  # 10 km — much larger than the polygon

    # Minimal regions GDF
    crs = CRS.from_epsg(4326).to_3d()  # placeholder; just needs a CRS
    from pyproj import CRS as ProjCRS
    proj_crs = ProjCRS.from_proj4(
        "+proj=laea +lat_0=40 +lon_0=-3 +datum=WGS84 +units=m"
    )
    regions = gpd.GeoDataFrame(
        {"region_id": [0], "region_name": ["TestRegion"]},
        geometry=[tiny_poly],
        crs=proj_crs,
    )
    return tiny_poly, regions, cell_size


def test_subcell_fallback_returns_one_cell(tiny_landmass_data):
    tiny_poly, regions, cell_size = tiny_landmass_data
    cells = generate_cells(tiny_poly, regions, cell_size, landmass_id=99)
    assert len(cells) == 1


def test_subcell_fallback_correct_region(tiny_landmass_data):
    tiny_poly, regions, cell_size = tiny_landmass_data
    cells = generate_cells(tiny_poly, regions, cell_size, landmass_id=99)
    assert cells["region_name"].iloc[0] == "TestRegion"


def test_subcell_fallback_correct_landmass_id(tiny_landmass_data):
    tiny_poly, regions, cell_size = tiny_landmass_data
    cells = generate_cells(tiny_poly, regions, cell_size, landmass_id=99)
    assert cells["landmass_id"].iloc[0] == 99


def test_subcell_fallback_cell_centred_on_polygon_centroid(tiny_landmass_data):
    tiny_poly, regions, cell_size = tiny_landmass_data
    cells = generate_cells(tiny_poly, regions, cell_size, landmass_id=99)
    # Cell centre should be close to polygon centroid
    cx = cells["cx"].iloc[0]
    cy = cells["cy"].iloc[0]
    assert cx == pytest.approx(tiny_poly.centroid.x, abs=1.0)
    assert cy == pytest.approx(tiny_poly.centroid.y, abs=1.0)


# ── generate_cells — empty regions guard ─────────────────────────────────────

def test_empty_regions_returns_empty_geodataframe():
    from pyproj import CRS
    proj_crs = CRS.from_proj4(
        "+proj=laea +lat_0=40 +lon_0=-3 +datum=WGS84 +units=m"
    )
    empty_regions = gpd.GeoDataFrame(
        columns=["region_id", "region_name", "geometry"],
        crs=proj_crs,
    )
    poly = box(0, 0, 100_000, 100_000)
    cells = generate_cells(poly, empty_regions, 10_000, landmass_id=0)
    assert isinstance(cells, gpd.GeoDataFrame)
    assert len(cells) == 0

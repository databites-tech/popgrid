"""
Tests for popgrid.geometry — LAEA projection and landmass detection.
"""

import pytest
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon

from popgrid.data import load_country_regions
from popgrid.geometry import (
    laea_crs,
    project_regions,
    detect_landmasses,
    Landmass,
)


# ── laea_crs ──────────────────────────────────────────────────────────────────

def test_laea_crs_returns_crs():
    from pyproj import CRS
    crs = laea_crs(0.0, 0.0)
    assert isinstance(crs, CRS)


def test_laea_crs_is_metres():
    crs = laea_crs(0.0, 0.0)
    assert crs.axis_info[0].unit_name == "metre"


def test_laea_crs_different_centres_differ():
    crs1 = laea_crs(0.0, 0.0)
    crs2 = laea_crs(10.0, 40.0)
    assert crs1 != crs2


# ── project_regions ───────────────────────────────────────────────────────────

def test_project_regions_changes_crs():
    regions = load_country_regions("DEU")
    projected, crs = project_regions(regions)
    assert projected.crs != regions.crs


def test_project_regions_returns_metres():
    regions = load_country_regions("DEU")
    projected, crs = project_regions(regions)
    assert projected.crs.axis_info[0].unit_name == "metre"


def test_project_regions_preserves_row_count():
    regions = load_country_regions("DEU")
    projected, _ = project_regions(regions)
    assert len(projected) == len(regions)


def test_project_regions_area_positive():
    regions = load_country_regions("DEU")
    projected, _ = project_regions(regions)
    total_area = projected.geometry.union_all().area
    assert total_area > 0


# ── detect_landmasses ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def germany_projected():
    regions = load_country_regions("DEU")
    projected, _ = project_regions(regions)
    return projected


@pytest.fixture(scope="module")
def spain_projected():
    regions = load_country_regions("ESP", dissolve_by="region")
    projected, _ = project_regions(regions)
    return projected


def test_detect_germany_returns_list(germany_projected):
    lms = detect_landmasses(germany_projected)
    assert isinstance(lms, list)


def test_detect_germany_at_least_one_landmass(germany_projected):
    lms = detect_landmasses(germany_projected)
    assert len(lms) >= 1


def test_detect_germany_returns_landmass_objects(germany_projected):
    lms = detect_landmasses(germany_projected)
    for lm in lms:
        assert isinstance(lm, Landmass)


def test_detect_germany_first_is_largest(germany_projected):
    lms = detect_landmasses(germany_projected)
    areas = [lm.geom.area for lm in lms]
    assert areas[0] == max(areas)


def test_detect_germany_region_idx_valid(germany_projected):
    lms = detect_landmasses(germany_projected)
    n_regions = len(germany_projected)
    for lm in lms:
        for idx in lm.region_idx:
            assert 0 <= idx < n_regions


def test_detect_spain_has_multiple_landmasses(spain_projected):
    # Spain has mainland + Canaries + Balearics + micro-territories
    lms = detect_landmasses(spain_projected)
    assert len(lms) >= 3


def test_detect_spain_canaries_and_balearics_present(spain_projected):
    lms = detect_landmasses(spain_projected)
    # Collect all region names across landmasses
    all_names = set()
    for lm in lms:
        for idx in lm.region_idx:
            all_names.add(spain_projected.iloc[idx]["region_name"])
    assert "Canary Is." in all_names
    assert "Islas Baleares" in all_names


def test_detect_spain_ceuta_melilla_separate(spain_projected):
    lms = detect_landmasses(spain_projected)
    # Ceuta and Melilla must be in different landmasses from each other
    ceuta_lm = None
    melilla_lm = None
    for i, lm in enumerate(lms):
        names = [spain_projected.iloc[idx]["region_name"] for idx in lm.region_idx]
        if "Ceuta" in names:
            ceuta_lm = i
        if "Melilla" in names:
            melilla_lm = i
    assert ceuta_lm is not None
    assert melilla_lm is not None
    assert ceuta_lm != melilla_lm

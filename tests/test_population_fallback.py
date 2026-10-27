"""
Tests for the GHS-POP population fallback and the country listing helpers.
"""

import csv
from pathlib import Path

import pytest

from popgrid import PopGrid, bundled_countries, supported_countries
from popgrid.data import _CURATED_CSV, _GHS_CSV, load_region_populations

RESOURCES = Path(__file__).resolve().parents[1] / "popgrid" / "resources"


def test_both_tables_ship():
    assert _CURATED_CSV.exists()
    assert _GHS_CSV.exists()


def test_ghs_table_schema_and_coverage():
    with open(_GHS_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert {"iso3", "region_name", "population", "year", "source"} <= set(reader.fieldnames)
        rows = list(reader)
    assert len(rows) > 4000
    isos = {r["iso3"] for r in rows}
    assert len(isos) > 200
    for iso in ("CHL", "IND", "GRC", "VNM", "NGA"):
        assert iso in isos


def test_bundled_countries_is_the_curated_set():
    b = bundled_countries()
    assert len(b) == 24
    assert "ESP" in b and "CHL" not in b


def test_supported_countries_superset_of_bundled():
    s = supported_countries()
    assert set(bundled_countries()) <= set(s)
    assert "CHL" in s and "IND" in s
    assert "VAT" not in s   # fewer than three regions: no map worth drawing
    assert len(s) > 150


def test_curated_rows_win_over_ghs():
    # Spain is curated: the loader must return the census figure, not GHS.
    pops = load_region_populations("ESP", ["Madrid"], name_to_parent=None)
    with open(_CURATED_CSV, newline="", encoding="utf-8") as fh:
        curated = {
            r["region_name"]: int(float(r["population"]))
            for r in csv.DictReader(fh) if r["iso3"] == "ESP"
        }
    assert pops["Madrid"] == curated["Madrid"]


def test_uncurated_country_falls_back_to_ghs():
    pops = load_region_populations("CHL", ["Región Metropolitana de Santiago", "Atacama"])
    assert pops["Región Metropolitana de Santiago"] > 5_000_000
    assert pops["Atacama"] < 1_000_000
    assert pops["Región Metropolitana de Santiago"] > 10 * pops["Atacama"]


def test_ghs_country_totals_are_plausible():
    with open(_GHS_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    total = {}
    for r in rows:
        total[r["iso3"]] = total.get(r["iso3"], 0) + int(r["population"])
    assert 1.3e9 < total["IND"] < 1.6e9
    assert 17e6 < total["CHL"] < 22e6
    assert 60e6 < total["FRA"] < 72e6


def test_popgrid_builds_an_uncurated_country():
    pg = PopGrid("CHL", n=200)
    pg.build()
    assert 150 <= pg.total_cells <= 250
    names = set(pg.regions["region_name"])
    assert "Región Metropolitana de Santiago" in names


# ── cartogram engines ─────────────────────────────────────────────────────────

def test_engine_argument_validated():
    with pytest.raises(ValueError):
        PopGrid("ESP", engine="magic")


def test_diffusion_engine_preserves_topology_and_matches_areas():
    import numpy as np
    from popgrid.cartogram import diffusion_cartogram
    from popgrid.data import load_country_regions
    from popgrid.geometry import project_regions

    regs = load_country_regions("CHN")
    proj, _ = project_regions(regs)
    names = proj["region_name"].tolist()
    pops = load_region_populations("CHN", names)
    out = diffusion_cartogram(proj, pops, resolution=192)
    # Every polygon survives as a real polygon: nothing folded into lines.
    assert all(g.geom_type in ("Polygon", "MultiPolygon") for g in out.geometry)
    assert out.geometry.is_valid.all()
    # Areas track population to within the raster's accuracy.
    a = out.geometry.area.to_numpy()
    v = np.array([pops[n] for n in names], float)
    share = v / v.sum()
    ratio = (a / a.sum()) / share
    assert np.median(ratio) == pytest.approx(1.0, abs=0.2)
    # Regions whose target is a handful of grid cells (Tibet, the Paracels)
    # cannot shrink below the raster's resolution; fit_quota fixes their block
    # counts downstream. Everything else must land within 2x.
    big = share > 0.01
    assert ratio[big].max() < 2.0 and ratio[big].min() > 0.5


def test_auto_engine_switches_for_china_and_not_for_spain(caplog):
    import logging
    caplog.set_level(logging.INFO, logger="popgrid")
    PopGrid("ESP", n=300).build()
    assert "using diffusion engine" not in caplog.text
    caplog.clear()
    PopGrid("CHN", n=300).build()
    assert "using diffusion engine" in caplog.text


def test_folded_detector():
    import shapely
    from popgrid.core import _is_folded
    assert not _is_folded(shapely.box(0, 0, 1, 1))
    assert _is_folded(shapely.GeometryCollection([shapely.box(0, 0, 1, 1), shapely.LineString([(0, 0), (2, 2)])]))
    assert not _is_folded(None)


# ── exclude ───────────────────────────────────────────────────────────────────

def test_country_default_exclude_removes_specks():
    from popgrid import AreaGrid
    ag = AreaGrid("AUS", n=300)
    ag.build()
    names = set(ag.regions["region_name"])
    assert "Macquarie Island" not in names and "Lord Howe Island" not in names
    assert "Tasmania" in names


def test_constructor_exclude_adds_to_defaults(caplog):
    import logging
    from popgrid import AreaGrid
    caplog.set_level(logging.WARNING, logger="popgrid")
    ag = AreaGrid("AUS", n=300, exclude=["Tasmania", "Atlantis"])
    ag.build()
    names = set(ag.regions["region_name"])
    assert "Tasmania" not in names and "Macquarie Island" not in names
    assert "Atlantis" in caplog.text   # unknown names are reported, not fatal


def test_exclude_keeps_area_and_population_in_step():
    from popgrid import AreaGrid, PopGrid
    import pandas as pd
    ag = AreaGrid("AUS", n=300); ag.build()
    pg = PopGrid("AUS", n=300); pg.build()
    land = set(pd.concat([lm["cells"] for lm in ag._landmasses if not lm["cells"].empty])["region_name"])
    assert "Macquarie Island" not in land
    assert ag.color_map == pg.color_map

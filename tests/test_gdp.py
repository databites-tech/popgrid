"""
Tests for the GDP variable: the bundled table, the loader, ``PopGrid(by="gdp")``
and the layout guarantees that GDP's larger distortions exposed (no holes,
no detached connector specks, exact block counts).
"""

import csv
from collections import Counter

import numpy as np
import pytest

from popgrid import AreaGrid, PopGrid
from popgrid.data import (
    VARIABLES,
    _GDP_CSV,
    gdp_quality,
    load_region_gdp,
    load_region_values,
    supported_countries,
)


# ── Table ─────────────────────────────────────────────────────────────────────

def test_gdp_table_ships_with_schema_and_coverage():
    assert _GDP_CSV.exists()
    with open(_GDP_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert {"iso3", "region_name", "gdp", "year", "quality", "source"} <= set(reader.fieldnames)
        rows = list(reader)
    assert len(rows) > 4000
    isos = {r["iso3"] for r in rows}
    assert set(supported_countries()) <= isos, "every drawable country needs GDP rows"
    assert {r["quality"] for r in rows} == {"reported", "modelled"}


def test_gdp_quality_flags():
    assert gdp_quality("ESP") == "reported"
    assert gdp_quality("DEU") == "reported"
    assert gdp_quality("ETH") == "modelled"     # national GDP only in the source
    assert gdp_quality("XXX") is None


def test_spain_gdp_shares_match_official_statistics():
    # Eurostat regional GDP 2022 (nama_10r_2gdp), share of national total.
    from popgrid.data import admin1_parent_map, load_country_regions
    regions = load_country_regions("ESP", dissolve_by="region")["region_name"].tolist()
    gdp = load_region_gdp("ESP", regions, name_to_parent=admin1_parent_map("ESP"))
    total = sum(gdp.values())
    share = {k: 100 * v / total for k, v in gdp.items()}
    official = {"Madrid": 19.4, "Cataluña": 18.8, "Andalucía": 13.4, "Valenciana": 9.3,
                "País Vasco": 5.8, "Galicia": 5.2, "Extremadura": 1.7}
    for region, pct in official.items():
        assert abs(share[region] - pct) < 3.0, (region, share[region], pct)
    # The point of the GDP map: Madrid's share of money is far above its share of land.
    assert share["Madrid"] > 18


def test_load_region_values_dispatch():
    names = ["Bayern", "Berlin"]
    pop = load_region_values("DEU", names, by="population")
    gdp = load_region_values("DEU", names, by="gdp")
    assert pop["Bayern"] > pop["Berlin"] and gdp["Bayern"] > gdp["Berlin"]
    assert gdp["Bayern"] > pop["Bayern"]       # dollars outnumber people
    with pytest.raises(ValueError):
        load_region_values("DEU", names, by="happiness")


# ── API ───────────────────────────────────────────────────────────────────────

def test_by_argument_validated_and_defaults_to_population():
    assert PopGrid("DEU").by == "population"
    assert PopGrid("DEU", by="gdp").by == "gdp"
    assert set(VARIABLES) == {"population", "gdp"}
    with pytest.raises(ValueError):
        PopGrid("DEU", by="happiness")


@pytest.fixture(scope="module")
def spain_gdp():
    return PopGrid("ESP", by="gdp").build()


@pytest.fixture(scope="module")
def spain_pop():
    return PopGrid("ESP").build()


def test_gdp_grid_sizes_regions_by_gdp(spain_gdp, spain_pop):
    g, p = spain_gdp.block_counts, spain_pop.block_counts
    assert sum(spain_gdp.block_counts.values()) == 1000
    assert g["Madrid"] > p["Madrid"] > 100
    assert g["Andalucía"] < p["Andalucía"]
    assert "GDP" in spain_gdp.subtitle and "population" in spain_pop.subtitle


def test_same_colours_in_every_layout(spain_gdp, spain_pop):
    area = AreaGrid("ESP").build()
    assert spain_gdp.color_map == spain_pop.color_map == area.color_map


def test_default_colours_are_the_0_6_ones(spain_pop):
    """The validated population maps keep their colours release to release."""
    cm = spain_pop.color_map
    assert cm["Madrid"] == "#EDC948" and cm["Andalucía"] == "#F1CE63"
    assert cm["Cataluña"] == "#F28E2B" and cm["Galicia"] == "#FABFD2"
    de = AreaGrid("DEU").build().color_map
    assert de["Bayern"] == "#4E79A7" and de["Berlin"] == "#FFBE7D"


def test_layouts_colour_mode_separates_regions_that_meet_when_deformed():
    """Madrid and Andalucía share a yellow in the default scheme and touch in
    the GDP layout; the "layouts" mode keeps them apart in every layout and
    still gives land, population and GDP maps one colour map."""
    from popgrid.render import _color_distance
    area = AreaGrid("ESP", color_mode="layouts").build()
    gdp = PopGrid("ESP", by="gdp", color_mode="layouts").build()
    assert area.color_map == gdp.color_map
    cm = gdp.color_map
    assert _color_distance(cm["Madrid"], cm["Andalucía"]) > 0.15
    with pytest.raises(ValueError):
        AreaGrid("ESP", color_mode="rainbow")


# ── Layout guarantees ─────────────────────────────────────────────────────────

def _lattice(grid):
    """{landmass_id: {(i, j): region_name}} on one shared lattice."""
    cs = grid._cell_size
    cells = [lm["cells"] for lm in grid._landmasses if len(lm["cells"])]
    ox = min(float(c["cx"].min()) for c in cells)
    oy = min(float(c["cy"].min()) for c in cells)
    out = {}
    for lm in grid._landmasses:
        c = lm["cells"]
        if len(c) == 0:
            continue
        ix = np.round((c["cx"].to_numpy() - ox) / cs).astype(int)
        iy = np.round((c["cy"].to_numpy() - oy) / cs).astype(int)
        out[lm["id"]] = dict(zip(zip(ix.tolist(), iy.tolist()), c["region_name"]))
    return out


def _holes(occ):
    """Empty lattice cells enclosed by occupied ones."""
    xs = [k[0] for k in occ]; ys = [k[1] for k in occ]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 3, max(ys) - y0 + 3
    filled = np.zeros((w, h), bool)
    for (i, j) in occ:
        filled[i - x0 + 1, j - y0 + 1] = True
    seen = np.zeros_like(filled); stack = [(0, 0)]; seen[0, 0] = True
    while stack:
        i, j = stack.pop()
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < w and 0 <= b < h and not seen[a, b] and not filled[a, b]:
                seen[a, b] = True; stack.append((a, b))
    return int((~filled & ~seen).sum())


@pytest.mark.parametrize("iso,by", [("ESP", "gdp"), ("DEU", "gdp"), ("ESP", "population")])
def test_no_holes_and_counts_within_tolerance(iso, by):
    grid = PopGrid(iso, by=by).build()
    counts = Counter()
    for occ in _lattice(grid).values():
        assert _holes(occ) == 0
        counts.update(occ.values())
    off = {r: counts.get(r, 0) - q for r, q in grid.quotas.items() if q > 0 and counts.get(r, 0) != q}
    assert all(abs(e) <= 2 for e in off.values()), off


def test_madrid_keeps_its_real_borders_in_the_gdp_map():
    """Madrid balloons to 22% of the map by GDP; it must still touch only the
    two Castillas, never Andalucía."""
    from popgrid.cartogram import real_adjacency
    grid = PopGrid("ESP", by="gdp").build()
    radj = real_adjacency(grid._regions_proj)
    for occ in _lattice(grid).values():
        for (i, j), r in occ.items():
            if r != "Madrid":
                continue
            for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                other = occ.get(nb)
                if other and other != "Madrid":
                    assert other in radj["Madrid"], f"Madrid touches {other}"


@pytest.mark.parametrize("iso,by", [("GBR", "population"), ("CHN", "population"), ("JPN", "population"), ("ESP", "gdp")])
def test_separate_landmasses_never_touch(iso, by):
    """Deforming the mainland used to move its blocks over a detached island
    (Northern Ireland ended up inside Great Britain)."""
    grid = PopGrid(iso, by=by).build()
    lat = _lattice(grid)
    ids = list(lat)
    for a in range(len(ids)):
        A = set(lat[ids[a]])
        grown = A | {(i + di, j + dj) for (i, j) in A for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))}
        for b in range(a + 1, len(ids)):
            assert not (grown & set(lat[ids[b]])), f"landmasses {ids[a]} and {ids[b]} touch"

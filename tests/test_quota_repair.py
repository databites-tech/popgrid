"""
Tests for repair_quotas: every region ends at its quota, including regions
fully enclosed by others (issue #3), without breaking the map.
"""

from collections import Counter

import geopandas as gpd
import numpy as np
import pytest
import shapely

import popgrid.cartogram as cg
from popgrid import PopGrid
from popgrid.cartogram import _exterior, _n4, repair_quotas

CS = 10.0  # cell size for the synthetic lattices


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cells(rows: list[str], names: dict[str, str]) -> gpd.GeoDataFrame:
    """Build a cell frame from an ASCII picture; '.' is empty, top row is north."""
    recs = {"cx": [], "cy": [], "region_id": [], "region_name": [], "landmass_id": []}
    ids = {name: i for i, name in enumerate(sorted(set(names.values())))}
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            if ch == ".":
                continue
            name = names[ch]
            recs["cx"].append(c * CS)
            recs["cy"].append((len(rows) - 1 - r) * CS)
            recs["region_name"].append(name)
            recs["region_id"].append(ids[name])
            recs["landmass_id"].append(0)
    cx, cy = np.array(recs["cx"]), np.array(recs["cy"])
    return gpd.GeoDataFrame(recs, geometry=shapely.box(cx - 5, cy - 5, cx + 5, cy + 5))


def _grids(cells: gpd.GeoDataFrame, cs: float) -> list[dict]:
    out = []
    for _, g in cells.groupby("landmass_id"):
        mx, my = g["cx"].min(), g["cy"].min()
        out.append({(round((x - mx) / cs), round((y - my) / cs)): r
                    for x, y, r in zip(g["cx"], g["cy"], g["region_name"])})
    return out


def _components(keys: set) -> int:
    seen, n = set(), 0
    for start in keys:
        if start in seen:
            continue
        n += 1
        stack = [start]
        seen.add(start)
        while stack:
            c = stack.pop()
            for d in _n4(c):
                if d in keys and d not in seen:
                    seen.add(d)
                    stack.append(d)
    return n


def _invariants(cells: gpd.GeoDataFrame, cs: float, adj: dict) -> dict:
    """Enclosed holes, extra region pieces, illegal border edges, landmass parts."""
    holes = pieces = illegal = parts = 0
    for grid in _grids(cells, cs):
        ext = _exterior(grid)
        ii = [k[0] for k in grid]
        jj = [k[1] for k in grid]
        holes += sum(1 for i in range(min(ii), max(ii) + 1) for j in range(min(jj), max(jj) + 1)
                     if (i, j) not in grid and (i, j) not in ext)
        by: dict = {}
        for k, r in grid.items():
            by.setdefault(r, set()).add(k)
        pieces += sum(_components(s) - 1 for s in by.values())
        parts += _components(set(grid))
        for k, a in grid.items():
            for d in ((k[0] + 1, k[1]), (k[0], k[1] + 1)):
                b = grid.get(d)
                if b and b != a and a in adj and b in adj and b not in adj[a]:
                    illegal += 1
    return {"holes": holes, "pieces": pieces, "illegal": illegal, "parts": parts}


def _counts(cells: gpd.GeoDataFrame) -> Counter:
    return Counter(cells["region_name"])


# ── Synthetic lattices: one mechanism each ────────────────────────────────────

def test_enclosed_region_grows_to_quota():
    # B is one cell in the middle of A; it can only grow by taking A's cells.
    rows = ["AAAAAAA"] * 3 + ["AAABAAA"] + ["AAAAAAA"] * 3
    cells = _cells(rows, {"A": "A", "B": "B"})
    adj = {"A": {"B"}, "B": {"A"}}
    out = repair_quotas(cells, {"A": 44, "B": 5}, CS, adj)
    assert _counts(out) == {"A": 44, "B": 5}
    inv = _invariants(out, CS, adj)
    assert inv["holes"] == 0 and inv["pieces"] == 0 and inv["parts"] == 1


def test_enclosed_region_shrinks_to_quota():
    rows = ["AAAAAAA", "AAAAAAA", "AABBBAA", "AABBBAA", "AABBBAA", "AAAAAAA", "AAAAAAA"]
    cells = _cells(rows, {"A": "A", "B": "B"})
    adj = {"A": {"B"}, "B": {"A"}}
    out = repair_quotas(cells, {"A": 44, "B": 5}, CS, adj)
    assert _counts(out) == {"A": 44, "B": 5}
    assert _invariants(out, CS, adj) == {"holes": 0, "pieces": 0, "illegal": 0, "parts": 1}


def test_surplus_leaves_through_the_outer_edge_when_totals_differ():
    # B has 4 too many and nobody is short: the surplus must exit at the edge.
    rows = ["AAAAAAA", "AAAAAAA", "AABBBAA", "AABBBAA", "AABBBAA", "AAAAAAA", "AAAAAAA"]
    cells = _cells(rows, {"A": "A", "B": "B"})
    adj = {"A": {"B"}, "B": {"A"}}
    out = repair_quotas(cells, {"A": 40, "B": 5}, CS, adj)
    assert _counts(out) == {"A": 40, "B": 5}
    assert _invariants(out, CS, adj)["holes"] == 0


def test_deficit_enters_through_the_outer_edge_when_totals_differ():
    rows = ["AAAAA", "AABAA", "AAAAA"]
    cells = _cells(rows, {"A": "A", "B": "B"})
    adj = {"A": {"B"}, "B": {"A"}}
    out = repair_quotas(cells, {"A": 14, "B": 4}, CS, adj)
    assert _counts(out) == {"A": 14, "B": 4}
    assert _invariants(out, CS, adj)["holes"] == 0


def test_a_wall_between_non_neighbours_is_never_breached():
    # W separates L and R, which do not border in reality. W is over quota and
    # may shrink from its ends at the outer edge, but never from the middle:
    # L and R must never touch, and W must stay one piece.
    rows = ["LLWRR"] * 6
    cells = _cells(rows, {"L": "L", "W": "W", "R": "R"})
    adj = {"L": {"W"}, "W": {"L", "R"}, "R": {"W"}}
    out = repair_quotas(cells, {"L": 13, "W": 1, "R": 13}, CS, adj)
    inv = _invariants(out, CS, adj)
    assert inv["illegal"] == 0 and inv["pieces"] == 0 and inv["holes"] == 0
    assert _counts(out)["W"] >= 1


def test_region_lost_in_layout_is_seeded_next_to_its_neighbours():
    # G has a quota but no cells; it borders A and B (not C) in reality.
    rows = ["AAAABBBB"] * 4 + ["CCCCCCCC"] * 2
    cells = _cells(rows, {"A": "A", "B": "B", "C": "C"})
    adj = {"A": {"B", "C", "G"}, "B": {"A", "C", "G"}, "C": {"A", "B"}, "G": {"A", "B"}}
    out = repair_quotas(cells, {"A": 14, "B": 14, "C": 16, "G": 4}, CS, adj)
    counts = _counts(out)
    assert counts["G"] == 4
    assert _invariants(out, CS, adj)["illegal"] == 0


def test_regions_without_quota_are_left_alone():
    rows = ["AAAB", "AAAB"]
    cells = _cells(rows, {"A": "A", "B": "B"})
    out = repair_quotas(cells, {"A": 6}, CS, {"A": {"B"}, "B": {"A"}})
    assert _counts(out) == {"A": 6, "B": 2}


# ── Real countries ────────────────────────────────────────────────────────────

# Residual tolerance: a region may stay off by up to 2 blocks only where it is
# the only thing keeping two non-neighbours apart (Argentina's La Pampa is a
# one-cell wall between Buenos Aires and Mendoza/Neuquén). Correct borders win
# over the last blocks.
COUNTRIES = ["CHE", "ESP", "ARG", "DEU", "FRA", "SWE", "ITA", "GBR", "CHN"]


@pytest.fixture(scope="module")
def built():
    """Build each country once, capturing the input and output of the repair."""
    captured = {}
    original = cg.repair_quotas

    def spy(cells, quotas, cell_size, real_adj=None, **kw):
        out = original(cells, quotas, cell_size, real_adj, **kw)
        spy.last = {"before": cells, "after": out, "cs": cell_size, "adj": real_adj or {}}
        return out

    cg.repair_quotas = spy
    try:
        for iso in COUNTRIES:
            pg = PopGrid(iso)
            pg.build()
            captured[iso] = (pg, spy.last)
    finally:
        cg.repair_quotas = original
    return captured


@pytest.mark.parametrize("iso", COUNTRIES)
def test_every_region_at_its_quota(built, iso):
    pg, _ = built[iso]
    quotas, counts = pg.quotas, pg.block_counts
    errors = {r: counts.get(r, 0) - q for r, q in quotas.items() if q > 0}
    off = {r: e for r, e in errors.items() if e}
    assert all(abs(e) <= 2 for e in errors.values()), off
    assert sum(abs(e) for e in errors.values()) <= 2, off
    assert abs(pg.total_cells - pg.n) <= 2


@pytest.mark.parametrize("iso", COUNTRIES)
def test_repair_never_makes_the_map_worse(built, iso):
    _, cap = built[iso]
    before = _invariants(cap["before"], cap["cs"], cap["adj"])
    after = _invariants(cap["after"], cap["cs"], cap["adj"])
    assert after["holes"] == 0
    for key in ("holes", "pieces", "illegal", "parts"):
        assert after[key] <= before[key], (key, before, after)


def test_issue_3_la_rioja_and_zug(built):
    """La Rioja (quota 7) came out with 2, Zug (quota 15) with 26."""
    esp, _ = built["ESP"]
    che, _ = built["CHE"]
    assert esp.block_counts["La Rioja"] == esp.quotas["La Rioja"] == 7
    assert che.block_counts["Zug"] == che.quotas["Zug"] == 15
    assert esp.total_cells == 1000 and che.total_cells == 1000


def test_quotas_and_block_counts_properties():
    pg = PopGrid("PRT", n=300)
    assert sum(pg.quotas.values()) == 300
    assert pg.block_counts == {r: pg.block_counts[r] for r in pg.block_counts}
    assert set(pg.block_counts) <= set(pg.quotas)

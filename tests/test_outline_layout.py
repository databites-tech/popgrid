"""
Tests for the outline layout (the default from 0.7.1): blocks follow the
cartogram's deformed outlines instead of being grown into place, with every
region at its exact quota, no holes, real borders kept where the grid allows,
and separate landmasses kept apart.
"""

from collections import Counter

import numpy as np
import pytest

from popgrid import AreaGrid, PopGrid
from popgrid.cartogram import _balanced_prices, real_adjacency


def _lattice(grid):
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


def _straight_share(occ, run=6):
    """Share of region-border edges lying on straight runs of `run`+ blocks."""
    edges = set()
    for (i, j), r in occ.items():
        for di, dj, key in ((1, 0, ("v", i + 1, j)), (-1, 0, ("v", i, j)),
                            (0, 1, ("h", j + 1, i)), (0, -1, ("h", j, i))):
            if occ.get((i + di, j + dj)) != r:
                edges.add(key)
    lines = {}
    for o, l, pos in edges:
        lines.setdefault((o, l), []).append(pos)
    total = long = 0
    for poss in lines.values():
        poss.sort(); n = 1
        for a, b in zip(poss, poss[1:]):
            if b == a + 1:
                n += 1
            else:
                total += n; long += n if n >= run else 0; n = 1
        total += n; long += n if n >= run else 0
    return long / total if total else 0.0


# ── API ───────────────────────────────────────────────────────────────────────

def test_outline_is_the_default_and_classic_is_available():
    assert PopGrid("ESP").layout == "outline"
    assert PopGrid("ESP", layout="classic").layout == "classic"
    with pytest.raises(ValueError):
        PopGrid("ESP", layout="hexagons")


def test_balanced_prices_hit_every_quota_exactly():
    rng = np.random.default_rng(0)
    cost = rng.normal(size=(300, 7))
    quotas = np.array([100, 80, 50, 40, 20, 9, 1])
    winner, _ = _balanced_prices(cost, quotas)
    assert np.array_equal(np.bincount(winner, minlength=7), quotas)


# ── Layout quality ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def spain_gdp():
    return PopGrid("ESP", by="gdp").build()


def test_spain_gdp_exact_counts_no_holes(spain_gdp):
    counts = Counter()
    for occ in _lattice(spain_gdp).values():
        assert _holes(occ) == 0
        counts.update(occ.values())
    assert sum(counts.values()) == 1000
    assert all(counts.get(r, 0) == q for r, q in spain_gdp.quotas.items() if q > 0)


def test_spain_gdp_has_no_box_borders(spain_gdp):
    """The point of the outline layout: 0.7.0 drew a third of Spain's GDP
    borders as straight runs of 6+ blocks (Cataluña and Madrid as boxes)."""
    main = max(_lattice(spain_gdp).values(), key=len)
    assert _straight_share(main) < 0.15


def test_spain_gdp_madrid_touches_only_the_castillas(spain_gdp):
    radj = real_adjacency(spain_gdp._regions_proj)
    for occ in _lattice(spain_gdp).values():
        for (i, j), r in occ.items():
            if r != "Madrid":
                continue
            for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                o = occ.get(nb)
                if o and o != "Madrid":
                    assert o in radj["Madrid"], o


@pytest.mark.parametrize("iso,by,n", [
    ("COL", "gdp", 1000),          # Bogotá: a quarter of GDP on a speck of land
    ("HUN", "gdp", 1000),          # Budapest: 38% of GDP
    ("ESP", "gdp", 2000),
    ("DEU", "population", 500),
])
def test_concentrations_and_other_n_do_not_collapse(iso, by, n):
    grid = PopGrid(iso, by=by, n=n).build()
    counts = Counter()
    for occ in _lattice(grid).values():
        assert _holes(occ) == 0
        counts.update(occ.values())
    off = {r: counts.get(r, 0) - q for r, q in grid.quotas.items() if q > 0 and counts.get(r, 0) != q}
    assert not off, off


def test_islands_stay_off_the_mainland():
    grid = PopGrid("GBR").build()
    lat = _lattice(grid)
    ids = list(lat)
    for a in range(len(ids)):
        A = set(lat[ids[a]])
        grown = A | {(i + di, j + dj) for (i, j) in A for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))}
        for b in range(a + 1, len(ids)):
            assert not (grown & set(lat[ids[b]]))


def test_land_maps_are_untouched_by_the_layout():
    """Only population and GDP maps change; land maps rasterise geography."""
    a = AreaGrid("ESP").build()
    assert a.total_cells == 999


def test_same_map_in_every_python_process():
    """Ties are broken by name, so string-hash randomisation (a different set
    order in every interpreter) cannot change the map."""
    import os
    import subprocess
    import sys

    code = (
        "import hashlib, warnings; warnings.filterwarnings('ignore');"
        "from popgrid import PopGrid;"
        "g = PopGrid('ESP').build();"
        "rows = sorted((round(float(r.cx)), round(float(r.cy)), r.region_name)"
        " for lm in g._landmasses for r in lm['cells'].itertuples());"
        "print(hashlib.md5(repr(rows).encode()).hexdigest())"
    )
    out = set()
    for seed in ("1", "2", "4"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        res = subprocess.run([sys.executable, "-c", code], env=env,
                             capture_output=True, text=True, check=True)
        out.add(res.stdout.strip().splitlines()[-1])
    assert len(out) == 1

"""
Contiguous area cartogram (Dougenik–Chrisman–Niemeyer, 1985).

Why this module exists
----------------------
The grid generator (:func:`popgrid.grid.generate_cells`) rasterises whatever
polygon it is given and assigns each cell to the region that contains it.  If
the region polygons are left in their *geographic* shape, the country outline
is locked to geography — a populous-but-tiny region like Madrid can never grow
past its borders, and the map looks like a plain geographic map regardless of
population.

A **contiguous area cartogram** deforms the region polygons so that each
region's *area* becomes proportional to a target value (population), while
preserving:

* relative position (north stays north),
* adjacency / topology (neighbours stay neighbours),
* each region's recognisable silhouette (it is the same polygon, stretched).

Feeding the deformed polygons into ``generate_cells`` then yields, for free, a
grid in which every region receives population-proportional cells *and* keeps
its shape.  The country outline distorts — dense regions bulge, sparse regions
pinch — which is precisely the Civixplorer "population visualised" look.

Algorithm (Dougenik, Chrisman & Niemeyer 1985, "rubber-sheet" forces)
---------------------------------------------------------------------
For each iteration:

1. For every region *j* compute current area ``A_j``, centroid ``C_j`` and the
   equal-area-circle radius ``R_j = sqrt(A_j / π)``.
2. Compute the desired area ``T_j`` (value share × total area, so total area is
   conserved) and desired radius ``D_j = sqrt(T_j / π)``.  The region's "mass"
   is ``m_j = D_j − R_j`` (positive ⇒ must grow, negative ⇒ must shrink).
3. A global damping term ``force_reduction = 1 / (1 + mean_size_error)`` keeps
   the deformation stable.
4. Every boundary vertex *v* is displaced by the sum of radial forces from all
   regions::

       d = |v − C_j|
       F_j =  m_j · R_j / d        if d >  R_j   (decays with distance outside)
       F_j =  m_j · d  / R_j       if d <= R_j   (zero at the centroid, linear inside)
       v  +=  Σ_j  F_j · (v − C_j)/d · force_reduction

Repeating drives each region's area towards its target while the whole sheet
stays connected.  This is the same family of algorithm used by Tilegrams /
cartogram.js under the hood, implemented here in pure NumPy + Shapely so the
library keeps a clean ``pip install`` with no compiled dependencies.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import shapely

logger = logging.getLogger(__name__)


def _components(keys: set) -> list[set]:
    """4-connected components of a set of lattice cells."""
    seen: set = set()
    out: list[set] = []
    for start in keys:
        if start in seen:
            continue
        comp = {start}
        stack = [start]
        seen.add(start)
        while stack:
            i, j = stack.pop()
            for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if nb in keys and nb not in seen:
                    seen.add(nb)
                    comp.add(nb)
                    stack.append(nb)
        out.append(comp)
    return out


def _region_stats(
    gdf: gpd.GeoDataFrame, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (cx, cy, radius, mass, force_reduction) for the current geometries."""
    areas = gdf.geometry.area.to_numpy()
    areas = np.maximum(areas, 1e-9)
    total_area = areas.sum()

    cents = gdf.geometry.centroid
    cx = cents.x.to_numpy()
    cy = cents.y.to_numpy()

    radius = np.sqrt(areas / np.pi)

    # Desired area: value share of the (conserved) total area.
    total_value = values.sum()
    desired_area = values / total_value * total_area
    desired_radius = np.sqrt(desired_area / np.pi)

    mass = desired_radius - radius  # >0 grow, <0 shrink

    # Global damping from the mean size error (Dougenik's force-reduction term).
    size_error = np.maximum(areas, desired_area) / np.minimum(areas, desired_area)
    mean_error = float(np.mean(size_error))
    force_reduction = 1.0 / (1.0 + mean_error)

    return cx, cy, radius, mass, force_reduction


def _make_displacer(cx, cy, radius, mass, frf):
    """
    Build a vectorised coordinate-displacement function for ``shapely.transform``.

    The returned closure takes an ``(N, 2)`` array of vertices and returns the
    displaced ``(N, 2)`` array, applying the summed radial forces from every
    region (all region arrays are fixed for the iteration).
    """

    def displace(coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        px = coords[:, 0:1]  # (N, 1)
        py = coords[:, 1:2]
        dx = px - cx[np.newaxis, :]  # (N, J)
        dy = py - cy[np.newaxis, :]
        dist = np.hypot(dx, dy)  # (N, J)
        dist_safe = np.maximum(dist, 1e-9)

        # Force magnitude per region: decays outside, linear-to-zero inside.
        outside = dist > radius[np.newaxis, :]
        f_out = mass[np.newaxis, :] * radius[np.newaxis, :] / dist_safe
        f_in = mass[np.newaxis, :] * dist_safe / np.maximum(radius[np.newaxis, :], 1e-9)
        force = np.where(outside, f_out, f_in)  # (N, J)

        ux = dx / dist_safe
        uy = dy / dist_safe
        new_x = coords[:, 0] + frf * np.sum(force * ux, axis=1)
        new_y = coords[:, 1] + frf * np.sum(force * uy, axis=1)
        return np.column_stack([new_x, new_y])

    return displace


def dougenik_cartogram(
    gdf: gpd.GeoDataFrame,
    values: dict[str, float] | np.ndarray,
    value_key: str = "region_name",
    iterations: int = 20,
    max_step_fraction: float = 0.35,
    diagnostics: dict | None = None,
) -> gpd.GeoDataFrame:
    """
    Deform region polygons so area becomes proportional to ``values``.

    Parameters
    ----------
    gdf:
        Projected (equal-area, metres) GeoDataFrame with a ``region_name``
        column and polygon geometries.
    values:
        Either a ``{region_name: value}`` mapping or an array aligned with
        ``gdf`` rows.  Values are the target weights (population).
    value_key:
        Column used to look up ``values`` when a dict is passed.
    iterations:
        Number of rubber-sheet iterations.  15–25 is plenty for ~20 regions.
    max_step_fraction:
        Per-iteration displacement is clamped to this fraction of the mean
        region radius to prevent self-intersection during large inflations
        (e.g. Madrid ~9×).  Lower = more stable but slower convergence.
    diagnostics:
        Optional dict filled with ``converged`` (False when the divergence
        guard stopped the loop early), ``iterations`` actually run and the
        final ``mean_error`` / ``max_error`` area ratios.

    Returns
    -------
    A copy of ``gdf`` with deformed geometries.  Region order, ids and names
    are preserved; only ``geometry`` changes.
    """
    out = gdf.copy().reset_index(drop=True)

    if isinstance(values, dict):
        val = np.array(
            [max(float(values.get(str(name), 0.0)), 0.0) for name in out[value_key]],
            dtype=float,
        )
    else:
        val = np.asarray(values, dtype=float)
    val = np.maximum(val, 1e-9)

    # Vertex budget: if make_valid starts manufacturing geometry (severe
    # self-intersection from over-inflation), its cost explodes super-linearly.
    # Stop iterating once complexity balloons — the layout is already a good
    # starting point and fit_quota fixes the exact cell counts downstream.
    try:
        _start_verts = int(shapely.get_num_coordinates(out.geometry.values).sum())
    except Exception:  # noqa: BLE001
        _start_verts = 0
    _vert_budget = max(2000, _start_verts * 4)
    _prev_geoms = list(out.geometry)
    converged = True
    err = np.ones(len(out))
    it = -1

    for it in range(iterations):
        cx, cy, radius, mass, frf = _region_stats(out, val)

        # Clamp the step so a single iteration cannot fold the sheet.
        step_cap = max_step_fraction * float(np.mean(radius))
        mass_clamped = np.clip(mass, -step_cap, step_cap)

        displace = _make_displacer(cx, cy, radius, mass_clamped, frf)

        new_geoms = [
            shapely.transform(geom, displace, include_z=False)
            for geom in out.geometry
        ]
        # Repair any self-intersection introduced by large local displacement.
        new_geoms = [shapely.make_valid(g) for g in new_geoms]
        out["geometry"] = gpd.GeoSeries(new_geoms, crs=out.crs)

        # Divergence guard: if complexity has ballooned past the budget, the
        # sheet is folding on itself — revert to the last clean state and stop.
        try:
            nverts = int(shapely.get_num_coordinates(out.geometry.values).sum())
        except Exception:  # noqa: BLE001
            nverts = 0
        if nverts > _vert_budget:
            logger.debug(
                "cartogram stop at iter %d/%d — vertices %d exceed budget %d",
                it + 1, iterations, nverts, _vert_budget,
            )
            out["geometry"] = gpd.GeoSeries(_prev_geoms, crs=out.crs)
            converged = False
            break
        _prev_geoms = list(out.geometry)

        # Convergence diagnostic.
        areas = out.geometry.area.to_numpy()
        desired = val / val.sum() * areas.sum()
        err = np.maximum(areas, desired) / np.minimum(np.maximum(areas, 1e-9), desired)
        logger.debug(
            "cartogram iter %2d/%d — mean area error %.3f, max %.3f",
            it + 1, iterations, float(err.mean()), float(err.max()),
        )

    if diagnostics is not None:
        diagnostics.update(
            converged=converged, iterations=it + 1,
            mean_error=float(err.mean()), max_error=float(err.max()),
        )
    return out


def area_error(gdf: gpd.GeoDataFrame, values: dict[str, float], value_key="region_name"):
    """Return per-region (target_share, actual_share, ratio) for validation."""
    areas = gdf.geometry.area.to_numpy()
    val = np.array([float(values.get(str(n), 0.0)) for n in gdf[value_key]])
    tshare = val / val.sum()
    ashare = areas / areas.sum()
    ratio = np.where(ashare > 0, tshare / ashare, np.nan)
    return list(zip(gdf[value_key].tolist(), tshare, ashare, ratio))


# ── Population sizing + exact-quota fitting (used by PopGrid) ──────────────────

def scale_to_area(gdf: gpd.GeoDataFrame, target_area: float) -> gpd.GeoDataFrame:
    """
    Uniformly scale every geometry about the group's centroid so the total
    area becomes ``target_area``.  Shape is preserved exactly (similarity
    transform); only size changes.

    Used to size a *detached* landmass (island, enclave) to its population
    quota before rasterising — the population analogue of leaving it at its
    geographic land area.  Mainland regions get their sizing from the
    cartogram instead; this is for single-region or detached landmasses
    that should keep their literal silhouette.
    """
    import shapely.affinity as aff

    cur = float(gdf.geometry.area.sum())
    if cur <= 0 or target_area <= 0:
        return gdf.copy()
    factor = (target_area / cur) ** 0.5
    union = gdf.geometry.union_all()
    ox, oy = float(union.centroid.x), float(union.centroid.y)
    out = gdf.copy()
    out["geometry"] = [
        aff.scale(g, xfact=factor, yfact=factor, origin=(ox, oy))
        for g in out.geometry
    ]
    return out


def fit_quota(
    cells: gpd.GeoDataFrame,
    quotas: dict[str, int],
    cell_size: float,
) -> gpd.GeoDataFrame:
    """
    Adjust each region's cell count to **exactly** ``quotas[region]`` by
    trimming peripheral cells (over quota) or growing compact cells
    (under quota) on the integer lattice.

    Shape-preserving: trimming removes the least-connected boundary cells
    first (ragged edges go before the body), and growth adds the most-
    connected adjacent empty cells first (fills concavities, not spikes).

    Operates per landmass (cells already carry ``landmass_id``); lattice
    indices are derived from each landmass's own cell positions.
    """
    from collections import defaultdict

    if cells is None or cells.empty:
        return cells

    out_frames = []
    for lm_id, lm_cells in cells.groupby("landmass_id"):
        minx = float(lm_cells["cx"].min())
        miny = float(lm_cells["cy"].min())

        def key(cx, cy):
            return (round((cx - minx) / cell_size), round((cy - miny) / cell_size))

        occupied: dict[tuple[int, int], str] = {}
        region_id_of: dict[str, int] = {}
        reg_keys: dict[str, set] = defaultdict(set)
        for _, r in lm_cells.iterrows():
            k = key(r["cx"], r["cy"])
            occupied[k] = r["region_name"]
            reg_keys[r["region_name"]].add(k)
            region_id_of[r["region_name"]] = r["region_id"]

        def neighbours(k):
            i, j = k
            return [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]

        # ── Drop disconnected specks: keep each region's largest 4-connected
        #    component (removes 1–2 cell fragments left by rasterising a thin
        #    deformed neck).  Skipped for single-region landmasses, whose
        #    multiple components are real islands of an archipelago (Canary,
        #    Balearics) that must be preserved. ───────────────────────────────
        if len(reg_keys) > 1:
            for name in list(reg_keys):
                keys = reg_keys[name]
                if len(keys) <= 1:
                    continue
                seen: set = set()
                components: list[set] = []
                for start in keys:
                    if start in seen:
                        continue
                    comp = {start}
                    stack = [start]
                    seen.add(start)
                    while stack:
                        cur = stack.pop()
                        for nb in neighbours(cur):
                            if nb in keys and nb not in seen:
                                seen.add(nb)
                                comp.add(nb)
                                stack.append(nb)
                    components.append(comp)
                if len(components) > 1:
                    largest = max(components, key=len)
                    for comp in components:
                        if comp is not largest:
                            for k in comp:
                                occupied.pop(k, None)
                    reg_keys[name] = largest

        # ── Trim over-quota regions by eroding the OUTER silhouette ─────────
        # Remove the cell with the most empty (background) neighbours, so
        # internal region borders are never cut (cutting them leaves a hole or
        # scrambles a surrounded region). A fully-interior region with no outer
        # cell is left slightly over quota rather than punctured.
        for name in list(reg_keys):
            quota = int(quotas.get(name, len(reg_keys[name])))
            while len(reg_keys[name]) > max(quota, 1):
                cand, cand_score = None, None
                for k in reg_keys[name]:
                    empty_nb = sum(1 for nb in neighbours(k) if nb not in occupied)
                    if empty_nb == 0:
                        continue  # interior cell — removing it makes a hole
                    same_nb = sum(1 for nb in neighbours(k) if occupied.get(nb) == name)
                    score = (empty_nb, -same_nb)
                    if cand_score is None or score > cand_score:
                        cand_score, cand = score, k
                if cand is None:
                    break  # no safe outer cell to remove
                reg_keys[name].discard(cand)
                occupied.pop(cand, None)

        # ── Grow under-quota regions (add most-connected empties first) ─────
        for name in list(reg_keys):
            quota = int(quotas.get(name, len(reg_keys[name])))
            guard = 0
            while len(reg_keys[name]) < quota and guard < 50_000:
                guard += 1
                cands: dict[tuple[int, int], int] = {}
                for k in reg_keys[name]:
                    for nb in neighbours(k):
                        if nb not in occupied:
                            cands[nb] = sum(
                                1 for nb2 in neighbours(nb) if nb2 in reg_keys[name]
                            )
                if not cands:
                    break
                best = max(cands, key=lambda c: cands[c])
                reg_keys[name].add(best)
                occupied[best] = name

        # ── Rebuild GeoDataFrame for this landmass ──────────────────────────
        half = cell_size / 2.0
        recs = {"cx": [], "cy": [], "region_id": [], "region_name": [], "landmass_id": []}
        for name, keys in reg_keys.items():
            for (i, j) in keys:
                recs["cx"].append(minx + i * cell_size)
                recs["cy"].append(miny + j * cell_size)
                recs["region_id"].append(region_id_of[name])
                recs["region_name"].append(name)
                recs["landmass_id"].append(lm_id)
        if not recs["cx"]:
            continue
        cx = np.array(recs["cx"]); cy = np.array(recs["cy"])
        frame = gpd.GeoDataFrame(
            recs,
            geometry=shapely.box(cx - half, cy - half, cx + half, cy + half),
            crs=cells.crs,
        )
        out_frames.append(frame)

    import pandas as pd
    if not out_frames:
        return cells.iloc[:0].copy()
    return gpd.GeoDataFrame(pd.concat(out_frames, ignore_index=True), crs=cells.crs)


# ── Outline-following assignment (multi-region landmasses) ────────────────────
#
# Why this exists
# ---------------
# The cartogram gets each region's *share* of the landmass right, but the
# rubber-sheet engine also shrinks the landmass as a whole (Spain by GDP came
# back at half its target area). Rasterising that and letting quota fitting
# grow every region to its count by "most connected empty cell first" is
# what drew boxes: 40% of Cataluña was added by the growth rule, not by the
# cartogram. Here the deformed landmass is rescaled to its exact block count
# and the blocks are handed out so every region gets exactly its quota while
# following the deformed outlines as closely as possible.
#
# How it works
# ------------
# 1. Cells: the lattice cells deepest inside the (hole-filled) deformed
#    outline, exactly as many as the landmass's quota. No growth step, so no
#    square silhouette and no enclosed holes.
# 2. Cost of giving cell c to region r: the signed distance from c to r's
#    deformed polygon, in cells (negative inside).
# 3. Each region r carries a price u_r; c goes to the region minimising
#    cost - u. Prices are set region by region so each region wins exactly
#    its quota given the others (dual coordinate ascent of the transport
#    problem). A region whose polygon came out too small grows by a uniform
#    offset around it; one that came out too large shrinks the same way, so
#    borders stay parallel to the cartogram's borders instead of turning into
#    straight lines.
# Real-border legality, enclosed regions and the last few counts are left to
# enforce_topology / fill_gaps / repair_quotas, exactly as before.

def _signed_distances(geoms: list, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(n_cells, n_regions) signed distance to each geometry, negative inside."""
    pts = shapely.points(x, y)
    out = np.empty((len(x), len(geoms)))
    for j, g in enumerate(geoms):
        if g is None or g.is_empty:
            out[:, j] = np.inf
            continue
        shapely.prepare(g)
        inside = shapely.contains_xy(g, x, y)
        d = shapely.distance(g, pts)
        if inside.any():
            d[inside] = -shapely.distance(g.boundary, pts[inside])
        out[:, j] = d
    return out


def _balanced_prices(
    cost: np.ndarray, quotas: np.ndarray, sweeps: int = 400, u: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assignment ``argmin(cost - u)`` with each column winning ``quotas`` rows.

    Dual coordinate ascent: each region's price is set so it wins exactly its
    quota given the others' prices; sweeps repeat until every count matches
    (or ``sweeps`` run out, leaving a few cells for repair_quotas).
    Returns ``(winner, u)``; pass ``u`` back in to warm-start.
    """
    n, r = cost.shape
    u = np.zeros(r) if u is None else u.copy()
    order = np.argsort(-quotas, kind="stable")
    winner = np.argmin(cost - u, axis=1)
    if np.array_equal(np.bincount(winner, minlength=r), quotas):
        return winner, u
    for _ in range(sweeps):
        for j in order:
            adj = cost - u
            adj[:, j] = np.inf
            others = adj.min(axis=1)
            t = cost[:, j] - others          # j wins c iff u_j > t_c
            q = int(quotas[j])
            if q >= n:
                u[j] = float(np.max(t[np.isfinite(t)], initial=0.0)) + 1.0
                continue
            part = np.partition(t, [q - 1, q]) if q > 0 else np.partition(t, 0)
            lo = part[q - 1] if q > 0 else part[0] - 1.0
            hi = part[q]
            if np.isfinite(lo) and np.isfinite(hi):
                u[j] = 0.5 * (lo + hi)
            elif np.isfinite(lo):
                u[j] = lo + 1.0
        winner = np.argmin(cost - u, axis=1)
        counts = np.bincount(winner, minlength=r)
        if np.array_equal(counts, quotas):
            break
    return winner, u


def _connected_selection(x, y, order, total, cell_size, min_piece=None):
    """
    Indices of ``total`` cells, deepest first (``order``), without specks.

    Taking the deepest cells straight off a thin peninsula or a scatter of
    islets leaves single blocks floating in the sea (Calabria's toe, the
    Hebrides). Pieces smaller than ``min_piece`` blocks are dropped and the
    count is made up with the next deepest cells that touch what is kept, so
    the outline stays the cartogram's and the coast stays clean.
    """
    if min_piece is None:
        min_piece = max(4, int(round(total * 0.004)))
    order = [int(i) for i in order]
    rank = {i: r for r, i in enumerate(order)}
    ix = np.round((x - x.min()) / cell_size).astype(int)
    iy = np.round((y - y.min()) / cell_size).astype(int)
    keys = list(zip(ix.tolist(), iy.tolist()))
    chosen = set(order[:total])
    for _ in range(4):
        where = {keys[i]: i for i in chosen}
        comps = _components(set(where))
        small = [c for c in comps if len(c) < min_piece]
        if not small or len(small) == len(comps):
            break
        dropped = {where[k] for c in small for k in c}
        chosen -= dropped
        kept = {keys[i] for i in chosen}
        # Grow back by depth, only into cells touching the kept set.
        while len(chosen) < total:
            added = False
            for i in order:
                if i in chosen or i in dropped:
                    continue
                a, b = keys[i]
                if ((a + 1, b) in kept or (a - 1, b) in kept
                        or (a, b + 1) in kept or (a, b - 1) in kept):
                    chosen.add(i)
                    kept.add(keys[i])
                    added = True
                    break
            if not added:
                break
    return np.array(sorted(chosen, key=rank.__getitem__))


def assign_cells(
    shaped: gpd.GeoDataFrame,
    quotas: dict[str, int],
    cell_size: float,
    landmass_id: int,
    real_adj: dict[str, set] | None = None,
    agree: float = 0.35,
    wall: float = 1.0,
    smooth_rounds: int = 12,
) -> gpd.GeoDataFrame:
    """
    Blocks for one multi-region landmass, following the deformed outlines.

    ``shaped`` holds the deformed region polygons, already scaled so their
    total area equals the landmass's block count. Returns one row per block
    (``cx, cy, region_id, region_name, landmass_id, geometry``) with every
    region at, or within a cell or two of, its quota.
    """
    names = [str(n) for n in shaped["region_name"]]
    ids = list(shaped["region_id"]) if "region_id" in shaped else list(range(len(names)))
    q = np.array([int(quotas.get(n, 0)) for n in names])
    total = int(q.sum())
    if total <= 0:
        return gpd.GeoDataFrame()

    geoms = [shapely.make_valid(g) if g is not None else None for g in shaped.geometry]
    union = shapely.make_valid(shapely.union_all([g for g in geoms if g is not None]))
    filled = _fill_holes_any(union)

    # Candidate lattice: the outline's bounding box plus a margin wide enough
    # to grow into if the outline rasterises short.
    minx, miny, maxx, maxy = filled.bounds
    pad = 3 * cell_size + np.sqrt(total) * cell_size * 0.1
    xs = np.arange(minx - pad, maxx + pad + cell_size, cell_size)
    ys = np.arange(miny - pad, maxy + pad + cell_size, cell_size)
    gx, gy = np.meshgrid(xs, ys)
    x, y = gx.ravel(), gy.ravel()

    # 1. The `total` cells deepest inside the outline, skipping water that
    #    the deformation closed between two regions that only face each
    #    other across the sea (no shared border, no shared neighbour).
    sd_union = _signed_distances([filled], x, y)[:, 0]
    pool = np.argsort(sd_union, kind="stable")[: int(total * 1.5) + 50]
    x, y = x[pool], y[pool]
    sd_all = _signed_distances(geoms, x, y) / cell_size
    if real_adj:
        two = np.argsort(sd_all, axis=1)[:, :2]
        water = np.zeros(len(x), bool)
        nameset = set(names)
        for i in range(len(x)):
            a, b = names[two[i, 0]], names[two[i, 1]]
            if a not in real_adj or b not in real_adj or b in real_adj[a]:
                continue
            if real_adj[a] & real_adj[b] & nameset:
                continue
            if sd_all[i, two[i, 0]] > -0.75 and sd_all[i, two[i, 1]] < 0.75:
                water[i] = True
        order = np.argsort(np.where(water, np.inf, sd_union[pool]), kind="stable")
    else:
        order = np.arange(len(x))
    keep = _connected_selection(x, y, order, total, cell_size)
    x, y = x[keep], y[keep]

    # 2-3. Signed distance to every region, then balanced prices.
    cost = sd_all[keep].copy()
    for j, g in enumerate(geoms):
        if not np.isfinite(cost[:, j]).any():
            # Region lost in the deformation: fall back to its original centre.
            c = shaped.geometry.iloc[j].centroid if shaped.geometry.iloc[j] is not None else None
            if c is not None and not c.is_empty:
                cost[:, j] = np.hypot(x - c.x, y - c.y) / cell_size
    cost = np.where(np.isfinite(cost), cost, 1e6)
    winner, u = _balanced_prices(cost, q)

    # 4. Smooth and legal: re-balance a few times with two extra terms per
    #    block, from its four neighbours' current owners. A small bonus for
    #    joining the region most of its neighbours belong to (no dashed
    #    ribbons, no one-block specks), and a penalty for every neighbour it
    #    would form a border with that does not exist in reality (walls come
    #    out of the assignment itself). Quotas stay exact throughout.
    ix = np.round((x - x.min()) / cell_size).astype(int)
    iy = np.round((y - y.min()) / cell_size).astype(int)
    index = {(a, b): i for i, (a, b) in enumerate(zip(ix.tolist(), iy.tolist()))}
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    nbr = np.full((len(x), 8), -1)
    for i, (a, b) in enumerate(zip(ix.tolist(), iy.tolist())):
        for kk, (da, db) in enumerate(steps):
            nbr[i, kk] = index.get((a + da, b + db), -1)
    # Diagonal neighbours count for the smoothness bonus too, at 1/sqrt(2):
    # with edge neighbours alone the bonus rewards axis-aligned straight
    # borders, the very thing this layout removes.
    w_nb = np.array([1, 1, 1, 1, 0.7071, 0.7071, 0.7071, 0.7071])
    illegal = np.zeros((len(names), len(names)))
    if real_adj:
        for a_i, a in enumerate(names):
            if a not in real_adj:
                continue
            for b_i, b in enumerate(names):
                if b in real_adj and a != b and b not in real_adj[a]:
                    illegal[a_i, b_i] = 1.0
    onehot = np.zeros((len(x) + 1, len(names)))  # last row: "no neighbour"
    rows = np.arange(len(x))
    for _ in range(smooth_rounds):
        onehot[:] = 0.0
        onehot[rows, winner] = 1.0
        stacked = onehot[np.where(nbr >= 0, nbr, len(x))]               # (n, 8, R)
        around = np.einsum("nkr,k->nr", stacked, w_nb)
        edge = stacked[:, :4, :].sum(axis=1)
        trial = cost - agree * around + wall * (edge @ illegal.T)
        new_winner, u = _balanced_prices(trial, q, u=u)
        if np.array_equal(new_winner, winner):
            break
        winner = new_winner

    half = cell_size / 2.0
    return gpd.GeoDataFrame(
        {
            "cx": x,
            "cy": y,
            "region_id": [ids[i] for i in winner],
            "region_name": [names[i] for i in winner],
            "landmass_id": landmass_id,
        },
        geometry=shapely.box(x - half, y - half, x + half, y + half),
        crs=shaped.crs,
    )


def _fill_holes_any(geom):
    """Drop interior rings of every polygon part."""
    from shapely.geometry import MultiPolygon, Polygon

    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    parts = [p for p in getattr(geom, "geoms", []) if p.geom_type == "Polygon"]
    if not parts:
        return geom
    return shapely.make_valid(MultiPolygon([Polygon(p.exterior) for p in parts]))


def fold_stray_pieces(
    cells: gpd.GeoDataFrame,
    cell_size: float,
    real_adj: dict[str, set] | None = None,
) -> gpd.GeoDataFrame:
    """
    Hand every detached piece of a region that touches another region to the
    neighbour it shares the longest border with. Pieces surrounded only by
    sea are real islands and stay. Counts drift by the pieces moved;
    repair_quotas restores them afterwards.

    With ``real_adj``, a piece only goes to a neighbour when that creates no
    border that does not exist in reality; a piece that is the only thing
    keeping two non-neighbours apart (a wall) stays where it is.
    """
    import pandas as pd
    from collections import defaultdict

    if cells is None or cells.empty:
        return cells
    frames = []
    for lm_id, lm_cells in cells.groupby("landmass_id"):
        if lm_cells["region_name"].nunique() < 2:
            frames.append(lm_cells)
            continue
        minx = float(lm_cells["cx"].min()); miny = float(lm_cells["cy"].min())
        keys = list(zip(np.round((lm_cells["cx"].to_numpy() - minx) / cell_size).astype(int).tolist(),
                        np.round((lm_cells["cy"].to_numpy() - miny) / cell_size).astype(int).tolist()))
        occ = dict(zip(keys, lm_cells["region_name"]))
        rid = dict(zip(lm_cells["region_name"], lm_cells["region_id"]))
        for _ in range(3):
            moved = False
            by_reg: dict = defaultdict(set)
            for k, r in occ.items():
                by_reg[r].add(k)
            for name, ks in by_reg.items():
                comps = _components(ks)
                if len(comps) < 2:
                    continue
                largest = max(comps, key=len)
                for comp in comps:
                    if comp is largest:
                        continue
                    border: dict = defaultdict(int)
                    for (i, j) in comp:
                        for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                            o = occ.get(nb)
                            if o is not None and o != name:
                                border[o] += 1
                    if not border:
                        continue  # an island of this region: keep it
                    candidates = sorted(border, key=lambda r: (-border[r], r))
                    if real_adj:
                        def legal(t: str) -> bool:
                            ok_with = real_adj.get(t, set()) | {t}
                            for (i, j) in comp:
                                for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                                    if nb in comp:
                                        continue
                                    o = occ.get(nb)
                                    if o is not None and o not in ok_with and t in real_adj and o in real_adj:
                                        return False
                            return True
                        candidates = [t for t in candidates if legal(t)]
                    if not candidates:
                        continue
                    target = candidates[0]
                    for k in comp:
                        occ[k] = target
                    moved = True
            if not moved:
                break
        out = lm_cells.copy()
        out["region_name"] = [occ[k] for k in keys]
        out["region_id"] = [rid[n] for n in out["region_name"]]
        frames.append(out)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=cells.crs)


def join_dashes(
    cells: gpd.GeoDataFrame,
    real_adj: dict[str, set],
    cell_size: float,
    max_len: int = 3,
) -> gpd.GeoDataFrame:
    """
    Reconnect the dashes of a dashed wall.

    A wall region (Castilla-La Mancha between Madrid and Andalucía) sometimes
    comes out as a row of short pieces with the walled-off regions touching
    in the gaps. Each detached piece is joined back to its region's main body
    along the shortest path of at most ``max_len`` blocks, provided every
    block on the path can legally become that region and no donor region is
    split. That removes the piece and, in the gaps, the border that should
    not exist. Counts drift by the blocks moved; repair_quotas restores them.
    """
    import pandas as pd
    from collections import deque

    if cells is None or cells.empty or not real_adj:
        return cells
    frames = []
    for lm_id, lm_cells in cells.groupby("landmass_id"):
        if lm_cells["region_name"].nunique() < 2:
            frames.append(lm_cells)
            continue
        minx = float(lm_cells["cx"].min()); miny = float(lm_cells["cy"].min())
        keys = list(zip(np.round((lm_cells["cx"].to_numpy() - minx) / cell_size).astype(int).tolist(),
                        np.round((lm_cells["cy"].to_numpy() - miny) / cell_size).astype(int).tolist()))
        occ = dict(zip(keys, lm_cells["region_name"]))
        rid = dict(zip(lm_cells["region_name"], lm_cells["region_id"]))

        def legal_as(region, k):
            ok = real_adj.get(region, set()) | {region}
            for nb in _n4(k):
                o = occ.get(nb)
                if o is not None and o not in ok and o in real_adj and region in real_adj:
                    return False
            return True

        for name in sorted(set(occ.values())):
            for _ in range(20):
                mine = {k for k, v in occ.items() if v == name}
                comps = sorted(_components(mine), key=len, reverse=True)
                if len(comps) < 2:
                    break
                body = comps[0]
                joined = False
                for comp in comps[1:]:
                    if not any(occ.get(nb) not in (None, name) for k in comp for nb in _n4(k)):
                        continue  # an island: leave it
                    # BFS through other regions' blocks from the piece to the body.
                    prev = {k: None for k in comp}
                    q = deque((k, 0) for k in sorted(comp))
                    hit = None
                    while q and hit is None:
                        k, dist = q.popleft()
                        for nb in _n4(k):
                            if nb in prev or nb not in occ:
                                continue
                            if nb in body:
                                hit = k
                                break
                            if dist + 1 > max_len or occ[nb] == name:
                                continue
                            prev[nb] = k
                            q.append((nb, dist + 1))
                    if hit is None:
                        continue
                    path = []
                    k = hit
                    while k is not None and k not in comp:
                        path.append(k)
                        k = prev[k]
                    if not path:
                        continue
                    old = {k: occ[k] for k in path}
                    for k in path:
                        occ[k] = name
                    ok = all(legal_as(name, k) for k in path)
                    if ok:
                        for donor in set(old.values()):
                            rest = {k for k, v in occ.items() if v == donor}
                            before = len(_components(rest | {k for k, v in old.items() if v == donor}))
                            if rest and len(_components(rest)) > before:
                                ok = False
                                break
                    if not ok:
                        for k, v in old.items():
                            occ[k] = v
                        continue
                    joined = True
                    break
                if not joined:
                    break
        out = lm_cells.copy()
        out["region_name"] = [occ[k] for k in keys]
        out["region_id"] = [rid[n] for n in out["region_name"]]
        frames.append(out)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=cells.crs)


def open_water_gaps(
    cells: gpd.GeoDataFrame,
    real_adj: dict[str, set],
    cell_size: float,
    passes: int = 6,
) -> gpd.GeoDataFrame:
    """
    Reopen sea between regions that only face each other across water.

    Two regions that do not border each other and share no neighbour (Vestfold
    and Østfold across the Oslofjord, East Wales and the South West across
    the Bristol Channel) cannot be walled apart by a connector region. When
    the deformed coastlines close the water between them, the contact block
    on the coast is turned back into sea. The region regains the block
    elsewhere in repair_quotas, which never adds a block that creates a
    border that does not exist.
    """
    import pandas as pd

    if cells is None or cells.empty or not real_adj:
        return cells
    frames = []
    for lm_id, lm_cells in cells.groupby("landmass_id"):
        if lm_cells["region_name"].nunique() < 2:
            frames.append(lm_cells)
            continue
        minx = float(lm_cells["cx"].min()); miny = float(lm_cells["cy"].min())
        keys = list(zip(np.round((lm_cells["cx"].to_numpy() - minx) / cell_size).astype(int).tolist(),
                        np.round((lm_cells["cy"].to_numpy() - miny) / cell_size).astype(int).tolist()))
        occ = dict(zip(keys, lm_cells["region_name"]))
        present = set(occ.values())
        removed: set = set()
        for _ in range(passes):
            ext = _exterior(occ)
            drop = set()
            for k, a in occ.items():
                if not any(nb in ext for nb in _n4(k)):
                    continue  # only coastal blocks can become sea
                for nb in _n4(k):
                    b = occ.get(nb)
                    if b is None or b == a or a not in real_adj or b not in real_adj:
                        continue
                    if b in real_adj[a]:
                        continue
                    if real_adj[a] & real_adj[b] & present:
                        continue  # a connector exists: enforce_topology's job
                    drop.add(k)
            if not drop:
                break
            for k in sorted(drop):
                r = occ[k]
                mine = {c for c, v in occ.items() if v == r}
                if len(mine) > 1 and _connected_without(mine, k):
                    del occ[k]
                    removed.add(k)
        if removed:
            keep = [k in occ for k in keys]
            frames.append(lm_cells[keep])
        else:
            frames.append(lm_cells)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=cells.crs)


# ── Topology enforcement (real borders) ───────────────────────────────────────

def real_adjacency(regions_proj: gpd.GeoDataFrame) -> dict[str, set]:
    """
    Return ``{region_name: {neighbouring region names}}`` from the *geographic*
    polygons — the ground-truth border graph (e.g. Madrid touches only the two
    Castillas).  Computed with a small buffer so regions that share only a
    border (not area) still register as neighbours.
    """
    names = regions_proj["region_name"].tolist()
    geoms = list(regions_proj.geometry)
    # Buffer by a small fraction of the typical region size to catch shared
    # edges and near-touching coastlines.
    scale = float(np.sqrt(regions_proj.geometry.area.mean())) * 0.02
    buffed = [shapely.buffer(g, scale) for g in geoms]
    tree = shapely.STRtree(buffed)
    adj: dict[str, set] = {n: set() for n in names}
    for i, g in enumerate(buffed):
        for j in tree.query(g):
            if j == i:
                continue
            if shapely.intersects(g, buffed[j]):
                adj[names[i]].add(names[j])
                adj[names[j]].add(names[i])
    return adj


def enforce_topology(
    cells: gpd.GeoDataFrame,
    real_adj: dict[str, set],
    quotas: dict[str, int],
    cell_size: float,
    passes: int = 30,
    contiguous_rings: bool = False,
) -> gpd.GeoDataFrame:
    """
    Repair grid adjacencies so two regions touch on the lattice **only if they
    touch in reality**, then rebalance toward quota with topology-safe swaps.

    The repair relabels an illegal border cell (region A touching region B that
    is not a real neighbour) to a *connector* region C that borders both A and
    B in reality.  Iterated, the connector (e.g. Castilla y León / La Mancha)
    grows into a thin band wrapping the inflated region (Madrid), which is
    exactly the real topology: Madrid bordered only by the two Castillas, the
    sparse Castillas reduced to lines around it.

    A light quota rebalance then moves cells from over- to under-quota *real*
    neighbours, but only when the move keeps every adjacency legal — so the
    1-cell connective ring is never eaten away.

    Operates per landmass; single-region landmasses are returned unchanged.
    """
    import pandas as pd
    from collections import defaultdict

    if cells is None or cells.empty:
        return cells

    out_frames = []
    for lm_id, lm_cells in cells.groupby("landmass_id"):
        present = set(lm_cells["region_name"])
        if len(present) < 2:
            out_frames.append(lm_cells.copy())
            continue

        minx = float(lm_cells["cx"].min())
        miny = float(lm_cells["cy"].min())

        def key(cx, cy):
            return (round((cx - minx) / cell_size), round((cy - miny) / cell_size))

        grid: dict[tuple, str] = {}
        region_id_of: dict[str, int] = {}
        for _, r in lm_cells.iterrows():
            grid[key(r["cx"], r["cy"])] = r["region_name"]
            region_id_of[r["region_name"]] = r["region_id"]

        def nbrs(k):
            i, j = k
            return ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))

        def reg_keys_of(region):
            return {k for k, v in grid.items() if v == region}

        def connected_without(region, drop):
            keys = reg_keys_of(region) - {drop}
            if not keys:
                return False
            start = next(iter(keys))
            seen = {start}
            stack = [start]
            while stack:
                c = stack.pop()
                for d in nbrs(c):
                    if d in keys and d not in seen:
                        seen.add(d)
                        stack.append(d)
            return len(seen) == len(keys)

        def legal_at(region, k):
            for d in nbrs(k):
                o = grid.get(d)
                if o is not None and o != region and o not in real_adj.get(region, set()):
                    return False
            return True

        def double_wall(k, A, d, B) -> bool:
            best = None
            for C1 in real_adj.get(A, set()) & present:
                if C1 == B:
                    continue
                for C2 in real_adj.get(B, set()) & present:
                    if C2 == A or (C2 != C1 and C2 not in real_adj.get(C1, set())):
                        continue
                    touch = (sum(1 for x in nbrs(k) if grid.get(x) == C1)
                             + sum(1 for x in nbrs(d) if grid.get(x) == C2))
                    if not touch:
                        continue
                    room = min(quotas.get(C1, 0) - counts.get(C1, 0),
                               quotas.get(C2, 0) - counts.get(C2, 0))
                    key = (-touch, -room, str(C1), str(C2))
                    if best is None or key < best[0]:
                        best = (key, C1, C2)
            if best is None:
                return False
            _, C1, C2 = best
            old_k, old_d = grid[k], grid[d]
            grid[k], grid[d] = C1, C2
            ok = legal_at(C1, k) and legal_at(C2, d)
            grid[k], grid[d] = old_k, old_d
            if not ok:
                return False
            if not (connected_without(A, k) and connected_without(B, d)):
                return False
            grid[k], grid[d] = C1, C2
            counts[A] -= 1; counts[B] -= 1
            counts[C1] += 1; counts[C2] += 1
            return True

        for _p in range(passes):
            changed = 0
            counts = defaultdict(int)
            for v in grid.values():
                counts[v] += 1

            # ── 1. Repair illegal adjacencies ──────────────────────────────
            for k in list(grid.keys()):
                A = grid.get(k)
                if A is None:
                    continue
                for d in nbrs(k):
                    B = grid.get(d)
                    if B is None or B == A or B in real_adj.get(A, set()):
                        continue
                    conn = real_adj.get(A, set()) & real_adj.get(B, set()) & present
                    if not conn and contiguous_rings:
                        # No region borders both (two cities, each enclosed by
                        # its own county: Budapest and Székesfehérvár). Build a
                        # two-block wall instead: A's block goes to a neighbour
                        # C1 of A, B's block to a neighbour C2 of B, with C1
                        # and C2 neighbours of each other.
                        if double_wall(k, A, d, B):
                            changed += 1
                            break
                        continue
                    if not conn:
                        continue
                    over_A = counts[A] - quotas.get(A, 0)
                    over_B = counts[B] - quotas.get(B, 0)
                    if not contiguous_rings:
                        # smallest connector grows (forms the thin band)
                        C = min(conn, key=lambda c: counts.get(c, 0))
                        # Forming the ring is a *rearrangement*: relabel the cell of
                        # whichever side is more over (or less under) its quota.
                        # The connector briefly exceeds quota; the gradient flow
                        # below sheds its BODY cells to deficit regions, while the
                        # legal_at guard prevents it from ever shedding the RING
                        # cells (a cell separating two non-neighbours can't legally
                        # become either) — so rings persist and quotas still settle.
                        target_k, target_region = (k, A) if over_A >= over_B else (d, B)
                        if (legal_at(C, target_k)
                                and connected_without(target_region, target_k)):
                            grid[target_k] = C
                            counts[target_region] -= 1
                            counts[C] += 1
                            changed += 1
                            break
                        continue
                    # Contiguous rings: among every (connector, side) option,
                    # prefer the connector that already touches the cell, so
                    # the ring grows out of the connector as one band instead
                    # of being planted in alternating dashes of two
                    # connectors (the Castillas around Madrid). Then the side
                    # more over its quota, then the smaller connector.
                    sides = [(k, A, over_A), (d, B, over_B)]
                    options = []
                    for C in conn:
                        for cell, region, over in sides:
                            touch = sum(1 for d2 in nbrs(cell) if grid.get(d2) == C)
                            room = quotas.get(C, 0) - counts.get(C, 0)
                            options.append((touch == 0, room < -2, -touch, -over,
                                            counts.get(C, 0), str(C), cell, region))
                    options.sort(key=lambda o: o[:6])
                    for _, *_, C, cell, region in options:
                        # A connector may run over its quota while it forms a
                        # wall (the gradient flow below sheds its body cells),
                        # but not past what it could ever shed: a canton of 4
                        # blocks (Uri) cannot wall off a 10-block front. Then
                        # the contact stays; exact counts win over the last
                        # borders.
                        qc = quotas.get(C, 0)
                        if counts.get(C, 0) - qc >= (qc // 8 if qc < 24 else qc // 2):
                            continue
                        if legal_at(C, cell) and connected_without(region, cell):
                            grid[cell] = C
                            counts[region] -= 1
                            counts[C] += 1
                            changed += 1
                            break
                    else:
                        continue
                    break

            # ── 2. Topology-safe gradient quota flow (multi-hop) ───────────
            # Over-quota regions push boundary cells "downhill" to real
            # neighbours that are more in deficit.  Surplus from a connector
            # that grew to wrap Madrid diffuses across the graph to the small
            # regions it was taken from — restoring quotas without unwrapping.
            # Ties broken by name in the outline layout, so the map does not
            # depend on Python's per-process string hashing (set order).
            flow_order = (
                sorted(present, key=lambda r: (-(counts[r] - quotas.get(r, 0)), str(r)))
                if contiguous_rings
                else sorted(present, key=lambda r: -(counts[r] - quotas.get(r, 0)))
            )
            for region in flow_order:
                guard = 0
                while counts[region] - quotas.get(region, 0) > 0 and guard < 4000:
                    guard += 1
                    err_R = counts[region] - quotas.get(region, 0)
                    moved = False
                    for k in list(reg_keys_of(region)):
                        for d in nbrs(k):
                            S = grid.get(d)
                            if S in present and S != region:
                                err_S = counts[S] - quotas.get(S, 0)
                                if (err_R - err_S >= 2
                                        and legal_at(S, k)
                                        and connected_without(region, k)):
                                    grid[k] = S
                                    counts[region] -= 1
                                    counts[S] += 1
                                    moved = True
                                    break
                        if moved:
                            break
                    if not moved:
                        break

            if changed == 0:
                break

        half = cell_size / 2.0
        recs = {"cx": [], "cy": [], "region_id": [], "region_name": [], "landmass_id": []}
        for (i, j), name in grid.items():
            recs["cx"].append(minx + i * cell_size)
            recs["cy"].append(miny + j * cell_size)
            recs["region_id"].append(region_id_of[name])
            recs["region_name"].append(name)
            recs["landmass_id"].append(lm_id)
        cx = np.array(recs["cx"]); cy = np.array(recs["cy"])
        out_frames.append(gpd.GeoDataFrame(
            recs, geometry=shapely.box(cx - half, cy - half, cx + half, cy + half),
            crs=cells.crs,
        ))

    if not out_frames:
        return cells.iloc[:0].copy()
    return gpd.GeoDataFrame(pd.concat(out_frames, ignore_index=True), crs=cells.crs)


# ── Gap filling (close seams left between deformed regions) ────────────────────

def fill_gaps(
    cells: gpd.GeoDataFrame,
    cell_size: float,
    quotas: dict | None = None,
    min_neighbours: int = 3,
    passes: int = 12,
) -> gpd.GeoDataFrame:
    """
    Close empty cells that sit *inside* the filled area — the seams left
    between adjacent deformed region polygons and any enclosed holes.

    An empty lattice cell is filled (assigned to the majority region among its
    occupied 4-neighbours) when it is clearly interior: either it has at least
    ``min_neighbours`` occupied neighbours, or it is a one-cell-wide seam
    (occupied cells on two opposite sides).  Cells on the open outer edge of a
    region keep fewer neighbours and are left alone, so the country silhouette
    is preserved while internal white gaps disappear.

    Iterated, so holes that only become interior after their neighbours fill
    are also closed.  Operates per landmass.
    """
    import pandas as pd
    from collections import Counter

    if cells is None or cells.empty:
        return cells

    out_frames = []
    for lm_id, lm_cells in cells.groupby("landmass_id"):
        minx = float(lm_cells["cx"].min())
        miny = float(lm_cells["cy"].min())

        occ: dict[tuple, str] = {}
        region_id_of: dict[str, int] = {}
        for _, r in lm_cells.iterrows():
            occ[(round((r["cx"] - minx) / cell_size),
                 round((r["cy"] - miny) / cell_size))] = r["region_name"]
            region_id_of[r["region_name"]] = r["region_id"]

        def nbrs(k):
            i, j = k
            return ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))

        orig_counts = Counter(occ.values())

        for _p in range(passes):
            iis = [k[0] for k in occ]
            jjs = [k[1] for k in occ]
            imin, imax = min(iis) - 1, max(iis) + 1
            jmin, jmax = min(jjs) - 1, max(jjs) + 1

            # Flood-fill empty space inward from the bounding-box border.  Any
            # empty cell the outside cannot reach is an *enclosed* interior hole
            # — fill it (whatever its size) with the majority bordering region.
            exterior: set = set()
            stack = []
            for i in range(imin, imax + 1):
                for j in (jmin, jmax):
                    if (i, j) not in occ:
                        stack.append((i, j))
            for j in range(jmin, jmax + 1):
                for i in (imin, imax):
                    if (i, j) not in occ:
                        stack.append((i, j))
            while stack:
                c = stack.pop()
                if c in exterior or c in occ:
                    continue
                exterior.add(c)
                ci, cj = c
                for nb in ((ci + 1, cj), (ci - 1, cj), (ci, cj + 1), (ci, cj - 1)):
                    ni, nj = nb
                    if imin <= ni <= imax and jmin <= nj <= jmax and nb not in occ:
                        stack.append(nb)

            # Enclosed empty cells the exterior flood-fill didn't reach. Group
            # into connected pockets and fill only the small ones (the seams
            # between adjacent regions). A very large enclosed void — produced
            # only by extreme deformation, e.g. China — is left alone, since
            # filling it would dump dozens of cells into a surrounded region.
            MAX_POCKET = 50
            enclosed = [
                (i, j)
                for i in range(imin, imax + 1)
                for j in range(jmin, jmax + 1)
                if (i, j) not in occ and (i, j) not in exterior
            ]
            enclosed_set = set(enclosed)
            seen_p: set = set()
            to_fill: dict[tuple, str] = {}
            for start in enclosed:
                if start in seen_p:
                    continue
                pocket = [start]
                seen_p.add(start)
                qi = 0
                while qi < len(pocket):
                    ci, cj = pocket[qi]
                    qi += 1
                    for nb in ((ci + 1, cj), (ci - 1, cj), (ci, cj + 1), (ci, cj - 1)):
                        if nb in enclosed_set and nb not in seen_p:
                            seen_p.add(nb)
                            pocket.append(nb)
                if len(pocket) > MAX_POCKET:
                    continue  # leave large voids unfilled (China)
                for k in pocket:
                    occ_nb = [occ[d] for d in nbrs(k) if d in occ]
                    if occ_nb:
                        to_fill[k] = Counter(occ_nb).most_common(1)[0][0]
            if not to_fill:
                break
            occ.update(to_fill)

        # Count-neutral PER REGION: every region sheds exactly as many outer-
        # silhouette cells as it gained from hole-filling, so closing seams
        # never changes any region's block count (no quota drift).
        now_counts = Counter(occ.values())
        for region in now_counts:
            surplus = now_counts[region] - orig_counts.get(region, 0)
            for _ in range(surplus):
                cand, cand_e = None, 0
                for k in [c for c, v in occ.items() if v == region]:
                    e = sum(1 for nb in nbrs(k) if nb not in occ)
                    if e > cand_e:
                        cand_e, cand = e, k
                if cand is None or cand_e == 0:
                    break
                del occ[cand]

        half = cell_size / 2.0
        recs = {"cx": [], "cy": [], "region_id": [], "region_name": [], "landmass_id": []}
        for (i, j), name in occ.items():
            recs["cx"].append(minx + i * cell_size)
            recs["cy"].append(miny + j * cell_size)
            recs["region_id"].append(region_id_of[name])
            recs["region_name"].append(name)
            recs["landmass_id"].append(lm_id)
        cx = np.array(recs["cx"]); cy = np.array(recs["cy"])
        out_frames.append(gpd.GeoDataFrame(
            recs, geometry=shapely.box(cx - half, cy - half, cx + half, cy + half),
            crs=cells.crs,
        ))

    if not out_frames:
        return cells.iloc[:0].copy()
    return gpd.GeoDataFrame(pd.concat(out_frames, ignore_index=True), crs=cells.crs)


# ── Quota repair (final pass) ─────────────────────────────────────────────────
#
# Why this exists (issue #3)
# --------------------------
# fit_quota, enforce_topology and fill_gaps can only change a region's block
# count where the region touches empty space: they trim cells next to the
# outside and grow into the outside. A region that ends up fully surrounded by
# other regions has no such cell, so it keeps whatever count the layout gave it.
# La Rioja (quota 7) came out with 2, Zug (quota 15) with 26, and country
# totals drifted to 969 or 1,045 instead of 1,000.
#
# How it works
# ------------
# Cells are moved one at a time through CHAINS of neighbouring regions:
#
# * surplus region S, deficit region D, connected through regions R1..Rk:
#   S gives one cell to R1, R1 gives one to R2, ..., Rk gives one to D.
#   S -1, D +1, everyone in between unchanged, outline untouched.
# * surplus with no deficit left to pair with: the chain runs to a region on
#   the outer edge, which drops one outer cell.
# * deficit with no surplus left: an edge region adds one outer cell and the
#   chain passes cells inward to D.
#
# Every single move must keep the map honest:
# * only between regions that really border each other (real_adjacency);
#   no move may create a border that doesn't exist or erase one that does;
# * the giving region stays in one piece, and so does the landmass;
# * "outside" means empty cells reachable by flood fill from beyond the
#   bounding box. Cells next to an enclosed hole are never dropped (that would
#   grow the hole) and no added cell may seal off part of the outside.
# * Cell choice keeps shapes compact: a region gives away the cell touching the
#   receiver most, farthest from its own centroid; an edge region drops its most
#   exposed outer cell; an added cell goes into the deepest concavity, closest
#   to the region's centroid.
# Each accepted move strictly reduces the total quota error, so the loop ends.

_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _n4(c):
    i, j = c
    return ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))


def _exterior(grid: dict) -> set:
    """Empty lattice cells reachable from outside the padded bounding box."""
    if not grid:
        return set()
    iis = [k[0] for k in grid]
    jjs = [k[1] for k in grid]
    imin, imax = min(iis) - 1, max(iis) + 1
    jmin, jmax = min(jjs) - 1, max(jjs) + 1
    seen: set = set()
    stack = [(i, j) for i in range(imin, imax + 1) for j in (jmin, jmax)]
    stack += [(i, j) for j in range(jmin, jmax + 1) for i in (imin, imax)]
    while stack:
        c = stack.pop()
        if c in seen or c in grid:
            continue
        i, j = c
        if not (imin <= i <= imax and jmin <= j <= jmax):
            continue
        seen.add(c)
        stack.extend(_n4(c))
    return seen


def _connected_without(cells_set: set, c) -> bool:
    """Do ``c``'s 4-neighbours inside ``cells_set`` stay connected once ``c`` is removed?"""
    nbs = [d for d in _n4(c) if d in cells_set]
    if len(nbs) <= 1:
        return True
    target = set(nbs)
    seen = {nbs[0]}
    stack = [nbs[0]]
    found = 1
    while stack:
        cur = stack.pop()
        for d in _n4(cur):
            if d == c or d in seen or d not in cells_set:
                continue
            seen.add(d)
            if d in target:
                found += 1
                if found == len(target):
                    return True
            stack.append(d)
    return found == len(target)


class _Ctx:
    """Per-landmass snapshot used to score and validate moves."""

    def __init__(self, grid: dict):
        from collections import Counter, defaultdict

        self.cells_of: dict = defaultdict(set)
        for k, r in grid.items():
            self.cells_of[r].add(k)
        self.centroid = {
            r: (sum(k[0] for k in ks) / len(ks), sum(k[1] for k in ks) / len(ks))
            for r, ks in self.cells_of.items()
        }
        self.contacts: Counter = Counter()
        for k, a in grid.items():
            for d in ((k[0] + 1, k[1]), (k[0], k[1] + 1)):
                b = grid.get(d)
                if b is not None and b != a:
                    self.contacts[frozenset((a, b))] += 1
        self.exterior = _exterior(grid)
        self.occupied = set(grid)


def repair_quotas(
    cells: gpd.GeoDataFrame,
    quotas: dict[str, int],
    cell_size: float,
    real_adj: dict[str, set] | None = None,
    max_moves: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Bring every region to exactly its quota, including regions fully enclosed
    by others, by moving single cells through chains of neighbouring regions.

    Runs after :func:`fill_gaps`. See the block comment above for the method.
    Regions without a quota entry are left as they are. Returns a new
    GeoDataFrame with the same columns.
    """
    import pandas as pd
    from collections import Counter, deque

    if cells is None or cells.empty or not quotas:
        return cells
    real_adj = real_adj or {}

    def legal(a, b) -> bool:
        if a == b or not real_adj or a not in real_adj or b not in real_adj:
            return True
        return b in real_adj[a]

    # ── Lattice per landmass ──────────────────────────────────────────────────
    grids: dict = {}
    origins: dict = {}
    region_id_of: dict = {}
    for lm_id, g in cells.groupby("landmass_id"):
        minx = float(g["cx"].min())
        miny = float(g["cy"].min())
        origins[lm_id] = (minx, miny)
        grid: dict = {}
        for cx, cy, name, rid in zip(g["cx"].to_numpy(), g["cy"].to_numpy(),
                                     g["region_name"], g["region_id"]):
            grid[(int(round((cx - minx) / cell_size)),
                  int(round((cy - miny) / cell_size)))] = name
            region_id_of[name] = rid
        grids[lm_id] = grid

    per_lm = {lm: Counter(g.values()) for lm, g in grids.items()}
    home = {r: max(per_lm, key=lambda lm: per_lm[lm].get(r, 0))
            for c in per_lm.values() for r in c}

    def total_counts() -> Counter:
        out: Counter = Counter()
        for g in grids.values():
            out.update(g.values())
        return out

    # ── Move validity ─────────────────────────────────────────────────────────
    def topology_ok(grid, ctx, c, old, new) -> bool:
        delta: Counter = Counter()
        for d in _n4(c):
            z = grid.get(d)
            if z is None:
                continue
            if old is not None and z != old:
                delta[frozenset((old, z))] -= 1
            if new is not None and z != new:
                delta[frozenset((new, z))] += 1
        for pair, dv in delta.items():
            if dv == 0:
                continue
            before = ctx.contacts.get(pair, 0)
            after = before + dv
            a, b = tuple(pair)
            if before > 0 and after <= 0 and legal(a, b):
                return False      # would erase a border that really exists
            if dv > 0 and not legal(a, b):
                return False      # would create or lengthen a border that doesn't exist
        return True

    def best_transfer(grid, ctx, X, Y):
        xs = ctx.cells_of.get(X, set())
        if len(xs) <= 1:
            return None
        cx, cy = ctx.centroid[X]
        cands = []
        for c in xs:
            n_y = sum(1 for d in _n4(c) if grid.get(d) == Y)
            if n_y:
                # Combined perimeter of X and Y changes by 2·(n_x − n_y): give
                # away cells that stick out of X and tuck into Y, so regions
                # that only relay a cell along a chain keep their shape.
                n_x = sum(1 for d in _n4(c) if grid.get(d) == X)
                dist = (c[0] - cx) ** 2 + (c[1] - cy) ** 2
                cands.append(((n_y - n_x, dist, c), c))
        cands.sort(reverse=True)
        for _, c in cands:
            if topology_ok(grid, ctx, c, X, Y) and _connected_without(xs, c):
                return c
        return None

    def best_drop(grid, ctx, E):
        es = ctx.cells_of.get(E, set())
        if len(es) <= 1:
            return None
        cx, cy = ctx.centroid[E]
        cands = []
        for c in es:
            n_out = sum(1 for d in _n4(c) if d in ctx.exterior)
            if n_out:
                cands.append(((n_out, (c[0] - cx) ** 2 + (c[1] - cy) ** 2, c), c))
        cands.sort(reverse=True)
        for _, c in cands:
            if (topology_ok(grid, ctx, c, E, None)
                    and _connected_without(es, c)
                    and _connected_without(ctx.occupied, c)):
                return c
        return None

    def seals_outside(grid, ctx, e) -> bool:
        grid[e] = "__probe__"
        ext = _exterior(grid)
        del grid[e]
        return any(d in ctx.exterior and d not in ext for d in _n4(e))

    def best_add(grid, ctx, Y):
        ys = ctx.cells_of.get(Y, set())
        if not ys:
            return None
        cx, cy = ctx.centroid[Y]
        empties = {d for c in ys for d in _n4(c) if d not in grid}
        cands = []
        for e in empties:
            n_occ = sum(1 for d in _n4(e) if d in grid)
            n_y = sum(1 for d in _n4(e) if grid.get(d) == Y)
            enclosed = e not in ctx.exterior     # filling a hole beats growing outward
            score = (enclosed, n_occ, n_y, -((e[0] - cx) ** 2 + (e[1] - cy) ** 2), e)
            cands.append((score, e))
        cands.sort(reverse=True)
        for _, e in cands:
            if topology_ok(grid, ctx, e, None, Y) and not seals_outside(grid, ctx, e):
                return e
        return None

    # ── Chain search (BFS over the region graph of one landmass) ─────────────
    def neighbours_of(ctx, R):
        out = []
        for pair in ctx.contacts:
            if R in pair:
                (other,) = tuple(pair - {R})
                out.append(other)
        return sorted(out)

    def path_forward(grid, ctx, S, is_target):
        """S -> ... -> T where each step is a valid one-cell transfer."""
        if is_target(S):
            return [S]
        parent = {S: None}
        q = deque([S])
        while q:
            X = q.popleft()
            for Y in neighbours_of(ctx, X):
                if Y in parent or best_transfer(grid, ctx, X, Y) is None:
                    continue
                parent[Y] = X
                if is_target(Y):
                    path = [Y]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    return path[::-1]
                q.append(Y)
        return None

    def path_backward(grid, ctx, D, is_source):
        """E -> ... -> D, returned as [D, ..., E]."""
        if is_source(D):
            return [D]
        parent = {D: None}
        q = deque([D])
        while q:
            Y = q.popleft()
            for X in neighbours_of(ctx, Y):
                if X in parent or best_transfer(grid, ctx, X, Y) is None:
                    continue
                parent[X] = Y
                if is_source(X):
                    path = [X]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    return path[::-1]
                q.append(X)
        return None

    # ── Execution with rollback ───────────────────────────────────────────────
    def run(grid, steps) -> bool:
        snapshot = dict(grid)
        for kind, a, b in steps:
            ctx = _Ctx(grid)
            if kind == "transfer":
                c = best_transfer(grid, ctx, a, b)
                if c is None:
                    break
                grid[c] = b
            elif kind == "drop":
                c = best_drop(grid, ctx, a)
                if c is None:
                    break
                del grid[c]
            else:  # add
                e = best_add(grid, ctx, a)
                if e is None:
                    break
                grid[e] = a
        else:
            return True
        grid.clear()
        grid.update(snapshot)
        return False

    # ── Seed regions the layout lost entirely ─────────────────────────────────
    # A region with a quota but no cells (small, dense, squeezed out during the
    # deformation: Goa, Chandigarh) can't grow through chains. Give it one cell
    # taken from a real neighbour, at the spot that touches the most of its
    # other real neighbours, i.e. where it sits between them geographically.
    counts = total_counts()
    for D in sorted(r for r, q in quotas.items() if int(q) > 0 and counts.get(r, 0) == 0):
        nbrs_D = real_adj.get(D, set())
        if not nbrs_D:
            continue
        best = None
        for lm_id, grid in grids.items():
            ctx = None
            for c, N in grid.items():
                if N not in nbrs_D:
                    continue
                around = {grid.get((c[0] + di, c[1] + dj))
                          for di in (-1, 0, 1) for dj in (-1, 0, 1)}
                touch = len((around - {None, N}) & nbrs_D) + 1
                spare = counts.get(N, 0) - int(quotas.get(N, 0))
                score = (touch, spare, c)
                if best is not None and score <= best[0]:
                    continue
                ctx = ctx or _Ctx(grid)
                if (len(ctx.cells_of[N]) > 1 and topology_ok(grid, ctx, c, N, D)
                        and _connected_without(ctx.cells_of[N], c)):
                    best = (score, lm_id, c, N)
        if best is not None:
            _, lm_id, c, N = best
            grids[lm_id][c] = D
            home[D] = lm_id
            region_id_of.setdefault(D, region_id_of.get(N))
            counts = total_counts()
            logger.debug("repair_quotas: seeded %r in a cell of %r", D, N)

    # ── Main loop ─────────────────────────────────────────────────────────────
    counts = total_counts()
    err = {r: counts.get(r, 0) - int(q) for r, q in quotas.items() if r in home}
    limit = max_moves if max_moves is not None else 4 * sum(abs(e) for e in err.values()) + 20
    moves = 0
    while moves < limit:
        counts = total_counts()
        err = {r: counts.get(r, 0) - int(q) for r, q in quotas.items() if r in home}
        surplus = sorted((r for r, e in err.items() if e > 0), key=lambda r: (-err[r], r))
        deficit = sorted((r for r, e in err.items() if e < 0), key=lambda r: (err[r], r))
        if not surplus and not deficit:
            break
        done = False

        # 1. surplus -> deficit through neighbours (outline untouched)
        for S in surplus:
            grid = grids[home[S]]
            ctx = _Ctx(grid)
            def_here = {D for D in deficit if home[D] == home[S]}
            if not def_here:
                continue
            path = path_forward(grid, ctx, S, lambda R: R in def_here)
            if path and run(grid, [("transfer", path[i], path[i + 1])
                                   for i in range(len(path) - 1)]):
                done = True
                break

        # 2. surplus -> outer edge
        if not done:
            for S in surplus:
                grid = grids[home[S]]
                ctx = _Ctx(grid)
                path = path_forward(grid, ctx, S,
                                    lambda R, g=grid, c=ctx: best_drop(g, c, R) is not None)
                if path and run(grid, [("transfer", path[i], path[i + 1])
                                       for i in range(len(path) - 1)]
                                + [("drop", path[-1], None)]):
                    done = True
                    break

        # 3. outer edge -> deficit
        if not done:
            for D in deficit:
                grid = grids[home[D]]
                ctx = _Ctx(grid)
                path = path_backward(grid, ctx, D,
                                     lambda R, g=grid, c=ctx: best_add(g, c, R) is not None)
                if path and run(grid, [("add", path[-1], None)]
                                + [("transfer", path[i], path[i - 1])
                                   for i in range(len(path) - 1, 0, -1)]):
                    done = True
                    break

        if not done:
            break
        moves += 1

    counts = total_counts()
    left = {r: counts.get(r, 0) - int(q) for r, q in quotas.items()
            if r in home and counts.get(r, 0) != int(q)}
    if left:
        logger.info("repair_quotas: %d move(s); still off: %s", moves, left)
    else:
        logger.debug("repair_quotas: %d move(s); every region at quota", moves)

    # ── Rebuild ──────────────────────────────────────────────────────────────
    half = cell_size / 2.0
    frames = []
    for lm_id, grid in grids.items():
        if not grid:
            continue
        minx, miny = origins[lm_id]
        keys = list(grid)
        cx = np.array([minx + k[0] * cell_size for k in keys])
        cy = np.array([miny + k[1] * cell_size for k in keys])
        names = [grid[k] for k in keys]
        frames.append(gpd.GeoDataFrame(
            {"cx": cx, "cy": cy,
             "region_id": [region_id_of[n] for n in names],
             "region_name": names,
             "landmass_id": [lm_id] * len(keys)},
            geometry=shapely.box(cx - half, cy - half, cx + half, cy + half),
            crs=cells.crs,
        ))
    if not frames:
        return cells.iloc[:0].copy()
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=cells.crs)


# ── Diffusion cartogram (Gastner & Newman, 2004) ─────────────────────────────
#
# Why a second engine
# -------------------
# The rubber-sheet method above moves vertices with pairwise radial forces and
# has to be damped hard to stay stable. When one region must shrink twenty-fold
# while its neighbour grows (Tibet next to Sichuan, Nunavut next to Ontario),
# the sheet folds before the areas converge, the divergence guard stops the
# loop, and the map tears.
#
# The diffusion method instead treats population density as a fluid that
# diffuses until it is uniform everywhere, and carries the map along with the
# flow. Because the flow field is the gradient of a smooth scalar, the mapping
# is continuous and topology-preserving by construction: regions can shrink or
# grow by any factor and never cross each other. The price is a raster: the
# density lives on a grid, and the method's accuracy is bounded by its
# resolution, which is why block counts are still fixed downstream by
# ``fit_quota``.
#
# Implementation notes
# --------------------
# * Density is rasterised on an L × L grid over a padded bounding box. Sea
#   cells get the country's mean density (the Gastner–Newman convention), so
#   the outline deforms but the total area is conserved.
# * The diffusion equation is solved exactly in Fourier space on the mirrored
#   2L × 2L domain, which gives zero-flux (Neumann) boundaries with plain
#   ``numpy.fft`` and no extra dependency: rho_hat(k, t) = rho_hat(k, 0) ·
#   exp(-|k|² t).
# * Velocity is v = -∇rho / rho. Vertices are integrated with an adaptive
#   midpoint scheme: a step is accepted when the midpoint and Euler results
#   agree to ``tol`` grid cells, otherwise halved.
# * Polygon edges are densified to about one vertex per grid cell before
#   integration so long straight borders bend with the flow.


def _rasterise_density(
    gdf: gpd.GeoDataFrame,
    val: np.ndarray,
    resolution: int,
    pad: float,
) -> tuple[np.ndarray, float, float, float]:
    """
    Density grid for the diffusion engine.

    Returns ``(rho, x0, y0, cell)``: the L × L density array (row = y, column
    = x, origin bottom-left) and the affine mapping from grid to data
    coordinates ``x = x0 + (i + 0.5) * cell``.
    """
    minx, miny, maxx, maxy = gdf.total_bounds
    w, h = maxx - minx, maxy - miny
    side = max(w, h) * (1.0 + 2.0 * pad)
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    x0, y0 = cx - side / 2.0, cy - side / 2.0
    cell = side / resolution

    centres = (np.arange(resolution) + 0.5) * cell
    gx, gy = np.meshgrid(x0 + centres, y0 + centres)   # shape (L, L), row = y
    flat_x, flat_y = gx.ravel(), gy.ravel()

    owner = np.full(flat_x.shape, -1, dtype=np.int64)
    for j, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        inside = shapely.contains_xy(geom, flat_x, flat_y)
        owner[inside & (owner < 0)] = j

    counts = np.bincount(owner[owner >= 0], minlength=len(gdf)).astype(float)
    land = counts.sum()
    mean_density = val.sum() / max(land, 1.0)
    region_density = np.where(counts > 0, val / np.maximum(counts, 1.0), mean_density)

    rho = np.full(flat_x.shape, mean_density, dtype=float)
    mask = owner >= 0
    rho[mask] = region_density[owner[mask]]
    return rho.reshape(resolution, resolution), x0, y0, cell


def _bilinear(field: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Sample ``field[row=y, col=x]`` at continuous grid positions (cell units)."""
    L = field.shape[0]
    fx = np.clip(px - 0.5, 0.0, L - 1.0001)
    fy = np.clip(py - 0.5, 0.0, L - 1.0001)
    ix = fx.astype(int)
    iy = fy.astype(int)
    tx = fx - ix
    ty = fy - iy
    ix1 = np.minimum(ix + 1, L - 1)
    iy1 = np.minimum(iy + 1, L - 1)
    return (
        field[iy, ix] * (1 - tx) * (1 - ty)
        + field[iy, ix1] * tx * (1 - ty)
        + field[iy1, ix] * (1 - tx) * ty
        + field[iy1, ix1] * tx * ty
    )


def diffusion_cartogram(
    gdf: gpd.GeoDataFrame,
    values: dict[str, float] | np.ndarray,
    value_key: str = "region_name",
    resolution: int = 384,
    pad: float = 0.35,
    tol: float = 0.05,
    max_steps: int = 2000,
) -> gpd.GeoDataFrame:
    """
    Deform region polygons with a Gastner–Newman diffusion cartogram.

    Continuous and topology-preserving for any ratio of target values, at the
    cost of a raster whose ``resolution`` bounds the accuracy of the areas.
    Exact block counts are fixed downstream by :func:`fit_quota`, so a few
    percent of area error here is harmless.

    Parameters
    ----------
    gdf:
        Projected (equal-area, metres) GeoDataFrame with polygon geometries.
    values:
        ``{region_name: value}`` or an array aligned with ``gdf`` rows.
    resolution:
        Density grid size per side. 256 is fast, 384 (default) is a good
        balance, 512 for very detailed outlines.
    pad:
        Sea margin around the country as a fraction of its larger dimension.
        The outline needs room to expand into.
    tol:
        Integration tolerance in grid cells per step.
    """
    out = gdf.copy().reset_index(drop=True)
    if isinstance(values, dict):
        val = np.array(
            [max(float(values.get(str(name), 0.0)), 0.0) for name in out[value_key]],
            dtype=float,
        )
    else:
        val = np.asarray(values, dtype=float)
    val = np.maximum(val, 1e-9)

    L = int(resolution)
    rho0, x0, y0, cell = _rasterise_density(out, val, L, pad)

    # Densify edges so straight borders can bend with the flow.
    geoms = [shapely.segmentize(g, cell * 1.5) for g in out.geometry]
    coords = shapely.get_coordinates(np.array(geoms, dtype=object))
    px = (coords[:, 0] - x0) / cell          # continuous grid coordinates
    py = (coords[:, 1] - y0) / cell

    # Exact diffusion in Fourier space on the mirrored (Neumann) domain.
    mirrored = np.block([
        [rho0, rho0[:, ::-1]],
        [rho0[::-1, :], rho0[::-1, ::-1]],
    ])
    rho_hat = np.fft.rfft2(mirrored)
    ky = 2.0 * np.pi * np.fft.fftfreq(2 * L)[:, None]
    kx = 2.0 * np.pi * np.fft.rfftfreq(2 * L)[None, :]
    k2 = kx ** 2 + ky ** 2
    rho_mean = float(rho0.mean())

    def velocity(t: float) -> tuple[np.ndarray, np.ndarray, float]:
        rho = np.fft.irfft2(rho_hat * np.exp(-k2 * t), s=(2 * L, 2 * L))
        gy_, gx_ = np.gradient(rho)                  # d/drow, d/dcol
        rho_c = rho[:L, :L]
        vx = -gx_[:L, :L] / rho_c
        vy = -gy_[:L, :L] / rho_c
        spread = float((rho_c.max() - rho_c.min()) / rho_mean)
        return vx, vy, spread

    t = 0.0
    dt = 1e-3
    x = px.copy()
    y = py.copy()
    steps = 0
    while steps < max_steps:
        vx, vy, spread = velocity(t)
        if spread < 1e-4:
            break
        v1x = _bilinear(vx, x, y)
        v1y = _bilinear(vy, x, y)
        # Midpoint step with the Euler step as the error estimate.
        while True:
            mx = x + 0.5 * dt * v1x
            my = y + 0.5 * dt * v1y
            vmx, vmy, _ = velocity(t + 0.5 * dt)
            v2x = _bilinear(vmx, mx, my)
            v2y = _bilinear(vmy, mx, my)
            nx = x + dt * v2x
            ny = y + dt * v2y
            err = float(np.max(np.hypot(nx - (x + dt * v1x), ny - (y + dt * v1y))))
            if err <= tol or dt < 1e-6:
                break
            dt *= 0.5
        x, y = nx, ny
        t += dt
        steps += 1
        dt = min(dt * 1.3, 0.25 * L * L)
        if err < 0.2 * tol:
            dt *= 1.3
    logger.debug("diffusion cartogram: %d steps, t=%.1f, spread=%.2e", steps, t, spread)

    new_coords = np.column_stack([x0 + x * cell, y0 + y * cell])
    moved = shapely.set_coordinates(np.array(geoms, dtype=object), new_coords)
    moved = [shapely.make_valid(g) for g in moved]
    out["geometry"] = gpd.GeoSeries(moved, crs=out.crs)
    return out

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
                    if not conn:
                        continue
                    # smallest connector grows (forms the thin band)
                    C = min(conn, key=lambda c: counts.get(c, 0))
                    # Forming the ring is a *rearrangement*: relabel the cell of
                    # whichever side is more over (or less under) its quota.
                    # The connector briefly exceeds quota; the gradient flow
                    # below sheds its BODY cells to deficit regions, while the
                    # legal_at guard prevents it from ever shedding the RING
                    # cells (a cell separating two non-neighbours can't legally
                    # become either) — so rings persist and quotas still settle.
                    over_A = counts[A] - quotas.get(A, 0)
                    over_B = counts[B] - quotas.get(B, 0)
                    target_k, target_region = (k, A) if over_A >= over_B else (d, B)
                    if (legal_at(C, target_k)
                            and connected_without(target_region, target_k)):
                        grid[target_k] = C
                        counts[target_region] -= 1
                        counts[C] += 1
                        changed += 1
                        break

            # ── 2. Topology-safe gradient quota flow (multi-hop) ───────────
            # Over-quota regions push boundary cells "downhill" to real
            # neighbours that are more in deficit.  Surplus from a connector
            # that grew to wrap Madrid diffuses across the graph to the small
            # regions it was taken from — restoring quotas without unwrapping.
            for region in sorted(present, key=lambda r: -(counts[r] - quotas.get(r, 0))):
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

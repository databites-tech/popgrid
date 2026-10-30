#!/usr/bin/env python3
"""
Generate popgrid maps for one or more countries.

Grid modes:
  pop  (default)  every cell = an equal share of the country's POPULATION
  gdp             every cell = an equal share of the country's GDP (PPP, 2024)
  area            every cell = an equal share of the country's LAND AREA

Usage
-----
    python generate.py                       # default set, population maps
    python generate.py ESP DEU JPN           # specific countries
    python generate.py ALL                   # the 24 countries with curated census data
    python generate.py WORLD                 # every country (GHS-POP estimates elsewhere)
    python generate.py ESP --mode area       # land-area map
    python generate.py ESP --mode gdp        # GDP map (official regional GDP in 89 countries)
    python generate.py ESP --mode compare    # area + population side by side
    python generate.py ESP DEU --mode area   # land-area maps, batch
    python generate.py ESP --n 1500          # finer detail
    python generate.py ESP --out maps        # custom output folder
    python generate.py ESP --palette databites --background solid
    python generate.py ESP --mainland        # mainland only, no territories
    python generate.py ESP --no-panels       # keep inline islands/enclaves, drop panels
    python generate.py ESP --show            # display instead of saving

Outputs PNGs to ./out/ (or --out) named <ISO3>_<mode>.png. dissolve_by='auto'
picks the right administrative level per country automatically.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.WARNING, format="%(message)s")

from popgrid import AreaGrid, PopGrid               # noqa: E402
from popgrid.data import bundled_countries, gdp_quality, supported_countries  # noqa: E402

# Default batch: the reference set plus a couple of stress tests.
DEFAULT_SET = ["ESP", "DEU", "ITA", "FRA", "GBR"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def country_name(iso3: str) -> str:
    """Human-readable country name for titles; falls back to the ISO code."""
    try:
        import pycountry
        rec = pycountry.countries.get(alpha_3=iso3)
        if rec is not None:
            return getattr(rec, "common_name", None) or rec.name
    except Exception:  # noqa: BLE001
        pass
    return iso3


def resolve_palette(raw: str) -> "str | list[str]":
    """Allow --palette '#hex,#hex,…' for custom colour lists."""
    return [c.strip() for c in raw.split(",")] if "," in raw else raw


def resolve_dissolve(raw: "str | None") -> "str | None":
    if raw is None or raw.lower() in ("none", ""):
        return None
    return raw


def resolve_codes(countries: list[str]) -> list[str]:
    if not countries:
        return DEFAULT_SET
    if len(countries) == 1 and countries[0].upper() == "ALL":
        return bundled_countries()
    if len(countries) == 1 and countries[0].upper() == "WORLD":
        return supported_countries()
    return [c.upper() for c in countries]


def title_for(iso: str, mode: str, override: "str | None") -> "str | None":
    if override:
        return override
    name = country_name(iso)
    if mode == "pop":
        return f"Population of {name}, Visualised"
    if mode == "gdp":
        return f"Economy of {name}, Visualised"
    if mode == "area":
        return f"Land Area of {name}, Visualised"
    return None


FOREST = "#1A3829"  # DataBites brand dark green, used for the arrow


def style_from(args) -> dict:
    """Collect the --edge-*/--grid-*/--label-* flags into a plot() style dict."""
    return {
        "edge_color": args.edge_color,
        "edge_width": args.edge_width,
        "grid_color": args.grid_color,
        "label_size": args.label_size,
    }


def _build(GridClass, iso, args):
    grid = GridClass(
        iso, n=args.n, cluster_distance_km=args.cluster,
        palette=resolve_palette(args.palette), dissolve_by=resolve_dissolve(args.dissolve),
        engine=args.engine, exclude=args.exclude,
    )
    grid.build()
    return grid


def _build_from_gdf(GridClass, gdf, args):
    kwargs = dict(
        gdf=gdf, region_col=args.region_col, n=args.n,
        cluster_distance_km=args.cluster, palette=resolve_palette(args.palette),
        engine=args.engine, exclude=args.exclude,
    )
    if GridClass is PopGrid:
        kwargs["weight_col"] = args.weight_col
    grid = GridClass.from_geodataframe(**kwargs)
    grid.build()
    return grid


def load_data_gdf(path, region_col, bbox=None):
    """Load a custom dataset and return a GeoDataFrame ready for
    ``AreaGrid/PopGrid.from_geodataframe``.

    Format is dispatched on file extension:

    * ``.geojson`` / ``.json`` / ``.gpkg`` / ``.shp`` / ``.parquet``
      Read directly with geopandas. Geometry is already in the file.
    * ``.csv``
      Read as a table and joined to Natural Earth 1:50m country borders on
      ``region_col`` (typically ``iso3``). The shapefile is downloaded to
      ``~/.popgrid/cache/`` on first use.

    A bbox ``(lon_min, lat_min, lon_max, lat_max)`` clips the result.
    """
    import geopandas as gpd
    from shapely.geometry import box as _box
    from pathlib import Path

    p = Path(path)
    ext = p.suffix.lower()

    if ext in {".geojson", ".json", ".gpkg", ".shp", ".parquet"}:
        gdf = gpd.read_file(p) if ext != ".parquet" else gpd.read_parquet(p)
        if region_col not in gdf.columns:
            raise ValueError(
                f"{p.name} missing region column {region_col!r}. "
                f"Available: {list(gdf.columns)}"
            )
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
    elif ext == ".csv":
        import urllib.request
        import pandas as pd
        from popgrid.data import cache_dir

        url = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
        zip_path = cache_dir() / "ne_50m_admin_0_countries.zip"
        if not zip_path.exists() or zip_path.stat().st_size < 100_000:
            req = urllib.request.Request(url, headers={"User-Agent": "popgrid"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                zip_path.write_bytes(resp.read())

        df = pd.read_csv(p)
        if region_col not in df.columns:
            raise ValueError(
                f"{p.name} missing region column {region_col!r}. "
                f"Available: {list(df.columns)}"
            )
        world = gpd.read_file(f"zip://{zip_path}")
        iso_src = next(
            (c for c in ("ISO_A3_EH", "ADM0_A3", "ISO_A3") if c in world.columns), None,
        )
        if iso_src is None:
            raise RuntimeError("No ISO3 column on Natural Earth countries.")
        world = world.rename(columns={iso_src: region_col})
        gdf = world.merge(df, on=region_col, how="inner")
        missing = set(df[region_col]) - set(gdf[region_col])
        if missing:
            print(f"  WARNING: no geometry for {sorted(missing)}")
    else:
        raise ValueError(
            f"Unsupported file type {ext!r}. Use .geojson, .gpkg, .shp, .parquet, or .csv."
        )

    if bbox is not None:
        gdf = gpd.clip(gdf, _box(*bbox))
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

    return gdf


def _grid_to_image(grid, args, figsize=None):
    """Render a grid's map WITHOUT title/subtitle/source and return
    ``(rgba_array, cell_px, size_in)``: the image, one grid cell's size in
    pixels of that image, and the figure size it was drawn at. The caller uses
    cell_px to bring both maps to a common block size so a block is identical
    across the pair, and size_in to re-render at the final printed scale."""
    import io
    import matplotlib.pyplot as plt

    fig = grid.plot(
        title="", subtitle="", data_source=" ", figsize=figsize,
        show_labels=not args.no_labels, label_min_cells=args.label_min,
        background_color=args.bg, scope=args.scope,
        background_style=args.background, show_region_borders=not args.no_region_borders,
        style=style_from(args), label_mode=args.label_mode,
    )
    size_in = tuple(fig.get_size_inches())
    bands_in = getattr(fig, "_popgrid_bands_in", 0.0)
    # Biggest axes is the main map; measure one cell's pixel span there.
    ax = max(fig.axes, key=lambda a: a.get_position().width * a.get_position().height)
    cs = grid.cell_size_km * 1000.0  # data units (metres)
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((cs, 0))
    cell_px_fig = abs(p1[0] - p0[0])
    cell_px = cell_px_fig * (args.dpi / fig.dpi)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=args.dpi, bbox_inches="tight",
                pad_inches=0.05, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf), cell_px, (size_in, bands_in)


def _rescale(img, factor):
    """Resize an RGBA float image by ``factor`` (LANCZOS), returning a float array."""
    if abs(factor - 1.0) < 1e-3:
        return img
    import numpy as np
    from PIL import Image
    h, w = img.shape[:2]
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype("uint8"))
    im = im.resize((max(1, round(w * factor)), max(1, round(h * factor))), Image.LANCZOS)
    return np.asarray(im).astype(float) / 255.0


def _pad_to(img, H, W, bg):
    """Centre ``img`` on an ``H×W`` canvas filled with background colour ``bg``."""
    import numpy as np
    h, w = img.shape[:2]
    ch = img.shape[2]
    out = np.zeros((H, W, ch), dtype=float)
    out[:, :, 0], out[:, :, 1], out[:, :, 2] = bg[0], bg[1], bg[2]
    if ch == 4:
        out[:, :, 3] = 1.0
    y, x = (H - h) // 2, (W - w) // 2
    out[y:y + h, x:x + w, :] = img
    return out


def _same_block_size(img_a, cpa, img_b, cpb, bg):
    """Bring two map images to one block size and one canvas size."""
    target = min(cpa, cpb)
    img_a = _rescale(img_a, target / cpa)
    img_b = _rescale(img_b, target / cpb)
    H = max(img_a.shape[0], img_b.shape[0])
    W = max(img_a.shape[1], img_b.shape[1])
    return _pad_to(img_a, H, W, bg), _pad_to(img_b, H, W, bg), target


def population_source(iso: str) -> str:
    """Short population credit: the curated table's source, else GHS-POP."""
    import csv
    from collections import Counter
    from popgrid import data as pgdata

    found: Counter = Counter()
    with open(pgdata._RESOURCES / "population.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["iso3"] == iso and (row.get("source") or "").strip():
                found[row["source"].strip()] += 1
    if found:
        return found.most_common(1)[0][0]
    return "GHS-POP 2025 (JRC, European Commission)"


def _text_width_in(text: str, **kw) -> float:
    """Rendered width of a string in inches."""
    import matplotlib.pyplot as plt
    f = plt.figure()
    t = f.text(0, 0, text, **kw)
    w = t.get_window_extent(f.canvas.get_renderer()).width / f.dpi
    plt.close(f)
    return w


# Layout of the side-by-side figure, in inches.
_SIDE = 0.7          # outer left/right margin
_GUTTER = 1.4        # space between the two columns, holds the arrow
_HEAD = 2.0          # title + subtitle band
_FOOT = 1.15         # credit + source band
_TEXT_PAD = 0.45     # minimum air on each side of a title or subtitle
_MAP_MAX_W = 10.0    # a wide country fills this width
_MAP_MAX_H = 13.0    # a tall country (Chile, Norway) fills this height
_MAX_RATIO = 2.0     # never wider than 2:1, so the post holds up in a feed

CREDIT = "Made with popgrid"


def compare_figure(iso, args):
    """
    Build an AreaGrid and a PopGrid of the same country and place them side by
    side, land area on the left and population on the right, at the SAME block
    size, so a block in one map equals a block in the other. Region colours are
    identical across both (stable-colour feature).

    The canvas adapts to the country's shape: a wide country gets a wide figure,
    a tall one (Chile) a near-square one where the maps run the full height.
    Both maps are drawn twice: once to learn their shape, then again at the
    exact size they are printed, so labels are fitted to the size people see.

    ``iso`` may be either an ISO3 string (bundled country) or a GeoDataFrame
    (custom dataset via ``--data``). When a GDF is passed, ``args.name`` is used
    in the panel titles.
    """
    import matplotlib.pyplot as plt
    import geopandas as gpd

    if isinstance(iso, gpd.GeoDataFrame):
        name = args.name or "Custom Region"
        ag = _build_from_gdf(AreaGrid, iso, args)
        pg = _build_from_gdf(PopGrid, iso, args)
        pop_src = None
    else:
        name = country_name(iso)
        ag = _build(AreaGrid, iso, args)
        pg = _build(PopGrid, iso, args)
        pop_src = population_source(iso)
    bg = tuple(int(args.bg.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))

    # ── Pass 1: natural render, to learn the shape of the pair ───────────────
    img_a, cpa, size_a = _grid_to_image(ag, args)
    img_p, cpp, size_p = _grid_to_image(pg, args)
    canvas_a, _, target = _same_block_size(img_a, cpa, img_p, cpp, bg)
    H, W = canvas_a.shape[:2]
    aspect = H / W

    # ── Layout ────────────────────────────────────────────────────────────────
    pct = round(100 / args.n, 2) if args.n else 0
    muted = "#5E7D6A"
    title_kw = dict(fontsize=30, fontweight="bold")
    sub_kw = dict(fontsize=19, fontstyle="italic")
    titles = (f"Land Area of {name}", f"Population of {name}")
    subs = (
        f"Every block = {pct}% of the national land area",
        f"Every block = {pct}% of the national population",
    )
    counts = (f"{ag.total_cells:,} blocks total", f"{pg.total_cells:,} blocks total")

    map_w = min(_MAP_MAX_W, _MAP_MAX_H / aspect)
    map_h = map_w * aspect
    widest = max([_text_width_in(t, **title_kw) for t in titles]
                 + [_text_width_in(s, **sub_kw) for s in subs])
    col_w = max(map_w, widest + 2 * _TEXT_PAD)
    fig_w = 2 * _SIDE + 2 * col_w + _GUTTER
    # A wide country (Switzerland) would make a banner wider than 2:1, which
    # feeds crop; pad it with air above and below the maps instead.
    body_h = max(map_h, fig_w / _MAX_RATIO - _HEAD - _FOOT)
    fig_h = _HEAD + body_h + _FOOT

    # ── Pass 2: redraw each map at the size it will be printed ───────────────
    cell_in = target * map_h / H          # one block, in final inches
    imgs = []
    for grid, cp, size in ((ag, cpa, size_a), (pg, cpp, size_p)):
        s = cell_in / (cp / args.dpi)
        (w_in, h_in), bands = size
        img, cp2, _ = _grid_to_image(
            grid, args, figsize=(w_in * s, (h_in - bands) * s + bands))
        imgs.append((img, cp2))
    canvas_a, canvas_p, target2 = _same_block_size(imgs[0][0], imgs[0][1],
                                                   imgs[1][0], imgs[1][1], bg)
    H2, W2 = canvas_a.shape[:2]
    # Printed size of the redrawn canvas; cropping can shift it a little, so
    # shrink to fit if it no longer sits inside its column.
    map_w, map_h = W2 * cell_in / target2, H2 * cell_in / target2
    fit = min(1.0, col_w / map_w, body_h / map_h)
    map_w, map_h = map_w * fit, map_h * fit

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=args.bg)

    def rect(x_in, y_in, w_in, h_in):
        return [x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h]

    map_y = _FOOT + (body_h - map_h) / 2
    centres = (_SIDE + col_w / 2, _SIDE + col_w + _GUTTER + col_w / 2)
    for cx, canvas in zip(centres, (canvas_a, canvas_p)):
        ax = fig.add_axes(rect(cx - map_w / 2, map_y, map_w, map_h))
        ax.imshow(canvas, interpolation="none")
        ax.axis("off")

    gx = _SIDE + col_w
    axM = fig.add_axes(rect(gx + 0.25, map_y + map_h / 2 - 0.3, _GUTTER - 0.5, 0.6))
    axM.axis("off")
    axM.set_xlim(0, 1)
    axM.set_ylim(0, 1)
    axM.annotate(
        "", xy=(1.0, 0.5), xytext=(0.0, 0.5),
        arrowprops=dict(arrowstyle="-|>", lw=3.5, color=FOREST, mutation_scale=30),
    )

    for cx, title, sub, count in zip(centres, titles, subs, counts):
        fig.text(cx / fig_w, (fig_h - 0.72) / fig_h, title, ha="center", va="center",
                 color=FOREST, **title_kw)
        fig.text(cx / fig_w, (fig_h - 1.28) / fig_h, sub, ha="center", va="center",
                 color=muted, **sub_kw)
        fig.text(cx / fig_w, (fig_h - 1.68) / fig_h, count, ha="center", va="center",
                 color=muted, fontsize=14)

    credit = CREDIT if args.credit is None else args.credit
    if args.source:
        source = args.source
    elif pop_src:
        source = f"Boundaries: Natural Earth  ·  Population: {pop_src}"
    else:
        source = "Boundaries: Natural Earth"
    x0 = _SIDE / fig_w
    if credit:
        fig.text(x0, 0.72 / fig_h, credit, ha="left", va="center",
                 fontsize=17, fontweight="bold", color=FOREST)
    fig.text(x0, (0.36 if credit else 0.55) / fig_h, source, ha="left", va="center",
             fontsize=14, color=muted)
    return fig, ag, pg


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate popgrid population or land-area maps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("countries", nargs="*",
                   help="ISO3 codes, ALL (curated census countries) or WORLD (every country).")
    p.add_argument("--mode", default="pop", choices=["pop", "area", "gdp", "compare"],
                   help="pop (population, default) | area (land area) | "
                        "compare (area + population side by side).")

    # Constructor options
    p.add_argument("--n", type=int, default=1000, help="Target block count.")
    p.add_argument("--cluster", type=float, default=200.0, metavar="KM",
                   help="Landmass cluster distance in km (default: 200).")
    p.add_argument("--palette", default="qual",
                   help="qual | databites | bold | #hex,#hex,… (default: qual).")
    p.add_argument("--dissolve", default="auto", metavar="COLUMN",
                   help="Dissolve admin-1 by: auto | region | none (default: auto).")

    # Data mode: build one map from a custom dataset (GeoJSON, shapefile,
    # geopackage, parquet, or CSV+iso3). Bypasses the bundled-country loop.
    p.add_argument("--data", default=None, metavar="PATH",
                   help="Custom dataset: .geojson/.gpkg/.shp/.parquet (with "
                        "geometry) or .csv (joined to Natural Earth countries "
                        "on iso3). Mutually exclusive with positional ISO3 codes.")
    p.add_argument("--region-col", default="iso3", metavar="COL",
                   help="Column to use as the region label (default: iso3).")
    p.add_argument("--weight-col", default="population", metavar="COL",
                   help="Column to weight PopGrid by (default: population).")
    p.add_argument("--bbox", default=None, metavar="LONMIN,LATMIN,LONMAX,LATMAX",
                   help="Optional clip box in lon/lat, e.g. -11,34,35,71.5 for Europe.")
    p.add_argument("--name", default=None, metavar="STR",
                   help="Display name for the region in data mode (e.g. 'Europe').")

    # Plot options
    p.add_argument("--scope", default="all", choices=["all", "islands", "mainland"],
                   help="Landmasses to show (default: all).")
    p.add_argument("--mainland", action="store_true",
                   help="Plot only the mainland (drop detached territories and "
                        "panels). Shorthand for --scope mainland.")
    p.add_argument("--no-panels", action="store_true",
                   help="Keep territories that sit inline near the mainland "
                        "(islands, enclaves like Ceuta/Melilla, Corsica, Sicily) "
                        "but drop the distant panel insets. Shorthand for "
                        "--scope islands.")
    p.add_argument("--no-labels", action="store_true",
                   help="Hide region labels (same as --label-mode none).")
    p.add_argument("--label-min", type=int, default=3, metavar="N",
                   help="Min cells for a region label (default: 3).")
    p.add_argument("--bg", default="#FAF6F0", metavar="HEX",
                   help="Background colour (default: #FAF6F0).")
    p.add_argument("--background", default="grid", choices=["grid", "solid", "none"],
                   help="Background style (default: grid).")
    p.add_argument("--exclude", nargs="+", default=None, metavar="REGION",
                   help="Region names to drop before building (they take no blocks). "
                        "Adds to the country's built-in exclusions.")
    p.add_argument("--engine", default="auto", choices=["auto", "dougenik", "diffusion"],
                   help="Cartogram engine for population maps: auto (rubber-sheet, "
                        "diffusion when it tears), dougenik, or diffusion (Gastner-Newman).")
    p.add_argument("--edge-color", default=None, metavar="HEX",
                   help="Region outline colour (default: black).")
    p.add_argument("--edge-width", type=float, default=None, metavar="PT",
                   help="Region outline width in points (default: auto).")
    p.add_argument("--grid-color", default=None, metavar="HEX",
                   help="Background grid colour (default: auto).")
    p.add_argument("--label-size", type=float, default=None, metavar="PT",
                   help="Fixed label size in points (default: fit each region).")
    p.add_argument("--credit", default=None, metavar="STR",
                   help="Bold credit line in compare mode (default: "
                        "'Made with popgrid'; pass '' to hide it).")
    p.add_argument("--label-mode", default="auto",
                   choices=["auto", "full", "short", "none"],
                   help="auto: fit, wrap or abbreviate per region (default); "
                        "full: never abbreviate; short: always abbreviate; none: no labels.")
    p.add_argument("--no-region-borders", action="store_true",
                   help="Disable region outlines (on by default).")
    p.add_argument("--source", default=None, metavar="STR",
                   help="Data source attribution shown bottom-left.")
    p.add_argument("--title", default=None, help="Custom title (single country).")
    p.add_argument("--subtitle", default=None, help="Custom subtitle.")

    # Output
    p.add_argument("--out", default="out", help="Output directory (default: out).")
    p.add_argument("--dpi", type=int, default=170, help="PNG resolution (default: 170).")
    p.add_argument("--show", action="store_true",
                   help="Display interactively instead of saving.")
    p.add_argument("--verbose", "-v", action="store_true", help="Show build progress.")

    args = p.parse_args()

    if args.verbose:
        logging.getLogger("popgrid").setLevel(logging.INFO)

    if args.mainland:
        args.scope = "mainland"
    elif args.no_panels:
        args.scope = "islands"

    palette = resolve_palette(args.palette)
    dissolve = resolve_dissolve(args.dissolve)

    if not args.show:
        os.makedirs(args.out, exist_ok=True)

    import matplotlib.pyplot as plt

    # Data mode: one figure from a custom GeoDataFrame, then exit.
    if args.data:
        if args.countries:
            print("WARNING: positional ISO3 codes are ignored in --data mode.")
        bbox = None
        if args.bbox:
            bbox = tuple(float(x) for x in args.bbox.split(","))
            if len(bbox) != 4:
                print("ERROR: --bbox must be LONMIN,LATMIN,LONMAX,LATMAX")
                return 1
        print(f"Loading {args.data}...")
        gdf = load_data_gdf(args.data, region_col=args.region_col, bbox=bbox)
        print(f"  {len(gdf)} regions after load/clip")
        stem = args.name.lower().replace(" ", "_") if args.name else "custom"

        if args.mode == "compare":
            fig, ag, pg = compare_figure(gdf, args)
            blocks = f"{ag.total_cells}/{pg.total_cells}"
        else:
            GridClass = PopGrid if args.mode == "pop" else AreaGrid
            grid = _build_from_gdf(GridClass, gdf, args)
            display = args.name or "Custom Region"
            title = args.title or (
                f"Population of {display}" if args.mode == "pop"
                else f"Land Area of {display}"
            )
            fig = grid.plot(
                title=title, subtitle=args.subtitle,
                show_labels=not args.no_labels, label_min_cells=args.label_min,
                background_color=args.bg, scope=args.scope,
                background_style=args.background,
                show_region_borders=not args.no_region_borders,
                data_source=args.source,
            )
            blocks = str(grid.total_cells)

        if args.show:
            print(f"  {stem}: {blocks} blocks (showing)")
            plt.show()
        else:
            path = os.path.join(args.out, f"{stem}_{args.mode}.png")
            fig.savefig(path, dpi=args.dpi,
                            bbox_inches=None if args.mode == "compare" else "tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  {stem}: {blocks} blocks → {path}")
        return 0

    codes = resolve_codes(args.countries)
    print(f"Generating {len(codes)} {args.mode} map(s) at n={args.n}"
          + ("" if args.show else f" → {args.out}/") + "\n")

    ok: list[str] = []
    failed: list[str] = []
    for iso in codes:
        try:
            if args.mode == "compare":
                fig, ag, pg = compare_figure(iso, args)
                blocks = f"{ag.total_cells}/{pg.total_cells}"
            else:
                kwargs = dict(
                    n=args.n, cluster_distance_km=args.cluster,
                    palette=palette, dissolve_by=dissolve, engine=args.engine,
                    exclude=args.exclude,
                )
                if args.mode == "area":
                    grid = AreaGrid(iso, **kwargs)
                else:
                    by = "gdp" if args.mode == "gdp" else "population"
                    grid = PopGrid(iso, by=by, **kwargs)
                    if by == "gdp" and gdp_quality(iso) != "reported":
                        print(f"  {iso}: note: regional GDP is a model estimate here "
                              "(no official regional accounts); it closely follows population.")
                grid.build()
                fig = grid.plot(
                    title=title_for(iso, args.mode, args.title),
                    subtitle=args.subtitle,
                    show_labels=not args.no_labels,
                    label_min_cells=args.label_min,
                    background_color=args.bg,
                    scope=args.scope,
                    background_style=args.background,
                    show_region_borders=not args.no_region_borders,
                    data_source=args.source,
                    style=style_from(args),
                    label_mode=args.label_mode,
                )
                blocks = str(grid.total_cells)

            if args.show:
                print(f"  {iso}: {blocks} blocks (showing)")
                plt.show()
            else:
                path = os.path.join(args.out, f"{iso}_{args.mode}.png")
                fig.savefig(path, dpi=args.dpi,
                            bbox_inches=None if args.mode == "compare" else "tight",
                            facecolor=fig.get_facecolor())
                plt.close(fig)
                print(f"  {iso}: {blocks} blocks → {path}")
            ok.append(iso)
        except Exception as e:  # noqa: BLE001
            print(f"  {iso}: FAILED — {type(e).__name__}: {e}")
            failed.append(iso)

    print(f"\nDone. {len(ok)} ok"
          + (f", {len(failed)} failed: {', '.join(failed)}" if failed else "."))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
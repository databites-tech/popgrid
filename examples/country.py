"""
country.py — visualise any country's population grid.

Usage
-----
    python examples/country.py <ISO3> [options]

Examples
--------
    python examples/country.py ESP
    python examples/country.py USA --scope mainland
    python examples/country.py FRA --n 500 --palette databites
    python examples/country.py DEU --save germany.png --dpi 300
    python examples/country.py NLD --no-labels
    python examples/country.py GBR --dissolve region --scope islands
    python examples/country.py JPN --n 2000 --save japan_hi.png

All AreaGrid parameters
-----------------------
AreaGrid(iso3, n, cluster_distance_km, palette, dissolve_by)

  iso3                  ISO 3166-1 alpha-3 code            e.g. 'ESP'
  --n          INT      Target cell count (default: 1000)  each = 1/n of land area
  --cluster    FLOAT    Landmass cluster distance in km     default: 200
                        Increase for countries with remote territories
  --palette    STR      Colour palette:
                          qual        qualitative, 20 distinct colours (default)
                          databites   DataBites green brand palette
                          bold        high-contrast saturated colours
                          #hex,#hex   comma-separated custom hex list
  --dissolve   STR      How to group Natural Earth admin-1 rows (default: auto):
                          auto        recommended setting per country (from RECOMMENDED_SETTINGS)
                          region      dissolve by NE parent-region field
                          None / omit raw admin-1 rows

.plot() / .save() parameters

  --scope      STR      Which landmasses to show:
                          all         main + nearby inline islands + distant panel insets (default)
                          islands     main + nearby inline islands only (no distant panels)
                          mainland    main landmass only
  --no-labels           Hide region name labels
  --label-min  INT      Min cells a region needs to receive a label (default: 3)
  --bg         HEX      Background colour (default: #FAF6F0  DataBites cream)
  --background STR      Background: grid (default) | solid | none
  --no-region-borders   Disable region boundary outlines (on by default)
  --source     STR      Data source attribution shown bottom-left
  --save       PATH     Save to file instead of showing interactively
                        Format is inferred from extension: .png .svg .pdf
  --dpi        INT      Resolution for raster output (default: 200)

Quick reference for dissolve_by per country
-------------------------------------------
Run:  python examples/inspect_country.py <ISO3>
Or check popgrid.RECOMMENDED_SETTINGS for pre-verified defaults.

Countries where dissolve_by='auto' gives a clean result:
  ESP → 19 CCAA      ITA → 20 regioni    FRA → 18 régions
  USA → 51 states    DEU → 16 Länder     BRA → 27 states
  NLD → 12 provinces JPN → 47 prefectures ... (24 countries in RECOMMENDED_SETTINGS)
"""

from __future__ import annotations

import argparse
import logging
import sys

import matplotlib.pyplot as plt

from popgrid import AreaGrid, RECOMMENDED_SETTINGS


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="country.py",
        description="Visualise any country as an equal-area population grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("iso3", metavar="ISO3",
                   help="ISO 3166-1 alpha-3 country code, e.g. ESP, DEU, USA")

    # AreaGrid() params
    p.add_argument("--n", type=int, default=1000,
                   help="Number of grid cells (default: 1000)")
    p.add_argument("--cluster", type=float, default=200.0, metavar="KM",
                   help="Landmass cluster distance in km (default: 200)")
    p.add_argument("--palette", default="qual",
                   help="Colour palette: qual | databites | bold | #hex,#hex,… (default: qual)")
    p.add_argument("--dissolve", default="auto", metavar="COLUMN",
                   help="Dissolve admin-1 by column: auto | region | None (default: auto)")

    # plot() params
    p.add_argument("--scope", default="all", choices=["all", "islands", "mainland"],
                   help="Landmasses to show: all | islands | mainland (default: all)")
    p.add_argument("--no-labels", action="store_true",
                   help="Hide region name labels")
    p.add_argument("--label-min", type=int, default=3, metavar="N",
                   help="Min cells for a region label (default: 3)")
    p.add_argument("--bg", default="#FAF6F0", metavar="HEX",
                   help="Background hex colour (default: #FAF6F0)")
    p.add_argument("--background", default="grid",
                   choices=["grid", "solid", "none"],
                   help="Background style: grid (default) | solid | none")
    p.add_argument("--no-region-borders", action="store_true",
                   help="Disable region boundary outlines (enabled by default)")
    p.add_argument("--source", default=None, metavar="STR",
                   help="Data source attribution shown bottom-left")

    # Output
    p.add_argument("--save", default=None, metavar="PATH",
                   help="Output file path (default: <ISO3>.png). Use 'show' to display interactively.")
    p.add_argument("--dpi", type=int, default=200,
                   help="Resolution for raster output (default: 200)")
    p.add_argument("--title", default=None,
                   help="Custom figure title (auto-generated if omitted)")
    p.add_argument("--subtitle", default=None,
                   help="Custom subtitle (auto-generated if omitted)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show build progress")

    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_palette(raw: str) -> str | list[str]:
    """Allow --palette '#hex,#hex,…' for custom colour lists."""
    if "," in raw:
        return [c.strip() for c in raw.split(",")]
    return raw


def resolve_dissolve(raw: str | None) -> str | None:
    if raw is None or raw.lower() in ("none", ""):
        return None
    return raw


def print_summary(pg: AreaGrid) -> None:
    iso = pg.country_iso3
    rec = RECOMMENDED_SETTINGS.get(iso, {})
    print()
    print(f"  Country   : {iso}")
    print(f"  Cells     : {pg.total_cells}  (target {pg.n})")
    print(f"  Cell size : {pg.cell_size_km:.1f} km")
    print(f"  Landmasses: {pg.n_landmasses}")
    print(f"  Regions   : {len(pg.regions)}")
    print(f"  Recommended dissolve_by: {rec.get('dissolve_by', 'None (not in table)')}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    palette = resolve_palette(args.palette)
    dissolve = resolve_dissolve(args.dissolve)

    pg = AreaGrid(
        country_iso3=args.iso3,
        n=args.n,
        cluster_distance_km=args.cluster,
        palette=palette,
        dissolve_by=dissolve,
    )

    # Eager build so we can print summary before plotting
    pg.build()
    print_summary(pg)

    fig = pg.plot(
        title=args.title,
        subtitle=args.subtitle,
        show_labels=not args.no_labels,
        label_min_cells=args.label_min,
        background_color=args.bg,
        scope=args.scope,
        background_style=args.background,
        show_region_borders=not args.no_region_borders,
        data_source=args.source,
    )

    save_path = args.save
    if save_path is None:
        save_path = f"{args.iso3.upper()}.png"

    if save_path.lower() == "show":
        plt.show()
    else:
        fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved → {save_path}")


if __name__ == "__main__":
    main()

"""
Build ``popgrid/resources/population_ghs.csv``: population of every Natural
Earth admin-1 unit on Earth, summed from the GHS-POP gridded population
raster (JRC Global Human Settlement Layer).

This is the table popgrid falls back to for countries that have no
hand-curated rows in ``population.csv``. It is regenerated, not edited.

Usage
-----
    python scripts/build_population_ghs.py \
        --raster path/to/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif

The raster (about 480 MB zipped) is not part of the repository. Download it
from the JRC open data server, GHS_POP_GLOBE_R2023A, epoch 2025, WGS84,
30 arc-second grid (about 1 km at the equator), unzip, and point ``--raster``
at the ``.tif``.

Method
------
For each admin-1 polygon, ``exactextract`` computes the area-weighted sum of
the raster cells it covers (partial cells count in proportion to the covered
fraction). Polygon and raster share the WGS84 lon/lat grid, so no
reprojection is needed. GHS-POP cells hold absolute population counts, so the
sum is the region's population estimate.

Units that share a name inside one country are summed under that name,
because popgrid groups cells by ``region_name``.

Output columns match ``population.csv`` (``iso3, region_name, population,
year, source``) so the loader reads both tables the same way, plus
``iso_3166_2`` for traceability.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from popgrid.data import _download_admin1  # noqa: E402

SOURCE = (
    "GHS-POP R2023A, epoch 2025, 30 arc-second grid (JRC, EC); "
    "zonal sum over Natural Earth 1:10m admin-1 polygons"
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raster", required=True, help="Path to the GHS-POP GeoTIFF.")
    p.add_argument(
        "--out",
        default=str(ROOT / "popgrid" / "resources" / "population_ghs.csv"),
        help="Output CSV (default: popgrid/resources/population_ghs.csv).",
    )
    p.add_argument("--year", type=int, default=2025, help="Epoch of the raster.")
    args = p.parse_args()

    from exactextract import exact_extract

    shp = _download_admin1()
    ne = gpd.read_file(shp)
    ne = ne[ne.geometry.notna()].copy()
    ne["geometry"] = ne.geometry.make_valid()
    ne["iso3"] = ne["adm0_a3"].str.strip().str.upper()
    ne["region_name"] = ne["name"].fillna("").astype(str).str.strip()
    ne = ne[ne["region_name"] != ""]
    print(f"{len(ne)} admin-1 units, {ne['iso3'].nunique()} countries", flush=True)

    print("Running zonal sums (a few minutes)…", flush=True)
    stats = exact_extract(
        args.raster,
        ne,
        ["sum"],
        include_cols=["iso3", "region_name", "iso_3166_2"],
        output="pandas",
        progress=True,
    )

    # Same name inside one country → one popgrid region: sum them.
    agg = (
        stats.groupby(["iso3", "region_name"], as_index=False)
        .agg(population=("sum", "sum"), iso_3166_2=("iso_3166_2", "first"))
        .sort_values(["iso3", "region_name"])
    )
    agg["population"] = agg["population"].fillna(0).round().astype("int64")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["iso3", "region_name", "population", "year", "source", "iso_3166_2"])
        for r in agg.itertuples(index=False):
            w.writerow([r.iso3, r.region_name, r.population, args.year, SOURCE, r.iso_3166_2 or ""])

    totals = agg.groupby("iso3")["population"].sum().sort_values(ascending=False)
    print(f"\nWrote {len(agg)} rows for {agg['iso3'].nunique()} countries → {out}")
    print("Largest country totals (millions):")
    for iso, v in totals.head(12).items():
        print(f"  {iso}  {v / 1e6:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

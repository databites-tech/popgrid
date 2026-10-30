"""
Build ``popgrid/resources/gdp.csv``: total GDP (PPP) of every Natural Earth
admin-1 unit on Earth, summed from the Kummu et al. gridded GDP raster.

This is the table ``PopGrid(..., by="gdp")`` reads. It is regenerated, not
edited.

Usage
-----
    python scripts/build_gdp.py \
        --raster path/to/rast_gdpTot_1990_2024_30arcsec.tif \
        --metadata path/to/adm1_metaData.csv

Data
----
Kummu, M. et al. (2025) "Downscaled gridded global dataset for gross domestic
product (GDP) per capita PPP over 1990-2022", Scientific Data, updated to
1990-2024 (v2026). CC BY 4.0. https://doi.org/10.5281/zenodo.16741980

Download from the Zenodo record:
  * ``rast_gdpTot_1990_2024_30arcsec.tif`` (about 1.4 GB): total GDP per
    cell, 30 arc-second grid (about 1 km at the equator), 2017 international
    USD at purchasing power parity, one band every five years plus 2024.
  * ``adm1_metaData.csv``: which countries have *reported* regional GDP.

Method
------
For each admin-1 polygon, ``exactextract`` computes the area-weighted sum of
the raster cells it covers (partial cells count in proportion to the covered
fraction). Polygon and raster share the WGS84 lon/lat grid. Units that share
a name inside one country are summed under that name, because popgrid groups
cells by ``region_name``.

The ``quality`` column matters when you present a GDP map. The source
dataset has official regional GDP for 89 countries ("reported"); everywhere
else it spreads the national GDP across regions with a model that closely
tracks population ("modelled"), so a GDP map of such a country mostly repeats
its population map. ``popgrid.data.gdp_quality(iso3)`` exposes the flag.

Output columns: ``iso3, region_name, gdp, year, quality, source, iso_3166_2``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from popgrid.data import _download_admin1  # noqa: E402

SOURCE = "Kummu et al. 2025 gridded GDP PPP v2026 (CC BY 4.0)"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raster", required=True, help="Path to the total-GDP GeoTIFF.")
    p.add_argument("--metadata", required=True, help="Path to adm1_metaData.csv.")
    p.add_argument("--year", type=int, default=2024, help="Band year to use (default 2024).")
    p.add_argument(
        "--out",
        default=str(ROOT / "popgrid" / "resources" / "gdp.csv"),
        help="Output CSV (default: popgrid/resources/gdp.csv).",
    )
    args = p.parse_args()

    import rasterio
    from exactextract import exact_extract

    with rasterio.open(args.raster) as r:
        bands = [i for i, d in enumerate(r.descriptions) if d and d.endswith(str(args.year))]
        if not bands:
            sys.exit(f"No band for year {args.year}; bands are {r.descriptions}")
        band = bands[0]

    ne = gpd.read_file(_download_admin1())
    ne = ne[ne.geometry.notna()].copy()
    ne["geometry"] = ne.geometry.make_valid()
    ne["iso3"] = ne["adm0_a3"].str.strip().str.upper()
    ne["region_name"] = ne["name"].fillna("").astype(str).str.strip()
    ne = ne[ne["region_name"] != ""]
    print(f"{len(ne)} admin-1 units, {ne['iso3'].nunique()} countries", flush=True)

    print("Running zonal sums (a few minutes)...", flush=True)
    stats = exact_extract(
        args.raster, ne, "sum", output="pandas",
        include_cols=["iso3", "region_name", "iso_3166_2"],
    )
    sum_cols = [c for c in stats.columns if c.endswith("sum")]
    stats["gdp"] = stats[sum_cols[band]] if len(sum_cols) > 1 else stats[sum_cols[0]]

    out = stats.groupby(["iso3", "region_name"], as_index=False).agg(
        gdp=("gdp", "sum"), iso_3166_2=("iso_3166_2", "first")
    )
    out["gdp"] = out["gdp"].fillna(0).round().astype("int64")

    meta = pd.read_csv(args.metadata)
    reported = set(meta.loc[meta["YearRanges"].notna() & (meta["YearRanges"].astype(str) != "NA"), "iso3"])
    out["quality"] = out["iso3"].map(lambda i: "reported" if i in reported else "modelled")
    out["year"] = args.year
    out["source"] = SOURCE
    out = out[["iso3", "region_name", "gdp", "year", "quality", "source", "iso_3166_2"]]
    out = out.sort_values(["iso3", "region_name"])
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows for {out['iso3'].nunique()} countries "
          f"({(out.groupby('iso3')['quality'].first() == 'reported').sum()} reported) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
inspect_country.py
------------------
Print the full Natural Earth admin-1 hierarchy for any country.

Usage:
    python inspect_country.py ESP
    python inspect_country.py DEU
    python inspect_country.py FRA

Shows:
  - Every admin-1 row (province / state / region)
  - Key hierarchy columns: codes, names (EN + local + ES/FR/DE), type,
    gadm_level, parent region, area, coordinates, Wikidata/GeoNames IDs
  - A summary of how many rows exist and which dissolve_by values work
"""

import sys
import geopandas as gpd
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

CACHE_PATH = Path.home() / ".popgrid" / "cache" / "ne_10m_admin_1_states_provinces.shp"

# The columns we care about — grouped by purpose
COLS = {
    "country":    ["adm0_a3", "iso_a2", "admin"],
    "codes":      ["adm1_code", "iso_3166_2", "code_hasc"],
    "names":      ["name", "name_en", "name_local", "name_es", "name_fr", "name_de"],
    "type":       ["type", "type_en", "gadm_level"],
    "hierarchy":  ["region", "region_cod", "region_sub", "sub_code"],
    "geography":  ["area_sqkm", "latitude", "longitude"],
    "ids":        ["wikidataid", "gn_id"],
}

ALL_COLS = [c for group in COLS.values() for c in group]

# ── Main ──────────────────────────────────────────────────────────────────────

def inspect(iso3: str) -> None:
    iso3 = iso3.strip().upper()

    if not CACHE_PATH.exists():
        print(
            f"Natural Earth shapefile not found at {CACHE_PATH}.\n"
            "Run any AreaGrid build first to trigger the download:\n\n"
            "    from popgrid import AreaGrid\n"
            "    AreaGrid('DEU').build()\n"
        )
        sys.exit(1)

    gdf = gpd.read_file(CACHE_PATH)

    # Filter country
    for col in ("adm0_a3", "iso_a3"):
        if col in gdf.columns:
            df = gdf[gdf[col].str.strip().str.upper() == iso3].copy()
            if not df.empty:
                break
    else:
        df = pd.DataFrame()

    if df.empty:
        available = sorted(gdf["adm0_a3"].dropna().unique())
        print(f"No rows found for '{iso3}'.")
        print(f"Sample valid codes: {available[:30]}")
        return

    # Keep only useful columns that exist in this shapefile
    present = [c for c in ALL_COLS if c in df.columns]
    df = df[present].reset_index(drop=True)

    # ── Print ──────────────────────────────────────────────────────────────────
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 28)

    print(f"\n{'═'*80}")
    print(f"  Natural Earth admin-1 data for: {iso3}  ({len(df)} rows)")
    print(f"{'═'*80}\n")

    # Print each section separately for readability
    sections = {
        "Identity & codes":   ["adm1_code", "iso_3166_2", "code_hasc", "adm0_a3"],
        "Names":              ["name", "name_en", "name_local", "name_es"],
        "Classification":     ["type_en", "gadm_level"],
        "Hierarchy (dissolve_by candidates)": ["region", "region_cod", "region_sub"],
        "Geography":          ["area_sqkm", "latitude", "longitude"],
        "External IDs":       ["wikidataid", "gn_id"],
    }

    for section_title, cols in sections.items():
        cols_present = [c for c in cols if c in df.columns]
        if not cols_present:
            continue
        print(f"── {section_title} {'─' * (56 - len(section_title))}")
        print(df[cols_present].to_string())
        print()

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"{'═'*80}")
    print("  SUMMARY")
    print(f"{'═'*80}")
    print(f"  Total rows           : {len(df)}")

    if "type_en" in df.columns:
        types = df["type_en"].value_counts()
        print(f"  type_en values       : {dict(types)}")

    if "gadm_level" in df.columns:
        levels = df["gadm_level"].value_counts().sort_index()
        print(f"  gadm_level values    : {dict(levels)}")

    print()
    print("  dissolve_by='region' →", end=" ")
    if "region" in df.columns:
        filled = df["region"].notna() & (df["region"].str.strip() != "")
        n_filled = filled.sum()
        n_groups = df[filled]["region"].nunique()
        if n_filled == 0:
            print("❌  column is empty — do NOT use dissolve_by for this country")
        elif n_filled < len(df):
            print(f"⚠️   {n_filled}/{len(df)} rows filled → {n_groups} parent groups")
            print("     (some rows would be excluded)")
        else:
            print(f"✅  all {len(df)} rows filled → {n_groups} parent groups")
            print(f"     Parent names: {sorted(df['region'].unique())}")
    else:
        print("❌  column not present")

    print()
    print("  Recommended AreaGrid usage:")
    if "region" in df.columns and df["region"].notna().all() and df["region"].str.strip().ne("").all():
        n_groups = df["region"].nunique()
        if n_groups < len(df):
            print(f"    AreaGrid('{iso3}', dissolve_by='region')  "
                f"# {len(df)} rows → {n_groups} regions")
        else:
            print(f"    AreaGrid('{iso3}')  # already at the right level ({len(df)} rows)")
    else:
        print(f"    AreaGrid('{iso3}')  # no useful parent grouping available")
    print()


if __name__ == "__main__":
    iso = sys.argv[1] if len(sys.argv) > 1 else "ESP"
    inspect(iso)

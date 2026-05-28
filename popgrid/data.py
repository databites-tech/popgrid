"""
Natural Earth admin-1 data: download once, cache in ~/.popgrid/cache/.

Admin level strategy
--------------------
Natural Earth's ne_10m_admin_1_states_provinces contains one row per
administrative unit.  The granularity differs per country:

  Country              NE rows   Right level?  Use dissolve_by?
  ─────────────────────────────────────────────────────────────
  Germany (DEU)           16     Bundesländer  No  (already right)
  France  (FRA)          101     Départements  Yes → 'region' → 18 Régions
  Italy   (ITA)          110     Province      Yes → 'region' → 20 Regioni
  Spain   (ESP)           52     Provincias    Yes → 'region' → 17+2 CCAA
  UK      (GBR)          232     Districts     No  ('region' gives English
                                                    sub-regions, not nations)
  USA     (USA)           51     States        No  ('region'=Census division)

The public API exposes a ``dissolve_by`` parameter that accepts:
  - ``None`` (default) — return raw admin-1 rows, no dissolve
  - ``'region'``       — dissolve by the NE 'region' parent field
  - any column name    — dissolve by that column (advanced use)

This is honest: the caller must know their country's data structure.
The module provides a KNOWN_DISSOLVE dict with pre-tested defaults.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
import requests
import shapely

from .exceptions import CountryNotFoundError, DataNotFoundError

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".popgrid" / "cache"

_NE_URLS: list[str] = [
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip",
    (
        "https://github.com/nvkelso/natural-earth-vector/raw/master/zips/"
        "ne_10m_admin_1_states_provinces.zip"
    ),
]
_NE_SHP = "ne_10m_admin_1_states_provinces.shp"

# ── Tested country → dissolve column mapping ──────────────────────────────────
# Only countries where dissolve_by='region' has been verified to give the
# correct, human-meaningful parent regions.
KNOWN_DISSOLVE: dict[str, str] = {
    "ESP": "region",  # 52 provinces → 17 CCAA + Ceuta + Melilla
    "ITA": "region",  # 110 province → 20 regioni
    "FRA": "region",  # 101 départements → 18 régions
    "PHL": "region",  # 118 provinces → 17 regions
}

# Countries where admin-1 is already the right display level.
# dissolve_by=None is correct — listed here for documentation.
ADMIN1_IS_CORRECT: list[str] = [
    "DEU",  # 16 Bundesländer
    "USA",  # 51 states (region = census division — too coarse)
    "BRA",  # 27 states
    "AUS",  # 8 states/territories
    "CAN",  # 13 provinces/territories
    "GBR",  # NOTE: 232 districts; 'region' gives English sub-regions not nations
    "MEX",  # 32 states
    "CHN",  # 34 provinces/regions
]

# ── Per-country recommended settings ─────────────────────────────────────────
# Use with ``dissolve_by='auto'`` or look up directly.
# Values are kwargs forwarded to ``load_country_regions`` and ``AreaGrid``.
#
# Verified against Natural Earth ne_10m_admin_1_states_provinces (May 2026).
# Run ``python examples/inspect_country.py <ISO3>`` to check any country.
RECOMMENDED_SETTINGS: dict[str, dict] = {
    # ── dissolve_by='region' gives the correct admin-1 parent level ──────────
    "ESP": {"dissolve_by": "region"},  # 52 provinces  → 19 CCAA
    "ITA": {"dissolve_by": "region"},  # 110 province  → 20 regioni
    "FRA": {"dissolve_by": "region"},  # 101 depts     → 18 régions
    "PHL": {"dissolve_by": "region"},  # 118 provinces → 17 regions

    # ── Already at the correct admin-1 level (no dissolve needed) ────────────
    "DEU": {"dissolve_by": None},   # 16 Bundesländer
    "USA": {"dissolve_by": None},   # 51 states  ← 'region' = census division (4 groups, wrong)
    "BRA": {"dissolve_by": None},   # 27 states
    "MEX": {"dissolve_by": None},   # 32 states
    "ARG": {"dissolve_by": None},   # 24 provinces
    "AUS": {"dissolve_by": None},   # 8  states/territories
    "CAN": {"dissolve_by": None},   # 13 provinces/territories
    "JPN": {"dissolve_by": None},   # 47 prefectures
    "CHN": {"dissolve_by": None},   # 32 provinces
    "NLD": {"dissolve_by": None},   # 12 provinces
    "POL": {"dissolve_by": None},   # 16 voivodeships
    "SWE": {"dissolve_by": None},   # 21 counties
    "NOR": {"dissolve_by": None},   # 15 regions
    "TUR": {"dissolve_by": None},   # 81 provinces
    "ZAF": {"dissolve_by": None},   # 9  provinces
    "KOR": {"dissolve_by": None},   # 17 regions
    "PRT": {"dissolve_by": None},   # 18 districts + 2 autonomous regions
    "CHE": {"dissolve_by": None},   # 26 cantons
    "BEL": {"dissolve_by": None},   # 11 provinces

    # ── Special cases ─────────────────────────────────────────────────────────
    # GBR: Natural Earth stores 232 UK districts. 'region' gives 16 English
    # sub-regions (not the 4 nations). No clean single-level option exists in
    # this dataset. Use dissolve_by=None (232 districts) and accept the detail.
    "GBR": {"dissolve_by": None},
}


# ── Region display labels ────────────────────────────────────────────────────
# Human-readable plural label for the administrative divisions shown.
# Used in the default figure title: "{Country} by {label} in a square-based grid"
# For dissolve_by='region' countries the label reflects the PARENT level.
REGION_LABELS: dict[str, str] = {
    "ESP": "Autonomous Communities",   # CCAA
    "ITA": "Regions",                  # dissolved from 110 province → 20 regioni
    "FRA": "Regions",                  # dissolved from 101 depts → 18 régions
    "PHL": "Regions",                  # dissolved from 118 provinces → 17 regions
    "DEU": "States",                   # 16 Bundesländer (NE type_en = State)
    "USA": "States",                   # 51 states
    "BRA": "States",                   # 27 states
    "MEX": "States",                   # 32 states
    "ARG": "Provinces",                # 24 provinces
    "AUS": "States & Territories",     # 8 states/territories
    "CAN": "Provinces & Territories",  # 13
    "JPN": "Prefectures",              # 47 prefectures
    "CHN": "Provinces",                # 34
    "NLD": "Provinces",                # 12
    "POL": "Voivodeships",             # 16
    "SWE": "Counties",                 # 21
    "NOR": "Counties",                 # 15
    "TUR": "Provinces",                # 81
    "ZAF": "Provinces",                # 9
    "KOR": "Regions",                  # 17 (mixed Metro Cities + Provinces)
    "PRT": "Districts",                # 18 + 2 autonomous regions
    "CHE": "Cantons",                  # 26
    "BEL": "Provinces",                # 11
    "GBR": "Districts",                # 232 NE districts (no clean higher level)
}


# ── Cache helpers ─────────────────────────────────────────────────────────────


def cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def clear_cache() -> None:
    import shutil
    shutil.rmtree(_CACHE_DIR, ignore_errors=True)
    logger.info("Cache cleared: %s", _CACHE_DIR)


# ── Download ──────────────────────────────────────────────────────────────────


def _download_admin1() -> Path:
    cdir = cache_dir()
    shp = cdir / _NE_SHP
    if shp.exists():
        logger.debug("Using cached shapefile: %s", shp)
        return shp

    last_exc: Exception | None = None
    for url in _NE_URLS:
        try:
            logger.info("Downloading Natural Earth admin-1 from %s …", url)
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()

            zip_path = cdir / "ne_admin1_tmp.zip"
            with open(zip_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65_536):
                    fh.write(chunk)

            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(cdir)
            zip_path.unlink(missing_ok=True)

            if shp.exists():
                logger.info("Cached at %s", shp)
                return shp
            raise FileNotFoundError(f"{_NE_SHP} not found after extraction.")

        except Exception as exc:
            last_exc = exc
            logger.warning("Download failed from %s: %s", url, exc)

    raise DataNotFoundError(
        f"Could not download Natural Earth admin-1 shapefile. "
        f"Last error: {last_exc}"
    )


# ── Core loader ───────────────────────────────────────────────────────────────


def load_country_regions(
    country_iso3: str,
    dissolve_by: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Load admin-1 regions for a country from Natural Earth.

    Parameters
    ----------
    country_iso3:
        ISO 3166-1 alpha-3 code, e.g. ``'ESP'``, ``'DEU'``, ``'GBR'``.
    dissolve_by:
        Column name to dissolve sub-units into parent regions.

        - ``None`` (default) — return raw Natural Earth admin-1 rows.
        - ``'auto'``  — look up ``RECOMMENDED_SETTINGS[iso3]['dissolve_by']``
          and apply it.  Falls back to ``None`` if the country is not listed.
        - ``'region'`` — dissolve by the NE parent-region field.
          **Verified** for: ESP, ITA, FRA, PHL.
          Use ``popgrid.data.KNOWN_DISSOLVE`` to see the tested mapping.

        If the column is absent or entirely empty for this country, a
        warning is issued and raw admin-1 rows are returned.

    Returns
    -------
    GeoDataFrame (WGS84) with columns:
        ``region_id``, ``region_name``, ``geometry``
    """
    shp_path = _download_admin1()
    all_regions: gpd.GeoDataFrame = gpd.read_file(shp_path)

    iso = country_iso3.strip().upper()

    # Resolve 'auto' → look up RECOMMENDED_SETTINGS
    if dissolve_by == "auto":
        dissolve_by = RECOMMENDED_SETTINGS.get(iso, {}).get("dissolve_by", None)
        logger.debug("dissolve_by='auto' resolved to %r for %s", dissolve_by, iso)

    # Filter by country
    regions: gpd.GeoDataFrame = gpd.GeoDataFrame()
    for col in ("adm0_a3", "iso_a3"):
        if col in all_regions.columns:
            mask = all_regions[col].str.strip().str.upper() == iso
            regions = all_regions[mask].copy()
            if not regions.empty:
                break

    if regions.empty:
        sample = sorted(
            all_regions.get("adm0_a3", all_regions.index).dropna().unique()
        )[:30]
        raise CountryNotFoundError(
            f"No regions found for ISO3 code '{iso}'. "
            f"Sample of available codes: {sample}"
        )

    # Repair geometries
    valid_mask = regions["geometry"].notna()
    regions = regions[valid_mask].copy()
    regions["geometry"] = shapely.make_valid(regions["geometry"].values)

    # ── Optional dissolve ─────────────────────────────────────────────────────
    if dissolve_by is not None:
        col = dissolve_by
        if col not in regions.columns:
            logger.warning(
                "dissolve_by='%s' column not found for %s — "
                "returning raw admin-1 rows. Available columns: %s",
                col, iso, sorted(regions.columns.tolist()),
            )
        else:
            filled = regions[col].notna() & (regions[col].str.strip() != "")
            n_filled = filled.sum()

            if n_filled == 0:
                logger.warning(
                    "dissolve_by='%s' is empty for all %d rows of %s — "
                    "returning raw admin-1 rows.",
                    col, len(regions), iso,
                )
            else:
                if n_filled < len(regions):
                    logger.warning(
                        "%d of %d rows have empty '%s' for %s — "
                        "those rows are excluded from the dissolve.",
                        len(regions) - n_filled, len(regions), col, iso,
                    )

                n_before = len(regions)
                regions = regions[filled].copy()
                regions[col] = regions[col].str.strip()

                dissolved = (
                    regions[[col, "geometry"]]
                    .dissolve(by=col, as_index=False)
                    .rename(columns={col: "region_name"})
                )
                dissolved["geometry"] = shapely.make_valid(dissolved["geometry"].values)
                dissolved = dissolved.reset_index(drop=True)
                dissolved["region_id"] = range(len(dissolved))
                regions = dissolved[["region_id", "region_name", "geometry"]]

                logger.info(
                    "Dissolved %d admin-1 units → %d '%s' parent regions for %s",
                    n_before, len(regions), col, iso,
                )
                return regions

    # ── Standard path ─────────────────────────────────────────────────────────
    regions = regions.rename(columns={"name": "region_name"})
    regions["region_id"] = range(len(regions))
    keep = ["region_id", "region_name", "geometry"]
    regions = regions[keep].reset_index(drop=True)

    logger.info("Loaded %d admin-1 regions for %s", len(regions), iso)
    return regions

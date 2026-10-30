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
    # 'exclude' lists uninhabited specks that would otherwise earn one land
    # block and zero population blocks: noise on the land side, absent on the
    # population side. Inhabited or large territories (Svalbard, Ceuta) stay.
    "ESP": {"dissolve_by": "region",   # 52 provinces  → 19 CCAA
            "display_names": {"Foral de Navarra": "Navarra", "Canary Is.": "Canary Islands",
                              "Islas Baleares": "Baleares"}},
    "ITA": {"dissolve_by": "region"},  # 110 province  → 20 regioni
    "FRA": {"dissolve_by": "region"},  # 101 depts     → 18 régions
    "PHL": {"dissolve_by": "region"},  # 118 provinces → 17 regions

    # ── Already at the correct admin-1 level (no dissolve needed) ────────────
    "DEU": {"dissolve_by": None},   # 16 Bundesländer
    # 51 states ('region' = census division, 4 groups, wrong). Alaska's
    # panhandle sits close enough to Washington to pass the inline test, so
    # it is forced into a panel together with Hawaii.
    "USA": {"dissolve_by": None, "panel_regions": ["Alaska", "Hawaii"]},
    "BRA": {"dissolve_by": None},   # 27 states
    "MEX": {"dissolve_by": None},   # 32 states
    "ARG": {"dissolve_by": None},   # 24 provinces
    "AUS": {"dissolve_by": None,    # 8  states/territories
            "exclude": ["Macquarie Island", "Lord Howe Island"]},
    "CAN": {"dissolve_by": None},   # 13 provinces/territories
    "JPN": {"dissolve_by": "region", "cluster_distance_km": 100},  # 9 regions; 100km peels Okinawa into its own panel
    "CHN": {"dissolve_by": None, "exclude": ["Paracel Islands"]},   # 32 provinces
    "NLD": {"dissolve_by": None},   # 12 provinces
    "POL": {"dissolve_by": None},   # 16 voivodeships
    "SWE": {"dissolve_by": None},   # 21 counties
    "NOR": {"dissolve_by": None, "exclude": ["Bouvet Island"],
            "panel_regions": ["Svalbard"]},   # 15 regions; Bjørnøya would float inline
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
    # GBR: Natural Earth stores 232 UK districts — far too granular. The
    # 'region' field groups them into 16 sub-regions (English regions + Welsh,
    # Scottish, NI groupings). Coarser and far more legible than 232 districts.
    "GBR": {"dissolve_by": "region"},

    # ── Countries served by the GHS-POP fallback table ───────────────────────
    # Natural Earth is far more granular than the level people recognise;
    # its 'region' field gives the sensible parent level.
    "SVN": {"dissolve_by": "region"},   # 193 municipalities → 12 statistical regions
    "LVA": {"dissolve_by": "region"},   # 119 municipalities → 5 planning regions
    "UGA": {"dissolve_by": "region"},   # 112 districts      → 4 regions
    "MLT": {"dissolve_by": "region"},   # 68 local councils  → 3 regions
    "AZE": {"dissolve_by": "region"},   # 78 districts       → 10 economic regions
    "THA": {"dissolve_by": "region"},   # 77 provinces       → 6 regions
    # RUS (86 federal subjects) and VNM (63 provinces) stay at admin-1, as TUR does.
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


def admin1_parent_map(country_iso3: str, parent_col: str = "region") -> dict[str, str]:
    """
    Return ``{admin1_name: parent_region}`` for a country, read straight from
    Natural Earth.  Used to aggregate fine-grained populations up to the coarse
    regions produced by ``load_country_regions(dissolve_by=parent_col)``.
    Names with an empty parent map to themselves.
    """
    shp_path = _download_admin1()
    all_regions = gpd.read_file(shp_path)
    iso = country_iso3.strip().upper()
    regions = gpd.GeoDataFrame()
    for col in ("adm0_a3", "iso_a3"):
        if col in all_regions.columns:
            m = all_regions[col].str.strip().str.upper() == iso
            regions = all_regions[m]
            if not regions.empty:
                break
    out: dict[str, str] = {}
    if regions.empty or "name" not in regions.columns or parent_col not in regions.columns:
        return out
    for _, r in regions.iterrows():
        name = str(r["name"]).strip() if r["name"] is not None else ""
        parent = r[parent_col]
        parent = str(parent).strip() if parent is not None else ""
        if name:
            out[name] = parent if parent else name
    return out


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
    # Natural Earth has a handful of nameless islets (a few km² each, e.g. one
    # off Russia, one off Mexico). They have no population row and would only
    # ever get a fallback weight, so they are dropped.
    named = regions["region_name"].notna() & (regions["region_name"].astype(str).str.strip() != "")
    if (~named).any():
        logger.debug("Dropping %d nameless admin-1 unit(s) for %s", int((~named).sum()), iso)
        regions = regions[named].copy()
    regions["region_id"] = range(len(regions))
    keep = ["region_id", "region_name", "geometry"]
    regions = regions[keep].reset_index(drop=True)

    logger.info("Loaded %d admin-1 regions for %s", len(regions), iso)
    return regions


_RESOURCES = Path(__file__).parent / "resources"
#: Hand-curated census figures for a few countries, at the level popgrid draws.
_CURATED_CSV = _RESOURCES / "population.csv"
#: Every Natural Earth admin-1 unit on Earth, summed from the GHS-POP 2025
#: raster. Regenerate with ``scripts/build_population_ghs.py``; never edit.
_GHS_CSV = _RESOURCES / "population_ghs.csv"


#: Total GDP (PPP, 2017 international USD) of every Natural Earth admin-1 unit,
#: summed from the Kummu et al. gridded GDP raster. Regenerate with
#: ``scripts/build_gdp.py``; never edit.
_GDP_CSV = _RESOURCES / "gdp.csv"

#: Variables :class:`PopGrid` can size regions by (its ``by=`` argument),
#: mapped to the label used in subtitles.
VARIABLES: dict[str, str] = {
    "population": "population",
    "gdp": "GDP",
}


def _read_population_rows(
    csv_path: Path, iso3: str, value_col: str = "population"
) -> dict[str, int]:
    """``{region_name: value}`` for one country from one table."""
    import csv as _csv

    out: dict[str, int] = {}
    if not csv_path.exists():
        logger.warning("data table not found at %s", csv_path)
        return out
    iso = iso3.strip().upper()
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            if (row.get("iso3") or "").strip().upper() != iso:
                continue
            name = (row.get("region_name") or "").strip()
            try:
                pop = int(float(row[value_col]))
            except (ValueError, KeyError, TypeError):
                pop = 0
            if name:
                out[name] = out.get(name, 0) + pop
    return out


def _table_iso3s(csv_path: Path, min_regions: int = 1) -> list[str]:
    import csv as _csv
    from collections import Counter

    if not csv_path.exists():
        return []
    counts: Counter = Counter()
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            iso = (row.get("iso3") or "").strip().upper()
            if iso:
                counts[iso] += 1
    return sorted(iso for iso, n in counts.items() if n >= min_regions)


def bundled_countries() -> list[str]:
    """ISO3 codes with hand-curated census populations (``population.csv``)."""
    return _table_iso3s(_CURATED_CSV)


def supported_countries(min_regions: int = 3) -> list[str]:
    """
    Every ISO3 code :class:`PopGrid` can draw: the curated set plus all
    countries in the GHS-POP table with at least ``min_regions`` admin-1
    units (fewer than that makes no map worth looking at).
    """
    return sorted(set(bundled_countries()) | set(_table_iso3s(_GHS_CSV, min_regions)))


def load_region_populations(
    iso3: str,
    region_names: list[str],
    name_to_parent: Optional[dict[str, str]] = None,
) -> dict[str, int]:
    """
    Return ``{region_name: population}`` for the given country.

    Reads the hand-curated ``resources/population.csv`` first. Countries
    without curated rows fall back to ``resources/population_ghs.csv``,
    GHS-POP 2025 estimates summed over every Natural Earth admin-1 unit,
    which covers every country. For regions found in neither, the mean of
    known values is used so the total stays roughly proportional.

    Parameters
    ----------
    iso3:
        ISO 3166-1 alpha-3 country code.
    region_names:
        Exact region name strings as they appear in Natural Earth
        (after any dissolve).  Used as lookup keys.

    Returns
    -------
    dict mapping region_name → integer population estimate.
    """
    country_pops = _read_population_rows(_CURATED_CSV, iso3)
    ghs_pops = _read_population_rows(_GHS_CSV, iso3)
    if not country_pops:
        country_pops = ghs_pops
        if country_pops:
            logger.info(
                "%s: no curated population rows; using GHS-POP 2025 estimates", iso3
            )

    result: dict[str, int] = {}
    missing: list[str] = []

    # If a fine→parent name map is supplied (because geometries were dissolved
    # to a coarser level), aggregate the fine-grained populations up to the
    # parent regions before looking up the requested coarse names.
    if name_to_parent:
        agg: dict[str, int] = {}
        for fine_name, pop in country_pops.items():
            parent = name_to_parent.get(fine_name, fine_name)
            agg[parent] = agg.get(parent, 0) + pop
        country_pops = agg

    if name_to_parent and ghs_pops is not country_pops:
        agg_ghs: dict[str, int] = {}
        for fine_name, pop in ghs_pops.items():
            parent = name_to_parent.get(fine_name, fine_name)
            agg_ghs[parent] = agg_ghs.get(parent, 0) + pop
        ghs_pops = agg_ghs

    for name in region_names:
        if name in country_pops and country_pops[name] > 0:
            result[name] = country_pops[name]
        elif name in ghs_pops:
            # A region the curated table skipped (Macquarie Island, Bouvet
            # Island): take the GHS estimate, which is 0 for uninhabited
            # land, so the region gets no blocks instead of the country mean.
            result[name] = ghs_pops[name]
            logger.info("%s: %r not in curated table; GHS estimate %d",
                        iso3, name, ghs_pops[name])
        else:
            missing.append(name)

    # Fallback: mean of known values so missing regions are not invisible
    if missing:
        fallback = max(1, sum(result.values()) // len(result)) if result else 1
        for name in missing:
            result[name] = fallback
            logger.warning(
                "No population data for %r (%s) — using fallback weight %d",
                name, iso3, fallback,
            )

    return result


def load_region_gdp(
    iso3: str,
    region_names: list[str],
    name_to_parent: Optional[dict[str, str]] = None,
) -> dict[str, int]:
    """
    Return ``{region_name: GDP}`` (PPP, 2017 international USD, 2024).

    Reads ``resources/gdp.csv``. A region missing from the table (a name the
    table spells differently) gets the country's GDP per person times its
    population, so it is sized like an average region rather than dropped.

    Check :func:`gdp_quality` before presenting a GDP map: only countries
    marked ``"reported"`` have official regional GDP behind them.
    """
    gdp = _read_population_rows(_GDP_CSV, iso3, value_col="gdp")
    if not gdp:
        raise DataNotFoundError(f"No GDP data for {iso3!r} in {_GDP_CSV.name}.")
    if name_to_parent:
        agg: dict[str, int] = {}
        for fine, v in gdp.items():
            parent = name_to_parent.get(fine, fine)
            agg[parent] = agg.get(parent, 0) + v
        gdp = agg

    result = {name: gdp[name] for name in region_names if name in gdp}
    missing = [name for name in region_names if name not in gdp]
    if missing:
        pops = load_region_populations(iso3, region_names, name_to_parent=name_to_parent)
        known_pop = sum(pops.get(n, 0) for n in result) or 1
        per_person = sum(result.values()) / known_pop
        for name in missing:
            result[name] = int(per_person * pops.get(name, 0))
            logger.warning(
                "No GDP data for %r (%s); using national GDP per person x population",
                name, iso3,
            )
    return result


def gdp_quality(iso3: str) -> Optional[str]:
    """
    ``"reported"`` when the country's regional GDP comes from official
    regional statistics, ``"modelled"`` when only national GDP is reported
    and the regional split is a model estimate (closely tracks population),
    ``None`` when the country is not in the table.
    """
    import csv as _csv

    iso = iso3.strip().upper()
    with open(_GDP_CSV, newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            if row["iso3"] == iso:
                return row["quality"]
    return None


def load_region_values(
    iso3: str,
    region_names: list[str],
    by: str = "population",
    name_to_parent: Optional[dict[str, str]] = None,
) -> dict[str, int]:
    """``{region_name: value}`` for the variable ``by`` (see :data:`VARIABLES`)."""
    if by == "population":
        return load_region_populations(iso3, region_names, name_to_parent=name_to_parent)
    if by == "gdp":
        return load_region_gdp(iso3, region_names, name_to_parent=name_to_parent)
    raise ValueError(f"by must be one of {tuple(VARIABLES)}; got {by!r}")

"""
Tests for popgrid.data — region loading and dissolve logic.
"""

import pytest
import geopandas as gpd

from popgrid.data import load_country_regions, KNOWN_DISSOLVE
from popgrid.exceptions import CountryNotFoundError


# ── KNOWN_DISSOLVE ────────────────────────────────────────────────────────────

def test_known_dissolve_is_dict():
    assert isinstance(KNOWN_DISSOLVE, dict)
    assert len(KNOWN_DISSOLVE) > 0


def test_known_dissolve_keys_are_iso3():
    for key in KNOWN_DISSOLVE:
        assert len(key) == 3, f"Expected ISO3 key, got '{key}'"
        assert key.isupper(), f"Expected uppercase ISO3 key, got '{key}'"


# ── load_country_regions — basic ──────────────────────────────────────────────

def test_load_germany_returns_geodataframe():
    gdf = load_country_regions("DEU")
    assert isinstance(gdf, gpd.GeoDataFrame)


def test_load_germany_correct_columns():
    gdf = load_country_regions("DEU")
    assert "region_id" in gdf.columns
    assert "region_name" in gdf.columns
    assert "geometry" in gdf.columns


def test_load_germany_crs_is_wgs84():
    gdf = load_country_regions("DEU")
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 4326


def test_load_germany_row_count():
    gdf = load_country_regions("DEU")
    # Natural Earth has exactly 16 Bundesländer for Germany
    assert len(gdf) == 16


def test_load_germany_no_null_geometries():
    gdf = load_country_regions("DEU")
    assert gdf["geometry"].notna().all()


def test_load_germany_no_empty_geometries():
    gdf = load_country_regions("DEU")
    assert gdf["geometry"].apply(lambda g: not g.is_empty).all()


def test_load_case_insensitive():
    gdf_upper = load_country_regions("DEU")
    gdf_lower = load_country_regions("deu")
    assert len(gdf_upper) == len(gdf_lower)


def test_load_invalid_iso_raises():
    with pytest.raises(CountryNotFoundError):
        load_country_regions("ZZZ")


# ── load_country_regions — dissolve ──────────────────────────────────────────

def test_spain_raw_has_52_rows():
    gdf = load_country_regions("ESP")
    assert len(gdf) == 52


def test_spain_dissolve_region_gives_19_ccaa():
    gdf = load_country_regions("ESP", dissolve_by="region")
    # 17 CCAA + Ceuta + Melilla
    assert len(gdf) == 19


def test_spain_dissolve_includes_ceuta_melilla():
    gdf = load_country_regions("ESP", dissolve_by="region")
    names = gdf["region_name"].tolist()
    assert "Ceuta" in names
    assert "Melilla" in names


def test_italy_dissolve_gives_20_regioni():
    gdf = load_country_regions("ITA", dissolve_by="region")
    assert len(gdf) == 20


def test_france_dissolve_gives_18_regions():
    gdf = load_country_regions("FRA", dissolve_by="region")
    assert len(gdf) == 18


def test_dissolve_result_has_correct_columns():
    gdf = load_country_regions("ESP", dissolve_by="region")
    assert set(gdf.columns) == {"region_id", "region_name", "geometry"}


def test_dissolve_result_no_null_geometries():
    gdf = load_country_regions("ESP", dissolve_by="region")
    assert gdf["geometry"].notna().all()


def test_dissolve_nonexistent_column_falls_back_gracefully():
    # Should warn but return raw rows, not raise
    gdf = load_country_regions("DEU", dissolve_by="nonexistent_col")
    assert len(gdf) == 16  # Falls back to raw admin-1


def test_dissolve_empty_column_falls_back_gracefully():
    # Germany has no 'region' values — should warn and return raw rows
    gdf_raw = load_country_regions("DEU")
    gdf_dissolve = load_country_regions("DEU", dissolve_by="region")
    assert len(gdf_raw) == len(gdf_dissolve)


# ── Rotated Natural Earth names (issue #10) ───────────────────────────────────

from popgrid.data import _rotation_fixes, admin1_name_fixes


def test_rotation_swap_is_fixed():
    fixes = _rotation_fixes(["Napo", "Tungurahua", "Pichincha"],
                            ["Provincia del Tungurahua", "Provincia de Napo", "Pichincha"])
    assert fixes == {"Napo": "Tungurahua", "Tungurahua": "Napo"}


def test_rotation_closed_by_elimination():
    # The last link is a translated name that matches no polygon.
    names = ["Semenawi Keyih Bahri", "Anseba", "Maekel", "Debub", "Gash Barka"]
    gn = ["Anseba Region", "Maekel Region", "Debub Region",
          "Northern Red Sea Region", "Gash-Barka Region"]
    assert _rotation_fixes(names, gn) == {
        "Semenawi Keyih Bahri": "Anseba", "Anseba": "Maekel",
        "Maekel": "Debub", "Debub": "Semenawi Keyih Bahri",
    }


def test_city_pointing_to_its_district_is_not_a_rotation():
    names = ["Francistown", "North-East", "Selebi-Phikwe", "Sowa", "Central"]
    gn = ["North East District", "North East District", "Central District",
          "Central District", "Central District"]
    assert _rotation_fixes(names, gn) == {}


def test_single_name_coincidence_is_not_a_rotation():
    # Bermuda: the city's GeoNames name is also the name of another parish.
    assert _rotation_fixes(["City of Hamilton", "Hamilton", "Pembroke"],
                           ["Hamilton", "Hamilton Parish", "Pembroke Parish"]) == {}


def test_bundled_countries_with_rotated_names():
    fixed = {iso for iso in ("ECU", "GUY", "ERI", "BMU", "ESP", "DEU", "BWA", "TWN")
             if admin1_name_fixes(iso)}
    assert fixed == {"ECU", "GUY", "ERI"}


def test_ecuador_napo_is_the_large_eastern_province():
    gdf = load_country_regions("ECU").to_crs(6933)
    area = gdf.set_index("region_name").geometry.area / 1e6
    assert area["Napo"] > 10_000 > area["Tungurahua"]


# ── Other names for the same region (issue #17) ───────────────────────────────

from popgrid.data import _match_names, match_region_names, region_aliases


def test_current_names_find_superseded_regions():
    assert match_region_names("MEX", ["Ciudad de México"]) == {"Ciudad de México": "Distrito Federal"}
    assert match_region_names("VEN", ["La Guaira"]) == {"La Guaira": "Vargas"}
    assert match_region_names("NIC", ["Costa Caribe Norte", "Costa Caribe Sur"]) == {
        "Costa Caribe Norte": "Atlántico Norte", "Costa Caribe Sur": "Atlántico Sur"}


def test_matching_ignores_accents_case_and_generic_words():
    out = _match_names(["nuevo leon", "Province of Napo", "Unknownia"],
                       ["Nuevo León", "Napo"], {})
    assert out == {"nuevo leon": "Nuevo León", "Province of Napo": "Napo", "Unknownia": None}


def test_ambiguous_names_are_not_guessed():
    assert _match_names(["Central"], ["Central Region", "Central District"], {}) == {"Central": None}


def test_alias_never_shadows_another_region_name():
    aliases = region_aliases("ECU")
    assert aliases.get("Napo") in (None, "Napo")
    assert aliases.get("Tungurahua") in (None, "Tungurahua")


def test_exclude_and_color_map_accept_official_names():
    from popgrid import AreaGrid

    ag = AreaGrid("MEX", exclude=["Ciudad de México"], color_map={"Nuevo Leon": "#ff0000"})
    ag.build()
    assert "Distrito Federal" not in ag.block_counts
    assert ag.color_map["Nuevo León"] == "#ff0000"

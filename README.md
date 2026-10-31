# popgrid

**Tiled grid maps for Python: land area, population and GDP, clearly explained.**

[![PyPI](https://img.shields.io/pypi/v/popgrid.svg)](https://pypi.org/project/popgrid/)
[![Python](https://img.shields.io/pypi/pyversions/popgrid.svg)](https://pypi.org/project/popgrid/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

popgrid turns a country (or any set of regions) into a block cartogram where **every block is an equal share of the whole**: 0.1% of the national land area, or 0.1% of the population. Same regions, same colours, two ways of seeing them. Where land and people diverge is the whole point.

<p align="center">
  <img src="https://raw.githubusercontent.com/databites-tech/popgrid/main/docs/img/spain_compare.png?v=0.7.1" width="100%" alt="Land area vs population of Spain, side by side">
</p>

Madrid is a sliver of land and a giant of population. The empty interior shrinks; the coasts and the capital swell. A block on the left is the same size as a block on the right, so the comparison is literally true.

---

## Install

```bash
pip install popgrid
```

Requires Python 3.10+. Country boundaries are downloaded once from Natural Earth and cached locally; population data for the bundled countries ships with the package.

---

## Quickstart

```python
from popgrid import PopGrid, AreaGrid

# Population: every block = 0.1% of the national population
PopGrid("ESP").plot().savefig("spain_population.png", dpi=170, bbox_inches="tight")

# Land area: every block = 0.1% of the national land area
AreaGrid("ESP").plot().savefig("spain_area.png", dpi=170, bbox_inches="tight")
```

| Population | Land area |
|---|---|
| ![Population of Spain](https://raw.githubusercontent.com/databites-tech/popgrid/main/docs/img/spain_pop.png?v=0.7.1) | ![Land area of Spain](https://raw.githubusercontent.com/databites-tech/popgrid/main/docs/img/spain_area.png?v=0.7.1) |

`.plot()` builds the grid on first call and returns a Matplotlib figure, so you can save, show, or restyle it however you like.

### GDP

`by="gdp"` sizes regions by GDP (purchasing power parity, 2024) instead of people. Same regions, same colours as the other two maps, so you can read all three side by side.

```python
PopGrid("ESP", by="gdp").plot(title="Economy of Spain")
```

Colours are assigned once per country from real borders, so a region that balloons in one layout can end up touching a region it never borders in reality, sometimes one of the same colour (Madrid and Andalucía are both yellow by default). `color_mode="layouts"` chooses colours with every layout in mind: regions two borders apart stay distinct and a region is weighted by its largest share of land, people or GDP. Pass it to all three maps of a country and they share one colour map, as below. The default stays the colouring of earlier versions, so existing maps do not change.

```python
for g in (AreaGrid("ESP", color_mode="layouts"),
          PopGrid("ESP", color_mode="layouts"),
          PopGrid("ESP", by="gdp", color_mode="layouts")):
    g.plot()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/databites-tech/popgrid/main/docs/img/spain_land_people_gdp.png?v=0.7.1" width="100%" alt="Spain by land, population and GDP">
</p>

Madrid is 1.6% of the land, 14% of the people and 22% of the money. Regional GDP comes from the [Kummu et al. gridded GDP dataset](https://doi.org/10.5281/zenodo.16741980), which carries **official regional accounts for 89 countries**. Elsewhere it spreads the national GDP across regions with a model that closely follows population, so a GDP map of such a country mostly repeats its population map. Check before you publish:

```python
from popgrid.data import gdp_quality
gdp_quality("ESP")   # 'reported': official regional GDP
gdp_quality("ETH")   # 'modelled': national GDP only, regional split estimated
```

---

## Your own data

Both classes accept any GeoDataFrame through `from_geodataframe`, so popgrid is not limited to the bundled countries. Point it at a shapefile, a GeoJSON, or a PostGIS query.

- `AreaGrid.from_geodataframe(gdf, region_col=...)` sizes each region by its land area.
- `PopGrid.from_geodataframe(gdf, region_col=..., weight_col=...)` sizes each region by a numeric column you supply, such as population.

```python
import geopandas as gpd
from popgrid import AreaGrid, PopGrid

gdf = gpd.read_file("examples/data/barcelona_districts.geojson")

# Sized by district land area
AreaGrid.from_geodataframe(gdf, region_col="district").plot(
    title="Land Area of Barcelona",
    subtitle="Every block = 0.1% of city land area",
)

# Sized by your own population column
PopGrid.from_geodataframe(gdf, region_col="district", weight_col="population").plot(
    title="Population of Barcelona",
    subtitle="Every block = 0.1% of city population",
)
```

Barcelona's 10 districts, built entirely from a custom GeoDataFrame. Eixample, the densest district, balloons under population; Les Corts and the green hills of Sarrià shrink.

| Land area | Population |
|---|---|
| ![Land area of Barcelona](https://raw.githubusercontent.com/databites-tech/popgrid/main/docs/img/barcelona_area.png?v=0.7.1) | ![Population of Barcelona](https://raw.githubusercontent.com/databites-tech/popgrid/main/docs/img/barcelona_population.png?v=0.7.1) |

A full runnable version is in [`examples/barcelona.py`](examples/barcelona.py), using the district boundaries in [`examples/data/barcelona_districts.geojson`](examples/data/barcelona_districts.geojson).

---

## Use with matplotlib

`.plot()` gives you a finished figure. `.draw(ax)` gives you just the map, inside an Axes you already own, so popgrid fits into any layout: a row of countries, a map beside a chart, your own title block.

```python
import matplotlib.pyplot as plt
from popgrid import AreaGrid, PopGrid

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
AreaGrid("ESP").draw(ax1, background_style="none")
PopGrid("ESP").draw(ax2, background_style="none")
ax1.set_title("Land area")
ax2.set_title("Population")
```

<p align="center">
  <img src="https://raw.githubusercontent.com/databites-tech/popgrid/main/docs/img/spain_subplots.png?v=0.7.1" width="100%" alt="Spain drawn into two matplotlib subplots">
</p>

`draw` returns the Axes, sets its aspect to equal and turns the axis off. It draws the main landmass and nearby islands; distant territories (Canary Islands, Hawaii, overseas France) need their own Axes, which is what `plot()` arranges for you.

### Styling

Both `plot()` and `draw()` take a `style` dict. Leave a key out and it keeps its automatic value.

| Key | Default | What it controls |
|---|---|---|
| `edge_color` | `#000000` | Region outline colour |
| `edge_width` | auto | Outline width in points (auto: 8% of a block) |
| `grid_color` | auto | Background grid lines or tiles |
| `grid_width` | `0.35` | Grid line width in points |
| `cell_gap` | auto | Gap between blocks, as a fraction of a block |
| `label_color` | auto | Label text colour (auto: black or white by region luminance) |
| `label_size` | auto | Fixed label size in points (auto: fit each region) |
| `label_min_size`, `label_max_size` | `7`, `18` | Bounds for auto-fitted labels |
| `label_weight`, `label_font` | `bold`, default | Label font |

```python
PopGrid("DEU", color_map={"Bayern": "#1A3829"}).plot(
    style={"edge_color": "#1A3829", "edge_width": 1.0, "label_max_size": 22},
    label_mode="auto",
    labels={"Baden-Württemberg": "BaWü"},
)
```

`color_map` pins regions to colours; the rest keep their automatic, adjacency-aware palette colour. `exclude=[...]` drops regions before the grid is built (they take no blocks); countries come with a short default list of uninhabited specks, and `--exclude` adds to it on the CLI. `label_mode` is `auto` (fit the full name, else wrap it on two lines, else abbreviate), `full`, `short` or `none`. `labels` overrides individual names; an empty string hides one.

---

## Command line

`generate.py` (at the repo root) renders one or many countries from the terminal.

```bash
# Population map of Spain
python generate.py ESP --mode pop

# Land-area map
python generate.py ESP --mode area

# GDP map
python generate.py ESP --mode gdp

# Side-by-side comparison (land area vs population, equal block size)
python generate.py ESP --mode compare

# Several countries at once, into a folder
python generate.py ESP DEU FRA --mode compare --out compare/

# The 24 countries with curated census data
python generate.py ALL --mode pop

# Every supported country
python generate.py WORLD --mode compare --out world/
```

Useful flags: `--n` (target block count, default 1000), `--layout classic` (the 0.7.0 block layout, to reproduce older maps), `--mainland` (drop detached territories), `--no-panels` (keep nearby islands inline, drop distant panels), `--palette`, `--dissolve`, `--label-mode auto|full|short|none`, `--label-size`, `--edge-color`, `--edge-width`, `--grid-color`, `--bg`, `--background grid|solid|none`, `--title`, `--subtitle`, `--source`, `--dpi`, `--out`. Run `python generate.py --help` for the full list.

---

## ArcGIS Online

The optional helpers in `popgrid.arcgis` publish popgrid blocks to ArcGIS Online as Hosted Feature Layers. The ArcGIS dependency is not required for normal popgrid usage; it is imported only by the publishing helpers.

Install the optional publishing dependencies:

```bash
pip install "popgrid[arcgis]"
```

Create an ArcGIS API for Python profile:

```python
from getpass import getpass
from arcgis.gis import GIS

# Saves an encrypted profile once; the password is never written in your code
GIS("https://www.arcgis.com", "YOUR_USERNAME", getpass(), profile="popgrid_agol")
```

Preview the output locally without publishing:

```bash
python examples/publish_arcgis_online.py ESP --n 200 --mode both --dry-run
```

Publish area and population as separate Hosted Feature Layers:

```bash
python examples/publish_arcgis_online.py ESP --n 1000 --mode both --profile popgrid_agol --folder PopGrid
```

When `--mode both` is used, two services are created so the undeformed land-area grid and the population-adjusted grid do not overlap in the same Feature Service:

```text
popgrid_ESP_area_n1000
popgrid_ESP_population_n1000
```

Each layer is published with region-based symbology from popgrid's `color_hex` values. To reapply that symbology to an existing layer:

```bash
python examples/style_arcgis_layer.py --profile popgrid_agol --title popgrid_ESP_population_n1000
```

The published attributes include `iso3`, `mode`, `block_id`, `region`, `color_hex`, `target_n`, `blocks`, `share_pct`, `cell_km`, `land_id`, `cx`, and `cy`.

---

## Countries

Any country works: `PopGrid("CHL")`, `PopGrid("IND")`, `PopGrid("NGA")`. Boundaries come from Natural Earth admin-1 and population from one of two bundled tables:

- **Curated census figures** for 24 countries, at the level the map shows (autonomous communities for Spain, Länder for Germany, and so on):
  `ARG` `AUS` `BEL` `BRA` `CAN` `CHE` `CHN` `DEU` `ESP` `FRA` `GBR` `ITA` `JPN` `KOR` `MEX` `NLD` `NOR` `PHL` `POL` `PRT` `SWE` `TUR` `USA` `ZAF`
- **GHS-POP 2025 estimates** for everything else: the JRC's 1 km gridded population summed over every Natural Earth admin-1 polygon on Earth. Modelled, not census; within a few percent of official regional figures, and always consistent with the polygon being drawn.

`popgrid.supported_countries()` lists the 207 countries with enough regions to make a map; `popgrid.bundled_countries()` the 24 curated ones. `python generate.py WORLD --mode compare` renders all of them.

For sub-national data such as cities, or to use your own population numbers, use `from_geodataframe` with your own polygons.

---

## How it works

1. Load admin-1 regions, project to an equal-area CRS, and choose a cell size so the country tiles into roughly `n` blocks (default 1000, so one block ≈ 0.1%).
2. Allocate an integer block quota per region using the Hamilton (largest-remainder) method, weighted by area or population.
3. For population and GDP maps, deform each landmass with a contiguous-area cartogram so a region's area tracks its value. Dense regions bulge, sparse regions pinch, relative position and silhouette are preserved. Two engines: a rubber-sheet cartogram (Dougenik et al.) by default, and a diffusion cartogram (Gastner-Newman) that `engine="auto"` switches to when the sheet would tear, which happens for countries with an empty interior such as China or Canada. Pass `engine="diffusion"` to use it everywhere.
4. Hand out the blocks along the deformed outlines (`layout="outline"`, the default). The deformed landmass is rescaled to exactly its block count, the blocks deepest inside it are kept, and each region gets exactly its quota by a balanced assignment on the signed distance to its deformed outline: a region the cartogram drew too small grows by a uniform offset, one drawn too large shrinks the same way, so borders follow the cartogram instead of turning into straight lines. The assignment also discourages borders that do not exist in reality, so walls (the Castillas around Madrid) come out of it; a repair pass then enforces the real border graph, rejoins dashed walls, keeps enclosed regions at their exact count and reopens sea that the deformation closed between regions that only face each other across water. `layout="classic"` keeps the 0.7.0 pipeline (rasterise, then grow regions to their counts) to reproduce maps made with earlier versions.
5. Place detached landmasses relative to the deformed mainland and push them clear of it if the deformation moved blocks over them, so Northern Ireland never ends up inside Great Britain.
6. Assign colours once per country from the geographic border graph, so the same region keeps the same colour across the land, population and GDP maps. By default regions are weighted by land area and only direct neighbours are kept apart; `color_mode="layouts"` also weights by the largest share of people or GDP and keeps regions two borders apart distinct.

---

## Notes and limitations

- **Population currency.** Bundled population figures are a fixed snapshot, not live data. For up-to-date or custom numbers, pass your own values through `PopGrid.from_geodataframe(..., weight_col=...)`.
- **Disputed borders.** Natural Earth draws them its own way (Kashmir, Crimea, Western Sahara). Check before publishing a map of a contested region.
- **Exact counts.** Population and GDP maps draw every region at its exact quota (`grid.quotas` vs `grid.block_counts`). When a wall region is too small to keep two non-neighbours apart, the count wins and the two may touch along a short stretch; countries with dozens of tiny units next to one dominant city (North Macedonia, Hungary or Turkey by GDP) show a few such contacts at `n=1000`, fewer at `n=2000`. Land-area maps rasterise real geography and can be a few blocks off `n`.
- **More blocks.** `n=2000` gives small regions more room (fewer one-block regions, fewer forced contacts) at roughly twice the build time; the layout itself does not depend on `n`.

---

## Made with popgrid

Blocks don't have to be people. Any number that adds up across regions works.

<p align="center">
  <img src="https://raw.githubusercontent.com/databites-tech/popgrid/main/docs/user_examples/showcase_germany_companies.png" width="70%" alt="German states by area vs active companies in the commercial register">
</p>

**Companies in the German commercial register, by state**, by [Julius Maneth](https://www.linkedin.com/in/juliusmaneth/) at [business.watch](https://www.business.watch). Every block is 0.1% of 2.35 million active firms. Hamburg and Berlin swell far beyond their land.

Built something with popgrid? Share it in [Show and tell](https://github.com/databites-tech/popgrid/discussions/categories/show-and-tell) and it may end up here.


---

## Data sources

- Country and admin-1 boundaries: [Natural Earth](https://www.naturalearthdata.com/) (`ne_10m_admin_1_states_provinces`), public domain.
- Curated population figures: national statistical offices (see the `source` column of `population.csv`).
- Fallback population figures: [GHS-POP R2023A](https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php), epoch 2025, European Commission JRC, summed over Natural Earth admin-1 polygons.
- GDP: Kummu, M. et al. (2025), [Downscaled gridded global dataset for GDP per capita PPP 1990-2022](https://doi.org/10.1038/s41597-025-04487-x), Scientific Data, updated to 2024 ([Zenodo](https://doi.org/10.5281/zenodo.16741980), CC BY 4.0). Total GDP at 30 arc-seconds, 2017 international dollars, summed over Natural Earth admin-1 polygons; `quality` marks the 89 countries with reported regional accounts.
- Barcelona example: district boundaries from [Ajuntament de Barcelona](https://opendata-ajuntament.barcelona.cat/) open data; population from the 2023 municipal register.

---

## License & credits

MIT licensed. Built by [Josep Ferrer](https://github.com/rfeers) at **[databites.tech](https://databites.tech)**, data and AI, clearly explained through diagrams.

The block-cartogram format is inspired by the "Population of X, Visualised" maps popularised by [@Civixplorer](https://twitter.com/Civixplorer).

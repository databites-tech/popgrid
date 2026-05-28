# popgrid

**Tiled grid maps for Python — land area and population.**

`popgrid` turns any country into a square grid map where every cell represents an equal fraction of the country's land area (or, in a future release, its population). Regions are automatically colored and labeled. Geographically detached territories are shown as panel insets.

Inspired by [@Civixplorer](https://twitter.com/Civixplorer).

---

## Installation

```bash
pip install -e .
```

Natural Earth boundary data (~14 MB) is downloaded automatically on first use and cached at `~/.popgrid/cache/`.

---

## Quick start

```python
from popgrid import AreaGrid

ag = AreaGrid("ESP", dissolve_by="auto")
fig = ag.plot()
fig.savefig("spain.png", dpi=200, bbox_inches="tight")
```

```bash
python examples/country.py ESP
python examples/country.py USA --background grid
python examples/country.py FRA --scope mainland --save france.png
```

---

## Classes

### `AreaGrid` — land-area tiled map

Each cell represents an equal share of the country's **land area**.

```python
from popgrid import AreaGrid

ag = AreaGrid(
    iso3="DEU",           # ISO 3166-1 alpha-3 country code
    n=1000,               # target number of cells (default: 1000)
    dissolve_by="auto",   # group Natural Earth admin-1 rows into regions
    palette="qual",       # colour palette: "qual", "databites", "bold", or list of hex
)
ag.build()               # downloads data, projects, rasterises
fig = ag.plot(
    background_style="grid",   # "grid" | "solid" | "none"
    show_region_borders=True,  # black outline around each region
    scope="all",               # "all" | "islands" | "mainland"
    data_source="Natural Earth · popgrid",
)
ag.save("germany.png", dpi=200)
```

### `PopGrid` — population-weighted tiled map *(coming in v0.4.0)*

Each cell will represent an equal share of the country's **population**.

```python
from popgrid import PopGrid

pg = PopGrid("ESP")
pg.build()   # raises NotImplementedError — not yet implemented
```

---

## Parameters

### `AreaGrid(iso3, n, dissolve_by, palette, cluster_distance_km)`

| Parameter | Default | Description |
|---|---|---|
| `iso3` | required | ISO 3166-1 alpha-3 code (`"ESP"`, `"USA"`, `"DEU"`, …) |
| `n` | `1000` | Target number of cells |
| `dissolve_by` | `"auto"` | Column to dissolve admin-1 rows into parent regions. `"auto"` uses `RECOMMENDED_SETTINGS`. |
| `palette` | `"qual"` | Named palette or list of hex colours |
| `cluster_distance_km` | `200` | Max distance (km) to treat landmasses as the same group |

### `.plot(...)` / `.save(path, dpi)`

| Parameter | Default | Description |
|---|---|---|
| `background_style` | `"grid"` | `"grid"` — light grey lines · `"solid"` — filled rectangle · `"none"` — plain |
| `show_region_borders` | `True` | Draw black outline around each region |
| `scope` | `"all"` | `"all"` · `"islands"` · `"mainland"` |
| `title` | auto | Override the auto-generated title |
| `subtitle` | auto | Override the auto-generated subtitle |
| `data_source` | auto | Attribution shown bottom-left |

---

## CLI

```bash
python examples/country.py ISO3 [options]

python examples/country.py ESP
python examples/country.py USA --background grid --dpi 300
python examples/country.py FRA --scope mainland --save france.png
python examples/country.py DEU --no-region-borders --background solid
python examples/country.py NLD --save show
```

**Options:**

| Flag | Description |
|---|---|
| `--n N` | Number of cells (default: 1000) |
| `--background` | `grid` / `solid` / `none` (default: `grid`) |
| `--no-region-borders` | Disable region outlines |
| `--scope` | `all` / `islands` / `mainland` |
| `--dissolve COLUMN` | Override dissolve column |
| `--save PATH` | Output path (default: `ISO3.png`). Use `show` for interactive window. |
| `--dpi N` | Export resolution (default: 200) |
| `--source STR` | Data source attribution |

Inspect Natural Earth data for any country:

```bash
python examples/inspect_country.py ESP
python examples/inspect_country.py FRA
```

---

## Supported countries

`dissolve_by="auto"` is verified and tested for 24 countries:

| Country | Regions shown |
|---|---|
| Spain (ESP) | 19 Autonomous Communities |
| France (FRA) | 18 Regions |
| Italy (ITA) | 20 Regions |
| Philippines (PHL) | 17 Regions |
| Germany (DEU) | 16 States |
| USA | 51 States |
| Brazil (BRA) | 27 States |
| Mexico (MEX) | 32 States |
| China (CHN) | 34 Provinces |
| Netherlands (NLD) | 12 Provinces |
| Japan (JPN) | 47 Prefectures |
| + 13 more | see `RECOMMENDED_SETTINGS` |

Any country in Natural Earth admin-1 works — `dissolve_by="auto"` falls back gracefully for unconfigured countries.

---

## Data source

**Natural Earth** — `ne_10m_admin_1_states_provinces` v5.1.1
[naturalearthdata.com](https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-1-states-provinces/)
Public domain. Auto-downloaded on first use, cached at `~/.popgrid/cache/`.

---

## Roadmap

| Version | Feature |
|---|---|
| **v0.1.0** ✅ | `AreaGrid` — land-area tiled maps, 24 countries, CLI |
| **v0.2.0** 🔜 | `AreaGrid.from_geodataframe()` — custom shapefiles |
| **v0.3.0** 🔜 | MkDocs documentation site, GitHub Actions CI |
| **v0.4.0** 🔜 | `PopGrid` — population-weighted cells |
| **v0.5.0** 🔜 | PyPI publish (`pip install popgrid`) |

---

## Development

```bash
git clone https://github.com/databites-tech/popgrid.git
cd popgrid
pip install -e ".[dev]"
python -m pytest tests/ -v
```

---

## License

MIT © Josep Ferrer — [databites.tech](https://databites.tech)

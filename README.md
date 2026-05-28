# popgrid

**Population grid visualizations for Python.**

`popgrid` generates Civixplorer-style population cartograms where every square represents an equal fraction of a country's population. Regions are color-coded; cities are marked with a lighter shade.

```python
from popgrid import SquareGrid

m = SquareGrid.from_naturalearth("ESP")
m.build(resolution=0.1, cities=10)
m.save("spain_population.png")
```

## Installation

```bash
pip install popgrid
```

## Chart types

| Class | Description | Status |
|---|---|---|
| `SquareGrid` | Civixplorer-style square grid | ✅ v0.1 |
| `HexGrid` | Hex tilegram (Tilegrams-style) | 🔜 v0.2 |
| `Dorling` | Dorling cartogram (circles) | 🔜 v0.3 |

## Quick start

```python
from popgrid import SquareGrid

# Auto-downloads Natural Earth data on first run (~30 MB, cached locally)
m = SquareGrid.from_naturalearth("DEU")
m.build(resolution=0.1, cities=10)
fig = m.plot(style="default")
m.save("germany_population.png", dpi=300)
```

### With your own data

```python
m = SquareGrid.from_file(
    "my_regions.geojson",
    pop_col="population_2023",
    iso3="NLD",
)
m.build(resolution=0.05)  # finer grid: each cell = 0.05% of population
m.save("netherlands.svg")
```

### Parameters

**`build(resolution, cities)`**
- `resolution` — fraction of total population per cell. `0.1` = 1 000 cells. `0.05` = 2 000 cells.
- `cities` — number of top cities to mark (0 = none). Requires `iso3`.

**`plot(style, figsize, title, show_legend, show_title, colors)`**
- `style` — `"default"` or `"databites"` color palette.
- `colors` — custom list of hex colors, one per region.

## Data sources

- **Admin-1 boundaries**: Natural Earth 1:10m cultural (`ne_10m_admin_1_states_provinces`)
- **Cities**: Natural Earth 1:10m cultural (`ne_10m_populated_places`)
- Data is auto-downloaded on first use and cached at `~/.popgrid/cache/`.

## License

MIT © Josep Ferrer / DataBites

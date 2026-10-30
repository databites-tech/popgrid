# Changelog

All notable changes to popgrid are documented here.

## [Unreleased]

### Fixed
- **Southern Chile no longer drops into a "Detached territories" box on the
  population map.** Territories are placed relative to the deformed
  mainland, and that frame was taken from the deformed polygons, which keep
  a thin sliver of Aysén reaching ~600 km past the last drawn block. It now
  comes from the mainland's actual blocks, so Magallanes sits inline below
  Aysén, as on the land map. Across 38 countries checked, the only other
  change is Okinawa, now inline in Japan's population map as it is on the
  land map.

### Changed
- **`generate.py --mode compare` looks better at any shape.** The canvas now
  follows the country: wide countries get a wide figure (never wider than
  2:1), tall ones like Chile or Norway a near-square one where the maps run
  the full height instead of shrinking to a sliver. Each map is drawn twice,
  the second time at the exact size it is printed, so region labels are
  fitted to the size people actually see.
- New footer in compare mode, left-aligned and readable: a bold credit line
  (`--credit`, default "Made with popgrid", `''` hides it) above the sources, which now name the population source too
  (the national office for curated countries, GHS-POP otherwise).
- Subtitles keep clear margins: the block total moved to its own line, and
  each column is at least as wide as its title plus padding.
- `plot(figsize=...)` keeps the title, "Detached territories" and footer bands
  at their size in inches and gives the rest to the maps, so a small figure no
  longer squeezes their text together.

## [0.6.1] - 2026-10-29

Every region gets its blocks, including the ones nobody can reach.

### Fixed
- **Regions enclosed by other regions now hit their quota** (#3, reported by
  @jgomezsa). Quota fitting only added or removed blocks where a region
  touched empty space, so a region surrounded on all sides kept whatever
  count the cartogram layout gave it: La Rioja drew 2 blocks for a quota of
  7, Zug 26 for 15, Tucumán 33 for 38. A new final pass,
  `cartogram.repair_quotas`, moves blocks one at a time along chains of
  real neighbours (a region with too many passes a block to a neighbour,
  which passes one on, until it reaches a region with too few or the outer
  edge). Every move keeps each region in one piece, never opens a hole,
  never creates or lengthens a border between regions that are not
  neighbours in reality, and never erases one that is.
- Regions that the layout squeezed out entirely (Goa in India, for example)
  are seeded next to their real neighbours and grown to their quota.
- Totals now land on `n`: Spain, Switzerland, Germany, France, Italy,
  Sweden, the UK, China and India all draw exactly 1000 blocks with every
  region at its quota.

### Added
- **`PopGrid.quotas`**: the block quota each region was allocated.
- **`block_counts`** on both classes: the blocks each region actually drew.
  Compare the two to see exactly how faithful a map is.

### Known limits
- A region that is the only thing keeping two non-neighbours apart can stay
  a block or two over quota, because correct borders win over the last
  blocks (La Pampa in Argentina, 10 for 8).
- Hungary's 23 cities with county rights are separate Natural Earth regions,
  and a few end up as one-block rings that cannot be repaired. Egypt's
  layout is broken upstream of this pass. Both are tracked separately.
- `AreaGrid` rasterises real geography and is not repaired; its counts can
  still be a few blocks off.

## [0.6.0] - 2026-10-27

Every country. popgrid no longer depends on a hand-maintained population
table for the countries it can draw.

### Added
- **`popgrid/resources/population_ghs.csv`**: population of every Natural
  Earth admin-1 unit on Earth (4,561 rows, 251 countries), summed from the
  JRC GHS-POP 2025 raster (30 arc-second grid, about 1 km) with
  area-weighted zonal statistics. `PopGrid` falls back to it for any country
  without rows in the curated `population.csv`, so `PopGrid("CHL")`,
  `PopGrid("IND")` or `PopGrid("NGA")` now work with no setup. The curated
  census figures for the original 24 countries still take priority.
- **`scripts/build_population_ghs.py`** regenerates that table from the
  raster (not in the repo, about 480 MB from the JRC open data server).
- **`popgrid.bundled_countries()`** (the 24 curated) and
  **`popgrid.supported_countries()`** (207 countries with at least three
  admin-1 units).
- **`generate.py WORLD`** renders every supported country. `ALL` keeps its
  meaning: the 24 curated ones.
- Dissolve settings for countries whose Natural Earth level is far finer than
  what people recognise: Slovenia (193 municipalities → 12 regions), Latvia
  (119 → 5), Uganda (112 → 4), Malta (68 → 3), Azerbaijan (78 → 10),
  Thailand (77 → 6).

- **Diffusion cartogram engine** (Gastner & Newman 2004), pure NumPy, as
  `engine="diffusion"` on `PopGrid` and `--engine` on the CLI. It solves the
  density-equalising flow in Fourier space and carries the map along it, so
  the mapping is continuous by construction: regions can shrink twenty-fold
  without tearing. The default `engine="auto"` keeps the rubber-sheet engine
  and switches to diffusion only when that one fails, detected by folded
  (degenerate) polygons in its output, a tripped divergence guard, or a
  region more than 2× off target. That is what used to produce the hole in
  China and the detached west; China, the UK, Canada, India, Mongolia and
  Algeria now take the diffusion path, the other curated countries render as
  before. The "China leaves a void" limitation is gone.
- **`exclude=`** on the constructors, `from_geodataframe` and the CLI
  (`--exclude NAME ...`): regions removed before the grid is built, so they
  take no blocks and the others' shares are computed without them. Countries
  carry a default list of uninhabited specks that would otherwise earn one
  land block and zero population blocks: Macquarie and Lord Howe Islands
  (AUS), the Paracel Islands (CHN), Bouvet Island (NOR). Svalbard and the
  like stay, since empty land next to a shrunken population map is the
  point of the comparison.
- `panel_regions` per-country setting: regions always drawn as a panel inset.
  Set for the USA so Alaska stops rendering inline above Washington (its
  panhandle passed the proximity test) while its islands sat in a panel.

### Changed
- **Readability pass on the default figure.** Subtitle 13 → 17 pt (19 pt in
  compare mode), panel-inset heading and names 12/10 → 14/13 pt with room so
  they no longer touch the inset border, source line 9 → 10 pt. Label colour
  is now chosen by WCAG contrast ratio (black or white, whichever contrasts
  more with the block colour) instead of a luminance threshold, and the halo
  behind every label is heavier and more opaque, so labels read on dark blue,
  red and purple blocks as well as on light ones.
- Per-country `display_names` in `RECOMMENDED_SETTINGS`: label renames that
  still go through the fitter (Spain: `Foral de Navarra` → `Navarra`,
  `Islas Baleares` → `Baleares`, `Canary Is.` → `Canary Islands`).
- `generate.py --data` (any GeoJSON, GeoPackage, Shapefile, Parquet or CSV
  with an `iso3` column) and `examples/data/europe.geojson` landed with 0.5.0's
  commit; documented here since the 0.5.0 notes predate them.

### Fixed
- **Region colours are now identical between the land-area and population
  maps of every country.** `PopGrid` assigned colours after simplifying
  high-vertex coastlines and dropping zero-block regions; both change the
  border graph and region order that the colouring derives from, so the same
  state could come out brown on one side and grey on the other (Australia).
  Colours are now assigned from the raw projected geometry, before any
  population-specific step, and a test checks seven countries end to end.
- Leader-line labels for one- and two-block territories (Ceuta, Melilla,
  Lord Howe Island) were 5.5 pt italic; they are now 10 pt bold with a halo.
- **Uninhabited islands no longer get the country's mean population.** A
  region without a curated row used to receive the mean of the known ones:
  Macquarie Island was drawn with 2.5 million people (10 % of Australia's
  blocks), Bouvet Island with 270,000 (5 % of Norway's). Such regions now take
  their GHS-POP value first, which is zero for uninhabited land, so they get
  no blocks.
- Packaging metadata uses the SPDX license form (`license = "MIT"`,
  `license-files`) instead of the deprecated table and classifier; builds
  are warning-free and keep working after setuptools drops the old form.
- The "no cell centres fell inside landmass" message for single-cell islets
  is logged at INFO, not WARNING, so `generate.py` output is quiet by default.
- Polygons that become invalid when projected (rings crossing the
  antimeridian: Chukotka, Fiji) are repaired right after projection. Some
  GEOS versions raise `TopologyException` from plain `intersects` on them,
  which broke `AreaGrid("RUS")` on those installs.
- Natural Earth's seven nameless admin-1 islets (a few km² each, off Russia,
  Mexico, Colombia, Venezuela and others) are dropped instead of receiving a
  made-up fallback population.

### Notes
- GHS-POP is a modelled estimate, not a census. Against the curated census
  rows the median region-level difference is 2 to 10 % per country, with
  larger gaps where Natural Earth's polygon does not match the statistical
  unit (the UK's "Merseyside", Korea's metropolitan cities). For a map where
  one block is 0.1 % of the country, that is well below what the eye can see,
  and the number always matches the shape actually drawn.
- Geometry still comes from Natural Earth, including its treatment of
  disputed areas (Kashmir, Crimea, Western Sahara). Check before publishing a
  map of a contested region.

## [0.5.0] - 2026-10-27

The matplotlib release. popgrid maps now drop into your own figures, every
visual knob is exposed, and region labels size themselves to the space they
actually have.

### Added
- **`AreaGrid.draw(ax)` / `PopGrid.draw(ax)`.** Draw the map into any
  Matplotlib Axes you already have: subplots, a map next to a chart, a custom
  layout. Returns the same Axes. `plot()` is unchanged and still builds a
  complete figure with title, insets and source line; it now uses `draw`
  internally, so the two render identically.
- **`style=` dict on `plot()` and `draw()`** with `edge_color`, `edge_width`,
  `grid_color`, `grid_width`, `cell_gap`, `label_color`, `label_size`,
  `label_min_size`, `label_max_size`, `label_weight`, `label_font`. Any key
  left out keeps its automatic value; unknown keys raise. Mirrored on the CLI
  as `--edge-color`, `--edge-width`, `--grid-color`, `--label-size`.
- **`color_map=` on the constructors and `from_geodataframe`** to pin
  specific regions to specific colours; unlisted regions keep their automatic,
  adjacency-aware palette colour.
- **`label_mode=`** (`auto` | `full` | `short` | `none`) and **`labels=`**
  (`{region: text}`, empty string hides a label) on `plot()` and `draw()`,
  and `--label-mode` on the CLI.
- **Optional ArcGIS Online publishing** (`popgrid.arcgis`), contributed by
  [@aitorcalero](https://github.com/aitorcalero) in #2: convert popgrid blocks to
  WGS84 GeoDataFrames, publish them as Hosted Feature Layers with region colours,
  and two example CLIs. Install with `pip install "popgrid[arcgis]"`; normal use
  does not need ArcGIS.

### Changed
- **Region labels are sized to fit.** Each label is measured against the
  largest rectangle inscribed in its region (not the bounding box, which
  overstated the room in L-shaped regions) using real glyph widths. Big
  regions now get labels up to 18 pt instead of 11; narrow ones wrap on two
  lines (`Castilla-` / `La Mancha`), fall back to the name proper for
  "X de Y" names (`Foral de Navarra` → `Navarra`), and only then abbreviate
  (`Castilla y León` → `CL`). Abbreviations are capped at 1.5× the minimum
  size so they never outshout a neighbour's full name.
- Inline territories below `label_min_cells` get a single leader-line label.
  Previously two-cell territories such as Ceuta were labelled twice (in-cell
  and leader line) because the two paths used different thresholds.
- `from_geodataframe` now goes through `__init__` instead of `cls.__new__`,
  so new constructor options reach custom-data grids automatically.
- `publish_hosted_feature_layer` now requires an authenticated `GIS` and raises a
  clear error instead of falling back to an anonymous session that cannot publish.

## [0.4.0] - 2026-06-03

Custom data release. `from_geodataframe` now drives both `AreaGrid` and
`PopGrid`, so any GeoDataFrame — a city's districts, a custom shapefile, a
PostGIS query — can be turned into an area or population cartogram, not just the
24 bundled countries.

### Added
- **`PopGrid.from_geodataframe(gdf, region_col=..., weight_col=...)`.** Sizes
  each region by a numeric column you supply (e.g. population), so population
  cartograms work on arbitrary polygon data. `AreaGrid.from_geodataframe`
  continues to weight by land area and ignores `weight_col`.
- **Barcelona worked example** (`examples/barcelona.py` +
  `examples/data/barcelona_districts.geojson`): the city's 10 districts by land
  area vs population, built entirely from a custom GeoDataFrame. District
  boundaries from Ajuntament de Barcelona open data; population from the 2023
  municipal register.

### Changed
- `_prepare_custom_gdf` carries an optional weight column through as
  `region_weight`; `PopGrid.build` uses it when present and otherwise falls back
  to the bundled population table for the 24 known countries.

### Notes
- Calling `PopGrid.from_geodataframe` without `weight_col` raises a clear error
  pointing you to `AreaGrid.from_geodataframe` for an area-weighted map.

## [0.3.0] - 2026-06-03

The population release. `PopGrid` turns a country into a block cartogram where
every cell is an equal share of the population, alongside the existing
land-area `AreaGrid`.

### Added
- **`PopGrid` — population-weighted maps.** Regions are deformed with a
  Dougenik rubber-sheet cartogram so each region's area is proportional to its
  population, then rasterised to a grid (each cell = an equal share of the
  national population, allocated by the Hamilton / largest-remainder method).
  Population data for 24 countries ships in `popgrid/resources/population.csv`.
- **Stable region colours across grid modes.** Colours are assigned from
  mode-invariant geography (real border adjacency + land area), so an
  `AreaGrid` and a `PopGrid` of the same country share an identical region
  colour map and can be compared region by region (e.g. Madrid keeps its colour
  while growing from land area to population).
- **Adjacency- and size-aware colouring.** Neighbouring regions are always
  visually distinct, and the largest regions are pushed apart in colour space so
  big blocks never share a hue.
- **Unified `generate.py` CLI.** One script generates population *or* land-area
  maps (`--mode pop|area`), for one country, several, or `ALL`, with full
  styling flags (palette, scope, background, labels, etc.). Replaces the old
  single-country `examples/country.py`.
- **Per-country segmentation** for over-granular countries: Japan renders its 9
  traditional regions (Kanto, Kinki, …) rather than 47 prefectures, the UK its
  16 sub-regions rather than 232 districts. Population is aggregated from the
  fine source level up to the parent regions automatically.

### Fixed
- **Gap-free rasterisation.** Cells in the seams between deformed regions are now
  filled (interior holes of the deformed outline are closed before gridding),
  and remaining enclosed pockets up to 50 cells are filled count-neutrally so
  block totals don't drift. All 24 bundled countries build with zero interior
  holes except China.
- **Grid-aligned territories.** Detached and inline territories (Spain's Ceuta,
  Melilla, Balearic Islands, etc.) now snap to the mainland's global lattice
  instead of sitting up to half a cell off; re-anchoring is a rigid translation
  that preserves each territory's internal spacing.
- **Build performance / no more hangs.** Pathologically high-vertex coastlines
  (e.g. UK 'Highlands and Islands', 4,200+ vertices) are simplified before
  deformation, and the cartogram stops iterating if geometry complexity balloons
  from over-inflation. UK went from a 260s+ timeout to a few seconds. Normal
  adjacent regions are never simplified, so quotas stay correct.

### Changed
- **Japan layout:** Hokkaido renders inline (it was being pushed into a detached
  panel) and Okinawa moves to its own panel, via a higher inline margin (near
  islands like Hokkaido and Corsica sit inline; far territories like the
  Canaries, Hawaii, Alaska and French overseas stay in panels) plus a tighter
  per-country cluster distance for Japan.

### Known limitations
- **China-class geography.** Countries with an extreme population-density
  gradient and long, thin provinces (China is the clearest case) tear the
  contiguous cartogram during deformation, leaving a central void and
  disconnected-looking pieces. They build without error but are not as clean as
  compact, evenly-populated countries. A diffusion-based cartogram
  (Gastner-Newman) would be needed to handle them well.
- **Population currency.** Bundled population figures are a fixed snapshot, not
  live data.

## [0.2.0]

### Added
- `AreaGrid.from_geodataframe()` — build a land-area grid from any GeoDataFrame
  (custom shapefiles / GeoJSON), not just bundled Natural Earth countries.
- Configurable `title`, `subtitle`, and `data_source` at construction.

## [0.1.0]

### Added
- Initial release: `AreaGrid` land-area tiled maps for Natural Earth countries,
  qualitative palettes, panel insets for detached territories, and CLI tools.

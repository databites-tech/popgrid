"""
Tests for draw(ax), the style dict, colour overrides and region labels.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from popgrid import AreaGrid, PopGrid  # noqa: E402
from popgrid.render import (  # noqa: E402
    DEFAULT_STYLE,
    _abbreviate,
    _key_word,
    _largest_rectangle,
    _wrap_name,
    resolve_style,
)


@pytest.fixture(scope="module")
def germany():
    ag = AreaGrid("DEU", n=300)
    ag.build()
    return ag


@pytest.fixture(scope="module")
def spain_pop():
    pg = PopGrid("ESP", n=300)
    pg.build()
    return pg


# ── draw(ax) ──────────────────────────────────────────────────────────────────

def test_draw_returns_same_axes(germany):
    fig, ax = plt.subplots()
    out = germany.draw(ax)
    assert out is ax
    plt.close(fig)


def test_draw_adds_artists_to_given_axes_only(germany):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    germany.draw(ax1)
    assert len(ax1.collections) > 0
    assert len(ax2.collections) == 0 and len(ax2.texts) == 0
    assert len(fig.axes) == 2  # no axes were added
    plt.close(fig)


def test_draw_sets_equal_aspect_and_hides_axis(germany):
    fig, ax = plt.subplots()
    germany.draw(ax)
    assert ax.get_aspect() == 1.0
    assert not ax.axison
    plt.close(fig)


def test_draw_leaves_axes_background_alone_by_default(germany):
    fig, ax = plt.subplots()
    ax.set_facecolor("#123456")
    germany.draw(ax)
    assert matplotlib.colors.to_hex(ax.get_facecolor()) == "#123456"
    plt.close(fig)


def test_draw_background_color_applied(germany):
    fig, ax = plt.subplots()
    germany.draw(ax, background_color="#ffffff")
    assert matplotlib.colors.to_hex(ax.get_facecolor()) == "#ffffff"
    plt.close(fig)


def test_draw_mainland_scope_draws_fewer_cells(spain_pop):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    spain_pop.draw(ax1, scope="islands", show_labels=False)
    spain_pop.draw(ax2, scope="mainland", show_labels=False)
    n_islands = sum(len(c.get_paths()) for c in ax1.collections)
    n_mainland = sum(len(c.get_paths()) for c in ax2.collections)
    assert n_mainland < n_islands
    plt.close(fig)


def test_draw_invalid_scope_raises(germany):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        germany.draw(ax, scope="everything")
    plt.close(fig)


def test_draw_builds_lazily():
    ag = AreaGrid("DEU", n=200)
    fig, ax = plt.subplots()
    ag.draw(ax)
    assert ag._built
    plt.close(fig)


# ── style ─────────────────────────────────────────────────────────────────────

def test_resolve_style_defaults_when_none():
    assert resolve_style(None) == DEFAULT_STYLE


def test_resolve_style_rejects_unknown_key():
    with pytest.raises(ValueError, match="Unknown style key"):
        resolve_style({"edge_colour": "#000"})


def test_resolve_style_none_values_keep_defaults():
    st = resolve_style({"edge_color": None, "edge_width": 2.5})
    assert st["edge_color"] == DEFAULT_STYLE["edge_color"]
    assert st["edge_width"] == 2.5


def test_edge_color_and_width_reach_the_lines(germany):
    fig, ax = plt.subplots()
    germany.draw(ax, show_labels=False, style={"edge_color": "#ff0000", "edge_width": 3.3})
    lines = ax.get_lines()
    assert lines
    assert all(matplotlib.colors.to_hex(ln.get_color()) == "#ff0000" for ln in lines)
    assert all(ln.get_linewidth() == 3.3 for ln in lines)
    plt.close(fig)


def test_label_color_and_size_reach_the_text(germany):
    fig, ax = plt.subplots(figsize=(10, 10))
    germany.draw(ax, style={"label_color": "#00ff00", "label_size": 9.5})
    texts = [t for t in ax.texts if t.get_text()]
    assert texts
    assert all(matplotlib.colors.to_hex(t.get_color()) == "#00ff00" for t in texts)
    assert all(t.get_fontsize() == 9.5 for t in texts)
    plt.close(fig)


def test_plot_accepts_style(germany):
    fig = germany.plot(show_labels=False, style={"edge_color": "#0000ff"})
    assert fig is not None
    plt.close(fig)


def test_plot_rejects_bad_style_key(germany):
    with pytest.raises(ValueError):
        germany.plot(style={"nope": 1})


# ── colour overrides ──────────────────────────────────────────────────────────

def test_color_map_override_applied():
    ag = AreaGrid("DEU", n=200, color_map={"Bayern": "#123456"})
    ag.build()
    assert ag.color_map["Bayern"] == "#123456"
    others = {k: v for k, v in ag.color_map.items() if k != "Bayern"}
    assert others  # the rest still coloured automatically
    assert "#123456" not in others.values()


def test_color_map_override_unknown_region_ignored():
    ag = AreaGrid("DEU", n=200, color_map={"Atlantis": "#123456"})
    ag.build()
    assert "Atlantis" not in ag.color_map


def test_from_geodataframe_accepts_color_map():
    import geopandas as gpd
    gdf = gpd.read_file("examples/data/barcelona_districts.geojson")
    ag = AreaGrid.from_geodataframe(gdf, region_col="district", n=200,
                                    color_map={"Eixample": "#abcdef"})
    ag.build()
    assert ag.color_map["Eixample"] == "#abcdef"


# ── labels ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,short", [
    ("Madrid", "MAD"),
    ("Castilla y León", "CL"),
    ("Castilla-La Mancha", "CM"),
    ("New South Wales", "NSW"),
    ("Foral de Navarra", "FN"),
    ("Baden-Württemberg", "BW"),
    ("La Rioja", "RIO"),
])
def test_abbreviate(name, short):
    assert _abbreviate(name) == short


@pytest.mark.parametrize("name,key", [
    ("Foral de Navarra", "Navarra"),
    ("Region of Murcia", "Murcia"),
    ("Castilla-La Mancha", None),   # article, not a genitive: stays one name
    ("Madrid", None),
])
def test_key_word(name, key):
    assert _key_word(name) == key


def test_wrap_name_splits_near_middle():
    assert _wrap_name("Castilla-La Mancha") == ["Castilla-", "La Mancha"]
    assert _wrap_name("Foral de Navarra") == ["Foral de", "Navarra"]
    assert _wrap_name("Madrid") is None


def test_largest_rectangle_in_l_shape():
    # 5-wide, 1-tall bar with a 1-wide, 4-tall stem on the left: the bar wins.
    comp = {(i, 0) for i in range(5)} | {(0, j) for j in range(1, 5)}
    i0, j0, cols, rows = _largest_rectangle(comp)
    assert (cols, rows) == (5, 1) and (i0, j0) == (0, 0)


def test_largest_rectangle_full_block():
    comp = {(i, j) for i in range(3) for j in range(4)}
    assert _largest_rectangle(comp) == (0, 0, 3, 4)


def test_label_mode_none_draws_no_text(germany):
    fig, ax = plt.subplots()
    germany.draw(ax, label_mode="none")
    assert not [t for t in ax.texts if t.get_text()]
    plt.close(fig)


def test_label_mode_short_abbreviates(germany):
    fig, ax = plt.subplots(figsize=(10, 10))
    germany.draw(ax, label_mode="short")
    texts = {t.get_text() for t in ax.texts}
    assert "BAY" in texts        # Bayern
    assert "Bayern" not in texts
    plt.close(fig)


def test_label_mode_full_never_abbreviates(germany):
    fig, ax = plt.subplots(figsize=(3, 3))   # tiny figure: auto would abbreviate
    germany.draw(ax, label_mode="full")
    texts = {t.get_text().replace("\n", " ") for t in ax.texts}
    assert "Bayern" in texts
    plt.close(fig)


def test_label_override_and_suppression(germany):
    fig, ax = plt.subplots(figsize=(10, 10))
    germany.draw(ax, labels={"Bayern": "Bavaria", "Hessen": ""})
    texts = {t.get_text() for t in ax.texts}
    assert "Bavaria" in texts
    assert "Bayern" not in texts and "Hessen" not in texts
    plt.close(fig)


def test_auto_labels_scale_with_figure(germany):
    sizes = {}
    for w in (4, 14):
        fig, ax = plt.subplots(figsize=(w, w))
        germany.draw(ax)
        bayern = [t for t in ax.texts if t.get_text().startswith("Bay")]
        sizes[w] = bayern[0].get_fontsize()
        plt.close(fig)
    assert sizes[14] > sizes[4]
    assert DEFAULT_STYLE["label_min_size"] <= sizes[4] <= DEFAULT_STYLE["label_max_size"]
    assert sizes[14] <= DEFAULT_STYLE["label_max_size"]


def test_invalid_label_mode_raises(germany):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        germany.draw(ax, label_mode="tiny")
    plt.close(fig)


# ── contrast and display names ────────────────────────────────────────────────

def test_text_color_picks_higher_contrast():
    from popgrid.render import _contrast, _text_color
    for bg in ("#4E79A7", "#1A3829", "#E15759", "#EDC948", "#ffffff", "#000000"):
        fg = _text_color(bg)
        other = "#111111" if fg == "#ffffff" else "#ffffff"
        assert _contrast(bg, fg) >= _contrast(bg, other)
        assert _contrast(bg, fg) >= 4.4   # readable on every palette colour


def test_display_names_apply_to_labels():
    pg = AreaGrid("ESP", n=400)
    pg.build()
    fig, ax = plt.subplots(figsize=(12, 10))
    pg.draw(ax)
    texts = {t.get_text().replace("\n", " ") for t in ax.texts}
    # Renamed before fitting: either the full new name or its abbreviation,
    # never the Natural Earth name or an abbreviation of it.
    assert texts & {"Baleares", "BAL"}
    assert not texts & {"Islas Baleares", "IB"}
    plt.close(fig)


# ── every region drawn ────────────────────────────────────────────────────────

@pytest.mark.parametrize("iso", ["GBR", "ESP", "DNK", "GRC"])
def test_every_region_with_blocks_is_drawn(iso):
    """
    A region that earned blocks must appear on the figure. Single-cell islets
    of a mainland region are filtered on purpose, so a region may lose a few
    blocks to that filter, never all of them.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.colors as mc
    ag = AreaGrid(iso, n=400)
    ag.build()
    cells = pd.concat([lm["cells"] for lm in ag._landmasses if not lm["cells"].empty])
    fig = ag.plot(show_labels=False)
    drawn = []
    for ax in fig.axes:
        for col in ax.collections:
            fcs = col.get_facecolors()
            if len(fcs):
                drawn.append(fcs[:, :3])
    drawn = np.concatenate(drawn)
    for region, n in cells["region_name"].value_counts().items():
        rgb = np.array(mc.to_rgb(ag.color_map[region]))
        hits = int(np.all(np.isclose(drawn, rgb, atol=0.01), axis=1).sum())
        assert hits > 0, f"{iso}: {region} has {n} blocks but none drawn"
        assert n - hits <= 4, f"{iso}: {region} has {n} blocks but only {hits} drawn"
    plt.close(fig)

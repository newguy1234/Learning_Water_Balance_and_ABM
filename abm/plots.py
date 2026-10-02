# -*- coding: utf-8 -*-
"""Matplotlib figures for the ABM field UI — cyclic lon/lat map when available."""

from __future__ import annotations

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


_POND_COLORS = [
    "#f4f0e6",
    "#c8ddd3",
    "#7fb3a6",
    "#3d7a6e",
    "#1b4a42",
    "#0d2c28",
]
POND_CMAP = ListedColormap(_POND_COLORS)

OWNER_COLORS = [
    "#2f6f5e",
    "#b86b2b",
    "#3a5f8a",
    "#8a4b5c",
    "#5c6b2f",
    "#6b4f8a",
    "#8a6b2f",
    "#2f5c6b",
]


def _values_for_mode(world, mode: str) -> tuple[np.ndarray, str, object, str]:
    if mode == "ownership":
        return (
            world.owner_id.astype(float),
            "Ownership",
            ListedColormap([OWNER_COLORS[i % len(OWNER_COLORS)] for i in range(max(world.n_farmers(), 1))]),
            "farmer id",
        )
    if mode == "leftover":
        vals = world.last_leftover if world.day > 0 else np.zeros(world.n)
        return vals, "Leftover (IR+rain−D−ET, Okura Eq. 3)", "YlGnBu", "leftover mm"
    if mode == "irrigation":
        vals = world.last_ir if world.day > 0 else np.zeros(world.n)
        return vals, "Irrigation applied", "Blues", "applied mm"
    if mode == "drainage":
        vals = world.last_dr if world.day > 0 else np.zeros(world.n)
        return vals, "Drainage", "Oranges", "drainage mm"
    if mode == "et":
        vals = getattr(world, "last_eta", None)
        vals = vals if vals is not None and world.day > 0 else np.zeros(world.n)
        return vals, "Actual ET (demand met from pond, then topsoil)", "YlOrRd", "ET mm/d"
    if mode == "yield":
        return getattr(world, "relative_yield", np.ones(world.n)), "Relative yield", "YlGn", "Ya/Ym"
    if mode == "stress":
        vals = getattr(world, "last_stress", None)
        vals = vals if vals is not None and world.day > 0 else np.zeros(world.n)
        return vals, "Water stress (unmet ET)", "Reds", "unmet mm"
    if mode == "hops":
        return world.last_hop_in, "Perched exchange with neighbours", "Greens", "net mm"
    if mode == "capture":
        vals = getattr(world, "last_to_pool", None)
        vals = vals if vals is not None and world.day > 0 else np.zeros(world.n)
        return vals, "Underground capture to pool (LFV)", "PuBuGn", "LFV mm"
    return world.ponding_mm, "Ponding", POND_CMAP, "ponding mm"


def _color_range(mode: str, vals: np.ndarray) -> tuple[float, float]:
    vmin = float(np.nanmin(vals)) if np.isfinite(vals).any() else 0.0
    vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
    if mode == "yield":
        vmin, vmax = 0.0, 1.0
    if mode == "ponding":
        vmax = max(vmax, 80.0)
    if mode == "stress":
        vmin, vmax = 0.0, max(vmax, 1.0)  # stress is never negative; 1 mm floor for a calm day
    return vmin, vmax


_DISTRICT_OUTLINE = None


def _district_outline():
    """The district boundary around the member parcels (Okura Fig. 1a), cached; the older
    traced outline if that file is missing."""
    global _DISTRICT_OUTLINE
    if _DISTRICT_OUTLINE is None:
        import os

        import geopandas as gpd

        from stem.params import DATA

        path = os.path.join(DATA, "district_outline_okura.geojson")
        if not os.path.exists(path):
            path = os.path.join(DATA, "district_outline.geojson")
        _DISTRICT_OUTLINE = gpd.read_file(path)
    return _DISTRICT_OUTLINE


def draw_infrastructure(ax, key=True):
    """Drainage channels, culverts, structures and the pumping station, with a key saying
    what is what (``stem.infrastructure.draw``). The key sits below the map."""
    try:
        from stem import infrastructure
    except Exception:
        return []
    handles = infrastructure.draw(ax, key=False)
    if key and handles:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2,
                  fontsize=6, frameon=False)
    return handles


def figure_field(world, mode: str = "ponding", selected: int | None = None,
                 block: str | None = None) -> Figure:
    """District map: real parcel outlines (not fixed-size markers — with the full
    ~4,000-paddy district these tile edge-to-edge, unlike a small subsample) coloured
    by each paddy's actual value, plus the district boundary and pumping station for
    context. Falls back to a centroid scatter when polygons aren't available, and to
    the legacy grid when there is no cyclic map at all.

    Earlier versions scattered a fixed-size dot per paddy regardless of geometry;
    with several thousand near-regularly-spaced dots of similar size, the Agg
    rasteriser's anti-aliased circle edges beat against each other and produce a
    moire/hatching artifact that has nothing to do with the data. Real polygons
    don't have that problem — adjacent parcels share an edge instead of overlapping.

    ``block`` ("North" / "South") draws only that area's paddies, cyclic and
    non-cyclic, zoomed to them; colours are scaled to that area alone.
    """
    fig = Figure(figsize=(8.6, 7.6), dpi=105)
    fig.patch.set_facecolor("#f7f4ef")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#efeae2")

    vals, title, cmap, label = _values_for_mode(world, mode)
    use_map = hasattr(world, "lon")
    keep = np.ones(world.n, dtype=bool) if block is None else (np.asarray(world.block) == block)
    if block is not None:
        title = f"{block} field · {title}"

    if use_map:
        station = getattr(world, "station_lonlat", None)
        if station is None:
            from stem.infrastructure import STATION_LONLAT as station

        lon, lat = world.lon, world.lat
        geometry = getattr(world, "geometry", None)
        owners = np.asarray(world.owner_id)[keep]
        shown = np.asarray(vals)[keep]

        if block is None:
            try:
                _district_outline().boundary.plot(ax=ax, color="#8a8574", linewidth=1.0, zorder=1)
            except Exception:
                pass  # outline is decorative only

        if geometry is not None:
            import geopandas as gpd

            geoms = geometry.reset_index(drop=True)[keep].reset_index(drop=True)
            if block is not None:
                # Cyclic paddies drain back to the pool; outline them so they stand out.
                cyc_geoms = geometry.reset_index(drop=True)[keep & np.asarray(world.cyclic)]
                cyc_geoms.boundary.plot(ax=ax, color="#1a1a1a", linewidth=0.25, alpha=0.5, zorder=4)
            if mode == "ownership":
                colors = [OWNER_COLORS[int(o) % len(OWNER_COLORS)] for o in owners]
                gpd.GeoDataFrame({"c": colors}, geometry=geoms).plot(
                    ax=ax, color=colors, edgecolor="none", zorder=3
                )
            else:
                vmin, vmax = _color_range(mode, shown)
                gpd.GeoDataFrame({"v": shown}, geometry=geoms).plot(
                    ax=ax, column="v", cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="none", zorder=3
                )
                sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
                sm.set_array([])
                fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03).set_label(label)
        elif mode == "ownership":
            colors = [OWNER_COLORS[int(o) % len(OWNER_COLORS)] for o in owners]
            ax.scatter(lon[keep], lat[keep], c=colors, s=14, alpha=0.92, linewidths=0, zorder=3)
        else:
            vmin, vmax = _color_range(mode, shown)
            sc = ax.scatter(lon[keep], lat[keep], c=shown, cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax),
                             s=14, alpha=0.95, linewidths=0, zorder=3)
            fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03).set_label(label)

        if not draw_infrastructure(ax):
            ax.scatter([station[0]], [station[1]], marker="*", s=280,
                       color="#1a1a1a", edgecolors="white", linewidths=1.0, zorder=6)

        if selected is not None and 0 <= selected < world.n and keep[selected]:
            ax.scatter(
                [lon[selected]],
                [lat[selected]],
                s=90,
                facecolors="none",
                edgecolors="#1a1a1a",
                linewidths=2.0,
                zorder=7,
            )
            for j in world.neighbour_ids(selected):
                ax.plot(
                    [lon[selected], lon[j]],
                    [lat[selected], lat[j]],
                    color="#2b6cb0",
                    lw=1.2,
                    alpha=0.85,
                    zorder=5,
                )
                ax.scatter([lon[j]], [lat[j]], s=40, facecolors="none",
                           edgecolors="#2b6cb0", linewidths=1.4, zorder=6)

        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_aspect("equal", adjustable="datalim")
        if block is not None and keep.any():
            pad_x = 0.04 * float(np.ptp(lon[keep]) or 1e-3)
            pad_y = 0.04 * float(np.ptp(lat[keep]) or 1e-3)
            ax.set_xlim(float(lon[keep].min()) - pad_x, float(lon[keep].max()) + pad_x)
            ax.set_ylim(float(lat[keep].min()) - pad_y, float(lat[keep].max()) + pad_y)
            ax.set_aspect("equal", adjustable="box")  # keep the zoom; resize the box instead
            ax.set_title(f"{title} · day {world.day} · outlined = cyclic")
        else:
            ax.set_title(f"{title} · day {world.day} · ★ = pumping station")
    else:
        data = world.grid(vals)
        im = ax.imshow(data, cmap=cmap, origin="upper")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(label)
        if selected is not None and 0 <= selected < world.n:
            sr, sc = int(world.row[selected]), int(world.col[selected])
            ax.add_patch(Rectangle((sc - 0.5, sr - 0.5), 1, 1, fill=False,
                                   edgecolor="#1a1a1a", linewidth=2.4))
        ax.set_title(f"{title} · day {world.day}")

    fig.tight_layout()
    return fig


def figure_neighbours(world, paddy: int) -> Figure:
    view = world.neighbour_view(paddy)
    pond = view["ponding"]
    owners = view["owners"]
    fig = Figure(figsize=(3.6, 3.4), dpi=110)
    fig.patch.set_facecolor("#f7f4ef")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#efeae2")

    vmax = max(np.nanmax(pond) if np.isfinite(pond).any() else 1.0, 40.0)
    im = ax.imshow(pond, cmap=POND_CMAP, vmin=0, vmax=vmax, origin="upper")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("mm")

    for r in range(3):
        for c in range(3):
            if not np.isfinite(pond[r, c]):
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color="#d9d3c7"))
                continue
            oid = int(owners[r, c])
            ax.text(
                c,
                r,
                f"{pond[r, c]:.0f}\nF{oid}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if pond[r, c] > vmax * 0.45 else "#1a1a1a",
            )
            edge = "#1a1a1a" if r == 1 and c == 1 else "#c45c26"
            ls = "-" if r == 1 and c == 1 else "--"
            ax.add_patch(
                Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=edge,
                          linewidth=2.2 if r == 1 and c == 1 else 1.2, linestyle=ls)
            )

    ax.set_xticks([0, 1, 2], ["W", "·", "E"])
    ax.set_yticks([0, 1, 2], ["N", "·", "S"])
    pid = view.get("paddy", paddy)
    ax.set_title(f"Paddy {pid} · cardinal nbrs")
    fig.tight_layout()
    return fig


def figure_series(world) -> Figure:
    fig = Figure(figsize=(5.2, 2.8), dpi=110)
    fig.patch.set_facecolor("#f7f4ef")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#efeae2")
    if not world.history:
        ax.text(0.5, 0.5, "Run a few days to see trajectories",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    days = [s.day for s in world.history]
    pond = [float(s.ponding_mm.mean()) for s in world.history]
    ir = [float(s.irrigation_mm.mean()) for s in world.history]
    dr = [float(s.drainage_mm.mean()) for s in world.history]
    ax.plot(days, pond, color="#1b4a42", lw=2.0, label="mean ponding")
    ax.plot(days, ir, color="#3a5f8a", lw=1.5, label="mean applied")
    ax.plot(days, dr, color="#b86b2b", lw=1.5, label="mean D")
    if hasattr(world.history[0], "mean_relative_yield"):
        yld = [float(s.mean_relative_yield) * 100 for s in world.history]
        ax2 = ax.twinx()
        ax2.plot(days, yld, color="#5c6b2f", lw=1.2, ls="--", label="Ya/Ym %")
        ax2.set_ylabel("relative yield %")
        ax2.set_ylim(0, 110)
    ax.set_xlabel("day")
    ax.set_ylabel("mm")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Field averages")
    fig.tight_layout()
    return fig


def figure_et(world) -> Figure:
    """ET0 (weather, uncontrollable) -> ETc (potential, Kc(stage)*ET0) -> ETa (actual,
    water-stress-limited) over the season, plus rain. This is the crop-growth driver
    the farmer cannot set directly — only irrigate to try to keep Ks near 1."""
    fig = Figure(figsize=(5.2, 2.8), dpi=110)
    fig.patch.set_facecolor("#f7f4ef")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#efeae2")
    if not world.history:
        ax.text(0.5, 0.5, "Run a few days to see ET develop",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    days = [s.day for s in world.history]
    et0 = [float(s.et0_mm) for s in world.history]
    etc = [float(s.etc_mm) for s in world.history]
    eta = [float(s.eta_mm.mean()) for s in world.history]
    rain = [float(s.rain_mm) for s in world.history]
    ax.bar(days, rain, color="lightsteelblue", width=0.9, alpha=0.6, label="rain (mm, weather)")
    ax.plot(days, et0, color="#8a6b2f", lw=1.1, ls=":", label="ET0 (weather, uncontrollable)")
    ax.plot(days, etc, color="#b86b2b", lw=1.6, label="ETc (potential, Kc·ET0)")
    ax.plot(days, eta, color="#1b4a42", lw=2.0, label="ETa (actual, field mean)")
    ax.set_xlabel("day")
    ax.set_ylabel("mm/d")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)
    ax.set_title("ET over the season: weather in, crop stress out")
    fig.tight_layout()
    return fig


def figure_metrics(eff: dict) -> Figure:
    fig = Figure(figsize=(4.4, 2.6), dpi=110)
    fig.patch.set_facecolor("#f7f4ef")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#efeae2")
    if not eff:
        ax.text(0.5, 0.5, "No season metrics yet", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    labels = ["WUE'%/100", "RUE", "LUE", "Ya/Ym"]
    vals = [
        float(eff.get("WUE_prime_pct", 0.0)) / 100.0,
        float(eff.get("RUE_all", 0.0)),
        float(eff.get("LUE", 0.0)),
        float(eff.get("relative_yield", 0.0)),
    ]
    pins = [None, eff.get("paper_RUE_all"), eff.get("paper_LUE"), None]
    xs = np.arange(len(labels))
    ax.bar(xs, vals, color=["#2f6f5e", "#3a5f8a", "#b86b2b", "#5c6b2f"], width=0.55)
    for i, pin in enumerate(pins):
        if pin is not None:
            ax.plot([i - 0.3, i + 0.3], [pin, pin], color="#1a1a1a", lw=1.8)
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, max(max(vals) * 1.25, 0.5))
    ax.set_title("Efficiencies (black = paper pin)")
    fig.tight_layout()
    return fig

# -*- coding: utf-8 -*-
"""Every figure of docs/conceptual_model.tex, drawn from the model's own data and code.

    python docs/figures/validation_runs.py   # once: default seasons vs Okura (data/)
    python docs/figures/make_figures.py      # all figures -> docs/figures/*.png

Data figures read the bundled district (cyclic), the ABM world and the newest
sensitivity batch (outputs/sensitivity/LATEST). Schematics are drawn with matplotlib so
that they change with the text, not with a drawing tool.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

# Reference palette (light mode, print): categorical slots in fixed order, one-hue blue ramp.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
NEUTRAL = "#d9d8d2"        # non-cyclic parcels / inactive
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
WATER = "#86b6ef"
SOIL, PAN, PERCHED = "#e9dcc6", "#c9b48f", "#b7d3f6"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.dpi": 220, "savefig.bbox": "tight", "legend.frameon": False,
})


def save(fig, name):
    path = os.path.join(HERE, name + ".png")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def clean(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=.6)
        ax.set_axisbelow(True)


# --- data figures --------------------------------------------------------------------------

def _world(year=2022):
    from abm.world import SeasonConfig, WaterWorld
    return WaterWorld(SeasonConfig(year=year, days=2, ownership_seed=0))


def fig_district(w):
    from cyclic import infrastructure

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import ScalarFormatter

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    geo = w.geometry
    classes = np.where(~w.cyclic, "non", np.where(w.block == "North", "north", "south"))
    for cls, color in (("non", NEUTRAL), ("north", BLUE), ("south", ORANGE)):
        geo[classes == cls].plot(ax=ax, color=color, edgecolor="white", linewidth=.15)
    for role, color, lw in (("connected_drain", INK2, .8), ("main_canal", INK, 2.6)):
        for feature in infrastructure.channels(role):
            xs, ys = zip(*feature["geometry"]["coordinates"])
            ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=5)
    ax.plot(*infrastructure.STATION_LONLAT, marker="s", ms=8, color=INK, zorder=9)
    ax.annotate("pumping station\nand pool", infrastructure.STATION_LONLAT, xytext=(60, -10),
                textcoords="offset points", fontsize=8, color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=.6))
    n_north = int(((w.block == "North") & w.cyclic).sum())
    n_south = int(((w.block == "South") & w.cyclic).sum())
    leg = [Patch(color=BLUE, label=f"North block, cyclic ({n_north:,} parcels)"),
           Patch(color=ORANGE, label=f"South block, cyclic ({n_south:,} parcels)"),
           Patch(color=NEUTRAL, label=f"non-cyclic ({int((~w.cyclic).sum()):,} parcels)"),
           Line2D([], [], color=INK, lw=2.6, label="main drainage canal"),
           Line2D([], [], color=INK2, lw=.8, label="drains reaching the pool"),
           Line2D([], [], color=INK, marker="s", lw=0, label="Hakusan-Jimbee station")]
    ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.set_xlabel("longitude (°E)")
    ax.set_ylabel("latitude (°N)")
    for axis in (ax.xaxis, ax.yaxis):
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        axis.set_major_formatter(fmt)
    ax.set_aspect(1 / np.cos(np.radians(35.8)))
    ax.tick_params(labelsize=7)
    ax.set_title("The Shiroyama-Jimbee district (Okura et al. 2025, Fig. 1a, on Fude parcels)")
    save(fig, "fig_district")


def fig_recapture(w):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    geo = w.geometry
    geo[~w.cyclic].plot(ax=ax, color=NEUTRAL, edgecolor="white", linewidth=.1)
    share = w.lateral_capture_i
    gc = geo[w.cyclic].to_frame("geometry").assign(share=share[w.cyclic])
    gc.plot(ax=ax, column="share", cmap=cmap, vmin=.25, vmax=1., edgecolor="white", linewidth=.1,
            legend=True, legend_kwds=dict(label="recapture share $\\kappa_i$", shrink=.55))
    ax.plot(*w.station_lonlat, marker="s", ms=7, color=INK)
    ax.set_aspect(1 / np.cos(np.radians(35.8)))
    ax.set_axis_off()
    ax.set_title("(a) recapture share κ$_i$, joint 2021–2022 fit")
    d = w.canal_distance_m[w.cyclic]
    for blk, color in (("North", BLUE), ("South", ORANGE)):
        m = (w.block[w.cyclic] == blk)
        bx.scatter(d[m], share[w.cyclic][m], s=9, color=color, alpha=.55, lw=0, label=f"{blk} block")
    mean = float(np.average(share[w.cyclic], weights=w.area_m2[w.cyclic]))
    bx.axhline(mean, color=INK2, lw=1, ls="--")
    bx.text(bx.get_xlim()[1] if False else d.max(), mean + .02, f"area-weighted mean {mean:.3f}",
            ha="right", fontsize=8, color=INK2)
    bx.set_xlabel("distance to the nearest drain reaching the pool (m)")
    bx.set_ylabel("recapture share")
    bx.legend(loc="upper right", fontsize=8)
    clean(bx)
    bx.set_title("(b) κ$_i$ against drain distance")
    save(fig, "fig_recapture")


def fig_forcing():
    from cyclic import infrastructure, okura
    from abm.world import SeasonConfig, WaterWorld

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.0), sharex=True)
    for year, color in ((2021, BLUE), (2022, ORANGE)):
        w = WaterWorld(SeasonConfig(year=year, days=None, ownership_seed=0))
        days = np.arange(len(w.rain_series))
        axes[0].plot(days, np.cumsum(w.rain_series), color=color, lw=2, label=f"{year}")
        axes[1].plot(days, pd.Series(w.et0_series).rolling(7, min_periods=1).mean(), color=color, lw=2)
        depth = infrastructure.canal_depth(year)
        if depth is not None:
            axes[2].plot(np.arange(len(depth["North"])), depth["North"], color=color, lw=2)
            axes[2].plot(np.arange(len(depth["South"])), depth["South"], color=color, lw=1.2, ls="--")
    axes[2].axhline(infrastructure.normal_canal_depth_m(), color=INK2, lw=.8, ls=":")
    axes[2].text(130, infrastructure.normal_canal_depth_m() - .02, "normal depth", ha="right", va="top",
                 fontsize=7.5, color=INK2)
    axes[0].set_ylabel("cumulative rain (mm)")
    axes[0].legend(loc="upper left")
    axes[0].set_title("Forcing: gauge-corrected rain, reference ET, main-canal depth")
    axes[1].set_ylabel("ET$_0$, 7-day mean (mm d$^{-1}$)")
    axes[2].set_ylabel("canal depth (m)")
    axes[2].set_xlabel("season day (0 = 17 April)")
    axes[2].text(2, axes[2].get_ylim()[1], "solid: North arm   dashed: South arm", va="top", fontsize=7.5, color=INK2)
    for ax in axes:
        clean(ax)
    save(fig, "fig_forcing")


def fig_validation():
    path = os.path.join(HERE, "data", "okura_comparison.csv")
    if not os.path.exists(path):
        print("skip fig_validation: run validation_runs.py first")
        return
    t = pd.read_csv(path)
    t = t[t.scale.str.contains("block") | t.term.isin(["LFV_pump", "dWB", "IR_all", "W"])].copy()
    t["label"] = np.where(t.scale.str.contains("block"), t.scale.str.split(" ").str[0] + " " + t.term, t.term)
    t = t[t.term != "rain"]
    labels = list(dict.fromkeys(t.label))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(labels))
    for k, (year, color, off) in enumerate(((2021, BLUE, -.17), (2022, ORANGE, .17))):
        r = t[t.year == year].set_index("label").reindex(labels)["ratio"]
        ax.barh(y + off, r.values, height=.32, color=color, label=str(year))
    ax.axvline(1, color=INK, lw=1)
    ax.axvspan(.88, 1.12, color=GRID, alpha=.6, zorder=0)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("simulated ÷ Okura et al. (2025)   (1 = exact; shaded band ±12 %)")
    ax.legend(loc="lower right")
    clean(ax, "x")
    ax.set_title("Default season against the measured water balance")
    save(fig, "fig_validation")


def _latest_batch():
    from abm import sensitivity as sens
    path = sens.latest()
    return None if path is None else sens.load(path)


def fig_trait_effects():
    d = _latest_batch()
    if d is None:
        print("skip fig_trait_effects: no sensitivity batch")
        return
    from abm import sensitivity as sens
    te = sens.trait_effects(d["runs"])
    e = te[te.metric == "request_m3"].pivot_table(index="trait", columns="scenario", values="rel_pct")
    ci = te[te.metric == "request_m3"].pivot_table(index="trait", columns="scenario", values="ci95")
    base = te[te.metric == "request_m3"].pivot_table(index="trait", columns="scenario", values="base")
    short = "baseline" + sens.SHORTAGE
    order = e["baseline"].abs().sort_values().index
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y = np.arange(len(order))
    for col, color, off, label in (("baseline", BLUE, .17, "full pump capacity"),
                                   (short, ORANGE, -.17, "pump shortage (25 %)")):
        if col not in e:
            continue
        err = 100 * ci.loc[order, col] / base.loc[order, col].abs()
        ax.barh(y + off, e.loc[order, col], xerr=err, height=.32, color=color, label=label,
                error_kw=dict(ecolor=INK2, lw=.8, capsize=2))
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, [t.replace("_", " ") for t in order])
    ax.set_xlabel("effect on season irrigation demand, high − low (% of base run); 95 % CI over 3 seeds")
    ax.legend(loc="lower right")
    clean(ax, "x")
    ax.set_title("Which traits move the demand the model generates")
    save(fig, "fig_trait_effects")


def fig_scenario_demand():
    d = _latest_batch()
    if d is None:
        return
    runs, daily = d["runs"], d["daily"]
    scen = runs.get("scenario", runs.weather)
    base = runs[runs.trait.isna()].assign(scenario=scen[runs.trait.isna()])
    show = [("baseline", INK, 2.2), ("drought", ORANGE, 1.6), ("wet", BLUE, 1.6), ("heatwave", YELLOW, 1.6),
            ("baseline + pump shortage", AQUA, 1.6)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for name, color, lw in show:
        ids = base[base.scenario == name].run_id
        x = daily[daily.run_id.isin(ids)].groupby("day")["request_m3"].mean()
        if len(x):
            ax.plot(x.index, x.rolling(7, min_periods=1).mean() / 1000, color=color, lw=lw, label=name)
    ax.set_xlabel("season day (0 = 17 April)")
    ax.set_ylabel("district demand, 7-day mean (10$^3$ m$^3$ d$^{-1}$)")
    ax.legend(fontsize=8, ncol=2)
    clean(ax)
    ax.set_title("Generated irrigation demand under weather and supply scenarios (2022, 3 seeds)")
    save(fig, "fig_scenario_demand")


# --- schematics ------------------------------------------------------------------------------

def box(ax, x, y, w, h, text, fc="#f3f2ee", ec=AXIS, size=8.5, bold=False, color=INK, round_=True):
    style = "round,pad=0.02,rounding_size=0.06" if round_ else "square,pad=0.0"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, p, q, text=None, color=INK2, size=7.5, rad=0.0, lw=1.2, off=(0, 0), ha="center", style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        mx, my = (p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1]
        ax.text(mx, my, text, fontsize=size, color=color, ha=ha, va="center")


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()
    return fig, ax


def fig_perceptual():
    """Cross-section of two paddies, a farm drain, the main canal, the pool and the lake."""
    fig, ax = canvas(10.4, 5.6, (0, 22), (0, 11.2))
    for x0 in (0.5, 6.7):
        ax.add_patch(Rectangle((x0, 6.0), 6.0, .7, fc=WATER, ec="none"))           # pond
        ax.add_patch(Rectangle((x0, 4.9), 6.0, 1.1, fc=SOIL, ec="none"))            # topsoil / root zone
        ax.add_patch(Rectangle((x0, 4.55), 6.0, .35, fc=PAN, ec="none"))            # plow pan
        ax.add_patch(Rectangle((x0, 2.6), 6.0, 1.95, fc="#f1ead9", ec="none"))      # subsoil
        ax.add_patch(Rectangle((x0, 2.6), 6.0, 1.0, fc=PERCHED, ec="none"))         # perched table
        ax.add_patch(Rectangle((x0 - .15, 6.0), .15, 1.0, fc=PAN, ec="none"))       # bund
    ax.add_patch(Rectangle((12.7, 6.0), .15, 1.0, fc=PAN, ec="none"))
    ax.add_patch(Rectangle((13.1, 3.0), 1.0, 3.0, fc="#f1ead9", ec=AXIS))          # farm drain trench
    ax.add_patch(Rectangle((13.1, 3.0), 1.0, 1.0, fc=WATER, ec="none"))
    ax.add_patch(Polygon([[15.0, 6.5], [15.5, 2.6], [16.8, 2.6], [17.3, 6.5]], fc="#f1ead9", ec=AXIS))
    ax.add_patch(Polygon([[15.35, 4.0], [15.5, 2.6], [16.8, 2.6], [16.95, 4.0]], fc=WATER, ec="none"))
    box(ax, 18.8, 3.6, 2.6, 1.6, "pool\n18 × 68 × 4 m", fc="#e6f0fc", ec=BLUE, size=8)
    box(ax, 18.8, .4, 2.6, 1.3, "Lake\nKita-Imbanuma", fc="#e6f0fc", ec=BLUE, size=8)
    ax.text(3.5, 7.15, "paddy i", ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.text(9.7, 7.15, "neighbour j", ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.text(13.6, 2.3, "farm drain", ha="center", fontsize=7.5, color=INK2)
    ax.text(16.15, 6.85, "main drainage canal", ha="center", fontsize=7.5, color=INK2)
    for y, label in ((6.35, "pond h"), (5.45, "topsoil + root zone"), (4.72, "plow pan"), (3.1, "perched table")):
        ax.text(.6, y, label, fontsize=7, color=INK2, va="center")
    arrow(ax, (1.2, 9.4), (1.2, 6.75), "rain R", ha="left", off=(.15, .55))
    ax.plot([21.1, 21.1, 2.6], [5.2, 10.0, 10.0], color=BLUE, lw=1.2)
    arrow(ax, (2.6, 10.0), (2.6, 6.75), None, color=BLUE)
    ax.text(11.4, 10.3, "irrigation IR: pumped from the pool, applied ≤ what was sent", fontsize=7.5,
            color=BLUE, ha="center")
    arrow(ax, (5.6, 6.75), (5.6, 9.0), "ET (dual $K_c$)", ha="left", off=(.15, .45))
    arrow(ax, (12.4, 6.3), (13.4, 5.5), None)
    ax.text(12.95, 7.6, "D: discharge, weir spill,\noverflow, outlet leak", fontsize=7, color=INK2, ha="left")
    arrow(ax, (3.0, 6.0), (3.0, 4.0), "percolation\n$K_{pan}h/h_{ref}$", ha="left", off=(.15, -.25), size=7)
    arrow(ax, (6.1, 3.1), (7.2, 3.1), None, style="<|-|>")
    ax.text(6.65, 2.25, "Darcy\nexchange", fontsize=7, color=INK2, ha="center")
    arrow(ax, (12.2, 3.2), (13.2, 3.5), None)
    ax.text(12.0, 1.5, "Hooghoudt lateral\nto the farm drain", fontsize=7, color=INK2, ha="center")
    arrow(ax, (9.0, 2.6), (9.0, 1.2), "deep\npercolation", ha="left", off=(.15, 0), size=7)
    arrow(ax, (14.1, 3.5), (15.6, 3.4), None)
    arrow(ax, (16.8, 3.3), (18.8, 4.2), None)
    ax.text(17.5, 5.9, "returns: routed cyclic D\n+ recaptured lateral", fontsize=7, color=INK2, ha="left")
    arrow(ax, (19.4, 3.6), (19.4, 1.7), None, color=BLUE)
    ax.text(19.25, 2.65, "outflow O,\nspill", fontsize=7, color=BLUE, ha="right", va="center")
    arrow(ax, (20.8, 1.7), (20.8, 3.6), None, color=BLUE)
    ax.text(20.95, 2.65, "gate W", fontsize=7, color=BLUE, ha="left", va="center")
    ax.text(16.15, 1.7, "high canal → backwater:\nlow parcels' drains held", ha="center", fontsize=7, color=ORANGE)
    ax.set_title("Perceptual model: how water moves through a cyclic paddy district", loc="left")
    save(fig, "fig_perceptual")


def fig_stores():
    """Conceptual stores of one parcel and the district stores they connect to."""
    fig, ax = canvas(9.0, 5.8, (0, 18), (-.7, 11))
    box(ax, 1, 8.2, 4.2, 1.4, "Pond  $h$", fc="#e6f0fc", ec=BLUE, bold=True)
    box(ax, 1, 5.9, 4.2, 1.4, "Surface layer\n(depletion, TEW 25 mm)", fc="#f6efe2", ec=PAN)
    box(ax, 1, 3.6, 4.2, 1.4, "Root zone\n(depletion, TAW 100 mm)", fc="#f6efe2", ec=PAN)
    box(ax, 1, 1.3, 4.2, 1.4, "Perched table\n(above farm-drain level)", fc="#e6f0fc", ec=BLUE)
    ax.add_patch(Rectangle((.6, .9), 5.0, 9.2, fc="none", ec=INK2, ls="--", lw=.8))
    ax.text(.6, -.35, "dashed: parcel i, one control volume (mm over its area)", fontsize=7.5, color=INK2)
    box(ax, 7.6, 1.3, 3.0, 1.4, "Neighbour\nparcels (4)", fc="#f3f2ee")
    box(ax, 12.2, 5.9, 2.8, 1.4, "Drains\n(in transit)", fc="#f3f2ee")
    box(ax, 12.2, 3.6, 2.8, 1.4, "Block storage\nΔWB", fc="#f3f2ee")
    box(ax, 15.3, 8.2, 2.4, 1.4, "Pool", fc="#e6f0fc", ec=BLUE, bold=True)
    box(ax, 15.3, 1.3, 2.4, 1.4, "Lake\n(boundary)", fc="#e6f0fc", ec=BLUE)
    arrow(ax, (3.1, 10.6), (3.1, 9.65), "R + IR", ha="left", off=(.15, .1))
    arrow(ax, (5.2, 9.1), (6.6, 9.9), "E", ha="left", off=(.35, .2))
    arrow(ax, (5.2, 4.4), (6.6, 5.2), "T", ha="left", off=(.35, .2))
    arrow(ax, (2.2, 8.2), (2.2, 7.35), "I", ha="left", off=(.12, 0))
    arrow(ax, (2.2, 5.9), (2.2, 5.05), "I (rest)", ha="left", off=(.12, 0))
    arrow(ax, (2.2, 3.6), (2.2, 2.75), "I (rest)", ha="left", off=(.12, 0))
    arrow(ax, (5.2, 2.0), (7.6, 2.0), "Darcy $Q_{ij}$", style="<|-|>", off=(0, .35))
    arrow(ax, (5.2, 8.9), (12.2, 6.8), "D (×(1 − backwater))", off=(0, .55))
    arrow(ax, (15.0, 6.9), (15.6, 8.2), "routed,\ncyclic only", ha="right", off=(-.35, .15))
    arrow(ax, (5.2, 1.6), (12.2, 4.0), None)
    ax.text(8.8, 3.55, "Hooghoudt lateral q: (1 − κ$_i$)", fontsize=7.5, color=INK2, ha="center")
    arrow(ax, (5.2, 2.5), (15.3, 8.6), None, rad=-.25, color=BLUE)
    ax.text(10.4, 4.85, "κ$_i$ · q → pool\n(same-day recapture)", fontsize=7.5, color=BLUE)
    arrow(ax, (1.6, 1.3), (1.6, .2), "deep", ha="left", off=(.12, 0))
    arrow(ax, (15.7, 8.2), (15.7, 2.75), "O, spill", ha="left", off=(.1, 0), color=BLUE)
    arrow(ax, (17.3, 2.75), (17.3, 8.2), "W", ha="left", off=(.1, 0), color=BLUE)
    arrow(ax, (15.3, 9.3), (5.6, 10.6), "IR (sent ≤ pumps)", rad=.08, color=BLUE, off=(0, .4))
    ax.set_title("Conceptual model: stores (boxes) and fluxes (arrows)", loc="left")
    save(fig, "fig_stores")


def _flow(name, title, steps, w=9.4, notes=None, colors=None):
    n = len(steps)
    fig, ax = canvas(w, .62 * n + .6, (0, 14), (0, n * 1.25 + .3))
    for k, text in enumerate(steps):
        y = (n - 1 - k) * 1.25 + .2
        fc = (colors or {}).get(k, "#f3f2ee")
        box(ax, .2, y, 8.4, .95, f"{k + 1}. {text}", fc=fc, size=8)
        if notes and notes.get(k):
            ax.text(8.85, y + .475, notes[k], fontsize=7.3, color=INK2, va="center")
        if k < n - 1:
            arrow(ax, (4.4, y), (4.4, y - .3))
    ax.set_title(title, loc="left")
    save(fig, name)


def fig_physics_order():
    _flow("fig_physics_order", "Procedural order of one physical day (every parcel at once)", [
        "Rain R and applied irrigation IR land on the pond",
        "Dual-K$_c$ ET demand; met from pond, then topsoil",
        "Farmer discharge; pond above the weir crest leaves",
        "Bund overflow; outlet leak (13 % of the pond)",
        "Backwater: share of D held on low parcels",
        "Plow-pan percolation I = K$_{pan}$ h / h$_{ref}$",
        "Fill surface layer → root zone → perched table",
        "Darcy exchange of perched water with 4 neighbours",
        "Hooghoudt lateral flow; deep percolation",
        "Recapture κ$_i$·lateral → pool; rest → ΔWB",
        "Route cyclic D to the pool (travel time)",
        "Settle pool: sends, returns, outflow, gate, spill",
    ], notes={1: "stress = demand not supplied", 4: "Okura Fig. 3 canal depth + GSI DEM",
              9: "cyclic parcels; non-cyclic → lake", 10: "route length / 0.2 m s$^{-1}$",
              11: "lake is an unlimited boundary"})


def fig_abm_day():
    fig, ax = canvas(9.4, 3.7, (0, 22), (0, 8.4))
    lanes = [("Station agent", 6.4), ("Farmer agents", 4.2), ("Physical model", 2.0)]
    for name, y in lanes:
        ax.add_patch(Rectangle((0, y - .1), 22, 2.0, fc="#f9f9f7", ec=GRID))
        ax.text(.3, y + .9, name, fontsize=8.5, color=INK, fontweight="bold", va="center")
    box(ax, 4.6, 6.75, 4.6, 1.3, "observe pool, forecast,\nfield deficit → dispatch\n(gate W, outflow O, send s$_i$)", size=7.5, fc="#e6f0fc", ec=BLUE)
    box(ax, 9.6, 4.55, 4.8, 1.3, "each farmer: observe own\nparcels → request,\napply ≤ s$_i$, drain, set weir", size=7.5, fc="#fde9e0", ec=ORANGE)
    box(ax, 15.0, 2.35, 3.6, 1.3, "advance physics\n(Fig. order) and\nsettle the pool", size=7.5)
    box(ax, 19.0, 4.55, 2.8, 1.3, "record score;\nlearn on next\ncheck day", size=7.5, fc="#fde9e0", ec=ORANGE)
    box(ax, 4.6, 4.55, 3.6, 1.3, "read neighbours'\nstrategy + score\n(yesterday's snapshot)", size=7.5, fc="#fde9e0", ec=ORANGE)
    arrow(ax, (9.2, 7.2), (10.6, 5.85))
    arrow(ax, (14.4, 5.0), (15.2, 3.65))
    arrow(ax, (18.6, 3.0), (19.0, 4.9))
    arrow(ax, (8.2, 5.2), (9.6, 5.2))
    ax.text(11, .6, "One model day = one calendar day; 131 days from 17 April. An external optimiser can replace the station lane.",
            ha="center", fontsize=7.8, color=INK2)
    ax.set_title("ODD+D process overview and scheduling", loc="left")
    save(fig, "fig_abm_day")


def fig_farmer_cycle():
    _flow("fig_farmer_cycle", "Farmer decision cycle (every day, for every owned parcel)", [
        "Check day? (always on days off; on a working day if home and due)",
        "Target T = m·h*(t') + 8(1 − r), drifted toward neighbours",
        "Bounds: L = T − 25 d,   U = min(T + 20 o, bund − 20)",
        "Will the pond fall below L before the next check?",
        "Request refill to U − forecast rain (+ trickle if needed)",
        "Add flow-through on hot days / while away",
        "Cut the refill by 25 %·c if the canal is short",
        "Set weir crest at U",
        "Hard rules: apply ≤ sent, drain ≤ stock, no-drain holds",
    ], notes={0: "no → inlet and weir stay as left", 1: "t' = t + round(4 × calendar lead)",
              3: "no → request nothing", 4: "water accounting, forecast reliance",
              6: "c = cooperation", 8: "herbicide 7 d ≥ 30 mm; top-dressing 4 d"},
          colors={0: "#fde9e0", 3: "#fde9e0"})


def fig_farmer_states():
    fig, ax = canvas(7.2, 3.6, (0, 16), (0, 8))
    states = {"Routine": (1.0, 5.2), "Wait": (6.2, 5.2), "Irrigating": (11.4, 5.2), "Waiting_For_Water": (6.2, 1.2)}
    for name, (x, y) in states.items():
        box(ax, x, y, 3.8 if name != "Waiting_For_Water" else 4.6, 1.3, name, fc="#fde9e0", ec=ORANGE, bold=True)
    arrow(ax, (4.8, 6.1), (6.2, 6.1), "check day", off=(0, .6))
    arrow(ax, (6.2, 5.6), (4.8, 5.6), "not a\ncheck day", off=(0, -.85))
    arrow(ax, (10.0, 6.1), (11.4, 6.1), "apply > 0", off=(0, .6))
    arrow(ax, (11.4, 5.6), (10.0, 5.6), "no refill\nneeded", off=(0, -.85))
    arrow(ax, (8.1, 5.2), (8.1, 2.5), "request > 0,\nnothing sent", ha="left", off=(.15, 0))
    arrow(ax, (10.8, 2.0), (13.0, 5.2), "water sent", ha="left", off=(.2, -.3))
    ax.text(.2, .3, "State is recorded after each day's decision; between check days the inlet and weir stay as left.",
            fontsize=7.5, color=INK2)
    ax.set_title("Farmer states", loc="left")
    save(fig, "fig_farmer_states")


def fig_station():
    _flow("fig_station", "Station agent (receding-horizon, reuse-first)", [
        "Forecast tomorrow: rain R$_1$, ET demand E$_1$ (6-day forecast)",
        "Least send per paddy: s = (max(E$_1$ − R$_1$, 0) + 3)/ρ + E$_0$ − h$_0$ − R$_0$",
        "Clip to 60 mm d$^{-1}$ and the bund",
        "Free water = pool + forecast returns (cyclic D + recaptured lateral)",
        "Gate W = min(shortfall × 1.05, 150,000 m$^3$)",
        "Too little? scale sends down;  surplus? bank it in cyclic paddies",
        "Flood: R$_0$ + R$_1$ > 40 mm → pump pool to the lake (O)",
        "World: pumps ≤ 3.304 m$^3$ s$^{-1}$; ration nearest the pool first",
    ], notes={1: "ρ = (1 − leak)(1 − K$_{pan}$/h$_{ref}$)", 4: "lake opened only for the shortfall",
              7: "physical limit, any station"}, colors={6: "#e6f0fc"})


def fig_classes():
    fig, ax = canvas(9.4, 5.0, (0, 22), (0, 11.6))
    def cls(x, y, w, title, lines, fc="#f3f2ee", ec=AXIS):
        h = .55 * (len(lines) + 1) + .3
        ax.add_patch(Rectangle((x, y - h), w, h, fc=fc, ec=ec))
        ax.add_patch(Rectangle((x, y - .7), w, .7, fc=ec, ec=ec, alpha=.35))
        ax.text(x + .15, y - .35, title, fontsize=8.5, fontweight="bold", va="center", color=INK)
        for k, line in enumerate(lines):
            ax.text(x + .15, y - 1.0 - .55 * k, line, fontsize=7, va="center", color=INK2, family="monospace")
        return (x, y - h, w, h)
    cls(.3, 11.3, 6.6, "DistrictModel (mesa.Model)", ["world: WaterWorld", "farmer_agents: {id: FarmerMesaAgent}",
        "station_agent: StationMesaAgent", "step(station_action=None)"], fc="#f9f9f7")
    cls(8.0, 11.3, 6.4, "WaterWorld", ["soil: State (4 stores × 4,002)", "lateral_capture_i, edges",
        "drain routes, backwater, pool", "apply_station(), step()"], fc="#e6f0fc", ec=BLUE)
    cls(15.5, 11.3, 6.2, "cyclic.physics", ["Physics (constants)", "advance(): one parcel day",
        "dual K$_c$, supply_et_demand()"], fc="#e6f0fc", ec=BLUE)
    cls(.3, 5.6, 6.6, "BaseFarmer", ["traits (12), days_off", "plan(): target, bounds, refill",
        "bounds_mm(), checks_today() … hooks", "enforce(): hard Constraints"], fc="#fde9e0", ec=ORANGE)
    cls(8.0, 5.6, 6.4, "PersonalityFarmer", ["learning_rate, social_influence", "learn(), record_outcome()",
        "7 archetypes (+ user farmers)"], fc="#fde9e0", ec=ORANGE)
    cls(15.5, 5.6, 6.2, "StationAgent", ["decide(obs) → W, O, sends", "MPCStation (default)",
        "quota, adaptive, level, closed"], fc="#f3f2ee")
    arrow(ax, (6.9, 10.4), (8.0, 10.4), "1", off=(0, .25))
    arrow(ax, (14.4, 10.4), (15.5, 10.4), "uses", off=(0, .25))
    arrow(ax, (3.6, 8.3), (3.6, 5.6), "1..n (via FarmerMesaAgent)", ha="left", off=(.15, 0))
    arrow(ax, (6.9, 4.4), (8.0, 4.4), None, style="<|-")
    ax.text(7.45, 3.85, "inherits", ha="center", fontsize=7, color=INK2)
    arrow(ax, (6.0, 8.3), (17.5, 5.6), "1 (via StationMesaAgent)", rad=-.12, off=(1.0, .5))
    ax.set_title("Classes of the implementation", loc="left")
    save(fig, "fig_classes")


def main(which=None):
    w = _world(2022)
    jobs = {
        "district": lambda: fig_district(w), "recapture": lambda: fig_recapture(w), "forcing": fig_forcing,
        "validation": fig_validation, "trait_effects": fig_trait_effects, "scenario_demand": fig_scenario_demand,
        "perceptual": fig_perceptual, "stores": fig_stores, "physics_order": fig_physics_order,
        "abm_day": fig_abm_day, "farmer_cycle": fig_farmer_cycle, "farmer_states": fig_farmer_states,
        "station": fig_station, "classes": fig_classes,
    }
    for name, job in jobs.items():
        if which and name not in which:
            continue
        job()


if __name__ == "__main__":
    main(sys.argv[1:] or None)

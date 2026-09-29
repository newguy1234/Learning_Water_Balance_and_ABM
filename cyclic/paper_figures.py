"""Okura et al. paper-style graphs for the Field over time page.

Recreates the published time-series / water-balance views from the digitised
Fig. 6 / Fig. 8 files and the observed block series (same sources as imitation):

    Fig. 4   rainfall and ET (Table 2 monthly shape)
    Fig. 5   ADCP-style block drainage (North / South)
    Fig. 6   monthly water balance bars (North / South)
    Fig. 8   irrigation to North and South + lake withdrawal
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from . import data, imitation, okura
from .paper_plots import load_fig6, load_fig8

PAPER_VIEWS = (
    "Fig. 8 — IR North / South + lake W",
    "Fig. 6 — monthly water balance",
    "Fig. 4 — rainfall and ET",
    "Fig. 5 — ADCP drainage North / South",
)


def fig8_irrigation(year, model_blocks=None, model_station=None):
    """Fig. 8: paper IR N/S + W vs this-run **cyclic** IR (CSV / Table 4 scale).

    Digitised Fig. 8 is cyclic-only. IR_all (= cyclic + NCB, Table 5 scale) is
    drawn on a separate panel so it is not mistaken for a hard-copy miss.
    """
    paper = load_fig8(year)
    paper_n = np.asarray(paper["irrigation_North_m3"], float)
    paper_s = np.asarray(paper["irrigation_South_m3"], float)
    paper_all = paper_n + paper_s
    paper_w = np.asarray(paper["withdrawal_m3"], float)

    fig = Figure(figsize=(16, 12.5))
    ax_ns = fig.add_subplot(4, 1, 1)
    ax_all = fig.add_subplot(4, 1, 2, sharex=ax_ns)
    ax_irall = fig.add_subplot(4, 1, 3, sharex=ax_ns)
    ax_w = fig.add_subplot(4, 1, 4, sharex=ax_ns)

    ax_ns.plot(paper["date"], paper_n / 1000.0, color="C0", lw=1.8,
               label="holdout paper — IR North (cyclic)")
    ax_ns.plot(paper["date"], paper_s / 1000.0, color="C1", lw=1.8,
               label="holdout paper — IR South (cyclic)")

    model_n = model_s = model_dates = None
    if model_blocks is not None and "irrigation_m3" in model_blocks.columns:
        for block, color in (("North", "C0"), ("South", "C1")):
            part = model_blocks[model_blocks["block"] == block].sort_values("date")
            if part.empty:
                continue
            ax_ns.plot(part["date"], part["irrigation_m3"] / 1000.0,
                       color=color, lw=1.2, alpha=0.9, ls=":",
                       label=f"this run — cyclic IR {block}")
            if model_dates is None and "date" in part.columns:
                model_dates = part["date"]
            vals = np.asarray(part["irrigation_m3"], float)
            if block == "North":
                model_n = vals
            else:
                model_s = vals

    ax_ns.set_ylabel("×10³ m³/d")
    ax_ns.set_title(f"Fig. 8 — cyclic IR North / South only ({year})")
    ax_ns.legend(fontsize=8, ncol=2, loc="upper right")
    ax_ns.grid(True, alpha=0.25)

    ax_all.plot(paper["date"], paper_all / 1000.0, color="0.25", lw=2.0,
                label="paper CSV — cyclic IR (N+S)")
    if model_n is not None and model_s is not None and model_dates is not None:
        n = min(len(model_n), len(model_s), len(np.asarray(model_dates)))
        ax_all.plot(np.asarray(model_dates)[:n], (model_n[:n] + model_s[:n]) / 1000.0,
                    color="C3", lw=1.4, alpha=0.9, ls=":",
                    label="this run — cyclic IR (N+S)")
    ax_all.set_ylabel("×10³ m³/d")
    ax_all.set_title("Aggregate cyclic IR (must match Fig. 8 CSV; Table 4 scale)")
    ax_all.legend(fontsize=9, loc="upper right")
    ax_all.grid(True, alpha=0.25)

    # Separate panel: IR_all is larger by design (adds non-cyclic supply).
    if model_station is not None and "IR_all_North_m3" in getattr(model_station, "columns", []):
        st = model_station.sort_values("date") if "date" in model_station.columns else model_station
        dates = st["date"] if "date" in st.columns else paper["date"].to_numpy()[:len(st)]
        ax_irall.plot(dates, np.asarray(st["IR_all_North_m3"], float) / 1000.0,
                      color="C0", lw=1.2, ls="-.",
                      label="this run — IR_all North (= cyclic + NCB)")
        ax_irall.plot(dates, np.asarray(st["IR_all_South_m3"], float) / 1000.0,
                      color="C1", lw=1.2, ls="-.",
                      label="this run — IR_all South (= cyclic + NCB)")
        if model_n is not None and model_s is not None and model_dates is not None:
            n = min(len(model_n), len(model_s), len(np.asarray(model_dates)))
            ax_irall.plot(np.asarray(model_dates)[:n], model_n[:n] / 1000.0,
                          color="C0", lw=0.8, alpha=0.5, ls=":",
                          label="cyclic North (Fig. 8 scale)")
            ax_irall.plot(np.asarray(model_dates)[:n], model_s[:n] / 1000.0,
                          color="C1", lw=0.8, alpha=0.5, ls=":",
                          label="cyclic South (Fig. 8 scale)")
    ax_irall.set_ylabel("×10³ m³/d")
    ax_irall.set_title("IR_all (Table 5 scale) — higher than Fig. 8 by design, not a miss")
    ax_irall.legend(fontsize=8, ncol=2, loc="upper right")
    ax_irall.grid(True, alpha=0.25)

    ax_w.plot(paper["date"], paper_w / 1000.0, color="C2", lw=1.6, ls="--",
              label="holdout paper — W (lake)")
    if model_station is not None and "W_m3" in model_station.columns:
        st = model_station.sort_values("date") if "date" in model_station.columns else model_station
        dates = st["date"] if "date" in st.columns else paper["date"].to_numpy()[:len(st)]
        w_true = np.asarray(st["W_m3"], float)
        ax_w.plot(dates, w_true / 1000.0, color="C2", lw=1.2, alpha=0.85, ls=":",
                  label="this run — W")
    ax_w.set_ylabel("×10³ m³/d")
    ax_w.set_title("Lake withdrawal W")
    ax_w.legend(fontsize=9, loc="upper right")
    ax_w.grid(True, alpha=0.25)

    fig.suptitle(
        f"Fig. 8 — cyclic IR hard-copy (solid≈dotted) · IR_all is a separate, larger quantity\n"
        f"Surface cyclic IR/W = digitised Fig. 8; underground ratios are the claim",
        fontsize=11, y=1.01)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig5_adcp_drainage(year, model_blocks=None):
    """Fig. 5 style: daily drainage at North and South (ADCP / observed block D)."""
    blocks, _, source = imitation.load_observed(year)
    fig = Figure(figsize=(16, 8.5))
    for i, block in enumerate(okura.BLOCKS):
        ax = fig.add_subplot(2, 1, i + 1)
        part = blocks[blocks["block"] == block].sort_values("date")
        ax.plot(part["date"], part["drainage_m3"] / 1000.0,
                color="black", lw=1.8, label=f"observed ADCP / Table series — D {block}")
        if model_blocks is not None:
            sim = model_blocks[model_blocks["block"] == block].sort_values("date")
            ax.plot(sim["date"], sim["drainage_m3"] / 1000.0,
                    color="C3", lw=1.2, label=f"this run — D {block}")
        ax.set_ylabel("×10³ m³/d")
        ax.set_title(f"Fig. 5 — {block} block drainage ({year})")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"ADCP-style drainage (source: {source})", fontsize=11, y=1.01)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig4_rain_et(year):
    """Fig. 4 style: daily rainfall and ET (Table 2 monthly totals, paper weather shape)."""
    wx = data.weather(year).reindex(okura.season_dates(year))
    dates = okura.season_dates(year)
    month = dates.month.to_numpy()
    rain = np.zeros(len(dates))
    et = np.zeros(len(dates))
    for m in okura.MONTHS:
        sel = month == m
        shape_r = np.clip(np.nan_to_num(wx["rain_mm"].to_numpy()[sel]), 0.0, None)
        shape_e = np.clip(np.nan_to_num(wx["et0_mm"].to_numpy()[sel]), 0.0, None)
        rain[sel] = okura._spread(shape_r, okura.TABLE2_MM[year]["rain"][m])
        et[sel] = okura._spread(shape_e, okura.TABLE2_MM[year]["et"][m])
    fig = Figure(figsize=(16, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(dates, rain, width=0.9, color="steelblue", alpha=0.7, label="rainfall (mm/d)")
    ax.plot(dates, et, color="darkorange", lw=1.8, label="ET₀ / ETc shape (mm/d)")
    ax.set_ylabel("mm/d")
    ax.set_title(f"Fig. 4 — Rainfall and evapotranspiration ({year})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig6_water_balance(year):
    """Fig. 6 style: monthly IR, R, ET, D, Lf for North and South (mm/d averages)."""
    frame = load_fig6(year)
    months = sorted(frame["month"].unique())
    labels = [pd.Timestamp(year=int(year), month=int(m), day=1).strftime("%b") for m in months]
    quantities = [
        ("irrigation", "IR", "C0"),
        ("rainfall", "R", "C1"),
        ("et", "ET", "C2"),
        ("discharge", "D", "C3"),
        ("lateral", "Lf", "C4"),
    ]
    fig = Figure(figsize=(16, 8.5))
    x = np.arange(len(months))
    width = 0.15
    for col, block in enumerate(okura.BLOCKS):
        ax = fig.add_subplot(1, 2, col + 1)
        part = frame[frame["block"] == block].set_index("month").reindex(months)
        for i, (col_name, label, color) in enumerate(quantities):
            vals = part[col_name].to_numpy(dtype=float)
            ax.bar(x + (i - 2) * width, vals, width=width, color=color, label=label)
        t4 = okura.TABLE4_MM[(block, year)]
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("mm/d (monthly mean)")
        ax.set_title(
            f"Fig. 6 — {block}  ·  Table 4 season: "
            f"IR {t4['irrigation']:.0f}  R {t4['rain']:.0f}  "
            f"ET {t4['et']:.0f}  D {t4['drainage']:.0f}  Lf {t4['lateral']:.0f} mm")
        ax.axhline(0.0, color="black", lw=0.6)
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def paper_figure(view, year, model_blocks=None, model_station=None):
    """Dispatch one paper-style figure by view name."""
    if view.startswith("Fig. 8"):
        return fig8_irrigation(year, model_blocks=model_blocks, model_station=model_station)
    if view.startswith("Fig. 6"):
        return fig6_water_balance(year)
    if view.startswith("Fig. 4"):
        return fig4_rain_et(year)
    if view.startswith("Fig. 5"):
        return fig5_adcp_drainage(year, model_blocks=model_blocks)
    return fig8_irrigation(year, model_blocks=model_blocks, model_station=model_station)

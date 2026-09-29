"""The field over time: Play steps a day and the maps and totals update.

    FieldTab(records)                  the interface tab (records: {label: run record})
    season_arrays(results)             a finished season as day x paddy arrays
    frame_figure(arrays, day, title)   one day's picture
    save_gif(arrays, title, path)      the whole season as an animated GIF

Play shows, changing each day:

    topsoil volume      evaporable topsoil stock as m³ on each paddy
    ET need missing     satellite ET the topsoil could not supply (mm/d)
    into the ground     water the paddy puts into the soil that day (mm/d)
    lateral sent        leftover sent sideways toward the drains (m³/d)
    toward main drain   cyclic D returning to the pumping-station (m³/d)
    non-cyclic D        non-cyclic drainage leaving to the lake / Tone (m³/d)
    W / O               lake withdrawal and same-day return to the lake (m³/d)
    ncb drainage stock  non-cyclic D accumulated out of reuse

The station pool is only a small same-day transit buffer (not shown as a stock).

Colour scales stay fixed over the season. Each map prints today's district total,
so a change is readable even when colours move slowly.
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import solara
from matplotlib.figure import Figure

from . import data, imitation, okura

OUTPUTS = os.path.join(data.PROJECT, "outputs")
MAP_PANELS = [
    ("topsoil_m3", "topsoil volume stock (m³)", "YlOrBr"),
    ("need", "ET need missing (mm/d)", "Reds"),
    ("into_ground_m3", "into ground soil (m³/d)", "Greens"),
    ("lateral_m3", "lateral flow sent (m³/d)", "BuPu"),
    ("canal_m3", "D back to the pool (m³/d)", "PuRd"),
    ("ncb_m3", "non-cyclic D to the lake (m³/d)", "Oranges"),
    ("pond_m3", "pond stock (m³)", "Blues"),
    ("stay_m3", "stays in the ground (m³/d)", "Greys"),
]
PANELS = {key: (label, cmap) for key, label, cmap in MAP_PANELS}
MAP_LABELS = [label for _key, label, _cmap in MAP_PANELS]
MAP_KEY_BY_LABEL = {label: key for key, label, _cmap in MAP_PANELS}
STOCK_SERIES = [
    ("topsoil_m3", "topsoil volume", "tab:orange"),
    ("pond_m3", "pond volume", "tab:blue"),
    ("ncb_stock_m3", "NCB drainage stock", "darkorange"),
]
FLUX_SERIES = [
    ("IR_cyclic_North_m3", "IR North (cyclic)", "tab:blue"),
    ("IR_cyclic_South_m3", "IR South (cyclic)", "tab:orange"),
    ("IR_all_North_m3", "IR_all North", "deepskyblue"),
    ("IR_all_South_m3", "IR_all South", "darkorange"),
    ("ir_pool_m3", "IR_all (N+S)", "tab:green"),
    ("D_North_m3", "D North (ADCP)", "tab:brown"),
    ("D_South_m3", "D South (ADCP)", "tab:olive"),
    ("LFV_pump_m3", "LFV_pump", "tab:purple"),
    ("W_m3", "W from lake", "tab:cyan"),
    ("O_m3", "O to lake", "tab:red"),
    ("d_ncb_m3", "non-cyclic D", "tab:orange"),
]
STOCK_LABELS = [label for _k, label, _c in STOCK_SERIES]
FLUX_LABELS = [label for _k, label, _c in FLUX_SERIES]
STOCK_KEY_BY_LABEL = {label: key for key, label, _c in STOCK_SERIES}
FLUX_KEY_BY_LABEL = {label: key for key, label, _c in FLUX_SERIES}
PIXEL_LOG = {
    "pond": "pond_mm", "topsoil": "topsoil_mm", "leftover": "leftover_mm",
    "into_ground": "infiltration_mm", "lateral": "lateral_mm", "stay": "stay_mm",
    "need": "need_mm", "canal": "canal_mm", "need_stock": "need_stock_mm",
}

# Defaults: fewer maps so the page is readable; pick more if you want.
_DEFAULT_MAPS = [
    "into ground soil (m³/d)",
    "lateral flow sent (m³/d)",
    "D back to the pool (m³/d)",
    "pond stock (m³)",
]
# Pool is a same-day transit buffer (~few thousand m³), not a season stock — omit it.
_DEFAULT_STOCK = ["pond volume"]
_DEFAULT_FLUX = ["IR North (cyclic)", "IR South (cyclic)", "D North (ADCP)", "D South (ADCP)",
                 "W from lake", "O to lake"]

chosen_run = solara.reactive("")
day = solara.reactive(0)
playing = solara.reactive(False)
seconds_per_day = solara.reactive("0.6")
et_display = solara.reactive("Difference from map mean")
gif_message = solara.reactive("")
gif_file = solara.reactive("")
selected_maps = solara.reactive(list(_DEFAULT_MAPS))
selected_stock = solara.reactive(list(_DEFAULT_STOCK))
selected_flux = solara.reactive(list(_DEFAULT_FLUX))
paper_view = solara.reactive("Fig. 8 — IR North / South + lake W")
paper_year = solara.reactive("2022")
show_paper = solara.reactive(True)
show_field = solara.reactive(True)

# Play must not use solara.use_thread: each day.set re-renders and can cancel the
# wait, racing through the whole season in one flash.
_play_generation = 0
_play_lock = threading.Lock()


def _log_mm(model, name, days):
    if name in model.log:
        return model.log[name][:days].astype(float)
    return np.zeros((days, model.n))


def _station_m3(station, name, days, fallback):
    if station is not None and name in getattr(station, "columns", []):
        return station[name].to_numpy()[:days].astype(float)
    return np.asarray(fallback, dtype=float)


def season_arrays(results):
    """A finished season as arrays [day, paddy], plus the district series and fixed colour limits."""
    model, inputs = results.model, results.model.inputs
    if not model.log:
        raise ValueError("this run did not record paddies; run it again from Run & compare")
    days = len(results.daily)
    area = model.area
    share = area / area.sum()
    et = _log_mm(model, "et_mm", days)
    satellite_today = inputs.sat_pass_et_mm[:days].astype(float)
    pond = _log_mm(model, "pond_mm", days)
    if not np.any(pond) and "depth_mm" in model.log:
        pond = model.log["depth_mm"][:days].astype(float)
    topsoil = _log_mm(model, "topsoil_mm", days)
    into_ground = _log_mm(model, "infiltration_mm", days)
    leftover = _log_mm(model, "leftover_mm", days)
    if not np.any(into_ground):
        into_ground = np.maximum(leftover, 0.0)
    lateral = _log_mm(model, "lateral_mm", days)
    cyclic = np.asarray(inputs.cyclic, dtype=float)
    canal = _log_mm(model, "canal_mm", days)
    if not np.any(canal) and "drainage_mm" in model.log:
        canal = model.log["drainage_mm"][:days].astype(float)
    canal_m3 = canal * area / 1000.0 * cyclic
    ncb_m3 = canal * area / 1000.0 * (1.0 - cyclic)
    station = results.daily
    zeros = np.zeros(days)
    pool = _station_m3(station, "pool_m3", days, zeros)
    d_pool = _station_m3(station, "D_pool_m3", days, canal_m3.sum(axis=1))
    ir_pool = _station_m3(station, "IR_all_m3", days,
                         (model.log["irrigation_mm"][:days] * area / 1000.0).sum(axis=1))
    d_ncb = _station_m3(station, "D_ncb_m3", days, ncb_m3.sum(axis=1))
    ncb_stock = _station_m3(station, "ncb_stock_m3", days, np.cumsum(d_ncb))
    stay = _log_mm(model, "stay_mm", days)
    arrays = SimpleNamespace(
        dates=inputs.dates[:days], lon=inputs.frame["lon"].to_numpy(), lat=inputs.frame["lat"].to_numpy(),
        passes=inputs.is_pass[:days], area=area, cyclic=cyclic,
        irrigation=model.log["irrigation_mm"][:days] * area / 1000.0,
        drainage=model.log["drainage_mm"][:days] * area / 1000.0,
        et=et,
        rain=np.broadcast_to(np.asarray(inputs.rain_mm[:days], dtype=float), (days,)),
        satellite=pd.DataFrame(satellite_today).ffill().to_numpy(),
        pond=pond,
        pond_m3=pond * area / 1000.0,
        topsoil=topsoil,
        topsoil_m3=topsoil * area / 1000.0,
        leftover=leftover,
        into_ground=into_ground,
        into_ground_m3=into_ground * area / 1000.0,
        groundwater=_log_mm(model, "et_groundwater_mm", days),
        need=_log_mm(model, "need_mm", days),
        need_stock=_log_mm(model, "need_stock_mm", days),
        canal=canal * cyclic,
        canal_m3=canal_m3,
        ncb_m3=ncb_m3,
        pool_m3=pool,
        d_pool_m3=d_pool,
        ir_pool_m3=ir_pool,
        d_ncb_m3=d_ncb,
        ncb_stock_m3=ncb_stock,
        W_m3=_station_m3(station, "W_m3", days, zeros),
        O_m3=_station_m3(station, "O_m3", days, zeros),
        LFV_pump_m3=_station_m3(station, "LFV_pump_m3", days, zeros),
        D_North_m3=_station_m3(station, "D_North_m3", days, zeros),
        D_South_m3=_station_m3(station, "D_South_m3", days, zeros),
        IR_all_North_m3=_station_m3(station, "IR_all_North_m3", days, zeros),
        IR_all_South_m3=_station_m3(station, "IR_all_South_m3", days, zeros),
        IR_cyclic_North_m3=zeros.copy(),
        IR_cyclic_South_m3=zeros.copy(),
        lateral=lateral,
        lateral_m3=lateral * area / 1000.0,
        stay=stay,
        stay_m3=stay * area / 1000.0,
    )
    if not np.any(arrays.stay):
        arrays.stay = _log_mm(model, "deep_mm", days)
        arrays.stay_m3 = arrays.stay * area / 1000.0
    blocks = results.blocks.groupby("date")[["irrigation_m3", "drainage_m3"]].sum()
    observed = imitation._complete(imitation.load_observed(results.year)[0].copy())
    observed = observed.groupby("date")[["irrigation_m3", "drainage_m3"]].sum().reindex(blocks.index)
    # Cyclic IR per block (Table 4 unit) — separate from IR_all which includes NCB.
    by_block = results.blocks.copy()
    by_block["date"] = pd.to_datetime(by_block["date"]).dt.normalize()
    date_index = pd.to_datetime(pd.Index(inputs.dates[:days])).normalize()
    for name in ("North", "South"):
        part = by_block[by_block["block"] == name].set_index("date")
        key = f"IR_cyclic_{name}_m3"
        setattr(arrays, key, part.reindex(date_index)["irrigation_m3"].fillna(0.0).to_numpy())
    field = sum(inputs.field_et_mm[block][:days] * share[inputs.block == block].sum() for block in okura.BLOCKS)

    fly_days = satellite_today.copy()
    fly_mean = np.array([np.nansum(np.where(np.isfinite(row), row, 0.0) * share) /
                         max(share[np.isfinite(row)].sum(), 1e-12) if np.isfinite(row).any() else np.nan
                         for row in fly_days])
    arrays.district = pd.DataFrame({
        "irrigation": blocks["irrigation_m3"].to_numpy() / 1000.0,
        "irrigation_observed": observed["irrigation_m3"].to_numpy() / 1000.0,
        "drainage": blocks["drainage_m3"].to_numpy() / 1000.0,
        "drainage_observed": observed["drainage_m3"].to_numpy() / 1000.0,
        "et": et @ share, "field_et": field, "satellite_et": fly_mean,
        "pond": pond @ share,
        "topsoil": arrays.topsoil @ share,
        "topsoil_m3": arrays.topsoil_m3.sum(axis=1),
        "leftover": arrays.leftover @ share,
        "into_ground": arrays.into_ground @ share,
        "into_ground_m3": arrays.into_ground_m3.sum(axis=1),
        "groundwater": arrays.groundwater @ share,
        "need": arrays.need @ share,
        "need_stock": arrays.need_stock @ share,
        "canal": arrays.canal @ share,
        "canal_m3": arrays.canal_m3.sum(axis=1),
        "ncb_m3": arrays.ncb_m3.sum(axis=1),
        "pool_m3": arrays.pool_m3,
        "pond_m3": arrays.pond_m3.sum(axis=1),
        "d_pool_m3": arrays.d_pool_m3,
        "ir_pool_m3": arrays.ir_pool_m3,
        "d_ncb_m3": arrays.d_ncb_m3,
        "ncb_stock_m3": arrays.ncb_stock_m3,
        "W_m3": arrays.W_m3, "O_m3": arrays.O_m3, "LFV_pump_m3": arrays.LFV_pump_m3,
        "D_North_m3": arrays.D_North_m3, "D_South_m3": arrays.D_South_m3,
        "IR_all_North_m3": arrays.IR_all_North_m3, "IR_all_South_m3": arrays.IR_all_South_m3,
        "IR_cyclic_North_m3": arrays.IR_cyclic_North_m3,
        "IR_cyclic_South_m3": arrays.IR_cyclic_South_m3,
        "rain": arrays.rain, "lateral": arrays.lateral @ share,
        "lateral_m3": arrays.lateral_m3.sum(axis=1),
        "stay": arrays.stay @ share,
        "stay_m3": arrays.stay_m3.sum(axis=1),
    }, index=arrays.dates)
    et_high = max(float(np.nanpercentile(et, 98)), float(np.nanpercentile(satellite_today, 98))
                  if np.isfinite(satellite_today).any() else 0.0, 1e-6)
    arrays.limits = {
        "irrigation": (0.0, max(float(np.nanpercentile(arrays.irrigation, 98)), 1e-6)),
        "drainage": (0.0, max(float(np.nanpercentile(arrays.drainage, 98)), 1e-6)),
        "et": (0.0, et_high),
        "satellite": (0.0, et_high),
        "pond": (0.0, max(float(np.nanpercentile(arrays.pond, 98)), 1e-6)),
        "pond_m3": (0.0, max(float(np.nanpercentile(arrays.pond_m3, 98)), 1e-6)),
        "topsoil": (0.0, max(float(np.nanpercentile(arrays.topsoil, 98)), 1e-6)),
        "topsoil_m3": (0.0, max(float(np.nanpercentile(arrays.topsoil_m3, 98)), 1e-6)),
        "leftover": (
            min(float(np.nanpercentile(arrays.leftover, 2)), 0.0),
            max(float(np.nanpercentile(arrays.leftover, 98)), 1e-6),
        ),
        "into_ground": (0.0, max(float(np.nanpercentile(arrays.into_ground, 98)), 1e-6)),
        "into_ground_m3": (0.0, max(float(np.nanpercentile(arrays.into_ground_m3, 98)), 1e-6)),
        "groundwater": (0.0, max(float(np.nanpercentile(arrays.groundwater, 98)), 1e-6)),
        "need": (0.0, max(float(np.nanpercentile(arrays.need, 98)), 1e-6)),
        "canal": (0.0, max(float(np.nanpercentile(arrays.canal, 98)), 1e-6)),
        "canal_m3": (0.0, max(float(np.nanpercentile(arrays.canal_m3, 98)), 1e-6)),
        "ncb_m3": (0.0, max(float(np.nanpercentile(arrays.ncb_m3, 98)), 1e-6)),
        "need_stock": (0.0, max(float(np.nanpercentile(arrays.need_stock, 98)), 1e-6)),
        "lateral": (0.0, max(float(np.nanpercentile(arrays.lateral, 98)), 1e-6)),
        "lateral_m3": (0.0, max(float(np.nanpercentile(arrays.lateral_m3, 98)), 1e-6)),
        "stay": (0.0, max(float(np.nanpercentile(arrays.stay, 98)), 1e-6)),
        "stay_m3": (0.0, max(float(np.nanpercentile(arrays.stay_m3, 98)), 1e-6)),
    }
    if model.cell_et_mm is not None:
        grid = model.cells
        arrays.pixel_lon, arrays.pixel_lat = grid.lon, grid.lat
        arrays.pixel_et = model.cell_et_mm[:days]
        arrays.pixel_satellite = pd.DataFrame(grid.observed_et_mm[:days]).ffill().to_numpy()
        arrays.pixel_passes = grid.is_pass[:days]
        cell_area = model.area[grid.owner] * grid.weights
        for key in ("et", "satellite"):
            values = getattr(arrays, "pixel_" + key)
            valid = np.isfinite(values)
            denominator = valid @ cell_area
            means = np.divide(np.where(valid, values, 0) @ cell_area, denominator,
                              out=np.full(days, np.nan), where=denominator > 0)
            setattr(arrays, "pixel_" + key + "_deviation", values - means[:, None])
        spread = np.concatenate([arrays.pixel_et_deviation.ravel(), arrays.pixel_satellite_deviation.ravel()])
        spread = spread[np.isfinite(spread)]
        arrays.deviation_limit = max(float(np.percentile(abs(spread), 98)) if spread.size else 0, 0.01)
        arrays.pixel_source = f"{grid.source}, approximately {grid.resolution_m} m source cells"
        finite = grid.observed_et_mm[np.isfinite(grid.observed_et_mm)]
        high = max(float(np.nanpercentile(arrays.pixel_et, 98)),
                   float(np.percentile(finite, 98)) if finite.size else 0, 1e-6)
        arrays.limits["et"] = arrays.limits["satellite"] = (0.0, high)
        if model.cell_log:
            for key, name in PIXEL_LOG.items():
                if name in model.cell_log:
                    setattr(arrays, "pixel_" + key, model.cell_log[name][:days])
            if hasattr(arrays, "pixel_canal"):
                cyclic_cell = arrays.cyclic[grid.owner]
                arrays.pixel_canal_m3 = arrays.pixel_canal * cell_area / 1000.0 * cyclic_cell
                arrays.pixel_ncb_m3 = arrays.pixel_canal * cell_area / 1000.0 * (1.0 - cyclic_cell)
            if hasattr(arrays, "pixel_into_ground"):
                arrays.pixel_into_ground_m3 = arrays.pixel_into_ground * cell_area / 1000.0
            if hasattr(arrays, "pixel_stay"):
                arrays.pixel_stay_m3 = arrays.pixel_stay * cell_area / 1000.0
            if hasattr(arrays, "pixel_pond"):
                arrays.pixel_pond_m3 = arrays.pixel_pond * cell_area / 1000.0
    return arrays


def _today_total(arrays, key, d):
    values = np.asarray(getattr(arrays, key)[d], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    if key.endswith("_m3"):
        return float(finite.sum())
    area = getattr(arrays, "area", None)
    if area is None or len(area) != len(values):
        return float(np.nanmean(finite))
    weight = np.where(np.isfinite(values), area, 0.0)
    return float(np.nansum(values * area) / max(weight.sum(), 1e-12))


def today_readout(arrays, d):
    """Numbers that must move when Play advances the day."""
    date = arrays.dates[d]
    topsoil = _today_total(arrays, "topsoil_m3", d)
    need = _today_total(arrays, "need", d)
    rain = float(np.nan_to_num(arrays.rain[d]))
    et = _today_total(arrays, "et", d)
    into_ground = _today_total(arrays, "into_ground_m3", d)
    lateral = _today_total(arrays, "lateral_m3", d)
    discharge = float(np.nan_to_num(arrays.d_pool_m3[d]))
    pumped = float(np.nan_to_num(arrays.ir_pool_m3[d]))
    ncb = float(np.nan_to_num(arrays.d_ncb_m3[d]))
    ncb_stock = float(np.nan_to_num(arrays.ncb_stock_m3[d]))
    w = float(np.nan_to_num(arrays.W_m3[d]))
    o = float(np.nan_to_num(arrays.O_m3[d]))
    lfv = float(np.nan_to_num(arrays.LFV_pump_m3[d]))
    dn = float(np.nan_to_num(arrays.D_North_m3[d]))
    ds = float(np.nan_to_num(arrays.D_South_m3[d]))
    return (
        f"**Day {d} — {date:%d %b %Y}**  \n"
        f"Rain **{rain:.1f} mm** · ET **{et:.2f} mm/d** · "
        f"ET need missing from topsoil **{need:.2f} mm/d**  \n"
        f"Station (m³/d): W **{w:,.0f}** · O **{o:,.0f}** · "
        f"LFV_pump **{lfv:,.0f}** · D_N **{dn:,.0f}** · D_S **{ds:,.0f}** · "
        f"IR_all **{pumped:,.0f}**  \n"
        f"Topsoil **{topsoil:,.0f} m³** · into ground **{into_ground:,.0f} m³** · "
        f"lateral **{lateral:,.0f} m³** · cyclic D **{discharge:,.0f} m³** · "
        f"NCB D **{ncb:,.0f} m³/d** · NCB stock **{ncb_stock:,.0f} m³**"
    )


def _map_panel_specs(map_labels):
    """Ordered (key, label, cmap) for the labels the user picked."""
    wanted = set(map_labels or ())
    return [(key, label, cmap) for key, label, cmap in MAP_PANELS if label in wanted]


def _series_specs(labels, catalog, key_by_label):
    wanted = set(labels or ())
    return [(key_by_label[label], label, color)
            for _key, label, color in catalog if label in wanted]


def frame_figure(arrays, d, title, et_deviation=False, map_labels=None,
                 stock_labels=None, flux_labels=None):
    """One day's picture. Only the selected maps / series are drawn, larger."""
    panels = _map_panel_specs(map_labels if map_labels is not None else MAP_LABELS)
    stock = _series_specs(
        stock_labels if stock_labels is not None else STOCK_LABELS,
        STOCK_SERIES, STOCK_KEY_BY_LABEL)
    flux = _series_specs(
        flux_labels if flux_labels is not None else FLUX_LABELS,
        FLUX_SERIES, FLUX_KEY_BY_LABEL)

    n_maps = len(panels)
    show_stock = bool(stock)
    show_flux = bool(flux)
    n_chart_cols = int(show_stock) + int(show_flux)
    show_charts = n_chart_cols > 0

    if n_maps <= 0 and not show_charts:
        fig = Figure(figsize=(12, 3))
        fig.text(0.5, 0.5, "Select at least one map or series above.",
                 ha="center", va="center", fontsize=14)
        return fig

    if n_maps <= 1:
        ncols = 1
    elif n_maps <= 2:
        ncols = 2
    elif n_maps <= 6:
        ncols = 3
    else:
        ncols = 4
    nrows_maps = int(np.ceil(n_maps / ncols)) if n_maps else 0
    map_row_h = 4.6 if n_maps <= 4 else 3.8
    chart_row_h = 3.8
    fig_w = 18.5 if ncols >= 3 else (16.0 if ncols == 2 else 12.0)
    height_ratios = ([map_row_h] * nrows_maps) + ([chart_row_h] if show_charts else [])
    fig_h = sum(height_ratios) + 0.9
    fig = Figure(figsize=(fig_w, fig_h))
    grid = fig.add_gridspec(
        nrows_maps + int(show_charts), max(ncols, n_chart_cols or 1),
        height_ratios=height_ratios, hspace=0.28, wspace=0.12)

    passed = arrays.passes[d]
    point_size = 9 if n_maps <= 2 else (6 if n_maps <= 4 else 4)
    pixel_size = 0.7 if n_maps <= 2 else (0.5 if n_maps <= 4 else 0.35)
    title_fs = 12 if n_maps <= 4 else 10

    for i, (key, label, cmap) in enumerate(panels):
        ax = fig.add_subplot(grid[i // ncols, i % ncols])
        low, high = arrays.limits[key]
        values = getattr(arrays, key)[d]
        lon, lat = arrays.lon, arrays.lat
        pixel_name = "pixel_" + key
        if hasattr(arrays, pixel_name):
            values = getattr(arrays, pixel_name)[d]
            lon, lat = arrays.pixel_lon, arrays.pixel_lat
        shown = np.isfinite(values)
        if (~shown).any():
            ax.scatter(lon[~shown], lat[~shown], s=max(point_size * 0.5, 2), color="lightgrey")
        size = pixel_size if hasattr(arrays, pixel_name) else point_size
        points = ax.scatter(lon[shown], lat[shown], c=values[shown], s=size, linewidths=0,
                            cmap=cmap, vmin=low, vmax=high)
        ax.plot(*data.POOL_LONLAT, marker="*", color="red", markersize=14, markeredgecolor="white")
        ax.set_aspect(1 / 0.81)
        ax.set_xticks([])
        ax.set_yticks([])
        today_value = _today_total(arrays, key, d)
        unit = "m³" if key.endswith("_m3") else ("mm" if "stock" in key or key == "pond" else "mm/d")
        if key.endswith("_m3"):
            caption = f"today {today_value:,.0f} {unit}"
        else:
            caption = f"today {today_value:.2f} {unit}"
        ax.set_title(f"{label}\n{caption}", fontsize=title_fs)
        fig.colorbar(points, ax=ax, orientation="horizontal", shrink=0.85, pad=0.03)

    today = arrays.dates[d]
    district = arrays.district
    if show_charts:
        chart_row = nrows_maps
        n_gs = max(ncols, n_chart_cols or 1)
        if show_stock and show_flux:
            if n_gs >= 4:
                water = fig.add_subplot(grid[chart_row, 0:2])
                flux_ax = fig.add_subplot(grid[chart_row, 2:4])
            elif n_gs == 3:
                water = fig.add_subplot(grid[chart_row, 0:2])
                flux_ax = fig.add_subplot(grid[chart_row, 2])
            else:
                water = fig.add_subplot(grid[chart_row, 0])
                flux_ax = fig.add_subplot(grid[chart_row, 1])
        elif show_stock:
            water = fig.add_subplot(grid[chart_row, :])
            flux_ax = None
        else:
            water = None
            flux_ax = fig.add_subplot(grid[chart_row, :])

        if water is not None:
            for key, label, color in stock:
                water.plot(district.index, district[key], color=color, lw=1.8, label=label)
            water.axvline(today, color="red", lw=2.0)
            water.set_ylabel("m³", fontsize=11)
            water.legend(fontsize=9, ncol=2, loc="upper left")
            water.tick_params(labelsize=9)
            water.set_title("Stocks", fontsize=11)
        if flux_ax is not None:
            for key, label, color in flux:
                flux_ax.plot(district.index, district[key], color=color, lw=1.7, label=label)
            flux_ax.axvline(today, color="red", lw=2.0)
            flux_ax.set_ylabel("m³/d", fontsize=11)
            flux_ax.legend(fontsize=9, ncol=2, loc="upper left")
            flux_ax.tick_params(labelsize=9)
            flux_ax.set_title("Daily flows", fontsize=11)

    suffix = f" · satellite pass over {int(passed.sum()):,} paddies" if passed.any() else ""
    fig.suptitle(f"{title} — {today:%d %b %Y} (day {d}){suffix}", fontsize=14)
    return fig


def save_gif(arrays, title, path, every=3, fps=5, dpi=70, on_progress=None, et_deviation=False,
             map_labels=None, stock_labels=None, flux_labels=None):
    """Every `every`-th day of the season as one animated GIF."""
    from PIL import Image

    frames = []
    days = list(range(0, len(arrays.dates), every))
    for count, d in enumerate(days):
        buffer = io.BytesIO()
        frame_figure(
            arrays, d, title, et_deviation=et_deviation,
            map_labels=map_labels, stock_labels=stock_labels, flux_labels=flux_labels,
        ).savefig(buffer, format="png", dpi=dpi)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("P", palette=Image.ADAPTIVE))
        if on_progress:
            on_progress(count + 1, len(days))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    return path


def _stop_play():
    global _play_generation
    with _play_lock:
        _play_generation += 1
        playing.set(False)


def _start(last_day):
    """Advance one day per delay in a plain thread, through to the last day."""
    global _play_generation
    if day.peek() >= last_day:
        day.set(0)
    with _play_lock:
        _play_generation += 1
        gen = _play_generation
        playing.set(True)

    def run():
        end = int(last_day)
        while playing.peek():
            with _play_lock:
                if gen != _play_generation:
                    return
            current = int(day.peek())
            if current >= end:
                playing.set(False)
                return
            try:
                delay = float(seconds_per_day.peek())
            except (TypeError, ValueError):
                delay = 0.6
            delay = max(delay, 0.15)
            time.sleep(delay)
            with _play_lock:
                if gen != _play_generation or not playing.peek():
                    return
            # Re-read in case the slider moved during the wait.
            current = int(day.peek())
            nxt = current + 1
            if nxt > end:
                day.set(end)
                playing.set(False)
                return
            day.set(nxt)
            if nxt >= end:
                playing.set(False)
                return

    threading.Thread(target=run, daemon=True).start()


@solara.component
def FieldTab(records, year=None, on_year=None):
    from .paper_figures import PAPER_VIEWS, paper_figure

    names = list(records)
    chosen = chosen_run.value if chosen_run.value in names else (names[0] if names else None)
    record = records.get(chosen)
    arrays = solara.use_memo(
        lambda: season_arrays(record.results) if record else None,
        dependencies=[id(record), chosen])
    last_day = len(arrays.dates) - 1 if arrays is not None else 0
    last = last_day
    current = min(max(int(day.value), 0), last) if arrays is not None else 0
    maps = list(selected_maps.value)
    stocks = list(selected_stock.value)
    fluxes = list(selected_flux.value)
    year_for_paper = int(year) if year is not None else int(paper_year.value)

    def change_year(v):
        paper_year.set(str(v))
        if on_year is not None:
            on_year(v)

    solara.Markdown(
        "Use **Paper graphs** for Okura Figs. 4–8 (IR North/South, ADCP D, monthly balance, rain/ET). "
        "After a **Run & compare** finish, select the run to overlay the model and to play "
        "field maps. Flow lines include **IR North / South** separately from IR_all. "
        f"**Year {year_for_paper}** follows the header toggle (or Year below).")

    solara.Markdown("### Paper graphs (Okura et al.)")
    with solara.Row(style={"align-items": "center", "flex-wrap": "wrap"}, gap="12px"):
        solara.Checkbox(label="Show paper graph", value=show_paper)
        solara.Select(
            "Year", values=["2021", "2022"], value=str(year_for_paper),
            on_value=change_year)
        solara.Select("Paper figure", values=list(PAPER_VIEWS), value=paper_view)
    if show_paper.value:
        plot_year = int(year_for_paper)
        model_blocks = record.results.blocks if record is not None else None
        model_station = getattr(record.results, "daily", None) if record is not None else None
        if model_station is not None and "date" not in getattr(model_station, "columns", []):
            model_station = model_station.reset_index().rename(columns={"index": "date"})
        if record is not None and int(record.year) != plot_year:
            model_blocks = None
            model_station = None
        solara.FigureMatplotlib(
            paper_figure(paper_view.value, plot_year,
                         model_blocks=model_blocks, model_station=model_station),
            dependencies=[paper_view.value, plot_year, chosen, paper_year.value,
                          show_paper.value],
            format="png")
        if record is not None and int(record.year) == plot_year:
            from .season_targets import compare_to_paper, holdout_summary_text
            hold = compare_to_paper(record.results.blocks, int(record.year))
            solara.Info(holdout_summary_text(hold))

    if record is None:
        solara.Markdown(
            f"_No run for **{year_for_paper}** yet. Paper graphs above still show that year. "
            f"Run a farmer on **Run & compare** (year {year_for_paper}) to unlock Play "
            f"and the field maps._")
        return

    def choose_run(name):
        _stop_play()
        chosen_run.set(name)
        day.set(0)

    solara.Markdown("### Field over time")
    with solara.Row(style={"align-items": "center", "flex-wrap": "wrap"}, gap="12px"):
        solara.Select("Run", values=names, value=chosen, on_value=choose_run)
        if playing.value:
            solara.Button("Pause", icon_name="mdi-pause", color="primary", on_click=_stop_play)
        else:
            solara.Button("Play", icon_name="mdi-play", color="primary",
                          on_click=lambda: _start(last))
        solara.Button("Start", icon_name="mdi-skip-previous", outlined=True,
                      on_click=lambda: (_stop_play(), day.set(0)))
        solara.Button("End", icon_name="mdi-skip-next", outlined=True,
                      on_click=lambda: (_stop_play(), day.set(last)))
        solara.Select("Seconds per day", values=["0.3", "0.6", "1.0", "2.0"],
                      value=str(seconds_per_day.value), on_value=seconds_per_day.set)

    with solara.Row(style={"align-items": "flex-start", "flex-wrap": "wrap"}, gap="16px"):
        solara.SelectMultiple("Field maps", values=selected_maps, all_values=MAP_LABELS,
                              style={"min-width": "280px", "max-width": "420px"})
        solara.SelectMultiple("Stock lines", values=selected_stock, all_values=STOCK_LABELS,
                              style={"min-width": "220px", "max-width": "320px"})
        solara.SelectMultiple("Flow lines", values=selected_flux, all_values=FLUX_LABELS,
                              style={"min-width": "260px", "max-width": "380px"})
    with solara.Row(style={"align-items": "center", "flex-wrap": "wrap"}, gap="8px"):
        solara.Button("Paper IR/D", outlined=True,
                      on_click=lambda: (
                          selected_maps.set([]),
                          selected_stock.set([]),
                          selected_flux.set([
                              "IR North (cyclic)", "IR South (cyclic)",
                              "D North (ADCP)", "D South (ADCP)",
                              "W from lake",
                          ]),
                      ))
        solara.Button("Soil focus", outlined=True,
                      on_click=lambda: (
                          selected_maps.set([
                              "into ground soil (m³/d)",
                              "lateral flow sent (m³/d)",
                              "pond stock (m³)",
                              "stays in the ground (m³/d)",
                          ]),
                          selected_stock.set(["pond volume", "topsoil volume"]),
                          selected_flux.set([]),
                      ))
        solara.Button("Drainage focus", outlined=True,
                      on_click=lambda: (
                          selected_maps.set([
                              "D back to the pool (m³/d)",
                              "non-cyclic D to the lake (m³/d)",
                              "lateral flow sent (m³/d)",
                          ]),
                          selected_stock.set(["pond volume", "NCB drainage stock"]),
                          selected_flux.set([
                              "D North (ADCP)", "D South (ADCP)",
                              "IR_all (N+S)", "LFV_pump",
                          ]),
                      ))
        solara.Button("Show all", outlined=True,
                      on_click=lambda: (
                          selected_maps.set(list(MAP_LABELS)),
                          selected_stock.set(list(STOCK_LABELS)),
                          selected_flux.set(list(FLUX_LABELS)),
                      ))
        solara.Button("Reset view", outlined=True,
                      on_click=lambda: (
                          selected_maps.set(list(_DEFAULT_MAPS)),
                          selected_stock.set(list(_DEFAULT_STOCK)),
                          selected_flux.set(list(_DEFAULT_FLUX)),
                      ))

    solara.Info(today_readout(arrays, current))
    solara.SliderInt(f"Day {current}: {arrays.dates[current]:%d %b %Y}",
                     value=day, min=0, max=last)

    solara.Markdown("### Field maps and selected series")
    with solara.Row(style={"align-items": "center"}, gap="12px"):
        solara.Checkbox(label="Show field figure", value=show_field)
    if show_field.value:
        title = f"{record.label} ({record.year})"
        fig = frame_figure(
            arrays, current, title,
            map_labels=maps, stock_labels=stocks, flux_labels=fluxes)
        solara.FigureMatplotlib(
            fig,
            dependencies=[current, chosen, tuple(maps), tuple(stocks), tuple(fluxes)],
            format="png")

    def export():
        path = os.path.join(
            OUTPUTS,
            f"field_{record.label.replace(' ', '_').replace(':', '_')}_{record.year}.gif")

        def work():
            try:
                save_gif(
                    arrays, f"{record.label} ({record.year})", path,
                    map_labels=list(selected_maps.peek()),
                    stock_labels=list(selected_stock.peek()),
                    flux_labels=list(selected_flux.peek()),
                    on_progress=lambda i, n: gif_message.set(f"Drawing frame {i} of {n} …"))
                gif_file.set(path)
                gif_message.set(f"Saved {path}")
            except Exception as error:
                gif_message.set(f"Could not save the animation: {error}")

        gif_message.set("Starting …")
        threading.Thread(target=work, daemon=True).start()

    with solara.Row(style={"align-items": "center"}):
        solara.Button("Save the season as a GIF", icon_name="mdi-movie-open-outline",
                      outlined=True, on_click=export)
        if gif_message.value:
            solara.Text(gif_message.value)
    if gif_file.value and os.path.exists(gif_file.value):
        with open(gif_file.value, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        solara.HTML(
            tag="img",
            attributes={"src": f"data:image/gif;base64,{encoded}", "style": "max-width: 100%;"})

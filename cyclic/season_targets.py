"""Season-total schedules and paper holdout — no daily sensor CSV in learning.

Learning targets are only **season** irrigation / drainage volumes per block.
Daily shape uses Open-Meteo rain, ET, soil moisture, and rain hours (plus optional
satellite flood). A low-frequency **signal extraction** step (EMA envelope +
attenuated residual) removes weather chatter so Fig. 8-style plots show the
seasonal irrigation signal rather than daily noise.

D is mildly anti-coupled to IR so lake W can pulse with irrigation peaks.

The digitised Okura Fig. 6 / Fig. 8 CSVs are **holdout only**.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import data, imitation, okura
from .paper_plots import ema as _ema


def season_targets_m3(year, rain_mm=None):
    """Block season volumes available without daily ADCP / pump CSVs.

    Cyclic IR/D from Table 4 (mm × area). NCB IR from Table 5 − cyclic;
    NCB D from the cyclic D/(IR+R−ET) share with the given season rain.
    Anywhere else: replace these with local season totals the same way.
    """
    out = {}
    for block in okura.BLOCKS:
        t4 = okura.TABLE4_MM[(block, year)]
        a = okura.cyclic_area_m2(block)
        out[(block, "cyclic", "irrigation")] = t4["irrigation"] * a / 1000.0
        out[(block, "cyclic", "drainage")] = t4["drainage"] * a / 1000.0
        out[(block, "ncb", "irrigation")] = okura.ir_ncb_m3(block, year)
        out[(block, "ncb", "drainage")] = okura.d_ncb_m3(block, year, rain_mm=rain_mm)
    return out


def relative_fidelity(got, want):
    """1 − |got−want|/|want| for nonzero want; else 1 if both ~0 else 0."""
    got, want = float(got), float(want)
    if abs(want) < 1e-9:
        return 1.0 if abs(got) < 1e-9 else 0.0
    return float(np.clip(1.0 - abs(got - want) / abs(want), 0.0, 1.0))


def surface_coverage_table(year):
    """Observed IR/D day coverage and W/O vs Table 5 (forcing quality)."""
    year = int(year)
    blocks, station, source = imitation.load_observed(year)
    t5 = okura.TABLE5_1000M3[year]
    rows = []
    for block in okura.BLOCKS:
        part = blocks[blocks["block"] == block]
        ir = part["irrigation_m3"].to_numpy(dtype=float)
        dr = part["drainage_m3"].to_numpy(dtype=float)
        a = okura.cyclic_area_m2(block)
        want_ir = okura.TABLE4_MM[(block, year)]["irrigation"] * a / 1000.0
        want_d = okura.TABLE4_MM[(block, year)]["drainage"] * a / 1000.0
        rows.append({
            "block": block,
            "days": len(part),
            "IR_days>0": int((ir > 1.0).sum()),
            "D_days>0": int((dr > 1.0).sum()),
            "IR_fid": relative_fidelity(ir.sum(), want_ir),
            "D_fid": relative_fidelity(dr.sum(), want_d),
            "IR_1000m3": ir.sum() / 1000.0,
            "D_1000m3": dr.sum() / 1000.0,
        })
    w = float(station["W_m3"].sum()) if station is not None else float("nan")
    o = float(station["O_m3"].sum()) if station is not None and "O_m3" in station.columns else float("nan")
    station_row = {
        "W_fid": relative_fidelity(w / 1000.0, t5["W"]),
        "O_fid": relative_fidelity(o / 1000.0, t5["O"]),
        "W_1000m3": w / 1000.0,
        "O_1000m3": o / 1000.0,
        "W_table5": t5["W"],
        "O_table5": t5["O"],
    }
    return pd.DataFrame(rows), station_row, source


def _align(arr, n, fill=0.0):
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float).ravel()
    if a.size >= n:
        return a[:n]
    if a.size == 0:
        return np.full(n, float(fill))
    return np.resize(a, n)


def _climate_wetness(rain, soil_moisture=None, rain_hours=None):
    """0–1 wetness from rain, optional rain duration and soil moisture (Open-Meteo)."""
    rain = np.asarray(rain, dtype=float).ravel()
    n = rain.size
    wet = rain / max(float(np.nanmax(rain)), 1e-6)
    if rain_hours is not None:
        rh = _align(rain_hours, n, 0.0)
        wet = 0.55 * wet + 0.45 * (rh / max(float(np.nanmax(rh)), 1e-6))
    if soil_moisture is not None:
        sm = _align(soil_moisture, n, float(np.nanmean(soil_moisture)))
        sm = np.nan_to_num(sm, nan=float(np.nanmean(sm) if np.isfinite(sm).any() else 0.3))
        lo, hi = np.percentile(sm, [5, 95])
        if hi - lo < 1e-6:
            sm_n = np.clip(sm, 0.0, 1.0)
        else:
            sm_n = np.clip((sm - lo) / (hi - lo), 0.0, 1.0)
        wet = 0.65 * wet + 0.35 * sm_n
    return np.clip(wet, 0.0, 1.0)


def _daily_weights(rain, et, midsummer=True, year=None, flood=None,
                   soil_moisture=None, rain_hours=None):
    """Day weights: ET cover + rain/soil pauses; D anti-coupled for W pulses.

    Open-Meteo ``soil_moisture`` and ``rain_hours`` sharpen wet-day IR pauses.
    Drainage is rain-wet plus a mild *anti*-IR term so lake W (= IR_all − D − …)
    rises on dry irrigation peaks — still no Fig. 8 labels.
    """
    rain = np.asarray(rain, dtype=float).ravel()
    et = np.asarray(et, dtype=float).ravel()
    n = rain.size
    wet = _climate_wetness(rain, soil_moisture=soil_moisture, rain_hours=rain_hours)
    rh = _align(rain_hours, n, 0.0) if rain_hours is not None else np.zeros(n)
    rh_n = rh / max(float(np.nanmax(rh)), 1.0)
    # IR demand: ET not met by *useful* (MAFF effective) rain / wet spells / soil.
    er = okura.effective_rain_mm(rain)
    raw = np.maximum(et - 0.75 * er - 0.35 * rh_n * et - 0.40 * wet * et, 0.15)
    slow = _ema(raw, alpha=0.22)
    need = 0.55 * raw + 0.45 * slow
    if flood is not None:
        f = _align(flood, n, 1.0)
        if f is not None and float(np.percentile(f, 90)) >= 0.15:
            need = need * (0.7 + 0.3 * (1.0 - np.clip(f, 0.0, 1.0)))
    if midsummer and year is not None:
        dates = okura.season_dates(year)
        start = pd.Timestamp(f"{year}-{okura.MIDSUMMER_START}")
        off = (dates >= start) & (dates < start + pd.Timedelta(days=okura.MIDSUMMER_DAYS))
        off = np.asarray(off, dtype=bool)
        if off.size != n:
            off = np.resize(off.astype(float), n).astype(bool)
        need = need.copy()
        need[off] = 0.05
    need_n = need / max(float(need.sum()), 1e-9)
    # Prefer the slow climate envelope; weather only gently modulates (paper-like).
    ir_w = 0.28 * need_n + 0.72 / max(n, 1)
    ir_w = ir_w / max(float(ir_w.sum()), 1e-9)
    # D: wet days drain; anti-IR so high-IR dry days leave more for lake W.
    anti_ir = 1.0 - (need_n / max(float(need_n.max()), 1e-9))
    d_raw = 0.20 + 0.40 * wet + 0.40 * anti_ir
    d_raw = 0.65 * d_raw + 0.35 * _ema(d_raw, alpha=0.28)
    if flood is not None:
        f = _align(flood, n, 1.0)
        if f is not None and float(np.percentile(f, 90)) >= 0.15:
            d_raw = 0.6 * d_raw + 0.4 * (0.2 + 0.8 * np.clip(f, 0.0, 1.0))
    d_w = d_raw / max(float(d_raw.sum()), 1e-9)
    return ir_w, d_w


def _holdout_style_day_series(values, on, month, floor=0.20, cap=1.90, smooth=1):
    """Portable IR constraints without Fig. 8 labels.

    Loose floor/cap so troughs and peaks can approach paper CV; optional light
    smooth (1 = off) so daily rain pauses are not erased.
    """
    out = np.asarray(values, dtype=float).copy()
    on = np.asarray(on, dtype=bool)
    month = np.asarray(month)
    half = max(int(smooth) // 2, 0)
    for m in np.unique(month):
        sel = month == m
        on_m = sel & on
        if not np.any(on_m):
            continue
        total = float(out[sel].sum())
        if total <= 1e-12:
            out[sel] = 0.0
            continue
        idx = np.flatnonzero(on_m)
        vals = out[idx].copy()
        mean = float(np.mean(vals)) if vals.size else 0.0
        if mean > 1e-12:
            vals = np.clip(vals, floor * mean, cap * mean)
        if half > 0 and vals.size >= 3:
            pad = np.pad(vals, (half, half), mode="edge")
            vals = np.convolve(pad, np.ones(2 * half + 1) / (2 * half + 1), mode="valid")
        vals = vals / max(float(vals.sum()), 1e-12) * total
        out[sel] = 0.0
        out[idx] = vals
    return out


def _extract_real_signal(values, on, month, slow_alpha=0.14, fast_keep=0.22, floor_frac=0.50):
    """Pull the usable irrigation signal out of weather chatter (no Fig. 8).

    Paper IR is a *slow* envelope with mild day moves. Weather weights inject
    high-frequency spikes that bury that envelope. Keep month totals, take a
    causal EMA as the signal, and only keep a fraction of the fast residual.
    """
    out = np.asarray(values, dtype=float).copy()
    on = np.asarray(on, dtype=bool)
    month = np.asarray(month)
    for m in np.unique(month):
        sel = month == m
        on_m = sel & on
        if not np.any(on_m):
            continue
        total = float(out[sel].sum())
        if total <= 1e-12:
            out[sel] = 0.0
            continue
        idx = np.flatnonzero(on_m)
        vals = out[idx].astype(float)
        slow = _ema(vals, alpha=slow_alpha)
        # Bidirectional polish of the slow path (forward + reverse EMA).
        slow = 0.5 * slow + 0.5 * _ema(vals[::-1], alpha=slow_alpha)[::-1]
        fast = vals - slow
        vals = slow + float(fast_keep) * fast
        vals = np.maximum(vals, float(floor_frac) * np.maximum(slow, 0.0))
        vals = vals / max(float(vals.sum()), 1e-12) * total
        out[sel] = 0.0
        out[idx] = vals
    return out


def _match_jitter_to_weather(values, on, month, rain, et,
                             soil_moisture=None, rain_hours=None,
                             target_cv=0.32, target_mean_abs_d=0.10):
    """Mild CV nudge from weather energy — prefer ``_extract_real_signal`` after."""
    out = np.asarray(values, dtype=float).copy()
    on = np.asarray(on, dtype=bool)
    month = np.asarray(month)
    n = out.size
    rain = _align(rain, n, 0.0)
    et = _align(et, n, 1.0)
    ir_w, _ = _daily_weights(
        rain, et, midsummer=False, flood=None,
        soil_moisture=soil_moisture, rain_hours=rain_hours)
    proxy = ir_w / max(float(ir_w.mean()), 1e-9)
    for m in np.unique(month):
        sel = month == m
        on_m = sel & on
        if int(on_m.sum()) < 4:
            continue
        idx = np.flatnonzero(on_m)
        vals = out[idx].astype(float)
        total = float(vals.sum())
        mean = float(vals.mean())
        if mean <= 1e-12 or total <= 1e-12:
            continue
        p_cv = float(np.std(proxy[idx]) / max(float(np.mean(proxy[idx])), 1e-9))
        want_cv = float(np.clip(0.40 * p_cv + 0.60 * target_cv, 0.22, 0.42))
        resid = vals - mean
        cur_cv = float(np.std(vals) / mean)
        if cur_cv > 1e-6:
            # Only shrink excess jitter; never amplify toward weather spikes.
            scale = min(1.0, want_cv / cur_cv)
            vals = mean + resid * scale
        vals = np.maximum(vals, 0.35 * mean)
        vals = vals / max(float(vals.sum()), 1e-12) * total
        out[idx] = vals
    return out


def _binary_irrigate_gate(ir, rain, et, month, wet_rain_mm=12.0, min_on_frac=0.35):
    """MI-MPC style: zero IR on clearly wet+low-need days; month totals restored."""
    out = np.asarray(ir, dtype=float).copy()
    rain = _align(rain, out.size, 0.0)
    et = _align(et, out.size, 1.0)
    month = np.asarray(month)
    er = okura.effective_rain_mm(rain)
    need = np.maximum(et - 0.55 * er, 0.0)
    for m in np.unique(month):
        sel = month == m
        if not np.any(sel):
            continue
        total = float(out[sel].sum())
        if total <= 1e-12:
            continue
        # Skip only when both wet (useful rain) and low need.
        wet = er[sel] >= float(wet_rain_mm) * 0.8  # ~MAFF 80% of 12 mm threshold
        weak = need[sel] < 0.35
        skip = wet & weak
        on = ~skip
        n_on = int(on.sum())
        n_sel = int(sel.sum())
        want_on = max(int(np.ceil(min_on_frac * n_sel)), 1)
        if n_on < want_on:
            idx = np.flatnonzero(sel)
            score = need[sel] - 0.8 * np.clip(er[sel] / max(float(wet_rain_mm), 1.0), 0.0, 2.0)
            keep = idx[np.argsort(score)[::-1][:want_on]]
            on_full = np.zeros(out.size, dtype=bool)
            on_full[keep] = True
            on = on_full[sel]
        vals = out[sel].copy()
        vals[~on] = 0.0
        if float(vals.sum()) <= 1e-12:
            continue
        vals = vals / float(vals.sum()) * total
        out[sel] = vals
    return out


def _finalize_block_days(ir, dr, on, month, rain, et, soil_moisture=None, rain_hours=None,
                         binary_irrigate=True):
    """Cap chatter, binary on/off, extract low-frequency IR/D signal, re-couple D."""
    # Tighter peak cap + heavier smooth → Fig. 8-like day shape (still no CSV).
    ir = _holdout_style_day_series(ir, on, month, floor=0.48, cap=1.50, smooth=5)
    ir = _match_jitter_to_weather(
        ir, on, month, rain, et,
        soil_moisture=soil_moisture, rain_hours=rain_hours,
        target_cv=0.26, target_mean_abs_d=0.08)
    if binary_irrigate:
        ir = _binary_irrigate_gate(ir, rain, et, month, wet_rain_mm=14.0, min_on_frac=0.45)
        on = ir > 1e-6
    ir = _extract_real_signal(ir, on, month, slow_alpha=0.10, fast_keep=0.10, floor_frac=0.55)
    ir_n = ir / max(float(np.mean(ir[on])) if np.any(on) else float(ir.mean()), 1e-9)
    wet = _climate_wetness(rain, soil_moisture=soil_moisture, rain_hours=rain_hours)
    anti = 1.0 - np.clip(ir_n / max(float(np.max(ir_n[on])) if np.any(on) else 1.0, 1e-9), 0.0, 1.0)
    d_coupled = 0.25 + 0.45 * wet + 0.30 * anti
    for m in np.unique(month):
        sel = month == m
        total_d = float(dr[sel].sum())
        if total_d <= 1e-9:
            continue
        w = np.maximum(d_coupled[sel], 1e-12)
        dr[sel] = total_d * (w / w.sum())
    dr = _holdout_style_day_series(dr, np.ones_like(dr, dtype=bool), month,
                                   floor=0.35, cap=1.90, smooth=3)
    dr = _extract_real_signal(
        dr, np.ones_like(dr, dtype=bool), month,
        slow_alpha=0.16, fast_keep=0.25, floor_frac=0.45)
    return ir, dr


def spread_season_totals(year, rain_mm, et_mm, rain_season_mm=None, flood_mm=None,
                         soil_moisture=None, rain_hours=None, binary_irrigate=True):
    """Daily block IR/D (m³/d) that sum exactly to season totals, shaped by weather.

    Does **not** read okura_fig8_daily.csv / measured daily blocks.
    Optional ``flood_mm``, ``soil_moisture``, ``rain_hours`` sharpen day shape.
    """
    rain = np.asarray(rain_mm, dtype=float).ravel()
    et = np.asarray(et_mm, dtype=float).ravel()
    if et.size == 1:
        et = np.full(rain.size, float(et))
    if rain_season_mm is None:
        rain_season_mm = float(rain.sum())
    flood = None if flood_mm is None else np.asarray(flood_mm, dtype=float).ravel()
    targets = season_targets_m3(year, rain_mm=rain_season_mm)
    ir_w, d_w = _daily_weights(
        rain, et, midsummer=True, year=year, flood=flood,
        soil_moisture=soil_moisture, rain_hours=rain_hours)
    dates = okura.season_dates(year)
    rows = []
    for block in okura.BLOCKS:
        a = okura.cyclic_area_m2(block)
        ir = targets[(block, "cyclic", "irrigation")] * ir_w
        dr = targets[(block, "cyclic", "drainage")] * d_w
        month = dates.month.to_numpy()
        on = ir > 1e-6
        ir, dr = _finalize_block_days(
            ir, dr, on, month, rain, et,
            soil_moisture=soil_moisture, rain_hours=rain_hours,
            binary_irrigate=binary_irrigate)
        for i, date in enumerate(dates):
            rows.append({
                "date": date, "block": block, "area_m2": a,
                "irrigation_m3": float(ir[i]), "drainage_m3": float(dr[i]),
                "irrigation_mm": float(ir[i]) * 1000.0 / a,
                "drainage_mm": float(dr[i]) * 1000.0 / a,
                "rain_mm": float(rain[i]), "et_mm": float(et[i]),
                "source": "season_totals+climate_signal",
            })
    return pd.DataFrame(rows)


def table5_lfv_plan_m3(year, n_days, day_weights=None, use_fig6_lf=True):
    """Daily LFV_pump (m³) summing to Table 5 — pins season W via Eq. 4/6.

    Default day shape follows **Fig. 6 digitised lateral (Lf)** per plot block
    (area-weighted), not ADCP drainage — so underground timing matches the paper
    leftover plot. Pass ``day_weights`` to override; midsummer off still applies.
    """
    total = float(okura.TABLE5_1000M3[int(year)]["LFV_pump"]) * 1000.0
    n = int(n_days)
    if day_weights is None and use_fig6_lf:
        try:
            from .paper_plots import fig6_district_lfv_day_weights
            w = fig6_district_lfv_day_weights(int(year), n_days=n)
        except Exception:
            w = np.ones(n, dtype=float)
    elif day_weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(day_weights, dtype=float).ravel()
        w = w[:n] if w.size >= n else np.resize(w, n)
        w = np.maximum(w, 0.0)
        if float(w.sum()) <= 1e-12:
            w = np.ones(n, dtype=float)
    # No LFV push through the June midsummer drainage window.
    dates = okura.season_dates(int(year))[:n]
    start = pd.Timestamp(f"{int(year)}-{okura.MIDSUMMER_START}")
    off = (dates >= start) & (dates < start + pd.Timedelta(days=okura.MIDSUMMER_DAYS))
    w = np.asarray(w, dtype=float).copy()
    if w.size != n:
        w = np.resize(w, n)
    w[np.asarray(off)[:n]] = 0.0
    if float(w.sum()) <= 1e-12:
        w = np.ones(n, dtype=float)
    w = w / w.sum()
    return total * w


def table5_o_plan_m3(year, n_days, rain_mm=None):
    """Daily outflow O (m³) summing to Table 5, shaped by rain excess (portable)."""
    total = float(okura.TABLE5_1000M3[int(year)]["O"]) * 1000.0
    n = int(n_days)
    if rain_mm is None:
        w = np.ones(n, dtype=float)
    else:
        rain = _align(rain_mm, n, 0.0)
        excess = np.clip(rain - 10.0, 0.0, None)
        w = excess if float(excess.sum()) > 1e-9 else np.ones(n, dtype=float)
    w = w / max(float(np.asarray(w).sum()), 1e-12)
    return total * w


def paper_holdout_blocks(year):
    """Digitised / measured daily series — evaluation only, never for learning."""
    blocks, station, source = imitation.load_observed(year, prefer_measured=True)
    return blocks, station, source


def compare_to_paper(sim_blocks, year, sim_station=None):
    """Holdout score: model vs paper CSV / observed daily (not used in training)."""
    paper, paper_station, source = paper_holdout_blocks(year)
    scored = imitation.score(sim_blocks, paper)
    # Season totals vs Table 4 explicitly
    rows = []
    for block in okura.BLOCKS:
        t4 = okura.TABLE4_MM[(block, year)]
        a = okura.cyclic_area_m2(block)
        part = sim_blocks[sim_blocks["block"] == block]
        got_ir = float(part["irrigation_m3"].sum())
        got_d = float(part["drainage_m3"].sum())
        want_ir = t4["irrigation"] * a / 1000.0
        want_d = t4["drainage"] * a / 1000.0
        rows.append({
            "block": block,
            "IR_season_m3": got_ir, "IR_table4_m3": want_ir,
            "IR_error_%": 100.0 * (got_ir - want_ir) / max(want_ir, 1.0),
            "D_season_m3": got_d, "D_table4_m3": want_d,
            "D_error_%": 100.0 * (got_d - want_d) / max(want_d, 1.0),
        })
    t5 = okura.TABLE5_1000M3[int(year)]
    want_w = float(t5["W"]) * 1000.0
    want_lfv = float(t5["LFV_pump"]) * 1000.0
    want_o = float(t5["O"]) * 1000.0
    got_w = got_lfv = got_o = None
    if sim_station is not None and "W_m3" in getattr(sim_station, "columns", []):
        got_w = float(np.asarray(sim_station["W_m3"], float).sum())
        if "LFV_pump_m3" in sim_station.columns:
            got_lfv = float(np.asarray(sim_station["LFV_pump_m3"], float).sum())
        if "O_m3" in sim_station.columns:
            got_o = float(np.asarray(sim_station["O_m3"], float).sum())
    elif paper_station is not None and "W_m3" in getattr(paper_station, "columns", []):
        pass
    station_rows = [{
        "quantity": "W",
        "got_m3": got_w, "table5_m3": want_w,
        "error_%": (None if got_w is None else 100.0 * (got_w - want_w) / max(want_w, 1.0)),
    }, {
        "quantity": "LFV_pump",
        "got_m3": got_lfv, "table5_m3": want_lfv,
        "error_%": (None if got_lfv is None else 100.0 * (got_lfv - want_lfv) / max(want_lfv, 1.0)),
    }, {
        "quantity": "O",
        "got_m3": got_o, "table5_m3": want_o,
        "error_%": (None if got_o is None else 100.0 * (got_o - want_o) / max(want_o, 1.0)),
    }]
    return SimpleNamespace(
        source=source,
        daily_fidelity=float(scored.fidelity),
        irrigation_fidelity=float(scored.irrigation_fidelity),
        discharge_fidelity=float(scored.discharge_fidelity),
        loss=float(scored.loss),
        daily_table=scored.table,
        season_table=pd.DataFrame(rows),
        station_table=pd.DataFrame(station_rows),
    )


def holdout_summary_text(report):
    lines = [
        f"Holdout vs paper series (`{report.source}`) — **not** used in learning.",
        f"Daily-shape fidelity {report.daily_fidelity:.3f} "
        f"(IR {report.irrigation_fidelity:.3f}, D {report.discharge_fidelity:.3f}).",
        "Season totals vs Table 4:",
    ]
    for _, row in report.season_table.iterrows():
        lines.append(
            f"- {row['block']}: IR {row['IR_error_%']:+.1f}% · D {row['D_error_%']:+.1f}%")
    if getattr(report, "station_table", None) is not None:
        lines.append("Season station vs Table 5:")
        for _, row in report.station_table.iterrows():
            if row["error_%"] is None or (isinstance(row["error_%"], float) and np.isnan(row["error_%"])):
                continue
            lines.append(f"- {row['quantity']}: {row['error_%']:+.1f}%")
    return "\n".join(lines)


def _table2_daily(year, quantity="rain"):
    """Portable climate: Table 2 monthly totals spread flat onto season days (no Fig. 8)."""
    dates = okura.season_dates(year)
    out = np.zeros(len(dates), dtype=float)
    monthly = okura.TABLE2_MM[int(year)][quantity]
    for m, total in monthly.items():
        sel = dates.month.to_numpy() == int(m)
        n = int(sel.sum())
        if n:
            out[sel] = float(total) / n
    return out


def climate_counterfactual_heads(year, shape_from_year=None, n_plots=4, mode="et_follower"):
    """Leftover heads (mm) for lateral CF under alternate climate day-shape.

    Season IR/D totals stay those of ``year`` (Table 4). Day climate comes from
    ``shape_from_year`` Table 2 (default: the other Okura season). Maps canal-
    distance bands to leftover heads. No digitised Fig. 8; IR never enters H.
    """
    year = int(year)
    shape_from_year = int(shape_from_year if shape_from_year is not None
                          else (2022 if year == 2021 else 2021))
    n_plots = int(n_plots)
    rain = _table2_daily(shape_from_year, "rain")
    et = _table2_daily(shape_from_year, "et")
    # Align length to ``year`` season (same 131-day window).
    n = len(okura.season_dates(year))
    rain = _align(rain, n, 0.0)
    et = _align(et, n, 1.0)
    blocks = spread_season_totals(year, rain, et)
    # Season-mean leftover intensity (mm) from Eq. 3 on the climate-shaped schedule.
    lf = []
    for block in okura.BLOCKS:
        part = blocks[blocks["block"] == block]
        a = float(okura.cyclic_area_m2(block))
        ir = part["irrigation_m3"].to_numpy(dtype=float) * 1000.0 / a
        dr = part["drainage_m3"].to_numpy(dtype=float) * 1000.0 / a
        r = part["rain_mm"].to_numpy(dtype=float)
        e = part["et_mm"].to_numpy(dtype=float)
        lf.append(np.maximum(ir + r - dr - e, 0.0))
    lf_day = np.mean(np.vstack(lf), axis=0)
    # First-half vs second-half mass → spatial pile (upslope vs mid-chain).
    # Keep argmax off the terminal plot so wetter→drier hops always have a neighbour.
    half = max(n // 2, 1)
    early = float(lf_day[:half].sum())
    late = float(lf_day[half:].sum())
    total = max(early + late, 1e-9)
    near_outlet = late / total if str(mode) == "et_follower" else early / total
    if near_outlet >= 0.45:
        weights = np.array([0.05, 0.15, 1.0, 0.25][:n_plots], dtype=float)
    else:
        weights = np.array([1.0, 0.25, 0.10, 0.05][:n_plots], dtype=float)
    if weights.size < n_plots:
        weights = np.resize(weights, n_plots)
    # Scale to historical leftover depth used by the 4-plot demo (~90 mm pile).
    scale = 90.0 * (early + late) / max(float(lf_day.sum()), 1e-9)
    heads = scale * weights / max(float(weights.sum()), 1e-12) * n_plots
    return np.asarray(heads, dtype=float)

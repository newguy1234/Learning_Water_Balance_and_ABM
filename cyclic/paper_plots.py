"""Digitised Okura paper plots / tables → shared loaders and model targets.

Single place for Fig. 6 / Fig. 8 CSV paths. UI figures and FieldAggregate W pins
both read through here.

    Fig. 6   monthly water-balance bars → LFV day shape / leftover intensity
    Fig. 8   daily IR N/S + lake W
    Table 4  season Lf mm per cyclic block
    Table 5  LFV / LfA capture pin (via construct_ratios elsewhere)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import data, okura

FIG6_PATH = os.path.join(data.DATA, "observed", "okura_fig6_monthly.csv")
FIG8_PATH = os.path.join(data.DATA, "observed", "okura_fig8_daily.csv")


def ema(x, alpha=0.25):
    """Causal exponential moving average (shared by figures and day-shape helpers)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.empty_like(x)
    if x.size == 0:
        return y
    y[0] = x[0]
    a = float(np.clip(alpha, 0.0, 1.0))
    for i in range(1, x.size):
        y[i] = a * x[i] + (1.0 - a) * y[i - 1]
    return y


def load_fig6(year=None):
    """Digitised Fig. 6 monthly water-balance bars (mm/d means)."""
    frame = pd.read_csv(FIG6_PATH)
    if year is not None:
        frame = frame[frame["year"] == int(year)].copy()
    return frame


def load_fig8(year=None):
    """Digitised Fig. 8 daily IR North/South + lake withdrawal (m³)."""
    frame = pd.read_csv(FIG8_PATH, parse_dates=["date"])
    if year is not None:
        frame = frame[frame["year"] == int(year)].copy()
    return frame.sort_values("date")


def _fig8_aligned(year, dates, column):
    """One Fig. 8 column (m³/d) on ``dates``; missing days → 0 (hard copy, not interpolated)."""
    year = int(year)
    fig8 = load_fig8(year)
    fig8_dates = pd.to_datetime(fig8["date"]).dt.normalize()
    out = np.zeros(len(dates), dtype=float)
    for d, date in enumerate(dates):
        hit = fig8.loc[fig8_dates == pd.Timestamp(date), column]
        out[d] = float(hit.iloc[0]) if len(hit) else 0.0
    return out


def fig8_w_plan_m3(year, dates):
    """Lake W (m³/d) — digitised Fig. 8 hard copy; missing days → 0."""
    return _fig8_aligned(year, dates, "withdrawal_m3")


def fig8_ir_plan_m3(year, dates):
    """Cyclic IR North/South (m³/d) — digitised Fig. 8 hard copy; missing days → 0."""
    return {
        "North": _fig8_aligned(year, dates, "irrigation_North_m3"),
        "South": _fig8_aligned(year, dates, "irrigation_South_m3"),
    }


def table4_block_lf_mm(year):
    """Season lateral leftover depth (mm) per cyclic block — paper Table 4."""
    return {b: float(okura.TABLE4_MM[(b, int(year))]["lateral"]) for b in okura.BLOCKS}


def fig6_monthly_lateral_mm_d(year):
    """Fig. 6 lateral mm/d by (block, month); negatives clipped."""
    frame = load_fig6(year)
    out = {}
    for _, row in frame.iterrows():
        out[(str(row["block"]), int(row["month"]))] = max(float(row["lateral"]), 0.0)
    return out


def fig6_daily_leftover_weights(year, n_days=None):
    """Daily leftover mm from Fig. 6 Lf shape, rescaled to Table 4 season Lf."""
    year = int(year)
    dates = okura.season_dates(year)
    n = int(n_days) if n_days is not None else len(dates)
    dates = dates[:n]
    months = dates.month.to_numpy()
    monthly = fig6_monthly_lateral_mm_d(year)
    start = pd.Timestamp(f"{year}-{okura.MIDSUMMER_START}")
    mid = (dates >= start) & (dates < start + pd.Timedelta(days=okura.MIDSUMMER_DAYS))
    out = {}
    for block in okura.BLOCKS:
        raw = np.array([monthly.get((block, int(m)), 0.0) for m in months], dtype=float)
        raw = np.maximum(raw, 0.0)
        raw[np.asarray(mid)[:n]] = 0.0
        want = float(okura.TABLE4_MM[(block, year)]["lateral"])
        if float(raw.sum()) <= 1e-12:
            raw = np.ones(n, dtype=float)
            raw[np.asarray(mid)[:n]] = 0.0
        out[block] = raw / float(raw.sum()) * want
    return out


def fig6_district_lfv_day_weights(year, n_days=None):
    """District daily weights for LFV_pump from Fig. 6 Lf (area-weighted N+S)."""
    year = int(year)
    daily = fig6_daily_leftover_weights(year, n_days=n_days)
    n = len(next(iter(daily.values())))
    w = np.zeros(n, dtype=float)
    for block in okura.BLOCKS:
        w += daily[block] * okura.cyclic_area_m2(block)
    if float(w.sum()) <= 1e-12:
        return np.ones(n, dtype=float)
    return w / float(w.sum())


def block_leftover_intensity(year):
    """Relative leftover intensity per block from Table 4 Lf (mean 1)."""
    year = int(year)
    lf = table4_block_lf_mm(year)
    areas = {b: okura.cyclic_area_m2(b) for b in okura.BLOCKS}
    mean = sum(lf[b] * areas[b] for b in okura.BLOCKS) / max(sum(areas.values()), 1.0)
    if mean <= 1e-12:
        return {b: 1.0 for b in okura.BLOCKS}
    return {b: float(lf[b] / mean) for b in okura.BLOCKS}

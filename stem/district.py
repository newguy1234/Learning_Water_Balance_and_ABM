"""The district: 4,002 paddies of the Hakusan-Jimbee service area, their neighbours,
static features and one season's weather.

Paddies, blocks, cyclic status, cardinal neighbours, ground elevation (GSI 5 m DEM) and
distances come from the bundled export `stem/bundled/paddy_abm_<year>.nc`. Weather is
Open-Meteo (`data/open_meteo_<year>.csv`); rain is scaled month by month to the on-site
gauge (Okura et al. 2025, Table 2), which Open-Meteo overstates by 30–50 %.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import okura
from .params import BUNDLED, DATA, OUTPUTS, PROJECT  # noqa: F401  (re-exported)

CARDINAL = ("east", "west", "north", "south")
GROUPS = (("North", True), ("South", True), ("North", False), ("South", False))


@dataclass
class District:
    year: int
    n: int
    paddy_id: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    area_m2: np.ndarray
    block: np.ndarray
    cyclic: np.ndarray
    neighbours: np.ndarray        # (n, 4) east, west, north, south; -1 = none
    elevation_m: np.ndarray
    pool_distance_m: np.ndarray
    canal_distance_m: np.ndarray
    drain_entry_m: np.ndarray     # paddy → nearest mapped channel reaching the pool
    drain_path_m: np.ndarray      # along the channels to the station
    geometry_wkt: np.ndarray
    dates: pd.DatetimeIndex
    rain_mm: np.ndarray           # gauge-corrected
    et0_mm: np.ndarray            # Open-Meteo FAO-56 reference ET
    weather: pd.DataFrame         # all Open-Meteo daily columns, season rows

    @property
    def days(self):
        return len(self.dates)

    def group_index(self):
        """0..3 per paddy: (North, cyclic), (South, cyclic), (North, non-cyclic), (South, non-cyclic)."""
        out = np.empty(self.n, dtype=int)
        for g, (block, cyc) in enumerate(GROUPS):
            out[(self.block == block) & (self.cyclic == cyc)] = g
        return out


def bundled_path(year):
    return os.path.join(BUNDLED, f"paddy_abm_{int(year)}.nc")


def weather(year):
    frame = pd.read_csv(os.path.join(DATA, f"open_meteo_{int(year)}.csv"), index_col=0, parse_dates=True)
    dates = okura.season_dates(int(year))
    return frame.reindex(dates)


def gauge_rain(rain_mm, dates, year):
    """Scale each month to the site gauge (Okura Table 2); other years: mean gauged ratio."""
    rain = np.asarray(rain_mm, dtype=float).copy()
    months = np.asarray(pd.DatetimeIndex(dates).month)
    tables = okura.TABLE2_MM
    if int(year) in tables:
        factors = {m: total / max(rain[months == m].sum(), 1e-9)
                   for m, total in tables[int(year)]["rain"].items() if rain[months == m].sum() > 0}
    else:
        ratios = {}
        for y in tables:
            wx = weather(y)
            other = np.nan_to_num(wx["rain_mm"].to_numpy())
            om = np.asarray(wx.index.month)
            for m, total in tables[y]["rain"].items():
                if other[om == m].sum() > 0:
                    ratios.setdefault(m, []).append(total / other[om == m].sum())
        factors = {m: float(np.mean(v)) for m, v in ratios.items()}
    for m, f in factors.items():
        rain[months == m] *= f
    return rain


@functools.lru_cache(maxsize=4)
def load_district(year=2021):
    import xarray as xr

    year = int(year)
    ds = xr.open_dataset(bundled_path(year))
    wx = weather(year)
    dates = pd.DatetimeIndex(wx.index)
    elev = ds.ground_elev_m.values.astype(float)
    elev = np.where(np.isfinite(elev), elev, np.nanmean(elev))
    district = District(
        year=year, n=ds.sizes["paddy"], paddy_id=ds.paddy.values.astype(str),
        lon=ds.lon.values.astype(float), lat=ds.lat.values.astype(float),
        area_m2=ds.area_m2.values.astype(float), block=ds.block.values.astype(str),
        cyclic=ds.cyclic.values.astype(bool),
        neighbours=np.column_stack([ds["neighbour_" + c].values for c in CARDINAL]).astype(int),
        elevation_m=elev, pool_distance_m=ds.pool_distance_m.values.astype(float),
        canal_distance_m=ds.canal_distance_m.values.astype(float),
        drain_entry_m=ds.drain_entry_m.values.astype(float), drain_path_m=ds.drain_path_m.values.astype(float),
        geometry_wkt=ds.geometry_wkt.values.astype(str), dates=dates,
        rain_mm=gauge_rain(np.nan_to_num(wx["rain_mm"].to_numpy()), dates, year),
        et0_mm=np.nan_to_num(wx["et0_mm"].to_numpy()), weather=wx,
    )
    ds.close()
    return district


def bbox(district, pad_deg=0.005):
    return (float(district.lon.min()) - pad_deg, float(district.lat.min()) - pad_deg,
            float(district.lon.max()) + pad_deg, float(district.lat.max()) + pad_deg)

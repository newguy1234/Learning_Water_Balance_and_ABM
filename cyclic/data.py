"""The district's inputs: paddies, what the satellite saw on each, and the weather.

    paddies()               every paddy parcel: block, cyclic flag (near canal/pool), area
    paddy_ndvi(frame, year) Sentinel-2 NDVI averaged over each parcel's 10 m pixels
    weather(year)           Open-Meteo daily weather at the district centroid
    season_inputs(year)     all of it as arrays for one season, cached

Files it reads from data/ (copied from Main_Show/Project/Satellite):
    paddy_et_s2.nc          Sentinel-2 L2A NDVI, 10 m, 10-day medians, 2021
    paddy_et_<year>.nc      NDVI at 300 m, 10-day composites, used for a year no 10 m file covers
    area_weather_<year>.nc  Open-Meteo daily weather at the district centroid
Anything missing is rebuilt or downloaded: paddies from cyclic/blocks.py (the Fude
rebuild), weather from the Open-Meteo archive API.
"""

from __future__ import annotations

import functools
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import okura
from . import infrastructure
from .canal import distance_to_canal_m
from .physics import KC_FALLBACK, canopy_cover, kc_from_ndvi

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT, "data")
# The Hakusan-Jimbee irrigation pumping station and its pool (cyclic/infrastructure.py).
POOL_LONLAT = infrastructure.POOL_LONLAT
AREA_CRS = "EPSG:6677"                   # JGD2011 plane rectangular zone IX, metres
# Tried in order for a season: the first file holding that year's imagery is used. Put a
# 10 m Sentinel-2 series for another year in as paddy_et_s2_<year>.nc and it wins over 300 m.
SATELLITE_FILES = ("paddy_et_s2_{year}.nc", "paddy_et_s2.nc", "paddy_et_{year}.nc")

# Open-Meteo archive daily fields, stored under these names and shown to farmers as yesterday.
OPEN_METEO_DAILY = (
    ("rain_mm", "precipitation_sum"),
    ("et0_mm", "et0_fao_evapotranspiration"),
    ("radiation_mj", "shortwave_radiation_sum"),
    ("tmean_c", "temperature_2m_mean"),
    ("tmax_c", "temperature_2m_max"),
    ("tmin_c", "temperature_2m_min"),
    ("atmean_c", "apparent_temperature_mean"),
    ("atmax_c", "apparent_temperature_max"),
    ("atmin_c", "apparent_temperature_min"),
    ("wind_mean_kmh", "wind_speed_10m_mean"),
    ("wind_max_kmh", "wind_speed_10m_max"),
    ("wind_gust_kmh", "wind_gusts_10m_max"),
    ("wind_dir_deg", "wind_direction_10m_dominant"),
    ("rain_hours", "precipitation_hours"),
    ("snow_cm", "snowfall_sum"),
    ("weather_code", "weather_code"),
    ("sunshine_s", "sunshine_duration"),
    ("daylight_s", "daylight_duration"),
    ("rh_mean_pct", "relative_humidity_2m_mean"),
    ("rh_max_pct", "relative_humidity_2m_max"),
    ("rh_min_pct", "relative_humidity_2m_min"),
    ("dewpoint_c", "dew_point_2m_mean"),
    ("vpd_max_kpa", "vapour_pressure_deficit_max"),
    ("cloud_pct", "cloud_cover_mean"),
    ("pressure_hpa", "pressure_msl_mean"),
    ("soil_temp_c", "soil_temperature_0_to_7cm_mean"),
    ("soil_moisture", "soil_moisture_0_to_7cm_mean"),
)
WEATHER_OBS = tuple(name for name, _ in OPEN_METEO_DAILY) + ("sunshine_h", "daylight_h")


# --- paddies -------------------------------------------------------------------------------

def assign_cyclic_layout(block, area_raw_m2, dist_canal_m, dist_pool_m=None, cyclic_flag=None):
    """Mark cyclic vs non-cyclic from the paper's layout, scaled to Table 1.

    Okura et al. Section 2.1 / Fig. 1a: cyclic blocks (yellow/red) lie along Lake
    Kita-Imbanuma, connected to the main drainage canal that feeds the pool. A
    non-cyclic block (gray) is not connected to that canal; Fig. 2a shows it
    hydrologically separated, inland / away from the pool, and its drainage returns
    to the lake or the Tone rather than the ADCP canal. Table 1 gives the areas.
    ``cyclic_flag`` (from Okura Fig. 1a, cyclic/infrastructure.py) says which parcels are
    cyclic. Without it, each North/South parcel is ranked nearest the canal (then the
    pool) first and cyclic area is filled to Table 1, the rest is NCB. Either way each
    part's areas are scaled to total exactly Table 1.
    """
    block = np.asarray(block)
    raw = np.asarray(area_raw_m2, dtype=float)
    canal = np.asarray(dist_canal_m, dtype=float)
    pool = np.asarray(dist_pool_m if dist_pool_m is not None else canal, dtype=float)
    cyclic = np.zeros(block.shape, dtype=bool)
    area_m2 = raw.copy()
    for name, sizes in okura.TABLE1_HA.items():
        rows = np.flatnonzero(block == name)
        if rows.size == 0:
            continue
        rank = np.lexsort((pool[rows], canal[rows]))
        ordered = rows[rank]
        values = raw[ordered]
        if cyclic_flag is not None:
            take = np.asarray(cyclic_flag, dtype=bool)[ordered]
        else:
            before = np.cumsum(values) - values
            take = before < sizes["cyclic"] * 1e4
        cyclic[ordered[take]] = True
        if np.any(take):
            area_m2[ordered[take]] = values[take] * sizes["cyclic"] * 1e4 / values[take].sum()
        if np.any(~take):
            area_m2[ordered[~take]] = (values[~take] * sizes["non_cyclic"] * 1e4
                                       / values[~take].sum())
    return cyclic, area_m2


def _apply_cyclic_layout(frame):
    """Stamp cyclic flags and Table-1 areas onto a parcel table that already has lon/lat."""
    canal = distance_to_canal_m(frame["lon"].to_numpy(), frame["lat"].to_numpy())
    cyclic, area_m2 = assign_cyclic_layout(
        frame["block"].to_numpy(),
        frame["area_raw_m2"].to_numpy() if "area_raw_m2" in frame.columns else frame["area_m2"].to_numpy(),
        canal,
        frame["distance_m"].to_numpy() if "distance_m" in frame.columns else canal,
        cyclic_flag=frame["fig1_cyclic"].to_numpy() if "fig1_cyclic" in frame.columns else None,
    )
    frame = frame.copy()
    frame["canal_distance_m"] = canal
    heights = infrastructure.elevation()
    if len(heights) and "polygon_uuid" in frame.columns:
        # Ground elevation (GSI 5 m DEM): whether a paddy's farm drain can empty into the
        # main drainage canal when the canal runs high (backwater).
        frame["ground_elev_m"] = heights.reindex(frame["polygon_uuid"]).to_numpy(dtype=float)
    frame["cyclic"] = cyclic
    frame["area_m2"] = area_m2
    return frame


# Columns data/paddies.gpkg keeps so the district layout loads without recomputing anything.
SAVED_LAYOUT = ("block", "cyclic", "area_m2", "canal_distance_m", "distance_m", "paddy_id")


def _layout_stale(cache) -> bool:
    """True when a layout source (cyclic/bundled infrastructure files) is newer than the
    saved layout, so it must be rebuilt once."""
    sources = (infrastructure.NETWORK_FILE, infrastructure.MEMBERSHIP_FILE, infrastructure.ELEVATION_FILE)
    saved = os.path.getmtime(cache)
    return any(os.path.exists(path) and os.path.getmtime(path) > saved for path in sources)


def paddies(refresh=False):
    """Every paddy parcel in the district.

    Parcels, blocks and cyclic status come from cyclic/blocks.py: MAFF Fude polygons
    kept, blocked and flagged by Okura Fig. 1a (cyclic/infrastructure.py), areas scaled
    to Table 1 by assign_cyclic_layout. `drain_entry_m` / `drain_path_m` are each
    parcel's drainage route to the pool through the mapped channels.

    SAVED LAYOUT. The first build writes data/paddies.gpkg with every derived column
    (blocks, cyclic flags, areas, canal distances, drainage routes, ground elevation); later
    calls load it as it is. It is rebuilt only with ``refresh=True`` or when an
    infrastructure file in cyclic/bundled is newer than it.

    AREA. The rebuilt district is still larger than Table 1 (gross parcel outlines include
    levees and roads). `area_m2` is each parcel's own area scaled so that the cyclic and
    non-cyclic parts of each block total exactly Table 1; `area_raw_m2` is the polygon.
    """
    import geopandas as gpd

    cache = os.path.join(DATA, "paddies.gpkg")
    if os.path.exists(cache) and not refresh and not _layout_stale(cache):
        frame = gpd.read_file(cache)
        if all(name in frame.columns for name in SAVED_LAYOUT):
            # The saved layout already holds everything; load it as it is.
            frame["cyclic"] = frame["cyclic"].astype(bool)
            return frame
        return _apply_cyclic_layout(frame)

    from . import blocks as rebuild_script

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parcels = rebuild_script.load_parcels()
        rebuilt, _, _ = rebuild_script.rebuild(parcels)

    columns = ["polygon_uuid", "block", "geometry"] + (["fig1_cyclic"] if "fig1_cyclic" in rebuilt else [])
    frame = rebuilt[columns].reset_index(drop=True)
    metric = frame.to_crs(AREA_CRS)
    frame["area_raw_m2"] = metric.area.to_numpy()
    centres = metric.geometry.centroid
    pump = gpd.GeoSeries(gpd.points_from_xy([POOL_LONLAT[0]], [POOL_LONLAT[1]]),
                         crs="EPSG:4326").to_crs(AREA_CRS).iloc[0]
    frame["distance_m"] = centres.distance(pump).to_numpy()
    lonlat = centres.to_crs("EPSG:4326")
    frame["lon"] = lonlat.x.to_numpy()
    frame["lat"] = lonlat.y.to_numpy()
    if infrastructure.available():
        routes = infrastructure.drainage_routes(frame["lon"].to_numpy(), frame["lat"].to_numpy())
        frame["drain_entry_m"] = routes["entry_m"]
        frame["drain_path_m"] = routes["path_m"]
    frame = _apply_cyclic_layout(frame)
    frame["paddy_id"] = [f"{b[0]}{i:05d}" for i, b in enumerate(frame["block"])]

    os.makedirs(DATA, exist_ok=True)
    saved = frame.copy()
    saved["cyclic"] = saved["cyclic"].astype(int)
    saved.to_file(cache, driver="GPKG")
    return frame


# --- satellite -------------------------------------------------------------------------------

def paddy_ndvi(frame, year, refresh=False):
    """NDVI per paddy per Sentinel-2 composite: the mean over the 10 m pixels whose
    centres fall inside the parcel, or the pixel under its centroid for a parcel too
    small to contain one. NaN where cloud masked every pixel. None if no imagery."""
    try:
        import shapely
        import xarray as xr
    except ImportError:
        return None

    cache = os.path.join(DATA, f"paddy_ndvi_{year}.nc")
    found = None
    for pattern in SATELLITE_FILES:
        path = os.path.join(DATA, pattern.format(year=year))
        if not os.path.exists(path):
            continue
        with xr.open_dataset(path) as ds:
            times = pd.to_datetime(ds["time"].values)
            keep = np.flatnonzero(times.year == year)
            if keep.size == 0:
                continue
            found = os.path.basename(path)
            stamp = f"{os.stat(path).st_size}:{os.stat(path).st_mtime_ns}"
            if os.path.exists(cache) and not refresh:
                with xr.open_dataset(cache) as cached:
                    if (cached.attrs.get("source") == found
                            and cached.attrs.get("source_stamp") == stamp
                            and "kc_observed" in cached and "flood_frac" in cached
                            and np.array_equal(cached.paddy.values, frame["paddy_id"].to_numpy())):
                        return cached.load()
            lat = ds["lat"].values
            lon = ds["lon"].values
            cube = ds["ndvi"].isel(time=keep).values.astype(np.float32)
        break
    if found is None:
        return None
    from .satellite_et import FLOOD_VOTES, LSWI_DRY, LSWI_WET, kc_observed, water_indices
    lswi_cube, votes_cube = water_indices(year, times[keep], cube.shape[1:])
    lat0, dlat = lat[0], lat[1] - lat[0]
    lon0, dlon = lon[0], lon[1] - lon[0]

    n = len(frame)
    out = np.full((cube.shape[0], n), np.nan, dtype=np.float32)
    pixels = np.zeros(n, dtype=np.int32)
    kc_out = np.full_like(out, np.nan)
    flood_out = np.full_like(out, np.nan)
    for k, geom in enumerate(frame.geometry.values):
        minx, miny, maxx, maxy = geom.bounds
        rows = sorted((int(np.floor((maxy - lat0) / dlat)), int(np.ceil((miny - lat0) / dlat))))
        cols = (int(np.floor((minx - lon0) / dlon)), int(np.ceil((maxx - lon0) / dlon)))
        ii, jj = np.meshgrid(np.arange(max(rows[0], 0), min(rows[1], lat.size - 1) + 1),
                             np.arange(max(cols[0], 0), min(cols[1], lon.size - 1) + 1),
                             indexing="ij")
        inside = shapely.contains_xy(geom, lon[jj], lat[ii])
        if inside.any():
            ri, rj = ii[inside], jj[inside]
        else:
            centre = geom.centroid
            # Clipped: on a coarse grid a paddy at the district's edge can round one pixel past it.
            ri = np.array([int(np.clip(round((centre.y - lat0) / dlat), 0, lat.size - 1))])
            rj = np.array([int(np.clip(round((centre.x - lon0) / dlon), 0, lon.size - 1))])
        values = cube[:, ri, rj]
        # The satellite ET of each 10 m pixel, then its mean over the paddy: a paddy half
        # flooded and half dry is the average of the two, not the ET of its average NDVI.
        kc_pixels = kc_observed(values, None if lswi_cube is None else lswi_cube[:, ri, rj],
                                None if votes_cube is None else votes_cube[:, ri, rj])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[:, k] = np.nanmean(values, axis=1)
            kc_out[:, k] = np.nanmean(kc_pixels, axis=1)
            if votes_cube is not None:
                flood_out[:, k] = np.nanmean(
                    (votes_cube[:, ri, rj] >= FLOOD_VOTES).astype(np.float32), axis=1)
            elif lswi_cube is not None:
                wet = np.clip((lswi_cube[:, ri, rj] - LSWI_DRY) / (LSWI_WET - LSWI_DRY), 0.0, 1.0)
                flood_out[:, k] = np.nanmean(wet, axis=1)
            else:
                flood_out[:, k] = 1.0
        pixels[k] = ri.size

    ds = xr.Dataset(
        {"ndvi": (("time", "paddy"), out), "kc_observed": (("time", "paddy"), kc_out),
         "flood_frac": (("time", "paddy"), flood_out), "pixels": (("paddy",), pixels)},
        coords={"time": times[keep], "paddy": frame["paddy_id"].to_numpy()},
        attrs={"source": found, "source_stamp": stamp,
               "resolution_m": round(abs(float(lat[1] - lat[0])) * 111_000.0),
               "method": "mean NDVI over pixel centres inside parcel, else the pixel under its centroid",
               "kc_observed": ("FAO-56 dual Kc per pixel from NDVI canopy cover and Sentinel-2 flood votes / LSWI "
                               + ("(water indices used)" if lswi_cube is not None else "(no water indices: surface assumed wet)"))},
    )
    ds.to_netcdf(cache)
    return ds


# --- weather ---------------------------------------------------------------------------------

def weather(year, span="season"):
    """Open-Meteo daily weather at the district centroid.

    span  "season"  April 17–August 25 (the paper)
          "year"    1 January–31 December
    """
    dates = okura.calendar_dates(year) if span == "year" else okura.season_dates(year)
    path = os.path.join(DATA, f"area_weather_{year}.nc")
    csv = os.path.join(DATA, f"open_meteo_{year}.csv" if span == "season" else f"open_meteo_{year}_year.csv")
    frame = None
    if os.path.exists(csv):
        frame = pd.read_csv(csv, index_col=0, parse_dates=True)
    elif span == "season" and os.path.exists(path):
        import xarray as xr

        with xr.open_dataset(path) as ds:
            columns = {}
            for name, api in OPEN_METEO_DAILY:
                key = api if api in ds else name
                if key in ds:
                    columns[name] = ds[key].values
            frame = pd.DataFrame(columns, index=pd.to_datetime(ds["time"].values))
            if "sunshine_s" in frame and "sunshine_h" not in frame:
                frame["sunshine_h"] = frame["sunshine_s"] / 3600.0
            if "daylight_s" in frame and "daylight_h" not in frame:
                frame["daylight_h"] = frame["daylight_s"] / 3600.0
    if frame is None or any(column not in frame.columns for column in WEATHER_OBS):
        frame = download_weather(year, dates)
        os.makedirs(DATA, exist_ok=True)
        frame.to_csv(csv)
    return frame.reindex(dates).astype(float)


def download_weather(year, dates=None):
    """The Open-Meteo archive API at the district centroid. No key needed."""
    import requests

    dates = okura.season_dates(year) if dates is None else dates
    response = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": 35.8204, "longitude": 140.2667,
        "start_date": pd.Timestamp(dates[0]).date().isoformat(),
        "end_date": pd.Timestamp(dates[-1]).date().isoformat(),
        "daily": ",".join(api for _, api in OPEN_METEO_DAILY),
        "timezone": "Asia/Tokyo",
    }, timeout=60)
    response.raise_for_status()
    daily = response.json()["daily"]
    frame = pd.DataFrame({name: daily[api] for name, api in OPEN_METEO_DAILY},
                         index=pd.to_datetime(daily["time"]))
    frame["sunshine_h"] = frame["sunshine_s"] / 3600.0
    frame["daylight_h"] = frame["daylight_s"] / 3600.0
    return frame


def station_rain(year, open_meteo):
    """Open-Meteo's daily rain rescaled to the site station's monthly totals (Table 2).

    Open-Meteo put 1,035 mm on the 2021 season where the station on site measured 791,
    and a water balance scored against that station should not start 244 mm wetter.
    Months the station did not publish keep the Open-Meteo value (calendar-year runs).
    """
    rain = np.nan_to_num(open_meteo["rain_mm"].to_numpy())
    if year not in okura.TABLE2_MM:
        return rain
    month = open_meteo.index.month.to_numpy()
    for m, total in okura.TABLE2_MM[year]["rain"].items():
        sel = month == m
        rain[sel] = okura._spread(open_meteo["rain_mm"].to_numpy()[sel], total)
    return rain


# --- one season as arrays --------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def season_inputs(year=okura.COMPARISON_YEAR, forcing="open_meteo", potential="pixel", span="season"):
    """Everything a season run reads, as arrays indexed [day] or [day, paddy].

    forcing  "open_meteo"  physics rain is Open-Meteo as published (default; accurate)
             "station"     rain rescaled to on-site Table 2 monthly totals
    Farmers always see Open-Meteo in observations either way.

    Satellite, per paddy:
      ndvi_daily    what the crop is doing each day -- a straight line through that
                    paddy's cloud-free passes; drives Kc in the physics
      sat_ndvi      what a farmer could know that morning: the latest cloud-free pass
                    up to yesterday (imagery arrives the day after acquisition)
      sat_et_mm     Kc(sat_ndvi) x Open-Meteo ET0 on the pass day
      sat_days_ago  how old that pass is; NaN before the first one
    """
    frame = paddies()
    wx = weather(year, span)
    dates = wx.index
    days, n = len(dates), len(frame)

    om_rain = np.nan_to_num(wx["rain_mm"].to_numpy())
    om_et0 = np.nan_to_num(wx["et0_mm"].to_numpy())
    rain = station_rain(year, wx) if forcing == "station" else om_rain
    seen_weather = {f"seen_{name}": np.nan_to_num(wx[name].to_numpy()) for name in WEATHER_OBS}

    sat = paddy_ndvi(frame, year)
    flood_daily = np.ones((days, n), dtype=np.float32)
    if sat is None:
        ndvi_daily = np.full((days, n), np.nan, dtype=np.float32)
        sat_ndvi = np.full((days, n), np.nan, dtype=np.float32)
        sat_kc = np.full((days, n), np.nan, dtype=np.float32)
        last_day = np.full((days, n), np.nan)
        pass_dates = pd.DatetimeIndex([])
    else:
        pass_dates = pd.to_datetime(sat["time"].values)
        nd = sat["ndvi"].values
        kc_obs = sat["kc_observed"].values if "kc_observed" in sat else kc_from_ndvi(nd)
        flood_obs = sat["flood_frac"].values if "flood_frac" in sat else np.ones_like(nd)
        x = (pass_dates - dates[0]).days.to_numpy()
        t = np.arange(days)
        ndvi_daily = np.full((days, n), np.nan, dtype=np.float32)
        for k in range(n):
            ok = np.isfinite(nd[:, k])
            if ok.any():
                ndvi_daily[:, k] = np.interp(t, x[ok], nd[ok, k])
            wet = np.isfinite(flood_obs[:, k])
            if wet.any():
                flood_daily[:, k] = np.clip(np.interp(t, x[wet], flood_obs[wet, k]), 0.0, 1.0)

        seen = np.full((days, n), np.nan, dtype=np.float32)
        seen_on = np.full((days, n), np.nan)
        seen_kc = np.full((days, n), np.nan, dtype=np.float32)
        for p, day in enumerate(x):
            visible = max(day + 1, 0)
            if visible >= days:
                continue
            ok = np.isfinite(nd[p])
            seen[visible, ok] = nd[p, ok]
            seen_on[visible, ok] = day
            seen_kc[visible, ok] = kc_obs[p, ok]
        sat_ndvi = pd.DataFrame(seen).ffill().to_numpy(dtype=np.float32)
        last_day = pd.DataFrame(seen_on).ffill().to_numpy()
        sat_kc = pd.DataFrame(seen_kc).ffill().to_numpy(dtype=np.float32)

    days_ago = np.arange(days)[:, None] - last_day
    pass_et0 = np.where(np.isfinite(last_day),
                        np.interp(np.nan_to_num(last_day), np.arange(days), om_et0), np.nan)
    sat_et = sat_kc * pass_et0
    kc_ndvi = np.where(np.isfinite(ndvi_daily), kc_from_ndvi(ndvi_daily), KC_FALLBACK)

    # --- the ET each paddy should transpire -------------------------------------------------
    # On a paddy's own cloud-free pass days its target is its own pixels' ET, Kc(NDVI) x ET0.
    # Between passes it follows the field-wide ET curve -- the block's area-weighted Kc x ET0
    # -- scaled by that paddy's ratio to the field at its passes, interpolated. So a paddy that
    # ran 5% above its field at one pass and 2% below at the next is asked to go from one to
    # the other along the field's own day-to-day shape.
    blocks_of = frame["block"].to_numpy()
    area = frame["area_m2"].to_numpy(dtype=float)
    field_et = kc_ndvi * om_et0[:, None]
    block_et = {b: (field_et[:, blocks_of == b] @ area[blocks_of == b]) / area[blocks_of == b].sum()
                for b in okura.BLOCKS}
    field_curve = np.column_stack([block_et[b] for b in blocks_of])
    ratio = np.ones((days, n))
    is_pass = np.zeros((days, n), dtype=bool)
    sat_pass_et = np.full((days, n), np.nan, dtype=np.float32)     # a value only on a fly-over
    if sat is not None:
        t = np.arange(days)
        inside = (x >= 0) & (x < days)
        for k in range(n):
            ok = inside & np.isfinite(nd[:, k])
            if not ok.any():
                continue
            pass_days = x[ok]
            observed_et = kc_obs[ok, k] * om_et0[pass_days]
            sat_pass_et[pass_days, k] = observed_et
            ratio[:, k] = np.interp(t, pass_days, observed_et / np.maximum(field_curve[pass_days, k], 1e-6))
            is_pass[pass_days, k] = True

    # potential="field"  every paddy's potential ET is its field's average from Open-Meteo,
    #                    ET0 x the field's Kc. The spread across paddies then has to come from
    #                    how each one was watered, and is what flyover.py scores against the
    #                    satellite. Its target is the field average, and its own pixels' ET on
    #                    its fly-over days.
    # potential="pixel" (default): retain each paddy's canopy in the dual-coefficient
    #                    physics and interpolate its own ET target between passes.
    #                    Actual ET still responds to local water stores and stress;
    #                    the satellite target is not imposed on the simulated flux.
    if potential == "pixel":
        et_target = ratio * field_curve
        kc_daily = np.where(om_et0[:, None] > 0, et_target / np.maximum(om_et0[:, None], 1e-6), kc_ndvi)
    elif potential == "field":
        et_target = np.where(is_pass, sat_pass_et, field_curve)
        kc_daily = np.where(om_et0[:, None] > 0, field_curve / np.maximum(om_et0[:, None], 1e-6), kc_ndvi)
    else:
        raise ValueError(f"potential is 'field' or 'pixel', not {potential!r}")

    # Canopy cover, for the simulation's transpiration (physics.advance). From each paddy's
    # own NDVI in pixel mode; in field mode the block's area-weighted average, so that every
    # paddy in a field starts from the same crop and any difference in their ET comes from how
    # each one was watered. Without imagery, a ramp to 0.9 over the first 60 days.
    ramp = np.clip(np.arange(days) / 60.0, 0.05, 0.9)[:, None]
    fc_pixel = canopy_cover(ndvi_daily)
    fc_pixel = np.where(np.isfinite(fc_pixel), fc_pixel, ramp)
    fc_block = {b: (fc_pixel[:, blocks_of == b] @ area[blocks_of == b]) / area[blocks_of == b].sum()
                for b in okura.BLOCKS}
    fc_field = np.column_stack([fc_block[b] for b in blocks_of])
    fc_daily = fc_pixel if potential == "pixel" else fc_field

    return SimpleNamespace(
        year=year, forcing=forcing, frame=frame, dates=dates,
        paddy_id=frame["paddy_id"].to_numpy(), block=blocks_of,
        cyclic=frame["cyclic"].to_numpy(dtype=bool), area_m2=area,
        rain_mm=rain, et0_mm=om_et0, radiation_mj=wx["radiation_mj"].to_numpy(),
        **seen_weather,
        ndvi_daily=ndvi_daily, kc_ndvi_daily=kc_ndvi, kc_daily=kc_daily,
        et_target_mm=et_target.astype(np.float32), is_pass=is_pass, field_et_mm=block_et,
        sat_pass_et_mm=sat_pass_et, potential=potential,
        fc_daily=fc_daily.astype(np.float32), fc_pixel_daily=fc_pixel.astype(np.float32),
        sat_ndvi=sat_ndvi, sat_et_mm=sat_et.astype(np.float32), sat_days_ago=days_ago.astype(np.float32),
        flood_daily=flood_daily.astype(np.float32),
        pass_dates=pass_dates,
        canal_distance_m=distance_to_canal_m(frame["lon"].to_numpy(), frame["lat"].to_numpy()),
        satellite_source=("none" if sat is None else
                          f"{sat.attrs.get('source', 'paddy_et_s2.nc')} "
                          f"({sat.attrs.get('resolution_m', 10)} m)"),
    )

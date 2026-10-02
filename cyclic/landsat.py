"""Thermal ET per paddy from Landsat 7/8/9 Collection 2 Level-2 (Planetary Computer).

Why: Sentinel-2 ET in this project is FAO-56 Kc(NDVI) × ET0, the same model the
simulation runs, so it cannot show a paddy short of water (satellite_et.py). Land
surface temperature can: a paddy that has dried out is hotter than a flooded one.

Pipeline, per clear Landsat pass:
    1. read the district window of surface temperature (ST), red, NIR and QA_PIXEL
       (Collection 2 L2, delivered at 30 m; thermal native 60 m Landsat 7, 100 m 8/9)
    2. mask fill, cloud, dilated cloud, cirrus and shadow (QA_PIXEL bits 0–4)
    3. sharpen ST to 30 m with NDVI (TsHARP, Kustas et al. 2003): at the native
       thermal scale fit ST = a + b·NDVI, then ST_30 = a + b·NDVI_30 + coarse residual,
       so the block mean stays what the sensor measured
    4. SSEBop ET fraction (Senay et al. 2013; Senay 2018):
           Tc  = c · Tmax_air        c from the coldest well-vegetated clear pixels
           dT  = Rn · r_ah / (ρ Cp)  clear-sky daily Rn (FAO-56), r_ah = 110 s/m
           ETf = (Tc + dT − Ts) / dT, clipped to 0–1.05
           ETa = ETf × k × ET0       k = 1.25 for grass-reference ET0 (Senay 2013)
    5. area-weighted mean over each paddy polygon on a 10 m subgrid

`download_season(year)` caches each scene window under data/landsat/;
`paddy_thermal_et(year)` writes data/landsat/paddy_et_landsat_<year>.npz.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np

from . import data, okura

LANDSAT_DIR = os.path.join(data.PROJECT, "data", "landsat")
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "landsat-c2-l2"

# QA_PIXEL bits that make a pixel unusable: fill, dilated cloud, cirrus, cloud, shadow.
QA_BAD_BITS = (0, 1, 2, 3, 4)
ST_SCALE, ST_OFFSET = 0.00341802, 149.0          # Kelvin
SR_SCALE, SR_OFFSET = 0.0000275, -0.2
NATIVE_THERMAL_M = {"landsat-7": 60.0, "landsat-8": 100.0, "landsat-9": 100.0}

# SSEBop constants (Senay et al. 2013; Senay 2018).
R_AH = 110.0          # s/m, aerodynamic resistance of the dry reference
CP = 1013.0           # J/kg/K
K_GRASS = 1.25        # ETa = ETf · k · ET0 with grass-reference ET0
COLD_NDVI = 0.7       # well-vegetated pixels that define the cold reference
MIN_CLEAR_FRAC = 0.3  # of district pixels, for a pass to be used


def _season_window(year):
    dates = okura.season_dates(int(year))
    return str(dates[0].date()), str(dates[-1].date())


def _catalog():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def search_scenes(year, bbox=None):
    from .copernicus import district_bbox

    bbox = bbox or district_bbox()
    start, end = _season_window(year)
    items = list(_catalog().search(collections=[COLLECTION], bbox=bbox,
                                   datetime=f"{start}/{end}").items())
    return sorted(items, key=lambda it: it.datetime), bbox


def _asset_keys(item):
    thermal = "lwir11" if "lwir11" in item.assets else "lwir"
    return {"st": thermal, "red": "red", "nir": "nir08", "qa": "qa_pixel"}


def _read_window(href, bbox, ref=None):
    """Read `bbox` (lon/lat) from a COG; with `ref` (transform, shape, crs) align to it."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    with rasterio.open(href) as src:
        if ref is None:
            bounds = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
            window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=window, boundless=True, fill_value=0)
            return arr, src.window_transform(window), str(src.crs)
        transform, shape, crs = ref
        window = from_bounds(transform.c, transform.f + transform.e * shape[0],
                             transform.c + transform.a * shape[1], transform.f,
                             transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=window, boundless=True, fill_value=0)
        return arr[:shape[0], :shape[1]], transform, crs


def download_season(year, progress=print):
    """Cache the district window of every season scene; returns the cached paths."""
    os.makedirs(LANDSAT_DIR, exist_ok=True)
    items, bbox = search_scenes(year)
    paths = []
    for item in items:
        path = os.path.join(LANDSAT_DIR, f"{item.id}.npz")
        if os.path.isfile(path):
            paths.append(path)
            continue
        keys = _asset_keys(item)
        qa, transform, crs = _read_window(item.assets[keys["qa"]].href, bbox)
        ref = (transform, qa.shape, crs)
        bands = {name: _read_window(item.assets[key].href, bbox, ref)[0]
                 for name, key in keys.items() if name != "qa"}
        np.savez_compressed(
            path, qa=qa, **bands, transform=np.array(transform)[:6], crs=crs,
            meta=json.dumps({"id": item.id, "datetime": item.datetime.isoformat(),
                             "platform": item.properties.get("platform"),
                             "scene_cloud": item.properties.get("eo:cloud_cover")}))
        if progress:
            progress(f"{item.id}: {qa.shape} px, scene cloud {item.properties.get('eo:cloud_cover')}%")
        paths.append(path)
    return paths


def load_scene(path):
    from affine import Affine

    blob = np.load(path, allow_pickle=False)
    meta = json.loads(str(blob["meta"]))
    qa = blob["qa"].astype(np.int64)
    bad = np.zeros(qa.shape, dtype=bool)
    for bit in QA_BAD_BITS:
        bad |= (qa >> bit) & 1 == 1
    st = blob["st"].astype(float) * ST_SCALE + ST_OFFSET
    red = blob["red"].astype(float) * SR_SCALE + SR_OFFSET
    nir = blob["nir"].astype(float) * SR_SCALE + SR_OFFSET
    valid = ~bad & (blob["st"] > 0) & (blob["red"] > 0) & (blob["nir"] > 0)
    ndvi = np.where(valid, (nir - red) / np.maximum(nir + red, 1e-6), np.nan)
    st = np.where(valid, st, np.nan)
    return SimpleNamespace(st_k=st, ndvi=ndvi, valid=valid, transform=Affine(*blob["transform"]),
                           crs=str(blob["crs"]), **meta)


def sharpen(st_k, ndvi, factor):
    """TsHARP: Ts–NDVI regression at the native thermal scale, residual kept per block."""
    if factor <= 1:
        return st_k
    h, w = st_k.shape
    hh, ww = h // factor * factor, w // factor * factor
    blocks = lambda a: a[:hh, :ww].reshape(hh // factor, factor, ww // factor, factor)
    with np.errstate(invalid="ignore"):
        ts_c = np.nanmean(blocks(st_k), axis=(1, 3))
        nd_c = np.nanmean(blocks(ndvi), axis=(1, 3))
    ok = np.isfinite(ts_c) & np.isfinite(nd_c)
    if ok.sum() < 30:
        return st_k
    b, a = np.polyfit(nd_c[ok], ts_c[ok], 1)
    resid = np.repeat(np.repeat(ts_c - (a + b * nd_c), factor, 0), factor, 1)
    out = st_k.copy()
    fine = a + b * ndvi[:hh, :ww] + resid
    out[:hh, :ww] = np.where(np.isfinite(fine), fine, st_k[:hh, :ww])
    return out


def _clear_sky_rn(day_inputs, elevation_m=5.0):
    """FAO-56 daily net radiation (MJ/m²/d → W/m²) from shortwave, Tmax/Tmin and RH."""
    rs = day_inputs["radiation_mj"]
    tmax, tmin = day_inputs["tmax_c"], day_inputs["tmin_c"]
    rh = day_inputs["rh_mean"]
    lat = np.radians(35.81)
    j = day_inputs["doy"]
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365)
    decl = 0.409 * np.sin(2 * np.pi * j / 365 - 1.39)
    ws = np.arccos(-np.tan(lat) * np.tan(decl))
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(lat) * np.sin(decl)
                                         + np.cos(lat) * np.cos(decl) * np.sin(ws))
    rso = (0.75 + 2e-5 * elevation_m) * ra
    es = lambda t: 0.6108 * np.exp(17.27 * t / (t + 237.3))
    ea = rh / 100.0 * 0.5 * (es(tmax) + es(tmin))
    sigma = 4.903e-9
    rnl = sigma * 0.5 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) * (0.34 - 0.14 * np.sqrt(ea)) \
        * (1.35 * min(rs / max(rso, 1e-6), 1.0) - 0.35)
    rn = (1 - 0.23) * rs - rnl
    return rn * 1e6 / 86400.0


def ssebop_etf(st_k, ndvi, tmax_c, rn_w_m2, air_density=1.18):
    """SSEBop ET fraction and the scene's c-factor."""
    tmax_k = tmax_c + 273.15
    cold = np.isfinite(st_k) & (ndvi >= COLD_NDVI)
    if cold.sum() < 20:
        cold = np.isfinite(st_k) & (ndvi >= np.nanpercentile(ndvi, 90))
    tc_pixels = st_k[cold]
    # Cold reference: a low percentile of well-vegetated pixels, not the single coldest.
    c = float(np.nanpercentile(tc_pixels, 10) / tmax_k)
    tc = c * tmax_k
    dt = max(rn_w_m2 * R_AH / (air_density * CP), 1.0)
    etf = np.clip((tc + dt - st_k) / dt, 0.0, 1.05)
    return etf, c, dt


def _paddy_index_grid(transform, shape, crs, subsample=3):
    """Paddy index (-1 none) on a subgrid `subsample`× finer than the scene grid."""
    from affine import Affine
    from rasterio.features import rasterize
    import geopandas as gpd
    from .paddy_export import load_paddy_dataset

    ds = load_paddy_dataset(2022)
    geoms = gpd.GeoSeries.from_wkt(ds.geometry_wkt.values, crs="EPSG:4326").to_crs(crs)
    fine = transform * Affine.scale(1.0 / subsample)
    grid = rasterize(((g, i) for i, g in enumerate(geoms)), out_shape=(shape[0] * subsample, shape[1] * subsample),
                     transform=fine, fill=-1, dtype="int32", all_touched=False)
    return grid, ds.paddy.values.astype(str)


def _paddy_means(values, grid, subsample, n):
    fine = np.repeat(np.repeat(values, subsample, 0), subsample, 1)
    fine = fine[:grid.shape[0], :grid.shape[1]]
    inside = (grid >= 0) & np.isfinite(fine)
    total = np.bincount(grid[inside], weights=fine[inside], minlength=n)
    count = np.bincount(grid[inside], minlength=n)
    total_all = np.bincount(grid[grid >= 0], minlength=n)
    mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    clear = np.where(total_all > 0, count / np.maximum(total_all, 1), 0.0)
    return mean, clear


def paddy_thermal_et(year, sharpen_thermal=True, min_paddy_clear=0.6, progress=print):
    """Per-paddy SSEBop ETf / ETa on every usable pass; saved and returned."""
    year = int(year)
    paths = download_season(year, progress=progress)
    inputs = data.season_inputs(year, "station", "pixel", "season")
    dates = [d.date() for d in inputs.dates]
    weather = {
        "radiation_mj": np.asarray(inputs.radiation_mj, dtype=float),
        "tmax_c": np.asarray(inputs.seen_tmax_c, dtype=float),
        "tmin_c": np.asarray(inputs.seen_tmin_c, dtype=float),
        "rh_mean": np.asarray(inputs.seen_rh_mean_pct, dtype=float),
    }
    n = len(inputs.paddy_id)
    grid = ids = None
    rows = []
    for path in paths:
        scene = load_scene(path)
        day = np.datetime64(scene.datetime[:10]).astype(object)
        if day not in dates:
            continue
        d = dates.index(day)
        district = np.isfinite(scene.st_k)
        if grid is None:
            grid, ids = _paddy_index_grid(scene.transform, scene.st_k.shape, scene.crs)
            if not np.array_equal(ids, np.asarray(inputs.paddy_id).astype(str)):
                raise ValueError("paddy order differs between the export and season inputs")
        in_district = grid[::3, ::3][:scene.st_k.shape[0], :scene.st_k.shape[1]] >= 0
        clear_frac = float(district[in_district].mean()) if in_district.any() else 0.0
        if clear_frac < MIN_CLEAR_FRAC:
            if progress:
                progress(f"{scene.id}: {clear_frac:.0%} of district clear — skipped")
            continue
        factor = int(round(NATIVE_THERMAL_M.get(scene.platform, 100.0) / 30.0)) if sharpen_thermal else 1
        st = sharpen(scene.st_k, scene.ndvi, factor)
        day_w = {k: float(v[d]) for k, v in weather.items()}
        day_w["doy"] = day.timetuple().tm_yday
        rn = _clear_sky_rn(day_w)
        etf, c, dt = ssebop_etf(st, scene.ndvi, day_w["tmax_c"], rn)
        etf_p, clear_p = _paddy_means(etf, grid, 3, n)
        st_p, _ = _paddy_means(st, grid, 3, n)
        etf_p = np.where(clear_p >= min_paddy_clear, etf_p, np.nan)
        rows.append(dict(day=d, date=str(day), id=scene.id, platform=scene.platform, c=c, dt=dt,
                         rn=rn, clear=clear_frac, etf=etf_p, st=np.where(clear_p >= min_paddy_clear, st_p, np.nan),
                         eta=etf_p * K_GRASS * float(inputs.et0_mm[d])))
        if progress:
            progress(f"{scene.id} ({scene.platform}): district {clear_frac:.0%} clear, "
                     f"paddies {np.isfinite(etf_p).mean():.0%}, c={c:.3f} dT={dt:.1f} K, "
                     f"ETf median {np.nanmedian(etf_p):.2f}")
    out = os.path.join(LANDSAT_DIR, f"paddy_et_landsat_{year}.npz")
    np.savez_compressed(
        out, paddy_id=np.asarray(inputs.paddy_id).astype(str),
        day=np.array([r["day"] for r in rows], dtype=int),
        date=np.array([r["date"] for r in rows]), scene=np.array([r["id"] for r in rows]),
        platform=np.array([str(r["platform"]) for r in rows]),
        etf=np.array([r["etf"] for r in rows]).reshape(len(rows), n),
        eta_mm=np.array([r["eta"] for r in rows]).reshape(len(rows), n),
        st_k=np.array([r["st"] for r in rows]).reshape(len(rows), n),
        c_factor=np.array([r["c"] for r in rows]), dt_k=np.array([r["dt"] for r in rows]),
        district_clear=np.array([r["clear"] for r in rows]),
        meta=json.dumps({"sharpened": bool(sharpen_thermal), "k": K_GRASS, "r_ah": R_AH,
                         "min_paddy_clear": min_paddy_clear}))
    return out


def daily_thermal_et(year, et0_mm, days=None):
    """Daily ET per paddy (days, n): thermal ETf interpolated between clear passes × k × ET0.

    Each paddy's ETf is linear in time between its own clear passes and held flat before
    the first and after the last (the usual SSEBop / METRIC ETrF interpolation). A paddy
    never seen clear takes each pass's district median.
    """
    thermal = load_paddy_thermal_et(year)
    if thermal is None:
        return None
    et0 = np.asarray(et0_mm, dtype=float)
    days = len(et0) if days is None else int(days)
    pass_days = np.asarray(thermal["day"], dtype=float)
    etf = np.asarray(thermal["etf"], dtype=float)
    median = np.nanmedian(etf, axis=1)
    t = np.arange(days, dtype=float)
    out = np.empty((days, etf.shape[1]))
    for i in range(etf.shape[1]):
        seen = np.isfinite(etf[:, i])
        x, y = (pass_days[seen], etf[seen, i]) if seen.any() else (pass_days, median)
        out[:, i] = np.interp(t, x, y)
    return out * K_GRASS * et0[:days, None]


def load_paddy_thermal_et(year):
    path = os.path.join(LANDSAT_DIR, f"paddy_et_landsat_{int(year)}.npz")
    if not os.path.isfile(path):
        return None
    blob = np.load(path, allow_pickle=False)
    return {k: blob[k] for k in blob.files}

"""Per-paddy ET from historical satellite data, interpolated between fly-overs with Open-Meteo.

Land surface temperature shows which paddies are wet and which have dried out, so ET
comes from Landsat 7/8/9 Collection 2 Level-2 thermal imagery (Microsoft Planetary
Computer, no account needed). For each clear pass:

    1. read the district window of surface temperature (ST), red, NIR and QA_PIXEL
       (delivered at 30 m; thermal native 60 m Landsat 7, 100 m Landsat 8/9)
    2. mask fill, cloud, dilated cloud, cirrus and shadow (QA_PIXEL bits 0–4)
    3. sharpen ST to 30 m with NDVI (TsHARP, Kustas et al. 2003)
    4. SSEBop ET fraction (Senay et al. 2013; Senay 2018):
           Tc = c · Tmax_air (c from cool, well-vegetated pixels), dT = Rn · r_ah / (ρ Cp)
           ETf = (Tc + dT − Ts) / dT, clipped to 0–1.05
    5. area-weighted mean over each paddy polygon (10 m subgrid)

Between passes each paddy's ETf is interpolated linearly in time and multiplied by that
day's Open-Meteo ET0 (× k = 1.25 for grass reference, Senay 2013): `daily_et(district)`.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np

from stem.district import DATA, bbox as district_bbox, load_district

LANDSAT_DIR = os.path.join(DATA, "landsat")
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "landsat-c2-l2"
QA_BAD_BITS = (0, 1, 2, 3, 4)
ST_SCALE, ST_OFFSET = 0.00341802, 149.0
SR_SCALE, SR_OFFSET = 0.0000275, -0.2
NATIVE_THERMAL_M = {"landsat-7": 60.0, "landsat-8": 100.0, "landsat-9": 100.0}
R_AH, CP, K_GRASS = 110.0, 1013.0, 1.25
COLD_NDVI = 0.7
MIN_CLEAR_FRAC = 0.3
MIN_PADDY_CLEAR = 0.6


# --- download ---------------------------------------------------------------------------------

def search_scenes(district):
    import planetary_computer
    import pystac_client

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    window = f"{district.dates[0].date()}/{district.dates[-1].date()}"
    items = catalog.search(collections=[COLLECTION], bbox=district_bbox(district), datetime=window).items()
    return sorted(items, key=lambda it: it.datetime)


def _read_window(href, box, ref=None):
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    with rasterio.open(href) as src:
        if ref is None:
            bounds = transform_bounds("EPSG:4326", src.crs, *box, densify_pts=21)
            window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            return src.read(1, window=window, boundless=True, fill_value=0), src.window_transform(window), str(src.crs)
        transform, shape, crs = ref
        window = from_bounds(transform.c, transform.f + transform.e * shape[0],
                             transform.c + transform.a * shape[1], transform.f,
                             transform=src.transform).round_offsets().round_lengths()
        return src.read(1, window=window, boundless=True, fill_value=0)[:shape[0], :shape[1]], transform, crs


def download_season(district, progress=print):
    """Cache the district window of every scene in the season under data/landsat/."""
    os.makedirs(LANDSAT_DIR, exist_ok=True)
    box = district_bbox(district)
    paths = []
    for item in search_scenes(district):
        path = os.path.join(LANDSAT_DIR, f"{item.id}.npz")
        if not os.path.isfile(path):
            thermal = "lwir11" if "lwir11" in item.assets else "lwir"
            qa, transform, crs = _read_window(item.assets["qa_pixel"].href, box)
            ref = (transform, qa.shape, crs)
            bands = {name: _read_window(item.assets[key].href, box, ref)[0]
                     for name, key in (("st", thermal), ("red", "red"), ("nir", "nir08"))}
            np.savez_compressed(path, qa=qa, **bands, transform=np.array(transform)[:6], crs=crs,
                                meta=json.dumps({"id": item.id, "datetime": item.datetime.isoformat(),
                                                 "platform": item.properties.get("platform")}))
            if progress:
                progress(f"downloaded {item.id}")
        paths.append(path)
    return paths


# --- one scene → ETf --------------------------------------------------------------------------

def load_scene(path):
    from affine import Affine

    blob = np.load(path, allow_pickle=False)
    qa = blob["qa"].astype(np.int64)
    bad = np.zeros(qa.shape, dtype=bool)
    for bit in QA_BAD_BITS:
        bad |= (qa >> bit) & 1 == 1
    valid = ~bad & (blob["st"] > 0) & (blob["red"] > 0) & (blob["nir"] > 0)
    red = blob["red"] * SR_SCALE + SR_OFFSET
    nir = blob["nir"] * SR_SCALE + SR_OFFSET
    return SimpleNamespace(
        st_k=np.where(valid, blob["st"] * ST_SCALE + ST_OFFSET, np.nan),
        ndvi=np.where(valid, (nir - red) / np.maximum(nir + red, 1e-6), np.nan),
        transform=Affine(*blob["transform"]), crs=str(blob["crs"]), **json.loads(str(blob["meta"])))


def sharpen(st_k, ndvi, factor):
    """TsHARP: Ts–NDVI regression at the native thermal scale, block residual kept."""
    if factor <= 1:
        return st_k
    h, w = st_k.shape
    hh, ww = h // factor * factor, w // factor * factor
    blocks = lambda a: a[:hh, :ww].reshape(hh // factor, factor, ww // factor, factor)
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
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


def net_radiation_w_m2(row, doy, lat_deg=35.81, elevation_m=5.0):
    """FAO-56 daily net radiation from Open-Meteo shortwave, Tmax/Tmin and RH."""
    rs, tmax, tmin, rh = row["radiation_mj"], row["tmax_c"], row["tmin_c"], row["rh_mean_pct"]
    lat = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(-np.tan(lat) * np.tan(decl))
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.sin(ws))
    rso = (0.75 + 2e-5 * elevation_m) * ra
    es = lambda t: 0.6108 * np.exp(17.27 * t / (t + 237.3))
    ea = rh / 100.0 * 0.5 * (es(tmax) + es(tmin))
    rnl = 4.903e-9 * 0.5 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) * (0.34 - 0.14 * np.sqrt(ea)) \
        * (1.35 * min(rs / max(rso, 1e-6), 1.0) - 0.35)
    return ((1 - 0.23) * rs - rnl) * 1e6 / 86400.0


def ssebop_etf(st_k, ndvi, tmax_c, rn_w_m2, air_density=1.18):
    cold = np.isfinite(st_k) & (ndvi >= COLD_NDVI)
    if cold.sum() < 20:
        cold = np.isfinite(st_k) & (ndvi >= np.nanpercentile(ndvi, 90))
    tc = np.nanpercentile(st_k[cold], 10)
    dt = max(rn_w_m2 * R_AH / (air_density * CP), 1.0)
    return np.clip((tc + dt - st_k) / dt, 0.0, 1.05), float(tc / (tmax_c + 273.15)), dt


def _paddy_grid(district, transform, shape, crs, sub=3):
    import geopandas as gpd
    from affine import Affine
    from rasterio.features import rasterize

    geoms = gpd.GeoSeries.from_wkt(district.geometry_wkt, crs="EPSG:4326").to_crs(crs)
    return rasterize(((g, i) for i, g in enumerate(geoms)), out_shape=(shape[0] * sub, shape[1] * sub),
                     transform=transform * Affine.scale(1.0 / sub), fill=-1, dtype="int32")


def _paddy_mean(values, grid, n, sub=3):
    fine = np.repeat(np.repeat(values, sub, 0), sub, 1)[:grid.shape[0], :grid.shape[1]]
    inside = (grid >= 0) & np.isfinite(fine)
    total = np.bincount(grid[inside], weights=fine[inside], minlength=n)
    count = np.bincount(grid[inside], minlength=n)
    whole = np.bincount(grid[grid >= 0], minlength=n)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan), np.where(whole > 0, count / np.maximum(whole, 1), 0.0)


# --- season → per-paddy ETf on passes, daily ET -----------------------------------------------

def thermal_path(year):
    return os.path.join(LANDSAT_DIR, f"paddy_et_landsat_{int(year)}.npz")


def paddy_thermal_etf(year, progress=print):
    """Per-paddy SSEBop ETf on every usable pass of the season; saved and returned."""
    district = load_district(year)
    dates = [d.date() for d in district.dates]
    grid = None
    rows = []
    for path in download_season(district, progress):
        scene = load_scene(path)
        day = np.datetime64(scene.datetime[:10]).astype(object)
        if day not in dates:
            continue
        d = dates.index(day)
        if grid is None:
            grid = _paddy_grid(district, scene.transform, scene.st_k.shape, scene.crs)
        in_district = grid[::3, ::3][:scene.st_k.shape[0], :scene.st_k.shape[1]] >= 0
        clear = float(np.isfinite(scene.st_k)[in_district].mean())
        if clear < MIN_CLEAR_FRAC:
            continue
        st = sharpen(scene.st_k, scene.ndvi, int(round(NATIVE_THERMAL_M.get(scene.platform, 100.0) / 30.0)))
        row = district.weather.iloc[d]
        etf, c, dt = ssebop_etf(st, scene.ndvi, row["tmax_c"], net_radiation_w_m2(row, day.timetuple().tm_yday))
        etf_p, seen = _paddy_mean(etf, grid, district.n)
        rows.append((d, scene.id, np.where(seen >= MIN_PADDY_CLEAR, etf_p, np.nan)))
        if progress:
            progress(f"{scene.id}: {clear:.0%} clear, ETf median {np.nanmedian(rows[-1][2]):.2f}")
    np.savez_compressed(thermal_path(year), paddy_id=district.paddy_id,
                        day=np.array([r[0] for r in rows]), scene=np.array([r[1] for r in rows]),
                        etf=np.array([r[2] for r in rows]))
    return thermal_path(year)


def load_thermal_etf(year):
    path = thermal_path(year)
    if not os.path.isfile(path):
        paddy_thermal_etf(year)
    blob = np.load(path, allow_pickle=False)
    return {k: blob[k] for k in blob.files}


def daily_et(district):
    """(days, n) ET per paddy: ETf interpolated between its clear passes × k × ET0."""
    thermal = load_thermal_etf(district.year)
    if not np.array_equal(thermal["paddy_id"], district.paddy_id):
        raise ValueError("thermal ET paddy order differs from the district")
    days = np.asarray(thermal["day"], dtype=float)
    etf = np.asarray(thermal["etf"], dtype=float)
    median = np.nanmedian(etf, axis=1)
    t = np.arange(district.days, dtype=float)
    out = np.empty((district.days, district.n))
    for i in range(district.n):
        seen = np.isfinite(etf[:, i])
        out[:, i] = np.interp(t, *(days[seen], etf[seen, i]) if seen.any() else (days, median))
    return out * K_GRASS * district.et0_mm[:, None]

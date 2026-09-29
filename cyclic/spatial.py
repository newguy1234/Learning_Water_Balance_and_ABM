"""Satellite-cell water balances nested within paddies.

Pixel centres inside a parcel represent equal shares of that parcel's modelled
area. A parcel containing no centre uses its nearest source cell, explicitly
flagged as a fallback. This does not create detail finer than the source imagery.
Irrigation is uniform within a paddy; canopy and water stores vary by cell.
"""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import data
from .physics import canopy_cover, kc_from_ndvi


@lru_cache(maxsize=4)
def cells(year, forcing="open_meteo", span="season"):
    import shapely
    import xarray as xr

    inputs = data.season_inputs(year, forcing, "pixel", span)
    frame = inputs.frame.to_crs("EPSG:4326")
    source = None
    for pattern in data.SATELLITE_FILES:
        path = Path(data.DATA) / pattern.format(year=year)
        if not path.exists():
            continue
        with xr.open_dataset(path) as ds:
            times = pd.to_datetime(ds.time.values)
            keep = np.flatnonzero(times.year == year)
            if not keep.size:
                continue
            lat, lon = ds.lat.values, ds.lon.values
            cube = ds.ndvi.isel(time=keep).transpose("time", "lat", "lon").values
            times = times[keep]
            source = path.name
        break
    if source is None:
        return None
    from .satellite_et import kc_observed, water_indices
    lswi_cube, votes_cube = water_indices(year, times, cube.shape[1:])

    owners, rows, cols, fallback = [], [], [], []
    for k, geom in enumerate(frame.geometry):
        xmin, ymin, xmax, ymax = geom.bounds
        ri = np.flatnonzero((lat >= ymin) & (lat <= ymax))
        cj = np.flatnonzero((lon >= xmin) & (lon <= xmax))
        ii, jj = np.meshgrid(ri, cj, indexing="ij")
        mask = shapely.contains_xy(geom, lon[jj], lat[ii])
        r, c = ii[mask], jj[mask]
        missing = not r.size
        if missing:
            centre = geom.centroid
            r = np.array([np.argmin(abs(lat - centre.y))])
            c = np.array([np.argmin(abs(lon - centre.x))])
        owners.extend([k] * len(r))
        rows.extend(r)
        cols.extend(c)
        fallback.extend([missing] * len(r))
    owner = np.asarray(owners, dtype=int)
    rows, cols = np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
    values = cube[:, rows, cols]
    kc_values = kc_observed(values, None if lswi_cube is None else lswi_cube[:, rows, cols],
                            None if votes_cube is None else votes_cube[:, rows, cols])
    days = len(inputs.dates)
    x = (times - inputs.dates[0]).days.to_numpy()
    daily = np.empty((days, len(owner)), dtype=np.float32)
    passes = np.zeros_like(daily, dtype=bool)
    observed = np.full_like(daily, np.nan)
    for j, k in enumerate(owner):
        valid = np.isfinite(values[:, j])
        if valid.any():
            daily[:, j] = np.interp(np.arange(days), x[valid], values[valid, j])
        else:
            daily[:, j] = inputs.ndvi_daily[:, k]
        visible = valid & (x >= 0) & (x < days)
        d = x[visible]
        passes[d, j] = True
        observed[d, j] = kc_values[visible, j] * inputs.et0_mm[d]
    fc = canopy_cover(daily).astype(np.float32)
    fc = np.where(np.isfinite(fc), fc, inputs.fc_pixel_daily[:, owner])
    weights = 1.0 / np.bincount(owner, minlength=len(frame))[owner]
    from .lateral import grid_edges
    from .canal import distance_to_canal_m
    cell_lon, cell_lat = lon[cols], lat[rows]
    return SimpleNamespace(owner=owner, weights=weights, lon=cell_lon, lat=cell_lat,
                           row=rows, col=cols, edges=grid_edges(rows, cols, cell_lon, cell_lat),
                           canopy=fc, is_pass=passes, observed_et_mm=observed,
                           fallback=np.asarray(fallback), source=source,
                           resolution_m=round(abs(float(lat[1] - lat[0])) * 111_000),
                           canal_distance_m=distance_to_canal_m(cell_lon, cell_lat))


def aggregate(values, grid, n):
    """Cell depths/fluxes to paddy means; preserves modelled water volume."""
    return np.bincount(grid.owner, weights=np.asarray(values) * grid.weights, minlength=n)

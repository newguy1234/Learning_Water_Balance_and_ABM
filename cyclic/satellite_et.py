"""Satellite ET at the imagery's own resolution, tied to the water standing on the plot.

The NDVI-only crop coefficient (physics.kc_from_ndvi, Vu et al. 2005) spans 1.10-1.155, so
10 m NDVI from 0.1 to 0.9 became almost the same ET everywhere on a given day. This uses the
same FAO-56 dual crop coefficient as the simulation (physics.advance), so the observed and
simulated ET are the same quantity, per 10 m pixel:

    Kc_obs = Kcb + Ke

    Kcb  from canopy cover:  fc = ((NDVI - 0.15) / (0.80 - 0.15))^2   (Carlson & Ripley 1997)
                             Kcb = Kcb_bare + (Kcb_full - Kcb_bare) x fc
    Ke   evaporation from water or wet soil between the plants:
             Ke = min(Kr x (Kc_max - Kcb), few x Kc_max),   few = 1 - fc
         Kr = 1 where the pixel is flooded -- at least two of Sentinel-2's three flood tests
         agree (Xiao's LSWI + 0.05 >= NDVI, NDWI > 0, scene class water); otherwise from the
         land-surface water index as a wetness scale,
             Kr = clip((LSWI - LSWI_dry) / (LSWI_wet - LSWI_dry), 0, 1)

    satellite ET = Kc_obs x ET0 on the pass day

Without the water indices (years where only NDVI was kept) the surface is assumed wet, Kr = 1.

WHAT THIS STILL CANNOT DO. Every pixel shares one ET0, and no transpiration stress is
observable from reflectance, so at closed canopy in mid-season -- when the crop hides the
water and the flood tests see nothing -- neighbouring pixels still differ little. A measured,
not modelled, spatial ET needs a thermal band: Landsat 8/9 surface temperature (100 m,
delivered at 30 m) through an energy-balance ET fraction such as SSEBop, sharpened with this
10 m NDVI. That is the next source to add, not something reflectance can supply.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import data
from .physics import Physics, canopy_cover

LSWI_DRY = -0.10
LSWI_WET = 0.20
FLOOD_VOTES = 2


def kc_observed(ndvi, lswi=None, flood_votes=None, physics=None):
    """The observed dual crop coefficient. Arrays of any matching shape; NaN where NDVI is NaN."""
    p = physics or Physics()
    ndvi = np.asarray(ndvi, dtype=np.float32)
    fc = canopy_cover(ndvi)
    kcb = p.kcb_bare + (p.kcb_full - p.kcb_bare) * fc
    few = np.clip(1.0 - fc, 0.05, 1.0)
    kr = np.ones_like(ndvi, dtype=np.float32)
    if lswi is not None:
        lswi = np.asarray(lswi, dtype=np.float32)
        kr = np.where(np.isfinite(lswi), np.clip((lswi - LSWI_DRY) / (LSWI_WET - LSWI_DRY), 0.0, 1.0), kr)
    if flood_votes is not None:
        votes = np.asarray(flood_votes, dtype=np.float32)
        kr = np.where(np.nan_to_num(votes) >= FLOOD_VOTES, 1.0, kr)
    ke = np.clip(np.minimum(kr * (p.kc_max - kcb), few * p.kc_max), 0.0, None)
    return (kcb + ke).astype(np.float32)


def water_indices(year, times, shape):
    """LSWI and flood votes on the NDVI grid for these acquisition times, cloud-masked;
    (None, None) if there is no water-index file for the year or it does not line up."""
    import xarray as xr

    path = os.path.join(data.DATA, f"paddy_water_s2_{year}.nc")
    if not os.path.exists(path):
        return None, None
    with xr.open_dataset(path) as ds:
        index = pd.to_datetime(ds["time"].values).get_indexer(pd.DatetimeIndex(times))
        if (index < 0).any() or tuple(ds["lswi"].shape[1:]) != tuple(shape):
            return None, None
        valid = ds["valid"].isel(time=index).values.astype(bool)
        lswi = np.where(valid, ds["lswi"].isel(time=index).values, np.nan).astype(np.float32)
        votes = np.where(valid, ds["flood_votes"].isel(time=index).values, np.nan).astype(np.float32)
    return lswi, votes

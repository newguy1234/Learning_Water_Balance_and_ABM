# -*- coding: utf-8 -*-
"""Load the cyclic district map + discovered underground physics for the ABM.

This is the point of the ABM: agents act on the **same** spatial skeleton and
leftover-fate rules the cyclic project discovered — not a toy square grid.

Sources (no rediscovery):
  * ``cyclic.data.paddies()`` — Fude parcels, North/South, cyclic flag, lon/lat
  * ``cyclic.lateral.cardinal_edges`` + ``graph.leftover_graph`` — neighbour hops
  * ``cyclic.bundle.physics_as_namespace`` — joint capture / neighbour / stay
  * ``cyclic.canal.pixel_ratio_fields`` — per-paddy LFV vs neighbour vs stay
  * ``cyclic.agents.HeadEqualise`` — discovered baseline neighbour-trickle law
  * ``cyclic.data.weather`` — Open-Meteo rain / ET0 at the district
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from .agents import HeadEqualise
from .bundle import load_physics_bundle, physics_as_namespace
from .canal import distance_to_canal_m, pixel_ratio_fields
from .data import paddies as load_paddies
from .data import weather as load_weather
from .graph import leftover_graph
from .identify import construct_ratios
from .lateral import cardinal_edges
from .physics import Physics


@dataclass
class CyclicLandscape:
    """District map + frozen underground shares for one ABM world."""

    n: int
    lon: np.ndarray
    lat: np.ndarray
    area_m2: np.ndarray
    block: np.ndarray
    cyclic: np.ndarray
    paddy_id: np.ndarray
    canal_distance_m: np.ndarray
    pool_distance_m: np.ndarray
    owner_seed_order: np.ndarray  # indices sorted for contiguous ownership chunks
    edges: Any
    graph: Any
    neighbours: list[list[int]]
    capture_i: np.ndarray
    neighbour_i: np.ndarray
    stay_i: np.ndarray
    neighbour_of_remain: float
    shares: dict
    physics: Any
    hop: Any
    rain_mm: np.ndarray
    et0_mm: np.ndarray
    year: int
    claim: str
    # Optional geometries for plotting (may be None).
    geometry: Any = None
    ground_water_law: Any = None
    # Drainage route to the pool through the mapped channels (cyclic/infrastructure.py):
    # to the nearest connected channel, then along the channels to the station.
    drain_entry_m: Any = None
    drain_path_m: Any = None
    ground_elev_m: Any = None     # median ground elevation, m above T.P. (GSI 5 m DEM)


def _neighbour_lists(edges, n: int) -> list[list[int]]:
    nbrs: list[list[int]] = [[] for _ in range(n)]
    for name in ("east", "west", "north", "south"):
        arr = np.asarray(getattr(edges, name), dtype=int)
        for i, j in enumerate(arr):
            if 0 <= int(j) < n and int(j) != i:
                nbrs[i].append(int(j))
    # Unique preserve order.
    return [list(dict.fromkeys(row)) for row in nbrs]


def _build_landscape(year=2022, days=None):
    frame = load_paddies().reset_index(drop=True)
    n = len(frame)

    lon = frame["lon"].to_numpy(dtype=float)
    lat = frame["lat"].to_numpy(dtype=float)
    area = frame["area_m2"].to_numpy(dtype=float)
    block = frame["block"].to_numpy()
    cyclic = frame["cyclic"].to_numpy(dtype=bool)
    paddy_id = frame["paddy_id"].to_numpy()
    if "canal_distance_m" in frame.columns:
        dist = frame["canal_distance_m"].to_numpy(dtype=float)
    else:
        dist = distance_to_canal_m(lon, lat)
    # Straight-line distance to the pumping-station pool (cyclic.data.POOL_LONLAT) —
    # who the station serves first when it cannot cover every request.
    if "distance_m" in frame.columns:
        pool_dist = frame["distance_m"].to_numpy(dtype=float)
    else:
        pool_dist = dist

    edges = cardinal_edges(lon, lat)
    graph = leftover_graph(dist, edges, cyclic)
    neighbours = _neighbour_lists(edges, n)

    phys = physics_as_namespace(load_physics_bundle())
    tables = construct_ratios(year)
    # Prefer joint discovered capture; fall back to year pin / table shares.
    capture = float(phys.capture)
    pins = phys.pins.get(int(year)) or phys.pins.get(str(year))
    if pins and "capture" in pins:
        # Hold joint field mean as default; year pin available for scenarios.
        pass
    farm_to_main = float(tables.get("farm_to_main", 1.0))
    topsoil = float(
        tables["kz_ratio"]
        / max(tables["kz_ratio"] + tables["ground_ratio"], 1e-9)
    )
    nbr_remain = float(tables.get("neighbour_of_remain", 0.5))

    # Block intensity from Table 4 lateral (relative) — same idea as cyclic attach.
    from . import okura

    inten = {}
    for b in ("North", "South"):
        inten[b] = float(okura.TABLE4_MM[(b, year)]["lateral"])
    mean_i = 0.5 * (inten["North"] + inten["South"])
    inten = {k: v / max(mean_i, 1e-9) for k, v in inten.items()}

    p = Physics()
    shares = pixel_ratio_fields(
        n,
        dist,
        edges,
        cyclic,
        capture=capture,
        farm_to_main=farm_to_main,
        topsoil_frac=topsoil,
        canal_length_m=float(p.canal_length_m),
        area_m2=area,
        block=block,
        block_intensity=inten,
        hard_capture=False,
        neighbour_of_remain=nbr_remain,
    )
    capture_i = np.asarray(shares["pool_capture_share"], dtype=float)
    neighbour_i = np.asarray(shares["kx_ratio"], dtype=float) + np.asarray(
        shares["ky_ratio"], dtype=float
    )
    stay_i = np.asarray(shares["kz_ratio"], dtype=float) + np.asarray(
        shares["ground_ratio"], dtype=float
    )
    # Renormalise per cell to a simplex (numerical safety).
    total = np.maximum(capture_i + neighbour_i + stay_i, 1e-12)
    capture_i, neighbour_i, stay_i = capture_i / total, neighbour_i / total, stay_i / total

    # Ownership order: by block then canal distance (contiguous along canal).
    order = np.lexsort((dist, block == "South"))

    wx = load_weather(year, span="season")
    rain = np.nan_to_num(wx["rain_mm"].to_numpy(dtype=float))
    et0 = np.nan_to_num(wx["et0_mm"].to_numpy(dtype=float))
    if days is not None:
        rain = rain[: int(days)]
        et0 = et0[: int(days)]

    hop = HeadEqualise()
    geometry = frame["geometry"] if "geometry" in frame.columns else None
    route = {name: frame[name].to_numpy(dtype=float) if name in frame.columns else None
             for name in ("drain_entry_m", "drain_path_m", "ground_elev_m")}

    return CyclicLandscape(
        n=n,
        lon=lon,
        lat=lat,
        area_m2=area,
        block=block,
        cyclic=cyclic,
        paddy_id=paddy_id,
        canal_distance_m=dist,
        pool_distance_m=pool_dist,
        owner_seed_order=order,
        edges=edges,
        graph=graph,
        neighbours=neighbours,
        capture_i=capture_i,
        neighbour_i=neighbour_i,
        stay_i=stay_i,
        neighbour_of_remain=nbr_remain,
        shares=shares,
        physics=phys,
        hop=hop,
        rain_mm=rain,
        et0_mm=et0,
        year=int(year),
        claim=str(phys.claim or ""),
        geometry=geometry,
        **route,
    )


def export_paddy_dataset(year=2022, path=None):
    """Write satellite observations, parcel status and frozen fate inputs to NetCDF.

    No fitting occurs. Missing fitted coefficients are explicitly labelled as a
    bundled-share fallback. Ownership belongs to an ABM scenario, not this file.
    """
    import json
    from pathlib import Path
    import xarray as xr
    from . import data
    from .identify import load_paddy_ground_water_law

    land = _build_landscape(year)
    frame = data.paddies().reset_index(drop=True)
    satellite = data.paddy_ndvi(frame, year)
    if satellite is None:
        raise FileNotFoundError(f'No satellite observations for {year}')
    ds = satellite.sel(paddy=land.paddy_id).copy(deep=True)
    for name in ('lon', 'lat', 'area_m2', 'block', 'cyclic', 'canal_distance_m',
                 'pool_distance_m', 'capture_i', 'neighbour_i', 'stay_i',
                 'drain_entry_m', 'drain_path_m', 'ground_elev_m'):
        values = getattr(land, name)
        if values is None:
            continue
        ds[name] = ('paddy', values.astype('int8') if name == 'cyclic' else values)
    ds['geometry_wkt'] = ('paddy', frame.geometry.to_wkt().to_numpy())
    for name, values in land.shares.items():
        values = np.asarray(values)
        if values.shape == (land.n,):
            ds['share_' + name] = ('paddy', values)
    for name in ('i', 'j', 'dist_m', 'ux', 'uy'):
        ds['edge_' + name] = ('edge', getattr(land.edges, name))
    for direction, feature in zip(('east', 'west', 'north', 'south'),
                                  ('right', 'left', 'up', 'down')):
        indices = getattr(land.edges, direction)
        ds['neighbour_' + direction] = ('paddy', indices)
        ds['has_' + feature] = ('paddy', (indices >= 0).astype('int8'))
    wx = data.weather(year, span='season')
    ds = ds.assign_coords(day=wx.index.to_numpy())
    for name in ('rain_mm', 'et0_mm'):
        ds[name] = ('day', wx[name].to_numpy(dtype=float))
        ds[name].attrs['units'] = 'mm/day'
    law = load_paddy_ground_water_law()
    if law is not None and not (law.lon.shape == land.lon.shape and np.allclose(law.lon, land.lon)
                                and np.allclose(law.lat, land.lat)):
        # Fitted on another parcel layout: refit (outputs/lateral) before it can be used.
        import warnings
        warnings.warn('Fitted leftover-fate equation was fitted on a different parcel layout; '
                      'exporting bundled shares instead. Rerun the season runs and '
                      'identify.equation_from_cache to refit it.')
        law = None
    ds.attrs.update(schema_version=1, year=int(year), physics_json=json.dumps(land.physics.raw),
                    neighbour_of_remain=land.neighbour_of_remain,
                    cyclic_status_source=('Okura et al. (2025) Fig. 1a registered onto MAFF Fude parcels '
                                          '(cyclic/infrastructure.py); areas scaled to Table 1'
                                          if 'drain_path_m' in ds else
                                          'cyclic.data.assign_cyclic_layout; canal/pool rank scaled to Table 1, inferred layout'),
                    groundwater_mode='fitted_equation' if law is not None else 'bundled_shares',
                    equation='pool=clip(X @ pool_weights,0,leftover) on cyclic paddies; remaining=leftover-pool',
                    dynamic_inputs='leftover_mm and infiltrated_mm are supplied by the simulation each day')
    if law is not None:
        if not (np.allclose(law.lon, land.lon) and np.allclose(law.lat, land.lat)):
            raise ValueError('Fitted equation parcel layout does not match current cyclic paddies')
        ds.attrs['training_cyclic_status_differences'] = int(np.count_nonzero(law.cyclic != land.cyclic))
        ds = ds.assign_coords(fate_feature=list(law.fate_names))
        ds['pool_weights'] = ('fate_feature', law.pool_weights)
        ds['stay_weights'] = ('fate_feature', law.stay_weights)
        ds.attrs.update(train_year=law.train_year, eval_year=law.eval_year)
    ds['cyclic'].attrs.update(flag_values=np.array([0, 1], dtype='int8'), flag_meanings='non_cyclic cyclic')
    for name in ('canal_distance_m', 'pool_distance_m', 'drain_entry_m', 'drain_path_m', 'ground_elev_m'):
        if name in ds:
            ds[name].attrs['units'] = 'm'
    ds['area_m2'].attrs['units'] = 'm2'
    target = Path(path) if path is not None else Path(data.DATA) / f'paddy_abm_{year}.nc'
    target.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(target)
    return target


def load_paddy_dataset(year=2022, path=None):
    """Load an explicit export or the installed package's bundled season file."""
    from pathlib import Path
    import xarray as xr
    target = Path(path) if path is not None else Path(__file__).parent / 'bundled' / f'paddy_abm_{year}.nc'
    if not target.exists() and path is None:
        target = export_paddy_dataset(year)
    with xr.open_dataset(target) as source:
        ds = source.load()
    if ds.attrs.get('schema_version') != 1 or int(ds.attrs['year']) != int(year):
        raise ValueError('Incompatible paddy dataset schema or year')
    if len(np.unique(ds.paddy.values)) != ds.sizes['paddy']:
        raise ValueError('Duplicate paddy IDs')
    return ds


def load_cyclic_landscape(year=2022, days=None, dataset_path=None):
    """Consume exactly the parcel ordering, status and coefficients in the export."""
    import json
    import geopandas as gpd
    ds = load_paddy_dataset(year, dataset_path)
    n = ds.sizes['paddy']
    edges = SimpleNamespace(n=n, **{name: ds['edge_' + name].values for name in ('i', 'j', 'dist_m', 'ux', 'uy')},
                            **{name: ds['neighbour_' + name].values for name in ('east', 'west', 'north', 'south')})
    values = {name: ds[name].values for name in ('lon', 'lat', 'area_m2', 'block', 'canal_distance_m',
              'pool_distance_m', 'capture_i', 'neighbour_i', 'stay_i')}
    values.update({name: ds[name].values if name in ds else None
                   for name in ('drain_entry_m', 'drain_path_m', 'ground_elev_m')})
    cyclic = ds.cyclic.values.astype(bool)
    law = None
    if ds.attrs['groundwater_mode'] == 'fitted_equation':
        law = SimpleNamespace(fate_names=tuple(ds.fate_feature.values), pool_weights=ds.pool_weights.values,
                              stay_weights=ds.stay_weights.values)
    return CyclicLandscape(n=n, **values, cyclic=cyclic, paddy_id=ds.paddy.values,
        owner_seed_order=np.arange(n), edges=edges,
        graph=leftover_graph(values['canal_distance_m'], edges, cyclic), neighbours=_neighbour_lists(edges, n),
        neighbour_of_remain=float(ds.attrs['neighbour_of_remain']),
        shares={name[6:]: ds[name].values for name in ds if name.startswith('share_')},
        physics=physics_as_namespace(json.loads(ds.attrs['physics_json'])), hop=HeadEqualise(),
        rain_mm=ds.rain_mm.values[:days], et0_mm=ds.et0_mm.values[:days], year=int(year),
        claim=json.loads(ds.attrs['physics_json']).get('claim', ''),
        geometry=gpd.GeoSeries.from_wkt(ds.geometry_wkt.values, crs='EPSG:4326'), ground_water_law=law)

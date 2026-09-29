"""Anisotropic lateral flow between neighbouring points.

Okura et al. obtain block lateral flow Lf as the Eq. 3 residual (IR + R - D - ET), then
Eq. 7 splits what of that residual reaches the pool (LFV_pump) from what stays in the
block (dWB). This module is the plot-scale process those residuals come from:

    1. vertical leakage through the plow pan, proportional to ponded depth
       (K_pan h / h_ref), into the plot itself
    2. sideways flow of the perched table between neighbouring points
       (Kx east-west, Ky north-south)
    3. Hooghoudt drain flow from that perched table toward the farm drains / pool

Kx and Ky are unknown anisotropic permeability. They are knobs, not fitted unless you
call `scan_anisotropy`.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .physics import kx_of, ky_of


def grid_edges(row, col, lon, lat):
    """4-connected raster neighbours. Each edge stores distance (m) and unit direction."""
    index = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(row, col))}
    i_from, i_to, dist, ux, uy = [], [], [], [], []
    seen = set()
    for i, (r, c) in enumerate(zip(row, col)):
        for dr, dc in ((0, 1), (1, 0)):
            j = index.get((int(r) + dr, int(c) + dc))
            if j is None:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            dx, dy = _metres(lon[i], lat[i], lon[j], lat[j])
            length = max(np.hypot(dx, dy), 1.0)
            i_from.append(i)
            i_to.append(j)
            dist.append(length)
            ux.append(dx / length)
            uy.append(dy / length)
    return SimpleNamespace(i=np.asarray(i_from, dtype=int), j=np.asarray(i_to, dtype=int),
                           dist_m=np.asarray(dist, dtype=float),
                           ux=np.asarray(ux, dtype=float), uy=np.asarray(uy, dtype=float))


CARDINAL = ("east", "west", "north", "south")  # right, left, up, down
CARDINAL_LABEL = {"east": "right", "west": "left", "north": "up", "south": "down"}


def cardinal_neighbours(lon, lat, k=8):
    """For every paddy, the nearest neighbour in each of east, west, north, south.

    Up/down/left/right of an irregular paddy layout: among the k nearest plots,
    assign each to the dominant axis (E–W vs N–S) and keep the closest in that
    slot. Missing edge slots are -1. The lateral equation is evaluated at every
    paddy and reads these four neighbours, not a kNN mean.
    """
    lon = np.asarray(lon, dtype=float).ravel()
    lat = np.asarray(lat, dtype=float).ravel()
    n = lon.size
    slots = {name: np.full(n, -1, dtype=int) for name in CARDINAL}
    best = {name: np.full(n, np.inf) for name in CARDINAL}
    if n < 2:
        return SimpleNamespace(**slots)
    k = min(max(int(k), 1), n - 1)
    xy = np.column_stack([lon, lat])
    try:
        from scipy.spatial import cKDTree
        _, neighbours = cKDTree(xy).query(xy, k=k + 1)
        neighbours = np.asarray(neighbours)[:, 1:]
    except Exception:
        d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(d2, np.inf)
        neighbours = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    for i in range(n):
        for j in np.atleast_1d(neighbours[i]):
            j = int(j)
            if j == i:
                continue
            dx, dy = _metres(lon[i], lat[i], lon[j], lat[j])
            length = max(np.hypot(dx, dy), 1.0)
            if abs(dx) >= abs(dy):
                name = "east" if dx > 0 else "west"
            else:
                name = "north" if dy > 0 else "south"
            if length < best[name][i]:
                best[name][i] = length
                slots[name][i] = j
    return SimpleNamespace(**slots)


def cardinal_edges(lon, lat, k=8):
    """Undirected 4-neighbour graph: up, down, left, right of every paddy."""
    nbr = cardinal_neighbours(lon, lat, k=k)
    n = len(np.asarray(lon).ravel())
    i_from, i_to, dist, ux, uy = [], [], [], [], []
    seen = set()
    lon = np.asarray(lon, dtype=float).ravel()
    lat = np.asarray(lat, dtype=float).ravel()
    for name in CARDINAL:
        for i, j in enumerate(np.asarray(getattr(nbr, name), dtype=int)):
            if j < 0 or j == i:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            dx, dy = _metres(lon[i], lat[i], lon[j], lat[j])
            length = max(np.hypot(dx, dy), 1.0)
            i_from.append(i)
            i_to.append(j)
            dist.append(length)
            ux.append(dx / length)
            uy.append(dy / length)
    return SimpleNamespace(
        i=np.asarray(i_from, dtype=int), j=np.asarray(i_to, dtype=int),
        dist_m=np.asarray(dist, dtype=float),
        ux=np.asarray(ux, dtype=float), uy=np.asarray(uy, dtype=float),
        east=np.asarray(nbr.east, dtype=int), west=np.asarray(nbr.west, dtype=int),
        north=np.asarray(nbr.north, dtype=int), south=np.asarray(nbr.south, dtype=int),
        n=n,
    )


def cardinal_features(values, nbr, missing=0.0):
    """(n, 4) neighbour values in order east, west, north, south, and a present mask.

    Missing neighbours (district edge) are `missing` and False in the mask, so they
    do not invent a dry plot at 0 mm as a real neighbour.
    """
    values = np.asarray(values, dtype=float).ravel()
    n = values.size
    cols, mask = [], []
    for name in CARDINAL:
        idx = np.asarray(getattr(nbr, name), dtype=int).ravel()
        if idx.size != n:
            idx = np.full(n, -1, dtype=int)
        present = idx >= 0
        col = np.full(n, missing, dtype=float)
        col[present] = values[idx[present]]
        cols.append(col)
        mask.append(present)
    return np.column_stack(cols), np.column_stack(mask)


def cardinal_drops(own, neighbour_values, present):
    """Own minus each cardinal neighbour; 0 where that neighbour does not exist."""
    own = np.asarray(own, dtype=float).ravel()[:, None]
    drop = own - np.asarray(neighbour_values, dtype=float)
    return np.where(present, drop, 0.0)


def knn_edges(lon, lat, k=4):
    """Fallback when points are not on a raster: k nearest neighbours."""
    n = len(lon)
    if n < 2:
        return SimpleNamespace(i=np.zeros(0, dtype=int), j=np.zeros(0, dtype=int),
                               dist_m=np.zeros(0), ux=np.zeros(0), uy=np.zeros(0))
    k = min(k, n - 1)
    xy = np.column_stack([lon, lat])
    try:
        from scipy.spatial import cKDTree
        _, neighbours = cKDTree(xy).query(xy, k=k + 1)
        neighbours = np.asarray(neighbours)[:, 1:]
    except Exception:
        d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(d2, np.inf)
        neighbours = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    i_from, i_to, dist, ux, uy = [], [], [], [], []
    seen = set()
    for i in range(n):
        for j in np.atleast_1d(neighbours[i]):
            j = int(j)
            if j == i:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            dx, dy = _metres(lon[i], lat[i], lon[j], lat[j])
            length = max(np.hypot(dx, dy), 1.0)
            i_from.append(i)
            i_to.append(j)
            dist.append(length)
            ux.append(dx / length)
            uy.append(dy / length)
    return SimpleNamespace(i=np.asarray(i_from, dtype=int), j=np.asarray(i_to, dtype=int),
                           dist_m=np.asarray(dist, dtype=float),
                           ux=np.asarray(ux, dtype=float), uy=np.asarray(uy, dtype=float))


def _metres(lon_a, lat_a, lon_b, lat_b):
    mid = np.radians(0.5 * (lat_a + lat_b))
    dx = (lon_b - lon_a) * 111_320.0 * np.cos(mid)
    dy = (lat_b - lat_a) * 110_540.0
    return dx, dy


def exchange_perched(perched_mm, edges, physics, k_x=None, k_y=None, area_m2=None):
    """Net perched-water change (mm) from Darcy flow along `edges`. Conserves volume
    when `area_m2` is given; otherwise conserves millimetres (equal-area points)."""
    n = perched_mm.size
    if edges is None or len(edges.i) == 0:
        return np.zeros(n)
    kx = kx_of(physics) if k_x is None else k_x
    ky = ky_of(physics) if k_y is None else k_y
    kx = np.broadcast_to(np.asarray(kx, dtype=float), (n,))
    ky = np.broadcast_to(np.asarray(ky, dtype=float), (n,))
    area = np.broadcast_to(np.asarray(1.0 if area_m2 is None else area_m2, dtype=float), (n,))
    i, j = edges.i, edges.j
    k_edge = 0.5 * ((kx[i] + kx[j]) * edges.ux ** 2 + (ky[i] + ky[j]) * edges.uy ** 2)
    head = np.maximum(perched_mm, 0.0) / 1000.0 / physics.drainable_porosity
    # Q m3/d = (K m/d) * thickness * width * dh / L; width ~ L so K * d * dh.
    k_m_d = k_edge / 1000.0
    q_m3 = k_m_d * physics.equivalent_depth_m * (head[i] - head[j])
    net = np.zeros(n)
    np.add.at(net, i, -1000.0 * q_m3 / np.maximum(area[i], 1e-9))
    np.add.at(net, j, 1000.0 * q_m3 / np.maximum(area[j], 1e-9))
    # Do not pull a dry node negative; scale the whole step if needed.
    trial = perched_mm + net
    if np.any(trial < -1e-9):
        room = np.where(net < 0, perched_mm / np.maximum(-net, 1e-12), 1.0)
        net = net * np.min(np.clip(room, 0.0, 1.0))
    return net


def neighbour_mean(values, edges):
    """Mean of `values` on neighbour plots (undirected knn edges)."""
    values = np.asarray(values, dtype=float).ravel()
    n = values.size
    mean = np.zeros(n)
    if edges is None or len(getattr(edges, "i", [])) == 0:
        return mean
    i = np.asarray(edges.i, dtype=int)
    j = np.asarray(edges.j, dtype=int)
    total = np.zeros(n)
    count = np.zeros(n)
    np.add.at(total, i, values[j])
    np.add.at(count, i, 1.0)
    np.add.at(total, j, values[i])
    np.add.at(count, j, 1.0)
    np.divide(total, count, out=mean, where=count > 0)
    return mean


def neighbour_head_features(depth_mm, edges):
    """Mean neighbour ponding and own-minus-neighbour head drop (mm).

    A soil-flow equation may see how much water each neighbour holds. These
    arrays are not a share of applied irrigation.
    """
    depth = np.asarray(depth_mm, dtype=float).ravel()
    mean = neighbour_mean(depth, edges)
    return mean, depth - mean


def neighbour_et_features(et_mm, edges):
    """Mean neighbour FAO-56 ET and own-minus-neighbour ET drop (mm).

    A soil-flow equation may see how much the neighbour evaporated. That is
    not a licence to send leftover toward high ET, and it is not the satellite
    Kc×ET0 map.
    """
    et = np.asarray(et_mm, dtype=float).ravel()
    mean = neighbour_mean(et, edges)
    return mean, et - mean

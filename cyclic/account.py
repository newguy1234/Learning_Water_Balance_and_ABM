"""Forced satellite-ET water accounting, without Darcy.

Given irrigation and discharge at a point, the satellite ET is taken as observed
(it must occur). Dual Kc splits that total: transpiration from the root zone, evaporation
from the pond if flooded otherwise from the surface. Rain and irrigation only replenish
those stores. If root zone and surface cannot meet T and E, the shortfall is groundwater
used for ET at that point (`need`), so the observed total is never cut back.

Anything left after rain, irrigation, discharge and that ET is leftover underground
stock. Darcy / Hooghoudt are not used. Where that leftover goes — lateral flow (to the
drains / pool) vs remaining in the ground — is an inverse problem of the same kind as
O'Malley (2018), Sci. Rep. 8:6919: a binary field on the points, with a data misfit
and neighbour coupling. The QUBO is solved classically here; the matrix is kept so a
quantum annealer can be swapped in later.

    IR + R + ET_from_groundwater = ET_satellite + D + d(pond) + d(surface) + d(root) + leftover
    leftover = lateral + stay
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from . import okura
from .physics import Physics, State, apply_et, dual_coefficients, split_satellite_et


def force_et(state, rain_mm, et_target_mm, irrigation_mm, discharge_mm, p,
             et0_mm=None, canopy=0.5):
    """One day of bookkeeping. Satellite ET is applied in full. No Darcy.

    T is taken from the root zone; E from the pond if flooded, else the surface.
    `canopy` is fractional cover (Sentinel-2). `et0_mm` only sets the T:E split;
    the total is always the satellite target.
    """
    water = state.water
    n = water.size
    et = np.maximum(np.nan_to_num(np.broadcast_to(np.asarray(et_target_mm, dtype=float), (n,))), 0.0)
    rain = np.broadcast_to(np.asarray(rain_mm, dtype=float), (n,))
    irrigation = np.maximum(np.broadcast_to(np.asarray(irrigation_mm, dtype=float), (n,)), 0.0)
    deficit_0 = state.deficit.copy()

    water += rain + irrigation
    wetted = (rain + irrigation) > 0.0
    fc, kcb, few, ponded, ks, kr, ke, _kc_cap = dual_coefficients(
        water, state.surface, deficit_0, canopy, p, wetted=wetted)
    if et0_mm is None:
        transpiration, evaporation = split_satellite_et(et, ks, kcb, ke, ponded)
    else:
        et0 = np.broadcast_to(np.asarray(et0_mm, dtype=float), (n,))
        transpiration, evaporation = split_satellite_et(et, ks * et0, kcb, ke * et0, ponded)
    taken = apply_et(state, transpiration, evaporation, p, force=True)
    water = state.water

    discharge = np.minimum(np.maximum(np.broadcast_to(np.asarray(discharge_mm, dtype=float), (n,)), 0.0), water)
    water -= discharge
    overflow = np.maximum(water - p.bund_mm, 0.0)
    water -= overflow
    spill = np.zeros(n)
    drainage = discharge + overflow + spill

    leftover = rain + irrigation - drainage - et
    zeros = np.zeros(n)
    return {
        "et_potential_mm": et,
        "et_mm": et,
        "et_pond_mm": taken["et_pond_mm"],
        "et_surface_mm": taken["et_surface_mm"],
        "et_root_mm": taken["et_root_mm"],
        "et_groundwater_mm": taken["need_mm"],
        "need_mm": taken["need_mm"],
        "canal_mm": drainage,
        "transpiration_mm": taken["transpiration_mm"],
        "evaporation_mm": taken["evaporation_mm"],
        "ks": ks,
        "kr": kr,
        "ke": ke,
        "kcb": kcb,
        "ponded": taken["ponded"],
        "discharge_mm": discharge,
        "overflow_mm": overflow,
        "spill_mm": spill,
        "drainage_mm": drainage,
        "infiltration_mm": np.maximum(leftover, 0.0),
        "leftover_mm": leftover,
        "neighbor_mm": zeros,
        "lateral_mm": zeros,
        "deep_mm": zeros,
        "stay_mm": zeros,
        "assigned_lateral": zeros,
    }


def leftover_volumes(stock_mm, area_m2):
    return np.asarray(stock_mm, dtype=float) * np.asarray(area_m2, dtype=float) / 1000.0


def _neighbour_lists(n, edges):
    adj = [[] for _ in range(n)]
    if edges is None:
        return adj
    for i, j in zip(np.asarray(edges.i, dtype=int), np.asarray(edges.j, dtype=int)):
        adj[i].append(j)
        adj[j].append(i)
    return adj


def solve_omalley(stock_mm, area_m2, cyclic, block, edges, capture, year,
                  smoothness=0.05, sweeps=4, seed=0):
    """Classical local search on an O'Malley-style binary inverse (Sci. Rep. 2018).

    x_i = 1 leftover at i is lateral flow; x_i = 0 it stays in the ground.
    Misfit is season cyclic Lf (Table 4 / Eq. 3) and LFV_pump (Table 5 / Eq. 7), plus
    neighbour agreement. Dense QUBO is not materialised; a flip's ΔE is O(degree).
    The same Hamiltonian is what a quantum annealer would minimise.
    """
    stock_mm = np.asarray(stock_mm, dtype=float)
    vol = leftover_volumes(stock_mm, area_m2)
    cyclic = np.asarray(cyclic, dtype=bool)
    block = np.asarray(block)
    n = vol.size
    weights = {name: np.where(cyclic & (block == name), vol, 0.0) for name in okura.BLOCKS}
    targets = {name: okura.TABLE4_MM[(name, year)]["lateral"] * okura.cyclic_area_m2(name) / 1000.0
               for name in okura.BLOCKS}
    lfv_weight = np.where(cyclic, float(capture) * vol, 0.0)
    lfv_target = okura.TABLE5_1000M3[year]["LFV_pump"] * 1000.0
    adj = _neighbour_lists(n, edges)
    mean_vol2 = float(np.mean(vol[vol > 0]) ** 2) if np.any(vol > 0) else 1.0
    j_couple = float(smoothness) * mean_vol2

    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=np.int8)
    free = np.flatnonzero(cyclic)
    for name in okura.BLOCKS:
        mask = np.flatnonzero(weights[name] > 0)
        order = mask[np.argsort(-weights[name][mask])]
        running = 0.0
        for i in order:
            if running >= targets[name]:
                break
            x[i] = 1
            running += weights[name][i]

    sums = {name: float(weights[name] @ x) for name in okura.BLOCKS}
    lfv = float(lfv_weight @ x)

    def misfit():
        value = sum((sums[name] - targets[name]) ** 2 for name in okura.BLOCKS)
        value += (lfv - lfv_target) ** 2
        if j_couple:
            for i, nbrs in enumerate(adj):
                for j in nbrs:
                    if i < j:
                        value += j_couple * (x[i] - x[j]) ** 2
        return value

    def delta(i):
        # x_i -> 1-x_i
        sign = 1.0 if x[i] == 0 else -1.0
        change = 0.0
        for name in okura.BLOCKS:
            w = weights[name][i]
            if w:
                s = sums[name]
                t = targets[name]
                change += (s + sign * w - t) ** 2 - (s - t) ** 2
        if lfv_weight[i]:
            change += (lfv + sign * lfv_weight[i] - lfv_target) ** 2 - (lfv - lfv_target) ** 2
        if j_couple and adj[i]:
            xi = int(x[i])
            for j in adj[i]:
                change += j_couple * (((1 - xi) - x[j]) ** 2 - (xi - x[j]) ** 2)
        return change, sign

    best = x.copy()
    best_e = misfit()
    for _ in range(max(sweeps, 1)):
        for i in rng.permutation(free if free.size else n):
            change, sign = delta(i)
            if change <= 0:
                x[i] = 1 - x[i]
                for name in okura.BLOCKS:
                    sums[name] += sign * weights[name][i]
                lfv += sign * lfv_weight[i]
                best_e += change
                best = x.copy()

    return SimpleNamespace(
        x=x, energy=best_e,
        lateral_m3=dict(sums), lfv_m3=lfv,
        targets=targets, lfv_target=lfv_target,
        lf_error={name: (sums[name] - targets[name]) / max(targets[name], 1.0) for name in okura.BLOCKS},
        lfv_error=(lfv - lfv_target) / max(lfv_target, 1.0),
        j_couple=j_couple,
    )


def apply_assignment(log, x):
    """Split recorded leftover into lateral vs stay using the binary field x."""
    leftover = np.asarray(log["leftover_mm" if "leftover_mm" in log else "infiltration_mm"])
    x = np.asarray(x, dtype=float)
    positive = np.maximum(leftover, 0.0)
    log["lateral_mm"][:] = positive * x
    stay = positive * (1.0 - x)
    log["deep_mm"][:] = stay
    if "stay_mm" in log:
        log["stay_mm"][:] = stay
    if "assigned_lateral" in log:
        log["assigned_lateral"][:] = x
    return log

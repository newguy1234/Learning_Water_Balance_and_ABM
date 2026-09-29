"""Discharge toward the main drainage canal, which returns to the pool.

Okura et al. measure surface drainage D with an ADCP in the main canal. That D is
each cyclic paddy's discharge back to the pumping-station pool the same day (Eq. 4):
irrigation is then pumped out of that pool, so the pool stock goes down. D is not
leftover seepage. What remains after IR + R − D − ET is leftover (Eq. 3), the
paper's Lf (plus what stays as dWB).

Okura §2.1: farm drains collect at the plot; the 低地排水路 is the district
collector; cyclic D and LFV_pump must not share a lining-catch sink. Same-day LFV
is the dedicated ``pool_capture_share`` of leftover (Table 5 LFV_pump/LfA), taken
from infiltrated water. Kx+Ky send leftover to neighbours to cover ET shortfalls.
Pond leftover cannot be LFV. Hops equalise the neighbour share; they do not catch
at the lining as LFV. Non-cyclic leftover never feeds LFV_pump.
"""

from __future__ import annotations

import numpy as np

from . import infrastructure
from .blocks import DIVIDER as _PLACEHOLDER_DIVIDER
from .graph import catch_at_canal, hop_stock, leftover_graph, move_stock, optional_qubo


# The main drainage canal (Okura Fig. 1b): the channel along the lake dyke through the
# pumping station, from cyclic/bundled/infrastructure.geojson. The straight placeholder in
# cyclic/blocks.py is used only when that file is missing.
DIVIDER = (infrastructure.main_canal_vertices() if infrastructure.available()
           else list(_PLACEHOLDER_DIVIDER))
# What canal_distance_m measures to: every channel that reaches the pool (main canal and the
# drains joined to it). Lateral flow is recaptured where it enters any of these drains, not
# only at the main canal (Okura §4.2), so this sets each paddy's recapture share.
POOL_NETWORK = (infrastructure.pool_network_vertices() if infrastructure.available()
                else [DIVIDER])


def _metres(lon_a, lat_a, lon_b, lat_b):
    mid = np.radians(0.5 * (lat_a + lat_b))
    dx = (lon_b - lon_a) * 111_320.0 * np.cos(mid)
    dy = (lat_b - lat_a) * 110_540.0
    return dx, dy


def _point_to_segment_m(px, py, ax, ay, bx, by):
    abx, aby = _metres(ax, ay, bx, by)
    apx, apy = _metres(ax, ay, px, py)
    length2 = abx * abx + aby * aby
    t = 0.0 if length2 <= 0 else np.clip((apx * abx + apy * aby) / length2, 0.0, 1.0)
    cx = apx - t * abx
    cy = apy - t * aby
    return float(np.hypot(cx, cy))


def distance_to_canal_m(lon, lat, vertices=None):
    """Plan-view distance (m) from each point to the nearest channel that drains to the pool
    (``POOL_NETWORK``), or to the one polyline ``vertices`` if given (e.g. ``DIVIDER``, the
    main canal alone)."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lines = POOL_NETWORK if vertices is None else [vertices]
    x, y = np.atleast_1d(lon).ravel(), np.atleast_1d(lat).ravel()
    out = np.full(x.shape, np.inf)
    for line in lines:
        line = list(line)
        for (ax, ay), (bx, by) in zip(line, line[1:]):
            # Same local projection as _point_to_segment_m, for every point at once.
            abx, aby = _metres(ax, ay, bx, by)
            apx, apy = _metres(ax, ay, x, y)
            length2 = abx * abx + aby * aby
            t = np.zeros_like(x) if length2 <= 0 else np.clip((apx * abx + apy * aby) / length2, 0.0, 1.0)
            out = np.minimum(out, np.hypot(apx - t * abx, apy - t * aby))
    return out.reshape(lon.shape) if lon.shape else float(out[0])


def farm_drain_factor(p, stage_mm=None):
    """1 at the reference farm-drain stage; rises as the drains run higher."""
    stage = p.farm_drain_stage_mm if stage_mm is None else stage_mm
    return max(float(stage), 0.0) / max(float(p.farm_drain_ref_mm), 1.0)


def darcy_shares(p):
    """(kx, ky, kz, ground, lfv) leftover shares, clipped and renormalised to 1."""
    raw = np.array([
        max(float(getattr(p, "kx_ratio", 0.0)), 0.0),
        max(float(getattr(p, "ky_ratio", 0.0)), 0.0),
        max(float(getattr(p, "kz_ratio", 0.0)), 0.0),
        max(float(getattr(p, "ground_ratio", 0.0)), 0.0),
        max(float(getattr(p, "pool_capture_share", 0.0)), 0.0),
    ], dtype=float)
    total = float(raw.sum())
    if total <= 0:
        raw[:] = 0.2
    else:
        raw /= total
    return {"kx": raw[0], "ky": raw[1], "kz": raw[2], "ground": raw[3], "lfv": raw[4]}


def _flow_signs(n, dist, edges):
    """Per-point downhill-to-canal direction: +x east, +y north, −x west, −y south."""
    sx, sy = np.zeros(n), np.zeros(n)
    if edges is None or len(getattr(edges, "i", [])) == 0:
        return np.ones(n), np.ones(n)
    dist = np.asarray(dist, dtype=float)
    i = np.asarray(edges.i, dtype=int)
    j = np.asarray(edges.j, dtype=int)
    ux = np.asarray(edges.ux, dtype=float)
    uy = np.asarray(edges.uy, dtype=float)
    high = np.where(dist[i] > dist[j], i, j)
    toward_j = dist[i] > dist[j]
    # Flow high → low. Edge ux,uy is i → j.
    flow_x = np.where(toward_j, ux, -ux)
    flow_y = np.where(toward_j, uy, -uy)
    drop = np.abs(dist[i] - dist[j])
    np.add.at(sx, high, flow_x * drop)
    np.add.at(sy, high, flow_y * drop)
    sign_x = np.sign(sx)
    sign_y = np.sign(sy)
    sign_x = np.where(sign_x == 0.0, 1.0, sign_x)
    sign_y = np.where(sign_y == 0.0, 1.0, sign_y)
    return sign_x, sign_y


def _axis_fractions(n, dist, edges):
    """Per-point share of downhill drop that is east-west vs north-south."""
    wx, wy = np.zeros(n), np.zeros(n)
    if edges is None or len(getattr(edges, "i", [])) == 0:
        return np.full(n, 0.5), np.full(n, 0.5)
    dist = np.asarray(dist, dtype=float)
    for axis, bucket in (("x", wx), ("y", wy)):
        keep = _axis_edges(edges, axis)
        if not np.any(keep):
            continue
        i = np.asarray(edges.i, dtype=int)[keep]
        j = np.asarray(edges.j, dtype=int)[keep]
        drop = np.abs(dist[i] - dist[j])
        np.add.at(bucket, i, drop)
        np.add.at(bucket, j, drop)
    total = wx + wy
    fx = np.divide(wx, total, out=np.full(n, 0.5), where=total > 0.0)
    return fx, 1.0 - fx


def _scale_field(raw, cyclic, target, weights=None):
    """Proximity field on cyclic cells whose weighted mean matches `target` in [0, 1].

    Rescaling and then clipping at 1 leaves the mean below the target whenever cells
    near the canal saturate, so the multiplier is solved for instead. `weights` is the
    measure the mean is taken in (area); unweighted if None.
    """
    out = np.zeros(raw.shape, dtype=float)
    mask = np.asarray(cyclic, dtype=bool)
    if not np.any(mask):
        return out
    w = np.maximum(raw[mask], 0.0)
    target = float(np.clip(target, 0.0, 1.0))
    if weights is None:
        weight = np.ones(w.size)
    else:
        weight = np.maximum(np.asarray(weights, dtype=float).ravel()[:mask.size][mask], 0.0)
    if float(weight.sum()) <= 0.0:
        weight = np.ones(w.size)
    if target <= 0.0 or float(w.max()) <= 0.0:
        out[mask] = target
        return out

    def mean_at(scale):
        return float(np.average(np.clip(w * scale, 0.0, 1.0), weights=weight))

    hi = 1.0 / float(w.max())
    for _ in range(80):
        if mean_at(hi) >= target:
            break
        hi *= 2.0
    else:
        out[mask] = target
        return out
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mean_at(mid) < target:
            lo = mid
        else:
            hi = mid
    out[mask] = np.clip(w * hi, 0.0, 1.0)
    return out


def pixel_ratio_fields(n, dist_m, edges, cyclic, capture, farm_to_main, topsoil_frac,
                       canal_length_m, area_m2=None, block=None, block_intensity=None,
                       hard_capture=False, neighbour_of_remain=0.5):
    """Per-point leftover shares with separate LFV and neighbour roles.

        pool_capture_share  → LFV (district mean = ``capture`` = Table 5 LFV/LfA);
                              near-canal cells get a higher LFV share.
        kx_ratio, ky_ratio  → leftover sent to neighbours (ET cover); half each of
                              ``neighbour_of_remain`` × (1 − LFV).
        kz_ratio, ground    → stay / deep on the rest.
    """
    n = int(n)
    cyclic = np.broadcast_to(np.asarray(cyclic, dtype=bool), (n,)).copy()
    cap = float(np.clip(capture, 0.0, 1.0))
    farm = float(np.clip(farm_to_main, 0.0, 1.0))
    nbr_frac = float(np.clip(neighbour_of_remain, 0.0, 1.0))
    dist = np.asarray(dist_m, dtype=float).ravel()
    if dist.size != n:
        dist = np.resize(dist, n) if dist.size else np.zeros(n)
    length = max(float(canal_length_m), 1.0)
    prox = np.exp(-np.maximum(dist, 0.0) / length)
    prox = np.where(cyclic, np.maximum(prox, 1e-6), 0.0)
    if block is not None and block_intensity:
        names = np.asarray(block).ravel()
        if names.size != n:
            names = np.resize(names, n)
        inten = np.ones(n, dtype=float)
        for name, val in block_intensity.items():
            inten[names == name] = float(val)
        prox = prox * np.where(cyclic, np.maximum(inten, 1e-6), 0.0)
    if area_m2 is not None:
        area = np.asarray(area_m2, dtype=float).ravel()
        if area.size != n:
            area = np.resize(area, n) if area.size else np.ones(n)
        w = np.where(cyclic, np.maximum(area, 0.0), 0.0)
    else:
        w = cyclic.astype(float)
    w_sum = float(w.sum()) or 1.0
    mean_prox = float((prox * w).sum() / w_sum)
    if mean_prox > 1e-12 and cap > 0.0:
        raw = prox / mean_prox
        mx = float(np.max(raw[cyclic])) if np.any(cyclic) else 1.0
        if cap * mx > 1.0 - 1e-12 and mx > 1.0 + 1e-12:
            t = (1.0 / cap - 1.0) / (mx - 1.0)
            t = float(np.clip(t, 0.0, 1.0))
            raw = (1.0 - t) + t * raw
        capture_i = np.where(cyclic, np.clip(cap * raw, 0.0, 1.0), 0.0)
    else:
        capture_i = np.where(cyclic, cap, 0.0).astype(float)
    farm_i = np.where(cyclic, farm, 0.0).astype(float)
    remain = np.maximum(1.0 - capture_i, 0.0)
    neighbour = remain * nbr_frac
    stay = remain - neighbour
    top = float(np.clip(topsoil_frac, 0.0, 1.0))
    kx = neighbour * 0.5
    ky = neighbour * 0.5
    sign_x, sign_y = np.ones(n), np.ones(n)
    return {
        "kx_ratio": kx,
        "ky_ratio": ky,
        "kz_ratio": stay * top,
        "ground_ratio": stay * (1.0 - top),
        "farm_to_main": farm_i,
        "pool_capture_share": capture_i,
        "kx_signed": kx * sign_x,
        "ky_signed": ky * sign_y,
        "kx_sign": sign_x,
        "ky_sign": sign_y,
    }


def mean_pixel_shares(fields, area_m2, cyclic=None):
    """Area-weighted mean of a pixel-ratio dict (cyclic cells only)."""
    area = np.asarray(area_m2, dtype=float)
    mask = np.ones(area.size, dtype=bool) if cyclic is None else np.asarray(cyclic, dtype=bool)
    w = np.where(mask.ravel()[:area.size], np.maximum(area, 0.0), 0.0)
    total = float(w.sum()) or 1.0
    out = {}
    for key, values in fields.items():
        arr = _share_vector(values, w.size)
        out[key] = float((arr * w).sum() / total)
    return out


def _share_vector(values, n):
    """Reduce a share field to length `n` (mean over time if needed)."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.ndim == 2:
        if arr.shape[-1] == n:
            arr = arr.mean(axis=0)
        elif arr.shape[0] == n:
            arr = arr.mean(axis=1)
        else:
            arr = arr.reshape(-1)
    arr = np.ravel(arr)
    if arr.size == n:
        return arr
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    return np.resize(arr, n)


def weighted_share_mean(values, area_m2, cyclic=None, extra_weights=None):
    area = np.asarray(area_m2, dtype=float)
    mask = np.ones(area.size, dtype=bool) if cyclic is None else np.asarray(cyclic, dtype=bool).ravel()
    w = np.maximum(area, 0.0) * mask[:area.size]
    if extra_weights is not None:
        w = w * np.asarray(extra_weights, dtype=float).ravel()[:area.size]
    w = np.maximum(w, 0.0)
    arr = _share_vector(values, area.size)
    total = float(w.sum()) or 1.0
    return float((arr * w).sum() / total)


def _axis_edges(edges, axis):
    ux, uy = np.asarray(edges.ux, dtype=float), np.asarray(edges.uy, dtype=float)
    if axis == "x":
        return np.abs(ux) >= np.abs(uy)
    return np.abs(uy) > np.abs(ux)


def _move_stock(stock, dist, edges, area, connected, p, axis):
    graph = leftover_graph(dist, edges, connected, one_way=bool(getattr(p, "one_way_to_canal", False)))
    return move_stock(stock, graph, area, p, axis)


def _hop_stock(stock, dist, edges, area, connected, p):
    graph = leftover_graph(dist, edges, connected, one_way=bool(getattr(p, "one_way_to_canal", False)))
    return hop_stock(stock, graph, area, p)


def _catch_at_main_drain(stock, dist, connected, p):
    return catch_at_canal(stock, dist, connected, p)


def divert_recapture_to_unsaturated(flux, surface, deficit, edges, area_m2, cyclic, p):
    """Some same-day LFV fills nearby unsaturated soil instead of the pool.

    That water is there so the neighbour can evaporate and transpire later (Ke, Ks)
    against the satellite ET score. It is not today's et_mm and not a leftover
    ratio. Limited by neighbour TEW depletion plus TAW deficit. Only wetter soil
    (lower demand) gives to drier soil; remainder stays LFV. Own-plot stores are
    not refilled from LFV. (Kx+Ky neighbour hops are a separate path in apply_canal.)
    """
    lfv = np.asarray(flux.get("lfv_mm", 0.0), dtype=float)
    n = int(np.size(lfv))
    diverted = np.zeros(n)
    filled = np.zeros(n)
    if n == 0 or edges is None or len(getattr(edges, "i", [])) == 0:
        flux["lfv_to_soil_mm"] = diverted
        flux["neighbor_fill_mm"] = filled
        return flux
    lfv = np.broadcast_to(lfv, (n,)).copy()
    surface = np.asarray(surface, dtype=float)
    deficit = np.asarray(deficit, dtype=float)
    if surface.size != n or deficit.size != n:
        flux["lfv_to_soil_mm"] = diverted
        flux["neighbor_fill_mm"] = filled
        return flux
    area = np.broadcast_to(np.asarray(area_m2, dtype=float), (n,))
    connected = np.ones(n, dtype=bool) if cyclic is None else np.broadcast_to(
        np.asarray(cyclic, dtype=bool), (n,)).copy()
    graph = leftover_graph(np.zeros(n), edges, connected, one_way=False)
    pairs = np.atleast_2d(getattr(graph, "undirected", np.zeros((0, 2), dtype=int)))
    demand = np.maximum(surface, 0.0) + np.maximum(deficit, 0.0)
    for pair in pairs:
        if pair.size < 2:
            continue
        a, b = int(pair[0]), int(pair[1])
        if a == b or not connected[a] or not connected[b]:
            continue
        if demand[a] == demand[b]:
            continue
        recv, donor = (a, b) if demand[a] > demand[b] else (b, a)
        if lfv[donor] <= 0.0 or demand[recv] <= 0.0:
            continue
        gap = demand[recv] - demand[donor]
        if gap <= 0.0:
            continue
        cap = gap / (1.0 + area[donor] / max(float(area[recv]), 1e-9))
        moved = min(float(lfv[donor]), float(demand[recv]), float(cap))
        if moved <= 0.0:
            continue
        to_s = min(moved, max(float(surface[recv]), 0.0))
        surface[recv] -= to_s
        rest = moved - to_s
        to_d = min(rest, max(float(deficit[recv]), 0.0))
        deficit[recv] -= to_d
        took = to_s + to_d
        if took <= 0.0:
            continue
        lfv[donor] -= took
        diverted[donor] += took
        filled[recv] += took
        demand[recv] = max(float(surface[recv]), 0.0) + max(float(deficit[recv]), 0.0)
        demand[donor] = max(float(surface[donor]), 0.0) + max(float(deficit[donor]), 0.0)
    flux["lfv_mm"] = lfv
    flux["lfv_to_soil_mm"] = diverted
    flux["neighbor_fill_mm"] = filled
    return flux


def apply_canal(flux, dist_m, edges, area_m2, p, drain_stock=None, leftover_split=True,
                cyclic=None, shares=None):
    """Toward the main drain is per-paddy discharge D.

    Cyclic D returns to the pool today. Non-cyclic D is not connected to that canal
    (Okura Section 2.1): leftover stays in the NCB catchment and does not feed LFV_pump.

    leftover_split: partition leftover (IR+R−D−ET) into LFV / neighbour / stay. Off on the
    FAO/Hooghoudt path, which already has its own lateral.

    Identified ratios on Physics (see cyclic.identify), or per-pixel `shares`:
        farm_to_main          farm-drain D that arrives in the main canal
        pool_capture_share    leftover → same-day LFV (Table 5 pin)
        kx, ky                leftover → neighbours (ET cover)
        kz, ground            leftover → stay / deep (dWB)
    """
    n = np.asarray(flux.get("leftover_mm", flux.get("drainage_mm", [0.0])), dtype=float).size
    discharge = np.maximum(np.broadcast_to(
        np.asarray(flux.get("drainage_mm", 0.0), dtype=float), (n,)), 0.0)
    dist = np.broadcast_to(np.asarray(dist_m, dtype=float), (n,)).copy()
    area = np.broadcast_to(np.asarray(area_m2, dtype=float), (n,))
    factor = farm_drain_factor(p)
    near = np.exp(-dist / max(float(p.canal_length_m), 1.0))
    connected = np.ones(n, dtype=bool) if cyclic is None else np.broadcast_to(
        np.asarray(cyclic, dtype=bool), (n,)).copy()

    def field(name, default):
        if shares is not None and name in shares:
            values = np.asarray(shares[name], dtype=float)
            if values.size == 1:
                return np.full(n, float(np.clip(values, 0.0, 1.0)))
            if values.size == n:
                return np.clip(values, 0.0, 1.0)
        return np.full(n, float(np.clip(default, 0.0, 1.0)))

    rho_sm = field("farm_to_main", float(np.clip(getattr(p, "farm_to_main", 1.0), 0.0, 1.0)))
    rho_ld = max(float(getattr(p, "lateral_per_discharge", 0.0)), 0.0)
    use_transit = leftover_split and (getattr(p, "ratio_split", False) or shares is not None)
    transit = np.zeros(n)
    if drain_stock is not None and np.size(drain_stock) == n:
        if use_transit:
            transit = np.maximum(np.asarray(drain_stock, dtype=float).ravel()[:n], 0.0)
        drain_stock[:] = 0.0

    flux["canal_mm"] = discharge
    flux["canal_out_mm"] = discharge * rho_sm * connected
    flux["drain_stock_mm"] = np.zeros(n)
    flux["farm_drain_factor"] = np.full(n, factor)
    flux["ld_ratio"] = np.full(n, rho_ld)
    flux["farm_to_main"] = rho_sm
    induced = rho_ld * discharge * connected

    if not leftover_split:
        current = np.asarray(flux.get("lateral_mm", 0.0), dtype=float)
        flux["lateral_mm"] = (current + induced) if np.size(current) == n else induced
        return flux

    leftover = np.maximum(np.asarray(flux.get("leftover_mm", 0.0), dtype=float), 0.0)
    leftover = np.broadcast_to(leftover, (n,)).copy()

    if getattr(p, "ratio_split", False) or shares is not None:
        if shares is None:
            scalar = darcy_shares(p)
            kx = np.full(n, scalar["kx"])
            ky = np.full(n, scalar["ky"])
            kz = np.full(n, scalar["kz"])
            ground = np.full(n, scalar["ground"])
            lfv_r = np.full(n, scalar["lfv"])
        else:
            kx = np.abs(field("kx_ratio", 0.0))
            ky = np.abs(field("ky_ratio", 0.0))
            kz = np.abs(field("kz_ratio", 0.0))
            ground = np.abs(field("ground_ratio", 0.0))
            lfv_r = np.abs(field("pool_capture_share", float(getattr(p, "pool_capture_share", 0.0))))
            total = np.maximum(kx + ky + kz + ground + lfv_r, 1e-12)
            kx, ky, kz, ground, lfv_r = (kx / total, ky / total, kz / total,
                                         ground / total, lfv_r / total)
        cyclic_mask = connected.astype(float)
        ncb_mask = 1.0 - cyclic_mask
        neighbour = kx + ky
        infiltration = np.asarray(flux.get("infiltration_mm", leftover), dtype=float)
        if np.size(infiltration) != n:
            infiltration = leftover.copy()
        else:
            infiltration = np.maximum(np.broadcast_to(infiltration, (n,)).copy(), 0.0)
        infiltrated = np.minimum(leftover, infiltration)
        flow_kx = leftover * kx * cyclic_mask
        flow_ky = leftover * ky * cyclic_mask
        flow_kz = leftover * kz
        flow_ground = leftover * ground + leftover * (neighbour + lfv_r) * ncb_mask
        # LFV: dedicated pool_capture share (not Kx+Ky).
        intended = leftover * lfv_r * cyclic_mask
        lfv = np.minimum(intended, infiltrated * cyclic_mask)
        shortfall = intended - lfv
        # Kx+Ky mass hops to neighbours to cover ET shortfalls.
        nbr_mass = leftover * neighbour * cyclic_mask + transit * cyclic_mask
        graph = leftover_graph(
            dist, edges, connected, one_way=bool(getattr(p, "one_way_to_canal", False)))
        _hopped, neighbor = hop_stock(nbr_mass, graph, area, p)
        previous = np.asarray(flux.get("neighbor_mm", 0.0), dtype=float)
        flux["neighbor_mm"] = previous + neighbor if np.size(previous) == n else neighbor
        flux["flow_kx_mm"] = flow_kx
        flux["flow_ky_mm"] = flow_ky
        flux["flow_kz_mm"] = flow_kz
        flux["flow_ground_mm"] = flow_ground
        flux["lateral_mm"] = intended  # LFV path volume (table pin)
        flux["stay_mm"] = leftover * (kz + ground)
        flux["deep_mm"] = flow_ground
        flux["lateral_from_d_mm"] = np.zeros(n)
        flux["lfv_mm"] = lfv
        flux["recapture_intended_mm"] = intended
        flux["lfv_shortfall_mm"] = shortfall
        flux["neighbour_leftover_mm"] = nbr_mass
        flux["drain_stock_mm"] = np.zeros(n)
        flux["graph_n_nodes"] = graph.n
        flux["graph_n_edges"] = int(graph.i.size)
        flux["graph_one_way"] = bool(graph.one_way)
        flux["graph_qubo"] = optional_qubo(nbr_mass, area, graph, p)
        if drain_stock is not None and np.size(drain_stock) == n:
            drain_stock[:] = 0.0
        return flux

    leftover, neighbor = _hop_stock(leftover, dist, edges, area, connected, p)
    from_d = np.minimum(leftover, induced)
    leftover = leftover - from_d
    lat_frac = np.clip(float(p.canal_capture_frac) * factor * near, 0.0, 1.0) * connected
    lateral = leftover * lat_frac + from_d
    stay = leftover - leftover * lat_frac
    previous = np.asarray(flux.get("neighbor_mm", 0.0), dtype=float)
    flux["neighbor_mm"] = previous + neighbor if np.size(previous) == n else neighbor
    flux["lateral_mm"] = lateral
    flux["stay_mm"] = stay
    flux["deep_mm"] = stay
    flux["lateral_from_d_mm"] = from_d
    return flux

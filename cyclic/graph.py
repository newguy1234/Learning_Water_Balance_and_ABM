"""Optional leftover graph: neighbours, hop routing, QUBO.

Soil permeability is two-way. Darcy is q = −K ∇h: Kx, Ky, Kz are a conductivity
tensor, not a diode. Water can move either way along an edge if the head is higher
on the other side (see lateral.exchange_perched).

Default leftover hops are that soil graph: bund seepage equalises leftover head
between neighbours. Recapture (Kx+Ky) does not use hops or the main-drain lining.
`one_way_to_canal=True` is an optional diode toward decreasing canal distance; it
is not the default and is not the LFV path.

Input (always)
    leftover mm, area, neighbour edges, cyclic mask, Physics
    (canal distance is used only if one_way_to_canal is True)

Output (always)
    neighbour delta mm, in-transit stock

Optional
    Physics.use_graph=False  skip graph hops (stock stays put)
    Physics.one_way_to_canal=True  hop only downhill toward the main drain
    Physics.as_qubo=True     also return a QUBO: bit i = send leftover into the graph
                             vs stay on the plot (O'Malley-style, coarsened if large)

Poisson hops (time T = one day, no inner clock loop)
    μ = leftover_speed_m_d / median edge length, unless canal_hops is set.
    P(k) = e^{-μ} μ^k / k!. The day's stock is the mixture Σ P(k) H^k stock.
    One hop H only sends leftover toward a drier neighbour, and at most enough
    to equalise the two plots.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def leftover_graph(dist, edges, connected, one_way=False):
    """Undirected soil graph plus leftover hop edges.

    Nodes are plots. Undirected edges are two-way permeability. Default
    (`one_way=False`): both directions on every neighbour edge; leftover head
    chooses which way water moves. Optional (`one_way=True`): keep only edges
    that decrease plan-view canal distance.
    """
    dist = np.asarray(dist, dtype=float)
    n = dist.size
    connected = np.broadcast_to(np.asarray(connected, dtype=bool), (n,)).copy()
    empty = SimpleNamespace(
        n=n, dist=dist, connected=connected, one_way=bool(one_way),
        i=np.zeros(0, dtype=int), j=np.zeros(0, dtype=int),
        ux=np.zeros(0), uy=np.zeros(0), drop=np.zeros(0), length_m=np.zeros(0),
        undirected=np.zeros((0, 2), dtype=int),
        **{name: np.full(n, -1, dtype=int) for name in ("east", "west", "north", "south")},
    )
    if edges is None or len(getattr(edges, "i", [])) == 0:
        return empty
    i = np.asarray(edges.i, dtype=int)
    j = np.asarray(edges.j, dtype=int)
    ux = np.asarray(edges.ux, dtype=float)
    uy = np.asarray(edges.uy, dtype=float)
    length = np.ones(i.size, dtype=float)
    if getattr(edges, "dist_m", None) is not None and np.size(edges.dist_m) == i.size:
        length = np.maximum(np.asarray(edges.dist_m, dtype=float), 1.0)
    same = connected[i] & connected[j] & (connected[i] == connected[j])
    undirected = np.column_stack([i[same], j[same]])
    cardinal = {}
    for name in ("east", "west", "north", "south"):
        val = getattr(edges, name, None)
        cardinal[name] = (np.full(n, -1, dtype=int) if val is None
                          else np.asarray(val, dtype=int))
    if not one_way:
        src = np.concatenate([i[same], j[same]])
        dst = np.concatenate([j[same], i[same]])
        dpx = np.concatenate([ux[same], -ux[same]])
        dpy = np.concatenate([uy[same], -uy[same]])
        lengths = np.concatenate([length[same], length[same]])
        drops = np.ones_like(lengths)
        return SimpleNamespace(
            n=n, dist=dist, connected=connected, one_way=False,
            i=src.astype(int), j=dst.astype(int), ux=dpx, uy=dpy, drop=drops,
            length_m=lengths, undirected=undirected, **cardinal,
        )
    high = np.where(dist[i] > dist[j], i, j)
    low = np.where(dist[i] > dist[j], j, i)
    drop = np.abs(dist[i] - dist[j])
    downhill = same & (dist[i] != dist[j])
    toward_j = dist[i] > dist[j]
    dpx = np.where(toward_j[downhill], ux[downhill], -ux[downhill])
    dpy = np.where(toward_j[downhill], uy[downhill], -uy[downhill])
    drops = drop[downhill]
    lengths = length[downhill]
    src, dst = high[downhill], low[downhill]
    return SimpleNamespace(
        n=n, dist=dist, connected=connected, one_way=True,
        i=src.astype(int), j=dst.astype(int), ux=dpx, uy=dpy, drop=drops,
        length_m=lengths, undirected=undirected, **cardinal,
    )


def leftover_mu(graph, p):
    """Mean hop count for the day: canal_hops, or leftover_speed_m_d / median edge length."""
    hops = getattr(p, "canal_hops", None)
    if hops is not None:
        return max(float(hops), 0.0)
    speed = max(float(getattr(p, "leftover_speed_m_d", 2.0)), 0.0)
    lengths = np.asarray(getattr(graph, "length_m", []), dtype=float)
    dx = 10.0 if lengths.size == 0 else float(np.median(np.maximum(lengths, 1.0)))
    return speed / max(dx, 1.0)


def move_stock(stock, graph, area, p, axis):
    """Send leftover along axis-aligned edges, only wetter to drier, at most equalising."""
    n = stock.size
    neighbor = np.zeros(n)
    if graph is None or graph.i.size == 0:
        return neighbor
    ux, uy = np.asarray(graph.ux), np.asarray(graph.uy)
    keep = np.abs(ux) >= np.abs(uy) if axis == "x" else np.abs(uy) > np.abs(ux)
    if not np.any(keep):
        return neighbor
    high = np.asarray(graph.i, dtype=int)[keep]
    low = np.asarray(graph.j, dtype=int)[keep]
    drop = np.asarray(graph.drop, dtype=float)[keep]
    area = np.broadcast_to(np.asarray(area, dtype=float), (n,))
    factor = max(float(getattr(p, "farm_drain_stage_mm", 0.0)), 0.0)
    ref = max(float(getattr(p, "farm_drain_ref_mm", 1.0)), 1.0)
    factor = factor / ref
    if bool(getattr(graph, "one_way", False)):
        frac = np.clip(drop / max(float(p.canal_length_m), 1.0) * factor,
                       0.0, float(p.canal_max_frac))
    else:
        frac = np.full(drop.shape, np.clip(factor, 0.0, float(p.canal_max_frac)))
    connected = graph.connected
    frac = np.where(connected[high] == connected[low], frac, 0.0)
    water_drop = stock[high] - stock[low]
    frac = np.where(water_drop > 0.0, frac, 0.0)
    cap = np.maximum(water_drop, 0.0) / (1.0 + area[high] / np.maximum(area[low], 1e-9))
    moved = np.minimum(np.maximum(stock[high], 0.0) * frac, cap)
    vol = moved * area[high] / 1000.0
    np.add.at(neighbor, high, -moved)
    np.add.at(neighbor, low, 1000.0 * vol / np.maximum(area[low], 1e-9))
    trial = stock + neighbor
    if np.any(trial < -1e-9):
        room = np.where(neighbor < 0, stock / np.maximum(-neighbor, 1e-12), 1.0)
        neighbor = neighbor * np.min(np.clip(room, 0.0, 1.0))
    return neighbor


def one_hop(stock, graph, area, p):
    """One neighbour step: east–west then north–south on the leftover graph."""
    current = np.maximum(np.asarray(stock, dtype=float), 0.0)
    current = np.maximum(current + move_stock(current, graph, area, p, "x"), 0.0)
    current = np.maximum(current + move_stock(current, graph, area, p, "y"), 0.0)
    return current


def poisson_hop_weights(mu, tail=1e-12):
    """P(k) = e^{-μ} μ^k / k! truncated when the remaining tail is below `tail`."""
    mu = max(float(mu), 0.0)
    if mu == 0.0:
        return np.array([1.0])
    weights = [np.exp(-mu)]
    term = weights[0]
    k_max = int(np.ceil(mu + 8.0 * np.sqrt(max(mu, 0.0)) + 8.0))
    for k in range(1, max(k_max, 1) + 1):
        term *= mu / k
        weights.append(term)
        if term < tail and k >= mu:
            break
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum()) or 1.0
    return weights / total


def hop_stock(stock, graph, area, p):
    """Spread leftover for one day as a Poisson mixture of neighbour hops.

    μ is leftover_speed / Δx unless canal_hops is set. k = 0 leaves the pile
    where it is; each hop H only equalises toward a drier neighbour.
    """
    start = np.maximum(np.asarray(stock, dtype=float), 0.0)
    if not bool(getattr(p, "use_graph", True)):
        return start, np.zeros(start.size)
    mu = leftover_mu(graph, p)
    if mu == 0.0 or graph is None or getattr(graph, "i", np.zeros(0)).size == 0:
        return start, np.zeros(start.size)
    weights = poisson_hop_weights(mu)
    current = start
    mixed = weights[0] * start
    for k in range(1, len(weights)):
        current = one_hop(current, graph, area, p)
        mixed = mixed + weights[k] * current
    return mixed, mixed - start


def catch_at_canal(stock, dist, connected, p):
    """Unused on the recapture path. Kept for the non-ratio leftover_split=False branch."""
    length = max(float(p.canal_length_m), 1.0)
    near = np.exp(-np.maximum(np.asarray(dist, dtype=float), 0.0) / length)
    power = max(float(getattr(p, "canal_catch_power", 4.0)), 0.5)
    frac = np.clip(near ** power, 0.0, 1.0) * np.asarray(connected, dtype=float)
    lfv = np.maximum(stock, 0.0) * frac
    remain = np.maximum(stock - lfv, 0.0)
    return lfv, remain


def assignment_qubo(leftover_m3, graph, lfv_target_m3, smoothness=0.05, capture=None):
    """QUBO on graph nodes: x_i=1 leftover enters the graph, x_i=0 it stays.

    H = (x · (capture_i V_i) − LFV)² + J Σ_{undirected edges} (x_i − x_j)²

    capture_i is nearness to the canal (sinks count more toward LFV). Same shape as
    O'Malley Eq. 2. Callers that do not want a QUBO simply skip this.
    """
    vol = np.maximum(np.asarray(leftover_m3, dtype=float).ravel(), 0.0)
    n = vol.size
    dist = graph.dist
    length = max(float(np.median(dist[np.isfinite(dist)]) if np.any(np.isfinite(dist)) else 400.0), 1.0)
    near = np.exp(-np.maximum(dist, 0.0) / length)
    cap = near if capture is None else np.broadcast_to(np.asarray(capture, dtype=float), (n,))
    weight = cap * vol * np.asarray(graph.connected, dtype=float)
    target = float(lfv_target_m3)
    mean_vol2 = float(np.mean(vol[vol > 0]) ** 2) if np.any(vol > 0) else 1.0
    j_couple = float(smoothness) * mean_vol2
    q = {}

    def add(i, j, value):
        if abs(value) < 1e-18:
            return
        key = (i, j) if i <= j else (j, i)
        q[key] = q.get(key, 0.0) + float(value)

    offset = target * target
    for i in range(n):
        wi = float(weight[i])
        add(i, i, wi * wi - 2.0 * target * wi)
        for j in range(i + 1, n):
            add(i, j, 2.0 * wi * float(weight[j]))
    for a, b in np.atleast_2d(graph.undirected):
        a, b = int(a), int(b)
        if a == b:
            continue
        add(a, a, j_couple)
        add(b, b, j_couple)
        add(a, b, -2.0 * j_couple)
    return SimpleNamespace(Q=q, offset=offset, n_var=n, kind="leftover_graph")


def optional_qubo(leftover_mm, area, graph, p, lfv_target_m3=None):
    """Attach a QUBO only when Physics.as_qubo is set. Large fields are refused here
    (use cyclic.quantum.omalley_qubo to coarsen first)."""
    if not bool(getattr(p, "as_qubo", False)):
        return None
    n = int(np.asarray(leftover_mm).size)
    if n > 64:
        return SimpleNamespace(
            Q={}, offset=0.0, n_var=n, kind="leftover_graph",
            skipped="n_var>64; coarsen with cyclic.quantum.omalley_qubo",
        )
    vol = np.asarray(leftover_mm, dtype=float) * np.asarray(area, dtype=float) / 1000.0
    target = 0.0 if lfv_target_m3 is None else float(lfv_target_m3)
    if lfv_target_m3 is None:
        target = float((vol * np.exp(-np.maximum(graph.dist, 0.0) / max(float(p.canal_length_m), 1.0))).sum())
    return assignment_qubo(vol, graph, target, smoothness=float(getattr(p, "graph_smoothness", 0.05)))

"""O'Malley (2018) binary hydrologic inverse, solved on IBM Quantum (QAOA).

Daniel O'Malley, Sci. Rep. 8:6919. The D-Wave annealer minimised a binary quadratic
program for high vs low permeability. Here leftover at each coarsened cell is the bit:

    x_i = 1  leftover recapture (Kx, Ky) counted as same-day LFV
    x_i = 0  leftover stays (Kz topsoil + ground)

The Hamiltonian is the same shape as the paper's Eq. 2: data misfit (Table 4 Lf, Table 5
LFV) plus neighbour couplings on the cell graph. IBM gate hardware cannot hold 4,925
paddies, so cyclic paddies are clustered to `n_nodes` bits (O'Malley's 1D/2D grids were
also small). QAOA minimises that QUBO; without Qiskit, an exact enumerator or the
classical bit-flip loop is used. Set IBM_QUANTUM_TOKEN (or QISKIT_IBM_TOKEN) to send
the circuit to IBM Quantum Runtime. Optional: IBM_QUANTUM_INSTANCE (Cloud CRN) and
IBM_QUANTUM_CHANNEL (`ibm_quantum_platform` default, or `ibm_cloud`).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

from . import okura
from .identify import (
    ACTIVE, BOUNDS, construct_ratios, qubo_energy, solve_qubo_classical, table_lfv_m3,
)


def _kmeans(xy, k, seed=0, sweeps=12):
    rng = np.random.default_rng(seed)
    k = max(1, min(int(k), len(xy)))
    centres = xy[rng.choice(len(xy), size=k, replace=False)].copy()
    labels = np.zeros(len(xy), dtype=int)
    for _ in range(sweeps):
        d2 = ((xy[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)
        for c in range(k):
            members = xy[labels == c]
            if len(members):
                centres[c] = members.mean(axis=0)
    return labels, centres


def _ibm_token():
    return os.environ.get("IBM_QUANTUM_TOKEN") or os.environ.get("QISKIT_IBM_TOKEN")


def _ibm_runtime_service():
    """Connect to the current IBM Quantum Platform (legacy `ibm_quantum` channel is gone)."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = _ibm_token()
    instance = os.environ.get("IBM_QUANTUM_INSTANCE") or os.environ.get("QISKIT_IBM_INSTANCE")
    channel = os.environ.get("IBM_QUANTUM_CHANNEL") or "ibm_quantum_platform"
    if not token:
        try:
            return QiskitRuntimeService()
        except Exception:
            raise RuntimeError("Set IBM_QUANTUM_TOKEN or QISKIT_IBM_TOKEN for IBM Quantum")
    kwargs = {"channel": channel, "token": token}
    if instance:
        kwargs["instance"] = instance
    try:
        return QiskitRuntimeService(**kwargs)
    except Exception:
        if channel != "ibm_cloud":
            kwargs["channel"] = "ibm_cloud"
            return QiskitRuntimeService(**kwargs)
        raise


def leftover_volume_m3(year, frame):
    """Eq. 3 leftover from Table 4, as m³ on each cyclic paddy."""
    area = frame["area_m2"].to_numpy(dtype=float)
    block = np.asarray(frame["block"])
    cyclic = frame["cyclic"].to_numpy(dtype=bool)
    vol = np.zeros(len(frame))
    for name in okura.BLOCKS:
        row = okura.TABLE4_MM[(name, year)]
        leftover_mm = max(float(row["irrigation"] + row["rain"] - row["drainage"] - row["et"]), 0.0)
        mask = cyclic & (block == name)
        vol[mask] = leftover_mm * area[mask] / 1000.0
    return vol


def cluster_cyclic(frame, n_nodes=12, seed=0):
    """Coarsen cyclic paddies to n_nodes bits (O'Malley-scale graph)."""
    from .canal import distance_to_canal_m
    from .lateral import knn_edges

    cyclic = frame["cyclic"].to_numpy(dtype=bool)
    idx = np.flatnonzero(cyclic)
    lon = frame["lon"].to_numpy(dtype=float)
    lat = frame["lat"].to_numpy(dtype=float)
    xy = np.column_stack([lon[idx], lat[idx]])
    labels_local, centres = _kmeans(xy, n_nodes, seed=seed)
    n_nodes = int(centres.shape[0])
    labels = np.full(len(frame), -1, dtype=int)
    labels[idx] = labels_local
    edges = knn_edges(lon[idx], lat[idx], k=4)
    couple = set()
    for a, b in zip(edges.i, edges.j):
        ca, cb = int(labels_local[a]), int(labels_local[b])
        if ca != cb:
            couple.add((min(ca, cb), max(ca, cb)))
    return SimpleNamespace(
        n_nodes=n_nodes, labels=labels, centres=centres, idx=idx,
        couple=sorted(couple),
        lon=lon, lat=lat, cyclic=cyclic,
        block=np.asarray(frame["block"]),
        area=frame["area_m2"].to_numpy(dtype=float),
        dist_m=distance_to_canal_m(lon, lat),
        paddy_edges=knn_edges(lon, lat, k=4),
    )


def omalley_qubo(year=okura.COMPARISON_YEAR, n_nodes=12, smoothness=0.05, bits=3, seed=0):
    """Binary quadratic program: cluster leftover assignment + farm-to-main bits.

    H = Σ_block (x·V_block − Lf_block)² + (x·V_cyclic − LFV/capture_scale)²
        + J Σ_{adjacent clusters} (x_i − x_j)²
        + (ρ_sm D4 − D5)²

    Same structure as O'Malley Eq. 2 (linear + neighbour-quadratic), with Okura
    Table 4/5 as the observations instead of hydraulic head.
    """
    from . import data
    from .identify import table_main_d_m3

    frame = data.paddies()
    grid = cluster_cyclic(frame, n_nodes=n_nodes, seed=seed)
    vol = leftover_volume_m3(year, frame)
    n = grid.n_nodes
    cluster_vol = np.zeros(n)
    cluster_block = {name: np.zeros(n) for name in okura.BLOCKS}
    for i in range(len(frame)):
        c = int(grid.labels[i])
        if c < 0:
            continue
        cluster_vol[c] += vol[i]
        name = str(grid.block[i])
        if name in cluster_block:
            cluster_block[name][c] += vol[i]
    targets = {name: okura.TABLE4_MM[(name, year)]["lateral"] * okura.cyclic_area_m2(name) / 1000.0
               for name in okura.BLOCKS}
    capture = float(construct_ratios(year)["pool_capture_share"])
    lfv_target = table_lfv_m3(year)
    # If every leftover bit is lateral, LFV is capture × leftover; bits choose which
    # leftover is the moving share (paper: high-k vs low-k).
    lfv_weight = capture * cluster_vol
    mean_vol2 = float(np.mean(cluster_vol[cluster_vol > 0]) ** 2) if np.any(cluster_vol > 0) else 1.0
    j_couple = float(smoothness) * mean_vol2

    q = {}

    def add(i, j, value):
        if abs(value) < 1e-18:
            return
        key = (i, j) if i <= j else (j, i)
        q[key] = q.get(key, 0.0) + float(value)

    offset = 0.0

    def square_dot(weights, target):
        nonlocal offset
        offset += target * target
        for i in range(n):
            wi = float(weights[i])
            add(i, i, wi * wi - 2.0 * target * wi)
            for j in range(i + 1, n):
                add(i, j, 2.0 * wi * float(weights[j]))

    for name in okura.BLOCKS:
        square_dot(cluster_block[name], targets[name])
    square_dot(lfv_weight, lfv_target)

    for i, j in grid.couple:
        add(i, i, j_couple)
        add(j, j, j_couple)
        add(i, j, -2.0 * j_couple)

    sm_lo, sm_hi = BOUNDS["farm_to_main"]
    denom = max(2 ** bits - 1, 1)
    sm_w = (sm_hi - sm_lo) / denom * (2 ** np.arange(bits, dtype=float))
    d4 = sum(okura.TABLE4_MM[(b, year)]["drainage"] * okura.cyclic_area_m2(b) / 1000.0
             for b in okura.BLOCKS)
    d5 = table_main_d_m3(year)
    c0 = n
    c = sm_lo * d4 - d5
    offset += c * c
    for k, wk in enumerate(sm_w):
        add(c0 + k, c0 + k, (wk * d4) ** 2 + 2.0 * c * wk * d4)
        for m in range(k + 1, bits):
            add(c0 + k, c0 + m, 2.0 * wk * sm_w[m] * d4 ** 2)

    n_var = n + bits
    return SimpleNamespace(
        Q=q, offset=offset, n_var=n_var, n_points=n, bits=bits, c0=c0,
        sm_lo=sm_lo, sm_w=sm_w, grid=grid, cluster_vol=cluster_vol,
        year=year, smoothness=smoothness, capture=capture,
        decode=lambda z: decode_omalley(z, n, bits, sm_lo, sm_w, c0, grid),
    )


def decode_omalley(bitstring, n_points, bits, sm_lo, sm_w, c0, grid):
    z = np.asarray(bitstring, dtype=float).ravel()
    x = z[:n_points]
    farm = float(np.clip(sm_lo + z[c0:c0 + bits] @ sm_w, 0.0, 1.0))
    paddy_x = np.zeros(len(grid.labels))
    for i, c in enumerate(grid.labels):
        if c >= 0:
            paddy_x[i] = x[int(c)]
    return SimpleNamespace(x=x, paddy_x=paddy_x, farm_to_main=farm, grid=grid)


def assignment_to_shares(decoded, year=None):
    """Binary leftover field → per-paddy Kx/Ky/Kz/ground (signed Kx/Ky)."""
    from .canal import _axis_fractions, _flow_signs
    from .physics import Physics

    grid = decoded.grid
    x = np.asarray(decoded.paddy_x, dtype=float)
    n = x.size
    p = Physics()
    top = float(p.total_evaporable_mm / max(p.total_evaporable_mm + p.total_available_mm, 1e-9))
    fx, fy = _axis_fractions(n, grid.dist_m, grid.paddy_edges)
    sx, sy = _flow_signs(n, grid.dist_m, grid.paddy_edges)
    stay = 1.0 - x
    kx, ky = x * fx, x * fy
    constructed = construct_ratios(year or okura.COMPARISON_YEAR)
    return {
        "kx_ratio": kx,
        "ky_ratio": ky,
        "kz_ratio": stay * top,
        "ground_ratio": stay * (1.0 - top),
        "farm_to_main": np.where(grid.cyclic, decoded.farm_to_main, 0.0),
        "pool_capture_share": x,
        "kx_signed": kx * sx,
        "ky_signed": ky * sy,
        "kx_sign": sx,
        "ky_sign": sy,
        "assigned_lateral": x,
        "ratio_split": True,
        "force_satellite_et": False,
        "kx_ratio_mean": float(np.average(kx, weights=np.maximum(grid.area * grid.cyclic, 1e-12))),
        "ky_ratio_mean": float(np.average(ky, weights=np.maximum(grid.area * grid.cyclic, 1e-12))),
        "kz_ratio_mean": float(np.average(stay * top, weights=np.maximum(grid.area * grid.cyclic, 1e-12))),
        "ground_ratio_mean": float(np.average(stay * (1.0 - top), weights=np.maximum(grid.area * grid.cyclic, 1e-12))),
        "farm_to_main_mean": float(decoded.farm_to_main),
        "pool_capture_share_mean": constructed["pool_capture_share"],
    }


def shares_as_physics(shares):
    return {
        "kx_ratio": shares["kx_ratio_mean"],
        "ky_ratio": shares["ky_ratio_mean"],
        "kz_ratio": shares["kz_ratio_mean"],
        "ground_ratio": shares["ground_ratio_mean"],
        "farm_to_main": shares["farm_to_main_mean"],
        "pool_capture_share": shares["pool_capture_share_mean"],
        "ratio_split": True,
        "force_satellite_et": False,
    }


def solve_exact(encoding):
    """Enumerate all 2^n bitstrings. For O'Malley-scale n (≤16)."""
    n = encoding.n_var
    if n > 16:
        raise ValueError(f"exact enumerator only for n≤16, got {n}")
    best_z, best_e = None, np.inf
    for code in range(1 << n):
        z = np.array([(code >> i) & 1 for i in range(n)], dtype=float)
        energy = qubo_energy(encoding.Q, z, encoding.offset)
        if energy < best_e:
            best_e, best_z = energy, z
    decoded = encoding.decode(best_z)
    decoded.energy = best_e
    decoded.bitstring = best_z
    decoded.backend = "exact"
    return decoded


def _ising_from_qubo(Q, n):
    """x=(1−z)/2 maps QUBO to Ising Σ h_i z_i + Σ J_ij z_i z_j + const, z=±1."""
    h = np.zeros(n)
    J = {}
    const = 0.0
    for (i, j), qij in Q.items():
        if i == j:
            h[i] += 0.5 * qij
            const += 0.5 * qij
        else:
            J[(i, j)] = J.get((i, j), 0.0) + 0.25 * qij
            h[i] += 0.25 * qij
            h[j] += 0.25 * qij
            const += 0.25 * qij
    return h, J, const


def _sparse_pauli(h, J, n):
    from qiskit.quantum_info import SparsePauliOp

    labels, coeffs = ["I" * n], [0.0]
    for i, hi in enumerate(h):
        if abs(hi) < 1e-15:
            continue
        pauli = ["I"] * n
        pauli[i] = "Z"
        labels.append("".join(pauli))
        coeffs.append(float(hi))
    for (i, j), jij in J.items():
        if abs(jij) < 1e-15:
            continue
        pauli = ["I"] * n
        pauli[i] = "Z"
        pauli[j] = "Z"
        labels.append("".join(pauli))
        coeffs.append(float(jij))
    return SparsePauliOp.from_list(list(zip(labels, coeffs)))


def solve_qaoa_qiskit(encoding, reps=2, shots=1024, seed=0, ibm=False, backend_name=None):
    """QAOA on the O'Malley QUBO. ibm=True uses IBM Quantum Runtime if a token is set."""
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.primitives import StatevectorSampler
    from scipy.optimize import minimize

    n = encoding.n_var
    h, J, _ = _ising_from_qubo(encoding.Q, n)
    hamiltonian = _sparse_pauli(h, J, n)
    ansatz = QAOAAnsatz(hamiltonian, reps=int(reps))
    ansatz.measure_all()
    ansatz = ansatz.decompose(reps=8)
    rng = np.random.default_rng(seed)
    x0 = rng.normal(scale=0.3, size=ansatz.num_parameters)

    sampler = None
    service_backend = None
    if ibm:
        from qiskit_ibm_runtime import SamplerV2

        service = _ibm_runtime_service()
        service_backend = service.least_busy(simulator=False, operational=True) if not backend_name \
            else service.backend(backend_name)
        sampler = SamplerV2(mode=service_backend)
    else:
        try:
            from qiskit_aer.primitives import SamplerV2 as AerSampler
            sampler = AerSampler()
        except Exception:
            sampler = StatevectorSampler(seed=int(seed))

    def energy_from_counts(counts):
        total = sum(counts.values()) or 1
        e = 0.0
        for bitstr, times in counts.items():
            z = np.array([int(b) for b in bitstr[::-1]], dtype=float)[:n]
            e += qubo_energy(encoding.Q, z, encoding.offset) * times / total
        return e

    def objective(theta):
        bound = ansatz.assign_parameters(theta)
        if ibm:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            pm = generate_preset_pass_manager(backend=service_backend, optimization_level=1)
            isa = pm.run(bound)
            job = sampler.run([isa], shots=shots)
            pub = job.result()[0]
            counts = pub.data.meas.get_counts() if hasattr(pub.data, "meas") else pub.data.c.get_counts()
        else:
            job = sampler.run([bound], shots=shots)
            result = job.result()
            try:
                counts = result[0].data.meas.get_counts()
            except Exception:
                counts = result[0].data.c.get_counts()
        return energy_from_counts(counts)

    opt = minimize(objective, x0, method="COBYLA", options={"maxiter": 25, "rhobeg": 0.4})
    final = ansatz.assign_parameters(opt.x)
    if ibm:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        pm = generate_preset_pass_manager(backend=service_backend, optimization_level=1)
        job = sampler.run([pm.run(final)], shots=max(shots, 2048))
        pub = job.result()[0]
        counts = pub.data.meas.get_counts() if hasattr(pub.data, "meas") else pub.data.c.get_counts()
        backend_label = getattr(service_backend, "name", "ibm")
    else:
        job = sampler.run([final], shots=max(shots, 2048))
        result = job.result()
        try:
            counts = result[0].data.meas.get_counts()
        except Exception:
            counts = result[0].data.c.get_counts()
        backend_label = "aer"
    best_bits = max(counts, key=counts.get)
    z = np.array([int(b) for b in best_bits[::-1]], dtype=float)[:n]
    decoded = encoding.decode(z)
    decoded.energy = qubo_energy(encoding.Q, z, encoding.offset)
    decoded.bitstring = z
    decoded.backend = backend_label
    decoded.qaoa_energy = float(opt.fun)
    decoded.counts = dict(counts)
    return decoded


def solve_omalley_quantum(year=okura.COMPARISON_YEAR, n_nodes=12, smoothness=0.05,
                          bits=3, backend="auto", seed=0, shots=1024, reps=2,
                          ibm_backend=None):
    """Build the O'Malley QUBO for leftover ratios and minimise it.

    backend
        auto     IBM if a token is set, else Aer if Qiskit is installed, else exact/classical
        exact    brute force (n_nodes+bits ≤ 16)
        classical  local bit flips (same loop as a QPU would replace)
        aer      Qiskit Aer QAOA
        ibm      IBM Quantum Runtime QAOA
    """
    encoding = omalley_qubo(year, n_nodes=n_nodes, smoothness=smoothness, bits=bits, seed=seed)
    token = _ibm_token()
    choice = backend
    if choice == "auto":
        if token:
            choice = "ibm"
        else:
            try:
                import qiskit  # noqa: F401
                choice = "aer"
            except Exception:
                choice = "exact" if encoding.n_var <= 16 else "classical"
    if choice in ("aer", "ibm"):
        try:
            solved = solve_qaoa_qiskit(encoding, reps=reps, shots=shots, seed=seed,
                                       ibm=(choice == "ibm"), backend_name=ibm_backend)
        except Exception as error:
            if encoding.n_var <= 16:
                solved = solve_exact(encoding)
                solved.backend = f"exact (QAOA failed: {error})"
            else:
                solved = solve_qubo_classical(encoding, sweeps=8, seed=seed)
                solved.backend = f"classical (QAOA failed: {error})"
    elif choice == "exact":
        solved = solve_exact(encoding)
    else:
        solved = solve_qubo_classical(encoding, sweeps=8, seed=seed)
        solved.backend = "classical"
    shares = assignment_to_shares(solved, year=year)
    physics = shares_as_physics(shares)
    return SimpleNamespace(
        encoding=encoding, solved=solved, shares=shares, physics=physics,
        backend=getattr(solved, "backend", choice),
        energy=float(getattr(solved, "energy", np.nan)),
        n_var=encoding.n_var, n_nodes=encoding.n_points,
    )

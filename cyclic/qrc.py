"""Daily quantum reservoir computer for leftover soil ratios.

The QAOA path is a static QUBO. Here each irrigation-season day is one reservoir
drive. Inputs are measured (or table-interpolated) IR, rain, D and ET. A small
quantum circuit (default 4 qubits) is the frozen reservoir; only a linear readout
is trained. After training, inference is classical if the reservoir is simulated.

IBM: optional Estimator job for a few probe days (`ibm_days`), not 131 separate
optimisations. The time equation is a linear-reservoir / SINDy fit on the QRC
stay share vs leftover, compared to a two-box TEW/TAW filling law.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

from . import okura
from .identify import construct_ratios
from .physics import Physics
from .quantum import _ibm_token, _ibm_runtime_service


FEATURE_NAMES = ("rain", "irrigation", "drainage", "et", "leftover", "leftover_lag",
                 "inflow", "season_frac")
GENERALIST_NAMES = (
    "leftover", "leftover_lag", "et", "rain", "irrigation", "drainage",
    "inflow", "season_frac", "lon", "lat", "dist_m", "near",
    "leftover*near", "leftover*dist_km",
)
RATIO_NAMES = ("kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio")


def qrc_path(year, folder=None):
    from . import data as _data
    folder = folder or os.path.join(_data.PROJECT, "outputs", "ratios")
    return os.path.join(folder, f"qrc_{int(year)}.npz")


def daily_forcing(year):
    """Area-weighted cyclic-block daily mm: IR, R, D, ET, leftover (Eq. 3)."""
    from .imitation import load_observed

    blocks, _station, source = load_observed(year)
    blocks = blocks.copy()
    if "leftover_mm" not in blocks.columns:
        ir = blocks["irrigation_mm"] if "irrigation_mm" in blocks.columns else (
            blocks["irrigation_m3"] / blocks["area_m2"] * 1000.0)
        d = blocks["drainage_mm"] if "drainage_mm" in blocks.columns else (
            blocks["drainage_m3"] / blocks["area_m2"] * 1000.0)
        rain = blocks["rain_mm"] if "rain_mm" in blocks.columns else 0.0
        et = blocks["et_mm"] if "et_mm" in blocks.columns else 0.0
        blocks["irrigation_mm"] = ir
        blocks["drainage_mm"] = d
        blocks["rain_mm"] = rain
        blocks["et_mm"] = et
        blocks["leftover_mm"] = np.asarray(ir) + np.asarray(rain) - np.asarray(d) - np.asarray(et)
    dates = np.array(sorted(blocks["date"].unique()))
    rows = []
    for day in dates:
        g = blocks[blocks["date"] == day]
        w = np.maximum(g["area_m2"].to_numpy(dtype=float), 1e-9)
        def avg(name):
            return float(np.average(g[name].to_numpy(dtype=float), weights=w))
        leftover = avg("leftover_mm") if "leftover_mm" in g.columns else (
            avg("irrigation_mm") + avg("rain_mm") - avg("drainage_mm") - avg("et_mm"))
        rows.append({
            "date": day,
            "rain": avg("rain_mm"),
            "irrigation": avg("irrigation_mm"),
            "drainage": avg("drainage_mm"),
            "et": avg("et_mm"),
            "leftover": leftover,
            "area_m2": float(w.sum()),
        })
    frame = pd_frame(rows)
    frame.attrs["source"] = source
    return frame


def pd_frame(rows):
    import pandas as pd
    return pd.DataFrame(rows)


def soil_box_ratios(frame, constructed=None, p=None):
    """Daily leftover simplex from a two-box TEW/TAW fill, ET drawn from the stores.

    This is the time-dependent *target* the QRC tries to imitate. Seasonal means stay
    close to Tables 4–5 via construct_ratios; the day-to-day shape is the filling law.
    """
    constructed = constructed or construct_ratios()
    p = p or Physics()
    tew, taw = float(p.total_evaporable_mm), float(p.total_available_mm)
    top_frac = tew / max(tew + taw, 1e-9)
    capture = float(constructed["pool_capture_share"])
    n = len(frame)
    leftover = np.asarray(frame["leftover"], dtype=float)
    et = np.asarray(frame["et"], dtype=float)
    top = 0.0
    deep = 0.0
    y = np.zeros((n, 4))
    stores = np.zeros((n, 2))
    for t in range(n):
        L = max(float(leftover[t]), 0.0)
        move = capture * L
        stay = L - move
        room_top = max(tew - top, 0.0)
        to_top = min(stay, room_top)
        rest = stay - to_top
        room_deep = max(taw - deep, 0.0)
        to_deep = min(rest, room_deep)
        extra_lat = rest - to_deep
        move += extra_lat
        if L > 1e-9:
            y[t, 0] = 0.5 * move / L
            y[t, 1] = 0.5 * move / L
            y[t, 2] = to_top / L
            y[t, 3] = to_deep / L
        else:
            y[t] = np.array([constructed["kx_ratio"], constructed["ky_ratio"],
                             constructed["kz_ratio"], constructed["ground_ratio"]], dtype=float)
        top = min(tew, max(0.0, top + to_top - et[t] * top_frac))
        deep = min(taw, max(0.0, deep + to_deep - et[t] * (1.0 - top_frac)))
        stores[t] = (top, deep)
    row_sum = y.sum(axis=1, keepdims=True)
    y = np.divide(y, np.maximum(row_sum, 1e-12))
    return y, stores


def _features(frame):
    leftover = np.asarray(frame["leftover"], dtype=float)
    lag = np.concatenate([[leftover[0]], leftover[:-1]])
    inflow = np.asarray(frame["rain"], dtype=float) + np.asarray(frame["irrigation"], dtype=float)
    n = len(frame)
    raw = np.column_stack([
        frame["rain"], frame["irrigation"], frame["drainage"], frame["et"],
        leftover, lag, inflow, np.arange(n) / max(n - 1, 1),
    ]).astype(float)
    scale = np.maximum(np.std(raw, axis=0), 1e-6)
    mean = raw.mean(axis=0)
    z = (raw - mean) / scale
    return z, raw, mean, scale


def _ry(theta):
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta):
    return np.diag([np.exp(-1j * theta / 2.0), np.exp(1j * theta / 2.0)])


def _apply_1q(psi, n, q, u):
    tensor = psi.reshape([2] * n)
    tensor = np.moveaxis(tensor, q, 0)
    tensor = np.tensordot(u, tensor, axes=([1], [0]))
    tensor = np.moveaxis(tensor, 0, q)
    return tensor.reshape(-1)


def _apply_cx(psi, n, c, t):
    tensor = psi.reshape([2] * n)
    idx = [slice(None)] * n
    # |11> target flip: swap target on control==1
    moved = np.moveaxis(tensor, (c, t), (0, 1))
    a, b = moved[1, 0].copy(), moved[1, 1].copy()
    moved[1, 0], moved[1, 1] = b, a
    tensor = np.moveaxis(moved, (0, 1), (c, t))
    return tensor.reshape(-1)


def reservoir_statevector(features, n_qubits=4, layers=2, seed=0):
    """<Z> on each qubit after a fixed random reservoir, one row per day."""
    rng = np.random.default_rng(seed)
    twists = rng.normal(scale=0.7, size=(layers, n_qubits))
    days, n_feat = features.shape
    z = np.zeros((days, n_qubits))
    dim = 1 << n_qubits
    for t in range(days):
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1.0
        for q, feat_i in enumerate(range(n_feat)):
            angle = float(features[t, feat_i]) * (np.pi / 2.0)
            psi = _apply_1q(psi, n_qubits, q % n_qubits, _ry(angle))
        for layer in range(layers):
            for q in range(n_qubits):
                psi = _apply_1q(psi, n_qubits, q, _rz(twists[layer, q]))
            for q in range(n_qubits):
                psi = _apply_cx(psi, n_qubits, q, (q + 1) % n_qubits)
        for q in range(n_qubits):
            tensor = psi.reshape([2] * n_qubits)
            tensor = np.moveaxis(tensor, q, 0)
            p0 = float(np.vdot(tensor[0].ravel(), tensor[0].ravel()).real)
            z[t, q] = 2.0 * p0 - 1.0
    return z


def _qiskit_reservoir_circuit(n_qubits, layers, twists, x_params):
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    for feat_i, param in enumerate(x_params):
        qc.ry(param, feat_i % n_qubits)
    for layer in range(layers):
        for q in range(n_qubits):
            qc.rz(float(twists[layer, q]), q)
        for q in range(n_qubits):
            qc.cx(q, (q + 1) % n_qubits)
    return qc


def reservoir_ibm(features, n_qubits=4, layers=2, seed=0, day_index=None, shots=2048):
    """Estimator <Z> for selected rows on IBM Runtime. One job, several pubs."""
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import EstimatorV2

    rng = np.random.default_rng(seed)
    twists = rng.normal(scale=0.7, size=(layers, n_qubits))
    n_feat = int(features.shape[1])
    x_params = ParameterVector("x", n_feat)
    qc = _qiskit_reservoir_circuit(n_qubits, layers, twists, x_params)
    service = _ibm_runtime_service()
    backend = service.least_busy(simulator=False, operational=True)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa = pm.run(qc)
    obs = []
    for q in range(n_qubits):
        label = ["I"] * n_qubits
        label[q] = "Z"
        obs.append(SparsePauliOp.from_list([("".join(label), 1.0)]))
    isa_obs = [o.apply_layout(isa.layout) for o in obs]
    estimator = EstimatorV2(mode=backend)
    days = np.arange(len(features)) if day_index is None else np.asarray(day_index, dtype=int)
    pubs = []
    for t in days:
        angles = [float(features[t, i]) * (np.pi / 2.0) for i in range(n_feat)]
        pubs.append((isa, isa_obs, angles))
    print("IBM Estimator", getattr(backend, "name", "ibm"), "pubs", len(pubs), flush=True)
    job = estimator.run(pubs, precision=1.0 / np.sqrt(shots))
    print("IBM job submitted", job.job_id() if callable(getattr(job, "job_id", None)) else job, flush=True)
    result = job.result()
    z = np.zeros((len(days), n_qubits))
    for i, pub in enumerate(result):
        z[i] = np.asarray(pub.data.evs, dtype=float).ravel()[:n_qubits]
    job_id = job.job_id() if callable(getattr(job, "job_id", None)) else str(getattr(job, "job_id", ""))
    return z, getattr(backend, "name", "ibm"), job_id


def _ridge(x, y, lam=1e-2):
    xtx = x.T @ x + lam * np.eye(x.shape[1])
    return np.linalg.solve(xtx, x.T @ y)


def _softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def fit_readout(r, target, lam=1e-2):
    """Affine map then simplex: ratios = softmax(R W)."""
    bias = np.ones((len(r), 1))
    design = np.hstack([r, bias])
    # Train in log-space of the target simplex.
    clipped = np.clip(target, 1e-6, 1.0)
    logits = np.log(clipped)
    weights = _ridge(design, logits, lam=lam)
    pred = _softmax(design @ weights)
    return weights, pred


def linear_time_equation(stay, leftover, stores_top, et, rain):
    """SINDy-style discrete update for topsoil water (mm).

    S(t+1) = a0 + a1 S(t) + a2 stay(t) L+(t) + a3 ET(t) + a4 rain(t)
    """
    s = np.asarray(stores_top, dtype=float)
    stay = np.asarray(stay, dtype=float)
    lp = np.maximum(np.asarray(leftover, dtype=float), 0.0)
    et = np.asarray(et, dtype=float)
    rain = np.asarray(rain, dtype=float)
    if len(s) < 3:
        return SimpleNamespace(coeffs=np.zeros(5), r2=0.0, names=(), text="too few days")
    y = s[1:]
    x = np.column_stack([
        np.ones(len(y)),
        s[:-1],
        stay[:-1] * lp[:-1],
        et[:-1],
        rain[:-1],
    ])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    fit = x @ coef
    ss_res = float(((y - fit) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    names = ("1", "S(t)", "stay(t)*L+(t)", "ET(t)", "rain(t)")
    terms = [f"{c:+.4f} {n}" for c, n in zip(coef, names)]
    text = "S_top(t+1) = " + " ".join(terms) + f"    (R^2={r2:.3f})"
    return SimpleNamespace(coeffs=coef, r2=r2, names=names, text=text, fit=fit)


def stay_feature_equation(stay, features_raw, feature_names=FEATURE_NAMES):
    """stay(t) ~= b·u(t)  (raw mm / day-fraction, not z-scored)."""
    y = np.asarray(stay, dtype=float)
    x = np.column_stack([np.ones(len(y)), np.asarray(features_raw, dtype=float)])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    fit = x @ coef
    ss_res = float(((y - fit) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    names = ("1",) + tuple(feature_names)
    terms = [f"{c:+.4g} {n}" for c, n in zip(coef, names)]
    text = "stay(t) = kz+ground ~= " + " ".join(terms) + f"    (R^2={r2:.3f})"
    return SimpleNamespace(coeffs=coef, r2=r2, names=names, text=text, feat_mean=None,
                           stay_target=None)


def generalist_drive(leftover, irrigation, drainage, et, rain, lon, lat, dist,
                     canal_m=400.0):
    """(days, n, F) design without the intercept. Broadcasts 1d forcings across paddies."""
    leftover = np.asarray(leftover, dtype=float)
    if leftover.ndim == 1:
        leftover = leftover[:, None]
    days, n = leftover.shape

    def field(val, as_time=True):
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 0:
            arr = np.full((days, n), float(arr))
        elif arr.ndim == 1:
            if arr.size == days and as_time:
                arr = np.broadcast_to(arr[:, None], (days, n))
            else:
                arr = np.broadcast_to(arr.ravel()[None, :], (days, n))
        elif arr.shape != (days, n):
            arr = np.broadcast_to(np.reshape(arr, leftover.shape), leftover.shape)
        return np.asarray(arr, dtype=float)

    irrigation = field(irrigation, True)
    drainage = field(drainage, True)
    et = field(et, True)
    rain = field(rain, True)
    lon = field(lon, False)
    lat = field(lat, False)
    dist = field(dist, False)
    lag = np.empty_like(leftover)
    lag[0] = leftover[0]
    if days > 1:
        lag[1:] = leftover[:-1]
    season = (np.arange(days) / max(days - 1, 1))[:, None] * np.ones((1, n))
    near = np.exp(-np.maximum(dist, 0.0) / max(float(canal_m), 1.0))
    return np.stack([
        leftover, lag, et, rain, irrigation, drainage,
        rain + irrigation, season, lon, lat, dist, near,
        leftover * near, leftover * (dist / 1000.0),
    ], axis=-1)


def fit_centered_generalist_stay(stay, drive, stay_target, lam=1e-1):
    """stay = clip(stay_target + b · (u - u_bar), 0, 1). Intercept is Table 5, not free."""
    y = np.asarray(stay, dtype=float) - float(stay_target)
    x = np.asarray(drive, dtype=float)
    mean = x.mean(axis=0)
    xc = x - mean
    xtx = xc.T @ xc + float(lam) * np.eye(xc.shape[1])
    beta = np.linalg.solve(xtx, xc.T @ y)
    fit = float(stay_target) + xc @ beta
    ss_res = float(((np.asarray(stay, dtype=float) - fit) ** 2).sum())
    ss_tot = float(((np.asarray(stay, dtype=float) - np.mean(stay)) ** 2).sum()) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    names = ("stay_target",) + tuple(GENERALIST_NAMES)
    coef = np.concatenate([[float(stay_target)], beta])
    terms = [f"{c:+.4g} ({n}-mean)" if n != "stay_target" else f"{c:.4g}"
             for c, n in zip(coef, names)]
    text = ("stay(t,x,y) = clip(stay_target + b.(u - u_bar), 0, 1)  with stay_target="
            f"{stay_target:.4f}    (R^2={r2:.3f})\n  b = " + " ".join(terms[1:]))
    return SimpleNamespace(coeffs=coef, r2=r2, names=names, text=text, feat_mean=mean,
                           stay_target=float(stay_target))


def eval_generalist_stay(coeffs, leftover, irrigation, drainage, et, rain, lon, lat, dist,
                         canal_m=400.0, feat_mean=None, stay_target=None):
    drive = generalist_drive(leftover, irrigation, drainage, et, rain, lon, lat, dist, canal_m)
    days, n, n_feat = drive.shape
    flat = drive.reshape(days * n, n_feat)
    coef = np.asarray(coeffs, dtype=float)
    if feat_mean is not None:
        intercept = float(stay_target if stay_target is not None else coef[0])
        beta = coef[1:] if coef.size == n_feat + 1 else coef
        stay = intercept + (flat - np.asarray(feat_mean, dtype=float)) @ beta
    else:
        design = np.column_stack([np.ones(days * n), flat])
        stay = design @ coef
    return np.clip(stay.reshape(days, n), 0.0, 1.0)


def pooled_store_equation(stores_top, stay, leftover, et, rain):
    """One S_top(t+1) law pooled over cluster time series."""
    xs, ys = [], []
    for s, st, L, e, r in zip(stores_top, stay, leftover, et, rain):
        s = np.asarray(s, dtype=float)
        if len(s) < 3:
            continue
        lp = np.maximum(np.asarray(L, dtype=float), 0.0)
        st = np.asarray(st, dtype=float)
        e = np.asarray(e, dtype=float)
        r = np.asarray(r, dtype=float)
        ys.append(s[1:])
        xs.append(np.column_stack([
            np.ones(len(s) - 1), s[:-1], st[:-1] * lp[:-1], e[:-1], r[:-1],
        ]))
    if not ys:
        return SimpleNamespace(coeffs=np.zeros(5), r2=0.0, names=(), text="too few days")
    x = np.vstack(xs)
    y = np.concatenate(ys)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    fit = x @ coef
    ss_res = float(((y - fit) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    names = ("1", "S(t)", "stay(t,x,y)*L+(t)", "ET(t)", "rain(t)")
    terms = [f"{c:+.4f} {n}" for c, n in zip(coef, names)]
    text = "S_top(t+1) = " + " ".join(terms) + f"    (R^2={r2:.3f})"
    return SimpleNamespace(coeffs=coef, r2=r2, names=names, text=text)


def format_generalist(eq_stay, eq_store, stay_target, top, n_clusters, backend="statevector"):
    lines = [
        f"Generalist leftover QRC - reservoir {backend} - {n_clusters} train clusters",
        "Applies at any parcel: stay = clip(stay_target + b.(u - u_bar), 0, 1); capture = 1-stay.",
        f"Kz = stay * {top:.4f} (TEW/TAW); ground = stay * {1.0 - top:.4f}.",
        "Kx, Ky = capture split on the local canal axis (not in b).",
        f"Season leftover-weighted stay pinned to {stay_target:.4f}.",
        eq_stay.text,
        eq_store.text,
        "u = [1, leftover, leftover_lag, ET, rain, IR, D, inflow, season_frac, "
        "lon, lat, dist_m, near=exp(-dist/400), leftover*near, leftover*dist_km]",
    ]
    return "\n".join(lines)


def train_qrc(year=okura.COMPARISON_YEAR, n_qubits=4, layers=2, seed=0,
              backend="auto", ibm_days=8, shots=2048, holdout_year=None, folder=None):
    """Fit daily leftover ratios with a quantum reservoir; extract a time equation."""
    frame = daily_forcing(year)
    constructed = construct_ratios(year)
    target, stores = soil_box_ratios(frame, constructed)
    z, raw, mean, scale = _features(frame)
    r_class = reservoir_statevector(z, n_qubits=n_qubits, layers=layers, seed=seed)
    ibm_meta = SimpleNamespace(backend="statevector", job_id="", days=np.array([], dtype=int),
                               z=None)
    choice = backend
    if choice == "auto":
        choice = "ibm" if _ibm_token() else "classical"
    if choice == "ibm":
        idx = np.linspace(0, len(z) - 1, num=max(int(ibm_days), 1), dtype=int)
        idx = np.unique(idx)
        try:
            z_ibm, name, job_id = reservoir_ibm(
                z, n_qubits=n_qubits, layers=layers, seed=seed, day_index=idx, shots=shots)
            r = r_class.copy()
            r[idx] = z_ibm
            ibm_meta = SimpleNamespace(backend=name, job_id=job_id, days=idx, z=z_ibm)
        except Exception as error:
            ibm_meta = SimpleNamespace(backend=f"statevector (IBM failed: {error})",
                                       job_id="", days=np.array([], dtype=int), z=None)
            r = r_class
    else:
        r = r_class
    weights, pred = fit_readout(r, target)
    stay = pred[:, 2] + pred[:, 3]
    stay_tgt = target[:, 2] + target[:, 3]
    means = {name: float(pred[:, i].mean()) for i, name in enumerate(RATIO_NAMES)}
    means["farm_to_main"] = float(constructed["farm_to_main"])
    means["pool_capture_share"] = float(means["kx_ratio"] + means["ky_ratio"])
    eq_s = linear_time_equation(stay, frame["leftover"], stores[:, 0], frame["et"], frame["rain"])
    eq_stay = stay_feature_equation(stay, raw)
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    stay_rmse = float(np.sqrt(np.mean((stay - stay_tgt) ** 2)))
    constant = stay.std() < 0.02
    transfer = None
    if holdout_year is not None and int(holdout_year) != int(year):
        transfer = evaluate_qrc(holdout_year, weights, mean, scale, n_qubits=n_qubits,
                                layers=layers, seed=seed)
    result = SimpleNamespace(
        year=int(year), frame=frame, target=target, pred=pred, stores=stores,
        features=z, features_raw=raw, reservoir=r, weights=weights,
        means=means, constructed=constructed, rmse=rmse, stay_rmse=stay_rmse,
        eq_store=eq_s, eq_stay=eq_stay, constant_stay=bool(constant),
        ibm=ibm_meta, n_qubits=n_qubits, layers=layers, seed=seed,
        transfer=transfer,
    )
    path = save_qrc(result, folder=folder)
    result.path = path
    result.equation_text = format_equations(result)
    _write_equation_file(result, folder=folder)
    return result


def evaluate_qrc(year, weights, feat_mean, feat_scale, n_qubits=4, layers=2, seed=0):
    frame = daily_forcing(year)
    constructed = construct_ratios(year)
    target, stores = soil_box_ratios(frame, constructed)
    raw = np.column_stack([
        frame["rain"], frame["irrigation"], frame["drainage"], frame["et"],
        frame["leftover"],
        np.concatenate([[frame["leftover"].iloc[0]], frame["leftover"].to_numpy()[:-1]]),
        np.asarray(frame["rain"]) + np.asarray(frame["irrigation"]),
        np.arange(len(frame)) / max(len(frame) - 1, 1),
    ]).astype(float)
    z = (raw - feat_mean) / np.maximum(feat_scale, 1e-6)
    r = reservoir_statevector(z, n_qubits=n_qubits, layers=layers, seed=seed)
    design = np.hstack([r, np.ones((len(r), 1))])
    pred = _softmax(design @ weights)
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    stay = pred[:, 2] + pred[:, 3]
    stay_tgt = target[:, 2] + target[:, 3]
    return SimpleNamespace(year=int(year), rmse=rmse,
                           stay_rmse=float(np.sqrt(np.mean((stay - stay_tgt) ** 2))),
                           means={name: float(pred[:, i].mean()) for i, name in enumerate(RATIO_NAMES)})


def format_equations(result):
    lines = [
        f"Daily QRC leftover ratios - {result.year} - reservoir {result.ibm.backend}",
        f"Season-mean Kx={result.means['kx_ratio']:.3f} Ky={result.means['ky_ratio']:.3f} "
        f"Kz={result.means['kz_ratio']:.3f} ground={result.means['ground_ratio']:.3f}",
        f"QRC vs two-box target RMSE={result.rmse:.4f}  stay RMSE={result.stay_rmse:.4f}",
        result.eq_store.text,
        result.eq_stay.text,
    ]
    if result.constant_stay:
        lines.append(
            "Stay share is nearly constant in time: the QRC did not find an extra "
            "seasonal wave beyond the two-box fill. The useful equation is S(t+1).")
    else:
        lines.append(
            "Stay share moves with the day: use stay(t) in the S(t+1) update, not a "
            "single Table 4/5 constant.")
    if result.transfer is not None:
        lines.append(
            f"Held-out {result.transfer.year}: RMSE={result.transfer.rmse:.4f} "
            f"stay RMSE={result.transfer.stay_rmse:.4f}")
    if result.ibm.job_id:
        days = [int(i) for i in np.asarray(result.ibm.days).ravel()]
        lines.append(f"IBM job {result.ibm.job_id} on days {days}")
    return "\n".join(lines)


def save_qrc(result, folder=None):
    from . import data as _data

    folder = folder or os.path.join(_data.PROJECT, "outputs", "ratios")
    os.makedirs(folder, exist_ok=True)
    path = qrc_path(result.year, folder)
    payload = {
        "year": np.int32(result.year),
        "pred": np.asarray(result.pred, dtype=np.float32),
        "target": np.asarray(result.target, dtype=np.float32),
        "stores": np.asarray(result.stores, dtype=np.float32),
        "reservoir": np.asarray(result.reservoir, dtype=np.float32),
        "weights": np.asarray(result.weights, dtype=np.float32),
        "rmse": np.float64(result.rmse),
        "stay_rmse": np.float64(result.stay_rmse),
        "backend": np.array(str(result.ibm.backend)),
        "equation": np.array(format_equations(result)),
    }
    for i, name in enumerate(RATIO_NAMES):
        payload[name] = np.float64(result.means[name])
    np.savez_compressed(path, **payload)
    return path


def _write_equation_file(result, folder=None):
    from . import data as _data
    folder = folder or os.path.join(_data.PROJECT, "outputs", "ratios")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"qrc_{result.year}_equation.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(result.equation_text if hasattr(result, "equation_text")
                     else format_equations(result))
        handle.write("\n")
    return path


def load_qrc_result(year, folder=None):
    """Arrays and text from the last QRC save, or None."""
    path = qrc_path(year, folder)
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=True) as data:
        pred = np.asarray(data["pred"], dtype=float)
        target = np.asarray(data["target"], dtype=float)
        stores = np.asarray(data["stores"], dtype=float)
        eq = ""
        if "equation" in data.files:
            raw = np.asarray(data["equation"])
            eq = str(raw.item() if raw.shape == () else raw)
        return SimpleNamespace(
            year=int(data["year"]) if "year" in data.files else int(year),
            pred=pred, target=target, stores=stores,
            rmse=float(data["rmse"]) if "rmse" in data.files else float("nan"),
            stay_rmse=float(data["stay_rmse"]) if "stay_rmse" in data.files else float("nan"),
            backend=str(data["backend"]) if "backend" in data.files else "",
            means={name: float(data[name]) for name in RATIO_NAMES if name in data.files},
            equation=eq,
            path=path,
        )


def load_qrc_equation(year, folder=None):
    path = qrc_path(year, folder)
    if folder is None:
        from . import data as _data
        text = os.path.join(_data.PROJECT, "outputs", "ratios", f"qrc_{int(year)}_equation.txt")
    else:
        text = os.path.join(folder, f"qrc_{int(year)}_equation.txt")
    if os.path.isfile(text):
        return open(text, encoding="utf-8").read()
    loaded = load_qrc_result(year, folder)
    if loaded is None:
        return None
    return loaded.equation or None


def load_spatial_qrc_equation(year, folder=None):
    if folder is None:
        from . import data as _data
        folder = os.path.join(_data.PROJECT, "outputs", "ratios")
    text = os.path.join(folder, f"qrc_spatial_{int(year)}_equation.txt")
    if os.path.isfile(text):
        return open(text, encoding="utf-8").read()
    loaded = load_spatial_qrc(year, folder)
    if loaded is None:
        return None
    packed = np.load(spatial_qrc_path(year, folder), allow_pickle=True)
    if "equation" not in packed.files:
        return None
    raw = np.asarray(packed["equation"])
    return str(raw.item() if raw.shape == () else raw)


def qrc_figure(year=okura.COMPARISON_YEAR, folder=None):
    """Daily leftover shares, two-box target, and topsoil store from the saved QRC."""
    from matplotlib.figure import Figure

    loaded = load_qrc_result(year, folder)
    if loaded is None:
        return None
    pred, target, stores = loaded.pred, loaded.target, loaded.stores
    n = len(pred)
    t = np.arange(n)
    fig = Figure(figsize=(12.4, 9.2))
    grid = fig.add_gridspec(3, 1, hspace=0.38)
    ax0 = fig.add_subplot(grid[0, 0])
    ax0.plot(t, pred[:, 0], color="#1f4e79", lw=1.4, label="Kx QRC")
    ax0.plot(t, pred[:, 1], color="#2e75b6", lw=1.1, ls="--", label="Ky QRC")
    ax0.plot(t, pred[:, 2], color="#c45911", lw=1.6, label="Kz QRC")
    ax0.plot(t, pred[:, 3], color="#7f7f7f", lw=1.2, label="ground QRC")
    ax0.plot(t, target[:, 2], color="#c45911", lw=0.9, ls=":", alpha=0.85, label="Kz two-box")
    ax0.set_ylabel("share of leftover")
    ax0.set_ylim(-0.02, 1.02)
    ax0.legend(fontsize=8, ncol=5, loc="upper right")
    ax0.set_title(
        f"Daily leftover ratios {loaded.year}  ({loaded.backend})  "
        f"RMSE {loaded.rmse:.3f} vs two-box fill"
    )
    ax0.set_xlim(0, n - 1)

    ax1 = fig.add_subplot(grid[1, 0], sharex=ax0)
    stay_p = pred[:, 2] + pred[:, 3]
    stay_t = target[:, 2] + target[:, 3]
    ax1.plot(t, stay_p, color="#c45911", lw=1.6, label="stay QRC (Kz+ground)")
    ax1.plot(t, stay_t, color="black", lw=1.1, ls=":", label="stay two-box")
    ax1.plot(t, pred[:, 0] + pred[:, 1], color="#1f4e79", lw=1.2, label="lateral QRC (Kx+Ky)")
    ax1.set_ylabel("share")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_title("Stay in soil vs leftover recapture")

    ax2 = fig.add_subplot(grid[2, 0], sharex=ax0)
    ax2.plot(t, stores[:, 0], color="#c45911", lw=1.6, label="topsoil S (mm)")
    ax2.plot(t, stores[:, 1], color="#7f7f7f", lw=1.2, label="ground S (mm)")
    ax2.set_ylabel("store (mm)")
    ax2.set_xlabel("day of irrigation season (0 = 17 Apr)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_title("Two-box stores that the QRC stay share is fitted to")
    fig.align_ylabels([ax0, ax1, ax2])
    return fig


def pixel_vs_district(year=okura.COMPARISON_YEAR, folder=None):
    """Apply the QRC stay share and S(t+1) equation on every cyclic paddy, then
    area-weight back to the district.

    Leftover mm is the block daily series (same on every paddy in the block),
    plus a canal-scaled variant so leftover is larger near the canal. Rain and
    ET stay the site series. stay(t) is the district QRC readout.
    """
    from . import data
    from .canal import distance_to_canal_m
    from .imitation import load_observed

    loaded = load_qrc_result(year, folder)
    if loaded is None:
        raise FileNotFoundError(f"no QRC save for {year}")
    frame = daily_forcing(year)
    stay = loaded.pred[:, 2] + loaded.pred[:, 3]
    eq = linear_time_equation(
        stay, frame["leftover"], loaded.stores[:, 0], frame["et"], frame["rain"])
    a0, a1, a2, a3, a4 = eq.coeffs
    leftover_d = np.asarray(frame["leftover"], dtype=float)
    et = np.asarray(frame["et"], dtype=float)
    rain = np.asarray(frame["rain"], dtype=float)
    n_days = len(stay)

    def integrate(lp, stay_t, s0=0.0):
        s = np.empty(n_days)
        s[0] = s0
        for t in range(n_days - 1):
            s[t + 1] = a0 + a1 * s[t] + a2 * stay_t[t] * lp[t] + a3 * et[t] + a4 * rain[t]
        return s

    s_district = integrate(np.maximum(leftover_d, 0.0), stay)
    rmse_eq_vs_box = float(np.sqrt(np.mean((s_district - loaded.stores[:, 0]) ** 2)))

    blocks, _station, _src = load_observed(year)
    dates = np.asarray(frame["date"])
    leftover_block = {}
    for name in okura.BLOCKS:
        g = blocks[blocks["block"] == name].copy()
        g = g.set_index("date").reindex(dates)
        ir = g["irrigation_mm"].to_numpy(dtype=float) if "irrigation_mm" in g else (
            g["irrigation_m3"].to_numpy(dtype=float) / g["area_m2"].to_numpy(dtype=float) * 1000.0)
        d = g["drainage_mm"].to_numpy(dtype=float) if "drainage_mm" in g else (
            g["drainage_m3"].to_numpy(dtype=float) / g["area_m2"].to_numpy(dtype=float) * 1000.0)
        r = g["rain_mm"].to_numpy(dtype=float)
        e = g["et_mm"].to_numpy(dtype=float)
        leftover_block[name] = ir + r - d - e

    pad = data.paddies()
    cyclic = pad["cyclic"].to_numpy(dtype=bool)
    area = pad["area_m2"].to_numpy(dtype=float)
    names = np.asarray(pad["block"])
    lon = pad["lon"].to_numpy(dtype=float)
    lat = pad["lat"].to_numpy(dtype=float)
    w = np.where(cyclic, np.maximum(area, 0.0), 0.0)
    w_sum = float(w.sum()) or 1.0
    idx = np.flatnonzero(cyclic)
    leftover_pix = np.zeros((n_days, len(pad)))
    for i in idx:
        leftover_pix[:, i] = leftover_block[str(names[i])]

    s_pix = np.zeros((n_days, len(pad)))
    lp = np.maximum(leftover_pix, 0.0)
    for t in range(n_days - 1):
        s_pix[t + 1, idx] = (
            a0 + a1 * s_pix[t, idx] + a2 * stay[t] * lp[t, idx] + a3 * et[t] + a4 * rain[t])
    s_mean = (s_pix * w).sum(axis=1) / w_sum
    rmse_pix_vs_district_eq = float(np.sqrt(np.mean((s_mean - s_district) ** 2)))
    rmse_pix_vs_box = float(np.sqrt(np.mean((s_mean - loaded.stores[:, 0]) ** 2)))

    dist = distance_to_canal_m(lon, lat)
    near = np.exp(-dist / 400.0)
    leftover_near = leftover_pix.copy()
    for name in okura.BLOCKS:
        mask = cyclic & (names == name)
        wn = w * mask
        mean_near = float((near * wn).sum() / max(float(wn.sum()), 1e-12))
        scale = near / max(mean_near, 1e-12)
        leftover_near[:, mask] = leftover_pix[:, mask] * scale[mask]
    s_near = np.zeros((n_days, len(pad)))
    lp_n = np.maximum(leftover_near, 0.0)
    for t in range(n_days - 1):
        s_near[t + 1, idx] = (
            a0 + a1 * s_near[t, idx] + a2 * stay[t] * lp_n[t, idx] + a3 * et[t] + a4 * rain[t])
    s_near_mean = (s_near * w).sum(axis=1) / w_sum
    rmse_near_vs_district_eq = float(np.sqrt(np.mean((s_near_mean - s_district) ** 2)))

    stay_vol_pix = float((stay[:, None] * lp * w).sum() / 1000.0)
    lat_vol_pix = float(((1.0 - stay)[:, None] * lp * w).sum() / 1000.0)
    stay_vol_d = float((stay * np.maximum(leftover_d, 0.0) * w_sum).sum() / 1000.0)
    lat_vol_d = float(((1.0 - stay) * np.maximum(leftover_d, 0.0) * w_sum).sum() / 1000.0)
    constructed = construct_ratios(year)
    lfa = (okura.TABLE5_1000M3[year]["LfA_North"] + okura.TABLE5_1000M3[year]["LfA_South"]) * 1000.0
    lfv = okura.TABLE5_1000M3[year]["LFV_pump"] * 1000.0
    dwb = okura.TABLE5_1000M3[year]["dWB"] * 1000.0
    clip_gap = float(np.mean(np.maximum(leftover_d, 0.0)
                             - np.average(lp, axis=1, weights=w)))
    return SimpleNamespace(
        year=int(year), coeffs=eq.coeffs, eq_text=eq.text, eq_r2=eq.r2,
        rmse_eq_vs_box_mm=rmse_eq_vs_box,
        rmse_pixel_vs_district_eq_mm=rmse_pix_vs_district_eq,
        rmse_pixel_vs_box_mm=rmse_pix_vs_box,
        rmse_canal_pixel_vs_district_eq_mm=rmse_near_vs_district_eq,
        s_end_district_mm=float(s_district[-1]),
        s_end_pixel_mm=float(s_mean[-1]),
        s_end_box_mm=float(loaded.stores[-1, 0]),
        stay_m3_pixel=stay_vol_pix, lateral_m3_pixel=lat_vol_pix,
        stay_m3_district=stay_vol_d, lateral_m3_district=lat_vol_d,
        table_dwb_m3=dwb, table_lfv_m3=lfv, table_lfa_m3=lfa,
        qrc_stay_mean=float(stay.mean()),
        table_stay=1.0 - float(constructed["pool_capture_share"]),
        leftover_clip_mm=clip_gap,
        n_cyclic=int(np.count_nonzero(cyclic)),
        area_m2=w_sum,
    )


def spatial_qrc_path(year, folder=None):
    from . import data as _data
    folder = folder or os.path.join(_data.PROJECT, "outputs", "ratios")
    return os.path.join(folder, f"qrc_spatial_{int(year)}.npz")


def fit_spatial_qrc(year, leftover, irrigation, drainage, et, rain, lon, lat, dist,
                    cyclic, area, n_clusters=16, n_qubits=4, layers=2, seed=0,
                    backend="classical", ibm_days=8, shots=2048, stay_target=None,
                    folder=None):
    """QRC leftover shares that depend on parcel (lon, lat) and day.

    Cyclic paddies are clustered in space (O'Malley-scale graph). Each cluster's
    daily leftover/IR/D/ET plus its (x, y) drive the reservoir. Shares are
    written back to every paddy in the cluster. Recapture is uniform on cyclic
    cells (Kx = Ky); season leftover-weighted stay is pinned to Table 5.
    """
    from .canal import pixel_ratio_fields
    from .identify import leftover_weighted_stay
    from .lateral import knn_edges
    from .quantum import _kmeans

    leftover = np.asarray(leftover, dtype=float)
    irrigation = np.asarray(irrigation, dtype=float)
    drainage = np.asarray(drainage, dtype=float)
    et = np.asarray(et, dtype=float)
    rain = np.asarray(rain, dtype=float)
    days, n = leftover.shape
    if rain.ndim == 1:
        rain = np.broadcast_to(rain[:, None], leftover.shape)
    if et.ndim == 1:
        et = np.broadcast_to(et[:, None], leftover.shape)
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    dist = np.asarray(dist, dtype=float)
    cyclic = np.asarray(cyclic, dtype=bool)
    area = np.asarray(area, dtype=float)
    constructed = construct_ratios(year)
    stay_target = (float(constructed["kz_ratio"] + constructed["ground_ratio"])
                   if stay_target is None else float(stay_target))
    nbr_frac = float(constructed.get("neighbour_of_remain", 0.5))
    idx = np.flatnonzero(cyclic)
    if idx.size == 0:
        raise ValueError("no cyclic paddies for spatial QRC")
    k = max(2, min(int(n_clusters), idx.size))
    local, centres = _kmeans(np.column_stack([lon[idx], lat[idx]]), k, seed=seed)
    k = int(centres.shape[0])
    labels = np.full(n, -1, dtype=int)
    labels[idx] = local
    feat_rows = []
    tgt_rows = []
    store_top = []
    series_L, series_stay_et, series_rain = [], [], []
    series_ir, series_drn, series_e = [], [], []
    series_xy = []
    for c in range(k):
        mem = labels == c
        w = np.maximum(area[mem], 1e-12)
        w = w / w.sum()
        L = leftover[:, mem] @ w
        ir = irrigation[:, mem] @ w
        drn = drainage[:, mem] @ w
        e = et[:, mem] @ w
        r = rain[:, mem] @ w
        frame = pd_frame([{"leftover": L[t], "et": e[t], "rain": r[t],
                           "irrigation": ir[t], "drainage": drn[t]} for t in range(days)])
        y, stores = soil_box_ratios(frame, constructed)
        lon_c, lat_c = float(centres[c, 0]), float(centres[c, 1])
        dist_c = float(np.average(dist[mem], weights=area[mem]))
        near_c = float(np.exp(-dist_c / 400.0))
        lag = np.concatenate([[L[0]], L[:-1]])
        season = np.arange(days) / max(days - 1, 1)
        raw = np.column_stack([
            r, ir, drn, e, L, lag, r + ir, season,
            np.full(days, lon_c), np.full(days, lat_c),
            np.full(days, dist_c), np.full(days, near_c),
        ])
        feat_rows.append(raw)
        tgt_rows.append(y)
        store_top.append(stores[:, 0])
        series_L.append(L)
        series_stay_et.append(e)
        series_rain.append(r)
        series_ir.append(ir)
        series_drn.append(drn)
        series_xy.append((lon_c, lat_c, dist_c))
    raw = np.vstack(feat_rows)
    target = np.vstack(tgt_rows)
    mean = raw.mean(axis=0)
    scale = np.maximum(raw.std(axis=0), 1e-6)
    z = (raw - mean) / scale
    r_state = reservoir_statevector(z, n_qubits=n_qubits, layers=layers, seed=seed)
    ibm_meta = SimpleNamespace(backend="statevector", job_id="", rows=np.array([], dtype=int))
    choice = backend
    if choice == "auto":
        choice = "ibm" if _ibm_token() else "classical"
    if choice == "ibm":
        idx = np.linspace(0, len(z) - 1, num=max(int(ibm_days), 1), dtype=int)
        idx = np.unique(idx)
        try:
            z_ibm, name, job_id = reservoir_ibm(
                z, n_qubits=n_qubits, layers=layers, seed=seed, day_index=idx, shots=shots)
            r_state = r_state.copy()
            r_state[idx] = z_ibm
            ibm_meta = SimpleNamespace(backend=name, job_id=job_id, rows=idx, z=z_ibm)
        except Exception as error:
            ibm_meta = SimpleNamespace(
                backend=f"statevector (IBM failed: {error})",
                job_id="", rows=np.array([], dtype=int), z=None)
    weights, pred_flat = fit_readout(r_state, target)
    pred_c = pred_flat.reshape(k, days, 4).transpose(1, 0, 2)
    stay_teacher = target[:, 2] + target[:, 3]
    graw = np.vstack([
        generalist_drive(series_L[c], series_ir[c], series_drn[c], series_stay_et[c],
                         series_rain[c], series_xy[c][0], series_xy[c][1],
                         series_xy[c][2])[:, 0, :]
        for c in range(k)
    ])
    eq_stay = fit_centered_generalist_stay(stay_teacher, graw, stay_target)
    tgt_c = target.reshape(k, days, 4)
    stay_box = [tgt_c[c, :, 2] + tgt_c[c, :, 3] for c in range(k)]
    eq_store = pooled_store_equation(
        store_top, stay_box, series_L, series_stay_et, series_rain)
    p = Physics()
    canal_m = float(p.canal_length_m)
    top = float(p.total_evaporable_mm / max(p.total_evaporable_mm + p.total_available_mm, 1e-9))
    edges = knn_edges(lon, lat, k=4)
    axis = pixel_ratio_fields(n, dist, edges, cyclic, constructed["pool_capture_share"],
                              constructed["farm_to_main"], top, canal_m,
                              neighbour_of_remain=nbr_frac)
    stay_qrc = eval_generalist_stay(
        eq_stay.coeffs, leftover, irrigation, drainage, et, rain, lon, lat, dist, canal_m,
        feat_mean=eq_stay.feat_mean, stay_target=eq_stay.stay_target)
    # LFV pinned from canal geometry; QRC modulates stay within the non-LFV remainder.
    lfv = np.asarray(axis["pool_capture_share"], dtype=float)
    remain = np.maximum(1.0 - lfv, 0.0)
    stay_cap = remain
    stay_field = np.clip(0.65 * stay_qrc + 0.35 * (remain * (1.0 - nbr_frac))[None, :], 0.0, None)
    stay_field = np.minimum(stay_field, stay_cap[None, :])
    stay_field = np.where(cyclic[None, :], stay_field, 0.0)
    neighbour = np.where(cyclic[None, :], remain[None, :] - stay_field, 0.0)
    daily = {name: np.zeros((days, n)) for name in RATIO_NAMES}
    daily["kx_ratio"] = neighbour * 0.5
    daily["ky_ratio"] = neighbour * 0.5
    daily["kz_ratio"] = stay_field * top
    daily["ground_ratio"] = stay_field * (1.0 - top)
    daily["kx_signed"] = daily["kx_ratio"] * axis["kx_sign"]
    daily["ky_signed"] = daily["ky_ratio"] * axis["ky_sign"]
    daily["kx_sign"] = np.broadcast_to(axis["kx_sign"], (days, n)).copy()
    daily["ky_sign"] = np.broadcast_to(axis["ky_sign"], (days, n)).copy()
    daily["farm_to_main"] = np.broadcast_to(axis["farm_to_main"], (days, n)).copy()
    daily["pool_capture_share"] = np.broadcast_to(lfv, (days, n)).copy()
    mean_shares = {name: daily[name].mean(axis=0) for name in
                   (*RATIO_NAMES, "farm_to_main", "pool_capture_share", "kx_signed", "ky_signed",
                    "kx_sign", "ky_sign")}
    leftover_weighted_stay(mean_shares, leftover, area, cyclic, stay_target, top)
    old_stay = daily["kz_ratio"] + daily["ground_ratio"]
    new_stay = mean_shares["kz_ratio"] + mean_shares["ground_ratio"]
    old_m = np.maximum(old_stay.mean(axis=0), 1e-12)
    factor = new_stay / old_m
    stay = np.clip(old_stay * factor, 0.0, None)
    stay = np.minimum(stay, np.maximum(1.0 - daily["pool_capture_share"], 0.0))
    neighbour = np.maximum(1.0 - daily["pool_capture_share"] - stay, 0.0)
    daily["kx_ratio"] = neighbour * 0.5
    daily["ky_ratio"] = neighbour * 0.5
    daily["kz_ratio"] = stay * top
    daily["ground_ratio"] = stay * (1.0 - top)
    daily["kx_signed"] = daily["kx_ratio"] * axis["kx_sign"]
    daily["ky_signed"] = daily["ky_ratio"] * axis["ky_sign"]
    stay_w = float(mean_shares.get("stay_weighted", stay_target))
    for name, val in daily.items():
        arr = np.asarray(val)
        mean_shares[name] = arr.mean(axis=0) if arr.ndim == 2 else arr
    mean_shares["stay_weighted"] = stay_w
    mean_shares["stay_target"] = stay_target
    equation_text = format_generalist(eq_stay, eq_store, stay_target, top, k, ibm_meta.backend)
    result = SimpleNamespace(
        year=int(year), n_clusters=k, labels=labels, daily=daily, shares=mean_shares,
        weights=weights, rmse=float(np.sqrt(np.mean((pred_flat - target) ** 2))),
        stay_weighted=float(mean_shares.get("stay_weighted", stay_target)),
        stay_target=stay_target,
        ibm=ibm_meta, n_qubits=n_qubits, layers=layers,
        eq_stay=eq_stay, eq_store=eq_store, equation_text=equation_text,
        generalist_coeffs=eq_stay.coeffs, store_coeffs=eq_store.coeffs,
    )
    try:
        path = spatial_qrc_path(year, folder)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        extra_mean = ("farm_to_main", "pool_capture_share", "kx_sign", "ky_sign")
        extra_daily = ("kx_signed", "ky_signed", "farm_to_main", "pool_capture_share",
                       "kx_sign", "ky_sign")
        np.savez_compressed(
            path, year=np.int32(year), n_clusters=np.int32(k), labels=labels,
            rmse=np.float64(result.rmse), stay_weighted=np.float64(result.stay_weighted),
            ibm_backend=np.array(str(ibm_meta.backend)),
            ibm_job_id=np.array(str(ibm_meta.job_id)),
            ibm_rows=np.asarray(ibm_meta.rows, dtype=np.int32),
            generalist_coeffs=np.asarray(eq_stay.coeffs, dtype=np.float64),
            generalist_feat_mean=np.asarray(eq_stay.feat_mean, dtype=np.float64),
            stay_target=np.float64(stay_target),
            store_coeffs=np.asarray(eq_store.coeffs, dtype=np.float64),
            equation=np.array(equation_text),
            **{name + "_mean": mean_shares[name] for name in (*RATIO_NAMES, *extra_mean)
               if name in mean_shares},
            **{name + "_daily": daily[name] for name in (*RATIO_NAMES, *extra_daily)
               if name in daily},
        )
        result.path = path
        text_path = os.path.join(os.path.dirname(path), f"qrc_spatial_{int(year)}_equation.txt")
        with open(text_path, "w", encoding="utf-8") as handle:
            handle.write(equation_text)
            handle.write("\n")
        result.equation_path = text_path
    except Exception:
        result.path = None
        result.equation_path = None
    return result


def load_spatial_qrc(year, folder=None):
    path = spatial_qrc_path(year, folder)
    if not os.path.isfile(path):
        return None
    packed = np.load(path, allow_pickle=False)
    daily, shares = {}, {}
    for key in packed.files:
        if key.endswith("_daily"):
            daily[key[: -len("_daily")]] = packed[key]
        elif key.endswith("_mean"):
            shares[key[: -len("_mean")]] = packed[key]
    if not daily or "kx_ratio" not in daily:
        return None
    return SimpleNamespace(
        year=int(packed["year"]) if "year" in packed.files else int(year),
        n_clusters=int(packed["n_clusters"]) if "n_clusters" in packed.files else 0,
        labels=packed["labels"] if "labels" in packed.files else None,
        daily=daily, shares=shares, path=path,
        rmse=float(packed["rmse"]) if "rmse" in packed.files else float("nan"),
        stay_weighted=float(packed["stay_weighted"]) if "stay_weighted" in packed.files else float("nan"),
        ibm_backend=str(packed["ibm_backend"]) if "ibm_backend" in packed.files else "",
        ibm_job_id=str(packed["ibm_job_id"]) if "ibm_job_id" in packed.files else "",
        feat_mean=packed["generalist_feat_mean"] if "generalist_feat_mean" in packed.files else None,
        generalist_coeffs=packed["generalist_coeffs"] if "generalist_coeffs" in packed.files else None,
        stay_target=float(packed["stay_target"]) if "stay_target" in packed.files else None,
    )


def apply_spatial_qrc(model, leftover, irrigation, drainage, et, rain, folder=None,
                      n_clusters=16, backend="classical"):
    """Load qrc_spatial_{year}.npz or fit from leftover + parcel (lon, lat, dist)."""
    leftover = np.asarray(leftover, dtype=float)
    loaded = load_spatial_qrc(model.year, folder)
    if (loaded is not None and np.asarray(loaded.daily["kx_ratio"]).shape[-1] == leftover.shape[1]
            and loaded.feat_mean is not None):
        return loaded
    frame = getattr(model.inputs, "frame", None)
    if frame is None or len(frame) != leftover.shape[1]:
        from . import data as _data
        frame = _data.paddies()
    if len(frame) != leftover.shape[1]:
        return None
    lon = frame["lon"].to_numpy() if hasattr(frame["lon"], "to_numpy") else np.asarray(frame["lon"])
    lat = frame["lat"].to_numpy() if hasattr(frame["lat"], "to_numpy") else np.asarray(frame["lat"])
    return fit_spatial_qrc(
        model.year, leftover, irrigation, drainage, et, rain, lon, lat,
        model.inputs.canal_distance_m, model.inputs.cyclic, model.area,
        n_clusters=n_clusters, folder=folder, backend=backend)


def spatial_leftover_fields(year):
    """Per-paddy daily leftover from Okura block IR/D, no satellite / xarray."""
    import pandas as pd

    from . import data as _data
    from . import imitation
    from . import okura as _okura
    from .identify import match_block_volumes
    from .physics import Physics

    frame = _data.paddies()
    blocks, _, source = imitation.load_observed(year)
    obs_dates = pd.to_datetime(blocks["date"])
    if getattr(obs_dates.dt, "tz", None) is not None:
        obs_dates = obs_dates.dt.tz_convert("UTC").dt.tz_localize(None)
    obs_dates = obs_dates.dt.normalize()
    dates = pd.DatetimeIndex(sorted(obs_dates.unique()))
    n = len(frame)
    days = len(dates)
    area = frame["area_m2"].to_numpy(dtype=float)
    block = frame["block"].to_numpy()
    cyclic = frame["cyclic"].to_numpy(dtype=bool)
    dist = frame["canal_distance_m"].to_numpy(dtype=float)
    lon = frame["lon"].to_numpy(dtype=float)
    lat = frame["lat"].to_numpy(dtype=float)
    near = np.exp(-np.maximum(dist, 0.0) / max(float(Physics().canal_length_m), 1.0))
    drain_score = 0.25 + 0.75 * near
    ir_score = 0.4 + 0.6 / (0.35 + near)
    irrigation = np.zeros((days, n))
    drainage = np.zeros((days, n))
    et = np.zeros((days, n))
    rain = np.zeros((days, n))
    for d, date in enumerate(dates):
        part_all = blocks[obs_dates == pd.Timestamp(date)]
        w = np.maximum(part_all["area_m2"].to_numpy(dtype=float), 1e-9) if len(part_all) else np.array([1.0])
        rain_mm = float(np.average(part_all["rain_mm"].to_numpy(dtype=float), weights=w)) if (
            len(part_all) and "rain_mm" in part_all.columns) else 0.0
        et_mm = float(np.average(part_all["et_mm"].to_numpy(dtype=float), weights=w)) if (
            len(part_all) and "et_mm" in part_all.columns) else 0.0
        rain[d] = rain_mm
        et[d] = et_mm
        ir, dr = {}, {}
        for name in _okura.BLOCKS:
            part = part_all[part_all["block"] == name]
            ir[name] = float(part["irrigation_m3"].sum()) if len(part) else 0.0
            dr[name] = float(part["drainage_m3"].sum()) if len(part) else 0.0
        irrigation[d] = match_block_volumes(ir, area, block, cyclic, weights=area * ir_score)
        drainage[d] = match_block_volumes(dr, area, block, cyclic, weights=area * drain_score)
    year = int(year)
    for name in _okura.BLOCKS:
        sel = (block == name) & cyclic
        if not np.any(sel):
            continue
        have_ir = float((irrigation[:, sel] * area[sel]).sum() / 1000.0)
        have_d = float((drainage[:, sel] * area[sel]).sum() / 1000.0)
        want_ir = float(_okura.TABLE4_MM[(name, year)]["irrigation"] * _okura.cyclic_area_m2(name) / 1000.0)
        want_d = float(_okura.TABLE4_MM[(name, year)]["drainage"] * _okura.cyclic_area_m2(name) / 1000.0)
        if have_ir > 1e-9:
            irrigation[:, sel] *= want_ir / have_ir
        if have_d > 1e-9:
            drainage[:, sel] *= want_d / have_d
    leftover = irrigation + rain - drainage - et
    return SimpleNamespace(
        year=year, source=source, dates=dates,
        leftover=leftover, irrigation=irrigation, drainage=drainage, et=et, rain=rain,
        lon=lon, lat=lat, dist=dist, cyclic=cyclic, area=area, n=n, days=days,
    )


def train_spatial_qrc(year=okura.COMPARISON_YEAR, n_clusters=16, folder=None,
                      backend="classical", ibm_days=8, shots=2048):
    """Fit parcel QRC, then a generalist stay(t,x,y) / S_top equation for any plot."""
    fields = spatial_leftover_fields(year)
    result = fit_spatial_qrc(
        year, fields.leftover, fields.irrigation, fields.drainage, fields.et, fields.rain,
        fields.lon, fields.lat, fields.dist, fields.cyclic, fields.area,
        n_clusters=n_clusters, folder=folder, backend=backend, ibm_days=ibm_days, shots=shots)
    result.source = fields.source
    result.n_paddies = fields.n
    result.n_days = fields.days
    return result


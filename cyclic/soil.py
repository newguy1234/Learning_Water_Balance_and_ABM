"""Model B: per-paddy soil permeability (Kz, Kx, Ky) by ensemble data assimilation.

Model A (`cyclic.et_net`) says where the block irrigation and discharge went. Model B
takes that forcing as given and infers, for every paddy,

    k_z   plow-pan permeability (mm/d): pond → soil, and perched water back up to the
          root zone of a drained paddy (Physics.pan_capillary_rise)
    k_x   east-west lateral permeability (mm/d): perched exchange with E/W neighbours
    k_y   north-south lateral permeability (mm/d): the same with N/S neighbours

Hooghoudt drain flow to the farm drains uses sqrt(k_x k_y) (physics.drain_k_mm_d), and
the paddy's recapture share sends that drain flow to the pool. So a wet paddy's water
can reach a drained neighbour's ET only through k_x / k_y and that neighbour's k_z.

Observations, one season:
    Lf per block   Σ infiltration on cyclic paddies = Okura Table 5 LfA_North / LfA_South
                   (Lf = IR + R − D − ET is the water that went into the ground)
    LFV            Σ recaptured drain flow (capture_i × Hooghoudt lateral) = Table 5 LFV_pump
    unmet ET       every paddy must meet its thermal ET: Landsat SSEBop ETf interpolated
                   between clear passes × daily ET0 (cyclic.landsat.daily_thermal_et) is
                   the ET demand; what its pond and topsoil cannot supply, after rise
                   through the pan and perched water from neighbours (Kx, Ky), is unmet.
                   Observation: each paddy's unmet ET in each of N_WINDOWS windows = 0.
                   et_source="sentinel" instead compares Sentinel-2 Kc×ET0 anomalies,
                   which cannot show water stress (kept for comparison).

Parameterisation (few numbers, so 4,002 paddies cannot be overfitted):

    log K_f,i = μ_f + β_f · features_i + Σ_m a_f,m φ_m(i)        f ∈ {z, x, y}

features: canal distance, ground elevation, North block, cyclic status (standardised);
φ_m: Gaussian bumps on N_PILOT pilot points (k-means of the paddy positions), the
pilot-point device of groundwater calibration (Doherty 2003).

Fit: ES-MDA (Emerick & Reynolds 2013), an iterative ensemble smoother. An ensemble of
parameter vectors is run through cyclic.physics for the season and updated toward the
observations four times with inflated noise. The final ensemble's spread is the
uncertainty on every paddy's K.

Validation:
    synthetic_twin()  a truth drawn from the prior makes its own observations (+ noise);
                      fitting from the prior, is each K field recovered?
    holdout()         fit on 2021, freeze, run 2022 on model A's 2022 forcing: do the
                      fitted K predict 2022's Lf, LFV and ET pattern better than default K?

Physics is the ABM's (bund 250 mm, stress="supply") with pan_capillary_rise on, so the
fitted field means the same thing where the ABM uses it.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np

from . import data, okura
from .physics import Physics, State, advance

SOIL_DIR = os.path.join(data.PROJECT, "outputs", "soil")
FIELD_NAME = "soil_field.npz"

FIELDS = ("k_z", "k_x", "k_y")
FEATURES = ("canal_distance", "elevation", "north", "cyclic")
N_PILOT = 12
N_WINDOWS = 5
BUND_MM = 250.0  # abm SeasonConfig.bund_height_mm

# Prior: centred on the Physics defaults, a factor of ~2 either way for the level.
PRIOR_LOG_MEAN = {"k_z": np.log(9.5), "k_x": np.log(2000.0), "k_y": np.log(2000.0)}
PRIOR_SD = {"level": 0.8, "feature": 0.3, "pilot": 0.4}
LOG_BOUNDS = {"k_z": (np.log(0.5), np.log(200.0)),
              "k_x": (np.log(20.0), np.log(1e5)),
              "k_y": (np.log(20.0), np.log(1e5))}

# Observation errors. Table volumes: 10 %. ET anomaly: 0.5 mm/d per paddy-window,
# inflated by sqrt(n / ET_EFFECTIVE_N) because satellite errors are spatially
# correlated, so 4,002 paddies are not 4,002 independent measurements.
TABLE_REL_SD = 0.10
ET_SD_MM = 0.5
ET_EFFECTIVE_N = 500
# Unmet thermal ET per paddy-window: SSEBop ET is good to ~15 % (Senay et al. 2013).
UNMET_REL_SD = 0.15
UNMET_FLOOR_MM = 2.0

ALPHAS = (28.0 / 3.0, 7.0, 4.0, 2.0)  # ES-MDA inflation, Σ 1/α = 1


def soil_physics():
    return Physics().replace(bund_mm=BUND_MM, pan_capillary_rise=True)


# --- the season a fit runs on -----------------------------------------------------------------

def _forcing_path(year):
    return os.path.join(SOIL_DIR, f"forcing_{int(year)}.npz")


def model_a_forcing(year, rebuild=False):
    """Per-paddy daily irrigation and farmer discharge from model A, run on default soil.

    Cached in outputs/soil/forcing_<year>.npz; `rebuild=True` after retraining model A.
    """
    path = _forcing_path(year)
    if not rebuild and os.path.isfile(path):
        blob = np.load(path, allow_pickle=False)
        return blob["irrigation_mm"], blob["discharge_mm"]
    from .agents import make_agent
    from .model import CyclicModel

    model = CyclicModel(int(year), agents=make_agent("et_net", use_soil_field=0.0),
                        potential="field", forcing="station", record_paddies=True,
                        save_csv=False)
    model.run()
    ir = np.asarray(model.log["irrigation_mm"], dtype=float)
    dis = np.asarray(model.log["farmer_discharge_mm"], dtype=float)
    os.makedirs(SOIL_DIR, exist_ok=True)
    np.savez_compressed(path, irrigation_mm=ir, discharge_mm=dis,
                        paddy_id=np.asarray(model.inputs.paddy_id).astype(str))
    return ir, dis


def _pilot_basis(x_m, y_m, n_pilot=N_PILOT, seed=0):
    """Gaussian bumps on k-means centres of the paddy positions, columns scaled to max 1."""
    rng = np.random.default_rng(seed)
    xy = np.column_stack([x_m, y_m])
    centres = xy[rng.choice(len(xy), n_pilot, replace=False)]
    for _ in range(50):
        label = ((xy[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
        centres = np.array([xy[label == k].mean(0) if np.any(label == k) else centres[k]
                            for k in range(n_pilot)])
    gap = np.sqrt(((centres[:, None] - centres[None]) ** 2).sum(-1))
    np.fill_diagonal(gap, np.inf)
    length = float(np.median(gap.min(1)))
    d2 = ((xy[:, None, :] - centres[None]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / length ** 2), centres, length


def _standardise(values):
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, np.nanmean(values))
    sd = float(values.std())
    return (values - values.mean()) / (sd if sd > 0 else 1.0)


def soil_setup(year, forcing=None, et_source="landsat"):
    """Everything one season's forward runs need, plus that season's observations."""
    from .paddy_export import load_paddy_dataset

    year = int(year)
    inputs = data.season_inputs(year, "station", "pixel", "season")
    ds = load_paddy_dataset(year)
    if not np.array_equal(np.asarray(inputs.paddy_id).astype(str), ds.paddy.values.astype(str)):
        raise ValueError("season inputs and the ABM paddy export disagree on paddy order")
    n = len(inputs.paddy_id)
    ir, dis = forcing if forcing is not None else model_a_forcing(year)
    days = min(len(inputs.dates), ir.shape[0])
    edges = SimpleNamespace(n=n, **{k: ds["edge_" + k].values for k in ("i", "j", "dist_m", "ux", "uy")},
                            **{k: ds["neighbour_" + k].values for k in ("east", "west", "north", "south")})
    cyclic = ds.cyclic.values.astype(bool)
    block = np.asarray(inputs.block)
    area = np.asarray(inputs.area_m2, dtype=float)
    capture = np.where(cyclic, np.clip(ds["share_pool_capture_share"].values.astype(float), 0.0, 1.0), 0.0)

    lon, lat = ds.lon.values.astype(float), ds.lat.values.astype(float)
    x_m = (lon - lon.mean()) * 111_320.0 * np.cos(np.radians(lat.mean()))
    y_m = (lat - lat.mean()) * 110_540.0
    features = np.column_stack([
        _standardise(np.log1p(np.asarray(inputs.canal_distance_m, dtype=float))),
        _standardise(ds.ground_elev_m.values),
        _standardise(block == "North"),
        _standardise(cyclic),
    ])
    pilot, centres, length = _pilot_basis(x_m, y_m)

    et_obs, et_weight = _et_observations(year, inputs, days, n, et_source)
    et_demand = None
    if et_source == "landsat":
        from .landsat import daily_thermal_et

        et_demand = daily_thermal_et(year, inputs.et0_mm, days)

    setup = SimpleNamespace(
        year=year, n=n, days=days, paddy_id=np.asarray(inputs.paddy_id).astype(str),
        rain=np.asarray(inputs.rain_mm, dtype=float)[:days],
        et0=np.asarray(inputs.et0_mm, dtype=float)[:days],
        canopy=np.asarray(inputs.fc_daily, dtype=float)[:days],
        irrigation=np.asarray(ir, dtype=float)[:days], discharge=np.asarray(dis, dtype=float)[:days],
        edges=edges, area=area, block=block, cyclic=cyclic, capture=capture,
        features=features, pilot=pilot, pilot_centres=centres, pilot_length_m=length,
        et_source=et_source, et_weight=et_weight, et_demand=et_demand, physics=soil_physics(),
    )
    setup.et_obs_mm = et_obs
    setup.et_valid = np.isfinite(et_obs)
    setup.d_obs, setup.sd = _observations(setup)
    return setup


def _et_observations(year, inputs, days, n, et_source):
    """Observed ET rows (rows, n; NaN = not observed) and each row's day weights
    (rows, days, n), so simulated ET is compared on exactly the observed days."""
    if et_source == "landsat":
        from .landsat import load_paddy_thermal_et

        thermal = load_paddy_thermal_et(year)
        if thermal is None:
            raise FileNotFoundError(f"no Landsat thermal ET for {year}; run "
                                    f"cyclic.landsat.paddy_thermal_et({year}) first")
        if not np.array_equal(thermal["paddy_id"], np.asarray(inputs.paddy_id).astype(str)):
            raise ValueError("Landsat paddy order differs from the season inputs")
        # Unmet ET summed over each window, every paddy; observed value 0.
        window = np.minimum((np.arange(days) * N_WINDOWS) // days, N_WINDOWS - 1)
        weight = np.zeros((N_WINDOWS, days, n), dtype=np.float32)
        for w in range(N_WINDOWS):
            weight[w, window == w] = 1.0
        return np.zeros((N_WINDOWS, n)), weight
    if et_source == "sentinel":
        sat = np.asarray(inputs.sat_pass_et_mm, dtype=float)[:days]
        passes = np.asarray(inputs.is_pass, dtype=bool)[:days] & np.isfinite(sat)
        window = np.minimum((np.arange(days) * N_WINDOWS) // days, N_WINDOWS - 1)
        weight = np.zeros((N_WINDOWS, days, n), dtype=np.float32)
        for w in range(N_WINDOWS):
            sel = (window == w)[:, None] & passes
            count = sel.sum(0)
            weight[w] = np.where(sel, 1.0 / np.maximum(count, 1), 0.0)
        obs = np.einsum("rdn,dn->rn", weight, np.where(passes, sat, 0.0))
        obs[weight.sum(1) == 0] = np.nan
        return obs, weight
    raise ValueError(f"et_source is 'landsat' or 'sentinel', not {et_source!r}")


def _anomalies(setup, window_means):
    """Each row's paddy values minus that row's area-weighted mean, valid ones only."""
    valid = setup.et_valid
    values = np.where(valid, window_means, 0.0)
    weight = np.where(valid, setup.area[None, :], 0.0)
    mean = (values * weight).sum(1, keepdims=True) / np.maximum(weight.sum(1, keepdims=True), 1e-9)
    return (values - mean)[valid]


def _table_targets(year):
    t5 = okura.TABLE5_1000M3[int(year)]
    return np.array([t5["LfA_North"], t5["LfA_South"], t5["LFV_pump"]], dtype=float) * 1000.0


def _observations(setup):
    tables = _table_targets(setup.year)
    if setup.et_demand is not None:
        # Unmet thermal ET = 0, with an error of UNMET_REL_SD of that window's demand.
        demand = np.einsum("rdn,dn->rn", setup.et_weight, setup.et_demand)[setup.et_valid]
        et = np.zeros(demand.size)
        inflate = np.sqrt(max(et.size, 1) / ET_EFFECTIVE_N)
        et_sd = (UNMET_REL_SD * demand + UNMET_FLOOR_MM) * inflate
    else:
        et = _anomalies(setup, setup.et_obs_mm)
        inflate = np.sqrt(max(et.size, 1) / ET_EFFECTIVE_N)
        et_sd = np.full(et.size, ET_SD_MM * inflate)
    return np.concatenate([tables, et]), np.concatenate([TABLE_REL_SD * tables, et_sd])


# --- parameters → K fields ---------------------------------------------------------------------

def n_params():
    return len(FIELDS) * (1 + len(FEATURES) + N_PILOT)


def prior_mean():
    out = []
    for f in FIELDS:
        out += [PRIOR_LOG_MEAN[f]] + [0.0] * (len(FEATURES) + N_PILOT)
    return np.array(out)


def prior_sd():
    one = [PRIOR_SD["level"]] + [PRIOR_SD["feature"]] * len(FEATURES) + [PRIOR_SD["pilot"]] * N_PILOT
    return np.array(one * len(FIELDS))


def sample_prior(n_ens, seed=0):
    rng = np.random.default_rng(seed)
    return prior_mean() + prior_sd() * rng.standard_normal((n_ens, n_params()))


def log_k_fields(theta, setup):
    """{field: log K per paddy (mm/d)} for one parameter vector, clipped to LOG_BOUNDS."""
    theta = np.asarray(theta, dtype=float)
    per = 1 + len(FEATURES) + N_PILOT
    out = {}
    for k, f in enumerate(FIELDS):
        block = theta[k * per:(k + 1) * per]
        value = block[0] + setup.features @ block[1:1 + len(FEATURES)] + setup.pilot @ block[1 + len(FEATURES):]
        out[f] = np.clip(value, *LOG_BOUNDS[f])
    return out


def k_fields(theta, setup):
    return {f: np.exp(v) for f, v in log_k_fields(theta, setup).items()}


# --- forward model -----------------------------------------------------------------------------

def simulate(setup, k=None, record=False):
    """One season of cyclic.physics with these per-paddy K (None = Physics defaults)."""
    p = setup.physics
    k = k or {}
    state = State.start(setup.n)
    infiltration = np.zeros(setup.n)
    lateral = np.zeros(setup.n)
    et_daily = np.zeros((setup.days, setup.n))
    demand = getattr(setup, "et_demand", None)
    totals = {key: np.zeros(setup.n) for key in
              ("neighbor_out_x_mm", "neighbor_out_y_mm", "neighbor_mm", "rise_mm", "deep_mm",
               "et_mm", "stress_mm", "drainage_mm")} if record else None
    start_storage = state.storage().copy()
    for d in range(setup.days):
        flux = advance(state, setup.rain[d], setup.et0[d], setup.canopy[d],
                       setup.irrigation[d], setup.discharge[d], p,
                       k_z=k.get("k_z"), k_x=k.get("k_x"), k_y=k.get("k_y"),
                       edges=setup.edges, area_m2=setup.area, stress="supply",
                       et_demand_mm=None if demand is None else demand[d])
        infiltration += flux["infiltration_mm"]
        lateral += flux["lateral_mm"]
        et_daily[d] = flux["et_mm"] if demand is None else flux["stress_mm"]
        if record:
            for key in totals:
                totals[key] += flux[key]
    out = SimpleNamespace(infiltration_mm=infiltration, lateral_mm=lateral,
                          et_window_mm=np.einsum("rdn,dn->rn", setup.et_weight, et_daily))
    if record:
        for key, value in totals.items():
            setattr(out, key, value)
        out.storage_change_mm = state.storage() - start_storage
    return out


def predicted(setup, run):
    """Observation vector a season run predicts (same layout as setup.d_obs)."""
    vol = lambda mm, mask: float((mm[mask] * setup.area[mask]).sum() / 1000.0)
    north = setup.cyclic & (setup.block == "North")
    south = setup.cyclic & (setup.block == "South")
    lfv = vol(setup.capture * run.lateral_mm, setup.cyclic)
    tables = np.array([vol(run.infiltration_mm, north), vol(run.infiltration_mm, south), lfv])
    if getattr(setup, "et_demand", None) is not None:
        return np.concatenate([tables, run.et_window_mm[setup.et_valid]])
    return np.concatenate([tables, _anomalies(setup, run.et_window_mm)])


def forward(theta, setup):
    return predicted(setup, simulate(setup, k_fields(theta, setup)))


# --- ES-MDA -------------------------------------------------------------------------------------

def es_mda(setup, ensemble, d_obs=None, sd=None, alphas=ALPHAS, seed=0, progress=None):
    """Iterative ensemble smoother with multiple data assimilation (Emerick & Reynolds 2013).

    Each pass runs every member, then moves each toward perturbed observations by the
    ensemble Kalman gain with noise inflated by α. Solved in ensemble space, so the
    thousands of ET observations cost an (n_ens × n_ens) solve.
    """
    rng = np.random.default_rng(seed)
    d_obs = setup.d_obs if d_obs is None else np.asarray(d_obs, dtype=float)
    sd = setup.sd if sd is None else np.asarray(sd, dtype=float)
    X = np.asarray(ensemble, dtype=float).copy()
    n_ens = X.shape[0]
    history = []
    for step, alpha in enumerate(alphas):
        D = np.stack([forward(x, setup) for x in X])
        history.append(_misfit(D, d_obs, sd))
        if progress:
            progress(f"ES-MDA pass {step + 1}/{len(alphas)}  alpha={alpha:.2f}  "
                     f"chi2/n tables={history[-1]['tables']:.3f} ET={history[-1]['et']:.3f}")
        Xa = (X - X.mean(0)) / np.sqrt(n_ens - 1)
        S = (D - D.mean(0)) / np.sqrt(n_ens - 1) / sd
        perturbed = d_obs + np.sqrt(alpha) * sd * rng.standard_normal(D.shape)
        R = (perturbed - D) / sd
        M = np.linalg.solve(S @ S.T + alpha * np.eye(n_ens), S @ R.T)
        X = X + M.T @ Xa
    D = np.stack([forward(x, setup) for x in X])
    history.append(_misfit(D, d_obs, sd))
    if progress:
        progress(f"ES-MDA final  chi2/n tables={history[-1]['tables']:.3f} ET={history[-1]['et']:.3f}")
    return SimpleNamespace(ensemble=X, predicted=D, misfit=history)


def _misfit(D, d_obs, sd):
    """Ensemble-mean χ²/n of the three table volumes and of the ET anomalies, apart, so
    thousands of ET observations do not hide the table fit."""
    z2 = ((D - d_obs) / sd) ** 2
    return {"tables": float(z2[:, :3].mean()), "et": float(z2[:, 3:].mean())}


def fit(setup, n_ens=200, seed=0, d_obs=None, progress=None):
    prior = sample_prior(n_ens, seed)
    result = es_mda(setup, prior, d_obs=d_obs, seed=seed + 1, progress=progress)
    result.prior = prior
    result.setup = setup
    return result


def field_summary(ensemble, setup):
    """Per-paddy posterior: geometric-mean K, log-sd and 5–95 % log bounds for each field."""
    logs = {f: np.stack([log_k_fields(x, setup)[f] for x in ensemble]) for f in FIELDS}
    out = {}
    for f, v in logs.items():
        out[f] = np.exp(v.mean(0))
        out[f + "_log_sd"] = v.std(0)
        out[f + "_log_p05"] = np.percentile(v, 5, axis=0)
        out[f + "_log_p95"] = np.percentile(v, 95, axis=0)
    return out


# --- what the fitted soil does with the water ---------------------------------------------------

def flow_shares(setup, k):
    """Season fate of the water that entered each paddy's soil, as shares of it.

    In = infiltration + perched water received from neighbours. Out:
        to_pool       recaptured Hooghoudt drain flow (LFV)
        drain_other   drain flow not recaptured (block storage dWB, or the lake off-cycle)
        to_x / to_y   sent to east-west / north-south neighbours
        rise          back up through the pan to this paddy's own ET
        deep          deep percolation
        stored        left in the perched layer at season end (remainder)
    """
    run = simulate(setup, k, record=True)
    inflow = run.neighbor_mm + run.neighbor_out_x_mm + run.neighbor_out_y_mm
    total = np.maximum(run.infiltration_mm + inflow, 1e-9)
    to_pool = setup.capture * run.lateral_mm
    parts = {
        "to_pool": to_pool,
        "drain_other": run.lateral_mm - to_pool,
        "to_x": run.neighbor_out_x_mm,
        "to_y": run.neighbor_out_y_mm,
        "rise": run.rise_mm,
        "deep": run.deep_mm,
    }
    shares = {key: value / total for key, value in parts.items()}
    shares["stored"] = 1.0 - sum(shares.values())
    shares["in_mm"] = total
    shares["stress_mm"] = run.stress_mm
    return shares


# --- validation ---------------------------------------------------------------------------------

def _score(setup, D):
    """Table relative errors and ET-anomaly skill of a predicted observation vector."""
    D = np.atleast_2d(D)
    tables = _table_targets(setup.year)
    rel = (D[:, :3] - tables) / tables
    et_obs = setup.d_obs[3:]
    et_sim = D[:, 3:]
    rmse = np.sqrt(((et_sim - et_obs) ** 2).mean(1))
    corr = np.array([np.corrcoef(row, et_obs)[0, 1] if row.std() > 0 else np.nan for row in et_sim])
    return {
        "lf_north_rel_err": rel[:, 0], "lf_south_rel_err": rel[:, 1], "lfv_rel_err": rel[:, 2],
        "et_anomaly_rmse_mm": rmse, "et_anomaly_corr": corr,
        "et_anomaly_obs_sd_mm": float(et_obs.std()),
    }


def _summarise(score):
    out = {}
    for key, value in score.items():
        if np.ndim(value):
            out[key] = {"mean": float(np.nanmean(value)), "p05": float(np.nanpercentile(value, 5)),
                        "p95": float(np.nanpercentile(value, 95))}
        else:
            out[key] = float(value)
    return out


def synthetic_twin(year=2021, n_ens=200, truth_seed=12345, seed=0, progress=None):
    """Draw a true K from the prior, observe it through the physics with noise, fit, compare."""
    setup = soil_setup(year)
    truth = sample_prior(1, truth_seed)[0]
    rng = np.random.default_rng(truth_seed + 1)
    d_true = forward(truth, setup)
    d_syn = d_true + setup.sd * rng.standard_normal(d_true.size)
    result = fit(setup, n_ens=n_ens, seed=seed, d_obs=d_syn, progress=progress)

    true_log = log_k_fields(truth, setup)
    report = {"year": int(year), "n_ens": int(n_ens), "misfit": result.misfit, "fields": {}}
    for which, ens in (("prior", result.prior), ("posterior", result.ensemble)):
        logs = {f: np.stack([log_k_fields(x, setup)[f] for x in ens]) for f in FIELDS}
        for f in FIELDS:
            mean = logs[f].mean(0)
            lo, hi = np.percentile(logs[f], 5, axis=0), np.percentile(logs[f], 95, axis=0)
            entry = report["fields"].setdefault(f, {})
            entry[which] = {
                "rmse_log": float(np.sqrt(((mean - true_log[f]) ** 2).mean())),
                "corr_spatial": float(np.corrcoef(mean, true_log[f])[0, 1]) if mean.std() > 0 else float("nan"),
                "level_err_log": float(mean.mean() - true_log[f].mean()),
                "coverage_90": float(((true_log[f] >= lo) & (true_log[f] <= hi)).mean()),
                "mean_log_sd": float(logs[f].std(0).mean()),
            }
        aniso_true = true_log["k_x"] - true_log["k_y"]
        aniso = (logs["k_x"] - logs["k_y"]).mean(0)
        report.setdefault("anisotropy", {})[which] = {
            "rmse_log": float(np.sqrt(((aniso - aniso_true) ** 2).mean())),
            "corr_spatial": float(np.corrcoef(aniso, aniso_true)[0, 1]) if aniso.std() > 0 else float("nan"),
        }
    report["parameters"] = {
        "truth": truth.tolist(),
        "posterior_mean": result.ensemble.mean(0).tolist(),
        "posterior_sd": result.ensemble.std(0).tolist(),
        "prior_sd": prior_sd().tolist(),
        "sd_reduction": (1.0 - result.ensemble.std(0) / prior_sd()).tolist(),
    }
    report["obs_fit"] = {"posterior": _summarise(_score_against(setup, result.predicted, d_syn))}
    return SimpleNamespace(report=report, result=result, truth=truth, setup=setup)


def _score_against(setup, D, d_obs):
    saved = setup.d_obs
    setup.d_obs = d_obs
    try:
        tables = _table_targets(setup.year)
        D = np.atleast_2d(D)
        rel = (D[:, :3] - d_obs[:3]) / tables
        et = D[:, 3:]
        return {"lf_north_rel_err": rel[:, 0], "lf_south_rel_err": rel[:, 1], "lfv_rel_err": rel[:, 2],
                "et_anomaly_rmse_mm": np.sqrt(((et - d_obs[3:]) ** 2).mean(1))}
    finally:
        setup.d_obs = saved


def holdout(train_year=2021, eval_year=2022, n_ens=200, seed=0, progress=None):
    """Fit on one season, freeze the K fields, predict the other season."""
    train = soil_setup(train_year)
    result = fit(train, n_ens=n_ens, seed=seed, progress=progress)
    evaluate = soil_setup(eval_year)
    if not np.array_equal(train.paddy_id, evaluate.paddy_id):
        raise ValueError("train and eval seasons have different paddies")
    default = prior_mean()
    report = {"train_year": int(train_year), "eval_year": int(eval_year), "n_ens": int(n_ens),
              "misfit_train": result.misfit}
    for name, setup in (("train", train), ("eval", evaluate)):
        if progress:
            progress(f"scoring {name} season {setup.year}")
        post = np.stack([forward(x, setup) for x in result.ensemble])
        prior = np.stack([forward(x, setup) for x in result.prior])
        base = forward(default, setup)
        physics_default = predicted(setup, simulate(setup, None))
        report[name] = {
            "posterior": _summarise(_score(setup, post)),
            "posterior_mean_k": _summarise(_score(setup, forward(result.ensemble.mean(0), setup))),
            "prior": _summarise(_score(setup, prior)),
            "default_k": _summarise(_score(setup, base)),
            "physics_default_uniform": _summarise(_score(setup, physics_default)),
        }
    return SimpleNamespace(report=report, result=result, train=train, evaluate=evaluate)


# --- saving and using a fitted field ------------------------------------------------------------

def field_path():
    return os.path.join(SOIL_DIR, FIELD_NAME)


def save_field(result, setup, path=None, extra=None):
    path = path or field_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    summary = field_summary(result.ensemble, setup)
    mean_k = {f: summary[f] for f in FIELDS}
    shares = flow_shares(setup, mean_k)
    meta = {"train_year": int(setup.year), "n_ens": int(result.ensemble.shape[0]),
            "features": list(FEATURES), "n_pilot": N_PILOT, "alphas": list(ALPHAS),
            "physics": {"bund_mm": BUND_MM, "pan_capillary_rise": True, "stress": "supply"}}
    meta.update(extra or {})
    np.savez_compressed(
        path, paddy_id=setup.paddy_id, ensemble=result.ensemble,
        **{key: np.asarray(value, dtype=float) for key, value in summary.items()},
        **{"share_" + key: np.asarray(value, dtype=float) for key, value in shares.items()},
        meta=json.dumps(meta))
    return path


def load_field(path=None):
    path = path or field_path()
    if not os.path.isfile(path):
        return None
    blob = np.load(path, allow_pickle=False)
    out = {key: blob[key] for key in blob.files if key != "meta"}
    out["meta"] = json.loads(str(blob["meta"]))
    return out


def soil_k_for(paddy_id, path=None):
    """(k_z, k_x, k_y) per paddy in this order, or None if no matching field is saved."""
    field = load_field(path)
    if field is None:
        return None
    if not np.array_equal(np.asarray(field["paddy_id"]).astype(str), np.asarray(paddy_id).astype(str)):
        return None
    return tuple(np.asarray(field[f], dtype=float) for f in FIELDS)


def apply_soil_field(model, path=None):
    """Put the fitted per-paddy K on a CyclicModel (and turn pan rise on). False if none."""
    k = soil_k_for(model.inputs.paddy_id, path)
    if k is None:
        return False
    model.k_z, model.k_x, model.k_y = k
    model.physics = model.physics.replace(pan_capillary_rise=True)
    return True

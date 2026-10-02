# -*- coding: utf-8 -*-
"""Does each personality trait and weather scenario move the simulated demand, and the
way it should? A batch of paired season runs plus the statistics to read them.

Design (see the references at the end of this docstring):

  noise floor      the default farmer mix on baseline weather, once per seed. The spread
                   across seeds is the model's own randomness (who owns what, each
                   farmer's drawn personality, day-off schedules).
  trait sweep      extended one-factor-at-a-time: for each trait, every farmer gets the
                   trait's low end, then its high end, with everything else as drawn.
                   Trait draws use a fixed order (abm.farmer), so pinning one trait
                   leaves every other trait, the ownership and the schedules identical.
                   The effect is the paired difference high − low for the same seed
                   (common random numbers), so seed-to-seed noise cancels.
  weather          each stem.weather preset vs baseline weather, paired by seed.
  pump shortage    the pumping station limited to SHORTAGE_PUMP_FRACTION of its MAFF
                   capacity, so the canal cannot meet demand and crops go short of
                   water. At full capacity no paddy is ever stressed, which leaves the
                   stress-driven traits (experience) and cooperation's scarcity cut with
                   nothing to react to; the trait sweep is repeated here to test them.
  interactions     optional: the trait sweep repeated under every weather preset, to
                   see whether a trait matters more in a drought than a wet year.

A run's ``scenario`` is its weather, plus " + pump shortage" when supply is limited.
Each trait carries an expected sign on some outputs, read off the decision rules in
abm.farmer (e.g. refill_overshoot raises the refill level, so it should raise demand).
A verdict compares the paired effect against that expectation with a t-test across
seeds.

References
  ten Broeke, van Voorn & Ligtenberg 2016, JASSS 19(1):5 — extended OAT first
  Lee et al. 2015, JASSS 18(4):4 — replicates, variance stability, comparing variants
  Seri & Secchi 2017 — power analysis for the number of runs
  Grimm et al. 2005, Science 310:987 — patterns as tests of structural realism
  Brown et al. 2012 — decision scaling / climate stress test of a water system

    from abm import sensitivity
    plan = sensitivity.build_plan(seeds=[0, 1, 2])
    runs, daily = sensitivity.run_plan(plan, workers=8)
    sensitivity.trait_effects(runs)
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .farmer import TRAIT_RANGE, TRAITS

# Season outputs a demand generator is judged on. (column, label, unit)
METRICS: tuple[tuple[str, str, str], ...] = (
    ("request_m3", "Season demand (requested)", "m³"),
    ("applied_m3", "Season irrigation applied", "m³"),
    ("unmet_m3", "Demand not supplied", "m³"),
    ("drainage_m3", "Season drainage", "m³"),
    ("peak7_request_m3", "Peak 7-day demand", "m³"),
    ("peak_day", "Day of peak 7-day demand", "day"),
    ("withdraw_m3", "Station lake withdrawal", "m³"),
    ("mean_ponding_mm", "Mean ponding depth", "mm"),
    ("stress_mm", "Season ET shortfall", "mm"),
    ("relative_yield", "Relative yield", "–"),
    ("WUE_prime_pct", "WUE′ (Okura Eq. 11)", "%"),
)
METRIC_LABEL = {k: label for k, label, _ in METRICS}
METRIC_UNIT = {k: unit for k, _, unit in METRICS}

# Pump shortage: share of the station's pump capacity (SeasonConfig.pump_capacity_m3_s,
# MAFF 3.304 m3/s). At 0.2 a drought season loses ~30 % of yield; 0.25 is a clear but
# not total shortage.
SHORTAGE_PUMP_FRACTION = .25
SHORTAGE = " + pump shortage"

# Expected sign of (high − low) for each trait, read off the rules in abm.farmer; they
# apply in every scenario. Missing = no prior (the run reports what the trait did).
EXPECTED: dict[str, dict[str, int]] = {
    "risk_tolerance": {"request_m3": -1, "mean_ponding_mm": -1},      # drops the risk buffer
    "drawdown_tolerance": {"mean_ponding_mm": -1},                    # lower refill trigger
    "refill_overshoot": {"request_m3": +1},                           # higher refill level
    "flow_through_habit": {"request_m3": +1, "drainage_m3": +1},      # inlet left running
    "forecast_reliance": {"request_m3": -1},                          # leaves room for rain
    "neighbour_wetness_bias": {"mean_ponding_mm": +1},                # sits wetter than peers
    "calendar_lead": {"peak_day": -1},                                # calendar runs early
}
# Extra expectations that only hold when the canal is short (crops stressed).
EXPECTED_SHORTAGE: dict[str, dict[str, int]] = {
    # Experienced farmers raise their target less after a stressed day, so ask for less.
    "experience": {"request_m3": -1},
    # Cooperative farmers trim their refill when the canal cannot supply everyone.
    "cooperation": {"request_m3": -1},
}
# Expected sign of (scenario − baseline), default farmer mix.
EXPECTED_WEATHER: dict[str, dict[str, int]] = {
    "drought": {"request_m3": +1},
    "wet": {"request_m3": -1, "drainage_m3": +1},
    "heatwave": {"request_m3": +1},
    "dry_july": {"request_m3": +1},
    "storm": {"drainage_m3": +1},
    "baseline" + SHORTAGE: {"unmet_m3": +1, "stress_mm": +1, "relative_yield": -1},
}


def expected(trait: str, metric: str, scenario: str):
    if scenario.endswith(SHORTAGE) and metric in EXPECTED_SHORTAGE.get(trait, {}):
        return EXPECTED_SHORTAGE[trait][metric]
    return EXPECTED.get(trait, {}).get(metric)


@dataclass(frozen=True)
class RunSpec:
    """One season. ``trait``/``value`` pin that trait for every farmer (None = as drawn)."""
    seed: int
    experiment: str = "noise"          # noise | trait | weather | shortage | interaction
    weather: str = "baseline"
    trait: str | None = None
    level: str = "base"                # base | low | high
    value: float | None = None
    year: int = 2022
    days: int | None = None
    paddies_per_farmer: int = 50
    pump_fraction: float = 1.0         # share of the station's pump capacity

    @property
    def scenario(self) -> str:
        return self.weather + (SHORTAGE if self.pump_fraction < 1 else "")

    @property
    def run_id(self) -> str:
        parts = [self.experiment, self.scenario, self.trait or "-", self.level, f"s{self.seed}"]
        return "|".join(parts)


def build_plan(seeds=(0, 1, 2), traits=None, weathers=None, interactions: bool = False,
               shortage: bool = True, **common) -> list[RunSpec]:
    """Every run the sensitivity tests need; duplicates (the shared baselines) dropped."""
    from stem.weather import list_weather_scenarios

    traits = list(traits or [t.name for t in TRAITS])
    weathers = list(weathers or list_weather_scenarios())
    if "baseline" not in weathers:
        weathers = ["baseline"] + weathers
    specs: list[RunSpec] = []

    def sweep(seed, kind, weather, **extra):
        for name in traits:
            low, high = TRAIT_RANGE[name]
            specs.append(RunSpec(seed, kind, weather, name, "low", float(low), **common, **extra))
            specs.append(RunSpec(seed, kind, weather, name, "high", float(high), **common, **extra))

    for seed in seeds:
        for w in weathers:
            specs.append(RunSpec(seed, "weather" if w != "baseline" else "noise", w, **common))
        for w in (weathers if interactions else ["baseline"]):
            sweep(seed, "trait" if w == "baseline" else "interaction", w)
        if shortage:
            specs.append(RunSpec(seed, "shortage", "baseline", **common, pump_fraction=SHORTAGE_PUMP_FRACTION))
            sweep(seed, "shortage", "baseline", pump_fraction=SHORTAGE_PUMP_FRACTION)
    seen, out = set(), []
    for s in specs:
        if s.run_id not in seen:
            seen.add(s.run_id)
            out.append(s)
    return out


def _build_model(spec: RunSpec):
    from dataclasses import fields

    from .agents import DEFAULT_ROSTER
    from .model import DistrictModel
    from .world import SeasonConfig

    pump = next(f.default for f in fields(SeasonConfig) if f.name == "pump_capacity_m3_s")
    if pump is not None and spec.pump_fraction < 1:
        pump = float(pump) * spec.pump_fraction
    cfg = SeasonConfig(year=spec.year, days=spec.days, seed=spec.seed, ownership_seed=spec.seed,
                       paddies_per_farmer=spec.paddies_per_farmer,
                       weather=spec.weather, pump_capacity_m3_s=pump)
    model = DistrictModel(cfg, agent_roster=list(DEFAULT_ROSTER), seed=spec.seed)
    if spec.trait is not None:
        for fid, agent in list(model.farmer_agents.items()):
            # Same seed, same type: only the pinned trait differs from the base run.
            model.set_farmer_agent(fid, agent.policy.name, **{spec.trait: spec.value})
    return model


def run_one(spec: RunSpec) -> tuple[dict, pd.DataFrame]:
    """Roll one season; return (one summary row, per-day district totals)."""
    from .metrics import season_efficiencies

    t0 = time.time()
    model = _build_model(spec)
    while not model.done():
        model.step()
    w = model.world
    area = w.area_m2
    m3 = lambda mm: float(np.dot(np.asarray(mm, dtype=float), area) / 1000.0)
    rows = []
    for s in w.history:
        req = s.requested_mm if s.requested_mm is not None else s.irrigation_mm
        rows.append(dict(
            day=s.day, rain_mm=float(s.rain_mm), et0_mm=float(s.et0_mm),
            request_m3=m3(req), applied_m3=m3(s.irrigation_mm), sent_m3=m3(s.irrigation_sent_mm),
            unmet_m3=m3(np.maximum(np.asarray(req) - s.irrigation_sent_mm, 0.)),
            drainage_m3=m3(s.drainage_mm), withdraw_m3=float(s.withdraw_m3),
            mean_ponding_mm=float(np.average(s.ponding_mm, weights=area)),
            stress_mm=float(np.average(s.stress_mm, weights=area)),
        ))
    daily = pd.DataFrame(rows)
    roll = daily["request_m3"].rolling(7, min_periods=1).sum()
    eff = season_efficiencies(w)
    summary = dict(asdict(spec), run_id=spec.run_id, scenario=spec.scenario,
                   request_m3=daily["request_m3"].sum(), applied_m3=daily["applied_m3"].sum(),
                   unmet_m3=daily["unmet_m3"].sum(), drainage_m3=daily["drainage_m3"].sum(),
                   withdraw_m3=daily["withdraw_m3"].sum(),
                   peak7_request_m3=float(roll.max()), peak_day=int(daily["day"].iloc[int(roll.values.argmax())]),
                   mean_ponding_mm=daily["mean_ponding_mm"].mean(), stress_mm=daily["stress_mm"].sum(),
                   relative_yield=float(np.average(w.relative_yield, weights=area)),
                   WUE_prime_pct=float(eff.get("WUE_prime_pct", np.nan)),
                   rain_total_mm=daily["rain_mm"].sum(), et0_total_mm=daily["et0_mm"].sum(),
                   n_paddies=int(w.n), n_farmers=int(w.n_farmers()), seconds=time.time() - t0)
    daily.insert(0, "run_id", spec.run_id)
    _release(model)
    return summary, daily


def _release(model) -> None:
    """Free a finished season. Mesa keeps a class-level id counter per model
    (``mesa.Agent._ids``, keyed by the model object) that is never cleared, so without
    this every season a worker runs stays in memory (~150 MB each on the full map)."""
    import gc

    import mesa

    getattr(mesa.Agent, "_ids", {}).pop(model, None)
    gc.collect()


# Memory one worker needs for a full-map season (the 4,002-paddy district): measured
# ~350 MB peak with one BLAS thread and the season released afterwards, plus headroom.
WORKER_MEMORY_GB = 0.5

# Each worker is one season on one core. Left at their default, numpy's BLAS libraries
# start a thread per core in every worker and reserve ~370 MB for them up front, which
# both oversubscribes the CPU and multiplies memory by the worker count.
SINGLE_THREAD_ENV = dict(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def default_workers() -> int:
    """Every core, but no more than free RAM / WORKER_MEMORY_GB: other programs, or a
    second batch running at the same time, would otherwise run the machine out of memory
    and Windows kills a worker (BrokenProcessPool)."""
    by_cpu = max(1, os.cpu_count() or 1)
    try:
        import psutil

        free_gb = psutil.virtual_memory().available / 1e9
        return max(1, min(by_cpu, int(free_gb // WORKER_MEMORY_GB)))
    except ImportError:
        return by_cpu


def code_version() -> str:
    """Hash of the abm and cyclic sources: a cached season is reused only while the code
    that produced it is unchanged."""
    import glob
    import hashlib

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha1()
    for path in sorted(glob.glob(os.path.join(root, "abm", "*.py")) + glob.glob(os.path.join(root, "cyclic", "*.py"))):
        if os.path.basename(path) != "sensitivity_ui.py":
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:12]


def _cache_path(spec: RunSpec, cache_dir: str, version: str) -> str:
    import hashlib

    key = hashlib.sha1(json.dumps(dict(asdict(spec), code=version), sort_keys=True).encode()).hexdigest()[:20]
    return os.path.join(cache_dir, f"{key}.pkl")


def _run_cached(spec: RunSpec, cache_dir: str | None, version: str) -> tuple[dict, pd.DataFrame]:
    """Worker entry: run one season and, with a cache, save it the moment it finishes, so
    a batch that dies (out of memory, Ctrl+C, a crash) keeps every season already done."""
    import pickle

    result = run_one(spec)
    if cache_dir:
        path = _cache_path(spec, cache_dir, version)
        with open(path + ".tmp", "wb") as f:
            pickle.dump(result, f)
        os.replace(path + ".tmp", path)
    return result


def run_plan(specs: list[RunSpec], workers: int | None = None, progress=print,
             cache_dir: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every spec (in parallel when workers > 1).

    ``cache_dir``: each finished season is saved there and reused by a later call with
    the same spec and unchanged code, so an interrupted batch resumes where it stopped.
    If a worker runs out of memory or dies, the unfinished seasons run again with half
    the workers, down to one at a time, instead of losing the batch."""
    import pickle
    from concurrent.futures.process import BrokenProcessPool

    for key, value in SINGLE_THREAD_ENV.items():
        os.environ.setdefault(key, value)   # inherited by the worker processes
    workers = max(1, int(workers or default_workers()))
    version = code_version()
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    results: dict[str, tuple[dict, pd.DataFrame]] = {}
    t0 = time.time()
    fresh = 0

    def done(spec, result, cached=False):
        nonlocal fresh
        results[spec.run_id] = result
        fresh += not cached
        if progress and not cached:
            i, n_left = len(results), len(specs) - len(results)
            left = (time.time() - t0) / max(fresh, 1) * n_left
            progress(f"[{i}/{len(specs)}] {spec.run_id}  ({result[0]['seconds']:.0f}s, ~{left / 60:.1f} min left)")

    def load_cached(pending):
        still = []
        for spec in pending:
            path = _cache_path(spec, cache_dir, version) if cache_dir else None
            if path and os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        done(spec, pickle.load(f), cached=True)
                    continue
                except Exception:
                    os.remove(path)
            if spec.run_id not in results:
                still.append(spec)
        return still

    pending = load_cached(list(specs))
    if progress:
        if len(pending) < len(specs):
            progress(f"{len(specs) - len(pending)} seasons reused from {cache_dir}")
        progress(f"{len(pending)} to run with {workers} worker process{'es' if workers > 1 else ''}")
    while pending:
        if workers == 1:
            for spec in pending:
                done(spec, _run_cached(spec, cache_dir, version))
            break
        pool = ProcessPoolExecutor(max_workers=workers)
        try:
            futures = {pool.submit(_run_cached, s, cache_dir, version): s for s in pending}
            for fut in as_completed(futures):
                done(futures[fut], fut.result())
            pool.shutdown()
            pending = []
        except (BrokenProcessPool, MemoryError) as error:
            pool.shutdown(wait=True, cancel_futures=True)
            pending = load_cached([s for s in pending if s.run_id not in results])
            workers = max(1, workers // 2)
            if progress:
                progress(f"{type(error).__name__}: a worker ran out of memory or died. Rerunning the "
                         f"{len(pending)} unfinished seasons with {workers} worker{'s' if workers > 1 else ''}.")
    summaries = [results[s.run_id][0] for s in specs]
    dailies = [results[s.run_id][1] for s in specs]
    runs = pd.DataFrame(summaries).sort_values(["experiment", "scenario", "trait", "level", "seed"],
                                               na_position="first").reset_index(drop=True)
    return runs, pd.concat(dailies, ignore_index=True)


# --- analysis ------------------------------------------------------------------------------

def _paired(diff: np.ndarray) -> dict:
    from scipy import stats

    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = diff.size
    mean = float(diff.mean()) if n else np.nan
    sd = float(diff.std(ddof=1)) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan
    if n > 1 and sd > 0:
        t = mean / se
        p = float(2 * stats.t.sf(abs(t), n - 1))
        half = float(stats.t.ppf(.975, n - 1) * se)
    elif n > 1:  # identical difference every seed: deterministic
        t, p, half = (np.inf if mean != 0 else 0.), (0. if mean != 0 else 1.), 0.
    else:
        t, p, half = np.nan, np.nan, np.nan
    return dict(n=n, diff=mean, diff_sd=sd, ci95=half, t=t, p=p)


def _verdict(sign, diff, p, rel, alpha=.05, material_pct=1.) -> str:
    detected = np.isfinite(p) and p < alpha and abs(rel) >= material_pct
    if sign is None:
        return "effect (no prior)" if detected else "no effect (no prior)"
    if sign == 0:
        return "LEAK: unexpected effect" if detected else "pass: no effect, as expected"
    if not detected:
        return "FAIL: no effect detected"
    return "pass: expected direction" if np.sign(diff) == sign else "FAIL: opposite direction"


def _relative_pct(diff: float, ref: float) -> float:
    """Effect as % of the base value. From a base of exactly 0 (no ET shortfall at full
    supply, say) any change is infinitely large in relative terms, not undefined."""
    if ref:
        return 100 * diff / abs(ref)
    return float(np.sign(diff) * np.inf) if np.isfinite(diff) and diff else 0.


def _scenarios(runs: pd.DataFrame) -> pd.Series:
    """Scenario label per run (older batches have no scenario column: weather only)."""
    return runs["scenario"] if "scenario" in runs else runs["weather"]


def _noise_sd(runs: pd.DataFrame, scenario: str, metric: str) -> float:
    base = runs[(runs.trait.isna()) & (_scenarios(runs) == scenario)][metric]
    return float(base.std(ddof=1)) if len(base) > 1 else np.nan


def trait_effects(runs: pd.DataFrame, metrics=None, alpha=.05, material_pct=1.) -> pd.DataFrame:
    """Paired high − low effect of every trait on every metric, per scenario."""
    metrics = list(metrics or [k for k, _, _ in METRICS])
    runs = runs.assign(scenario=_scenarios(runs))
    rows = []
    sweep = runs[runs.trait.notna()]
    for (scenario, trait), grp in sweep.groupby(["scenario", "trait"]):
        lo = grp[grp.level == "low"].set_index("seed")
        hi = grp[grp.level == "high"].set_index("seed")
        base = runs[(runs.trait.isna()) & (runs.scenario == scenario)].set_index("seed")
        seeds = lo.index.intersection(hi.index)
        for metric in metrics:
            st = _paired(hi.loc[seeds, metric].values - lo.loc[seeds, metric].values)
            ref = float(base[metric].mean()) if len(base) else float(lo.loc[seeds, metric].mean())
            rel = _relative_pct(st["diff"], ref)
            exp = expected(trait, metric, scenario)
            noise = _noise_sd(runs, scenario, metric)
            rows.append(dict(scenario=scenario, trait=trait, metric=metric,
                             low=float(lo.loc[seeds, metric].mean()), base=ref,
                             high=float(hi.loc[seeds, metric].mean()), **st, rel_pct=rel,
                             noise_sd=noise, signal_to_noise=abs(st["diff"]) / noise if noise else np.nan,
                             expected=exp, verdict=_verdict(exp, st["diff"], st["p"], rel, alpha, material_pct)))
    return pd.DataFrame(rows)


def weather_effects(runs: pd.DataFrame, metrics=None, alpha=.05, material_pct=1.) -> pd.DataFrame:
    """Paired (scenario − baseline) effect on the default farmer mix: every weather preset,
    and the pump shortage."""
    metrics = list(metrics or [k for k, _, _ in METRICS])
    runs = runs.assign(scenario=_scenarios(runs))
    base = runs[(runs.trait.isna()) & (runs.scenario == "baseline")].set_index("seed")
    rows = []
    for scenario, grp in runs[(runs.trait.isna()) & (runs.scenario != "baseline")].groupby("scenario"):
        g = grp.set_index("seed")
        seeds = g.index.intersection(base.index)
        for metric in metrics:
            st = _paired(g.loc[seeds, metric].values - base.loc[seeds, metric].values)
            ref = float(base.loc[seeds, metric].mean())
            rel = _relative_pct(st["diff"], ref)
            exp = EXPECTED_WEATHER.get(scenario, {}).get(metric)
            rows.append(dict(scenario=scenario, metric=metric, base=ref,
                             value=float(g.loc[seeds, metric].mean()), **st, rel_pct=rel,
                             rain_total_mm=float(g["rain_total_mm"].mean()),
                             et0_total_mm=float(g["et0_total_mm"].mean()),
                             expected=exp, verdict=_verdict(exp, st["diff"], st["p"], rel, alpha, material_pct)))
    return pd.DataFrame(rows)


def noise_table(runs: pd.DataFrame, effects: pd.DataFrame | None = None, power=.8, alpha=.05) -> pd.DataFrame:
    """Seed-to-seed spread of the base runs, and the paired-design runs (seeds) needed to
    detect a 1 % and a 5 % change at the given power (normal approximation)."""
    from scipy import stats

    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    base = runs[(runs.trait.isna()) & (_scenarios(runs) == "baseline")]
    rows = []
    for metric, _, _ in METRICS:
        v = base[metric].astype(float)
        mean, sd = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else np.nan
        row = dict(metric=metric, mean=mean, sd=sd, cv_pct=100 * sd / abs(mean) if mean else np.nan)
        # Paired designs care about the sd of the difference; use the median across traits.
        if effects is not None and not effects.empty:
            dsd = effects[(effects.metric == metric) & (effects.scenario == "baseline")]["diff_sd"]
            row["paired_diff_sd"] = float(np.nanmedian(dsd)) if dsd.notna().any() else np.nan
        sd_use = row.get("paired_diff_sd", sd)
        for pct in (1, 5):
            delta = abs(mean) * pct / 100
            row[f"seeds_for_{pct}pct"] = int(np.ceil((z * sd_use / delta) ** 2)) if delta and np.isfinite(sd_use) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --- persistence ---------------------------------------------------------------------------

RESULTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "sensitivity")


def save(runs: pd.DataFrame, daily: pd.DataFrame, meta: dict, root: str = RESULTS_ROOT) -> str:
    out = os.path.join(root, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    runs.to_csv(os.path.join(out, "runs.csv"), index=False)
    daily.to_csv(os.path.join(out, "daily.csv"), index=False)
    effects = trait_effects(runs)
    effects.to_csv(os.path.join(out, "trait_effects.csv"), index=False)
    weather_effects(runs).to_csv(os.path.join(out, "weather_effects.csv"), index=False)
    noise_table(runs, effects).to_csv(os.path.join(out, "noise.csv"), index=False)
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    with open(os.path.join(root, "LATEST"), "w", encoding="utf-8") as f:
        f.write(out)
    return out


def latest(root: str = RESULTS_ROOT) -> str | None:
    try:
        with open(os.path.join(root, "LATEST"), encoding="utf-8") as f:
            path = f.read().strip()
        return path if os.path.isdir(path) else None
    except OSError:
        return None


def load(path: str) -> dict:
    """runs, daily, trait_effects, weather_effects, noise (DataFrames) and meta (dict)."""
    out = {name: pd.read_csv(os.path.join(path, f"{name}.csv"))
           for name in ("runs", "daily", "trait_effects", "weather_effects", "noise")}
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
        out["meta"] = json.load(f)
    out["path"] = path
    return out

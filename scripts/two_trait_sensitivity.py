"""District screen of the two farmer traits: cooperation and work pattern.

4,002 paddies, 131 days of 2022, three ownership seeds, full pumps and 25% pumps.
Every farmer is a steward. Learning and social copying stay off. A paired
target-depth offset (normal SD 0.05) gives neighbours something to level toward.
Ranges are scenario assumptions, not fitted trait distributions.
"""
from pathlib import Path
import sys, os, json, time, itertools, hashlib, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "outputs/abm_testing_review/runtime"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
OUT = ROOT / "outputs" / "two_trait_sensitivity"
OUT.mkdir(exist_ok=True)


def run(job):
    import numpy as np
    from abm.model import DistrictModel
    from abm.world import SeasonConfig
    warnings.filterwarnings("ignore", category=FutureWarning)
    label, cooperation, work_pattern, scenario, seed = job
    start = time.perf_counter()
    cfg = SeasonConfig(
        year=2022, days=131, seed=seed, ownership_seed=seed, paddies_per_farmer=50,
        pump_capacity_m3_s=3.304 if scenario == "historical" else 3.304 * 0.25,
    )
    model = DistrictModel(cfg, agent_roster=["steward"], station_name="mpc", seed=seed)
    rng = np.random.default_rng(seed + 10000)
    for fid in sorted(model.farmer_agents):
        initial = float(np.clip(1.0 + rng.normal(0, 0.05), 0.65, 1.5))
        model.set_farmer_agent(
            fid, "steward", cooperation=cooperation, work_pattern=work_pattern,
            target_multiplier=initial, learning_rate=0.0, social_influence=0.0,
            trait_variability=0.0, schedule_jitter=0.0, week_swap=0.0,
        )
    for _ in range(131):
        model.step()
    world = model.world
    area = world.area_m2
    volume = lambda name: np.array([float((getattr(day, name) * area / 1000).sum()) for day in world.history])
    requested = volume("requested_mm")
    applied = volume("irrigation_mm")
    drained = volume("drainage_mm")
    result = dict(
        label=label, cooperation=cooperation, work_pattern=work_pattern,
        scenario=scenario, seed=seed, n_paddies=world.n, n_farmers=len(model.farmer_agents),
        demand_m3=float(requested.sum()), applied_m3=float(applied.sum()),
        peak7_m3=float(np.convolve(requested, np.ones(7), "valid").max()),
        drainage_m3=float(drained.sum()),
        lake_m3=float(sum(day.withdraw_m3 for day in world.history)),
        stress_mm=float(sum(np.average(day.stress_mm, weights=area) for day in world.history)),
        yield_fraction=float(np.average(world.relative_yield, weights=area)),
        seconds=time.perf_counter() - start,
    )
    assert all(np.isfinite(v) for v in result.values() if isinstance(v, (float, int)))
    assert all(np.all(day.irrigation_mm <= day.irrigation_sent_mm + 1e-8) for day in world.history)
    (OUT / f"{scenario}_{seed}_{label}.json").write_text(json.dumps(result, indent=2))
    return result


def _effects(rows):
    import pandas as pd
    frame = pd.DataFrame(rows)
    metrics = ["demand_m3", "applied_m3", "peak7_m3", "drainage_m3", "lake_m3", "stress_mm", "yield_fraction"]
    out = []
    for scenario, block in frame.groupby("scenario"):
        for seed, seeded in block.groupby("seed"):
            centre = seeded[(seeded.cooperation == 0.5) & (seeded.work_pattern == 0.5)].iloc[0]
            corners = seeded[seeded.label.str.startswith("f")]
            for factor, low, high in (("cooperation", 0.0, 1.0), ("work_pattern", 0.0, 1.0)):
                other = "work_pattern" if factor == "cooperation" else "cooperation"
                for metric in metrics:
                    high_mean = corners[corners[factor] == high].groupby(other)[metric].mean().mean()
                    low_mean = corners[corners[factor] == low].groupby(other)[metric].mean().mean()
                    effect = float(high_mean - low_mean)
                    base = float(centre[metric])
                    percent = 100.0 * effect / base if base else float("nan")
                    out.append(dict(scenario=scenario, seed=int(seed), factor=factor, metric=metric,
                                    effect=effect, percent=percent))
    detail = pd.DataFrame(out)
    summary = (detail.groupby(["scenario", "factor", "metric"], as_index=False)
               .agg(mean_effect=("effect", "mean"), mean_percent=("percent", "mean"),
                    min_percent=("percent", "min"), max_percent=("percent", "max")))
    return detail, summary


def main():
    corners = [(f"f{i}", *values) for i, values in enumerate(itertools.product([0.0, 1.0], [0.0, 1.0]))]
    corners.append(("centre", 0.5, 0.5))
    jobs = [(*values, scenario, seed) for scenario in ("historical", "pump25") for seed in (11, 29, 47) for values in corners]
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in [ROOT / "abm/farmer.py", ROOT / "abm/agents.py", ROOT / "abm/world.py",
                            ROOT / "cyclic/physics.py", ROOT / "abm/model.py"]}
    (OUT / "design.json").write_text(json.dumps(dict(
        jobs=jobs, source_hashes=sources, year=2022, days=131,
        note="2x2 of cooperation and work_pattern, plus a 0.5/0.5 centre. Three ownership seeds. Not a Sobol index.",
    ), indent=2))
    pending = [job for job in jobs if not (OUT / f"{job[-2]}_{job[-1]}_{job[0]}.json").exists()]
    if pending:
        with ProcessPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(run, job): job for job in pending}
            for future in as_completed(futures):
                row = future.result()
                print(row["scenario"], row["seed"], row["label"], round(row["seconds"], 1), flush=True)
    rows = [json.loads((OUT / f"{job[-2]}_{job[-1]}_{job[0]}.json").read_text()) for job in jobs]
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)
    detail, summary = _effects(rows)
    detail.to_csv(OUT / "effects_by_seed.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    print("COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    main()

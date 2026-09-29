"""Experiments reported in the Environmental Modelling & Software manuscript.

    py scripts/ems_experiments.py            # everything
    py scripts/ems_experiments.py ratios gym # a subset

Writes CSV/JSON to outputs/paper/ems/.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

import cyclic
from cyclic import data
from cyclic.benchmark import farmer_track
from cyclic.identify import construct_ratios, equation_from_cache, equation_score_table, ratio_holdout
from cyclic.underground import underground_ratio_train_both

OUT = os.path.join(data.PROJECT, "outputs", "paper", "ems")
YEARS = (2021, 2022)
FARMERS = ("rainfed", "flooded", "persist", "et_follower", "et_match", "satellite")
GYM_POLICIES = ("zero", "random", "spill", "et_follower")
SPILL_CEILING_MM = 60.0


def _save(name, frame):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    frame.to_csv(path, index=False)
    print(frame.to_string(index=False), flush=True)
    print("wrote", path, flush=True)


def ratios():
    """Leftover-simplex pins (Table 5), joint training and single-year freezes."""
    rows = []
    for year in YEARS:
        pin = construct_ratios(year)
        rows.append({"protocol": "closed_form", "train": str(year), "eval": year,
                     "capture": pin["pool_capture_share"], "pin": pin["pool_capture_share"],
                     "neighbour": pin["kx_ratio"] + pin["ky_ratio"],
                     "stay": pin["kz_ratio"] + pin["ground_ratio"]})
    both = underground_ratio_train_both(YEARS)
    stats = both.train.stats
    for year in YEARS:
        rows.append({"protocol": "joint", "train": "2021+2022", "eval": year,
                     "capture": both.train_capture, "pin": both.table_captures[year],
                     "neighbour": stats.kx_mean + stats.ky_mean,
                     "stay": stats.kz_mean + stats.ground_mean,
                     "capture_cv": stats.capture_cv})
    for train_year, eval_year in ((2021, 2022), (2022, 2021)):
        hold = ratio_holdout(train_year=train_year, eval_year=eval_year)
        rows.append({"protocol": "freeze", "train": str(train_year), "eval": eval_year,
                     "capture": hold.train_capture, "pin": hold.eval_capture})
    frame = pd.DataFrame(rows)
    frame["abs_error"] = (frame["capture"] - frame["pin"]).abs()
    frame["fidelity"] = 1.0 - frame["abs_error"] / frame["pin"]
    _save("ratios.csv", frame)
    return frame


def farmers(days=None):
    """Track A: registered farmer baselines scored on block IR/D against Okura."""
    tables = []
    for year in YEARS:
        start = time.time()
        report = farmer_track(FARMERS, year=year, days=days)
        table = report.table.drop(columns=["observed"])
        table["seconds"] = round(time.time() - start, 1)
        tables.append(table)
    frame = pd.concat(tables, ignore_index=True)
    _save("farmers.csv", frame)
    return frame


def _gym_season(policy, year, obs_role="farmer", seed=0, hide=(), env_id="Cyclic-Imbanuma-v0",
                spill_mm=SPILL_CEILING_MM):
    import gymnasium as gym

    env = gym.make(env_id, year=year, obs_role=obs_role, disable_env_checker=True)
    open_ir = env_id != "Cyclic-Imbanuma-v0"
    if hide:
        env.unwrapped.set_obs_channels("farmer", hide=hide)
    obs, info = env.reset(seed=seed)
    core = env.unwrapped.core
    rng = np.random.default_rng(seed)
    rewards, steps = [], 0
    start = time.time()
    farmer = None
    if policy not in ("random", "zero", "spill"):
        farmer = cyclic.make_agent(policy)
        farmer.reset(core.model, range(core.model.n))
    while True:
        batch = info["obs_farmer"]
        if policy == "random":
            action = rng.uniform(0.0, 20.0, size=env.action_space.shape).astype(np.float32)
        elif policy == "zero":
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        elif policy == "spill":
            action = np.maximum(np.asarray(batch.depth_mm, dtype=np.float32) - spill_mm, 0.0)
        else:
            irrigation, discharge = farmer.decide_batch(batch)
            if open_ir:
                action = np.column_stack([irrigation, discharge]).astype(np.float32)
            else:
                action = np.asarray(discharge, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        steps += 1
        if terminated or truncated:
            break
    elapsed = time.time() - start
    score = info.get("score")
    env.close()
    return {
        "env": env_id, "policy": policy if policy != "spill" else f"spill_{spill_mm:g}mm",
        "year": year, "obs_role": obs_role, "hidden": ";".join(hide), "steps": steps,
        "mean_reward": float(np.mean(rewards)), "seconds": round(elapsed, 1),
        "steps_per_second": round(steps / elapsed, 2),
        "n_paddies": int(core.n_paddies),
        "obs_dim": int(len(core.visible_fields(obs_role))),
        "irrigation_fidelity": None if score is None else float(score.irrigation_fidelity),
        "discharge_fidelity": None if score is None else float(score.discharge_fidelity),
        "et_fidelity": None if score is None else float(getattr(score, "et_fidelity", np.nan)),
    }


def gym():
    """Gymnasium API: season rollouts for simple policies, throughput and obs sizes."""
    rows = [_gym_season(policy, year) for year in YEARS for policy in GYM_POLICIES]
    rows += [_gym_season("spill", year, spill_mm=20.0) for year in YEARS]
    rows += [_gym_season("et_follower", year, hide=("satellite",)) for year in YEARS]
    for hide in ((), ("satellite",)):
        rows += [_gym_season("et_follower", year, hide=hide, env_id="Cyclic-Imbanuma-Open-v0")
                 for year in YEARS]
    frame = pd.DataFrame(rows)
    _save("gym_rollouts.csv", frame)
    return frame


def observations():
    """Default channel packs per role and the fields each hides."""
    env = cyclic.CyclicEnv(year=2022, given_irrigation=True, record_paddies=False)
    env.reset(seed=0)
    full = set(env.visible_fields("full"))
    rows = []
    for role in ("farmer", "equation", "ratio", "full"):
        visible = set(env.visible_fields(role))
        rows.append({"role": role, "n_visible": len(visible),
                     "hidden": ";".join(sorted(full - visible)),
                     "sees_satellite": "sat_et_mm" in visible})
    frame = pd.DataFrame(rows)
    _save("observation_roles.csv", frame)
    return frame


def equations():
    """Track C: neighbour-head hop law fitted on 2021, frozen, scored on 2022."""
    report = equation_from_cache(train_year=2021, eval_year=2022, save=False)
    frame = pd.DataFrame(equation_score_table(report))
    _save("equations.csv", frame)
    return frame


def figures():
    """Fig. 2 (leftover capture protocols) and Fig. 3 (farmer track) from the CSVs above."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratio = pd.read_csv(os.path.join(OUT, "ratios.csv"))
    farm = pd.read_csv(os.path.join(OUT, "farmers.csv"))

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    labels, values, pins, colours = [], [], [], []
    for _, row in ratio.iterrows():
        labels.append(f"{row['protocol']}\n{row['train']}→{row['eval']}")
        values.append(row["capture"])
        pins.append(row["pin"])
        colours.append({"closed_form": "#9e9e9e", "joint": "#1f77b4", "freeze": "#ff7f0e"}[row["protocol"]])
    x = np.arange(len(labels))
    ax.bar(x, values, color=colours, width=0.6, label="inferred capture")
    ax.scatter(x, pins, color="black", marker="_", s=400, zorder=3, label="Table 5 pin (eval year)")
    for i, fid in enumerate(ratio["fidelity"]):
        ax.text(x[i], values[i] + 0.01, f"{fid:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylim(0.6, 0.88)
    ax.set_ylabel("capture share LFV/LfA (–)")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_capture_protocols.png"), dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    names = list(dict.fromkeys(farm["name"]))
    for ax, quantity in zip(axes, ("irrigation_fidelity", "discharge_fidelity")):
        for k, year in enumerate(YEARS):
            sub = farm[farm["year"] == year].set_index("name").loc[names]
            ax.bar(np.arange(len(names)) + (k - 0.5) * 0.38, sub[quantity], width=0.38, label=str(year))
        ax.set_xticks(np.arange(len(names)), names, rotation=35, ha="right", fontsize=7)
        ax.set_title(quantity.replace("_", " "), fontsize=8)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("season-total fidelity (–)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_farmer_track.png"), dpi=300)
    plt.close(fig)
    print("wrote figures to", OUT)


def environment():
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor(), "cyclic": "0.2.0",
            "numpy": np.__version__, "pandas": pd.__version__}
    try:
        import gymnasium
        info["gymnasium"] = gymnasium.__version__
    except ImportError:
        pass
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "environment.json"), "w") as handle:
        json.dump(info, handle, indent=2)
    print(info)


EXPERIMENTS = {"ratios": ratios, "observations": observations, "gym": gym,
               "equations": equations, "farmers": farmers, "figures": figures}


if __name__ == "__main__":
    environment()
    for name in sys.argv[1:] or EXPERIMENTS:
        print(f"\n=== {name} ===", flush=True)
        EXPERIMENTS[name]()

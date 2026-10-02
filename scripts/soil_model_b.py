"""Fit model B (per-paddy soil K) and validate it: synthetic twin + 2021->2022 holdout.

    python scripts/soil_model_b.py [--n-ens 200] [--rebuild-forcing]

Writes outputs/soil/validation.json and outputs/soil/soil_field.npz (fitted on 2021).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cyclic import soil  # noqa: E402


def _fmt(entry):
    return f"{entry['mean']:+.3f} [{entry['p05']:+.3f}, {entry['p95']:+.3f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-ens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rebuild-forcing", action="store_true")
    args = parser.parse_args()

    for year in (2021, 2022):
        soil.model_a_forcing(year, rebuild=args.rebuild_forcing)

    print("== synthetic twin (2021) ==")
    twin = soil.synthetic_twin(2021, n_ens=args.n_ens, seed=args.seed, progress=print)
    for f, entry in twin.report["fields"].items():
        pr, po = entry["prior"], entry["posterior"]
        print(f"{f}: rmse log {pr['rmse_log']:.3f} -> {po['rmse_log']:.3f}   "
              f"spatial r {pr['corr_spatial']:+.2f} -> {po['corr_spatial']:+.2f}   "
              f"level err {pr['level_err_log']:+.3f} -> {po['level_err_log']:+.3f}   "
              f"90% coverage {po['coverage_90']:.2f}")
    an = twin.report["anisotropy"]
    print(f"log(kx/ky): rmse {an['prior']['rmse_log']:.3f} -> {an['posterior']['rmse_log']:.3f}   "
          f"spatial r {an['posterior']['corr_spatial']:+.2f}")

    print("== holdout (fit 2021, predict 2022) ==")
    hold = soil.holdout(2021, 2022, n_ens=args.n_ens, seed=args.seed, progress=print)
    for season in ("train", "eval"):
        print(f"-- {season} ({hold.report[season == 'train' and 'train_year' or 'eval_year']})")
        for model in ("physics_default_uniform", "default_k", "prior", "posterior", "posterior_mean_k"):
            s = hold.report[season][model]
            print(f"  {model:24s} Lf N {_fmt(s['lf_north_rel_err'])}  Lf S {_fmt(s['lf_south_rel_err'])}  "
                  f"LFV {_fmt(s['lfv_rel_err'])}  ET rmse {s['et_anomaly_rmse_mm']['mean']:.3f} "
                  f"r {s['et_anomaly_corr']['mean']:+.4f}")

    path = soil.save_field(hold.result, hold.train, extra={"validation": "outputs/soil/validation.json"})
    field = soil.load_field(path)
    m = hold.train.cyclic
    shares = {k[6:]: float(np.average(field[k][m], weights=hold.train.area[m]))
              for k in field if k.startswith("share_") and k not in ("share_in_mm", "share_stress_mm")}
    k_summary = {f: {"median_mm_d": float(np.median(field[f])), "p05": float(np.percentile(field[f], 5)),
                     "p95": float(np.percentile(field[f], 95)),
                     "mean_log_sd": float(field[f + "_log_sd"].mean())} for f in soil.FIELDS}
    print("fitted K:", json.dumps(k_summary, indent=1))
    print("cyclic flow shares (area-weighted):", json.dumps(shares, indent=1))

    report = {"synthetic_twin": twin.report, "holdout": hold.report, "fitted_k": k_summary,
              "cyclic_flow_shares": shares, "field": path}
    out = os.path.join(soil.SOIL_DIR, "validation.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", out, "and", path)


if __name__ == "__main__":
    main()

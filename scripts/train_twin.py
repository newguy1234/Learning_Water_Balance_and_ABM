"""Synthetic-twin validation: ABM with random ratios → retrain A+B → recovered?

    python scripts/train_twin.py [--seed 0] [--epochs 300]

Writes outputs/learned/twin_<seed>.json and the true / learned ratios (.npz).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abm.twin import run_twin  # noqa: E402
from cyclic import learn  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    report, learned, truth = run_twin(seed=args.seed, epochs=args.epochs)
    base = os.path.join(learn.LEARNED_DIR, f"twin_{args.seed}")
    os.makedirs(learn.LEARNED_DIR, exist_ok=True)
    truth.save(base + "_truth.npz")
    learned.save(base + "_learned.npz")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print("ratio recovery (all paddies): corr / MAE   [uniform-guess MAE]")
    for name, r in report["recovery_all"].items():
        b = report["baseline_recovery_all"][name]
        print(f"  {name:8s} r={r['corr']:+.3f}  MAE={r['mae']:.4f}  [{b['mae']:.4f}]  truth sd {r['truth_sd']:.4f}")
    for key in ("fit_train", "frozen_test", "baseline_test"):
        v = report[key]
        print(f"{key:13s} unmet {v['unmet_et_share']:.4f}  NSE N {v['drainage_nse_north']:.3f} "
              f"S {v['drainage_nse_south']:.3f}  Lf N {v['lf_north_rel_err']:+.3f} S {v['lf_south_rel_err']:+.3f}  "
              f"LFV {v['lfv_rel_err']:+.3f}")


if __name__ == "__main__":
    main()

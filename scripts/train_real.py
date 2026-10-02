"""Fit models A and B together on one measured season, freeze them, score the other.

    python scripts/train_real.py [--epochs 300]

Writes outputs/learned/real.pt + ratios_real.npz (fitted on 2021, used by the ABM) and
outputs/learned/real_validation.json (2021→2022 and 2022→2021, against an untrained
baseline and a model fitted on the evaluated season itself).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cyclic import learn  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    seasons = {y: learn.Season.from_real(y) for y in (2021, 2022)}
    fitted = {}
    for year, season in seasons.items():
        print(f"== fitting A+B on {year}")
        allocator, ratio_field, history = learn.train(season, epochs=args.epochs)
        fitted[year] = (allocator, ratio_field)
        learn.save(allocator, ratio_field, os.path.join(learn.LEARNED_DIR, f"real_{year}.pt"),
                   train_year=year, final_loss=history[-1])

    report = {}
    for train_year, (allocator, ratio_field) in fitted.items():
        for eval_year, season in seasons.items():
            key = f"fit_{train_year}_eval_{eval_year}"
            report[key] = learn.evaluate(allocator, ratio_field, season)
    for eval_year, season in seasons.items():
        report[f"baseline_eval_{eval_year}"] = learn.evaluate(*learn.baseline(season), season)

    allocator, ratio_field = fitted[2021]
    learn.save(allocator, ratio_field, learn.REAL_WEIGHTS, train_year=2021)
    ratio_field.ratios().save(learn.REAL_RATIOS, train_year=2021)
    with open(os.path.join(learn.LEARNED_DIR, "real_validation.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    for key, value in report.items():
        print(f"{key:24s} unmet {value['unmet_et_share']:.4f}  NSE N {value['drainage_nse_north']:.3f} "
              f"S {value['drainage_nse_south']:.3f}  Lf N {value['lf_north_rel_err']:+.3f} "
              f"S {value['lf_south_rel_err']:+.3f}  LFV {value['lfv_rel_err']:+.3f}")


if __name__ == "__main__":
    main()

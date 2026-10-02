# -*- coding: utf-8 -*-
"""Sensitivity tests for the ABM farmer traits and weather scenarios, then a results UI.

Runs a batch of paired seasons (abm.sensitivity), saves them under
outputs/sensitivity/<timestamp>/ and opens a viewer with every graph (port 8771, apart
from the main ABM UI on 8770).

    python scripts/abm_sensitivity.py                      # 3 seeds, all traits, weather, pump shortage
    python scripts/abm_sensitivity.py --seeds 5 --interactions
    python scripts/abm_sensitivity.py --traits refill_overshoot,experience --weather baseline,drought
    python scripts/abm_sensitivity.py --quick              # toy grid, seconds: checks the pipeline only
    python scripts/abm_sensitivity.py --view               # just open the newest results
    python scripts/abm_sensitivity.py --no-ui              # CI / headless

Every finished season is saved under outputs/sensitivity/_cache, so if a batch dies
(out of memory, Ctrl+C) running the same command again resumes it. Seasons are reused
only while the abm/ and cyclic/ code is unchanged; --fresh reruns everything.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _csv(text):
    return [t.strip() for t in text.split(",") if t.strip()] if text else None


def main(argv=None) -> int:
    from abm import sensitivity as sens

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=3, help="replicate seeds (paired across every variant)")
    p.add_argument("--traits", type=_csv, help="comma list (default: every trait in abm.farmer.TRAITS)")
    p.add_argument("--weather", type=_csv, help="comma list of abm.weather presets (default: all)")
    p.add_argument("--interactions", action="store_true", help="repeat the trait sweep under every weather")
    p.add_argument("--year", type=int, default=2022)
    p.add_argument("--days", type=int, default=None, help="season length (default: full season)")
    p.add_argument("--per-farmer", type=int, default=50, help="paddies per farmer")
    p.add_argument("--no-shortage", action="store_true",
                   help="skip the pump-shortage scenario (the trait sweep with the canal short of water)")
    p.add_argument("--workers", type=int, default=None, help="parallel processes (default: every core, capped by free RAM)")
    p.add_argument("--fresh", action="store_true",
                   help="ignore seasons saved by earlier (interrupted) batches and rerun everything")
    p.add_argument("--quick", action="store_true", help="toy grid, 40 days, 2 seeds: pipeline check only")
    p.add_argument("--no-ui", action="store_true", help="do not open the results viewer")
    p.add_argument("--view", nargs="?", const="", default=None, metavar="DIR",
                   help="only open the viewer (newest results, or DIR)")
    p.add_argument("--port", type=int, default=8771)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)

    if args.view is not None:
        from abm.sensitivity_ui import serve
        return serve(args.view or None, port=args.port, open_browser=not args.no_browser) or 0

    common = dict(year=args.year, days=args.days, paddies_per_farmer=args.per_farmer)
    seeds = list(range(args.seeds))
    if args.quick:
        common.update(use_cyclic_map=False, days=args.days or 40, paddies_per_farmer=5)
        seeds = list(range(max(2, min(args.seeds, 2))))
    plan = sens.build_plan(seeds=seeds, traits=args.traits, weathers=args.weather,
                           interactions=args.interactions, shortage=not args.no_shortage, **common)
    print(f"{len(plan)} season runs over seeds {seeds}", flush=True)
    t0 = time.time()
    # Every finished season is saved here, so rerunning after a crash resumes the batch.
    cache = os.path.join(sens.RESULTS_ROOT, "_cache")
    if args.fresh and os.path.isdir(cache):
        import shutil
        shutil.rmtree(cache)
    runs, daily = sens.run_plan(plan, workers=args.workers, progress=lambda s: print(s, flush=True),
                                cache_dir=cache)
    meta = dict(seeds=seeds, n_runs=len(plan), interactions=args.interactions,
                shortage=not args.no_shortage, shortage_pump_fraction=sens.SHORTAGE_PUMP_FRACTION,
                minutes=(time.time() - t0) / 60, argv=sys.argv[1:], **common)
    meta.setdefault("use_cyclic_map", True)
    out = sens.save(runs, daily, meta)

    effects = sens.trait_effects(runs)
    weff = sens.weather_effects(runs)
    main_scenarios = ["baseline", "baseline" + sens.SHORTAGE]
    checked = effects[effects.expected.notna() & effects.scenario.isin(main_scenarios)]
    allv = list(checked.verdict) + list(weff[weff.expected.notna()].verdict)
    bad = [v for v in allv if v.startswith(("FAIL", "LEAK"))]
    print(f"\nSaved {out}")
    print(f"{len(allv) - len(bad)}/{len(allv)} expectations met")
    for _, r in checked[checked.verdict.str.startswith(("FAIL", "LEAK"))].iterrows():
        print(f"  {r.trait:>22} - {r.metric:<16} {r.scenario:<26} {r.verdict}  ({r.rel_pct:+.2f}%, p={r.p:.3g})")
    for _, r in weff[weff.expected.notna() & weff.verdict.str.startswith("FAIL")].iterrows():
        print(f"  {r.scenario:>22} - {r.metric:<16} {r.verdict}  ({r.rel_pct:+.2f}%, p={r.p:.3g})")

    if not args.no_ui:
        from abm.sensitivity_ui import serve
        serve(out, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

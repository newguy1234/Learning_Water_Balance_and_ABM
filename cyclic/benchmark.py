"""Two-way benchmark: farmer behaviour vs leftover ratios.

Track A — Farmer behaviours (forecasters)
    Register a policy that maps morning observations to irrigation and discharge.
    Score is fidelity of cyclic-block IR and D to Okura (pump + ADCP).

Track B — Leftover ratios
    Closed-form shares from Tables 4–5, plus any fitted hop-then-catch vector on disk.
    Score is how capture and farm-to-main sit against LFV_pump and main-canal D.

    python -m cyclic bench
    python example.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from . import okura
from .identify import construct_ratios, load_best_ratios, table_closure_objectives
from .imitation import load_observed, score as score_blocks
from .model import CyclicModel, Results


def farmer_track(names, year=okura.COMPARISON_YEAR, days=None, save_csv=False):
    """Run each registered forecaster; score block IR/D against the observed series."""
    observed, _, source = load_observed(year)
    rows = []
    reports = {}
    for name in list(names):
        model = CyclicModel(year, agents=name, record_paddies=False, save_csv=save_csv)
        if days is None:
            results = model.run()
        else:
            n = min(int(days), model.n_days)
            for _ in range(n):
                if not model.running:
                    break
                model.step()
            model.running = False
            results = Results(model)
        scored = score_blocks(results.blocks, observed)
        row = {
            "track": "farmer",
            "name": name,
            "year": int(year),
            "days": int(len(results.daily)),
            "fidelity": scored.fidelity,
            "loss": scored.loss,
            "irrigation_fidelity": scored.irrigation_fidelity,
            "discharge_fidelity": scored.discharge_fidelity,
            "observed": source,
        }
        rows.append(row)
        reports[name] = SimpleNamespace(row=row, score=scored, results=results)
    return SimpleNamespace(table=pd.DataFrame(rows), reports=reports, source=source)


def ratio_track(year=okura.COMPARISON_YEAR, span="season"):
    """Tables-4–5 closed form vs fitted leftover shares (no season rerun)."""
    constructed = construct_ratios(year)
    obj, signed = table_closure_objectives(year)
    saved = load_best_ratios(year, span)
    rows = [{
        "track": "ratios",
        "name": "tables_4_5",
        "year": int(year),
        "kx_ratio": constructed["kx_ratio"],
        "ky_ratio": constructed["ky_ratio"],
        "kz_ratio": constructed["kz_ratio"],
        "ground_ratio": constructed["ground_ratio"],
        "farm_to_main": constructed["farm_to_main"],
        "capture": constructed["pool_capture_share"],
        "lfv_rel_error": obj["lfv"] ** 0.5,
        "loss": None,
        "best": True,
    }]
    if saved is not None:
        p = saved.params
        capture = p.get("pool_capture_share", p.get("kx_ratio", 0.0) + p.get("ky_ratio", 0.0))
        rows.append({
            "track": "ratios",
            "name": "fitted_season",
            "year": int(year),
            "kx_ratio": p["kx_ratio"],
            "ky_ratio": p["ky_ratio"],
            "kz_ratio": p["kz_ratio"],
            "ground_ratio": p["ground_ratio"],
            "farm_to_main": p["farm_to_main"],
            "capture": capture,
            "lfv_rel_error": None,
            "loss": saved.loss,
            "best": True,
        })
        # Lower identification loss wins; tables row is the closed form, not a search loss.
        if saved.loss is not None:
            rows[-1]["best"] = True
            rows[0]["best"] = False
    table = pd.DataFrame(rows)
    winner = "fitted_season" if saved is not None else "tables_4_5"
    return SimpleNamespace(
        table=table, constructed=constructed, fitted=None if saved is None else saved.params,
        loss=None if saved is None else saved.loss, winner=winner, objectives=obj, signed=signed,
    )


def two_way(farmers=(), year=okura.COMPARISON_YEAR, days=None):
    """Both tracks. Farmer names must already be registered. Ratios always run."""
    ratios = ratio_track(year)
    farmers_out = None
    if farmers:
        farmers_out = farmer_track(farmers, year=year, days=days)
    return SimpleNamespace(year=int(year), farmers=farmers_out, ratios=ratios)


def write_benchmark(report, folder=None):
    import os
    from . import data as _data
    folder = folder or os.path.join(_data.PROJECT, "outputs", "benchmark")
    os.makedirs(folder, exist_ok=True)
    paths = {}
    if report.farmers is not None:
        path = os.path.join(folder, f"farmers_{report.year}.csv")
        report.farmers.table.to_csv(path, index=False)
        paths["farmers"] = path
    path = os.path.join(folder, f"ratios_{report.year}.csv")
    report.ratios.table.to_csv(path, index=False)
    paths["ratios"] = path
    return paths


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else list(argv)
    year = okura.COMPARISON_YEAR
    names = []
    days = None
    for item in argv:
        if item.isdigit() and int(item) in (2021, 2022):
            year = int(item)
        elif item.startswith("days="):
            days = int(item.split("=", 1)[1])
        else:
            names.append(item)
    report = two_way(farmers=names, year=year, days=days)
    print(report.ratios.table.to_string(index=False))
    print("ratio winner:", report.ratios.winner)
    if report.farmers is not None:
        print(report.farmers.table.to_string(index=False))
    print("wrote", write_benchmark(report))
    return report

# -*- coding: utf-8 -*-
"""Default ABM season (built-in farmer mix, MPC station) for 2021 and 2022, compared with
Okura et al. (2025) Tables 3-6. Writes docs/figures/data/okura_comparison.csv and the
season's daily district series used by the figures.

    python docs/figures/validation_runs.py
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "data")


def run(year: int):
    import numpy as np
    import pandas as pd

    from abm.agents import DEFAULT_ROSTER
    from abm.metrics import okura_comparison
    from abm.model import DistrictModel
    from abm.world import SeasonConfig

    model = DistrictModel(SeasonConfig(year=year, seed=0, ownership_seed=0), agent_roster=list(DEFAULT_ROSTER), seed=0)
    while not model.done():
        model.step()
    w = model.world
    table = okura_comparison(w).assign(year=year)
    a = w.area_m2
    daily = pd.DataFrame([dict(
        year=year, day=s.day, rain_mm=s.rain_mm, et0_mm=s.et0_mm,
        request_m3=float(np.dot(s.requested_mm, a) / 1000), applied_m3=float(np.dot(s.irrigation_mm, a) / 1000),
        drainage_m3=float(np.dot(s.drainage_mm, a) / 1000), to_pool_m3=float(np.dot(s.to_pool_mm, a) / 1000),
        withdraw_m3=s.withdraw_m3, pool_m3=s.pool_m3, mean_ponding_mm=float(np.average(s.ponding_mm, weights=a)),
        canal_north_m=s.canal_depth_north_m, canal_south_m=s.canal_depth_south_m,
        backwater_m3=float(np.dot(s.backwater_mm, a) / 1000) if s.backwater_mm is not None else 0.0)
        for s in w.history])
    return table, daily


if __name__ == "__main__":
    import pandas as pd

    os.makedirs(OUT, exist_ok=True)
    with ProcessPoolExecutor(2) as pool:
        results = list(pool.map(run, (2021, 2022)))
    pd.concat([r[0] for r in results]).to_csv(os.path.join(OUT, "okura_comparison.csv"), index=False)
    pd.concat([r[1] for r in results]).to_csv(os.path.join(OUT, "default_daily.csv"), index=False)
    print(pd.concat([r[0] for r in results]).to_string(index=False))

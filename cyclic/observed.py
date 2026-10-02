"""What Okura et al. (2025) measured: daily block irrigation and drainage, season tables.

Cyclic blocks, daily (data/observed/okura_<year>_blocks.csv): irrigation from Fig. 8
(digitised), drainage from Fig. 6 monthly discharge scaled to Table 4. Non-cyclic
irrigation is the station's total minus the cyclic part (`irrigation_all_m3`); their
drainage was not measured daily, so the season total (Table 4 closure, okura.d_ncb_m3)
follows the cyclic drainage's daily shape.

Season targets for the soil: Table 5 LfA_North / LfA_South (water that went into the
ground, Lf = IR + R − D − ET) and LFV_pump (lateral flow recaptured at the pool).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from stem import okura
from stem.district import DATA, GROUPS


def _blocks_csv(year):
    return pd.read_csv(os.path.join(DATA, "observed", f"okura_{int(year)}_blocks.csv"), parse_dates=["date"])


def block_forcing(district):
    """Daily m³ per group (4, days) for irrigation and drainage, in district.GROUPS order."""
    frame = _blocks_csv(district.year)
    dates = pd.DatetimeIndex(district.dates).normalize()
    ir = np.zeros((len(GROUPS), len(dates)))
    dr = np.zeros((len(GROUPS), len(dates)))
    for block in okura.BLOCKS:
        rows = frame[frame["block"] == block].set_index("date").reindex(dates).fillna(0.0)
        cyc_ir = rows["irrigation_m3"].to_numpy()
        cyc_dr = rows["drainage_m3"].to_numpy()
        g_c = GROUPS.index((block, True))
        g_n = GROUPS.index((block, False))
        ir[g_c], dr[g_c] = cyc_ir, cyc_dr
        if "irrigation_all_m3" in rows and rows["irrigation_all_m3"].sum() > cyc_ir.sum():
            ir[g_n] = np.maximum(rows["irrigation_all_m3"].to_numpy() - cyc_ir, 0.0)
        else:
            ir[g_n] = cyc_ir / max(cyc_ir.sum(), 1e-9) * okura.ir_ncb_m3(block, district.year)
        season_rain = float(district.rain_mm.sum())
        dr[g_n] = cyc_dr / max(cyc_dr.sum(), 1e-9) * okura.d_ncb_m3(block, district.year, season_rain)
    return ir, dr


def soil_targets(year):
    t5 = okura.TABLE5_1000M3[int(year)]
    return SimpleNamespace(lf_m3={"North": t5["LfA_North"] * 1e3, "South": t5["LfA_South"] * 1e3},
                           lfv_m3=t5["LFV_pump"] * 1e3, dwb_m3=t5["dWB"] * 1e3)

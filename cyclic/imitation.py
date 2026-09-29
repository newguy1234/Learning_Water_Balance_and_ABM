"""The environment's side of imitation: the demonstration, the score, and the hook that
training algorithms register through. The algorithms themselves live outside the
package, in example.py.

THE DEMONSTRATION. Okura et al. record the irrigation pumped to each cyclic block and the
drainage leaving it and flowing back to the pool -- one number per block per day. Nobody
recorded what any one farmer did, so an agent has to reproduce block totals from plot
decisions.

Both are VOLUMES (m3/d): irrigation from the pump records, drainage back to the pool from
the ADCPs in the main drainage canal. tools/build_observed.py writes

    data/observed/okura_<year>_blocks.csv    date, block, area_m2, irrigation_m3, drainage_m3,
                                             rain_mm, et_mm, source
    data/observed/okura_<year>_station.csv   date, W_m3, O_m3, LFV_pump_m3

from the paper's Fig. 8 (daily irrigation), Fig. 6 (monthly drainage) and tables. Replace
them with the measured files when they arrive -- irrigation_m3 and drainage_m3 are all a
blocks file needs. With no files at all, load_observed() falls back on
okura.interpolate_daily and says so.

IDENTIFIABILITY. With only IR and D observed, seepage (plow pan K: physics) and ponding
depth (behaviour) trade off, since both set IR - D. Fix the physics, or add LFV_pump to
the score, before reading a learned depth as behaviour.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import data, okura

BLOCKS = okura.BLOCKS
OBSERVED_DIR = os.path.join(data.DATA, "observed")
# How much the day-to-day shape counts against the season totals.
DAILY_WEIGHT = 0.25
QUANTITIES = ("irrigation", "drainage")


class Cancelled(Exception):
    """Raise from a learner's callback to stop it."""


def _complete(blocks):
    """Fill in whichever of the m3 / mm columns a blocks file left out."""
    if "area_m2" not in blocks:
        blocks["area_m2"] = blocks["block"].map(okura.cyclic_area_m2)
    for q in QUANTITIES:
        if f"{q}_m3" not in blocks and f"{q}_mm" in blocks:
            blocks[f"{q}_m3"] = blocks[f"{q}_mm"] * blocks["area_m2"] / 1000.0
        if f"{q}_mm" not in blocks and f"{q}_m3" in blocks:
            blocks[f"{q}_mm"] = blocks[f"{q}_m3"] / blocks["area_m2"] * 1000.0
    return blocks


def load_observed(year=okura.COMPARISON_YEAR, prefer_measured=True):
    """(blocks, station, source). Files in data/observed win; otherwise the paper's tables,
    interpolated."""
    measured = os.path.join(OBSERVED_DIR, f"okura_{year}_blocks.csv")
    if prefer_measured and os.path.exists(measured):
        blocks = _complete(pd.read_csv(measured, parse_dates=["date"]))
        station_path = os.path.join(OBSERVED_DIR, f"okura_{year}_station.csv")
        station = pd.read_csv(station_path, parse_dates=["date"]) if os.path.exists(station_path) else None
        source = str(blocks["source"].iloc[0]) if "source" in blocks else f"measured ({measured})"
        return blocks, station, source

    cache_blocks = os.path.join(data.DATA, f"okura_interpolated_{year}_blocks.csv")
    cache_station = os.path.join(data.DATA, f"okura_interpolated_{year}_station.csv")
    if os.path.exists(cache_blocks) and os.path.exists(cache_station):
        blocks = pd.read_csv(cache_blocks, parse_dates=["date"])
        station = pd.read_csv(cache_station, parse_dates=["date"])
    else:
        blocks, station = okura.interpolate_daily(year, data.weather(year))
        blocks.to_csv(cache_blocks, index=False)
        station.to_csv(cache_station, index=False)
    return _complete(blocks), station, "interpolated from Okura et al. (2025) tables"


def block_series(blocks, block, dates):
    """One block's observed irrigation_m3 and drainage_m3, one value per date (0 if missing)."""
    rows = _complete(blocks[blocks["block"] == block].copy()).set_index("date").reindex(dates)
    return {q + "_m3": rows[q + "_m3"].fillna(0.0).to_numpy() for q in QUANTITIES}


def score(sim_blocks, observed_blocks, quantities=QUANTITIES):
    """How closely a season reproduced the observed block series: irrigation into and
    drainage back out of each cyclic block, both as volumes (x10^3 m3).

    loss      mean over (block, quantity) of season_error^2 + DAILY_WEIGHT x nRMSE^2
    fidelity  1 - mean |season error|, floored at 0: 1 is exact season totals
    """
    observed_blocks = _complete(observed_blocks.copy())
    sim_blocks = _complete(sim_blocks.copy())
    rows, losses, errors = [], [], []
    for block in BLOCKS:
        sim = sim_blocks[sim_blocks["block"] == block].set_index("date")
        obs = observed_blocks[observed_blocks["block"] == block].set_index("date")
        common = sim.index.intersection(obs.index)
        if common.empty:
            continue
        for q in quantities:
            s = sim.loc[common, f"{q}_m3"].to_numpy() / 1000.0
            o = obs.loc[common, f"{q}_m3"].to_numpy() / 1000.0
            season_error = (s.sum() - o.sum()) / o.sum() if o.sum() else np.nan
            rmse = float(np.sqrt(np.mean((s - o) ** 2)))
            nrmse = rmse / o.mean() if o.mean() else np.nan
            spread = np.sum((o - o.mean()) ** 2)
            rows.append({"block": block, "quantity": q, "model_1000m3": s.sum(), "observed_1000m3": o.sum(),
                         "season_error_%": 100.0 * season_error, "rmse_1000m3_d": rmse,
                         "nse": 1.0 - np.sum((s - o) ** 2) / spread if spread else np.nan,
                         "corr": float(np.corrcoef(s, o)[0, 1]) if s.std() and o.std() else np.nan})
            losses.append(season_error ** 2 + DAILY_WEIGHT * nrmse ** 2)
            errors.append(abs(season_error))
    table = pd.DataFrame(rows)
    def component(quantity):
        selected = [abs(row["season_error_%"]) / 100.0 for row in rows if row["quantity"] == quantity]
        return max(0.0, 1.0 - float(np.mean(selected))) if selected else float("nan")

    return SimpleNamespace(table=table, loss=float(np.mean(losses)),
                           fidelity=max(0.0, 1.0 - float(np.mean(errors))),
                           irrigation_fidelity=component("irrigation"),
                           discharge_fidelity=component("drainage"))


# --- the hook learners register through -------------------------------------------------------

LEARNERS = {}


def register_learner(name):
    def add(cls):
        cls.name = name
        LEARNERS[name] = cls
        return cls
    return add


@dataclass
class LearnResult:
    name: str
    policies: dict                 # {"North": agent, "South": agent}
    score: SimpleNamespace
    history: pd.DataFrame          # one row per step: step, loss, best_loss, fidelity, ...
    results: object = None         # the evaluated season (model.Results)
    seconds: float = 0.0
    notes: dict = field(default_factory=dict)


class Learner:
    """A training algorithm the interface can launch.

    `settings` are {name: (default, low, high)}; the interface shows them as sliders and
    builds the learner as Learner(base=<the Base farmer>, **settings), then calls
    fit(year, observed, callback). callback(row) takes a dict with at least step, loss,
    best_loss and fidelity; raise Cancelled from it to stop. Return a LearnResult.
    """

    name = "learner"
    settings: dict = {}
    # True if the learner trains the agent through its own learn(); the interface then refuses
    # to launch from a Base farmer that has none, and says why.
    requires_learn = False

    def fit(self, year=2021, observed=None, callback=None, evaluation_year=None):
        raise NotImplementedError

    @staticmethod
    def rollout(year, policies, physics=None, record=False):
        from .model import CyclicModel

        return CyclicModel(year, agents=policies, physics=physics, record_paddies=record).run()

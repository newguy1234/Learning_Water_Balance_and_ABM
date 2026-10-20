"""Register your own agents, then open a UI.

    python example.py            # cyclic UI (forecasters + ratio finders)
    python example.py abm        # ABM UI (farmers + station staff)

Anything registered here shows up in the UIs' menus. In the ABM, a farmer registered
here joins the default farmer mix, and a station registered here becomes the default
staff (otherwise the built-in "mpc" optimiser is used).
"""

from __future__ import annotations

import os
import sys

import numpy as np

import abm
import cyclic


# 1. cyclic forecaster: a farmer policy for the cyclic model -------------------------------
@cyclic.register_forecaster("my_forecaster")
class MyForecaster(cyclic.FarmerPolicy):
    """Refill shallow paddies, send back anything above a ceiling."""

    defaults = dict(refill_below_mm=10.0, refill_mm=20.0, ceiling_mm=60.0)
    bounds = dict(refill_below_mm=(0.0, 50.0), refill_mm=(0.0, 60.0), ceiling_mm=(20.0, 150.0))

    def decide_batch(self, observation):
        p = self.params
        depth = observation.depth_mm
        irrigation = np.where(depth < p["refill_below_mm"], p["refill_mm"], 0.0)
        discharge = np.maximum(depth - p["ceiling_mm"], 0.0)
        return irrigation, discharge


# 2. cyclic ratio finder: leftover shares (Kx / Ky / Kz / ground) at every paddy ----------
@cyclic.register_ratio("my_ratios")
class MyRatios(cyclic.RatioPolicy):
    """Start from the Okura table targets, then reweight the stay share to the leftover."""

    def partition(self, leftover, observation, targets):
        shares = {name: np.array(targets.shares[name], copy=True)
                  for name in cyclic.agents.RATIO_SHARE_KEYS if name in targets.shares}
        if leftover is not None:
            self.fit_stay_to_leftover(shares, leftover, observation, targets)
        return shares


# 3. ABM farmer: decides for every paddy it owns --------------------------------------------
# Every farmer must inherit abm.BaseFarmer (register_farmer refuses anything else). It
# already brings the stage ponding depth, the outlet weir board and the hard
# no-drain holds. Change what you like: exact params (defaults), any hook,
# and extra constraints.
class NoDrainBeforeHarvestInspection(abm.Constraint):
    """Extra hard rule: keep the weir up for 3 days before the harvest drain so the
    inspector sees the crop standing in water (just an example of a custom constraint)."""

    name = "inspection"

    def active(self, farmer, obs, crop_day):
        return abm.crop.HARVEST_DRAIN_DAY - 3 <= crop_day < abm.crop.HARVEST_DRAIN_DAY

    def enforce(self, farmer, obs, plan, pond):
        plan.drain[:] = 0.0
        plan.weir[:] = obs["bund_mm"]


@abm.register_farmer("my_farmer")
class MyFarmer(abm.BaseFarmer):
    """Custom farmer: same stage depth as everyone else, plus one extra hard rule."""

    constraints = abm.BaseFarmer.constraints + (NoDrainBeforeHarvestInspection(),)


# 4. ABM station: the LID staff deciding lake intake, outflow and what each paddy is sent ---
@abm.register_station("my_station")
class MyStation(abm.StationAgent):
    """Send each paddy what it needs (deficit + ET + seepage), open the lake gate for the rest."""

    defaults = dict(max_send_mm=30.0, max_withdraw_m3=150_000.0)

    def decide(self, obs):
        send = np.minimum(obs["water_needed_mm"], self.params["max_send_mm"])
        need_m3 = float((send * obs["area_m2"]).sum() / 1000.0)
        self.state = "irrigation" if need_m3 > 0 else "wait"
        return {
            "withdraw_m3": self._gate_m3(obs, need_m3),
            "outflow_m3": 0.0,
            "irrigation_sent_mm": send,
            "state": self.state,
        }


# 5. Launch a UI ------------------------------------------------------------------------------
# The UI server imports this file (so the agents above are registered there) and serves
# its `Page`; EXAMPLE_UI tells it which UI's page that is.
if cyclic.is_server():
    if os.environ.get("EXAMPLE_UI") == "abm":
        from abm.ui import Page  # noqa: F401
    else:
        from cyclic.app import Page  # noqa: F401

if __name__ == "__main__":
    ui = "abm"
    os.environ["EXAMPLE_UI"] = ui
    if ui == "abm":
        abm.serve(__file__)
    else:
        cyclic.serve(__file__)

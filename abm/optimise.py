# -*- coding: utf-8 -*-
"""Baseline optimising LID staff: receding-horizon (MPC) water dispatch.

Model predictive control is the standard modern approach to canal and paddy
irrigation scheduling: each day, predict every paddy's water with a process model
and a short weather forecast, choose today's releases to meet tomorrow's needs at
least cost, apply only today's decision, then re-plan tomorrow on the new state.
This agent is a deliberately simple, transparent version of that loop, meant as the
baseline any learned or heuristic staff policy has to beat.

Objective, per day (maximising Okura WUE' / RUE / LUE while protecting yield):
  * no paddy is short of water tomorrow (stress = weather ET demand the pond and
    topsoil cannot supply, see ``cyclic.physics.supply_et_demand``);
  * irrigation is the *minimum* that achieves that — shallow ponds percolate and
    leak less (percolation ∝ depth), so less applied water is lost;
  * water already flowing back (pool + same-day cyclic D + recaptured lateral flow)
    is used before any lake water, so reuse ratios rise and withdrawal W falls;
  * the lake gate only opens for the remaining shortfall, and the pool is never
    pumped to the lake except ahead of flood rain (outflow O).

Pond model used for the plan (the same processes as ``cyclic.physics.advance``,
with no farmer discharge assumed), per paddy with pond h₀, today's rain R₀, send s
and ET demand E₀:

    end-of-day pond  h₁ = max(h₀ + R₀ + s − E₀, 0) · ρ,
    ρ = (1 − outlet_loss) · (1 − K_pan / h_ref)      (leak, then plow-pan percolation)

Tomorrow is stress-free if h₁ covers tomorrow's demand net of rain, plus a margin:
h₁ ≥ max(E₁ − R₁, 0) + margin. Inverting gives the least send s for each paddy.
"""

from __future__ import annotations

import numpy as np

from .station import StationAgent, register_station


@register_station("mpc")
class MPCStation(StationAgent):
    """Receding-horizon staff: least-water, reuse-first, stress-free dispatch."""

    name = "mpc"
    defaults = dict(
        margin_mm=3.0,            # safety pond kept above tomorrow's net demand
        max_send_mm=60.0,         # canal delivery limit per paddy per day
        max_withdraw_m3=150_000.0,  # lake culvert capacity per day
        max_outflow_m3=50_000.0,
        heavy_rain_mm=40.0,       # flood-control drawdown trigger (today + tomorrow)
        overflow_spill_m3=30_000.0,
        bank_returns=True,        # store surplus returns in cyclic ponds up to target
        forecast_safety=0.05,     # extra gate opening against return-forecast error
    )

    def plan_sends(self, obs: dict) -> np.ndarray:
        """Least send (mm) per paddy so no paddy is short of water tomorrow."""
        p = self.params
        phys = obs["pond_physics"]
        pond = np.asarray(obs["ponding_mm"], dtype=float)
        rain = np.asarray(obs["rain_forecast_mm"], dtype=float)
        demand = np.asarray(obs["et_demand_forecast_mm"], dtype=float)
        r0, r1 = float(rain[0]), float(rain[min(1, rain.size - 1)])
        e0, e1 = float(demand[0]), float(demand[min(1, demand.size - 1)])
        keep = (1.0 - phys["outlet_loss_per_day"]) * (1.0 - min(phys["percolation_k_mm_d"] / phys["pond_ref_mm"], 0.95))
        need_end = max(e1 - r1, 0.0) + float(p["margin_mm"])
        send = need_end / max(keep, 1e-6) + e0 - pond - r0
        # Never plan a pond above the bund.
        send = np.minimum(send, phys["bund_mm"] - pond - r0 + e0)
        return np.clip(send, 0.0, float(p["max_send_mm"]))

    def _bank(self, obs: dict, send: np.ndarray, surplus_m3: float) -> np.ndarray:
        """Spend returns that would otherwise spill on cyclic ponds below target
        (their seepage flows back to the pool), then on the rest."""
        area = np.asarray(obs["area_m2"], dtype=float)
        pond = np.asarray(obs["ponding_mm"], dtype=float)
        rain = float(np.asarray(obs["rain_forecast_mm"])[0])
        room = np.maximum(float(obs["target_depth_mm"]) - (pond + rain + send), 0.0)
        room = np.minimum(room, float(self.params["max_send_mm"]) - send)
        cyclic = np.asarray(obs["cyclic"], dtype=bool)
        for group in (cyclic, ~cyclic):
            want_m3 = float((room[group] * area[group]).sum() / 1000.0)
            if surplus_m3 <= 0 or want_m3 <= 0:
                continue
            share = min(surplus_m3 / want_m3, 1.0)
            send = send + np.where(group, room * share, 0.0)
            surplus_m3 -= want_m3 * share
        return send

    def decide(self, obs: dict) -> dict:
        p = self.params
        area = np.asarray(obs["area_m2"], dtype=float)
        rain = np.asarray(obs["rain_forecast_mm"], dtype=float)
        send = self.plan_sends(obs)
        need_m3 = float((send * area).sum() / 1000.0)
        free_m3 = max(float(obs["pool_m3"]), 0.0) + float(obs.get("forecast_return_m3", 0.0))

        flood = float(rain[:2].sum()) > float(p["heavy_rain_mm"]) or \
            float(obs.get("last_spill_m3", 0.0)) > float(p["overflow_spill_m3"])
        self.state = "drain_to_lake" if flood else ("irrigation" if need_m3 > 0 else "wait")

        shortfall = max(need_m3 - free_m3, 0.0)
        gate = min(shortfall * (1.0 + float(p["forecast_safety"])), float(p["max_withdraw_m3"]))
        available = free_m3 + gate
        if need_m3 > available > 0:
            # Culvert-limited: cut every paddy by the same fraction (no distance bias).
            send = send * (available / need_m3)
        elif need_m3 < free_m3 and p["bank_returns"]:
            send = self._bank(obs, send, free_m3 - need_m3)

        # Flood control: empty the pool ahead of heavy rain so the drainage pumps
        # have room; otherwise never pump to the lake (the pool spills on its own).
        outflow = min(max(float(obs["pool_m3"]), 0.0), float(p["max_outflow_m3"])) if flood else 0.0
        return {
            "withdraw_m3": gate,
            "outflow_m3": outflow,
            "irrigation_sent_mm": send,
            "state": self.state,
        }

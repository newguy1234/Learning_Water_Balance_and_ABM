# -*- coding: utf-8 -*-
"""Irrigation-station (LID) staff registry — pool / lake control.

Conceptual model (Craciun Tables 8–9, Fig. 8):
  states Wait ↔ Irrigation ↔ Drain-out-to-lake
  actions: open the lake gate up to W (lake→pool), outflow pool→lake (O), dispatch
  to fields.

The pool is small (``obs["pool_capacity_m3"]``, 18 × 68 m) next to a day's pumping;
what the station can send is pool + gate + the returns expected back while pumping
(``obs["forecast_return_m3"]``). Above capacity the pool spills to the lake on its
own, so outflow is only needed to draw it down ahead of heavy rain.

Default staff: ``default_station()`` — the most recently registered station from
outside the ``abm`` package (your own script, e.g. ``example.py``), else
the ``mpc`` optimiser (``abm.optimise.MPCStation``). Resolved when a model is
built, so register before creating the session/env.

Farmers may only apply what this agent **sends**. Register outside::

    @abm.register_station("my_staff")
    class MyStaff(abm.StationAgent):
        def decide(self, obs): ...
"""

from __future__ import annotations

import numpy as np

STATIONS: dict[str, type] = {}


DEFAULT_STATION = "mpc"


def register_station(name: str):
    def deco(cls):
        cls.name = name
        STATIONS.pop(name, None)  # re-registering moves it to the most recent slot
        STATIONS[name] = cls
        return cls

    return deco


def _is_builtin(cls) -> bool:
    module = getattr(cls, "__module__", "") or ""
    return module == "abm" or module.startswith("abm.")


def default_station() -> str:
    """The staff agent used when none is named: the latest station registered
    outside the ``abm`` package, otherwise the ``mpc`` optimiser."""
    from . import optimise  # noqa: F401  (registers "mpc")

    custom = [name for name, cls in STATIONS.items() if not _is_builtin(cls)]
    return custom[-1] if custom else DEFAULT_STATION


def make_station(name: str, **kwargs):
    if name not in STATIONS:
        raise KeyError(f"unknown station {name!r}; choose from {sorted(STATIONS)}")
    return STATIONS[name](**kwargs)


def list_stations() -> list[str]:
    return sorted(STATIONS)


class StationAgent:
    """LID irrigation-management staff — single daily decision-maker."""

    name = "station"
    defaults: dict = {}
    bounds: dict = {}
    # Conceptual-model state labels.
    STATES = ("wait", "irrigation", "drain_to_lake")

    def __init__(self, **kwargs):
        merged = dict(getattr(self, "defaults", {}) or {})
        merged.update(kwargs)
        self.params = merged
        self.state = "wait"

    def decide(self, obs: dict) -> dict:
        n = int(obs["n_paddies"])
        return {
            "withdraw_m3": 0.0,
            "outflow_m3": 0.0,
            "irrigation_sent_mm": np.zeros(n),
            "state": self.state,
        }

    def _transition(self, obs: dict) -> str:
        """Table 8 style transitions (adaptive by default)."""
        rain = float(obs.get("rain_mm", 0.0))
        # "Overflow currently occurring": yesterday's pool spill to the lake.
        overflow = float(obs.get("last_spill_m3", 0.0)) > float(self.params.get("overflow_spill_m3", 30_000.0))
        if rain > float(self.params.get("heavy_rain_mm", 25.0)) or overflow:
            return "drain_to_lake"
        deficit = float(obs.get("field_deficit_m3", 0.0))
        if deficit > float(self.params.get("irrigate_trigger_m3", 5_000.0)):
            return "irrigation"
        # Timetable fallback: irrigate most mornings in season.
        day = int(obs.get("day", 0))
        if day % max(int(self.params.get("timetable_every", 1)), 1) == 0:
            return "irrigation"
        return "wait"

    def _gate_m3(self, obs: dict, need_m3: float) -> float:
        """Lake gate for today: what the send needs beyond the pool and the expected
        returns, plus enough to refill the pool, capped by the culvert."""
        short = need_m3 - float(obs["pool_m3"]) - float(obs.get("forecast_return_m3", 0.0))
        refill = float(obs.get("pool_capacity_m3", 0.0))
        return min(max(short, 0.0) + refill, float(self.params.get("max_withdraw_m3", np.inf)))

    def _drain_down(self, obs: dict, fraction: float) -> dict:
        n = int(obs["n_paddies"])
        return {
            "withdraw_m3": 0.0,
            "outflow_m3": min(float(self.params["max_outflow_m3"]), float(obs["pool_m3"]) * fraction),
            "irrigation_sent_mm": np.zeros(n),
            "state": self.state,
        }


@register_station("quota")
class QuotaStation(StationAgent):
    """Fixed daily send depth; top up pool from the lake (fixed timetable style)."""

    name = "quota"
    defaults = dict(
        send_mm=14.0,
        max_withdraw_m3=80_000.0,
        max_outflow_m3=40_000.0,
        overflow_spill_m3=30_000.0,
        heavy_rain_mm=25.0,
        irrigate_trigger_m3=1_000.0,
        timetable_every=1,
    )

    def decide(self, obs: dict) -> dict:
        self.state = self._transition(obs)
        p = self.params
        n = int(obs["n_paddies"])
        area = np.asarray(obs["area_m2"], dtype=float)
        if self.state == "drain_to_lake":
            return self._drain_down(obs, 0.5)
        if self.state == "wait":
            return {
                "withdraw_m3": 0.0,
                "outflow_m3": 0.0,
                "irrigation_sent_mm": np.zeros(n),
                "state": self.state,
            }
        send = np.full(n, float(p["send_mm"]))
        # Match stage demand a bit: send at least median field need when dry.
        need = np.asarray(obs.get("water_needed_mm", np.zeros(n)), dtype=float)
        if need.size == n:
            send = np.maximum(send, np.minimum(need, float(p["send_mm"]) * 1.5))
        need_m3 = float((send * area / 1000.0).sum())
        return {
            "withdraw_m3": self._gate_m3(obs, need_m3),
            "outflow_m3": 0.0,
            "irrigation_sent_mm": send,
            "state": self.state,
        }


@register_station("adaptive")
class AdaptiveStation(StationAgent):
    """Adaptive pump scheduling (Factor 5): send to field deficit + FAO ETc + seepage."""

    name = "adaptive"
    defaults = dict(
        max_send_mm=28.0,
        max_withdraw_m3=100_000.0,
        max_outflow_m3=50_000.0,
        overflow_spill_m3=30_000.0,
        heavy_rain_mm=20.0,
        irrigate_trigger_m3=2_000.0,
        timetable_every=1,
    )

    def decide(self, obs: dict) -> dict:
        self.state = self._transition(obs)
        p = self.params
        n = int(obs["n_paddies"])
        area = np.asarray(obs["area_m2"], dtype=float)
        if self.state == "drain_to_lake":
            return self._drain_down(obs, 0.6)
        if self.state == "wait":
            return {
                "withdraw_m3": 0.0,
                "outflow_m3": 0.0,
                "irrigation_sent_mm": np.zeros(n),
                "state": self.state,
            }
        # Paddy water requirement: refill the deficit + ETc + yesterday's seepage.
        need = np.asarray(obs.get("water_needed_mm", obs.get("field_deficit_mm", np.zeros(n))), dtype=float)
        send = np.minimum(need, float(p["max_send_mm"]))
        # Equity: boost driest quartile slightly.
        if send.size:
            q = np.quantile(send, 0.75)
            send = np.where(send >= q, send * 1.05, send)
        need_m3 = float((send * area / 1000.0).sum())
        return {
            "withdraw_m3": self._gate_m3(obs, need_m3),
            "outflow_m3": 0.0,
            "irrigation_sent_mm": send,
            "state": self.state,
        }


@register_station("level")
class LevelSeekingStation(AdaptiveStation):
    """Alias kept for earlier API — same as adaptive."""

    name = "level"


@register_station("closed")
class ClosedStation(StationAgent):
    """No lake exchange — recycle only what returns to the pool."""

    name = "closed"
    defaults = dict(send_mm=10.0, reserve_m3=500.0)

    def decide(self, obs: dict) -> dict:
        self.state = "irrigation"
        p = self.params
        n = int(obs["n_paddies"])
        area = np.asarray(obs["area_m2"], dtype=float)
        # Gate shut: only the pool and the returns flowing back can be pumped.
        pool = float(obs["pool_m3"]) + float(obs.get("forecast_return_m3", 0.0))
        usable = max(pool - float(p["reserve_m3"]), 0.0)
        need = np.asarray(obs.get("water_needed_mm", np.full(n, p["send_mm"])), dtype=float)
        send = np.minimum(np.maximum(need, 0.0), float(p["send_mm"]) * 1.2)
        cost = float((send * area / 1000.0).sum())
        if cost > usable > 0:
            send = send * (usable / cost)
        elif usable <= 0:
            send = np.zeros(n)
        return {
            "withdraw_m3": 0.0,
            "outflow_m3": 0.0,
            "irrigation_sent_mm": send,
            "state": self.state,
        }

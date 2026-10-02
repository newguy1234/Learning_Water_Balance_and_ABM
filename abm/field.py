# -*- coding: utf-8 -*-
"""Multi-agent field session: a thin, UI/CLI-facing wrapper around the Mesa
``DistrictModel`` (station + farmer mesa.Agents on WaterWorld physics).

Step order (conceptual twin):
  1. Station observes pool/lake/field deficit → withdraw / outflow / send
  2. Farmers observe canal send + crop target → apply ≤ sent, discharge ≤ stock
  3. World updates ponding, FAO Ks/yield, leftover hops, pool returns
"""

from __future__ import annotations

from abm.agents import default_roster

from dataclasses import dataclass, field

from .agents import list_agents
from .metrics import season_efficiencies
from .model import DistrictModel
from .station import list_stations
from .world import SeasonConfig, WaterWorld


@dataclass
class FieldSession:
    year: int = 2022
    days: int | None = None
    seed: int = 0
    paddies_per_farmer: int = 50
    # 0 = every farmer holds exactly paddies_per_farmer ("fixed"); up to 1.0 lets
    # holdings vary between 0.5x and 2x that average instead (scattered
    # across the district with the scenario seed).
    farm_size_variability: float = 0.0
    # Default: built-in archetypes + any farmer you registered (abm.agents.default_roster).
    agent_roster: list[str] = field(default_factory=default_roster)
    # Share of farmers on each roster name; missing names split the remainder evenly.
    roster_weights: dict[str, float] = field(default_factory=dict, repr=False)
    # None → abm.station.default_station(): a user-registered station, else "mpc".
    station_name: str | None = None
    # 0 = replay that year's historical Open-Meteo record exactly; >0 (e.g. 0.3)
    # jitters rain/ET0 day to day so weather differs each seed/reset — an input
    # no farmer or station policy controls. See world.perturb_weather.
    weather_variability: float = 0.0
    # Share of farmers made collaborative (None = each keeps its own cooperation trait).
    collaborative_share: float | None = None
    model: DistrictModel | None = field(default=None, repr=False)
    last_station_state: str = "wait"

    def __post_init__(self):
        self.rebuild()

    def rebuild(self) -> None:
        cfg = SeasonConfig(
            year=self.year,
            days=self.days,
            seed=self.seed,
            paddies_per_farmer=self.paddies_per_farmer,
            farm_size_variability=self.farm_size_variability,
            weather_variability=self.weather_variability,
            collaborative_share=self.collaborative_share,
        )
        self.model = DistrictModel(
            cfg,
            agent_roster=self.agent_roster,
            roster_weights=self.roster_weights,
            station_name=self.station_name,
            seed=self.seed,
        )
        self.days = self.model.world.config.days
        self.agent_roster = self.model.agent_roster
        self.station_name = self.model.station_name

    # --- passthroughs onto the Mesa model ------------------------------------------------------

    @property
    def world(self) -> WaterWorld:
        assert self.model is not None
        return self.model.world

    @world.setter
    def world(self, value: WaterWorld) -> None:
        assert self.model is not None
        self.model.world = value

    @property
    def farmers(self) -> dict[int, object]:
        """Farmer id -> its decision policy (the Mesa agent wrapping it is internal)."""
        assert self.model is not None
        return {fid: agent.policy for fid, agent in self.model.farmer_agents.items()}

    @property
    def station(self):
        assert self.model is not None
        return self.model.station_agent.policy if self.model.station_agent else None

    def _spawn_farmers(self) -> None:
        assert self.model is not None
        self.model._spawn_farmers()
        self.agent_roster = self.model.agent_roster

    def _spawn_station(self) -> None:
        assert self.model is not None
        self.model._spawn_station()
        self.station_name = self.model.station_name

    def set_paddies_per_farmer(self, value: int) -> None:
        assert self.model is not None
        self.paddies_per_farmer = max(1, int(value))
        self.model.set_paddies_per_farmer(self.paddies_per_farmer)

    def set_farm_size_variability(self, value: float) -> None:
        assert self.model is not None
        self.farm_size_variability = max(float(value), 0.0)
        self.model.set_farm_size_variability(self.farm_size_variability)

    def set_weather_variability(self, value: float) -> None:
        """Changes the weather series itself, so this rebuilds the season."""
        self.weather_variability = max(float(value), 0.0)
        self.rebuild()

    def set_roster(self, names: list[str]) -> None:
        assert self.model is not None
        self.model.set_roster(names)
        self.agent_roster = self.model.agent_roster

    def set_roster_weights(self, weights: dict[str, float]) -> None:
        assert self.model is not None
        self.model.set_roster_weights(weights)
        self.roster_weights = dict(self.model.roster_weights)

    def effective_roster_weights(self) -> dict[str, float]:
        assert self.model is not None
        return self.model.effective_roster_weights()

    def set_station(self, name: str, **kwargs) -> None:
        assert self.model is not None
        self.model.set_station(name, **kwargs)
        self.station_name = self.model.station_name

    def set_farmer_agent(self, farmer_id: int, agent_name: str, **kwargs) -> None:
        assert self.model is not None
        self.model.set_farmer_agent(farmer_id, agent_name, **kwargs)

    def reset(self) -> dict:
        assert self.model is not None
        return self.model.reset()

    def step(self):
        assert self.model is not None
        result = self.model.step()
        self.last_station_state = self.model.last_station_state
        return result

    def run_days(self, n: int | None = None) -> dict:
        assert self.model is not None
        w = self.model.world
        remaining = w.config.days - w.day
        steps = remaining if n is None else min(int(n), remaining)
        for _ in range(steps):
            if self.model.done():
                break
            self.step()
        return season_efficiencies(w)

    def run_season(self) -> dict:
        self.reset()
        return self.run_days(None)

    def status(self) -> dict:
        assert self.model is not None
        w = self.model.world
        g = w.today_growth()
        return {
            "day": w.day,
            "days": w.config.days,
            "stage": g.stage,
            "kc": g.kc,
            "target_depth_mm": g.target_depth_mm,
            "n_paddies": w.n,
            "n_farmers": w.n_farmers(),
            "paddies_per_farmer": w.config.paddies_per_farmer,
            "roster": self.model.roster_names(),
            "roster_weights": dict(self.model.roster_weights),
            "station": self.station_name,
            "station_state": self.last_station_state,
            "pool_m3": w.pool_m3,
            "lake_m3": w.lake_m3,
            "mean_sent_mm": float(w.irrigation_sent_mm.mean()) if w.day == 0 else float(w.last_ir.mean()),
            "mean_relative_yield": float(w.relative_yield.mean()),
            "rain_mm": float(w.rain_series[min(w.day, len(w.rain_series) - 1)]),
            "et_mm": float(g.etc_mm),
            "mean_ponding_mm": float(w.ponding_mm.mean()),
            "claim": (w.claim[:80] + "…") if len(w.claim) > 80 else w.claim,
            "n_cyclic": int(w.cyclic.sum()),
            "capture_mean": float(w.capture),
            "learned_ratios": bool(w.learned_ratios),
            "agents_available": list_agents(),
            "stations_available": list_stations(),
        }

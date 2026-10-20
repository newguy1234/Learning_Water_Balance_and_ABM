# -*- coding: utf-8 -*-
"""Gymnasium-style environment for the ABM district: plug in farmers and staff.

    from abm.gym_env import AbmEnv

    env = AbmEnv(year=2022, farmers=["steward", "opportunist"], staff="mpc")
    obs, info = env.reset(seed=0)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step()   # staff agent decides
        done = terminated or truncated
    info["season"]            # Okura WUE' / RUE / LUE, W, O, yield, stress

Farmers: registered names (``abm.list_agents()``), or ``{name: share}`` for a mix.
``collaborative_share`` (0..1) sets that share of farmers to cooperation 0.95 and the rest to 0.05. None keeps each farmer's own cooperation.
Weather: a preset name (``abm.weather.PRESETS``: drought, wet, heatwave, storm,
dry_july, erratic) or an ``abm.weather.WeatherScenario`` you tweak.
Staff: a registered station name or ``StationAgent`` instance, used whenever
``step()`` gets no action. None (default) → ``abm.station.default_station()``: a
station you registered yourself if there is one, otherwise the ``mpc`` optimiser. Pass an action to drive the staff role yourself —
``{"withdraw_m3", "outflow_m3", "irrigation_sent_mm"}`` or a flat array
``[withdraw_m3, outflow_m3, send_mm_paddy_0, ..., send_mm_paddy_n-1]``.

Reward per day, in 10^3 m3 over the district (higher is better):

    − w_lake · W − w_outflow · O − w_stress · Σ stress·A

so it favours reuse over lake water (RUE/LUE up), no pumping to the lake, and
never trading yield for it (stress weighted heavily). ``info["season"]`` carries
the season's Okura efficiencies at termination.

Subclasses ``gymnasium.Env`` and exposes ``observation_space`` / ``action_space``
when Gymnasium is installed (``pip install -e ".[gym]"``); the same API works
without it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .agents import default_roster
from .metrics import season_efficiencies
from .model import DistrictModel
from .station import StationAgent, make_station
from .world import SeasonConfig
from . import optimise  # noqa: F401  (registers the "mpc" staff agent)

try:
    import gymnasium as gym
    from gymnasium import spaces
    _Base = gym.Env
except ImportError:  # pragma: no cover - the API works without gymnasium
    gym = None
    spaces = None
    _Base = object

ENV_ID = "Abm-Imbanuma-v0"
DAY_FEATURES = ("day", "rain_mm", "et0_mm", "rain_tomorrow_mm", "et_demand_mm",
                "target_depth_mm", "pool_m3", "forecast_return_m3")


class AbmEnv(_Base):
    metadata = {"render_modes": [], "name": "AbmEnv",
                "district": "Shiroyama-Jinbee / Imbanuma, Chiba (Okura et al. 2025)"}

    def __init__(self, year: int = 2022, days: int | None = None, seed: int = 0,
                 farmers: list[str] | dict[str, float] | None = None,
                 staff: str | StationAgent | None = None,
                 reward_weights: dict[str, float] | None = None,
                 paddies_per_farmer: int = 50, weather_variability: float = 0.0,
                 weather=None, collaborative_share: float | None = None,
                 config: SeasonConfig | None = None):
        super().__init__()
        self.year = int(year)
        self.days = days
        self.seed_value = int(seed)
        if isinstance(farmers, dict):
            self.roster, self.roster_weights = list(farmers), dict(farmers)
        else:
            self.roster, self.roster_weights = list(farmers or default_roster()), {}
        self.staff = staff
        self.reward_weights = dict(lake=1.0, outflow=1.0, stress=20.0, **(reward_weights or {}))
        self.config = config or SeasonConfig(year=self.year, days=days, seed=self.seed_value,
                                             paddies_per_farmer=paddies_per_farmer,
                                             weather_variability=weather_variability,
                                             weather=weather,
                                             collaborative_share=collaborative_share)
        self.model: DistrictModel | None = None
        self._build(self.seed_value)
        n = self.model.world.n
        if spaces is not None:
            self.observation_space = spaces.Dict({
                "ponding_mm": spaces.Box(0.0, np.inf, (n,), np.float32),
                "topsoil_available_mm": spaces.Box(0.0, np.inf, (n,), np.float32),
                "stress_mm": spaces.Box(0.0, np.inf, (n,), np.float32),
                "seepage_mm": spaces.Box(0.0, np.inf, (n,), np.float32),
                "day": spaces.Box(-np.inf, np.inf, (len(DAY_FEATURES),), np.float32),
            })
            self.action_space = spaces.Box(0.0, np.inf, (n + 2,), np.float32)

    # --- setup ----------------------------------------------------------------------------

    def _build(self, seed: int) -> None:
        self.config.seed = int(seed)
        self.config.days = self.days
        self.model = DistrictModel(self.config, agent_roster=self.roster,
                                   roster_weights=self.roster_weights, seed=int(seed))
        self.set_staff(self.staff)

    def set_staff(self, staff: str | StationAgent) -> None:
        """Swap the staff agent used when step() gets no action."""
        self.staff = staff
        if staff is None:
            from .station import default_station

            staff = default_station()
        if isinstance(staff, str):
            self.model.set_station(staff)
        else:
            self.model.station_agent.policy = staff
        self.model.station_name = getattr(self.model.station_agent.policy, "name", str(staff))

    @property
    def world(self):
        return self.model.world

    # --- gym API --------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed_value = int(seed)
        self._build(self.seed_value)
        return self._obs(), self._info()

    def step(self, action: Any = None):
        world = self.world
        station_action = None if action is None else self._as_station_action(action)
        self.model.step(station_action)
        step = world.history[-1]
        area = world.area_m2
        stress_m3 = float(np.dot(step.stress_mm, area) / 1000.0)
        w = self.reward_weights
        reward = -(w["lake"] * step.withdraw_m3 + w["outflow"] * step.outflow_m3
                   + w["stress"] * stress_m3) / 1000.0
        terminated = world.done()
        info = self._info()
        info.update(withdraw_m3=step.withdraw_m3, outflow_m3=step.outflow_m3, stress_m3=stress_m3,
                    irrigation_m3=float(np.dot(step.irrigation_mm, area) / 1000.0),
                    staff_state=self.model.last_station_state)
        if terminated:
            info["season"] = season_efficiencies(world)
        return self._obs(), float(reward), bool(terminated), False, info

    # --- helpers --------------------------------------------------------------------------

    def _as_station_action(self, action: Any) -> dict:
        if isinstance(action, dict):
            return action
        flat = np.asarray(action, dtype=float).ravel()
        n = self.world.n
        if flat.size != n + 2:
            raise ValueError(f"flat action needs {n + 2} values: withdraw, outflow, then one send per paddy")
        return {"withdraw_m3": flat[0], "outflow_m3": flat[1], "irrigation_sent_mm": flat[2:]}

    def staff_observation(self) -> dict:
        """The full observation a rule-based staff agent sees (``StationAgent.decide``)."""
        return self.world.observe_station()

    def _obs(self) -> dict:
        o = self.world.observe_station()
        day = np.array([o["day"], o["rain_mm"], o["et0_mm"], o["rain_forecast_mm"][1],
                        o["et_demand_forecast_mm"][0], o["target_depth_mm"], o["pool_m3"],
                        o["forecast_return_m3"]], dtype=np.float32)
        return {
            "ponding_mm": o["ponding_mm"].astype(np.float32),
            "topsoil_available_mm": o["topsoil_available_mm"].astype(np.float32),
            "stress_mm": o["stress_mm"].astype(np.float32),
            "seepage_mm": o["seepage_mm"].astype(np.float32),
            "day": day,
        }

    def _info(self) -> dict:
        world = self.world
        weather = getattr(world, "weather", None)
        return {"day": world.day, "n_paddies": world.n, "n_farmers": world.n_farmers(),
                "staff": self.model.station_name, "farmers": self.model.roster_names(),
                "weather": getattr(weather, "name", "historical"),
                "season_rain_mm": float(world.rain_series.sum()),
                "season_et0_mm": float(world.et0_series.sum())}


if gym is not None:
    try:
        gym.register(id=ENV_ID, entry_point="abm.gym_env:AbmEnv")
    except Exception:  # pragma: no cover - already registered
        pass

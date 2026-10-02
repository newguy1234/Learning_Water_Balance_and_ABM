# -*- coding: utf-8 -*-
"""The ABM's world: stem's water world plus who owns which paddy and what agents see.

The physics, weather, pool and station hardware live in ``stem.field.Field`` (a
mesa.Model of the paddies). WaterWorld adds farm holdings and the observations the
farmer and station agents act on; abm.DistrictModel adds the agents themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stem.field import Field, FieldConfig, StepResult, perturb_weather  # noqa: F401  (re-exported)

from .landscape import assign_owners_along_canal_variable


@dataclass
class SeasonConfig(FieldConfig):
    """FieldConfig plus who owns the paddies and how they cooperate."""

    paddies_per_farmer: int = 50
    # Ownership is scattered reproducibly across the district. 0 = every farmer holds
    # exactly paddies_per_farmer ("fixed"); up to 1.0 lets holdings vary between
    # 0.5x and 2x that average instead of all being identical.
    farm_size_variability: float = 0.0
    # Which paddies each farmer owns. None (default) = a fresh random ownership every time the
    # world is built or reset; the seed drawn is kept as WaterWorld.ownership_seed_used so a
    # run can be repeated by passing it back here. An int = that fixed ownership.
    ownership_seed: int | None = None
    # Collaboration scenario: this share (0..1) of all farmers, whatever their type, is made
    # collaborative (cooperation 0.95, levels its water with the neighbours) and the rest
    # uncollaborative (0.05, keeps its own level). None = every farmer keeps its own trait.
    collaborative_share: float | None = None


def assign_owners(n: int, paddies_per_farmer: int) -> np.ndarray:
    if paddies_per_farmer <= 0 or paddies_per_farmer >= n:
        return np.zeros(n, dtype=int)
    owners = np.zeros(n, dtype=int)
    farmer = 0
    for start in range(0, n, paddies_per_farmer):
        owners[start : start + paddies_per_farmer] = farmer
        farmer += 1
    return owners



class WaterWorld(Field):
    """Field + farm holdings + agent observations."""

    def __init__(self, config: SeasonConfig | None = None, seed=None):
        self.config = config or SeasonConfig()
        super().__init__(self.config, seed=seed)
        self.owner_id = assign_owners_along_canal_variable(
            np.arange(self.n), self.n, self.config.paddies_per_farmer,
            self.config.farm_size_variability, self._draw_ownership_seed())
        self.row = np.argsort(np.argsort(self.lat))
        self.col = np.argsort(np.argsort(self.lon))
        self.nrows = self.ncols = int(np.ceil(np.sqrt(self.n)))

    def _draw_ownership_seed(self) -> int:
        """The config's ownership_seed, or a fresh random one (remembered for replay)."""
        seed = self.config.ownership_seed
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        self.ownership_seed_used = int(seed)
        return self.ownership_seed_used

    def set_paddies_per_farmer(self, value: int) -> None:
        self.config.paddies_per_farmer = int(value)
        self._reassign_owners()

    def set_farm_size_variability(self, value: float) -> None:
        self.config.farm_size_variability = max(float(value), 0.0)
        self._reassign_owners()

    def _reassign_owners(self) -> None:
        self.owner_id = assign_owners_along_canal_variable(
            np.arange(self.n), self.n, self.config.paddies_per_farmer,
            self.config.farm_size_variability, self._draw_ownership_seed(),
        )

    def n_farmers(self) -> int:
        return int(self.owner_id.max()) + 1 if self.n else 0

    def paddies_of(self, farmer_id: int) -> np.ndarray:
        return np.where(self.owner_id == farmer_id)[0]

    def neighbour_ids(self, paddy: int) -> list[int]:
        return list(self.neighbours[int(paddy)])

    def neighbour_view(self, paddy: int) -> dict:
        """Centre paddy + cardinal neighbours (cyclic edges), not a fake 3×3 grid."""
        paddy = int(paddy)
        ids = [paddy] + self.neighbour_ids(paddy)
        pond = {i: float(self.ponding_mm[i]) for i in ids}
        sent = {i: float(self.irrigation_sent_mm[i]) for i in ids}
        grid_pond = np.full((3, 3), np.nan)
        grid_own = np.full((3, 3), -1, dtype=int)
        grid_ids = np.full((3, 3), -1, dtype=int)
        grid_pond[1, 1] = self.ponding_mm[paddy]
        grid_own[1, 1] = self.owner_id[paddy]
        grid_ids[1, 1] = paddy
        # Place cardinal neighbours in NSEW cells when possible.
        dirs = {
            "north": (0, 1),
            "south": (2, 1),
            "west": (1, 0),
            "east": (1, 2),
        }
        for col, name in enumerate(("east", "west", "north", "south")):
            j = int(self.neighbour_index[paddy, col])
            if j >= 0:
                rr, cc = dirs[name]
                grid_pond[rr, cc] = self.ponding_mm[j]
                grid_own[rr, cc] = self.owner_id[j]
                grid_ids[rr, cc] = j
        return {
            "paddy": paddy,
            "ponding": grid_pond,
            "owners": grid_own,
            "ids": grid_ids,
            "sent": sent,
            "pond_map": pond,
            "own_owner": int(self.owner_id[paddy]),
            "cardinal": self.neighbour_ids(paddy),
            "lon": float(self.lon[paddy]),
            "lat": float(self.lat[paddy]),
        }

    def reset(self) -> dict:
        self.set_ratios(self.config.ratios)
        self.reset_season()
        self._reassign_owners()
        return self.observe()

    def observe(self) -> dict:
        g = self.today_growth()
        i = min(self.day, len(self.rain_series) - 1)
        return {
            "day": self.day,
            "rain_mm": float(self.rain_series[i]),
            "et_mm": float(g.etc_mm),
            "et0_mm": float(self.et0_series[i]),
            "kc": float(g.kc),
            "stage": g.stage,
            "target_depth_mm": float(g.target_depth_mm),
            "ponding_mm": self.ponding_mm.copy(),
            "soil_mm": self.soil_mm.copy(),
            # Yesterday's pond loss other than ET and chosen discharge (plow-pan
            # percolation + outlet leak) — what a farmer reads off the falling water.
            "seepage_mm": self.last_seepage.copy(),
            # Yesterday's unmet weather-driven ET (mm) and the season total so far.
            "stress_mm": self.last_stress.copy(),
            "stress_cum_mm": self.stress_cum.copy(),
            "block": self.block.copy(),
            "area_m2": self.area_m2.copy(),
            "cyclic": self.cyclic.copy(),
            "capture": self.capture,
            "neighbour": self.neighbour,
            "stay": self.stay,
            "capture_i": self.lateral_capture_i.copy(),
            "neighbour_i": self.neighbour_i.copy(),
            "stay_i": self.stay_i.copy(),
            "year": self.config.year,
            "owner_id": self.owner_id.copy(),
            "lon": self.lon.copy(),
            "lat": self.lat.copy(),
            "row": self.row.copy(),
            "col": self.col.copy(),
            "n_paddies": self.n,
            "neighbours": [list(n) for n in self.neighbours],
            "hop_in_mm": self.last_hop_in.copy(),
            "irrigation_sent_mm": self.irrigation_sent_mm.copy(),
            "last_irrigation_mm": self.last_ir.copy(),
            "last_drainage_mm": self.last_dr.copy(),
            "n_farmers": self.n_farmers(),
            "paddies_per_farmer": int(self.config.paddies_per_farmer),
            "pool_m3": float(self.pool_m3),
            "pool_capacity_m3": float(self.pool_capacity_m3),
            # Yesterday's cyclic D + recaptured lateral flow: the station's forecast of
            # what will flow back into the pool while it pumps today.
            "forecast_return_m3": float(self.last_return_m3),
            "last_spill_m3": float(self.last_spill_m3),
            "lake_m3": float(self.lake_m3),
            "last_withdraw_m3": float(self.last_withdraw_m3),
            "last_outflow_m3": float(self.last_outflow_m3),
            "ks": self.ks.copy(),
            "relative_yield": self.relative_yield.copy(),
            "mean_relative_yield": float(self.relative_yield.mean()),
            "claim": self.claim,
            "pool_distance_m": self.pool_distance_m.copy(),
            "learned_ratios": bool(self.learned_ratios),
        }

    def observe_station(self) -> dict:
        obs = self.observe()
        obs["role"] = "station"
        target = float(obs["target_depth_mm"])
        # Every paddy can draw from the pool now, cyclic or not — the station only
        # rations by pool_distance_m (see apply_station) when supply falls short.
        deficit = np.maximum(target - self.ponding_mm, 0.0)
        obs["field_deficit_mm"] = deficit
        obs["field_deficit_m3"] = float((deficit * self.area_m2 / 1000.0).sum())
        obs["water_needed_mm"] = deficit + float(obs["et_mm"]) + self.last_seepage
        # Short weather forecast (today first) and the pond physics a staff model may use.
        horizon = 3
        days = [min(self.day + k, len(self.rain_series) - 1) for k in range(horizon)]
        obs["rain_forecast_mm"] = np.array([self.rain_series[d] for d in days], dtype=float)
        obs["et_demand_forecast_mm"] = np.array([self.et_demand_mm(self.day + k) for k in range(horizon)])
        obs["topsoil_available_mm"] = self.soil_mm.copy()
        # Drainage already on its way to the pool (arrives today and later).
        obs["drain_transit_m3"] = self.drain_transit_m3
        cap = self.config.pump_capacity_m3_s
        obs["pump_capacity_m3"] = float(cap * 86_400.0) if cap else float("inf")
        if self.canal_depth_m is not None:
            d = min(self.day, len(self.canal_depth_m["North"]) - 1)
            obs["canal_depth_m"] = {arm: float(v[d]) for arm, v in self.canal_depth_m.items()}
        # Share of the pond each paddy keeps overnight (1 − kz) and the bund height.
        obs["pond_physics"] = {"keep": 1.0 - self.ratios.kz, "bund_mm": float(self.bund_height_mm)}
        return obs

    def observe_farmer(self, farmer_id: int) -> dict:
        obs = self.observe()
        owned = self.paddies_of(farmer_id)
        visible = set(owned.tolist())
        for p in owned:
            visible.update(self.neighbour_ids(int(p)))
        visible = sorted(visible)
        obs["role"] = "farmer"
        obs["farmer_id"] = int(farmer_id)
        obs["owned_ids"] = owned
        obs["visible_ids"] = np.array(visible, dtype=int)
        obs["owned_ponding_mm"] = self.ponding_mm[owned].copy()
        obs["owned_irrigation_sent_mm"] = self.irrigation_sent_mm[owned].copy()
        obs["owned_ks"] = self.ks[owned].copy()
        obs["owned_relative_yield"] = self.relative_yield[owned].copy()
        obs["owned_seepage_mm"] = self.last_seepage[owned].copy()
        obs["owned_stress_mm"] = self.last_stress[owned].copy()
        obs["visible_ponding_mm"] = self.ponding_mm[visible].copy()
        obs["owned_area_m2"] = self.area_m2[owned].copy()
        # How far each owned paddy's drainage travels to the pool (entry + along channels).
        obs["owned_drain_route_m"] = (self.drain_entry_m[owned] + self.drain_path_m[owned]).copy()
        # Yesterday's share of each owned paddy's drainage the high canal held back.
        obs["owned_backwater"] = self.last_backwater_frac[owned].copy()
        obs["owned_calendar_shift_days"] = self.calendar_shift_days[owned].copy()
        obs["owned_et_demand_mm"] = self.et_demand_for(self.day, obs["owned_calendar_shift_days"])
        date = self.season_date(self.day)
        obs["date"] = str(date.date())
        obs["weekday"] = int(date.weekday())  # 0 Monday .. 6 Sunday
        # Rain forecast, today first (the farmer decides how far ahead to trust it).
        obs["rain_forecast_mm"] = np.array(
            [self.rain_series[min(self.day + k, len(self.rain_series) - 1)] for k in range(6)], dtype=float)
        obs["bund_mm"] = float(self.bund_height_mm)
        obs["neighbour_mean_mm"] = np.array(
            [
                float(np.mean(self.ponding_mm[self.neighbours[int(p)]]))
                if self.neighbours[int(p)]
                else float(self.ponding_mm[int(p)])
                for p in owned
            ]
        )
        target = float(obs["target_depth_mm"])
        obs["owned_water_needed_mm"] = np.maximum(
            target - obs["owned_ponding_mm"] + float(obs["et_mm"]) + obs["owned_seepage_mm"], 0.0
        )
        return obs

    def grid(self, values: np.ndarray) -> np.ndarray:
        """Legacy square reshape — prefer lon/lat scatter for cyclic maps."""
        side = int(np.ceil(np.sqrt(self.n)))
        out = np.full((side, side), np.nan)
        flat = np.asarray(values, dtype=float).ravel()
        out.ravel()[: flat.size] = flat[: side * side]
        return out

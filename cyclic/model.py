"""The district as a Mesa model: a PaddyAgent per paddy, one PumpingStation, a day a step.

    from cyclic import CyclicModel

    results = CyclicModel(2021, agents="et_follower").run()
    results.blocks                        # daily per block: m3 and mm, as Okura et al. measure
    results.station                       # daily W, O, LFV_pump, pool storage
    results.paddy_season()                # every paddy's season: irrigation, discharge, ET vs target ...
    results.paddy_series("N00012")        # one paddy's days: depth, irrigation, drainage, ET vs its satellite
    results.paddy_daily("irrigation_mm")  # dates x paddies

A DAY
  1. every farmer sees its Observation and asks for (irrigation_mm, discharge_mm)
  2. the station serves the irrigation (all of it, unless a lake allowance binds)
  3. physics.advance moves the water on every plot
     4. cyclic D_North and D_South enter the pool (Table 5); LFV_pump (Eq. 7 share of
        leftover) enters; IR_all is pumped to North and South (cyclic + ncb); W fills a
        shortfall from the lake; O sends surplus to the lake (Eq. 4). Non-cyclic D does
        not enter the pool.
  5. the station settles the pool: lake withdrawal W for what the returns did not
     cover, outflow O for what the pool could not hold (Okura et al. Eq. 4)
  6. the day is recorded per paddy and summed per block, cyclic and non-cyclic apart
"""

from __future__ import annotations

from functools import cached_property
import os
import re

import mesa
import numpy as np
import pandas as pd

from . import data, okura
from .agents import PaddyAgent, PumpingStation, build_batch, resolve_policies
from .physics import Physics, State, advance
from .account import apply_assignment, force_et, solve_omalley
from .canal import apply_canal, divert_recapture_to_unsaturated
from .lateral import cardinal_edges

BLOCKS = okura.BLOCKS
GROUPS = ("cyclic", "non_cyclic")
FLUXES = ("rain", "irrigation", "requested", "drainage", "farmer_discharge", "passive_drainage",
          "lateral_sim", "leftover", "deep", "et", "et_potential", "infiltration", "neighbor")
STATION = ("W_m3", "W_policy_m3", "O_m3", "LFV_pump_m3", "LFV_eq6_m3",
           "D_North_m3", "D_South_m3", "D_pool_m3",
           "IR_all_North_m3", "IR_all_South_m3", "IR_all_m3", "IR_ncb_m3",
           "pool_m3", "d_pool_m3", "D_ncb_m3", "ncb_stock_m3")
PADDY_LOG = ("depth_mm", "requested_mm", "irrigation_mm", "farmer_discharge_mm",
             "passive_drainage_mm", "drainage_mm", "lateral_mm", "deep_mm", "et_mm",
             "et_potential_mm", "infiltration_mm", "neighbor_mm", "et_pond_mm",
             "et_surface_mm", "et_root_mm", "et_groundwater_mm", "leftover_mm",
             "stay_mm", "ponded", "assigned_lateral", "pond_mm", "topsoil_mm",
             "need_mm", "need_stock_mm", "canal_mm", "canal_out_mm", "drain_stock_mm",
             "transpiration_mm", "evaporation_mm",
             "flow_kx_mm", "flow_ky_mm", "flow_kz_mm", "flow_ground_mm", "lfv_mm")
CELL_LOG = ("lateral_mm", "infiltration_mm", "deep_mm", "drainage_mm", "neighbor_mm",
            "et_mm", "et_pond_mm", "et_surface_mm", "et_root_mm", "et_groundwater_mm",
            "leftover_mm", "stay_mm", "ponded", "pond_mm", "topsoil_mm",
            "need_mm", "need_stock_mm", "canal_mm", "canal_out_mm", "drain_stock_mm",
            "transpiration_mm", "evaporation_mm",
            "flow_kx_mm", "flow_ky_mm", "flow_kz_mm", "flow_ground_mm", "lfv_mm")


def _daily_columns():
    columns = ["date", *STATION]
    for block in BLOCKS:
        for group in GROUPS:
            prefix = f"{block}|{group}|"
            columns += [prefix + "area_m2", prefix + "storage_m3"]
            columns += [f"{prefix}{q}_m3" for q in FLUXES]
    return columns


DAILY_COLUMNS = _daily_columns()


def _reporter(key):
    return lambda model: model.today[key]


class CyclicModel(mesa.Model):
    """
    year              season to run (satellite imagery on disk covers 2021)
    agents            who farms: a registered name, FarmerPolicy class or instance, a dict
                      {"North": ..., "South": ...}, or a function (index, inputs) -> spec
    physics           Physics constants; defaults are the Okura-scaled ones
    forcing           "open_meteo" rain (default) or "station" rain (Table 2 monthly totals)
    potential         "pixel" (default): each paddy's satellite-derived canopy and ET target;
                      "field": block-average canopy. Actual ET responds to local water stores.
    lake_allowance_m3 cap on the season's withdrawal from the lake; None is unlimited
    record_paddies    keep every paddy's day (needed for paddy_* results and cloning);
                      switch off for fast rollouts inside a learner
    record_cells      keep a daily log on every satellite cell. Default is off when there
                      are more than 20k cells (would be ~1.5 GB). CSV cell export can
                      turn it on explicitly.
    k_x, k_y, k_z     unused when force_satellite_et (no Darcy). Optional per-paddy K for the
                      older Darcy path only.
    """

    def __init__(self, year=okura.COMPARISON_YEAR, agents="et_follower", physics=None, forcing="open_meteo",
                 lake_allowance_m3=None, record_paddies=True, initial_depth_mm=30.0, seed=0,
                 potential="pixel", k_x=None, k_y=None, k_z=None, save_csv=True, span="season",
                 record_cells=None):
        super().__init__(rng=seed)
        self.year = year
        self.span = span
        self.hard_surface = False
        self.physics = physics or Physics()
        self.inputs = inputs = data.season_inputs(year, forcing, potential, span)
        self.n = len(inputs.paddy_id)
        self.n_days = len(inputs.dates)
        self.area = inputs.area_m2
        self.day = 0
        self.state = State.start(self.n, initial_depth_mm)
        from .spatial import cells
        self.cells = cells(year, forcing, span) if potential == "pixel" else None
        self.cell_state = (State.start(len(self.cells.owner), initial_depth_mm)
                           if self.cells is not None else None)
        n_cells = 0 if self.cells is None else len(self.cells.owner)
        if record_cells is None:
            record_cells = bool(record_paddies) and n_cells > 0 and n_cells <= 20000
        self.record_cells = bool(record_cells) and self.cells is not None
        self.cell_et_mm = (np.zeros((self.n_days, n_cells), dtype=np.float32)
                           if self.record_cells else None)
        self.cell_log = ({k: np.zeros((self.n_days, n_cells), dtype=np.float32)
                          for k in CELL_LOG}
                         if self.record_cells else {})
        self.k_x = None if k_x is None else np.asarray(k_x, dtype=float)
        self.k_y = None if k_y is None else np.asarray(k_y, dtype=float)
        self.k_z = None if k_z is None else np.asarray(k_z, dtype=float)
        lon = inputs.frame["lon"].to_numpy()
        lat = inputs.frame["lat"].to_numpy()
        self.paddy_edges = cardinal_edges(lon, lat)

        self.policy_of, self.groups = resolve_policies(agents, inputs)
        self.paddies = [
            PaddyAgent(self, i, inputs.paddy_id[i], inputs.block[i], bool(inputs.cyclic[i]),
                       float(self.area[i]), self.policy_of[i])
            for i in range(self.n)
        ]
        # Station before policy.reset so FieldAggregate can pin Fig. 8 W / hard_surface.
        self.station = PumpingStation(self, self.physics.pool_capacity_m3, lake_allowance_m3)
        for policy, rows in self.groups:
            policy.reset(self, rows)

        self.request_irrigation = np.zeros(self.n)
        self.request_discharge = np.zeros(self.n)
        self.observations = None
        # The last day's per-paddy water movements, in mm -- what the environment rewards on.
        self.last_flux = {}
        self.last_irrigation = np.zeros(self.n)
        self.record_paddies = record_paddies
        self.save_csv = bool(save_csv)
        self.log = ({k: np.zeros((self.n_days, self.n), dtype=np.float32) for k in PADDY_LOG}
                    if record_paddies else {})
        self._weights = {
            (block, group): np.where((inputs.block == block) & (inputs.cyclic == (group == "cyclic")),
                                     self.area, 0.0)
            for block in BLOCKS for group in GROUPS
        }
        self._cyclic_area = np.where(inputs.cyclic, self.area, 0.0)
        self._ncb_area = np.where(~inputs.cyclic, self.area, 0.0)
        # Largest per-plot error in the storage identity seen so far, mm. Should stay ~1e-9.
        self.balance_error_mm = 0.0
        self.inverse = None
        self.today = {}
        self.datacollector = mesa.DataCollector(
            model_reporters={column: _reporter(column) for column in DAILY_COLUMNS})

    def _conductivity(self, owner=None):
        def expand(values):
            if values is None:
                return None
            values = np.asarray(values, dtype=float)
            if values.ndim == 0 or values.size == 1:
                return values
            return values[owner] if owner is not None else values
        return expand(self.k_z), expand(self.k_x), expand(self.k_y)

    def step(self):
        self.advance()

    def _observe(self):
        p, n = self.physics, self.n
        flux = self.last_flux or {}
        zeros = np.zeros(n)
        return build_batch(
            self.inputs, self.day, self.state.water.copy(), self.area,
            pond_mm=self.state.water.copy(),
            topsoil_mm=np.maximum(p.total_evaporable_mm - self.state.surface, 0.0),
            need_mm=np.asarray(flux.get("need_mm", zeros), dtype=float),
            need_stock_mm=self.state.need.copy(),
            leftover_mm=np.asarray(flux.get("leftover_mm", zeros), dtype=float),
            canal_mm=np.asarray(flux.get("canal_mm", zeros), dtype=float),
            drain_stock_mm=self.state.drain.copy(),
            canal_distance_m=np.asarray(getattr(self.inputs, "canal_distance_m", zeros), dtype=float),
            farm_drain_mm=np.full(n, p.farm_drain_stage_mm),
            last_discharge_mm=np.asarray(flux.get("discharge_mm", flux.get("drainage_mm", zeros)), dtype=float),
            last_et_mm=np.asarray(flux.get("et_mm", zeros), dtype=float),
            last_lfv_mm=np.asarray(flux.get("lfv_mm", zeros), dtype=float),
            **self._neighbour_heads(flux),
        )

    def _neighbour_heads(self, flux=None):
        from .lateral import cardinal_features, neighbour_et_features, neighbour_head_features
        edges = getattr(self, "paddy_edges", None)
        depth = self.state.water
        mean, drop = neighbour_head_features(depth, edges)
        et = np.zeros(depth.size) if flux is None else np.asarray(
            flux.get("et_mm", np.zeros(depth.size)), dtype=float).ravel()
        if et.size != depth.size:
            et = np.zeros(depth.size)
        et_mean, et_drop = neighbour_et_features(et, edges)
        water, present = cardinal_features(depth, edges)
        et_nbr, et_ok = cardinal_features(et, edges)
        out = {
            "neighbour_depth_mm": mean,
            "neighbour_head_drop_mm": drop,
            "neighbour_et_mm": et_mean,
            "neighbour_et_drop_mm": et_drop,
        }
        labels = ("right", "left", "up", "down")
        for col, label in enumerate(labels):
            out[f"neighbour_{label}_mm"] = water[:, col]
            out[f"neighbour_{label}_et_mm"] = np.where(et_ok[:, col], et_nbr[:, col], 0.0)
        return out

    def _finish_flux(self, flux, state, dist_m, edges, area_m2, cyclic=None):
        if self.physics.canal_routing:
            n_here = int(np.asarray(area_m2).size)
            shares = getattr(self, "pixel_shares", None)
            daily = getattr(self, "pixel_shares_daily", None)
            if daily is not None and "kx_ratio" in daily:
                day = int(np.clip(self.day, 0, np.asarray(daily["kx_ratio"]).shape[0] - 1))
                if np.asarray(daily["kx_ratio"]).shape[-1] == n_here:
                    shares = {key: np.asarray(val)[day] for key, val in daily.items()
                              if np.ndim(val) == 2 and np.asarray(val).shape[-1] == n_here}
            if shares is not None and np.size(shares.get("kx_ratio", [])) != n_here:
                shares = None
            leftover_split = bool(
                getattr(self.physics, "ratio_split", False)
                or getattr(self, "pixel_shares", None) is not None
                or self.physics.force_satellite_et
            )
            apply_canal(flux, dist_m, edges, area_m2, self.physics, drain_stock=state.drain,
                        leftover_split=leftover_split, cyclic=cyclic,
                        shares=shares)
            if leftover_split:
                divert_recapture_to_unsaturated(
                    flux, state.surface, state.deficit, edges, area_m2, cyclic, self.physics)
        kz = np.asarray(flux.get("flow_kz_mm", 0.0), dtype=float)
        if np.size(kz) == state.surface.size:
            take = np.minimum(np.maximum(kz, 0.0), np.maximum(state.surface, 0.0))
            state.surface -= take
            extra = np.maximum(kz - take, 0.0)
            flux["flow_kz_mm"] = take
            if extra.any():
                stay = np.asarray(flux.get("stay_mm", np.zeros(state.surface.size)), dtype=float)
                flux["stay_mm"] = stay + extra
                flux["deep_mm"] = flux.get("deep_mm", stay) + extra
                flux["flow_ground_mm"] = np.asarray(
                    flux.get("flow_ground_mm", np.zeros(state.surface.size)), dtype=float) + extra
        need = np.asarray(flux.get("need_mm", 0.0), dtype=float)
        if np.size(need) == state.need.size:
            state.need += need
            flux["need_stock_mm"] = state.need.copy()
        return flux

    def advance(self, irrigation=None, discharge=None):
        """One day. With no arguments every paddy's own policy decides. Pass arrays of mm
        (one per paddy, or a scalar for all) to act for every paddy from outside instead --
        which is what CyclicEnv does."""
        if not self.running:
            return
        d, inputs, p = self.day, self.inputs, self.physics

        depth = self.state.water.copy()
        self.observations = self._observe()
        self.request_irrigation[:] = 0.0
        self.request_discharge[:] = 0.0
        if irrigation is not None or discharge is not None:
            self.request_irrigation[:] = np.broadcast_to(
                np.asarray(0.0 if irrigation is None else irrigation, dtype=float), (self.n,))
            self.request_discharge[:] = np.broadcast_to(
                np.asarray(0.0 if discharge is None else discharge, dtype=float), (self.n,))
        else:
            for policy, rows in self.groups:
                if policy.batched:
                    wanted_in, wanted_off = policy.decide_batch(self.observations.subset(rows))
                    self.request_irrigation[rows] = wanted_in
                    self.request_discharge[rows] = wanted_off
                else:
                    for i in rows:
                        self.paddies[i].decide()

        requested = np.clip(np.nan_to_num(self.request_irrigation), 0.0, None)
        wanted_out = np.clip(np.nan_to_num(self.request_discharge), 0.0, None)
        # IR_all ≤ pool + same-day cyclic D + LFV + lake − planned O (Eq. 4).
        # Protect ET-cover demand: only scale the free (above-cover) slice.
        cyclic = np.asarray(inputs.cyclic, dtype=bool)
        d_cyc_m3 = float((wanted_out[cyclic] * self.area[cyclic]).sum() / 1000.0) if np.any(cyclic) else 0.0
        lfv_m3 = 0.0
        if getattr(self, "lfv_plan_m3", None) is not None:
            lfv_m3 = float(np.asarray(self.lfv_plan_m3, dtype=float)[
                min(d, len(self.lfv_plan_m3) - 1)])
        o_m3 = 0.0
        if getattr(self, "o_plan_m3", None) is not None:
            o_m3 = float(np.asarray(self.o_plan_m3, dtype=float)[
                min(d, len(self.o_plan_m3) - 1)])
        from . import okura as _okura
        rain_d = np.asarray(inputs.rain_mm[d], dtype=float)
        er_d = _okura.effective_rain_mm(rain_d)
        et_d = np.asarray(inputs.et_target_mm[d], dtype=float)
        depth_d = np.asarray(self.state.water, dtype=float)
        cover_mm = np.maximum(et_d - er_d - np.maximum(depth_d, 0.0), 0.0)
        cover_req = np.minimum(requested, cover_mm)
        free_req = np.maximum(requested - cover_req, 0.0)
        cover_m3 = float(cover_req @ self.area) / 1000.0
        free_m3 = float(free_req @ self.area) / 1000.0
        avail = None if getattr(self, "hard_surface", False) else self.station.available_irrigation_m3(
            cyclic_d_m3=d_cyc_m3, lfv_m3=lfv_m3, o_m3=o_m3)
        if avail is None:
            irrigation = requested
        else:
            serve_cover = min(cover_m3, float(avail))
            remain = max(float(avail) - serve_cover, 0.0)
            serve_free = min(free_m3, remain)
            cover_scale = (serve_cover / cover_m3) if cover_m3 > 1e-12 else 1.0
            free_scale = (serve_free / free_m3) if free_m3 > 1e-12 else 1.0
            irrigation = cover_req * cover_scale + free_req * free_scale

        before = self.state.storage()
        if p.force_satellite_et:
            if self.cells is None:
                flux = force_et(self.state, inputs.rain_mm[d], inputs.et_target_mm[d],
                                irrigation, wanted_out, p,
                                et0_mm=inputs.et0_mm[d], canopy=inputs.fc_daily[d])
                self._finish_flux(flux, self.state, inputs.canal_distance_m, self.paddy_edges, self.area,
                                  cyclic=inputs.cyclic)
            else:
                from .spatial import aggregate
                grid = self.cells
                cell_area = self.area[grid.owner] * grid.weights
                paddy_et = inputs.et_target_mm[d, grid.owner]
                observed = grid.observed_et_mm[d]
                cell_et = np.where(np.isfinite(observed), observed, paddy_et)
                cell_flux = force_et(self.cell_state, inputs.rain_mm[d],
                                     cell_et,
                                     irrigation[grid.owner], wanted_out[grid.owner], p,
                                     et0_mm=inputs.et0_mm[d], canopy=grid.canopy[d])
                self._finish_flux(cell_flux, self.cell_state, grid.canal_distance_m, grid.edges, cell_area,
                                  cyclic=inputs.cyclic[grid.owner])
                flux = {key: aggregate(value, grid, self.n) for key, value in cell_flux.items()
                        if np.ndim(value) == 1 and np.size(value) == len(grid.owner)}
                for key in ("water", "surface", "deficit", "perched", "drain"):
                    getattr(self.state, key)[:] = aggregate(getattr(self.cell_state, key), grid, self.n)
                self.state.need[:] = aggregate(self.cell_state.need, grid, self.n)
                flux["need_stock_mm"] = self.state.need.copy()
                if self.cell_et_mm is not None:
                    self.cell_et_mm[d] = cell_flux["et_mm"]
                if self.cell_log:
                    for key in CELL_LOG:
                        if key in cell_flux:
                            self.cell_log[key][d] = cell_flux[key]
        elif self.cells is None:
            kz, kx, ky = self._conductivity()
            flux = advance(self.state, inputs.rain_mm[d], inputs.et0_mm[d], inputs.fc_daily[d],
                           irrigation, wanted_out, p, k_z=kz, k_x=kx, k_y=ky,
                           edges=self.paddy_edges, area_m2=self.area)
            self._finish_flux(flux, self.state, inputs.canal_distance_m, self.paddy_edges, self.area,
                              cyclic=inputs.cyclic)
        else:
            from .spatial import aggregate
            grid = self.cells
            cell_before = self.cell_state.storage()
            kz, kx, ky = self._conductivity(grid.owner)
            cell_area = self.area[grid.owner] * grid.weights
            cell_flux = advance(self.cell_state, inputs.rain_mm[d], inputs.et0_mm[d],
                                grid.canopy[d], irrigation[grid.owner], wanted_out[grid.owner], p,
                                k_z=kz, k_x=kx, k_y=ky, edges=grid.edges, area_m2=cell_area)
            self._finish_flux(cell_flux, self.cell_state, grid.canal_distance_m, grid.edges, cell_area,
                              cyclic=inputs.cyclic[grid.owner])
            cell_residual = (self.cell_state.storage() - cell_before) - (
                inputs.rain_mm[d] + irrigation[grid.owner] - cell_flux["et_mm"]
                - cell_flux["drainage_mm"] - cell_flux["lateral_mm"] - cell_flux["deep_mm"]
                + cell_flux["neighbor_mm"])
            self.balance_error_mm = max(self.balance_error_mm, float(np.abs(cell_residual).max()))
            flux = {key: aggregate(value, grid, self.n) for key, value in cell_flux.items()
                    if np.ndim(value) == 1 and np.size(value) == len(grid.owner)}
            for key in ("water", "surface", "deficit", "perched", "drain"):
                getattr(self.state, key)[:] = aggregate(getattr(self.cell_state, key), grid, self.n)
            self.state.need[:] = aggregate(self.cell_state.need, grid, self.n)
            flux["need_stock_mm"] = self.state.need.copy()
            if self.cell_et_mm is not None:
                self.cell_et_mm[d] = cell_flux["et_mm"]
            if self.cell_log:
                for key in CELL_LOG:
                    if key in cell_flux:
                        self.cell_log[key][d] = cell_flux[key]
        passive = flux["overflow_mm"] + flux["spill_mm"]
        if not p.force_satellite_et:
            residual = (self.state.storage() - before) - (
                inputs.rain_mm[d] + irrigation - flux["et_mm"] - flux["drainage_mm"]
                - flux["lateral_mm"] - flux["deep_mm"] + flux["neighbor_mm"])
            self.balance_error_mm = max(self.balance_error_mm, float(np.abs(residual).max()))
        self.last_flux = flux
        self.last_irrigation = irrigation

        d_mm = flux.get("canal_out_mm", flux["drainage_mm"])
        if getattr(self, "lfv_plan_m3", None) is not None:
            lfv_m3 = float(np.asarray(self.lfv_plan_m3, dtype=float)[
                min(d, len(self.lfv_plan_m3) - 1)])
        else:
            lfv_m3 = float(np.asarray(
                flux.get("lfv_mm",
                         p.pool_capture_share * np.asarray(flux.get("leftover_mm", flux["lateral_mm"]), dtype=float)),
                dtype=float) @ self._cyclic_area) / 1000.0
        o_force = None
        if getattr(self, "o_plan_m3", None) is not None:
            o_force = float(np.asarray(self.o_plan_m3, dtype=float)[
                min(d, len(self.o_plan_m3) - 1)])
        w_force = None
        if getattr(self, "w_plan_m3", None) is not None:
            w_force = float(np.asarray(self.w_plan_m3, dtype=float)[
                min(d, len(self.w_plan_m3) - 1)])
        self.station.settle(
            IR_North_all_m3=float(irrigation @ np.where(inputs.block == "North", self.area, 0.0)) / 1000.0,
            IR_South_all_m3=float(irrigation @ np.where(inputs.block == "South", self.area, 0.0)) / 1000.0,
            D_North_m3=float(d_mm @ np.where(
                (inputs.block == "North") & inputs.cyclic, self.area, 0.0)) / 1000.0,
            D_South_m3=float(d_mm @ np.where(
                (inputs.block == "South") & inputs.cyclic, self.area, 0.0)) / 1000.0,
            LFV_pump_m3=lfv_m3,
            D_ncb_m3=float(flux["drainage_mm"] @ self._ncb_area) / 1000.0,
            IR_ncb_m3=float(irrigation @ self._ncb_area) / 1000.0,
            O_force_m3=o_force,
            W_force_m3=w_force,
        )

        per_plot = {
            "irrigation": irrigation, "requested": requested, "drainage": flux["drainage_mm"],
            "farmer_discharge": flux["discharge_mm"], "passive_drainage": passive,
            "lateral_sim": flux["lateral_mm"], "leftover": np.asarray(
                flux.get("leftover_mm", flux["lateral_mm"]), dtype=float),
            "deep": flux["deep_mm"], "et": flux["et_mm"],
            "et_potential": flux["et_potential_mm"], "infiltration": flux["infiltration_mm"],
            "neighbor": flux["neighbor_mm"],
        }
        storage = self.state.storage()
        today = {"date": inputs.dates[d], **self.station.today}
        for (block, group), weight in self._weights.items():
            prefix = f"{block}|{group}|"
            area = float(weight.sum())
            today[prefix + "area_m2"] = area
            today[prefix + "rain_m3"] = float(inputs.rain_mm[d]) * area / 1000.0
            today[prefix + "storage_m3"] = float(storage @ weight) / 1000.0
            for name, values in per_plot.items():
                today[f"{prefix}{name}_m3"] = float(values @ weight) / 1000.0
        self.today = today

        if self.record_paddies:
            for key, values in (("depth_mm", depth), ("requested_mm", requested),
                                ("irrigation_mm", irrigation), ("farmer_discharge_mm", flux["discharge_mm"]),
                                ("passive_drainage_mm", passive), ("drainage_mm", flux["drainage_mm"]),
                                ("lateral_mm", flux["lateral_mm"]), ("deep_mm", flux["deep_mm"]),
                                ("et_mm", flux["et_mm"]), ("et_potential_mm", flux["et_potential_mm"]),
                                ("infiltration_mm", flux["infiltration_mm"]),
                                ("neighbor_mm", flux["neighbor_mm"]),
                                ("et_pond_mm", flux["et_pond_mm"]),
                                ("et_surface_mm", flux["et_surface_mm"]),
                                ("et_root_mm", flux["et_root_mm"]),
                                ("et_groundwater_mm", flux["et_groundwater_mm"]),
                                ("transpiration_mm", flux.get("transpiration_mm", np.zeros(self.n))),
                                ("evaporation_mm", flux.get("evaporation_mm", np.zeros(self.n))),
                                ("leftover_mm", flux["leftover_mm"]),
                                ("stay_mm", flux["stay_mm"]),
                                ("ponded", flux["ponded"]),
                                ("assigned_lateral", flux["assigned_lateral"]),
                                ("need_mm", flux.get("need_mm", np.zeros(self.n))),
                                ("need_stock_mm", flux.get("need_stock_mm", self.state.need)),
                                ("canal_mm", flux.get("canal_mm", np.zeros(self.n))),
                                ("canal_out_mm", flux.get("canal_out_mm", np.zeros(self.n))),
                                ("drain_stock_mm", flux.get("drain_stock_mm", self.state.drain)),
                                ("flow_kx_mm", flux.get("flow_kx_mm", np.zeros(self.n))),
                                ("flow_ky_mm", flux.get("flow_ky_mm", np.zeros(self.n))),
                                ("flow_kz_mm", flux.get("flow_kz_mm", np.zeros(self.n))),
                                ("flow_ground_mm", flux.get("flow_ground_mm", np.zeros(self.n))),
                                ("lfv_mm", flux.get("lfv_mm", np.zeros(self.n)))):
                self.log[key][d] = values
            self.log["pond_mm"][d] = self.state.water
            self.log["topsoil_mm"][d] = np.maximum(p.total_evaporable_mm - self.state.surface, 0.0)
            if self.cell_log and self.cell_state is not None:
                self.cell_log["pond_mm"][d] = self.cell_state.water
                self.cell_log["topsoil_mm"][d] = np.maximum(
                    p.total_evaporable_mm - self.cell_state.surface, 0.0)

        self.datacollector.collect(self)
        self.day += 1
        if self.day >= self.n_days:
            self.running = False

    def run(self):
        while self.running:
            self.step()
        self.inverse = None
        if self.physics.force_satellite_et and self.log and not self.physics.canal_routing:
            stock = np.maximum(self.log["leftover_mm"].sum(axis=0), 0.0)
            self.inverse = solve_omalley(
                stock, self.area, self.inputs.cyclic, self.inputs.block,
                self.paddy_edges, self.physics.pool_capture_share, self.year)
            apply_assignment(self.log, self.inverse.x)
            if self.cell_log and "leftover_mm" in self.cell_log:
                apply_assignment(self.cell_log, self.inverse.x[self.cells.owner])
        results = Results(self)
        if self.save_csv:
            results.write_csv()
        return results


class Results:
    """A finished season, in the units the paper reports."""

    def __init__(self, model):
        self.model = model
        self.year = model.year
        self.daily = model.datacollector.get_model_vars_dataframe().reset_index(drop=True)
        self.csv_dir = None
        if getattr(model, "inverse", None) is not None and model.log:
            lat = model.log["lateral_mm"]
            stay = model.log["deep_mm"]
            capture = model.physics.pool_capture_share
            self.daily["LFV_pump_m3"] = capture * (lat @ model._cyclic_area) / 1000.0
            for block in BLOCKS:
                weight = model._weights[(block, "cyclic")]
                self.daily[f"{block}|cyclic|lateral_sim_m3"] = (lat @ weight) / 1000.0
                self.daily[f"{block}|cyclic|deep_m3"] = (stay @ weight) / 1000.0

    @cached_property
    def blocks(self):
        """Daily, one row per cyclic block per day -- the scale Okura et al. measured.
        irrigation_all_m3 adds the block area's non-cyclic supply (their IR_all);
        et_target_mm is the block's area-weighted satellite ET target."""
        inputs = self.model.inputs
        days = len(self.daily)
        frames = []
        for block in BLOCKS:
            cyclic, non_cyclic = f"{block}|cyclic|", f"{block}|non_cyclic|"
            frame = pd.DataFrame({"date": pd.to_datetime(self.daily["date"]), "block": block,
                                  "area_m2": self.daily[cyclic + "area_m2"]})
            for q in FLUXES:
                frame[f"{q}_m3"] = self.daily[f"{cyclic}{q}_m3"]
                frame[f"{q}_mm"] = frame[f"{q}_m3"] / frame["area_m2"] * 1000.0
            frame["storage_m3"] = self.daily[cyclic + "storage_m3"]
            frame["irrigation_ncb_m3"] = self.daily[non_cyclic + "irrigation_m3"]
            frame["drainage_ncb_m3"] = self.daily[non_cyclic + "drainage_m3"]
            frame["irrigation_all_m3"] = frame["irrigation_m3"] + frame["irrigation_ncb_m3"]
            frame["er_mm"] = okura.effective_rain_mm(frame["rain_mm"].to_numpy())
            weights = np.where((inputs.block == block) & inputs.cyclic, inputs.area_m2, 0.0)
            frame["et_target_mm"] = (inputs.et_target_mm[:days].astype(float) @ weights) / weights.sum()
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    @cached_property
    def station(self):
        frame = self.daily[["date", *STATION]].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        return frame

    def _require_log(self):
        if not self.model.log:
            raise RuntimeError("run the model with record_paddies=True")
        return self.model.log

    def paddy_daily(self, name):
        log = self._require_log()
        inputs = self.model.inputs
        return pd.DataFrame(log[name], index=inputs.dates, columns=inputs.paddy_id)

    def pixel_series(self, paddy):
        """Daily cell ET within one paddy, including source coordinates and fallback flags."""
        m = self.model
        if m.cell_et_mm is None:
            raise RuntimeError("pixel_series requires potential='pixel' and record_paddies=True with imagery")
        k = int(np.flatnonzero(m.inputs.paddy_id == paddy)[0]) if isinstance(paddy, str) else int(paddy)
        selected = np.flatnonzero(m.cells.owner == k)
        days = len(self.daily)
        return pd.DataFrame({
            "date": np.repeat(m.inputs.dates[:days], len(selected)),
            "cell": np.tile(selected, days),
            "lon": np.tile(m.cells.lon[selected], days),
            "lat": np.tile(m.cells.lat[selected], days),
            "centroid_fallback": np.tile(m.cells.fallback[selected], days),
            "area_m2": np.tile(m.area[k] * m.cells.weights[selected], days),
            "et_mm": m.cell_et_mm[:days, selected].ravel(),
            "satellite_et_mm": m.cells.observed_et_mm[:days, selected].ravel(),
        })

    def paddy_series(self, paddy):
        """One paddy's season, a row a day: its water, what it was sent and let go (mm and
        m3), and the ET it transpired against its own satellite target. `paddy` is a
        paddy_id or an index."""
        log = self._require_log()
        inputs = self.model.inputs
        k = int(np.flatnonzero(inputs.paddy_id == paddy)[0]) if isinstance(paddy, str) else int(paddy)
        area = float(self.model.area[k])
        days = len(self.daily)
        extra = {}
        for name in ("et_pond_mm", "et_surface_mm", "et_root_mm", "et_groundwater_mm",
                     "transpiration_mm", "evaporation_mm", "leftover_mm",
                     "stay_mm", "pond_mm", "topsoil_mm", "deep_mm", "need_mm", "need_stock_mm",
                     "canal_mm", "canal_out_mm", "drain_stock_mm"):
            if name in log:
                extra[name] = log[name][:days, k]
        frame = pd.DataFrame({
            "date": inputs.dates[:days],
            "depth_mm": log["depth_mm"][:days, k],
            "irrigation_mm": log["irrigation_mm"][:days, k],
            "drainage_mm": log["drainage_mm"][:days, k],
            "farmer_discharge_mm": log["farmer_discharge_mm"][:days, k],
            "lateral_mm": log["lateral_mm"][:days, k],
            "et_mm": log["et_mm"][:days, k],
            "et_target_mm": inputs.et_target_mm[:days, k],
            "field_et_mm": inputs.field_et_mm[inputs.block[k]][:days],
            "is_pass": inputs.is_pass[:days, k],
            "satellite_et_mm": inputs.sat_pass_et_mm[:days, k],
            "rain_mm": inputs.rain_mm[:days],
            **extra,
        })
        frame["irrigation_m3"] = frame["irrigation_mm"] * area / 1000.0
        frame["drainage_m3"] = frame["drainage_mm"] * area / 1000.0
        frame.attrs.update(paddy_id=str(inputs.paddy_id[k]), block=str(inputs.block[k]),
                           cyclic=bool(inputs.cyclic[k]), area_m2=area, index=k)
        return frame

    def paddy_season(self):
        """Every paddy's season: what it took, let go, seeped and transpired, its WUE' by
        Eq. 11 at plot scale, and how closely its ET followed its own satellite target --
        on its pass days, and over every day."""
        model, inputs = self.model, self.model.inputs
        log = self._require_log()
        days = len(self.daily)
        frame = pd.DataFrame({
            "paddy_id": inputs.paddy_id, "block": inputs.block, "cyclic": inputs.cyclic,
            "area_ha": model.area / 1e4, "distance_m": inputs.frame["distance_m"].to_numpy(),
            "lon": inputs.frame["lon"].to_numpy(), "lat": inputs.frame["lat"].to_numpy(),
            "rain_mm": inputs.rain_mm[:days].sum(),
        })
        for key in ("irrigation_mm", "farmer_discharge_mm", "passive_drainage_mm", "drainage_mm",
                    "lateral_mm", "deep_mm", "et_mm", "et_potential_mm", "leftover_mm",
                    "et_groundwater_mm", "stay_mm", "need_mm", "canal_mm", "canal_out_mm"):
            if key in log:
                frame[key] = log[key][:days].sum(axis=0)
        frame["mean_depth_mm"] = log["depth_mm"][:days].mean(axis=0)
        if "pond_mm" in log:
            frame["mean_pond_mm"] = log["pond_mm"][:days].mean(axis=0)
        if "topsoil_mm" in log:
            frame["mean_topsoil_mm"] = log["topsoil_mm"][:days].mean(axis=0)
        if "drain_stock_mm" in log:
            frame["mean_drain_stock_mm"] = log["drain_stock_mm"][:days].mean(axis=0)
        frame["days_irrigated"] = (log["irrigation_mm"][:days] > 0).sum(axis=0)
        frame["irrigation_m3"] = frame["irrigation_mm"] * model.area / 1000.0
        frame["drainage_m3"] = frame["drainage_mm"] * model.area / 1000.0
        effective = okura.effective_rain_mm(inputs.rain_mm[:days]).sum()
        net = frame["irrigation_mm"] - frame["drainage_mm"] + effective
        frame["wue_prime_pct"] = np.where(net > 0, 100.0 * frame["et_mm"] / net.where(net > 0, 1), np.nan)
        frame["et_ratio"] = frame["et_mm"] / frame["et_potential_mm"]

        target = inputs.et_target_mm[:days].astype(float)
        passes = inputs.is_pass[:days]
        miss = np.abs(log["et_mm"][:days] - target)
        frame["et_target_mm"] = target.sum(axis=0)
        frame["et_fidelity"] = np.clip(1.0 - miss.sum(axis=0) / np.maximum(target.sum(axis=0), 1e-9), 0.0, 1.0)
        pass_target = (target * passes).sum(axis=0)
        frame["et_pass_fidelity"] = np.where(
            pass_target > 0, np.clip(1.0 - (miss * passes).sum(axis=0) / np.maximum(pass_target, 1e-9), 0.0, 1.0),
            np.nan)
        frame["passes"] = passes.sum(axis=0)
        return frame

    def underground_daily(self, grain="paddy"):
        """Water that went into the soil and sideways, per point per day.

        grain 'paddy'   one row per parcel
              'cell'    one row per satellite cell (needs record_paddies and imagery)

        lateral_to_pool_m3 is the captured Hooghoudt drain flow (Eq. 7 share) for cyclic
        parcels; non-cyclic discharge and lateral leave to the lake / Tone and are 0 here.
        """
        model, inputs = self.model, self.model.inputs
        days = len(self.daily)
        capture = model.physics.pool_capture_share
        if grain == "cell":
            if not model.cell_log:
                raise RuntimeError(
                    "cell CSV needs CyclicModel(..., record_cells=True); "
                    "the default skips the ~120k-cell daily log to save RAM")
            grid = model.cells
            n = len(grid.owner)
            date = np.repeat(inputs.dates[:days], n)
            owner = np.tile(grid.owner, days)
            cyclic = inputs.cyclic[owner]
            area = np.tile(model.area[grid.owner] * grid.weights, days)
            lateral = model.cell_log["lateral_mm"][:days].ravel()
            to_pool = np.where(cyclic, capture * lateral * area / 1000.0, 0.0)
            return pd.DataFrame({
                "date": date, "cell": np.tile(np.arange(n), days),
                "paddy_id": inputs.paddy_id[owner], "block": inputs.block[owner],
                "cyclic": cyclic, "lon": np.tile(grid.lon, days), "lat": np.tile(grid.lat, days),
                "area_m2": area,
                "irrigation_mm": model.log["irrigation_mm"][:days][:, grid.owner].ravel(),
                "drainage_mm": model.cell_log["drainage_mm"][:days].ravel(),
                "infiltration_mm": model.cell_log["infiltration_mm"][:days].ravel(),
                "neighbor_mm": model.cell_log["neighbor_mm"][:days].ravel(),
                "lateral_mm": lateral,
                "deep_mm": model.cell_log["deep_mm"][:days].ravel(),
                "lateral_to_pool_m3": to_pool,
                "et_mm": model.cell_log["et_mm"][:days].ravel(),
                "et_pond_mm": model.cell_log["et_pond_mm"][:days].ravel(),
                "et_surface_mm": model.cell_log["et_surface_mm"][:days].ravel(),
                "et_root_mm": model.cell_log["et_root_mm"][:days].ravel(),
                "ponded": model.cell_log["ponded"][:days].ravel(),
            })
        log = self._require_log()
        n = model.n
        date = np.repeat(inputs.dates[:days], n)
        cyclic = np.tile(inputs.cyclic, days)
        area = np.tile(model.area, days)
        lateral = log["lateral_mm"][:days].ravel()
        to_pool = np.where(cyclic, capture * lateral * area / 1000.0, 0.0)
        return pd.DataFrame({
            "date": date, "paddy_id": np.tile(inputs.paddy_id, days),
            "block": np.tile(inputs.block, days), "cyclic": cyclic,
            "lon": np.tile(inputs.frame["lon"].to_numpy(), days),
            "lat": np.tile(inputs.frame["lat"].to_numpy(), days),
            "area_m2": area,
            "irrigation_mm": log["irrigation_mm"][:days].ravel(),
            "discharge_mm": log["farmer_discharge_mm"][:days].ravel(),
            "drainage_mm": log["drainage_mm"][:days].ravel(),
            "infiltration_mm": log["infiltration_mm"][:days].ravel(),
            "neighbor_mm": log["neighbor_mm"][:days].ravel(),
            "lateral_mm": lateral,
            "deep_mm": log["deep_mm"][:days].ravel(),
            "lateral_to_pool_m3": to_pool,
            "leftover_mm": log["leftover_mm"][:days].ravel() if "leftover_mm" in log else 0,
            "et_groundwater_mm": log["et_groundwater_mm"][:days].ravel() if "et_groundwater_mm" in log else 0,
            "stay_mm": log["stay_mm"][:days].ravel() if "stay_mm" in log else log["deep_mm"][:days].ravel(),
            "assigned_lateral": log["assigned_lateral"][:days].ravel() if "assigned_lateral" in log else 0,
            "et_mm": log["et_mm"][:days].ravel(),
            "et_pond_mm": log["et_pond_mm"][:days].ravel(),
            "et_surface_mm": log["et_surface_mm"][:days].ravel(),
            "et_root_mm": log["et_root_mm"][:days].ravel(),
            "ponded": log["ponded"][:days].ravel(),
            "flood_frac": inputs.flood_daily[:days].ravel(),
            "canal_mm": log["canal_mm"][:days].ravel() if "canal_mm" in log else 0,
            "canal_out_mm": log["canal_out_mm"][:days].ravel() if "canal_out_mm" in log else 0,
            "drain_stock_mm": log["drain_stock_mm"][:days].ravel() if "drain_stock_mm" in log else 0,
            "drainage_to_pool_m3": np.where(cyclic, log["drainage_mm"][:days].ravel() * area / 1000.0, 0.0),
            "drainage_ncb_m3": np.where(~cyclic, log["drainage_mm"][:days].ravel() * area / 1000.0, 0.0),
        })

    def underground_csv(self, path, grain="paddy"):
        """Write underground_daily to CSV. Season totals go next to it as *_totals.csv."""
        daily = self.underground_daily(grain)
        daily.to_csv(path, index=False)
        group = ["paddy_id"] if grain == "paddy" else ["cell", "paddy_id"]
        sums = [c for c in daily.columns if c.endswith("_mm") or c.endswith("_m3")]
        totals = daily.groupby(group, as_index=False)[sums].sum()
        first = daily.groupby(group, as_index=False).first()
        keep = [c for c in ("block", "cyclic", "lon", "lat", "area_m2") if c in first]
        totals = first[group + keep].merge(totals, on=group)
        totals_path = str(path).replace(".csv", "_totals.csv")
        totals.to_csv(totals_path, index=False)
        return path, totals_path

    def paddy_log_table(self):
        """Every paddy, every day, as one CSV-ready table (stocks and fluxes)."""
        log = self._require_log()
        inputs = self.model.inputs
        days = len(self.daily)
        n = self.model.n
        frame = {
            "date": np.repeat(inputs.dates[:days], n),
            "paddy_id": np.tile(inputs.paddy_id, days),
            "block": np.tile(inputs.block, days),
            "cyclic": np.tile(inputs.cyclic, days),
            "lon": np.tile(inputs.frame["lon"].to_numpy(), days),
            "lat": np.tile(inputs.frame["lat"].to_numpy(), days),
            "area_m2": np.tile(self.model.area, days),
            "rain_mm": np.repeat(np.asarray(getattr(inputs, "rain_mm", np.zeros(days))), n),
            "et_target_mm": np.asarray(getattr(inputs, "et_target_mm", np.zeros((days, n))))[:days].ravel(),
            "is_pass": np.asarray(getattr(inputs, "is_pass", np.zeros((days, n), dtype=bool)))[:days].ravel(),
        }
        for key, values in log.items():
            frame[key] = values[:days].ravel()
        return pd.DataFrame(frame)

    def write_csv(self, folder=None):
        """Write this run to CSV. Plots are not drawn — open Field over time / Paddies later.

        Folder defaults to outputs/runs/<year>_<policies>/. Overwrites the same farmer+year.
        """
        folder = folder or self.default_csv_dir()
        os.makedirs(folder, exist_ok=True)
        self.blocks.to_csv(os.path.join(folder, "blocks.csv"), index=False)
        self.station.to_csv(os.path.join(folder, "station.csv"), index=False)
        if self.model.log:
            self.paddy_log_table().to_csv(os.path.join(folder, "paddy_daily.csv"), index=False)
            try:
                self.paddy_season().to_csv(os.path.join(folder, "paddies_season.csv"), index=False)
            except (KeyError, RuntimeError):
                pass
            try:
                self.underground_csv(os.path.join(folder, "underground.csv"), grain="paddy")
            except (KeyError, RuntimeError):
                pass
        self.csv_dir = folder
        return folder

    def default_csv_dir(self):
        names = []
        for policy, _ in getattr(self.model, "groups", []):
            names.append(getattr(policy, "name", type(policy).__name__))
        slug = "+".join(dict.fromkeys(names)) or "run"
        slug = re.sub(r"[^\w.+-]+", "_", slug)[:80]
        return os.path.join(data.PROJECT, "outputs", "runs", f"{self.year}_{slug}")

# -*- coding: utf-8 -*-
"""ABM world on the cyclic district map, run on cyclic's own paddy hydrology.

Agents (LID staff + farmers) act on:
  * real Imbanuma paddies (``cyclic.data.paddies``)
  * ``cyclic.physics.advance`` for every paddy each day: FAO-56 dual-Kc ET demand, the
    farmer's discharge + bund overflow + outlet leak (D), plow-pan percolation in
    proportion to ponded depth, root zone, perched layer, Darcy exchange with
    neighbouring paddies through the perched layer, Hooghoudt lateral drain flow
    and deep percolation
  * the discovered per-paddy recapture share (``pool_capture_share``, district mean
    = Okura Table 5 LFV_pump / ΣLf·A ≈ 0.76) applied to that lateral flow on cyclic
    paddies — the rest stays in the blocks as Okura's ΔWB (Eq. 7)
  * Open-Meteo rain/ET0 from ``cyclic.data.weather``, rain scaled to the on-site
    gauge's monthly totals (Okura Table 2)

Seepage is therefore a continuous, depth-driven loss from every flooded paddy
(Okura Lf ≈ 7–11 mm/d), not a share of whatever surplus a day happens to leave.

Water stress (``advance(..., stress="supply")``): each day the weather drives a
full, well-watered ET demand (dual Kc × ET0). Rice meets it from standing water,
then the topsoil layer (not the deeper root zone); the millimetres it cannot supply are
that day's stress (demand 8 mm, 2 mm available → 6 mm). Ks = ETa / demand, and
FAO-33 relative yield = 1 − Ky · Σstress / Σdemand over the season so far.

Infrastructure (``cyclic.infrastructure``; sources in cyclic/bundled/SOURCES.md): the
district's parcels, North/South blocks and cyclic status follow Okura Fig. 1a, and the pool
sits at the Hakusan-Jimbee pumping station. A cyclic paddy's surface drainage D reaches the
pool along a real route: to the nearest mapped channel connected to the main drainage
canal, then along the channels to the station. It travels at ``drain_velocity_m_s``, so D
released late in the day on a long route arrives the next day; water on its way is
``drain_transit_m3``.

Backwater (Okura §4.1): when the main drainage canal runs high, low paddies' farm drains
cannot empty into it and the water stays on the field (it then leaves as seepage and lateral
flow). The canal's daily depth at the North and South sites comes from Okura Fig. 3 for 2021
and 2022 (``canal_depth="okura"``). A cyclic paddy's drain is free while its ground
(GSI 5 m DEM) stands at least ``backwater_band_m`` above the drain reference plus the canal's
rise over its normal (2022) depth; the reference is set so that at normal depth even the
lowest 2% of cyclic land drains freely. Below that, the blocked share of its drainage D
rises linearly to all of it.

Irrigation supply: the station cannot deliver more than its pumps' capacity
(``pump_capacity_m3_s``, MAFF: 3.304 m³/s); beyond it the nearest paddies are served first.

Pool: the 18 × 68 m pool is a flow-through buffer (``Physics.pool_capacity_m3``).
Same-day cyclic drainage and recaptured lateral flow are pumped straight back out;
Lake Kita-Imbanuma (a boundary, never exhausted) makes up what returns do not
cover through the station's gate (Okura Eq. 4/6), and anything above capacity
spills to the lake at the drainage pumping station (O).

Hard rules: farmers apply ≤ station send; discharge ≤ ponded water.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cyclic.physics import Physics, State, advance

from . import crop
from .landscape import (
    CyclicLandscape,
    assign_owners_along_canal_variable,
    load_cyclic_landscape,
)


@dataclass
class SeasonConfig:
    year: int = 2022
    days: int | None = None  # None → full Open-Meteo season length
    seed: int = 0
    paddies_per_farmer: int = 50
    # Ownership is scattered reproducibly across the district. 0 = every farmer holds
    # exactly paddies_per_farmer ("fixed"); up to 1.0 lets holdings vary between
    # 0.5x and 2x that average instead of all being identical.
    farm_size_variability: float = 0.0
    # Which paddies each farmer owns. None (default) = a fresh random ownership every time the
    # world is built or reset; the seed drawn is kept as WaterWorld.ownership_seed_used so a
    # run can be repeated by passing it back here. An int = that fixed ownership.
    ownership_seed: int | None = None
    # None → full pool (pool_capacity_m3).
    initial_pool_m3: float | None = None
    # None → cyclic.physics.Physics.pool_capacity_m3 (18 × 68 m, 4 m deep).
    pool_capacity_m3: float | None = None
    # Running tally of net lake exchange only. The lake is a boundary condition
    # (Conceptual Model, Scope table): withdrawals are never limited by it.
    initial_lake_m3: float = 5_000_000.0
    # Bund/levee height: real paddy bunds run ~15-30 cm and spill uncontrolled surface
    # runoff once exceeded (IRRI water-management guidance; Chapagain & Yamaji 2010).
    # Passed to cyclic.physics as bund_mm.
    bund_height_mm: float = 250.0
    # Scale Open-Meteo rain to the on-site gauge's monthly totals (Okura Table 2).
    # Open-Meteo overstates season rain by ~30-50% at Imbanuma; years without a gauge
    # record use the mean monthly ratio of the gauged years.
    rain_gauge_correction: bool = True
    initial_perched_mm: float = 10.0
    # Day-to-day weather jitter around the historical record (0 = exact replay of
    # that year's rain/ET0). Weather the farmer cannot see coming or
    # steer, only react to; see perturb_weather.
    weather_variability: float = 0.0
    # Weather scenario (abm.weather.WeatherScenario or a preset name such as
    # "drought"): rain/ET0 scaling, dry spells, heatwaves, storms, jitter. When set,
    # its own ``variability`` replaces weather_variability.
    weather: object | None = None
    # Collaboration scenario: this share (0..1) of all farmers, whatever their type, is made
    # collaborative (cooperation 0.95, levels its water with the neighbours) and the rest
    # uncollaborative (0.05, keeps its own level). None = every farmer keeps its own trait.
    collaborative_share: float | None = None
    # Volumetric water price (per m3 applied). 0 = area-based fees only (the Japanese
    # land-improvement-district norm: no marginal cost). Farmers run a pricing check
    # against it (abm.farmer.BaseFarmer.price_multiplier).
    water_price_per_m3: float = 0.0
    # Flow speed (m/s) of surface drainage through farm drains and the channel network to
    # the pool (assumption: slow flat-lowland drains). D is released evenly through the day,
    # so the share arriving the same day is 1 − travel time (days). None or 0 = every
    # cyclic D reaches the pool the same day (Okura's assumption).
    drain_velocity_m_s: float | None = 0.2
    # Main drainage canal water level for backwater. "okura": daily North/South depths
    # digitised from Okura Fig. 3 (2021, 2022), the normal 2022 level in other years; a
    # float: that depth (m) every day; None: no backwater.
    canal_depth: str | float | None = "okura"
    # Margin (m) over which a paddy's farm drain goes from free to fully drowned.
    backwater_band_m: float = 0.10
    # Irrigation pumping station capacity (m3/s). MAFF: 白山甚兵衛機場 max supply 3.304 m3/s.
    # None = no limit beyond the water available.
    pump_capacity_m3_s: float | None = 3.304
    use_cyclic_map: bool = True
    # Toy-grid fallback only if use_cyclic_map=False.
    nrows: int | None = None
    ncols: int | None = None
    area_ha_each: float = 0.5
    capture: float | None = None
    neighbour: float | None = None
    stay: float | None = None


@dataclass
class StepResult:
    day: int
    stage: str
    rain_mm: float
    et0_mm: float
    kc: float
    etc_mm: float
    target_depth_mm: float
    irrigation_sent_mm: np.ndarray
    irrigation_mm: np.ndarray
    drainage_mm: np.ndarray       # D = farmer discharge + bund overflow + outlet leak
    leftover_mm: np.ndarray       # Okura Eq. 3 residual IR + R − D − ET (accounting only)
    to_pool_mm: np.ndarray        # recaptured lateral flow → pool (LFV_pump)
    to_neighbour_mm: np.ndarray   # perched water sent to neighbouring paddies
    stay_mm: np.ndarray           # lateral flow not recaptured → block storage (ΔWB)
    hop_in_mm: np.ndarray         # net perched exchange with neighbours (+ = received)
    ponding_mm: np.ndarray
    ks: np.ndarray
    eta_mm: np.ndarray
    et_potential_mm: np.ndarray   # weather-driven ET demand
    stress_mm: np.ndarray         # demand the paddy's water could not supply
    infiltration_mm: np.ndarray
    lateral_mm: np.ndarray        # Hooghoudt drain flow out of the perched layer
    deep_mm: np.ndarray
    owner_id: np.ndarray
    withdraw_m3: float
    outflow_m3: float             # planned station outflow + automatic pool spill
    pool_spill_m3: float
    pool_m3: float
    lake_m3: float
    return_d_m3: float                # cyclic D arriving at the pool today (routed)
    return_lfv_m3: float
    unapplied_m3: float
    dwb_m3: float                 # cumulative block storage change (Okura Eq. 7)
    mean_relative_yield: float
    requested_mm: np.ndarray | None = None  # farmers' demand on the canal (before supply)
    weir_mm: np.ndarray | None = None       # outlet weir crest each farmer set (inf = none)
    drain_transit_m3: float = 0.0           # cyclic D still on its way to the pool
    backwater_mm: np.ndarray | None = None  # drainage held on the field by a high canal
    canal_depth_north_m: float = float("nan")
    canal_depth_south_m: float = float("nan")


def perturb_weather(
    rain_mm: np.ndarray, et0_mm: np.ndarray, variability: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Day-to-day multiplicative log-normal jitter around one year's historical
    rain/ET0 record — weather the farmer cannot see coming or control, only react
    to. ``variability`` 0 replays that year exactly (default; keeps runs comparable
    to the paper's pins); >0 (e.g. 0.2-0.4) makes every reset a different, but still
    climatologically plausible, season. Rain gets more relative jitter than ET0:
    day-to-day rainfall is bursty and event-driven, while ET0 tracks smoother
    radiation/temperature swings.
    """
    if variability <= 0:
        return rain_mm, et0_mm
    rng = np.random.default_rng((int(seed) * 2654435761 + 0x9E3779B9) & 0xFFFFFFFF)
    et0_noise = rng.normal(0.0, variability * 0.5, size=et0_mm.shape)
    rain_noise = rng.normal(0.0, variability * 1.5, size=rain_mm.shape)
    et0_out = np.clip(et0_mm * np.exp(et0_noise), 0.0, None)
    rain_out = np.clip(rain_mm * np.exp(rain_noise), 0.0, None)
    return rain_out, et0_out


def _monthly_gauge_factors(year: int, rain_mm: np.ndarray) -> dict[int, float]:
    """Gauge / Open-Meteo rain per calendar month over the season (Okura Table 2)."""
    from cyclic import okura

    dates = okura.season_dates(year)[: len(rain_mm)]
    gauge = okura.TABLE2_MM[year]["rain"]
    factors = {}
    for month, total in gauge.items():
        model = float(rain_mm[np.asarray(dates.month) == month].sum())
        if model > 0:
            factors[int(month)] = float(total) / model
    return factors


def gauge_corrected_rain(rain_mm: np.ndarray, year: int) -> np.ndarray:
    """Scale each month of an Open-Meteo season so it matches the site gauge.

    Gauged years (2021, 2022) are matched month by month; any other year gets the
    mean monthly ratio of the gauged years, so scenario seasons start from the same
    bias-corrected climatology rather than Open-Meteo's wetter one.
    """
    from cyclic import okura

    rain_mm = np.asarray(rain_mm, dtype=float)
    if year in okura.TABLE2_MM:
        factors = _monthly_gauge_factors(year, rain_mm)
    else:
        from cyclic.paddy_export import load_paddy_dataset

        per_year = [
            _monthly_gauge_factors(y, load_paddy_dataset(y).rain_mm.values)
            for y in okura.TABLE2_MM
        ]
        factors = {
            m: float(np.mean([f[m] for f in per_year if m in f]))
            for m in set().union(*per_year)
        }
    dates = okura.season_dates(year)[: len(rain_mm)]
    scale = np.array([factors.get(int(m), 1.0) for m in dates.month])
    return rain_mm * scale


def assign_owners(n: int, paddies_per_farmer: int) -> np.ndarray:
    if paddies_per_farmer <= 0 or paddies_per_farmer >= n:
        return np.zeros(n, dtype=int)
    owners = np.zeros(n, dtype=int)
    farmer = 0
    for start in range(0, n, paddies_per_farmer):
        owners[start : start + paddies_per_farmer] = farmer
        farmer += 1
    return owners


@dataclass
class WaterWorld:
    config: SeasonConfig = field(default_factory=SeasonConfig)
    landscape: CyclicLandscape | None = field(default=None, repr=False)
    physics: Any = field(default=None, repr=False)

    def __post_init__(self):
        cfg = self.config
        self.bund_height_mm = float(cfg.bund_height_mm)
        # cyclic's soil/drain constants; only the bund height is an ABM scenario knob.
        self.soil_physics = Physics().replace(bund_mm=self.bund_height_mm)
        self.pool_capacity_m3 = float(
            cfg.pool_capacity_m3 if cfg.pool_capacity_m3 is not None
            else self.soil_physics.pool_capacity_m3
        )
        if cfg.use_cyclic_map:
            self._init_from_cyclic()
        else:
            self._init_toy_grid()

    # --- ponding is cyclic.physics.State.water -------------------------------------------

    @property
    def ponding_mm(self) -> np.ndarray:
        return self.soil.water

    @ponding_mm.setter
    def ponding_mm(self, value) -> None:
        self.soil.water = np.asarray(value, dtype=float).copy()

    def _init_from_cyclic(self):
        cfg = self.config
        land = self.landscape or load_cyclic_landscape(year=cfg.year, days=cfg.days)
        self.landscape = land
        self.physics = land.physics
        self.n = land.n
        self.lon = land.lon
        self.lat = land.lat
        self.area_m2 = land.area_m2
        self.block = land.block
        self.cyclic = land.cyclic
        self.paddy_id = land.paddy_id
        self.canal_distance_m = land.canal_distance_m
        self.pool_distance_m = land.pool_distance_m
        # Drainage route to the pool through the mapped channels (entry + along channels).
        zeros = np.zeros(land.n)
        self.drain_entry_m = np.asarray(getattr(land, "drain_entry_m", None) if getattr(land, "drain_entry_m", None) is not None else zeros, dtype=float)
        self.drain_path_m = np.asarray(getattr(land, "drain_path_m", None) if getattr(land, "drain_path_m", None) is not None else zeros, dtype=float)
        from cyclic.infrastructure import STATION_LONLAT
        self.station_lonlat = STATION_LONLAT
        heights = getattr(land, "ground_elev_m", None)
        self.ground_elev_m = None if heights is None else np.asarray(heights, dtype=float)
        self.edges = land.edges
        self.graph = land.graph
        self.neighbours = land.neighbours
        self.capture_i = land.capture_i.copy()
        self.neighbour_i = land.neighbour_i.copy()
        self.stay_i = land.stay_i.copy()
        self.neighbour_of_remain = float(land.neighbour_of_remain)
        self.hop = land.hop
        self.geometry = land.geometry
        self.claim = land.claim
        self.ground_water_law = land.ground_water_law
        # Share of each paddy's lateral drain flow the pool recaptures (0 off-cycle).
        share = land.shares.get("pool_capture_share", self.capture_i)
        self.lateral_capture_i = np.where(self.cyclic, np.clip(np.asarray(share, dtype=float), 0.0, 1.0), 0.0)

        # District means (for status / metrics).
        mask = land.cyclic
        self.capture = float(self.lateral_capture_i[mask].mean()) if mask.any() else float(land.physics.capture)
        self.neighbour = float(self.neighbour_i[mask].mean()) if mask.any() else float(land.physics.neighbour)
        self.stay = float(self.stay_i[mask].mean()) if mask.any() else float(land.physics.stay)

        self.owner_id = assign_owners_along_canal_variable(
            land.owner_seed_order, self.n, cfg.paddies_per_farmer,
            cfg.farm_size_variability, self._draw_ownership_seed(),
        )
        # Pseudo row/col for neighbour_view fallback (spatial rank).
        self.row = np.argsort(np.argsort(self.lat))  # north→south rank
        self.col = np.argsort(np.argsort(self.lon))
        self.nrows = self.ncols = int(np.ceil(np.sqrt(self.n)))

        rain = land.rain_mm
        if cfg.rain_gauge_correction:
            rain = gauge_corrected_rain(rain, cfg.year)
        if cfg.weather is not None:
            from .weather import weather_scenario

            self.weather = weather_scenario(cfg.weather)
            self.rain_series, self.et0_series = self.weather.apply(rain, land.et0_mm, cfg.year, cfg.seed)
        else:
            self.weather = None
            self.rain_series, self.et0_series = perturb_weather(
                rain, land.et0_mm, cfg.weather_variability, cfg.seed
            )
        cfg.days = len(self.rain_series)
        self._reset_runtime(cfg)

    def _init_toy_grid(self):
        """Legacy synthetic grid (only if use_cyclic_map=False)."""
        from cyclic.bundle import load_physics_bundle, physics_as_namespace

        cfg = self.config
        self.physics = physics_as_namespace(load_physics_bundle())
        self.capture = float(cfg.capture if cfg.capture is not None else self.physics.capture)
        self.neighbour = float(
            cfg.neighbour if cfg.neighbour is not None else self.physics.neighbour
        )
        self.stay = float(cfg.stay if cfg.stay is not None else self.physics.stay)
        side = int(np.ceil(np.sqrt(36)))
        self.nrows, self.ncols = side, side
        self.n = side * side
        self.neighbours = [
            [
                j
                for j in (
                    i - side if i >= side else -1,
                    i + side if i + side < self.n else -1,
                    i - 1 if i % side else -1,
                    i + 1 if (i + 1) % side else -1,
                )
                if j >= 0
            ]
            for i in range(self.n)
        ]
        self.owner_id = assign_owners(self.n, cfg.paddies_per_farmer)
        self.area_m2 = np.full(self.n, cfg.area_ha_each * 1e4)
        self.block = np.array(
            ["North" if i < self.n // 2 else "South" for i in range(self.n)]
        )
        self.cyclic = np.ones(self.n, dtype=bool)
        self.lon = np.array([i % side for i in range(self.n)], dtype=float)
        self.lat = np.array([i // side for i in range(self.n)], dtype=float)
        self.row = np.array([i // side for i in range(self.n)])
        self.col = np.array([i % side for i in range(self.n)])
        self.capture_i = np.full(self.n, self.capture)
        self.neighbour_i = np.full(self.n, self.neighbour)
        self.stay_i = np.full(self.n, self.stay)
        self.lateral_capture_i = np.full(self.n, self.capture)
        self.edges = None
        self.graph = None
        self.hop = None
        self.geometry = None
        self.neighbour_of_remain = 0.5
        self.ground_water_law = None
        self.claim = str(getattr(self.physics, "claim", ""))
        self.paddy_id = np.arange(self.n)
        self.canal_distance_m = np.zeros(self.n)
        # No real pump location on the toy grid — treat the (0, 0) corner as the pool.
        self.pool_distance_m = np.hypot(self.lon, self.lat)
        self.drain_entry_m = np.zeros(self.n)
        self.drain_path_m = self.pool_distance_m.copy()
        self.station_lonlat = (0.0, 0.0)
        self.ground_elev_m = None
        from cyclic import okura

        days = cfg.days or okura.SEASON_DAYS
        self.rain_series = self._monthly_toy(cfg.year, days, "rain")
        self.et0_series = self._monthly_toy(cfg.year, days, "et")
        cfg.days = days
        self._reset_runtime(cfg)

    @staticmethod
    def _monthly_toy(year, days, kind):
        from cyclic import okura

        table = okura.TABLE2_MM[year][kind]
        dates = okura.season_dates(year)[:days]
        out = np.zeros(days)
        for i, ts in enumerate(dates):
            m = int(ts.month)
            md = sum(1 for d in dates if d.month == m)
            out[i] = float(table.get(m, 0.0)) / max(md, 1)
        return out

    def _reset_runtime(self, cfg: SeasonConfig):
        self.day = 0
        self.history: list[StepResult] = []
        rng = np.random.default_rng(cfg.seed)
        self.soil = State.start(self.n, 0.0, cfg.initial_perched_mm)
        self.soil.water = rng.uniform(40.0, 70.0, size=self.n)
        self.pool_m3 = float(
            self.pool_capacity_m3 if cfg.initial_pool_m3 is None
            else min(cfg.initial_pool_m3, self.pool_capacity_m3)
        )
        self.lake_m3 = float(cfg.initial_lake_m3)
        self.irrigation_sent_mm = np.zeros(self.n)
        self.gate_m3 = 0.0
        self.planned_outflow_m3 = 0.0
        self.last_hop_in = np.zeros(self.n)
        self.last_ir = np.zeros(self.n)
        self.last_dr = np.zeros(self.n)
        self.last_leftover = np.zeros(self.n)
        self.last_to_pool = np.zeros(self.n)
        self.last_eta = np.zeros(self.n)
        self.last_seepage = np.zeros(self.n)
        self.last_withdraw_m3 = 0.0
        self.last_outflow_m3 = 0.0
        self.last_spill_m3 = 0.0
        self.last_return_m3 = 0.0
        self._set_drain_routing(cfg)
        self._set_backwater(cfg)
        self.dwb_m3 = 0.0
        self.deep_m3 = 0.0
        self.ks = np.ones(self.n)
        self.eta_cum = np.zeros(self.n)
        self.etp_cum = np.zeros(self.n)
        self.stress_cum = np.zeros(self.n)
        self.last_stress = np.zeros(self.n)
        self.etc_cum = 0.0
        self.relative_yield = np.ones(self.n)
        # Each paddy's crop calendar shift (days ahead of the district calendar), set by
        # its owner's eagerness (abm.model.DistrictModel keeps it in sync).
        self.calendar_shift_days = np.zeros(self.n, dtype=int)
        self.last_request = np.zeros(self.n)
        self.request_cum = np.zeros(self.n)

    def _set_drain_routing(self, cfg: SeasonConfig) -> None:
        """Share of each paddy's D that reaches the pool k days later (k = 0 today).

        Travel time t = route length / speed, in days. D is released evenly over the day,
        so it arrives spread evenly over [t, t + 1]; day k receives the overlap with
        [k, k + 1]."""
        route = np.asarray(self.drain_entry_m, dtype=float) + np.asarray(self.drain_path_m, dtype=float)
        speed = float(cfg.drain_velocity_m_s or 0.0)
        travel = route / (speed * 86_400.0) if speed > 0 else np.zeros(self.n)
        self.drain_travel_days = travel
        lags = int(np.floor(travel.max())) + 2 if self.n else 1
        share = np.zeros((self.n, lags))
        for k in range(lags):
            share[:, k] = np.clip(np.minimum(travel + 1.0, k + 1.0) - np.maximum(travel, float(k)), 0.0, 1.0)
        self.drain_share = share / np.maximum(share.sum(axis=1, keepdims=True), 1e-12)
        self.drain_pending_m3 = np.zeros(lags)

    def _set_backwater(self, cfg: SeasonConfig) -> None:
        """Daily canal depth per arm and each cyclic paddy's drain reference (see module doc)."""
        days = int(cfg.days or len(self.rain_series))
        self.last_backwater_frac = np.zeros(self.n)
        self.canal_depth_m = None
        if cfg.canal_depth is None or self.ground_elev_m is None or not np.isfinite(self.ground_elev_m).any():
            return
        from cyclic import infrastructure

        self.normal_canal_depth_m = infrastructure.normal_canal_depth_m()
        if cfg.canal_depth == "okura":
            series = infrastructure.canal_depth(int(cfg.year))
            if series is None:
                series = {arm: np.full(days, self.normal_canal_depth_m) for arm in ("North", "South")}
        else:
            series = {arm: np.full(days, float(cfg.canal_depth)) for arm in ("North", "South")}
        pad = lambda a: np.concatenate([a, np.full(max(days - len(a), 0), a[-1])])[:days]
        self.canal_depth_m = {arm: pad(np.asarray(values, dtype=float)) for arm, values in series.items()}
        band = max(float(cfg.backwater_band_m), 1e-6)
        ground, cyc = self.ground_elev_m, np.asarray(self.cyclic, dtype=bool)
        order = np.argsort(ground[cyc])
        cumulative = np.cumsum(self.area_m2[cyc][order]) / self.area_m2[cyc].sum()
        lowest = float(ground[cyc][order][np.searchsorted(cumulative, 0.02)])
        self.drain_reference_m = lowest - band

    def backwater_fraction(self, day: int) -> np.ndarray:
        """Share of each paddy's drainage the canal holds back on ``day`` (0 = drains freely)."""
        if self.canal_depth_m is None:
            return np.zeros(self.n)
        d = min(max(int(day), 0), len(self.canal_depth_m["North"]) - 1)
        depth = np.where(np.asarray(self.block) == "North", self.canal_depth_m["North"][d],
                         self.canal_depth_m["South"][d])
        margin = (self.ground_elev_m - self.drain_reference_m) - (depth - self.normal_canal_depth_m)
        band = max(float(self.config.backwater_band_m), 1e-6)
        blocked = np.clip(1.0 - margin / band, 0.0, 1.0)
        return np.where(np.asarray(self.cyclic, dtype=bool) & np.isfinite(margin), blocked, 0.0)

    @property
    def drain_transit_m3(self) -> float:
        """Cyclic surface drainage released but not yet at the pool (m3)."""
        return float(self.drain_pending_m3.sum())

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
        if self.landscape is not None:
            self.owner_id = assign_owners_along_canal_variable(
                self.landscape.owner_seed_order, self.n, self.config.paddies_per_farmer,
                self.config.farm_size_variability, self._draw_ownership_seed(),
            )
        else:
            self.owner_id = assign_owners(self.n, self.config.paddies_per_farmer)

    def n_farmers(self) -> int:
        return int(self.owner_id.max()) + 1 if self.n else 0

    def paddies_of(self, farmer_id: int) -> np.ndarray:
        return np.where(self.owner_id == farmer_id)[0]

    def neighbour_ids(self, paddy: int) -> list[int]:
        return list(self.neighbours[int(paddy)])

    def today_growth(self) -> crop.GrowthDay:
        i = min(self.day, len(self.et0_series) - 1)
        return crop.growth_day(self.day, self.et0_series[i])

    def et_demand_mm(self, day: int) -> float:
        """Weather-driven ET demand (mm) of a well-watered paddy on ``day`` — the same
        dual-Kc formula ``cyclic.physics.advance`` uses, at the wet-day Kc cap."""
        d = min(max(int(day), 0), len(self.et0_series) - 1)
        p = self.soil_physics
        fc = crop.canopy_cover(d)
        kcb = p.kcb_bare + (p.kcb_full - p.kcb_bare) * fc
        few = min(max(1.0 - fc, 0.05), 1.0)
        cap = float(p.kc_wet)
        return float(min(cap, kcb + few * cap) * self.et0_series[d])

    def canopy_cover(self, day: int) -> np.ndarray:
        """Per-paddy canopy cover on each paddy's own (shifted) crop calendar."""
        shifts = np.asarray(self.calendar_shift_days, dtype=int)
        cover = {int(s): crop.canopy_cover(int(day) + int(s)) for s in np.unique(shifts)}
        return np.array([cover[int(s)] for s in shifts], dtype=float)

    def et_demand_for(self, day: int, shift_days) -> np.ndarray:
        """``et_demand_mm`` for paddies whose crop runs ``shift_days`` ahead."""
        d = min(max(int(day), 0), len(self.et0_series) - 1)
        p = self.soil_physics
        fc = np.array([crop.canopy_cover(int(day) + int(s)) for s in np.atleast_1d(shift_days)])
        kcb = p.kcb_bare + (p.kcb_full - p.kcb_bare) * fc
        few = np.clip(1.0 - fc, 0.05, 1.0)
        cap = float(p.kc_wet)
        return np.minimum(cap, kcb + few * cap) * float(self.et0_series[d])

    def season_date(self, day: int):
        from cyclic import okura

        import pandas as pd
        return okura.season_dates(self.config.year)[0] + pd.Timedelta(days=int(day))

    def soil_storage_m3(self) -> float:
        """Water held on and in every paddy (pond + soil + perched), m3."""
        return float(np.dot(self.soil.storage(), self.area_m2) / 1000.0)

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
        edges = getattr(self, "edges", None)
        if edges is not None:
            for name, (rr, cc) in dirs.items():
                j = int(getattr(edges, name)[paddy])
                if j >= 0:
                    grid_pond[rr, cc] = self.ponding_mm[j]
                    grid_own[rr, cc] = self.owner_id[j]
                    grid_ids[rr, cc] = j
        else:
            for j in self.neighbour_ids(paddy):
                # Fall back: first free slot.
                for rr, cc in ((0, 1), (2, 1), (1, 0), (1, 2)):
                    if not np.isfinite(grid_pond[rr, cc]):
                        grid_pond[rr, cc] = self.ponding_mm[j]
                        grid_own[rr, cc] = self.owner_id[j]
                        grid_ids[rr, cc] = j
                        break
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
        self.__post_init__()
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
            "perched_mm": self.soil.perched.copy(),
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
            "use_cyclic_map": bool(self.config.use_cyclic_map),
            "pool_distance_m": self.pool_distance_m.copy(),
            "using_real_ground_water_law": self.ground_water_law is not None,
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
        obs["topsoil_available_mm"] = np.maximum(self.soil_physics.total_evaporable_mm - self.soil.surface, 0.0)
        # Drainage already on its way to the pool (arrives today and later).
        obs["drain_transit_m3"] = self.drain_transit_m3
        cap = self.config.pump_capacity_m3_s
        obs["pump_capacity_m3"] = float(cap * 86_400.0) if cap else float("inf")
        if self.canal_depth_m is not None:
            d = min(self.day, len(self.canal_depth_m["North"]) - 1)
            obs["canal_depth_m"] = {arm: float(v[d]) for arm, v in self.canal_depth_m.items()}
        p = self.soil_physics
        obs["pond_physics"] = {
            "percolation_k_mm_d": float(p.plow_pan_k_mm_d),
            "pond_ref_mm": float(p.pond_infiltrate_ref_mm),
            "outlet_loss_per_day": float(p.outlet_loss_per_day),
            "bund_mm": float(p.bund_mm),
        }
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
        obs["water_price_per_m3"] = float(self.config.water_price_per_m3)
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

    def _ration_by_pool_distance(self, sent: np.ndarray, available_m3: float) -> np.ndarray:
        """When the pool can't cover every request, paddies nearest the pool draw
        first (full request), in ascending ``pool_distance_m`` order, until the
        pool runs dry — not a uniform proportional cut across the whole field."""
        cost_m3 = sent * self.area_m2 / 1000.0
        order = np.argsort(self.pool_distance_m)
        rationed = np.zeros_like(sent)
        remaining = max(float(available_m3), 0.0)
        for idx in order:
            cost = cost_m3[idx]
            if cost <= 0.0:
                continue
            if cost <= remaining:
                rationed[idx] = sent[idx]
                remaining -= cost
            else:
                rationed[idx] = sent[idx] * (remaining / cost)
                remaining = 0.0
        return rationed

    def apply_station(self, action: dict) -> np.ndarray:
        """Dispatch plan for today. ``withdraw_m3`` is the most the lake gate may let in
        today (0 = gate shut); ``outflow_m3`` is pumped from the pool to the lake.

        What can be sent is the pool, plus the gate, plus the returns expected to flow
        back while the pumps run (yesterday's cyclic D + recaptured lateral flow).
        Short of that, the nearest paddies are served first. The pool itself is settled
        in ``step`` once the day's real returns are known.
        """
        self.gate_m3 = max(float(action.get("withdraw_m3", 0.0)), 0.0)
        outflow = max(float(action.get("outflow_m3", 0.0)), 0.0)
        sent = np.clip(
            np.asarray(action.get("irrigation_sent_mm", np.zeros(self.n)), dtype=float),
            0.0,
            None,
        ).reshape(self.n)
        available = self.pool_m3 + self.gate_m3 + self.last_return_m3
        self.planned_outflow_m3 = min(outflow, available)
        available -= self.planned_outflow_m3
        cap = self.config.pump_capacity_m3_s
        if cap:
            # The irrigation pumps cannot lift more than their capacity in a day.
            available = min(available, float(cap) * 86_400.0)
        need_m3 = float((sent * self.area_m2 / 1000.0).sum())
        if need_m3 > available and need_m3 > 0:
            sent = self._ration_by_pool_distance(sent, available)
        self.irrigation_sent_mm = sent
        return sent

    def step(self, apply_mm, drainage_mm, weir_mm=None, requested_mm=None) -> StepResult:
        """One day. ``weir_mm``: each paddy's outlet weir crest (pond above it leaves as
        D; None/inf = no board). ``requested_mm``: the farmers' demand, recorded only."""
        g = self.today_growth()
        area = self.area_m2
        sent = self.irrigation_sent_mm
        want = np.clip(np.asarray(apply_mm, dtype=float).reshape(self.n), 0.0, None)
        ir = np.minimum(want, sent)
        unapplied_m3 = float(((sent - ir) * area / 1000.0).sum())
        sent_m3 = float((sent * area / 1000.0).sum())

        i = min(self.day, len(self.rain_series) - 1)
        rain = float(self.rain_series[i])
        et0 = float(self.et0_series[i])
        dr_request = np.clip(np.asarray(drainage_mm, dtype=float).reshape(self.n), 0.0, None)

        # cyclic's paddy day: ET, D (discharge + overflow + outlet leak), percolation,
        # root zone, perched exchange with neighbours, Hooghoudt lateral, deep.
        weir = None if weir_mm is None else np.broadcast_to(np.asarray(weir_mm, dtype=float), (self.n,))
        request = np.zeros(self.n) if requested_mm is None else np.broadcast_to(
            np.asarray(requested_mm, dtype=float), (self.n,)).copy()
        flux = advance(
            self.soil, rain, et0, self.canopy_cover(self.day), ir, dr_request,
            self.soil_physics, edges=self.edges, area_m2=area, stress="supply",
            weir_mm=weir,
        )
        # Backwater: what a high canal holds back stays on the field (it seeps out later).
        frac = self.backwater_fraction(self.day)
        held = frac * flux["drainage_mm"]
        if held.any():
            self.soil.water += held
        dr = flux["drainage_mm"] - held
        leftover = flux["leftover_mm"] + held
        self.last_backwater_frac = frac
        eta = flux["et_mm"]
        lateral = flux["lateral_mm"]
        neighbour = flux["neighbor_mm"]
        to_pool = self.lateral_capture_i * lateral
        stay = np.where(self.cyclic, lateral - to_pool, 0.0)

        vol = lambda mm, mask=slice(None): float((mm[mask] * area[mask] / 1000.0).sum())
        # Cyclic D travels the drains to the pool; today's arrivals include earlier days' D.
        released = np.where(self.cyclic, dr * area / 1000.0, 0.0)
        routed = self.drain_pending_m3 + released @ self.drain_share
        return_d_m3 = float(routed[0])
        self.drain_pending_m3 = np.append(routed[1:], 0.0)
        return_lfv_m3 = vol(to_pool)
        # Non-cyclic drains (surface and lateral) run back to the lake / Tone River.
        to_lake_m3 = vol(dr, ~self.cyclic) + vol(lateral, ~self.cyclic)
        self.dwb_m3 += vol(stay)
        self.deep_m3 += vol(flux["deep_mm"])

        # Settle the pool: pumps drew `sent`, unused send and today's returns came back,
        # the station pumped its outflow, the lake gate refilled toward the lake level.
        pool = self.pool_m3 - sent_m3 + unapplied_m3 + return_d_m3 + return_lfv_m3
        outflow = min(self.planned_outflow_m3, max(pool, 0.0))
        pool -= outflow
        withdraw = min(self.gate_m3, max(self.pool_capacity_m3 - pool, 0.0))
        pool += withdraw
        if pool < 0.0 and self.gate_m3 > 0.0:
            # Returns fell short of the forecast the dispatch relied on; the water is
            # already on the fields, so the open culvert let the lake cover it.
            withdraw -= pool
            pool = 0.0
        # With the gate shut a shortfall stays as negative pool: the pumps drew the
        # main drainage canal down (Okura Fig. 3), repaid from later returns before
        # anything more can be sent (apply_station counts pool_m3 as available).
        spill = max(pool - self.pool_capacity_m3, 0.0)
        pool -= spill
        self.pool_m3 = pool
        self.lake_m3 += outflow + spill + to_lake_m3 - withdraw

        self.last_hop_in = neighbour.copy()
        self.last_ir = ir.copy()
        self.last_dr = dr.copy()
        self.last_leftover = leftover.copy()
        self.last_to_pool = to_pool.copy()
        self.last_eta = eta.copy()
        self.last_seepage = flux["infiltration_mm"] + flux["spill_mm"] * (1.0 - frac)
        self.last_withdraw_m3 = withdraw
        self.last_request = request
        self.request_cum += request
        self.last_outflow_m3 = outflow + spill
        self.last_spill_m3 = spill
        self.last_return_m3 = return_d_m3 + return_lfv_m3
        self.ks = np.asarray(flux["ks"], dtype=float).copy()
        self.eta_cum += eta
        self.etp_cum += flux["et_potential_mm"]
        self.last_stress = flux["stress_mm"].copy()
        self.stress_cum += self.last_stress
        self.etc_cum += g.etc_mm
        self.relative_yield = np.clip(
            1.0 - crop.KY_RICE * (1.0 - self.eta_cum / np.maximum(self.etp_cum, 1e-9)), 0.0, 1.0
        )

        result = StepResult(
            day=self.day,
            stage=g.stage,
            rain_mm=rain,
            et0_mm=et0,
            kc=g.kc,
            etc_mm=g.etc_mm,
            target_depth_mm=g.target_depth_mm,
            irrigation_sent_mm=sent.copy(),
            irrigation_mm=ir,
            drainage_mm=dr.copy(),
            leftover_mm=leftover.copy(),
            to_pool_mm=to_pool,
            to_neighbour_mm=np.maximum(-neighbour, 0.0),
            stay_mm=stay,
            hop_in_mm=neighbour.copy(),
            ponding_mm=self.ponding_mm.copy(),
            ks=self.ks.copy(),
            eta_mm=eta.copy(),
            et_potential_mm=flux["et_potential_mm"].copy(),
            stress_mm=flux["stress_mm"].copy(),
            infiltration_mm=flux["infiltration_mm"].copy(),
            lateral_mm=lateral.copy(),
            deep_mm=flux["deep_mm"].copy(),
            owner_id=self.owner_id.copy(),
            withdraw_m3=float(withdraw),
            outflow_m3=float(outflow + spill),
            pool_spill_m3=float(spill),
            pool_m3=float(self.pool_m3),
            lake_m3=float(self.lake_m3),
            return_d_m3=return_d_m3,
            return_lfv_m3=return_lfv_m3,
            unapplied_m3=unapplied_m3,
            dwb_m3=float(self.dwb_m3),
            mean_relative_yield=float(self.relative_yield.mean()),
            drain_transit_m3=self.drain_transit_m3,
            backwater_mm=held.copy(),
            canal_depth_north_m=float(self.canal_depth_m["North"][min(self.day, len(self.canal_depth_m["North"]) - 1)]) if self.canal_depth_m else float("nan"),
            canal_depth_south_m=float(self.canal_depth_m["South"][min(self.day, len(self.canal_depth_m["South"]) - 1)]) if self.canal_depth_m else float("nan"),
            requested_mm=request,
            weir_mm=None if weir is None else weir.copy(),
        )
        self.history.append(result)
        self.day += 1
        self.irrigation_sent_mm = np.zeros(self.n)
        self.gate_m3 = 0.0
        self.planned_outflow_m3 = 0.0
        return result

    def done(self) -> bool:
        return self.day >= int(self.config.days or 0)

    def grid(self, values: np.ndarray) -> np.ndarray:
        """Legacy square reshape — prefer lon/lat scatter for cyclic maps."""
        side = int(np.ceil(np.sqrt(self.n)))
        out = np.full((side, side), np.nan)
        flat = np.asarray(values, dtype=float).ravel()
        out.ravel()[: flat.size] = flat[: side * side]
        return out

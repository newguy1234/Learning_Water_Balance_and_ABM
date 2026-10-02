"""The water world as a Mesa model: the paddies, their water, the pool and the station's
pumps. abm.DistrictModel is this model plus farmer and station agents; cyclic.learn
steps the same stem.water equations to fit the ratios.

    field = Field(FieldConfig(year=2021))       # 4,002 Paddy agents on the real district
    field.apply_station({"irrigation_sent_mm": send, "withdraw_m3": gate})
    result = field.day_step(apply_mm, drainage_mm)          # one day → StepResult

What is in it (sources in stem/bundled/SOURCES.md):
  * the district's paddies as `Paddy` agents (block, cyclic status, area, neighbours,
    hydrology ratios, pond and soil water); water stepped for all at once by stem.water
  * weather: Open-Meteo rain (scaled to the site gauge, Okura Table 2) and ET0, optionally
    jittered (`weather_variability`) or reshaped by a scenario (stem.weather)
  * ET demand: FAO-56 dual Kc × ET0 on each paddy's crop calendar (stem.crop); the pond,
    then the soil, supply it and the rest is stress; FAO-33 relative yield
  * cyclic drainage reaching the pool along the mapped channels at `drain_velocity_m_s`
  * backwater (Okura §4.1): a high main canal (Okura Fig. 3) holds low paddies' drainage
  * the 18 × 68 m pool, the lake gate, spill (O) and the pumps' capacity; when supply is
    short the paddies nearest the station are served first

Hard rules: applied ≤ sent; discharge ≤ ponded water.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mesa
import numpy as np

from . import crop, water
from .district import load_district
from .params import BUND_MM, KC_WET, KCB_BARE, KCB_FULL, POOL_CAPACITY_M3, PUMP_CAPACITY_M3_S, SOIL_MAX_MM


@dataclass
class FieldConfig:
    """The physical season: weather, infrastructure, ratios."""

    year: int = 2022
    days: int | None = None  # None → full Open-Meteo season length
    seed: int = 0
    # None → full pool (pool_capacity_m3).
    initial_pool_m3: float | None = None
    # None → POOL_CAPACITY_M3 (18 × 68 m, 4 m deep).
    pool_capacity_m3: float | None = None
    # Running tally of net lake exchange only. The lake is a boundary condition
    # (Conceptual Model, Scope table): withdrawals are never limited by it.
    initial_lake_m3: float = 5_000_000.0
    # Bund/levee height: real paddy bunds run ~15-30 cm and spill uncontrolled surface
    # runoff once exceeded (IRRI water-management guidance; Chapagain & Yamaji 2010).
    # Passed to stem.water as the bund height.
    bund_height_mm: float = BUND_MM
    # Scale Open-Meteo rain to the on-site gauge's monthly totals (Okura Table 2).
    # Open-Meteo overstates season rain by ~30-50% at Imbanuma; years without a gauge
    # record use the mean monthly ratio of the gauged years.
    rain_gauge_correction: bool = True
    # Day-to-day weather jitter around the historical record (0 = exact replay of
    # that year's rain/ET0). Weather the farmer cannot see coming or
    # steer, only react to; see perturb_weather.
    weather_variability: float = 0.0
    # Weather scenario (stem.weather.WeatherScenario or a preset name such as
    # "drought"): rain/ET0 scaling, dry spells, heatwaves, storms, jitter. When set,
    # its own ``variability`` replaces weather_variability.
    weather: object | None = None
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
    pump_capacity_m3_s: float | None = PUMP_CAPACITY_M3_S
    # Per-paddy hydrology ratios (stem.water.Ratios). None = the ratios fitted on the
    # measured seasons (stem.water.fitted_ratios), or uniform ones if none are fitted yet.
    ratios: Any = None


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
    to_pool_mm: np.ndarray        # cyclic lateral flow → pool (LFV_pump)
    to_neighbour_mm: np.ndarray   # soil water sent to neighbouring paddies (net)
    stay_mm: np.ndarray           # cyclic infiltration not reaching the pool → ΔWB
    hop_in_mm: np.ndarray         # net soil exchange with neighbours (+ = received)
    ponding_mm: np.ndarray
    ks: np.ndarray
    eta_mm: np.ndarray
    et_potential_mm: np.ndarray   # weather-driven ET demand
    stress_mm: np.ndarray         # demand the paddy's water could not supply
    infiltration_mm: np.ndarray
    lateral_mm: np.ndarray        # soil water to the main drainage (lateral share)
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


class Paddy(mesa.Agent):
    """One parcel. Static attributes come from the district; pond, soil and ratios are
    read from (and written to) the Field's arrays."""

    def __init__(self, model: "Field", index: int):
        super().__init__(model)
        self.index = int(index)
        d = model.district
        self.paddy_id = str(d.paddy_id[index])
        self.block = str(d.block[index])
        self.cyclic = bool(d.cyclic[index])
        self.area_m2 = float(d.area_m2[index])
        self.lonlat = (float(d.lon[index]), float(d.lat[index]))
        self.neighbour_index = {name: int(j) for name, j in zip(("east", "west", "north", "south"),
                                                                  d.neighbours[index]) if j >= 0}

    @property
    def neighbours(self) -> list["Paddy"]:
        return [self.model.paddies[j] for j in self.neighbour_index.values()]

    @property
    def pond(self) -> float:
        return float(self.model.pond[self.index])

    @pond.setter
    def pond(self, value: float) -> None:
        self.model.pond[self.index] = float(value)

    @property
    def soil(self) -> float:
        return float(self.model.soil[self.index])

    @property
    def kz(self) -> float:
        return float(self.model.ratios.kz[self.index])

    @property
    def shares(self) -> dict:
        return dict(zip(water.SHARES, map(float, self.model.ratios.shares[self.index])))


class Field(mesa.Model):
    """The district's water world. Subclasses add who decides (abm.WaterWorld, DistrictModel)."""

    def __init__(self, config: FieldConfig | None = None, seed=None):
        config = config or FieldConfig()
        super().__init__(seed=config.seed if seed is None else seed)
        self.config = config
        self.district = d = load_district(config.year)
        self.n = d.n
        self.lon, self.lat = d.lon, d.lat
        self.area_m2, self.block, self.cyclic, self.paddy_id = d.area_m2, d.block, d.cyclic, d.paddy_id
        self.canal_distance_m, self.pool_distance_m = d.canal_distance_m, d.pool_distance_m
        self.drain_entry_m, self.drain_path_m = d.drain_entry_m, d.drain_path_m
        self.ground_elev_m = d.elevation_m
        self.neighbour_index = d.neighbours
        self.neighbours = [[int(j) for j in row if j >= 0] for row in d.neighbours]
        from .infrastructure import STATION_LONLAT
        self.station_lonlat = STATION_LONLAT
        self.grid_net = water.Grid(d.neighbours, d.area_m2)
        self.pond = np.zeros(self.n)
        self.soil = np.zeros(self.n)
        self.set_ratios(config.ratios)
        self.paddies = [Paddy(self, i) for i in range(self.n)]
        self.reset_season()

    # --- ratios ---------------------------------------------------------------------------------

    def set_ratios(self, ratios: water.Ratios | None = None) -> None:
        """These ratios, else the ones fitted on measured data (cyclic.learn), else defaults."""
        fitted = water.fitted_ratios() if ratios is None else None
        self.ratios_source = "given" if ratios is not None else ("fitted" if fitted is not None else "default")
        self.ratios = r = ratios if ratios is not None else (fitted or water.Ratios.uniform(self.n))
        self.claim = {"given": "hydrology ratios given in the config",
                      "fitted": "hydrology ratios fitted on the measured seasons",
                      "default": "uniform hydrology ratios (none fitted yet)"}[self.ratios_source]
        self.learned_ratios = self.ratios_source == "fitted"
        self.capture_i = np.where(self.cyclic, r.share("lateral"), 0.0)
        self.lateral_capture_i = self.capture_i
        self.neighbour_i = r.share("kx") + r.share("ky")
        self.stay_i = r.share("stay")
        mask = self.cyclic if self.cyclic.any() else slice(None)
        self.capture = float(self.capture_i[mask].mean())
        self.neighbour = float(self.neighbour_i[mask].mean())
        self.stay = float(self.stay_i[mask].mean())

    # --- season ---------------------------------------------------------------------------------

    def reset_season(self) -> None:
        """Weather series for the config and fresh water, pool and counters."""
        cfg = self.config
        self.bund_height_mm = float(cfg.bund_height_mm)
        self.pool_capacity_m3 = float(cfg.pool_capacity_m3 if cfg.pool_capacity_m3 is not None else POOL_CAPACITY_M3)
        d = self.district
        rain = d.rain_mm if cfg.rain_gauge_correction else np.nan_to_num(d.weather["rain_mm"].to_numpy())
        if cfg.weather is not None:
            from .weather import weather_scenario

            self.weather = weather_scenario(cfg.weather)
            self.rain_series, self.et0_series = self.weather.apply(rain, d.et0_mm, cfg.year, cfg.seed)
        else:
            self.weather = None
            self.rain_series, self.et0_series = perturb_weather(rain, d.et0_mm, cfg.weather_variability, cfg.seed)
        if cfg.days:
            self.rain_series, self.et0_series = self.rain_series[:cfg.days], self.et0_series[:cfg.days]
        cfg.days = len(self.rain_series)
        self._reset_runtime(cfg)

    @property
    def geometry(self):
        """Paddy polygons (GeoSeries, lon/lat) for maps."""
        if getattr(self, "_geometry", None) is None:
            import geopandas as gpd

            self._geometry = gpd.GeoSeries.from_wkt(self.district.geometry_wkt, crs="EPSG:4326")
        return self._geometry

    @property
    def ponding_mm(self) -> np.ndarray:
        return self.pond

    @ponding_mm.setter
    def ponding_mm(self, value) -> None:
        self.pond = np.asarray(value, dtype=float).copy()

    @property
    def soil_mm(self) -> np.ndarray:
        return self.soil

    def water_day(self, rain, irrigation, discharge, et_demand, weir=None) -> dict:
        """stem.water on every paddy for one day; the day's fluxes (mm per paddy)."""
        self.pond, self.soil, flux = water.step_numpy(
            self.pond, self.soil, rain, irrigation, discharge, et_demand, self.ratios, self.grid_net,
            bund=self.bund_height_mm, weir=weir)
        return flux

    @staticmethod
    def et_potential(et0_mm, canopy_cover):
        """Well-watered ET demand: FAO-56 dual Kc at the wet-day cap × ET0 (Allen et al. 1998)."""
        fc = np.asarray(canopy_cover, dtype=float)
        kcb = KCB_BARE + (KCB_FULL - KCB_BARE) * fc
        few = np.clip(1.0 - fc, 0.05, 1.0)
        return np.minimum(KC_WET, kcb + few * KC_WET) * float(et0_mm)

    def _reset_runtime(self, cfg: FieldConfig):
        self.day = 0
        self.history: list[StepResult] = []
        rng = np.random.default_rng(cfg.seed)
        self.pond = rng.uniform(40.0, 70.0, size=self.n)
        self.soil = np.full(self.n, SOIL_MAX_MM)
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
        # its owner's calendar_lead (abm.model.DistrictModel keeps it in sync).
        self.calendar_shift_days = np.zeros(self.n, dtype=int)
        self.last_request = np.zeros(self.n)
        self.request_cum = np.zeros(self.n)

    def _set_drain_routing(self, cfg: FieldConfig) -> None:
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

    def _set_backwater(self, cfg: FieldConfig) -> None:
        """Daily canal depth per arm and each cyclic paddy's drain reference (see module doc)."""
        days = int(cfg.days or len(self.rain_series))
        self.last_backwater_frac = np.zeros(self.n)
        self.canal_depth_m = None
        if cfg.canal_depth is None or self.ground_elev_m is None or not np.isfinite(self.ground_elev_m).any():
            return
        from stem import infrastructure

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

    def today_growth(self) -> crop.GrowthDay:
        i = min(self.day, len(self.et0_series) - 1)
        return crop.growth_day(self.day, self.et0_series[i])

    def et_demand_mm(self, day: int) -> float:
        """Weather-driven ET demand (mm) of a well-watered paddy on ``day`` — the same
        FAO-56 dual-Kc formula at the wet-day Kc cap."""
        d = min(max(int(day), 0), len(self.et0_series) - 1)
        return float(Field.et_potential(self.et0_series[d], crop.canopy_cover(d)))

    def canopy_cover(self, day: int) -> np.ndarray:
        """Per-paddy canopy cover on each paddy's own (shifted) crop calendar."""
        shifts = np.asarray(self.calendar_shift_days, dtype=int)
        cover = {int(s): crop.canopy_cover(int(day) + int(s)) for s in np.unique(shifts)}
        return np.array([cover[int(s)] for s in shifts], dtype=float)

    def et_demand_for(self, day: int, shift_days) -> np.ndarray:
        """``et_demand_mm`` for paddies whose crop runs ``shift_days`` ahead."""
        d = min(max(int(day), 0), len(self.et0_series) - 1)
        shifts = np.atleast_1d(np.asarray(shift_days, dtype=int))
        cover = {int(v): crop.canopy_cover(int(day) + int(v)) for v in np.unique(shifts)}
        return Field.et_potential(self.et0_series[d], np.array([cover[int(v)] for v in shifts]))

    def season_date(self, day: int):
        from stem import okura

        import pandas as pd
        return okura.season_dates(self.config.year)[0] + pd.Timedelta(days=int(day))

    def soil_storage_m3(self) -> float:
        """Water held on and in every paddy (pond + soil), m3."""
        return float(np.dot(self.pond + self.soil_mm, self.area_m2) / 1000.0)

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

    def day_step(self, apply_mm, drainage_mm, weir_mm=None, requested_mm=None, et_demand_mm=None) -> StepResult:
        """One day. ``weir_mm``: each paddy's outlet weir crest (pond above it leaves as
        D; None/inf = no board). ``requested_mm``: the farmers' demand, recorded only.
        ``et_demand_mm``: each paddy's ET demand today (e.g. satellite ET); None = FAO-56
        potential on its crop calendar."""
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

        # cyclic.water: ET from pond then soil, discharge (+ weir) and overflow, kz into
        # the soil, soil water routed by the paddy's ratios.
        weir = None if weir_mm is None else np.broadcast_to(np.asarray(weir_mm, dtype=float), (self.n,))
        request = np.zeros(self.n) if requested_mm is None else np.broadcast_to(
            np.asarray(requested_mm, dtype=float), (self.n,)).copy()
        demand = (self.et_demand_for(self.day, self.calendar_shift_days) if et_demand_mm is None
                  else np.broadcast_to(np.asarray(et_demand_mm, dtype=float), (self.n,)))
        flux = self.water_day(rain, ir, dr_request, demand, weir=weir)
        # Backwater: what a high canal holds back stays on the field (it seeps out later).
        frac = self.backwater_fraction(self.day)
        held = frac * flux["drainage_mm"]
        if held.any():
            self.pond = self.pond + held
        dr = flux["drainage_mm"] - held
        eta = flux["et_mm"]
        leftover = rain + ir - dr - eta
        self.last_backwater_frac = frac
        lateral = flux["lateral_mm"]
        neighbour = flux["received_mm"] - flux["to_x_mm"] - flux["to_y_mm"]
        to_pool = np.where(self.cyclic, lateral, 0.0)
        stay = np.where(self.cyclic, flux["infiltration_mm"] - lateral, 0.0)
        ks = np.divide(eta, demand, out=np.ones(self.n), where=demand > 0)

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
        self.last_seepage = flux["infiltration_mm"]
        self.last_withdraw_m3 = withdraw
        self.last_request = request
        self.request_cum += request
        self.last_outflow_m3 = outflow + spill
        self.last_spill_m3 = spill
        self.last_return_m3 = return_d_m3 + return_lfv_m3
        self.ks = ks
        self.eta_cum += eta
        self.etp_cum += demand
        self.last_stress = flux["unmet_mm"].copy()
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
            et_potential_mm=np.asarray(demand, dtype=float).copy(),
            stress_mm=flux["unmet_mm"].copy(),
            infiltration_mm=flux["infiltration_mm"].copy(),
            lateral_mm=lateral.copy(),
            deep_mm=flux["deep_mm"].copy(),
            owner_id=np.array(getattr(self, "owner_id", np.zeros(self.n, dtype=int))),
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

"""Farmers, leftover-ratio agents, the registries they are picked from, and the pumping station.

Three kinds of agent are registered outside this package (see example.py). They
are not one joint net. The gym couples a farmer model to a hydrology equation:

    Farmer       FarmerPolicy.decide_batch(obs) -> irrigation_mm, discharge_mm
                 Applied water and sent-back D. IR is overwritten by the gym when
                 given_irrigation is True (Okura block total; the algorithm may
                 only propose a per-paddy split, renormalised).
    Lateral      LateralPolicy.flux(...) -> after, delta
                 LateralPolicy.partition(...) -> Kx, Ky, Kz, ground at every paddy
                 Equation builder H: neighbour water and neighbour FAO-56 ET on
                 up/down/left/right of every paddy, plus leftover collection shares.
                 No soil K; no applied IR share. Frozen if the farmer irrigates
                 differently. Optional `@register_ratio` can override partition.

    A farmer answers one question for one paddy on one morning: given yesterday's weather,
the last satellite pass, local stores, and (in the gym) how much the pool sent, how
much surface water to send back? `agents=` is one name / class / instance, a dict
{"North": ..., "South": ...}, or a function (index, inputs) -> spec per paddy.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import mesa
import numpy as np
import pandas as pd

from .physics import KC_INI, NDVI_BARE, kc_from_ndvi
from .data import WEATHER_OBS

AGENTS = {}
RATIOS = {}
LATERALS = {}
RATIO_SHARE_KEYS = ("kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio",
                    "farm_to_main", "pool_capture_share")


def _register(registry, name_or_class=None, *, name=None):
    def add(cls, key=None):
        key = key or name or getattr(cls, "name", None) or cls.__name__
        cls.name = key
        registry[key] = cls
        return cls

    if isinstance(name_or_class, type):
        return add(name_or_class)
    if isinstance(name_or_class, str):
        return lambda cls: add(cls, name_or_class)
    return add


def register_agent(name_or_class=None, *, name=None):
    """Add a FarmerPolicy to the registry. Works as @register_agent, @register_agent("x")
    or register_agent(MyClass)."""
    return _register(AGENTS, name_or_class, name=name)


def register_ratio(name_or_class=None, *, name=None):
    """Add a RatioPolicy to the leftover-share registry. Same decorator shapes as
    register_agent."""
    return _register(RATIOS, name_or_class, name=name)


def register_lateral(name_or_class=None, *, name=None):
    """Add a neighbour-head lateral-flow equation. Same decorator shapes as
    register_agent. The law may not take soil properties."""
    return _register(LATERALS, name_or_class, name=name)


def make_agent(spec, **params):
    """A farmer policy instance from a registered name, a class, or an instance."""
    if isinstance(spec, FarmerPolicy):
        return spec
    if isinstance(spec, type) and issubclass(spec, FarmerPolicy):
        return spec(**params)
    if isinstance(spec, str):
        if spec not in AGENTS:
            raise KeyError(f"no agent registered as {spec!r}; registered: {sorted(AGENTS)}")
        return AGENTS[spec](**params)
    raise TypeError(f"cannot make a farmer policy from {spec!r}")


def make_ratio(spec, **params):
    """A leftover-ratio policy from a registered name, a class, or an instance."""
    if isinstance(spec, RatioPolicy):
        return spec
    if isinstance(spec, type) and issubclass(spec, RatioPolicy):
        return spec(**params)
    if isinstance(spec, str):
        if spec not in RATIOS:
            raise KeyError(f"no ratio agent registered as {spec!r}; registered: {sorted(RATIOS)}")
        return RATIOS[spec](**params)
    raise TypeError(f"cannot make a ratio policy from {spec!r}")


def make_lateral(spec, **params):
    """A neighbour-head lateral equation from a registered name, class, or instance."""
    if isinstance(spec, LateralPolicy):
        return spec
    if isinstance(spec, type) and issubclass(spec, LateralPolicy):
        return spec(**params)
    if isinstance(spec, str):
        if spec not in LATERALS:
            raise KeyError(f"no lateral equation registered as {spec!r}; "
                           f"registered: {sorted(LATERALS)}")
        return LATERALS[spec](**params)
    raise TypeError(f"cannot make a lateral equation from {spec!r}")


# --- what a farmer sees ------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    paddy_id: str
    block: str
    cyclic: bool
    area_m2: float
    day: int                  # 0 = April 17
    date: pd.Timestamp
    sat_ndvi: float           # last cloud-free Sentinel-2 pass over this plot (NaN: none yet)
    sat_et_mm: float          # Kc(sat_ndvi) x ET0 on that pass day
    sat_days_ago: float       # age of that pass in days
    rain_mm: float            # yesterday, Open-Meteo precipitation
    et0_mm: float             # yesterday, Open-Meteo FAO-56 ET0
    radiation_mj: float       # yesterday, shortwave MJ/m²
    tmean_c: float
    tmax_c: float
    tmin_c: float
    atmean_c: float           # apparent (feels-like) temperature
    atmax_c: float
    atmin_c: float
    wind_mean_kmh: float      # 10 m
    wind_max_kmh: float
    wind_gust_kmh: float
    wind_dir_deg: float       # dominant direction
    rain_hours: float
    snow_cm: float
    weather_code: float       # WMO code
    sunshine_s: float
    daylight_s: float
    sunshine_h: float
    daylight_h: float
    rh_mean_pct: float
    rh_max_pct: float
    rh_min_pct: float
    dewpoint_c: float
    vpd_max_kpa: float
    cloud_pct: float
    pressure_hpa: float
    soil_temp_c: float        # 0-7 cm
    soil_moisture: float      # 0-7 cm m³/m³
    depth_mm: float           # standing water this morning
    pond_mm: float = 0.0      # pond stock (same water as depth_mm)
    topsoil_mm: float = 0.0   # evaporable topsoil remaining
    need_mm: float = 0.0      # yesterday's topsoil shortfall for satellite ET
    need_stock_mm: float = 0.0
    leftover_mm: float = 0.0
    canal_mm: float = 0.0
    drain_stock_mm: float = 0.0
    canal_distance_m: float = 0.0
    farm_drain_mm: float = 0.0
    irrigation_sent_mm: float = 0.0   # today's Okura pool IR at this point (gym forcing)
    last_discharge_mm: float = 0.0    # yesterday's surface return
    last_et_mm: float = 0.0
    last_lfv_mm: float = 0.0
    neighbour_depth_mm: float = 0.0      # mean ponding of neighbour plots
    neighbour_head_drop_mm: float = 0.0  # own depth minus that mean (wetter > 0)
    neighbour_et_mm: float = 0.0         # mean FAO-56 ET of neighbour plots
    neighbour_et_drop_mm: float = 0.0    # own ET minus that mean
    neighbour_right_mm: float = 0.0      # east neighbour water (0 if none)
    neighbour_left_mm: float = 0.0       # west
    neighbour_up_mm: float = 0.0         # north
    neighbour_down_mm: float = 0.0       # south
    neighbour_right_et_mm: float = 0.0
    neighbour_left_et_mm: float = 0.0
    neighbour_up_et_mm: float = 0.0
    neighbour_down_et_mm: float = 0.0


if set(WEATHER_OBS) - set(Observation.__dataclass_fields__):
    raise RuntimeError("Observation is missing Open-Meteo fields: "
                       + ", ".join(sorted(set(WEATHER_OBS) - set(Observation.__dataclass_fields__))))


STOCK_OBS = ("pond_mm", "topsoil_mm", "need_mm", "need_stock_mm", "leftover_mm",
             "canal_mm", "drain_stock_mm", "canal_distance_m", "farm_drain_mm",
             "irrigation_sent_mm", "last_discharge_mm", "last_et_mm", "last_lfv_mm",
             "neighbour_depth_mm", "neighbour_head_drop_mm",
             "neighbour_et_mm", "neighbour_et_drop_mm",
             "neighbour_right_mm", "neighbour_left_mm", "neighbour_up_mm", "neighbour_down_mm",
             "neighbour_right_et_mm", "neighbour_left_et_mm",
             "neighbour_up_et_mm", "neighbour_down_et_mm")

# Identity / geometry always kept when masking (needed to index paddies).
OBS_ALWAYS = ("paddy_id", "block", "cyclic", "area_m2")

# Named channel packs so a gym can hide/show different variables per agent role.
OBS_CHANNEL_PACKS = {
    "identity": OBS_ALWAYS,
    "weather": WEATHER_OBS,
    "satellite": ("sat_ndvi", "sat_et_mm", "sat_days_ago"),
    "own_stock": ("depth_mm", "pond_mm", "topsoil_mm", "need_mm", "need_stock_mm",
                  "leftover_mm", "canal_mm", "drain_stock_mm", "farm_drain_mm"),
    "forcing": ("irrigation_sent_mm", "canal_distance_m"),
    "yesterday_fate": ("last_discharge_mm", "last_et_mm", "last_lfv_mm"),
    "neighbour_summary": ("neighbour_depth_mm", "neighbour_head_drop_mm",
                          "neighbour_et_mm", "neighbour_et_drop_mm"),
    "neighbour_cardinal": ("neighbour_right_mm", "neighbour_left_mm",
                           "neighbour_up_mm", "neighbour_down_mm",
                           "neighbour_right_et_mm", "neighbour_left_et_mm",
                           "neighbour_up_et_mm", "neighbour_down_et_mm"),
}

# Default visibility: farmer sees weather/stock/forcing; equation learner sees
# neighbour heads/ET (and may hide satellite ET to block shortcuts).
DEFAULT_ROLE_CHANNELS = {
    "farmer": ("identity", "weather", "satellite", "own_stock", "forcing", "yesterday_fate"),
    "equation": ("identity", "own_stock", "forcing", "neighbour_summary", "neighbour_cardinal"),
    "ratio": ("identity", "own_stock", "forcing", "yesterday_fate", "neighbour_summary"),
    "full": tuple(OBS_CHANNEL_PACKS.keys()),
}


def resolve_observation_channels(channels=None, hide=None, show=None):
    """Expand pack names / field names into a frozenset of ObservationBatch fields.

    channels: None | 'full' | role name | iterable of pack or field names
    hide / show: optional field or pack names applied after the base set
    """
    if channels is None or channels == "full":
        names = set(ObservationBatch.FIELDS)
    elif isinstance(channels, str) and channels in DEFAULT_ROLE_CHANNELS:
        names = set()
        for pack in DEFAULT_ROLE_CHANNELS[channels]:
            names.update(OBS_CHANNEL_PACKS.get(pack, (pack,)))
    else:
        names = set(OBS_ALWAYS)
        for item in channels:
            if item in OBS_CHANNEL_PACKS:
                names.update(OBS_CHANNEL_PACKS[item])
            elif item in ObservationBatch.FIELDS or item in DEFAULT_ROLE_CHANNELS.get("full", ()):
                names.add(item)
            elif item in DEFAULT_ROLE_CHANNELS:
                for pack in DEFAULT_ROLE_CHANNELS[item]:
                    names.update(OBS_CHANNEL_PACKS.get(pack, (pack,)))
            else:
                names.add(item)
    for item in hide or ():
        drop = OBS_CHANNEL_PACKS.get(item, (item,))
        names.difference_update(drop)
    for item in show or ():
        add = OBS_CHANNEL_PACKS.get(item, (item,))
        names.update(add)
    names.update(OBS_ALWAYS)
    return frozenset(k for k in ObservationBatch.FIELDS if k in names)


class ObservationBatch:
    """The same fields as Observation, as arrays over many paddies on one morning."""

    FIELDS = ("paddy_id", "block", "cyclic", "area_m2", "sat_ndvi", "sat_et_mm",
              "sat_days_ago") + WEATHER_OBS + ("depth_mm",) + STOCK_OBS

    def __init__(self, day, date, **arrays):
        self.day = int(day)
        self.date = date
        n = len(np.asarray(arrays["depth_mm"]))
        for key in self.FIELDS:
            if key in arrays:
                setattr(self, key, np.asarray(arrays[key]))
            elif key in WEATHER_OBS or key in STOCK_OBS:
                setattr(self, key, np.zeros(n))
            else:
                raise KeyError(key)

    def __len__(self):
        return len(self.depth_mm)

    def subset(self, rows):
        return ObservationBatch(self.day, self.date, **{k: getattr(self, k)[rows] for k in self.FIELDS})

    def as_dict(self):
        """Plain numpy arrays, plus day, for agents that want a feature matrix."""
        return {"day": self.day, **{k: getattr(self, k) for k in self.FIELDS}}

    def masked(self, channels=None, hide=None, show=None, fill=0.0):
        """Copy with selected fields zeroed (hidden) for POMDP / ablation studies.

        Hidden numeric fields become `fill`; string/bool identity fields in
        OBS_ALWAYS are never removed. Use role presets via channels='farmer' etc.
        """
        keep = resolve_observation_channels(channels, hide=hide, show=show)
        arrays = {}
        n = len(self)
        for key in self.FIELDS:
            values = getattr(self, key)
            if key in keep:
                arrays[key] = values
            elif key in OBS_ALWAYS:
                arrays[key] = values
            elif key in ("paddy_id", "block"):
                arrays[key] = values
            elif key == "cyclic":
                arrays[key] = values
            else:
                arrays[key] = np.full(n, fill, dtype=float)
        out = ObservationBatch(self.day, self.date, **arrays)
        out.visible_channels = sorted(keep)
        return out

    def feature_matrix(self, channels=None, hide=None, show=None):
        """(n, d) float matrix of currently visible numeric channels only."""
        keep = resolve_observation_channels(channels, hide=hide, show=show)
        cols = []
        names = []
        for key in self.FIELDS:
            if key not in keep or key in ("paddy_id", "block", "cyclic"):
                continue
            cols.append(np.asarray(getattr(self, key), dtype=float).ravel())
            names.append(key)
        if not cols:
            return np.zeros((len(self), 0)), ()
        return np.column_stack(cols), tuple(names)

    def row(self, i):
        values = {}
        for key in self.FIELDS:
            value = getattr(self, key)[i]
            if key in ("paddy_id", "block"):
                values[key] = str(value)
            elif key == "cyclic":
                values[key] = bool(value)
            else:
                values[key] = float(value)
        return Observation(day=self.day, date=self.date, **values)

    @classmethod
    def of(cls, obs):
        return cls(obs.day, obs.date, **{k: np.array([getattr(obs, k)]) for k in cls.FIELDS})


def build_batch(inputs, day, depth_mm, area_m2, **stock):
    """Every paddy's observation on a morning. Day 0 has no yesterday in the season file,
    so it is shown its own day's weather. Stock fields (pond, topsoil, need, leftover,
    canal) are the paddy's own stores — not a farmer's target."""
    yesterday = max(day - 1, 0)
    n = len(depth_mm)
    weather = {name: np.full(n, getattr(inputs, f"seen_{name}")[yesterday]) for name in WEATHER_OBS}
    depth = np.asarray(depth_mm, dtype=float)
    extra = {}
    for key in STOCK_OBS:
        extra[key] = np.asarray(stock[key], dtype=float) if key in stock else (
            depth if key == "pond_mm" else np.zeros(n))
    if "canal_distance_m" not in stock and hasattr(inputs, "canal_distance_m"):
        extra["canal_distance_m"] = np.asarray(inputs.canal_distance_m, dtype=float)
    return ObservationBatch(
        day, inputs.dates[day],
        paddy_id=inputs.paddy_id, block=inputs.block, cyclic=inputs.cyclic, area_m2=area_m2,
        sat_ndvi=inputs.sat_ndvi[day], sat_et_mm=inputs.sat_et_mm[day],
        sat_days_ago=inputs.sat_days_ago[day],
        depth_mm=depth,
        **weather,
        **extra,
    )


def leftover_from_season(results):
    """Eq. 3 leftover on every plot from a finished (or partial) season: logged leftover
    if the model recorded it, otherwise IR + R − D − ET."""
    model = results.model if hasattr(results, "model") else results
    log = model.log or {}
    days = len(getattr(results, "daily", [])) or int(getattr(model, "day", 0)) or 1
    days = min(int(days), model.n_days)
    if "leftover_mm" in log:
        return np.asarray(log["leftover_mm"][:days], dtype=float)
    rain = np.asarray(model.inputs.rain_mm[:days], dtype=float)
    if rain.ndim == 1:
        rain = rain[:, None]
    irrigation = np.asarray(log.get("irrigation_mm", np.zeros((days, model.n)))[:days], dtype=float)
    discharge = np.asarray(log.get("farmer_discharge_mm", log.get("drainage_mm",
                           np.zeros((days, model.n))))[:days], dtype=float)
    et = np.asarray(log.get("et_mm", model.inputs.et_target_mm)[:days], dtype=float)
    return irrigation + rain - discharge - et


def point_targets(year=None, day=0, inputs=None, edges=None):
    """What is scored at every paddy on one morning.

    Forecaster (same number on every plot in a block — Okura only measured the pump /
    ADCP totals):
        block_irrigation_m3, block_drainage_m3
        block_irrigation_mm, block_drainage_mm   volume / cyclic area that day
    Plot ET:
        et_target_mm   satellite-derived target that day
    Leftover ratios (every cyclic point, Tables 4–5 means, canal geometry):
        kx_ratio, ky_ratio, kz_ratio, ground_ratio, farm_to_main, capture
    """
    from types import SimpleNamespace

    from . import imitation, okura
    from .canal import pixel_ratio_fields
    from .identify import construct_ratios
    from .lateral import cardinal_edges
    from .physics import Physics

    if inputs is None:
        from . import data
        year = okura.COMPARISON_YEAR if year is None else int(year)
        inputs = data.season_inputs(year)
    else:
        year = int(year if year is not None else getattr(inputs, "year", okura.COMPARISON_YEAR))
    n = len(inputs.paddy_id)
    day = int(np.clip(day, 0, len(inputs.dates) - 1))
    constructed = construct_ratios(year)
    physics = Physics()
    stay = float(constructed["kz_ratio"] + constructed["ground_ratio"])
    top = float(physics.total_evaporable_mm /
                max(physics.total_evaporable_mm + physics.total_available_mm, 1e-9))
    if edges is None:
        lon = inputs.frame["lon"].to_numpy(dtype=float)
        lat = inputs.frame["lat"].to_numpy(dtype=float)
        edges = cardinal_edges(lon, lat)
    dist = np.asarray(getattr(inputs, "canal_distance_m", np.zeros(n)), dtype=float)
    shares = pixel_ratio_fields(
        n, dist, edges, inputs.cyclic,
        constructed["pool_capture_share"], constructed["farm_to_main"],
        top, physics.canal_length_m,
        neighbour_of_remain=float(constructed.get(
            "neighbour_of_remain", physics.neighbour_of_remain)))
    observed, _, source = imitation.load_observed(year)
    dates = pd.to_datetime(inputs.dates)
    ir_m3 = np.zeros(n)
    d_m3 = np.zeros(n)
    ir_mm = np.zeros(n)
    d_mm = np.zeros(n)
    cyclic_area = {
        block: float(inputs.area_m2[(np.asarray(inputs.block) == block) & inputs.cyclic].sum())
        for block in okura.BLOCKS
    }
    for block in okura.BLOCKS:
        series = imitation.block_series(observed, block, dates)
        sel = np.asarray(inputs.block) == block
        ir_m3[sel] = series["irrigation_m3"][day]
        d_m3[sel] = series["drainage_m3"][day]
        area = max(cyclic_area[block], 1.0)
        ir_mm[sel] = series["irrigation_m3"][day] / area * 1000.0
        d_mm[sel] = series["drainage_m3"][day] / area * 1000.0
    et = np.asarray(inputs.et_target_mm[day], dtype=float)
    table = pd.DataFrame({
        "paddy_id": inputs.paddy_id,
        "block": inputs.block,
        "cyclic": inputs.cyclic,
        "area_m2": inputs.area_m2,
        "canal_distance_m": dist,
        "et_target_mm": et,
        "block_irrigation_m3": ir_m3,
        "block_drainage_m3": d_m3,
        "block_irrigation_mm": ir_mm,
        "block_drainage_mm": d_mm,
        "kx_ratio": shares["kx_ratio"],
        "ky_ratio": shares["ky_ratio"],
        "kz_ratio": shares["kz_ratio"],
        "ground_ratio": shares["ground_ratio"],
        "capture": shares["pool_capture_share"],
        "farm_to_main": shares["farm_to_main"],
    })
    return SimpleNamespace(
        year=year, day=day, date=inputs.dates[day], table=table, shares=shares,
        constructed=constructed, stay_target=stay, topsoil_frac=top, source=source,
        lfv_target_m3=okura.TABLE5_1000M3[year]["LFV_pump"] * 1000.0,
    )


# --- policies ------------------------------------------------------------------------------

class FarmerPolicy:
    """Base class. Override decide(obs) or decide_batch(batch); both return
    (irrigation_mm, discharge_mm). `defaults` are the parameters, `bounds` the ranges a
    learner may search."""

    name = "farmer"
    defaults: dict = {}
    bounds: dict = {}

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise TypeError(f"{type(self).__name__} has no parameters {sorted(unknown)}")
        self.params = {**self.defaults, **params}

    def __repr__(self):
        shown = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.params.items()
                 if not isinstance(v, np.ndarray)}
        return f"{type(self).__name__}({shown})"

    @property
    def batched(self):
        return type(self).decide_batch is not FarmerPolicy.decide_batch

    def reset(self, model, rows):
        """Called once when a model is built, with the paddy indices this policy farms."""

    # --- learning hooks, called by cyclic.imitate while training ----------------------------

    def learn(self, batch, irrigation, discharge, reward, info):
        """After every training day: the morning's batch, what this farmer did (arrays, mm),
        the day's reward in (0, 1], and info -- info["blocks"][block] holds the simulated
        and observed irrigation and drainage (mm) and that block's reward;
        info["paddy_reward"] is the reward per paddy. Does nothing unless overridden."""

    def end_episode(self, score, info):
        """After every training season, with imitation.score (fidelity, loss, table)."""

    @property
    def learns(self):
        return type(self).learn is not FarmerPolicy.learn

    def decide(self, obs):
        if not self.batched:
            raise NotImplementedError("implement decide(obs) or decide_batch(batch)")
        irrigation, discharge = self.decide_batch(ObservationBatch.of(obs))
        return float(irrigation[0]), float(discharge[0])

    def decide_batch(self, batch):
        pairs = [self.decide(batch.row(i)) for i in range(len(batch))]
        return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])

    @staticmethod
    def expected_et_mm(batch):
        """ET a farmer would expect today: Kc from the last satellite pass (Kc-ini before
        the first) times yesterday's ET0."""
        kc = np.where(np.isfinite(batch.sat_ndvi),
                      kc_from_ndvi(np.nan_to_num(batch.sat_ndvi, nan=NDVI_BARE)), KC_INI)
        return kc * batch.et0_mm


class RatioPolicy:
    """Leftover shares at every point: Kx (E–W), Ky (N–S), Kz (topsoil), ground.

    Override partition(leftover, observation, targets). leftover is mm, shape (days, n)
    or (n,); observation is an ObservationBatch; targets is point_targets(...).
    absorb(results) feeds leftover from a finished forecaster season.
    """

    name = "ratios"
    defaults: dict = {}

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise TypeError(f"{type(self).__name__} has no parameters {sorted(unknown)}")
        self.params = {**self.defaults, **params}
        self.shares = None

    def __repr__(self):
        shown = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.params.items()
                 if not isinstance(v, np.ndarray)}
        return f"{type(self).__name__}({shown})"

    def partition(self, leftover, observation, targets):
        shares = {key: np.array(targets.shares[key], copy=True) for key in RATIO_SHARE_KEYS
                  if key in targets.shares}
        if leftover is not None and np.nanmax(np.abs(np.asarray(leftover, dtype=float))) > 0:
            self.fit_stay_to_leftover(shares, leftover, observation, targets)
        return shares

    def fit_stay_to_leftover(self, shares, leftover, observation, targets):
        """Reweight Kz+ground so leftover-and-area-weighted stay matches Table 5."""
        from .identify import leftover_weighted_stay

        leftover_weighted_stay(
            shares, leftover, observation.area_m2, observation.cyclic,
            targets.stay_target, targets.topsoil_frac)
        return shares

    def absorb(self, results):
        """Use leftover from a finished (or partial) season, typically a forecaster run."""
        model = results.model
        leftover = leftover_from_season(results)
        day = int(np.clip(getattr(model, "day", 0) - 1, 0, model.n_days - 1))
        observation = getattr(model, "observations", None) or build_batch(
            model.inputs, day, model.state.water, model.area,
            leftover_mm=leftover[-1] if np.ndim(leftover) == 2 else leftover,
            canal_distance_m=np.asarray(getattr(model.inputs, "canal_distance_m", 0.0), dtype=float),
        )
        targets = point_targets(model.year, day=day, inputs=model.inputs, edges=model.paddy_edges)
        self.shares = self.partition(leftover, observation, targets)
        return self.shares

    def observe(self, results):
        """What a ratio agent can read off a finished season: leftover and the fluxes
        that made it, plus canal distance. Not a farmer target."""
        model = results.model
        leftover = leftover_from_season(results)
        days = leftover.shape[0] if np.ndim(leftover) == 2 else 1
        log = model.log or {}

        def season_sum(name, fallback=0.0):
            if name in log:
                arr = np.asarray(log[name][:days], dtype=float)
                return arr.sum(axis=0) if arr.ndim == 2 else arr
            return np.full(model.n, fallback, dtype=float)

        pile = leftover.sum(axis=0) if np.ndim(leftover) == 2 else leftover
        rain = np.asarray(model.inputs.rain_mm[:days], dtype=float)
        rain_mm = float(rain.sum()) if rain.ndim == 1 else rain.sum(axis=0)
        return {
            "paddy_id": model.inputs.paddy_id,
            "block": model.inputs.block,
            "cyclic": model.inputs.cyclic,
            "area_m2": model.area,
            "canal_distance_m": np.asarray(getattr(model.inputs, "canal_distance_m", 0.0), dtype=float),
            "leftover_mm": pile,
            "irrigation_mm": season_sum("irrigation_mm"),
            "discharge_mm": season_sum("farmer_discharge_mm"),
            "et_mm": season_sum("et_mm"),
            "et_target_mm": np.asarray(model.inputs.et_target_mm[:days], dtype=float).sum(axis=0),
            "rain_mm": rain_mm if np.ndim(rain_mm) else np.full(model.n, rain_mm),
        }

    def attach(self, model, shares=None):
        """Write shares onto a model (pixel_shares + district-mean Physics)."""
        from dataclasses import fields as physics_fields

        from .canal import weighted_share_mean
        from .physics import Physics

        shares = shares or self.shares
        if shares is None:
            raise RuntimeError("partition or absorb first")
        model.pixel_shares = {key: np.asarray(val) for key, val in shares.items()}
        allowed = {item.name for item in physics_fields(Physics)}
        means = {}
        for key in ("kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio", "farm_to_main"):
            if key in model.pixel_shares:
                means[key] = weighted_share_mean(
                    model.pixel_shares[key], model.area, model.inputs.cyclic)
        means["ratio_split"] = True
        means["force_satellite_et"] = False
        if "pool_capture_share" in model.pixel_shares:
            means["pool_capture_share"] = weighted_share_mean(
                model.pixel_shares["pool_capture_share"], model.area, model.inputs.cyclic)
        else:
            means["pool_capture_share"] = means.get("kx_ratio", 0.0) + means.get("ky_ratio", 0.0)
        model.physics = model.physics.replace(**{k: v for k, v in means.items() if k in allowed})
        self.params.update({k: means[k] for k in means if k in (
            "kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio", "farm_to_main", "pool_capture_share")})
        return means

    def district_means(self, shares=None, area_m2=None, cyclic=None):
        from .canal import weighted_share_mean

        shares = shares or self.shares
        if shares is None:
            raise RuntimeError("partition or absorb first")
        means = {}
        for key in RATIO_SHARE_KEYS:
            if key in shares:
                means[key] = (weighted_share_mean(shares[key], area_m2, cyclic)
                              if area_m2 is not None else float(np.mean(shares[key])))
        return means


@register_ratio
class FrozenStay(RatioPolicy):
    """Fit leftover-weighted stay once (`absorb`), then keep that per-paddy field.

    The next year's leftover is not allowed to retune the shares. That is the
    year-holdout: are these plot ratios a property of the field?
    """

    name = "frozen_stay"

    def partition(self, leftover, observation, targets):
        if self.shares is not None:
            return {key: np.array(val, copy=True) for key, val in self.shares.items()}
        shares = super().partition(leftover, observation, targets)
        self.shares = shares
        return shares


@register_ratio
class TablesRatio(RatioPolicy):
    """Table 4–5 geometry at every point. absorb(season) reweights stay to leftover mass."""

    name = "tables"

    def partition(self, leftover, observation, targets):
        shares = {name: np.array(targets.shares[name], copy=True)
                  for name in ("kx_ratio", "ky_ratio", "kz_ratio",
                               "ground_ratio", "farm_to_main", "pool_capture_share")}
        if leftover is not None:
            self.fit_stay_to_leftover(shares, leftover, observation, targets)
        return shares


RATIOS["demo_ratios"] = TablesRatio  # older CLI / finder name


class LateralPolicy:
    """Neighbour-head hydrology equation for leftover hops.

    Inputs are leftover heads and, for every paddy, the water and FAO-56 ET of
    its up, down, left and right neighbours (north, south, west, east), plus
    topology, area, and kinematic Physics (μ = vT/Δx, bund window). A missing
    edge neighbour is skipped. Soil properties are not an input: no Dupuit K,
    no plow-pan conductivity as a learned ratio. Applied irrigation share is
    not an input either. Neighbour ET may change if farmers irrigate
    differently; H is frozen and is fed the new heads and new neighbour ET.
    Knowing neighbour ET is not sending leftover toward high ET. Leftover
    collection shares (Kx, Ky, Kz, ground) are outputs of the same per-paddy
    equation (`partition`), not a separate ratio finder. Override
    flux(stock, graph, area, physics, neighbour_et_mm=, et_mm=).
    """

    name = "lateral"
    uses_soil_properties = False
    sees_neighbour_water = True
    sees_neighbour_et = True
    form = "physics"
    defaults: dict = {}
    share_keys = ("kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio")

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise TypeError(f"{type(self).__name__} has no parameters {sorted(unknown)}")
        self.params = {**self.defaults, **params}

    def flux(self, stock, graph, area, physics, neighbour_et_mm=None, et_mm=None,
             infiltration=None, use_paddy=False):
        from .graph import hop_stock

        return hop_stock(stock, graph, area, physics)

    def partition(self, stock, graph, et_mm=None, year=None, physics=None, infiltration=None,
                  cyclic=None):
        """Leftover simplex at every paddy: Kx, Ky, Kz, ground. Recapture/stay, not hops."""
        n = int(np.asarray(stock).size)
        return table_leftover_shares(n, year=year, physics=physics, cyclic=cyclic)


def table_leftover_shares(n, year=None, cyclic=None, physics=None, stay_frac=None):
    """District Table 4-5 leftover shares, optionally a per-paddy stay fraction."""
    from .identify import construct_ratios

    year = 2022 if year is None else int(year)
    ratios = construct_ratios(year)
    capture_pin = float(ratios["pool_capture_share"])
    nbr_pin = float(ratios["kx_ratio"] + ratios["ky_ratio"])
    stay_pin = float(ratios["kz_ratio"] + ratios["ground_ratio"])
    stay_mass = stay_pin if stay_pin > 1e-12 else 1.0
    if stay_frac is None:
        stay_frac = np.full(n, stay_pin)
    stay_frac = np.clip(np.asarray(stay_frac, dtype=float).ravel(), 0.0, 1.0)
    if stay_frac.size == 1:
        stay_frac = np.full(n, float(stay_frac))
    if cyclic is not None:
        stay_frac = np.where(np.asarray(cyclic, dtype=bool), stay_frac, 1.0)
    # stay_frac is Kz+ground; LFV stays pinned; remainder is neighbour hops.
    stay_frac = np.minimum(stay_frac, np.maximum(1.0 - capture_pin, 0.0))
    neighbour = np.clip(1.0 - capture_pin - stay_frac, 0.0, 1.0)
    if nbr_pin <= 1e-12 and capture_pin <= 1e-12:
        neighbour = np.zeros(n)
    kz = stay_frac * (float(ratios["kz_ratio"]) / stay_mass)
    ground = np.maximum(stay_frac - kz, 0.0)
    kx = 0.5 * neighbour
    ky = 0.5 * neighbour
    stack = np.clip(np.column_stack([kx, ky, kz, ground]), 0.0, None)
    keys = ("kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio")
    out = {key: stack[:, i] for i, key in enumerate(keys)}
    out["pool_capture_share"] = np.where(
        np.asarray(cyclic, dtype=bool) if cyclic is not None else True,
        capture_pin, 0.0)
    return out


@register_lateral
class HeadEqualise(LateralPolicy):
    """Baseline: leftover hops equalise head, wetter plot to the drier neighbour.

    This is the equation the package already runs. Register a different class
    with `@register_lateral` to propose another neighbour-head law. A win on
    historical year holdout is not claimed for this baseline.
    """

    name = "head_equalise"
    form = "physics"


class LearnedLateral(LateralPolicy):
    """Fit a frozen linear or nonlinear H from neighbour water and neighbour ET.

    This is the ML problem: propose H in a hypothesis class, fit on one historical
    year, freeze the weights, score the other year and a counterfactual farmer IR
    map. Plot-scale ΔL is unobserved in Okura; `fit` is against a teacher delta
    (the hop operator in the demo). A field ranking is holdout year leftover /
    satellite ET, not a new soil K.
    """

    fitted = False
    weights = None
    share_weights = None
    paddy_weights = None
    use_stock_features = False
    extra_names = ()

    def design(self, drops, neighbour_et, extra=None):
        raise NotImplementedError

    def _design_row(self, stock, graph, et_mm=None, infiltration=None):
        from .lateral import cardinal_drops, cardinal_features

        stock = np.asarray(stock, dtype=float).ravel()
        et_field = np.zeros(stock.size) if et_mm is None else np.asarray(et_mm, dtype=float).ravel()
        if et_field.size != stock.size:
            et_field = np.zeros(stock.size)
        water, present = cardinal_features(stock, graph)
        drops = cardinal_drops(stock, water, present)
        et_nbr, et_ok = cardinal_features(et_field, graph)
        et_nbr = np.where(et_ok, et_nbr, 0.0)
        extra, names = leftover_extra(stock, present)
        if not self.use_stock_features:
            extra, names = None, ()
        self.extra_names = names
        return self.design(drops, et_nbr, extra=extra)

    def fit(self, stocks, neighbour_ets, deltas, graph=None, shares=None,
            infiltrations=None, use_stock_features=None, per_paddy=False):
        if use_stock_features is not None:
            self.use_stock_features = bool(use_stock_features)
        rows, targets = [], []
        infs = infiltrations if infiltrations is not None else [None] * len(stocks)
        for stock, et_field, delta, inf in zip(stocks, neighbour_ets, deltas, infs):
            rows.append(self._design_row(stock, graph, et_mm=et_field, infiltration=inf))
            targets.append(np.asarray(delta, dtype=float).ravel())
        x = np.stack(rows, axis=0)
        y = np.stack(targets, axis=0)
        days, n, n_feat = x.shape
        self.weights, *_ = np.linalg.lstsq(x.reshape(days * n, n_feat), y.ravel(), rcond=None)
        self.paddy_weights = None
        if per_paddy:
            local = np.repeat(np.asarray(self.weights, dtype=float)[None, :], n, axis=0)
            for i in range(n):
                local[i] = ridge_fit(x[:, i], y[:, i], self.weights)
            self.paddy_weights = local
        self.fitted = True
        if shares is not None:
            self.fit_shares(stocks, neighbour_ets, shares, graph=graph, infiltrations=infs)
        return self.weights

    def fit_shares(self, stocks, neighbour_ets, shares, graph=None, infiltrations=None):
        """Kx, Ky, Kz, ground at every paddy from the same features as hops."""
        rows, targets = [], []
        infs = infiltrations if infiltrations is not None else [None] * len(stocks)
        for stock, et_field, share, inf in zip(stocks, neighbour_ets, shares, infs):
            stock = np.asarray(stock, dtype=float).ravel()
            rows.append(self._design_row(stock, graph, et_mm=et_field, infiltration=inf))
            share = np.asarray(share, dtype=float)
            if share.ndim == 1:
                share = np.repeat(share.reshape(1, -1), stock.size, axis=0)
            if share.shape[0] != stock.size:
                share = np.broadcast_to(share, (stock.size, share.shape[-1]))
            targets.append(np.clip(share.reshape(stock.size, -1), 0.0, None))
        x = np.vstack(rows)
        y = np.vstack(targets)
        self.share_weights, *_ = np.linalg.lstsq(x, y, rcond=None)
        self.fitted = True
        return self.share_weights

    def partition(self, stock, graph, et_mm=None, year=None, physics=None, infiltration=None,
                  cyclic=None):
        stock = np.asarray(stock, dtype=float).ravel()
        if self.share_weights is None:
            return table_leftover_shares(stock.size, year=year, physics=physics, cyclic=cyclic)
        raw = np.clip(self._design_row(stock, graph, et_mm=et_mm, infiltration=infiltration)
                      @ np.asarray(self.share_weights, dtype=float), 0.0, None)
        total = np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)
        shares = raw / total
        keys = ("kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio")
        return {key: shares[:, i] for i, key in enumerate(keys)}

    def flux(self, stock, graph, area, physics, neighbour_et_mm=None, et_mm=None,
             infiltration=None, use_paddy=False):
        stock = np.asarray(stock, dtype=float).ravel()
        et_field = et_mm if et_mm is not None else neighbour_et_mm
        design = self._design_row(stock, graph, et_mm=et_field, infiltration=infiltration)
        if use_paddy and self.paddy_weights is not None:
            delta = np.sum(design * np.asarray(self.paddy_weights, dtype=float), axis=1)
        else:
            delta = design @ np.asarray(self.weights, dtype=float)
        area = np.broadcast_to(np.asarray(area, dtype=float).ravel(), stock.shape)
        vol = delta * area
        vol = vol - float(vol.sum()) * area / max(float(area.sum()), 1e-9)
        delta = vol / np.maximum(area, 1e-9)
        after = np.maximum(stock + delta, 0.0)
        return after, after - stock


def ridge_fit(x, y, fallback, ridge=1e-2):
    """Per-paddy least squares. Falls back to the general weights if the plot is singular."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    fallback = np.asarray(fallback, dtype=float).ravel()
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] != fallback.size:
        return fallback
    xtx = x.T @ x
    try:
        weights = np.linalg.solve(xtx + float(ridge) * np.eye(x.shape[1]), x.T @ y)
    except np.linalg.LinAlgError:
        return fallback
    if not np.isfinite(weights).all():
        return fallback
    return weights


def leftover_extra(stock, present, infiltration=None):
    """Own leftover pile and which neighbours exist. Used by the hop / trickle equation."""
    leftover = np.maximum(np.asarray(stock, dtype=float).ravel(), 0.0)
    present = np.asarray(present, dtype=float)
    if present.ndim == 1:
        present = present.reshape(leftover.size, -1)
    cols = [leftover]
    names = ["leftover_mm"]
    if infiltration is not None:
        inf = np.clip(np.asarray(infiltration, dtype=float).ravel(), 0.0, None)
        if inf.size == leftover.size:
            cols.append(np.minimum(inf, leftover))
            names.append("infiltrated_mm")
    cols.append(present)
    names.extend(["has_right", "has_left", "has_up", "has_down"])
    return np.column_stack(cols), tuple(names)


def leftover_fate_design(stock, present, infiltration=None, cyclic=None, dist=None):
    """Features for recapture and soil-stay. Not neighbour head drops.

    Width is frozen (9): intercept, leftover, infiltrated, four has_*, cyclic,
    distance. Missing infiltration / cyclic / distance become zeros so a year
    without `infiltration_mm` can still score a year that has it.
    """
    leftover = np.maximum(np.asarray(stock, dtype=float).ravel(), 0.0)
    n = leftover.size
    inf = np.zeros(n)
    if infiltration is not None:
        raw = np.clip(np.asarray(infiltration, dtype=float).ravel(), 0.0, None)
        if raw.size == n:
            inf = np.minimum(raw, leftover)
    present = np.asarray(present, dtype=float)
    if present.ndim == 1:
        present = present.reshape(n, -1)
    if present.shape[0] != n:
        present = np.zeros((n, 4))
    elif present.shape[1] != 4:
        padded = np.zeros((n, 4))
        cols = min(4, present.shape[1])
        padded[:, :cols] = present[:, :cols]
        present = padded
    cyc = np.zeros(n)
    if cyclic is not None:
        raw = np.asarray(cyclic, dtype=float).ravel()
        if raw.size == 1:
            cyc = np.full(n, float(raw[0]))
        elif raw.size == n:
            cyc = raw
    distance = np.zeros(n)
    if dist is not None:
        raw = np.asarray(dist, dtype=float).ravel()
        if raw.size == 1:
            distance = np.full(n, float(raw[0]))
        elif raw.size == n:
            distance = raw
    names = ("intercept", "leftover_mm", "infiltrated_mm",
             "has_right", "has_left", "has_up", "has_down",
             "cyclic", "distance_m")
    return np.column_stack([np.ones(n), leftover, inf, present, cyc, distance]), names


class LeftoverFate:
    """Two leftover-mass equations: recapture to the pool, and leftover that stays in the soil.

    Neighbour trickle is a different equation (LinearHops / NonlinearHops). Recapture can
    only come from leftover that infiltrated. After the linear fit, pool + stay is scaled
    to the leftover pile so the two mass equations still partition leftover.
    """

    name = "leftover_fate"
    fitted = False
    pool_weights = None
    stay_weights = None
    pool_paddy_weights = None
    stay_paddy_weights = None
    feature_names = ()

    def _row(self, stock, graph, infiltration=None, cyclic=None, dist=None):
        from .lateral import cardinal_features

        stock = np.asarray(stock, dtype=float).ravel()
        _, present = cardinal_features(stock, graph)
        x, names = leftover_fate_design(stock, present, infiltration=infiltration,
                                        cyclic=cyclic, dist=dist)
        self.feature_names = names
        return x

    def fit(self, stocks, shares, graph, infiltrations=None, cyclic=None, dist=None,
            per_paddy=False):
        infs = infiltrations if infiltrations is not None else [None] * len(stocks)
        rows, pool, stay = [], [], []
        for stock, share, inf in zip(stocks, shares, infs):
            stock = np.maximum(np.asarray(stock, dtype=float).ravel(), 0.0)
            rows.append(self._row(stock, graph, inf, cyclic, dist))
            share = np.asarray(share, dtype=float)
            if share.ndim == 1:
                share = np.repeat(share.reshape(1, -1), stock.size, axis=0)
            share = np.clip(share.reshape(stock.size, -1), 0.0, None)
            pool.append(stock * (share[:, 0] + share[:, 1]))
            stay.append(stock * (share[:, 2] + share[:, 3]))
        x = np.stack(rows, axis=0)
        pool_y = np.stack(pool, axis=0)
        stay_y = np.stack(stay, axis=0)
        days, n, n_feat = x.shape
        flat = x.reshape(days * n, n_feat)
        self.pool_weights, *_ = np.linalg.lstsq(flat, pool_y.ravel(), rcond=None)
        self.stay_weights, *_ = np.linalg.lstsq(flat, stay_y.ravel(), rcond=None)
        self.pool_paddy_weights = None
        self.stay_paddy_weights = None
        if per_paddy:
            pool_local = np.repeat(np.asarray(self.pool_weights, dtype=float)[None, :], n, axis=0)
            stay_local = np.repeat(np.asarray(self.stay_weights, dtype=float)[None, :], n, axis=0)
            for i in range(n):
                pool_local[i] = ridge_fit(x[:, i], pool_y[:, i], self.pool_weights)
                stay_local[i] = ridge_fit(x[:, i], stay_y[:, i], self.stay_weights)
            self.pool_paddy_weights = pool_local
            self.stay_paddy_weights = stay_local
        self.fitted = True
        return self

    def predict_mm(self, stock, graph, infiltration=None, cyclic=None, dist=None,
                   use_paddy=False):
        stock = np.maximum(np.asarray(stock, dtype=float).ravel(), 0.0)
        if self.pool_weights is None:
            part = table_leftover_shares(stock.size, cyclic=cyclic)
            capture = part.get("pool_capture_share", part["kx_ratio"] + part["ky_ratio"])
            return stock * capture, stock * (1.0 - capture)
        x = self._row(stock, graph, infiltration, cyclic, dist)
        if use_paddy and self.pool_paddy_weights is not None:
            pool = np.clip(np.sum(x * np.asarray(self.pool_paddy_weights, dtype=float), axis=1), 0.0, None)
            stay = np.clip(np.sum(x * np.asarray(self.stay_paddy_weights, dtype=float), axis=1), 0.0, None)
        else:
            pool = np.clip(x @ np.asarray(self.pool_weights, dtype=float), 0.0, None)
            stay = np.clip(x @ np.asarray(self.stay_weights, dtype=float), 0.0, None)
        total = pool + stay
        scale = np.divide(stock, total, out=np.zeros_like(stock), where=total > 1e-12)
        return pool * scale, stay * scale

    def partition(self, stock, graph, et_mm=None, year=None, physics=None, infiltration=None,
                  cyclic=None, dist=None):
        stock = np.asarray(stock, dtype=float).ravel()
        pool, stay = self.predict_mm(stock, graph, infiltration=infiltration,
                                     cyclic=cyclic, dist=dist)
        leftover = np.maximum(stock, 0.0)
        capture = np.divide(pool, leftover, out=np.zeros_like(leftover), where=leftover > 1e-12)
        stay_frac = np.clip(1.0 - capture, 0.0, 1.0)
        return table_leftover_shares(stock.size, year=year, physics=physics, cyclic=cyclic,
                                     stay_frac=stay_frac)


@register_lateral
class LinearHops(LearnedLateral):
    """Linear equation at every paddy: ΔL_i = w0 + Σ_d [w_h,d (h_i−h_d) + w_e,d ET_d]
    for d in right, left, up, down (east, west, north, south)."""

    name = "linear"
    form = "linear"

    def __init__(self, **params):
        super().__init__(**params)
        self.weights = np.array([0.0, -0.25, -0.25, -0.25, -0.25, 0.0, 0.0, 0.0, 0.0])

    def design(self, drops, neighbour_et, extra=None):
        drops = np.asarray(drops, dtype=float)
        neighbour_et = np.asarray(neighbour_et, dtype=float)
        cols = [np.ones(drops.shape[0]), drops, neighbour_et]
        if extra is not None:
            cols.append(np.asarray(extra, dtype=float))
        return np.column_stack(cols)


@register_lateral
class NonlinearHops(LearnedLateral):
    """Nonlinear-in-features: linear UDLR terms plus drop², tanh(drop), drop×ET per side."""

    name = "nonlinear"
    form = "nonlinear"

    def __init__(self, **params):
        super().__init__(**params)
        self.weights = np.zeros(21)
        self.weights[1:5] = -0.25

    def design(self, drops, neighbour_et, extra=None):
        drops = np.asarray(drops, dtype=float)
        neighbour_et = np.asarray(neighbour_et, dtype=float)
        cols = [
            np.ones(drops.shape[0]),
            drops,
            neighbour_et,
            drops ** 2,
            np.tanh(drops / 20.0),
            drops * neighbour_et,
        ]
        if extra is not None:
            cols.append(np.asarray(extra, dtype=float))
        return np.column_stack(cols)


@register_agent
class ETFollower(FarmerPolicy):
    """The base farmer: replace what the plot will lose, keep it at a target depth.

        irrigation = clip(target + et_gain x Kc(last pass) x ET0(yesterday)
                          + seepage_allowance - depth, 0, max_irrigation)
        discharge  = max(depth - ceiling, 0)          # let heavy rain off

    and the two dry spells the paper describes (Section 2.5): mid-summer drainage, when
    irrigation stops and the plot is emptied, and the end of irrigation before harvest.
    Rain arrives through `depth`: a wet yesterday leaves more standing, so less is sent.
    """

    name = "et_follower"
    defaults = dict(target_depth_mm=30.0, ceiling_mm=60.0, et_gain=1.0, seepage_allowance_mm=11.0,
                    max_irrigation_mm=80.0, midsummer_start_day=54.0, midsummer_days=14.0,
                    final_drain_day=120.0, farmer_variation=0.15, visit_interval_days=3.0)
    bounds = dict(target_depth_mm=(5.0, 80.0), ceiling_mm=(20.0, 120.0), et_gain=(0.5, 1.5),
                  seepage_allowance_mm=(0.0, 25.0), midsummer_start_day=(40.0, 75.0),
                  midsummer_days=(0.0, 28.0), final_drain_day=(100.0, 131.0),
                  farmer_variation=(0.0, 0.5), visit_interval_days=(1.0, 7.0))

    def reset(self, model, rows):
        # Scenario assumptions, not measured farmer characteristics. Draw once per
        # season, not daily ET noise. Model seed makes comparisons reproducible.
        self._traits = {str(model.inputs.paddy_id[i]): tuple(model.rng.uniform(-1, 1, 3))
                        for i in rows}

    def dry_day(self, day):
        p = self.params
        start = p["midsummer_start_day"]
        return start <= day < start + p["midsummer_days"] or day >= p["final_drain_day"]

    def decide_batch(self, batch):
        p = self.params
        depth = batch.depth_mm
        traits = getattr(self, "_traits", {})
        offsets = np.array([traits.get(str(k), (0.0, 0.0, 0.0)) for k in batch.paddy_id])
        variation = p.get("farmer_variation", 0.0)
        target = p["target_depth_mm"] * (1 + variation * offsets[:, 0])
        shift = np.rint(20 * variation * offsets[:, 1]).astype(int)
        local_day = batch.day - shift
        start = p["midsummer_start_day"]
        dry = ((local_day >= start) & (local_day < start + p["midsummer_days"])) | (local_day >= p["final_drain_day"])
        interval = max(1, int(round(p.get("visit_interval_days", 1))))
        phase = np.minimum(((offsets[:, 2] + 1) * 0.5 * interval).astype(int), interval - 1)
        visit = (batch.day % interval) == phase
        wanted = target + p["et_gain"] * self.expected_et_mm(batch) + p["seepage_allowance_mm"]
        irrigation = np.clip(wanted - depth, 0.0, p["max_irrigation_mm"])
        discharge = np.maximum(depth - np.maximum(p["ceiling_mm"], target), 0.0)
        return np.where(visit & ~dry, irrigation, 0.0), np.where(dry, depth, discharge)


@register_agent
class SatellitePhysics(FarmerPolicy):
    """Not a farmer. Irrigation and outlet discharge follow the satellite flood fraction:
    flooded points are topped up to `pond_mm`; dry points drain the pond so ET is taken
    from the soil stores. Between fly-overs the flood fraction is interpolated."""

    name = "satellite"
    defaults = dict(pond_mm=30.0)

    def reset(self, model, rows):
        self.rows = np.asarray(rows)
        self.flood = model.inputs.flood_daily

    def decide_batch(self, batch):
        flood = np.clip(self.flood[batch.day, self.rows], 0.0, 1.0)
        target = float(self.params["pond_mm"]) * flood
        depth = batch.depth_mm
        irrigation = np.maximum(target - depth, 0.0)
        discharge = np.maximum(depth - target, 0.0)
        return irrigation, discharge


@register_agent
class FieldAggregate(FarmerPolicy):
    """Hard-copy surface IR/D/W; Table 5 underground leftover ratios.

    **Known surface:** Fig. 8 IR/W and observed D (no supply clip / satellite zeroing).

    **Underground:** district capture = Table 5 LFV/LfA (no artificial floor).
    Per-point LFV is that capture share (near-canal higher); neighbour hops move
    leftover so ET-short plots are refilled; stay (Kz/ground) is deep/dWB.
    ``solve_penetration`` sets plow-pan k so infiltrated leftover can supply both.
    """

    name = "field_aggregate"
    defaults = dict(spill_fudge=0.02, pool_capture_share=None,
                    plow_pan_k_mm_d=13.0, irrigate_ncb=1.0,
                    schedule="fig8",
                    hard_surface=1.0,
                    pin_fig8_w=1.0,
                    satellite_empty_dry=0.0,
                    flood_empty_below=0.25,
                    flood_min_wet_frac=0.02,
                    flood_min_p90=0.15,
                    pin_table5_lfv=1.0,
                    pin_table5_o=1.0,
                    pin_fig6_lf=1.0,
                    paper_plot_ratios=1.0,
                    solve_penetration=1.0,
                    w_ema=0.0,
                    w_cap_frac=3.5,
                    ir_supply_cap=0.0)

    def reset(self, model, rows):
        from . import imitation
        from .identify import match_block_volumes

        self.rows = np.asarray(rows)
        self._model = model
        self._area = np.asarray(model.area, dtype=float)
        self._block = np.asarray(model.inputs.block)
        self._cyclic = np.asarray(model.inputs.cyclic, dtype=bool)
        self._outlet = float(getattr(model.physics, "outlet_loss_per_day", 0.13))
        year = int(getattr(model, "year", 2022))
        self._spill_fudge = float(self.params.get("spill_fudge", 0.02))
        self._cum_d_obs = {"North": 0.0, "South": 0.0}
        self._cum_d_est = {"North": 0.0, "South": 0.0}
        self._cum_d_obs_ncb = {"North": 0.0, "South": 0.0}
        self._cum_d_est_ncb = {"North": 0.0, "South": 0.0}
        # Linear pond→soil I=K h/h_ref is already in physics; pin K and capture so
        # LFV_pump / drainage can reach Tables 4–5 on Okura match runs.
        changes = {}
        if self.params.get("pool_capture_share") is not None:
            changes["pool_capture_share"] = float(self.params["pool_capture_share"])
        if self.params.get("plow_pan_k_mm_d") is not None:
            changes["plow_pan_k_mm_d"] = float(self.params["plow_pan_k_mm_d"])
        if changes:
            model.physics = model.physics.replace(**changes)
        rain = np.asarray(model.inputs.rain_mm, dtype=float)
        self._rain_mm = rain[:, 0] if rain.ndim == 2 and rain.shape[1] == 1 else (
            rain if rain.ndim == 1 else rain.mean(axis=1))
        from . import okura as _okura
        table_rain = float(_okura.TABLE4_MM[("North", year)]["rain"])
        rain_season = float(np.sum(np.asarray(self._rain_mm[:model.n_days], dtype=float)))
        wetter = rain_season > table_rain * 1.02
        if "d_catchup" in self.params:
            self._d_catchup = float(self.params["d_catchup"])
        else:
            # Wetter Open-Meteo than Table 4: pull harder so spill+D tracks ADCP.
            self._d_catchup = 1.0 if wetter else (0.55 if year == 2021 else 0.35)
        if "d_catchup_ncb" in self.params:
            self._d_catchup_ncb = float(self.params["d_catchup_ncb"])
        else:
            self._d_catchup_ncb = 1.0 if wetter else 0.85
        blocks, _, _ = imitation.load_observed(model.year)
        dates = pd.to_datetime(model.inputs.dates)
        if getattr(dates, "tz", None) is not None:
            dates = dates.tz_convert("UTC").tz_localize(None)
        dates = dates.normalize()
        obs_dates = pd.to_datetime(blocks["date"])
        if obs_dates.dt.tz is not None:
            obs_dates = obs_dates.dt.tz_convert("UTC").dt.tz_localize(None)
        obs_dates = obs_dates.dt.normalize()
        self.irrigation = np.zeros((model.n_days, model.n))
        self.discharge = np.zeros((model.n_days, model.n))
        area = self._area
        block = self._block
        cyclic = self._cyclic
        et = np.asarray(getattr(model.inputs, "et_target_mm", np.zeros((model.n_days, model.n))), dtype=float)
        if et.ndim == 1:
            et = np.broadcast_to(et[:, None], (model.n_days, model.n))
        dist = np.asarray(getattr(model.inputs, "canal_distance_m", np.zeros(model.n)), dtype=float)
        near = np.exp(-np.maximum(dist, 0.0) / max(float(model.physics.canal_length_m), 1.0))
        drain_score = 0.25 + 0.75 * near
        schedule = str(self.params.get("schedule", "fig8")).lower().strip()
        if schedule not in ("fig8", "observed", "paper", "csv"):
            schedule = "fig8"
        hard = float(self.params.get("hard_surface", 1.0)) >= 0.5
        model.hard_surface = hard
        # Hard-copy cyclic IR from Fig. 8 CSV; D from observed blocks (Fig. 6 / ADCP).
        fig8_ir = None
        if hard and schedule in ("fig8", "paper", "csv"):
            try:
                from .paper_plots import fig8_ir_plan_m3
                fig8_ir = fig8_ir_plan_m3(year, dates)
            except Exception:
                fig8_ir = None
        for d, date in enumerate(dates):
            ir, dr = {}, {}
            for name in ("North", "South"):
                part = blocks[(obs_dates == pd.Timestamp(date)) & (blocks["block"] == name)]
                if fig8_ir is not None:
                    ir[name] = float(fig8_ir[name][d])
                else:
                    ir[name] = float(part["irrigation_m3"].sum()) if len(part) else 0.0
                dr[name] = float(part["drainage_m3"].sum()) if len(part) else 0.0
            # Within-block split is unobserved; area-only when hard_surface (no ET guess).
            if hard:
                self.irrigation[d] = match_block_volumes(ir, area, block, cyclic, weights=area)
            else:
                ir_score = np.maximum(et[min(d, et.shape[0] - 1)], 0.05) * (0.4 + 0.6 / (0.35 + near))
                self.irrigation[d] = match_block_volumes(ir, area, block, cyclic, weights=area * ir_score)
            self.discharge[d] = match_block_volumes(dr, area, block, cyclic, weights=area * drain_score)
        from . import okura as _okura
        year = int(model.year)
        for name in ("North", "South"):
            sel = (block == name) & cyclic
            if not np.any(sel):
                continue
            have_d = float((self.discharge[:, sel] * area[sel]).sum() / 1000.0)
            want_d = float(_okura.TABLE4_MM[(name, year)]["drainage"] * _okura.cyclic_area_m2(name) / 1000.0)
            # Keep Fig. 8 IR days exact; pin D season to Table 4.
            if have_d > 1e-9:
                self.discharge[:, sel] *= want_d / have_d
        if float(self.params.get("irrigate_ncb", 1.0)):
            rain_season_mm = float(np.sum(np.asarray(self._rain_mm[:model.n_days], dtype=float)))
            self._fill_ncb_irrigation(year, et, near, rain_mm=rain_season_mm)
        # Fig. 8 daily lake W pin (digitised CSV hard copy).
        model.w_plan_m3 = None
        if float(self.params.get("pin_fig8_w", 1.0)):
            try:
                from .paper_plots import fig8_w_plan_m3
                model.w_plan_m3 = fig8_w_plan_m3(year, dates)
            except Exception:
                model.w_plan_m3 = None
        # Table 5 LFV_pump: day shape from paper Fig. 6 lateral (Lf) plot bars.
        # Pin O from Table 5 even with Fig. 8 W — settle will not draw extra lake
        # beyond W_force to meet O (pool/spill only).
        from . import season_targets as _season
        if float(self.params.get("pin_table5_lfv", 1.0)):
            if float(self.params.get("pin_fig6_lf", 1.0)):
                model.lfv_plan_m3 = _season.table5_lfv_plan_m3(
                    year, model.n_days, day_weights=None, use_fig6_lf=True)
            else:
                cyclic = self._cyclic
                area = self._area
                d_day = (self.discharge[:, cyclic] * area[cyclic]).sum(axis=1) if np.any(cyclic) else np.ones(model.n_days)
                model.lfv_plan_m3 = _season.table5_lfv_plan_m3(
                    year, model.n_days, day_weights=d_day, use_fig6_lf=False)
        else:
            model.lfv_plan_m3 = None
        if float(self.params.get("pin_table5_o", 1.0)):
            model.o_plan_m3 = _season.table5_o_plan_m3(
                year, model.n_days, rain_mm=self._rain_mm[:model.n_days])
        else:
            model.o_plan_m3 = None
        # Paper plot-block leftover shares: Table 5 capture pin + Fig. 6 intensity.
        # Per-point LFV = capture share; neighbour hops cover ET shortfalls.
        if float(self.params.get("paper_plot_ratios", 1.0)):
            try:
                from .identify import attach_pixel_shares, construct_ratios
                from .physics import Physics
                ratios = construct_ratios(year)
                if self.params.get("pool_capture_share") is not None:
                    # Optional override only when explicitly set (not a floor).
                    c = float(self.params["pool_capture_share"])
                    stay = 1.0 - c
                    top = float(model.physics.total_evaporable_mm /
                                max(model.physics.total_evaporable_mm
                                    + model.physics.total_available_mm, 1e-9))
                    ratios = dict(ratios)
                    ratios.update({
                        "pool_capture_share": c,
                        "kx_ratio": 0.5 * c,
                        "ky_ratio": 0.5 * c,
                        "kz_ratio": stay * top,
                        "ground_ratio": stay * (1.0 - top),
                    })
                allowed = {item.name for item in fields(Physics)}
                model.physics = model.physics.replace(
                    **{k: v for k, v in ratios.items() if k in allowed})
                attach_pixel_shares(model, ratios, use_paper_plots=True)
            except Exception:
                pass
        # Given IR/D: solve soil penetration so ET + LFV can both be covered, then
        # enable underground hops (lateral) + LFV to pool + deep stay.
        if float(self.params.get("solve_penetration", 1.0)):
            try:
                from .water_fate import configure_given_ir_d_water_fate
                report = configure_given_ir_d_water_fate(
                    model, self.irrigation, self.discharge, year=year)
                self.params["plow_pan_k_mm_d"] = float(report.plow_pan_k_mm_d)
                self.params["penetration_et_cover"] = float(report.et_cover)
                self.params["penetration_lfv_room"] = float(report.lfv_room)
            except Exception:
                try:
                    from .water_fate import apply_water_fate_physics
                    apply_water_fate_physics(model)
                except Exception:
                    pass
        # Hard surface: deliver known IR (no supply clip); W is Fig. 8 exact (no O borrow).
        st = getattr(model, "station", None)
        if st is not None:
            st.w_ema = float(np.clip(self.params.get("w_ema", 0.0), 0.0, 1.0))
            st._w_filt = 0.0
            st.w_hard_copy = hard and float(self.params.get("pin_fig8_w", 1.0)) >= 0.5
            if hard or not float(self.params.get("ir_supply_cap", 0.0)):
                st.w_daily_cap_m3 = None
            elif float(self.params.get("ir_supply_cap", 0.0)):
                frac = float(self.params.get("w_cap_frac", 3.5))
                want_w = float(_okura.TABLE5_1000M3[year]["W"]) * 1000.0
                want_o = float(_okura.TABLE5_1000M3[year]["O"]) * 1000.0
                if model.w_plan_m3 is not None and len(model.w_plan_m3):
                    peak_w = float(np.max(model.w_plan_m3))
                    st.w_daily_cap_m3 = max(
                        ((want_w + want_o) / max(model.n_days, 1)) * frac,
                        peak_w * 1.05) if frac > 0 else None
                else:
                    st.w_daily_cap_m3 = ((want_w + want_o) / max(model.n_days, 1)) * frac if frac > 0 else None
            if float(self.params.get("pin_table5_o", 1.0)):
                want_o = float(_okura.TABLE5_1000M3[year]["O"]) * 1000.0
                want_w = float(_okura.TABLE5_1000M3[year]["W"]) * 1000.0
                mean_o = want_o / max(model.n_days, 1)
                mean_w = want_w / max(model.n_days, 1)
                st.capacity = max(
                    float(st.capacity),
                    mean_o * 15.0,
                    mean_w * 0.35,
                    float(model.physics.pool_capacity_m3))

    def _fill_ncb_irrigation(self, year, et, near, rain_mm=None):
        """Spread table IR_ncb and D_ncb (Open-Meteo rain in the D balance) onto NCB."""
        from . import okura as _okura
        from .identify import match_block_volumes

        area = self._area
        block = self._block
        cyclic = self._cyclic
        ncb = ~cyclic
        if not np.any(ncb):
            return
        day_w_ir, day_w_d = {}, {}
        for name in ("North", "South"):
            sel = (block == name) & cyclic
            if np.any(sel):
                w_ir = (self.irrigation[:, sel] * area[sel]).sum(axis=1)
                w_d = (self.discharge[:, sel] * area[sel]).sum(axis=1)
            else:
                w_ir = np.ones(self.irrigation.shape[0])
                w_d = np.ones(self.discharge.shape[0])
            for store, w in ((day_w_ir, w_ir), (day_w_d, w_d)):
                store[name] = np.maximum(w, 0.0)
                if float(store[name].sum()) <= 1e-9:
                    store[name] = np.ones_like(store[name])
                store[name] = store[name] / store[name].sum()
        for d in range(self.irrigation.shape[0]):
            want_ir, want_d = {}, {}
            for name in ("North", "South"):
                want_ir[name] = float(_okura.ir_ncb_m3(name, year)) * float(day_w_ir[name][d])
                want_d[name] = float(_okura.d_ncb_m3(name, year, rain_mm=rain_mm)) * float(day_w_d[name][d])
            score = np.maximum(et[min(d, et.shape[0] - 1)], 0.05) * (0.4 + 0.6 / (0.35 + near))
            drain_score = 0.25 + 0.75 * near
            ir_mm = match_block_volumes(want_ir, area, block, cyclic=ncb, weights=area * score)
            dr_mm = match_block_volumes(want_d, area, block, cyclic=ncb, weights=area * drain_score)
            self.irrigation[d, ncb] = ir_mm[ncb]
            self.discharge[d, ncb] = dr_mm[ncb]

    def _merge_ncb_irrigation(self, irrigation_rows, day, rows):
        """Keep Table 5 NCB IR when a subclass re-pins only cyclic volumes."""
        rows = np.asarray(rows)
        out = np.asarray(irrigation_rows, dtype=float).copy()
        ncb_local = ~self._cyclic[rows]
        if np.any(ncb_local):
            out[ncb_local] = self.irrigation[int(day), rows][ncb_local]
        return out

    def _merge_ncb_discharge(self, discharge_rows, day, rows):
        """Keep NCB drainage targets when a subclass re-pins only cyclic D."""
        rows = np.asarray(rows)
        out = np.asarray(discharge_rows, dtype=float).copy()
        ncb_local = ~self._cyclic[rows]
        if np.any(ncb_local):
            out[ncb_local] = self.discharge[int(day), rows][ncb_local]
        return out

    def _spill_adjusted_discharge(self, irrigation_mm, depth_mm, et_mm, want_discharge_mm, rows):
        """Farmer D so farmer D + outlet spill ≈ cyclic ADCP and NCB D targets.

        Uses a linear pond→soil infiltration world (I = K_pan h / h_ref) already in
        physics; here we only invert the outlet leak so scored drainage matches.
        A running North/South catch-up term removes leftover season error on cyclic.
        """
        from .identify import match_block_volumes

        outlet_true = float(np.clip(getattr(self, "_outlet", 0.13), 0.0, 0.5))
        outlet = float(np.clip(outlet_true + float(getattr(self, "_spill_fudge", 0.02)), 0.0, 0.4))
        retain = max(1.0 - outlet, 0.5)
        catchup = float(getattr(self, "_d_catchup", 0.0))
        area = self._area
        block = self._block
        cyclic = self._cyclic
        ncb = ~cyclic
        n = area.size
        ir = np.zeros(n)
        depth = np.zeros(n)
        et = np.zeros(n)
        ir[rows] = np.asarray(irrigation_mm, dtype=float)
        depth[rows] = np.asarray(depth_mm, dtype=float)
        et[rows] = np.maximum(np.asarray(et_mm, dtype=float), 0.0)
        rain = np.zeros(n)
        if hasattr(self, "_rain_mm"):
            day = int(getattr(self, "_day", 0))
            rain_src = self._rain_mm
            rain[:] = float(rain_src[min(day, len(rain_src) - 1)])
        water = np.maximum(depth + ir + rain, 0.0)
        water_after_et = np.maximum(water - et, 0.0)
        want_c, want_n = {}, {}
        for name in ("North", "South"):
            sel_c = (block == name) & cyclic
            sel_n = (block == name) & ncb
            if np.any(sel_c):
                v_obs = float((want_discharge_mm[sel_c] * area[sel_c]).sum() / 1000.0)
                lag = catchup * (self._cum_d_obs.get(name, 0.0) - self._cum_d_est.get(name, 0.0))
                v_target = max(v_obs + lag, 0.0)
                w_plan = float((water[sel_c] * area[sel_c]).sum() / 1000.0)
                w_et = float((water_after_et[sel_c] * area[sel_c]).sum() / 1000.0)
                v_farmer = float(np.clip((v_target - outlet * w_plan) / retain, 0.0, w_et))
                want_c[name] = v_farmer
                if not getattr(getattr(self, "_model", None), "log", None):
                    est_total = v_farmer + outlet_true * max(w_et - v_farmer, 0.0)
                    self._cum_d_obs[name] = self._cum_d_obs.get(name, 0.0) + v_obs
                    self._cum_d_est[name] = self._cum_d_est.get(name, 0.0) + est_total
            else:
                want_c[name] = 0.0
            if np.any(sel_n):
                v_obs = float((want_discharge_mm[sel_n] * area[sel_n]).sum() / 1000.0)
                lag = float(getattr(self, "_d_catchup_ncb", 0.85)) * (
                    self._cum_d_obs_ncb.get(name, 0.0) - self._cum_d_est_ncb.get(name, 0.0))
                v_target = max(v_obs + lag, 0.0)
                w_plan = float((water[sel_n] * area[sel_n]).sum() / 1000.0)
                w_et = float((water_after_et[sel_n] * area[sel_n]).sum() / 1000.0)
                v_farmer = float(np.clip((v_target - outlet * w_plan) / retain, 0.0, w_et))
                want_n[name] = v_farmer
                if not getattr(getattr(self, "_model", None), "log", None):
                    est = v_farmer + outlet_true * max(w_et - v_farmer, 0.0)
                    self._cum_d_obs_ncb[name] = self._cum_d_obs_ncb.get(name, 0.0) + v_obs
                    self._cum_d_est_ncb[name] = self._cum_d_est_ncb.get(name, 0.0) + est
            else:
                want_n[name] = 0.0
        weights = area * (0.05 + water_after_et)
        out = match_block_volumes(want_c, area, block, cyclic, weights=weights)
        if float(self.params.get("irrigate_ncb", 1.0)) and np.any(ncb):
            out_n = match_block_volumes(want_n, area, block, cyclic=ncb, weights=weights)
            out = np.where(ncb, out_n, out)
        return out[rows]

    def _sync_drainage_catchup(self, day):
        """Book cyclic and NCB catch-up from scored drainage vs planned targets."""
        model = getattr(self, "_model", None)
        if model is None or day <= 0 or not getattr(model, "log", None):
            return
        if "drainage_mm" not in model.log:
            return
        drain = np.asarray(model.log["drainage_mm"], dtype=float)
        for name in ("North", "South"):
            sel_c = (self._block == name) & self._cyclic
            sel_n = (self._block == name) & (~self._cyclic)
            if np.any(sel_c):
                self._cum_d_est[name] = float(
                    (drain[:day, sel_c] * self._area[sel_c]).sum() / 1000.0)
                self._cum_d_obs[name] = float(
                    (self.discharge[:day, sel_c] * self._area[sel_c]).sum() / 1000.0)
            if np.any(sel_n):
                self._cum_d_est_ncb[name] = float(
                    (drain[:day, sel_n] * self._area[sel_n]).sum() / 1000.0)
                self._cum_d_obs_ncb[name] = float(
                    (self.discharge[:day, sel_n] * self._area[sel_n]).sum() / 1000.0)

    def _flood_fraction(self, day, rows=None):
        """Satellite flood/ponding 0–1 for each paddy (interpolated between passes)."""
        n = self._area.size
        flood = np.ones(n, dtype=float)
        flood_in = getattr(getattr(self, "_model", None), "inputs", None)
        flood_in = getattr(flood_in, "flood_daily", None) if flood_in is not None else None
        if flood_in is not None:
            arr = np.asarray(flood_in, dtype=float)
            if arr.ndim == 2:
                arr = arr[min(int(day), arr.shape[0] - 1)]
            flood = np.clip(arr, 0.0, 1.0)
        if rows is None:
            return flood
        return flood[np.asarray(rows)]

    def _flood_signal_usable(self, flood_full):
        """True only when the satellite actually shows some ponded area today.

        The Imbanuma flood stack is often near-zero; treating that as 'all dry'
        empties the district and destroys season IR. Skip the hard empty rule then.
        """
        flood_full = np.asarray(flood_full, dtype=float)
        wet_frac = float((flood_full >= float(self.params.get("flood_empty_below", 0.25))).mean())
        p90 = float(np.percentile(flood_full, 90))
        min_wet = float(self.params.get("flood_min_wet_frac", 0.02))
        min_p90 = float(self.params.get("flood_min_p90", 0.15))
        return wet_frac >= min_wet or p90 >= min_p90

    def _relative_flood_weights(self, flood_full):
        """Soft 0–1 weights: prefer wetter paddies even when absolute flood is weak."""
        f = np.asarray(flood_full, dtype=float)
        lo, hi = np.percentile(f, [10, 90])
        if hi - lo < 1e-6:
            return np.ones_like(f)
        return np.clip((f - lo) / (hi - lo), 0.0, 1.0)

    def _apply_satellite_empty_when_dry(self, irrigation_mm, discharge_mm, depth_mm, day, rows):
        """If the satellite shows flood/ponding somewhere, dry spots stay empty.

        When the flood map is unusable (almost all zeros), only apply a soft relative
        preference — do not zero the whole district's irrigation.
        """
        from .identify import match_block_volumes

        rows = np.asarray(rows)
        irrigation = np.asarray(irrigation_mm, dtype=float).copy()
        discharge = np.asarray(discharge_mm, dtype=float).copy()
        depth = np.asarray(depth_mm, dtype=float)
        flood_full = self._flood_fraction(day)
        flood = flood_full[rows]
        empty_below = float(self.params.get("flood_empty_below", 0.25))

        if not float(self.params.get("satellite_empty_dry", 1.0)):
            return irrigation, discharge

        if not self._flood_signal_usable(flood_full):
            # Soft: slightly prefer relatively wetter paddies; keep block totals via rematch.
            rel = self._relative_flood_weights(flood_full)
            area = self._area
            block = self._block
            cyclic = self._cyclic
            ncb = ~cyclic
            n = area.size
            want_c, want_n = {}, {}
            for name in ("North", "South"):
                sel_c = (block == name) & cyclic
                sel_n = (block == name) & ncb
                want_c[name] = float((self.irrigation[int(day), sel_c] * area[sel_c]).sum() / 1000.0) if np.any(sel_c) else 0.0
                want_n[name] = float((self.irrigation[int(day), sel_n] * area[sel_n]).sum() / 1000.0) if np.any(sel_n) else 0.0
            full_ir = np.zeros(n)
            full_ir[rows] = irrigation
            w = np.maximum(full_ir, 0.05) * (0.35 + 0.65 * rel)
            ir_c = match_block_volumes(want_c, area, block, cyclic, weights=area * w)
            if float(self.params.get("irrigate_ncb", 1.0)) and np.any(ncb):
                ir_n = match_block_volumes(want_n, area, block, cyclic=ncb, weights=area * w)
                full_ir = np.where(ncb, ir_n, ir_c)
            else:
                full_ir = ir_c
            return full_ir[rows], discharge

        wet = flood >= empty_below
        dry = ~wet
        if not np.any(wet):
            return np.zeros_like(irrigation), np.maximum(depth, 0.0)

        irrigation[dry] = 0.0
        discharge[dry] = np.maximum(depth[dry], 0.0)

        area = self._area
        block = self._block
        cyclic = self._cyclic
        ncb = ~cyclic
        n = area.size
        full_ir = np.zeros(n)
        full_ir[rows] = irrigation
        wet_w = np.where(flood_full >= empty_below, np.maximum(flood_full, 0.05), 0.0)

        want_c, want_n = {}, {}
        for name in ("North", "South"):
            sel_c = (block == name) & cyclic
            sel_n = (block == name) & ncb
            want_c[name] = float((self.irrigation[int(day), sel_c] * area[sel_c]).sum() / 1000.0) if np.any(sel_c) else 0.0
            want_n[name] = float((self.irrigation[int(day), sel_n] * area[sel_n]).sum() / 1000.0) if np.any(sel_n) else 0.0
        base_w = np.maximum(full_ir, 0.0) * wet_w
        if float(base_w.sum()) < 1e-9:
            base_w = wet_w
        ir_c = match_block_volumes(want_c, area, block, cyclic, weights=area * np.maximum(base_w, 1e-12))
        if float(self.params.get("irrigate_ncb", 1.0)) and np.any(ncb):
            ir_n = match_block_volumes(want_n, area, block, cyclic=ncb, weights=area * np.maximum(base_w, 1e-12))
            full_ir = np.where(ncb, ir_n, ir_c)
        else:
            full_ir = ir_c
        full_ir = np.where(flood_full >= empty_below, full_ir, 0.0)
        irrigation = full_ir[rows]
        irrigation[dry] = 0.0
        discharge[dry] = np.maximum(depth[dry], 0.0)
        return irrigation, discharge

    def decide_batch(self, batch):
        rows = self.rows
        day = int(batch.day)
        self._day = day
        self._sync_drainage_catchup(day)
        irrigation = self.irrigation[day, rows]
        demand = self.expected_et_mm(batch)
        discharge = self._spill_adjusted_discharge(
            irrigation, batch.depth_mm, demand, self.discharge[day], rows)
        # Hard-copy surface: never zero Fig. 8 IR from a satellite dry guess.
        if float(self.params.get("satellite_empty_dry", 0.0)) and float(
                self.params.get("hard_surface", 1.0)) < 0.5:
            irrigation, discharge = self._apply_satellite_empty_when_dry(
                irrigation, discharge, batch.depth_mm, day, rows)
        return irrigation, discharge


@register_agent
class ETMatch(FieldAggregate):
    """Okura block IR/D totals, redistributed each morning toward satellite ET.

    Field totals match the pump / ADCP (same as field_aggregate, with outlet-spill
    compensation on D). Within a block, irrigation goes to plots that would starve
    FAO-56; discharge comes from plots above an ET pond. Midsummer keeps a shallow
    pond instead of emptying.
    """

    name = "et_match"
    defaults = dict(
        pond_mm=22.0,
        et_cover_days=1.0,
        overshoot=1.1,
        seepage_mm=12.0,
        ceiling_mm=95.0,
        min_pond_mm=6.0,
        catchup_gain=1.2,
        miss_gain=0.8,
        neighbour_et_gain=0.15,
        surface_gain=0.35,
        sat_fresh_days=16.0,
        midsummer_start_day=54.0,
        midsummer_days=14.0,
        midsummer_pond_mm=16.0,
        final_drain_day=122.0,
        ir_need_gain=1.5,
        ir_dry_gain=0.8,
        spill_fudge=0.02,
        pool_capture_share=None,
        plow_pan_k_mm_d=13.0,
        irrigate_ncb=1.0,
    )

    def _demand_mm(self, batch):
        expected = np.maximum(self.expected_et_mm(batch), 0.0)
        sat = np.nan_to_num(np.asarray(batch.sat_et_mm, dtype=float), nan=0.0)
        age = np.nan_to_num(np.asarray(batch.sat_days_ago, dtype=float), nan=99.0)
        fresh = (sat > 0.0) & (age <= self.params["sat_fresh_days"])
        return np.maximum(expected, np.where(fresh, sat, expected))

    def _keep_mm(self, batch, demand):
        p = self.params
        day = float(batch.day)
        mid = p["midsummer_start_day"] <= day < p["midsummer_start_day"] + p["midsummer_days"]
        pond = p["midsummer_pond_mm"] if mid else p["pond_mm"]
        need = np.maximum(np.asarray(batch.need_mm, dtype=float), 0.0)
        topsoil = np.maximum(np.asarray(getattr(batch, "topsoil_mm", 0.0), dtype=float), 0.0)
        hotter = p["neighbour_et_gain"] * np.maximum(
            demand - np.maximum(np.asarray(batch.neighbour_et_mm, dtype=float), 0.0), 0.0)
        dry_surface = p["surface_gain"] * np.maximum(9.0 - topsoil, 0.0)
        from . import okura as _okura
        er = _okura.effective_rain_mm(np.asarray(batch.rain_mm, dtype=float))
        fill = (pond + p["et_cover_days"] * np.maximum(demand - er, 0.0)
                + p["catchup_gain"] * need
                + hotter + dry_surface + p["seepage_mm"]) * p["overshoot"]
        return np.minimum(fill, p["ceiling_mm"])

    def _renorm_blocks(self, mm, want_mm, rows):
        from .identify import match_block_volumes

        area = self._area
        block = self._block
        cyclic = self._cyclic
        full = np.zeros(area.size)
        full[rows] = np.maximum(mm, 0.0)
        want = {}
        for name in ("North", "South"):
            sel = (block == name) & cyclic
            want[name] = float((want_mm[sel] * area[sel]).sum() / 1000.0) if np.any(sel) else 0.0
        weights = area * np.maximum(full, 1e-6)
        return match_block_volumes(want, area, block, cyclic, weights=weights)[rows]

    def decide_batch(self, batch):
        p = self.params
        rows = self.rows
        day = int(batch.day)
        self._day = day
        self._sync_drainage_catchup(day)
        depth = np.asarray(batch.depth_mm, dtype=float)
        cyclic = np.asarray(batch.cyclic, dtype=bool)
        demand = self._demand_mm(batch)
        keep = self._keep_mm(batch, demand)
        need = np.maximum(np.asarray(batch.need_mm, dtype=float), 0.0)
        last_et = np.maximum(np.asarray(getattr(batch, "last_et_mm", 0.0), dtype=float), 0.0)
        miss = p["miss_gain"] * np.maximum(demand - last_et, 0.0)
        dry = np.maximum(keep - depth, 0.0)
        ir_score = np.where(
            cyclic, 0.05 + demand + p["ir_need_gain"] * need + p["ir_dry_gain"] * dry + miss, 0.0)
        irrigation = self._renorm_blocks(ir_score, self.irrigation[day], rows)
        irrigation = self._merge_ncb_irrigation(irrigation, day, rows)
        # Spill-compensated Okura D, then reweight toward wet plots.
        base_dr = self._spill_adjusted_discharge(
            irrigation, depth, demand, self.discharge[day], rows)
        if day >= p["final_drain_day"]:
            return irrigation, self._merge_ncb_discharge(base_dr, day, rows)
        excess = np.maximum(depth + irrigation - keep, 0.0)
        dr_score = np.where(cyclic, 0.05 + excess, 0.0)
        # Rebuild want mm on full field from spill-adjusted row values.
        want_full = np.zeros(self._area.size)
        want_full[rows] = base_dr
        discharge = self._renorm_blocks(dr_score, want_full, rows)
        return irrigation, self._merge_ncb_discharge(discharge, day, rows)


@register_agent
class RatioFinder(FieldAggregate):
    """Contribution agent: inferred underground flow ratios (Kx, Ky, Kz, ground).

    Surface IR/D come from Fig. 8 CSV via FieldAggregate (known forcing, not novelty).
    This agent attaches the per-point leftover simplex: pool_capture → same-day LFV,
    Kx+Ky → neighbours (ET), stay → deep. See ``cyclic.infer_underground_ratios``.
    Near-canal cells get more LFV; district mean matches Table 5. Simulated ET is
    FAO-56, not forced satellite ET.
    """

    name = "ratio_finder"

    def reset(self, model, rows):
        super().reset(model, rows)
        # Default off: QRC is optional; core path uses paper plot shares + penetration.
        if not hasattr(model, "use_spatial_qrc"):
            model.use_spatial_qrc = False
        from .identify import (attach_pixel_shares, construct_ratios, leftover_weighted_stay,
                               load_best_ratios)
        from .physics import Physics
        span = getattr(model, "span", "season")
        saved = load_best_ratios(model.year, span)
        p = model.physics
        has = float(p.kx_ratio) + float(p.ky_ratio) + float(p.kz_ratio) + float(p.ground_ratio) > 1e-9
        constructed = construct_ratios(model.year)
        stay_target = float(constructed["kz_ratio"] + constructed["ground_ratio"])
        if has:
            ratios = {key: float(getattr(p, key)) for key in
                      ("kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio", "farm_to_main")}
            ratios["pool_capture_share"] = float(getattr(
                p, "pool_capture_share", constructed["pool_capture_share"]))
            ratios["neighbour_of_remain"] = float(getattr(
                p, "neighbour_of_remain", constructed.get("neighbour_of_remain", 0.5)))
            ratios["ratio_split"] = True
            ratios["force_satellite_et"] = False
        elif saved is not None:
            ratios = saved.params
        else:
            ratios = constructed
        allowed = {item.name for item in fields(Physics)}
        model.physics = model.physics.replace(**{k: v for k, v in ratios.items() if k in allowed})
        attach_pixel_shares(model, ratios)
        if (not has) and saved is not None and saved.fields:
            n = np.size(model.pixel_shares["kx_ratio"])
            if np.size(saved.fields.get("kx_ratio", [])) == n:
                model.pixel_shares.update(saved.fields)
        rain = np.asarray(model.inputs.rain_mm, dtype=float)
        if rain.ndim == 1:
            rain = rain[:, None]
        et = np.asarray(model.inputs.et_target_mm, dtype=float)
        leftover = self.irrigation + rain[:self.irrigation.shape[0]] - self.discharge
        leftover = leftover - et[:self.irrigation.shape[0]]
        top = float(model.physics.total_evaporable_mm /
                    max(model.physics.total_evaporable_mm + model.physics.total_available_mm, 1e-9))
        shares = model.pixel_shares
        n_share = np.size(shares["kx_ratio"])
        spatial = None
        if leftover.shape[1] == model.n and getattr(model, "use_spatial_qrc", False):
            try:
                from .qrc import apply_spatial_qrc
                rain_1d = rain[:, 0] if rain.ndim == 2 and rain.shape[1] == 1 else (
                    rain if rain.ndim == 1 else rain.mean(axis=1))
                spatial = apply_spatial_qrc(
                    model, leftover, self.irrigation, self.discharge,
                    et[:leftover.shape[0]], rain_1d)
                if spatial is not None:
                    paddy_shares, paddy_daily = spatial.shares, spatial.daily
                    if (model.cells is not None
                            and np.size(paddy_shares.get("kx_ratio", [])) == model.n):
                        owner = model.cells.owner
                        paddy_shares = {key: (np.asarray(val)[owner]
                                              if np.ndim(val) == 1 and np.size(val) == model.n
                                              else (np.asarray(val)[:, owner]
                                                    if np.ndim(val) == 2
                                                    and np.shape(val)[-1] == model.n
                                                    else val))
                                        for key, val in paddy_shares.items()}
                        paddy_daily = {key: (np.asarray(val)[:, owner]
                                             if np.ndim(val) == 2
                                             and np.shape(val)[-1] == model.n
                                             else val)
                                       for key, val in paddy_daily.items()}
                    model.pixel_shares.update(paddy_shares)
                    model.pixel_daily = paddy_daily
                    shares = model.pixel_shares
                    n_share = np.size(shares["kx_ratio"])
                    self.params["qrc_spatial_clusters"] = spatial.n_clusters
                    self.params["qrc_spatial_rmse"] = spatial.rmse
            except Exception:
                spatial = None
        # Mark contribution on the model for reports / UI.
        model.contribution = "inferred_underground_flow_ratios"
        model.underground_claim = (
            "per-point LFV (pool_capture) vs neighbours (Kx+Ky) vs stay (Kz+ground); "
            "surface IR known")
        if not has and spatial is None:
            if leftover.shape[1] == n_share:
                leftover_weighted_stay(shares, leftover, model.area, model.inputs.cyclic,
                                       stay_target, top)
            elif model.cells is not None and n_share == len(model.cells.owner):
                owner = model.cells.owner
                leftover_weighted_stay(shares, leftover[:, owner],
                                       model.area[owner] * model.cells.weights,
                                       np.asarray(model.inputs.cyclic[owner], dtype=bool),
                                       stay_target, top)
        from .canal import weighted_share_mean
        w_area, w_cyclic = model.area, model.inputs.cyclic
        extra = None
        if model.cells is not None and np.size(shares["kx_ratio"]) == len(model.cells.owner):
            owner = model.cells.owner
            w_area = model.area[owner]
            w_cyclic = np.asarray(model.inputs.cyclic[owner], dtype=bool)
            extra = model.cells.weights
        ratios["kx_ratio"] = weighted_share_mean(shares["kx_ratio"], w_area, w_cyclic, extra)
        ratios["ky_ratio"] = weighted_share_mean(shares["ky_ratio"], w_area, w_cyclic, extra)
        ratios["kz_ratio"] = weighted_share_mean(shares["kz_ratio"], w_area, w_cyclic, extra)
        ratios["ground_ratio"] = weighted_share_mean(shares["ground_ratio"], w_area, w_cyclic, extra)
        if "pool_capture_share" in shares:
            ratios["pool_capture_share"] = weighted_share_mean(
                shares["pool_capture_share"], w_area, w_cyclic, extra)
        for key in ("kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio", "farm_to_main",
                    "pool_capture_share"):
            if key in ratios:
                self.params[key] = ratios[key]
        self.params["stay_target"] = stay_target
        self.params["stay_weighted"] = float(shares.get("stay_weighted", stay_target))


@register_agent
class Rainfed(FarmerPolicy):
    """Never irrigates, never opens the outlet. The lower bound."""

    name = "rainfed"

    def decide_batch(self, batch):
        return np.zeros(len(batch)), np.zeros(len(batch))


@register_agent
class Flooded(FarmerPolicy):
    """Tops up to a fixed depth every day and never drains. The upper bound."""

    name = "flooded"
    defaults = dict(depth_mm=60.0)
    bounds = dict(depth_mm=(10.0, 100.0))

    def decide_batch(self, batch):
        return np.maximum(self.params["depth_mm"] - batch.depth_mm, 0.0), np.zeros(len(batch))


@register_agent
class LinearPolicy(FarmerPolicy):
    """irrigation = relu(theta_I . phi), discharge = relu(theta_D . phi), where phi is what
    the farmer sees plus smooth bumps over the season (so a calendar can be learned).
    Fitted by imitation.AggregateCloning."""

    name = "linear"
    CENTRES = np.linspace(0.0, 130.0, 12)
    WIDTH_DAYS = 8.0
    FEATURES = ["bias", "depth/10", "expected_et", "rain/10", "pass_age/10"] + [
        f"day~{int(c)}" for c in CENTRES]
    defaults = dict(theta_irrigation=None, theta_discharge=None, max_irrigation_mm=150.0)

    @classmethod
    def features(cls, batch):
        n = len(batch)
        season = np.exp(-0.5 * ((batch.day - cls.CENTRES) / cls.WIDTH_DAYS) ** 2)
        return np.column_stack([
            np.ones(n), batch.depth_mm / 10.0, cls.expected_et_mm(batch), batch.rain_mm / 10.0,
            np.nan_to_num(batch.sat_days_ago, nan=30.0) / 10.0, np.tile(season, (n, 1)),
        ])

    def decide_batch(self, batch):
        phi = self.features(batch)
        k = phi.shape[1]
        theta_i = self.params["theta_irrigation"]
        theta_d = self.params["theta_discharge"]
        theta_i = np.zeros(k) if theta_i is None else theta_i
        theta_d = np.zeros(k) if theta_d is None else theta_d
        irrigation = np.clip(phi @ theta_i, 0.0, self.params["max_irrigation_mm"])
        discharge = np.maximum(phi @ theta_d, 0.0)
        return irrigation, discharge


class WaterFatePolicy(FarmerPolicy):
    """Gym policy: given today's pool IR (`obs.irrigation_sent_mm`), output surface
    discharge and optional leftover shares. Irrigation in the return is a proposed
    disaggregation of the known Okura block total; the env renormalises it."""

    name = "water_fate"

    def decide_batch(self, batch):
        n = len(batch)
        return {"irrigation_mm": np.zeros(n), "discharge_mm": np.zeros(n)}


@register_agent
class PersistDischarge(FarmerPolicy):
    """Copy yesterday's surface return. First morning is zeros."""

    name = "persist"

    def decide_batch(self, batch):
        d = np.asarray(getattr(batch, "last_discharge_mm", np.zeros(len(batch))), dtype=float)
        return np.zeros(len(batch)), np.maximum(d, 0.0)


@register_agent
class LinearFate(LinearPolicy):
    """Linear D from weather, pond, given IR and yesterday's return. IR output is unused
    when the gym supplies Okura irrigation."""

    name = "linear_fate"
    FEATURES = LinearPolicy.FEATURES + ["ir_sent/10", "last_d/10"]

    @classmethod
    def features(cls, batch):
        base = super().features(batch)
        n = len(batch)
        ir = np.asarray(getattr(batch, "irrigation_sent_mm", np.zeros(n)), dtype=float)
        last = np.asarray(getattr(batch, "last_discharge_mm", np.zeros(n)), dtype=float)
        return np.column_stack([base, ir / 10.0, last / 10.0])

    def decide_batch(self, batch):
        _, discharge = super().decide_batch(batch)
        return np.zeros(len(batch)), discharge


def resolve_policies(agents, inputs):
    """Which policy farms which paddy. Returns (policy per paddy, [(policy, rows)])."""
    n = len(inputs.paddy_id)
    made = {}

    def once(key, spec):
        if key not in made:
            made[key] = make_agent(spec)
        return made[key]

    if isinstance(agents, dict):
        policy_of = [once(("block", b), agents.get(b, "et_follower")) for b in inputs.block]
    elif callable(agents) and not isinstance(agents, (FarmerPolicy, type)):
        policy_of = []
        for i in range(n):
            spec = agents(i, inputs)
            key = id(spec) if isinstance(spec, FarmerPolicy) else spec
            policy_of.append(once(key, spec))
    else:
        policy = make_agent(agents)
        policy_of = [policy] * n

    groups = {}
    for i, policy in enumerate(policy_of):
        groups.setdefault(id(policy), (policy, []))[1].append(i)
    return policy_of, [(policy, np.array(rows)) for policy, rows in groups.values()]


# --- Mesa agents -----------------------------------------------------------------------------

class PaddyAgent(mesa.Agent):
    """One paddy and the farmer deciding for it. Its water lives in the model's arrays."""

    def __init__(self, model, index, paddy_id, block, cyclic, area_m2, policy):
        super().__init__(model)
        self.index = index
        self.paddy_id = paddy_id
        self.block = block
        self.cyclic = cyclic
        self.area_m2 = area_m2
        self.policy = policy

    @property
    def depth_mm(self):
        return float(self.model.state.water[self.index])

    def observe(self):
        return self.model.observations.row(self.index)

    def decide(self):
        irrigation, discharge = self.policy.decide(self.observe())
        self.model.request_irrigation[self.index] = irrigation
        self.model.request_discharge[self.index] = discharge


class PumpingStation(mesa.Agent):
    """The pool at the Shiroyama-Jimbee irrigation pumping station (Okura et al. Eq. 4–6, Table 5).

    Actions each day, all as volumes (m³), matching Fig. 2b–c and Table 5:

      1. Cyclic D_North and D_South enter the main drainage canal / pool (ADCP D).
      2. LFV_pump (same-day recapture of infiltrated leftover, Eq. 7 share of cyclic Lf) enters.
      3. IR_N_all and IR_S_all are pumped to the whole North and South areas (cyclic
         plus non-cyclic; Eq. 5 splits IR_all = IR + IR_ncb by command area in the paper;
         here they are the sums of what was actually served).
      4. Lake withdrawal W fills any shortfall (gravity culvert from Kita-Imbanuma).
      5. Drainage-station O sends surplus from the canal/pool back to the lake.
      6. Non-cyclic D does not enter the pool; it is the NCB drainage stock to the
         lake / Tone.

    Eq. 4 (pool, rainfall on the canal ignored):
        Δpool = W − O + LFV_pump + D_N + D_S − IR_N_all − IR_S_all
    Eq. 6 residual LFV is stored as LFV_eq6_m3 for comparison with Table 5.

    The lake is unlimited unless `allowance_m3` or `w_daily_cap_m3` is set.
    Irrigation to both fields cannot exceed pool stock + same-day cyclic D
    returns + LFV + today's lake take (`supply_fraction`); NCB discharge does
    not credit the pool.
    """

    def __init__(self, model, capacity_m3, allowance_m3=None,
                 w_ema=0.0, w_daily_cap_m3=None):
        super().__init__(model)
        self.capacity = float(capacity_m3)
        self.storage = float(capacity_m3)
        self.allowance = allowance_m3
        self.withdrawn = 0.0
        self.expected_inflow = 0.0
        self.ncb_storage = 0.0
        # Station W policy (shape only): EMA + daily cap for plotted W_policy_m3.
        # Physical W_m3 still equals the Eq. 4 shortfall so Table 5 ΣW closes —
        # unless w_daily_cap_m3 also hard-caps lake take (and IR via supply_fraction).
        self.w_ema = float(np.clip(w_ema, 0.0, 1.0))
        self.w_daily_cap_m3 = None if w_daily_cap_m3 is None else float(w_daily_cap_m3)
        self._w_filt = 0.0
        self._w_o_debt = 0.0  # lake borrowed for O; repaid from later W_force
        self.w_hard_copy = False
        self.today = {key: 0.0 for key in (
            "W_m3", "W_policy_m3", "O_m3", "LFV_pump_m3", "LFV_eq6_m3",
            "D_North_m3", "D_South_m3", "D_pool_m3",
            "IR_all_North_m3", "IR_all_South_m3", "IR_all_m3", "IR_ncb_m3",
            "pool_m3", "d_pool_m3", "D_ncb_m3", "ncb_stock_m3")}

    def lake_room_m3(self):
        """How much lake water may still be taken today (m³)."""
        if self.allowance is not None:
            return max(float(self.allowance) - float(self.withdrawn), 0.0)
        cap = getattr(self, "w_daily_cap_m3", None)
        if cap is not None and float(cap) > 0.0:
            return float(cap)
        return None  # unlimited

    def available_irrigation_m3(self, cyclic_d_m3=0.0, lfv_m3=0.0, o_m3=0.0):
        """Max IR_all today: pool + cyclic D + LFV + lake − planned O (Eq. 4).

        Outflow O leaves the pool the same day, so it cannot also fund irrigation.
        NCB discharge does not credit the pool.
        """
        lake = self.lake_room_m3()
        base = (float(self.storage)
                + max(float(cyclic_d_m3), 0.0)
                + max(float(lfv_m3), 0.0)
                - max(float(o_m3), 0.0))
        if lake is None:
            return None
        return max(base + float(lake), 0.0)

    def supply_fraction(self, demand_m3, cyclic_d_m3=0.0, lfv_m3=0.0, o_m3=0.0):
        """Share of requested IR_all that pool + cyclic returns + lake − O can serve."""
        if demand_m3 <= 0:
            return 1.0
        available = self.available_irrigation_m3(
            cyclic_d_m3=cyclic_d_m3, lfv_m3=lfv_m3, o_m3=o_m3)
        if available is None:
            return 1.0
        return float(min(1.0, max(available, 0.0) / demand_m3))

    def settle(self, IR_North_all_m3, IR_South_all_m3, D_North_m3, D_South_m3,
               LFV_pump_m3, D_ncb_m3=0.0, IR_ncb_m3=0.0, O_force_m3=None,
               W_force_m3=None):
        """One day of Table 5 / Eq. 4 bookkeeping. Every argument is m³.

        The station pool is a **transit** node (water moves to/from the lake many
        times per day). At daily resolution it only keeps a small working buffer
        (``capacity``); anything above that leaves the same day as O to the lake.
        ``pool_m3`` is that buffer, not a season store.

        When ``O_force_m3`` is set (Table 5 O plan), outflow is matched using
        capacity spill, then pool, then remaining lake room — not an uncapped
        lake draw that would break IR ≤ W + D + LFV − O.

        When ``W_force_m3`` is set (Fig. 8 digitised lake withdrawal), lake take
        follows that plan (capped by allowance / daily cap), and O absorbs the
        pool residual.
        """
        from . import okura

        ir_n = max(float(IR_North_all_m3), 0.0)
        ir_s = max(float(IR_South_all_m3), 0.0)
        d_n = max(float(D_North_m3), 0.0)
        d_s = max(float(D_South_m3), 0.0)
        lfv = max(float(LFV_pump_m3), 0.0)
        ir_all = ir_n + ir_s
        d_pool = d_n + d_s
        before = self.storage
        stock = self.storage + d_pool + lfv
        room = self.lake_room_m3()  # None = unlimited; else today's lake budget
        if W_force_m3 is not None:
            want_w = max(float(W_force_m3), 0.0)
            # Hard-copy Fig. 8 W: no borrow / repay that would change daily lake take.
            if not getattr(self, "w_hard_copy", False):
                debt = float(getattr(self, "_w_o_debt", 0.0) or 0.0)
                if debt > 0.0 and want_w > 0.0:
                    pay = min(debt, want_w)
                    want_w -= pay
                    self._w_o_debt = debt - pay
            lake = want_w if room is None else min(want_w, float(room))
        else:
            need = max(ir_all - stock, 0.0)
            lake = need
            if room is not None:
                lake = min(lake, float(room))
        self.withdrawn += lake
        self.storage = max(stock + lake - ir_all, 0.0)
        if O_force_m3 is not None:
            # Table 5 O pin: match daily plan from pool/spill.
            # When W is also forced (Fig. 8), borrow lake for O shortfalls and
            # repay from later W_force days (`_w_o_debt`) so season W and O both hold.
            want_o = max(float(O_force_m3), 0.0)
            soft_cap = max(float(self.capacity), want_o)
            w_locked = W_force_m3 is not None
            if w_locked:
                take = min(want_o, self.storage)
                short = want_o - take
                self.storage -= take
                # Hard-copy W: O from pool/spill only — never borrow lake past Fig. 8.
                if short > 0.0 and not getattr(self, "w_hard_copy", False):
                    room_left = None if room is None else max(float(room) - lake, 0.0)
                    borrow = short if room_left is None else min(short, room_left)
                    lake += borrow
                    self.withdrawn += borrow
                    self._w_o_debt = float(getattr(self, "_w_o_debt", 0.0) or 0.0) + borrow
                    outflow = take + borrow
                else:
                    outflow = take
            elif self.storage <= soft_cap + want_o:
                if self.storage >= want_o:
                    self.storage -= want_o
                    outflow = want_o
                else:
                    still = want_o - self.storage
                    self.storage = 0.0
                    room_left = None if room is None else max(float(room) - lake, 0.0)
                    take = still if room_left is None else min(still, room_left)
                    lake += take
                    self.withdrawn += take
                    outflow = want_o - still + take
                if self.storage > soft_cap:
                    outflow += self.storage - soft_cap
                    self.storage = soft_cap
            else:
                # Surplus beyond buffer + plan: must spill the excess.
                outflow = self.storage - soft_cap
                self.storage = soft_cap
        else:
            spill = max(self.storage - self.capacity, 0.0)
            self.storage -= spill
            outflow = spill
        # Causal EMA + daily cap → W_policy_m3 (Fig. 8 overlay); W_m3 is Eq. 4 / force.
        alpha = float(getattr(self, "w_ema", 0.0) or 0.0)
        if alpha > 0.0:
            self._w_filt = alpha * lake + (1.0 - alpha) * float(getattr(self, "_w_filt", 0.0))
            w_pol = self._w_filt
        else:
            self._w_filt = lake
            w_pol = lake
        cap = getattr(self, "w_daily_cap_m3", None)
        if cap is not None and float(cap) > 0.0:
            w_pol = min(float(w_pol), float(cap))
        self.expected_inflow = d_pool + lfv
        ncb = max(float(D_ncb_m3), 0.0)
        self.ncb_storage += ncb
        lfv_eq6 = okura.eq6_lfv_pump(lake, outflow, d_n, d_s, ir_n, ir_s)
        self.today = {
            "W_m3": lake, "W_policy_m3": float(w_pol),
            "O_m3": outflow, "LFV_pump_m3": lfv, "LFV_eq6_m3": lfv_eq6,
            "D_North_m3": d_n, "D_South_m3": d_s, "D_pool_m3": d_pool,
            "IR_all_North_m3": ir_n, "IR_all_South_m3": ir_s, "IR_all_m3": ir_all,
            "IR_ncb_m3": max(float(IR_ncb_m3), 0.0),
            "pool_m3": self.storage, "d_pool_m3": self.storage - before,
            "D_ncb_m3": ncb, "ncb_stock_m3": self.ncb_storage,
        }

"""One day of paddy hydrology for every plot at once.

Each paddy carries four stores, all in mm over its own area:

    water      ponded depth above the soil surface
    surface    depletion of the thin top soil layer evaporation draws on (0 = wet)
    deficit    depletion of the root zone below saturation (0 = saturated)
    perched    the perched water table above the farm-drain level, below the plow pan
    need       millimetres of T or E the root zone / surface could not supply; a missing
               stock, not water present on the plot

EVAPOTRANSPIRATION, TIED TO THE WATER A FARMER PUTS ON. FAO-56's dual crop coefficient
(Allen, Pereira, Raes & Smith 1998, Chapter 7):

    ET = (Ks x Kcb + Ke) x ET0

    Kcb   transpiration, from the canopy cover fc the satellite sees (Carlson & Ripley 1997):
              fc  = ((NDVI - NDVI_soil) / (NDVI_canopy - NDVI_soil))^2, clipped to 0-1
              Kcb = Kcb_bare + (Kcb_full - Kcb_bare) x fc
    Ks    root-zone water stress: 1 while water stands or the root zone has lost less than
          the readily available water RAW, falling linearly to 0 at the total available TAW:
              Ks = (TAW - deficit) / (TAW - RAW)
          FAO-56 Table 22 gives rice a depletion fraction p = 0.20, the most sensitive of
          any crop listed: RAW = p x TAW
    Ke    evaporation from the water surface, or the wet soil, between the plants:
              Ke = min(Kr x (Kc_max - Ks x Kcb), few x Kc_max),     few = 1 - fc
          Kr = 1 while water stands or the surface layer has lost less than the readily
          evaporable water REW, falling to 0 at the total evaporable water TEW:
              Kr = (TEW - surface) / (TEW - REW)

What that makes of a farmer's choices. Kept flooded, a paddy evaporates from open water
between the plants and transpires fully: ET = min(Kc_max, Kcb + few x Kc_max) x ET0, the
most it can be. Drained, the open-water evaporation stops within a day or two as the
surface dries (Kr), and once the root zone is drawn past RAW the crop transpires less (Ks).
Never watered, ET falls to what rain sustains. The satellite's ET lies between those
bounds, and where between is what the farmers' irrigation decides.

The coefficients below are starting values in the range FAO-56 gives for rice and a
clay-loam paddy soil. They are meant to be calibrated, not quoted.

The day, in order: rain and irrigation land on the pond; Ks from beginning-of-day root-zone
depletion (1 if the plot is ponded); T = Ks Kcb ET0 taken from the root zone, never from
rainfall or irrigation directly; E = Ke ET0 from the pond if water stands, else from the
surface layer; the farmer's discharge, overflow and outlet leak; infiltration from the pond
in proportion to ponded depth (K_pan h / h_ref, capped at the pond) into the
surface layer, then the root zone (replenishing T), then the perched layer; lateral
flow and deep percolation. Pond, root zone and the station pool are the three stores.

Storage identity, per plot and per day, which the model checks:
    d(water - surface - deficit + perched) = rain + irrigation - ET - D - lateral - deep + neighbor
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

# --- the satellite's rice crop coefficient, for the observed ET -------------------------------
# Vu, Watanabe & Takagi (2005), Agricultural Water Management 76, 195-210, Table 4:
# lysimeters at Tsukuba and Fuchu on the Kanto plain. Kc-ini 1.10, Kc-mid 1.155. This is the
# *observed* ET the satellite implies (data.season_inputs); the simulation uses the dual
# coefficient below, which responds to water.
KC_INI = 1.10
KC_MID = 1.155
NDVI_BARE = 0.426
NDVI_FULL = 0.736
KC_FALLBACK = 0.5 * (KC_INI + KC_MID)

# --- canopy cover from NDVI, for the simulation's transpiration -------------------------------
# Carlson & Ripley (1997), Remote Sensing of Environment 62, 241-252: fc = N*^2 with N* the
# NDVI scaled between bare soil and full canopy.
NDVI_SOIL = 0.15
NDVI_CANOPY = 0.80


def kc_from_ndvi(ndvi):
    """Crop coefficient interpolated between the two measured anchors. NaN stays NaN."""
    ndvi = np.asarray(ndvi, dtype=float)
    fraction = np.clip((ndvi - NDVI_BARE) / (NDVI_FULL - NDVI_BARE), 0.0, 1.0)
    value = KC_INI + (KC_MID - KC_INI) * fraction
    return float(value) if value.ndim == 0 else value


def canopy_cover(ndvi):
    """Carlson & Ripley's fractional vegetation cover. NaN stays NaN."""
    scaled = np.clip((np.asarray(ndvi, dtype=float) - NDVI_SOIL) / (NDVI_CANOPY - NDVI_SOIL), 0.0, 1.0)
    return scaled ** 2


@dataclass
class Physics:
    """Soil, drainage and crop constants. Any of them can be handed to a learner as well."""

    # Pond → soil, proportional to ponded depth, not a leftover-simplex ratio and not
    # gated by TEW dryness. plow_pan_k_mm_d is the infiltration when the pond equals
    # pond_infiltrate_ref_mm (30 mm puddled start). Old (h+D_pan)/D_pan made a 10 mm
    # pond infiltrate almost the same millimetres as a 50 mm pond. Capped at the pond
    # in advance(). Not in identify.ACTIVE.
    plow_pan_k_mm_d: float = 9.5
    plow_pan_depth_mm: float = 150.0
    pond_infiltrate_ref_mm: float = 30.0
    # Hooghoudt. The land improvement project left a dense drain network, hence short L.
    lateral_k_mm_d: float = 2000.0
    # Anisotropic horizontal conductivity (mm/d). None = use lateral_k_mm_d both ways.
    # kx is east-west, ky is north-south. The unknown we may try to recover.
    kx_mm_d: Optional[float] = None
    ky_mm_d: Optional[float] = None
    drain_spacing_m: float = 10.0
    equivalent_depth_m: float = 0.8
    drainable_porosity: float = 0.15
    deep_percolation_mm_d: float = 1.0
    # Outlet pipes and bare-soil farm drains leak a share of standing water each day.
    outlet_loss_per_day: float = 0.13
    bund_mm: float = 100.0
    # He et al. (2025): nitrogen lateral seepage through paddy bunds is concentrated
    # in the shallow bund (the 10–20 cm layer does not delay under controlled irrigation).
    # Leftover hops are that bund path, not Tsubo Dupuit through a sloping aquifer.
    bund_seepage_mm: float = 150.0
    # FAO-56 dual crop coefficient for rice.
    kc_max: float = 1.20            # the most a wet paddy can evaporate and transpire (Eq. 72, unadjusted)
    # Xu et al. (2017): FAO Kcp = 1.2 is too low for the ET pulse after wetting under
    # drying–wetting rice; they raised it to 1.29. Used as the Ke cap on rain/irrigation
    # days only. Kcb stays FAO/Vu (Japanese), not Xu's East-China 1.52.
    kc_wet: float = 1.29
    kcb_bare: float = 0.15          # basal coefficient with no canopy
    kcb_full: float = 1.15          # basal coefficient at full canopy
    readily_evaporable_mm: float = 9.0     # REW, clay loam (Table 19)
    total_evaporable_mm: float = 25.0      # TEW, clay loam (Table 19)
    # Root zone: rice p = 0.20 (Table 22) on ~100 mm available in a puddled 0.5 m profile.
    readily_available_mm: float = 20.0
    total_available_mm: float = 100.0
    force_satellite_et: bool = False
    # The pool at the station is about 18 x 68 m (Section 2.1); depth assumed 4 m.
    pool_capacity_m3: float = 18.0 * 68.0 * 4.0
    # Farm-drain stage (mm of water in the farm drains). Higher stage increases
    # leftover captured as lateral toward the pool (paper Lf). Cyclic D itself
    # returns to the pool the same day.
    farm_drain_stage_mm: float = 200.0
    farm_drain_ref_mm: float = 200.0
    farm_drain_lateral_gain: float = 1.0
    canal_length_m: float = 400.0
    canal_capture_frac: float = 0.35
    canal_max_frac: float = 0.5
    canal_routing: bool = True
    # Leftover travel (m/day). Poisson μ = leftover_speed_m_d / median neighbour spacing.
    leftover_speed_m_d: float = 2.0
    canal_hops: Optional[float] = None
    canal_catch_power: float = 4.0
    drain_travel_frac: float = 0.4
    lateral_per_discharge: float = 0.0
    farm_to_main: float = 1.0
    # Leftover (IR+R−D−ET) simplex:
    #   pool_capture_share  → same-day LFV (district mean = Table 5 LFV/LfA)
    #   kx_ratio, ky_ratio  → leftover sent to neighbours (cover ET shortfalls)
    #   kz_ratio, ground    → stay / deep (dWB)
    # Of the non-LFV remainder, neighbour_of_remain goes to Kx+Ky (rest stay).
    pool_capture_share: float = 0.74
    neighbour_of_remain: float = 0.50
    kx_ratio: float = 0.0
    ky_ratio: float = 0.0
    kz_ratio: float = 0.0
    ground_ratio: float = 0.0
    ratio_split: bool = False
    # Leftover routing as a graph (neighbours + canal sinks). Off: catch in place, no hops.
    use_graph: bool = True
    one_way_to_canal: bool = False
    as_qubo: bool = False
    graph_smoothness: float = 0.05

    def replace(self, **changes):
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values.update(changes)
        return Physics(**values)


def kx_of(p):
    return p.lateral_k_mm_d if p.kx_mm_d is None else float(p.kx_mm_d)


def ky_of(p):
    return p.lateral_k_mm_d if p.ky_mm_d is None else float(p.ky_mm_d)


def dual_coefficients(water, surface, deficit, canopy, p, wetted=None):
    """FAO-56 dual Kc at the start of ET. `deficit` is beginning-of-day root-zone depletion.

    Kcb from satellite canopy cover. Ks from the root-zone water balance (1 while ponded).
    Ke from Kr and uncovered wet fraction; Kr = 1 while ponded or the surface is still wet.
    On rain or irrigation days the Ke cap is Xu et al.'s (2017) wetting-pulse Kcp
    (`kc_wet`), otherwise FAO-56 `kc_max`.
    """
    water = np.asarray(water, dtype=float)
    n = water.size
    fc = np.clip(np.nan_to_num(np.broadcast_to(np.asarray(canopy, dtype=float), (n,)), nan=0.5), 0.0, 1.0)
    kcb = p.kcb_bare + (p.kcb_full - p.kcb_bare) * fc
    few = np.clip(1.0 - fc, 0.05, 1.0)
    ponded = water > 0
    taw, raw = p.total_available_mm, p.readily_available_mm
    tew, rew = p.total_evaporable_mm, p.readily_evaporable_mm
    ks = np.where(ponded | (deficit <= raw), 1.0,
                  np.clip((taw - deficit) / (taw - raw), 0.0, 1.0))
    kr = np.where(ponded | (surface <= rew), 1.0,
                  np.clip((tew - surface) / (tew - rew), 0.0, 1.0))
    kc_cap = np.full(n, float(p.kc_max))
    if wetted is not None:
        wet = np.broadcast_to(np.asarray(wetted, dtype=bool), (n,))
        kc_cap = np.where(wet, float(getattr(p, "kc_wet", p.kc_max)), kc_cap)
    ke = np.clip(np.minimum(kr * (kc_cap - ks * kcb), few * kc_cap), 0.0, None)
    return fc, kcb, few, ponded, ks, kr, ke, kc_cap


def apply_et(state, transpiration, evaporation, p, force=False):
    """Take T from the root zone and E from the pond (flooded) or surface (dry).

    Rain and irrigation are not ET sources. They sit on the pond until infiltration
    refills the root zone. Unmet T or E is `need` (groundwater wanted at that point).
    If `force`, that need is counted in `et_mm` so satellite ET still occurs in full.
    """
    water, surface, deficit = state.water, state.surface, state.deficit
    n = water.size
    transpiration = np.maximum(np.broadcast_to(np.asarray(transpiration, dtype=float), (n,)), 0.0)
    evaporation = np.maximum(np.broadcast_to(np.asarray(evaporation, dtype=float), (n,)), 0.0)
    ponded = water > 0
    from_root = np.minimum(transpiration, np.maximum(p.total_available_mm - deficit, 0.0))
    deficit += from_root
    need_t = transpiration - from_root
    from_pond = np.where(ponded, np.minimum(evaporation, np.maximum(water, 0.0)), 0.0)
    water -= from_pond
    e_rest = evaporation - from_pond
    from_surface = np.minimum(e_rest, np.maximum(p.total_evaporable_mm - surface, 0.0))
    surface += from_surface
    need_e = e_rest - from_surface
    need = need_t + need_e
    realized = from_pond + from_surface + from_root
    return {
        "et_pond_mm": from_pond,
        "et_surface_mm": from_surface,
        "et_root_mm": from_root,
        "et_groundwater_mm": need,
        "need_mm": need,
        "transpiration_mm": from_root + (need_t if force else 0.0),
        "evaporation_mm": from_pond + from_surface + (need_e if force else 0.0),
        "et_mm": realized + (need if force else 0.0),
        "ponded": ponded.astype(float),
    }


def supply_et_demand(state, demand_mm, p):
    """Meet the weather's ET demand from the water rice can actually use.

    ``demand_mm`` is the full ET the day's weather would drive from a well-watered
    crop. Rice draws it from the pond first, then the topsoil layer (down to TEW);
    the deeper root-zone store does not count. Whatever cannot be supplied is the
    day's water stress in mm: demand 8 mm with 2 mm available is 6 mm of stress.
    Mutates ``state`` in place; same keys as ``apply_et`` plus ``stress_mm``.
    """
    water, surface = state.water, state.surface
    n = water.size
    demand = np.maximum(np.broadcast_to(np.asarray(demand_mm, dtype=float), (n,)), 0.0)
    ponded = water > 0
    from_pond = np.minimum(demand, np.maximum(water, 0.0))
    water -= from_pond
    rest = demand - from_pond
    from_surface = np.minimum(rest, np.maximum(p.total_evaporable_mm - surface, 0.0))
    surface += from_surface
    stress = rest - from_surface
    realized = from_pond + from_surface
    zeros = np.zeros(n)
    return {
        "et_pond_mm": from_pond,
        "et_surface_mm": from_surface,
        "et_root_mm": zeros,
        "et_groundwater_mm": stress,
        "need_mm": stress,
        "stress_mm": stress,
        "transpiration_mm": zeros,
        "evaporation_mm": realized,
        "et_mm": realized,
        "ponded": ponded.astype(float),
    }


def split_satellite_et(et_mm, ks, kcb, ke, ponded):
    """Partition an observed ET total using the dual-Kc shares T : E, not NDVI alone."""
    et = np.maximum(np.asarray(et_mm, dtype=float), 0.0)
    transpiration = np.maximum(ks * kcb, 0.0)
    evaporation = np.maximum(ke, 0.0)
    dual = transpiration + evaporation
    t_frac = np.divide(transpiration, dual, out=np.where(ponded, 0.0, 1.0).astype(float), where=dual > 0)
    t = et * t_frac
    return t, et - t


def drain_k_mm_d(p, kx=None, ky=None):
    """Hooghoudt equivalent horizontal K: geometric mean of the two axes."""
    x = kx_of(p) if kx is None else kx
    y = ky_of(p) if ky is None else ky
    return np.sqrt(np.asarray(x, dtype=float) * np.asarray(y, dtype=float))


def percolation_mm(depth_mm, p, k_z=None):
    """Pond → soil in millimetres, proportional to ponded depth.

    I = K_pan * h / h_ref. Not a share of leftover L, not a function of TEW
    depletion. Remaining pond after this rate is taken stays on the plot.
    """
    k = p.plow_pan_k_mm_d if k_z is None else k_z
    h = np.maximum(np.asarray(depth_mm, dtype=float), 0.0)
    ref = max(float(getattr(p, "pond_infiltrate_ref_mm", 30.0)), 1.0)
    return np.asarray(k, dtype=float) * (h / ref)


def lateral_mm(perched_mm, p, k_h=None):
    head_m = np.maximum(perched_mm, 0.0) / 1000.0 / p.drainable_porosity
    k = (drain_k_mm_d(p) if k_h is None else k_h) / 1000.0
    q = (8 * k * p.equivalent_depth_m * head_m + 4 * k * head_m ** 2) / p.drain_spacing_m ** 2
    return q * 1000.0


@dataclass
class State:
    water: np.ndarray
    surface: np.ndarray
    deficit: np.ndarray
    perched: np.ndarray
    need: np.ndarray
    drain: np.ndarray

    @classmethod
    def start(cls, n, depth_mm=30.0, perched_mm=10.0):
        """Puddled and flooded, which is where the irrigation period starts (Section 2.5)."""
        return cls(np.full(n, float(depth_mm)), np.zeros(n), np.zeros(n),
                   np.full(n, float(perched_mm)), np.zeros(n), np.zeros(n))

    def storage(self):
        return self.water - self.surface - self.deficit + self.perched


def advance(state, rain_mm, et0_mm, canopy, irrigation_mm, discharge_mm, p,
            k_z=None, k_x=None, k_y=None, edges=None, area_m2=None, stress="root_zone",
            weir_mm=None):
    """One day, in place on `state`. `canopy` is each paddy's cover fc (0-1). Returns the
    day's fluxes in mm per plot, and the day's Ks and Kr.

    `stress` picks how ET meets the weather. "root_zone" (default, the paper runs):
    FAO-56 Ks/Kr gate T and E, with no stress while any water stands. "supply": the
    full weather-driven ET (well-watered dual Kc x ET0) is demanded every day and
    drawn from pond, then topsoil (the water rice uses); `stress_mm` is the unmet part
    and Ks = ETa / demand (see supply_et_demand).

    `k_z`, `k_x`, `k_y` are optional per-point conductivities (mm/d). `edges` is an
    anisotropic neighbour graph (see lateral.grid_edges); perched water then moves
    sideways before Hooghoudt drains it toward the pool.

    `weir_mm` (optional, per plot): crest height of the outlet weir board the farmer has
    set. Pond above the crest after ET and any chosen discharge leaves through the
    outlet the same day and counts as discharge (D), before the bund can overflow.
    None (default) = no board, only the chosen discharge.
    """
    water, surface, deficit, perched = state.water, state.surface, state.deficit, state.perched
    n = water.size
    deficit_0 = deficit.copy()

    water += rain_mm + irrigation_mm
    rain = np.broadcast_to(np.asarray(rain_mm, dtype=float), (n,))
    irrigation = np.broadcast_to(np.asarray(irrigation_mm, dtype=float), (n,))
    wetted = (rain + irrigation) > 0.0
    fc, kcb, few, ponded, ks, kr, ke, kc_cap = dual_coefficients(
        water, surface, deficit_0, canopy, p, wetted=wetted)
    et_potential = np.minimum(kc_cap, kcb + few * kc_cap) * et0_mm
    if stress == "supply":
        taken = supply_et_demand(state, et_potential, p)
        ks = np.divide(taken["et_mm"], et_potential, out=np.ones(n), where=et_potential > 0)
    else:
        taken = apply_et(state, ks * kcb * et0_mm, ke * et0_mm, p, force=False)
        taken["stress_mm"] = np.zeros(n)
    water, surface, deficit = state.water, state.surface, state.deficit

    discharge = np.minimum(np.maximum(discharge_mm, 0.0), water)
    water -= discharge
    if weir_mm is not None:
        over_weir = np.maximum(water - np.maximum(np.asarray(weir_mm, dtype=float), 0.0), 0.0)
        water -= over_weir
        discharge = discharge + over_weir
    overflow = np.maximum(water - p.bund_mm, 0.0)
    water -= overflow
    spill = water * p.outlet_loss_per_day
    water -= spill

    infiltration = np.where(water > 0, np.minimum(percolation_mm(water, p, k_z), water), 0.0)
    water -= infiltration
    to_surface = np.minimum(infiltration, surface)
    surface -= to_surface
    to_root_zone = np.minimum(infiltration - to_surface, deficit)
    deficit -= to_root_zone
    perched += infiltration - to_surface - to_root_zone

    neighbor_net = np.zeros(n)
    ratio_split = bool(getattr(p, "ratio_split", False))
    if edges is not None and not ratio_split:
        from .lateral import exchange_perched
        neighbor_net = exchange_perched(perched, edges, p, k_x, k_y, area_m2)
        perched += neighbor_net

    if ratio_split:
        lateral = np.zeros(n)
    else:
        k_h = drain_k_mm_d(p, k_x, k_y)
        from .canal import farm_drain_factor
        lateral = np.minimum(lateral_mm(perched, p, k_h) * farm_drain_factor(p),
                             np.maximum(perched, 0.0))
        perched -= lateral
    deep = np.minimum(p.deep_percolation_mm_d, np.maximum(perched, 0.0))
    perched -= deep

    leftover = rain + irrigation - (discharge + overflow + spill) - taken["et_mm"]

    return {
        "et_potential_mm": et_potential,
        "et_mm": taken["et_mm"],
        "stress_mm": taken["stress_mm"],
        "et_pond_mm": taken["et_pond_mm"],
        "et_surface_mm": taken["et_surface_mm"],
        "et_root_mm": taken["et_root_mm"],
        "transpiration_mm": taken["transpiration_mm"],
        "evaporation_mm": taken["evaporation_mm"],
        "ks": ks,
        "kr": kr,
        "ke": ke,
        "kcb": kcb,
        "ponded": taken["ponded"],
        "discharge_mm": discharge,
        "overflow_mm": overflow,
        "runoff_mm": overflow,  # bund overflow = surface runoff (not a second mass path)
        "spill_mm": spill,
        "drainage_mm": discharge + overflow + spill,
        "infiltration_mm": infiltration,
        "leftover_mm": leftover,
        "et_groundwater_mm": taken["need_mm"],
        "need_mm": taken["need_mm"],
        "canal_mm": np.zeros(n),
        "neighbor_mm": neighbor_net,
        "lateral_mm": lateral,
        "deep_mm": deep,
        "stay_mm": deep,
        "assigned_lateral": np.zeros(n),
    }

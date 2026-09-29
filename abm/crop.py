# -*- coding: utf-8 -*-
"""FAO-56 rice growth cycle and yield response for the ABM field.

References (crop water / stress):
  * Allen et al. (1998) FAO Irrigation and Drainage Paper 56 — Kc curve,
    transplanted paddy rice Kc_ini≈1.05, Kc_mid≈1.20, Kc_end≈0.90–0.60.
  * Doorenbos & Kassam (1979) FAO 33 — yield response
        (1 − Ya/Ym) = Ky (1 − ETa/ETc), Ky_rice ≈ 1.0–1.15.
  * Conceptual twin (Craciun): crop coefficient links growth stage to demand;
    farmers maintain ideal ponding; insufficient applied irrigation hurts growth.

Season length matches Okura Imbanuma (131 d, 17 Apr – 25 Aug). Mid-summer
drainage window follows Japanese practice (shallow / drained mid-season).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# FAO-56 Table 12 / text: transplanted rice under flooded culture.
KC_INI = 1.05
KC_MID = 1.20
KC_END = 0.90  # wet harvest; use 0.60 if dry-down to harvest

# Lengths (days) summing to Okura SEASON_DAYS = 131.
L_INI = 25
L_DEV = 30
L_MID = 50
L_LATE = 26  # 25+30+50+26 = 131

# FAO 33 indicative Ky for paddy rice (seasonal aggregate ≈ 1.1).
KY_RICE = 1.10

# Target ponding by stage (mm) — farmer "ideal height" from conceptual model.
TARGET_DEPTH = {
    "initial": 80.0,       # establishment flood
    "development": 60.0,
    "mid": 50.0,
    "midsummer_drain": 10.0,  # mid-season drainage practice
    "late": 30.0,
    "harvest_drain": 0.0,
}

# Mid-summer drainage window (day index within season).
MIDSUMMER_START = 70
MIDSUMMER_DAYS = 14
HARVEST_DRAIN_DAY = 120


@dataclass
class GrowthDay:
    stage: str
    kc: float
    target_depth_mm: float
    etc_mm: float  # Kc * ET0 (potential)


def stage_name(day: int) -> str:
    if day < L_INI:
        return "initial"
    if day < L_INI + L_DEV:
        return "development"
    if day < L_INI + L_DEV + L_MID:
        return "mid"
    return "late"


def kc_of_day(day: int) -> float:
    """Piecewise-linear FAO-56 single-crop-coefficient curve."""
    d = max(int(day), 0)
    if d < L_INI:
        return KC_INI
    if d < L_INI + L_DEV:
        # Linear climb ini → mid
        t = (d - L_INI) / max(L_DEV, 1)
        return KC_INI + t * (KC_MID - KC_INI)
    if d < L_INI + L_DEV + L_MID:
        return KC_MID
    # Late: mid → end
    t = (d - (L_INI + L_DEV + L_MID)) / max(L_LATE, 1)
    t = min(max(t, 0.0), 1.0)
    return KC_MID + t * (KC_END - KC_MID)


def target_depth_mm(day: int) -> float:
    if day >= HARVEST_DRAIN_DAY:
        return TARGET_DEPTH["harvest_drain"]
    if MIDSUMMER_START <= day < MIDSUMMER_START + MIDSUMMER_DAYS:
        return TARGET_DEPTH["midsummer_drain"]
    return TARGET_DEPTH[stage_name(day)]


# Canopy cover fc for cyclic.physics' FAO-56 dual Kc (Kcb = Kcb_bare + (Kcb_full −
# Kcb_bare)·fc). Scenario seasons have no satellite NDVI, so fc follows the same stage
# lengths as the Kc curve: sparse transplants, closing canopy, full, senescing.
FC_INI = 0.15
FC_MID = 0.90
FC_END = 0.70


def canopy_cover(day: int) -> float:
    d = max(int(day), 0)
    if d < L_INI:
        return FC_INI
    if d < L_INI + L_DEV:
        return FC_INI + (d - L_INI) / max(L_DEV, 1) * (FC_MID - FC_INI)
    if d < L_INI + L_DEV + L_MID:
        return FC_MID
    t = min((d - (L_INI + L_DEV + L_MID)) / max(L_LATE, 1), 1.0)
    return FC_MID + t * (FC_END - FC_MID)


def growth_day(day: int, et0_mm: float) -> GrowthDay:
    kc = kc_of_day(day)
    return GrowthDay(
        stage=stage_name(day),
        kc=kc,
        target_depth_mm=target_depth_mm(day),
        etc_mm=kc * float(et0_mm),
    )


def stress_ks(ponding_mm: np.ndarray, target_mm: float, half_stress_mm: float = 25.0) -> np.ndarray:
    """Ponding-based stress coefficient Ks ∈ (0, 1].

    Full demand when ponding ≥ target. Smoothly falls as the field dries below
    target (proxy for FAO-56 Ks when flooded culture is the standard).
    """
    pond = np.asarray(ponding_mm, dtype=float)
    deficit = np.maximum(float(target_mm) - pond, 0.0)
    return 1.0 / (1.0 + deficit / max(half_stress_mm, 1e-6))


def actual_et_mm(etc_mm: float, ks: np.ndarray) -> np.ndarray:
    return float(etc_mm) * np.asarray(ks, dtype=float)


def yield_increment(eta_mm: np.ndarray, etc_mm: float, ky: float = KY_RICE) -> np.ndarray:
    """Daily relative-yield contribution (FAO 33 Ky form, normalised by ETc).

    Accumulating ``yield_increment`` over the season and dividing by season
    length gives Ya/Ym ≈ 1 − Ky (1 − ΣETa/ΣETc) when ETc > 0.
    """
    etc = max(float(etc_mm), 1e-9)
    eta = np.asarray(eta_mm, dtype=float)
    # Relative ET attainment today; Ky scales the shortfall into yield loss.
    relative = np.clip(eta / etc, 0.0, 1.0)
    return 1.0 - ky * (1.0 - relative)


def season_relative_yield(eta_sum: np.ndarray, etc_sum: float, ky: float = KY_RICE) -> np.ndarray:
    """Season Ya/Ym from cumulative ETa vs ETc (FAO 33)."""
    etc = max(float(etc_sum), 1e-9)
    eta = np.asarray(eta_sum, dtype=float)
    return np.clip(1.0 - ky * (1.0 - eta / etc), 0.0, 1.0)


def kc_series(days: int) -> np.ndarray:
    return np.array([kc_of_day(d) for d in range(days)], dtype=float)


def target_series(days: int) -> np.ndarray:
    return np.array([target_depth_mm(d) for d in range(days)], dtype=float)

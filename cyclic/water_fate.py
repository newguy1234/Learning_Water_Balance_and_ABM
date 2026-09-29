"""Given IR/D: cover ET, then **inferred underground flow ratios** take over.

Contribution path (one day, every paddy):

  1. Apply given IR and D (Fig. 8 / Table 4 — known forcing, not novelty).
  2. Cover ET from pond + soil (FAO-56).
  3. Penetration ``plow_pan_k_mm_d``: pond → soil (enables underground leftover).
  4. Neighbour hops equalise leftover head — wetter plots refill ET-short neighbours.
  5. Per-point capture (Kx+Ky) → same-day **LFV**; district mean = Table 5 LFV/LfA.
  6. Stay (Kz + ground) → deep / dWB.

No artificial capture floor: Table 5 sets the pin; hops + penetration satisfy ET.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from . import okura
from .physics import percolation_mm, Physics


def _align2d(arr, n_days, n, fill=0.0):
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        if a.size == n_days:
            return np.broadcast_to(a[:, None], (n_days, n)).copy()
        if a.size == n:
            return np.broadcast_to(a[None, :], (n_days, n)).copy()
        return np.full((n_days, n), float(fill))
    out = np.zeros((n_days, n), dtype=float)
    out[:, :] = fill
    out[: min(n_days, a.shape[0]), : min(n, a.shape[1])] = a[:n_days, :n]
    return out


def estimate_infiltrated_leftover_m3(
        plow_pan_k_mm_d,
        irrigation_mm,
        discharge_mm,
        rain_mm,
        et_mm,
        area_m2,
        cyclic,
        pond_ref_mm=30.0,
        outlet_loss=0.13,
        bund_mm=100.0,
):
    """Fast season estimate of cyclic infiltrated leftover volume (m³).

    Day loop (no neighbour hops): pond ← IR+R; take ET; apply D; infiltrate with k;
    leftover L = IR+R−D−ET; infiltrated leftover = min(I, max(L,0)) on cyclic cells.
    """
    ir = np.asarray(irrigation_mm, dtype=float)
    dr = np.asarray(discharge_mm, dtype=float)
    n_days, n = ir.shape
    rain = _align2d(rain_mm, n_days, n, 0.0)
    et = _align2d(et_mm, n_days, n, 0.0)
    area = np.asarray(area_m2, dtype=float).ravel()
    cyclic = np.asarray(cyclic, dtype=bool).ravel()
    p = Physics(plow_pan_k_mm_d=float(plow_pan_k_mm_d),
                pond_infiltrate_ref_mm=float(pond_ref_mm),
                outlet_loss_per_day=float(outlet_loss),
                bund_mm=float(bund_mm))
    water = np.full(n, 30.0)  # puddled start
    total_m3 = 0.0
    et_got = 0.0
    et_want = 0.0
    for d in range(n_days):
        water = water + rain[d] + ir[d]
        # ET cover from pond first (simple), then residual need.
        take = np.minimum(np.maximum(et[d], 0.0), np.maximum(water, 0.0))
        water = water - take
        et_got += float((take * area).sum() / 1000.0)
        et_want += float((np.maximum(et[d], 0.0) * area).sum() / 1000.0)
        disc = np.minimum(np.maximum(dr[d], 0.0), water)
        water = water - disc
        overflow = np.maximum(water - p.bund_mm, 0.0)
        water = water - overflow
        spill = water * p.outlet_loss_per_day
        water = water - spill
        infil = np.minimum(percolation_mm(water, p), water)
        water = water - infil
        leftover = rain[d] + ir[d] - (disc + overflow + spill) - take
        inf_left = np.minimum(infil, np.maximum(leftover, 0.0))
        inf_left = np.where(cyclic, inf_left, 0.0)
        total_m3 += float((inf_left * area).sum() / 1000.0)
    et_cover = et_got / max(et_want, 1e-9)
    return total_m3, float(np.clip(et_cover, 0.0, 1.0))


def lfv_infiltration_target_m3(year):
    """Minimum cyclic infiltrated leftover (m³) so Table 5 LFV can be delivered.

    With capture = LFV/LfA, need infiltrated leftover ≥ LfA (= LFV/capture) so the
    pin can reach LFV_pump. Using LfA_North + LfA_South is the paper identity.
    """
    t5 = okura.TABLE5_1000M3[int(year)]
    return (float(t5["LfA_North"]) + float(t5["LfA_South"])) * 1000.0


def solve_penetration_k(
        year,
        irrigation_mm,
        discharge_mm,
        rain_mm,
        et_mm,
        area_m2,
        cyclic,
        k_lo=4.0,
        k_hi=80.0,
        iters=14,
        pond_ref_mm=30.0,
        et_cover_floor=0.85,
):
    """Binary-search plow_pan_k so infiltrated leftover ≥ LfA and ET cover is high.

    Returns namespace(plow_pan_k_mm_d, infiltrated_m3, target_m3, et_cover, ok).
    """
    year = int(year)
    target = lfv_infiltration_target_m3(year)
    lo, hi = float(k_lo), float(k_hi)
    best = None
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        inf_m3, et_c = estimate_infiltrated_leftover_m3(
            mid, irrigation_mm, discharge_mm, rain_mm, et_mm, area_m2, cyclic,
            pond_ref_mm=pond_ref_mm)
        row = SimpleNamespace(
            plow_pan_k_mm_d=float(mid),
            infiltrated_m3=float(inf_m3),
            target_m3=float(target),
            et_cover=float(et_c),
            lfv_room=float(inf_m3) / max(float(target), 1.0),
            ok=bool(inf_m3 >= target * 0.98 and et_c >= et_cover_floor),
        )
        best = row
        # Need more infiltration → higher k; enough → try lower k (prefer modest pan).
        if inf_m3 < target or et_c < et_cover_floor:
            lo = mid
        else:
            hi = mid
    # Final evaluate at hi (sufficient side) if best not ok.
    if best is not None and not best.ok:
        mid = float(hi)
        inf_m3, et_c = estimate_infiltrated_leftover_m3(
            mid, irrigation_mm, discharge_mm, rain_mm, et_mm, area_m2, cyclic,
            pond_ref_mm=pond_ref_mm)
        best = SimpleNamespace(
            plow_pan_k_mm_d=float(mid),
            infiltrated_m3=float(inf_m3),
            target_m3=float(target),
            et_cover=float(et_c),
            lfv_room=float(inf_m3) / max(float(target), 1.0),
            ok=bool(inf_m3 >= target * 0.95),
        )
    return best


def apply_water_fate_physics(model, penetration_k=None, deep_mm_d=None):
    """Turn on ratio-split canal path: lateral hops + LFV + deep soil stays."""
    changes = {
        "ratio_split": True,
        "canal_routing": True,
    }
    if penetration_k is not None:
        changes["plow_pan_k_mm_d"] = float(penetration_k)
    if deep_mm_d is not None:
        changes["deep_percolation_mm_d"] = float(deep_mm_d)
    # Prefer a bit more deep flow when stay share is large (dry-year dWB).
    model.physics = model.physics.replace(**changes)
    return model.physics


def configure_given_ir_d_water_fate(model, irrigation_mm, discharge_mm, year=None):
    """End-to-end: given IR/D arrays → solve penetration → enable underground fate.

    Call after the schedule fills ``irrigation_mm`` / ``discharge_mm`` (n_days × n).
    """
    year = int(year if year is not None else model.year)
    rain = np.asarray(model.inputs.rain_mm, dtype=float)
    if rain.ndim == 2:
        rain_1d = rain[:model.n_days].mean(axis=1) if rain.shape[1] > 1 else rain[:model.n_days, 0]
    else:
        rain_1d = rain[:model.n_days]
    et = np.asarray(model.inputs.et_target_mm, dtype=float)
    if et.ndim == 1:
        et = np.broadcast_to(et[:, None], (model.n_days, model.n))
    solved = solve_penetration_k(
        year,
        irrigation_mm[:model.n_days],
        discharge_mm[:model.n_days],
        rain_1d,
        et[:model.n_days],
        model.area,
        model.inputs.cyclic,
        pond_ref_mm=float(model.physics.pond_infiltrate_ref_mm),
    )
    # Deep percolation: scale mildly with stay fraction (1 - capture).
    from .identify import construct_ratios
    pin = construct_ratios(year)
    stay = float(pin["kz_ratio"] + pin["ground_ratio"])
    deep = max(0.5, min(4.0, 0.8 + 3.0 * stay))
    apply_water_fate_physics(
        model,
        penetration_k=solved.plow_pan_k_mm_d,
        deep_mm_d=deep,
    )
    model.penetration_report = solved
    model.penetration_report.deep_percolation_mm_d = float(deep)
    return solved

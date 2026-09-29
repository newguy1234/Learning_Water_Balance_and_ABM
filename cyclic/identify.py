"""Identification of **inferred underground flow ratios** (contribution core).

Contribution
    With surface IR/D known (Fig. 8 / Okura pumps), leftover after FAO-56 ET is
    Okura's unobserved Lf residual. We infer per-point shares of that residual:

        pool_capture_share   → same-day LFV (district mean = LFV_pump/LfA)
        kx_ratio, ky_ratio   → neighbours (ET cover; axis not identified)
        kz_ratio, ground     → stay / deep (dWB)

    Canonical API: ``cyclic.infer_underground_ratios``, ``cyclic.underground_ratio_holdout``.
    Surface IR timing is forcing, not novelty. Kx vs Ky axis is not identified.

`run_finder` freezes irrigation and discharge from the best currently registered
forecaster. Simulated ET comes from FAO-56 dual Kc and local water. Satellite ET,
irrigation, discharge, and Table 4 lateral (Lf) volumes are all in the loss; the
satellite map is not imposed as the flux. The unknown is how leftover IR+R−D−ET
is partitioned:

    pool_capture_share  same-day LFV (Table 5 LFV_pump/LfA; nearer-canal weighted)
    kx_ratio            leftover → neighbours, E-W half of the neighbour share
    ky_ratio            leftover → neighbours, N-S half of the same share
    kz_ratio            remains in the topsoil
    ground_ratio        remains in the deeper soil
    farm_to_main        farm-drain discharge that reaches the main canal

Each satellite cell has its own simplex. LFV is nearer-canal weighted with
area-weighted mean pinned to Table 5. Of the non-LFV remainder, neighbour_of_remain
goes to Kx+Ky (hops cover ET); the rest stays. Same-day LFV is from infiltrated
leftover, not hop-then-lining-catch.

    from cyclic import infer_underground_ratios, underground_ratio_holdout
    from cyclic.identify import HydrologyProblem, construct_ratios, solve_classical
"""

from __future__ import annotations

import ast
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import okura
from .physics import Physics
from . import data as _data

BLOCKS = okura.BLOCKS
# Leftover simplex + farm-drain → main-canal rate. Conductivity Kx/Ky (mm/d) stays later.
ACTIVE = ("kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio", "farm_to_main")
LATER = ("kx_mm_d", "ky_mm_d")
BOUNDS = {
    "kx_ratio": (0.0, 1.0),
    "ky_ratio": (0.0, 1.0),
    "kz_ratio": (0.0, 1.0),
    "ground_ratio": (0.0, 1.0),
    "farm_to_main": (0.0, 1.0),
    "lateral_per_discharge": (0.0, 3.0),
    "kx_mm_d": (50.0, 20000.0),
    "ky_mm_d": (50.0, 20000.0),
}
OBJECTIVES = ("et", "irrigation", "discharge", "ir_daily", "d_daily",
              "pond", "lateral", "ld_ratio", "farm_to_main", "lfv", "darcy")
IMITATOR_OBJ = ("et", "irrigation", "discharge", "pond")
RATIO_OBJ = ("lateral", "ld_ratio", "farm_to_main", "lfv", "darcy")
# Prieto et al. (2021, 2022): routing mechanisms were identifiable from streamflow-like
# aggregates; percolation was not. Applied here: lfv/farm_to_main/darcy are routing-like;
# Kz vs ground is the percolation split we do not identify; leftover volume terms are
# algebraic consequences of ET/IR/D, not independent routing observations.
PROCESS_ROLE = {
    "et": "flux",
    "irrigation": "p1",
    "discharge": "p1",
    "ir_daily": "p1",
    "d_daily": "p1",
    "pond": "store",
    "lateral": "algebraic",
    "ld_ratio": "algebraic",
    "farm_to_main": "routing",
    "lfv": "routing",
    "darcy": "routing",
}
ROUTING_COORDINATES = ("kx_ratio", "ky_ratio")   # only the sum is identified
PERCOLATION_COORDINATES = ("kz_ratio", "ground_ratio")  # not identified
# Beven (2006) limit of acceptability, set before the leftover-speed sweep: simulated
# LFV_pump at least this fraction of Table 5. Failing it is structure, not a miss on v.
LFV_ACCEPT_FRAC = 0.5
# `irrigation` and `discharge` are season totals: any daily schedule with the right
# total scores the same. `ir_daily` and `d_daily` are the day-by-day errors. They are
# reported at weight 0 so the fitted loss stays comparable to saved checkpoints; the
# daily numbers are what a forecaster has to win on once measured series are loaded.
REPORTED_ONLY = ("ir_daily", "d_daily")
DEFAULT_WEIGHTS = {
    "et": 1.0, "irrigation": 1.0, "discharge": 1.0, "pond": 0.5,
    "lateral": 1.0, "ld_ratio": 1.0, "farm_to_main": 1.0, "lfv": 1.0, "darcy": 1.0,
    "ir_daily": 0.0, "d_daily": 0.0,
}


def table_lateral_m3(year, block):
    """Table 4 leftover (Lf) for one cyclic block, m³/season."""
    return okura.TABLE4_MM[(block, year)]["lateral"] * okura.cyclic_area_m2(block) / 1000.0


def table_ld_ratio(year, block):
    row = okura.TABLE4_MM[(block, year)]
    d = max(float(row["drainage"]), 1e-9)
    return float(row["lateral"]) / d


def table_main_d_m3(year):
    row = okura.TABLE5_1000M3[year]
    return (row["D_North"] + row["D_South"]) * 1000.0


def table_lfv_m3(year):
    return okura.TABLE5_1000M3[year]["LFV_pump"] * 1000.0


def construct_ratios(year=okura.COMPARISON_YEAR, neighbour_of_remain=None):
    """Closed-form leftover shares from Okura Tables 4–5.

    Simplex (sums to 1 on cyclic leftover):
        pool_capture_share  = LFV_pump / LfA   → same-day LFV (Table 5)
        kx_ratio + ky_ratio                   → to neighbours (ET cover)
        kz_ratio + ground_ratio               → stay / deep (dWB)

    Of the non-LFV remainder, ``neighbour_of_remain`` (default Physics 0.5) goes to
    Kx+Ky (half each); the rest is stay split by TEW/(TEW+TAW).
    """
    area = {block: okura.cyclic_area_m2(block) for block in BLOCKS}
    lfa = (okura.TABLE5_1000M3[year]["LfA_North"] + okura.TABLE5_1000M3[year]["LfA_South"]) * 1000.0
    lfv = table_lfv_m3(year)
    capture = float(np.clip(lfv / max(lfa, 1.0), 0.0, 1.0))
    p = Physics()
    nbr_frac = float(np.clip(
        p.neighbour_of_remain if neighbour_of_remain is None else neighbour_of_remain,
        0.0, 1.0))
    remain = 1.0 - capture
    neighbour = remain * nbr_frac
    stay = remain - neighbour
    topsoil_frac = float(p.total_evaporable_mm /
                         max(p.total_evaporable_mm + p.total_available_mm, 1e-9))
    d4 = sum(okura.TABLE4_MM[(block, year)]["drainage"] * area[block] for block in BLOCKS) / 1000.0
    d5 = table_main_d_m3(year)
    return {
        "kx_ratio": 0.5 * neighbour,
        "ky_ratio": 0.5 * neighbour,
        "kz_ratio": stay * topsoil_frac,
        "ground_ratio": stay * (1.0 - topsoil_frac),
        "farm_to_main": float(np.clip(d5 / max(d4, 1e-9), 0.0, 1.0)),
        "pool_capture_share": capture,
        "neighbour_of_remain": nbr_frac,
        "ratio_split": True,
        "force_satellite_et": False,
    }


def attach_pixel_shares(model, constructed=None, use_paper_plots=True):
    """Write a leftover simplex onto every satellite cell (or paddy if no grid).

    When ``use_paper_plots`` is True, North/South intensity follows paper Table 4
    Lf mm (Fig. 6 season bars) so plot-block leftover matches Okura's figures.
    """
    from .canal import pixel_ratio_fields

    constructed = constructed or construct_ratios(model.year)
    capture = float(constructed["pool_capture_share"])
    farm = float(constructed["farm_to_main"])
    p = model.physics
    topsoil = float(p.total_evaporable_mm / max(p.total_evaporable_mm + p.total_available_mm, 1e-9))
    length = float(p.canal_length_m)
    nbr = float(constructed.get(
        "neighbour_of_remain", getattr(p, "neighbour_of_remain", 0.5)))
    intensity = None
    if use_paper_plots:
        try:
            from .paper_plots import block_leftover_intensity
            intensity = block_leftover_intensity(model.year)
        except Exception:
            intensity = None
    if model.cells is not None:
        grid = model.cells
        cyclic = np.asarray(model.inputs.cyclic[grid.owner], dtype=bool)
        block = np.asarray(model.inputs.block[grid.owner])
        model.pixel_shares = pixel_ratio_fields(
            len(grid.owner), grid.canal_distance_m, grid.edges, cyclic,
            capture, farm, topsoil, length,
            area_m2=model.area[grid.owner] * grid.weights,
            block=block, block_intensity=intensity,
            hard_capture=False, neighbour_of_remain=nbr)
    else:
        model.pixel_shares = pixel_ratio_fields(
            model.n, model.inputs.canal_distance_m, model.paddy_edges, model.inputs.cyclic,
            capture, farm, topsoil, length, area_m2=model.area,
            block=model.inputs.block, block_intensity=intensity,
            hard_capture=False, neighbour_of_remain=nbr)
    return model.pixel_shares


def best_ratio_path(year, span="season", folder=None):
    import os
    folder = folder or os.path.join(_data.PROJECT, "outputs", "ratios")
    return os.path.join(folder, f"best_{int(year)}_{span}.npz")


def load_best_ratios(year, span="season", folder=None):
    """Last saved best district vector and pixel shares, or None."""
    import os

    path = best_ratio_path(year, span, folder)
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=True) as data:
        params = {key: float(data[key]) for key in ACTIVE if key in data.files}
        params["pool_capture_share"] = float(data["pool_capture_share"]) if "pool_capture_share" in data.files else (
            params.get("kx_ratio", 0.0) + params.get("ky_ratio", 0.0))
        params["ratio_split"] = True
        params["force_satellite_et"] = False
        fields = None
        extra = ("pool_capture_share", "kx_signed", "ky_signed", "kx_sign", "ky_sign")
        if "kx_ratio_field" in data.files:
            fields = {name: np.asarray(data[name + "_field"], dtype=float)
                      for name in (*ACTIVE, *extra)
                      if name + "_field" in data.files}
        return SimpleNamespace(
            params=params, fields=fields,
            loss=float(data["loss"]) if "loss" in data.files else None,
            path=path,
        )


QUANTUM_FIELD_KEYS = (
    "kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio", "farm_to_main",
    "pool_capture_share", "kx_signed", "ky_signed", "kx_sign", "ky_sign",
    "assigned_lateral",
)


def quantum_ratio_path(year, folder=None):
    import os
    folder = folder or os.path.join(_data.PROJECT, "outputs", "ratios")
    return os.path.join(folder, f"quantum_{int(year)}.npz")


def overlay_pixel_maps(maps, pixel_override):
    """Put per-paddy leftover shares onto an existing field-share map."""
    if pixel_override is None:
        return maps
    maps.fields.update({k: v for k, v in pixel_override.items() if isinstance(v, np.ndarray)})
    maps.volumes.update({k: np.asarray(pixel_override[k])
                         for k, _, _, _ in SHARE_LABELS if k in pixel_override})
    if "kx_signed" in pixel_override:
        maps.volumes["kx_ratio"] = np.asarray(pixel_override["kx_signed"])
    if "ky_signed" in pixel_override:
        maps.volumes["ky_ratio"] = np.asarray(pixel_override["ky_signed"])
    return maps


def save_quantum_ratios(year, quantum, maps=None, params=None, loss=None, signed=None,
                        split=None, span="tables", folder=None):
    """Always overwrite the last QAOA / IBM assignment so the UI can draw it."""
    import os

    folder = folder or os.path.join(_data.PROJECT, "outputs", "ratios")
    os.makedirs(folder, exist_ok=True)
    path = quantum_ratio_path(year, folder)
    params = params or getattr(quantum, "physics", {})
    payload = {key: float(params[key]) for key in ACTIVE if key in params}
    payload["pool_capture_share"] = float(params.get(
        "pool_capture_share", params.get("kx_ratio", 0.0) + params.get("ky_ratio", 0.0)))
    payload["loss"] = np.float64(loss if loss is not None else getattr(quantum, "energy", np.nan))
    payload["energy"] = np.float64(getattr(quantum, "energy", np.nan))
    payload["n_var"] = np.int32(getattr(quantum, "n_var", 0))
    payload["n_nodes"] = np.int32(getattr(quantum, "n_nodes", 0))
    payload["year"] = np.int32(year)
    payload["backend"] = np.array(str(getattr(quantum, "backend", "")))
    payload["span"] = np.array(str(span))
    fields = None if maps is None else maps.fields
    if fields is None and getattr(quantum, "shares", None) is not None:
        fields = {name: quantum.shares[name] for name in QUANTUM_FIELD_KEYS
                  if name in quantum.shares and isinstance(quantum.shares[name], np.ndarray)}
    if fields:
        for name, values in fields.items():
            if isinstance(values, np.ndarray):
                payload[name + "_field"] = np.asarray(values, dtype=np.float32)
    solved = getattr(quantum, "solved", None)
    if solved is not None and getattr(solved, "bitstring", None) is not None:
        payload["bitstring"] = np.asarray(solved.bitstring, dtype=np.float32)
    if signed:
        payload["signed_json"] = np.array(json.dumps({k: float(v) for k, v in signed.items()}))
    if split is not None:
        payload["imitator_pct"] = np.float64(split.imitator_pct)
        payload["ratios_pct"] = np.float64(split.ratios_pct)
    np.savez_compressed(path, **payload)
    return path


def load_quantum_ratios(year, folder=None):
    """Last IBM / QAOA leftover assignment, or None."""
    import os

    path = quantum_ratio_path(year, folder)
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=True) as data:
        params = {key: float(data[key]) for key in ACTIVE if key in data.files}
        params["pool_capture_share"] = float(data["pool_capture_share"]) if "pool_capture_share" in data.files else (
            params.get("kx_ratio", 0.0) + params.get("ky_ratio", 0.0))
        params["ratio_split"] = True
        params["force_satellite_et"] = False
        extra = ("pool_capture_share", "kx_signed", "ky_signed", "kx_sign", "ky_sign",
                 "assigned_lateral")
        fields = {name: np.asarray(data[name + "_field"], dtype=float)
                  for name in (*ACTIVE, *extra)
                  if name + "_field" in data.files} or None
        signed = {}
        if "signed_json" in data.files:
            raw = str(np.asarray(data["signed_json"]).item() if np.asarray(data["signed_json"]).shape == ()
                      else data["signed_json"])
            try:
                signed = json.loads(raw)
            except Exception:
                try:
                    signed = ast.literal_eval(raw)
                except Exception:
                    signed = {}
        split = None
        if "imitator_pct" in data.files:
            split = SimpleNamespace(
                imitator_pct=float(data["imitator_pct"]),
                ratios_pct=float(data["ratios_pct"]),
                imitator_signed=0.0, ratios_signed=0.0, signed_ratio=0.0, total=1.0,
            )
        quantum = SimpleNamespace(
            backend=str(data["backend"]) if "backend" in data.files else "",
            energy=float(data["energy"]) if "energy" in data.files else float("nan"),
            n_var=int(data["n_var"]) if "n_var" in data.files else 0,
            n_nodes=int(data["n_nodes"]) if "n_nodes" in data.files else 0,
        )
        return SimpleNamespace(
            params=params, fields=fields, signed=signed, split=split, quantum=quantum,
            loss=float(data["loss"]) if "loss" in data.files else None,
            span=str(data["span"]) if "span" in data.files else "tables",
            path=path,
        )


def report_from_quantum(year, folder=None):
    """Rebuild a ratio-finder report from the last saved IBM / QAOA file."""
    loaded = load_quantum_ratios(year, folder)
    if loaded is None:
        return None
    maps = overlay_pixel_maps(field_share_maps(year, loaded.params), loaded.fields)
    obj, signed = table_closure_objectives(year)
    signed = loaded.signed or signed
    split = loaded.split or loss_split(obj, signed=signed)
    from .canal import mean_pixel_shares
    means = mean_pixel_shares(maps.fields, maps.area, maps.cyclic)
    params = {k: loaded.params.get(k, means.get(k, 0.0)) for k in ACTIVE}
    params["kx_signed"] = means.get("kx_signed", loaded.params.get("kx_signed", 0.0))
    params["ky_signed"] = means.get("ky_signed", loaded.params.get("ky_signed", 0.0))
    params["pool_capture_share"] = loaded.params.get(
        "pool_capture_share", params.get("kx_ratio", 0.0) + params.get("ky_ratio", 0.0))
    return SimpleNamespace(
        params=params, objectives=obj, signed=signed, split=split,
        loss=loaded.loss if loaded.loss is not None else split.total,
        maps=maps, seconds=0.0, span=loaded.span, days=0, year=int(year),
        saved=loaded.path, quantum=loaded.quantum, from_disk=True,
    )


def save_best_ratios(year, span, params, fields=None, loss=None, signed=None, split=None,
                     folder=None, force=False):
    """Keep this run if it beats the file on disk (or if none exists).

    force=True writes even if the stored loss is smaller — needed after a physics
    change, when the old score is not comparable.
    """
    import os

    previous = load_best_ratios(year, span, folder)
    if (not force and previous is not None and previous.loss is not None
            and loss is not None and loss > previous.loss + 1e-12):
        return previous.path, False
    folder = folder or os.path.join(_data.PROJECT, "outputs", "ratios")
    os.makedirs(folder, exist_ok=True)
    path = best_ratio_path(year, span, folder)
    payload = {key: float(params[key]) for key in ACTIVE if key in params}
    payload["pool_capture_share"] = float(params.get(
        "pool_capture_share", params.get("kx_ratio", 0.0) + params.get("ky_ratio", 0.0)))
    payload["loss"] = np.float64(loss if loss is not None else np.nan)
    payload["year"] = np.int32(year)
    if fields:
        for name, values in fields.items():
            payload[name + "_field"] = np.asarray(values, dtype=np.float32)
    if signed:
        payload["signed_json"] = np.array(json.dumps({k: float(v) for k, v in signed.items()}))
    if split is not None:
        payload["imitator_pct"] = np.float64(split.imitator_pct)
        payload["ratios_pct"] = np.float64(split.ratios_pct)
    np.savez_compressed(path, **payload)
    return path, True


def spread_field_action(irrigation, discharge, area_m2, block=None, cyclic=None):
    """Broadcast a field or per-block action onto every paddy.

    `irrigation` / `discharge` may be:
        a scalar            the same mm on every plot
        length-2 array      North, South (cyclic.okura.BLOCKS order)
        dict {block: mm}    named blocks
        length-n array      already per paddy

    Non-cyclic plots get 0 unless the value is already per-paddy. Totals then
    scale with area, so one field decision is the district's aggregate action.
    """
    n = np.asarray(area_m2).size
    blocks = None if block is None else np.asarray(block)
    mask = np.ones(n, dtype=bool) if cyclic is None else np.asarray(cyclic, dtype=bool)

    def expand(value):
        if isinstance(value, dict):
            out = np.zeros(n)
            for name, mm in value.items():
                out[(blocks == name) & mask] = float(mm)
            return out
        array = np.asarray(value, dtype=float)
        if array.ndim == 0 or array.size == 1:
            out = np.zeros(n)
            out[mask] = float(array)
            return out
        if array.size == n:
            return array.reshape(n)
        if array.size == len(BLOCKS) and blocks is not None:
            out = np.zeros(n)
            for i, name in enumerate(BLOCKS):
                out[(blocks == name) & mask] = float(array.flat[i])
            return out
        raise ValueError("irrigation/discharge must be a scalar, per-block, or per-paddy")

    return expand(irrigation), expand(discharge)


def match_block_volumes(volume_m3, area_m2, block, cyclic=None, weights=None):
    """mm on each cyclic paddy so that block totals equal `volume_m3` (scalar, length-2, or dict).

    `weights` are volume shares (typically area × a per-plot score). Default is area,
    i.e. the same mm on every cyclic plot in the block.
    """
    n = np.asarray(area_m2).size
    area = np.asarray(area_m2, dtype=float)
    blocks = np.asarray(block)
    mask = np.ones(n, dtype=bool) if cyclic is None else np.asarray(cyclic, dtype=bool)
    if isinstance(volume_m3, dict):
        wanted = volume_m3
    else:
        array = np.broadcast_to(np.asarray(volume_m3, dtype=float), (len(BLOCKS),))
        wanted = {name: float(array[i]) for i, name in enumerate(BLOCKS)}
    scores = area.copy() if weights is None else np.asarray(weights, dtype=float)
    scores = np.where(mask, np.maximum(scores, 0.0), 0.0)
    mm_out = np.zeros(n)
    for name in BLOCKS:
        sel = (blocks == name) & mask
        mass = float(scores[sel].sum())
        want = float(wanted.get(name, 0.0))
        if mass <= 0.0 or want == 0.0:
            continue
        mm_out[sel] = want * 1000.0 * scores[sel] / (np.maximum(area[sel], 1e-9) * mass)
    return mm_out


def match_et_cover_then_free(volume_m3, area_m2, block, cyclic, cover_mm, free_weights):
    """Block IR totals fixed: cover ET first, then place leftover however `free_weights` say.

    1. Give each paddy ``cover_mm`` (ET − rain − pond, clipped) as far as the day's
       block budget allows (scaled down together if cover > budget).
    2. Any remaining m³ is spread by ``free_weights`` (neural scores, etc.).
    """
    n = np.asarray(area_m2).size
    area = np.asarray(area_m2, dtype=float)
    blocks = np.asarray(block)
    mask = np.asarray(cyclic, dtype=bool)
    cover = np.maximum(np.asarray(cover_mm, dtype=float), 0.0)
    free_w = np.maximum(np.asarray(free_weights, dtype=float), 0.0)
    if isinstance(volume_m3, dict):
        wanted = volume_m3
    else:
        array = np.broadcast_to(np.asarray(volume_m3, dtype=float), (len(BLOCKS),))
        wanted = {name: float(array[i]) for i, name in enumerate(BLOCKS)}
    mm_out = np.zeros(n)
    for name in BLOCKS:
        sel = (blocks == name) & mask
        if not np.any(sel):
            continue
        want = float(wanted.get(name, 0.0))
        if want <= 0.0:
            continue
        a = area[sel]
        cover_m3 = float((cover[sel] * a).sum() / 1000.0)
        if cover_m3 <= want:
            scale = 1.0
            leftover = want - cover_m3
            mm_out[sel] = cover[sel]
        else:
            scale = want / max(cover_m3, 1e-12)
            leftover = 0.0
            mm_out[sel] = cover[sel] * scale
        if leftover > 1e-12:
            w = free_w[sel] * a
            mass = float(w.sum())
            if mass <= 1e-12:
                w = a
                mass = float(w.sum())
            mm_out[sel] = mm_out[sel] + leftover * 1000.0 * w / (np.maximum(a, 1e-9) * mass)
    return mm_out


def leftover_weighted_stay(shares, leftover_mm, area_m2, cyclic, stay_target, topsoil_frac):
    """Rescale stay so leftover-weighted mean matches Table 5 dWB share.

    Keeps ``pool_capture_share`` (LFV) fixed; puts the rest into Kx+Ky (neighbours)
    vs Kz+ground (stay).
    """
    stay = np.asarray(shares["kz_ratio"], dtype=float) + np.asarray(shares["ground_ratio"], dtype=float)
    lfv = np.asarray(shares.get("pool_capture_share", 0.0), dtype=float)
    area = np.asarray(area_m2, dtype=float)
    mask = np.asarray(cyclic, dtype=bool)
    L = np.asarray(leftover_mm, dtype=float)
    if L.ndim == 2:
        mass = np.maximum(L, 0.0).sum(axis=0) * area
    else:
        mass = np.maximum(L, 0.0) * area
    mass = np.where(mask, mass, 0.0)
    total = float(mass.sum()) or 1.0
    current = float((stay * mass).sum() / total)
    target = float(np.clip(stay_target, 0.0, 1.0))
    if current > 1e-12:
        stay = np.clip(stay * (target / current), 0.0, 1.0)
        current = float((stay * mass).sum() / total)
        if current > 1e-12:
            stay = np.clip(stay * (target / current), 0.0, 1.0)
    else:
        stay = np.where(mask, target, 0.0)
    stay = np.where(mask, stay, 0.0)
    lfv = np.where(mask, np.clip(lfv, 0.0, 1.0), 0.0)
    # LFV pin first; stay cannot eat the LFV share.
    stay = np.minimum(stay, np.maximum(1.0 - lfv, 0.0))
    neighbour = np.where(mask, np.clip(1.0 - lfv - stay, 0.0, 1.0), 0.0)
    top = float(np.clip(topsoil_frac, 0.0, 1.0))
    shares["kx_ratio"] = neighbour * 0.5
    shares["ky_ratio"] = neighbour * 0.5
    shares["kz_ratio"] = np.where(mask, stay * top, 0.0)
    shares["ground_ratio"] = np.where(mask, stay * (1.0 - top), 0.0)
    shares["pool_capture_share"] = lfv
    if "kx_sign" in shares:
        shares["kx_signed"] = shares["kx_ratio"] * np.asarray(shares["kx_sign"], dtype=float)
        shares["ky_signed"] = shares["ky_ratio"] * np.asarray(shares["ky_sign"], dtype=float)
    shares["stay_target"] = target
    shares["stay_weighted"] = float((stay * mass).sum() / total)
    return shares


def leftover_pinned(irrigation_mm, rain_mm, drainage_mm, et_target_mm):
    """Okura Eq. 3 leftover with satellite ET as the observed flux.

    Does not copy satellite into FAO-56 `et_mm`. Groundwater that must have been
    available for that ET is `pin_groundwater_for_et`.
    """
    irrigation = np.asarray(irrigation_mm, dtype=float)
    rain = np.asarray(rain_mm, dtype=float)
    drainage = np.asarray(drainage_mm, dtype=float)
    et = np.asarray(et_target_mm, dtype=float)
    if rain.ndim < irrigation.ndim:
        rain = rain.reshape(rain.shape + (1,) * (irrigation.ndim - rain.ndim))
    return irrigation + rain - drainage - et


def pin_groundwater_for_et(et_target_mm, et_mm=None, et_potential_mm=None,
                           et_groundwater_mm=None):
    """Extra water that had to be in the root/pond (or taken from groundwater).

    Unstressed FAO-56 is a lower bound: if simulated `et_mm` is below satellite
    ET, the plot was too dry by that gap (capped at potential). Satellite is not
    written as the FAO-56 flux.
    """
    target = np.maximum(np.asarray(et_target_mm, dtype=float), 0.0)
    if et_potential_mm is not None:
        cap = np.maximum(np.asarray(et_potential_mm, dtype=float), 0.0)
        target = np.minimum(target, cap)
    if et_mm is None:
        gap = target
    else:
        gap = np.maximum(target - np.maximum(np.asarray(et_mm, dtype=float), 0.0), 0.0)
    if et_groundwater_mm is None:
        return gap
    already = np.maximum(np.asarray(et_groundwater_mm, dtype=float), 0.0)
    return np.maximum(gap, already)


def table_lf_m3(year):
    return sum(okura.TABLE4_MM[(block, year)]["lateral"] * okura.cyclic_area_m2(block) / 1000.0
               for block in BLOCKS)


def _volume_m3(mm, area_m2, cyclic=None):
    mm = np.asarray(mm, dtype=float)
    area = np.asarray(area_m2, dtype=float)
    if cyclic is not None:
        mm = np.where(np.asarray(cyclic, dtype=bool), mm, 0.0)
    return float((mm * area).sum() / 1000.0)


def pin_capture_to_lfv(leftover_mm, area_m2, cyclic, year, infiltration_mm=None,
                       neighbor_mm=None, et_target_mm=None, et_mm=None,
                       et_potential_mm=None, et_groundwater_mm=None, capture=None):
    """Split ET-pinned leftover into hops, drain capture, and soil stay.

    Recapture is cyclic infiltrated leftover. If `capture` is None it is scaled
    so district volume matches Table 5 LFV_pump as closely as a [0, 1] same-day
    share can; if set, that share is frozen (holdout). Pond leftover cannot be
    LFV. Neighbour hops redistribute leftover heads; they do not consume leftover
    mass. Stay is the leftover remainder (Table 5 dWB).
    """
    leftover = np.asarray(leftover_mm, dtype=float)
    squeeze = leftover.ndim == 1
    if squeeze:
        leftover = leftover[None, ...]
    pile = np.maximum(leftover, 0.0)
    cyclic = np.asarray(cyclic, dtype=bool)
    area = np.asarray(area_m2, dtype=float)
    if infiltration_mm is None:
        infiltrated = pile.copy()
        infiltration_limited = False
    else:
        inf = np.maximum(np.asarray(infiltration_mm, dtype=float), 0.0)
        if inf.ndim == 1:
            inf = inf[None, ...]
        infiltrated = np.minimum(inf, pile)
        infiltration_limited = True
    recapturable = np.where(cyclic, infiltrated, 0.0)
    recapturable_m3 = _volume_m3(recapturable, area)
    leftover_m3 = _volume_m3(pile, area, cyclic)
    cyclic_area = float(np.asarray(area_m2, dtype=float)[cyclic].sum()) or 1.0
    lfv_target = table_lfv_m3(year)
    lf_target = table_lf_m3(year)
    dwb_target = okura.TABLE5_1000M3[year]["dWB"] * 1000.0
    frozen_capture = capture is not None
    if recapturable_m3 <= 1e-9:
        capture = 0.0 if capture is None else float(np.clip(capture, 0.0, 1.0))
        delivered = 0.0
    elif capture is None:
        capture = float(np.clip(lfv_target / recapturable_m3, 0.0, 1.0))
        delivered = capture * recapturable_m3
    else:
        capture = float(np.clip(capture, 0.0, 1.0))
        delivered = capture * recapturable_m3
    to_pool = recapturable * capture
    stay = pile - to_pool
    if neighbor_mm is None:
        hop = np.zeros_like(pile)
        hop_from_cache = False
    else:
        hop = np.asarray(neighbor_mm, dtype=float)
        if hop.ndim == 1:
            hop = hop[None, ...]
        hop_from_cache = True
    hop_m3 = 0.5 * _volume_m3(np.abs(hop), area, cyclic)
    groundwater = None
    groundwater_m3 = float("nan")
    if et_target_mm is not None:
        groundwater = pin_groundwater_for_et(
            et_target_mm, et_mm=et_mm, et_potential_mm=et_potential_mm,
            et_groundwater_mm=et_groundwater_mm)
        groundwater_m3 = _volume_m3(np.maximum(groundwater, 0.0), area, cyclic)
    et_fao_m3 = float("nan") if et_mm is None else _volume_m3(np.maximum(et_mm, 0.0), area, cyclic)
    et_sat_m3 = (float("nan") if et_target_mm is None
                 else _volume_m3(np.maximum(et_target_mm, 0.0), area, cyclic))
    report = SimpleNamespace(
        year=int(year),
        leftover_m3=leftover_m3,
        leftover_table4_m3=lf_target,
        leftover_rel_error=(leftover_m3 - lf_target) / max(abs(lf_target), 1.0),
        drain_capture_m3=delivered,
        lfv_target_m3=lfv_target,
        lfv_rel_error=(delivered - lfv_target) / max(abs(lfv_target), 1.0),
        capture_share=capture,
        ground_stay_m3=leftover_m3 - delivered,
        dwb_target_m3=dwb_target,
        dwb_rel_error=((leftover_m3 - delivered) - dwb_target) / max(abs(dwb_target), 1.0),
        lateral_hop_m3=hop_m3,
        hop_from_cache=hop_from_cache,
        recapturable_m3=recapturable_m3,
        infiltration_limited=infiltration_limited,
        lfv_hit=bool(recapturable_m3 + 1.0 >= lfv_target),
        groundwater_for_et_m3=groundwater_m3,
        et_fao_m3=et_fao_m3,
        et_satellite_m3=et_sat_m3,
        leftover_mm=leftover_m3 * 1000.0 / cyclic_area,
        drain_capture_mm=delivered * 1000.0 / cyclic_area,
        ground_stay_mm=(leftover_m3 - delivered) * 1000.0 / cyclic_area,
        lateral_hop_mm=hop_m3 * 1000.0 / cyclic_area,
        force_satellite_et=False,
        frozen_capture=frozen_capture,
    )
    if squeeze:
        to_pool = to_pool[0]
        stay = stay[0]
        leftover = leftover[0]
        if groundwater is not None and np.ndim(groundwater) > leftover.ndim:
            groundwater = np.asarray(groundwater)[0]
    return leftover, to_pool, stay, report, groundwater


def pin_summary_text(pin):
    """Season volumes after pinning leftover to satellite ET and capture to LFV."""
    def _m3(value):
        return float(value) / 1000.0

    lines = [
        f"{pin.year} leftover after satellite ET: {_m3(pin.leftover_m3):.0f} x10^3 m3 "
        f"({pin.leftover_mm:.0f} mm; Table 4 Lf {_m3(pin.leftover_table4_m3):.0f}, "
        f"{100.0 * pin.leftover_rel_error:+.1f}%)",
        f"  paddy-to-paddy hops (conserved): {_m3(pin.lateral_hop_m3):.0f} x10^3 m3 "
        f"({pin.lateral_hop_mm:.0f} mm)",
        f"  drain capture (LFV): {_m3(pin.drain_capture_m3):.0f} x10^3 m3 "
        f"({pin.drain_capture_mm:.0f} mm; Table 5 {_m3(pin.lfv_target_m3):.0f}, "
        f"{100.0 * pin.lfv_rel_error:+.1f}%, share {pin.capture_share:.3f})",
        f"  ground stay (dWB): {_m3(pin.ground_stay_m3):.0f} x10^3 m3 "
        f"({pin.ground_stay_mm:.0f} mm; Table 5 {_m3(pin.dwb_target_m3):.0f}, "
        f"{100.0 * pin.dwb_rel_error:+.1f}%)",
    ]
    if np.isfinite(pin.groundwater_for_et_m3):
        lines.append(
            f"  extra groundwater to meet satellite ET: {_m3(pin.groundwater_for_et_m3):.0f} x10^3 m3 "
            "(FAO-56 flux not overwritten)"
        )
    if np.isfinite(pin.et_fao_m3) and np.isfinite(pin.et_satellite_m3):
        lines.append(
            f"  FAO-56 ET {_m3(pin.et_fao_m3):.0f} vs satellite {_m3(pin.et_satellite_m3):.0f} x10^3 m3"
        )
    if not pin.lfv_hit and pin.capture_share >= 0.999:
        lines.append(
            "  infiltrated leftover is smaller than Table 5 LFV_pump; capture is 1.0 and LFV is short"
        )
    if getattr(pin, "frozen_capture", False) and pin.capture_share < 0.999:
        lines.append(f"  capture share frozen from the other year ({pin.capture_share:.3f})")
    return "\n".join(lines)


def pin_field_to_et_and_lfv(field, irrigation=None, rain=None, drainage=None, et_target=None,
                            et_mm=None, et_potential=None, et_groundwater=None, capture=None):
    """Replace leftover with the satellite-ET identity and LFV-scaled shares."""
    leftover = field.leftover
    if irrigation is not None and rain is not None and drainage is not None and et_target is not None:
        leftover = leftover_pinned(irrigation, rain, drainage, et_target)
        field.leftover = np.asarray(leftover, dtype=np.float32)
        field.et_target = np.asarray(et_target, dtype=np.float32)
        field.irrigation = np.asarray(irrigation, dtype=np.float32)
        field.drainage = np.asarray(drainage, dtype=np.float32)
        field.rain = np.asarray(rain, dtype=np.float32)
    if getattr(field, "neighbor", None) is None:
        field.neighbor = _leftover_hops(leftover, field)
    # Keep the ET inputs with the field so a cached field re-pins identically.
    if et_potential is not None:
        field.et_potential = np.asarray(et_potential, dtype=np.float32)
    if et_groundwater is not None:
        field.et_groundwater = np.asarray(et_groundwater, dtype=np.float32)
    et_potential = getattr(field, "et_potential", None) if et_potential is None else et_potential
    et_groundwater = getattr(field, "et_groundwater", None) if et_groundwater is None else et_groundwater
    et_fao = field.et if et_mm is None else et_mm
    leftover, to_pool, stay, pin, groundwater = pin_capture_to_lfv(
        leftover, field.area, field.cyclic, field.year,
        infiltration_mm=getattr(field, "infiltration", None),
        neighbor_mm=getattr(field, "neighbor", None),
        et_target_mm=et_target if et_target is not None else getattr(field, "et_target", None),
        et_mm=et_fao,
        et_potential_mm=et_potential,
        et_groundwater_mm=et_groundwater,
        capture=capture)
    field.leftover = np.asarray(leftover, dtype=np.float32)
    field.to_pool = np.asarray(to_pool, dtype=np.float32)
    field.stay_mm = np.asarray(stay, dtype=np.float32)
    field.pin = pin
    if groundwater is not None:
        field.groundwater_for_et = np.asarray(groundwater, dtype=np.float32)
    field.shares = _shares_from_pool_stay(leftover, to_pool, stay, field.cyclic)
    return field


def _leftover_hops(leftover, field):
    from .graph import hop_stock

    leftover = np.asarray(leftover, dtype=float)
    graph = _field_graph(field)
    physics = Physics()
    rows = []
    for day in range(leftover.shape[0]):
        _, delta = hop_stock(np.maximum(leftover[day], 0.0), graph, field.area, physics)
        rows.append(delta)
    return np.asarray(rows, dtype=np.float32)


def _shares_from_pool_stay(leftover, to_pool, stay, cyclic):
    pile = np.maximum(np.asarray(leftover, dtype=float), 0.0)
    capture = np.divide(np.asarray(to_pool, dtype=float), np.maximum(pile, 1e-6))
    stay_frac = np.clip(1.0 - capture, 0.0, 1.0)
    p = Physics()
    top = float(p.total_evaporable_mm / max(p.total_evaporable_mm + p.total_available_mm, 1e-9))
    mask = np.asarray(cyclic, dtype=bool)
    kx = np.where(mask, 0.5 * capture, 0.0)
    ky = np.where(mask, 0.5 * capture, 0.0)
    kz = np.where(mask, stay_frac * top, 0.0)
    ground = np.where(mask, stay_frac * (1.0 - top), 1.0)
    stack = np.stack([kx, ky, kz, ground], axis=-1)
    total = np.maximum(stack.sum(axis=-1, keepdims=True), 1e-12)
    return (stack / total).astype(np.float32)


def pin_water_fate(year, folder=None, field=None, progress=None, compact=False, capture=None):
    """Load a cached year, pin leftover to satellite ET, scale capture to LFV."""
    if field is None:
        field = load_year_field(year, folder=folder, progress=progress, compact=False)
    pin_field_to_et_and_lfv(field, capture=capture)
    if compact:
        save_year_field(field, folder)
    return field


def leftover_split_text(capture=None):
    return (
        "leftover_i = IR_i + rain - D_i - ET_sat_i\n"
        "need_d_i  = ET-deficit of neighbour_d + downhill drop leftover_i - leftover_d\n"
        "Z_i       = need_right + need_left + need_up + need_down\n"
        "flow_d_i  = surplus_i * need_d_i / Z_i   (capped so leftover equalises; hops repeat)\n"
        "ground_i  = leftover still > 0 after hops\n"
        "gw_ET_i   = leftover still < 0 after hops   # pins satellite ET"
    )


FLOW_DIRS = ("right", "left", "up", "down")
FLOW_KEYS = ("east", "west", "north", "south")


def relative_et_route_day(leftover_mm, graph, area_m2=None, hops=8):
    """Move leftover downhill / toward ET-short UDLR neighbours.

    leftover_i = IR+R−D−ET_sat. Negative leftover is unmet satellite ET. Each hop
    sends surplus relatively to neighbours with a deficit or a lower leftover
    (relay). Remaining deficit is groundwater so satellite ET is still matched.
    `force_satellite_et` is not used.
    """
    leftover = np.asarray(leftover_mm, dtype=float).ravel().copy()
    n = leftover.size
    area = np.ones(n) if area_m2 is None else np.broadcast_to(
        np.asarray(area_m2, dtype=float).ravel(), leftover.shape).copy()
    idx = []
    for name in FLOW_KEYS:
        j = np.asarray(getattr(graph, name, np.full(n, -1)), dtype=int).ravel()
        idx.append(j if j.size == n else np.full(n, -1, dtype=int))
    total_send = np.zeros((n, 4))
    total_received = np.zeros(n)
    first_send = np.zeros((n, 4))
    for hop in range(max(int(hops), 1)):
        surplus = np.maximum(leftover, 0.0)
        if float(surplus.sum()) < 1e-9:
            break
        need = np.zeros((n, 4))
        for k, j in enumerate(idx):
            present = j >= 0
            if not np.any(present):
                continue
            drop = leftover[present] - leftover[j[present]]
            nbr_def = np.maximum(-leftover[j[present]], 0.0)
            need[present, k] = nbr_def + np.maximum(drop, 0.0)
        z = need.sum(axis=1)
        offer = np.zeros((n, 4))
        ok = z > 1e-12
        offer[ok] = surplus[ok, None] * (need[ok] / z[ok, None])
        send = np.zeros((n, 4))
        recv = np.zeros(n)
        moved = False
        for k, j in enumerate(idx):
            present = j >= 0
            if not np.any(present):
                continue
            i = np.flatnonzero(present)
            dest = j[i]
            drop = np.maximum(leftover[i] - leftover[dest], 0.0)
            cap = drop * area[dest] / np.maximum(area[i] + area[dest], 1e-9)
            moved_mm = np.minimum(offer[i, k], cap)
            send[i, k] = moved_mm
            vol = moved_mm * area[i]
            recv_mm = vol / np.maximum(area[dest], 1e-9)
            np.add.at(recv, dest, recv_mm)
            if float(moved_mm.sum()) > 1e-9:
                moved = True
        leftover = leftover - send.sum(axis=1) + recv
        total_send = total_send + send
        total_received = total_received + recv
        if hop == 0:
            first_send = send.copy()
        if not moved:
            break
    ground = np.maximum(leftover, 0.0)
    unfilled = np.maximum(-leftover, 0.0)
    return SimpleNamespace(
        leftover=leftover, surplus=np.maximum(np.asarray(leftover_mm, dtype=float).ravel(), 0.0),
        deficit=np.maximum(-np.asarray(leftover_mm, dtype=float).ravel(), 0.0),
        flow=first_send, relay_flow=total_send, ground=ground, received=total_received, unfilled=unfilled,
        groundwater=unfilled,
    )


def relative_et_route_field(field, progress=None, hops=8):
    """Apply relative ET routing to every day of a leftover field."""
    leftover = np.asarray(field.leftover, dtype=float)
    graph = _field_graph(field)
    days, n = leftover.shape
    flow = np.zeros((days, n, 4), dtype=np.float32)
    ground = np.zeros((days, n), dtype=np.float32)
    received = np.zeros((days, n), dtype=np.float32)
    unfilled = np.zeros((days, n), dtype=np.float32)
    for day in range(days):
        routed = relative_et_route_day(leftover[day], graph, field.area, hops=hops)
        flow[day] = routed.flow
        ground[day] = routed.ground
        received[day] = routed.received
        unfilled[day] = routed.unfilled
    cyclic = np.asarray(field.cyclic, dtype=bool)
    area = np.asarray(field.area, dtype=float)
    def vol(mm):
        return _volume_m3(mm, area, cyclic)

    flow_m3 = {name: vol(flow[..., k]) for k, name in enumerate(FLOW_DIRS)}
    et_sat = getattr(field, "et_target", None)
    if et_sat is None:
        et_sat = field.et
    et_sat = np.maximum(np.asarray(et_sat, dtype=float), 0.0)
    et_m3 = vol(et_sat)
    gap0 = vol(np.maximum(-leftover, 0.0))
    gap1 = vol(unfilled)
    matched = max(et_m3 - gap1, 0.0)
    lateral_to_et_m3 = max(gap0 - gap1, 0.0)
    udlr_m3 = lateral_to_et_m3
    cyclic_area = float(area[cyclic].sum()) or 1.0
    lf_m3 = vol(np.maximum(leftover, 0.0))
    lf_target = table_lf_m3(field.year)
    lfv_target = table_lfv_m3(field.year)
    if getattr(field, "infiltration", None) is None:
        recapturable = vol(ground)
    else:
        recapturable = vol(np.minimum(np.maximum(field.infiltration, 0.0), ground))
    capture = float(np.clip(lfv_target / max(recapturable, 1.0), 0.0, 1.0))
    lfv_m3 = capture * recapturable
    mask = np.broadcast_to(cyclic, unfilled.shape)
    et_rmse = float(np.sqrt(np.mean(unfilled[mask] ** 2)))
    et_rmse_gw = 0.0
    lf_rel = (lf_m3 - lf_target) / max(abs(lf_target), 1.0)
    lfv_rel = (lfv_m3 - lfv_target) / max(abs(lfv_target), 1.0)
    return SimpleNamespace(
        year=int(field.year),
        hops=int(hops),
        flow=flow, ground=ground, received=received, unfilled=unfilled,
        flow_m3=flow_m3,
        udlr_m3=udlr_m3,
        ground_m3=vol(ground),
        received_m3=vol(received),
        surplus_m3=vol(np.maximum(leftover, 0.0)),
        deficit_m3=gap0,
        unfilled_m3=gap1,
        groundwater_m3=gap1,
        et_satellite_m3=et_m3,
        et_matched_m3=et_m3,
        et_matched_without_gw_m3=matched,
        et_match=1.0,
        et_match_without_gw=matched / max(et_m3, 1.0),
        et_rmse_mm=et_rmse,
        et_rmse_after_gw_mm=et_rmse_gw,
        gap_closed=1.0 - gap1 / max(gap0, 1.0) if gap0 > 1.0 else 1.0,
        lf_m3=lf_m3,
        lf_table4_m3=lf_target,
        lf_rel_error=lf_rel,
        lf_rmse_mm=abs(lf_m3 - lf_target) * 1000.0 / cyclic_area,
        lfv_m3=lfv_m3,
        lfv_target_m3=lfv_target,
        lfv_rel_error=lfv_rel,
        capture_share=capture,
        force_satellite_et=False,
    )


def relative_et_table(train_route, eval_route=None):
    rows = []
    for route in (train_route, eval_route):
        if route is None:
            continue
        rows.append({
            "year": route.year,
            "ET_RMSE_mm": route.et_rmse_mm,
            "ET_RMSE_after_gw": route.et_rmse_after_gw_mm,
            "ET_from_local_lateral_pct": 100.0 * route.et_match_without_gw,
            "Lf": route.lf_m3 / 1000.0,
            "Table4_Lf": route.lf_table4_m3 / 1000.0,
            "Lf_pct": 100.0 * route.lf_rel_error,
            "Lf_RMSE_mm": route.lf_rmse_mm,
            "UDLR": route.udlr_m3 / 1000.0,
            "to_right": route.flow_m3["right"] / 1000.0,
            "to_left": route.flow_m3["left"] / 1000.0,
            "to_up": route.flow_m3["up"] / 1000.0,
            "to_down": route.flow_m3["down"] / 1000.0,
            "to_ground": route.ground_m3 / 1000.0,
            "LFV": route.lfv_m3 / 1000.0,
            "Table5_LFV": route.lfv_target_m3 / 1000.0,
            "LFV_pct": 100.0 * route.lfv_rel_error,
            "gw_for_ET": route.groundwater_m3 / 1000.0,
        })
    return pd.DataFrame(rows)


def relative_et_holdout(train_year=2021, eval_year=2022, folder=None, progress=None):
    """Pin leftover to satellite ET, route surplus to short neighbours, rest to ground."""
    from .validation import split

    train_year, eval_year = split(train_year, eval_year)
    if progress:
        progress(f"Pinning {train_year} leftover to satellite ET ...")
    train = pin_water_fate(train_year, folder=folder, progress=progress)
    if progress:
        progress(f"Routing {train_year} surplus toward ET-short neighbours ...")
    train_route = relative_et_route_field(train)
    if progress:
        progress(f"Pinning {eval_year} leftover to satellite ET ...")
    eval_field = pin_water_fate(eval_year, folder=folder, progress=progress)
    if progress:
        progress(f"Routing {eval_year} surplus toward ET-short neighbours ...")
    eval_route = relative_et_route_field(eval_field)
    return SimpleNamespace(
        simple=True,
        relative=True,
        train_year=int(train_year),
        eval_year=int(eval_year),
        force_satellite_et=False,
        farmer_model_separate=True,
        equation_text=leftover_split_text(),
        train=train,
        eval=eval_field,
        train_route=train_route,
        eval_route=eval_route,
        train_pin=train.pin,
        eval_pin=eval_field.pin,
        train_source=train.source,
        eval_source=eval_field.source,
        capture=float(train.pin.capture_share),
        eval_ceiling=None,
    )


def simple_leftover_holdout(train_year=2021, eval_year=2022, folder=None, progress=None):
    """The leftover problem, closed form: pin ET, freeze 2021 capture, score 2022.

    Farmer IR/D is not this object. Hops equalise leftover head. Drain capture is
    a single infiltrated-leftover share identified on the train year against
    Table 5 LFV_pump. `force_satellite_et` stays False.
    """
    from .validation import split

    train_year, eval_year = split(train_year, eval_year)
    if progress:
        progress(f"Pinning {train_year} leftover to satellite ET, capture to LFV ...")
    train = pin_water_fate(train_year, folder=folder, progress=progress)
    capture = float(train.pin.capture_share)
    if progress:
        progress(f"Applying frozen capture {capture:.3f} to {eval_year} ...")
    eval_field = pin_water_fate(eval_year, folder=folder, progress=progress, capture=capture)
    ceiling = None
    if eval_field.pin.capture_share < 1.0 or not eval_field.pin.lfv_hit:
        leftover, to_pool, stay, ceiling, _gw = pin_capture_to_lfv(
            eval_field.leftover, eval_field.area, eval_field.cyclic, eval_year,
            infiltration_mm=getattr(eval_field, "infiltration", None),
            neighbor_mm=getattr(eval_field, "neighbor", None),
            et_target_mm=getattr(eval_field, "et_target", None),
            et_mm=getattr(eval_field, "et", None),
            capture=1.0)
        eval_field.ceiling_pin = ceiling
        eval_field.ceiling_to_pool = to_pool
        eval_field.ceiling_stay = stay
    return SimpleNamespace(
        simple=True,
        train_year=int(train_year),
        eval_year=int(eval_year),
        capture=capture,
        force_satellite_et=False,
        farmer_model_separate=True,
        equation_text=leftover_split_text(capture),
        train=train,
        train_pin=train.pin,
        eval=eval_field,
        eval_pin=eval_field.pin,
        eval_ceiling=ceiling,
        train_source=train.source,
        eval_source=eval_field.source,
    )


def simple_leftover_table(report):
    """Train vs frozen holdout vs Okura Tables 4–5, in 10^3 m3."""
    rows = []
    for label, pin in (
        (f"{report.train_year} (identify)", report.train_pin),
        (f"{report.eval_year} (frozen {report.capture:.3f})", report.eval_pin),
    ):
        rows.append({
            "year": label,
            "leftover": pin.leftover_m3 / 1000.0,
            "Table4_Lf": pin.leftover_table4_m3 / 1000.0,
            "leftover_pct": 100.0 * pin.leftover_rel_error,
            "hops": pin.lateral_hop_m3 / 1000.0,
            "drain_LFV": pin.drain_capture_m3 / 1000.0,
            "Table5_LFV": pin.lfv_target_m3 / 1000.0,
            "LFV_pct": 100.0 * pin.lfv_rel_error,
            "stay": pin.ground_stay_m3 / 1000.0,
            "Table5_dWB": pin.dwb_target_m3 / 1000.0,
            "dWB_pct": 100.0 * pin.dwb_rel_error,
            "capture": pin.capture_share,
        })
    if report.eval_ceiling is not None:
        pin = report.eval_ceiling
        rows.append({
            "year": f"{report.eval_year} (capture=1 ceiling)",
            "leftover": pin.leftover_m3 / 1000.0,
            "Table4_Lf": pin.leftover_table4_m3 / 1000.0,
            "leftover_pct": 100.0 * pin.leftover_rel_error,
            "hops": pin.lateral_hop_m3 / 1000.0,
            "drain_LFV": pin.drain_capture_m3 / 1000.0,
            "Table5_LFV": pin.lfv_target_m3 / 1000.0,
            "LFV_pct": 100.0 * pin.lfv_rel_error,
            "stay": pin.ground_stay_m3 / 1000.0,
            "Table5_dWB": pin.dwb_target_m3 / 1000.0,
            "dWB_pct": 100.0 * pin.dwb_rel_error,
            "capture": pin.capture_share,
        })
    return pd.DataFrame(rows)


def _rel(sim, target):
    return float((sim - target) ** 2 / max(target * target, 1.0))


def _signed(sim, target):
    """(sim − target) / |target|. Positive = too much simulated, negative = too little."""
    return float((sim - target) / max(abs(target), 1.0))


def _daily_rel(results, in_obs, sim_dates, part, sim_column, obs_column):
    """Mean squared relative error day by day, scaled by the season's mean daily target.

    The season-total terms cannot see a wrong schedule. This can: it is the error a
    forecaster has to reduce once measured daily pump / ADCP series are loaded.
    """
    if sim_column not in results.daily or not len(part):
        return 0.0
    sim = results.daily.loc[in_obs, sim_column].to_numpy(dtype=float)
    dates = pd.Series(sim_dates)[np.asarray(in_obs, dtype=bool)].to_numpy()
    obs_dates = pd.to_datetime(part["date"])
    if obs_dates.dt.tz is not None:
        obs_dates = obs_dates.dt.tz_convert("UTC").dt.tz_localize(None)
    want = (pd.Series(part[obs_column].to_numpy(dtype=float), index=obs_dates.dt.normalize())
            .groupby(level=0).sum().reindex(pd.DatetimeIndex(dates)).to_numpy(dtype=float))
    ok = np.isfinite(sim) & np.isfinite(want)
    if not np.any(ok):
        return 0.0
    scale = max(float(np.mean(np.abs(want[ok]))), 1.0)
    return float(np.mean((sim[ok] - want[ok]) ** 2) / (scale * scale))


def loss_split(obj, weights=None, signed=None):
    """Weighted squared loss from the paddy imitator vs from leftover ratios.

    The imitator is FieldAggregate / ratio_finder spreading measured IR, D and ET.
    Ratio terms are L/D, farm-to-main, LFV and the capture simplex.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    signed = signed or {}

    def weighted(keys):
        return float(sum(weights.get(name, 0.0) * float(obj.get(name, 0.0)) for name in keys))

    def signed_sum(keys):
        return float(sum(weights.get(name, 0.0) * float(signed.get(name, 0.0)) for name in keys))

    imitator = weighted(IMITATOR_OBJ)
    ratios = weighted(RATIO_OBJ)
    total = imitator + ratios
    mag_i = float(sum(weights.get(name, 0.0) * abs(float(signed.get(name, 0.0))) for name in IMITATOR_OBJ))
    mag_r = float(sum(weights.get(name, 0.0) * abs(float(signed.get(name, 0.0))) for name in RATIO_OBJ))
    mag = mag_i + mag_r
    return SimpleNamespace(
        imitator=imitator,
        ratios=ratios,
        total=total,
        imitator_pct=0.0 if total <= 0 else 100.0 * imitator / total,
        ratios_pct=0.0 if total <= 0 else 100.0 * ratios / total,
        imitator_signed=signed_sum(IMITATOR_OBJ),
        ratios_signed=signed_sum(RATIO_OBJ),
        signed_ratio=(signed_sum(IMITATOR_OBJ) / signed_sum(RATIO_OBJ)
                      if abs(signed_sum(RATIO_OBJ)) > 1e-12 else 0.0),
        imitator_signed_pct=0.0 if mag <= 0 else 100.0 * mag_i / mag,
        ratios_signed_pct=0.0 if mag <= 0 else 100.0 * mag_r / mag,
    )


class HydrologyProblem:
    """Vector-valued identification problem. `scalar` is the weighted sum for a
    classical solver; `qubo` is the same ratios as a binary Hamiltonian."""

    def __init__(self, year=okura.COMPARISON_YEAR, physics=None, weights=None,
                 agents="ratio_finder", include_darcy=False):
        self.year = year
        constructed = construct_ratios(year)
        self.constructed = constructed
        self.physics = physics or Physics().replace(**constructed)
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.agents = agents
        self.include_darcy = bool(include_darcy)
        self.names = ACTIVE + (LATER if self.include_darcy else ())
        self.ld_target = {block: table_ld_ratio(year, block) for block in BLOCKS}
        self.d_main_target_m3 = table_main_d_m3(year)
        self.lfv_target_m3 = table_lfv_m3(year)
        from . import imitation
        self.observed_blocks = imitation.load_observed(year)[0]

    def bounds_array(self):
        return np.array([BOUNDS[name] for name in self.names], dtype=float)

    def pack(self, params=None, **extra):
        values = {}
        for name in self.names:
            value = getattr(self.physics, name, None)
            if value is None:
                value = BOUNDS[name][0]
            values[name] = float(value)
        values.update(params or {})
        values.update(extra)
        return np.array([float(values[name]) for name in self.names], dtype=float)

    def unpack(self, vector):
        values = {name: float(vector[i]) for i, name in enumerate(self.names)}
        if all(name in values for name in ("kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio")):
            raw = np.array([values["kx_ratio"], values["ky_ratio"],
                            values["kz_ratio"], values["ground_ratio"]], dtype=float)
            raw = np.maximum(raw, 0.0)
            total = float(raw.sum()) or 1.0
            values["kx_ratio"], values["ky_ratio"], values["kz_ratio"], values["ground_ratio"] = raw / total
        return values

    def physics_with(self, vector):
        return self.physics.replace(**self.unpack(vector))

    def objectives(self, results):
        """Squared relative errors from a finished (or partial) season. 0 is exact."""
        model = results.model
        log = model.log or {}
        inputs = model.inputs
        days = len(results.daily)
        area = model.area
        cyclic = np.asarray(inputs.cyclic, dtype=bool)
        et = np.asarray(log.get("et_mm", np.zeros((days, model.n))))[:days]
        target = np.asarray(inputs.et_target_mm[:days], dtype=float)
        is_pass = np.asarray(getattr(inputs, "is_pass", np.ones_like(target)), dtype=float)[:days]
        if is_pass.shape != et.shape:
            is_pass = np.ones_like(et)
        et_miss = (et - target) ** 2 / np.maximum(target, 0.5) ** 2
        et_ok = np.isfinite(et) & np.isfinite(target)
        if et.size and np.any(et_ok):
            et_w = np.where(is_pass > 0, 2.0, 1.0)
            et_err = float(np.average(et_miss[et_ok], weights=et_w[et_ok]))
        else:
            et_err = 0.0
        pond = np.asarray(log.get("pond_mm", log.get("depth_mm", np.zeros((days, model.n)))))[:days]
        flood = np.asarray(getattr(inputs, "flood_daily", np.zeros_like(pond)))[:days]
        if flood.shape == pond.shape and np.any(flood):
            ponded = (pond > 0).astype(float)
            flood_c = np.clip(flood, 0.0, 1.0)
            pond_err = float(np.mean((ponded - flood_c) ** 2))
            pond_signed = float(np.mean(ponded - flood_c))
        else:
            pond_err = 0.0
            pond_signed = 0.0
        leftover = np.asarray(log.get("leftover_mm", np.zeros((days, model.n))))[:days]
        drainage = np.asarray(log.get("drainage_mm", np.zeros((days, model.n))))[:days]
        ld_err = 0.0
        ld_signed = 0.0
        lat_err = 0.0
        lat_signed = 0.0
        for block in BLOCKS:
            w = np.where((inputs.block == block) & cyclic, area, 0.0)
            l_m3 = float((leftover * w).sum() / 1000.0)
            d_m3 = float((drainage * w).sum() / 1000.0)
            sim = l_m3 / max(d_m3, 1e-9)
            ld_err += _rel(sim, self.ld_target[block]) / len(BLOCKS)
            ld_signed += _signed(sim, self.ld_target[block]) / len(BLOCKS)
            want_l = table_lateral_m3(self.year, block)
            lat_err += _rel(l_m3, want_l) / len(BLOCKS)
            lat_signed += _signed(l_m3, want_l) / len(BLOCKS)
        obs = getattr(self, "observed_blocks", None)
        sim_dates = pd.to_datetime(results.daily["date"])
        if getattr(sim_dates.dt, "tz", None) is not None:
            sim_dates = sim_dates.dt.tz_convert("UTC").dt.tz_localize(None)
        sim_dates = sim_dates.dt.normalize()
        if obs is not None and days:
            obs_dates = pd.to_datetime(obs["date"])
            if obs_dates.dt.tz is not None:
                obs_dates = obs_dates.dt.tz_convert("UTC").dt.tz_localize(None)
            obs_dates = obs_dates.dt.normalize()
            in_obs = sim_dates.isin(pd.Index(obs_dates.unique()))
        else:
            in_obs = np.ones(len(results.daily), dtype=bool)
        d_main = float(results.daily.loc[in_obs, "D_pool_m3"].sum()) if "D_pool_m3" in results.daily else 0.0
        lfv = float(results.daily.loc[in_obs, "LFV_pump_m3"].sum()) if "LFV_pump_m3" in results.daily else 0.0
        ir_err = dr_err = ir_signed = dr_signed = 0.0
        ir_day_err = d_day_err = 0.0
        if obs is not None and days:
            for block in BLOCKS:
                prefix = f"{block}|cyclic|"
                part = obs[obs["block"] == block]
                want_ir = float(part["irrigation_m3"].sum()) if len(part) else 1.0
                want_d = float(part["drainage_m3"].sum()) if len(part) else 1.0
                sim_ir = float(results.daily.loc[in_obs, prefix + "irrigation_m3"].sum()) if prefix + "irrigation_m3" in results.daily else 0.0
                sim_d = float(results.daily.loc[in_obs, prefix + "drainage_m3"].sum()) if prefix + "drainage_m3" in results.daily else 0.0
                ir_err += _rel(sim_ir, want_ir) / len(BLOCKS)
                dr_err += _rel(sim_d, want_d) / len(BLOCKS)
                ir_signed += _signed(sim_ir, want_ir) / len(BLOCKS)
                dr_signed += _signed(sim_d, want_d) / len(BLOCKS)
                # Day-by-day, not the season total. A schedule that only gets the total
                # right scores 0 above and non-zero here.
                ir_day_err += _daily_rel(results, in_obs, sim_dates, part,
                                         prefix + "irrigation_m3", "irrigation_m3") / len(BLOCKS)
                d_day_err += _daily_rel(results, in_obs, sim_dates, part,
                                        prefix + "drainage_m3", "drainage_m3") / len(BLOCKS)
        from .canal import darcy_shares, mean_pixel_shares
        capture = float(self.constructed.get("pool_capture_share", 0.0))
        pixel = getattr(model, "pixel_shares", None)
        if pixel is not None and "pool_capture_share" in pixel:
            area_w = area
            cyclic_w = cyclic
            if model.cells is not None and np.size(pixel["pool_capture_share"]) == len(model.cells.owner):
                grid = model.cells
                area_w = model.area[grid.owner] * grid.weights
                cyclic_w = cyclic[grid.owner]
            sim_cap = mean_pixel_shares(pixel, area_w, cyclic_w)["pool_capture_share"]
        else:
            sim_share = darcy_shares(model.physics)
            sim_cap = float(sim_share.get("lfv", sim_share["kx"] + sim_share["ky"]))
        et_signed = float(np.mean((et - target) / np.maximum(np.abs(target), 0.5))) if et.size else 0.0
        squared = {
            "et": et_err,
            "irrigation": ir_err,
            "discharge": dr_err,
            "ir_daily": ir_day_err,
            "d_daily": d_day_err,
            "pond": pond_err,
            "lateral": lat_err,
            "ld_ratio": ld_err,
            "farm_to_main": _rel(d_main, self.d_main_target_m3),
            "lfv": _rel(lfv, self.lfv_target_m3),
            "darcy": float((sim_cap - capture) ** 2),
        }
        signed = {
            "et": et_signed,
            "irrigation": ir_signed,
            "discharge": dr_signed,
            "ir_daily": 0.0,
            "d_daily": 0.0,
            "pond": pond_signed,
            "lateral": lat_signed,
            "ld_ratio": ld_signed,
            "farm_to_main": _signed(d_main, self.d_main_target_m3),
            "lfv": _signed(lfv, self.lfv_target_m3),
            "darcy": float(sim_cap - capture),
        }
        results.signed_objectives = signed
        return squared

    def scalar(self, obj):
        return float(sum(self.weights.get(name, 0.0) * obj.get(name, 0.0) for name in OBJECTIVES))

    def evaluate(self, vector, days=None, save_csv=False, record_paddies=True,
                 potential="field", span="season", pixel_shares=None,
                 pixel_shares_daily=None, leftover_mm=None):
        from .model import CyclicModel, Results

        model = CyclicModel(self.year, agents=self.agents, physics=self.physics_with(vector),
                            record_paddies=record_paddies, save_csv=save_csv,
                            potential=potential, span=span, record_cells=False)
        if pixel_shares is not None:
            model.pixel_shares = _reshape_pixel_shares(pixel_shares, model)
        else:
            params = self.unpack(vector)
            capture = float(self.constructed.get(
                "pool_capture_share",
                params.get("pool_capture_share", 0.0)))
            params["pool_capture_share"] = capture
            params.setdefault(
                "neighbour_of_remain",
                float(self.constructed.get("neighbour_of_remain", 0.5)))
            remain = max(1.0 - capture, 0.0)
            free = (params.get("kx_ratio", 0.0) + params.get("ky_ratio", 0.0)
                    + params.get("kz_ratio", 0.0) + params.get("ground_ratio", 0.0))
            if free > 1e-12 and remain >= 0.0:
                scale = remain / free
                for key in ("kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio"):
                    if key in params:
                        params[key] = float(params[key]) * scale
            attach_pixel_shares(model, params)
            leftover = leftover_mm if leftover_mm is not None else getattr(self, "leftover_mm", None)
            if leftover is not None:
                _reweight_shares_to_leftover(model, leftover, params)
        if pixel_shares_daily is not None:
            model.pixel_shares_daily = _reshape_pixel_daily(pixel_shares_daily, model)
        limit = model.n_days if days is None else min(int(days), model.n_days)
        for _ in range(limit):
            if not model.running:
                break
            model.step()
        results = Results(model)
        obj = self.objectives(results)
        signed = getattr(results, "signed_objectives", {})
        return SimpleNamespace(vector=np.asarray(vector, dtype=float), params=self.unpack(vector),
                               objectives=obj, signed=signed,
                               split=loss_split(obj, self.weights, signed),
                               loss=self.scalar(obj), results=results)

    def qubo(self, leftover_m3, discharge_m3, bits=3, smoothness=0.0):
        """Binary Hamiltonian for ρ_ld and ρ_sm (and leftover assignment if leftover_m3 given).

        Each ratio is encoded as bits bits on its bound interval. The same Q dict is what a
        quantum annealer would minimise; `decode_qubo` maps a bitstring back to ratios.
        Darcy bits are omitted until include_darcy is set.
        """
        return ratio_qubo(leftover_m3, discharge_m3, self.ld_target, self.d_main_target_m3,
                          bits=bits, smoothness=smoothness, include_assignment=True)

    def graph_qubo(self, leftover_m3, dist_m, edges, cyclic, smoothness=0.05):
        """Optional leftover-graph QUBO (include or skip). Same LFV target as Table 5."""
        from .graph import assignment_qubo, leftover_graph
        graph = leftover_graph(dist_m, edges, cyclic,
                               one_way=bool(getattr(self.physics, "one_way_to_canal", False)))
        return assignment_qubo(leftover_m3, graph, self.lfv_target_m3, smoothness=smoothness)


def ratio_qubo(leftover_m3, discharge_m3, ld_target, d_main_target_m3,
               bits=3, smoothness=0.0, include_assignment=True):
    """Q dict {(i, j): Qij} with i <= j. Bits: leftover x_i, then ρ_ld, then ρ_sm.

    H = (x·leftover - ρ_ld D)^2 + (ρ_ld - L/D_table)^2 + (ρ_sm D - D_main_table)^2
    """
    leftover = np.asarray(leftover_m3, dtype=float).ravel()
    discharge = np.asarray(discharge_m3, dtype=float).ravel()
    n = leftover.size
    discharge = np.broadcast_to(discharge, (n,))
    n_bits = n if include_assignment else 0
    ld_lo, ld_hi = BOUNDS["lateral_per_discharge"]
    sm_lo, sm_hi = BOUNDS["farm_to_main"]
    denom = max(2 ** bits - 1, 1)
    ld_w = (ld_hi - ld_lo) / denom * (2 ** np.arange(bits, dtype=float))
    sm_w = (sm_hi - sm_lo) / denom * (2 ** np.arange(bits, dtype=float))
    d_vol = float(np.maximum(discharge.sum(), 1e-9))
    target_ld = float(np.mean(list(ld_target.values()))) if isinstance(ld_target, dict) else float(ld_target)
    q = {}

    def add(i, j, value):
        if abs(value) < 1e-18:
            return
        key = (i, j) if i <= j else (j, i)
        q[key] = q.get(key, 0.0) + float(value)

    def square_linear(offset, weights, start, scale=1.0, target=0.0):
        """(offset + w·b - target)^2 in binary b, times scale."""
        c = offset - target
        add_const = scale * c * c
        for k, wk in enumerate(weights):
            add(start + k, start + k, scale * (wk ** 2 + 2.0 * c * wk))
            for m in range(k + 1, len(weights)):
                add(start + k, start + m, scale * 2.0 * wk * weights[m])
        return add_const

    b0 = n_bits
    c0 = n_bits + bits
    offset = 0.0
    if include_assignment:
        L = leftover
        for i in range(n):
            add(i, i, L[i] ** 2)
            for j in range(i + 1, n):
                add(i, j, 2.0 * L[i] * L[j])
            add(i, i, -2.0 * L[i] * ld_lo * d_vol)
            for k, wk in enumerate(ld_w):
                add(i, b0 + k, -2.0 * L[i] * wk * d_vol)
        offset += (ld_lo * d_vol) ** 2
        for k, wk in enumerate(ld_w):
            add(b0 + k, b0 + k, (wk * d_vol) ** 2 + 2.0 * ld_lo * d_vol * wk * d_vol)
            for m in range(k + 1, bits):
                add(b0 + k, b0 + m, 2.0 * wk * ld_w[m] * d_vol ** 2)
    offset += square_linear(ld_lo, ld_w, b0, scale=1.0, target=target_ld)
    offset += square_linear(sm_lo * d_vol, sm_w * d_vol, c0, scale=1.0, target=d_main_target_m3)
    n_var = c0 + bits
    return SimpleNamespace(
        Q=q, offset=offset, n_var=n_var, n_points=n_bits, bits=bits,
        ld_lo=ld_lo, sm_lo=sm_lo, ld_w=ld_w, sm_w=sm_w, b0=b0, c0=c0,
        decode=lambda bits_: decode_qubo(bits_, n_bits, bits, ld_lo, sm_lo, ld_w, sm_w, b0, c0),
    )


def decode_qubo(bitstring, n_points, bits, ld_lo, sm_lo, ld_w, sm_w, b0, c0):
    z = np.asarray(bitstring, dtype=float).ravel()
    x = z[:n_points] if n_points else np.zeros(0)
    rho_ld = ld_lo + float(z[b0:b0 + bits] @ ld_w)
    rho_sm = sm_lo + float(z[c0:c0 + bits] @ sm_w)
    return SimpleNamespace(x=x, lateral_per_discharge=rho_ld, farm_to_main=np.clip(rho_sm, 0.0, 1.0))


def qubo_energy(Q, bitstring, offset=0.0):
    z = np.asarray(bitstring, dtype=float).ravel()
    energy = float(offset)
    for (i, j), qij in Q.items():
        energy += qij * z[i] * z[j] if i != j else qij * z[i]
    return energy


def solve_classical(problem, n_eval=16, seed=0, days=None, start=None,
                    potential="pixel", span="season"):
    """Weighted-sum search over the active ratios (random walk, clipped to bounds).

    Starts from `start` or a saved best. Does not search Darcy K. Each evaluation is a
    season (or `days`). For a quantum annealer, build `problem.qubo(...)` instead.
    """
    bounds = problem.bounds_array()
    rng = np.random.default_rng(seed)
    x0 = problem.pack() if start is None else np.asarray(start, dtype=float)

    def clip(z):
        return np.clip(z, bounds[:, 0], bounds[:, 1])

    def run(z):
        return problem.evaluate(clip(z), days=days, potential=potential, span=span)

    best = run(x0)
    history = [best]
    scale = bounds[:, 1] - bounds[:, 0]
    for k in range(max(int(n_eval) - 1, 0)):
        step = scale * (0.35 * (0.7 ** k) * rng.standard_normal(len(x0)))
        trial = run(best.vector + step)
        history.append(trial)
        if trial.loss < best.loss:
            best = trial
    return SimpleNamespace(best=best, history=history, names=problem.names, later=LATER)


ROUTING_SPEEDS = (2.0, 10.0, 25.0, 50.0, 100.0, 200.0)


def routing_delivery(year=okura.COMPARISON_YEAR, speeds=ROUTING_SPEEDS, folder=None):
    """How much of the recapture share the same-day collection puts in the pool.

    Recapture Kx+Ky is pinned to Table 5 LFV_pump / LfA. That share of leftover is
    counted as LFV the same day if the pan has already sent it underground; pond
    leftover cannot be LFV. Hops no longer deliver LFV. Written to
    outputs/ratios/routing_{year}.txt so the manuscript can quote it without
    re-running a season per test.
    """
    import os

    from .canal import mean_pixel_shares
    from .physics import Physics

    shares = construct_ratios(year)
    pin = float(shares["pool_capture_share"])
    target = table_lfv_m3(year)
    rows = []
    for speed in speeds:
        physics = Physics(leftover_speed_m_d=float(speed)).replace(**shares)
        problem = HydrologyProblem(year, physics=physics)
        out = problem.evaluate(problem.pack(shares))
        model = out.results.model
        cyclic_area = np.where(model.inputs.cyclic, model.area, 0.0)
        entered = float((model.log["lateral_mm"] @ cyclic_area).sum()) / 1000.0
        leftover = float((np.maximum(model.log["leftover_mm"], 0.0) @ cyclic_area).sum()) / 1000.0
        infil = float((np.maximum(model.log["infiltration_mm"], 0.0) @ cyclic_area).sum()) / 1000.0
        lfv = float(out.results.daily["LFV_pump_m3"].sum())
        short = max(entered - lfv, 0.0)
        mean_cap = mean_pixel_shares(
            model.pixel_shares, model.area,
            np.asarray(model.inputs.cyclic, dtype=bool))["pool_capture_share"]
        rows.append({
            "leftover_speed_m_d": float(speed),
            "mean_capture": float(mean_cap),
            "entered_over_leftover": float(entered / max(leftover, 1e-9)),
            "infiltration_m3": infil,
            "lfv_sim_m3": lfv,
            "lfv_target_m3": float(target),
            "delivered": float(lfv / max(target, 1e-9)),
            "infil_over_target": float(infil / max(target, 1e-9)),
            "shortfall_m3": short,
            "lfv_objective": float(out.objectives["lfv"]),
            "loss": float(out.loss),
            "accepts": float(lfv / max(target, 1e-9)) >= LFV_ACCEPT_FRAC,
        })
    frame = pd.DataFrame(rows)
    any_ok = bool(frame["accepts"].any()) if len(frame) else False
    text = "\n".join([
        f"Leftover recapture delivery - {year} - closed-form Table 4-5 vector",
        f"collection ratio Kx+Ky pinned to LFV_pump/LfA = {pin:.4f}; Table 5 LFV_pump = {target:,.0f} m3",
        f"Beven limit of acceptability (set before the sweep): delivered >= {LFV_ACCEPT_FRAC:.2f} of Table 5",
        f"limit met at any leftover speed: {any_ok}",
        "",
        frame.to_string(index=False, float_format=lambda v: f"{v:,.4f}"),
        "",
        "Delivered is simulated LFV_pump over Table 5. Recapture is same-day collection "
        "of infiltrated leftover, not hop-then-lining-catch, so leftover speed should "
        "not close or open the gap. Recapture is uniform on cyclic cells, so leftover-"
        "weighted capture matches the area-weighted pin. Remaining gap is leftover "
        "volume, daily pan clip, and recapture taken by unsaturated neighbours.",
    ])
    folder = folder or os.path.join(_data.PROJECT, "outputs", "ratios")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"routing_{int(year)}.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    return SimpleNamespace(frame=frame, text=text, path=path, pin=pin, target=float(target))


def solve_qubo_classical(encoding, sweeps=4, seed=0):
    """Local bit flips on the ratio QUBO — the same loop a QPU would replace."""
    n = encoding.n_var
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, n).astype(float)
    energy = qubo_energy(encoding.Q, z, encoding.offset)
    best, best_e = z.copy(), energy
    for _ in range(max(sweeps, 1)):
        for i in rng.permutation(n):
            z[i] = 1.0 - z[i]
            new = qubo_energy(encoding.Q, z, encoding.offset)
            if new <= energy:
                energy = new
                if energy < best_e:
                    best, best_e = z.copy(), energy
            else:
                z[i] = 1.0 - z[i]
    decoded = encoding.decode(best)
    decoded.energy = best_e
    decoded.bitstring = best
    return decoded


SHARE_LABELS = (
    ("kx_ratio", "Kx east (+) / west (−)", "flow_kx_mm", "RdBu_r"),
    ("ky_ratio", "Ky north (+) / south (−)", "flow_ky_mm", "RdBu_r"),
    ("kz_ratio", "Kz topsoil", "flow_kz_mm", "YlOrBr"),
    ("ground_ratio", "ground soil", "flow_ground_mm", "Greys"),
)
SPANS = ("tables", "season", "year")


def table_closure_objectives(year):
    """Loss from Tables 4–5 alone: IR, D and ET are taken as measured (error 0)."""
    area = {block: okura.cyclic_area_m2(block) for block in BLOCKS}
    total = sum(area.values())

    def wavg(key):
        return sum(okura.TABLE4_MM[(block, year)][key] * area[block] for block in BLOCKS) / total

    leftover = max(wavg("irrigation") + wavg("rain") - wavg("drainage") - wavg("et"), 1e-9)
    lf = wavg("lateral")
    lfv = table_lfv_m3(year)
    lfa = (okura.TABLE5_1000M3[year]["LfA_North"] + okura.TABLE5_1000M3[year]["LfA_South"]) * 1000.0
    constructed = construct_ratios(year)
    capture = constructed["pool_capture_share"]
    d4_mm = wavg("drainage")
    obj = {
        "et": 0.0, "irrigation": 0.0, "discharge": 0.0, "pond": 0.0,
        "ir_daily": 0.0, "d_daily": 0.0,
        "lateral": _rel(leftover, lf),
        "ld_ratio": _rel(leftover / max(d4_mm, 1e-9), lf / max(d4_mm, 1e-9)),
        "farm_to_main": 0.0,
        "lfv": _rel(capture * lfa, lfv),
        "darcy": (constructed["pool_capture_share"] - capture) ** 2,
        "closure": ((leftover - lf) / leftover) ** 2,
    }
    signed = {
        "et": 0.0, "irrigation": 0.0, "discharge": 0.0, "pond": 0.0,
        "ir_daily": 0.0, "d_daily": 0.0,
        "lateral": _signed(leftover, lf),
        "ld_ratio": _signed(leftover / max(d4_mm, 1e-9), lf / max(d4_mm, 1e-9)),
        "farm_to_main": 0.0,
        "lfv": _signed(capture * lfa, lfv),
        "darcy": constructed["pool_capture_share"] - capture,
        "closure": (leftover - lf) / leftover,
    }
    return obj, signed


def field_share_maps(year, params, results=None):
    from . import data
    from .canal import distance_to_canal_m, pixel_ratio_fields
    from .lateral import knn_edges
    from .physics import Physics

    frame = data.paddies()
    lon = frame["lon"].to_numpy()
    lat = frame["lat"].to_numpy()
    cyclic = frame["cyclic"].to_numpy(dtype=bool)
    area = frame["area_m2"].to_numpy() if "area_m2" in frame.columns else np.ones(len(frame))
    block = frame["block"].to_numpy() if "block" in frame.columns else None
    model = None if results is None else results.model
    fields = getattr(model, "pixel_shares", None) if model is not None else None
    if fields is not None and model.cells is not None and np.size(fields["kx_ratio"]) == len(model.cells.owner):
        grid = model.cells
        lon, lat = grid.lon, grid.lat
        cyclic = np.asarray(model.inputs.cyclic[grid.owner], dtype=bool)
        area = model.area[grid.owner] * grid.weights
        block = np.asarray(model.inputs.block[grid.owner])
    elif fields is not None and model is not None and np.size(fields["kx_ratio"]) == model.n:
        lon = model.inputs.frame["lon"].to_numpy()
        lat = model.inputs.frame["lat"].to_numpy()
        cyclic = np.asarray(model.inputs.cyclic, dtype=bool)
        area = model.area
        block = np.asarray(model.inputs.block)
    else:
        p = Physics()
        topsoil = float(p.total_evaporable_mm / max(p.total_evaporable_mm + p.total_available_mm, 1e-9))
        intensity = None
        try:
            from .paper_plots import block_leftover_intensity
            intensity = block_leftover_intensity(int(year))
        except Exception:
            intensity = None
        fields = pixel_ratio_fields(
            len(frame), distance_to_canal_m(lon, lat), knn_edges(lon, lat, k=4), cyclic,
            float(params.get("pool_capture_share",
                             params.get("kx_ratio", 0.0) + params.get("ky_ratio", 0.0))),
            float(params.get("farm_to_main", 1.0)), topsoil, float(p.canal_length_m),
            area_m2=area, block=block, block_intensity=intensity,
            neighbour_of_remain=float(params.get(
                "neighbour_of_remain", getattr(p, "neighbour_of_remain", 0.5))))
    volumes = {key: np.asarray(fields[key], dtype=float) for key, _, _, _ in SHARE_LABELS}
    if "kx_signed" in fields:
        volumes["kx_ratio"] = np.asarray(fields["kx_signed"], dtype=float)
    if "ky_signed" in fields:
        volumes["ky_ratio"] = np.asarray(fields["ky_signed"], dtype=float)
    return SimpleNamespace(lon=lon, lat=lat, cyclic=cyclic, area=area, volumes=volumes,
                           params=params, fields=fields)


def ratio_map_figure(report):
    """One season-constant ratio set: shares, loss, and field maps. Not a daily animation."""
    from matplotlib.figure import Figure

    params = report.params
    maps = report.maps
    fig = Figure(figsize=(13.5, 10.2))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.15, 1.15, 1.35], hspace=0.42, wspace=0.28)
    pie = fig.add_subplot(grid[0, :2])
    shares = [max(float(params[k]), 0.0) for k, _, _, _ in SHARE_LABELS]
    labels = [f"{lab.split(' ', 1)[0]} {100 * s:.1f}%" for (k, lab, _, _), s in zip(SHARE_LABELS, shares)]
    colours = ["#4c78a8", "#59a14f", "#f28e2b", "#8c8c8c"]
    if sum(shares) <= 0:
        shares = [1.0]
        labels = ["none"]
        colours = ["#ccc"]
    pie.pie(shares, labels=labels, colors=colours[:len(shares)], startangle=90,
            wedgeprops={"linewidth": 0.6, "edgecolor": "white"})
    pie.set_title("Area-weighted mean leftover shares", fontsize=11)

    bars = fig.add_subplot(grid[0, 2:])
    signed = getattr(report, "signed", None) or {}
    names = [k for k in ("et", "irrigation", "discharge", "pond", "lateral", "ld_ratio", "farm_to_main", "lfv", "darcy", "closure")
             if k in (signed if signed else report.objectives)]
    values = [float(signed.get(k, report.objectives.get(k, 0.0))) for k in names]
    colours = ["#f28e2b" if k in IMITATOR_OBJ else "#4c78a8" for k in names]
    bars.barh(names, values, color=colours)
    bars.axvline(0.0, color="0.3", lw=0.8)
    bars.set_xlabel("signed relative error  (sim − target) / |target|")
    split = getattr(report, "split", None)
    if split is not None:
        bars.set_title(
            f"Imitator {split.imitator_pct:.0f}%  ·  ratios {split.ratios_pct:.0f}%   "
            f"(signed {split.imitator_signed:+.2f} / {split.ratios_signed:+.2f})",
            fontsize=10)
    else:
        bars.set_title(f"Loss {report.loss:.4g}", fontsize=11)
    bars.invert_yaxis()

    farm = fig.add_subplot(grid[1, :2])
    if split is not None:
        farm.barh(["loss share"], [split.imitator_pct], color="#f28e2b", label="paddy imitator")
        farm.barh(["loss share"], [split.ratios_pct], left=[split.imitator_pct],
                  color="#4c78a8", label="leftover ratios")
        farm.set_xlim(0, 100)
        farm.set_xlabel("% of weighted squared loss")
        farm.legend(loc="lower right", fontsize=8)
        farm.set_title("Where the loss comes from", fontsize=11)
    else:
        farm.bar(["farm drain → main canal"], [float(params.get("farm_to_main", 0.0))], color="#b07aa1")
        farm.set_ylim(0, 1.05)
        farm.set_ylabel("ratio")
        farm.set_title("Farm-to-main (district mean)", fontsize=11)

    text = fig.add_subplot(grid[1, 2:])
    text.axis("off")
    lines = [
        f"span: {report.span} · {report.days} days",
        f"Kx {params['kx_ratio']:.3f}  Ky {params['ky_ratio']:.3f}",
        f"Kx signed {params.get('kx_signed', 0):+.3f}  (+E/−W)",
        f"Ky signed {params.get('ky_signed', 0):+.3f}  (+N/−S)",
        f"Kz {params['kz_ratio']:.3f}  ground {params['ground_ratio']:.3f}",
        f"farm_to_main {params.get('farm_to_main', 0):.3f}",
        f"scalar loss {report.loss:.4g}  ({report.seconds:.1f} s)",
        (f"best saved: {getattr(report, 'saved', '—')}" if getattr(report, "saved", None) else
         "Each pixel has its own Kx/Ky/Kz/ground; maps are those shares."),
    ]
    text.text(0.0, 0.95, "\n".join(lines), va="top", fontsize=10, family="monospace",
              transform=text.transAxes)

    for i, (key, label, _, cmap) in enumerate(SHARE_LABELS):
        ax = fig.add_subplot(grid[2, i])
        values = maps.volumes[key]
        cyclic = maps.cyclic
        ax.scatter(maps.lon[~cyclic], maps.lat[~cyclic], s=3, color="lightgrey", linewidths=0)
        n_pts = int(np.count_nonzero(cyclic))
        size = 1.2 if n_pts > 8000 else 4
        points = ax.scatter(maps.lon[cyclic], maps.lat[cyclic], c=values[cyclic], s=size,
                            linewidths=0, cmap=cmap)
        ax.set_aspect(1 / 0.81)
        ax.set_xticks([])
        ax.set_yticks([])
        if key in ("kx_ratio", "ky_ratio"):
            unit = "+east / −west" if key == "kx_ratio" else "+north / −south"
            vmax = max(float(np.nanmax(np.abs(values[cyclic]))), 1e-6) if np.any(cyclic) else 1.0
            points.set_clim(-vmax, vmax)
        else:
            unit = "share at pixel"
            points.set_clim(0.0, max(float(np.nanmax(values[cyclic])) if np.any(cyclic) else 1.0, 1e-6))
        ax.set_title(f"{label}\n{unit}", fontsize=9)
        fig.colorbar(points, ax=ax, orientation="horizontal", shrink=0.85, pad=0.08)
    fig.suptitle("Leftover partition per pixel, held constant in time", fontsize=13)
    return fig


def parse_finder_method(method="classical", ratio_agent=None):
    """(kind, registered_name). kind is classical, omalley, ibm, qrc_daily, qrc_spatial, registered."""
    if ratio_agent:
        return "registered", str(ratio_agent)
    raw = (method or "classical").strip()
    aliases = {
        "classical": ("classical", None),
        "ibm": ("ibm", None),
        "qaoa": ("omalley", None),
        "omalley": ("omalley", None),
        "quantum": ("omalley", None),
        "qrc": ("qrc_daily", None),
        "qrc_daily": ("qrc_daily", None),
        "daily_qrc": ("qrc_daily", None),
        "qrc_spatial": ("qrc_spatial", None),
        "spatial_qrc": ("qrc_spatial", None),
    }
    if raw in aliases:
        return aliases[raw]
    if raw.startswith("registered:") or raw.startswith("ratio:"):
        return "registered", raw.split(":", 1)[1]
    from .agents import RATIOS
    if raw in RATIOS:
        return "registered", raw
    raise ValueError(
        f"unknown ratio method {method!r}; use classical, omalley, qrc_daily, "
        f"qrc_spatial, or a registered ratio {sorted(RATIOS)}")


PACKAGE_FORECASTERS = (
    "et_match", "et_net", "et_follower", "satellite", "field_aggregate", "ratio_finder",
    "rainfed", "flooded", "linear", "linear_fate", "persist",
)


def registered_forecasters():
    """Farmer names registered from example.py / the page, not package builtins."""
    from .agents import AGENTS, FarmerPolicy

    names = []
    for name, cls in AGENTS.items():
        if name in PACKAGE_FORECASTERS:
            continue
        if isinstance(cls, type) and issubclass(cls, FarmerPolicy):
            names.append(name)
    return sorted(names)


def _forecaster_span_days(span):
    return 365 if span == "year" else int(okura.SEASON_DAYS)


def _forecaster_fits_span(entry, span):
    results = getattr(entry, "results", None)
    model = getattr(results, "model", None) if results is not None else None
    if model is None:
        return False
    return int(getattr(model, "n_days", 0)) == _forecaster_span_days(span)


def score_forecaster(year, name, observed=None, span="season"):
    """One season of a registered farmer, scored on observed block IR/D."""
    from .agents import leftover_from_season
    from .imitation import load_observed, score
    from .model import CyclicModel

    observed = observed if observed is not None else load_observed(year)[0]
    results = CyclicModel(year, agents=name, record_paddies=True, save_csv=False, span=span).run()
    scored = score(results.blocks, observed)
    return SimpleNamespace(
        name=name, loss=float(scored.loss), fidelity=float(scored.fidelity),
        score=scored, results=results, leftover=leftover_from_season(results),
    )


def best_registered_forecaster(year=okura.COMPARISON_YEAR, known=None, span="season", run=True):
    """Lowest IR/D loss among farmers registered at this moment.

    `known` may reuse Run & compare scores: {name: namespace(loss, fidelity, results)}.
    If nobody is registered outside the package, fall back to et_follower.
    """
    from .agents import leftover_from_season
    from .imitation import load_observed

    names = registered_forecasters() or ["et_match"]
    known = known or {}
    ranked = []
    observed = None
    for name in names:
        entry = known.get(name)
        if entry is not None and getattr(entry, "loss", None) is not None:
            if run and not _forecaster_fits_span(entry, span):
                entry = None
            else:
                if getattr(entry, "leftover", None) is None and getattr(entry, "results", None) is not None:
                    try:
                        entry.leftover = leftover_from_season(entry.results)
                    except Exception:
                        entry.leftover = None
                ranked.append(entry)
                continue
        if not run:
            ranked.append(SimpleNamespace(name=name, loss=None, fidelity=None,
                                          results=None, leftover=None))
            continue
        if observed is None:
            observed = load_observed(year)[0]
        try:
            ranked.append(score_forecaster(year, name, observed=observed, span=span))
        except Exception:
            continue
    if not ranked:
        return None
    scored = [row for row in ranked if getattr(row, "loss", None) is not None]
    if scored:
        scored.sort(key=lambda row: (row.loss, -(row.fidelity or 0.0)))
        return scored[0]
    if not run and ranked:
        ranked.sort(key=lambda row: row.name or "")
        return ranked[0]
    ranked.sort(key=lambda row: (float("inf") if row.loss is None else row.loss,
                                 -(row.fidelity or 0.0)))
    return ranked[0]


def freeze_forecaster(results, name="forecaster"):
    """Replay a finished farmer's irrigation and discharge every morning."""
    from .agents import FarmerPolicy

    model = results.model
    log = model.log or {}
    days, n = model.n_days, model.n
    ir = np.asarray(log.get("irrigation_mm", np.zeros((days, n))), dtype=float)
    dr = np.asarray(log.get("farmer_discharge_mm", log.get("drainage_mm", np.zeros((days, n)))),
                    dtype=float)

    class FrozenForecaster(FarmerPolicy):
        def reset(self, model, rows):
            self.rows = np.asarray(rows)

        def decide_batch(self, batch):
            day = int(np.clip(batch.day, 0, ir.shape[0] - 1))
            return ir[day, self.rows], dr[day, self.rows]

    FrozenForecaster.name = f"frozen:{name}"
    FrozenForecaster.__name__ = "FrozenForecaster"
    return FrozenForecaster()


def _reweight_shares_to_leftover(model, leftover, params):
    capture = float(params.get(
        "pool_capture_share", params.get("kx_ratio", 0.0) + params.get("ky_ratio", 0.0)))
    nbr = float(params.get("neighbour_of_remain",
                           getattr(model.physics, "neighbour_of_remain", 0.5)))
    stay_target = max(1.0 - capture, 0.0) * (1.0 - nbr)
    p = model.physics
    top = float(p.total_evaporable_mm / max(p.total_evaporable_mm + p.total_available_mm, 1e-9))
    shares = model.pixel_shares
    n_share = np.size(shares["kx_ratio"])
    pile = np.asarray(leftover, dtype=float)
    if pile.ndim == 2 and pile.shape[1] == n_share:
        leftover_weighted_stay(shares, pile, model.area, model.inputs.cyclic, stay_target, top)
    elif model.cells is not None and n_share == len(model.cells.owner) and pile.ndim == 2 and pile.shape[1] == model.n:
        owner = model.cells.owner
        leftover_weighted_stay(
            shares, pile[:, owner], model.area[owner] * model.cells.weights,
            np.asarray(model.inputs.cyclic[owner], dtype=bool), stay_target, top)


def _reshape_pixel_shares(pixel_shares, model):
    shares = {}
    owner = None if model.cells is None else model.cells.owner
    n_here = model.n if owner is None else len(owner)
    for key, value in pixel_shares.items():
        if key in ("ratio_split", "force_satellite_et") or str(key).endswith("_mean"):
            continue
        array = np.asarray(value, dtype=float)
        if array.ndim == 0 or array.size == 1:
            array = np.full(model.n, float(array))
        if array.ndim == 2:
            array = array.mean(axis=0)
        if owner is not None and array.ndim == 1 and array.size == model.n:
            shares[key] = array[owner]
        else:
            shares[key] = array
        if np.ndim(shares[key]) == 1 and np.size(shares[key]) not in (n_here, model.n):
            shares[key] = np.full(n_here, float(np.mean(shares[key])))
    return shares


def _reshape_pixel_daily(pixel_daily, model):
    owner = None if model.cells is None else model.cells.owner
    out = {}
    for key, value in pixel_daily.items():
        array = np.asarray(value, dtype=float)
        if array.ndim != 2:
            continue
        if owner is not None and array.shape[-1] == model.n:
            out[key] = array[:, owner]
        else:
            out[key] = array
    return out


def shares_from_registered(year, name, leftover=None):
    """Pixel leftover shares from a `@register_ratio` agent."""
    from .agents import build_batch, make_ratio, point_targets
    from . import data as _data

    agent = make_ratio(name)
    inputs = _data.season_inputs(year)
    n = len(inputs.paddy_id)
    pile = leftover
    last = np.zeros(n)
    if leftover is not None:
        pile = np.asarray(leftover, dtype=float)
        if pile.ndim == 2 and pile.shape[-1] == n:
            last = pile[-1]
        elif pile.ndim == 1 and pile.size == n:
            last = pile
        else:
            pile = None
    day = 0 if pile is None or np.ndim(pile) < 2 else int(np.clip(pile.shape[0] - 1, 0, len(inputs.dates) - 1))
    observation = build_batch(
        inputs, day, np.zeros(n), inputs.area_m2, leftover_mm=last)
    targets = point_targets(year, day=day, inputs=inputs)
    return agent.partition(pile, observation, targets)


def shares_from_daily_qrc(year, n, cyclic):
    """Broadcast the saved district QRC readout onto every cyclic paddy."""
    from .qrc import RATIO_NAMES, load_qrc_result

    loaded = load_qrc_result(year)
    if loaded is None:
        raise ValueError(
            f"No daily QRC save for {year}. Train it outside the UI: "
            f"python -m cyclic identify qrc classical {year}")
    constructed = construct_ratios(year)
    cyclic = np.asarray(cyclic, dtype=bool)
    mask = cyclic.astype(float)
    shares, daily = {}, {}
    for i, name in enumerate(RATIO_NAMES):
        mean = loaded.means.get(name, float(loaded.pred[:, i].mean()))
        shares[name] = mask * float(mean)
        daily[name] = loaded.pred[:, i:i + 1] * mask[None, :]
    shares["farm_to_main"] = mask * float(constructed["farm_to_main"])
    # QRC still emits a 4-way leftover simplex; treat Kx+Ky as LFV for that path.
    shares["pool_capture_share"] = shares["kx_ratio"] + shares["ky_ratio"]
    days = int(loaded.pred.shape[0])
    daily["farm_to_main"] = np.broadcast_to(shares["farm_to_main"][None, :], (days, n)).copy()
    daily["pool_capture_share"] = daily["kx_ratio"] + daily["ky_ratio"]
    return shares, daily, loaded


def shares_from_spatial_qrc(year):
    """Per-paddy leftover shares from the last spatial QRC save."""
    from .qrc import load_spatial_qrc

    loaded = load_spatial_qrc(year)
    if loaded is None:
        raise ValueError(
            f"No spatial QRC save for {year}. Train it outside the UI: "
            f"python -m cyclic identify qrc spatial classical {year}")
    daily = {key: np.asarray(val) for key, val in loaded.daily.items()}
    shares = {}
    for key, arr in daily.items():
        shares[key] = arr.mean(axis=0) if arr.ndim == 2 else np.asarray(arr)
    for key, val in (loaded.shares or {}).items():
        shares.setdefault(key, np.asarray(val))
    return shares, daily, loaded


def _finder_pixel_source(kind, name, year, span, leftover=None):
    """(pixel_shares, pixel_daily) for a selected leftover approach, or (None, None)."""
    if kind == "registered":
        if leftover is None and span != "tables":
            from .qrc import spatial_leftover_fields
            leftover = spatial_leftover_fields(year).leftover
        return shares_from_registered(year, name, leftover=leftover), None
    if kind == "qrc_daily":
        frame = _data.paddies()
        shares, daily, _loaded = shares_from_daily_qrc(
            year, len(frame), frame["cyclic"].to_numpy(dtype=bool))
        return shares, daily
    if kind == "qrc_spatial":
        shares, daily, _loaded = shares_from_spatial_qrc(year)
        return shares, daily
    return None, None


def run_finder(year=okura.COMPARISON_YEAR, span="season", n_eval=8, method="classical",
               n_nodes=12, quantum_backend="auto", force=False, ratio_agent=None,
               known_forecasters=None):
    """Learn leftover shares; keep the best on disk and reuse it next time.

    Irrigation and discharge come from the best currently registered forecaster
    (lowest block IR/D loss). Pass `known_forecasters` to reuse Run & compare scores.

    method  classical  random walk on district means
            ibm / qaoa / omalley  O'Malley 2018 QUBO (QAOA on IBM or exact/Aer)
            qrc_daily / qrc_spatial  apply a previously trained QRC save
            <registered name>  a `@register_ratio` agent from example.py
    force   ignore a saved best (start from Table 4–5 construct, always write)
    """
    import time
    from .canal import mean_pixel_shares

    t0 = time.time()
    kind, registered = parse_finder_method(method, ratio_agent)
    constructed = construct_ratios(year)
    quantum = None
    pixel_override = None
    pixel_daily = None
    sim_span = "year" if span == "year" else "season"
    farmer = best_registered_forecaster(
        year, known=known_forecasters, span=sim_span, run=(span != "tables"))
    leftover = None if farmer is None else getattr(farmer, "leftover", None)
    farmer_name = None if farmer is None else farmer.name
    farmer_loss = None if farmer is None else getattr(farmer, "loss", None)
    farmer_fid = None if farmer is None else getattr(farmer, "fidelity", None)
    agents = "ratio_finder"
    if farmer is not None and getattr(farmer, "results", None) is not None:
        agents = freeze_forecaster(farmer.results, name=farmer.name)
    if kind in ("ibm", "omalley"):
        from .quantum import solve_omalley_quantum
        backend = "ibm" if kind == "ibm" else quantum_backend
        quantum = solve_omalley_quantum(year, n_nodes=n_nodes, backend=backend)
        constructed = {**constructed, **quantum.physics}
        pixel_override = quantum.shares
    elif kind in ("registered", "qrc_daily", "qrc_spatial"):
        pixel_override, pixel_daily = _finder_pixel_source(
            kind, registered, year, span, leftover=leftover)
    if span == "tables":
        obj, signed = table_closure_objectives(year)
        problem = HydrologyProblem(year)
        loss = problem.scalar(obj) + float(obj.get("closure", 0.0))
        split = loss_split(obj, problem.weights, signed)
        maps = overlay_pixel_maps(field_share_maps(year, constructed), pixel_override)
        means = mean_pixel_shares(maps.fields, maps.area, maps.cyclic)
        params = {k: means.get(k, constructed[k]) for k in (*ACTIVE,)}
        params.setdefault("pool_capture_share",
                          float(means.get("pool_capture_share", constructed["pool_capture_share"])))
        params["kx_signed"] = means.get("kx_signed", 0.0)
        params["ky_signed"] = means.get("ky_signed", 0.0)
        saved = None
        if quantum is not None:
            saved = save_quantum_ratios(
                year, quantum, maps=maps, params=params, loss=loss, signed=signed,
                split=split, span=span)
        return SimpleNamespace(
            params=params, objectives=obj, signed=signed, split=split, loss=loss, maps=maps,
            seconds=time.time() - t0, span=span, days=0, year=year, saved=saved,
            quantum=quantum, method=kind, ratio_agent=registered,
            forecaster=farmer_name, forecaster_loss=farmer_loss, forecaster_fidelity=farmer_fid,
        )
    saved = None if (force or pixel_override is not None) else load_best_ratios(year, sim_span)
    start_params = {**constructed, **(saved.params if saved is not None else {})}
    problem = HydrologyProblem(year, physics=Physics().replace(**start_params), agents=agents)
    problem.leftover_mm = leftover
    if pixel_override is not None:
        out = problem.evaluate(problem.pack(), days=None, save_csv=False, record_paddies=True,
                               potential="pixel", span=sim_span, pixel_shares=pixel_override,
                               pixel_shares_daily=pixel_daily, leftover_mm=leftover)
        learned = SimpleNamespace(best=out, history=[out])
    else:
        learned = solve_classical(
            problem, n_eval=max(int(n_eval), 1), start=problem.pack(),
            potential="pixel", span=sim_span)
    out = learned.best
    maps = overlay_pixel_maps(field_share_maps(year, out.params, results=out.results), pixel_override)
    means = mean_pixel_shares(maps.fields, maps.area, maps.cyclic)
    params = {**out.params, **{k: means.get(k, out.params[k]) for k in ACTIVE if k in means}}
    if "pool_capture_share" in means:
        params["pool_capture_share"] = float(means["pool_capture_share"])
    else:
        params.setdefault(
            "pool_capture_share",
            float(constructed.get("pool_capture_share",
                                  params.get("kx_ratio", 0.0) + params.get("ky_ratio", 0.0))))
    params["kx_signed"] = means.get("kx_signed", 0.0)
    params["ky_signed"] = means.get("ky_signed", 0.0)
    skip_best = kind in ("registered", "qrc_daily", "qrc_spatial")
    if skip_best:
        saved_note = None
    else:
        path, wrote = save_best_ratios(
            year, sim_span, params, fields=maps.fields, loss=out.loss,
            signed=out.signed, split=out.split, force=force)
        saved_note = path if wrote else f"{path} (kept previous, loss {saved.loss if saved else '—'})"
    if quantum is not None:
        saved_note = save_quantum_ratios(
            year, quantum, maps=maps, params=params, loss=out.loss, signed=out.signed,
            split=out.split, span=span)
    return SimpleNamespace(
        params=params, objectives=out.objectives, signed=out.signed, split=out.split,
        loss=out.loss, maps=maps,
        seconds=time.time() - t0, span=span, days=out.results.model.n_days, year=year,
        saved=saved_note,
        trials=len(learned.history),
        quantum=quantum, method=kind, ratio_agent=registered,
        forecaster=farmer_name, forecaster_loss=farmer_loss, forecaster_fidelity=farmer_fid,
    )


def discover_pixel_shares(year, method="tables", leftover=None, results=None,
                          agents="field_aggregate", potential="field", simulate=False):
    """Per-paddy leftover simplex using only `year`. Does not look at the other year.

    tables       Table 5 capture pin (LFV/LfA) + canal / Fig. 6 spatial shape
    stay         same, then leftover-weighted stay (needs leftover or a season)
    classical    saved `best_{year}_season.npz` field, else tables
    """
    method = str(method).lower().replace("-", "_")
    pin = construct_ratios(year)
    from .canal import pixel_ratio_fields, distance_to_canal_m
    from . import data as _data
    from .lateral import knn_edges
    frame = _data.paddies()
    lon = frame["lon"].to_numpy()
    lat = frame["lat"].to_numpy()
    cyclic = frame["cyclic"].to_numpy(dtype=bool)
    area = (frame["area_m2"].to_numpy() if "area_m2" in frame.columns
            else np.ones(len(frame), dtype=float))
    block = frame["block"].to_numpy() if "block" in frame.columns else None
    p = Physics()
    top = float(p.total_evaporable_mm /
                max(p.total_evaporable_mm + p.total_available_mm, 1e-9))
    intensity = None
    try:
        from .paper_plots import block_leftover_intensity
        intensity = block_leftover_intensity(year)
    except Exception:
        intensity = None
    fields = pixel_ratio_fields(
        len(frame), distance_to_canal_m(lon, lat), knn_edges(lon, lat, k=4),
        cyclic, float(pin["pool_capture_share"]), float(pin["farm_to_main"]),
        top, float(p.canal_length_m),
        area_m2=area, block=block, block_intensity=intensity, hard_capture=False,
        neighbour_of_remain=float(pin.get("neighbour_of_remain",
                                          getattr(p, "neighbour_of_remain", 0.5))))
    maps = SimpleNamespace(area=area, cyclic=cyclic, fields=fields)
    params = dict(pin)

    if method in ("classical", "fitted", "best"):
        saved = load_best_ratios(year, "season")
        if saved is not None and saved.fields:
            return SimpleNamespace(fields=saved.fields, params=saved.params, area=maps.area,
                                   cyclic=maps.cyclic, year=int(year), method=method)
        if saved is not None:
            maps = field_share_maps(year, saved.params, results=results)
            return SimpleNamespace(fields=maps.fields, params=saved.params, area=maps.area,
                                   cyclic=maps.cyclic, year=int(year), method=method)

    if method in ("stay", "frozen_stay", "spatial"):
        pile = leftover
        if pile is None and results is not None:
            from .agents import leftover_from_season
            pile = leftover_from_season(results)
        if pile is None and simulate:
            from .model import CyclicModel
            from .agents import leftover_from_season
            model = CyclicModel(year, agents=agents, potential=potential,
                                record_paddies=False, save_csv=False)
            results = model.run()
            pile = leftover_from_season(results)
            attach_pixel_shares(model, pin)
            fields = {key: np.array(val, copy=True) for key, val in model.pixel_shares.items()}
            maps = SimpleNamespace(area=model.area, cyclic=np.asarray(model.inputs.cyclic, dtype=bool),
                                   fields=fields)
        if pile is not None:
            stay = float(pin["kz_ratio"] + pin["ground_ratio"])
            leftover_weighted_stay(fields, pile, maps.area, maps.cyclic, stay, top)

    elif method not in ("tables", "table", "classical", "fitted", "best"):
        raise ValueError(f"unknown leftover discoverer {method!r}; "
                         "use tables, stay, or classical")

    return SimpleNamespace(fields=fields, params=params, area=maps.area, cyclic=maps.cyclic,
                           year=int(year), method=method,
                           table_capture=float(pin["pool_capture_share"]))


def ratio_holdout(method="tables", train_year=2021, eval_year=None, simulate=False,
                  agents="field_aggregate", potential="field"):
    """Fit a leftover-share field on one year, freeze it, score the other year.

    Ranking is the held-out year, not the training loss. `simulate=False` reports
    whether last year's area-weighted capture describes this year's Table 5 pin
    (no season run). `simulate=True` replays Okura IR/D on the eval year with the
    frozen K_i and scores leftover / LFV / ET.
    """
    from .canal import mean_pixel_shares
    from .validation import split

    train_year, eval_year = split(train_year, eval_year)
    train = discover_pixel_shares(train_year, method, agents=agents, potential=potential,
                                  simulate=simulate)
    eval_target = construct_ratios(eval_year)
    means = mean_pixel_shares(train.fields, train.area, train.cyclic)
    train_capture = float(means.get("pool_capture_share",
                                    means.get("kx_ratio", 0.0) + means.get("ky_ratio", 0.0)))
    eval_capture = float(eval_target["pool_capture_share"])
    scored = None
    if simulate:
        problem = HydrologyProblem(eval_year)
        scored = problem.evaluate(
            problem.pack(), pixel_shares=train.fields, potential=potential,
            record_paddies=False, save_csv=False)
    return SimpleNamespace(
        method=method, train_year=int(train_year), eval_year=int(eval_year),
        train_capture=train_capture, eval_capture=eval_capture,
        capture_error=float((train_capture - eval_capture) ** 2),
        fields=train.fields, params=train.params, simulate=bool(simulate),
        eval=scored,
        loss=None if scored is None else scored.loss,
        objectives=None if scored is None else scored.objectives,
        stage_frozen=True,
    )


def soil_flow_holdout(train_year=2021, eval_year=None, counterfactual="et_follower",
                      equation="head_equalise"):
    """Discover a neighbour-head lateral equation without soil properties.

    The hydrology object is ΔL = H(h_i, {h_j, ET_j of neighbours}). That is an
    ML problem: discover a linear or nonlinear equation in those features, freeze
    it, and score it on the other historical year. H may see how much water each
    neighbour holds and how much FAO-56 ET that neighbour used. It does not take
    soil K or the applied IR share. `equation` is a `@register_lateral` name:
    `head_equalise` (physics baseline), `linear`, or `nonlinear`. H is evaluated
    at every paddy and reads that paddy's up, down, left and right neighbours
    (north, south, west, east). A leftover *share* of IR+R−D−ET is not this
    object (`ratio_holdout`). A win of linear or nonlinear over head-equalise
    is not claimed.

    Counterfactual IR uses the same Table 4 season totals with the *other*
    year's Table 2 climate day-shape (`season_targets.climate_counterfactual_heads`).
    """
    import inspect

    from .agents import make_lateral
    from .graph import hop_stock, leftover_graph
    from .lateral import cardinal_edges, neighbour_et_features, neighbour_head_features
    from .season_targets import climate_counterfactual_heads
    from .validation import split

    train_year, eval_year = split(train_year, eval_year)
    law = make_lateral(equation)
    forbidden = {"irrigation", "irrigation_mm", "irrigation_sent_mm", "ksat", "k_soil",
                 "soil_k", "plow_pan_k", "conductivity"}
    flux_args = set(inspect.signature(law.flux).parameters)
    lon = np.array([140.267, 140.267, 140.267, 140.267])
    lat = np.array([35.818, 35.814, 35.810, 35.807])
    dist = np.array([1200.0, 800.0, 400.0, 0.0])
    area = np.full(4, 1000.0)
    edges = cardinal_edges(lon, lat)
    graph = leftover_graph(dist, edges, np.ones(4, dtype=bool), one_way=False)
    physics = Physics()
    # Historical stand-ins: 2021 wet leftover piled upslope; 2022 drier, smaller pile.
    scenarios = {
        int(train_year): np.array([80.0, 10.0, 0.0, 0.0]),
        int(eval_year): np.array([25.0, 8.0, 4.0, 1.0]),
    }
    # Same season totals, other-year climate shape → different leftover heads (no Fig. 8).
    cf_fallback = np.array([5.0, 5.0, 50.0, 10.0])
    try:
        cf_heads = climate_counterfactual_heads(
            int(train_year), shape_from_year=int(eval_year),
            n_plots=4, mode=str(counterfactual))
        if not np.all(np.isfinite(cf_heads)) or float(np.max(cf_heads)) < 1.0:
            cf_heads = cf_fallback
    except Exception:
        cf_heads = cf_fallback
    farmer_heads = {
        "okura": scenarios[int(train_year)],
        str(counterfactual): np.asarray(cf_heads, dtype=float),
    }

    def _et_from_heads(heads):
        return np.clip(np.asarray(heads, dtype=float) / 10.0, 0.5, 8.0)

    def _follow_head(heads, after):
        wetter = int(np.argmax(heads))
        if not (after[wetter] < heads[wetter] - 1e-9 and after.size == heads.size):
            return False
        for name in ("east", "west", "north", "south"):
            j = int(np.asarray(getattr(graph, name, [-1]))[wetter])
            if j < 0:
                continue
            if heads[j] < heads[wetter] - 1e-9 and after[j] < heads[j] - 1e-6:
                return False
        return True

    def _score(heads, et_mm):
        et_mean, _ = neighbour_et_features(et_mm, edges)
        after, delta = law.flux(heads, graph, area, physics, et_mm=et_mm)
        follow_head = _follow_head(heads, after)
        et_corr = float(np.corrcoef(delta, et_mm)[0, 1])
        return after, delta, follow_head, et_corr, et_mean

    train_heads = scenarios[int(train_year)]
    if callable(getattr(law, "fit", None)):
        # Teacher is the hop operator (plot-scale ΔL is unobserved in Okura).
        # Include a same-heads / different-ET case so neighbour ET is not collinear
        # with head and the fit is not forced to send leftover toward high ET.
        fit_heads = [train_heads, 0.6 * train_heads + 5.0, train_heads]
        fit_ets = [_et_from_heads(train_heads), _et_from_heads(0.6 * train_heads + 5.0),
                   np.full(4, 3.0)]
        stocks, ets, deltas = [], [], []
        for heads, et_mm in zip(fit_heads, fit_ets):
            _, delta = hop_stock(heads, graph, area, physics)
            stocks.append(heads)
            ets.append(et_mm)
            deltas.append(delta)
        law.fit(stocks, ets, deltas, graph=graph)

    train_et = _et_from_heads(scenarios[int(train_year)])
    eval_et = _et_from_heads(scenarios[int(eval_year)])
    cf_et = _et_from_heads(farmer_heads[str(counterfactual)])
    train_after, _, train_head, train_et_corr, train_n_et = _score(
        scenarios[int(train_year)], train_et)
    eval_after, _, eval_head, eval_et_corr, _ = _score(
        scenarios[int(eval_year)], eval_et)
    _, _, cf_head, cf_et_corr, _ = _score(
        farmer_heads[str(counterfactual)], cf_et)
    same_again, _, _, _, _ = _score(scenarios[int(train_year)], train_et)
    mean_depth, drop = neighbour_head_features(scenarios[int(train_year)], edges)
    return SimpleNamespace(
        train_year=int(train_year), eval_year=int(eval_year),
        equation=str(getattr(law, "name", equation)),
        form=str(getattr(law, "form", "physics")),
        hypothesis_class=str(getattr(law, "form", "physics")),
        ml_problem=True,
        fitted=bool(getattr(law, "fitted", False)),
        n_parameters=0 if getattr(law, "weights", None) is None else int(np.size(law.weights)),
        uses_soil_properties=bool(getattr(law, "uses_soil_properties", False)),
        sees_neighbour_water=bool(getattr(law, "sees_neighbour_water", True)),
        sees_neighbour_et=bool(getattr(law, "sees_neighbour_et", True)),
        applied_share_known=False,
        leftover_is_residual=True,
        historical_scenarios=(int(train_year), int(eval_year)),
        counterfactual_ir=str(counterfactual),
        counterfactual_heads=np.asarray(farmer_heads[str(counterfactual)], dtype=float),
        climate_shaped_cf=True,
        farmer_invariant=bool(np.allclose(train_after, same_again)
                              and not (flux_args & forbidden)
                              and not bool(getattr(law, "uses_soil_properties", False))),
        hops_follow_head=bool(train_head and eval_head and cf_head),
        hops_follow_et=bool(train_et_corr > 0.2 or eval_et_corr > 0.2 or cf_et_corr > 0.2),
        per_paddy=True,
        n_paddies=int(train_after.size),
        checks_cardinal_neighbours=bool(
            int(np.asarray(getattr(graph, "north", [-1]))[1]) >= 0
            and int(np.asarray(getattr(graph, "south", [-1]))[1]) >= 0),
        directions=("up", "down", "left", "right"),
        neighbour_head_drop_mm=drop,
        neighbour_depth_mm=mean_depth,
        neighbour_et_mm=train_n_et,
        capture_train=float(construct_ratios(train_year)["pool_capture_share"]),
        capture_eval=float(construct_ratios(eval_year)["pool_capture_share"]),
    )


def discover_lateral_equation(equation=None, train_year=2021, eval_year=None,
                              counterfactual="et_follower"):
    """One per-paddy UDLR equation H; linear and nonlinear are forms, not two models.

    `equation=None` (default) fits both forms on the training year, freezes them,
    and scores each on the other year and a counterfactual farmer IR map, against
    the physics baseline `head_equalise`. Pass `linear`, `nonlinear`, or
    `head_equalise` to score a single form. Do not retune to leftover share.

    The 4-plot demo does not read season caches. Whole-year 2021→2022 holdout
    is `equation_from_cache(train_year=2021, eval_year=2022)`.
    """
    if equation in (None, "", "both", "compare"):
        physics = soil_flow_holdout(train_year=train_year, eval_year=eval_year,
                                    counterfactual=counterfactual, equation="head_equalise")
        linear = soil_flow_holdout(train_year=train_year, eval_year=eval_year,
                                   counterfactual=counterfactual, equation="linear")
        nonlinear = soil_flow_holdout(train_year=train_year, eval_year=eval_year,
                                      counterfactual=counterfactual, equation="nonlinear")
        return SimpleNamespace(
            two_separate_models=False,
            one_equation=True,
            forms=("linear", "nonlinear"),
            baseline="head_equalise",
            train_year=physics.train_year,
            eval_year=physics.eval_year,
            per_paddy=True,
            checks_cardinal_neighbours=bool(physics.checks_cardinal_neighbours),
            directions=physics.directions,
            physics=physics,
            linear=linear,
            nonlinear=nonlinear,
            hops_follow_head=bool(linear.hops_follow_head and nonlinear.hops_follow_head
                                  and physics.hops_follow_head),
            hops_follow_et=bool(linear.hops_follow_et or nonlinear.hops_follow_et
                                or physics.hops_follow_et),
        )
    return soil_flow_holdout(train_year=train_year, eval_year=eval_year,
                             counterfactual=counterfactual, equation=equation)


# --- whole-year cache: fit H on 2021 leftover/ET, freeze, score 2022 -----------------------

DAILY_FIELD_COLS = ("date", "paddy_id", "leftover_mm", "et_mm", "neighbor_mm",
                    "stay_mm", "infiltration_mm", "flow_kx_mm", "flow_ky_mm",
                    "flow_kz_mm", "flow_ground_mm", "irrigation_mm", "drainage_mm",
                    "rain_mm", "et_target_mm", "et_potential_mm", "et_groundwater_mm")
SHARE_KEYS = ("kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio")


def _outputs_root(folder=None):
    import os

    return folder or os.path.join(_data.PROJECT, "outputs")


def _lateral_dir(folder=None):
    import os

    path = os.path.join(_outputs_root(folder), "lateral")
    os.makedirs(path, exist_ok=True)
    return path


def linear_term_names(extra_names=()):
    from .lateral import CARDINAL, CARDINAL_LABEL

    return (("intercept",)
            + tuple(f"drop_{CARDINAL_LABEL[name]}" for name in CARDINAL)
            + tuple(f"et_{CARDINAL_LABEL[name]}" for name in CARDINAL)
            + tuple(extra_names))


def linear_equation_text(weights, extra_names=()):
    """Frozen linear neighbour-trickle law."""
    names = linear_term_names(extra_names)
    return affine_equation_text("dL_i (neighbour trickle)", weights, names)


def affine_equation_text(left_hand, weights, names):
    names = tuple(names)
    weights = np.asarray(weights, dtype=float).ravel()
    if weights.size > len(names):
        names = names + tuple(f"x{i}" for i in range(len(names), weights.size))
    if weights.size == 0:
        return f"{left_hand} = (unfitted)"
    parts = [f"{weights[0]:+.4g}"]
    for name, value in zip(names[1:], weights[1:]):
        parts.append(f"{value:+.4g}*{name}")
    return f"{left_hand} = " + " ".join(parts)


def leftover_equations_text(hop_weights, extra_names=(), pool_weights=None,
                            stay_weights=None, fate_names=()):
    """Three leftover equations: neighbour trickle, recapture to the pool, soil stay."""
    lines = [linear_equation_text(hop_weights, extra_names=extra_names)]
    if pool_weights is not None:
        lines.append(affine_equation_text("to_pool_i (recapture / LFV)", pool_weights, fate_names))
    if stay_weights is not None:
        lines.append(affine_equation_text("stay_i (ground soil)", stay_weights, fate_names))
    return "\n".join(lines)


def leftover_fate_text(weights, extra_names=(), capture=None, pool_weights=None,
                       stay_weights=None, fate_names=()):
    if pool_weights is not None or stay_weights is not None:
        return leftover_equations_text(weights, extra_names=extra_names,
                                       pool_weights=pool_weights, stay_weights=stay_weights,
                                       fate_names=fate_names)
    hop = linear_equation_text(weights, extra_names=extra_names)
    if capture is None:
        return hop
    capture = float(capture)
    stay = 1.0 - capture
    return (
        hop
        + f"\nto_pool_i (LFV) = leftover_i * {capture:.4f}   # pool_capture_share; same-day collection"
        + f"\nstay_i (ground soil) = leftover_i * {stay:.4f}   # Kz+ground (+ neighbour remainder handled separately)"
    )


def _sample_days(n_days, sample_days=16):
    n_days = int(n_days)
    if n_days <= 0:
        return np.zeros(0, dtype=int)
    if sample_days is None or int(sample_days) >= n_days:
        return np.arange(n_days, dtype=int)
    return np.unique(np.linspace(0, n_days - 1, int(sample_days)).astype(int))


def _rmse(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return float(np.sqrt(np.mean((left - right) ** 2)))


def _season_geometry(path):
    frame = pd.read_csv(path)
    ids = frame["paddy_id"].astype(str).to_numpy()
    area = (frame["area_m2"].to_numpy(dtype=float) if "area_m2" in frame.columns
            else frame["area_ha"].to_numpy(dtype=float) * 10000.0)
    dist = (frame["distance_m"].to_numpy(dtype=float) if "distance_m" in frame.columns
            else np.zeros(len(frame)))
    cyclic = frame["cyclic"].to_numpy()
    if cyclic.dtype != bool:
        cyclic = np.asarray([str(v).strip().lower() in ("1", "true", "yes") for v in cyclic])
    return SimpleNamespace(
        paddy_id=ids,
        lon=frame["lon"].to_numpy(dtype=float),
        lat=frame["lat"].to_numpy(dtype=float),
        area=area,
        cyclic=np.asarray(cyclic, dtype=bool),
        dist=dist,
        et_season=frame["et_mm"].to_numpy(dtype=float) if "et_mm" in frame.columns else None,
        leftover_season=(frame["leftover_mm"].to_numpy(dtype=float)
                         if "leftover_mm" in frame.columns else None),
        stay_season=(frame["stay_mm"].to_numpy(dtype=float) if "stay_mm" in frame.columns
                     else (frame["deep_mm"].to_numpy(dtype=float) if "deep_mm" in frame.columns
                           else None)),
    )


def _field(year, leftover, et, neighbor, geometry, dates, source, shares=None,
           infiltration=None):
    leftover = np.asarray(leftover, dtype=np.float32)
    et = np.asarray(et, dtype=np.float32)
    if neighbor is not None:
        neighbor = np.asarray(neighbor, dtype=np.float32)
        if not np.isfinite(neighbor).any() or float(np.nanmax(np.abs(neighbor))) < 1e-9:
            neighbor = None
    if shares is not None:
        shares = np.asarray(shares, dtype=np.float32)
    if infiltration is not None:
        infiltration = np.asarray(infiltration, dtype=np.float32)
        if not np.isfinite(infiltration).any() or float(np.nanmax(np.abs(infiltration))) < 1e-9:
            infiltration = None
    stay_frac = None
    leftover_season = getattr(geometry, "leftover_season", None)
    stay_season = getattr(geometry, "stay_season", None)
    if stay_season is not None:
        pile = leftover_season if leftover_season is not None else leftover.sum(axis=0)
        stay_frac = np.clip(np.asarray(stay_season, dtype=float)
                            / np.maximum(np.asarray(pile, dtype=float), 1e-6), 0.0, 1.0)
    field = SimpleNamespace(
        year=int(year), leftover=leftover, et=et, neighbor=neighbor, shares=shares,
        infiltration=infiltration, stay_frac=stay_frac,
        lon=np.asarray(geometry.lon, dtype=float), lat=np.asarray(geometry.lat, dtype=float),
        area=np.asarray(geometry.area, dtype=float), cyclic=np.asarray(geometry.cyclic, dtype=bool),
        dist=np.asarray(geometry.dist, dtype=float), paddy_id=np.asarray(geometry.paddy_id),
        dates=pd.to_datetime(dates), source=str(source), n_days=int(leftover.shape[0]),
        n=int(leftover.shape[1]),
    )
    if field.shares is None:
        field.shares = leftover_share_teacher(field)
    return field


def leftover_share_teacher(field):
    """Per-paddy leftover simplex: infiltrated leftover scaled to Table 5 LFV_pump."""
    if getattr(field, "shares", None) is not None:
        return np.asarray(field.shares, dtype=np.float32)
    leftover, to_pool, stay, pin, _groundwater = pin_capture_to_lfv(
        field.leftover, field.area, field.cyclic, field.year,
        infiltration_mm=getattr(field, "infiltration", None),
        neighbor_mm=getattr(field, "neighbor", None),
        et_target_mm=getattr(field, "et_target", None),
        et_mm=getattr(field, "et", None))
    field.pin = pin
    field.to_pool = np.asarray(to_pool, dtype=np.float32)
    field.stay_mm = np.asarray(stay, dtype=np.float32)
    return _shares_from_pool_stay(leftover, to_pool, stay, field.cyclic)


def _field_npz_path(year, folder=None):
    import os

    return os.path.join(_lateral_dir(folder), f"field_{int(year)}.npz")


FIELD_FORCING = ("irrigation", "drainage", "rain", "et_target", "et_potential", "et_groundwater",
                 "groundwater_for_et")


def save_year_field(field, folder=None):
    import os

    path = _field_npz_path(field.year, folder)
    payload = dict(
        leftover=np.asarray(field.leftover, dtype=np.float32),
        et=np.asarray(field.et, dtype=np.float32),
        lon=np.asarray(field.lon, dtype=float),
        lat=np.asarray(field.lat, dtype=float),
        area=np.asarray(field.area, dtype=float),
        cyclic=np.asarray(field.cyclic, dtype=bool),
        dist=np.asarray(field.dist, dtype=float),
        paddy_id=np.asarray(field.paddy_id).astype(str),
        dates=np.asarray(pd.to_datetime(field.dates).strftime("%Y-%m-%d")),
        year=int(field.year),
        source=str(field.source),
    )
    if field.neighbor is not None:
        payload["neighbor"] = np.asarray(field.neighbor, dtype=np.float32)
    if getattr(field, "shares", None) is not None:
        payload["shares"] = np.asarray(field.shares, dtype=np.float32)
    if getattr(field, "infiltration", None) is not None:
        payload["infiltration"] = np.asarray(field.infiltration, dtype=np.float32)
    if getattr(field, "stay_frac", None) is not None:
        payload["stay_frac"] = np.asarray(field.stay_frac, dtype=float)
    # Forcing kept with the field so the cache fully replaces the run's paddy_daily.csv.
    for name in FIELD_FORCING:
        if getattr(field, name, None) is not None:
            payload[name] = np.asarray(getattr(field, name), dtype=np.float32)
    np.savez_compressed(path, **payload)
    return path


def load_year_field_npz(year, folder=None):
    import os

    path = _field_npz_path(year, folder)
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=True) as data:
        neighbor = np.asarray(data["neighbor"]) if "neighbor" in data.files else None
        infiltration = np.asarray(data["infiltration"]) if "infiltration" in data.files else None
        geometry = SimpleNamespace(
            paddy_id=np.asarray(data["paddy_id"]).astype(str),
            lon=np.asarray(data["lon"], dtype=float),
            lat=np.asarray(data["lat"], dtype=float),
            area=np.asarray(data["area"], dtype=float),
            cyclic=np.asarray(data["cyclic"], dtype=bool),
            dist=np.asarray(data["dist"], dtype=float),
            leftover_season=None,
            stay_season=(np.asarray(data["stay_frac"], dtype=float)
                         * np.maximum(np.asarray(data["leftover"], dtype=float).sum(axis=0), 1e-6)
                         if "stay_frac" in data.files else None),
        )
        field = _field(int(data["year"]), data["leftover"], data["et"], neighbor, geometry,
                       data["dates"], str(data["source"]), shares=None,
                       infiltration=infiltration)
        for name in FIELD_FORCING:
            if name in data.files:
                setattr(field, name, np.asarray(data[name], dtype=np.float32))
        return field


def _newest_run_daily(year, folder):
    import os

    runs = os.path.join(folder, "runs")
    if not os.path.isdir(runs):
        return None
    found = []
    prefix = f"{int(year)}_"
    for name in os.listdir(runs):
        if not name.startswith(prefix):
            continue
        path = os.path.join(runs, name, "paddy_daily.csv")
        if os.path.isfile(path):
            found.append(path)
    if not found:
        return None
    return max(found, key=os.path.getmtime)


def _pivot_daily(frame, ids, column):
    if column not in frame.columns:
        return None
    wide = frame.pivot_table(index="date", columns="paddy_id", values=column, aggfunc="first")
    wide.columns = wide.columns.astype(str)
    return wide.reindex(columns=list(ids)).fillna(0.0).to_numpy(dtype=np.float32)


def _read_run_paddy_daily(daily_path, season_path, year, progress=None):
    import os

    if progress:
        progress(f"Reading cached {year} paddy_daily.csv ...")
    geometry = _season_geometry(season_path)
    usecols = [name for name in DAILY_FIELD_COLS]
    header = pd.read_csv(daily_path, nrows=0).columns.tolist()
    usecols = [name for name in usecols if name in header]
    daily = pd.read_csv(daily_path, usecols=usecols, parse_dates=["date"])
    daily["paddy_id"] = daily["paddy_id"].astype(str)
    leftover = _pivot_daily(daily, geometry.paddy_id, "leftover_mm")
    et = _pivot_daily(daily, geometry.paddy_id, "et_mm")
    neighbor = _pivot_daily(daily, geometry.paddy_id, "neighbor_mm")
    irrigation = _pivot_daily(daily, geometry.paddy_id, "irrigation_mm")
    drainage = _pivot_daily(daily, geometry.paddy_id, "drainage_mm")
    rain = _pivot_daily(daily, geometry.paddy_id, "rain_mm")
    et_target = _pivot_daily(daily, geometry.paddy_id, "et_target_mm")
    et_potential = _pivot_daily(daily, geometry.paddy_id, "et_potential_mm")
    et_groundwater = _pivot_daily(daily, geometry.paddy_id, "et_groundwater_mm")
    if leftover is None or et is None:
        raise ValueError(f"{daily_path} needs leftover_mm and et_mm columns")
    if irrigation is not None and drainage is not None and rain is not None and et_target is not None:
        leftover = leftover_pinned(irrigation, rain, drainage, et_target).astype(np.float32)
    dates = np.sort(pd.to_datetime(daily["date"]).unique())
    if leftover.shape[0] != dates.size:
        leftover_wide = daily.pivot_table(
            index="date", columns="paddy_id", values="leftover_mm", aggfunc="first")
        dates = pd.to_datetime(leftover_wide.index)
    infiltration = _pivot_daily(daily, geometry.paddy_id, "infiltration_mm")
    source = os.path.relpath(daily_path, _data.PROJECT).replace("\\", "/")
    field = _field(year, leftover, et, neighbor, geometry, dates, source, shares=None,
                   infiltration=infiltration)
    return pin_field_to_et_and_lfv(
        field, irrigation=irrigation, rain=rain, drainage=drainage, et_target=et_target,
        et_mm=et, et_potential=et_potential, et_groundwater=et_groundwater)


def _reconstruct_from_wide(year, folder, progress=None):
    import os

    ir_path = os.path.join(folder, f"paddy_daily_irrigation_mm_{int(year)}.csv")
    d_path = os.path.join(folder, f"paddy_daily_drainage_mm_{int(year)}.csv")
    season_path = os.path.join(folder, f"paddies_season_{int(year)}.csv")
    missing = [path for path in (ir_path, d_path, season_path) if not os.path.isfile(path)]
    if missing:
        raise ValueError(
            f"No cached {year} leftover field. Need a run paddy_daily.csv under "
            f"outputs/runs/{year}_*/ or {os.path.basename(ir_path)}, "
            f"{os.path.basename(d_path)} and {os.path.basename(season_path)}.")
    if progress:
        progress(f"Reconstructing {year} leftover from cached IR, D, rain and season ET ...")
    geometry = _season_geometry(season_path)
    ids = list(geometry.paddy_id)
    irrigation = pd.read_csv(ir_path, index_col=0, parse_dates=True)
    drainage = pd.read_csv(d_path, index_col=0, parse_dates=True)
    irrigation.columns = irrigation.columns.astype(str)
    drainage.columns = drainage.columns.astype(str)
    irrigation = irrigation.reindex(columns=ids).fillna(0.0)
    drainage = drainage.reindex(columns=ids).fillna(0.0)
    drainage = drainage.reindex(irrigation.index).fillna(0.0)
    dates = irrigation.index
    wx = _data.weather(int(year)).reindex(dates)
    rain = np.nan_to_num(_data.station_rain(int(year), wx.fillna(0.0)))
    et0 = np.nan_to_num(wx["et0_mm"].to_numpy(dtype=float))
    if geometry.et_season is None:
        et = np.clip(et0[:, None] * 1.1, 0.0, None)
    else:
        scale = float(et0.sum()) or 1.0
        et = geometry.et_season[None, :] * (et0[:, None] / scale)
    leftover = leftover_pinned(irrigation.to_numpy(dtype=float), rain,
                               drainage.to_numpy(dtype=float), et)
    source = (f"reconstructed leftover = IR + R − D − satellite ET from "
              f"{os.path.basename(ir_path)} / {os.path.basename(d_path)}")
    field = _field(year, leftover, et, None, geometry, dates, source)
    rain_2d = np.broadcast_to(rain[:, None], leftover.shape)
    return pin_field_to_et_and_lfv(
        field, irrigation=irrigation.to_numpy(dtype=float), rain=rain_2d,
        drainage=drainage.to_numpy(dtype=float), et_target=et, et_mm=et)


def equation_cache_status(year, folder=None):
    """What leftover/ET cache exists for this year (npz, run CSV, or IR/D reconstruction)."""
    import os

    year = int(year)
    folder = _outputs_root(folder)
    npz = _field_npz_path(year, folder)
    daily = _newest_run_daily(year, folder)
    ir_path = os.path.join(folder, f"paddy_daily_irrigation_mm_{year}.csv")
    d_path = os.path.join(folder, f"paddy_daily_drainage_mm_{year}.csv")
    season = os.path.join(folder, f"paddies_season_{year}.csv")
    if daily:
        source = os.path.relpath(daily, _data.PROJECT).replace("\\", "/") + " (leftover pinned to satellite ET, capture to LFV)"
    elif os.path.isfile(npz):
        source = "compact leftover/ET npz (capture scaled to LFV)"
    elif os.path.isfile(ir_path) and os.path.isfile(d_path) and os.path.isfile(season):
        source = f"IR/D CSVs + {os.path.basename(season)} (leftover = IR+R−D−ET, capture to LFV)"
    else:
        source = None
    return SimpleNamespace(
        year=year,
        npz=npz if os.path.isfile(npz) else None,
        paddy_daily=daily,
        irrigation=ir_path if os.path.isfile(ir_path) else None,
        drainage=d_path if os.path.isfile(d_path) else None,
        season=season if os.path.isfile(season) else None,
        usable=source is not None,
        source=source or f"no cached {year} leftover field",
    )


def load_year_field(year, folder=None, progress=None, compact=True):
    """Leftover heads and ET for one cached year, as arrays (days × paddies)."""
    import os

    year = int(year)
    folder = _outputs_root(folder)
    daily = _newest_run_daily(year, folder)
    if daily:
        season = os.path.join(os.path.dirname(daily), "paddies_season.csv")
        if not os.path.isfile(season):
            season = os.path.join(folder, f"paddies_season_{year}.csv")
        field = _read_run_paddy_daily(daily, season, year, progress=progress)
    else:
        loaded = load_year_field_npz(year, folder)
        if loaded is not None:
            return pin_field_to_et_and_lfv(loaded)
        field = _reconstruct_from_wide(year, folder, progress=progress)
    if compact:
        save_year_field(field, folder)
    return field


def _field_graph(field):
    from .graph import leftover_graph
    from .lateral import cardinal_edges

    edges = cardinal_edges(field.lon, field.lat)
    return leftover_graph(field.dist, edges, field.cyclic, one_way=False)


def _hop_teacher(leftover, graph, area, physics, days):
    from .graph import hop_stock

    rows = []
    stock = np.asarray(leftover, dtype=float)
    for day in days:
        _, delta = hop_stock(np.maximum(stock[day], 0.0), graph, area, physics)
        rows.append(delta)
    return np.vstack(rows)


def _day_slices(array, days):
    if array is None:
        return [None] * len(days)
    array = np.asarray(array, dtype=float)
    return [np.clip(array[day], 0.0, None) for day in days]


def _flux_days(law, leftover, et, graph, area, physics, days, infiltration=None,
               use_paddy=False):
    rows = []
    stock = np.asarray(leftover, dtype=float)
    et = np.asarray(et, dtype=float)
    infs = _day_slices(infiltration, days)
    for day, inf in zip(days, infs):
        _, delta = law.flux(np.maximum(stock[day], 0.0), graph, area, physics,
                            et_mm=et[day], infiltration=inf, use_paddy=use_paddy)
        rows.append(delta)
    return np.vstack(rows)


def _share_days(law, leftover, et, graph, days, year, infiltration=None, cyclic=None):
    rows = []
    stock = np.asarray(leftover, dtype=float)
    et = np.asarray(et, dtype=float)
    infs = _day_slices(infiltration, days)
    for day, inf in zip(days, infs):
        part = law.partition(np.maximum(stock[day], 0.0), graph, et_mm=et[day], year=year,
                             infiltration=inf, cyclic=cyclic)
        rows.append(np.column_stack([part[key] for key in SHARE_KEYS]))
    return np.stack(rows, axis=0)


def _fate_teacher(leftover, shares, days):
    leftover = np.asarray(leftover, dtype=float)
    shares = np.asarray(shares, dtype=float)
    pool, stay = [], []
    for day in days:
        pile = np.maximum(leftover[day], 0.0)
        share = shares[day]
        pool.append(pile * (share[..., 0] + share[..., 1]))
        stay.append(pile * (share[..., 2] + share[..., 3]))
    return np.vstack(pool), np.vstack(stay)


def _fate_mm_days(fate, leftover, graph, days, infiltration=None, cyclic=None, dist=None,
                  use_paddy=False):
    leftover = np.asarray(leftover, dtype=float)
    infs = _day_slices(infiltration, days)
    pool, stay = [], []
    for day, inf in zip(days, infs):
        p, s = fate.predict_mm(np.maximum(leftover[day], 0.0), graph, infiltration=inf,
                               cyclic=cyclic, dist=dist, use_paddy=use_paddy)
        pool.append(p)
        stay.append(s)
    return np.vstack(pool), np.vstack(stay)


def _rmse_per_paddy(hat, target):
    err = np.asarray(hat, dtype=float) - np.asarray(target, dtype=float)
    return np.sqrt(np.mean(err ** 2, axis=0))


def _scope_compare(general_hat, paddy_hat, target):
    general = _rmse_per_paddy(general_hat, target)
    local = _rmse_per_paddy(paddy_hat, target)
    return SimpleNamespace(
        general_median_rmse=float(np.median(general)),
        paddy_median_rmse=float(np.median(local)),
        paddy_wins=float(np.mean(local < general - 1e-12)),
        n_paddies=int(general.size),
    )


def _capture_mean(share_stack, cyclic):
    capture = share_stack[..., 0] + share_stack[..., 1]
    mask = np.asarray(cyclic, dtype=bool)
    if not mask.any():
        return float(capture.mean())
    return float(capture[:, mask].mean())


def _eval_target(field, graph, area, physics, days):
    if field.neighbor is not None:
        return np.vstack([np.asarray(field.neighbor[day], dtype=float) for day in days]), "cached neighbor_mm"
    return _hop_teacher(field.leftover, graph, area, physics, days), "hop_stock teacher"


def equation_report_path(train_year=2021, eval_year=2022, folder=None):
    import os

    return os.path.join(_lateral_dir(folder), f"equation_{int(train_year)}_to_{int(eval_year)}.npz")


def load_equation_report(train_year=2021, eval_year=2022, folder=None):
    """Last frozen 2021→2022 equation comparison, or None."""
    import os

    path = equation_report_path(train_year, eval_year, folder)
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=True) as data:
        extra = (tuple(np.asarray(data["extra_names"]).astype(str).tolist())
                 if "extra_names" in data.files else ())

        def _metric(key, default=float("nan")):
            return float(data[key]) if key in data.files else default

        def one(form, train_key, eval_key, n_par, text, fitted, teacher_key=None):
            return SimpleNamespace(
                form=form, fitted=fitted, n_parameters=n_par,
                weights=np.asarray(data[form + "_weights"]) if form + "_weights" in data.files else None,
                equation_text=text,
                extra_names=extra,
                train_rmse=_metric(train_key),
                eval_rmse=_metric(eval_key),
                eval_rmse_teacher=_metric(teacher_key) if teacher_key else float("nan"),
                share_train_rmse=_metric(form + "_share_train_rmse"),
                share_eval_rmse=_metric(form + "_share_eval_rmse"),
                capture_train=_metric(form + "_capture_train"),
                capture_eval=_metric(form + "_capture_eval"),
            )
        linear_text = str(data["equation_text"]) if "equation_text" in data.files else "linear H"
        report = SimpleNamespace(
            two_separate_models=False, one_equation=True,
            forms=("linear", "nonlinear"), baseline="head_equalise",
            train_year=int(data["train_year"]), eval_year=int(data["eval_year"]),
            from_disk=True, saved=path,
            eval_target="cached leftover hops",
            train_source="cached year", eval_source="cached year",
            n_paddies=int(np.asarray(data["lon"]).size) if "lon" in data.files else 0,
            sample_days=None,
            farmer_model_separate=True,
            head_equalise=one("head_equalise", "physics_train_rmse", "physics_eval_rmse", 0,
                              "head equalise (physics baseline)", False,
                              "physics_eval_rmse_teacher"),
            linear=one("linear", "linear_train_rmse", "linear_eval_rmse", 9, linear_text, True,
                       "linear_eval_rmse_teacher"),
            nonlinear=one("nonlinear", "nonlinear_train_rmse", "nonlinear_eval_rmse", 21,
                          "nonlinear H", True, "nonlinear_eval_rmse_teacher"),
            eval_target_delta=np.asarray(data["eval_target"]) if "eval_target" in data.files else None,
            eval_linear=np.asarray(data["eval_linear"]) if "eval_linear" in data.files else None,
            eval_nonlinear=np.asarray(data["eval_nonlinear"]) if "eval_nonlinear" in data.files else None,
            eval_physics=np.asarray(data["eval_physics"]) if "eval_physics" in data.files else None,
        )
        if report.linear.weights is not None:
            pool_w = np.asarray(data["pool_weights"]) if "pool_weights" in data.files else None
            stay_w = np.asarray(data["stay_weights"]) if "stay_weights" in data.files else None
            fate_names = (tuple(np.asarray(data["fate_names"]).astype(str).tolist())
                          if "fate_names" in data.files else ())
            if pool_w is not None and pool_w.size == 0:
                pool_w = None
            if stay_w is not None and stay_w.size == 0:
                stay_w = None
            capture = report.linear.capture_eval
            if not np.isfinite(capture):
                capture = None
            report.linear.equation_text = leftover_fate_text(
                report.linear.weights, extra_names=extra, capture=capture,
                pool_weights=pool_w, stay_weights=stay_w, fate_names=fate_names)
            report.linear.n_parameters = int(np.size(report.linear.weights))
            report.equations_text = report.linear.equation_text
            if pool_w is not None:
                report.recapture = SimpleNamespace(
                    form="recapture", fitted=True, n_parameters=int(np.size(pool_w)),
                    weights=pool_w, extra_names=fate_names,
                    equation_text=affine_equation_text("to_pool_i (recapture / LFV)", pool_w, fate_names),
                    train_rmse=_metric("recapture_train_rmse"),
                    eval_rmse=_metric("recapture_eval_rmse"),
                    capture_eval=_metric("linear_capture_eval"),
                )
            if stay_w is not None:
                report.stay = SimpleNamespace(
                    form="stay", fitted=True, n_parameters=int(np.size(stay_w)),
                    weights=stay_w, extra_names=fate_names,
                    equation_text=affine_equation_text("stay_i (ground soil)", stay_w, fate_names),
                    train_rmse=_metric("stay_train_rmse"),
                    eval_rmse=_metric("stay_eval_rmse"),
                )
        if report.nonlinear.weights is not None:
            report.nonlinear.n_parameters = int(np.size(report.nonlinear.weights))
            report.nonlinear.equation_text = (
                f"nonlinear neighbour trickle, {report.nonlinear.n_parameters} weights")
        n_pad = int(report.n_paddies)
        n_hop = int(np.size(report.linear.weights)) if report.linear.weights is not None else 0
        report.linear_paddy = SimpleNamespace(
            form="linear_paddy", fitted=True, scope="per paddy",
            n_parameters=int(n_hop * n_pad) if n_hop else 0,
            weights=report.linear.weights,
            equation_text=(f"per-paddy dL_i: {n_pad} paddies x {n_hop} weights "
                           "(same variables as the general trickle law)"),
            extra_names=extra,
            train_rmse=_metric("linear_paddy_train_rmse"),
            eval_rmse=_metric("linear_paddy_eval_rmse"),
            eval_rmse_teacher=float("nan"),
        )
        if getattr(report, "recapture", None) is not None:
            n_fate = int(np.size(report.recapture.weights))
            report.recapture_paddy = SimpleNamespace(
                form="recapture_paddy", fitted=True, scope="per paddy",
                n_parameters=int(n_fate * n_pad),
                weights=None,
                extra_names=getattr(report.recapture, "extra_names", extra),
                equation_text=f"per-paddy to_pool_i: {n_pad} paddies x {n_fate} weights",
                train_rmse=_metric("recapture_paddy_train_rmse"),
                eval_rmse=_metric("recapture_paddy_eval_rmse"),
            )
            report.stay_paddy = SimpleNamespace(
                form="stay_paddy", fitted=True, scope="per paddy",
                n_parameters=int(n_fate * n_pad),
                weights=None,
                extra_names=getattr(report.stay, "extra_names", extra) if getattr(report, "stay", None) else extra,
                equation_text=f"per-paddy stay_i: {n_pad} paddies x {n_fate} weights",
                train_rmse=_metric("stay_paddy_train_rmse"),
                eval_rmse=_metric("stay_paddy_eval_rmse"),
            )
        report.trickle_compare = SimpleNamespace(
            paddy_wins=_metric("trickle_paddy_wins"),
            general_median_rmse=float("nan"),
            paddy_median_rmse=float("nan"),
        )
        report.recapture_compare = SimpleNamespace(paddy_wins=_metric("recapture_paddy_wins"))
        report.stay_compare = SimpleNamespace(paddy_wins=_metric("stay_paddy_wins"))
        report.scopes = ("general", "per paddy")
        report.eval_linear_paddy = (np.asarray(data["eval_linear_paddy"])
                                    if "eval_linear_paddy" in data.files else None)
        return report


def load_paddy_ground_water_law(train_year=2021, eval_year=2022, folder=None):
    """The real, fitted general leftover-fate equation — to_pool_i (recapture/LFV)
    and stay_i (ground soil), both in mm, as linear functions of the day's actual
    leftover_mm, each paddy's distance to the canal, its cyclic flag, and which
    cardinal neighbours it has — frozen from the last
    ``equation_<train_year>_to_<eval_year>.npz`` (produced by the Equation Builder /
    ``relative_et_holdout``). This is what a caller like the ABM should evaluate
    per paddy per day instead of assuming a fixed per-paddy capture/stay split.

    Returns ``None`` if that cache doesn't exist yet (run the Equation Builder tab
    once to create it).

    The per-paddy (not just general) neighbour-trickle law is included too, as
    ``trickle_paddy_weights`` (n_paddies x n_features, or None if not cached) with
    its own ``trickle_fate_names`` — a real fitted weight vector per paddy, not a
    single shared formula.

    lon/lat/cyclic here are positionally aligned with ``cyclic.data.paddies()``
    (verified: same source data, same row order) — a caller with its own paddy
    subset should still match on paddy_id/lon-lat rather than assume this forever.
    """
    import os

    path = equation_report_path(train_year, eval_year, folder)
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=True) as data:
        if "pool_weights" not in data.files or "stay_weights" not in data.files:
            return None
        return SimpleNamespace(
            train_year=int(data["train_year"]),
            eval_year=int(data["eval_year"]),
            fate_names=tuple(np.asarray(data["fate_names"]).astype(str).tolist()),
            pool_weights=np.asarray(data["pool_weights"], dtype=float),
            stay_weights=np.asarray(data["stay_weights"], dtype=float),
            lon=np.asarray(data["lon"], dtype=float),
            lat=np.asarray(data["lat"], dtype=float),
            cyclic=np.asarray(data["cyclic"], dtype=bool),
            trickle_paddy_weights=(np.asarray(data["linear_paddy_weights"], dtype=float)
                                   if "linear_paddy_weights" in data.files else None),
            trickle_fate_names=(tuple(str(n) for n in ("intercept", "drop_right", "drop_left",
                                     "drop_up", "drop_down", "et_right", "et_left", "et_up",
                                     "et_down", "leftover_mm", "has_right", "has_left",
                                     "has_up", "has_down"))
                                if "linear_paddy_weights" in data.files else None),
        )


def evaluate_ground_water_ratio(
    law, *, leftover_mm, distance_m, cyclic, has_right, has_left, has_up, has_down,
    infiltrated_mm=None,
):
    """to_pool_mm, stay_mm for every paddy today, from ``load_paddy_ground_water_law``'s
    general equation. Feature order follows ``law.fate_names`` — robust to the cache
    reordering its columns, as long as the names stay the same.

    ``leftover_mm`` is clipped to >= 0 first (this equation, like the rest of the
    leftover routing, only ever splits non-negative leftover). The raw regression
    is fit so ``to_pool_i + stay_i == leftover_i`` exactly, but nothing stops a raw
    evaluation from going negative for a small leftover_mm where the intercept
    dominates, or from giving non-cyclic paddies a nonzero to_pool — both violate
    hard physical rules stated elsewhere in this package (non-cyclic leftover never
    feeds LFV_pump; see cyclic/canal.py). Both are enforced here, with the
    conservation identity re-derived afterwards (stay = leftover - to_pool) rather
    than trusting the raw stay_weights evaluation once to_pool has been clamped.
    """
    leftover_mm = np.maximum(np.asarray(leftover_mm, dtype=float), 0.0)
    n = leftover_mm.shape[0]
    infiltrated_mm = np.zeros(n) if infiltrated_mm is None else np.broadcast_to(
        np.asarray(infiltrated_mm, dtype=float), (n,)
    )
    cyclic = np.broadcast_to(np.asarray(cyclic, dtype=bool), (n,))
    available = {
        "intercept": np.ones(n),
        "leftover_mm": leftover_mm,
        "infiltrated_mm": infiltrated_mm,
        "has_right": np.broadcast_to(np.asarray(has_right, dtype=float), (n,)),
        "has_left": np.broadcast_to(np.asarray(has_left, dtype=float), (n,)),
        "has_up": np.broadcast_to(np.asarray(has_up, dtype=float), (n,)),
        "has_down": np.broadcast_to(np.asarray(has_down, dtype=float), (n,)),
        "cyclic": cyclic.astype(float),
        "distance_m": np.broadcast_to(np.asarray(distance_m, dtype=float), (n,)),
    }
    x = np.stack([available[name] for name in law.fate_names], axis=1)
    to_pool = x @ law.pool_weights
    to_pool = np.where(cyclic, np.clip(to_pool, 0.0, leftover_mm), 0.0)
    stay = leftover_mm - to_pool
    return to_pool, stay


def equation_score_table(report):
    rows = []

    def add(item, equation, form, scope, fitted_on):
        rows.append({
            "equation": equation,
            "form": form,
            "scope": scope,
            "fitted on": fitted_on,
            "n parameters": int(item.n_parameters),
            f"{report.train_year} RMSE mm": item.train_rmse,
            f"{report.eval_year} RMSE mm": item.eval_rmse,
            "law": (item.equation_text.split("\n")[0] if item.equation_text else ""),
        })

    add(report.head_equalise, "neighbour trickle", "head_equalise", "general", "-")
    add(report.linear, "neighbour trickle", "linear", "general", str(report.train_year))
    if getattr(report, "linear_paddy", None) is not None and np.isfinite(
            getattr(report.linear_paddy, "eval_rmse", float("nan"))):
        add(report.linear_paddy, "neighbour trickle", "linear_paddy", "per paddy",
            str(report.train_year))
    add(report.nonlinear, "neighbour trickle", "nonlinear", "general", str(report.train_year))
    recapture = getattr(report, "recapture", None)
    if recapture is not None:
        add(recapture, "recapture to pool", recapture.form, "general", str(report.train_year))
    recapture_paddy = getattr(report, "recapture_paddy", None)
    if recapture_paddy is not None and np.isfinite(getattr(recapture_paddy, "eval_rmse", float("nan"))):
        add(recapture_paddy, "recapture to pool", recapture_paddy.form, "per paddy",
            str(report.train_year))
    stay = getattr(report, "stay", None)
    if stay is not None:
        add(stay, "stay in soil", stay.form, "general", str(report.train_year))
    stay_paddy = getattr(report, "stay_paddy", None)
    if stay_paddy is not None and np.isfinite(getattr(stay_paddy, "eval_rmse", float("nan"))):
        add(stay_paddy, "stay in soil", stay_paddy.form, "per paddy", str(report.train_year))
    return pd.DataFrame(rows)


def equation_scope_summary(report):
    """Plain-language general vs per-paddy holdout comparison."""
    pairs = (
        ("neighbour trickle", getattr(report, "linear", None), getattr(report, "linear_paddy", None),
         getattr(report, "trickle_compare", None)),
        ("recapture to pool", getattr(report, "recapture", None), getattr(report, "recapture_paddy", None),
         getattr(report, "recapture_compare", None)),
        ("stay in soil", getattr(report, "stay", None), getattr(report, "stay_paddy", None),
         getattr(report, "stay_compare", None)),
    )
    lines = []
    for label, general, local, compare in pairs:
        if general is None or local is None:
            continue
        if not np.isfinite(getattr(local, "eval_rmse", float("nan"))):
            continue
        wins = getattr(compare, "paddy_wins", float("nan")) if compare is not None else float("nan")
        win_text = (f"; per-paddy closer on {100.0 * wins:.0f}% of paddies"
                    if np.isfinite(wins) else "")
        lines.append(
            f"{label}: general {general.eval_rmse:.4f} mm vs per-paddy {local.eval_rmse:.4f} mm "
            f"on {report.eval_year}{win_text}"
        )
    return "\n".join(lines)


def equation_from_fields(train, eval_field, sample_days=16, progress=None, folder=None,
                         save=True):
    """Fit neighbour-trickle, recapture, and soil-stay equations on `train`, score `eval_field`."""
    from .agents import HeadEqualise, LeftoverFate, LinearHops, NonlinearHops
    from .validation import split

    train_year, eval_year = split(train.year, eval_field.year)
    physics = Physics()
    if progress:
        progress(f"Building UDLR neighbour graph ({train.n} paddies) ...")
    graph = _field_graph(train)
    area = train.area
    train_days = _sample_days(train.n_days, sample_days)
    eval_days = _sample_days(eval_field.n_days, sample_days)
    if progress:
        progress(f"Hop teacher on {train_year} ({len(train_days)} days) ...")
    train_teacher = _hop_teacher(train.leftover, graph, area, physics, train_days)
    eval_target, eval_target_name = _eval_target(eval_field, graph, eval_field.area, physics, eval_days)
    eval_teacher = _hop_teacher(eval_field.leftover, graph, eval_field.area, physics, eval_days)
    train_share_teacher = np.stack([np.asarray(train.shares[day], dtype=float) for day in train_days])
    share_rows = [train_share_teacher[i] for i in range(len(train_days))]
    train_pool_y, train_stay_y = _fate_teacher(train.leftover, train.shares, train_days)
    eval_pool_y, eval_stay_y = _fate_teacher(eval_field.leftover, eval_field.shares, eval_days)

    def pack_hop(name, law, train_hat, eval_hat, scope="general", n_par=None, text=None):
        weights = None if getattr(law, "weights", None) is None else np.asarray(law.weights, dtype=float)
        extra = tuple(getattr(law, "extra_names", ()) or ())
        if n_par is None:
            n_par = 0 if name == "head_equalise" else int(np.size(weights))
        if text is None:
            if name == "linear" and weights is not None:
                text = linear_equation_text(weights, extra_names=extra)
            elif name == "head_equalise":
                text = "head equalise (physics baseline)"
            else:
                text = f"nonlinear neighbour trickle, {n_par} weights"
        return SimpleNamespace(
            form=name, fitted=bool(getattr(law, "fitted", False)),
            scope=scope, n_parameters=n_par, weights=weights, equation_text=text,
            extra_names=extra,
            train_rmse=_rmse(train_hat, train_teacher),
            eval_rmse=_rmse(eval_hat, eval_target),
            eval_rmse_teacher=_rmse(eval_hat, eval_teacher),
            share_weights=None,
            capture_eval=float("nan"),
        )

    baseline = HeadEqualise()
    linear = LinearHops()
    nonlinear = NonlinearHops()
    fate = LeftoverFate()
    if progress:
        progress(f"Fitting general and per-paddy leftover equations on cached {train_year} ...")
    stocks = [np.maximum(np.asarray(train.leftover[day], dtype=float), 0.0) for day in train_days]
    ets = [np.asarray(train.et[day], dtype=float) for day in train_days]
    deltas = [train_teacher[i] for i in range(len(train_days))]
    infs = _day_slices(train.infiltration, train_days)
    linear.fit(stocks, ets, deltas, graph=graph, infiltrations=infs,
               use_stock_features=True, per_paddy=True)
    nonlinear.fit(stocks, ets, deltas, graph=graph, infiltrations=infs, use_stock_features=True)
    fate.fit(stocks, share_rows, graph, infiltrations=infs, cyclic=train.cyclic,
             dist=train.dist, per_paddy=True)
    if progress:
        progress(f"Applying frozen leftover equations to cached {eval_year} ...")
    physics_train = _flux_days(baseline, train.leftover, train.et, graph, area, physics, train_days)
    physics_eval = _flux_days(baseline, eval_field.leftover, eval_field.et, graph,
                              eval_field.area, physics, eval_days)
    linear_train = _flux_days(linear, train.leftover, train.et, graph, area, physics, train_days)
    linear_eval = _flux_days(linear, eval_field.leftover, eval_field.et, graph,
                             eval_field.area, physics, eval_days)
    linear_paddy_train = _flux_days(linear, train.leftover, train.et, graph, area, physics,
                                    train_days, use_paddy=True)
    linear_paddy_eval = _flux_days(linear, eval_field.leftover, eval_field.et, graph,
                                   eval_field.area, physics, eval_days, use_paddy=True)
    nonlinear_train = _flux_days(nonlinear, train.leftover, train.et, graph, area, physics,
                                 train_days)
    nonlinear_eval = _flux_days(nonlinear, eval_field.leftover, eval_field.et, graph,
                                eval_field.area, physics, eval_days)
    train_pool_hat, train_stay_hat = _fate_mm_days(
        fate, train.leftover, graph, train_days, train.infiltration, train.cyclic, train.dist)
    eval_pool_hat, eval_stay_hat = _fate_mm_days(
        fate, eval_field.leftover, graph, eval_days, eval_field.infiltration,
        eval_field.cyclic, eval_field.dist)
    train_pool_paddy, train_stay_paddy = _fate_mm_days(
        fate, train.leftover, graph, train_days, train.infiltration, train.cyclic, train.dist,
        use_paddy=True)
    eval_pool_paddy, eval_stay_paddy = _fate_mm_days(
        fate, eval_field.leftover, graph, eval_days, eval_field.infiltration,
        eval_field.cyclic, eval_field.dist, use_paddy=True)
    fate_names = tuple(getattr(fate, "feature_names", ()) or ())
    recapture = SimpleNamespace(
        form="recapture", scope="general", fitted=bool(fate.fitted),
        n_parameters=int(np.size(fate.pool_weights)),
        weights=np.asarray(fate.pool_weights, dtype=float),
        extra_names=fate_names,
        equation_text=affine_equation_text("to_pool_i (recapture / LFV)", fate.pool_weights, fate_names),
        train_rmse=_rmse(train_pool_hat, train_pool_y),
        eval_rmse=_rmse(eval_pool_hat, eval_pool_y),
        capture_eval=float(np.mean(np.divide(
            eval_pool_hat, np.maximum(np.vstack([np.maximum(eval_field.leftover[day], 0.0)
                                                 for day in eval_days]), 1e-6)))),
    )
    stay = SimpleNamespace(
        form="stay", scope="general", fitted=bool(fate.fitted),
        n_parameters=int(np.size(fate.stay_weights)),
        weights=np.asarray(fate.stay_weights, dtype=float),
        extra_names=fate_names,
        equation_text=affine_equation_text("stay_i (ground soil)", fate.stay_weights, fate_names),
        train_rmse=_rmse(train_stay_hat, train_stay_y),
        eval_rmse=_rmse(eval_stay_hat, eval_stay_y),
    )
    n_feat_hop = int(np.size(linear.weights))
    n_feat_fate = int(np.size(fate.pool_weights))
    recapture_paddy = SimpleNamespace(
        form="recapture_paddy", scope="per paddy", fitted=bool(fate.fitted),
        n_parameters=int(n_feat_fate * train.n),
        weights=None if fate.pool_paddy_weights is None else np.asarray(fate.pool_paddy_weights, dtype=float),
        extra_names=fate_names,
        equation_text=(f"per-paddy to_pool_i: {train.n} paddies x {n_feat_fate} weights "
                       "(same leftover/infiltration/distance variables as the general recapture law)"),
        train_rmse=_rmse(train_pool_paddy, train_pool_y),
        eval_rmse=_rmse(eval_pool_paddy, eval_pool_y),
    )
    stay_paddy = SimpleNamespace(
        form="stay_paddy", scope="per paddy", fitted=bool(fate.fitted),
        n_parameters=int(n_feat_fate * train.n),
        weights=None if fate.stay_paddy_weights is None else np.asarray(fate.stay_paddy_weights, dtype=float),
        extra_names=fate_names,
        equation_text=(f"per-paddy stay_i: {train.n} paddies x {n_feat_fate} weights "
                       "(same leftover/infiltration/distance variables as the general stay law)"),
        train_rmse=_rmse(train_stay_paddy, train_stay_y),
        eval_rmse=_rmse(eval_stay_paddy, eval_stay_y),
    )
    linear_hop = pack_hop("linear", linear, linear_train, linear_eval, scope="general")
    linear_hop.equation_text = leftover_equations_text(
        linear_hop.weights, extra_names=linear_hop.extra_names,
        pool_weights=recapture.weights, stay_weights=stay.weights, fate_names=fate_names)
    linear_hop.capture_eval = recapture.capture_eval
    linear_hop.share_weights = np.column_stack([fate.pool_weights, fate.stay_weights])
    linear_paddy = pack_hop(
        "linear_paddy", linear, linear_paddy_train, linear_paddy_eval, scope="per paddy",
        n_par=n_feat_hop * train.n,
        text=(f"per-paddy dL_i: {train.n} paddies x {n_feat_hop} weights "
              "(same drop/ET/leftover variables as the general trickle law; ridge fit, "
              "falls back to the general weights if a plot is singular)"))
    linear_paddy.paddy_weights = None if linear.paddy_weights is None else np.asarray(
        linear.paddy_weights, dtype=float)
    report = SimpleNamespace(
        two_separate_models=False,
        one_equation=True,
        leftover_equations=("neighbour_trickle", "recapture", "stay"),
        scopes=("general", "per paddy"),
        forms=("linear", "nonlinear"),
        baseline="head_equalise",
        train_year=int(train_year),
        eval_year=int(eval_year),
        per_paddy=True,
        checks_cardinal_neighbours=True,
        directions=("up", "down", "left", "right"),
        farmer_model_separate=True,
        train_source=train.source,
        eval_source=eval_field.source,
        eval_target=eval_target_name,
        sample_days=int(len(train_days)),
        n_paddies=int(train.n),
        head_equalise=pack_hop("head_equalise", baseline, physics_train, physics_eval),
        linear=linear_hop,
        linear_paddy=linear_paddy,
        nonlinear=pack_hop("nonlinear", nonlinear, nonlinear_train, nonlinear_eval),
        recapture=recapture,
        recapture_paddy=recapture_paddy,
        stay=stay,
        stay_paddy=stay_paddy,
        trickle_compare=_scope_compare(linear_eval, linear_paddy_eval, eval_target),
        recapture_compare=_scope_compare(eval_pool_hat, eval_pool_paddy, eval_pool_y),
        stay_compare=_scope_compare(eval_stay_hat, eval_stay_paddy, eval_stay_y),
        equations_text=linear_hop.equation_text,
        train_teacher=train_teacher,
        eval_target_delta=eval_target,
        eval_linear=linear_eval,
        eval_linear_paddy=linear_paddy_eval,
        eval_nonlinear=nonlinear_eval,
        eval_physics=physics_eval,
        lon=train.lon, lat=train.lat, cyclic=train.cyclic,
        water_pin=SimpleNamespace(
            train=getattr(train, "pin", None),
            eval=getattr(eval_field, "pin", None),
        ),
    )
    if save:
        path = equation_report_path(train_year, eval_year, folder)
        np.savez_compressed(
            path,
            train_year=report.train_year, eval_year=report.eval_year,
            linear_weights=report.linear.weights,
            nonlinear_weights=report.nonlinear.weights,
            pool_weights=report.recapture.weights,
            stay_weights=report.stay.weights,
            fate_names=np.asarray(fate_names, dtype=str),
            linear_train_rmse=report.linear.train_rmse,
            linear_eval_rmse=report.linear.eval_rmse,
            linear_paddy_train_rmse=report.linear_paddy.train_rmse,
            linear_paddy_eval_rmse=report.linear_paddy.eval_rmse,
            nonlinear_train_rmse=report.nonlinear.train_rmse,
            nonlinear_eval_rmse=report.nonlinear.eval_rmse,
            linear_eval_rmse_teacher=report.linear.eval_rmse_teacher,
            nonlinear_eval_rmse_teacher=report.nonlinear.eval_rmse_teacher,
            physics_eval_rmse_teacher=report.head_equalise.eval_rmse_teacher,
            physics_train_rmse=report.head_equalise.train_rmse,
            physics_eval_rmse=report.head_equalise.eval_rmse,
            recapture_train_rmse=report.recapture.train_rmse,
            recapture_eval_rmse=report.recapture.eval_rmse,
            recapture_paddy_train_rmse=report.recapture_paddy.train_rmse,
            recapture_paddy_eval_rmse=report.recapture_paddy.eval_rmse,
            stay_train_rmse=report.stay.train_rmse,
            stay_eval_rmse=report.stay.eval_rmse,
            stay_paddy_train_rmse=report.stay_paddy.train_rmse,
            stay_paddy_eval_rmse=report.stay_paddy.eval_rmse,
            trickle_paddy_wins=report.trickle_compare.paddy_wins,
            recapture_paddy_wins=report.recapture_compare.paddy_wins,
            stay_paddy_wins=report.stay_compare.paddy_wins,
            equation_text=np.array(report.linear.equation_text),
            extra_names=np.asarray(report.linear.extra_names, dtype=str),
            eval_linear=np.asarray(linear_eval, dtype=np.float32),
            eval_linear_paddy=np.asarray(linear_paddy_eval, dtype=np.float32),
            eval_nonlinear=np.asarray(nonlinear_eval, dtype=np.float32),
            eval_physics=np.asarray(physics_eval, dtype=np.float32),
            eval_target=np.asarray(eval_target, dtype=np.float32),
            lon=train.lon, lat=train.lat, cyclic=train.cyclic,
            linear_capture_eval=report.recapture.capture_eval,
            linear_paddy_weights=(np.asarray(linear.paddy_weights, dtype=np.float32)
                                  if linear.paddy_weights is not None else np.zeros(0)),
        )
        report.saved = path
    return report


def equation_from_cache(train_year=2021, eval_year=2022, sample_days=16, folder=None,
                        progress=None, save=True):
    """Fit H on a cached whole-year leftover/ET field, freeze it, score the other year.

    Leftover is IR+R−D−satellite ET at each paddy (FAO-56 `et_mm` is not overwritten).
    Drain capture is cyclic infiltrated leftover scaled to Table 5 LFV_pump. 2021
    default cache is reconstructed leftover from saved daily IR/D plus season ET.
    2022 prefers `outputs/runs/2022_*/paddy_daily.csv`. Linear and nonlinear are
    forms of the neighbour-trickle equation, not two hydrology models. Recapture
    and soil stay are leftover-mass equations. The farmer IR/D model is not this
    object.
    """
    from .validation import split

    train_year, eval_year = split(train_year, eval_year)
    if progress:
        progress(f"Loading cached {train_year} leftover / ET ...")
    train = load_year_field(train_year, folder=folder, progress=progress)
    if progress:
        progress(f"Loading cached {eval_year} leftover / ET ...")
    eval_field = load_year_field(eval_year, folder=folder, progress=progress)
    if train.n != eval_field.n:
        raise ValueError(f"{train_year} has {train.n} paddies, {eval_year} has {eval_field.n}")
    return equation_from_fields(train, eval_field, sample_days=sample_days,
                                progress=progress, folder=folder, save=save)


def equation_figure(report):
    """2021-fit vs 2022 leftover-hop comparison: RMSE bars and predicted vs target scatter."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12.5, 5.2))
    ax_bar, ax_sc = fig.subplots(1, 2)
    names = ["head_equalise", "linear", "linear_paddy", "nonlinear"]
    names = [name for name in names if getattr(report, name, None) is not None]
    train_rmse = [getattr(report, name).train_rmse for name in names]
    eval_rmse = [getattr(report, name).eval_rmse for name in names]
    x = np.arange(len(names))
    ax_bar.bar(x - 0.18, train_rmse, width=0.36, label=f"{report.train_year} (fit)")
    ax_bar.bar(x + 0.18, eval_rmse, width=0.36, label=f"{report.eval_year} (holdout)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([name.replace("_", "\n") for name in names], fontsize=8)
    ax_bar.set_ylabel("leftover-hop RMSE (mm)")
    ax_bar.set_title(f"General vs per-paddy H from {report.train_year} on {report.eval_year}",
                     fontsize=10)
    ax_bar.legend(fontsize=8)
    target = getattr(report, "eval_target_delta", None)
    pred = getattr(report, "eval_linear", None)
    pred_paddy = getattr(report, "eval_linear_paddy", None)
    if target is not None and pred is not None:
        target = np.asarray(target).ravel()
        pred = np.asarray(pred).ravel()
        step = max(int(target.size / 4000), 1)
        ax_sc.scatter(target[::step], pred[::step], s=6, alpha=0.35, label="linear general")
        if pred_paddy is not None:
            ax_sc.scatter(target[::step], np.asarray(pred_paddy).ravel()[::step],
                          s=6, alpha=0.35, label="linear per paddy")
        lo = float(np.min(target[::step]))
        hi = float(np.max(target[::step]))
        if hi <= lo:
            hi = lo + 1.0
        ax_sc.plot([lo, hi], [lo, hi], color="black", lw=0.8)
        ax_sc.set_aspect("equal", adjustable="datalim")
        ax_sc.legend(fontsize=8)
    ax_sc.set_xlabel(f"{report.eval_year} {report.eval_target} (mm)")
    ax_sc.set_ylabel(f"frozen trickle law on {report.eval_year} (mm)")
    ax_sc.set_title("Holdout leftover hops, every sampled paddy-day", fontsize=10)
    fig.tight_layout()
    return fig


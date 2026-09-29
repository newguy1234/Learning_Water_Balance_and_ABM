"""Neural Okura matcher: IR/D placement + per-paddy soil / sideways permeability.

Season IR/D aggregates are the hard target. Day shape uses Open-Meteo rain, ET,
soil moisture, and rain hours (no Fig. 8). Within a day: cover ET first, then
free leftover placement. Leftover kx/ky/kz/ground ratios are a separate agent.
Train with `train_et_net()` → et_net.pt.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import data, okura
from .agents import FieldAggregate, register_agent
from .identify import match_block_volumes

AGENT_DIR = os.path.join(data.PROJECT, "outputs", "agents")
WEIGHTS_NAME = "et_net.pt"
BASE_KZ = 13.0
BASE_KLAT = 2000.0


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


class SpillNet:
    """MLP → IR score, D score, spill scale, vertical K, lateral K."""

    FEATURE_NAMES = (
        "depth", "et", "rain", "need", "sat_et", "topsoil", "flood", "day",
        "okura_ir", "okura_d", "north", "south", "cyclic", "canal_dist",
    )

    def __init__(self, hidden=64):
        torch, nn = _torch()
        self.torch = torch
        n_in = len(self.FEATURE_NAMES)
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 5),
        )
        self.hidden = int(hidden)

    def parameters(self):
        return self.net.parameters()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state):
        self.net.load_state_dict(state)

    def forward(self, x):
        torch = self.torch
        raw = self.net(x)
        ir = torch.nn.functional.softplus(raw[:, 0]) + 0.05
        dr = torch.nn.functional.softplus(raw[:, 1]) + 0.05
        spill = 0.15 + 0.85 * torch.sigmoid(raw[:, 2])
        # Centered log-range so K is not pinned at a sigmoid ceiling.
        # Kz ~ 5–34 mm/d around BASE_KZ; K_lat ~ 0.3–3.3× BASE_KLAT.
        kz = BASE_KZ * torch.exp(0.95 * torch.tanh(raw[:, 3]))
        klat = BASE_KLAT * torch.exp(1.2 * torch.tanh(raw[:, 4]))
        return ir, dr, spill, kz, klat


def feature_matrix(depth, et, rain, need, sat_et, topsoil, day, okura_ir, okura_d,
                   block, cyclic, canal_dist=None, n_days=131, flood=None):
    depth = np.asarray(depth, dtype=np.float32)
    n = depth.size
    north = (np.asarray(block) == "North").astype(np.float32)
    cyc = np.asarray(cyclic, dtype=np.float32)
    day_frac = np.full(n, float(day) / max(int(n_days) - 1, 1), dtype=np.float32)
    rain_arr = np.full(n, float(np.asarray(rain).reshape(-1)[0]), dtype=np.float32)
    if flood is None:
        flood_arr = np.ones(n, dtype=np.float32)
    else:
        flood_arr = np.asarray(flood, dtype=np.float32).ravel()
        if flood_arr.size == 1:
            flood_arr = np.full(n, float(flood_arr[0]), dtype=np.float32)
        elif flood_arr.size != n:
            flood_arr = np.ones(n, dtype=np.float32)
        flood_arr = np.clip(flood_arr, 0.0, 1.0)
    if canal_dist is None:
        dist = np.zeros(n, dtype=np.float32)
    else:
        dist = np.asarray(canal_dist, dtype=np.float32).ravel()
        if dist.size != n:
            dist = np.zeros(n, dtype=np.float32)
    return np.column_stack([
        depth / 50.0,
        np.asarray(et, dtype=np.float32) / 8.0,
        rain_arr / 20.0,
        np.asarray(need, dtype=np.float32) / 5.0,
        np.asarray(sat_et, dtype=np.float32) / 8.0,
        np.asarray(topsoil, dtype=np.float32) / 25.0,
        flood_arr,
        day_frac,
        np.asarray(okura_ir, dtype=np.float32) / 50.0,
        np.asarray(okura_d, dtype=np.float32) / 50.0,
        north,
        1.0 - north,
        cyc,
        np.clip(dist / 500.0, 0.0, 3.0),
    ]).astype(np.float32)


def weights_path(name=WEIGHTS_NAME):
    return os.path.join(AGENT_DIR, name)


def save_weights(net, path=None, train_years=()):
    torch, _ = _torch()
    path = path or weights_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "hidden": net.hidden,
        "n_in": len(SpillNet.FEATURE_NAMES),
        "features": list(SpillNet.FEATURE_NAMES),
        "train_years": tuple(int(y) for y in train_years),
    }, path)
    return path


def weights_status():
    """Whether et_net.pt exists and matches the current feature layout."""
    path = weights_path()
    if not os.path.isfile(path):
        return SimpleNamespace(path=path, exists=False, compatible=False, train_years=(),
                               detail="no weights file — will auto-train on first run")
    torch, _ = _torch()
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        return SimpleNamespace(path=path, exists=True, compatible=False, train_years=(),
                               detail=f"could not read weights: {error}")
    n_in = int(blob.get("n_in", -1))
    want = len(SpillNet.FEATURE_NAMES)
    years = tuple(int(y) for y in blob.get("train_years", ()) or ())
    if n_in > 0 and n_in != want:
        return SimpleNamespace(path=path, exists=True, compatible=False, train_years=years,
                               detail=f"stale weights ({n_in} inputs, need {want}) — retrain")
    net = SpillNet(hidden=int(blob.get("hidden", 64)))
    try:
        net.load_state_dict(blob["state_dict"])
    except RuntimeError as error:
        return SimpleNamespace(path=path, exists=True, compatible=False, train_years=years,
                               detail=f"incompatible weights: {error}")
    if not years:
        years = (2021,)
    return SimpleNamespace(path=path, exists=True, compatible=True, train_years=years,
                           detail=f"trained on {years} ({want} features, hidden={net.hidden})")


def load_weights(net, path=None):
    torch, _ = _torch()
    path = path or weights_path()
    if not os.path.isfile(path):
        return False
    blob = torch.load(path, map_location="cpu", weights_only=False)
    try:
        net.load_state_dict(blob["state_dict"])
    except RuntimeError:
        return False
    years = tuple(int(y) for y in blob.get("train_years", ()) or ())
    net.train_years = years if years else (2021,)
    return True


def _as_volume(want, i):
    torch = _torch()[0]
    v = want[i]
    if isinstance(v, torch.Tensor):
        return v
    return torch.as_tensor(float(v), dtype=torch.float32)


def _torch_match_mask(scores, want_m3, area, block, mask):
    torch = _torch()[0]
    out = torch.zeros_like(scores)
    block_arr = np.asarray(block)
    mask_arr = np.asarray(mask, dtype=bool)
    for i, name in enumerate(("North", "South")):
        sel = torch.as_tensor((block_arr == name) & mask_arr)
        if not bool(sel.any()):
            continue
        w = torch.clamp(scores[sel], min=1e-6) * area[sel]
        mass = w.sum().clamp_min(1e-9)
        out[sel] = _as_volume(want_m3, i) * 1000.0 * w / (area[sel].clamp_min(1e-9) * mass)
    return out


def _torch_et_cover_then_free(cover_mm, free_scores, want_m3, area, block, mask):
    """IR mm with fixed block totals: fund ET cover, then free_scores place the rest."""
    torch = _torch()[0]
    cover = torch.clamp(cover_mm, min=0.0)
    free = torch.clamp(free_scores, min=1e-6)
    out = torch.zeros_like(cover)
    block_arr = np.asarray(block)
    mask_arr = np.asarray(mask, dtype=bool)
    for i, name in enumerate(("North", "South")):
        sel = torch.as_tensor((block_arr == name) & mask_arr)
        if not bool(sel.any()):
            continue
        want = _as_volume(want_m3, i)
        if float(want.detach()) <= 0.0:
            continue
        cover_m3 = (cover[sel] * area[sel]).sum() / 1000.0
        if cover_m3 <= want:
            base = cover[sel]
            leftover = want - cover_m3
        else:
            base = cover[sel] * (want / cover_m3.clamp_min(1e-9))
            leftover = want * 0.0
        w = free[sel] * area[sel]
        mass = w.sum().clamp_min(1e-9)
        extra = leftover * 1000.0 * w / (area[sel].clamp_min(1e-9) * mass)
        out[sel] = base + extra
    return out


def _volume_map(mm, area, block, mask):
    area = np.asarray(area, dtype=float)
    block = np.asarray(block)
    mask = np.asarray(mask, dtype=bool)
    mm = np.asarray(mm, dtype=float)
    out = {}
    for name in ("North", "South"):
        sel = (block == name) & mask
        out[name] = float((mm[sel] * area[sel]).sum() / 1000.0) if np.any(sel) else 0.0
    return out


def build_training_year(year=2021, physics=None):
    """One year of mornings + table targets for an unrolled surrogate."""
    from .model import CyclicModel
    from .physics import Physics

    physics = physics if physics is not None else Physics()
    model = CyclicModel(year, agents="field_aggregate", potential="field",
                        physics=physics, record_paddies=True, save_csv=False,
                        forcing="open_meteo")
    teacher = model.groups[0][0]
    rows = np.asarray(model.groups[0][1])
    area = np.asarray(model.area, dtype=np.float32)
    block = np.asarray(model.inputs.block)
    cyclic = np.asarray(model.inputs.cyclic, dtype=bool)
    ncb = ~cyclic
    canal = np.asarray(getattr(model.inputs, "canal_distance_m", np.zeros(model.n)), dtype=np.float32)
    days = []
    for d in range(model.n_days):
        morn = np.asarray(model.state.water, dtype=np.float32).copy()
        batch = model._observe().subset(rows)
        ir_t, dr_t = teacher.decide_batch(batch)
        full_ir = np.zeros(model.n)
        full_dr = np.zeros(model.n)
        full_ir[rows] = ir_t
        full_dr[rows] = dr_t
        model.advance(full_ir, full_dr)
        need_full = np.zeros(model.n, dtype=np.float32)
        need_full[rows] = np.maximum(np.asarray(batch.need_mm, dtype=np.float32), 0.0)
        top_full = np.zeros(model.n, dtype=np.float32)
        top_full[rows] = np.maximum(
            np.asarray(getattr(batch, "topsoil_mm", 0.0), dtype=np.float32), 0.0)
        rain = float(np.asarray(model.inputs.rain_mm[d]).reshape(-1)[0])
        et = np.asarray(model.inputs.et_target_mm[d], dtype=np.float32)
        sat = np.nan_to_num(np.asarray(model.inputs.sat_et_mm[d], dtype=np.float32), nan=0.0)
        flood = np.asarray(getattr(model.inputs, "flood_daily", np.ones(model.n)), dtype=np.float32)
        if flood.ndim == 2:
            flood = flood[min(d, flood.shape[0] - 1)]
        plan_ir = teacher.irrigation[d]
        plan_d = teacher.discharge[d]
        days.append(dict(
            day=d, year=int(year), morn=morn, et=et, rain=rain, need_full=need_full, sat_et=sat,
            topsoil=top_full, flood=flood, okura_ir=plan_ir, okura_d=plan_d,
            want_ir_c=_volume_map(plan_ir, area, block, cyclic),
            want_d_c=_volume_map(plan_d, area, block, cyclic),
            want_ir_n=_volume_map(plan_ir, area, block, ncb),
            want_d_n=_volume_map(plan_d, area, block, ncb),
            area=area, block=block, cyclic=cyclic, ncb=ncb, canal_dist=canal,
            outlet=float(model.physics.outlet_loss_per_day),
            bund=float(model.physics.bund_mm),
            href=float(getattr(model.physics, "pond_infiltrate_ref_mm", 30.0)),
            n_days=model.n_days,
            lfv_day_m3=float(getattr(model, "lfv_plan_m3", np.zeros(model.n_days))[
                min(d, model.n_days - 1)]) if getattr(model, "lfv_plan_m3", None) is not None else 0.0,
        ))
    return days


def _surrogate_day(net, day, depth):
    """One morning: pin IR/D, apply Kz infiltration + lateral mix, return loss and new depth."""
    torch = _torch()[0]
    x = feature_matrix(
        depth.detach().cpu().numpy() if torch.is_tensor(depth) else day["morn"],
        day["et"], day["rain"], day["need_full"], day["sat_et"], day["topsoil"],
        day["day"], day["okura_ir"], day["okura_d"], day["block"], day["cyclic"],
        day.get("canal_dist"), day["n_days"], flood=day.get("flood"))
    # Use live depth in features when unrolling.
    if torch.is_tensor(depth):
        x = torch.as_tensor(x)
        x = x.clone()
        x[:, 0] = torch.clamp(depth, min=0.0) / 50.0
    else:
        x = torch.as_tensor(x)
        depth = torch.as_tensor(day["morn"], dtype=torch.float32)

    ir_s, dr_s, spill, kz, klat = net.forward(x)
    area = torch.as_tensor(day["area"], dtype=torch.float32)
    block = day["block"]
    cyclic = day["cyclic"]
    ncb = day["ncb"]
    want_ir_c = [day["want_ir_c"]["North"], day["want_ir_c"]["South"]]
    want_d_c = [day["want_d_c"]["North"], day["want_d_c"]["South"]]
    want_ir_n = [day["want_ir_n"]["North"], day["want_ir_n"]["South"]]
    want_d_n = [day["want_d_n"]["North"], day["want_d_n"]["South"]]

    et = torch.as_tensor(day["et"], dtype=torch.float32)
    rain = float(day["rain"])
    er = float(okura.effective_rain_mm(rain))
    # Cover ET with pond + *useful* rain first; net places leftover IR freely.
    cover = torch.clamp(et - er - torch.clamp(depth, min=0.0), min=0.0)
    ir_mm = (_torch_et_cover_then_free(cover, ir_s, want_ir_c, area, block, cyclic)
             + _torch_et_cover_then_free(cover, ir_s, want_ir_n, area, block, ncb))

    spill_c, spill_n = [], []
    for name in ("North", "South"):
        sel_c = (np.asarray(block) == name) & np.asarray(cyclic, dtype=bool)
        sel_n = (np.asarray(block) == name) & np.asarray(ncb, dtype=bool)
        spill_c.append(spill[torch.as_tensor(sel_c)].mean() if np.any(sel_c) else torch.tensor(0.7))
        spill_n.append(spill[torch.as_tensor(sel_n)].mean() if np.any(sel_n) else torch.tensor(0.7))
    farmer_c = [want_d_c[i] * spill_c[i] for i in range(2)]
    farmer_n = [want_d_n[i] * spill_n[i] for i in range(2)]
    dr_mm = (_torch_match_mask(dr_s, farmer_c, area, block, cyclic)
             + _torch_match_mask(dr_s, farmer_n, area, block, ncb))

    outlet = float(day["outlet"])
    bund = float(day["bund"])
    href = max(float(day["href"]), 1.0)

    # Physics order: wet → ET → farmer D → overflow → spill → infiltrate → sideways mix.
    water = torch.clamp(depth + ir_mm + rain - et, min=0.0)
    water = torch.clamp(water - dr_mm, min=0.0)
    overflow = torch.clamp(water - bund, min=0.0)
    water = water - overflow
    spill_mm = water * outlet
    water = water - spill_mm
    infil = torch.minimum(kz * (water / href), water)
    water = water - infil
    scored = dr_mm + overflow + spill_mm

    # Sideways: plots with higher K_lat equalise pond toward block mean (heterogeneous soil).
    mix = torch.clamp(klat / (BASE_KLAT * 4.0), 0.0, 0.35)
    new_depth = water.clone()
    for name in ("North", "South"):
        for mask in (cyclic, ncb):
            sel_np = (np.asarray(block) == name) & np.asarray(mask, dtype=bool)
            if not np.any(sel_np):
                continue
            sel = torch.as_tensor(sel_np)
            mass = (new_depth[sel] * area[sel]).sum().clamp_min(1e-9)
            mean = mass / area[sel].sum().clamp_min(1e-9)
            new_depth = torch.where(sel, new_depth + mix * (mean - new_depth), new_depth)

    loss = torch.zeros((), dtype=torch.float32)
    n_terms = 0
    for i, name in enumerate(("North", "South")):
        for mask, want_ir, want_d in (
            (cyclic, want_ir_c, want_d_c),
            (ncb, want_ir_n, want_d_n),
        ):
            sel = torch.as_tensor((np.asarray(block) == name) & np.asarray(mask, dtype=bool))
            if not bool(sel.any()):
                continue
            got_ir = (ir_mm[sel] * area[sel]).sum() / 1000.0
            got_d = (scored[sel] * area[sel]).sum() / 1000.0
            scale = max(float(want_ir[i]), float(want_d[i]), 1.0)
            loss = loss + ((got_ir - want_ir[i]) / scale) ** 2 + ((got_d - want_d[i]) / scale) ** 2
            n_terms += 2

    # --- Theory-guided / SOTA-style soft physics (no Fig. 8 labels) ---------------------
    # FAO-56 / IPICS: cover ET (already); MPC zone control: keep pond in a soft band.
    shortfall = torch.relu(et - (torch.clamp(depth, min=0.0) + ir_mm + er))
    et_scale = et.mean().clamp_min(0.5)
    loss = loss + 0.35 * ((shortfall * area).sum() / (area.sum().clamp_min(1.0) * et_scale))
    n_terms += 1
    pond_lo, pond_hi = 8.0, 55.0
    zone = (torch.relu(pond_lo - new_depth) + torch.relu(new_depth - pond_hi)) / 30.0
    loss = loss + 0.12 * (zone * area).sum() / area.sum().clamp_min(1.0)
    n_terms += 1

    # Wet-day IR pause: MAFF useful rain (not raw drizzle / uncapped storms).
    wet = float(np.clip(er / 12.0, 0.0, 1.0))
    ir_all = (ir_mm * area).sum() / 1000.0
    d_cyc = torch.zeros((), dtype=torch.float32)
    sel_c = torch.as_tensor(np.asarray(cyclic, dtype=bool).copy())
    if bool(sel_c.any()):
        d_cyc = (scored[sel_c] * area[sel_c]).sum() / 1000.0
    ir_scale = max(float(sum(want_ir_c) + sum(want_ir_n)), 1.0)
    loss = loss + 0.18 * wet * (ir_all / ir_scale)
    n_terms += 1

    # Midsummer drainage window: cyclic IR ≈ 0 (Okura midsummer).
    year = int(day.get("year", 2022))
    d_idx = int(day["day"])
    dates = okura.season_dates(year)
    if d_idx < len(dates):
        start = pd.Timestamp(f"{year}-{okura.MIDSUMMER_START}")
        if start <= dates[d_idx] < start + pd.Timedelta(days=okura.MIDSUMMER_DAYS):
            loss = loss + 0.40 * (ir_all / ir_scale)
            n_terms += 1

    # IR–D anti-coupling → lake W proxy pulses on dry high-IR days (Eq. 4 style).
    lfv = float(day.get("lfv_day_m3", 0.0))
    w_proxy = torch.relu(ir_all - d_cyc - lfv)
    # Prefer some lake draw when dry; prefer little when wet (soft, portable).
    loss = loss + 0.08 * ((1.0 - wet) * torch.relu(0.15 * ir_all - w_proxy)
                          + wet * (w_proxy / ir_scale)) / ir_scale
    n_terms += 1

    # Spatial continuity / starve-burst (within-day).
    for i, name in enumerate(("North", "South")):
        sel = torch.as_tensor((np.asarray(block) == name) & np.asarray(cyclic, dtype=bool))
        if not bool(sel.any()):
            continue
        want = float(want_ir_c[i])
        if want < 1.0:
            continue
        got_mm = ir_mm[sel]
        mean_mm = (got_mm * area[sel]).sum() / area[sel].sum().clamp_min(1e-9)
        starve = torch.relu(0.15 * mean_mm - got_mm)
        loss = loss + 0.04 * (starve * area[sel]).sum() / (area[sel].sum().clamp_min(1.0) * mean_mm.clamp_min(0.5))
        loss = loss + 0.03 * (torch.relu(got_mm - 2.0 * mean_mm) * area[sel]).sum() / (
            area[sel].sum().clamp_min(1.0) * mean_mm.clamp_min(0.5))
        n_terms += 2

    # Flood-aware empty (only when satellite flood is usable).
    flood = day.get("flood")
    if flood is not None:
        f = np.asarray(flood, dtype=float).ravel()
        if f.size == area.numel() and float(np.percentile(f, 90)) >= 0.15:
            dry = torch.as_tensor(f < 0.25)
            if bool(dry.any()):
                loss = loss + 0.10 * (ir_mm[dry] * area[dry]).sum() / (
                    area.sum().clamp_min(1.0) * (ir_all.detach().clamp_min(1.0) / area.sum().clamp_min(1.0) * 1000.0 + 1.0))
                n_terms += 1

    loss = loss + 0.03 * ((kz.mean() / BASE_KZ - 1.0) ** 2
                          + (klat.mean() / BASE_KLAT - 1.0) ** 2)
    for k, base in ((kz, BASE_KZ), (klat, BASE_KLAT)):
        cv = k.std() / k.mean().clamp_min(1e-3)
        loss = loss + 0.02 * torch.relu(0.25 - cv) ** 2
        loss = loss + 0.004 * ((k.mean() - base) / base) ** 2
    n_terms += 3
    # Eq. 3 leftover proxy on cyclic paddies (IR + R − D − ET), m³.
    et_t = torch.as_tensor(day["et"], dtype=torch.float32)
    if bool(sel_c.any()):
        lf_proxy = ((ir_mm[sel_c] + rain - et_t[sel_c] - scored[sel_c]) * area[sel_c]).sum() / 1000.0
    else:
        lf_proxy = torch.zeros((), dtype=torch.float32)
    lfv_day = torch.as_tensor(float(day.get("lfv_day_m3", 0.0)), dtype=torch.float32)
    extras = dict(ir_all_m3=ir_all, d_cyc_m3=d_cyc, w_proxy_m3=w_proxy,
                  lf_proxy_m3=lf_proxy, lfv_day_m3=lfv_day)
    return loss, n_terms, new_depth, kz, klat, extras


def train_et_net(train_years=(2021,), epochs=40, lr=0.02, hidden=64, progress=None):
    """Fit SpillNet with theory-guided soft physics (no Fig. 8 daily labels).

    Default train year is 2021 only so 2022 is a true unseen holdout.
    Hard: Table 4/5 day IR/D aggregates. Soft: ET cover, pond zone (MPC), wet-day
    pause, midsummer off, IR–D→W proxy, TgLSTM ΔIR, season W / Lf / LFV / dWB.
    """
    torch, _ = _torch()
    if isinstance(train_years, int):
        train_years = (train_years,)
    years_days = []
    for year in train_years:
        if progress:
            progress(f"building {year} training days…")
        years_days.append(build_training_year(int(year)))
    net = SpillNet(hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    history = []
    for epoch in range(int(epochs)):
        opt.zero_grad()
        loss = torch.zeros((), dtype=torch.float32)
        n_terms = 0
        for days in years_days:
            depth = torch.as_tensor(days[0]["morn"], dtype=torch.float32)
            prev_ir = None
            w_season = torch.zeros((), dtype=torch.float32)
            lf_season = torch.zeros((), dtype=torch.float32)
            lfv_season = torch.zeros((), dtype=torch.float32)
            year = int(days[0].get("year", 2021))
            for t, day in enumerate(days):
                part, n, depth, _kz, _klat, extras = _surrogate_day(net, day, depth)
                loss = loss + part
                n_terms += n
                ir_all = extras["ir_all_m3"]
                w_season = w_season + extras["w_proxy_m3"]
                lf_season = lf_season + extras["lf_proxy_m3"]
                lfv_season = lfv_season + extras["lfv_day_m3"]
                # TgLSTM-style fluctuation: mild day-to-day IR change (not Fig. 8).
                if prev_ir is not None:
                    scale = max(float(ir_all.detach()), float(prev_ir.detach()), 1.0)
                    loss = loss + 0.04 * ((ir_all - prev_ir) / scale) ** 2
                    n_terms += 1
                prev_ir = ir_all
                if (t + 1) % 8 == 0:
                    depth = depth.detach()
                    prev_ir = prev_ir.detach()
            # Soft season identities (portable aggregates — not digitised daily ops).
            t5 = okura.TABLE5_1000M3[year]
            want_w = float(t5["W"]) * 1000.0
            want_lfv = float(t5["LFV_pump"]) * 1000.0
            want_dwb = float(t5["dWB"]) * 1000.0
            want_lf = float(sum(
                okura.TABLE4_MM[(b, year)]["lateral"] * okura.cyclic_area_m2(b) / 1000.0
                for b in okura.BLOCKS))
            loss = loss + 0.05 * ((w_season - want_w) / max(want_w, 1.0)) ** 2
            loss = loss + 0.04 * ((lf_season - want_lf) / max(want_lf, 1.0)) ** 2
            loss = loss + 0.04 * ((lfv_season - want_lfv) / max(want_lfv, 1.0)) ** 2
            # dWB residual of leftover vs LFV capture (Eq. 3 / Table 5 identity).
            loss = loss + 0.03 * (
                ((lf_season - lfv_season) - want_dwb) / max(want_dwb, 1.0)) ** 2
            n_terms += 4
        loss = loss / max(n_terms, 1)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
        if progress and (epoch % 5 == 0 or epoch == epochs - 1):
            progress(f"epoch {epoch + 1}/{epochs}  loss={history[-1]:.6f}")
    path = save_weights(net, train_years=tuple(train_years))
    return SimpleNamespace(net=net, history=history, path=path, train_years=tuple(train_years))


@register_agent
class ETNet(FieldAggregate):
    """Neural Okura matcher with spatially varying plow-pan and lateral permeability."""

    name = "et_net"
    training_years = ()
    defaults = dict(
        hidden=64,
        spill_fudge=0.02,
        pool_capture_share=None,
        plow_pan_k_mm_d=13.0,
        auto_train=0.0,   # 1 = train on the spot if no saved weights (never by default)
        irrigate_ncb=1.0,
        neural_discharge=0.0,
        k_ema=0.08,
        place_ema=0.22,
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
        ir_supply_cap=0.0,
    )

    def reset(self, model, rows):
        super().reset(model, rows)
        hidden = int(self.params.get("hidden", 64))
        self._net = SpillNet(hidden=hidden)
        self._trained = load_weights(self._net)
        if self._trained:
            self.training_years = tuple(getattr(self._net, "train_years", ()) or (2021,))
        elif float(self.params.get("auto_train", 0.0)):
            report = train_et_net(train_years=(2021,), hidden=hidden,
                                  progress=lambda m: print(m, flush=True))
            self._net = report.net
            self._trained = True
            self.training_years = tuple(report.train_years)
        else:
            # Never train behind a run: use the saved weights, or say how to make them.
            raise RuntimeError(
                f"et_net has no usable trained weights ({weights_status().detail}). "
                "Train once with cyclic.train_et_net() (saved to outputs/agents/et_net.pt), "
                "or pass auto_train=1 to this farmer.")
        n = model.n
        model.k_z = np.full(n, BASE_KZ, dtype=float)
        model.k_x = np.full(n, BASE_KLAT, dtype=float)
        model.k_y = np.full(n, BASE_KLAT, dtype=float)
        self._kz = model.k_z
        self._kx = model.k_x
        self._ky = model.k_y
        self._k_ready = False
        self._ir_w = None

    def __repr__(self):
        status = weights_status()
        tag = "trained" if status.compatible else "needs train"
        shown = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.params.items()
                 if not isinstance(v, np.ndarray)}
        return f"ETNet({tag}; {status.detail}; {shown})"

    def decide_batch(self, batch):
        torch = _torch()[0]
        rows = self.rows
        day = int(batch.day)
        self._day = day
        self._sync_drainage_catchup(day)

        depth = np.asarray(batch.depth_mm, dtype=float)
        et = np.asarray(self.expected_et_mm(batch), dtype=float)
        rain = float(np.mean(np.asarray(batch.rain_mm, dtype=float)))
        need = np.maximum(np.asarray(batch.need_mm, dtype=float), 0.0)
        sat = np.nan_to_num(np.asarray(batch.sat_et_mm, dtype=float), nan=0.0)
        top = np.maximum(np.asarray(getattr(batch, "topsoil_mm", 0.0), dtype=float), 0.0)
        n = self._area.size
        full_depth = np.zeros(n)
        full_et = np.zeros(n)
        full_need = np.zeros(n)
        full_sat = np.zeros(n)
        full_top = np.zeros(n)
        full_flood = np.ones(n)
        full_depth[rows] = depth
        full_et[rows] = et
        full_need[rows] = need
        full_sat[rows] = sat
        full_top[rows] = top
        flood_in = getattr(self._model.inputs, "flood_daily", None)
        if flood_in is not None:
            flood_day = np.asarray(flood_in, dtype=float)
            if flood_day.ndim == 2:
                flood_day = flood_day[min(day, flood_day.shape[0] - 1)]
            full_flood = np.clip(flood_day, 0.0, 1.0)
        canal = np.asarray(getattr(self._model.inputs, "canal_distance_m", np.zeros(n)), dtype=float)
        n_days = int(getattr(self._model, "n_days", 131))
        x = feature_matrix(
            full_depth, full_et, rain, full_need, full_sat, full_top, day,
            self.irrigation[day], self.discharge[day], self._block, self._cyclic,
            canal, n_days, flood=full_flood)
        self._net.net.eval()
        with torch.no_grad():
            ir_s, dr_s, _spill, kz, klat = self._net.forward(torch.as_tensor(x))
            ir_s = ir_s.cpu().numpy()
            dr_s = dr_s.cpu().numpy()
            kz = kz.cpu().numpy()
            klat = klat.cpu().numpy()

        # Soil permeability drifts slowly — it is not rewritten every morning.
        a_k = float(np.clip(self.params.get("k_ema", 0.08), 0.0, 1.0))
        if self._k_ready and a_k < 1.0:
            kz = (1.0 - a_k) * np.asarray(self._kz, dtype=float) + a_k * kz
            klat = (1.0 - a_k) * np.asarray(self._kx, dtype=float) + a_k * klat
        self._k_ready = True
        self._model.k_z = kz
        self._model.k_x = klat
        self._model.k_y = klat
        self._kz = kz
        self._kx = klat
        self._ky = klat

        area = self._area
        block = self._block
        cyclic = self._cyclic
        ncb = ~cyclic
        # ET cover first (mm): pond + MAFF useful rain; then free leftover IR.
        er = float(okura.effective_rain_mm(rain))
        cover_full = np.maximum(full_et - er - np.maximum(full_depth, 0.0), 0.0)
        free_w = np.maximum(ir_s, 1e-6)
        if self._flood_signal_usable(full_flood):
            free_w = free_w * (0.15 + 0.85 * np.maximum(full_flood, 0.0))
        a_p = float(np.clip(self.params.get("place_ema", 0.22), 0.0, 1.0))
        if self._ir_w is not None and a_p < 1.0:
            free_w = (1.0 - a_p) * self._ir_w + a_p * free_w
        self._ir_w = free_w

        from .identify import match_et_cover_then_free

        want_ir_c, want_ir_n = {}, {}
        for name in ("North", "South"):
            sel_c = (block == name) & cyclic
            sel_n = (block == name) & ncb
            want_ir_c[name] = float((self.irrigation[day, sel_c] * area[sel_c]).sum() / 1000.0) if np.any(sel_c) else 0.0
            want_ir_n[name] = float((self.irrigation[day, sel_n] * area[sel_n]).sum() / 1000.0) if np.any(sel_n) else 0.0
        ir_c = match_et_cover_then_free(want_ir_c, area, block, cyclic, cover_full, free_w)
        ir_n = match_et_cover_then_free(want_ir_n, area, block, ncb, cover_full, free_w)
        irrigation_rows = np.where(ncb, ir_n, ir_c)[rows]

        base_dr = self._spill_adjusted_discharge(
            irrigation_rows, depth, et, self.discharge[day], rows)
        if not float(self.params.get("neural_discharge", 0.0)):
            irrigation_rows, base_dr = self._apply_satellite_empty_when_dry(
                irrigation_rows, base_dr, depth, day, rows)
            return irrigation_rows, base_dr

        want_full = np.zeros(n)
        want_full[rows] = base_dr
        want_d = {}
        for name in ("North", "South"):
            sel = (block == name) & cyclic
            want_d[name] = float((want_full[sel] * area[sel]).sum() / 1000.0) if np.any(sel) else 0.0
        discharge = match_block_volumes(
            want_d, area, block, cyclic, weights=area * np.maximum(dr_s, 1e-6))
        irrigation_rows, discharge_rows = self._apply_satellite_empty_when_dry(
            irrigation_rows, self._merge_ncb_discharge(discharge[rows], day, rows),
            depth, day, rows)
        return irrigation_rows, discharge_rows

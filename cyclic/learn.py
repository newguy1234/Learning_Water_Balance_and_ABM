"""Models A and B, trained together through the shared water balance (cyclic.water).

Model A (Allocator) is a neural net that splits each day's measured block irrigation
and drainage across the block's paddies, from the weather, each paddy's state and ET
demand, its static features and a per-paddy embedding (so it can overfit per paddy).

Model B (RatioField) is each paddy's fixed hydrology ratios (cyclic.water.SHARES):
kz into the soil, kx / ky to neighbours, ground to deep soil, lateral to the pool.

Both are fitted at once by gradient descent through a season of `water.step`, so that
    • every paddy's satellite ET is met as fully as the water it was given allows
      (and what it cannot get from its own pond and soil must come from neighbours)
    • the block drainage matches what was measured, day by day
    • the water entering the ground per block matches Okura Table 5 LfA (Lf)
    • the lateral flow reaching the pool matches Table 5 LFV_pump
They are meant to overfit the training season; `evaluate` then runs them frozen on
another season. `Season.from_real(year)` is the measured data; the ABM makes
synthetic seasons with known ratios (abm.twin) to check the ratios are recoverable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from stem import water
from stem.district import GROUPS, OUTPUTS, load_district

LEARNED_DIR = os.path.join(OUTPUTS, "learned")
REAL_WEIGHTS = os.path.join(LEARNED_DIR, "real.pt")
REAL_RATIOS = os.path.join(LEARNED_DIR, "ratios_real.npz")
DRAIN_GROUP_WEIGHT = (1.0, 1.0, 0.3, 0.3)   # non-cyclic daily drainage was not measured
TABLE_WEIGHT = 5.0


@dataclass
class Season:
    """Everything one season's training or evaluation needs (numpy)."""

    year: int
    area: np.ndarray
    group: np.ndarray            # 0..3, district.GROUPS
    neighbours: np.ndarray
    static: np.ndarray           # (n, k) standardised static features
    rain: np.ndarray             # (days,)
    et0: np.ndarray              # (days,)
    et_demand: np.ndarray        # (days, n) the ET each paddy must meet
    irrigation_m3: np.ndarray    # (4, days) applied per group
    drainage_m3: np.ndarray      # (4, days) measured per group
    lf_m3: np.ndarray            # (2,) North, South cyclic: water into the ground
    lfv_m3: float                # lateral flow recaptured at the pool
    source: str = "real"
    truth: water.Ratios | None = None
    extra: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.area)

    @property
    def days(self):
        return len(self.rain)

    @classmethod
    def from_real(cls, year):
        from . import satellite
        from .observed import block_forcing, soil_targets

        d = load_district(year)
        ir, dr = block_forcing(d)
        t = soil_targets(year)
        return cls(year=d.year, area=d.area_m2, group=d.group_index(), neighbours=d.neighbours,
                   static=static_features(d), rain=d.rain_mm, et0=d.et0_mm, et_demand=satellite.daily_et(d),
                   irrigation_m3=ir, drainage_m3=dr, lf_m3=np.array([t.lf_m3["North"], t.lf_m3["South"]]),
                   lfv_m3=t.lfv_m3, source="real")


def static_features(district):
    z = lambda a: (np.asarray(a, float) - np.mean(a)) / (np.std(a) or 1.0)
    return np.column_stack([
        z(district.block == "North"), z(district.cyclic), z(district.elevation_m),
        z(np.log1p(district.pool_distance_m)), z(np.log1p(district.canal_distance_m)),
        z(district.lon), z(district.lat),
    ])


# --- the two models -----------------------------------------------------------------------------

N_DYNAMIC = 8


class Allocator(nn.Module):
    """Model A: per-paddy irrigation and discharge scores for one day."""

    def __init__(self, n, n_static, embed=8, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(n, embed)
        nn.init.normal_(self.embed.weight, std=0.1)
        self.net = nn.Sequential(nn.Linear(n_static + embed + N_DYNAMIC, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.config = dict(n=n, n_static=n_static, embed=embed, hidden=hidden)

    def forward(self, static, dynamic):
        x = torch.cat([static, self.embed.weight, dynamic], dim=1)
        return nn.functional.softplus(self.net(x)) + 1e-3


class RatioField(nn.Module):
    """Model B: each paddy's fixed kz and soil shares (free parameters per paddy)."""

    def __init__(self, n, init: water.Ratios | None = None):
        super().__init__()
        init = init or water.Ratios.uniform(n)
        kz = np.clip(init.kz, 1e-4, 1 - 1e-4)
        self.kz_logit = nn.Parameter(torch.tensor(np.log(kz / (1 - kz)), dtype=torch.float64))
        self.share_logits = nn.Parameter(torch.tensor(np.log(np.clip(init.shares, 1e-6, 1)), dtype=torch.float64))

    def tensors(self):
        return torch.sigmoid(self.kz_logit), torch.softmax(self.share_logits, dim=1)

    def ratios(self):
        kz, shares = self.tensors()
        return water.Ratios(kz.detach().numpy().copy(), shares.detach().numpy().copy())


# --- one season through the water balance -------------------------------------------------------

def _allocate(total_m3, score, group, area, n_groups=4):
    """mm per paddy so each group's paddies receive `total_m3` in proportion to score × area."""
    weight = score * area
    denom = torch.zeros(n_groups, dtype=weight.dtype).index_add(0, group, weight)
    return total_m3[group] * 1000.0 * score / denom[group].clamp_min(1e-9)


def farmer_day(allocator, season, d, pond, soil, static, area, group, group_area, ir_tot, dr_tot, demand):
    """Model A's decision for day `d`: irrigation and discharge (mm) per paddy."""
    dyn = torch.stack([
        pond / 50.0, soil / 100.0, torch.full_like(pond, season.rain[d] / 20.0),
        torch.full_like(pond, season.et0[d] / 8.0), demand[d] / 8.0,
        torch.full_like(pond, d / max(season.days - 1, 1)),
        (ir_tot[:, d] / group_area * 1000.0)[group] / 20.0,
        (dr_tot[:, d] / group_area * 1000.0)[group] / 20.0,
    ], dim=1).float()
    score = allocator(static, dyn).double()
    return _allocate(ir_tot[:, d], score[:, 0], group, area), _allocate(dr_tot[:, d], score[:, 1], group, area)


def simulate(allocator, ratio_field, season, keep_daily=False):
    """Run the season; returns season sums, daily group drainage and the ET loss term."""
    T = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt)
    area, group = T(season.area), T(season.group, torch.long)
    grid = water.Grid(season.neighbours, season.area)
    static = T(season.static, torch.float32)
    ir_tot, dr_tot = T(season.irrigation_m3), T(season.drainage_m3)
    group_area = torch.zeros(4, dtype=torch.float64).index_add(0, group, area)
    demand = T(season.et_demand)
    kz, shares = ratio_field.tensors()
    pond, soil = water.start(season.n)
    cyclic = group < 2
    north = group % 2 == 0
    infil = {"North": 0.0, "South": 0.0}
    lfv = 0.0
    drain_daily = []
    et_loss = 0.0
    unmet_total = 0.0
    daily = {"irrigation_mm": [], "drainage_mm": [], "et_mm": [], "unmet_mm": []} if keep_daily else None
    e_scale = float(season.et_demand.mean()) or 1.0
    for d in range(season.days):
        irrigation, discharge = farmer_day(allocator, season, d, pond, soil, static, area, group, group_area,
                                           ir_tot, dr_tot, demand)
        pond, soil, f = water.step(pond, soil, season.rain[d], irrigation, discharge, demand[d], kz, shares, grid)
        drain_daily.append(torch.zeros(4, dtype=torch.float64).index_add(0, group, f["drainage_mm"] * area / 1000.0))
        infil["North"] = infil["North"] + (f["infiltration_mm"] * area)[cyclic & north].sum() / 1000.0
        infil["South"] = infil["South"] + (f["infiltration_mm"] * area)[cyclic & ~north].sum() / 1000.0
        lfv = lfv + (f["lateral_mm"] * area)[cyclic].sum() / 1000.0
        et_loss = et_loss + ((f["unmet_mm"] / e_scale) ** 2).mean()
        unmet_total = unmet_total + (f["unmet_mm"] * area).sum()
        if keep_daily:
            for key, value in (("irrigation_mm", irrigation), ("drainage_mm", f["drainage_mm"]),
                               ("et_mm", f["et_mm"]), ("unmet_mm", f["unmet_mm"])):
                daily[key].append(value.detach().numpy())
    out = dict(drain_daily=torch.stack(drain_daily, dim=1), lf=torch.stack([infil["North"], infil["South"]]),
               lfv=lfv, et_loss=et_loss / season.days,
               unmet_share=unmet_total / (demand * area).sum())
    if keep_daily:
        out["daily"] = {k: np.stack(v) for k, v in daily.items()}
    return out


def loss_terms(out, season):
    obs = torch.as_tensor(season.drainage_m3)
    scale = obs.mean(1, keepdim=True).clamp_min(1.0)
    w = torch.tensor(DRAIN_GROUP_WEIGHT, dtype=torch.float64)[:, None]
    drain = (w * ((out["drain_daily"] - obs) / scale) ** 2).mean()
    lf_t = torch.as_tensor(season.lf_m3)
    lf = (((out["lf"] - lf_t) / lf_t) ** 2).sum()
    lfv = ((out["lfv"] - season.lfv_m3) / season.lfv_m3) ** 2
    return {"et": out["et_loss"], "drainage": drain, "lf": lf, "lfv": lfv,
            "total": out["et_loss"] + drain + TABLE_WEIGHT * (lf + lfv)}


# --- training, evaluation, saving ---------------------------------------------------------------

def train(season, epochs=300, lr_allocator=3e-3, lr_ratios=5e-2, seed=0, progress=print, every=25):
    """Fit A and B together on one season, as tightly as they will go (no regularisation)."""
    torch.manual_seed(seed)
    allocator = Allocator(season.n, season.static.shape[1])
    ratio_field = RatioField(season.n)
    opt = torch.optim.Adam([{"params": allocator.parameters(), "lr": lr_allocator},
                            {"params": ratio_field.parameters(), "lr": lr_ratios}])
    history = []
    for epoch in range(int(epochs)):
        opt.zero_grad()
        terms = loss_terms(simulate(allocator, ratio_field, season), season)
        terms["total"].backward()
        opt.step()
        history.append({k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in terms.items()})
        if progress and (epoch % every == 0 or epoch == epochs - 1):
            h = history[-1]
            progress(f"epoch {epoch + 1}/{epochs}  total {h['total']:.4f}  ET {h['et']:.4f}  "
                     f"drain {h['drainage']:.4f}  Lf {h['lf']:.4f}  LFV {h['lfv']:.4f}")
    return allocator, ratio_field, history


def evaluate(allocator, ratio_field, season):
    """Frozen run of a season: how well are its measurements reproduced?"""
    with torch.no_grad():
        out = simulate(allocator, ratio_field, season)
        terms = loss_terms(out, season)
    obs = season.drainage_m3
    sim = out["drain_daily"].numpy()
    nse = [1 - ((sim[g] - obs[g]) ** 2).sum() / max(((obs[g] - obs[g].mean()) ** 2).sum(), 1e-9) for g in range(2)]
    lf = out["lf"].numpy()
    return {
        "loss": {k: float(v) for k, v in terms.items()},
        "unmet_et_share": float(out["unmet_share"]),
        "drainage_nse_north": float(nse[0]), "drainage_nse_south": float(nse[1]),
        "lf_north_rel_err": float(lf[0] / season.lf_m3[0] - 1), "lf_south_rel_err": float(lf[1] / season.lf_m3[1] - 1),
        "lfv_rel_err": float(out["lfv"] / season.lfv_m3 - 1),
    }


def replay(allocator, ratio_field, season):
    """Recreate a season in the stem Field (the ABM's own Mesa water world): the station
    sends exactly the historical block irrigation, every farmer is model A, the ratios
    are model B's. Returns the same scores as `evaluate`, from the Field's own StepResults."""
    from stem.field import Field, FieldConfig
    from stem.params import SOIL_MAX_MM, START_POND_MM

    field = Field(FieldConfig(year=season.year, days=season.days, ratios=ratio_field.ratios(),
                              canal_depth=None, drain_velocity_m_s=None, pump_capacity_m3_s=None))
    field.rain_series = np.asarray(season.rain, float)
    field.et0_series = np.asarray(season.et0, float)
    field.pond = np.full(season.n, START_POND_MM)
    field.soil = np.full(season.n, SOIL_MAX_MM)
    T = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt)
    area, group = T(season.area), T(season.group, torch.long)
    group_area = torch.zeros(4, dtype=torch.float64).index_add(0, group, area)
    static, ir_tot, dr_tot, demand = T(season.static, torch.float32), T(season.irrigation_m3),         T(season.drainage_m3), T(season.et_demand)
    with torch.no_grad():
        for d in range(season.days):
            ir, dis = farmer_day(allocator, season, d, T(field.pond), T(field.soil), static, area, group,
                                 group_area, ir_tot, dr_tot, demand)
            ir = ir.numpy()
            field.apply_station({"irrigation_sent_mm": ir, "withdraw_m3": 1e12})
            field.day_step(ir, dis.numpy(), et_demand_mm=season.et_demand[d])
    H = field.history
    cyc = season.group < 2
    vol = lambda key, mask: float(sum((getattr(h, key)[mask] * season.area[mask]).sum() for h in H) / 1000.0)
    sim = np.stack([np.bincount(season.group, weights=h.drainage_mm * season.area / 1000.0, minlength=4) for h in H], 1)
    obs = season.drainage_m3
    nse = [1 - ((sim[g] - obs[g]) ** 2).sum() / max(((obs[g] - obs[g].mean()) ** 2).sum(), 1e-9) for g in range(2)]
    lf = [vol("infiltration_mm", cyc & (season.group == g)) for g in (0, 1)]
    unmet = sum((h.stress_mm * season.area).sum() for h in H) / (season.et_demand * season.area).sum()
    return {
        "unmet_et_share": float(unmet),
        "drainage_nse_north": float(nse[0]), "drainage_nse_south": float(nse[1]),
        "lf_north_rel_err": float(lf[0] / season.lf_m3[0] - 1), "lf_south_rel_err": float(lf[1] / season.lf_m3[1] - 1),
        "lfv_rel_err": float(vol("to_pool_mm", cyc) / season.lfv_m3 - 1),
    }


def baseline(season):
    """Untrained reference: equal scores (area-weighted split) and uniform ratios."""
    torch.manual_seed(0)
    allocator = Allocator(season.n, season.static.shape[1])
    for p in allocator.net[-1].parameters():
        nn.init.zeros_(p)
    return allocator, RatioField(season.n)


def ratio_recovery(learned: water.Ratios, truth: water.Ratios, mask=None):
    """How well learned per-paddy ratios match the true ones (synthetic twin)."""
    mask = np.ones(truth.n, bool) if mask is None else mask
    out = {}
    for name, a, b in [("kz", learned.kz, truth.kz)] + [
            (s, learned.shares[:, k], truth.shares[:, k]) for k, s in enumerate(water.SHARES)]:
        a, b = a[mask], b[mask]
        out[name] = {"corr": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan"),
                     "mae": float(np.abs(a - b).mean()), "bias": float((a - b).mean()),
                     "truth_sd": float(b.std())}
    return out


def save(allocator, ratio_field, path, **meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"allocator": allocator.state_dict(), "config": allocator.config,
                "kz_logit": ratio_field.kz_logit.detach(), "share_logits": ratio_field.share_logits.detach(),
                "meta": meta}, path)
    return path


def load(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    allocator = Allocator(**blob["config"])
    allocator.load_state_dict(blob["allocator"])
    ratio_field = RatioField(blob["config"]["n"])
    with torch.no_grad():
        ratio_field.kz_logit.copy_(blob["kz_logit"])
        ratio_field.share_logits.copy_(blob["share_logits"])
    return allocator, ratio_field, blob.get("meta", {})


def real_ratios():
    """The ratios fitted on the measured data (for the ABM), or None if not trained yet."""
    return water.Ratios.load(REAL_RATIOS) if os.path.isfile(REAL_RATIOS) else None

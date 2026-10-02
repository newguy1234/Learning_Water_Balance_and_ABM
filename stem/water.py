"""The physics: one day of water on every paddy, shared by cyclic (the learner) and abm.

Two stores per paddy, in mm over its own area: the pond and the soil (0 … SOIL_MAX_MM).
Fixed per-paddy hydrology ratios decide where water goes:

    kz        share of the pond that enters the soil each day
    of the soil water each day, shares summing to 1:
      kx      to the east / west neighbours (split between those that exist)
      ky      to the north / south neighbours
      ground  to deep soil (lost to the district)
      lateral to the main drainage → the station pool on cyclic paddies (Okura's
              LFV_pump), to the lake / river off the cyclic area
      stay    stays in this paddy's soil

The day: rain and irrigation land on the pond; ET demand is met from the pond, then the
soil (what neither can supply is unmet); the farmer's discharge (and pond above an
outlet weir) leaves, then bund overflow; kz of the pond infiltrates (up to the soil's
room); the soil water is routed by the shares; water a neighbour cannot hold surfaces
on its pond. Everything is torch so the learner can differentiate through it; the ABM
calls `step_numpy`.

Storage identity, per paddy per day:
    Δ(pond + soil) = rain + irrigation − ET − drainage − deep − lateral − sent + received
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from .params import BUND_MM, DEFAULT_KZ, DEFAULT_SHARES, FITTED_RATIOS, SOIL_MAX_MM, START_POND_MM

SHARES = ("kx", "ky", "ground", "lateral", "stay")


@dataclass
class Ratios:
    """Per-paddy hydrology ratios: kz (n,) and shares (n, 5) in SHARES order, rows sum to 1."""

    kz: np.ndarray
    shares: np.ndarray

    @property
    def n(self):
        return len(self.kz)

    def share(self, name):
        return self.shares[:, SHARES.index(name)]

    @classmethod
    def uniform(cls, n, kz=DEFAULT_KZ, shares=DEFAULT_SHARES):
        return cls(np.full(n, float(kz)), np.tile(np.asarray(shares, dtype=float), (n, 1)))

    @classmethod
    def random(cls, n, seed=0, spatial=None):
        """Random but plausible ratios. `spatial` (n, k) features make them smooth in space."""
        rng = np.random.default_rng(seed)
        base = np.log(np.array(DEFAULT_SHARES))
        if spatial is None:
            z = rng.normal(0, 0.6, (n, 6))
        else:
            z = spatial @ rng.normal(0, 0.6, (spatial.shape[1], 6)) + rng.normal(0, 0.25, (n, 6))
        logits = base + z[:, 1:]
        shares = np.exp(logits - logits.max(1, keepdims=True))
        shares /= shares.sum(1, keepdims=True)
        kz = 1.0 / (1.0 + np.exp(-(np.log(DEFAULT_KZ / (1 - DEFAULT_KZ)) + z[:, 0])))
        return cls(kz, shares)

    def save(self, path, **meta):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, kz=self.kz, shares=self.shares, share_names=np.array(SHARES),
                            **{k: np.asarray(v) for k, v in meta.items()})

    @classmethod
    def load(cls, path):
        blob = np.load(path, allow_pickle=False)
        return cls(blob["kz"].astype(float), blob["shares"].astype(float))


def fitted_ratios(path=FITTED_RATIOS):
    """The ratios cyclic.learn fitted on the measured seasons, or None if not trained yet."""
    return Ratios.load(path) if os.path.isfile(path) else None


def start(n, dtype=torch.float64):
    """Pond START_POND_MM and a saturated soil, as torch tensors."""
    return torch.full((n,), START_POND_MM, dtype=dtype), torch.full((n,), SOIL_MAX_MM, dtype=dtype)


class Grid:
    """Paddy areas and the cardinal neighbour network, as tensors, built once per world."""

    def __init__(self, neighbours, area_m2, dtype=torch.float64):
        nbr = torch.as_tensor(np.asarray(neighbours), dtype=torch.long)
        self.area = torch.as_tensor(np.asarray(area_m2, dtype=float), dtype=dtype)
        self.n = len(self.area)
        self.axes = []
        for cols in ([0, 1], [2, 3]):              # east/west, north/south
            sub = nbr[:, cols]
            present = sub >= 0
            count = present.sum(1)
            src = present.nonzero(as_tuple=True)[0]
            self.axes.append((count > 0, count.clamp_min(1).to(dtype), src, sub[present]))

    def send(self, amount_mm, axis):
        """Split `amount_mm` of each paddy evenly among its neighbours on `axis` (0 = x, 1 = y)."""
        has, count, src, dst = self.axes[axis]
        sent = torch.where(has, amount_mm, torch.zeros_like(amount_mm))
        per = sent * self.area / count
        received = torch.zeros_like(amount_mm).index_add(0, dst, per[src]) / self.area
        return sent, received


def step(pond, soil, rain, irrigation, discharge, et_demand, kz, shares, grid,
         bund=BUND_MM, soil_max=SOIL_MAX_MM, weir=None):
    """One day on every paddy. Per-paddy arguments are torch tensors (n,); `shares`
    (n, 5); `grid` a Grid. Returns the new pond and soil and the day's fluxes in mm."""
    pond = pond + rain + irrigation
    et_pond = torch.minimum(et_demand, pond)
    pond = pond - et_pond
    et_soil = torch.minimum(et_demand - et_pond, soil)
    soil = soil - et_soil
    unmet = et_demand - et_pond - et_soil

    out = torch.minimum(discharge.clamp_min(0.0), pond)
    pond = pond - out
    if weir is not None:
        over_weir = (pond - weir).clamp_min(0.0)
        pond = pond - over_weir
        out = out + over_weir
    overflow = (pond - bund).clamp_min(0.0)
    pond = pond - overflow

    infiltration = torch.minimum(kz * pond, (soil_max - soil).clamp_min(0.0))
    pond = pond - infiltration
    soil = soil + infiltration

    kx, ky, ground, lateral = (shares[:, k] * soil for k in range(4))
    to_x, from_x = grid.send(kx, 0)
    to_y, from_y = grid.send(ky, 1)
    soil = soil - to_x - to_y - ground - lateral + from_x + from_y
    surfaced = (soil - soil_max).clamp_min(0.0)
    soil = soil - surfaced
    pond = pond + surfaced
    return pond, soil, {
        "et_mm": et_pond + et_soil, "unmet_mm": unmet, "discharge_mm": out, "overflow_mm": overflow,
        "drainage_mm": out + overflow, "infiltration_mm": infiltration, "to_x_mm": to_x, "to_y_mm": to_y,
        "received_mm": from_x + from_y, "deep_mm": ground, "lateral_mm": lateral, "surfaced_mm": surfaced,
        "rise_mm": et_soil,
    }


def step_numpy(pond, soil, rain, irrigation, discharge, et_demand, ratios, grid,
               bund=BUND_MM, soil_max=SOIL_MAX_MM, weir=None):
    """`step` on numpy arrays (the ABM). Returns numpy pond, soil and fluxes."""
    t = lambda a: torch.from_numpy(np.require(np.broadcast_to(np.asarray(a, dtype=float), (grid.n,)), requirements="CW"))
    with torch.no_grad():
        p, s, flux = step(t(pond), t(soil), t(rain), t(irrigation), t(discharge), t(et_demand),
                          torch.from_numpy(np.require(ratios.kz, float, "CW")),
                          torch.from_numpy(np.require(ratios.shares, float, "CW")), grid,
                          bund=bund, soil_max=soil_max, weir=None if weir is None else t(weir))
    return p.numpy().copy(), s.numpy().copy(), {k: v.numpy().copy() for k, v in flux.items()}

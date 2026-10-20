# -*- coding: utf-8 -*-
"""One-at-a-time sensitivity of every farmer trait.

A single paddy, 131 days, station sends whatever the farmer asks. Weather is a
fixed season with a dry spell and a hot spell, so forecast use, heat flow-through
and days away can show up. Neighbours sit 15 mm wetter than this field, so
cooperation and the wetness offset can move the target. This isolates the
decision rule. It is not the 4,002-paddy district run.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load the farmer modules without pulling geopandas through abm/__init__.py.
pkg = types.ModuleType("abm")
pkg.__path__ = [str(ROOT / "abm")]
pkg.__package__ = "abm"
sys.modules["abm"] = pkg

import abm.crop as crop  # noqa: E402
import abm.farmer as farmer  # noqa: E402
import abm.agents as agents  # noqa: E402

DAYS = 131
# 17 April 2022 was a Sunday.
WEEKDAY0 = 6


def weather():
    rng = np.random.default_rng(2022)
    rain = rng.gamma(0.35, 7.0, DAYS)
    rain[60:84] = 0.0
    et0 = 3.2 + 1.6 * np.sin(np.linspace(0.2, np.pi, DAYS))
    et0[75:89] = 6.2
    return rain, et0


RAIN, ET0 = weather()


def run(overrides: dict, price: float = 0.0, learn: bool = False) -> dict:
    cls = agents.PersonalityFarmer if learn else farmer.BaseFarmer
    person = cls(
        trait_variability=0.0, schedule_jitter=0.0, week_swap=0.0, seed=1,
        target_multiplier=1.0, neighbour_weight=0.3, **overrides,
    )
    pond = 30.0
    demand = applied = overflow = unmet = 0.0
    checks = 0
    daily = []
    for day in range(DAYS):
        weekday = (WEEKDAY0 + day) % 7
        cd = person.crop_day(day)
        etc = crop.kc_of_day(max(cd, 0)) * float(ET0[day])
        forecast = np.zeros(8)
        for k in range(1, 8):
            if day + k < DAYS:
                forecast[k] = RAIN[day + k]
        obs = dict(
            day=day, weekday=weekday,
            owned_ids=np.array([0]),
            owned_ponding_mm=np.array([pond]),
            owned_irrigation_sent_mm=np.array([1.0e6]),
            ponding_mm=np.array([pond]),
            irrigation_sent_mm=np.array([1.0e6]),
            rain_mm=float(RAIN[day]),
            et_mm=etc, et0_mm=float(ET0[day]),
            owned_et_demand_mm=np.array([etc]),
            owned_seepage_mm=np.array([7.0 if pond > 1 else 0.0]),
            neighbour_mean_mm=np.array([pond + 15.0]),
            rain_forecast_mm=forecast,
            bund_mm=250.0,
            water_price_per_m3=price,
            area_m2=np.array([1000.0]),
        )
        if learn and person.checked_today is not None:
            person.memory.append(dict(ks=0.7, scarcity=0.4))
        plan = person.act(obs)
        take = float(plan.apply[0])
        weir = float(plan.weir[0])
        seep = 7.0 if pond > 1 else 0.0
        pond = pond + float(RAIN[day]) + take - etc - seep
        over = max(pond - weir, 0.0)
        pond -= over
        short = max(-pond, 0.0)
        pond = max(pond, 0.0)
        demand += float(plan.request[0])
        applied += take
        overflow += over
        unmet += short
        checks += int(person.checked_today)
        daily.append(float(plan.request[0]))
    roll = np.convolve(daily, np.ones(7), mode="valid")
    return dict(demand=demand, applied=applied, overflow=overflow, unmet=unmet,
                checks=checks, peak7=float(roll.max()) if len(roll) else 0.0)


def main():
    base = run({})
    print(f"{'trait':22} {'low':>8} {'high':>8} {'dDemand':>8} {'dApply':>8} {'dOver':>8} {'dCheck':>8} {'dPeak':>8}")
    rows = []
    for trait in farmer.TRAITS:
        low = run({trait.name: trait.low})
        high = run({trait.name: trait.high})
        def pct(key):
            den = base[key]
            return 100.0 * (high[key] - low[key]) / den if den else 0.0
        rec = dict(
            trait=trait.name,
            demand=pct("demand"),
            applied=pct("applied"),
            overflow=pct("overflow"),
            checks=pct("checks"),
            peak=pct("peak7"),
        )
        rows.append(rec)
        print(f"{trait.name:22} {trait.low:8.2f} {trait.high:8.2f} "
              f"{rec['demand']:8.1f} {rec['applied']:8.1f} {rec['overflow']:8.1f} "
              f"{rec['checks']:8.1f} {rec['peak']:8.1f}")
    print("\nreference", {k: round(v, 1) for k, v in base.items()})
    print("traits kept", len(farmer.TRAITS))


if __name__ == "__main__":
    main()

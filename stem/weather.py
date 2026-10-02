# -*- coding: utf-8 -*-
"""Tweakable weather scenarios for ABM seasons.

A scenario reshapes one base year's gauge-corrected rain and ET0 — weather no
farmer or staff policy controls, only reacts to. Every knob is optional and they
compose in this order:

  1. ``rain_scale`` / ``et0_scale`` over the whole season, then per calendar month
     (``monthly_rain_scale`` / ``monthly_et0_scale``, e.g. ``{7: 0.3}`` = dry July)
  2. ``dry_spells``   [(start_day, length)]: no rain on those days
  3. ``heatwaves``    [(start_day, length, et0_factor)]: ET0 multiplied on those days
  4. ``storms``       [(day, mm)]: rain added on that day
  5. ``variability``  day-to-day log-normal jitter (0 = none), drawn from ``seed``

Days count from the season start (17 April, day 0).

    from stem.weather import WeatherScenario, weather_scenario

    weather_scenario("drought")                               # a preset
    weather_scenario("drought", heatwaves=[(80, 10, 1.3)])    # preset, tweaked
    WeatherScenario(rain_scale=0.7, storms=[(95, 150.0)])     # built from scratch

Pass one as ``SeasonConfig(weather=...)`` or ``AbmEnv(weather=...)`` (a preset
name works too). ``stem.weather.PRESETS`` lists the named scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class WeatherScenario:
    name: str = "custom"
    rain_scale: float = 1.0
    et0_scale: float = 1.0
    monthly_rain_scale: dict[int, float] = field(default_factory=dict)
    monthly_et0_scale: dict[int, float] = field(default_factory=dict)
    dry_spells: list[tuple[int, int]] = field(default_factory=list)
    heatwaves: list[tuple[int, int, float]] = field(default_factory=list)
    storms: list[tuple[int, float]] = field(default_factory=list)
    variability: float = 0.0
    seed: int | None = None  # None → the season's seed

    def apply(self, rain_mm, et0_mm, year: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        from stem import okura
        from .field import perturb_weather

        rain = np.asarray(rain_mm, dtype=float) * float(self.rain_scale)
        et0 = np.asarray(et0_mm, dtype=float) * float(self.et0_scale)
        months = np.asarray(okura.season_dates(year)[: rain.size].month)
        for month, factor in self.monthly_rain_scale.items():
            rain = np.where(months == int(month), rain * float(factor), rain)
        for month, factor in self.monthly_et0_scale.items():
            et0 = np.where(months == int(month), et0 * float(factor), et0)
        for start, length in self.dry_spells:
            rain[int(start): int(start) + int(length)] = 0.0
        for start, length, factor in self.heatwaves:
            et0[int(start): int(start) + int(length)] *= float(factor)
        for day, mm in self.storms:
            if 0 <= int(day) < rain.size:
                rain[int(day)] += float(mm)
        jitter_seed = int(seed if self.seed is None else self.seed)
        return perturb_weather(rain, et0, float(self.variability), jitter_seed)

    def summary(self, rain_mm, et0_mm) -> dict:
        return {"scenario": self.name, "rain_mm": float(np.sum(rain_mm)), "et0_mm": float(np.sum(et0_mm))}


PRESETS: dict[str, WeatherScenario] = {
    "baseline": WeatherScenario(name="baseline"),
    # Okura 2022 was already a drought year (410 mm vs 638 mm 30-year mean); these push further.
    "drought": WeatherScenario(name="drought", rain_scale=0.5, et0_scale=1.1, dry_spells=[(60, 25)]),
    "wet": WeatherScenario(name="wet", rain_scale=1.5, et0_scale=0.95),
    "heatwave": WeatherScenario(name="heatwave", heatwaves=[(75, 14, 1.35)], dry_spells=[(75, 14)]),
    "storm": WeatherScenario(name="storm", storms=[(90, 150.0), (91, 80.0)]),
    "dry_july": WeatherScenario(name="dry_july", monthly_rain_scale={7: 0.2}, monthly_et0_scale={7: 1.15}),
    "erratic": WeatherScenario(name="erratic", variability=0.4),
}


def weather_scenario(name: str | WeatherScenario | None = "baseline", **overrides) -> WeatherScenario:
    """A preset (or an existing scenario) with any fields overridden."""
    if isinstance(name, WeatherScenario):
        base = name
    else:
        key = name or "baseline"
        if key not in PRESETS:
            raise KeyError(f"unknown weather scenario {key!r}; choose from {sorted(PRESETS)}")
        base = PRESETS[key]
    return replace(base, **overrides) if overrides else replace(base)


def list_weather_scenarios() -> list[str]:
    return sorted(PRESETS)

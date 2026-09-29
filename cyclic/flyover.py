"""Relative spatial consistency against an NDVI-derived ET proxy, not actual-ET validation.

Current scoring normalizes simulated and proxy values by their own area-weighted
means. Fidelity averages relative distribution and paired spatial agreement.
The legacy level_fidelity field is diagnostic only and is excluded from the score.
The absolute proxy still uses weather ET0. It is not an independent observation.

A confirmed fly-over is a satellite pass with cloud-free pixels over a paddy. On those days,
for each block, two things are asked of the ET the farmers produced:

    level         the block's mean ET should be the field-average ET from Open-Meteo --
                  ET0 times the field's Kc -- which is what the district as a whole must
                  have transpired that day
                      level fidelity = 1 - |simulated mean - field mean| / field mean

    distribution  the spread of ET across the block's paddies should be the satellite's:
                  which paddies ran high and which ran low, and by how much. The area-
                  weighted quantiles of simulated and satellite ET are compared, as the
                  1-Wasserstein distance W1 (mean absolute gap between matching quantiles),
                  relative to the satellite mean
                      distribution fidelity = 1 - W1 / satellite mean

    Historical scoring (superseded): fidelity = the mean of the two, each floored at 0. Paired fidelity -- every paddy
    against its own pixels, 1 - sum|sim - sat| / sum sat -- is reported beside it: a
    district can have the right distribution with the wrong paddies in each part of it.

Every row also carries the actual values: the simulated and satellite mean ET and their
10th and 90th percentiles across the block's paddies, in mm/d.

    score_day(inputs, day, simulated_et)   rows for one day (empty without a fly-over)
    summary(rows)                          a season's rows as a table and area-weighted scores
    from_results(results)                  the same, from a finished run
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import okura

QUANTILES = np.linspace(0.01, 0.99, 99)
MIN_PADDIES = 10
COLUMNS = ["date", "day", "block", "paddies", "area_ha", "simulated_mean_mm", "simulated_p10_mm",
           "simulated_p90_mm", "satellite_mean_mm", "satellite_p10_mm", "satellite_p90_mm", "field_mean_mm",
           "simulated_sd_mm", "satellite_sd_mm", "w1_mm", "level_fidelity", "distribution_fidelity",
           "paired_fidelity", "fidelity"]


def weighted_quantiles(values, weights, quantiles=QUANTILES):
    order = np.argsort(values)
    values, weights = np.asarray(values, dtype=float)[order], np.asarray(weights, dtype=float)[order]
    position = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(quantiles, position, values)


def _weighted_sd(values, weights):
    mean = np.average(values, weights=weights)
    return float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))


def score_day(inputs, day, simulated_et):
    """One row per block with a confirmed fly-over on this day."""
    passed = inputs.is_pass[day]
    if not passed.any():
        return []
    satellite_today = inputs.sat_pass_et_mm[day]
    rows = []
    for block in okura.BLOCKS:
        mask = passed & (inputs.block == block) & np.isfinite(satellite_today)
        if mask.sum() < MIN_PADDIES:
            continue
        weights = inputs.area_m2[mask]
        simulated = np.asarray(simulated_et, dtype=float)[mask]
        satellite = satellite_today[mask].astype(float)
        field = float(inputs.field_et_mm[block][day])
        simulated_mean = float(np.average(simulated, weights=weights))
        satellite_mean = float(np.average(satellite, weights=weights))
        sim_p10, sim_p90 = weighted_quantiles(simulated, weights, [0.1, 0.9])
        sat_p10, sat_p90 = weighted_quantiles(satellite, weights, [0.1, 0.9])
        w1 = float(np.mean(np.abs(weighted_quantiles(simulated, weights) - weighted_quantiles(satellite, weights))))
        level = max(0.0, 1.0 - abs(simulated_mean - field) / max(field, 1e-6))
        # Compare relative spatial patterns, not the shared weather-derived ET
        # magnitude. This is a proxy-consistency diagnostic, not ET validation.
        if simulated_mean <= 1e-6 or satellite_mean <= 1e-6:
            continue
        sim_relative = simulated / simulated_mean
        sat_relative = satellite / satellite_mean
        shape_gap = np.mean(np.abs(weighted_quantiles(sim_relative, weights)
                                   - weighted_quantiles(sat_relative, weights)))
        distribution = max(0.0, 1.0 - float(shape_gap))
        paired = max(0.0, 1.0 - float(np.average(abs(sim_relative - sat_relative), weights=weights)))
        rows.append({
            "date": inputs.dates[day], "day": int(day), "block": block, "paddies": int(mask.sum()),
            "area_ha": float(weights.sum()) / 1e4,
            "simulated_mean_mm": simulated_mean, "simulated_p10_mm": float(sim_p10), "simulated_p90_mm": float(sim_p90),
            "satellite_mean_mm": satellite_mean, "satellite_p10_mm": float(sat_p10), "satellite_p90_mm": float(sat_p90),
            "field_mean_mm": field, "simulated_sd_mm": _weighted_sd(simulated, weights),
            "satellite_sd_mm": _weighted_sd(satellite, weights), "w1_mm": w1, "level_fidelity": level,
            "distribution_fidelity": distribution, "paired_fidelity": paired,
            "fidelity": 0.5 * (paired + distribution),
        })
    return rows


def summary(rows):
    """A season of fly-over rows: the table, and each score weighted by the area it covered."""
    table = pd.DataFrame(rows, columns=COLUMNS)
    if table.empty:
        nan = float("nan")
        return SimpleNamespace(table=table, fidelity=nan, level=nan, distribution=nan, paired=nan, passes=0)
    weights = table["area_ha"]

    def mean(column):
        return float(np.average(table[column], weights=weights))

    return SimpleNamespace(table=table, fidelity=mean("fidelity"), level=mean("level_fidelity"),
                           distribution=mean("distribution_fidelity"), paired=mean("paired_fidelity"),
                           passes=int(table["day"].nunique()))


def from_results(results):
    """The fly-over scores of a finished run that recorded its paddies."""
    model = results.model
    if not model.log:
        raise ValueError("fly-over scores need a run that recorded its paddies")
    days = len(results.daily)
    rows = []
    for day in np.flatnonzero(model.inputs.is_pass[:days].any(axis=1)):
        rows.extend(score_day(model.inputs, int(day), model.log["et_mm"][day]))
    return summary(rows)

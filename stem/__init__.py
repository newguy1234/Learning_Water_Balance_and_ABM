"""stem: the physical world the cyclic learner and the ABM share.

    district   the 4,002 Hakusan-Jimbee paddies, their neighbours, static features and
               one season's weather (Open-Meteo, rain scaled to the site gauge)
    water      the water balance: one day on every paddy, driven by each paddy's fixed
               hydrology ratios (kz, kx, ky, ground, lateral, stay)
    okura      Okura et al. (2025): season dates and the published water-balance tables
    infrastructure  pumping station, drainage channels, main-canal depth (backwater)
    params     physical parameter settings

Nothing here learns or decides: cyclic fits the ratios, abm's agents decide the water.
"""

from . import district, infrastructure, okura, params, water
from .district import District, load_district
from .water import Grid, Ratios, SHARES, fitted_ratios, step, step_numpy

__all__ = ["district", "infrastructure", "okura", "params", "water", "District", "load_district",
           "Grid", "Ratios", "SHARES", "fitted_ratios", "step", "step_numpy"]

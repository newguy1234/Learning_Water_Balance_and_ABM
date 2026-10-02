"""cyclic: learn each paddy's fixed hydrology ratios from past irrigation seasons.

    satellite  per-paddy ET from Landsat thermal (SSEBop), interpolated between fly-overs
               with Open-Meteo ET0
    observed   what Okura et al. (2025) measured: daily block irrigation / drainage,
               season lateral flow (Lf) and recapture at the pool (LFV)
    learn      model A (neural net placing the block irrigation and discharge on paddies)
               and model B (per-paddy ratios on the simplex), fitted together through the
               stem water balance; frozen evaluation and a replay in the stem Field

The world itself (paddies, water balance, Mesa field) is `stem`.
"""

from . import learn, observed, satellite

__all__ = ["learn", "observed", "satellite"]

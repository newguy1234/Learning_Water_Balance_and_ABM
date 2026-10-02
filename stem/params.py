"""Physical parameter settings shared by the cyclic learner and the ABM.

Every constant here is a property of the district or of rice, with its source.
Per-paddy hydrology ratios are not here: they are fitted (cyclic.learn) and loaded
with `stem.water.fitted_ratios`.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
DATA = os.path.join(PROJECT, "data")
BUNDLED = os.path.join(HERE, "bundled")
OUTPUTS = os.path.join(PROJECT, "outputs")
FITTED_RATIOS = os.path.join(OUTPUTS, "learned", "ratios_real.npz")

# Soil and pond.
SOIL_MAX_MM = 100.0    # FAO-56 total available water of a puddled 0.5 m clay-loam profile
BUND_MM = 250.0        # 15–30 cm bunds (IRRI guidance; Chapagain & Yamaji 2010)
START_POND_MM = 30.0   # puddled and flooded when the irrigation period starts (Okura §2.5)

# Ratios a paddy has before any are fitted: kz, then kx, ky, ground, lateral, stay.
DEFAULT_KZ = 0.25
DEFAULT_SHARES = (0.05, 0.05, 0.05, 0.40, 0.45)

# FAO-56 dual crop coefficient for rice (Allen et al. 1998); wet-day Kc cap Xu et al. (2017).
KCB_BARE, KCB_FULL, KC_WET = 0.15, 1.15, 1.29

# Station and pool (Okura et al. 2025 §2.1; MAFF 白山甚兵衛機場).
POOL_CAPACITY_M3 = 18.0 * 68.0 * 4.0   # 18 × 68 m pool, depth assumed 4 m
PUMP_CAPACITY_M3_S = 3.304

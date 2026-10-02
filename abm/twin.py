"""Synthetic twin: can cyclic.learn recover hydrology ratios it did not see?

The ABM runs a season on the real district with random per-paddy ratios (smooth in
space, like a soil map, plus paddy-to-paddy noise) and its own farmers and station.
What a measurement campaign would have recorded becomes the training data:
    • each block's daily irrigation and drainage (cyclic and non-cyclic)
    • each paddy's daily ET (what the satellite would see)
    • water into the ground per cyclic block (Lf) and lateral flow to the pool (LFV)
Models A and B are then trained from scratch on that alone (cyclic.learn.train) and the
learned ratios are compared with the true ones. A second ABM season (the other year,
same true ratios) checks the frozen models on weather they were not trained on.
"""

from __future__ import annotations

import numpy as np

from cyclic import learn, water
from stem.district import load_district

from .model import DistrictModel
from .world import SeasonConfig


def random_truth(year=2021, seed=0):
    district = load_district(year)
    return water.Ratios.random(district.n, seed=seed, spatial=learn.static_features(district))


def synthetic_season(truth, year=2021, seed=0):
    """Run the ABM with `truth` ratios; return what was 'measured' as a learn.Season."""
    model = DistrictModel(SeasonConfig(year=year, seed=seed, ownership_seed=seed, ratios=truth), seed=seed)
    while not model.world.done():
        model.step()
    w = model.world
    district = w.district
    group = district.group_index()
    area = w.area_m2
    hist = w.history
    per_group = lambda key: np.stack([np.bincount(group, weights=getattr(h, key) * area / 1000.0, minlength=4)
                                      for h in hist], axis=1)
    infil = np.stack([h.infiltration_mm for h in hist])
    lf = np.array([(infil[:, w.cyclic & (w.block == b)] * area[w.cyclic & (w.block == b)]).sum() / 1000.0
                   for b in ("North", "South")])
    lfv = float(sum((h.to_pool_mm * area).sum() for h in hist) / 1000.0)
    return learn.Season(year=year, area=area, group=group, neighbours=district.neighbours,
                        static=learn.static_features(district), rain=np.asarray(w.rain_series, float),
                        et0=np.asarray(w.et0_series, float), et_demand=np.stack([h.eta_mm for h in hist]),
                        irrigation_m3=per_group("irrigation_mm"), drainage_m3=per_group("drainage_mm"),
                        lf_m3=lf, lfv_m3=lfv, source="abm", truth=truth)


def run_twin(seed=0, train_year=2021, test_year=2022, epochs=300, progress=print):
    """Train A+B on an ABM season with random ratios; score ratio recovery and a frozen year."""
    truth = random_truth(train_year, seed)
    train = synthetic_season(truth, train_year, seed)
    test = synthetic_season(truth, test_year, seed + 1)
    allocator, ratio_field, history = learn.train(train, epochs=epochs, progress=progress)
    learned = ratio_field.ratios()
    cyclic = train.group < 2
    return {
        "seed": seed, "train_year": train_year, "test_year": test_year, "epochs": epochs,
        "final_loss": history[-1],
        "recovery_all": learn.ratio_recovery(learned, truth),
        "recovery_cyclic": learn.ratio_recovery(learned, truth, cyclic),
        "baseline_recovery_all": learn.ratio_recovery(water.Ratios.uniform(truth.n), truth),
        "fit_train": learn.evaluate(allocator, ratio_field, train),
        "frozen_test": learn.evaluate(allocator, ratio_field, test),
        "baseline_test": learn.evaluate(*learn.baseline(test), test),
    }, learned, truth

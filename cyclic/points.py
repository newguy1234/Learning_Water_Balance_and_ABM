"""Per-point irrigation, discharge and underground accounting, without Darcy.

Satellite flood sets irrigation and discharge. Satellite ET is forced to occur:
transpiration from the root zone, evaporation from the pond if flooded else the
surface. Leftover IR+R−D−ET is an underground budget.

    python -m cyclic points --csv underground.csv
"""

from __future__ import annotations

import os

from .agents import SatellitePhysics
from . import okura
from .model import CyclicModel
from .physics import Physics


def run(year=okura.COMPARISON_YEAR, physics=None, csv=None, grain="paddy",
        pond_mm=30.0, forcing="station"):
    physics = physics or Physics(force_satellite_et=True)
    if not physics.force_satellite_et:
        physics = physics.replace(force_satellite_et=True)
    model = CyclicModel(year, agents=SatellitePhysics(pond_mm=pond_mm), physics=physics,
                        forcing=forcing, record_paddies=True)
    results = model.run()
    paths = None
    if csv:
        os.makedirs(os.path.dirname(os.path.abspath(csv)) or ".", exist_ok=True)
        paths = results.underground_csv(csv, grain=grain)
    return results, paths


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=okura.COMPARISON_YEAR)
    parser.add_argument("--csv", default="underground.csv")
    parser.add_argument("--grain", choices=("paddy", "cell"), default="paddy")
    parser.add_argument("--pond-mm", type=float, default=30.0)
    args = parser.parse_args(argv)
    results, paths = run(args.year, csv=args.csv, grain=args.grain, pond_mm=args.pond_mm)
    inverse = results.model.inverse
    n = len(results.model.inputs.paddy_id)
    print(f"season {args.year}: {n} points, satellite ET forced, no Darcy")
    if inverse is not None:
        print("O'Malley split (leftover → lateral vs stay):")
        for block, err in inverse.lf_error.items():
            print(f"  Lf {block} error {100 * err:.1f}%  ({inverse.lateral_m3[block]:.0f} m3)")
        print(f"  LFV_pump error {100 * inverse.lfv_error:.1f}%  ({inverse.lfv_m3:.0f} m3)")
        print(f"  points assigned lateral {int(inverse.x.sum())} / {n}")
    if paths:
        print("wrote", paths[0], "and", paths[1])

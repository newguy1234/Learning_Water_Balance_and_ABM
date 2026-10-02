# Contribution: inferred underground flow ratios

## Train on both 2021 and 2022

```
CONTRIBUTION — inferred underground flow ratios
====================================================
inferred underground flow ratios: per-point leftover capture (Kx+Ky -> LFV) vs stay (Kz+ground -> deep/dWB), under hard-copy surface IR/D/W (Fig. 8 + ADCP / Table 4; not estimated schedules)

Not claimed: measured subsurface paths; unique Kx vs Ky axis; novel IR schedule / Fig. 8 timing; that district Lf or LFV_pump totals are newly observed; estimating surface IR/D/W when digitised paper series exist

Train both years: 2021, 2022
  joint capture (area-weighted) = 0.8500
  2021 Table 5 pin = 0.7384   error^2 = 0.012460
  2022 Table 5 pin = 0.7873   error^2 = 0.003925
  mean capture error (sq)     = 0.008192

Spatial field (2061 cyclic / 4925 points)
  capture mean / std / CV = 0.8500 / 0.0411 / 0.048
  capture range           = [0.8187, 1.0000]
  stay mean               = 0.1500
  Kx~Ky (half capture)    = 0.4250 each (axis not identified)
  Kz / ground             = 0.0300 / 0.1200

Pipeline: known IR/D -> cover ET -> penetrate plow pan -> neighbour hops -> per-point (Kx+Ky) LFV vs (Kz+ground) deep stay.
```

## Optional single-year freeze (2021 -> 2022)

```
CONTRIBUTION — inferred underground flow ratios
====================================================
inferred underground flow ratios: per-point leftover capture (Kx+Ky -> LFV) vs stay (Kz+ground -> deep/dWB), under hard-copy surface IR/D/W (Fig. 8 + ADCP / Table 4; not estimated schedules)

Not claimed: measured subsurface paths; unique Kx vs Ky axis; novel IR schedule / Fig. 8 timing; that district Lf or LFV_pump totals are newly observed; estimating surface IR/D/W when digitised paper series exist

Holdout: train 2021 -> eval 2022
  train capture (area-weighted) = 0.8500
  eval Table 5 pin              = 0.7873
  capture error (sq)            = 0.003925

Spatial field (2061 cyclic / 4925 points)
  capture mean / std / CV = 0.8500 / 0.0396 / 0.047
  capture range           = [0.8198, 1.0000]
  stay mean               = 0.1500
  Kx~Ky (half capture)    = 0.4250 each (axis not identified)
  Kz / ground             = 0.0300 / 0.1200

Pipeline: known IR/D -> cover ET -> penetrate plow pan -> neighbour hops -> per-point (Kx+Ky) LFV vs (Kz+ground) deep stay.
```

## Claim boundary

Surface IR/W (Fig. 8) and ADCP drainage are **forcing**, not novelty.
We infer how leftover after ET is partitioned underground, per point,
with district mean capture trained on both Okura seasons (Table 5 LFV/LfA),
hard-floored so the frozen field holds across wet and dry years.

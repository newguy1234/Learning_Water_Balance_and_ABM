# Inferred underground flow ratios under cyclic irrigation

*Working manuscript generated from the `cyclic` package. Re-run `python -m cyclic paper 2022` after a code change. `tests/test_paper_claims.py` fails if the text and the code disagree.*

## Abstract

**Contribution: inferred underground flow ratios.** With today's pool irrigation and surface drainage treated as known forcing (Okura, Kamei and Kato, 2025; digitised Fig. 8 IR/W), we infer how leftover after FAO-56 ET is partitioned *underground* at each point: capture \(K_x+K_y\) as same-day LFV to the pool (district mean pinned to Table 5 \(LFV_{pump}/LfA\)), versus stay \(K_z+K_g\) as deep / district storage. Spatial contrast follows canal proximity and Table 4 / Fig. 6 Lf intensity; ratios freeze across the wet↔dry year (`underground_ratio_holdout` / `ratio_holdout`). We do **not** claim measured subsurface paths, a unique \(K_x\) vs \(K_y\) axis, or novelty in IR timing.

We pose **one next-day water-fate predictor** on a closed cyclic-irrigation district, packaged as a Gymnasium loop that other algorithms can plug into. See the generated manuscript (`python -m cyclic paper`) for the full text; this file is a short pointer — regenerate for live numbers.

**Keywords:** inferred underground flow ratios; leftover lateral flow; cyclic irrigation; FAO-56 dual Kc; year holdout; Imbanuma

## Canonical API

```python
import cyclic
report = cyclic.infer_underground_ratios(2022)
hold = cyclic.underground_ratio_holdout()
print(cyclic.contribution_text(hold))
```


# Methods — inferred underground flow ratios (Okura, Kamei & Kato (2025), J. Hydrology: Regional Studies 57, 102111 -- Shiroyama-Jimbee cyclic irrigation area, Imbanuma, Chiba)

## Contribution

**Inferred underground flow ratios.** Surface IR/D (Fig. 8 / Okura pumps) are known forcing. After FAO-56 ET, leftover Lf (Okura Eq. 3) is partitioned per point into capture \(K_x+K_y\) (same-day LFV) and stay \(K_z+K_g\) (deep/dWB). District mean capture = Table 5 LFV_pump/LfA. Spatial structure: nearer the main drain → higher capture; Table 4 / Fig. 6 Lf intensity scales North vs South. Protocol: `infer_underground_ratios`, `underground_ratio_holdout`. Not claimed: measured subsurface paths; unique \(K_x\) vs \(K_y\); novel IR timing.

## Site and period

Shiroyama-Jimbee district, Imbanuma Land Improvement District, Chiba, Japan (Lake Kita Imbanuma). Cyclic irrigation season 17 April–25 August (131 days), years 2021 (wet) and 2022 (dry).

Observed irrigation and surface drainage are pump records and ADCPs in the main drainage canal (Okura §2). Daily IR/D from that campaign are the intended forecaster labels; the loader currently interpolates published tables until those series are ingested. Lateral flow Lf is **not** observed; it is the Eq. 3 residual. Simulated ET is FAO-56 dual Kc on local water; satellite Kc×ET0 is a loss term, not the flux (`force_satellite_et=False`). The main drainage canal stretches alongside Lake Kita Imbanuma, along the edge of the North and South blocks; drainage converges at the pumping-station pool; LFV_pump is same-day recapture of infiltrated leftover (collection ratio Kx+Ky, **per cyclic cell**, area-weighted mean = LFV_pump/LfA — nearer the main drain recaptures more; the main-drain polyline is not an LFV hop target), not hop-then-lining-catch. **Novelty is this underground map** (per-point inferred underground flow ratios Kx, Ky, Kz, ground under year freeze). Surface IR/W may be taken from digitised Fig. 8 as known forcing.

## Governing identities (Okura)

- **Okura 1** (District storage): `dWB = W - O + (R - ET) A_cyc / 1000 - IR_ncb`
- **Okura 3** (Block leftover (Lf), not measured): `L = IR + R - D - ET`
- **Okura 4** (Pumping-station pool): `Δpool = W - O + LFV_pump + D_N + D_S - IR_all`
- **Okura 6** (LFV from the pool residual): `LFV_pump = IR_all - W + O - D_N - D_S`
- **Okura 7** (Block storage after LFV leaves): `dWB = LfA_N + LfA_S - LFV_pump`
- **Okura 11–14** (Efficiencies): `WUE' = ET/(IR-D+ER);  RUE = D/IR_all;  LUE = LFV_pump/IR_all`

## Plot-scale model (this work)

FAO-56 dual crop coefficient (Allen et al. 1998) splits evaporation (pond or TEW surface) from transpiration (TAW root zone). Simulated ET is scored against the satellite Kc×ET0 map; that map is not imposed as the flux. One next-day water-fate predictor registers outside the package: `@register_forecaster` is the **farmer model** (per-paddy applied irrigation as a split of the Okura block total, plus surface discharge); leftover collection shares (Kx, Ky, Kz, ground) are coefficients of the per-paddy equation builder `@register_lateral` (`partition`, the **equation builder**, without soil properties); optional `@register_ratio` can override those shares. Linear and nonlinear are forms of the same H, not two hydrology models; physics baseline `head_equalise`. `CyclicEnv(given_irrigation=True)` is the gym. Leftover \(K_i\) are fitted on one year and frozen on the other (`ratio_holdout`). Per-paddy applied IR share is unknown; hops may see neighbour leftover head and neighbour FAO-56 ET on the up, down, left and right plots of every paddy (`neighbour_up_mm` … `neighbour_right_et_mm`), still equalise head rather than sending leftover toward high ET, and the same law is scored on historical 2021/2022 even if farmers irrigate differently (`soil_flow_holdout`). After IR, R, D and ET, leftover L is partitioned once (M1). Kz and ground are **stay** in the plot (dWB), not a second outlet. Kx+Ky is the **collection / recapture** share: same-day LFV from leftover that has already gone through the pan (pond leftover cannot be LFV). Unsaturated neighbours can take some of that recapture into TEW/TAW so those plots can still meet later FAO-56 ET; that intercept is a store capacity, not a leftover ratio. Only the sum is used, so the axis split is reported, not identified. A hop moves infiltrated remainder only toward a drier neighbour, at most until the two equalise; hops do not catch at the lining as LFV. Mean hops in a day are leftover speed over spacing (μ = vT/Δx), a kinematic count, not a fifth leftover ratio. `canal_hops` is not searched (Song et al. 2020: constrain drainage parameters or return-flow uncertainty explodes). Dupuit conductivity leftover is off because the district is flat (Tsubo et al. 2007). Hops equalise leftover head rather than following satellite ET, at bund-seepage scale (He et al. 2025: ~150 mm bund window; leftover speed 2 m d⁻¹). Simulated Ke uses Xu et al. (2017) wetting-pulse Kcp = 1.29 on rain or irrigation days. Pond→soil is `plow_pan_k_mm_d` × (pond / `pond_infiltrate_ref_mm`), capped at the pond — a Physics rate, not a leftover ratio, and not gated by TEW saturation. Perched Darcy/Hooghoudt runs only when `ratio_split` is off.

- **M1** (Leftover simplex on each plot): `L = (K_x + K_y + K_z + K_g) L,  with K_x+K_y+K_z+K_g = 1`
- **M2** (Stay vs recapture (collection ratio)): `stay = K_z + K_g,  recapture = K_x + K_y = LFV_pump / LfA  (Table 5/4)`
- **M3** (Leftover hops equalise neighbour head; not Darcy): `H sends infiltrated remainder only wetter→drier, at most equalising. Recapture does not hop.`
- **M4** (Same-day recapture from infiltrated leftover): `LFV = min((K_x+K_y) L^+, I); pond leftover cannot be LFV`
- **M5** (Optional QUBO (O'Malley)): `x_i=1 send leftover into the graph, x_i=0 stay; H = data misfit + J(x_i-x_j)^2`
- **M6** (Neighbour-head lateral equation (no soil K)): `One H per paddy: ΔL_i = H(h_i, h_up, h_down, h_left, h_right, ET of each); linear and nonlinear are forms, not two models`

## Closed-form shares from Tables 4–5 (2022)

- recapture Kx+Ky = LFV_pump / LfA = **0.7873** (same-day collection from infiltrated leftover)
- stay = **0.2127** (Kz = 0.0213, ground = 0.0851; split TEW/(TEW+TAW) = 25/125)
- farm-to-main = Table 5 D / Table 4 D = **1.0000**
- table start Kx = Ky = half of capture = **0.0532**

## Fitted shares on recapture routing (2022 season)

Loss 1.4429. File `best_2022_season.npz`.

- Kx = 0.4085, Ky = 0.4085 (reported only: routing sees Kx+Ky, so the axis split is not identified)
- Kz = 0.0366, ground = 0.1464
- farm-to-main = 1.0000
- capture = 0.8170

## Table self-consistency

Okura Tables 3–6 identities for 2022: all core checks pass. Table 7 North-block season WUE' is known not to close against Table 4 under the MAFF effective-rain rule (Okura code comments); monthly shape is retained, scaled to Table 4.

## Reproducibility

- Season length: 131 days; comparison year default 2022.
- Identification: IR/D from the best registered forecaster (frozen); leftover shares searched. Satellite ET scored, not imposed.
- Graph routing: `Physics.use_graph=True`, `one_way_to_canal=False` (bund seepage, not a canal diode).
- Optional QUBO: `Physics.as_qubo` or `python -m cyclic identify qubo-graph` (O'Malley clusters, not 4,925 paddies).
- Command: `python -m cyclic paper 2022` (METHODS.md and PAPER.md).

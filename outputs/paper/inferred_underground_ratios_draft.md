# Inferred underground leftover ratios under known surface forcing in a cyclic paddy district

**Working draft** — aligned to the `cyclic` implementation (pool capture → LFV; Kx+Ky → neighbours; Kz/ground → stay).  
**Not a claim of measured subsurface paths.** Surface IR/D/W are hard-copied from Okura et al. (2025); novelty is the per-point leftover partition and the protocols that freeze it.

**Suggested target:** *Journal of Hydrology: Regional Studies* (primary); *Agricultural Water Management* (alternate).  
**Suggested conference:** EGU General Assembly (HS) or PAWEES / ICID regional session (see §9).

---

## Abstract

In lowland cyclic irrigation, surface irrigation (IR), rainfall (R), surface drainage (D), and district leftover (Lf) are reported at block or pump scale, while the fate of leftover water beneath the paddy—same-day reuse (LFV), neighbour exchange, and on-plot stay—is not observed plot by plot. Building on the closed district water balance of Okura, Kamei and Kato (2025) for Shiroyama–Jinbee (Imbanuma, Chiba), we treat digitised surface IR/D (and related Table 4/5 volumes) as **known forcing**, not as something to rediscover. The contribution is an **inferred underground leftover simplex** at every cyclic point:

\[
L = \mathrm{IR}+R-D-\mathrm{ET},\qquad
L = L_{\mathrm{LFV}}+L_{\mathrm{nbr}}+L_{\mathrm{top}}+L_{\mathrm{deep}},
\]

with district-mean \(L_{\mathrm{LFV}}/L\) pinned to Table 5 \(LFV_{\mathrm{pump}}/LfA\), nearer-canal weighting of that share, neighbour hops covering local ET shortfalls, and the remaining stay split between evaporable topsoil (\(K_z\)) and deeper storage (ground / dWB-like). Joint training on 2021–2022 yields an area-weighted capture field with **mean relative fidelity ≈ 0.97** against each year’s Table 5 pin (\(|\Delta|\approx 0.0245\); mean squared error \(6\times10^{-4}\)). Year holdout of a frozen field against the other year’s pin shows the expected inter-annual gap (\(|\Delta|\approx 0.049\)).

Separately, we detail a **physics teacher** for leftover *hops*: bund seepage that equalises leftover head between neighbours under a Poisson mixture of hop counts. That teacher supplies labelled per-paddy-day \(\Delta L\) for optional equation discovery (linear / nonlinear / per-paddy trickle laws) and year holdout; a win over the teacher is **not** claimed. Recapture is not lining-catch; hops are not the LFV path.

**Keywords:** cyclic irrigation; leftover (Lf); LFV; underground flow ratios; neighbour hops; head equalisation; year holdout; Imbanuma; FAO-56 dual \(K_c\)

---

## 1. Introduction

### 1.1 Motivation

Cyclic irrigation returns farm drainage to a pumping-station pool and reuses it. Okura et al. (2025) closed the seasonal water balance for a flat lake-edge district using pumps, ADCPs, and published Tables 4–5. Their leftover \(L_f\) (Eq. 3 style: IR + R − D − ET) and the pumped reuse volume \(LFV_{\mathrm{pump}}\) define a district **capture ratio** \(LFV_{\mathrm{pump}}/LfA\), but they do not publish a map of how leftover is partitioned *at each paddy* into reuse, neighbour exchange, and stay.

That map matters for management: reuse efficiency, ET stress under uneven leftover, and whether plot ratios are a property of the field (holdout) or of the year. It also matters for modelling: calibrating lumped tanks to canal NSE can absorb pan-supply error into storage coefficients (cf. Iwasaki and Iida, 2024, on the same district). We refuse that absorption.

### 1.2 Contribution (claimed)

1. **Hard-copy surface forcing.** Digitized Fig. 8 / Okura IR+D (and Table 4 block volumes) enter leftover as known inputs. Surface IR *timing* is not the novelty.
2. **Per-point underground leftover simplex** with explicit roles:
   - `pool_capture_share` → same-day LFV (Table 5 pin; nearer-canal weighted);
   - \(K_x+K_y\) → leftover sent to **neighbours** (ET cover via hops);
   - \(K_z\) → stay in evaporable **topsoil** (TEW);
   - `ground_ratio` → deeper stay (TAW / dWB-like).
3. **Joint train / year holdout protocols** (`underground_ratio_train_both`, `underground_ratio_holdout`) reporting capture fidelity against Table 5.
4. **Physics teacher for leftover hops** (head equalise + Poisson \(\mu\)), used as optional labels for trickle-equation discovery—not as a claimed new law.

### 1.3 Not claimed

Measured subsurface pathlines; unique identification of \(K_x\) vs \(K_y\); new observation of district \(L_f\) or \(LFV_{\mathrm{pump}}\); that farmer IR schedules are novel; a holdout win of learned \(H\) over head equalise.

### 1.4 Relation to Okura et al. (2025)

Okura et al. provide the identities and labels. This work asks a *spatial inverse* question: given those identities and hard surface series, what per-point leftover shares are consistent with Table 5 capture and with neighbour-head physics? The natural venue companion is the same regional hydrology literature that published the field study.

---

## 2. Site identities and leftover residual

### 2.1 District

Shiroyama–Jinbee cyclic blocks (North / South), Imbanuma Second Phase area, Chiba, Japan. Seasons 2021 (wetter) and 2022 (drier) as in Okura et al. (2025). Cyclic paddies connect to the reuse pool; non-cyclic (NCB) leftover does not feed \(LFV_{\mathrm{pump}}\).

### 2.2 Surface water balance (known)

For plot \(i\) on day \(t\) (or season totals where only tables exist):

\[
L_{i,t} = IR_{i,t} + R_{t} - D_{i,t} - ET_{i,t}.
\]

- \(IR\), \(D\): hard-copied / ADCP–consistent forcing (Fig. 8 digitisation / Table 4).
- \(ET\): FAO-56 dual \(K_c\) on local water (Allen et al., 1998; wetting-day \(K_{cp}\) following Xu et al., 2017). Satellite \(K_c\times ET_0\) is a **score**, not a copied flux (`force_satellite_et = False`).
- Pond → soil **infiltration** (plow-pan conductivity) is a separate process from leftover ratios. Ratios partition *leftover*, not pond millimetres directly.

### 2.3 District pins (Tables 4–5)

\[
\rho_{\mathrm{LFV}}(y) = \frac{LFV_{\mathrm{pump}}(y)}{LfA(y)}.
\]

Closed-form construct for year \(y\):

\[
\begin{aligned}
\rho_{\mathrm{LFV}} &= \mathrm{clip}(LFV_{\mathrm{pump}}/LfA,\,0,1),\\
\rho_{\mathrm{nbr}} &= (1-\rho_{\mathrm{LFV}})\,\nu,\\
\rho_{\mathrm{stay}} &= (1-\rho_{\mathrm{LFV}})(1-\nu),\\
K_x = K_y &= \tfrac12\rho_{\mathrm{nbr}},\\
K_z &= \rho_{\mathrm{stay}}\frac{\mathrm{TEW}}{\mathrm{TEW}+\mathrm{TAW}},\\
\mathrm{ground} &= \rho_{\mathrm{stay}}\frac{\mathrm{TAW}}{\mathrm{TEW}+\mathrm{TAW}},
\end{aligned}
\]

with neighbour fraction \(\nu=\) `neighbour_of_remain` (default 0.5). Farm-to-main \(\rho_{sm}\) pins Table 5 main-canal D to Table 4 drainage.

**Numerical pins (this codebase):**

| Year | \(\rho_{\mathrm{LFV}}\) | \(K_x=K_y\) | \(K_z\) | ground |
|------|------------------------|-------------|---------|--------|
| 2021 | 0.7384 | 0.0654 | 0.0262 | 0.1046 |
| 2022 | 0.7873 | 0.0532 | 0.0213 | 0.0851 |
| Joint mean | 0.7629 | 0.0593 | — | — |

---

## 3. Methods I — Leftover simplex and spatial field

### 3.1 Roles (implementation)

| Share | Role in `apply_canal` |
|-------|------------------------|
| `pool_capture_share` | Intended same-day LFV; clipped by infiltrated leftover |
| \(K_x,K_y\) | Leftover mass that **hops to neighbours** (ET cover) |
| \(K_z\) | Bookkept stay that fills evaporable **surface** deficit |
| `ground_ratio` | Bookkept **deep** stay (`deep_mm`) |

Simplex: five non-negative shares summing to 1 on cyclic leftover. NCB: LFV and neighbour mass fold into ground stay (no pool feed).

### 3.2 Nearer-canal LFV field

Let \(d_i\) be plan distance to the main drain, \(\ell\) a canal length scale. Proximity \(p_i=e^{-d_i/\ell}\) (optionally × North/South Table 4 intensity). Soft-rescale so the **area-weighted cyclic mean** of capture equals \(\rho_{\mathrm{LFV}}\), with per-cell clip to \([0,1]\). Remainder \(1-\rho_i\) splits by \(\nu\) into neighbour vs stay; \(K_x=K_y\) each take half of the neighbour share (axis **not** identified).

### 3.3 Joint train and holdout

- **`underground_ratio_train_both`:** one field with district capture = mean of 2021 and 2022 pins; score squared error to each year’s pin.
- **`underground_ratio_holdout`:** fit/freeze on train year; score the frozen mean capture against the eval year’s Table 5 pin (no claim that years share one capture).

Relative fidelity:

\[
F = 1 - \frac{|\hat\rho-\rho_{\mathrm{pin}}|}{\rho_{\mathrm{pin}}}.
\]

### 3.4 What \(K_z\) is not

\(K_z\) is **not** “pond water infiltrating into topsoil.” Pond infiltration is plow-pan flux into TEW/TAW/perched stores. \(K_z\) is the share of *leftover residual* attributed to shallow stay once \(L\) has already been formed.

---

## 4. Methods II — Physics teacher for leftover hops (detailed)

This section is the mechanism behind the “leftover-hop RMSE / holdout scatter” figures. It produces **net sideways leftover transfer** \(\Delta_i\) (mm per paddy-day). It does **not** use the Table 5 simplex to invent those millimetres; the simplex only decides *how much* leftover is eligible for neighbour hops vs LFV vs stay. The teacher moves the neighbour-eligible (or full leftover, in the hop-teacher cache) mass on a graph.

### 4.1 Graph

Nodes = paddies (or satellite cells). Undirected edges = neighbour contacts (cardinal UDLR or \(k\)-NN). Default permeability is **two-way**: Darcy \(q=-K\nabla h\) is not a diode. Optional `one_way_to_canal` keeps only edges that decrease canal distance (not the default LFV path).

### 4.2 Leftover as head

On day \(t\), treat leftover depth \(L_i\) (mm) as the head for hopping. Higher \(L\) = wetter for this operator. This is a modelling choice: Okura does not publish plot ponding for Playán-style residuals; leftover is the available residual pile.

### 4.3 One directional move (`move_stock`)

For edges oriented along an axis (E–W or N–S):

1. Compare leftover on the two ends.
2. **Only if** the source is wetter than the sink (\(L_{\mathrm{high}}-L_{\mathrm{low}}>0\)) may water move.
3. A stage factor (farm-drain stage / reference, clipped by `canal_max_frac`) limits the fraction of the wetter pile that can leave.
4. Volume conservation: mass leaving the high plot equals mass arriving on the low plot (areas convert mm ↔ m³).
5. Cap: at most enough to **equalise** heads after the move (no overshoot that would invert the gradient).

Thus one move is **head-equalising bund seepage**, not “flow toward high ET,” and not catch at the lake-edge lining as LFV.

### 4.4 One hop (`one_hop`)

One hop = E–W equalisation pass, then N–S pass, clipping negative leftover to zero. Multiple hops allow leftover to travel farther along the graph within the day.

### 4.5 Poisson mixture for one day (`hop_stock`)

Mean hop count:

\[
\mu = \begin{cases}
\texttt{canal\_hops} & \text{if set},\\
v_L / \Delta x & \text{otherwise},
\end{cases}
\]

with leftover speed \(v_L\) (default 2 m d⁻¹) and \(\Delta x\) = median edge length. Hop-count weights:

\[
P(k)=e^{-\mu}\frac{\mu^k}{k!}\quad(k=0,1,2,\ldots),
\]

truncated and renormalised. Let \(H\) be the one-hop operator and \(L^{(0)}\) the morning leftover. The day’s mixed leftover is

\[
\tilde L = \sum_{k\ge 0} P(k)\, H^{k}(L^{(0)}).
\]

Teacher label:

\[
\Delta = \tilde L - L^{(0)}.
\]

- \(\Delta_i>0\): plot \(i\) **received** leftover from neighbours that day.  
- \(\Delta_i<0\): plot \(i\) **sent** leftover away.  
- Typical magnitude in holdout scatters: order **1 mm d⁻¹** (not hundreds), except when a bad nonlinear law explodes on transfer.

Mass is conserved over the cyclic graph (aside from numerical clip safeguards).

### 4.6 Why a “teacher,” not a “truth”

Plot-scale leftover hops are **unobserved** in Okura. The physics teacher is a **mechanistic label generator** consistent with:

- He et al. (2025): bund seepage concentrated in shallow layers, driven by irrigation–drainage regime / ponding, not canopy demand;
- flat lake-edge geometry (Dupuit slope conductivity as a leftover *share* is forbidden);
- Beven-style limits: hops may equalise head but must not become a free path that delivers LFV to fit Table 5.

Learned trickle laws (\(H_{\mathrm{lin}}\), \(H_{\mathrm{nonlin}}\), per-paddy) regress \(\Delta\) from neighbour head drops and neighbour FAO-56 ET, freeze on year \(y_1\), score RMSE on \(y_2\). **A holdout win is not claimed** in this draft; the nonlinear form can fail catastrophically on transfer (large holdout RMSE), which is itself informative about hypothesis class.

### 4.7 Coupling to the simplex

In the live simulator with `ratio_split=True`:

- LFV uses `pool_capture_share` of leftover (from infiltrated water);
- **neighbour-eligible** mass \(\propto(K_x+K_y)L\) is what `hop_stock` redistributes;
- stay \((K_z+\mathrm{ground})L\) does not hop.

So the physics teacher answers: *given a leftover map, how does neighbour-eligible water rearrange?* The simplex answers: *how much leftover is neighbour-eligible vs LFV vs stay?*

---

## 5. Results (current package numbers)

### 5.1 Joint-train capture fidelity

| Quantity | Value |
|----------|-------|
| Joint area-weighted capture | 0.7629 |
| 2021 Table 5 pin | 0.7384 → \(F=0.9668\), \(\|\Delta\|=0.0245\) |
| 2022 Table 5 pin | 0.7873 → \(F=0.9689\), \(\|\Delta\|=0.0245\) |
| Mean relative fidelity | **≈ 0.97** |
| Mean squared capture error | \(6.0\times10^{-4}\) |

Interpretation: equation fidelity of the **leftover-ratio / LFV pin**, not farmer IR/D volume fidelity and not ET map fidelity.

### 5.2 Year holdout (frozen field vs other pin)

| Train → eval | \(\|\hat\rho_{\mathrm{train}}-\rho_{\mathrm{eval}}\|\) | error² |
|--------------|------------------------------------------------------|--------|
| 2021 → 2022 | ≈ 0.049 | ≈ 0.0024 |
| 2022 → 2021 | ≈ 0.049 | ≈ 0.0024 |

The gap tracks the wet/dry year difference in Table 5 capture, not a failure of nearer-canal geometry within a year.

### 5.3 Hop equation figures (qualitative)

- Holdout scatter: predicted vs cached \(\Delta\) on sampled paddy-days; points near the origin at ±1–3 mm scale for linear forms.
- RMSE bars: nonlinear can show very large 2022 holdout RMSE relative to linear / head equalise—evidence against transferring that hypothesis class, not evidence against the teacher itself.

---

## 6. Discussion

### 6.1 Identifiability

Following Prieto et al. (2021, 2022): routing-like coordinates (LFV capture, farm-to-main) are constrained by pool/canal aggregates; the percolation split \(K_z\) vs ground is **not** identified—only their sum (stay) enters the claim. \(K_x\) vs \(K_y\) is an equivalence class under district aggregation.

### 6.2 Separation from farmer skill

Forecasters propose IR/D. Underground ratios and hop laws are scored under frozen or hard-copied IR/D so hydrology is not rewarded for leaking the IR interpolation rule (Kapoor and Narayanan, 2023).

### 6.3 Limits of acceptability

If seasonal infiltrated leftover cannot supply Table 5 LFV, that is a **pan-supply / structure** miss (Beven), not a signal to retune `plow_pan_k` against LFV.

---

## 7. Conclusions

We infer per-point underground leftover ratios under hard-copied surface IR/D for a cyclic district, with LFV pinned to Table 5, neighbours receiving a dedicated leftover share for ET cover, and stay split shallow/deep by TEW/TAW. Joint-train capture fidelity is **≈ 0.97**. A physics teacher—Poisson mixtures of head-equalising hops—labels leftover neighbour exchange for optional equation discovery without claiming a new holdout-winning law.

---

## 8. Data and code

- Field labels: Okura et al. (2025), Tables 4–5; Fig. 8 digitisation in the `cyclic` package.
- Reproduction: `python -m cyclic` underground / paper targets; `underground_ratio_train_both()`; tests in `tests/test_identify.py`.

---

## 9. Where to target (recommendation)

### Primary journal — **Journal of Hydrology: Regional Studies**

**Why:** Okura et al. (2025) already published the Imbanuma cyclic water balance there (*J. Hydrol. Reg. Stud.*, DOI [10.1016/j.ejrh.2024.102111](https://doi.org/10.1016/j.ejrh.2024.102111)). A methods paper that (i) keeps their identities, (ii) adds per-point leftover ratios under hard surface forcing, and (iii) documents hop physics + holdout protocol is a natural **regional follow-on**. Editors and reviewers will already know the district, the ADCP/pump setting, and the leftover vocabulary (Lf, LFV).

**Fit:** regional hydrologic insight + reproducible spatial inference; not a claim of a universal conductivity.

### Strong alternate — **Agricultural Water Management**

**Why:** reuse efficiency, drainage water recycling, and management implications of leftover partition are core AWM topics (recent DWRS / return-flow papers). Better if you emphasise **management utility** of the simplex (who needs neighbour cover vs LFV) over regional process detail.

### If you lead with equation discovery / ML holdout

- **Environmental Modelling & Software** — Gym-style env, registered learners, year holdout protocol.
- Avoid overselling QRC/QAOA unless those beats are real (current package does not claim a quantum win).

### Conferences (talk / extended abstract first)

| Venue | Angle |
|-------|--------|
| **EGU General Assembly** (Hydrological Sciences) | Physics teacher + identifiability + holdout |
| **AGU Fall Meeting** (H13 / irrigation sessions) | Broader visibility; poster or talk |
| **PAWEES** or **ICID** congress / regional | Paddy / irrigation engineering audience closest to Okura |
| **ASABE Annual International Meeting** | If framing as decision-support for cyclic systems |

**Practical path:** EGU or PAWEES abstract → full paper to *J. Hydrol. Reg. Stud.* (primary) with AWM as backup if the regional editor wants more management framing.

### Positioning one-liner for the cover letter

> We do not re-estimate Okura’s surface IR/D. We infer the unobserved per-point leftover partition (LFV vs neighbour hops vs stay) under that forcing, with ≈0.97 fidelity to Table 5 capture under joint training, and we document a head-equalising physics teacher for leftover hops used in year-holdout equation protocols.

---

## References (seed list; expand in final)

Allen, R.G., et al., 1998. *Crop evapotranspiration.* FAO Irrigation and Drainage Paper 56.  
Beven, K., 2006. A manifesto for the equifinality thesis. *J. Hydrol.*  
He, Y., et al., 2025. Nitrogen lateral seepage through paddy bunds…  
Iwasaki, K., Iida, T., 2024. Tank-model recalibration, Imbanuma canals.  
Okura, D., Kamei, T., Kato, T., 2025. Water balance analysis… cyclic irrigation, Imbanuma. *J. Hydrol. Reg. Stud.* 102111.  
Prieto, C., et al., 2021/2022. Identifiability of hydrological processes from streamflow.  
Xu, J., et al., 2017. Dual crop coefficients for drying–wetting rice.  
Kapoor, S., Narayanan, A., 2023. Leakage and the reproducibility crisis…  
Playán, E., et al., 2008. Inter-paddy discharge from water-balance residuals.  
Tsubo, M., et al., 2007. Dupuit seepage on a toposequence.  
Song, J., et al., 2020. Uncertainty in irrigation return-flow estimates.  

---

## Appendix A — Glossary

| Term | Meaning |
|------|---------|
| Leftover \(L\) | IR+R−D−ET residual (Okura Lf family) |
| LFV | Same-day reuse volume counted at the pool |
| Physics teacher | Head-equalise + Poisson hops → \(\Delta\) labels |
| Simplex | Shares of \(L\) summing to 1 |
| Hop \(\Delta\) | Net leftover mm gained/lost from neighbours that day |
| Equation fidelity (~0.97) | Relative match of capture field to Table 5 pin |

## Appendix B — Implementation map

| Paper object | Code |
|--------------|------|
| Construct pins | `cyclic.identify.construct_ratios` |
| Spatial field | `cyclic.canal.pixel_ratio_fields` |
| Routing | `cyclic.canal.apply_canal`, `darcy_shares` |
| Joint / holdout | `cyclic.underground.underground_ratio_train_both`, `…_holdout` |
| Hop teacher | `cyclic.graph.hop_stock`, `one_hop`, `move_stock` |
| Equation holdout | `cyclic.identify.equation_from_fields`, `equation_figure` |

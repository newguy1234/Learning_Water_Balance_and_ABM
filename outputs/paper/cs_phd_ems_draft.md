# Learning per-plot water fate from district aggregates under cyclic irrigation: a Gymnasium benchmark with frozen equation holdout

**Working draft — CS / environmental modelling framing**  
**Primary venue:** *Environmental Modelling & Software* (Elsevier)  
**Alternate:** *Computers and Electronics in Agriculture*; conference first: EGU ITS / AGU (ML for Earth) or AAAI / IJCAI AI for Social Impact track (if the gym + leakage story is foregrounded)

*Author role (assumed):* CS PhD student. Hydrology (Okura et al., 2025) supplies the **benchmark identities and labels**; the contribution is the **learning problem, constraints, environment API, and evaluation protocol**.

---

## Abstract

Many scientific ML settings supervise models with **coarse aggregates** while decisions or diagnoses are needed at **fine spatial resolution**. We package one such inverse problem as a **Gymnasium environment** for a real cyclic-irrigation district (Shiroyama–Jinbee, Imbanuma), where surface irrigation and drainage are treated as **known forcing** (digitised Okura pump/ADCP series / Tables 4–5) and the learning target is an **unobserved per-plot leftover partition**.

The environment exposes three plug-in interfaces: (i) `@register_forecaster` — plot-level IR/D policies; (ii) `@register_ratio` — leftover simplex shares; (iii) `@register_lateral` — neighbour-head leftover-hop equations. Supervision for underground shares is a **district identity** (Table 5 \(LFV_{\mathrm{pump}}/LfA\)), not plot labels. We enforce **anti-shortcut constraints** (no copying satellite ET into the flux; hops may equalise leftover head but must not deliver LFV; no absorbing Table 5 error into tank-style storage coefficients). Evaluation is **year holdout with frozen parameters** (2021↔2022), not in-sample capture.

Under these constraints, a closed-form Table-pinned leftover simplex (LFV vs neighbour hops vs topsoil/deep stay), joint-trained on both seasons, matches each year’s capture pin at **mean relative fidelity ≈ 0.97**. A **physics teacher** (Poisson mixtures of head-equalising hops) generates plot-day hop labels for optional equation discovery; we do **not** claim a learned hop law beats that teacher on holdout. The artefact is the **benchmark + protocol**, not a new field campaign and not a claimed SOTA hydrology model.

**Keywords:** Gymnasium; learning from aggregates; scientific machine learning; year holdout; equation discovery; identifiability; shortcut learning; irrigation return flow; environmental modelling

---

## 1. Why this is a CS problem

### 1.1 The generic template

| Ingredient | Here |
|------------|------|
| Fine-scale object | Per-plot leftover fate shares / hop \(\Delta\) |
| Coarse supervision | District Table 5 capture; block IR/D |
| Simulator / aggregator | FAO-56 + routing that maps shares → district totals |
| Failure mode | Shortcuts that fit aggregates by cheating physics |
| Protocol | Freeze on year \(y_1\), score year \(y_2\) |

This matches **learning from label proportions / aggregate observations** (Quadrianto et al., 2009; Zhang et al., 2020), with a **non-mean aggregator** (a season-long water-balance simulator) and an equivalence class of solutions (axis \(K_x\) vs \(K_y\) invisible to the aggregator).

### 1.2 What we refuse to call novelty

- Closing the Imbanuma water balance (Okura et al., 2025).
- FAO-56 dual \(K_c\) (Allen et al., 1998; Xu et al., 2017).
- Gym APIs for crop management in general (CropGym, gym-DSSAT, CyclesGym, …).
- A claimed win of QAOA/QRC over classical search.

### 1.3 Claimed CS contribution

1. **Problem formulation:** per-plot leftover fate under **hard known surface forcing**, supervised by **district capture identity**.
2. **Environment + registration API** so external algorithms plug in without editing the simulator.
3. **Constraint set** that blocks known shortcuts (ET copy; hops-as-LFV; tank absorption).
4. **Evaluation protocol:** joint vs holdout fidelity; frozen hop-equation RMSE against a physics teacher.
5. **Negative result space:** nonlinear hop hypotheses can fail transfer; \(K_z\) vs ground and \(K_x\) vs \(K_y\) are not identified — reported as such (Prieto-style ranking).

**PhD thesis sentence:** *I build and validate a constrained inverse-learning benchmark for spatial water-fate inference from district aggregates, with year-holdout protocols and an explicit physics teacher for equation discovery.*

---

## 2. Related work (CS lens)

**Crop / irrigation RL environments.** Existing envs optimise yield and applied water at field scale with observed rewards (Kallenberg et al., 2023; Gautron et al., 2022; Turchetta et al., 2022; Chen et al., 2021). They rarely (i) take district IR as *known forcing* and ask for unobserved leftover fate, or (ii) score against a published district identity under anti-cheat routing.

**Learning from aggregates.** Bag-level labels and label proportions are studied in ML; our bag operator is a physics simulator, and the fine label is a residual of other predicted quantities — closer to scientific inverse problems than to standard LLP.

**Shortcut learning and leakage.** Kapoor and Narayanan (2023); Geirhos et al. (2020). Our proxy IR/D interpolations can leak if discharge is ranked on them; the protocol therefore hard-copies or freezes surface series when scoring underground ratios.

**Identifiability.** Beven equifinality; Prieto et al. (2021, 2022) ranking of routing vs percolation identifyability — used as a **design principle** for which simplex coordinates we claim.

**Equation discovery / symbolic regression adjacent.** We fit parametric linear/nonlinear neighbour-head maps and hold them out; we do not claim AI-Feynman-style symbolic recovery.

---

## 3. Benchmark domain (not the CS claim)

**Site.** Cyclic paddy district, Imbanuma, Chiba (Okura et al., 2025).  
**Forcing.** Surface IR/D hard-copied (Fig. 8 / tables).  
**Residual.** \(L = \mathrm{IR}+R-D-\mathrm{ET}\).  
**District pin.** \(\rho = LFV_{\mathrm{pump}}/LfA\) (year-dependent).

The domain is to this benchmark what Atari is to RL papers: **necessary context**, not the contribution title.

---

## 4. Environment API

### 4.1 Loop

`CyclicEnv` / `CyclicModel`: daily `reset` / `step`. Observations include weather, stocks, optional neighbour features. Actions (when farming): irrigation and discharge mm.

### 4.2 Three registered learners

```text
@register_forecaster  →  (IR, D) at plots     # surface policy
@register_ratio       →  leftover simplex     # underground shares
@register_lateral     →  hop law H            # neighbour ΔL
```

External code (e.g. `example.py`) registers agents **outside** the core package — the CS artefact is the **stable interface + scoring**, not a single handcrafted farmer.

### 4.3 Leftover simplex (structured output)

Per cyclic plot, shares sum to 1:

| Coordinate | Semantic role in the env |
|------------|--------------------------|
| `pool_capture_share` | Same-day LFV (district mean pinned) |
| \(K_x,K_y\) | Neighbour-eligible leftover (hops) |
| \(K_z\) | Topsoil stay (TEW bookkeeping) |
| `ground_ratio` | Deep stay |

**Design choice (CS):** output is a **probability simplex** (structured prediction), not unconstrained mm/d conductivities. Travel distance is a separate kinematic parameter \(\mu\), not a fifth share — so search cannot trade “fake conductivity” against capture.

### 4.4 Anti-shortcut constraints (first-class)

1. `force_satellite_et = False` — satellite ET is a **loss target**, not the flux.  
2. LFV only from **infiltrated** leftover × capture share — pond leftover cannot be LFV.  
3. Hops **equalise head** (wet→dry); they do not catch at the lining as LFV.  
4. Pan-supply miss vs Table 5 is reported as **structure**, not folded into `plow_pan_k`.

These are to the benchmark what action masks / physics constraints are in safe RL: they define the legal hypothesis class.

---

## 5. Physics teacher for hop labels (for equation discovery)

Plot-scale leftover hops are unobserved. For `@register_lateral`, we need a **teacher** \(\Delta^\star\).

### 5.1 Algorithm (head equalise + Poisson hops)

**Input:** leftover map \(L\), neighbour graph, areas, speed \(v_L\) (or fixed `canal_hops`).  
**One hop \(H\):** along E–W then N–S edges, move mass only wet→dry, at most equalising heads, volume-conserving.  
**Day mixture:** \(\mu = v_L/\Delta x\), \(P(k)=e^{-\mu}\mu^k/k!\), \(\tilde L=\sum_k P(k) H^k(L)\).  
**Label:** \(\Delta^\star = \tilde L - L\).

### 5.2 Learning problem

Fit \(H_\theta\) (linear / nonlinear / per-paddy) so \(H_\theta(L, \text{neighbour ET}, \ldots)\approx\Delta^\star\) on year \(y_1\); freeze \(\theta\); score RMSE on year \(y_2\).

**Important for a CS thesis:** the teacher is part of the **benchmark definition** (like a simulator reward). Beating it is an open challenge; this draft does **not** claim a beat. Reporting that nonlinear fails transfer is a **valid negative result**.

---

## 6. Protocols and metrics

### 6.1 Ratio fidelity (aggregate→field)

\[
F = 1 - \frac{|\hat\rho - \rho_{\mathrm{pin}}|}{\rho_{\mathrm{pin}}}
\]

on area-weighted cyclic capture vs Table 5 pin.

**Joint train (both years):** mean \(F \approx \mathbf{0.97}\) (\(|\Delta|\approx 0.0245\), MSE \(\approx 6\times10^{-4}\)).  
**Holdout year pin miss:** \(|\Delta|\approx 0.049\) (inter-annual pin shift).

Interpretation for CS reviewers: this is **constraint-satisfying reconstruction fidelity**, not “accuracy of a black-box regressor on i.i.d. test plots.”

### 6.2 Hop equation holdout

RMSE of frozen \(H_\theta\) vs teacher \(\Delta^\star\) on held-out year; scatter of predicted vs cached \(\Delta\) (typical scale ~1 mm d⁻¹).

### 6.3 What not to put in the teaser figure

Do not lead with “we discovered Imbanuma hydrology.” Lead with: **env diagram → aggregate label → constrained simplex → holdout arrows.**

---

## 7. Identifiability as a first-class result

| Quantity | Status |
|----------|--------|
| Capture sum / `pool_capture` | Constrained by Table 5 |
| \(K_x\) vs \(K_y\) | **Not identified** (report half–half) |
| \(K_z\) vs ground | **Not identified** (TEW/TAW split only) |
| Hop law beyond teacher | Open; no win claimed |

For a CS PhD: **stating the equivalence class** is a contribution (Zhang et al. consistency up to equivalence), not a weakness to hide.

---

## 8. Threats to validity (write these before reviewers do)

1. **Teacher ≠ truth.** Hop \(\Delta^\star\) is mechanistic, not measured.  
2. **Label leakage** if IR/D proxies are interpolated from the same weather the agent sees — underground scoring must freeze/hard-copy surface.  
3. **Two seasons only** — holdout is real but thin; protocol generalises, \(N_{\mathrm{year}}=2\) does not.  
4. **Single district** — benchmark is regional; claim is the **problem class**, not universal coefficients.

---

## 9. Positioning vs hydrology journals

| Venue | Lead claim |
|-------|------------|
| *J. Hydrol. Reg. Stud.* | Regional leftover ratios on Okura’s site |
| ***Environmental Modelling & Software*** | **Env + constrained inverse learning + holdout protocol** |
| NeurIPS/ICML main | Only with a new algorithm that **wins** a clear metric |

As a CS PhD student, prefer **EM&S** (or *Computers and Electronics in Agriculture* if the editor wants agri-AI). Use EGU/AGU for community feedback; use a hydrology journal only for a **second** paper that deepens the regional numbers without the gym framing.

---

## 10. Suggested title / cover letter (CS)

**Title options**
1. *A Gymnasium Benchmark for Learning Per-Plot Water Fate from District Aggregates*  
2. *Constrained Inverse Learning of Leftover Partition under Cyclic Irrigation*  
3. *From District Labels to Plot Shares: Holdout Protocols for Irrigation Return-Flow Equations*

**Cover letter (short)**  
> We submit a reusable Gymnasium benchmark and evaluation protocol for a scientific inverse problem: inferring per-plot leftover water-fate shares from district aggregate identities under known surface forcing. The contribution is the problem formulation, anti-shortcut constraints, registration API, and year-holdout metrics—not a new field campaign. A Table-pinned reconstruction reaches ≈0.97 relative fidelity to the published capture identity under joint training; learned hop equations are evaluated against an explicit physics teacher without claiming a holdout win.

---

## 11. Thesis chapter map (optional)

1. Related work: envs, LLP, shortcuts, identifiability  
2. Formal problem + constraints  
3. Benchmark implementation  
4. Baseline ratio construct + fidelity  
5. Physics teacher + equation discovery experiments  
6. Negative results and open challenges  
7. Broader impact: other aggregate→fine environmental inverses

---

## 12. Seed references (CS-heavy)

Gautron et al., 2022 — crop management RL survey.  
Kallenberg et al., 2023 — CropGym.  
Turchetta et al., 2022 — CyclesGym.  
Chen et al., 2021 — rice irrigation DQN.  
Quadrianto et al., 2009 — learning from label proportions.  
Zhang et al., 2020 — learning from aggregate observations.  
Geirhos et al., 2020 — shortcut learning.  
Kapoor & Narayanan, 2023 — leakage.  
Beven, 2006 — equifinality.  
Prieto et al., 2021/2022 — process identifiability.  
Okura et al., 2025 — domain labels (*J. Hydrol. Reg. Stud.*).  
Allen et al., 1998 — FAO-56.  

---

## Appendix — Implementation pointers

| Concept | Code |
|---------|------|
| Env / model | `cyclic.env`, `cyclic.model` |
| Registration | `cyclic.agents` (`@register_*`) |
| Simplex construct | `cyclic.identify.construct_ratios` |
| Joint / holdout ratios | `cyclic.underground` |
| Hop teacher | `cyclic.graph.hop_stock` |
| Equation holdout + figure | `cyclic.identify.equation_from_fields`, `equation_figure` |

---

## Bottom line for you (CS PhD)

Your discovery, in CS terms, is not “mm went that way.” It is:

> **A constrained, year-holdout benchmark for learning fine-scale water-fate structure from coarse hydrologic identities, with a physics teacher for equation discovery and explicit non-identifiability.**

Head to ***Environmental Modelling & Software***. Keep the ~0.97 number as a **baseline fidelity under constraints**, and keep the hop scatter as **protocol illustration / open challenge**, not as the abstract’s hero claim.

"""Working manuscript generated from the cyclic package.

Numbers come from cyclic.okura, cyclic.identify, cyclic.physics and any saved
ratio / QRC files. Run:

    python -m cyclic paper 2022

That writes outputs/paper/manuscript.md and PAPER.md. tests/test_paper_claims.py
recomputes the same numbers and fails if the prose disagrees with the code.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

from . import okura
from .identify import (DEFAULT_WEIGHTS, LFV_ACCEPT_FRAC, OBJECTIVES, PERCOLATION_COORDINATES,
                       PROCESS_ROLE, REPORTED_ONLY, ROUTING_COORDINATES, construct_ratios,
                       load_best_ratios)
from .physics import KC_INI, KC_MID, NDVI_BARE, NDVI_FULL, Physics, canopy_cover, kc_from_ndvi


def live_numbers(year=okura.COMPARISON_YEAR):
    """Every quantity the manuscript quotes. Tests compare this dict to the code."""
    p = Physics()
    shares = construct_ratios(year)
    fitted = load_best_ratios(year, "season")
    tew, taw = p.total_evaporable_mm, p.total_available_mm
    topsoil = tew / (tew + taw)
    t5 = okura.TABLE5_1000M3[year]
    lfa = (t5["LfA_North"] + t5["LfA_South"]) * 1000.0
    lfv = t5["LFV_pump"] * 1000.0
    capture = lfv / lfa
    area = {block: okura.cyclic_area_m2(block) for block in okura.BLOCKS}
    from .identify import table_main_d_m3

    d4 = sum(okura.TABLE4_MM[(block, year)]["drainage"] * area[block] for block in okura.BLOCKS) / 1000.0
    d5 = table_main_d_m3(year)
    farm_raw = d5 / d4
    from .env import DEFAULT_WEIGHTS as ENV_WEIGHTS, GYM_WEIGHTS

    qrc_txt = _read_if_exists(os.path.join(_outputs(), "ratios", f"qrc_{year}_equation.txt"))
    qrc_spatial_txt = _read_if_exists(os.path.join(_outputs(), "ratios", f"qrc_spatial_{year}_equation.txt"))
    routing_txt = _read_if_exists(os.path.join(_outputs(), "ratios", f"routing_{year}.txt"))
    return SimpleNamespace(
        year=int(year),
        season_days=int(okura.SEASON_DAYS),
        comparison_year=int(okura.COMPARISON_YEAR),
        force_satellite_et=bool(p.force_satellite_et),
        ratio_split_default=bool(p.ratio_split),
        use_graph=bool(p.use_graph),
        one_way_to_canal=bool(p.one_way_to_canal),
        leftover_speed_m_d=float(p.leftover_speed_m_d),
        bund_seepage_mm=float(p.bund_seepage_mm),
        plow_pan_k_mm_d=float(p.plow_pan_k_mm_d),
        pond_infiltrate_ref_mm=float(p.pond_infiltrate_ref_mm),
        canal_hops=p.canal_hops,
        canal_catch_power=float(p.canal_catch_power),
        canal_length_m=float(p.canal_length_m),
        canal_max_frac=float(p.canal_max_frac),
        kc_max=float(p.kc_max),
        kc_wet=float(p.kc_wet),
        kcb_bare=float(p.kcb_bare),
        kcb_full=float(p.kcb_full),
        rew=float(p.readily_evaporable_mm),
        tew=float(tew),
        raw=float(p.readily_available_mm),
        taw=float(taw),
        rice_p=float(p.readily_available_mm / p.total_available_mm),
        topsoil_frac=float(topsoil),
        kc_ini=float(KC_INI),
        kc_mid=float(KC_MID),
        ndvi_bare=float(NDVI_BARE),
        ndvi_full=float(NDVI_FULL),
        capture=float(capture),
        stay=float(1.0 - capture),
        farm_to_main=float(shares["farm_to_main"]),
        farm_to_main_raw=float(farm_raw),
        kx=float(shares["kx_ratio"]),
        ky=float(shares["ky_ratio"]),
        kz=float(shares["kz_ratio"]),
        ground=float(shares["ground_ratio"]),
        simplex=float(shares["kx_ratio"] + shares["ky_ratio"] + shares["kz_ratio"] + shares["ground_ratio"]),
        shares_force_et=bool(shares["force_satellite_et"]),
        shares_ratio_split=bool(shares["ratio_split"]),
        lfv_1000m3=float(t5["LFV_pump"]),
        lfa_1000m3=float(t5["LfA_North"] + t5["LfA_South"]),
        north_ha=okura.TABLE1_HA["North"],
        south_ha=okura.TABLE1_HA["South"],
        table4=okura.TABLE4_MM,
        table5=t5,
        table3=okura.TABLE3_1000M3[year],
        table6=okura.TABLE6[year],
        rain_2021=okura.TABLE4_MM[("North", 2021)]["rain"],
        rain_2022=okura.TABLE4_MM[("North", 2022)]["rain"],
        capture_2021=float(construct_ratios(2021)["pool_capture_share"]),
        capture_2022=float(construct_ratios(2022)["pool_capture_share"]),
        identify_weights=dict(DEFAULT_WEIGHTS),
        process_role=dict(PROCESS_ROLE),
        routing_coordinates=tuple(ROUTING_COORDINATES),
        percolation_coordinates=tuple(PERCOLATION_COORDINATES),
        lfv_accept_frac=float(LFV_ACCEPT_FRAC),
        env_weights=dict(ENV_WEIGHTS),
        gym_weights=dict(GYM_WEIGHTS),
        objectives=tuple(OBJECTIVES),
        fitted=None if fitted is None else {
            "kx": float(fitted.params["kx_ratio"]),
            "ky": float(fitted.params["ky_ratio"]),
            "kz": float(fitted.params["kz_ratio"]),
            "ground": float(fitted.params["ground_ratio"]),
            "farm_to_main": float(fitted.params["farm_to_main"]),
            "capture": float(fitted.params.get(
                "pool_capture_share",
                fitted.params["kx_ratio"] + fitted.params["ky_ratio"])),
            "loss": fitted.loss,
            "path": os.path.basename(fitted.path),
            "force_et": bool(fitted.params.get("force_satellite_et", False)),
        },
        qrc_txt=qrc_txt,
        qrc_spatial_txt=qrc_spatial_txt,
        routing_txt=routing_txt,
        reported_only=tuple(REPORTED_ONLY),
        fc_at_bare=float(canopy_cover(0.15)),
        fc_at_full=float(canopy_cover(0.80)),
        kc_at_bare=float(kc_from_ndvi(NDVI_BARE)),
        kc_at_full=float(kc_from_ndvi(NDVI_FULL)),
    )


def _outputs():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def _read_if_exists(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return handle.read().strip()


def _fmt(value, digits=4):
    return f"{float(value):.{digits}f}"


def _weights(mapping):
    return ", ".join(f"{key}={float(val):g}" for key, val in mapping.items())


def manuscript_markdown(year=okura.COMPARISON_YEAR):
    n = live_numbers(year)
    t4n = n.table4[("North", year)]
    t4s = n.table4[("South", year)]
    fitted_block = _fitted_section(n)
    qrc_block = _qrc_section(n)
    return "\n".join([
        "# Inferred underground flow ratios under cyclic irrigation",
        "",
        "*Working manuscript generated from the `cyclic` package. Every number in the "
        "Results section is either written by `cyclic.manuscript.live_numbers` or quoted "
        "verbatim from a file the package generated (§7). Re-run "
        "`python -m cyclic paper 2022` after a code change. "
        "`tests/test_paper_claims.py` fails if the text and the code disagree.*",
        "",
        "## Abstract",
        "",
        "**Contribution: inferred underground flow ratios.** With today's pool "
        "irrigation and surface drainage treated as known forcing (Okura, Kamei and "
        "Kato, 2025; digitised Fig. 8 IR/W), we infer how leftover after FAO-56 ET is "
        "partitioned *underground* at each point: capture \(K_x+K_y\) as same-day LFV "
        "to the pool (district mean pinned to Table 5 "
        f"\(LFV_{{pump}}/LfA\) = {_fmt(n.capture)} in {n.year}), versus stay "
        "\(K_z+K_g\) as deep / district storage. Spatial contrast follows canal "
        "proximity and Table 4 / Fig. 6 Lf intensity; ratios freeze across the wet↔dry "
        "year (`underground_ratio_holdout` / `ratio_holdout`). We do **not** claim "
        "measured subsurface paths, a unique \(K_x\) vs \(K_y\) axis, or novelty in IR "
        "timing.",
        "",
        "We pose **one next-day water-fate predictor** on a closed cyclic-irrigation "
        "district, packaged as a Gymnasium loop that other algorithms can plug into. "
        "Today's pool irrigation is a known forcing (Okura pump totals for "
        "Shiroyama–Jinbee, Imbanuma, Chiba, spread onto cyclic paddies). "
        "The predictor outputs how much each point sent back as surface drainage. An "
        "FAO-56 dual-Kc simulator (Allen et al., 1998) then says how much water the crop "
        "got; leftover shares say how much moved laterally and how much underground "
        "drains recaptured as LFV. Satellite Kc×ET0 is the crop-water score, not a flux "
        "the model is allowed to copy (`force_satellite_et = False`). The intended IR/D "
        "labels are Okura's daily pump and ADCP series; the article publishes monthly "
        "and seasonal totals, so the loader currently interpolates them. **The proxy "
        "labels leak** for discharge until those files are present; irrigation is no "
        "longer a target in the gym, so that leak is not a ranking of farmer IR. "
        "Leftover lateral flow Lf remains Okura's unobserved residual. An optional "
        "side protocol discovers a **linear or nonlinear neighbour-head lateral-flow "
        "equation without soil properties** (`@register_lateral`: "
        "`linear` and `nonlinear` are forms of the same H, against "
        "`head_equalise`). Algorithms register "
        "outside the package (`@register_forecaster`, optional `@register_ratio`, "
        "optional `@register_lateral`) and are scored on real historical years "
        "(2021↔2022). Leftover shares \(K_x,K_y,K_z,K_g\) are coefficients of that "
        "same per-paddy equation H, fitted on one year and frozen on the "
        "other (`ratio_holdout`); hops freeze the same way "
        "(`soil_flow_holdout`). Recapture \(K_x+K_y\) is a same-day "
        "collection share of leftover already sent underground, pinned to Table 5. "
        "Quantum-reservoir and QAOA encodings of the leftover inverse are a side "
        "piece: identifying per-plot ratios that way is underdetermined, and neither "
        "beats a classical walk here.",
        "",
        "**Keywords:** water-fate prediction; gymnasium benchmark; leftover lateral "
        "flow; neighbour-head hops; linear and nonlinear equation discovery; "
        "cyclic irrigation; FAO-56 dual Kc; year holdout; "
        "identifiability; label leakage; Imbanuma",
        "",
        "## 1. Introduction",
        "",
        "Cyclic irrigation reuses drainage at a pumping-station pool. Okura et al. "
        "(2025) closed that water balance for Shiroyama–Jinbee: pumps, ADCPs, leftover "
        "as Eq. 3. The computer-science problem is a **next-day water-fate map**. This "
        "morning we know how much water the pool sent (Okura irrigation, a block total "
        "spread onto points). We ask how much each point got as crop water, how much it "
        "sent back to the pool, how much it sent laterally, and how much underground "
        "drains captured. That is one predictor in one gym, not a new field campaign, "
        "and not two sequential tasks that hide irrigation as a leaked target.",
        "",
        "Lf is a **block residual**, and the share of water applied at a paddy is not "
        "observed: Okura IR is a pump total. A leftover ratio of \(IR+R-D-ET\) at a "
        "point is therefore a ratio of an unknown split. If farmers irrigate a different "
        "pattern, FAO-56 ET changes and that residual pile is a different object — a "
        "frozen leftover *share* need not hold out. The equation we set up to "
        "discover is a **linear or nonlinear** soil-flow map driven by "
        "**neighbour water levels** and **neighbour "
        "FAO-56 ET**, without soil properties: hops equalise leftover head, wetter "
        "plot to the drier neighbour, and do not send leftover toward high ET or "
        "copy the satellite ET map. \(H\) may see how much water and how much ET "
        "each neighbour has. That law has to be the same law if farmers behave "
        "completely differently (the leftover pile and the neighbour ET map both "
        "change; \(H\) is frozen), and it is checked on the real 2021 "
        "and 2022 seasons (`cyclic.soil_flow_holdout`, `@register_lateral`).",
        "",
        "Applied irrigation and discharge are a **separate farmer model** "
        "(`@register_forecaster`): how much water was put on each paddy and how much "
        "was sent back. The equation builder is hydrology, not farmer skill: \(H\) "
        "does not take applied IR as a feature, and it is frozen when that farmer "
        "model changes. One gym couples them — farmer IR/D set today's leftover "
        "heads and FAO-56 ET, then \(H\) moves leftover among up/down/left/right "
        "neighbours — not a chained IR-then-hops learner that hides irrigation as a leaked "
        "target.",
        "",
        "A predictor that writes the satellite ET map into "
        "the flux, treats millimetre-per-day conductivity as a leftover share, or sends "
        "leftover toward high-ET pixels will fit the score and stop being Okura's "
        "identity. So the predictor acts in a constrained simulator: capture is pinned "
        "to Table 5 as an area-weighted mean and is the leftover collection ratio "
        "(same-day LFV from infiltrated leftover); travel is a kinematic hop count for "
        "bund seepage, not a fifth simplex coordinate; simulated ET is FAO-56 on local "
        "water (`force_satellite_et = False`); hops only equalise leftover head, wetter "
        "plot to the drier plot, and do not deliver LFV.",
        "",
        "Three things follow, and we separate them because two are protocol and one is "
        "a finding.",
        "",
        "1. **The protocol.** One gym (`CyclicEnv(given_irrigation=True)`). A "
        "**farmer model** (`@register_forecaster`) proposes per-paddy applied IR "
        "(a split of the known Okura *block* total; the true applied share is "
        "unknown) and surface discharge D. A separate **equation builder** "
        "(`@register_lateral`) is the per-paddy UDLR leftover law \(H\), with linear "
        "and nonlinear as forms of that same \(H\). Leftover \(K_i\) are coefficients "
        "of \(H\) at every paddy and freeze across years "
        "(`ratio_holdout`); hops freeze the same way, including under a "
        "counterfactual IR schedule that would change ET (`soil_flow_holdout`, "
        "`discover_lateral_equation`). In-sample capture is not that ranking.",
        "2. **What is identified, and what is not.** Only the sum \(K_x+K_y\) reaches "
        "the routing, so the axis split is reported, not identified. Leftover volume is "
        "an algebraic consequence of the ET, irrigation and discharge errors, so it adds "
        "no independent constraint. That leaves roughly two independent observations of "
        "routing for four declared unknowns (§4.5).",
        "3. **Recapture is collection, not lining-catch.** With capture pinned to "
        "\(LFV_{pump}/LfA\), that share of leftover is counted as LFV the same day if "
        "the pan has already sent it underground (§4.4, §5.2). Pond leftover cannot be "
        "LFV. Hops equalise infiltrated remainder as bund seepage; they are not the "
        "LFV path. If seasonal infiltration cannot supply Table 5, that is a Beven "
        "limit on the pan assumption, not a licence to fit `plow_pan_k`.",
        "",
        "What this is not: a claimed new leftover flow law (head-equalise hops are the "
        "baseline equation, not a reported win), a demonstrated win for QAOA or QRC, "
        "or a claim that the daily pump files are in the repository. Daily irrigation "
        "and discharge from Okura et al. are the intended IR/D labels; the loader has "
        "the hook for them (`imitation.load_observed` prefers "
        "`data/observed/okura_{year}_blocks.csv`) and interpolates the published tables "
        "until they are present. Okura IR is a block/pump total, not a measured "
        "per-plot delivery: the gym's default disaggregation is uniform millimetres on "
        "cyclic paddies. An algorithm may propose another spatial split; the env "
        "renormalises it so each block still matches Okura's volume.",
        "",
        "### 1.1 Related work, and what we took from it",
        "",
        "Three literatures already cover most of what this repository does. We cite them "
        "to bound novelty, and then we say which design choices we took from them rather "
        "than rediscovering.",
        "",
        "**Crop-management environments are established.** Gymnasium environments wrapping "
        "process-based crop models exist and are maintained: CropGym over PCSE "
        "(Kallenberg et al., 2023), gym-DSSAT over DSSAT, reviewed with the wider setting "
        "by Gautron et al. (2022), CyclesGym over the multi-year Cycles model (Turchetta "
        "et al., 2022), and AquaCrop-Gym / AquaCropGymnasium over AquaCrop-OSPy "
        "(Alkaff et al., 2025). Chen et al. (2021) trained a deep Q-network to irrigate "
        "paddy rice from short-term rainfall forecasts at Nanchang, with conventional "
        "flooding as the baseline. Daily `reset`/`step` irrigation decisions against a "
        "water-balance model are therefore **not** a contribution of this work. What those "
        "environments do not have is the setting here: they are single-field and "
        "single-policy, their reward is yield and applied water, and every quantity the "
        "agent is scored on is observed at the scale the agent acts on. None of them "
        "contains a district that reuses its own drainage, and none asks an agent for a "
        "quantity nobody measured. None of them takes district irrigation as a known "
        "forcing and asks for the four-way fate of that water.",
        "",
        "*What we took.* CropGym and CyclesGym evaluate on held-out weather; Chen et al. "
        "(2021) go further and keep the rainfall the agent sees (a forecast) distinct from "
        "the rainfall that generates the water-balance outcome (the station observation). "
        "That is the opposite of our proxy, which interpolates IR/D labels from the same "
        "Open-Meteo series the policy observes (§2.2). We therefore refuse to rank "
        "farmer discharge on the proxy, keep yesterday-weather as an observation rather "
        "than a forecast, and treat a wet-year versus dry-year split as the gym protocol "
        "(`cyclic.holdout`, §2.4). CyclesGym's weather generator is the right tool once "
        "labels are independent of it; it is the wrong tool while they are not.",
        "",
        "**Cyclic irrigation has been modelled before.** Fukuda et al. (1999) simulated a "
        "reused-water irrigation system in a low-lying creek district with a complex tank "
        "model, calibrating 157 parameters; Hatcho et al. (2018) built a system-dynamics "
        "water and nitrogen model of a cyclic district in Shiga and used it for management "
        "scenarios. On the same Imbanuma cyclic district as Okura et al. (2025), Iwasaki "
        "and Iida (2024) recalibrated a tank model of canal water level, improving NSE "
        "from −1.02 to 0.17 by adjusting field-stage tank parameters. All three are lumped "
        "or block-scale with calibrated storage parameters. Simulating cyclic irrigation is "
        "therefore not new either. The difference here is that the reuse pathway is a "
        "*per-plot partition that an external algorithm supplies*, rather than a calibrated "
        "coefficient of the model.",
        "",
        "*What we took.* Iwasaki and Iida (2024) closed canal-level NSE on this district "
        "by retuning field-stage tanks. That is the absorption we refuse: a lumped storage "
        "coefficient that improves a canal hydrograph without identifying leftover. "
        "Fukuda's 157-parameter tank is the same failure mode at larger scale. Leftover "
        "shares stay outside the simulator, and a pan-supply miss on LFV_pump is reported "
        "as structure rather than folded into `plow_pan_k` or a tank parameter.",
        "",
        "**Leftover as a plot residual, and lateral seepage as a conductivity, are not new "
        "either.** Playán et al. (2008) estimated inter-paddy discharge on six connected "
        "paddies by writing the water-balance residual forward and backward and averaging, "
        "with measured ponding depths and a known field inflow and outflow. Tsubo et al. "
        "(2007) coupled FAO crop ET with Dupuit seepage through bunds along a rainfed "
        "toposequence and validated ponding against field water levels. Song et al. (2020) "
        "showed that irrigation return-flow estimates from a water-balance residual have "
        "wide uncertainty unless drainage parameters are physically constrained. He et al. "
        "(2025) measured nitrogen lateral seepage through paddy bunds under four "
        "irrigation–drainage regimes and found the flux concentrated in the shallow bund "
        "(the 10–20 cm layer did not delay with controlled irrigation) and reduced by "
        "controlled irrigation and drainage, not by evaporative demand. Those papers are "
        "the nearest hydrologic neighbours. They do not pose leftover shares as the output "
        "of a registered predictor scored only on a district-scale identity, and they do "
        "not forbid treating millimetre-per-day conductivity as a leftover share. Tsubo's "
        "Dupuit seepage is exactly the cheat §3.1 forbids; Playán's residual is computed "
        "on six instrumented paddies, not on an unobserved block Lf.",
        "",
        "*What we took.* Playán et al. needed measured ponding depths to estimate a plot "
        "residual; we do not have those depths, so P2 is not Playán's residual at plot "
        "scale and we do not claim leftover is observed. Tsubo et al.'s Dupuit seepage "
        "needs a toposequence slope; Okura's Imbanuma blocks are lake-edge paddies on "
        "essentially flat land, so leftover as millimetre-per-day conductivity is even "
        "less justified here — that is why §3.1 forbids it. Song et al. showed return-flow "
        "uncertainty explodes unless drainage parameters are physically constrained; we "
        "put `canal_hops` outside `identify.ACTIVE` for that reason, not as a style "
        "preference. He et al. (2025) give independent physical support for hops that "
        "equalise leftover head and do not follow the satellite ET map: bund seepage is "
        "driven by the irrigation–drainage regime and shallow ponding, not by canopy "
        "demand.",
        "",
        "**Non-identifiability is old news in hydrology, and that helps rather than hurts.** "
        "Beven's equifinality thesis (Beven, 2001, 2006; Beven and Binley, 2014) is that "
        "many parameter sets and model structures reproduce the available data acceptably, "
        "so a single optimum should not be expected, and that models should be judged "
        "against limits of acceptability set before running them. Prieto et al. (2021) "
        "turned that into a ranking: from streamflow they could identify routing "
        "mechanisms with high power, while percolation was the least identifiable process, "
        "and the same ranking held when they moved to ungauged catchments (Prieto et al., "
        "2022). Beven's position also gives §5.2 its standing: when no candidate satisfies "
        "the limits, that is evidence about model structure, inputs or data, and is "
        "scientifically informative rather than a failed experiment. Our order-of-magnitude "
        "LFV_pump shortfall is a structural-inadequacy result of exactly that kind, "
        "reported with the mechanism identified rather than absorbed into a calibrated "
        "coefficient.",
        "",
        "*What we took.* Limits of acceptability are set before the search: capture pinned "
        "to Table 5, hops that only equalise head, μ not on the simplex. Prieto et al.'s "
        "ranking is how we read the leftover vector. \(K_x+K_y\) (whether leftover enters "
        "the drain graph) is the routing-like coordinate we can hope to constrain from "
        "pool inflow; \(K_z\) versus ground is the unsaturated/percolation split we do "
        "not claim to identify (§4.5). Preferring 'not identified' to a wrong mechanism "
        "is their reliability criterion and ours.",
        "",
        "**The learning setting has a name.** Task P2 asks for per-plot shares while the "
        "only supervision is a published district aggregate. That is learning from "
        "aggregate observations — learning from label proportions (Quadrianto et al., "
        "2009) generalised to real-valued bag-level targets and regression by Zhang et al. "
        "(2020). Our aggregation operator is not their mean, sum or rank: it is a "
        "season-long FAO-56 and routing simulator, and the quantity being aggregated "
        "\(L\) is itself a residual of other predicted quantities. Zhang et al.'s notion "
        "of consistency only up to an equivalence relation is the right way to read §3.2 "
        "and §4.5: the axis split \(K_x\) versus \(K_y\) is precisely a direction the "
        "aggregation operator cannot see, so we report the equivalence class rather than "
        "a point estimate of each axis.",
        "",
        "**And the failure modes have names too.** The proxy-label problem in §2.2 is "
        "leakage in the sense catalogued by Kapoor and Narayanan (2023), whose survey found "
        "it across 294 papers in 17 fields and whose civil-war case study lost its "
        "ML-over-regression advantage once leakage was fixed. A predictor that recovers our "
        "interpolation rule rather than reasoning about water is learning a shortcut in the "
        "sense of Geirhos et al. (2020): a rule that scores well on the benchmark and does "
        "not transfer. Chen et al. (2021) already avoided that particular shortcut for "
        "rice irrigation by scoring against observed rain while the agent saw a forecast; "
        "our proxy does the reverse, which is why we report both failure modes against "
        "ourselves rather than waiting to be told.",
        "",
        "**Dual Kc on wetting–drying rice is not new either.** Xu et al. (2017) calibrated "
        "FAO-56 dual crop coefficients for paddy rice under drying–wetting cycles in East "
        "China: locally adjusted \(K_{cb}\) matched measured ET, while the FAO-recommended "
        "\(K_{cp} = 1.2\) was too low for the pulse after wetting. *What we took.* Dual Kc "
        "is the simulated flux because wetting–drying paddies are the case Xu tested, and "
        "the wetting-day Ke cap is their \(K_{cp}=1.29\) (`physics.kc_wet`); the satellite "
        "map remains a single-coefficient *target* in the loss. Replacing FAO-56 "
        "with that map (`force_satellite_et`) would undo the separation Xu showed is "
        "needed after a wetting pulse (§4.2).",
        "",
        "So the honest novelty claim is narrow. The **core contribution is inferred "
        "underground flow ratios**: a per-point leftover simplex (capture \(K_x+K_y\) "
        "→ same-day LFV vs stay \(K_z+K_g\) → deep/dWB) under known surface IR/D, with "
        "Table 5's \(LFV_{pump}/LfA\) as the district pin on same-day collection of "
        "infiltrated leftover, hops forbidden from delivering LFV — surface IR/W may be "
        "known (including digitised Fig. 8) and are **not** the novelty "
        "(`cyclic.infer_underground_ratios`, `underground_ratio_holdout`). Around that "
        "core sit supporting pieces: (i) a prediction task in which the "
        "supervised quantity is an unobserved block residual and the aggregation operator "
        "is a water-balance simulator, posed with the constraints that keep that residual "
        "equal to the published identity; (ii) the two-stage composition in which a "
        "sequential policy's rollout becomes the input distribution for an inverse problem, "
        "with an explicit freeze between the stages; and (iii) a "
        "protocol to discover a linear or nonlinear neighbour-head lateral-flow "
        "equation without soil properties, required to remain the same if farmers irrigate differently, "
        "and scored on the historical 2021 and 2022 seasons (`@register_lateral`: "
        "`linear`, `nonlinear`, or `head_equalise`; `soil_flow_holdout`). A win of "
        "that equation over head-equalise hops is not "
        "claimed. It is not a new "
        "environment API, not a new cyclic "
        "irrigation timing model, not a claim that leftover is observed in the field, "
        "and not lining-catch of hopped leftover. A search of crop-management Gymnasium environments "
        "(including CyclesGym), rice irrigation DQN with weather forecasts, cyclic-"
        "irrigation tank and system-dynamics models, plot-scale paddy water balances with "
        "FAO ET and Dupuit or bund seepage, dual-Kc rice under drying–wetting, irrigation "
        "return-flow identification, and hydrological mechanism-identifiability rankings "
        "did not turn up a registered leftover-share predictor scored on Okura's Table 5 "
        "capture identity under the anti-cheat hop rule.",
        "",
        "## 2. Prediction tasks, labels, and protocol",
        "",
        "### 2.1 Site and hydrologic identities",
        "",
        "The site is the Shiroyama–Jinbee (白山甚兵衛) district of the Imbanuma Land "
        "Improvement District, on Lake Kita Imbanuma, Chiba, Japan (Okura et al., 2025). "
        "Okura's North block is Shiroyama; South is Jinbee. The cyclic irrigation season "
        f"is 17 April–25 August, {n.season_days} days, in 2021 (wet) and 2022 (dry). "
        "The main drainage canal runs along the lake edge; drainage from both blocks "
        "converges at the pumping-station pool; LFV_pump is subsurface recapture into "
        "that reuse system, not lining-catch of hopped leftover.",
        "",
        "Okura et al. (2025, §2) measured irrigation IR from pump records, surface "
        "drainage D with ADCPs in the main drainage canal (including under backwater), "
        "lake withdrawal W and outflow O from the station, and on-site weather. Those "
        "IR and D series are daily in the field campaign and are the **intended IR/D labels**. "
        "Irrigation is a gym forcing; discharge remains a predicted farmer head. The "
        "published article reports monthly and seasonal totals. The "
        "loader currently interpolates those totals onto 131 days "
        "(`cyclic.okura.interpolate_daily`, Open-Meteo daily shape, rescaled to Table 2, "
        "IR−D from Table 7 WUE' rescaled to Table 4). Ingesting the measured files is a "
        "data-path change — `imitation.load_observed` already prefers "
        "`data/observed/okura_{year}_blocks.csv` — but it is not score-neutral, for two "
        "reasons set out in §2.2 and §2.4: the interpolation leaks the observations, and "
        "the fitted loss compares season totals. Lateral flow Lf was not "
        "measured. Eq. 3 is still the residual that closes each cyclic block. Okura's "
        "own comments note that North-block season WUE' does not close against Table 4 "
        "under the MAFF effective-rain rule.",
        "",
        "Plot geometry is rebuilt from MAFF Fude polygons clipped to the district "
        "outline and split at the pumping station (`cyclic.blocks`). Cyclic versus "
        "non-cyclic membership is not taken from a GIS attribute. Following Okura "
        "§2.1 / Fig. 1a, parcels nearest the main canal (then the pool) are filled "
        "until Table 1 cyclic area is met; the remainder is non-cyclic. Parcel areas "
        "are then scaled so each block's cyclic and non-cyclic totals match Table 1 "
        "exactly. Non-cyclic drainage does not enter the ADCP canal or LFV_pump.",
        "",
        "Canopy and the satellite ET *target* come from Sentinel-2 NDVI (Drusch et al., "
        "2012) stored as 10 m, 10-day medians where available, otherwise 300 m "
        "composites. Reference ET0 and the farmer's yesterday-weather vector come from "
        "the Open-Meteo Historical Weather API (Zippenfenig, 2024), which serves "
        "reanalysis including ERA5 (Hersbach et al., 2020). The satellite ET target "
        "on a pass day is Kc(NDVI) × ET0, not a thermal ET product.",
        "",
        f"Table 1 areas: North cyclic {n.north_ha['cyclic']} ha, non-cyclic "
        f"{n.north_ha['non_cyclic']} ha; South cyclic {n.south_ha['cyclic']} ha, "
        f"non-cyclic {n.south_ha['non_cyclic']} ha. Seasonal cyclic-block rain is "
        f"{n.rain_2021:.0f} mm in 2021 and {n.rain_2022:.0f} mm in 2022 (Table 4).",
        "",
        "### 2.2 Task: next-day water fate, given pool irrigation",
        "",
        "A **water-fate policy** enters from `example.py`, not from inside `cyclic` "
        "(`@register_forecaster`, `FarmerPolicy.decide_batch` or `WaterFatePolicy`). "
        "Each morning an `ObservationBatch` (ponding, Open-Meteo yesterday, last "
        "Sentinel-2 NDVI and Kc×ET0, own stores, **irrigation_sent_mm**) maps to "
        "discharge_mm per plot. Irrigation_sent_mm is today's Okura pool delivery at "
        "that point: a block/pump total spread uniformly on cyclic paddies unless the "
        "algorithm proposes another disaggregation, which the env renormalises to the "
        "Okura volume. Crop-got is simulated FAO-56 ET, scored against satellite "
        "Kc×ET0. Sent-back is surface D, scored against Okura ADCP (or the interpolation "
        "until those files are loaded). Lateral is leftover versus Table 4 Lf. LFV is "
        "same-day recapture versus Table 5 LFV_pump spread over the season. "
        "`CyclicEnv(given_irrigation=True)` is that loop; irrigation weight is 0 so the "
        "known forcing is not a leaked target.",
        "",
        "Baselines that already run in the gym:",
        "",
        "- **persist** — copy yesterday's discharge (`last_discharge_mm`).",
        "- **rainfed** — send nothing back.",
        "- **et_follower** — drain above a pond ceiling; IR is ignored.",
        "- **linear** / **linear_fate** — linear map from weather, pond, given IR and "
        "yesterday's return to D.",
        "- **field_aggregate** — replay Okura D spread onto paddies (a D oracle, not a "
        "forecast).",
        "- **table leftover shares** — `construct_ratios` / `leftover_shares='tables'`, "
        "Kx+Ky pinned to Table 5; the default gym leftover field.",
        "- **classical walk** — `identify.solve_classical`, eight-eval Gaussian walk on "
        "district-mean shares (in-sample unless frozen on the other year).",
        "- **quantum (side piece)** — QRC or QAOA on the same leftover inverse; optional, "
        "CLI-only, not a claimed win.",
        "",
        "The historical coupling is the same loop on a training year: the policy sees "
        "what farmers did (or the proxy of it) and is then scored on the held-out year. "
        "That is how we propose to predict current farmer return flow, and how we test "
        "whether leftover hydrology holds out through the years.",
        "",
        "**The proxy labels leak.** `okura.interpolate_daily` builds the interpolation "
        "from the same Open-Meteo series the policy observes: rain and ET0 carry the "
        "daily shape rescaled to Table 2, drainage is spread half by day count and half "
        "by that day's rainfall share, and irrigation is uniform over each month's "
        "irrigating days with a June mid-summer gap. A discharge head that reconstructs "
        "that rule — D linear in rain — will beat one that reasons about water. So a "
        "sent-back score on the proxy measures how well an agent inverted our "
        "interpolation, not how well it forecasts a farmer. We report D numbers on the "
        "proxy only as a smoke test. Rankings are meaningful once measured daily series "
        "are in `data/observed/`; they are not transferable from the proxy, and the "
        "earlier claim in this repository that the proxy and the measured series are "
        "interchangeable was wrong. Giving the agent today's IR does not create that "
        "leak — the leak was already in the D labels — but it does stop us pretending "
        "that reconstructing interpolated IR is farmer skill.",
        "",
        "### 2.3 Discovering a neighbour-head lateral equation",
        "",
        "The hydrology we want is an equation for how leftover moves between "
        "neighbouring paddies **without soil properties**: "
        r"\(\Delta L_i = H(h_i, \{h_j, ET_j\}_{j\sim i})\). "
        "\(H\) sees how much water each neighbour holds and how much FAO-56 ET "
        "that neighbour used (`neighbour_depth_mm`, `neighbour_et_mm`). It does not "
        "see Dupuit \(K\), a learned plow-pan conductivity, or the applied "
        "irrigation share (Okura IR is a pump total: the **per-paddy irrigation share "
        "is unknown**). Knowing neighbour ET is not sending leftover toward high ET, "
        "and it is not copying the satellite Kc×ET0 map into the hop "
        "(`force_satellite_et = False`). If farmers irrigate a completely different "
        "pattern, FAO-56 ET and leftover \(IR+R-D-ET\) both change, and a frozen "
        "leftover *share* (`ratio_holdout`) need not survive that counterfactual. "
        "\(H\) is frozen and is fed the new neighbour heads and new neighbour ET. "
        "Baseline hops still equalise leftover head, wetter to drier "
        "(`neighbour_head_drop_mm` on the observation). "
        "Accuracy is checked on the real historical seasons 2021 (wet) and 2022 "
        "(dry): freeze \(H\) on one year and score it on the other "
        "(`cyclic.soil_flow_holdout`, `discover_lateral_equation`). That is an "
        "**ML problem**: linear and nonlinear are **forms of the same equation** "
        "H, not two separate models (`linear`, `nonlinear` against the physics "
        "baseline `head_equalise`). \(H\) is evaluated at **every paddy** "
        "and checks that paddy's **up, down, left and right** neighbours (north, "
        "south, west, east); a missing edge neighbour is skipped, not treated as "
        "a dry plot. Register a candidate with "
        "`@register_lateral`; a win of either class over head-equalise hops is not "
        "claimed. Plot-scale ΔL is unobserved, so a fit in the demo uses the hop "
        "operator as teacher; the ranking is the frozen equation on the other year. "
        "Applied IR and D are **not** this object: they are a separate farmer model "
        "(`@register_forecaster`). Leftover \(K_x, K_y, K_z, K_g\) are **part of the "
        "same per-paddy equation** (`LateralPolicy.partition`), not a separate Ratio "
        "finder. `@register_ratio` remains an optional override of those coefficients "
        "(`RatioPolicy.partition`, `FrozenStay`). `absorb(season)` can still pin stay "
        "on the training year. Capture "
        "\(K_x+K_y\) is set to that year's "
        r"\(LFV_{pump}/LfA\) in the constructed fields and only penalised during a search "
        "(§3.2); stay stands in for district storage. The leftover-share **ranking** "
        "is `cyclic.ratio_holdout`: freeze the fitted \(K_i\) and score leftover, LFV "
        "and FAO-56 ET on the other year. Refitting stay to the evaluation leftover "
        "is not allowed (`FrozenStay`). District-mean capture is not a constant of "
        "the field: "
        f"it is {_fmt(n.capture_2021)} in 2021 and {_fmt(n.capture_2022)} in 2022. A 2021 "
        "field is scored against 2022's Table 5. That gap is the share holdout, not "
        "the soil-flow holdout. Quantum encodings of the same simplex are a side "
        "experiment, not a second required predictor. The score still does not "
        "invent a plot-scale observation of leftover (§4.5).",
        "",
        "### 2.4 Evaluation protocol, and which score is daily",
        "",
        "There are two scoring paths and they differ in an important way.",
        "",
        "- **`CyclicEnv` (Gymnasium `reset`/`step`) is daily.** Each step compares that "
        "day's crop-got (FAO-56 vs Kc×ET0, pass days ×2), sent-back (block drainage), "
        "leftover against Table 4 Lf spread over the season, and LFV against Table 5 "
        "spread over the days, each as "
        r"\(\exp(-|\mathrm{sim}-\mathrm{obs}|/\mathrm{scale})\) in (0, 1]. Gym weights "
        f"are {_weights(n.gym_weights)}. Open-loop `given_irrigation=False` still uses "
        f"{_weights(n.env_weights)} and scores irrigation as well.",
        "- **`HydrologyProblem` fits season totals.** Its `irrigation` and `discharge` "
        "terms compare season sums, so any daily schedule with the right total scores "
        "identically. This is deliberate for leftover identification, where only the season "
        f"closes against Tables 4–5, but it means the fitted loss cannot see a wrong "
        f"schedule. The day-by-day errors are therefore computed as well and reported at "
        f"weight 0: `{'`, `'.join(n.reported_only)}` in `identify.OBJECTIVES`. They are "
        "the numbers a discharge head has to reduce once measured series are loaded; keeping "
        "their weight at 0 leaves the fitted loss comparable with saved checkpoints.",
        "",
        "`run_finder` freezes the best registered water-fate agent (lowest block IR/D "
        "loss among `@register_forecaster` maps defined outside the package; fallback "
        "`et_follower` if nobody is registered) before evaluating leftover shares, so "
        "ratio identification cannot retune irrigation. Package builtins (rainfed, "
        "flooded, linear, linear_fate, persist, field_aggregate, ratio_finder, "
        "et_follower, satellite) are not in the "
        "`registered_forecasters()` contest. Freezing is necessary but not sufficient: "
        "`farm_to_main` multiplies simulated drainage into the pool, so it absorbs "
        "whatever drainage bias the frozen policy has (§5.3).",
        "",
        "Cross-year generalisation is the CropGym / CyclesGym split: "
        "`cyclic.holdout` (farmer D), `cyclic.ratio_holdout` (leftover \(K_i\)), "
        "`cyclic.soil_flow_holdout` (neighbour-head hops, no soil properties), and "
        "`cyclic.validation.split` train "
        "on one of 2021/2022 and evaluate on the other. Held-out leftover scores never "
        "retune the frozen field. Chen et al. (2021) keep forecast rain (what the agent "
        "sees) distinct from observed rain (what generates the water-balance outcome); we "
        "keep yesterday-weather as the observation and today's rain as the physics, which "
        "is that split for one-day lag. `solve_classical` still writes an in-sample "
        "checkpoint (`best_{year}_season.npz`); that file is not a holdout ranking until "
        "it is frozen through `ratio_holdout`. The proxy still "
        "cannot support a discharge ranking, because both years' daily shapes are "
        "interpolated from the same weather family (§2.2).",
        "",
        "## 3. Constraints and penalties",
        "",
        "These rules keep the leftover inverse identifiable. They divide into rules the "
        "code enforces unconditionally and targets it merely penalises, and the division "
        "matters: only the first kind cannot be traded away by a search. The list is not "
        "invented from scratch. Song et al. (2020) required physically constrained "
        "drainage parameters before a return-flow residual is usable; Tsubo et al. (2007) "
        "used Dupuit seepage on a sloping toposequence, which Imbanuma is not; He et al. "
        "(2025) measured bund seepage as a shallow, management-driven flux rather than an "
        "ET-driven one; Prieto et al. (2021) preferred 'not identified' to a wrong "
        "mechanism. The five enforced rules below are those papers applied to this "
        "simplex.",
        "",
        "### 3.1 Enforced",
        "",
        "1. **Leftover is a ratio of today's pile**, not mm d⁻¹ conductivity. "
        r"\(K_x+K_y+K_z+K_g=1\) is imposed by renormalisation in both "
        "`HydrologyProblem.unpack` and `canal.apply_canal`, so no search point can leave "
        "the simplex. Stay \(K_z+K_g\) is district storage (dWB), not a second canal.",
        "2. **Travel is not a fifth leftover ratio.** Mean hops in a day are "
        r"\(\mu=vT/\Delta x\) (Berkowitz et al., 2006, review Poisson waiting times in "
        "CTRW; here the mix is discrete neighbour hops, and because the hop operator is "
        "nonlinear the mixture is a discrete construction rather than that paper's "
        "semigroup). If `canal_hops` is set, μ is that count. `canal_hops` is not in "
        "`identify.ACTIVE`, so no solver can tune it. That is Song et al.'s physical "
        "constraint on a drainage parameter, written as a search mask rather than a "
        "prior.",
        f"3. **Satellite ET is scored, not copied.** Default `force_satellite_et = "
        f"{n.force_satellite_et}`, and `physics.advance` calls "
        "`apply_et(..., force=False)` on the default path. FAO-56 already says "
        "ET = (Ks Kcb + Ke) ET0 and that Ks and Ke fall when the plot is dry (Allen et "
        "al., 1998, Ch. 7). Writing the map in as the day's ET makes leftover no longer "
        "the FAO-56 residual.",
        "4. **High satellite ET does not pull leftover.** A hop moves leftover only "
        "from the currently wetter plot to the drier plot, at most until leftover "
        "heads equalise (`graph.move_stock`). He et al. (2025) measured the analogous "
        "physical flux as bund seepage driven by irrigation–drainage regime and shallow "
        "ponding, not by canopy demand; Tsubo et al.'s Dupuit conductivity is forbidden "
        "here because the district is flat. Hops are bund seepage of infiltrated "
        "remainder, not the LFV path (§4.4).",
        "5. **Non-cyclic leftover does not feed LFV_pump.** Cyclic D returns to the "
        "pool the same day (× farm_to_main). QAOA, if used, is on O'Malley-scale "
        "clusters, not one bit per paddy.",
        "",
        "### 3.2 Penalised, not enforced",
        "",
        r"**Capture.** \(K_x+K_y\) is set to \(LFV_{pump}/LfA\) wherever the shares are "
        "*constructed*: `construct_ratios` returns it, and `canal.pixel_ratio_fields` "
        "puts that value uniformly on cyclic cells (no lining-proximity target). The "
        f"area-weighted mean is therefore the pin exactly ({_fmt(n.capture)} in {n.year}). "
        "But during a search capture is only *penalised*, by the `darcy` term "
        r"\((\mathrm{sim}-\mathrm{table})^2\) at weight "
        f"{n.identify_weights['darcy']:.1f}. At the deviation in §5.3 that penalty is "
        "about 0.002 against a loss near 1.6, so it is close to free. Calling capture "
        "'pinned' during identification would be wrong.",
        "",
        "**The axis split.** \(K_x\) and \(K_y\) enter the simulation only through their "
        "sum: `apply_canal` forms recapture as leftover times \(K_x+K_y\) and sends "
        "that share to the pool the same day from infiltrated leftover. "
        "`graph.move_stock` still equalises infiltrated remainder as bund seepage. "
        "`flow_kx_mm` and `flow_ky_mm` are logged and mapped but never fed back. So the "
        "split is **not identified** and is reported like the reserved conductivities "
        "`kx_mm_d`, `ky_mm_d`. Reading a fitted \(K_y>K_x\) as evidence about drain "
        "orientation, as an earlier version of this manuscript did, is unsupported.",
        "",
        "## 4. Simulator (FAO-56 stores, leftover simplex, hops)",
        "",
        "### 4.1 Okura identities as scoring contracts",
        "",
        "Volumes in m³ unless noted. A_cyc is cyclic area. The code in `cyclic.okura` "
        "recomputes every derived table entry; §5.1 reports which identities close.",
        "",
        r"- **Okura 1** (district storage): \(\Delta WB = W - O + (R - ET)\,A_{cyc}/1000 - IR_{ncb}\)",
        r"- **Okura 3** (block leftover, not measured): \(L = IR + R - D - ET\)",
        r"- **Okura 4** (pumping-station pool): \(\Delta pool = W - O + LFV_{pump} + D_N + D_S - IR_{all}\)",
        r"- **Okura 6** (LFV from the pool residual): \(LFV_{pump} = IR_{all} - W + O - D_N - D_S\)",
        r"- **Okura 7** (block storage after LFV leaves): \(\Delta WB = LfA_N + LfA_S - LFV_{pump}\)",
        r"- **Okura 11–14**: \(WUE' = ET/(IR - D + ER)\), \(RUE = D/IR_{all}\), \(LUE = LFV_{pump}/IR_{all}\)",
        "",
        "Effective rain ER follows the MAFF daily rule as stated by Okura et al. "
        "(2025, §2.6): none below 5 mm, 80% between 5 and 80 mm, 64 mm above 80 mm "
        "(`okura.effective_rain_mm`).",
        "",
        "### 4.2 FAO-56 dual crop coefficient (the simulated flux)",
        "",
        "Each plot (or satellite cell aggregated to a plot) carries ponded depth, "
        "surface-layer depletion, root-zone deficit, perched water above the farm-drain "
        "level, unmet evaporative demand `need`, and in-transit drain stock. One day "
        "(`physics.advance`) adds rain and irrigation to the pond, then takes "
        "evapotranspiration from the stores, then farmer discharge, bund overflow and "
        "outlet leak, then infiltration proportional to ponded depth "
        f"(`plow_pan_k_mm_d` = {_fmt(n.plow_pan_k_mm_d, 1)} mm d⁻¹ at "
        f"`pond_infiltrate_ref_mm` = {_fmt(n.pond_infiltrate_ref_mm, 0)} mm, capped at "
        "the pond; not a leftover-simplex ratio and not gated by TEW depletion), then "
        "leftover routing.",
        "",
        "Evapotranspiration is FAO-56 Chapter 7 (Allen et al., 1998):",
        "",
        r"\[ ET = (K_s K_{cb} + K_e)\,ET_0 \]",
        "",
        r"Fractional cover is Carlson and Ripley's (1997) \(f_c = N^{*2}\), "
        r"\(N^* = (NDVI - NDVI_{soil})/(NDVI_{canopy} - NDVI_{soil})\) clipped to [0, 1], "
        f"with NDVI_soil = 0.15 and NDVI_canopy = 0.80 (`physics.canopy_cover`). "
        f"Basal coefficient interpolates from Kcb_bare = {_fmt(n.kcb_bare, 2)} to "
        f"Kcb_full = {_fmt(n.kcb_full, 2)}. Stress Ks is 1 while the plot is ponded or "
        f"root-zone depletion is below RAW = {_fmt(n.raw, 0)} mm; it falls linearly to 0 "
        f"at TAW = {_fmt(n.taw, 0)} mm. FAO-56 Table 22 gives rice the smallest listed "
        f"depletion fraction p = {_fmt(n.rice_p, 2)}. Evaporation Ke uses Kr on the "
        f"surface layer (REW = {_fmt(n.rew, 0)} mm, TEW = {_fmt(n.tew, 0)} mm; clay-loam "
        f"anchors from FAO-56 Table 19) and the uncovered fraction 1 − fc, capped at "
        f"Kc_max = {_fmt(n.kc_max, 2)} (Eq. 72, unadjusted) except on rain or irrigation "
        f"days, when Xu et al.'s wetting-pulse cap `kc_wet` = {_fmt(n.kc_wet, 2)} is used. "
        "Transpiration is taken from "
        "the root zone, never from today's rain or irrigation directly. Evaporation is "
        "taken from the pond if water stands, otherwise from the surface layer. Unmet "
        "demand is recorded as `need` and is **not** added to `et_mm` unless "
        "`force_satellite_et` is turned on. The default path calls "
        "`apply_et(..., force=False)`.",
        "",
        "Xu et al. (2017) calibrated dual Kc for rice under drying–wetting cycles and "
        "found the dual method feasible once basal coefficients were localised; they also "
        f"found FAO's recommended \\(K_{{cp}} = 1.2\\) too low for the pulse after wetting "
        f"and raised it to {_fmt(n.kc_wet, 2)}. We use that number: on a rain or irrigation "
        f"day the simulated Ke cap is `kc_wet` = {_fmt(n.kc_wet, 2)}, otherwise FAO-56 "
        f"`kc_max` = {_fmt(n.kc_max, 2)}. Kcb stays the Japanese FAO/Vu range "
        f"({_fmt(n.kcb_bare, 2)}–{_fmt(n.kcb_full, 2)}), not Xu's East-China mid-season "
        "1.52. Wetting–drying paddies are the case here, so dual Kc stays the simulated "
        "flux and the satellite map stays a single-coefficient target. Copying the map "
        "into `et_mm` would erase the wetting pulse Xu separated.",
        "",
        "The satellite *target* used in the loss is a single-coefficient map, not the "
        "dual-Kc flux. Vu, Watanabe and Takagi (2005, Table 4) compared FAO-56 Kc for "
        "Japanese rice at Tsukuba (NIAES) and Fuchu (TUAT) lysimeters. Their climate-"
        "adjusted typical values were Kc-ini = 1.10 on all four seasons and Kc-mid "
        "between 1.13 and 1.17. This package interpolates",
        "",
        rf"\[ K_c(NDVI) = {n.kc_ini:.2f} + ({n.kc_mid:.3f} - {n.kc_ini:.2f})\,"
        rf"\mathrm{{clip}}\bigl((NDVI - {n.ndvi_bare:.3f})/"
        rf"({n.ndvi_full:.3f} - {n.ndvi_bare:.3f}), 0, 1\bigr) \]",
        "",
        "and multiplies by Open-Meteo ET0. NDVI_BARE and NDVI_FULL are the interpolation "
        "anchors used in code, not additional Vu table entries. On a Sentinel-2 pass "
        "day the identification loss weights that target twice (`et_w = 2` if `is_pass`). "
        "The environment does the same with `pass_weight = 2`. Neither path copies the "
        "map into `et_mm`.",
        "",
        "When leftover ratios are **not** in use, perched water can still exchange with "
        "neighbours (`lateral.exchange_perched`) and drain by a Hooghoudt equation "
        "(Hooghoudt, 1940) with geometric-mean horizontal K. When `ratio_split` is "
        "true, those two steps are skipped: leftover is the simplex below, not a "
        "conductivity-driven perched flux. Darcy Kx, Ky in mm d⁻¹ remain on Physics "
        "for a later identification and are not in the active leftover vector.",
        "",
        "### 4.3 Leftover simplex (M1–M2)",
        "",
        "After the FAO-56 day,",
        "",
        r"\[ L = R + IR - D_{surface} - ET_{sim} \]",
        "",
        "with D_surface the sum of farmer discharge, bund overflow and outlet leak. This "
        "is signed: on a heavy-rain day with little irrigation it goes negative, which is "
        "what the Table 4 target does too. `physics.advance` returns and logs it signed, "
        "and the `lateral` and `ld_ratio` loss terms use the signed values. Routing is "
        r"the one place it is clipped: `apply_canal` and `graph.hop_stock` take "
        r"\(\max(L, 0)\), because a negative pile has nothing to send downhill. The same "
        "symbol therefore means the signed residual in the loss and the clipped pile in "
        "the routing, and the clip biases seasonal routed volume upward relative to the "
        "signed sum. The positive pile is partitioned once:",
        "",
        r"\[ K_x + K_y + K_z + K_g = 1,\quad K_\bullet \ge 0 \]",
        "",
        r"\[ \mathrm{capture} = K_x + K_y,\qquad \mathrm{stay} = K_z + K_g. \]",
        "",
        "Kx is nominally the east–west share of leftover recapture "
        "and Ky the north–south share, but only their sum is used (§3.2). That sum is "
        "same-day LFV from infiltrated leftover; pond leftover cannot be LFV. Kz returns to "
        "the evaporable topsoil; ground goes to the deeper store. Stay stands in for "
        "Okura's dWB rather than a second canal outlet, with one caveat worth stating: "
        "`CyclicModel._finish_flux` returns the Kz share to `state.surface`, so that "
        "water can evaporate again later in the season and be counted in `et_mm` against "
        "the satellite target. To that extent Kz is delayed ET rather than a storage "
        "change, and equating stay with dWB is approximate. "
        f"The district-mean closed form (`construct_ratios`) sets capture = LFV_pump / "
        f"LfA = {_fmt(n.capture)} in {n.year}, splits stay by TEW/(TEW+TAW) = "
        f"{_fmt(n.topsoil_frac)} so that Kz = {_fmt(n.kz)} and ground = {_fmt(n.ground)}, "
        f"and splits capture equally so Kx = Ky = {_fmt(n.kx)}. farm_to_main is "
        f"Table 5 main-canal D / Table 4 farm D = {_fmt(n.farm_to_main_raw)} before "
        f"clipping to {_fmt(n.farm_to_main)}. "
        "Per-pixel fields (`canal.pixel_ratio_fields`) put that recapture uniformly on "
        "cyclic cells; they do not raise it toward a traced main-drain lining. Kx and Ky "
        "each take half. The same TEW/TAW stay split is kept. A leftover-mass reweight "
        "(`leftover_weighted_stay`) then rescales stay so the leftover-and-area-weighted "
        "mean matches Table 5, without introducing neighbour-relative extra ratios. "
        "DIVIDER is still the North/South block split and the ranking used to fill "
        "Table 1 cyclic area; it is not an LFV sink.",
        "",
        "### 4.4 Graph routing and Poisson hops (M3–M4)",
        "",
        "Soil permeability is two-way in the unused perched-Darcy path. Leftover hops "
        "are two-way on that soil graph as well. Default "
        f"`one_way_to_canal = {n.one_way_to_canal}`: recapture does not use the "
        "main-drain lining as a sink, so hops are bund seepage between neighbours, not "
        "a canal diode.",
        "",
        "One hop H on an axis-aligned leftover edge moves water only from the currently "
        "wetter plot to the drier plot, and at most until the two leftover heads "
        "equalise, accounting for area (`graph.move_stock`). The amount moved is "
        r"\(\min(\mathrm{stock}_{high}\,f,\ \mathrm{cap})\) with "
        r"\(f=\mathrm{clip}(\mathrm{stage}/\mathrm{ref}, 0, "
        f"{n.canal_max_frac:.2f})\\) from the farm-drain stage (not canal-distance "
        "drop), and cap the equalising amount. A hop never moves toward a wetter "
        "neighbour and never inverts leftover head; because the Poisson mixture is a "
        "convex combination of such states, the mixture does not invert head either. "
        "High satellite ET does not pull leftover; only leftover head does. He et al. "
        f"(2025) measured the physical analogue as bund seepage through about "
        f"{n.bund_seepage_mm:.0f} mm of bund (their 10–20 cm layer), driven by the "
        "irrigation–drainage regime rather than canopy demand; leftover hops are that "
        "path (`bund_seepage_mm`), not Tsubo Dupuit on a slope. The "
        "consequence is that the hop operator is **diffusive**: it levels infiltrated "
        "remainder instead of advecting leftover to a lining sink. Recapture does not "
        "use that operator.",
        "",
        "A day is not an inner clock loop. If leftover speed is v and median neighbour "
        r"spacing is \(\Delta x\), the mean number of hops in time T = 1 day is",
        "",
        rf"\[ \mu = vT/\Delta x \quad\text{{(default }} v = {n.leftover_speed_m_d:.1f}\,\mathrm{{m\,d}}^{{-1}}\text{{, bund-seepage scale)}} \]",
        "",
        "unless `canal_hops` is set, in which case μ is that count. The day's stock is "
        "the Poisson mixture",
        "",
        r"\[ \mathrm{stock}(T) = \sum_{k=0}^{k_{\max}} e^{-\mu} \frac{\mu^k}{k!}\, H^k\,\mathrm{stock}(0) \]",
        "",
        "truncated when the remaining tail is below 10⁻¹² and renormalised "
        "(`graph.poisson_hop_weights`, `graph.hop_stock`). k = 0 leaves the pile where "
        "it is. Mass is conserved to numerical precision. Hops act only on infiltrated "
        "leftover that was not recaptured (bund seepage). Recapture itself is",
        "",
        r"\[ LFV = \min\bigl((K_x+K_y)\,L^+,\ I\bigr) \]",
        "",
        "on cyclic cells only: \(L^+=\\max(L,0)\) and \(I\) is that day's pond→soil "
        "infiltration (`canal.apply_canal`). Pond leftover cannot be LFV. Some of that "
        "recapture then fills nearby unsaturated soil (neighbour TEW depletion then TAW "
        "deficit) instead of the pool (`canal.divert_recapture_to_unsaturated`): only "
        "drier neighbours take it, and only up to their remaining capacity, so those "
        "plots can evaporate and transpire on later days (Ke, Ks) against the satellite "
        "ET score. It is delayed ET, like Kz returning to the topsoil, not a copy of "
        "today's map into `et_mm`. That is not a leftover-simplex ratio and does not "
        "use leftover speed. Saturated neighbours leave recapture as LFV. Non-cyclic "
        "leftover never feeds LFV_pump. Cyclic surface drainage that has already been "
        "counted as D returns to the pool the same day, multiplied by farm_to_main.",
        "",
        "The pool (`PumpingStation.settle`) then applies Okura Eq. 4: D_N + D_S + "
        "LFV_pump enter; IR_all is pumped out; W covers a shortfall from the lake; O "
        "spills surplus. Pool geometry is the Okura §2.1 station pool, about 18 × 68 m, "
        "with assumed 4 m depth.",
        "",
        "### 4.5 Evaluation loop",
        "",
        "The unknown vector is (Kx, Ky, Kz, ground, farm_to_main), each in [0, 1], "
        "renormalised to a simplex on the first four. Conductivities kx_mm_d, ky_mm_d "
        "are reserved (`identify.LATER`) and are not searched.",
        "",
        "`run_finder` is the evaluation loop for P2 given a frozen P1. It does **not** "
        "hold satellite ET as the flux. It freezes irrigation and discharge from the "
        "best currently registered forecaster. The leftover pile used to reweight stay "
        "is `leftover_from_season` of that frozen run. The `irrigation` and `discharge` "
        "terms compare season totals against whatever series the loader exposes "
        "(interpolated tables now; Okura pump/ADCP days when those files are present); "
        f"`{'` and `'.join(n.reported_only)}` report the same comparison day by day at "
        "weight 0 (§2.4).",
        "",
        "HydrologyProblem.objectives then scores a full season of FAO-56 + hop routing "
        "with squared relative errors (0 is exact). Default weights are 1 on et, "
        "irrigation, discharge, lateral, ld_ratio, farm_to_main, lfv and darcy "
        f"(the last is (simulated capture − Table 5 capture)²), and 0.5 on pond. "
        f"Explicitly: {n.identify_weights}. Pass days use weight 2 on the ET term. "
        "Lateral compares leftover volume to Table 4 Lf; lfv compares same-day recapture "
        "to Table 5 LFV_pump.",
        "",
        "**How many of those nine fitted terms constrain the routing?** Fewer than the "
        "list suggests, and it is worth being precise because the whole point of the "
        "constraints is identifiability.",
        "",
        "- Simulated leftover is \(R+IR-D-ET\) plot by plot, and Table 4 Lf is the same "
        "residual on the table — §5.1 shows Eq. 3 closing there to within 1 mm. So "
        "`lateral` and `ld_ratio` are algebraic consequences of the `et`, `irrigation` "
        "and `discharge` errors. They reweight the loss; they add no independent "
        "information about where leftover goes. The sentence 'leftover is in the score' "
        "is true and, on its own, misleading.",
        "- `darcy` compares the mean of the constructed capture field against the value "
        "that field was built from, so on the closed-form vector it is a near-tautology.",
        "- `lfv` and `farm_to_main` are the two terms that genuinely test routing: "
        "seasonal pool inflow, and the share of farm drainage arriving in the main canal.",
        "",
        "So the count is about two independent observations of routing against four "
        "declared unknowns, of which the axis split is unidentified and `farm_to_main` is "
        "confounded with the frozen forecaster's drainage bias. The identifiable content "
        "is roughly capture and farm-to-main, with the Kz-versus-ground split entering "
        "only weakly through topsoil refill. In `table_closure_objectives`, which scores "
        "Tables 4–5 alone, `darcy` and `lfv` are identically zero by construction; that "
        "function measures table closure, not model skill. Prieto et al. (2021, 2022) "
        "found the same split in catchment models: routing was among the most identifiable "
        "processes from streamflow, percolation among the least. We encode that ranking "
        "as `identify.PROCESS_ROLE`. Routing-like terms are "
        + ", ".join(f"`{k}`" for k, role in n.process_role.items() if role == "routing")
        + f"; the percolation-like coordinates {n.percolation_coordinates} are reported, "
        f"not identified; only the sum of {n.routing_coordinates} reaches the hop graph.",
        "",
        "The Gymnasium environment (`CyclicEnv(given_irrigation=True)`) uses a four-term "
        "daily water-fate reward, each "
        r"\(\exp(-|\mathrm{sim}-\mathrm{obs}|/\mathrm{scale})\) in (0, 1], with gym "
        f"weights {_weights(n.gym_weights)}. Open-loop `given_irrigation=False` still uses "
        f"{_weights(n.env_weights)} and scores irrigation as a target. Crop-got is FAO-56 versus "
        "Kc×ET0; sent-back is D versus Okura; lateral is leftover versus Table 4 Lf; "
        "lfv is same-day recapture versus Table 5. Satellite ET, discharge, leftover "
        "and LFV are all in the gym score; irrigation is a known forcing.",
        "",
        "Classical identification is a bounded random walk on the district means "
        "(`solve_classical`). Saved best vectors live in `outputs/ratios/best_{year}_season.npz`. "
        "QRC training is CLI-only (`python -m cyclic identify qrc classical` and "
        "`identify qrc spatial classical`); the UI only applies a saved readout.",
        "",
        "### 4.6 Optional solvers (side piece, not the contribution)",
        "",
        "Quantum is a side experiment for identifying leftover ratios, and it is a "
        "tricky one: the simplex is underdetermined (§4.5) before any circuit is run. "
        "O'Malley (2018) posed hydrologic inverse analysis as a binary quadratic "
        "program on a small grid and minimised it on a D-Wave annealer. Here the bit "
        "is leftover assignment, not permeability: x_i = 1 sends cluster leftover into "
        "the graph (Kx+Ky), x_i = 0 keeps it as stay. The Hamiltonian has the same "
        "shape as O'Malley Eq. 2: data misfit to Table 4/5 plus neighbour coupling "
        "J (x_i − x_j)². IBM gate hardware cannot hold one bit per paddy, so cyclic "
        "parcels are k-means clustered to n_nodes bits (default 12) before QAOA "
        "(Farhi et al., 2014) or a classical bit-flip loop. `Physics.as_qubo` on the "
        "full leftover graph refuses n > 64 and tells the caller to coarsen. This is "
        "an optional encoding, not the default identification.",
        "",
        "The quantum reservoir (Fujii and Nakajima, 2017) is a different object. Each "
        "irrigation-season day drives a small frozen circuit (default 4 qubits, 2 "
        "layers of random Rz and ring CX). Only a linear readout is trained, against "
        "a two-box TEW/TAW filling target whose seasonal mean stays on Tables 4–5. "
        "District QRC predicts a daily simplex; spatial QRC predicts stay at a parcel "
        "from leftover and geometry, then splits capture on the local canal axis. "
        "Inference after a statevector train is classical. IBM Runtime is optional "
        "for a few probe days, not 131 separate optimisations.",
        "",
        "## 5. Results (baselines on the current labels)",
        "",
        "### 5.1 Table self-consistency",
        "",
        f"For {n.year}, `okura.check_tables` recomputes Eqs. 1, 3, 6, 7, 12–14, "
        "lake ratio, SDR, and the Table 4 ↔ Table 5 volume translations. Core checks "
        "(everything except Table 7 North WUE' and the Section 3.2.2 prose irrigation "
        "rates) pass in the unit tests for both 2021 and 2022. Known disagreements "
        "kept from Okura's own notes: Table 3 rain/outflow rows were swapped in the "
        "published PDF and are stored corrected (2021 R = 4144, O = 2666 ×10³ m³); "
        "Section 3.2.2 North irrigation 15.3 / 17.5 mm d⁻¹ does not match Table 4 "
        "13.6 / 15.6 mm d⁻¹, and Table 4 is used; Table 6 RUE_N is D_N / IR_N_all; "
        "Table 7 North season WUE' would need more than 80% of rain as effective rain.",
        "",
        f"Closed-form leftover shares for {n.year} from Tables 4–5:",
        "",
        f"- capture Kx+Ky = LFV_pump / LfA = {n.lfv_1000m3:.0f} / {n.lfa_1000m3:.0f} "
        f"= **{_fmt(n.capture)}**",
        f"- stay = **{_fmt(n.stay)}** (Kz = {_fmt(n.kz)}, ground = {_fmt(n.ground)}; "
        f"TEW/(TEW+TAW) = {n.tew:.0f}/{n.tew + n.taw:.0f})",
        f"- farm-to-main = **{_fmt(n.farm_to_main)}** "
        f"(raw Table 5/4 ratio {_fmt(n.farm_to_main_raw)})",
        f"- table-start Kx = Ky = **{_fmt(n.kx)}**",
        f"- simplex sum = **{_fmt(n.simplex)}**",
        f"- `force_satellite_et` on that vector = **{n.shares_force_et}**",
        "",
        f"Cyclic-block Table 4 ({n.year}), mm over the season: North rain {t4n['rain']}, "
        f"IR {t4n['irrigation']}, ET {t4n['et']}, D {t4n['drainage']}, Lf {t4n['lateral']}; "
        f"South rain {t4s['rain']}, IR {t4s['irrigation']}, ET {t4s['et']}, D {t4s['drainage']}, "
        f"Lf {t4s['lateral']}. Eq. 3 on the stored millimetres is within 1 mm of Table 4 "
        f"Lf (South 2021 is 1362 vs 1363). `check_tables` treats that as closed.",
        "",
        _routing_section(n),
        fitted_block,
        qrc_block,
        "### 5.5 What the tests require of the hop operator",
        "",
        "`tests/test_identify.py` builds a four-plot downhill chain and checks: Poisson "
        "weights sum to 1; μ = 0 leaves stock unmoved; larger μ moves more leftover "
        "onto the canal cell while conserving mass; a wetter downhill cell does not "
        "receive leftover from a drier uphill cell. Canal catch plus in-transit stock "
        "equals the captured leftover. `tests/test_paper_claims.py` additionally checks "
        "that two share vectors with the same \(K_x+K_y\) route identically (§3.2), that "
        "the constructed capture field's area-weighted mean equals the Table 5 pin, and "
        "that the reported daily IR/D terms exist at weight 0. Those are the statements "
        "in §3 and §4.4, not extra physics.",
        "",
        "## 6. Discussion",
        "",
        "The contribution is **inferred underground flow ratios**: a per-point leftover "
        "simplex under known surface IR/D, with capture pinned to Table 5 and frozen "
        "across years (`infer_underground_ratios`, `underground_ratio_holdout`). The "
        "gym packages that claim: a farmer model for applied irrigation and discharge "
        "(forcing, not novelty), and an optional equation builder for leftover hops. "
        "Linear and nonlinear are forms of that one \(H\), not a second "
        "hydrology model. Okura et al. (2025) remain "
        "the source of the labels and of the leftover identities. Historical rollouts "
        "are how we propose to predict current farmer return flow and to test "
        "whether that lateral equation holds out through the years.",
        "",
        "Keeping leftover as a ratio of the day's pile is what makes Table 5's "
        "LFV_pump / LfA a paper identity rather than a calibrated conductivity. Travel "
        "distance is handled by μ, which is not on the simplex. And the constraint that "
        "hops only equalise head is what stops a search from discovering that the "
        "cheapest way to fit LFV_pump is to send leftover wherever the loss wants it.",
        "",
        "**Collection is implemented as recapture, not lining-catch.** \(K_x+K_y\) is "
        "the share of leftover that enters the reuse network at the plot and is counted "
        "as LFV the same day, taken only from infiltrated leftover. Farm-drain travel "
        "at daily resolution is hours, not days, so conveyance is not a fifth simplex "
        "coordinate. Pond leftover stays as dWB or tomorrow's D. Hops equalise "
        "infiltrated remainder as bund seepage and do not catch at the lake-edge lining. "
        "If infiltration cannot supply Table 5, that is a pan-supply miss (§5.2), not a "
        "reason to retune `plow_pan_k` against LFV (Iwasaki and Iida's absorption).",
        "",
        "A hydrology result would be one fitted per-paddy UDLR equation H "
        "(linear or nonlinear form of `@register_lateral`) that, with no soil "
        "properties, beats frozen head-equalise hops on the held-out year "
        "(`discover_lateral_equation`) and still equalises leftover head under a "
        "counterfactual farmer IR that would change ET. Linear and nonlinear are "
        "not two separate models. Leftover collection shares are coefficients of that "
        "same H at every paddy; freezing them on the held-out year is `ratio_holdout`, "
        "with daily D scored against Okura's ADCP series and crop-got against satellite Kc×ET0, given "
        "Okura IR. Those comparisons are the protocol; a win is not claimed here. Chen et "
        "al. (2021) already keep forecast rain off the physics rain, which we match by "
        "showing yesterday-weather. Iwasaki and Iida (2024) improved this district's "
        "canal NSE by retuning tanks — the move §5.2 refuses, because it would hide "
        "pan-supply error inside a storage coefficient.",
        "",
        "What this is **not**:",
        "",
        "1. A claim that daily pump/ADCP files are in the repository. The intended IR/D "
        "labels are those series; the current score uses table interpolations. Because "
        "those interpolations are built from the weather the policy observes (§2.2), "
        "discharge scores on them are a smoke test, not a ranking. Cyclic membership is a "
        "canal-distance fill to Table 1.",
        "2. A daily fit. `HydrologyProblem` fits season totals; the daily errors are "
        f"reported at weight 0 (`{'`, `'.join(n.reported_only)}`). A schedule that is "
        "wrong day by day and right over the season is invisible to the fitted loss.",
        "3. An identified axis split. \(K_x\) and \(K_y\) reach the simulation only "
        "through their sum, so reported values of each separately mean nothing.",
        "4. A proven quantum inverse. The spatial QRC collapsed to the Table 5 stay "
        "target exactly (§5.4). QAOA runs on O'Malley-scale clusters, not one bit per "
        "paddy. Both are optional encodings on the same inverse.",
        "5. A demonstrated improvement over a simpler split. Saved `best_*_season.npz` is "
        "an in-sample checkpoint (`load_best_ratios` stamps `force_satellite_et=False`). "
        "Default leftover speed 2 m d⁻¹ is a kinematic starting value for bund seepage, "
        "and recapture does not use it.",
        "6. Forced satellite ET as identification. `force_satellite_et=True` still "
        "exists for bookkeeping (`cyclic.account.force_et`). It is not the default.",
        "",
        "## 7. Reproducibility",
        "",
        "```",
        "python -m cyclic paper 2022",
        "python -m cyclic routing 2022",
        "python -m cyclic identify qrc classical 2022",
        "python -m cyclic identify qrc spatial classical 2022",
        "python example.py",
        "python -m unittest tests.test_paper_claims tests.test_methods tests.test_identify tests.test_scoring",
        "```",
        "",
        f"Season length {n.season_days} days (17 April–25 August). Comparison year "
        f"{n.comparison_year}. Graph routing default `use_graph={n.use_graph}`, "
        f"`one_way_to_canal={n.one_way_to_canal}`. Physics default "
        f"`force_satellite_et={n.force_satellite_et}`. Constructed ratios set "
        f"`ratio_split={n.shares_ratio_split}` and "
        f"`force_satellite_et={n.shares_force_et}`.",
        "",
        "Every Results number is written by `cyclic.manuscript.live_numbers` or copied "
        "verbatim from a file that `cyclic.identify` or `cyclic.qrc` generated. §5.2 "
        "needs a full season per leftover speed, so it is quoted from "
        "`outputs/ratios/routing_{year}.txt` rather than recomputed on import; re-run "
        "`python -m cyclic routing` after changing the routing code.",
        "",
        "## References",
        "",
        "Alkaff, M., Basuhail, A., Sari, Y., 2025. Optimizing water use in maize "
        "irrigation with reinforcement learning. Mathematics 13, 595. "
        "https://doi.org/10.3390/math13040595",
        "",
        "Allen, R.G., Pereira, L.S., Raes, D., Smith, M., 1998. Crop evapotranspiration: "
        "Guidelines for computing crop water requirements. FAO Irrigation and Drainage "
        "Paper 56. FAO, Rome. ISBN 92-5-104219-5. "
        "https://www.fao.org/4/X0490E/X0490E00.htm",
        "",
        "Berkowitz, B., Cortis, A., Dentz, M., Scher, H., 2006. Modeling non-Fickian "
        "transport in geological formations as a continuous time random walk. Reviews "
        "of Geophysics 44, RG2003. https://doi.org/10.1029/2005RG000178",
        "",
        "Beven, K., 2001. How far can we go in distributed hydrological modelling? "
        "Hydrology and Earth System Sciences 5, 1–12. "
        "https://doi.org/10.5194/hess-5-1-2001",
        "",
        "Beven, K., 2006. A manifesto for the equifinality thesis. Journal of Hydrology "
        "320, 18–36. https://doi.org/10.1016/j.jhydrol.2005.07.007",
        "",
        "Beven, K., Binley, A., 2014. GLUE: 20 years on. Hydrological Processes 28, "
        "5897–5918. https://doi.org/10.1002/hyp.10082",
        "",
        "Carlson, T.N., Ripley, D.A., 1997. On the relation between NDVI, fractional "
        "vegetation cover, and leaf area index. Remote Sensing of Environment 62, "
        "241–252. https://doi.org/10.1016/S0034-4257(97)00104-1",
        "",
        "Chen, M., Cui, Y., Wang, X., Xie, H., Liu, F., Luo, T., Zheng, S., Luo, Y., "
        "2021. A reinforcement learning approach to irrigation decision-making for rice "
        "using weather forecasts. Agricultural Water Management 250, 106838. "
        "https://doi.org/10.1016/j.agwat.2021.106838",
        "",
        "Drusch, M., Del Bello, U., Carlier, S., Colin, O., Fernandez, V., Gascon, F., "
        "Hoersch, B., Isola, C., Laberinti, P., Martimort, P., Meygret, A., Spoto, F., "
        "Sy, O., Marchese, F., Bargellini, P., 2012. Sentinel-2: ESA's optical "
        "high-resolution mission for GMES operational services. Remote Sensing of "
        "Environment 120, 25–36. https://doi.org/10.1016/j.rse.2011.11.026",
        "",
        "Farhi, E., Goldstone, J., Gutmann, S., 2014. A quantum approximate optimization "
        "algorithm. arXiv:1411.4028. https://doi.org/10.48550/arXiv.1411.4028",
        "",
        "Fukuda, T., Jayadi, R., Nakano, Y., Kuroda, M., 1999. Application of complex tank "
        "model for evaluating performance of water operation in a reused water irrigation "
        "system. Journal of the Faculty of Agriculture, Kyushu University 44, 189–198. "
        "https://doi.org/10.5109/24320",
        "",
        "Fujii, K., Nakajima, K., 2017. Harnessing disordered-ensemble quantum dynamics "
        "for machine learning. Physical Review Applied 8, 024030. "
        "https://doi.org/10.1103/PhysRevApplied.8.024030",
        "",
        "Gautron, R., Maillard, O.-A., Preux, P., Corbeels, M., Sabbadin, R., 2022. "
        "Reinforcement learning for crop management support: review, prospects and "
        "challenges. Computers and Electronics in Agriculture 200, 107182. "
        "https://doi.org/10.1016/j.compag.2022.107182",
        "",
        "Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., "
        "Wichmann, F.A., 2020. Shortcut learning in deep neural networks. Nature Machine "
        "Intelligence 2, 665–673. https://doi.org/10.1038/s42256-020-00257-z",
        "",
        "Hatcho, N., Kurihara, K., Matsuno, Y., Horino, H., 2018. Characteristics of "
        "drainage water quality and loading from paddy field under cyclic irrigation and "
        "its management options. Journal of Water Resource and Protection 10, 73–84. "
        "https://doi.org/10.4236/jwarp.2018.101005",
        "",
        "He, Y., Wang, S., Qi, W., Gu, N., Fu, G., 2025. Characteristics of nitrogen "
        "loss through lateral seepage in paddy fields under irrigation and drainage "
        "regulation. Archives of Agronomy and Soil Science. "
        "https://doi.org/10.1080/03650340.2025.2550632",
        "",
        "Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A., "
        "Muñoz-Sabater, J., Nicolas, J., Peubey, C., Radu, R., Schepers, D., Simmons, A., "
        "Soci, C., Abdalla, S., Abellan, X., Balsamo, G., Bechtold, P., Biavati, G., "
        "Bidlot, J., Bonavita, M., De Chiara, G., Dahlgren, P., Dee, D., Diamantakis, M., "
        "Dragani, R., Flemming, J., Forbes, R., Fuentes, M., Geer, A., Haimberger, L., "
        "Healy, S., Hogan, R.J., Hólm, E., Janisková, M., Keeley, S., Laloyaux, P., "
        "Lopez, P., Lupu, C., Radnoti, G., de Rosnay, P., Rozum, I., Vamborg, F., "
        "Villaume, S., Thépaut, J.-N., 2020. The ERA5 global reanalysis. Quarterly "
        "Journal of the Royal Meteorological Society 146, 1999–2049. "
        "https://doi.org/10.1002/qj.3803",
        "",
        "Hooghoudt, S.B., 1940. Bijdragen tot de kennis van eenige natuurkundige "
        "grootheden van den grond, 7: Algemeene beschouwing van het probleem van de "
        "detailontwatering en de infiltratie door middel van parallel loopende drains, "
        "greppels, slooten en kanalen. Verslagen van Landbouwkundige Onderzoekingen "
        "46, 515–707.",
        "",
        "Iwasaki, S., Iida, T., 2024. Improvement of the drainage canal water level model "
        "for Lake Imba cyclic irrigation area. Proceedings of the 2024 Annual Conference, "
        "Japan Society of Hydrology and Water Resources, PS-2-46. "
        "https://doi.org/10.11520/jshwr.37.0_309",
        "",
        "Kallenberg, M.G.J., Overweg, H., van Bree, R., Athanasiadis, I.N., 2023. Nitrogen "
        "management with reinforcement learning and crop growth models. Environmental "
        "Data Science 2, e34. https://doi.org/10.1017/eds.2023.28",
        "",
        "Kapoor, S., Narayanan, A., 2023. Leakage and the reproducibility crisis in "
        "machine-learning-based science. Patterns 4, 100804. "
        "https://doi.org/10.1016/j.patter.2023.100804",
        "",
        "Okura, F., Kamei, A., Kato, T., 2025. Water balance analysis of hydrological "
        "processes in cyclic irrigation: a case study of the Imbanuma irrigation area "
        "in Chiba, Japan. Journal of Hydrology: Regional Studies 57, 102111. "
        "https://doi.org/10.1016/j.ejrh.2024.102111",
        "",
        "O'Malley, D., 2018. An approach to quantum-computational hydrologic inverse "
        "analysis. Scientific Reports 8, 6919. https://doi.org/10.1038/s41598-018-25206-0",
        "",
        "Playán, E., Pérez-Coveta, O., Martínez-Cob, A., Herrero, J., García-Navarro, P., "
        "Latorre, B., Brufau, P., Garcés, J., 2008. Overland water and salt flows in a set "
        "of rice paddies. Agricultural Water Management 95, 645–658. "
        "https://doi.org/10.1016/j.agwat.2008.01.012",
        "",
        "Prieto, C., Kavetski, D., Le Vine, N., Álvarez, C., Medina, R., 2021. "
        "Identification of dominant hydrological mechanisms using Bayesian inference, "
        "multiple statistical hypothesis testing, and flexible models. Water Resources "
        "Research 57, e2020WR028338. https://doi.org/10.1029/2020WR028338",
        "",
        "Prieto, C., Le Vine, N., Kavetski, D., Fenicia, F., Scheidegger, A., Vitolo, C., "
        "2022. An exploration of Bayesian identification of dominant hydrological "
        "mechanisms in ungauged catchments. Water Resources Research 58, e2021WR030705. "
        "https://doi.org/10.1029/2021WR030705",
        "",
        "Quadrianto, N., Smola, A.J., Caetano, T.S., Le, Q.V., 2009. Estimating labels "
        "from label proportions. Journal of Machine Learning Research 10, 2349–2374. "
        "https://www.jmlr.org/papers/v10/quadrianto09a.html",
        "",
        "Song, J.-H., Her, Y., Hwang, S., Kang, M.-S., 2020. Uncertainty in irrigation "
        "return flow estimation: comparing conceptual and physically-based parameterization "
        "approaches. Water 12, 1125. https://doi.org/10.3390/w12041125",
        "",
        "Tsubo, M., Fukai, S., Tuong, T.P., Ouk, M., 2007. A water balance model for "
        "rainfed lowland rice fields emphasising lateral water movement within a "
        "toposequence. Ecological Modelling 204, 503–515. "
        "https://doi.org/10.1016/j.ecolmodel.2007.02.001",
        "",
        "Turchetta, M., Corinzia, L., Sussex, S., Burton, A., Herrera, J., "
        "Athanasiadis, I.N., Buhmann, J.M., Krause, A., 2022. Learning long-term crop "
        "management strategies with CyclesGym. Advances in Neural Information Processing "
        "Systems 35 (NeurIPS 2022, Datasets and Benchmarks). "
        "https://doi.org/10.52202/068431-0828",
        "",
        "Vu, S.H., Watanabe, H., Takagi, K., 2005. Application of FAO-56 for evaluating "
        "evapotranspiration in simulation of pollutant runoff from paddy rice field in "
        "Japan. Agricultural Water Management 76, 195–210. "
        "https://doi.org/10.1016/j.agwat.2005.01.012",
        "",
        "Xu, J., Lv, Y., Ai, L., Yang, S., He, Y., Dalson, T., 2017. Validation of "
        "dual-crop coefficient method for calculation of rice evapotranspiration under "
        "drying–wetting cycle condition. Paddy and Water Environment 15, 381–393. "
        "https://doi.org/10.1007/s10333-016-0556-9",
        "",
        "Zhang, Y., Charoenphakdee, N., Wu, Z., Sugiyama, M., 2020. Learning from "
        "aggregate observations. Advances in Neural Information Processing Systems 33 "
        "(NeurIPS 2020). https://doi.org/10.48550/arXiv.2004.06316",
        "",
        "Zippenfenig, P., 2024. Open-Meteo.com Weather API (version 1.4.0). Zenodo. "
        "https://doi.org/10.5281/zenodo.14582479",
        "",
    ])


def _routing_section(n):
    """Recapture delivery: same-day collection clipped by infiltration, not hops."""
    lines = [
        "### 5.2 Recapture is collection, clipped by the pan",
        "",
        "Capture \(K_x+K_y\) is the leftover collection ratio, pinned to Table 5's "
        "\(LFV_{pump}/LfA\). That share of leftover is counted as LFV the same day if "
        "the pan has already sent it underground. Pond leftover cannot be LFV. "
        "`python -m cyclic routing` runs the closed-form vector at a range of leftover "
        "speeds and reports the share delivered:",
        "",
    ]
    if n.routing_txt:
        lines += ["```", n.routing_txt, "```", ""]
    else:
        lines += ["No `routing_{year}.txt` is present; run `python -m cyclic routing`.", ""]
    lines += [
        "Leftover speed does not move delivered LFV: hops are bund seepage of the "
        "infiltrated remainder, not the LFV path. Recapture is uniform on cyclic "
        "cells, so leftover-weighted capture matches the area-weighted pin. The "
        "remaining gap to Table 5 is leftover volume, the daily pan clip "
        "(`lfv_shortfall`), and recapture taken by unsaturated neighbours. Seasonal "
        "infiltration can still exceed Table 5 while daily coincidences of leftover "
        "and infiltration do not. That is a Beven limit of "
        f"acceptability (`identify.LFV_ACCEPT_FRAC` = {n.lfv_accept_frac:.2f} of Table 5 "
        "LFV_pump), not a licence to fit `plow_pan_k`. `plow_pan_k` stays a Physics "
        "rate, like leftover speed, and is not in `identify.ACTIVE`.",
        "",
        "This is the §6 split, implemented: collection (enter the reuse network at the "
        "plot) is the simplex share; conveyance to the pool is same-day at daily "
        "resolution, so it is not a fifth leftover ratio. Farm drains are not mapped.",
        "",
    ]
    return "\n".join(lines)


def _fitted_section(n):
    if n.fitted is None:
        return (
            f"### 5.3 Fitted district vector ({n.year})\n\n"
            "No `best_{year}_season.npz` is on disk. Closed-form Table 4–5 shares are "
            "the start of `solve_classical`.\n"
        )
    f = n.fitted
    return "\n".join([
        f"### 5.3 Fitted district vector ({n.year})",
        "",
        f"Saved random-walk result `{f['path']}`, scalar loss {f['loss']:.4f} under the "
        "objectives in §4.5, fitted in-sample on this year. Loaded parameters stamp "
        f"`force_satellite_et={f['force_et']}`:",
        "",
        f"- Kx = {f['kx']:.4f}, Ky = {f['ky']:.4f} — **reported, not identified**: the "
        "routing sees only the sum (§3.2), so this split carries no information about "
        "drain orientation",
        f"- Kz = {f['kz']:.4f}, ground = {f['ground']:.4f}",
        f"- farm-to-main = {f['farm_to_main']:.4f}, against a Table 5 / Table 4 target of "
        f"{n.farm_to_main_raw:.4f}. The gap is not a measurement: `farm_to_main` "
        "multiplies simulated drainage into the pool, so it absorbs the frozen "
        "forecaster's drainage bias. Freezing P1 stops leftover identification from "
        "retuning irrigation, but it does not decouple this parameter from P1 error.",
        f"- capture = {f['capture']:.4f}, against the Table 5 pin of {n.capture:.4f}. The "
        "`darcy` penalty on that gap is "
        f"{(f['capture'] - n.capture) ** 2:.4f} against a loss of {f['loss']:.4f}. "
        "This checkpoint may predate same-day recapture; re-run "
        "`python -m cyclic identify` after a physics change before quoting the loss.",
        "",
        "This file is a checkpoint, not an independent field campaign, and its loss is "
        "in-sample. Re-run `python -m cyclic identify` after a physics change before "
        "quoting the loss against a new code path.",
        "",
    ])


def _qrc_section(n):
    lines = [
        f"### 5.4 Quantum reservoir readout ({n.year})",
        "",
        "Training is outside the UI. The text below is copied from the last "
        "`qrc_*_equation.txt` written by `cyclic.qrc`, not re-derived in prose.",
        "",
    ]
    if n.qrc_txt:
        lines += ["**District daily QRC**", "", "```", n.qrc_txt, "```", ""]
    else:
        lines += ["No district QRC equation file is present.", ""]
    if n.qrc_spatial_txt:
        lines += ["**Spatial QRC**", "", "```", n.qrc_spatial_txt, "```", ""]
    else:
        lines += ["No spatial QRC equation file is present.", ""]
    lines += [
        "The spatial readout **collapsed to its own target**, and we state that as a "
        "result rather than a caveat. No coefficient in b exceeds 1e-17 in magnitude, the "
        "reported R² is −0.000, and the printed rule reduces to "
        "`stay = stay_target`. So the spatial equation above is a restatement of the "
        "Table 5 stay share, not a learned spatial field. The mechanism is leakage: "
        "`leftover_weighted_stay` rescales any P2 output so its leftover-weighted mean "
        "matches Table 5, which means a readout of exactly zero already satisfies the "
        "target the loss is checking. That rescale applies to every registered ratio "
        "finder, so `lateral` and `darcy` are partly insensitive to which finder produced "
        "the field. The district daily readout is better posed — its stay regression has "
        "a genuine if weak R², and it reports a held-out year — but it is fitted against "
        "a two-box filling target rather than an observation, so it should be read as "
        "internal consistency, not skill.",
        "",
    ]
    return "\n".join(lines)


def write_manuscript(year=okura.COMPARISON_YEAR, folder=None, write_repo=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_folder = os.path.join(root, "outputs", "paper")
    folder = folder or default_folder
    os.makedirs(folder, exist_ok=True)
    text = manuscript_markdown(year)
    methods_path = os.path.join(folder, "manuscript.md")
    with open(methods_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    repo_path = os.path.join(root, "PAPER.md")
    if write_repo is None:
        write_repo = os.path.abspath(folder) == os.path.abspath(default_folder)
    if write_repo:
        with open(repo_path, "w", encoding="utf-8") as handle:
            handle.write(text)
    return SimpleNamespace(manuscript=methods_path, repo=repo_path if write_repo else None,
                           numbers=live_numbers(year))

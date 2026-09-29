"""Paper methods: **inferred underground flow ratios** under cyclic irrigation.

Contribution (see ``cyclic.underground``)
    With surface IR/D known, infer per-point leftover capture (Kx+Ky → LFV) vs stay
    (Kz+ground → deep). District mean capture = Table 5 LFV_pump/LfA. Freeze across
    years. Surface Fig. 8 timing is forcing, not novelty.

This module is the methods text a manuscript can cite. Numbers come from
cyclic.okura (tables) and cyclic.identify / cyclic.underground (leftover shares).
Run:

    python -m cyclic paper [year]

That writes outputs/paper/: METHODS.md, CONTRIBUTION.md, table checks, leftover
shares, a routing figure.

Citations (full manuscript in cyclic.manuscript / PAPER.md)
    Okura, F., Kamei, A. and Kato, T. (2025). Journal of Hydrology: Regional Studies
    57, 102111. https://doi.org/10.1016/j.ejrh.2024.102111
    Allen et al. (1998). FAO Irrigation and Drainage Paper 56 (dual Kc).
    O'Malley, D. (2018). Sci. Rep. 8:6919 (binary hydrologic inverse / QUBO).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

from . import okura
from .identify import construct_ratios, load_best_ratios
from .physics import Physics


SITE = (
    "Shiroyama-Jimbee district, Imbanuma Land Improvement District, Chiba, Japan "
    "(Lake Kita Imbanuma). Cyclic irrigation season 17 April–25 August (131 days), "
    "years 2021 (wet) and 2022 (dry)."
)


def model_equations():
    """Numbered identities used in the code (Okura Eq. 1–14 plus model M1–M5)."""
    return [
        ("Okura 1", "District storage",
         r"dWB = W - O + (R - ET) A_cyc / 1000 - IR_ncb"),
        ("Okura 3", "Block leftover (Lf), not measured",
         r"L = IR + R - D - ET"),
        ("Okura 4", "Pumping-station pool",
         r"Δpool = W - O + LFV_pump + D_N + D_S - IR_all"),
        ("Okura 6", "LFV from the pool residual",
         r"LFV_pump = IR_all - W + O - D_N - D_S"),
        ("Okura 7", "Block storage after LFV leaves",
         r"dWB = LfA_N + LfA_S - LFV_pump"),
        ("Okura 11–14", "Efficiencies",
         r"WUE' = ET/(IR-D+ER);  RUE = D/IR_all;  LUE = LFV_pump/IR_all"),
        ("M1", "Leftover simplex on each plot",
         r"L = (K_x + K_y + K_z + K_g) L,  with K_x+K_y+K_z+K_g = 1"),
        ("M2", "Stay vs recapture (collection ratio)",
         r"stay = K_z + K_g,  recapture = K_x + K_y = LFV_pump / LfA  (Table 5/4)"),
        ("M3", "Leftover hops equalise neighbour head; not Darcy",
         r"H sends infiltrated remainder only wetter→drier, at most equalising. Recapture does not hop."),
        ("M4", "Same-day recapture from infiltrated leftover",
         r"LFV = min((K_x+K_y) L^+, I); pond leftover cannot be LFV"),
        ("M5", "Optional QUBO (O'Malley)",
         r"x_i=1 send leftover into the graph, x_i=0 stay; H = data misfit + J(x_i-x_j)^2"),
        ("M6", "Neighbour-head lateral equation (no soil K)",
         r"One H per paddy: ΔL_i = H(h_i, h_up, h_down, h_left, h_right, ET of each); linear and nonlinear are forms, not two models"),
    ]


def closed_form_shares(year=okura.COMPARISON_YEAR):
    """Table 4–5 leftover partition (district mean), before any search."""
    return construct_ratios(year)


def fitted_shares(year=okura.COMPARISON_YEAR, span="season"):
    saved = load_best_ratios(year, span)
    if saved is None:
        return None
    return saved.params, saved.loss, saved.path


def table_checks(year):
    frame = okura.check_tables(year)
    # North Table 7 WUE' cannot be reproduced from Table 4 under the MAFF rain rule
    # (documented in okura.py). Exclude that known incompatibility from "paper ready".
    skip = frame["check"].str.contains("Table 7 WUE'|Prose irrigation")
    core = frame.loc[~skip]
    return SimpleNamespace(all=frame, core=core, core_ok=bool(core["ok"].all()))


def methods_markdown(year=okura.COMPARISON_YEAR):
    shares = closed_form_shares(year)
    fitted = fitted_shares(year)
    checks = table_checks(year)
    p = Physics()
    lines = [
        f"# Methods — inferred underground flow ratios ({okura.CITATION})",
        "",
        "## Contribution",
        "",
        "**Inferred underground flow ratios.** Surface IR/D (Fig. 8 / Okura pumps) are "
        "known forcing. After FAO-56 ET, leftover Lf (Okura Eq. 3) is partitioned per "
        "point into capture \(K_x+K_y\) (same-day LFV) and stay \(K_z+K_g\) (deep/dWB). "
        "District mean capture = Table 5 LFV_pump/LfA. Spatial structure: nearer the "
        "main drain → higher capture; Table 4 / Fig. 6 Lf intensity scales North vs "
        "South. Protocol: `infer_underground_ratios`, `underground_ratio_holdout`. "
        "Not claimed: measured subsurface paths; unique \(K_x\) vs \(K_y\); novel IR timing.",
        "",
        "## Site and period",
        "",
        SITE,
        "",
        "Observed irrigation and surface drainage are pump records and ADCPs in the main "
        "drainage canal (Okura §2). Daily IR/D from that campaign are the intended "
        "forecaster labels; the loader currently interpolates published tables until "
        "those series are ingested. Lateral flow Lf is **not** observed; it is the Eq. 3 "
        "residual. Simulated ET is FAO-56 dual Kc on local water; satellite Kc×ET0 is a "
        "loss term, not the flux (`force_satellite_et=False`). "
        "The main drainage canal stretches alongside Lake Kita Imbanuma, along the edge of "
        "the North and South blocks; drainage converges at the pumping-station pool; "
        "LFV_pump is same-day recapture of infiltrated leftover (collection ratio "
        "Kx+Ky, **per cyclic cell**, area-weighted mean = LFV_pump/LfA — nearer the "
        "main drain recaptures more; the main-drain polyline is not an LFV hop target), "
        "not hop-then-lining-catch. **Novelty is this underground map** (per-point "
        "inferred underground flow ratios Kx, Ky, Kz, ground under year freeze). "
        "Surface IR/W may be taken from digitised Fig. 8 as known forcing.",
        "",
        "## Governing identities (Okura)",
        "",
    ]
    for tag, title, eq in model_equations()[:6]:
        lines.append(f"- **{tag}** ({title}): `{eq}`")
    lines.extend([
        "",
        "## Plot-scale model (this work)",
        "",
        "FAO-56 dual crop coefficient (Allen et al. 1998) splits evaporation (pond or "
        "TEW surface) from transpiration (TAW root zone). Simulated ET is scored against "
        "the satellite Kc×ET0 map; that map is not imposed as the flux. One next-day "
        "water-fate predictor registers outside the package: `@register_forecaster` is "
        "the **farmer model** (per-paddy applied irrigation as a split of the Okura "
        "block total, plus surface discharge); leftover collection shares "
        "(Kx, Ky, Kz, ground) are coefficients of the per-paddy equation builder "
        "`@register_lateral` (`partition`, the **equation builder**, without soil properties); optional `@register_ratio` can override "
        "those shares. Linear and nonlinear are forms of the same H, "
        "not two hydrology models; physics baseline `head_equalise`. `CyclicEnv(given_irrigation=True)` is the gym. Leftover \(K_i\) are "
        "fitted on one year and frozen on the other (`ratio_holdout`). Per-paddy "
        "applied IR share is unknown; hops may see neighbour leftover head and "
        "neighbour FAO-56 ET on the up, down, left and right plots of every paddy "
        "(`neighbour_up_mm` … `neighbour_right_et_mm`), still equalise head rather than "
        "sending leftover toward high ET, and the same law is scored on historical "
        "2021/2022 even if farmers irrigate differently (`soil_flow_holdout`). After IR, R, D "
        "and ET, leftover L is partitioned once (M1). Kz and ground are **stay** in the "
        "plot (dWB), not a second outlet. Kx+Ky is the **collection / recapture** share: "
        "same-day LFV from leftover that has already gone through the pan "
        "(pond leftover cannot be LFV). Unsaturated neighbours can take some of that "
        "recapture into TEW/TAW so those plots can still meet later FAO-56 ET; that "
        "intercept is a store capacity, not a leftover "
        "ratio. Only the sum is used, so the axis split is "
        "reported, not identified. A hop moves infiltrated remainder only toward a drier "
        "neighbour, at most until the two equalise; hops do not catch at the lining as "
        "LFV. Mean hops in a day are leftover speed over spacing (μ = vT/Δx), a kinematic "
        "count, not a fifth leftover ratio. `canal_hops` is not searched (Song et al. "
        "2020: constrain drainage parameters or return-flow uncertainty explodes). Dupuit "
        "conductivity leftover is off because the district is flat (Tsubo et al. 2007). "
        "Hops equalise leftover head rather than following satellite ET, at bund-seepage "
        "scale (He et al. 2025: ~150 mm bund window; leftover speed 2 m d⁻¹). Simulated "
        "Ke uses Xu et al. (2017) wetting-pulse Kcp = 1.29 on rain or irrigation days. "
        "Pond→soil is `plow_pan_k_mm_d` × (pond / `pond_infiltrate_ref_mm`), capped at "
        "the pond — a Physics rate, not a leftover ratio, and not gated by TEW "
        "saturation. Perched Darcy/Hooghoudt runs only when "
        "`ratio_split` is off.",
        "",
    ])
    for tag, title, eq in model_equations()[6:]:
        lines.append(f"- **{tag}** ({title}): `{eq}`")
    lines.extend([
        "",
        f"## Closed-form shares from Tables 4–5 ({year})",
        "",
        f"- recapture Kx+Ky = LFV_pump / LfA = **{shares['pool_capture_share']:.4f}** "
        "(same-day collection from infiltrated leftover)",
        f"- stay = **{1.0 - shares['pool_capture_share']:.4f}** "
        f"(Kz = {shares['kz_ratio']:.4f}, ground = {shares['ground_ratio']:.4f}; "
        f"split TEW/(TEW+TAW) = {p.total_evaporable_mm:.0f}/"
        f"{p.total_evaporable_mm + p.total_available_mm:.0f})",
        f"- farm-to-main = Table 5 D / Table 4 D = **{shares['farm_to_main']:.4f}**",
        f"- table start Kx = Ky = half of capture = **{shares['kx_ratio']:.4f}**",
        "",
    ])
    if fitted is not None:
        params, loss, path = fitted
        lines.extend([
            f"## Fitted shares on recapture routing ({year} season)",
            "",
            f"Loss {loss:.4f}. File `{os.path.basename(path)}`.",
            "",
            f"- Kx = {params['kx_ratio']:.4f}, Ky = {params['ky_ratio']:.4f} "
            f"(reported only: routing sees Kx+Ky, so the axis split is not identified)",
            f"- Kz = {params['kz_ratio']:.4f}, ground = {params['ground_ratio']:.4f}",
            f"- farm-to-main = {params['farm_to_main']:.4f}",
            f"- capture = {params.get('pool_capture_share', params['kx_ratio']+params['ky_ratio']):.4f}",
            "",
        ])
    n_fail = int((~checks.core["ok"]).sum())
    lines.extend([
        "## Table self-consistency",
        "",
        f"Okura Tables 3–6 identities for {year}: "
        f"{'all core checks pass' if checks.core_ok else f'{n_fail} core check(s) fail'}. "
        "Table 7 North-block season WUE' is known not to close against Table 4 under the "
        "MAFF effective-rain rule (Okura code comments); monthly shape is retained, "
        "scaled to Table 4.",
        "",
        "## Reproducibility",
        "",
        "- Season length: 131 days; comparison year default 2022.",
        "- Identification: IR/D from the best registered forecaster (frozen); leftover "
        "shares searched. Satellite ET scored, not imposed.",
        "- Graph routing: `Physics.use_graph=True`, `one_way_to_canal=False` "
        "(bund seepage, not a canal diode).",
        "- Optional QUBO: `Physics.as_qubo` or `python -m cyclic identify qubo-graph` "
        "(O'Malley clusters, not 4,925 paddies).",
        "- Command: `python -m cyclic paper 2022` (METHODS.md and PAPER.md).",
        "",
    ])
    return "\n".join(lines)


def routing_figure(path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (0.3, 1.6, "Plot\nL = IR+R−D−ET"),
        (3.3, 1.6, "Stay\nKz, ground"),
        (3.3, 3.2, "Recapture\nKx+Ky, if infiltrated"),
        (6.5, 3.2, "Bund hops\ninfiltrated remainder"),
        (9.3, 3.2, "Pool\nLFV + D"),
        (9.3, 1.2, "Lake\nW, O"),
    ]
    for x, y, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 2.2, 1.2, boxstyle="round,pad=0.04",
                                    facecolor="#f4f1ea", edgecolor="#333", lw=1.0))
        ax.text(x + 1.1, y + 0.6, text, ha="center", va="center", fontsize=8)
    arrows = [
        ((2.5, 2.2), (3.3, 2.2)),
        ((2.5, 2.4), (3.3, 3.6)),
        ((5.5, 3.8), (6.5, 3.8)),
        ((5.5, 3.5), (9.3, 3.8)),
        ((10.4, 3.2), (10.4, 2.4)),
        ((8.7, 3.5), (9.3, 2.2)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                     color="#333", lw=1.1))
    ax.set_title("Leftover routing (M1–M4). Recapture is same-day LFV; hops are bund seepage.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_paper_pack(year=okura.COMPARISON_YEAR, folder=None):
    """Write METHODS.md, PAPER.md/manuscript.md, CSV checks/shares, routing schematic."""
    import pandas as pd

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = folder or os.path.join(root, "outputs", "paper")
    os.makedirs(folder, exist_ok=True)
    methods_path = os.path.join(folder, "METHODS.md")
    with open(methods_path, "w", encoding="utf-8") as handle:
        handle.write(methods_markdown(year))
    checks = table_checks(year)
    checks.all.to_csv(os.path.join(folder, f"okura_table_checks_{year}.csv"), index=False)
    shares = closed_form_shares(year)
    rows = [{"source": "tables_4_5", **{k: v for k, v in shares.items() if k in (
        "kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio", "farm_to_main", "pool_capture_share")}}]
    fitted = fitted_shares(year)
    if fitted is not None:
        params, loss, _ = fitted
        row = {"source": "fitted_season", "loss": loss}
        row.update({k: params[k] for k in (
            "kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio", "farm_to_main") if k in params})
        row["pool_capture_share"] = params.get(
            "pool_capture_share", params.get("kx_ratio", 0) + params.get("ky_ratio", 0))
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(folder, f"leftover_shares_{year}.csv"), index=False)
    fig_path = routing_figure(os.path.join(folder, "leftover_routing.png"))
    from .underground import write_contribution
    contribution_path = write_contribution(folder=folder, year=year)
    # Copy methods next to the package for the manuscript working folder.
    repo_methods = os.path.join(root, "METHODS.md")
    with open(repo_methods, "w", encoding="utf-8") as handle:
        handle.write(methods_markdown(year))
    from .manuscript import write_manuscript
    paper = write_manuscript(year, folder=folder)
    return SimpleNamespace(
        folder=folder, methods=methods_path, repo_methods=repo_methods,
        figure=fig_path, contribution=contribution_path, core_ok=checks.core_ok, year=year,
        manuscript=paper.manuscript, repo_paper=paper.repo,
    )


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else list(argv)
    year = int(argv[0]) if argv else okura.COMPARISON_YEAR
    out = write_paper_pack(year)
    print("year", out.year)
    print("core table checks", "OK" if out.core_ok else "FAIL")
    print("wrote", out.methods)
    print("wrote", out.contribution)
    print("wrote", out.repo_methods)
    print("wrote", out.manuscript)
    if out.repo_paper:
        print("wrote", out.repo_paper)
    print("wrote", out.figure)
    return out

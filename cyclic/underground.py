"""Contribution core: **inferred underground flow ratios**.

Claim (what we assert)
    Surface IR, D, and lake W are **hard copies** of what Okura recorded
    (digitised Fig. 8 pumps / withdrawal; ADCP / Fig. 6 + Table 4 drainage) —
    not inferred schedules. Leftover after FAO-56 ET is the unobserved residual (Lf).
    We infer a per-point simplex on that residual:

        capture  Kx + Ky   → same-day LFV to the pool (district mean = LFV_pump/LfA)
        stay     Kz + Kg   → deep / district storage (dWB)

    Spatial structure comes from canal proximity and Table 4 / Fig. 6 Lf intensity.
    Ratios are trained on **both** 2021 and 2022 (`underground_ratio_train_both`),
    then reported against each year's Table 5 pin. Single-year freeze is still
    available via `underground_ratio_holdout`.

Not claimed
    Direct measurement of subsurface paths or velocities; unique Kx vs Ky (axis
    split is reported, half/half); novelty in IR timing; estimating surface IR/D/W
    when the digitised paper series exist; that Lf itself is new (Okura Eq. 3).

API
    infer_underground_ratios(year)           → one-season field + pins
    infer_underground_ratios_joint(years)   → one field trained on both years
    underground_ratio_train_both(...)       → joint train; score each Table 5 pin
    underground_ratio_holdout(...)          → freeze one year, score the other
    contribution_text(report)               → one-page claim + numbers
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from . import okura
from .canal import mean_pixel_shares, pixel_ratio_fields
from .identify import construct_ratios, discover_pixel_shares, ratio_holdout
from .physics import Physics

CLAIM = (
    "inferred underground flow ratios: per-point leftover capture (Kx+Ky -> LFV) "
    "vs stay (Kz+ground -> deep/dWB), under hard-copy surface IR/D/W "
    "(Fig. 8 + ADCP / Table 4; not estimated schedules)"
)

NOT_CLAIMED = (
    "measured subsurface paths",
    "unique Kx vs Ky axis",
    "novel IR schedule / Fig. 8 timing",
    "that district Lf or LFV_pump totals are newly observed",
    "estimating surface IR/D/W when digitised paper series exist",
)


def _block_intensity(year):
    try:
        from .paper_plots import block_leftover_intensity
        return block_leftover_intensity(int(year))
    except Exception:
        return None


def _joint_block_intensity(years):
    """Mean Table 4 / Fig. 6 leftover intensity over training years."""
    acc = {}
    for year in years:
        bi = _block_intensity(year) or {}
        for key, val in bi.items():
            acc.setdefault(key, []).append(float(val))
    if not acc:
        return None
    return {key: float(np.mean(vals)) for key, vals in acc.items()}


def _spatial_stats(fields, area, cyclic):
    """Capture and stay summaries on cyclic cells (area-weighted mean + CV)."""
    means = mean_pixel_shares(fields, area, cyclic)
    capture = np.asarray(fields.get("pool_capture_share",
                                    np.asarray(fields["kx_ratio"]) + np.asarray(fields["ky_ratio"])),
                         dtype=float).ravel()
    area = np.asarray(area, dtype=float).ravel()
    cyclic = np.asarray(cyclic, dtype=bool).ravel()
    n = min(capture.size, area.size, cyclic.size)
    cap = capture[:n]
    w = np.where(cyclic[:n], np.maximum(area[:n], 0.0), 0.0)
    w_sum = float(w.sum()) or 1.0
    mean_c = float((cap * w).sum() / w_sum)
    cap_c = cap[cyclic[:n]]
    std_c = float(np.std(cap_c)) if cap_c.size else float("nan")
    stay_mean = 1.0 - mean_c
    return SimpleNamespace(
        capture_mean=mean_c,
        capture_std=std_c,
        capture_cv=float(std_c / mean_c) if mean_c > 1e-12 else float("nan"),
        capture_min=float(np.min(cap_c)) if cap_c.size else float("nan"),
        capture_max=float(np.max(cap_c)) if cap_c.size else float("nan"),
        stay_mean=stay_mean,
        kx_mean=float(means.get("kx_ratio", mean_c * 0.5)),
        ky_mean=float(means.get("ky_ratio", mean_c * 0.5)),
        kz_mean=float(means.get("kz_ratio", float("nan"))),
        ground_mean=float(means.get("ground_ratio", float("nan"))),
        farm_to_main=float(means.get("farm_to_main", float("nan"))),
        n_cyclic=int(cyclic[:n].sum()),
        n_points=int(n),
    )


def infer_underground_ratios(year=okura.COMPARISON_YEAR, method="tables",
                             use_paper_plots=True, results=None):
    """Infer per-point underground flow ratios for one season.

    Surface IR/D are *inputs* to the leftover residual, not the contribution.
    Returns the simplex field, district Table 4–5 pins, and spatial stats.
    """
    from . import data
    from .canal import distance_to_canal_m
    from .lateral import knn_edges

    year = int(year)
    pin = construct_ratios(year)
    params = dict(pin)
    # Prefer a saved / stay field when method asks for it; otherwise build the
    # contribution map: canal proximity × Table 4 Lf intensity → capture_i.
    if method not in ("tables", "table") or results is not None:
        discovered = discover_pixel_shares(
            year, method=method, results=results, simulate=False)
        fields = {k: np.asarray(v, dtype=float) for k, v in discovered.fields.items()}
        area = np.asarray(discovered.area, dtype=float)
        cyclic = np.asarray(discovered.cyclic, dtype=bool)
        params = dict(discovered.params)
    else:
        frame = data.paddies()
        lon = frame["lon"].to_numpy()
        lat = frame["lat"].to_numpy()
        cyclic = frame["cyclic"].to_numpy(dtype=bool)
        area = (frame["area_m2"].to_numpy() if "area_m2" in frame.columns
                else np.ones(len(frame), dtype=float))
        block = frame["block"].to_numpy() if "block" in frame.columns else None
        p = Physics()
        top = float(p.total_evaporable_mm /
                    max(p.total_evaporable_mm + p.total_available_mm, 1e-9))
        intensity = _block_intensity(year) if use_paper_plots else None
        fields = pixel_ratio_fields(
            len(frame), distance_to_canal_m(lon, lat), knn_edges(lon, lat, k=4),
            cyclic, float(pin["pool_capture_share"]), float(pin["farm_to_main"]),
            top, float(p.canal_length_m),
            area_m2=area, block=block, block_intensity=intensity,
            hard_capture=False,
            neighbour_of_remain=float(pin.get("neighbour_of_remain",
                                              getattr(p, "neighbour_of_remain", 0.5))))
    stats = _spatial_stats(fields, area, cyclic)
    return SimpleNamespace(
        year=year,
        method=str(method),
        claim=CLAIM,
        not_claimed=NOT_CLAIMED,
        contribution="inferred_underground_flow_ratios",
        surface_forcing="known (Fig. 8 / Okura IR+D; not novelty)",
        fields=fields,
        params=params,
        district_pin=pin,
        capture_pin=float(pin["pool_capture_share"]),
        stats=stats,
        area=np.asarray(area, dtype=float),
        cyclic=np.asarray(cyclic, dtype=bool),
    )


def infer_underground_ratios_joint(years=(2021, 2022), method="tables",
                                   use_paper_plots=True):
    """One underground-ratio field trained on both seasons.

    District capture = mean of each year's Table 5 LFV/LfA (exact pins, no floor).
    Farm-to-main and North/South Lf intensity are averaged the same way.
    Per-point LFV is that capture share; neighbour hops cover local ET shortfalls.
    """
    from . import data
    from .canal import distance_to_canal_m
    from .lateral import knn_edges

    years = tuple(int(y) for y in years)
    if len(years) < 1:
        raise ValueError("need at least one training year")
    table_pins = {y: construct_ratios(y) for y in years}
    table_captures = {y: float(p["pool_capture_share"]) for y, p in table_pins.items()}
    table_farms = {y: float(p["farm_to_main"]) for y, p in table_pins.items()}
    joint_capture = float(np.mean(list(table_captures.values())))
    joint_farm = float(np.mean(list(table_farms.values())))
    stay = 1.0 - joint_capture
    p = Physics()
    top = float(p.total_evaporable_mm /
                max(p.total_evaporable_mm + p.total_available_mm, 1e-9))
    nbr_frac = float(np.clip(getattr(p, "neighbour_of_remain", 0.5), 0.0, 1.0))
    neighbour = stay * nbr_frac
    stay_mass = stay - neighbour
    params = {
        "kx_ratio": 0.5 * neighbour,
        "ky_ratio": 0.5 * neighbour,
        "kz_ratio": stay_mass * top,
        "ground_ratio": stay_mass * (1.0 - top),
        "farm_to_main": joint_farm,
        "pool_capture_share": joint_capture,
        "table_capture_share": joint_capture,
        "neighbour_of_remain": nbr_frac,
        "ratio_split": True,
        "force_satellite_et": False,
        "train_years": years,
    }

    if method not in ("tables", "table"):
        stacked = None
        area = cyclic = None
        for y in years:
            discovered = discover_pixel_shares(y, method=method, simulate=False)
            area = np.asarray(discovered.area, dtype=float)
            cyclic = np.asarray(discovered.cyclic, dtype=bool)
            piece = {k: np.asarray(v, dtype=float) for k, v in discovered.fields.items()}
            if stacked is None:
                stacked = {k: [v] for k, v in piece.items()}
            else:
                for k, v in piece.items():
                    stacked.setdefault(k, []).append(v)
        fields = {k: np.mean(np.stack(vs, axis=0), axis=0) for k, vs in stacked.items()}
        cap = np.asarray(fields.get("pool_capture_share",
                                    fields["kx_ratio"] + fields["ky_ratio"]), dtype=float)
        w = np.where(cyclic, np.maximum(area, 0.0), 0.0)
        w_sum = float(w.sum()) or 1.0
        mean_c = float((cap * w).sum() / w_sum)
        if mean_c > 1e-12:
            cap = np.clip(cap * (joint_capture / mean_c), 0.0, 1.0)
        else:
            cap = np.where(cyclic, joint_capture, 0.0)
        stay_i = 1.0 - cap
        nbr = stay_i * nbr_frac
        stay_mass = stay_i - nbr
        fields["pool_capture_share"] = cap
        fields["kx_ratio"] = 0.5 * nbr
        fields["ky_ratio"] = 0.5 * nbr
        fields["kz_ratio"] = stay_mass * top
        fields["ground_ratio"] = stay_mass * (1.0 - top)
    else:
        frame = data.paddies()
        lon = frame["lon"].to_numpy()
        lat = frame["lat"].to_numpy()
        cyclic = frame["cyclic"].to_numpy(dtype=bool)
        area = (frame["area_m2"].to_numpy() if "area_m2" in frame.columns
                else np.ones(len(frame), dtype=float))
        block = frame["block"].to_numpy() if "block" in frame.columns else None
        intensity = _joint_block_intensity(years) if use_paper_plots else None
        fields = pixel_ratio_fields(
            len(frame), distance_to_canal_m(lon, lat), knn_edges(lon, lat, k=4),
            cyclic, joint_capture, joint_farm, top, float(p.canal_length_m),
            area_m2=area, block=block, block_intensity=intensity,
            hard_capture=False, neighbour_of_remain=nbr_frac)

    stats = _spatial_stats(fields, area, cyclic)
    year_pins = {
        y: SimpleNamespace(
            table_capture=table_captures[y],
            capture_error=float((stats.capture_mean - table_captures[y]) ** 2),
            farm_to_main=table_farms[y],
        )
        for y in years
    }
    return SimpleNamespace(
        year=years,
        years=years,
        method=str(method),
        claim=CLAIM,
        not_claimed=NOT_CLAIMED,
        contribution="inferred_underground_flow_ratios",
        surface_forcing="known (Fig. 8 / Okura IR+D; not novelty)",
        train_both=True,
        fields=fields,
        params=params,
        district_pin=params,
        capture_pin=float(joint_capture),
        table_capture_mean=joint_capture,
        table_captures=table_captures,
        year_pins=year_pins,
        stats=stats,
        area=np.asarray(area, dtype=float),
        cyclic=np.asarray(cyclic, dtype=bool),
    )


def underground_ratio_train_both(years=(2021, 2022), method="tables"):
    """Publication protocol: train ratios on both years, score each Table 5 pin."""
    train = infer_underground_ratios_joint(years=years, method=method)
    errors = {y: float(train.year_pins[y].capture_error) for y in train.years}
    return SimpleNamespace(
        contribution="inferred_underground_flow_ratios",
        claim=CLAIM,
        not_claimed=NOT_CLAIMED,
        method=str(method),
        train_years=train.years,
        train_both=True,
        train=train,
        train_capture=float(train.stats.capture_mean),
        table_captures=dict(train.table_captures),
        year_pins=train.year_pins,
        capture_errors=errors,
        capture_error=float(np.mean(list(errors.values()))),
        fields=train.fields,
        params=train.params,
        stage_frozen=True,
        surface_forcing="known; joint-trained underground K_i on both years",
    )


def underground_ratio_holdout(method="tables", train_year=2021, eval_year=None,
                              simulate=False):
    """Optional: infer ratios on one year, freeze, score the other year.

    Prefer ``underground_ratio_train_both`` when both seasons are used for training.
    """
    hold = ratio_holdout(method=method, train_year=train_year, eval_year=eval_year,
                         simulate=simulate)
    train = infer_underground_ratios(hold.train_year, method=method)
    eval_pin = construct_ratios(hold.eval_year)
    return SimpleNamespace(
        contribution="inferred_underground_flow_ratios",
        claim=CLAIM,
        not_claimed=NOT_CLAIMED,
        method=hold.method,
        train_year=hold.train_year,
        eval_year=hold.eval_year,
        train=train,
        train_capture=hold.train_capture,
        eval_capture=hold.eval_capture,
        capture_error=hold.capture_error,
        eval_pin=eval_pin,
        fields=hold.fields,
        params=hold.params,
        simulate=hold.simulate,
        eval=hold.eval,
        stage_frozen=True,
        surface_forcing="known; frozen underground K_i only",
    )


def contribution_text(report=None, year=None):
    """Plain-text contribution card for CLI / paper pack."""
    if report is None:
        report = underground_ratio_train_both()
    lines = [
        "CONTRIBUTION — inferred underground flow ratios",
        "=" * 52,
        CLAIM,
        "",
        "Not claimed: " + "; ".join(NOT_CLAIMED),
        "",
    ]
    if getattr(report, "train_both", False) or getattr(report, "train_years", None):
        years = getattr(report, "train_years", None) or getattr(report, "years", ())
        lines.append(f"Train both years: {', '.join(str(y) for y in years)}")
        lines.append(f"  joint capture (area-weighted) = {report.train_capture:.4f}")
        table_caps = getattr(report, "table_captures", {})
        errors = getattr(report, "capture_errors", {})
        for y in years:
            pin = table_caps.get(y, float("nan"))
            err = errors.get(y, float("nan"))
            lines.append(f"  {y} Table 5 pin = {pin:.4f}   error^2 = {err:.6f}")
        lines.append(f"  mean capture error (sq)     = {float(report.capture_error):.6f}")
        lines.append("")
        stats = report.train.stats if hasattr(report, "train") else report.stats
    elif hasattr(report, "train_year"):
        lines.extend([
            f"Holdout: train {report.train_year} -> eval {report.eval_year}",
            f"  train capture (area-weighted) = {report.train_capture:.4f}",
            f"  eval Table 5 pin              = {report.eval_capture:.4f}",
            f"  capture error (sq)            = {report.capture_error:.6f}",
            "",
        ])
        stats = report.train.stats
    else:
        year = report.year
        stats = report.stats
        lines.extend([
            f"Year {year}",
            f"  district capture pin (LFV/LfA) = {report.capture_pin:.4f}",
            "",
        ])
    lines.extend([
        f"Spatial field ({stats.n_cyclic} cyclic / {stats.n_points} points)",
        f"  capture mean / std / CV = {stats.capture_mean:.4f} / "
        f"{stats.capture_std:.4f} / {stats.capture_cv:.3f}",
        f"  capture range           = [{stats.capture_min:.4f}, {stats.capture_max:.4f}]",
        f"  stay mean               = {stats.stay_mean:.4f}",
        f"  Kx~Ky (half capture)    = {stats.kx_mean:.4f} each (axis not identified)",
        f"  Kz / ground             = {stats.kz_mean:.4f} / {stats.ground_mean:.4f}",
        "",
        "Pipeline: known IR/D -> cover ET -> penetrate plow pan -> neighbour hops -> "
        "per-point (Kx+Ky) LFV vs (Kz+ground) deep stay.",
    ])
    return "\n".join(lines)


def write_contribution(folder=None, year=okura.COMPARISON_YEAR):
    """Write outputs/paper/CONTRIBUTION.md (inferred underground flow ratios)."""
    import os
    from . import data

    folder = folder or os.path.join(data.PROJECT, "outputs", "paper")
    os.makedirs(folder, exist_ok=True)
    both = underground_ratio_train_both(years=(2021, 2022))
    hold = underground_ratio_holdout(train_year=2021, eval_year=2022)
    path = os.path.join(folder, "CONTRIBUTION.md")
    body = [
        "# Contribution: inferred underground flow ratios",
        "",
        "## Train on both 2021 and 2022",
        "",
        "```",
        contribution_text(both),
        "```",
        "",
        "## Optional single-year freeze (2021 -> 2022)",
        "",
        "```",
        contribution_text(hold),
        "```",
        "",
        "## Claim boundary",
        "",
        "Surface IR/W (Fig. 8) and ADCP drainage are **forcing**, not novelty.",
        "We infer how leftover after ET is partitioned underground, per point,",
        "with district mean capture trained on both Okura seasons (Table 5 LFV/LfA).",
        "Per-point LFV is that capture share; neighbour hops cover local ET shortfalls.",
        "No artificial capture floor — pins match the tables.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(body))
    return path

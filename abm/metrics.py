# -*- coding: utf-8 -*-
"""Season WUE'/RUE/LUE from an ABM history using cyclic.okura equations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cyclic import okura


def history_to_frame(world) -> pd.DataFrame:
    """Flatten WaterWorld.history into per-day district totals."""
    rows = []
    for step in world.history:
        area = world.area_m2
        ir_m3 = (step.irrigation_mm * area / 1000.0).sum()
        d_m3 = (step.drainage_mm * area / 1000.0).sum()
        et_mm_mean = float(np.average(step.eta_mm, weights=area))
        rain_mm = float(step.rain_mm)
        lfv_m3 = (step.to_pool_mm * area / 1000.0).sum()
        for block in ("North", "South"):
            mask = world.block == block
            a = area[mask]
            rows.append({
                "day": step.day,
                "block": block,
                "rain_mm": rain_mm,
                "et_mm": et_mm_mean,
                "irrigation_mm": float(step.irrigation_mm[mask].mean()) if mask.any() else 0.0,
                "drainage_mm": float(step.drainage_mm[mask].mean()) if mask.any() else 0.0,
                "irrigation_m3": float((step.irrigation_mm[mask] * a / 1000.0).sum()),
                "drainage_m3": float((step.drainage_mm[mask] * a / 1000.0).sum()),
                "leftover_mm": float(step.leftover_mm[mask].mean()) if mask.any() else 0.0,
                "to_pool_m3": float((step.to_pool_mm[mask] * a / 1000.0).sum()),
            })
        rows.append({
            "day": step.day,
            "block": "district",
            "rain_mm": rain_mm,
            "et_mm": et_mm_mean,
            "irrigation_mm": float(step.irrigation_mm.mean()),
            "drainage_mm": float(step.drainage_mm.mean()),
            "irrigation_m3": ir_m3,
            "drainage_m3": d_m3,
            "leftover_mm": float(step.leftover_mm.mean()),
            "to_pool_m3": lfv_m3,
        })
    return pd.DataFrame(rows)


def _season_mm(world, attr: str, mask: np.ndarray) -> float:
    """Area-weighted season total (mm) of a per-paddy StepResult field over ``mask``."""
    a = world.area_m2[mask]
    if not a.size:
        return 0.0
    return float(sum(np.dot(getattr(s, attr)[mask], a) for s in world.history) / a.sum())


def _season_m3(world, attr: str, mask: np.ndarray) -> float:
    a = world.area_m2[mask]
    return float(sum(np.dot(getattr(s, attr)[mask], a) for s in world.history) / 1000.0)


def season_efficiencies(world) -> dict:
    """Okura Eq. 11–14 + FAO-33 relative yield on the simulated season.

    As in the paper: D and Lf are the *cyclic* blocks' (only their drainage reaches
    the pool); RUE and LUE divide by IR_all, everything the station sent to the North
    or South area, cyclic and non-cyclic alike (Eq. 5, 12–14); WUE' is per cyclic
    block with ETc and effective rain (Eq. 11).
    """
    if not world.history:
        return {}

    rain = np.array([s.rain_mm for s in world.history])
    etc = np.array([s.etc_mm for s in world.history])
    er_season = float(okura.effective_rain_mm(rain).sum())
    cyc = np.asarray(world.cyclic, dtype=bool)

    out_block = {}
    ir_all, d_cyc = {}, {}
    for block in ("North", "South"):
        in_block = world.block == block
        mask = in_block & cyc
        ir_all[block] = _season_m3(world, "irrigation_mm", in_block)
        d_cyc[block] = _season_m3(world, "drainage_mm", mask)
        ir_mm = _season_mm(world, "irrigation_mm", mask)
        d_mm = _season_mm(world, "drainage_mm", mask)
        et_mm = _season_mm(world, "eta_mm", mask)
        etp_mm = _season_mm(world, "et_potential_mm", mask)
        out_block[block] = dict(
            IR_mm=ir_mm, D_mm=d_mm, ET_mm=et_mm,
            # Okura Eq. 3: the residual the paper calls lateral flow.
            Lf_mm=ir_mm + float(rain.sum()) - d_mm - et_mm,
            WUE_prime_pct=float(okura.eq11_wue_prime_pct(etp_mm, ir_mm, d_mm, er_season)),
            RUE=float(okura.eq12_rue(d_cyc[block], ir_all[block])),
        )

    lfv = _season_m3(world, "to_pool_mm", cyc)
    lue = okura.eq14_lue(lfv, ir_all["North"], ir_all["South"])
    wue = {b: out_block[b]["WUE_prime_pct"] for b in out_block}
    cyc_area = world.area_m2[cyc]
    cyc_wue = float(np.average([wue["North"], wue["South"]],
                               weights=[world.area_m2[cyc & (world.block == b)].sum() for b in ("North", "South")]))         if cyc_area.size else float("nan")

    pins = getattr(world.physics, "efficiency_pins", {}) or {}
    pin = pins.get(str(world.config.year), {})

    sent = np.stack([s.irrigation_sent_mm for s in world.history], axis=0)
    sent_total = float((sent * world.area_m2 / 1000.0).sum())

    return {
        "year": world.config.year,
        "days": len(world.history),
        "capture": world.capture,
        "neighbour": world.neighbour,
        "stay": world.stay,
        "WUE_prime_pct": cyc_wue,
        "WUE_prime_North": wue["North"],
        "WUE_prime_South": wue["South"],
        "RUE_North": out_block["North"]["RUE"],
        "RUE_South": out_block["South"]["RUE"],
        "RUE_all": float(okura.eq13_rue_all(d_cyc["North"], d_cyc["South"], ir_all["North"], ir_all["South"])),
        "LUE": float(lue),
        "LFV_m3": lfv,
        "IR_m3": ir_all["North"] + ir_all["South"],
        "IR_sent_m3": sent_total,
        "D_m3": d_cyc["North"] + d_cyc["South"],
        "IR_cyclic_mm": float(np.average([out_block[b]["IR_mm"] for b in out_block])),
        "D_cyclic_mm": float(np.average([out_block[b]["D_mm"] for b in out_block])),
        "Lf_North_mm": out_block["North"]["Lf_mm"],
        "Lf_South_mm": out_block["South"]["Lf_mm"],
        "W_m3": float(sum(s.withdraw_m3 for s in world.history)),
        "O_m3": float(sum(s.outflow_m3 for s in world.history)),
        "dWB_m3": float(world.history[-1].dwb_m3),
        "ET_mm": _season_mm(world, "eta_mm", cyc),
        "ET_demand_mm": _season_mm(world, "et_potential_mm", np.ones(world.n, bool)),
        "stress_mm": _season_mm(world, "stress_mm", np.ones(world.n, bool)),
        "stress_cyclic_mm": _season_mm(world, "stress_mm", cyc),
        "stress_noncyclic_mm": _season_mm(world, "stress_mm", ~cyc),
        "ETc_mm": float(etc.sum()),
        "rain_mm": float(rain.sum()),
        "relative_yield": float(getattr(world, "relative_yield", np.array([1.0])).mean()),
        "mean_ks": float(getattr(world, "ks", np.array([1.0])).mean()),
        "pool_m3": float(getattr(world, "pool_m3", 0.0)),
        "paper_LUE": pin.get("LUE"),
        "paper_RUE_all": pin.get("RUE_all"),
    }


def okura_comparison(world) -> pd.DataFrame:
    """Simulated season beside Okura et al. (2025) Tables 3–6 for the same year.

    Per cyclic block (mm, Table 4) and station scale (10^3 m3, Tables 3/5/6). Only
    for 2021/2022 — the years the paper monitored; scenarios have no reference.
    """
    year = int(world.config.year)
    if year not in okura.TABLE5_1000M3 or not world.history:
        return pd.DataFrame()
    eff = season_efficiencies(world)
    rows = []
    for block in ("North", "South"):
        ref = okura.TABLE4_MM[(block, year)]
        mask = (world.block == block) & world.cyclic
        sim = dict(rain=eff["rain_mm"], irrigation=_season_mm(world, "irrigation_mm", mask),
                   et=_season_mm(world, "eta_mm", mask), drainage=_season_mm(world, "drainage_mm", mask),
                   lateral=eff[f"Lf_{block}_mm"])
        for term in ("rain", "irrigation", "et", "drainage", "lateral"):
            rows.append(dict(scale=f"{block} block (mm)", term=term, simulated=sim[term], okura=float(ref[term])))
    t5, t6 = okura.TABLE5_1000M3[year], okura.TABLE6[year]
    for term, sim, ref in (("W", eff["W_m3"], t5["W"]), ("O", eff["O_m3"], t5["O"]),
                           ("LFV_pump", eff["LFV_m3"], t5["LFV_pump"]), ("dWB", eff["dWB_m3"], t5["dWB"]),
                           ("IR_all", eff["IR_m3"], t5["IR_all_North"] + t5["IR_all_South"])):
        rows.append(dict(scale="station (10^3 m3)", term=term, simulated=sim / 1000.0, okura=float(ref)))
    for term in ("RUE_North", "RUE_South", "RUE_all", "LUE"):
        rows.append(dict(scale="ratio", term=term, simulated=eff[term], okura=float(t6[term])))
    frame = pd.DataFrame(rows)
    frame["ratio"] = frame.simulated / frame.okura.where(frame.okura != 0)
    return frame


def summarise_run(world, agent_name: str = "") -> pd.DataFrame:
    eff = season_efficiencies(world)
    eff["agent"] = agent_name
    return pd.DataFrame([eff])


def record_farmer_performance(model, step):
    """Freeze ownership/personality at decision time; later roster edits cannot relabel history."""
    rows = []
    world = model.world
    for fid, agent in model.farmer_agents.items():
        for cyclic in (True, False):
            for block in np.unique(world.block):
                ids = np.flatnonzero((step.owner_id == fid) & (world.cyclic == cyclic) & (world.block == block))
                if not len(ids):
                    continue
                area = world.area_m2[ids]
                mean = lambda values: float(np.average(np.asarray(values)[ids], weights=area))
                policy = agent.policy
                rows.append(dict(cohort=model.farmer_cohort, farmer_id=fid, farmer_type=agent.name,
                    day=step.day, cyclic_status='Cyclic' if cyclic else 'Non-cyclic', block=str(block), paddies=len(ids), area_ha=float(area.sum()/1e4),
                    irrigation_mm=mean(step.irrigation_mm), drainage_mm=mean(step.drainage_mm),
                    sent_mm=mean(step.irrigation_sent_mm),
                    requested_mm=mean(step.requested_mm) if step.requested_mm is not None else np.nan,
                    capture_mm=mean(step.to_pool_mm),
                    eta_mm=mean(step.eta_mm), etc_mm=mean(step.et_potential_mm), mean_ks=mean(step.ks),
                    stress_mm=mean(step.stress_mm),
                    score=mean(step.ks)-.15*mean(step.irrigation_mm)/40.-.05*mean(step.drainage_mm)/30.,
                    cooperation=float(policy.cooperation_level) if hasattr(policy, 'cooperation_level') else np.nan,
                    neighbour_offset=float(getattr(policy, 'traits', {}).get('neighbour_offset', np.nan)),
                    calendar_shift_days=int(getattr(policy, 'calendar_shift_days', 0)),
                    part_time=float(getattr(policy, 'traits', {}).get('part_time', np.nan)),
                    checked=int(bool(getattr(policy, 'checked_today', True))),
                    spend=float(getattr(policy, 'spend', 0.)),
                    copying_events=int(getattr(policy, 'copied_from', None) is not None)))
    return rows


def farmer_performance(model, *, statuses=None, blocks=None, farmer_types=None, rank_by="score") -> pd.DataFrame:
    """Area-weight within farms; accumulate over recorded days for each ownership cohort."""
    from .crop import season_relative_yield
    records = pd.DataFrame(model.farmer_performance_history)
    if records.empty:
        return pd.DataFrame()
    for column, selected in [('cyclic_status', statuses), ('block', blocks), ('farmer_type', farmer_types)]:
        if selected is not None:
            records = records[records[column].isin(selected)]
    if records.empty:
        return pd.DataFrame()
    # Merge selected parcel strata within each farmer/day before summing days.
    # This prevents double-counting days, ETc, cooperation, or copying events.
    daily_keys = ['cohort', 'farmer_id', 'farmer_type', 'day']
    weighted = ['irrigation_mm', 'drainage_mm', 'sent_mm', 'requested_mm', 'capture_mm', 'eta_mm', 'etc_mm', 'mean_ks', 'score', 'stress_mm']
    for col in weighted:
        records[col] = records[col] * records.area_ha
    agg = {col: 'sum' for col in weighted}
    agg.update(area_ha='sum', paddies='sum', cooperation='first', neighbour_offset='first',
               copying_events='max', calendar_shift_days='first', part_time='first',
               checked='max', spend='last')
    records = records.groupby(daily_keys, as_index=False).agg(agg)
    for col in weighted:
        records[col] /= records.area_ha
    keys = ['cohort', 'farmer_id', 'farmer_type']
    result = records.groupby(keys, sort=True).agg(
        days=('day','size'), paddies=('paddies','last'), area_ha=('area_ha','last'),
        score=('score','mean'), mean_ks=('mean_ks','mean'),
        irrigation_mm=('irrigation_mm','sum'), drainage_mm=('drainage_mm','sum'),
        sent_mm=('sent_mm','sum'), requested_mm=('requested_mm','sum'), capture_mm=('capture_mm','sum'),
        eta_mm=('eta_mm','sum'), etc_mm=('etc_mm','sum'), stress_mm=('stress_mm','sum'),
        cooperation=('cooperation','last'), neighbour_offset=('neighbour_offset','last'),
        copying_events=('copying_events','sum'), calendar_shift_days=('calendar_shift_days','last'),
        part_time=('part_time','last'), check_days=('checked','sum'), spend=('spend','last')).reset_index()
    result['relative_yield_pct'] = [100*float(season_relative_yield(np.array([eta]), etc)[0])
        for eta, etc in zip(result.eta_mm, result.etc_mm)]
    result['sent_used_pct'] = 100*result.irrigation_mm / result.sent_mm.where(result.sent_mm > 0)
    result['request_met_pct'] = 100*result.irrigation_mm / result.requested_mm.where(result.requested_mm > 0)
    return _rank_farmers(result, rank_by)


RANK_METRICS = {'score': False, 'relative_yield_pct': False, 'mean_ks': False,
                'irrigation_mm': True, 'drainage_mm': True, 'capture_mm': False}


def _rank_farmers(frame, rank_by):
    if rank_by not in RANK_METRICS:
        raise ValueError(f'Unknown ranking metric: {rank_by}')
    return frame.sort_values(rank_by, ascending=RANK_METRICS[rank_by], kind='stable').reset_index(drop=True)


def farmer_type_performance(model, **filters) -> pd.DataFrame:
    """Equal-weight mean of individual farm metrics, not a pooled district mean."""
    farms = farmer_performance(model, **filters)
    if farms.empty:
        return farms
    numeric = ['days','area_ha','score','relative_yield_pct','mean_ks','irrigation_mm',
               'drainage_mm','capture_mm','sent_used_pct','copying_events','cooperation',
               'neighbour_offset','requested_mm','request_met_pct','calendar_shift_days',
               'part_time','check_days','spend']
    group = farms.groupby('farmer_type')
    result = group[numeric].mean()
    result.insert(0, 'farmer_periods', group.size())
    return _rank_farmers(result.reset_index(), filters.get('rank_by', 'score'))


def best_farmers(model, limit=10, **filters) -> pd.DataFrame:
    """Rank by mean social-learning score: Ks - .15*IR/40 - .05*D/30 per day."""
    if limit < 0:
        raise ValueError('limit must be non-negative')
    return farmer_performance(model, **filters).head(limit)

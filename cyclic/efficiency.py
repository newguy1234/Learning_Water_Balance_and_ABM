"""WUE', RUE and LUE as Okura et al. define them, for any daily block series -- the model's,
the one interpolated from the paper, or the measured one when it arrives -- and what
actually sets them.

THE EQUATIONS AND WHAT THEY RESPOND TO. Lf is the Eq. 3 residual (the paper's lateral
flow) and R - ER is the rain the MAFF rule does not count.

  WUE' = ET / (IR - D + ER)                                          Eq. 11
       = ET / (ET + Lf - (R - ER))              because IR - D = ET + Lf - R  (Eq. 3)

    In the model Lf = lateral + deep percolation + storage change, and lateral + deep is
    what gets through the plow pan, K_pan h / h_ref while water stands (proportional to
    ponded depth, not a leftover ratio). So WUE'
    is seepage against ET: pond deeper or longer and Lf rises and WUE' falls. Draining
    more does not lower Lf; it lowers IR - D for the same ET -- which is how the North
    block's WUE' rose in 2022 when its drainage went up 63%.

  RUE_i = D_i A_i / IR_all_i = (D_i / IR_i) x (IR_i / IR_all_i)        Eq. 12

    The share of the cyclic block's own supply leaving over the surface (the farmer's
    discharge + overflow on rain days + the outlet leak), times the block's share of its
    area's supply.

  LUE = LFV_pump / IR_all = c x sum_i (Lf_i A_i) / IR_all                Eq. 14 with Eq. 7

    The same seepage that lowers WUE', counted coming back, times the capture share c.
    Ponding depth moves WUE' down and LUE up together: one physical quantity on both
    sides of the trade-off the paper describes.

    season(blocks)             Table 4 form, plus WUE and WUE'
    ratios(blocks, station)    Table 6 form
    monthly_wue_prime(blocks)  Table 7 form
    compare(blocks, station, year)   all of the above beside the paper
    identities(blocks, station)      the decompositions above, evaluated
    sensitivity(year, ...)     elasticity of each efficiency to each parameter
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import okura

BLOCKS = okura.BLOCKS


def season(blocks):
    rows = []
    for block in BLOCKS:
        g = blocks[blocks["block"] == block]
        if g.empty:
            continue
        row = {"block": block, "days": len(g),
               "rain_mm": g["rain_mm"].sum(),
               "er_mm": okura.effective_rain_mm(g["rain_mm"].to_numpy()).sum(),
               "irrigation_mm": g["irrigation_mm"].sum(),
               "drainage_mm": g["drainage_mm"].sum(),
               "et_mm": g["et_mm"].sum()}
        row["lateral_residual_mm"] = (row["irrigation_mm"] + row["rain_mm"]
                                      - row["drainage_mm"] - row["et_mm"])
        row["wue_pct"] = okura.eq10_wue_pct(row["et_mm"], row["irrigation_mm"], row["drainage_mm"],
                                            row["rain_mm"])
        row["wue_prime_pct"] = okura.eq11_wue_prime_pct(row["et_mm"], row["irrigation_mm"],
                                                        row["drainage_mm"], row["er_mm"])
        for extra in ("lateral_sim_mm", "deep_mm", "farmer_discharge_mm", "passive_drainage_mm"):
            if extra in g:
                row[extra] = g[extra].sum()
        rows.append(row)
    return pd.DataFrame(rows).set_index("block")


def ratios(blocks, station):
    drainage = {b: blocks.loc[blocks["block"] == b, "drainage_m3"].sum() for b in BLOCKS}
    supply = {b: blocks.loc[blocks["block"] == b, "irrigation_all_m3"].sum() for b in BLOCKS}
    total = sum(supply.values())
    out = {
        "RUE_North": okura.eq12_rue(drainage["North"], supply["North"]),
        "RUE_South": okura.eq12_rue(drainage["South"], supply["South"]),
        "RUE_all": okura.eq13_rue_all(drainage["North"], drainage["South"], supply["North"], supply["South"]),
    }
    if station is not None and "LFV_pump_m3" in station:
        lfv = station["LFV_pump_m3"].sum()
        out["LUE"] = okura.eq14_lue(lfv, supply["North"], supply["South"])
        out["SDR"] = okura.surface_drainage_ratio(drainage["North"] + drainage["South"], lfv)
    if station is not None and {"W_m3", "O_m3"} <= set(station.columns):
        out["lake_ratio"] = okura.lake_ratio(station["W_m3"].sum(), station["O_m3"].sum(), total)
    return out


def monthly_wue_prime(blocks):
    frame = blocks.assign(month=pd.to_datetime(blocks["date"]).dt.month,
                          er=okura.effective_rain_mm(blocks["rain_mm"].to_numpy()))
    table = {}
    for block in BLOCKS:
        g = frame[frame["block"] == block].groupby("month")
        sums = g[["et_mm", "irrigation_mm", "drainage_mm", "er"]].sum()
        table[block] = {m: okura.eq11_wue_prime_pct(r.et_mm, r.irrigation_mm, r.drainage_mm, r.er)
                        for m, r in sums.iterrows()}
    return pd.DataFrame(table)


def compare(blocks, station, year=okura.COMPARISON_YEAR):
    """The model beside every figure the paper reports that it can be compared with."""
    rows = []

    def add(group, quantity, block, model, paper):
        error = (model - paper) / paper * 100.0 if paper not in (None, 0) else np.nan
        rows.append({"table": group, "quantity": quantity, "block": block,
                     "model": model, "paper": paper, "error_%": error})

    s = season(blocks)
    months = monthly_wue_prime(blocks)
    for block in BLOCKS:
        t4 = okura.TABLE4_MM[(block, year)]
        add("Table 4", "irrigation mm", block, s.loc[block, "irrigation_mm"], t4["irrigation"])
        add("Table 4", "drainage mm", block, s.loc[block, "drainage_mm"], t4["drainage"])
        add("Table 4", "ET mm", block, s.loc[block, "et_mm"], t4["et"])
        add("Table 4", "rain mm", block, s.loc[block, "rain_mm"], t4["rain"])
        add("Table 4", "Lf (Eq.3 residual) mm", block, s.loc[block, "lateral_residual_mm"], t4["lateral"])
        t7 = okura.TABLE7_WUE_PCT[(block, year)]
        add("Table 7", "WUE' season %", block, s.loc[block, "wue_prime_pct"], t7["season"])
        for m, name in okura.MONTHS.items():
            if m in months.index:
                add("Table 7", f"WUE' {name} %", block, months.loc[m, block], t7[m])

    t6 = okura.TABLE6[year]
    for key, value in ratios(blocks, station).items():
        add("Table 6", key, "district" if "_" not in key or key.endswith("all") else key.split("_")[1],
            value, t6.get(key))

    t5 = okura.TABLE5_1000M3[year]
    if station is not None:
        for key, column in (("W", "W_m3"), ("O", "O_m3"), ("LFV_pump", "LFV_pump_m3"),
                            ("D_North", "D_North_m3"), ("D_South", "D_South_m3"),
                            ("IR_all_North", "IR_all_North_m3"), ("IR_all_South", "IR_all_South_m3")):
            if column in station:
                add("Table 5", f"{key} x10^3 m3", "district" if key in ("W", "O", "LFV_pump") else
                    ("North" if "North" in key else "South"),
                    station[column].sum() / 1000.0, t5[key])
    return pd.DataFrame(rows)


def identities(blocks, station, physics=None):
    """The decompositions in the module docstring, evaluated, with their closure."""
    s = season(blocks)
    supply_total = blocks["irrigation_all_m3"].sum()
    rows = []
    for block in BLOCKS:
        r = s.loc[block]
        g = blocks[blocks["block"] == block]
        ineffective = r["rain_mm"] - r["er_mm"]
        row = {
            "block": block,
            "WUE' Eq.11 %": r["wue_prime_pct"],
            "WUE' = ET/(ET+Lf-(R-ER)) %": 100.0 * r["et_mm"] / (r["et_mm"] + r["lateral_residual_mm"] - ineffective),
            "ET mm": r["et_mm"], "Lf mm": r["lateral_residual_mm"], "R-ER mm": ineffective,
            "D/IR": r["drainage_mm"] / r["irrigation_mm"],
            "IR/IR_all": g["irrigation_m3"].sum() / g["irrigation_all_m3"].sum(),
        }
        row["RUE = D/IR x IR/IR_all"] = row["D/IR"] * row["IR/IR_all"]
        if "lateral_sim_mm" in r:
            storage = g["storage_m3"].to_numpy() / g["area_m2"].iloc[0] * 1000.0
            first_day = (g["rain_mm"].iloc[0] + g["irrigation_mm"].iloc[0] - g["et_mm"].iloc[0]
                         - g["drainage_mm"].iloc[0] - g["lateral_sim_mm"].iloc[0] - g["deep_mm"].iloc[0])
            change = storage[-1] - (storage[0] - first_day)
            row.update({"lateral mm": r["lateral_sim_mm"], "deep mm": r["deep_mm"], "storage change mm": change,
                        "Lf - (lateral+deep+dS) mm": r["lateral_residual_mm"] - r["lateral_sim_mm"] - r["deep_mm"] - change,
                        "farmer discharge share of D": r["farmer_discharge_mm"] / r["drainage_mm"]
                        if r["drainage_mm"] else np.nan})
            if physics is not None:
                row["LUE share = c x lateral A / IR_all"] = (physics.pool_capture_share
                                                             * g["lateral_sim_m3"].sum() / supply_total)
        rows.append(row)
    return pd.DataFrame(rows).set_index("block").T


def headline(blocks, station):
    s = season(blocks)
    out = {f"WUE'_{b}": s.loc[b, "wue_prime_pct"] for b in BLOCKS}
    out.update({f"IR_{b}_mm": s.loc[b, "irrigation_mm"] for b in BLOCKS})
    out.update({f"D_{b}_mm": s.loc[b, "drainage_mm"] for b in BLOCKS})
    out.update(ratios(blocks, station))
    return out


def sensitivity(year=okura.COMPARISON_YEAR, agent_params=None, agent_names=(), physics=None, physics_names=(),
                step=0.1, policy=None):
    """Elasticity of every headline efficiency to each parameter: the % change in the
    metric per % change in the parameter, by central difference at +-`step`."""
    from .agents import ETFollower
    from .model import CyclicModel
    from .physics import Physics

    policy = policy or ETFollower
    base = {**policy.defaults, **(agent_params or {})}
    physics = physics or Physics()

    def run(params, phys):
        results = CyclicModel(year, agents=policy(**params), physics=phys, record_paddies=False).run()
        return headline(results.blocks, results.station)

    reference = run(base, physics)
    rows = []
    for name in agent_names:
        hi = run({**base, name: base[name] * (1 + step)}, physics)
        lo = run({**base, name: base[name] * (1 - step)}, physics)
        rows.append({"parameter": f"farmer.{name}",
                     **{k: (hi[k] - lo[k]) / reference[k] / (2 * step) for k in reference}})
    for name in physics_names:
        value = getattr(physics, name)
        hi = run(base, physics.replace(**{name: value * (1 + step)}))
        lo = run(base, physics.replace(**{name: value * (1 - step)}))
        rows.append({"parameter": f"physics.{name}",
                     **{k: (hi[k] - lo[k]) / reference[k] / (2 * step) for k in reference}})
    return pd.DataFrame(rows).set_index("parameter"), reference

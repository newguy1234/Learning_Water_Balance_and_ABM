"""Everything Okura, Kamei & Kato (2025) report, their equations, and a daily series
interpolated from their tables until the measured series arrive.

    Okura, F., Kamei, A. and Kato, T. (2025). Water balance analysis of hydrological
    processes in cyclic irrigation: a case study of the Imbanuma irrigation area in
    Chiba, Japan. Journal of Hydrology: Regional Studies 57, 102111.

    okura.check_tables(2021)            # do the paper's own tables agree with each other
    okura.interpolate_daily(2021, wx)   # a daily block series consistent with the tables

WHAT THEY MEASURED. Irrigation IR (pump records, m3/d), surface drainage D (ADCP in the
main drainage canal, every 3 h), lake withdrawal W and outflow O (pump records), and
weather (on-site station). Lateral flow Lf was *not* measured: Eq. 3 obtains it as the
residual that closes each block's balance. So only IR and D are observations of what
the farmers did; everything else is arithmetic on them.

WHAT THE TABLES SAY ABOUT THEMSELVES, checked by check_tables():
  * Table 3 has rainfall and lake outflow in each other's rows. Its subtotals only add
    up with R = 4144 and O = 2666 (x10^3 m3) in 2021, which is also what 791 mm over the
    523.9 ha of cyclic blocks gives. The values below are the corrected ones.
  * The prose of Section 3.2.2 gives North irrigation as 15.3 and 17.5 mm/d. Table 4
    gives 1787 and 2043 mm, i.e. 13.6 and 15.6 mm/d, and it is Table 4 that closes
    against Tables 3 and 5 (cyclic irrigation 9,797 + non-cyclic 4,937 = 14,734 x10^3
    m3 supplied). Table 4 is used; the prose figures are kept in TEXT_IRRIGATION_MM_D.
  * Table 6 labels RUE_N as D_N / IR_N, but its values are D_N / IR_N_all -- the whole
    North area's supply, cyclic and non-cyclic (1608 / 6550 = 0.245). Eq. 12 says the
    same, so eq12_rue follows Eq. 12.
  * Table 7's season WUE' for the North block cannot be reproduced from Table 4 under the
    MAFF effective-rainfall rule: it would need more effective rain than 80% of all rain.
    The South block's can. The monthly *shape* of Table 7 is used, rescaled to Table 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CITATION = ("Okura, Kamei & Kato (2025), J. Hydrology: Regional Studies 57, 102111 -- "
            "Shiroyama-Jimbee cyclic irrigation area, Imbanuma, Chiba")

BLOCKS = ("North", "South")
YEARS = (2021, 2022)
# The year runs are compared against by default. 2022 is the drought year: less rain, more
# irrigation, and surface drainage and lateral flow swap places against 2021 (Section 4.1).
COMPARISON_YEAR = 2022
SEASON_DAYS = 131                       # April 17 to August 25 inclusive, both years
MONTHS = {4: "April", 5: "May", 6: "June", 7: "July", 8: "August"}


def season_dates(year):
    return pd.date_range(f"{year}-04-17", periods=SEASON_DAYS, freq="D")


def calendar_dates(year):
    return pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")


# --- Table 1: paddy area (ha) ------------------------------------------------------
TABLE1_HA = {
    "North": {"cyclic": 260.4, "non_cyclic": 307.3, "total": 567.7},
    "South": {"cyclic": 263.5, "non_cyclic": 151.7, "total": 415.2},
}


def cyclic_area_m2(block):
    return TABLE1_HA[block]["cyclic"] * 1e4


def non_cyclic_area_m2(block):
    return TABLE1_HA[block]["non_cyclic"] * 1e4


def ir_ncb_m3(block, year):
    """Table 5 IR_all − Table 4 cyclic IR (m³/season)."""
    cyclic_ir = TABLE4_MM[(block, year)]["irrigation"] * cyclic_area_m2(block) / 1000.0
    return max(float(TABLE5_1000M3[year][f"IR_all_{block}"]) * 1000.0 - cyclic_ir, 0.0)


def d_ncb_m3(block, year, rain_mm=None):
    """NCB season drainage (m³) from Okura table structure + accurate rain.

    IR_ncb is Table 5 IR_all − Table 4 cyclic IR. The drainage *share* of net water
    is taken from the cyclic Table 4 closure D/(IR+R−ET). Rain on the NCB area is
    Open-Meteo season depth when `rain_mm` is given (preferred); otherwise Table 4
    rain. ET depth stays Table 4 (same seasonal demand surface as the cyclic block).
    """
    t4 = TABLE4_MM[(block, year)]
    a_n = non_cyclic_area_m2(block)
    net_c_mm = t4["irrigation"] + t4["rain"] - t4["et"]
    if net_c_mm <= 1e-9:
        return 0.0
    d_frac = t4["drainage"] / net_c_mm
    ir_n = ir_ncb_m3(block, year)
    r_mm = float(t4["rain"] if rain_mm is None else rain_mm)
    et_mm = float(t4["et"])
    net_n = ir_n + r_mm * a_n / 1000.0 - et_mm * a_n / 1000.0
    return max(net_n * d_frac, 0.0)


# --- Table 2: monthly totals at the site weather station (mm) ----------------------
TABLE2_MM = {
    2021: {"rain": {4: 43.2, 5: 72.6, 6: 86.6, 7: 313.2, 8: 275.4},
           "et": {4: 49.8, 5: 102.1, 6: 118.8, 7: 138.6, 8: 112.8}},
    2022: {"rain": {4: 99.2, 5: 114.0, 6: 67.4, 7: 70.2, 8: 59.2},
           "et": {4: 45.1, 5: 123.3, 6: 137.2, 7: 159.3, 8: 126.5}},
}

# --- Table 3: whole study site (x10^3 m3), R and O rows corrected -------------------
TABLE3_1000M3 = {
    2021: {"W": 8114, "R": 4144, "O": 2666, "ET": 2735, "IR_ncb": 4937, "dWB": 1919},
    2022: {"W": 7955, "R": 2148, "O": 848, "ET": 3099, "IR_ncb": 5172, "dWB": 984},
}

# --- Table 4: cyclic block water balance, season totals (mm) -------------------------
TABLE4_MM = {
    ("North", 2021): {"rain": 791, "irrigation": 1787, "et": 522, "drainage": 618, "lateral": 1438},
    ("North", 2022): {"rain": 410, "irrigation": 2043, "et": 591, "drainage": 1010, "lateral": 851},
    ("South", 2021): {"rain": 791, "irrigation": 1952, "et": 522, "drainage": 859, "lateral": 1363},
    ("South", 2022): {"rain": 410, "irrigation": 1868, "et": 591, "drainage": 770, "lateral": 917},
}

# Section 3.2.2 prose, kept because it disagrees with Table 4 for the North block.
TEXT_IRRIGATION_MM_D = {("North", 2021): 15.3, ("North", 2022): 17.5,
                        ("South", 2021): 14.9, ("South", 2022): 14.2}

# --- Table 5: irrigation pumping station (x10^3 m3 over the season) -----------------
TABLE5_1000M3 = {
    2021: {"W": 8114, "D_North": 1608, "D_South": 2262, "O": 2666, "IR_all_North": 6550,
           "IR_all_South": 8184, "LFV_pump": 5416, "LfA_North": 3744, "LfA_South": 3591,
           "dWB": 1919},
    2022: {"W": 7955, "D_North": 2631, "D_South": 2029, "O": 848, "IR_all_North": 7425,
           "IR_all_South": 7989, "LFV_pump": 3647, "LfA_North": 2216, "LfA_South": 2416,
           "dWB": 984},
}

# --- Table 6: reuse ratios -----------------------------------------------------------
TABLE6 = {
    2021: {"lake_ratio": 0.37, "LUE": 0.37, "RUE_all": 0.26, "RUE_North": 0.25,
           "RUE_South": 0.28, "SDR": 0.42},
    2022: {"lake_ratio": 0.46, "LUE": 0.24, "RUE_all": 0.30, "RUE_North": 0.35,
           "RUE_South": 0.25, "SDR": 0.56},
}

# --- Table 7: monthly WUE' (%) -------------------------------------------------------
TABLE7_WUE_PCT = {
    ("North", 2021): {4: 18.1, 5: 24.1, 6: 34.2, 7: 24.5, 8: 45.7, "season": 28.1},
    ("North", 2022): {4: 39.2, 5: 49.4, 6: 60.1, 7: 28.5, 8: 44.0, "season": 41.1},
    ("South", 2021): {4: 21.8, 5: 28.8, 6: 32.8, 7: 33.2, 8: 41.1, "season": 31.9},
    ("South", 2022): {4: 25.6, 5: 45.0, 6: 50.2, 7: 38.7, 8: 52.7, "season": 43.0},
}

# Section 2.5: mid-summer drainage "in June by stopping irrigation for approximately two
# weeks". The paper gives no dates; mid-June is an assumption, and a learnable one.
MIDSUMMER_START = "06-10"
MIDSUMMER_DAYS = 14


# --- the equations -------------------------------------------------------------------

def effective_rain_mm(rain_mm):
    """MAFF Japan, as Okura et al. Section 2.6: under 5 mm none counts, 5-80 mm counts
    80%, above 80 mm it is 64 mm. Daily rule; takes a number or an array."""
    rain = np.asarray(rain_mm, dtype=float)
    out = np.where(rain < 5.0, 0.0, np.where(rain < 80.0, 0.8 * rain, 64.0))
    return float(out) if out.ndim == 0 else out


def eq1_storage_change(W, O, R_mm, ET_mm, area_cyclic_m2, IR_ncb):
    """dWB = W - O + (R - ETc)(A_N + A_S)/1000 - IR_N_ncb - IR_S_ncb   [m3]"""
    return W - O + (R_mm - ET_mm) * area_cyclic_m2 / 1000.0 - IR_ncb


def eq3_lateral_mm(IR_m3, area_m2, R_mm, D_mm, ET_mm):
    """Lf_i = 1000 IR_i / A_i + R - D_i - ETc   [mm]: the residual, not a measurement."""
    return 1000.0 * IR_m3 / area_m2 + R_mm - D_mm - ET_mm


def eq6_lfv_pump(W, O, D_North_m3, D_South_m3, IR_all_North, IR_all_South):
    """LFV_pump from the pool balance (Eq. 4): what must have seeped in for the pool to
    close. LFV = IR_N_all + IR_S_all - W + O - D_N A_N/1000 - D_S A_S/1000   [m3]"""
    return IR_all_North + IR_all_South - W + O - D_North_m3 - D_South_m3


def eq7_storage_change(LfA_North_m3, LfA_South_m3, LFV_pump):
    """dWB = A_N Lf_N/1000 + A_S Lf_S/1000 - LFV_pump   [m3]"""
    return LfA_North_m3 + LfA_South_m3 - LFV_pump


def eq10_wue_pct(ET_mm, IR_mm, D_mm, R_mm):
    """WUE_i = ETc / (IR_i - D_i + R) x 100, all as depths."""
    net = IR_mm - D_mm + R_mm
    return 100.0 * ET_mm / net if net > 0 else float("nan")


def eq11_wue_prime_pct(ET_mm, IR_mm, D_mm, ER_mm):
    """WUE'_i = ETc / (IR_i - D_i + ER) x 100. The paper's reported efficiency."""
    net = IR_mm - D_mm + ER_mm
    return 100.0 * ET_mm / net if net > 0 else float("nan")


def eq12_rue(D_m3, IR_all_m3):
    """RUE_i = D_i A_i / (1000 IR_all_i): drainage coming back over what the area got."""
    return D_m3 / IR_all_m3 if IR_all_m3 > 0 else float("nan")


def eq13_rue_all(D_North_m3, D_South_m3, IR_all_North, IR_all_South):
    total = IR_all_North + IR_all_South
    return (D_North_m3 + D_South_m3) / total if total > 0 else float("nan")


def eq14_lue(LFV_pump, IR_all_North, IR_all_South):
    """LUE = LFV_pump / (IR_N_all + IR_S_all): seepage coming back over supply."""
    total = IR_all_North + IR_all_South
    return LFV_pump / total if total > 0 else float("nan")


def lake_ratio(W, O, IR_all):
    """Table 6, (W - O) / IR_all: the share of supply that is net lake water."""
    return (W - O) / IR_all if IR_all > 0 else float("nan")


def surface_drainage_ratio(D_m3, LFV_pump):
    """Table 6, SDR = D / (LFV + D): the share of returns arriving over the surface."""
    total = D_m3 + LFV_pump
    return D_m3 / total if total > 0 else float("nan")


# --- self-consistency ------------------------------------------------------------------

def check_tables(year=2021):
    """Recompute every derived figure in the paper from its inputs. Returns a DataFrame
    of (check, computed, reported, ok). Only failures that matter are flagged."""
    t3, t5, t6 = TABLE3_1000M3[year], TABLE5_1000M3[year], TABLE6[year]
    area = sum(cyclic_area_m2(b) for b in BLOCKS)
    ir_all = t5["IR_all_North"] + t5["IR_all_South"]
    rows = []

    def add(name, computed, reported, tolerance):
        ok = abs(computed - reported) <= tolerance * max(abs(reported), 1e-9)
        rows.append({"check": name, "computed": computed, "reported": reported, "ok": ok})

    # Table 3 already holds (R - ET) x area as volumes, so Eq. 1 is a sum of its rows.
    add("Eq.1 dWB (Table 3)", t3["W"] - t3["O"] + t3["R"] - t3["ET"] - t3["IR_ncb"], t3["dWB"], 0.01)
    add("Eq.6 LFV_pump (Table 5)", eq6_lfv_pump(t5["W"], t5["O"], t5["D_North"], t5["D_South"],
                                                t5["IR_all_North"], t5["IR_all_South"]), t5["LFV_pump"], 0.01)
    add("Eq.7 dWB (Table 5)", eq7_storage_change(t5["LfA_North"], t5["LfA_South"], t5["LFV_pump"]),
        t5["dWB"], 0.01)
    add("Eq.12 RUE_North", eq12_rue(t5["D_North"], t5["IR_all_North"]), t6["RUE_North"], 0.05)
    add("Eq.12 RUE_South", eq12_rue(t5["D_South"], t5["IR_all_South"]), t6["RUE_South"], 0.05)
    add("Eq.13 RUE_all", eq13_rue_all(t5["D_North"], t5["D_South"], t5["IR_all_North"],
                                      t5["IR_all_South"]), t6["RUE_all"], 0.05)
    add("Eq.14 LUE", eq14_lue(t5["LFV_pump"], t5["IR_all_North"], t5["IR_all_South"]), t6["LUE"], 0.05)
    add("(W-O)/IR_all", lake_ratio(t5["W"], t5["O"], ir_all), t6["lake_ratio"], 0.05)
    add("SDR", surface_drainage_ratio(t5["D_North"] + t5["D_South"], t5["LFV_pump"]), t6["SDR"], 0.05)
    add("Rain x cyclic area = Table 3 R", TABLE4_MM[("North", year)]["rain"] * area / 1e6, t3["R"], 0.01)

    cyclic_ir = 0.0
    for block in BLOCKS:
        t4 = TABLE4_MM[(block, year)]
        a = cyclic_area_m2(block)
        cyclic_ir += t4["irrigation"] * a / 1e6
        add(f"Table 4 D x A = Table 5 D_{block}", t4["drainage"] * a / 1e6, t5[f"D_{block}"], 0.01)
        add(f"Table 4 Lf x A = Table 5 LfA_{block}", t4["lateral"] * a / 1e6, t5[f"LfA_{block}"], 0.01)
        add(f"Eq.3 Lf_{block} closes Table 4", t4["irrigation"] + t4["rain"] - t4["drainage"] - t4["et"],
            t4["lateral"], 0.01)
        add(f"Prose irrigation {block} mm/d = Table 4", TEXT_IRRIGATION_MM_D[(block, year)],
            t4["irrigation"] / SEASON_DAYS, 0.02)
        # What effective rain would Table 7's season WUE' need? Physically at most 80% of R.
        er_needed = 100.0 * t4["et"] / TABLE7_WUE_PCT[(block, year)]["season"] - (t4["irrigation"] - t4["drainage"])
        rows.append({"check": f"Table 7 WUE' {block} needs ER <= 0.8 R",
                     "computed": er_needed, "reported": 0.8 * t4["rain"],
                     "ok": er_needed <= 0.8 * t4["rain"]})
    add("Cyclic IR + IR_ncb = Table 5 IR_all", cyclic_ir + t3["IR_ncb"], ir_all, 0.01)
    return pd.DataFrame(rows)


# --- a daily series from the tables ------------------------------------------------------

def _spread(shape, total):
    shape = np.clip(np.nan_to_num(np.asarray(shape, dtype=float)), 0.0, None)
    if shape.sum() > 0:
        return shape / shape.sum() * total
    return np.full(shape.size, total / max(shape.size, 1))


def interpolate_daily(year, weather, drainage_rain_weight=0.5, outflow_rain_threshold_mm=10.0,
                      midsummer=(MIDSUMMER_START, MIDSUMMER_DAYS)):
    """A daily series for each cyclic block that reproduces the paper's tables exactly.

    Until Okura et al. share the measured daily series, this is what the model is scored
    against. Every assumption is a line below, and every season total matches Table 4/5.

      rain, ET     Open-Meteo's daily shape, rescaled to Table 2's monthly totals
      IR - D       per month from Table 7: 100 ET_m / WUE'_m - ER_m, then rescaled so the
                   season equals Table 4 (which Table 7 does not, for the North block)
      D            the season's Table 4 total, spread over months and days half by day
                   count and half by rainfall (`drainage_rain_weight`)
      IR           (IR - D) + D per month, uniform over the month's irrigating days, and
                   zero through the June mid-summer drainage (dates assumed)
      Lf           Eq. 3, daily, so it goes negative on heavy rain days as the paper says
      station      IR_all from the Table 5 / Table 4 ratio, LFV_pump from Lf shares,
                   O on days with rain above `outflow_rain_threshold_mm`, W from Eq. 4

    `weather` is a DataFrame indexed by date with rain_mm and et0_mm.
    Returns (blocks, station): long daily frames with m3 and mm columns.
    """
    dates = season_dates(year)
    wx = weather.reindex(dates)
    month = dates.month.to_numpy()
    n = len(dates)

    rain = np.zeros(n)
    et = np.zeros(n)
    for m in MONTHS:
        sel = month == m
        rain[sel] = _spread(wx["rain_mm"].to_numpy()[sel], TABLE2_MM[year]["rain"][m])
        et[sel] = _spread(wx["et0_mm"].to_numpy()[sel], TABLE2_MM[year]["et"][m])
    er = effective_rain_mm(rain)
    w = drainage_rain_weight

    start = pd.Timestamp(f"{year}-{midsummer[0]}") if midsummer else None
    frames = []
    for block in BLOCKS:
        t4 = TABLE4_MM[(block, year)]
        wue = TABLE7_WUE_PCT[(block, year)]
        net = {m: max(100.0 * TABLE2_MM[year]["et"][m] / wue[m] - er[month == m].sum(), 0.0)
               for m in MONTHS}
        scale = (t4["irrigation"] - t4["drainage"]) / sum(net.values())

        irrigation = np.zeros(n)
        drainage = np.zeros(n)
        for m in MONTHS:
            sel = np.flatnonzero(month == m)
            share = (1 - w) * len(sel) / n + w * rain[sel].sum() / rain.sum()
            d_month = t4["drainage"] * share
            ir_month = net[m] * scale + d_month

            on = np.ones(len(sel), dtype=bool)
            if start is not None:
                on &= ~((dates[sel] >= start) & (dates[sel] < start + pd.Timedelta(days=midsummer[1])))
            irrigation[sel[on]] = ir_month / on.sum()

            r = rain[sel]
            by_rain = r / r.sum() if r.sum() > 0 else np.full(len(sel), 1.0 / len(sel))
            drainage[sel] = d_month * ((1 - w) / len(sel) + w * by_rain)

        area = cyclic_area_m2(block)
        lateral = irrigation + rain - drainage - et
        frame = pd.DataFrame({
            "date": dates, "block": block, "area_m2": area,
            "rain_mm": rain, "er_mm": er, "et_mm": et,
            "irrigation_mm": irrigation, "drainage_mm": drainage, "lateral_mm": lateral,
        })
        for q in ("rain", "et", "irrigation", "drainage", "lateral"):
            frame[f"{q}_m3"] = frame[f"{q}_mm"] * area / 1000.0
        frame["wue_scale"] = scale
        frames.append(frame)
    blocks = pd.concat(frames, ignore_index=True)

    t5 = TABLE5_1000M3[year]
    by = {b: blocks[blocks.block == b].reset_index(drop=True) for b in BLOCKS}
    for b in BLOCKS:
        ratio = t5[f"IR_all_{b}"] * 1000.0 / by[b]["irrigation_m3"].sum()
        blocks.loc[blocks.block == b, "irrigation_all_m3"] = by[b]["irrigation_m3"].to_numpy() * ratio

    lfa = by["North"]["lateral_m3"].to_numpy() + by["South"]["lateral_m3"].to_numpy()
    lfv = t5["LFV_pump"] * 1000.0 * lfa / lfa.sum()
    excess = np.clip(rain - outflow_rain_threshold_mm, 0.0, None)
    outflow = _spread(excess, t5["O"] * 1000.0)
    ir_all = sum(blocks.loc[blocks.block == b, "irrigation_all_m3"].to_numpy() for b in BLOCKS)
    d_all = by["North"]["drainage_m3"].to_numpy() + by["South"]["drainage_m3"].to_numpy()
    station = pd.DataFrame({
        "date": dates,
        "W_m3": ir_all - d_all - lfv + outflow,        # Eq. 4, day by day
        "O_m3": outflow,
        "LFV_pump_m3": lfv,
        "IR_all_m3": ir_all,
        "D_m3": d_all,
    })
    blocks["source"] = "interpolated from Okura et al. (2025) Tables 2, 4, 5, 7"
    return blocks, station

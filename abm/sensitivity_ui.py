# -*- coding: utf-8 -*-
"""Solara viewer for an abm.sensitivity batch (python scripts/abm_sensitivity.py).

    python scripts/abm_sensitivity.py --view            # newest results
    python scripts/abm_sensitivity.py --view outputs/sensitivity/20261001-120000

Tabs
    Trait guide   what every personality trait does, in plain words and in the code
    Verdicts      every trait / scenario expectation: pass, fail, leak, or no prior
    Traits        tornado of paired high − low effects on one output, noise band behind
    Trait detail  one trait: daily demand at low / base / high, and the per-seed pairs
    Scenarios     each weather preset and the pump shortage vs baseline
    Interactions  trait effect under every weather scenario (when the batch ran them)
    Noise & runs  seed-to-seed spread and how many seeds a 1 % / 5 % effect needs
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import solara
import solara.lab
from matplotlib.figure import Figure

from abm import sensitivity as sens

RESULTS_ENV = "ABM_SENSITIVITY_RESULTS"

LOW, BASE, HIGH = "#3a5f8a", "#6b6b6b", "#b86b2b"
PASS, FAIL, LEAK, NEUTRAL = "#2f6f5e", "#a23b3b", "#b8860b", "#6b6b6b"

CSS = """
.sens-panel { background: var(--sens-panel, #fbfaf6); border: 1px solid #e3ded2; border-radius: 8px;
              padding: 12px 16px; margin: 8px 0; overflow-wrap: anywhere; }
.sens-table { border-collapse: collapse; font-size: 13px; width: 100%; }
.sens-table th, .sens-table td { padding: 4px 8px; border-bottom: 1px solid #e3ded2; text-align: right; }
.sens-table th:first-child, .sens-table td:first-child,
.sens-table th:nth-child(2), .sens-table td:nth-child(2) { text-align: left; }
.v-pass { color: #2f6f5e; font-weight: 600; } .v-fail { color: #a23b3b; font-weight: 600; }
.v-leak { color: #b8860b; font-weight: 600; } .v-none { color: #6b6b6b; }
"""


def _load():
    path = os.environ.get(RESULTS_ENV) or sens.latest()
    if not path or not os.path.isdir(path):
        return None
    d = sens.load(path)
    # Recompute the statistics from the runs, so batches saved before a trait was renamed
    # or an expectation added read with today's names and rules.
    from abm.farmer import RENAMED_TRAITS, RETIRED_TRAITS

    runs = d["runs"]
    runs = runs[~runs.trait.isin(RETIRED_TRAITS)].copy()
    runs["trait"] = runs.trait.map(lambda t: RENAMED_TRAITS.get(t, t) if isinstance(t, str) else t)
    if "scenario" not in runs:
        runs["scenario"] = runs["weather"]
    d["runs"] = runs
    d["trait_effects"] = sens.trait_effects(runs)
    d["weather_effects"] = sens.weather_effects(runs)
    d["noise"] = sens.noise_table(runs, d["trait_effects"])
    return d


data = solara.reactive(_load())
metric = solara.reactive("request_m3")
weather_sel = solara.reactive("baseline")
trait_sel = solara.reactive("refill_overshoot")
selected_tab = solara.reactive(0)


def _fmt(v, digits=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    a = abs(float(v))
    if a >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{digits}g}"


def _verdict_class(v: str) -> str:
    return ("v-pass" if v.startswith("pass") else "v-fail" if v.startswith("FAIL")
            else "v-leak" if v.startswith("LEAK") else "v-none")


def _table(frame: pd.DataFrame, cols: list[tuple[str, str]], verdict_col: str | None = "verdict") -> str:
    head = "".join(f"<th>{label}</th>" for _, label in cols)
    body = []
    for _, r in frame.iterrows():
        cells = []
        for key, _ in cols:
            v = r[key]
            if key == verdict_col:
                cells.append(f'<td class="{_verdict_class(str(v))}">{v}</td>')
            elif key == "metric":
                cells.append(f"<td>{sens.METRIC_LABEL.get(v, v)}</td>")
            elif key == "expected":
                cells.append("<td>" + ({1: "↑", -1: "↓", 0: "none"}.get(v, "–") if pd.notna(v) else "–") + "</td>")
            else:
                cells.append(f"<td>{_fmt(v) if not isinstance(v, str) else v}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="sens-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


# --- figures -------------------------------------------------------------------------------

def figure_tornado(effects: pd.DataFrame, runs: pd.DataFrame, metric_key: str, weather: str) -> Figure:
    e = effects[(effects.metric == metric_key) & (effects.scenario == weather)].copy()
    fig = Figure(figsize=(8, max(3, .38 * len(e) + 1.2)), layout="constrained")
    ax = fig.subplots()
    if e.empty:
        ax.text(.5, .5, "No trait sweep for this scenario", ha="center", transform=ax.transAxes)
        return fig
    e["abs"] = e["diff"].abs()
    e = e.sort_values("abs")
    colors = [PASS if v.startswith("pass") else FAIL if v.startswith("FAIL") else LEAK if v.startswith("LEAK")
              else NEUTRAL for v in e.verdict]
    y = np.arange(len(e))
    ax.barh(y, e["diff"], xerr=e["ci95"].fillna(0), color=colors, alpha=.85, capsize=3)
    noise = e["noise_sd"].iloc[0]
    if np.isfinite(noise):
        ax.axvspan(-noise, noise, color="#cfcac0", alpha=.45, zorder=0, label="± 1 sd across seeds (base)")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.axvline(0, color="#333", lw=.8)
    ax.set_yticks(y, [f"{t}  ({r:+.1f}%)" if np.isfinite(r) else t for t, r in zip(e.trait, e.rel_pct)])
    ax.set_xlabel(f"high − low: {sens.METRIC_LABEL[metric_key]} ({sens.METRIC_UNIT[metric_key]})")
    ax.set_title(f"Trait effects · {weather} · 95% CI over seeds", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def _band(ax, daily: pd.DataFrame, runs: pd.DataFrame, mask, col, color, label, cumulative=False):
    ids = runs.loc[mask, "run_id"]
    d = daily[daily.run_id.isin(ids)]
    if d.empty:
        return
    piv = d.pivot_table(index="day", columns="run_id", values=col)
    if cumulative:
        piv = piv.cumsum()
    ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), color=color, alpha=.18, lw=0)
    ax.plot(piv.index, piv.mean(axis=1), color=color, lw=1.6, label=label)


def figure_trait_daily(daily, runs, trait, weather, col) -> Figure:
    fig = Figure(figsize=(9, 5.4), layout="constrained")
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    base = runs.trait.isna() & (runs.scenario == weather)
    lo = (runs.trait == trait) & (runs.level == "low") & (runs.scenario == weather)
    hi = (runs.trait == trait) & (runs.level == "high") & (runs.scenario == weather)
    for ax, cum in ((ax1, False), (ax2, True)):
        _band(ax, daily, runs, lo, col, LOW, f"{trait} low", cum)
        _band(ax, daily, runs, base, col, BASE, "as drawn", cum)
        _band(ax, daily, runs, hi, col, HIGH, f"{trait} high", cum)
        ax.spines[["top", "right"]].set_visible(False)
    label = col.replace("_m3", " (m³)").replace("_mm", " (mm)").replace("_", " ")
    ax1.set_ylabel(f"daily {label}")
    ax2.set_ylabel(f"cumulative {label}")
    ax2.set_xlabel("season day (0 = 17 April)")
    ax1.legend(fontsize=8, frameon=False, ncol=3)
    ax1.set_title(f"{trait} · {weather} · line = mean, band = range over seeds", fontsize=11, loc="left")
    return fig


def figure_pairs(runs, trait, weather, metric_key) -> Figure:
    fig = Figure(figsize=(4.6, 4), layout="constrained")
    ax = fig.subplots()
    sub = runs[(runs.trait == trait) & (runs.scenario == weather)]
    base = runs[runs.trait.isna() & (runs.scenario == weather)].set_index("seed")[metric_key]
    for seed, g in sub.groupby("seed"):
        g = g.set_index("level")[metric_key]
        xs, ys = [], []
        for x, lvl in ((0, "low"), (1, "base"), (2, "high")):
            v = base.get(seed, np.nan) if lvl == "base" else g.get(lvl, np.nan)
            xs.append(x)
            ys.append(v)
        ax.plot(xs, ys, "-o", color="#8a8a8a", ms=4, lw=.8)
    ax.set_xticks([0, 1, 2], ["low", "as drawn", "high"])
    ax.set_ylabel(f"{sens.METRIC_LABEL[metric_key]} ({sens.METRIC_UNIT[metric_key]})")
    ax.set_title("each line = one seed (paired)", fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def figure_weather_bars(weff: pd.DataFrame, metric_key: str) -> Figure:
    e = weff[weff.metric == metric_key].sort_values("rel_pct")
    fig = Figure(figsize=(7, 3.6), layout="constrained")
    ax = fig.subplots()
    if e.empty:
        ax.text(.5, .5, "No scenarios in this batch", ha="center", transform=ax.transAxes)
        return fig
    colors = [PASS if v.startswith("pass") else FAIL if v.startswith("FAIL") else NEUTRAL for v in e.verdict]
    err = 100 * e["ci95"].fillna(0) / e["base"].abs().replace(0, np.nan)
    ax.barh(e.scenario, e.rel_pct, xerr=err.fillna(0), color=colors, alpha=.85, capsize=3)
    ax.axvline(0, color="#333", lw=.8)
    ax.set_xlabel(f"% change vs baseline: {sens.METRIC_LABEL[metric_key]}")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def figure_weather_daily(daily, runs, col) -> Figure:
    fig = Figure(figsize=(9, 6.6), layout="constrained")
    ax_r, ax_e, ax_d = fig.subplots(3, 1, sharex=True)
    base = runs[runs.trait.isna()]
    weathers = sorted(base.scenario.unique(), key=lambda w: (w != "baseline", w))
    cmap = matplotlib.colormaps["tab10"]
    for i, w in enumerate(weathers):
        color = "#222" if w == "baseline" else cmap(i % 10)
        ids = base[base.scenario == w].run_id
        d = daily[daily.run_id.isin(ids)].groupby("day").mean(numeric_only=True)
        ax_r.plot(d.index, d.rain_mm.cumsum(), color=color, lw=1.4, label=w)
        ax_e.plot(d.index, d.et0_mm.rolling(7, min_periods=1).mean(), color=color, lw=1.4)
        ax_d.plot(d.index, d[col].rolling(7, min_periods=1).mean(), color=color, lw=1.4)
    ax_r.set_ylabel("cumulative rain (mm)")
    ax_e.set_ylabel("ET0, 7-day mean (mm/d)")
    ax_d.set_ylabel(col.replace("_m3", " (m³/d)").replace("_", " ") + ", 7-day mean")
    ax_d.set_xlabel("season day (0 = 17 April)")
    ax_r.legend(fontsize=8, frameon=False, ncol=4)
    for ax in (ax_r, ax_e, ax_d):
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def figure_interactions(effects: pd.DataFrame, metric_key: str) -> Figure:
    e = effects[effects.metric == metric_key]
    piv = e.pivot_table(index="trait", columns="scenario", values="rel_pct")
    fig = Figure(figsize=(8, max(3.5, .36 * len(piv) + 1.5)), layout="constrained")
    ax = fig.subplots()
    if piv.shape[1] < 2:
        ax.text(.5, .5, "Run the batch with --interactions to fill this view",
                ha="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    cols = ["baseline"] + [c for c in piv.columns if c != "baseline"]
    piv = piv[cols]
    lim = float(np.nanmax(np.abs(piv.values))) or 1.
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(piv)), piv.index)
    for (i, j), v in np.ndenumerate(piv.values):
        if np.isfinite(v):
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > .6 * lim else "#222")
    fig.colorbar(im, ax=ax, label="high − low, % of base run")
    ax.set_title(f"Trait effect by scenario · {sens.METRIC_LABEL[metric_key]}", fontsize=11, loc="left")
    return fig


def figure_noise(runs: pd.DataFrame, metric_key: str) -> Figure:
    base = runs[runs.trait.isna() & (runs.scenario == "baseline")].sort_values("seed")
    fig = Figure(figsize=(6, 3.2), layout="constrained")
    ax = fig.subplots()
    ax.plot(base.seed.astype(str), base[metric_key], "o", color=BASE)
    ax.axhline(base[metric_key].mean(), color=BASE, lw=.8, ls="--")
    ax.set_xlabel("seed")
    ax.set_ylabel(f"{sens.METRIC_LABEL[metric_key]} ({sens.METRIC_UNIT[metric_key]})")
    ax.set_title("Base runs on baseline weather: the model's own noise", fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


# --- tabs ----------------------------------------------------------------------------------

def _metric_select():
    solara.Select(label="Output", value=metric, values=[k for k, _, _ in sens.METRICS])


@solara.component
def VerdictsTab():
    d = data.value
    te, we = d["trait_effects"], d["weather_effects"]
    checked = te[te.expected.notna() & te.scenario.isin(["baseline", "baseline" + sens.SHORTAGE])]
    wchecked = we[we.expected.notna()]
    allv = pd.concat([checked.verdict, wchecked.verdict])
    n_pass = int(allv.str.startswith("pass").sum())
    n_fail = int(allv.str.startswith("FAIL").sum())
    n_leak = int(allv.str.startswith("LEAK").sum())
    with solara.Div(classes=["sens-panel"]):
        solara.Markdown(
            f"**{n_pass} pass · {n_fail} fail · {n_leak} leak** of {len(allv)} expectations.  \n"
            "An effect counts when the paired high − low difference is significant across seeds "
            f"(p < 0.05) and at least 1 % of the base value. Expected signs are read off the rules in "
            "`abm/farmer.py` and `abm/weather.py` (`EXPECTED`, `EXPECTED_SHORTAGE`, `EXPECTED_WEATHER` in "
            "`abm/sensitivity.py`). The Trait guide tab explains each expectation.")
        solara.Markdown("**Traits (baseline weather, normal supply and pump shortage)**")
        cols = [("trait", "Trait"), ("scenario", "Scenario"), ("metric", "Output"), ("expected", "Expected"), ("low", "Low"),
                ("high", "High"), ("diff", "High − low"), ("rel_pct", "% of base"), ("p", "p"),
                ("n", "Seeds"), ("verdict", "Verdict")]
        solara.HTML(tag="div", unsafe_innerHTML=_table(checked.sort_values(["verdict", "trait", "scenario"]), cols))
        solara.Markdown("**Scenarios (vs baseline weather and normal supply)**")
        wcols = [("scenario", "Scenario"), ("metric", "Output"), ("expected", "Expected"), ("base", "Baseline"),
                 ("value", "Scenario"), ("rel_pct", "% change"), ("p", "p"), ("n", "Seeds"), ("verdict", "Verdict")]
        solara.HTML(tag="div", unsafe_innerHTML=_table(wchecked.sort_values(["verdict", "scenario"]), wcols))
    with solara.Div(classes=["sens-panel"]):
        solara.Markdown("**Traits with no prior: what they did**")
        free = te[te.expected.isna() & te.scenario.isin(["baseline", "baseline" + sens.SHORTAGE])
                  & (te.metric == metric.value)]
        _metric_select()
        cols = [("trait", "Trait"), ("scenario", "Scenario"), ("metric", "Output"), ("low", "Low"), ("high", "High"),
                ("diff", "High − low"), ("rel_pct", "% of base"), ("signal_to_noise", "|effect| / noise sd"),
                ("p", "p"), ("verdict", "Verdict")]
        solara.HTML(tag="div", unsafe_innerHTML=_table(free.sort_values("rel_pct", key=abs, ascending=False), cols))


@solara.component
def TraitsTab():
    d = data.value
    weathers = sorted(d["trait_effects"].scenario.unique(), key=lambda w: (w != "baseline", w))
    with solara.Div(classes=["sens-panel"]):
        with solara.Row():
            _metric_select()
            solara.Select(label="Scenario", value=weather_sel, values=weathers)
        solara.Markdown("Bar colour = verdict (green pass, red fail, amber leak, grey no prior). "
                        "Grey band = seed-to-seed spread of the base run: a bar inside it would be lost "
                        "in noise without the paired design.")
        solara.FigureMatplotlib(figure_tornado(d["trait_effects"], d["runs"], metric.value, weather_sel.value),
                                dependencies=[metric.value, weather_sel.value])


@solara.component
def TraitDetailTab():
    d = data.value
    runs = d["runs"]
    traits = sorted(runs.trait.dropna().unique())
    weathers = sorted(runs[runs.trait.notna()].scenario.unique(), key=lambda w: (w != "baseline", w))
    daily_cols = ["request_m3", "applied_m3", "drainage_m3", "unmet_m3", "withdraw_m3", "mean_ponding_mm", "stress_mm"]
    col, set_col = solara.use_state("request_m3")
    with solara.Div(classes=["sens-panel"]):
        with solara.Row():
            solara.Select(label="Trait", value=trait_sel, values=traits)
            solara.Select(label="Scenario", value=weather_sel, values=weathers)
            solara.Select(label="Daily series", value=col, values=daily_cols, on_value=set_col)
        t = next((x for x in sens.TRAITS if x.name == trait_sel.value), None)
        if t:
            solara.Markdown(f"**{t.name}** ({t.low} … {t.high}, default {t.default}): {t.meaning}")
    with solara.Div(classes=["sens-panel"]):
        solara.FigureMatplotlib(figure_trait_daily(d["daily"], runs, trait_sel.value, weather_sel.value, col),
                                dependencies=[trait_sel.value, weather_sel.value, col])
    with solara.Div(classes=["sens-panel"]):
        _metric_select()
        solara.FigureMatplotlib(figure_pairs(runs, trait_sel.value, weather_sel.value, metric.value),
                                dependencies=[trait_sel.value, weather_sel.value, metric.value])


@solara.component
def WeatherTab():
    d = data.value
    daily_cols = ["request_m3", "applied_m3", "drainage_m3", "unmet_m3", "withdraw_m3", "mean_ponding_mm", "stress_mm"]
    col, set_col = solara.use_state("request_m3")
    with solara.ColumnsResponsive(5, 7, gutters=True):
        with solara.Div(classes=["sens-panel"]):
            _metric_select()
            solara.FigureMatplotlib(figure_weather_bars(d["weather_effects"], metric.value),
                                    dependencies=[metric.value])
            we = d["weather_effects"]
            tot = we.drop_duplicates("scenario")[["scenario", "rain_total_mm", "et0_total_mm"]]
            solara.HTML(tag="div", unsafe_innerHTML=_table(
                tot, [("scenario", "Scenario"), ("rain_total_mm", "Season rain (mm)"),
                      ("et0_total_mm", "Season ET0 (mm)")], verdict_col=None))
        with solara.Div(classes=["sens-panel"]):
            solara.Select(label="Daily series", value=col, values=daily_cols, on_value=set_col)
            solara.FigureMatplotlib(figure_weather_daily(d["daily"], d["runs"], col), dependencies=[col])


@solara.component
def InteractionsTab():
    d = data.value
    with solara.Div(classes=["sens-panel"]):
        _metric_select()
        solara.Markdown("Does a trait matter more in some weather, or when the canal is short? Each cell is the paired high − low "
                        "effect as a % of that scenario's base run.")
        solara.FigureMatplotlib(figure_interactions(d["trait_effects"], metric.value), dependencies=[metric.value])


@solara.component
def NoiseTab():
    d = data.value
    with solara.ColumnsResponsive(6, 6, gutters=True):
        with solara.Div(classes=["sens-panel"]):
            solara.Markdown(
                "**Seeds needed** (80 % power, α = 0.05) to detect a 1 % or 5 % change with the paired "
                "design, from the median sd of the paired differences. If it exceeds the seeds you ran, "
                "a \"no effect\" verdict on that output is not conclusive (Seri & Secchi 2017).")
            cols = [("metric", "Output"), ("mean", "Mean"), ("sd", "SD across seeds"), ("cv_pct", "CV %"),
                    ("paired_diff_sd", "SD of paired diff"), ("seeds_for_1pct", "Seeds for 1 %"),
                    ("seeds_for_5pct", "Seeds for 5 %")]
            noise = d["noise"]
            solara.HTML(tag="div", unsafe_innerHTML=_table(noise, [c for c in cols if c[0] in noise.columns],
                                                           verdict_col=None))
        with solara.Div(classes=["sens-panel"]):
            _metric_select()
            solara.FigureMatplotlib(figure_noise(d["runs"], metric.value), dependencies=[metric.value])


# Plain-words explanation of each trait (the mechanism in the code is Trait.meaning).
PLAIN = {
    "experience": (
        "Years of growing rice. After a day when its crop was short of water, a learning farmer "
        "raises the water depth it aims for. An experienced farmer reacts about half as much, "
        "treating one bad day as bad luck rather than a trend.",
        "Only acts when crops are actually stressed, which never happens at full pump capacity. "
        "It is tested under the pump shortage."),
    "water_accounting": (
        "How carefully it budgets water. At 1 it notices that its paddy also loses water into the "
        "ground (seepage) and tops up for that, and it counts the rain that fell today so it does "
        "not irrigate on top of it. At 0 it only replaces what the crop uses.",
        "Pulls both ways (more for seepage, less for rain), so there is no expected direction."),
    "cooperation": (
        "How much it fits in with its neighbours. A cooperative farmer lets its water level drift "
        "toward the neighbours' and, when the canal cannot supply everyone, asks for up to 25 % "
        "less so others get some. An uncooperative one follows its own preference instead.",
        "Under the pump shortage it should lower demand. With full supply the levelling can go "
        "either way."),
    "work_ethic": (
        "How often it walks its fields: 1 = every day, 0.5 = every 3 days, 0 = every 5 days. "
        "Between visits the inlet and weir stay as it left them. It replaces the earlier labour "
        "and work ethic, which the model only ever used together.",
        "More frequent checks mean smaller, more frequent top-ups. That can mean less waste or "
        "more water, so there is no expected direction."),
    "risk_tolerance": (
        "How much safety margin it keeps. A cautious farmer (0) keeps 8 mm of extra water above "
        "the crop's ideal depth in case of a hot spell or a missed visit. A risk-tolerant one (1) "
        "keeps none.",
        "Higher tolerance means less demand and shallower water."),
    "neighbour_wetness_bias": (
        "Whether it likes its paddies wetter (+) or drier (−) than the ones next door, by up to "
        "15 mm. It only acts to the extent the farmer is uncooperative.",
        "Higher means deeper water."),
    "calendar_lead": (
        "Whether it plants early or late compared with the district: +1 is 4 days ahead, −1 is 4 "
        "days behind. The crop's water needs, the mid-season drain and the harvest drain all move "
        "with it.",
        "An early farmer should reach peak demand earlier."),
    "drawdown_tolerance": (
        "How low it lets the water fall before refilling. At 1 the pond can drop 25 mm below the "
        "target before the inlet opens. At 0 it refills as soon as the level dips.",
        "Higher means shallower water on average."),
    "refill_overshoot": (
        "How high it fills when it does refill: up to 20 mm above the target. This is also where "
        "it sets the outlet weir board, so water above it spills to the drain.",
        "Higher means more demand. Average depth can even fall, because fewer, bigger refills "
        "leave the pond low for longer."),
    "off_farm_work": (
        "Share of working days it is away at another job; most Japanese rice farmers farm part "
        "time. It always checks on its days off. On away days whatever it left running keeps "
        "running.",
        "Fewer visits means fewer refills, but inlets may be left open, so there is no expected "
        "direction."),
    "forecast_reliance": (
        "How far ahead it trusts the rain forecast. At 0 it ignores the forecast. At 1 it looks 4 "
        "days ahead, trusting each later day 20 % less, and fills less to leave room for the rain.",
        "Higher means less demand."),
    "flow_through_habit": (
        "Kakenagashi: leaving the inlet open so water keeps flowing through the paddy, to cool the "
        "crop on hot days and as insurance while away. The extra water leaves over the weir to the "
        "drain.",
        "Higher means more demand and more drainage."),
}


def _arrow(sign):
    return {1: "↑", -1: "↓", 0: "no change"}[sign]


@solara.component
def TraitGuideTab():
    from abm.agents import ARCHETYPES
    from abm.farmer import FARMER_DEFAULTS, RENAMED_TRAITS, RETIRED_TRAITS

    d = data.value
    te = d["trait_effects"] if d else None
    with solara.Div(classes=["sens-panel"]):
        solara.Markdown(
            "## The farmer personality traits\n"
            "Every farmer carries these traits. Each farmer type (below) has its own average, and "
            "every farmer gets a small random offset from it (`trait_variability`, 8 % of the range), "
            "so no two farmers are identical. The code is in `abm/farmer.py` (`TRAITS`), and the "
            "farmer-type averages are in `abm/agents.py` (`ARCHETYPES`).\n\n"
            "**How a trait is tested:** in the sensitivity batch every farmer's trait is set to its "
            "lowest value, then its highest, keeping everything else identical for the same seed. "
            "*Effect on demand* below is that high − low difference as a % of the season's demand, "
            "with normal supply and under the pump shortage.")
    for t in sens.TRAITS:
        plain, test = PLAIN.get(t.name, ("", ""))
        exp = dict(sens.EXPECTED.get(t.name, {}))
        exp_short = sens.EXPECTED_SHORTAGE.get(t.name, {})
        exp_text = ", ".join(f"{sens.METRIC_LABEL[m].lower()} {_arrow(v)}" for m, v in exp.items())
        if exp_short:
            exp_text += ("; " if exp_text else "") + "under pump shortage: " + ", ".join(
                f"{sens.METRIC_LABEL[m].lower()} {_arrow(v)}" for m, v in exp_short.items())
        results = ""
        if te is not None and not te.empty:
            parts = []
            for scen, label in (("baseline", "normal supply"), ("baseline" + sens.SHORTAGE, "pump shortage")):
                row = te[(te.trait == t.name) & (te.scenario == scen) & (te.metric == "request_m3")]
                if len(row):
                    r = row.iloc[0]
                    parts.append(f"{label} **{r.rel_pct:+.1f} %** (p = {_fmt(r.p)})")
            if parts:
                results = "  \n*Effect on demand in this batch (high − low):* " + " · ".join(parts)
        old = [o for o, n in RENAMED_TRAITS.items() if n == t.name]
        with solara.Div(classes=["sens-panel"]):
            solara.Markdown(
                f"### {t.name}\n"
                f"Range {t.low:g} … {t.high:g}, default {t.default:g}"
                + (f" · earlier name `{old[0]}`" if old else "") + "\n\n"
                f"{plain}\n\n"
                f"*In the code:* {t.meaning}.  \n"
                f"*Expected:* {exp_text or 'no prior'}. {test}"
                + results)
    with solara.Div(classes=["sens-panel"]):
        solara.Markdown("### Each farmer type's average traits")
        names = [t.name for t in sens.TRAITS]
        rows = [{"type": k, **{n: v["traits"].get(n, np.nan) for n in names}} for k, v in ARCHETYPES.items()]
        frame = pd.DataFrame(rows)
        head = "".join(f"<th>{c.replace('_', ' ')}</th>" for c in frame.columns)
        body = "".join("<tr>" + "".join(f"<td>{v if isinstance(v, str) else f'{v:g}'}</td>" for v in r) + "</tr>"
                       for r in frame.itertuples(index=False))
        solara.HTML(tag="div", unsafe_innerHTML=f'<div style="overflow-x:auto"><table class="sens-table">'
                                                  f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
        solara.Markdown(
            "### Removed traits\n"
            f"`{'`, `'.join(RETIRED_TRAITS)}`: `wealth` and `price_sensitivity` only acted on a per-m³ "
            "water price, which Japanese land-improvement districts do not charge. `labour` was "
            "merged into `work_ethic` because the check interval only ever used the two together. "
            "Old names are still accepted in code, and a given `labour` is folded into `work_ethic`.\n\n"
            "### Shared settings behind the traits\n"
            "The trait sets how far along each scale a farmer sits. These settings in "
            "`FARMER_DEFAULTS` set the scale itself: "
            + ", ".join(f"`{k}` = {FARMER_DEFAULTS[k]:g}" for k in (
                "max_shift_days", "lower_gap_mm", "upper_gap_mm", "risk_buffer_mm", "max_forecast_days",
                "forecast_decay", "heat_et0_mm", "heat_flow_mm", "away_slack", "scarcity_cut")) + ".")


TABS = {
    "Trait guide": TraitGuideTab,
    "Verdicts": VerdictsTab,
    "Traits": TraitsTab,
    "Trait detail": TraitDetailTab,
    "Scenarios": WeatherTab,
    "Interactions": InteractionsTab,
    "Noise & runs": NoiseTab,
}


@solara.component
def Page():
    solara.Style(CSS)
    solara.Title("ABM sensitivity")
    d = data.value
    if d is None:
        with solara.Div(classes=["sens-panel"]):
            solara.Markdown("## No sensitivity results yet\nRun `python scripts/abm_sensitivity.py` "
                            "to fill the other tabs.")
        TraitGuideTab()
        return
    meta = d["meta"]
    runs = d["runs"]
    with solara.Div(classes=["sens-panel"]):
        solara.Markdown(
            f"## ABM sensitivity tests\n"
            f"`{d['path']}` · {len(runs)} season runs · seeds {meta.get('seeds')} · year {meta.get('year')} · "
            f"{int(runs.n_paddies.iloc[0]):,} paddies, {int(runs.n_farmers.iloc[0])} farmers · "
            f"{'cyclic map' if meta.get('use_cyclic_map') else 'toy grid (weather presets have no effect)'} · "
            f"{meta.get('minutes', 0):.1f} min")
    names = list(TABS)
    with solara.lab.Tabs(value=selected_tab.value, on_value=selected_tab.set):
        for name in names:
            solara.lab.Tab(name)
    TABS[names[selected_tab.value]]()


def serve(path: str | None = None, port: int = 8771, open_browser: bool = True):
    """Open this viewer on ``path`` (default: the newest batch). Blocks until Ctrl+C."""
    import cyclic

    if path:
        os.environ[RESULTS_ENV] = os.path.abspath(path)
    return cyclic.serve(script=os.path.abspath(__file__), port=port, open_browser=open_browser)


Page  # noqa: B018 — Solara discovers Page in module scope

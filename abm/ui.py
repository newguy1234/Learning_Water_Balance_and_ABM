# -*- coding: utf-8 -*-
"""Solara UI for the ABM field.

    python -m abm ui
    # or with your own farmers / station registered:
    python example.py abm

Tabs
    Field       district map (ownership / irrigation / drainage / lateral flow / …) + neighbours
    North field the North area only (cyclic + non-cyclic), zoomed, with its own summary
    South field the same for the South area
    Run         levers, farmer-personality mix, station policy, Reset / Run / Step
    Metrics     season time series + the efficiency table the EMS paper reports

Built on the cyclic satellite paddy map and the discovered underground leftover
shares (joint capture / neighbour hops / stay). LID staff send pool water;
farmers apply ≤ send. Crop demand and ET are homegrown FAO-56/33 (Kc curve +
ponding-stress Ks + Ky yield response) driven by what the farmer actually
applied plus rain — the cyclic satellite ET target is not fed into the model.
Registered agents (including those defined outside ``abm``) appear in the
menus automatically.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import matplotlib

matplotlib.use("Agg")

import numpy as np
import solara
import solara.lab

from abm.agents import list_agents
from abm.field import FieldSession
from abm.metrics import season_efficiencies
from abm.plots import (
    figure_et,
    figure_field,
    figure_metrics,
    figure_neighbours,
    figure_series,
)
from abm.station import list_stations

# Reactive state ---------------------------------------------------------------
session = solara.reactive(FieldSession())
map_mode = solara.reactive("ponding")
selected_paddy = solara.reactive(0)
status_text = solara.reactive("Ready — adjust levers, then Run or Step.")
busy = solara.reactive(False)
playing = solara.reactive(False)
metrics = solara.reactive({})
step_chunk = solara.reactive(5)
play_delay_s = solara.reactive(0.2)
tick = solara.reactive(0)  # force redraw
selected_tab = solara.reactive("Field")

MAP_LAYERS = [
    ("ponding", "Ponding"),
    ("ownership", "Ownership"),
    ("irrigation", "Irrigation applied"),
    ("drainage", "Drainage"),
    ("et", "ET (actual)"),
    ("leftover", "Leftover"),
    ("hops", "Lateral flow"),
    ("yield", "Yield"),
    ("stress", "Water stress (unmet ET)"),
    ("capture", "Capture to pool (LFV)"),
]


def _refresh():
    tick.value = tick.value + 1


def _safe(fn):
    def wrapped(*a, **k):
        if busy.value or playing.value:
            return
        busy.value = True
        try:
            fn(*a, **k)
        except Exception:
            status_text.value = traceback.format_exc(limit=4)
        finally:
            busy.value = False
            _refresh()

    return wrapped


def _status_after_step(s):
    st = s.status()
    status_text.value = (
        f"Day {st['day']}/{st['days']} · {st['stage']} Kc={st['kc']:.2f} · "
        f"target {st['target_depth_mm']:.0f} mm · pool {st['pool_m3']:,.0f} m³ · "
        f"Ya/Ym {st['mean_relative_yield']:.2f} · station {st['station']}/{st['station_state']}"
    )


@_safe
def do_reset():
    s = session.value
    s.reset()
    metrics.value = {}
    status_text.value = (
        f"Reset · {s.world.n} paddies · {s.world.n_farmers()} farmers · "
        f"{s.paddies_per_farmer} paddies/farmer"
    )


@_safe
def do_step():
    s = session.value
    n = int(step_chunk.value)
    for _ in range(n):
        if s.world.done():
            break
        s.step()
    if s.world.history:
        metrics.value = season_efficiencies(s.world)
    _status_after_step(s)


def _play_loop():
    s = session.value
    n = max(int(step_chunk.value), 1)
    delay = max(float(play_delay_s.value), 0.0)
    try:
        while playing.value and not s.world.done():
            for _ in range(n):
                if s.world.done():
                    break
                s.step()
            if s.world.history:
                metrics.value = season_efficiencies(s.world)
            _status_after_step(s)
            _refresh()
            if s.world.done():
                break
            time.sleep(delay)
    finally:
        playing.value = False
        _refresh()


def do_play():
    if playing.value or busy.value:
        return
    playing.value = True
    threading.Thread(target=_play_loop, daemon=True).start()


def do_pause():
    playing.value = False


def on_paddies_per_farmer(value):
    s = session.value
    s.set_paddies_per_farmer(int(value))
    selected_paddy.value = min(selected_paddy.value, s.world.n - 1)
    status_text.value = (
        f"Ownership · {s.world.n_farmers()} farmers · "
        f"{s.paddies_per_farmer} paddies each"
    )
    _refresh()


def on_farm_size_variability(value):
    s = session.value
    s.set_farm_size_variability(float(value))
    selected_paddy.value = min(selected_paddy.value, s.world.n - 1)
    counts = np.bincount(s.world.owner_id)
    status_text.value = (
        f"Farm sizes · {s.world.n_farmers()} farmers · "
        f"{counts.min()}-{counts.max()} paddies each (avg {s.paddies_per_farmer})"
    )
    _refresh()


def on_weather_variability(value):
    s = session.value
    s.set_weather_variability(float(value))
    metrics.value = {}
    selected_paddy.value = 0
    status_text.value = (
        f"Weather variability {float(value):.2f} · rebuilt with a fresh "
        f"{'randomised' if value else 'historical'} season"
    )
    _refresh()


def on_roster_change(names: list[str]):
    s = session.value
    s.set_roster(list(names))
    s.set_roster_weights({})  # equal split on a fresh roster
    status_text.value = f"Roster · {s.status()['roster']}"
    _refresh()


def on_roster_weight(name: str, pct: float):
    s = session.value
    weights = {**s.effective_roster_weights(), name: max(float(pct), 0.0) / 100.0}
    s.set_roster_weights(weights)
    _refresh()


def on_station_change(name: str):
    s = session.value
    s.set_station(str(name))
    status_text.value = f"Station staff · {s.station_name}"
    _refresh()


def on_select_paddy(value):
    w = session.value.world
    selected_paddy.value = max(0, min(int(value), max(w.n - 1, 0)))
    _refresh()


CSS = """
.abm-shell {
  font-family: "Source Serif 4", "Iowan Old Style", Georgia, serif;
  background: linear-gradient(165deg, #e8f0ec 0%, #f7f4ef 42%, #efe6d8 100%);
  min-height: 100vh;
  color: #1c2421;
}
.abm-hero {
  padding: 1.4rem 1.6rem 0.6rem;
  border-bottom: 1px solid rgba(27, 74, 66, 0.18);
}
.abm-hero h1 {
  margin: 0;
  font-size: 1.85rem;
  letter-spacing: 0.02em;
  color: #1b4a42;
  font-weight: 650;
}
.abm-hero p {
  margin: 0.35rem 0 0;
  max-width: 46rem;
  color: #3d4f49;
  font-size: 0.98rem;
}
.abm-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
  margin: 0.75rem 0 0.15rem;
}
.abm-stat {
  background: rgba(255, 252, 247, 0.9);
  border: 1px solid rgba(27, 74, 66, 0.14);
  border-radius: 8px;
  padding: 0.35rem 0.85rem;
  min-width: 6.5rem;
}
.abm-stat .k {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: #5a6e67;
}
.abm-stat .v {
  font-size: 1.12rem;
  font-weight: 700;
  color: #1b4a42;
  font-variant-numeric: tabular-nums;
}
.abm-status {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.8rem;
  color: #2f4a43;
  white-space: pre-wrap;
  margin-top: 0.4rem;
}
.abm-panel {
  background: rgba(255, 252, 247, 0.82);
  border: 1px solid rgba(27, 74, 66, 0.12);
  border-radius: 10px;
  padding: 0.85rem 1rem;
  margin: 0.4rem;
}
.abm-label {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #5a6e67;
  margin-bottom: 0.25rem;
}
.abm-table {
  border-collapse: collapse;
  font-size: 0.86rem;
  width: 100%;
}
.abm-table th, .abm-table td {
  border-bottom: 1px solid rgba(27, 74, 66, 0.14);
  padding: 0.3rem 0.7rem;
  text-align: right;
}
.abm-table th:first-child, .abm-table td:first-child {
  text-align: left;
}
.abm-table thead th {
  color: #5a6e67;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
"""


def _stat(key: str, value: str) -> str:
    return f'<div class="abm-stat"><div class="k">{key}</div><div class="v">{value}</div></div>'


@solara.component
def Header():
    _ = tick.value
    s = session.value
    w = s.world
    st = s.status()
    solara.HTML(
        tag="div",
        unsafe_innerHTML=(
            "<h1>Imbanuma field · ABM</h1>"
            "<p>Built on the <strong>cyclic</strong> satellite paddy map and the "
            "discovered underground leftover rules (joint capture / neighbour hops / stay). "
            "LID staff send pool water; farmers apply ≤ send. FAO-56 rice + FAO-33 yield.</p>"
        ),
    )
    tiles = "".join([
        _stat("Day", f"{st['day']} / {st['days']}"),
        _stat("Stage", st["stage"].replace("_", " ").title()),
        _stat("Paddies", f"{w.n:,} ({int(w.cyclic.sum()):,} cyclic)"),
        _stat("Farmers", f"{w.n_farmers():,}"),
        _stat("Pool", f"{w.pool_m3:,.0f} m³"),
        _stat("Ya/Ym", f"{w.relative_yield.mean():.2f}"),
        _stat("Groundwater", "stem.water ratios"),
    ])
    solara.HTML(tag="div", classes=["abm-stats"], unsafe_innerHTML=tiles)
    solara.HTML(tag="div", classes=["abm-status"], unsafe_innerHTML=status_text.value)


@solara.component
def RunControls():
    _ = tick.value
    w = session.value.world
    done = w.done()

    with solara.Row(style={"align-items": "center"}, gap="10px"):
        if playing.value:
            solara.Button("Pause", icon_name="mdi-pause", color="primary", on_click=do_pause)
        else:
            solara.Button(
                "Run" if not done else "Finished",
                icon_name="mdi-play",
                color="primary",
                disabled=busy.value or done,
                on_click=do_play,
            )
        solara.Button("Step", icon_name="mdi-skip-next", disabled=busy.value or playing.value or done,
                      on_click=do_step)
        solara.Button("Reset", icon_name="mdi-restart", disabled=busy.value or playing.value,
                      on_click=do_reset)
        solara.Text("Days/frame", classes=["abm-label"])
        solara.SliderInt(label="", value=step_chunk, min=1, max=30)
        solara.Text("Sec/frame", classes=["abm-label"])
        solara.SliderFloat(label="", value=play_delay_s, min=0.0, max=1.5, step=0.05)


@solara.component
def FieldTab():
    _ = tick.value
    s = session.value
    w = s.world

    with solara.Div(classes=["abm-panel"]):
        RunControls()

    with solara.ColumnsResponsive(4, 8, gutters=True):
        with solara.Div(classes=["abm-panel"]):
            solara.Markdown("**Map layer**")
            solara.ToggleButtonsSingle(
                value=map_mode,
                values=[key for key, _ in MAP_LAYERS],
            )
            solara.Markdown("**Inspect paddy**")
            with solara.Row(style={"align-items": "center"}, gap="10px"):
                solara.InputInt(
                    label="Selected paddy id",
                    value=selected_paddy.value,
                    on_value=on_select_paddy,
                )
                solara.Text(f"range 0–{max(w.n - 1, 0):,}", classes=["abm-label"])
            oid = int(w.owner_id[selected_paddy.value])
            nbrs = w.neighbour_ids(selected_paddy.value)
            is_cyclic = bool(w.cyclic[selected_paddy.value])
            solara.Markdown(
                f"Owner **F{oid}** (`{s.status()['roster'].get(oid, '?')}`) · "
                f"{'cyclic' if is_cyclic else 'non-cyclic'} · "
                f"{w.pool_distance_m[selected_paddy.value]:,.0f} m from the pool · "
                f"neighbours `{nbrs}`"
            )
            solara.FigureMatplotlib(
                figure_neighbours(w, selected_paddy.value),
                dependencies=[tick.value, selected_paddy.value],
            )
        with solara.Div(classes=["abm-panel"]):
            solara.FigureMatplotlib(
                figure_field(w, mode=map_mode.value, selected=selected_paddy.value),
                dependencies=[tick.value, map_mode.value, selected_paddy.value],
            )


@solara.component
def SettingsTab():
    _ = tick.value
    s = session.value
    w = s.world
    agents = list_agents()
    roster = list(s.agent_roster)
    weights = s.effective_roster_weights()

    with solara.ColumnsResponsive(6, 6, gutters=True):
        with solara.Div(classes=["abm-panel"]):
            solara.Markdown("**Levers**")
            solara.Text("Paddies per farmer (average holding)", classes=["abm-label"])
            solara.SliderInt(
                label="",
                value=s.paddies_per_farmer,
                min=1,
                max=max(w.n, 2),
                on_value=on_paddies_per_farmer,
                disabled=playing.value,
            )
            solara.Text(
                "Farm size variability (0 = fixed size; 1 = holdings range 0.5×–2× the average)",
                classes=["abm-label"],
            )
            solara.SliderFloat(
                label="",
                value=s.farm_size_variability,
                min=0.0,
                max=1.0,
                step=0.05,
                on_value=on_farm_size_variability,
                disabled=playing.value,
            )
            _counts = np.bincount(w.owner_id) if w.n else np.array([0])
            solara.Markdown(
                f"This run: **{w.n_farmers()}** farmers, holdings **{_counts.min()}–{_counts.max()}** "
                "paddies each" + (" (all equal)" if s.farm_size_variability <= 0 else "")
            )
            solara.Text("Weather variability (0 = exact historical year)", classes=["abm-label"])
            solara.SliderFloat(
                label="",
                value=s.weather_variability,
                min=0.0,
                max=0.6,
                step=0.05,
                on_value=on_weather_variability,
                disabled=playing.value,
            )
            solara.Markdown(
                "Rain/ET0 no farmer or station policy controls — 0 replays "
                f"{w.config.year}'s Open-Meteo record exactly (comparable to the "
                "paper's pins); higher jitters it day-to-day so every reset grows "
                "a different season."
            )

        with solara.Div(classes=["abm-panel"]):
            solara.Markdown("**LID station staff** (pool / lake control)")
            solara.Select(
                label="Station policy",
                value=s.station_name,
                values=list_stations(),
                on_value=on_station_change,
                disabled=playing.value,
            )
            solara.Markdown(
                f"State **{s.last_station_state}** · pool "
                f"**{s.world.pool_m3:,.0f}** m³ · lake **{s.world.lake_m3:,.0f}** m³"
            )
            solara.Markdown("**Farmer personality mix** (share of the {} farmers)".format(w.n_farmers()))
            solara.SelectMultiple(
                label="Farmer types in rotation",
                values=roster,
                all_values=agents,
                on_value=on_roster_change,
                disabled=playing.value,
            )
            solara.Markdown(
                "Personality is an initial tendency: farmers learn from their own outcomes "
                "and may copy more successful neighbouring owners using yesterday's observations. "
                "Numerical learning settings are scenario assumptions from the conceptual model."
            )
            policies = [a.policy for a in s.model.farmer_agents.values()]
            learned = [p for p in policies if hasattr(p, "copy_events")]
            if learned:
                solara.Markdown(
                    f"**Social learning:** {sum(p.copy_events for p in learned)} copying events · "
                    f"{sum(p.copied_from is not None for p in learned)} farmers copied on the latest day · "
                    f"mean cooperation {np.mean([p.cooperation for p in learned]):.2f}"
                )
            for name in roster:
                pct = round(weights.get(name, 0.0) * 100.0)
                solara.SliderInt(
                    label=f"{name} ({pct}%)",
                    value=pct,
                    min=0,
                    max=100,
                    disabled=playing.value,
                    on_value=lambda v, name=name: on_roster_weight(name, v),
                )


def _fmt(value, digits=3) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{digits}f}"


def _paper_table_html(eff: dict) -> str:
    rows = [
        ("WUE′ (%) — Eq. 11", eff.get("WUE_prime_pct"), None, 1),
        ("RUE, North — Eq. 12", eff.get("RUE_North"), None, 3),
        ("RUE, South — Eq. 12", eff.get("RUE_South"), None, 3),
        ("RUE, all — Eq. 13", eff.get("RUE_all"), eff.get("paper_RUE_all"), 3),
        ("LUE — Eq. 14", eff.get("LUE"), eff.get("paper_LUE"), 3),
        ("Relative yield Ya/Ym (FAO-33)", eff.get("relative_yield"), None, 3),
        ("Irrigation applied (m³)", eff.get("IR_m3"), None, 0),
        ("Irrigation sent by station (m³)", eff.get("IR_sent_m3"), None, 0),
        ("Drainage D (m³)", eff.get("D_m3"), None, 0),
        ("Captured to pool, LFV (m³)", eff.get("LFV_m3"), None, 0),
        ("ET, actual (mm)", eff.get("ET_mm"), None, 1),
        ("ETc, potential (mm)", eff.get("ETc_mm"), None, 1),
        ("Rain (mm)", eff.get("rain_mm"), None, 1),
    ]
    body = "".join(
        f"<tr><td>{label}</td><td>{_fmt(sim, digits)}</td><td>{_fmt(paper, digits)}</td></tr>"
        for label, sim, paper, digits in rows
    )
    return (
        '<table class="abm-table"><thead><tr><th>Quantity</th><th>This run</th>'
        f"<th>Paper (2022)</th></tr></thead><tbody>{body}</tbody></table>"
    )


def _farmer_table_html(frame):
    labels = {
        "cohort": "Cohort", "farmer_id": "Farmer", "farmer_type": "Personality",
        "farmer_periods": "Farm periods", "days": "Days", "area_ha": "Area (ha)",
        "score": "Balanced score", "relative_yield_pct": "Yield (%)", "mean_ks": "Mean Ks",
        "irrigation_mm": "Irrigation (mm)", "drainage_mm": "Drainage (mm)",
        "capture_mm": "Pool capture (mm)", "sent_used_pct": "Supply used (%)",
        "copying_events": "Copies", "cooperation": "Cooperation",
    }
    table = frame[[c for c in labels if c in frame]].rename(columns=labels)
    return '<div style="overflow-x:auto">' + table.to_html(
        index=False, classes="abm-table", border=0, escape=True,
        float_format=lambda v: f"{v:.3f}", na_rep="—") + '</div>'


@solara.component
def MetricsTab():
    _ = tick.value
    s = session.value
    w = s.world
    eff = metrics.value

    from abm.metrics import farmer_type_performance, best_farmers
    statuses, set_statuses = solara.use_state(["Cyclic", "Non-cyclic"])
    blocks, set_blocks = solara.use_state(["North", "South"])
    personalities, set_personalities = solara.use_state([])
    ranking, set_ranking = solara.use_state("Balanced score")
    ranks = {"Balanced score": "score", "Highest estimated yield": "relative_yield_pct",
             "Least crop stress": "mean_ks", "Lowest irrigation": "irrigation_mm",
             "Lowest drainage": "drainage_mm", "Highest pool capture": "capture_mm"}
    available_types = sorted(set(r['farmer_type'] for r in s.model.farmer_performance_history)
                             | set(s.agent_roster))
    with solara.Div(classes=["abm-panel"]):
        solara.Markdown("**Mix and compare farmer performance**")
        solara.SelectMultiple(label="Paddy status", values=statuses,
            all_values=["Cyclic", "Non-cyclic"], on_value=set_statuses)
        solara.SelectMultiple(label="Blocks", values=blocks,
            all_values=["North", "South"], on_value=set_blocks)
        solara.SelectMultiple(label="Personalities (empty = all)", values=personalities,
            all_values=available_types, on_value=set_personalities)
        solara.Select(label="Rank best by", value=ranking, values=list(ranks), on_value=set_ranking)
        solara.Markdown(
            "Filters combine. A farmer owning both statuses is evaluated separately on each "
            "subset of paddies. Within-farm metrics are area-weighted; type averages give each "
            "farm period equal weight. Copying and cooperation describe the farmer, not individual paddies. "
            "Depths accumulate across the displayed days; yield is estimated to date."
        )
        solara.Markdown(
            "Balanced score = mean daily Ks − 0.15 × irrigation/40 − 0.05 × drainage/30. "
            "Higher is better. Lowest irrigation alone does not imply good crop performance. "
            "Compare similar day counts; location and supply also affect results."
        )
        common = dict(blocks=blocks, farmer_types=personalities or None, rank_by=ranks[ranking])
        sections = [("Combined selection", statuses)] if len(statuses) > 1 else []
        sections += [(status + " paddies", [status]) for status in statuses]
        if not sections:
            solara.Markdown("_Select at least one paddy status._")
        for title, subset in sections:
            averages = farmer_type_performance(s.model, statuses=subset, **common)
            leaders = best_farmers(s.model, statuses=subset, **common)
            solara.Markdown(f"**{title} — averages by personality**")
            if averages.empty:
                solara.Markdown("_No recorded results match these filters. Run days or change the selection._")
            else:
                solara.HTML(tag="div", unsafe_innerHTML=_farmer_table_html(averages))
                solara.Markdown(f"**Best {title.lower()} farmers — top 10 · {ranking}**")
                solara.HTML(tag="div", unsafe_innerHTML=_farmer_table_html(leaders))
        solara.Markdown("Roster or ownership changes start a new cohort; past results keep their original labels.")
    with solara.ColumnsResponsive(6, 6, gutters=True):
        with solara.Div(classes=["abm-panel"]):
            solara.FigureMatplotlib(figure_series(w), dependencies=[tick.value])
        with solara.Div(classes=["abm-panel"]):
            solara.FigureMatplotlib(figure_et(w), dependencies=[tick.value])
    with solara.ColumnsResponsive(6, 6, gutters=True):
        with solara.Div(classes=["abm-panel"]):
            solara.FigureMatplotlib(
                figure_metrics(eff), dependencies=[tick.value, str(eff)]
            )
    with solara.Div(classes=["abm-panel"]):
        solara.Markdown(f"**Season ({eff.get('days', 0)} d, {eff.get('year', w.config.year)})** — "
                         "the quantities Okura et al. (2025) report, this run vs. the paper's 2022 pin.")
        if eff:
            solara.HTML(tag="div", unsafe_innerHTML=_paper_table_html(eff))
        else:
            solara.Markdown("_Run a few days to populate this table._")


def _block_summary(w, block: str) -> str:
    """One line of what is happening in one area (cyclic + non-cyclic paddies)."""
    m = np.asarray(w.block) == block
    if not m.any():
        return f"No paddies in the {block} field."
    a = w.area_m2[m]
    cyc = np.asarray(w.cyclic)[m]
    avg = lambda values: float(np.average(np.asarray(values)[m], weights=a))
    farmers = len(np.unique(np.asarray(w.owner_id)[m]))
    text = (f"**{int(m.sum()):,} paddies** · {a.sum() / 1e4:,.1f} ha "
            f"({a[cyc].sum() / 1e4:,.1f} ha cyclic, {a[~cyc].sum() / 1e4:,.1f} ha non-cyclic) · "
            f"{farmers} farmers · mean ponding **{avg(w.ponding_mm):.1f} mm**")
    if w.day > 0:
        text += (f" · yesterday: applied {avg(w.last_ir):.1f} mm, drained {avg(w.last_dr):.1f} mm, "
                 f"short of ET {avg(w.last_stress):.2f} mm")
    text += (f" · season stress **{avg(w.stress_cum):.1f} mm** · yield "
             f"**{avg(w.relative_yield):.3f}** (lowest {float(w.relative_yield[m].min()):.3f})")
    return text


def BlockTab(block: str):
    """Map of one area (North or South), zoomed to it, with its own summary."""
    _ = tick.value
    s = session.value
    w = s.world
    with solara.Div(classes=["abm-panel"]):
        RunControls()
    with solara.Div(classes=["abm-panel"]):
        solara.Markdown("**Map layer**")
        solara.ToggleButtonsSingle(value=map_mode, values=[key for key, _ in MAP_LAYERS])
        solara.Markdown(_block_summary(w, block))
    with solara.Div(classes=["abm-panel"]):
        solara.FigureMatplotlib(
            figure_field(w, mode=map_mode.value, selected=selected_paddy.value, block=block),
            dependencies=[tick.value, map_mode.value, selected_paddy.value, block],
        )


@solara.component
def NorthTab():
    BlockTab("North")


@solara.component
def SouthTab():
    BlockTab("South")


TABS = {
    "Field": ("mdi-grid", FieldTab),
    "North field": ("mdi-arrow-up-bold-box-outline", NorthTab),
    "South field": ("mdi-arrow-down-bold-box-outline", SouthTab),
    "Simulation settings": ("mdi-tune", SettingsTab),
    "Metrics": ("mdi-chart-line", MetricsTab),
}


@solara.component
def Page():
    # Touch tick so figures recompute after actions.
    _ = tick.value
    names = list(TABS)
    with solara.Column(classes=["abm-shell"], style={"width": "100%"}):
        solara.Style(CSS)
        with solara.Div(classes=["abm-hero"]):
            Header()
        with solara.lab.Tabs(
            value=names.index(selected_tab.value),
            on_value=lambda i: selected_tab.set(names[i]),
        ):
            for name in names:
                solara.lab.Tab(name, icon_name=TABS[name][0])
        TABS[selected_tab.value][1]()


def serve(script=None, port=8770, open_browser=True, reload=False):
    """Launch the ABM UI (blocks until Ctrl+C)."""
    from .serve import serve as _serve

    return _serve(script or __file__, port=port, open_browser=open_browser, reload=reload)


# Solara entry: `solara run abm/ui.py` or abm.serve()
Page  # noqa: B018 — Solara discovers Page in module scope

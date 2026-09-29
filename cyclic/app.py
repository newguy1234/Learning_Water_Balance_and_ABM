"""The interface to the cyclic irrigation environment.

    python example.py              # open this UI with the agents registered in example.py
    python -m cyclic               # command-line reports (paper, bench, copernicus, ...)

Tabs
    Farmers              farmer model (applied water and discharge)
    Run & compare        run farmers; writes the year caches the equation builder reads
    Equation builder     leftover fate: neighbour trickle, pool recapture, soil stay
    Field over time / Paddies   plots of farmer runs

The farmer model and the equation builder are separate. The builder fits three
leftover equations (neighbour trickle, recapture to the pool, soil stay) as a
general field law and as a per-paddy law, then compares them. Linear and
nonlinear are forms of the trickle equation, not two hydrology models.
"""

import copy
import math
import os
import sys
import threading
import time
import traceback
from html import escape
from types import SimpleNamespace

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the project root
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import solara
from matplotlib.figure import Figure

import cyclic
from cyclic import efficiency, imitation, okura
from cyclic.validation import output_label

try:
    import example  # noqa: F401  -- registers outside forecasters / ratio agents
except Exception:
    pass

YEARS = (2022, 2021)                 # 2022 is the comparison year

DEFAULT_CODE = '''# A quick farmer, written in the page. For anything you want to keep, write it
# in example.py instead. decide_batch(batch, params) runs every morning for every paddy.
#
# batch.depth_mm       standing water on the plot this morning
# batch.rain_mm        yesterday's rain (Open-Meteo)
# batch.et0_mm         yesterday's reference ET (Open-Meteo)
# batch.wind_mean_kmh, batch.wind_max_kmh, batch.wind_gust_kmh, batch.wind_dir_deg
# batch.tmean_c, batch.tmax_c, batch.tmin_c, batch.rh_mean_pct, batch.vpd_max_kpa
# batch.radiation_mj, batch.cloud_pct, batch.sunshine_h, batch.pressure_hpa
# batch.soil_temp_c, batch.soil_moisture, and other Open-Meteo daily fields on the batch
# batch.sat_ndvi       last cloud-free Sentinel-2 NDVI over this plot (nan before the first)
# batch.sat_et_mm      Kc(NDVI) x ET0 on that pass day
# batch.sat_days_ago   age of that pass
# batch.day, batch.date  day 0 is April 17
# batch.block, batch.cyclic, batch.area_m2, batch.paddy_id
#
# Return (irrigation_mm, discharge_mm), one entry per paddy.
# PARAMS become sliders: name -> (default, low, high).

NAME = "custom_farmer"
PARAMS = {
    "target_depth_mm": (30.0, 5.0, 80.0),
    "ceiling_mm": (90.0, 40.0, 120.0),
    "seepage_mm": (11.0, 0.0, 25.0),
    "overshoot": (1.3, 1.0, 2.0),
    "stop_day": (122.0, 100.0, 131.0),
}

def decide_batch(batch, params):
    depth = batch.depth_mm
    if batch.day >= params["stop_day"]:
        return np.zeros_like(depth), depth.copy()
    et = np.where(np.isfinite(batch.sat_et_mm), batch.sat_et_mm, 1.1 * batch.et0_mm)
    need = np.maximum(getattr(batch, "need_mm", 0.0), 0.0)
    keep = (params["target_depth_mm"] + et + params["seepage_mm"] + need) * params["overshoot"] / 0.87
    irrigation = np.clip(keep - depth, 0.0, 180.0)
    irrigation = np.where(batch.cyclic, irrigation, 0.0)
    discharge = np.maximum(depth + irrigation - params["ceiling_mm"], 0.0)
    return irrigation, discharge
'''

REGISTER_EXAMPLE = '''
Put a farmer in **example.py**, then run `python example.py`. The name in
`register_forecaster` is what appears in the **Agent** menus on **Run & compare**.

```python
import numpy as np
import cyclic

@cyclic.register_forecaster("simple_farmer")
class SimpleFarmer(cyclic.FarmerPolicy):
    defaults = dict(refill_below_mm=10.0, refill_mm=20.0, ceiling_mm=60.0)
    bounds = dict(refill_below_mm=(0.0, 50.0), refill_mm=(0.0, 60.0),
                  ceiling_mm=(20.0, 150.0))

    def decide_batch(self, observation):
        p = self.params
        depth = observation.depth_mm
        irrigation = np.where(depth < p["refill_below_mm"], p["refill_mm"], 0.0)
        discharge = np.maximum(depth - p["ceiling_mm"], 0.0)
        return irrigation, discharge
```
'''

RATIO_REGISTER_EXAMPLE = '''
Leftover **soil ratios** (Kx / Ky / Kz / ground) are a separate `@register_ratio`
class — not the farmer. Put it in **example.py** next to the farmer. The name
appears in leftover / ratio holdout (`cyclic.make_ratio(...)`, `ratio_holdout`).

```python
import numpy as np
import cyclic

@cyclic.register_ratio("my_soil_ratios")
class MySoilRatios(cyclic.RatioPolicy):
    def partition(self, leftover, observation, targets):
        # Start from Table 4–5 point targets, then reweight stay to leftover.
        shares = {name: np.array(targets.shares[name], copy=True)
                  for name in ("kx_ratio", "ky_ratio", "kz_ratio",
                               "ground_ratio", "farm_to_main", "pool_capture_share")}
        if leftover is not None:
            self.fit_stay_to_leftover(shares, leftover, observation, targets)
        return shares

# After a farmer season:
# shares = cyclic.make_ratio("my_soil_ratios").absorb(season)
# Holdout: cyclic.ratio_holdout("my_soil_ratios", train_year=2021, eval_year=2022)
```

`leftover` is IR+R−D−ET (Okura Eq. 3). Do **not** put soil K or applied IR into
`@register_lateral` hop laws — that is a different registry.
'''

CSS = """
.lwb-code textarea { font-family: Consolas, 'Courier New', monospace !important; font-size: 13px !important; line-height: 1.35 !important; }
.lwb-table { border-collapse: collapse; font-size: 13px; margin: 4px 0 12px 0; }
.lwb-table th, .lwb-table td { border-bottom: 1px solid #ddd; padding: 3px 10px; text-align: right; white-space: nowrap; }
.lwb-table th:first-child, .lwb-table td:first-child { text-align: left; }
.lwb-table thead th { background: #f3f5f7; }
.lwb-scroll { overflow-x: auto; max-width: 100%; }
.lwb-muted { color: #666; font-size: 13px; }
"""

TRAINED = "trained:"


# --- farmers written in the page --------------------------------------------------------------

def register_code(source):
    """Compile the editor's code into a registered FarmerPolicy. Returns its name."""
    namespace = {"np": np, "numpy": np, "math": math}
    exec(compile(source, "<farmer>", "exec"), namespace)
    name = str(namespace.get("NAME", "custom_farmer"))
    spec = namespace.get("PARAMS", {})
    body = {
        "__doc__": "Written in the page.",
        "defaults": {key: float(value[0]) for key, value in spec.items()},
        "bounds": {key: (float(value[1]), float(value[2])) for key, value in spec.items()},
    }
    if callable(namespace.get("decide_batch")):
        batch_rule = namespace["decide_batch"]
        body["decide_batch"] = lambda self, batch: batch_rule(batch, self.params)
    elif callable(namespace.get("decide")):
        rule = namespace["decide"]
        body["decide"] = lambda self, obs: rule(obs, self.params)
    else:
        raise ValueError("define decide(obs, params) or decide_batch(batch, params)")
    cls = type("".join(part.title() for part in name.split("_")) or "Custom", (cyclic.FarmerPolicy,), body)
    cyclic.register_agent(name)(cls)

    # Fail now, on a real morning, rather than two minutes into a run.
    inputs = cyclic.data.season_inputs(YEAR.value)
    sample = cyclic.agents.build_batch(inputs, 60, np.full(len(inputs.paddy_id), 20.0), inputs.area_m2)
    irrigation, discharge = cls().decide_batch(sample.subset(np.arange(5)))
    if np.shape(irrigation) != (5,) or np.shape(discharge) != (5,):
        raise ValueError("decide must return two numbers (or two arrays, one per paddy)")
    return name


# --- trained agents -----------------------------------------------------------------------------

_trained_cache = {}


def trained_agent(name):
    """A saved agent, reloaded only when its file changes."""
    path = cyclic.saved_agents()[name]
    stamp = os.path.getmtime(path)
    if _trained_cache.get(name, (None,))[0] != stamp:
        _trained_cache[name] = (stamp, cyclic.load_agent(name))
    return _trained_cache[name][1]


def agent_choices():
    return sorted(cyclic.AGENTS) + [TRAINED + name for name in cyclic.saved_agents()]


# --- state ------------------------------------------------------------------------------------

YEAR = solara.reactive(okura.COMPARISON_YEAR)


class Slot:
    """One of the two farmers the page works with."""

    def __init__(self, label, name):
        self.label = label
        self.name = solara.reactive(name)
        self.params = solara.reactive({})
        self.learned = solara.reactive(None)       # {"North": policy, "South": policy}
        self.learned_by = solara.reactive("")

    @property
    def trained(self):
        return self.name.value.startswith(TRAINED)

    def trained_instance(self):
        return trained_agent(self.name.value[len(TRAINED):])

    @property
    def policy_class(self):
        if self.trained:
            return type(self.trained_instance())
        return cyclic.AGENTS[self.name.value]

    def current_params(self):
        if self.trained:
            return {}
        cls = self.policy_class
        return {k: v for k, v in self.params.value.items() if k in cls.defaults}

    def agents(self):
        if self.learned.value:
            return dict(self.learned.value)
        if self.trained:
            return copy.deepcopy(self.trained_instance())
        return self.policy_class(**self.current_params())

    def describe(self):
        if self.learned.value:
            return f"learned by {self.learned_by.value}"
        if self.trained:
            return self.name.value
        shown = ", ".join(f"{k}={v:.3g}" for k, v in self.current_params().items())
        return f"{self.name.value}({shown})"

    def choose(self, name):
        self.name.set(name)
        self.params.set({})
        self.learned.set(None)


code = solara.reactive(DEFAULT_CODE)
code_message = solara.reactive(("", "info"))
try:
    register_code(DEFAULT_CODE)
except Exception as error:                      # data missing: the page still opens
    code_message.value = (f"Example farmer not registered: {error}", "warning")

base = Slot("Base farmer", "et_follower")
_farmer_default = ("et_net" if "et_net" in cyclic.AGENTS
                   else "et_match" if "et_match" in cyclic.AGENTS else "et_follower")
farmer = Slot("Farmer", TRAINED + _farmer_default if _farmer_default in cyclic.saved_agents() else _farmer_default)

runs = solara.reactive({})
run_busy = solara.reactive("")
run_error = solara.reactive("")
run_mode = solara.reactive("One farmer")

equation_busy = solara.reactive("")
equation_error = solara.reactive("")
equation_report = solara.reactive(None)
EQUATION_TRAIN_YEAR = 2021
EQUATION_EVAL_YEAR = 2022


paddy_run = solara.reactive("")
paddy_metric = solara.reactive("irrigation_mm")

_observed = {}


def observed(year):
    if year not in _observed:
        _observed[year] = imitation.load_observed(year)
    return _observed[year]


def short_error(error):
    """A ValueError is a message meant for the person using the page; anything else is a bug,
    so show where it happened."""
    if isinstance(error, ValueError):
        return str(error)
    return f"{type(error).__name__}: {error}\n" + "".join(traceback.format_exc().splitlines(True)[-6:])


# --- helpers ----------------------------------------------------------------------------------

def Table(frame, index=False, digits=3):
    html = frame.to_html(index=index, classes="lwb-table", border=0, na_rep="—",
                         float_format=lambda v: f"{v:,.{digits}f}")
    return solara.HTML(tag="div", unsafe_innerHTML=html, classes=["lwb-scroll"])


def blocks_figure(observed_blocks, series):
    """Per cyclic block: irrigation in and drainage back to the pool, both as volumes against
    what was observed, and the block's ET against its satellite target."""
    observed_blocks = imitation._complete(observed_blocks.copy())
    fig = Figure(figsize=(13, 9.5))
    axes = fig.subplots(3, 2, sharex=True)
    titles = {"irrigation": "irrigation into the block (×10³ m³/d)",
              "drainage": "drainage back to the pool (×10³ m³/d)"}
    for col, block in enumerate(okura.BLOCKS):
        o = observed_blocks[observed_blocks["block"] == block]
        for row, q in enumerate(("irrigation", "drainage")):
            ax = axes[row, col]
            ax.plot(o["date"], o[f"{q}_m3"] / 1000.0, color="black", lw=1.8, label="observed")
            for label, frame in series.items():
                s = frame[frame["block"] == block]
                ax.plot(s["date"], s[f"{q}_m3"] / 1000.0, lw=1.1, label=label)
            ax.set_title(f"{block} — {titles[q]}", fontsize=10)
            ax.legend(fontsize=8)
        ax = axes[2, col]
        first = True
        for label, frame in series.items():
            s = frame[frame["block"] == block]
            if first and "et_target_mm" in s:
                ax.plot(s["date"], s["et_target_mm"], color="black", lw=1.8, label="satellite ET target")
                first = False
            ax.plot(s["date"], s["et_mm"], lw=1.1, label=f"{label} ET")
        ax.set_title(f"{block} — ET the farmers produced vs satellite (mm/d, block mean)", fontsize=10)
        ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def run_record(label, description, results, year, seconds, agent_name=None):
    obs_blocks = observed(year)[0]
    return SimpleNamespace(
        label=label, description=description, results=results, year=year, seconds=seconds,
        agent_name=agent_name if agent_name is not None else label,
        score=imitation.score(results.blocks, obs_blocks),
        compare=efficiency.compare(results.blocks, results.station, year),
        flyover=cyclic.flyover.from_results(results),
    )


def agent_from_choice(name):
    """A policy instance from an Agent menu name (registered or trained:<name>)."""
    if name.startswith(TRAINED):
        return copy.deepcopy(trained_agent(name[len(TRAINED):]))
    return cyclic.make_agent(name)


def jobs_for_run():
    if run_mode.value == "All farmers":
        jobs, skipped = [], []
        for name in agent_choices():
            try:
                jobs.append((name, agent_from_choice(name), name))
            except Exception as error:
                skipped.append(f"{name} ({error})")
        if not jobs:
            raise ValueError("No farmers could be built" +
                             (": " + "; ".join(skipped) if skipped else "."))
        return jobs
    return [(farmer.name.value, farmer.agents(), farmer.describe())]


# --- actions ---------------------------------------------------------------------------------

def start_runs(jobs=None, year=None):
    year = YEAR.value if year is None else year
    run_error.set("")
    try:
        jobs = jobs if jobs is not None else jobs_for_run()
        jobs = copy.deepcopy(jobs)
    except Exception as error:
        run_error.set(short_error(error))
        return

    def work():
        found = {}
        try:
            observed(year)
            for label, agents, description in jobs:
                agent_name = label
                status = output_label(agents, year)
                label = f"{label} · {year} · {status}"
                description = f"{description}; {status}"
                run_busy.set(f"Running {label}: {description} on {year} …")
                started = time.time()
                results = cyclic.CyclicModel(year, agents=agents).run()
                found[label] = run_record(label, description, results, year, time.time() - started,
                                          agent_name=agent_name)
                csv_dir = getattr(results, "csv_dir", "")
                if csv_dir:
                    found[label].csv_dir = csv_dir
                    run_busy.set(f"Saved {label} to {csv_dir}")
                runs.set({**runs.value, **found})
        except Exception as error:
            run_error.set(short_error(error))
        finally:
            run_busy.set("")

    run_busy.set("Starting …")
    threading.Thread(target=work, daemon=True).start()


def start_equation_builder():
    equation_error.set("")
    equation_busy.set(
        f"Routing leftover toward ET-short neighbours on {EQUATION_TRAIN_YEAR} and {EQUATION_EVAL_YEAR}…")

    def work():
        try:
            from cyclic.identify import relative_et_holdout

            report = relative_et_holdout(
                train_year=EQUATION_TRAIN_YEAR, eval_year=EQUATION_EVAL_YEAR,
                progress=equation_busy.set)
            equation_report.set(report)
            equation_busy.set("")
        except Exception as error:
            equation_busy.set("")
            equation_error.set(short_error(error))

    threading.Thread(target=work, daemon=True).start()


# --- page -------------------------------------------------------------------------------------

def field_tab():
    """The animation of the field over time — runs for the header year only."""
    from cyclic import animation

    def set_year(v):
        year = int(v)
        YEAR.set(year)
        animation.paper_year.set(str(year))
        animation.chosen_run.set("")
        animation.day.set(0)

    year = int(YEAR.value)
    filtered = {label: record for label, record in runs.value.items()
                if int(getattr(record, "year", year)) == year}
    return animation.FieldTab(filtered, year=year, on_year=set_year)


TABS = {
    "Farmers": ("mdi-account-cowboy-hat", lambda: FarmersTab()),
    "Run & compare": ("mdi-play-circle-outline", lambda: RunTab()),
    "Equation builder": ("mdi-function-variant", lambda: EquationBuilderTab()),
    "Field over time": ("mdi-play-box-outline", field_tab),
    "Paddies": ("mdi-grid", lambda: PaddiesTab()),
}
selected_tab = solara.reactive("Farmers")


@solara.component
def Page():
    solara.Style(CSS)
    solara.Title("Cyclic irrigation")
    names = list(TABS)
    with solara.Column(style={"padding": "12px 24px", "max-width": "1800px"}):
        Header()
        # The tabs are labels only and the chosen tab is rendered below them, the way
        # Main_Show's interface does it. Nesting content inside solara.lab.Tab depends on the
        # Vuetify version underneath: with ipyvuetify 3 the content ends up inside the tab bar
        # and the page body stays empty. This way only the open tab renders at all.
        with solara.lab.Tabs(value=names.index(selected_tab.value),
                             on_value=lambda i: selected_tab.set(names[i])):
            for name in names:
                solara.lab.Tab(name, icon_name=TABS[name][0])
        TABS[selected_tab.value][1]()


@solara.component
def Header():
    from cyclic import animation

    def set_year(v):
        year = int(v)
        YEAR.set(year)
        animation.paper_year.set(str(year))
        animation.chosen_run.set("")
        animation.day.set(0)

    source = observed(YEAR.value)[2]
    with solara.Row(style={"align-items": "center", "flex-wrap": "wrap"}, gap="18px"):
        solara.Markdown("## Shiroyama-Jimbee cyclic irrigation — farmer model and leftover hydrology equation")
        solara.ToggleButtonsSingle(value=str(YEAR.value), values=[str(y) for y in YEARS],
                                   on_value=set_year)
    solara.HTML(tag="div", classes=["lwb-muted"], unsafe_innerHTML=(
        f"{len(cyclic.data.paddies()):,} paddies · {okura.SEASON_DAYS}-day season from April 17 · observed block series: "
        f"<b>{escape(source)}</b> · satellite: "
        f"<b>{escape(cyclic.data.season_inputs(YEAR.value).satellite_source)}</b> · "
        f"Farmer: <b>{escape(farmer.describe())}</b> · "
        f"Base farmer: <b>{escape(base.describe())}</b> · "
        f"Equation: leftover after satellite ET, {EQUATION_TRAIN_YEAR} capture frozen on {EQUATION_EVAL_YEAR}"))


@solara.component
def EquationBuilderTab():
    from cyclic.identify import (
        equation_cache_status, equation_figure, equation_score_table,
        equation_scope_summary, load_equation_report, pin_summary_text,
        simple_leftover_table, relative_et_table,
    )

    solara.Markdown(
        "Leftover at each paddy is `IR + rain − D − satellite ET`. Satellite ET is the "
        "loss. Surplus leftover moves **right / left / up / down** in proportion to how "
        "short that neighbour is for its own satellite ET; what neighbours do not take "
        "stays **in the ground**. Farmer irrigation can be odd — the routing still uses "
        "the ET that actually happened. FAO-56 is not overwritten."
    )
    train_status = equation_cache_status(EQUATION_TRAIN_YEAR)
    eval_status = equation_cache_status(EQUATION_EVAL_YEAR)
    solara.Markdown(
        f"- **{EQUATION_TRAIN_YEAR} cache:** {train_status.source}\n"
        f"- **{EQUATION_EVAL_YEAR} cache:** {eval_status.source}\n"
        "- **2021** without a run folder is reconstructed leftover "
        "`IR + R − D − satellite ET` from `outputs/paddy_daily_irrigation_mm_2021.csv` "
        "and season ET.\n"
        "- **2022** prefers `outputs/runs/2022_*/paddy_daily.csv` so leftover can be "
        "re-pinned from irrigation, rain, drainage, and `et_target_mm`."
    )
    if not train_status.usable or not eval_status.usable:
        solara.Warning(
            "Need a leftover/ET cache for both years. Run a farmer on **Run & compare** "
            "for 2022, and keep the 2021 daily irrigation/drainage CSVs "
            "(or run 2021 so a `outputs/runs/2021_*/paddy_daily.csv` exists)."
        )
    with solara.Row(style={"align-items": "flex-end", "flex-wrap": "wrap"}, gap="16px"):
        solara.Button(
            f"Route leftover to ET-short neighbours ({EQUATION_TRAIN_YEAR} and {EQUATION_EVAL_YEAR})",
            icon_name="mdi-function-variant", color="primary",
            disabled=bool(equation_busy.value) or not train_status.usable or not eval_status.usable,
            on_click=lambda: start_equation_builder())
        def load_saved():
            loaded = load_equation_report(EQUATION_TRAIN_YEAR, EQUATION_EVAL_YEAR)
            if loaded is None:
                equation_error.set(
                    f"No saved equation yet. Fit H on {EQUATION_TRAIN_YEAR} first.")
            else:
                equation_error.set("")
                equation_report.set(loaded)
        solara.Button("Show last saved equation", icon_name="mdi-content-save",
                      disabled=bool(equation_busy.value), on_click=load_saved)
    if equation_busy.value:
        solara.Info(equation_busy.value)
        solara.ProgressLinear(True)
    if equation_error.value:
        solara.Error(equation_error.value)

    report = equation_report.value
    if report is None:
        solara.Markdown(
            "_Run the leftover routing to send surplus "
            "toward neighbours that were short for satellite ET._"
        )
        return
    if getattr(report, "relative", False):
        solara.Markdown(
            f"### Relative ET routing · {report.train_year} → {report.eval_year}\n\n"
            f"```\n{report.equation_text}\n```"
        )
        Table(relative_et_table(report.train_route, report.eval_route), digits=1)
        solara.Markdown(
            f"Train source: `{getattr(report, 'train_source', '')}`  \n"
            f"Eval source: `{getattr(report, 'eval_source', '')}`  \n"
            "ET_match is satellite ET covered by local IR+rain−D plus neighbour surplus. "
            "Gap closed is the share of ET-short plots that neighbours could fill. "
            "Unaccepted surplus is ground."
        )
        return
    if getattr(report, "simple", False):
        solara.Markdown(
            f"### Leftover split · {report.train_year} capture {report.capture:.3f} frozen on {report.eval_year}\n\n"
            f"```\n{report.equation_text}\n```"
        )
        Table(simple_leftover_table(report), digits=1)
        solara.Markdown(
            f"```\n{pin_summary_text(report.train_pin)}\n{pin_summary_text(report.eval_pin)}\n```"
        )
        if report.eval_ceiling is not None:
            solara.Markdown(
                f"{report.eval_year} ceiling if every infiltrated millimetre went to the drains:\n\n"
                f"```\n{pin_summary_text(report.eval_ceiling)}\n```"
            )
        solara.Markdown(
            f"Train source: `{getattr(report, 'train_source', '')}`  \n"
            f"Eval source: `{getattr(report, 'eval_source', '')}`  \n"
            "Hops equalise leftover head; they are not a mass sink. Drain capture only "
            "uses infiltrated leftover. A 2022 miss on leftover itself is the farmer IR/D, "
            "not this split."
        )
        return
    disk = " · **loaded from disk**" if getattr(report, "from_disk", False) else ""
    solara.Markdown(
        f"### Frozen H from {report.train_year} · scored on {report.eval_year}"
        + (f" · `{report.eval_target}`" if getattr(report, "eval_target", None) else "")
        + (f" · saved `{report.saved}`" if getattr(report, "saved", None) else "")
        + disk
    )
    pin = getattr(report, "water_pin", None)
    if pin is not None:
        texts = []
        if getattr(pin, "train", None) is not None:
            texts.append(pin_summary_text(pin.train))
        if getattr(pin, "eval", None) is not None:
            texts.append(pin_summary_text(pin.eval))
        if texts:
            solara.Markdown(
                "Leftover is pinned to satellite ET at each paddy; drain capture is scaled "
                "to Table 5 LFV. FAO-56 `et_mm` is not overwritten.\n\n"
                f"```\n{chr(10).join(texts)}\n```"
            )
    solara.Markdown(
        "**General field laws** (one weight vector for every paddy). Neighbour trickle uses "
        "head drops. Recapture and soil stay are leftover-mass laws; after the linear fit they "
        "are scaled so `to_pool_i + stay_i = leftover_i`.\n\n"
        f"```\n{getattr(report, 'equations_text', report.linear.equation_text)}\n```"
    )
    summary = equation_scope_summary(report)
    if summary:
        solara.Markdown("**General vs per-paddy** on holdout (same variables; per-paddy fits "
                        "one law per plot and falls back to the general weights if that plot "
                        f"is singular)\n\n{summary}")
    solara.Markdown("Per-paddy laws are not printed in full (thousands of weight vectors). "
                    "The table compares RMSE.")
    Table(equation_score_table(report), digits=4)
    if getattr(report, "eval_linear", None) is not None:
        solara.FigureMatplotlib(equation_figure(report), format="png")
    solara.Markdown(
        f"Train source: `{getattr(report, 'train_source', '')}`  \n"
        f"Eval source: `{getattr(report, 'eval_source', '')}`  \n"
        "Head-equalise is the physics teacher for neighbour trickle, not a second fitted model. "
        "The general law is one equation for the field. The per-paddy law is the same equation "
        "form with a different weight vector on each plot. "
        "A trickle form that fits 2021 leftover hops but has a large 2022 RMSE did not transfer."
    )


@solara.component
def FarmersTab():
    solara.Markdown("## How to register a farmer")
    solara.Markdown(REGISTER_EXAMPLE)
    solara.Markdown("## How to register a soil ratio finder")
    solara.Markdown(RATIO_REGISTER_EXAMPLE)


@solara.component
def SlotEditor(slot):
    with solara.Card(slot.label):
        if slot.learned.value:
            solara.Success(f"Using the policies learned by {slot.learned_by.value}.")
            for block, policy in slot.learned.value.items():
                solara.Text(f"{block}: {policy!r}"[:400])
            solara.Button("Back to a registered agent", text=True, on_click=lambda: slot.learned.set(None))
            return

        choices = agent_choices()
        if slot.name.value not in choices:           # a saved file deleted while the page was open
            slot.name.value = choices[0]
        solara.Select("Agent", values=choices, value=slot.name.value, on_value=slot.choose)
        if slot.trained:
            try:
                solara.Success("Trained agent, saved with cyclic.save_agent.")
                solara.Text(repr(slot.trained_instance())[:500])
            except Exception as error:
                solara.Error(f"Could not load it: {error}")
            return

        cls = slot.policy_class
        doc = (cls.__doc__ or "").strip().split("\n\n")[0]
        if doc:
            solara.HTML(tag="div", classes=["lwb-muted"], unsafe_innerHTML=escape(doc).replace("\n", "<br>"))
        numeric = {k: v for k, v in cls.defaults.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if not numeric:
            solara.Text("No parameters." if cls.name != "linear" else
                        "A linear policy has no hand-set parameters: it is fitted by aggregate cloning.")
        for key, default in numeric.items():
            low, high = cls.bounds.get(key, (0.0, max(2.0 * float(default), 1.0)))
            value = float(slot.params.value.get(key, default))
            span = high - low
            step = 0.01 if span <= 2 else 0.1 if span <= 20 else 0.5
            solara.SliderFloat(key, value=min(max(value, low), high), min=low, max=high,
                               step=step, thumb_label=False,
                               on_value=lambda v, k=key: slot.params.set({**slot.params.value, k: v}))
        if slot.params.value:
            solara.Button("Reset to defaults", text=True, on_click=lambda: slot.params.set({}))


flyover_pass = solara.reactive("")


def weighted_cdf(values, weights):
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return values, np.cumsum(weights) / weights.sum()


def flyover_figure(records, pass_date):
    """Top: every run's fly-over fidelity through the season. Bottom: at one fly-over, the
    distribution of ET across the block's paddies -- satellite against each run -- and the
    Open-Meteo field average the block's mean should sit on."""
    fig = Figure(figsize=(13, 8))
    axes = fig.subplots(2, 2)
    inputs = records[0].results.model.inputs
    day = int(np.flatnonzero(inputs.dates == pd.Timestamp(pass_date))[0])
    for col, block in enumerate(okura.BLOCKS):
        ax = axes[0, col]
        for r in records:
            table = r.flyover.table[r.flyover.table["block"] == block]
            line = ax.plot(table["date"], table["fidelity"], marker="o", label=f"{r.label}: fidelity")[0]
            ax.plot(table["date"], table["distribution_fidelity"], ls=":", marker=".", color=line.get_color(),
                    label=f"{r.label}: distribution")
        ax.axvline(pd.Timestamp(pass_date), color="red", lw=1)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{block} — ET fidelity at each fly-over", fontsize=10)
        ax.legend(fontsize=7)

        ax = axes[1, col]
        mask = inputs.is_pass[day] & (inputs.block == block) & np.isfinite(inputs.sat_pass_et_mm[day])
        if mask.sum():
            weights = inputs.area_m2[mask]
            ax.step(*weighted_cdf(inputs.sat_pass_et_mm[day, mask].astype(float), weights), where="post",
                    color="black", lw=2, label="satellite")
            for r in records:
                ax.step(*weighted_cdf(r.results.model.log["et_mm"][day, mask].astype(float), weights), where="post",
                        lw=1.3, label=r.label)
            ax.axvline(float(inputs.field_et_mm[block][day]), color="grey", ls="--", label="field average (Open-Meteo)")
        ax.set_xlabel("ET mm/d")
        ax.set_ylabel("share of block area")
        ax.set_title(f"{block} — ET across paddies on {pd.Timestamp(pass_date):%d %b %Y}", fontsize=10)
        ax.legend(fontsize=7)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


@solara.component
def FlyoverSection(records):
    solara.Markdown(
        "### Satellite fly-overs\n"
        "These are **NDVI-derived spatial proxy diagnostics**, not independent measurements of actual ET. "
        "Each map is divided by its own mean before comparing its distribution and paired paddy values. "
        "The score averages those two spatial comparisons; matching Open-Meteo's mean earns no score. "
        "The weather-based absolute target is excluded from training by default.")
    Table(pd.DataFrame([{"run": r.label, "fly-overs": r.flyover.passes, "spatial proxy consistency": r.flyover.fidelity,
                         "distribution": r.flyover.distribution,
                         "paired": r.flyover.paired} for r in records]))
    dates = sorted(records[0].flyover.table["date"].dt.strftime("%Y-%m-%d").unique())
    if not dates:
        solara.Markdown("_No confirmed fly-overs in this season._")
        return
    chosen = flyover_pass.value if flyover_pass.value in dates else dates[0]
    solara.Select("Fly-over", values=dates, value=chosen, on_value=flyover_pass.set)
    solara.FigureMatplotlib(flyover_figure(records, chosen), format="png")
    with solara.Details("Every fly-over, every block"):
        for r in records:
            solara.Markdown(f"**{r.label}**")
            Table(r.flyover.table.assign(date=r.flyover.table["date"].dt.strftime("%Y-%m-%d")))


@solara.component
def FarmerSelect(slot):
    choices = agent_choices()
    if not choices:
        solara.Warning("No farmers registered. Add one in example.py.")
        return
    current = slot.name.value if slot.name.value in choices else choices[0]
    if slot.name.value not in choices:
        slot.name.value = choices[0]
    solara.Select(slot.label, values=choices, value=current, on_value=slot.choose)


@solara.component
def RunTab():
    obs_blocks = observed(YEAR.value)[0]
    one = run_mode.value == "One farmer"
    n_all = len(agent_choices())
    solara.Markdown(
        "Run **one** registered farmer, or **all** of them. Each run writes CSV under "
        "`outputs/runs/<year>_<agent>/` (blocks, station, paddies, daily stocks) and **does not "
        "draw maps while it runs**. Open **Field over time** or **Paddies** afterwards if you "
        "want plots. **Fidelity** 1 means exact season totals.")
    with solara.Row(style={"align-items": "flex-end", "flex-wrap": "wrap"}, gap="16px"):
        solara.ToggleButtonsSingle(value=run_mode, values=["One farmer", "All farmers"])
        if one:
            FarmerSelect(farmer)
        solara.Select("Output year", values=list(YEARS), value=YEAR)
        label = (f"Run {farmer.name.value} on {YEAR.value}" if one
                 else f"Run all {n_all} farmers on {YEAR.value}")
        solara.Button(label, icon_name="mdi-play", color="primary",
                      disabled=bool(run_busy.value), on_click=lambda: start_runs())
    if one:
        solara.Markdown(f"Running **{farmer.describe()}**.")
    else:
        solara.Markdown(f"Running every registered farmer ({n_all}), each at its default parameters.")
    if run_busy.value:
        solara.Info(run_busy.value)
        solara.ProgressLinear(True)
    if run_error.value:
        solara.Error(run_error.value)

    records = [r for r in runs.value.values() if r.year == YEAR.value]
    if not records:
        solara.Markdown("_Nothing run for this season yet._")
        return

    summary = []
    for r in records:
        row = {"run": r.label, "policy": r.description, "water-volume fidelity": r.score.fidelity,
               "irrigation fidelity": r.score.irrigation_fidelity,
               "discharge fidelity": r.score.discharge_fidelity,
               "water-volume loss": r.score.loss, "spatial ET proxy consistency": r.flyover.fidelity,
               "seconds": r.seconds, "csv": getattr(r, "csv_dir", "")}
        for _, s in r.score.table.iterrows():
            row[f"{s['block']} {s['quantity']} err %"] = s["season_error_%"]
        summary.append(row)
    solara.Markdown("### Score against the observed block series")
    solara.Markdown("Irrigation and discharge fidelity remain part of scoring alongside ET. "
                    "Water-volume loss includes both season totals and daily errors; the daily training reward "
                    "uses irrigation and discharge (default weights 1:1). Weather-derived absolute ET is diagnostic only. "
                    "Discharge here is total drainage leaving the block, including passive drainage.")
    Table(pd.DataFrame(summary))

    with solara.Details("Daily block plots (slow — skip this while comparing runs)"):
        solara.FigureMatplotlib(blocks_figure(obs_blocks, {r.label: r.results.blocks for r in records}),
                                format="png")
        FlyoverSection(records)

    solara.Markdown("### Beside the paper (Tables 4–7)")
    merged = None
    for r in records:
        frame = r.compare[["table", "quantity", "block", "model", "paper"]].rename(columns={"model": r.label})
        merged = frame if merged is None else merged.merge(frame, on=["table", "quantity", "block", "paper"])
    columns = ["table", "quantity", "block", *[r.label for r in records], "paper"]
    Table(merged[columns])

    with solara.Details("What sets WUE', RUE and LUE in each run (identities)"):
        solara.Markdown("WUE′ = ET / (ET + Lf − (R − ER)); RUE = D/IR × IR/IR_all; "
                        "LUE = capture × lateral / IR_all. See cyclic/efficiency.py.")
        for r in records:
            solara.Markdown(f"**{r.label}**")
            Table(efficiency.identities(r.results.blocks, r.results.station, r.results.model.physics).reset_index())


paddy_pick = solara.reactive("")
paddy_block = solara.reactive("North")


def _series_mm(series, name, fallback=None):
    if name in series.columns:
        return series[name]
    if fallback and fallback in series.columns:
        return series[fallback]
    return np.zeros(len(series))


def paddy_figure(series):
    """One paddy's season: ET, irrigation/drainage, pond/topsoil stocks, leftover split."""
    fig = Figure(figsize=(13, 11.5))
    ax_et, ax_water, ax_stock, ax_under = fig.subplots(4, 1, sharex=True)
    dates = series["date"]
    ax_et.plot(dates, series["et_target_mm"], color="black", lw=1.6, label="its satellite ET target")
    ax_et.plot(dates, series["field_et_mm"], color="grey", lw=1.0, ls="--", label="field-wide ET (interpolates between passes)")
    ax_et.plot(dates, series["et_mm"], color="tab:blue", lw=1.3, label="ET this farmer produced")
    ax_et.plot(dates, _series_mm(series, "et_pond_mm"), color="tab:cyan", lw=1.0, label="E from pond")
    ax_et.plot(dates, _series_mm(series, "et_surface_mm"), color="tab:olive", lw=1.0, label="E from surface soil")
    ax_et.plot(dates, _series_mm(series, "et_root_mm"), color="tab:green", lw=1.0, label="T from root zone")
    ax_et.plot(dates, _series_mm(series, "et_groundwater_mm"), color="tab:purple", lw=1.2, label="ET from groundwater")
    passes = series[series["is_pass"]]
    ax_et.scatter(passes["date"], passes["et_target_mm"], s=40, color="red", zorder=5,
                  label="its own satellite pass")
    ax_et.set_ylabel("mm/d")
    ax_et.legend(fontsize=8, ncol=3)
    ax_et.set_title("Evapotranspiration", fontsize=10)

    ax_water.bar(dates, series["irrigation_m3"], color="tab:green", width=0.9, label="irrigation in (m³)")
    ax_water.bar(dates, -series["drainage_m3"], color="tab:brown", width=0.9, label="drainage out (m³)")
    ax_water.axhline(0, color="black", lw=0.6)
    ax_water.set_ylabel("m³/d")
    ax_water.legend(fontsize=8)
    ax_water.set_title("Water sent to and let go from this paddy", fontsize=10)

    pond = _series_mm(series, "pond_mm", "depth_mm")
    topsoil = _series_mm(series, "topsoil_mm")
    area = float(series.attrs.get("area_m2", 1.0))
    ax_stock.plot(dates, pond * area / 1000.0, color="tab:blue", lw=1.2, label="pond volume (m³)")
    ax_stock.plot(dates, topsoil * area / 1000.0, color="tab:orange", lw=1.2, label="topsoil volume stock (m³)")
    ax_rain = ax_stock.twinx()
    ax_rain.bar(dates, series["rain_mm"], color="lightsteelblue", width=0.9, alpha=0.6, label="rain (mm)")
    ax_stock.set_ylabel("stock m³")
    ax_rain.set_ylabel("rain mm")
    ax_stock.legend(fontsize=8, loc="upper left")
    ax_rain.legend(fontsize=8, loc="upper right")
    ax_stock.set_title("Pond and topsoil volume stocks", fontsize=10)

    leftover = _series_mm(series, "leftover_mm")
    stay = _series_mm(series, "stay_mm", "deep_mm")
    into_ground = _series_mm(series, "infiltration_mm")
    if np.all(into_ground == 0):
        into_ground = np.maximum(leftover, 0.0)
    ax_under.plot(dates, _series_mm(series, "need_mm"), color="tab:red", lw=1.3,
                  label="ET need missing from topsoil")
    ax_under.plot(dates, into_ground, color="tab:green", lw=1.3, label="into ground soil")
    ax_under.plot(dates, leftover, color="tab:purple", lw=1.0, ls="--", label="leftover underground")
    ax_under.plot(dates, series["lateral_mm"], color="tab:pink", lw=1.2, label="lateral flow sent")
    cyclic = bool(series.attrs.get("cyclic", True))
    ax_under.plot(dates, _series_mm(series, "canal_mm", "drainage_mm"), color="tab:brown", lw=1.2,
                  label="D back to the pool" if cyclic else "non-cyclic D to the lake")
    ax_under.plot(dates, stay, color="0.35", lw=1.1, label="stays in ground")
    ax_under.axhline(0, color="black", lw=0.5)
    ax_under.set_ylabel("mm/d")
    ax_under.legend(fontsize=8, ncol=2)
    ax_under.set_title("Need, leftover, lateral, and D back to the pool", fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


@solara.component
def PaddiesTab():
    year = int(YEAR.value)
    records = {label: record for label, record in runs.value.items()
               if int(getattr(record, "year", year)) == year}
    names = list(records)
    chosen = paddy_run.value if paddy_run.value in names else (names[0] if names else None)
    record = records.get(chosen)
    # Before any return: hooks have to run on every render.
    season = solara.use_memo(lambda: record.results.paddy_season() if record else None,
                             dependencies=[id(record)])
    if record is None:
        solara.Markdown(f"_No run for **{year}** yet. Run something on **Run & compare**._")
        return
    metrics = ["et_fidelity", "et_pass_fidelity", "irrigation_m3", "drainage_m3", "irrigation_mm", "drainage_mm",
               "farmer_discharge_mm", "passive_drainage_mm", "lateral_mm", "leftover_mm", "stay_mm",
               "et_groundwater_mm", "et_mm", "et_target_mm", "wue_prime_pct", "mean_depth_mm",
               "mean_pond_mm", "mean_topsoil_mm", "days_irrigated", "passes"]
    with solara.Row():
        solara.Select("Run", values=names, value=chosen, on_value=paddy_run.set)
        solara.Select("Season value", values=metrics, value=paddy_metric)

    ids = season["paddy_id"].tolist()
    picked = paddy_pick.value if paddy_pick.value in ids else ids[0]
    row = season[season["paddy_id"] == picked].iloc[0]

    fig = Figure(figsize=(8.5, 9))
    ax = fig.subplots()
    metric = paddy_metric.value if paddy_metric.value in season else "et_fidelity"
    points = ax.scatter(season["lon"], season["lat"], c=season[metric], s=4, cmap="viridis")
    edge = season[season["cyclic"]]
    ax.scatter(edge["lon"], edge["lat"], s=18, facecolors="none", edgecolors="red", linewidths=0.15, alpha=0.4)
    ax.plot(*cyclic.data.POOL_LONLAT, marker="*", markersize=16, color="red", markeredgecolor="white")
    ax.scatter([row["lon"]], [row["lat"]], s=260, facecolors="none", edgecolors="magenta", linewidths=2.5, zorder=6)
    ax.set_aspect(1 / 0.81)
    ax.set_title(f"{record.label}: {metric}  (red rings: cyclic, star: pumping station, magenta: selected paddy)",
                 fontsize=9)
    fig.colorbar(points, ax=ax, shrink=0.6)
    fig.tight_layout()
    with solara.Columns([3, 2]):
        solara.FigureMatplotlib(fig, format="png")
        with solara.Column():
            grouped = season.groupby(["block", "cyclic"])[
                ["area_ha", "irrigation_m3", "drainage_m3", "et_mm", "et_target_mm", "et_fidelity",
                 "et_pass_fidelity", "mean_depth_mm"]].mean().reset_index()
            solara.Markdown("**Mean per paddy**")
            Table(grouped, digits=2)
            solara.Markdown(f"**{metric}: spread across paddies**")
            Table(season.groupby("block")[metric].describe().reset_index(), digits=3)

    solara.Markdown("### One paddy")
    block = paddy_block.value
    in_block = season[season["block"] == block]["paddy_id"].tolist()
    position = in_block.index(picked) if picked in in_block else 0
    with solara.Row(style={"align-items": "center", "flex-wrap": "wrap"}, gap="12px"):
        solara.InputText("Paddy id", value=picked, on_value=paddy_pick.set, continuous_update=False)
        solara.Select("Block", values=list(okura.BLOCKS), value=block, on_value=paddy_block.set)
        solara.Button("Worst ET fit", outlined=True,
                      on_click=lambda: paddy_pick.set(season.sort_values("et_fidelity")["paddy_id"].iloc[0]))
        solara.Button("Best ET fit", outlined=True,
                      on_click=lambda: paddy_pick.set(season.sort_values("et_fidelity")["paddy_id"].iloc[-1]))
        solara.Button("Random", outlined=True,
                      on_click=lambda: paddy_pick.set(str(np.random.choice(ids))))
    solara.SliderInt(f"{block} paddies, nearest the pumping station first", value=position, min=0,
                     max=max(len(in_block) - 1, 0), on_value=lambda i: paddy_pick.set(in_block[i]))

    series = record.results.paddy_series(picked)
    Table(pd.DataFrame([{
        "paddy": picked, "block": row["block"], "cyclic": bool(row["cyclic"]), "area_ha": row["area_ha"],
        "satellite passes": int(row["passes"]), "ET fidelity (all days)": row["et_fidelity"],
        "ET fidelity (its passes)": row["et_pass_fidelity"], "ET mm": row["et_mm"],
        "ET target mm": row["et_target_mm"], "irrigation m³": row["irrigation_m3"],
        "drainage m³": row["drainage_m3"], "mean depth mm": row["mean_depth_mm"],
    }]))
    solara.FigureMatplotlib(paddy_figure(series), format="png")

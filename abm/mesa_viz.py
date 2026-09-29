# -*- coding: utf-8 -*-
"""Mesa's own SolaraViz shell for the ABM ``DistrictModel`` — a framework-native
view (play/pause/step/reset + parameter widgets all provided by Mesa itself)
alongside the hand-built ``abm.ui`` page.

    python -m abm mesa-ui

DistrictModel (abm/model.py) is a real mesa.Model with FarmerMesaAgent/StationMesaAgent
mesa.Agent instances, so it plugs into mesa.visualization.SolaraViz directly —
no adapter needed. The map/series/metrics figures are the same ones abm.ui uses.
"""

from __future__ import annotations


import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from abm.agents import default_roster

import matplotlib

matplotlib.use("Agg")

import solara
from mesa.visualization import SolaraViz
from mesa.visualization.user_param import Slider

from abm.metrics import season_efficiencies
from abm.model import DistrictModel
from abm.plots import figure_field, figure_metrics, figure_series
from abm.station import default_station, list_stations
from abm.world import SeasonConfig


def SpaceComponent(model: DistrictModel):
    fig = figure_field(model.world, mode="ponding")
    return solara.FigureMatplotlib(fig, format="png")


def SeriesComponent(model: DistrictModel):
    fig = figure_series(model.world)
    return solara.FigureMatplotlib(fig, format="png")


def MetricsComponent(model: DistrictModel):
    eff = season_efficiencies(model.world) if model.world.history else {}
    fig = figure_metrics(eff)
    return solara.FigureMatplotlib(fig, format="png")


def StatusComponent(model: DistrictModel):
    w = model.world
    return solara.Markdown(
        f"Day **{w.day}/{w.config.days}** · pool **{w.pool_m3:,.0f}** m³ · "
        f"Ya/Ym **{w.relative_yield.mean():.2f}** · station **{model.station_name}**/"
        f"**{model.last_station_state}** · roster {model.effective_roster_weights()}"
    )


# Fixed at construction (agent_roster/roster_weights); station_name and seed are the
# parameters Mesa's own ModelCreator sidebar exposes as widgets.
model_params = {
    "config": SeasonConfig(),
    "agent_roster": default_roster(),
    "roster_weights": {},
    "station_name": {
        "type": "Select",
        "label": "Station policy",
        "value": default_station(),
        "values": list_stations(),
    },
    "seed": Slider("Seed", value=0, min=0, max=100, step=1),
}

_model = DistrictModel(SeasonConfig())


@solara.component
def Page():
    SolaraViz(
        _model,
        components=[StatusComponent, SpaceComponent, SeriesComponent, MetricsComponent],
        model_params=model_params,
        name="Imbanuma field · ABM (Mesa SolaraViz)",
    )


def serve(script=None, port=8775, open_browser=True, reload=False):
    """Launch Mesa's own SolaraViz shell for the ABM (blocks until Ctrl+C)."""
    import cyclic

    target = os.path.abspath(script) if script else os.path.abspath(__file__)
    return cyclic.serve(script=target, port=port, open_browser=open_browser, reload=reload)


# Solara entry: `solara run abm/mesa_viz.py` or cyclic.serve(this file)
Page  # noqa: B018 — Solara discovers Page in module scope

# -*- coding: utf-8 -*-
"""Agent-based models on frozen cyclic physics (conceptual twin gym).

    python -m abm ui
    python example.py abm      # with the farmer + station registered in example.py
"""

from .world import WaterWorld, SeasonConfig, StepResult, assign_owners
from .farmer import (
    BaseFarmer,
    Constraint,
    NoDrainHold,
    Plan,
    Trait,
    TRAITS,
    FARMER_DEFAULTS,
)
from .agents import (
    AGENTS,
    FarmerAgent,
    PersonalityFarmer,
    DEFAULT_ROSTER,
    default_roster,
    make_agent,
    list_agents,
    register,
    register_farmer,
    register_agent,
)
from .station import (
    STATIONS,
    StationAgent,
    make_station,
    list_stations,
    register_station,
)
from .metrics import season_efficiencies, summarise_run
from .simulate import run_season, compare_agents
from .landscape import load_cyclic_landscape, landscape_summary, CyclicLandscape
from .field import FieldSession
from .model import DistrictModel, FarmerMesaAgent, StationMesaAgent
from .optimise import MPCStation
from .gym_env import AbmEnv
from .weather import WeatherScenario, weather_scenario, list_weather_scenarios
from . import crop

__all__ = [
    "WaterWorld",
    "SeasonConfig",
    "StepResult",
    "assign_owners",
    "load_cyclic_landscape",
    "landscape_summary",
    "CyclicLandscape",
    "AGENTS",
    "BaseFarmer",
    "Constraint",
    "NoDrainHold",
    "Plan",
    "Trait",
    "TRAITS",
    "FARMER_DEFAULTS",
    "FarmerAgent",
    "PersonalityFarmer",
    "DEFAULT_ROSTER",
    "default_roster",
    "make_agent",
    "list_agents",
    "register",
    "register_farmer",
    "register_agent",
    "STATIONS",
    "StationAgent",
    "make_station",
    "list_stations",
    "register_station",
    "season_efficiencies",
    "summarise_run",
    "run_season",
    "compare_agents",
    "FieldSession",
    "DistrictModel",
    "FarmerMesaAgent",
    "StationMesaAgent",
    "MPCStation",
    "AbmEnv",
    "WeatherScenario",
    "weather_scenario",
    "list_weather_scenarios",
    "crop",
    "serve",
    "page",
]


def page():
    from .ui import Page

    return Page


def serve(script=None, port=8770, open_browser=True, reload=False):
    from .ui import serve as _serve

    return _serve(script=script, port=port, open_browser=open_browser, reload=reload)

# -*- coding: utf-8 -*-
"""Run ABM seasons — single policy or multi-farmer field session."""

from __future__ import annotations

from abm.agents import default_roster

import pandas as pd

from .field import FieldSession
from .world import WaterWorld


def run_season(
    agent_name: str = "steward",
    year: int = 2022,
    days: int = 131,
    seed: int = 0,
    agent_kwargs: dict | None = None,
    world: WaterWorld | None = None,
    paddies_per_farmer: int | None = None,
    station_name: str | None = None,
) -> tuple[WaterWorld, dict]:
    """Roll one season on every cyclic paddy, with LID station + farmers (apply ≤ sent)."""
    kwargs = dict(
        year=year,
        days=days,
        seed=seed,
        agent_roster=[agent_name],
        station_name=station_name,
    )
    if paddies_per_farmer is not None:
        kwargs["paddies_per_farmer"] = paddies_per_farmer
    session = FieldSession(**kwargs)
    if agent_kwargs:
        for fid in list(session.farmers):
            session.set_farmer_agent(fid, agent_name, **agent_kwargs)
    if world is not None:
        session.world = world
        session._spawn_farmers()
        session._spawn_station()
    eff = session.run_season()
    eff["agent"] = agent_name
    eff["station"] = session.station_name
    return session.world, eff


def compare_agents(
    names: list[str] | None = None,
    year: int = 2022,
    days: int = 131,
) -> pd.DataFrame:
    names = names or default_roster()
    rows = []
    for i, name in enumerate(names):
        _, eff = run_season(
            agent_name=name,
            year=year,
            days=days,
            seed=i,
        )
        rows.append(eff)
    return pd.DataFrame(rows)

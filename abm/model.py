# -*- coding: utf-8 -*-
"""The ABM district as a Mesa model: one mesa.Agent per farmer + one station agent,
physics on WaterWorld (mirrors cyclic/model.py + cyclic/agents.py's own PaddyAgent /
PumpingStation / CyclicModel split — thin per-entity agents that decide, a model that
holds the real vectorized state and advances it).

    model = DistrictModel(SeasonConfig(), agent_roster=["steward", "adaptive_experimenter"])
    while not model.done():
        model.step()

A step, in order:
  1. the station agent observes pool/lake/field deficit and dispatches -> WaterWorld.apply_station
  2. every farmer agent observes its own paddies and decides apply_mm / discharge_mm
  3. WaterWorld.step() moves the water with cyclic.physics.advance (ET, D, percolation,
     perched exchange, Hooghoudt lateral), recaptures lateral flow, settles the pool
"""

from __future__ import annotations

from abm.agents import default_roster

import numpy as np

try:
    import mesa
except ImportError as error:  # pragma: no cover - mesa is a hard abm dependency
    raise ImportError("mesa is required for abm.model (pip install mesa>=3.3,<4)") from error

from .agents import list_agents, make_agent
from .station import default_station, list_stations, make_station
from .world import SeasonConfig, WaterWorld


class FarmerMesaAgent(mesa.Agent):
    """One farmer: decides apply_mm / discharge_mm for every paddy it owns.

    A thin Mesa wrapper — the decision logic lives in the wrapped ``policy``
    (an ``abm.BaseFarmer`` instance, e.g. an archetype); this class only tells
    Mesa the farmer exists and asks the policy each step. Named distinctly from
    ``abm.FarmerAgent`` (the older name of the policy base class).
    """

    def __init__(self, model: "DistrictModel", farmer_id: int, policy):
        super().__init__(model)
        self.farmer_id = int(farmer_id)
        self.policy = policy

    @property
    def name(self) -> str:
        return getattr(self.policy, "name", type(self.policy).__name__)

    def step(self) -> None:
        obs = self.model.world.observe_farmer(self.farmer_id)
        obs.update(self.model.social_observation(self.farmer_id))
        plan = self.policy.act(obs)
        ids = plan.ids
        self.model.pending_ir[ids] += plan.apply
        self.model.pending_dr[ids] += plan.drain
        self.model.pending_weir[ids] = plan.weir
        self.model.pending_request[ids] += plan.request


class StationMesaAgent(mesa.Agent):
    """The single LID irrigation-management staff agent (pool / lake control).

    Named distinctly from ``abm.station.StationAgent`` (the policy base class).
    """

    def __init__(self, model: "DistrictModel", policy):
        super().__init__(model)
        self.policy = policy

    def step(self) -> None:
        obs = self.model.world.observe_station()
        action = self.policy.decide(obs)
        self.model.last_station_state = str(action.get("state", getattr(self.policy, "state", "wait")))
        self.model.world.apply_station(action)


class DistrictModel(mesa.Model):
    """Mesa model: WaterWorld physics + a FarmerMesaAgent per farmer + one StationMesaAgent."""

    def __init__(
        self,
        config: SeasonConfig | None = None,
        agent_roster: list[str] | None = None,
        roster_weights: dict[str, float] | None = None,
        station_name: str | None = None,
        seed: int = 0,
    ):
        super().__init__(seed=seed)
        self.world = WaterWorld(config or SeasonConfig(seed=seed))
        self.agent_roster: list[str] = list(agent_roster or default_roster())
        self.roster_weights: dict[str, float] = dict(roster_weights or {})
        # None → abm.station.default_station(): a user-registered station, else "mpc".
        self.station_name = station_name or default_station()
        self.last_station_state = "wait"
        self.station_agent: StationMesaAgent | None = None
        self.farmer_agents: dict[int, FarmerMesaAgent] = {}
        self.pending_ir = np.zeros(self.world.n)
        self.pending_dr = np.zeros(self.world.n)
        self.pending_weir = np.full(self.world.n, np.inf)
        self.pending_request = np.zeros(self.world.n)
        self.farmer_performance_history = []
        self.farmer_cohort = -1
        self._spawn_station()
        self._spawn_farmers()

    # --- roster / weights -------------------------------------------------------------------

    def weight_fractions(self, roster: list[str]) -> np.ndarray:
        """Each roster name's share (sums to 1): explicit roster_weights first,
        any unset names split whatever share is left over evenly."""
        weights = np.array([max(float(self.roster_weights.get(name, 0.0)), 0.0) for name in roster])
        unset = weights <= 0
        if unset.any():
            leftover = max(1.0 - weights.sum(), 0.0)
            weights[unset] = leftover / unset.sum()
        if weights.sum() <= 0:
            weights = np.ones(len(roster))
        return weights / weights.sum()

    def effective_roster_weights(self) -> dict[str, float]:
        roster = self.agent_roster or ["steward"]
        return dict(zip(roster, self.weight_fractions(roster)))

    def _roster_assignment(self, roster: list[str], n_farmers: int) -> list[str]:
        """One roster name per farmer id, honouring roster_weights, spatially
        shuffled so a personality mix isn't a contiguous run of canal-adjacent farmers."""
        if n_farmers <= 0:
            return []
        weights = self.weight_fractions(roster)
        raw = weights * n_farmers
        counts = np.floor(raw).astype(int)
        remainder = n_farmers - int(counts.sum())
        if remainder > 0:
            order = np.argsort(-(raw - counts))
            for i in range(remainder):
                counts[order[i % len(order)]] += 1
        assignment: list[str] = []
        for name, count in zip(roster, counts):
            assignment += [name] * int(count)
        rng = np.random.default_rng(int(self.world.config.seed))
        rng.shuffle(assignment)
        return assignment

    # --- (re)spawning -------------------------------------------------------------------------

    def _spawn_farmers(self) -> None:
        self.farmer_cohort += 1
        for agent in list(self.farmer_agents.values()):
            agent.remove()
        stations = set(list_stations())
        roster = [n for n in self.agent_roster if n in list_agents() and n not in stations] or ["steward"]
        self.agent_roster = roster
        n_farmers = self.world.n_farmers()
        assignment = self._roster_assignment(roster, n_farmers)
        self.farmer_agents = {
            fid: FarmerMesaAgent(self, fid, make_agent(name, seed=self._farmer_seed(fid))) for fid, name in enumerate(assignment)
        }
        self._apply_collaboration()

    def collaborative_farmers(self) -> set[int]:
        """Farmer ids made collaborative by SeasonConfig.collaborative_share (seeded, so the
        same farmers are picked on every reset); empty when the share is None."""
        share = self.world.config.collaborative_share
        n = self.world.n_farmers()
        if share is None or n == 0:
            return set()
        order = np.random.default_rng([int(self.world.config.seed), 7919]).permutation(n)
        return set(int(f) for f in order[: int(round(float(np.clip(share, 0, 1)) * n))])

    def _apply_collaboration(self, farmer_ids=None) -> None:
        if self.world.config.collaborative_share is None:
            return
        from .agents import COLLABORATIVE, UNCOLLABORATIVE

        collaborative = self.collaborative_farmers()
        for fid in (self.farmer_agents if farmer_ids is None else farmer_ids):
            agent = self.farmer_agents[fid]
            agent.policy.set_cooperation(COLLABORATIVE if fid in collaborative else UNCOLLABORATIVE)

    def set_collaborative_share(self, share: float | None) -> None:
        self.world.config.collaborative_share = None if share is None else float(share)
        self._spawn_farmers()

    def _spawn_station(self) -> None:
        name = self.station_name if self.station_name in list_stations() else default_station()
        self.station_name = name
        if self.station_agent is not None:
            self.station_agent.remove()
        self.station_agent = StationMesaAgent(self, make_station(name))

    def set_roster(self, names: list[str]) -> None:
        valid = set(list_agents())
        self.agent_roster = [n for n in names if n in valid] or ["steward"]
        self._spawn_farmers()

    def set_roster_weights(self, weights: dict[str, float]) -> None:
        self.roster_weights = dict(weights)
        self._spawn_farmers()

    def set_station(self, name: str, **kwargs) -> None:
        self.station_name = name
        self._spawn_station()
        if kwargs:
            self.station_agent.policy = make_station(self.station_name, **kwargs)

    def set_paddies_per_farmer(self, value: int) -> None:
        self.world.set_paddies_per_farmer(int(value))
        self._spawn_farmers()

    def set_farm_size_variability(self, value: float) -> None:
        self.world.set_farm_size_variability(float(value))
        self._spawn_farmers()

    def set_farmer_agent(self, farmer_id: int, agent_name: str, **kwargs) -> None:
        self.farmer_cohort += 1
        policy = make_agent(agent_name, **dict({"seed": self._farmer_seed(farmer_id)}, **kwargs))
        existing = self.farmer_agents.get(int(farmer_id))
        if existing is not None:
            existing.policy = policy
        else:
            self.farmer_agents[int(farmer_id)] = FarmerMesaAgent(self, int(farmer_id), policy)
        self._apply_collaboration([int(farmer_id)])

    def _farmer_seed(self, farmer_id):
        return int(np.random.SeedSequence([int(self.world.config.seed), int(farmer_id)]).generate_state(1)[0])

    def social_observation(self, farmer_id):
        """Only other owners of spatially adjacent paddies; one vote per farmer."""
        snapshot = getattr(self, '_social_snapshot', {})
        neighbour_owners = set()
        for p in self.world.paddies_of(farmer_id):
            neighbour_owners.update(int(self.world.owner_id[j]) for j in self.world.neighbour_ids(p))
        neighbour_owners.discard(farmer_id)
        return dict(social_neighbours=[dict(snapshot[f]) for f in sorted(neighbour_owners) if f in snapshot],
                    previous_social_state=dict(snapshot.get(farmer_id, {})),
                    field_deficit_m3=float(np.dot(np.maximum(self.world.today_growth().target_depth_mm - self.world.ponding_mm, 0), self.world.area_m2)/1000))

    def sync_crop_calendars(self) -> None:
        """Each paddy's crop runs on its owner's calendar (eagerness shift), so the
        physics' canopy/ET follow the farmer's own transplanting date."""
        shift = np.zeros(self.world.n, dtype=int)
        for fid, agent in self.farmer_agents.items():
            shift[self.world.paddies_of(fid)] = int(getattr(agent.policy, "calendar_shift_days", 0))
        self.world.calendar_shift_days = shift

    # --- run loop -----------------------------------------------------------------------------

    def step(self, station_action: dict | None = None):
        """One day. ``station_action`` (withdraw_m3 / outflow_m3 / irrigation_sent_mm)
        replaces the station agent's own decision — how an external optimiser or RL
        policy drives the staff role (see abm.gym_env)."""
        self.pending_ir = np.zeros(self.world.n)
        self.pending_dr = np.zeros(self.world.n)
        self.pending_weir = np.full(self.world.n, np.inf)
        self.pending_request = np.zeros(self.world.n)
        self.sync_crop_calendars()
        self._social_snapshot = {fid: dict(agent.policy.social_state(), farmer_id=fid)
            for fid, agent in self.farmer_agents.items() if hasattr(agent.policy, "social_state")}
        if station_action is None:
            self.station_agent.step()
        else:
            self.last_station_state = str(station_action.get("state", "external"))
            self.world.apply_station(station_action)
        for agent in self.farmer_agents.values():
            agent.step()
        result = self.world.step(self.pending_ir, self.pending_dr, weir_mm=self.pending_weir,
                                 requested_mm=self.pending_request)
        for fid, agent in self.farmer_agents.items():
            if hasattr(agent.policy, "record_outcome"):
                agent.policy.record_outcome(self.world, self.world.paddies_of(fid))
        from .metrics import record_farmer_performance
        self.farmer_performance_history.extend(record_farmer_performance(self, result))
        return result

    def done(self) -> bool:
        return self.world.done()

    def reset(self) -> dict:
        obs = self.world.reset()
        self.farmer_performance_history = []
        self.farmer_cohort = -1
        self._spawn_station()
        self._spawn_farmers()
        return obs

    def roster_names(self) -> dict[int, str]:
        return {fid: agent.name for fid, agent in self.farmer_agents.items()}

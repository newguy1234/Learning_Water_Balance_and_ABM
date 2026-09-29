# -*- coding: utf-8 -*-
"""Farmer registry and the built-in personality archetypes.

Every farmer — built-in or registered from your own script — inherits from
``abm.farmer.BaseFarmer`` (the common personality variables, crop calendar, lower/upper
bounds, part-time checking, forecast use, weir board, pricing check and hard
constraints). ``register_farmer`` refuses any other class.

Conceptual model (Craciun): farmers open/close inlet/outlet to hold ideal height;
canal water must already be present (``irrigation_sent_mm``). They cannot invent IR.
"""

from __future__ import annotations

import numpy as np

from .farmer import (  # noqa: F401  (re-exported for older imports)
    BaseFarmer,
    Constraint,
    FARMER_DEFAULTS,
    NEIGHBOUR_DEFAULTS,
    NoDrainHold,
    Plan,
    TRAITS,
)

AGENTS: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        if not (isinstance(cls, type) and issubclass(cls, BaseFarmer)):
            raise TypeError(f"farmer {name!r} must inherit from abm.BaseFarmer "
                            f"(got {getattr(cls, '__mro__', cls)})")
        cls.name = name
        AGENTS[name] = cls
        return cls

    return deco


register_farmer = register
register_agent = register


def make_agent(name: str, **kwargs):
    if name not in AGENTS:
        raise KeyError(f"unknown agent {name!r}; choose from {sorted(AGENTS)}")
    return AGENTS[name](**kwargs)


def list_agents() -> list[str]:
    return sorted(AGENTS)


def default_roster() -> list[str]:
    """Farmer types used when none are named: the built-in archetypes plus every farmer
    registered outside the ``abm`` package (your own script, e.g. ``example.py``)."""
    custom = [name for name, cls in AGENTS.items()
              if not (getattr(cls, "__module__", "") or "").startswith("abm")]
    return list(DEFAULT_ROSTER) + [name for name in custom if name not in DEFAULT_ROSTER]


# Older name for the base class.
FarmerAgent = BaseFarmer


# Section 7 / Tables 5-6 of Conceptual_Model.pdf, extended with the calendar, band,
# checking, forecast, flow-through and price traits. Numerical values are scenario
# assumptions, not fitted parameters or measurements from the paper.
ARCHETYPES = {
    # Wealthy, full-time, well staffed; keeps fields full and starts first.
    'maximiser': dict(
        traits=dict(wealth=.9, experience=.6, knowledge=.6, labour=.9, cooperation=.2, work_ethic=.9,
                    risk_tolerance=.3, neighbour_offset=.6, eagerness=.6, drawdown=.2, overfill=.8,
                    part_time=.1, foresight=.5, flow_through=.5, price_sensitivity=.2),
        target_multiplier=1.3, learning_rate=.08, social_influence=.15, evidence_threshold=.05),
    # Poor, short of hands, off-farm job on weekdays; lets the pond run low, watches cost.
    'survivalist': dict(
        traits=dict(wealth=.2, experience=.5, knowledge=.5, labour=.3, cooperation=.8, work_ethic=.7,
                    risk_tolerance=.4, neighbour_offset=-.5, eagerness=-.3, drawdown=.7, overfill=.2,
                    part_time=.7, foresight=.3, flow_through=.1, price_sensitivity=.9),
        target_multiplier=.95, learning_rate=.12, social_influence=.45, evidence_threshold=.03),
    # Experienced, knowledgeable, cooperative; reads forecasts, modest band.
    'steward': dict(
        traits=dict(wealth=.6, experience=.9, knowledge=.9, labour=.7, cooperation=.95, work_ethic=.9,
                    risk_tolerance=.6, neighbour_offset=-.3, eagerness=.2, drawdown=.6, overfill=.3,
                    part_time=.2, foresight=.8, flow_through=.1, price_sensitivity=.6),
        target_multiplier=.95, learning_rate=.10, social_influence=.3, evidence_threshold=.03),
    # Risk-averse veteran: never lets the pond drop far, overfills, early.
    'guardian': dict(
        traits=dict(wealth=.6, experience=.95, knowledge=.7, labour=.6, cooperation=.5, work_ethic=.8,
                    risk_tolerance=.1, neighbour_offset=.5, eagerness=.4, drawdown=.1, overfill=.7,
                    part_time=.2, foresight=.4, flow_through=.3, price_sensitivity=.3),
        target_multiplier=1.2, learning_rate=.02, social_influence=.1, evidence_threshold=.15),
    # Tries drying (deep drawdown) and trusts forecasts; copies what works.
    'adaptive_experimenter': dict(
        traits=dict(wealth=.6, experience=.4, knowledge=.85, labour=.8, cooperation=.7, work_ethic=.85,
                    risk_tolerance=.85, neighbour_offset=-.2, eagerness=0., drawdown=.8, overfill=.4,
                    part_time=.3, foresight=.9, flow_through=.1, price_sensitivity=.6),
        target_multiplier=1., learning_rate=.3, social_influence=.65, evidence_threshold=.01),
    # Low effort, mostly away, late; leaves the inlet running rather than visiting.
    'passive_operator': dict(
        traits=dict(wealth=.5, experience=.5, knowledge=.4, labour=.35, cooperation=.4, work_ethic=.2,
                    risk_tolerance=.4, neighbour_offset=0., eagerness=-.7, drawdown=.5, overfill=.5,
                    part_time=.8, foresight=.1, flow_through=.6, price_sensitivity=.3),
        target_multiplier=1.1, learning_rate=.01, social_influence=.1, evidence_threshold=.1),
    # Takes whatever the canal offers; runs water through when it can.
    'opportunist': dict(
        traits=dict(wealth=.5, experience=.6, knowledge=.6, labour=.7, cooperation=.1, work_ethic=.7,
                    risk_tolerance=.75, neighbour_offset=.3, eagerness=.1, drawdown=.4, overfill=.6,
                    part_time=.4, foresight=.6, flow_through=.7, price_sensitivity=.4),
        target_multiplier=1.15, learning_rate=.2, social_influence=.5, evidence_threshold=.02),
}
DEFAULT_ROSTER = tuple(ARCHETYPES)

# Collaboration is one variable every farmer carries (the ``cooperation`` trait, 0..1). A
# scenario can overwrite it for everyone with ``SeasonConfig(collaborative_share=...)``;
# see BaseFarmer.set_cooperation.
COLLABORATIVE = .95
UNCOLLABORATIVE = .05


class PersonalityFarmer(BaseFarmer):
    """BaseFarmer plus lagged, outcome-based social learning (the archetypes).

    On a check day it compares its last score with the adjacent owners' and, if one did
    clearly better, moves its target multiplier, cooperation and drawdown part of the
    way toward that farmer's. Stress pushes its target up, canal scarcity pulls it down
    and makes it more cooperative.
    """
    defaults = dict(learning_rate=.1, social_influence=.3, evidence_threshold=.03)
    bounds = dict(learning_rate=(0., 1.), social_influence=(0., 1.),
                  trait_variability=(0., .25), evidence_threshold=(0., .5),
                  target_multiplier=(.65, 1.5), neighbour_offset_mm=(0., 40.),
                  neighbour_weight=(0., 1.), neighbour_max_shift_mm=(1., 60.),
                  neighbour_memory=(0., 1.), max_shift_days=(0, 10))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.copied_from = None
        self.copy_events = 0
        self.score = None
        self.memory = []

    def social_state(self):
        return dict(target_multiplier=self.target_multiplier, cooperation=self.cooperation,
                    drawdown=self.traits['drawdown'], score=self.score)

    def plan(self, obs):
        self.copied_from = None
        return super().plan(obs)

    def learn(self, obs):
        rate = float(np.clip(self.params['learning_rate'], 0, 1))
        peers = [p for p in obs.get('social_neighbours', []) if p['score'] is not None]
        own = obs.get('previous_social_state', self.social_state())
        best = max(peers, key=lambda p: (p['score'], -p['farmer_id'])) if peers else None
        if best is not None and own.get('score') is not None and best['score'] > own['score'] + self.params['evidence_threshold']:
            copy = rate * float(np.clip(self.params['social_influence'], 0, 1))
            if copy > 0:
                self.target_multiplier += copy * (best['target_multiplier'] - self.target_multiplier)
                self.cooperation += copy * (best['cooperation'] - self.cooperation)
                if 'drawdown' in best:
                    self.traits['drawdown'] += copy * (best['drawdown'] - self.traits['drawdown'])
                self.copied_from = int(best['farmer_id'])
                self.copy_events += 1
        if self.memory:
            last = self.memory[-1]
            # Scarcity promotes lower demand and coordination; experienced farmers
            # adjust less to one stressful observation.
            self.target_multiplier += rate * ((1-last['ks']) * (1-.5*self.traits['experience']) - .2*last['scarcity'])
            self.cooperation += rate * .1 * last['scarcity'] * (1-self.cooperation)
        self.target_multiplier = float(np.clip(self.target_multiplier, .65, 1.5))
        self.traits['drawdown'] = float(np.clip(self.traits['drawdown'], 0, 1))

    def adjust(self, obs, plan):
        rate = float(np.clip(self.params['learning_rate'], 0, 1))
        sent = np.asarray(obs['owned_irrigation_sent_mm'], dtype=float)
        if self.name == 'opportunist':
            # Takes surplus canal water beyond what it asked for.
            plan.apply = plan.apply + .25 * np.maximum(sent - plan.request, 0.) * (1-self.cooperation)
        if self.name == 'adaptive_experimenter' and rate > 0:
            # Reproducible bounded exploration, keyed by day, never iteration order.
            rng = np.random.default_rng(np.random.SeedSequence([int(self.params['seed']), int(obs['day'])]))
            factor = float(np.clip(1 + rng.normal(0, .1*rate), .9, 1.1))
            plan.request, plan.apply = plan.request * factor, plan.apply * factor

    def record_outcome(self, world, ids):
        if not len(ids):
            return
        weights = world.area_m2[ids]
        avg = lambda a: float(np.average(np.asarray(a)[ids], weights=weights))
        ks, irrigation = avg(world.ks), avg(world.last_ir)
        # world.step has already cleared irrigation_sent_mm; the day's send is in history.
        sent = world.history[-1].irrigation_sent_mm if world.history else world.irrigation_sent_mm
        scarcity = float(np.clip(1 - avg(sent)/max(self.last_demand_mm, 1e-9), 0, 1)) if self.last_demand_mm else 0.
        # Shared observable proxy, not economic profit or a causal yield estimate.
        self.score = ks - .15*irrigation/40. - .05*avg(world.last_dr)/30.
        self.memory.append(dict(day=world.day-1, ks=ks, irrigation_mm=irrigation,
                                scarcity=scarcity, score=self.score, copied_from=self.copied_from))


for _name, _spec in ARCHETYPES.items():
    _defaults = dict(PersonalityFarmer.defaults, **{k: v for k, v in _spec.items() if k != 'traits'})
    register(_name)(type(''.join(p.title() for p in _name.split('_')) + 'Farmer', (PersonalityFarmer,),
                         {'defaults': _defaults, 'traits_mean': dict(_spec['traits']), '__module__': __name__}))

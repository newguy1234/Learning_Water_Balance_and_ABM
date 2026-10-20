# -*- coding: utf-8 -*-
"""Farmer registry and the built-in personality archetypes.

Every farmer — built-in or registered from your own script — inherits from
``abm.farmer.BaseFarmer`` (cooperation, work pattern, the stage ponding depth, the
weir board and the hard constraints). ``register_farmer``
refuses any other class.

Conceptual model (Craciun): farmers open/close inlet/outlet to hold ideal height;
canal water must already be present (``irrigation_sent_mm``). They cannot invent IR.
"""

from __future__ import annotations

import numpy as np

from .farmer import (  # noqa: F401  (re-exported for older imports)
    BaseFarmer,
    Constraint,
    FARMER_DEFAULTS,
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


# Two traits only. Centres are scenario assumptions, not fitted parameters.
# Learning and social copying stay off: the district screen showed social copying
# does not move water, and the adaptation rule's demand swing is the controller
# reacting to unmet requests.
ARCHETYPES = {
    'maximiser': dict(traits=dict(cooperation=.2, work_pattern=.1)),
    'survivalist': dict(traits=dict(cooperation=.8, work_pattern=.7)),
    'steward': dict(traits=dict(cooperation=.95, work_pattern=.2)),
    'guardian': dict(traits=dict(cooperation=.5, work_pattern=.2)),
    'adaptive_experimenter': dict(traits=dict(cooperation=.7, work_pattern=.3)),
    'passive_operator': dict(traits=dict(cooperation=.4, work_pattern=.8)),
    'opportunist': dict(traits=dict(cooperation=.1, work_pattern=.4)),
}
DEFAULT_ROSTER = tuple(ARCHETYPES)

# A scenario can overwrite cooperation with SeasonConfig(collaborative_share=...).
COLLABORATIVE = .95
UNCOLLABORATIVE = .05


class PersonalityFarmer(BaseFarmer):
    """BaseFarmer. Archetypes differ on cooperation and work pattern.

    Learning and social copying are present and off (rate 0, social weight 0). A
    scenario that sets ``learning_rate`` above 0 can still move the stage-depth
    multiplier from stress, scarcity, and a better neighbour.
    """
    defaults = dict(learning_rate=0., social_influence=0., evidence_threshold=.03)
    bounds = dict(learning_rate=(0., 1.), social_influence=(0., 1.),
                  trait_variability=(0., .25), evidence_threshold=(0., .5),
                  target_multiplier=(.65, 1.5))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.copied_from = None
        self.copy_events = 0
        self.score = None
        self.memory = []

    def social_state(self):
        return dict(target_multiplier=self.target_multiplier, score=self.score)

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
                self.copied_from = int(best['farmer_id'])
                self.copy_events += 1
        if self.memory:
            last = self.memory[-1]
            # Scarcity lowers the target. Stress raises it.
            self.target_multiplier += rate * ((1-last['ks']) - .2*last['scarcity'])
        self.target_multiplier = float(np.clip(self.target_multiplier, .65, 1.5))

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
                         {'defaults': _defaults, 'traits_mean': dict(_spec.get('traits', {})), '__module__': __name__}))

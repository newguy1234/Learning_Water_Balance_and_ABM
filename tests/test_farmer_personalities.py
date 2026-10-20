import numpy as np
import pytest
from abm.agents import DEFAULT_ROSTER, make_agent, list_agents
from abm.model import DistrictModel
from abm.world import SeasonConfig

pytestmark = pytest.mark.usefixtures("abm_builtins_only")


def observation():
    return dict(day=1, owned_ids=np.array([0,1]), owned_ponding_mm=np.array([10.,20.]),
        owned_irrigation_sent_mm=np.array([20.,20.]), ponding_mm=np.array([10.,20.]),
        irrigation_sent_mm=np.array([20.,20.]), rain_mm=0., et_mm=5., target_depth_mm=50.,
        pool_m3=100., neighbour_mean_mm=np.array([30.,30.]))


def test_default_roster_replaces_base_farmers():
    assert set(list_agents()) == set(DEFAULT_ROSTER)


def test_copy_successful_neighbour_changes_actions():
    obs=observation()
    farmer=make_agent('adaptive_experimenter', seed=9, learning_rate=1., social_influence=1.)
    control=make_agent('adaptive_experimenter', seed=9, learning_rate=1., social_influence=0.)
    farmer.score=control.score=.1
    obs['social_neighbours']=[dict(farmer_id=3, score=.9, target_multiplier=.8)]
    obs.update(owned_irrigation_sent_mm=np.array([200.,200.]), irrigation_sent_mm=np.array([200.,200.]))
    ir, dr=farmer.decide(obs)
    baseline,_=control.decide(obs)
    assert farmer.copied_from == 3
    assert farmer.copy_events == 1
    assert np.any(ir != baseline)
    assert farmer.target_multiplier < control.target_multiplier
    assert farmer.traits["cooperation"] == pytest.approx(control.traits["cooperation"])
    assert farmer.traits["work_pattern"] == pytest.approx(control.traits["work_pattern"])


def test_guardian_needs_stronger_evidence():
    obs=observation()
    obs['social_neighbours']=[dict(farmer_id=3, score=.55, target_multiplier=.8)]
    guardian=make_agent('guardian'); guardian.score=.5
    guardian.decide(obs)
    assert guardian.copy_events == 0


@pytest.mark.parametrize('name', DEFAULT_ROSTER)
def test_personality_respects_ownership_and_supply(name):
    p=make_agent(name)
    obs=observation()
    obs.update(ponding_mm=np.array([10.,20.,30.]), irrigation_sent_mm=np.array([20.,20.,20.]))
    ir,dr=p.decide(obs)
    assert ir[2] == dr[2] == 0
    assert np.all(ir >= 0) and np.all(ir <= obs['irrigation_sent_mm'])
    assert np.all(dr >= 0)


def model():
    return DistrictModel(SeasonConfig(use_cyclic_map=False, days=6, paddies_per_farmer=3, seed=7))


def test_snapshot_order_independence_reset_and_local_network():
    a,b=model(),model()
    b.farmer_agents=dict(reversed(list(b.farmer_agents.items())))
    for _ in range(6):
        a.step(); b.step()
    np.testing.assert_allclose(a.world.ponding_mm,b.world.ponding_mm)
    for fid in a.farmer_agents:
        p,q=a.farmer_agents[fid].policy,b.farmer_agents[fid].policy
        assert p.memory == q.memory
        assert p.social_state() == q.social_state()
        neighbours=a.social_observation(fid)['social_neighbours']
        visible={int(a.world.owner_id[j]) for i in a.world.paddies_of(fid) for j in a.world.neighbour_ids(i)}-{fid}
        assert {p['farmer_id'] for p in neighbours} == visible
    traits=a.farmer_agents[0].policy.initial_traits.copy()
    a.reset()
    assert a.farmer_agents[0].policy.memory == []
    assert a.farmer_agents[0].policy.traits == traits
    assert set(traits) == {"cooperation", "work_pattern"}


def test_archetypes_differ_on_cooperation_and_work_pattern():
    from abm.farmer import TRAIT_NAMES
    assert TRAIT_NAMES == ("cooperation", "work_pattern")
    pinned = dict(seed=0, trait_variability=0.)
    farmers = {name: make_agent(name, **pinned) for name in DEFAULT_ROSTER}
    assert farmers["steward"].cooperation > farmers["survivalist"].cooperation > farmers["maximiser"].cooperation
    assert farmers["passive_operator"].traits["work_pattern"] > farmers["survivalist"].traits["work_pattern"] > farmers["maximiser"].traits["work_pattern"]
    assert all(f.params["learning_rate"] == 0 and f.params["social_influence"] == 0 for f in farmers.values())


def test_cooperation_pulls_the_target_toward_neighbours():
    obs = observation()
    obs["neighbour_mean_mm"] = np.array([90., 90.])

    def target(cooperation):
        farmer = make_agent("passive_operator", seed=1, trait_variability=0., cooperation=cooperation)
        farmer.decide(obs)
        return farmer.last_target_mm

    assert np.all(target(1.0) > target(0.0))

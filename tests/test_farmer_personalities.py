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
    obs['social_neighbours']=[dict(farmer_id=3, score=.9, target_multiplier=.8,
        cooperation=.9, drawdown_tolerance=0.)]
    obs.update(owned_irrigation_sent_mm=np.array([200.,200.]), irrigation_sent_mm=np.array([200.,200.]))
    ir, dr=farmer.decide(obs)
    baseline,_=control.decide(obs)
    assert farmer.copied_from == 3
    assert farmer.copy_events == 1
    assert np.any(ir != baseline)
    assert farmer.target_multiplier < control.target_multiplier
    assert farmer.traits['drawdown_tolerance'] < control.traits['drawdown_tolerance']


def test_guardian_needs_stronger_evidence():
    obs=observation()
    obs['social_neighbours']=[dict(farmer_id=3, score=.55, target_multiplier=.8,
        cooperation=.9, drawdown_tolerance=0.)]
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
    return DistrictModel(SeasonConfig(days=6, paddies_per_farmer=3, seed=7))


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
    assert a.farmer_agents[0].policy.traits != a.farmer_agents[1].policy.traits


def _target_with(offset, neighbour_mm, **params):
    obs = observation()
    obs['neighbour_mean_mm'] = np.array([neighbour_mm, neighbour_mm], dtype=float)
    farmer = make_agent('passive_operator', seed=1, learning_rate=0., **params)
    farmer.traits['neighbour_wetness_bias'] = offset
    farmer.decide(obs)
    return farmer, farmer.last_target_mm


def test_neighbour_wetness_bias_sets_direction_relative_to_neighbours():
    _, wet = _target_with(1.0, 50.0)
    _, neutral = _target_with(0.0, 50.0)
    _, dry = _target_with(-1.0, 50.0)
    assert np.all(wet > neutral) and np.all(neutral > dry)
    # Wetter neighbours pull the same farmer's target up.
    _, beside_wet = _target_with(0.0, 90.0)
    assert np.all(beside_wet > neutral)


def test_neighbour_pull_is_soft_and_capped():
    base, _ = _target_with(0.0, 0.0, neighbour_weight=0.0)
    stage = base.last_target_mm
    _, extreme = _target_with(1.0, 10_000.0)
    p = make_agent('passive_operator').params
    assert np.all(extreme - stage <= p['neighbour_weight'] * p['neighbour_max_shift_mm'] + 1e-9)
    assert np.all(extreme - stage > 0)


def test_neighbour_level_is_remembered_not_snapshotted():
    farmer, _ = _target_with(0.0, 30.0)
    obs = observation()
    obs['neighbour_mean_mm'] = np.array([80., 80.])
    obs['day'] = 1 + farmer.monitor_interval  # next day this farmer checks its paddies
    farmer.decide(obs)
    memory = farmer.params['neighbour_memory']
    np.testing.assert_allclose(list(farmer.neighbour_level.values()), 30 + memory * (80 - 30))


def test_archetypes_differ_in_neighbour_wetness_bias():
    offsets = {name: make_agent(name, seed=0).traits['neighbour_wetness_bias'] for name in DEFAULT_ROSTER}
    assert offsets['maximiser'] > offsets['passive_operator'] > offsets['survivalist']
    assert all(-1 <= v <= 1 for v in offsets.values())


def test_any_registered_farmer_can_use_the_neighbour_target():
    import abm

    class Plain(abm.FarmerAgent):
        defaults = dict(neighbour_wetness_bias=1.0)

        def decide_owned(self, obs):
            target = self.neighbour_target(obs, obs['target_depth_mm'])
            return np.zeros_like(target), np.zeros_like(target)

    obs = observation()
    obs['neighbour_mean_mm'] = np.array([50., 50.])  # neighbours sit at the stage target
    farmer = Plain()
    wetter = farmer.neighbour_target(obs, 50.0)
    assert np.all(wetter > 50.0)
    assert np.all(wetter - 50.0 <= 0.3 * 20.0 + 1e-9)


def test_collaborative_levels_with_neighbours_uncollaborative_keeps_its_own():
    wet_obs = observation()
    wet_obs['neighbour_mean_mm'] = np.array([90., 90.])
    dry_obs = observation()
    dry_obs['neighbour_mean_mm'] = np.array([10., 10.])

    def target(cooperation, obs, offset):
        farmer = make_agent('passive_operator', seed=1, learning_rate=0.)
        farmer.traits['neighbour_wetness_bias'] = offset
        farmer.set_cooperation(cooperation)
        farmer.decide(obs)
        return farmer.last_target_mm

    # Collaborative: follows the neighbours up and down, whatever its own offset.
    assert np.all(target(1.0, wet_obs, -1.0) > target(1.0, dry_obs, -1.0))
    # Uncollaborative: the neighbours make no difference, only its own offset does.
    np.testing.assert_allclose(target(0.0, wet_obs, 0.5), target(0.0, dry_obs, 0.5))
    assert np.all(target(0.0, wet_obs, 1.0) > target(0.0, wet_obs, -1.0))


def test_collaborative_share_sets_every_farmers_cooperation():
    from abm.agents import COLLABORATIVE, UNCOLLABORATIVE
    for share, expected in ((1.0, {COLLABORATIVE}), (0.0, {UNCOLLABORATIVE})):
        model = DistrictModel(SeasonConfig(days=1, collaborative_share=share), agent_roster=list(DEFAULT_ROSTER))
        levels = {a.policy.cooperation_level for a in model.farmer_agents.values()}
        assert levels == expected
    model = DistrictModel(SeasonConfig(days=1, collaborative_share=0.5, seed=4), agent_roster=list(DEFAULT_ROSTER))
    n = len(model.farmer_agents)
    collaborative = [a for a in model.farmer_agents.values() if a.policy.cooperation_level == COLLABORATIVE]
    assert len(collaborative) == round(0.5 * n)
    # Same farmers picked again after a reset (seeded).
    before = model.collaborative_farmers()
    model.reset()
    assert model.collaborative_farmers() == before


def test_collaborative_share_reaches_custom_registered_farmers():
    import abm
    from abm.agents import AGENTS, COLLABORATIVE

    @abm.register_farmer("plain_test_farmer")
    class Plain(abm.FarmerAgent):
        pass

    try:
        model = DistrictModel(SeasonConfig(days=1, collaborative_share=1.0), agent_roster=["plain_test_farmer"])
        assert all(a.policy.params["cooperation"] == COLLABORATIVE for a in model.farmer_agents.values())
    finally:
        AGENTS.pop("plain_test_farmer", None)

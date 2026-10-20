import numpy as np
import pytest

import abm
from abm import crop
from abm.agents import AGENTS, make_agent
from abm.farmer import BaseFarmer, NoDrainHold, in_drain_window

pytestmark = pytest.mark.usefixtures("abm_builtins_only")

# A plain farmer with the stage depth unscaled, for exact checks.
PINNED = dict(cooperation=0., work_pattern=0., neighbour_weight=0.,
              target_multiplier=1., schedule_jitter=0., week_swap=0.)


def farmer(**params):
    return BaseFarmer(**{**PINNED, **params})


def observation(day=20, pond=60., sent=100., **extra):
    obs = dict(day=day, owned_ids=np.array([0]), owned_ponding_mm=np.array([pond]),
               owned_irrigation_sent_mm=np.array([sent]), ponding_mm=np.array([pond]),
               irrigation_sent_mm=np.array([sent]), rain_mm=0., et_mm=5., et0_mm=4.,
               owned_et_demand_mm=np.array([5.]), owned_seepage_mm=np.array([0.]),
               target_depth_mm=crop.target_depth_mm(day), neighbour_mean_mm=np.array([pond]),
               rain_forecast_mm=np.zeros(6), bund_mm=250.)
    obs.update(extra)
    return obs


def test_register_refuses_farmers_that_do_not_inherit_base_farmer():
    with pytest.raises(TypeError):
        @abm.register_farmer("not_a_farmer")
        class NotAFarmer:
            pass
    assert "not_a_farmer" not in AGENTS
    assert abm.FarmerAgent is BaseFarmer


def test_only_cooperation_and_work_pattern():
    from abm.farmer import TRAIT_NAMES
    assert TRAIT_NAMES == ("cooperation", "work_pattern")
    a, b = make_agent("steward", seed=1), make_agent("steward", seed=2)
    assert a.traits["work_pattern"] != b.traits["work_pattern"]
    assert make_agent("steward", seed=1, work_pattern=.3).traits["work_pattern"] == .3


def test_part_time_work_pattern_checks_on_days_off():
    f = farmer(work_pattern=1.)
    f.act(observation(day=20, weekday=5))
    assert f.checked_today
    f.act(observation(day=21, weekday=0))
    assert not f.checked_today
    assert f.days_to_next_check(observation(day=21, weekday=0)) == 5


def test_midseason_drain_opens_the_lower_bound():
    f = farmer()
    day = crop.MIDSUMMER_START
    t = f.stage_target_mm({}, day)
    lower, upper = f.bounds_mm(np.array([t]), day, {"bund_mm": 250.})
    assert lower[0] == 0 and upper[0] == t
    day = 20
    assert not in_drain_window(day)
    t = f.stage_target_mm({}, day)
    lower, upper = f.bounds_mm(np.array([t]), day, {"bund_mm": 250.})
    assert lower[0] == t and upper[0] == t


def test_tops_up_to_the_stage_depth():
    target = crop.target_depth_mm(20)
    # Already above the stage depth after today's loss: no request.
    assert farmer().act(observation(pond=target + 30)).request[0] == 0
    # Below the stage depth: replace the deficit and today's loss.
    plan = farmer().act(observation(pond=target - 30))
    np.testing.assert_allclose(plan.request, 35)
    np.testing.assert_allclose(plan.weir, target)


def test_no_drain_hold_is_hard_even_for_a_custom_decision():
    class Drainer(BaseFarmer):
        def decide_owned(self, obs):
            return np.zeros(1), np.full(1, 50.)

    f = Drainer(**PINNED)
    held = f.act(observation(day=6, pond=10.))       # herbicide hold, crop days 5..11
    assert held.drain[0] == 0 and held.weir[0] == 250.
    assert held.apply[0] >= 30 - 10                  # topped up toward the 30 mm minimum
    free = Drainer(**PINNED).act(observation(day=30, pond=60.))
    assert free.drain[0] == 50.


def test_extra_constraints_can_be_added():
    class Careful(BaseFarmer):
        constraints = BaseFarmer.constraints + (NoDrainHold("extra", start_day=40, days=2),)

    f = Careful(**PINNED)
    assert f.act(observation(day=40)).weir[0] == 250.
    assert Careful(**PINNED).act(observation(day=45)).weir[0] < 250.


def test_days_off_vary_slightly_between_farmers_and_weeks():
    farmers = [BaseFarmer(seed=s) for s in range(200)]
    usual = {f.days_off for f in farmers}
    assert (5, 6) in usual and usual <= {(5, 6), (4, 5), (0, 6)} and len(usual) > 1
    share = sum(f.days_off == (5, 6) for f in farmers) / len(farmers)
    assert .55 < share < .85  # most keep Sat+Sun
    f = farmers[0]
    weeks = {f.days_off_in_week(7 * w) for w in range(40)}
    assert f.days_off in weeks and len(weeks) > 1  # an occasional week moves a day off


def test_weir_board_spills_above_the_crest():
    from cyclic.physics import Physics, State, advance

    p = Physics().replace(bund_mm=250.)
    s = State.start(2, 0., 10.)
    s.water[:] = 100.
    flux = advance(s, 0., 0., .5, np.zeros(2), np.zeros(2), p, weir_mm=np.array([60., np.inf]))
    assert flux["discharge_mm"][0] == pytest.approx(40.)
    assert flux["discharge_mm"][1] == 0.


def test_world_runs_each_paddy_on_its_owners_calendar():
    from abm.model import DistrictModel
    from abm.world import SeasonConfig

    model = DistrictModel(SeasonConfig(use_cyclic_map=False, days=3, paddies_per_farmer=6, seed=2))
    model.step()
    shifts = model.world.calendar_shift_days
    for fid, agent in model.farmer_agents.items():
        assert set(shifts[model.world.paddies_of(fid)]) == {agent.policy.calendar_shift_days}
    day = crop.L_INI  # canopy starts to close here, so a shift shows
    cover = model.world.canopy_cover(day)
    expected = [crop.canopy_cover(day + s) for s in shifts]
    np.testing.assert_allclose(cover, expected)
    step = model.world.history[-1]
    assert step.requested_mm is not None and step.weir_mm is not None

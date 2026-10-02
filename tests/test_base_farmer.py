import numpy as np
import pytest

import abm
from stem import crop
from abm.agents import AGENTS, make_agent
from abm.farmer import BaseFarmer, NoDrainHold, in_drain_window

pytestmark = pytest.mark.usefixtures("abm_builtins_only")

# A plain farmer with every trait pinned (no personality noise) for exact checks.
PINNED = dict(calendar_lead=0., drawdown_tolerance=1., refill_overshoot=1., off_farm_work=0.,
              forecast_reliance=0., flow_through_habit=0., risk_tolerance=1., water_accounting=1.,
              cooperation=0., neighbour_wetness_bias=0., work_ethic=1., neighbour_weight=0.,
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


def test_calendar_lead_shifts_the_crop_calendar_up_to_four_days():
    assert farmer(calendar_lead=1.).calendar_shift_days == 4
    assert farmer(calendar_lead=-1.).calendar_shift_days == -4
    assert farmer(calendar_lead=.5).crop_day(10) == 12


def test_traits_vary_between_farmers_of_one_kind_unless_pinned():
    a, b = make_agent("steward", seed=1), make_agent("steward", seed=2)
    assert a.traits["off_farm_work"] != b.traits["off_farm_work"]
    assert make_agent("steward", seed=1, off_farm_work=.3).traits["off_farm_work"] == .3


def test_bounds_follow_the_shifted_target():
    early, late = farmer(calendar_lead=1.), farmer(calendar_lead=-1.)
    day = crop.MIDSUMMER_START - 2  # early farmer is already in its mid-season drain
    assert in_drain_window(early.crop_day(day)) and not in_drain_window(late.crop_day(day))
    t = early.stage_target_mm({}, early.crop_day(day))
    lower, upper = early.bounds_mm(np.array([t]), early.crop_day(day), {"bund_mm": 250.})
    assert lower[0] == 0 and upper[0] == t
    t = late.stage_target_mm({}, late.crop_day(day))
    lower, upper = late.bounds_mm(np.array([t]), late.crop_day(day), {"bund_mm": 250.})
    assert lower[0] == t - 25 and upper[0] == t + 20


def test_fill_and_let_drop():
    f = farmer()
    target = crop.target_depth_mm(20)
    # Inside the band with little loss: let it drop, no request.
    plan = f.act(observation(pond=target - 5))
    assert plan.request[0] == 0
    # Below the lower bound: refill to the upper bound (+ today's loss).
    f = farmer()
    plan = f.act(observation(pond=target - 30))
    np.testing.assert_allclose(plan.request, target + 20 - (target - 30) + 5)
    np.testing.assert_allclose(plan.weir, target + 20)


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


def test_forecast_rain_holds_back_irrigation_for_farmers_who_trust_it():
    wet = observation(pond=20., rain_forecast_mm=np.array([0., 30., 30., 0., 0., 0.]))
    blind = farmer(forecast_reliance=0.).act(dict(wet)).request[0]
    seeing = farmer(forecast_reliance=1.).act(dict(wet)).request[0]
    assert seeing < blind


def test_off_farm_worker_checks_on_weekends_not_while_away():
    f = farmer(off_farm_work=1., work_ethic=0.)
    f.act(observation(day=20, weekday=5))            # Saturday: checks
    assert f.checked_today
    f.act(observation(day=21, weekday=0))            # Monday: away
    assert not f.checked_today
    assert f.days_to_next_check(observation(day=21, weekday=0)) == 5  # next Saturday


def test_days_off_vary_slightly_between_farmers_and_weeks():
    farmers = [BaseFarmer(seed=s) for s in range(200)]
    usual = {f.days_off for f in farmers}
    assert (5, 6) in usual and usual <= {(5, 6), (4, 5), (0, 6)} and len(usual) > 1
    share = sum(f.days_off == (5, 6) for f in farmers) / len(farmers)
    assert .55 < share < .85  # most keep Sat+Sun
    f = farmers[0]
    weeks = {f.days_off_in_week(7 * w) for w in range(40)}
    assert f.days_off in weeks and len(weeks) > 1  # an occasional week moves a day off


def test_flow_through_on_hot_days():
    hot = observation(pond=20., et0_mm=6.)
    assert farmer(flow_through_habit=1.).act(dict(hot)).request[0] > farmer(flow_through_habit=0.).act(dict(hot)).request[0]


def test_work_ethic_sets_the_check_interval():
    assert [farmer(work_ethic=w).monitor_interval for w in (1., .75, .5, .25, 0.)] == [1, 2, 3, 4, 5]


def test_earlier_trait_names_still_work_and_labour_folds_into_work_ethic():
    from abm.farmer import RENAMED_TRAITS

    for old, new in RENAMED_TRAITS.items():
        lo, hi = abm.farmer.TRAIT_RANGE[new]
        value = lo + .3 * (hi - lo)
        assert BaseFarmer(**{old: value}).traits[new] == pytest.approx(value)
    # The old interval 1 + 4(1 - labour)(1 - work_ethic) is kept exactly.
    old = BaseFarmer(labour=.5, work_ethic=.5)
    assert old.traits["work_ethic"] == pytest.approx(.75)
    assert old.monitor_interval == 2
    assert "wealth" not in BaseFarmer(wealth=.9, price_sensitivity=.9).traits


def test_removed_traits_do_not_change_the_other_traits_drawn_for_a_seed():
    # Retired traits keep their slot in the seeded draw: a farmer's other traits are the
    # same as when wealth / labour / price_sensitivity existed (values from that version).
    f = make_agent("steward", seed=1)
    assert f.traits["experience"] == pytest.approx(min(.9 + _old_noise(1, 1), 1.), abs=1e-12)
    assert f.traits["off_farm_work"] == pytest.approx(min(max(.2 + _old_noise(1, 11), 0.), 1.), abs=1e-12)


def _old_noise(seed, slot, spread=.08):
    rng = np.random.default_rng(seed)
    draws = [rng.normal(0., spread * (2. if name in ("neighbour_offset", "eagerness") else 1.))
             for name in ("wealth", "experience", "knowledge", "labour", "cooperation", "work_ethic",
                          "risk_tolerance", "neighbour_offset", "eagerness", "drawdown", "overfill",
                          "part_time", "foresight", "flow_through", "price_sensitivity")]
    return draws[slot]


def test_experience_damps_the_reaction_to_a_stressed_day():
    """The learning step: target rises by rate × (1 − ks) × (1 − experience / 2)."""
    rises = {}
    for exp in (0., 1.):
        f = make_agent("steward", seed=3, experience=exp, learning_rate=1., social_influence=0.)
        f.memory.append(dict(day=0, ks=.5, irrigation_mm=0., scarcity=0., score=.5, copied_from=None))
        before = f.target_multiplier
        f.learn(observation())
        rises[exp] = f.target_multiplier - before
    assert rises[0.] == pytest.approx(.5)
    assert rises[1.] == pytest.approx(.25)


def test_weir_board_spills_above_the_crest():
    from stem import water

    ratios = water.Ratios.uniform(2, kz=0.0)
    grid = water.Grid(np.full((2, 4), -1), np.ones(2))
    _, _, flux = water.step_numpy(np.full(2, 100.), np.zeros(2), 0., 0., 0., 0., ratios, grid,
                                  weir=np.array([60., np.inf]))
    assert flux["discharge_mm"][0] == pytest.approx(40.)
    assert flux["discharge_mm"][1] == 0.


def test_world_runs_each_paddy_on_its_owners_calendar():
    from abm.model import DistrictModel
    from abm.world import SeasonConfig

    model = DistrictModel(SeasonConfig(days=3, paddies_per_farmer=200, seed=2))
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

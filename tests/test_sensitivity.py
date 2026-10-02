import numpy as np
import pytest

from abm import sensitivity as sens
from abm.farmer import TRAIT_NAMES

pytestmark = pytest.mark.usefixtures("abm_builtins_only")

TOY = dict(use_cyclic_map=False, days=25, paddies_per_farmer=4)


def test_plan_pairs_every_trait_on_every_seed_without_duplicates():
    plan = sens.build_plan(seeds=[0, 1], weathers=["baseline", "drought"], **TOY)
    ids = [s.run_id for s in plan]
    assert len(ids) == len(set(ids))
    # per seed: 2 weather base runs + 2 levels x every trait, then the same under pump shortage
    assert len(plan) == 2 * (2 + 2 * len(TRAIT_NAMES) + 1 + 2 * len(TRAIT_NAMES))
    assert len(sens.build_plan(seeds=[0], weathers=["baseline"], shortage=False, **TOY)) == 1 + 2 * len(TRAIT_NAMES)
    lows = {(s.seed, s.trait) for s in plan if s.level == "low"}
    highs = {(s.seed, s.trait) for s in plan if s.level == "high"}
    assert lows == highs


def test_pinning_one_trait_leaves_the_rest_of_the_population_identical():
    base = sens._build_model(sens.RunSpec(3, **TOY))
    pinned = sens._build_model(sens.RunSpec(3, "trait", trait="refill_overshoot", level="high", value=1., **TOY))
    assert base.roster_names() == pinned.roster_names()
    assert np.array_equal(base.world.owner_id, pinned.world.owner_id)
    for fid, agent in base.farmer_agents.items():
        other = pinned.farmer_agents[fid].policy
        assert other.traits["refill_overshoot"] == 1.
        assert {k: v for k, v in agent.policy.traits.items() if k != "refill_overshoot"} == \
               {k: v for k, v in other.traits.items() if k != "refill_overshoot"}
        assert agent.policy.days_off == other.days_off


def test_same_spec_reproduces_exactly():
    a, _ = sens.run_one(sens.RunSpec(1, **TOY))
    b, _ = sens.run_one(sens.RunSpec(1, **TOY))
    for key, *_ in sens.METRICS:
        assert a[key] == pytest.approx(b[key], nan_ok=True)


def test_effects_detect_refill_overshoot():
    plan = sens.build_plan(seeds=[0, 1, 2], traits=["refill_overshoot"], weathers=["baseline"],
                           shortage=False, **TOY)
    runs, daily = sens.run_plan(plan, workers=1, progress=None)
    assert set(daily.run_id) == set(runs.run_id)
    eff = sens.trait_effects(runs).set_index(["trait", "metric"])
    assert eff.loc[("refill_overshoot", "request_m3"), "verdict"] == "pass: expected direction"
    noise = sens.noise_table(runs, sens.trait_effects(runs))
    assert set(noise.metric) == {k for k, *_ in sens.METRICS}


def test_pump_shortage_limits_the_station_and_carries_its_own_expectations():
    spec = sens.RunSpec(0, "shortage", pump_fraction=sens.SHORTAGE_PUMP_FRACTION, **TOY)
    assert spec.scenario == "baseline" + sens.SHORTAGE
    full = sens._build_model(sens.RunSpec(0, **TOY)).world.config.pump_capacity_m3_s
    short = sens._build_model(spec).world.config.pump_capacity_m3_s
    assert short == pytest.approx(full * sens.SHORTAGE_PUMP_FRACTION)
    assert sens.expected("experience", "request_m3", spec.scenario) == -1
    assert sens.expected("experience", "request_m3", "baseline") is None
    assert sens.expected("refill_overshoot", "request_m3", spec.scenario) == +1


def test_finished_seasons_are_saved_and_reused(tmp_path):
    plan = sens.build_plan(seeds=[0, 1], traits=["refill_overshoot"], weathers=["baseline"],
                           shortage=False, **TOY)
    first, _ = sens.run_plan(plan, workers=1, progress=None, cache_dir=str(tmp_path))
    assert len(list(tmp_path.glob("*.pkl"))) == len(plan)
    messages = []
    again, _ = sens.run_plan(plan, workers=1, progress=messages.append, cache_dir=str(tmp_path))
    assert "0 to run with 1 worker process" in messages
    assert again["request_m3"].tolist() == first["request_m3"].tolist()


def test_a_change_from_a_zero_base_counts_as_an_effect():
    assert sens._relative_pct(5., 0.) == float("inf")
    assert sens._relative_pct(0., 0.) == 0.
    assert sens._verdict(+1, 5., 1e-4, sens._relative_pct(5., 0.)) == "pass: expected direction"

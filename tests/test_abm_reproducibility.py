"""Replay checks on a synthetic district, without downloads or satellite files."""
from dataclasses import fields

import numpy as np

from abm.landscape import assign_owners_along_canal_variable
from abm.model import DistrictModel
from abm.world import SeasonConfig, StepResult, perturb_weather


def test_ownership_and_weather_replay_with_explicit_seeds():
    order = np.arange(60)
    owners = assign_owners_along_canal_variable(order, 60, 4, .8, 137)
    # Intervening random work must not change explicit-seed results.
    assign_owners_along_canal_variable(order, 60, 4, .8, 999)
    np.testing.assert_array_equal(owners, assign_owners_along_canal_variable(order, 60, 4, .8, 137))
    assert not np.array_equal(owners, assign_owners_along_canal_variable(order, 60, 4, .8, 138))
    rain, et0 = np.linspace(0, 20, 30), np.full(30, 5.)
    original_rain, original_et0 = rain.copy(), et0.copy()
    first = perturb_weather(rain, et0, .3, 137)
    perturb_weather(rain, et0, .3, 999)
    replay = perturb_weather(rain, et0, .3, 137)
    for a, b in zip(first, replay):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(rain, original_rain)
    np.testing.assert_array_equal(et0, original_et0)
    assert not np.array_equal(first[0], perturb_weather(rain, et0, .3, 138)[0])


def test_district_trajectory_replays_across_instances_and_reset(abm_builtins_only):
    def model():
        return DistrictModel(SeasonConfig(use_cyclic_map=False, days=12, seed=37,
                                          paddies_per_farmer=3),
                             agent_roster=['steward', 'opportunist'],
                             station_name='mpc', seed=37)

    def trajectory(district):
        for _ in range(12):
            district.step()
        return list(district.world.history)

    a, b = model(), model()
    reference = trajectory(a)
    replay = trajectory(b)
    a.reset()
    reset_replay = trajectory(a)
    assert len(reference) == len(replay) == len(reset_replay) == 12
    for expected, repeated, reset in zip(reference, replay, reset_replay):
        for field in fields(StepResult):
            value = getattr(expected, field.name)
            for result in (repeated, reset):
                actual = getattr(result, field.name)
                if value is None or isinstance(value, str):
                    assert actual == value, field.name
                else:
                    np.testing.assert_allclose(actual, value, rtol=0, atol=0,
                                               equal_nan=True, err_msg=field.name)
    # Guard against a vacuous replay of an inert model.
    assert any(np.any(day.irrigation_mm > 0) for day in reference)
    assert not np.array_equal(reference[0].ponding_mm, reference[-1].ponding_mm)

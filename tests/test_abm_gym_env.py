import numpy as np
import pytest

from abm.gym_env import AbmEnv
from abm.optimise import MPCStation

pytestmark = pytest.mark.usefixtures("abm_builtins_only")


def run(env, action_fn=None):
    obs, info = env.reset(seed=0)
    total, done = 0.0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(None if action_fn is None else action_fn(env))
        total += reward
        done = terminated or truncated
    return total, info


def test_env_api_with_plugged_in_farmers_and_staff():
    env = AbmEnv(year=2022, days=6, farmers={"steward": 0.5, "opportunist": 0.5}, staff="mpc")
    obs, info = env.reset(seed=0)
    assert obs["ponding_mm"].shape == (env.world.n,)
    assert set(info["farmers"].values()) <= {"steward", "opportunist"}
    assert info["staff"] == "mpc"
    total, info = run(env)
    assert np.isfinite(total)
    assert {"WUE_prime_pct", "RUE_all", "LUE", "W_m3", "O_m3", "stress_mm"} <= set(info["season"])


def test_external_staff_action_flat_array():
    env = AbmEnv(year=2022, days=3, farmers=["steward"])
    n = env.world.n
    action = lambda e: np.concatenate([[1e6, 0.0], np.full(n, 10.0)])
    _, info = run(env, action)
    assert info["staff_state"] == "external"
    assert info["irrigation_m3"] > 0


def test_mpc_keeps_every_paddy_unstressed_with_less_water_than_adaptive():
    mpc = AbmEnv(year=2022, days=30, farmers=["steward"], staff="mpc")
    _, info_mpc = run(mpc)
    assert np.all(mpc.world.stress_cum == 0)
    ada = AbmEnv(year=2022, days=30, farmers=["steward"], staff="adaptive")
    _, info_ada = run(ada)
    assert info_mpc["season"]["IR_m3"] < info_ada["season"]["IR_m3"]


def test_staff_can_be_an_instance():
    env = AbmEnv(year=2022, days=2, farmers=["steward"], staff=MPCStation(margin_mm=5.0))
    run(env)
    assert env.model.station_agent.policy.params["margin_mm"] == 5.0


def test_weather_scenarios_are_tweakable():
    from abm.weather import WeatherScenario, weather_scenario
    from abm.world import SeasonConfig, WaterWorld

    base = WaterWorld(SeasonConfig(year=2022))
    drought = WaterWorld(SeasonConfig(year=2022, weather="drought"))
    assert drought.rain_series.sum() < 0.5 * base.rain_series.sum() + 1e-9
    assert np.all(drought.rain_series[60:85] == 0)
    assert drought.et0_series.sum() == pytest.approx(1.1 * base.et0_series.sum())
    storm = WaterWorld(SeasonConfig(year=2022, weather=WeatherScenario(storms=[(10, 150.0)])))
    assert storm.rain_series[10] == pytest.approx(base.rain_series[10] + 150.0)
    tweaked = weather_scenario("drought", dry_spells=[], rain_scale=1.0)
    assert tweaked.et0_scale == 1.1 and tweaked.dry_spells == []
    env = AbmEnv(year=2022, days=2, farmers=["steward"], weather="heatwave")
    _, info = env.reset(seed=0)
    assert info["weather"] == "heatwave"


def test_default_staff_is_mpc_unless_user_registers_one():
    import abm
    from abm.station import STATIONS, default_station

    assert default_station() == "mpc"
    assert abm.FieldSession(days=1).station_name == "mpc"
    assert AbmEnv(year=2022, days=1, farmers=["steward"]).model.station_name == "mpc"

    @abm.register_station("my_test_staff")
    class MyStaff(abm.StationAgent):
        pass

    try:
        assert default_station() == "my_test_staff"
        assert abm.FieldSession(days=1).station_name == "my_test_staff"
        env = AbmEnv(year=2022, days=1, farmers=["steward"])
        assert env.model.station_name == "my_test_staff"
        assert isinstance(env.model.station_agent.policy, MyStaff)
    finally:
        STATIONS.pop("my_test_staff", None)
    assert default_station() == "mpc"


def test_user_registered_farmer_joins_default_roster():
    import abm
    from abm.agents import AGENTS, DEFAULT_ROSTER, default_roster

    assert default_roster() == list(DEFAULT_ROSTER)

    @abm.register_farmer("my_test_farmer")
    class MyFarmer(abm.FarmerAgent):
        pass

    try:
        assert default_roster() == [*DEFAULT_ROSTER, "my_test_farmer"]
        session = abm.FieldSession(days=1, seed=3)
        assert "my_test_farmer" in session.agent_roster
        assert "my_test_farmer" in session.model.roster_names().values()
    finally:
        AGENTS.pop("my_test_farmer", None)

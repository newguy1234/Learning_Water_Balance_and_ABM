import numpy as np
import pytest
import cyclic
from abm.landscape import assign_owners_along_canal_variable, load_cyclic_landscape
from abm.world import SeasonConfig, WaterWorld

@pytest.mark.parametrize('year', [2021, 2022])
def test_bundled_parcels_match_package(year):
    ds = cyclic.load_paddy_dataset(year)
    frame = cyclic.data.paddies()
    np.testing.assert_array_equal(ds.paddy, frame.paddy_id)
    np.testing.assert_array_equal(ds.cyclic.astype(bool), frame.cyclic)
    assert 'ndvi' in ds
    assert 'pool_weights' in ds
    land = load_cyclic_landscape(year, days=2)
    np.testing.assert_array_equal(land.cyclic, ds.cyclic)
    np.testing.assert_allclose(land.ground_water_law.pool_weights, ds.pool_weights)
    assert len(land.rain_mm) == 2

@pytest.mark.parametrize('variability', [0, 0.8])
def test_scattered_ownership(variability):
    order = np.arange(1000)
    owners = assign_owners_along_canal_variable(order, 1000, 50, variability, 42)
    np.testing.assert_array_equal(owners, assign_owners_along_canal_variable(order, 1000, 50, variability, 42))
    assert not np.array_equal(owners, assign_owners_along_canal_variable(order, 1000, 50, variability, 43))
    assert np.all(np.bincount(owners) > 0)
    assert all(np.ptp(np.flatnonzero(owners == k)) > 500 for k in np.unique(owners)[:-1])


def test_step_conserves_water_and_noncyclic_drainage():
    world = WaterWorld(SeasonConfig(days=2))
    volume = lambda mm: float(np.dot(mm, world.area_m2) / 1000)
    stores = lambda: (world.pool_m3 + world.lake_m3 + world.soil_storage_m3()
                      + world.dwb_m3 + world.deep_m3 + world.drain_transit_m3)
    before = stores()
    sent = np.full(world.n, 5.0)
    world.apply_station({"withdraw_m3": 1e7, "irrigation_sent_mm": sent})
    result = world.step(np.full(world.n, 3.0), np.full(world.n, 2.0))
    assert np.all(result.to_pool_mm[~world.cyclic] == 0)
    assert np.all(result.stay_mm[~world.cyclic] == 0)
    # Recaptured lateral flow + block storage is exactly the cyclic lateral flow.
    np.testing.assert_allclose((result.to_pool_mm + result.stay_mm)[world.cyclic],
                               result.lateral_mm[world.cyclic])
    # Only rain and ET cross the district boundary (the lake is inside the tally).
    assert stores() == pytest.approx(before + volume(np.full(world.n, result.rain_mm) - result.eta_mm), abs=1e-3)


def test_seepage_is_depth_driven_not_surplus_driven():
    """A paddy held at target (zero daily surplus) still percolates and feeds lateral
    flow — Okura's Lf is a continuous loss, not a share of the day's leftover."""
    def infiltration(depth):
        world = WaterWorld(SeasonConfig(days=1))
        world.ponding_mm = np.full(world.n, depth)
        return world.step(np.zeros(world.n), np.zeros(world.n)).infiltration_mm

    deep, shallow = infiltration(50.0), infiltration(10.0)
    assert np.all(deep > 0)
    assert shallow.mean() < deep.mean()


def test_pool_is_capped_and_spills_to_lake():
    world = WaterWorld(SeasonConfig(days=2))
    assert world.pool_m3 <= world.pool_capacity_m3
    world.apply_station({"withdraw_m3": 0.0, "irrigation_sent_mm": np.zeros(world.n)})
    result = world.step(np.zeros(world.n), np.full(world.n, 40.0))
    assert world.pool_m3 <= world.pool_capacity_m3 + 1e-6
    assert result.pool_spill_m3 > 0


def test_rain_matches_site_gauge():
    from cyclic import okura
    world = WaterWorld(SeasonConfig(year=2021))
    assert world.rain_series.sum() == pytest.approx(sum(okura.TABLE2_MM[2021]["rain"].values()), rel=1e-6)


def test_stress_is_unmet_weather_et():
    """Demand 8 mm, only 2 mm of pond + topsoil water -> 6 mm of stress. The root
    zone below is full but rice cannot use it."""
    from cyclic.physics import Physics, State, supply_et_demand
    p = Physics()
    state = State.start(1, depth_mm=1.0, perched_mm=0.0)
    state.surface[:] = p.total_evaporable_mm - 1.0  # 1 mm left in the topsoil
    taken = supply_et_demand(state, 8.0, p)
    assert taken["et_mm"][0] == pytest.approx(2.0)
    assert taken["stress_mm"][0] == pytest.approx(6.0)
    assert state.water[0] == pytest.approx(0.0)
    assert state.deficit[0] == pytest.approx(0.0)


def test_ample_water_means_no_stress():
    """If every paddy is offered more water than it could use, no farmer type is ever
    short: stress only comes from supply limits, never from the physics itself."""
    import abm
    from abm.model import DistrictModel

    class Unlimited(abm.StationAgent):
        def decide(self, obs):
            n = int(obs["n_paddies"])
            return {"withdraw_m3": 1e12, "outflow_m3": 0.0,
                    "irrigation_sent_mm": np.full(n, 200.0), "state": "irrigation"}

    # Unlimited supply: also lift the pumps' real capacity (SeasonConfig.pump_capacity_m3_s).
    model = DistrictModel(SeasonConfig(year=2022, days=40, pump_capacity_m3_s=None),
                          agent_roster=list(abm.agents.DEFAULT_ROSTER))
    model.station_agent.policy = Unlimited()
    while not model.done():
        model.step()
    assert np.all(model.world.stress_cum == 0)
    assert np.all(model.world.relative_yield == pytest.approx(1.0))

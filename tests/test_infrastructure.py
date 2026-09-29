import numpy as np
import pytest

from cyclic import data, infrastructure, okura
from cyclic.canal import distance_to_canal_m


def metres_between(a, b):
    x, y = infrastructure.metres([b[0]], [b[1]], origin=a)
    return float(np.hypot(x[0], y[0]))


def test_district_matches_okura_table1():
    frame = data.paddies()
    raw = frame.groupby(["block", "cyclic"]).area_raw_m2.sum() / 1e4
    for block, sizes in okura.TABLE1_HA.items():
        assert raw[(block, True)] == pytest.approx(sizes["cyclic"], rel=0.05)
        assert raw[(block, False)] == pytest.approx(sizes["non_cyclic"], rel=0.05)
        # Areas are then scaled to Table 1 exactly.
        scaled = frame[(frame.block == block)].groupby("cyclic").area_m2.sum() / 1e4
        assert scaled[True] == pytest.approx(sizes["cyclic"], rel=1e-6)
        assert scaled[False] == pytest.approx(sizes["non_cyclic"], rel=1e-6)


def test_pool_is_at_the_station_on_the_main_canal():
    assert data.POOL_LONLAT == infrastructure.STATION_LONLAT
    station = infrastructure.STATION_LONLAT
    assert distance_to_canal_m(station[0], station[1]) < 60.0
    canal = infrastructure.main_canal_vertices()
    assert canal[0][1] > station[1] > canal[-1][1]   # arms north and south of the station


def test_cyclic_paddies_drain_to_the_pool_through_the_network():
    frame = data.paddies()
    cyclic = frame.cyclic.to_numpy()
    path = frame.drain_path_m.to_numpy()
    assert np.all(np.isfinite(path))
    # Cyclic paddies sit on the network; non-cyclic ones are far from it (they drain out).
    assert np.median(frame.drain_entry_m[cyclic]) < 400
    assert np.median(frame.drain_entry_m[~cyclic]) > 1000
    # A point at the station has (almost) no route left.
    route = infrastructure.drainage_routes([infrastructure.STATION_LONLAT[0]], [infrastructure.STATION_LONLAT[1]])
    assert route["total_m"][0] < 50


def test_every_channel_has_a_role_and_sources_are_cited():
    roles = {f["properties"]["role"] for f in infrastructure.network()["features"]}
    assert {"main_canal", "connected_drain", "other_drain", "station"} <= roles
    assert "OpenStreetMap" in infrastructure.network()["attribution"]
    import os
    text = open(os.path.join(infrastructure.BUNDLED, "SOURCES.md"), encoding="utf-8").read()
    assert "10.1016/j.ejrh.2024.102111" in text and "ODbL" in text


def test_recapture_follows_distance_to_the_pool_network():
    frame = data.paddies()
    main_only = distance_to_canal_m(frame.lon.to_numpy(), frame.lat.to_numpy(),
                                    vertices=infrastructure.main_canal_vertices())
    network = frame.canal_distance_m.to_numpy()
    assert np.all(network <= main_only + 0.1)   # the main canal is part of the network (projection jitter < 0.1 m)
    from cyclic.paddy_export import _build_landscape
    land = _build_landscape(2022)
    c = land.cyclic
    share = land.shares["pool_capture_share"][c]
    near, far = network[c] < 100, network[c] > 500
    assert share[near].mean() > share[far].mean()
    assert np.average(share, weights=land.area_m2[c]) == pytest.approx(land.physics.capture, abs=0.02)


def test_canal_depth_and_elevation_are_bundled():
    d21, d22 = infrastructure.canal_depth(2021), infrastructure.canal_depth(2022)
    assert len(d21["North"]) == len(d22["South"]) == 131
    # Okura §4.1: the North arm ran high all of wet 2021; 2022 sat at its normal level.
    assert d21["North"].mean() - d22["North"].mean() > 0.1
    assert infrastructure.normal_canal_depth_m() == pytest.approx(0.80, abs=0.03)
    assert infrastructure.canal_depth(2019) is None
    frame = data.paddies()
    h = frame.ground_elev_m
    assert h.notna().all()
    # The South cyclic block stands higher than the North (Okura §4.1).
    assert h[frame.cyclic & (frame.block == "South")].median() > h[frame.cyclic & (frame.block == "North")].median()


def test_backwater_holds_drainage_on_low_cyclic_fields():
    from abm.world import SeasonConfig, WaterWorld

    high = WaterWorld(SeasonConfig(days=2, canal_depth=1.3))
    normal = WaterWorld(SeasonConfig(days=2, canal_depth=None))
    for world in (high, normal):
        world.ponding_mm = np.full(world.n, 80.0)
        world.apply_station({"withdraw_m3": 0.0, "irrigation_sent_mm": np.zeros(world.n)})
    before = high.soil_storage_m3() + high.pool_m3 + high.lake_m3 + high.dwb_m3 + high.deep_m3 + high.drain_transit_m3
    a = high.step(np.zeros(high.n), np.full(high.n, 30.0))
    b = normal.step(np.zeros(normal.n), np.full(normal.n, 30.0))
    frac = high.last_backwater_frac
    assert np.all(frac[~high.cyclic] == 0)
    low = high.cyclic & (high.ground_elev_m < np.percentile(high.ground_elev_m[high.cyclic], 20))
    assert frac[low].mean() > 0.5
    assert a.drainage_mm[high.cyclic].sum() < b.drainage_mm[normal.cyclic].sum()
    after = high.soil_storage_m3() + high.pool_m3 + high.lake_m3 + high.dwb_m3 + high.deep_m3 + high.drain_transit_m3
    vol = lambda mm: float(np.dot(mm, high.area_m2) / 1000)
    assert after == pytest.approx(before + vol(np.full(high.n, a.rain_mm) - a.eta_mm), abs=1e-3)
    # At the canal's normal depth nothing is held back on 98 % of cyclic land.
    calm = WaterWorld(SeasonConfig(days=1, canal_depth=infrastructure.normal_canal_depth_m()))
    assert np.average(calm.backwater_fraction(0)[calm.cyclic] > 0, weights=calm.area_m2[calm.cyclic]) <= 0.03


def test_pump_capacity_limits_delivery():
    from abm.world import SeasonConfig, WaterWorld

    world = WaterWorld(SeasonConfig(days=1, pump_capacity_m3_s=1.0))
    sent = world.apply_station({"withdraw_m3": 1e9, "irrigation_sent_mm": np.full(world.n, 100.0)})
    delivered = float(np.dot(sent, world.area_m2) / 1000)
    assert delivered == pytest.approx(86_400.0, rel=1e-6)
    # Nearest paddies (to the pool) are served first.
    near = np.argsort(world.pool_distance_m)[:10]
    assert np.all(sent[near] == 100.0)


def test_abm_routes_drainage_to_the_pool_and_conserves_it():
    from abm.world import SeasonConfig, WaterWorld

    slow = WaterWorld(SeasonConfig(days=3, drain_velocity_m_s=0.02))
    instant = WaterWorld(SeasonConfig(days=3, drain_velocity_m_s=None))
    assert np.allclose(slow.station_lonlat, infrastructure.STATION_LONLAT)
    released = arrived = 0.0
    for world in (slow, instant):
        for _ in range(3):
            world.apply_station({"withdraw_m3": 1e7, "irrigation_sent_mm": np.zeros(world.n)})
            step = world.step(np.zeros(world.n), np.full(world.n, 5.0))
            if world is slow:
                released += float(np.dot(step.drainage_mm[world.cyclic], world.area_m2[world.cyclic]) / 1000)
                arrived += step.return_d_m3
    # Slow drains hold water in transit; what was released = what arrived + what is on its way.
    assert slow.drain_transit_m3 > 0
    assert released == pytest.approx(arrived + slow.drain_transit_m3, rel=1e-9)
    assert instant.drain_transit_m3 == 0
    assert np.all(slow.drain_travel_days[slow.cyclic] > 0)


def test_map_key_names_every_drawn_feature():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([140.25, 140.28], [35.78, 35.83])
    handles = infrastructure.draw(ax, key=True)
    labels = {h.get_label() for h in handles}
    for role in ("main_canal", "connected_drain", "culvert", "other_drain", "river", "station",
                 "irrigation_pipeline", "sluice_gate"):
        assert infrastructure.KEY[role][0] in labels
    assert ax.get_xlim()[1] < 140.30   # still framed on what was drawn first
    plt.close(fig)
    notes = infrastructure.network()
    assert notes["irrigation_pipelines"]["hakusan_main_m"] == 940
    assert "pile" in notes["lateral_collection"]["pool"]

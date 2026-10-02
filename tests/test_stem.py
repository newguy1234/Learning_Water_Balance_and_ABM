import os

import numpy as np
import pytest

from stem import infrastructure, okura, water
from stem.district import load_district
from stem.field import Field, FieldConfig


@pytest.fixture(scope="module")
def district():
    return load_district(2022)


def test_district_matches_okura_table1(district):
    for block, sizes in okura.TABLE1_HA.items():
        for cyc, key in ((True, "cyclic"), (False, "non_cyclic")):
            area = district.area_m2[(district.block == block) & (district.cyclic == cyc)].sum() / 1e4
            assert area == pytest.approx(sizes[key], rel=1e-6)


def test_cyclic_paddies_drain_to_the_pool_through_the_network(district):
    c = district.cyclic
    assert np.all(np.isfinite(district.drain_path_m))
    assert np.median(district.drain_entry_m[c]) < 400
    assert np.median(district.drain_entry_m[~c]) > 1000


def test_sources_are_cited():
    roles = {f["properties"]["role"] for f in infrastructure.network()["features"]}
    assert {"main_canal", "connected_drain", "other_drain", "station"} <= roles
    text = open(os.path.join(infrastructure.BUNDLED, "SOURCES.md"), encoding="utf-8").read()
    assert "10.1016/j.ejrh.2024.102111" in text and "ODbL" in text


def test_canal_depth_and_elevation(district):
    d21, d22 = infrastructure.canal_depth(2021), infrastructure.canal_depth(2022)
    assert len(d21["North"]) == len(d22["South"]) == 131
    assert d21["North"].mean() - d22["North"].mean() > 0.1      # Okura §4.1: wet 2021 ran high
    h, c = district.elevation_m, district.cyclic
    assert np.median(h[c & (district.block == "South")]) > np.median(h[c & (district.block == "North")])


def test_gauge_rain_matches_table2(district):
    assert district.rain_mm.sum() == pytest.approx(sum(okura.TABLE2_MM[2022]["rain"].values()), rel=1e-6)


def test_water_balance_closes_and_neighbour_flow_conserves_volume(district):
    ratios = water.Ratios.random(district.n, seed=3)
    grid = water.Grid(district.neighbours, district.area_m2)
    pond, soil = np.full(district.n, 30.0), np.full(district.n, 60.0)
    for _ in range(5):
        p0, s0 = pond.copy(), soil.copy()
        pond, soil, f = water.step_numpy(pond, soil, 2.0, 8.0, 3.0, 4.5, ratios, grid)
        res = (pond + soil - p0 - s0) - (2.0 + 8.0 - f["et_mm"] - f["drainage_mm"] - f["deep_mm"]
                                         - f["lateral_mm"] - f["to_x_mm"] - f["to_y_mm"] + f["received_mm"])
        assert np.abs(res).max() < 1e-9
        sent = ((f["to_x_mm"] + f["to_y_mm"]) * district.area_m2).sum()
        assert (f["received_mm"] * district.area_m2).sum() == pytest.approx(sent, rel=1e-9)


def test_field_is_a_mesa_model_of_the_paddies():
    ratios = water.Ratios.random(4002, seed=1)
    field = Field(FieldConfig(year=2022, days=2, ratios=ratios))
    assert len(field.agents_by_type[type(field.paddies[0])]) == field.n
    paddy = field.paddies[10]
    assert paddy.kz == pytest.approx(ratios.kz[10])
    assert sum(paddy.shares.values()) == pytest.approx(1.0)
    paddy.pond = 12.0
    assert field.pond[10] == 12.0
    assert all(n.index in field.neighbours[10] for n in paddy.neighbours)


def test_backwater_holds_drainage_on_low_cyclic_fields():
    high = Field(FieldConfig(days=2, canal_depth=1.3))
    normal = Field(FieldConfig(days=2, canal_depth=None))
    for f in (high, normal):
        f.pond = np.full(f.n, 80.0)
        f.apply_station({"withdraw_m3": 0.0, "irrigation_sent_mm": np.zeros(f.n)})
    a = high.day_step(np.zeros(high.n), np.full(high.n, 30.0))
    b = normal.day_step(np.zeros(normal.n), np.full(normal.n, 30.0))
    frac = high.last_backwater_frac
    assert np.all(frac[~high.cyclic] == 0)
    low = high.cyclic & (high.ground_elev_m < np.percentile(high.ground_elev_m[high.cyclic], 20))
    assert frac[low].mean() > 0.5
    assert a.drainage_mm[high.cyclic].sum() < b.drainage_mm[normal.cyclic].sum()


def test_pump_capacity_serves_the_nearest_paddies_first():
    field = Field(FieldConfig(days=1, pump_capacity_m3_s=1.0))
    sent = field.apply_station({"withdraw_m3": 1e9, "irrigation_sent_mm": np.full(field.n, 100.0)})
    assert float(np.dot(sent, field.area_m2) / 1000) == pytest.approx(86_400.0, rel=1e-6)
    assert np.all(sent[np.argsort(field.pool_distance_m)[:10]] == 100.0)


def test_drainage_in_transit_is_conserved():
    slow = Field(FieldConfig(days=3, drain_velocity_m_s=0.02))
    released = arrived = 0.0
    for _ in range(3):
        slow.apply_station({"withdraw_m3": 1e7, "irrigation_sent_mm": np.zeros(slow.n)})
        step = slow.day_step(np.zeros(slow.n), np.full(slow.n, 5.0))
        released += float(np.dot(step.drainage_mm[slow.cyclic], slow.area_m2[slow.cyclic]) / 1000)
        arrived += step.return_d_m3
    assert slow.drain_transit_m3 > 0
    assert released == pytest.approx(arrived + slow.drain_transit_m3, rel=1e-9)

import os
import tempfile
import unittest

import numpy as np

from cyclic.account import force_et, solve_omalley
from cyclic.lateral import exchange_perched, knn_edges
from cyclic.physics import Physics, State, advance
from cyclic.model import Results


class AnisotropyTests(unittest.TestCase):
    def test_dry_et_comes_from_soil_not_pond(self):
        state = State.start(2, depth_mm=0.0, perched_mm=0.0)
        state.surface[:] = 0.0
        state.deficit[:] = 0.0
        flux = advance(state, 0.0, 5.0, np.array([0.2, 0.8]), np.zeros(2), np.zeros(2), Physics())
        self.assertTrue(np.all(flux["et_pond_mm"] == 0))
        self.assertTrue(np.all(flux["et_surface_mm"] + flux["et_root_mm"] > 0))
        self.assertTrue(flux["et_root_mm"][1] > flux["et_root_mm"][0])

    def test_flooded_e_from_pond_t_from_root(self):
        state = State.start(1, depth_mm=40.0)
        flux = advance(state, 0.0, 5.0, 0.5, 0.0, 0.0, Physics())
        self.assertGreater(float(flux["et_pond_mm"][0]), 0)
        self.assertGreater(float(flux["et_root_mm"][0]), 0)
        self.assertEqual(float(flux["et_surface_mm"][0]), 0)
        self.assertAlmostEqual(float(flux["et_pond_mm"][0]), float(flux["evaporation_mm"][0]), places=6)

    def test_neighbour_exchange_conserves_volume_and_prefers_kx(self):
        perched = np.array([30.0, 0.0, 0.0])
        lon = np.array([140.0, 140.0002, 140.0])
        lat = np.array([35.7, 35.7, 35.702])
        edges = knn_edges(lon, lat, k=2)
        area = np.full(3, 1000.0)
        east = exchange_perched(perched, edges, Physics(kx_mm_d=8000, ky_mm_d=200),
                                area_m2=area)
        north = exchange_perched(perched, edges, Physics(kx_mm_d=200, ky_mm_d=8000),
                                 area_m2=area)
        self.assertAlmostEqual(float(east @ area), 0.0, places=6)
        self.assertGreater(abs(east[1]), abs(east[2]))
        self.assertGreater(abs(north[2]), abs(north[1]))

    def test_vertical_k_changes_infiltration(self):
        wet = State.start(1, depth_mm=40.0)
        dry_k = Physics(plow_pan_k_mm_d=2.0)
        wet_k = Physics(plow_pan_k_mm_d=20.0)
        a = advance(State.start(1, depth_mm=40.0), 0.0, 0.0, 0.5, 0.0, 0.0, dry_k)
        b = advance(wet, 0.0, 0.0, 0.5, 0.0, 0.0, wet_k)
        self.assertGreater(float(b["infiltration_mm"][0]), float(a["infiltration_mm"][0]))

    def test_infiltration_scales_with_pond_not_topsoil_depletion(self):
        p = Physics()
        wet_soil = State.start(1, depth_mm=30.0)
        dry_soil = State.start(1, depth_mm=30.0)
        dry_soil.surface[:] = p.total_evaporable_mm
        same_pond = advance(wet_soil, 0.0, 0.0, 0.5, 0.0, 0.0, p)
        dry_top = advance(dry_soil, 0.0, 0.0, 0.5, 0.0, 0.0, p)
        self.assertAlmostEqual(float(same_pond["infiltration_mm"][0]),
                               float(dry_top["infiltration_mm"][0]))
        shallow = advance(State.start(1, depth_mm=15.0), 0.0, 0.0, 0.5, 0.0, 0.0, p)
        deep = advance(State.start(1, depth_mm=30.0), 0.0, 0.0, 0.5, 0.0, 0.0, p)
        self.assertAlmostEqual(float(deep["infiltration_mm"][0]) / 30.0,
                               float(shallow["infiltration_mm"][0]) / 15.0, places=5)

    def test_need_stock_when_root_and_surface_are_empty(self):
        p = Physics()
        state = State.start(1, depth_mm=0.0, perched_mm=0.0)
        state.surface[:] = p.total_evaporable_mm
        state.deficit[:] = p.total_available_mm
        flux = force_et(state, 0.0, 8.0, 0.0, 0.0, p)
        self.assertAlmostEqual(float(flux["need_mm"][0]), 8.0)
        self.assertAlmostEqual(float(flux["et_surface_mm"][0]), 0.0)
        self.assertAlmostEqual(float(flux["et_root_mm"][0]), 0.0)
        self.assertAlmostEqual(float(flux["et_mm"][0]), 8.0)

    def test_dry_surface_still_takes_t_from_root_zone(self):
        p = Physics()
        state = State.start(1, depth_mm=0.0, perched_mm=0.0)
        state.surface[:] = p.total_evaporable_mm
        state.deficit[:] = 0.0
        flux = force_et(state, 0.0, 8.0, 0.0, 0.0, p)
        self.assertAlmostEqual(float(flux["need_mm"][0]), 0.0)
        self.assertAlmostEqual(float(flux["et_surface_mm"][0]), 0.0)
        self.assertAlmostEqual(float(flux["et_root_mm"][0]), 8.0)

    def test_need_is_zero_when_surface_and_root_cover_et(self):
        state = State.start(1, depth_mm=0.0, perched_mm=0.0)
        state.surface[:] = 0.0
        state.deficit[:] = 0.0
        flux = force_et(state, 0.0, 8.0, 0.0, 0.0, Physics())
        self.assertAlmostEqual(float(flux["need_mm"][0]), 0.0)
        self.assertAlmostEqual(float(flux["et_surface_mm"][0] + flux["et_root_mm"][0]), 8.0)
        self.assertGreater(float(flux["et_root_mm"][0]), 0.0)

    def test_higher_farm_drains_increase_lateral(self):
        wet = State.start(1, depth_mm=40.0, perched_mm=40.0)
        low = Physics(farm_drain_stage_mm=50.0, farm_drain_ref_mm=200.0)
        high = Physics(farm_drain_stage_mm=400.0, farm_drain_ref_mm=200.0)
        a = advance(State.start(1, depth_mm=40.0, perched_mm=40.0), 0.0, 0.0, 0.5, 0.0, 0.0, low)
        b = advance(wet, 0.0, 0.0, 0.5, 0.0, 0.0, high)
        self.assertGreater(float(b["lateral_mm"][0]), float(a["lateral_mm"][0]))

    def test_leftover_moves_toward_the_canal(self):
        from cyclic.canal import apply_canal, distance_to_canal_m
        leftover = np.array([20.0, 0.0])
        lon = np.array([140.267, 140.267])
        # Point 0 is ~170 m from the nearest drain reaching the pool, point 1 ~40 m.
        lat = np.array([35.807, 35.82])
        dist = distance_to_canal_m(lon, lat)
        self.assertGreater(float(dist[0]), float(dist[1]))
        edges = knn_edges(lon, lat, k=1)
        flux = {"leftover_mm": leftover, "neighbor_mm": np.zeros(2)}
        apply_canal(flux, dist, edges, np.full(2, 1000.0),
                    Physics(canal_length_m=50.0, canal_max_frac=0.8, canal_capture_frac=0.2,
                            canal_hops=4, farm_drain_stage_mm=200.0, farm_drain_ref_mm=200.0))
        self.assertLess(float(flux["neighbor_mm"][0]), 0.0)
        self.assertGreater(float(flux["neighbor_mm"][1]), 0.0)
        leftover_split = flux["lateral_mm"] + flux["stay_mm"]
        self.assertAlmostEqual(float(leftover_split.sum()), float(leftover.sum()), places=6)
        self.assertTrue(np.allclose(flux["canal_mm"], 0.0))

    def test_toward_main_drain_is_discharge_to_the_pool(self):
        from cyclic.canal import apply_canal
        dist = np.array([0.0, 2000.0])
        p = Physics(farm_drain_stage_mm=200.0, farm_drain_ref_mm=200.0,
                    canal_length_m=1000.0, canal_capture_frac=0.2)
        stock = np.ones(2)
        leftover = np.array([20.0, 20.0])
        flux = {"drainage_mm": np.array([10.0, 10.0]), "leftover_mm": leftover.copy()}
        apply_canal(flux, dist, None, np.full(2, 1000.0), p, drain_stock=stock)
        self.assertTrue(np.allclose(flux["canal_mm"], [10.0, 10.0]))
        self.assertTrue(np.allclose(flux["canal_out_mm"], [10.0, 10.0]))
        self.assertTrue(np.allclose(stock, 0.0))
        self.assertTrue(np.allclose(flux["lateral_mm"] + flux["stay_mm"], leftover))

    def test_farm_to_main_and_lateral_per_discharge_ratios(self):
        from cyclic.canal import apply_canal
        p = Physics(farm_to_main=0.5, lateral_per_discharge=1.0, canal_capture_frac=0.0)
        leftover = np.array([20.0, 20.0])
        flux = {"drainage_mm": np.array([10.0, 10.0]), "leftover_mm": leftover.copy()}
        apply_canal(flux, np.array([0.0, 0.0]), None, np.full(2, 1000.0), p,
                    cyclic=np.array([True, False]))
        self.assertTrue(np.allclose(flux["canal_mm"], [10.0, 10.0]))
        self.assertTrue(np.allclose(flux["canal_out_mm"], [5.0, 0.0]))
        self.assertAlmostEqual(float(flux["lateral_from_d_mm"][0]), 10.0)
        self.assertAlmostEqual(float(flux["lateral_from_d_mm"][1]), 0.0)
        self.assertAlmostEqual(float((flux["lateral_mm"] + flux["stay_mm"]).sum()), 40.0)

    def test_non_cyclic_drainage_stock_is_separate_from_the_pool(self):
        from cyclic.agents import PumpingStation
        station = PumpingStation.__new__(PumpingStation)
        station.capacity = 5_000.0
        station.storage = 5_000.0
        station.allowance = None
        station.withdrawn = 0.0
        station.expected_inflow = 0.0
        station.ncb_storage = 0.0
        station.settle(IR_North_all_m3=60.0, IR_South_all_m3=40.0,
                       D_North_m3=25.0, D_South_m3=15.0, LFV_pump_m3=10.0,
                       D_ncb_m3=25.0, IR_ncb_m3=30.0)
        self.assertEqual(station.today["D_pool_m3"], 40.0)
        self.assertEqual(station.today["D_North_m3"], 25.0)
        self.assertEqual(station.today["D_South_m3"], 15.0)
        self.assertEqual(station.today["IR_all_m3"], 100.0)
        self.assertEqual(station.today["LFV_pump_m3"], 10.0)
        self.assertEqual(station.today["D_ncb_m3"], 25.0)
        self.assertEqual(station.today["ncb_stock_m3"], 25.0)
        station.settle(IR_North_all_m3=60.0, IR_South_all_m3=40.0,
                       D_North_m3=25.0, D_South_m3=15.0, LFV_pump_m3=10.0,
                       D_ncb_m3=15.0)
        self.assertEqual(station.today["ncb_stock_m3"], 40.0)
        self.assertNotEqual(station.today["pool_m3"], station.today["ncb_stock_m3"])
        # Eq. 4: Δpool = W − O + LFV + D_N + D_S − IR_all
        d = station.today
        self.assertAlmostEqual(
            d["d_pool_m3"],
            d["W_m3"] - d["O_m3"] + d["LFV_pump_m3"] + d["D_pool_m3"] - d["IR_all_m3"],
            places=6)

    def test_non_cyclic_is_land_away_from_the_canal(self):
        from cyclic.data import assign_cyclic_layout
        block = np.array(["North", "North", "North"])
        area = np.array([1.5e6, 1.5e6, 3.0e6])
        canal = np.array([20.0, 80.0, 4000.0])
        cyclic, scaled = assign_cyclic_layout(block, area, canal)
        self.assertTrue(cyclic[0] and cyclic[1])
        self.assertFalse(cyclic[2])
        self.assertAlmostEqual(scaled[cyclic].sum() / 1e4, 260.4, places=4)
        self.assertAlmostEqual(scaled[~cyclic].sum() / 1e4, 307.3, places=4)

    def test_ncb_leftover_does_not_cross_into_cyclic_canal(self):
        from cyclic.canal import apply_canal
        leftover = np.array([20.0, 0.0])
        dist = np.array([400.0, 10.0])
        edges = knn_edges(np.array([140.267, 140.267]), np.array([35.82, 35.807]), k=1)
        flux = {"leftover_mm": leftover, "neighbor_mm": np.zeros(2), "drainage_mm": np.zeros(2)}
        apply_canal(flux, dist, edges, np.full(2, 1000.0),
                    Physics(canal_length_m=50.0, canal_max_frac=0.8, canal_capture_frac=0.5),
                    leftover_split=True, cyclic=np.array([False, True]))
        self.assertEqual(float(flux["neighbor_mm"][0]), 0.0)
        self.assertEqual(float(flux["lateral_mm"][0]), 0.0)
        self.assertAlmostEqual(float(flux["stay_mm"][0]), 20.0)

    def test_higher_farm_drain_stage_captures_more_toward_canal(self):
        from cyclic.canal import apply_canal, farm_drain_factor
        leftover = np.array([30.0, 30.0])
        dist = np.array([80.0, 80.0])
        edges = knn_edges(np.array([140.0, 140.001]), np.array([35.8, 35.8]), k=1)
        area = np.full(2, 1000.0)
        low = Physics(farm_drain_stage_mm=50.0, farm_drain_ref_mm=200.0, canal_routing=True)
        high = Physics(farm_drain_stage_mm=400.0, farm_drain_ref_mm=200.0, canal_routing=True)
        drainage = np.array([12.0, 12.0])
        a = apply_canal({"leftover_mm": leftover.copy(), "drainage_mm": drainage.copy()},
                        dist, edges, area, low)
        b = apply_canal({"leftover_mm": leftover.copy(), "drainage_mm": drainage.copy()},
                        dist, edges, area, high)
        self.assertTrue(np.allclose(a["canal_mm"], drainage))
        self.assertTrue(np.allclose(b["canal_mm"], drainage))
        self.assertTrue(np.allclose(a["canal_out_mm"], drainage))
        self.assertGreater(float(b["lateral_mm"].sum()), float(a["lateral_mm"].sum()))
        self.assertGreater(farm_drain_factor(high), farm_drain_factor(low))

    def test_satellite_et_must_occur_from_groundwater_if_needed(self):
        p = Physics()
        state = State.start(1, depth_mm=0.0, perched_mm=0.0)
        state.surface[:] = p.total_evaporable_mm
        state.deficit[:] = p.total_available_mm
        flux = force_et(state, 0.0, 8.0, 0.0, 0.0, p)
        self.assertAlmostEqual(float(flux["et_mm"][0]), 8.0)
        self.assertAlmostEqual(float(flux["et_groundwater_mm"][0]), 8.0)
        self.assertAlmostEqual(float(flux["et_pond_mm"][0]), 0.0)
        self.assertAlmostEqual(float(flux["et_root_mm"][0]), 0.0)

    def test_leftover_is_eq3_residual(self):
        state = State.start(1, depth_mm=20.0)
        flux = force_et(state, 5.0, 4.0, 10.0, 3.0, Physics())
        self.assertAlmostEqual(float(flux["leftover_mm"][0]), 5 + 10 - float(flux["drainage_mm"][0]) - 4)

    def test_omalley_split_assigns_lateral_or_stay(self):
        n = 6
        stock = np.array([50., 40., 30., 20., 10., 5.])
        area = np.full(n, 1e4)
        cyclic = np.array([True, True, True, True, False, False])
        block = np.array(["North", "North", "South", "South", "North", "South"])
        edges = knn_edges(np.linspace(140, 140.01, n), np.full(n, 35.7), k=2)
        solved = solve_omalley(stock, area, cyclic, block, edges, 0.74, 2022, sweeps=2)
        self.assertEqual(solved.x.size, n)
        self.assertTrue(set(solved.x).issubset({0, 1}))
        self.assertEqual(int(solved.x[4] + solved.x[5]), 0)

    def test_satellite_drains_dry_points(self):
        from cyclic.agents import SatellitePhysics
        policy = SatellitePhysics(pond_mm=30.0)
        policy.rows = np.array([0, 1])
        policy.flood = np.array([[1.0, 0.0]])
        batch = type("B", (), {"day": 0, "depth_mm": np.array([10.0, 25.0])})()
        irrigation, discharge = policy.decide_batch(batch)
        self.assertGreater(irrigation[0], 0)
        self.assertEqual(irrigation[1], 0)
        self.assertEqual(discharge[0], 0)
        self.assertGreater(discharge[1], 0)


class UndergroundCsvTests(unittest.TestCase):
    def test_csv_from_short_log(self):
        from types import SimpleNamespace
        import pandas as pd
        n, days = 3, 2
        dates = pd.date_range("2022-04-17", periods=days)
        log = {k: np.ones((days, n), dtype=np.float32) for k in
               ("irrigation_mm", "farmer_discharge_mm", "drainage_mm", "infiltration_mm",
                "neighbor_mm", "lateral_mm", "deep_mm", "et_mm", "et_pond_mm",
                "et_surface_mm", "et_root_mm", "ponded")}
        model = SimpleNamespace(
            n=n, physics=Physics(), cell_log={},
            area=np.array([1000., 2000., 1500.]),
            inputs=SimpleNamespace(
                dates=dates,
                paddy_id=np.array(["N00000", "N00001", "S00002"]),
                block=np.array(["North", "North", "South"]),
                cyclic=np.array([True, True, False]),
                rain_mm=np.ones(days),
                et_target_mm=np.ones((days, n)),
                is_pass=np.zeros((days, n), dtype=bool),
                flood_daily=np.ones((days, n), dtype=np.float32),
                frame=pd.DataFrame({"lon": [140.1, 140.2, 140.3], "lat": [35.7, 35.7, 35.8]}),
            ),
            log=log,
        )
        results = Results.__new__(Results)
        results.model = model
        results.daily = pd.DataFrame({"date": dates})
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "u.csv")
            daily_path, totals_path = results.underground_csv(path)
            daily = pd.read_csv(daily_path)
            totals = pd.read_csv(totals_path)
            self.assertEqual(len(daily), 6)
            self.assertIn("lateral_to_pool_m3", daily.columns)
            self.assertEqual(daily.loc[~daily.cyclic, "lateral_to_pool_m3"].sum(), 0)
            self.assertGreater(daily.loc[daily.cyclic, "lateral_to_pool_m3"].sum(), 0)
            self.assertEqual(len(totals), 3)
            results.year = 2022
            results.blocks = pd.DataFrame({"date": dates, "block": ["North", "North"]})
            results.station = pd.DataFrame({"date": dates})
            saved = results.write_csv(os.path.join(folder, "run"))
            self.assertTrue(os.path.isfile(os.path.join(saved, "blocks.csv")))
            self.assertTrue(os.path.isfile(os.path.join(saved, "paddy_daily.csv")))
            daily_all = pd.read_csv(os.path.join(saved, "paddy_daily.csv"))
            self.assertEqual(len(daily_all), 6)
            self.assertIn("et_mm", daily_all.columns)


if __name__ == "__main__":
    unittest.main()

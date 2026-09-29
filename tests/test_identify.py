import unittest

import numpy as np

from cyclic.identify import (
    HydrologyProblem, construct_ratios, qubo_energy, ratio_qubo, solve_qubo_classical,
    spread_field_action, match_block_volumes, loss_split, save_best_ratios, load_best_ratios,
)
from cyclic.physics import Physics
from types import SimpleNamespace

from cyclic.canal import apply_canal, darcy_shares, pixel_ratio_fields
import cyclic




def package_forecasters_only():
    """Hide forecasters registered outside the package (e.g. by example.py) so tests of
    the package fallback don't depend on what other tests imported first."""
    from unittest import mock
    from cyclic.agents import AGENTS
    from cyclic.identify import PACKAGE_FORECASTERS

    kept = {name: cls for name, cls in AGENTS.items() if name in PACKAGE_FORECASTERS}
    return mock.patch.dict(AGENTS, kept, clear=True)

class IdentifyTests(unittest.TestCase):
    def test_spread_field_action_matches_block_volumes(self):
        area = np.array([100.0, 300.0, 200.0, 400.0])
        block = np.array(["North", "North", "South", "South"])
        cyclic = np.array([True, True, True, False])
        irrigation, discharge = spread_field_action(
            {"North": 10.0, "South": 4.0}, 2.0, area, block, cyclic)
        self.assertTrue(np.allclose(irrigation, [10.0, 10.0, 4.0, 0.0]))
        self.assertTrue(np.allclose(discharge, [2.0, 2.0, 2.0, 0.0]))
        mm = match_block_volumes({"North": 4.0, "South": 2.0}, area, block, cyclic)
        self.assertAlmostEqual(float(mm[cyclic & (block == "North")] @ area[cyclic & (block == "North")] / 1000.0), 4.0)
        self.assertAlmostEqual(float(mm[cyclic & (block == "South")] @ area[cyclic & (block == "South")] / 1000.0), 2.0)
        self.assertEqual(float(mm[~cyclic].sum()), 0.0)
        weighted = match_block_volumes(
            {"North": 4.0, "South": 2.0}, area, block, cyclic,
            weights=area * np.array([1.0, 3.0, 1.0, 1.0]))
        self.assertAlmostEqual(float(weighted[cyclic & (block == "North")] @ area[cyclic & (block == "North")] / 1000.0), 4.0)
        self.assertGreater(float(weighted[1]), float(weighted[0]))

    def test_problem_packs_active_ratios_and_holds_darcy_for_later(self):
        physics = Physics(force_satellite_et=True, ratio_split=True,
                          kx_ratio=0.2, ky_ratio=0.2, kz_ratio=0.1,
                          ground_ratio=0.5, farm_to_main=0.7)
        problem = HydrologyProblem(physics=physics)
        vector = problem.pack()
        self.assertEqual(list(problem.names),
                         ["kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio", "farm_to_main"])
        self.assertTrue(np.allclose(vector, [0.2, 0.2, 0.1, 0.5, 0.7]))
        self.assertNotIn("kx_mm_d", problem.names)
        later = HydrologyProblem(include_darcy=True)
        self.assertEqual(later.names[-2:], ("kx_mm_d", "ky_mm_d"))
        obj = {name: 0.0 for name in ("et", "irrigation", "discharge", "pond",
                                      "lateral", "ld_ratio", "farm_to_main", "lfv", "darcy")}
        obj["et"] = 2.0
        self.assertAlmostEqual(problem.scalar(obj), 2.0)
        obj["et"] = 0.0
        obj["lateral"] = 3.0
        self.assertAlmostEqual(problem.scalar(obj), 3.0)

    def test_loss_split_and_saved_best(self):
        import tempfile
        obj = {name: 0.0 for name in ("et", "irrigation", "discharge", "pond",
                                      "lateral", "ld_ratio", "farm_to_main", "lfv", "darcy")}
        obj["irrigation"] = 1.0
        obj["lfv"] = 1.0
        signed = dict(obj)
        signed["irrigation"] = 0.5
        signed["lfv"] = -0.25
        split = loss_split(obj, signed=signed)
        self.assertAlmostEqual(split.imitator_pct + split.ratios_pct, 100.0)
        self.assertAlmostEqual(split.imitator_pct, 50.0, places=5)
        self.assertAlmostEqual(split.ratios_pct, 50.0, places=5)
        self.assertLess(split.signed_ratio, 0.0)
        with tempfile.TemporaryDirectory() as folder:
            params = construct_ratios(2022)
            path, wrote = save_best_ratios(2022, "season", params, loss=1.0, folder=folder)
            self.assertTrue(wrote)
            path2, wrote2 = save_best_ratios(2022, "season", params, loss=2.0, folder=folder)
            self.assertFalse(wrote2)
            path3, wrote3 = save_best_ratios(2022, "season", {**params, "kx_ratio": 0.1},
                                             loss=0.5, folder=folder)
            self.assertTrue(wrote3)
            loaded = load_best_ratios(2022, "season", folder)
            self.assertAlmostEqual(loaded.loss, 0.5)
            self.assertAlmostEqual(loaded.params["kx_ratio"], 0.1)

    def test_construct_ratios_are_a_simplex(self):
        ratios = construct_ratios(2022)
        allowed = {f.name for f in Physics.__dataclass_fields__.values()}
        shares = darcy_shares(Physics().replace(**{k: v for k, v in ratios.items() if k in allowed}))
        self.assertAlmostEqual(sum(shares.values()), 1.0, places=6)
        self.assertGreater(shares["lfv"], 0.0)
        self.assertGreater(shares["kx"] + shares["ky"], 0.0)
        self.assertGreater(shares["kz"] + shares["ground"], 0.0)
        self.assertAlmostEqual(
            shares["kx"] + shares["ky"] + shares["kz"] + shares["ground"] + shares["lfv"],
            1.0, places=6)
        self.assertTrue(0.0 <= ratios["farm_to_main"] <= 1.0)
        self.assertIn("ratio_finder", cyclic.AGENTS)
        agent = cyclic.make_agent("ratio_finder")
        self.assertTrue(agent.batched)

    def test_ratio_split_partitions_leftover(self):
        leftover = np.array([20.0, 20.0])
        p = Physics(ratio_split=True, pool_capture_share=0.3, kx_ratio=0.2, ky_ratio=0.2,
                    kz_ratio=0.1, ground_ratio=0.2, farm_to_main=0.5, canal_capture_frac=0.0)
        flux = {"drainage_mm": np.array([10.0, 10.0]), "leftover_mm": leftover.copy()}
        apply_canal(flux, np.array([0.0, 0.0]), None, np.full(2, 1000.0), p,
                    cyclic=np.array([True, True]))
        self.assertTrue(np.allclose(flux["canal_out_mm"], [5.0, 5.0]))
        parts = (flux["flow_kx_mm"] + flux["flow_ky_mm"] + flux["flow_kz_mm"]
                 + flux["flow_ground_mm"] + flux["lateral_mm"])
        self.assertTrue(np.allclose(parts, leftover))
        self.assertTrue(np.allclose(flux["lateral_mm"], leftover * 0.3))
        self.assertTrue(np.allclose(flux["stay_mm"], leftover * 0.3))
        self.assertTrue(np.allclose(flux["stay_mm"], flux["flow_kz_mm"] + flux["flow_ground_mm"]))
        self.assertTrue(np.allclose(flux["lfv_mm"], flux["lateral_mm"]))
        self.assertTrue(np.allclose(flux["drain_stock_mm"], 0.0))
        self.assertTrue(np.allclose(flux["lfv_shortfall_mm"], 0.0))

    def test_pixel_ratio_fields_are_uniform_and_sum_to_one(self):
        dist = np.array([0.0, 0.0, 0.0])
        cyclic = np.array([True, True, True])
        fields = pixel_ratio_fields(3, dist, None, cyclic, 0.6, 0.5, 0.2, 400.0,
                                    neighbour_of_remain=0.5)
        parts = (fields["kx_ratio"] + fields["ky_ratio"] + fields["kz_ratio"]
                 + fields["ground_ratio"] + fields["pool_capture_share"])
        self.assertTrue(np.allclose(parts, 1.0))
        self.assertTrue(np.allclose(fields["pool_capture_share"], 0.6))
        self.assertTrue(np.allclose(fields["kx_ratio"] + fields["ky_ratio"], 0.2))
        self.assertTrue(np.allclose(fields["farm_to_main"], 0.5))
        leftover = np.array([20.0, 20.0, 20.0])
        p = Physics(ratio_split=True, farm_to_main=0.5)
        flux = {"drainage_mm": np.array([10.0, 10.0, 10.0]), "leftover_mm": leftover.copy()}
        apply_canal(flux, dist, None, np.full(3, 1000.0), p, cyclic=cyclic, shares=fields)
        self.assertTrue(np.allclose(flux["flow_kx_mm"], flux["flow_ky_mm"]))
        self.assertTrue(np.allclose(
            flux["flow_kx_mm"] + flux["flow_ky_mm"] + flux["flow_kz_mm"]
            + flux["flow_ground_mm"] + flux["lateral_mm"],
            leftover))
        self.assertTrue(np.allclose(flux["lfv_mm"], leftover * 0.6))
        self.assertTrue(np.allclose(flux["drain_stock_mm"], 0.0))

    def test_weighted_share_mean_accepts_daily_maps(self):
        from cyclic.canal import weighted_share_mean, mean_pixel_shares
        area = np.array([1.0, 3.0, 2.0])
        cyclic = np.array([True, True, False])
        daily = np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]])
        mean = weighted_share_mean(daily, area, cyclic)
        expect = (0.2 * 1.0 + 0.3 * 3.0) / 4.0
        self.assertAlmostEqual(mean, expect)
        out = mean_pixel_shares({"kx_ratio": daily}, area, cyclic)
        self.assertAlmostEqual(out["kx_ratio"], expect)

    def test_poisson_hops_mix_over_hop_count(self):
        from cyclic.graph import leftover_graph, hop_stock, poisson_hop_weights
        from cyclic.lateral import knn_edges
        leftover = np.array([100.0, 0.0, 0.0, 0.0])
        dist = np.array([1200.0, 800.0, 400.0, 0.0])
        lon = np.full(4, 140.267)
        lat = np.array([35.818, 35.814, 35.810, 35.807])
        edges = knn_edges(lon, lat, k=3)
        connected = np.ones(4, dtype=bool)
        area = np.full(4, 1000.0)
        graph = leftover_graph(dist, edges, connected, one_way=True)
        weights = poisson_hop_weights(4.0)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(float(weights[0]), 0.0)
        self.assertGreater(len(weights), 5)
        stay, delta = hop_stock(leftover, graph, area, Physics(canal_hops=0.0))
        self.assertTrue(np.allclose(stay, leftover))
        self.assertTrue(np.allclose(delta, 0.0))
        slow = Physics(canal_hops=1.0, canal_length_m=400.0, canal_max_frac=0.9,
                       farm_drain_stage_mm=200.0, farm_drain_ref_mm=200.0)
        fast = Physics(canal_hops=8.0, canal_length_m=400.0, canal_max_frac=0.9,
                       farm_drain_stage_mm=200.0, farm_drain_ref_mm=200.0)
        s1, _ = hop_stock(leftover, graph, area, slow)
        s8, _ = hop_stock(leftover, graph, area, fast)
        self.assertAlmostEqual(float(s1.sum()), 100.0, places=5)
        self.assertAlmostEqual(float(s8.sum()), 100.0, places=5)
        self.assertGreater(float(s8[-1]), float(s1[-1]))
        self.assertLess(float(s8[0]), float(s1[0]))
        pair = leftover_graph(np.array([400.0, 0.0]), knn_edges(
            np.array([140.267, 140.267]), np.array([35.810, 35.807]), k=1),
            np.ones(2, dtype=bool), one_way=True)
        blocked, _ = hop_stock(np.array([20.0, 30.0]), pair, np.full(2, 1000.0), fast)
        self.assertAlmostEqual(float(blocked[0]), 20.0, places=5)
        self.assertAlmostEqual(float(blocked[1]), 30.0, places=5)

    def test_recapture_goes_to_the_pool_the_same_day(self):
        leftover = np.array([20.0, 0.0, 0.0, 0.0])
        dist = np.array([1200.0, 800.0, 400.0, 0.0])
        p = Physics(ratio_split=True, pool_capture_share=1.0, kx_ratio=0.0, ky_ratio=0.0,
                    kz_ratio=0.0, ground_ratio=0.0, canal_hops=6)
        cyclic = np.ones(4, dtype=bool)
        area = np.full(4, 1000.0)
        stock = np.zeros(4)
        flux = {"leftover_mm": leftover.copy(), "drainage_mm": np.zeros(4)}
        apply_canal(flux, dist, None, area, p, drain_stock=stock, cyclic=cyclic)
        self.assertAlmostEqual(float(flux["lfv_mm"][0]), 20.0)
        self.assertAlmostEqual(float(flux["lfv_mm"][1:].sum()), 0.0)
        self.assertTrue(np.allclose(flux["drain_stock_mm"], 0.0))
        self.assertTrue(np.allclose(stock, 0.0))

    def test_recapture_is_clipped_by_infiltration(self):
        leftover = np.array([20.0, 20.0])
        p = Physics(ratio_split=True, pool_capture_share=0.8, kx_ratio=0.0, ky_ratio=0.0,
                    kz_ratio=0.1, ground_ratio=0.1)
        flux = {"drainage_mm": np.zeros(2), "leftover_mm": leftover.copy(),
                "infiltration_mm": np.array([5.0, 20.0])}
        apply_canal(flux, np.array([0.0, 0.0]), None, np.full(2, 1000.0), p,
                    cyclic=np.array([True, True]))
        self.assertAlmostEqual(float(flux["recapture_intended_mm"][0]), 16.0)
        self.assertAlmostEqual(float(flux["lfv_mm"][0]), 5.0)
        self.assertAlmostEqual(float(flux["lfv_shortfall_mm"][0]), 11.0)
        self.assertAlmostEqual(float(flux["lfv_mm"][1]), 16.0)
        self.assertAlmostEqual(float(flux["lfv_shortfall_mm"][1]), 0.0)

    def test_recapture_fills_unsaturated_neighbours_not_saturated_ones(self):
        from cyclic.canal import divert_recapture_to_unsaturated
        from cyclic.lateral import knn_edges
        lon = np.array([140.267, 140.267])
        lat = np.array([35.810, 35.807])
        edges = knn_edges(lon, lat, k=1)
        cyclic = np.array([True, True])
        area = np.full(2, 1000.0)
        p = Physics()
        wet = divert_recapture_to_unsaturated(
            {"lfv_mm": np.array([10.0, 0.0])},
            np.array([0.0, 0.0]), np.array([0.0, 0.0]),
            edges, area, cyclic, p)
        self.assertAlmostEqual(float(wet["lfv_mm"][0]), 10.0)
        self.assertTrue(np.allclose(wet["lfv_to_soil_mm"], 0.0))
        surface = np.array([0.0, 8.0])
        deficit = np.array([0.0, 0.0])
        dry = divert_recapture_to_unsaturated(
            {"lfv_mm": np.array([10.0, 0.0])},
            surface, deficit, edges, area, cyclic, p)
        self.assertLess(float(dry["lfv_mm"][0]), 10.0)
        self.assertGreater(float(dry["lfv_to_soil_mm"][0]), 0.0)
        self.assertGreater(float(dry["neighbor_fill_mm"][1]), 0.0)
        self.assertLess(float(surface[1]), 8.0)

    def test_lateral_hops_then_is_caught_at_the_main_canal(self):
        leftover = np.array([20.0, 0.0, 0.0, 0.0])
        dist = np.array([1200.0, 800.0, 400.0, 0.0])
        p = Physics(ratio_split=True, pool_capture_share=0.0, kx_ratio=0.5, ky_ratio=0.5,
                    kz_ratio=0.0, ground_ratio=0.0, canal_hops=6, canal_max_frac=0.9)
        from cyclic.lateral import knn_edges
        lon = np.full(4, 140.267)
        lat = np.array([35.818, 35.814, 35.810, 35.807])
        edges = knn_edges(lon, lat, k=3)
        cyclic = np.ones(4, dtype=bool)
        area = np.full(4, 1000.0)
        flux = {"leftover_mm": leftover.copy(), "drainage_mm": np.zeros(4),
                "infiltration_mm": leftover.copy()}
        apply_canal(flux, dist, edges, area, p, cyclic=cyclic)
        self.assertTrue(np.allclose(flux["lfv_mm"], 0.0))
        self.assertLess(float(flux["neighbor_mm"][0]), 0.0)
        self.assertGreater(float(np.maximum(flux["neighbor_mm"][1:], 0.0).sum()), 0.0)

    def test_graph_routing_is_optional_and_qubo_is_opt_in(self):
        from cyclic.graph import leftover_graph, assignment_qubo
        from cyclic.lateral import knn_edges
        leftover = np.array([20.0, 0.0, 0.0, 0.0])
        dist = np.array([1200.0, 800.0, 400.0, 0.0])
        lon = np.full(4, 140.267)
        lat = np.array([35.818, 35.814, 35.810, 35.807])
        edges = knn_edges(lon, lat, k=3)
        cyclic = np.ones(4, dtype=bool)
        area = np.full(4, 1000.0)
        p = Physics(ratio_split=True, pool_capture_share=1.0, kx_ratio=0.0, ky_ratio=0.0,
                    kz_ratio=0.0, ground_ratio=0.0, canal_length_m=400.0, canal_max_frac=0.9,
                    canal_hops=6, use_graph=False, as_qubo=True)
        flux = {"leftover_mm": leftover.copy(), "drainage_mm": np.zeros(4)}
        apply_canal(flux, dist, edges, area, p, cyclic=cyclic)
        self.assertTrue(np.allclose(flux["neighbor_mm"], 0.0))
        self.assertAlmostEqual(float(flux["lfv_mm"][0]), 20.0)
        self.assertIsNotNone(flux["graph_qubo"])
        self.assertEqual(flux["graph_qubo"].n_var, 4)
        p2 = Physics(ratio_split=True, pool_capture_share=0.0, kx_ratio=0.5, ky_ratio=0.5,
                     kz_ratio=0.0, ground_ratio=0.0, canal_length_m=400.0, canal_max_frac=0.9,
                     canal_hops=6, one_way_to_canal=False)
        flux2 = {"leftover_mm": np.array([0.0, 0.0, 0.0, 20.0]), "drainage_mm": np.zeros(4),
                 "infiltration_mm": np.array([0.0, 0.0, 0.0, 20.0])}
        apply_canal(flux2, dist, edges, area, p2, cyclic=cyclic)
        self.assertGreater(float(np.maximum(flux2["neighbor_mm"][:3], 0.0).sum()), 0.0)
        self.assertTrue(np.allclose(flux2["lfv_mm"], 0.0))
        graph0 = leftover_graph(np.zeros(4), edges, cyclic, one_way=False)
        self.assertGreater(graph0.i.size, 0)
        graph = leftover_graph(dist, edges, cyclic, one_way=True)
        qubo = assignment_qubo(np.full(4, 10.0), graph, 20.0)
        self.assertGreater(len(qubo.Q), 0)
        self.assertEqual(qubo.n_var, 4)

    def test_pixel_lfv_is_nearer_canal_weighted(self):
        edges = SimpleNamespace(
            i=np.array([0]), j=np.array([1]),
            ux=np.array([0.0]), uy=np.array([1.0]),
        )
        near = pixel_ratio_fields(
            2, np.array([0.0, 100.0]), edges, np.array([True, True]), 0.8, 0.5, 0.2, 400.0,
            neighbour_of_remain=0.5)
        far = pixel_ratio_fields(
            2, np.array([2000.0, 2100.0]), edges, np.array([True, True]), 0.8, 0.5, 0.2, 400.0,
            neighbour_of_remain=0.5)
        self.assertAlmostEqual(float(np.mean(near["pool_capture_share"])), 0.8, places=5)
        self.assertAlmostEqual(float(np.mean(far["pool_capture_share"])), 0.8, places=5)
        self.assertGreater(float(near["pool_capture_share"][0]), float(near["pool_capture_share"][1]))
        self.assertTrue(np.allclose(near["kx_ratio"], near["ky_ratio"]))
        parts = (near["kx_ratio"] + near["ky_ratio"] + near["kz_ratio"]
                 + near["ground_ratio"] + near["pool_capture_share"])
        self.assertTrue(np.allclose(parts, 1.0))

    def test_registered_forecasters_pick_lowest_ird_loss(self):
        with package_forecasters_only():
            self._check_registered_forecasters_pick_lowest_ird_loss()

    def _check_registered_forecasters_pick_lowest_ird_loss(self):
        from cyclic.identify import (best_registered_forecaster, freeze_forecaster,
                                     registered_forecasters)
        names = registered_forecasters()
        self.assertNotIn("et_match", names)
        self.assertNotIn("ratio_finder", names)
        self.assertNotIn("et_follower", names)
        # No outside farmers: package fallback is et_match.
        winner = best_registered_forecaster(2022, known={}, run=False)
        self.assertEqual(winner.name, "et_match")
        known = {"custom_out": SimpleNamespace(
            name="custom_out", loss=0.25, fidelity=0.8, results=None)}
        # Outside name only wins when it appears in registered_forecasters();
        # with an empty outside list, fallback still returns et_match.
        days, n = 3, 4
        ir = np.arange(days * n, dtype=float).reshape(days, n)
        dr = ir + 1.0
        frozen = freeze_forecaster(
            SimpleNamespace(model=SimpleNamespace(
                n_days=days, n=n,
                log={"irrigation_mm": ir, "farmer_discharge_mm": dr})),
            name="et_match")
        frozen.reset(None, np.array([1, 3]))
        got_ir, got_dr = frozen.decide_batch(SimpleNamespace(day=1))
        np.testing.assert_allclose(got_ir, ir[1, [1, 3]])
        np.testing.assert_allclose(got_dr, dr[1, [1, 3]])

    def test_parse_finder_method_registered_and_qrc(self):
        from cyclic.identify import parse_finder_method
        import example  # noqa: F401
        self.assertEqual(parse_finder_method("classical"), ("classical", None))
        self.assertEqual(parse_finder_method("omalley"), ("omalley", None))
        self.assertEqual(parse_finder_method("qrc_daily"), ("qrc_daily", None))
        self.assertEqual(parse_finder_method("qrc_spatial"), ("qrc_spatial", None))
        self.assertEqual(parse_finder_method("demo_ratios"), ("registered", "demo_ratios"))
        self.assertEqual(parse_finder_method("registered:demo_ratios"),
                         ("registered", "demo_ratios"))
        with self.assertRaises(ValueError):
            parse_finder_method("not_a_method")

    def test_registered_ratio_tables_run_finder(self):
        from cyclic.identify import run_finder
        import example  # noqa: F401
        with package_forecasters_only():
            report = run_finder(2022, span="tables", method="demo_ratios")
        self.assertEqual(report.method, "registered")
        self.assertEqual(report.ratio_agent, "demo_ratios")
        self.assertEqual(report.forecaster, "et_match")
        self.assertAlmostEqual(
            report.params.get("pool_capture_share", 0.0)
            + report.params["kx_ratio"] + report.params["ky_ratio"]
            + report.params["kz_ratio"] + report.params["ground_ratio"], 1.0, places=5)
        self.assertEqual(len(report.maps.lon), len(report.maps.volumes["kx_ratio"]))

    def test_daily_qrc_tables_run_finder_if_saved(self):
        from cyclic.identify import run_finder
        from cyclic.qrc import load_qrc_result
        if load_qrc_result(2022) is None:
            self.skipTest("no daily QRC save")
        report = run_finder(2022, span="tables", method="qrc_daily")
        self.assertEqual(report.method, "qrc_daily")
        self.assertGreaterEqual(report.params["kx_ratio"], 0.0)
        self.assertLessEqual(report.params["kx_ratio"] + report.params["ky_ratio"]
                             + report.params["kz_ratio"] + report.params["ground_ratio"], 1.0 + 1e-6)

    def test_tables_run_finder_reports_loss_and_maps(self):
        from cyclic.identify import run_finder, ratio_map_figure
        from cyclic import okura
        report = run_finder(2022, span="tables")
        self.assertEqual(getattr(report, "method", "classical"), "classical")
        self.assertEqual(report.days, 0)
        self.assertIn("et", report.objectives)
        self.assertIsNotNone(report.split)
        if report.split.total > 0:
            self.assertAlmostEqual(report.split.imitator_pct + report.split.ratios_pct, 100.0, places=5)
        self.assertGreaterEqual(report.loss, 0.0)
        self.assertEqual(len(report.maps.lon), len(report.maps.volumes["kx_ratio"]))
        cyclic = report.maps.cyclic
        kx = report.maps.volumes["kx_ratio"][cyclic]
        self.assertGreaterEqual(float(np.nanmin(kx)), 0.0)
        self.assertLessEqual(float(np.nanmax(kx)), 1.0)
        fig = ratio_map_figure(report)
        self.assertGreaterEqual(len(fig.axes), 4)
        self.assertEqual(len(okura.calendar_dates(2022)), 365)

    def test_ratio_qubo_decodes_and_prefers_table_ld(self):
        leftover = np.array([3.0, 1.0, 0.0])
        discharge = np.array([2.0, 2.0, 2.0])
        encoding = ratio_qubo(leftover, discharge, {"North": 0.8, "South": 0.8},
                              d_main_target_m3=6.0, bits=3, include_assignment=False)
        self.assertEqual(encoding.n_var, 6)
        zeros = np.zeros(encoding.n_var)
        ones = np.ones(encoding.n_var)
        decoded0 = encoding.decode(zeros)
        decoded1 = encoding.decode(ones)
        self.assertAlmostEqual(decoded0.lateral_per_discharge, 0.0)
        self.assertAlmostEqual(decoded0.farm_to_main, 0.0)
        self.assertAlmostEqual(decoded1.lateral_per_discharge, 3.0)
        self.assertAlmostEqual(decoded1.farm_to_main, 1.0)
        self.assertGreater(qubo_energy(encoding.Q, zeros, encoding.offset),
                           -1e9)
        found = solve_qubo_classical(encoding, sweeps=3, seed=1)
        self.assertEqual(found.bitstring.size, encoding.n_var)
        self.assertTrue(0.0 <= found.lateral_per_discharge <= 3.0)
        self.assertTrue(0.0 <= found.farm_to_main <= 1.0)

    def test_quantum_save_reloads_on_report(self):
        import tempfile
        from cyclic.quantum import solve_omalley_quantum
        from cyclic.identify import save_quantum_ratios, report_from_quantum
        result = solve_omalley_quantum(2022, n_nodes=4, bits=2, backend="exact", seed=0)
        with tempfile.TemporaryDirectory() as folder:
            path = save_quantum_ratios(2022, result, folder=folder)
            self.assertTrue(path.endswith("quantum_2022.npz"))
            report = report_from_quantum(2022, folder=folder)
            self.assertIsNotNone(report)
            self.assertTrue(report.from_disk)
            self.assertEqual(report.quantum.backend, "exact")
            self.assertEqual(len(report.maps.lon), len(report.maps.volumes["kx_ratio"]))
            self.assertAlmostEqual(report.params["farm_to_main"], result.physics["farm_to_main"], places=5)

    def test_omalley_qubo_exact_assigns_bits(self):
        from cyclic.quantum import omalley_qubo, solve_omalley_quantum
        from cyclic.identify import qubo_energy as energy
        encoding = omalley_qubo(2022, n_nodes=4, bits=2, seed=0)
        self.assertLessEqual(encoding.n_var, 16)
        zeros = np.zeros(encoding.n_var)
        ones = np.ones(encoding.n_var)
        self.assertTrue(np.isfinite(energy(encoding.Q, zeros, encoding.offset)))
        self.assertTrue(np.isfinite(energy(encoding.Q, ones, encoding.offset)))
        result = solve_omalley_quantum(2022, n_nodes=4, bits=2, backend="exact", seed=0)
        self.assertEqual(result.backend, "exact")
        self.assertEqual(result.solved.paddy_x.size, result.encoding.grid.labels.size)
        self.assertTrue(0.0 <= result.physics["farm_to_main"] <= 1.0)
        parts = (result.shares["kx_ratio"] + result.shares["ky_ratio"]
                 + result.shares["kz_ratio"] + result.shares["ground_ratio"])
        cyclic = result.encoding.grid.cyclic
        self.assertTrue(np.allclose(parts[cyclic], 1.0, atol=1e-6))

    def test_qrc_figure_from_saved_npz(self):
        from cyclic.qrc import qrc_figure, load_qrc_result
        loaded = load_qrc_result(2022)
        if loaded is None:
            self.skipTest("no qrc_2022.npz")
        fig = qrc_figure(2022)
        self.assertIsNotNone(fig)
        self.assertGreaterEqual(len(fig.axes), 3)

    def test_leftover_weighted_stay_hits_table_target(self):
        from cyclic.identify import leftover_weighted_stay
        shares = {
            "kx_ratio": np.array([0.5, 0.4, 0.3]),
            "ky_ratio": np.array([0.4, 0.4, 0.3]),
            "kz_ratio": np.array([0.05, 0.1, 0.2]),
            "ground_ratio": np.array([0.05, 0.1, 0.2]),
            "kx_sign": np.ones(3), "ky_sign": np.ones(3),
        }
        leftover = np.array([[10.0, 20.0, 5.0], [10.0, 20.0, 5.0]])
        area = np.array([100.0, 100.0, 100.0])
        leftover_weighted_stay(shares, leftover, area, np.ones(3, dtype=bool), 0.21, 0.2)
        stay = shares["kz_ratio"] + shares["ground_ratio"]
        w = leftover.sum(axis=0) * area
        self.assertAlmostEqual(float((stay * w).sum() / w.sum()), 0.21, places=3)
        self.assertTrue(np.allclose(
            shares["kx_ratio"] + shares["ky_ratio"] + shares["kz_ratio"] + shares["ground_ratio"], 1.0))

    def test_daily_qrc_fits_simplex_and_writes_equation(self):
        import tempfile
        from cyclic.qrc import soil_box_ratios, daily_forcing, train_qrc, RATIO_NAMES
        frame = daily_forcing(2022)
        self.assertGreaterEqual(len(frame), 20)
        y, stores = soil_box_ratios(frame)
        self.assertTrue(np.allclose(y.sum(axis=1), 1.0, atol=1e-6))
        self.assertEqual(stores.shape[1], 2)
        with tempfile.TemporaryDirectory() as folder:
            result = train_qrc(2022, backend="classical", n_qubits=3, layers=1,
                               holdout_year=None, folder=folder)
        self.assertLess(result.rmse, 0.25)
        for name in RATIO_NAMES:
            self.assertTrue(0.0 <= result.means[name] <= 1.0)
        self.assertIn("S_top(t+1)", result.equation_text)
        self.assertIn("stay(t)", result.equation_text)
        self.assertGreaterEqual(result.eq_store.r2, 0.0)

    def test_spatial_qrc_uniform_recapture_and_pinned_stay(self):
        import tempfile
        from cyclic.qrc import fit_spatial_qrc
        n, days = 8, 6
        lon = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        lat = np.array([0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3])
        dist = np.array([10.0, 20.0, 30.0, 40.0, 400.0, 450.0, 500.0, 550.0])
        cyclic = np.ones(n, dtype=bool)
        area = np.full(n, 1000.0)
        leftover = np.linspace(12.0, 45.0, days)[:, None] * (0.4 + dist / 200.0)
        ir = leftover + 5.0
        dr = np.full_like(leftover, 3.0)
        et = np.full_like(leftover, 2.0)
        rain = np.full(days, 1.0)
        result = fit_spatial_qrc(2022, leftover, ir, dr, et, rain, lon, lat, dist,
                                 cyclic, area, n_clusters=2, n_qubits=3, layers=1, seed=0,
                                 folder=tempfile.mkdtemp())
        lfv = result.shares["pool_capture_share"]
        neighbour = result.shares["kx_ratio"] + result.shares["ky_ratio"]
        stay = result.shares["kz_ratio"] + result.shares["ground_ratio"]
        self.assertEqual(result.n_clusters, 2)
        self.assertTrue(np.allclose(lfv + neighbour + stay, 1.0, atol=1e-5))
        self.assertTrue(np.allclose(result.shares["kx_ratio"], result.shares["ky_ratio"], atol=1e-6))
        self.assertTrue(np.allclose(
            result.daily["kx_ratio"] + result.daily["ky_ratio"]
            + result.daily["kz_ratio"] + result.daily["ground_ratio"]
            + result.daily["pool_capture_share"], 1.0, atol=1e-5))
        self.assertIn("lon", result.equation_text)
        self.assertIn("stay(t,x,y)", result.equation_text)
        self.assertAlmostEqual(result.stay_weighted, result.stay_target, places=3)
        self.assertIn("S_top(t+1)", result.equation_text)

    def test_ratio_holdout_freezes_train_year_capture(self):
        from cyclic.identify import ratio_holdout
        out = ratio_holdout("tables", train_year=2021, eval_year=2022, simulate=False)
        self.assertEqual(out.train_year, 2021)
        self.assertEqual(out.eval_year, 2022)
        self.assertAlmostEqual(out.train_capture, construct_ratios(2021)["pool_capture_share"], places=4)
        self.assertAlmostEqual(out.eval_capture, construct_ratios(2022)["pool_capture_share"], places=6)
        self.assertGreater(abs(out.train_capture - out.eval_capture), 0.03)
        self.assertGreater(out.capture_error, 0.0)
        self.assertIsNone(out.eval)
        stay = ratio_holdout("stay", train_year=2021, eval_year=2022, simulate=False)
        self.assertAlmostEqual(stay.train_capture, out.train_capture, places=4)

    def test_soil_flow_follows_neighbour_head_not_et(self):
        from cyclic.identify import soil_flow_holdout
        from cyclic.lateral import cardinal_edges, cardinal_neighbours, neighbour_head_features, knn_edges
        depth = np.array([40.0, 10.0, 10.0])
        lon = np.array([140.0, 140.0, 140.0])
        lat = np.array([35.81, 35.80, 35.79])
        mean, drop = neighbour_head_features(depth, knn_edges(lon, lat, k=2))
        self.assertGreater(float(drop[0]), 0.0)
        self.assertLess(float(drop[1]), float(drop[0]))
        chain = np.array([140.267, 140.267, 140.267, 140.267])
        chain_lat = np.array([35.818, 35.814, 35.810, 35.807])
        nbr = cardinal_neighbours(chain, chain_lat)
        self.assertEqual(int(nbr.south[0]), 1)
        self.assertEqual(int(nbr.north[1]), 0)
        self.assertEqual(int(nbr.south[1]), 2)
        self.assertEqual(int(nbr.east[1]), -1)
        self.assertEqual(int(nbr.west[1]), -1)
        edges = cardinal_edges(chain, chain_lat)
        self.assertGreater(edges.i.size, 0)
        out = soil_flow_holdout(train_year=2021, eval_year=2022, counterfactual="et_follower")
        self.assertFalse(out.applied_share_known)
        self.assertTrue(out.leftover_is_residual)
        self.assertTrue(out.hops_follow_head)
        self.assertFalse(out.hops_follow_et)
        self.assertEqual(out.counterfactual_ir, "et_follower")
        self.assertFalse(out.uses_soil_properties)
        self.assertTrue(out.farmer_invariant)
        self.assertTrue(out.sees_neighbour_water)
        self.assertTrue(out.sees_neighbour_et)
        self.assertEqual(out.equation, "head_equalise")
        self.assertEqual(out.form, "physics")
        self.assertTrue(out.ml_problem)
        self.assertFalse(out.fitted)
        lin = soil_flow_holdout(train_year=2021, eval_year=2022, equation="linear")
        self.assertEqual(lin.form, "linear")
        self.assertTrue(lin.fitted)
        self.assertEqual(lin.n_parameters, 9)
        self.assertTrue(lin.hops_follow_head)
        self.assertFalse(lin.hops_follow_et)
        self.assertTrue(lin.per_paddy)
        self.assertTrue(lin.checks_cardinal_neighbours)
        self.assertEqual(lin.directions, ("up", "down", "left", "right"))
        self.assertEqual(lin.n_paddies, 4)
        nonlin = soil_flow_holdout(train_year=2021, eval_year=2022, equation="nonlinear")
        self.assertEqual(nonlin.form, "nonlinear")
        self.assertTrue(nonlin.fitted)
        self.assertEqual(nonlin.n_parameters, 21)
        self.assertTrue(nonlin.hops_follow_head)
        self.assertFalse(nonlin.hops_follow_et)
        self.assertEqual(out.historical_scenarios, (2021, 2022))
        self.assertEqual(np.asarray(out.neighbour_et_mm).size, 4)
        from cyclic.lateral import neighbour_et_features
        et = np.array([8.0, 2.0, 1.0])
        et_mean, et_drop = neighbour_et_features(et, knn_edges(lon, lat, k=2))
        self.assertGreater(float(et_mean[1]), 1.0)
        self.assertGreater(float(et_drop[0]), 0.0)
        from cyclic.identify import discover_lateral_equation
        ranked = discover_lateral_equation(train_year=2021, eval_year=2022)
        self.assertFalse(ranked.two_separate_models)
        self.assertTrue(ranked.one_equation)
        self.assertEqual(ranked.forms, ("linear", "nonlinear"))
        self.assertEqual(ranked.baseline, "head_equalise")
        self.assertTrue(ranked.linear.fitted)
        self.assertTrue(ranked.nonlinear.fitted)
        self.assertEqual(ranked.physics.equation, "head_equalise")
        one = discover_lateral_equation(equation="head_equalise", train_year=2021, eval_year=2022)
        self.assertEqual(one.equation, "head_equalise")

    def test_equation_from_cache_fits_2021_scores_2022(self):
        import os
        import tempfile

        import pandas as pd

        from cyclic.identify import (
            _field, equation_from_fields, equation_score_table, linear_equation_text,
            save_year_field, load_year_field, equation_from_cache, equation_cache_status,
        )

        lon = np.array([140.267, 140.267, 140.267, 140.267])
        lat = np.array([35.818, 35.814, 35.810, 35.807])
        geometry = SimpleNamespace(
            paddy_id=np.array(["a", "b", "c", "d"]),
            lon=lon, lat=lat, area=np.full(4, 1000.0),
            cyclic=np.ones(4, dtype=bool), dist=np.array([1200.0, 800.0, 400.0, 0.0]),
        )
        dates_2021 = pd.date_range("2021-04-17", periods=5, freq="D")
        dates_2022 = pd.date_range("2022-04-17", periods=5, freq="D")
        leftover_2021 = np.vstack([
            np.array([80.0, 10.0, 0.0, 0.0]),
            np.array([70.0, 12.0, 2.0, 0.0]),
            np.array([60.0, 14.0, 4.0, 1.0]),
            np.array([50.0, 16.0, 6.0, 2.0]),
            np.array([40.0, 18.0, 8.0, 3.0]),
        ])
        leftover_2022 = leftover_2021 * 0.4 + 2.0
        et_2021 = np.clip(leftover_2021 / 10.0, 0.5, 8.0)
        et_2022 = np.clip(leftover_2022 / 10.0, 0.5, 8.0)
        inf_2021 = leftover_2021 * 0.35
        inf_2022 = leftover_2022 * 0.35
        train = _field(2021, leftover_2021, et_2021, None, geometry, dates_2021, "test-2021",
                       infiltration=inf_2021)
        eval_field = _field(2022, leftover_2022, et_2022, None, geometry, dates_2022, "test-2022",
                            infiltration=inf_2022)
        report = equation_from_fields(train, eval_field, sample_days=5, save=False)
        self.assertEqual(report.train_year, 2021)
        self.assertEqual(report.eval_year, 2022)
        self.assertFalse(report.two_separate_models)
        self.assertTrue(report.one_equation)
        self.assertTrue(report.farmer_model_separate)
        self.assertTrue(report.linear.fitted)
        self.assertTrue(report.nonlinear.fitted)
        self.assertEqual(report.linear.n_parameters, 14)
        self.assertEqual(report.nonlinear.n_parameters, 26)
        self.assertEqual(report.linear_paddy.n_parameters, 14 * 4)
        self.assertEqual(report.scopes, ("general", "per paddy"))
        self.assertTrue(np.isfinite(report.linear_paddy.eval_rmse))
        self.assertTrue(np.isfinite(report.recapture_paddy.eval_rmse))
        self.assertEqual(report.leftover_equations, ("neighbour_trickle", "recapture", "stay"))
        self.assertIsNotNone(report.linear.share_weights)
        self.assertGreater(report.linear.capture_eval, 0.0)
        self.assertLessEqual(report.linear.capture_eval, 1.0 + 1e-6)
        self.assertIn("dL_i", report.linear.equation_text)
        self.assertIn("leftover_mm", report.linear.equation_text)
        self.assertIn("infiltrated_mm", report.recapture.equation_text)
        self.assertIn("to_pool_i", report.linear.equation_text)
        self.assertIn("stay_i", report.stay.equation_text)
        self.assertIn("drop_up", linear_equation_text(report.linear.weights,
                                                      extra_names=report.linear.extra_names))
        self.assertLess(report.linear.train_rmse, 5.0)
        self.assertLess(report.head_equalise.train_rmse, 1e-6)
        table = equation_score_table(report)
        self.assertEqual(list(table["form"])[:2], ["head_equalise", "linear"])
        self.assertIn("linear_paddy", list(table["form"]))
        self.assertIn("per paddy", list(table["scope"]))
        self.assertIn("recapture", list(table["form"]))
        self.assertIn("stay", list(table["form"]))
        with tempfile.TemporaryDirectory() as folder:
            save_year_field(train, folder)
            save_year_field(eval_field, folder)
            self.assertTrue(equation_cache_status(2021, folder).usable)
            loaded = load_year_field(2021, folder)
            self.assertEqual(loaded.n, 4)
            self.assertEqual(loaded.n_days, 5)
            self.assertIsNotNone(loaded.infiltration)
            cached = equation_from_cache(2021, 2022, sample_days=5, folder=folder, save=True)
            self.assertEqual(cached.eval_year, 2022)
            self.assertTrue(os.path.isfile(cached.saved))
            self.assertIn("to_pool_i", cached.linear.equation_text)

    def test_pin_leftover_to_et_and_lfv(self):
        from cyclic.identify import (
            leftover_pinned, pin_capture_to_lfv, pin_groundwater_for_et, table_lfv_m3,
        )

        irrigation = np.array([[20.0, 10.0], [8.0, 12.0]])
        rain = np.array([[1.0, 1.0], [2.0, 2.0]])
        drainage = np.array([[4.0, 3.0], [1.0, 2.0]])
        et_target = np.array([[5.0, 6.0], [3.0, 4.0]])
        leftover = leftover_pinned(irrigation, rain, drainage, et_target)
        self.assertTrue(np.allclose(leftover, irrigation + rain - drainage - et_target))
        et_fao = et_target - 2.0
        need = pin_groundwater_for_et(et_target, et_mm=et_fao)
        self.assertTrue(np.allclose(need, 2.0))
        area = np.array([5.0e8, 5.0e8])
        cyclic = np.array([True, True])
        infiltration = leftover * 0.8
        _leftover, to_pool, stay, pin, groundwater = pin_capture_to_lfv(
            leftover, area, cyclic, 2022, infiltration_mm=infiltration,
            et_target_mm=et_target, et_mm=et_fao)
        self.assertFalse(pin.force_satellite_et)
        self.assertAlmostEqual(pin.drain_capture_m3 + pin.ground_stay_m3, pin.leftover_m3)
        self.assertTrue(np.allclose(to_pool + stay, np.maximum(leftover, 0.0)))
        recapturable = float((np.minimum(infiltration, np.maximum(leftover, 0.0)) * area).sum() / 1000.0)
        self.assertAlmostEqual(pin.capture_share, table_lfv_m3(2022) / recapturable, places=6)
        self.assertLess(pin.capture_share, 1.0)
        self.assertTrue(pin.lfv_hit)
        self.assertTrue(np.allclose(groundwater, 2.0))
        frozen_leftover, frozen_pool, frozen_stay, frozen, _ = pin_capture_to_lfv(
            leftover, area, cyclic, 2022, infiltration_mm=infiltration, capture=0.5)
        self.assertTrue(frozen.frozen_capture)
        self.assertAlmostEqual(frozen.capture_share, 0.5)
        self.assertAlmostEqual(frozen.drain_capture_m3, 0.5 * recapturable, places=3)
        self.assertTrue(np.allclose(frozen_pool + frozen_stay, np.maximum(leftover, 0.0)))

    def test_leftover_fate_width_does_not_depend_on_infiltration(self):
        from cyclic.agents import LeftoverFate, leftover_fate_design
        from cyclic.identify import _field_graph

        lon = np.array([140.267, 140.267, 140.267, 140.267])
        lat = np.array([35.818, 35.814, 35.810, 35.807])
        field = SimpleNamespace(
            lon=lon, lat=lat, cyclic=np.ones(4, dtype=bool),
            dist=np.array([1200.0, 800.0, 400.0, 0.0]), area=np.full(4, 1000.0),
        )
        graph = _field_graph(field)
        leftover = np.array([10.0, 8.0, 4.0, 1.0])
        present = np.ones((4, 4))
        x_none, names_none = leftover_fate_design(leftover, present)
        x_inf, names_inf = leftover_fate_design(leftover, present, infiltration=leftover * 0.4,
                                                cyclic=field.cyclic, dist=field.dist)
        self.assertEqual(x_none.shape[1], 9)
        self.assertEqual(x_inf.shape[1], 9)
        self.assertEqual(names_none, names_inf)
        shares = [{
            "kx_ratio": np.full(4, 0.2), "ky_ratio": np.full(4, 0.2),
            "kz_ratio": np.full(4, 0.3), "ground_ratio": np.full(4, 0.3),
        }]
        share_row = np.column_stack([shares[0][k] for k in ("kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio")])
        fate = LeftoverFate()
        fate.fit([leftover], [share_row], graph, infiltrations=[None],
                 cyclic=field.cyclic, dist=field.dist)
        pool, stay = fate.predict_mm(leftover, graph, infiltration=leftover * 0.4,
                                     cyclic=field.cyclic, dist=field.dist)
        self.assertEqual(pool.shape, leftover.shape)
        self.assertTrue(np.allclose(pool + stay, leftover))

    def test_relative_et_route_fills_short_neighbour(self):
        from cyclic.identify import relative_et_route_day

        leftover = np.array([10.0, -6.0, 2.0])
        graph = SimpleNamespace(
            east=np.array([1, 2, -1]), west=np.array([-1, 0, 1]),
            north=np.array([-1, -1, -1]), south=np.array([-1, -1, -1]),
        )
        routed = relative_et_route_day(leftover, graph, hops=8)
        self.assertLess(float(routed.unfilled[1]), 1e-6)
        self.assertGreaterEqual(float(routed.received[1]), 6.0 - 1e-6)
        self.assertFalse(bool(routed.flow.sum() < 0))
        chain = np.array([10.0, 0.0, -6.0])
        relayed = relative_et_route_day(chain, graph, hops=8)
        self.assertLess(float(relayed.unfilled[2]), 1e-4)


if __name__ == "__main__":
    unittest.main()

"""The manuscript must describe this repository, not a different model."""

import inspect
import unittest

import numpy as np

from cyclic import env as env_mod
from cyclic import identify, manuscript, methods, okura, physics
from cyclic.graph import hop_stock, leftover_graph, leftover_mu, poisson_hop_weights
from cyclic.lateral import knn_edges
from cyclic.physics import Physics, State, advance


class PaperClaimTests(unittest.TestCase):
    def setUp(self):
        self.n = manuscript.live_numbers(2022)
        self.text = manuscript.manuscript_markdown(2022)
        self.methods = methods.methods_markdown(2022)

    def test_defaults_match_what_the_paper_states(self):
        p = Physics()
        self.assertFalse(p.force_satellite_et)
        self.assertFalse(self.n.force_satellite_et)
        self.assertFalse(self.n.shares_force_et)
        self.assertTrue(p.use_graph)
        self.assertFalse(p.one_way_to_canal)
        self.assertIsNone(p.canal_hops)
        self.assertEqual(p.leftover_speed_m_d, 2.0)
        self.assertEqual(p.pond_infiltrate_ref_mm, 30.0)
        self.assertNotIn("plow_pan_k_mm_d", identify.ACTIVE)
        self.assertFalse(p.ratio_split)
        shares = identify.construct_ratios(2022)
        self.assertTrue(shares["ratio_split"])
        self.assertFalse(shares["force_satellite_et"])

    def test_capture_is_lfv_over_lfa_and_simplex_sums_to_one(self):
        t5 = okura.TABLE5_1000M3[2022]
        lfa = (t5["LfA_North"] + t5["LfA_South"]) * 1000.0
        lfv = t5["LFV_pump"] * 1000.0
        self.assertAlmostEqual(self.n.capture, lfv / lfa)
        self.assertAlmostEqual(self.n.farm_to_main, identify.construct_ratios(2022)["farm_to_main"])
        self.assertLessEqual(self.n.farm_to_main, 1.0)
        self.assertGreater(self.n.farm_to_main_raw, 0.99)
        self.assertAlmostEqual(self.n.kx + self.n.ky + self.n.kz + self.n.ground, self.n.stay)
        top = Physics().total_evaporable_mm / (
            Physics().total_evaporable_mm + Physics().total_available_mm)
        self.assertAlmostEqual(self.n.kz, (self.n.kz + self.n.ground) * top)
        self.assertIn(f"{self.n.capture:.4f}", self.text)
        self.assertIn(f"{self.n.stay:.4f}", self.text)

    def test_manuscript_does_not_claim_forced_or_held_et(self):
        forbidden = (
            "ET in the identification runs is the satellite target (forced)",
            "Satellite ET is forced",
            "measured IR, D, ET held",
            "that map is imposed",
        )
        blob = self.text + "\n" + self.methods
        for phrase in forbidden:
            self.assertNotIn(phrase, blob)
        self.assertIn("one next-day water-fate predictor", self.text)
        self.assertIn("@register_forecaster", self.text)
        self.assertIn("@register_ratio", self.text)
        self.assertIn("@register_lateral", self.text)
        self.assertIn("without soil properties", self.text.lower())
        self.assertIn("intended IR/D labels", self.text)
        self.assertIn("leftover lateral flow", self.text.lower())
        self.assertIn("cyclic irrigation", self.text)
        self.assertNotIn("two sequential predictors", self.text)
        self.assertIn("separate farmer model", self.text.lower())
        self.assertIn("equation builder", self.text.lower())

    def test_manuscript_does_not_overclaim_what_is_identified(self):
        """Claims the earlier drafts made that the code does not support."""
        forbidden = (
            "Ky larger is consistent with an east–west main drain",
            "Ky larger: main drain is east–west",
            "the predictor APIs and losses do not change when the proxy is replaced",
            "enforced in code, not suggested in comments",
        )
        blob = self.text + "\n" + self.methods
        for phrase in forbidden:
            self.assertNotIn(phrase, blob)
        # The replacements have to be present, not just the old phrasing absent.
        self.assertIn("reported, not identified", self.text)
        self.assertIn("The proxy labels leak", self.text)
        self.assertIn("fits season totals", self.text)
        self.assertIn("algebraic consequences", self.text)
        self.assertIn("not identified", self.methods)
        self.assertIn("force_satellite_et = False", self.text)
        self.assertIn("`force_satellite_et=False`", self.methods)
        self.assertIn("not imposed as the flux", self.methods)

    def test_genuine_dois_are_in_the_manuscript(self):
        dois = (
            "10.1016/j.ejrh.2024.102111",
            "10.1038/s41598-018-25206-0",
            "10.1016/S0034-4257(97)00104-1",
            "10.1016/j.agwat.2005.01.012",
            "10.1103/PhysRevApplied.8.024030",
            "10.1029/2005RG000178",
            "10.1002/qj.3803",
            "10.1016/j.rse.2011.11.026",
            "10.5281/zenodo.14582479",
            "1411.4028",
        )
        for doi in dois:
            self.assertIn(doi, self.text, msg=doi)
        self.assertIn("Allen, R.G., Pereira, L.S., Raes, D., Smith, M., 1998", self.text)
        self.assertIn("FAO Irrigation and Drainage Paper 56", self.text)
        self.assertIn("Hooghoudt, S.B., 1940", self.text)

    def test_related_work_positions_the_novelty_claim(self):
        """§1.1: the prior art that limits what may be claimed as new."""
        dois = (
            "10.1017/eds.2023.28",          # CropGym
            "10.1016/j.compag.2022.107182",  # RL for crop management review
            "10.3390/math13040595",          # AquaCropGymnasium
            "10.5109/24320",                 # cyclic irrigation tank model
            "10.4236/jwarp.2018.101005",     # cyclic irrigation system dynamics
            "10.1016/j.jhydrol.2005.07.007",  # equifinality manifesto
            "10.5194/hess-5-1-2001",         # distributed modelling limits
            "10.1002/hyp.10082",             # GLUE 20 years on
            "10.1016/j.patter.2023.100804",  # leakage in ML-based science
            "10.1038/s42256-020-00257-z",    # shortcut learning
            "10.48550/arXiv.2004.06316",     # learning from aggregate observations
            "10.1016/j.agwat.2008.01.012",   # Playán inter-paddy residual
            "10.1016/j.ecolmodel.2007.02.001",  # Tsubo FAO + Dupuit
            "10.3390/w12041125",             # Song return-flow uncertainty
            "10.11520/jshwr.37.0_309",       # Iwasaki Imbanuma tank model
            "10.1016/j.agwat.2021.106838",   # Chen rice DQN, forecast vs observed rain
            "10.52202/068431-0828",          # CyclesGym
            "10.1029/2020WR028338",          # Prieto mechanism identifiability
            "10.1029/2021WR030705",          # Prieto ungauged ranking
            "10.1007/s10333-016-0556-9",     # Xu dual Kc rice drying-wetting
            "10.1080/03650340.2025.2550632", # He bund lateral seepage
        )
        for doi in dois:
            self.assertIn(doi, self.text, msg=doi)
        self.assertIn("Quadrianto, N., Smola, A.J., Caetano, T.S., Le, Q.V., 2009", self.text)
        self.assertIn("Beven, K., 2006", self.text)
        # The claim has to stay bounded by that prior art.
        self.assertIn("are therefore **not** a contribution of this work", self.text)
        self.assertIn("Simulating cyclic irrigation is therefore not new", self.text)
        self.assertIn("the honest novelty claim is narrow", self.text)
        self.assertIn("inferred underground flow ratios", self.text.lower())
        self.assertIn("Playán et al. (2008)", self.text)
        self.assertIn("Tsubo et al. (2007)", self.text)
        self.assertIn("Iwasaki and Iida (2024)", self.text)
        self.assertIn("Song et al. (2020)", self.text)
        self.assertIn("Chen et al. (2021)", self.text)
        self.assertIn("Turchetta et al., 2022", self.text)
        self.assertIn("Prieto et al. (2021)", self.text)
        self.assertIn("Xu et al. (2017)", self.text)
        self.assertIn("He et al. (2025)", self.text)
        self.assertIn("kc_wet", self.text)
        self.assertIn("PROCESS_ROLE", self.text)
        self.assertIn("LFV_ACCEPT_FRAC", self.text)
        self.assertIn("cyclic.validation.split", self.text)
        self.assertIn("ratio_holdout", self.text)
        self.assertIn("FrozenStay", self.text)
        self.assertIn("per-paddy irrigation share is unknown", self.text.lower())
        self.assertIn("neighbour water levels", self.text.lower())
        self.assertIn("neighbour_et_mm", self.text)
        self.assertIn("neighbour fao-56 et", self.text.lower())
        self.assertNotIn("because \\(h\\) does not take ir or et", self.text.lower())
        self.assertIn("soil_flow_holdout", self.text)
        self.assertIn("discover_lateral_equation", self.text)
        self.assertIn("@register_lateral", self.text)
        self.assertIn("without soil properties", self.text.lower())
        self.assertIn("linear or nonlinear", self.text.lower())
        self.assertIn("`linear`", self.text)
        self.assertIn("`nonlinear`", self.text)
        self.assertIn("not two separate models", self.text.lower())
        self.assertIn("forms of the same equation", self.text.lower())
        self.assertIn("separate farmer model", self.text.lower())
        self.assertIn("bund_seepage_mm", self.text)
        self.assertIn("pond_infiltrate_ref_mm", self.text)
        self.assertIn("not a leftover-simplex ratio", self.text)
        self.assertEqual(set(identify.PROCESS_ROLE), set(identify.OBJECTIVES))
        self.assertEqual(identify.PROCESS_ROLE["lfv"], "routing")
        self.assertEqual(identify.PROCESS_ROLE["lateral"], "algebraic")
        self.assertEqual(identify.LFV_ACCEPT_FRAC, 0.5)
        # Lessons have to change the design, not only the citation list.
        self.assertGreaterEqual(self.text.count("What we took"), 5)
        self.assertIn("refuse to rank farmer discharge on the proxy", self.text)
        self.assertIn("put `canal_hops` outside `identify.ACTIVE`", self.text)
        for overclaim in ("the first benchmark", "the first simulator",
                          "we are the first", "a novel framework"):
            self.assertNotIn(overclaim, self.text.lower())

    def test_advance_does_not_force_satellite_et_and_leftover_is_eq3(self):
        source = inspect.getsource(advance)
        self.assertIn("apply_et(state, ks * kcb * et0_mm, ke * et0_mm, p, force=False)", source)
        self.assertIn("if ratio_split:", source)
        state = State.start(3, depth_mm=40.0)
        rain = np.array([5.0, 5.0, 5.0])
        et0 = np.array([4.0, 4.0, 4.0])
        canopy = np.array([0.5, 0.5, 0.5])
        ir = np.array([10.0, 0.0, 8.0])
        d = np.array([3.0, 0.0, 1.0])
        flux = advance(state, rain, et0, canopy, ir, d, Physics())
        leftover = rain + ir - flux["drainage_mm"] - flux["et_mm"]
        self.assertTrue(np.allclose(flux["leftover_mm"], leftover))
        cap = max(Physics().kc_max, Physics().kc_wet)
        self.assertTrue(np.all(flux["et_mm"] <= et0 * cap + 1e-9))
        dry = State.start(1, depth_mm=40.0)
        wet = State.start(1, depth_mm=40.0)
        et0_1 = np.array([4.0])
        canopy_1 = np.array([0.5])
        dry_flux = advance(dry, np.array([0.0]), et0_1, canopy_1, np.array([0.0]), np.array([0.0]), Physics())
        wet_flux = advance(wet, np.array([5.0]), et0_1, canopy_1, np.array([8.0]), np.array([0.0]), Physics())
        self.assertGreater(float(wet_flux["ke"][0]), float(dry_flux["ke"][0]))
        self.assertAlmostEqual(Physics().kc_wet, 1.29)
        self.assertAlmostEqual(Physics().bund_seepage_mm, 150.0)

    def test_identification_and_env_loss_terms(self):
        for key in ("et", "irrigation", "discharge", "lateral", "lfv"):
            self.assertEqual(identify.DEFAULT_WEIGHTS[key], 1.0)
            self.assertIn(key, identify.OBJECTIVES)
        self.assertEqual(env_mod.DEFAULT_WEIGHTS["et"], 1.0)
        self.assertEqual(env_mod.DEFAULT_WEIGHTS["irrigation"], 1.0)
        self.assertEqual(env_mod.DEFAULT_WEIGHTS["drainage"], 1.0)
        self.assertEqual(env_mod.DEFAULT_WEIGHTS["lateral"], 1.0)
        self.assertEqual(env_mod.GYM_WEIGHTS["irrigation"], 0.0)
        self.assertEqual(env_mod.GYM_WEIGHTS["lfv"], 1.0)
        self.assertIn("Satellite ET, discharge, leftover", self.text)

    def test_run_finder_uses_best_registered_forecaster(self):
        source = inspect.getsource(identify.run_finder)
        self.assertIn("best_registered_forecaster", source)
        self.assertIn("freeze_forecaster", source)
        self.assertIn("PACKAGE_FORECASTERS", inspect.getsource(identify.registered_forecasters))
        self.assertIn("ratio_finder", identify.PACKAGE_FORECASTERS)
        self.assertIn("et_follower", identify.PACKAGE_FORECASTERS)

    def test_poisson_mixture_and_head_equalising_hops(self):
        weights = poisson_hop_weights(4.0)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        leftover = np.array([100.0, 0.0, 0.0, 0.0])
        dist = np.array([1200.0, 800.0, 400.0, 0.0])
        lon = np.full(4, 140.267)
        lat = np.array([35.818, 35.814, 35.810, 35.807])
        graph = leftover_graph(dist, knn_edges(lon, lat, k=3), np.ones(4, dtype=bool), one_way=True)
        p = Physics()
        self.assertGreater(leftover_mu(graph, p), 0.0)
        stay, _ = hop_stock(leftover, graph, np.full(4, 1000.0), Physics(canal_hops=0.0))
        self.assertTrue(np.allclose(stay, leftover))
        slow = Physics(canal_hops=1.0, canal_max_frac=0.9)
        fast = Physics(canal_hops=8.0, canal_max_frac=0.9)
        s1, _ = hop_stock(leftover, graph, np.full(4, 1000.0), slow)
        s8, _ = hop_stock(leftover, graph, np.full(4, 1000.0), fast)
        self.assertAlmostEqual(float(s1.sum()), 100.0, places=5)
        self.assertAlmostEqual(float(s8.sum()), 100.0, places=5)
        self.assertGreater(float(s8[-1]), float(s1[-1]))
        pair = leftover_graph(
            np.array([400.0, 0.0]),
            knn_edges(np.array([140.267, 140.267]), np.array([35.810, 35.807]), k=1),
            np.ones(2, dtype=bool), one_way=True)
        blocked, _ = hop_stock(np.array([20.0, 30.0]), pair, np.full(2, 1000.0), fast)
        self.assertAlmostEqual(float(blocked[0]), 20.0, places=5)
        self.assertAlmostEqual(float(blocked[1]), 30.0, places=5)
        self.assertIn("wetter plot to the drier plot", self.text)
        self.assertIn("High satellite ET does not pull leftover", self.text)

    def test_routing_sees_only_the_sum_of_kx_and_ky(self):
        """§3.2: the axis split is reported, not identified."""
        from cyclic.canal import apply_canal

        n = 6
        lon = np.full(n, 140.267)
        lat = np.linspace(35.807, 35.820, n)
        dist = np.linspace(0.0, 1500.0, n)[::-1].copy()
        edges = knn_edges(lon, lat, k=3)

        def routed(kx, ky):
            p = Physics(ratio_split=True, canal_routing=True)
            shares = {"kx_ratio": np.full(n, kx), "ky_ratio": np.full(n, ky),
                      "kz_ratio": np.full(n, 0.04), "ground_ratio": np.full(n, 0.17),
                      "farm_to_main": np.ones(n)}
            flux = {"leftover_mm": np.full(n, 12.0), "drainage_mm": np.full(n, 3.0)}
            apply_canal(flux, dist, edges, np.full(n, 10000.0), p,
                        drain_stock=np.zeros(n), leftover_split=True,
                        cyclic=np.ones(n, dtype=bool), shares=shares)
            return np.asarray(flux["lfv_mm"], dtype=float)

        self.assertTrue(np.allclose(routed(0.40, 0.39), routed(0.10, 0.69)))
        self.assertIn("only their sum is used", self.text)
        self.assertIn("Pond leftover cannot be LFV", self.text)
        self.assertIn("divert_recapture_to_unsaturated", self.text)
        self.assertIn("delayed ET", self.text)

    def test_constructed_capture_field_mean_equals_the_table_pin(self):
        """§3.2: recapture is uniform on cyclic cells, mean equals the Table 5 pin."""
        from cyclic.canal import mean_pixel_shares, pixel_ratio_fields

        n = 400
        rng = np.random.default_rng(0)
        dist = rng.uniform(0.0, 2000.0, n)
        area = rng.uniform(1000.0, 30000.0, n)
        lon = 140.26 + rng.uniform(0.0, 0.02, n)
        lat = 35.80 + rng.uniform(0.0, 0.02, n)
        cyclic = np.ones(n, dtype=bool)
        fields = pixel_ratio_fields(n, dist, knn_edges(lon, lat, k=4), cyclic,
                                    self.n.capture, self.n.farm_to_main, 0.2, 400.0,
                                    area_m2=area)
        got = mean_pixel_shares(fields, area, cyclic)
        self.assertAlmostEqual(got["pool_capture_share"], self.n.capture, places=4)
        self.assertTrue(np.all(fields["pool_capture_share"] <= 1.0 + 1e-9))

    def test_daily_ir_and_d_errors_are_reported_at_weight_zero(self):
        """§2.4: the fitted loss is on season totals; daily errors are reported."""
        for key in identify.REPORTED_ONLY:
            self.assertIn(key, identify.OBJECTIVES)
            self.assertEqual(identify.DEFAULT_WEIGHTS[key], 0.0)
        obj, signed = identify.table_closure_objectives(2022)
        for key in identify.REPORTED_ONLY:
            self.assertIn(key, obj)
            self.assertIn(key, signed)
        self.assertIn("ir_daily", self.text)

    def test_mu_is_speed_over_spacing_not_a_ratio(self):
        graph = leftover_graph(
            np.array([10.0, 0.0]),
            knn_edges(np.array([140.0, 140.0]), np.array([35.80, 35.81]), k=1),
            np.ones(2, dtype=bool))
        p = Physics(leftover_speed_m_d=2.0, canal_hops=None)
        mu = leftover_mu(graph, p)
        dx = float(np.median(np.maximum(graph.length_m, 1.0)))
        self.assertAlmostEqual(mu, 2.0 / dx)
        self.assertNotIn("canal_hops", identify.ACTIVE)
        self.assertIn("not a fifth leftover ratio", self.methods)
        self.assertIn("not a fifth simplex coordinate", self.text)

    def test_vu_kc_anchors_and_carlson_cover(self):
        self.assertEqual(physics.KC_INI, 1.10)
        self.assertEqual(physics.KC_MID, 1.155)
        self.assertAlmostEqual(self.n.kc_at_bare, 1.10)
        self.assertAlmostEqual(self.n.kc_at_full, 1.155)
        self.assertAlmostEqual(float(physics.canopy_cover(0.15)), 0.0)
        self.assertAlmostEqual(float(physics.canopy_cover(0.80)), 1.0)
        self.assertIn("Vu, Watanabe and Takagi (2005", self.text)
        self.assertIn("Carlson and Ripley", self.text)

    def test_okura_eq3_closes_table4_both_years(self):
        for year in okura.YEARS:
            checks = methods.table_checks(year)
            eq3 = checks.all[checks.all["check"].str.startswith("Eq.3 Lf_")]
            self.assertTrue(eq3["ok"].all(), msg=eq3.to_string())
            for block in okura.BLOCKS:
                row = okura.TABLE4_MM[(block, year)]
                leftover = row["irrigation"] + row["rain"] - row["drainage"] - row["et"]
                self.assertLessEqual(abs(leftover - row["lateral"]), 1.0)

    def test_paper_pack_writes_manuscript_without_forced_et_sentence(self):
        import os
        import tempfile

        folder = tempfile.mkdtemp()
        out = methods.write_paper_pack(2022, folder=folder)
        self.assertTrue(os.path.isfile(out.manuscript))
        with open(out.manuscript, encoding="utf-8") as handle:
            body = handle.read()
        self.assertIn("Okura, F., Kamei, A., Kato, T., 2025", body)
        self.assertNotIn("ET in the identification runs is the satellite target (forced)", body)
        self.assertIsNone(out.repo_paper)


if __name__ == "__main__":
    unittest.main()

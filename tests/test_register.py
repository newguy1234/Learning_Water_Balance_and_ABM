import unittest

import numpy as np

import cyclic
import example  # noqa: F401


class RegisterTests(unittest.TestCase):
    def test_et_match_is_package_preset(self):
        self.assertIn("et_match", cyclic.FORECASTERS)
        self.assertTrue(issubclass(cyclic.FORECASTERS["et_match"], cyclic.FarmerPolicy))
        self.assertIn("et_match", cyclic.identify.PACKAGE_FORECASTERS)
        self.assertNotIn("et_match", cyclic.identify.registered_forecasters())
        self.assertIn("et_net", cyclic.FORECASTERS)
        self.assertIn("et_net", cyclic.identify.PACKAGE_FORECASTERS)
        self.assertNotIn("et_net", cyclic.identify.registered_forecasters())
        self.assertIn("tables", cyclic.RATIOS)
        self.assertIn("demo_ratios", cyclic.RATIOS)

    def test_example_registers_one_agent_of_each_kind(self):
        import abm
        self.assertIn("my_forecaster", cyclic.identify.registered_forecasters())
        self.assertIn("my_ratios", cyclic.RATIOS)
        self.assertIn("my_farmer", abm.list_agents())
        self.assertIn("my_station", abm.list_stations())

    def test_et_match_keeps_okura_block_totals(self):
        model = cyclic.CyclicModel(2022, agents="et_match", potential="field",
                                   record_paddies=False, save_csv=False)
        model.step()
        day = 0
        area = model.area
        block = model.inputs.block
        cyclic_mask = model.inputs.cyclic
        ir = model.last_irrigation
        base = model.groups[0][0]
        for name in ("North", "South"):
            sel = (block == name) & cyclic_mask
            if not np.any(sel):
                continue
            want = float((base.irrigation[day, sel] * area[sel]).sum())
            got = float((ir[sel] * area[sel]).sum())
            self.assertAlmostEqual(got, want, places=3)

    def test_et_match_does_not_empty_at_midsummer(self):
        import pandas as pd
        from cyclic.agents import ObservationBatch

        model = cyclic.CyclicModel(2022, agents="et_match", potential="field",
                                   record_paddies=False, save_csv=False)
        agent = model.groups[0][0]
        rows = model.groups[0][1]
        n = len(rows)
        batch = ObservationBatch(
            60, pd.Timestamp("2022-06-16"),
            paddy_id=model.inputs.paddy_id[rows],
            block=model.inputs.block[rows],
            cyclic=model.inputs.cyclic[rows],
            area_m2=model.area[rows],
            sat_ndvi=np.full(n, 0.62),
            sat_et_mm=np.full(n, 5.2),
            sat_days_ago=np.full(n, 1.0),
            depth_mm=np.full(n, 25.0),
            et0_mm=np.full(n, 4.8),
            need_mm=np.zeros(n),
            last_et_mm=np.full(n, 4.0),
            neighbour_et_mm=np.full(n, 4.0),
            topsoil_mm=np.full(n, 20.0),
        )
        _, dr = agent.decide_batch(batch)
        # Midsummer must not dump the whole pond (et_follower does).
        self.assertLess(float(np.mean(dr[model.inputs.cyclic[rows]])), 20.0)

    def test_point_targets_cover_each_paddy(self):
        targets = cyclic.point_targets(2022, day=0)
        self.assertGreater(len(targets.table), 0)
        for column in ("et_target_mm", "block_irrigation_mm", "block_drainage_mm",
                       "kx_ratio", "ky_ratio", "kz_ratio", "ground_ratio",
                       "capture", "farm_to_main", "canal_distance_m"):
            self.assertIn(column, targets.table.columns)
        parts = (targets.table["capture"] + targets.table["kx_ratio"] + targets.table["ky_ratio"]
                 + targets.table["kz_ratio"] + targets.table["ground_ratio"])
        cyclic_rows = targets.table["cyclic"]
        np.testing.assert_allclose(parts[cyclic_rows], 1.0, atol=1e-6)
        c = targets.constructed
        self.assertAlmostEqual(targets.stay_target,
                               (1.0 - c["pool_capture_share"]) * (1.0 - c["neighbour_of_remain"]), places=6)

    def test_ratio_agent_absorbs_forecaster_leftover(self):
        model = cyclic.CyclicModel(2022, agents="et_match", potential="field",
                                   record_paddies=True, save_csv=False)
        model.step()
        model.running = False
        season = cyclic.Results(model)
        leftover = cyclic.leftover_from_season(season)
        self.assertEqual(leftover.shape[-1], model.n)
        ratios = cyclic.make_ratio("tables")
        shares = ratios.absorb(season)
        self.assertEqual(len(shares["kx_ratio"]), model.n)
        seen = ratios.observe(season)
        self.assertEqual(len(seen["leftover_mm"]), model.n)
        means = ratios.district_means(shares, model.area, model.inputs.cyclic)
        self.assertAlmostEqual(means["pool_capture_share"] + means["kx_ratio"] + means["ky_ratio"]
                               + means["kz_ratio"] + means["ground_ratio"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()

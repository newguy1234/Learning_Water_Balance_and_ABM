import unittest

import cyclic


class BenchmarkTests(unittest.TestCase):
    def test_ratio_track_names_a_winner(self):
        report = cyclic.ratio_track(2022)
        self.assertIn(report.winner, ("tables_4_5", "fitted_season"))
        self.assertIn("capture", report.table.columns)
        self.assertGreater(float(report.constructed["pool_capture_share"]), 0.0)

    def test_two_way_without_farmers_is_ratios_only(self):
        report = cyclic.two_way(farmers=(), year=2022)
        self.assertIsNone(report.farmers)
        self.assertEqual(report.year, 2022)
        paths = cyclic.write_benchmark(report)
        self.assertIn("ratios", paths)

    def test_register_forecaster_is_the_agent_registry(self):
        @cyclic.register_forecaster("unit_demo")
        class UnitDemo(cyclic.FarmerPolicy):
            def decide_batch(self, obs):
                import numpy as np
                z = np.zeros(obs.depth_mm.shape)
                return z, z
        self.addCleanup(cyclic.FORECASTERS.pop, "unit_demo", None)
        self.assertIn("unit_demo", cyclic.FORECASTERS)
        self.assertIs(cyclic.register_forecaster, cyclic.register_agent)


if __name__ == "__main__":
    unittest.main()

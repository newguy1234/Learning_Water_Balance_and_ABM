import unittest

import numpy as np
import pandas as pd

from cyclic import imitation
from cyclic.env import CyclicEnv, DEFAULT_WEIGHTS


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.observed = pd.DataFrame([
            {"date": pd.Timestamp("2022-06-01") + pd.Timedelta(days=d), "block": block,
             "area_m2": 1000., "irrigation_m3": value, "drainage_m3": value / 2}
            for block in imitation.BLOCKS for d, value in enumerate([100., 200., 300.])])

    def test_irrigation_and_discharge_each_penalize_season_score(self):
        perfect = imitation.score(self.observed, self.observed)
        self.assertEqual(perfect.loss, 0)
        self.assertEqual(perfect.irrigation_fidelity, 1)
        self.assertEqual(perfect.discharge_fidelity, 1)
        for quantity, component in (("irrigation", "irrigation_fidelity"), ("drainage", "discharge_fidelity")):
            with self.subTest(quantity=quantity):
                changed = self.observed.copy()
                changed[quantity + "_m3"] *= 0.5
                score = imitation.score(changed, self.observed)
                self.assertAlmostEqual(getattr(score, component), 0.5)
                self.assertLess(score.fidelity, perfect.fidelity)
                self.assertGreater(score.loss, perfect.loss)

    def test_wrong_daily_timing_penalized_even_with_matching_totals(self):
        for quantity in ("irrigation", "drainage"):
            changed = self.observed.copy()
            changed[quantity + "_m3"] = changed.groupby("block")[quantity + "_m3"].transform(lambda x: x.iloc[::-1].to_numpy())
            score = imitation.score(changed, self.observed)
            self.assertAlmostEqual(score.fidelity, 1)
            self.assertGreater(score.loss, 0)

    def test_default_reward_uses_et_irrigation_discharge_and_lateral(self):
        for key in ("et", "irrigation", "drainage", "lateral"):
            self.assertEqual(DEFAULT_WEIGHTS[key], 1.0)
        env = CyclicEnv(potential="field")
        self.assertFalse(env.physics.force_satellite_et)
        observation, info = env.reset()
        _, reward, _, _, info = env.step((0., 0.))
        self.assertEqual(info["et"]["training_weight"], 1.0)
        self.assertEqual(info["lateral"]["training_weight"], 1.0)
        north = info["blocks"][imitation.BLOCKS[0]]
        self.assertIn("irrigation_reward", north)
        self.assertIn("drainage_reward", north)
        self.assertIn("lateral_reward", north)
        self.assertEqual(len(info["et"]["simulated_mm"]), len(observation))
        self.assertTrue(np.isfinite(info["et"]["reward"]).all())
        self.assertTrue(np.isfinite(reward))
        for name in ("wind_mean_kmh", "wind_max_kmh", "wind_gust_kmh", "wind_dir_deg",
                     "rh_mean_pct", "vpd_max_kpa", "tmean_c", "soil_moisture", "pressure_hpa"):
            values = getattr(observation, name)
            self.assertEqual(len(values), len(observation))
            self.assertTrue(np.isfinite(values).all(), name)

    def test_given_irrigation_gym_scores_fate_not_irrigation(self):
        env = CyclicEnv(potential="field", given_irrigation=True)
        self.assertEqual(env.weights["irrigation"], 0.0)
        self.assertEqual(env.weights["lfv"], 1.0)
        observation, info = env.reset()
        self.assertTrue(info["given_irrigation"])
        self.assertEqual(len(observation.irrigation_sent_mm), len(observation))
        self.assertTrue(np.all(np.isfinite(observation.irrigation_sent_mm)))
        _, reward, _, _, info = env.step((np.zeros(len(observation)), np.zeros(len(observation))))
        self.assertIn("fate", info)
        for key in ("irrigation_sent_mm", "got_mm", "sent_back_mm", "lateral_mm", "lfv_mm"):
            self.assertIn(key, info["fate"])
            self.assertEqual(len(info["fate"][key]), len(observation))
        self.assertEqual(info["lfv"]["training_weight"], 1.0)
        self.assertTrue(np.isfinite(reward))
        self.assertTrue(np.all(np.isfinite(observation.irrigation_sent_mm)))

    def test_spatial_proxy_score_does_not_reward_weather_mean(self):
        from types import SimpleNamespace
        from cyclic.flyover import score_day
        n = 20
        target = np.linspace(2., 5., n)
        inputs = SimpleNamespace(is_pass=np.ones((1,n),dtype=bool),
            sat_pass_et_mm=target[None,:], block=np.array([imitation.BLOCKS[0]]*n),
            area_m2=np.ones(n), dates=pd.to_datetime(['2022-06-01']),
            field_et_mm={imitation.BLOCKS[0]: np.array([3.5])})
        first = score_day(inputs,0,target)[0]
        inputs.field_et_mm[imitation.BLOCKS[0]][0] = 100
        scaled = score_day(inputs,0,target*4)[0]
        self.assertAlmostEqual(first['fidelity'],scaled['fidelity'])
        shuffled = score_day(inputs,0,target[::-1])[0]
        self.assertLess(shuffled['paired_fidelity'],first['paired_fidelity'])


if __name__ == "__main__":
    unittest.main()

"""Gymnasium API smoke tests (requires gymnasium)."""

import unittest

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


@unittest.skipUnless(HAS_GYM, "gymnasium not installed")
class GymEnvTests(unittest.TestCase):
    def test_make_reset_step(self):
        import cyclic.gym_env

        env = gym.make("Cyclic-Imbanuma-v0", year=2021, given_irrigation=True)
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.ndim, 2)
        self.assertIn("feature_names", info)
        self.assertIn("obs_equation", info)
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(action)
        self.assertIsInstance(reward, float)
        self.assertFalse(truncated)
        env.close()

    def test_equation_hides_satellite_by_default(self):
        from cyclic.gym_env import CyclicGym

        env = CyclicGym(year=2021, given_irrigation=True)
        env.reset(seed=0)
        farmer = env.core.visible_fields("farmer")
        equation = env.core.visible_fields("equation")
        self.assertIn("sat_et_mm", farmer)
        self.assertNotIn("sat_et_mm", equation)
        env.close()


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cyclic import soil
from cyclic.lateral import exchange_perched, exchange_perched_parts
from cyclic.physics import Physics, State, advance


def _line_edges():
    # Three paddies east-west, the middle one with a north neighbour.
    return SimpleNamespace(i=np.array([0, 1, 1]), j=np.array([1, 2, 3]),
                           dist_m=np.full(3, 50.0), ux=np.array([1.0, 1.0, 0.0]),
                           uy=np.array([0.0, 0.0, 1.0]))


class ExchangePartsTests(unittest.TestCase):
    def test_parts_match_net_and_split_by_axis(self):
        edges = _line_edges()
        perched = np.array([60.0, 20.0, 5.0, 5.0])
        area = np.array([1000.0, 1500.0, 1000.0, 800.0])
        p = Physics()
        net, out_x, out_y = exchange_perched_parts(perched, edges, p, k_x=500.0, k_y=50.0, area_m2=area)
        np.testing.assert_allclose(net, exchange_perched(perched, edges, p, 500.0, 50.0, area))
        self.assertAlmostEqual(float((net * area).sum()), 0.0, places=6)
        # Paddy 1 sends east along x and north along y; its N–S neighbour edge is y-only.
        self.assertGreater(out_x[1], 0.0)
        self.assertGreater(out_y[1], 0.0)
        self.assertEqual(out_y[0], 0.0)
        # What was sent equals what the receivers gained plus the senders' own net loss.
        sent = (out_x + out_y) * area
        received = (net * area) + sent
        self.assertAlmostEqual(float(received.sum()), float(sent.sum()), places=6)


class PanRiseTests(unittest.TestCase):
    def test_off_by_default(self):
        state = State.start(2, depth_mm=0.0, perched_mm=40.0)
        state.surface[:] = 20.0
        flux = advance(state, 0.0, 4.0, 0.5, np.zeros(2), np.zeros(2), Physics(), stress="supply")
        self.assertTrue(np.all(flux["rise_mm"] == 0.0))

    def test_rise_refills_a_dry_paddy_and_keeps_the_balance(self):
        p = Physics().replace(pan_capillary_rise=True)
        state = State.start(2, depth_mm=0.0, perched_mm=40.0)
        state.surface[:] = 20.0
        before = state.storage().copy()
        k_z = np.array([2.0, 30.0])
        flux = advance(state, 0.0, 4.0, 0.5, np.zeros(2), np.zeros(2), p, k_z=k_z, stress="supply")
        self.assertLessEqual(flux["rise_mm"][0], 2.0 + 1e-12)
        self.assertGreater(flux["rise_mm"][1], flux["rise_mm"][0])
        residual = (state.storage() - before) - (
            -flux["et_mm"] - flux["drainage_mm"] - flux["lateral_mm"] - flux["deep_mm"] + flux["neighbor_mm"])
        np.testing.assert_allclose(residual, 0.0, atol=1e-9)


class EsMdaTests(unittest.TestCase):
    def test_recovers_a_linear_gaussian_problem(self):
        rng = np.random.default_rng(3)
        G = rng.standard_normal((40, 4))
        truth = np.array([1.0, -0.5, 0.3, 2.0])
        sd = np.full(40, 0.05)
        d_obs = G @ truth + sd * rng.standard_normal(40)
        setup = SimpleNamespace(d_obs=d_obs, sd=sd)
        prior = rng.standard_normal((300, 4)) * 2.0
        with patch.object(soil, "forward", lambda x, _setup: G @ x):
            result = soil.es_mda(setup, prior, seed=1)
        np.testing.assert_allclose(result.ensemble.mean(0), truth, atol=0.05)
        self.assertLess(result.misfit[-1]["tables"] + result.misfit[-1]["et"],
                        result.misfit[0]["tables"] + result.misfit[0]["et"])


class FieldTests(unittest.TestCase):
    def test_prior_mean_is_the_physics_default_and_bounds_hold(self):
        n = 5
        setup = SimpleNamespace(features=np.zeros((n, len(soil.FEATURES))),
                                pilot=np.zeros((n, soil.N_PILOT)))
        k = soil.k_fields(soil.prior_mean(), setup)
        np.testing.assert_allclose(k["k_z"], Physics().plow_pan_k_mm_d)
        np.testing.assert_allclose(k["k_x"], Physics().lateral_k_mm_d)
        wild = soil.prior_mean() + 50.0
        logs = soil.log_k_fields(wild, SimpleNamespace(features=np.ones((n, len(soil.FEATURES))),
                                                       pilot=np.ones((n, soil.N_PILOT))))
        for f, (lo, hi) in soil.LOG_BOUNDS.items():
            self.assertTrue(np.all(logs[f] <= hi + 1e-12))


class ModelATests(unittest.TestCase):
    def test_old_five_head_weights_load_into_three_heads(self):
        from cyclic.et_net import N_OUT, SpillNet
        import torch

        old = SpillNet(hidden=8)
        state = old.state_dict()
        last = f"{len(old.net) - 1}"
        state[f"{last}.weight"] = torch.randn(5, 8)
        state[f"{last}.bias"] = torch.randn(5)
        net = SpillNet(hidden=8)
        net.load_state_dict(state)
        out = net.forward(torch.zeros(3, len(SpillNet.FEATURE_NAMES)))
        self.assertEqual(len(out), N_OUT)


if __name__ == "__main__":
    unittest.main()

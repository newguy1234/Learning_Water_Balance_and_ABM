import unittest
from types import SimpleNamespace

import numpy as np

from cyclic.agents import ETFollower, build_batch
from cyclic.model import CyclicModel
from cyclic.physics import Physics, State, advance
from cyclic.spatial import aggregate


class SpatialETTests(unittest.TestCase):
    def test_cell_fluxes_conserve_water_and_aggregate_volume(self):
        grid = SimpleNamespace(owner=np.array([0, 0, 1]), weights=np.array([0.5, 0.5, 1.0]))
        state = State.start(3, depth_mm=0)
        state.deficit[:] = [25, 70, 40]
        before = state.storage()
        irrigation = np.array([2., 2., 8.])
        flux = advance(state, 0., 5., np.array([0.2, 0.8, 0.5]), irrigation, 0., Physics())
        np.testing.assert_allclose(state.storage() - before,
                                   irrigation - flux['et_mm'] - flux['drainage_mm']
                                   - flux['lateral_mm'] - flux['deep_mm'], atol=1e-10)
        self.assertNotEqual(flux['et_mm'][0], flux['et_mm'][1])
        paddy_area = np.array([200., 500.])
        self.assertAlmostEqual(aggregate(flux['et_mm'], grid, 2) @ paddy_area,
                               flux['et_mm'] @ (paddy_area[grid.owner] * grid.weights))

    def test_default_pixel_run_and_repeatable_management(self):
        a, b = CyclicModel(seed=7, record_cells=True), CyclicModel(seed=7)
        a.step()
        b.step()
        self.assertEqual(a.inputs.potential, 'pixel')
        np.testing.assert_array_equal(a.last_irrigation, b.last_irrigation)
        self.assertGreater(np.ptp(a.last_irrigation), 0)
        np.testing.assert_allclose(aggregate(a.cell_et_mm[0], a.cells, a.n), a.log['et_mm'][0])
        self.assertLess(a.balance_error_mm, 1e-9)

    def test_explicit_uniform_daily_farmer(self):
        m = CyclicModel(potential='field')
        policy = ETFollower(farmer_variation=0, visit_interval_days=1)
        policy.reset(m, np.arange(m.n))
        batch = build_batch(m.inputs, 0, np.full(m.n, 10.), m.area)
        irrigation, _ = policy.decide_batch(batch)
        expected = np.clip(30 + policy.expected_et_mm(batch) + 11 - 10, 0, 80)
        np.testing.assert_allclose(irrigation, expected)


if __name__ == '__main__':
    unittest.main()

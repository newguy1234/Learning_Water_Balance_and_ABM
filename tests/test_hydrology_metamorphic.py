"""Physical relations independent of a fitted trajectory or observational dataset.

These verify implementation, not empirical validity. The boundary is the paddy
network: neighbour exchange is internal, lateral/deep flow leaves this boundary.
"""
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from cyclic.physics import Physics, State, advance


def scenario(seed):
    rng = np.random.default_rng(seed)
    n = 8
    state = State.start(n)
    state.water[:] = rng.uniform(0, 150, n)
    state.surface[:] = rng.uniform(0, 25, n)
    state.deficit[:] = rng.uniform(0, 100, n)
    state.perched[:] = rng.uniform(0, 80, n)
    # Disjoint pairs allow an isolation test without hidden connecting edges.
    edges = SimpleNamespace(i=np.array([0, 2, 4, 6]), j=np.array([1, 3, 5, 7]),
                            dist_m=np.full(4, 20.), ux=np.array([1., 0., 1., 0.]),
                            uy=np.array([0., 1., 0., 1.]))
    area = rng.uniform(100, 3000, n)
    return rng, state, edges, area


def copy_state(state, order):
    return State(**{f.name: getattr(state, f.name)[order].copy() for f in fields(State)})


@pytest.mark.parametrize('seed', [3, 17, 91])
@pytest.mark.parametrize('stress', ['root_zone', 'supply'])
@pytest.mark.parametrize('ratio_split', [False, True])
def test_daily_and_cumulative_network_water_balance(seed, stress, ratio_split):
    rng, state, edges, area = scenario(seed)
    p = Physics(ratio_split=ratio_split)
    initial_volume = state.storage() @ area / 1000
    cumulative_external = 0.
    for day in range(30):
        rain = np.full(8, 180. if day == 10 else (5. if day % 7 == 0 else 0.))
        irrigation = rng.uniform(0, 40, 8) if day < 15 else np.zeros(8)
        before = state.storage().copy()
        flux = advance(state, rain, 7., rng.uniform(0, 1, 8), irrigation,
                       rng.uniform(0, 20, 8), p, edges=edges, area_m2=area,
                       stress=stress, weir_mm=np.linspace(20, 100, 8))
        external = rain + irrigation - flux['et_mm'] - flux['drainage_mm'] \
            - flux['lateral_mm'] - flux['deep_mm']
        np.testing.assert_allclose(state.storage() - before,
                                   external + flux['neighbor_mm'], rtol=0, atol=1e-10)
        assert abs(flux['neighbor_mm'] @ area / 1000) < 1e-10
        cumulative_external += external @ area / 1000
        assert abs(state.storage() @ area / 1000 - initial_volume - cumulative_external) < 1e-8
        assert np.isfinite(state.storage()).all()
        assert (state.water >= -1e-10).all() and (state.perched >= -1e-10).all()
        assert (state.surface >= -1e-10).all() and (state.surface <= p.total_evaporable_mm + 1e-10).all()
        assert (state.deficit >= -1e-10).all() and (state.deficit <= p.total_available_mm + 1e-10).all()


@pytest.mark.parametrize('stress', ['root_zone', 'supply'])
def test_relabelling_paddies_preserves_states_and_fluxes(stress):
    rng, original, edges, area = scenario(17)
    order = np.array([5, 2, 7, 0, 3, 1, 6, 4])
    inverse = np.argsort(order)
    relabelled = copy_state(original, order)
    remapped = SimpleNamespace(i=inverse[edges.i], j=inverse[edges.j],
                              dist_m=edges.dist_m, ux=edges.ux, uy=edges.uy)
    p = Physics()
    for _ in range(12):
        rain, et0, canopy, irrigation, discharge = [rng.uniform(0, hi, 8)
                                                   for hi in (15, 8, 1, 30, 10)]
        kz, kx, ky = [rng.uniform(1, hi, 8) for hi in (15, 2000, 500)]
        args = (rain, et0, canopy, irrigation, discharge)
        a = advance(original, *args, p, k_z=kz, k_x=kx, k_y=ky,
                    edges=edges, area_m2=area, stress=stress)
        b = advance(relabelled, *(x[order] for x in args), p,
                    k_z=kz[order], k_x=kx[order], k_y=ky[order],
                    edges=remapped, area_m2=area[order], stress=stress)
        for key in a:
            np.testing.assert_allclose(b[key], a[key][order], rtol=0, atol=1e-10, err_msg=key)
        for f in fields(State):
            np.testing.assert_allclose(getattr(relabelled, f.name),
                                       getattr(original, f.name)[order], rtol=0, atol=1e-10)


def test_closed_network_redistributes_without_creating_water():
    _, state, edges, area = scenario(91)
    p = Physics(outlet_loss_per_day=0, bund_mm=1000,
                deep_percolation_mm_d=0, farm_drain_stage_mm=0)
    before = state.storage() @ area / 1000
    initial_perched = state.perched.copy()
    for _ in range(20):
        flux = advance(state, 0., 0., .5, 0., 0., p, edges=edges, area_m2=area)
        for key in ('et_mm', 'drainage_mm', 'lateral_mm', 'deep_mm'):
            np.testing.assert_allclose(flux[key], 0., rtol=0, atol=1e-12)
        assert abs(state.storage() @ area / 1000 - before) < 1e-9
    assert not np.allclose(state.perched, initial_perched)


def test_disconnected_paddies_do_not_respond_to_other_components_rain():
    _, a, edges, area = scenario(3)
    b = copy_state(a, np.arange(8))
    # Modest conductivity avoids triggering a network-wide positivity limiter.
    p = Physics(lateral_k_mm_d=20., deep_percolation_mm_d=0)
    for _ in range(10):
        rain = np.zeros(8)
        rain[:2] = 60.
        fa = advance(a, 0., 5., .5, 0., 0., p, edges=edges, area_m2=area)
        fb = advance(b, rain, 5., .5, 0., 0., p, edges=edges, area_m2=area)
        for key in fa:
            np.testing.assert_allclose(fa[key][2:], fb[key][2:], rtol=0, atol=1e-10)
        for f in fields(State):
            np.testing.assert_allclose(getattr(a, f.name)[2:], getattr(b, f.name)[2:],
                                       rtol=0, atol=1e-10)
    assert not np.allclose(a.storage()[:2], b.storage()[:2])

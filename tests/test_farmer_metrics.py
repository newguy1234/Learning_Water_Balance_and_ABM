import numpy as np
import pytest
from abm.model import DistrictModel
from abm.world import SeasonConfig
from abm.metrics import farmer_performance, farmer_type_performance, best_farmers


def test_farmer_metrics_aggregate_and_reset():
    m=DistrictModel(SeasonConfig(days=3, paddies_per_farmer=3))
    assert farmer_performance(m).empty
    for _ in range(3): m.step()
    farms=farmer_performance(m)
    assert len(farms)==m.world.n_farmers()
    assert (farms.days==3).all()
    assert farms.score.is_monotonic_decreasing
    assert len(best_farmers(m,2))==2
    assert (farms.relative_yield_pct.between(0,100)).all()
    for row in farmer_type_performance(m).itertuples():
        individual=farms[farms.farmer_type==row.farmer_type]
        assert row.irrigation_mm==pytest.approx(individual.irrigation_mm.mean())
        assert row.score==pytest.approx(individual.score.mean())
    row=farms.iloc[0]
    ids=m.world.paddies_of(row.farmer_id)
    expected=sum(np.average(h.irrigation_mm[ids],weights=m.world.area_m2[ids]) for h in m.world.history)
    assert row.irrigation_mm==pytest.approx(expected)
    m.reset()
    assert farmer_performance(m).empty


def test_roster_change_preserves_historical_labels():
    m=DistrictModel(SeasonConfig(days=2,paddies_per_farmer=3),agent_roster=['steward'])
    m.step()
    m.set_roster(['guardian'])
    m.step()
    rows=farmer_performance(m)
    assert set(rows.farmer_type)=={'steward','guardian'}
    assert (rows.days==1).all()


def test_mixed_holdings_filters_and_recombination():
    m=DistrictModel(SeasonConfig(days=2,paddies_per_farmer=4002),agent_roster=['steward'])
    for _ in range(2): m.step()
    all_farms=farmer_performance(m)
    cyclic=farmer_performance(m,statuses=['Cyclic'])
    noncyclic=farmer_performance(m,statuses=['Non-cyclic'])
    assert all_farms.iloc[0].days == cyclic.iloc[0].days == noncyclic.iloc[0].days == 2
    total=all_farms.iloc[0]; c=cyclic.iloc[0]; n=noncyclic.iloc[0]
    assert total.area_ha == pytest.approx(c.area_ha+n.area_ha)
    assert total.irrigation_mm == pytest.approx((c.irrigation_mm*c.area_ha+n.irrigation_mm*n.area_ha)/total.area_ha)
    assert total.copying_events == c.copying_events == n.copying_events
    north=farmer_performance(m,statuses=['Non-cyclic'],blocks=['North'],farmer_types=['steward'])
    mask=(~m.world.cyclic)&(m.world.block=='North')
    assert north.iloc[0].paddies == mask.sum()
    expected=sum(np.average(h.irrigation_mm[mask],weights=m.world.area_m2[mask]) for h in m.world.history)
    assert north.iloc[0].irrigation_mm == pytest.approx(expected)
    assert farmer_performance(m,statuses=[]).empty
    assert farmer_performance(m,farmer_types=['guardian']).empty


def test_alternative_ranking():
    m=DistrictModel(SeasonConfig(days=2,paddies_per_farmer=3))
    m.step()
    assert best_farmers(m,rank_by='irrigation_mm').irrigation_mm.is_monotonic_increasing
    assert farmer_type_performance(m,rank_by='relative_yield_pct').relative_yield_pct.is_monotonic_decreasing


def test_block_field_maps_show_only_that_area():
    import numpy as np
    from abm.plots import figure_field
    from abm.world import SeasonConfig, WaterWorld

    world = WaterWorld(SeasonConfig(days=1))
    for block in ("North", "South"):
        fig = figure_field(world, mode="ponding", block=block)
        ax = fig.axes[0]
        assert ax.get_title().startswith(f"{block} field")
        m = np.asarray(world.block) == block
        lo, hi = ax.get_ylim()
        assert lo <= world.lat[m].min() and hi >= world.lat[m].max()
    # The two areas together cover the district, with no overlap.
    assert int((np.asarray(world.block) == "North").sum() + (np.asarray(world.block) == "South").sum()) == world.n

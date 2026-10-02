# cyclic-irrigation (CyclicEnv)

**Gymnasium package** for leftover water-fate modelling under cyclic irrigation  
(Imbanuma / Shiroyama–Jinbee; Okura et al. 2025).

**Target journal:** [*Environmental Modelling & Software*](https://www.sciencedirect.com/journal/environmental-modelling-and-software)  
Manuscript draft: [`outputs/paper/ems_research_article.md`](outputs/paper/ems_research_article.md)  
Reproduce every table and figure: `python scripts/ems_experiments.py`

## Install

```bash
# Recommended for the EMS software paper (Gymnasium + spatial + Copernicus):
pip install -e ".[ems]"
# Or explicitly:
pip install -e ".[gym,copernicus]"
# Everything including Solara UI:
pip install -e ".[all]"
```

The `[gym]` extra includes Gymnasium **and** the spatial stack (geopandas / xarray) required to reset `Cyclic-Imbanuma-v0`.

A frozen EMS physics bundle (leftover capture / neighbour / stay + efficiency pins) ships under `cyclic/bundled/` so a fresh `pip install` does not need to re-run discovery. Rebuild after experiments with `python -m cyclic.bundle rebuild`.

## Agent-based models (`abm/`)

Conceptual twin gym on the **cyclic** satellite paddy map, run on cyclic's own
paddy hydrology. LID staff send pool water; farmers apply ≤ send; FAO-33 Ya/Ym.

Every day each paddy goes through `cyclic.physics.advance`: FAO-56 dual-Kc ET
(canopy from the crop stage, since scenario seasons have no satellite NDVI), the
farmer's discharge + bund overflow + outlet leak (D), plow-pan percolation in
proportion to ponded depth, root zone, perched layer, Darcy exchange with
neighbouring paddies, Hooghoudt lateral drain flow and deep percolation. Seepage
is therefore a continuous loss from every flooded paddy, as Okura et al. (2025)
measured (Lf ≈ 7–11 mm/d). The discovered per-paddy recapture share
(`pool_capture_share`, mean ≈ 0.76 = Table 5 LFV_pump / ΣLf·A) sends that lateral
flow on cyclic paddies to the pool; the rest is block storage ΔWB (Eq. 7).

The district is the Hakusan-Jimbee (白山甚兵衛) pumping station's 983 ha service
area: 4,002 MAFF Fude parcels whose membership, North/South block and cyclic
status are read from Okura et al. Fig. 1a (`cyclic/infrastructure.py`,
`tools/build_infrastructure.py`). The pool sits at the station, the main
drainage canal is the real channel along the lake dyke, and each cyclic paddy's
drainage reaches the pool along the mapped channels (OpenStreetMap), taking
`SeasonConfig.drain_velocity_m_s` to get there. A paddy's recaptured share of
lateral flow depends on its distance to the nearest channel that reaches the pool.
Backwater: when the main canal runs high (daily depths from Okura Fig. 3,
`SeasonConfig.canal_depth`), low cyclic paddies (GSI 5 m DEM) cannot drain and the
water stays on the field. The station delivers at most its pumps' 3.304 m³/s
(`SeasonConfig.pump_capacity_m3_s`). Sources and remaining assumptions:
`cyclic/bundled/SOURCES.md`.

The pool is the real 18 × 68 m flow-through buffer (`Physics.pool_capacity_m3`).
Routed cyclic drainage and recaptured lateral flow are pumped straight back
out; the lake (a boundary, never exhausted) makes up the rest through the
station's gate, and anything above capacity spills to the lake (O). When the
station's send exceeds pool + gate + expected returns, paddies nearest the
pumping station (`cyclic.data.POOL_LONLAT`) are served in full first. Open-Meteo
rain is scaled to the on-site gauge's monthly totals (Table 2; ~30–50 % wetter
otherwise). Ponding is capped at a bund height (`SeasonConfig.bund_height_mm`,
default 250 mm; IRRI guidance, Chapagain & Yamaji 2010,
[Paddy and Water Environment](https://link.springer.com/article/10.1007/s10333-012-0324-4)).
`SeasonConfig.weather_variability` (0 = exact historical replay) jitters rain/ET0
day-to-day for scenario seasons. `abm.metrics.okura_comparison(world)` puts a
2021/2022 season beside Okura Tables 3–6.

Farmers and the station are real `mesa.Agent`s on a `mesa.Model`
(`abm.model.DistrictModel`) — one `FarmerMesaAgent` per farmer, one
`StationMesaAgent`, both thin wrappers around the registered policy classes
below; the physics (cyclic.physics paddy day, recapture, pool accounting) stays vectorized
on `WaterWorld`, the same split `cyclic/model.py` uses for `CyclicModel` +
`PaddyAgent`/`PumpingStation`. `abm.FieldSession` is the UI/CLI-facing wrapper
around `DistrictModel`.

```bash
pip install -e ".[abm]"
python -m abm ui           # hand-built Solara dashboard: map, run controls, paper metrics
python -m abm mesa-ui       # the same DistrictModel through Mesa's own SolaraViz shell
python example.py abm      # ABM UI with the farmer + station registered in example.py
```

```python
@abm.register_station("my_staff")
class MyStaff(abm.StationAgent):
    def decide(self, obs):
        return {"withdraw_m3": ..., "outflow_m3": ...,
                "irrigation_sent_mm": ...}  # length n_paddies

@abm.register_farmer("my_farmer")
class MyFarmer(abm.FarmerAgent):
    def decide_owned(self, obs):
        # apply ≤ obs["owned_irrigation_sent_mm"]; discharge from stock
        return apply_mm, drainage_mm
```

### ABM gym environment + optimising staff baseline

```python
from abm.gym_env import AbmEnv

env = AbmEnv(year=2022, farmers={"steward": 0.6, "opportunist": 0.4}, staff="mpc")
obs, info = env.reset(seed=0)
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step()   # or step(your_action)
    done = terminated or truncated
info["season"]   # Okura WUE' / RUE / LUE, W, O, yield, stress
```

Default staff everywhere (`FieldSession`, `DistrictModel`, `run_season`, `AbmEnv`,
both UIs): the latest station you register yourself with `@abm.register_station`
(e.g. `my_station` in `example.py`), otherwise `mpc`
(`abm.station.default_station()`). `mpc` (`abm.optimise.MPCStation`) is the
baseline optimiser: each day it
plans, with the pond physics and a 3-day rain/ET forecast, the least water per
paddy that keeps it stress-free tomorrow; uses returning water (pool + cyclic D +
recaptured lateral flow) before the lake; opens the lake gate only for the
shortfall; banks surplus returns in cyclic ponds instead of spilling; pumps to
the lake only ahead of flood rain. Pass an action to `step()` to drive the staff
role yourself (`{"withdraw_m3", "outflow_m3", "irrigation_sent_mm"}` or a flat
`[W, O, send_0 … send_n]`). Reward = −(W + O + 20·stress volume) / 10^3 m3.
Weather scenarios: `AbmEnv(weather="drought")` or
`SeasonConfig(weather=...)` with a preset from `abm.weather.PRESETS` (baseline,
drought, wet, heatwave, storm, dry_july, erratic) or a tweaked
`abm.weather.WeatherScenario` (season/monthly rain and ET0 scaling, dry spells,
heatwaves, storms, day-to-day variability), e.g.
`weather_scenario("drought", heatwaves=[(80, 10, 1.3)])`.
Note WUE' rises when losses fall while RUE and LUE (returns / irrigation) rise
when more water passes through, so they cannot all be maximised at once.

## Gymnasium (recommended)

```python
import gymnasium as gym
import cyclic.gym_env   # registers Cyclic-Imbanuma-v0

env = gym.make("Cyclic-Imbanuma-v0", year=2022, given_irrigation=True)
obs, info = env.reset(seed=0)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

| Env id | Meaning |
|--------|---------|
| `Cyclic-Imbanuma-v0` | Okura IR known; action = discharge |
| `Cyclic-Imbanuma-Open-v0` | IR + D both actions |

Multi-role observations: `info["obs_farmer"]`, `info["obs_equation"]`, `info["obs_ratio"]`.

Hide / show channels:

```python
env.unwrapped.set_obs_channels("equation", hide=("yesterday_fate",))
env.unwrapped.set_obs_channels("farmer", hide=("satellite",))
```

## Register your own learners

```python
import cyclic

@cyclic.register_forecaster("my_farmer")
class MyFarmer(cyclic.FarmerPolicy):
    def decide_batch(self, obs):
        return irrigation_mm, discharge_mm

@cyclic.register_ratio("my_ratios")
class MyRatios(cyclic.RatioPolicy):
    def partition(self, leftover, obs, targets):
        return shares

@cyclic.register_lateral("my_hops")
class MyHops(cyclic.LateralPolicy):
    def flux(self, stock, graph, area, physics, neighbour_et_mm=None):
        ...
```

## Copernicus (Sentinel-2 over Imbanuma)

1. Create a free account at https://dataspace.copernicus.eu/  
2. Register credentials with this package:

```bash
python -m cyclic copernicus register
python -m cyclic copernicus download 2022 --max-cloud 40
```

## Contribution (science baseline)

Surface IR/D are hard-copied forcing. Inferred underground leftover ratios:

```python
import cyclic
print(cyclic.contribution_text(cyclic.underground_ratio_train_both()))
```

## CLI

```bash
python -m cyclic copernicus status
python -m cyclic paper 2022
python -m cyclic bench
python example.py          # cyclic UI with the forecaster + ratio finder registered in example.py
python example.py abm      # ABM UI with the farmer + station registered in example.py
```

### Per-paddy satellite and groundwater handoff to the ABM

```python
import cyclic

ds = cyclic.load_paddy_dataset(2022)  # shipped NetCDF, loaded into memory
path = cyclic.export_paddy_dataset(2022, "paddy_abm_2022.nc")  # rebuild from local sources
```

The installed package includes `cyclic/bundled/paddy_abm_2021.nc` and
`paddy_abm_2022.nc`. Importing `cyclic` exposes these APIs without writing files
or downloading data. Install `cyclic-irrigation[abm]` for the ABM and spatial stack.
Each dataset joins satellite NDVI/Kc/flood observations to stable `paddy` IDs,
parcel geometry, area, block, cyclic status, canal/pool distances, cardinal
neighbours, daily weather, fallback shares and fitted groundwater coefficients.
Satellite composites use `time`; daily weather uses `day`. Groundwater feature
names use `fate_feature`; `leftover_mm` and `infiltrated_mm` are runtime inputs,
not fixed satellite observations. The ABM currently evaluates the equation with
its daily positive leftover and zero infiltration (the evaluator default).

The ABM reads the bundled contract by default. To consume another export:

```python
from cyclic.paddy_export import load_cyclic_landscape
from abm.world import WaterWorld, SeasonConfig
world = WaterWorld(SeasonConfig(year=2022),
                   landscape=load_cyclic_landscape(2022, dataset_path="paddy_abm_2022.nc"))
```

Ownership is a reproducible shuffle across parcels, controlled by the scenario
seed and holding-size settings. It does not alter cyclic status. Cyclic status
remains the package's inferred canal/pool layout scaled to Table 1, not a measured
satellite classification. The export records equation provenance and any change
between training and current cyclic flags. Non-cyclic drainage goes to the lake;
only cyclic drainage and capture return to the pool. Both categories retain the
existing shared station supply policy. Regenerate the export explicitly after
changing source data or refitting equations.

### Adaptive farmer personalities

The ABM's old `fixed`, `rule`, `efficiency`, `noise` and example-only farmer
policies have been replaced by the seven archetypes in `Conceptual_Model.pdf`,
Section 7, Tables 5-6 (pages 10-12): `maximiser`, `survivalist`, `steward`,
`guardian`, `adaptive_experimenter`, `passive_operator`, and `opportunist`.
The example launcher and both UIs use these by default; station policies remain
separate. These names replace the previous ABM names in scripts and saved rosters.

**Collaboration and the neighbour water level:** every farmer carries a
`cooperation` level (0–1; archetypes from their traits, other farmers from
params, default 0.5) and a `neighbour_wetness_bias` (−1 drier … +1 wetter than its
neighbours; archetype bases in `abm.agents.ARCHETYPES`). Each paddy's
target drifts softly by
`cooperation·(neighbour level − target) + (1 − cooperation)·bias·15 mm`:
collaborative farmers level their water with the (up to 4) neighbouring paddies,
uncollaborative ones hold their own preferred level. Weight 0.3, tanh-capped at
20 mm, neighbour level remembered at 30 %/day, so targets move at most ±6 mm.
`SeasonConfig(collaborative_share=0.7)` / `AbmEnv(collaborative_share=0.7)`
makes that share of all farmers (any type, seeded choice) collaborative (0.95)
and the rest uncollaborative (0.05); cooperation still evolves through learning
afterwards. Custom farmers use it via `self.neighbour_target(obs, target)`
(see `my_farmer` in `example.py`).

Each farmer has 12 persistent, seed-reproducible personality traits
(`abm.farmer.TRAITS`): `experience`, `water_accounting`, `cooperation`,
`work_ethic`, `risk_tolerance`, `neighbour_wetness_bias`, `calendar_lead`,
`drawdown_tolerance`, `refill_overshoot`, `off_farm_work`, `forecast_reliance`
and `flow_through_habit`, plus descriptive age. Each trait's `meaning` says what
it does in the decision rules; the **Trait guide** tab of the sensitivity viewer
(`python scripts/abm_sensitivity.py --view`) explains every one in plain words
with its tested effect on demand. Earlier names (`part_time`, `overfill`,
`knowledge` …) are still accepted (`abm.farmer.RENAMED_TRAITS`). `wealth` and
`price_sensitivity` were removed (they only acted on a volumetric water price,
which land-improvement districts do not charge), and `labour` was merged into
`work_ethic` (the check interval only used the two together). Age does not
prescribe a behaviour. Between monitoring visits, farmers retain
inlet/outlet settings. Farmers learn from water scarcity and crop stress, and can
copy the previous day's inlet/outlet settings and target strategy of a better
performing neighbouring owner. Neighbours are owners of adjacent paddies,
excluding self and duplicates. Every farmer reads the same frozen daily snapshot;
there is no same-day learning cascade from iteration order. Reset clears memory
and reconstructs the same initial population for the same seed.

**Implementation assumptions:** the document provides qualitative archetypes,
not fitted numerical parameters. `ARCHETYPES` in `abm/agents.py` contains editable
scenario defaults. Learning uses an observable score (crop stress coefficient
minus irrigation and drainage penalties), not measured profit or a validated
causal yield effect. Fertiliser, water pricing and economic incentives are
not simulated. The opportunist responds to immediately available water. The
adaptive experimenter explores bounded variations in irrigation demand; it does
not invoke a separate optimisation service. `learning_rate=0` disables adaptation
and exploration; `social_influence=0` disables copying. Individual traits,
`memory`, `copied_from`, and `copy_events` are accessible on each farmer policy;
the Run UI reports aggregate copying events and cooperation.

### Farmer performance tables

The Metrics tab shows equal-weight averages per farmer personality and the top
10 individual farm periods. Within a farm, depths and stress are area-weighted.
The ranking uses the mean daily social-learning score:
`Ks - 0.15 * irrigation_mm / 40 - 0.05 * drainage_mm / 30` (higher is better).
Tables include estimated relative yield to date, mean Ks, cumulative irrigation,
drainage and pool capture, percentage of station supply used, copying events,
and cooperation. This is a model score, not profit or measured yield. A zero
supply denominator displays as missing, not infinite efficiency. A change in
ownership or personality starts a new cohort without relabelling past results;
compare periods with similar lengths. Reset clears the records.

Programmatic access: `abm.metrics.farmer_performance(model)`,
`farmer_type_performance(model)`, and `best_farmers(model, limit=10)` return
pandas tables suitable for CSV export.

Farmer metrics can also be filtered by `statuses=['Cyclic', 'Non-cyclic']`,
`blocks=['North', 'South']`, and `farmer_types=['steward', 'guardian']` in any
combination. Omit a filter to include all; an empty API filter selects no rows.
The UI's empty personality selection means all types. Choose `rank_by` from
`score`, `relative_yield_pct`, `mean_ks`, `irrigation_mm`, `drainage_mm`, or
`capture_mm`; irrigation and drainage rank lowest first, the others highest first.
Daily records preserve paddy-status/block strata, which are area-weighted back
into a single farm-day for combined selections. A mixed holding contributes only
its cyclic parcels to the cyclic leaderboard and only its non-cyclic parcels to
the non-cyclic leaderboard. Copying events and cooperation are farmer-level
context repeated in split views; do not add them across status leaderboards.

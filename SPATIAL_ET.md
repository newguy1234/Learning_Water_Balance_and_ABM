ET now runs at the satellite source-cell scale by default (`potential="pixel"`).
Scoring update: absolute NDVI-times-Open-Meteo ET is not independent actual-ET
validation. Its default training weight is now zero. Irrigation and discharge
retain weight one each. The fly-over diagnostic now compares relative spatial
patterns (each map normalized by its own mean); Open-Meteo mean agreement is
excluded. Earlier numerical ET fidelity results use the old metric and are not
directly comparable. Weather ET0 still forces the physics and informs farmer
expectations; removing it from the reward does not remove that model uncertainty.
An independent actual-ET product or validated energy-balance retrieval is still
needed before claiming satellite-based absolute ET accuracy.
Each cell has its own canopy and four water stores. Farmers still act on paddies:
the requested irrigation depth is applied to each cell in the paddy, and cell
fluxes are averaged back to paddy depths before calculating station volumes.
Cells represent equal shares of their paddy's modelled area; polygon-edge
intersection fractions and within-paddy redistribution are not modelled.

Both 2021 and 2022 now have nominal 10 m Sentinel-2 source imagery (stored on
the same geographic grid, approximately 11 m north-south). The 2022 file was
imported from `Main_Show/Project/Satellite/paddy_et_s2_2022.nc`: 23 acquisition
dates from May 4 to September 29. Invalid NDVI outside [-1, 1] was masked
(0.66% of finite source values). Cloud gaps remain missing; the model
interpolates canopy between valid observations. Before the first observation,
the existing model uses the first available canopy, not an April measurement.
The older 330 m input remains available as a fallback. Paddies containing no source-cell centre use the nearest
cell and retain a fallback flag. No random ET texture or artificial finer grid
is generated. Canopy between composites is retrospectively interpolated;
farmer observations retain their existing delayed imagery visibility.

The base `et_follower` now has persistent, seeded differences in target depth
(default +/-15%), drainage calendar (default +/-3 days), and visit phase
(default once every three days). These are adjustable scenario assumptions,
not measured farmer traits. Custom and trained policies keep their own decision
rules. Set `farmer_variation=0, visit_interval_days=1` for the previous uniform
daily policy, and `potential="field"` for block-average canopy physics.

ET can still be equal in some cells: the existing dual-coefficient model caps
wet-condition ET, and nearby cells can have the same canopy and water status.
Changing those physical coefficients requires calibration, not arbitrary noise.

The field animation offers absolute ET and deviations from each map's
area-weighted mean, with fixed seasonal colour limits. The satellite panel
holds the latest available estimate, so its pixels can refer to different
composite dates. Satellite ET is derived from NDVI and reference ET, not a
direct flux measurement. Source resolution is displayed on the map.

`results.pixel_series(paddy_id)` exports daily cell ET, source coordinates,
represented area, satellite estimates on composite dates, and fallback flags.
Restart the app and rerun seasons to replace previously computed results.
Paddy caches are checked against the chosen source file and parcel IDs, so a
new high-resolution source invalidates a previously cached coarse input.

Scoring retains irrigation and total discharge (drainage) against observed
cyclic-block volumes. The daily training reward mixes irrigation, discharge,
and ET with default weights 1:1:1; satellite days increase the ET weight to 2.
Season water-volume loss penalizes both season totals and daily timing for
irrigation and discharge. The comparison table reports irrigation fidelity,
discharge fidelity, combined water-volume fidelity/loss, and satellite ET
fidelity separately. Discharge observations describe total outflow, not just
the farmer-controlled outlet action.

Training and final evaluation use distinct years. The CLI defaults to training
on 2021 and evaluating on 2022; use `--train-year` and `--evaluation-year` to
reverse them. Equal years and training observations from another year are
rejected. In the app, the selected page year is the evaluation year and the
other available year is used for training. All three built-in learners select
candidates using training data only and evaluate a deep copy without learning
hooks. `LearnResult.score` and `.results` are held-out evaluation results;
`.notes['training_score']` and `.history` contain training diagnostics.
Saved policies carry their training years. Agents already trained on the
evaluation year, and legacy saved agents with unknown training history, cannot
be used as a training baseline for that split; start from a registered agent.

Reward training now replays each end-of-episode snapshot with learning off on
the training year before selecting it. The online season's score cannot rank
an end-of-season policy that did not generate those fluxes. The best frozen
checkpoint and its separate irrigation/discharge/ET scores are saved under
`outputs/checkpoints/<run timestamp>/` after each improvement, with history
updated every episode. The app shows these best-so-far training diagnostics.
The selected checkpoint minimizes the existing irrigation/discharge loss;
a better ET score cannot replace a checkpoint with a worse water-flow loss.

Output viewing is separate from training/evaluation selection. Use "Output year"
in Run & compare, or "View learned policy on year" after training, to run a copy
of the fixed policy on 2021 or 2022 with learning off. Training-year and unknown-
provenance outputs are explicitly labelled diagnostics. Outputs from different
years remain available in the field animation and paddy views; viewing them
does not replace the learner's held-out score or alter its saved policy.

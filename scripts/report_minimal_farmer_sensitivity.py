from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/minimal_farmer_sensitivity'
def report():
 d=pd.read_csv(OUT/'runs.csv');s=pd.read_csv(OUT/'summary.csv');r=pd.read_csv(OUT/'reduction.csv');inter=pd.read_csv(OUT/'interactions.csv')
 labels={'target':'Target depth','rate':'Adaptation','social':'Social learning'}
 fig,axs=plt.subplots(1,2,figsize=(12,5),sharey=False)
 for ax,metric,title in zip(axs,['demand_m3','applied_m3'],['Requested water','Applied water']):
  for j,(sc,name,col) in enumerate([('historical','Nominal pumps','#087F8C'),('pump25','25% pump capacity','#B76D25')]):
   sub=s[(s.scenario==sc)&(s.metric==metric)].set_index('factor').loc[list(labels)]
   x=np.arange(3)+(j-.5)*.34;means=sub.mean_percent.to_numpy()
   ax.bar(x,means,width=.32,color=col,label=name)
   ax.errorbar(x,means,yerr=np.array([means-sub.min_percent.to_numpy(),sub.max_percent.to_numpy()-means]),fmt='none',ecolor='#333333',capsize=3)
  ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(np.arange(3),list(labels.values()));ax.set_title(title);ax.set_ylabel('High minus low / centre (%)');ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=9)
 fig.suptitle('Current rule-based ABM: factorial screening of three controls')
 fig.text(.5,.01,'60 seasons; 4,002 paddies; 2022 forcing; three paired seeds. Whiskers show seed range, not confidence intervals.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(OUT/'sensitivity.png',dpi=180);plt.close(fig)
 def fmt(v):return f'{v:+.2f}'
 lines=['# A minimal farmer model for the Imbanuma PhD project','',
 '**Recommendation: start with target water depth as the single behavioural dimension. Treat adaptation as a separate optional mechanism and social learning as an ablation, rather than restoring twelve personality traits or seven named archetypes.**',
 '', 'This is a recommendation for the next model comparison, not evidence that one dimension explains real farmers. The LLM should face the same small decision problem as an explicit-rule baseline and must earn its additional complexity on independent observations.',
 '', '## What changed between the manuscript and current code','',
 'The supplied conceptual-model PDF describes twelve traits and seven archetypes. The current `abm/farmer.py` has an empty `TRAITS` tuple and daily checking. Seven named classes still exist, differing mainly in target multiplier, adaptation rate, social influence and evidence threshold. The experiment leaves that code unchanged and uses one class (`steward`) throughout. It screens three existing controls, with evidence threshold fixed at 0.03. It does not implement or benchmark LLM agents.',
 '', '## Atlas evidence and what it does not establish','',
 '| Candidate | Evidence | Boundary of support |', '|---|---|---|',
 '| Target water depth | [Anan et al. (2004)](https://doi.org/10.1007/s10333-004-0035-6) studies empirical water management, intake and ponding depth in Japanese paddies. | Supports an operational depth preference. Our 0.85–1.15 range is a scenario assumption, not an estimated personality distribution. Publisher abstract inspected. |',
 '| Adaptation to outcomes | [Apetrei et al. (2024)](https://doi.org/10.1016/j.ecolmodel.2023.110609) explicitly models learning-by-doing and social learning. | Supports testing learning mechanisms. It does not calibrate our daily Imbanuma learning rate or scarcity equation. Full text inspected from the local Atlas PDF. |',
 '| Social learning | Apetrei et al.; [Du et al. (2025)](https://doi.org/10.3390/su17083468) finds spatial associations in AWD adoption in Shiga. | Adoption association is not proof of daily copying, a causal effect, or a numeric copying coefficient. Du abstract inspected through Atlas metadata. |',
 '| Labour constraints | [Yagi and Hayashi (2021)](https://doi.org/10.22434/IFAMR2020.0013) describes weather-dependent labour scheduling in Japanese rice corporations. | Does not establish a universal field-check interval. Hold daily checking fixed for this screen; revisit only if field observations justify it. Atlas abstract inspected. |',
 '', 'Atlas returned 21 unique records from three app searches. Because its cross-field CLI adds computer-science terms, four additional targeted queries used Atlas’s OpenAlex provider directly without those terms (five records per query, not necessarily unique). Search metadata is retained in `outputs/farmer_traits_review/minimal_traits_atlas.json` and `minimal_traits_targeted.json`. Four selected evidence records are in `evidence.csv`. Broad search results were screened for relevance, not treated as equally supporting evidence.',
 '', '## Sensitivity design','',
 'The full 4,002-paddy district ran for 131 days per experiment using 2022 weather and the existing MPC station. There were 60 runs: eight factorial corners, one centre, and one depth-only baseline, repeated for three explicitly fixed ownership seeds and two pump capacities. Nominal capacity is 3.304 m³/s; the shortage experiment uses 25% of that. Initial target-depth offsets have assumed SD 0.05 and are held identical within paired runs. Physical parameters, crop calendar, daily checking, holds and station policy remain fixed.',
 '', '| Control | Low | High | Centre |', '|---|---:|---:|---:|', '| Target multiplier | 0.85 | 1.15 | 1.00 |', '| Adaptation rate | 0.00 | 0.20 | 0.10 |', '| Social multiplier | 0.00 | 0.60 | 0.30 |',
 '', 'A main effect is the high-minus-low outcome averaged across the other two controls, calculated separately for each scenario and seed. Percentages divide by the matching centre outcome. Means and minimum–maximum seed effects follow below. This is two-level factorial screening with interactions, not a Sobol analysis, calibration or a statistical estimate of farmer populations.',
 '', '## Computed effects','',
 '| Pump setting | Control | Demand change (%) | Applied change (%) | Peak 7-day demand (%) | Lake intake (%) |', '|---|---|---:|---:|---:|---:|']
 for sc in ['historical','pump25']:
  for f in labels:
   z=s[(s.scenario==sc)&(s.factor==f)].set_index('metric')
   vals=[f"{z.loc[m,'mean_percent']:+.2f} [{z.loc[m,'min_percent']:+.2f}, {z.loc[m,'max_percent']:+.2f}]" for m in ['demand_m3','applied_m3','peak7_m3','lake_m3']]
   lines.append('| '+('Nominal' if sc=='historical' else '25% capacity')+' | '+labels[f]+' | '+' | '.join(vals)+' |')
 lines += ['', '![Factorial effects](sensitivity.png)', '',
 'Values in brackets are the range across three paired seeds, not confidence intervals. A large demand effect with a small application effect means the delivery constraint is suppressing the behavioural difference. It does not show equivalent changes in withdrawals or water savings.',
 '', '## Adaptation and social influence are coupled','',
 'In the current code, the copying step is `learning_rate × social_influence × (neighbour_target − own_target)`, conditional on a score advantage above the evidence threshold. Thus social influence has exactly zero effect when learning rate is zero. The experiment checks this invariance for both target levels, all seeds and both capacity scenarios. The mean social main effect includes these zero-rate cases, so it can understate the effect among adaptive agents. The conditional contrasts below therefore restrict adaptation to 0.2.',
 '', '| Pump setting | Social effect on demand (%) at rate 0.2 | Social effect on application (%) at rate 0.2 |', '|---|---:|---:|']
 for sc in ['historical','pump25']:
  vals=[]
  for metric in ['demand_m3','applied_m3']:
   effects=[]
   for seed,g in d[d.scenario==sc].groupby('seed'):
    f=g[g.label.str.startswith('f') & (g.rate==.2)];den=g[g.label=='centre'][metric].iloc[0]
    effects.append(100*(f[f.social==.6][metric].mean()-f[f.social==0][metric].mean())/den)
   vals.append(f'{np.mean(effects):+.3f}')
  lines.append('| '+sc+' | '+' | '.join(vals)+' |')
 lines += ['', 'Adaptation also changes the target by `rate × ((1 − previous Ks) − 0.2 × scarcity)` and clips it to 0.65–1.5. Scarcity is defined from sent/requested water, so the current controller can produce this feedback even when ET stress is zero. Large effects may reflect controller-policy mismatch or repeated drift to a bound. They should not be described as measured learning, risk tolerance or cooperation.',
 '', '## Can the simpler model replace the centre configuration?','',
 'The depth-only baseline keeps the same target distribution but turns adaptation and social learning off. Changes below are relative to the centre configuration; this is a model-reduction comparison, not an accuracy comparison against observations.',
 '', '| Pump setting | Demand (%) | Applied (%) | Peak (%) | Lake (%) | Unmet ET (mm) | Yield (percentage points) |','|---|---:|---:|---:|---:|---:|---:|']
 for sc,g in r.groupby('scenario'):
  avg=g.groupby('metric')[['effect','percent']].mean()
  vals=[fmt(avg.loc[m,'percent']) for m in ['demand_m3','applied_m3','peak7_m3','lake_m3']]
  vals += [fmt(avg.loc['stress_mm','effect']),fmt(100*avg.loc['yield_fraction','effect'])]
  lines.append('| '+sc+' | '+' | '.join(vals)+' |')
 lines += ['', 'Screening tolerances were documented as 5% for demand/application/peak/lake intake, 5 mm stress, and 1 percentage point of the yield proxy. These are review triggers, not validated agronomic acceptance thresholds. If simplification exceeds a trigger, retain the affected mechanism as a comparison until observational validation resolves it. Sensitivity alone cannot decide which version is more accurate.',
 '', '## Minimal LLM specification','',
 '1. A fixed profile describes desired wetness relative to crop stage. Avoid moral labels such as maximiser or steward in the model comparison.',
 '2. Supply crop stage, pond/soil summaries, water delivered, available forecasts and prior outcomes as state. Keep hard constraints outside the prompt.',
 '3. Return a compact JSON action: target depth, refill permission and drain mode. Deterministic code converts it to per-field actions and enforces supply, storage and no-drain limits.',
 '4. Compare a fixed profile with an optional adaptation instruction. Keep social observation off in the minimal baseline and test it as a controlled ablation.',
 '5. Fit a small statement vocabulary on development data, then freeze it. Compare with direct optimisation of the same simple numerical policy on an unseen season. Satellite similarity alone cannot identify farmer personality or ownership.',
 '', '## Limits and reproducibility','',
 '- This is a conditional screen of one weather year, one station policy and one physical parameterisation. Three seeds cover ownership and initial target heterogeneity, not all relevant uncertainties.',
 '- The 25% pump scenario isolates capacity; it does not include a new drought climate. Future weather and physical ensembles may change the ranking.',
 '- The source PDF describes a different model version. Its reported 2021/2022 errors and twelve-trait sensitivities must not be mixed with these new results or attributed to LLMs.',
 '- `runs.csv` contains all 60 outcomes. `main_effects.csv`, `interactions.csv`, `reduction.csv` and `summary.csv` contain the calculations. `design.json` records all jobs and hashes of the core model source files.',
 '- Scripts: `scripts/minimal_farmer_sensitivity.py`, `scripts/analyse_minimal_farmer_sensitivity.py`, `scripts/report_minimal_farmer_sensitivity.py`. Run with the existing `.venv-atlas` interpreter and `outputs/abm_testing_review/runtime` dependencies. The simulation script resumes completed jobs; use a new output directory if changing the design or source.',
 '- All runs check finite metrics and application ≤ station delivery. The analysis checks the zero-learning/social invariance. No production farmer defaults were changed.',
 '- A preliminary isolated full season took 26.5 seconds on this host. Experiment wall times include three concurrent processes and are not an LLM speed benchmark.',
 '', 'The revised presentation uses the supplied conceptual-model figures and labels LLM components as proposed. It removes the fixed funding timetable and the twelve-trait personality catalogue.']
 (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
 print('Report and figure written')
if __name__=='__main__':report()

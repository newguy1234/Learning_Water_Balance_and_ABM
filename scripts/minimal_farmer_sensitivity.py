"""Reproducible full-district screening of three existing behavioural controls.
Production ABM defaults are not changed. All ranges are modelling assumptions.
"""
from pathlib import Path
import sys, os, json, time, itertools, hashlib, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'outputs/abm_testing_review/runtime'))
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
OUT=ROOT/'outputs/minimal_farmer_sensitivity';OUT.mkdir(exist_ok=True)

def run(job):
 import numpy as np
 from abm.model import DistrictModel
 from abm.world import SeasonConfig
 warnings.filterwarnings('ignore',category=FutureWarning)
 label,target,rate,social,scenario,seed=job
 start=time.perf_counter()
 cfg=SeasonConfig(year=2022,days=131,seed=seed,ownership_seed=seed,paddies_per_farmer=50,
                  pump_capacity_m3_s=3.304 if scenario=='historical' else 3.304*.25)
 m=DistrictModel(cfg,agent_roster=['steward'],station_name='mpc',seed=seed)
 rng=np.random.default_rng(seed+10000)
 # Identical farmer offsets across paired runs, including reduced baselines.
 for fid in sorted(m.farmer_agents):
  initial=float(np.clip(target+rng.normal(0,.05),.65,1.5))
  m.set_farmer_agent(fid,'steward',target_multiplier=initial,learning_rate=rate,
                     social_influence=social,evidence_threshold=.03,trait_variability=0,
                     schedule_jitter=0,week_swap=0)
 for _ in range(131):m.step()
 w=m.world;a=w.area_m2; total=a.sum()
 volume=lambda name:np.array([float((getattr(d,name)*a/1000).sum()) for d in w.history])
 req=volume('requested_mm');ir=volume('irrigation_mm');dr=volume('drainage_mm')
 result=dict(label=label,target=target,rate=rate,social=social,scenario=scenario,seed=seed,
   n_paddies=w.n,n_farmers=len(m.farmer_agents),demand_m3=float(req.sum()),
   applied_m3=float(ir.sum()),peak7_m3=float(np.convolve(req,np.ones(7),'valid').max()),
   drainage_m3=float(dr.sum()),lake_m3=float(sum(d.withdraw_m3 for d in w.history)),
   stress_mm=float(sum(np.average(d.stress_mm,weights=a) for d in w.history)),
   yield_fraction=float(np.average(w.relative_yield,weights=a)),
   copies=sum(f.policy.copy_events for f in m.farmer_agents.values()),
   mean_final_target=float(np.mean([f.policy.target_multiplier for f in m.farmer_agents.values()])),
   seconds=time.perf_counter()-start)
 assert all(np.isfinite(v) for v in result.values() if isinstance(v,(float,int)))
 assert all(np.all(d.irrigation_mm <= d.irrigation_sent_mm+1e-8) for d in w.history)
 (OUT/f'{scenario}_{seed}_{label}.json').write_text(json.dumps(result,indent=2))
 return result

def main():
 factors=[(f'f{i}',*v) for i,v in enumerate(itertools.product([.85,1.15],[0,.2],[0,.6]))]
 factors += [('centre',1.,.1,.3),('depth_only',1.,0.,0.)]
 jobs=[(*v,sc,seed) for sc in ['historical','pump25'] for seed in [11,29,47] for v in factors]
 sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'abm/farmer.py',ROOT/'abm/agents.py',ROOT/'abm/world.py',ROOT/'cyclic/physics.py',ROOT/'abm/model.py']}
 (OUT/'design.json').write_text(json.dumps(dict(jobs=jobs,source_hashes=sources,year=2022,days=131,
   note='Two-level full factorial plus centre and reduced baseline. Three paired ownership seeds. Conditional screening, not empirical validation or Sobol indices.'),indent=2))
 rows=[]
 with ProcessPoolExecutor(max_workers=3) as ex:
  futures={ex.submit(run,j):j for j in jobs if not (OUT/f'{j[-2]}_{j[-1]}_{j[0]}.json').exists()}
  for f in as_completed(futures):
   r=f.result();print(r['scenario'],r['seed'],r['label'],round(r['seconds'],1),flush=True)
 for j in jobs:rows.append(json.loads((OUT/f'{j[-2]}_{j[-1]}_{j[0]}.json').read_text()))
 import pandas as pd
 pd.DataFrame(rows).to_csv(OUT/'runs.csv',index=False)
 print('COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()

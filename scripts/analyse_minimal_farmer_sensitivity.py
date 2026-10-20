from pathlib import Path
import itertools,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/minimal_farmer_sensitivity'
metrics=['demand_m3','applied_m3','peak7_m3','lake_m3','drainage_m3','stress_mm','yield_fraction']
factors=['target','rate','social']
def analyse():
 d=pd.read_csv(OUT/'runs.csv');rows=[];pairs=[];reduce=[]
 assert len(d)==60 and not d.duplicated(['scenario','seed','label']).any()
 for (sc,seed),g in d.groupby(['scenario','seed']):
  f=g[g.label.str.startswith('f')];c=g[g.label=='centre'].iloc[0];b=g[g.label=='depth_only'].iloc[0]
  for x,y in [('f0','f1'),('f4','f5')]:
   np.testing.assert_allclose(g[g.label==x][metrics].to_numpy(),g[g.label==y][metrics].to_numpy(),rtol=0,atol=1e-8)
  for metric in metrics:
   denom=c[metric]
   for name in factors:
    low,high=sorted(f[name].unique());effect=f.loc[f[name]==high,metric].mean()-f.loc[f[name]==low,metric].mean()
    rows.append(dict(scenario=sc,seed=seed,factor=name,metric=metric,effect=effect,percent=100*effect/denom if abs(denom)>1e-8 else np.nan))
   for a,bb in itertools.combinations(factors,2):
    a0,a1=sorted(f[a].unique());b0,b1=sorted(f[bb].unique())
    mean=lambda aa,bbb:f.loc[(f[a]==aa)&(f[bb]==bbb),metric].mean()
    effect=mean(a1,b1)-mean(a0,b1)-mean(a1,b0)+mean(a0,b0)
    pairs.append(dict(scenario=sc,seed=seed,pair=a+' x '+bb,metric=metric,effect=effect,percent=100*effect/denom if abs(denom)>1e-8 else np.nan))
   effect=b[metric]-c[metric]
   reduce.append(dict(scenario=sc,seed=seed,metric=metric,centre=denom,reduced=b[metric],effect=effect,percent=100*effect/denom if abs(denom)>1e-8 else np.nan))
 for name,data in [('main_effects',rows),('interactions',pairs),('reduction',reduce)]:pd.DataFrame(data).to_csv(OUT/(name+'.csv'),index=False)
 e=pd.DataFrame(rows);summary=e.groupby(['scenario','factor','metric']).agg(mean_effect=('effect','mean'),mean_percent=('percent','mean'),min_percent=('percent','min'),max_percent=('percent','max')).reset_index();summary.to_csv(OUT/'summary.csv',index=False)
 payload=dict(main=summary.to_dict('records'),reduction=pd.DataFrame(reduce).groupby(['scenario','metric'])[['effect','percent']].mean().reset_index().to_dict('records'),runtime_median=float(d.seconds.median()),runs=len(d),invariance_pass=True)
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2,allow_nan=True))
 print(summary[summary.metric.isin(['demand_m3','applied_m3','peak7_m3','lake_m3'])].to_string(index=False))
 print('Reduction',pd.DataFrame(reduce).groupby(['scenario','metric'])['percent'].mean().to_string())
if __name__=='__main__':analyse()

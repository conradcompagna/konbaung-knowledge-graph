from pathlib import Path
import json, sys, os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import pandas as pd
from scipy import stats
O=Path(__file__).resolve().parents[1]
RESULTS=O/'results'; RESULTS.mkdir(exist_ok=True)
SEED=20260911
CAT=json.loads((O/'recovered_source/data/axial_category_catalog.json').read_text(encoding='utf-8'))
LABEL={x['id']:x['label'] for mode in ['entities','relations'] for x in CAT[mode]}
E=sorted(x['id'] for x in CAT['entities']); R=sorted(x['id'] for x in CAT['relations'])
def load(): return pd.read_csv(O/'inputs/analysis_triples.csv.gz',keep_default_na=False)
def save(df,name):
 df.to_csv(RESULTS/(name+'.csv'),index=False,encoding='utf-8-sig')
def jsave(obj,name):
 (RESULTS/(name+'.json')).write_text(json.dumps(obj,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
def bh(p):
 p=np.asarray(p,float);out=np.full(p.shape,np.nan); valid=np.flatnonzero(np.isfinite(p));ix=valid[np.argsort(p[valid])];n=len(ix)
 if n:out[ix]=np.minimum(1,np.minimum.accumulate((p[ix]*n/np.arange(1,n+1))[::-1])[::-1])
 return out
def unit(a):
 a=np.asarray(a,float);return np.divide(a,np.linalg.norm(a,axis=-1,keepdims=True),out=np.zeros_like(a),where=np.linalg.norm(a,axis=-1,keepdims=True)>0)
def tensor(d,weights=None):
 s=d.S.map(dict(zip(E,range(len(E))))).to_numpy();p=d.P.map(dict(zip(R,range(len(R))))).to_numpy();o=d.O.map(dict(zip(E,range(len(E))))).to_numpy()
 return np.bincount((s*len(R)+p)*len(E)+o,weights=weights,minlength=len(E)*len(R)*len(E)).reshape(len(E),len(R),len(E)).astype(float)
def permutation_summary(obs,null):
 null=np.asarray(null,float)
 if not np.isfinite(obs) or not np.all(np.isfinite(null)):
  return dict(observed=float(obs) if np.isfinite(obs) else None,null_mean=None,null_sd=None,z=None,p_greater=None,p_two_sided=None,draws=len(null),status='undefined statistic; no p-value',finite_null_draws=int(np.isfinite(null).sum()))
 mu=null.mean();sd=null.std(ddof=1)
 # Both tails around the null, including observed in empirical reference.
 plo=(1+(null<=obs).sum())/(len(null)+1);phi=(1+(null>=obs).sum())/(len(null)+1)
 return dict(observed=float(obs),null_mean=float(mu),null_sd=float(sd),z=float((obs-mu)/sd) if sd else None,p_greater=float(phi),p_two_sided=float(min(1,2*min(plo,phi))),draws=len(null))
def pretty(x): return LABEL.get(x,x)

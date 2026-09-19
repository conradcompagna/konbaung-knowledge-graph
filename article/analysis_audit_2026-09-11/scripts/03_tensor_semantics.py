"""Nonnegative multiway factors and semantic/structural QAP comparisons."""
from common import *
sys.path.insert(0,str(O/'vendor'))
import tensorly as tl
from tensorly.decomposition import non_negative_parafac_hals, tucker
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

def cpfit(a,rank,seed,iters=300):
 cp,errors=non_negative_parafac_hals(a,rank=rank,n_iter_max=iters,init='random',random_state=seed,tol=1e-7,return_errors=True,exact=False)
 return cp,dict(iterations=len(errors),relative_error=float(errors[-1]),last_error_change=float(abs(errors[-1]-errors[-2])) if len(errors)>1 else None,seed=seed,rank=rank)
def factors_table(cp,modes,prefix):
 rows=[]
 for mode,(name,ids) in enumerate(modes):
  F=np.asarray(cp.factors[mode]);F=F/np.maximum(F.sum(0),1e-30)
  for k in range(F.shape[1]):
   for j,code in enumerate(ids):rows.append(dict(mode=name,component=k+1,code=code,label=LABEL.get(code,code),loading=F[j,k]))
 save(pd.DataFrame(rows),prefix+'_loadings')

d=load();x=tensor(d);rng=np.random.default_rng(SEED+3)
# Split by pages within volumes, never by individual claims.
hold=set()
for vol,g in d.groupby('volume'):
 pages=np.array(sorted(g.page_key.unique()));hold.update(rng.choice(pages,size=max(1,round(.2*len(pages))),replace=False))
train=d[~d.page_key.isin(hold)];test=d[d.page_key.isin(hold)];xt=tensor(train);xv=tensor(test);fits=[];best=None
save(pd.DataFrame({'page_key':sorted(d.page_key.unique()),'test_page':[p in hold for p in sorted(d.page_key.unique())]}),'03_page_split')
for rank in [3,5,7]:
 for seed in [SEED,SEED+1]:
  cp,meta=cpfit(np.sqrt(xt),rank,seed)
  rates=np.maximum(tl.cp_to_tensor(cp)**2,1e-10);prob=rates/rates.sum();nll=float(-(xv*np.log(prob)).sum()/xv.sum())
  meta.update(validation_nll_per_claim=nll,fit='sqrt count CP, score squared reconstruction');fits.append(meta)
  print('CP',meta,flush=True)
  if best is None or nll<best[0]:best=(nll,rank,seed)
ind=xt.sum((1,2))[:,None,None]*xt.sum((0,2))[None,:,None]*xt.sum((0,1))[None,None,:]+.5
ind=ind/ind.sum();fits.append(dict(rank=1,seed=None,fit='marginal independence baseline',validation_nll_per_claim=float(-(xv*np.log(ind)).sum()/xv.sum())))
save(pd.DataFrame(fits),'03_tensor_model_selection')
cp,meta=cpfit(np.sqrt(x),best[1],best[2],600);meta.update(selected_rank=best[1],train_claims=len(train),test_claims=len(test),selection='minimum held-out owner-page NLL among ranks 3,5,7 and two seeds; descriptive model, not calibrated missing-fact probabilities')
factors_table(cp,[('subject',E),('relation',R),('object',E)],'03_static_cp');jsave(meta,'03_static_cp_fit')
np.savez_compressed(RESULTS/'03_static_cp.npz',weights=cp.weights,S=cp.factors[0],P=cp.factors[1],O=cp.factors[2])
# HOSVD/Tucker is a separate, signed higher-order decomposition diagnostic.
tk=tucker(np.sqrt(x),rank=[8,8,8],init='svd',n_iter_max=50,tol=1e-6)
jsave({'ranks':[8,8,8],'relative_frobenius_error':float(np.linalg.norm(np.sqrt(x)-tl.tucker_to_tensor(tk))/np.linalg.norm(np.sqrt(x)))},'03_tucker_fit')
np.savez_compressed(RESULTS/'03_tucker.npz',core=tk.core,S=tk.factors[0],P=tk.factors[1],O=tk.factors[2])
# Reign is an ordered narrative segment, not event time. Normalize by claim exposure.
chron=pd.read_csv(O/'inputs/page_chronology.csv');order=chron.reign.drop_duplicates().tolist();reigns=[r for r in order if r in set(d.reign_segment)]
reigns += [r for r in d.reign_segment.unique() if r not in reigns]
z=np.stack([tensor(g) for r in reigns for g in [d[d.reign_segment==r]]],axis=3)
exposure=z.sum((0,1,2));zn=np.sqrt(z/exposure[None,None,None,:]*1000)
cp4,meta4=cpfit(zn,5,SEED,250);factors_table(cp4,[('subject',E),('relation',R),('object',E),('reign',reigns)],'03_reign_cp')
meta4.update(reign_order=reigns,claims_per_reign=exposure.tolist(),transformation='sqrt occurrences per 1000 claims within reign; all 12 narrative segments including succession crises',time_semantics='reign-linked textual composition, not durations or dated events')
jsave(meta4,'03_reign_cp_fit');np.savez_compressed(RESULTS/'03_reign_cp.npz',weights=cp4.weights,S=cp4.factors[0],P=cp4.factors[1],O=cp4.factors[2],T=cp4.factors[3],reigns=reigns)
# Relation semantics vs structural endpoint distributions; three embedding views.
structure=unit(np.sqrt(x.transpose(1,0,2).reshape(81,-1)));similarity=structure@structure.T
mask=np.triu(np.ones((81,81),bool),1);y=similarity[mask]
recs=[json.loads(l) for l in (O/'recovered_source/embeddings/relation_records.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(recs)==11886
for rec in recs:
 found=d[d.prow==rec['index']]
 assert not len(found) or set(found.p)=={rec['tag']}
W=np.zeros((81,11886));np.add.at(W,(d.P.map(dict(zip(R,range(81)))).to_numpy(),d.prow.to_numpy()),1)
views={'base':'embeddings/relation_base_vectors.npy','context':'similarity/features/relation_context_vectors.npy','fused':'similarity/features/relation_fused_vectors.npy'}
results=[];pair_table=pd.DataFrame({'r1':[R[i] for i,j in zip(*np.where(mask))],'r2':[R[j] for i,j in zip(*np.where(mask))],'structural_cosine':y})
domain=pd.read_csv(O/'inputs/domain_map.csv');print('Domain map columns',domain.columns.tolist(),flush=True)
# Domain stratification if explicit map is present, otherwise use prefix-free unstratified null only.
idcol=next(c for c in domain if c in ['relation_id','rid','id']);domcol=next(c for c in domain if 'domain' in c and c!=idcol)
dom=dict(zip(domain[idcol],domain[domcol]));strata=[np.array([i for i,r in enumerate(R) if dom.get(r)==v]) for v in set(dom.values())]
sets=[set(d.loc[d.P==r,'p']) for r in R];overlap=np.zeros((81,81))
for i in range(81):
 for j in range(81):overlap[i,j]=len(sets[i]&sets[j])/max(1,len(sets[i]|sets[j]))
freq=np.log1p(W.sum(1));freqdist=np.abs(freq[:,None]-freq[None,:]);cov=np.column_stack([np.ones(mask.sum()),freqdist[mask],overlap[mask]])
for view,path in views.items():
 V=np.load(O/'recovered_source'/path,mmap_mode='r');assert V.shape[0]==11886
 # Centering removes the claim-weighted common embedding direction.
 cent=W@V/W.sum(1,keepdims=True);mean=np.average(cent,weights=W.sum(1),axis=0);cent=unit(cent-mean);sem=cent@cent.T;v=sem[mask]
 pair_table[view+'_centered_cosine']=v
 for mode in ['unrestricted','within_relation_domain']:
  obs=np.corrcoef(v,y)[0,1];null=[]
  for b in range(1999):
   perm=rng.permutation(81)
   if mode=='within_relation_domain':
    perm=np.arange(81)
    for ix in strata:perm[ix]=rng.permutation(ix)
   null.append(float(np.corrcoef(v,similarity[np.ix_(perm,perm)][mask])[0,1]))
  results.append(dict(view=view,null_model=mode,**permutation_summary(obs,null)))
 # Partial correlation controlling log-frequency difference and raw-tag overlap,
 # with Freedman-Lane QAP of reduced-model residuals, refitted each draw.
 fit0=cov@np.linalg.lstsq(cov,y,rcond=None)[0];res=y-fit0;resM=np.zeros((81,81));resM[mask]=res;resM+=resM.T
 X=np.column_stack([cov,v]);pinv=np.linalg.pinv(X);observed=(pinv@y)[-1];null=[]
 for b in range(1999):
  perm=rng.permutation(81);yp=fit0+resM[np.ix_(perm,perm)][mask];null.append((pinv@yp)[-1])
 results.append(dict(view=view,null_model='Freedman_Lane_QAP_frequency_and_shared_predicates',**permutation_summary(observed,null)))
results=pd.DataFrame(results);results['q_two_sided']=bh(results.p_two_sided);save(results,'03_semantic_structure_qap');save(pair_table,'03_semantic_structure_pairs')
print('Tensor and semantics completed',flush=True)

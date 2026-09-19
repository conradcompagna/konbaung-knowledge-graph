from common import *
from scipy.spatial.distance import jensenshannon
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,str(O/'vendor'))
import tensorly as tl
from tensorly.decomposition import non_negative_parafac_hals

d=load();rng=np.random.default_rng(SEED+6);focal=pd.read_csv(RESULTS/'02_layer_qap.csv')[['r1','r2']].drop_duplicates().values.tolist()
# Distinguish a shared royal endpoint from coupling distributed across categories.
rows=[]
for specification,codes,z in [('without_sovereign',[e for e in E if e!='E01'],d),('without_sovereign_or_status_object',[e for e in E if e not in ['E01','E30']],d),('without_registers',E,d[d.register.isin(['','NaN'])])]:
 x=tensor(z).transpose(1,0,2);idx=np.array([E.index(c) for c in codes]);x=x[:,idx][:,:,idx];n=len(idx);mask=~np.eye(n,dtype=bool)
 strata=[np.array([i for i,e in enumerate(codes) if ((int(e[1:])<=23 or int(e[1:])>=46) if k==0 else (24<=int(e[1:])<=29) if k==1 else (30<=int(e[1:])<=43) if k==2 else (44<=int(e[1:])<=45))]) for k in range(4)]
 for r1,r2 in focal:
  a=np.log1p(x[R.index(r1)]);b=np.log1p(x[R.index(r2)]);obs=np.corrcoef(a[mask],b[mask])[0,1];null=[]
  for j in range(1999):
   perm=np.arange(n)
   for ix in strata:perm[ix]=rng.permutation(ix)
   null.append(np.corrcoef(a[mask],b[np.ix_(perm,perm)][mask])[0,1])
  rows.append(dict(specification=specification,r1=r1,r2=r2,**permutation_summary(obs,null)))
rows=pd.DataFrame(rows);rows['q_two_sided']=rows.groupby('specification').p_two_sided.transform(bh);save(rows,'06_layer_qap_sensitivity')
# Resample owner pages within volume. Confidence intervals describe page-sampling
# variability in this corpus, not a random sample of Konbaung political life.
page_ids=sorted(d.page_key.unique());pmap=dict(zip(page_ids,range(len(page_ids))));pg=d.page_key.map(pmap).to_numpy();s=d.S.map(dict(zip(E,range(52)))).to_numpy();o=d.O.map(dict(zip(E,range(52)))).to_numpy();r=d.P.map(dict(zip(R,range(81)))).to_numpy()
flat=(r*52+s)*52+o;grouped=[np.array([pmap[p] for p in g.page_key.unique()]) for v,g in d.groupby('volume')];mask=~np.eye(52,dtype=bool);boot=[]
for draw in range(999):
 w=np.zeros(len(page_ids))
 for ids in grouped:w+=np.bincount(rng.choice(ids,len(ids),replace=True),minlength=len(page_ids))
 a=np.bincount(flat,weights=w[pg],minlength=81*52*52).reshape(81,52,52)
 out={'draw':draw}
 for r1,r2 in focal:out[r1+'_'+r2]=np.corrcoef(np.log1p(a[R.index(r1)])[mask],np.log1p(a[R.index(r2)])[mask])[0,1]
 boot.append(out)
b=pd.DataFrame(boot);save(b,'06_page_bootstrap_layer_draws');save(pd.DataFrame([dict(r1=r1,r2=r2,lower95=b[r1+'_'+r2].quantile(.025),upper95=b[r1+'_'+r2].quantile(.975),median=b[r1+'_'+r2].median()) for r1,r2 in focal]),'06_layer_bootstrap_intervals')
# CP seed and register sensitivity: align components across fits, never labels.
base=np.load(RESULTS/'03_static_cp.npz');ref=[base['S'],base['P'],base['O']];ref=[unit(v.T) for v in ref];stability=[];loads=[]
specs=[('alternate_seed',d,7,SEED+10),('without_registers',d[d.register.isin(['','NaN'])],7,SEED),('without_title_and_identity',d[~d.P.isin(['R12','R64'])],5,SEED)]
for name,z,rank,seed in specs:
 cp,errors=non_negative_parafac_hals(np.sqrt(tensor(z)),rank=rank,n_iter_max=500,init='random',random_state=seed,tol=1e-7,return_errors=True)
 similarity=np.ones((7,rank))
 for rf,F in zip(ref,cp.factors):similarity*=rf@unit(np.asarray(F).T).T
 aidx,bidx=linear_sum_assignment(-similarity)
 for i,j in zip(aidx,bidx):stability.append(dict(specification=name,reference_component=i+1,new_component=j+1,multimode_cosine_product=similarity[i,j],relative_error=float(errors[-1]),iterations=len(errors)))
 for mode,codes,F in zip(['subject','relation','object'],[E,R,E],cp.factors):
  F=F/np.maximum(F.sum(0),1e-30)
  for j in range(rank):
   for i,c in enumerate(codes):loads.append(dict(specification=name,mode=mode,component=j+1,code=c,label=LABEL[c],loading=F[i,j]))
save(pd.DataFrame(stability),'06_tensor_component_stability');save(pd.DataFrame(loads),'06_tensor_sensitivity_loadings')
# Adjacent reign segments: whole S-P-O composition, page-block permutation.
order=['Alaungpaya','Naungdawgyi','Hsinbyushin','Singu','1782 succession crisis','Bodawpaya','Bagyidaw','1837 succession war','Tharrawaddy','Pagan Min','Mindon','Thibaw'];comp=[]
for ra,rb in zip(order[:-1],order[1:]):
 z=d[d.reign_segment.isin([ra,rb])].copy();pages=z[['page_key','volume','reign_segment']].drop_duplicates();ambig=pages.page_key.duplicated(keep=False);excluded=pages.loc[ambig,'page_key'].unique();pages=pages[~ambig].reset_index(drop=True);z=z[~z.page_key.isin(excluded)]
 pi=dict(zip(pages.page_key,range(len(pages))));coord=((z.S.map(dict(zip(E,range(52))))*81+z.P.map(dict(zip(R,range(81)))))*52+z.O.map(dict(zip(E,range(52))))).to_numpy();mat=csr_matrix((np.ones(len(z)),(z.page_key.map(pi),coord)),shape=(len(pages),52*81*52));lab=(pages.reign_segment==ra).to_numpy();strata=list(pages.groupby('volume').indices.values())
 def stat(l):
  a=np.asarray(mat[l].sum(0)).ravel();b=np.asarray(mat[~l].sum(0)).ravel();return jensenshannon(a,b,base=2)**2
 obs=stat(lab);null=[]
 movable=any(len(set(lab[ix]))>1 for ix in strata)
 if movable:
  for j in range(999):
   lp=lab.copy()
   for ix in strata:lp[ix]=rng.permutation(lab[ix])
   null.append(stat(lp))
  row=permutation_summary(obs,null)
 else:row=dict(observed=obs,p_greater=None,p_two_sided=None,draws=0)
 comp.append(dict(reign1=ra,reign2=rb,pages1=int(lab.sum()),pages2=int((~lab).sum()),mixed_pages_excluded=len(excluded),null_exchangeable=movable,**row))
comp=pd.DataFrame(comp);comp['q_greater']=bh(comp.p_greater);save(comp,'06_adjacent_reign_tensor_comparison')
jsave(dict(bootstrap_draws=999,bootstrap_unit='owner page, sampled within volume',QAP_sensitivity_draws=1999,
 comparisons='Adjacent reign segments compare composition, not evolution of an observed social risk set. Reign labels permuted at page level within volume; no historical cause inferred.',unusable_for_historical_event_models=['No verified occurrence timestamps for every triple','Within-sentence triple order is extraction order, not event order','Page order includes retrospective registers','No complete office spell onset/exit/censoring ledger or population at risk']), '06_scope')
print('Robustness and comparisons completed',flush=True)

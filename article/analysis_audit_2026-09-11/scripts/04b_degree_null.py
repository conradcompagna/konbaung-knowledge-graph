"""Uniform-proposal directed degree MCMC with edge switches and cycle reversal.

Rejections count as transitions. Fixed proposal counts avoid accepted-swap sampling
bias. Directed cycle reversal supplements double-edge switches for irreducibility.
"""
from common import *
import networkx as nx
d=load();A=tensor(d).sum(1)>0;np.fill_diagonal(A,False);g=nx.from_numpy_array(A,create_using=nx.DiGraph);obs=nx.triadic_census(g);draw=[];diags=[]
for chain in [0,1]:
 rng=np.random.default_rng(SEED+40+chain);b=A.copy();edges=np.argwhere(b).tolist();index={tuple(e):i for i,e in enumerate(edges)};m=len(edges);accepted=0;cycleaccepted=0;proposals=0
 for it in range(500):
  steps=50*m if it==0 else 3*m
  for aidx,bidx in rng.integers(m,size=(steps,2)):
   proposals+=1
   if aidx==bidx:continue
   u,v=edges[aidx];x,y=edges[bidx]
   if len({u,v,x,y})!=4 or b[u,y] or b[x,v]:continue
   b[u,v]=b[x,y]=False;b[u,y]=b[x,v]=True
   del index[(u,v)];del index[(x,y)];edges[aidx]=[u,y];edges[bidx]=[x,v];index[(u,y)]=aidx;index[(x,v)]=bidx;accepted+=1
  for u,v,w in rng.integers(52,size=(max(1,steps//10),3)):
   if len({u,v,w})!=3:continue
   if b[u,v] and b[v,w] and b[w,u] and not b[v,u] and not b[w,v] and not b[u,w]:
    old=[(u,v),(v,w),(w,u)];new=[(v,u),(w,v),(u,w)];ix=[index[e] for e in old]
    for e in old:b[e]=False;del index[e]
    for j,e in zip(ix,new):b[e]=True;edges[j]=list(e);index[e]=j
    cycleaccepted+=1
  assert np.array_equal(A.sum(0),b.sum(0)) and np.array_equal(A.sum(1),b.sum(1)) and not np.diag(b).any()
  row=nx.triadic_census(nx.from_numpy_array(b,create_using=nx.DiGraph));row.update(chain=chain,draw=it);draw.append(row)
  if it%100==0:print('degree null',chain,it,flush=True)
 diags.append(dict(chain=chain,double_switch_proposals=proposals,accepted_switches=accepted,accepted_cycle_reversals=cycleaccepted))
df=pd.DataFrame(draw);save(df,'04_degree_preserving_triad_draws');rr=[]
for k,v in obs.items():rr.append(dict(triad=k,**permutation_summary(v,df[k])))
rr=pd.DataFrame(rr);rr['q_two_sided']=bh(rr.p_two_sided);save(rr,'04_degree_preserving_triads')
jsave(dict(draws=1000,chains=2,diagnostics=diags,degree_invariants_verified=True,lag1_autocorrelations={k:[df[df.chain==c][k].autocorr() if df[df.chain==c][k].std()>0 else None for c in [0,1]] for k in obs},
 chain_means={k:[df[df.chain==c][k].mean() for c in [0,1]] for k in obs},method='fixed-proposal double-edge switches plus directed triangle reversal; symmetric proposals and rejection self-loops. 50*m burn-in plus 3*m switch proposals and 0.3*m cycle proposals per saved draw. Approximate MCMC, no guarantee of complete mixing.'),'04_degree_null_diagnostics')

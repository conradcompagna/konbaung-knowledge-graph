from common import *
from collections import defaultdict,Counter
from itertools import product
d=load();edges=d[d.s!=d.o].drop_duplicates(['page_key','s','P','o']).reset_index(drop=True)
scope=json.loads((RESULTS/'04_scope.json').read_text());F=list(scope['families']);families={k:set(v) for k,v in scope['families'].items() if isinstance(v,list)};K=len(F)
fm={r:next((i for i,k in enumerate(F) if r in families.get(k,set())),K-1) for r in R};labels=edges.P.map(fm).to_numpy();pages=edges.page_key.astype('category').cat.codes.to_numpy(dtype=np.int64)
chains=[];ances=[]
focal={('information','authorization'),('allegiance','authorization'),('resistance','coercion'),('authorization','execution')}
for page,g in edges.groupby('page_key'):
 outgoing=defaultdict(list)
 for t in g.itertuples():outgoing[t.s].append(t)
 for a in g.itertuples():
  for b in outgoing.get(a.o,[]):
   if b.o==a.s:continue
   chains.append((a.Index,b.Index));key=(F[fm[a.P]],F[fm[b.P]])
   if key in focal:ances.append(dict(pattern=' -> '.join(key),page_key=page,subject=a.s,intermediate=a.o,object=b.o,predicate1=a.p,predicate2=b.p,relation1=a.P,relation2=b.P,sentence1=a.sentence_id,sentence2=b.sentence_id,triple1=a.triple_id,triple2=b.triple_id,intermediate_category=a.O,same_sentence=a.sentence_id==b.sentence_id))
ar=np.asarray(chains);page=pages[ar[:,0]]
def count(lab,binary=False):
 code=lab[ar[:,0]]*K+lab[ar[:,1]]
 if binary:code=np.unique(page*K*K+code)%(K*K)
 return np.bincount(code,minlength=K*K)
obs=count(labels);obsp=count(labels,True);rng=np.random.default_rng(SEED+9);groups=[np.asarray(g) for g in edges.groupby(['page_key','S','O']).indices.values() if len(g)>1]
null=np.zeros((1999,K*K),int);nullp=np.zeros_like(null)
for b in range(1999):
 l=labels.copy()
 for ix in groups:l[ix]=rng.permutation(labels[ix])
 null[b]=count(l);nullp[b]=count(l,True)
out=[]
for j,(a,b) in enumerate(product(F,repeat=2)):
 for mode,ob,draws in [('path_count',obs,null),('page_presence',obsp,nullp)]:out.append(dict(pattern=a+' -> '+b,unit=mode,**permutation_summary(ob[j],draws[:,j])))
q=pd.DataFrame(out);q['q_greater']=q.groupby('unit').p_greater.transform(bh);q['q_two_sided']=q.groupby('unit').p_two_sided.transform(bh);save(q,'09_category_conditioned_chain_null')
anchors=pd.DataFrame(ances);save(anchors,'09_all_focal_path_anchors')
for pat,g in anchors.groupby('pattern'):
 j=next(j for j,(a,b) in enumerate(product(F,repeat=2)) if a+' -> '+b==pat)
 assert obs[j]==len(g) and obsp[j]==g.page_key.nunique(),pat
overview=[]
for pat,g in anchors.groupby('pattern'):
 perpage=g.groupby('page_key').size();overview.append(dict(pattern=pat,paths=len(g),pages=len(perpage),largest_page_share=perpage.max()/len(g),top5_page_share=perpage.nlargest(5).sum()/len(g),same_sentence_share=g.same_sentence.mean(),sovereign_intermediate_share=(g.intermediate_category=='E01').mean()))
save(pd.DataFrame(overview),'09_path_concentration')
jsave(dict(conditioning='Predicate-family labels shuffled within owner page and ordered subject/object axial-category pair',swappable_groups=len(groups),variable_family_groups=sum(len(set(labels[ix]))>1 for ix in groups),
 meaning='Controls broad category direction and page content, testing finer endpoint-label alignment. Page-presence sensitivity avoids large-page path multiplication. Neither statistic establishes event succession.'),'09_scope')
# Bring every selected source passage into a compact readable ledger.
sents={s['sentence_id']:s for s in map(json.loads,(O/'inputs/sentences.jsonl').read_text(encoding='utf-8').splitlines())}
chosen=anchors.groupby('pattern',group_keys=False).apply(lambda x:x.drop_duplicates('page_key').head(12)).reset_index(drop=True)
ids=set(chosen.sentence1)|set(chosen.sentence2)
cells=pd.read_csv(RESULTS/'01_source_anchors.csv');ids.update(cells.sentence_id)
rules=pd.read_csv(RESULTS/'04_rule_source_anchors.csv');ids.update(rules.sentence1);ids.update(rules.sentence2)
save(pd.DataFrame([dict(sentence_id=sid,volume=sents[sid]['volume_id'],page=sents[sid]['owner_page'],burmese=sents[sid]['burmese'],english_translation=sents[sid]['english_translation']) for sid in sorted(ids)]),'09_selected_source_passages')
save(chosen,'09_selected_path_anchors')
print('Motif sensitivity completed',overview,flush=True)

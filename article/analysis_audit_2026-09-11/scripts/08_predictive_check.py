"""ComplEx vs DistMult and frequency baselines on held-out category facts.

This is a diagnostic of representation, never evidence of an unrecorded event.
"""
from common import *
import torch
import copy
torch.set_num_threads(2);torch.manual_seed(SEED);np.random.seed(SEED)
d=load();facts=d[['S','P','O']].drop_duplicates().reset_index(drop=True);arr=np.column_stack([facts.S.map(dict(zip(E,range(52)))),facts.P.map(dict(zip(R,range(81)))),facts.O.map(dict(zip(E,range(52))))]).astype(np.int64)
rng=np.random.default_rng(SEED+8);ix=rng.permutation(len(arr));ntrain=int(.7*len(arr));nval=int(.1*len(arr));train=arr[ix[:ntrain]];valid=arr[ix[ntrain:ntrain+nval]];test=arr[ix[ntrain+nval:]]
factsplit=np.full(len(arr),'test',object);factsplit[ix[:ntrain]]='train';factsplit[ix[ntrain:ntrain+nval]]='validation';facts['split']=factsplit;save(facts,'08_fact_split')
known=np.zeros((52,81,52),bool);known[arr[:,0],arr[:,1],arr[:,2]]=True
trainknown=np.zeros_like(known);trainknown[train[:,0],train[:,1],train[:,2]]=True
validation_known=trainknown.copy();validation_known[valid[:,0],valid[:,1],valid[:,2]]=True
class Bilinear(torch.nn.Module):
 def __init__(self,complex_model):
  super().__init__();self.complex_model=complex_model;self.E=torch.nn.Embedding(52,64 if complex_model else 32);self.R=torch.nn.Embedding(81,64 if complex_model else 32)
  torch.nn.init.normal_(self.E.weight,std=.15);torch.nn.init.normal_(self.R.weight,std=.15)
 def forward(self,s,p):
  a=self.E(s);r=self.R(p);b=self.E.weight
  if not self.complex_model:return (a*r)@b.T
  ar,ai=a.chunk(2,dim=-1);rr,ri=r.chunk(2,dim=-1);br,bi=b.chunk(2,dim=-1)
  return (ar*rr-ai*ri)@br.T+(ar*ri+ai*rr)@bi.T
def score_model(model,z):
 with torch.no_grad():return model(torch.tensor(z[:,0]),torch.tensor(z[:,1])).numpy()
def metrics(score,z,filter_known=True,reference=None):
 score=score.copy();target=score[np.arange(len(z)),z[:,2]].copy()
 if filter_known:
  source=known if reference is None else reference
  exclude=source[z[:,0],z[:,1]].copy();exclude[np.arange(len(z)),z[:,2]]=False;score[exclude]=-np.inf
 # Average rank for ties, including the correct candidate.
 ranks=1+(score>target[:,None]).sum(1)+.5*((score==target[:,None]).sum(1)-1)
 return dict(MRR=float((1/ranks).mean()),hits1=float((ranks<=1).mean()),hits3=float((ranks<=3).mean()),hits10=float((ranks<=10).mean()),facts=len(z)),ranks
results=[];traces=[]
for name,complex_model in [('DistMult',False),('ComplEx',True)]:
 torch.manual_seed(SEED);model=Bilinear(complex_model);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=1e-4);best=(-1,None,0)
 for epoch in range(1,201):
  loss_sum=0
  for batch in np.array_split(rng.permutation(len(train)),max(1,len(train)//256)):
   z=train[batch];scores=model(torch.tensor(z[:,0]),torch.tensor(z[:,1]));exclude=trainknown[z[:,0],z[:,1]].copy();exclude[np.arange(len(z)),z[:,2]]=False;scores=scores.masked_fill(torch.tensor(exclude),-1e6)
   loss=torch.nn.functional.cross_entropy(scores,torch.tensor(z[:,2]));opt.zero_grad();loss.backward();opt.step();loss_sum+=float(loss.detach())
  if epoch%10==0:
   m,_=metrics(score_model(model,valid),valid,reference=validation_known);traces.append(dict(model=name,epoch=epoch,train_batch_loss_sum=loss_sum,validation_MRR=m['MRR']))
   if m['MRR']>best[0]:best=(m['MRR'],copy.deepcopy(model.state_dict()),epoch)
   if epoch-best[2]>=50:break
 model.load_state_dict(best[1]);score=score_model(model,test)
 for mode in [True,False]:
  m,ranks=metrics(score,test,mode);results.append(dict(model=name,filter='all_other_known_category_facts' if mode else 'unfiltered',selected_epoch=best[2],**m))
 torch.save(model.state_dict(),RESULTS/('08_'+name+'_parameters.pt'))
freq=np.ones((81,52))*.5;np.add.at(freq,(train[:,1],train[:,2]),1);scores=np.log(freq[test[:,1]])
for mode in [True,False]:
 m,_=metrics(scores,test,mode);results.append(dict(model='predicate_object_frequency',filter='all_other_known_category_facts' if mode else 'unfiltered',**m))
save(pd.DataFrame(results),'08_link_prediction');save(pd.DataFrame(traces),'08_training_trace')
jsave(dict(unique_category_facts=len(arr),train=len(train),validation=len(valid),test=len(test),embedding_dimension=32,
 limits='Held-out facts are unique category S-P-O cells, not people or dated events. Labels from test facts enter only the conventional filtered-ranking exclusion, never fitting. Incomplete observations are not verified negative facts. No predicted fact is added to the data. No claim of causal inference or historical discovery. Evaluation is object-category ranking among 52 categories, not raw-tag link prediction.'),'08_scope')
print('Prediction diagnostic completed',results,flush=True)

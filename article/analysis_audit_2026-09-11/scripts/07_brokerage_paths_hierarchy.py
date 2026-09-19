from common import *
import networkx as nx
from scipy.linalg import expm
from itertools import combinations
from collections import Counter

d=load();A=tensor(d).sum(1);np.fill_diagonal(A,0);G=nx.from_numpy_array(A,create_using=nx.DiGraph);U=nx.from_numpy_array(A+A.T);B=A>0;n=len(E)
# Existing directed centrality file uses eigenvectors only on the largest SCC.
edge=nx.edge_betweenness_centrality(G);save(pd.DataFrame([dict(source=E[i],target=E[j],betweenness=w) for (i,j),w in edge.items()]),'07_edge_betweenness')
loadc=nx.load_centrality(G);constraint=nx.constraint(U,weight='weight');effective=nx.effective_size(U,weight='weight');largest=U.subgraph(max(nx.connected_components(U),key=len)).copy();current=nx.current_flow_betweenness_centrality(largest,weight=None);currentcl=nx.current_flow_closeness_centrality(largest,weight=None)
rows=[]
for i,e in enumerate(E):rows.append(dict(category=e,label=LABEL[e],burt_constraint=constraint[i],effective_size=effective[i],ego_efficiency=effective[i]/U.degree(i) if U.degree(i) else None,load_centrality=loadc[i],current_flow_betweenness=current.get(i),current_flow_closeness=currentcl.get(i)))
save(pd.DataFrame(rows),'07_brokerage_centrality')
# G-F roles are counted on the 30 actor/organization categories with explicit groups.
actors=E[:23]+E[45:];grp={}
for e in actors:
 k=int(e[1:]);grp[e]='dynasty' if k<=4 else 'central_court' if k in [5,6,23,52] else 'military' if k in [7,8] else 'religious' if k in [20,21,22] else 'foreign_diplomatic' if k in [17,18,19] else 'provincial_social'
idx=[E.index(e) for e in actors];H=G.subgraph(idx);brokers=[]
for b in H:
 count=Counter()
 for a in H.predecessors(b):
  for c in H.successors(b):
   if a==c or H.has_edge(a,c):continue
   ga,gb,gc=grp[E[a]],grp[E[b]],grp[E[c]]
   role='coordinator' if ga==gb==gc else 'itinerant' if ga==gc else 'representative' if ga==gb else 'gatekeeper' if gb==gc else 'liaison'
   count[role]+=1
 brokers.append(dict(category=E[b],group=grp[E[b]],**{k:count[k] for k in ['coordinator','itinerant','representative','gatekeeper','liaison']}))
save(pd.DataFrame(brokers),'07_gould_fernandez_brokerage')
# Every reachable ordered pair and a weighted path sensitivity.
W=G.copy()
for i,j in W.edges:W[i][j]['distance']=1/W[i][j]['weight']
paths=[]
for a in G:
 un=nx.single_source_shortest_path_length(G,a);weighted=nx.single_source_dijkstra_path_length(W,a,weight='distance')
 for b in G:
  if a!=b:paths.append(dict(source=E[a],target=E[b],reachable=b in un,shortest_path=un.get(b),inverse_frequency_distance=weighted.get(b)))
save(pd.DataFrame(paths),'07_all_pairs_paths')
mut=int((B&B.T).sum()//2);asym=int((B&~B.T).sum());null=n*(n-1)//2-mut-asym
degree=np.array([U.degree(i) for i in U]);common=(nx.to_numpy_array(U,weight=None)@nx.to_numpy_array(U,weight=None));squares=sum(v*(v-1)/2 for v in common[np.triu_indices(n,1)])/2
summary=dict(mutual_dyads=mut,asymmetric_dyads=asym,null_dyads=null,undirected_triangles=int(sum(nx.triangles(U).values())//3),undirected_squares=int(squares),
 degree_assortativity_directed=nx.degree_assortativity_coefficient(G,x='out',y='in'),degree_assortativity_undirected=nx.degree_assortativity_coefficient(U),
 node_connectivity_largest_undirected=nx.node_connectivity(largest),edge_connectivity_largest_undirected=nx.edge_connectivity(largest),k_shell_sizes=dict(Counter(nx.core_number(U).values())))
jsave(summary,'07_topology_supplement')
tree=nx.maximum_spanning_tree(U,weight='weight');save(pd.DataFrame([dict(source=E[i],target=E[j],summed_directional_claims=w['weight']) for i,j,w in tree.edges(data=True)]),'07_maximum_weight_spanning_forest')
# Unit edge capacity measures independent routes, not material transport volume.
unitG=G.copy();nx.set_edge_attributes(unitG,1,'capacity');cut=[]
for j in G:
 if j==0:continue
 flow,part=nx.minimum_cut(unitG,0,j,capacity='capacity');cut.append(dict(source='E01',target=E[j],edge_disjoint_route_capacity=flow,source_side_nodes=len(part[0])))
save(pd.DataFrame(cut),'07_sovereign_minimum_cuts')
# Reachability vulnerability to removal of each analytical category.
def efficiency(g):
 ds=[1/l for i in g for j,l in nx.single_source_shortest_path_length(g,i).items() if i!=j];return sum(ds)/(len(g)*(len(g)-1)) if len(g)>1 else 0
v=[];baseeff=efficiency(G)
for i,e in enumerate(E):
 h=G.copy();h.remove_node(i);v.append(dict(removed_category=e,remaining_nodes=len(h),remaining_edges=h.number_of_edges(),remaining_efficiency=efficiency(h),baseline_efficiency=baseeff))
save(pd.DataFrame(v),'07_category_removal_sensitivity')
# Hitting / commute times only on an irreducible component (no teleportation).
scc=sorted(max(nx.strongly_connected_components(G),key=len));a=A[np.ix_(scc,scc)];p=a/a.sum(1,keepdims=True);vals,vec=np.linalg.eig(p.T);pi=np.real(vec[:,np.argmin(abs(vals-1))]);pi=pi/pi.sum();Z=np.linalg.inv(np.eye(len(p))-p+np.ones((len(p),1))*pi[None,:]);hit=(np.diag(Z)[None,:]-Z)/pi[None,:]
walk=[]
for i,a in enumerate(scc):
 for j,b in enumerate(scc):
  if i!=j:walk.append(dict(source=E[a],target=E[b],expected_walk_steps=hit[i,j],commute_steps=hit[i,j]+hit[j,i]))
save(pd.DataFrame(walk),'07_random_walk_hitting_times')
# Generalized trophic levels: least-squares directed differences, valid with cycles.
lap=np.diag((A+A.T).sum(1))-(A+A.T);imbalance=A.sum(0)-A.sum(1);level=np.linalg.lstsq(lap,imbalance,rcond=None)[0];level-=level.min();incoherence=float((A*(level[None,:]-level[:,None]-1)**2).sum()/A.sum())
save(pd.DataFrame({'category':E,'generalized_trophic_level':level}),'07_trophic_levels')
# Degree-ordered rich club and NODF-like nestedness on adjacency, no significance claim.
rich=nx.rich_club_coefficient(U,normalized=False);save(pd.DataFrame([dict(degree_threshold=k,rich_club_coefficient=v) for k,v in rich.items()]),'07_rich_club')
bb=nx.to_numpy_array(U,weight=None);deg=bb.sum(1);nested=[]
for i,j in combinations(range(n),2):
 if deg[i]!=deg[j] and min(deg[i],deg[j])>0:nested.append(np.dot(bb[i],bb[j])/min(deg[i],deg[j]))
jsave(dict(generalized_trophic_incoherence=incoherence,degree_ordered_nestedness=float(np.mean(nested)),nestedness_pairs=len(nested),
 scope='Category-graph diagnostics; raw extraction direction is retained. Degree, routes, random walks, trophic level and current flow do not measure literal historical movement, resources or social rank. G-F groups are analyst choices. Weighted distance is inverse claim count, not physical or temporal distance.'),'07_hierarchy_scope')
# Save an explicitly attenuated communicability matrix to avoid overflow/popularity domination.
adj=nx.to_numpy_array(U,weight=None);scale=max(abs(np.linalg.eigvalsh(adj)));com=expm(adj/scale)
save(pd.DataFrame([dict(category_a=E[i],category_b=E[j],spectrally_scaled_communicability=com[i,j]) for i,j in combinations(range(n),2)]),'07_communicability')
print('Brokerage hierarchy and paths completed',summary,flush=True)

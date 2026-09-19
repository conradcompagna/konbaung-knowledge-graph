"""Encoded category graphs and relation layers; QAP and positional profiles."""

from common import *
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.spatial.distance import jensenshannon


def graph(a):
    b = a.copy()
    np.fill_diagonal(b, 0)
    return nx.from_numpy_array(b, create_using=nx.DiGraph)


def metrics(g):
    n = len(g)
    m = g.number_of_edges()
    ug = g.to_undirected()
    deg = np.array([g.degree(i) for i in g])
    out = np.array([g.out_degree(i) for i in g])
    inn = np.array([g.in_degree(i) for i in g])
    scc = list(nx.strongly_connected_components(g))
    wcc = list(nx.weakly_connected_components(g))
    reach = sum(len(nx.descendants(g, i)) for i in g)
    r = [len(nx.descendants(g, i)) / (n - 1) for i in g] if n > 1 else [0]
    return dict(
        nodes=n,
        edges=m,
        density=nx.density(g),
        reciprocity=nx.reciprocity(g) if m else 0,
        weak_components=len(wcc),
        strong_components=len(scc),
        largest_weak=max(map(len, wcc)),
        largest_strong=max(map(len, scc)),
        isolates=len(list(nx.isolates(g))),
        transitivity=nx.transitivity(ug),
        average_clustering=nx.average_clustering(ug),
        directed_average_clustering=nx.average_clustering(g),
        flow_hierarchy=nx.flow_hierarchy(g) if m else 0,
        reachable_pair_share=reach / (n * (n - 1)),
        global_reaching_centralization=sum(max(r) - a for a in r) / (n - 1),
        outdegree_centralization=float((out.max() - out).sum() / (n - 1) ** 2),
        indegree_centralization=float((inn.max() - inn).sum() / (n - 1) ** 2),
        bridges=len(list(nx.bridges(ug))),
        articulation_points=len(list(nx.articulation_points(ug))),
        global_efficiency=nx.global_efficiency(ug),
    )


d = load()
x = tensor(d)
layers = x.transpose(1, 0, 2)
A = layers.sum(0)
G = graph(A)
jsave(metrics(G), "02_category_topology")
pr = nx.pagerank(G, weight="weight")
bw = nx.betweenness_centrality(G, normalized=True, weight=None)
cl = nx.closeness_centrality(G)
ha = nx.harmonic_centrality(G)
largest_scc = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
eig0 = nx.eigenvector_centrality_numpy(largest_scc, weight="weight")
eig = {i: eig0.get(i, np.nan) for i in G}
hubs, auth = nx.hits(G, max_iter=1000)
alpha = 0.85 / max(abs(np.linalg.eigvals(nx.to_numpy_array(G))))
katz = nx.katz_centrality_numpy(G, alpha=float(alpha), weight="weight")
core = nx.core_number(G.to_undirected())
rows = []
stren = layers.sum(1) + layers.sum(2)
den = stren.sum(0)
part = 1 - np.divide((stren**2).sum(0), den**2, out=np.zeros_like(den), where=den > 0)
for i, e in enumerate(E):
    rows.append(
        dict(
            entity=e,
            label=LABEL[e],
            in_degree=G.in_degree(i),
            out_degree=G.out_degree(i),
            in_strength=G.in_degree(i, weight="weight"),
            out_strength=G.out_degree(i, weight="weight"),
            pagerank=pr[i],
            betweenness=bw[i],
            in_closeness=cl[i],
            harmonic=ha[i],
            eigenvector=eig[i],
            katz=katz[i],
            HITS_hub=hubs[i],
            HITS_authority=auth[i],
            coreness=core[i],
            layer_participation=part[i],
        )
    )
save(pd.DataFrame(rows), "02_category_centralities")
ly = []
lr = []
for j, r in enumerate(R):
    g = graph(layers[j])
    v = metrics(g)
    v.update(
        relation_id=r,
        label=LABEL[r],
        occurrences=int(layers[j].sum()),
        category_self_occurrences=int(np.trace(layers[j])),
    )
    ly.append(v)
    p = nx.pagerank(g, weight="weight")
    for i, e in enumerate(E):
        lr.append(
            dict(
                relation_id=r,
                entity_id=e,
                pagerank=p[i],
                out_strength=float(layers[j, i].sum()),
                in_strength=float(layers[j, :, i].sum()),
            )
        )
save(pd.DataFrame(ly), "02_layer_statistics")
save(pd.DataFrame(lr), "02_layer_centralities")
mask = ~np.eye(52, dtype=bool)
pairs = []
for a in range(81):
    for b in range(a + 1, 81):
        u = layers[a][mask]
        v = layers[b][mask]
        union = ((u > 0) | (v > 0)).sum()
        pairs.append(
            dict(
                r1=R[a],
                r2=R[b],
                edge_jaccard=float(((u > 0) & (v > 0)).sum() / union) if union else 0,
                log_weight_correlation=float(np.corrcoef(np.log1p(u), np.log1p(v))[0, 1])
                if u.std() * v.std()
                else None,
                outdegree_spearman=float(
                    stats.spearmanr(layers[a].sum(1), layers[b].sum(1)).statistic
                ),
                indegree_spearman=float(
                    stats.spearmanr(layers[a].sum(0), layers[b].sum(0)).statistic
                ),
            )
        )
save(pd.DataFrame(pairs), "02_all_layer_pairs")
# Prespecified comparisons of authorization, implementation, information, resources,
# coercion and resistance. QAP permutes category labels on one layer.
focal = [
    ("R11", "R12"),
    ("R11", "R13"),
    ("R11", "R14"),
    ("R11", "R23"),
    ("R23", "R24"),
    ("R25", "R26"),
    ("R26", "R29"),
    ("R26", "R21"),
    ("R33", "R34"),
    ("R36", "R44"),
    ("R54", "R52"),
    ("R76", "R52"),
    ("R09", "R12"),
    ("R72", "R73"),
    ("R23", "R76"),
]
rng = np.random.default_rng(SEED + 2)
qrows = []
# Four explicit ontology strata, used only as a sensitivity null.
strata = [
    np.array(list(range(23)) + list(range(45, 52))),
    np.arange(23, 29),
    np.arange(29, 43),
    np.arange(43, 45),
]
for r1, r2 in focal:
    a = np.log1p(layers[R.index(r1)])
    b = np.log1p(layers[R.index(r2)])
    u = a[mask]
    u = (u - u.mean()) / u.std()
    obs = np.corrcoef(a[mask], b[mask])[0, 1]
    for mode in ["unrestricted", "ontology_stratified"]:
        null = []
        for j in range(1999):
            perm = rng.permutation(52)
            if mode == "ontology_stratified":
                perm = np.arange(52)
                for ix in strata:
                    perm[ix] = rng.permutation(ix)
            v = b[np.ix_(perm, perm)][mask]
            null.append(float(np.mean(u * (v - v.mean()) / v.std())))
        qrows.append(dict(r1=r1, r2=r2, null_model=mode, **permutation_summary(obs, null)))
q = pd.DataFrame(qrows)
q["q_two_sided"] = q.groupby("null_model").p_two_sided.transform(bh)
save(q, "02_layer_qap")
# Positions from incoming/outgoing predicate profiles, with Hellinger transform.
prof = np.concatenate([x.sum(2), x.sum(0).T], axis=1)
prof = np.sqrt(
    np.divide(
        prof,
        prof.sum(1, keepdims=True),
        out=np.zeros_like(prof),
        where=prof.sum(1, keepdims=True) > 0,
    )
)
score = []
best = None
for k in range(2, 9):
    fit = KMeans(k, n_init=50, random_state=SEED).fit(prof)
    sil = silhouette_score(prof, fit.labels_)
    score.append(dict(k=k, silhouette=sil))
    if best is None or sil > best[0]:
        best = (sil, k, fit.labels_)
save(pd.DataFrame(score), "02_role_model_selection")
save(
    pd.DataFrame(
        {
            "entity_id": E,
            "label": [LABEL[e] for e in E],
            "profile_role": best[2],
            "selected_k": best[1],
        }
    ),
    "02_profile_roles",
)
sim = prof @ prof.T
nr = []
for i, e in enumerate(E):
    for j in [j for j in np.argsort(-sim[i]) if j != i][:5]:
        nr.append(dict(entity=e, neighbor=E[j], profile_cosine=sim[i, j]))
save(pd.DataFrame(nr), "02_role_neighbors")
com = []
cmaps = []
community_graph = nx.from_numpy_array(nx.to_numpy_array(G) + nx.to_numpy_array(G).T)
for seed in range(20):
    comm = nx.community.louvain_communities(community_graph, weight="weight", seed=seed)
    cmap = {n: j for j, c in enumerate(comm) for n in c}
    cmaps.append([cmap[i] for i in range(52)])
    com.append(
        dict(
            seed=seed,
            communities=len(comm),
            modularity=nx.community.modularity(community_graph, comm, weight="weight"),
            ARI_to_seed0=adjusted_rand_score(cmaps[0], cmaps[-1]),
        )
    )
save(pd.DataFrame(com), "02_category_community_stability")
save(pd.DataFrame({"entity_id": E, "community_seed0": cmaps[0]}), "02_category_communities")
# Extend the archived cautious named-person graph without inventing directed ties.
edges = pd.read_csv(O / "inputs/person_edges.csv")
ng = nx.from_pandas_edgelist(edges, "a", "b", ["claims", "pages"])
largest = ng.subgraph(max(nx.connected_components(ng), key=len)).copy()
meta = dict(
    nodes=len(ng),
    edges=ng.number_of_edges(),
    components=nx.number_connected_components(ng),
    largest_component=len(largest),
    density=nx.density(ng),
    transitivity=nx.transitivity(ng),
    bridges=len(list(nx.bridges(ng))),
    articulation_points=len(list(nx.articulation_points(ng))),
    diameter_largest_component=nx.diameter(largest),
    radius_largest_component=nx.radius(largest),
    mean_distance_largest_component=nx.average_shortest_path_length(largest),
    global_efficiency=nx.global_efficiency(ng),
)
jsave(meta, "02_named_person_topology")
b = nx.betweenness_centrality(ng)
h = nx.harmonic_centrality(ng)
c = nx.closeness_centrality(ng)
core = nx.core_number(ng)
central = pd.read_csv(O / "inputs/person_centrality.csv")
central["betweenness"] = central.cluster_id.map(b)
central["harmonic"] = central.cluster_id.map(h)
central["closeness"] = central.cluster_id.map(c)
central["coreness"] = central.cluster_id.map(core)
save(central.sort_values("betweenness", ascending=False), "02_named_person_extended_centrality")
print("Graph layers completed", meta, flush=True)

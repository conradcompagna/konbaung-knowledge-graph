"""Page-scoped exact-tag paths/rules; predicate-label and degree nulls."""

from common import *
from collections import defaultdict, Counter
import networkx as nx
from itertools import product, combinations

d = load()
split = pd.read_csv(RESULTS / "03_page_split.csv")
testpages = set(split.loc[split.test_page, "page_key"])
edges = d[d.s != d.o].drop_duplicates(["page_key", "s", "P", "o"]).reset_index(drop=True)
families = {
    "authorization": {"R06", "R11", "R12", "R14", "R15", "R23", "R69", "R75"},
    "execution": {"R13", "R24", "R33", "R36", "R37", "R40", "R43", "R44", "R45", "R46", "R47"},
    "information": {"R25", "R26", "R27", "R28", "R29"},
    "coercion": {"R21", "R39", "R41", "R51", "R52", "R53"},
    "resistance": {"R54", "R55", "R73", "R76", "R77", "R79"},
    "allegiance": {"R16", "R30", "R34", "R56", "R57", "R58"},
    "other": set(R),
}
F = list(families)
fm = {r: next(i for i, f in enumerate(F) if r in families[f]) for r in R}
labels = edges.P.map(fm).to_numpy()
K = len(F)
body = defaultdict(set)
head = defaultdict(set)
headsubject = defaultdict(set)
rule_support = defaultdict(set)
chain_anchors = {}
fork = Counter()
trans = []
cycle = []
chains = []
for page, g in edges.groupby("page_key", sort=True):
    out = defaultdict(list)
    pair = defaultdict(list)
    for t in g.itertuples():
        out[t.s].append(t)
        pair[(t.s, t.o)].append(t)
        key = (page, t.s, t.o)
        head[t.P].add(key)
        headsubject[t.P].add((page, t.s))
    for s, elist in out.items():
        rels = sorted({e.P for e in elist})
        for r1, r2 in combinations(rels, 2):
            fork[(r1, r2)] += 1
        for a in elist:
            for b in out.get(a.o, []):
                if b.o == a.s:
                    continue
                key = (page, a.s, b.o)
                rp = (a.P, b.P)
                body[rp].add(key)
                chains.append((a.Index, b.Index))
                chain_anchors.setdefault(
                    (rp, key), (a.triple_id, b.triple_id, a.sentence_id, b.sentence_id, a.o)
                )
                for c in pair.get((a.s, b.o), []):
                    rule_support[(a.P, b.P, c.P)].add(key)
                    trans.append((a.Index, b.Index, c.Index))
                for c in pair.get((b.o, a.s), []):
                    cycle.append((a.Index, b.Index, c.Index))
save(
    pd.DataFrame(
        [
            dict(
                p1=k[0],
                p2=k[1],
                distinct_page_endpoint_pairs=len(v),
                support_pages=len({x[0] for x in v}),
            )
            for k, v in body.items()
        ]
    ).sort_values("distinct_page_endpoint_pairs", ascending=False),
    "04_frequent_paths",
)
save(
    pd.DataFrame(
        [dict(p1=k[0], p2=k[1], source_page_support=n) for k, n in fork.items()]
    ).sort_values("source_page_support", ascending=False),
    "04_frequent_outgoing_relation_pairs",
)
rules = []
anchors = []
for (a, b, c), support in rule_support.items():
    bod = body[(a, b)]
    trainbod = {k for k in bod if k[0] not in testpages}
    testbod = bod - trainbod
    trainsup = {k for k in support if k[0] not in testpages}
    testsup = support - trainsup
    if len(trainsup) < 3:
        continue
    htrain = {k for k in head[c] if k[0] not in testpages}
    eligible = {k for k in trainbod if (k[0], k[1]) in headsubject[c]}
    rules.append(
        dict(
            body1=a,
            body2=b,
            head=c,
            train_support=len(trainsup),
            train_body_pairs=len(trainbod),
            train_confidence=len(trainsup) / len(trainbod),
            head_coverage=len(trainsup) / len(htrain),
            PCA_denominator=len(eligible),
            PCA_confidence=len(trainsup) / len(eligible) if eligible else None,
            test_support=len(testsup),
            test_body_pairs=len(testbod),
            test_confidence=len(testsup) / len(testbod) if testbod else None,
            train_pages=len({k[0] for k in trainsup}),
            test_pages=len({k[0] for k in testsup}),
        )
    )
    for key in sorted(support)[:3]:
        anc = chain_anchors[((a, b), key)]
        anchors.append(
            dict(
                body1=a,
                body2=b,
                head=c,
                page_key=key[0],
                subject=key[1],
                intermediate=anc[4],
                object=key[2],
                triple1=anc[0],
                triple2=anc[1],
                sentence1=anc[2],
                sentence2=anc[3],
            )
        )
rules = pd.DataFrame(rules)
if len(rules):
    rules = rules.sort_values(["train_support", "train_confidence"], ascending=False)
save(rules, "04_length2_horn_rules")
save(pd.DataFrame(anchors), "04_rule_source_anchors")
single = []
for a, b in product(R, R):
    if a == b:
        continue
    h1 = {k for k in head[a] if k[0] not in testpages}
    h2 = {k for k in head[b] if k[0] not in testpages}
    sup = h1 & h2
    if len(sup) < 3:
        continue
    t1 = {k for k in head[a] if k[0] in testpages}
    t2 = {k for k in head[b] if k[0] in testpages}
    single.append(
        dict(
            body=a,
            head=b,
            train_support=len(sup),
            train_body_pairs=len(h1),
            train_confidence=len(sup) / len(h1),
            head_coverage=len(sup) / len(h2),
            test_body_pairs=len(t1),
            test_support=len(t1 & t2),
            test_confidence=len(t1 & t2) / len(t1) if t1 else None,
        )
    )
save(pd.DataFrame(single), "04_single_edge_rules")
# Count edge-instance configurations (parallel predicates retained); cycle counts
# are divided by 3 only in untyped summaries, because labels distinguish rotation.
arrays = {
    "chain": np.asarray(chains, dtype=int).reshape(-1, 2),
    "transitive": np.asarray(trans, dtype=int).reshape(-1, 3),
    "cyclic": np.asarray(cycle, dtype=int).reshape(-1, 3),
}


def counts(lab, ar):
    codes = np.zeros(len(ar), dtype=int)
    for c in range(ar.shape[1]):
        codes = codes * K + lab[ar[:, c]]
    return np.bincount(codes, minlength=K ** ar.shape[1])


observed = {key: counts(labels, ar) for key, ar in arrays.items()}
null = {key: np.zeros((1999, len(obs)), int) for key, obs in observed.items()}
rng = np.random.default_rng(SEED + 4)
groups = list(edges.groupby("page_key").indices.values())
for i in range(1999):
    lp = labels.copy()
    for ix in groups:
        lp[ix] = rng.permutation(labels[ix])
    for key, ar in arrays.items():
        null[key][i] = counts(lp, ar)
mr = []
for key, obs in observed.items():
    n = arrays[key].shape[1]
    for j, pattern in enumerate(product(F, repeat=n)):
        mr.append(
            dict(
                motif=key,
                pattern=" -> ".join(pattern),
                **permutation_summary(obs[j], null[key][:, j]),
            )
        )
mr = pd.DataFrame(mr)
mr["q_two_sided"] = bh(mr.p_two_sided)
mr["q_enrichment"] = bh(mr.p_greater)
save(mr, "04_labelled_motif_tests")
np.savez_compressed(RESULTS / "04_labelled_motif_null_draws.npz", **null)
jsave(
    dict(
        page_scoped_edges=len(edges),
        exact_tag_paths=len(chains),
        transitive_configurations=len(trans),
        cyclic_oriented_configurations=len(cycle),
        rule_candidates=len(rules),
        note="Raw exact tags are scoped within owner page. Labels are axial; edges are grammatical extraction directions. Motif null preserves page predicate-family totals and endpoint topology, not per-layer degree. Rule holdout is pages, not historical future. Missing edges are unrecorded, not false.",
        families={
            k: sorted(v) if k != "other" else "remaining relations" for k, v in families.items()
        },
    ),
    "04_scope",
)
# The separate unlabelled degree-preserving triad test is 04b_degree_null.py.
print("Paths rules motifs completed", len(rules), flush=True)

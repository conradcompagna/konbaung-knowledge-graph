"""Human-reviewed family inventory and an explicit, conservative method checklist."""

from common import *
import re
from collections import Counter

inv = json.loads((O / "inventory.json").read_text(encoding="utf-8"))


def evidence(names):
    result = []
    for name in names:
        hits = [r["source"] for r in inv if r["source"].endswith("/" + name) or r["source"] == name]
        if hits:
            result.append(min(hits, key=len))
        else:
            raise AssertionError("Missing evidence " + name)
    return " | ".join(result)


# Each record distinguishes what was actually found from what was added.
catalog = [
    (
        1,
        "Native triple and RDF profiling",
        "Already covered",
        "relation_frequency.csv;relation_royal_subject_and_actor_diversity.csv;actor_relation_profiles.csv",
        "Canonical counts, frequency and category participation, sovereign roles and relation diversity. RDF export metadata are not an additional historical test.",
        "Data revalidated against intact original source.",
        "data_validation.json",
        "",
    ),
    (
        2,
        "Three-way categorical analysis",
        "Partly covered",
        "reproduce_derived_tests.py;oriented_revocations_actor_analysis_clean.csv",
        "Pairwise chi-square/Cramer V, odds ratios, GEE and conditional contrasts existed. No full S-P-O loglinear fit located.",
        "Mutual independence, SP/PO conditional independence, all-pairwise IPF, Pearson residuals and a page/predicate constrained permutation.",
        "results/01_loglinear_models.csv;results/01_conditional_permutation.json;results/01_threeway_cells.csv",
        "Three-way residuals are descriptive; no asymptotic chi-square p-values for sparse cells. The randomization tests S-O association given P/page, not the isolated three-way interaction.",
    ),
    (
        3,
        "Directed graph topology and centrality",
        "Partly covered",
        "named_person_network_centrality.csv;named_person_network_edges.csv",
        "Named-person network already had undirected degree/strength, PageRank and components.",
        "Directed category centralities and connectivity, paths, cuts, walks, named-person graph extensions.",
        "results/02_category_centralities.csv;results/02_named_person_topology.json;results/07_topology_supplement.json",
        "Category graphs retain extraction direction; named-person graph remains the archived conservative undirected graph.",
    ),
    (
        4,
        "Dyads triads and motifs",
        "Not found as network tests",
        [],
        "Burmese orthographic motifs and relation cliques are present, but are different objects.",
        "Typed chains/transitive/cyclic patterns with page-label nulls; stricter page/category null; unlabelled triads against two degree-preserving MCMC chains.",
        "results/04_labelled_motif_tests.csv;results/09_category_conditioned_chain_null.csv;results/04_degree_preserving_triads.csv",
        "No exhaustive census of every larger graphlet size.",
    ),
    (
        5,
        "Paths reachability and flow",
        "Not found as a dedicated family",
        [],
        "Person career order and adjacent page windows are not graph-path analysis.",
        "Shortest/weighted paths, reachability, typed page-scoped paths, hitting/commute times, cuts, removal sensitivity.",
        "results/07_all_pairs_paths.csv;results/04_frequent_paths.csv;results/07_random_walk_hitting_times.csv",
        "Graph walk steps and inverse-frequency distances are mathematical diagnostics, not measured travel, time or resources.",
    ),
    (
        6,
        "Communities positions and roles",
        "Partly covered",
        "community_seed_resolution_audit.csv;community_stability_summary.csv;relation_category_centered_k8.csv",
        "Page relation communities, semantic clusters, Louvain, clique and seed/resolution checks existed.",
        "Predicate-specific directional role profiles, category communities, an explicit Bernoulli SBM and seed checks.",
        "results/02_profile_roles.csv;results/05_sbm_memberships.csv;results/05_sbm_selection.csv",
        "Leiden/Infomap and exact regular or automorphic equivalence were not run; profile similarity is not exact regular equivalence.",
    ),
    (
        7,
        "Multiplex relation layers",
        "Not found as actual S-to-O layers",
        [],
        "Existing relation co-occurrence graphs connect predicates to predicates; they are not 81 endpoint-adjacency layers.",
        "All 81 layers, overlap, centrality, participation, 3240 descriptive layer comparisons, 15 prespecified QAP comparisons and sensitivities.",
        "results/02_layer_statistics.csv;results/02_all_layer_pairs.csv;results/02_layer_qap.csv;results/06_layer_qap_sensitivity.csv",
        "No supra-adjacency community/centrality fit or multilayer SBM.",
    ),
    (
        8,
        "Tensor and multiway decomposition",
        "Not found",
        [],
        "Embedding clustering was not S-P-O tensor decomposition.",
        "Nonnegative CP rank 3/5/7 selection, full rank-7 descriptive fit, Tucker 8/8/8 and seed/register/status sensitivities.",
        "results/03_tensor_model_selection.csv;results/03_static_cp_loadings.csv;results/03_tucker_fit.json;results/06_tensor_component_stability.csv",
        "RESCAL, DEDICOM and coupled tensor/matrix variants remain unrun alternatives.",
    ),
    (
        9,
        "Statistical relational learning",
        "Not found as a fitted triple model",
        "title_embedding_classifier_summary.csv",
        "Title-category embedding classifiers existed, but not joint S-P-O completion models.",
        "Held-out category-fact DistMult and ComplEx, plus frequency baseline; page-held-out CP selection.",
        "results/08_link_prediction.csv;results/08_scope.json",
        "Scores do not give calibrated probabilities that an unrecorded historical event occurred.",
    ),
    (
        10,
        "Formal statistical network models",
        "Not found",
        [],
        "GEE on observations is not an ERGM.",
        "Regularized p1 dyad-independent exponential-family likelihood, held-out dyad comparison, triad adequacy simulations and Bernoulli SBM.",
        "results/05_p1_models.csv;results/05_p1_triad_adequacy.csv;results/05_sbm_selection.csv",
        "Category adjacency only. No historical-person causal ERGM, p2, latent-space, growth or dynamic process fit.",
    ),
    (
        11,
        "Permutation bootstrap and QAP",
        "Partly covered",
        "corpus_clustering_permutations.csv;community_block_bootstrap.csv;title_semantic_grammar_permutation.csv",
        "Extensive page/reign permutations, circular shifts, bootstraps and BH correction existed.",
        "Layer QAP, residual-permutation multiple-matrix QAP, degree/predicate nulls and page bootstrap for new layer effects.",
        "results/02_layer_qap.csv;results/03_semantic_structure_qap.csv;results/06_layer_bootstrap_intervals.csv",
        "Empirical p-value floors and the family of corrected comparisons are documented.",
    ),
    (
        12,
        "Mixing and assortativity",
        "Partly covered",
        "contingency_actor_by_status_event_mode.csv;actor_relation_profiles.csv",
        "Category participation and selected categorical associations existed.",
        "Full category mixing tensor, conditional endpoint randomization and degree assortativity diagnostics.",
        "results/01_observed_and_expected.npz;results/01_conditional_permutation.json;results/07_topology_supplement.json",
        "Shared analytical category is not necessarily shared social identity; no claim of individual homophily.",
    ),
    (
        13,
        "Brokerage and structural holes",
        "Not found as formal measures",
        [],
        "Historical discussion of intermediaries was present; formal Burt or G-F calculations were not found.",
        "Burt constraint/effective size, brokerage centrality, and five G-F brokerage roles with explicit analyst groups.",
        "results/07_brokerage_centrality.csv;results/07_gould_fernandez_brokerage.csv",
        "These measures describe category connectivity, not a measured monopoly held by a person.",
    ),
    (
        14,
        "Hierarchy and core periphery",
        "Partly covered",
        "actor_credential_capacity_profiles.csv;01_spatial_core_periphery.py",
        "Agency concentration, credential/capacity ratios and a geographic core/frontier comparison existed.",
        "Flow/reachability hierarchy, generalized trophic levels, cores/shells, rich-club and nestedness diagnostics; SBM.",
        "results/02_layer_statistics.csv;results/07_trophic_levels.csv;results/07_rich_club.csv;results/07_hierarchy_scope.json",
        "No calibrated dominance ranking; geographic core/periphery is not graph core/periphery.",
    ),
    (
        15,
        "Frequent subgraph pattern mining",
        "Partly covered",
        "strict_named_cluster_membership.csv",
        "Relation cliques and narrative pattern inventories existed, not unrestricted labelled endpoint subgraph mining.",
        "Exact-tag page-scoped frequent edges, outgoing relation forks and length-two labelled paths; small closed patterns via rule supports.",
        "results/04_frequent_paths.csv;results/04_frequent_outgoing_relation_pairs.csv;results/04_length2_horn_rules.csv",
        "No exhaustive closed/maximal general-subgraph miner; node identity is not asserted across pages.",
    ),
    (
        16,
        "Logical and association rules",
        "Not found",
        [],
        "No AMIE-style grounded triple rules with support/head coverage/confidence located.",
        "One-body-edge and two-edge Horn rules with standard/PCA confidence, head coverage and page holdout.",
        "results/04_single_edge_rules.csv;results/04_length2_horn_rules.csv",
        "Custom bounded rule enumeration, not a run of AMIE itself. Low support and weak holdout results are retained.",
    ),
    (
        17,
        "Temporal networks",
        "Partly covered as narrative order",
        "reign_personnel_turnover_by_resolution_spec.csv;page_chronology_full.csv",
        "Reign personnel turnover, chronological composition, page-window order and circular-shift tests existed.",
        "Additional reign-level S-P-O comparison and reign tensor; no event-time paths claimed.",
        "results/06_adjacent_reign_tensor_comparison.csv",
        "Validated event timestamps and within-sentence event order are absent. Page order includes retrospect and is not elapsed time.",
    ),
    (
        18,
        "Relational event models",
        "Not found",
        [],
        "Page-window associations and pre/post activity are not REM likelihood fits.",
        "Not fitted to historical events.",
        "results/06_scope.json",
        "Needs defensible event order, actor identities and a defined set of possible next events. Extraction order and narrative pages cannot supply those without historical annotation.",
    ),
    (
        19,
        "Survival and event history",
        "Not found",
        [],
        "Career span in pages is not duration in office. Earlier pre/post activation was withdrawn.",
        "Not fitted.",
        "results/06_scope.json",
        "No validated office spells, dates of entry/exit, right censoring or population at risk; Kaplan-Meier/Cox estimates would imply unsupported exposure.",
    ),
    (
        20,
        "Dynamic network models",
        "Partly covered descriptively",
        "page_chronology_full.csv;community_stability_summary.csv",
        "Narrative composition changes and community stability existed; no historical TERGM/SAOM/dynamic SBM located.",
        "Reign-segment distribution comparisons and four-mode factors.",
        "results/06_adjacent_reign_tensor_comparison.csv;results/03_reign_cp_loadings.csv",
        "Chronicle mentions are not repeated complete network states. No process model fitted.",
    ),
    (
        21,
        "Tensor through time",
        "Not found",
        [],
        "Separate reign frequency charts are not four-mode tensor models.",
        "Five-component nonnegative subject-relation-object-reign tensor normalized per 1000 claims within segment.",
        "results/03_reign_cp_loadings.csv;results/03_reign_cp_fit.json",
        "The fourth mode is 12 narrative reign/crisis segments, not dated event time. No causal temporal factor dynamics claimed.",
    ),
    (
        22,
        "Text and graph",
        "Already extensively covered",
        "relation_embedding_analysis_v3.py;title_semantic_grammar_permutation.csv;03_burmese_royal_marking.py",
        "Burmese honorific/morphological tests, predicate formulaicity, lexical/context analysis and Gemini semantic structure existed.",
        "Base/context/fused semantic-to-endpoint-structure QAP, within-domain and frequency/shared-predicate controls.",
        "results/03_semantic_structure_qap.csv;results/03_semantic_structure_pairs.csv",
        "Context embeddings include the same Burmese/English source context; agreement is internal consistency, not independent validation.",
    ),
    (
        23,
        "Spatial networks",
        "Partly covered",
        "01_spatial_core_periphery.py;02_gradient_interaction.py",
        "A transparent place-string gazetteer and core/province/frontier agency tests already existed.",
        "No coordinate model added.",
        "",
        "No validated coordinates linked to the canonical entity/mention graph. Moran/Geary/gravity/distance/physical-flow models need geolocation and an explicit spatial weights/risk-set definition.",
    ),
    (
        24,
        "Robustness and measurement uncertainty",
        "Already extensively covered",
        "community_block_bootstrap.csv;gee_sensitivity.csv;title_concentration_chronology_sensitivity.csv",
        "Sentence/page/reign controls, density matching, semantic/category/identity/window alternatives, manual source corrections and FDR existed.",
        "New page bootstraps, king/status/register exclusions, CP seed and label-null sensitivity, numerical/source invariants.",
        "results/06_layer_bootstrap_intervals.csv;results/06_tensor_component_stability.csv;results/09_category_conditioned_chain_null.csv",
        "No calibrated triple extraction confidence or independent full-corpus reliability sample is available for confidence weighting/posterior measurement-error models.",
    ),
    (
        25,
        "Comparative networks",
        "Partly covered",
        "relation_by_reign.csv;reign_trend_permutation_tests.csv;01_spatial_core_periphery.py",
        "Reign composition, core/frontier comparisons and trend/permutation tests existed.",
        "Layer topology comparison, QAP coupling, whole S-P-O adjacent-reign JS divergence with page permutations.",
        "results/02_layer_statistics.csv;results/06_adjacent_reign_tensor_comparison.csv",
        "Corpus attention and exposure differ across reigns; null results for small segments are reported.",
    ),
    (
        26,
        "Predictive representation learning",
        "Partly covered",
        "title_embedding_classifier_summary.csv;relation_embedding_analysis_v3.py",
        "Gemini vectors, nearest neighbors, semantic communities and title classifiers existed. These are not trained graph embeddings.",
        "DistMult and ComplEx with disjoint category-fact train/validation/test sets and a frequency baseline.",
        "results/08_link_prediction.csv;results/08_training_trace.csv",
        "No TransE/RotatE/GNN/DeepWalk benchmark. The two added predictors underperform the simple baseline, so no missing historical facts are imputed.",
    ),
]
rows = []
for f, title, before, names, prior, new, outputs, limits in catalog:
    names = names.split(";") if isinstance(names, str) else names
    rows.append(
        dict(
            family=f,
            family_name=title,
            prior_status=before,
            prior_evidence=evidence(names),
            prior_coverage=prior,
            new_work=new,
            new_outputs=outputs,
            remaining_limits=limits,
        )
    )
families = pd.DataFrame(rows)
save(families, "10_family_audit")

# Specific method coverage; unmatched alternatives stay visibly unrun.
old = {
    1: r"number of triples|properties/predicates|classes|property usage|property use|triples per instance|incoming|outgoing|co-occurrence|class distributions|frequency|sovereign|subject/object|relation diversity|unique subjects|participation",
    2: r"marginal|conditional distributions|pairwise independence|χ²|odds ratios|association parameters|model comparison",
    3: r"^(in-degree|out-degree|total degree|weighted degree / strength|degree centrality|PageRank|connected components)$",
    6: r"modularity|Louvain|clique communities|overlapping communities",
    11: r"node-label permutation|temporal randomization|constrained permutation|bootstrap|empirical p|Z-score|confidence interval|effect size",
    12: r"categorical mixing|conditional mixing|permutation tests",
    14: r"by reign|by actor class",
    15: r"frequent edges",
    17: r"temporal correlation|persistence|edge turnover|entropy|temporal randomization",
    20: r"network change-point detection",
    22: r"predicate × lexical|entity × lexical|event type × surrounding|relation × discourse|predicate co-occurrence|multilevel models",
    23: r"relation frequency by space|regional network comparison|geographically constrained permutation",
    24: r"bootstrap by sentence|bootstrap by page|bootstrap by episode|bootstrap by reign|leave-one-document|leave-one-reign|entity-resolution|alternate entity|predicate-collapse|edge-weight|alternate time-window|alternate null|multiple-testing|community structure|centrality ranking",
    25: r"relation composition|permutation|bootstrap|distributional|Jensen|community structure",
}
new = {
    2: r"conditional independence|three-way interaction|likelihood-ratio|log-linear|Pearson residuals|observed/expected|model comparison|residual-based|iterative proportional",
    3: r".*",
    4: r"^(?!squares|graphlets|larger local).*",
    5: r"^(?!simple paths|path-constrained|relation-path similarity).*",
    6: r"spectral clustering|structural equivalence|blockmodeling|stochastic blockmodels|role equivalence|core/periphery",
    7: r"layer density|layer-specific centrality|layer overlap|edge overlap|degree correlation|inter-layer dependence|participation|cross-layer triangles|cross-layer motifs|layer similarity",
    8: r"CP /|Tucker$|Tucker3|non-negative|actor groups|recipient groups|relation families",
    9: r"latent relational models|matrix/tensor|DistMult",
    10: r"p_1|fixed-degree|stochastic blockmodels|reciprocity|transitivity|triangles|degree|particular structural",
    11: r"node-label|edge rewiring|degree-preserving|configuration-model|relation-label|QAP|MRQAP|constrained|bootstrap|empirical|Z-score|confidence|effect",
    12: r"assortativity|categorical mixing|degree assortativity|conditional mixing|permutation",
    13: r"Burt|effective size|efficiency|brokerage|Gould",
    14: r"hierarchy measures|flow hierarchy|trophic|reachability hierarchy|cycle-based|core/periphery|coreness|k-core|k-shell|rich club|nestedness|globally|by predicate|by actor class",
    15: r"frequent edges|frequent paths|frequent trees|frequent subgraphs|labelled subgraphs",
    16: r".*",
    21: r"temporal CP|non-negative|latent-factor trajectories",
    22: r"graph structure × textual|relation communities × sentence|structural role × lexical",
    24: r"bootstrap by page|predicate-collapse|edge-weight|alternate null|multiple-testing|community structure|centrality ranking",
    25: r"density|centralization|reciprocity|transitivity|assortativity|degree distribution|path structure|centrality distribution|community structure|motif profile|relation composition|multiplex coupling|block structure|hierarchy|core/periphery|permutation|bootstrap|distributional|graph distances|Jensen|model-parameter",
    26: r"^DistMult$|^ComplEx$|neural link prediction",
}
text = (O / "inputs/requested_method_catalogue.txt").read_text(encoding="utf-8")
text = text[text.index("# 1. Native") : text.index("# The master map")]
methodrows = []
family = 0
for line in text.splitlines():
    m = re.match(r"#+ (\d+)\. ", line)
    if m:
        family = int(m[1])
    if not line.startswith("* "):
        continue
    method = line[2:].replace("**", "").strip()
    row = rows[family - 1]
    status = "Not run alternative"
    detail = "Named variant not separately executed; see exact family coverage and outputs."
    if family in old and re.search(old[family], method, re.I):
        status = "Prior evidence located"
        detail = row["prior_coverage"]
    if family in new and re.search(new[family], method, re.I):
        status = "New analysis executed"
        detail = row["new_work"]
    if (
        family in [18, 19]
        or (family == 17 and status == "Not run alternative")
        or (family == 20 and status == "Not run alternative")
        or (family == 23 and status == "Not run alternative")
    ):
        status = "Data prerequisites missing"
        detail = row["remaining_limits"]
    if family == 24 and re.search(
        "extraction-confidence|posterior uncertainty|relation-classification uncertainty", method
    ):
        status = "Data prerequisites missing"
        detail = row["remaining_limits"]
    if family == 1 and status == "Not run alternative":
        status = "Not a variable in the canonical analysis table"
        detail = "URI/literal/datatype/numeric-measure profiling is not represented as a varying substantive field in the canonical tag table; original JSON/N-Quads metadata retained."
    if family == 3 and method == "assortativity":
        detail = "Added degree assortativity in 07_topology_supplement.json."
    if family == 4 and method == "squares":
        status = "New analysis executed"
        detail = "Undirected category four-cycle count; 07_topology_supplement.json."
    if family == 10 and method == "exponential random graph models":
        status = "Partial new analysis"
        detail = "p1 dyad-independent exponential-family special case only. No dependent-tie historical ERGM."
    if family in [6, 13] and re.search(
        "role equivalence|structural equivalence|bridging centrality|brokerage$", method
    ):
        detail += " Similarity/Burt/betweenness diagnostics; no claim of exact equivalence or a separate bridging-centrality formula."
    if family == 22 and re.search("sentence embeddings", method):
        status = "Partial new analysis"
        detail = "Occurrence-context relation centroids include Burmese and English, not an independent sentence-only embedding model."
    if family == 21 and status == "New analysis executed":
        detail += " Fourth mode is coarse narrative reign segment."
    if family == 6 and method == "spectral clustering":
        status = "Not run alternative"
        detail = "K-means on directional relation profiles and fitted SBM were run; no separate spectral clustering estimator."
    if family == 6 and method in ["structural equivalence", "role equivalence"]:
        status = "Partial new analysis"
        detail = "Directional predicate-profile similarity and role clustering; not an exact equivalence relation on actors."
    if family == 15 and method == "frequent trees":
        status = "Partial new analysis"
        detail = "Two-edge paths and outgoing relation-pair forks only; no general tree miner."
    if family == 22 and method == "structural role × lexical profile":
        status = "Not run alternative"
        detail = "New QAP compares relation semantic centroids with relation endpoint structure, not actor-role lexical profiles."
    methodrows.append(
        dict(
            family=family,
            family_name=row["family_name"],
            requested_method=method,
            status=status,
            detail=detail,
            prior_evidence=row["prior_evidence"],
            new_outputs=row["new_outputs"],
        )
    )
save(pd.DataFrame(methodrows), "10_requested_method_checklist")
# Preserve exact search evidence for absence audits, with line numbers and source paths.
patterns = r"log.linear|iterative proportional|triadic_census|degree.preserving|\bQAP\b|MRQAP|\bERGM\b|stochastic.block|\bRESCAL\b|\bTucker\b|PARAFAC|DistMult|ComplEx|relational.event.model|Kaplan|\bCox\b|Burt|effective_size|pagerank|betweenness|chi2_contingency"
hits = []
seen = set()
for r in inv:
    if (
        r["extension"] not in [".py", ".md", ".docx", ".txt"]
        or r["text_file"] in seen
        or not r["text_file"]
    ):
        continue
    seen.add(r["text_file"])
    for i, line in enumerate((O / r["text_file"]).read_text(encoding="utf-8").splitlines()):
        if re.search(patterns, line, re.I):
            hits.append(
                dict(
                    source=r["source"], extracted_text=r["text_file"], line=i + 1, text=line[:1000]
                )
            )
save(pd.DataFrame(hits), "10_archive_method_search_hits")
summary = dict(
    inventory_entries=len(inv),
    top_level_zip_files=sum(r["extension"] == ".zip" and "!/" not in r["source"] for r in inv),
    archived_and_loose_docx=sum(r["extension"] == ".docx" for r in inv),
    script_occurrences=sum(r["extension"] == ".py" for r in inv),
    unique_scripts=len({r["sha256"] for r in inv if r["extension"] == ".py"}),
    families=len(rows),
    requested_bullet_items=len(methodrows),
    method_status_counts=dict(Counter(r["status"] for r in methodrows)),
    family_prior_status_counts=dict(Counter(r["prior_status"] for r in rows)),
    scope="All 26 families audited. Named algorithms within a family are alternatives, not assumed complete from one representative fit. The method checklist records unrun variants explicitly. Original project source was searched for additional advanced implementations; its recovered originals and validation assets were used, without treating extraction trials as statistical historical results.",
)
jsave(summary, "10_audit_summary")
print(json.dumps(summary, indent=2))

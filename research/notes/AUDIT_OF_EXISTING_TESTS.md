# Audit of existing Konbaung analyses

The supplied catalogue contains **26 families and 380 bullet items**, including overlapping metrics, alternative algorithms and proposed applications. The audit found extensive earlier work, added a historically focused battery of missing analyses, and records the unrun alternatives explicitly. Coverage of a family does not imply that every named algorithm in it has been executed.

## What was inspected

The article folder contains **14 top-level ZIP packages**. Including nested members and loose files, the inventory has **3,350 entries**, **70 Word-document occurrences**, **154 Python-script occurrences representing 127 distinct scripts**, and 29 PDFs (117 pages of extracted text). Repeated archive copies are not counted as independent studies. Reports, scripts, result tables, correction notes and withdrawal lists were cross-checked. A keyword hit was used to locate evidence, not to certify a completed test. The original project was also searched for the missing advanced implementations and used to retrieve the intact source.

The primary source for the new work is `../../DIGHUM_WEBGPT_ANALYSIS_PACKAGE_20260831.zip` in the original `dighumproject` directory. The copy nested in the large final reproducibility package is truncated. Forty-one complete members were initially recovered with CRC and manifest checks; the intact original subsequently supplied the missing relation context/fused matrices. The original and recovered canonical triples, sentences, taxonomy and base vectors match byte for byte. No original inputs were edited.

## Coverage by family

| family | family_name | prior_status | new_work |
|---|---|---|---|
| 1 | Native triple and RDF profiling | Already covered | Data revalidated against intact original source. |
| 2 | Three-way categorical analysis | Partly covered | Mutual independence, SP/PO conditional independence, all-pairwise IPF, Pearson residuals and a page/predicate constrained permutation. |
| 3 | Directed graph topology and centrality | Partly covered | Directed category centralities and connectivity, paths, cuts, walks, named-person graph extensions. |
| 4 | Dyads triads and motifs | Not found as network tests | Typed chains/transitive/cyclic patterns with page-label nulls; stricter page/category null; unlabelled triads against two degree-preserving MCMC chains. |
| 5 | Paths reachability and flow | Not found as a dedicated family | Shortest/weighted paths, reachability, typed page-scoped paths, hitting/commute times, cuts, removal sensitivity. |
| 6 | Communities positions and roles | Partly covered | Predicate-specific directional role profiles, category communities, an explicit Bernoulli SBM and seed checks. |
| 7 | Multiplex relation layers | Not found as actual S-to-O layers | All 81 layers, overlap, centrality, participation, 3240 descriptive layer comparisons, 15 prespecified QAP comparisons and sensitivities. |
| 8 | Tensor and multiway decomposition | Not found | Nonnegative CP rank 3/5/7 selection, full rank-7 descriptive fit, Tucker 8/8/8 and seed/register/status sensitivities. |
| 9 | Statistical relational learning | Not found as a fitted triple model | Held-out category-fact DistMult and ComplEx, plus frequency baseline; page-held-out CP selection. |
| 10 | Formal statistical network models | Not found | Regularized p1 dyad-independent exponential-family likelihood, held-out dyad comparison, triad adequacy simulations and Bernoulli SBM. |
| 11 | Permutation bootstrap and QAP | Partly covered | Layer QAP, residual-permutation multiple-matrix QAP, degree/predicate nulls and page bootstrap for new layer effects. |
| 12 | Mixing and assortativity | Partly covered | Full category mixing tensor, conditional endpoint randomization and degree assortativity diagnostics. |
| 13 | Brokerage and structural holes | Not found as formal measures | Burt constraint/effective size, brokerage centrality, and five G-F brokerage roles with explicit analyst groups. |
| 14 | Hierarchy and core periphery | Partly covered | Flow/reachability hierarchy, generalized trophic levels, cores/shells, rich-club and nestedness diagnostics; SBM. |
| 15 | Frequent subgraph pattern mining | Partly covered | Exact-tag page-scoped frequent edges, outgoing relation forks and length-two labelled paths; small closed patterns via rule supports. |
| 16 | Logical and association rules | Not found | One-body-edge and two-edge Horn rules with standard/PCA confidence, head coverage and page holdout. |
| 17 | Temporal networks | Partly covered as narrative order | Additional reign-level S-P-O comparison and reign tensor; no event-time paths claimed. |
| 18 | Relational event models | Not found | Not fitted to historical events. |
| 19 | Survival and event history | Not found | Not fitted. |
| 20 | Dynamic network models | Partly covered descriptively | Reign-segment distribution comparisons and four-mode factors. |
| 21 | Tensor through time | Not found | Five-component nonnegative subject-relation-object-reign tensor normalized per 1000 claims within segment. |
| 22 | Text and graph | Already extensively covered | Base/context/fused semantic-to-endpoint-structure QAP, within-domain and frequency/shared-predicate controls. |
| 23 | Spatial networks | Partly covered | No coordinate model added. |
| 24 | Robustness and measurement uncertainty | Already extensively covered | New page bootstraps, king/status/register exclusions, CP seed and label-null sensitivity, numerical/source invariants. |
| 25 | Comparative networks | Partly covered | Layer topology comparison, QAP coupling, whole S-P-O adjacent-reign JS divergence with page permutations. |
| 26 | Predictive representation learning | Partly covered | DistMult and ComplEx with disjoint category-fact train/validation/test sets and a frequency baseline. |

## Exact previous evidence and new outputs

Each entry below names the archived evidence rather than relying on a report title. `!/` separates an archive from its member path.

### 1 Native triple and RDF profiling

**Before:** Already covered. Canonical counts, frequency and category participation, sovereign roles and relation diversity. RDF export metadata are not an additional historical test.

**Located evidence:**

- `Konbaung_Power_Relations_Evidence_Package.zip!/Konbaung_Power_Relations_Evidence_Package/02_ARCHIVED_RAW_RESULTS/relation_frequency.csv`
- `Konbaung_Power_Relations_Evidence_Package.zip!/Konbaung_Power_Relations_Evidence_Package/02_ARCHIVED_RAW_RESULTS/relation_royal_subject_and_actor_diversity.csv`
- `Konbaung_Power_Relations_Evidence_Package.zip!/Konbaung_Power_Relations_Evidence_Package/02_ARCHIVED_RAW_RESULTS/actor_relation_profiles.csv`

**New work:** Data revalidated against intact original source.

**Outputs:** [data_validation.json](data_validation.json).

### 2 Three-way categorical analysis

**Before:** Partly covered. Pairwise chi-square/Cramer V, odds ratios, GEE and conditional contrasts existed. No full S-P-O loglinear fit located.

**Located evidence:**

- `Konbaung_Power_Relations_Evidence_Package.zip!/Konbaung_Power_Relations_Evidence_Package/01_ARCHIVED_SCRIPTS/reproduce_derived_tests.py`
- `Konbaung_46_Episode_Evidence_Package.zip!/Konbaung_46_Episode_Evidence/inputs/oriented_revocations_actor_analysis_clean.csv`

**New work:** Mutual independence, SP/PO conditional independence, all-pairwise IPF, Pearson residuals and a page/predicate constrained permutation.

**Outputs:** [01_loglinear_models.csv](results/01_loglinear_models.csv), [01_conditional_permutation.json](results/01_conditional_permutation.json), [01_threeway_cells.csv](results/01_threeway_cells.csv).

**Boundary:** Three-way residuals are descriptive; no asymptotic chi-square p-values for sparse cells. The randomization tests S-O association given P/page, not the isolated three-way interaction.

### 3 Directed graph topology and centrality

**Before:** Partly covered. Named-person network already had undirected degree/strength, PageRank and components.

**Located evidence:**

- `Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED (2).zip!/Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED/09_ENTITY_RESOLUTION_ANALYSIS/04_OUTPUTS/named_person_network_centrality.csv`
- `Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED (2).zip!/Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED/09_ENTITY_RESOLUTION_ANALYSIS/04_OUTPUTS/named_person_network_edges.csv`

**New work:** Directed category centralities and connectivity, paths, cuts, walks, named-person graph extensions.

**Outputs:** [02_category_centralities.csv](results/02_category_centralities.csv), [02_named_person_topology.json](results/02_named_person_topology.json), [07_topology_supplement.json](results/07_topology_supplement.json).

**Boundary:** Category graphs retain extraction direction; named-person graph remains the archived conservative undirected graph.

### 4 Dyads triads and motifs

**Before:** Not found as network tests. Burmese orthographic motifs and relation cliques are present, but are different objects.

**New work:** Typed chains/transitive/cyclic patterns with page-label nulls; stricter page/category null; unlabelled triads against two degree-preserving MCMC chains.

**Outputs:** [04_labelled_motif_tests.csv](results/04_labelled_motif_tests.csv), [09_category_conditioned_chain_null.csv](results/09_category_conditioned_chain_null.csv), [04_degree_preserving_triads.csv](results/04_degree_preserving_triads.csv).

**Boundary:** No exhaustive census of every larger graphlet size.

### 5 Paths reachability and flow

**Before:** Not found as a dedicated family. Person career order and adjacent page windows are not graph-path analysis.

**New work:** Shortest/weighted paths, reachability, typed page-scoped paths, hitting/commute times, cuts, removal sensitivity.

**Outputs:** [07_all_pairs_paths.csv](results/07_all_pairs_paths.csv), [04_frequent_paths.csv](results/04_frequent_paths.csv), [07_random_walk_hitting_times.csv](results/07_random_walk_hitting_times.csv).

**Boundary:** Graph walk steps and inverse-frequency distances are mathematical diagnostics, not measured travel, time or resources.

### 6 Communities positions and roles

**Before:** Partly covered. Page relation communities, semantic clusters, Louvain, clique and seed/resolution checks existed.

**Located evidence:**

- `Konbaung_Complete_Narrative_Analysis.zip!/Konbaung_Narrative_Atlas_Complete/results/community_seed_resolution_audit.csv`
- `Konbaung_Complete_Narrative_Analysis.zip!/Konbaung_Narrative_Atlas_Complete/results/community_stability_summary.csv`
- `Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED (2).zip!/Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED/04_EMBEDDING_ANALYSIS/06_OUTPUTS/tables/relation_category_centered_k8.csv`

**New work:** Predicate-specific directional role profiles, category communities, an explicit Bernoulli SBM and seed checks.

**Outputs:** [02_profile_roles.csv](results/02_profile_roles.csv), [05_sbm_memberships.csv](results/05_sbm_memberships.csv), [05_sbm_selection.csv](results/05_sbm_selection.csv).

**Boundary:** Leiden/Infomap and exact regular or automorphic equivalence were not run; profile similarity is not exact regular equivalence.

### 7 Multiplex relation layers

**Before:** Not found as actual S-to-O layers. Existing relation co-occurrence graphs connect predicates to predicates; they are not 81 endpoint-adjacency layers.

**New work:** All 81 layers, overlap, centrality, participation, 3240 descriptive layer comparisons, 15 prespecified QAP comparisons and sensitivities.

**Outputs:** [02_layer_statistics.csv](results/02_layer_statistics.csv), [02_all_layer_pairs.csv](results/02_all_layer_pairs.csv), [02_layer_qap.csv](results/02_layer_qap.csv), [06_layer_qap_sensitivity.csv](results/06_layer_qap_sensitivity.csv).

**Boundary:** No supra-adjacency community/centrality fit or multilayer SBM.

### 8 Tensor and multiway decomposition

**Before:** Not found. Embedding clustering was not S-P-O tensor decomposition.

**New work:** Nonnegative CP rank 3/5/7 selection, full rank-7 descriptive fit, Tucker 8/8/8 and seed/register/status sensitivities.

**Outputs:** [03_tensor_model_selection.csv](results/03_tensor_model_selection.csv), [03_static_cp_loadings.csv](results/03_static_cp_loadings.csv), [03_tucker_fit.json](results/03_tucker_fit.json), [06_tensor_component_stability.csv](results/06_tensor_component_stability.csv).

**Boundary:** RESCAL, DEDICOM and coupled tensor/matrix variants remain unrun alternatives.

### 9 Statistical relational learning

**Before:** Not found as a fitted triple model. Title-category embedding classifiers existed, but not joint S-P-O completion models.

**Located evidence:**

- `Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED (2).zip!/Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED/04_EMBEDDING_ANALYSIS/06_OUTPUTS/tables/title_embedding_classifier_summary.csv`

**New work:** Held-out category-fact DistMult and ComplEx, plus frequency baseline; page-held-out CP selection.

**Outputs:** [08_link_prediction.csv](results/08_link_prediction.csv), [08_scope.json](results/08_scope.json).

**Boundary:** Scores do not give calibrated probabilities that an unrecorded historical event occurred.

### 10 Formal statistical network models

**Before:** Not found. GEE on observations is not an ERGM.

**New work:** Regularized p1 dyad-independent exponential-family likelihood, held-out dyad comparison, triad adequacy simulations and Bernoulli SBM.

**Outputs:** [05_p1_models.csv](results/05_p1_models.csv), [05_p1_triad_adequacy.csv](results/05_p1_triad_adequacy.csv), [05_sbm_selection.csv](results/05_sbm_selection.csv).

**Boundary:** Category adjacency only. No historical-person causal ERGM, p2, latent-space, growth or dynamic process fit.

### 11 Permutation bootstrap and QAP

**Before:** Partly covered. Extensive page/reign permutations, circular shifts, bootstraps and BH correction existed.

**Located evidence:**

- `Konbaung_46_Episode_Statistical_Evidence.zip!/Konbaung_46_Episode_Statistical_Evidence/results/corpus_clustering_permutations.csv`
- `Konbaung_Complete_Narrative_Analysis.zip!/Konbaung_Narrative_Atlas_Complete/results/community_block_bootstrap.csv`
- `Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED (2).zip!/Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED/04_EMBEDDING_ANALYSIS/06_OUTPUTS/extension/title_semantic_grammar_permutation.csv`

**New work:** Layer QAP, residual-permutation multiple-matrix QAP, degree/predicate nulls and page bootstrap for new layer effects.

**Outputs:** [02_layer_qap.csv](results/02_layer_qap.csv), [03_semantic_structure_qap.csv](results/03_semantic_structure_qap.csv), [06_layer_bootstrap_intervals.csv](results/06_layer_bootstrap_intervals.csv).

**Boundary:** Empirical p-value floors and the family of corrected comparisons are documented.

### 12 Mixing and assortativity

**Before:** Partly covered. Category participation and selected categorical associations existed.

**Located evidence:**

- `Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED (2).zip!/Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED/03_BASE_STATISTICAL_ANALYSIS/06_REPRODUCED_OUTPUTS/contingency_actor_by_status_event_mode.csv`
- `Konbaung_Power_Relations_Evidence_Package.zip!/Konbaung_Power_Relations_Evidence_Package/02_ARCHIVED_RAW_RESULTS/actor_relation_profiles.csv`

**New work:** Full category mixing tensor, conditional endpoint randomization and degree assortativity diagnostics.

**Outputs:** [01_observed_and_expected.npz](results/01_observed_and_expected.npz), [01_conditional_permutation.json](results/01_conditional_permutation.json), [07_topology_supplement.json](results/07_topology_supplement.json).

**Boundary:** Shared analytical category is not necessarily shared social identity; no claim of individual homophily.

### 13 Brokerage and structural holes

**Before:** Not found as formal measures. Historical discussion of intermediaries was present; formal Burt or G-F calculations were not found.

**New work:** Burt constraint/effective size, brokerage centrality, and five G-F brokerage roles with explicit analyst groups.

**Outputs:** [07_brokerage_centrality.csv](results/07_brokerage_centrality.csv), [07_gould_fernandez_brokerage.csv](results/07_gould_fernandez_brokerage.csv).

**Boundary:** These measures describe category connectivity, not a measured monopoly held by a person.

### 14 Hierarchy and core periphery

**Before:** Partly covered. Agency concentration, credential/capacity ratios and a geographic core/frontier comparison existed.

**Located evidence:**

- `Konbaung_Power_Relations_Evidence_Package.zip!/Konbaung_Power_Relations_Evidence_Package/02_ARCHIVED_RAW_RESULTS/actor_credential_capacity_profiles.csv`
- `Konbaung_Reproduction_Package.zip!/repro/src/01_spatial_core_periphery.py`

**New work:** Flow/reachability hierarchy, generalized trophic levels, cores/shells, rich-club and nestedness diagnostics; SBM.

**Outputs:** [02_layer_statistics.csv](results/02_layer_statistics.csv), [07_trophic_levels.csv](results/07_trophic_levels.csv), [07_rich_club.csv](results/07_rich_club.csv), [07_hierarchy_scope.json](results/07_hierarchy_scope.json).

**Boundary:** No calibrated dominance ranking; geographic core/periphery is not graph core/periphery.

### 15 Frequent subgraph pattern mining

**Before:** Partly covered. Relation cliques and narrative pattern inventories existed, not unrestricted labelled endpoint subgraph mining.

**Located evidence:**

- `Konbaung_Chat_Analysis_Reproducibility (1).zip!/Konbaung_Chat_Analysis_Reproducibility/results/01_strict_relation_cliques/strict_named_cluster_membership.csv`

**New work:** Exact-tag page-scoped frequent edges, outgoing relation forks and length-two labelled paths; small closed patterns via rule supports.

**Outputs:** [04_frequent_paths.csv](results/04_frequent_paths.csv), [04_frequent_outgoing_relation_pairs.csv](results/04_frequent_outgoing_relation_pairs.csv), [04_length2_horn_rules.csv](results/04_length2_horn_rules.csv).

**Boundary:** No exhaustive closed/maximal general-subgraph miner; node identity is not asserted across pages.

### 16 Logical and association rules

**Before:** Not found. No AMIE-style grounded triple rules with support/head coverage/confidence located.

**New work:** One-body-edge and two-edge Horn rules with standard/PCA confidence, head coverage and page holdout.

**Outputs:** [04_single_edge_rules.csv](results/04_single_edge_rules.csv), [04_length2_horn_rules.csv](results/04_length2_horn_rules.csv).

**Boundary:** Custom bounded rule enumeration, not a run of AMIE itself. Low support and weak holdout results are retained.

### 17 Temporal networks

**Before:** Partly covered as narrative order. Reign personnel turnover, chronological composition, page-window order and circular-shift tests existed.

**Located evidence:**

- `Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED (2).zip!/Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED/09_ENTITY_RESOLUTION_ANALYSIS/04_OUTPUTS/reign_personnel_turnover_by_resolution_spec.csv`
- `Konbaung_Complete_Narrative_Analysis.zip!/Konbaung_Narrative_Atlas_Complete/results/page_chronology_full.csv`

**New work:** Additional reign-level S-P-O comparison and reign tensor; no event-time paths claimed.

**Outputs:** [06_adjacent_reign_tensor_comparison.csv](results/06_adjacent_reign_tensor_comparison.csv).

**Boundary:** Validated event timestamps and within-sentence event order are absent. Page order includes retrospect and is not elapsed time.

### 18 Relational event models

**Before:** Not found. Page-window associations and pre/post activity are not REM likelihood fits.

**New work:** Not fitted to historical events.

**Outputs:** [06_scope.json](results/06_scope.json).

**Boundary:** Needs defensible event order, actor identities and a defined set of possible next events. Extraction order and narrative pages cannot supply those without historical annotation.

### 19 Survival and event history

**Before:** Not found. Career span in pages is not duration in office. Earlier pre/post activation was withdrawn.

**New work:** Not fitted.

**Outputs:** [06_scope.json](results/06_scope.json).

**Boundary:** No validated office spells, dates of entry/exit, right censoring or population at risk; Kaplan-Meier/Cox estimates would imply unsupported exposure.

### 20 Dynamic network models

**Before:** Partly covered descriptively. Narrative composition changes and community stability existed; no historical TERGM/SAOM/dynamic SBM located.

**Located evidence:**

- `Konbaung_Complete_Narrative_Analysis.zip!/Konbaung_Narrative_Atlas_Complete/results/page_chronology_full.csv`
- `Konbaung_Complete_Narrative_Analysis.zip!/Konbaung_Narrative_Atlas_Complete/results/community_stability_summary.csv`

**New work:** Reign-segment distribution comparisons and four-mode factors.

**Outputs:** [06_adjacent_reign_tensor_comparison.csv](results/06_adjacent_reign_tensor_comparison.csv), [03_reign_cp_loadings.csv](results/03_reign_cp_loadings.csv).

**Boundary:** Chronicle mentions are not repeated complete network states. No process model fitted.

### 21 Tensor through time

**Before:** Not found. Separate reign frequency charts are not four-mode tensor models.

**New work:** Five-component nonnegative subject-relation-object-reign tensor normalized per 1000 claims within segment.

**Outputs:** [03_reign_cp_loadings.csv](results/03_reign_cp_loadings.csv), [03_reign_cp_fit.json](results/03_reign_cp_fit.json).

**Boundary:** The fourth mode is 12 narrative reign/crisis segments, not dated event time. No causal temporal factor dynamics claimed.

### 22 Text and graph

**Before:** Already extensively covered. Burmese honorific/morphological tests, predicate formulaicity, lexical/context analysis and Gemini semantic structure existed.

**Located evidence:**

- `Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED (2).zip!/Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED/04_EMBEDDING_ANALYSIS/04_SCRIPTS/relation_embedding_analysis_v3.py`
- `Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED (2).zip!/Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED/04_EMBEDDING_ANALYSIS/06_OUTPUTS/extension/title_semantic_grammar_permutation.csv`
- `Konbaung_Reproduction_Package.zip!/repro/src/03_burmese_royal_marking.py`

**New work:** Base/context/fused semantic-to-endpoint-structure QAP, within-domain and frequency/shared-predicate controls.

**Outputs:** [03_semantic_structure_qap.csv](results/03_semantic_structure_qap.csv), [03_semantic_structure_pairs.csv](results/03_semantic_structure_pairs.csv).

**Boundary:** Context embeddings include the same Burmese/English source context; agreement is internal consistency, not independent validation.

### 23 Spatial networks

**Before:** Partly covered. A transparent place-string gazetteer and core/province/frontier agency tests already existed.

**Located evidence:**

- `Konbaung_Reproduction_Package.zip!/repro/src/01_spatial_core_periphery.py`
- `Konbaung_Reproduction_Package.zip!/repro/src/02_gradient_interaction.py`

**New work:** No coordinate model added.

**Boundary:** No validated coordinates linked to the canonical entity/mention graph. Moran/Geary/gravity/distance/physical-flow models need geolocation and an explicit spatial weights/risk-set definition.

### 24 Robustness and measurement uncertainty

**Before:** Already extensively covered. Sentence/page/reign controls, density matching, semantic/category/identity/window alternatives, manual source corrections and FDR existed.

**Located evidence:**

- `Konbaung_Complete_Narrative_Analysis.zip!/Konbaung_Narrative_Atlas_Complete/results/community_block_bootstrap.csv`
- `Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED (2).zip!/Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED/08_NEW_STRUCTURAL_AUDITS/07_INDEPENDENT_AUDIT/custom_audit_outputs/gee_sensitivity.csv`
- `Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED (2).zip!/Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED/09_ENTITY_RESOLUTION_ANALYSIS/04_OUTPUTS/title_concentration_chronology_sensitivity.csv`

**New work:** New page bootstraps, king/status/register exclusions, CP seed and label-null sensitivity, numerical/source invariants.

**Outputs:** [06_layer_bootstrap_intervals.csv](results/06_layer_bootstrap_intervals.csv), [06_tensor_component_stability.csv](results/06_tensor_component_stability.csv), [09_category_conditioned_chain_null.csv](results/09_category_conditioned_chain_null.csv).

**Boundary:** No calibrated triple extraction confidence or independent full-corpus reliability sample is available for confidence weighting/posterior measurement-error models.

### 25 Comparative networks

**Before:** Partly covered. Reign composition, core/frontier comparisons and trend/permutation tests existed.

**Located evidence:**

- `Konbaung_Chronological_Trends_Evidence.zip!/Konbaung_Chronological_Trends_Evidence/results/relation_by_reign.csv`
- `Konbaung_Chronological_Trends_Evidence.zip!/Konbaung_Chronological_Trends_Evidence/results/reign_trend_permutation_tests.csv`
- `Konbaung_Reproduction_Package.zip!/repro/src/01_spatial_core_periphery.py`

**New work:** Layer topology comparison, QAP coupling, whole S-P-O adjacent-reign JS divergence with page permutations.

**Outputs:** [02_layer_statistics.csv](results/02_layer_statistics.csv), [06_adjacent_reign_tensor_comparison.csv](results/06_adjacent_reign_tensor_comparison.csv).

**Boundary:** Corpus attention and exposure differ across reigns; null results for small segments are reported.

### 26 Predictive representation learning

**Before:** Partly covered. Gemini vectors, nearest neighbors, semantic communities and title classifiers existed. These are not trained graph embeddings.

**Located evidence:**

- `Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED (2).zip!/Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED/04_EMBEDDING_ANALYSIS/06_OUTPUTS/tables/title_embedding_classifier_summary.csv`
- `Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED (2).zip!/Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED/04_EMBEDDING_ANALYSIS/04_SCRIPTS/relation_embedding_analysis_v3.py`

**New work:** DistMult and ComplEx with disjoint category-fact train/validation/test sets and a frequency baseline.

**Outputs:** [08_link_prediction.csv](results/08_link_prediction.csv), [08_training_trace.csv](results/08_training_trace.csv).

**Boundary:** No TransE/RotatE/GNN/DeepWalk benchmark. The two added predictors underperform the simple baseline, so no missing historical facts are imputed.

## Previously attempted but withdrawn or qualified

The earlier archives explicitly withdraw or limit several attractive results. They remain classified as attempted, not missing:

- **Titling causes subsequent activity:** exposure imbalance and identity uncertainty undermine the earlier pre/post activation result. Do not reinstate it from page order.
- **Nearest-prior-title semantic continuity:** equal-sized random prior sets perform similarly; chronological inheritance was not established.
- **Positive title/appointment/access/appanage sentence association:** early results were sensitive to sentence density; later matched-page analyses replace them.
- **R12–R64 semantic proximity as independent validation:** these categories share raw predicates, so part of the proximity is circular.
- **Exact title strings or embedding neighbors as identities:** recurrent titles, homonyms and circular title bridges can merge distinct people.
- **Sovereign share correlated with an HHI including the sovereign:** the quantities are mechanically coupled.

The controlling files are `WITHDRAWN_OR_SUPERSEDED_RESULTS.md`, `WITHDRAWN_OR_NOT_USED.md`, and `CLEAR_ERRORS_AND_CAUTIONS.md` in the indexed archives. Their exact extracted paths are searchable in `inventory.csv`.

## Method-level enumeration and remaining work

The [380-item checklist](results/10_requested_method_checklist.csv) distinguishes previous evidence, new execution, partial execution, missing data prerequisites and unrun alternatives. It deliberately does not claim an exhaustive benchmark of every centrality, community algorithm, tensor factorization or neural model. Exact equivalence algorithms, RESCAL/DEDICOM, supra-adjacency multilayer models, general closed/maximal subgraph mining, and additional embedding/GNN architectures remain listed as unrun variants.

Historical event-time paths, REMs, survival models and dynamic actor/network processes require validated event order, identities, exposure or network-state observations. Geographic distance/autocorrelation/gravity models require validated geolocation. Those inputs are not supplied by narrative page order or the existing three-tier place-string gazetteer. The new results therefore do not relabel page windows as elapsed historical time, or mentions as complete population risk sets.

The inventory is [CSV](inventory.csv) and [JSON](inventory.json); the [method-search evidence](results/10_archive_method_search_hits.csv) records source paths and line numbers. The HTML reader provides a filterable version of the method checklist.

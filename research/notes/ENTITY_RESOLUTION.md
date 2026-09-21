# Entity resolution: from similar names to a review queue

The chronicle contains spelling variants, transliterations, recurring titles, and
different people described in almost the same words. Resolving them requires a
pipeline that retrieves plausible matches, ranks them using several kinds of
evidence, and preserves source-based review before identities are combined.

## From similarity to supervised ranking

The entity-only embedding pass generated 1,530,847 candidate pairs using the top
100 neighbours per cluster. Its highest-ranked pair—“Hlaingdet Prince's younger
son” and “Hlaingdet Prince's youngest son”—had cosine similarity 0.9956. The
adjudication record identifies sentence `vol3_s001416` as describing two sons with
different titles, records zero accepted merges, and stops the raw-similarity pass.
This made the distinction between a useful retrieval signal and an identity
decision concrete.

The [full-label trainer](../../pipeline/resolution/train_full_gold_pair_classifier.py)
combines entity/context similarities with pair features, compares logistic
regression and histogram gradient boosting, and builds a ranked review queue.
Its 16,944 labelled pairs include 16,122 positive alias pairs derived from GPT Pro
identity clusters, supplemented by corrections and adjudicated cluster pairs;
the [label-source counts](../findings/entity_resolution/classifier_results.json)
make that mixture explicit.

Connected identity/adjudication families stay together in five-fold
cross-validation, repeated eight times. Identity weighting prevents large alias
clusters from dominating the comparison. The saved
[fold counts](../findings/entity_resolution/cv_folds.csv) make the sizes of these
partitions inspectable.

| Model | Identity-weighted average precision | Identity-weighted ROC AUC |
|---|---:|---:|
| Raw embedding similarity | 0.7481 | 0.3304 |
| Logistic regression | 0.9847 | 0.9434 |
| Histogram gradient boosting | 0.9929 | 0.9736 |

Gradient boosting was selected. On the 822 unresolved-cluster pairs—the subset
matching the review task—it achieved out-of-fold **AP 0.8172 and ROC AUC 0.9250**.
These subset results distinguish the hard identity decisions from within-cluster
alias pairs in the larger evaluation population.

## Reviewing decisions and the retrieval boundary

The model ranks candidates for review; automatic ML merging remains disabled.
A separate high-precision threshold retained 18 of 241 positive cluster pairs
with no observed false positives in the saved out-of-fold predictions. Keeping
that narrow queue separate from the broader review queue makes the precision/
coverage tradeoff visible. The results support the selected review policy on
this labelled population; model and threshold selection used these evaluations.

The upstream retrieval audit found 236 of 240 eligible known positive pairs
(98.33%). Four—including “Dagon fort” / “DA_PON_FORT”—never reached the classifier.
That diagnosis separates a retrieval problem from a ranking problem and identifies
where broader candidate generation is needed.

A separate 40-page contextual-resolution trial tested whether an LLM could
produce inventory-conforming groups. Its two mechanical-conformance reports
record removal of 70 unknown aliases and restoration of 340 tags as singletons.
The repair step restored coverage without inventing additional identity merges;
complete final coverage therefore includes those repairs.

The [selected review records](../findings/entity_resolution/review_decisions.json)
preserve the rejected top candidate, recorded stop decision, four retrieval misses,
and per-window repair counts. Together with the classifier aggregates and fold
counts, they show how model comparison, domain evidence, and review policy fit
together. These are excerpts from the July/September 2026 research stages, not a
new evaluation of every later resolution wave; full source passages, model weights,
and entity inventories remain outside this repository.

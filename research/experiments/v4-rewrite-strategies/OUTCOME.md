# v4 repair strategies

When an extraction pass produces output that is mostly right and partly wrong, there
are several ways to fix it, and they differ in cost and in how much correct output they
put at risk.

Four were tried against the v4 pass:

| Strategy | Approach |
|---|---|
| delta repair | send only the defective records back for correction |
| flat verbatim | re-emit everything verbatim, correcting in place |
| full rewrite | discard and regenerate the page |
| diagnostic | classify the failures first, then choose |

Delta repair is cheapest and safest and became the pattern the pipeline uses:
`pipeline/extraction/predicate_gloss_repair_batch.py`,
`relation_predicate_grounding_batch.py`, `cross_page_repair.py` and
`page_grounding_repair.py` are all delta repairs over specific defect classes,
with `pipeline/audit/` deciding what counts as a defect.

The diagnostic step turned out to matter more than the repair strategy. Classifying
failures before repairing them is what makes a delta pass possible at all.

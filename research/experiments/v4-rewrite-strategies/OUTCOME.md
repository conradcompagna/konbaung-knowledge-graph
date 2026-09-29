# v4 repair strategies

Targeted repair preserves accepted extraction records while correcting specific
defect classes. This comparison explored the cost and preservation tradeoffs
between record-level corrections and full-page regeneration.

Four were tried against the v4 pass:

| Strategy | Approach |
|---|---|
| delta repair | send only the defective records back for correction |
| flat verbatim | re-emit everything verbatim, correcting in place |
| full rewrite | discard and regenerate the page |
| diagnostic | classify the failures first, then choose |

The pipeline adopted delta repair to focus requests on identified defects and
preserve accepted records:
`pipeline/extraction/predicate_gloss_repair_batch.py`,
`relation_predicate_grounding_batch.py`, `cross_page_repair.py` and
`page_grounding_repair.py` are all delta repairs over specific defect classes,
with `pipeline/audit/` deciding what counts as a defect.

The diagnostic step turned out to matter more than the repair strategy. Classifying
failures before repairing them is what makes a delta pass possible at all.

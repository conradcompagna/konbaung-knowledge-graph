# v4 repair strategies

Four ways of repairing defective records in the v4 extraction output:

| Strategy | Approach |
|---|---|
| delta repair | send only the defective records back for correction |
| flat verbatim | re-emit everything verbatim, correcting in place |
| full rewrite | discard and regenerate the page |
| diagnostic | classify the failures first, then choose |

The pipeline uses delta repair, which leaves accepted records unchanged:
`pipeline/extraction/predicate_gloss_repair_batch.py`,
`relation_predicate_grounding_batch.py`, `cross_page_repair.py` and
`page_grounding_repair.py` are all delta repairs over specific defect classes,
with `pipeline/audit/` identifying the defects. Classifying failures first, as in the
diagnostic strategy, is what makes a targeted repair pass possible.

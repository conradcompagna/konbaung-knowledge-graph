# Research record

This directory is the record of how the knowledge graph was produced and what the
analysis of it found. It is not the dataset. The extracted triples, the RDF store, the
embedding matrices and the source scans are not published; see
[../docs/PUBLICATION.md](../docs/PUBLICATION.md).

The question this directory is meant to answer is whether someone could follow the
method — not whether they could download the result.

## Layout

| Directory | Contents |
|---|---|
| [`analysis/`](analysis/) | The statistical and network analysis source: contingency analysis, graph layers, tensor semantics, path/rule motifs, degree-based null models, network models, robustness, brokerage and hierarchy, predictive checks, sensitivity, validation, figures, and the audit catalogue. |
| [`findings/`](findings/) | What the analysis produced: model selection tables, fit statistics, scope declarations, stability and sensitivity intervals, validation output. Summary statistics only. |
| [`figures/`](figures/) | The four figures, in PNG and SVG. |
| [`reproduce/`](reproduce/) | The runner, per-stage logs, output checksums, input provenance, and the data-validation record. |
| [`notes/`](notes/) | Methodology, the audit of existing tests, the new-analyses write-up, and a preliminary historical interpretation. |
| [`datasets/`](datasets/) | Label inventory distributions and the most frequent labels. |
| [`experiments/`](experiments/) | Extraction approaches that were superseded, with an `OUTCOME.md` each. |

`inputs/`, `recovered_source/` and `results/` remain untracked: those are the local
directories holding the source data and the full analysis output.
`findings/` is the published subset.

## Where to start

For a concrete model-development example, read
[Entity resolution: from similar names to a review queue](notes/ENTITY_RESOLUTION.md),
which connects candidate retrieval, supervised ranking, and source-based review to
the saved evaluation records.

1. [`notes/METHODOLOGY.md`](notes/METHODOLOGY.md) — how the data was built and embedded.
2. [`notes/ANALYSIS_READ_ME_FIRST.md`](notes/ANALYSIS_READ_ME_FIRST.md) — what the
   analysis run was for and how to read its output.
3. [`reproduce/`](reproduce/) — the eleven-stage run, in order, with its logs.
4. [`notes/NEW_ANALYSES_AND_RESULTS.md`](notes/NEW_ANALYSES_AND_RESULTS.md) — what the
   analysis found.
5. [`notes/PRELIMINARY_HISTORICAL_INTERPRETATION.md`](notes/PRELIMINARY_HISTORICAL_INTERPRETATION.md)
   — what it might mean, stated as preliminary.

## What is in `findings/`, and what is not

Published: model selection and fit statistics, community and topology summaries, QAP
results, bootstrap intervals, component stability, brokerage and hierarchy summaries,
link-prediction scores, null-model diagnostics, the method-family audit, the
environment record, and the validation output.

Not published: the full contingency cells, all layer pairs, tensor loadings,
per-entity centralities, null-draw matrices, the trained link-prediction parameters,
and the three `*_anchors.csv` tables. The anchors quote triples verbatim — subject,
predicate, object and source sentence — and are the dataset rather than a finding.

The boundary is deliberate: every statistic needed to judge whether the analysis was
done properly is here; the material needed to reconstruct the corpus is not.

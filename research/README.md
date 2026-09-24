# Research: building and analyzing a historical knowledge graph

Start with the [construction story and supporting evidence](../docs/BUILD_PROCESS.md)
for the selected artifacts and the process connecting them to the application.

This package connects a Burmese primary source to structured extraction, embeddings,
entity resolution, and statistical analysis. It records both the engineering that
produced the graph and the historical questions the graph makes possible to explore.

## A route through the work

1. **The historical question and findings:** [interpretation](notes/PRELIMINARY_HISTORICAL_INTERPRETATION.md)
   and [analysis results](notes/NEW_ANALYSES_AND_RESULTS.md) examine differentiated
   forms of power and the relationships among royal allocation, administration,
   military action, information flows, and religious patronage.
2. **Building the research data:** [methodology](notes/METHODOLOGY.md) and the
   [pipeline guide](../pipeline/README.md) explain corpus reconstruction,
   sentence-linked extraction, analytical categories, embeddings, and validation.
3. **Making the work inspectable:** the [reproduction record](reproduce/README.md)
   provides stage logs, environment details, input provenance, and output checksums;
   the [analysis guide](notes/ANALYSIS_READ_ME_FIRST.md) introduces the test battery.

## Supporting material

| Directory | Contents |
|---|---|
| [analysis/](analysis/) | Statistical and network-analysis source, from contingency models and graph layers to tensor decomposition, null models, prediction, and sensitivity checks |
| [findings/](findings/) | Selected model comparisons, fit statistics, network summaries, stability intervals, and validation records |
| [figures/](figures/) | Four research figures in PNG and SVG |
| [reproduce/](reproduce/) | Runner, logs, checksums, provenance, and environment records |
| [notes/](notes/) | Methodology, findings, historical interpretation, and detailed evaluation coverage |
| [datasets/](datasets/) | Label-inventory distributions and frequency summaries |
| [experiments/](experiments/) | The development of extraction schemas and repair strategies |

## Entity resolution case study

[From similar names to a review queue](notes/ENTITY_RESOLUTION.md) connects candidate
retrieval, supervised ranking, error analysis, and source-based review to saved
evaluation records. It shows how similarity search became a practical curation tool.

The [construction record](../docs/BUILD_PROCESS.md) connects the served graph to
its canonical annotations, embedding model and categories; the analysis records
identify the input snapshots behind each set of findings.

# Annotator generations

The annotator developed across four connected requirements: reading historical
Burmese, producing structured output, linking claims to source sentences, and
connecting entities across pages. The sequence below records how each schema
iteration addressed those requirements.

The generations, in order, all against the same corpus:

| Generation | Design change | Next design requirement |
|---|---|---|
| `plaintext_triple_annotator` | free-text triples | output was not parseable reliably |
| `triple_annotator` | structured triple schema | triples were not tied to source sentences |
| `simple_linked_triple_annotator` | sentence identifiers on each triple | entities were re-invented per page |
| `minimal_linked_triple_annotator` | minimal schema, cross-page identifiers | too minimal to carry evidence |
| `spine_triple_annotator` | an explicit argument "spine" per page | the spine crowded out secondary claims |
| `page_kg_annotator`, `kg_annotator` | whole-page knowledge graph, published in `pipeline/extraction/` | — |
| `structured_open_coding_annotator` | open coding with structure, published | the approach that scaled |

The published pipeline uses the last two. The earlier ones are here.

Version families under `konbaung_spine_triple_page0177_test_v1` through `v9` in the
development workspace tested the spine schema against one page with definitions
present or absent, examples fictional or real, context windows of different sizes, and
the evidence field first or last. That kind of single-page A/B against a fixed
reference page is how each schema change was judged before a full-corpus run.

The gold standards those tests were scored against are in
[`../../../prompts/gold_standards/`](../../../prompts/gold_standards/).

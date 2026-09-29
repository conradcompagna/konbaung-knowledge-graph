# Annotator generations

Earlier versions of the annotator that extracts triples from chronicle pages. Each
version addressed a problem found in the one before. In order, all run on the same
corpus:

| Generation | Design change | Next design requirement |
|---|---|---|
| `plaintext_triple_annotator` | free-text triples | output was not parseable reliably |
| `triple_annotator` | structured triple schema | triples were not tied to source sentences |
| `simple_linked_triple_annotator` | sentence identifiers on each triple | entities were re-invented per page |
| `minimal_linked_triple_annotator` | minimal schema, cross-page identifiers | too minimal to carry evidence |
| `spine_triple_annotator` | an explicit argument "spine" per page | the spine crowded out secondary claims |
| `page_kg_annotator`, `kg_annotator` | whole-page knowledge graph, published in `pipeline/extraction/` | — |
| `structured_open_coding_annotator` | open coding with structure, published in `pipeline/extraction/` | — |

The current pipeline uses the last two; the scripts for the earlier ones are in this
folder. Each schema change was tested on a single reference page before a full-corpus
run. The reference annotations used for those tests are in
[`../../../prompts/gold_standards/`](../../../prompts/gold_standards/).

## Other trials

| Script | Purpose |
|---|---|
| `chunk_triple_annotator.py` | Annotates each Burmese evidence chunk before extracting its triple |
| `delta_enrichment_test.py` | Tests adding enrichment fields to existing triples without changing them |
| `enrichment_review_trial.py` | Runs the sentence-analysis enrichment schema without modifying the corpus |
| `metadata_recall_trial.py`, `metadata_recall_prompt.txt` | Metadata recall over existing chronicle triples |
| `minimal_semantic_triple_trial.py` | Minimal subject–predicate–object extraction from sentence translations |
| `v3_ungrounded_entity_span_trial.py` | Locates the Burmese spans of V3 entities on one page |

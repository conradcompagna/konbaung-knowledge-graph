# Served graph: build summary

The deployed graph and reader use these snapshots:

| Component | Snapshot | Built by |
|---|---|---|
| Claims | `konbaung_historiography_v3_canonical_20260724` (27,129 claims, 11,222 sentences, 1,215 pages) from run `konbaung_historiography_ungrounded_v3_full_batch_20260723`, Gemini 3.1 Flash Lite | [`pipeline/extraction/historiography.py`](../pipeline/extraction/historiography.py), [`pipeline/corpus/build_historiography_v3_reader_data.py`](../pipeline/corpus/build_historiography_v3_reader_data.py) |
| Categories | `konbaung_axial_categories_v2` (52 entity, 81 relation categories) | [`pipeline/extraction/flashlite_axial_full_corpus_batch.py`](../pipeline/extraction/flashlite_axial_full_corpus_batch.py) |
| Embeddings | Gemini Embedding 2, 768 dimensions, eight views | [`pipeline/embeddings/v3_eight_view_embeddings.py`](../pipeline/embeddings/v3_eight_view_embeddings.py) |
| Graph | `konbaung_knowledge_graph_v3` (Oxigraph RDF) | [`konbaung_reader_app/build_graph_database.py`](../konbaung_reader_app/build_graph_database.py) |

Where a sentence appears on more than one page, one annotation is kept: the one with
the most triples, then the one on the sentence's own page, then the lowest page number.

File hashes for these snapshots are in
[`research/reproduce/served_artifacts.json`](../research/reproduce/served_artifacts.json).
The full construction method is in [`research/notes/METHODOLOGY.md`](../research/notes/METHODOLOGY.md),
and the stages are mapped to code in [`pipeline/README.md`](../pipeline/README.md).

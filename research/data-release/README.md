# Konbaung Chronicle V3 dataset

I built this dataset to investigate kingship, office, military command, religious
patronage, tribute and kinship in the Konbaung chronicles. It connects structured
historical claims to Burmese source sentences, English translations, source-page
identifiers, semantic categories and vector representations.

**[Citable dataset on Zenodo](https://doi.org/10.5281/zenodo.22949204)** ·
**[Download on GitHub](https://github.com/conradcompagna/konbaung-knowledge-graph/releases/tag/data-v3-2026-09-25)** ·
**[Construction and methods](https://github.com/conradcompagna/konbaung-knowledge-graph/blob/main/docs/BUILD_PROCESS.md)**

## What is in the release

| Material | Contents |
|---|---|
| Canonical claims | 27,129 subject–predicate–object records across three volumes and 1,215 pages |
| Source sentences | 11,282 records, including Burmese text, English translations and page routing |
| Entity representations | 23,890 labels and a 23,890 × 768 float32 matrix |
| Relation representations | 11,886 labels and an 11,886 × 768 float32 matrix |
| Contextual representations | 27,135 full-triple/context vectors, 768 dimensions each, with explicit links for all 27,129 claims |
| Entity resolution | 17,658 clusters, membership tables and 54,258 subject/object occurrence records with alternative person-resolution assignments |
| Provenance | Original selection metadata, source-record hashes, category assignments, matrix row identifiers and a SHA-256 manifest |

`Konbaung_Chronicle_V3_Data.zip` contains the 23 payload files from my original
`Konbaung_Raw_Triples_and_Embeddings.zip`, unchanged byte for byte, plus this guide,
the license, citation, manifest and loading example. The release packages three
matrices from the wider eight-view embedding workflow documented in the methods.

## How I constructed it

I processed the three volumes through OCR, source restoration and sentence
reconstruction, then used Gemini 3.1 Flash Lite for translation and V3 structured
claim extraction. I selected one canonical annotation per sentence from the
`konbaung_historiography_ungrounded_v3_full_batch_20260723` run, producing
`konbaung_historiography_v3_canonical_20260724`.

For repeated sentence appearances, selection prioritizes the annotation with the
most triples, then the owner page, then the lowest page number. The exported
sentences retain that selection rule, decisions and source appearances. There are
11,222 canonical V3 sentence records and 60 source sentences whose original status
is `v3_unavailable`; 10,498 sentences contain claims.

I assigned the claims to a system of 52 entity categories and 81 relation
categories, generated 768-dimensional representations with Gemini Embedding 2,
and developed entity clustering and guarded person-resolution workflows. Raw
labels and alternative identity assignments remain available alongside one
another, so historical interpretation can follow the source and resolution trail.

These records encode assertions in the chronicles. The source sentences,
translations and page references provide the context for interpreting each
model-assisted annotation.

## Files and joins

| Path in the ZIP | Role and join |
|---|---|
| `data/triples.jsonl` | One claim per `triple_id`; join `sentence_id` to source sentences |
| `data/sentences.jsonl` | One record per `sentence_id`; `volume_id`, `owner_page`, `page_numbers` and `cross_page` preserve page context |
| `embeddings/entity_records.jsonl` | Entity `index` is the zero-based row in `entity_base_vectors.npy`; `baseKey` matches the claim's `embedding_record_key` |
| `embeddings/relation_records.jsonl` | The corresponding row/key mapping for `relation_base_vectors.npy` |
| `embeddings/triple_context.keys.jsonl` | Context-vector `row` and original embedding `key` |
| `embeddings/triple_context_links.jsonl` | `triple_id` → `vector_rows` in `triple_context.npy` |
| `entity_resolution/cluster_tables/` | Compact cluster records, label membership and cluster summaries |
| `entity_resolution/guarded_person_tables/` | Title bridges, review decisions, exclusions, revocations and occurrence-level person assignments |
| `DATA_INDEX.json` | Original export notes and payload counts |

Each claim's `subject`, `predicate` and `object` contains a `tag`,
`embedding_record_key`, `embedding_row` and `embedding_matrix`. Its
`axial_categories` records subject, relation and object categories.

Context links retain original chunking: 27,111 claims have one vector row and 18
have two. Some rows are shared between claims. Use `vector_rows` as supplied;
the release does not average or regenerate them.

Cluster tables use rank-based identifiers such as `e00000`; these are a separate
namespace from embedding keys such as `e_...`. Resolve cluster membership through
the member labels/table rather than treating the two kinds of identifier as equal.

`occurrence_level_person_resolution_all_specs.csv` contains one subject and one
object occurrence for each claim. The `role` column distinguishes them. Alongside
source context, it preserves `person_id_raw_tag`, `person_id_supplied_cluster`,
`person_id_guarded_cluster`, `person_id_preferred_high` and
`person_id_preferred_inclusive`. These columns record alternative resolution
specifications rather than interchangeable identifiers.

## Read an actual record

With Python and NumPy installed, run the included example against the downloaded
archive (no extraction, server or model inference is needed):

```sh
python read_dataset.py Konbaung_Chronicle_V3_Data.zip
```

It prints a real claim, its linked Burmese/English sentence, and the shapes of its
entity, relation and contextual vectors. Use `--triple-id ID` to select a claim
and `--verify` to compare every original payload with the manifest. The script
reads the `.npy` matrices with `allow_pickle=False`.

## Release integrity and citation

The [manifest](manifest.json) records every original file's bytes and SHA-256,
row counts, schemas, matrix dimensions and successful record-link checks. All
27,129 exported claims and 11,222 canonical sentence texts/translations were
compared with the retained canonical V3 source files. The entity and relation
matrices also match the graph-build input fingerprints recorded in
[the served-artifact manifest](https://github.com/conradcompagna/konbaung-knowledge-graph/blob/main/research/reproduce/served_artifacts.json).

The archive checksum is published separately as `SHA256SUMS.txt`. This V3 release
is distinct from the earlier analysis snapshot described in
[`research/datasets/`](https://github.com/conradcompagna/konbaung-knowledge-graph/tree/main/research/datasets).

Compagna, Conrad (2026). *Konbaung Chronicle V3: Canonical Claims, Source Sentences
and Embeddings* (3.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.22949204

Machine-readable citation: [CITATION.cff](CITATION.cff).

## Terms

The [Research Data Evaluation License](LICENSE.txt) permits research, education,
experimentation and evaluation, including hiring evaluation, with attribution.
Commercial use and redistribution of the dataset or embeddings require my
permission. Underlying source material and other third-party rights retain their
own status. Contact: conradcompagna@gmail.com.

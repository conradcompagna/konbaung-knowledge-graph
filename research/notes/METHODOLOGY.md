# Konbaung Historical-Relations Dataset and Embedding Tables

## Methodology and reproducibility report

**Version 1.0 · 31 August 2026**

### Scope

This report documents the research database and embedding-derived tables in `DIGHUM_WEBGPT_ANALYSIS_PACKAGE_20260831_WITH_METHODOLOGY.zip`. It covers corpus preparation, sentence translation, subject–predicate–object extraction, cross-page deduplication, analytical categories, embeddings, similarity tables, clustering, and archive validation. The report follows each transformation from the source pages to the analytical tables, with explicit identifiers, validation rules, and recorded parameters. Application architecture is described in the repository and pipeline guides.

## 1. Dataset and textual units

The corpus comprises three volumes of Burmese historical prose concerning the Konbaung dynasty. Four audited duplicate scans were excluded from 1,219 source-page records, leaving 1,215 annotation pages. The final statistical tables contain:

| Item | Count |
|---|---:|
| Unique sentence rows | 11,282 |
| Canonical V3-annotated sentences | 11,222 |
| Sentences unavailable to V3 annotation | 60 |
| Triple-bearing sentences | 10,498 |
| Canonical triple occurrences | 27,129 |
| Distinct entity-tag strings | 23,890 |
| Distinct relation-tag strings | 11,886 |
| Axial categories | 52 entity; 81 relation |

Sentence boundaries were defined only by U+104B MYANMAR SIGN SECTION (`။`). OCR line wraps and other whitespace runs were normalized to one space and did not create boundaries. Running headers, printed page numbers, confirmed editorial matter, duplicate terminators, and audited OCR debris were removed by deterministic rules or explicit allowlists. Ambiguous OCR was preserved. Validation reconstructed every sentence from exact page substrings, checked complete retained-character coverage, and verified source SHA-256 hashes.

A sentence continuing onto a following page retained source pieces from every page but belonged to the page on which it began (`owner_page`). Cross-page routing files were subsets of the sentence corpus and were not appended as new observations. A restoration pass added eleven chronicle sentences and repaired eleven cross-page records using exact canonical-page offsets. The final volume counts are 3,254, 3,987, and 4,041 sentences.

## 2. English translations

The principal translation pass used `gemini-3.1-flash-lite`, temperature 0, one candidate, an 8,000-token output limit, and zero thinking budget. Requests were grouped by owner page and supplied ordered sentence IDs, Burmese sentence text, and cleaned page context. The prompt required full translation rather than summary, preservation of clauses, actors, quantities, negation, modality, lists, and reported speech, and prohibited merging or splitting sentences. Page context could resolve omitted referents; unambiguous additions were marked in square brackets.

Structured validation required exactly one nonempty translation per input ID in the original order. The main run produced 11,271 accepted translations with no missing or flagged pages. Eleven newly restored sentences and corrected translations for eleven repaired records were imported from the audited restoration package, yielding 11,282 sentence–translation pairs.

## 3. Open-coded triples and cross-page deduplication

The extraction prompt asked how power operated in the Konbaung dynasty as a lived social system. It covered royal and military authority, administration, service, taxation and tribute, local intermediaries, succession, kinship, Buddhist legitimacy, ritual, punishment, rebellion, diplomacy, war, tributary relations, and colonial displacement.

Each sentence received either `annotate` with one or more triples or `skip` with no triples and a short justification. Every triple contained an English analytical `subject`, `predicate`, and `object`. Labels were intended to be slightly more abstract than literal glosses but close to the evidence. This generation links triples to their source sentence IDs. Its output schema contains semantic triples rather than token-level evidence spans, offsets, or confidence scores.

Extraction used `gemini-3.1-flash-lite`, temperature 0, one candidate, a 16,000-token output limit, and a structured schema. Pages were attempted at high thinking; truncations were retried at medium and then minimal thinking. Structural validation required every supplied sentence ID exactly once, preserved order, valid decisions, and complete triple fields. Of 1,215 pages, 1,213 were accepted. The two pages outside the accepted extraction set (`vol2-p0132` and `vol3-p0511`) account for 60 sentences; their text and translations remain in the corpus with annotation coverage recorded explicitly.

Cross-page sentences sometimes received annotations in more than one page context. Canonicalization grouped results by stable sentence ID and selected one result using this ordered rule:

1. greatest number of triples;
2. if tied, the sentence’s owner page;
3. if still tied, the lower page number.

There were 806 multiply submitted sentence IDs and 813 duplicate appearances beyond the retained observations. The source had 30,180 page-occurrence triples; canonicalization produced 27,129. The archive re-executed the selection rule and stores sentence text only once. Page appearances remain provenance, not independent observations.

## 4. Axial analytical categories

Each canonical subject and object was classified into one of 52 entity categories, and each predicate into one of 81 relation categories. This used `gemini-3.1-flash-lite`, prompt version 2.0, temperature 0, low thinking, and a 15,000-token output limit. Inputs included the page summary, Burmese sentences, translations, and existing triples. The model could classify but not rewrite, add, delete, merge, split, or reorder triples.

Fifteen previously validated pages covering 334 triples were reused after exact ordered-subset verification. Remaining pages were processed in volume batches; length failures were completed with minimal-thinking and ordered chunk retries. All 27,129 triples received three category assignments. Final closed-schema import removed provisional codes, including 14 audit-specified field remaps and 22 documented manual closures. The final category manifest reports zero unresolved triples and no provisional categories.

## 5. Embedding construction

Eight embeddings were defined for every canonical triple using `gemini-embedding-2` at 768 dimensions:

| View | Input text |
|---|---|
| S / O | `Historical argument: {subject or object}` |
| P | `Directed historical relation: {predicate}` |
| SPO | Separate `Subject`, `Predicate`, and `Object` lines |
| SBE / OBE | S/O header plus Burmese sentence and English translation |
| PBE | P header plus Burmese sentence and English translation |
| SPOBE | SPO text plus Burmese sentence and English translation |

An embedding key was `e_` plus the first 32 hexadecimal characters of SHA-256 over the exact UTF-8 request text. Identical text therefore reused one vector. The 27,129 triples generated 217,032 view references but 151,275 unique requests. These were stored in six physical source matrices: argument (23,890 rows), predicate (11,886), triple (26,867), argument-context (39,493), predicate-context (22,004), and triple-context (27,135).

Requests above 9,000 characters received exact token counting. Forty-six inputs above the safe 7,800-token threshold were split near their midpoint at punctuation or whitespace; Burmese and English context were split in parallel, and chunk keys were recorded. Every response was checked for key coverage, 768 values, errors, and vector norm. All 151,275 keys were collected, and all six shards were certified. Stored vectors are `float32` and effectively unit normalized.

For statistical tables, subjects and objects were pooled into one entity vocabulary; predicates formed a relation vocabulary. Records were ordered by descending mention frequency and then raw tag. Each tag’s context vector was the normalized sum of its occurrence-level contextual vectors; split occurrences were first averaged and normalized. The fused vector was:

\[
f = \operatorname{norm}\left[\sqrt{0.70}\,b\ ;\ \sqrt{0.30}\,c\right],
\]

where `b` is the base vector and `c` the contextual centroid. Fused-vector dot products therefore equal `0.70 × cosine(base) + 0.30 × cosine(context)`. The archive contains aligned 768-dimensional base/context matrices and 1,536-dimensional fused matrices for 23,890 entity and 11,886 relation tags. Every triple contains the exact entity or relation base-matrix row for its three fields.

## 6. Similarity and exploratory clusters

Nearest-neighbor candidates were generated by sparse random projection of fused vectors to 256 dimensions using seed 20,260,724. Projected cosine search returned 60 non-self candidates. Exact fused-vector dot products were computed within that candidate set, and the best 30 were retained. The archive supplies both GPT-readable JSONL rows and NPZ `indices`/`similarities` arrays: 716,700 directed entity links and 356,580 relation links. Scores are exact for retained candidates, but candidate retrieval is approximate rather than exhaustive all-pairs search.

For clustering, undirected graphs used the top 15 neighbors, minimum similarity 0.84 for entities and 0.85 for relations, plus weight-1 links for exact normalized lexical matches. Louvain resolutions from 0.35 to 2.0 were evaluated over three seeds; resolution 2.0 was selected. Communities with fewer than five tag rows were marked unresolved. Results were 68 substantive plus 181 unresolved entity communities (249 total IDs) and 54 substantive plus 14 unresolved relation communities (68 total IDs).

Substantive communities were labeled by `gemini-3.1-flash-lite` at temperature 0 and low thinking, followed by 18 documented manual label overrides. These clusters are exploratory semantic neighborhoods, not entity resolution and not substitutes for the 52/81 axial categories.

## 7. Archive construction and validation

The primary files are `data/pages.jsonl`, `data/sentences.jsonl`, and `data/triples.jsonl`, keyed by `page_id`, `sentence_id`, and `triple_id`. Triples join to sentences by `sentence_id`, to matrices by `embedding_row`, and to the analytical taxonomy by category ID. Sentence text is deliberately absent from triple rows, preventing repetition.

The archive also contains tag records, base/context/fused matrices, neighbor JSONL and NPZ tables, frequency tables, cluster assignments and inventories, parameter reports, manifests, and validation records. Before packaging, the builder checked primary-key uniqueness, sentence identity, canonical selection, foreign keys, triple-to-embedding mappings, category coverage, matrix shape/dtype/norm, sorted non-self neighbor tables, and complete cluster assignments.

Every staged file was hashed in `CHECKSUMS.sha256`. The ZIP used ZIP64 and DEFLATE level 6, passed CRC testing, and was compared with the staged member inventory before atomic installation. A sidecar SHA-256 verifies the complete archive.

## 8. Analysis units and validation scope

Use 11,282 as the full sentence denominator, 11,222 for V3-available sentences, 10,498 for triple-bearing sentences, and 27,129 for claim-level analysis. Entity positions total 54,258; relation positions total 27,129. Use `owner_page` for unique page allocation and do not expand cross-page provenance as independent observations. Multiple triples within a sentence and sentences within a page are dependent; inferential models should use clustered or hierarchical errors where appropriate.

Validation combines exact source reconstruction, schema and coverage checks, documented corrections, and parameter-based sensitivity analysis. These checks establish computational integrity and make the transformations inspectable. Translations, triples, categories, and cluster labels are model-assisted; independent inter-annotator reliability has not been measured.

The analytical choices are recorded explicitly: V3 traceability is at sentence level; contextual embeddings combine Burmese text with model-generated English; raw labels can have multiple senses, with subject/object occurrences pooled for entity vectors. Nearest-neighbor retrieval uses approximate candidates with exact rescoring. Louvain communities depend on graph thresholds, resolution, and seed, so comparisons should use the recorded settings and stability analyses. These details define the scope within which the tables support historical interpretation.

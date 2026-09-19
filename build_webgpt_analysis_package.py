from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


ROOT = Path(__file__).resolve().parent
APP_ROOT = ROOT / "konbaung_reader_app"
SUMMARY_PATH = ROOT / "DIGHUM_PROJECT_DATABASE_SUMMARY_20260810.json"
CANONICAL_ROOT = (
    APP_ROOT / "data" / "konbaung_historiography_v3_canonical_20260724"
)
AXIAL_ROOT = APP_ROOT / "data" / "konbaung_axial_categories_v2"
GRAPH_ROOT = APP_ROOT / "data" / "konbaung_knowledge_graph_v3"
EMBEDDING_ROOT = ROOT / "konbaung_v3_node_edge_clustering_first_pass_20260724"
OCCURRENCES_PATH = (
    ROOT / "konbaung_v3_eight_view_embeddings_20260724" / "occurrences.jsonl"
)
METHODOLOGY_MD = ROOT / "KONBAUNG_DATA_AND_EMBEDDINGS_METHODOLOGY.md"
METHODOLOGY_DOCX = ROOT / "KONBAUNG_DATA_AND_EMBEDDINGS_METHODOLOGY.docx"
OUTPUT_PATH = ROOT / "DIGHUM_WEBGPT_ANALYSIS_PACKAGE_20260831_WITH_METHODOLOGY.zip"


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON document from an explicit source path."""
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSON Lines records without loading the full file at once."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc


def write_json(path: Path, value: Any) -> None:
    """Write human-readable UTF-8 JSON with Burmese text preserved as Unicode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write normalized records as UTF-8 JSON Lines and return the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file using bounded-memory reads."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_canonical_sentences() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load the already-deduplicated V3 sentence snapshot and its selection audit."""
    index = read_json(CANONICAL_ROOT / "index.json")
    sentences: dict[str, dict[str, Any]] = {}
    for volume_id in ("vol1", "vol2", "vol3"):
        document = read_json(CANONICAL_ROOT / "sentences" / f"{volume_id}.json")
        if document["selectionRule"] != index["selectionRule"]:
            raise RuntimeError(f"Selection rule mismatch in {volume_id}")
        for sentence_id, sentence in document["sentences"].items():
            if sentence_id in sentences:
                raise RuntimeError(f"Duplicate canonical sentence ID: {sentence_id}")
            if sentence["sid"] != sentence_id:
                raise RuntimeError(f"Sentence key/ID mismatch: {sentence_id}")
            sentences[sentence_id] = sentence
    expected = int(index["totals"]["canonicalSentences"])
    if len(sentences) != expected:
        raise RuntimeError(
            f"Expected {expected:,} canonical sentences, found {len(sentences):,}"
        )
    return sentences, index


def validate_selection_rule(
    canonical_sentences: dict[str, dict[str, Any]], canonical_index: dict[str, Any]
) -> dict[str, int]:
    """Recheck the max-triples/owner-page/lower-page rule recorded by the source."""
    duplicate_sentence_ids = 0
    checked_appearances = 0
    for sentence in canonical_sentences.values():
        appearances = sentence["sourceAppearances"]
        checked_appearances += len(appearances)
        if len(appearances) > 1:
            duplicate_sentence_ids += 1
        owner_page = int(sentence["ownerPage"])
        best = min(
            appearances,
            key=lambda row: (
                -int(row["tripleCount"]),
                0 if int(row["page"]) == owner_page else 1,
                int(row["page"]),
            ),
        )
        selected = sentence["selectedFrom"]
        if best["key"] != selected["key"]:
            raise RuntimeError(
                f"Canonical selection rule failed for {sentence['sid']}: "
                f"expected {best['key']}, found {selected['key']}"
            )
    expected_duplicates = int(canonical_index["totals"]["duplicateSentenceIds"])
    if duplicate_sentence_ids != expected_duplicates:
        raise RuntimeError(
            f"Expected {expected_duplicates} duplicate annotation sentence IDs, "
            f"found {duplicate_sentence_ids}"
        )
    return {
        "duplicateSentenceIds": duplicate_sentence_ids,
        "sourceAppearancesChecked": checked_appearances,
    }


def normalize_pages_and_sentences(
    summary: dict[str, Any], canonical_sentences: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    """Normalize page appearances into one page row and one row per sentence ID."""
    pages: list[dict[str, Any]] = []
    appearances: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for volume in summary["volumes"]:
        volume_id = volume["id"]
        for page in volume["pages"]:
            pages.append(
                {
                    "page_id": page["pageId"],
                    "volume_id": volume_id,
                    "volume_label": volume["label"],
                    "page_number": int(page["pageNumber"]),
                    "summary": page.get("summary", ""),
                }
            )
            for sentence in page["sentences"]:
                appearances[sentence["sentenceId"]].append(sentence)

    sentence_rows: dict[str, dict[str, Any]] = {}
    conflicts = 0
    for sentence_id, rows in appearances.items():
        first = rows[0]
        invariant = (
            int(first["ownerPage"]),
            tuple(int(page) for page in first["pages"]),
            bool(first["crossPage"]),
            first["burmese"],
            first["english"],
        )
        for row in rows[1:]:
            candidate = (
                int(row["ownerPage"]),
                tuple(int(page) for page in row["pages"]),
                bool(row["crossPage"]),
                row["burmese"],
                row["english"],
            )
            if candidate != invariant:
                conflicts += 1
        canonical = canonical_sentences.get(sentence_id)
        volume_id = sentence_id.split("_", 1)[0]
        if canonical is not None:
            if canonical["my"] != first["burmese"] or canonical["en"] != first["english"]:
                raise RuntimeError(f"Canonical/reader text mismatch for {sentence_id}")
            sentence_rows[sentence_id] = {
                "sentence_id": sentence_id,
                "volume_id": canonical["volumeId"],
                "owner_page": int(canonical["ownerPage"]),
                "page_numbers": [int(page) for page in canonical["pages"]],
                "cross_page": len(canonical["pages"]) > 1,
                "burmese": canonical["my"],
                "english_translation": canonical["en"],
                "annotation_status": "canonical_v3",
                "decision": canonical["decision"],
                "justification": canonical.get("justification", ""),
                "selection_rule": canonical["selectionRule"],
                "selected_from": canonical["selectedFrom"],
                "source_appearances": canonical["sourceAppearances"],
                "triple_count": len(canonical["triples"]),
            }
        else:
            sentence_rows[sentence_id] = {
                "sentence_id": sentence_id,
                "volume_id": volume_id,
                "owner_page": int(first["ownerPage"]),
                "page_numbers": [int(page) for page in first["pages"]],
                "cross_page": bool(first["crossPage"]),
                "burmese": first["burmese"],
                "english_translation": first["english"],
                "annotation_status": "v3_unavailable",
                "decision": None,
                "justification": "",
                "selection_rule": None,
                "selected_from": None,
                "source_appearances": [],
                "triple_count": 0,
            }

    if conflicts:
        raise RuntimeError(
            f"Found {conflicts} conflicting representations of repeated sentence IDs"
        )
    expected_pages = int(summary["counts"]["pages"])
    expected_unique = (
        int(summary["counts"]["uniqueCanonicalSentences"])
        + int(summary["counts"]["unavailableSentenceIds"])
    )
    if len(pages) != expected_pages or len(sentence_rows) != expected_unique:
        raise RuntimeError(
            f"Normalized count mismatch: {len(pages)} pages, "
            f"{len(sentence_rows)} sentences"
        )
    stats = {
        "pageAppearanceRows": sum(len(rows) for rows in appearances.values()),
        "uniqueSentenceRows": len(sentence_rows),
        "crossPageSentenceRows": sum(
            1 for row in sentence_rows.values() if row["cross_page"]
        ),
        "canonicalV3SentenceRows": sum(
            1
            for row in sentence_rows.values()
            if row["annotation_status"] == "canonical_v3"
        ),
        "unavailableV3SentenceRows": sum(
            1
            for row in sentence_rows.values()
            if row["annotation_status"] == "v3_unavailable"
        ),
        "conflictingRepeatedSentenceRows": conflicts,
    }
    return pages, sentence_rows, stats


def load_embedding_records(
    records_path: Path, matrix_path: Path, expected_rows: int
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, dict[str, Any]], dict[str, Any]]:
    """Load one app embedding table and validate its row mapping and vector norms."""
    records = list(iter_jsonl(records_path))
    matrix = np.load(matrix_path, mmap_mode="r")
    if len(records) != expected_rows or matrix.shape != (expected_rows, 768):
        raise RuntimeError(
            f"Embedding shape mismatch for {matrix_path.name}: "
            f"{len(records)} records and shape {matrix.shape}"
        )
    if matrix.dtype != np.float32:
        raise RuntimeError(f"Expected float32 vectors in {matrix_path.name}")
    indices = [int(record["index"]) for record in records]
    if sorted(indices) != list(range(expected_rows)):
        raise RuntimeError(f"Non-contiguous embedding indices in {records_path.name}")
    by_tag = {record["tag"]: record for record in records}
    if len(by_tag) != expected_rows:
        raise RuntimeError(f"Duplicate tags in {records_path.name}")
    norms = np.linalg.norm(np.asarray(matrix), axis=1)
    if not np.isfinite(norms).all() or float(norms.min()) < 0.999 or float(norms.max()) > 1.001:
        raise RuntimeError(f"Embedding vectors are not finite unit vectors: {matrix_path}")
    stats = {
        "rows": expected_rows,
        "dimensions": 768,
        "dtype": str(matrix.dtype),
        "normMin": float(norms.min()),
        "normMax": float(norms.max()),
        "normMean": float(norms.mean()),
    }
    return records, matrix, by_tag, stats


def load_knn_table(
    path: Path, records: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Validate one precomputed top-30 neighbor table against the shared tag rows."""
    archive = np.load(path)
    if set(archive.files) != {"indices", "similarities"}:
        raise RuntimeError(f"Unexpected KNN arrays in {path.name}: {archive.files}")
    indices = np.asarray(archive["indices"])
    similarities = np.asarray(archive["similarities"])
    expected_shape = (len(records), 30)
    if indices.shape != expected_shape or similarities.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected KNN shape in {path.name}: "
            f"{indices.shape}, {similarities.shape}"
        )
    if indices.dtype != np.int32 or similarities.dtype != np.float32:
        raise RuntimeError(f"Unexpected KNN dtypes in {path.name}")
    if int(indices.min()) < 0 or int(indices.max()) >= len(records):
        raise RuntimeError(f"Out-of-range KNN index in {path.name}")
    if not np.isfinite(similarities).all():
        raise RuntimeError(f"Non-finite KNN similarity in {path.name}")
    row_numbers = np.arange(len(records), dtype=np.int32)[:, None]
    if np.any(indices == row_numbers):
        raise RuntimeError(f"Self-neighbor found in {path.name}")
    if np.any(similarities[:, 1:] > similarities[:, :-1] + 1e-7):
        raise RuntimeError(f"KNN rows are not similarity-sorted in {path.name}")
    stats = {
        "rows": len(records),
        "neighborsPerRow": 30,
        "links": int(indices.size),
        "similarityMin": float(similarities.min()),
        "similarityMax": float(similarities.max()),
        "indicesInRange": True,
        "selfNeighbors": 0,
        "rowsSortedDescending": True,
    }
    return indices, similarities, stats


def validate_feature_matrix(
    path: Path, expected_rows: int, expected_dimensions: int
) -> dict[str, Any]:
    """Validate a context or fused feature matrix without loading it wholly into RAM."""
    matrix = np.load(path, mmap_mode="r")
    if matrix.shape != (expected_rows, expected_dimensions) or matrix.dtype != np.float32:
        raise RuntimeError(
            f"Unexpected feature matrix at {path}: {matrix.shape}, {matrix.dtype}"
        )
    norm_min = float("inf")
    norm_max = float("-inf")
    norm_sum = 0.0
    for start in range(0, expected_rows, 4096):
        block = np.asarray(matrix[start : start + 4096], dtype=np.float32)
        if not np.isfinite(block).all():
            raise RuntimeError(f"Non-finite feature value in {path.name}")
        norms = np.linalg.norm(block, axis=1)
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
        norm_sum += float(norms.sum())
    if norm_min < 0.999 or norm_max > 1.001:
        raise RuntimeError(f"Feature rows are not unit normalized in {path.name}")
    return {
        "rows": expected_rows,
        "dimensions": expected_dimensions,
        "dtype": str(matrix.dtype),
        "normMin": norm_min,
        "normMax": norm_max,
        "normMean": norm_sum / expected_rows,
    }


def neighbor_rows(
    records: list[dict[str, Any]],
    indices: np.ndarray,
    similarities: np.ndarray,
) -> Iterator[dict[str, Any]]:
    """Expose compact NPZ neighbor arrays as GPT-readable tag-name JSONL rows."""
    for row_index, record in enumerate(records):
        yield {
            "index": row_index,
            "tag": record["tag"],
            "neighbors": [
                {
                    "rank": rank,
                    "index": int(neighbor_index),
                    "tag": records[int(neighbor_index)]["tag"],
                    "similarity": float(similarity),
                }
                for rank, (neighbor_index, similarity) in enumerate(
                    zip(indices[row_index], similarities[row_index]), start=1
                )
            ],
        }


def validate_labeled_assignments(
    path: Path, records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Confirm every tag row has exactly one final labeled-cluster assignment."""
    assignments = list(iter_jsonl(path))
    if len(assignments) != len(records):
        raise RuntimeError(f"Assignment row count mismatch in {path.name}")
    cluster_ids: set[str] = set()
    unresolved = 0
    for expected_index, (assignment, record) in enumerate(zip(assignments, records)):
        if (
            int(assignment["index"]) != expected_index
            or assignment["tag"] != record["tag"]
            or assignment["baseKey"] != record["baseKey"]
        ):
            raise RuntimeError(f"Assignment/tag row mismatch at {path.name}:{expected_index}")
        cluster_ids.add(assignment["clusterId"])
        unresolved += int(bool(assignment["unresolved"]))
    return {
        "rows": len(assignments),
        "uniqueClusters": len(cluster_ids),
        "unresolvedRows": unresolved,
        "everyTagAssignedOnce": True,
        "indicesMatchEmbeddingRecords": True,
    }


def load_axial_assignments() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load the final app taxonomy and its sentence/ordinal category assignments."""
    catalog = read_json(AXIAL_ROOT / "catalog.json")
    assignments: dict[str, dict[str, Any]] = {}
    for volume_id in ("vol1", "vol2", "vol3"):
        document = read_json(AXIAL_ROOT / "assignments" / f"{volume_id}.json")
        for sentence_id, assignment in document["sentences"].items():
            if sentence_id in assignments:
                raise RuntimeError(f"Duplicate axial assignment: {sentence_id}")
            assignments[sentence_id] = assignment
    return catalog, assignments


def category_lookup(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index both entity and relation category definitions by their stable ID."""
    lookup: dict[str, dict[str, Any]] = {}
    for kind in ("entities", "relations"):
        for category in catalog[kind]:
            category_id = category["id"]
            if category_id in lookup:
                raise RuntimeError(f"Duplicate axial category ID: {category_id}")
            lookup[category_id] = category
    return lookup


def build_triples(
    sentence_rows: dict[str, dict[str, Any]],
    canonical_sentences: dict[str, dict[str, Any]],
    node_by_tag: dict[str, dict[str, Any]],
    edge_by_tag: dict[str, dict[str, Any]],
    assignments: dict[str, dict[str, Any]],
    categories: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join canonical triples to taxonomy tags and exact app embedding matrix rows."""
    triples: list[dict[str, Any]] = []
    occurrence_ids: set[str] = set()
    sentence_ordinals: set[tuple[str, int]] = set()
    triples_per_sentence: Counter[str] = Counter()
    for occurrence in iter_jsonl(OCCURRENCES_PATH):
        occurrence_id = occurrence["occurrenceId"]
        sentence_id = occurrence["sid"]
        ordinal = int(occurrence["tripleOrdinal"])
        if occurrence_id in occurrence_ids:
            raise RuntimeError(f"Duplicate occurrence ID: {occurrence_id}")
        if (sentence_id, ordinal) in sentence_ordinals:
            raise RuntimeError(f"Duplicate sentence/ordinal: {sentence_id}:{ordinal}")
        occurrence_ids.add(occurrence_id)
        sentence_ordinals.add((sentence_id, ordinal))
        if sentence_id not in canonical_sentences or sentence_id not in sentence_rows:
            raise RuntimeError(f"Triple references unknown sentence: {sentence_id}")
        sentence = canonical_sentences[sentence_id]
        if (
            occurrence["sentenceMy"] != sentence["my"]
            or occurrence["sentenceEn"] != sentence["en"]
        ):
            raise RuntimeError(f"Triple/source sentence mismatch: {occurrence_id}")
        source_triple = sentence["triples"][ordinal - 1]
        raw_tags = (
            occurrence["subject"],
            occurrence["predicate"],
            occurrence["object"],
        )
        if raw_tags != (
            source_triple["subject"],
            source_triple["predicate"],
            source_triple["object"],
        ):
            raise RuntimeError(f"Canonical triple mismatch: {occurrence_id}")

        subject_record = node_by_tag[occurrence["subject"]]
        predicate_record = edge_by_tag[occurrence["predicate"]]
        object_record = node_by_tag[occurrence["object"]]
        embedding_keys = occurrence["embeddingKeys"]
        if embedding_keys["S"] != [subject_record["baseKey"]]:
            raise RuntimeError(f"Subject embedding key mismatch: {occurrence_id}")
        if embedding_keys["P"] != [predicate_record["baseKey"]]:
            raise RuntimeError(f"Predicate embedding key mismatch: {occurrence_id}")
        if embedding_keys["O"] != [object_record["baseKey"]]:
            raise RuntimeError(f"Object embedding key mismatch: {occurrence_id}")

        assignment = assignments[sentence_id]
        category_ids = assignment["triples"].get(str(ordinal))
        if assignment["status"] != "accepted" or category_ids is None:
            raise RuntimeError(f"Missing accepted axial assignment: {occurrence_id}")
        for category_id in category_ids.values():
            if category_id not in categories:
                raise RuntimeError(f"Unknown axial category: {category_id}")

        triples.append(
            {
                "triple_id": occurrence_id,
                "canonical_key": f"v3-{sentence_id}-t{ordinal:03d}",
                "sentence_id": sentence_id,
                "volume_id": occurrence["volumeId"],
                "ordinal": ordinal,
                "owner_page": int(occurrence["ownerPage"]),
                "page_numbers": [int(page) for page in occurrence["pageOccurrences"]],
                "subject": {
                    "tag": occurrence["subject"],
                    "embedding_record_key": subject_record["baseKey"],
                    "embedding_row": int(subject_record["index"]),
                    "embedding_matrix": "embeddings/entity_base_vectors.npy",
                },
                "predicate": {
                    "tag": occurrence["predicate"],
                    "embedding_record_key": predicate_record["baseKey"],
                    "embedding_row": int(predicate_record["index"]),
                    "embedding_matrix": "embeddings/relation_base_vectors.npy",
                },
                "object": {
                    "tag": occurrence["object"],
                    "embedding_record_key": object_record["baseKey"],
                    "embedding_row": int(object_record["index"]),
                    "embedding_matrix": "embeddings/entity_base_vectors.npy",
                },
                "axial_categories": {
                    "subject": {
                        "id": category_ids["s"],
                        "label": categories[category_ids["s"]]["label"],
                    },
                    "relation": {
                        "id": category_ids["r"],
                        "label": categories[category_ids["r"]]["label"],
                    },
                    "object": {
                        "id": category_ids["o"],
                        "label": categories[category_ids["o"]]["label"],
                    },
                },
                "axial_assignment_source_page": assignment["sourcePage"],
                "source_record_hash": occurrence["sourceRecordHash"],
            }
        )
        triples_per_sentence[sentence_id] += 1

    for sentence_id, sentence in canonical_sentences.items():
        if triples_per_sentence[sentence_id] != len(sentence["triples"]):
            raise RuntimeError(f"Triple count mismatch for {sentence_id}")
    if len(triples) != 27_129:
        raise RuntimeError(f"Expected 27,129 triples, found {len(triples):,}")
    return triples, {
        "rows": len(triples),
        "uniqueTripleIds": len(occurrence_ids),
        "uniqueSentenceOrdinalPairs": len(sentence_ordinals),
        "tripleBearingSentences": len(triples_per_sentence),
        "entityEmbeddingReferences": len(triples) * 2,
        "relationEmbeddingReferences": len(triples),
        "missingEmbeddingReferences": 0,
        "missingAxialAssignments": 0,
    }


def write_table_overview(path: Path, counts: dict[str, int]) -> None:
    """Write a tiny CSV inventory for tools that inspect tabular files first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("table", "rows", "primary_key"))
        writer.writerow(("data/pages.jsonl", counts["pages"], "page_id"))
        writer.writerow(("data/sentences.jsonl", counts["sentences"], "sentence_id"))
        writer.writerow(("data/triples.jsonl", counts["triples"], "triple_id"))
        writer.writerow(
            ("embeddings/entity_records.jsonl", counts["entityTags"], "index")
        )
        writer.writerow(
            ("embeddings/relation_records.jsonl", counts["relationTags"], "index")
        )
        writer.writerow(
            ("similarity/entity_top30_neighbors.jsonl", counts["entityTags"], "index")
        )
        writer.writerow(
            (
                "similarity/relation_top30_neighbors.jsonl",
                counts["relationTags"],
                "index",
            )
        )
        writer.writerow(
            ("clustering/entity_assignments_labeled.jsonl", counts["entityTags"], "index")
        )
        writer.writerow(
            (
                "clustering/relation_assignments_labeled.jsonl",
                counts["relationTags"],
                "index",
            )
        )


def package_readme(counts: dict[str, int], canonical_index: dict[str, Any]) -> str:
    """Describe the normalized joins and the cross-page counting rules for Web GPT."""
    totals = canonical_index["totals"]
    return f"""# DIGHUM / Konbaung Web GPT analysis package

This archive is a normalized, read-only export of the final database used by the
Konbaung reader app. Start with `manifest.json`, then use the three JSONL tables.

## What is here

- `methodology/`: the article-facing methodology report in Markdown and Word format.
- `data/pages.jsonl`: {counts['pages']:,} pages, one row per page, including page summaries.
- `data/sentences.jsonl`: {counts['sentences']:,} unique sentence IDs, with Burmese text,
  English translation, page provenance, annotation status, and the canonical-selection
  audit. Sentence text occurs only here; it is not repeated inside every triple.
- `data/triples.jsonl`: {counts['triples']:,} canonical V3 claim rows. Join
  `sentence_id` to the sentence table. Each subject, predicate, and object includes its
  exact app embedding-matrix row and its raw tag. Axial category IDs and labels are also
  present.
- `data/axial_category_catalog.json`: definitions for all {counts['entityCategories']}
  entity categories and {counts['relationCategories']} relation categories.
- `embeddings/entity_records.jsonl` + `entity_base_vectors.npy`: {counts['entityTags']:,}
  raw entity tags and their 768-dimensional float32 vectors.
- `embeddings/relation_records.jsonl` + `relation_base_vectors.npy`:
  {counts['relationTags']:,} raw relation tags and their 768-dimensional float32 vectors.
- `similarity/entity_top30_neighbors.jsonl` and
  `similarity/relation_top30_neighbors.jsonl`: GPT-readable precomputed neighbor tables,
  containing {counts['entityNeighborLinks']:,} entity similarities and
  {counts['relationNeighborLinks']:,} relation similarities. Exact source NPZ tables and
  their manifests are beside them under `similarity/precomputed/`.
- `similarity/features/`: base-context centroids and the 1,536-dimensional fused feature
  matrices used by the precomputed neighbor and clustering passes. Similarity is
  `0.70 * cosine(base) + 0.30 * cosine(context centroid)`.
- `clustering/`: unlabeled and labeled row assignments, cluster inventories, label maps,
  selected-resolution and sweep reports, frequency tables, and audit documents. It covers
  {counts['entityClusters']:,} entity clusters and {counts['relationClusters']:,} relation
  clusters.
- `database/konbaung_knowledge_graph_v3.nq`: the complete portable N-Quads serialization
  of the app's Oxigraph RDF database ({counts['rdfStatements']:,} statements).

## Critical counting rule

Do not treat page appearances as independent sentences or claims. Use `sentence_id` and
`triple_id` as primary keys.

The source annotation run had {totals['sourceTriples']:,} page-occurrence triples and
{totals['duplicateSentenceIds']:,} sentence IDs submitted more than once because some
sentences crossed page boundaries. The canonical source keeps exactly one annotation set
per sentence ID: most triples first, then owner page, then lower page number. It removed
{totals['removedDuplicateAppearances']:,} duplicate annotation appearances and yielded
{totals['canonicalTriples']:,} canonical triples. This package re-validates that rule and
contains one sentence row per ID.

The sentence table has {totals['canonicalSentences']:,} canonical V3 sentences plus
{totals['unavailableSentenceIds']:,} translated source sentences from the two invalid
annotation pages. Those unavailable rows are retained with `annotation_status` equal to
`v3_unavailable` and have no invented triples.

## Embedding joins

The app stores embeddings for raw entity and relation tags, not a separate invented vector
for each claim. Every triple explicitly points to all three vectors:

```python
import json
import numpy as np

entity_vectors = np.load("embeddings/entity_base_vectors.npy", mmap_mode="r")
relation_vectors = np.load("embeddings/relation_base_vectors.npy", mmap_mode="r")

with open("data/triples.jsonl", encoding="utf-8") as handle:
    triple = json.loads(next(handle))

s = entity_vectors[triple["subject"]["embedding_row"]]
p = relation_vectors[triple["predicate"]["embedding_row"]]
o = entity_vectors[triple["object"]["embedding_row"]]
```

The source embedding pipeline reports model `gemini-embedding-2`. The older RDF build
manifest labels the same app-facing matrices `gemini-embedding-001`; both provenance
records are preserved in `manifest.json` instead of silently hiding that metadata
discrepancy.

## Precomputed similarities and clusters

The JSONL neighbor tables are keyed by the same `index` used in the entity and relation
record tables. Each row names the tag and its 30 retained neighbors in descending
similarity order. These are the existing pipeline results: 60 projected candidates were
generated, then the 30 highest exact fused-feature cosine scores among those candidates
were retained. They are not presented as a newly computed exhaustive all-pairs table.

The exact compact arrays are:

- `similarity/precomputed/entity_knn.npz`: `indices` and `similarities`, shape
  `({counts['entityTags']}, 30)`.
- `similarity/precomputed/relation_knn.npz`: the same arrays, shape
  `({counts['relationTags']}, 30)`.

Cluster assignments use those row indices too. Prefer the files ending in `_labeled` when
you want the final candidate meta-tags; the unlabeled versions are preserved as provenance.

## Suggested preliminary analyses

- Frequency tables and cross-tabs over raw tags or axial category IDs.
- Per-volume and per-page distributions using owner pages, without double-counting
  cross-page appearances.
- Cosine similarity between raw tag vectors (the stored rows are unit-normalized).
- Sentence-level co-occurrence or claim density by joining triples to sentences.
- RDF queries after loading the N-Quads file into an RDF engine.

Use `CHECKSUMS.sha256` to verify every packaged file. `validation_report.json` records the
deduplication, foreign-key, taxonomy, and embedding-coverage checks performed at build time.
"""


def usage_example() -> str:
    """Provide a compact loader that a code-enabled GPT can run after extraction."""
    return '''from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def read_jsonl(relative_path: str):
    with (ROOT / relative_path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


sentences = {row["sentence_id"]: row for row in read_jsonl("data/sentences.jsonl")}
entity_vectors = np.load(ROOT / "embeddings/entity_base_vectors.npy", mmap_mode="r")
relation_vectors = np.load(ROOT / "embeddings/relation_base_vectors.npy", mmap_mode="r")

for triple in read_jsonl("data/triples.jsonl"):
    sentence = sentences[triple["sentence_id"]]
    subject_vector = entity_vectors[triple["subject"]["embedding_row"]]
    predicate_vector = relation_vectors[triple["predicate"]["embedding_row"]]
    object_vector = entity_vectors[triple["object"]["embedding_row"]]
    # Run analysis here. Sentence text is intentionally joined rather than duplicated.
    break
'''


def copy_sources(stage: Path) -> None:
    """Copy exact portable database and app embedding artifacts into the package."""
    copies = {
        GRAPH_ROOT / "konbaung_knowledge_graph_v3.nq": (
            stage / "database" / "konbaung_knowledge_graph_v3.nq"
        ),
        GRAPH_ROOT / "manifest.json": stage / "database" / "source_manifest.json",
        EMBEDDING_ROOT / "node_records.jsonl": (
            stage / "embeddings" / "entity_records.jsonl"
        ),
        EMBEDDING_ROOT / "edge_records.jsonl": (
            stage / "embeddings" / "relation_records.jsonl"
        ),
        EMBEDDING_ROOT / "node_base_vectors.npy": (
            stage / "embeddings" / "entity_base_vectors.npy"
        ),
        EMBEDDING_ROOT / "edge_base_vectors.npy": (
            stage / "embeddings" / "relation_base_vectors.npy"
        ),
        EMBEDDING_ROOT / "node_feature_manifest.json": (
            stage / "embeddings" / "entity_feature_manifest.source.json"
        ),
        EMBEDDING_ROOT / "edge_feature_manifest.json": (
            stage / "embeddings" / "relation_feature_manifest.source.json"
        ),
        EMBEDDING_ROOT / "feature_build_report.json": (
            stage / "embeddings" / "feature_build_report.source.json"
        ),
        EMBEDDING_ROOT / "node_context_vectors.npy": (
            stage / "similarity" / "features" / "entity_context_vectors.npy"
        ),
        EMBEDDING_ROOT / "edge_context_vectors.npy": (
            stage / "similarity" / "features" / "relation_context_vectors.npy"
        ),
        EMBEDDING_ROOT / "node_fused_vectors.npy": (
            stage / "similarity" / "features" / "entity_fused_vectors.npy"
        ),
        EMBEDDING_ROOT / "edge_fused_vectors.npy": (
            stage / "similarity" / "features" / "relation_fused_vectors.npy"
        ),
        EMBEDDING_ROOT / "node_knn.npz": (
            stage / "similarity" / "precomputed" / "entity_knn.npz"
        ),
        EMBEDDING_ROOT / "edge_knn.npz": (
            stage / "similarity" / "precomputed" / "relation_knn.npz"
        ),
        EMBEDDING_ROOT / "node_knn_manifest.json": (
            stage / "similarity" / "precomputed" / "entity_knn_manifest.json"
        ),
        EMBEDDING_ROOT / "edge_knn_manifest.json": (
            stage / "similarity" / "precomputed" / "relation_knn_manifest.json"
        ),
        EMBEDDING_ROOT / "knn_build_report.json": (
            stage / "similarity" / "precomputed" / "knn_build_report.json"
        ),
        EMBEDDING_ROOT / "node_assignments.jsonl": (
            stage / "clustering" / "entity_assignments.jsonl"
        ),
        EMBEDDING_ROOT / "edge_assignments.jsonl": (
            stage / "clustering" / "relation_assignments.jsonl"
        ),
        EMBEDDING_ROOT / "node_assignments_labeled.jsonl": (
            stage / "clustering" / "entity_assignments_labeled.jsonl"
        ),
        EMBEDDING_ROOT / "edge_assignments_labeled.jsonl": (
            stage / "clustering" / "relation_assignments_labeled.jsonl"
        ),
        EMBEDDING_ROOT / "node_clusters.json": (
            stage / "clustering" / "entity_clusters.json"
        ),
        EMBEDDING_ROOT / "edge_clusters.json": (
            stage / "clustering" / "relation_clusters.json"
        ),
        EMBEDDING_ROOT / "node_clusters_labeled.json": (
            stage / "clustering" / "entity_clusters_labeled.json"
        ),
        EMBEDDING_ROOT / "edge_clusters_labeled.json": (
            stage / "clustering" / "relation_clusters_labeled.json"
        ),
        EMBEDDING_ROOT / "node_cluster_labels.json": (
            stage / "clustering" / "entity_cluster_labels.json"
        ),
        EMBEDDING_ROOT / "edge_cluster_labels.json": (
            stage / "clustering" / "relation_cluster_labels.json"
        ),
        EMBEDDING_ROOT / "selected_resolution.json": (
            stage / "clustering" / "selected_resolution.json"
        ),
        EMBEDDING_ROOT / "resolution_sweep.json": (
            stage / "clustering" / "resolution_sweep.json"
        ),
        EMBEDDING_ROOT / "cluster_labeling_audit.json": (
            stage / "clustering" / "cluster_labeling_audit.json"
        ),
        EMBEDDING_ROOT / "konbaung_v3_node_tags_by_frequency.csv": (
            stage / "clustering" / "entity_tags_by_frequency.csv"
        ),
        EMBEDDING_ROOT / "konbaung_v3_edge_tags_by_frequency.csv": (
            stage / "clustering" / "relation_tags_by_frequency.csv"
        ),
        EMBEDDING_ROOT / "FIRST_PASS_CLUSTER_REPORT.md": (
            stage / "clustering" / "FIRST_PASS_CLUSTER_REPORT.md"
        ),
        EMBEDDING_ROOT / "PARED_DOWN_METATAG_MAPPING.md": (
            stage / "clustering" / "PARED_DOWN_METATAG_MAPPING.md"
        ),
        EMBEDDING_ROOT / "UNLABELED_CLUSTER_PREVIEW.md": (
            stage / "clustering" / "UNLABELED_CLUSTER_PREVIEW.md"
        ),
        CANONICAL_ROOT / "index.json": stage / "provenance" / "canonical_index.json",
        AXIAL_ROOT / "manifest.json": stage / "provenance" / "axial_manifest.json",
        METHODOLOGY_MD: (
            stage / "methodology" / "KONBAUNG_DATA_AND_EMBEDDINGS_METHODOLOGY.md"
        ),
        METHODOLOGY_DOCX: (
            stage / "methodology" / "KONBAUNG_DATA_AND_EMBEDDINGS_METHODOLOGY.docx"
        ),
        Path(__file__).resolve(): (
            stage / "provenance" / "build_webgpt_analysis_package.py"
        ),
    }
    for source, destination in copies.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def source_hashes() -> dict[str, str]:
    """Record checksums for the authoritative inputs used to assemble normalized rows."""
    sources = {
        "DIGHUM_PROJECT_DATABASE_SUMMARY_20260810.json": SUMMARY_PATH,
        "canonical_index.json": CANONICAL_ROOT / "index.json",
        "canonical_vol1.json": CANONICAL_ROOT / "sentences" / "vol1.json",
        "canonical_vol2.json": CANONICAL_ROOT / "sentences" / "vol2.json",
        "canonical_vol3.json": CANONICAL_ROOT / "sentences" / "vol3.json",
        "axial_catalog.json": AXIAL_ROOT / "catalog.json",
        "axial_assignments_vol1.json": AXIAL_ROOT / "assignments" / "vol1.json",
        "axial_assignments_vol2.json": AXIAL_ROOT / "assignments" / "vol2.json",
        "axial_assignments_vol3.json": AXIAL_ROOT / "assignments" / "vol3.json",
        "canonical_occurrences.jsonl": OCCURRENCES_PATH,
        "entity_records.jsonl": EMBEDDING_ROOT / "node_records.jsonl",
        "relation_records.jsonl": EMBEDDING_ROOT / "edge_records.jsonl",
        "entity_base_vectors.npy": EMBEDDING_ROOT / "node_base_vectors.npy",
        "relation_base_vectors.npy": EMBEDDING_ROOT / "edge_base_vectors.npy",
        "entity_context_vectors.npy": EMBEDDING_ROOT / "node_context_vectors.npy",
        "relation_context_vectors.npy": EMBEDDING_ROOT / "edge_context_vectors.npy",
        "entity_fused_vectors.npy": EMBEDDING_ROOT / "node_fused_vectors.npy",
        "relation_fused_vectors.npy": EMBEDDING_ROOT / "edge_fused_vectors.npy",
        "entity_knn.npz": EMBEDDING_ROOT / "node_knn.npz",
        "relation_knn.npz": EMBEDDING_ROOT / "edge_knn.npz",
        "entity_assignments_labeled.jsonl": (
            EMBEDDING_ROOT / "node_assignments_labeled.jsonl"
        ),
        "relation_assignments_labeled.jsonl": (
            EMBEDDING_ROOT / "edge_assignments_labeled.jsonl"
        ),
        "entity_clusters_labeled.json": EMBEDDING_ROOT / "node_clusters_labeled.json",
        "relation_clusters_labeled.json": EMBEDDING_ROOT / "edge_clusters_labeled.json",
        "knowledge_graph_v3.nq": GRAPH_ROOT / "konbaung_knowledge_graph_v3.nq",
        "methodology.md": METHODOLOGY_MD,
        "methodology.docx": METHODOLOGY_DOCX,
    }
    return {name: sha256(path) for name, path in sources.items()}


def build() -> Path:
    """Build, validate, checksum, and atomically install the Web GPT ZIP archive."""
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing package: {OUTPUT_PATH}")
    required = (
        SUMMARY_PATH,
        CANONICAL_ROOT / "index.json",
        AXIAL_ROOT / "catalog.json",
        GRAPH_ROOT / "konbaung_knowledge_graph_v3.nq",
        EMBEDDING_ROOT / "node_base_vectors.npy",
        EMBEDDING_ROOT / "edge_base_vectors.npy",
        OCCURRENCES_PATH,
        METHODOLOGY_MD,
        METHODOLOGY_DOCX,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    summary = read_json(SUMMARY_PATH)
    canonical_sentences, canonical_index = load_canonical_sentences()
    selection_stats = validate_selection_rule(canonical_sentences, canonical_index)
    pages, sentence_rows, sentence_stats = normalize_pages_and_sentences(
        summary, canonical_sentences
    )
    node_records, _, node_by_tag, node_stats = load_embedding_records(
        EMBEDDING_ROOT / "node_records.jsonl",
        EMBEDDING_ROOT / "node_base_vectors.npy",
        23_890,
    )
    edge_records, _, edge_by_tag, edge_stats = load_embedding_records(
        EMBEDDING_ROOT / "edge_records.jsonl",
        EMBEDDING_ROOT / "edge_base_vectors.npy",
        11_886,
    )
    node_knn_indices, node_knn_similarities, node_knn_stats = load_knn_table(
        EMBEDDING_ROOT / "node_knn.npz", node_records
    )
    edge_knn_indices, edge_knn_similarities, edge_knn_stats = load_knn_table(
        EMBEDDING_ROOT / "edge_knn.npz", edge_records
    )
    feature_stats = {
        "entityContext": validate_feature_matrix(
            EMBEDDING_ROOT / "node_context_vectors.npy", 23_890, 768
        ),
        "relationContext": validate_feature_matrix(
            EMBEDDING_ROOT / "edge_context_vectors.npy", 11_886, 768
        ),
        "entityFused": validate_feature_matrix(
            EMBEDDING_ROOT / "node_fused_vectors.npy", 23_890, 1536
        ),
        "relationFused": validate_feature_matrix(
            EMBEDDING_ROOT / "edge_fused_vectors.npy", 11_886, 1536
        ),
    }
    cluster_assignment_stats = {
        "entity": validate_labeled_assignments(
            EMBEDDING_ROOT / "node_assignments_labeled.jsonl", node_records
        ),
        "relation": validate_labeled_assignments(
            EMBEDDING_ROOT / "edge_assignments_labeled.jsonl", edge_records
        ),
    }
    entity_clusters = read_json(EMBEDDING_ROOT / "node_clusters_labeled.json")
    relation_clusters = read_json(EMBEDDING_ROOT / "edge_clusters_labeled.json")
    if len(entity_clusters) != cluster_assignment_stats["entity"]["uniqueClusters"]:
        raise RuntimeError("Entity cluster inventory/assignment count mismatch")
    if len(relation_clusters) != cluster_assignment_stats["relation"]["uniqueClusters"]:
        raise RuntimeError("Relation cluster inventory/assignment count mismatch")
    catalog, assignments = load_axial_assignments()
    categories = category_lookup(catalog)
    triples, triple_stats = build_triples(
        sentence_rows,
        canonical_sentences,
        node_by_tag,
        edge_by_tag,
        assignments,
        categories,
    )

    graph_manifest = read_json(GRAPH_ROOT / "manifest.json")
    counts = {
        "pages": len(pages),
        "sentences": len(sentence_rows),
        "canonicalSentences": len(canonical_sentences),
        "unavailableSentences": sentence_stats["unavailableV3SentenceRows"],
        "triples": len(triples),
        "tripleBearingSentences": triple_stats["tripleBearingSentences"],
        "entityTags": len(node_records),
        "relationTags": len(edge_records),
        "entityCategories": len(catalog["entities"]),
        "relationCategories": len(catalog["relations"]),
        "entityNeighborLinks": node_knn_stats["links"],
        "relationNeighborLinks": edge_knn_stats["links"],
        "entityClusters": len(entity_clusters),
        "relationClusters": len(relation_clusters),
        "rdfStatements": int(graph_manifest["statements"]),
    }

    with tempfile.TemporaryDirectory(prefix="dighum_webgpt_", dir=ROOT) as temp_name:
        stage = Path(temp_name)
        write_jsonl(stage / "data" / "pages.jsonl", pages)
        write_jsonl(
            stage / "data" / "sentences.jsonl",
            (sentence_rows[key] for key in sorted(sentence_rows)),
        )
        write_jsonl(stage / "data" / "triples.jsonl", triples)
        write_json(stage / "data" / "axial_category_catalog.json", catalog)
        write_table_overview(stage / "data" / "table_overview.csv", counts)
        write_jsonl(
            stage / "similarity" / "entity_top30_neighbors.jsonl",
            neighbor_rows(node_records, node_knn_indices, node_knn_similarities),
        )
        write_jsonl(
            stage / "similarity" / "relation_top30_neighbors.jsonl",
            neighbor_rows(edge_records, edge_knn_indices, edge_knn_similarities),
        )
        copy_sources(stage)

        validation_report = {
            "schemaVersion": 1,
            "validatedAt": datetime.now(timezone.utc).isoformat(),
            "passed": True,
            "sentenceDeduplication": {
                **sentence_stats,
                **selection_stats,
                "canonicalSelectionRule": canonical_index["selectionRule"],
                "sourceAnnotationTriplesBeforeDeduplication": canonical_index["totals"][
                    "sourceTriples"
                ],
                "removedDuplicateAnnotationAppearances": canonical_index["totals"][
                    "removedDuplicateAppearances"
                ],
                "sentenceIdPrimaryKeyUnique": True,
            },
            "triples": triple_stats,
            "embeddingCoverage": {
                "entity": node_stats,
                "relation": edge_stats,
                "everyTripleHasSubjectPredicateObjectEmbeddingRows": True,
            },
            "precomputedSimilarity": {
                "entityKnn": node_knn_stats,
                "relationKnn": edge_knn_stats,
                "featureMatrices": feature_stats,
                "formula": (
                    "0.70 * cosine(base, base) + "
                    "0.30 * cosine(contextCentroid, contextCentroid)"
                ),
            },
            "clustering": {
                "assignments": cluster_assignment_stats,
                "entityClusterInventoryRows": len(entity_clusters),
                "relationClusterInventoryRows": len(relation_clusters),
                "labeledInventoriesPresent": True,
                "unlabeledProvenancePresent": True,
            },
            "taxonomy": {
                "entityCategories": len(catalog["entities"]),
                "relationCategories": len(catalog["relations"]),
                "everyTripleHasThreeAcceptedCategoryAssignments": True,
            },
            "knowledgeGraph": {
                "portableFormat": "N-Quads",
                "statements": int(graph_manifest["statements"]),
                "claims": int(graph_manifest["counts"]["claims"]),
                "sentences": int(graph_manifest["counts"]["sentences"]),
            },
        }
        write_json(stage / "validation_report.json", validation_report)

        manifest = {
            "schemaVersion": 1,
            "title": "DIGHUM Konbaung Web GPT statistical analysis package",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "description": (
                "Normalized final app database export with unique sentences, canonical "
                "V3 triples, translations, raw tags, axial tags, exact app embedding "
                "matrix rows, precomputed similarities, labeled and unlabeled cluster "
                "tables, page summaries, and portable RDF."
            ),
            "counts": counts,
            "primaryKeys": {
                "data/pages.jsonl": "page_id",
                "data/sentences.jsonl": "sentence_id",
                "data/triples.jsonl": "triple_id",
                "embeddings/entity_records.jsonl": "index",
                "embeddings/relation_records.jsonl": "index",
                "similarity/entity_top30_neighbors.jsonl": "index",
                "similarity/relation_top30_neighbors.jsonl": "index",
            },
            "joins": {
                "tripleToSentence": "data/triples.jsonl.sentence_id -> data/sentences.jsonl.sentence_id",
                "subjectAndObjectEmbedding": (
                    "embedding_row -> embeddings/entity_base_vectors.npy[row]"
                ),
                "predicateEmbedding": (
                    "embedding_row -> embeddings/relation_base_vectors.npy[row]"
                ),
                "axialCategory": (
                    "axial_categories.*.id -> data/axial_category_catalog.json"
                ),
                "entitySimilarityAndClusterRows": (
                    "index -> embeddings/entity_records.jsonl.index"
                ),
                "relationSimilarityAndClusterRows": (
                    "index -> embeddings/relation_records.jsonl.index"
                ),
            },
            "deduplication": {
                "key": "sentence_id",
                "selectionRule": canonical_index["selectionRule"],
                "sourceDuplicateSentenceIds": canonical_index["totals"][
                    "duplicateSentenceIds"
                ],
                "removedDuplicateAnnotationAppearances": canonical_index["totals"][
                    "removedDuplicateAppearances"
                ],
                "note": (
                    "Repeated page appearances are provenance only. Sentence text is "
                    "stored once in data/sentences.jsonl."
                ),
            },
            "embeddings": {
                "dimensions": 768,
                "dtype": "float32",
                "normalized": True,
                "sourcePipelineModel": "gemini-embedding-2",
                "appGraphManifestModelLabel": graph_manifest["embeddingSort"]["model"],
                "metadataDiscrepancyPreserved": True,
                "entityMatrix": "embeddings/entity_base_vectors.npy",
                "relationMatrix": "embeddings/relation_base_vectors.npy",
                "roleSemantics": (
                    "Subject and object use the shared entity matrix; predicate uses "
                    "the relation matrix."
                ),
            },
            "precomputedSimilarity": {
                "formula": (
                    "0.70 * cosine(base, base) + "
                    "0.30 * cosine(contextCentroid, contextCentroid)"
                ),
                "candidateMethod": (
                    "60 projected candidate neighbors followed by exact fused-feature "
                    "scoring and retention of 30 rows"
                ),
                "entityJsonl": "similarity/entity_top30_neighbors.jsonl",
                "relationJsonl": "similarity/relation_top30_neighbors.jsonl",
                "entityNpz": "similarity/precomputed/entity_knn.npz",
                "relationNpz": "similarity/precomputed/relation_knn.npz",
                "contextAndFusedMatrices": "similarity/features/",
                "entityLinks": node_knn_stats["links"],
                "relationLinks": edge_knn_stats["links"],
            },
            "clustering": {
                "root": "clustering/",
                "entityClusters": len(entity_clusters),
                "relationClusters": len(relation_clusters),
                "finalAssignments": [
                    "clustering/entity_assignments_labeled.jsonl",
                    "clustering/relation_assignments_labeled.jsonl",
                ],
                "finalInventories": [
                    "clustering/entity_clusters_labeled.json",
                    "clustering/relation_clusters_labeled.json",
                ],
                "provenanceVersionsIncluded": True,
            },
            "database": {
                "engine": graph_manifest["database"],
                "portableExport": "database/konbaung_knowledge_graph_v3.nq",
                "format": "N-Quads",
                "statements": graph_manifest["statements"],
                "namedGraphs": graph_manifest["namedGraphs"],
            },
            "intentionalOmissions": [
                (
                    "The page-denormalized DIGHUM summary is not copied because it "
                    "repeats cross-page sentence and triple appearances. Its complete "
                    "content is normalized into pages, sentences, and triples tables."
                ),
                (
                    "The native Oxigraph RocksDB files are not copied because the exact "
                    "portable N-Quads serialization is complete and substantially easier "
                    "for Web GPT and statistical tools to read."
                ),
            ],
            "sourceSha256": source_hashes(),
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "README.md").write_text(
            package_readme(counts, canonical_index), encoding="utf-8"
        )
        (stage / "load_example.py").write_text(usage_example(), encoding="utf-8")

        packaged_files = sorted(path for path in stage.rglob("*") if path.is_file())
        checksum_lines = [
            f"{sha256(path)}  {path.relative_to(stage).as_posix()}"
            for path in packaged_files
        ]
        (stage / "CHECKSUMS.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )
        packaged_files = sorted(path for path in stage.rglob("*") if path.is_file())

        temporary_fd, temporary_zip_name = tempfile.mkstemp(
            prefix="dighum_webgpt_", suffix=".zip.building", dir=ROOT
        )
        os.close(temporary_fd)
        temporary_zip = Path(temporary_zip_name)
        try:
            with zipfile.ZipFile(
                temporary_zip,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=True,
            ) as archive:
                for path in packaged_files:
                    archive.write(path, path.relative_to(stage).as_posix())
            with zipfile.ZipFile(temporary_zip, mode="r") as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise RuntimeError(f"ZIP CRC validation failed: {bad_member}")
                expected_members = {
                    path.relative_to(stage).as_posix() for path in packaged_files
                }
                if set(archive.namelist()) != expected_members:
                    raise RuntimeError("ZIP member inventory does not match staging files")
            os.replace(temporary_zip, OUTPUT_PATH)
        finally:
            if temporary_zip.exists():
                temporary_zip.unlink()

    zip_hash = sha256(OUTPUT_PATH)
    OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".sha256").write_text(
        f"{zip_hash}  {OUTPUT_PATH.name}\n", encoding="utf-8"
    )
    return OUTPUT_PATH


if __name__ == "__main__":
    package = build()
    print(
        json.dumps(
            {
                "package": str(package),
                "bytes": package.stat().st_size,
                "sha256": sha256(package),
            },
            indent=2,
        )
    )

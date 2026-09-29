from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[2]
FIRST_PASS_ROOT = ROOT / "konbaung_v3_node_edge_clustering_first_pass_20260724"
WORKBOOK = ROOT / "konbaung_manual_clusters_top1000_nodes_edges.xlsx"
OUTPUT_ROOT = ROOT / "konbaung_v3_manual_seeded_clustering_20260724"
RANDOM_SEED = 20260724

CONFIG = {
    "node": {
        "assignment_sheet": "Node_Assignments",
        "cluster_sheet": "Node_Clusters",
        "prefix": "NODE",
        "minimum_similarity": 0.84,
        "minimum_lexical_similarity": 0.95,
        "minimum_head_similarity": 0.0,
        "neighbor_count": 15,
        "resolution": 2.0,
    },
    "edge": {
        "assignment_sheet": "Edge_Assignments",
        "cluster_sheet": "Edge_Clusters",
        "prefix": "EDGE",
        "minimum_similarity": 0.85,
        "minimum_lexical_similarity": 0.80,
        "minimum_head_similarity": 0.50,
        "neighbor_count": 15,
        "resolution": 2.0,
    },
}

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("Cannot normalize zero or non-finite vector")
    return np.asarray(vector / norm, dtype=np.float32)


def column_number(reference: str) -> int:
    letters = "".join(char for char in reference if char.isalpha())
    value = 0
    for char in letters:
        value = value * 26 + ord(char.upper()) - ord("A") + 1
    return value - 1


def workbook_rows(path: Path) -> dict[str, list[dict[str, str | None]]]:
    namespaces = {"m": MAIN_NS, "r": REL_NS}
    output: dict[str, list[dict[str, str | None]]] = {}
    with ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", namespaces):
                shared.append(
                    "".join(node.text or "" for node in item.findall(".//m:t", namespaces))
                )

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
        }
        sheet_targets: dict[str, str] = {}
        for item in workbook.findall("m:sheets/m:sheet", namespaces):
            relationship_id = item.attrib[f"{{{REL_NS}}}id"]
            target = targets[relationship_id].lstrip("/")
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            sheet_targets[item.attrib["name"]] = target

        for sheet_name, target in sheet_targets.items():
            root = ET.fromstring(archive.read(target))
            raw_rows: list[list[str | None]] = []
            maximum_column = 0
            for row in root.findall("m:sheetData/m:row", namespaces):
                values: dict[int, str | None] = {}
                for cell in row.findall("m:c", namespaces):
                    index = column_number(cell.attrib["r"])
                    maximum_column = max(maximum_column, index)
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find("m:v", namespaces)
                    inline_node = cell.find("m:is", namespaces)
                    value: str | None = None
                    if cell_type == "s" and value_node is not None:
                        value = shared[int(value_node.text or "0")]
                    elif cell_type == "inlineStr" and inline_node is not None:
                        value = "".join(
                            node.text or "" for node in inline_node.findall(".//m:t", namespaces)
                        )
                    elif value_node is not None:
                        value = value_node.text
                    values[index] = value
                raw_rows.append([values.get(index) for index in range(maximum_column + 1)])
            if not raw_rows:
                output[sheet_name] = []
                continue
            width = max(len(row) for row in raw_rows)
            raw_rows = [row + [None] * (width - len(row)) for row in raw_rows]
            headers = [str(value) if value is not None else "" for value in raw_rows[0]]
            output[sheet_name] = [
                {
                    header: row[index] if index < len(row) else None
                    for index, header in enumerate(headers)
                    if header
                }
                for row in raw_rows[1:]
                if any(value is not None for value in row)
            ]
    return output


def load_records(kind: str) -> list[dict[str, Any]]:
    records = list(jsonl(FIRST_PASS_ROOT / f"{kind}_records.jsonl"))
    if [int(row["index"]) for row in records] != list(range(len(records))):
        raise RuntimeError(f"{kind} records are not complete and ordered")
    return records


def effective_seeds(
    kind: str,
    sheets: dict[str, list[dict[str, str | None]]],
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, int], dict[str, Any]]:
    config = CONFIG[kind]
    assignments = sheets[config["assignment_sheet"]]
    cluster_rows = sheets[config["cluster_sheet"]]
    clusters = {row["cluster_id"]: row for row in cluster_rows}
    record_by_tag = {row["tag"]: row for row in records}

    if len(assignments) != 1000:
        raise RuntimeError(f"Expected 1,000 manual {kind} assignments, found {len(assignments)}")
    tags = [str(row["tag"]) for row in assignments]
    if len(tags) != len(set(tags)):
        raise RuntimeError(f"Duplicate manual {kind} tags")

    seed_members: dict[str, list[int]] = defaultdict(list)
    seed_meta: dict[str, dict[str, Any]] = {}
    frequency_mismatches = []
    missing_tags = []
    for row in assignments:
        tag = str(row["tag"])
        if tag not in record_by_tag:
            missing_tags.append(tag)
            continue
        record = record_by_tag[tag]
        expected_frequency = int(str(row["frequency"]))
        if int(record["frequency"]) != expected_frequency:
            frequency_mismatches.append(
                {
                    "tag": tag,
                    "workbook": expected_frequency,
                    "records": int(record["frequency"]),
                }
            )
        parent_cluster_id = str(row["cluster_id"])
        parent = clusters[parent_cluster_id]
        action = str(parent["action"])
        if action == "DO_NOT_AUTOMERGE":
            effective_id = f"{parent_cluster_id}::{tag}"
            canonical_label = tag
            effective_action = "DIRECTIONAL_SINGLETON"
        else:
            effective_id = parent_cluster_id
            canonical_label = str(parent["canonical_label"])
            effective_action = action
        seed_members[effective_id].append(int(record["index"]))
        seed_meta[effective_id] = {
            "seedClusterId": effective_id,
            "manualClusterId": parent_cluster_id,
            "canonicalLabel": canonical_label,
            "action": effective_action,
            "manualAction": action,
            "confidence": parent.get("confidence"),
            "basis": parent.get("basis"),
            "notes": parent.get("notes"),
            "qualityFlag": row.get("quality_flag"),
        }

    if missing_tags or frequency_mismatches:
        raise RuntimeError(
            json.dumps(
                {
                    "missingTags": missing_tags,
                    "frequencyMismatches": frequency_mismatches,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    seeds: list[dict[str, Any]] = []
    manual_assignment: dict[int, int] = {}
    for seed_index, effective_id in enumerate(seed_members):
        members = sorted(seed_members[effective_id])
        meta = seed_meta[effective_id]
        seed = {
            "seedIndex": seed_index,
            **meta,
            "manualMemberIndices": members,
            "manualMemberTags": [records[index]["tag"] for index in members],
        }
        seeds.append(seed)
        for member in members:
            if member in manual_assignment:
                raise RuntimeError(f"Manual {kind} record assigned twice: {member}")
            manual_assignment[member] = seed_index

    cluster_member_counts = Counter(str(row["cluster_id"]) for row in assignments)
    workbook_cluster_mismatches = []
    for cluster_id, row in clusters.items():
        expected = int(str(row["member_count"]))
        actual = cluster_member_counts[cluster_id]
        if expected != actual:
            workbook_cluster_mismatches.append(
                {
                    "clusterId": cluster_id,
                    "expected": expected,
                    "actual": actual,
                }
            )
    if workbook_cluster_mismatches:
        raise RuntimeError(
            f"Workbook {kind} cluster/member mismatch: {workbook_cluster_mismatches[:10]}"
        )

    validation = {
        "kind": kind,
        "manualAssignments": len(assignments),
        "manualClusters": len(clusters),
        "effectiveSeedCentroids": len(seeds),
        "expandableSeedCentroids": sum(
            seed["action"] in {"MERGE", "REVIEW_MERGE"} for seed in seeds
        ),
        "fixedSingletonSeedCentroids": sum(
            seed["action"] not in {"MERGE", "REVIEW_MERGE"} for seed in seeds
        ),
        "doNotAutomergeFamilies": sum(
            row["action"] == "DO_NOT_AUTOMERGE" for row in clusters.values()
        ),
        "frequencyMismatches": 0,
        "missingTags": 0,
        "manualMemberCountValidated": True,
    }
    return seeds, manual_assignment, validation


def build_centroids(
    vectors: np.ndarray,
    seeds: list[dict[str, Any]],
) -> np.ndarray:
    centroids = np.empty((len(seeds), vectors.shape[1]), dtype=np.float32)
    for seed in seeds:
        members = np.asarray(seed["manualMemberIndices"], dtype=np.int32)
        centroids[seed["seedIndex"]] = normalized(
            np.asarray(vectors[members], dtype=np.float32).mean(axis=0)
        )
    return centroids


def build_lexical_matrices(
    records: list[dict[str, Any]],
) -> tuple[csr_matrix, csr_matrix, dict[str, Any]]:
    texts = [str(row["normalizedTag"]) for row in records]
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True,
        norm="l2",
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        sublinear_tf=True,
        norm="l2",
    )
    word_matrix = word_vectorizer.fit_transform(texts).tocsr()
    char_matrix = char_vectorizer.fit_transform(texts).tocsr()
    return (
        word_matrix,
        char_matrix,
        {
            "wordAnalyzer": "word",
            "wordNgramRange": [1, 2],
            "wordFeatures": len(word_vectorizer.vocabulary_),
            "charAnalyzer": "char_wb",
            "charNgramRange": [3, 5],
            "charFeatures": len(char_vectorizer.vocabulary_),
            "wordWeight": 0.45,
            "charWeight": 0.55,
            "comparison": "maximum combined cosine to a reviewed member of the proposed seed",
        },
    )


def build_head_matrix(
    records: list[dict[str, Any]],
) -> tuple[csr_matrix, dict[str, Any]]:
    heads = [
        (str(row["normalizedTag"]).split()[0] if str(row["normalizedTag"]).split() else "")
        for row in records
    ]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        lowercase=True,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(heads).tocsr()
    return (
        matrix,
        {
            "analyzer": "char_wb",
            "ngramRange": [2, 5],
            "features": len(vectorizer.vocabulary_),
            "source": "first normalized predicate token",
        },
    )


def head_match(
    record_index: int,
    member_indices: list[int],
    head_matrix: csr_matrix | None,
    direction_signatures: list[bool] | None,
) -> tuple[float, int | None]:
    if head_matrix is None:
        return 1.0, None
    candidates = [
        index
        for index in member_indices
        if index != record_index
        and (
            direction_signatures is None
            or direction_signatures[index] == direction_signatures[record_index]
        )
    ]
    if not candidates:
        return 0.0, None
    scores = np.asarray((head_matrix[candidates] @ head_matrix[record_index].T).toarray()).reshape(
        -1
    )
    best_local = int(np.argmax(scores))
    return float(scores[best_local]), int(candidates[best_local])


def lexical_match(
    record_index: int,
    member_indices: list[int],
    word_matrix: csr_matrix,
    char_matrix: csr_matrix,
    direction_signatures: list[bool] | None = None,
) -> tuple[float, int | None, float, float]:
    candidates = [
        index
        for index in member_indices
        if index != record_index
        and (
            direction_signatures is None
            or direction_signatures[index] == direction_signatures[record_index]
        )
    ]
    if not candidates:
        return 0.0, None, 0.0, 0.0
    word_scores = np.asarray(
        (word_matrix[candidates] @ word_matrix[record_index].T).toarray()
    ).reshape(-1)
    char_scores = np.asarray(
        (char_matrix[candidates] @ char_matrix[record_index].T).toarray()
    ).reshape(-1)
    combined = 0.45 * word_scores + 0.55 * char_scores
    best_local = int(np.argmax(combined))
    return (
        float(combined[best_local]),
        int(candidates[best_local]),
        float(word_scores[best_local]),
        float(char_scores[best_local]),
    )


def top_two(similarities: np.ndarray) -> tuple[int, float, float]:
    if similarities.shape[0] == 1:
        return 0, float(similarities[0]), -1.0
    candidate = np.argpartition(similarities, -2)[-2:]
    ordered = candidate[np.argsort(-similarities[candidate])]
    return (
        int(ordered[0]),
        float(similarities[ordered[0]]),
        float(similarities[ordered[1]]),
    )


def calibrate_thresholds(
    kind: str,
    vectors: np.ndarray,
    seeds: list[dict[str, Any]],
    centroids: np.ndarray,
    manual_assignment: dict[int, int],
    word_matrix: csr_matrix,
    char_matrix: csr_matrix,
    direction_signatures: list[bool] | None,
    head_matrix: csr_matrix | None,
) -> dict[str, Any]:
    eligible_seed_indices = [
        int(seed["seedIndex"]) for seed in seeds if seed["action"] in {"MERGE", "REVIEW_MERGE"}
    ]
    eligible_position = {
        seed_index: position for position, seed_index in enumerate(eligible_seed_indices)
    }
    eligible_centroids = np.asarray(centroids[eligible_seed_indices], dtype=np.float32)
    knn = np.load(FIRST_PASS_ROOT / f"{kind}_knn.npz")
    neighbor_indices = knn["indices"]
    neighbor_similarities = knn["similarities"]
    evaluations = []
    for seed in seeds:
        members = seed["manualMemberIndices"]
        seed_index = int(seed["seedIndex"])
        if len(members) < 2 or seed_index not in eligible_position:
            continue
        member_matrix = np.asarray(vectors[members], dtype=np.float32)
        member_sum = member_matrix.sum(axis=0)
        for local, member_index in enumerate(members):
            leave_one_out = normalized(member_sum - member_matrix[local])
            similarities = (
                np.asarray(vectors[member_index], dtype=np.float32) @ eligible_centroids.T
            )
            similarities = np.asarray(similarities, dtype=np.float32)
            true_position = eligible_position[seed_index]
            similarities[true_position] = float(
                np.asarray(vectors[member_index], dtype=np.float32) @ leave_one_out
            )
            predicted_position, score, second = top_two(similarities)
            predicted = eligible_seed_indices[predicted_position]
            (
                lexical_similarity,
                lexical_member_index,
                lexical_word_similarity,
                lexical_char_similarity,
            ) = lexical_match(
                member_index,
                seeds[predicted]["manualMemberIndices"],
                word_matrix,
                char_matrix,
                direction_signatures,
            )
            head_similarity, head_member_index = head_match(
                member_index,
                seeds[predicted]["manualMemberIndices"],
                head_matrix,
                direction_signatures,
            )
            supporting_neighbors = [
                (
                    int(neighbor),
                    float(neighbor_similarity),
                )
                for neighbor, neighbor_similarity in zip(
                    neighbor_indices[member_index],
                    neighbor_similarities[member_index],
                )
                if int(neighbor) != member_index
                and manual_assignment.get(int(neighbor)) == predicted
            ]
            evaluations.append(
                {
                    "recordIndex": member_index,
                    "tag": seeds[seed_index]["manualMemberTags"][local],
                    "trueSeedIndex": seed_index,
                    "predictedSeedIndex": predicted,
                    "correct": predicted == seed_index,
                    "score": score,
                    "margin": score - second,
                    "trueSimilarity": float(similarities[true_position]),
                    "predictedSeedManualNeighborCount": len(supporting_neighbors),
                    "predictedSeedBestManualNeighborSimilarity": (
                        max(value for _, value in supporting_neighbors)
                        if supporting_neighbors
                        else None
                    ),
                    "lexicalSimilarity": lexical_similarity,
                    "lexicalWordSimilarity": lexical_word_similarity,
                    "lexicalCharSimilarity": lexical_char_similarity,
                    "lexicalBestManualMemberIndex": lexical_member_index,
                    "headSimilarity": head_similarity,
                    "headBestManualMemberIndex": head_member_index,
                }
            )

    minimum = float(CONFIG[kind]["minimum_similarity"])
    minimum_lexical = float(CONFIG[kind]["minimum_lexical_similarity"])
    minimum_head = float(CONFIG[kind]["minimum_head_similarity"])
    target_precision = 0.995
    best: tuple[int, float, float, float, float] | None = None
    selected: dict[str, Any] | None = None
    for score_threshold in np.arange(minimum, 0.991, 0.005):
        for margin_threshold in np.arange(0.0, 0.151, 0.005):
            for lexical_threshold in np.arange(minimum_lexical, 0.991, 0.01):
                accepted = [
                    row
                    for row in evaluations
                    if row["score"] >= score_threshold
                    and row["margin"] >= margin_threshold
                    and row["predictedSeedManualNeighborCount"] >= 1
                    and row["lexicalSimilarity"] >= lexical_threshold
                    and row["headSimilarity"] >= minimum_head
                ]
                if len(accepted) < 20:
                    continue
                correct = sum(row["correct"] for row in accepted)
                precision = correct / len(accepted)
                if precision < target_precision:
                    continue
                key = (
                    len(accepted),
                    precision,
                    -float(score_threshold),
                    -float(margin_threshold),
                    -float(lexical_threshold),
                )
                if best is None or key > best:
                    best = key
                    selected = {
                        "scoreThreshold": round(float(score_threshold), 6),
                        "marginThreshold": round(float(margin_threshold), 6),
                        "lexicalThreshold": round(float(lexical_threshold), 6),
                        "acceptedHoldouts": len(accepted),
                        "correctHoldouts": correct,
                        "estimatedPrecision": precision,
                        "estimatedCoverage": len(accepted) / len(evaluations),
                    }
    if selected is None:
        selected = {
            "scoreThreshold": round(minimum + 0.04, 6),
            "marginThreshold": 0.03,
            "lexicalThreshold": minimum_lexical,
            "acceptedHoldouts": 0,
            "correctHoldouts": 0,
            "estimatedPrecision": None,
            "estimatedCoverage": 0.0,
        }
    selected.update(
        {
            "kind": kind,
            "targetPrecision": target_precision,
            "domainMinimumLexicalSimilarity": minimum_lexical,
            "domainMinimumHeadSimilarity": minimum_head,
            "eligibleExpandableSeeds": len(eligible_seed_indices),
            "holdoutRows": len(evaluations),
            "rawLeaveOneOutAccuracy": (
                sum(row["correct"] for row in evaluations) / len(evaluations)
                if evaluations
                else None
            ),
            "evaluations": evaluations,
        }
    )
    return selected


def automatic_seed_assignments(
    kind: str,
    vectors: np.ndarray,
    records: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    centroids: np.ndarray,
    manual_assignment: dict[int, int],
    calibration: dict[str, Any],
    word_matrix: csr_matrix,
    char_matrix: csr_matrix,
    direction_signatures: list[bool] | None,
    head_matrix: csr_matrix | None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    score_threshold = float(calibration["scoreThreshold"])
    margin_threshold = float(calibration["marginThreshold"])
    lexical_threshold = float(calibration["lexicalThreshold"])
    head_threshold = float(calibration["domainMinimumHeadSimilarity"])
    eligible_seed_indices = [
        int(seed["seedIndex"]) for seed in seeds if seed["action"] in {"MERGE", "REVIEW_MERGE"}
    ]
    centroid_transpose = np.asarray(centroids[eligible_seed_indices].T, dtype=np.float32)
    assignments: dict[int, dict[str, Any]] = {}
    score_rows = []
    knn = np.load(FIRST_PASS_ROOT / f"{kind}_knn.npz")
    neighbor_indices = knn["indices"]
    neighbor_similarities = knn["similarities"]
    unseeded = [index for index in range(len(records)) if index not in manual_assignment]
    for start in range(0, len(unseeded), 512):
        indices = np.asarray(unseeded[start : start + 512], dtype=np.int32)
        matrix = np.asarray(vectors[indices], dtype=np.float32)
        similarities = matrix @ centroid_transpose
        partitioned = np.argpartition(similarities, -2, axis=1)[:, -2:]
        partition_scores = np.take_along_axis(similarities, partitioned, axis=1)
        order = np.argsort(-partition_scores, axis=1)
        ordered_indices = np.take_along_axis(partitioned, order, axis=1)
        ordered_scores = np.take_along_axis(partition_scores, order, axis=1)
        for local, record_index in enumerate(indices):
            best_seed = eligible_seed_indices[int(ordered_indices[local, 0])]
            score = float(ordered_scores[local, 0])
            second = float(ordered_scores[local, 1])
            margin = score - second
            (
                lexical_similarity,
                lexical_member_index,
                lexical_word_similarity,
                lexical_char_similarity,
            ) = lexical_match(
                int(record_index),
                seeds[best_seed]["manualMemberIndices"],
                word_matrix,
                char_matrix,
                direction_signatures,
            )
            head_similarity, head_member_index = head_match(
                int(record_index),
                seeds[best_seed]["manualMemberIndices"],
                head_matrix,
                direction_signatures,
            )
            supporting_neighbors = [
                (
                    int(neighbor),
                    float(neighbor_similarity),
                )
                for neighbor, neighbor_similarity in zip(
                    neighbor_indices[int(record_index)],
                    neighbor_similarities[int(record_index)],
                )
                if manual_assignment.get(int(neighbor)) == best_seed
            ]
            accepted = (
                score >= score_threshold
                and margin >= margin_threshold
                and len(supporting_neighbors) >= 1
                and lexical_similarity >= lexical_threshold
                and head_similarity >= head_threshold
            )
            row = {
                "recordIndex": int(record_index),
                "tag": records[int(record_index)]["tag"],
                "bestSeedIndex": best_seed,
                "bestSeedClusterId": seeds[best_seed]["seedClusterId"],
                "bestCanonicalLabel": seeds[best_seed]["canonicalLabel"],
                "score": round(score, 6),
                "margin": round(margin, 6),
                "lexicalSimilarity": round(lexical_similarity, 6),
                "lexicalWordSimilarity": round(lexical_word_similarity, 6),
                "lexicalCharSimilarity": round(lexical_char_similarity, 6),
                "lexicalBestManualMemberTag": (
                    records[lexical_member_index]["tag"]
                    if lexical_member_index is not None
                    else None
                ),
                "headSimilarity": round(head_similarity, 6),
                "headBestManualMemberTag": (
                    records[head_member_index]["tag"] if head_member_index is not None else None
                ),
                "manualNeighborSupportCount": len(supporting_neighbors),
                "bestManualNeighborTag": (
                    records[
                        max(
                            supporting_neighbors,
                            key=lambda item: item[1],
                        )[0]
                    ]["tag"]
                    if supporting_neighbors
                    else None
                ),
                "bestManualNeighborSimilarity": (
                    round(max(value for _, value in supporting_neighbors), 6)
                    if supporting_neighbors
                    else None
                ),
                "accepted": accepted,
            }
            score_rows.append(row)
            if accepted:
                assignments[int(record_index)] = row

    summary = {
        "unseededRecordsScored": len(unseeded),
        "eligibleExpandableSeeds": len(eligible_seed_indices),
        "automaticSeedAssignmentsAccepted": len(assignments),
        "automaticSeedAssignmentRate": len(assignments) / len(unseeded),
        "scoreThreshold": score_threshold,
        "marginThreshold": margin_threshold,
        "lexicalThreshold": lexical_threshold,
        "headSimilarityThreshold": head_threshold,
        "manualNeighborSupportRequired": True,
        "lexicalSimilarityRequired": True,
        "predicateHeadSimilarityRequired": head_matrix is not None,
        "passiveByDirectionCompatibilityRequired": (direction_signatures is not None),
        "scoreQuantiles": {
            str(quantile): float(np.quantile([row["score"] for row in score_rows], quantile))
            for quantile in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "marginQuantiles": {
            str(quantile): float(np.quantile([row["margin"] for row in score_rows], quantile))
            for quantile in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "lexicalSimilarityQuantiles": {
            str(quantile): float(
                np.quantile(
                    [row["lexicalSimilarity"] for row in score_rows],
                    quantile,
                )
            )
            for quantile in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "scoreRows": score_rows,
    }
    return assignments, summary


def residual_communities(
    kind: str,
    records: list[dict[str, Any]],
    excluded: set[int],
) -> tuple[list[set[int]], dict[str, Any]]:
    config = CONFIG[kind]
    knn = np.load(FIRST_PASS_ROOT / f"{kind}_knn.npz")
    indices = knn["indices"]
    similarities = knn["similarities"]
    residual = set(range(len(records))) - excluded
    graph = nx.Graph()
    graph.add_nodes_from(residual)
    minimum = float(config["minimum_similarity"])
    for source in residual:
        for rank in range(min(int(config["neighbor_count"]), indices.shape[1])):
            target = int(indices[source, rank])
            if target not in residual:
                continue
            similarity = float(similarities[source, rank])
            if similarity < minimum:
                continue
            weight = max(1e-6, (similarity - minimum) / max(1e-6, 1.0 - minimum))
            if not graph.has_edge(source, target) or weight > graph[source][target]["weight"]:
                graph.add_edge(
                    source,
                    target,
                    weight=weight,
                    similarity=similarity,
                )

    lexical_groups: dict[str, list[int]] = defaultdict(list)
    for index in residual:
        lexical_groups[records[index]["normalizedTag"]].append(index)
    for members in lexical_groups.values():
        if len(members) <= 1:
            continue
        anchor = members[0]
        for member in members[1:]:
            graph.add_edge(
                anchor,
                member,
                weight=1.0,
                similarity=1.0,
                lexicalExact=True,
            )

    communities = [
        set(group)
        for group in nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=float(config["resolution"]),
            seed=RANDOM_SEED,
        )
    ]
    communities.sort(
        key=lambda group: (
            -sum(int(records[index]["frequency"]) for index in group),
            -len(group),
            min(records[index]["tag"] for index in group),
        )
    )
    report = {
        "residualRecords": len(residual),
        "graphEdges": graph.number_of_edges(),
        "isolatedRecords": nx.number_of_isolates(graph),
        "communities": len(communities),
        "communitiesAtLeast5": sum(len(group) >= 5 for group in communities),
        "recordsInCommunitiesUnder5": sum(len(group) for group in communities if len(group) < 5),
    }
    return communities, report


def exemplar_rows(
    members: list[int],
    records: list[dict[str, Any]],
    similarities: dict[int, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    central_indices = sorted(
        members,
        key=lambda index: (
            -similarities[index],
            -int(records[index]["frequency"]),
            records[index]["tag"],
        ),
    )[:20]
    frequent_indices = sorted(
        members,
        key=lambda index: (
            -int(records[index]["frequency"]),
            -similarities[index],
            records[index]["tag"],
        ),
    )[:20]

    def row(index: int) -> dict[str, Any]:
        return {
            "tag": records[index]["tag"],
            "frequency": int(records[index]["frequency"]),
            "similarityToCentroid": round(similarities[index], 6),
        }

    return (
        [row(index) for index in central_indices],
        [row(index) for index in frequent_indices],
    )


def materialize_outputs(
    kind: str,
    records: list[dict[str, Any]],
    vectors: np.ndarray,
    seeds: list[dict[str, Any]],
    centroids: np.ndarray,
    manual_assignment: dict[int, int],
    automatic_assignment: dict[int, dict[str, Any]],
    residual: list[set[int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = CONFIG[kind]
    cluster_rows: list[dict[str, Any]] = []
    assignment_by_index: dict[int, dict[str, Any]] = {}
    automatic_by_seed: dict[int, list[int]] = defaultdict(list)
    for record_index, row in automatic_assignment.items():
        automatic_by_seed[int(row["bestSeedIndex"])].append(record_index)

    for seed in seeds:
        seed_index = int(seed["seedIndex"])
        manual_members = list(seed["manualMemberIndices"])
        automatic_members = sorted(automatic_by_seed.get(seed_index, []))
        members = sorted(manual_members + automatic_members)
        centroid = np.asarray(centroids[seed_index], dtype=np.float32)
        similarities = {
            index: float(np.asarray(vectors[index], dtype=np.float32) @ centroid)
            for index in members
        }
        central, frequent = exemplar_rows(members, records, similarities)
        cluster_rows.append(
            {
                "kind": kind,
                "clusterId": seed["seedClusterId"],
                "clusterType": "manual_seed",
                "manualClusterId": seed["manualClusterId"],
                "canonicalLabel": seed["canonicalLabel"],
                "action": seed["action"],
                "manualAction": seed["manualAction"],
                "confidence": seed["confidence"],
                "basis": seed["basis"],
                "notes": seed["notes"],
                "manualMemberCount": len(manual_members),
                "automaticMemberCount": len(automatic_members),
                "uniqueTags": len(members),
                "mentionCount": sum(int(records[index]["frequency"]) for index in members),
                "meanSimilarityToManualCentroid": round(
                    float(np.mean(list(similarities.values()))), 6
                ),
                "minimumSimilarityToManualCentroid": round(
                    float(np.min(list(similarities.values()))), 6
                ),
                "centralTags": central,
                "frequentTags": frequent,
            }
        )
        for index in manual_members:
            assignment_by_index[index] = {
                "kind": kind,
                "index": index,
                "tag": records[index]["tag"],
                "frequency": int(records[index]["frequency"]),
                "clusterId": seed["seedClusterId"],
                "canonicalLabel": seed["canonicalLabel"],
                "clusterType": "manual_seed",
                "assignmentMethod": "fixed_manual_assignment",
                "manualSeedMember": True,
                "similarityToCentroid": round(similarities[index], 6),
                "assignmentMargin": None,
            }
        for index in automatic_members:
            proposal = automatic_assignment[index]
            assignment_by_index[index] = {
                "kind": kind,
                "index": index,
                "tag": records[index]["tag"],
                "frequency": int(records[index]["frequency"]),
                "clusterId": seed["seedClusterId"],
                "canonicalLabel": seed["canonicalLabel"],
                "clusterType": "manual_seed",
                "assignmentMethod": "manual_centroid",
                "manualSeedMember": False,
                "similarityToCentroid": proposal["score"],
                "assignmentMargin": proposal["margin"],
                "lexicalSimilarity": proposal["lexicalSimilarity"],
                "lexicalBestManualMemberTag": proposal["lexicalBestManualMemberTag"],
                "headSimilarity": proposal["headSimilarity"],
                "headBestManualMemberTag": proposal["headBestManualMemberTag"],
            }

    prefix = config["prefix"]
    for ordinal, members_set in enumerate(residual, start=1):
        members = sorted(members_set)
        matrix = np.asarray(vectors[members], dtype=np.float32)
        centroid = normalized(matrix.mean(axis=0))
        similarities = {
            index: float(np.asarray(vectors[index], dtype=np.float32) @ centroid)
            for index in members
        }
        central, frequent = exemplar_rows(members, records, similarities)
        cluster_id = f"{prefix}_NEW_{ordinal:04d}"
        canonical_label = central[0]["tag"]
        cluster_rows.append(
            {
                "kind": kind,
                "clusterId": cluster_id,
                "clusterType": "residual_louvain",
                "manualClusterId": None,
                "canonicalLabel": canonical_label,
                "action": "NEW_CLUSTER",
                "manualAction": None,
                "confidence": None,
                "basis": "Louvain community among tags not confidently assigned to a manual centroid",
                "notes": None,
                "manualMemberCount": 0,
                "automaticMemberCount": len(members),
                "uniqueTags": len(members),
                "mentionCount": sum(int(records[index]["frequency"]) for index in members),
                "meanSimilarityToManualCentroid": None,
                "minimumSimilarityToManualCentroid": None,
                "meanSimilarityToClusterCentroid": round(
                    float(np.mean(list(similarities.values()))), 6
                ),
                "minimumSimilarityToClusterCentroid": round(
                    float(np.min(list(similarities.values()))), 6
                ),
                "centralTags": central,
                "frequentTags": frequent,
            }
        )
        for index in members:
            assignment_by_index[index] = {
                "kind": kind,
                "index": index,
                "tag": records[index]["tag"],
                "frequency": int(records[index]["frequency"]),
                "clusterId": cluster_id,
                "canonicalLabel": canonical_label,
                "clusterType": "residual_louvain",
                "assignmentMethod": "residual_louvain",
                "manualSeedMember": False,
                "similarityToCentroid": round(similarities[index], 6),
                "assignmentMargin": None,
            }

    if sorted(assignment_by_index) != list(range(len(records))):
        missing = sorted(set(range(len(records))) - set(assignment_by_index))
        raise RuntimeError(f"Incomplete {kind} assignments: {missing[:20]}")
    assignments = [assignment_by_index[index] for index in range(len(records))]
    return cluster_rows, assignments


def write_assignment_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "kind",
        "index",
        "tag",
        "frequency",
        "clusterId",
        "canonicalLabel",
        "clusterType",
        "assignmentMethod",
        "manualSeedMember",
        "similarityToCentroid",
        "assignmentMargin",
        "lexicalSimilarity",
        "lexicalBestManualMemberTag",
        "headSimilarity",
        "headBestManualMemberTag",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_kind(
    kind: str,
    sheets: dict[str, list[dict[str, str | None]]],
) -> dict[str, Any]:
    records = load_records(kind)
    vectors = np.load(FIRST_PASS_ROOT / f"{kind}_fused_vectors.npy", mmap_mode="r")
    seeds, manual_assignment, manual_validation = effective_seeds(kind, sheets, records)
    centroids = build_centroids(vectors, seeds)
    word_matrix, char_matrix, lexical_config = build_lexical_matrices(records)
    direction_signatures = (
        ["by" in str(row["normalizedTag"]).split() for row in records] if kind == "edge" else None
    )
    if kind == "edge":
        head_matrix, head_config = build_head_matrix(records)
    else:
        head_matrix, head_config = None, None
    np.save(OUTPUT_ROOT / f"{kind}_seed_centroids.npy", centroids)
    write_json(
        OUTPUT_ROOT / f"{kind}_seed_index.json",
        [
            {key: value for key, value in seed.items() if key != "manualMemberIndices"}
            for seed in seeds
        ],
    )

    calibration = calibrate_thresholds(
        kind,
        vectors,
        seeds,
        centroids,
        manual_assignment,
        word_matrix,
        char_matrix,
        direction_signatures,
        head_matrix,
    )
    write_json(
        OUTPUT_ROOT / f"{kind}_calibration.json",
        calibration,
    )
    automatic, scoring = automatic_seed_assignments(
        kind,
        vectors,
        records,
        seeds,
        centroids,
        manual_assignment,
        calibration,
        word_matrix,
        char_matrix,
        direction_signatures,
        head_matrix,
    )
    write_jsonl(
        OUTPUT_ROOT / f"{kind}_centroid_scores.jsonl",
        scoring.pop("scoreRows"),
    )
    excluded = set(manual_assignment) | set(automatic)
    communities, residual_report = residual_communities(kind, records, excluded)
    clusters, assignments = materialize_outputs(
        kind,
        records,
        vectors,
        seeds,
        centroids,
        manual_assignment,
        automatic,
        communities,
    )
    write_json(OUTPUT_ROOT / f"{kind}_clusters.json", clusters)
    write_jsonl(OUTPUT_ROOT / f"{kind}_assignments.jsonl", assignments)
    write_assignment_csv(OUTPUT_ROOT / f"{kind}_assignments.csv", assignments)

    manual_preserved = all(
        assignments[index]["clusterId"] == seeds[seed_index]["seedClusterId"]
        and assignments[index]["assignmentMethod"] == "fixed_manual_assignment"
        for index, seed_index in manual_assignment.items()
    )
    return {
        "kind": kind,
        "records": len(records),
        "mentions": sum(int(row["frequency"]) for row in records),
        "manualValidation": manual_validation,
        "manualAssignmentsPreserved": manual_preserved,
        "lexicalModel": lexical_config,
        "passiveByDirectionCompatibility": kind == "edge",
        "predicateHeadModel": head_config,
        "calibration": {key: value for key, value in calibration.items() if key != "evaluations"},
        "centroidScoring": scoring,
        "residualClustering": residual_report,
        "finalClusters": len(clusters),
        "seededClusters": len(seeds),
        "residualClusters": len(communities),
        "outputFiles": {
            "clusters": str(OUTPUT_ROOT / f"{kind}_clusters.json"),
            "assignmentsJsonl": str(OUTPUT_ROOT / f"{kind}_assignments.jsonl"),
            "assignmentsCsv": str(OUTPUT_ROOT / f"{kind}_assignments.csv"),
            "centroids": str(OUTPUT_ROOT / f"{kind}_seed_centroids.npy"),
        },
    }


def report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Konbaung V3 manual-seeded clustering",
        "",
        "This is a separate second clustering pass. It reads the original node and "
        "edge fused Gemini embedding matrices and the copied manual workbook. It "
        "does not read the entity-resolution/coreference outputs and does not "
        "modify the first clustering pass.",
        "",
        "## Method",
        "",
        "1. The 1,000 reviewed tags in each domain are fixed to their workbook clusters.",
        "2. Each effective manual cluster becomes an equal-tag-weighted centroid "
        "in the original 70% tag / 30% sentence-context Gemini embedding space.",
        "3. Word and character n-gram TF-IDF cosine similarity measures lexical "
        "identity against the reviewed members of each proposed cluster.",
        "4. Leave-one-out recovery of members from multi-member manual clusters "
        "sets centroid-score, margin, and lexical thresholds at a 99.5% target "
        "precision. The lexical search floor is 0.95 for nodes, where identity "
        "must be protected from related-name containment, and 0.80 for edges, "
        "where reviewed relation clusters legitimately contain more paraphrase.",
        "5. Edge proposals must also match a reviewed member's active/passive "
        "`BY` signature. This direction guard is derived from the workbook's "
        "explicit `KILLED` versus `KILLED_BY` do-not-automerge family.",
        "6. Edge proposals require character n-gram similarity of at least 0.50 "
        "between the predicate head and a reviewed member's predicate head, "
        "preserving inflectional variants while rejecting unrelated head verbs.",
        "7. Unreviewed tags crossing all thresholds join the corresponding fixed "
        "manual cluster only when one of its original 30 nearest Gemini neighbors "
        "is a reviewed member of that same cluster. Manual centroids never move.",
        "8. Remaining tags are clustered separately with the original pass's "
        "15-neighbor weighted graph, similarity floors, Louvain resolution 2.0, "
        "and random seed.",
        "",
        "The workbook's `DO_NOT_AUTOMERGE` edge families are treated as review "
        "families, not collapsed relations: their members become separate "
        "directional seed centroids.",
        "",
    ]
    for kind in ("node", "edge"):
        row = result[kind]
        validation = row["manualValidation"]
        calibration = row["calibration"]
        scoring = row["centroidScoring"]
        residual = row["residualClustering"]
        lines.extend(
            [
                f"## {kind.title()} results",
                "",
                f"- Source tags: **{row['records']:,}**",
                f"- Source mentions: **{row['mentions']:,}**",
                f"- Workbook assignments fixed: **{validation['manualAssignments']:,}**",
                f"- Workbook clusters: **{validation['manualClusters']:,}**",
                f"- Effective seed centroids: **{validation['effectiveSeedCentroids']:,}**",
                f"- Expandable `MERGE`/`REVIEW_MERGE` centroids: **"
                f"{validation['expandableSeedCentroids']:,}**",
                f"- Fixed singleton centroids: **{validation['fixedSingletonSeedCentroids']:,}**",
                f"- Manual assignments preserved: **"
                f"{str(row['manualAssignmentsPreserved']).lower()}**",
                f"- Leave-one-out rows: **{calibration['holdoutRows']:,}**",
                f"- Raw leave-one-out accuracy: **"
                f"{100 * calibration['rawLeaveOneOutAccuracy']:.2f}%**",
                f"- Selected centroid score threshold: **{calibration['scoreThreshold']:.3f}**",
                f"- Selected margin threshold: **{calibration['marginThreshold']:.3f}**",
                f"- Selected lexical threshold: **{calibration['lexicalThreshold']:.3f}**",
                f"- Predicate-head threshold: **{calibration['domainMinimumHeadSimilarity']:.3f}**",
                f"- Estimated precision among accepted holdouts: **"
                f"{100 * calibration['estimatedPrecision']:.2f}%**",
                f"- Automatically attached unreviewed tags: **"
                f"{scoring['automaticSeedAssignmentsAccepted']:,}**",
                f"- Tags left for residual Louvain: **{residual['residualRecords']:,}**",
                f"- Residual communities: **{residual['communities']:,}**",
                f"- Final clusters: **{row['finalClusters']:,}**",
                "",
            ]
        )
    lines.extend(
        [
            "## Safety and provenance",
            "",
            f"- Copied workbook SHA-256: `{result['workbookSha256']}`",
            "- Original first-pass files were hashed before and after this run and "
            f"remained unchanged: **{str(result['firstPassUnchanged']).lower()}**.",
            "- The old entity-resolution outputs were not inputs.",
            "- All output is under `konbaung_v3_manual_seeded_clustering_20260724`.",
            "",
            "## Interpretation",
            "",
            "Manual assignments are authoritative in this run. Automatic centroid "
            "attachments are statistically conservative proposals; residual "
            "communities are new, unlabeled clusters and should not inherit a "
            "manual label merely because one is nearby.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("node", "edge", "all"), default="all")
    args = parser.parse_args()
    if not WORKBOOK.exists():
        raise RuntimeError(f"Workbook not found: {WORKBOOK}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sheets = workbook_rows(WORKBOOK)
    required = {
        CONFIG[kind][field]
        for kind in ("node", "edge")
        for field in ("assignment_sheet", "cluster_sheet")
    }
    missing = required - set(sheets)
    if missing:
        raise RuntimeError(f"Workbook missing sheets: {sorted(missing)}")

    first_pass_paths = [
        FIRST_PASS_ROOT / "node_clusters.json",
        FIRST_PASS_ROOT / "node_assignments.jsonl",
        FIRST_PASS_ROOT / "edge_clusters.json",
        FIRST_PASS_ROOT / "edge_assignments.jsonl",
        FIRST_PASS_ROOT / "selected_resolution.json",
    ]
    before = {str(path): sha256(path) for path in first_pass_paths}
    result: dict[str, Any] = {
        "method": "fixed manual centroids plus calibrated centroid attachment and residual Louvain",
        "workbook": str(WORKBOOK),
        "workbookSha256": sha256(WORKBOOK),
        "entityResolutionInputsUsed": False,
    }
    kinds = ("node", "edge") if args.kind == "all" else (args.kind,)
    for kind in kinds:
        result[kind] = run_kind(kind, sheets)
    after = {str(path): sha256(path) for path in first_pass_paths}
    result["firstPassHashesBefore"] = before
    result["firstPassHashesAfter"] = after
    result["firstPassUnchanged"] = before == after
    if not result["firstPassUnchanged"]:
        raise RuntimeError("Original first-pass files changed during seeded run")
    write_json(OUTPUT_ROOT / "run_report.json", result)
    if args.kind == "all":
        (OUTPUT_ROOT / "MANUAL_SEEDED_CLUSTERING_REPORT.md").write_text(
            report_markdown(result), encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

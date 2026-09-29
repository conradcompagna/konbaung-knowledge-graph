from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from pipeline.resolution import v3_node_disambiguation as identity_model


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "konbaung_node_identity_disambiguation_complete_bundle_20260725"
FIRST_PASS = ROOT / "konbaung_v3_node_edge_clustering_first_pass_20260724"
OUTPUT = ROOT / "konbaung_node_identity_statistical_refinement_20260725"
DOCUMENTED_REDIRECTS = {"N-0397": "N-0501"}
DOCUMENTED_CANONICALS = {"N-0501": "Min Hla Min Khaung Kyaw"}
RANDOM_SEED = 20260725
CENTROID_NEIGHBORS = 30
REVIEW_SIZE = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def redirected(cluster_id: str) -> str:
    while cluster_id in DOCUMENTED_REDIRECTS:
        cluster_id = DOCUMENTED_REDIRECTS[cluster_id]
    return cluster_id


def add_candidate(
    sources: dict[tuple[int, int], set[str]],
    left: int,
    right: int,
    source: str,
) -> None:
    if left == right:
        return
    pair = (left, right) if left < right else (right, left)
    sources[pair].add(source)


def exact_centroid_knn(
    matrix: np.ndarray,
    neighbors: int,
    batch_size: int = 256,
) -> np.ndarray:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact centroid KNN")
    device = torch.device("cuda")
    corpus = torch.from_numpy(np.ascontiguousarray(normalize_rows(matrix))).to(device)
    result = np.empty((matrix.shape[0], neighbors), dtype=np.int32)
    with torch.inference_mode():
        for start in range(0, matrix.shape[0], batch_size):
            stop = min(start + batch_size, matrix.shape[0])
            similarities = corpus[start:stop] @ corpus.T
            local_rows = torch.arange(stop - start, device=device)
            similarities[local_rows, torch.arange(start, stop, device=device)] = -2.0
            indices = torch.topk(similarities, k=neighbors, dim=1, largest=True).indices
            result[start:stop] = indices.cpu().numpy().astype(np.int32)
    del corpus
    torch.cuda.empty_cache()
    return result


def build_cluster_state(
    assignments: list[dict[str, str]],
    records: list[dict[str, Any]],
    base_vectors: np.ndarray,
    context_vectors: np.ndarray,
    fused_vectors: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_cluster: dict[str, list[int]] = defaultdict(list)
    canonical_by_cluster: dict[str, str] = {}
    original_ids: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(assignments):
        if int(records[index]["index"]) != index:
            raise RuntimeError("Source records are unordered")
        if str(records[index]["tag"]) != row["display_tag"]:
            raise RuntimeError(f"Tag/vector alignment mismatch at rank {row['rank']}")
        original = row["final_cluster_id"]
        cluster_id = redirected(original)
        by_cluster[cluster_id].append(index)
        original_ids[cluster_id].add(original)
        canonical_by_cluster.setdefault(cluster_id, row["canonical_label"])
    canonical_by_cluster.update(DOCUMENTED_CANONICALS)

    cluster_ids = sorted(
        by_cluster,
        key=lambda value: int(value.split("-")[1]),
    )
    clusters: list[dict[str, Any]] = []
    tag_to_cluster = np.empty(len(records), dtype=np.int32)
    base_centroids = np.empty((len(cluster_ids), base_vectors.shape[1]), dtype=np.float32)
    context_centroids = np.empty((len(cluster_ids), context_vectors.shape[1]), dtype=np.float32)
    fused_centroids = np.empty((len(cluster_ids), fused_vectors.shape[1]), dtype=np.float32)
    for cluster_index, cluster_id in enumerate(cluster_ids):
        members = sorted(by_cluster[cluster_id])
        tag_to_cluster[members] = cluster_index
        base_centroids[cluster_index] = np.asarray(base_vectors[members], dtype=np.float32).mean(
            axis=0
        )
        context_centroids[cluster_index] = np.asarray(
            context_vectors[members], dtype=np.float32
        ).mean(axis=0)
        fused_centroids[cluster_index] = np.asarray(fused_vectors[members], dtype=np.float32).mean(
            axis=0
        )
        display_tags = [assignments[index]["display_tag"] for index in members]
        clusters.append(
            {
                "clusterIndex": cluster_index,
                "clusterId": cluster_id,
                "canonicalLabel": canonical_by_cluster[cluster_id],
                "tagCount": len(members),
                "totalFrequency": sum(int(assignments[index]["frequency"]) for index in members),
                "memberIndices": members,
                "memberTags": display_tags,
                "sourceClusterIds": sorted(original_ids[cluster_id]),
                "documentedFrozenMerge": (set(original_ids[cluster_id]) == {"N-0397", "N-0501"}),
            }
        )
    return (
        clusters,
        tag_to_cluster,
        normalize_rows(base_centroids),
        normalize_rows(context_centroids),
        normalize_rows(fused_centroids),
    )


def build_cluster_candidates(
    clusters: list[dict[str, Any]],
    tag_to_cluster: np.ndarray,
    fused_centroids: np.ndarray,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], set[str]]]:
    sources: dict[tuple[int, int], set[str]] = defaultdict(set)

    centroid_knn = exact_centroid_knn(fused_centroids, CENTROID_NEIGHBORS)
    for left in range(centroid_knn.shape[0]):
        for right in centroid_knn[left]:
            add_candidate(sources, left, int(right), "fused_centroid_knn")
    np.save(OUTPUT / "fused_centroid_knn_indices.npy", centroid_knn)

    tag_knn = np.load(FIRST_PASS / "node_knn.npz")["indices"]
    for tag_index in range(tag_knn.shape[0]):
        left = int(tag_to_cluster[tag_index])
        for neighbor in tag_knn[tag_index]:
            right = int(tag_to_cluster[int(neighbor)])
            add_candidate(sources, left, right, "member_tag_knn")

    normalized_groups: dict[str, list[int]] = defaultdict(list)
    compact_groups: dict[str, list[int]] = defaultdict(list)
    order_groups: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        tokens = identity_model.normalized_tokens(str(cluster["canonicalLabel"]))
        normalized_groups[" ".join(tokens)].append(index)
        compact_groups["".join(tokens)].append(index)
        order_groups[
            " ".join(sorted(token for token in tokens if token not in {"of", "the"}))
        ].append(index)
    for name, groups in (
        ("canonical_normalized_exact", normalized_groups),
        ("canonical_compact_exact", compact_groups),
        ("canonical_order_signature", order_groups),
    ):
        for key, members in groups.items():
            if not key or len(members) < 2 or len(members) > 100:
                continue
            for left_position in range(len(members)):
                for right_position in range(left_position + 1, len(members)):
                    add_candidate(
                        sources,
                        members[left_position],
                        members[right_position],
                        name,
                    )
    return sorted(sources), sources


def train_tag_pair_model(
    records: list[dict[str, Any]],
    assignments: list[dict[str, str]],
    base_vectors: np.ndarray,
    context_vectors: np.ndarray,
    fused_vectors: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    manual_seed = [redirected(row["final_cluster_id"]) for row in assignments]
    manual_members: dict[str, list[int]] = defaultdict(list)
    for index, seed in enumerate(manual_seed):
        manual_members[seed].append(index)

    tokens = [identity_model.normalized_tokens(str(row["tag"])) for row in records]
    normalized_text = [" ".join(row) for row in tokens]
    numeric = [identity_model.numeric_signature(row) for row in tokens]
    grams = [identity_model.character_ngrams(row) for row in tokens]
    token_document_frequency = Counter(token for row in tokens for token in set(row))
    token_idf = {
        token: np.log((len(records) + 1) / (token_document_frequency[token] + 1)) + 1
        for token in token_document_frequency
    }
    tag_candidates, candidate_report = identity_model.build_candidate_pairs(records, tokens, grams)
    word_matrix, char_matrix, text_report = identity_model.build_text_features(
        records, normalized_text
    )
    builder = identity_model.PairFeatures(
        base_vectors,
        context_vectors,
        fused_vectors,
        word_matrix,
        char_matrix,
        tokens,
        normalized_text,
        numeric,
        token_idf,
    )
    rng = np.random.default_rng(RANDOM_SEED)
    training_pairs, labels, training_report = identity_model.manual_training_pairs(
        tag_candidates, manual_seed, manual_members, rng
    )
    features = builder.matrix(training_pairs)
    model, calibration = identity_model.train_model(features, labels)
    return (
        model,
        {
            "candidateGeneration": candidate_report,
            "textFeatures": text_report,
            "training": training_report,
            "calibration": calibration,
        },
    )


def cluster_pair_features(
    clusters: list[dict[str, Any]],
    candidate_pairs: list[tuple[int, int]],
    base_centroids: np.ndarray,
    context_centroids: np.ndarray,
    fused_centroids: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    tokens = [
        identity_model.normalized_tokens(str(cluster["canonicalLabel"])) for cluster in clusters
    ]
    normalized_text = [" ".join(row) for row in tokens]
    numeric = [identity_model.numeric_signature(row) for row in tokens]
    token_document_frequency = Counter(token for row in tokens for token in set(row))
    token_idf = {
        token: np.log((len(clusters) + 1) / (token_document_frequency[token] + 1)) + 1
        for token in token_document_frequency
    }
    word_matrix, char_matrix, text_report = identity_model.build_text_features(
        clusters, normalized_text
    )
    builder = identity_model.PairFeatures(
        base_centroids,
        context_centroids,
        fused_centroids,
        word_matrix,
        char_matrix,
        tokens,
        normalized_text,
        numeric,
        token_idf,
    )
    features = np.empty(
        (
            len(candidate_pairs),
            len(identity_model.FEATURE_NAMES),
        ),
        dtype=np.float32,
    )
    batch_size = 10000
    for start in range(0, len(candidate_pairs), batch_size):
        stop = min(start + batch_size, len(candidate_pairs))
        features[start:stop] = builder.matrix(candidate_pairs[start:stop])
    return features, text_report


def pair_type(left: dict[str, Any], right: dict[str, Any]) -> str:
    left_singleton = int(left["tagCount"]) == 1
    right_singleton = int(right["tagCount"]) == 1
    if left_singleton and right_singleton:
        return "singleton_singleton"
    if left_singleton or right_singleton:
        return "singleton_cluster"
    return "cluster_cluster"


def candidate_rows(
    clusters: list[dict[str, Any]],
    pairs: list[tuple[int, int]],
    sources: dict[tuple[int, int], set[str]],
    features: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, (left_index, right_index) in enumerate(pairs):
        left = clusters[left_index]
        right = clusters[right_index]
        row: dict[str, Any] = {
            "candidateIndex": row_index,
            "leftIndex": left_index,
            "rightIndex": right_index,
            "leftClusterId": left["clusterId"],
            "rightClusterId": right["clusterId"],
            "leftCanonical": left["canonicalLabel"],
            "rightCanonical": right["canonicalLabel"],
            "leftTagCount": left["tagCount"],
            "rightTagCount": right["tagCount"],
            "leftFrequency": left["totalFrequency"],
            "rightFrequency": right["totalFrequency"],
            "pairType": pair_type(left, right),
            "modelProbability": float(probabilities[row_index]),
            "candidateSources": sorted(sources[(left_index, right_index)]),
        }
        for feature_index, feature_name in enumerate(identity_model.FEATURE_NAMES):
            row[feature_name] = float(features[row_index, feature_index])
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -float(row["modelProbability"]),
            -float(row["geminiFusedCosine"]),
            str(row["leftClusterId"]),
            str(row["rightClusterId"]),
        )
    )
    return rows


def select_nonoverlapping_review(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    used: set[int] = set()
    selected: list[dict[str, Any]] = []
    for row in rows:
        left = int(row["leftIndex"])
        right = int(row["rightIndex"])
        if left in used or right in used:
            continue
        selected.append(row)
        used.add(left)
        used.add(right)
        if len(selected) == size:
            break
    if len(selected) != size:
        raise RuntimeError(f"Could select only {len(selected)} disjoint review pairs")
    return selected


def write_review_markdown(
    path: Path,
    selected: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
) -> None:
    lines = [
        "# Statistical Pairwise Refinement — Pass 1 Review",
        "",
        "Tentative proposals are ordered by model score. No proposal in this "
        "file is committed until reviewed. Stop at the first pair that is "
        "not the same underlying identity.",
        "",
    ]
    for ordinal, row in enumerate(selected, start=1):
        left = clusters[int(row["leftIndex"])]
        right = clusters[int(row["rightIndex"])]
        lines.extend(
            [
                f"## {ordinal:03d}. {left['canonicalLabel']} ↔ {right['canonicalLabel']}",
                "",
                f"- Type: `{row['pairType']}`",
                f"- Model probability: `{float(row['modelProbability']):.6f}`",
                f"- Fused centroid cosine: `{float(row['geminiFusedCosine']):.6f}`",
                f"- Tag centroid cosine: `{float(row['geminiTagCosine']):.6f}`",
                f"- Context centroid cosine: `{float(row['geminiContextCosine']):.6f}`",
                f"- Candidate sources: `{', '.join(row['candidateSources'])}`",
                f"- Left `{left['clusterId']}` tags: " + " | ".join(left["memberTags"]),
                f"- Right `{right['clusterId']}` tags: " + " | ".join(right["memberTags"]),
                "- Review: `PENDING`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protected = [
        BUNDLE / "konbaung_node_identity_tag_assignments_complete.csv",
        BUNDLE / "konbaung_node_identity_component_assignments_complete.csv",
        BUNDLE / "konbaung_node_identity_clusters_complete.csv",
        BUNDLE / "konbaung_node_identity_disambiguation_validation.json",
        FIRST_PASS / "node_records.jsonl",
        FIRST_PASS / "node_base_vectors.npy",
        FIRST_PASS / "node_context_vectors.npy",
        FIRST_PASS / "node_fused_vectors.npy",
        FIRST_PASS / "node_knn.npz",
    ]
    hashes_before = {str(path): sha256(path) for path in protected}

    assignments = read_csv(BUNDLE / "konbaung_node_identity_tag_assignments_complete.csv")
    records = list(identity_model.jsonl(FIRST_PASS / "node_records.jsonl"))
    if len(assignments) != len(records) or len(records) != 23890:
        raise RuntimeError("Expected exactly 23,890 aligned tags")
    assignment_by_display_tag = {row["display_tag"]: row for row in assignments}
    if len(assignment_by_display_tag) != len(assignments):
        raise RuntimeError("Bundle display tags are not unique")
    if set(assignment_by_display_tag) != {str(row["tag"]) for row in records}:
        raise RuntimeError("Bundle/source tag sets do not align")
    assignments = [assignment_by_display_tag[str(row["tag"])] for row in records]

    base_vectors = np.load(FIRST_PASS / "node_base_vectors.npy", mmap_mode="r")
    context_vectors = np.load(FIRST_PASS / "node_context_vectors.npy", mmap_mode="r")
    fused_vectors = np.load(FIRST_PASS / "node_fused_vectors.npy", mmap_mode="r")
    (
        clusters,
        tag_to_cluster,
        base_centroids,
        context_centroids,
        fused_centroids,
    ) = build_cluster_state(
        assignments,
        records,
        base_vectors,
        context_vectors,
        fused_vectors,
    )
    if len(clusters) != 19002:
        raise RuntimeError(f"Expected 19,002 corrected clusters, found {len(clusters)}")

    np.save(OUTPUT / "base_cluster_centroids.npy", base_centroids)
    np.save(OUTPUT / "context_cluster_centroids.npy", context_centroids)
    np.save(OUTPUT / "fused_cluster_centroids.npy", fused_centroids)
    np.save(OUTPUT / "tag_to_corrected_cluster_index.npy", tag_to_cluster)
    write_jsonl(OUTPUT / "corrected_initial_clusters.jsonl", clusters)

    corrected_assignments: list[dict[str, Any]] = []
    for index, row in enumerate(assignments):
        cluster = clusters[int(tag_to_cluster[index])]
        corrected_assignments.append(
            {
                **row,
                "working_cluster_id": cluster["clusterId"],
                "working_canonical_label": cluster["canonicalLabel"],
                "documented_frozen_merge_applied": (row["final_cluster_id"] == "N-0397"),
            }
        )
    write_csv(
        OUTPUT / "corrected_initial_tag_assignments.csv",
        corrected_assignments,
        list(corrected_assignments[0]),
    )

    print("Training pair classifier...", flush=True)
    model, training_report = train_tag_pair_model(
        records,
        corrected_assignments,
        base_vectors,
        context_vectors,
        fused_vectors,
    )
    joblib.dump(model, OUTPUT / "cluster_pair_classifier.joblib")

    print("Generating all candidate types...", flush=True)
    pairs, sources = build_cluster_candidates(clusters, tag_to_cluster, fused_centroids)
    feature_matrix, cluster_text_report = cluster_pair_features(
        clusters,
        pairs,
        base_centroids,
        context_centroids,
        fused_centroids,
    )
    probabilities = model.predict_proba(feature_matrix)[:, 1]
    rows = candidate_rows(clusters, pairs, sources, feature_matrix, probabilities)
    write_jsonl(OUTPUT / "all_cluster_pair_candidates.jsonl", rows)

    top_fields = [key for key in rows[0] if key != "candidateSources"] + ["candidateSources"]
    csv_rows = [
        {
            **row,
            "candidateSources": "; ".join(row["candidateSources"]),
        }
        for row in rows[:5000]
    ]
    write_csv(
        OUTPUT / "top_5000_cluster_pair_candidates.csv",
        csv_rows,
        top_fields,
    )

    selected = select_nonoverlapping_review(rows, REVIEW_SIZE)
    write_json(OUTPUT / "pass_01_tentative_review_pairs.json", selected)
    write_review_markdown(OUTPUT / "PASS_01_MANUAL_REVIEW.md", selected, clusters)

    pair_type_counts = Counter(row["pairType"] for row in rows)
    selected_type_counts = Counter(row["pairType"] for row in selected)
    hashes_after = {str(path): sha256(path) for path in protected}
    if hashes_before != hashes_after:
        raise RuntimeError("A protected source changed during refinement")
    report = {
        "method": (
            "manual-cluster centroid averaging plus supervised pairwise "
            "classification and human stop-on-first-error review"
        ),
        "inputManualClusters": 19003,
        "documentedFrozenMerge": {
            "retiredClusterId": "N-0397",
            "survivingClusterId": "N-0501",
            "canonicalLabel": "Min Hla Min Khaung Kyaw",
        },
        "correctedInitialClusters": len(clusters),
        "rawTags": len(records),
        "centroidWeighting": "equal weight per unique raw tag",
        "centroidViews": {
            "tag": int(base_centroids.shape[1]),
            "context": int(context_centroids.shape[1]),
            "fused": int(fused_centroids.shape[1]),
        },
        "candidatePairs": len(rows),
        "candidatePairTypes": dict(pair_type_counts),
        "centroidNeighborsPerItem": CENTROID_NEIGHBORS,
        "reviewPairs": len(selected),
        "reviewPairTypes": dict(selected_type_counts),
        "training": training_report,
        "clusterTextFeatures": cluster_text_report,
        "protectedHashesBefore": hashes_before,
        "protectedHashesAfter": hashes_after,
        "protectedSourcesUnchanged": hashes_before == hashes_after,
        "newEmbeddingApiCalls": 0,
        "reviewStatus": "PENDING",
    }
    write_json(OUTPUT / "pass_01_run_report.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()

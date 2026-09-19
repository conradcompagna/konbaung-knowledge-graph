from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.random_projection import SparseRandomProjection

import konbaung_v3_eight_view_embeddings as embedding_run


EMBEDDING_ROOT = embedding_run.OUTPUT_ROOT
OUTPUT_ROOT = (
    embedding_run.ROOT / "konbaung_v3_node_edge_clustering_first_pass_20260724"
)

BASE_WEIGHT = 0.70
CONTEXT_WEIGHT = 0.30
PROJECTION_DIMENSIONS = 256
PROJECTED_NEIGHBORS = 61
EXACT_NEIGHBORS = 30
RANDOM_SEED = 20260724

KIND_CONFIG = {
    "node": {
        "base_shard": "argument",
        "context_shard": "argument_context",
        "record_count": 23_890,
    },
    "edge": {
        "base_shard": "predicate",
        "context_shard": "predicate_context",
        "record_count": 11_886,
    },
}


def now_iso() -> str:
    return embedding_run.now_iso()


def jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    return embedding_run.iter_jsonl(path)


def atomic_json(path: Path, value: Any) -> None:
    embedding_run.atomic_write_json(path, value)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    embedding_run.write_jsonl(path, rows)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("Cannot normalize a zero or non-finite vector")
    return vector / norm


def normalized_lexical_tag(tag: str) -> str:
    value = tag.casefold()
    value = re.sub(r"[_\-/]+", " ", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


class VectorStore:
    def __init__(self, shard: str) -> None:
        self.shard = shard
        vector_path = EMBEDDING_ROOT / "vectors" / f"{shard}.npy"
        keys_path = EMBEDDING_ROOT / "vectors" / f"{shard}.keys.jsonl"
        manifest_path = EMBEDDING_ROOT / "vectors" / f"{shard}.manifest.json"
        manifest = embedding_run.read_json(manifest_path)
        if not manifest.get("certified"):
            raise RuntimeError(f"Embedding shard is not certified: {shard}")
        self.vectors = np.load(vector_path, mmap_mode="r")
        self.key_to_row = {
            row["key"]: int(row["row"]) for row in jsonl_rows(keys_path)
        }
        if len(self.key_to_row) != self.vectors.shape[0]:
            raise RuntimeError(f"Key/vector count mismatch in {shard}")

    def get(self, key: str) -> np.ndarray:
        return np.asarray(self.vectors[self.key_to_row[key]], dtype=np.float32)

    def combine(self, keys: list[str]) -> np.ndarray:
        if not keys:
            raise RuntimeError(f"Empty embedding key list for {self.shard}")
        if len(keys) == 1:
            return self.get(keys[0])
        combined = np.mean([self.get(key) for key in keys], axis=0)
        return l2_normalize(combined.astype(np.float32))


def feature_paths(kind: str) -> dict[str, Path]:
    return {
        "records": OUTPUT_ROOT / f"{kind}_records.jsonl",
        "base": OUTPUT_ROOT / f"{kind}_base_vectors.npy",
        "context": OUTPUT_ROOT / f"{kind}_context_vectors.npy",
        "fused": OUTPUT_ROOT / f"{kind}_fused_vectors.npy",
        "feature_manifest": OUTPUT_ROOT / f"{kind}_feature_manifest.json",
        "knn": OUTPUT_ROOT / f"{kind}_knn.npz",
        "knn_manifest": OUTPUT_ROOT / f"{kind}_knn_manifest.json",
    }


def collect_tag_metadata() -> tuple[
    dict[str, dict[str, Any]], dict[str, dict[str, Any]]
]:
    node_data: dict[str, dict[str, Any]] = {}
    edge_data: dict[str, dict[str, Any]] = {}
    for occurrence in jsonl_rows(EMBEDDING_ROOT / "occurrences.jsonl"):
        keys = occurrence["embeddingKeys"]
        occurrence_id = occurrence["occurrenceId"]
        sid = occurrence["sid"]

        # Subject and object feed exactly the same node pool. Position is not retained.
        for tag, base_view, context_view in (
            (occurrence["subject"], "S", "SBE"),
            (occurrence["object"], "O", "OBE"),
        ):
            base_keys = keys[base_view]
            if len(base_keys) != 1:
                raise RuntimeError(
                    f"Node base view unexpectedly chunked: {occurrence_id}:{base_view}"
                )
            item = node_data.setdefault(
                tag,
                {
                    "tag": tag,
                    "frequency": 0,
                    "baseKey": base_keys[0],
                    "contextKeyLists": [],
                    "sampleOccurrences": [],
                },
            )
            if item["baseKey"] != base_keys[0]:
                raise RuntimeError(f"Node tag maps to multiple base keys: {tag}")
            item["frequency"] += 1
            item["contextKeyLists"].append(keys[context_view])
            if len(item["sampleOccurrences"]) < 5:
                item["sampleOccurrences"].append(
                    {"occurrenceId": occurrence_id, "sid": sid}
                )

        tag = occurrence["predicate"]
        base_keys = keys["P"]
        if len(base_keys) != 1:
            raise RuntimeError(f"Edge base view unexpectedly chunked: {occurrence_id}")
        item = edge_data.setdefault(
            tag,
            {
                "tag": tag,
                "frequency": 0,
                "baseKey": base_keys[0],
                "contextKeyLists": [],
                "sampleOccurrences": [],
            },
        )
        if item["baseKey"] != base_keys[0]:
            raise RuntimeError(f"Edge tag maps to multiple base keys: {tag}")
        item["frequency"] += 1
        item["contextKeyLists"].append(keys["PBE"])
        if len(item["sampleOccurrences"]) < 5:
            item["sampleOccurrences"].append(
                {"occurrenceId": occurrence_id, "sid": sid}
            )
    return node_data, edge_data


def build_kind_features(
    kind: str,
    metadata: dict[str, dict[str, Any]],
    base_store: VectorStore,
    context_store: VectorStore,
) -> dict[str, Any]:
    paths = feature_paths(kind)
    expected = KIND_CONFIG[kind]["record_count"]
    if len(metadata) != expected:
        raise RuntimeError(
            f"Expected {expected:,} {kind} tags, found {len(metadata):,}"
        )

    ordered = sorted(
        metadata.values(), key=lambda row: (-int(row["frequency"]), row["tag"])
    )
    count = len(ordered)
    base_vectors = np.lib.format.open_memmap(
        paths["base"], mode="w+", dtype=np.float32, shape=(count, 768)
    )
    context_vectors = np.lib.format.open_memmap(
        paths["context"], mode="w+", dtype=np.float32, shape=(count, 768)
    )
    fused_vectors = np.lib.format.open_memmap(
        paths["fused"], mode="w+", dtype=np.float32, shape=(count, 1536)
    )
    record_rows = []

    for index, item in enumerate(ordered):
        base = l2_normalize(base_store.get(item["baseKey"]).astype(np.float32))
        context_sum = np.zeros(768, dtype=np.float32)
        for key_list in item["contextKeyLists"]:
            context_sum += context_store.combine(key_list)
        context = l2_normalize(context_sum)
        fused = np.concatenate(
            (
                math.sqrt(BASE_WEIGHT) * base,
                math.sqrt(CONTEXT_WEIGHT) * context,
            )
        ).astype(np.float32)
        fused = l2_normalize(fused)
        base_vectors[index] = base
        context_vectors[index] = context
        fused_vectors[index] = fused
        record_rows.append(
            {
                "index": index,
                "tag": item["tag"],
                "normalizedTag": normalized_lexical_tag(item["tag"]),
                "frequency": int(item["frequency"]),
                "baseKey": item["baseKey"],
                "contextOccurrences": len(item["contextKeyLists"]),
                "sampleOccurrences": item["sampleOccurrences"],
            }
        )

    base_vectors.flush()
    context_vectors.flush()
    fused_vectors.flush()
    write_jsonl(paths["records"], record_rows)
    manifest = {
        "createdAt": now_iso(),
        "kind": kind,
        "records": count,
        "mentions": sum(int(row["frequency"]) for row in record_rows),
        "dimensions": {
            "base": 768,
            "context": 768,
            "fused": 1536,
        },
        "similarityFormula": (
            f"{BASE_WEIGHT:.2f} * cosine(base, base) + "
            f"{CONTEXT_WEIGHT:.2f} * cosine(contextCentroid, contextCentroid)"
        ),
        "baseWeight": BASE_WEIGHT,
        "contextWeight": CONTEXT_WEIGHT,
        "subjectObjectRoleUsed": False if kind == "node" else None,
        "paths": {name: str(path) for name, path in paths.items() if path.exists()},
    }
    atomic_json(paths["feature_manifest"], manifest)
    return manifest


def build_features() -> None:
    OUTPUT_ROOT.mkdir(exist_ok=True)
    node_data, edge_data = collect_tag_metadata()
    stores = {
        shard: VectorStore(shard)
        for shard in (
            "argument",
            "argument_context",
            "predicate",
            "predicate_context",
        )
    }
    result = {
        "createdAt": now_iso(),
        "embeddingRoot": str(EMBEDDING_ROOT),
        "sourceManifestSha256": embedding_run.sha256_file(
            EMBEDDING_ROOT / "source_manifest.json"
        ),
        "embeddingFinalReportSha256": embedding_run.sha256_file(
            EMBEDDING_ROOT / "final_report.json"
        ),
        "node": build_kind_features(
            "node", node_data, stores["argument"], stores["argument_context"]
        ),
        "edge": build_kind_features(
            "edge", edge_data, stores["predicate"], stores["predicate_context"]
        ),
    }
    atomic_json(OUTPUT_ROOT / "feature_build_report.json", result)
    print(json.dumps(result, indent=2))


def build_knn(kind: str) -> dict[str, Any]:
    paths = feature_paths(kind)
    vectors = np.load(paths["fused"], mmap_mode="r")
    projector = SparseRandomProjection(
        n_components=PROJECTION_DIMENSIONS,
        density="auto",
        random_state=RANDOM_SEED,
    )
    projected = projector.fit_transform(vectors)
    projected = normalize(projected, norm="l2", copy=False)
    candidate_count = min(PROJECTED_NEIGHBORS, vectors.shape[0])
    neighbors = NearestNeighbors(
        n_neighbors=candidate_count,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    neighbors.fit(projected)
    _, candidate_indices = neighbors.kneighbors(projected, return_distance=True)

    exact_indices = np.empty(
        (vectors.shape[0], EXACT_NEIGHBORS), dtype=np.int32
    )
    exact_similarities = np.empty(
        (vectors.shape[0], EXACT_NEIGHBORS), dtype=np.float32
    )
    for start in range(0, vectors.shape[0], 512):
        end = min(vectors.shape[0], start + 512)
        for local, row_index in enumerate(range(start, end)):
            candidates = candidate_indices[row_index]
            candidates = candidates[candidates != row_index]
            candidate_vectors = np.asarray(vectors[candidates], dtype=np.float32)
            similarities = candidate_vectors @ np.asarray(
                vectors[row_index], dtype=np.float32
            )
            order = np.argsort(-similarities)[:EXACT_NEIGHBORS]
            selected = candidates[order]
            selected_similarities = similarities[order]
            if len(selected) < EXACT_NEIGHBORS:
                raise RuntimeError(f"Insufficient exact neighbors for {kind}")
            exact_indices[row_index] = selected
            exact_similarities[row_index] = selected_similarities

    np.savez(
        paths["knn"],
        indices=exact_indices,
        similarities=exact_similarities,
    )
    quantiles = {}
    for rank in (1, 5, 10, 15, 20, 30):
        values = exact_similarities[:, rank - 1]
        quantiles[str(rank)] = {
            "min": float(values.min()),
            "p05": float(np.quantile(values, 0.05)),
            "p25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "p75": float(np.quantile(values, 0.75)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
        }
    manifest = {
        "createdAt": now_iso(),
        "kind": kind,
        "records": int(vectors.shape[0]),
        "fusedDimensions": int(vectors.shape[1]),
        "projectionDimensions": PROJECTION_DIMENSIONS,
        "projectedCandidateNeighbors": candidate_count - 1,
        "retainedExactNeighbors": EXACT_NEIGHBORS,
        "similarityQuantilesByNeighborRank": quantiles,
        "path": str(paths["knn"]),
    }
    atomic_json(paths["knn_manifest"], manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def build_all_knn() -> None:
    reports = {kind: build_knn(kind) for kind in ("node", "edge")}
    atomic_json(OUTPUT_ROOT / "knn_build_report.json", reports)


def load_records(kind: str) -> list[dict[str, Any]]:
    return list(jsonl_rows(feature_paths(kind)["records"]))


def build_graph(
    kind: str,
    neighbor_count: int,
    minimum_similarity: float,
) -> nx.Graph:
    knn = np.load(feature_paths(kind)["knn"])
    indices = knn["indices"]
    similarities = knn["similarities"]
    records = load_records(kind)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(records)))

    for source in range(indices.shape[0]):
        for rank in range(min(neighbor_count, indices.shape[1])):
            target = int(indices[source, rank])
            similarity = float(similarities[source, rank])
            if similarity < minimum_similarity:
                continue
            weight = max(
                1e-6,
                (similarity - minimum_similarity)
                / max(1e-6, 1.0 - minimum_similarity),
            )
            if graph.has_edge(source, target):
                if weight > graph[source][target]["weight"]:
                    graph[source][target]["weight"] = weight
                    graph[source][target]["similarity"] = similarity
            else:
                graph.add_edge(
                    source, target, weight=weight, similarity=similarity
                )

    lexical_groups: dict[str, list[int]] = defaultdict(list)
    for row in records:
        lexical_groups[row["normalizedTag"]].append(int(row["index"]))
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
    return graph


def communities_to_labels(
    communities: list[set[int]], record_count: int
) -> np.ndarray:
    labels = np.full(record_count, -1, dtype=np.int32)
    for community_index, members in enumerate(communities):
        for member in members:
            labels[member] = community_index
    if np.any(labels < 0):
        raise RuntimeError("Community assignment left records unlabeled")
    return labels


def cohesion_summary(
    vectors: np.ndarray, communities: list[set[int]]
) -> dict[str, float]:
    weighted_sum = 0.0
    total = 0
    cluster_means = []
    for members in communities:
        member_indices = np.fromiter(members, dtype=np.int32)
        cluster = np.asarray(vectors[member_indices], dtype=np.float32)
        centroid = l2_normalize(cluster.mean(axis=0))
        similarities = cluster @ centroid
        mean_similarity = float(similarities.mean())
        cluster_means.append(mean_similarity)
        weighted_sum += float(similarities.sum())
        total += len(member_indices)
    return {
        "weightedMean": weighted_sum / total,
        "medianCommunityMean": float(np.median(cluster_means)),
        "p10CommunityMean": float(np.quantile(cluster_means, 0.10)),
    }


def run_resolution_sweep(
    kind: str,
    neighbor_count: int,
    minimum_similarity: float,
    resolutions: list[float],
    seeds: list[int],
) -> list[dict[str, Any]]:
    graph = build_graph(kind, neighbor_count, minimum_similarity)
    vectors = np.load(feature_paths(kind)["fused"], mmap_mode="r")
    records = load_records(kind)
    results = []
    for resolution in resolutions:
        label_runs = []
        communities_by_seed = []
        for seed in seeds:
            communities = nx.community.louvain_communities(
                graph,
                weight="weight",
                resolution=resolution,
                seed=seed,
            )
            communities = [set(group) for group in communities]
            communities_by_seed.append(communities)
            label_runs.append(communities_to_labels(communities, len(records)))
        aris = [
            adjusted_rand_score(label_runs[0], labels)
            for labels in label_runs[1:]
        ]
        chosen = communities_by_seed[0]
        sizes = np.array([len(group) for group in chosen], dtype=np.int32)
        meaningful = sizes >= 5
        unresolved_records = int(sizes[~meaningful].sum())
        mention_counts = np.array(
            [
                sum(int(records[index]["frequency"]) for index in group)
                for group in chosen
            ],
            dtype=np.int64,
        )
        results.append(
            {
                "kind": kind,
                "resolution": resolution,
                "neighborCount": neighbor_count,
                "minimumSimilarity": minimum_similarity,
                "graphNodes": graph.number_of_nodes(),
                "graphEdges": graph.number_of_edges(),
                "isolatedNodes": nx.number_of_isolates(graph),
                "communities": len(chosen),
                "communitiesAtLeast5": int(meaningful.sum()),
                "recordsInCommunitiesUnder5": unresolved_records,
                "fractionRecordsUnder5": unresolved_records / len(records),
                "minimumSize": int(sizes.min()),
                "medianSize": float(np.median(sizes)),
                "p90Size": float(np.quantile(sizes, 0.90)),
                "maximumSize": int(sizes.max()),
                "maximumMentionCount": int(mention_counts.max()),
                "modularity": float(
                    nx.community.modularity(
                        graph,
                        chosen,
                        weight="weight",
                        resolution=resolution,
                    )
                ),
                "stabilityAdjustedRandMean": (
                    float(np.mean(aris)) if aris else 1.0
                ),
                "cohesion": cohesion_summary(vectors, chosen),
            }
        )
    return results


def sweep(
    node_minimum_similarity: float,
    edge_minimum_similarity: float,
    neighbor_count: int,
) -> None:
    resolutions = [0.35, 0.50, 0.70, 0.90, 1.10, 1.35, 1.65, 2.0]
    seeds = [RANDOM_SEED, RANDOM_SEED + 1, RANDOM_SEED + 2]
    result = {
        "createdAt": now_iso(),
        "method": "weighted cosine kNN graph plus Louvain communities",
        "resolutions": resolutions,
        "seeds": seeds,
        "node": run_resolution_sweep(
            "node",
            neighbor_count,
            node_minimum_similarity,
            resolutions,
            seeds,
        ),
        "edge": run_resolution_sweep(
            "edge",
            neighbor_count,
            edge_minimum_similarity,
            resolutions,
            seeds,
        ),
    }
    atomic_json(OUTPUT_ROOT / "resolution_sweep.json", result)
    print(json.dumps(result, indent=2))


def cluster_member_rows(
    kind: str,
    cluster_id: str,
    members: set[int],
    records: list[dict[str, Any]],
    vectors: np.ndarray,
    unresolved: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    member_indices = np.array(sorted(members), dtype=np.int32)
    matrix = np.asarray(vectors[member_indices], dtype=np.float32)
    centroid = l2_normalize(matrix.mean(axis=0))
    similarities = matrix @ centroid
    order_by_centrality = np.argsort(-similarities)
    order_by_frequency = sorted(
        range(len(member_indices)),
        key=lambda offset: (
            -int(records[int(member_indices[offset])]["frequency"]),
            -float(similarities[offset]),
            records[int(member_indices[offset])]["tag"],
        ),
    )

    def exemplar(offset: int) -> dict[str, Any]:
        record = records[int(member_indices[offset])]
        return {
            "tag": record["tag"],
            "frequency": int(record["frequency"]),
            "similarityToCentroid": round(float(similarities[offset]), 6),
        }

    central = [exemplar(int(offset)) for offset in order_by_centrality[:20]]
    frequent = [exemplar(int(offset)) for offset in order_by_frequency[:20]]
    assignments = []
    for local_offset, record_index in enumerate(member_indices):
        record = records[int(record_index)]
        assignments.append(
            {
                "kind": kind,
                "index": int(record_index),
                "tag": record["tag"],
                "normalizedTag": record["normalizedTag"],
                "frequency": int(record["frequency"]),
                "clusterId": cluster_id,
                "unresolved": unresolved,
                "similarityToCentroid": round(
                    float(similarities[local_offset]), 6
                ),
                "baseKey": record["baseKey"],
                "sampleOccurrences": record["sampleOccurrences"],
            }
        )

    cluster = {
        "kind": kind,
        "clusterId": cluster_id,
        "unresolved": unresolved,
        "candidateMetaTag": None,
        "alternativeMetaTag": None,
        "description": None,
        "coherenceAssessment": None,
        "splitNote": None,
        "uniqueTags": len(member_indices),
        "mentionCount": sum(
            int(records[int(index)]["frequency"]) for index in member_indices
        ),
        "meanSimilarityToCentroid": round(float(similarities.mean()), 6),
        "p10SimilarityToCentroid": round(
            float(np.quantile(similarities, 0.10)), 6
        ),
        "minimumSimilarityToCentroid": round(float(similarities.min()), 6),
        "centralTags": central,
        "frequentTags": frequent,
    }
    return cluster, assignments


def select_kind(
    kind: str,
    resolution: float,
    neighbor_count: int,
    minimum_similarity: float,
) -> dict[str, Any]:
    graph = build_graph(kind, neighbor_count, minimum_similarity)
    records = load_records(kind)
    vectors = np.load(feature_paths(kind)["fused"], mmap_mode="r")
    communities = nx.community.louvain_communities(
        graph,
        weight="weight",
        resolution=resolution,
        seed=RANDOM_SEED,
    )
    communities = [set(group) for group in communities]
    substantive = [group for group in communities if len(group) >= 5]
    unresolved = [group for group in communities if len(group) < 5]
    substantive.sort(
        key=lambda group: (
            -sum(int(records[index]["frequency"]) for index in group),
            -len(group),
            min(records[index]["tag"] for index in group),
        )
    )
    unresolved.sort(
        key=lambda group: (
            -len(group),
            min(records[index]["tag"] for index in group),
        )
    )

    prefix = "NODE" if kind == "node" else "EDGE"
    cluster_rows = []
    assignment_rows = []
    for ordinal, members in enumerate(substantive, start=1):
        cluster_id = f"{prefix}_C{ordinal:03d}"
        cluster, assignments = cluster_member_rows(
            kind, cluster_id, members, records, vectors, False
        )
        cluster_rows.append(cluster)
        assignment_rows.extend(assignments)
    for ordinal, members in enumerate(unresolved, start=1):
        cluster_id = f"{prefix}_U{ordinal:03d}"
        cluster, assignments = cluster_member_rows(
            kind, cluster_id, members, records, vectors, True
        )
        cluster["candidateMetaTag"] = "UNRESOLVED"
        cluster["coherenceAssessment"] = "unresolved"
        cluster_rows.append(cluster)
        assignment_rows.extend(assignments)

    assignment_rows.sort(key=lambda row: int(row["index"]))
    if [row["index"] for row in assignment_rows] != list(range(len(records))):
        raise RuntimeError(f"Assignments are not complete and ordered for {kind}")
    cluster_path = OUTPUT_ROOT / f"{kind}_clusters.json"
    assignment_path = OUTPUT_ROOT / f"{kind}_assignments.jsonl"
    atomic_json(cluster_path, cluster_rows)
    write_jsonl(assignment_path, assignment_rows)
    result = {
        "kind": kind,
        "resolution": resolution,
        "neighborCount": neighbor_count,
        "minimumSimilarity": minimum_similarity,
        "records": len(records),
        "mentions": sum(int(row["frequency"]) for row in records),
        "substantiveClusters": len(substantive),
        "unresolvedClusters": len(unresolved),
        "unresolvedRecords": sum(len(group) for group in unresolved),
        "clusterPath": str(cluster_path),
        "assignmentPath": str(assignment_path),
    }
    return result


def write_unlabeled_preview(results: dict[str, Any]) -> None:
    lines = [
        "# Konbaung V3 Node/Edge Clustering — Unlabeled Preview",
        "",
        "This preview shows central and high-frequency tags before candidate "
        "meta-tags are assigned.",
        "",
    ]
    for kind in ("node", "edge"):
        lines.extend(
            [
                f"## {kind.title()} communities",
                "",
                f"- Substantive clusters: {results[kind]['substantiveClusters']}",
                f"- Unresolved records: {results[kind]['unresolvedRecords']}",
                "",
            ]
        )
        clusters = embedding_run.read_json(
            OUTPUT_ROOT / f"{kind}_clusters.json"
        )
        for cluster in clusters:
            if cluster["unresolved"]:
                continue
            central = ", ".join(
                item["tag"] for item in cluster["centralTags"][:10]
            )
            frequent = ", ".join(
                f"{item['tag']} ({item['frequency']})"
                for item in cluster["frequentTags"][:8]
            )
            lines.extend(
                [
                    f"### {cluster['clusterId']}",
                    "",
                    f"- Tags: {cluster['uniqueTags']:,}; mentions: "
                    f"{cluster['mentionCount']:,}; mean cosine: "
                    f"{cluster['meanSimilarityToCentroid']:.3f}",
                    f"- Central: {central}",
                    f"- Frequent: {frequent}",
                    "",
                ]
            )
    path = OUTPUT_ROOT / "UNLABELED_CLUSTER_PREVIEW.md"
    temp = path.with_suffix(".md.tmp")
    temp.write_text("\n".join(lines), encoding="utf-8")
    temp.replace(path)


def select_clusters(
    node_resolution: float,
    edge_resolution: float,
    node_minimum_similarity: float,
    edge_minimum_similarity: float,
    neighbor_count: int,
) -> None:
    result = {
        "createdAt": now_iso(),
        "method": "weighted cosine kNN graph plus Louvain communities",
        "node": select_kind(
            "node",
            node_resolution,
            neighbor_count,
            node_minimum_similarity,
        ),
        "edge": select_kind(
            "edge",
            edge_resolution,
            neighbor_count,
            edge_minimum_similarity,
        ),
    }
    atomic_json(OUTPUT_ROOT / "selected_resolution.json", result)
    write_unlabeled_preview(result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="First-pass clustering of Konbaung V3 node and edge tags."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build-features")
    subparsers.add_parser("build-knn")
    sweep_parser = subparsers.add_parser("sweep")
    sweep_parser.add_argument("--node-min-similarity", type=float, required=True)
    sweep_parser.add_argument("--edge-min-similarity", type=float, required=True)
    sweep_parser.add_argument("--neighbors", type=int, default=15)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--node-resolution", type=float, required=True)
    select_parser.add_argument("--edge-resolution", type=float, required=True)
    select_parser.add_argument(
        "--node-min-similarity", type=float, required=True
    )
    select_parser.add_argument(
        "--edge-min-similarity", type=float, required=True
    )
    select_parser.add_argument("--neighbors", type=int, default=15)
    args = parser.parse_args()

    if args.command == "build-features":
        build_features()
    elif args.command == "build-knn":
        build_all_knn()
    elif args.command == "sweep":
        sweep(
            args.node_min_similarity,
            args.edge_min_similarity,
            args.neighbors,
        )
    else:
        select_clusters(
            args.node_resolution,
            args.edge_resolution,
            args.node_min_similarity,
            args.edge_min_similarity,
            args.neighbors,
        )


if __name__ == "__main__":
    main()

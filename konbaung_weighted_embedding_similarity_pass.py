from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path(r"C:\Users\conra\Desktop\dighumproject")
EMBEDDINGS = ROOT / "konbaung_v3_eight_view_embeddings_20260724"
FIRST_PASS = ROOT / "konbaung_v3_node_edge_clustering_first_pass_20260724"
CLUSTER_SOURCE = ROOT / "konbaung_node_identity_statistical_refinement_20260725"
OUTPUT = ROOT / "konbaung_node_identity_weighted_embedding_refinement_20260725"

OCCURRENCES = EMBEDDINGS / "occurrences.jsonl"
TRIPLE_VECTORS = EMBEDDINGS / "vectors" / "triple.npy"
TRIPLE_KEYS = EMBEDDINGS / "vectors" / "triple.keys.jsonl"
NODE_RECORDS = FIRST_PASS / "node_records.jsonl"
CLUSTERS = CLUSTER_SOURCE / "corrected_initial_clusters.jsonl"
TAG_CENTROIDS = CLUSTER_SOURCE / "base_cluster_centroids.npy"
CONTEXT_CENTROIDS = CLUSTER_SOURCE / "context_cluster_centroids.npy"

TAG_WEIGHT = 0.50
TRIPLE_WEIGHT = 0.20
CONTEXT_WEIGHT = 0.30
NEIGHBORS = 100
EXPECTED_TAGS = 23_890
EXPECTED_CLUSTERS = 19_002


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise RuntimeError("Cannot normalize a zero vector")
    return matrix / norms


def build_triple_tag_vectors(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    tag_to_index = {record["tag"]: int(record["index"]) for record in records}
    if len(tag_to_index) != EXPECTED_TAGS:
        raise RuntimeError(f"Expected {EXPECTED_TAGS:,} unique node tags")

    key_to_row = {row["key"]: int(row["row"]) for row in jsonl(TRIPLE_KEYS)}
    vector_store = np.load(TRIPLE_VECTORS, mmap_mode="r")
    if len(key_to_row) != vector_store.shape[0]:
        raise RuntimeError("Triple key/vector count mismatch")

    sums = np.zeros((EXPECTED_TAGS, vector_store.shape[1]), dtype=np.float32)
    counts = np.zeros(EXPECTED_TAGS, dtype=np.int32)

    for occurrence in jsonl(OCCURRENCES):
        keys = occurrence["embeddingKeys"]["SPO"]
        if not keys:
            raise RuntimeError(f"Missing SPO vector: {occurrence['occurrenceId']}")
        if len(keys) == 1:
            triple_vector = np.asarray(vector_store[key_to_row[keys[0]]], dtype=np.float32)
        else:
            triple_vector = np.mean(
                [
                    np.asarray(vector_store[key_to_row[key]], dtype=np.float32)
                    for key in keys
                ],
                axis=0,
                dtype=np.float32,
            )
            triple_norm = float(np.linalg.norm(triple_vector))
            if triple_norm == 0.0:
                raise RuntimeError("Chunked triple produced a zero vector")
            triple_vector /= triple_norm

        for field in ("subject", "object"):
            index = tag_to_index[occurrence[field]]
            sums[index] += triple_vector
            counts[index] += 1

    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0)
        raise RuntimeError(f"{len(missing)} tags have no triple occurrences")
    return normalize_rows(sums), counts


def build_triple_cluster_centroids(
    tag_vectors: np.ndarray, clusters: list[dict]
) -> np.ndarray:
    centroids = np.empty(
        (len(clusters), tag_vectors.shape[1]), dtype=np.float32
    )
    for index, cluster in enumerate(clusters):
        members = np.asarray(cluster["memberIndices"], dtype=np.int32)
        centroids[index] = np.mean(
            tag_vectors[members], axis=0, dtype=np.float32
        )
    return normalize_rows(centroids)


def exact_neighbors(
    weighted_vectors: np.ndarray, neighbors: int, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact weighted cosine search")
    device = torch.device("cuda")
    corpus = torch.from_numpy(np.ascontiguousarray(weighted_vectors)).to(device)
    indices = np.empty((len(weighted_vectors), neighbors), dtype=np.int32)
    scores = np.empty((len(weighted_vectors), neighbors), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(weighted_vectors), batch_size):
            stop = min(start + batch_size, len(weighted_vectors))
            similarity = corpus[start:stop] @ corpus.T
            local = torch.arange(stop - start, device=device)
            similarity[local, torch.arange(start, stop, device=device)] = -2.0
            values, found = torch.topk(
                similarity, k=neighbors, largest=True, sorted=True, dim=1
            )
            indices[start:stop] = found.cpu().numpy().astype(np.int32)
            scores[start:stop] = values.cpu().numpy().astype(np.float32)
    del corpus
    torch.cuda.empty_cache()
    return indices, scores


def unique_ranked_pairs(
    indices: np.ndarray, scores: np.ndarray
) -> list[tuple[float, int, int]]:
    pairs: dict[tuple[int, int], float] = {}
    for left in range(indices.shape[0]):
        for position in range(indices.shape[1]):
            right = int(indices[left, position])
            pair = (left, right) if left < right else (right, left)
            score = float(scores[left, position])
            if score > pairs.get(pair, -2.0):
                pairs[pair] = score
    ranked = [(score, left, right) for (left, right), score in pairs.items()]
    ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
    return ranked


def clean(value: object) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = list(jsonl(NODE_RECORDS))
    clusters = list(jsonl(CLUSTERS))
    if len(records) != EXPECTED_TAGS:
        raise RuntimeError(f"Expected {EXPECTED_TAGS:,} node records")
    if len(clusters) != EXPECTED_CLUSTERS:
        raise RuntimeError(f"Expected {EXPECTED_CLUSTERS:,} clusters")

    tag_centroids = normalize_rows(np.load(TAG_CENTROIDS))
    context_centroids = normalize_rows(np.load(CONTEXT_CENTROIDS))
    triple_tag_vectors, triple_occurrence_counts = build_triple_tag_vectors(records)
    triple_centroids = build_triple_cluster_centroids(triple_tag_vectors, clusters)
    np.save(OUTPUT / "triple_tag_vectors.npy", triple_tag_vectors)
    np.save(OUTPUT / "triple_tag_occurrence_counts.npy", triple_occurrence_counts)
    np.save(OUTPUT / "triple_cluster_centroids.npy", triple_centroids)

    weighted_vectors = np.concatenate(
        [
            math.sqrt(TAG_WEIGHT) * tag_centroids,
            math.sqrt(TRIPLE_WEIGHT) * triple_centroids,
            math.sqrt(CONTEXT_WEIGHT) * context_centroids,
        ],
        axis=1,
    ).astype(np.float32)
    weighted_vectors = normalize_rows(weighted_vectors)
    np.save(OUTPUT / "weighted_cluster_vectors.npy", weighted_vectors)

    neighbor_indices, neighbor_scores = exact_neighbors(
        weighted_vectors, NEIGHBORS
    )
    np.savez_compressed(
        OUTPUT / "weighted_embedding_neighbors_top100.npz",
        indices=neighbor_indices,
        similarities=neighbor_scores,
    )
    ranked_pairs = unique_ranked_pairs(neighbor_indices, neighbor_scores)

    candidates_path = OUTPUT / "weighted_embedding_candidate_pairs.jsonl"
    markdown_path = OUTPUT / "ALL_WEIGHTED_GEMINI_EMBEDDING_CANDIDATES.md"
    top100_path = OUTPUT / "WEIGHTED_GEMINI_EMBEDDING_TOP100.md"

    with (
        candidates_path.open("w", encoding="utf-8", newline="\n") as candidates,
        markdown_path.open("w", encoding="utf-8", newline="\n") as markdown,
    ):
        markdown.write("# All weighted Gemini embedding candidates\n\n")
        markdown.write(
            "Similarity = 50% entity tag + 20% whole triple + 30% entity with "
            "Burmese and English sentence context.\n\n"
        )
        for score, left, right in ranked_pairs:
            left_identity = clean(clusters[left]["canonicalLabel"])
            right_identity = clean(clusters[right]["canonicalLabel"])
            candidates.write(
                json.dumps(
                    {
                        "weightedEmbeddingSimilarity": score,
                        "leftClusterId": clusters[left]["clusterId"],
                        "leftIdentity": left_identity,
                        "rightClusterId": clusters[right]["clusterId"],
                        "rightIdentity": right_identity,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            markdown.write(
                f"- **{score:.6%}** — {left_identity} ↔ {right_identity}\n"
            )

    with top100_path.open("w", encoding="utf-8", newline="\n") as top100:
        top100.write("# Weighted Gemini embedding top candidates\n\n")
        top100.write(
            "Similarity = 50% entity tag + 20% whole triple + 30% entity with "
            "Burmese and English sentence context.\n\n"
        )
        for score, left, right in ranked_pairs[:100]:
            top100.write(
                f"- **{score:.6%}** — {clean(clusters[left]['canonicalLabel'])} "
                f"↔ {clean(clusters[right]['canonicalLabel'])}\n"
            )

    report = {
        "formula": {
            "entityTagSimilarity": TAG_WEIGHT,
            "wholeTripleSimilarity": TRIPLE_WEIGHT,
            "entityBurmeseEnglishContextSimilarity": CONTEXT_WEIGHT,
        },
        "implementation": (
            "exact weighted sum of cosine similarities, implemented as a "
            "sqrt-weighted concatenation of separately normalized view centroids"
        ),
        "clusterMemberWeighting": "equal weight per unique tag",
        "tripleOccurrenceAggregation": "mean per tag, then equal tag mean per cluster",
        "contextOccurrenceAggregation": "mean per tag, then equal tag mean per cluster",
        "clusters": len(clusters),
        "neighborsPerCluster": NEIGHBORS,
        "uniqueCandidatePairs": len(ranked_pairs),
        "newEmbeddingApiCalls": 0,
        "sourceHashes": {
            str(OCCURRENCES): sha256(OCCURRENCES),
            str(TRIPLE_VECTORS): sha256(TRIPLE_VECTORS),
            str(NODE_RECORDS): sha256(NODE_RECORDS),
            str(CLUSTERS): sha256(CLUSTERS),
            str(TAG_CENTROIDS): sha256(TAG_CENTROIDS),
            str(CONTEXT_CENTROIDS): sha256(CONTEXT_CENTROIDS),
        },
        "outputs": {
            "allCandidatesMarkdown": str(markdown_path),
            "top100Markdown": str(top100_path),
            "candidatePairsJsonl": str(candidates_path),
        },
    }
    (OUTPUT / "weighted_embedding_pass_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

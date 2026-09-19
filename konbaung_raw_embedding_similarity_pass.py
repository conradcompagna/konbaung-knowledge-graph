from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(r"C:\Users\conra\Desktop\dighumproject")
SOURCE = ROOT / "konbaung_node_identity_statistical_refinement_20260725"
OUTPUT = ROOT / "konbaung_node_identity_raw_embedding_refinement_20260725"
CLUSTERS_PATH = SOURCE / "corrected_initial_clusters.jsonl"
CENTROIDS_PATH = SOURCE / "base_cluster_centroids.npy"
NEIGHBORS_PER_CLUSTER = 100
REVIEW_SIZE = 100
EXPECTED_CLUSTERS = 19_002


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_clusters() -> list[dict]:
    with CLUSTERS_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def exact_cosine_neighbors(
    matrix: np.ndarray, neighbors: int, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this exact all-cluster cosine search")
    device = torch.device("cuda")
    corpus = torch.from_numpy(np.ascontiguousarray(normalize(matrix))).to(device)
    indices = np.empty((len(matrix), neighbors), dtype=np.int32)
    scores = np.empty((len(matrix), neighbors), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(matrix), batch_size):
            stop = min(start + batch_size, len(matrix))
            similarities = corpus[start:stop] @ corpus.T
            local = torch.arange(stop - start, device=device)
            similarities[local, torch.arange(start, stop, device=device)] = -2.0
            values, neighbors_found = torch.topk(
                similarities, k=neighbors, dim=1, largest=True, sorted=True
            )
            indices[start:stop] = neighbors_found.cpu().numpy().astype(np.int32)
            scores[start:stop] = values.cpu().numpy().astype(np.float32)
    del corpus
    torch.cuda.empty_cache()
    return indices, scores


def unique_ranked_pairs(
    neighbor_indices: np.ndarray, neighbor_scores: np.ndarray
) -> list[tuple[float, int, int]]:
    best: dict[tuple[int, int], float] = {}
    for left in range(neighbor_indices.shape[0]):
        for position in range(neighbor_indices.shape[1]):
            right = int(neighbor_indices[left, position])
            pair = (left, right) if left < right else (right, left)
            score = float(neighbor_scores[left, position])
            if score > best.get(pair, -2.0):
                best[pair] = score
    rows = [(score, left, right) for (left, right), score in best.items()]
    rows.sort(key=lambda row: (-row[0], row[1], row[2]))
    return rows


def disjoint_review(
    ranked_pairs: list[tuple[float, int, int]], size: int
) -> list[tuple[float, int, int]]:
    used: set[int] = set()
    selected: list[tuple[float, int, int]] = []
    for score, left, right in ranked_pairs:
        if left in used or right in used:
            continue
        selected.append((score, left, right))
        used.add(left)
        used.add(right)
        if len(selected) == size:
            return selected
    raise RuntimeError(f"Could select only {len(selected)} disjoint review pairs")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    clusters = load_clusters()
    centroids = np.load(CENTROIDS_PATH)
    if len(clusters) != EXPECTED_CLUSTERS or centroids.shape[0] != EXPECTED_CLUSTERS:
        raise RuntimeError(
            f"Expected {EXPECTED_CLUSTERS:,} clusters/vectors; found "
            f"{len(clusters):,}/{centroids.shape[0]:,}"
        )

    centroids = normalize(centroids)
    neighbor_indices, neighbor_scores = exact_cosine_neighbors(
        centroids, NEIGHBORS_PER_CLUSTER
    )
    np.savez_compressed(
        OUTPUT / "raw_entity_cosine_neighbors_top100.npz",
        indices=neighbor_indices,
        similarities=neighbor_scores,
    )

    ranked_pairs = unique_ranked_pairs(neighbor_indices, neighbor_scores)
    with (OUTPUT / "raw_embedding_candidate_pairs.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for score, left, right in ranked_pairs:
            handle.write(
                json.dumps(
                    {
                        "embeddingSimilarity": score,
                        "leftClusterId": clusters[left]["clusterId"],
                        "leftIdentity": clusters[left]["canonicalLabel"],
                        "rightClusterId": clusters[right]["clusterId"],
                        "rightIdentity": clusters[right]["canonicalLabel"],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    review = disjoint_review(ranked_pairs, REVIEW_SIZE)
    review_json: list[dict] = []
    review_lines = [
        "# Raw entity-embedding similarity review",
        "",
        "The only score is cosine similarity between averaged entity-only Gemini vectors.",
        "",
    ]
    for score, left, right in review:
        row = {
            "embeddingSimilarity": score,
            "leftClusterId": clusters[left]["clusterId"],
            "leftIdentity": clusters[left]["canonicalLabel"],
            "leftMembers": clusters[left]["memberTags"],
            "rightClusterId": clusters[right]["clusterId"],
            "rightIdentity": clusters[right]["canonicalLabel"],
            "rightMembers": clusters[right]["memberTags"],
        }
        review_json.append(row)
        review_lines.append(
            f"- **{score:.6%}** — {row['leftIdentity']} ↔ {row['rightIdentity']}"
        )

    (OUTPUT / "raw_embedding_review_top100.json").write_text(
        json.dumps(review_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "RAW_EMBEDDING_REVIEW_TOP100.md").write_text(
        "\n".join(review_lines) + "\n", encoding="utf-8"
    )

    report = {
        "method": "raw cosine similarity of averaged entity-only Gemini embeddings",
        "excluded": [
            "supervised classifier",
            "context embeddings",
            "fused embeddings",
            "TF-IDF",
            "lexical normalization",
            "token rules",
            "manual cross-cluster negative labels",
        ],
        "clusters": len(clusters),
        "embeddingDimensions": int(centroids.shape[1]),
        "neighborsPerCluster": NEIGHBORS_PER_CLUSTER,
        "uniqueCandidatePairs": len(ranked_pairs),
        "reviewPairs": len(review),
        "sourceHashes": {
            str(CLUSTERS_PATH): sha256(CLUSTERS_PATH),
            str(CENTROIDS_PATH): sha256(CENTROIDS_PATH),
        },
        "newEmbeddingApiCalls": 0,
        "status": "AWAITING_MANUAL_ADJUDICATION",
    }
    (OUTPUT / "raw_embedding_pass_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

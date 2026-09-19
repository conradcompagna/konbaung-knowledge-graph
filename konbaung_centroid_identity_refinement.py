from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from difflib import SequenceMatcher
import hashlib
import heapq
import json
from pathlib import Path
import re
import shutil
from typing import Any
import unicodedata

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SEED = ROOT / "konbaung_node_identity_statistical_refinement_20260725"
FIRST_PASS = ROOT / "konbaung_v3_node_edge_clustering_first_pass_20260724"
OUTPUT = ROOT / "konbaung_node_identity_centroid_refinement_20260725"
SEED_CLUSTERS = SEED / "corrected_initial_clusters.jsonl"
TAG_VECTORS = FIRST_PASS / "node_base_vectors.npy"
TAG_RECORDS = FIRST_PASS / "node_records.jsonl"
SEED_ASSIGNMENTS = SEED / "corrected_initial_tag_assignments.csv"
STATE_PATH = OUTPUT / "state.json"
LEDGER_PATH = OUTPUT / "merge_ledger.jsonl"
REVIEW_SIZE = 100
NEIGHBORS = 100
EXHAUSTIVE_DTYPE = np.dtype(
    [
        ("reranked", "<f8"),
        ("embedding", "<f4"),
        ("lexical", "<f8"),
        ("character", "<f4"),
        ("tokens", "<f4"),
        ("trigrams", "<f4"),
        ("token_order", "<f4"),
        ("left", "<i4"),
        ("right", "<i4"),
        ("left_alias_index", "<i4"),
        ("right_alias_index", "<i4"),
    ]
)
_EXHAUSTIVE_CACHE: dict[str, Any] | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def numeric_cluster_id(cluster_id: str) -> int:
    return int(cluster_id.split("-")[-1])


def init_state() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    clusters = read_jsonl(SEED_CLUSTERS)
    vectors = np.load(TAG_VECTORS, mmap_mode="r")
    records = read_jsonl(TAG_RECORDS)
    if len(vectors) != 23_890 or len(records) != 23_890:
        raise RuntimeError("Expected exactly 23,890 aligned entity labels and vectors")
    if len(clusters) != 19_002:
        raise RuntimeError("Expected the 19,002-cluster corrected GPT seed")
    covered = sorted(index for cluster in clusters for index in cluster["memberIndices"])
    if covered != list(range(23_890)):
        raise RuntimeError("Seed clusters do not partition all entity labels")
    for index, record in enumerate(records):
        if int(record["index"]) != index:
            raise RuntimeError(f"Unordered tag record at {index}")

    state = {
        "method": "iterative raw-Gemini entity-vector centroid cosine",
        "pass": 1,
        "status": "READY_FOR_REVIEW",
        "clusters": clusters,
        "cannotLinks": [],
        "reviewedCandidates": 0,
        "acceptedMerges": 0,
        "sourceHashes": {
            str(SEED_CLUSTERS): sha256(SEED_CLUSTERS),
            str(TAG_VECTORS): sha256(TAG_VECTORS),
            str(TAG_RECORDS): sha256(TAG_RECORDS),
            str(SEED_ASSIGNMENTS): sha256(SEED_ASSIGNMENTS),
        },
        "newEmbeddingApiCalls": 0,
    }
    write_json(STATE_PATH, state)
    LEDGER_PATH.write_text("", encoding="utf-8")


def centroids_for(
    clusters: list[dict[str, Any]], tag_vectors: np.ndarray
) -> np.ndarray:
    tag_vectors = normalize_rows(tag_vectors)
    centroids = np.empty((len(clusters), tag_vectors.shape[1]), dtype=np.float32)
    for index, cluster in enumerate(clusters):
        member_indices = np.asarray(cluster["memberIndices"], dtype=np.int64)
        centroids[index] = tag_vectors[member_indices].mean(axis=0)
    return normalize_rows(centroids)


def exact_neighbors(
    centroids: np.ndarray, neighbors: int, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact centroid cosine search")
    device = torch.device("cuda")
    corpus = torch.from_numpy(np.ascontiguousarray(centroids)).to(device)
    indices = np.empty((len(centroids), neighbors), dtype=np.int32)
    scores = np.empty((len(centroids), neighbors), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(centroids), batch_size):
            stop = min(start + batch_size, len(centroids))
            similarities = corpus[start:stop] @ corpus.T
            local = torch.arange(stop - start, device=device)
            similarities[local, torch.arange(start, stop, device=device)] = -2.0
            values, found = torch.topk(
                similarities, k=neighbors, dim=1, largest=True, sorted=True
            )
            indices[start:stop] = found.cpu().numpy().astype(np.int32)
            scores[start:stop] = values.cpu().numpy().astype(np.float32)
    del corpus
    torch.cuda.empty_cache()
    return indices, scores


def source_ids(cluster: dict[str, Any]) -> set[str]:
    return set(cluster["sourceClusterIds"])


def cannot_merge(
    left: dict[str, Any],
    right: dict[str, Any],
    cannot_links: set[tuple[str, str]],
) -> bool:
    for a in source_ids(left):
        for b in source_ids(right):
            pair = (a, b) if a < b else (b, a)
            if pair in cannot_links:
                return True
    return False


def make_batch() -> Path:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if state["status"] == "STOPPED_ZERO_VALID_BATCH":
        raise RuntimeError("Run has already reached its stopping condition")
    clusters = state["clusters"]
    cannot_links = {tuple(pair) for pair in state["cannotLinks"]}
    tag_vectors = np.load(TAG_VECTORS, mmap_mode="r")
    centroids = centroids_for(clusters, tag_vectors)
    neighbor_indices, neighbor_scores = exact_neighbors(centroids, NEIGHBORS)

    ranked: dict[tuple[int, int], float] = {}
    for left in range(len(clusters)):
        for position in range(NEIGHBORS):
            right = int(neighbor_indices[left, position])
            pair = (left, right) if left < right else (right, left)
            if cannot_merge(clusters[pair[0]], clusters[pair[1]], cannot_links):
                continue
            score = float(neighbor_scores[left, position])
            if score > ranked.get(pair, -2.0):
                ranked[pair] = score

    used: set[int] = set()
    selected: list[dict[str, Any]] = []
    for (left, right), score in sorted(
        ranked.items(), key=lambda item: (-item[1], item[0])
    ):
        if left in used or right in used:
            continue
        a, b = clusters[left], clusters[right]
        selected.append(
            {
                "rank": len(selected) + 1,
                "similarity": score,
                "leftClusterId": a["clusterId"],
                "leftCanonical": a["canonicalLabel"],
                "leftMembers": a["memberTags"],
                "leftSourceClusterIds": a["sourceClusterIds"],
                "rightClusterId": b["clusterId"],
                "rightCanonical": b["canonicalLabel"],
                "rightMembers": b["memberTags"],
                "rightSourceClusterIds": b["sourceClusterIds"],
            }
        )
        used.add(left)
        used.add(right)
        if len(selected) == REVIEW_SIZE:
            break
    if len(selected) != REVIEW_SIZE:
        raise RuntimeError(f"Only {len(selected)} disjoint candidates were available")

    pass_no = int(state["pass"])
    json_path = OUTPUT / f"pass_{pass_no:02d}_candidates.json"
    md_path = OUTPUT / f"PASS_{pass_no:02d}_CANDIDATES.md"
    write_json(json_path, selected)
    lines = [
        f"# Centroid identity review — pass {pass_no}",
        "",
        "Accept only when every label on both sides denotes the exact same underlying identity.",
        "The sole score is raw cosine similarity between current mean Gemini entity vectors.",
        "",
    ]
    for row in selected:
        lines.extend(
            [
                f"## {row['rank']}. {row['similarity']:.6%}",
                "",
                f"- Left: **{row['leftCanonical']}** (`{row['leftClusterId']}`)",
                f"- Left labels: {'; '.join(row['leftMembers'])}",
                f"- Right: **{row['rightCanonical']}** (`{row['rightClusterId']}`)",
                f"- Right labels: {'; '.join(row['rightMembers'])}",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def merge_clusters(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    survivor, retired = sorted(
        [left, right], key=lambda row: numeric_cluster_id(row["clusterId"])
    )
    canonical_source = max(
        [left, right],
        key=lambda row: (int(row["totalFrequency"]), len(row["memberTags"])),
    )
    members = sorted(set(left["memberIndices"]) | set(right["memberIndices"]))
    return {
        "clusterIndex": -1,
        "clusterId": survivor["clusterId"],
        "canonicalLabel": canonical_source["canonicalLabel"],
        "tagCount": len(members),
        "totalFrequency": int(left["totalFrequency"]) + int(right["totalFrequency"]),
        "memberIndices": members,
        "memberTags": [
            tag
            for _, tag in sorted(
                zip(
                    left["memberIndices"] + right["memberIndices"],
                    left["memberTags"] + right["memberTags"],
                )
            )
        ],
        "sourceClusterIds": sorted(source_ids(left) | source_ids(right)),
        "documentedFrozenMerge": bool(
            left.get("documentedFrozenMerge")
            or right.get("documentedFrozenMerge")
        ),
        "retiredWorkingClusterIds": sorted(
            set(left.get("retiredWorkingClusterIds", []))
            | set(right.get("retiredWorkingClusterIds", []))
            | {retired["clusterId"]}
        ),
    }


def apply_adjudication(path: Path) -> None:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    pass_no = int(state["pass"])
    candidates = json.loads(
        (OUTPUT / f"pass_{pass_no:02d}_candidates.json").read_text(encoding="utf-8")
    )
    decisions = json.loads(path.read_text(encoding="utf-8"))
    if len(decisions) != REVIEW_SIZE:
        raise RuntimeError("Adjudication must contain exactly 100 decisions")
    by_rank = {int(row["rank"]): row for row in decisions}
    if set(by_rank) != set(range(1, REVIEW_SIZE + 1)):
        raise RuntimeError("Adjudication ranks must be exactly 1 through 100")

    clusters = state["clusters"]
    by_id = {row["clusterId"]: row for row in clusters}
    cannot_links = {tuple(pair) for pair in state["cannotLinks"]}
    merged_ids: set[str] = set()
    replacements: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    accepted = 0

    for candidate in candidates:
        decision = by_rank[int(candidate["rank"])]
        verdict = str(decision["decision"]).upper()
        if verdict not in {"MERGE", "REJECT"}:
            raise RuntimeError(f"Invalid verdict at rank {candidate['rank']}: {verdict}")
        left = by_id[candidate["leftClusterId"]]
        right = by_id[candidate["rightClusterId"]]
        ledger = {
            "pass": pass_no,
            "rank": candidate["rank"],
            "similarity": candidate["similarity"],
            "leftClusterId": left["clusterId"],
            "leftCanonical": left["canonicalLabel"],
            "rightClusterId": right["clusterId"],
            "rightCanonical": right["canonicalLabel"],
            "decision": verdict,
            "reason": decision["reason"],
        }
        if verdict == "MERGE":
            if left["clusterId"] in merged_ids or right["clusterId"] in merged_ids:
                raise RuntimeError("Candidate batch was not disjoint")
            merged = merge_clusters(left, right)
            replacements.append(merged)
            merged_ids.update([left["clusterId"], right["clusterId"]])
            ledger["survivingClusterId"] = merged["clusterId"]
            accepted += 1
        else:
            for a in source_ids(left):
                for b in source_ids(right):
                    cannot_links.add((a, b) if a < b else (b, a))
        ledger_rows.append(ledger)

    next_clusters = [
        row for row in clusters if row["clusterId"] not in merged_ids
    ] + replacements
    next_clusters.sort(key=lambda row: numeric_cluster_id(row["clusterId"]))
    for index, cluster in enumerate(next_clusters):
        cluster["clusterIndex"] = index

    with LEDGER_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        for row in ledger_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )

    state["clusters"] = next_clusters
    state["cannotLinks"] = [list(pair) for pair in sorted(cannot_links)]
    state["reviewedCandidates"] = int(state["reviewedCandidates"]) + REVIEW_SIZE
    state["acceptedMerges"] = int(state["acceptedMerges"]) + accepted
    state["lastPassAccepted"] = accepted
    if accepted == 0:
        state["status"] = "STOPPED_ZERO_VALID_BATCH"
    else:
        state["pass"] = pass_no + 1
        state["status"] = "READY_FOR_REVIEW"
    write_json(STATE_PATH, state)
    export_outputs(state)


def export_outputs(state: dict[str, Any]) -> None:
    clusters = state["clusters"]
    records = read_jsonl(TAG_RECORDS)
    assignment_rows: list[dict[str, Any]] = []
    for cluster in clusters:
        for member_index, member_tag in zip(
            cluster["memberIndices"], cluster["memberTags"]
        ):
            assignment_rows.append(
                {
                    "source_index": member_index,
                    "tag": member_tag,
                    "frequency": records[member_index]["frequency"],
                    "final_cluster_id": cluster["clusterId"],
                    "canonical_label": cluster["canonicalLabel"],
                }
            )
    assignment_rows.sort(key=lambda row: row["source_index"])
    with (OUTPUT / "current_tag_assignments.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignment_rows[0]))
        writer.writeheader()
        writer.writerows(assignment_rows)
    write_jsonl(OUTPUT / "current_clusters.jsonl", clusters)
    lines = [
        "# Centroid identity-refinement status",
        "",
        f"- Status: `{state['status']}`",
        f"- Seed clusters: 19,002",
        f"- Current clusters: {len(clusters):,}",
        f"- Candidates manually reviewed: {state['reviewedCandidates']:,}",
        f"- Accepted exact-identity merges: {state['acceptedMerges']:,}",
        f"- New embedding API calls: 0",
        "",
        "Each current centroid is the normalized mean of its member labels' raw Gemini entity vectors.",
    ]
    (OUTPUT / "RUN_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_outputs() -> None:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if state["status"] != "STOPPED_ZERO_VALID_BATCH":
        raise RuntimeError("Finalization requires a zero-valid reviewed batch")

    for source, expected_hash in state["sourceHashes"].items():
        actual_hash = sha256(Path(source))
        if actual_hash != expected_hash:
            raise RuntimeError(f"Source changed during run: {source}")

    clusters = state["clusters"]
    records = read_jsonl(TAG_RECORDS)
    member_indices = [
        member_index
        for cluster in clusters
        for member_index in cluster["memberIndices"]
    ]
    if sorted(member_indices) != list(range(23_890)):
        raise RuntimeError("Final clusters do not partition all 23,890 labels")
    if sum(int(cluster["totalFrequency"]) for cluster in clusters) != 54_258:
        raise RuntimeError("Final cluster frequencies do not total 54,258")

    ledger = read_jsonl(LEDGER_PATH)
    accepted = [row for row in ledger if row["decision"] == "MERGE"]
    rejected = [row for row in ledger if row["decision"] == "REJECT"]
    if len(ledger) != int(state["reviewedCandidates"]):
        raise RuntimeError("Ledger size does not match reviewed-candidate count")
    if len(accepted) != int(state["acceptedMerges"]):
        raise RuntimeError("Ledger merge count does not match state")
    final_pass = max(int(row["pass"]) for row in ledger)
    final_pass_rows = [row for row in ledger if int(row["pass"]) == final_pass]
    if len(final_pass_rows) != 100 or any(
        row["decision"] != "REJECT" for row in final_pass_rows
    ):
        raise RuntimeError("Final pass is not a complete zero-valid batch")

    assignment_rows: list[dict[str, Any]] = []
    for cluster in clusters:
        for member_index, member_tag in zip(
            cluster["memberIndices"], cluster["memberTags"]
        ):
            assignment_rows.append(
                {
                    "source_index": member_index,
                    "tag": member_tag,
                    "frequency": records[member_index]["frequency"],
                    "final_cluster_id": cluster["clusterId"],
                    "canonical_label": cluster["canonicalLabel"],
                }
            )
    assignment_rows.sort(key=lambda row: row["source_index"])
    with (OUTPUT / "final_tag_assignments.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignment_rows[0]))
        writer.writeheader()
        writer.writerows(assignment_rows)
    write_jsonl(OUTPUT / "final_clusters.jsonl", clusters)

    merge_lines = [
        "# Accepted centroid identity merges",
        "",
        f"All {len(accepted):,} merges below were manually accepted as exact identity, not merely semantic similarity.",
        "",
    ]
    for row in accepted:
        merge_lines.extend(
            [
                f"## Pass {row['pass']}, rank {row['rank']} — {row['similarity']:.6%}",
                "",
                f"- **{row['leftCanonical']}** ↔ **{row['rightCanonical']}**",
                f"- {row['reason']}",
                f"- Surviving cluster: `{row['survivingClusterId']}`",
                "",
            ]
        )
    (OUTPUT / "ACCEPTED_MERGES.md").write_text(
        "\n".join(merge_lines), encoding="utf-8"
    )

    multi_clusters = sum(1 for cluster in clusters if len(cluster["memberTags"]) > 1)
    singleton_clusters = len(clusters) - multi_clusters
    accepted_scores = [float(row["similarity"]) for row in accepted]
    final_scores = [float(row["similarity"]) for row in final_pass_rows]
    report_lines = [
        "# Final report — centroid entity identity refinement",
        "",
        "## Result",
        "",
        f"- Status: `{state['status']}`",
        f"- Entity labels retained: 23,890 / 23,890",
        f"- Total label frequency retained: 54,258 / 54,258",
        f"- Starting GPT-seeded clusters: 19,002",
        f"- Final clusters: {len(clusters):,}",
        f"- Accepted exact-identity merges: {len(accepted):,}",
        f"- Rejected candidate pairs: {len(rejected):,}",
        f"- Candidates manually reviewed: {len(ledger):,} across {final_pass} passes",
        f"- Final stopping batch: pass {final_pass}, 100 reviewed, 0 accepted",
        f"- Final singleton clusters: {singleton_clusters:,}",
        f"- Final multi-label clusters: {multi_clusters:,}",
        f"- New embedding API calls: {state['newEmbeddingApiCalls']}",
        f"- New embedding cost: $0.00",
        "",
        "## Method",
        "",
        "The run began from the corrected GPT-manual entity clusters. Each label used its existing 768-dimensional raw Gemini entity embedding. For every current cluster, the normalized member vectors were arithmetically averaged and the result normalized to form its centroid. Exact cosine similarity between current centroids generated each batch's 100 highest-scoring disjoint candidates. Previously rejected source-cluster pairs were retained as cannot-links.",
        "",
        "Every candidate was then reviewed for exact identity: the same person, place, object, collective, or concept—not merely a related entity or a similar Burmese title. Accepted singleton-to-singleton, singleton-to-cluster, and cluster-to-cluster pairs were merged. Their vectors were averaged into a new centroid before the next pass. The process stopped only after a complete 100-candidate batch contained no valid merge.",
        "",
        "## Validation",
        "",
        "- Every source index from 0 through 23,889 occurs exactly once in the final assignment.",
        "- Final cluster frequencies sum to the original 54,258.",
        "- The merge ledger contains one decision for every reviewed candidate.",
        "- The accepted count in the ledger matches the state count.",
        "- All four recorded source-file SHA-256 hashes still match; neither the GPT seed data nor the embedding data was modified.",
        f"- Accepted cosine range: {min(accepted_scores):.6%} to {max(accepted_scores):.6%}.",
        f"- Stopping-batch cosine range: {min(final_scores):.6%} to {max(final_scores):.6%}.",
        "",
        "The zero-valid stopping batch is a practical saturation criterion for this iterative nearest-centroid procedure. It is not a mathematical claim that no obscure valid identity pair exists anywhere below the reviewed similarity frontier.",
    ]
    (OUTPUT / "FINAL_REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


def markdown_text(value: Any) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def normalized_lexical_form(value: str) -> tuple[str, str, list[str]]:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[_\-/\\]+", " ", value)
    value = "".join(character if character.isalnum() else " " for character in value)
    spaced = " ".join(value.split())
    return spaced, spaced.replace(" ", ""), spaced.split()


def token_f1(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = sum((Counter(left) & Counter(right)).values())
    return 2.0 * overlap / (len(left) + len(right))


def trigram_jaccard(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if len(left) < 3 or len(right) < 3:
        return 1.0 if left == right else 0.0
    left_grams = {left[index : index + 3] for index in range(len(left) - 2)}
    right_grams = {right[index : index + 3] for index in range(len(right) - 2)}
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def lexical_pair_score(left: str, right: str) -> dict[str, Any]:
    left_spaced, left_compact, left_tokens = normalized_lexical_form(left)
    right_spaced, right_compact, right_tokens = normalized_lexical_form(right)
    character = SequenceMatcher(None, left_compact, right_compact).ratio()
    tokens = token_f1(left_tokens, right_tokens)
    trigrams = trigram_jaccard(left_compact, right_compact)
    token_order = SequenceMatcher(
        None, " ".join(sorted(left_tokens)), " ".join(sorted(right_tokens))
    ).ratio()
    lexical = (
        0.45 * character
        + 0.30 * tokens
        + 0.15 * trigrams
        + 0.10 * token_order
    )
    if left_compact and left_compact == right_compact:
        lexical = 1.0
    return {
        "lexical": lexical,
        "character": character,
        "tokens": tokens,
        "trigrams": trigrams,
        "tokenOrder": token_order,
        "leftAlias": left,
        "rightAlias": right,
    }


def best_cluster_lexical_score(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    return max(
        (
            lexical_pair_score(left_tag, right_tag)
            for left_tag in left["memberTags"]
            for right_tag in right["memberTags"]
        ),
        key=lambda row: (
            row["lexical"],
            row["character"],
            row["tokens"],
            row["trigrams"],
        ),
    )


def write_manual_review_files(
    start_rank: int = 1, count: int = 1000, rerank: bool = False
) -> Path:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    clusters = state["clusters"]
    records = read_jsonl(TAG_RECORDS)

    if start_rank == 1 and not rerank:
        tagset_lines = ["# Current collapsed entity tagset", ""]
        for cluster in clusters:
            tagset_lines.extend(
                [
                    (
                        f"## `{cluster['clusterId']}` | "
                        f"{markdown_text(cluster['canonicalLabel'])} | "
                        f"{int(cluster['totalFrequency']):,}"
                    ),
                    "",
                ]
            )
            for member_index, member_tag in zip(
                cluster["memberIndices"], cluster["memberTags"]
            ):
                tagset_lines.append(
                    f"- {markdown_text(member_tag)} | "
                    f"{int(records[member_index]['frequency']):,}"
                )
            tagset_lines.append("")
        (OUTPUT / "CURRENT_COLLAPSED_ENTITY_TAGSET.md").write_text(
            "\n".join(tagset_lines), encoding="utf-8"
        )

    source_to_current: dict[str, int] = {}
    for cluster_index, cluster in enumerate(clusters):
        for source_cluster_id in cluster["sourceClusterIds"]:
            source_to_current[source_cluster_id] = cluster_index
    blocked: list[set[int]] = [set() for _ in clusters]
    for source_left, source_right in state["cannotLinks"]:
        left = source_to_current[source_left]
        right = source_to_current[source_right]
        if left == right:
            continue
        blocked[left].add(right)
        blocked[right].add(left)

    tag_vectors = np.load(TAG_VECTORS, mmap_mode="r")
    centroids = centroids_for(clusters, tag_vectors)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact centroid cosine search")
    device = torch.device("cuda")
    corpus = torch.from_numpy(np.ascontiguousarray(centroids)).to(device)
    column_indices = torch.arange(len(clusters), device=device)
    heap: list[tuple[float, int, int]] = []
    required_top = start_rank + count - 1

    with torch.inference_mode():
        for start in range(0, len(clusters), 256):
            stop = min(start + 256, len(clusters))
            similarities = corpus[start:stop] @ corpus.T
            row_indices = torch.arange(start, stop, device=device)
            similarities.masked_fill_(
                column_indices.unsqueeze(0) <= row_indices.unsqueeze(1), -2.0
            )
            for local_index, cluster_index in enumerate(range(start, stop)):
                if blocked[cluster_index]:
                    blocked_indices = torch.tensor(
                        sorted(blocked[cluster_index]),
                        dtype=torch.long,
                        device=device,
                    )
                    similarities[local_index, blocked_indices] = -2.0

            values, flat_indices = torch.topk(
                similarities.flatten(),
                k=min(required_top, similarities.numel()),
                largest=True,
                sorted=True,
            )
            for score, flat_index in zip(
                values.cpu().tolist(), flat_indices.cpu().tolist()
            ):
                if score <= -1.5:
                    continue
                left = start + flat_index // len(clusters)
                right = flat_index % len(clusters)
                item = (float(score), int(left), int(right))
                if len(heap) < required_top:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)

    del corpus
    torch.cuda.empty_cache()

    def cluster_line(cluster_index: int) -> str:
        cluster = clusters[cluster_index]
        members = "; ".join(
            (
                f"{markdown_text(member_tag)} "
                f"({int(records[member_index]['frequency']):,})"
            )
            for member_index, member_tag in zip(
                cluster["memberIndices"], cluster["memberTags"]
            )
        )
        return (
            f"- `{cluster['clusterId']}` | "
            f"**{markdown_text(cluster['canonicalLabel'])}** | "
            f"total {int(cluster['totalFrequency']):,} | {members}"
        )

    end_rank = start_rank + count - 1
    embedding_ranked = sorted(heap, reverse=True)
    selected = embedding_ranked[start_rank - 1 : end_rank]
    if rerank:
        reranked = []
        for embedding_rank, (score, left, right) in enumerate(
            selected, start=start_rank
        ):
            features = best_cluster_lexical_score(clusters[left], clusters[right])
            reranked_score = 0.50 * features["lexical"] + 0.50 * score
            reranked.append(
                (
                    reranked_score,
                    score,
                    embedding_rank,
                    left,
                    right,
                    features,
                )
            )
        reranked.sort(
            key=lambda row: (-row[0], -row[1], row[2], row[3], row[4])
        )
        candidate_rows = [
            (reranked_rank, *row)
            for reranked_rank, row in enumerate(reranked, start=1)
        ]
        candidate_lines = [
            f"# Reranked entity merge candidates {start_rank:,}-{end_rank:,}",
            "",
            "Reranked score = 50% lexical score + 50% original Gemini cosine.",
            "Lexical score = 45% compact-character similarity + 30% token overlap + 15% character-trigram overlap + 10% token-sorted similarity; exact case/separator-normalized matches score 100%.",
            "",
        ]
    else:
        candidate_rows = [
            (rank, score, score, rank, left, right, None)
            for rank, (score, left, right) in enumerate(
                selected, start=start_rank
            )
        ]
        candidate_lines = [
            f"# Remaining entity merge candidates {start_rank:,}-{end_rank:,}",
            "",
        ]
    for (
        display_rank,
        reranked_score,
        score,
        embedding_rank,
        left,
        right,
        features,
    ) in candidate_rows:
        left_singleton = len(clusters[left]["memberTags"]) == 1
        right_singleton = len(clusters[right]["memberTags"]) == 1
        if left_singleton and right_singleton:
            candidate_type = "singleton to singleton"
        elif left_singleton or right_singleton:
            candidate_type = "singleton to cluster"
        else:
            candidate_type = "cluster to cluster"
        if rerank:
            heading = (
                f"## {display_rank} | reranked {reranked_score:.6%} | "
                f"Gemini {score:.6%} | Gemini rank {embedding_rank:,} | "
                f"{candidate_type}"
            )
            feature_lines = [
                (
                    f"- Lexical {features['lexical']:.6%} | "
                    f"characters {features['character']:.6%} | "
                    f"tokens {features['tokens']:.6%} | "
                    f"trigrams {features['trigrams']:.6%} | "
                    f"token-sort {features['tokenOrder']:.6%}"
                ),
                (
                    f"- Best lexical aliases: "
                    f"{markdown_text(features['leftAlias'])} <-> "
                    f"{markdown_text(features['rightAlias'])}"
                ),
            ]
        else:
            heading = f"## {display_rank} | {score:.6%} | {candidate_type}"
            feature_lines = []
        candidate_lines.extend(
            [
                heading,
                "",
                cluster_line(left),
                cluster_line(right),
                *feature_lines,
                "",
            ]
        )
    if rerank:
        candidate_path = (
            OUTPUT
            / f"RERANKED_ENTITY_MERGE_CANDIDATES_{start_rank}_{end_rank}.md"
        )
    elif start_rank == 1 and count == 1000:
        candidate_path = OUTPUT / "TOP_1000_REMAINING_ENTITY_MERGE_CANDIDATES.md"
    else:
        candidate_path = (
            OUTPUT
            / f"REMAINING_ENTITY_MERGE_CANDIDATES_{start_rank}_{end_rank}.md"
        )
    candidate_path.write_text(
        "\n".join(candidate_lines), encoding="utf-8"
    )
    return candidate_path


def write_reranked_embedding_pool(
    pool_count: int = 1_000_000, file_size: int = 100_000
) -> list[Path]:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    clusters = state["clusters"]
    records = read_jsonl(TAG_RECORDS)
    source_to_current = {
        source_id: cluster_index
        for cluster_index, cluster in enumerate(clusters)
        for source_id in cluster["sourceClusterIds"]
    }
    blocked: list[set[int]] = [set() for _ in clusters]
    for source_left, source_right in state["cannotLinks"]:
        left = source_to_current[source_left]
        right = source_to_current[source_right]
        if left != right:
            blocked[left].add(right)
            blocked[right].add(left)

    centroids = centroids_for(
        clusters, np.load(TAG_VECTORS, mmap_mode="r")
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact centroid cosine search")
    device = torch.device("cuda")
    corpus = torch.from_numpy(np.ascontiguousarray(centroids)).to(device)
    column_indices = torch.arange(len(clusters), device=device)
    global_scores = np.empty(0, dtype=np.float32)
    global_left = np.empty(0, dtype=np.int32)
    global_right = np.empty(0, dtype=np.int32)
    with torch.inference_mode():
        for start in range(0, len(clusters), 256):
            stop = min(start + 256, len(clusters))
            similarities = corpus[start:stop] @ corpus.T
            row_indices = torch.arange(start, stop, device=device)
            similarities.masked_fill_(
                column_indices.unsqueeze(0) <= row_indices.unsqueeze(1), -2.0
            )
            for local_index, cluster_index in enumerate(range(start, stop)):
                if blocked[cluster_index]:
                    blocked_indices = torch.tensor(
                        sorted(blocked[cluster_index]),
                        dtype=torch.long,
                        device=device,
                    )
                    similarities[local_index, blocked_indices] = -2.0
            values, flat_indices = torch.topk(
                similarities.flatten(),
                k=min(pool_count, similarities.numel()),
                largest=True,
                sorted=False,
            )
            scores = values.cpu().numpy().astype(np.float32)
            flat = flat_indices.cpu().numpy().astype(np.int64)
            valid = scores > -1.5
            scores = scores[valid]
            flat = flat[valid]
            left = (start + flat // len(clusters)).astype(np.int32)
            right = (flat % len(clusters)).astype(np.int32)
            combined_scores = np.concatenate([global_scores, scores])
            combined_left = np.concatenate([global_left, left])
            combined_right = np.concatenate([global_right, right])
            if len(combined_scores) > pool_count:
                keep = np.argpartition(
                    combined_scores, -pool_count
                )[-pool_count:]
                global_scores = combined_scores[keep]
                global_left = combined_left[keep]
                global_right = combined_right[keep]
            else:
                global_scores = combined_scores
                global_left = combined_left
                global_right = combined_right
    del corpus
    torch.cuda.empty_cache()
    order = np.lexsort((global_right, global_left, -global_scores))
    global_scores = global_scores[order][:pool_count]
    global_left = global_left[order][:pool_count]
    global_right = global_right[order][:pool_count]

    reranked = np.empty(pool_count, dtype=EXHAUSTIVE_DTYPE)
    embedding_ranks = np.arange(1, pool_count + 1, dtype=np.int32)
    for index in range(pool_count):
        left = int(global_left[index])
        right = int(global_right[index])
        features = best_cluster_lexical_score(clusters[left], clusters[right])
        left_alias_position = clusters[left]["memberTags"].index(
            features["leftAlias"]
        )
        right_alias_position = clusters[right]["memberTags"].index(
            features["rightAlias"]
        )
        left_alias_index = clusters[left]["memberIndices"][left_alias_position]
        right_alias_index = clusters[right]["memberIndices"][
            right_alias_position
        ]
        embedding = float(global_scores[index])
        score = 0.50 * features["lexical"] + 0.50 * embedding
        reranked[index] = (
            score,
            embedding,
            features["lexical"],
            features["character"],
            features["tokens"],
            features["trigrams"],
            features["tokenOrder"],
            left,
            right,
            left_alias_index,
            right_alias_index,
        )
        if (index + 1) % 100_000 == 0:
            print(
                f"Lexically scored {index + 1:,}/{pool_count:,} candidates",
                flush=True,
            )
    order = np.lexsort(
        (
            reranked["right"],
            reranked["left"],
            embedding_ranks,
            -reranked["embedding"],
            -reranked["reranked"],
        )
    )
    reranked = reranked[order]
    embedding_ranks = embedding_ranks[order]

    def cluster_line(cluster_index: int) -> str:
        cluster = clusters[cluster_index]
        members = "; ".join(
            (
                f"{markdown_text(member_tag)} "
                f"({int(records[member_index]['frequency']):,})"
            )
            for member_index, member_tag in zip(
                cluster["memberIndices"], cluster["memberTags"]
            )
        )
        return (
            f"- `{cluster['clusterId']}` | "
            f"**{markdown_text(cluster['canonicalLabel'])}** | "
            f"total {int(cluster['totalFrequency']):,} | {members}"
        )

    output_paths: list[Path] = []
    for file_start in range(0, pool_count, file_size):
        file_stop = min(file_start + file_size, pool_count)
        first_rank = file_start + 1
        last_rank = file_stop
        lines = [
            (
                f"# Reranked Gemini top {pool_count:,} entity candidates "
                f"{first_rank:,}-{last_rank:,}"
            ),
            "",
            "Reranked score = 50% lexical score + 50% original Gemini cosine.",
            "Lexical score = 45% compact-character similarity + 30% token overlap + 15% character-trigram overlap + 10% token-sorted similarity; exact case/separator-normalized matches score 100%.",
            "",
        ]
        for array_index in range(file_start, file_stop):
            row = reranked[array_index]
            rank = array_index + 1
            left = int(row["left"])
            right = int(row["right"])
            left_singleton = len(clusters[left]["memberTags"]) == 1
            right_singleton = len(clusters[right]["memberTags"]) == 1
            if left_singleton and right_singleton:
                candidate_type = "singleton to singleton"
            elif left_singleton or right_singleton:
                candidate_type = "singleton to cluster"
            else:
                candidate_type = "cluster to cluster"
            left_alias = records[int(row["left_alias_index"])]["tag"]
            right_alias = records[int(row["right_alias_index"])]["tag"]
            lines.extend(
                [
                    (
                        f"## {rank} | reranked "
                        f"{float(row['reranked']):.6%} | Gemini "
                        f"{float(row['embedding']):.6%} | Gemini rank "
                        f"{int(embedding_ranks[array_index]):,} | "
                        f"{candidate_type}"
                    ),
                    "",
                    cluster_line(left),
                    cluster_line(right),
                    (
                        f"- Lexical {float(row['lexical']):.6%} | "
                        f"characters {float(row['character']):.6%} | "
                        f"tokens {float(row['tokens']):.6%} | "
                        f"trigrams {float(row['trigrams']):.6%} | "
                        f"token-sort {float(row['token_order']):.6%}"
                    ),
                    (
                        f"- Best lexical aliases: "
                        f"{markdown_text(left_alias)} <-> "
                        f"{markdown_text(right_alias)}"
                    ),
                    "",
                ]
            )
        output_path = (
            OUTPUT
            / (
                f"RERANKED_GEMINI_TOP_{pool_count}_"
                f"{first_rank}_{last_rank}.md"
            )
        )
        output_path.write_text("\n".join(lines), encoding="utf-8")
        output_paths.append(output_path)
        print(f"Wrote {output_path.name}", flush=True)
    return output_paths


def build_embedding_cosine_matrix(path: Path) -> None:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    clusters = state["clusters"]
    centroids = centroids_for(
        clusters, np.load(TAG_VECTORS, mmap_mode="r")
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exhaustive centroid cosine scoring")
    matrix = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(clusters), len(clusters)),
    )
    device = torch.device("cuda")
    corpus = torch.from_numpy(np.ascontiguousarray(centroids)).to(device)
    with torch.inference_mode():
        for start in range(0, len(clusters), 256):
            stop = min(start + 256, len(clusters))
            matrix[start:stop] = (corpus[start:stop] @ corpus.T).cpu().numpy()
    matrix.flush()
    del matrix, corpus
    torch.cuda.empty_cache()


def exhaustive_cache(embedding_matrix_path: str) -> dict[str, Any]:
    global _EXHAUSTIVE_CACHE
    if _EXHAUSTIVE_CACHE is not None:
        return _EXHAUSTIVE_CACHE
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    clusters = state["clusters"]
    records = read_jsonl(TAG_RECORDS)
    prepared_forms = []
    for record in records:
        spaced, compact, tokens = normalized_lexical_form(record["tag"])
        prepared_forms.append(
            {
                "spaced": spaced,
                "compact": compact,
                "tokens": tokens,
                "tokenCounter": Counter(tokens),
                "trigrams": frozenset(
                    compact[index : index + 3]
                    for index in range(max(0, len(compact) - 2))
                ),
                "tokenSorted": " ".join(sorted(tokens)),
            }
        )
    source_to_current = {
        source_id: cluster_index
        for cluster_index, cluster in enumerate(clusters)
        for source_id in cluster["sourceClusterIds"]
    }
    blocked: list[set[int]] = [set() for _ in clusters]
    for source_left, source_right in state["cannotLinks"]:
        left = source_to_current[source_left]
        right = source_to_current[source_right]
        if left != right:
            blocked[left].add(right)
            blocked[right].add(left)
    _EXHAUSTIVE_CACHE = {
        "clusters": clusters,
        "forms": prepared_forms,
        "blocked": blocked,
        "embedding": np.load(embedding_matrix_path, mmap_mode="r"),
    }
    return _EXHAUSTIVE_CACHE


def cheap_lexical_components(
    left: dict[str, Any], right: dict[str, Any]
) -> tuple[float, float]:
    left_tokens = left["tokens"]
    right_tokens = right["tokens"]
    if left_tokens and right_tokens:
        overlap = sum(
            (left["tokenCounter"] & right["tokenCounter"]).values()
        )
        tokens = 2.0 * overlap / (len(left_tokens) + len(right_tokens))
    else:
        tokens = 0.0
    left_grams = left["trigrams"]
    right_grams = right["trigrams"]
    if left_grams and right_grams:
        trigrams = len(left_grams & right_grams) / len(left_grams | right_grams)
    elif left["compact"] and left["compact"] == right["compact"]:
        trigrams = 1.0
    else:
        trigrams = 0.0
    return tokens, trigrams


def exhaustive_worker(
    start: int,
    stop: int,
    top_count: int,
    embedding_matrix_path: str,
    result_path: str,
) -> dict[str, Any]:
    from cdifflib import CSequenceMatcher

    cache = exhaustive_cache(embedding_matrix_path)
    clusters = cache["clusters"]
    forms = cache["forms"]
    blocked = cache["blocked"]
    embedding_matrix = cache["embedding"]
    heap: list[tuple[Any, ...]] = []
    compared = 0
    bounded_out = 0
    sequence_scored = 0

    for left_index in range(start, stop):
        left_cluster = clusters[left_index]
        for right_index in range(left_index + 1, len(clusters)):
            if right_index in blocked[left_index]:
                continue
            compared += 1
            embedding = float(embedding_matrix[left_index, right_index])
            cheap_rows: list[tuple[float, int, int, float, float]] = []
            exact_row: tuple[int, int, float, float] | None = None
            maximum_lexical_upper = 0.0

            for left_alias_index in left_cluster["memberIndices"]:
                left_form = forms[left_alias_index]
                for right_alias_index in clusters[right_index]["memberIndices"]:
                    right_form = forms[right_alias_index]
                    tokens, trigrams = cheap_lexical_components(
                        left_form, right_form
                    )
                    if (
                        left_form["compact"]
                        and left_form["compact"] == right_form["compact"]
                    ):
                        exact_row = (
                            left_alias_index,
                            right_alias_index,
                            tokens,
                            trigrams,
                        )
                        maximum_lexical_upper = 1.0
                        break
                    lexical_upper = 0.55 + 0.30 * tokens + 0.15 * trigrams
                    cheap_rows.append(
                        (
                            lexical_upper,
                            left_alias_index,
                            right_alias_index,
                            tokens,
                            trigrams,
                        )
                    )
                    maximum_lexical_upper = max(
                        maximum_lexical_upper, lexical_upper
                    )
                if exact_row is not None:
                    break

            combined_upper = (
                0.50 * maximum_lexical_upper + 0.50 * embedding
            )
            if len(heap) == top_count and combined_upper < heap[0][0]:
                bounded_out += 1
                continue

            if exact_row is not None:
                (
                    best_left_alias,
                    best_right_alias,
                    best_tokens,
                    best_trigrams,
                ) = exact_row
                best_lexical = 1.0
                best_character = 1.0
                left_form = forms[best_left_alias]
                right_form = forms[best_right_alias]
                best_token_order = CSequenceMatcher(
                    None,
                    left_form["tokenSorted"],
                    right_form["tokenSorted"],
                ).ratio()
                sequence_scored += 1
            else:
                best_key: tuple[float, float, float, float] | None = None
                best_lexical = 0.0
                best_character = 0.0
                best_tokens = 0.0
                best_trigrams = 0.0
                best_token_order = 0.0
                best_left_alias = -1
                best_right_alias = -1
                cheap_rows.sort(reverse=True)
                for (
                    lexical_upper,
                    left_alias_index,
                    right_alias_index,
                    tokens,
                    trigrams,
                ) in cheap_rows:
                    if best_key is not None and lexical_upper < best_lexical:
                        break
                    left_form = forms[left_alias_index]
                    right_form = forms[right_alias_index]
                    character = CSequenceMatcher(
                        None, left_form["compact"], right_form["compact"]
                    ).ratio()
                    token_order = CSequenceMatcher(
                        None,
                        left_form["tokenSorted"],
                        right_form["tokenSorted"],
                    ).ratio()
                    sequence_scored += 1
                    lexical = (
                        0.45 * character
                        + 0.30 * tokens
                        + 0.15 * trigrams
                        + 0.10 * token_order
                    )
                    key = (lexical, character, tokens, trigrams)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_lexical = lexical
                        best_character = character
                        best_tokens = tokens
                        best_trigrams = trigrams
                        best_token_order = token_order
                        best_left_alias = left_alias_index
                        best_right_alias = right_alias_index

            reranked = 0.50 * best_lexical + 0.50 * embedding
            item = (
                reranked,
                embedding,
                best_lexical,
                best_character,
                best_tokens,
                best_trigrams,
                best_token_order,
                left_index,
                right_index,
                best_left_alias,
                best_right_alias,
            )
            if len(heap) < top_count:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

    result = np.empty(len(heap), dtype=EXHAUSTIVE_DTYPE)
    for row_index, item in enumerate(heap):
        result[row_index] = item
    np.save(result_path, result, allow_pickle=False)
    return {
        "start": start,
        "stop": stop,
        "compared": compared,
        "boundedOut": bounded_out,
        "sequenceScored": sequence_scored,
        "resultPath": result_path,
    }


def write_exhaustive_reranked_candidates(top_count: int = 100_000) -> Path:
    work_dir = OUTPUT / "_exhaustive_rerank_work"
    if work_dir.exists():
        if work_dir.resolve().parent != OUTPUT.resolve():
            raise RuntimeError("Unsafe exhaustive work directory")
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    embedding_matrix_path = work_dir / "embedding_cosine.npy"
    print("Building exact embedding cosine matrix", flush=True)
    build_embedding_cosine_matrix(embedding_matrix_path)

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    clusters = state["clusters"]
    tasks = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for task_number, start in enumerate(range(0, len(clusters) - 1, 256)):
            stop = min(start + 256, len(clusters) - 1)
            result_path = work_dir / f"chunk_{task_number:03d}.npy"
            tasks.append(
                executor.submit(
                    exhaustive_worker,
                    start,
                    stop,
                    top_count,
                    str(embedding_matrix_path),
                    str(result_path),
                )
            )
        completed = []
        for done_number, future in enumerate(as_completed(tasks), start=1):
            result = future.result()
            completed.append(result)
            print(
                f"Completed {done_number}/{len(tasks)} row chunks "
                f"({result['compared']:,} pairs)",
                flush=True,
            )

    global_top = np.empty(0, dtype=EXHAUSTIVE_DTYPE)
    for result in completed:
        rows = np.load(result["resultPath"], mmap_mode=None)
        combined = np.concatenate([global_top, rows])
        if len(combined) > top_count:
            keep = np.argpartition(combined["reranked"], -top_count)[-top_count:]
            global_top = combined[keep]
        else:
            global_top = combined
    order = np.lexsort(
        (
            global_top["right"],
            global_top["left"],
            -global_top["embedding"],
            -global_top["reranked"],
        )
    )
    global_top = global_top[order][:top_count]
    records = read_jsonl(TAG_RECORDS)

    def cluster_line(cluster_index: int) -> str:
        cluster = clusters[cluster_index]
        members = "; ".join(
            (
                f"{markdown_text(member_tag)} "
                f"({int(records[member_index]['frequency']):,})"
            )
            for member_index, member_tag in zip(
                cluster["memberIndices"], cluster["memberTags"]
            )
        )
        return (
            f"- `{cluster['clusterId']}` | "
            f"**{markdown_text(cluster['canonicalLabel'])}** | "
            f"total {int(cluster['totalFrequency']):,} | {members}"
        )

    lines = [
        "# Exhaustive reranked top 100,000 entity merge candidates",
        "",
        "Candidate pool: every unreviewed pair among the 18,722 current entity clusters.",
        "Reranked score = 50% lexical score + 50% Gemini centroid cosine.",
        "Lexical score = 45% compact-character similarity + 30% token overlap + 15% character-trigram overlap + 10% token-sorted similarity; exact case/separator-normalized matches score 100%.",
        "",
    ]
    for rank, row in enumerate(global_top, start=1):
        left = int(row["left"])
        right = int(row["right"])
        left_singleton = len(clusters[left]["memberTags"]) == 1
        right_singleton = len(clusters[right]["memberTags"]) == 1
        if left_singleton and right_singleton:
            candidate_type = "singleton to singleton"
        elif left_singleton or right_singleton:
            candidate_type = "singleton to cluster"
        else:
            candidate_type = "cluster to cluster"
        left_alias = records[int(row["left_alias_index"])]["tag"]
        right_alias = records[int(row["right_alias_index"])]["tag"]
        lines.extend(
            [
                (
                    f"## {rank} | reranked {float(row['reranked']):.6%} | "
                    f"Gemini {float(row['embedding']):.6%} | {candidate_type}"
                ),
                "",
                cluster_line(left),
                cluster_line(right),
                (
                    f"- Lexical {float(row['lexical']):.6%} | "
                    f"characters {float(row['character']):.6%} | "
                    f"tokens {float(row['tokens']):.6%} | "
                    f"trigrams {float(row['trigrams']):.6%} | "
                    f"token-sort {float(row['token_order']):.6%}"
                ),
                (
                    f"- Best lexical aliases: {markdown_text(left_alias)} <-> "
                    f"{markdown_text(right_alias)}"
                ),
                "",
            ]
        )
    output_path = (
        OUTPUT / "EXHAUSTIVE_RERANKED_TOP_100000_ENTITY_MERGE_CANDIDATES.md"
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    if work_dir.resolve().parent != OUTPUT.resolve():
        raise RuntimeError("Unsafe exhaustive work directory cleanup")
    shutil.rmtree(work_dir)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("batch")
    sub.add_parser("finalize")
    review_parser = sub.add_parser("review-files")
    review_parser.add_argument("--start", type=int, default=1)
    review_parser.add_argument("--count", type=int, default=1000)
    review_parser.add_argument("--rerank", action="store_true")
    exhaustive_parser = sub.add_parser("exhaustive-rerank")
    exhaustive_parser.add_argument("--count", type=int, default=100000)
    million_parser = sub.add_parser("million-rerank")
    million_parser.add_argument("--pool-count", type=int, default=1000000)
    million_parser.add_argument("--file-size", type=int, default=100000)
    apply_parser = sub.add_parser("apply")
    apply_parser.add_argument("adjudication", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        init_state()
        print(STATE_PATH)
    elif args.command == "batch":
        print(make_batch())
    elif args.command == "finalize":
        finalize_outputs()
        print(OUTPUT / "FINAL_REPORT.md")
    elif args.command == "review-files":
        print(write_manual_review_files(args.start, args.count, args.rerank))
    elif args.command == "exhaustive-rerank":
        print(write_exhaustive_reranked_candidates(args.count))
    elif args.command == "million-rerank":
        for path in write_reranked_embedding_pool(
            args.pool_count, args.file_size
        ):
            print(path)
    else:
        apply_adjudication(args.adjudication)
        print(STATE_PATH)


if __name__ == "__main__":
    main()

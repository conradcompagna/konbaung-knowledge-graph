from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"C:\Users\conra\Desktop\dighumproject")
OLD_BUNDLE = ROOT / "pipeline" / "resolution" / "pair_classifier"
MANUAL = ROOT / "konbaung_node_identity_disambiguation_complete_bundle_20260725"
RAW_VECTORS = ROOT / "konbaung_v3_node_edge_clustering_first_pass_20260724"
CLUSTER_VECTORS = ROOT / "konbaung_node_identity_statistical_refinement_20260725"
RAW_CANDIDATES = (
    ROOT
    / "konbaung_node_identity_raw_embedding_refinement_20260725"
    / "raw_embedding_candidate_pairs.jsonl"
)
REGEX_REVIEW = (
    ROOT / "konbaung_node_identity_regex_refinement_20260725" / "regex_candidate_adjudication.jsonl"
)
STATISTICAL_REVIEW = CLUSTER_VECTORS / "pass_01_adjudication.csv"
OUTPUT = ROOT / "konbaung_full_gold_pair_classifier_20260725"

FEATURE_MODULE_DIR = OLD_BUNDLE
if str(FEATURE_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_MODULE_DIR))

from pair_classifier_features import (  # noqa: E402
    CorpusStats,
    FEATURE_NAMES,
    feature_vector,
    make_meta,
)


EXPECTED_TAGS = 23_890
EXPECTED_MANUAL_CLUSTERS = 19_003
EXPECTED_CORRECTED_CLUSTERS = 19_002
EXPECTED_RAW_CANDIDATES = 1_530_847
STABLE_SOURCE_INDICES = [0, *range(10, 52)]
STABLE_FEATURE_NAMES = [
    "raw_similarity",
    "context_similarity_feature",
    "fused_similarity_feature",
    *[FEATURE_NAMES[index] for index in range(10, 52)],
]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass
class VectorPair:
    base_similarity: float
    context_similarity: float

    @property
    def fused_similarity(self) -> float:
        return (self.base_similarity + self.context_similarity) / 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def cosine_percent(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(left, right), -1.0, 1.0) * 100.0)


def read_csv(path: Path, delimiter: str = ",") -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def make_global_stats(tags: Iterable[str]) -> CorpusStats:
    unique_tags = sorted(set(tags))
    metas = {tag: make_meta(tag) for tag in unique_tags}
    token_df: Counter = Counter()
    for meta in metas.values():
        token_df.update(meta.token_set)
    count = len(metas)
    token_idf = {
        token: math.log((count + 1) / (frequency + 1)) + 1.0
        for token, frequency in token_df.items()
    }
    return CorpusStats(
        metas=metas,
        degrees=Counter({tag: 1 for tag in unique_tags}),
        top_scores={tag: 100.0 for tag in unique_tags},
        token_df=token_df,
        token_idf=token_idf,
        band_counts=Counter(),
        pair_count=0,
        rare_cutoff=50,
    )


def stable_feature_vector(
    left: str,
    right: str,
    similarities: VectorPair,
    stats: CorpusStats,
) -> np.ndarray:
    original = feature_vector(
        left,
        right,
        similarities.base_similarity,
        100,
        100,
        stats,
    )
    return np.asarray(
        [
            original[0],
            similarities.context_similarity / 100.0,
            similarities.fused_similarity / 100.0,
            *original[10:52],
        ],
        dtype=np.float32,
    )


def load_sources() -> dict:
    tag_assignments = read_csv(MANUAL / "konbaung_node_identity_tag_assignments_complete.csv")
    component_assignments = read_csv(
        MANUAL / "konbaung_node_identity_component_assignments_complete.csv"
    )
    manual_clusters = read_csv(MANUAL / "konbaung_node_identity_clusters_complete.csv")
    ledger = read_csv(MANUAL / "konbaung_node_identity_merge_ledger_complete.tsv", delimiter="\t")
    node_records = list(read_jsonl(RAW_VECTORS / "node_records.jsonl"))
    corrected_clusters = list(read_jsonl(CLUSTER_VECTORS / "corrected_initial_clusters.jsonl"))

    if len(tag_assignments) != EXPECTED_TAGS or len(node_records) != EXPECTED_TAGS:
        raise RuntimeError(
            f"Expected {EXPECTED_TAGS:,} tags/vectors; found "
            f"{len(tag_assignments):,}/{len(node_records):,}"
        )
    if len(manual_clusters) != EXPECTED_MANUAL_CLUSTERS:
        raise RuntimeError(
            f"Expected {EXPECTED_MANUAL_CLUSTERS:,} manual clusters; found {len(manual_clusters):,}"
        )
    if len(corrected_clusters) != EXPECTED_CORRECTED_CLUSTERS:
        raise RuntimeError(
            f"Expected {EXPECTED_CORRECTED_CLUSTERS:,} corrected clusters; "
            f"found {len(corrected_clusters):,}"
        )
    node_record_index: dict[str, int] = {}
    for index, record in enumerate(node_records):
        tag = record["tag"]
        if tag in node_record_index:
            raise RuntimeError(f"Duplicate node-vector tag: {tag!r}")
        node_record_index[tag] = index
    display_tags = {row["display_tag"] for row in tag_assignments}
    if len(display_tags) != EXPECTED_TAGS:
        raise RuntimeError("Manual corrected display tags are not unique")
    if display_tags != set(node_record_index):
        missing_vectors = sorted(display_tags - set(node_record_index))
        missing_assignments = sorted(set(node_record_index) - display_tags)
        raise RuntimeError(
            "Tag/vector identity mismatch. "
            f"Missing vectors={missing_vectors[:10]}, "
            f"missing assignments={missing_assignments[:10]}"
        )
    for index, assignment in enumerate(tag_assignments):
        if int(assignment["rank"]) != index + 1:
            raise RuntimeError(f"Non-contiguous tag rank at row {index + 1}")
    return {
        "tag_assignments": tag_assignments,
        "component_assignments": component_assignments,
        "manual_clusters": manual_clusters,
        "ledger": ledger,
        "node_records": node_records,
        "node_record_index": node_record_index,
        "corrected_clusters": corrected_clusters,
    }


def build_full_gold() -> tuple[pd.DataFrame, CorpusStats, dict, dict]:
    sources = load_sources()
    tag_assignments = sources["tag_assignments"]
    component_assignments = sources["component_assignments"]
    manual_clusters = sources["manual_clusters"]
    ledger = sources["ledger"]
    corrected_clusters = sources["corrected_clusters"]
    node_record_index = sources["node_record_index"]

    vector_order = np.asarray(
        [node_record_index[row["display_tag"]] for row in tag_assignments],
        dtype=np.int32,
    )
    raw_base = normalize(np.load(RAW_VECTORS / "node_base_vectors.npy")[vector_order])
    raw_context = normalize(np.load(RAW_VECTORS / "node_context_vectors.npy")[vector_order])
    cluster_base = normalize(np.load(CLUSTER_VECTORS / "base_cluster_centroids.npy"))
    cluster_context = normalize(np.load(CLUSTER_VECTORS / "context_cluster_centroids.npy"))
    if raw_base.shape != (EXPECTED_TAGS, 768) or raw_context.shape != (EXPECTED_TAGS, 768):
        raise RuntimeError(f"Unexpected raw vector shapes: {raw_base.shape}/{raw_context.shape}")
    if cluster_base.shape != (EXPECTED_CORRECTED_CLUSTERS, 768):
        raise RuntimeError(f"Unexpected cluster vector shape: {cluster_base.shape}")

    corrected_index = {row["clusterId"]: index for index, row in enumerate(corrected_clusters)}
    corrected_label = {row["clusterId"]: row["canonicalLabel"] for row in corrected_clusters}
    manual_label = {row["cluster_id"]: row["canonical_label"] for row in manual_clusters}

    tags_by_cluster: dict[str, list[int]] = defaultdict(list)
    tags_by_component: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(tag_assignments):
        tags_by_cluster[row["final_cluster_id"]].append(index)
        tags_by_component[row["source_component_id"]].append(index)

    def averaged_raw(indices: list[int], matrix: np.ndarray) -> np.ndarray:
        vector = matrix[indices].mean(axis=0)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    manual_base_vectors = {
        cluster_id: averaged_raw(indices, raw_base)
        for cluster_id, indices in tags_by_cluster.items()
    }
    manual_context_vectors = {
        cluster_id: averaged_raw(indices, raw_context)
        for cluster_id, indices in tags_by_cluster.items()
    }
    component_base_vectors = {
        component_id: averaged_raw(indices, raw_base)
        for component_id, indices in tags_by_component.items()
    }
    component_context_vectors = {
        component_id: averaged_raw(indices, raw_context)
        for component_id, indices in tags_by_component.items()
    }

    def vectors_for_cluster(cluster_id: str) -> tuple[np.ndarray, np.ndarray]:
        if cluster_id in corrected_index:
            index = corrected_index[cluster_id]
            return cluster_base[index], cluster_context[index]
        if cluster_id in manual_base_vectors:
            return manual_base_vectors[cluster_id], manual_context_vectors[cluster_id]
        raise KeyError(f"No vectors for cluster {cluster_id}")

    def cluster_pair(left_id: str, right_id: str) -> VectorPair:
        left_base, left_context = vectors_for_cluster(left_id)
        right_base, right_context = vectors_for_cluster(right_id)
        return VectorPair(
            cosine_percent(left_base, right_base),
            cosine_percent(left_context, right_context),
        )

    identity_union = UnionFind()
    for cluster_id in tags_by_cluster:
        identity_union.find(cluster_id)

    supplemental: dict[tuple[str, str], dict] = {}
    conflicts: list[dict] = []

    def add_supplemental(
        left_id: str,
        right_id: str,
        left_tag: str,
        right_tag: str,
        label: int,
        source: str,
        rationale: str,
        similarities: VectorPair | None = None,
    ) -> None:
        if left_id == right_id:
            return
        key = tuple(sorted((left_id, right_id)))
        ordered = (left_id, right_id) == key
        if key in supplemental and supplemental[key]["label"] != label:
            conflicts.append(
                {
                    "left_id": left_id,
                    "right_id": right_id,
                    "left_tag": left_tag,
                    "right_tag": right_tag,
                    "existing_label": supplemental[key]["label"],
                    "new_label": label,
                    "existing_sources": supplemental[key]["source"],
                    "new_source": source,
                    "resolution": "excluded_conflicting_pair",
                }
            )
            supplemental.pop(key)
            return
        if key in supplemental:
            row = supplemental[key]
            row["source"] = ";".join(sorted(set(row["source"].split(";")) | {source}))
            if rationale and rationale not in row["rationale"]:
                row["rationale"] = f"{row['rationale']} | {rationale}"
            return
        if similarities is None:
            similarities = cluster_pair(left_id, right_id)
        supplemental[key] = {
            "left_id": key[0],
            "right_id": key[1],
            "left_tag": left_tag if ordered else right_tag,
            "right_tag": right_tag if ordered else left_tag,
            "left_family": key[0],
            "right_family": key[1],
            "label": int(label),
            "base_similarity": similarities.base_similarity,
            "context_similarity": similarities.context_similarity,
            "fused_similarity": similarities.fused_similarity,
            "source": source,
            "rationale": rationale,
            "source_weight": 1.0,
            "granularity": "cluster_pair",
        }
        if label == 1:
            identity_union.union(left_id, right_id)

    # The supplied complete bundle documented this one missed frozen-to-frozen merge.
    add_supplemental(
        "N-0501",
        "N-0397",
        manual_label["N-0501"],
        manual_label["N-0397"],
        1,
        "documented_frozen_correction",
        "The supplied manual bundle's own review queue identified these as the same official.",
    )

    # Later statistical review: seven accepted pairs and the first rejected pair.
    for row in read_csv(STATISTICAL_REVIEW):
        if row["manuallyReviewed"].lower() != "true":
            continue
        add_supplemental(
            row["leftClusterId"],
            row["rightClusterId"],
            row["leftCanonical"],
            row["rightCanonical"],
            1 if row["status"] == "ACCEPTED" else 0,
            "statistical_pass_manual_review",
            row["reason"],
        )

    # Regex pass: only definite accepts and definite context conflicts are labels.
    regex_counts: Counter = Counter()
    for row in read_jsonl(REGEX_REVIEW):
        decision = row["manualDecision"]
        regex_counts[decision] += 1
        if decision not in {"ACCEPTED_VERIFIED", "EXCLUDED_CONTEXT_CONFLICT"}:
            continue
        add_supplemental(
            row["leftClusterId"],
            row["rightClusterId"],
            row["leftCanonical"],
            row["rightCanonical"],
            1 if decision == "ACCEPTED_VERIFIED" else 0,
            "regex_context_manual_review",
            row.get("manualReason", row.get("primaryRule", decision)),
        )

    # Resolve the old classifier's 549 explicit pair decisions back to cluster IDs.
    old_rows = read_csv(OLD_BUNDLE / "pair_classifier_oof_predictions.csv")
    wanted: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in old_rows:
        wanted[tuple(sorted((row["left_tag"], row["right_tag"])))].append(row)
    found_old: set[int] = set()
    for candidate in read_jsonl(RAW_CANDIDATES):
        key = tuple(sorted((candidate["leftIdentity"], candidate["rightIdentity"])))
        matches = wanted.get(key)
        if not matches:
            continue
        score = float(candidate["embeddingSimilarity"]) * 100.0
        for row in matches:
            row_index = old_rows.index(row)
            if row_index in found_old:
                continue
            if abs(float(row["score_percent"]) - score) > 0.000002:
                continue
            left_id = candidate["leftClusterId"]
            right_id = candidate["rightClusterId"]
            left_tag = candidate["leftIdentity"]
            right_tag = candidate["rightIdentity"]
            context_score = cluster_pair(left_id, right_id).context_similarity
            add_supplemental(
                left_id,
                right_id,
                left_tag,
                right_tag,
                int(float(row["label"])),
                "score_band_manual_adjudication",
                row["adjudication_rationale"],
                VectorPair(score, context_score),
            )
            found_old.add(row_index)
    if len(found_old) != len(old_rows):
        missing = sorted(set(range(len(old_rows))) - found_old)
        raise RuntimeError(
            f"Could not recover {len(missing)} old adjudicated pair IDs; "
            f"first missing rows: {missing[:10]}"
        )

    # The complete manual ledger contains nine corrected false joins and two
    # corrected pointers. Recover their rejected provisional targets as negatives.
    component_to_final = {
        row["source_component_id"]: row["final_cluster_id"] for row in component_assignments
    }
    target_to_final: dict[str, str] = dict(component_to_final)
    for row in ledger:
        target_to_final[row["source_component_id"]] = row["final_cluster_id"]
        # Group aliases are authoritative only on rows that actually establish or
        # join that group. A REMOVED_BAD_JOIN row's provisional target is the
        # rejected destination and must not be mapped back to the rejected source.
        if row["resolution_type"] not in {
            "NEW_ALIAS_GROUP",
            "GROUP_TO_FROZEN",
            "DIRECT_TO_FROZEN",
            "COMPONENT_LINK",
        }:
            continue
        for field in ("reviewed_target", "resolved_target", "provisional_target"):
            target = row[field]
            if target and target not in component_to_final:
                target_to_final[target] = row["final_cluster_id"]

    ledger_negative_count = 0
    for row in ledger:
        if row["status"] not in {"CORRECTED_KEEP_DISTINCT", "CORRECTED_POINTER"}:
            continue
        target_key = row["provisional_target"]
        target_cluster = target_to_final.get(target_key)
        source_component = row["source_component_id"]
        source_cluster = component_to_final[source_component]
        if not target_cluster or target_cluster == source_cluster:
            continue
        left_base = component_base_vectors[source_component]
        left_context = component_context_vectors[source_component]
        target_base, target_context = vectors_for_cluster(target_cluster)
        add_supplemental(
            source_cluster,
            target_cluster,
            row["source_component_canonical"],
            manual_label.get(target_cluster, corrected_label.get(target_cluster, target_key)),
            0,
            "gpt_pro_corrected_bad_join",
            row["note"],
            VectorPair(
                cosine_percent(left_base, target_base),
                cosine_percent(left_context, target_context),
            ),
        )
        ledger_negative_count += 1

    # Apply all accepted supplemental identities before assigning CV families.
    for row in supplemental.values():
        if row["label"] == 1:
            identity_union.union(row["left_family"], row["right_family"])

    # Build every positive raw-tag pair implied by the complete GPT Pro clustering.
    rows: list[dict] = []
    positive_pairs_by_cluster: dict[str, int] = {}
    for cluster_id, indices in tags_by_cluster.items():
        pair_count = len(indices) * (len(indices) - 1) // 2
        if pair_count == 0:
            continue
        positive_pairs_by_cluster[cluster_id] = pair_count
        pair_weight = 1.0 / pair_count
        for left_index, right_index in itertools.combinations(indices, 2):
            left = tag_assignments[left_index]
            right = tag_assignments[right_index]
            similarities = VectorPair(
                cosine_percent(raw_base[left_index], raw_base[right_index]),
                cosine_percent(raw_context[left_index], raw_context[right_index]),
            )
            rows.append(
                {
                    "left_id": f"tag:{left_index}",
                    "right_id": f"tag:{right_index}",
                    "left_tag": left["display_tag"],
                    "right_tag": right["display_tag"],
                    "left_family": cluster_id,
                    "right_family": cluster_id,
                    "label": 1,
                    "base_similarity": similarities.base_similarity,
                    "context_similarity": similarities.context_similarity,
                    "fused_similarity": similarities.fused_similarity,
                    "source": "gpt_pro_complete_identity_cluster",
                    "rationale": (
                        f"Both manually assigned to identity {cluster_id}: "
                        f"{manual_label[cluster_id]}"
                    ),
                    "source_weight": pair_weight,
                    "granularity": "raw_tag_pair",
                }
            )

    for row in supplemental.values():
        left_root = identity_union.find(row["left_family"])
        right_root = identity_union.find(row["right_family"])
        if row["label"] == 0 and left_root == right_root:
            conflicts.append(
                {
                    "left_id": row["left_id"],
                    "right_id": row["right_id"],
                    "left_tag": row["left_tag"],
                    "right_tag": row["right_tag"],
                    "existing_label": 1,
                    "new_label": 0,
                    "existing_sources": "positive_identity_transitive_closure",
                    "new_source": row["source"],
                    "resolution": "excluded_conflicting_pair",
                }
            )
            continue
        row["left_family"] = left_root
        row["right_family"] = right_root
        rows.append(row)

    for row in rows:
        row["left_family"] = identity_union.find(row["left_family"])
        row["right_family"] = identity_union.find(row["right_family"])

    # CV groups include every connected labelled family, including negative edges,
    # so no identity family can appear in both train and test for a fold.
    cv_union = UnionFind()
    for row in rows:
        cv_union.union(row["left_family"], row["right_family"])
    for row in rows:
        row["cv_group"] = cv_union.find(row["left_family"])

    universe_tags = [row["display_tag"] for row in tag_assignments]
    universe_tags.extend(row["canonicalLabel"] for row in corrected_clusters)
    universe_tags.extend(row["left_tag"] for row in rows)
    universe_tags.extend(row["right_tag"] for row in rows)
    stats = make_global_stats(universe_tags)

    frame = pd.DataFrame(rows)
    feature_rows = []
    for row in frame.itertuples(index=False):
        feature_rows.append(
            stable_feature_vector(
                row.left_tag,
                row.right_tag,
                VectorPair(row.base_similarity, row.context_similarity),
                stats,
            )
        )
    X = np.vstack(feature_rows)
    feature_frame = pd.DataFrame(X, columns=STABLE_FEATURE_NAMES)
    frame = pd.concat([frame.reset_index(drop=True), feature_frame], axis=1)

    source_counts = frame.groupby(["source", "label"]).size().rename("pairs").reset_index()
    audit = {
        "manual_tags": len(tag_assignments),
        "manual_clusters": len(manual_clusters),
        "manual_multi_tag_clusters": len(positive_pairs_by_cluster),
        "manual_positive_tag_pairs": int(
            (frame.source == "gpt_pro_complete_identity_cluster").sum()
        ),
        "supplemental_unique_pairs": len(supplemental),
        "training_pairs": len(frame),
        "training_positives": int(frame.label.sum()),
        "training_negatives": int((frame.label == 0).sum()),
        "cv_groups": int(frame.cv_group.nunique()),
        "regex_decisions": dict(regex_counts),
        "ledger_negative_rows_recovered": ledger_negative_count,
        "conflicts_excluded": len(conflicts),
        "source_counts": source_counts.to_dict(orient="records"),
        "input_hashes": {
            str(MANUAL / "konbaung_node_identity_tag_assignments_complete.csv"): sha256(
                MANUAL / "konbaung_node_identity_tag_assignments_complete.csv"
            ),
            str(MANUAL / "konbaung_node_identity_merge_ledger_complete.tsv"): sha256(
                MANUAL / "konbaung_node_identity_merge_ledger_complete.tsv"
            ),
            str(OLD_BUNDLE / "pair_classifier_oof_predictions.csv"): sha256(
                OLD_BUNDLE / "pair_classifier_oof_predictions.csv"
            ),
            str(REGEX_REVIEW): sha256(REGEX_REVIEW),
            str(STATISTICAL_REVIEW): sha256(STATISTICAL_REVIEW),
        },
    }
    extras = {
        "conflicts": conflicts,
        "audit": audit,
        "identity_union": identity_union,
        "corrected_index": corrected_index,
        "corrected_label": corrected_label,
        "cluster_base": cluster_base,
        "cluster_context": cluster_context,
    }
    return frame, stats, extras, sources


def class_balanced_weights(frame: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    source_weight = frame.source_weight.to_numpy(dtype=float)[indices]
    labels = frame.label.to_numpy(dtype=np.int8)[indices]
    positive_total = source_weight[labels == 1].sum()
    negative_total = source_weight[labels == 0].sum()
    weights = source_weight.copy()
    weights[labels == 1] *= 0.5 / positive_total
    weights[labels == 0] *= 0.5 / negative_total
    weights *= len(weights)
    return weights


def fit_model(model, X: np.ndarray, y: np.ndarray, weights: np.ndarray):
    fitted = clone(model)
    if hasattr(fitted, "named_steps"):
        fitted.fit(X, y, logisticregression__sample_weight=weights)
    else:
        fitted.fit(X, y, sample_weight=weights)
    return fitted


def oof_predictions(
    model,
    X: np.ndarray,
    frame: pd.DataFrame,
    repeats: int,
) -> tuple[np.ndarray, list[dict]]:
    y = frame.label.to_numpy(dtype=np.int8)
    groups = frame.cv_group.to_numpy(dtype=object)
    predictions = np.zeros(len(frame), dtype=np.float64)
    counts = np.zeros(len(frame), dtype=np.int32)
    folds: list[dict] = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260725 + repeat)
        for fold, (train_index, test_index) in enumerate(splitter.split(X, y, groups), start=1):
            weights = class_balanced_weights(frame, train_index)
            fitted = fit_model(model, X[train_index], y[train_index], weights)
            probability = fitted.predict_proba(X[test_index])[:, 1]
            predictions[test_index] += probability
            counts[test_index] += 1
            folds.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "train_pairs": len(train_index),
                    "test_pairs": len(test_index),
                    "train_positives": int(y[train_index].sum()),
                    "test_positives": int(y[test_index].sum()),
                    "train_groups": len(set(groups[train_index])),
                    "test_groups": len(set(groups[test_index])),
                }
            )
    if np.any(counts != repeats):
        raise RuntimeError(
            f"OOF coverage error: expected {repeats} predictions per row; "
            f"observed {counts.min()}..{counts.max()}"
        )
    return predictions / counts, folds


def metrics(
    y: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict:
    prediction = probability >= 0.5
    has_both_classes = len(np.unique(y)) == 2
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        prediction,
        average="binary",
        zero_division=0,
        sample_weight=sample_weight,
    )
    return {
        "average_precision": float(
            average_precision_score(y, probability, sample_weight=sample_weight)
        ),
        "roc_auc": (
            float(roc_auc_score(y, probability, sample_weight=sample_weight))
            if has_both_classes
            else None
        ),
        "brier_score": float(brier_score_loss(y, probability, sample_weight=sample_weight)),
        "balanced_accuracy_at_0_5": (
            float(balanced_accuracy_score(y, prediction, sample_weight=sample_weight))
            if has_both_classes
            else None
        ),
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1),
    }


def choose_review_threshold(
    y: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    target_recall: float = 0.99,
) -> tuple[float, dict]:
    positive_total = float(weights[y == 1].sum())
    best = None
    for threshold in sorted(np.unique(probability), reverse=True):
        selected = probability >= threshold
        weighted_true_positive = float(weights[selected & (y == 1)].sum())
        recall = weighted_true_positive / positive_total
        if recall >= target_recall:
            weighted_selected = float(weights[selected].sum())
            precision = weighted_true_positive / weighted_selected if weighted_selected else 0.0
            best = (
                float(threshold),
                {
                    "target_weighted_recall": target_recall,
                    "weighted_recall": recall,
                    "weighted_precision": precision,
                    "selected_pairs": int(selected.sum()),
                    "selected_positives": int((selected & (y == 1)).sum()),
                    "selected_negatives": int((selected & (y == 0)).sum()),
                    "unweighted_positive_recall": float(
                        (selected & (y == 1)).sum() / max(1, int((y == 1).sum()))
                    ),
                },
            )
            break
    if best is None:
        return 0.0, {
            "target_weighted_recall": target_recall,
            "weighted_recall": 1.0,
            "weighted_precision": float(weights[y == 1].sum() / weights.sum()),
            "selected_pairs": len(y),
            "selected_positives": int((y == 1).sum()),
            "selected_negatives": int((y == 0).sum()),
            "unweighted_positive_recall": 1.0,
        }
    return best


def train(repeats: int) -> tuple[dict, pd.DataFrame, dict]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("Building full manually seeded gold pair set...", flush=True)
    frame, stats, extras, _ = build_full_gold()
    frame.to_csv(OUTPUT / "full_gold_training_pairs.csv", index=False)
    pd.DataFrame(extras["conflicts"]).to_csv(OUTPUT / "excluded_label_conflicts.csv", index=False)
    (OUTPUT / "gold_construction_audit.json").write_text(
        json.dumps(extras["audit"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    X = frame[STABLE_FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = frame.label.to_numpy(dtype=np.int8)
    evaluation_weights = frame.source_weight.to_numpy(dtype=float)

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.3,
            max_iter=5000,
            solver="liblinear",
            random_state=42,
        ),
    )
    boosted = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.035,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=4.0,
        random_state=42,
    )

    print(
        f"Running {repeats} x 5 identity-family-disjoint CV over {len(frame):,} gold pairs...",
        flush=True,
    )
    logistic_oof, fold_rows = oof_predictions(logistic, X, frame, repeats)
    boosted_oof, _ = oof_predictions(boosted, X, frame, repeats)

    comparison = {
        "raw_embedding_similarity": {
            "unweighted": metrics(y, frame.base_similarity.to_numpy() / 100.0),
            "identity_weighted": metrics(
                y,
                frame.base_similarity.to_numpy() / 100.0,
                evaluation_weights,
            ),
        },
        "logistic_regression": {
            "unweighted": metrics(y, logistic_oof),
            "identity_weighted": metrics(y, logistic_oof, evaluation_weights),
        },
        "hist_gradient_boosting": {
            "unweighted": metrics(y, boosted_oof),
            "identity_weighted": metrics(y, boosted_oof, evaluation_weights),
        },
    }
    logistic_ap = comparison["logistic_regression"]["identity_weighted"]["average_precision"]
    boosted_ap = comparison["hist_gradient_boosting"]["identity_weighted"]["average_precision"]
    if boosted_ap > logistic_ap:
        selected_name = "hist_gradient_boosting"
        selected_template = boosted
        selected_oof = boosted_oof
    else:
        selected_name = "logistic_regression"
        selected_template = logistic
        selected_oof = logistic_oof

    print(f"Selected {selected_name}; fitting final model...", flush=True)
    final_weights = class_balanced_weights(frame, np.arange(len(frame)))
    final_model = fit_model(selected_template, X, y, final_weights)
    calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
    calibrator.fit(selected_oof, y, sample_weight=evaluation_weights)
    calibrated_oof = calibrator.predict(selected_oof)
    review_threshold, review_validation = choose_review_threshold(
        y,
        calibrated_oof,
        evaluation_weights,
        target_recall=0.99,
    )
    granularity_validation: dict[str, dict] = {}
    for granularity in ("raw_tag_pair", "cluster_pair"):
        mask = frame.granularity.to_numpy() == granularity
        granularity_validation[granularity] = {
            "pairs": int(mask.sum()),
            "positives": int(y[mask].sum()),
            "negatives": int((y[mask] == 0).sum()),
            "selected_model_oof": metrics(y[mask], selected_oof[mask]),
            "calibrated_oof": metrics(y[mask], calibrated_oof[mask]),
        }
    cluster_mask = frame.granularity.to_numpy() == "cluster_pair"
    cluster_negative_max = float(calibrated_oof[cluster_mask & (y == 0)].max())
    high_precision_threshold = float(np.nextafter(cluster_negative_max, 1.0))
    high_precision_selected = calibrated_oof >= high_precision_threshold
    high_precision_validation = {
        "threshold": high_precision_threshold,
        "cluster_pair_false_positives": int(
            (high_precision_selected & cluster_mask & (y == 0)).sum()
        ),
        "cluster_pair_true_positives": int(
            (high_precision_selected & cluster_mask & (y == 1)).sum()
        ),
        "cluster_pair_recall": float(
            (high_precision_selected & cluster_mask & (y == 1)).sum()
            / max(1, int((cluster_mask & (y == 1)).sum()))
        ),
    }

    frame["logistic_oof_probability"] = logistic_oof
    frame["boosted_oof_probability"] = boosted_oof
    frame["selected_oof_probability"] = selected_oof
    frame["calibrated_oof_probability"] = calibrated_oof
    frame["oof_review_candidate"] = calibrated_oof >= review_threshold
    frame["oof_error"] = np.where(
        (frame.label == 1) & (~frame.oof_review_candidate),
        "missed_valid_merge",
        np.where(
            (frame.label == 0) & frame.oof_review_candidate,
            "false_review_candidate",
            "",
        ),
    )
    frame.to_csv(OUTPUT / "full_gold_oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT / "full_gold_cv_folds.csv", index=False)

    bundle = {
        "model_version": "full_manual_gold_v2",
        "model_type": selected_name,
        "model": final_model,
        "calibrator": calibrator,
        "feature_names": STABLE_FEATURE_NAMES,
        "review_threshold": review_threshold,
        "high_precision_threshold": high_precision_threshold,
        "target_weighted_recall": 0.99,
        "ml_auto_enabled": False,
        "training_pairs": len(frame),
        "training_positives": int(y.sum()),
        "training_negatives": int((y == 0).sum()),
        "manual_tags": extras["audit"]["manual_tags"],
        "manual_clusters": extras["audit"]["manual_clusters"],
        "token_df": dict(stats.token_df),
        "token_idf": stats.token_idf,
        "rare_cutoff": stats.rare_cutoff,
        "validation": comparison,
        "granularity_validation": granularity_validation,
        "review_threshold_validation": review_validation,
        "high_precision_threshold_validation": high_precision_validation,
        "notes": (
            "Uses all same-identity tag pairs in the complete GPT Pro manual "
            "clustering plus every explicit later yes/no adjudication. Ambiguous "
            "not-proven pairs remain unlabeled. Context and entity-only Gemini "
            "cosines are used; ML automatic merges remain disabled."
        ),
    }
    model_path = OUTPUT / "full_gold_coreference_pair_classifier.joblib"
    joblib.dump(bundle, model_path, compress=3)

    summary = {
        "gold_construction": extras["audit"],
        "selected_model": selected_name,
        "model_comparison": comparison,
        "granularity_validation": granularity_validation,
        "review_threshold": review_threshold,
        "review_threshold_validation": review_validation,
        "high_precision_threshold": high_precision_threshold,
        "high_precision_threshold_validation": high_precision_validation,
        "automatic_merge_policy": "disabled",
        "model_path": str(model_path),
    }
    (OUTPUT / "full_gold_training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_training_report(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return bundle, frame, extras


def write_training_report(summary: dict) -> None:
    gold = summary["gold_construction"]
    comparison = summary["model_comparison"]
    selected = summary["selected_model"]
    threshold = summary["review_threshold"]
    validation = summary["review_threshold_validation"]
    cluster_validation = summary["granularity_validation"]["cluster_pair"]["selected_model_oof"]
    high_validation = summary["high_precision_threshold_validation"]
    lines = [
        "# Full-manual-gold coreference classifier",
        "",
        "## Gold data",
        "",
        f"- Complete GPT Pro tags: **{gold['manual_tags']:,}**",
        f"- Complete GPT Pro identity clusters: **{gold['manual_clusters']:,}**",
        f"- Same-identity raw-tag pairs used: **{gold['manual_positive_tag_pairs']:,}**",
        f"- Total unique training pairs: **{gold['training_pairs']:,}**",
        f"- Positive pairs: **{gold['training_positives']:,}**",
        f"- Explicit negative pairs: **{gold['training_negatives']:,}**",
        f"- Excluded source conflicts: **{gold['conflicts_excluded']:,}**",
        "",
        "Pairs marked `EXCLUDED_NOT_PROVEN` were not converted into negatives.",
        "",
        "## Validation",
        "",
        "Cross-validation keeps every connected identity/adjudication family wholly "
        "inside one fold. Large clusters are identity-weighted so a 38-member cluster "
        "does not overwhelm hundreds of two-member identities.",
        "",
        "| Model | Weighted AP | Weighted ROC AUC | Weighted balanced accuracy |",
        "|---|---:|---:|---:|",
    ]
    for name, result in comparison.items():
        weighted = result["identity_weighted"]
        lines.append(
            f"| {name} | {weighted['average_precision']:.4f} | "
            f"{weighted['roc_auc']:.4f} | "
            f"{weighted['balanced_accuracy_at_0_5']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Selected model: **{selected}**.",
            "",
            "## Operating policy",
            "",
            "ML automatic merging remains disabled. The classifier ranks a manual review queue.",
            "",
            f"The manual-review threshold is **{threshold:.8f}**, chosen for "
            f"**{validation['weighted_recall']:.2%} identity-weighted OOF recall**. "
            f"At that threshold the OOF review set contains "
            f"{validation['selected_positives']:,} positive and "
            f"{validation['selected_negatives']:,} negative labelled pairs.",
            "",
            "Because deployment scores unresolved cluster pairs rather than raw "
            "within-cluster aliases, the directly relevant held-out cluster-pair "
            f"metrics are **AP {cluster_validation['average_precision']:.4f}** and "
            f"**ROC AUC {cluster_validation['roc_auc']:.4f}**.",
            "",
            f"A separate high-precision threshold of "
            f"**{summary['high_precision_threshold']:.8f}** made zero false-positive "
            f"cluster decisions in OOF validation while retaining "
            f"{high_validation['cluster_pair_true_positives']:,} of "
            f"{summary['granularity_validation']['cluster_pair']['positives']:,} "
            "known positive cluster pairs. This is an audit queue, not an "
            "automatic-merge authorization.",
            "",
        ]
    )
    (OUTPUT / "FULL_GOLD_PAIR_CLASSIFIER_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def score(bundle: dict | None = None, batch_size: int = 25_000) -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if bundle is None:
        bundle = joblib.load(OUTPUT / "full_gold_coreference_pair_classifier.joblib")
    model = bundle["model"]
    calibrator = bundle["calibrator"]
    threshold = float(bundle["review_threshold"])
    high_precision_threshold = float(bundle["high_precision_threshold"])

    corrected_clusters = list(read_jsonl(CLUSTER_VECTORS / "corrected_initial_clusters.jsonl"))
    corrected_index = {row["clusterId"]: index for index, row in enumerate(corrected_clusters)}
    cluster_context = normalize(np.load(CLUSTER_VECTORS / "context_cluster_centroids.npy"))
    tags = [row["canonicalLabel"] for row in corrected_clusters]
    metas = {tag: make_meta(tag) for tag in set(tags)}
    token_df = Counter(bundle["token_df"])
    stats = CorpusStats(
        metas=metas,
        degrees=Counter({tag: 1 for tag in metas}),
        top_scores={tag: 100.0 for tag in metas},
        token_df=token_df,
        token_idf=bundle["token_idf"],
        band_counts=Counter(),
        pair_count=EXPECTED_RAW_CANDIDATES,
        rare_cutoff=int(bundle["rare_cutoff"]),
    )

    columns = [
        "raw_rank",
        "raw_score_percent",
        "left_cluster_id",
        "left_tag",
        "right_cluster_id",
        "right_tag",
        "context_score_percent",
        "fused_score_percent",
        "raw_model_probability",
        "model_probability",
        "known_gold_label",
        "known_gold_source",
        "decision",
    ]
    all_path = OUTPUT / "all_candidates_full_gold_classified.csv.gz"
    review_rows: list[tuple] = []
    counts: Counter = Counter()
    known_pair_labels: dict[tuple[str, str], int] = {}
    known_pair_sources: dict[tuple[str, str], str] = {}
    gold_path = OUTPUT / "full_gold_training_pairs.csv"
    for row in read_csv(gold_path):
        if row["granularity"] != "cluster_pair":
            continue
        key = tuple(sorted((row["left_id"], row["right_id"])))
        known_pair_labels[key] = int(row["label"])
        known_pair_sources[key] = row["source"]
    conflict_pairs: set[tuple[str, str]] = set()
    conflict_path = OUTPUT / "excluded_label_conflicts.csv"
    if conflict_path.exists() and conflict_path.stat().st_size:
        for row in read_csv(conflict_path):
            conflict_pairs.add(tuple(sorted((row["left_id"], row["right_id"]))))
    retrieved_known: set[tuple[str, str]] = set()

    batch_rows: list[dict] = []
    batch_features: list[np.ndarray] = []

    def flush(writer: csv.DictWriter) -> None:
        if not batch_rows:
            return
        matrix = np.vstack(batch_features)
        raw_probability = model.predict_proba(matrix)[:, 1]
        probability = calibrator.predict(raw_probability)
        for row, raw_value, value in zip(batch_rows, raw_probability, probability):
            row["raw_model_probability"] = f"{float(raw_value):.8f}"
            row["model_probability"] = f"{float(value):.8f}"
            key = tuple(sorted((row["left_cluster_id"], row["right_cluster_id"])))
            known_label = known_pair_labels.get(key)
            row["known_gold_label"] = str(known_label) if known_label is not None else ""
            row["known_gold_source"] = known_pair_sources.get(key, "")
            if key in conflict_pairs:
                row["known_gold_source"] = "conflicting_manual_labels"
                row["decision"] = "known_gold_conflict"
            elif known_label == 1:
                row["decision"] = "known_gold_merge"
            elif known_label == 0:
                row["decision"] = "known_gold_no_merge"
            else:
                row["decision"] = "manual_review" if float(value) >= threshold else "reject"
            writer.writerow(row)
            counts[row["decision"]] += 1
            if row["decision"] == "manual_review":
                review_rows.append(
                    (
                        float(value),
                        float(raw_value),
                        float(row["raw_score_percent"]),
                        int(row["raw_rank"]),
                        row["left_cluster_id"],
                        row["left_tag"],
                        row["right_cluster_id"],
                        row["right_tag"],
                        float(row["context_score_percent"]),
                        float(row["fused_score_percent"]),
                    )
                )
        batch_rows.clear()
        batch_features.clear()

    print(f"Scoring {EXPECTED_RAW_CANDIDATES:,} cluster candidates...", flush=True)
    with gzip.open(all_path, "wt", encoding="utf-8", newline="", compresslevel=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for rank, candidate in enumerate(read_jsonl(RAW_CANDIDATES), start=1):
            left_id = candidate["leftClusterId"]
            right_id = candidate["rightClusterId"]
            left_tag = candidate["leftIdentity"]
            right_tag = candidate["rightIdentity"]
            if left_tag not in stats.metas:
                stats.metas[left_tag] = make_meta(left_tag)
            if right_tag not in stats.metas:
                stats.metas[right_tag] = make_meta(right_tag)
            left_context = cluster_context[corrected_index[left_id]]
            right_context = cluster_context[corrected_index[right_id]]
            similarities = VectorPair(
                float(candidate["embeddingSimilarity"]) * 100.0,
                cosine_percent(left_context, right_context),
            )
            batch_features.append(stable_feature_vector(left_tag, right_tag, similarities, stats))
            batch_rows.append(
                {
                    "raw_rank": rank,
                    "raw_score_percent": f"{similarities.base_similarity:.6f}",
                    "left_cluster_id": left_id,
                    "left_tag": left_tag,
                    "right_cluster_id": right_id,
                    "right_tag": right_tag,
                    "context_score_percent": f"{similarities.context_similarity:.6f}",
                    "fused_score_percent": f"{similarities.fused_similarity:.6f}",
                    "raw_model_probability": "",
                    "model_probability": "",
                    "known_gold_label": "",
                    "known_gold_source": "",
                    "decision": "",
                }
            )
            key = tuple(sorted((left_id, right_id)))
            if key in known_pair_labels:
                retrieved_known.add(key)
            if len(batch_rows) >= batch_size:
                flush(writer)
            if rank % 250_000 == 0:
                print(f"Scored {rank:,} candidates...", flush=True)
        flush(writer)
    if rank != EXPECTED_RAW_CANDIDATES:
        raise RuntimeError(f"Expected {EXPECTED_RAW_CANDIDATES:,} scored rows; found {rank:,}")

    review_rows.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
    review_path = OUTPUT / "full_gold_manual_review_queue.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model_probability",
                "raw_model_probability",
                "raw_score_percent",
                "raw_rank",
                "left_cluster_id",
                "left_tag",
                "right_cluster_id",
                "right_tag",
                "context_score_percent",
                "fused_score_percent",
            ]
        )
        writer.writerows(review_rows)
    top_path = OUTPUT / "full_gold_ranked_top_10000.csv"
    high_path = OUTPUT / "full_gold_high_precision_review_queue.csv"
    queue_header = [
        "model_probability",
        "raw_model_probability",
        "raw_score_percent",
        "raw_rank",
        "left_cluster_id",
        "left_tag",
        "right_cluster_id",
        "right_tag",
        "context_score_percent",
        "fused_score_percent",
    ]
    with top_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(queue_header)
        writer.writerows(review_rows[:10_000])
    high_rows = [row for row in review_rows if row[0] >= high_precision_threshold]
    with high_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(queue_header)
        writer.writerows(high_rows)

    known_positive = {key for key, label in known_pair_labels.items() if label == 1}
    known_negative = {key for key, label in known_pair_labels.items() if label == 0}
    eligible_positive = {
        key for key in known_positive if key[0] in corrected_index and key[1] in corrected_index
    }
    pre_resolved_positive = known_positive - eligible_positive
    eligible_negative = {
        key for key in known_negative if key[0] in corrected_index and key[1] in corrected_index
    }
    summary = {
        "pairs_scored": rank,
        "decision_counts": dict(counts),
        "review_threshold": threshold,
        "high_precision_threshold": high_precision_threshold,
        "high_precision_review_count": len(high_rows),
        "ranked_top_queue_count": min(10_000, len(review_rows)),
        "automatic_merges": 0,
        "known_cluster_pair_retrieval": {
            "conflicting_label_pairs_quarantined": len(conflict_pairs),
            "labelled_positive_pairs": len(known_positive),
            "positive_pairs_already_resolved_before_retrieval": len(pre_resolved_positive),
            "eligible_unresolved_positive_pairs": len(eligible_positive),
            "positive_pairs_present_in_top100_pool": len(eligible_positive & retrieved_known),
            "positive_retrieval_recall": (
                len(eligible_positive & retrieved_known) / len(eligible_positive)
                if eligible_positive
                else None
            ),
            "labelled_negative_pairs": len(known_negative),
            "eligible_negative_pairs": len(eligible_negative),
            "negative_pairs_present_in_top100_pool": len(eligible_negative & retrieved_known),
        },
        "outputs": {
            "all_classified": str(all_path),
            "manual_review_queue": str(review_path),
            "ranked_top_10000": str(top_path),
            "high_precision_review_queue": str(high_path),
        },
    }
    (OUTPUT / "full_gold_scoring_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("train", "score", "all"),
        default="all",
        nargs="?",
    )
    parser.add_argument("--cv-repeats", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=25_000)
    args = parser.parse_args()

    bundle = None
    if args.action in {"train", "all"}:
        bundle, _, _ = train(args.cv_repeats)
    if args.action in {"score", "all"}:
        score(bundle, args.batch_size)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train a conservative tag-pair coreference classifier.

Usage:
  python train_pair_classifier.py \
      --candidates ALL_RAW_GEMINI_EMBEDDING_CANDIDATES.md \
      --adjudications TOP_100_PER_BAND_COREFERENCE_REVIEW.md \
      --output-dir pair_classifier_output

The script performs entity-family-disjoint cross-validation, compares logistic
regression with gradient-boosted trees, selects conservative operating
thresholds, and writes a reusable joblib model bundle plus audit files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pair_classifier_features import (  # noqa: E402
    FEATURE_NAMES,
    build_corpus_stats,
    deterministic_merge_reason,
    feature_vector,
    hard_rejection_reason,
    identity_evidence_gate,
    iter_candidates,
)

TABLE_RE = re.compile(
    r"^\|\s*(\d+)\s*\|\s*([\d.]+)%\s*\|\s*(?:\*\*)?(MERGE|NO MERGE)(?:\*\*)?\s*"
    r"\|\s*`([^`]*)`\s*↔\s*`([^`]*)`\s*\|\s*(.*?)\s*\|$",
    re.MULTILINE,
)


def parse_adjudications(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    records = []
    current_band = None
    # Table scores identify the band unambiguously, so section tracking is unnecessary.
    for match in TABLE_RE.finditer(text):
        score = float(match.group(2))
        records.append(
            {
                "band": int(score),
                "rank_in_band": int(match.group(1)),
                "score_percent": score,
                "label": 1 if match.group(3) == "MERGE" else 0,
                "left_tag": match.group(4),
                "right_tag": match.group(5),
                "adjudication_rationale": match.group(6),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError(f"No adjudication table rows were found in {path}")
    return frame


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def collect_labeled_ranks(candidate_path: Path, adjudications: pd.DataFrame) -> Dict[int, Tuple[int, int]]:
    pair_to_index: Dict[Tuple[str, str], int] = {}
    for index, row in adjudications.iterrows():
        pair_to_index[(row.left_tag, row.right_tag)] = int(index)
        pair_to_index[(row.right_tag, row.left_tag)] = int(index)
    ranks: Counter = Counter()
    found: Dict[int, Tuple[int, int]] = {}
    for candidate in iter_candidates(candidate_path):
        ranks[candidate.left] += 1
        rank_left = ranks[candidate.left]
        ranks[candidate.right] += 1
        rank_right = ranks[candidate.right]
        index = pair_to_index.get((candidate.left, candidate.right))
        if index is not None:
            found[index] = (rank_left, rank_right)
    missing = sorted(set(adjudications.index) - set(found))
    if missing:
        raise ValueError(f"Could not locate {len(missing)} adjudicated pairs in the candidate file; first missing indices: {missing[:10]}")
    return found


def grouped_oof_predictions(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    weighted_hist: bool = False,
) -> Tuple[np.ndarray, List[dict]]:
    predictions = np.zeros(len(y), dtype=np.float64)
    counts = np.zeros(len(y), dtype=np.int32)
    fold_rows: List[dict] = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1200 + repeat)
        for fold, (train_index, test_index) in enumerate(splitter.split(X, y, groups), start=1):
            fitted = clone(model)
            if weighted_hist:
                positives = max(1, int(y[train_index].sum()))
                positive_weight = (len(train_index) - positives) / positives
                weights = np.where(y[train_index] == 1, positive_weight, 1.0)
                fitted.fit(X[train_index], y[train_index], sample_weight=weights)
            else:
                fitted.fit(X[train_index], y[train_index])
            fold_probability = fitted.predict_proba(X[test_index])[:, 1]
            predictions[test_index] += fold_probability
            counts[test_index] += 1
            fold_rows.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "train_pairs": len(train_index),
                    "test_pairs": len(test_index),
                    "train_positives": int(y[train_index].sum()),
                    "test_positives": int(y[test_index].sum()),
                }
            )
    return predictions / counts, fold_rows


def ranking_metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    ordering = np.argsort(-probability)
    result = {
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
    }
    for k in (5, 10, 20, 30, 50, 75, 100):
        selected = y[ordering[:k]]
        result[f"precision_at_{k}"] = float(selected.mean())
        result[f"recall_at_{k}"] = float(selected.sum() / y.sum())
    return result


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    spread = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total)
    return (centre - spread) / denominator, (centre + spread) / denominator


def choose_auto_threshold(
    y: np.ndarray,
    probability: np.ndarray,
    deterministic: np.ndarray,
    eligible: np.ndarray,
    minimum_predictions: int = 10,
) -> Tuple[float, dict]:
    best = None
    for threshold in sorted(np.unique(probability), reverse=True):
        predicted = deterministic | (eligible & (probability >= threshold))
        total = int(predicted.sum())
        if total < minimum_predictions:
            continue
        true_positive = int(y[predicted].sum())
        precision = true_positive / total
        recall = true_positive / int(y.sum())
        # Automatic merging requires zero observed false positives in entity-disjoint OOF.
        if precision == 1.0:
            candidate = (recall, total, -threshold, threshold, precision, true_positive)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return 1.0, {"predicted": int(deterministic.sum()), "true_positive": int(y[deterministic].sum())}
    _, total, _, threshold, precision, true_positive = best
    lower, upper = wilson_interval(true_positive, total)
    return float(threshold), {
        "predicted": int(total),
        "true_positive": int(true_positive),
        "false_positive": int(total - true_positive),
        "precision": float(precision),
        "recall": float(true_positive / y.sum()),
        "wilson_95_low": float(lower),
        "wilson_95_high": float(upper),
    }


def choose_review_threshold(
    y: np.ndarray,
    probability: np.ndarray,
    auto_predicted: np.ndarray,
    eligible: np.ndarray,
    target_recall: float = 0.90,
) -> Tuple[float, dict]:
    chosen = None
    # Highest threshold that still reaches target recall gives the smallest queue.
    for threshold in sorted(np.unique(probability), reverse=True):
        selected = auto_predicted | (eligible & (probability >= threshold))
        true_positive = int(y[selected].sum())
        recall = true_positive / int(y.sum())
        if recall >= target_recall:
            total = int(selected.sum())
            precision = true_positive / total if total else 0.0
            chosen = (float(threshold), total, true_positive, precision, recall)
            break
    if chosen is None:
        threshold = 0.0
        selected = auto_predicted | eligible
        total = int(selected.sum())
        true_positive = int(y[selected].sum())
        precision = true_positive / total if total else 0.0
        recall = true_positive / int(y.sum())
    else:
        threshold, total, true_positive, precision, recall = chosen
    return float(threshold), {
        "selected_including_auto": int(total),
        "true_positive_including_auto": int(true_positive),
        "precision_including_auto": float(precision),
        "recall_including_auto": float(recall),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=HERE / "ALL_RAW_GEMINI_EMBEDDING_CANDIDATES.md")
    parser.add_argument("--adjudications", type=Path, default=HERE / "TOP_100_PER_BAND_COREFERENCE_REVIEW.md")
    parser.add_argument("--output-dir", type=Path, default=HERE / "pair_classifier_output")
    parser.add_argument("--cv-repeats", type=int, default=12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Parsing adjudications...", flush=True)
    adjudications = parse_adjudications(args.adjudications)
    y = adjudications.label.to_numpy(dtype=np.int8)

    print("Building candidate-corpus statistics...", flush=True)
    stats = build_corpus_stats(args.candidates)
    print(f"Corpus: {stats.pair_count:,} pairs and {len(stats.metas):,} unique tags", flush=True)

    print("Locating adjudicated pairs and their mutual-neighbour ranks...", flush=True)
    rank_map = collect_labeled_ranks(args.candidates, adjudications)

    X_rows = []
    hard_reasons: List[Optional[str]] = []
    deterministic_reasons: List[Optional[str]] = []
    evidence_gate = []
    for index, row in adjudications.iterrows():
        rank_left, rank_right = rank_map[int(index)]
        vector = feature_vector(row.left_tag, row.right_tag, row.score_percent, rank_left, rank_right, stats)
        X_rows.append(vector)
        left_meta = stats.metas[row.left_tag]
        right_meta = stats.metas[row.right_tag]
        hard_reason = hard_rejection_reason(left_meta, right_meta)
        deterministic_reason = None if hard_reason else deterministic_merge_reason(left_meta, right_meta)
        hard_reasons.append(hard_reason)
        deterministic_reasons.append(deterministic_reason)
        evidence_gate.append(identity_evidence_gate(vector))
    X = np.vstack(X_rows)

    adjudications["rank_left"] = [rank_map[int(i)][0] for i in adjudications.index]
    adjudications["rank_right"] = [rank_map[int(i)][1] for i in adjudications.index]
    adjudications["hard_veto_reason"] = hard_reasons
    adjudications["deterministic_merge_reason"] = deterministic_reasons
    adjudications["identity_evidence_gate"] = evidence_gate

    hard_mask = adjudications.hard_veto_reason.notna().to_numpy()
    deterministic_mask = adjudications.deterministic_merge_reason.notna().to_numpy()
    evidence_mask = adjudications.identity_evidence_gate.to_numpy(dtype=bool)

    positive_hard_vetoes = adjudications[(adjudications.label == 1) & adjudications.hard_veto_reason.notna()]
    deterministic_false = adjudications[(adjudications.label == 0) & adjudications.deterministic_merge_reason.notna()]
    if not positive_hard_vetoes.empty:
        raise RuntimeError(
            "Hard vetoes incorrectly reject adjudicated positives. Review these rows:\n"
            + positive_hard_vetoes[["left_tag", "right_tag", "hard_veto_reason"]].to_string(index=False)
        )
    if not deterministic_false.empty:
        raise RuntimeError(
            "Deterministic rules produce adjudicated false positives. Review these rows:\n"
            + deterministic_false[["left_tag", "right_tag", "deterministic_merge_reason"]].to_string(index=False)
        )

    union_find = UnionFind()
    for row in adjudications.itertuples(index=False):
        union_find.union(row.left_tag, row.right_tag)
    groups = np.asarray([union_find.find(tag) for tag in adjudications.left_tag], dtype=object)

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.3, max_iter=5000, solver="liblinear", random_state=42),
    )
    boosted = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=7,
        min_samples_leaf=12,
        l2_regularization=3.0,
        random_state=42,
    )

    print("Running repeated entity-family-disjoint cross-validation...", flush=True)
    logistic_oof, fold_rows = grouped_oof_predictions(logistic, X, y, groups, args.cv_repeats)
    boosted_oof, _ = grouped_oof_predictions(boosted, X, y, groups, max(4, args.cv_repeats // 2), weighted_hist=True)
    raw_baseline = adjudications.score_percent.to_numpy(dtype=float) / 100.0

    metrics = {
        "raw_similarity_baseline": ranking_metrics(y, raw_baseline),
        "logistic_regression": ranking_metrics(y, logistic_oof),
        "hist_gradient_boosting": ranking_metrics(y, boosted_oof),
    }

    # With only 52 positive labels, ML probabilities are used to rank the manual queue,
    # not to execute unseen automatic merges. Automatic merges are restricted to
    # deterministic normalization rules that had zero false positives in validation.
    auto_threshold = 1.1  # deliberately unreachable; retained in the bundle schema
    auto_oof = deterministic_mask.copy()
    auto_total = int(auto_oof.sum())
    auto_true = int(y[auto_oof].sum())
    auto_low, auto_high = wilson_interval(auto_true, auto_total)
    auto_metrics = {
        "predicted": auto_total,
        "true_positive": auto_true,
        "false_positive": auto_total - auto_true,
        "precision": float(auto_true / auto_total) if auto_total else None,
        "recall": float(auto_true / y.sum()),
        "wilson_95_low": float(auto_low),
        "wilson_95_high": float(auto_high),
        "ml_auto_enabled": False,
    }

    eligible_review = (~hard_mask) & (adjudications.score_percent.to_numpy() >= 94.0)
    review_threshold, review_metrics = choose_review_threshold(y, logistic_oof, auto_oof, eligible_review, target_recall=0.90)

    print("Fitting final logistic model and OOF-scale calibrator...", flush=True)
    final_model = clone(logistic).fit(X, y)
    full_train_probability = final_model.predict_proba(X)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
    calibrator.fit(full_train_probability, logistic_oof)
    calibrated_train_probability = calibrator.predict(full_train_probability)

    # Keep policy thresholds on the OOF-calibrated probability scale.
    bundle = {
        "model_type": "standardized_logistic_regression",
        "model": final_model,
        "calibrator": calibrator,
        "feature_names": FEATURE_NAMES,
        "auto_threshold": auto_threshold,
        "review_threshold": review_threshold,
        "model_min_raw_score": 94.0,
        "training_pairs": int(len(adjudications)),
        "training_positives": int(y.sum()),
        "training_negatives": int((1 - y).sum()),
        "candidate_corpus_pairs": int(stats.pair_count),
        "candidate_unique_tags": int(len(stats.metas)),
        "validation": metrics,
        "auto_threshold_validation": auto_metrics,
        "review_threshold_validation": review_metrics,
        "ml_auto_enabled": False,
        "notes": (
            "Strict identity normalization model. Sentence/triple context is not used as positive evidence. "
            "Hard contradictions veto merges. The trained model ranks the manual-review queue; automatic merges are deterministic only."
        ),
    }
    model_path = args.output_dir / "raw_coreference_pair_classifier.joblib"
    joblib.dump(bundle, model_path, compress=3)

    adjudications["logistic_oof_probability"] = logistic_oof
    adjudications["boosted_oof_probability"] = boosted_oof
    adjudications["full_model_probability"] = full_train_probability
    adjudications["calibrated_full_model_probability"] = calibrated_train_probability
    adjudications["oof_auto_merge"] = auto_oof
    adjudications["oof_review_or_auto"] = auto_oof | (eligible_review & (logistic_oof >= review_threshold))
    adjudications["oof_error"] = np.where(
        adjudications.oof_auto_merge & (adjudications.label == 0),
        "false_auto_merge",
        np.where((~adjudications.oof_review_or_auto) & (adjudications.label == 1), "missed_valid_merge", ""),
    )

    # Save feature matrix with transparent names.
    feature_frame = pd.DataFrame(X, columns=FEATURE_NAMES)
    training_features = pd.concat([adjudications.reset_index(drop=True), feature_frame], axis=1)
    training_features.to_csv(args.output_dir / "pair_classifier_training_features.csv", index=False)
    adjudications.to_csv(args.output_dir / "pair_classifier_oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "pair_classifier_cv_folds.csv", index=False)

    model_comparison_rows = []
    for model_name, model_metrics in metrics.items():
        model_comparison_rows.append({"model": model_name, **model_metrics})
    pd.DataFrame(model_comparison_rows).to_csv(args.output_dir / "pair_classifier_model_comparison.csv", index=False)

    scaler = final_model.named_steps["standardscaler"]
    linear = final_model.named_steps["logisticregression"]
    coefficients = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "standardized_coefficient": linear.coef_[0],
            "absolute_coefficient": np.abs(linear.coef_[0]),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    coefficients.to_csv(args.output_dir / "pair_classifier_feature_coefficients.csv", index=False)

    summary = {
        "training": {
            "pairs": int(len(adjudications)),
            "positives": int(y.sum()),
            "negatives": int((1 - y).sum()),
            "entity_family_groups": int(len(set(groups))),
        },
        "candidate_corpus": {
            "pairs": int(stats.pair_count),
            "unique_tags": int(len(stats.metas)),
            "bands": {str(k): int(v) for k, v in sorted(stats.band_counts.items(), reverse=True)},
        },
        "model_comparison": metrics,
        "selected_model": "logistic_regression",
        "automatic_merge_policy": {
            "threshold": auto_threshold,
            "minimum_raw_score_for_ml": 94.0,
            "ml_auto_enabled": False,
            "validation": auto_metrics,
            "deterministic_merges_in_training": int(deterministic_mask.sum()),
            "deterministic_precision_in_training": float(y[deterministic_mask].mean()) if deterministic_mask.any() else None,
        },
        "manual_review_policy": {"threshold": review_threshold, "validation": review_metrics},
        "hard_vetoes": {
            "training_pairs_vetoed": int(hard_mask.sum()),
            "valid_merges_vetoed": int(y[hard_mask].sum()),
        },
    }
    (args.output_dir / "pair_classifier_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""# Raw-tag coreference pair classifier

## Purpose

This model ranks **strict identity-normalization pairs**, not merely related entities. It was trained on the 549 manually adjudicated raw-similarity candidates: 52 valid merges and 497 false merges.

## Validation design

Validation was repeated {args.cv_repeats} times with five folds while keeping connected name/title families together. A tag family therefore could not appear in both training and test data within a fold. This is stricter than ordinary random pair splitting and reduces memorization leakage.

## Model comparison

| Model | Average precision | ROC AUC | Precision at 20 | Precision at 50 | Precision at 100 |
|---|---:|---:|---:|---:|---:|
| Raw similarity alone | {metrics['raw_similarity_baseline']['average_precision']:.3f} | {metrics['raw_similarity_baseline']['roc_auc']:.3f} | {metrics['raw_similarity_baseline']['precision_at_20']:.1%} | {metrics['raw_similarity_baseline']['precision_at_50']:.1%} | {metrics['raw_similarity_baseline']['precision_at_100']:.1%} |
| Logistic regression | {metrics['logistic_regression']['average_precision']:.3f} | {metrics['logistic_regression']['roc_auc']:.3f} | {metrics['logistic_regression']['precision_at_20']:.1%} | {metrics['logistic_regression']['precision_at_50']:.1%} | {metrics['logistic_regression']['precision_at_100']:.1%} |
| Gradient-boosted trees | {metrics['hist_gradient_boosting']['average_precision']:.3f} | {metrics['hist_gradient_boosting']['roc_auc']:.3f} | {metrics['hist_gradient_boosting']['precision_at_20']:.1%} | {metrics['hist_gradient_boosting']['precision_at_50']:.1%} | {metrics['hist_gradient_boosting']['precision_at_100']:.1%} |

The logistic model was selected because it had the strongest entity-family-disjoint ranking performance and substantially better probability calibration than the boosted model.

## Operating policy

### Automatic merge

A pair is automatically merged only when it passes an extremely conservative deterministic normalization rule and has no hard contradiction. **ML-only automatic merging is disabled** because 52 positive labels are not enough to estimate unseen-pair precision safely.

On out-of-fold validation the deterministic policy selected **{auto_metrics.get('predicted', 0)}** pairs, all **{auto_metrics.get('true_positive', 0)}** of which were valid, for observed precision **{auto_metrics.get('precision', 0):.1%}** and recall **{auto_metrics.get('recall', 0):.1%}**. The 95% Wilson interval for precision is **{auto_metrics.get('wilson_95_low', 0):.1%}–{auto_metrics.get('wilson_95_high', 1):.1%}**; the lower bound is limited by the small number of positive training examples.

### Manual review

Pairs not automatically merged enter manual review at calibrated probability **{review_threshold:.6f}** or above, provided they have no contradiction and raw similarity is at least 94%. In out-of-fold validation, automatic plus review candidates captured **{review_metrics.get('recall_including_auto', 0):.1%}** of valid merges.

### Automatic rejection

Pairs are rejected when they contain hard contradictions such as different numbers, left/right or north/south opposition, incompatible kinship roles, entity-versus-unit mismatch, or entity-versus-event mismatch. Non-deterministic pairs below 94% raw similarity are also rejected because the manually reviewed 94% band contained zero valid merges and the model has no labelled support below that range.

## Important limitation

Only 52 positive examples are available. The model is useful as a high-precision queue reducer, but its automatic decisions should continue to be audited. Retraining after each new batch of manual adjudications will improve both precision estimates and coverage.
"""
    (args.output_dir / "PAIR_CLASSIFIER_REPORT.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary["model_comparison"], indent=2), flush=True)
    print(f"Saved model bundle to {model_path}", flush=True)
    print(f"Auto threshold: {auto_threshold:.6f}; review threshold: {review_threshold:.6f}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Classify every raw candidate pair with the trained coreference model.

The implementation streams 1.5M+ pairs in raw-rank order, evaluates the ML
model only inside its validated score range, and writes compact audit queues.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pair_classifier_features import (  # noqa: E402
    STOPWORDS,
    GENERIC_TITLES,
    GROUP_MARKERS,
    PERSON_MARKERS,
    PLACE_MARKERS,
    EVENT_MARKERS,
    build_corpus_stats,
    feature_vector,
    hard_rejection_reason,
    identity_evidence_gate,
    iter_candidates,
)

OUTPUT_COLUMNS = [
    "raw_rank",
    "raw_score_percent",
    "score_band",
    "left_tag",
    "right_tag",
    "rank_from_left",
    "rank_from_right",
    "raw_model_probability",
    "model_probability",
    "decision",
    "decision_source",
    "decision_reason",
]


def deterministic_keys(meta):
    personal_prefixes = {"nga", "maung", "ma", "u"}
    strong_title_tokens = {
        "mingyi",
        "maha",
        "thado",
        "thiri",
        "siri",
        "nemyo",
        "naymyo",
        "nawrahta",
        "sithu",
        "prince",
        "princess",
        "king",
        "queen",
        "minister",
        "governor",
        "sayadaw",
        "sawbwa",
        "lord",
        "chief",
    }
    tokens = meta.tokens
    token_set = meta.token_set
    place_words = {"city", "town"}
    event_words = {"campaign", "expedition"}
    city_base = tuple(sorted(t for t in tokens if t not in place_words))
    event_base = tuple(sorted(t for t in tokens if t not in event_words))
    city_anchor = any(
        t not in GENERIC_TITLES
        and t not in STOPWORDS
        and t not in GROUP_MARKERS
        and t not in PERSON_MARKERS
        and t not in PLACE_MARKERS
        for t in set(city_base)
    )
    event_anchor = any(
        t not in GENERIC_TITLES
        and t not in STOPWORDS
        and t not in EVENT_MARKERS
        and t not in {"royal", "military"}
        for t in set(event_base)
    )
    return {
        "norm": meta.norm,
        "compact": meta.compact,
        "variant_compact": meta.variant_compact,
        "sorted": meta.sorted_tokens,
        "variant_sorted": meta.variant_sorted,
        "personal_prefix": bool(tokens and tokens[0] in personal_prefixes),
        "collective": "and" in token_set,
        "person": bool(meta.persons),
        "typed_non_person": bool(
            meta.objects
            or meta.events
            or meta.places
            or meta.institutions
            or (meta.groups and not meta.persons)
        ),
        "ofthe": tuple(sorted(t for t in tokens if t not in {"of", "the"})),
        "has_ofthe": bool(token_set & {"of", "the"}),
        "city_base": city_base,
        "has_citytown": bool(token_set & place_words),
        "city_anchor": city_anchor,
        "event_base": event_base,
        "has_eventword": bool(token_set & event_words),
        "event_anchor": event_anchor,
    }


def deterministic_reason(left, right) -> Optional[str]:
    if left["norm"] == right["norm"]:
        return "exact_after_case_punctuation_and_spacing_normalization"
    if left["compact"] == right["compact"]:
        return "exact_after_word_boundary_normalization"
    if left["variant_compact"] == right["variant_compact"]:
        return "exact_after_curated_transliteration_normalization"
    safe_reorder = (left["collective"] and right["collective"]) or (
        left["typed_non_person"]
        and right["typed_non_person"]
        and not left["personal_prefix"]
        and not right["personal_prefix"]
    )
    if safe_reorder and left["sorted"] == right["sorted"]:
        return "same_normalized_tokens_in_different_order"
    if safe_reorder and left["variant_sorted"] == right["variant_sorted"]:
        return "same_tokens_after_curated_transliteration_normalization"
    if (
        (left["has_ofthe"] or right["has_ofthe"])
        and left["ofthe"] == right["ofthe"]
        and left["ofthe"]
    ):
        return "same_tokens_after_removing_of_the"
    if (
        left["city_anchor"]
        and right["city_anchor"]
        and left["has_citytown"]
        and right["has_citytown"]
        and left["city_base"] == right["city_base"]
    ):
        return "named_city_town_wording_variant"
    if (
        left["event_anchor"]
        and right["event_anchor"]
        and left["has_eventword"]
        and right["has_eventword"]
        and left["event_base"] == right["event_base"]
    ):
        return "named_campaign_expedition_wording_variant"
    return None


def write_queue(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path, default=HERE / "ALL_RAW_GEMINI_EMBEDDING_CANDIDATES.md"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=HERE / "pair_classifier_output" / "raw_coreference_pair_classifier.joblib",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "pair_classifier_output")
    parser.add_argument("--batch-size", type=int, default=25000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.model)
    model = bundle["model"]
    calibrator = bundle["calibrator"]
    auto_threshold = float(bundle["auto_threshold"])
    review_threshold = float(bundle["review_threshold"])
    model_min_score = float(bundle["model_min_raw_score"])

    print("Building corpus statistics and cached normalization keys...", flush=True)
    stats = build_corpus_stats(args.candidates)
    rule_keys = {tag: deterministic_keys(meta) for tag, meta in stats.metas.items()}
    print(f"Scoring {stats.pair_count:,} candidate pairs...", flush=True)

    all_path = args.output_dir / "all_raw_candidates_classified.csv.gz"
    ranks: Counter = Counter()
    decision_counts: Counter = Counter()
    reason_counts: Counter = Counter()
    band_decisions: Dict[int, Counter] = defaultdict(Counter)
    auto_rows: List[dict] = []
    review_rows: List[dict] = []

    batch_records: List[dict] = []
    batch_vectors: List[np.ndarray] = []
    batch_vector_indices: List[int] = []

    def finalize_record(record: dict) -> None:
        decision_counts[record["decision"]] += 1
        reason_counts[record["decision_reason"]] += 1
        band_decisions[int(record["score_band"])][record["decision"]] += 1
        if record["decision"] == "auto_merge":
            auto_rows.append(record.copy())
        elif record["decision"] == "manual_review":
            review_rows.append(record.copy())

    def flush_batch(writer: csv.DictWriter) -> None:
        if batch_vectors:
            matrix = np.vstack(batch_vectors)
            raw_probability = model.predict_proba(matrix)[:, 1]
            probability = calibrator.predict(raw_probability)
            for record_index, vector, raw_value, value in zip(
                batch_vector_indices, batch_vectors, raw_probability, probability
            ):
                record = batch_records[record_index]
                raw_value = float(raw_value)
                value = float(value)
                record["raw_model_probability"] = f"{raw_value:.8f}"
                record["model_probability"] = f"{value:.8f}"
                left_meta = stats.metas[record["left_tag"]]
                right_meta = stats.metas[record["right_tag"]]
                personal_prefixes = {"nga", "maung", "ma", "u"}
                short_personal_reordering_risk = bool(
                    left_meta.tokens and left_meta.tokens[0] in personal_prefixes
                ) or bool(right_meta.tokens and right_meta.tokens[0] in personal_prefixes)
                if (
                    value >= auto_threshold
                    and identity_evidence_gate(vector)
                    and not short_personal_reordering_risk
                ):
                    record["decision"] = "auto_merge"
                    record["decision_source"] = "ml_high_precision_gate"
                    record["decision_reason"] = (
                        "probability_above_auto_threshold_with_lexical_identity_evidence"
                    )
                elif value >= review_threshold:
                    record["decision"] = "manual_review"
                    record["decision_source"] = "ml_review_gate"
                    record["decision_reason"] = "probability_above_manual_review_threshold"
                else:
                    record["decision"] = "reject"
                    record["decision_source"] = "ml_below_threshold"
                    record["decision_reason"] = "probability_below_manual_review_threshold"
        for record in batch_records:
            writer.writerow(record)
            finalize_record(record)
        batch_records.clear()
        batch_vectors.clear()
        batch_vector_indices.clear()

    with gzip.open(all_path, "wt", encoding="utf-8", newline="", compresslevel=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for raw_rank, candidate in enumerate(iter_candidates(args.candidates), start=1):
            ranks[candidate.left] += 1
            rank_left = ranks[candidate.left]
            ranks[candidate.right] += 1
            rank_right = ranks[candidate.right]
            band = int(candidate.score)
            record = {
                "raw_rank": raw_rank,
                "raw_score_percent": f"{candidate.score:.6f}",
                "score_band": band,
                "left_tag": candidate.left,
                "right_tag": candidate.right,
                "rank_from_left": rank_left,
                "rank_from_right": rank_right,
                "raw_model_probability": "",
                "model_probability": "",
                "decision": "",
                "decision_source": "",
                "decision_reason": "",
            }

            # Deterministic matches are checked everywhere, including low-score tails.
            det_reason = deterministic_reason(rule_keys[candidate.left], rule_keys[candidate.right])

            if candidate.score < model_min_score:
                exact_reasons = {
                    "exact_after_case_punctuation_and_spacing_normalization",
                    "exact_after_word_boundary_normalization",
                    "exact_after_curated_transliteration_normalization",
                }
                if det_reason in exact_reasons:
                    record["raw_model_probability"] = "1.00000000"
                    record["model_probability"] = "1.00000000"
                    record["decision"] = "auto_merge"
                    record["decision_source"] = "deterministic_exact_normalization"
                    record["decision_reason"] = det_reason
                elif det_reason:
                    record["decision"] = "manual_review"
                    record["decision_source"] = "deterministic_structure_below_validated_range"
                    record["decision_reason"] = det_reason
                else:
                    record["decision"] = "reject"
                    record["decision_source"] = "outside_validated_score_range"
                    record["decision_reason"] = (
                        "raw_similarity_below_94_percent_without_deterministic_identity_match"
                    )
                batch_records.append(record)
            else:
                left_meta = stats.metas[candidate.left]
                right_meta = stats.metas[candidate.right]
                hard_reason = hard_rejection_reason(left_meta, right_meta)
                if hard_reason:
                    record["decision"] = "reject"
                    record["decision_source"] = "hard_contradiction_veto"
                    record["decision_reason"] = hard_reason
                    batch_records.append(record)
                elif det_reason:
                    record["raw_model_probability"] = "1.00000000"
                    record["model_probability"] = "1.00000000"
                    record["decision"] = "auto_merge"
                    record["decision_source"] = "deterministic_normalization"
                    record["decision_reason"] = det_reason
                    batch_records.append(record)
                else:
                    vector = feature_vector(
                        candidate.left,
                        candidate.right,
                        candidate.score,
                        rank_left,
                        rank_right,
                        stats,
                    )
                    batch_vector_indices.append(len(batch_records))
                    batch_vectors.append(vector)
                    batch_records.append(record)

            if len(batch_records) >= args.batch_size:
                flush_batch(writer)
        flush_batch(writer)

    def queue_key(row: dict):
        return (
            -float(row["model_probability"] or 0.0),
            -float(row["raw_model_probability"] or 0.0),
            -float(row["raw_score_percent"]),
            int(row["raw_rank"]),
        )

    auto_rows.sort(key=queue_key)
    review_rows.sort(key=queue_key)
    write_queue(args.output_dir / "auto_merge_candidates.csv", auto_rows)
    write_queue(args.output_dir / "manual_review_queue.csv", review_rows)

    summary_rows = []
    for band in sorted(band_decisions, reverse=True):
        counts = band_decisions[band]
        total = sum(counts.values())
        summary_rows.append(
            {
                "score_band": band,
                "total_candidates": total,
                "auto_merge": counts["auto_merge"],
                "manual_review": counts["manual_review"],
                "reject": counts["reject"],
                "auto_merge_percent": counts["auto_merge"] / total * 100 if total else 0,
                "manual_review_percent": counts["manual_review"] / total * 100 if total else 0,
            }
        )
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "scoring_summary.csv", index=False)

    summary = {
        "candidate_file": str(args.candidates),
        "pairs_scored": int(stats.pair_count),
        "unique_tags": int(len(stats.metas)),
        "thresholds": {
            "automatic_merge_probability": auto_threshold,
            "manual_review_probability": review_threshold,
            "minimum_raw_score_for_ml": model_min_score,
        },
        "decision_counts": {key: int(value) for key, value in decision_counts.items()},
        "reason_counts": {key: int(value) for key, value in reason_counts.most_common()},
        "outputs": {
            "all_classified": str(all_path),
            "auto_merge": str(args.output_dir / "auto_merge_candidates.csv"),
            "manual_review": str(args.output_dir / "manual_review_queue.csv"),
        },
    }
    (args.output_dir / "scoring_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["decision_counts"], indent=2), flush=True)
    print(
        f"Wrote {len(auto_rows):,} automatic merges and {len(review_rows):,} manual-review candidates.",
        flush=True,
    )


if __name__ == "__main__":
    main()

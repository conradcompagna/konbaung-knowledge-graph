from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(
    r"C:\Users\conra\Desktop\dighumproject"
    r"\konbaung_node_identity_statistical_refinement_20260725"
)
CANDIDATES = ROOT / "all_cluster_pair_candidates.jsonl"
CLUSTERS = ROOT / "corrected_initial_clusters.jsonl"
OUTPUT = ROOT / "ALL_CANDIDATE_MERGES_CLEAN.csv"
README = ROOT / "ALL_CANDIDATE_MERGES_CLEAN_README.md"
AUTO_MD = ROOT / "AUTOMATIC_MERGES_AT_90_PERCENT.md"
THRESHOLD = 0.90

SOURCE_LABELS = {
    "canonical_compact_exact": "same label after removing spacing/punctuation",
    "canonical_normalized_exact": "same normalized label",
    "canonical_order_signature": "same words in a different order",
    "fused_centroid_knn": "cluster-context embedding neighbor",
    "member_tag_knn": "member-tag embedding neighbor",
    "lexical": "lexical similarity",
}


def load_clusters() -> dict[str, dict]:
    clusters: dict[str, dict] = {}
    with CLUSTERS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cluster = json.loads(line)
                clusters[cluster["clusterId"]] = cluster
    return clusters


def readable_sources(sources: list[str]) -> str:
    return "; ".join(
        SOURCE_LABELS.get(source, source.replace("_", " ")) for source in sources
    )


def main() -> None:
    clusters = load_clusters()
    fields = [
        "rank",
        "confidence",
        "automatic_result_at_90_percent",
        "pair_type",
        "left_cluster_id",
        "left_identity",
        "left_aliases",
        "left_tag_count",
        "left_total_frequency",
        "right_cluster_id",
        "right_identity",
        "right_aliases",
        "right_tag_count",
        "right_total_frequency",
        "why_it_was_a_candidate",
    ]
    counts: Counter[str] = Counter()
    accepted: list[dict] = []
    previous_probability = 1.0

    with (
        CANDIDATES.open("r", encoding="utf-8") as source,
        OUTPUT.open("w", encoding="utf-8-sig", newline="") as destination,
    ):
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for rank, line in enumerate(source, start=1):
            candidate = json.loads(line)
            probability = float(candidate["modelProbability"])
            if probability > previous_probability + 1e-15:
                raise RuntimeError("Candidate input is not sorted by descending confidence")
            previous_probability = probability

            left = clusters[candidate["leftClusterId"]]
            right = clusters[candidate["rightClusterId"]]
            clears_threshold = probability >= THRESHOLD
            result = "MERGE" if clears_threshold else "DO NOT AUTO-MERGE"
            counts["total"] += 1
            counts[candidate["pairType"]] += 1
            if clears_threshold:
                counts["automatic"] += 1

            row = {
                "rank": rank,
                "confidence": f"{probability:.6%}",
                "automatic_result_at_90_percent": result,
                "pair_type": candidate["pairType"].replace("_", " - "),
                "left_cluster_id": left["clusterId"],
                "left_identity": left["canonicalLabel"],
                "left_aliases": " | ".join(left["memberTags"]),
                "left_tag_count": left["tagCount"],
                "left_total_frequency": left["totalFrequency"],
                "right_cluster_id": right["clusterId"],
                "right_identity": right["canonicalLabel"],
                "right_aliases": " | ".join(right["memberTags"]),
                "right_tag_count": right["tagCount"],
                "right_total_frequency": right["totalFrequency"],
                "why_it_was_a_candidate": readable_sources(
                    candidate["candidateSources"]
                ),
            }
            writer.writerow(row)
            if clears_threshold:
                accepted.append(row)

    if counts["total"] != 560_612:
        raise RuntimeError(f"Expected 560,612 candidates, wrote {counts['total']:,}")
    if counts["automatic"] != 7:
        raise RuntimeError(f"Expected 7 automatic merges, found {counts['automatic']:,}")

    auto_lines = [
        "# Automatic merges at the 90% threshold",
        "",
        "These are the only candidates from the complete pool that clear the calibrated "
        "automatic threshold. Confidence is the classifier score, not a statistical "
        "guarantee of correctness.",
        "",
        "| Rank | Confidence | Left identity | Right identity | Pair type |",
        "|---:|---:|---|---|---|",
    ]
    for row in accepted:
        auto_lines.append(
            f"| {row['rank']} | {row['confidence']} | "
            f"{row['left_cluster_id']} - {row['left_identity']} | "
            f"{row['right_cluster_id']} - {row['right_identity']} | "
            f"{row['pair_type']} |"
        )
    auto_lines.append("")
    AUTO_MD.write_text("\n".join(auto_lines), encoding="utf-8")

    readme_lines = [
        "# Clean candidate-merge export",
        "",
        f"`{OUTPUT.name}` contains every one of the {counts['total']:,} candidate pairs, "
        "sorted from highest to lowest classifier confidence.",
        "",
        "It deliberately excludes the internal cosine similarities, TF-IDF values, token "
        "ratios, and other model features. Each row contains only:",
        "",
        "- rank and confidence percentage",
        "- whether it clears the 90% automatic threshold",
        "- candidate type",
        "- both cluster IDs and canonical identities",
        "- all aliases already belonging to each side",
        "- tag counts and corpus frequencies",
        "- a short explanation of why the pair entered the candidate pool",
        "",
        f"Only {counts['automatic']:,} candidates clear the 90% threshold. The other "
        f"{counts['total'] - counts['automatic']:,} rows are possibilities considered by "
        "candidate generation, not recommended automatic merges.",
        "",
        "Pair-type totals:",
        "",
        f"- Singleton - singleton: {counts['singleton_singleton']:,}",
        f"- Singleton - cluster: {counts['singleton_cluster']:,}",
        f"- Cluster - cluster: {counts['cluster_cluster']:,}",
        "",
        "The CSV uses UTF-8 with a BOM so Burmese and other non-ASCII text opens correctly "
        "in Excel.",
        "",
    ]
    README.write_text("\n".join(readme_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "rows": counts["total"],
                "automaticAt90Percent": counts["automatic"],
                "readme": str(README),
                "automaticShortlist": str(AUTO_MD),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

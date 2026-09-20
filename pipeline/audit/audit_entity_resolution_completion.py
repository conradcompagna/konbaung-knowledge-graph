"""Independently verify the completed two-phase Konbaung entity-resolution archive."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(r"C:\Users\conra\Desktop\dighumproject")
OUTPUT = ROOT / "konbaung_singleton_completion_gemini_20260902"
BASELINE = ROOT / "konbaung_nonsingleton_top50_frequency_wave_gemini_trial_20260902_first100"


def load_json(path: Path):
    """Load one UTF-8 JSON artifact for cross-file validation."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV artifact while preserving every field as source text."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def usage_totals(clusters: list[dict]) -> dict[str, int]:
    """Sum Gemini usage fields directly from every saved cluster record."""
    prompt = sum(int(row.get("usage", {}).get("prompt_token_count", 0)) for row in clusters)
    answer = sum(int(row.get("usage", {}).get("candidates_token_count", 0)) for row in clusters)
    thinking = sum(int(row.get("usage", {}).get("thoughts_token_count", 0)) for row in clusters)
    total = sum(int(row.get("usage", {}).get("total_token_count", 0)) for row in clusters)
    return {
        "promptTokens": prompt,
        "answerTokens": answer,
        "thinkingTokens": thinking,
        "totalTokens": total,
    }


def count_data_rows(path: Path) -> int:
    """Count candidate rows without loading the thousands of tiny CSVs together."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def main() -> None:
    """Run independent set, cardinality, usage, and materialization checks."""
    manifest = load_json(OUTPUT / "FINAL_ALL_ENTITY_RESOLUTION_MANIFEST.json")
    baseline_manifest = load_json(BASELINE / "full_nonsingleton_run_manifest.json")
    final_audit = load_json(OUTPUT / "FINAL_GLOBAL_EXCLUSION_AUDIT.json")
    rows = load_csv(OUTPUT / "resolved_entities_all.csv")
    all_clusters = load_json(OUTPUT / "resolved_clusters_all.json")
    singleton_clusters = load_json(OUTPUT / "resolved_clusters_singleton_phase.json")
    wave_rows = load_csv(OUTPUT / "SINGLETON_WAVE_SUMMARIES.csv")

    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        """Record every failed invariant so a single run reports all defects."""
        if not condition:
            errors.append(message)

    ids = [row["id"] for row in rows]
    id_set = set(ids)
    entity_by_id = {row["id"]: row["entity"] for row in rows}
    row_by_id = {row["id"]: row for row in rows}
    expected_ids = {f"e{index:05d}" for index in range(manifest["sourceEntityTags"])}
    phase_counts = Counter(row["phase"] for row in rows)
    parent_ids = {row["parent_id"] for row in rows}

    check(len(rows) == 23_890, "resolved entity row count is not 23,890")
    check(len(ids) == len(id_set), "resolved entity IDs are not unique")
    check(id_set == expected_ids, "resolved entity IDs are not the complete e00000-e23889 set")
    check(phase_counts == {"non_singleton": 9_219, "singleton": 14_671}, "phase row counts differ")
    check(parent_ids <= id_set, "one or more parent IDs do not exist in the entity ledger")
    check(
        all(row["parent_entity"] == entity_by_id[row["parent_id"]] for row in rows),
        "one or more parent labels disagree with their parent ID",
    )
    check(
        all(row_by_id[parent_id]["parent_id"] == parent_id for parent_id in parent_ids),
        "one or more cluster parents are not self-assigned",
    )

    all_member_ids = [member_id for cluster in all_clusters for member_id in cluster["memberIds"]]
    all_cluster_parents = [cluster["parentId"] for cluster in all_clusters]
    check(len(all_clusters) == 17_658, "combined cluster/call count is not 17,658")
    check(len(all_cluster_parents) == len(set(all_cluster_parents)), "combined parent IDs repeat")
    check(len(all_member_ids) == len(set(all_member_ids)), "an entity appears in multiple clusters")
    check(set(all_member_ids) == id_set, "combined clusters do not cover the ledger exactly")
    check(set(all_cluster_parents) == parent_ids, "cluster parents and ledger parents differ")
    check(
        all(
            row_by_id[member_id]["parent_id"] == cluster["parentId"]
            for cluster in all_clusters
            for member_id in cluster["memberIds"]
        ),
        "cluster membership disagrees with the resolved ledger",
    )

    singleton_member_ids = [member_id for cluster in singleton_clusters for member_id in cluster["memberIds"]]
    singleton_parent_ids = [cluster["parentId"] for cluster in singleton_clusters]
    singleton_waves = sorted({int(cluster["wave"]) for cluster in singleton_clusters})
    check(len(singleton_clusters) == 12_735, "singleton cluster/call count is not 12,735")
    check(len(singleton_parent_ids) == len(set(singleton_parent_ids)), "singleton parent IDs repeat")
    check(len(singleton_member_ids) == len(set(singleton_member_ids)), "singleton members repeat")
    check(len(singleton_member_ids) == 14_671, "singleton clusters do not cover 14,671 tags")
    check(
        all(row_by_id[member_id]["phase"] == "singleton" for member_id in singleton_member_ids),
        "singleton clusters contain a baseline-phase entity",
    )
    check(all(int(cluster["parentMentions"]) == 1 for cluster in singleton_clusters), "singleton parent mentions differ")
    check(singleton_waves == list(range(96, 156)), "singleton wave numbers are not the complete 96-155 range")

    singleton_usage = usage_totals(singleton_clusters)
    baseline_clusters = all_clusters[: len(all_clusters) - len(singleton_clusters)]
    baseline_usage = usage_totals(baseline_clusters)
    combined_usage = usage_totals(all_clusters)
    check(singleton_usage == manifest["singletonUsage"], "singleton usage does not sum to its manifest")
    check(combined_usage == manifest["combinedUsage"], "combined usage does not sum to its manifest")
    check(baseline_usage == baseline_manifest["usage"], "baseline usage does not sum to its prior manifest")
    check(
        singleton_usage["promptTokens"] + singleton_usage["answerTokens"] + singleton_usage["thinkingTokens"]
        == singleton_usage["totalTokens"],
        "singleton token components do not equal total tokens",
    )

    dropped = sum(len(cluster.get("conformance", {}).get("dropped", [])) for cluster in singleton_clusters)
    check(dropped == manifest["conformanceDroppedStringsSingletonPhase"] == 17, "conformance-drop count differs")

    raw_responses = list((OUTPUT / "waves").rglob("raw_response.json"))
    page_manifests = list((OUTPUT / "waves").rglob("page_manifest.json"))
    results = list((OUTPUT / "waves").rglob("result.json"))
    check(len(raw_responses) == 12_735, "raw response file count is not 12,735")
    check(len(page_manifests) == 12_735, "page manifest file count is not 12,735")
    check(len(results) == 12_735, "result file count is not 12,735")

    wave_numbers = [int(row["wave"]) for row in wave_rows]
    submitted_calls = sum(int(row["submittedCalls"]) for row in wave_rows)
    assigned_entities = sum(int(row["newAssignedEntities"]) for row in wave_rows)
    avoided_calls = sum(int(row["futureCallsAvoided"]) for row in wave_rows)
    check(wave_numbers == list(range(96, 156)), "wave summary rows are not the complete 96-155 range")
    check(submitted_calls == 12_735, "wave summary submitted calls do not total 12,735")
    check(assigned_entities == 14_671, "wave summary assignments do not total 14,671")
    check(avoided_calls == 1_936, "wave summary avoided calls do not total 1,936")
    check(submitted_calls + avoided_calls == 14_671, "submitted plus avoided singleton calls is not 14,671")

    active_files = list((OUTPUT / "active_candidate_lists").glob("*.csv"))
    retired_files = list((OUTPUT / "retired_candidate_lists").glob("*.csv"))
    active_roster_rows = load_csv(OUTPUT / "active_roster.csv")
    retired_candidate_rows = sum(count_data_rows(path) for path in retired_files)
    check(len(active_files) == 0, "active candidate-list files remain")
    check(len(active_roster_rows) == 0, "active roster rows remain")
    check(len(retired_files) == 14_671, "retired candidate-list count is not 14,671")
    check(retired_candidate_rows == 0, "retired candidate-list rows remain after purge")
    check(final_audit["status"] == "passed", "saved final exclusion audit is not passed")

    calculated_singleton_cost = (
        singleton_usage["promptTokens"] * 0.25 / 1_000_000
        + (singleton_usage["answerTokens"] + singleton_usage["thinkingTokens"]) * 1.50 / 1_000_000
    )
    calculated_combined_cost = (
        combined_usage["promptTokens"] * 0.25 / 1_000_000
        + (combined_usage["answerTokens"] + combined_usage["thinkingTokens"]) * 1.50 / 1_000_000
    )
    check(abs(calculated_singleton_cost - manifest["singletonStandardListPriceUsd"]) < 0.000001, "singleton cost differs")
    check(abs(calculated_combined_cost - manifest["combinedStandardListPriceUsd"]) < 0.000001, "combined cost differs")

    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "entityRows": len(rows),
        "phaseRows": dict(phase_counts),
        "combinedClustersAndCalls": len(all_clusters),
        "singletonClustersAndCalls": len(singleton_clusters),
        "uniqueCombinedMemberIds": len(set(all_member_ids)),
        "singletonCallsAvoided": avoided_calls,
        "singletonWaveRange": [singleton_waves[0], singleton_waves[-1]],
        "singletonWaveCount": len(singleton_waves),
        "rawResponseFiles": len(raw_responses),
        "pageManifestFiles": len(page_manifests),
        "resultFiles": len(results),
        "activeCandidateFiles": len(active_files),
        "retiredCandidateFiles": len(retired_files),
        "retiredCandidateRows": retired_candidate_rows,
        "singletonUsage": singleton_usage,
        "combinedUsage": combined_usage,
        "singletonStandardListPriceUsd": round(calculated_singleton_cost, 6),
        "combinedStandardListPriceUsd": round(calculated_combined_cost, 6),
        "conformanceDroppedStringsSingletonPhase": dropped,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

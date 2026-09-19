#!/usr/bin/env python3
"""Resolve every remaining singleton with a resumable, globally pruned Gemini cascade."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import continue_konbaung_top50_frequency_waves as engine
import run_konbaung_nonsingleton_top50_wave_trial as base
from konbaung_gemini_summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "konbaung_entity_resolution_candidates_20260902_v4_nonsingleton_top50"
PRIOR = ROOT / "konbaung_nonsingleton_top50_frequency_wave_gemini_trial_20260902_first100"
OUTPUT = ROOT / "konbaung_singleton_completion_gemini_20260902"
ACTIVE_LISTS = "active_candidate_lists"
RETIRED_LISTS = "retired_candidate_lists"
STATE_FILE = "singleton_state.json"
LOCATION_DB = "singleton_candidate_locations.sqlite"
MAX_WORKERS = 100
FIRST_SINGLETON_WAVE = 96


# Require explicit paid execution while allowing initialization and inspection for free.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run or resume paid singleton waves until every entity is assigned.",
    )
    return parser.parse_args()


# Read a CSV without changing its canonical order.
def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# Write the singleton roster with the true top-20 source-page size.
def write_singleton_roster(
    root: Path,
    eligible: list[dict[str, Any]],
    assigned_ids: set[str],
) -> None:
    path = root / "active_roster.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("id", "entity", "mentions", "source_candidates", "file"))
        for row in eligible:
            if row["id"] not in assigned_ids:
                writer.writerow(
                    (row["id"], row["entity"], row["mentions"], row["candidates"], row["file"])
                )
    os.replace(temporary, path)


# Point the audited cascade engine at the separate singleton workspace for this process.
def configure_engine() -> None:
    engine.OUTPUT = OUTPUT
    engine.ACTIVE_LISTS = ACTIVE_LISTS
    engine.RETIRED_LISTS = RETIRED_LISTS
    engine.STATE_FILE = STATE_FILE
    engine.LOCATION_DB = LOCATION_DB
    engine.TARGET_COMPLETED_WAVES = 1_000_000
    base.write_active_roster = write_singleton_roster


# Load and verify the immutable completed non-singleton ledger used as the baseline.
def load_baseline() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(
        (PRIOR / "full_nonsingleton_run_manifest.json").read_text(encoding="utf-8")
    )
    state = json.loads((PRIOR / "ten_wave_state.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_non_singleton_pass":
        raise RuntimeError("The non-singleton manifest is not complete.")
    if state.get("status") != "completed_non_singleton_pass":
        raise RuntimeError("The non-singleton state is not complete.")
    if len(state["assignments"]) != 9219 or manifest["remainingSingletonEntities"] != 14671:
        raise RuntimeError("The non-singleton baseline has unexpected counts.")
    return manifest, state


# Atomically materialize only unresolved singleton pages after baseline tags are removed.
def initialize_output(
    all_rows: list[dict[str, Any]],
    baseline_manifest: dict[str, Any],
    baseline_state: dict[str, Any],
) -> None:
    if OUTPUT.exists():
        if not (OUTPUT / STATE_FILE).is_file():
            raise RuntimeError(f"Output exists without a recoverable state: {OUTPUT}")
        return
    index_by_id = {row["id"]: row for row in all_rows}
    remaining_roster = read_csv(
        PRIOR / "remaining_singletons_after_nonsingleton_completion.csv"
    )
    eligible = [index_by_id[row["id"]] for row in remaining_roster]
    baseline_assignments = baseline_state["assignments"]
    baseline_entities = {
        assignment["entity"] for assignment in baseline_assignments.values()
    }
    if len(eligible) != 14671 or any(row["mentions"] != 1 for row in eligible):
        raise RuntimeError("Expected exactly 14,671 remaining singleton parents.")
    if set(index_by_id) != set(baseline_assignments) | {row["id"] for row in eligible}:
        raise RuntimeError("Baseline assignments and singleton roster do not partition the index.")

    staging = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.staging_", dir=ROOT))
    try:
        active = staging / ACTIVE_LISTS
        retired = staging / RETIRED_LISTS
        active.mkdir()
        retired.mkdir()
        retained_rows = 0
        removed_rows = 0
        for number, parent in enumerate(eligible, start=1):
            source_rows = base.read_candidate_csv(SOURCE / "entities" / parent["file"])
            if len(source_rows) != 20:
                raise RuntimeError(f"Singleton source page is not top 20: {parent['file']}")
            retained = [
                row for row in source_rows if row["entity"] not in baseline_entities
            ]
            retained_rows += len(retained)
            removed_rows += len(source_rows) - len(retained)
            base.write_candidate_csv(active / parent["file"], retained)
            if number % 1000 == 0:
                print(f"materialized singleton pages: {number}/{len(eligible)}", flush=True)
        write_singleton_roster(staging, eligible, set(baseline_assignments))
        state = {
            "schemaVersion": 1,
            "status": "ready",
            "model": engine.MODEL,
            "thinkingLevel": "minimal",
            "includeThoughts": False,
            "completedWaves": FIRST_SINGLETON_WAVE - 1,
            "firstSingletonWave": FIRST_SINGLETON_WAVE,
            "pendingWave": None,
            "initialEligibleParents": len(eligible),
            "baselineAssignedEntities": len(baseline_assignments),
            "assignments": baseline_assignments,
            "clusters": [],
            "waveSummaries": [],
        }
        base.write_json(staging / STATE_FILE, state)
        base.write_json(
            staging / "plan_manifest.json",
            {
                "status": "ready_not_submitted",
                "apiCallsMade": 0,
                "model": engine.MODEL,
                "thinkingLevel": "minimal",
                "includeThoughts": False,
                "baselineAssignedEntities": len(baseline_assignments),
                "remainingSingletonParents": len(eligible),
                "sourceCandidatesPerPage": 20,
                "candidateRowsBeforeBaselinePurge": len(eligible) * 20,
                "candidateRowsRemovedByBaseline": removed_rows,
                "candidateRowsAfterBaselinePurge": retained_rows,
                "firstSingletonWave": FIRST_SINGLETON_WAVE,
                "workerCap": MAX_WORKERS,
                "priorManifest": str(
                    PRIOR / "full_nonsingleton_run_manifest.json"
                ),
                "priorUsage": baseline_manifest["usage"],
            },
        )
        (staging / "README.md").write_text(
            "# Remaining-singleton Gemini completion\n\n"
            "This separate cascade begins from the completed 95-wave non-singleton "
            "assignment ledger. It materializes only the 14,671 unresolved singleton "
            "pages, removes all 9,219 prior assignments before submission, and then "
            "uses strict pairwise-disjoint waves with global pruning after each fully "
            "validated wave. The immutable v4 candidate source and completed prior "
            "archive are unchanged.\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(staging, OUTPUT)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


# Submit a whole disjoint wave with the proven 100-request concurrency cap.
def submit_wave_capped(
    wave: int,
    wave_directory: Path,
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(selected))) as executor:
        futures = {
            executor.submit(engine.submit_or_resume_page, wave, wave_directory, parent): parent
            for parent in selected
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            parent = futures[future]
            try:
                result = future.result()
                results.append(result)
                outcome = f"yes={len(result['yes'])}"
            except Exception as error:
                failures.append(
                    {"id": parent["id"], "entity": parent["entity"], "error": repr(error)}
                )
                outcome = f"FAILED: {error}"
            base.write_json(
                wave_directory / "submission_progress.json",
                {
                    "completedWorkers": completed,
                    "validatedResponses": len(results),
                    "failures": failures,
                },
            )
            print(
                f"wave {wave} response {completed}/{len(selected)}: "
                f"{parent['entity']} - {outcome}",
                flush=True,
            )
    if failures:
        raise RuntimeError(
            f"Wave {wave} has {len(failures)} failures; no decisions were applied. "
            "Rerun to resume saved responses."
        )
    return sorted(results, key=lambda row: row["waveOrder"])


# Replace the generic wave review with singleton-specific pruning terminology.
def write_singleton_wave_review(
    wave_directory: Path,
    clusters: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    lines = [
        f"# Singleton wave {summary['wave']} review",
        "",
        f"- Calls: {summary['submittedCalls']}",
        f"- Singleton entities assigned: {summary['newAssignedEligibleParents']}",
        f"- Future singleton calls avoided: {summary['futureCallsAvoided']}",
        f"- Positive clusters: {summary['multiEntityClusters']}",
        f"- Remaining singleton parents: {summary['remainingEligibleParents']}",
        f"- Global exclusion audit: {summary['audit']['status']}",
        f"- Total tokens: {summary['usage']['totalTokens']}",
        "",
        "## Decisions",
        "",
    ]
    for cluster in clusters:
        yes = " | ".join(cluster["members"][1:]) or "none"
        lines.extend(
            [
                f"### {cluster['waveOrder']}. {cluster['parentEntity']} (1 mention)",
                "",
                f"- YES: {yes}",
                f"- NO: {len(cluster['different'])} omitted supplied candidates",
                f"- Candidates sent: {cluster['activeCandidateCount']}",
                "",
            ]
        )
    (wave_directory / "REVIEW.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


# Sum exact usage metadata across committed singleton waves.
def summed_usage(summaries: list[dict[str, Any]]) -> dict[str, int]:
    return {
        key: sum(int(summary["usage"].get(key) or 0) for summary in summaries)
        for key in ("promptTokens", "answerTokens", "thinkingTokens", "totalTokens")
    }


# Escape field separators so entity strings remain readable in compact Markdown lists.
def markdown_value(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


# Write only positive singleton decisions into one compact aggregate review.
def write_positive_singleton_review(
    clusters: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    positive = [cluster for cluster in clusters if len(cluster["memberIds"]) > 1]
    aliases = sum(len(cluster["memberIds"]) - 1 for cluster in positive)
    lines = [
        "# Master review: positive singleton resolutions",
        "",
        "Only singleton calls where Gemini accepted at least one supplied alias are included.",
        "",
        f"- Singleton calls: {manifest['singletonCalls']}",
        f"- Positive resolution pages included: {len(positive)}",
        f"- No-match pages excluded: {manifest['singletonCalls'] - len(positive)}",
        f"- Accepted aliases: {aliases}",
        "",
    ]
    current_wave = None
    number = 0
    for cluster in positive:
        wave = int(cluster["wave"])
        if wave != current_wave:
            current_wave = wave
            lines.extend([f"## Wave {wave}", ""])
        number += 1
        yes = " | ".join(markdown_value(value) for value in cluster["members"][1:])
        lines.extend(
            [
                f"### {number}. {markdown_value(cluster['parentEntity'])} (1 mention)",
                "",
                f"- YES: {yes}",
                f"- NO: {len(cluster['different'])} omitted supplied candidates",
                f"- Candidates sent: {cluster['activeCandidateCount']}",
                "",
            ]
        )
    (OUTPUT / "MASTER_POSITIVE_SINGLETON_RESOLUTIONS.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


# Export the complete 23,890-tag ledger and prove both phases form one exact partition.
def finalize(
    state: dict[str, Any],
    all_rows: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
    baseline_manifest: dict[str, Any],
    baseline_state: dict[str, Any],
) -> dict[str, Any]:
    assignments = state["assignments"]
    if set(assignments) != {row["id"] for row in all_rows}:
        raise RuntimeError("Final assignments do not cover the complete entity index.")
    audit = engine.audit_global_exclusion(assignments, eligible)
    if audit["activePages"] != 0 or audit["workingCandidateRowsAfterPurge"] != 0:
        raise RuntimeError("Completed singleton pages are not fully empty and retired.")

    baseline_ids = set(baseline_state["assignments"])
    singleton_clusters = state["clusters"]
    baseline_clusters = json.loads(
        (PRIOR / "resolved_clusters_after_nonsingleton_completion.json").read_text(
            encoding="utf-8"
        )
    )
    combined_clusters = [*baseline_clusters, *singleton_clusters]
    member_ids = [
        entity_id for cluster in combined_clusters for entity_id in cluster["memberIds"]
    ]
    if len(member_ids) != len(set(member_ids)) or set(member_ids) != set(assignments):
        raise RuntimeError("Combined clusters are not a unique cover of final assignments.")

    singleton_usage = summed_usage(state["waveSummaries"])
    combined_usage = {
        key: int(baseline_manifest["usage"][key]) + singleton_usage[key]
        for key in singleton_usage
    }
    singleton_calls = sum(
        int(summary["submittedCalls"]) for summary in state["waveSummaries"]
    )
    singleton_positive = sum(
        int(summary["multiEntityClusters"]) for summary in state["waveSummaries"]
    )
    singleton_aliases = len(eligible) - singleton_calls
    total_calls = int(baseline_manifest["totalCalls"]) + singleton_calls

    with (OUTPUT / "resolved_entities_all.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "id",
                "entity",
                "mentions",
                "parent_id",
                "parent_entity",
                "wave",
                "wave_order",
                "phase",
            )
        )
        for row in all_rows:
            assignment = assignments[row["id"]]
            writer.writerow(
                (
                    row["id"],
                    row["entity"],
                    row["mentions"],
                    assignment["parentId"],
                    assignment["parentEntity"],
                    assignment["wave"],
                    assignment["waveOrder"],
                    "non_singleton" if row["id"] in baseline_ids else "singleton",
                )
            )
    base.write_json(OUTPUT / "resolved_clusters_all.json", combined_clusters)
    base.write_json(OUTPUT / "resolved_clusters_singleton_phase.json", singleton_clusters)
    base.write_json(OUTPUT / "FINAL_GLOBAL_EXCLUSION_AUDIT.json", audit)

    summary_fields = [
        "wave",
        "submittedCalls",
        "newAssignedEntities",
        "newAssignedEligibleParents",
        "futureCallsAvoided",
        "multiEntityClusters",
        "candidateRowsRemoved",
        "remainingEligibleParents",
    ]
    with (OUTPUT / "SINGLETON_WAVE_SUMMARIES.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [
                {field: summary.get(field, 0) for field in summary_fields}
                for summary in state["waveSummaries"]
            ]
        )

    manifest = {
        "status": "completed_all_entities",
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "model": engine.MODEL,
        "thinkingLevel": "minimal",
        "includeThoughts": False,
        "sourceEntityTags": len(all_rows),
        "baselineAssignedTags": len(baseline_ids),
        "singletonParentsAtStart": len(eligible),
        "singletonCalls": singleton_calls,
        "singletonCallsAvoided": singleton_aliases,
        "singletonPositiveClusters": singleton_positive,
        "singletonNoMatchCalls": singleton_calls - singleton_positive,
        "singletonPhaseWaves": len(state["waveSummaries"]),
        "lastCombinedWave": state["completedWaves"],
        "totalCallsBothPhases": total_calls,
        "totalCallsAvoidedBothPhases": len(all_rows) - total_calls,
        "totalAssignedEntityTags": len(assignments),
        "singletonUsage": singleton_usage,
        "combinedUsage": combined_usage,
        "singletonStandardListPriceUsd": round(
            singleton_usage["promptTokens"] * 0.25 / 1_000_000
            + singleton_usage["answerTokens"] * 1.50 / 1_000_000,
            6,
        ),
        "combinedStandardListPriceUsd": round(
            combined_usage["promptTokens"] * 0.25 / 1_000_000
            + combined_usage["answerTokens"] * 1.50 / 1_000_000,
            6,
        ),
        "conformanceDroppedStringsSingletonPhase": sum(
            int(summary.get("conformanceDroppedStrings", 0))
            for summary in state["waveSummaries"]
        ),
        "finalGlobalExclusionAudit": audit,
    }
    base.write_json(OUTPUT / "FINAL_ALL_ENTITY_RESOLUTION_MANIFEST.json", manifest)
    write_positive_singleton_review(singleton_clusters, manifest)

    lines = [
        "# Completed singleton entity-resolution pass",
        "",
        f"- Singleton parents at start: {manifest['singletonParentsAtStart']}",
        f"- Singleton Gemini calls: {manifest['singletonCalls']}",
        f"- Future singleton calls avoided: {manifest['singletonCallsAvoided']}",
        f"- Positive singleton clusters: {manifest['singletonPositiveClusters']}",
        f"- No-match singleton calls: {manifest['singletonNoMatchCalls']}",
        f"- Singleton waves: {manifest['singletonPhaseWaves']}",
        f"- Singleton tokens: {singleton_usage['totalTokens']}",
        f"- Singleton standard list-price cost: ${manifest['singletonStandardListPriceUsd']:.2f}",
        f"- All entity tags assigned across both phases: {manifest['totalAssignedEntityTags']}",
        f"- Combined calls: {manifest['totalCallsBothPhases']}",
        f"- Combined tokens: {combined_usage['totalTokens']}",
        f"- Final exclusion audit: {audit['status']}",
        "",
        "## Singleton wave summaries",
        "",
        "| Wave | Calls | Assigned | Calls avoided | Positive | Remaining | Tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in state["waveSummaries"]:
        lines.append(
            f"| {summary['wave']} | {summary['submittedCalls']} | "
            f"{summary['newAssignedEligibleParents']} | {summary['futureCallsAvoided']} | "
            f"{summary['multiEntityClusters']} | {summary['remainingEligibleParents']} | "
            f"{summary['usage']['totalTokens']} |"
        )
    lines.extend(
        [
            "",
            "Use `MASTER_POSITIVE_SINGLETON_RESOLUTIONS.md` for only accepted merges. "
            "Exact decisions and prompts remain under `waves/wave_NNN/`.",
        ]
    )
    (OUTPUT / "FULL_SINGLETON_COMPLETION_REVIEW.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    return manifest


# Initialize, resume, execute, and finalize the authorized singleton cascade.
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args()
    configure_engine()
    all_rows = base.read_index()
    baseline_manifest, baseline_state = load_baseline()
    initialize_output(all_rows, baseline_manifest, baseline_state)
    eligible_ids = {
        row["id"]
        for row in read_csv(
            PRIOR / "remaining_singletons_after_nonsingleton_completion.csv"
        )
    }
    eligible = [row for row in all_rows if row["id"] in eligible_ids]
    index_by_id = {row["id"]: row for row in all_rows}
    index_by_entity = {row["entity"]: row for row in all_rows}
    engine.initialize_location_database()
    state = engine.resume_pending_if_needed(
        engine.read_state(), eligible, index_by_id
    )
    initial_audit = engine.audit_global_exclusion(state["assignments"], eligible)
    if not args.execute:
        selected, skipped = engine.select_wave(engine.read_active_roster())
        base.write_json(
            OUTPUT / "INITIAL_GLOBAL_EXCLUSION_AUDIT.json", initial_audit
        )
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "remainingSingletonParents": len(engine.read_active_roster()),
                    "firstWave": state["completedWaves"] + 1,
                    "firstWaveSelectedPages": len(selected),
                    "firstWaveOverlapSkips": len(skipped),
                    "initialAudit": initial_audit,
                    "apiCallsMadeThisInvocation": 0,
                },
                indent=2,
            )
        )
        return
    if not api_key():
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    if state.get("status") == "completed_all_entities":
        print(
            (OUTPUT / "FINAL_ALL_ENTITY_RESOLUTION_MANIFEST.json").read_text(
                encoding="utf-8"
            ),
            flush=True,
        )
        return

    while engine.read_active_roster():
        wave = int(state["completedWaves"]) + 1
        wave_directory, selected = engine.load_or_create_selection(wave)
        results = submit_wave_capped(wave, wave_directory, selected)
        pending = engine.build_pending(
            wave, results, index_by_entity, state["assignments"]
        )
        base.write_json(wave_directory / "pending_commit.json", pending)
        state["pendingWave"] = wave
        state["status"] = "responses_validated_pending_commit"
        engine.write_state(state)
        summary = engine.complete_pending(
            state, pending, eligible, index_by_id
        )
        write_singleton_wave_review(wave_directory, pending["clusters"], summary)
        print(
            f"wave {wave} committed: calls={summary['submittedCalls']}, "
            f"assigned={summary['newAssignedEligibleParents']}, "
            f"future calls avoided={summary['futureCallsAvoided']}, "
            f"remaining={summary['remainingEligibleParents']}",
            flush=True,
        )
        state = engine.read_state()

    manifest = finalize(
        state, all_rows, eligible, baseline_manifest, baseline_state
    )
    state["status"] = "completed_all_entities"
    state["pendingWave"] = None
    engine.write_state(state)
    plan = json.loads((OUTPUT / "plan_manifest.json").read_text(encoding="utf-8"))
    base.write_json(
        OUTPUT / "plan_manifest.json",
        {**plan, "status": "completed", "apiCallsMade": manifest["singletonCalls"]},
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

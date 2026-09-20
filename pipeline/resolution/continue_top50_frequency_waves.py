#!/usr/bin/env python3
"""Continue the audited non-singleton top-50 run through ten complete waves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google.genai import types

import run_konbaung_nonsingleton_top50_wave_trial as base
from konbaung_gemini_summary_claim_completion_annotator import api_key
from run_konbaung_binary_resolution_production import (
    BinaryDecision,
    binary_prompt,
    conform_binary_decision,
    validate_binary_decision,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = base.OUTPUT
ACTIVE_LISTS = base.ACTIVE_LISTS
RETIRED_LISTS = base.RETIRED_LISTS
MODEL = base.MODEL
TARGET_COMPLETED_WAVES = 10
STATE_FILE = "ten_wave_state.json"
LOCATION_DB = "ten_wave_candidate_locations.sqlite"
MAX_OUTPUT_TOKENS = 1600
RETRY_DELAYS = (2, 4, 8, 16, 30, 30)


def parse_args() -> argparse.Namespace:
    """Require explicit authorization before starting or resuming paid waves."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run or resume waves two through ten.",
    )
    parser.add_argument(
        "--until-empty",
        action="store_true",
        help="Continue beyond wave ten until every non-singleton parent is assigned.",
    )
    return parser.parse_args()


def read_state() -> dict[str, Any]:
    """Load the ten-wave ledger that supersedes the frozen first-wave ledger."""
    return json.loads((OUTPUT / STATE_FILE).read_text(encoding="utf-8"))


def write_state(state: dict[str, Any]) -> None:
    """Checkpoint the canonical continuation state atomically after each transition."""
    base.write_json(OUTPUT / STATE_FILE, state)


def read_active_roster() -> list[dict[str, Any]]:
    """Load unresolved non-singleton parents in their preserved frequency order."""
    with (OUTPUT / "active_roster.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for position, row in enumerate(rows, start=1):
        parsed.append(
            {
                **row,
                "position": position,
                "mentions": int(row["mentions"]),
                "source_candidates": int(row["source_candidates"]),
            }
        )
    if not all(
        left["mentions"] >= right["mentions"]
        for left, right in zip(parsed, parsed[1:])
    ):
        raise RuntimeError("Active roster is no longer frequency-descending")
    return parsed


def initialize_location_database() -> None:
    """Index candidate-to-file locations once so later global purges touch only targets."""
    path = OUTPUT / LOCATION_DB
    if path.is_file():
        return
    temporary = path.with_suffix(".sqlite.tmp")
    if temporary.exists():
        temporary.unlink()
    database = sqlite3.connect(temporary)
    try:
        database.execute(
            "CREATE TABLE locations (candidate TEXT NOT NULL, file TEXT NOT NULL, "
            "PRIMARY KEY (candidate, file)) WITHOUT ROWID"
        )
        pages = [
            *sorted((OUTPUT / ACTIVE_LISTS).glob("*.csv")),
            *sorted((OUTPUT / RETIRED_LISTS).glob("*.csv")),
        ]
        for number, page in enumerate(pages, start=1):
            rows = base.read_candidate_csv(page)
            database.executemany(
                "INSERT INTO locations(candidate, file) VALUES (?, ?)",
                ((row["entity"], page.name) for row in rows),
            )
            if number % 1000 == 0:
                database.commit()
                print(f"indexed current candidate locations: {number}/{len(pages)}", flush=True)
        database.execute("CREATE INDEX locations_candidate ON locations(candidate)")
        database.commit()
    finally:
        database.close()
    os.replace(temporary, path)


def first_wave_summary() -> dict[str, Any]:
    """Import exact wave-one totals so the cumulative ledger starts from audited data."""
    manifest = json.loads((OUTPUT / "run_manifest.json").read_text(encoding="utf-8"))
    return {
        "wave": 1,
        "submittedCalls": manifest["submittedPages"],
        "newAssignedEntities": manifest["assignedEntities"],
        "newAssignedEligibleParents": manifest["assignedEligibleParents"],
        "futureCallsAvoided": (
            manifest["assignedEligibleParents"] - manifest["submittedPages"]
        ),
        "singletonAliases": (
            manifest["assignedEntities"] - manifest["assignedEligibleParents"]
        ),
        "multiEntityClusters": manifest["multiEntityClusters"],
        "conformanceDroppedStrings": manifest["conformanceDroppedStrings"],
        "candidateRowsRemoved": manifest["mutation"][
            "candidateRowsRemovedThisRecoveryPass"
        ],
        "remainingEligibleParents": manifest["remainingEligibleParents"],
        "usage": manifest["usage"],
        "audit": manifest["globalExclusionAudit"],
    }


def initialize_continuation(all_rows: list[dict[str, Any]]) -> None:
    """Create a continuation ledger from the completed and audited first-wave state."""
    state_path = OUTPUT / STATE_FILE
    if state_path.is_file():
        return
    first_state = json.loads((OUTPUT / "state.json").read_text(encoding="utf-8"))
    if first_state.get("status") != "completed":
        raise RuntimeError("The first-wave checkpoint is not complete")
    if len(first_state["clusters"]) != 92 or len(first_state["assignments"]) != 299:
        raise RuntimeError("The first-wave checkpoint has unexpected counts")
    clusters: list[dict[str, Any]] = []
    for cluster in first_state["clusters"]:
        clusters.append({"wave": 1, **cluster})
    assignments = {
        entity_id: {"wave": 1, **assignment}
        for entity_id, assignment in first_state["assignments"].items()
    }
    state = {
        "schemaVersion": 2,
        "status": "ready",
        "model": MODEL,
        "thinkingLevel": "minimal",
        "includeThoughts": False,
        "targetCompletedWaves": TARGET_COMPLETED_WAVES,
        "completedWaves": 1,
        "pendingWave": None,
        "initialEligibleParents": 6498,
        "excludedSingletonParents": len(all_rows) - 6498,
        "assignments": assignments,
        "clusters": clusters,
        "waveSummaries": [first_wave_summary()],
    }
    write_state(state)


def select_wave(roster: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Greedily pack a maximal wave from cleaned footprints in strict frequency order."""
    occupied_by: dict[str, dict[str, Any]] = {}
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for scan_order, parent in enumerate(roster, start=1):
        path = OUTPUT / ACTIVE_LISTS / parent["file"]
        if not path.is_file():
            raise RuntimeError(f"Roster parent has no active page: {parent['entity']}")
        candidates = base.read_candidate_csv(path)
        footprint = [parent["entity"], *[row["entity"] for row in candidates]]
        if len(footprint) != len(set(footprint)):
            raise RuntimeError(f"Duplicate tag in active footprint: {parent['file']}")
        conflicts = [tag for tag in footprint if tag in occupied_by]
        if conflicts:
            first = conflicts[0]
            owner = occupied_by[first]
            skipped.append(
                {
                    "scan_order": scan_order,
                    "id": parent["id"],
                    "entity": parent["entity"],
                    "mentions": parent["mentions"],
                    "active_candidates": len(candidates),
                    "first_conflict": first,
                    "conflicting_wave_order": owner["wave_order"],
                    "conflicting_parent": owner["entity"],
                    "conflict_count": len(conflicts),
                }
            )
            continue
        record = {
            **parent,
            "scan_order": scan_order,
            "wave_order": len(selected) + 1,
            "active_candidates": len(candidates),
            "footprint_sha256": hashlib.sha256(
                "\n".join(footprint).encode("utf-8")
            ).hexdigest(),
        }
        selected.append(record)
        for tag in footprint:
            occupied_by[tag] = record
    if not selected:
        raise RuntimeError("No page could be selected from the active roster")
    return selected, skipped


def write_wave_selection(
    wave_directory: Path,
    selected: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> None:
    """Freeze the entire wave and its conflict proof before making any paid request."""
    selection_fields = [
        "wave_order",
        "scan_order",
        "position",
        "id",
        "entity",
        "mentions",
        "source_candidates",
        "active_candidates",
        "file",
        "footprint_sha256",
    ]
    with (wave_directory / "selection.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=selection_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [{field: row[field] for field in selection_fields} for row in selected]
        )
    skip_fields = [
        "scan_order",
        "id",
        "entity",
        "mentions",
        "active_candidates",
        "first_conflict",
        "conflicting_wave_order",
        "conflicting_parent",
        "conflict_count",
    ]
    with (wave_directory / "conflict_skips.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=skip_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(skipped)
    base.write_json(
        wave_directory / "selection_manifest.json",
        {
            "status": "frozen_not_submitted",
            "selectedPages": len(selected),
            "skippedForOverlap": len(skipped),
            "activeRosterPages": len(selected) + len(skipped),
            "packing": (
                "Strict mention-frequency-descending greedy maximal packing of current "
                "cleaned {parent + remaining candidates} footprints."
            ),
            "pairwiseOverlapCount": 0,
        },
    )


def load_or_create_selection(wave: int) -> tuple[Path, list[dict[str, Any]]]:
    """Create one wave once or reload its frozen page order during a safe resume."""
    wave_directory = OUTPUT / "waves" / f"wave_{wave:03d}"
    wave_directory.mkdir(parents=True, exist_ok=True)
    selection_path = wave_directory / "selection.csv"
    if not selection_path.is_file():
        selected, skipped = select_wave(read_active_roster())
        write_wave_selection(wave_directory, selected, skipped)
        print(
            f"wave {wave}: selected {len(selected)} pages; "
            f"skipped {len(skipped)} overlaps",
            flush=True,
        )
        return wave_directory, selected
    with selection_path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    selected = [
        {
            **row,
            "wave_order": int(row["wave_order"]),
            "scan_order": int(row["scan_order"]),
            "position": int(row["position"]),
            "mentions": int(row["mentions"]),
            "source_candidates": int(row["source_candidates"]),
            "active_candidates": int(row["active_candidates"]),
        }
        for row in raw_rows
    ]
    if not selected:
        raise RuntimeError(f"Frozen wave {wave} has no pages")
    return wave_directory, selected


def response_directory(wave_directory: Path, parent: dict[str, Any]) -> Path:
    """Give each wave page a stable, readable, and collision-free response folder."""
    return wave_directory / "responses" / (
        f"page_{parent['wave_order']:03d}_{parent['id']}_{base.safe_name(parent['entity'])}"
    )


def submit_or_resume_page(
    wave: int,
    wave_directory: Path,
    parent: dict[str, Any],
) -> dict[str, Any]:
    """Submit one frozen page once or recover and validate its raw response on resume."""
    directory = response_directory(wave_directory, parent)
    directory.mkdir(parents=True, exist_ok=True)
    candidates = base.read_candidate_csv(OUTPUT / ACTIVE_LISTS / parent["file"])
    if len(candidates) != parent["active_candidates"]:
        raise RuntimeError(
            f"Active candidate count changed before wave {wave} commit: {parent['id']}"
        )
    prompt = binary_prompt(parent["entity"], candidates)
    prompt_path = directory / "prompt_sent.txt"
    if prompt_path.is_file():
        if prompt_path.read_text(encoding="utf-8") != prompt:
            raise RuntimeError(f"Saved prompt differs: wave {wave}, {parent['id']}")
    else:
        prompt_path.write_text(prompt, encoding="utf-8", newline="\n")

    raw_path = directory / "raw_response.json"
    if raw_path.is_file():
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        client = base.get_client()
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                response = client.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=BinaryDecision.model_json_schema(),
                        temperature=0.0,
                        candidate_count=1,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        thinking_config=types.ThinkingConfig(
                            thinking_level=types.ThinkingLevel.MINIMAL,
                            include_thoughts=False,
                        ),
                    ),
                )
                raw = response.model_dump(mode="json", exclude_none=True)
                base.write_json(raw_path, raw)
                break
            except Exception as error:
                if attempt >= len(RETRY_DELAYS) or not base.is_transient(error):
                    raise
                time.sleep(RETRY_DELAYS[attempt] + random.random())
        else:
            raise RuntimeError("API retry loop ended unexpectedly")

    answer = base.answer_from_raw(raw)
    (directory / "answer_returned.json").write_text(
        answer + "\n", encoding="utf-8", newline="\n"
    )
    raw_decision = BinaryDecision.model_validate_json(answer)
    base.write_json(directory / "raw_decision.json", raw_decision.model_dump(mode="json"))
    decision, conformance = conform_binary_decision(
        raw_decision, parent["entity"], candidates
    )
    validate_binary_decision(decision, parent["entity"], candidates)
    base.write_json(directory / "conformance.json", conformance)
    base.write_json(directory / "result.json", decision.model_dump(mode="json"))
    usage = raw.get("usage_metadata") or {}
    base.write_json(
        directory / "page_manifest.json",
        {
            "wave": wave,
            "waveOrder": parent["wave_order"],
            "parent": parent,
            "activeCandidatesSent": len(candidates),
            "promptCharacters": len(prompt),
            "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "thinkingLevel": "minimal",
            "includeThoughts": False,
            "usage": usage,
            "conformance": conformance,
        },
    )
    return {
        "wave": wave,
        "waveOrder": parent["wave_order"],
        "parent": parent,
        "candidates": candidates,
        "yes": decision.y,
        "different": [
            row["entity"] for row in candidates if row["entity"] not in decision.y
        ],
        "usage": usage,
        "directory": str(directory.relative_to(OUTPUT)),
        "conformance": conformance,
    }


def submit_wave(
    wave: int,
    wave_directory: Path,
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Submit every non-overlapping page concurrently and wait for complete validation."""
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    workers = len(selected)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(submit_or_resume_page, wave, wave_directory, parent): parent
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
                    {
                        "id": parent["id"],
                        "entity": parent["entity"],
                        "error": repr(error),
                    }
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
                f"{parent['entity']} — {outcome}",
                flush=True,
            )
    if failures:
        raise RuntimeError(
            f"Wave {wave} has {len(failures)} failures; no decisions were applied. "
            "Rerun to resume saved responses."
        )
    return sorted(results, key=lambda row: row["waveOrder"])


def build_pending(
    wave: int,
    results: list[dict[str, Any]],
    index_by_entity: dict[str, dict[str, Any]],
    existing_assignments: dict[str, Any],
) -> dict[str, Any]:
    """Prove the new wave contains no duplicate or previously assigned entity."""
    assignments: dict[str, dict[str, Any]] = {}
    clusters: list[dict[str, Any]] = []
    for result in results:
        parent = result["parent"]
        members = [parent["entity"], *result["yes"]]
        member_ids = [index_by_entity[entity]["id"] for entity in members]
        for entity, entity_id in zip(members, member_ids, strict=True):
            if entity_id in existing_assignments:
                raise RuntimeError(f"Wave {wave} reassigns an existing entity: {entity}")
            if entity_id in assignments:
                raise RuntimeError(f"Wave {wave} assigns an entity twice: {entity}")
            assignments[entity_id] = {
                "wave": wave,
                "entity": entity,
                "parentId": parent["id"],
                "parentEntity": parent["entity"],
                "waveOrder": result["waveOrder"],
            }
        clusters.append(
            {
                "wave": wave,
                "waveOrder": result["waveOrder"],
                "parentId": parent["id"],
                "parentEntity": parent["entity"],
                "parentMentions": parent["mentions"],
                "activeCandidateCount": len(result["candidates"]),
                "memberIds": member_ids,
                "members": members,
                "different": result["different"],
                "usage": result["usage"],
                "directory": result["directory"],
                "conformance": result["conformance"],
            }
        )
    return {
        "wave": wave,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "assignments": assignments,
        "clusters": clusters,
    }


def target_files_for_entities(entities: set[str]) -> set[str]:
    """Use the immutable location index to find every page that may need pruning."""
    if not entities:
        return set()
    placeholders = ",".join("?" for _ in entities)
    database = sqlite3.connect(OUTPUT / LOCATION_DB)
    try:
        return {
            str(row[0])
            for row in database.execute(
                f"SELECT DISTINCT file FROM locations WHERE candidate IN ({placeholders})",
                tuple(entities),
            )
        }
    finally:
        database.close()


def apply_pending(
    state: dict[str, Any],
    pending: dict[str, Any],
    eligible: list[dict[str, Any]],
    index_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Idempotently remove a complete wave everywhere before exposing its new state."""
    new_assignments = pending["assignments"]
    new_ids = set(new_assignments)
    new_entities = {
        assignment["entity"] for assignment in new_assignments.values()
    }
    targets = target_files_for_entities(new_entities)
    rows_removed = 0
    pages_changed = 0
    for filename in sorted(targets):
        active = OUTPUT / ACTIVE_LISTS / filename
        retired = OUTPUT / RETIRED_LISTS / filename
        path = active if active.is_file() else retired
        if not path.is_file():
            raise RuntimeError(f"Indexed candidate page is missing: {filename}")
        rows = base.read_candidate_csv(path)
        retained = [row for row in rows if row["entity"] not in new_entities]
        if len(retained) != len(rows):
            rows_removed += len(rows) - len(retained)
            pages_changed += 1
            base.write_candidate_csv(path, retained)

    eligible_ids = {row["id"] for row in eligible}
    retired_now = 0
    for entity_id in new_ids & eligible_ids:
        filename = index_by_id[entity_id]["file"]
        active = OUTPUT / ACTIVE_LISTS / filename
        retired = OUTPUT / RETIRED_LISTS / filename
        if active.is_file():
            os.replace(active, retired)
            retired_now += 1
        elif not retired.is_file():
            raise RuntimeError(f"Assigned eligible page is missing: {filename}")

    combined_ids = set(state["assignments"]) | new_ids
    base.write_active_roster(OUTPUT, eligible, combined_ids)
    return {
        "newAssignedEntities": len(new_ids),
        "newAssignedEligibleParents": len(new_ids & eligible_ids),
        "candidateSourcePagesTargeted": len(targets),
        "candidateRowsRemovedThisRecoveryPass": rows_removed,
        "candidatePagesChangedThisRecoveryPass": pages_changed,
        "pagesRetiredThisRecoveryPass": retired_now,
    }


def audit_global_exclusion(
    assignments: dict[str, Any], eligible: list[dict[str, Any]]
) -> dict[str, Any]:
    """Exhaustively prove that all cumulative assignments are absent from lists and roster."""
    assigned_ids = set(assignments)
    assigned_entities = {
        assignment["entity"] for assignment in assignments.values()
    }
    eligible_ids = {row["id"] for row in eligible}
    active_files = {path.name for path in (OUTPUT / ACTIVE_LISTS).glob("*.csv")}
    retired_files = {path.name for path in (OUTPUT / RETIRED_LISTS).glob("*.csv")}
    if active_files & retired_files:
        raise RuntimeError("A materialized page exists in both active and retired folders")
    if len(active_files) + len(retired_files) != len(eligible):
        raise RuntimeError("Materialized eligible page count changed")
    expected_retired = {
        row["file"] for row in eligible if row["id"] in assigned_ids
    }
    if retired_files != expected_retired:
        raise RuntimeError("Retired pages do not match cumulative eligible assignments")
    candidate_rows = 0
    for folder_name in (ACTIVE_LISTS, RETIRED_LISTS):
        for path in (OUTPUT / folder_name).glob("*.csv"):
            rows = base.read_candidate_csv(path)
            candidate_rows += len(rows)
            leaked = assigned_entities & {row["entity"] for row in rows}
            if leaked:
                raise RuntimeError(f"Assigned tags leaked into {path.name}: {sorted(leaked)}")
    with (OUTPUT / "active_roster.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        roster_ids = {row["id"] for row in csv.DictReader(handle)}
    expected_roster = eligible_ids - assigned_ids
    if roster_ids != expected_roster:
        raise RuntimeError("Active roster does not match cumulative assignments")
    return {
        "status": "passed",
        "materializedEligiblePages": len(eligible),
        "activePages": len(active_files),
        "retiredPages": len(retired_files),
        "workingCandidateRowsAfterPurge": candidate_rows,
        "assignedEntitiesAbsentFromEveryMaterializedList": True,
        "activeRosterMatchesAssignments": True,
    }


def usage_totals(clusters: list[dict[str, Any]]) -> dict[str, int]:
    """Sum exact Gemini usage metadata without estimating prompt or answer tokens."""
    keys = {
        "promptTokens": "prompt_token_count",
        "answerTokens": "candidates_token_count",
        "thinkingTokens": "thoughts_token_count",
        "totalTokens": "total_token_count",
    }
    return {
        output_key: sum(
            int(cluster["usage"].get(api_key_name) or 0) for cluster in clusters
        )
        for output_key, api_key_name in keys.items()
    }


def complete_pending(
    state: dict[str, Any],
    pending: dict[str, Any],
    eligible: list[dict[str, Any]],
    index_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply, audit, summarize, and commit a pending wave as one recoverable transition."""
    wave = int(pending["wave"])
    mutation = apply_pending(state, pending, eligible, index_by_id)
    combined_assignments = {**state["assignments"], **pending["assignments"]}
    audit = audit_global_exclusion(combined_assignments, eligible)
    clusters = pending["clusters"]
    new_eligible = mutation["newAssignedEligibleParents"]
    summary = {
        "wave": wave,
        "submittedCalls": len(clusters),
        "newAssignedEntities": len(pending["assignments"]),
        "newAssignedEligibleParents": new_eligible,
        "futureCallsAvoided": new_eligible - len(clusters),
        "singletonAliases": len(pending["assignments"]) - new_eligible,
        "multiEntityClusters": sum(len(cluster["members"]) > 1 for cluster in clusters),
        "conformanceDroppedStrings": sum(
            len(cluster["conformance"]["dropped"]) for cluster in clusters
        ),
        "activeCandidates": {
            "minimum": min(cluster["activeCandidateCount"] for cluster in clusters),
            "maximum": max(cluster["activeCandidateCount"] for cluster in clusters),
            "mean": round(
                sum(cluster["activeCandidateCount"] for cluster in clusters) / len(clusters),
                3,
            ),
        },
        "candidateRowsRemoved": mutation["candidateRowsRemovedThisRecoveryPass"],
        "remainingEligibleParents": audit["activePages"],
        "usage": usage_totals(clusters),
        "audit": audit,
    }
    wave_directory = OUTPUT / "waves" / f"wave_{wave:03d}"
    base.write_json(wave_directory / "wave_summary.json", summary)
    write_wave_review(wave_directory, clusters, summary)
    state["assignments"] = combined_assignments
    state["clusters"].extend(clusters)
    state["waveSummaries"].append(summary)
    state["completedWaves"] = wave
    state["pendingWave"] = None
    state["status"] = "ready" if wave < TARGET_COMPLETED_WAVES else "completed"
    write_state(state)
    base.write_json(wave_directory / "GLOBAL_EXCLUSION_AUDIT.json", audit)
    return summary


def write_wave_review(
    wave_directory: Path,
    clusters: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """Write one compact human-readable decision review beside the raw responses."""
    lines = [
        f"# Wave {summary['wave']} review",
        "",
        f"- Calls: {summary['submittedCalls']}",
        f"- Entities assigned: {summary['newAssignedEntities']}",
        f"- Future calls avoided: {summary['futureCallsAvoided']}",
        f"- Singleton aliases: {summary['singletonAliases']}",
        f"- Remaining eligible parents: {summary['remainingEligibleParents']}",
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
                f"### {cluster['waveOrder']}. {cluster['parentEntity']} "
                f"({cluster['parentMentions']} mentions)",
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


def write_cumulative_exports(
    state: dict[str, Any], all_rows: list[dict[str, Any]]
) -> None:
    """Export a concise ten-wave ledger, manifest, and review for final inspection."""
    assignments = state["assignments"]
    with (OUTPUT / "resolved_entities_after_10_waves.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ("id", "entity", "mentions", "parent_id", "parent_entity", "wave", "wave_order")
        )
        for row in all_rows:
            assignment = assignments.get(row["id"])
            if assignment:
                writer.writerow(
                    (
                        row["id"],
                        row["entity"],
                        row["mentions"],
                        assignment["parentId"],
                        assignment["parentEntity"],
                        assignment["wave"],
                        assignment["waveOrder"],
                    )
                )
    base.write_json(OUTPUT / "resolved_clusters_after_10_waves.json", state["clusters"])
    summary_fields = [
        "wave",
        "submittedCalls",
        "newAssignedEntities",
        "newAssignedEligibleParents",
        "futureCallsAvoided",
        "singletonAliases",
        "multiEntityClusters",
        "candidateRowsRemoved",
        "remainingEligibleParents",
    ]
    with (OUTPUT / "WAVE_SUMMARIES.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [{field: summary.get(field, 0) for field in summary_fields} for summary in state["waveSummaries"]]
        )
    total_usage = {
        key: sum(int(summary["usage"].get(key) or 0) for summary in state["waveSummaries"])
        for key in ("promptTokens", "answerTokens", "thinkingTokens", "totalTokens")
    }
    total_calls = sum(summary["submittedCalls"] for summary in state["waveSummaries"])
    manifest = {
        "status": state["status"],
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "thinkingLevel": "minimal",
        "includeThoughts": False,
        "completedWaves": state["completedWaves"],
        "totalCalls": total_calls,
        "initialEligibleParents": state["initialEligibleParents"],
        "assignedEntities": len(assignments),
        "assignedEligibleParents": (
            state["initialEligibleParents"] - state["waveSummaries"][-1]["remainingEligibleParents"]
        ),
        "remainingEligibleParents": state["waveSummaries"][-1]["remainingEligibleParents"],
        "futureCallsAvoided": sum(
            summary["futureCallsAvoided"] for summary in state["waveSummaries"]
        ),
        "singletonAliases": sum(summary["singletonAliases"] for summary in state["waveSummaries"]),
        "multiEntityClusters": sum(
            summary["multiEntityClusters"] for summary in state["waveSummaries"]
        ),
        "conformanceDroppedStrings": sum(
            summary.get("conformanceDroppedStrings", 0) for summary in state["waveSummaries"]
        ),
        "usage": total_usage,
        "finalGlobalExclusionAudit": state["waveSummaries"][-1]["audit"],
    }
    base.write_json(OUTPUT / "ten_wave_run_manifest.json", manifest)
    lines = [
        "# Ten-wave entity-resolution review",
        "",
        f"- Completed waves: {manifest['completedWaves']}",
        f"- Gemini calls: {manifest['totalCalls']}",
        f"- Unique entity tags assigned: {manifest['assignedEntities']}",
        f"- Future calls avoided: {manifest['futureCallsAvoided']}",
        f"- Singleton aliases absorbed: {manifest['singletonAliases']}",
        f"- Eligible parents remaining: {manifest['remainingEligibleParents']}",
        f"- Total API tokens: {total_usage['totalTokens']}",
        f"- Thinking tokens: {total_usage['thinkingTokens']}",
        f"- Final exclusion audit: {manifest['finalGlobalExclusionAudit']['status']}",
        "",
        "## Wave summaries",
        "",
        "| Wave | Calls | Assigned | Calls avoided | Singletons | Remaining | Tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in state["waveSummaries"]:
        lines.append(
            f"| {summary['wave']} | {summary['submittedCalls']} | "
            f"{summary['newAssignedEntities']} | {summary['futureCallsAvoided']} | "
            f"{summary['singletonAliases']} | {summary['remainingEligibleParents']} | "
            f"{summary['usage']['totalTokens']} |"
        )
    lines.extend(
        [
            "",
            "Individual decisions for waves 2–10 are in each `waves/wave_NNN/REVIEW.md`; "
            "the original `REVIEW.md` contains wave one.",
        ]
    )
    (OUTPUT / "TEN_WAVE_REVIEW.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def write_full_completion_exports(
    state: dict[str, Any],
    all_rows: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
) -> dict[str, Any]:
    """Export the completed non-singleton ledger and exact unresolved-singleton roster."""
    assignments = state["assignments"]
    assigned_ids = set(assignments)
    eligible_ids = {row["id"] for row in eligible}
    if not eligible_ids <= assigned_ids:
        raise RuntimeError("Cannot finalize while eligible non-singleton parents remain")
    audit = audit_global_exclusion(assignments, eligible)
    if audit["activePages"] != 0:
        raise RuntimeError("Completed non-singleton pass still has active pages")

    singleton_rows = [row for row in all_rows if row["mentions"] == 1]
    assigned_singletons = [row for row in singleton_rows if row["id"] in assigned_ids]
    remaining_singletons = [row for row in singleton_rows if row["id"] not in assigned_ids]
    with (OUTPUT / "remaining_singletons_after_nonsingleton_completion.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("id", "entity", "mentions", "source_candidates", "file"))
        for row in remaining_singletons:
            writer.writerow(
                (row["id"], row["entity"], row["mentions"], row["candidates"], row["file"])
            )

    with (OUTPUT / "resolved_entities_after_nonsingleton_completion.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ("id", "entity", "mentions", "parent_id", "parent_entity", "wave", "wave_order")
        )
        for row in all_rows:
            assignment = assignments.get(row["id"])
            if assignment:
                writer.writerow(
                    (
                        row["id"],
                        row["entity"],
                        row["mentions"],
                        assignment["parentId"],
                        assignment["parentEntity"],
                        assignment["wave"],
                        assignment["waveOrder"],
                    )
                )
    base.write_json(
        OUTPUT / "resolved_clusters_after_nonsingleton_completion.json",
        state["clusters"],
    )

    summary_fields = [
        "wave",
        "submittedCalls",
        "newAssignedEntities",
        "newAssignedEligibleParents",
        "futureCallsAvoided",
        "singletonAliases",
        "multiEntityClusters",
        "candidateRowsRemoved",
        "remainingEligibleParents",
    ]
    with (OUTPUT / "ALL_WAVE_SUMMARIES.csv").open(
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

    total_usage = {
        key: sum(
            int(summary["usage"].get(key) or 0)
            for summary in state["waveSummaries"]
        )
        for key in ("promptTokens", "answerTokens", "thinkingTokens", "totalTokens")
    }
    total_calls = sum(
        int(summary["submittedCalls"]) for summary in state["waveSummaries"]
    )
    manifest = {
        "status": "completed_non_singleton_pass",
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "thinkingLevel": "minimal",
        "includeThoughts": False,
        "completedWaves": state["completedWaves"],
        "totalCalls": total_calls,
        "initialEligibleNonSingletonParents": len(eligible),
        "assignedNonSingletonEntities": len(eligible_ids),
        "futureNonSingletonCallsAvoided": len(eligible_ids) - total_calls,
        "initialSingletonEntities": len(singleton_rows),
        "singletonAliasesAbsorbed": len(assigned_singletons),
        "remainingSingletonEntities": len(remaining_singletons),
        "totalAssignedEntityTags": len(assignments),
        "totalAliasesAbsorbedWithoutOwnCall": len(assignments) - total_calls,
        "conformanceDroppedStrings": sum(
            int(summary.get("conformanceDroppedStrings", 0))
            for summary in state["waveSummaries"]
        ),
        "usage": total_usage,
        "finalGlobalExclusionAudit": audit,
        "remainingSingletonRoster": "remaining_singletons_after_nonsingleton_completion.csv",
    }
    base.write_json(OUTPUT / "full_nonsingleton_run_manifest.json", manifest)
    base.write_json(OUTPUT / "FULL_NONSINGLETON_GLOBAL_EXCLUSION_AUDIT.json", audit)

    lines = [
        "# Completed non-singleton entity-resolution pass",
        "",
        f"- Completed waves: {manifest['completedWaves']}",
        f"- Gemini calls: {manifest['totalCalls']}",
        f"- Non-singleton entities assigned: {manifest['assignedNonSingletonEntities']}",
        f"- Future non-singleton calls avoided: {manifest['futureNonSingletonCallsAvoided']}",
        f"- Singleton aliases absorbed: {manifest['singletonAliasesAbsorbed']}",
        f"- Singleton entities remaining: {manifest['remainingSingletonEntities']}",
        f"- Total entity tags assigned: {manifest['totalAssignedEntityTags']}",
        f"- Total API tokens: {total_usage['totalTokens']}",
        f"- Thinking tokens: {total_usage['thinkingTokens']}",
        f"- Final exclusion audit: {audit['status']}",
        "",
        "## Wave summaries",
        "",
        "| Wave | Calls | Assigned | Calls avoided | Singletons | Remaining parents | Tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in state["waveSummaries"]:
        lines.append(
            f"| {summary['wave']} | {summary['submittedCalls']} | "
            f"{summary['newAssignedEntities']} | {summary['futureCallsAvoided']} | "
            f"{summary['singletonAliases']} | {summary['remainingEligibleParents']} | "
            f"{summary['usage']['totalTokens']} |"
        )
    lines.extend(
        [
            "",
            "The singleton roster was counted but not submitted. Exact page-level decisions "
            "remain under `waves/wave_NNN/`, with the original wave-one decisions at the root.",
        ]
    )
    (OUTPUT / "FULL_NONSINGLETON_REVIEW.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    return manifest


def resume_pending_if_needed(
    state: dict[str, Any],
    eligible: list[dict[str, Any]],
    index_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Finish any already-paid wave mutation before selecting or submitting another wave."""
    pending_wave = state.get("pendingWave")
    if pending_wave is None:
        return state
    pending_path = OUTPUT / "waves" / f"wave_{int(pending_wave):03d}" / "pending_commit.json"
    if not pending_path.is_file():
        raise RuntimeError("State names a pending wave without a pending commit file")
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    summary = complete_pending(state, pending, eligible, index_by_id)
    print(
        f"resumed and committed wave {pending_wave}: "
        f"{summary['newAssignedEntities']} entities assigned",
        flush=True,
    )
    return read_state()


def main() -> None:
    """Safely resume the cascade until exactly ten complete waves have been committed."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args()
    all_rows = base.read_index()
    eligible = [row for row in all_rows if row["mentions"] >= base.MINIMUM_PARENT_MENTIONS]
    index_by_id = {row["id"]: row for row in all_rows}
    index_by_entity = {row["entity"]: row for row in all_rows}
    initialize_continuation(all_rows)
    initialize_location_database()
    state = resume_pending_if_needed(read_state(), eligible, index_by_id)
    if args.until_empty:
        state["completionTarget"] = "all_non_singleton_parents"
        if read_active_roster():
            state["status"] = "ready"
        write_state(state)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "completedWaves": state["completedWaves"],
                    "targetCompletedWaves": (
                        "until_non_singleton_roster_empty"
                        if args.until_empty
                        else TARGET_COMPLETED_WAVES
                    ),
                    "remainingEligibleParents": len(read_active_roster()),
                    "apiCallsMadeThisInvocation": 0,
                },
                indent=2,
            )
        )
        return
    if not api_key():
        raise RuntimeError("GEMINI_API_KEY is not configured")

    def work_remains() -> bool:
        """Stop at wave ten normally or at an empty roster for full completion."""
        if args.until_empty:
            return bool(read_active_roster())
        return int(state["completedWaves"]) < TARGET_COMPLETED_WAVES

    while work_remains():
        wave = int(state["completedWaves"]) + 1
        wave_directory, selected = load_or_create_selection(wave)
        results = submit_wave(wave, wave_directory, selected)
        pending = build_pending(
            wave,
            results,
            index_by_entity,
            state["assignments"],
        )
        base.write_json(wave_directory / "pending_commit.json", pending)
        state["pendingWave"] = wave
        state["status"] = "responses_validated_pending_commit"
        write_state(state)
        summary = complete_pending(state, pending, eligible, index_by_id)
        print(
            f"wave {wave} committed: calls={summary['submittedCalls']}, "
            f"assigned={summary['newAssignedEntities']}, "
            f"future calls avoided={summary['futureCallsAvoided']}, "
            f"remaining={summary['remainingEligibleParents']}",
            flush=True,
        )
        state = read_state()

    if args.until_empty:
        manifest = write_full_completion_exports(state, all_rows, eligible)
        state["status"] = "completed_non_singleton_pass"
        write_state(state)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    else:
        write_cumulative_exports(state, all_rows)
        print((OUTPUT / "TEN_WAVE_REVIEW.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()

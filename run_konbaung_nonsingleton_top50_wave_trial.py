#!/usr/bin/env python3
"""Submit up to 100 pages from a frequency-greedy, disjoint top-50 wave."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from konbaung_gemini_summary_claim_completion_annotator import api_key
from run_konbaung_binary_resolution_production import (
    BinaryDecision,
    binary_prompt,
    conform_binary_decision,
    validate_binary_decision,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "konbaung_entity_resolution_candidates_20260902_v4_nonsingleton_top50"
INDEX = SOURCE / "index.csv"
OUTPUT = ROOT / "konbaung_nonsingleton_top50_frequency_wave_gemini_trial_20260902_first100"
MODEL = "gemini-3.1-flash-lite"
MINIMUM_PARENT_MENTIONS = 2
TRIAL_PAGE_CAP = 100
MAX_WORKERS = 100
MAX_OUTPUT_TOKENS = 1600
RETRY_DELAYS = (2, 4, 8, 16, 30, 30)
ACTIVE_LISTS = "active_candidate_lists"
RETIRED_LISTS = "retired_candidate_lists"
THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    """Keep paid submission behind an explicit flag while allowing a zero-call plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit or resume up to 100 disjoint pages, then apply their decisions.",
    )
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    """Write a durable JSON checkpoint without exposing a partially written file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def safe_name(value: str) -> str:
    """Make a readable Windows-safe component for one response directory."""
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", "_", value).strip(" ._")
    return value[:60].rstrip(" ._") or "entity"


def read_index() -> list[dict[str, Any]]:
    """Load the immutable v4 index and verify its frequency and page-size invariants."""
    with INDEX.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    expected = ["id", "entity", "mentions", "candidates", "file"]
    if not raw_rows or list(raw_rows[0]) != expected:
        raise RuntimeError("Unexpected v4 candidate index schema")
    rows: list[dict[str, Any]] = []
    for position, raw in enumerate(raw_rows, start=1):
        rows.append(
            {
                **raw,
                "position": position,
                "mentions": int(raw["mentions"]),
                "candidates": int(raw["candidates"]),
            }
        )
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("Entity IDs are not unique")
    if len({row["entity"] for row in rows}) != len(rows):
        raise RuntimeError("Entity tags are not unique")
    if not all(
        left["mentions"] >= right["mentions"]
        for left, right in zip(rows, rows[1:])
    ):
        raise RuntimeError("Entity index is not frequency-descending")
    eligible = [row for row in rows if row["mentions"] >= MINIMUM_PARENT_MENTIONS]
    if len(eligible) != 6498 or any(row["candidates"] != 50 for row in eligible):
        raise RuntimeError("Expected exactly 6,498 non-singleton top-50 pages")
    return rows


def read_candidate_csv(path: Path) -> list[dict[str, str]]:
    """Read one compact entity-neighbor page and enforce its three-column schema."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["entity", "emb", "char"]:
            raise RuntimeError(f"Unexpected candidate schema: {path}")
        return list(reader)


def write_candidate_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Atomically replace a working page after globally removing resolved tags."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["entity", "emb", "char"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def build_frequency_greedy_wave(
    eligible: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Admit pages in strict mention order whenever their complete footprints are disjoint."""
    occupied_by: dict[str, dict[str, Any]] = {}
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for scan_order, parent in enumerate(eligible, start=1):
        candidates = read_candidate_csv(SOURCE / "entities" / parent["file"])
        if len(candidates) != 50:
            raise RuntimeError(f"Eligible page is not top 50: {parent['file']}")
        footprint = [parent["entity"], *[row["entity"] for row in candidates]]
        if len(set(footprint)) != 51:
            raise RuntimeError(f"Page footprint is not 51 distinct tags: {parent['file']}")
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
            "footprint_sha256": hashlib.sha256(
                "\n".join(footprint).encode("utf-8")
            ).hexdigest(),
        }
        selected.append(record)
        for tag in footprint:
            occupied_by[tag] = record
    return selected, skipped


def write_selection_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Materialize the selected page order without duplicating the 50 candidate rows."""
    fields = [
        "wave_order",
        "scan_order",
        "position",
        "id",
        "entity",
        "mentions",
        "candidates",
        "file",
        "footprint_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row[field] for field in fields} for row in rows])


def write_skipped_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Record why every excluded page could not coexist with the higher-priority packing."""
    fields = [
        "scan_order",
        "id",
        "entity",
        "mentions",
        "first_conflict",
        "conflicting_wave_order",
        "conflicting_parent",
        "conflict_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_active_roster(
    root: Path,
    eligible: list[dict[str, Any]],
    assigned_ids: set[str],
) -> None:
    """Write the unresolved non-singleton parent roster in corpus-frequency order."""
    path = root / "active_roster.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("id", "entity", "mentions", "source_candidates", "file"))
        for row in eligible:
            if row["id"] not in assigned_ids:
                writer.writerow(
                    (row["id"], row["entity"], row["mentions"], 50, row["file"])
                )
    os.replace(temporary, path)


def initialize_output(
    all_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> None:
    """Atomically install the frozen plan and materialized non-singleton working lists."""
    if OUTPUT.exists():
        if not (OUTPUT / "state.json").is_file():
            raise RuntimeError(f"Output exists without a recoverable state: {OUTPUT}")
        return
    eligible = [row for row in all_rows if row["mentions"] >= MINIMUM_PARENT_MENTIONS]
    staging = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.staging_", dir=ROOT))
    try:
        active = staging / ACTIVE_LISTS
        retired = staging / RETIRED_LISTS
        active.mkdir()
        retired.mkdir()
        for copied, row in enumerate(eligible, start=1):
            shutil.copy2(SOURCE / "entities" / row["file"], active / row["file"])
            if copied % 1000 == 0:
                print(f"materialized {copied}/{len(eligible)} eligible pages", flush=True)
        write_selection_csv(staging / "first_wave_all_pages.csv", selected)
        trial_selection = selected[:TRIAL_PAGE_CAP]
        write_selection_csv(staging / "first_wave_trial_first100.csv", trial_selection)
        write_skipped_csv(staging / "first_wave_conflict_skips.csv", skipped)
        write_active_roster(staging, eligible, set())
        state = {
            "schemaVersion": 1,
            "status": "initialized_no_api_calls",
            "source": str(SOURCE),
            "model": MODEL,
            "thinkingLevel": "minimal",
            "includeThoughts": False,
            "eligibleParentPages": len(eligible),
            "firstWavePages": len(selected),
            "trialPages": len(trial_selection),
            "submittedResponsesSaved": 0,
            "pendingCommit": None,
            "assignments": {},
            "clusters": [],
        }
        write_json(staging / "state.json", state)
        write_json(
            staging / "plan_manifest.json",
            {
                "status": "ready_not_submitted",
                "apiCallsMade": 0,
                "sourceEntities": len(all_rows),
                "excludedSingletonParents": len(all_rows) - len(eligible),
                "eligibleNonSingletonParents": len(eligible),
                "candidatesPerEligiblePage": 50,
                "packing": (
                    "Strict corpus-frequency-descending greedy maximal packing of "
                    "pairwise-disjoint {parent + 50 candidates} footprints."
                ),
                "firstWavePages": len(selected),
                "trialPageCap": TRIAL_PAGE_CAP,
                "pagesInTrial": len(trial_selection),
                "workerCap": MAX_WORKERS,
                "workersUsed": min(MAX_WORKERS, len(trial_selection)),
                "decisionSchema": {"y": ["exact supplied YES tags"]},
            },
        )
        (staging / "README.md").write_text(
            "# Non-singleton top-50 frequency-wave trial\n\n"
            "All 6,498 eligible parent pages have more than one corpus mention and "
            "exactly 50 candidates. `first_wave_all_pages.csv` is a strict "
            "frequency-first greedy packing whose complete 51-tag footprints do not "
            f"overlap. Up to its first {TRIAL_PAGE_CAP} pages are submitted; because "
            f"this wave contains {len(selected)}, this trial submits "
            f"{len(trial_selection)}.\n\n"
            "Gemini returns only `y`; membership means YES and omission means NO. All "
            "100 responses are saved and validated before any roster/list mutation. "
            "Afterward, each parent and accepted alias is assigned once, removed from "
            "the eligible roster, and physically deleted from every materialized "
            "non-singleton candidate list. The immutable v4 source is unchanged.\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(staging, OUTPUT)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def load_frozen_trial(index_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Reload the frozen capped selection so a resume cannot alter the paid inputs."""
    with (OUTPUT / "first_wave_trial_first100.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        frozen = list(csv.DictReader(handle))
    if not frozen or len(frozen) > TRIAL_PAGE_CAP:
        raise RuntimeError("Frozen trial must contain between 1 and 100 pages")
    rows: list[dict[str, Any]] = []
    for raw in frozen:
        source = index_by_id[raw["id"]]
        if source["entity"] != raw["entity"] or int(raw["mentions"]) != source["mentions"]:
            raise RuntimeError(f"Frozen selection differs from source: {raw['id']}")
        rows.append({**source, "wave_order": int(raw["wave_order"])})
    return rows


def get_client() -> genai.Client:
    """Give each simultaneous worker its own API client to avoid shared transport state."""
    client = getattr(THREAD_LOCAL, "client", None)
    if client is None:
        client = genai.Client(api_key=api_key())
        THREAD_LOCAL.client = client
    return client


def answer_from_raw(raw: dict[str, Any]) -> str:
    """Extract only the structured non-thought answer from a saved Gemini response."""
    texts: list[str] = []
    for candidate in raw.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            if part.get("text") and not part.get("thought", False):
                texts.append(part["text"])
    return "".join(texts).strip()


def is_transient(error: Exception) -> bool:
    """Retry only recognizable capacity, network, timeout, or server failures."""
    lowered = str(error).casefold()
    return any(
        marker in lowered
        for marker in (
            "429",
            "500",
            "502",
            "503",
            "504",
            "resource_exhausted",
            "rate limit",
            "temporarily unavailable",
            "deadline exceeded",
            "connection",
            "timeout",
        )
    )


def submit_or_resume_page(parent: dict[str, Any]) -> dict[str, Any]:
    """Save one paid response exactly once, or validate its raw checkpoint on resume."""
    order = int(parent["wave_order"])
    directory = OUTPUT / f"page_{order:03d}_{parent['id']}_{safe_name(parent['entity'])}"
    directory.mkdir(exist_ok=True)
    candidates = read_candidate_csv(SOURCE / "entities" / parent["file"])
    if len(candidates) != 50:
        raise RuntimeError(f"Trial page no longer contains 50 candidates: {parent['id']}")
    prompt = binary_prompt(parent["entity"], candidates)
    prompt_path = directory / "prompt_sent.txt"
    if prompt_path.exists():
        if prompt_path.read_text(encoding="utf-8") != prompt:
            raise RuntimeError(f"Saved prompt differs for wave page {order}")
    else:
        prompt_path.write_text(prompt, encoding="utf-8", newline="\n")

    raw_path = directory / "raw_response.json"
    if raw_path.exists():
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        client = get_client()
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
                write_json(raw_path, raw)
                break
            except Exception as error:
                if attempt >= len(RETRY_DELAYS) or not is_transient(error):
                    raise
                delay = RETRY_DELAYS[attempt] + random.random()
                time.sleep(delay)
        else:
            raise RuntimeError("API retry loop ended unexpectedly")

    answer = answer_from_raw(raw)
    (directory / "answer_returned.json").write_text(
        answer + "\n", encoding="utf-8", newline="\n"
    )
    raw_decision = BinaryDecision.model_validate_json(answer)
    write_json(directory / "raw_decision.json", raw_decision.model_dump(mode="json"))
    decision, conformance = conform_binary_decision(
        raw_decision, parent["entity"], candidates
    )
    validate_binary_decision(decision, parent["entity"], candidates)
    write_json(directory / "conformance.json", conformance)
    write_json(directory / "result.json", decision.model_dump(mode="json"))
    usage = raw.get("usage_metadata") or {}
    page_result = {
        "waveOrder": order,
        "parent": parent,
        "candidates": candidates,
        "yes": decision.y,
        "no": [row["entity"] for row in candidates if row["entity"] not in decision.y],
        "usage": usage,
        "directory": directory.name,
        "conformance": conformance,
    }
    write_json(
        directory / "page_manifest.json",
        {
            "waveOrder": order,
            "parent": parent,
            "candidatesSent": 50,
            "promptCharacters": len(prompt),
            "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "thinkingLevel": "minimal",
            "includeThoughts": False,
            "usage": usage,
            "conformance": conformance,
        },
    )
    return page_result


def submit_wave(trial: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Launch every trial page together and defer mutation until all responses validate."""
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(trial))) as executor:
        futures = {executor.submit(submit_or_resume_page, row): row for row in trial}
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
            write_json(
                OUTPUT / "submission_progress.json",
                {
                    "completedWorkers": completed,
                    "validatedResponses": len(results),
                    "failures": failures,
                },
            )
            print(
                f"response {completed}/{len(trial)}: {parent['entity']} — {outcome}",
                flush=True,
            )
    if failures:
        raise RuntimeError(
            f"{len(failures)} pages failed; no decisions were applied. Rerun to resume."
        )
    return sorted(results, key=lambda row: row["waveOrder"])


def build_pending_commit(
    results: list[dict[str, Any]], index_by_entity: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Prove each tag has one cluster before recording the all-at-once mutation intent."""
    assignments: dict[str, dict[str, Any]] = {}
    clusters: list[dict[str, Any]] = []
    for result in results:
        parent = result["parent"]
        members = [parent["entity"], *result["yes"]]
        member_ids = [index_by_entity[entity]["id"] for entity in members]
        for entity, entity_id in zip(members, member_ids, strict=True):
            if entity_id in assignments:
                raise RuntimeError(f"Entity would be assigned twice: {entity}")
            assignments[entity_id] = {
                "entity": entity,
                "parentId": parent["id"],
                "parentEntity": parent["entity"],
                "waveOrder": result["waveOrder"],
            }
        clusters.append(
            {
                "waveOrder": result["waveOrder"],
                "parentId": parent["id"],
                "parentEntity": parent["entity"],
                "parentMentions": parent["mentions"],
                "memberIds": member_ids,
                "members": members,
                "different": result["no"],
                "usage": result["usage"],
                "directory": result["directory"],
                "conformance": result["conformance"],
            }
        )
    return {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "assignments": assignments,
        "clusters": clusters,
    }


def apply_pending_commit(
    pending: dict[str, Any],
    eligible: list[dict[str, Any]],
    index_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Idempotently purge all resolved tags globally and retire eligible assigned pages."""
    assigned_ids = set(pending["assignments"])
    assigned_entities = {
        assignment["entity"] for assignment in pending["assignments"].values()
    }
    rows_removed = 0
    pages_changed = 0
    for folder_name in (ACTIVE_LISTS, RETIRED_LISTS):
        for path in (OUTPUT / folder_name).glob("*.csv"):
            rows = read_candidate_csv(path)
            retained = [row for row in rows if row["entity"] not in assigned_entities]
            if len(retained) != len(rows):
                rows_removed += len(rows) - len(retained)
                pages_changed += 1
                write_candidate_csv(path, retained)

    eligible_ids = {row["id"] for row in eligible}
    retired_now = 0
    for entity_id in assigned_ids & eligible_ids:
        filename = index_by_id[entity_id]["file"]
        active = OUTPUT / ACTIVE_LISTS / filename
        retired = OUTPUT / RETIRED_LISTS / filename
        if active.exists():
            os.replace(active, retired)
            retired_now += 1
        elif not retired.exists():
            raise RuntimeError(f"Assigned eligible page is missing: {filename}")
    write_active_roster(OUTPUT, eligible, assigned_ids)
    return {
        "assignedEntities": len(assigned_ids),
        "assignedEligibleParents": len(assigned_ids & eligible_ids),
        "candidateRowsRemovedThisRecoveryPass": rows_removed,
        "candidatePagesChangedThisRecoveryPass": pages_changed,
        "pagesRetiredThisRecoveryPass": retired_now,
    }


def audit_commit(
    pending: dict[str, Any], eligible: list[dict[str, Any]]
) -> dict[str, Any]:
    """Scan every working page and roster to prove the global exclusion invariant."""
    assigned_ids = set(pending["assignments"])
    assigned_entities = {
        assignment["entity"] for assignment in pending["assignments"].values()
    }
    eligible_ids = {row["id"] for row in eligible}
    active_files = {path.name for path in (OUTPUT / ACTIVE_LISTS).glob("*.csv")}
    retired_files = {path.name for path in (OUTPUT / RETIRED_LISTS).glob("*.csv")}
    if active_files & retired_files:
        raise RuntimeError("A page exists in both active and retired folders")
    if len(active_files) + len(retired_files) != len(eligible):
        raise RuntimeError("Materialized eligible page count changed")
    expected_retired = {
        row["file"] for row in eligible if row["id"] in assigned_ids
    }
    if retired_files != expected_retired:
        raise RuntimeError("Retired pages do not exactly match assigned eligible entities")

    candidate_rows = 0
    for folder_name in (ACTIVE_LISTS, RETIRED_LISTS):
        for path in (OUTPUT / folder_name).glob("*.csv"):
            rows = read_candidate_csv(path)
            candidate_rows += len(rows)
            leaked = assigned_entities & {row["entity"] for row in rows}
            if leaked:
                raise RuntimeError(f"Resolved tags leaked into {path.name}: {sorted(leaked)}")
    with (OUTPUT / "active_roster.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        roster_ids = {row["id"] for row in csv.DictReader(handle)}
    expected_roster = eligible_ids - assigned_ids
    if roster_ids != expected_roster:
        raise RuntimeError("Active roster does not match eligible unresolved IDs")
    return {
        "status": "passed",
        "materializedEligiblePages": len(eligible),
        "activePages": len(active_files),
        "retiredPages": len(retired_files),
        "workingCandidateRowsAfterPurge": candidate_rows,
        "assignedEntitiesAbsentFromEveryMaterializedList": True,
        "activeRosterMatchesAssignments": True,
    }


def write_exports(
    all_rows: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
    pending: dict[str, Any],
    mutation: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Create compact review files and exact API token totals for the completed trial."""
    assignments = pending["assignments"]
    with (OUTPUT / "resolved_entities.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("id", "entity", "mentions", "parent_id", "parent_entity", "wave_order"))
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
                        assignment["waveOrder"],
                    )
                )
    write_json(OUTPUT / "resolved_clusters.json", pending["clusters"])
    write_json(OUTPUT / "GLOBAL_EXCLUSION_AUDIT.json", audit)

    usage_keys = {
        "promptTokens": "prompt_token_count",
        "answerTokens": "candidates_token_count",
        "thinkingTokens": "thoughts_token_count",
        "totalTokens": "total_token_count",
    }
    usage = {
        output_key: sum(
            int(cluster["usage"].get(api_key_name) or 0)
            for cluster in pending["clusters"]
        )
        for output_key, api_key_name in usage_keys.items()
    }
    eligible_ids = {row["id"] for row in eligible}
    manifest = {
        "status": "completed",
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "thinkingLevel": "minimal",
        "includeThoughts": False,
        "decisionMode": "binary_yes_or_no",
        "submittedPages": len(pending["clusters"]),
        "candidatesPerPage": 50,
        "workerCap": MAX_WORKERS,
        "workersUsed": min(MAX_WORKERS, len(pending["clusters"])),
        "firstWavePages": sum(1 for _ in csv.DictReader((OUTPUT / "first_wave_all_pages.csv").open(encoding="utf-8", newline=""))),
        "initialEligibleParents": len(eligible),
        "excludedSingletonParents": len(all_rows) - len(eligible),
        "assignedEntities": len(assignments),
        "assignedEligibleParents": len(set(assignments) & eligible_ids),
        "remainingEligibleParents": len(eligible_ids - set(assignments)),
        "multiEntityClusters": sum(
            len(cluster["members"]) > 1 for cluster in pending["clusters"]
        ),
        "conformanceDroppedStrings": sum(
            len(cluster["conformance"]["dropped"])
            for cluster in pending["clusters"]
        ),
        "mutation": mutation,
        "globalExclusionAudit": audit,
        "usage": usage,
    }
    write_json(OUTPUT / "run_manifest.json", manifest)
    lines = [
        "# Review: non-singleton top-50 frequency-wave trial",
        "",
        f"- First-wave pages available: {manifest['firstWavePages']}",
        f"- Pages submitted in this test: {manifest['submittedPages']}",
        f"- Entities assigned: {manifest['assignedEntities']}",
        f"- Eligible parent pages remaining: {manifest['remainingEligibleParents']}",
        f"- Multi-entity clusters: {manifest['multiEntityClusters']}",
        f"- Global exclusion audit: {audit['status']}",
        f"- Total API tokens: {usage['totalTokens']}",
        f"- Thinking tokens: {usage['thinkingTokens']}",
        "",
        "## Decisions",
        "",
    ]
    for cluster in pending["clusters"]:
        yes = " | ".join(cluster["members"][1:]) or "none"
        lines.extend(
            [
                f"### {cluster['waveOrder']}. {cluster['parentEntity']} "
                f"({cluster['parentMentions']} mentions)",
                "",
                f"- YES: {yes}",
                f"- NO: {len(cluster['different'])} omitted supplied candidates",
                "",
            ]
        )
    (OUTPUT / "REVIEW.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    return manifest


def main() -> None:
    """Plan safely by default, or submit and atomically commit the authorized trial."""
    args = parse_args()
    all_rows = read_index()
    eligible = [row for row in all_rows if row["mentions"] >= MINIMUM_PARENT_MENTIONS]
    if not OUTPUT.exists():
        selected, skipped = build_frequency_greedy_wave(eligible)
        print(
            f"frequency-greedy first wave: {len(selected)} pages; "
            f"{len(skipped)} conflict skips",
            flush=True,
        )
        initialize_output(all_rows, selected, skipped)

    index_by_id = {row["id"]: row for row in all_rows}
    index_by_entity = {row["entity"]: row for row in all_rows}
    trial = load_frozen_trial(index_by_id)
    state = json.loads((OUTPUT / "state.json").read_text(encoding="utf-8"))
    if not args.execute:
        print(json.dumps(json.loads((OUTPUT / "plan_manifest.json").read_text(encoding="utf-8")), indent=2))
        return
    if not api_key():
        raise RuntimeError("GEMINI_API_KEY is not configured")

    if state["status"] == "completed":
        print((OUTPUT / "REVIEW.md").read_text(encoding="utf-8"), flush=True)
        return
    results = submit_wave(trial)
    pending = build_pending_commit(results, index_by_entity)
    write_json(OUTPUT / "pending_commit.json", pending)
    state["status"] = "responses_validated_pending_commit"
    state["submittedResponsesSaved"] = len(results)
    state["pendingCommit"] = "pending_commit.json"
    write_json(OUTPUT / "state.json", state)

    mutation = apply_pending_commit(pending, eligible, index_by_id)
    audit = audit_commit(pending, eligible)
    manifest = write_exports(all_rows, eligible, pending, mutation, audit)
    state["status"] = "completed"
    state["assignments"] = pending["assignments"]
    state["clusters"] = pending["clusters"]
    state["pendingCommit"] = None
    write_json(OUTPUT / "state.json", state)
    write_json(
        OUTPUT / "plan_manifest.json",
        {**json.loads((OUTPUT / "plan_manifest.json").read_text(encoding="utf-8")), "status": "completed", "apiCallsMade": len(results)},
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

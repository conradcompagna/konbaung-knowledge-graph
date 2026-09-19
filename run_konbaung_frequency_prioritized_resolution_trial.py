#!/usr/bin/env python3
"""Resolve 100 frequency-prioritized entity pages with global physical exclusion."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict

from konbaung_gemini_summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "konbaung_entity_resolution_candidates_20260902_v3_frequency_prioritized"
INDEX = SOURCE / "index.csv"
OUTPUT = ROOT / "konbaung_frequency_prioritized_gemini_resolution_trial_20260902_first100"
ACTIVE_LISTS = "active_candidate_lists"
RETIRED_LISTS = "retired_candidate_lists"
LOCATION_DB = "source_candidate_locations.sqlite"
MODEL = "gemini-3.1-flash-lite"
TARGET_SUBMISSIONS = 100
MAX_OUTPUT_TOKENS = 1600
TRANSIENT_RETRY_DELAYS = (2, 4, 8, 16, 30, 30)


class Decision(BaseModel):
    """Keep Gemini's response to accepted and uncertain exact candidate tags."""

    model_config = ConfigDict(extra="forbid")

    s: list[str]
    u: list[str]


def write_json(path: Path, value: Any) -> None:
    """Write JSON atomically so interrupted checkpoints remain recoverable."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def safe_name(value: str) -> str:
    """Create a short Windows-safe directory component for each submitted parent."""
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", "_", value).strip(" ._")
    return value[:60].rstrip(" ._") or "entity"


def read_index() -> list[dict[str, Any]]:
    """Read and validate the frequency-descending mixed-size source index."""
    with INDEX.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    expected = ["id", "entity", "mentions", "candidates", "file"]
    if not raw_rows or list(raw_rows[0]) != expected:
        raise RuntimeError("Unexpected frequency-prioritized index schema")
    rows: list[dict[str, Any]] = []
    for position, raw in enumerate(raw_rows):
        row = {
            **raw,
            "position": position,
            "mentions": int(raw["mentions"]),
            "candidates": int(raw["candidates"]),
        }
        rows.append(row)
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("Entity IDs are not unique")
    if len({row["entity"] for row in rows}) != len(rows):
        raise RuntimeError("Entity tags are not unique")
    if not all(
        left["mentions"] >= right["mentions"]
        for left, right in zip(rows, rows[1:])
    ):
        raise RuntimeError("Entity index is not frequency-descending")
    for row in rows:
        expected_candidates = 50 if row["mentions"] > 10 else 20
        if row["candidates"] != expected_candidates:
            raise RuntimeError(f"Wrong candidate count in index: {row['id']}")
    return rows


def read_candidate_csv(path: Path) -> list[dict[str, str]]:
    """Read one active or retired candidate page with strict schema checking."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and list(rows[0]) != ["entity", "emb", "char"]:
        raise RuntimeError(f"Unexpected candidate schema: {path}")
    return rows


def write_candidate_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Replace one working candidate page atomically after global exclusions."""
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


def initial_state() -> dict[str, Any]:
    """Create an empty resolution ledger for a newly materialized working archive."""
    return {
        "schemaVersion": 2,
        "source": str(SOURCE),
        "model": MODEL,
        "thinkingLevel": "minimal",
        "includeThoughts": False,
        "cursor": 0,
        "submittedPasses": 0,
        "assignments": {},
        "clusters": [],
        "skippedParentPages": [],
        "passes": [],
        "pending": None,
    }


def initialize_working_archive(index_rows: list[dict[str, Any]]) -> None:
    """Copy source pages and build their inverted lookup before installing the run."""
    if OUTPUT.exists():
        if not (OUTPUT / "state.json").is_file():
            raise RuntimeError(f"Incomplete output exists without state: {OUTPUT}")
        return

    staging = Path(
        tempfile.mkdtemp(prefix=f".{OUTPUT.name}.staging_", dir=OUTPUT.parent)
    )
    try:
        active = staging / ACTIVE_LISTS
        retired = staging / RETIRED_LISTS
        shutil.copytree(SOURCE / "entities", active)
        retired.mkdir()

        database = sqlite3.connect(staging / LOCATION_DB)
        try:
            database.execute(
                "CREATE TABLE locations (candidate TEXT NOT NULL, file TEXT NOT NULL, "
                "PRIMARY KEY (candidate, file)) WITHOUT ROWID"
            )
            for position, row in enumerate(index_rows, start=1):
                candidates = read_candidate_csv(active / row["file"])
                if len(candidates) != row["candidates"]:
                    raise RuntimeError(f"Source page-size mismatch: {row['file']}")
                database.executemany(
                    "INSERT INTO locations(candidate, file) VALUES (?, ?)",
                    ((candidate["entity"], row["file"]) for candidate in candidates),
                )
                if position % 1000 == 0:
                    database.commit()
                    print(
                        f"indexed candidate locations for {position}/{len(index_rows)} pages",
                        flush=True,
                    )
            database.execute(
                "CREATE INDEX locations_candidate ON locations(candidate)"
            )
            database.commit()
        finally:
            database.close()

        state = initial_state()
        write_json(staging / "state.json", state)
        write_active_roster(staging, index_rows, set())
        (staging / "README.md").write_text(
            "# Active frequency-prioritized resolution trial\n\n"
            "`active_candidate_lists/` is the materialized working set. After each "
            "accepted decision, the parent and accepted aliases are removed from "
            "`active_roster.csv`, their own pages move to `retired_candidate_lists/`, "
            "and their rows are physically deleted from every active and retired page. "
            "The immutable v3 source archive is never changed.\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(staging, OUTPUT)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def write_active_roster(
    root: Path, index_rows: list[dict[str, Any]], assigned_ids: set[str]
) -> None:
    """Materialize the current unresolved parent roster in frequency order."""
    path = root / "active_roster.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("id", "entity", "mentions", "source_candidates", "file"))
        for row in index_rows:
            if row["id"] not in assigned_ids:
                writer.writerow(
                    (
                        row["id"],
                        row["entity"],
                        row["mentions"],
                        row["candidates"],
                        row["file"],
                    )
                )
    os.replace(temporary, path)


def load_state() -> dict[str, Any]:
    """Load the durable ledger used to resume without repeating completed API calls."""
    return json.loads((OUTPUT / "state.json").read_text(encoding="utf-8"))


def next_parent(
    index_rows: list[dict[str, Any]], state: dict[str, Any]
) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
    """Select the next highest-frequency unresolved parent and record absorbed skips."""
    cursor = int(state["cursor"])
    skipped: list[dict[str, Any]] = []
    while cursor < len(index_rows):
        row = index_rows[cursor]
        cursor += 1
        assignment = state["assignments"].get(row["id"])
        if assignment is None:
            return row, cursor, skipped
        skipped.append(
            {
                "id": row["id"],
                "entity": row["entity"],
                "mentions": row["mentions"],
                "resolvedToId": assignment["parentId"],
                "resolvedToEntity": assignment["parentEntity"],
                "resolvedInPass": assignment["pass"],
            }
        )
    raise RuntimeError("The unresolved roster is exhausted")


def active_candidates(
    parent: dict[str, Any], state: dict[str, Any]
) -> list[dict[str, str]]:
    """Read the physically cleaned page and reject any leaked resolved candidate."""
    path = OUTPUT / ACTIVE_LISTS / parent["file"]
    if not path.is_file():
        raise RuntimeError(f"Unresolved parent has no active page: {parent['entity']}")
    rows = read_candidate_csv(path)
    assigned_entities = {
        assignment["entity"] for assignment in state["assignments"].values()
    }
    leaked = assigned_entities & {row["entity"] for row in rows}
    if leaked:
        raise RuntimeError(f"Resolved candidates leaked into active page: {sorted(leaked)}")
    return rows


def build_prompt(parent: str, candidates: list[dict[str, str]]) -> str:
    """Ask Gemini to make strict same-referent decisions from the cleaned neighbor page."""
    candidate_lines = [
        f"{row['entity']}\t{row['emb']}\t{row['char']}" for row in candidates
    ]
    return f"""Decide which candidate tags denote exactly the same underlying entity as PARENT.

Identity must be strict. Merge spelling, transliteration, capitalization, separator,
word-order, or genuine title/name variants of the same referent. Do not merge merely
related things: person/forces/family/reign, ruler/dynasty/state, place/fort/unit/residents,
title/office-holder, singular/plural class, or part/whole. For a generic parent, merge
only equivalent generic expressions, never named instances. emb and char are retrieval
clues, not proof. Use reliable historical knowledge only when confident.

Return only JSON {{"s":[...],"u":[...]}}. `s` contains exact candidate strings that are
the same entity as PARENT. `u` contains exact candidate strings that may be the same but
remain genuinely uncertain. Do not include PARENT. Omitted candidates are different.

PARENT
{parent}

CANDIDATES\tentity,emb,char
{chr(10).join(candidate_lines)}
"""


def validate_decision(
    decision: Decision, parent: str, candidates: list[dict[str, str]]
) -> None:
    """Reject invented, repeated, overlapping, or parent-valued output strings."""
    allowed = {row["entity"] for row in candidates}
    if parent in decision.s or parent in decision.u:
        raise RuntimeError("Gemini included the parent in its candidate lists")
    if len(decision.s) != len(set(decision.s)) or len(decision.u) != len(
        set(decision.u)
    ):
        raise RuntimeError("Gemini repeated a candidate")
    if set(decision.s) & set(decision.u):
        raise RuntimeError("Gemini marked a candidate both same and uncertain")
    unknown = (set(decision.s) | set(decision.u)) - allowed
    if unknown:
        raise RuntimeError(f"Gemini returned unknown candidates: {sorted(unknown)}")


def conform_decision(
    decision: Decision, parent: str, candidates: list[dict[str, str]]
) -> tuple[Decision, dict[str, Any]]:
    """Drop non-candidate or structurally invalid strings without making another API call."""
    allowed = {row["entity"] for row in candidates}
    accepted: list[str] = []
    uncertain: list[str] = []
    dropped: list[dict[str, str]] = []
    seen_same: set[str] = set()
    seen_uncertain: set[str] = set()

    for value in decision.s:
        reason = None
        if value == parent:
            reason = "parent_not_candidate"
        elif value not in allowed:
            reason = "not_in_supplied_candidates"
        elif value in seen_same:
            reason = "duplicate_same"
        if reason is not None:
            dropped.append({"list": "s", "entity": value, "reason": reason})
        else:
            accepted.append(value)
            seen_same.add(value)

    for value in decision.u:
        reason = None
        if value == parent:
            reason = "parent_not_candidate"
        elif value not in allowed:
            reason = "not_in_supplied_candidates"
        elif value in seen_same:
            reason = "already_in_same"
        elif value in seen_uncertain:
            reason = "duplicate_uncertain"
        if reason is not None:
            dropped.append({"list": "u", "entity": value, "reason": reason})
        else:
            uncertain.append(value)
            seen_uncertain.add(value)

    conformed = Decision(s=accepted, u=uncertain)
    return conformed, {
        "changed": bool(dropped),
        "dropped": dropped,
        "policy": "Discard strings that are not exact supplied candidates; do not retry.",
    }


def answer_from_raw(raw: dict[str, Any]) -> str:
    """Recover the structured answer from a saved raw response after interruption."""
    texts: list[str] = []
    for candidate in raw.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            if part.get("text") and not part.get("thought", False):
                texts.append(part["text"])
    return "".join(texts).strip()


def is_transient(error: Exception) -> bool:
    """Limit automatic retries to recognizable service or rate-limit failures."""
    text = str(error).casefold()
    markers = (
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
    return any(marker in text for marker in markers)


def submit_or_resume_page(
    client: genai.Client,
    pass_number: int,
    parent: dict[str, Any],
    candidates: list[dict[str, str]],
) -> tuple[Decision, dict[str, Any], Path, dict[str, Any]]:
    """Submit one page once, or recover its saved response without another paid call."""
    directory = OUTPUT / (
        f"pass_{pass_number:03d}_{parent['id']}_{safe_name(parent['entity'])}"
    )
    directory.mkdir(exist_ok=True)
    prompt = build_prompt(parent["entity"], candidates)
    prompt_path = directory / "prompt_sent.txt"
    if prompt_path.exists():
        existing = prompt_path.read_text(encoding="utf-8")
        if existing != prompt:
            raise RuntimeError(f"Saved prompt differs for pass {pass_number}")
    else:
        prompt_path.write_text(prompt, encoding="utf-8", newline="\n")

    raw_path = directory / "raw_response.json"
    if raw_path.exists():
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        last_error: Exception | None = None
        for attempt in range(len(TRANSIENT_RETRY_DELAYS) + 1):
            try:
                response = client.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=Decision.model_json_schema(),
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
                last_error = error
                if attempt >= len(TRANSIENT_RETRY_DELAYS) or not is_transient(error):
                    raise
                delay = TRANSIENT_RETRY_DELAYS[attempt]
                print(
                    f"transient API failure on pass {pass_number}; retrying in {delay}s: "
                    f"{error}",
                    flush=True,
                )
                time.sleep(delay)
        else:
            raise RuntimeError("API retry loop ended unexpectedly") from last_error

    answer = answer_from_raw(raw)
    (directory / "answer_returned.json").write_text(
        answer + "\n", encoding="utf-8", newline="\n"
    )
    raw_decision = Decision.model_validate_json(answer)
    write_json(
        directory / "raw_decision.json", raw_decision.model_dump(mode="json")
    )
    decision, conformance = conform_decision(
        raw_decision, parent["entity"], candidates
    )
    write_json(directory / "conformance.json", conformance)
    validate_decision(decision, parent["entity"], candidates)
    write_json(directory / "result.json", decision.model_dump(mode="json"))
    usage = raw.get("usage_metadata") or {}
    write_json(
        directory / "page_manifest.json",
        {
            "pass": pass_number,
            "parent": parent,
            "activeCandidatesSent": candidates,
            "activeCandidateCount": len(candidates),
            "sourceCandidateCount": parent["candidates"],
            "excludedByPriorResolutions": parent["candidates"] - len(candidates),
            "promptCharacters": len(prompt),
            "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "thinkingLevel": "minimal",
            "includeThoughts": False,
            "conformance": conformance,
            "usage": usage,
        },
    )
    return decision, usage, directory, conformance


def set_pending(
    state: dict[str, Any],
    pass_number: int,
    parent: dict[str, Any],
    cursor_after: int,
    skipped: list[dict[str, Any]],
    candidates: list[dict[str, str]],
    decision: Decision,
    usage: dict[str, Any],
    directory: Path,
    conformance: dict[str, Any],
    index_by_entity: dict[str, dict[str, Any]],
) -> None:
    """Checkpoint an accepted API response before mutating the global working lists."""
    members = [parent["entity"], *decision.s]
    member_ids = [parent["id"], *[index_by_entity[tag]["id"] for tag in decision.s]]
    if len(member_ids) != len(set(member_ids)):
        raise RuntimeError("Pending cluster repeats an entity ID")
    already_assigned = set(member_ids) & set(state["assignments"])
    if already_assigned:
        raise RuntimeError(f"Pending cluster reassigns IDs: {sorted(already_assigned)}")
    state["pending"] = {
        "pass": pass_number,
        "parent": parent,
        "cursorAfter": cursor_after,
        "skipped": skipped,
        "members": members,
        "memberIds": member_ids,
        "uncertain": decision.u,
        "different": [
            row["entity"]
            for row in candidates
            if row["entity"] not in decision.s and row["entity"] not in decision.u
        ],
        "activeCandidateCount": len(candidates),
        "sourceCandidateCount": parent["candidates"],
        "usage": usage,
        "directory": directory.name,
        "conformance": conformance,
    }
    write_json(OUTPUT / "state.json", state)


def located_files(database: sqlite3.Connection, entity: str) -> list[str]:
    """Look up every source page on which an entity was originally a candidate."""
    return [
        str(row[0])
        for row in database.execute(
            "SELECT file FROM locations WHERE candidate = ?", (entity,)
        )
    ]


def purge_entities_from_lists(
    entities: list[str], own_files: list[str]
) -> dict[str, Any]:
    """Physically remove resolved rows everywhere and retire their own parent pages."""
    entity_set = set(entities)
    candidate_rows_removed = 0
    affected_pages: set[str] = set()
    database = sqlite3.connect(OUTPUT / LOCATION_DB)
    try:
        targets: set[str] = set()
        for entity in entities:
            targets.update(located_files(database, entity))
    finally:
        database.close()

    for filename in sorted(targets):
        active_path = OUTPUT / ACTIVE_LISTS / filename
        retired_path = OUTPUT / RETIRED_LISTS / filename
        path = active_path if active_path.exists() else retired_path
        if not path.exists():
            raise RuntimeError(f"Candidate-location page is missing: {filename}")
        rows = read_candidate_csv(path)
        retained = [row for row in rows if row["entity"] not in entity_set]
        removed = len(rows) - len(retained)
        if removed:
            write_candidate_csv(path, retained)
            candidate_rows_removed += removed
            affected_pages.add(filename)

    retired_pages: list[str] = []
    for filename in own_files:
        active_path = OUTPUT / ACTIVE_LISTS / filename
        retired_path = OUTPUT / RETIRED_LISTS / filename
        if active_path.exists():
            os.replace(active_path, retired_path)
            retired_pages.append(filename)
        elif not retired_path.exists():
            raise RuntimeError(f"Resolved entity page is missing: {filename}")

    return {
        "resolvedEntitiesPurged": entities,
        "sourcePagesTargeted": len(targets),
        "affectedPages": len(affected_pages),
        "candidateRowsRemoved": candidate_rows_removed,
        "ownPagesRetired": retired_pages,
    }


def complete_pending(
    state: dict[str, Any],
    index_rows: list[dict[str, Any]],
    index_by_id: dict[str, dict[str, Any]],
) -> None:
    """Finish an idempotent global purge and commit the pending cluster to state."""
    pending = state.get("pending")
    if pending is None:
        return
    members = list(pending["members"])
    member_ids = list(pending["memberIds"])
    own_files = [index_by_id[member_id]["file"] for member_id in member_ids]
    mutation = purge_entities_from_lists(members, own_files)

    pass_number = int(pending["pass"])
    parent = pending["parent"]
    for member_id, entity in zip(member_ids, members, strict=True):
        state["assignments"][member_id] = {
            "entity": entity,
            "parentId": parent["id"],
            "parentEntity": parent["entity"],
            "pass": pass_number,
        }
    cluster = {
        "pass": pass_number,
        "parentId": parent["id"],
        "parentEntity": parent["entity"],
        "parentMentions": parent["mentions"],
        "memberIds": member_ids,
        "members": members,
        "uncertain": pending["uncertain"],
    }
    state["clusters"].append(cluster)
    state["passes"].append(
        {
            **cluster,
            "different": pending["different"],
            "activeCandidateCount": pending["activeCandidateCount"],
            "sourceCandidateCount": pending["sourceCandidateCount"],
            "usage": pending["usage"],
            "directory": pending["directory"],
            "conformance": pending["conformance"],
            "globalMutation": mutation,
        }
    )
    state["skippedParentPages"].extend(pending["skipped"])
    state["cursor"] = int(pending["cursorAfter"])
    state["submittedPasses"] = pass_number
    state["pending"] = None
    write_active_roster(OUTPUT, index_rows, set(state["assignments"]))
    write_json(OUTPUT / "state.json", state)
    write_json(
        OUTPUT / pending["directory"] / "global_mutation.json", mutation
    )


def audit_global_exclusion(
    index_rows: list[dict[str, Any]], state: dict[str, Any]
) -> dict[str, Any]:
    """Scan every working CSV to prove resolved entities occur in no candidate list."""
    assigned_ids = set(state["assignments"])
    assigned_entities = {
        assignment["entity"] for assignment in state["assignments"].values()
    }
    active_files = {path.name for path in (OUTPUT / ACTIVE_LISTS).glob("*.csv")}
    retired_files = {path.name for path in (OUTPUT / RETIRED_LISTS).glob("*.csv")}
    if active_files & retired_files:
        raise RuntimeError("A candidate page exists in both active and retired folders")
    if len(active_files) + len(retired_files) != len(index_rows):
        raise RuntimeError("Working candidate page count changed")
    expected_retired = {row["file"] for row in index_rows if row["id"] in assigned_ids}
    if retired_files != expected_retired:
        raise RuntimeError("Retired pages do not exactly match assigned entities")

    candidate_rows = 0
    leaks: list[dict[str, str]] = []
    for folder in (ACTIVE_LISTS, RETIRED_LISTS):
        for path in (OUTPUT / folder).glob("*.csv"):
            rows = read_candidate_csv(path)
            candidate_rows += len(rows)
            for row in rows:
                if row["entity"] in assigned_entities:
                    leaks.append({"file": path.name, "entity": row["entity"]})
                    if len(leaks) >= 20:
                        break
            if len(leaks) >= 20:
                break
        if leaks:
            break
    if leaks:
        raise RuntimeError(f"Resolved entities remain in candidate lists: {leaks}")

    with (OUTPUT / "active_roster.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        roster = list(csv.DictReader(handle))
    if {row["id"] for row in roster} != {
        row["id"] for row in index_rows if row["id"] not in assigned_ids
    }:
        raise RuntimeError("Active roster does not match assignment state")
    return {
        "status": "passed",
        "sourcePages": len(index_rows),
        "activePages": len(active_files),
        "retiredPages": len(retired_files),
        "workingCandidateRowsAfterPurges": candidate_rows,
        "resolvedEntitiesAbsentFromEveryList": True,
        "activeRosterMatchesAssignments": True,
    }


def normalize_conformance_records(state: dict[str, Any]) -> None:
    """Backfill unchanged conformance metadata for passes saved before logging existed."""
    changed = False
    unchanged = {
        "changed": False,
        "dropped": [],
        "policy": "Discard strings that are not exact supplied candidates; do not retry.",
    }
    for row in state["passes"]:
        if "conformance" in row:
            continue
        row["conformance"] = unchanged
        directory = OUTPUT / row["directory"]
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        write_json(directory / "raw_decision.json", result)
        write_json(directory / "conformance.json", unchanged)
        page_manifest_path = directory / "page_manifest.json"
        page_manifest = json.loads(page_manifest_path.read_text(encoding="utf-8"))
        page_manifest["conformance"] = unchanged
        write_json(page_manifest_path, page_manifest)
        changed = True
    if changed:
        write_json(OUTPUT / "state.json", state)


def write_exports(
    index_rows: list[dict[str, Any]], state: dict[str, Any], audit: dict[str, Any]
) -> None:
    """Export compact review tables, totals, and the global-exclusion audit."""
    assignments = state["assignments"]
    with (OUTPUT / "resolved_entities.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ("id", "entity", "mentions", "parent_id", "parent_entity", "pass")
        )
        for row in index_rows:
            assignment = assignments.get(row["id"])
            if assignment is not None:
                writer.writerow(
                    (
                        row["id"],
                        row["entity"],
                        row["mentions"],
                        assignment["parentId"],
                        assignment["parentEntity"],
                        assignment["pass"],
                    )
                )

    with (OUTPUT / "skipped_parent_pages.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "id",
            "entity",
            "mentions",
            "resolvedToId",
            "resolvedToEntity",
            "resolvedInPass",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(state["skippedParentPages"])
    write_json(OUTPUT / "resolved_clusters.json", state["clusters"])
    write_json(OUTPUT / "GLOBAL_EXCLUSION_AUDIT.json", audit)

    prompt_tokens = sum(
        int(row["usage"].get("prompt_token_count") or 0) for row in state["passes"]
    )
    answer_tokens = sum(
        int(row["usage"].get("candidates_token_count") or 0)
        for row in state["passes"]
    )
    thought_tokens = sum(
        int(row["usage"].get("thoughts_token_count") or 0)
        for row in state["passes"]
    )
    total_tokens = sum(
        int(row["usage"].get("total_token_count") or 0) for row in state["passes"]
    )
    parent_mentions = [int(row["parentMentions"]) for row in state["passes"]]
    if not all(left >= right for left, right in zip(parent_mentions, parent_mentions[1:])):
        raise RuntimeError("Submitted parents are not frequency-descending")
    manifest = {
        "status": "completed_with_manual_review_warnings",
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "thinkingLevel": "minimal",
        "includeThoughts": False,
        "targetSubmittedPages": TARGET_SUBMISSIONS,
        "submittedPages": state["submittedPasses"],
        "parentOrder": "corpus mention frequency descending",
        "highFrequencyRule": "more than 10 mentions receives source top 50",
        "skippedResolvedParentPages": len(state["skippedParentPages"]),
        "sourceEntities": len(index_rows),
        "resolvedEntitiesRemovedFromRoster": len(assignments),
        "remainingRosterEntities": len(index_rows) - len(assignments),
        "clusters": len(state["clusters"]),
        "multiEntityClusters": sum(
            len(cluster["members"]) > 1 for cluster in state["clusters"]
        ),
        "conformanceDroppedStrings": sum(
            len(row.get("conformance", {}).get("dropped", []))
            for row in state["passes"]
        ),
        "globalExclusionAudit": audit,
        "usage": {
            "promptTokens": prompt_tokens,
            "answerTokens": answer_tokens,
            "thinkingTokens": thought_tokens,
            "totalTokens": total_tokens,
        },
    }
    write_json(OUTPUT / "run_manifest.json", manifest)

    lines = [
        "# Frequency-prioritized entity-resolution trial",
        "",
        f"- Submitted pages: {state['submittedPasses']}",
        f"- Skipped absorbed parent pages: {len(state['skippedParentPages'])}",
        f"- Entities removed from roster: {len(assignments)}",
        f"- Entities remaining: {len(index_rows) - len(assignments)}",
        f"- Global exclusion audit: {audit['status']}",
        f"- Total API tokens: {total_tokens}",
        f"- Thinking tokens: {thought_tokens}",
        "",
        "## Decisions",
        "",
    ]
    for row in state["passes"]:
        same = " | ".join(row["members"][1:]) or "none"
        uncertain = " | ".join(row["uncertain"]) or "none"
        dropped = " | ".join(
            item["entity"]
            for item in row.get("conformance", {}).get("dropped", [])
        ) or "none"
        lines.extend(
            [
                f"### Pass {row['pass']}: {row['parentEntity']} "
                f"({row['parentMentions']} mentions)",
                "",
                f"- Same identity: {same}",
                f"- Uncertain: {uncertain}",
                f"- Non-candidate strings discarded: {dropped}",
                f"- Candidates sent: {row['activeCandidateCount']} / "
                f"{row['sourceCandidateCount']}",
                f"- Candidate rows globally removed: "
                f"{row['globalMutation']['candidateRowsRemoved']}",
                "",
            ]
        )
    (OUTPUT / "REVIEW.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def main() -> None:
    """Run or resume 100 accepted pages and finish with exhaustive exclusion QA."""
    if not api_key():
        raise RuntimeError("GEMINI_API_KEY is not configured")
    index_rows = read_index()
    index_by_entity = {row["entity"]: row for row in index_rows}
    index_by_id = {row["id"]: row for row in index_rows}
    initialize_working_archive(index_rows)
    state = load_state()
    complete_pending(state, index_rows, index_by_id)
    state = load_state()
    normalize_conformance_records(state)
    state = load_state()
    client = genai.Client(api_key=api_key())

    while int(state["submittedPasses"]) < TARGET_SUBMISSIONS:
        pass_number = int(state["submittedPasses"]) + 1
        parent, cursor_after, skipped = next_parent(index_rows, state)
        candidates = active_candidates(parent, state)
        decision, usage, directory, conformance = submit_or_resume_page(
            client, pass_number, parent, candidates
        )
        set_pending(
            state,
            pass_number,
            parent,
            cursor_after,
            skipped,
            candidates,
            decision,
            usage,
            directory,
            conformance,
            index_by_entity,
        )
        complete_pending(state, index_rows, index_by_id)
        state = load_state()
        print(
            f"completed {pass_number}/{TARGET_SUBMISSIONS}: {parent['entity']} "
            f"({parent['mentions']} mentions), accepted {len(decision.s)}, "
            f"uncertain {len(decision.u)}",
            flush=True,
        )

    audit_path = OUTPUT / "GLOBAL_EXCLUSION_AUDIT.json"
    audit = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path.is_file()
        else audit_global_exclusion(index_rows, state)
    )
    write_exports(index_rows, state, audit)
    print((OUTPUT / "REVIEW.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()

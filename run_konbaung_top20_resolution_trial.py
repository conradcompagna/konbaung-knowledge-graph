#!/usr/bin/env python3
"""Resolve ten top-20 entity pages sequentially while shrinking an active roster."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict

from konbaung_gemini_summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "konbaung_entity_resolution_candidates_20260902_v2"
INDEX = SOURCE / "index.csv"
OUTPUT = ROOT / "konbaung_top20_gemini_resolution_trial_20260902_first10"
MODEL = "gemini-3.1-flash-lite"
TARGET_SUBMISSIONS = 10
MAX_OUTPUT_TOKENS = 1000


class Decision(BaseModel):
    """Keep Gemini's output to two short lists of exact candidate tags."""

    model_config = ConfigDict(extra="forbid")

    s: list[str]
    u: list[str]


def read_index() -> list[dict[str, str]]:
    """Read the stable source order and exact entity-to-file mapping."""
    with INDEX.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or list(rows[0]) != ["id", "entity", "file"]:
        raise RuntimeError("Unexpected top-20 index schema")
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("Entity IDs are not unique")
    if len({row["entity"] for row in rows}) != len(rows):
        raise RuntimeError("Entity tags are not unique")
    return rows


def read_candidates(index_row: dict[str, str]) -> list[dict[str, str]]:
    """Read one parent's ranked top-20 candidates and their two source scores."""
    path = SOURCE / "entities" / index_row["file"]
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20 or (rows and list(rows[0]) != ["entity", "emb", "char"]):
        raise RuntimeError(f"Unexpected candidate page: {path}")
    return rows


def write_json(path: Path, value: Any) -> None:
    """Write JSON atomically so an interrupted call cannot corrupt checkpoint state."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def safe_name(value: str) -> str:
    """Create a short readable Windows-safe directory component."""
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", "_", value).strip(" ._")
    return value[:60].rstrip(" ._") or "entity"


def initial_state() -> dict[str, Any]:
    """Create an empty roster ledger whose unresolved set is implicit."""
    return {
        "schemaVersion": 1,
        "source": str(SOURCE),
        "model": MODEL,
        "cursor": 0,
        "submittedPasses": 0,
        "assignments": {},
        "clusters": [],
        "skippedParentPages": [],
        "passes": [],
    }


def load_state() -> dict[str, Any]:
    """Resume a partial trial without paying for already completed pages again."""
    state_path = OUTPUT / "state.json"
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return initial_state()


def next_parent(
    index_rows: list[dict[str, str]], state: dict[str, Any]
) -> dict[str, str]:
    """Skip parents absorbed by earlier passes and return the next unresolved page."""
    while int(state["cursor"]) < len(index_rows):
        position = int(state["cursor"])
        state["cursor"] = position + 1
        row = index_rows[position]
        assignment = state["assignments"].get(row["id"])
        if assignment is None:
            return row
        state["skippedParentPages"].append(
            {
                "id": row["id"],
                "entity": row["entity"],
                "resolvedToId": assignment["parentId"],
                "resolvedToEntity": assignment["parentEntity"],
                "resolvedInPass": assignment["pass"],
            }
        )
    raise RuntimeError("The unresolved roster is exhausted")


def active_candidates(
    parent: dict[str, str],
    index_by_entity: dict[str, dict[str, str]],
    state: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Remove candidates already assigned by earlier pages before prompting Gemini."""
    active: list[dict[str, str]] = []
    excluded: list[dict[str, Any]] = []
    for candidate in read_candidates(parent):
        candidate_index = index_by_entity.get(candidate["entity"])
        if candidate_index is None:
            raise RuntimeError(f"Unknown candidate tag: {candidate['entity']}")
        assignment = state["assignments"].get(candidate_index["id"])
        if assignment is None:
            active.append(candidate)
        else:
            excluded.append(
                {
                    "entity": candidate["entity"],
                    "resolvedToEntity": assignment["parentEntity"],
                    "resolvedInPass": assignment["pass"],
                }
            )
    return active, excluded


def build_prompt(parent: str, candidates: list[dict[str, str]]) -> str:
    """Ask only whether each active neighbor is exactly the same referent as its parent."""
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
    same = decision.s
    uncertain = decision.u
    if parent in same or parent in uncertain:
        raise RuntimeError("Gemini included the parent in its candidate lists")
    if len(same) != len(set(same)) or len(uncertain) != len(set(uncertain)):
        raise RuntimeError("Gemini repeated a candidate")
    if set(same) & set(uncertain):
        raise RuntimeError("Gemini marked a candidate both same and uncertain")
    unknown = (set(same) | set(uncertain)) - allowed
    if unknown:
        raise RuntimeError(f"Gemini returned unknown candidates: {sorted(unknown)}")


def submit_page(
    client: genai.Client,
    pass_number: int,
    parent: dict[str, str],
    candidates: list[dict[str, str]],
    excluded: list[dict[str, Any]],
) -> tuple[Decision, dict[str, Any], Path]:
    """Submit one compact page and preserve its exact request, response, and usage."""
    directory = OUTPUT / (
        f"pass_{pass_number:02d}_{parent['id']}_{safe_name(parent['entity'])}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(parent["entity"], candidates)
    (directory / "prompt_sent.txt").write_text(
        prompt, encoding="utf-8", newline="\n"
    )
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
    write_json(
        directory / "raw_response.json",
        response.model_dump(mode="json", exclude_none=True),
    )
    decision = Decision.model_validate_json(response.text)
    validate_decision(decision, parent["entity"], candidates)
    write_json(directory / "result.json", decision.model_dump(mode="json"))
    usage = (
        response.usage_metadata.model_dump(mode="json")
        if response.usage_metadata is not None
        else {}
    )
    write_json(
        directory / "page_manifest.json",
        {
            "pass": pass_number,
            "parent": parent,
            "activeCandidates": candidates,
            "excludedAlreadyResolvedCandidates": excluded,
            "promptCharacters": len(prompt),
            "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "usage": usage,
        },
    )
    return decision, usage, directory


def apply_decision(
    state: dict[str, Any],
    pass_number: int,
    parent: dict[str, str],
    decision: Decision,
    index_by_entity: dict[str, dict[str, str]],
    candidates: list[dict[str, str]],
    excluded: list[dict[str, Any]],
    usage: dict[str, Any],
    directory: Path,
) -> None:
    """Remove the processed parent and accepted aliases from the unresolved roster."""
    members = [parent["entity"], *decision.s]
    member_ids = [parent["id"], *[index_by_entity[tag]["id"] for tag in decision.s]]
    for member_id, member in zip(member_ids, members, strict=True):
        if member_id in state["assignments"]:
            raise RuntimeError(f"Entity was already resolved: {member}")
        state["assignments"][member_id] = {
            "parentId": parent["id"],
            "parentEntity": parent["entity"],
            "pass": pass_number,
        }
    cluster = {
        "pass": pass_number,
        "parentId": parent["id"],
        "parentEntity": parent["entity"],
        "memberIds": member_ids,
        "members": members,
        "uncertain": decision.u,
    }
    state["clusters"].append(cluster)
    state["passes"].append(
        {
            **cluster,
            "activeCandidateCount": len(candidates),
            "excludedAlreadyResolvedCandidates": excluded,
            "different": [
                row["entity"]
                for row in candidates
                if row["entity"] not in decision.s and row["entity"] not in decision.u
            ],
            "usage": usage,
            "directory": directory.name,
        }
    )
    state["submittedPasses"] = pass_number
    write_json(OUTPUT / "state.json", state)


def write_exports(index_rows: list[dict[str, str]], state: dict[str, Any]) -> None:
    """Export the shrunken roster, assignments, skips, clusters, and a review summary."""
    assignments = state["assignments"]
    with (OUTPUT / "remaining_roster.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("id", "entity", "file"))
        for row in index_rows:
            if row["id"] not in assignments:
                writer.writerow((row["id"], row["entity"], row["file"]))

    with (OUTPUT / "resolved_entities.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("id", "entity", "parent_id", "parent_entity", "pass"))
        for row in index_rows:
            assignment = assignments.get(row["id"])
            if assignment is not None:
                writer.writerow(
                    (
                        row["id"],
                        row["entity"],
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
            "resolvedToId",
            "resolvedToEntity",
            "resolvedInPass",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(state["skippedParentPages"])

    write_json(OUTPUT / "resolved_clusters.json", state["clusters"])
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
    write_json(
        OUTPUT / "run_manifest.json",
        {
            "status": "completed_with_manual_review_warning",
            "completedAt": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "thinkingLevel": "minimal",
            "submittedPages": state["submittedPasses"],
            "skippedResolvedParentPages": len(state["skippedParentPages"]),
            "sourceEntities": len(index_rows),
            "resolvedEntitiesRemovedFromRoster": len(assignments),
            "remainingRosterEntities": len(index_rows) - len(assignments),
            "clusters": len(state["clusters"]),
            "multiEntityClusters": sum(
                len(cluster["members"]) > 1 for cluster in state["clusters"]
            ),
            "usage": {
                "promptTokens": prompt_tokens,
                "answerTokens": answer_tokens,
                "thinkingTokens": thought_tokens,
                "totalTokens": total_tokens,
            },
        },
    )

    lines = [
        "# Top-20 sequential entity-resolution trial",
        "",
        f"- Submitted pages: {state['submittedPasses']}",
        f"- Skipped parent pages already resolved earlier: {len(state['skippedParentPages'])}",
        f"- Entities removed from roster: {len(assignments)}",
        f"- Entities remaining: {len(index_rows) - len(assignments)}",
        f"- Total API tokens: {total_tokens}",
        "",
        "## Decisions",
        "",
    ]
    for row in state["passes"]:
        same = " | ".join(row["members"][1:]) or "none"
        uncertain = " | ".join(row["uncertain"]) or "none"
        lines.extend(
            [
                f"### Pass {row['pass']}: {row['parentEntity']} ({row['parentId']})",
                "",
                f"- Same identity: {same}",
                f"- Uncertain: {uncertain}",
                f"- Active candidates sent: {row['activeCandidateCount']}",
                f"- Already-resolved candidates filtered: {len(row['excludedAlreadyResolvedCandidates'])}",
                "",
            ]
        )
    (OUTPUT / "REVIEW.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def main() -> None:
    """Submit ten actual unresolved pages, checkpointing the roster after every call."""
    if not api_key():
        raise RuntimeError("GEMINI_API_KEY is not configured")
    index_rows = read_index()
    index_by_entity = {row["entity"]: row for row in index_rows}
    OUTPUT.mkdir(exist_ok=True)
    state = load_state()
    client = genai.Client(api_key=api_key())

    while int(state["submittedPasses"]) < TARGET_SUBMISSIONS:
        pass_number = int(state["submittedPasses"]) + 1
        parent = next_parent(index_rows, state)
        candidates, excluded = active_candidates(
            parent, index_by_entity, state
        )
        decision, usage, directory = submit_page(
            client, pass_number, parent, candidates, excluded
        )
        apply_decision(
            state,
            pass_number,
            parent,
            decision,
            index_by_entity,
            candidates,
            excluded,
            usage,
            directory,
        )

    write_exports(index_rows, state)
    print((OUTPUT / "REVIEW.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

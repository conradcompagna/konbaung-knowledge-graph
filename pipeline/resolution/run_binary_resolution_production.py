#!/usr/bin/env python3
"""Prepare or explicitly execute the binary full entity-resolution production pass."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from pipeline.resolution import frequency_candidates as engine
from pipeline.extraction.summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parents[2]
PLAN_OUTPUT = ROOT / "konbaung_binary_resolution_production_plan_20260902"
LIVE_OUTPUT = ROOT / "konbaung_binary_resolution_production_20260902"
MINIMUM_PARENT_MENTIONS = 2


class BinaryDecision(BaseModel):
    """Return only accepted exact tags; every omitted supplied tag is a no decision."""

    model_config = ConfigDict(extra="forbid")

    y: list[str]

    @property
    def s(self) -> list[str]:
        """Expose accepted tags to the shared assignment engine."""
        return self.y

    @property
    def u(self) -> list[str]:
        """Expose an always-empty compatibility view because uncertainty is removed."""
        return []


def parse_args() -> argparse.Namespace:
    """Require an explicit execution flag so the default command cannot call Gemini."""
    parser = argparse.ArgumentParser(
        description="Prepare the binary production plan; pass --execute to call Gemini."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Start or resume paid Gemini production calls.",
    )
    return parser.parse_args()


def binary_prompt(parent: str, candidates: list[dict[str, str]]) -> str:
    """Request one token-efficient yes list with omission defined explicitly as no."""
    candidate_lines = [f"{row['entity']}\t{row['emb']}\t{row['char']}" for row in candidates]
    return f"""Decide whether each candidate tag denotes exactly the same underlying entity as PARENT.

Identity must be strict. Say yes only for spelling, transliteration, capitalization,
separator, word-order, or genuine title/name variants of the same referent. Say no for
merely related things: person/forces/family/reign, ruler/dynasty/state,
place/fort/unit/residents, title/office-holder, singular/plural class, or part/whole.
For a generic parent, say yes only to equivalent generic expressions, never named
instances. emb and char are retrieval clues, not proof. Use reliable historical
knowledge only when confident. There is no uncertain state: make a binary decision.

Return only JSON {{"y":[...]}}. `y` contains exact supplied candidate strings whose
answer is YES. Every supplied candidate omitted from `y` is NO. Do not include PARENT.

PARENT
{parent}

CANDIDATES\tentity,emb,char
{chr(10).join(candidate_lines)}
"""


def validate_binary_decision(
    decision: BinaryDecision, parent: str, candidates: list[dict[str, str]]
) -> None:
    """Reject parent-valued, repeated, or invented yes decisions."""
    allowed = {row["entity"] for row in candidates}
    if parent in decision.y:
        raise RuntimeError("Gemini included the parent in its yes list")
    if len(decision.y) != len(set(decision.y)):
        raise RuntimeError("Gemini repeated a yes candidate")
    unknown = set(decision.y) - allowed
    if unknown:
        raise RuntimeError(f"Gemini returned unknown candidates: {sorted(unknown)}")


def conform_binary_decision(
    decision: BinaryDecision, parent: str, candidates: list[dict[str, str]]
) -> tuple[BinaryDecision, dict[str, Any]]:
    """Discard invalid yes strings while preserving them in conformance audit data."""
    allowed = {row["entity"] for row in candidates}
    accepted: list[str] = []
    seen: set[str] = set()
    dropped: list[dict[str, str]] = []
    for value in decision.y:
        reason = None
        if value == parent:
            reason = "parent_not_candidate"
        elif value not in allowed:
            reason = "not_in_supplied_candidates"
        elif value in seen:
            reason = "duplicate_yes"
        if reason is not None:
            dropped.append({"list": "y", "entity": value, "reason": reason})
        else:
            accepted.append(value)
            seen.add(value)
    return BinaryDecision(y=accepted), {
        "changed": bool(dropped),
        "dropped": dropped,
        "policy": "Discard invalid yes strings; every valid omitted candidate remains no.",
    }


def configure_engine() -> None:
    """Configure the audited roster/exclusion engine for the new binary production run."""
    engine.OUTPUT = LIVE_OUTPUT
    engine.Decision = BinaryDecision
    engine.build_prompt = binary_prompt
    engine.validate_decision = validate_binary_decision
    engine.conform_decision = conform_binary_decision


def eligible_rows(index_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select frequency-ordered parent pages with at least two corpus mentions."""
    return [row for row in index_rows if row["mentions"] >= MINIMUM_PARENT_MENTIONS]


def prepare_plan(index_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Materialize the exact zero-call production inventory and binary schema."""
    eligible = eligible_rows(index_rows)
    PLAN_OUTPUT.mkdir(exist_ok=True)
    with (PLAN_OUTPUT / "eligible_parent_pages.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("order", "id", "entity", "mentions", "candidates", "file"))
        for order, row in enumerate(eligible, start=1):
            writer.writerow(
                (
                    order,
                    row["id"],
                    row["entity"],
                    row["mentions"],
                    row["candidates"],
                    row["file"],
                )
            )

    schema = BinaryDecision.model_json_schema()
    engine.write_json(PLAN_OUTPUT / "response_schema.json", schema)
    (PLAN_OUTPUT / "prompt_template.txt").write_text(
        binary_prompt("<PARENT>", [{"entity": "<CANDIDATE>", "emb": "<EMB>", "char": "<CHAR>"}]),
        encoding="utf-8",
        newline="\n",
    )
    frequent = sum(row["mentions"] > 10 for row in eligible)
    manifest = {
        "status": "ready_not_executed",
        "apiCallsMade": 0,
        "model": engine.MODEL,
        "thinkingLevel": "minimal",
        "includeThoughts": False,
        "decisionMode": "binary",
        "responseShape": {"y": ["exact supplied YES tags"]},
        "noRule": "Every supplied candidate omitted from y is NO.",
        "uncertainState": False,
        "parentEligibility": "corpus mentions > 1",
        "initialEligibleParentPages": len(eligible),
        "maximumGeminiCalls": len(eligible),
        "actualGeminiCalls": (
            "Cannot be known before resolution; accepted later parents are skipped."
        ),
        "excludedSingletonParentPages": len(index_rows) - len(eligible),
        "eligibleTop50Pages": frequent,
        "eligibleTop20Pages": len(eligible) - frequent,
        "singletonCandidatesRemainEligibleForAbsorption": True,
        "globalExclusion": (
            "Every yes-resolved tag is removed from the roster, its own page, and "
            "every other materialized candidate list before the next prompt."
        ),
    }
    engine.write_json(PLAN_OUTPUT / "plan_manifest.json", manifest)
    (PLAN_OUTPUT / "README.md").write_text(
        "# Binary entity-resolution production plan\n\n"
        f"There are **{len(eligible):,}** initially eligible parent pages with more "
        "than one corpus mention. This is the maximum call count; the actual count "
        "will be lower whenever an accepted alias would otherwise become a later "
        "parent. The default runner invocation only refreshes this plan and cannot "
        "call Gemini. Starting or resuming production requires the explicit "
        "`--execute` flag.\n\n"
        "The API response contains only `y`. Membership means YES; omission means NO. "
        "There is no uncertain state. Singleton tags can be accepted as aliases but "
        "never receive their own parent prompt.\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def next_eligible_parent(
    index_rows: list[dict[str, Any]], state: dict[str, Any]
) -> tuple[dict[str, Any], int, list[dict[str, Any]]] | None:
    """Return the next unresolved eligible parent or stop at the singleton boundary."""
    cursor = int(state["cursor"])
    skipped: list[dict[str, Any]] = []
    while cursor < len(index_rows):
        row = index_rows[cursor]
        if row["mentions"] < MINIMUM_PARENT_MENTIONS:
            return None
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
    return None


def write_binary_review(state: dict[str, Any], eligible_count: int) -> None:
    """Replace the shared review with binary terminology after production completes."""
    lines = [
        "# Binary frequency-prioritized entity resolution",
        "",
        f"- Initially eligible parent pages: {eligible_count}",
        f"- Gemini pages actually submitted: {state['submittedPasses']}",
        f"- Absorbed eligible parent pages skipped: {len(state['skippedParentPages'])}",
        f"- Entities assigned: {len(state['assignments'])}",
        "- Decision mode: YES/NO; no uncertain state",
        "",
        "## Decisions",
        "",
    ]
    for row in state["passes"]:
        yes = " | ".join(row["members"][1:]) or "none"
        no_count = len(row["different"])
        lines.extend(
            [
                f"### Pass {row['pass']}: {row['parentEntity']} ({row['parentMentions']} mentions)",
                "",
                f"- YES: {yes}",
                f"- NO: {no_count} omitted supplied candidates",
                f"- Candidates sent: {row['activeCandidateCount']} / {row['sourceCandidateCount']}",
                "",
            ]
        )
    (LIVE_OUTPUT / "REVIEW.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def execute(index_rows: list[dict[str, Any]]) -> None:
    """Run or resume all eligible binary pages only after explicit user authorization."""
    if not api_key():
        raise RuntimeError("GEMINI_API_KEY is not configured")
    configure_engine()
    index_by_entity = {row["entity"]: row for row in index_rows}
    index_by_id = {row["id"]: row for row in index_rows}
    engine.initialize_working_archive(index_rows)
    state = engine.load_state()
    engine.complete_pending(state, index_rows, index_by_id)
    state = engine.load_state()
    client = engine.genai.Client(api_key=api_key())

    while True:
        selection = next_eligible_parent(index_rows, state)
        if selection is None:
            break
        parent, cursor_after, skipped = selection
        pass_number = int(state["submittedPasses"]) + 1
        candidates = engine.active_candidates(parent, state)
        decision, usage, directory, conformance = engine.submit_or_resume_page(
            client, pass_number, parent, candidates
        )
        engine.set_pending(
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
        engine.complete_pending(state, index_rows, index_by_id)
        state = engine.load_state()
        print(
            f"completed {pass_number}: {parent['entity']} "
            f"({parent['mentions']} mentions), yes {len(decision.y)}",
            flush=True,
        )

    audit = engine.audit_global_exclusion(index_rows, state)
    engine.write_exports(index_rows, state, audit)
    manifest_path = LIVE_OUTPUT / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "completed_pending_manual_semantic_review",
            "decisionMode": "binary",
            "uncertainState": False,
            "parentEligibility": "corpus mentions > 1",
            "initialEligibleParentPages": len(eligible_rows(index_rows)),
            "actualGeminiCalls": state["submittedPasses"],
        }
    )
    engine.write_json(manifest_path, manifest)
    write_binary_review(state, len(eligible_rows(index_rows)))


def main() -> None:
    """Prepare the zero-call plan by default and execute only with explicit opt-in."""
    configure_engine()
    index_rows = engine.read_index()
    manifest = prepare_plan(index_rows)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    args = parse_args()
    if args.execute:
        execute(index_rows)


if __name__ == "__main__":
    main()

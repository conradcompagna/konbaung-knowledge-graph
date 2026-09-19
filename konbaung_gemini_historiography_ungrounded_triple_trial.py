#!/usr/bin/env python3
"""Historiography-heavy, ungrounded S-P-O extraction from Konbaung reader pages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel

from konbaung_gemini_summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "konbaung_reader_app" / "static" / "data" / "konbaung"
OUTPUT_ROOT = ROOT / "konbaung_historiography_ungrounded_trials"
OLD_PROMPT_PATH = (
    ROOT / "konbaung_translated_sentence_annotation_prompt.backup_shortened_20260713.md"
)
MODEL = "gemini-3.1-flash-lite"


class AnalyticalTriple(BaseModel):
    subject: str
    predicate: str
    object: str


class SentenceDecision(BaseModel):
    sid: str
    decision: Literal["annotate", "skip"]
    justification: str
    triples: list[AnalyticalTriple]


class TrialResult(BaseModel):
    sentences: list[SentenceDecision]


TASK_INSTRUCTION = """You are given an existing interpretive summary of one page and ordered sentence records containing each complete Burmese sentence and its faithful English translation. Do not create another summary or translate the sentences again. Use the supplied summary as an analytical coverage map: identify its historical arguments, then extract the sentence-level relations that support, exemplify, qualify, or explain those arguments.

This experiment is deliberately ungrounded. Do not quote or return Burmese spans. Do not return English glosses, evidence spans, entity tags, source markers, confidence scores, or offsets. Instead, express each historically meaningful relation directly as a normalized analytical subject-predicate-object combination:

- `subject`: a concise English identification of the actor, institution, group, document, place, resource, or other source of the relation;
- `predicate`: a concise analytical graph-edge label, preferably in `ALL_CAPS_WITH_UNDERSCORES`;
- `object`: a concise English identification of the relation's endpoint, target, recipient, status, action-result, complement, resource, place, or proposition.

The three fields must form a coherent, historically defensible proposition. Normalize omitted participants when the Burmese grammar, translation, page context, or chronicle convention makes them reasonably clear. Analytical interpretation and inference are allowed, but do not invent events or relations unsupported by the supplied page.

Return every supplied `sid` exactly once and preserve sentence order. Each sentence record must contain `sid`, `decision`, `justification`, and `triples`.

- Use `decision: "annotate"` when the sentence contains historically useful evidence for the sociology or history of Konbaung power. Set `justification` to an empty string and return one or more triples.
- Use `decision: "skip"` when the sentence is a fragment, an unusable continuation, purely formulaic material, or contains no historically useful relation. Return `triples: []` and give exactly one concise English sentence in `justification`.

A quotation, letter, proclamation, or reported statement is not automatically a narrator-verified fact, but it may be historically useful evidence of a claim, command, demand, political position, administrative practice, or relationship. Represent the evidentiary status in the triple when necessary instead of silently converting an attributed claim into fact.

Be thorough without padding. Extract all distinct historically relevant relations in every annotated sentence, especially all relations that provide evidence for the supplied summary. Longer sentences commonly need several triples when they contain distinct actors, commands, grants, offices, movements, conflicts, resources, ritual acts, institutional acts, or consequences. Do not omit a major proposition merely because another triple from the same sentence already supports the same summary claim. Conversely, do not atomize minor syntax or incidental detail that adds no useful relation to a historical triples database.

Return only the requested JSON through the supplied schema."""


def historiography_prompt() -> str:
    """Reuse the analytical frame and reading guidance from the old prompt."""
    old_prompt = OLD_PROMPT_PATH.read_text(encoding="utf-8")
    task_marker = (
        "You are given an existing interpretive summary of the page and ordered pairs"
    )
    reading_marker = "What to look for in Burmese chronicle prose:"
    grounding_marker = "Be accurate in pointing to explicit Burmese spans;"

    frame, remainder = old_prompt.split(task_marker, maxsplit=1)
    _, reading_section = remainder.split(reading_marker, maxsplit=1)
    reading_section, _ = reading_section.split(grounding_marker, maxsplit=1)
    frame = frame.replace(
        "extract span-grounded triples related to the sociology of power",
        "extract analytical subject-predicate-object triples related to the sociology of power",
    )
    return (
        frame.rstrip()
        + "\n\n"
        + TASK_INSTRUCTION.strip()
        + "\n\n"
        + reading_marker
        + reading_section.rstrip()
        + "\n\nCoverage, historical usefulness, and depth of interpretation are the most important factors."
    )


def read_page(volume: int, page_number: int) -> dict[str, Any]:
    path = DATA_ROOT / "pages" / f"vol{volume}" / f"{page_number:04d}.json"
    if not path.exists():
        raise FileNotFoundError(f"Unknown reader page: volume {volume}, page {page_number}")
    return json.loads(path.read_text(encoding="utf-8"))


def sentence_pairs(page: dict[str, Any]) -> list[dict[str, str]]:
    pairs = [
        {
            "sid": str(sentence["id"]),
            "my": str(sentence["text"]),
            "en": str(sentence["translation"]),
        }
        for sentence in page["sentences"]
    ]
    if not pairs or any(not pair["my"] or not pair["en"] for pair in pairs):
        raise ValueError("Selected page lacks complete Burmese/translation sentence pairs")
    return pairs


def build_prompt(summary: str, pairs: list[dict[str, str]]) -> str:
    supplied_page = {
        "page_summary": summary,
        "sentences": pairs,
    }
    return (
        historiography_prompt()
        + "\n\nSUPPLIED PAGE:\n"
        + json.dumps(supplied_page, ensure_ascii=False, indent=2)
        + "\n"
    )


def validate(pairs: list[dict[str, str]], result: TrialResult) -> dict[str, Any]:
    errors: list[str] = []
    expected_ids = [pair["sid"] for pair in pairs]
    observed_ids = [group.sid for group in result.sentences]
    if observed_ids != expected_ids:
        errors.append(f"Sentence IDs/order mismatch: expected {expected_ids}, got {observed_ids}")

    triple_count = 0
    annotated_count = 0
    skipped_count = 0
    forbidden_values = {"", "null", "none", "n/a", "-"}
    for group in result.sentences:
        justification = group.justification.strip()
        if group.decision == "skip":
            skipped_count += 1
            if group.triples:
                errors.append(f"{group.sid}: skipped sentence returned triples")
            if not justification:
                errors.append(f"{group.sid}: skipped sentence lacks a justification")
            elif "\n" in justification:
                errors.append(f"{group.sid}: skip justification is not one line")
            continue

        annotated_count += 1
        if justification:
            errors.append(f"{group.sid}: annotated sentence has a justification")
        if not group.triples:
            errors.append(f"{group.sid}: annotated sentence returned no triples")
        triple_count += len(group.triples)
        for index, triple in enumerate(group.triples, start=1):
            for field in ("subject", "predicate", "object"):
                value = str(getattr(triple, field)).strip()
                if value.lower() in forbidden_values:
                    errors.append(
                        f"{group.sid} triple {index}: invalid {field} value"
                    )

    return {
        "accepted": not errors,
        "errors": errors,
        "sentenceCount": len(result.sentences),
        "annotatedSentenceCount": annotated_count,
        "skippedSentenceCount": skipped_count,
        "tripleCount": triple_count,
    }


def review_markdown(
    pairs: list[dict[str, str]],
    result: TrialResult,
    validation: dict[str, Any],
) -> str:
    groups = {group.sid: group for group in result.sentences}
    lines = [
        "# Historiography-heavy ungrounded triple trial",
        "",
        "VALID" if validation["accepted"] else "FLAGGED",
        "",
    ]
    for error in validation["errors"]:
        lines.append(f"- {error}")
    if validation["errors"]:
        lines.append("")

    for pair in pairs:
        lines.extend([f"## {pair['sid']}", "", pair["my"], "", pair["en"], ""])
        group = groups.get(pair["sid"])
        if not group:
            continue
        lines.extend([f"**Decision:** {group.decision}", ""])
        if group.decision == "skip":
            lines.extend([f"**Justification:** {group.justification}", ""])
        else:
            for triple in group.triples:
                lines.append(
                    f"- **{triple.subject}** — `{triple.predicate}` → **{triple.object}**"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run(
    volume: int,
    page_number: int,
    output_name: str,
    thinking_level: str = "low",
) -> Path:
    page = read_page(volume, page_number)
    pairs = sentence_pairs(page)
    summary = str(page.get("summary", "")).strip()
    prompt = build_prompt(summary, pairs)

    output_dir = OUTPUT_ROOT / f"vol{volume}" / f"page_{page_number:04d}" / output_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "input.json").write_text(
        json.dumps(
            {"page_summary": summary, "sentences": pairs},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TrialResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=65536,
            thinking_config=types.ThinkingConfig(
                thinking_level={
                    "minimal": types.ThinkingLevel.MINIMAL,
                    "low": types.ThinkingLevel.LOW,
                    "medium": types.ThinkingLevel.MEDIUM,
                    "high": types.ThinkingLevel.HIGH,
                }[thinking_level],
                include_thoughts=True,
            ),
        ),
    )

    answer_parts: list[str] = []
    thought_parts: list[str] = []
    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts or []:
            text = part.text or ""
            if not text:
                continue
            if part.thought:
                thought_parts.append(text)
            else:
                answer_parts.append(text)
    raw = "".join(answer_parts).strip() or response.text or ""
    thoughts = "\n\n".join(thought_parts).strip()
    (output_dir / "raw_response.json").write_text(raw, encoding="utf-8")
    (output_dir / "thoughts.md").write_text(
        ("# Gemini thought summary\n\n" + thoughts + "\n")
        if thoughts
        else "# Gemini thought summary\n\nNo thought summary was returned.\n",
        encoding="utf-8",
    )

    result = TrialResult.model_validate_json(raw)
    validation = validate(pairs, result)
    usage = (
        response.usage_metadata.model_dump(mode="json")
        if response.usage_metadata
        else {}
    )
    (output_dir / "result.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "volume": volume,
                "page": page_number,
                "thinkingLevel": thinking_level,
                "usage": usage,
                "thoughts": thoughts,
                "validation": validation,
                "response": result.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "review.md").write_text(
        review_markdown(pairs, result, validation),
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int, required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument(
        "--output-name",
        default="historiography_ungrounded_low_thinking_01",
    )
    parser.add_argument(
        "--thinking-level",
        choices=("minimal", "low", "medium", "high"),
        default="low",
    )
    args = parser.parse_args()
    print(run(args.volume, args.page, args.output_name, args.thinking_level))


if __name__ == "__main__":
    main()

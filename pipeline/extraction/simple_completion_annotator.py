#!/usr/bin/env python3
"""Simplified third-pass Gemini extraction over uncovered page prose."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from build_konbaung_reader_data import span_candidates
from konbaung_gemini_cite_sources_annotator import ENV_FILE, MODEL, ORIGINAL_PROMPT_PATH, PROMPT


ROOT = Path(__file__).resolve().parent
READER_DATA = ROOT / "konbaung_reader_app" / "static" / "data" / "konbaung" / "pages"
SOURCE_ROOT = ROOT / "konbaung_viable_pages_manual_ranges"
OUTPUT_ROOT = ROOT / "konbaung_remainder_completion_test"


class CompletionTriple(BaseModel):
    e: str = Field(description="Exact contiguous evidence span from a target-page remainder block.")
    s: str
    sg: str
    st: str
    p: str
    o: str
    og: str
    ot: str
    d: str = Field(description="Exact Burmese date/time span, or empty string.")
    dg: str = Field(description="English date/time gloss, or empty string.")
    l: str = Field(description="Exact Burmese location span, or empty string.")
    lg: str = Field(description="English location gloss, or empty string.")
    q: str = Field(description="Exact Burmese quantity span, or empty string.")
    qg: str = Field(description="English quantity gloss, or empty string.")


class CompletionResult(BaseModel):
    T: list[CompletionTriple]
    remaining_characters: int = Field(
        description="Approximate number of useful target-page characters not covered by existing or new evidence spans."
    )


THIRD_PASS_PROMPT = """<NEW_THIRD_PASS_TASK>
This page has already been annotated twice. The first two prompts above are read-only context. Your new task is to complete their coverage.

Read TARGET_PAGE_TEXT and the existing triples. Find the useful page material that is not already included in an existing evidence span. Add new triples whose evidence spans cover that remaining material. Keep going until roughly 90% or more of the useful main text is covered by either an existing or a new evidence span.

Cover the complete useful passage around each new claim, including its predicate and relevant supporting details. Extract the subject, relation, object, English glosses, and any explicit date/time, location, or quantity metadata. Do not duplicate existing triples.

Ignore page headers, footers, page numbers, citations, footnotes, publisher text, and OCR debris. Do not return or explain ignored material. Return only the new triples and your estimate of how many useful TARGET_PAGE_TEXT characters remain outside all existing and new evidence spans.
</NEW_THIRD_PASS_TASK>"""


def api_key() -> str:
    direct = os.getenv("GEMINI_API_KEY", "").strip()
    if direct:
        return direct
    for raw in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        if raw.strip().startswith("GEMINI_API_KEY="):
            return raw.split("=", 1)[1].strip().strip("\"'")
    return ""


def utf16_to_codepoint(text: str, offset: int) -> int:
    return len(text.encode("utf-16-le")[: offset * 2].decode("utf-16-le", errors="ignore"))


def merge_intervals(intervals: list[tuple[int, int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def remainder_blocks(page: dict[str, Any]) -> list[dict[str, Any]]:
    text = page["canonicalText"]
    intervals = [
        (
            utf16_to_codepoint(text, int(item["evidence"]["startUtf16"])),
            utf16_to_codepoint(text, int(item["evidence"]["endUtf16"])),
        )
        for item in page["annotations"]
    ]
    merged = merge_intervals(intervals)
    blocks: list[dict[str, Any]] = []
    cursor = 0
    for start, end in merged:
        if cursor < start and text[cursor:start].strip():
            blocks.append(
                {
                    "id": f"R{len(blocks) + 1}",
                    "start": cursor,
                    "end": start,
                    "text": text[cursor:start],
                }
            )
        cursor = max(cursor, end)
    if cursor < len(text) and text[cursor:].strip():
        blocks.append(
            {"id": f"R{len(blocks) + 1}", "start": cursor, "end": len(text), "text": text[cursor:]}
        )
    return blocks


def page_path(volume: int, page: int) -> Path:
    return READER_DATA / f"vol{volume}" / f"{page:04d}.json"


def source_path(volume: int, page: int) -> Path:
    return SOURCE_ROOT / f"konbaung_vol{volume}" / "pages" / f"page_{page:04d}.txt"


def exact_source_text(volume: int, page: int) -> str:
    raw = source_path(volume, page).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    return raw.decode("utf-8")


def compact_existing(page: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": item["id"],
            "e": item["evidence"]["canonicalText"],
            "s": item["subject"]["text"],
            "sg": item["subject"].get("gloss"),
            "st": item["subject"].get("type"),
            "p": item["relation"]["rawLabel"],
            "o": item["object"]["text"],
            "og": item["object"].get("gloss"),
            "ot": item["object"].get("type"),
        }
        for item in page["annotations"]
    ]


def build_prompt(volume: int, page_number: int) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    page = json.loads(page_path(volume, page_number).read_text(encoding="utf-8"))
    blocks = remainder_blocks(page)
    original_prompt = ORIGINAL_PROMPT_PATH.read_text(encoding="utf-8-sig")
    prompt = f"""<ORIGINAL_FIRST_PASS_PROMPT_READ_ONLY>
{original_prompt}
</ORIGINAL_FIRST_PASS_PROMPT_READ_ONLY>

<SECOND_PASS_PROMPT_READ_ONLY>
{PROMPT}
</SECOND_PASS_PROMPT_READ_ONLY>

{THIRD_PASS_PROMPT}

<TARGET_PAGE_TEXT>
{page["canonicalText"]}
</TARGET_PAGE_TEXT>

<EXISTING_SUMMARY_READ_ONLY>
{page.get("summary") or ""}
</EXISTING_SUMMARY_READ_ONLY>

<EXISTING_TRIPLES_AND_EVIDENCE_READ_ONLY>
{json.dumps(compact_existing(page), ensure_ascii=False, indent=2)}
</EXISTING_TRIPLES_AND_EVIDENCE_READ_ONLY>
"""
    return prompt, page, blocks


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", "", text)


def contained(text: str, span: str) -> bool:
    return bool(span) and normalize_whitespace(span) in normalize_whitespace(text)


def validate(
    result: CompletionResult,
    blocks: list[dict[str, Any]],
    target_text: str,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    warnings: list[str] = []
    coverage = {item["id"]: [False] * len(item["text"]) for item in blocks}
    placements: list[dict[str, Any]] = []

    output_spans = [("T", index, item.e) for index, item in enumerate(result.T, start=1)]
    for kind, index, evidence in output_spans:
        candidates: list[tuple[int, str, dict[str, int]]] = []
        for block in blocks:
            for candidate in span_candidates(block["text"], evidence):
                new_characters = sum(
                    not coverage[block["id"]][position]
                    for position in range(candidate["start"], candidate["end"])
                    if not block["text"][position].isspace()
                )
                candidates.append((new_characters, block["id"], candidate))
        if not candidates:
            warnings.append(f"{kind}[{index}]: evidence is not inside any remainder block")
            placements.append({"kind": kind, "index": index, "block": None})
            continue
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]["start"]))
        _, block_id, selected = candidates[0]
        block_text = next(item["text"] for item in blocks if item["id"] == block_id)
        overlap = sum(
            coverage[block_id][position]
            for position in range(selected["start"], selected["end"])
            if not block_text[position].isspace()
        )
        if overlap:
            warnings.append(
                f"{kind}[{index}]: evidence overlaps {overlap} already assigned characters"
            )
        for position in range(selected["start"], selected["end"]):
            if not block_text[position].isspace():
                coverage[block_id][position] = True
        placements.append(
            {
                "kind": kind,
                "index": index,
                "block": block_id,
                "start": selected["start"],
                "end": selected["end"],
            }
        )

    for index, triple in enumerate(result.T, start=1):
        for field in ("s", "o"):
            value = getattr(triple, field)
            if not contained(triple.e, value) and not contained(target_text, value):
                warnings.append(
                    f"T[{index}].{field}: endpoint is absent from evidence and target page"
                )
        for field in ("d", "l", "q"):
            value = getattr(triple, field)
            if value and not contained(triple.e, value):
                warnings.append(f"T[{index}].{field}: modifier is not in target evidence")
    missing_by_block: dict[str, int] = {}
    for block in blocks:
        missing = sum(
            not coverage[block["id"]][position]
            for position, char in enumerate(block["text"])
            if not char.isspace()
        )
        if missing:
            missing_by_block[block["id"]] = missing
            warnings.append(f"{block['id']}: {missing} non-whitespace characters remain unassigned")
    coverage_report = {
        "complete": not missing_by_block,
        "missing_by_block": missing_by_block,
        "assigned_characters": sum(sum(flags) for flags in coverage.values()),
        "required_characters": sum(
            sum(not char.isspace() for char in block["text"]) for block in blocks
        ),
    }
    return warnings, placements, coverage_report


def run(volume: int, page_number: int, output_name: str) -> Path:
    prompt, page, blocks = build_prompt(volume, page_number)
    output_dir = OUTPUT_ROOT / f"vol{volume}" / f"page_{page_number:04d}" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_sent.txt").write_text(prompt, encoding="utf-8")
    write_payload = {
        "page": page_number,
        "volume": volume,
        "blocks": blocks,
        "coverage": 1 - sum(len(item["text"]) for item in blocks) / len(page["canonicalText"]),
    }
    (output_dir / "mechanical_remainder.json").write_text(
        json.dumps(write_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CompletionResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=10000,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    result = CompletionResult.model_validate_json(response.text)
    warnings, placements, coverage_report = validate(result, blocks, page["canonicalText"])
    usage = response.usage_metadata.model_dump() if response.usage_metadata else {}
    output = {
        "valid": not warnings,
        "warnings": warnings,
        "usage": usage,
        "coverage": coverage_report,
        "result": result.model_dump(),
    }
    (output_dir / "result.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    triple_placements = {item["index"]: item for item in placements if item["kind"] == "T"}
    postprocessed = {
        "T": [
            {
                **triple.model_dump(),
                "block": triple_placements.get(index, {}).get("block"),
                "s_inferred": not contained(triple.e, triple.s),
                "o_inferred": not contained(triple.e, triple.o),
            }
            for index, triple in enumerate(result.T, start=1)
        ],
        "model_remaining_characters": result.remaining_characters,
        "coverage": coverage_report,
    }
    (output_dir / "postprocessed.json").write_text(
        json.dumps(postprocessed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int, required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--output-name", default="third_pass_01")
    args = parser.parse_args()
    print(run(args.volume, args.page, args.output_name))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract remaining research-relevant evidence for quantitative historical analysis."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from calculate_debris_adjusted_coverage import debris_positions
from konbaung_gemini_cite_sources_annotator import ENV_FILE, MODEL, ORIGINAL_PROMPT_PATH
from konbaung_gemini_summary_claim_completion_annotator import api_key, utf16_to_codepoint


ROOT = Path(__file__).resolve().parent
READER_DATA = ROOT / "konbaung_reader_app" / "static" / "data" / "konbaung" / "pages"
OUTPUT_ROOT = ROOT / "konbaung_quantitative_fourth_pass_test"
COVERAGE_PATH = ROOT / "konbaung_fourth_pass_below_80pct.json"


class FourthPassTriple(BaseModel):
    e: str = Field(description="Exact contiguous Burmese evidence block from TARGET_PAGE_TEXT.")
    s: str = Field(description="Burmese subject span from TARGET_PAGE_TEXT.")
    sg: str = Field(description="Short English subject gloss.")
    st: str = Field(description="Concise open-coded subject type.")
    p: str = Field(description="Concise open-coded relation.")
    o: str = Field(description="Burmese object span from TARGET_PAGE_TEXT.")
    og: str = Field(description="Short English object gloss.")
    ot: str = Field(description="Concise open-coded object type.")
    d: str = Field(description="Exact Burmese date/time span, or empty string.")
    dg: str = Field(description="English date/time gloss, or empty string.")
    l: str = Field(description="Exact Burmese location span, or empty string.")
    lg: str = Field(description="English location gloss, or empty string.")
    q: str = Field(description="Exact Burmese quantity span, or empty string.")
    qg: str = Field(description="English quantity gloss, or empty string.")


class FourthPassResult(BaseModel):
    T: list[FourthPassTriple]


FOURTH_PASS_PROMPT = """<NEW_FOURTH_PASS_TASK>
This is the fourth and final annotation pass over a Burmese royal chronicle page. The original first-pass prompt above is read-only historiographical and research-question context. Follow only this task's output instructions.

Most of the page has already been tagged in earlier passes. The purpose of this new pass is to sift through any remaining unannotated sections and ensure exhaustive coverage of all triples that help to answer the research question.

Recover useful evidence relevant to the original research question wherever it remains: institutions, authority, administration, hierarchy, status, kinship, patronage, tribute, taxation, labor, warfare, diplomacy, religion, ritual, legitimacy, mobility, communication, resources, quantities, dates, places, and historically meaningful actions or conditions.

Do not, however, pad your output with useless or irrelevant details.

Read the list of existing triples first to establish what evidence has already been collected from the page.

However, do not produce redundant triples. Ensure that all new additions illuminate genuinely new relations rather than restating an existing triple using the exact same evidence base.

Then, read the original page with unannotated spans marked out in context. Focus new evidence on the marked passages, but an evidence block may extend into adjoining context when necessary to express a coherent claim.

Preserve meaningful dates, locations, and quantities in the marked metadata categories. Omit headers, footers, page numbers, citations, footnotes, publisher text, isolated punctuation, OCR debris, and other fragments.

The page's current evidence coverage and uncovered percentage are supplied below. Add enough useful evidence to make coverage exceed 80 percent wherever the source permits.
</NEW_FOURTH_PASS_TASK>"""


def page_path(volume: int, page: int) -> Path:
    return READER_DATA / f"vol{volume}" / f"{page:04d}.json"


def load_page(volume: int, page: int) -> dict[str, Any]:
    return json.loads(page_path(volume, page).read_text(encoding="utf-8-sig"))


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def coverage_record(page: dict[str, Any]) -> dict[str, Any]:
    text = page["canonicalText"]
    evidence_intervals = merge_intervals(
        [
            (
                utf16_to_codepoint(text, int(item["evidence"]["startUtf16"])),
                utf16_to_codepoint(text, int(item["evidence"]["endUtf16"])),
            )
            for item in page["annotations"]
            if item.get("evidence", {}).get("startUtf16") is not None
            and item.get("evidence", {}).get("endUtf16") is not None
        ]
    )
    evidence = set()
    for start, end in evidence_intervals:
        evidence.update(range(start, end))
    debris = debris_positions(text)
    retained = {
        index for index, char in enumerate(text) if not char.isspace() and index not in debris
    }
    uncovered = retained - evidence
    uncovered_pct = 100 * len(uncovered) / len(retained) if retained else 0.0
    return {
        "retained_characters": len(retained),
        "annotated_characters": len(retained - uncovered),
        "unannotated_characters": len(uncovered),
        "coverage_pct": 100 - uncovered_pct,
        "unannotated_pct": uncovered_pct,
        "evidence_intervals": evidence_intervals,
        "debris_positions": debris,
        "uncovered_positions": uncovered,
    }


def uncovered_intervals(page: dict[str, Any], coverage: dict[str, Any]) -> list[tuple[int, int]]:
    text = page["canonicalText"]
    evidence = set()
    for start, end in coverage["evidence_intervals"]:
        evidence.update(range(start, end))
    debris = coverage["debris_positions"]
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, char in enumerate(text):
        available = index not in evidence and index not in debris
        if available and start is None:
            start = index
        if not available and start is not None:
            if any(position in coverage["uncovered_positions"] for position in range(start, index)):
                intervals.append((start, index))
            start = None
    if start is not None and any(
        position in coverage["uncovered_positions"] for position in range(start, len(text))
    ):
        intervals.append((start, len(text)))
    return intervals


def marked_page_text(page: dict[str, Any], intervals: list[tuple[int, int]]) -> str:
    text = page["canonicalText"]
    parts: list[str] = []
    cursor = 0
    for number, (start, end) in enumerate(intervals, start=1):
        parts.append(text[cursor:start])
        parts.append(f"[UNANNOTATED U{number:03d}]")
        parts.append(text[start:end])
        parts.append(f"[/UNANNOTATED U{number:03d}]")
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def compact_existing(page: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in page["annotations"]:
        metadata = item.get("metadata", {})
        rows.append(
            {
                "id": item["id"],
                "e": item["evidence"]["canonicalText"],
                "s": item["subject"]["text"],
                "st": item["subject"].get("type"),
                "p": item["relation"]["rawLabel"],
                "o": item["object"]["text"],
                "ot": item["object"].get("type"),
                "d": metadata.get("date", {}).get("originalText"),
                "l": metadata.get("location", {}).get("originalText"),
                "q": metadata.get("quantity", {}).get("originalText"),
            }
        )
    return rows


def public_coverage(coverage: dict[str, Any]) -> dict[str, Any]:
    return {
        key: coverage[key]
        for key in (
            "retained_characters",
            "annotated_characters",
            "unannotated_characters",
            "coverage_pct",
            "unannotated_pct",
        )
    }


def build_prompt(
    volume: int, page_number: int
) -> tuple[str, dict[str, Any], dict[str, Any], list[tuple[int, int]]]:
    page = load_page(volume, page_number)
    coverage = coverage_record(page)
    intervals = uncovered_intervals(page, coverage)
    original_prompt = ORIGINAL_PROMPT_PATH.read_text(encoding="utf-8-sig")
    prompt = f"""<ORIGINAL_FIRST_PASS_PROMPT_READ_ONLY>
{original_prompt}
</ORIGINAL_FIRST_PASS_PROMPT_READ_ONLY>

{FOURTH_PASS_PROMPT}

<EXISTING_SUMMARY_READ_ONLY>
{page.get("summary") or ""}
</EXISTING_SUMMARY_READ_ONLY>

<EXISTING_TRIPLES_READ_ONLY>
{json.dumps(compact_existing(page), ensure_ascii=False, indent=2)}
</EXISTING_TRIPLES_READ_ONLY>

<PAGE_COVERAGE>
Current evidence coverage: {coverage["coverage_pct"]:.2f}%
Uncovered text: {coverage["unannotated_pct"]:.2f}% ({coverage["unannotated_characters"]} of {coverage["retained_characters"]} retained characters)
</PAGE_COVERAGE>

<TARGET_PAGE_TEXT_WITH_MARKERS>
{marked_page_text(page, intervals)}
</TARGET_PAGE_TEXT_WITH_MARKERS>
"""
    return prompt, page, coverage, intervals


def normalized(value: str) -> str:
    return re.sub(r"\s+", "", value)


def validate(result: FourthPassResult, page: dict[str, Any], coverage: dict[str, Any]) -> list[str]:
    text = page["canonicalText"]
    compact_text = normalized(text)
    existing = {
        (
            normalized(item["subject"]["text"]),
            item["relation"]["rawLabel"],
            normalized(item["object"]["text"]),
        )
        for item in page["annotations"]
    }
    warnings: list[str] = []
    for index, triple in enumerate(result.T, start=1):
        compact_evidence = normalized(triple.e)
        if not compact_evidence or compact_evidence not in compact_text:
            warnings.append(f"T[{index}].e does not resolve exactly after whitespace normalization")
        for field in ("s", "o", "d", "l", "q"):
            value = getattr(triple, field)
            if value and normalized(value) not in compact_text:
                warnings.append(f"T[{index}].{field} does not occur in TARGET_PAGE_TEXT")
        signature = (normalized(triple.s), triple.p, normalized(triple.o))
        if signature in existing:
            warnings.append(f"T[{index}] duplicates an existing subject-relation-object signature")
    return warnings


def write_coverage_catalogue() -> dict[str, Any]:
    rows = []
    for volume in (1, 2, 3):
        for path in sorted((READER_DATA / f"vol{volume}").glob("*.json")):
            page = json.loads(path.read_text(encoding="utf-8-sig"))
            coverage = coverage_record(page)
            if coverage["coverage_pct"] < 80:
                rows.append({"volume": volume, "page": int(path.stem), **public_coverage(coverage)})
    payload = {
        "method": "Non-whitespace canonical page text after conservative deterministic debris removal; evidence is the union of all reader-aligned first-, second-, and third-pass blocks.",
        "threshold": "coverage_pct < 80",
        "page_count": len(rows),
        "pages": rows,
    }
    COVERAGE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def run(volume: int, page_number: int, output_name: str) -> Path:
    prompt, page, coverage, intervals = build_prompt(volume, page_number)
    output_dir = OUTPUT_ROOT / f"vol{volume}" / f"page_{page_number:04d}" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_sent.txt").write_text(prompt, encoding="utf-8")
    (output_dir / "coverage_before.json").write_text(
        json.dumps(
            {**public_coverage(coverage), "unannotated_intervals": intervals},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FourthPassResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=10000,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    result = FourthPassResult.model_validate_json(response.text)
    warnings = validate(result, page, coverage)
    usage = response.usage_metadata.model_dump(mode="json") if response.usage_metadata else {}
    payload = {
        "valid": not warnings,
        "warnings": warnings,
        "usage": usage,
        "coverage_before": public_coverage(coverage),
        "result": result.model_dump(),
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "postprocessed.json").write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int)
    parser.add_argument("--page", type=int)
    parser.add_argument("--output-name", default="fourth_pass_01")
    parser.add_argument("--coverage-only", action="store_true")
    args = parser.parse_args()
    coverage = write_coverage_catalogue()
    print(f"below_80_pages={coverage['page_count']} coverage_file={COVERAGE_PATH}")
    if not args.coverage_only:
        if args.volume is None or args.page is None:
            parser.error("--volume and --page are required unless --coverage-only is used")
        print(run(args.volume, args.page, args.output_name))


if __name__ == "__main__":
    main()

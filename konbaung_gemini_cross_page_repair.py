#!/usr/bin/env python3
"""Propose repairs to page-boundary triples using two-page context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from konbaung_gemini_cite_sources_annotator import MODEL, ORIGINAL_PROMPT_PATH
from konbaung_gemini_summary_claim_completion_annotator import api_key
from konbaung_gemini_triple_metadata_semantic_frame import existing_triples, load_page, normalized


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "konbaung_cross_page_repair_tests"


class SpanGloss(BaseModel):
    my: str = Field(description="Exact Burmese span from either supplied page.")
    en: str = Field(description="Short English gloss.")


class EvidenceSpan(BaseModel):
    page: int = Field(description="Source page number.")
    my: str = Field(description="Exact Burmese evidence span from that page.")


class ReplacementTriple(BaseModel):
    evidence: list[EvidenceSpan]
    s: str
    sg: str
    st: str
    p: str
    o: str
    og: str
    ot: str
    time: SpanGloss | None = None
    place: SpanGloss | None = None
    quantity: SpanGloss | None = None
    manner: SpanGloss | None = None
    reason: SpanGloss | None = None


class RepairProposal(BaseModel):
    id: str = Field(description="Exact ID of the existing triple to replace.")
    replacement: ReplacementTriple


class RepairResult(BaseModel):
    R: list[RepairProposal]


REPAIR_PROMPT = """<CROSS_PAGE_REPAIR_TASK>
The existing chronicle annotations were created one page at a time. The model that created them could not see how a sentence, speaker, omitted subject, event, date, location, quantity, or argument continued across a page boundary. Consequently, some triples near the end of the first page or beginning of the second page may be incomplete or incorrectly interpreted.

Read both consecutive pages together and inspect the existing triples nearest their shared boundary. Propose a repair only when the two-page context materially changes or completes an existing triple. Relevant repairs include resolving a speaker or omitted participant, completing a cross-page predicate or object, correcting a relation, extending its evidence across pages, or correcting time, place, quantity, manner, or reason metadata whose scope becomes clear from the neighboring page.

For every repair, identify the exact existing triple ID and provide one complete replacement triple. Evidence may contain one span from either page or spans from both pages. Preserve an existing triple when cross-page context does not warrant a substantive change. Do not create unrelated new triples, rewrite correct triples stylistically, or use this task for ordinary within-page enrichment. Return an empty `R` when no boundary repair is necessary.

Every evidence span and Burmese metadata value must occur verbatim in one of the supplied page texts. Subject or object text may remain inferred when the chronicle omits it, but the evidence must support the inference.
</CROSS_PAGE_REPAIR_TASK>"""


def original_prefix() -> str:
    return ORIGINAL_PROMPT_PATH.read_text(encoding="utf-8-sig").rstrip()


def page_block(volume: int, page_number: int) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    page = load_page(volume, page_number)
    triples = existing_triples(page)
    block = f"""<PAGE number=\"{page_number}\">
<PAGE_TEXT>
{page['canonicalText']}
</PAGE_TEXT>
<EXISTING_TRIPLES_READ_ONLY>
{json.dumps(triples, ensure_ascii=False, indent=2)}
</EXISTING_TRIPLES_READ_ONLY>
</PAGE>"""
    return block, page, triples


def build_prompt(volume: int, first_page: int) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    first_block, first, first_triples = page_block(volume, first_page)
    second_block, second, second_triples = page_block(volume, first_page + 1)
    prompt = f"""<ORIGINAL_FIRST_PASS_PROMPT_READ_ONLY>
{original_prefix()}
</ORIGINAL_FIRST_PASS_PROMPT_READ_ONLY>

{REPAIR_PROMPT}

{first_block}

{second_block}
"""
    return prompt, [first, second], first_triples + second_triples


def validate(result: RepairResult, pages: list[dict[str, Any]], triples: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    ids = {triple["id"] for triple in triples}
    page_text = {int(page["pageNumber"]): normalized(page["canonicalText"]) for page in pages}
    combined = "".join(page_text.values())
    for index, repair in enumerate(result.R, 1):
        if repair.id not in ids:
            flags.append(f"R[{index}]: unknown triple ID {repair.id}")
        for evidence in repair.replacement.evidence:
            if evidence.page not in page_text or normalized(evidence.my) not in page_text.get(evidence.page, ""):
                flags.append(f"R[{index}]: evidence does not resolve on page {evidence.page}")
        for category in ("time", "place", "quantity", "manner", "reason"):
            value = getattr(repair.replacement, category)
            if value and normalized(value.my) not in combined:
                flags.append(f"R[{index}].{category}: span absent from both pages")
    return flags


def run(volume: int, first_page: int, output_name: str) -> Path:
    prompt, pages, triples = build_prompt(volume, first_page)
    output_dir = OUTPUT_ROOT / f"vol{volume}" / f"pages_{first_page:04d}_{first_page + 1:04d}" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_sent.txt").write_text(prompt, encoding="utf-8")
    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RepairResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=6000,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = response.text or ""
    (output_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
    flags: list[str] = []
    try:
        result = RepairResult.model_validate_json(raw)
        flags.extend(validate(result, pages, triples))
        parsed: dict[str, Any] | str = result.model_dump(exclude_none=True)
    except Exception as exc:
        flags.append(f"structured parse failed: {exc}")
        parsed = raw
    payload = {
        "accepted": True,
        "flagged": bool(flags),
        "flags": flags,
        "usage": response.usage_metadata.model_dump(mode="json") if response.usage_metadata else {},
        "result": parsed,
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int, required=True)
    parser.add_argument("--first-page", type=int, required=True)
    parser.add_argument("--output-name", default="cross_page_repair_01")
    args = parser.parse_args()
    print(run(args.volume, args.first_page, args.output_name))


if __name__ == "__main__":
    main()

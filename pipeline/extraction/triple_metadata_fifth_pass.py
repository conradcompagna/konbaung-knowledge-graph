#!/usr/bin/env python3
"""Augment existing chronicle triples with auxiliary metadata and arguments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from konbaung_gemini_cite_sources_annotator import MODEL, ORIGINAL_PROMPT_PATH
from konbaung_gemini_summary_claim_completion_annotator import READER_DATA, api_key


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "konbaung_triple_metadata_fifth_pass_test"


class SpanGloss(BaseModel):
    my: str = Field(description="Exact Burmese span from TARGET_PAGE_TEXT.")
    en: str = Field(description="Short English gloss.")


class AuxiliaryArgument(BaseModel):
    my: str = Field(description="Exact Burmese auxiliary argument span from TARGET_PAGE_TEXT.")
    en: str = Field(description="Short English gloss.")
    type: str = Field(
        description="Concise open type such as reason, instrument, method, condition, or result."
    )


class TripleMetadataAddition(BaseModel):
    id: str = Field(description="Exact existing triple ID receiving these additions.")
    time: list[SpanGloss] | None = Field(
        default=None, description="All new governing date or time details."
    )
    place: list[SpanGloss] | None = Field(
        default=None, description="All new governing location details."
    )
    quantity: list[SpanGloss] | None = Field(
        default=None, description="All new governing quantities or measurements."
    )
    othersubjects: list[SpanGloss] | None = Field(
        default=None, description="All additional subjects or co-agents beyond the main subject."
    )
    otherobjects: list[SpanGloss] | None = Field(
        default=None,
        description="All additional objects, recipients, targets, or affected entities beyond the main object.",
    )
    auxiliaryarguments: list[AuxiliaryArgument] | None = Field(
        default=None,
        description="All additional reasons, conditions, methods, instruments, routes, procedures, or results.",
    )


class FifthPassResult(BaseModel):
    M: list[TripleMetadataAddition] = Field(
        description="Only triples with useful additions, grouped by parent triple ID."
    )


FIFTH_PASS_PROMPT = """<NEW_FIFTH_PASS_TASK>
Augment the existing triples from one Burmese royal chronicle page with useful secondary metadata. The original first-pass prompt above is read-only historiographical and research-question context. This is not a new triple-extraction pass: preserve the existing triples and identify only additional information that belongs to them.

Inspect the existing triples, their evidence, and the full page context. Add further explicit date/time, place, or quantity spans when useful details are present beyond the metadata already recorded. These categories may contain multiple additions.

Work through the triples one by one. For each triple, inspect its evidence and the surrounding page context, then consider all six metadata categories before moving to the next triple: time, place, quantity, other subjects, other objects, and auxiliary arguments. Return every applicable nonredundant addition rather than only the most salient one. Omit a category only when the context supplies no genuinely new information for it.

Before writing the response, silently map the page's date/time expressions, named locations, quantities, participant lists, and causal, procedural, or consequential phrases to every existing triple they directly govern. Context shared across adjacent clauses may enrich more than one triple. Then emit the complete set of additions from that map.

When one predicate governs a coordinated list, treat unselected list members as other subjects or other objects. Preserve linked complements such as titles, offices, granted territories, destinations, transported assets, instruments, and measurements whenever they add information beyond the main endpoints. Do not mistake a group name containing a place name for an event location.

Also identify other subjects and other objects: additional participants in the same action or relation that are historically meaningful but are not the triple's main subject or object. Record only their Burmese spans and short English glosses.

Finally, record auxiliary arguments explaining why or under what condition something happened, the method, instrument, route, or procedure by which it happened, and its result or consequence. Complex predicates may have multiple auxiliary arguments. For each, record its Burmese span, English gloss, and a concise open type such as reason, instrument, method, condition, or result.

Group genuinely useful additions under their parent triple ID. Include only the applicable metadata fields in each record; do not emit empty fields. Do not repeat the main subject, main object, relation, evidence, or metadata already recorded. Do not invent unsupported participants or interpretations, create new triples, or pad the output. A triple with no useful additional metadata must have no output record at all. Explicit Burmese metadata and argument spans must be copied verbatim from TARGET_PAGE_TEXT; they may occur outside the immediate evidence block when page context supplies them.
</NEW_FIFTH_PASS_TASK>"""


def load_page(volume: int, page_number: int) -> dict[str, Any]:
    path = READER_DATA / f"vol{volume}" / f"{page_number:04d}.json"
    return json.loads(path.read_text(encoding="utf-8-sig"))


def existing_triples(page: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in page["annotations"]:
        metadata = item.get("metadata", {})
        rows.append(
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
                "time": metadata.get("date", {}).get("originalText"),
                "place": metadata.get("location", {}).get("originalText"),
                "quantity": metadata.get("quantity", {}).get("originalText"),
            }
        )
    return rows


def build_prompt(volume: int, page_number: int) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    page = load_page(volume, page_number)
    triples = existing_triples(page)
    original = ORIGINAL_PROMPT_PATH.read_text(encoding="utf-8-sig")
    prompt = f"""<ORIGINAL_FIRST_PASS_PROMPT_READ_ONLY>
{original}
</ORIGINAL_FIRST_PASS_PROMPT_READ_ONLY>

{FIFTH_PASS_PROMPT}

<EXISTING_SUMMARY_READ_ONLY>
{page.get("summary") or ""}
</EXISTING_SUMMARY_READ_ONLY>

<TARGET_PAGE_TEXT>
{page["canonicalText"]}
</TARGET_PAGE_TEXT>

<EXISTING_TRIPLES_READ_ONLY>
{json.dumps(triples, ensure_ascii=False, indent=2)}
</EXISTING_TRIPLES_READ_ONLY>
"""
    return prompt, page, triples


def normalized(value: str) -> str:
    return re.sub(r"\s+", "", value)


def validate(
    result: FifthPassResult, page: dict[str, Any], triples: list[dict[str, Any]]
) -> list[str]:
    target = normalized(page["canonicalText"])
    existing = {item["id"]: item for item in triples}
    warnings: list[str] = []
    for index, addition in enumerate(result.M, start=1):
        if addition.id not in existing:
            warnings.append(f"M[{index}]: unknown triple id {addition.id}")
        categories = {
            "time": addition.time or [],
            "place": addition.place or [],
            "quantity": addition.quantity or [],
            "othersubjects": addition.othersubjects or [],
            "otherobjects": addition.otherobjects or [],
            "auxiliaryarguments": addition.auxiliaryarguments or [],
        }
        if not any(categories.values()):
            warnings.append(f"M[{index}]: empty metadata record should have been omitted")
        for category, items in categories.items():
            for item in items:
                if item.my and normalized(item.my) not in target:
                    warnings.append(f"M[{index}].{category}: span is absent from page")
    return warnings


def run(volume: int, page_number: int, output_name: str) -> Path:
    prompt, page, triples = build_prompt(volume, page_number)
    output_dir = OUTPUT_ROOT / f"vol{volume}" / f"page_{page_number:04d}" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_sent.txt").write_text(prompt, encoding="utf-8")
    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FifthPassResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=10000,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    result = FifthPassResult.model_validate_json(response.text)
    warnings = validate(result, page, triples)
    normalized_result = result.model_dump(exclude_none=True)
    payload = {
        "valid": not warnings,
        "warnings": warnings,
        "usage": response.usage_metadata.model_dump(mode="json") if response.usage_metadata else {},
        "result": normalized_result,
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "postprocessed.json").write_text(
        json.dumps(normalized_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int, required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--output-name", default="fifth_pass_01")
    args = parser.parse_args()
    print(run(args.volume, args.page, args.output_name))


if __name__ == "__main__":
    main()

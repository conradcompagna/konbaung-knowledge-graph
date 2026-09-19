#!/usr/bin/env python3
"""Augment chronicle triples using the semantic-frame cached prefix."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from konbaung_gemini_cite_sources_annotator import MODEL
from konbaung_gemini_summary_claim_completion_annotator import READER_DATA, api_key


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "konbaung_triple_metadata_fifth_pass_test"
PREFIX_PATH = ROOT / "konbaung_fifth_pass_semantic_frame_prefix.txt"


class MetadataItem(BaseModel):
    operation: Literal["add", "replace"]
    span: str = Field(description="Exact Burmese span from TARGET_PAGE_TEXT.")
    gloss: str = Field(description="Short English gloss.")
    role: str = Field(description="Concise semantic role.")
    replaces: str | None = None
    note: str | None = None


class QuantityItem(MetadataItem):
    measures: str = Field(description="What the quantity counts, values, or measures.")


class ParticipantItem(BaseModel):
    span: str = Field(description="Exact Burmese span from TARGET_PAGE_TEXT.")
    gloss: str = Field(description="Short English gloss.")
    role: str = Field(description="Participant's semantic role in this relation.")
    open_type: str = Field(description="Concise open analytical type.")


class AuxiliaryArgument(BaseModel):
    span: str = Field(description="Exact Burmese argument span from TARGET_PAGE_TEXT.")
    gloss: str = Field(description="Short English gloss.")
    argument_type: str = Field(description="ALL_CAPS semantic role.")
    open_type: str = Field(description="Concise open analytical type.")
    note: str | None = None


class TripleMetadataAddition(BaseModel):
    triple_id: str = Field(description="Exact existing triple ID receiving these additions.")
    time: list[MetadataItem] | None = None
    locations: list[MetadataItem] | None = None
    quantities: list[QuantityItem] | None = None
    other_subjects: list[ParticipantItem] | None = None
    other_objects: list[ParticipantItem] | None = None
    auxiliary_arguments: list[AuxiliaryArgument] | None = None


class FifthPassResult(BaseModel):
    augmentations: list[TripleMetadataAddition]
    coverage_report: str


def fixed_prefix() -> str:
    text = PREFIX_PATH.read_text(encoding="utf-8-sig")
    start = text.find("<FIXED_PREFIX_BEGIN>")
    end = text.rfind("<FIXED_PREFIX_END>")
    if start < 0 or end < start:
        raise ValueError(f"Fixed-prefix markers missing from {PREFIX_PATH}")
    return text[start : end + len("<FIXED_PREFIX_END>")]


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


def runtime_input(volume: int, page_number: int) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    page = load_page(volume, page_number)
    triples = existing_triples(page)
    prompt = f"""<RUNTIME_INPUT_BEGIN>
<EXISTING_SUMMARY_READ_ONLY>
{page.get('summary') or ''}
</EXISTING_SUMMARY_READ_ONLY>

<TARGET_PAGE_TEXT>
{page['canonicalText']}
</TARGET_PAGE_TEXT>

<EXISTING_TRIPLES_READ_ONLY>
{json.dumps(triples, ensure_ascii=False, indent=2)}
</EXISTING_TRIPLES_READ_ONLY>
<RUNTIME_INPUT_END>
"""
    return prompt, page, triples


def normalized(value: str) -> str:
    return re.sub(r"\s+", "", value)


def validate(result: FifthPassResult, page: dict[str, Any], triples: list[dict[str, Any]]) -> list[str]:
    target = normalized(page["canonicalText"])
    existing = {item["id"]: item for item in triples}
    warnings: list[str] = []
    for index, addition in enumerate(result.augmentations, start=1):
        if addition.triple_id not in existing:
            warnings.append(f"augmentations[{index}]: unknown triple id {addition.triple_id}")
        categories = {
            "time": addition.time or [],
            "locations": addition.locations or [],
            "quantities": addition.quantities or [],
            "other_subjects": addition.other_subjects or [],
            "other_objects": addition.other_objects or [],
            "auxiliary_arguments": addition.auxiliary_arguments or [],
        }
        if not any(categories.values()):
            warnings.append(f"M[{index}]: empty metadata record should have been omitted")
        for category, items in categories.items():
            for item in items:
                if item.span and normalized(item.span) not in target:
                    warnings.append(f"augmentations[{index}].{category}: span is absent from page")
    return warnings


def create_cache(client: genai.Client, ttl_seconds: int) -> str:
    cached = client.caches.create(
        model=MODEL,
        config=types.CreateCachedContentConfig(
            display_name="konbaung_semantic_frame_metadata_prefix",
            contents=fixed_prefix(),
            ttl=f"{ttl_seconds}s",
        ),
    )
    if not cached.name:
        raise RuntimeError("Gemini did not return a cache name")
    return cached.name


def run(
    volume: int,
    page_number: int,
    output_name: str,
    cache_name: str | None = None,
    create_new_cache: bool = False,
    cache_ttl_seconds: int = 86400,
) -> Path:
    runtime, page, triples = runtime_input(volume, page_number)
    prefix = fixed_prefix()
    prompt = f"{prefix}\n\n{runtime}"
    output_dir = OUTPUT_ROOT / f"vol{volume}" / f"page_{page_number:04d}" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_sent.txt").write_text(prompt, encoding="utf-8")
    client = genai.Client(api_key=api_key())
    if create_new_cache:
        cache_name = create_cache(client, cache_ttl_seconds)
    config = types.GenerateContentConfig(
        cached_content=cache_name,
        response_mime_type="application/json",
        response_schema=FifthPassResult,
        temperature=0.0,
        candidate_count=1,
        max_output_tokens=10000,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=runtime if cache_name else prompt,
        config=config,
    )
    result = FifthPassResult.model_validate_json(response.text)
    warnings = validate(result, page, triples)
    normalized_result = result.model_dump(exclude_none=True)
    payload = {
        "valid": not warnings,
        "warnings": warnings,
        "usage": response.usage_metadata.model_dump(mode="json") if response.usage_metadata else {},
        "cache_name": cache_name,
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
    parser.add_argument("--cache-name")
    parser.add_argument("--create-cache", action="store_true")
    parser.add_argument("--cache-ttl-seconds", type=int, default=86400)
    args = parser.parse_args()
    print(
        run(
            args.volume,
            args.page,
            args.output_name,
            cache_name=args.cache_name,
            create_new_cache=args.create_cache,
            cache_ttl_seconds=args.cache_ttl_seconds,
        )
    )


if __name__ == "__main__":
    main()

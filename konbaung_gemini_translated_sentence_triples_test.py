#!/usr/bin/env python3
"""Extract triples from pretranslated Burmese chronicle sentences."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from konbaung_gemini_cite_sources_annotator import MODEL
from konbaung_gemini_summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parent
READER_DATA_ROOT = ROOT / "konbaung_reader_app" / "static" / "data" / "konbaung"
TRANSLATION_ROOT = ROOT / "konbaung_sentence_translation_full_batch_20260713_repaired" / "pages"
PROMPT_PATH = ROOT / "konbaung_translated_sentence_annotation_prompt.md"
OUTPUT_ROOT = ROOT / "konbaung_translated_sentence_triple_tests_unified"


class Endpoint(BaseModel):
    my: str = Field(description="Burmese referent span or concise inferred Burmese referent.")
    en: str = Field(description="Compact English gloss of the referent.")
    tag: str = Field(description="Concise open-coded entity tag.")
    source: Literal["sentence", "page", "inferred"]


class Triple(BaseModel):
    s: Endpoint
    p: str = Field(description="Concise ALL_CAPS open-coded relation label.")
    o: Endpoint


class SentenceTriples(BaseModel):
    sid: str = Field(description="ID of the sentence supporting these triples.")
    T: list[Triple] = Field(min_length=1)


class TripleResult(BaseModel):
    S: list[SentenceTriples]


def translation_dir(volume: int, page_number: int) -> Path:
    page_dir = TRANSLATION_ROOT / f"vol{volume}" / f"page_{page_number:04d}"
    if not (page_dir / "payload.json").exists() or not (page_dir / "result.json").exists():
        raise FileNotFoundError(f"No stored sentence translation for volume {volume}, page {page_number}")
    return page_dir


def build_payload(volume: int, page_number: int) -> dict[str, Any]:
    page_path = READER_DATA_ROOT / "pages" / f"vol{volume}" / f"{page_number:04d}.json"
    page = json.loads(page_path.read_text(encoding="utf-8"))
    source_dir = translation_dir(volume, page_number)
    translation_input = json.loads((source_dir / "payload.json").read_text(encoding="utf-8"))
    translation_output = json.loads((source_dir / "result.json").read_text(encoding="utf-8"))
    translations = {
        item["id"]: item["translation"]
        for item in translation_output["response"]["translations"]
    }
    sentence_pairs = []
    for sentence in translation_input["sentences"]:
        sentence_id = sentence["id"]
        if sentence_id not in translations:
            raise ValueError(f"Missing stored translation for {sentence_id}")
        sentence_pairs.append(
            {
                "id": sentence_id,
                "my": sentence["my"],
                "en": translations[sentence_id],
            }
        )
    return {
        "page_id": f"vol{volume}-p{page_number:04d}",
        "summary": page["summary"],
        "sentence_pairs": sentence_pairs,
    }


def full_prompt(payload: dict[str, Any]) -> str:
    annotation_prompt = PROMPT_PATH.read_text(encoding="utf-8-sig").rstrip()
    return f"""{annotation_prompt}

<ANNOTATION_INPUT>
{json.dumps(payload, ensure_ascii=False, indent=2)}
</ANNOTATION_INPUT>
"""


def validate(payload: dict[str, Any], result: TripleResult) -> list[str]:
    errors: list[str] = []
    sentence_ids = [item["id"] for item in payload["sentence_pairs"]]
    known_ids = set(sentence_ids)
    counts = Counter(group.sid for group in result.S)
    unknown = sorted(set(counts) - known_ids)
    missing = [sentence_id for sentence_id in sentence_ids if counts[sentence_id] == 0]
    duplicates = [sentence_id for sentence_id in sentence_ids if counts[sentence_id] > 1]
    if unknown:
        errors.append(f"Unknown sentence IDs: {unknown}")
    if missing:
        errors.append(f"Missing sentence records: {missing}")
    if duplicates:
        errors.append(f"Duplicate sentence records: {duplicates}")
    order = {sentence_id: index for index, sentence_id in enumerate(sentence_ids)}
    observed = [order.get(group.sid, len(order)) for group in result.S]
    if observed != sorted(observed):
        errors.append("Sentence records are not in supplied order")
    return errors


def review_text(payload: dict[str, Any], result: TripleResult, errors: list[str]) -> str:
    triples_by_sentence: dict[str, list[Triple]] = {}
    for group in result.S:
        triples_by_sentence.setdefault(group.sid, []).extend(group.T)
    lines = [
        f"# {payload['page_id']}",
        "",
        "VALID" if not errors else "FLAGGED",
        "",
        "## Summary",
        "",
        payload["summary"],
        "",
    ]
    lines.extend(f"- {error}" for error in errors)
    if errors:
        lines.append("")
    for sentence in payload["sentence_pairs"]:
        lines.extend(
            [
                f"## {sentence['id']}",
                "",
                sentence["my"],
                "",
                sentence["en"],
                "",
            ]
        )
        for triple in triples_by_sentence.get(sentence["id"], []):
            lines.extend(
                [
                    f"- {triple.s.my} [{triple.s.tag}; {triple.s.source}] "
                    f"--{triple.p}--> {triple.o.my} [{triple.o.tag}; {triple.o.source}]",
                    f"  {triple.s.en} -> {triple.o.en}",
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run(volume: int, page_number: int, output_name: str) -> Path:
    payload = build_payload(volume, page_number)
    prompt = full_prompt(payload)
    output_dir = OUTPUT_ROOT / f"vol{volume}" / f"page_{page_number:04d}" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "payload.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "prompt_sent.txt").write_text(prompt, encoding="utf-8")

    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TripleResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=8000,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
        ),
    )
    raw = response.text or ""
    (output_dir / "raw_response.json").write_text(raw, encoding="utf-8")
    result = TripleResult.model_validate_json(raw)
    errors = validate(payload, result)
    (output_dir / "result.json").write_text(
        json.dumps(
            {
                "accepted": not errors,
                "errors": errors,
                "usage": response.usage_metadata.model_dump(mode="json")
                if response.usage_metadata
                else {},
                "response": result.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "review.md").write_text(
        review_text(payload, result, errors), encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int, required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--output-name", default="unified_annotation_01")
    args = parser.parse_args()
    print(run(args.volume, args.page, args.output_name))


if __name__ == "__main__":
    main()

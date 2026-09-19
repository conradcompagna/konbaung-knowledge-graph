#!/usr/bin/env python3
"""Ground existing relation labels in exact Burmese predicate spans for one page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict

from konbaung_gemini_summary_claim_completion_annotator import api_key
from konbaung_gemini_translated_sentence_triple_enrichment_test import build_payload


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "konbaung_relation_predicate_grounding_trials"
MODEL = "gemini-3.1-flash-lite"
INSTRUCTION = """You are being given a list of linked Burmese  sentences and their existing triples coming from one page of Konbaung chronicle text.

For every triple, copy the exact contiguous Burmese span in the setnence that expresses the relation, and give a compact English gloss of that predicate span. If the predicate itself is not located in the sentence, list it if it is present in another sentence, or if truly inferred, mark that down. 

The Burmese predicate span must be copied verbatim from the assigned sentence; never invent change or simplify the original wording."""


class Grounding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    my: str
    en: str
    src: Literal["sentence", "page", "inferred"]


class Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    R: list[Grounding]


def prepare_records(volume: int, page: int) -> tuple[dict, list[dict], list[str]]:
    payload = build_payload(volume, page)
    records: list[dict] = []
    expected: list[str] = []
    for sentence in payload["sentences"]:
        triples = []
        for index, triple in enumerate(sentence["existing_triples"], start=1):
            tid = f"{sentence['id']}_t{index:03d}"
            expected.append(tid)
            triples.append(
                {
                    "id": tid,
                    "s": triple["s"]["my"],
                    "p": triple["p"],
                    "o": triple["o"]["my"],
                }
            )
        records.append(
            {
                "sid": sentence["id"],
                "my": sentence["my"],
                "en": sentence["en"],
                "T": triples,
            }
        )
    return payload, records, expected


def validate(records: list[dict], expected: list[str], result: Result) -> list[str]:
    errors: list[str] = []
    returned = [item.id for item in result.R]
    if returned != expected:
        errors.append(f"TID/order mismatch: expected {expected}, returned {returned}")
    sentences = {record["sid"]: record["my"] for record in records}
    for item in result.R:
        assigned_sid = item.id.rsplit("_t", 1)[0]
        if item.src == "sentence":
            if item.my not in sentences.get(assigned_sid, ""):
                errors.append(f"{item.id}: sentence span not found in assigned sentence")
        elif item.src == "page":
            if not any(item.my in text for text in sentences.values()):
                errors.append(f"{item.id}: page span not found on supplied page")
    return errors


def render_review(payload: dict, records: list[dict], result: Result, errors: list[str]) -> str:
    lookup = {item.id: item for item in result.R}
    lines = [
        f"# {payload['page_id']} predicate-grounding review",
        "",
        "VALID" if not errors else "FLAGGED",
        "",
    ]
    for record in records:
        if not record["T"]:
            continue
        lines.extend(
            [
                f"## {record['sid']}",
                "",
                "### Burmese",
                "",
                record["my"],
                "",
                "### English",
                "",
                record["en"],
                "",
            ]
        )
        for triple in record["T"]:
            item = lookup.get(triple["id"])
            lines.extend(
                [
                    f"### {triple['id']}",
                    "",
                    f"- Subject: `{triple['s']}`",
                    f"- Relation: `{triple['p']}`",
                    f"- Object: `{triple['o']}`",
                ]
            )
            if item:
                lines.extend(
                    [
                        f"- Predicate span: `{item.my}`",
                        f"- Predicate gloss: {item.en}",
                        f"- Source: `{item.src}`",
                        "",
                    ]
                )
    if errors:
        lines.extend(["## Validation errors", ""])
        lines.extend(f"- {error}" for error in errors)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run(volume: int, page: int, output_name: str) -> Path:
    payload, records, expected = prepare_records(volume, page)
    output_dir = OUTPUT_ROOT / f"vol{volume}" / f"page_{page:04d}" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    input_data = {"page_id": payload["page_id"], "records": records}
    prompt = INSTRUCTION + "\n\n<INPUT>\n" + json.dumps(input_data, ensure_ascii=False, indent=2) + "\n</INPUT>"
    (output_dir / "payload.json").write_text(
        json.dumps(input_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "prompt_sent.txt").write_text(prompt, encoding="utf-8")
    (output_dir / "response_schema.json").write_text(
        json.dumps(Result.model_json_schema(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=Result.model_json_schema(),
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,
                include_thoughts=False,
            ),
        ),
    )
    raw = response.text or ""
    (output_dir / "raw_response.json").write_text(raw, encoding="utf-8")
    result = Result.model_validate_json(raw)
    errors = validate(records, expected, result)
    stored = {
        "accepted": not errors,
        "errors": errors,
        "request": {
            "model": MODEL,
            "thinking_budget": 0,
            "include_thoughts": False,
            "temperature": 0.0,
            "candidate_count": 1,
            "max_output_tokens": 4096,
        },
        "usage": response.usage_metadata.model_dump(mode="json") if response.usage_metadata else {},
        "response": result.model_dump(mode="json"),
    }
    (output_dir / "result.json").write_text(
        json.dumps(stored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "review.md").write_text(
        render_review(payload, records, result, errors), encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int, required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--output-name", default="user_prompt_minimal_trial")
    args = parser.parse_args()
    output_dir = run(args.volume, args.page, args.output_name)
    print(json.dumps({"output_dir": str(output_dir), "review": str(output_dir / "review.md")}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rerun the same page with a grouped, mandatory full-rewrite schema."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from konbaung_gemini_summary_claim_completion_annotator import api_key
from konbaung_gemini_v4_delta_repair_trial import read_json, write_json


ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT
    / "konbaung_v4_delta_repair_trials"
    / "vol2"
    / "page_0353"
    / "concise_delta_prompt_trial_01"
    / "payload.json"
)
OUTPUT = (
    ROOT
    / "konbaung_v4_delta_repair_trials"
    / "vol2"
    / "page_0353"
    / "flat_readable_full_rewrite_no_thinking_trial_03"
)
MODEL = "gemini-3.1-flash-lite"

PROMPT = """You will receive one page of the Konbaung Chronicle containing every Burmese sentence, its English translation, and all existing triples attached to that sentence.

Rewrite every supplied triple exactly once. Do not skip or delete any ID, and do not add, split, or merge facts. Preserve the input IDs and their order.

For each triple, work in this order:

1. Write the most natural and semantically coherent Burmese subject, predicate, and object for the specific relation expressed by the sentence.
2. Write a direct English gloss of each selected Burmese subject, predicate, and object.
3. Write an analytical entity tag for the subject, an analytical relation tag for the predicate, and an analytical entity tag for the object.

Read every result literally as:

subject → predicate → object

The three components must make sense together. Repair repeated or wrongly attached spans, incorrect direction, confused recipients or contents, and predicates attached to the wrong subject or object.

Use exact Burmese text where the component is expressed. A grammatically omitted subject or object may remain represented by its inferred referent. Never invent Burmese wording for an explicit span.

The English values must translate or transliterate their Burmese values. In particular, the predicate English must translate the Burmese predicate itself rather than restating the analytical predicate tag.

Tags are open-coded analytical classifications one useful step above literal translation. Do not impose a closed ontology, but do not mechanically convert an English gloss into PascalCase or UPPER_SNAKE_CASE.

Return every input ID exactly once in input order. Return only schema-valid JSON with no explanation."""


def triple_ids(payload: dict[str, Any]) -> list[str]:
    return [triple["id"] for sentence in payload["S"] for triple in sentence["T"]]


def model_input_text(payload: dict[str, Any]) -> str:
    """Render the page exactly as readable sentence-linked blocks for Gemini."""
    lines = [f'<PAGE id="{payload["page_id"]}">']
    for sentence in payload["S"]:
        lines.extend(
            [
                f'<SENTENCE id="{sentence["sid"]}">',
                "BURMESE SENTENCE:",
                sentence["my"],
                "",
                "ENGLISH TRANSLATION:",
                sentence["en"],
                "",
                "EXISTING TRIPLES:",
            ]
        )
        for triple in sentence["T"]:
            lines.extend(
                [
                    f'<TRIPLE id="{triple["id"]}">',
                    f"s_my: {triple['s']['my']}",
                    f"p_my: {triple['predicate_grounding']['my']}",
                    f"o_my: {triple['o']['my']}",
                    f"s_en: {triple['s']['en']}",
                    f"p_en: {triple['predicate_grounding']['en']}",
                    f"o_en: {triple['o']['en']}",
                    f"s_tag: {triple['s']['tag']}",
                    f"p_tag: {triple['p']}",
                    f"o_tag: {triple['o']['tag']}",
                    "</TRIPLE>",
                    "",
                ]
            )
        lines.extend(["</SENTENCE>", ""])
    lines.append("</PAGE>")
    return "\n".join(lines)


def response_schema(ids: list[str]) -> dict[str, Any]:
    fields = (
        "s_my",
        "p_my",
        "o_my",
        "s_en",
        "p_en",
        "o_en",
        "s_tag",
        "p_tag",
        "o_tag",
    )
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string", "enum": ids},
            **{field: {"type": "string"} for field in fields},
        },
        "required": ["id", *fields],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "R": {
                "type": "array",
                "minItems": len(ids),
                "maxItems": len(ids),
                "items": item,
            }
        },
        "required": ["R"],
    }


def old_display(triple: dict[str, Any]) -> dict[str, Any]:
    return {
        "s": {key: triple["s"][key] for key in ("my", "en", "tag")},
        "p": {
            "my": triple["predicate_grounding"]["my"],
            "en": triple["predicate_grounding"]["en"],
            "tag": triple["p"],
        },
        "o": {key: triple["o"][key] for key in ("my", "en", "tag")},
    }


def new_display(item: dict[str, Any]) -> dict[str, Any]:
    return {
        role: {
            "my": item["my"][role],
            "en": item["en"][role],
            "tag": item["tag"][role],
        }
        for role in ("s", "p", "o")
    }


def review_markdown(payload: dict[str, Any], result: dict[str, Any]) -> str:
    replacements = {item["id"]: item for item in result["R"]}
    lines = [f"# {payload['page_id']}", ""]
    for sentence in payload["S"]:
        lines.extend([f"## {sentence['sid']}", "", sentence["my"], "", sentence["en"], ""])
        if not sentence["T"]:
            lines.extend(["No triples.", ""])
            continue
        for old in sentence["T"]:
            item_id = old["id"]
            lines.extend(
                [
                    f"### {item_id}",
                    "",
                    "Old triple:",
                    "",
                    "```json",
                    json.dumps(old_display(old), ensure_ascii=False, indent=2),
                    "```",
                    "",
                    "Rewritten triple:",
                    "",
                    "```json",
                    json.dumps(new_display(replacements[item_id]), ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def validate(result: dict[str, Any], ids: list[str]) -> list[str]:
    rows = result.get("R", [])
    output_ids = [item.get("id") for item in rows]
    errors = []
    if output_ids != ids:
        errors.append("IDs are missing, duplicated, or out of input order")
    for item in rows:
        for group in ("my", "en", "tag"):
            for role in ("s", "p", "o"):
                if not str(item[group][role]).strip():
                    errors.append(f"{item.get('id')}: blank {group}.{role}")
    return errors


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    payload = read_json(SOURCE)
    ids = triple_ids(payload)
    schema = response_schema(ids)
    prompt_sent = (
        PROMPT
        + "\n\nPAGE_DATA\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\nEND_PAGE_DATA"
    )
    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt_sent,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
            max_output_tokens=max(8192, len(ids) * 600),
        ),
    )
    text = response.text or ""
    result = json.loads(text)
    errors = validate(result, ids)
    usage = response.usage_metadata.model_dump(mode="json") if response.usage_metadata else {}
    write_json(OUTPUT / "payload.json", payload)
    write_json(OUTPUT / "schema.json", schema)
    (OUTPUT / "prompt_sent.txt").write_text(prompt_sent, encoding="utf-8")
    write_json(OUTPUT / "raw_response.json", {"text": text, "usage": usage})
    write_json(OUTPUT / "result.json", result)
    write_json(
        OUTPUT / "status.json",
        {
            "created": datetime.now(timezone.utc).isoformat(),
            "page_id": payload["page_id"],
            "model": MODEL,
            "thinking_config": None,
            "input_triples": len(ids),
            "output_triples": len(result.get("R", [])),
            "errors": errors,
            "usage": usage,
        },
    )
    (OUTPUT / "review.md").write_text(review_markdown(payload, result), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "triples": len(ids),
                "returned": len(result.get("R", [])),
                "errors": errors,
                "usage": usage,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Translate precomputed Burmese chronicle sentences one by one with Gemini."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from pipeline.extraction.cite_sources_annotator import MODEL
from pipeline.extraction.summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = ROOT / "konbaung_sentence_corpus_20260713"
OUTPUT_ROOT = ROOT / "konbaung_sentence_translation_tests"


TRANSLATION_PROMPT = """Translate every supplied Burmese sentence into full, faithful English.

This is translation, not summarization, annotation, interpretation, or triple extraction. Translate only each sentence item's `my` text. Use `page_context` only to resolve local syntax, omitted subjects, pronouns, names, titles, and continuity.

Preserve every explicit clause, actor, action, object, name, title, list item, direction, place, date, time, quantity, condition, purpose, cause, result, contrast, negation, modality, sequence, command, and reported statement. Do not shorten, condense, generalize, selectively report, or aggregate repetitive material. Preserve proposals as proposals, possibilities as possibilities, and plain statements of death as death rather than killing. Do not invent agents, actions, motives, causal links, or completed events absent from the target.

Write clear natural English while preserving the source's internal structure. Keep inventories as inventories rather than manufacturing event sentences. When page context resolves an omitted referent unambiguously, supply it in square brackets; otherwise preserve the ambiguity. Ignore OCR line wrapping without silently repairing lexical content. The sentences have already been segmented; do not merge or split them.

Return exactly one item for every sentence, preserving IDs and order. Return JSON only under the supplied schema, with no commentary."""


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def load_sentences(volume: int, page_number: int) -> list[dict[str, Any]]:
    path = CORPUS_ROOT / "sentences" / f"vol{volume}.jsonl"
    sentences: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("owner_page") == page_number:
            sentences.append(item)
    if not sentences:
        raise ValueError(f"No owned sentences found for volume {volume}, page {page_number}")
    return sentences


def build_payload(volume: int, page_number: int) -> dict[str, Any]:
    page_path = CORPUS_ROOT / "cleaned_pages" / f"vol{volume}" / f"page_{page_number:04d}.txt"
    source_sentences = load_sentences(volume, page_number)
    return {
        "page_id": f"vol{volume}-p{page_number:04d}",
        "page_context": normalize(page_path.read_text(encoding="utf-8")),
        "sentences": [
            {"id": item["id"], "my": normalize(item["text"])} for item in source_sentences
        ],
    }


def response_schema(ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "translations": {
                "type": "array",
                "minItems": len(ids),
                "maxItems": len(ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string", "enum": ids},
                        "translation": {
                            "type": "string",
                            "description": "Full faithful English translation of only the target sentence.",
                        },
                    },
                    "required": ["id", "translation"],
                },
            }
        },
        "required": ["translations"],
    }


def validate(payload: dict[str, Any], result: dict[str, Any]) -> list[str]:
    translations = result.get("translations")
    if not isinstance(translations, list):
        return ["translations must be an array"]

    errors: list[str] = []
    expected_ids = [item["id"] for item in payload["sentences"]]
    actual_ids = [item.get("id") for item in translations if isinstance(item, dict)]
    if actual_ids != expected_ids:
        errors.append(f"ID/order mismatch: expected {expected_ids!r}, received {actual_ids!r}")
    for index, item in enumerate(translations):
        if not isinstance(item, dict) or not str(item.get("translation", "")).strip():
            errors.append(f"translations[{index}] has an empty translation")
    return errors


def write_review(
    path: Path,
    payload: dict[str, Any],
    result: dict[str, Any],
    errors: list[str],
) -> None:
    lines = [
        f"# {payload['page_id']}",
        "",
        "VALID" if not errors else "FLAGGED",
        "",
    ]
    lines.extend(f"- {error}" for error in errors)
    if errors:
        lines.append("")
    for source, output in zip(payload["sentences"], result.get("translations", [])):
        lines.extend(
            [
                f"## {source['id']}",
                "",
                source["my"],
                "",
                str(output.get("translation", "")),
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run(volume: int, page_number: int, output_name: str) -> Path:
    payload = build_payload(volume, page_number)
    ids = [item["id"] for item in payload["sentences"]]
    prompt = (
        TRANSLATION_PROMPT
        + "\n\n<INPUT>\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n</INPUT>\n"
    )
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
            response_json_schema=response_schema(ids),
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=8000,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = response.text or ""
    (output_dir / "raw_response.json").write_text(raw, encoding="utf-8")
    result = json.loads(raw)
    errors = validate(payload, result)
    write_review(output_dir / "review.md", payload, result, errors)
    (output_dir / "result.json").write_text(
        json.dumps(
            {
                "accepted": not errors,
                "errors": errors,
                "usage": response.usage_metadata.model_dump(mode="json")
                if response.usage_metadata
                else {},
                "response": result,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int, required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--output-name", default="sentence_translation_01")
    args = parser.parse_args()
    print(run(args.volume, args.page, args.output_name))


if __name__ == "__main__":
    main()

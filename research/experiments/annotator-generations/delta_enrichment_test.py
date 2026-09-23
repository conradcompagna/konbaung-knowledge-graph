#!/usr/bin/env python3
"""Test the downloaded immutable-triple Gemini enrichment delta design."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from konbaung_gemini_cite_sources_annotator import MODEL
from konbaung_gemini_summary_claim_completion_annotator import api_key
from konbaung_gemini_triple_metadata_semantic_frame import load_page


ROOT = Path(__file__).resolve().parent
DOWNLOADS = Path.home() / "Downloads"
PROMPT_PATH = DOWNLOADS / "gemini_burmese_triple_enrichment_prompt.txt"
SCHEMA_PATH = DOWNLOADS / "gemini_burmese_triple_enrichment_schema.json"
VALIDATOR_PATH = DOWNLOADS / "burmese_enrichment_validator.py"
OUTPUT_ROOT = ROOT / "konbaung_delta_enrichment_tests"


def normalize_for_model(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def load_validator() -> Any:
    spec = importlib.util.spec_from_file_location("downloaded_enrichment_validator", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load validator from {VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_payload(volume: int, page_number: int) -> tuple[dict[str, Any], dict[str, Any]]:
    page = load_page(volume, page_number)
    records: list[dict[str, Any]] = []
    metadata_sources = (("date", "time"), ("location", "place"), ("quantity", "quantity"))

    for annotation in page["annotations"]:
        metadata: list[dict[str, str]] = []
        for source_key, tag in metadata_sources:
            source = annotation.get("metadata", {}).get(source_key, {})
            if source.get("originalText"):
                metadata.append(
                    {
                        "id": f"m{len(metadata)}",
                        "tag": tag,
                        "my": normalize_for_model(source["originalText"]),
                        "en": source.get("gloss") or "",
                    }
                )

        subject = {
            "my": normalize_for_model(annotation["subject"]["text"]),
            "en": annotation["subject"].get("gloss") or "",
            "tag": annotation["subject"].get("type") or "",
        }
        object_ = {
            "my": normalize_for_model(annotation["object"]["text"]),
            "en": annotation["object"].get("gloss") or "",
            "tag": annotation["object"].get("type") or "",
        }
        used: list[str] = []
        for span in [subject["my"], object_["my"], *(item["my"] for item in metadata)]:
            if span and span not in used:
                used.append(span)

        records.append(
            {
                "id": annotation["id"],
                "evidence": normalize_for_model(annotation["evidence"]["canonicalText"]),
                "subject": subject,
                "predicate_tag": annotation["relation"]["rawLabel"],
                "object": object_,
                "metadata": metadata,
                "used": used,
            }
        )

    return {"page_id": f"vol{volume}-p{page_number:04d}", "records": records}, page


def write_review(
    path: Path, payload: dict[str, Any], response: dict[str, Any], errors: list[str]
) -> None:
    outputs = {item.get("id"): item for item in response.get("D", []) if isinstance(item, dict)}
    lines = ["# Validation", "", "VALID" if not errors else "FLAGGED", ""]
    lines.extend(f"- {error}" for error in errors)
    if errors:
        lines.append("")
    lines.extend(["# Enrichment deltas", ""])
    for record in payload["records"]:
        out = outputs.get(record["id"], {})
        lines.extend(
            [
                f"## {record['id']}",
                "",
                f"**Triple:** {record['subject']['en']} | {record['predicate_tag']} | {record['object']['en']}",
                "",
                f"**Evidence:** {record['evidence']}",
                "",
            ]
        )
        predicate = out.get("p", {})
        lines.append(
            f"**Predicate:** {predicate.get('my', '')} | {predicate.get('en', '')} | {predicate.get('fit', '')}"
        )
        placements = out.get("metadata_placement", [])
        if placements:
            lines.extend(["", "**Metadata placement:**"])
            lines.extend(f"- {item.get('id')} -> {item.get('to')}" for item in placements)
        additions = [*(("S", item) for item in out.get("s", []))]
        additions.extend(("P", item) for item in predicate.get("details", []))
        additions.extend(("O", item) for item in out.get("o", []))
        if additions:
            lines.extend(["", "**New details:**"])
            lines.extend(
                f"- {part} / {item.get('tag')}: {item.get('my')} | {item.get('en')}"
                for part, item in additions
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run(volume: int, page_number: int, output_name: str) -> Path:
    payload, _ = build_payload(volume, page_number)
    fixed_prompt = PROMPT_PATH.read_text(encoding="utf-8-sig").rstrip()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8-sig"))
    full_prompt = (
        fixed_prompt
        + "\n\n<USER_PAYLOAD>\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n</USER_PAYLOAD>\n"
    )

    output_dir = OUTPUT_ROOT / f"vol{volume}" / f"page_{page_number:04d}" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "payload.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "prompt_sent.txt").write_text(full_prompt, encoding="utf-8")

    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=6000,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = response.text or ""
    (output_dir / "raw_response.json").write_text(raw, encoding="utf-8")
    parsed = json.loads(raw)
    validator = load_validator()
    errors = validator.validate(payload, parsed)
    if not errors:
        (output_dir / "merged.json").write_text(
            json.dumps(validator.merge(payload, parsed), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    write_review(output_dir / "review.md", payload, parsed, errors)
    result = {
        "accepted": not errors,
        "errors": errors,
        "usage": response.usage_metadata.model_dump(mode="json") if response.usage_metadata else {},
        "response": parsed,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int, required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--output-name", default="delta_enrichment_01")
    args = parser.parse_args()
    print(run(args.volume, args.page, args.output_name))


if __name__ == "__main__":
    main()

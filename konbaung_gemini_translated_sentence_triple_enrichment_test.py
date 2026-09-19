#!/usr/bin/env python3
"""Prepare or run an additive enrichment pass over existing sentence triples."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from konbaung_gemini_cite_sources_annotator import MODEL
from konbaung_gemini_summary_claim_completion_annotator import api_key
from konbaung_gemini_translated_sentence_triples_test import (
    SentenceTriples,
    Triple,
    TripleResult,
    build_payload as build_v2_payload,
)


ROOT = Path(__file__).resolve().parent
V1_PROMPT_PATH = (
    ROOT
    / "konbaung_translated_sentence_triples_full_batch_20260713_high_thinking"
    / "cached_prefix.txt"
)
ENRICHMENT_PROMPT_PATH = ROOT / "konbaung_translated_sentence_triple_enrichment_prompt.md"
EXISTING_ROOT = ROOT / "konbaung_translated_sentence_triples_full_batch_20260713_high_thinking" / "pages"
OUTPUT_ROOT = ROOT / "konbaung_translated_sentence_triple_enrichment_tests"
PROMPT_EXAMPLE_PAGES = {
    (3, 133),
    (3, 177),
    (3, 221),
}


class EnrichmentResult(BaseModel):
    S: list[SentenceTriples] = Field(
        description=(
            "Only sentence records with genuinely additional triples; an empty list is valid "
            "when the existing annotation is complete."
        )
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def existing_result_path(volume: int, page_number: int) -> Path:
    return EXISTING_ROOT / f"vol{volume}" / f"page_{page_number:04d}" / "result.json"


def load_existing_annotation(volume: int, page_number: int) -> TripleResult:
    path = existing_result_path(volume, page_number)
    if not path.exists():
        raise FileNotFoundError(f"No existing v2 sentence-triple result: {path}")
    stored = read_json(path)
    if stored.get("accepted") is not True:
        raise ValueError(f"Existing v2 result is not accepted: {path}")
    return TripleResult.model_validate(stored["response"])


def build_payload(volume: int, page_number: int) -> dict[str, Any]:
    v2_payload = build_v2_payload(volume, page_number)
    existing = load_existing_annotation(volume, page_number)
    existing_by_sid = {
        group.sid: [triple.model_dump(mode="json") for triple in group.T]
        for group in existing.S
    }
    return {
        "page_id": v2_payload["page_id"],
        "summary": v2_payload["summary"],
        "sentences": [
            {
                **sentence,
                "existing_triples": existing_by_sid.get(sentence["id"], []),
            }
            for sentence in v2_payload["sentence_pairs"]
        ],
    }


def static_prefix() -> str:
    v1_prompt = V1_PROMPT_PATH.read_text(encoding="utf-8-sig").rstrip()
    enrichment_prompt = ENRICHMENT_PROMPT_PATH.read_text(encoding="utf-8-sig").rstrip()
    return f"""<ORIGINAL_V1_ANNOTATION_PROMPT_READ_ONLY>
{v1_prompt}
</ORIGINAL_V1_ANNOTATION_PROMPT_READ_ONLY>

<CURRENT_ENRICHMENT_TASK_INSTRUCTIONS>
{enrichment_prompt}
</CURRENT_ENRICHMENT_TASK_INSTRUCTIONS>"""


def dynamic_input(payload: dict[str, Any]) -> str:
    records: list[str] = []
    for sentence in payload["sentences"]:
        records.append(
            f"""<SENTENCE_RECORD id="{sentence['id']}">
<BURMESE_SENTENCE>
{sentence['my']}
</BURMESE_SENTENCE>
<ENGLISH_TRANSLATION>
{sentence['en']}
</ENGLISH_TRANSLATION>
<EXISTING_TRIPLES_READ_ONLY>
{json.dumps(sentence['existing_triples'], ensure_ascii=False, indent=2)}
</EXISTING_TRIPLES_READ_ONLY>
</SENTENCE_RECORD>"""
        )
    sentence_records = "\n\n".join(records)
    return f"""<ENRICHMENT_INPUT page_id="{payload['page_id']}">
<PAGE_SUMMARY_READ_ONLY>
{payload['summary']}
</PAGE_SUMMARY_READ_ONLY>

{sentence_records}
</ENRICHMENT_INPUT>
"""


def full_prompt(payload: dict[str, Any]) -> str:
    return f"{static_prefix()}\n\n{dynamic_input(payload)}"


def prompt_from_template(payload: dict[str, Any], template_path: Path | None) -> str:
    if template_path is None:
        return full_prompt(payload)
    template = template_path.read_text(encoding="utf-8-sig")
    pattern = re.compile(
        r'<ENRICHMENT_INPUT page_id="\{\{PAGE_ID\}\}">.*?</ENRICHMENT_INPUT>',
        re.DOTALL,
    )
    rendered, replacement_count = pattern.subn(dynamic_input(payload).rstrip(), template)
    if replacement_count != 1:
        raise ValueError(
            f"Expected exactly one ENRICHMENT_INPUT template block in {template_path}; "
            f"found {replacement_count}"
        )
    return rendered


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def triple_key(triple: Triple) -> tuple[str, str, str]:
    return (
        normalize(triple.s.my),
        triple.p.strip().upper(),
        normalize(triple.o.my),
    )


def validate(payload: dict[str, Any], result: EnrichmentResult) -> list[str]:
    errors: list[str] = []
    sentence_pairs = payload["sentences"]
    sentence_ids = [item["id"] for item in sentence_pairs]
    sentence_lookup = {item["id"]: item for item in sentence_pairs}
    page_text = "\n".join(item["my"] for item in sentence_pairs)
    known_ids = set(sentence_ids)

    observed_ids = [group.sid for group in result.S]
    unknown = sorted(set(observed_ids) - known_ids)
    duplicates = sorted(sid for sid in set(observed_ids) if observed_ids.count(sid) > 1)
    if unknown:
        errors.append(f"Unknown sentence IDs: {unknown}")
    if duplicates:
        errors.append(f"Duplicate sentence records: {duplicates}")
    order = {sid: index for index, sid in enumerate(sentence_ids)}
    observed_order = [order.get(sid, len(order)) for sid in observed_ids]
    if observed_order != sorted(observed_order):
        errors.append("Sentence records are not in supplied order")

    existing_by_sid: dict[str, set[tuple[str, str, str]]] = {}
    for sentence in sentence_pairs:
        existing_by_sid[sentence["id"]] = {
            triple_key(Triple.model_validate(item))
            for item in sentence["existing_triples"]
        }

    returned_keys: set[tuple[str, str, str, str]] = set()
    for group in result.S:
        if group.sid not in sentence_lookup:
            continue
        sentence_text = sentence_lookup[group.sid]["my"]
        for index, triple in enumerate(group.T, start=1):
            key = triple_key(triple)
            if key in existing_by_sid.get(group.sid, set()):
                errors.append(f"{group.sid} T[{index}] exactly duplicates an existing triple")
            returned_key = (group.sid, *key)
            if returned_key in returned_keys:
                errors.append(f"{group.sid} T[{index}] duplicates another returned triple")
            returned_keys.add(returned_key)

            for role, endpoint in (("s", triple.s), ("o", triple.o)):
                endpoint_text = normalize(endpoint.my)
                in_sentence = bool(endpoint_text) and endpoint_text in normalize(sentence_text)
                in_page = bool(endpoint_text) and endpoint_text in normalize(page_text)
                label = f"{group.sid} T[{index}].{role}"
                if endpoint.source == "sentence" and not in_sentence:
                    errors.append(f"{label}: source=sentence but Burmese referent is not in that sentence")
                elif endpoint.source == "page":
                    if not in_page:
                        errors.append(f"{label}: source=page but Burmese referent is not on the page")
                    elif in_sentence:
                        errors.append(f"{label}: source=page but Burmese referent is in the assigned sentence")
                elif endpoint.source == "inferred" and in_page:
                    errors.append(f"{label}: source=inferred but Burmese referent is explicit on the page")
    return errors


def triple_lines(index: int, triple: Triple) -> list[str]:
    return [
        f"{index}. Subject: `{triple.s.my}` — {triple.s.en} "
        f"[`{triple.s.tag}`; source=`{triple.s.source}`]",
        f"   Relation: `{triple.p}`",
        f"   Object: `{triple.o.my}` — {triple.o.en} "
        f"[`{triple.o.tag}`; source=`{triple.o.source}`]",
    ]


def review_text(
    payload: dict[str, Any],
    result: EnrichmentResult,
    errors: list[str] | None = None,
) -> str:
    existing_by_sid = {
        sentence["id"]: [
            Triple.model_validate(item) for item in sentence["existing_triples"]
        ]
        for sentence in payload["sentences"]
    }
    additions_by_sid = {group.sid: group.T for group in result.S}
    status = "VALID" if not errors else "FLAGGED"
    lines = [f"# {payload['page_id']} enrichment review", "", status, ""]
    total_sentences = len(payload["sentences"])
    enriched_sentences = sum(
        1 for sentence in payload["sentences"] if additions_by_sid.get(sentence["id"])
    )
    added_triples = sum(len(triples) for triples in additions_by_sid.values())
    lines.extend(
        [
            f"Gemini proposed {added_triples} additional triple(s) across "
            f"{enriched_sentences} of {total_sentences} supplied sentences.",
            "",
        ]
    )
    for sentence in payload["sentences"]:
        sid = sentence["id"]
        lines.extend(
            [
                f"## {sid}",
                "",
                "### Original Burmese sentence",
                "",
                sentence["my"],
                "",
                "### English translation",
                "",
                sentence["en"],
                "",
                "### Existing triples",
                "",
            ]
        )
        existing_triples = existing_by_sid.get(sid, [])
        if existing_triples:
            for index, triple in enumerate(existing_triples, start=1):
                lines.extend(triple_lines(index, triple))
                lines.append("")
        else:
            lines.extend(["None.", ""])
        lines.extend(["### Gemini proposed additions", ""])
        additions = additions_by_sid.get(sid, [])
        if additions:
            for index, triple in enumerate(additions, start=1):
                lines.extend(triple_lines(index, triple))
                lines.append("")
        else:
            lines.extend(["None proposed.", ""])
    if errors:
        lines.extend(["## Validation warnings", ""])
        lines.extend(f"- {error}" for error in errors)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def prepare(
    volume: int,
    page_number: int,
    output_name: str,
    prompt_template: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    payload = build_payload(volume, page_number)
    output_dir = OUTPUT_ROOT / f"vol{volume}" / f"page_{page_number:04d}" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "payload.json", payload)
    write_json(
        output_dir / "existing_annotation.json",
        {
            "S": [
                {"sid": sentence["id"], "T": sentence["existing_triples"]}
                for sentence in payload["sentences"]
            ]
        },
    )
    write_json(output_dir / "response_schema.json", EnrichmentResult.model_json_schema())
    (output_dir / "prompt_sent.txt").write_text(
        prompt_from_template(payload, prompt_template),
        encoding="utf-8",
    )
    review_path = output_dir / "review.md"
    if review_path.exists():
        review_path.unlink()
    return output_dir, payload


def run(
    volume: int,
    page_number: int,
    output_name: str,
    max_output_tokens: int,
    thinking_budget: int,
    prompt_template: Path | None = None,
) -> Path:
    if not 0 <= thinking_budget <= 2000:
        raise ValueError("thinking_budget must be between 0 and the 2000-token hard cap")
    if (volume, page_number) in PROMPT_EXAMPLE_PAGES:
        raise ValueError(
            f"vol{volume} page {page_number} is represented in the prompt examples and "
            "cannot be used for a live evaluation call"
        )
    output_dir, payload = prepare(volume, page_number, output_name, prompt_template)
    prompt_text = (output_dir / "prompt_sent.txt").read_text(encoding="utf-8")
    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt_text,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EnrichmentResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        ),
    )
    raw = response.text or ""
    (output_dir / "raw_response.json").write_text(raw, encoding="utf-8")
    result = EnrichmentResult.model_validate_json(raw)
    errors = validate(payload, result)
    write_json(
        output_dir / "result.json",
        {
            "accepted": not errors,
            "errors": errors,
            "request": {
                "model": MODEL,
                "thinking_budget": thinking_budget,
                "max_output_tokens": max_output_tokens,
                "prompt_template": str(prompt_template) if prompt_template else None,
            },
            "usage": response.usage_metadata.model_dump(mode="json")
            if response.usage_metadata
            else {},
            "response": result.model_dump(mode="json"),
        },
    )
    (output_dir / "review.md").write_text(
        review_text(payload, result, errors),
        encoding="utf-8",
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int, default=3)
    parser.add_argument("--page", type=int, default=133)
    parser.add_argument("--output-name", default="prompt_review_01")
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument(
        "--prompt-template",
        type=Path,
        help="Optional standalone prompt template ending in the standard ENRICHMENT_INPUT placeholder block.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=2000,
        help="Hard Gemini thinking-token budget for this enrichment call.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Call Gemini. Without this flag the command only prepares review artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run:
        output_dir = run(
            args.volume,
            args.page,
            args.output_name,
            args.max_output_tokens,
            args.thinking_budget,
            args.prompt_template,
        )
        called_api = True
    else:
        output_dir, _ = prepare(
            args.volume,
            args.page,
            args.output_name,
            args.prompt_template,
        )
        called_api = False
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "prompt": str(output_dir / "prompt_sent.txt"),
                "called_api": called_api,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

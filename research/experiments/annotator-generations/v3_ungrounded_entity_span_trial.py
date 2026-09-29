#!/usr/bin/env python3
"""Ground one page of spanless historiography-v3 entities to Burmese spans."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel

from konbaung_gemini_summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parent
CANONICAL_ROOT = (
    ROOT / "konbaung_reader_app" / "data" / "konbaung_historiography_v3_canonical_20260724"
)
READER_PAGES = ROOT / "konbaung_reader_app" / "static" / "data" / "konbaung" / "pages"
OUTPUT_ROOT = ROOT / "konbaung_v3_canonical_entity_span_trials"
MODEL = "gemini-3.1-flash-lite"

PROMPT = """You are grounding the subject and object entities of existing English analytical triples to one Burmese chronicle page and its complete associated sentences.

The supplied triples are intentionally spanless. For every triple, identify the subject and object entities and return an exact Burmese surface phrase in the supplied Burmese context that names each entity. The purpose is entity disambiguation: use all supplied context to resolve an entity that is implicit in its corresponding sentence or named there only indirectly. Do not rewrite, add, delete, merge, or split triples. Return every `triple_id` exactly once and in the supplied order.

For each subject and object:

- Return `status: "found"` only when `span` is an exact contiguous substring of either `physical_page_text` or any supplied `burmese_sentence`, and the phrase explicitly denotes the English entity in this specific triple.
- Copy a found span verbatim. Preserve the original Burmese spelling and characters. Never translate, normalize, correct, complete, or synthesize Burmese.
- Return the smallest complete nominal phrase that identifies the entity. Keep names, titles, quantities, and qualifiers needed to distinguish it, but exclude surrounding clause material and grammatical particles when they are not part of the entity.
- A compound entity may use one contiguous list phrase when the complete list is explicitly present.
- Use the English translation, the full triple, its corresponding Burmese sentence, the physical page, and all other supplied sentences to disambiguate which Burmese phrase names the endpoint.
- When the entity is implicit, pronominal, or indirect in its corresponding sentence, use an explicit exact mention elsewhere in the supplied Burmese context when the context clearly establishes that it is the same entity.
- Return `status: "not_found"` and `span: ""` only when no explicit Burmese mention of the entity can be matched confidently to one exact contiguous phrase anywhere in the supplied Burmese context.

A supplied `burmese_sentence` may cross a physical page boundary. Its entire text is valid source context, including portions lying on an adjacent physical page.

Return only schema-valid JSON."""


class EndpointSpan(BaseModel):
    status: Literal["found", "not_found"]
    span: str


class TripleSpans(BaseModel):
    triple_id: str
    subject: EndpointSpan
    object: EndpointSpan


class SpanResult(BaseModel):
    triples: list[TripleSpans]


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_canonical_owner_page(
    volume: int,
    page_number: int,
) -> list[dict[str, Any]]:
    path = CANONICAL_ROOT / "sentences" / f"vol{volume}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    volume_data = json.loads(path.read_text(encoding="utf-8"))
    sentences = [
        sentence
        for sentence in volume_data["sentences"].values()
        if int(sentence["ownerPage"]) == page_number and sentence["triples"]
    ]
    if not sentences:
        raise ValueError(f"Canonical page vol{volume}-p{page_number:04d} has no triples")
    return sorted(sentences, key=lambda sentence: sentence["sid"])


def read_physical_page(volume: int, page_number: int) -> dict[str, Any]:
    path = READER_PAGES / f"vol{volume}" / f"{page_number:04d}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_input(
    canonical_sentences: list[dict[str, Any]],
    physical_page: dict[str, Any],
    volume: int,
    page_number: int,
) -> dict[str, Any]:
    sentences: list[dict[str, Any]] = []
    for source in canonical_sentences:
        triples = [
            {
                "triple_id": f"v3-{source['sid']}-t{ordinal:03d}",
                "subject": triple["subject"],
                "predicate": triple["predicate"],
                "object": triple["object"],
            }
            for ordinal, triple in enumerate(source["triples"], start=1)
        ]
        sentences.append(
            {
                "sentence_id": source["sid"],
                "owner_page": source["ownerPage"],
                "physical_pages": source["pages"],
                "cross_page": len(source["pages"]) > 1,
                "burmese_sentence": source["my"],
                "english_translation": source["en"],
                "triples": triples,
            }
        )
    return {
        "page_id": f"vol{volume}-p{page_number:04d}",
        "source": "reader_canonical_historiography_v3_27129",
        "deduplication": "one canonical sentence ID assigned to its owner page",
        "physical_page_text": normalize_whitespace(physical_page["canonicalText"]),
        "sentences": sentences,
    }


def validate(
    supplied: dict[str, Any],
    result: SpanResult,
) -> tuple[list[str], list[str], list[dict[str, Any]], dict[str, int]]:
    input_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for sentence in supplied["sentences"]:
        for triple in sentence["triples"]:
            input_rows.append((sentence, triple))

    expected_ids = [triple["triple_id"] for _, triple in input_rows]
    observed_ids = [triple.triple_id for triple in result.triples]
    errors: list[str] = []
    model_flags: list[str] = []
    if observed_ids != expected_ids:
        errors.append(f"triple IDs/order mismatch: expected {expected_ids}, got {observed_ids}")

    input_by_id = {triple["triple_id"]: (sentence, triple) for sentence, triple in input_rows}
    page_text = supplied["physical_page_text"]
    supplied_burmese_contexts = [
        page_text,
        *(sentence["burmese_sentence"] for sentence in supplied["sentences"]),
    ]
    audited: list[dict[str, Any]] = []
    counts = {
        "input_triples": len(expected_ids),
        "input_endpoints": len(expected_ids) * 2,
        "found": 0,
        "not_found": 0,
    }

    for returned in result.triples:
        source_pair = input_by_id.get(returned.triple_id)
        if source_pair is None:
            errors.append(f"{returned.triple_id}: unknown triple ID")
            continue
        sentence, original = source_pair
        audit_row: dict[str, Any] = {
            "triple_id": returned.triple_id,
            "subject": original["subject"],
            "predicate": original["predicate"],
            "object": original["object"],
        }
        for role in ("subject", "object"):
            endpoint = getattr(returned, role)
            model_span = endpoint.span
            role_errors: list[str] = []
            if endpoint.status == "found":
                if not model_span:
                    role_errors.append("found span is empty")
                if not any(model_span in context for context in supplied_burmese_contexts):
                    role_errors.append(
                        "span is not an exact substring of any supplied Burmese context"
                    )
            else:
                if model_span:
                    role_errors.append("not_found span is not empty")

            if role_errors:
                model_flags.append(f"{returned.triple_id} {role}: " + "; ".join(role_errors))
                final_status = "not_found"
                final_span = ""
            else:
                final_status = endpoint.status
                final_span = model_span
            counts[final_status] += 1

            audit_row[f"{role}_grounding"] = {
                "status": final_status,
                "span": final_span,
                "model_status": endpoint.status,
                "model_span": model_span,
                "in_physical_page": bool(model_span) and model_span in page_text,
                "in_corresponding_sentence": (
                    bool(model_span) and model_span in sentence["burmese_sentence"]
                ),
                "flag_reason": "; ".join(role_errors),
                "validation_errors": role_errors,
            }
        audited.append(audit_row)
    return errors, model_flags, audited, counts


def review_markdown(
    supplied: dict[str, Any],
    audited: list[dict[str, Any]],
    errors: list[str],
    model_flags: list[str],
) -> str:
    audits = {row["triple_id"]: row for row in audited}
    lines = [
        f"# Spanless v3 entity grounding trial: {supplied['page_id']}",
        "",
        (
            "VALIDATED"
            if not model_flags and not errors
            else (
                f"VALIDATED WITH {len(model_flags)} MODEL OUTPUTS FLAGGED"
                if not errors
                else "INVALID"
            )
        ),
        "",
    ]
    for error in errors:
        lines.append(f"- {error}")
    if errors:
        lines.append("")
    for flag in model_flags:
        lines.append(f"- {flag}")
    if model_flags:
        lines.append("")

    for sentence in supplied["sentences"]:
        lines.extend(
            [
                f"## {sentence['sentence_id']}",
                "",
                f"**Burmese:** {sentence['burmese_sentence']}",
                "",
                f"**English:** {sentence['english_translation']}",
                "",
            ]
        )
        for triple in sentence["triples"]:
            audit = audits.get(triple["triple_id"])
            if not audit:
                continue
            subject = audit["subject_grounding"]
            object_ = audit["object_grounding"]
            subject_display = (
                f"`{subject['span']}`" if subject["status"] == "found" else "**NOT FOUND**"
            )
            if subject["flag_reason"]:
                subject_display += (
                    f" — rejected model span `{subject['model_span']}` ({subject['flag_reason']})"
                )
            object_display = (
                f"`{object_['span']}`" if object_["status"] == "found" else "**NOT FOUND**"
            )
            if object_["flag_reason"]:
                object_display += (
                    f" — rejected model span `{object_['model_span']}` ({object_['flag_reason']})"
                )
            lines.extend(
                [
                    f"### {triple['triple_id']}",
                    "",
                    (
                        f"**English triple:** {triple['subject']} "
                        f"— `{triple['predicate']}` → {triple['object']}"
                    ),
                    "",
                    f"- Burmese subject: {subject_display}",
                    f"- Burmese object: {object_display}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def save_validated_result(
    output_dir: Path,
    supplied: dict[str, Any],
    result: SpanResult,
    usage: dict[str, Any],
    created: str | None = None,
) -> dict[str, Any]:
    errors, model_flags, audited, counts = validate(supplied, result)
    validation = {
        "accepted": not errors,
        "errors": errors,
        "model_output_flags": model_flags,
        **counts,
    }
    write_json(
        output_dir / "result.json",
        {
            "created": created or datetime.now(timezone.utc).isoformat(),
            "page_id": supplied["page_id"],
            "source": supplied["source"],
            "model": MODEL,
            "thinking_level": "minimal",
            "usage": usage,
            "validation": validation,
            "triples": audited,
        },
    )
    (output_dir / "review.md").write_text(
        review_markdown(supplied, audited, errors, model_flags),
        encoding="utf-8",
    )
    return validation


def reprocess_existing(output_dir: Path) -> dict[str, Any]:
    supplied = json.loads((output_dir / "input.json").read_text(encoding="utf-8"))
    raw = (output_dir / "raw_response.json").read_text(encoding="utf-8")
    result = SpanResult.model_validate_json(raw)
    old_result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    return save_validated_result(
        output_dir,
        supplied,
        result,
        old_result.get("usage", {}),
        old_result.get("created"),
    )


def run(volume: int, page_number: int, output_name: str) -> Path:
    canonical_sentences = read_canonical_owner_page(volume, page_number)
    physical_page = read_physical_page(volume, page_number)
    supplied = build_input(
        canonical_sentences,
        physical_page,
        volume,
        page_number,
    )
    prompt = (
        PROMPT
        + "\n\nSUPPLIED PAGE, SENTENCES, AND SPANLESS TRIPLES:\n"
        + json.dumps(supplied, ensure_ascii=False, indent=2)
        + "\n"
    )

    output_dir = OUTPUT_ROOT / f"vol{volume}" / f"page_{page_number:04d}" / output_name
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "input.json", supplied)
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SpanResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
                include_thoughts=False,
            ),
        ),
    )
    raw = response.text or ""
    (output_dir / "raw_response.json").write_text(raw + "\n", encoding="utf-8")
    result = SpanResult.model_validate_json(raw)
    usage = response.usage_metadata.model_dump(mode="json") if response.usage_metadata else {}
    validation = save_validated_result(output_dir, supplied, result, usage)
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "page_id": supplied["page_id"],
                "validation": validation,
                "usage": usage,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int, required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument(
        "--output-name",
        default="verbatim_entity_spans_minimal_thinking_01",
    )
    parser.add_argument(
        "--reprocess-existing",
        action="store_true",
        help="Revalidate an existing raw response without another API call.",
    )
    args = parser.parse_args()
    if args.reprocess_existing:
        output_dir = OUTPUT_ROOT / f"vol{args.volume}" / f"page_{args.page:04d}" / args.output_name
        validation = reprocess_existing(output_dir)
        print(
            json.dumps(
                {"output": str(output_dir), "validation": validation},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        run(args.volume, args.page, args.output_name)


if __name__ == "__main__":
    main()

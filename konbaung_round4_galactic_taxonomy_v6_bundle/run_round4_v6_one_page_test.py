#!/usr/bin/env python3
"""Run the Round 4 V6 galactic-vector taxonomy on one canonical Konbaung page."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = Path(__file__).resolve().parent
PAGE_ID = os.getenv("KONBAUNG_PAGE_ID", "vol1-p0047")
PAGE_VOLUME_ID = PAGE_ID.split("-", 1)[0]
OUT_DIR = ROOT / os.getenv(
    "KONBAUNG_OUT_DIR",
    "konbaung_round4_galactic_taxonomy_v6_one_page_test_20260802",
)
SELECTION_METHOD = os.getenv("KONBAUNG_SELECTION_METHOD", "manual")
MODEL = "gemini-3.1-flash-lite"
MAX_OUTPUT_TOKENS = int(os.getenv("KONBAUNG_MAX_OUTPUT_TOKENS", "12000"))

sys.path.insert(0, str(ROOT))
from konbaung_gemini_summary_claim_completion_annotator import api_key

V1_V2_CONTEXT_PATH = BUNDLE_DIR / "konbaung_v1_v2_prompt_no_axial_schema.txt"
TAXONOMY_PROMPT_PATH = BUNDLE_DIR / "konbaung_galactic_vector_taxonomy_v6.md"
CATEGORY_SCHEMA_PATH = BUNDLE_DIR / "konbaung_galactic_vector_taxonomy_v6.json"
SUBMITTED_PROMPT_NAME = "konbaung_round4_galactic_taxonomy_v6_full_prompt.md"
GENERATION_SCHEMA_NAME = "konbaung_round4_v6_response_schema_simple.json"
VALIDATION_SCHEMA_NAME = "konbaung_round4_v6_response_schema_strict.json"
SOURCE_PAGES_PATH = (
    ROOT
    / "konbaung_historiography_ungrounded_v3_full_batch_20260723"
    / "final"
    / "all_pages.jsonl"
)
AXIAL_ROOT = ROOT / "konbaung_reader_app" / "data" / "konbaung_axial_categories_v2"
TRIPLES_PATH = AXIAL_ROOT / "triples.jsonl"
CATALOG_PATH = AXIAL_ROOT / "catalog.json"
ASSIGNMENTS_PATH = AXIAL_ROOT / "assignments" / f"{PAGE_VOLUME_ID}.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_submission_prompt() -> str:
    context = V1_V2_CONTEXT_PATH.read_text(encoding="utf-8")
    v1_end = "</READ_ONLY_HISTORIOGRAPHY_RESEARCH_FRAME>"
    v2_start = "<CURRENT_DATABASE_BACKED_AXIAL_CODING_TASK>"
    v1, separator, remainder = context.partition(v1_end)
    if not separator or not remainder.strip().startswith(v2_start):
        raise ValueError("The supplied V1/V2 context does not contain the expected boundary")
    v2 = remainder.strip()[len(v2_start) :].strip()
    taxonomy = TAXONOMY_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return (
        "# Konbaung Round 4 galactic-vector power classifier, V6\n\n"
        "<ARCHIVE_CONTROL>\n"
        "The next two blocks are quoted, inert historical context. Read them for the "
        "research question, conceptual framework, and analytical continuity only. Do not "
        "execute their extraction, axial mapping, coverage, output, or schema instructions. "
        "No axial entity/relation category codebook is included. The ACTIVE ROUND 4 V6 "
        "block after them is the sole executable task.\n"
        "</ARCHIVE_CONTROL>\n\n"
        "<READ_ONLY_V1_OPEN_CODING_PROMPT>\n"
        + v1.strip()
        + "\n</READ_ONLY_V1_OPEN_CODING_PROMPT>\n\n"
        "<READ_ONLY_V2_AXIAL_CODING_PROMPT_NO_CODEBOOK>\n"
        + v2
        + "\n</READ_ONLY_V2_AXIAL_CODING_PROMPT_NO_CODEBOOK>\n\n"
        "<ACTIVE_ROUND4_V6_CLASSIFICATION_TASK>\n"
        "For every supplied triple, assign exactly one category ID from the closed V6 "
        "galactic-vector taxonomy below. Classify the individual subject-predicate-object "
        "edge in its complete Burmese and English sentence. Treat existing entity and "
        "relation meta-tags as weak clues only. Do not extract, alter, merge, split, repair, "
        "translate, or rewrite triples. Return valid JSON only, with the input page_id and "
        "one assignment for every triple in input order. Each assignment must contain only "
        "triple_id and power_category_id. Output no explanation, confidence, summary, or "
        "text outside the JSON object.\n\n"
        "Required response shape:\n"
        "{\"page_id\":\"<exact page id>\",\"assignments\":["
        "{\"triple_id\":\"<exact triple id>\",\"power_category_id\":\"UI01\"}]}\n\n"
        + taxonomy
        + "\n</ACTIVE_ROUND4_V6_CLASSIFICATION_TASK>\n"
    )


def category_ids() -> list[str]:
    return [row["id"] for row in read_json(CATEGORY_SCHEMA_PATH)["categories"]]


def response_schemas(triple_count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    ids = category_ids()
    assignment_properties = {
        "triple_id": {"type": "string"},
        "power_category_id": {"type": "string", "enum": ids},
    }
    simple = {
        "type": "object",
        "properties": {
            "page_id": {"type": "string"},
            "assignments": {
                "type": "array",
                "minItems": triple_count,
                "maxItems": triple_count,
                "items": {
                    "type": "object",
                    "properties": assignment_properties,
                    "required": ["triple_id", "power_category_id"],
                },
            },
        },
        "required": ["page_id", "assignments"],
    }
    strict = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "page_id": {"type": "string"},
            "assignments": {
                "type": "array",
                "minItems": triple_count,
                "maxItems": triple_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": assignment_properties,
                    "required": ["triple_id", "power_category_id"],
                },
            },
        },
        "required": ["page_id", "assignments"],
    }
    return simple, strict


def load_jsonl_match(path: Path, predicate: Any) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if predicate(record):
                    matches.append(record)
    return matches


def load_page_summary() -> str:
    matches = load_jsonl_match(SOURCE_PAGES_PATH, lambda row: row["key"] == PAGE_ID)
    if len(matches) != 1:
        raise ValueError(f"Expected one source-page record for {PAGE_ID}, found {len(matches)}")
    return matches[0]["summary"]


def verify_accepted_page(rows: list[dict[str, Any]]) -> None:
    assignment_document = read_json(ASSIGNMENTS_PATH)
    sentence_assignments = assignment_document["sentences"]
    sentence_ids = list(dict.fromkeys(row["sid"] for row in rows))
    for sentence_id in sentence_ids:
        assignment = sentence_assignments[sentence_id]
        if assignment["status"] != "accepted" or assignment["sourcePage"] != PAGE_ID:
            raise ValueError(f"{sentence_id} is not an accepted assignment from {PAGE_ID}")


def build_input() -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    rows = load_jsonl_match(TRIPLES_PATH, lambda row: row["selectedPage"] == PAGE_ID)
    if not rows:
        raise ValueError(f"No canonical triples found for {PAGE_ID}")
    verify_accepted_page(rows)

    catalog = read_json(CATALOG_PATH)
    entity_labels = {row["id"]: row["label"] for row in catalog["entities"]}
    relation_labels = {row["id"]: row["label"] for row in catalog["relations"]}

    sentences: list[dict[str, Any]] = []
    sentences_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["sid"] not in sentences_by_id:
            sentence = {
                "sentence_id": row["sid"],
                "my": row["sentenceMy"],
                "en": row["sentenceEn"],
                "triples": [],
            }
            sentences_by_id[row["sid"]] = sentence
            sentences.append(sentence)
        tags = row["categories"]
        sentences_by_id[row["sid"]]["triples"].append(
            {
                "triple_id": row["id"],
                "sentence_id": row["sid"],
                "subject": {
                    "en": row["subject"],
                    "entity_tag_id": tags["s"],
                    "entity_tag_label": entity_labels[tags["s"]],
                },
                "predicate": {
                    "en": row["predicate"],
                    "relation_tag_id": tags["r"],
                    "relation_tag_label": relation_labels[tags["r"]],
                },
                "object": {
                    "en": row["object"],
                    "entity_tag_id": tags["o"],
                    "entity_tag_label": entity_labels[tags["o"]],
                },
            }
        )

    payload = {
        "page_id": PAGE_ID,
        "page_summary": load_page_summary(),
        "sentences": sentences,
    }
    return payload, entity_labels, relation_labels


def payload_triples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        triple
        for sentence in payload["sentences"]
        for triple in sentence["triples"]
    ]


def validate_result(result: dict[str, Any], payload: dict[str, Any]) -> None:
    strict_schema = response_schemas(len(payload_triples(payload)))[1]
    errors = sorted(
        Draft202012Validator(strict_schema).iter_errors(result),
        key=lambda error: list(error.path),
    )
    if errors:
        messages = [f"{list(error.path)}: {error.message}" for error in errors]
        raise ValueError("Strict response-schema validation failed: " + "; ".join(messages))

    expected_ids = [row["triple_id"] for row in payload_triples(payload)]
    actual_ids = [row["triple_id"] for row in result["assignments"]]
    if result["page_id"] != payload["page_id"]:
        raise ValueError("Response page_id does not match the input")
    if actual_ids != expected_ids:
        raise ValueError("Response triple IDs do not exactly preserve input order")


def render_review(
    payload: dict[str, Any],
    result: dict[str, Any],
    usage: dict[str, Any],
) -> str:
    category_schema = read_json(CATEGORY_SCHEMA_PATH)
    category_labels = {row["id"]: row["name"] for row in category_schema["categories"]}
    vector_labels = {
        "UPWARD_INWARD": "Upward / Inward",
        "DOWNWARD_INWARD": "Downward / Inward",
        "DOWNWARD_OUTWARD": "Downward / Outward",
        "UPWARD_OUTWARD": "Upward / Outward",
        "NONE": "No clear vector",
    }
    category_vectors = {
        row["id"]: vector_labels[row["vector"]]
        for row in category_schema["categories"]
    }
    assignments = {row["triple_id"]: row for row in result["assignments"]}
    triples = payload_triples(payload)

    lines = [
        "# Konbaung Round 4 V6 galactic-vector annotation review",
        "",
        f"- Page: `{payload['page_id']}`",
        f"- Model: `{MODEL}`",
        f"- Page selection: {SELECTION_METHOD}",
        f"- Submitted and packaged prompt: `{SUBMITTED_PROMPT_NAME}`",
        "- Prompt composition: original open-coding V1 → V2 axial-coding task without its codebook → ACTIVE Round 4 V6 taxonomy",
        "- Giant axial E/R schema submitted: no; only each triple's existing S/P/O meta tags were supplied",
        f"- Category schema: `{CATEGORY_SCHEMA_PATH.name}`",
        f"- Generation schema: `{GENERATION_SCHEMA_NAME}`",
        f"- Validation schema: `{VALIDATION_SCHEMA_NAME}`",
        f"- Triple count: {len(triples)}",
        f"- Validation: passed",
        f"- Prompt tokens: {usage.get('prompt_token_count', 'n/a')}",
        f"- Output tokens: {usage.get('candidates_token_count', 'n/a')}",
        f"- Thinking tokens: {usage.get('thoughts_token_count', 'n/a')}",
        "",
        "## Existing page summary",
        "",
        payload["page_summary"],
        "",
        "## Sentences, triples, meta tags, and new annotations",
        "",
    ]

    for sentence in payload["sentences"]:
        sentence_id = sentence["sentence_id"]
        lines.extend(
            [
                f"### `{sentence_id}`",
                "",
                "**Burmese**",
                "",
                sentence["my"],
                "",
                "**English**",
                "",
                sentence["en"],
                "",
                "| Triple | Existing S/P/O | Existing meta tags | New annotation |",
                "|---|---|---|---|",
            ]
        )
        for triple in sentence["triples"]:
            assignment = assignments[triple["triple_id"]]
            category_id = assignment["power_category_id"]
            annotation = (
                f"{category_id} — {category_labels[category_id]} — "
                f"Vector: {category_vectors[category_id]}"
            )
            spo = (
                f"{triple['subject']['en']} → {triple['predicate']['en']} → "
                f"{triple['object']['en']}"
            )
            meta = (
                f"{triple['subject']['entity_tag_id']} "
                f"({triple['subject']['entity_tag_label']}) / "
                f"{triple['predicate']['relation_tag_id']} "
                f"({triple['predicate']['relation_tag_label']}) / "
                f"{triple['object']['entity_tag_id']} "
                f"({triple['object']['entity_tag_label']})"
            )
            lines.append(
                f"| `{triple['triple_id']}` | {spo} | {meta} | {annotation} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Raw new annotations",
            "",
            "```json",
            json.dumps(result, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    payload, _, _ = build_input()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "input.json", payload)
    packaged_category_schema_path = OUT_DIR / CATEGORY_SCHEMA_PATH.name
    shutil.copy2(CATEGORY_SCHEMA_PATH, packaged_category_schema_path)
    prompt = build_submission_prompt()
    packaged_submitted_prompt_path = OUT_DIR / SUBMITTED_PROMPT_NAME
    write_text(packaged_submitted_prompt_path, prompt)
    page_text = (
        "\n\n<PAGE_DATA_JSON>\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n</PAGE_DATA_JSON>\n"
    )
    triple_count = len(payload_triples(payload))
    generation_schema, validation_schema = response_schemas(triple_count)
    generation_schema_path = OUT_DIR / GENERATION_SCHEMA_NAME
    validation_schema_path = OUT_DIR / VALIDATION_SCHEMA_NAME
    write_json(generation_schema_path, generation_schema)
    write_json(validation_schema_path, validation_schema)

    key = api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=prompt), types.Part(text=page_text)],
            )
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=generation_schema,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW,
            ),
        ),
    )

    raw_response = response.model_dump(mode="json", exclude_none=True)
    write_json(OUT_DIR / "raw_response.json", raw_response)
    result = json.loads(response.text)
    write_json(OUT_DIR / "result.json", result)
    validate_result(result, payload)

    usage = raw_response.get("usage_metadata", {})
    write_text(OUT_DIR / "review.md", render_review(payload, result, usage))
    write_json(
        OUT_DIR / "run_manifest.json",
        {
            "status": "completed_valid",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "page_id": PAGE_ID,
            "page_selection": SELECTION_METHOD,
            "model": MODEL,
            "thinking_level": "low",
            "temperature": 0.0,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "packaged_submitted_prompt_path": str(packaged_submitted_prompt_path),
            "submitted_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_composition": "V1 open-coding research frame; V2 axial-coding task without codebook; ACTIVE Round 4 V6 taxonomy",
            "v1_v2_context_source": str(V1_V2_CONTEXT_PATH),
            "giant_axial_entity_relation_schema_submitted": False,
            "category_schema_source_path": str(CATEGORY_SCHEMA_PATH),
            "packaged_category_schema_path": str(packaged_category_schema_path),
            "category_schema_sha256": sha256(CATEGORY_SCHEMA_PATH),
            "generation_schema_path": str(generation_schema_path),
            "generation_schema_sha256": sha256(generation_schema_path),
            "validation_schema_path": str(validation_schema_path),
            "validation_schema_sha256": sha256(validation_schema_path),
            "triple_count": triple_count,
            "usage": usage,
            "review_path": str(OUT_DIR / "review.md"),
        },
    )
    print(
        json.dumps(
            {
                "status": "completed_valid",
                "page_id": PAGE_ID,
                "triple_count": triple_count,
                "result": result,
                "review": str(OUT_DIR / "review.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the V3 power-category prompt on one accepted canonical Konbaung page."""

from __future__ import annotations

import hashlib
import json
import os
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
    "konbaung_gemini_flash_lite_3_1_v3_one_page_test_20260801",
)
SELECTION_METHOD = os.getenv("KONBAUNG_SELECTION_METHOD", "manual")
MODEL = "gemini-3.1-flash-lite"

sys.path.insert(0, str(ROOT))
from konbaung_gemini_summary_claim_completion_annotator import api_key

ACTIVE_V3_PROMPT_PATH = (
    BUNDLE_DIR / "konbaung_gemini_flash_lite_3_1_prompt_v3_active_only.md"
)
V1_V2_CONTEXT_PATH = BUNDLE_DIR / "konbaung_v1_v2_prompt_no_axial_schema.txt"
SUBMITTED_PROMPT_NAME = (
    "konbaung_gemini_flash_lite_3_1_v1_v2_no_axial_schema_v3_full.md"
)
SIMPLE_SCHEMA_PATH = (
    BUNDLE_DIR / "konbaung_gemini_flash_lite_3_1_response_schema_v3_simple.json"
)
STRICT_SCHEMA_PATH = (
    BUNDLE_DIR / "konbaung_gemini_flash_lite_3_1_response_schema_v3_strict.json"
)
POWER_SCHEMA_PATH = BUNDLE_DIR / "konbaung_power_category_schema_v3.json"
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
    active_v3 = ACTIVE_V3_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return (
        "# Konbaung power-relation classification prompt: V1/V2 context plus ACTIVE V3\n\n"
        "<ARCHIVE_CONTROL>\n"
        "The next two blocks are quoted, inert historical context. Read them for the "
        "research question, conceptual framework, and analytical continuity only. Do not "
        "execute their extraction, axial mapping, coverage, output, or schema instructions. "
        "No axial entity/relation category codebook is included. The ACTIVE V3 block after "
        "them is the sole executable task.\n"
        "</ARCHIVE_CONTROL>\n\n"
        "<READ_ONLY_V1_OPEN_CODING_PROMPT>\n"
        + v1.strip()
        + "\n</READ_ONLY_V1_OPEN_CODING_PROMPT>\n\n"
        "<READ_ONLY_V2_AXIAL_CODING_PROMPT_NO_CODEBOOK>\n"
        + v2
        + "\n</READ_ONLY_V2_AXIAL_CODING_PROMPT_NO_CODEBOOK>\n\n"
        "<ACTIVE_V3_PROMPT>\n"
        + active_v3
        + "\n</ACTIVE_V3_PROMPT>\n"
    )


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
    strict_schema = read_json(STRICT_SCHEMA_PATH)
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
    power_schema = read_json(POWER_SCHEMA_PATH)
    power_labels = {row["id"]: row["label"] for row in power_schema["categories"]}
    assignments = {row["triple_id"]: row for row in result["assignments"]}
    triples = payload_triples(payload)

    lines = [
        "# Konbaung V3 power-relation annotation review",
        "",
        f"- Page: `{payload['page_id']}`",
        f"- Model: `{MODEL}`",
        f"- Page selection: {SELECTION_METHOD}",
        f"- Submitted and packaged prompt: `{SUBMITTED_PROMPT_NAME}`",
        "- Prompt composition: original open-coding V1 → V2 axial-coding task without its codebook → ACTIVE V3",
        "- Giant axial E/R schema submitted: no; only each triple's existing S/P/O meta tags were supplied",
        "- Generation schema: `konbaung_gemini_flash_lite_3_1_response_schema_v3_simple.json`",
        "- Validation schema: `konbaung_gemini_flash_lite_3_1_response_schema_v3_strict.json`",
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
            if category_id == "PROVISIONAL":
                annotation = (
                    f"PROVISIONAL — {assignment['provisional_category_name']}: "
                    f"{assignment['provisional_category_definition']}"
                )
            elif category_id == "NO_CLEAR_POWER_RELATION":
                annotation = "NO_CLEAR_POWER_RELATION"
            else:
                annotation = f"{category_id} — {power_labels[category_id]}"
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
    prompt = build_submission_prompt()
    packaged_submitted_prompt_path = OUT_DIR / SUBMITTED_PROMPT_NAME
    write_text(packaged_submitted_prompt_path, prompt)
    page_text = (
        "\n\n<PAGE_DATA_JSON>\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n</PAGE_DATA_JSON>\n"
    )
    generation_schema = read_json(SIMPLE_SCHEMA_PATH)
    triple_count = len(payload_triples(payload))
    generation_schema["properties"]["assignments"]["minItems"] = triple_count
    generation_schema["properties"]["assignments"]["maxItems"] = triple_count

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
            max_output_tokens=4000,
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
            "packaged_submitted_prompt_path": str(packaged_submitted_prompt_path),
            "submitted_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_composition": "V1 open-coding research frame; V2 axial-coding task without codebook; ACTIVE V3",
            "v1_v2_context_source": str(V1_V2_CONTEXT_PATH),
            "giant_axial_entity_relation_schema_submitted": False,
            "generation_schema_path": str(SIMPLE_SCHEMA_PATH),
            "generation_schema_sha256": sha256(SIMPLE_SCHEMA_PATH),
            "validation_schema_path": str(STRICT_SCHEMA_PATH),
            "validation_schema_sha256": sha256(STRICT_SCHEMA_PATH),
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

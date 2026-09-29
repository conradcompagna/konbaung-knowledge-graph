#!/usr/bin/env python3
"""Run one readable, flat, no-thinking full-rewrite repair trial."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

import konbaung_gemini_v4_delta_repair_trial as corpus
from konbaung_gemini_summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parent
MODEL = "gemini-3.1-flash-lite"
TRIAL_NAME = "flat_full_rewrite_verbatim_no_thinking_trial_01"
SEED = 2026071803

PROMPT = """You will receive one page of the Konbaung Chronicle. Each Burmese sentence is immediately followed by its English translation and its existing triples.

Rewrite every supplied triple. Every input ID must appear exactly once: either as a complete rewritten triple in `R` or as an ID in `D`. Do not skip any triple, add facts, split facts, or merge facts.

For each coherent triple, write these fields in the supplied order:

1. Burmese subject, predicate, and object.
2. English glosses of that subject, predicate, and object.
3. Analytical tags for that subject, predicate, and object.

Read each result literally as:

subject -> predicate -> object

The three components must form the most natural, semantically coherent version of the specific relation recorded by the input triple. Correct wrongly attached components, repeated components, incorrect direction, confused recipients or contents, and predicates assigned to the wrong subject or object.

VERBATIM BURMESE SPANS ARE MANDATORY. Copy every text-grounded `s_my`, `p_my`, and `o_my` exactly and contiguously from the supplied Burmese page. Preserve the original characters, spaces, particles, and spelling. Never translate back into Burmese, normalize the wording, combine separate fragments, insert ellipses, or invent a cleaner Burmese predicate. The predicate must be an actual Burmese predicate span in the supplied page.

A subject or object that is grammatically omitted may remain inferred only if the existing triple already marks that endpoint as inferred. In that case, preserve its existing Burmese referent rather than inventing another one. Predicates may not be inferred.

Each English field must directly translate or transliterate its corresponding Burmese field. In particular, `p_en` must translate `p_my`; it must not paraphrase the analytical relation tag.

The tags are open-coded analytical classifications one useful step above literal translation. `s_tag` and `o_tag` classify the historical entities; `p_tag` classifies the historical relation. Do not mechanically convert an English gloss into PascalCase or UPPER_SNAKE_CASE.

Place an ID in `D` only when the existing annotation adds no useful relation and cannot be rewritten as a coherent subject-predicate-object proposition supported by the supplied text. Do not delete a useful relation merely because its subject or object is legitimately inferred.

Return only schema-valid JSON. Do not include explanations or unchanged input formatting."""


FIELDS = (
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def select_payload() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    corpus.SEED = SEED
    corpus.USED_PAGES = set(corpus.USED_PAGES) | {"vol2-p0353"}
    return corpus.build_payload()


def all_input_triples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [triple for sentence in payload["S"] for triple in sentence["T"]]


def render_input(payload: dict[str, Any]) -> str:
    lines = [f'<PAGE id="{payload["page_id"]}">', ""]
    for sentence in payload["S"]:
        lines.extend(
            [
                f'<SENTENCE id="{sentence["sid"]}">',
                "BURMESE:",
                sentence["my"],
                "",
                "ENGLISH:",
                sentence["en"],
                "",
                "EXISTING TRIPLES:",
            ]
        )
        if not sentence["T"]:
            lines.append("NONE")
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
    row = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            **{field: {"type": "string"} for field in FIELDS},
        },
        "required": ["id", *FIELDS],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "R": {"type": "array", "items": row},
            "D": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["R", "D"],
    }


def old_flat(triple: dict[str, Any]) -> dict[str, str]:
    return {
        "id": triple["id"],
        "s_my": triple["s"]["my"],
        "p_my": triple["predicate_grounding"]["my"],
        "o_my": triple["o"]["my"],
        "s_en": triple["s"]["en"],
        "p_en": triple["predicate_grounding"]["en"],
        "o_en": triple["o"]["en"],
        "s_tag": triple["s"]["tag"],
        "p_tag": triple["p"],
        "o_tag": triple["o"]["tag"],
    }


def display(row: dict[str, str]) -> dict[str, Any]:
    return {
        "s": {"my": row["s_my"], "en": row["s_en"], "tag": row["s_tag"]},
        "p": {"my": row["p_my"], "en": row["p_en"], "tag": row["p_tag"]},
        "o": {"my": row["o_my"], "en": row["o_en"], "tag": row["o_tag"]},
    }


def validate(payload: dict[str, Any], result: dict[str, Any]) -> list[str]:
    triples = all_input_triples(payload)
    expected = [triple["id"] for triple in triples]
    old = {triple["id"]: triple for triple in triples}
    replacements = result.get("R", [])
    deletions = result.get("D", [])
    replacement_ids = [row.get("id") for row in replacements]
    errors: list[str] = []
    if len(replacement_ids) != len(set(replacement_ids)):
        errors.append("duplicate ID in R")
    if len(deletions) != len(set(deletions)):
        errors.append("duplicate ID in D")
    if set(replacement_ids).intersection(deletions):
        errors.append("an ID appears in both R and D")
    if set(replacement_ids).union(deletions) != set(expected):
        errors.append("R and D do not cover every input ID exactly once")

    page_text = "\n".join(sentence["my"] for sentence in payload["S"])
    for row in replacements:
        item_id = row.get("id")
        if item_id not in old:
            continue
        for field in FIELDS:
            if not str(row.get(field, "")).strip():
                errors.append(f"{item_id}: blank {field}")
        for role, endpoint in (("s", "s"), ("o", "o")):
            value = row[f"{role}_my"]
            if value not in page_text:
                original = old[item_id][endpoint]
                if original.get("source") != "inferred" or value != original["my"]:
                    errors.append(
                        f"{item_id}: {role}_my is neither verbatim page text nor preserved inference"
                    )
        if row["p_my"] not in page_text:
            errors.append(f"{item_id}: p_my is not a verbatim span from the page")
    return errors


def review_markdown(payload: dict[str, Any], result: dict[str, Any]) -> str:
    replacements = {row["id"]: row for row in result.get("R", [])}
    deletions = set(result.get("D", []))
    lines = [f"# {payload['page_id']}", ""]
    for sentence in payload["S"]:
        lines.extend(
            [
                f"## {sentence['sid']}",
                "",
                sentence["my"],
                "",
                sentence["en"],
                "",
            ]
        )
        for triple in sentence["T"]:
            item_id = triple["id"]
            lines.extend(
                [
                    f"### {item_id}",
                    "",
                    "Old triple:",
                    "",
                    "```json",
                    json.dumps(display(old_flat(triple)), ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
            if item_id in replacements:
                lines.extend(
                    [
                        "Rewritten triple:",
                        "",
                        "```json",
                        json.dumps(display(replacements[item_id]), ensure_ascii=False, indent=2),
                        "```",
                        "",
                    ]
                )
            elif item_id in deletions:
                lines.extend(["Proposed deletion.", ""])
    return "\n".join(lines)


def main() -> None:
    payload, _ = select_payload()
    triples = all_input_triples(payload)
    ids = [triple["id"] for triple in triples]
    schema = response_schema(ids)
    page_input = render_input(payload)
    prompt_sent = PROMPT + "\n\n" + page_input

    page_id = payload["page_id"]
    volume, page_number = page_id.split("-p")
    output = (
        ROOT
        / "konbaung_v4_delta_repair_trials"
        / volume
        / f"page_{int(page_number):04d}"
        / TRIAL_NAME
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt_sent,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
            max_output_tokens=8192,
        ),
    )
    response_text = response.text or ""
    result = json.loads(response_text)
    usage = response.usage_metadata.model_dump(mode="json") if response.usage_metadata else {}
    errors = validate(payload, result)

    write_json(output / "payload.json", payload)
    (output / "model_input.txt").write_text(page_input, encoding="utf-8")
    (output / "prompt_sent.txt").write_text(prompt_sent, encoding="utf-8")
    write_json(output / "schema.json", schema)
    (output / "response.json").write_text(response_text + "\n", encoding="utf-8")
    write_json(
        output / "status.json",
        {
            "created": datetime.now(timezone.utc).isoformat(),
            "page_id": page_id,
            "model": MODEL,
            "thinking_config": None,
            "input_triples": len(ids),
            "rewritten": len(result.get("R", [])),
            "deleted": len(result.get("D", [])),
            "errors": errors,
            "usage": usage,
        },
    )
    (output / "review.md").write_text(review_markdown(payload, result), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "page_id": page_id,
                "triples": len(ids),
                "R": len(result.get("R", [])),
                "D": len(result.get("D", [])),
                "errors": errors,
                "usage": usage,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

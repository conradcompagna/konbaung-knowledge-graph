#!/usr/bin/env python3
"""Diagnose only defective triples with low thinking and returned thought summaries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from konbaung_gemini_summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT
    / "konbaung_v4_delta_repair_trials"
    / "vol2"
    / "page_0067"
    / "flat_full_rewrite_verbatim_no_thinking_trial_01"
    / "payload.json"
)
OUTPUT = (
    ROOT
    / "konbaung_v4_delta_repair_trials"
    / "vol2"
    / "page_0067"
    / "selective_three_sentence_diagnosis_low_thinking_trial_02"
)
MODEL = "gemini-3.1-flash-lite"

PROMPT = """You will receive one page of the Konbaung Chronicle. Every Burmese sentence is immediately followed by its English translation and its existing triples.

Evaluate every supplied triple, but return only triples that contain at least one real error. Do not return sound triples.

For each defective triple, return exactly three concise English sentences in three fields:

1. `spans`: Are the Burmese subject, predicate, and object correctly grounded in the supplied Burmese text; do they refer to different components rather than reusing the same material; and, when read together as subject -> predicate -> object, do they form a coherent statement? Identify the exact error if not.
2. `glosses`: Do the three English glosses accurately translate or transliterate their corresponding Burmese values rather than paraphrasing the analytical tags? Identify the exact mistranslation if present.
3. `tags`: Do the subject, predicate, and object tags each provide a useful ontological or analytical abstraction above the grounded wording? Identify any tag that merely repeats a translation, misclassifies its component, or describes a relation not formed by the selected spans.

Write exactly one complete sentence in each field. Together, the three fields are the required three-sentence evaluation of that triple.

If the answer is no in any of those three sentences, return a complete corrected version of that triple under the same ID. Supply the corrected subject as `s`, predicate as `p`, and object as `o`; each contains its Burmese `my`, English `en`, and analytical `tag`. Rewrite the whole triple even if only one component changed. The correction must resolve every error identified in the three diagnostic sentences.

Grounding rules:

- A component marked `sentence` must be copied verbatim from its assigned Burmese sentence.
- A component marked `page` must be copied verbatim from another Burmese sentence on this page.
- A subject or object marked `inferred` may be absent from the text when Burmese grammar or immediate context requires it.
- An inferred subject or object is not an error merely because it is absent.
- A predicate should normally be an actual verbal or relational expression in the supplied Burmese text.
- Subject, predicate, and object must represent distinct semantic roles; they must not duplicate the same text or force a unary description into a meaningless binary statement.

Be literal and specific. Do not praise the triple, discuss general history, propose additional triples, or invent defects merely to produce output. If every triple is sound, return an empty `E` array. Never return a diagnostic record without its complete corrected triple.

Return only schema-valid JSON."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def triples(payload: dict[str, Any]) -> list[dict[str, Any]]:
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
                    f's_my: {triple["s"]["my"]}',
                    f'p_my: {triple["predicate_grounding"]["my"]}',
                    f'o_my: {triple["o"]["my"]}',
                    f's_en: {triple["s"]["en"]}',
                    f'p_en: {triple["predicate_grounding"]["en"]}',
                    f'o_en: {triple["o"]["en"]}',
                    f's_tag: {triple["s"]["tag"]}',
                    f'p_tag: {triple["p"]}',
                    f'o_tag: {triple["o"]["tag"]}',
                    f's_source: {triple["s"]["source"]}',
                    f'p_source: {triple["predicate_grounding"]["source"]}',
                    f'o_source: {triple["o"]["source"]}',
                    "</TRIPLE>",
                    "",
                ]
            )
        lines.extend(["</SENTENCE>", ""])
    lines.append("</PAGE>")
    return "\n".join(lines)


def response_schema() -> dict[str, Any]:
    component = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "my": {"type": "string"},
            "en": {"type": "string"},
            "tag": {"type": "string"},
        },
        "required": ["my", "en", "tag"],
    }
    evaluation = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "spans": {"type": "string"},
            "glosses": {"type": "string"},
            "tags": {"type": "string"},
            "s": component,
            "p": component,
            "o": component,
        },
        "required": ["id", "spans", "glosses", "tags", "s", "p", "o"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"E": {"type": "array", "items": evaluation}},
        "required": ["E"],
    }


def display(triple: dict[str, Any]) -> dict[str, Any]:
    return {
        "s": {
            "my": triple["s"]["my"],
            "en": triple["s"]["en"],
            "tag": triple["s"]["tag"],
        },
        "p": {
            "my": triple["predicate_grounding"]["my"],
            "en": triple["predicate_grounding"]["en"],
            "tag": triple["p"],
        },
        "o": {
            "my": triple["o"]["my"],
            "en": triple["o"]["en"],
            "tag": triple["o"]["tag"],
        },
    }


def validate(payload: dict[str, Any], result: dict[str, Any]) -> list[str]:
    all_triples = triples(payload)
    known = {triple["id"] for triple in all_triples}
    old = {triple["id"]: triple for triple in all_triples}
    page_text = "\n".join(sentence["my"] for sentence in payload["S"])
    rows = result.get("E", [])
    ids = [row.get("id") for row in rows]
    errors: list[str] = []
    if len(ids) != len(set(ids)):
        errors.append("duplicate diagnostic ID")
    unknown = set(ids).difference(known)
    if unknown:
        errors.append(f"unknown IDs: {sorted(unknown)}")
    for row in rows:
        for field in ("spans", "glosses", "tags"):
            value = str(row.get(field, "")).strip()
            if not value:
                errors.append(f"{row.get('id')}: blank {field}")
            elif value[-1] not in ".!?":
                errors.append(f"{row.get('id')}: {field} is not a complete sentence")
        item_id = row.get("id")
        for role in ("s", "p", "o"):
            for field in ("my", "en", "tag"):
                if not str(row.get(role, {}).get(field, "")).strip():
                    errors.append(f"{item_id}: blank {role}.{field}")
        if item_id not in old:
            continue
        for role, endpoint in (("s", "s"), ("o", "o")):
            value = row[role]["my"]
            if value not in page_text:
                original = old[item_id][endpoint]
                if original.get("source") != "inferred" or value != original["my"]:
                    errors.append(
                        f"{item_id}: {role}.my is neither verbatim page text nor preserved inference"
                    )
        if row["p"]["my"] not in page_text:
            errors.append(f"{item_id}: p.my is not a verbatim span from the page")
    return errors


def returned_thoughts(response: Any) -> str:
    texts: list[str] = []
    for candidate in response.candidates or []:
        content = candidate.content
        for part in (content.parts if content else []) or []:
            if getattr(part, "thought", False) and getattr(part, "text", None):
                texts.append(part.text)
    return "\n\n".join(texts)


def review_markdown(payload: dict[str, Any], result: dict[str, Any]) -> str:
    evaluations = {row["id"]: row for row in result.get("E", [])}
    lines = [f'# {payload["page_id"]} — flagged triples', ""]
    for sentence in payload["S"]:
        flagged = [triple for triple in sentence["T"] if triple["id"] in evaluations]
        if not flagged:
            continue
        lines.extend(
            [
                f'## {sentence["sid"]}',
                "",
                sentence["my"],
                "",
                sentence["en"],
                "",
            ]
        )
        for triple in flagged:
            row = evaluations[triple["id"]]
            lines.extend(
                [
                    f'### {triple["id"]}',
                    "",
                    "```json",
                    json.dumps(display(triple), ensure_ascii=False, indent=2),
                    "```",
                    "",
                    f'**Spans:** {row["spans"]}',
                    "",
                    f'**Glosses:** {row["glosses"]}',
                    "",
                    f'**Tags:** {row["tags"]}',
                    "",
                    "Proposed replacement:",
                    "",
                    "```json",
                    json.dumps(
                        {role: row[role] for role in ("s", "p", "o")},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "```",
                    "",
                ]
            )
    if not evaluations:
        lines.extend(["Gemini flagged no triples.", ""])
    return "\n".join(lines)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    payload = read_json(SOURCE)
    page_input = render_input(payload)
    prompt_sent = PROMPT + "\n\n" + page_input
    schema = response_schema()

    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt_sent,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(
                thinking_level="low",
                include_thoughts=True,
            ),
        ),
    )
    response_text = response.text or ""
    result = json.loads(response_text)
    thoughts = returned_thoughts(response)
    usage = response.usage_metadata.model_dump(mode="json") if response.usage_metadata else {}
    errors = validate(payload, result)

    write_json(OUTPUT / "payload.json", payload)
    (OUTPUT / "model_input.txt").write_text(page_input, encoding="utf-8")
    (OUTPUT / "prompt_sent.txt").write_text(prompt_sent, encoding="utf-8")
    write_json(OUTPUT / "schema.json", schema)
    (OUTPUT / "response.json").write_text(response_text + "\n", encoding="utf-8")
    (OUTPUT / "thought_summary.txt").write_text(thoughts, encoding="utf-8")
    write_json(
        OUTPUT / "status.json",
        {
            "created": datetime.now(timezone.utc).isoformat(),
            "page_id": payload["page_id"],
            "model": MODEL,
            "thinking_level": "low",
            "include_thoughts": True,
            "input_triples": len(triples(payload)),
            "flagged_triples": len(result.get("E", [])),
            "errors": errors,
            "usage": usage,
        },
    )
    (OUTPUT / "review.md").write_text(review_markdown(payload, result), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "page_id": payload["page_id"],
                "triples": len(triples(payload)),
                "flagged": len(result.get("E", [])),
                "thought_summary_chars": len(thoughts),
                "errors": errors,
                "usage": usage,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

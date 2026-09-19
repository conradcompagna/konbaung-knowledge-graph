#!/usr/bin/env python3
"""Test a lean, sentence-only repair pass for tag-like predicate glosses."""

from __future__ import annotations

import json
import re
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel

from konbaung_gemini_summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "konbaung_relation_predicate_grounding_full_batch_20260717" / "pages"
OUTPUT = ROOT / "konbaung_predicate_gloss_repair_trials" / "five_sentence_trial_01"
MODEL = "gemini-3.1-flash-lite"
SELECTION = {
    "vol1_s000011": (1, 49),
    "vol1_s000012": (1, 49),
    "vol1_s000515": (1, 81),
    "vol2_s000089": (2, 28),
    "vol3_s000032": (3, 27),
}

INSTRUCTION = """You are correcting faulty English glosses of exact Burmese predicate spans from Konbaung chronicle sentences.

For every supplied item, translate only the Burmese predicate span (`my`) as it functions in its sentence. Return a short, natural English verb phrase or relational phrase.

The gloss must be a faithful lexical translation of the Burmese span itself. Do not restate or infer the broader historical relation, subject, object, motive, consequence, or analytical interpretation.

Never output an ALL_CAPS relation label. Never use underscores. Never copy or reconstruct a database tag. Use ordinary English words in sentence case.

Use the full Burmese sentence and its English translation only as context for disambiguating the predicate span.

Return every supplied ID exactly once and in the supplied order. Return only JSON matching the schema."""


class Gloss(BaseModel):
    id: str
    en: str


class Result(BaseModel):
    R: list[Gloss]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tag_like(value: str) -> bool:
    return "_" in value or bool(re.fullmatch(r"[A-Z0-9 ]+", value.strip()))


def build_payload() -> tuple[dict, list[str], dict[str, str]]:
    records = []
    expected = []
    old_glosses: dict[str, str] = {}
    for sid, (volume, page) in SELECTION.items():
        page_dir = SOURCE / f"vol{volume}" / f"page_{page:04d}"
        payload = read_json(page_dir / "payload.json")
        result = read_json(page_dir / "result.json")["response"]["R"]
        sentence = next(record for record in payload["records"] if record["sid"] == sid)
        items = []
        for grounding in result:
            if not grounding["id"].startswith(sid + "_t") or not tag_like(grounding["en"]):
                continue
            items.append({"id": grounding["id"], "my": grounding["my"]})
            expected.append(grounding["id"])
            old_glosses[grounding["id"]] = grounding["en"]
        if not items:
            raise ValueError(f"No tag-like glosses selected for {sid}")
        records.append({"sid": sid, "my": sentence["my"], "en": sentence["en"], "items": items})
    return {"sentences": records}, expected, old_glosses


def validate(result: Result, expected: list[str]) -> list[str]:
    errors = []
    ids = [item.id for item in result.R]
    if ids != expected:
        errors.append(f"ID/order mismatch: expected {expected}, got {ids}")
    for item in result.R:
        if not item.en.strip():
            errors.append(f"{item.id}: empty gloss")
        if "_" in item.en:
            errors.append(f"{item.id}: gloss contains underscore")
        if re.fullmatch(r"[A-Z0-9 ]+", item.en.strip()):
            errors.append(f"{item.id}: gloss is an all-caps label")
    return errors


def render_review(payload: dict, old: dict[str, str], result: Result, errors: list[str]) -> str:
    returned = {item.id: item.en for item in result.R}
    lines = ["# Five-sentence predicate-gloss repair trial", "", "PASS" if not errors else "FLAGGED", ""]
    if errors:
        lines.extend(["## Validation errors", "", *[f"- {error}" for error in errors], ""])
    for sentence in payload["sentences"]:
        lines.extend([
            f"## {sentence['sid']}", "", "### Burmese", "", sentence["my"], "",
            "### English sentence", "", sentence["en"], "", "### Gloss repairs", "",
        ])
        for item in sentence["items"]:
            lines.extend([
                f"- `{item['id']}`", f"  - Burmese predicate: `{item['my']}`",
                f"  - Old gloss: `{old[item['id']]}`", f"  - Proposed gloss: `{returned.get(item['id'], '[MISSING]')}`",
            ])
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    payload, expected, old = build_payload()
    prompt = INSTRUCTION + "\n\n<INPUT>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n</INPUT>"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "payload.json", payload)
    (OUTPUT / "prompt_sent.txt").write_text(prompt, encoding="utf-8")
    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Result,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(thinking_budget=0, include_thoughts=False),
        ),
    )
    raw = response.text or ""
    parsed = Result.model_validate_json(raw)
    errors = validate(parsed, expected)
    (OUTPUT / "raw_response.json").write_text(raw, encoding="utf-8")
    write_json(
        OUTPUT / "result.json",
        {
            "accepted": not errors,
            "errors": errors,
            "model": MODEL,
            "thinking_budget": 0,
            "response": parsed.model_dump(mode="json"),
            "usage": response.usage_metadata.model_dump(mode="json") if response.usage_metadata else {},
        },
    )
    (OUTPUT / "review.md").write_text(render_review(payload, old, parsed, errors), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "sentences": len(payload["sentences"]), "items": len(expected), "errors": errors}, indent=2))


if __name__ == "__main__":
    main()

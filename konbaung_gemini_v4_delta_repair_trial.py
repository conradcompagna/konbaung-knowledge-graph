#!/usr/bin/env python3
"""Run one compact page-level v4 triple-repair trial."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from konbaung_gemini_summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parent
V3 = ROOT / "konbaung_dataset_v3_20260718"
OUTPUT_ROOT = ROOT / "konbaung_v4_delta_repair_trials"
MODEL = "gemini-3.1-flash-lite"
SEED = 20260718
USED_PAGES = {
    "vol1-p0082", "vol1-p0152", "vol1-p0282", "vol1-p0304", "vol1-p0340",
    "vol2-p0072", "vol2-p0201", "vol2-p0236", "vol2-p0258",
    "vol3-p0133", "vol3-p0221", "vol3-p0241", "vol3-p0271", "vol3-p0333",
}

PROMPT = """You will receive one page of the Konbaung Chronicle containing every Burmese sentence, its English translation, and all existing triples attached to that sentence.

Inspect every triple. Treat the existing triple as generally reliable and change it only when one of the following defects is genuinely present.

### 1. Subject, predicate grounding, and object

Read the triple literally as:

subject → relation → object

Check that the subject, predicate grounding, and object represent the correct components of the specific relation recorded by that triple.

When these components are explicit, their Burmese groundings must be exact spans from the supplied text. Do not reuse the same Burmese span as two different components. Prefer minimal spans and avoid including a subject or object inside the predicate grounding when a smaller verbal expression conveys the relation.

A legitimately omitted subject or object may remain inferred. Do not invent a textual span or delete an otherwise valid relation merely because Burmese grammar omits an actor or referent.

Repair a triple only when its components are wrongly attached, improperly repeated, semantically incoherent, incorrectly directed, or assigned to the wrong subject or object.

Do not collapse an entire multi-action sentence into one general relation. Each triple records one specific subject–relation–object proposition.

### 2. Source provenance

Use:

- `sentence` when the exact Burmese span appears in the sentence assigned to the triple;
- `page` when it appears elsewhere in the supplied page and supplies a clear antecedent, heading, or inherited predicate;
- `inferred` when the referent is absent from the supplied Burmese but is grammatically or contextually required.

Check inferred predicate groundings especially carefully. Recover an explicit predicate from the sentence or page when one exists. Never invent Burmese wording.

### 3. Glosses

Check that every English gloss directly translates or transliterates its corresponding Burmese value.

Pay particular attention to `predicate_grounding.en`. It must translate `predicate_grounding.my` itself, not repeat or paraphrase the analytical relation `p`.

Do not add identities, motives, historical explanations, or analytical interpretations that are absent from the corresponding Burmese value.

Repair only glosses that are inaccurate or contain information not expressed by their Burmese grounding.

### 4. Entity and relation tags

Check that `s.tag`, `o.tag`, and `p` are analytical open codes.

Entity tags classify what kind of entity the subject or object represents. The relation `p` classifies the historical relationship from the selected subject to the selected object.

Tags should provide one useful analytical step above literal translation. They must not be sentence-length translations mechanically converted into PascalCase or UPPER_SNAKE_CASE.

However, do not change a valid reusable tag merely because it resembles the translation. Ordinary relations such as `ATTACKS`, `COMMANDS`, `APPOINTS_TO_OFFICE`, or `MARRIES` may naturally resemble the underlying verb while still functioning as analytical relations.

Do not impose a closed ontology, normalize synonyms across the corpus, or replace a reasonable existing open code merely because another label might also work.

### 5. Conservative repair

Do not add new facts or IDs.

Do not split one input triple into multiple triples or merge separate triples.

If a triple is sound, leave it unchanged and do not return it.

If a triple is defective but repairable from the supplied text, return one complete replacement triple under its unchanged ID.

Place an ID in `D` only when the supplied text cannot support any coherent binary subject–relation–object repair. Do not delete a triple merely because an endpoint is legitimately inferred.

### Historiographical grounding

This database supports a systematic historical analysis of power relations in the Konbaung Chronicle. Its central question is: who exercised power over whom, through what actions, commands, appointments, punishments, grants, obligations, alliances, and other relationships?

The triples recover who did what to whom and who ordered, authorized, rewarded, constrained, attacked, appointed, or otherwise acted upon whom. Burmese groundings preserve the textual evidence, English glosses translate that evidence, and analytical tags identify the historical type of entity or relationship.

Remain grounded in concrete propositions expressed by the chronicle. Do not introduce motives, strategies, legitimacy claims, or broader historical significance unless the supplied text supports them.

### Output

Return only:

{
  "R": [complete replacement triple objects that genuinely changed],
  "D": ["IDs that cannot form a coherent binary triple"]
}

Preserve every input ID exactly.

Do not return unchanged triples, explanations, comments, markdown, or additional fields."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def endpoint_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "my": {"type": "string"},
            "en": {"type": "string"},
            "tag": {"type": "string"},
            "source": {"type": "string", "enum": ["sentence", "page", "inferred"]},
        },
        "required": ["my", "en", "tag", "source"],
    }


def response_schema(ids: list[str]) -> dict[str, Any]:
    grounding = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "my": {"type": "string"},
            "en": {"type": "string"},
            "source": {"type": "string", "enum": ["sentence", "page", "inferred"]},
        },
        "required": ["my", "en", "source"],
    }
    replacement = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string", "enum": ids},
            "s": endpoint_schema(),
            "p": {"type": "string"},
            "o": endpoint_schema(),
            "predicate_grounding": grounding,
        },
        "required": ["id", "s", "p", "o", "predicate_grounding"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "R": {"type": "array", "maxItems": len(ids), "items": replacement},
            "D": {"type": "array", "maxItems": len(ids), "items": {"type": "string", "enum": ids}},
        },
        "required": ["R", "D"],
    }


def choose_page(pages: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = []
    for page in pages:
        triple_count = sum(len(group.get("T", [])) for group in page.get("S", []))
        page_id = page["page_id"]
        record_path = V3 / "sentence_corpus" / "page_records" / f"vol{page['volume']}" / f"page_{int(page['page']):04d}.json"
        if page_id in USED_PAGES or not record_path.exists() or not 12 <= triple_count <= 24:
            continue
        page_record = read_json(record_path)
        if page_record.get("cross_page_sentence_ids"):
            continue
        eligible.append(page)
    if not eligible:
        raise ValueError("No eligible ordinary page found")
    return random.Random(SEED).choice(sorted(eligible, key=lambda item: item["page_id"]))


def build_payload() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    pages = read_jsonl(V3 / "triples" / "annotations" / "all_pages.jsonl")
    page = choose_page(pages)
    volume = int(page["volume"])
    page_number = int(page["page"])
    page_record = read_json(
        V3 / "sentence_corpus" / "page_records" / f"vol{volume}" / f"page_{page_number:04d}.json"
    )
    sentences = {
        item["id"]: item
        for item in read_jsonl(V3 / "sentence_corpus" / "sentences" / f"vol{volume}.jsonl")
    }
    translations = {
        item["id"]: item["en"]
        for item in read_jsonl(V3 / "sentence_translations" / "translations" / f"vol{volume}.jsonl")
    }
    groups = {group["sid"]: group.get("T", []) for group in page.get("S", [])}
    old_by_id: dict[str, dict[str, Any]] = {}
    records = []
    for sid in page_record["sentence_ids"]:
        triples = []
        for index, triple in enumerate(groups.get(sid, []), start=1):
            item = {"id": f"{sid}_t{index:03d}", **triple}
            triples.append(item)
            old_by_id[item["id"]] = triple
        records.append({"sid": sid, "my": sentences[sid]["text"], "en": translations[sid], "T": triples})
    payload = {"page_id": page["page_id"], "S": records}
    return payload, old_by_id


def validate(result: dict[str, Any], old_by_id: dict[str, dict[str, Any]]) -> list[str]:
    errors = []
    replacements = result.get("R", [])
    deletions = result.get("D", [])
    replacement_ids = [item.get("id") for item in replacements]
    if len(replacement_ids) != len(set(replacement_ids)):
        errors.append("duplicate replacement ID")
    if len(deletions) != len(set(deletions)):
        errors.append("duplicate deletion ID")
    overlap = set(replacement_ids).intersection(deletions)
    if overlap:
        errors.append(f"IDs both replaced and deleted: {sorted(overlap)}")
    unknown = (set(replacement_ids) | set(deletions)).difference(old_by_id)
    if unknown:
        errors.append(f"unknown IDs: {sorted(unknown)}")
    for item in replacements:
        item_id = item.get("id")
        proposed = {key: item[key] for key in ("s", "p", "o", "predicate_grounding") if key in item}
        if item_id in old_by_id and proposed == old_by_id[item_id]:
            errors.append(f"{item_id}: no-op replacement")
    return errors


def review_markdown(
    payload: dict[str, Any],
    result: dict[str, Any],
) -> str:
    def display_triple(triple: dict[str, Any]) -> dict[str, Any]:
        return {
            "s": {key: triple["s"][key] for key in ("my", "en", "tag")},
            "p": {
                "my": triple["predicate_grounding"]["my"],
                "en": triple["predicate_grounding"]["en"],
                "tag": triple["p"],
            },
            "o": {key: triple["o"][key] for key in ("my", "en", "tag")},
        }

    replacements = {item["id"]: item for item in result.get("R", [])}
    deletions = set(result.get("D", []))
    lines = [f"# {payload['page_id']}", ""]
    for sentence in payload["S"]:
        lines.extend([f"## {sentence['sid']}", "", sentence["my"], "", sentence["en"], ""])
        if not sentence["T"]:
            lines.extend(["No triples.", ""])
            continue
        for old in sentence["T"]:
            item_id = old["id"]
            old_triple = display_triple(old)
            lines.extend(
                [
                    f"### {item_id}",
                    "",
                    "Old triple:",
                    "",
                    "```json",
                    json.dumps(old_triple, ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
            if item_id in replacements:
                new = replacements[item_id]
                new_triple = display_triple(new)
                lines.extend(
                    [
                        "Proposed replacement:",
                        "",
                        "```json",
                        json.dumps(new_triple, ensure_ascii=False, indent=2),
                        "```",
                        "",
                    ]
                )
            elif item_id in deletions:
                lines.extend(["Proposed deletion.", ""])
    return "\n".join(lines)


def main() -> None:
    payload, old_by_id = build_payload()
    page_id = payload["page_id"]
    volume, page_number = page_id.split("-p")
    output = OUTPUT_ROOT / volume / f"page_{int(page_number):04d}" / "concise_delta_prompt_trial_01"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    ids = list(old_by_id)
    schema = response_schema(ids)
    prompt_sent = PROMPT + "\n\nPAGE_DATA\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\nEND_PAGE_DATA"
    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt_sent,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
            max_output_tokens=max(4096, min(65536, len(ids) * 400)),
            thinking_config=types.ThinkingConfig(thinking_level="low", include_thoughts=False),
        ),
    )
    raw_text = response.text or ""
    result = json.loads(raw_text)
    errors = validate(result, old_by_id)
    usage = response.usage_metadata.model_dump(mode="json") if response.usage_metadata else {}
    write_json(output / "payload.json", payload)
    write_json(output / "schema.json", schema)
    (output / "prompt_sent.txt").write_text(prompt_sent, encoding="utf-8")
    write_json(output / "raw_response.json", {"text": raw_text, "usage": usage})
    write_json(output / "result.json", result)
    write_json(
        output / "status.json",
        {
            "created": datetime.now(timezone.utc).isoformat(),
            "page_id": page_id,
            "model": MODEL,
            "thinking_level": "low",
            "input_triples": len(ids),
            "replacements": len(result.get("R", [])),
            "deletions": len(result.get("D", [])),
            "errors": errors,
            "usage": usage,
        },
    )
    (output / "review.md").write_text(review_markdown(payload, result), encoding="utf-8")
    print(json.dumps({"output": str(output), "page_id": page_id, "triples": len(ids), "R": len(result.get("R", [])), "D": len(result.get("D", [])), "errors": errors, "usage": usage}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

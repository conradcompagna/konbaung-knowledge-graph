#!/usr/bin/env python3
"""Run full-record metadata recall trials over existing chronicle triples."""

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
from konbaung_gemini_triple_metadata_semantic_frame import existing_triples, load_page, normalized


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "konbaung_triple_metadata_recall_trials"


class Addition(BaseModel):
    my: str = Field(description="Exact Burmese span from TARGET_PAGE_TEXT.")
    en: str = Field(description="Short English gloss.")


class TripleAdditions(BaseModel):
    id: str
    time: Addition | None = None
    place: Addition | None = None
    quantity: Addition | None = None
    manner: Addition | None = None
    reason: Addition | None = None


class RecallResult(BaseModel):
    M: list[TripleAdditions]


METADATA_ENRICHMENT_PROMPT = """<METADATA_ENRICHMENT_TASK>
These Burmese royal chronicles describe how power operated through kingship, court hierarchy, administration, warfare, diplomacy, tributary relations, religious patronage, ritual performance, resource mobilization, and claims to legitimacy. Existing open-coded triples have already captured the principal actors, actions, institutions, and objects relevant to that historical analysis. This is not another triple-extraction pass. Its purpose is to enrich those existing graph relations with concise contextual metadata that enables chronological, spatial, quantitative, procedural, and causal analysis.

For each existing triple, inspect its evidence and the complete page for genuinely additional metadata in five categories:

1. `time`: when the relation occurred. This includes explicit dates, relative dates, times of day, durations, deadlines, and meaningful sequence expressions.
2. `place`: where it occurred. This includes event sites, origins, destinations, routes, jurisdictions, palace spaces, ritual sites, camps, offices, and source locations.
3. `quantity`: its measurable scale. This includes counts, dimensions, distances, durations, money, rates, personnel, animals, objects, resources, offerings, and losses.
4. `manner`: how it occurred. This includes method, means, instrument, vehicle, clothing or equipment used in performance, procedure, formation, arrangement, and characteristic mode of action.
5. `reason`: why it occurred. This includes explicit cause, purpose, intention, motivation, trigger, justification, or intended goal.

Return only the triple ID and applicable metadata fields. Each metadata value contains an exact Burmese span in `my` and a short English gloss in `en`. Omit a field when no useful value exists, and omit the entire triple when it has no additions.

Metadata must add information. Do not repeat anything already expressed by the triple's subject, predicate, object, or populated time/place/quantity fields. This prohibition applies even when a candidate uses a shorter span, a longer overlapping span, a grammatical variant, or an English paraphrase. For example, if the object is already “the stupa,” do not return the stupa as place. If the predicate already means “travels by royal boat,” do not return “by royal boat” as manner. If the subject already contains “sixteen Buddhas,” do not return sixteen as quantity unless the number independently measures a different aspect of the relation.

Every Burmese metadata span must occur verbatim in the page text and must directly qualify the specific triple. Adjacent clauses may supply shared context, but do not import details from a separate event.

Example 1: an existing triple says that a king dispatched a military force, while its evidence additionally states “at dawn,” “from the western camp,” “five hundred soldiers,” “by royal boats,” and “because scouts reported an enemy approach.” These may respectively populate time, place, quantity, manner, and reason.

Example 2: an existing triple says that attendants escorted a relic palanquin. If the evidence says they wore ceremonial clothing and carried fans, that phrase is new manner metadata. The palanquin itself is not new metadata because it is already the object.

Example 3: an existing triple already has a date and location but its evidence gives a payment amount and says the judgment was conducted according to customary law. Return quantity and manner only; do not return another version of the existing date or location.

Prefer a small number of genuinely informative additions over comprehensive restatement. Before returning each field, ask: could a researcher learn this information from the existing subject, predicate, object, and metadata alone? If yes, omit it.
</METADATA_ENRICHMENT_TASK>"""


def masked_page_text(page: dict[str, Any]) -> str:
    return page["canonicalText"]


def build_prompt(volume: int, page_number: int) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    page = load_page(volume, page_number)
    triples = existing_triples(page)
    masked_text = masked_page_text(page)
    prompt_triples = [
        {
            key: value
            for key, value in triple.items()
            if key not in {"time", "place", "quantity"} or value is not None
        }
        for triple in triples
    ]
    return (
        f"""<TARGET_PAGE_TEXT>
{masked_text}
</TARGET_PAGE_TEXT>

<EXISTING_TRIPLES_READ_ONLY>
{json.dumps(prompt_triples, ensure_ascii=False, indent=2)}
</EXISTING_TRIPLES_READ_ONLY>

{METADATA_ENRICHMENT_PROMPT}
""",
        page,
        triples,
    )


def validate(
    result: RecallResult, page: dict[str, Any], triples: list[dict[str, Any]]
) -> list[str]:
    warnings: list[str] = []
    source = {row["id"]: row for row in triples}
    target = normalized(page["canonicalText"])
    for index, row in enumerate(result.M, 1):
        original = source.get(row.id)
        if original is None:
            warnings.append(f"M[{index}]: unknown id {row.id}")
            continue
        categories = {
            "time": row.time,
            "place": row.place,
            "quantity": row.quantity,
            "manner": row.manner,
            "reason": row.reason,
        }
        items = [item for item in categories.values() if item is not None]
        if not items:
            warnings.append(f"M[{index}]: empty record should have been omitted")
        for category in ("time", "place", "quantity"):
            if categories[category] is not None and original.get(category):
                warnings.append(
                    f"M[{index}]: category {category} already had data and will be stripped"
                )
        for item in items:
            if normalized(item.my) not in target:
                warnings.append(f"M[{index}]: addition span absent from page: {item.my}")
    return warnings


def postprocess(result: RecallResult, triples: list[dict[str, Any]]) -> dict[str, Any]:
    existing = {row["id"]: row for row in triples}
    payload = result.model_dump(exclude_none=True)
    for row in payload["M"]:
        original = existing.get(row["id"])
        if not original:
            continue
        for category in ("time", "place", "quantity"):
            if original.get(category):
                row.pop(category, None)
    payload["M"] = [row for row in payload["M"] if len(row) > 1]
    return payload


def write_review(
    path: Path,
    page: dict[str, Any],
    triples: list[dict[str, Any]],
    result: dict[str, Any] | str,
) -> None:
    additions = {}
    if isinstance(result, dict):
        additions = {row["id"]: row for row in result.get("M", [])}
    lines = ["# Summary", "", page.get("summary") or "", "", "# Triples", ""]
    for triple in triples:
        lines.append(
            f"**{triple['id']}** {triple['s']} ({triple['sg']}) "
            f"--{triple['p']}--> {triple['o']} ({triple['og']})"
        )
        added = additions.get(triple["id"], {})
        for category in ("time", "place", "quantity", "manner", "reason"):
            value = added.get(category)
            if value:
                lines.append(f"+ **{category}:** {value['my']} | {value['en']}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run(volume: int, page_number: int, output_name: str) -> Path:
    prompt, page, triples = build_prompt(volume, page_number)
    output_dir = OUTPUT_ROOT / f"vol{volume}" / f"page_{page_number:04d}" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_sent.txt").write_text(prompt, encoding="utf-8")
    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RecallResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=3000,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw_text = response.text or ""
    (output_dir / "raw_response.txt").write_text(raw_text, encoding="utf-8")
    flags: list[str] = []
    parsed_result: dict[str, Any] | str
    try:
        result = RecallResult.model_validate_json(raw_text)
        flags.extend(validate(result, page, triples))
        parsed_result = postprocess(result, triples)
    except Exception as exc:
        flags.append(f"structured parse failed: {exc}")
        parsed_result = raw_text
    payload = {
        "accepted": True,
        "flagged": bool(flags),
        "flags": flags,
        "usage": response.usage_metadata.model_dump(mode="json") if response.usage_metadata else {},
        "result": parsed_result,
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_review(output_dir / "review.md", page, triples, parsed_result)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int, required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    print(run(args.volume, args.page, args.output_name))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a one-page Gemini trial for incremental entity/relation categorization."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel

from pipeline.extraction.summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PAGES = (
    ROOT / "konbaung_historiography_ungrounded_v3_full_batch_20260723" / "final" / "all_pages.jsonl"
)
OUTPUT_ROOT = ROOT / "konbaung_incremental_category_trials"
HISTORIOGRAPHY_PROMPT_PATH = ROOT / "prompts/translated_sentence_annotation_prompt_short.md"
MODEL = "gemini-3.1-flash-lite"


class Category(BaseModel):
    id: str
    label: str


class EntityMapping(BaseModel):
    entity: str
    category_id: str


class RelationMapping(BaseModel):
    relation: str
    category_id: str


class CategorizationResult(BaseModel):
    entity_categories: list[Category]
    relation_categories: list[Category]
    entity_mappings: list[EntityMapping]
    relation_mappings: list[RelationMapping]


INSTRUCTION = """The existing triples are the open codes produced in the preceding research phase. Perform the next phase: axial coding.

Use the page summary and full Burmese/English sentences only as context for interpreting the triples.

Create reusable analytical categories that relate the open codes to the historiographical research question and conceptual frame in the read-only prefix. The categories should identify broader structures and mechanisms of Konbaung power without becoming vague umbrella types.

For entities, categorize each node by its analytical role or position in the power field—not merely by its surface ontology. For relations, categorize each predicate by the broader mechanism, direction, or function of power that it expresses—not merely by its grammatical action type. The axial category must remain one analytical level above the supplied open code and be reusable across later pages.

Do not use generic catch-all labels such as `Person`, `Group`, `Place`, `Concept`, `Institution`, `Administrative Action`, or `Military Action` where the supplied context supports a more meaningful analytical category.

This is categorization, not entity-identity resolution: do not merge, rename, correct, or rewrite the supplied entity or relation strings. Similar-looking strings are still separate required input items and must each receive their own mapping row.

Return:
1. `entity_categories`: the proposed entity category IDs and labels.
2. `relation_categories`: the proposed relation category IDs and labels.
3. `entity_mappings`: every distinct subject and object in the supplied triples, each exactly once, preserving its string exactly and mapped to exactly one proposed entity category.
4. `relation_mappings`: every distinct predicate in the supplied triples, each exactly once, preserving its string exactly and mapped to exactly one proposed relation category.

Use compact stable IDs such as `E01` and `R01`. Every category referenced by a mapping must appear in the corresponding category list. Make categories broad enough to reuse on later pages but specific enough to remain historically meaningful. Do not add definitions, rationales, confidence scores, summaries, commentary, or any other fields. Return only the requested JSON through the supplied schema."""

SEQUENTIAL_INSTRUCTION = """A previous axial codebook is supplied below. Treat it as the current binding vocabulary:

- Reuse an existing category ID and label whenever it fits the current open code.
- Preserve every previous category ID and label exactly in the returned category lists, even when no current-page item uses it.
- Do not rename, redefine, merge, split, or renumber a previous category.
- Create a new category only when no previous category adequately captures the current open code's analytical role or mechanism.
- Give new categories an `E` or `R` identifier whose number is exactly one greater than the highest identifier already present in that category set.
- Return the complete updated category lists, followed by mappings for every distinct entity and relation on the current page only."""


def read_only_historiography_prefix() -> str:
    original_prompt = HISTORIOGRAPHY_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "<READ_ONLY_HISTORIOGRAPHY_RESEARCH_FRAME>\n"
        "The text inside this block is the original open-coding prompt. Read it only "
        "for its research question, conceptual framework, and analytical priorities. "
        "Do not execute its extraction task, grounding rules, coverage targets, output "
        "instructions, or schema instructions. The CURRENT AXIAL-CODING TASK after "
        "this block is the only task to perform.\n\n"
        + original_prompt.rstrip()
        + "\n</READ_ONLY_HISTORIOGRAPHY_RESEARCH_FRAME>\n"
    )


def load_pages() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with SOURCE_PAGES.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def triples_for(record: dict[str, Any]) -> list[dict[str, str]]:
    triples: list[dict[str, str]] = []
    for sentence in record["response"]["sentences"]:
        for triple in sentence["triples"]:
            triples.append(
                {
                    "subject": str(triple["subject"]),
                    "predicate": str(triple["predicate"]),
                    "object": str(triple["object"]),
                }
            )
    return triples


def eligible(record: dict[str, Any]) -> bool:
    sentences = record.get("sentences") or []
    return bool(
        str(record.get("summary", "")).strip()
        and sentences
        and all(
            str(sentence.get("sid", "")).strip()
            and str(sentence.get("my", "")).strip()
            and str(sentence.get("en", "")).strip()
            for sentence in sentences
        )
        and triples_for(record)
    )


def choose_page(
    volume: int | None,
    page_number: int | None,
) -> dict[str, Any]:
    pages = [record for record in load_pages() if eligible(record)]
    if volume is not None or page_number is not None:
        if volume is None or page_number is None:
            raise ValueError("--volume and --page must be supplied together")
        matches = [
            record
            for record in pages
            if int(record["volume"]) == volume and int(record["page"]) == page_number
        ]
        if not matches:
            raise ValueError(f"No valid page found for volume {volume}, page {page_number}")
        return matches[0]
    return secrets.choice(pages)


def unique_inputs(
    triples: list[dict[str, str]],
) -> tuple[list[str], list[str]]:
    entities = list(
        dict.fromkeys(
            value for triple in triples for value in (triple["subject"], triple["object"])
        )
    )
    relations = list(dict.fromkeys(triple["predicate"] for triple in triples))
    return entities, relations


def supplied_page(record: dict[str, Any]) -> dict[str, Any]:
    decisions = {sentence["sid"]: sentence for sentence in record["response"]["sentences"]}
    sentences = []
    for sentence in record["sentences"]:
        decision = decisions[sentence["sid"]]
        sentences.append(
            {
                "sid": sentence["sid"],
                "my": sentence["my"],
                "en": sentence["en"],
                "triples": decision["triples"],
            }
        )
    return {
        "page_id": record["key"],
        "page_summary": record["summary"],
        "sentences": sentences,
    }


def validate(
    entities: list[str],
    relations: list[str],
    result: CategorizationResult,
    previous: CategorizationResult | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    entity_category_ids = [category.id for category in result.entity_categories]
    relation_category_ids = [category.id for category in result.relation_categories]
    mapped_entities = [mapping.entity for mapping in result.entity_mappings]
    mapped_relations = [mapping.relation for mapping in result.relation_mappings]

    if len(entity_category_ids) != len(set(entity_category_ids)):
        errors.append("Duplicate entity category IDs")
    if len(relation_category_ids) != len(set(relation_category_ids)):
        errors.append("Duplicate relation category IDs")
    if mapped_entities != entities:
        errors.append("Entity mappings do not exactly preserve source order and coverage")
    if mapped_relations != relations:
        errors.append("Relation mappings do not exactly preserve source order and coverage")
    if any(
        mapping.category_id not in set(entity_category_ids) for mapping in result.entity_mappings
    ):
        errors.append("An entity mapping references an unknown category")
    if any(
        mapping.category_id not in set(relation_category_ids)
        for mapping in result.relation_mappings
    ):
        errors.append("A relation mapping references an unknown category")
    if previous is not None:
        current_entity_categories = {
            category.id: category.label for category in result.entity_categories
        }
        current_relation_categories = {
            category.id: category.label for category in result.relation_categories
        }
        for category in previous.entity_categories:
            if current_entity_categories.get(category.id) != category.label:
                errors.append(
                    f"Previous entity category was not preserved: {category.id} — {category.label}"
                )
        for category in previous.relation_categories:
            if current_relation_categories.get(category.id) != category.label:
                errors.append(
                    f"Previous relation category was not preserved: "
                    f"{category.id} — {category.label}"
                )

    return {
        "accepted": not errors,
        "errors": errors,
        "entityCount": len(entities),
        "relationCount": len(relations),
        "entityCategoryCount": len(result.entity_categories),
        "relationCategoryCount": len(result.relation_categories),
    }


def reconcile_carried_mappings(
    entities: list[str],
    relations: list[str],
    result: CategorizationResult,
    previous: CategorizationResult | None,
) -> list[str]:
    if previous is None:
        return []

    repairs: list[str] = []
    previous_entities = {
        mapping.entity: mapping.category_id for mapping in previous.entity_mappings
    }
    previous_relations = {
        mapping.relation: mapping.category_id for mapping in previous.relation_mappings
    }
    current_entities = {mapping.entity: mapping.category_id for mapping in result.entity_mappings}
    current_relations = {
        mapping.relation: mapping.category_id for mapping in result.relation_mappings
    }

    if len(current_entities) == len(result.entity_mappings) and not (
        set(current_entities) - set(entities)
    ):
        reconciled_entities: list[EntityMapping] = []
        for entity in entities:
            category_id = current_entities.get(entity)
            if entity in previous_entities:
                stable_category_id = previous_entities[entity]
                if category_id is None:
                    repairs.append(
                        f"Restored recurring entity `{entity}` as `{stable_category_id}`"
                    )
                elif category_id != stable_category_id:
                    repairs.append(
                        f"Restored stable category for recurring entity `{entity}`: "
                        f"`{category_id}` → `{stable_category_id}`"
                    )
                category_id = stable_category_id
            if category_id is None:
                reconciled_entities = []
                break
            reconciled_entities.append(EntityMapping(entity=entity, category_id=category_id))
        if reconciled_entities:
            result.entity_mappings = reconciled_entities

    if len(current_relations) == len(result.relation_mappings) and not (
        set(current_relations) - set(relations)
    ):
        reconciled_relations: list[RelationMapping] = []
        for relation in relations:
            category_id = current_relations.get(relation)
            if relation in previous_relations:
                stable_category_id = previous_relations[relation]
                if category_id is None:
                    repairs.append(
                        f"Restored recurring relation `{relation}` as `{stable_category_id}`"
                    )
                elif category_id != stable_category_id:
                    repairs.append(
                        f"Restored stable category for recurring relation `{relation}`: "
                        f"`{category_id}` → `{stable_category_id}`"
                    )
                category_id = stable_category_id
            if category_id is None:
                reconciled_relations = []
                break
            reconciled_relations.append(RelationMapping(relation=relation, category_id=category_id))
        if reconciled_relations:
            result.relation_mappings = reconciled_relations

    return repairs


def review_markdown(
    supplied: dict[str, Any],
    result: CategorizationResult,
    validation: dict[str, Any],
    previous: CategorizationResult | None = None,
) -> str:
    previous_entity_ids = (
        {category.id for category in previous.entity_categories} if previous else set()
    )
    previous_relation_ids = (
        {category.id for category in previous.relation_categories} if previous else set()
    )
    new_entity_categories = [
        category for category in result.entity_categories if category.id not in previous_entity_ids
    ]
    new_relation_categories = [
        category
        for category in result.relation_categories
        if category.id not in previous_relation_ids
    ]
    carried_entity_categories = [
        category for category in result.entity_categories if category.id in previous_entity_ids
    ]
    carried_relation_categories = [
        category for category in result.relation_categories if category.id in previous_relation_ids
    ]

    lines = [
        f"# Category trial — {supplied['page_id']}",
        "",
        "## New categories introduced on this page",
        "",
        "### New entity categories",
        "",
    ]
    for category in new_entity_categories:
        lines.append(f"- `{category.id}` — {category.label}")
    if not new_entity_categories:
        lines.append("- None")

    lines.extend(["", "### New relation categories", ""])
    for category in new_relation_categories:
        lines.append(f"- `{category.id}` — {category.label}")
    if not new_relation_categories:
        lines.append("- None")

    lines.extend(["", "## Carried-forward category sets", ""])
    lines.extend(["### Carried entity categories", ""])
    for category in carried_entity_categories:
        lines.append(f"- `{category.id}` — {category.label}")
    if not carried_entity_categories:
        lines.append("- None (initial page)")

    lines.extend(["", "### Carried relation categories", ""])
    for category in carried_relation_categories:
        lines.append(f"- `{category.id}` — {category.label}")
    if not carried_relation_categories:
        lines.append("- None (initial page)")

    entity_groups = {
        category.id: [
            mapping.entity
            for mapping in result.entity_mappings
            if mapping.category_id == category.id
        ]
        for category in result.entity_categories
    }
    relation_groups = {
        category.id: [
            mapping.relation
            for mapping in result.relation_mappings
            if mapping.category_id == category.id
        ]
        for category in result.relation_categories
    }
    entity_category_labels = {category.id: category.label for category in result.entity_categories}
    relation_category_labels = {
        category.id: category.label for category in result.relation_categories
    }
    entity_axial_codes = {
        mapping.entity: entity_category_labels[mapping.category_id]
        for mapping in result.entity_mappings
    }
    relation_axial_codes = {
        mapping.relation: relation_category_labels[mapping.category_id]
        for mapping in result.relation_mappings
    }

    lines.extend(["", "## Entities grouped by category", ""])
    for category in result.entity_categories:
        if not entity_groups[category.id]:
            continue
        new_marker = " — **NEW**" if category.id not in previous_entity_ids else ""
        lines.extend([f"### `{category.id}` — {category.label}{new_marker}", ""])
        lines.extend(f"- {entity}" for entity in entity_groups[category.id])
        lines.append("")

    lines.extend(["## Relations grouped by category", ""])
    for category in result.relation_categories:
        if not relation_groups[category.id]:
            continue
        new_marker = " — **NEW**" if category.id not in previous_relation_ids else ""
        lines.extend([f"### `{category.id}` — {category.label}{new_marker}", ""])
        lines.extend(f"- `{relation}`" for relation in relation_groups[category.id])
        lines.append("")

    lines.extend(
        [
            "## Source page",
            "",
            "### Page summary",
            "",
            supplied["page_summary"],
            "",
            "### Sentences and triples",
            "",
        ]
    )
    for sentence in supplied["sentences"]:
        lines.extend(
            [
                f"#### {sentence['sid']}",
                "",
                sentence["my"],
                "",
                sentence["en"],
                "",
            ]
        )
        if sentence["triples"]:
            for triple in sentence["triples"]:
                lines.append(
                    f"- **{triple['subject']}** — `{triple['predicate']}` → **{triple['object']}**"
                )
                lines.append(
                    "  - **Axial codes:** "
                    f"`{entity_axial_codes.get(triple['subject'], 'UNMAPPED')}` — "
                    f"`{relation_axial_codes.get(triple['predicate'], 'UNMAPPED')}` → "
                    f"`{entity_axial_codes.get(triple['object'], 'UNMAPPED')}`"
                )
        else:
            lines.append("- No triples")
        lines.append("")

    lines.extend(
        [
            "## Validation",
            "",
            "VALID" if validation["accepted"] else "FLAGGED",
        ]
    )
    for error in validation["errors"]:
        lines.append(f"- {error}")
    return "\n".join(lines).rstrip() + "\n"


def rerender(output_dir: Path) -> Path:
    supplied = json.loads((output_dir / "input.json").read_text(encoding="utf-8"))
    payload = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    result = CategorizationResult.model_validate(payload["response"])
    previous = None
    if payload.get("previousCodebook"):
        previous = load_previous_result(Path(payload["previousCodebook"]))
    (output_dir / "review.md").write_text(
        review_markdown(supplied, result, payload["validation"], previous),
        encoding="utf-8",
    )
    return output_dir / "review.md"


def merge_codebooks(
    previous: CategorizationResult | None,
    current: CategorizationResult,
) -> CategorizationResult:
    entity_mappings: dict[str, EntityMapping] = {}
    relation_mappings: dict[str, RelationMapping] = {}
    if previous is not None:
        entity_mappings.update({mapping.entity: mapping for mapping in previous.entity_mappings})
        relation_mappings.update(
            {mapping.relation: mapping for mapping in previous.relation_mappings}
        )
    for mapping in current.entity_mappings:
        entity_mappings.setdefault(mapping.entity, mapping)
    for mapping in current.relation_mappings:
        relation_mappings.setdefault(mapping.relation, mapping)
    return CategorizationResult(
        entity_categories=current.entity_categories,
        relation_categories=current.relation_categories,
        entity_mappings=list(entity_mappings.values()),
        relation_mappings=list(relation_mappings.values()),
    )


def load_previous_result(path: Path) -> CategorizationResult:
    output_dir = path if path.is_dir() else path.parent
    codebook_path = output_dir / "codebook.json"
    if codebook_path.exists():
        return CategorizationResult.model_validate_json(codebook_path.read_text(encoding="utf-8"))

    result_path = output_dir / "result.json" if path.is_dir() else path
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    current = CategorizationResult.model_validate(payload["response"])
    previous = (
        load_previous_result(Path(payload["previousCodebook"]))
        if payload.get("previousCodebook")
        else None
    )
    cumulative = merge_codebooks(previous, current)
    codebook_path.write_text(
        cumulative.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return cumulative


def repair_existing(output_dir: Path) -> Path:
    supplied = json.loads((output_dir / "input.json").read_text(encoding="utf-8"))
    payload = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    result = CategorizationResult.model_validate(payload["response"])
    previous = (
        load_previous_result(Path(payload["previousCodebook"]))
        if payload.get("previousCodebook")
        else None
    )
    triples = [triple for sentence in supplied["sentences"] for triple in sentence["triples"]]
    entities, relations = unique_inputs(triples)
    repairs = reconcile_carried_mappings(entities, relations, result, previous)
    validation = validate(entities, relations, result, previous)
    payload["deterministicRepairs"] = repairs
    payload["validation"] = validation
    payload["response"] = result.model_dump(mode="json")
    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "review.md").write_text(
        review_markdown(supplied, result, validation, previous),
        encoding="utf-8",
    )
    cumulative = merge_codebooks(previous, result)
    (output_dir / "codebook.json").write_text(
        cumulative.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def run(
    volume: int | None = None,
    page_number: int | None = None,
    previous_path: Path | None = None,
) -> Path:
    record = choose_page(volume, page_number)
    supplied = supplied_page(record)
    triples = triples_for(record)
    entities, relations = unique_inputs(triples)
    previous = load_previous_result(previous_path) if previous_path else None
    previous_block = ""
    if previous is not None:
        previous_block = (
            "\n\n"
            + SEQUENTIAL_INSTRUCTION
            + "\n\n<PREVIOUS_AXIAL_CODEBOOK_READ_ONLY>\n"
            + json.dumps(previous.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n</PREVIOUS_AXIAL_CODEBOOK_READ_ONLY>"
        )
    prompt = (
        read_only_historiography_prefix()
        + "\n<CURRENT_AXIAL_CODING_TASK>\n"
        + INSTRUCTION
        + previous_block
        + "\n\nSUPPLIED PAGE:\n"
        + json.dumps(supplied, ensure_ascii=False, indent=2)
        + "\n\nEXACT REQUIRED INVENTORY:\n"
        + json.dumps(
            {
                "entity_count": len(entities),
                "entities": entities,
                "relation_count": len(relations),
                "relations": relations,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n</CURRENT_AXIAL_CODING_TASK>\n"
    )

    base_name = f"vol{int(record['volume'])}_page_{int(record['page']):04d}_high"
    output_dir = OUTPUT_ROOT / base_name
    suffix = 2
    while output_dir.exists():
        output_dir = OUTPUT_ROOT / f"{base_name}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True)

    (output_dir / "input.json").write_text(
        json.dumps(supplied, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CategorizationResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=65536,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH,
                include_thoughts=False,
            ),
        ),
    )

    raw = (response.text or "").strip()
    (output_dir / "raw_response.json").write_text(raw + "\n", encoding="utf-8")
    result = CategorizationResult.model_validate_json(raw)
    repairs = reconcile_carried_mappings(entities, relations, result, previous)
    validation = validate(entities, relations, result, previous)
    usage = response.usage_metadata.model_dump(mode="json") if response.usage_metadata else {}
    payload = {
        "model": MODEL,
        "thinkingLevel": "high",
        "volume": int(record["volume"]),
        "page": int(record["page"]),
        "previousCodebook": str(previous_path) if previous_path else None,
        "deterministicRepairs": repairs,
        "usage": usage,
        "validation": validation,
        "response": result.model_dump(mode="json"),
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "review.md").write_text(
        review_markdown(supplied, result, validation, previous),
        encoding="utf-8",
    )
    cumulative = merge_codebooks(previous, result)
    (output_dir / "codebook.json").write_text(
        cumulative.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int)
    parser.add_argument("--page", type=int)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--rerender", type=Path)
    parser.add_argument("--repair", type=Path)
    args = parser.parse_args()
    if args.rerender:
        print(rerender(args.rerender.resolve()))
        return
    if args.repair:
        print(repair_existing(args.repair.resolve()))
        return
    print(
        run(
            args.volume,
            args.page,
            args.previous.resolve() if args.previous else None,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Database-backed sequential axial coding for Konbaung triples."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel

from konbaung_gemini_incremental_category_trial import (
    MODEL,
    CategorizationResult,
    Category,
    EntityMapping,
    RelationMapping,
    choose_page,
    read_only_historiography_prefix,
    review_markdown,
    supplied_page,
    triples_for,
    unique_inputs,
)
from konbaung_gemini_summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "konbaung_incremental_category_trials"
DB_PATH = OUTPUT_ROOT / "axial_codebook.sqlite3"
CODEBOOK_EXPORT = OUTPUT_ROOT / "axial_codebook.json"
LEGACY_RUNS = [
    OUTPUT_ROOT / "vol2_page_0287_high_03",
    OUTPUT_ROOT / "vol2_page_0288_high",
    OUTPUT_ROOT / "vol2_page_0289_high",
    OUTPUT_ROOT / "vol2_page_0290_high",
]


def configure_fresh_pass(pass_root: Path) -> None:
    global OUTPUT_ROOT, DB_PATH, CODEBOOK_EXPORT, LEGACY_RUNS
    OUTPUT_ROOT = pass_root
    DB_PATH = OUTPUT_ROOT / "axial_codebook.sqlite3"
    CODEBOOK_EXPORT = OUTPUT_ROOT / "axial_codebook.json"
    LEGACY_RUNS = []


class CategoryProposal(BaseModel):
    temp_id: str
    label: str


class UnseenEntityMapping(BaseModel):
    entity: str
    category_ref: str


class UnseenRelationMapping(BaseModel):
    relation: str
    category_ref: str


class AxialDelta(BaseModel):
    new_entity_categories: list[CategoryProposal]
    new_relation_categories: list[CategoryProposal]
    entity_mappings: list[UnseenEntityMapping]
    relation_mappings: list[UnseenRelationMapping]


TASK = """Perform axial coding of the previously unassigned open codes on the supplied page.

The existing category codebook was read deterministically from the project's local database. It is binding. Each category may include up to ten of its most recently assigned tag examples to clarify its current semantic range. Reuse an existing category whenever it adequately captures an open code's analytical role or mechanism. Propose a new category only when none fits.

Entity categories describe an entity's analytical role or position in the Konbaung power field. Relation categories describe a broader mechanism, direction, or function of power. Avoid generic surface types when a historically meaningful axial category is warranted.

Return only a delta:

1. `new_entity_categories`: genuinely necessary new entity categories only, using temporary IDs `NE01`, `NE02`, and so on.
2. `new_relation_categories`: genuinely necessary new relation categories only, using temporary IDs `NR01`, `NR02`, and so on.
3. `entity_mappings`: every item in `unassigned_entities`, exactly once and preserving its string exactly, mapped either to an existing `E` ID or a proposed `NE` temporary ID.
4. `relation_mappings`: every item in `unassigned_relations`, exactly once and preserving its string exactly, mapped either to an existing `R` ID or a proposed `NR` temporary ID.

Do not return or restate existing categories. Do not return mappings for the already-assigned current-page items. Do not invent permanent IDs; the local application assigns them transactionally. Every proposal must be used by at least one mapping. Return no explanations, definitions, confidence scores, summaries, or other fields."""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS categories (
            kind TEXT NOT NULL CHECK (kind IN ('entity', 'relation')),
            category_id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            created_page_key TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (kind, label COLLATE NOCASE)
        );

        CREATE TABLE IF NOT EXISTS assignments (
            kind TEXT NOT NULL CHECK (kind IN ('entity', 'relation')),
            tag TEXT NOT NULL,
            category_id TEXT NOT NULL REFERENCES categories(category_id),
            first_page_key TEXT NOT NULL,
            last_page_key TEXT NOT NULL,
            page_count INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (kind, tag)
        );

        CREATE TABLE IF NOT EXISTS page_assignments (
            page_key TEXT NOT NULL,
            kind TEXT NOT NULL CHECK (kind IN ('entity', 'relation')),
            tag TEXT NOT NULL,
            category_id TEXT NOT NULL REFERENCES categories(category_id),
            PRIMARY KEY (page_key, kind, tag)
        );

        CREATE TABLE IF NOT EXISTS page_runs (
            page_key TEXT PRIMARY KEY,
            volume INTEGER NOT NULL,
            page INTEGER NOT NULL,
            source_dir TEXT NOT NULL,
            model TEXT NOT NULL,
            completed_at TEXT NOT NULL
        );
        """
    )
    return connection


def insert_category(
    connection: sqlite3.Connection,
    kind: str,
    category_id: str,
    label: str,
    page_key: str,
) -> None:
    row = connection.execute(
        "SELECT kind, label FROM categories WHERE category_id = ?",
        (category_id,),
    ).fetchone()
    if row is not None:
        if row["kind"] != kind or row["label"] != label:
            raise ValueError(
                f"Category collision for {category_id}: "
                f"{row['kind']}/{row['label']} versus {kind}/{label}"
            )
        return
    connection.execute(
        """
        INSERT INTO categories(kind, category_id, label, created_page_key, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            kind,
            category_id,
            label,
            page_key,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def record_assignment(
    connection: sqlite3.Connection,
    page_key: str,
    kind: str,
    tag: str,
    category_id: str,
) -> None:
    existing = connection.execute(
        "SELECT category_id FROM assignments WHERE kind = ? AND tag = ?",
        (kind, tag),
    ).fetchone()
    if existing is not None and existing["category_id"] != category_id:
        raise ValueError(
            f"Stable assignment conflict for {kind} `{tag}`: "
            f"{existing['category_id']} versus {category_id}"
        )
    if existing is None:
        connection.execute(
            """
            INSERT INTO assignments(
                kind, tag, category_id, first_page_key, last_page_key, page_count
            ) VALUES (?, ?, ?, ?, ?, 1)
            """,
            (kind, tag, category_id, page_key, page_key),
        )
    else:
        already_on_page = connection.execute(
            """
            SELECT 1 FROM page_assignments
            WHERE page_key = ? AND kind = ? AND tag = ?
            """,
            (page_key, kind, tag),
        ).fetchone()
        if already_on_page is None:
            connection.execute(
                """
                UPDATE assignments
                SET last_page_key = ?, page_count = page_count + 1
                WHERE kind = ? AND tag = ?
                """,
                (page_key, kind, tag),
            )
    connection.execute(
        """
        INSERT OR IGNORE INTO page_assignments(page_key, kind, tag, category_id)
        VALUES (?, ?, ?, ?)
        """,
        (page_key, kind, tag, category_id),
    )


def import_legacy_runs(connection: sqlite3.Connection) -> None:
    for directory in LEGACY_RUNS:
        if not (directory / "result.json").exists():
            continue
        payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        page_key = f"vol{int(payload['volume'])}-p{int(payload['page']):04d}"
        result = CategorizationResult.model_validate(payload["response"])
        with connection:
            for category in result.entity_categories:
                insert_category(
                    connection, "entity", category.id, category.label, page_key
                )
            for category in result.relation_categories:
                insert_category(
                    connection, "relation", category.id, category.label, page_key
                )
            for mapping in result.entity_mappings:
                record_assignment(
                    connection,
                    page_key,
                    "entity",
                    mapping.entity,
                    mapping.category_id,
                )
            for mapping in result.relation_mappings:
                record_assignment(
                    connection,
                    page_key,
                    "relation",
                    mapping.relation,
                    mapping.category_id,
                )
            connection.execute(
                """
                INSERT OR REPLACE INTO page_runs(
                    page_key, volume, page, source_dir, model, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    page_key,
                    int(payload["volume"]),
                    int(payload["page"]),
                    str(directory),
                    str(payload["model"]),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )


def category_rows(
    connection: sqlite3.Connection,
    kind: str,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT category_id, label
        FROM categories
        WHERE kind = ?
        ORDER BY CAST(SUBSTR(category_id, 2) AS INTEGER), category_id
        """,
        (kind,),
    ).fetchall()
    values: list[dict[str, Any]] = []
    for row in rows:
        examples = connection.execute(
            """
            SELECT tag FROM assignments
            WHERE kind = ? AND category_id = ?
            ORDER BY rowid DESC
            LIMIT 10
            """,
            (kind, row["category_id"]),
        ).fetchall()
        values.append(
            {
                "id": row["category_id"],
                "label": row["label"],
                "examples": [example["tag"] for example in examples],
            }
        )
    return values


def assignments_for(
    connection: sqlite3.Connection,
    kind: str,
    tags: list[str],
) -> dict[str, str]:
    if not tags:
        return {}
    placeholders = ",".join("?" for _ in tags)
    rows = connection.execute(
        f"""
        SELECT tag, category_id FROM assignments
        WHERE kind = ? AND tag IN ({placeholders})
        """,
        (kind, *tags),
    ).fetchall()
    return {row["tag"]: row["category_id"] for row in rows}


def next_id(connection: sqlite3.Connection, kind: str) -> str:
    prefix = "E" if kind == "entity" else "R"
    row = connection.execute(
        """
        SELECT COALESCE(MAX(CAST(SUBSTR(category_id, 2) AS INTEGER)), 0) AS maximum
        FROM categories WHERE kind = ?
        """,
        (kind,),
    ).fetchone()
    return f"{prefix}{int(row['maximum']) + 1:02d}"


def validate_delta(
    delta: AxialDelta,
    unseen_entities: list[str],
    unseen_relations: list[str],
    entity_category_ids: set[str],
    relation_category_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    if [mapping.entity for mapping in delta.entity_mappings] != unseen_entities:
        errors.append("Entity delta does not exactly match unassigned entity inventory")
    if [mapping.relation for mapping in delta.relation_mappings] != unseen_relations:
        errors.append("Relation delta does not exactly match unassigned relation inventory")

    entity_temp_ids = [proposal.temp_id for proposal in delta.new_entity_categories]
    relation_temp_ids = [proposal.temp_id for proposal in delta.new_relation_categories]
    if len(entity_temp_ids) != len(set(entity_temp_ids)):
        errors.append("Duplicate temporary entity category ID")
    if len(relation_temp_ids) != len(set(relation_temp_ids)):
        errors.append("Duplicate temporary relation category ID")
    if any(not temp_id.startswith("NE") for temp_id in entity_temp_ids):
        errors.append("Invalid temporary entity category ID")
    if any(not temp_id.startswith("NR") for temp_id in relation_temp_ids):
        errors.append("Invalid temporary relation category ID")

    valid_entity_refs = entity_category_ids | set(entity_temp_ids)
    valid_relation_refs = relation_category_ids | set(relation_temp_ids)
    if any(
        mapping.category_ref not in valid_entity_refs
        for mapping in delta.entity_mappings
    ):
        errors.append("Entity mapping references an unknown category")
    if any(
        mapping.category_ref not in valid_relation_refs
        for mapping in delta.relation_mappings
    ):
        errors.append("Relation mapping references an unknown category")

    used_entity_refs = {mapping.category_ref for mapping in delta.entity_mappings}
    used_relation_refs = {mapping.category_ref for mapping in delta.relation_mappings}
    if set(entity_temp_ids) - used_entity_refs:
        errors.append("An unused entity category was proposed")
    if set(relation_temp_ids) - used_relation_refs:
        errors.append("An unused relation category was proposed")
    return errors


def resolve_proposals(
    connection: sqlite3.Connection,
    kind: str,
    proposals: list[CategoryProposal],
) -> tuple[dict[str, str], list[Category]]:
    temp_to_stable: dict[str, str] = {}
    additions: list[Category] = []
    next_number = int(next_id(connection, kind)[1:])
    prefix = "E" if kind == "entity" else "R"
    existing_labels = {
        row["label"].casefold(): row["category_id"]
        for row in connection.execute(
            "SELECT category_id, label FROM categories WHERE kind = ?",
            (kind,),
        ).fetchall()
    }
    pending_labels: dict[str, str] = {}
    for proposal in proposals:
        normalized = proposal.label.strip().casefold()
        if normalized in existing_labels:
            stable_id = existing_labels[normalized]
        elif normalized in pending_labels:
            stable_id = pending_labels[normalized]
        else:
            stable_id = f"{prefix}{next_number:02d}"
            next_number += 1
            pending_labels[normalized] = stable_id
            additions.append(Category(id=stable_id, label=proposal.label.strip()))
        temp_to_stable[proposal.temp_id] = stable_id
    return temp_to_stable, additions


def export_codebook(connection: sqlite3.Connection) -> None:
    payload = {
        "database": str(DB_PATH),
        "entity_categories": category_rows(connection, "entity"),
        "relation_categories": category_rows(connection, "relation"),
        "entity_assignment_count": connection.execute(
            "SELECT COUNT(*) AS count FROM assignments WHERE kind = 'entity'"
        ).fetchone()["count"],
        "relation_assignment_count": connection.execute(
            "SELECT COUNT(*) AS count FROM assignments WHERE kind = 'relation'"
        ).fetchone()["count"],
    }
    CODEBOOK_EXPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(
    volume: int,
    page_number: int,
    thinking_level: str = "low",
) -> Path:
    connection = connect()
    import_legacy_runs(connection)
    export_codebook(connection)

    record = choose_page(volume, page_number)
    supplied = supplied_page(record)
    triples = triples_for(record)
    entities, relations = unique_inputs(triples)
    known_entities = assignments_for(connection, "entity", entities)
    known_relations = assignments_for(connection, "relation", relations)
    unseen_entities = [tag for tag in entities if tag not in known_entities]
    unseen_relations = [tag for tag in relations if tag not in known_relations]
    entity_catalog = category_rows(connection, "entity")
    relation_catalog = category_rows(connection, "relation")
    prompt_entity_catalog = [
        {
            "id": category["id"],
            "label": category["label"],
            "examples": category["examples"],
        }
        for category in entity_catalog
    ]
    prompt_relation_catalog = [
        {
            "id": category["id"],
            "label": category["label"],
            "examples": category["examples"],
        }
        for category in relation_catalog
    ]

    model_input = {
        "existing_entity_categories": prompt_entity_catalog,
        "existing_relation_categories": prompt_relation_catalog,
        "unassigned_entity_count": len(unseen_entities),
        "unassigned_entities": unseen_entities,
        "unassigned_relation_count": len(unseen_relations),
        "unassigned_relations": unseen_relations,
        "page": supplied,
    }
    prompt = (
        read_only_historiography_prefix()
        + "\n<CURRENT_DATABASE_BACKED_AXIAL_CODING_TASK>\n"
        + TASK
        + "\n\nDETERMINISTIC DATABASE SNAPSHOT AND PAGE INPUT:\n"
        + json.dumps(model_input, ensure_ascii=False, indent=2)
        + "\n</CURRENT_DATABASE_BACKED_AXIAL_CODING_TASK>\n"
    )

    base_name = f"vol{volume}_page_{page_number:04d}_{thinking_level}_db"
    output_dir = OUTPUT_ROOT / base_name
    suffix = 2
    while output_dir.exists():
        output_dir = OUTPUT_ROOT / f"{base_name}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True)
    (output_dir / "input.json").write_text(
        json.dumps(model_input, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AxialDelta,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=65536,
            thinking_config=types.ThinkingConfig(
                thinking_level={
                    "minimal": types.ThinkingLevel.MINIMAL,
                    "low": types.ThinkingLevel.LOW,
                    "medium": types.ThinkingLevel.MEDIUM,
                    "high": types.ThinkingLevel.HIGH,
                }[thinking_level],
                include_thoughts=False,
            ),
        ),
    )
    raw = (response.text or "").strip()
    (output_dir / "raw_response.json").write_text(raw + "\n", encoding="utf-8")
    delta = AxialDelta.model_validate_json(raw)
    errors = validate_delta(
        delta,
        unseen_entities,
        unseen_relations,
        {category["id"] for category in entity_catalog},
        {category["id"] for category in relation_catalog},
    )
    if errors:
        raise ValueError("; ".join(errors))

    entity_temp_map, new_entity_categories = resolve_proposals(
        connection, "entity", delta.new_entity_categories
    )
    relation_temp_map, new_relation_categories = resolve_proposals(
        connection, "relation", delta.new_relation_categories
    )
    resolved_entities = dict(known_entities)
    for mapping in delta.entity_mappings:
        resolved_entities[mapping.entity] = entity_temp_map.get(
            mapping.category_ref, mapping.category_ref
        )
    resolved_relations = dict(known_relations)
    for mapping in delta.relation_mappings:
        resolved_relations[mapping.relation] = relation_temp_map.get(
            mapping.category_ref, mapping.category_ref
        )

    page_key = supplied["page_id"]
    with connection:
        for category in new_entity_categories:
            insert_category(connection, "entity", category.id, category.label, page_key)
        for category in new_relation_categories:
            insert_category(
                connection, "relation", category.id, category.label, page_key
            )
        for entity in entities:
            record_assignment(
                connection, page_key, "entity", entity, resolved_entities[entity]
            )
        for relation in relations:
            record_assignment(
                connection, page_key, "relation", relation, resolved_relations[relation]
            )
        connection.execute(
            """
            INSERT INTO page_runs(
                page_key, volume, page, source_dir, model, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                page_key,
                volume,
                page_number,
                str(output_dir),
                MODEL,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    final_entity_categories = [
        Category(id=row["id"], label=row["label"])
        for row in category_rows(connection, "entity")
    ]
    final_relation_categories = [
        Category(id=row["id"], label=row["label"])
        for row in category_rows(connection, "relation")
    ]
    canonical = CategorizationResult(
        entity_categories=final_entity_categories,
        relation_categories=final_relation_categories,
        entity_mappings=[
            EntityMapping(entity=entity, category_id=resolved_entities[entity])
            for entity in entities
        ],
        relation_mappings=[
            RelationMapping(
                relation=relation, category_id=resolved_relations[relation]
            )
            for relation in relations
        ],
    )
    previous_snapshot = CategorizationResult(
        entity_categories=[
            Category(id=row["id"], label=row["label"]) for row in entity_catalog
        ],
        relation_categories=[
            Category(id=row["id"], label=row["label"]) for row in relation_catalog
        ],
        entity_mappings=[],
        relation_mappings=[],
    )
    usage = (
        response.usage_metadata.model_dump(mode="json")
        if response.usage_metadata
        else {}
    )
    validation = {
        "accepted": True,
        "errors": [],
        "entityCount": len(entities),
        "relationCount": len(relations),
        "knownEntityCount": len(known_entities),
        "knownRelationCount": len(known_relations),
        "newEntityAssignmentCount": len(unseen_entities),
        "newRelationAssignmentCount": len(unseen_relations),
        "newEntityCategoryCount": len(new_entity_categories),
        "newRelationCategoryCount": len(new_relation_categories),
    }
    result_payload = {
        "model": MODEL,
        "thinkingLevel": thinking_level,
        "volume": volume,
        "page": page_number,
        "database": str(DB_PATH),
        "usage": usage,
        "validation": validation,
        "modelDelta": delta.model_dump(mode="json"),
        "newEntityCategories": [
            category.model_dump(mode="json") for category in new_entity_categories
        ],
        "newRelationCategories": [
            category.model_dump(mode="json") for category in new_relation_categories
        ],
        "response": canonical.model_dump(mode="json"),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "review.md").write_text(
        review_markdown(supplied, canonical, validation, previous_snapshot),
        encoding="utf-8",
    )
    export_codebook(connection)
    connection.close()
    return output_dir


def initialize() -> None:
    connection = connect()
    import_legacy_runs(connection)
    export_codebook(connection)
    counts = connection.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM categories WHERE kind = 'entity') AS entity_categories,
            (SELECT COUNT(*) FROM categories WHERE kind = 'relation') AS relation_categories,
            (SELECT COUNT(*) FROM assignments WHERE kind = 'entity') AS entities,
            (SELECT COUNT(*) FROM assignments WHERE kind = 'relation') AS relations
        """
    ).fetchone()
    print(dict(counts))
    connection.close()


def combine_reviews() -> Path:
    connection = connect()
    runs = connection.execute(
        """
        SELECT page_key, source_dir
        FROM page_runs
        ORDER BY volume, page
        """
    ).fetchall()
    connection.close()
    if not runs:
        raise ValueError("The selected pass contains no completed page runs")

    sections: list[str] = []
    summary_path = OUTPUT_ROOT / "PASS_SUMMARY.md"
    if summary_path.exists():
        sections.append(summary_path.read_text(encoding="utf-8").rstrip())
    for run_record in runs:
        review_path = Path(run_record["source_dir"]) / "review.md"
        if not review_path.exists():
            raise FileNotFoundError(
                f"Missing review for {run_record['page_key']}: {review_path}"
            )
        sections.append(review_path.read_text(encoding="utf-8").rstrip())

    output_path = OUTPUT_ROOT / "COMBINED_REVIEW.md"
    output_path.write_text(
        "\n\n---\n\n".join(sections) + "\n",
        encoding="utf-8",
    )
    return output_path


def export_codebook_markdown() -> Path:
    connection = connect()
    lines = [
        "# Final axial codebook",
        "",
    ]
    for kind, heading in (
        ("entity", "Entity categories"),
        ("relation", "Relation categories"),
    ):
        category_records = connection.execute(
            """
            SELECT category_id, label, created_page_key
            FROM categories
            WHERE kind = ?
            ORDER BY CAST(SUBSTR(category_id, 2) AS INTEGER), category_id
            """,
            (kind,),
        ).fetchall()
        assignment_total = connection.execute(
            "SELECT COUNT(*) AS count FROM assignments WHERE kind = ?",
            (kind,),
        ).fetchone()["count"]
        lines.extend(
            [
                f"## {heading}",
                "",
                f"{len(category_records)} categories; {assignment_total} assigned tags.",
                "",
            ]
        )
        for category_record in category_records:
            assignments = connection.execute(
                """
                SELECT tag
                FROM assignments
                WHERE kind = ? AND category_id = ?
                ORDER BY page_count DESC, tag COLLATE NOCASE
                """,
                (kind, category_record["category_id"]),
            ).fetchall()
            lines.extend(
                [
                    f"### `{category_record['category_id']}` — "
                    f"{category_record['label']}",
                    "",
                    f"Introduced on `{category_record['created_page_key']}`; "
                    f"{len(assignments)} assigned tags.",
                    "",
                ]
            )
            lines.extend(f"- {assignment['tag']}" for assignment in assignments)
            if not assignments:
                lines.append("- No assigned tags")
            lines.append("")
    connection.close()

    output_path = OUTPUT_ROOT / "FINAL_CODEBOOK.md"
    output_path.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    return output_path


def refresh_pass_summary() -> Path:
    connection = connect()
    run_records = connection.execute(
        """
        SELECT page_key, source_dir
        FROM page_runs
        ORDER BY volume, page
        """
    ).fetchall()
    category_counts = connection.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM categories WHERE kind = 'entity')
                AS entity_categories,
            (SELECT COUNT(*) FROM categories WHERE kind = 'relation')
                AS relation_categories,
            (SELECT COUNT(*) FROM assignments WHERE kind = 'entity')
                AS entity_assignments,
            (SELECT COUNT(*) FROM assignments WHERE kind = 'relation')
                AS relation_assignments
        """
    ).fetchone()
    connection.close()
    if not run_records:
        raise ValueError("The selected pass contains no completed page runs")

    rows: list[dict[str, Any]] = []
    totals_by_level: dict[str, dict[str, int]] = {}
    for run_record in run_records:
        source_dir = Path(run_record["source_dir"])
        payload = json.loads(
            (source_dir / "result.json").read_text(encoding="utf-8")
        )
        level = str(payload["thinkingLevel"]).upper()
        usage = payload["usage"]
        totals = totals_by_level.setdefault(level, {"thinking": 0, "total": 0})
        totals["thinking"] += int(usage.get("thoughts_token_count") or 0)
        totals["total"] += int(usage.get("total_token_count") or 0)
        rows.append(
            {
                "page": int(payload["page"]),
                "level": level,
                "entities": int(payload["validation"]["entityCount"]),
                "relations": int(payload["validation"]["relationCount"]),
                "new_entities": int(
                    payload["validation"]["newEntityCategoryCount"]
                ),
                "new_relations": int(
                    payload["validation"]["newRelationCategoryCount"]
                ),
                "source_dir": source_dir,
            }
        )

    pages = [row["page"] for row in rows]
    lines = [
        "# Volume 1 fresh axial-coding pass",
        "",
        f"This pass began with an empty codebook and has processed {len(rows)} "
        f"valid volume-1 pages, from page {min(pages)} through page {max(pages)}.",
        "",
        "| Page | Thinking | Entities | Relations | New entity categories | "
        "New relation categories |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['page']} | `{row['level']}` | {row['entities']} | "
            f"{row['relations']} | {row['new_entities']} | "
            f"{row['new_relations']} |"
        )

    lines.extend(["", "## Page results", ""])
    for row in rows:
        relative = row["source_dir"].relative_to(OUTPUT_ROOT).as_posix()
        lines.append(f"- [Page {row['page']}]({relative}/review.md)")

    lines.extend(
        [
            "",
            "## Final database state",
            "",
            f"- Entity categories: {category_counts['entity_categories']}",
            f"- Relation categories: {category_counts['relation_categories']}",
            f"- Stored entity assignments: "
            f"{category_counts['entity_assignments']}",
            f"- Stored relation assignments: "
            f"{category_counts['relation_assignments']}",
        ]
    )
    complete_thinking = 0
    complete_total = 0
    for level, totals in totals_by_level.items():
        complete_thinking += totals["thinking"]
        complete_total += totals["total"]
        lines.append(
            f"- `{level}` calls: {totals['thinking']:,} thinking tokens; "
            f"{totals['total']:,} total tokens"
        )
    lines.extend(
        [
            f"- Complete pass: {complete_thinking:,} thinking tokens; "
            f"{complete_total:,} total tokens",
            "",
            "The canonical pass state is stored in `axial_codebook.sqlite3`; "
            "`axial_codebook.json` is its readable export.",
            "",
            "## Short review",
            "",
            "All page outputs passed exact inventory coverage validation before "
            "their assignments were committed. Exact recurring strings are resolved "
            "locally; spelling or formatting variants can still receive different "
            "assignments because this pass does not perform identity resolution.",
        ]
    )
    output_path = OUTPUT_ROOT / "PASS_SUMMARY.md"
    output_path.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    return output_path


def compile_full_results() -> Path:
    summary_path = refresh_pass_summary()
    codebook_path = export_codebook_markdown()
    connection = connect()
    run_records = connection.execute(
        """
        SELECT page_key, source_dir
        FROM page_runs
        ORDER BY volume, page
        """
    ).fetchall()
    connection.close()
    sections = [
        summary_path.read_text(encoding="utf-8").rstrip(),
        codebook_path.read_text(encoding="utf-8").rstrip(),
    ]
    for run_record in run_records:
        review_path = Path(run_record["source_dir"]) / "review.md"
        sections.append(review_path.read_text(encoding="utf-8").rstrip())
    output_path = OUTPUT_ROOT / "COMPILED_RESULTS.md"
    output_path.write_text(
        "\n\n---\n\n".join(sections) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--combine-reviews", action="store_true")
    parser.add_argument("--export-codebook-md", action="store_true")
    parser.add_argument("--refresh-summary", action="store_true")
    parser.add_argument("--compile-full-results", action="store_true")
    parser.add_argument("--volume", type=int)
    parser.add_argument("--page", type=int)
    parser.add_argument("--fresh-pass-root", type=Path)
    parser.add_argument(
        "--thinking-level",
        choices=("minimal", "low", "medium", "high"),
        default="low",
    )
    args = parser.parse_args()
    if args.fresh_pass_root:
        configure_fresh_pass(args.fresh_pass_root.resolve())
    if args.initialize:
        initialize()
        return
    if args.combine_reviews:
        print(combine_reviews())
        return
    if args.export_codebook_md:
        print(export_codebook_markdown())
        return
    if args.refresh_summary:
        print(refresh_pass_summary())
        return
    if args.compile_full_results:
        print(compile_full_results())
        return
    if args.volume is None or args.page is None:
        parser.error("--volume and --page are required unless --initialize is used")
    print(run(args.volume, args.page, args.thinking_level))


if __name__ == "__main__":
    main()

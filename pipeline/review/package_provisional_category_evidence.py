#!/usr/bin/env python3
"""Package provisional axial categories with their triple and sentence evidence."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import konbaung_gemini_flashlite_axial_full_corpus_batch as full


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "konbaung_flashlite_axial_full_corpus_20260729"
PACKAGE_NAME = "konbaung_provisional_categories_evidence_20260729"
OUTPUT_DIR = ROOT / PACKAGE_NAME
ZIP_PATH = ROOT / f"{PACKAGE_NAME}.zip"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tag_descriptor(
    page_key: str,
    tag_id: str,
    taxonomy_labels: dict[str, str],
    provisional_by_id: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    provisional = provisional_by_id.get((page_key, tag_id))
    if provisional is not None:
        return {
            "id": tag_id,
            "scoped_id": f"{tag_id}@{page_key}",
            "label": provisional["label"],
            "provisional": True,
        }
    return {
        "id": tag_id,
        "scoped_id": tag_id,
        "label": taxonomy_labels[tag_id],
        "provisional": False,
    }


def main() -> None:
    final_dir = RUN_ROOT / "final"
    category_source = final_dir / "all_provisional_categories.jsonl"
    triple_source = final_dir / "all_tagged_triples.jsonl"
    categories = read_jsonl(category_source)
    triples = read_jsonl(triple_source)
    if len(categories) != 10:
        raise ValueError(f"Expected 10 provisional categories, found {len(categories)}")

    prefix, _, _ = full.load_prompt_package()
    taxonomy_labels = full.taxonomy_labels(prefix)
    provisional_by_id = {(category["key"], category["id"]): category for category in categories}
    page_cache: dict[str, dict[str, Any]] = {}
    sentence_cache: dict[tuple[str, str], dict[str, str]] = {}
    for category in categories:
        key = category["key"]
        volume = int(key[3])
        page_data = full.read_json(RUN_ROOT / "pages" / f"vol{volume}" / key / "page_data.json")
        page_cache[key] = page_data
        for sentence in page_data["sentences"]:
            sentence_cache[(key, sentence["sid"])] = sentence

    evidence_rows: list[dict[str, Any]] = []
    evidence_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for category in categories:
        page_key = category["key"]
        scoped_id = f"{category['id']}@{page_key}"
        for triple in triples:
            if triple["key"] != page_key:
                continue
            positions = [
                field for field, tag_id in triple["tags"].items() if tag_id == category["id"]
            ]
            if not positions:
                continue
            sentence = sentence_cache[(page_key, triple["sid"])]
            page_data = page_cache[page_key]
            tags = {
                field: tag_descriptor(
                    page_key,
                    tag_id,
                    taxonomy_labels,
                    provisional_by_id,
                )
                for field, tag_id in triple["tags"].items()
            }
            row = {
                "association_id": (f"{scoped_id}:{triple['sid']}:{int(triple['i']):03d}"),
                "category_scoped_id": scoped_id,
                "category_positions": positions,
                "page": {
                    "key": page_key,
                    "volume": triple["volume"],
                    "page": triple["page"],
                    "summary": page_data["summary"],
                },
                "sentence": {
                    "sid": sentence["sid"],
                    "my": sentence["my"],
                    "en": sentence["en"],
                },
                "triple": {
                    "i": triple["i"],
                    "sid": triple["sid"],
                    "subject": triple["subject"],
                    "predicate": triple["predicate"],
                    "object": triple["object"],
                },
                "tags": tags,
            }
            evidence_rows.append(row)
            evidence_by_category[scoped_id].append(row)

    category_records: list[dict[str, Any]] = []
    for category in categories:
        scoped_id = f"{category['id']}@{category['key']}"
        usages = evidence_by_category[scoped_id]
        category_records.append(
            {
                "scoped_id": scoped_id,
                "page_key": category["key"],
                "kind": ("entity" if category["kind"] == "new_entity_categories" else "relation"),
                "temporary_id": category["id"],
                "label": category["label"],
                "definition": category["definition"],
                "justification": category["justification"],
                "usage_triples": len(usages),
                "usage_sentences": len({usage["sentence"]["sid"] for usage in usages}),
                "evidence": usages,
            }
        )

    unique_triples = {
        (
            row["page"]["key"],
            row["triple"]["sid"],
            int(row["triple"]["i"]),
        )
        for row in evidence_rows
    }
    sentence_groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in evidence_rows:
        sentence_key = (row["page"]["key"], row["sentence"]["sid"])
        value = sentence_groups.setdefault(
            sentence_key,
            {
                "page": row["page"],
                "sentence": row["sentence"],
                "category_scoped_ids": [],
                "triple_ids": [],
            },
        )
        if row["category_scoped_id"] not in value["category_scoped_ids"]:
            value["category_scoped_ids"].append(row["category_scoped_id"])
        triple_id = f"{row['page']['key']}:{row['triple']['sid']}:{int(row['triple']['i']):03d}"
        if triple_id not in value["triple_ids"]:
            value["triple_ids"].append(triple_id)
    sentence_rows = list(sentence_groups.values())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "categories_with_evidence.json", category_records)
    write_jsonl(OUTPUT_DIR / "category_triple_associations.jsonl", evidence_rows)
    write_jsonl(OUTPUT_DIR / "sentence_contexts.jsonl", sentence_rows)

    review_lines = [
        "# Konbaung provisional axial categories and evidence",
        "",
        f"- Provisional categories: **{len(category_records)}**",
        f"- Entity categories: **{sum(row['kind'] == 'entity' for row in category_records)}**",
        f"- Relation categories: **{sum(row['kind'] == 'relation' for row in category_records)}**",
        f"- Unique evidence triples: **{len(unique_triples)}**",
        f"- Category–triple associations: **{len(evidence_rows)}**",
        f"- Unique sentence contexts: **{len(sentence_rows)}**",
        "",
        (
            "Temporary IDs are page-scoped. For example, `NE01@vol1-p0216` "
            "and `NE01@vol1-p0249` are different proposals."
        ),
        "",
    ]
    for category in category_records:
        review_lines.extend(
            [
                "---",
                "",
                f"## {category['label']}",
                "",
                f"- Scoped ID: `{category['scoped_id']}`",
                f"- Kind: **{category['kind']}**",
                f"- Source page: `{category['page_key']}`",
                f"- Definition: {category['definition']}",
                f"- Justification: {category['justification']}",
                f"- Evidence triples: **{category['usage_triples']}**",
                "",
            ]
        )
        for index, evidence in enumerate(category["evidence"], start=1):
            triple = evidence["triple"]
            sentence = evidence["sentence"]
            tags = evidence["tags"]
            review_lines.extend(
                [
                    f"### Evidence {index}: `{sentence['sid']}` / triple {triple['i']}",
                    "",
                    f"**Triple:** {triple['subject']} — `{triple['predicate']}` → {triple['object']}",
                    "",
                    (
                        "**Axial tags:** "
                        f"{tags['s']['id']} — {tags['s']['label']}; "
                        f"{tags['r']['id']} — {tags['r']['label']}; "
                        f"{tags['o']['id']} — {tags['o']['label']}"
                    ),
                    "",
                    f"**Provisional position(s):** {', '.join(evidence['category_positions'])}",
                    "",
                    "**Burmese sentence:**",
                    "",
                    sentence["my"],
                    "",
                    "**English sentence:**",
                    "",
                    sentence["en"],
                    "",
                ]
            )
    (OUTPUT_DIR / "review.md").write_text(
        "\n".join(review_lines).rstrip() + "\n",
        encoding="utf-8",
    )

    readme = """# Provisional category evidence package

This package contains every provisional entity and relation category proposed
by Gemini during final Konbaung axial coding, every tagged triple that uses one
of those categories, and the complete Burmese and English sentence context.

Files:

- `categories_with_evidence.json`: categories with nested evidence records.
- `category_triple_associations.jsonl`: one row per category–triple link.
- `sentence_contexts.jsonl`: deduplicated sentence contexts.
- `review.md`: human-readable combined review with complete tag names.
- `manifest.json`: counts and source checksums.

Temporary category IDs are page-scoped and have therefore been namespaced as
`NE01@page-key` or `NR01@page-key` in this package.
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(RUN_ROOT),
        "counts": {
            "provisional_categories": len(category_records),
            "entity_categories": sum(row["kind"] == "entity" for row in category_records),
            "relation_categories": sum(row["kind"] == "relation" for row in category_records),
            "unique_evidence_triples": len(unique_triples),
            "category_triple_associations": len(evidence_rows),
            "unique_sentence_contexts": len(sentence_rows),
        },
        "sources": {
            "provisional_categories": {
                "path": str(category_source),
                "sha256": sha256(category_source),
            },
            "tagged_triples": {
                "path": str(triple_source),
                "sha256": sha256(triple_source),
            },
        },
    }
    write_json(OUTPUT_DIR / "manifest.json", manifest)

    with zipfile.ZipFile(
        ZIP_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(OUTPUT_DIR.iterdir(), key=lambda value: value.name):
            archive.write(path, f"{PACKAGE_NAME}/{path.name}")
    print(
        json.dumps(
            {
                "zip": str(ZIP_PATH),
                "size_bytes": ZIP_PATH.stat().st_size,
                **manifest["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = Path(
    r"C:\Users\conra\Desktop\dighumproject"
    r"\konbaung_flashlite_axial_full_corpus_20260729"
)
PROMPT_PATH = Path(
    r"C:\Users\conra\Desktop\dighumproject"
    r"\konbaung_flashlite_axial_coding_prompt_v2.md"
)
V3_ROOT = APP_ROOT / "data" / "konbaung_historiography_v3_canonical_20260724"
OUTPUT_ROOT = APP_ROOT / "data" / "konbaung_axial_categories_v2"
CLOSED_SCHEMA_ARCHIVE = Path(
    r"C:\Users\conra\Downloads"
    r"\konbaung_flashlite_axial_closed_schema_20260729.zip"
)
CLOSED_SCHEMA_ARCHIVE_SHA256 = (
    "dba65e85c8c969d15fcae2df0143859d920372a29b01872fdfea32d8b29ec3c7"
)
TAXONOMY_PATTERN = re.compile(
    r"^- \*\*(?P<id>[ER]\d{2}) (?P<label>[^*]+)\*\*: (?P<definition>.+)$"
)

# Exact replacements supplied by the closed-schema archive's
# provisional_remap_audit.json. Keys are (page, page-local triple index, field).
CLOSED_SCHEMA_FIELD_REMAPS = {
    ("vol1-p0216", 3, "o"): "E37",
    ("vol1-p0249", 0, "r"): "R42",
    ("vol1-p0249", 0, "o"): "E40",
    ("vol1-p0302", 9, "s"): "E05",
    ("vol1-p0302", 11, "o"): "E07",
    ("vol1-p0302", 17, "o"): "E07",
    ("vol1-p0302", 18, "o"): "E07",
    ("vol1-p0302", 22, "s"): "E07",
    ("vol1-p0374", 12, "r"): "R09",
    ("vol3-p0071", 5, "r"): "R65",
    ("vol3-p0185", 6, "o"): "E46",
    ("vol3-p0185", 12, "o"): "E46",
    ("vol3-p0185", 14, "o"): "E46",
    ("vol3-p0224", 3, "o"): "E46",
}

# The archive left vol2-p0217 unresolved because its response returned 19 of
# 20 annotations. The complete source run contains all 20. Close its two
# page-local categories into the existing vocabulary: sacred display objects
# are E31, and material cosmological representation is R65.
CLOSED_SCHEMA_CATEGORY_REMAPS = {
    ("vol2-p0217", "NE01"): "E31",
    ("vol2-p0217", "NR01"): "R65",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def parse_taxonomy() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    entities: dict[str, dict[str, Any]] = {}
    relations: dict[str, dict[str, Any]] = {}
    for line in PROMPT_PATH.read_text(encoding="utf-8").splitlines():
        match = TAXONOMY_PATTERN.match(line)
        if not match:
            continue
        category_id = match.group("id")
        descriptor = {
            "id": category_id,
            "tagId": category_id,
            "label": match.group("label").strip(),
            "definition": match.group("definition").strip(),
            "provisional": False,
        }
        (entities if category_id.startswith("E") else relations)[category_id] = (
            descriptor
        )
    if len(entities) != 52 or len(relations) != 81:
        raise ValueError(
            f"Expected 52 entity and 81 relation categories; "
            f"found {len(entities)} and {len(relations)}"
        )
    return entities, relations


def load_v3_sentences() -> dict[str, dict[str, Any]]:
    sentences: dict[str, dict[str, Any]] = {}
    for volume_id in ("vol1", "vol2", "vol3"):
        path = V3_ROOT / "sentences" / f"{volume_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        sentences.update(payload["sentences"])
    return sentences


def closed_schema_tag(
    key: str,
    triple_index: int,
    field: str,
    tag_id: str,
) -> tuple[str, str | None]:
    exact = CLOSED_SCHEMA_FIELD_REMAPS.get((key, triple_index, field))
    if exact is not None:
        return exact, "archive"
    if tag_id.startswith(("NE", "NR")):
        collapsed = CLOSED_SCHEMA_CATEGORY_REMAPS.get((key, tag_id))
        if collapsed is None:
            raise ValueError(
                f"Unmapped provisional category in {key} "
                f"triple {triple_index} field {field}: {tag_id}"
            )
        return collapsed, "manual"
    return tag_id, None


def main() -> None:
    tagged_path = SOURCE_ROOT / "final" / "all_tagged_triples.jsonl"
    pages_path = SOURCE_ROOT / "final" / "all_pages.jsonl"
    report_path = SOURCE_ROOT / "final_report.json"

    entity_catalog, relation_catalog = parse_taxonomy()
    page_records = read_jsonl(pages_path)
    page_by_key = {record["key"]: record for record in page_records}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for record in report["other_unresolved_pages"]:
        page_by_key.setdefault(
            record["key"],
            {
                "key": record["key"],
                "status": record["category"],
                "errors": record["errors"],
            },
        )
    for key in report["truncated_pages"]:
        page_by_key.setdefault(key, {"key": key, "status": "truncated"})

    v3_sentences = load_v3_sentences()
    assignment_rows: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    ordinal_by_sid: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    remap_counts: Counter[str] = Counter()
    enriched: list[dict[str, Any]] = []

    for row in read_jsonl(tagged_path):
        sid = row["sid"]
        source_sentence = v3_sentences.get(sid)
        if source_sentence is None:
            raise ValueError(f"Tagged row references unknown V3 sentence: {sid}")
        ordinal_by_sid[sid] += 1
        ordinal = ordinal_by_sid[sid]
        source_triples = source_sentence["triples"]
        if ordinal > len(source_triples):
            raise ValueError(f"Too many tagged triples for {sid}")
        source_triple = source_triples[ordinal - 1]
        expected = (
            source_triple["subject"],
            source_triple["predicate"],
            source_triple["object"],
        )
        actual = (row["subject"], row["predicate"], row["object"])
        if actual != expected:
            raise ValueError(
                f"Tagged/V3 triple mismatch for {sid} #{ordinal}: "
                f"{actual!r} != {expected!r}"
            )

        key = row["key"]
        raw_tags = row["tags"]
        tags = {}
        for field in ("s", "r", "o"):
            tags[field], remap_source = closed_schema_tag(
                key,
                int(row["i"]),
                field,
                raw_tags[field],
            )
            if remap_source is not None:
                remap_counts[remap_source] += 1
        if tags["s"] not in entity_catalog or tags["o"] not in entity_catalog:
            raise ValueError(f"Unknown entity category in {key}: {tags}")
        if tags["r"] not in relation_catalog:
            raise ValueError(f"Unknown relation category in {key}: {tags}")

        assignment_rows[sid][ordinal] = tags
        entity_counts.update((tags["s"], tags["o"]))
        relation_counts.update((tags["r"],))
        enriched.append(
            {
                "id": f"v3-{sid}-t{ordinal:03d}",
                "sid": sid,
                "ordinal": ordinal,
                "volumeId": source_sentence["volumeId"],
                "ownerPage": int(source_sentence["ownerPage"]),
                "pages": [int(page) for page in source_sentence["pages"]],
                "selectedPage": key,
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object": row["object"],
                "categories": tags,
                "sentenceMy": source_sentence["my"],
                "sentenceEn": source_sentence["en"],
            }
        )

    status_counts: Counter[str] = Counter()
    assignments_by_volume: dict[str, dict[str, Any]] = {
        "vol1": {},
        "vol2": {},
        "vol3": {},
    }
    unresolved_triples = 0
    for sid, sentence in sorted(v3_sentences.items()):
        if not sentence["triples"]:
            continue
        key = sentence["selectedFrom"]["key"]
        page_record = page_by_key.get(key, {})
        tagged = assignment_rows.get(sid, {})
        if len(tagged) == len(sentence["triples"]):
            status = "accepted"
            reason = None
        else:
            status = str(page_record.get("status") or "unresolved")
            errors = "; ".join(page_record.get("errors", []))
            reason = errors or (
                "The Gemini response was truncated."
                if status == "truncated"
                else "No valid final axial result is available for the selected page."
            )
            unresolved_triples += len(sentence["triples"]) - len(tagged)
        status_counts[status] += 1
        assignments_by_volume[sentence["volumeId"]][sid] = {
            "status": status,
            "sourcePage": key,
            "reason": reason,
            "triples": {str(ordinal): tags for ordinal, tags in sorted(tagged.items())},
        }

    if len(enriched) != int(report["tagged_triples"]):
        raise ValueError(
            f"Expected {report['tagged_triples']} tagged triples; found {len(enriched)}"
        )
    if remap_counts != {"archive": 14, "manual": 22}:
        raise ValueError(
            "Closed-schema remap coverage changed: "
            f"{dict(sorted(remap_counts.items()))}"
        )

    for category_id, descriptor in entity_catalog.items():
        descriptor["count"] = entity_counts[category_id]
    for category_id, descriptor in relation_catalog.items():
        descriptor["count"] = relation_counts[category_id]

    catalog = {
        "schemaVersion": 1,
        "entities": sorted(
            entity_catalog.values(),
            key=lambda item: (item["provisional"], item["id"]),
        ),
        "relations": sorted(
            relation_catalog.values(),
            key=lambda item: (item["provisional"], item["id"]),
        ),
    }
    write_json(OUTPUT_ROOT / "catalog.json", catalog)
    for volume_id, assignments in assignments_by_volume.items():
        write_json(
            OUTPUT_ROOT / "assignments" / f"{volume_id}.json",
            {"schemaVersion": 1, "sentences": assignments},
        )
    write_jsonl(OUTPUT_ROOT / "triples.jsonl", enriched)

    manifest = {
        "schemaVersion": 1,
        "id": "konbaung_axial_categories_v2",
        "description": (
            "Final axial category assignments over canonical V3 historiography triples"
        ),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "model": report["model"],
        "promptVersion": report["prompt_version"],
        "thinkingLevel": report["thinking_level"],
        "selectionRule": report["selection_rule"],
        "source": {
            "runRoot": str(SOURCE_ROOT),
            "taggedTriples": str(tagged_path),
            "taggedTriplesSha256": sha256(tagged_path),
            "prompt": str(PROMPT_PATH),
            "promptSha256": sha256(PROMPT_PATH),
            "v3Index": str(V3_ROOT / "index.json"),
            "v3IndexSha256": sha256(V3_ROOT / "index.json"),
            "closedSchemaArchive": str(CLOSED_SCHEMA_ARCHIVE),
            "closedSchemaArchiveSha256": CLOSED_SCHEMA_ARCHIVE_SHA256,
            "closedSchemaRemaps": {
                "archiveAuditFields": remap_counts["archive"],
                "manuallyClosedUnresolvedFields": remap_counts["manual"],
            },
        },
        "totals": {
            "taggedTriples": len(enriched),
            "unresolvedTriples": unresolved_triples,
            "entityCategories": len(entity_catalog),
            "relationCategories": len(relation_catalog),
            "provisionalEntityCategories": sum(
                bool(item["provisional"]) for item in entity_catalog.values()
            ),
            "provisionalRelationCategories": sum(
                bool(item["provisional"]) for item in relation_catalog.values()
            ),
            "sentenceStatuses": dict(sorted(status_counts.items())),
        },
    }
    write_json(OUTPUT_ROOT / "manifest.json", manifest)
    print(json.dumps(manifest["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

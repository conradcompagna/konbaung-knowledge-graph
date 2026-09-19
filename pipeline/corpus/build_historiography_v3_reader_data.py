#!/usr/bin/env python3
"""Build the reader's canonical, sentence-deduplicated V3 annotation snapshot."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "konbaung_historiography_ungrounded_v3_full_batch_20260723"
SOURCE_PAGES = RUN_ROOT / "final" / "all_pages.jsonl"
SOURCE_REPORT = RUN_ROOT / "final_report.json"
BASE_DATA_ROOT = ROOT / "konbaung_reader_app" / "static" / "data" / "konbaung"
OUTPUT_ROOT = (
    ROOT / "konbaung_reader_app" / "data" / "konbaung_historiography_v3_canonical_20260724"
)
SELECTION_RULE = "max_triples_then_owner_page_then_lowest_page"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def page_key(volume_id: str, page_number: int) -> str:
    return f"{volume_id}-p{page_number:04d}"


def validate_decision(
    key: str,
    sid: str,
    decision: dict[str, Any],
) -> None:
    triples = decision["triples"]
    if decision["decision"] == "annotate":
        if decision["justification"] or not triples:
            raise ValueError(f"{key}/{sid}: invalid annotate decision")
    elif decision["decision"] == "skip":
        if not decision["justification"] or triples:
            raise ValueError(f"{key}/{sid}: invalid skip decision")
    else:
        raise ValueError(f"{key}/{sid}: unknown decision")
    for triple in triples:
        for field in ("subject", "predicate", "object"):
            if not str(triple[field]).strip():
                raise ValueError(f"{key}/{sid}: empty {field}")


def base_sentence_catalog(
    base_index: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, int, str]]]:
    catalog: dict[str, dict[str, Any]] = {}
    appearances: list[tuple[str, int, str]] = []
    for volume in base_index["volumes"]:
        volume_id = volume["id"]
        for page_number in volume["availablePages"]:
            page = read_json(BASE_DATA_ROOT / "pages" / volume_id / f"{int(page_number):04d}.json")
            for sentence in page["sentences"]:
                sid = sentence["id"]
                existing = catalog.get(sid)
                identity = {
                    "sid": sid,
                    "volumeId": volume_id,
                    "ownerPage": int(sentence["ownerPage"]),
                    "my": sentence["text"],
                    "en": sentence["translation"],
                    "pages": [int(page) for page in sentence["pages"]],
                }
                if existing is not None and existing != identity:
                    raise ValueError(f"{sid}: inconsistent cross-page sentence identity")
                catalog[sid] = identity
                appearances.append((volume_id, int(page_number), sid))
    return catalog, appearances


def load_occurrences(
    catalog: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with SOURCE_PAGES.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            volume_id = f"vol{int(record['volume'])}"
            page_number = int(record["page"])
            key = page_key(volume_id, page_number)
            if record["key"] != key:
                raise ValueError(f"Line {line_number}: expected key {key}")
            source_sentences = record["sentences"]
            decisions = record["response"]["sentences"]
            source_ids = [sentence["sid"] for sentence in source_sentences]
            decision_ids = [sentence["sid"] for sentence in decisions]
            if source_ids != decision_ids:
                raise ValueError(f"{key}: input/output sentence order differs")

            source_by_sid = {sentence["sid"]: sentence for sentence in source_sentences}
            for decision in decisions:
                sid = decision["sid"]
                identity = catalog.get(sid)
                if identity is None:
                    raise ValueError(f"{key}/{sid}: absent from current reader corpus")
                source = source_by_sid[sid]
                if source["my"] != identity["my"]:
                    raise ValueError(f"{key}/{sid}: Burmese sentence differs")
                if source["en"] != identity["en"]:
                    raise ValueError(f"{key}/{sid}: translation differs")
                validate_decision(key, sid, decision)
                occurrences[sid].append(
                    {
                        "key": key,
                        "volumeId": volume_id,
                        "page": page_number,
                        "ownerPage": identity["ownerPage"],
                        "thinkingLevel": record["thinking_level"],
                        "usage": record["usage"],
                        "decision": decision["decision"],
                        "justification": decision["justification"],
                        "triples": decision["triples"],
                    }
                )
    return occurrences


def choose_occurrence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: (
            -len(row["triples"]),
            row["page"] != row["ownerPage"],
            row["page"],
        ),
    )


def occurrence_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": row["key"],
        "page": row["page"],
        "ownerPage": row["ownerPage"],
        "thinkingLevel": row["thinkingLevel"],
        "decision": row["decision"],
        "tripleCount": len(row["triples"]),
    }


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Canonical V3 snapshot already exists: {OUTPUT_ROOT}")
    staging = OUTPUT_ROOT.with_name(f"{OUTPUT_ROOT.name}.staging")
    if staging.exists():
        raise FileExistsError(f"Stale staging directory exists: {staging}")

    report = read_json(SOURCE_REPORT)
    base_index = read_json(BASE_DATA_ROOT / "index.json")
    catalog, page_sentence_appearances = base_sentence_catalog(base_index)
    occurrences = load_occurrences(catalog)
    chosen = {sid: choose_occurrence(rows) for sid, rows in occurrences.items()}

    duplicate_sentence_ids = {sid for sid, rows in occurrences.items() if len(rows) > 1}
    extra_duplicate_appearances = sum(len(occurrences[sid]) - 1 for sid in duplicate_sentence_ids)
    duplicate_different_counts = sum(
        len({len(row["triples"]) for row in occurrences[sid]}) > 1 for sid in duplicate_sentence_ids
    )
    duplicate_different_content = sum(
        len(
            {
                json.dumps(
                    {
                        "decision": row["decision"],
                        "justification": row["justification"],
                        "triples": row["triples"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                for row in occurrences[sid]
            }
        )
        > 1
        for sid in duplicate_sentence_ids
    )

    sentences_by_volume: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    volume_stats: dict[str, Counter[str]] = {
        volume["id"]: Counter() for volume in base_index["volumes"]
    }
    for sid, selected in sorted(chosen.items()):
        identity = catalog[sid]
        volume_id = identity["volumeId"]
        record = {
            **identity,
            "decision": selected["decision"],
            "justification": selected["justification"],
            "triples": selected["triples"],
            "selectedFrom": occurrence_summary(selected),
            "selectionRule": SELECTION_RULE,
            "sourceAppearances": [
                occurrence_summary(row)
                for row in sorted(occurrences[sid], key=lambda row: (row["page"], row["key"]))
            ],
        }
        sentences_by_volume[volume_id][sid] = record
        stats = volume_stats[volume_id]
        stats["canonicalSentences"] += 1
        stats["canonicalTriples"] += len(selected["triples"])
        stats[f"{selected['decision']}Sentences"] += 1
        stats[f"{selected['thinkingLevel']}Sentences"] += 1
        if len(occurrences[sid]) > 1:
            stats["duplicateSentenceIds"] += 1

    page_appearance_triples = 0
    page_appearance_sentences = 0
    pages_with_unavailable: dict[str, set[int]] = defaultdict(set)
    unavailable_sentence_ids = sorted(set(catalog) - set(chosen))
    for volume_id, page_number, sid in page_sentence_appearances:
        selected = chosen.get(sid)
        if selected is None:
            pages_with_unavailable[volume_id].add(page_number)
            continue
        page_appearance_sentences += 1
        page_appearance_triples += len(selected["triples"])
        volume_stats[volume_id]["pageAppearanceSentences"] += 1
        volume_stats[volume_id]["pageAppearanceTriples"] += len(selected["triples"])

    for volume in base_index["volumes"]:
        volume_id = volume["id"]
        write_json(
            staging / "sentences" / f"{volume_id}.json",
            {
                "schemaVersion": 2,
                "volumeId": volume_id,
                "selectionRule": SELECTION_RULE,
                "sentences": sentences_by_volume[volume_id],
            },
        )

    source_sentence_appearances = sum(len(rows) for rows in occurrences.values())
    source_triples = sum(len(row["triples"]) for rows in occurrences.values() for row in rows)
    if source_sentence_appearances != 12035:
        raise ValueError("Unexpected accepted source sentence appearance count")
    if source_triples != int(report["triples"]):
        raise ValueError("Source triple count differs from final report")

    totals = {
        "baseUniqueSentences": len(catalog),
        "baseSentenceAppearances": len(page_sentence_appearances),
        "sourceSentenceAppearances": source_sentence_appearances,
        "sourceTriples": source_triples,
        "canonicalSentences": len(chosen),
        "canonicalTriples": sum(len(selected["triples"]) for selected in chosen.values()),
        "pageAppearanceSentences": page_appearance_sentences,
        "pageAppearanceTriples": page_appearance_triples,
        "duplicateSentenceIds": len(duplicate_sentence_ids),
        "removedDuplicateAppearances": extra_duplicate_appearances,
        "duplicatesWithDifferentTripleCounts": duplicate_different_counts,
        "duplicatesWithDifferentContent": duplicate_different_content,
        "unavailableSentenceIds": len(unavailable_sentence_ids),
    }
    index = {
        "schemaVersion": 2,
        "id": "konbaung_historiography_v3_canonical",
        "description": (
            "One canonical historiography annotation set per sentence ID, "
            "without subject/predicate/object span grounding"
        ),
        "model": report["model"],
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "selectionRule": SELECTION_RULE,
        "source": {
            "pagesJsonl": str(SOURCE_PAGES),
            "pagesJsonlSha256": sha256(SOURCE_PAGES),
            "finalReport": str(SOURCE_REPORT),
            "finalReportSha256": sha256(SOURCE_REPORT),
        },
        "totals": totals,
        "unavailableSentenceIds": unavailable_sentence_ids,
        "pagesWithUnavailableSentences": {
            volume_id: sorted(pages) for volume_id, pages in sorted(pages_with_unavailable.items())
        },
        "volumes": [
            {
                "id": volume["id"],
                **dict(volume_stats[volume["id"]]),
                "pagesWithUnavailableSentences": sorted(pages_with_unavailable[volume["id"]]),
            }
            for volume in base_index["volumes"]
        ],
    }
    write_json(staging / "index.json", index)
    staging.replace(OUTPUT_ROOT)
    print(json.dumps({"output": str(OUTPUT_ROOT), **totals}, indent=2))


if __name__ == "__main__":
    main()

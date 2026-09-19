#!/usr/bin/env python3
"""Build the non-destructive v3 Konbaung dataset with grounded predicates on every triple."""

from __future__ import annotations

import json
import shutil
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "konbaung_dataset_v3_20260718"
SOURCE_SENTENCES = ROOT / "konbaung_sentence_corpus_20260718_restoration_integrated"
SOURCE_TRANSLATIONS = ROOT / "konbaung_sentence_translation_20260718_restoration_integrated"
SOURCE_TRIPLES = ROOT / "konbaung_triples_20260718_restoration_integrated"
ORIGINAL_GROUNDINGS = ROOT / "konbaung_relation_predicate_grounding_full_batch_20260717" / "pages"
GLOSS_CORRECTIONS = (
    ROOT
    / "konbaung_predicate_gloss_repair_full_batch_20260718"
    / "predicate_gloss_corrections.jsonl"
)
PAGE0208_REPAIR = (
    ROOT
    / "konbaung_relation_predicate_grounding_page0208_repair_20260718"
    / "predicate_groundings.jsonl"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n" for value in values
        ),
        encoding="utf-8",
    )


def normalized_with_map(value: str) -> tuple[str, list[int]]:
    characters = []
    mapping = []
    for index, character in enumerate(value):
        category = unicodedata.category(character)
        if character.isspace() or category.startswith("P"):
            continue
        characters.append(character.casefold())
        mapping.append(index)
    return "".join(characters), mapping


def recover_span(needle: str, text: str) -> tuple[str, str] | None:
    if needle in text:
        return needle, "exact"
    normalized_needle, _ = normalized_with_map(needle)
    normalized_text, mapping = normalized_with_map(text)
    if not normalized_needle:
        return None
    start = normalized_text.find(normalized_needle)
    if start < 0 or normalized_text.find(normalized_needle, start + 1) >= 0:
        return None
    return text[mapping[start] : mapping[start + len(normalized_needle) - 1] + 1], "normalized"


def load_original_groundings() -> tuple[
    dict[str, dict[str, str]], dict[str, dict[str, Any]], Counter, list[dict[str, Any]]
]:
    groundings: dict[str, dict[str, str]] = {}
    contexts: dict[str, dict[str, Any]] = {}
    counts = Counter()
    unresolved = []
    for result_path in sorted(ORIGINAL_GROUNDINGS.glob("vol*/page_*/result.json")):
        payload = read_json(result_path.parent / "payload.json")
        stored = read_json(result_path)
        sentence_by_sid = {record["sid"]: record for record in payload["records"]}
        page_sentences = list(sentence_by_sid.values())
        for item in stored.get("response", {}).get("R", []):
            item_id = item["id"]
            sid = item_id.rsplit("_t", 1)[0]
            assigned = sentence_by_sid.get(sid)
            if assigned is None:
                raise ValueError(f"No assigned sentence for {item_id}")
            recovered = recover_span(item["my"], assigned["my"])
            source = "sentence"
            status = "sentence_exact"
            span = item["my"]
            if recovered:
                span, method = recovered
                status = f"sentence_{method}"
            else:
                page_match = None
                for sentence in page_sentences:
                    candidate = recover_span(item["my"], sentence["my"])
                    if candidate:
                        page_match = candidate
                        break
                if page_match:
                    span, method = page_match
                    source = "page"
                    status = f"page_{method}"
                else:
                    source = "inferred"
                    status = "unresolved"
                    unresolved.append(
                        {"id": item_id, "my": item["my"], "page_id": payload["page_id"], "sid": sid}
                    )
            if item_id in groundings:
                raise ValueError(f"Duplicate grounding: {item_id}")
            groundings[item_id] = {"my": span, "en": item["en"], "source": source}
            contexts[item_id] = {"sid": sid, "page_id": payload["page_id"], "alignment": status}
            counts[status] += 1
    return groundings, contexts, counts, unresolved


def main() -> None:
    if V3.exists():
        raise FileExistsError(V3)
    if not PAGE0208_REPAIR.exists():
        raise FileNotFoundError("Page 208 repair is not complete")
    if not GLOSS_CORRECTIONS.exists():
        raise FileNotFoundError("Gloss correction batch is not complete")

    groundings, contexts, alignment_counts, unresolved = load_original_groundings()
    corrections = {item["id"]: item["en"] for item in read_jsonl(GLOSS_CORRECTIONS)}
    for item_id, gloss in corrections.items():
        if item_id in groundings:
            groundings[item_id]["en"] = gloss
    for item in read_jsonl(PAGE0208_REPAIR):
        groundings[item["id"]] = {"my": item["my"], "en": item["en"], "source": "sentence"}
        contexts[item["id"]] = {
            "sid": item["id"].rsplit("_t", 1)[0],
            "page_id": "vol1-p0208",
            "alignment": "sentence_exact_rerun",
        }

    shutil.copytree(SOURCE_SENTENCES, V3 / "sentence_corpus")
    shutil.copytree(SOURCE_TRANSLATIONS, V3 / "sentence_translations")
    shutil.copytree(SOURCE_TRIPLES, V3 / "triples")

    pages = read_jsonl(V3 / "triples" / "annotations" / "all_pages.jsonl")
    output_groundings = []
    missing = []
    total = 0
    restoration_groundings = 0
    for page in pages:
        for group in page.get("S", []):
            sid = group["sid"]
            for index, triple in enumerate(group.get("T", []), start=1):
                total += 1
                triple_id = f"{sid}_t{index:03d}"
                inline = triple.get("predicate_grounding")
                if inline:
                    grounding = {"my": inline["my"], "en": inline["en"], "source": inline["source"]}
                    restoration_groundings += 1
                else:
                    grounding = groundings.get(triple_id)
                if grounding is None:
                    missing.append(triple_id)
                    continue
                triple["predicate_grounding"] = grounding
                output_groundings.append({"id": triple_id, **grounding})
    if missing:
        raise ValueError(f"Missing predicate groundings ({len(missing)}): {missing[:20]}")
    if total != 23440:
        raise ValueError(f"Unexpected v3 triple total: {total}")

    write_jsonl(V3 / "triples" / "annotations" / "all_pages.jsonl", pages)
    for volume in (1, 2, 3):
        write_jsonl(
            V3 / "triples" / "annotations" / f"vol{volume}.jsonl",
            [page for page in pages if int(page["volume"]) == volume],
        )
    output_groundings.sort(key=lambda item: item["id"])
    write_jsonl(V3 / "triples" / "predicate_groundings" / "all.jsonl", output_groundings)
    write_json(
        V3 / "diagnostics" / "predicate_alignment.json",
        {"counts": dict(alignment_counts), "unresolved": unresolved},
    )

    manifest = {
        "version": "v3",
        "created": "2026-07-18",
        "sentences": 11282,
        "triples": total,
        "predicate_groundings": len(output_groundings),
        "restoration_units": 22,
        "restoration_groundings": restoration_groundings,
        "repaired_tag_like_glosses": len(corrections),
        "thinking_tokens_for_gloss_repair": 0,
        "page_0208_groundings_recovered": 104,
        "alignment_counts": dict(alignment_counts),
        "unresolved_original_anchors_retained": len(unresolved),
        "older_datasets_modified": False,
    }
    write_json(V3 / "manifest.json", manifest)
    (V3 / "README.md").write_text(
        "# Konbaung dataset v3\n\nV3 preserves the older datasets and adds conservative chronicle restorations plus `predicate_grounding` (`my`, `en`, `source`) to every triple. Tag-like English glosses were repaired using sentence-only, zero-thinking Gemini batches. Groundings from formerly flagged pages are retained individually; source provenance is recomputed deterministically.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

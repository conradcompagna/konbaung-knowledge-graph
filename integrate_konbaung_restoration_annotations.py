#!/usr/bin/env python3
"""Integrate conservatively corrected restoration sentences and triples into new corpus snapshots."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PACKAGE = Path(r"C:\Users\conra\Downloads\konbaung_restoration_annotations_predicate_spans_20260717.zip")
BASE_SENTENCES = ROOT / "konbaung_sentence_corpus_20260713_repaired"
BASE_TRANSLATIONS = ROOT / "konbaung_sentence_translation_full_batch_20260713_repaired"
BASE_TRIPLES = ROOT / "konbaung_translated_sentence_triples_full_batch_20260713_high_thinking"
LIVE_READER = ROOT / "konbaung_reader_app" / "static" / "data" / "konbaung"
OUT_SENTENCES = ROOT / "konbaung_sentence_corpus_20260718_restoration_integrated"
OUT_TRANSLATIONS = ROOT / "konbaung_sentence_translation_20260718_restoration_integrated"
OUT_TRIPLES = ROOT / "konbaung_triples_20260718_restoration_integrated"
IGNORED_MATCH_CHARS = set("]“”\"?J၀၁၂၃၄၅၆၇၈၉0123456789")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n" for value in values),
        encoding="utf-8",
    )


def load_package() -> list[dict[str, Any]]:
    with zipfile.ZipFile(PACKAGE) as archive:
        name = next(name for name in archive.namelist() if name.endswith("corrected_source_unit_annotations.json"))
        value = json.loads(archive.read(name).decode("utf-8"))
    return list(value["S"])


def relaxed(value: str) -> str:
    return "".join(char for char in value if not char.isspace() and char not in IGNORED_MATCH_CHARS)


def relaxed_with_map(value: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    mapping: list[int] = []
    for index, char in enumerate(value):
        if char.isspace() or char in IGNORED_MATCH_CHARS:
            continue
        chars.append(char)
        mapping.append(index)
    return "".join(chars), mapping


def utf16_offset(value: str, codepoint_offset: int) -> int:
    return len(value[:codepoint_offset].encode("utf-16-le")) // 2


def canonical_text(volume: int, page: int) -> str:
    return str(read_json(LIVE_READER / "pages" / f"vol{volume}" / f"{page:04d}.json")["canonicalText"])


def find_source_piece(volume: int, page: int, needle: str) -> dict[str, Any]:
    page_text = canonical_text(volume, page)
    haystack, mapping = relaxed_with_map(page_text)
    start_normalized = haystack.find(needle)
    if start_normalized < 0:
        raise ValueError(f"Restored text not found on vol{volume} page {page}: {needle[:80]}")
    if haystack.find(needle, start_normalized + 1) >= 0:
        raise ValueError(f"Restored text is ambiguous on vol{volume} page {page}: {needle[:80]}")
    start = mapping[start_normalized]
    end = mapping[start_normalized + len(needle) - 1] + 1
    source_text = page_text[start:end]
    return {
        "page": page,
        "clean_start": None,
        "clean_end": None,
        "start_codepoint": start,
        "end_codepoint": end,
        "start_utf16": utf16_offset(page_text, start),
        "end_utf16": utf16_offset(page_text, end),
        "text": re.sub(r"\s+", " ", source_text).strip(),
        "source_text": source_text,
        "restored_from_removed_text": True,
    }


def find_source_pieces(volume: int, pages: list[int], needle: str) -> list[dict[str, Any]]:
    whole_matches = []
    for page in pages:
        try:
            whole_matches.append(find_source_piece(volume, page, needle))
        except ValueError as error:
            if "not found" not in str(error):
                raise
    if len(whole_matches) == 1:
        return whole_matches
    if len(whole_matches) > 1:
        raise ValueError(f"Restored text occurs on multiple pages: vol{volume} {pages}")
    if len(pages) == 2:
        partitions = []
        for split in range(10, len(needle) - 9):
            try:
                first = find_source_piece(volume, pages[0], needle[:split])
                second = find_source_piece(volume, pages[1], needle[split:])
            except ValueError:
                continue
            partitions.append([first, second])
        if len(partitions) == 1:
            return partitions[0]
        if partitions:
            # Page-boundary OCR can permit several nearby splits within the same
            # word; choose the split that gives the longest first-page fragment.
            return max(partitions, key=lambda parts: len(relaxed(parts[0]["source_text"])))
    raise ValueError(f"Could not map restored text across vol{volume} pages {pages}: {needle[:80]}")


def apply_conservative_tweaks(units: list[dict[str, Any]]) -> None:
    indexed = {unit["sid"]: unit for unit in units}

    plot = indexed["vol1_fr005_s01"]["T"][1]
    plot["p"] = {
        "my": "မသင့်သောအကြံကို ကြံလေသောကြောင့်",
        "en": "formed an improper plot",
        "tag": "FORMS_IMPROPER_PLOT",
        "source": "sentence",
    }
    plot["o"] = {
        "my": "မသင့်သောအကြံ",
        "en": "an improper plot",
        "tag": "DynasticPlot",
        "source": "sentence",
    }

    ritual = indexed["vol2_fr010_s01"]
    ritual["T"].insert(
        0,
        {
            "s": {"my": "မင်းတရားကြီး", "en": "the King", "tag": "RoyalRitualAuthority", "source": "inferred"},
            "p": {"my": "ရထားတင်၍", "en": "placed on a chariot", "tag": "PLACES_ON_CHARIOT", "source": "sentence"},
            "o": {"my": "မဟာပိန္နဲနတ်", "en": "Maha Peinne nat", "tag": "RoyalCultDeity", "source": "sentence"},
        },
    )

    campaign = indexed["vol3_fr041_s01"]
    original = campaign["T"][0]
    official = original["o"]
    campaign["T"] = [
        {
            "s": original["s"],
            "p": {
                "my": "ကိုယ်ရံနောက်ထောက် အစုအမှုထမ်း လူတစ်ထောင်ကျော်ပေး၍",
                "en": "assigned more than one thousand personal-guard and supporting servicemen to",
                "tag": "ASSIGNS_MORE_THAN_ONE_THOUSAND_SERVICEMEN_TO",
                "source": "sentence",
            },
            "o": official,
        },
        {
            "s": original["s"],
            "p": {
                "my": "သိန်းနီကြောင်း ချီတက် ထမ်းရွက်စေသည်",
                "en": "ordered to advance and serve on the Hsenwi route",
                "tag": "ORDERS_CAMPAIGN_SERVICE_ON_HSENWI_ROUTE",
                "source": "sentence",
            },
            "o": official,
        },
    ]


def map_and_build_sentences(
    units: list[dict[str, Any]],
    base_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, dict[str, Any]]]:
    by_volume: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in base_records:
        by_volume[int(record["volume"])].append(record)
    replacements: dict[str, dict[str, Any]] = {}
    source_to_target: dict[str, str] = {}
    integrated_units: dict[str, dict[str, Any]] = {}

    for unit in units:
        volume = int(unit["volume"])
        pages = [int(page) for page in unit["pages"]]
        full_norm = relaxed(unit["my"])
        if unit["integration"]["action"] == "ADD_NEW_SENTENCE":
            if len(pages) != 1:
                raise ValueError(f"Unexpected cross-page new sentence: {unit['sid']}")
            source = [find_source_piece(volume, pages[0], full_norm)]
            target_id = unit["sid"]
            record = {
                "id": target_id,
                "volume": volume,
                "owner_page": pages[0],
                "pages": pages,
                "cross_page": False,
                "text": unit["my"],
                "source": source,
                "restoration_source": unit["sid"],
            }
        else:
            candidates = []
            for candidate in by_volume[volume]:
                candidate_norm = relaxed(candidate["text"])
                if len(candidate_norm) < 20 or not full_norm.endswith(candidate_norm):
                    continue
                if not set(int(page) for page in candidate["pages"]).intersection(pages):
                    continue
                candidates.append(candidate)
            if len(candidates) != 1:
                raise ValueError(f"Expected one current repair target for {unit['sid']}; got {[item['id'] for item in candidates]}")
            old = candidates[0]
            target_id = old["id"]
            old_norm = relaxed(old["text"])
            prefix_norm = full_norm[: -len(old_norm)]
            prefix_pieces = []
            if prefix_norm:
                prefix_pieces = find_source_pieces(volume, pages, prefix_norm)
            source = prefix_pieces + list(old["source"])
            source.sort(key=lambda item: (int(item["page"]), int(item["start_codepoint"])))
            record = dict(old)
            record.update(
                {
                    "owner_page": pages[0],
                    "pages": pages,
                    "cross_page": len(pages) > 1,
                    "text": unit["my"],
                    "source": source,
                    "restoration_source": unit["sid"],
                }
            )
        if target_id in replacements:
            raise ValueError(f"Duplicate integrated target: {target_id}")
        replacements[target_id] = record
        source_to_target[unit["sid"]] = target_id
        integrated_units[target_id] = unit

    output = [replacements.get(record["id"], record) for record in base_records]
    existing_ids = {record["id"] for record in base_records}
    output.extend(record for target_id, record in replacements.items() if target_id not in existing_ids)
    output.sort(key=lambda item: (int(item["volume"]), int(item["owner_page"]), int(item["source"][0]["start_codepoint"]), item["id"]))
    return output, source_to_target, integrated_units


def validate_units(units_by_target: dict[str, dict[str, Any]], sentences: dict[str, dict[str, Any]]) -> None:
    for target, unit in units_by_target.items():
        sentence = sentences[target]["text"]
        for index, triple in enumerate(unit["T"], start=1):
            for role in ("s", "p", "o"):
                endpoint = triple[role]
                if endpoint["source"] == "sentence" and endpoint["my"] not in sentence:
                    raise ValueError(f"{target} T{index} {role}: exact span not in restored sentence: {endpoint['my']}")


def update_sentence_root(records: list[dict[str, Any]]) -> None:
    shutil.copytree(BASE_SENTENCES, OUT_SENTENCES)
    for volume in (1, 2, 3):
        volume_records = [record for record in records if int(record["volume"]) == volume]
        write_jsonl(OUT_SENTENCES / "sentences" / f"vol{volume}.jsonl", volume_records)
        write_jsonl(
            OUT_SENTENCES / "cross_page_sentences" / f"vol{volume}.jsonl",
            [record for record in volume_records if record["cross_page"]],
        )
    write_jsonl(OUT_SENTENCES / "sentences" / "all_volumes.jsonl", records)

    by_page: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for source in record["source"]:
            by_page[(int(record["volume"]), int(source["page"]))].append(record)
    for path in OUT_SENTENCES.glob("page_records/vol*/page_*.json"):
        page_record = read_json(path)
        key = (int(page_record["volume"]), int(page_record["page"]))
        page_sentences = sorted(
            {record["id"]: record for record in by_page[key]}.values(),
            key=lambda record: min(
                int(source["start_codepoint"])
                for source in record["source"]
                if int(source["page"]) == key[1]
            ),
        )
        page_record["sentence_ids"] = [record["id"] for record in page_sentences]
        page_record["owned_sentence_ids"] = [record["id"] for record in page_sentences if int(record["owner_page"]) == key[1]]
        page_record["cross_page_sentence_ids"] = [record["id"] for record in page_sentences if record["cross_page"]]
        write_json(path, page_record)

    manifest = read_json(OUT_SENTENCES / "manifest.json")
    manifest["output_root"] = str(OUT_SENTENCES)
    manifest["integration"] = {
        "date": "2026-07-18",
        "source_package": PACKAGE.name,
        "new_sentence_units": 11,
        "repaired_sentence_units": 11,
        "policy": "chronicle-only conservative restoration",
    }
    for volume_record in manifest["volumes"]:
        volume = int(volume_record["volume"])
        subset = [record for record in records if int(record["volume"]) == volume]
        volume_record["sentence_count"] = len(subset)
        volume_record["cross_page_sentence_count"] = sum(bool(record["cross_page"]) for record in subset)
    manifest["totals"]["sentence_count"] = len(records)
    manifest["totals"]["cross_page_sentence_count"] = sum(bool(record["cross_page"]) for record in records)
    write_json(OUT_SENTENCES / "manifest.json", manifest)
    (OUT_SENTENCES / "README_RESTORATION_INTEGRATION.md").write_text(
        "# Restoration-integrated sentence corpus\n\nDerived from `konbaung_sentence_corpus_20260713_repaired`; originals are unchanged. Eleven chronicle sentences were restored and eleven cross-page sentence records were repaired using exact canonical-page offsets.\n",
        encoding="utf-8",
    )


def update_translation_root(
    sentence_records: list[dict[str, Any]],
    units_by_target: dict[str, dict[str, Any]],
) -> None:
    shutil.copytree(BASE_TRANSLATIONS, OUT_TRANSLATIONS)
    base = {record["id"]: record for record in read_jsonl(BASE_TRANSLATIONS / "translations" / "all_volumes.jsonl")}
    sentence_by_id = {record["id"]: record for record in sentence_records}
    for target, unit in units_by_target.items():
        sentence = sentence_by_id[target]
        record = dict(base.get(target, {}))
        record.update(
            {
                "id": target,
                "volume": sentence["volume"],
                "owner_page": sentence["owner_page"],
                "pages": sentence["pages"],
                "cross_page": sentence["cross_page"],
                "my": sentence["text"],
                "en": unit["en"],
                "restoration_source": unit["sid"],
            }
        )
        base[target] = record
    records = sorted(base.values(), key=lambda item: (int(item["volume"]), int(item["owner_page"]), item["id"]))
    write_jsonl(OUT_TRANSLATIONS / "translations" / "all_volumes.jsonl", records)
    for volume in (1, 2, 3):
        write_jsonl(OUT_TRANSLATIONS / "translations" / f"vol{volume}.jsonl", [item for item in records if int(item["volume"]) == volume])


def convert_triples(unit: dict[str, Any]) -> list[dict[str, Any]]:
    converted = []
    for triple in unit["T"]:
        predicate = triple["p"]
        converted.append(
            {
                "s": triple["s"],
                "p": predicate["tag"],
                "o": triple["o"],
                "predicate_grounding": {
                    "my": predicate["my"],
                    "en": predicate["en"],
                    "source": predicate["source"],
                },
            }
        )
    return converted


def update_triple_root(
    units_by_target: dict[str, dict[str, Any]],
    sentences: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    OUT_TRIPLES.mkdir(parents=True)
    shutil.copytree(BASE_TRIPLES / "annotations", OUT_TRIPLES / "annotations")
    pages = read_jsonl(BASE_TRIPLES / "annotations" / "all_pages.jsonl")
    page_by_key = {(int(page["volume"]), int(page["page"])): page for page in pages}
    affected = set(units_by_target)
    old_count = 0
    for page in pages:
        kept = []
        for group in page.get("S", []):
            if group["sid"] in affected:
                old_count += len(group.get("T", []))
            else:
                kept.append(group)
        page["S"] = kept
    for target, unit in units_by_target.items():
        sentence = sentences[target]
        key = (int(sentence["volume"]), int(sentence["owner_page"]))
        if key not in page_by_key:
            page = {"page_id": f"vol{key[0]}-p{key[1]:04d}", "volume": key[0], "page": key[1], "summary": "", "S": [], "provenance": "manual_restoration"}
            pages.append(page)
            page_by_key[key] = page
        page_by_key[key]["S"].append(
            {
                "sid": target,
                "T": convert_triples(unit),
                "provenance": "manual_chronicle_restoration_20260718",
                "restoration_source": unit["sid"],
                "formulaic_group": unit.get("formulaic_group"),
            }
        )
    pages.sort(key=lambda item: (int(item["volume"]), int(item["page"])))
    for page in pages:
        page["S"].sort(key=lambda group: group["sid"])
    write_jsonl(OUT_TRIPLES / "annotations" / "all_pages.jsonl", pages)
    for volume in (1, 2, 3):
        write_jsonl(OUT_TRIPLES / "annotations" / f"vol{volume}.jsonl", [page for page in pages if int(page["volume"]) == volume])

    grounding_rows = []
    for target, unit in units_by_target.items():
        for index, triple in enumerate(unit["T"], start=1):
            grounding_rows.append(
                {
                    "id": f"{target}_t{index:03d}",
                    "my": triple["p"]["my"],
                    "en": triple["p"]["en"],
                    "src": triple["p"]["source"],
                    "restoration_source": unit["sid"],
                }
            )
    write_jsonl(OUT_TRIPLES / "predicate_groundings" / "restoration.jsonl", grounding_rows)
    new_count = sum(len(group.get("T", [])) for page in pages for group in page.get("S", []))
    return old_count, new_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Create the new integrated snapshots.")
    args = parser.parse_args()
    if not PACKAGE.exists():
        raise FileNotFoundError(PACKAGE)
    units = load_package()
    apply_conservative_tweaks(units)
    base_records = read_jsonl(BASE_SENTENCES / "sentences" / "all_volumes.jsonl")
    records, source_to_target, units_by_target = map_and_build_sentences(units, base_records)
    sentences = {record["id"]: record for record in records}
    validate_units(units_by_target, sentences)
    report = {
        "package_units": len(units),
        "new_sentences": sum(unit["integration"]["action"] == "ADD_NEW_SENTENCE" for unit in units),
        "repaired_sentences": sum(unit["integration"]["action"] == "REPAIR_EXISTING_SENTENCE" for unit in units),
        "restoration_triples_after_tweaks": sum(len(unit["T"]) for unit in units),
        "source_to_current_target": source_to_target,
        "output_sentence_count": len(records),
    }
    if not args.apply:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    for path in (OUT_SENTENCES, OUT_TRANSLATIONS, OUT_TRIPLES):
        if path.exists():
            raise FileExistsError(path)
    update_sentence_root(records)
    update_translation_root(records, units_by_target)
    old_replaced_count, total_triples = update_triple_root(units_by_target, sentences)
    report.update(
        {
            "old_triples_replaced": old_replaced_count,
            "integrated_total_triples": total_triples,
            "sentence_root": str(OUT_SENTENCES),
            "translation_root": str(OUT_TRANSLATIONS),
            "triple_root": str(OUT_TRIPLES),
        }
    )
    write_json(OUT_TRIPLES / "integration_report.json", report)
    provenance = OUT_TRIPLES / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PACKAGE, provenance / PACKAGE.name)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

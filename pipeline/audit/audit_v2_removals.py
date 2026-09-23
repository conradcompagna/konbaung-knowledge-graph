#!/usr/bin/env python3
"""Create an exhaustive, copy-safe audit of text removed from Konbaung corpus v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import build_konbaung_sentence_corpus as base
import repair_konbaung_sentence_corpus as repaired


CORPUS_ROOT = Path("konbaung_sentence_corpus_20260713_repaired")
SOURCE_ARCHIVE = Path(
    "konbaung_reader_app/data_archives/konbaung_reader_pre_sentence_annotations_20260713/pages"
)
DEFAULT_OUTPUT = Path("konbaung_v2_removal_audit_20260715")
REPORTED_CASE = (3, 50)

REASON_DESCRIPTIONS = {
    "editorial_footnote": (
        "Automatic tail-block rule. In the last 14 nonempty lines before a detected publisher footer, "
        "the builder looks for an asterisk, editor marker, numbered/date/Latin signal, or a date in the "
        "last four lines. It then removes every line from the earliest inferred start through the footer "
        "boundary. This blanket range is the rule that incorrectly removed body text on vol3 page 50."
    ),
    "editorial_footnote_fragment": (
        "Automatic last-four-lines rule for a short numbered numeric fragment."
    ),
    "publisher_footer": (
        "Automatic publisher-line rule. A short line resembling the publisher name is detected among "
        "the last ten nonempty lines, and that line plus everything after it is removed."
    ),
    "page_number": "Automatic rule for an all-numeric line among the first three nonempty lines.",
    "page_number_ocr": (
        "Automatic rule for a short numeric/ASCII-like OCR page-number line among the first three "
        "nonempty lines when a running header is also present."
    ),
    "running_header": (
        "Automatic header-regex rule applied to the first three nonempty lines, including short lines "
        "between multiple header matches."
    ),
    "pre_header_debris": (
        "Automatic rule that removes a top-three nonempty line occurring before the first detected header."
    ),
    "trailing_numeric_debris": (
        "Automatic last-four-lines rule for a bare numeric string of eight characters or fewer."
    ),
    "trailing_ocr_debris": (
        "Automatic fallback tail rule used when no publisher footer is detected; repeatedly removes "
        "short punctuation/ASCII/non-Burmese junk from the end."
    ),
    "non_burmese_debris": (
        "Automatic whole-line rule for text containing Latin letters but no character in the Myanmar block."
    ),
    "duplicate_sentence_terminator": (
        "Automatic character rule that removes U+104B MYANMAR SIGN SECTION when the preceding nonspace "
        "character is already U+104B."
    ),
    "orphan_closing_bracket": (
        "Automatic character rule that removes a closing square bracket immediately following U+104B."
    ),
    "manual_running_header": "Explicit v2 audit allowlist entry for a confirmed running header.",
    "manual_page_number": "Explicit v2 audit allowlist entry for a confirmed printed page number.",
    "final_unterminated_caption": (
        "Automatic volume-end rule that removes an unterminated tail of 120 characters or fewer after "
        "the last U+104B on the volume's final included page."
    ),
    "excluded_duplicate_page": (
        "Explicit v2 allowlist exclusion of an entire OCR scan judged to duplicate the immediately "
        "preceding page."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--source-archive", type=Path, default=SOURCE_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--zip", type=Path, default=None, dest="zip_path")
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def write_text_exact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_json(path: Path, payload: Any) -> None:
    write_text_lf(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def workspace_relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_source_pages(source_root: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], list[str]]:
    pages: dict[tuple[int, int], dict[str, Any]] = {}
    errors: list[str] = []
    for volume in (1, 2, 3):
        for path in sorted((source_root / f"vol{volume}").glob("*.json")):
            raw = path.read_bytes()
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception as exc:  # pragma: no cover - audit error path
                errors.append(f"Cannot parse source JSON {path}: {exc}")
                continue
            page = int(payload.get("pageNumber", path.stem))
            key = (volume, page)
            if key in pages:
                errors.append(f"Duplicate source key vol{volume} page {page}")
                continue
            text = payload.get("canonicalText")
            if not isinstance(text, str):
                errors.append(f"Missing canonicalText in {path}")
                continue
            pages[key] = {
                "volume": volume,
                "page": page,
                "path": path,
                "raw_sha256": sha256_bytes(raw),
                "canonical_text": text,
                "canonical_text_sha256": sha256_text(text),
            }
    return pages, errors


def load_page_records(corpus: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], list[str]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    errors: list[str] = []
    for volume in (1, 2, 3):
        for path in sorted((corpus / "page_records" / f"vol{volume}").glob("page_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - audit error path
                errors.append(f"Cannot parse v2 page record {path}: {exc}")
                continue
            page = int(payload["page"])
            key = (volume, page)
            if key in records:
                errors.append(f"Duplicate v2 page record vol{volume} page {page}")
                continue
            records[key] = {"path": path, "payload": payload}
    return records, errors


def check_item(checks: list[dict[str, Any]], name: str, errors: list[str], details: str) -> None:
    checks.append(
        {
            "name": name,
            "status": "passed" if not errors else "failed",
            "details": details,
            "errors": errors[:100],
            "error_count": len(errors),
        }
    )


def removal_record(
    *,
    volume: int,
    page: int,
    ordinal: int,
    item: dict[str, Any],
    source: dict[str, Any],
    page_record_rel: str | None,
    duplicate_of: int | None = None,
) -> dict[str, Any]:
    text = item["text"]
    start = int(item["start_codepoint"])
    end = int(item["end_codepoint"])
    reason = item["reason"]
    return {
        "id": f"vol{volume}-p{page:04d}-r{ordinal:03d}",
        "volume": volume,
        "page": page,
        "source_line": int(item["line"]),
        "reason": reason,
        "reason_description": REASON_DESCRIPTIONS.get(reason, "Undocumented removal reason."),
        "start_codepoint": start,
        "end_codepoint": end,
        "removed_codepoints": end - start,
        "removed_non_whitespace_codepoints": sum(not char.isspace() for char in text),
        "removed_text": text,
        "duplicate_of_page": duplicate_of,
        "original_ocr_page": f"original_ocr_pages/vol{volume}/page_{page:04d}.txt",
        "v2_page_record": page_record_rel,
        "source_json_sha256": source["raw_sha256"],
        "canonical_text_sha256": source["canonical_text_sha256"],
    }


def render_reason_markdown(reason: str, records: list[dict[str, Any]]) -> str:
    page_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        page_groups[(record["volume"], record["page"])].append(record)
    total_codepoints = sum(record["removed_codepoints"] for record in records)
    total_nonspace = sum(record["removed_non_whitespace_codepoints"] for record in records)
    lines = [
        f"# Removed as `{reason}`",
        "",
        REASON_DESCRIPTIONS.get(reason, "Undocumented removal reason."),
        "",
        f"Records: {len(records)}  ",
        f"Pages: {len(page_groups)}  ",
        f"Removed codepoints (excluding line endings): {total_codepoints}  ",
        f"Removed non-whitespace codepoints: {total_nonspace}",
        "",
        "Each block below is copied exactly from the archived OCR input. Offsets use Python/Unicode "
        "codepoint indexing and half-open ranges `[start, end)`.",
        "",
    ]
    for (volume, page), page_records in sorted(page_groups.items()):
        lines.extend(
            [
                f"## Volume {volume}, page {page}",
                "",
                f"Original OCR page: [page_{page:04d}.txt](../original_ocr_pages/vol{volume}/page_{page:04d}.txt)",
            ]
        )
        if reason != "excluded_duplicate_page":
            lines.append(
                f"V2 page record: [page_{page:04d}.json](../v2_page_records/vol{volume}/page_{page:04d}.json)"
            )
        lines.append("")
        for record in page_records:
            duplicate_note = ""
            if record.get("duplicate_of_page") is not None:
                duplicate_note = f"; duplicate of page {record['duplicate_of_page']}"
            lines.extend(
                [
                    f"### {record['id']}",
                    "",
                    f"Source line {record['source_line']}; codepoints "
                    f"`[{record['start_codepoint']}, {record['end_codepoint']})`{duplicate_note}",
                    "",
                    "~~~text",
                    record["removed_text"],
                    "~~~",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def render_known_case(source_text: str, records: list[dict[str, Any]]) -> str:
    source_lines = source_text.splitlines()
    relevant = []
    for number in range(20, min(38, len(source_lines)) + 1):
        label = ""
        if 26 <= number <= 33:
            label = "  <<< CONFIRMED BODY TEXT REMOVED"
        elif 34 <= number <= 37:
            label = "  <<< DATE-CONVERSION FOOTNOTE"
        elif number == 38:
            label = "  <<< PUBLISHER FOOTER"
        relevant.append(f"{number:02d}: {source_lines[number - 1]}{label}")
    page_removals = [
        {
            key: record[key]
            for key in (
                "id",
                "source_line",
                "reason",
                "start_codepoint",
                "end_codepoint",
                "removed_text",
            )
        }
        for record in records
    ]
    return (
        "# Confirmed reported failure: Volume 3, page 50\n\n"
        "The v2 builder classified source lines 25-37 as `editorial_footnote`. The user's review "
        "confirmed that lines 26-33 are still main chronicle content; the Gregorian date-conversion "
        "footnotes begin at line 34. Therefore the blanket footnote range cut eight lines of body text, "
        "including the `၁၂၀၇- ခု` year heading and the resumed narrative.\n\n"
        "Original OCR page: [page_0050.txt](../original_ocr_pages/vol3/page_0050.txt)  \n"
        "V2 page record: [page_0050.json](../v2_page_records/vol3/page_0050.json)  \n"
        "Exact archived source JSON sample: [0050.json](../original_source_json_samples/vol3/0050.json)\n\n"
        "## Numbered source excerpt\n\n~~~text\n"
        + "\n".join(relevant)
        + "\n~~~\n\n## V2 removal records on this page\n\n~~~json\n"
        + json.dumps(page_removals, ensure_ascii=False, indent=2)
        + "\n~~~\n"
    )


def render_high_risk_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# Editorial-tail removals with prose before a date",
        "",
        "These are heuristic review candidates, not confirmed errors. A page is listed when the "
        "`editorial_footnote` block contains a Burmese prose-like line before its first detected Gregorian "
        "date line. This specifically tests the failure shape seen on Volume 3 page 50.",
        "",
        f"Candidate pages: {len(records)}",
        "",
    ]
    for record in records:
        volume = record["volume"]
        page = record["page"]
        lines.extend(
            [
                f"## Volume {volume}, page {page}",
                "",
                f"Original OCR page: [page_{page:04d}.txt](../original_ocr_pages/vol{volume}/page_{page:04d}.txt)  ",
                f"First detected date line inside removed block: {record['first_date_line']}  ",
                f"Earlier prose-like removed lines: {', '.join(map(str, record['prose_lines_before_date']))}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    workspace = Path.cwd().resolve()
    corpus = args.corpus.resolve()
    source_root = args.source_archive.resolve()
    output = args.output.resolve()
    zip_path = (args.zip_path or output.with_suffix(".zip")).resolve()

    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing audit directory: {output}")
    if zip_path.exists():
        raise SystemExit(f"Refusing to overwrite existing ZIP archive: {zip_path}")
    if not corpus.is_dir():
        raise SystemExit(f"Missing v2 corpus: {corpus}")
    if not source_root.is_dir():
        raise SystemExit(f"Missing archived OCR source: {source_root}")

    output.mkdir(parents=True)
    checks: list[dict[str, Any]] = []
    fatal_errors: list[str] = []

    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    repair_manifest = json.loads((corpus / "repair_manifest.json").read_text(encoding="utf-8"))
    sources, source_errors = load_source_pages(source_root)
    page_records, page_record_errors = load_page_records(corpus)
    fatal_errors.extend(source_errors)
    fatal_errors.extend(page_record_errors)

    expected_source_pages = int(repair_manifest["totals"]["source_pages"])
    errors = []
    if len(sources) != expected_source_pages:
        errors.append(f"Expected {expected_source_pages} archived pages, found {len(sources)}")
    check_item(
        checks,
        "archived_source_page_count",
        errors,
        f"Found {len(sources)} archived page JSON files; v2 repair manifest expects {expected_source_pages}.",
    )
    fatal_errors.extend(errors)

    expected_record_pages = int(repair_manifest["totals"]["annotation_pages"])
    errors = []
    if len(page_records) != expected_record_pages:
        errors.append(f"Expected {expected_record_pages} page records, found {len(page_records)}")
    check_item(
        checks,
        "v2_page_record_count",
        errors,
        f"Found {len(page_records)} v2 page records; v2 repair manifest expects {expected_record_pages}.",
    )
    fatal_errors.extend(errors)

    exclusions = {
        (int(item["volume"]), int(item["page"])): item
        for item in repair_manifest["excluded_duplicate_pages"]
    }
    errors = []
    combined_keys = set(page_records) | set(exclusions)
    if combined_keys != set(sources):
        missing = sorted(set(sources) - combined_keys)
        extra = sorted(combined_keys - set(sources))
        errors.extend([f"Unaccounted source pages: {missing}", f"Unknown accounted pages: {extra}"])
    if set(page_records) & set(exclusions):
        errors.append(
            f"Excluded pages also have v2 page records: {sorted(set(page_records) & set(exclusions))}"
        )
    check_item(
        checks,
        "every_source_page_accounted_for",
        errors,
        f"{len(page_records)} included pages plus {len(exclusions)} excluded duplicate pages account for the archive.",
    )
    fatal_errors.extend(errors)

    all_removals: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    reason_pages: dict[str, set[tuple[int, int]]] = defaultdict(set)
    reason_codepoints: Counter[str] = Counter()
    reason_nonspace: Counter[str] = Counter()
    hash_errors: list[str] = []
    span_errors: list[str] = []
    rebuild_errors: list[str] = []
    coverage_errors: list[str] = []

    for key in sorted(sources):
        volume, page = key
        source = sources[key]
        text = source["canonical_text"]
        original_page_rel = f"original_ocr_pages/vol{volume}/page_{page:04d}.txt"
        write_text_exact(output / original_page_rel, text)

        recorded_hash = None
        page_record_rel = None
        if key in page_records:
            info = page_records[key]
            payload = info["payload"]
            recorded_hash = payload["source_sha256"]
            page_record_rel = f"v2_page_records/vol{volume}/page_{page:04d}.json"
            destination = output / page_record_rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(info["path"], destination)

            if recorded_hash != source["raw_sha256"]:
                hash_errors.append(
                    f"vol{volume} page {page}: page-record source hash {recorded_hash} != archive {source['raw_sha256']}"
                )
            if int(payload["original_character_count"]) != len(text):
                span_errors.append(
                    f"vol{volume} page {page}: original_character_count mismatch "
                    f"({payload['original_character_count']} vs {len(text)})"
                )

            try:
                lookup = {repaired.text_key(text): key}
                rebuilt_cleaned, source_positions, rebuilt_removals, _ = (
                    repaired.cleaned_page_with_repairs(text, lookup)
                )
            except Exception as exc:  # pragma: no cover - audit error path
                rebuild_errors.append(f"vol{volume} page {page}: cleaner rerun failed: {exc}")
                rebuilt_cleaned, source_positions, rebuilt_removals = "", [], []

            record_removals = list(payload["removals"])
            final_items = [
                item for item in record_removals if item["reason"] == "final_unterminated_caption"
            ]
            nonfinal_items = [
                item for item in record_removals if item["reason"] != "final_unterminated_caption"
            ]
            if rebuilt_removals != nonfinal_items:
                rebuild_errors.append(
                    f"vol{volume} page {page}: rerun removal records differ from v2 record"
                )
            if final_items:
                if not rebuilt_cleaned.startswith(payload["cleaned_text"]):
                    rebuild_errors.append(
                        f"vol{volume} page {page}: v2 final-caption cleaned text is not a rerun prefix"
                    )
                source_positions = source_positions[: len(payload["cleaned_text"])]
            elif rebuilt_cleaned != payload["cleaned_text"]:
                rebuild_errors.append(
                    f"vol{volume} page {page}: rerun cleaned text differs from v2 record"
                )

            intervals: list[tuple[int, int, str]] = []
            for ordinal, item in enumerate(record_removals, start=1):
                start = int(item["start_codepoint"])
                end = int(item["end_codepoint"])
                exact = text[start:end] if 0 <= start <= end <= len(text) else None
                if exact != item["text"]:
                    span_errors.append(
                        f"vol{volume} page {page} removal {ordinal}: source slice mismatch or invalid range"
                    )
                calculated_line = 1 + text.count("\n", 0, max(0, start))
                if calculated_line != int(item["line"]):
                    span_errors.append(
                        f"vol{volume} page {page} removal {ordinal}: line {item['line']} != calculated {calculated_line}"
                    )
                intervals.append((start, end, item["reason"]))
                record = removal_record(
                    volume=volume,
                    page=page,
                    ordinal=ordinal,
                    item=item,
                    source=source,
                    page_record_rel=page_record_rel,
                )
                all_removals.append(record)
                reason_counts[record["reason"]] += 1
                reason_pages[record["reason"]].add(key)
                reason_codepoints[record["reason"]] += record["removed_codepoints"]
                reason_nonspace[record["reason"]] += record["removed_non_whitespace_codepoints"]

            for (start_a, end_a, reason_a), (start_b, end_b, reason_b) in zip(
                sorted(intervals), sorted(intervals)[1:]
            ):
                if end_a > start_b:
                    span_errors.append(
                        f"vol{volume} page {page}: overlapping removals {reason_a} [{start_a},{end_a}) "
                        f"and {reason_b} [{start_b},{end_b})"
                    )

            if len(source_positions) == len(payload["cleaned_text"]):
                retained_nonspace = {
                    position
                    for char, position in zip(payload["cleaned_text"], source_positions)
                    if not char.isspace()
                }
                removed_nonspace = {
                    position
                    for item in record_removals
                    for position in range(int(item["start_codepoint"]), int(item["end_codepoint"]))
                    if 0 <= position < len(text) and not text[position].isspace()
                }
                source_nonspace = {
                    position for position, char in enumerate(text) if not char.isspace()
                }
                overlap = retained_nonspace & removed_nonspace
                missing = source_nonspace - retained_nonspace - removed_nonspace
                extra = (retained_nonspace | removed_nonspace) - source_nonspace
                if overlap or missing or extra:
                    coverage_errors.append(
                        f"vol{volume} page {page}: overlap={len(overlap)}, missing={len(missing)}, extra={len(extra)}"
                    )
            else:
                coverage_errors.append(
                    f"vol{volume} page {page}: source mapping length {len(source_positions)} != cleaned length "
                    f"{len(payload['cleaned_text'])}"
                )
        else:
            excluded = exclusions[key]
            recorded_hash = excluded["source_sha256"]
            if recorded_hash != source["raw_sha256"]:
                hash_errors.append(
                    f"vol{volume} page {page}: exclusion source hash {recorded_hash} != archive {source['raw_sha256']}"
                )
            item = {
                "line": 1,
                "start_codepoint": 0,
                "end_codepoint": len(text),
                "reason": "excluded_duplicate_page",
                "text": text,
            }
            record = removal_record(
                volume=volume,
                page=page,
                ordinal=1,
                item=item,
                source=source,
                page_record_rel=None,
                duplicate_of=int(excluded["duplicate_of"]),
            )
            all_removals.append(record)
            reason_counts[record["reason"]] += 1
            reason_pages[record["reason"]].add(key)
            reason_codepoints[record["reason"]] += record["removed_codepoints"]
            reason_nonspace[record["reason"]] += record["removed_non_whitespace_codepoints"]

        source_manifest.append(
            {
                "volume": volume,
                "page": page,
                "included_in_v2": key in page_records,
                "excluded_duplicate_of": exclusions.get(key, {}).get("duplicate_of"),
                "archived_source_json": workspace_relative(source["path"], workspace),
                "archived_source_json_sha256": source["raw_sha256"],
                "v2_recorded_source_json_sha256": recorded_hash,
                "source_hash_match": recorded_hash == source["raw_sha256"],
                "canonical_text_sha256": source["canonical_text_sha256"],
                "canonical_text_codepoints": len(text),
                "bundle_ocr_page": original_page_rel,
            }
        )

    check_item(
        checks,
        "archived_source_hashes_match_v2",
        hash_errors,
        f"Checked SHA-256 provenance for {len(sources)} archived source JSON files.",
    )
    check_item(
        checks,
        "every_recorded_removal_is_an_exact_source_slice",
        span_errors,
        f"Checked {sum(reason_counts.values()) - len(exclusions)} v2 removal records for exact text, offsets, lines, and overlap.",
    )
    check_item(
        checks,
        "v2_cleaner_rerun_matches_page_records",
        rebuild_errors,
        f"Re-ran the v2 cleaning function on {len(page_records)} exact archived page inputs.",
    )
    check_item(
        checks,
        "all_non_whitespace_source_characters_are_retained_or_recorded_removed",
        coverage_errors,
        "Independently partitioned every included page's non-whitespace source positions into retained versus removed.",
    )
    fatal_errors.extend(hash_errors)
    fatal_errors.extend(span_errors)
    fatal_errors.extend(rebuild_errors)
    fatal_errors.extend(coverage_errors)

    expected_reason_counts: Counter[str] = Counter()
    for volume_manifest in manifest["volumes"]:
        expected_reason_counts.update(volume_manifest["removal_counts"])
    actual_v2_reason_counts = reason_counts.copy()
    actual_v2_reason_counts.pop("excluded_duplicate_page", None)
    errors = []
    if actual_v2_reason_counts != expected_reason_counts:
        errors.append(
            f"Manifest counts {dict(expected_reason_counts)} != page-record counts {dict(actual_v2_reason_counts)}"
        )
    check_item(
        checks,
        "removal_reason_counts_match_v2_manifest",
        errors,
        f"Matched {sum(actual_v2_reason_counts.values())} page-record removals across {len(actual_v2_reason_counts)} reasons.",
    )
    fatal_errors.extend(errors)

    reported_records = [
        record for record in all_removals if (record["volume"], record["page"]) == REPORTED_CASE
    ]
    reported_by_line = {record["source_line"]: record for record in reported_records}
    errors = []
    for line in range(26, 34):
        record = reported_by_line.get(line)
        if not record or record["reason"] != "editorial_footnote":
            errors.append(f"Expected vol3 page 50 line {line} to be removed as editorial_footnote")
    check_item(
        checks,
        "reported_vol3_page50_failure_reproduced",
        errors,
        "Confirmed that v2 removed source lines 26-33, which the user identified as main chronicle content.",
    )
    fatal_errors.extend(errors)

    all_removals.sort(
        key=lambda item: (item["volume"], item["page"], item["start_codepoint"], item["id"])
    )
    write_jsonl(output / "all_removed_spans.jsonl", all_removals)
    with (output / "all_removed_spans.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "id",
                "volume",
                "page",
                "source_line",
                "reason",
                "start_codepoint",
                "end_codepoint",
                "removed_codepoints",
                "removed_non_whitespace_codepoints",
                "duplicate_of_page",
                "original_ocr_page",
                "removed_text_escaped",
            ]
        )
        for record in all_removals:
            escaped = (
                record["removed_text"]
                .replace("\\", "\\\\")
                .replace("\r", "\\r")
                .replace("\n", "\\n")
                .replace("\t", "\\t")
            )
            writer.writerow(
                [
                    record["id"],
                    record["volume"],
                    record["page"],
                    record["source_line"],
                    record["reason"],
                    record["start_codepoint"],
                    record["end_codepoint"],
                    record["removed_codepoints"],
                    record["removed_non_whitespace_codepoints"],
                    record["duplicate_of_page"],
                    record["original_ocr_page"],
                    escaped,
                ]
            )

    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in all_removals:
        by_reason[record["reason"]].append(record)
    for reason, records in sorted(by_reason.items()):
        write_jsonl(output / "by_reason" / f"{reason}.jsonl", records)
        write_text_lf(
            output / "by_reason" / f"{reason}.md",
            render_reason_markdown(reason, records),
        )

    reason_summary = []
    for reason in sorted(reason_counts):
        reason_summary.append(
            {
                "reason": reason,
                "records": reason_counts[reason],
                "pages": len(reason_pages[reason]),
                "removed_codepoints": reason_codepoints[reason],
                "removed_non_whitespace_codepoints": reason_nonspace[reason],
                "description": REASON_DESCRIPTIONS.get(reason, "Undocumented removal reason."),
            }
        )
    with (output / "reason_summary.csv").open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reason_summary[0]))
        writer.writeheader()
        writer.writerows(reason_summary)

    write_jsonl(output / "source_pages_manifest.jsonl", source_manifest)

    # Preserve exact build inputs for the reported error and the four whole-page exclusions.
    sample_keys = {REPORTED_CASE, *exclusions.keys()}
    for volume, page in sorted(sample_keys):
        source = sources[(volume, page)]
        destination = output / "original_source_json_samples" / f"vol{volume}" / f"{page:04d}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source["path"], destination)

    provenance_files = [
        corpus / "README.md",
        corpus / "manifest.json",
        corpus / "repair_manifest.json",
        corpus / "review" / "removed_lines.tsv",
        corpus / "review" / "excluded_duplicate_pages.tsv",
        Path("build_konbaung_sentence_corpus.py").resolve(),
        Path("repair_konbaung_sentence_corpus.py").resolve(),
    ]
    for path in provenance_files:
        if path.exists():
            destination = output / "provenance" / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    reported_source = sources[REPORTED_CASE]["canonical_text"]
    write_text_lf(
        output / "high_risk_review" / "CONFIRMED_vol3_page0050.md",
        render_known_case(reported_source, reported_records),
    )
    write_json(
        output / "high_risk_review" / "CONFIRMED_vol3_page0050.json",
        {
            "volume": 3,
            "page": 50,
            "confirmed_body_lines_removed": list(range(26, 34)),
            "actual_footnotes_begin_line": 34,
            "v2_editorial_footnote_range": [25, 37],
            "records": reported_records,
        },
    )

    editorial_by_page: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in by_reason.get("editorial_footnote", []):
        editorial_by_page[(record["volume"], record["page"])].append(record)
    high_risk_candidates = []
    for (volume, page), records in sorted(editorial_by_page.items()):
        records.sort(key=lambda item: item["source_line"])
        first_date = next(
            (record for record in records if base.is_date_reference(record["removed_text"])),
            None,
        )
        if first_date is None:
            continue
        prose_before = []
        for record in records:
            if record["source_line"] >= first_date["source_line"]:
                break
            myanmar_count = sum("\u1000" <= char <= "\u109f" for char in record["removed_text"])
            if myanmar_count >= 15:
                prose_before.append(record["source_line"])
        if prose_before:
            high_risk_candidates.append(
                {
                    "volume": volume,
                    "page": page,
                    "first_date_line": first_date["source_line"],
                    "prose_lines_before_date": prose_before,
                    "original_ocr_page": f"original_ocr_pages/vol{volume}/page_{page:04d}.txt",
                }
            )
    write_jsonl(
        output / "high_risk_review" / "editorial_tail_pages_with_prose_before_date.jsonl",
        high_risk_candidates,
    )
    write_text_lf(
        output / "high_risk_review" / "editorial_tail_pages_with_prose_before_date.md",
        render_high_risk_markdown(high_risk_candidates),
    )

    total_v2_records = sum(actual_v2_reason_counts.values())
    total_all_records = len(all_removals)
    summary = {
        "audit_schema_version": 1,
        "target": "Konbaung sentence corpus v2 repaired build",
        "corpus": workspace_relative(corpus, workspace),
        "exact_source_archive": workspace_relative(source_root, workspace),
        "source_pages": len(sources),
        "included_v2_pages": len(page_records),
        "excluded_duplicate_pages": len(exclusions),
        "v2_removal_records": total_v2_records,
        "audit_records_including_whole_page_exclusions": total_all_records,
        "reason_count": len(reason_counts),
        "total_removed_codepoints_excluding_line_endings": sum(reason_codepoints.values()),
        "total_removed_non_whitespace_codepoints": sum(reason_nonspace.values()),
        "reason_summary": reason_summary,
        "confirmed_error": {
            "volume": 3,
            "page": 50,
            "body_lines_wrongly_removed": list(range(26, 34)),
            "misclassification": "editorial_footnote",
        },
        "high_risk_editorial_tail_candidate_pages": len(high_risk_candidates),
        "validation_status": "passed" if not fatal_errors else "failed",
    }
    write_json(output / "summary.json", summary)

    test_results = {
        "status": "passed" if not fatal_errors else "failed",
        "checks": checks,
        "total_error_count": len(fatal_errors),
        "errors": fatal_errors[:500],
    }
    write_json(output / "tests" / "test_results.json", test_results)
    test_lines = [
        "# Audit validation results",
        "",
        f"Overall status: **{test_results['status'].upper()}**",
        "",
    ]
    for check in checks:
        test_lines.extend(
            [
                f"## {check['name']}",
                "",
                f"Status: **{check['status']}**  ",
                check["details"],
                "",
            ]
        )
        if check["errors"]:
            test_lines.extend(["~~~text", *check["errors"], "~~~", ""])
    write_text_lf(output / "tests" / "test_results.md", "\n".join(test_lines).rstrip() + "\n")

    readme = f"""# Konbaung corpus v2 removal audit

This bundle exhaustively records textual content removed by `konbaung_sentence_corpus_20260713_repaired` (schema version 2). It does not modify the OCR source or either corpus.

## Start here

- `summary.json`: counts and validation status.
- `high_risk_review/CONFIRMED_vol3_page0050.md`: the reported false footnote cut.
- `by_reason/`: one Markdown file and one JSONL file for every removal reason.
- `all_removed_spans.jsonl`: the complete machine-readable deletion ledger.
- `all_removed_spans.tsv`: the same ledger in a spreadsheet-friendly format; embedded newlines are escaped.
- `original_ocr_pages/`: all {len(sources)} original page texts, copied exactly from each archived source JSON's `canonicalText` field.
- `v2_page_records/`: exact copies of all {len(page_records)} v2 page provenance records.
- `source_pages_manifest.jsonl`: archived source paths and SHA-256 hashes.
- `provenance/`: v2 manifests, original removal TSVs, and the two build scripts.
- `tests/`: independent audit results.

## Scope note

The deletion ledger contains every recorded non-layout deletion plus four whole-page exclusions. OCR line wraps and whitespace runs were normalized by the corpus builder; those layout-only changes are not treated as lost textual content. The coverage test proves that every non-whitespace character in every included archived OCR page is either retained by v2 or represented in this ledger.

## Confirmed problem

Volume 3 page 50 lines 26-33 are main chronicle content but v2 removed them as `editorial_footnote`. The actual Gregorian date-conversion footnotes begin at source line 34. See `high_risk_review/CONFIRMED_vol3_page0050.md`.

## Validation

Status: **{test_results["status"].upper()}**. The audit checked source hashes, exact source slices, line/offset correctness, non-overlap, manifest counts, a rerun of the v2 cleaner, full non-whitespace character coverage, and the reported failure.
"""
    write_text_lf(output / "README_FIRST.md", readme)

    if fatal_errors:
        raise RuntimeError(
            f"Audit validation failed with {len(fatal_errors)} errors; outputs were retained at {output}"
        )

    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, (Path(output.name) / path.relative_to(output)).as_posix())
    with zipfile.ZipFile(zip_path, "r") as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"ZIP integrity test failed at {corrupt}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Audit directory: {output}")
    print(f"ZIP archive: {zip_path}")
    print(f"ZIP SHA-256: {sha256_bytes(zip_path.read_bytes())}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise

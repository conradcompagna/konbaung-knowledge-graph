#!/usr/bin/env python3
"""Build a repaired copy of the Konbaung sentence corpus from explicit audit findings."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import build_konbaung_sentence_corpus as base


SOURCE_ROOT = Path("konbaung_reader_app/static/data/konbaung/pages")
DEFAULT_OUTPUT = Path("konbaung_sentence_corpus_20260713_repaired")

# These archive pages duplicate the immediately preceding printed page.
DUPLICATE_PAGES = {
    (1, 225): 224,
    (1, 242): 241,
    (1, 295): 294,
    (1, 301): 300,
}

# Restore only lines confirmed by the audit to be historical prose.
RESTORE_LINES = {
    (1, 60): {2: "pre_header_debris"},
    (1, 112): {2: "pre_header_debris"},
    (2, 128): {34: "editorial_footnote", 35: "editorial_footnote"},
    (3, 131): {3: "running_header"},
    (3, 445): {
        33: "editorial_footnote",
        34: "editorial_footnote",
        35: "editorial_footnote",
    },
}

# Remove only running headers and printed page numbers confirmed by the audit.
REMOVE_LINES = {
    (2, 30): {5: ("manual_running_header", "ကုန်းဘောင်ဆက် မဟာရာဇဝင်တော်ကြီး")},
    (2, 60): {5: ("manual_running_header", "ကုန်းဘောင်ဆက် မဟာရာဇဝင်တော်ကြီး")},
    (2, 230): {35: ("manual_running_header", "ကုန်းဘောင်ဆက် မဟာရာဇဝင်တော်ကြီး")},
    (1, 261): {1: ("manual_running_header", "စစ်ကိုင်းမြို့တည် မင်းတရားကြီး")},
    (1, 263): {1: ("manual_running_header", "စစ်ကိုင်းမြို့တည် မင်းတရားကြီး")},
    (1, 265): {1: ("manual_running_header", "စစ်ကိုင်းမြို့တည် မင်းတရားကြီး")},
    (1, 267): {1: ("manual_running_header", "စစ်ကိုင်းမြို့တည် မင်းတရားကြီး")},
    (1, 269): {1: ("manual_running_header", "စစ်ကိုင်းမြို့တည် မင်းတရားကြီး")},
    (1, 271): {1: ("manual_running_header", "စစ်ကိုင်းမြို့တည် မင်းတရားကြီး")},
    (1, 273): {1: ("manual_running_header", "စစ်ကိုင်းမြို့တည် မင်းတရားကြီး")},
    (1, 275): {1: ("manual_running_header", "စစ်ကိုင်းမြို့တည် မင်းတရားကြီး")},
    (1, 385): {1: ("manual_running_header", "ငစဉ့်ကူးမင်း အကြောင်း")},
    (1, 387): {1: ("manual_running_header", "ငစဉ့်ကူးမင်း အကြောင်း")},
    (1, 391): {1: ("manual_running_header", "ဘောင်းကားမင်း အကြောင်း")},
    (1, 59): {5: ("manual_page_number", "၅၇")},
    (1, 67): {14: ("manual_page_number", "၆၁")},
    (1, 165): {4: ("manual_page_number", "၁၅၉")},
    (2, 169): {7: ("manual_page_number", "၁၆၇")},
    (2, 203): {4: ("manual_page_number", "၂၀၁")},
    (2, 241): {4: ("manual_page_number", "၂၃၇")},
    (3, 173): {18: ("manual_page_number", "၁၄၇")},
    (3, 257): {4: ("manual_page_number", "၂၃၁")},
    (3, 273): {7: ("manual_page_number", "၂၄၇")},
    (3, 335): {4: ("manual_page_number", "၃ဝ၉")},
    (3, 467): {5: ("manual_page_number", "၄၃၃")},
    (3, 469): {5: ("manual_page_number", "၄၃၅")},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_page_lookup(source_root: Path) -> dict[str, tuple[int, int]]:
    lookup: dict[str, tuple[int, int]] = {}
    for volume in (1, 2, 3):
        for path in sorted((source_root / f"vol{volume}").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            page = int(payload.get("pageNumber", path.stem))
            key = text_key(payload["canonicalText"])
            previous = lookup.get(key)
            if previous and (previous in RESTORE_LINES or previous in REMOVE_LINES):
                raise RuntimeError(f"Ambiguous page text hash for repair page {previous}")
            lookup[key] = (volume, page)
    return lookup


def cleaned_page_with_repairs(
    text: str,
    page_lookup: dict[str, tuple[int, int]],
) -> tuple[str, list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    page_key = page_lookup[text_key(text)]
    removed, reasons, lines = base.classify_removed_lines(text)

    for line_number, expected_reason in RESTORE_LINES.get(page_key, {}).items():
        index = line_number - 1
        actual_reason = reasons.get(index)
        if index not in removed or actual_reason != expected_reason:
            raise RuntimeError(
                f"Repair precondition failed for vol{page_key[0]} page {page_key[1]} "
                f"line {line_number}: expected {expected_reason!r}, got {actual_reason!r}"
            )
        removed.remove(index)
        reasons.pop(index, None)

    for line_number, (reason, expected_text) in REMOVE_LINES.get(page_key, {}).items():
        index = line_number - 1
        actual_text = lines[index]["text"].strip()
        if actual_text != expected_text:
            raise RuntimeError(
                f"Removal precondition failed for vol{page_key[0]} page {page_key[1]} "
                f"line {line_number}: expected {expected_text!r}, got {actual_text!r}"
            )
        if index in removed:
            raise RuntimeError(
                f"Audited residual was already removed on vol{page_key[0]} page {page_key[1]} "
                f"line {line_number}"
            )
        removed.add(index)
        reasons[index] = reason

    chars: list[str] = []
    source_positions: list[int] = []
    removal_records: list[dict[str, Any]] = []
    for row in lines:
        if row["index"] in removed:
            removal_records.append(
                {
                    "line": row["line_number"],
                    "start_codepoint": row["start"],
                    "end_codepoint": row["end"],
                    "reason": reasons[row["index"]],
                    "text": row["text"],
                }
            )
            continue
        body = row["text"]
        leading = len(body) - len(body.lstrip())
        trailing = len(body.rstrip())
        if leading >= trailing:
            continue
        if chars and chars[-1] != " ":
            chars.append(" ")
            source_positions.append(row["start"] + leading)
        for relative, char in enumerate(body[leading:trailing], start=leading):
            source = row["start"] + relative
            if char.isspace():
                if chars and chars[-1] != " ":
                    chars.append(" ")
                    source_positions.append(source)
            else:
                chars.append(char)
                source_positions.append(source)

    while chars and chars[-1] == " ":
        chars.pop()
        source_positions.pop()

    filtered_chars: list[str] = []
    filtered_positions: list[int] = []
    last_nonspace = ""
    for char, source in zip(chars, source_positions):
        if char == base.BURMESE_STOP and last_nonspace == base.BURMESE_STOP:
            removal_records.append(
                {
                    "line": 1 + text.count("\n", 0, source),
                    "start_codepoint": source,
                    "end_codepoint": source + 1,
                    "reason": "duplicate_sentence_terminator",
                    "text": char,
                }
            )
            continue
        if char == "]" and last_nonspace == base.BURMESE_STOP:
            removal_records.append(
                {
                    "line": 1 + text.count("\n", 0, source),
                    "start_codepoint": source,
                    "end_codepoint": source + 1,
                    "reason": "orphan_closing_bracket",
                    "text": char,
                }
            )
            continue
        filtered_chars.append(char)
        filtered_positions.append(source)
        if not char.isspace():
            last_nonspace = char

    cleaned = "".join(filtered_chars)
    possible_footnotes = []
    retained_nonempty = [row for row in lines if row["text"].strip() and row["index"] not in removed]
    for row in retained_nonempty[-4:]:
        stripped = row["text"].strip()
        if base.NUMBERED_RE.match(stripped) and len(stripped) <= 80:
            possible_footnotes.append(
                {"line": row["line_number"], "text": row["text"], "reason": "numbered_tail_retained"}
            )
    return cleaned, filtered_positions, removal_records, possible_footnotes


def jsonl_write(path: Path, records: Iterable[dict[str, Any]]) -> None:
    base.jsonl_write(path, records)


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing derived corpus: {output}")
    for directory in ("cross_page_sentences", "boundary_fragments", "review"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    page_lookup = build_page_lookup(source_root)
    original_cleaned_page = base.cleaned_page
    base.cleaned_page = lambda text: cleaned_page_with_repairs(text, page_lookup)
    try:
        results = []
        excluded_pages = []
        for volume in (1, 2, 3):
            all_files = sorted((source_root / f"vol{volume}").glob("*.json"))
            files = []
            for path in all_files:
                payload = json.loads(path.read_text(encoding="utf-8"))
                page = int(payload.get("pageNumber", path.stem))
                duplicate_of = DUPLICATE_PAGES.get((volume, page))
                if duplicate_of is None:
                    files.append(path)
                    continue
                excluded_pages.append(
                    {
                        "volume": volume,
                        "page": page,
                        "duplicate_of": duplicate_of,
                        "source_path": str(path.resolve()),
                        "source_sha256": base.sha256_bytes(path.read_bytes()),
                    }
                )
            results.append(base.build_volume(volume, files, output))
    finally:
        base.cleaned_page = original_cleaned_page

    validation = base.validate(results)
    base.write_review_files(output, results)
    all_sentences = [sentence for result in results for sentence in result["sentences"]]
    jsonl_write(output / "sentences" / "all_volumes.jsonl", all_sentences)

    volumes = []
    for result in results:
        volumes.append(
            {
                "volume": result["volume"],
                "page_count": len(result["pages"]),
                "sentence_count": len(result["sentences"]),
                "cross_page_sentence_count": sum(s["cross_page"] for s in result["sentences"]),
                "boundary_fragment_count": len(result["fragments"]),
                "removed_line_count": sum(len(page["removals"]) for page in result["pages"]),
                "removal_counts": result["removal_counts"],
            }
        )

    repair_manifest = {
        "schema_version": 1,
        "source_root": str(source_root),
        "output_root": str(output),
        "policy": "Explicit audit allowlist only; no generalized cleaning changes.",
        "restored_lines": [
            {"volume": volume, "page": page, "lines": sorted(lines)}
            for (volume, page), lines in sorted(RESTORE_LINES.items())
        ],
        "removed_confirmed_running_matter": [
            {"volume": volume, "page": page, "lines": sorted(lines)}
            for (volume, page), lines in sorted(REMOVE_LINES.items())
        ],
        "excluded_duplicate_pages": excluded_pages,
        "deliberately_unchanged": [
            "bare numeric sentence records",
            "ambiguous numeric-only source lines",
            "Latin OCR and citation callouts",
            "asterisk editorial markers",
            "all other inherited OCR text",
        ],
        "volumes": volumes,
        "totals": {
            "source_pages": sum(len(list((source_root / f"vol{v}").glob("*.json"))) for v in (1, 2, 3)),
            "annotation_pages": sum(item["page_count"] for item in volumes),
            "excluded_duplicate_pages": len(excluded_pages),
            "sentences": len(all_sentences),
            "cross_page_sentences": sum(item["cross_page_sentence_count"] for item in volumes),
            "boundary_fragments": sum(item["boundary_fragment_count"] for item in volumes),
        },
        "validation": validation,
    }
    (output / "repair_manifest.json").write_text(
        json.dumps(repair_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source_root": str(source_root),
                "output_root": str(output),
                "ownership_rule": (
                    "A cross-page sentence belongs to the active page on which it begins; "
                    "four audited duplicate scans are skipped."
                ),
                "sentence_boundary": "U+104B MYANMAR SIGN SECTION (။) only.",
                "whitespace_policy": (
                    "OCR line wraps and other whitespace runs are normalized to one space; "
                    "they never create sentence boundaries."
                ),
                "volumes": volumes,
                "totals": repair_manifest["totals"],
                "validation": validation,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with (output / "review" / "excluded_duplicate_pages.tsv").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("volume\tpage\tduplicate_of\tsource_sha256\n")
        for item in excluded_pages:
            handle.write(
                f"{item['volume']}\t{item['page']}\t{item['duplicate_of']}\t{item['source_sha256']}\n"
            )
    (output / "README.md").write_text(
        "# Repaired Konbaung sentence corpus\n\n"
        "This is a new derived corpus. The original OCR, previous sentence corpus, and builder are unchanged.\n\n"
        "Repairs are restricted to the explicit audit allowlists in `repair_manifest.json`: five pages with "
        "restored prose, fourteen confirmed running headers, twelve confirmed printed page numbers, and four "
        "excluded duplicate scans. Ambiguous OCR and list-number records are deliberately preserved.\n\n"
        "Use `sentences/all_volumes.jsonl` for annotation. Do not append `cross_page_sentences/*.jsonl`; those "
        "files are a routing subset of the main sentence files.\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(repair_manifest["totals"], indent=2))
    print(output)


if __name__ == "__main__":
    main()

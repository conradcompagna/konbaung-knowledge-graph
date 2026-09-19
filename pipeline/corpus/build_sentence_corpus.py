#!/usr/bin/env python3
"""Build a copy-safe, sentence-segmented Konbaung chronicle corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("konbaung_reader_app/static/data/konbaung/pages")
DEFAULT_OUTPUT = Path("konbaung_sentence_corpus_20260713")
BURMESE_STOP = "\u104b"

HEADER_RE = re.compile(
    r"(?:\u1019\u101f\u102c\u101b\u102c\u1007\u101d\u1004\u103a\u1010\u1031\u102c\u103a\u1000\u103c\u102e\u1038|"
    r"\u1016\u103c\u1005\u103a\u1010\u1031\u102c\u103a\u1005\u1009\u103a|"
    r"\u1019\u103c\u102d\u102f\u1037\u1010\u100a\u103a\u1014\u1014\u103a\u1038\u1010\u100a\u103a.*\u1019\u1004\u103a\u1038\u1010\u101b\u102c\u1038\u1000\u103c\u102e\u1038|"
    r"\u1014\u1014\u103a\u1038\u1005\u1036.*\u1019\u1004\u103a\u1038\u1010\u101b\u102c\u1038\u1000\u103c\u102e\u1038)"
)
NUMBER_RE = re.compile(r"^[\u1040-\u10490-9\s]+$")
PAGE_NUMBER_OCR_RE = re.compile(r"^[\u1040-\u1049\u101d0-9A-Za-z?()\[\].,\-\s]+$")
NUMBERED_RE = re.compile(r"^[\u1040-\u10490-9]+\s*[.\u104a\u104b]")
SHORT_NON_BURMESE_RE = re.compile(r"^[^\u1000-\u109f]{1,16}$")
TAIL_JUNK_RE = re.compile(r"^[A-Za-z0-9\s*?+\-_=.,:;|/\\()\[\]]{1,20}$")
EDITOR_MARKER = "\u1005\u102c\u1010\u100a\u103a\u1038"
MONTH_NAMES = (
    "\u1007\u1014\u103a\u1014\u101d\u102b\u101b\u102e",
    "\u1016\u1031\u1016\u1031\u102c\u103a\u101d\u102b\u101b\u102e",
    "\u1019\u1010\u103a",
    "\u1027\u1015\u103c\u102e",
    "\u1019\u1031",
    "\u1007\u103d\u1014\u103a",
    "\u1007\u1030\u101c\u102d\u102f\u1004\u103a",
    "\u1007\u1030\u101c\u102d\u102f\u1004\u103a",
    "\u101e\u103c\u1002\u102f\u1010\u103a",
    "\u1029\u1002\u102f\u1010\u103a",
    "\u1005\u1000\u103a\u1010\u1004\u103a\u1018\u102c",
    "\u1021\u1031\u102c\u1000\u103a\u1010\u102d\u102f\u1018\u102c",
    "\u1014\u102d\u102f\u101d\u1004\u103a\u1018\u102c",
    "\u1012\u102e\u1007\u1004\u103a\u1018\u102c",
)
GREGORIAN_YEAR_RE = re.compile(
    r"(?:1[4-9][0-9]{2}|2[0-9]{3}|[\u1041\u1042][\u1044-\u1049\u101d][\u1040-\u1049\u101d]{2})"
)
LATIN_RE = re.compile(r"[A-Za-z]")
NUMERIC_FRAGMENT_RE = re.compile(r"^[\u1040-\u1049\u101d0-9\s.,\u104a\u104b()\-]+$")
DATE_DIGITS = r"[\u1040-\u1049\u101d0-9]+"
DATE_MONTH_RE = re.compile(
    rf"(?:{DATE_DIGITS}\s*(?:{'|'.join(re.escape(name) for name in MONTH_NAMES)})|"
    rf"(?:{'|'.join(re.escape(name) for name in MONTH_NAMES)})\s*[-,\u104a]?\s*{DATE_DIGITS})"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utf16_offset(text: str, codepoint_offset: int) -> int:
    return len(text[:codepoint_offset].encode("utf-16-le")) // 2


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def line_ranges(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    position = 0
    for index, raw in enumerate(text.splitlines(keepends=True), start=1):
        body = raw.rstrip("\r\n")
        rows.append(
            {
                "index": index - 1,
                "line_number": index,
                "start": position,
                "end": position + len(body),
                "text": body,
            }
        )
        position += len(raw)
    if not rows and text:
        rows.append({"index": 0, "line_number": 1, "start": 0, "end": len(text), "text": text})
    return rows


def compact(text: str) -> str:
    return re.sub(r"[\s\"'`\u201c\u201d]+", "", text)


def is_publisher_line(text: str) -> bool:
    value = compact(text)
    if len(value) > 40:
        return False
    return value.startswith(
        ("\u101b\u102c\u1015\u103c\u100a\u1037", "\u1005\u102c\u1015\u103c\u100a\u1037")
    ) and (
        "\u1021\u102f\u1015\u103a" in value
        or "\u1010\u102d\u102f\u1000\u103a" in value
        or "\u1000\u102d\u102f\u1000\u103a" in value
        or "\u1010\u102d\u102f\u1037" in value
    )


def is_date_reference(text: str) -> bool:
    return bool(
        GREGORIAN_YEAR_RE.search(text)
        or DATE_MONTH_RE.search(text)
        or "\u1018\u102e\u1005\u102e" in text
    )


def is_footnote_signal(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith("*") or stripped.lstrip("[(").lstrip().startswith("*"):
        return True
    if EDITOR_MARKER in stripped:
        return True
    return bool(
        NUMBERED_RE.match(stripped) and (is_date_reference(stripped) or LATIN_RE.search(stripped))
    )


def is_footnote_lead(text: str) -> bool:
    stripped = text.strip()
    return bool(
        is_footnote_signal(stripped)
        or (
            NUMBERED_RE.match(stripped)
            and len(stripped) <= 24
            and NUMERIC_FRAGMENT_RE.fullmatch(stripped)
        )
        or is_date_reference(stripped)
        or stripped in MONTH_NAMES
        or (len(stripped) <= 20 and LATIN_RE.search(stripped))
    )


def is_trailing_junk(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped in {BURMESE_STOP, "\u104a", "*", "**", "***", "-"}:
        return True
    if TAIL_JUNK_RE.fullmatch(stripped):
        return True
    return bool(len(stripped) <= 16 and SHORT_NON_BURMESE_RE.fullmatch(stripped))


def top_nonempty_indices(lines: list[dict[str, Any]], limit: int = 3) -> list[int]:
    return [row["index"] for row in lines if row["text"].strip()][:limit]


def classify_removed_lines(text: str) -> tuple[set[int], dict[int, str], list[dict[str, Any]]]:
    lines = line_ranges(text)
    removed: set[int] = set()
    reasons: dict[int, str] = {}
    top = top_nonempty_indices(lines)
    header_indices = [index for index in top if HEADER_RE.search(lines[index]["text"].strip())]
    has_header = bool(header_indices)
    if header_indices:
        first_header = min(header_indices)
        for index in top:
            if index < first_header:
                removed.add(index)
                reasons[index] = "pre_header_debris"
    if len(header_indices) >= 2:
        for index in range(min(header_indices), max(header_indices) + 1):
            stripped = lines[index]["text"].strip()
            if stripped and len(stripped) <= 80 and BURMESE_STOP not in stripped:
                removed.add(index)
                reasons[index] = "running_header"

    for index in top:
        stripped = lines[index]["text"].strip()
        if HEADER_RE.search(stripped):
            removed.add(index)
            reasons[index] = "running_header"
        elif NUMBER_RE.fullmatch(stripped):
            removed.add(index)
            reasons[index] = "page_number"
        elif has_header and len(stripped) <= 16 and PAGE_NUMBER_OCR_RE.fullmatch(stripped):
            removed.add(index)
            reasons[index] = "page_number_ocr"

    nonempty = [row["index"] for row in lines if row["text"].strip()]
    publisher_index = next(
        (index for index in reversed(nonempty[-10:]) if is_publisher_line(lines[index]["text"])),
        None,
    )
    footer_limit = (
        publisher_index if publisher_index is not None else (nonempty[-1] + 1 if nonempty else 0)
    )
    search_indices = [index for index in nonempty if index < footer_limit][-14:]
    signals = [index for index in search_indices if is_footnote_signal(lines[index]["text"])]
    immediate_tail = search_indices[-4:]
    for index in immediate_tail:
        stripped = lines[index]["text"].strip()
        if is_date_reference(stripped):
            signals.append(index)
    if signals:
        footnote_start = min(signals)
        has_editor_marker = any(EDITOR_MARKER in lines[index]["text"] for index in signals)
        while footnote_start > 0:
            previous = footnote_start - 1
            if previous not in nonempty:
                footnote_start = previous
                continue
            if is_footnote_lead(lines[previous]["text"]):
                footnote_start = previous
                continue
            if has_editor_marker and NUMBERED_RE.match(lines[previous]["text"].strip()):
                footnote_start = previous
                continue
            break
        for index in range(footnote_start, footer_limit):
            removed.add(index)
            reasons[index] = "editorial_footnote"

    for index in immediate_tail:
        stripped = lines[index]["text"].strip()
        if index in removed or not NUMBERED_RE.match(stripped):
            continue
        if len(stripped) <= 16 and NUMERIC_FRAGMENT_RE.fullmatch(stripped):
            removed.add(index)
            reasons[index] = "editorial_footnote_fragment"
    for index in immediate_tail:
        stripped = lines[index]["text"].strip()
        if index in removed:
            continue
        if len(stripped) <= 8 and re.fullmatch(r"[\u1040-\u1049\u101d0-9]+", stripped):
            removed.add(index)
            reasons[index] = "trailing_numeric_debris"

    if publisher_index is not None:
        for index in range(publisher_index, len(lines)):
            removed.add(index)
            reasons[index] = "publisher_footer"
    else:
        for index in reversed(nonempty):
            if index in removed:
                continue
            if is_trailing_junk(lines[index]["text"]):
                removed.add(index)
                reasons[index] = "trailing_ocr_debris"
                continue
            break

    for row in lines:
        stripped = row["text"].strip()
        if not stripped or row["index"] in removed:
            continue
        if LATIN_RE.search(stripped) and not re.search(r"[\u1000-\u109f]", stripped):
            removed.add(row["index"])
            reasons[row["index"]] = "non_burmese_debris"

    return removed, reasons, lines


def cleaned_page(text: str) -> tuple[str, list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    removed, reasons, lines = classify_removed_lines(text)
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
        if char == BURMESE_STOP and last_nonspace == BURMESE_STOP:
            line_number = 1 + text.count("\n", 0, source)
            removal_records.append(
                {
                    "line": line_number,
                    "start_codepoint": source,
                    "end_codepoint": source + 1,
                    "reason": "duplicate_sentence_terminator",
                    "text": char,
                }
            )
            continue
        if char == "]" and last_nonspace == BURMESE_STOP:
            line_number = 1 + text.count("\n", 0, source)
            removal_records.append(
                {
                    "line": line_number,
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
    chars = filtered_chars
    source_positions = filtered_positions
    cleaned = "".join(chars)
    possible_footnotes = []
    retained_nonempty = [
        row for row in lines if row["text"].strip() and row["index"] not in removed
    ]
    for row in retained_nonempty[-4:]:
        stripped = row["text"].strip()
        if NUMBERED_RE.match(stripped) and len(stripped) <= 80:
            possible_footnotes.append(
                {
                    "line": row["line_number"],
                    "text": row["text"],
                    "reason": "numbered_tail_retained",
                }
            )
    return cleaned, source_positions, removal_records, possible_footnotes


def trimmed_range(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def make_piece(page: dict[str, Any], start: int, end: int) -> dict[str, Any] | None:
    trimmed = trimmed_range(page["cleaned_text"], start, end)
    if trimmed is None:
        return None
    start, end = trimmed
    mapping = page["source_positions"]
    source_start = mapping[start]
    source_end = mapping[end - 1] + 1
    original = page["original_text"]
    return {
        "page": page["page"],
        "clean_start": start,
        "clean_end": end,
        "start_codepoint": source_start,
        "end_codepoint": source_end,
        "start_utf16": utf16_offset(original, source_start),
        "end_utf16": utf16_offset(original, source_end),
        "text": page["cleaned_text"][start:end],
        "source_text": original[source_start:source_end],
    }


def make_pieces(page: dict[str, Any], start: int, end: int) -> list[dict[str, Any]]:
    trimmed = trimmed_range(page["cleaned_text"], start, end)
    if trimmed is None:
        return []
    start, end = trimmed
    mapping = page["source_positions"]
    original = page["original_text"]
    boundaries = [start]
    for index in range(start + 1, end):
        previous_source = mapping[index - 1]
        current_source = mapping[index]
        if current_source <= previous_source:
            continue
        skipped = original[previous_source + 1 : current_source]
        if any(not char.isspace() for char in skipped):
            boundaries.append(index)
    boundaries.append(end)
    pieces = []
    for left, right in zip(boundaries, boundaries[1:]):
        piece = make_piece(page, left, right)
        if piece:
            pieces.append(piece)
    return pieces


def jsonl_write(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def page_text_for_models(text: str) -> str:
    return re.sub(rf"{BURMESE_STOP}\s*", f"{BURMESE_STOP}\n", text).strip()


def build_volume(volume: int, files: list[Path], output: Path) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    possible_footnotes: list[dict[str, Any]] = []
    removal_counts: Counter[str] = Counter()

    for path in files:
        raw = path.read_bytes()
        source_hashes[str(path)] = sha256_bytes(raw)
        payload = json.loads(raw.decode("utf-8"))
        original = payload["canonicalText"]
        cleaned, source_positions, removals, possible = cleaned_page(original)
        page_number = int(payload.get("pageNumber", path.stem))
        for item in possible:
            possible_footnotes.append({"volume": volume, "page": page_number, **item})
        removal_counts.update(item["reason"] for item in removals)
        pages.append(
            {
                "volume": volume,
                "page": page_number,
                "source_path": str(path),
                "source_sha256": source_hashes[str(path)],
                "original_text": original,
                "cleaned_text": cleaned,
                "source_positions": source_positions,
                "removals": removals,
                "sentence_ids": [],
                "owned_sentence_ids": [],
                "cross_page_sentence_ids": [],
                "boundary_fragment_ids": [],
            }
        )

    last_page = pages[-1]
    last_stop = last_page["cleaned_text"].rfind(BURMESE_STOP)
    tail_start = last_stop + 1
    tail_range = trimmed_range(
        last_page["cleaned_text"], tail_start, len(last_page["cleaned_text"])
    )
    if tail_range is not None:
        clean_start, clean_end = tail_range
        tail = last_page["cleaned_text"][clean_start:clean_end]
        if len(tail) <= 120:
            mapping = last_page["source_positions"]
            source_start = mapping[clean_start]
            source_end = mapping[clean_end - 1] + 1
            last_page["removals"].append(
                {
                    "line": 1 + last_page["original_text"].count("\n", 0, source_start),
                    "start_codepoint": source_start,
                    "end_codepoint": source_end,
                    "reason": "final_unterminated_caption",
                    "text": last_page["original_text"][source_start:source_end],
                }
            )
            removal_counts["final_unterminated_caption"] += 1
            truncate = tail_start
            while truncate > 0 and last_page["cleaned_text"][truncate - 1].isspace():
                truncate -= 1
            last_page["cleaned_text"] = last_page["cleaned_text"][:truncate]
            last_page["source_positions"] = last_page["source_positions"][:truncate]

    sentences: list[dict[str, Any]] = []
    fragments: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def emit_sentence(pieces: list[dict[str, Any]]) -> None:
        if not pieces:
            return
        sentence_id = f"vol{volume}_s{len(sentences) + 1:06d}"
        record = {
            "id": sentence_id,
            "volume": volume,
            "owner_page": pieces[0]["page"],
            "pages": list(dict.fromkeys(piece["page"] for piece in pieces)),
            "cross_page": len({piece["page"] for piece in pieces}) > 1,
            "text": normalize_whitespace(" ".join(piece["text"] for piece in pieces)),
            "source": pieces,
        }
        sentences.append(record)

    for page in pages:
        start = 0
        for index, char in enumerate(page["cleaned_text"]):
            if char != BURMESE_STOP:
                continue
            pending.extend(make_pieces(page, start, index + 1))
            emit_sentence(pending)
            pending = []
            start = index + 1
        pending.extend(make_pieces(page, start, len(page["cleaned_text"])))

    if pending:
        fragment_id = f"vol{volume}_f000001"
        fragments.append(
            {
                "id": fragment_id,
                "volume": volume,
                "owner_page": pending[0]["page"],
                "pages": list(dict.fromkeys(piece["page"] for piece in pending)),
                "text": normalize_whitespace(" ".join(piece["text"] for piece in pending)),
                "reason": "unterminated_at_end_of_valid_volume",
                "source": pending,
            }
        )

    page_by_number = {page["page"]: page for page in pages}
    for sentence in sentences:
        for page_number in sentence["pages"]:
            page_by_number[page_number]["sentence_ids"].append(sentence["id"])
            if sentence["cross_page"]:
                page_by_number[page_number]["cross_page_sentence_ids"].append(sentence["id"])
        page_by_number[sentence["owner_page"]]["owned_sentence_ids"].append(sentence["id"])
    for fragment in fragments:
        for page_number in fragment["pages"]:
            page_by_number[page_number]["boundary_fragment_ids"].append(fragment["id"])

    clean_dir = output / "cleaned_pages" / f"vol{volume}"
    record_dir = output / "page_records" / f"vol{volume}"
    clean_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    for page in pages:
        (clean_dir / f"page_{page['page']:04d}.txt").write_text(
            page_text_for_models(page["cleaned_text"]) + "\n", encoding="utf-8", newline="\n"
        )
        public_page = {
            key: value
            for key, value in page.items()
            if key not in {"original_text", "source_positions"}
        }
        public_page["original_character_count"] = len(page["original_text"])
        public_page["cleaned_character_count"] = len(page["cleaned_text"])
        (record_dir / f"page_{page['page']:04d}.json").write_text(
            json.dumps(public_page, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    sentence_dir = output / "sentences"
    sentence_dir.mkdir(parents=True, exist_ok=True)
    jsonl_write(sentence_dir / f"vol{volume}.jsonl", sentences)
    jsonl_write(
        output / "cross_page_sentences" / f"vol{volume}.jsonl",
        (s for s in sentences if s["cross_page"]),
    )
    jsonl_write(output / "boundary_fragments" / f"vol{volume}.jsonl", fragments)

    return {
        "volume": volume,
        "pages": pages,
        "sentences": sentences,
        "fragments": fragments,
        "possible_footnotes": possible_footnotes,
        "removal_counts": dict(removal_counts),
        "source_hashes": source_hashes,
    }


def validate(results: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    for result in results:
        page_lookup = {page["page"]: page for page in result["pages"]}
        coverage: dict[int, Counter[int]] = {page["page"]: Counter() for page in result["pages"]}
        records = [*result["sentences"], *result["fragments"]]
        for record in records:
            reconstructed = normalize_whitespace(
                " ".join(piece["text"] for piece in record["source"])
            )
            if reconstructed != record["text"]:
                errors.append(f"{record['id']}: sentence reconstruction mismatch")
            for piece in record["source"]:
                page = page_lookup[piece["page"]]
                original = page["original_text"]
                exact = original[piece["start_codepoint"] : piece["end_codepoint"]]
                if exact != piece["source_text"]:
                    errors.append(
                        f"{record['id']}: source substring mismatch on page {piece['page']}"
                    )
                if normalize_whitespace(exact) != normalize_whitespace(piece["text"]):
                    errors.append(
                        f"{record['id']}: normalized source mismatch on page {piece['page']}"
                    )
                for index in range(piece["clean_start"], piece["clean_end"]):
                    if not page["cleaned_text"][index].isspace():
                        coverage[piece["page"]][index] += 1

        for page in result["pages"]:
            for index, char in enumerate(page["cleaned_text"]):
                if char.isspace():
                    continue
                count = coverage[page["page"]][index]
                if count != 1:
                    errors.append(
                        f"vol{result['volume']} page {page['page']} char {index}: assigned {count} times"
                    )
                    if len(errors) >= 100:
                        break
            for removal in page["removals"]:
                exact = page["original_text"][removal["start_codepoint"] : removal["end_codepoint"]]
                if exact != removal["text"]:
                    errors.append(
                        f"vol{result['volume']} page {page['page']}: removal source mismatch"
                    )
        for source_path, expected in result["source_hashes"].items():
            if sha256_bytes(Path(source_path).read_bytes()) != expected:
                errors.append(f"source changed during build: {source_path}")

    if errors:
        raise RuntimeError("Validation failed:\n" + "\n".join(errors[:100]))
    return {
        "status": "passed",
        "checks": [
            "Every sentence source piece equals the exact original page substring.",
            "Whitespace-normalized source pieces reconstruct every sentence.",
            "Every retained non-whitespace character is assigned exactly once.",
            "Every removed line equals its recorded original substring.",
            "All source JSON SHA-256 hashes are unchanged after the build.",
        ],
    }


def write_review_files(output: Path, results: list[dict[str, Any]]) -> None:
    review = output / "review"
    review.mkdir(parents=True, exist_ok=True)
    with (review / "removed_lines.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("volume\tpage\tline\treason\ttext\n")
        for result in results:
            for page in result["pages"]:
                for item in page["removals"]:
                    text = item["text"].replace("\t", " ")
                    handle.write(
                        f"{result['volume']}\t{page['page']}\t{item['line']}\t{item['reason']}\t{text}\n"
                    )
    with (review / "possible_footnotes.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("volume\tpage\tline\treason\ttext\n")
        for result in results:
            for item in result["possible_footnotes"]:
                text = item["text"].replace("\t", " ")
                handle.write(
                    f"{item['volume']}\t{item['page']}\t{item['line']}\t{item['reason']}\t{text}\n"
                )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing derived corpus: {args.output.resolve()}")
    for directory in ("cross_page_sentences", "boundary_fragments", "review"):
        (args.output / directory).mkdir(parents=True, exist_ok=True)

    results = []
    for volume in (1, 2, 3):
        files = sorted((args.source / f"vol{volume}").glob("*.json"))
        if not files:
            raise SystemExit(f"No source pages found for vol{volume} under {args.source}")
        results.append(build_volume(volume, files, args.output))

    validation = validate(results)
    write_review_files(args.output, results)
    all_sentences = [sentence for result in results for sentence in result["sentences"]]
    jsonl_write(args.output / "sentences" / "all_volumes.jsonl", all_sentences)

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
    manifest = {
        "schema_version": 1,
        "source_root": str(args.source.resolve()),
        "output_root": str(args.output.resolve()),
        "ownership_rule": "A cross-page sentence belongs to the page on which it begins and retains source pieces for every page it touches.",
        "sentence_boundary": "U+104B MYANMAR SIGN SECTION (။) only.",
        "whitespace_policy": "OCR line wraps and other whitespace runs are normalized to one space; they never create sentence boundaries.",
        "volumes": volumes,
        "totals": {
            "pages": sum(item["page_count"] for item in volumes),
            "sentences": len(all_sentences),
            "cross_page_sentences": sum(item["cross_page_sentence_count"] for item in volumes),
            "boundary_fragments": sum(item["boundary_fragment_count"] for item in volumes),
            "removed_lines": sum(item["removed_line_count"] for item in volumes),
        },
        "validation": validation,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (args.output / "README.md").write_text(
        "# Konbaung sentence corpus\n\n"
        "This is a derived, copy-safe corpus. The source reader JSON files are never modified.\n\n"
        "- `cleaned_pages/`: page text after deterministic removal of running matter, with one sentence per line.\n"
        "- `sentences/`: model-ready JSONL; cross-page sentences are included and owned by their starting page.\n"
        "- `cross_page_sentences/`: duplicate routing index containing the complete cross-page sentence records.\n"
        "- `boundary_fragments/`: text still unterminated at the end of a valid volume.\n"
        "- `page_records/`: per-page provenance, removals, hashes, and sentence IDs.\n"
        "- `review/`: compact removal and possible-footnote audit tables.\n\n"
        "All offsets are against the untouched `canonicalText`; both Unicode code-point and UTF-16 offsets are supplied.\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest["totals"], indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Detect section breaks for Konbaung volumes from OCR JSON layout.

The detector is deterministic:
- parse TOC entries from OCR JSON layout, not an LLM
- map TOC printed page numbers onto approved valid OCR pages
- confirm/locate visible centered headings from OCR word bounding boxes

It does not modify source OCR text or page images.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


DIGITS = {
    **{str(i): str(i) for i in range(10)},
    "၀": "0",
    "၁": "1",
    "၂": "2",
    "၃": "3",
    "၄": "4",
    "၅": "5",
    "၆": "6",
    "၇": "7",
    "၈": "8",
    "၉": "9",
    # Common OCR confusion in Burmese numerals.
    "ဝ": "0",
    "O": "0",
    "o": "0",
}


@dataclass
class VisualLine:
    page_num: int
    line_index: int
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    page_width: int
    page_height: int

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def width_ratio(self) -> float:
        return (self.x1 - self.x0) / self.page_width

    @property
    def center_deviation(self) -> float:
        return abs(self.cx - self.page_width / 2) / self.page_width


@dataclass
class TocEntry:
    section_no: int
    title: str
    toc_page_num: int
    y0: int
    y1: int
    printed_page: Optional[int]
    printed_page_raw: str
    printed_page_candidates: list[dict[str, Any]]


@dataclass
class HeadingCandidate:
    page_num: int
    line_start: int
    line_end: int
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    score_reason: str


def parse_int_loose(text: str) -> Optional[int]:
    chars: list[str] = []
    for ch in text:
        if ch in DIGITS:
            chars.append(DIGITS[ch])
        elif ch.isspace() or ch in ".,၊။:-–—()[]":
            continue
        else:
            return None
    if not chars:
        return None
    value = int("".join(chars))
    return value if value > 0 else None


def normalize_for_match(text: str) -> str:
    return re.sub(r"[\s၊။.,:;\\-–—()\\[\\]\"'“”‘’]+", "", text)


def bbox_of_word(word: dict[str, Any]) -> tuple[int, int, int, int]:
    verts = word["boundingBox"]["vertices"]
    xs = [v.get("x", 0) for v in verts]
    ys = [v.get("y", 0) for v in verts]
    return min(xs), min(ys), max(xs), max(ys)


def word_text(word: dict[str, Any]) -> str:
    return "".join(symbol.get("text", "") for symbol in word.get("symbols", []))


def word_break(word: dict[str, Any]) -> Optional[str]:
    for symbol in reversed(word.get("symbols", [])):
        break_type = symbol.get("property", {}).get("detectedBreak", {}).get("type")
        if break_type:
            return break_type
    return None


def read_ocr_response(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["responses"][0]


def extract_visual_lines(json_path: Path, page_num: int) -> list[VisualLine]:
    response = read_ocr_response(json_path)
    page = response["fullTextAnnotation"]["pages"][0]
    page_width = int(page["width"])
    page_height = int(page["height"])
    lines: list[VisualLine] = []
    current_words: list[dict[str, Any]] = []

    def flush() -> None:
        if not current_words:
            return
        text = " ".join(word_text(word) for word in current_words).strip()
        bboxes = [bbox_of_word(word) for word in current_words]
        lines.append(
            VisualLine(
                page_num=page_num,
                line_index=len(lines) + 1,
                text=text,
                x0=min(b[0] for b in bboxes),
                y0=min(b[1] for b in bboxes),
                x1=max(b[2] for b in bboxes),
                y1=max(b[3] for b in bboxes),
                page_width=page_width,
                page_height=page_height,
            )
        )
        current_words.clear()

    for block in page.get("blocks", []):
        for paragraph in block.get("paragraphs", []):
            for word in paragraph.get("words", []):
                current_words.append(word)
                if word_break(word) in {"LINE_BREAK", "EOL_SURE_SPACE"}:
                    flush()
    flush()
    return lines


def numbered_toc_start(line: VisualLine) -> Optional[int]:
    if line.x0 > line.page_width * 0.35:
        return None
    text = line.text.strip()
    if "။" not in text:
        return None
    prefix = text.split("။", 1)[0]
    value = parse_int_loose(prefix)
    if value is None:
        return None
    if not (1 <= value <= 300):
        return None
    return value


def clean_toc_title(section_no: int, lines: list[VisualLine]) -> str:
    parts = [line.text.strip() for line in lines]
    if parts:
        parts[0] = re.sub(rf"^[\s0-9၀-၉ဝOo]+[။.]\s*", "", parts[0]).strip()
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def parse_toc_entries(ocr_json_dir: Path, toc_pages: list[int]) -> list[TocEntry]:
    entries: list[TocEntry] = []
    for page_num in toc_pages:
        lines = extract_visual_lines(ocr_json_dir / f"page_{page_num:04d}.json", page_num)
        body_lines = [line for line in lines if line.x0 < line.page_width * 0.82]
        page_number_lines = [
            line
            for line in lines
            if line.x0 >= line.page_width * 0.82 and line.y0 > line.page_height * 0.08
        ]
        starts: list[tuple[int, int]] = []
        for idx, line in enumerate(body_lines):
            section_no = numbered_toc_start(line)
            if section_no is not None:
                starts.append((idx, section_no))
        entry_records: list[dict[str, Any]] = []
        for entry_idx, (body_idx, section_no) in enumerate(starts):
            next_body_idx = (
                starts[entry_idx + 1][0] if entry_idx + 1 < len(starts) else len(body_lines)
            )
            entry_lines = body_lines[body_idx:next_body_idx]
            y0 = min(line.y0 for line in entry_lines)
            y1 = max(line.y1 for line in entry_lines)
            last_line = entry_lines[-1]
            target_y = (last_line.y0 + last_line.y1) / 2
            candidates: list[dict[str, Any]] = []
            for line in page_number_lines:
                center_y = (line.y0 + line.y1) / 2
                # Page numbers align roughly with the last line of each TOC entry.
                if y0 - 40 <= center_y <= y1 + 90:
                    value = parse_int_loose(line.text)
                    candidates.append(
                        {
                            "raw": line.text,
                            "value": value,
                            "x0": line.x0,
                            "y0": line.y0,
                            "x1": line.x1,
                            "y1": line.y1,
                            "dy_from_entry_last_line": round(abs(center_y - target_y), 1),
                        }
                    )
            entry_records.append(
                {
                    "section_no": section_no,
                    "entry_lines": entry_lines,
                    "y0": y0,
                    "y1": y1,
                    "target_y": target_y,
                    "candidates": candidates,
                }
            )

        assignments: dict[int, dict[str, Any]] = {}
        for line in sorted(page_number_lines, key=lambda item: item.y0):
            value = parse_int_loose(line.text)
            if value is None:
                continue
            center_y = (line.y0 + line.y1) / 2
            choices: list[tuple[float, int]] = []
            for record_idx, record in enumerate(entry_records):
                if record_idx in assignments:
                    continue
                dy = abs(center_y - record["target_y"])
                if dy <= 140:
                    choices.append((dy, record_idx))
            if not choices:
                continue
            dy, record_idx = min(choices, key=lambda item: item[0])
            assignments[record_idx] = {
                "raw": line.text,
                "value": value,
                "x0": line.x0,
                "y0": line.y0,
                "x1": line.x1,
                "y1": line.y1,
                "dy_from_entry_last_line": round(dy, 1),
            }

        for record_idx, record in enumerate(entry_records):
            section_no = record["section_no"]
            selected = assignments.get(record_idx)
            entries.append(
                TocEntry(
                    section_no=section_no,
                    title=clean_toc_title(section_no, record["entry_lines"]),
                    toc_page_num=page_num,
                    y0=record["y0"],
                    y1=record["y1"],
                    printed_page=selected["value"] if selected else None,
                    printed_page_raw=selected["raw"] if selected else "",
                    printed_page_candidates=record["candidates"],
                )
            )
    entries.sort(key=lambda item: item.section_no)
    return entries


def detect_printed_page(lines: list[VisualLine]) -> Optional[int]:
    candidates: list[int] = []
    for line in lines:
        if line.y0 > line.page_height * 0.12:
            continue
        if line.x0 < line.page_width * 0.18 or line.x0 > line.page_width * 0.78:
            value = parse_int_loose(line.text)
            if value is not None and 1 <= value <= 900:
                candidates.append(value)
    if not candidates:
        return None
    return max(candidates)


def build_printed_page_map(valid_page_nums: list[int], ocr_json_dir: Path) -> dict[int, int]:
    anchors: dict[int, int] = {}
    for page_num in valid_page_nums:
        value = detect_printed_page(
            extract_visual_lines(ocr_json_dir / f"page_{page_num:04d}.json", page_num)
        )
        if value is not None:
            anchors[page_num] = value

    by_ocr: dict[int, int] = {}
    sorted_valid = sorted(valid_page_nums)
    for i, page_num in enumerate(sorted_valid):
        if page_num in anchors:
            by_ocr[page_num] = anchors[page_num]
            continue
        previous_anchor = next((p for p in reversed(sorted_valid[:i]) if p in anchors), None)
        next_anchor = next((p for p in sorted_valid[i + 1 :] if p in anchors), None)
        if previous_anchor is not None:
            previous_index = sorted_valid.index(previous_anchor)
            by_ocr[page_num] = anchors[previous_anchor] + (i - previous_index)
        elif next_anchor is not None:
            next_index = sorted_valid.index(next_anchor)
            by_ocr[page_num] = anchors[next_anchor] - (next_index - i)
    return by_ocr


def is_punctuation_only(text: str) -> bool:
    return not re.search(r"[\u1000-\u109F]", text) or all(
        ch in "။၊.:-–—()[] 0123456789၀၁၂၃၄၅၆၇၈၉ " for ch in text
    )


def heading_line_score(lines: list[VisualLine], index: int) -> tuple[bool, str]:
    line = lines[index]
    text = line.text.strip()
    if len(text) < 8 or is_punctuation_only(text):
        return False, "empty_or_punctuation"
    if line.y0 < line.page_height * 0.11:
        return False, "top_running_header_band"
    if line.y0 > line.page_height * 0.90:
        return False, "footer_band"
    if line.x0 > line.page_width * 0.75 and parse_int_loose(text) is not None:
        return False, "page_number"
    if re.match(r"^[0-9၀-၉ဝOo\s]+[။.)]", text):
        return False, "numbered_body_line"

    previous_line = lines[index - 1] if index else None
    next_line = lines[index + 1] if index + 1 < len(lines) else None
    gap_above = line.y0 - previous_line.y1 if previous_line else 999
    gap_below = next_line.y0 - line.y1 if next_line else 999
    text_signal = any(key in text for key in ["ခြင်း", "အကြောင်း", "အနွယ်", "ဖြစ်တော်စဉ်", "အတ္ထုပ္ပတ္တိ"])
    centered = line.center_deviation <= 0.075 and line.width_ratio <= 0.62
    spaced_centered = (
        line.center_deviation <= 0.11
        and line.width_ratio <= 0.58
        and (gap_above >= 55 or gap_below >= 55)
    )
    signal_centered = (
        text_signal
        and line.center_deviation <= 0.13
        and line.width_ratio <= 0.68
        and (gap_above >= 45 or gap_below >= 45)
    )
    if centered or spaced_centered or signal_centered:
        return True, (
            f"width={line.width_ratio:.3f};center_dev={line.center_deviation:.3f};"
            f"gap_above={gap_above};gap_below={gap_below};text_signal={text_signal}"
        )
    return False, (
        f"width={line.width_ratio:.3f};center_dev={line.center_deviation:.3f};"
        f"gap_above={gap_above};gap_below={gap_below};text_signal={text_signal}"
    )


def detect_heading_candidates(page_num: int, ocr_json_dir: Path) -> list[HeadingCandidate]:
    lines = extract_visual_lines(ocr_json_dir / f"page_{page_num:04d}.json", page_num)
    accepted: list[tuple[int, str]] = []
    for idx in range(len(lines)):
        ok, reason = heading_line_score(lines, idx)
        if ok:
            accepted.append((idx, reason))

    groups: list[HeadingCandidate] = []
    pending: list[tuple[int, str]] = []

    def flush() -> None:
        if not pending:
            return
        group_lines = [lines[idx] for idx, _reason in pending]
        text = " ".join(line.text for line in group_lines)
        groups.append(
            HeadingCandidate(
                page_num=page_num,
                line_start=group_lines[0].line_index,
                line_end=group_lines[-1].line_index,
                text=re.sub(r"\s+", " ", text).strip(),
                x0=min(line.x0 for line in group_lines),
                y0=min(line.y0 for line in group_lines),
                x1=max(line.x1 for line in group_lines),
                y1=max(line.y1 for line in group_lines),
                score_reason=" | ".join(reason for _idx, reason in pending),
            )
        )
        pending.clear()

    for idx, reason in accepted:
        if pending and idx - pending[-1][0] > 2:
            flush()
        pending.append((idx, reason))
    flush()
    return groups


def similarity(a: str, b: str) -> float:
    aa = normalize_for_match(a)
    bb = normalize_for_match(b)
    if not aa or not bb:
        return 0.0
    if aa in bb or bb in aa:
        return min(len(aa), len(bb)) / max(len(aa), len(bb))
    grams_a = {aa[i : i + 3] for i in range(max(1, len(aa) - 2))}
    grams_b = {bb[i : i + 3] for i in range(max(1, len(bb) - 2))}
    if not grams_a or not grams_b:
        return 0.0
    return len(grams_a & grams_b) / len(grams_a | grams_b)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_crops(
    out_dir: Path, image_dir: Path, rows: list[dict[str, Any]], padding: int = 120
) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    crop_dir = out_dir / "review_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row.get("heading_x0") == "":
            continue
        image_path = image_dir / f"page_{int(row['ocr_page_num']):04d}.png"
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            x0 = max(0, int(row["heading_x0"]) - padding)
            y0 = max(0, int(row["heading_y0"]) - padding)
            x1 = min(image.width, int(row["heading_x1"]) + padding)
            y1 = min(image.height, int(row["heading_y1"]) + padding)
            crop = image.crop((x0, y0, x1, y1))
            draw = ImageDraw.Draw(crop)
            draw.rectangle(
                (
                    int(row["heading_x0"]) - x0,
                    int(row["heading_y0"]) - y0,
                    int(row["heading_x1"]) - x0,
                    int(row["heading_y1"]) - y0,
                ),
                outline="red",
                width=6,
            )
            crop_name = (
                f"section_{int(row['section_no']):04d}_page_{int(row['ocr_page_num']):04d}.png"
            )
            crop.save(crop_dir / crop_name)
            row["review_crop"] = str(crop_dir / crop_name)


def _resample_filter() -> Any:
    try:
        from PIL import Image

        return Image.Resampling.LANCZOS
    except AttributeError:
        from PIL import Image

        return Image.LANCZOS


def _row_review_label(row: dict[str, Any]) -> str:
    return (
        f"section {int(row['section_no']):03d} | "
        f"ocr page {int(row['ocr_page_num']):04d} | "
        f"printed {row['toc_printed_page']} | "
        f"score {row['heading_match_score']}"
    )


def _has_heading_box(row: dict[str, Any]) -> bool:
    return all(
        row.get(key) != "" for key in ("heading_x0", "heading_y0", "heading_x1", "heading_y1")
    )


def maybe_write_review_contact_sheets(
    out_dir: Path, image_dir: Path, rows: list[dict[str, Any]]
) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return

    sheet_dir = out_dir / "review_contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    resample = _resample_filter()

    def write_sheets(kind: str, tiles: list[Any], cols: int) -> list[str]:
        paths: list[str] = []
        if not tiles:
            return paths
        for sheet_idx, start in enumerate(range(0, len(tiles), cols * 5), start=1):
            chunk = tiles[start : start + cols * 5]
            tile_w = max(tile.width for tile in chunk)
            tile_h = max(tile.height for tile in chunk)
            rows_needed = (len(chunk) + cols - 1) // cols
            sheet = Image.new("RGB", (tile_w * cols, tile_h * rows_needed), "white")
            for idx, tile in enumerate(chunk):
                x = (idx % cols) * tile_w
                y = (idx // cols) * tile_h
                sheet.paste(tile, (x, y))
            path = sheet_dir / f"{kind}_sheet_{sheet_idx:03d}.png"
            sheet.save(path)
            paths.append(str(path))
        return paths

    full_page_tiles: list[Any] = []
    crop_tiles: list[Any] = []
    for row in rows:
        image_path = image_dir / f"page_{int(row['ocr_page_num']):04d}.png"
        if not image_path.exists():
            continue
        with Image.open(image_path) as original:
            image = original.convert("RGB")

            label_h = 42
            thumb_w = 420
            thumb_h = round(image.height * (thumb_w / image.width))
            full_tile = Image.new("RGB", (thumb_w, label_h + thumb_h), "white")
            full_draw = ImageDraw.Draw(full_tile)
            full_draw.text((8, 8), _row_review_label(row), fill="black")
            thumb = image.resize((thumb_w, thumb_h), resample)
            full_tile.paste(thumb, (0, label_h))
            if _has_heading_box(row):
                sx = thumb_w / image.width
                sy = thumb_h / image.height
                full_draw.rectangle(
                    (
                        int(row["heading_x0"]) * sx,
                        label_h + int(row["heading_y0"]) * sy,
                        int(row["heading_x1"]) * sx,
                        label_h + int(row["heading_y1"]) * sy,
                    ),
                    outline="red",
                    width=3,
                )
            full_page_tiles.append(full_tile)

            if _has_heading_box(row):
                pad = 260
                x0 = max(0, int(row["heading_x0"]) - pad)
                y0 = max(0, int(row["heading_y0"]) - pad)
                x1 = min(image.width, int(row["heading_x1"]) + pad)
                y1 = min(image.height, int(row["heading_y1"]) + pad)
                crop = image.crop((x0, y0, x1, y1))
                crop_draw = ImageDraw.Draw(crop)
                crop_draw.rectangle(
                    (
                        int(row["heading_x0"]) - x0,
                        int(row["heading_y0"]) - y0,
                        int(row["heading_x1"]) - x0,
                        int(row["heading_y1"]) - y0,
                    ),
                    outline="red",
                    width=6,
                )
                crop.thumbnail((900, 420), resample)
                crop_label_h = 38
                crop_tile = Image.new("RGB", (900, crop_label_h + 420), "white")
                crop_tile_draw = ImageDraw.Draw(crop_tile)
                crop_tile_draw.text((8, 8), _row_review_label(row), fill="black")
                crop_tile.paste(crop, (0, crop_label_h))
                crop_tiles.append(crop_tile)

    full_paths = write_sheets("full_pages", full_page_tiles, cols=4)
    crop_paths = write_sheets("heading_crops", crop_tiles, cols=2)
    for row in rows:
        row["review_contact_sheets_dir"] = str(sheet_dir)
    write_json(
        sheet_dir / "contact_sheet_manifest.json",
        {
            "full_pages": full_paths,
            "heading_crops": crop_paths,
        },
    )


def run(args: argparse.Namespace) -> None:
    volume_id = args.volume_id
    root = Path(args.ocr_root)
    valid_root = Path(args.valid_root)
    out_dir = Path(args.out_dir)
    ocr_json_dir = root / "data" / "ocr_json" / volume_id
    image_dir = root / "data" / "rendered_pages" / volume_id
    valid_page_nums = [
        int(path.stem.split("_", 1)[1])
        for path in sorted((valid_root / volume_id / "pages").glob("page_*.txt"))
    ]
    valid_page_set = set(valid_page_nums)

    toc_entries = parse_toc_entries(ocr_json_dir, list(range(args.toc_start, args.toc_end + 1)))
    printed_by_ocr = build_printed_page_map(valid_page_nums, ocr_json_dir)
    ocr_by_printed: dict[int, int] = {}
    for ocr_page, printed_page in printed_by_ocr.items():
        ocr_by_printed.setdefault(printed_page, ocr_page)

    heading_by_page = {
        page_num: detect_heading_candidates(page_num, ocr_json_dir) for page_num in valid_page_nums
    }

    rows: list[dict[str, Any]] = []
    for entry in toc_entries:
        if entry.printed_page is None:
            predicted_page = None
        else:
            predicted_page = ocr_by_printed.get(entry.printed_page)
        if predicted_page not in valid_page_set:
            continue

        search_pages = [predicted_page]
        if predicted_page - 1 in valid_page_set:
            search_pages.append(predicted_page - 1)
        if predicted_page + 1 in valid_page_set:
            search_pages.append(predicted_page + 1)

        candidates: list[tuple[float, HeadingCandidate]] = []
        for page_num in search_pages:
            for heading in heading_by_page.get(page_num, []):
                candidates.append((similarity(entry.title, heading.text), heading))
        candidates.sort(
            key=lambda pair: (pair[0], -abs(pair[1].page_num - predicted_page)), reverse=True
        )
        best_score, best_heading = candidates[0] if candidates else (0.0, None)

        row = {
            "section_no": entry.section_no,
            "toc_title": entry.title,
            "toc_printed_page": entry.printed_page if entry.printed_page is not None else "",
            "toc_printed_page_raw": entry.printed_page_raw,
            "toc_page_num": entry.toc_page_num,
            "ocr_page_num": predicted_page,
            "in_valid_range": predicted_page in valid_page_set,
            "image_path": str(image_dir / f"page_{predicted_page:04d}.png")
            if predicted_page
            else "",
            "heading_match_score": round(best_score, 4),
            "heading_text": best_heading.text if best_heading else "",
            "heading_line_start": best_heading.line_start if best_heading else "",
            "heading_line_end": best_heading.line_end if best_heading else "",
            "heading_x0": best_heading.x0 if best_heading else "",
            "heading_y0": best_heading.y0 if best_heading else "",
            "heading_x1": best_heading.x1 if best_heading else "",
            "heading_y1": best_heading.y1 if best_heading else "",
            "heading_reason": best_heading.score_reason if best_heading else "",
            "review_crop": "",
        }
        rows.append(row)

    maybe_write_crops(out_dir, image_dir, rows)
    maybe_write_review_contact_sheets(out_dir, image_dir, rows)
    write_csv(out_dir / f"{volume_id}_section_break_predictions.csv", rows)
    write_json(out_dir / f"{volume_id}_section_break_predictions.json", rows)
    write_json(out_dir / f"{volume_id}_toc_entries.json", [asdict(entry) for entry in toc_entries])
    write_json(out_dir / f"{volume_id}_printed_page_map.json", printed_by_ocr)
    write_json(
        out_dir / f"{volume_id}_all_heading_candidates.json",
        {
            str(page_num): [asdict(item) for item in items]
            for page_num, items in heading_by_page.items()
            if items
        },
    )
    summary = {
        "volume_id": volume_id,
        "valid_pages": len(valid_page_nums),
        "toc_entries": len(toc_entries),
        "predicted_breaks_in_valid_range": len(rows),
        "with_heading_candidate": sum(1 for row in rows if row["heading_text"]),
        "with_strong_heading_match_score_ge_0_25": sum(
            1 for row in rows if float(row["heading_match_score"]) >= 0.25
        ),
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministically compute Konbaung section break candidates."
    )
    parser.add_argument("--volume-id", default="konbaung_vol1")
    parser.add_argument("--ocr-root", default="konbaung-google-ocr")
    parser.add_argument("--valid-root", default="konbaung_viable_pages_manual_ranges")
    parser.add_argument("--out-dir", default="konbaung_section_breaks_vol1_review")
    parser.add_argument("--toc-start", type=int, default=9)
    parser.add_argument("--toc-end", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

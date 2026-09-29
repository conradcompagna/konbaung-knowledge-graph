#!/usr/bin/env python3
"""Calculate evidence coverage after conservative removal of page paratext."""

import json
import re
from pathlib import Path

from konbaung_gemini_remainder_completion_annotator import utf16_to_codepoint


ROOT = Path("konbaung_reader_app/static/data/konbaung/pages")
HEADER_RE = re.compile(r"(မဟာရာဇဝင်တော်ကြီး|ဖြစ်တော်စဉ်|မြို့တည်နန်းတည်\s+မင်းတရားကြီး|နန်းစံ\s+မင်းတရားကြီး)")
NUMBER_RE = re.compile(r"^[၀-၉0-9]+$")
FOOTNOTE_RE = re.compile(r"^[၁၂၃၄၅၆၇၈၉၀0-9]+\s*[။.]")
PUBLISHER_RE = re.compile(r"ရာပြည့်.*(?:စာ)?အုပ်တိုင်|ရာပြည့်.*(?:စာ)?အုပ်တိုက်|ရာပြည့်စာအုပ်တို့၏")


def line_ranges(text: str) -> list[tuple[int, int, str]]:
    lines = []
    position = 0
    for raw in text.splitlines(keepends=True):
        body = raw.rstrip("\r\n")
        lines.append((position, position + len(body), body))
        position += len(raw)
    return lines


def debris_positions(text: str) -> set[int]:
    lines = line_ranges(text)
    nonempty = [index for index, (_, _, line) in enumerate(lines) if line.strip()]
    debris: set[int] = set()
    for index in nonempty[:3]:
        start, end, line = lines[index]
        stripped = line.strip()
        if HEADER_RE.search(stripped) or NUMBER_RE.fullmatch(stripped):
            debris.update(range(start, end))

    publisher_lines = [
        index for index, (_, _, line) in enumerate(lines) if PUBLISHER_RE.search(line)
    ]
    if publisher_lines:
        publisher = publisher_lines[-1]
        start_line = publisher
        for index in range(max(0, publisher - 5), publisher):
            if FOOTNOTE_RE.match(lines[index][2].strip()):
                start_line = min(start_line, index)
        for index in range(start_line, len(lines)):
            debris.update(range(lines[index][0], lines[index][1]))
    return debris


def main() -> None:
    rows = []
    for volume in (1, 2, 3):
        for path in sorted((ROOT / f"vol{volume}").glob("*.json")):
            page = json.loads(path.read_text(encoding="utf-8"))
            text = page["canonicalText"]
            debris = debris_positions(text)
            evidence: set[int] = set()
            for annotation in page["annotations"]:
                start = utf16_to_codepoint(text, int(annotation["evidence"]["startUtf16"]))
                end = utf16_to_codepoint(text, int(annotation["evidence"]["endUtf16"]))
                evidence.update(range(start, end))
            retained = [i for i, char in enumerate(text) if not char.isspace() and i not in debris]
            uncovered = [i for i in retained if i not in evidence]
            rows.append(
                {
                    "volume": volume,
                    "page": int(path.stem),
                    "retained_characters": len(retained),
                    "unannotated_characters": len(uncovered),
                    "unannotated_pct": 100 * len(uncovered) / len(retained) if retained else 0,
                }
            )

    selected = [row for row in rows if row["unannotated_pct"] > 10]
    output = Path("konbaung_annotation_coverage_over_10pct_debris_adjusted.json")
    output.write_text(
        json.dumps(
            {
                "method": "Non-whitespace text after conservative deterministic removal of running headers, standalone page numbers, publisher lines, trailing numbered citations, and post-publisher OCR debris.",
                "pages": selected,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for volume in (1, 2, 3):
        volume_rows = [row for row in rows if row["volume"] == volume]
        volume_selected = [row for row in selected if row["volume"] == volume]
        print(
            f"vol{volume}: {len(volume_selected)}/{len(volume_rows)} "
            f"({100 * len(volume_selected) / len(volume_rows):.2f}%)"
        )
    print(f"all: {len(selected)}/{len(rows)} ({100 * len(selected) / len(rows):.2f}%)")
    print(output.resolve())


if __name__ == "__main__":
    main()

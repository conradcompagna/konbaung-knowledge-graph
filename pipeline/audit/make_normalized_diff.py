#!/usr/bin/env python3
"""Create reviewable normalized copies of the Konbaung OCR volumes.

This utility is intentionally copy-safe:
- original OCR text files are copied into the output directory first;
- all normalization and diff outputs are derived from those copied inputs;
- source files under konbaung-google-ocr are never modified.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import konbaung_gemini_kg_annotator as kg


DEFAULT_INPUTS = [
    Path("konbaung-google-ocr/data/ocr_text/konbaung_vol1/konbaung_vol1_full.txt"),
    Path("konbaung-google-ocr/data/ocr_text/konbaung_vol2/konbaung_vol2_full.txt"),
    Path("konbaung-google-ocr/data/ocr_text/konbaung_vol3/konbaung_vol3_full.txt"),
]


@dataclass
class DeletedLine:
    reason: str
    raw_line: str
    normalized_line: str
    line_number: int


@dataclass
class ReviewPage:
    volume_id: str
    page_num: int
    raw_text: str
    kept_lines: list[str]
    deleted_lines: list[DeletedLine]
    clean_text: str
    drop_reason: Optional[str]


def deletion_reason(raw_line: str, normalized_line: str, page_num: int) -> Optional[str]:
    """Return the line-level deletion reason used by the normalizer."""
    if not normalized_line:
        return "blank_or_zero_width"
    if normalized_line.startswith("[[BOOK="):
        return "page_marker"
    if kg.is_standalone_page_number(normalized_line, page_num):
        return "standalone_page_number"
    if any(p.search(normalized_line) for p in kg.FOOTER_PATTERNS):
        return "footer_or_printer_noise"
    if kg.mostly_latin_noise(normalized_line):
        return "latin_ocr_noise"
    if kg.looks_like_footnote_line(normalized_line):
        return "footnote_or_date_line"
    # Drop OCR-inserted footnote date clusters such as "၁။ ၁၈ ... ၂။ ..." at bottoms.
    ascii_line = normalized_line.translate(
        str.maketrans(kg.MYANMAR_DIGITS, kg.ASCII_DIGITS)
    ).replace("ဝ", "0")
    if (
        normalized_line.count("။") >= 2
        and re.search(r"[၁၂၃၄၅၆၇၈၉]\s*။", normalized_line)
        and re.search(r"1[6789]\d{2}|20\d{2}", ascii_line)
    ):
        return "footnote_date_cluster"
    return None


def join_kept_lines(cleaned: list[str]) -> str:
    """Match kg.clean_page_lines paragraph joining while keeping tracking separate."""
    paragraphs: list[str] = []
    buf = ""
    for line in cleaned:
        if not buf:
            buf = line
            continue
        prev_ends_sentence = buf.endswith(("။", "?", "!”", "။”", "။'", "။။"))
        line_is_headingish = len(line) < 55 and not line.endswith(
            ("သည်", "၏", "၍", "ပြီး", "ရာ", "လျှင်")
        )
        if prev_ends_sentence and line_is_headingish:
            paragraphs.append(buf)
            buf = line
        else:
            sep = "" if buf.endswith(("-", "၊")) else " "
            buf = buf + sep + line
    if buf:
        paragraphs.append(buf)

    text = "\n".join(paragraphs)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tracked_clean_page(raw_text: str, page_num: int) -> tuple[list[str], list[DeletedLine], str]:
    kept: list[str] = []
    deleted: list[DeletedLine] = []
    for idx, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = kg.normalize_line(raw_line)
        reason = deletion_reason(raw_line, line, page_num)
        if reason:
            deleted.append(
                DeletedLine(
                    reason=reason,
                    raw_line=raw_line,
                    normalized_line=line,
                    line_number=idx,
                )
            )
            continue
        kept.append(line)
    return kept, deleted, join_kept_lines(kept)


def parse_review_volume(
    path: Path,
    *,
    drop_front_matter: bool,
    drop_appendix: bool,
    body_starts: dict[str, int],
) -> list[ReviewPage]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    matches = list(kg.PAGE_RE.finditer(raw))
    if not matches:
        raise ValueError(f"No page markers found in {path}")

    pages: list[ReviewPage] = []
    in_back_matter = False
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        volume_id = match.group("book")
        page_num = int(match.group("page"))
        raw_text = raw[start:end]
        kept_lines, deleted_lines, clean_text = tracked_clean_page(raw_text, page_num)
        kg_page = kg.Page(
            volume_id=volume_id,
            page_num=page_num,
            raw_text=raw_text,
            clean_text=clean_text,
        )
        body_start = body_starts.get(volume_id, 23)
        drop_reason = kg.classify_page(
            kg_page,
            drop_front_matter=drop_front_matter,
            drop_appendix=drop_appendix,
            body_start=body_start,
        )
        if in_back_matter and drop_reason is None:
            drop_reason = "back_matter_after_index_or_picture_index"
        if drop_reason == "index_or_picture_index":
            in_back_matter = True
        pages.append(
            ReviewPage(
                volume_id=volume_id,
                page_num=page_num,
                raw_text=raw_text,
                kept_lines=kept_lines,
                deleted_lines=deleted_lines,
                clean_text=clean_text,
                drop_reason=drop_reason,
            )
        )
    return pages


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def copy_inputs(inputs: list[Path], source_copy_dir: Path) -> list[Path]:
    source_copy_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in inputs:
        if not src.exists():
            raise FileNotFoundError(src)
        dest = source_copy_dir / src.name
        shutil.copy2(src, dest)
        copied.append(dest)
    return copied


def reset_derived_outputs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_out = out_dir.resolve()
    for name in ("normalized_text", "normalized_pages", "diff_html", "reports"):
        target = (out_dir / name).resolve()
        if resolved_out not in target.parents:
            raise RuntimeError(f"Refusing to remove path outside output directory: {target}")
        if target.exists():
            shutil.rmtree(target)


def normalized_volume_text(pages: list[ReviewPage]) -> str:
    parts: list[str] = []
    for page in pages:
        if page.drop_reason or not page.clean_text.strip():
            continue
        parts.append(
            f"[[BOOK={page.volume_id} PAGE={page.page_num:04d}]]\n{page.clean_text}".rstrip()
        )
    return "\n\n".join(parts).rstrip() + "\n"


def deleted_display_text(text: str) -> str:
    """Capslock-style display for deleted text while preserving Burmese codepoints."""
    return text.upper()


def write_volume_html(path: Path, pages: list[ReviewPage]) -> None:
    volume_id = pages[0].volume_id if pages else path.stem
    removed_pages = [p for p in pages if p.drop_reason]
    kept_pages = [p for p in pages if not p.drop_reason]
    doc: list[str] = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{html.escape(volume_id)} Normalization Diff</title>",
        "<style>",
        "body{font-family:'Myanmar Text','Noto Sans Myanmar',Arial,sans-serif;margin:24px;line-height:1.5;color:#1f2328;background:#fff;}",
        "h1{font-size:24px;margin:0 0 12px;} h2{font-size:18px;margin:0 0 8px;} h3{font-size:14px;margin:12px 0 6px;}",
        ".summary{border:1px solid #d0d7de;padding:12px;margin:12px 0 20px;background:#f6f8fa;}",
        ".page{border-top:2px solid #d0d7de;padding:18px 0;} .page.removed{background:#fff8f8;padding-left:10px;padding-right:10px;}",
        ".grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;}",
        "pre{white-space:pre-wrap;word-break:break-word;border:1px solid #d0d7de;padding:10px;background:#fff;margin:0;font-family:'Myanmar Text','Noto Sans Myanmar',Consolas,monospace;font-size:13px;}",
        ".deleted{color:#b00000;text-transform:uppercase;font-family:Impact,'Arial Black','Myanmar Text','Noto Sans Myanmar',sans-serif;font-weight:900;}",
        ".deleted-block{border-color:#ffb3b3;background:#fff3f3;}",
        ".label{font-family:Impact,'Arial Black',Arial,sans-serif;font-weight:900;color:#b00000;text-transform:uppercase;}",
        ".meta{color:#57606a;font-size:13px;margin-bottom:8px;}",
        ".removed-banner{color:#b00000;font-family:Impact,'Arial Black',Arial,sans-serif;font-weight:900;text-transform:uppercase;margin:8px 0;}",
        "table{border-collapse:collapse;width:100%;font-size:13px;} td,th{border:1px solid #d0d7de;padding:4px 6px;text-align:left;}",
        "@media(max-width:900px){.grid{grid-template-columns:1fr;}}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(volume_id)} Normalization Diff</h1>",
        '<div class="summary">',
        f"<strong>Total pages:</strong> {len(pages)} &nbsp; ",
        f"<strong>Kept pages:</strong> {len(kept_pages)} &nbsp; ",
        f"<strong>Full pages removed:</strong> {len(removed_pages)}",
        "<h3>Full Pages Removed</h3>",
    ]
    if removed_pages:
        doc.append(
            "<table><thead><tr><th>Page</th><th>Reason</th><th>Clean chars before page removal</th></tr></thead><tbody>"
        )
        for p in removed_pages:
            doc.append(
                "<tr>"
                f"<td>{p.page_num:04d}</td>"
                f"<td>{html.escape(p.drop_reason or '')}</td>"
                f"<td>{len(p.clean_text)}</td>"
                "</tr>"
            )
        doc.append("</tbody></table>")
    else:
        doc.append("<p>No full pages removed.</p>")
    doc.append("</div>")

    for page in pages:
        page_class = "page removed" if page.drop_reason else "page"
        doc.append(f'<section class="{page_class}" id="page-{page.page_num:04d}">')
        doc.append(f"<h2>PAGE {page.page_num:04d}</h2>")
        if page.drop_reason:
            doc.append(
                f'<div class="removed-banner">FULL PAGE REMOVED FROM NORMALIZED COPY: '
                f"{html.escape(page.drop_reason)}</div>"
            )
        doc.append(
            f'<div class="meta">Kept lines after line cleaning: {len(page.kept_lines)}; '
            f"line-level deletions: {len(page.deleted_lines)}; clean chars: {len(page.clean_text)}</div>"
        )
        doc.append('<div class="grid">')
        doc.append("<div><h3>Normalized Copy Text For This Page</h3>")
        normalized_shown = "" if page.drop_reason else page.clean_text
        doc.append(f"<pre>{html.escape(normalized_shown)}</pre></div>")
        doc.append('<div><h3><span class="label">Deleted From This Page</span></h3>')
        deleted_chunks: list[str] = []
        if page.drop_reason and page.clean_text.strip():
            deleted_chunks.append(
                f"FULL PAGE REMOVED ({page.drop_reason})\n" + deleted_display_text(page.clean_text)
            )
        for d in page.deleted_lines:
            shown = d.normalized_line or d.raw_line
            if not shown.strip():
                continue
            deleted_chunks.append(
                f"LINE {d.line_number} REMOVED ({d.reason})\n" + deleted_display_text(shown)
            )
        deleted_text = "\n\n".join(deleted_chunks)
        doc.append(f'<pre class="deleted deleted-block">{html.escape(deleted_text)}</pre></div>')
        doc.append("</div></section>")

    doc.extend(["</body>", "</html>"])
    write_text(path, "\n".join(doc))


def write_reports(out_dir: Path, all_pages: list[ReviewPage], copied_inputs: list[Path]) -> None:
    removed_jsonl = out_dir / "reports/full_pages_removed.jsonl"
    deletion_jsonl = out_dir / "reports/line_deletions.jsonl"
    page_csv = out_dir / "reports/page_summary.csv"

    removed_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with removed_jsonl.open("w", encoding="utf-8", newline="\n") as f:
        for p in all_pages:
            if not p.drop_reason:
                continue
            f.write(
                json.dumps(
                    {
                        "volume_id": p.volume_id,
                        "page_num": p.page_num,
                        "drop_reason": p.drop_reason,
                        "raw_chars": len(p.raw_text),
                        "clean_chars_before_page_removal": len(p.clean_text),
                        "clean_preview": p.clean_text[:240],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with deletion_jsonl.open("w", encoding="utf-8", newline="\n") as f:
        for p in all_pages:
            for d in p.deleted_lines:
                f.write(
                    json.dumps(
                        {
                            "volume_id": p.volume_id,
                            "page_num": p.page_num,
                            "line_number": d.line_number,
                            "reason": d.reason,
                            "raw_line": d.raw_line,
                            "normalized_line": d.normalized_line,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    with page_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "volume_id",
                "page_num",
                "page_status",
                "drop_reason",
                "raw_chars",
                "clean_chars",
                "kept_lines",
                "deleted_lines",
            ],
        )
        writer.writeheader()
        for p in all_pages:
            writer.writerow(
                {
                    "volume_id": p.volume_id,
                    "page_num": f"{p.page_num:04d}",
                    "page_status": "removed" if p.drop_reason else "kept",
                    "drop_reason": p.drop_reason or "",
                    "raw_chars": len(p.raw_text),
                    "clean_chars": len(p.clean_text),
                    "kept_lines": len(p.kept_lines),
                    "deleted_lines": len(p.deleted_lines),
                }
            )

    by_reason: dict[str, int] = {}
    for p in all_pages:
        if p.drop_reason:
            by_reason[p.drop_reason] = by_reason.get(p.drop_reason, 0) + 1
    summary = {
        "source_copies": [str(p) for p in copied_inputs],
        "pages_total": len(all_pages),
        "pages_kept": sum(1 for p in all_pages if not p.drop_reason),
        "full_pages_removed": sum(1 for p in all_pages if p.drop_reason),
        "full_pages_removed_by_reason": by_reason,
        "line_deletions": sum(len(p.deleted_lines) for p in all_pages),
    }
    write_text(
        out_dir / "reports/summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )

    md_lines = [
        "# Full Pages Removed",
        "",
        "| Volume | Page | Reason | Clean chars before page removal |",
        "|---|---:|---|---:|",
    ]
    for p in all_pages:
        if p.drop_reason:
            md_lines.append(
                f"| {p.volume_id} | {p.page_num:04d} | {p.drop_reason} | {len(p.clean_text)} |"
            )
    write_text(out_dir / "reports/full_pages_removed.md", "\n".join(md_lines) + "\n")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    reset_derived_outputs(out_dir)
    source_copy_dir = out_dir / "source_copies"
    inputs = [Path(p) for p in args.inputs] if args.inputs else DEFAULT_INPUTS
    copied_inputs = copy_inputs(inputs, source_copy_dir)
    body_starts = kg.parse_body_starts(args.body_starts)

    all_pages: list[ReviewPage] = []
    by_volume: dict[str, list[ReviewPage]] = {}
    for copied in copied_inputs:
        pages = parse_review_volume(
            copied,
            drop_front_matter=args.drop_front_matter,
            drop_appendix=args.drop_appendix,
            body_starts=body_starts,
        )
        all_pages.extend(pages)
        if pages:
            by_volume[pages[0].volume_id] = pages

    for volume_id, pages in sorted(by_volume.items()):
        write_text(
            out_dir / "normalized_text" / f"{volume_id}_normalized.txt",
            normalized_volume_text(pages),
        )
        for page in pages:
            if page.drop_reason:
                continue
            write_text(
                out_dir / "normalized_pages" / volume_id / f"page_{page.page_num:04d}.txt",
                page.clean_text.rstrip() + "\n",
            )
        write_volume_html(out_dir / "diff_html" / f"{volume_id}_diff.html", pages)

    write_reports(out_dir, all_pages, copied_inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized Konbaung text copies and HTML deletion diffs."
    )
    parser.add_argument(
        "--inputs", nargs="*", default=None, help="Optional input full-text OCR files."
    )
    parser.add_argument(
        "--out-dir", default="konbaung_normalized_review", help="Derived output directory."
    )
    parser.add_argument("--drop-front-matter", action="store_true", default=True)
    parser.add_argument("--keep-front-matter", dest="drop_front_matter", action="store_false")
    parser.add_argument(
        "--body-starts",
        default="konbaung_vol1:41,konbaung_vol2:23,konbaung_vol3:23",
        help="Comma map of volume_id:first_body_page.",
    )
    parser.add_argument("--drop-appendix", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

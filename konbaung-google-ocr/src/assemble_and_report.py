import json
import re
from pathlib import Path

MYANMAR_RE = re.compile(r"[က-႟ꩠ-ꩿꧠ-꧿]")


def run(book_id: str):
    base = Path(__file__).resolve().parent.parent / "data"
    json_dir = base / "ocr_json" / book_id
    text_dir = base / "ocr_text" / book_id / "pages"
    text_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = base / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    page_files = sorted(json_dir.glob("page_*.json"))

    full_text_parts = []
    rows = []

    for pf in page_files:
        page_num = int(pf.stem.split("_")[1])
        data = json.loads(pf.read_text(encoding="utf-8"))
        response = data.get("responses", [{}])[0]
        text = response.get("fullTextAnnotation", {}).get("text", "")
        has_error = "error" in response

        page_txt_path = text_dir / f"page_{page_num:04d}.txt"
        page_txt_path.write_text(text, encoding="utf-8")

        full_text_parts.append(f"[[BOOK={book_id} PAGE={page_num:04d}]]\n{text}\n")

        char_count = len(text)
        myanmar_chars = len(MYANMAR_RE.findall(text))
        myanmar_ratio = (myanmar_chars / char_count) if char_count > 0 else 0.0

        # average word confidence if available
        confidences = []
        pages = response.get("fullTextAnnotation", {}).get("pages", [])
        for p in pages:
            for block in p.get("blocks", []):
                for para in block.get("paragraphs", []):
                    for word in para.get("words", []):
                        conf = word.get("confidence")
                        if conf is not None:
                            confidences.append(conf)
        avg_conf = (sum(confidences) / len(confidences)) if confidences else None

        empty_page = char_count == 0
        suspicious = (
            has_error
            or empty_page
            or char_count < 20
            or myanmar_ratio < 0.3
            or (avg_conf is not None and avg_conf < 0.6)
        )

        rows.append(
            {
                "page_number": page_num,
                "char_count": char_count,
                "myanmar_char_count": myanmar_chars,
                "myanmar_char_ratio": round(myanmar_ratio, 3),
                "avg_word_confidence": round(avg_conf, 3) if avg_conf is not None else "",
                "empty_page": empty_page,
                "suspicious_page": suspicious,
                "has_error": has_error,
            }
        )

    full_book_path = base / "ocr_text" / book_id / f"{book_id}_full.txt"
    full_book_path.write_text("\n".join(full_text_parts), encoding="utf-8")

    # CSV report
    csv_path = reports_dir / f"{book_id}_ocr_summary.csv"
    header = list(rows[0].keys())
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r[h]) for h in header))
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    # Markdown summary
    total_pages = len(rows)
    empty_pages = sum(1 for r in rows if r["empty_page"])
    suspicious_pages = sum(1 for r in rows if r["suspicious_page"])
    total_chars = sum(r["char_count"] for r in rows)
    avg_myanmar_ratio = sum(r["myanmar_char_ratio"] for r in rows) / total_pages

    md_lines = [
        f"# OCR Quality Summary: {book_id}",
        "",
        f"- Total pages: {total_pages}",
        f"- Empty pages: {empty_pages}",
        f"- Suspicious pages (flagged): {suspicious_pages}",
        f"- Total characters extracted: {total_chars}",
        f"- Average Myanmar character ratio: {avg_myanmar_ratio:.3f}",
        "",
        "## Suspicious pages",
        "",
    ]
    for r in rows:
        if r["suspicious_page"]:
            md_lines.append(
                f"- page {r['page_number']:04d}: chars={r['char_count']}, "
                f"myanmar_ratio={r['myanmar_char_ratio']}, "
                f"avg_conf={r['avg_word_confidence']}, error={r['has_error']}"
            )

    md_path = reports_dir / f"{book_id}_ocr_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Assembled {total_pages} pages.")
    print(f"Full book text: {full_book_path}")
    print(f"CSV report: {csv_path}")
    print(f"MD report: {md_path}")
    print(f"Empty pages: {empty_pages}, Suspicious pages: {suspicious_pages}")
    print(f"Avg Myanmar ratio: {avg_myanmar_ratio:.3f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--book-id", required=True)
    args = ap.parse_args()
    run(args.book_id)

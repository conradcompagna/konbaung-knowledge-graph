import sys
import fitz  # pymupdf
from pathlib import Path

def render_pdf(pdf_path: str, book_id: str, dpi: int = 350, max_pages: int = None):
    out_dir = Path(__file__).resolve().parent.parent / "data" / "rendered_pages" / book_id
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    n_pages = len(doc)
    if max_pages:
        n_pages = min(n_pages, max_pages)

    for i in range(n_pages):
        page = doc[i]
        pix = page.get_pixmap(matrix=mat)
        out_path = out_dir / f"page_{i+1:04d}.png"
        pix.save(str(out_path))
        print(f"rendered {out_path}")

    doc.close()
    print(f"done: {n_pages} pages -> {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--book-id", required=True)
    ap.add_argument("--dpi", type=int, default=350)
    ap.add_argument("--max-pages", type=int, default=None)
    args = ap.parse_args()
    render_pdf(args.pdf, args.book_id, args.dpi, args.max_pages)

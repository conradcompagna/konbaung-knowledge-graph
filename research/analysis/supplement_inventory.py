"""Extract the archived PDF reports/atlases into the read-only search index."""

from pathlib import Path
import json, zipfile, io, hashlib
from pypdf import PdfReader

O = Path(__file__).resolve().parents[1]
data = json.loads((O / "inventory.json").read_text(encoding="utf-8"))
rows = []
for r in data:
    if r["extension"] != ".pdf":
        continue
    archive, name = r["source"].split("!/", 1)
    with zipfile.ZipFile(O.parent / archive) as z:
        raw = z.read(name)
    reader = PdfReader(io.BytesIO(raw))
    text = "\n\n".join(
        f"PDF PAGE {i + 1}\n" + (p.extract_text() or "") for i, p in enumerate(reader.pages)
    )
    h = hashlib.sha256(raw).hexdigest()
    path = O / "source_text" / (h[:20] + ".txt")
    path.write_text(text, encoding="utf-8")
    rows.append(
        dict(
            source=r["source"],
            pages=len(reader.pages),
            sha256=h,
            text_file=str(path.relative_to(O)).replace("\\", "/"),
        )
    )
(O / "pdf_inventory.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print("PDFs indexed", len(rows), "pages", sum(r["pages"] for r in rows))

"""Read-only inventory of original files and nested archives; deduplicate text by SHA256."""

from pathlib import Path
import zipfile, io, json, hashlib, csv, re, collections, xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_audit_2026-09-11"
TEXT = OUT / "source_text"
TEXT.mkdir(parents=True, exist_ok=True)
records = []
errors = []
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def doc_text(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        parts = []
        for name in z.namelist():
            if name == "word/document.xml" or re.match(
                r"word/(footnotes|endnotes|comments|header\d+|footer\d+)\.xml$", name
            ):
                root = ET.fromstring(z.read(name))
                # Visible, inserted text; omit deleted runs in tracked revisions.
                paragraphs = [
                    "".join(t.text or "" for t in p.findall(".//w:t", NS))
                    for p in root.findall(".//w:p", NS)
                ]
                parts.append(name + "\n" + "\n".join(paragraphs))
        return "\n\n".join(parts)


def consume(source, size, getbytes, depth=0):
    ext = Path(source.split("!/")[-1]).suffix.lower()
    rec = {
        "source": source,
        "size": size,
        "extension": ext,
        "text_file": "",
        "sha256": "",
        "preview": "",
    }
    records.append(rec)
    try:
        if ext == ".zip" and depth < 5:
            with zipfile.ZipFile(io.BytesIO(getbytes())) as z:
                for info in z.infolist():
                    if not info.is_dir():
                        consume(
                            source + "!/" + info.filename,
                            info.file_size,
                            lambda n=info.filename: z.read(n),
                            depth + 1,
                        )
        elif ext in {
            ".docx",
            ".txt",
            ".md",
            ".py",
            ".r",
            ".ipynb",
            ".json",
            ".yaml",
            ".yml",
            ".html",
            ".csv",
            ".tsv",
        }:
            raw = getbytes()
            rec["sha256"] = hashlib.sha256(raw).hexdigest()
            if ext == ".docx":
                txt = doc_text(raw)
            elif ext in {".csv", ".tsv"}:
                txt = raw.decode("utf-8-sig", errors="replace")
                reader = csv.reader(io.StringIO(txt), delimiter="\t" if ext == ".tsv" else ",")
                rows = []
                for i, row in enumerate(reader):
                    if i < 3:
                        rows.append(row)
                rec["rows"] = i if txt else 0
                rec["columns"] = json.dumps(rows[0] if rows else [], ensure_ascii=False)
                txt = json.dumps(
                    {"rows": rec.get("rows"), "sample": rows}, ensure_ascii=False, indent=2
                )
            else:
                txt = raw.decode("utf-8-sig", errors="replace")
            h = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:20]
            target = TEXT / (h + ".txt")
            if not target.exists():
                target.write_text(txt, encoding="utf-8")
            rec["text_file"] = str(target.relative_to(OUT)).replace("\\", "/")
            rec["preview"] = txt[:200].replace("\n", " ")
    except Exception as e:
        errors.append({"source": source, "error": repr(e)})


for p in sorted(ROOT.iterdir()):
    if p.is_file() and not p.name.startswith("~$"):
        consume(p.name, p.stat().st_size, p.read_bytes)

(OUT / "inventory.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
)
fields = sorted(set().union(*(r.keys() for r in records)))
with (OUT / "inventory.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(records)
(OUT / "inventory_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
print(
    json.dumps(
        {
            "items": len(records),
            "extensions": dict(collections.Counter(r["extension"] for r in records)),
            "unique_texts": len(list(TEXT.glob("*.txt"))),
            "errors": errors,
        },
        indent=2,
    )
)
print("\nDATA / EMBEDDING / CODE CANDIDATES")
for r in records:
    if re.search(
        r"embedd|\.np[yz]$|\.parquet$|\.pkl$|\.py$|\.r$|triple.*\.(json|csv)$|axial|mapping|cluster",
        r["source"],
        re.I,
    ):
        print(f"{r['size']:>11} {r['source']} ({r.get('rows', '')} rows)")

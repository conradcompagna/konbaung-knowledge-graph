#!/usr/bin/env python3
"""Package canonical Konbaung pages and grounded annotations for manual review."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_READER_DATA = ROOT / "konbaung_reader_app" / "static" / "data" / "konbaung"
DEFAULT_BUILD_DIR = ROOT / "konbaung_manual_review_bundle"
DEFAULT_ZIP = ROOT / "konbaung_manual_review_bundle.zip"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("date", "location", "quantity"):
        value = metadata.get(key) or {}
        if value.get("originalText") or value.get("gloss"):
            result[key] = {
                "my": value.get("originalText"),
                "en": value.get("gloss"),
            }
    return result


def compact_triple(annotation: dict[str, Any]) -> dict[str, Any]:
    subject = annotation["subject"]
    obj = annotation["object"]
    record: dict[str, Any] = {
        "id": annotation["id"],
        "pass": annotation["provenance"]["pass"],
        "evidence_my": annotation["evidence"]["canonicalText"],
        "subject": {
            "my": subject["text"],
            "en": subject.get("gloss"),
            "tag": subject.get("type"),
            "inferred": bool(subject.get("inferred")),
        },
        "relation": annotation["relation"]["rawLabel"],
        "object": {
            "my": obj["text"],
            "en": obj.get("gloss"),
            "tag": obj.get("type"),
            "inferred": bool(obj.get("inferred")),
        },
    }
    record.update(compact_metadata(annotation.get("metadata") or {}))
    return record


def build(reader_data: Path, build_dir: Path, zip_path: Path) -> dict[str, Any]:
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    page_paths = sorted((reader_data / "pages").glob("vol*/*.json"))
    if not page_paths:
        raise FileNotFoundError(f"No compiled page records under {reader_data}")

    manifest_pages: list[dict[str, Any]] = []
    pass_names = ("first_pass", "second_pass", "third_pass", "fourth_pass")
    totals = {"pages": 0, "triples": 0, **{name: 0 for name in pass_names}}
    for page_path in page_paths:
        page = json.loads(page_path.read_text(encoding="utf-8"))
        volume = page["volumeId"]
        number = int(page["pageNumber"])
        stem = f"page_{number:04d}"
        output_dir = build_dir / volume
        canonical_text = page["canonicalText"]
        summary = page.get("summary") or ""
        triples = [compact_triple(item) for item in page["annotations"]]

        write_text(output_dir / f"{stem}.txt", canonical_text)
        write_text(output_dir / f"{stem}_summary.txt", summary + ("\n" if summary else ""))
        write_json(output_dir / f"{stem}_triples.json", triples)

        pass_counts = {
            name: sum(item["pass"] == name for item in triples)
            for name in pass_names
        }
        totals["pages"] += 1
        totals["triples"] += len(triples)
        for name, count in pass_counts.items():
            totals[name] += count
        manifest_pages.append(
            {
                "volume": volume,
                "page": number,
                "triples": len(triples),
                **pass_counts,
                "text_sha1": hashlib.sha1(canonical_text.encode("utf-8")).hexdigest(),
            }
        )

    readme = """# Konbaung Chronicle Manual Review Bundle

Each volume directory contains three UTF-8 files per source page:

- `page_NNNN.txt`: exact canonical OCR page text.
- `page_NNNN_summary.txt`: Gemini first-pass interpretive summary.
- `page_NNNN_triples.json`: all first-through-fourth-pass triples with canonical Burmese evidence blocks, Burmese/English entities, entity tags, inference flags, and all supplied date/location/quantity metadata.

The `pass` field records `first_pass`, `second_pass`, `third_pass`, or `fourth_pass`. An `inferred` entity may be implicit in the chronicle's royal or narrative grammar rather than appearing inside its evidence block. Originals were read only and were not modified.
"""
    write_text(build_dir / "README.md", readme)
    manifest = {"totals": totals, "pages": manifest_pages}
    write_json(build_dir / "manifest.json", manifest)

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(build_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(build_dir.parent))

    return {
        **totals,
        "build_dir": str(build_dir),
        "zip_path": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader-data", type=Path, default=DEFAULT_READER_DATA)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    args = parser.parse_args()
    result = build(args.reader_data.resolve(), args.build_dir.resolve(), args.zip.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

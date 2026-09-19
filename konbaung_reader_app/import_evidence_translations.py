#!/usr/bin/env python3
"""Copy canonical evidence translations into the reader's static data tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = APP_ROOT.parent / "konbaung_evidence_translation_full_batch_20260713" / "pages"
DEFAULT_DESTINATION = APP_ROOT / "static" / "data" / "konbaung" / "evidence_translations"


def run(source: Path, destination: Path) -> dict[str, int]:
    counts = {"pages": 0, "translations": 0}
    for result_path in sorted(source.glob("vol*/page_*/result.json")):
        volume_id = result_path.parent.parent.name
        page_number = int(result_path.parent.name.removeprefix("page_"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        translations = result.get("response", {}).get("translations", [])
        payload = {
            "pageId": f"{volume_id}-p{page_number:04d}",
            "translations": [
                {"id": str(item["id"]), "translation": str(item["translation"])}
                for item in translations
            ],
        }
        output_path = destination / volume_id / f"{page_number:04d}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        counts["pages"] += 1
        counts["translations"] += len(payload["translations"])
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()

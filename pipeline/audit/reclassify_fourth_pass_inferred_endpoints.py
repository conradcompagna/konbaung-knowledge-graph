#!/usr/bin/env python3
"""Remove obsolete inferred-endpoint warnings from fourth-pass output buckets."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path("konbaung_quantitative_fourth_pass_full_batch_20260712")
ENDPOINT_WARNING = re.compile(r"^T\[\d+\]\.(?:s|o):")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    moved = 0
    retained = 0
    for validation_path in sorted((ROOT / "validation" / "invalid").glob("**/page_*.json")):
        record = json.loads(validation_path.read_text(encoding="utf-8-sig"))
        warnings = [
            warning
            for warning in record.get("validation_warnings", [])
            if not ENDPOINT_WARNING.match(str(warning))
        ]
        record["validation_warnings"] = warnings
        relative = validation_path.relative_to(ROOT / "validation" / "invalid")
        if warnings:
            write_json(validation_path, record)
            retained += 1
            continue

        valid_validation = ROOT / "validation" / "valid" / relative
        write_json(valid_validation, record)
        validation_path.unlink()
        for phase in ("raw", "postprocessed"):
            source = ROOT / phase / "invalid" / relative
            destination = ROOT / phase / "valid" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
        moved += 1
    print(json.dumps({"moved_to_valid": moved, "retained_for_review": retained}, indent=2))


if __name__ == "__main__":
    main()

import json
from pathlib import Path


ROOT = Path("konbaung_summary_gap_full_batch_20260711")


def page_records(bucket: str) -> list[dict[str, object]]:
    records = []
    for path in sorted((ROOT / "validation" / bucket).glob("**/page_*.json")):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        records.append(
            {
                "volume_id": data["volume_id"],
                "page_num": data["page_num"],
                "sha1": data["sha1"],
                "bucket": bucket,
                "warnings": data.get("validation_warnings", []),
            }
        )
    return records


def main() -> None:
    pages = page_records("valid") + page_records("invalid")
    pages.sort(key=lambda row: (str(row["volume_id"]), int(row["page_num"])))
    payload = {
        "prompt_variant": "claim_anchor_gap_analysis",
        "model": "gemini-3.1-flash-lite",
        "page_count": len(pages),
        "valid_count": sum(row["bucket"] == "valid" for row in pages),
        "invalid_review_count": sum(row["bucket"] == "invalid" for row in pages),
        "pages": pages,
    }
    (ROOT / "production_completion_index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

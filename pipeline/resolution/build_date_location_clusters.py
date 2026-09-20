import json
import re
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path("konbaung_open_coding_structured_full_batch_20260709")
INPUT_ROOT = ROOT / "postprocessed"
OUTPUT_ROOT = ROOT / "aggregate_clusters"


def normalize_gloss(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def source_info(path: Path) -> dict:
    relative = path.relative_to(INPUT_ROOT)
    return {
        "validation_status": relative.parts[0],
        "volume": relative.parts[1],
        "page": path.stem,
        "source_file": relative.as_posix(),
    }


def slim_triple(triple: dict) -> dict:
    fields = ("s", "sg", "st", "p", "o", "og", "ot", "d", "dg", "l", "lg", "q", "qg")
    return {field: triple.get(field, "") for field in fields if triple.get(field, "") != ""}


def build_clusters() -> tuple[list[dict], list[dict], dict]:
    dates = defaultdict(lambda: {"gloss_forms": set(), "burmese_spans": set(), "triples": []})
    locations = defaultdict(lambda: {"gloss_forms": set(), "burmese_spans": set(), "triples": []})
    files_seen = triples_seen = 0

    for path in sorted(INPUT_ROOT.glob("*/*/page_*.json")):
        files_seen += 1
        data = json.loads(path.read_text(encoding="utf-8"))
        source = source_info(path)
        for ordinal, triple in enumerate(data.get("T", []), start=1):
            triples_seen += 1
            record = {**source, "triple_number": ordinal, "triple": slim_triple(triple)}

            date_gloss = str(triple.get("dg", "")).strip()
            date_span = str(triple.get("d", "")).strip()
            if date_gloss:
                cluster = dates[normalize_gloss(date_gloss)]
                cluster["gloss_forms"].add(date_gloss)
                if date_span:
                    cluster["burmese_spans"].add(date_span)
                cluster["triples"].append(record)

            location_gloss = str(triple.get("lg", "")).strip()
            location_span = str(triple.get("l", "")).strip()
            if location_gloss:
                cluster = locations[normalize_gloss(location_gloss)]
                cluster["gloss_forms"].add(location_gloss)
                if location_span:
                    cluster["burmese_spans"].add(location_span)
                cluster["triples"].append(record)

    def finalize(raw: dict) -> list[dict]:
        result = []
        for key, cluster in raw.items():
            result.append({
                "cluster_key": key,
                "mention_count": len(cluster["triples"]),
                "gloss_forms": sorted(cluster["gloss_forms"], key=str.casefold),
                "burmese_spans": sorted(cluster["burmese_spans"]),
                "triples": cluster["triples"],
            })
        return sorted(result, key=lambda item: (-item["mention_count"], item["cluster_key"]))

    date_clusters = finalize(dates)
    location_clusters = finalize(locations)
    stats = {
        "files_seen": files_seen,
        "triples_seen": triples_seen,
        "date_clusters": len(date_clusters),
        "date_associated_triples": sum(item["mention_count"] for item in date_clusters),
        "location_clusters": len(location_clusters),
        "location_associated_triples": sum(item["mention_count"] for item in location_clusters),
    }
    return date_clusters, location_clusters, stats


def main() -> None:
    date_clusters, location_clusters, stats = build_clusters()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    def write_review_csv(path: Path, heading: str, clusters: list[dict]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(heading, "subject", "relation", "object"),
            )
            writer.writeheader()
            for cluster in clusters:
                label = cluster["gloss_forms"][0]
                for item in cluster["triples"]:
                    triple = item["triple"]
                    writer.writerow({
                        heading: label,
                        "subject": triple.get("sg", ""),
                        "relation": triple.get("p", ""),
                        "object": triple.get("og", ""),
                    })

    write_review_csv(OUTPUT_ROOT / "triples_by_date_compact.csv", "date", date_clusters)
    write_review_csv(OUTPUT_ROOT / "triples_by_location_compact.csv", "location", location_clusters)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

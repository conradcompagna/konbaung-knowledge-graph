from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(
    r"C:\Users\conra\Desktop\dighumproject"
    r"\konbaung_node_identity_statistical_refinement_20260725"
)
INPUT = ROOT / "all_cluster_pair_candidates.jsonl"
OUTPUT = ROOT / "ALL_CANDIDATE_MERGES_MINIMAL.md"
THRESHOLD = 0.90
EXPECTED_ROWS = 560_612
EXPECTED_MERGES = 7


def clean(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


def render(candidate: dict) -> str:
    confidence = float(candidate["modelProbability"])
    left = clean(candidate["leftCanonical"])
    right = clean(candidate["rightCanonical"])
    return f"- **{confidence:.6%}** — {left} ↔ {right}\n"


def main() -> None:
    merge_lines: list[str] = []
    nonmerge_temp = OUTPUT.with_suffix(".nonmerge.tmp")
    row_count = 0
    merge_count = 0
    previous_confidence = 1.0

    with (
        INPUT.open("r", encoding="utf-8") as source,
        nonmerge_temp.open("w", encoding="utf-8", newline="\n") as nonmerges,
    ):
        for line in source:
            candidate = json.loads(line)
            confidence = float(candidate["modelProbability"])
            if confidence > previous_confidence + 1e-15:
                raise RuntimeError("Candidate input is not sorted by descending confidence")
            previous_confidence = confidence
            row_count += 1
            rendered = render(candidate)
            if confidence >= THRESHOLD:
                merge_lines.append(rendered)
                merge_count += 1
            else:
                nonmerges.write(rendered)

    if row_count != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS:,} candidates; found {row_count:,}")
    if merge_count != EXPECTED_MERGES:
        raise RuntimeError(f"Expected {EXPECTED_MERGES} automatic merges; found {merge_count}")

    with OUTPUT.open("w", encoding="utf-8", newline="\n") as destination:
        destination.write("# Candidate merges\n\n")
        destination.write(
            "Each bullet contains only the classifier confidence and the two proposed "
            "identities. The candidates remain sorted from highest to lowest confidence.\n\n"
        )
        destination.write("## Would merge automatically\n\n")
        destination.writelines(merge_lines)
        destination.write("\n## Would not merge automatically\n\n")
        with nonmerge_temp.open("r", encoding="utf-8") as nonmerges:
            for block in iter(lambda: nonmerges.read(1024 * 1024), ""):
                destination.write(block)

    nonmerge_temp.unlink()
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "candidateBullets": row_count,
                "wouldMerge": merge_count,
                "wouldNotMerge": row_count - merge_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

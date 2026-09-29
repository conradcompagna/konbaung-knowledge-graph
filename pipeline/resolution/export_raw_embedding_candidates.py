from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(
    r"C:\Users\conra\Desktop\dighumproject"
    r"\konbaung_node_identity_raw_embedding_refinement_20260725"
)
INPUT = ROOT / "raw_embedding_candidate_pairs.jsonl"
OUTPUT = ROOT / "ALL_RAW_GEMINI_EMBEDDING_CANDIDATES.md"
EXPECTED_ROWS = 1_530_847


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: object) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def main() -> None:
    rows = 0
    previous_similarity = 1.0
    with (
        INPUT.open("r", encoding="utf-8") as source,
        OUTPUT.open("w", encoding="utf-8", newline="\n") as destination,
    ):
        destination.write("# All raw Gemini embedding candidates\n\n")
        destination.write(
            "Each bullet contains only raw entity-embedding cosine similarity and the "
            "two proposed identities. Candidates are sorted from highest to lowest "
            "similarity.\n\n"
        )
        for line in source:
            candidate = json.loads(line)
            similarity = float(candidate["embeddingSimilarity"])
            if similarity > previous_similarity + 1e-12:
                raise RuntimeError("Input candidates are not in descending similarity order")
            previous_similarity = similarity
            left = clean(candidate["leftIdentity"])
            right = clean(candidate["rightIdentity"])
            destination.write(f"- **{similarity:.6%}** — {left} ↔ {right}\n")
            rows += 1

    if rows != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS:,} candidates; wrote {rows:,}")

    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "candidateBullets": rows,
                "bytes": OUTPUT.stat().st_size,
                "sha256": sha256(OUTPUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

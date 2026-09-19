#!/usr/bin/env python3
"""Build 50-candidate pages for frequent entities and retain 20 for the rest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz import fuzz, process


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = ROOT / "DIGHUM_WEBGPT_ANALYSIS_PACKAGE_20260831_WITH_METHODOLOGY.zip"
BASE_CANDIDATES = ROOT / "konbaung_entity_resolution_candidates_20260902_v2"
DEFAULT_OUTPUT = ROOT / "konbaung_entity_resolution_candidates_20260902_v3_frequency_prioritized"
ENTITY_RECORDS = "embeddings/entity_records.jsonl"
ENTITY_FUSED_VECTORS = "similarity/features/entity_fused_vectors.npy"
FREQUENT_MENTION_THRESHOLD = 10
FREQUENT_CANDIDATES = 50
NORMAL_CANDIDATES = 20
FREQUENT_BATCH_SIZE = 128


def parse_args() -> argparse.Namespace:
    """Accept alternate source and output paths while retaining canonical defaults."""
    parser = argparse.ArgumentParser(
        description="Build frequency-prioritized entity-resolution candidate pages."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--base-candidates", type=Path, default=BASE_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expanded-threshold",
        type=int,
        default=FREQUENT_MENTION_THRESHOLD,
        help="Use exact top 50 when corpus frequency is greater than this value.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Hash a source file to preserve exact build provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(archive: zipfile.ZipFile) -> list[dict[str, Any]]:
    """Read the canonical frequency-sorted entity inventory from the ZIP archive."""
    rows: list[dict[str, Any]] = []
    with archive.open(ENTITY_RECORDS) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8") as text:
            for line_number, line in enumerate(text, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"Invalid entity record at line {line_number}: {error}"
                    ) from error
    return rows


def extract_fused_vectors(archive: zipfile.ZipFile, temporary_root: Path) -> np.ndarray:
    """Extract the fused Gemini feature matrix to temporary storage for NumPy loading."""
    vector_path = temporary_root / "entity_fused_vectors.npy"
    with archive.open(ENTITY_FUSED_VECTORS) as source:
        with vector_path.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    vectors = np.load(vector_path)
    if vectors.ndim != 2 or vectors.shape[1] != 1536:
        raise RuntimeError(f"Unexpected fused-vector shape: {vectors.shape}")
    return vectors


def normalized_character_form(value: str) -> str:
    """Normalize case and separators before order-sensitive character comparison."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def normalized_token_sorted_character_form(value: str) -> str:
    """Sort normalized tokens so word-order variants can receive full similarity."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    spaced = "".join(character if character.isalnum() else " " for character in normalized)
    return "".join(sorted(spaced.split()))


def safe_filename(entity: str, index: int, width: int) -> str:
    """Create the same stable Windows-safe filenames used by the prior archive."""
    prefix = unicodedata.normalize("NFKC", entity)
    prefix = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", prefix)
    prefix = re.sub(r"\s+", "_", prefix).strip(" ._")
    prefix = prefix[:72].rstrip(" ._") or "entity"
    return f"{prefix}__e{index:0{width}d}.csv"


def write_candidate_csv(path: Path, rows: list[tuple[str, float, float]]) -> None:
    """Write one compact candidate page using the established three-column schema."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("entity", "emb", "char"))
        for entity, embedding_score, character_score in rows:
            writer.writerow((entity, f"{embedding_score:.4f}", f"{character_score:.4f}"))


def exact_top_rows(
    entity_index: int,
    tags: list[str],
    embedding_scores: np.ndarray,
    character_scores: np.ndarray,
) -> list[tuple[str, float, float]]:
    """Select the exact hybrid top 50 from the complete entity inventory."""
    embedding_scores = np.asarray(embedding_scores, dtype=np.float32)
    character_scores = np.asarray(character_scores, dtype=np.float32) / 100.0
    hybrid_scores = 0.5 * embedding_scores + 0.5 * character_scores
    hybrid_scores[entity_index] = -np.inf
    selected = np.argpartition(hybrid_scores, -FREQUENT_CANDIDATES)[-FREQUENT_CANDIDATES:]
    ordered = sorted(
        (int(index) for index in selected),
        key=lambda index: (
            -float(hybrid_scores[index]),
            -float(character_scores[index]),
            -float(embedding_scores[index]),
            tags[index].casefold(),
            index,
        ),
    )
    return [
        (
            tags[index],
            float(embedding_scores[index]),
            float(character_scores[index]),
        )
        for index in ordered
    ]


def write_readme(
    output: Path, archive_name: str, frequent_count: int, expanded_threshold: int
) -> None:
    """Document the threshold, ranking order, and two candidate-page sizes."""
    text = f"""# Frequency-prioritized Konbaung entity candidates

`index.csv` is ordered by canonical corpus mention frequency, highest first.

- {frequent_count:,} entities with more than {expanded_threshold} mentions have
  exactly {FREQUENT_CANDIDATES} candidates.
- All remaining entities have exactly {NORMAL_CANDIDATES} candidates.
- Frequent-entity pages are the exact full-inventory top {FREQUENT_CANDIDATES} under
  `0.5 * emb + 0.5 * char`.
- Normal pages are preserved byte-for-byte from the audited v2 20-candidate archive.

The CSVs retain only `entity,emb,char` to minimize Gemini input tokens. `emb` is the
archive's fused Gemini score. `char` is the maximum of normalized original-order and
token-sorted RapidFuzz ratios. Source archive: `{archive_name}`.

During resolution, a separate active working copy must remove every resolved entity
from the parent roster, retire its own page, and delete it from every other active page.
The immutable files here remain the reproducible source archive.
"""
    (output / "README.md").write_text(text, encoding="utf-8", newline="\n")


def build(
    archive_path: Path,
    base_path: Path,
    output_path: Path,
    expanded_threshold: int = FREQUENT_MENTION_THRESHOLD,
) -> None:
    """Build the mixed 50/20 archive atomically and validate every generated page."""
    archive_path = archive_path.resolve()
    base_path = base_path.resolve()
    output_path = output_path.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if not (base_path / "index.csv").is_file():
        raise FileNotFoundError(base_path / "index.csv")
    if output_path.exists():
        raise FileExistsError(output_path)

    staging: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.staging_", dir=output_path.parent)
    )
    vector_root: tempfile.TemporaryDirectory[str] | None = None
    try:
        entities_output = staging / "entities"
        shutil.copytree(base_path / "entities", entities_output)
        with (base_path / "index.csv").open(encoding="utf-8", newline="") as handle:
            base_index = list(csv.DictReader(handle))

        with zipfile.ZipFile(archive_path) as archive:
            records = read_records(archive)
            vector_root = tempfile.TemporaryDirectory(prefix="konbaung_frequency_vectors_")
            fused_vectors = extract_fused_vectors(archive, Path(vector_root.name))

        if len(records) != len(base_index) or fused_vectors.shape[0] != len(records):
            raise RuntimeError("Entity records, base index, and vectors are not aligned")
        frequencies = [int(record["frequency"]) for record in records]
        if not all(left >= right for left, right in zip(frequencies, frequencies[1:])):
            raise RuntimeError("Canonical entity records are not frequency-descending")
        tags = [str(record["tag"]) for record in records]
        if len(set(tags)) != len(tags):
            raise RuntimeError("Canonical entity tags are not unique")

        width = len(str(len(records) - 1))
        filenames = [safe_filename(tag, index, width) for index, tag in enumerate(tags)]
        for index, (record, base_row, filename) in enumerate(
            zip(records, base_index, filenames, strict=True)
        ):
            expected_id = f"e{index:0{width}d}"
            if (
                int(record["index"]) != index
                or base_row["id"] != expected_id
                or base_row["entity"] != record["tag"]
                or base_row["file"] != filename
            ):
                raise RuntimeError(f"Source alignment failed at entity {index}")

        frequent_indices = [
            index for index, frequency in enumerate(frequencies) if frequency > expanded_threshold
        ]
        character_forms = [normalized_character_form(tag) for tag in tags]
        token_sorted_forms = [normalized_token_sorted_character_form(tag) for tag in tags]
        if any(not form for form in character_forms):
            raise RuntimeError("At least one entity lacks an alphanumeric form")

        for batch_start in range(0, len(frequent_indices), FREQUENT_BATCH_SIZE):
            batch_indices = frequent_indices[batch_start : batch_start + FREQUENT_BATCH_SIZE]
            embedding_matrix = np.asarray(
                fused_vectors[batch_indices] @ fused_vectors.T, dtype=np.float32
            )
            character_matrix = process.cdist(
                [character_forms[index] for index in batch_indices],
                character_forms,
                scorer=fuzz.ratio,
                dtype=np.float32,
                workers=-1,
            )
            token_sorted_matrix = process.cdist(
                [token_sorted_forms[index] for index in batch_indices],
                token_sorted_forms,
                scorer=fuzz.ratio,
                dtype=np.float32,
                workers=-1,
            )
            np.maximum(character_matrix, token_sorted_matrix, out=character_matrix)
            for local_index, entity_index in enumerate(batch_indices):
                rows = exact_top_rows(
                    entity_index,
                    tags,
                    embedding_matrix[local_index],
                    character_matrix[local_index],
                )
                write_candidate_csv(entities_output / filenames[entity_index], rows)
            print(
                f"built frequent pages {batch_start + 1}-"
                f"{batch_start + len(batch_indices)} of {len(frequent_indices)}",
                flush=True,
            )

        with (staging / "index.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("id", "entity", "mentions", "candidates", "file"))
            for index, (tag, frequency, filename) in enumerate(
                zip(tags, frequencies, filenames, strict=True)
            ):
                candidate_count = (
                    FREQUENT_CANDIDATES if frequency > expanded_threshold else NORMAL_CANDIDATES
                )
                writer.writerow(
                    (
                        f"e{index:0{width}d}",
                        tag,
                        frequency,
                        candidate_count,
                        filename,
                    )
                )

        expected_rows = (
            len(frequent_indices) * FREQUENT_CANDIDATES
            + (len(records) - len(frequent_indices)) * NORMAL_CANDIDATES
        )
        observed_rows = 0
        for index, filename in enumerate(filenames):
            with (entities_output / filename).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            expected = (
                FREQUENT_CANDIDATES
                if frequencies[index] > expanded_threshold
                else NORMAL_CANDIDATES
            )
            if len(rows) != expected or (rows and list(rows[0]) != ["entity", "emb", "char"]):
                raise RuntimeError(f"Candidate-page validation failed: {filename}")
            if tags[index] in {row["entity"] for row in rows}:
                raise RuntimeError(f"Self-candidate found: {filename}")
            observed_rows += len(rows)
        if observed_rows != expected_rows:
            raise RuntimeError("Candidate-row total does not match expectation")

        write_readme(
            staging,
            archive_path.name,
            len(frequent_indices),
            expanded_threshold,
        )
        manifest = {
            "schemaVersion": 2,
            "sourceArchive": archive_path.name,
            "sourceArchiveSha256": sha256(archive_path),
            "baseCandidateArchive": base_path.name,
            "entities": len(records),
            "frequencyOrder": "descending; stable canonical index breaks ties",
            "frequentMentionRule": f"frequency > {expanded_threshold}",
            "frequentEntities": len(frequent_indices),
            "frequentCandidatesPerEntity": FREQUENT_CANDIDATES,
            "normalEntities": len(records) - len(frequent_indices),
            "normalCandidatesPerEntity": NORMAL_CANDIDATES,
            "candidateRows": observed_rows,
            "columns": ["entity", "emb", "char"],
            "ordering": "0.50 * emb + 0.50 * char",
            "frequentCandidateSearch": "exact full-inventory hybrid top 50",
            "normalCandidateSearch": "audited v2 pages preserved byte-for-byte",
        }
        write_candidate_manifest = staging / "manifest.json"
        write_candidate_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(staging, output_path)
        staging = None
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    finally:
        if vector_root is not None:
            vector_root.cleanup()
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    """Build the candidate archive selected by command-line arguments."""
    args = parse_args()
    build(
        args.archive,
        args.base_candidates,
        args.output,
        args.expanded_threshold,
    )


if __name__ == "__main__":
    main()

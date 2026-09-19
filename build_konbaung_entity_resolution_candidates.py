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


DEFAULT_ARCHIVE = Path(__file__).with_name(
    "DIGHUM_WEBGPT_ANALYSIS_PACKAGE_20260831_WITH_METHODOLOGY.zip"
)
DEFAULT_OUTPUT = Path(__file__).with_name(
    "konbaung_entity_resolution_candidates_20260902_v2"
)
ENTITY_RECORDS = "embeddings/entity_records.jsonl"
ENTITY_NEIGHBORS = "similarity/entity_top30_neighbors.jsonl"
ENTITY_FUSED_VECTORS = "similarity/features/entity_fused_vectors.npy"
EMBEDDING_CANDIDATES = 30
CHARACTER_CANDIDATES = 30
OUTPUT_CANDIDATES = 20
CHARACTER_BATCH_SIZE = 128


def parse_args() -> argparse.Namespace:
    """Parse optional paths while keeping the canonical archive and output as defaults."""
    parser = argparse.ArgumentParser(
        description="Build compact per-entity resolution candidate CSV files."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Hash the source archive so the generated folder records its exact provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl_member(archive: zipfile.ZipFile, name: str) -> list[dict[str, Any]]:
    """Read one UTF-8 JSONL member directly from the source ZIP without unpacking it."""
    rows: list[dict[str, Any]] = []
    with archive.open(name) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8") as text:
            for line_number, line in enumerate(text, start=1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise RuntimeError(
                            f"Invalid JSON in {name} at line {line_number}: {error}"
                        ) from error
    return rows


def load_source_rows(
    archive: zipfile.ZipFile,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Load and validate the aligned entity inventory and embedding-neighbor indices."""
    records = read_jsonl_member(archive, ENTITY_RECORDS)
    neighbor_rows = read_jsonl_member(archive, ENTITY_NEIGHBORS)
    if len(records) != len(neighbor_rows):
        raise RuntimeError(
            f"Entity/neighbor count mismatch: {len(records)} != {len(neighbor_rows)}"
        )

    neighbor_indices = np.empty(
        (len(records), EMBEDDING_CANDIDATES), dtype=np.int32
    )
    for expected_index, (record, neighbor_row) in enumerate(
        zip(records, neighbor_rows, strict=True)
    ):
        if record.get("index") != expected_index:
            raise RuntimeError(f"Entity record index {expected_index} is not aligned")
        if neighbor_row.get("index") != expected_index:
            raise RuntimeError(f"Neighbor row index {expected_index} is not aligned")
        if neighbor_row.get("tag") != record.get("tag"):
            raise RuntimeError(f"Entity tag {expected_index} differs across source tables")
        neighbors = neighbor_row.get("neighbors", [])
        if len(neighbors) != EMBEDDING_CANDIDATES:
            raise RuntimeError(
                f"Entity {expected_index} has {len(neighbors)} embedding neighbors"
            )
        neighbor_indices[expected_index] = [
            int(neighbor["index"]) for neighbor in neighbors
        ]
    return records, neighbor_indices


def extract_fused_vectors(
    archive: zipfile.ZipFile, temporary_root: Path
) -> np.ndarray:
    """Extract only the fused matrix to temporary storage and load it into memory."""
    vector_path = temporary_root / "entity_fused_vectors.npy"
    with archive.open(ENTITY_FUSED_VECTORS) as source:
        with vector_path.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    vectors = np.load(vector_path)
    if vectors.ndim != 2 or vectors.shape[1] != 1536:
        raise RuntimeError(f"Unexpected fused-vector shape: {vectors.shape}")
    return vectors


def normalized_character_form(value: str) -> str:
    """Normalize case and separators, then retain only alphanumeric characters."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def normalized_token_sorted_character_form(value: str) -> str:
    """Sort normalized word tokens before character comparison so word order is neutral."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    spaced = "".join(
        character if character.isalnum() else " " for character in normalized
    )
    return "".join(sorted(spaced.split()))


def safe_filename(entity: str, index: int, width: int) -> str:
    """Create a readable Windows-safe filename with a stable index for uniqueness."""
    prefix = unicodedata.normalize("NFKC", entity)
    prefix = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", prefix)
    prefix = re.sub(r"\s+", "_", prefix).strip(" ._")
    prefix = prefix[:72].rstrip(" ._") or "entity"
    return f"{prefix}__e{index:0{width}d}.csv"


def write_candidate_csv(
    path: Path,
    candidate_rows: list[tuple[str, float, float]],
) -> None:
    """Write one compact three-column candidate table for a single source entity."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("entity", "emb", "char"))
        for entity, embedding_score, character_score in candidate_rows:
            writer.writerow((entity, f"{embedding_score:.4f}", f"{character_score:.4f}"))


def candidate_rows_for_entity(
    entity_index: int,
    tags: list[str],
    embedding_neighbor_indices: np.ndarray,
    character_neighbor_indices: np.ndarray,
    character_scores: np.ndarray,
    fused_vectors: np.ndarray,
) -> list[tuple[str, float, float]]:
    """Fuse the embedding and spelling pools, then rank them by both scores equally."""
    candidate_indices = np.unique(
        np.concatenate(
            (embedding_neighbor_indices, character_neighbor_indices)
        ).astype(np.int32, copy=False)
    )
    candidate_indices = candidate_indices[candidate_indices != entity_index]
    query_vector = np.asarray(fused_vectors[entity_index], dtype=np.float32)
    embedding_scores = np.asarray(
        fused_vectors[candidate_indices] @ query_vector, dtype=np.float32
    )
    selected_character_scores = (
        np.asarray(character_scores[candidate_indices], dtype=np.float32) / 100.0
    )
    hybrid_scores = 0.5 * embedding_scores + 0.5 * selected_character_scores

    ordered_positions = sorted(
        range(len(candidate_indices)),
        key=lambda position: (
            -float(hybrid_scores[position]),
            -float(selected_character_scores[position]),
            -float(embedding_scores[position]),
            tags[int(candidate_indices[position])].casefold(),
            int(candidate_indices[position]),
        ),
    )[:OUTPUT_CANDIDATES]
    return [
        (
            tags[int(candidate_indices[position])],
            float(embedding_scores[position]),
            float(selected_character_scores[position]),
        )
        for position in ordered_positions
    ]


def write_readme(output_root: Path, archive_name: str) -> None:
    """Document the terse file layout and the exact interpretation of both scores."""
    text = f"""# Konbaung entity-resolution candidates

Each CSV in `entities/` represents the source entity named before its `__eNNNNN`
suffix. `index.csv` preserves the exact entity text and its filename.

Each entity CSV has exactly three columns and 20 non-self candidates:

- `entity`: candidate raw entity tag.
- `emb`: exact fused Gemini similarity, where the archive defines the fused score as
  70% base-vector cosine plus 30% context-centroid cosine.
- `char`: RapidFuzz character ratio after Unicode NFKC normalization, case folding,
  and removal of spaces, punctuation, and separators. It is the better of the
  original-order ratio and a token-sorted ratio, so `Pindale Prince` and
  `Prince Pindale` score 1.0000 while extra or missing words still reduce the score.

Rows are ordered by `0.5 * emb + 0.5 * char`; the combined score is omitted to keep
the files to the requested three columns. The candidate pool is the union of the
archive's 30 precomputed embedding neighbors and the 30 exhaustive character-nearest
entities from the full 23,890-tag inventory.

Source: `{archive_name}`.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8", newline="\n")


def build(archive_path: Path, output_path: Path) -> None:
    """Generate and validate the complete per-entity candidate folder atomically."""
    archive_path = archive_path.resolve()
    output_path = output_path.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if output_path.exists():
        raise FileExistsError(
            f"Output already exists; choose a new path or move it first: {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.staging_", dir=output_path.parent)
    )
    temporary_vector_root: tempfile.TemporaryDirectory[str] | None = None
    try:
        entities_path = staging_path / "entities"
        entities_path.mkdir()
        source_archive_sha256 = sha256(archive_path)

        with zipfile.ZipFile(archive_path) as archive:
            records, embedding_neighbor_indices = load_source_rows(archive)
            temporary_vector_root = tempfile.TemporaryDirectory(
                prefix="konbaung_entity_vectors_"
            )
            fused_vectors = extract_fused_vectors(
                archive, Path(temporary_vector_root.name)
            )

            if fused_vectors.shape[0] != len(records):
                raise RuntimeError(
                    f"Entity/vector count mismatch: {len(records)} != {fused_vectors.shape[0]}"
                )

            tags = [str(record["tag"]) for record in records]
            character_forms = [normalized_character_form(tag) for tag in tags]
            token_sorted_character_forms = [
                normalized_token_sorted_character_form(tag) for tag in tags
            ]
            if any(not form for form in character_forms):
                empty_index = next(
                    index for index, form in enumerate(character_forms) if not form
                )
                raise RuntimeError(
                    f"Entity {empty_index} has no alphanumeric character form"
                )

            width = len(str(len(tags) - 1))
            filenames = [
                safe_filename(tag, index, width) for index, tag in enumerate(tags)
            ]
            if len({name.casefold() for name in filenames}) != len(filenames):
                raise RuntimeError("Generated filenames are not unique on Windows")

            with (staging_path / "index.csv").open(
                "w", encoding="utf-8", newline=""
            ) as index_handle:
                index_writer = csv.writer(index_handle, lineterminator="\n")
                index_writer.writerow(("id", "entity", "file"))

                for batch_start in range(0, len(tags), CHARACTER_BATCH_SIZE):
                    batch_end = min(batch_start + CHARACTER_BATCH_SIZE, len(tags))
                    score_matrix = process.cdist(
                        character_forms[batch_start:batch_end],
                        character_forms,
                        scorer=fuzz.ratio,
                        dtype=np.float32,
                        workers=-1,
                    )
                    token_sorted_score_matrix = process.cdist(
                        token_sorted_character_forms[batch_start:batch_end],
                        token_sorted_character_forms,
                        scorer=fuzz.ratio,
                        dtype=np.float32,
                        workers=-1,
                    )
                    np.maximum(
                        score_matrix, token_sorted_score_matrix, out=score_matrix
                    )
                    for local_index, entity_index in enumerate(
                        range(batch_start, batch_end)
                    ):
                        scores = score_matrix[local_index]
                        scores[entity_index] = -1.0
                        top_character_indices = np.argpartition(
                            scores, -CHARACTER_CANDIDATES
                        )[-CHARACTER_CANDIDATES:]
                        top_character_indices = top_character_indices[
                            np.lexsort(
                                (top_character_indices, -scores[top_character_indices])
                            )
                        ]
                        rows = candidate_rows_for_entity(
                            entity_index,
                            tags,
                            embedding_neighbor_indices[entity_index],
                            top_character_indices,
                            scores,
                            fused_vectors,
                        )
                        if len(rows) != OUTPUT_CANDIDATES:
                            raise RuntimeError(
                                f"Entity {entity_index} produced {len(rows)} candidates"
                            )
                        filename = filenames[entity_index]
                        write_candidate_csv(entities_path / filename, rows)
                        index_writer.writerow(
                            (f"e{entity_index:0{width}d}", tags[entity_index], filename)
                        )

        write_readme(staging_path, archive_path.name)
        manifest = {
            "schemaVersion": 1,
            "sourceArchive": archive_path.name,
            "sourceArchiveSha256": source_archive_sha256,
            "entities": len(records),
            "files": len(records),
            "candidatesPerEntity": OUTPUT_CANDIDATES,
            "candidateRows": len(records) * OUTPUT_CANDIDATES,
            "columns": ["entity", "emb", "char"],
            "embeddingScore": (
                "Exact archive fused similarity: 0.70 * base cosine + "
                "0.30 * context-centroid cosine"
            ),
            "characterScore": (
                "Maximum RapidFuzz ratio over original-order and token-sorted "
                "NFKC-casefolded alphanumeric characters"
            ),
            "ordering": "0.50 * emb + 0.50 * char",
            "candidatePool": {
                "embeddingNeighbors": EMBEDDING_CANDIDATES,
                "characterNeighbors": CHARACTER_CANDIDATES,
                "method": "union",
            },
        }
        (staging_path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        csv_files = list(entities_path.glob("*.csv"))
        if len(csv_files) != len(records):
            raise RuntimeError(
                f"Output file count mismatch: {len(csv_files)} != {len(records)}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_path, output_path)
        staging_path = None
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    finally:
        if temporary_vector_root is not None:
            temporary_vector_root.cleanup()
        if staging_path is not None and staging_path.exists():
            shutil.rmtree(staging_path)


def main() -> None:
    """Run the complete candidate-folder build from command-line arguments."""
    args = parse_args()
    build(args.archive, args.output)


if __name__ == "__main__":
    main()

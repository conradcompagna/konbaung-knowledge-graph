from __future__ import annotations

import gzip
import json
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CLUSTER_ROOT = ROOT / "konbaung_v3_node_edge_clustering_first_pass_20260724"
MODEL_ROOT = ROOT / "models" / "fasttext"
OUTPUT_ROOT = ROOT / "konbaung_v3_entity_resolution_two_pass_20260724"
TOKEN_ROOT = OUTPUT_ROOT / "fasttext_tag_tokens"


def jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def script_kind(token: str) -> str:
    for char in token:
        codepoint = ord(char)
        if 0x1000 <= codepoint <= 0x109F or 0xAA60 <= codepoint <= 0xAA7F:
            return "my"
    return "en"


def tag_tokens(tag: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", tag).casefold()
    characters: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category[0] in {"L", "N", "M"}:
            characters.append(char)
        else:
            characters.append(" ")
    parts = " ".join("".join(characters).split()).split()
    candidates = list(parts)
    if len(parts) > 1:
        candidates.extend(
            "_".join(parts[start : start + width])
            for width in (2, 3)
            for start in range(0, len(parts) - width + 1)
        )
        candidates.append("_".join(parts))
    raw = tag.strip()
    if raw:
        candidates.extend([raw, raw.casefold()])
    return list(dict.fromkeys(token for token in candidates if token))


def scan_vectors(
    archive: Path,
    wanted: set[str],
) -> tuple[list[str], np.ndarray, dict]:
    found: dict[str, np.ndarray] = {}
    lines = 0
    header_count = None
    dimensions = None
    with gzip.open(archive, "rt", encoding="utf-8", errors="replace", newline="\n") as handle:
        header = handle.readline().strip().split()
        if len(header) != 2:
            raise RuntimeError(f"Unexpected fastText header in {archive}: {header}")
        header_count, dimensions = map(int, header)
        for line in handle:
            lines += 1
            token, separator, values = line.partition(" ")
            if not separator or token not in wanted:
                continue
            vector = np.fromstring(values, dtype=np.float32, sep=" ")
            if vector.size != dimensions:
                raise RuntimeError(
                    f"Bad vector length for {token!r}: {vector.size} != {dimensions}"
                )
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector /= norm
            found[token] = vector
    if lines != header_count:
        raise RuntimeError(
            f"Truncated fastText archive {archive}: read {lines}, expected {header_count}"
        )
    tokens = sorted(found)
    matrix = (
        np.stack([found[token] for token in tokens]).astype(np.float32)
        if tokens
        else np.empty((0, dimensions), dtype=np.float32)
    )
    return tokens, matrix, {
        "archive": str(archive),
        "headerCount": header_count,
        "linesRead": lines,
        "dimensions": dimensions,
        "wantedTokens": len(wanted),
        "foundTokens": len(tokens),
        "coverage": len(tokens) / len(wanted) if wanted else 1.0,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TOKEN_ROOT.mkdir(parents=True, exist_ok=True)
    records = list(jsonl(CLUSTER_ROOT / "node_records.jsonl"))
    requested: dict[str, set[str]] = {"en": set(), "my": set()}
    token_document_frequency: Counter[str] = Counter()
    tag_token_map: dict[str, list[str]] = {}

    for record in records:
        tokens = tag_tokens(record["tag"])
        tag_token_map[record["tag"]] = tokens
        token_document_frequency.update(set(tokens))
        for token in tokens:
            requested[script_kind(token)].add(token)

    reports = {}
    token_indexes = {}
    for language in ("en", "my"):
        archive = MODEL_ROOT / f"cc.{language}.300.vec.gz"
        if not requested[language]:
            reports[language] = {
                "archive": str(archive),
                "wantedTokens": 0,
                "foundTokens": 0,
                "coverage": 1.0,
                "skipped": True,
            }
            token_indexes[language] = {}
            continue
        tokens, vectors, report = scan_vectors(archive, requested[language])
        np.save(TOKEN_ROOT / f"{language}_token_vectors.npy", vectors)
        with (TOKEN_ROOT / f"{language}_tokens.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for index, token in enumerate(tokens):
                handle.write(
                    json.dumps(
                        {
                            "index": index,
                            "token": token,
                            "documentFrequency": token_document_frequency[token],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        token_indexes[language] = {token: index for index, token in enumerate(tokens)}
        reports[language] = report
        print(json.dumps(report, ensure_ascii=False), flush=True)

    covered_tags = 0
    with (TOKEN_ROOT / "tag_tokens.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            tokens = tag_token_map[record["tag"]]
            covered = [
                {
                    "language": script_kind(token),
                    "token": token,
                    "vectorIndex": token_indexes[script_kind(token)].get(token),
                    "documentFrequency": token_document_frequency[token],
                }
                for token in tokens
                if token in token_indexes[script_kind(token)]
            ]
            if covered:
                covered_tags += 1
            handle.write(
                json.dumps(
                    {
                        "index": record["index"],
                        "tag": record["tag"],
                        "frequency": record["frequency"],
                        "tokens": tokens,
                        "coveredTokens": covered,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    report = {
        "nodeTags": len(records),
        "coveredNodeTags": covered_tags,
        "tagCoverage": covered_tags / len(records),
        "languages": reports,
    }
    (TOKEN_ROOT / "extraction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import math
import os
import re
import unicodedata
from zipfile import ZipFile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[2]
FIRST_PASS = ROOT / "konbaung_v3_node_edge_clustering_first_pass_20260724"
MANUAL_PASS = ROOT / "konbaung_v3_manual_seeded_clustering_20260724"
MANUAL_WORKBOOK = (
    Path(os.environ["KONBAUNG_ENTITY_MANUAL_WORKBOOK"]).resolve()
    if os.environ.get("KONBAUNG_ENTITY_MANUAL_WORKBOOK")
    else None
)
OUTPUT = ROOT / os.environ.get(
    "KONBAUNG_ENTITY_DISAMBIGUATION_OUTPUT",
    "konbaung_v3_node_disambiguation_20260724",
)
RANDOM_SEED = 20260724
CP1252_REVERSE: dict[str, int] = {}
for _byte in range(256):
    try:
        CP1252_REVERSE[bytes([_byte]).decode("cp1252")] = _byte
    except UnicodeDecodeError:
        continue

FEATURE_NAMES = [
    "geminiTagCosine",
    "geminiContextCosine",
    "geminiFusedCosine",
    "wordTfidfCosine",
    "charTfidfCosine",
    "tokenJaccard",
    "sharedTokenCoverage",
    "characterLengthRatio",
    "tokenCountRatio",
    "normalizedExact",
    "tokenSetContainment",
    "numericSignatureEqual",
    "bothHaveNumericSignature",
    "compactNormalizedExact",
]


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def repair_utf8_mojibake(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        raw = bytes(ord(char) if ord(char) <= 255 else CP1252_REVERSE[char] for char in value)
        return raw.decode("utf-8")
    except (KeyError, UnicodeDecodeError, ValueError):
        return value


def xlsx_table(path: Path, sheet_name: str) -> list[dict[str, Any]]:
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    relation_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    main = f"{{{main_ns}}}"
    relation = f"{{{relation_ns}}}"

    def column_index(reference: str) -> int:
        match = re.match(r"([A-Z]+)", reference)
        if match is None:
            raise RuntimeError(f"Invalid XLSX cell reference: {reference}")
        result = 0
        for char in match.group(1):
            result = result * 26 + ord(char) - 64
        return result - 1

    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(text.text or "" for text in item.iter(f"{main}t"))
                for item in shared_root.findall(f"{main}si")
            ]

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {row.attrib["Id"]: row.attrib["Target"] for row in relationships}
        sheets = workbook.find(f"{main}sheets")
        if sheets is None:
            raise RuntimeError("XLSX workbook has no sheets")
        sheet = next(
            (row for row in sheets if row.attrib.get("name") == sheet_name),
            None,
        )
        if sheet is None:
            raise RuntimeError(f"Missing XLSX sheet: {sheet_name}")
        target = targets[sheet.attrib[f"{relation}id"]].lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        sheet_root = ET.fromstring(archive.read(target))

    raw_rows: list[dict[int, Any]] = []
    for row in sheet_root.findall(f".//{main}sheetData/{main}row"):
        values: dict[int, Any] = {}
        for cell in row.findall(f"{main}c"):
            cell_type = cell.attrib.get("t")
            value_element = cell.find(f"{main}v")
            inline_element = cell.find(f"{main}is")
            if cell_type == "s" and value_element is not None:
                value: Any = shared_strings[int(value_element.text)]
            elif cell_type == "inlineStr" and inline_element is not None:
                value = "".join(text.text or "" for text in inline_element.iter(f"{main}t"))
            elif cell_type == "b" and value_element is not None:
                value = value_element.text == "1"
            else:
                value = value_element.text if value_element is not None else None
            values[column_index(cell.attrib["r"])] = value
        if values:
            raw_rows.append(values)
    if not raw_rows:
        raise RuntimeError(f"XLSX sheet is empty: {sheet_name}")
    header_width = max(raw_rows[0]) + 1
    headers = [raw_rows[0].get(index) for index in range(header_width)]
    if any(header is None for header in headers):
        raise RuntimeError(f"XLSX sheet has blank headers: {sheet_name}")
    return [
        {str(headers[index]): row.get(index) for index in range(header_width)}
        for row in raw_rows[1:]
    ]


def load_expanded_manual_seeds(
    records: list[dict[str, Any]],
    workbook_path: Path,
) -> tuple[
    list[str | None],
    dict[str, list[int]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    rows = xlsx_table(workbook_path, "Entity Assignments")
    text_fields = (
        "tag",
        "cluster_id",
        "canonical_label",
        "action",
        "confidence",
        "basis",
        "quality_flag",
        "notes",
        "taxonomy_origin",
    )
    repair_log: list[dict[str, str]] = []
    for row in rows:
        for field in text_fields:
            original = row.get(field)
            repaired = repair_utf8_mojibake(original)
            row[field] = repaired
            if field == "tag" and original != repaired:
                repair_log.append(
                    {
                        "workbookValue": str(original),
                        "sourceValue": str(repaired),
                    }
                )

    source_by_tag = {str(row["tag"]): row for row in records}
    manual_by_tag: dict[str, dict[str, Any]] = {}
    for row in rows:
        tag = str(row["tag"])
        if tag in manual_by_tag:
            raise RuntimeError(f"Duplicate manual entity tag: {tag}")
        manual_by_tag[tag] = row
    expected_tags = {str(row["tag"]) for row in records if int(row["frequency"]) > 1}
    if set(manual_by_tag) != expected_tags:
        missing = sorted(expected_tags - set(manual_by_tag))
        extra = sorted(set(manual_by_tag) - expected_tags)
        raise RuntimeError(
            "Expanded manual workbook does not exactly cover frequency > 1 "
            f"tags after text repair; missing={missing[:10]}, extra={extra[:10]}"
        )

    manual_seed: list[str | None] = [None] * len(records)
    manual_members: dict[str, list[int]] = defaultdict(list)
    seed_meta: dict[str, dict[str, Any]] = {}
    for tag, row in manual_by_tag.items():
        source = source_by_tag[tag]
        if int(row["frequency"]) != int(source["frequency"]):
            raise RuntimeError(f"Manual frequency mismatch for {tag}")
        index = int(source["index"])
        seed_id = str(row["cluster_id"])
        canonical = str(row["canonical_label"])
        manual_seed[index] = seed_id
        manual_members[seed_id].append(index)
        existing = seed_meta.get(seed_id)
        if existing is not None and existing["canonicalLabel"] != canonical:
            raise RuntimeError(f"Conflicting canonical labels for manual seed {seed_id}")
        seed_meta[seed_id] = {
            "seedClusterId": seed_id,
            "canonicalLabel": canonical,
            "action": row.get("action"),
            "confidence": row.get("confidence"),
            "basis": row.get("basis"),
            "qualityFlag": row.get("quality_flag"),
            "notes": row.get("notes"),
            "taxonomyOrigin": row.get("taxonomy_origin"),
        }

    observed_member_counts = Counter(str(row["cluster_id"]) for row in rows)
    observed_frequency_totals: Counter[str] = Counter()
    for row in rows:
        observed_frequency_totals[str(row["cluster_id"])] += int(row["frequency"])
    for row in rows:
        seed_id = str(row["cluster_id"])
        if int(row["member_count"]) != observed_member_counts[seed_id]:
            raise RuntimeError(f"Manual member-count mismatch for seed {seed_id}")
        if int(row["cluster_total_frequency"]) != observed_frequency_totals[seed_id]:
            raise RuntimeError(f"Manual frequency-total mismatch for seed {seed_id}")

    audit = {
        "mode": "expanded_frequency_gt1_workbook",
        "workbook": str(workbook_path),
        "workbookSha256": sha256(workbook_path),
        "reviewedTags": len(rows),
        "reviewedMentions": sum(int(row["frequency"]) for row in rows),
        "manualSeedClusters": len(manual_members),
        "singletonManualClusters": sum(len(members) == 1 for members in manual_members.values()),
        "multiTagManualClusters": sum(len(members) > 1 for members in manual_members.values()),
        "utf8MojibakeTagRepairs": repair_log,
        "actions": dict(Counter(str(row["action"]) for row in rows)),
        "confidence": dict(Counter(str(row["confidence"]) for row in rows)),
        "qualityFlags": dict(Counter(str(row["quality_flag"] or "") for row in rows)),
    }
    return manual_seed, manual_members, seed_meta, audit


def normalized_tokens(value: str) -> tuple[str, ...]:
    value = unicodedata.normalize("NFKC", value).casefold()
    cleaned = "".join(char if char.isalnum() else " " for char in value)
    return tuple(cleaned.split())


def numeric_signature(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(token for token in tokens if token.isdigit())


def character_ngrams(tokens: tuple[str, ...]) -> frozenset[str]:
    compact = "".join(tokens)
    if not compact:
        return frozenset()
    padded = f"^{compact}$"
    if len(padded) < 3:
        return frozenset({padded})
    return frozenset(padded[index : index + 3] for index in range(len(padded) - 2))


def build_candidate_pairs(
    records: list[dict[str, Any]],
    tokens: list[tuple[str, ...]],
    grams: list[frozenset[str]],
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    pairs: set[tuple[int, int]] = set()
    source_counts: Counter[str] = Counter()

    def add_pair(left: int, right: int, source: str) -> None:
        if left == right:
            return
        pair = (left, right) if left < right else (right, left)
        if pair not in pairs:
            source_counts[source] += 1
        pairs.add(pair)

    knn = np.load(FIRST_PASS / "node_knn.npz")
    neighbor_indices = knn["indices"]
    for source in range(neighbor_indices.shape[0]):
        for target in neighbor_indices[source]:
            add_pair(source, int(target), "gemini_knn")

    exact_groups: dict[str, list[int]] = defaultdict(list)
    compact_groups: dict[str, list[int]] = defaultdict(list)
    order_signature_groups: dict[str, list[int]] = defaultdict(list)
    for index, row_tokens in enumerate(tokens):
        exact_groups[" ".join(row_tokens)].append(index)
        compact_groups["".join(row_tokens)].append(index)
        order_signature_groups[
            " ".join(sorted(token for token in row_tokens if token not in {"of", "the"}))
        ].append(index)
    exact_pair_count = 0
    for key, members in exact_groups.items():
        if not key or len(members) < 2:
            continue
        for left_position in range(len(members)):
            for right_position in range(left_position + 1, len(members)):
                before = len(pairs)
                add_pair(
                    members[left_position],
                    members[right_position],
                    "normalized_exact",
                )
                exact_pair_count += len(pairs) - before

    compact_pair_count = 0
    for key, members in compact_groups.items():
        if not key or len(members) < 2:
            continue
        for left_position in range(len(members)):
            for right_position in range(left_position + 1, len(members)):
                before = len(pairs)
                add_pair(
                    members[left_position],
                    members[right_position],
                    "compact_normalized_exact",
                )
                compact_pair_count += len(pairs) - before

    order_signature_pair_count = 0
    for key, members in order_signature_groups.items():
        if not key or len(members) < 2 or len(members) > 100:
            continue
        for left_position in range(len(members)):
            for right_position in range(left_position + 1, len(members)):
                before = len(pairs)
                add_pair(
                    members[left_position],
                    members[right_position],
                    "order_insensitive_token_signature",
                )
                order_signature_pair_count += len(pairs) - before

    gram_postings: dict[str, list[int]] = defaultdict(list)
    for index, row_grams in enumerate(grams):
        for gram in row_grams:
            gram_postings[gram].append(index)

    lexical_pairs_before = len(pairs)
    for source, row_grams in enumerate(grams):
        rare = sorted(
            (
                (len(gram_postings[gram]), gram)
                for gram in row_grams
                if 2 <= len(gram_postings[gram]) <= 80
            )
        )[:10]
        shared_counts: Counter[int] = Counter()
        for _, gram in rare:
            for target in gram_postings[gram]:
                if target != source:
                    shared_counts[target] += 1
        ranked: list[tuple[float, int, int]] = []
        for target, shared in shared_counts.items():
            union = len(row_grams | grams[target])
            if union <= 0:
                continue
            jaccard = shared / union
            if jaccard < 0.12:
                continue
            ranked.append((jaccard, shared, target))
        ranked.sort(reverse=True)
        for _, _, target in ranked[:20]:
            add_pair(source, target, "rare_char_ngram")

    ordered = sorted(pairs)
    return (
        ordered,
        {
            "candidatePairs": len(ordered),
            "newPairsByFirstSource": dict(source_counts),
            "normalizedExactPairs": exact_pair_count,
            "compactNormalizedExactPairs": compact_pair_count,
            "orderInsensitiveSignaturePairs": order_signature_pair_count,
            "rareCharacterPairsAdded": len(pairs) - lexical_pairs_before,
            "geminiNeighborsPerRecord": int(neighbor_indices.shape[1]),
            "rareCharacterPostingMaximum": 80,
            "rareCharacterGramsPerRecord": 10,
            "rareCharacterCandidatesPerRecord": 20,
        },
    )


def build_text_features(
    records: list[dict[str, Any]],
    normalized_text: list[str],
) -> tuple[csr_matrix, csr_matrix, dict[str, Any]]:
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=False,
        sublinear_tf=True,
        norm="l2",
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        lowercase=False,
        sublinear_tf=True,
        norm="l2",
    )
    word_matrix = word_vectorizer.fit_transform(normalized_text).tocsr()
    char_matrix = char_vectorizer.fit_transform(normalized_text).tocsr()
    return (
        word_matrix,
        char_matrix,
        {
            "wordFeatures": len(word_vectorizer.vocabulary_),
            "wordNgramRange": [1, 2],
            "charFeatures": len(char_vectorizer.vocabulary_),
            "charNgramRange": [2, 5],
            "records": len(records),
        },
    )


class PairFeatures:
    def __init__(
        self,
        base_vectors: np.ndarray,
        context_vectors: np.ndarray,
        fused_vectors: np.ndarray,
        word_matrix: csr_matrix,
        char_matrix: csr_matrix,
        tokens: list[tuple[str, ...]],
        normalized_text: list[str],
        numeric: list[tuple[str, ...]],
        token_idf: dict[str, float],
    ) -> None:
        self.base_vectors = base_vectors
        self.context_vectors = context_vectors
        self.fused_vectors = fused_vectors
        self.word_matrix = word_matrix
        self.char_matrix = char_matrix
        self.tokens = tokens
        self.normalized_text = normalized_text
        self.numeric = numeric
        self.token_idf = token_idf

    def matrix(self, pairs: list[tuple[int, int]]) -> np.ndarray:
        if not pairs:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        left = np.asarray([pair[0] for pair in pairs], dtype=np.int32)
        right = np.asarray([pair[1] for pair in pairs], dtype=np.int32)
        matrix = np.empty((len(pairs), len(FEATURE_NAMES)), dtype=np.float32)
        matrix[:, 0] = np.einsum(
            "ij,ij->i",
            np.asarray(self.base_vectors[left], dtype=np.float32),
            np.asarray(self.base_vectors[right], dtype=np.float32),
        )
        matrix[:, 1] = np.einsum(
            "ij,ij->i",
            np.asarray(self.context_vectors[left], dtype=np.float32),
            np.asarray(self.context_vectors[right], dtype=np.float32),
        )
        matrix[:, 2] = np.einsum(
            "ij,ij->i",
            np.asarray(self.fused_vectors[left], dtype=np.float32),
            np.asarray(self.fused_vectors[right], dtype=np.float32),
        )
        matrix[:, 3] = np.asarray(
            self.word_matrix[left].multiply(self.word_matrix[right]).sum(axis=1)
        ).reshape(-1)
        matrix[:, 4] = np.asarray(
            self.char_matrix[left].multiply(self.char_matrix[right]).sum(axis=1)
        ).reshape(-1)

        for row, (left_index, right_index) in enumerate(pairs):
            left_tokens = self.tokens[left_index]
            right_tokens = self.tokens[right_index]
            left_set = set(left_tokens)
            right_set = set(right_tokens)
            shared = left_set & right_set
            union = left_set | right_set
            matrix[row, 5] = len(shared) / len(union) if union else 1.0
            shared_weight = sum(self.token_idf.get(token, 0.0) for token in shared)
            left_weight = sum(self.token_idf.get(token, 0.0) for token in left_set)
            right_weight = sum(self.token_idf.get(token, 0.0) for token in right_set)
            denominator = min(left_weight, right_weight)
            matrix[row, 6] = shared_weight / denominator if denominator > 0 else 0.0
            left_length = len(self.normalized_text[left_index])
            right_length = len(self.normalized_text[right_index])
            matrix[row, 7] = (
                min(left_length, right_length) / max(left_length, right_length)
                if max(left_length, right_length) > 0
                else 1.0
            )
            matrix[row, 8] = (
                min(len(left_tokens), len(right_tokens)) / max(len(left_tokens), len(right_tokens))
                if max(len(left_tokens), len(right_tokens)) > 0
                else 1.0
            )
            matrix[row, 9] = float(
                self.normalized_text[left_index] == self.normalized_text[right_index]
            )
            matrix[row, 10] = float(
                bool(left_set)
                and bool(right_set)
                and (left_set.issubset(right_set) or right_set.issubset(left_set))
            )
            matrix[row, 11] = float(self.numeric[left_index] == self.numeric[right_index])
            matrix[row, 12] = float(
                bool(self.numeric[left_index]) and bool(self.numeric[right_index])
            )
            matrix[row, 13] = float("".join(left_tokens) == "".join(right_tokens))
        return matrix


def role_transform_tokens(
    left_tokens: tuple[str, ...],
    right_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    connectives = {"of", "the"}
    left = tuple(token for token in left_tokens if token not in connectives)
    right = tuple(token for token in right_tokens if token not in connectives)
    if len(left) < 2 or len(left) != len(right):
        return ()
    if Counter(left) != Counter(right) or left == right:
        return ()
    candidates = set()
    if left[0] == right[-1] and left[1:] == right[:-1]:
        candidates.add(left[0])
    if left[-1] == right[0] and left[:-1] == right[1:]:
        candidates.add(left[-1])
    return tuple(sorted(candidates))


def orthographic_variant_compatible(
    left_tokens: tuple[str, ...],
    right_tokens: tuple[str, ...],
) -> bool:
    """Accept spelling variation, not mere shared-topic containment."""
    ignored = {"of", "the", "and"}
    left = tuple(token for token in left_tokens if token not in ignored)
    right = tuple(token for token in right_tokens if token not in ignored)
    if not left or not right or left == right:
        return False
    if max(len(left), len(right)) > 6:
        return False
    left_counter = Counter(left)
    right_counter = Counter(right)
    if left_counter == right_counter:
        return False
    left_set = set(left)
    right_set = set(right)
    if left_set < right_set or right_set < left_set:
        return False

    left_compact = "".join(left)
    right_compact = "".join(right)
    compact_similarity = difflib.SequenceMatcher(None, left_compact, right_compact).ratio()
    length_ratio = min(len(left_compact), len(right_compact)) / max(
        len(left_compact), len(right_compact)
    )

    if len(left) == len(right):
        token_similarities = [
            difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(left, right) if a != b
        ]
        return bool(
            token_similarities
            and min(token_similarities) >= 0.82
            and compact_similarity >= 0.88
            and length_ratio >= 0.82
        )

    # Narrowly allow token-boundary changes in transliterations. Strict-set
    # containment was rejected above, so appended qualifiers cannot pass.
    return compact_similarity >= 0.94 and length_ratio >= 0.84


def strict_conjunction_reordering(
    left_tokens: tuple[str, ...],
    right_tokens: tuple[str, ...],
) -> bool:
    """Accept an actual conjunct swap without changing modifier scope."""
    if left_tokens.count("and") != 1 or right_tokens.count("and") != 1:
        return False
    left_and = left_tokens.index("and")
    right_and = right_tokens.index("and")
    left_first = left_tokens[:left_and]
    left_second = left_tokens[left_and + 1 :]
    right_first = right_tokens[:right_and]
    right_second = right_tokens[right_and + 1 :]
    if not all((left_first, left_second, right_first, right_second)):
        return False
    if left_first == right_second and left_second == right_first:
        return True

    # Also allow "A and B X" <-> "B and A X", where X is an explicitly
    # repeated common head such as "forces", "affairs", or "officials".
    common_suffix_length = 0
    for left_token, right_token in zip(reversed(left_second), reversed(right_second)):
        if left_token != right_token:
            break
        common_suffix_length += 1
    if common_suffix_length == 0:
        return False
    left_second_core = left_second[:-common_suffix_length]
    right_second_core = right_second[:-common_suffix_length]
    return (
        bool(left_second_core)
        and bool(right_second_core)
        and left_first == right_second_core
        and left_second_core == right_first
    )


def learn_safe_role_tokens(
    training_pairs: list[tuple[int, int]],
    labels: np.ndarray,
    tokens: list[tuple[str, ...]],
) -> tuple[set[str], dict[str, Any]]:
    evidence: dict[str, Counter[str]] = defaultdict(Counter)
    for pair, label in zip(training_pairs, labels):
        roles = role_transform_tokens(tokens[pair[0]], tokens[pair[1]])
        for role in roles:
            evidence[role]["positive" if int(label) == 1 else "negative"] += 1
    safe = {
        role
        for role, counts in evidence.items()
        if counts["positive"] >= 1 and counts["negative"] == 0
    }
    report = {
        role: {
            "positivePairs": counts["positive"],
            "negativePairs": counts["negative"],
            "acceptedAsSafeRole": role in safe,
        }
        for role, counts in sorted(evidence.items())
    }
    return safe, report


def manual_training_pairs(
    candidate_pairs: list[tuple[int, int]],
    manual_seed: list[str | None],
    manual_members: dict[str, list[int]],
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], np.ndarray, dict[str, Any]]:
    positives: set[tuple[int, int]] = set()
    for members in manual_members.values():
        for left_position in range(len(members)):
            for right_position in range(left_position + 1, len(members)):
                positives.add((members[left_position], members[right_position]))

    hard_negatives = {
        pair
        for pair in candidate_pairs
        if manual_seed[pair[0]] is not None
        and manual_seed[pair[1]] is not None
        and manual_seed[pair[0]] != manual_seed[pair[1]]
    }

    manual_indices = np.asarray(
        [index for index, seed in enumerate(manual_seed) if seed is not None],
        dtype=np.int32,
    )
    random_negatives: set[tuple[int, int]] = set()
    attempts = 0
    while len(random_negatives) < 12000 and attempts < 200000:
        attempts += 1
        left, right = rng.choice(manual_indices, size=2, replace=False)
        left = int(left)
        right = int(right)
        if manual_seed[left] == manual_seed[right]:
            continue
        pair = (left, right) if left < right else (right, left)
        random_negatives.add(pair)

    negatives = hard_negatives | random_negatives
    training_pairs = sorted(positives) + sorted(negatives)
    labels = np.asarray(
        [1] * len(positives) + [0] * len(negatives),
        dtype=np.int8,
    )
    return (
        training_pairs,
        labels,
        {
            "positivePairs": len(positives),
            "hardNegativePairs": len(hard_negatives),
            "randomNegativePairs": len(random_negatives),
            "negativePairs": len(negatives),
            "trainingPairs": len(training_pairs),
        },
    )


def new_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )


def train_model(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
) -> tuple[HistGradientBoostingClassifier, dict[str, Any]]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(labels), dtype=np.float64)
    for training, testing in splitter.split(feature_matrix, labels):
        model = new_classifier()
        model.fit(feature_matrix[training], labels[training])
        oof[testing] = model.predict_proba(feature_matrix[testing])[:, 1]

    target_precision = 0.995

    def select_threshold(
        lane_name: str,
        eligible: np.ndarray,
        minimum_threshold: float,
    ) -> dict[str, Any]:
        selected: dict[str, Any] | None = None
        for threshold in np.linspace(minimum_threshold, 0.9999, 1200):
            predicted = eligible & (oof >= threshold)
            accepted = int(predicted.sum())
            if accepted < 20:
                continue
            true_positive = int(((labels == 1) & predicted).sum())
            false_positive = int(((labels == 0) & predicted).sum())
            precision = true_positive / accepted
            recall = true_positive / int((labels == 1).sum())
            if precision < target_precision:
                continue
            candidate = {
                "lane": lane_name,
                "threshold": float(threshold),
                "eligibleRows": int(eligible.sum()),
                "accepted": accepted,
                "truePositive": true_positive,
                "falsePositive": false_positive,
                "precision": precision,
                "recall": recall,
            }
            if (
                selected is None
                or candidate["recall"] > selected["recall"]
                or (
                    candidate["recall"] == selected["recall"]
                    and candidate["precision"] > selected["precision"]
                )
            ):
                selected = candidate
        if selected is None:
            threshold = 0.999
            predicted = eligible & (oof >= threshold)
            accepted = int(predicted.sum())
            true_positive = int(((labels == 1) & predicted).sum())
            false_positive = int(((labels == 0) & predicted).sum())
            selected = {
                "lane": lane_name,
                "threshold": threshold,
                "eligibleRows": int(eligible.sum()),
                "accepted": accepted,
                "truePositive": true_positive,
                "falsePositive": false_positive,
                "precision": (true_positive / accepted if accepted else None),
                "recall": (
                    true_positive / int((labels == 1).sum()) if int((labels == 1).sum()) else None
                ),
            }
        return selected

    lane_masks = {
        "general": np.ones(len(labels), dtype=bool),
        "token_reordering": (
            (feature_matrix[:, 5] >= 0.72)
            & (feature_matrix[:, 6] >= 0.95)
            & (feature_matrix[:, 4] >= 0.60)
            & (feature_matrix[:, 8] >= 0.70)
        ),
        "orthographic_variant": (
            (feature_matrix[:, 4] >= 0.78)
            & (feature_matrix[:, 7] >= 0.70)
            & (feature_matrix[:, 0] >= 0.86)
        ),
    }
    lanes = {
        "general": select_threshold("general", lane_masks["general"], 0.90),
        "token_reordering": select_threshold(
            "token_reordering",
            lane_masks["token_reordering"],
            0.75,
        ),
        "orthographic_variant": select_threshold(
            "orthographic_variant",
            lane_masks["orthographic_variant"],
            0.80,
        ),
    }
    selected = dict(lanes["general"])
    selected["targetPrecision"] = target_precision
    selected["folds"] = 5
    selected["lanes"] = {name: dict(row) for name, row in lanes.items()}
    selected["positiveProbabilityQuantiles"] = {
        str(value): float(np.quantile(oof[labels == 1], value))
        for value in (0.1, 0.25, 0.5, 0.75, 0.9)
    }
    selected["negativeProbabilityQuantiles"] = {
        str(value): float(np.quantile(oof[labels == 0], value))
        for value in (0.5, 0.9, 0.95, 0.99, 0.999)
    }
    model = new_classifier()
    model.fit(feature_matrix, labels)
    return model, selected


class UnionFind:
    def __init__(self, size: int, manual_seed: list[str | None]) -> None:
        self.parent = list(range(size))
        self.members = [[index] for index in range(size)]
        self.seed = list(manual_seed)

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> tuple[bool, str | None]:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return True, None
        left_seed = self.seed[left_root]
        right_seed = self.seed[right_root]
        if left_seed is not None and right_seed is not None and left_seed != right_seed:
            return False, "different_manual_seeds"
        if len(self.members[left_root]) < len(self.members[right_root]):
            left_root, right_root = right_root, left_root
            left_seed, right_seed = right_seed, left_seed
        self.parent[right_root] = left_root
        self.members[left_root].extend(self.members[right_root])
        self.members[right_root] = []
        self.seed[left_root] = left_seed if left_seed is not None else right_seed
        self.seed[right_root] = None
        return True, None


def markdown_text(value: Any) -> str:
    return (
        str(value).replace("\r", " ").replace("\n", " ").replace("\\", "\\\\").replace("|", "\\|")
    )


def write_markdown_inventory(
    path: Path,
    clusters: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    include_singletons: bool,
) -> None:
    assignment_by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignments:
        assignment_by_cluster[row["clusterId"]].append(row)
    selected = [
        cluster for cluster in clusters if include_singletons or int(cluster["uniqueTags"]) > 1
    ]
    selected.sort(
        key=lambda row: (
            -int(row["mentionCount"]),
            -int(row["uniqueTags"]),
            str(row["clusterId"]),
        )
    )
    lines = [
        "# Konbaung V3 node identity-disambiguation inventory",
        "",
        (
            "This file lists every identity component, including singletons."
            if include_singletons
            else "This file lists every multi-tag identity component. Singleton "
            "tags remain available in the CSV and JSONL assignments."
        ),
        "",
        f"- Listed clusters: **{len(selected):,}**",
        f"- Listed tags: **{sum(int(row['uniqueTags']) for row in selected):,}**",
        "",
    ]
    for cluster in selected:
        cluster_id = str(cluster["clusterId"])
        rows = sorted(
            assignment_by_cluster[cluster_id],
            key=lambda row: (
                -int(row["frequency"]),
                str(row["tag"]).casefold(),
                str(row["tag"]),
            ),
        )
        lines.extend(
            [
                f"## {markdown_text(cluster_id)} — {markdown_text(cluster['canonicalLabel'])}",
                "",
                f"- Cluster type: `{cluster['clusterType']}`",
                f"- Total count: **{int(cluster['mentionCount']):,}**",
                f"- Unique tags: **{int(cluster['uniqueTags']):,}**",
                f"- Manual seed: {markdown_text(cluster.get('manualSeedId') or 'none')}",
                "",
                "| Tag | Frequency | Assignment | Link | Confidence |",
                "|---|---:|---|---|---:|",
            ]
        )
        for row in rows:
            confidence = row.get("linkConfidence")
            lines.append(
                f"| {markdown_text(row['tag'])} | {int(row['frequency'])} | "
                f"{markdown_text(row['assignmentMethod'])} | "
                f"{markdown_text(row.get('linkedTag') or '')} | "
                f"{'' if confidence is None else f'{float(confidence):.6f}'} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_assignment_csv(path: Path, assignments: list[dict[str, Any]]) -> None:
    fields = [
        "index",
        "tag",
        "frequency",
        "clusterId",
        "canonicalLabel",
        "clusterType",
        "manualSeedId",
        "manualSeedMember",
        "assignmentMethod",
        "linkedTag",
        "linkConfidence",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(assignments)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protected_paths = [
        FIRST_PASS / "node_records.jsonl",
        FIRST_PASS / "node_base_vectors.npy",
        FIRST_PASS / "node_context_vectors.npy",
        FIRST_PASS / "node_fused_vectors.npy",
        FIRST_PASS / "node_knn.npz",
    ]
    if MANUAL_WORKBOOK is not None:
        protected_paths.append(MANUAL_WORKBOOK)
    else:
        protected_paths.extend(
            [
                MANUAL_PASS / "node_assignments.jsonl",
                MANUAL_PASS / "node_seed_index.json",
            ]
        )
    hashes_before = {str(path): sha256(path) for path in protected_paths}

    records = list(jsonl(FIRST_PASS / "node_records.jsonl"))
    if [int(row["index"]) for row in records] != list(range(len(records))):
        raise RuntimeError("Node records are incomplete or unordered")
    if MANUAL_WORKBOOK is not None:
        (
            manual_seed,
            manual_members,
            seed_meta,
            manual_audit,
        ) = load_expanded_manual_seeds(records, MANUAL_WORKBOOK)
    else:
        manual_source = list(jsonl(MANUAL_PASS / "node_assignments.jsonl"))
        if len(manual_source) != len(records):
            raise RuntimeError("Manual assignment source does not align with records")
        seed_index = json.loads((MANUAL_PASS / "node_seed_index.json").read_text(encoding="utf-8"))
        seed_meta = {str(row["seedClusterId"]): row for row in seed_index}
        manual_seed = [None] * len(records)
        manual_members = defaultdict(list)
        for row in manual_source:
            if not bool(row["manualSeedMember"]):
                continue
            index = int(row["index"])
            seed_id = str(row["clusterId"])
            manual_seed[index] = seed_id
            manual_members[seed_id].append(index)
        if sum(seed is not None for seed in manual_seed) != 1000:
            raise RuntimeError("Expected exactly 1,000 reviewed manual node tags")
        if set(manual_members) != set(seed_meta):
            raise RuntimeError("Manual seed metadata/member mismatch")
        manual_audit = {
            "mode": "original_top1000_manual_seed_pass",
            "reviewedTags": 1000,
            "manualSeedClusters": len(manual_members),
        }

    tokens = [normalized_tokens(str(row["tag"])) for row in records]
    normalized_text = [" ".join(row_tokens) for row_tokens in tokens]
    numeric = [numeric_signature(row_tokens) for row_tokens in tokens]
    grams = [character_ngrams(row_tokens) for row_tokens in tokens]
    token_document_frequency = Counter(token for row_tokens in tokens for token in set(row_tokens))
    token_idf = {
        token: math.log((len(records) + 1) / (frequency + 1)) + 1
        for token, frequency in token_document_frequency.items()
    }

    candidate_pairs, candidate_report = build_candidate_pairs(records, tokens, grams)
    word_matrix, char_matrix, text_report = build_text_features(records, normalized_text)
    base_vectors = np.load(FIRST_PASS / "node_base_vectors.npy", mmap_mode="r")
    context_vectors = np.load(FIRST_PASS / "node_context_vectors.npy", mmap_mode="r")
    fused_vectors = np.load(FIRST_PASS / "node_fused_vectors.npy", mmap_mode="r")
    feature_builder = PairFeatures(
        base_vectors,
        context_vectors,
        fused_vectors,
        word_matrix,
        char_matrix,
        tokens,
        normalized_text,
        numeric,
        token_idf,
    )

    rng = np.random.default_rng(RANDOM_SEED)
    training_pairs, labels, training_report = manual_training_pairs(
        candidate_pairs, manual_seed, manual_members, rng
    )
    training_features = feature_builder.matrix(training_pairs)
    safe_role_tokens, role_evidence = learn_safe_role_tokens(training_pairs, labels, tokens)
    model, calibration = train_model(training_features, labels)
    threshold = float(calibration["threshold"])
    lane_thresholds = {name: float(row["threshold"]) for name, row in calibration["lanes"].items()}

    all_candidate_features = np.empty((len(candidate_pairs), len(FEATURE_NAMES)), dtype=np.float32)
    all_candidate_probabilities = np.empty(len(candidate_pairs), dtype=np.float32)
    batch_size = 10000
    for start in range(0, len(candidate_pairs), batch_size):
        stop = min(start + batch_size, len(candidate_pairs))
        batch_pairs = candidate_pairs[start:stop]
        features = feature_builder.matrix(batch_pairs)
        all_candidate_features[start:stop] = features
        all_candidate_probabilities[start:stop] = model.predict_proba(features)[:, 1]

    bootstrap_role_evidence: Counter[str] = Counter()
    bootstrap_role_with_of_evidence: Counter[str] = Counter()
    for pair_index, (left, right) in enumerate(candidate_pairs):
        features = all_candidate_features[pair_index]
        probability = float(all_candidate_probabilities[pair_index])
        roles = role_transform_tokens(tokens[left], tokens[right])
        if not roles:
            continue
        shared_tokens = set(tokens[left]) & set(tokens[right])
        rare_shared_anchor = any(token_document_frequency[token] <= 50 for token in shared_tokens)
        structurally_eligible = (
            probability >= lane_thresholds["token_reordering"]
            and float(features[5]) >= 0.72
            and float(features[6]) >= 0.95
            and float(features[4]) >= 0.60
            and float(features[8]) >= 0.70
            and float(features[0]) >= 0.82
            and float(features[2]) >= 0.82
            and bool(features[11])
            and rare_shared_anchor
        )
        if structurally_eligible:
            bootstrap_role_evidence.update(roles)
            if "of" in tokens[left] or "of" in tokens[right]:
                bootstrap_role_with_of_evidence.update(roles)
    bootstrapped_safe_roles = {
        role
        for role, count in bootstrap_role_evidence.items()
        if count >= 2
        or (bootstrap_role_with_of_evidence[role] >= 1 and token_document_frequency[role] >= 50)
    }
    safe_role_tokens.update(bootstrapped_safe_roles)

    accepted_edges: list[dict[str, Any]] = []
    rejected_numeric = 0
    rejected_without_identity_structure = 0
    rejected_unsafe_permutation = 0
    for pair_index, pair in enumerate(candidate_pairs):
        left, right = pair
        features = all_candidate_features[pair_index]
        probability = float(all_candidate_probabilities[pair_index])
        normalized_exact = bool(features[9])
        compact_exact = bool(features[13])
        numeric_equal = bool(features[11])
        if not numeric_equal and not (
            manual_seed[left] is not None and manual_seed[left] == manual_seed[right]
        ):
            rejected_numeric += 1
            continue
        if normalized_exact:
            accepted = True
            method = "normalized_exact"
            confidence = 1.0
        elif compact_exact:
            accepted = True
            method = "compact_normalized_exact"
            confidence = 1.0
        else:
            left_tokens = tokens[left]
            right_tokens = tokens[right]
            left_set = set(left_tokens)
            right_set = set(right_tokens)
            shared_tokens = left_set & right_set
            rare_shared_anchor = any(
                token_document_frequency[token] <= 50 for token in shared_tokens
            )
            pure_permutation = (
                Counter(left_tokens) == Counter(right_tokens) and left_tokens != right_tokens
            )
            roles = role_transform_tokens(left_tokens, right_tokens)
            safe_role_reordering = bool(set(roles) & safe_role_tokens)
            conjunction_reordering = pure_permutation and strict_conjunction_reordering(
                left_tokens, right_tokens
            )
            direct_manual_attachment = (manual_seed[left] is None) != (manual_seed[right] is None)
            token_reordering = (
                float(features[5]) >= 0.72
                and float(features[6]) >= 0.95
                and float(features[4]) >= 0.60
                and float(features[8]) >= 0.70
                and (safe_role_reordering or conjunction_reordering)
            )
            learned_role_template = (
                safe_role_reordering
                and float(features[5]) >= 0.72
                and float(features[6]) >= 0.95
                and float(features[4]) >= 0.60
                and float(features[8]) >= 0.70
            )
            orthographic_variant = (
                orthographic_variant_compatible(left_tokens, right_tokens)
                and float(features[4]) >= 0.78
                and float(features[7]) >= 0.70
                and float(features[0]) >= 0.86
                and float(features[2]) >= 0.86
            )
            accepted_lane = (
                "learned_role_template"
                if learned_role_template
                else "token_reordering"
                if (token_reordering and probability >= lane_thresholds["token_reordering"])
                else "orthographic_variant"
                if (orthographic_variant and probability >= lane_thresholds["orthographic_variant"])
                else "general"
                if (
                    (
                        safe_role_reordering
                        or conjunction_reordering
                        or (direct_manual_attachment and rare_shared_anchor)
                    )
                    and probability >= threshold
                )
                else None
            )
            accepted = (
                accepted_lane is not None
                and (
                    learned_role_template
                    or conjunction_reordering
                    or orthographic_variant
                    or (direct_manual_attachment and rare_shared_anchor)
                )
                and (not pure_permutation or safe_role_reordering or conjunction_reordering)
                and float(features[0]) >= 0.82
                and float(features[2]) >= 0.82
                and (float(features[4]) >= 0.55 or float(features[3]) >= 0.70)
            )
            method = (
                (
                    f"statistical_equivalence:{accepted_lane}:"
                    f"{'safe_role' if safe_role_reordering else 'conjunction' if conjunction_reordering else 'orthographic_variant' if orthographic_variant else 'manual_seed_attachment'}"
                )
                if accepted_lane is not None
                else "statistical_equivalence:rejected"
            )
            confidence = probability
            if (
                accepted_lane is not None
                and not learned_role_template
                and not conjunction_reordering
                and not orthographic_variant
                and not (direct_manual_attachment and rare_shared_anchor)
            ):
                rejected_without_identity_structure += 1
            if (
                accepted_lane is not None
                and pure_permutation
                and not safe_role_reordering
                and not conjunction_reordering
            ):
                rejected_unsafe_permutation += 1
        if not accepted:
            continue
        accepted_edges.append(
            {
                "left": left,
                "right": right,
                "leftTag": records[left]["tag"],
                "rightTag": records[right]["tag"],
                "method": method,
                "confidence": confidence,
                "features": {
                    name: round(float(features[index]), 6)
                    for index, name in enumerate(FEATURE_NAMES)
                },
            }
        )

    direct_manual_options: dict[int, dict[str, float]] = defaultdict(dict)
    for edge in accepted_edges:
        left = int(edge["left"])
        right = int(edge["right"])
        if manual_seed[left] is None and manual_seed[right] is not None:
            seed_id = str(manual_seed[right])
            direct_manual_options[left][seed_id] = max(
                direct_manual_options[left].get(seed_id, 0.0),
                float(edge["confidence"]),
            )
        elif manual_seed[right] is None and manual_seed[left] is not None:
            seed_id = str(manual_seed[left])
            direct_manual_options[right][seed_id] = max(
                direct_manual_options[right].get(seed_id, 0.0),
                float(edge["confidence"]),
            )
    ambiguous_direct = set()
    for index, options in direct_manual_options.items():
        ordered = sorted(options.values(), reverse=True)
        if len(ordered) >= 2 and ordered[0] - ordered[1] < 0.05:
            ambiguous_direct.add(index)

    union_find = UnionFind(len(records), manual_seed)
    for seed_id, members in manual_members.items():
        anchor = members[0]
        for member in members[1:]:
            merged, reason = union_find.union(anchor, member)
            if not merged:
                raise RuntimeError(f"Could not initialize manual seed {seed_id}: {reason}")

    accepted_edges.sort(
        key=lambda row: (
            row["method"] != "normalized_exact",
            -float(row["confidence"]),
            int(row["left"]),
            int(row["right"]),
        )
    )
    conflicts: list[dict[str, Any]] = []
    used_edges: list[dict[str, Any]] = []
    adjacency: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in accepted_edges:
        left = int(edge["left"])
        right = int(edge["right"])
        if (left in ambiguous_direct and manual_seed[right] is not None) or (
            right in ambiguous_direct and manual_seed[left] is not None
        ):
            conflicts.append({**edge, "rejectionReason": "ambiguous_manual_seed"})
            continue
        merged, reason = union_find.union(left, right)
        if not merged:
            conflicts.append({**edge, "rejectionReason": reason})
            continue
        used_edges.append(edge)
        adjacency[left].append(edge)
        adjacency[right].append(edge)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components[union_find.find(index)].append(index)

    manual_components: list[tuple[str, list[int]]] = []
    new_multi: list[list[int]] = []
    singletons: list[list[int]] = []
    for root, members in components.items():
        seed_id = union_find.seed[union_find.find(root)]
        if seed_id is not None:
            manual_components.append((str(seed_id), members))
        elif len(members) > 1:
            new_multi.append(members)
        else:
            singletons.append(members)
    manual_components.sort(key=lambda item: item[0])
    new_multi.sort(
        key=lambda members: (
            -sum(int(records[index]["frequency"]) for index in members),
            -len(members),
            min(str(records[index]["tag"]) for index in members),
        )
    )
    singletons.sort(
        key=lambda members: (
            -int(records[members[0]]["frequency"]),
            str(records[members[0]]["tag"]),
        )
    )

    cluster_specs: list[tuple[str, str | None, list[int]]] = []
    for seed_id, members in manual_components:
        cluster_specs.append((seed_id, seed_id, members))
    for ordinal, members in enumerate(new_multi, start=1):
        cluster_specs.append((f"NODE_ALIAS_NEW_{ordinal:05d}", None, members))
    for ordinal, members in enumerate(singletons, start=1):
        cluster_specs.append((f"NODE_SINGLETON_{ordinal:05d}", None, members))

    clusters: list[dict[str, Any]] = []
    assignment_by_index: dict[int, dict[str, Any]] = {}
    for cluster_id, seed_id, members in cluster_specs:
        members = sorted(members)
        if seed_id is not None:
            canonical = str(seed_meta[seed_id]["canonicalLabel"])
            cluster_type = "manual_seed_identity"
        elif len(members) > 1:
            canonical_index = sorted(
                members,
                key=lambda index: (
                    -int(records[index]["frequency"]),
                    len(str(records[index]["tag"])),
                    str(records[index]["tag"]),
                ),
            )[0]
            canonical = str(records[canonical_index]["tag"])
            cluster_type = "new_alias_component"
        else:
            canonical = str(records[members[0]]["tag"])
            cluster_type = "singleton"

        component_edges = [
            edge
            for index in members
            for edge in adjacency.get(index, [])
            if int(edge["left"]) < int(edge["right"])
        ]
        clusters.append(
            {
                "clusterId": cluster_id,
                "canonicalLabel": canonical,
                "clusterType": cluster_type,
                "manualSeedId": seed_id,
                "uniqueTags": len(members),
                "mentionCount": sum(int(records[index]["frequency"]) for index in members),
                "reviewedMemberCount": sum(manual_seed[index] is not None for index in members),
                "automaticMemberCount": sum(manual_seed[index] is None for index in members),
                "acceptedEquivalenceEdges": len(component_edges),
                "minimumEdgeConfidence": (
                    min(float(edge["confidence"]) for edge in component_edges)
                    if component_edges
                    else None
                ),
                "memberIndices": members,
            }
        )

        for index in members:
            if manual_seed[index] is not None:
                method = "fixed_manual_alias"
                linked_tag = None
                link_confidence = None
            elif len(members) == 1:
                method = "unresolved_singleton"
                linked_tag = None
                link_confidence = None
            else:
                incident = sorted(
                    adjacency.get(index, []),
                    key=lambda edge: (
                        -float(edge["confidence"]),
                        edge["method"] != "normalized_exact",
                    ),
                )
                if incident:
                    link = incident[0]
                    other = int(link["right"]) if int(link["left"]) == index else int(link["left"])
                    method = str(link["method"])
                    linked_tag = str(records[other]["tag"])
                    link_confidence = float(link["confidence"])
                else:
                    method = "transitive_alias_component"
                    linked_tag = None
                    link_confidence = None
            assignment_by_index[index] = {
                "index": index,
                "tag": records[index]["tag"],
                "frequency": int(records[index]["frequency"]),
                "clusterId": cluster_id,
                "canonicalLabel": canonical,
                "clusterType": cluster_type,
                "manualSeedId": seed_id,
                "manualSeedMember": manual_seed[index] is not None,
                "assignmentMethod": method,
                "linkedTag": linked_tag,
                "linkConfidence": link_confidence,
            }

    if sorted(assignment_by_index) != list(range(len(records))):
        raise RuntimeError("Output node assignments are incomplete")
    assignments = [assignment_by_index[index] for index in range(len(records))]

    write_json(OUTPUT / "node_identity_clusters.json", clusters)
    write_jsonl(OUTPUT / "node_identity_assignments.jsonl", assignments)
    write_assignment_csv(OUTPUT / "node_identity_assignments.csv", assignments)
    write_jsonl(OUTPUT / "accepted_equivalence_edges.jsonl", used_edges)
    write_jsonl(OUTPUT / "rejected_conflicting_edges.jsonl", conflicts)
    write_markdown_inventory(
        OUTPUT / "NODE_IDENTITY_ALIAS_CLUSTERS.md",
        clusters,
        assignments,
        include_singletons=False,
    )
    write_markdown_inventory(
        OUTPUT / "NODE_IDENTITY_ALL_CLUSTERS.md",
        clusters,
        assignments,
        include_singletons=True,
    )

    hashes_after = {str(path): sha256(path) for path in protected_paths}
    if hashes_before != hashes_after:
        raise RuntimeError("A protected source file changed during the run")

    size_distribution = Counter(
        "1"
        if len(members) == 1
        else "2"
        if len(members) == 2
        else "3-4"
        if len(members) <= 4
        else "5-9"
        if len(members) <= 9
        else "10+"
        for members in components.values()
    )
    report = {
        "method": "supervised high-precision pair equivalence plus constrained components",
        "records": len(records),
        "mentions": sum(int(row["frequency"]) for row in records),
        "manualReviewedTags": sum(seed is not None for seed in manual_seed),
        "manualSeedClusters": len(manual_members),
        "manualSeedAudit": manual_audit,
        "candidateGeneration": candidate_report,
        "textFeatures": text_report,
        "training": training_report,
        "learnedRoleTokenEvidence": role_evidence,
        "bootstrappedRoleTokenEvidence": dict(sorted(bootstrap_role_evidence.items())),
        "bootstrappedRoleWithOfEvidence": dict(sorted(bootstrap_role_with_of_evidence.items())),
        "bootstrappedSafeRoleTokens": sorted(bootstrapped_safe_roles),
        "safeRoleTokens": sorted(safe_role_tokens),
        "features": FEATURE_NAMES,
        "calibration": calibration,
        "acceptanceSafetyFloors": {
            "geminiTagCosine": 0.82,
            "geminiFusedCosine": 0.82,
            "charTfidfCosineOr": 0.55,
            "wordTfidfCosineOr": 0.70,
            "numericSignatureMustMatch": True,
            "manualSeedMargin": 0.05,
        },
        "acceptedCandidateEdges": len(accepted_edges),
        "usedEquivalenceEdges": len(used_edges),
        "rejectedNumericCandidatePairs": rejected_numeric,
        "rejectedWithoutIdentityStructure": rejected_without_identity_structure,
        "rejectedUnsafeTokenPermutations": rejected_unsafe_permutation,
        "rejectedConflictingEdges": len(conflicts),
        "ambiguousDirectManualCandidates": len(ambiguous_direct),
        "outputClusters": len(clusters),
        "manualSeedIdentityClusters": len(manual_components),
        "newAliasComponents": len(new_multi),
        "unseededSingletons": len(singletons),
        "allSingletonComponents": sum(len(members) == 1 for members in components.values()),
        "multiTagClusters": len(manual_components)
        - sum(len(members) == 1 for _, members in manual_components)
        + len(new_multi),
        "sizeDistribution": dict(size_distribution),
        "sourceHashesBefore": hashes_before,
        "sourceHashesAfter": hashes_after,
        "sourcesUnchanged": hashes_before == hashes_after,
        "discardedEntityResolutionInputsUsed": False,
    }
    write_json(OUTPUT / "run_report.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

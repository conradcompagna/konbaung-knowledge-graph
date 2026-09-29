from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import GaussianRandomProjection


ROOT = Path(__file__).resolve().parents[2]
CLUSTER_ROOT = ROOT / "konbaung_v3_node_edge_clustering_first_pass_20260724"
EMBEDDING_ROOT = ROOT / "konbaung_v3_eight_view_embeddings_20260724"
OUTPUT_ROOT = ROOT / "konbaung_v3_entity_resolution_two_pass_20260724"
TOKEN_ROOT = OUTPUT_ROOT / "fasttext_tag_tokens"
ALIAS_ROOT = OUTPUT_ROOT / "pass1_alias_resolution"
COREF_ROOT = OUTPUT_ROOT / "pass2_coreference"
RANDOM_SEED = 20260724

# These are identity seeds, not string-matching rules.  Each seed names a
# statistically resolved pass-1 alias cluster whose occurrences can serve as
# contextual examples for the second-pass linker.  Separate pass-1 clusters
# are retained here when the deliberately conservative alias threshold did
# not merge them automatically.
RULER_IDENTITY_SEEDS = {
    "Alaungpaya": (
        "Alaungpaya",
        "Alaungmintayagyi",
        "King_Alaungpaya",
        "King Alaungmintaya",
        "Alaungmintayagyi_Phaya",
        "Alaungmintaya-gyi",
        "alaungpaya",
        "Alaungmintaya-gyi-phaya",
        "alaungmintaya_gyi",
        "Alaung Mintayagyi",
    ),
    "Hsinbyushin": (
        "King Hsinbyushin",
        "Hsinbyushin_King",
        "Hsinbyushin",
        "King Hsinbyushin (Siri Dhamma Raja)",
    ),
    "Singu Min": (
        "Singu Min",
        "King Singu Min",
    ),
    "Phaungkaza Maung Maung": (
        "Maung Maung",
        "Prince Maung Maung",
    ),
    "Bodawpaya": (
        "King Bodawpaya",
        "Bodawpaya",
        "King_Bodawpaya",
        "Late King Bodawpaya",
    ),
    "Bagyidaw": (
        "King Bagyidaw",
        "King_Bagyidaw",
        "Bagyidaw",
    ),
    "Tharrawaddy": (
        "King Tharrawaddy",
        "Tharrawaddy",
        "King_Tharrawaddy",
        "Tharrawaddy Min",
        "Tharyarwady Min",
        "Tharyarwady King",
    ),
    "Pagan Min": (
        "Pagan Min",
        "King Pagan",
        "Pagan_King",
    ),
    "Mindon": (
        "King Mindon",
        "King_Mindon",
    ),
    "Thibaw": ("King Thibaw",),
}


def jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_rows(matrix: np.ndarray, chunk: int = 4096) -> None:
    for start in range(0, matrix.shape[0], chunk):
        block = np.asarray(matrix[start : start + chunk], dtype=np.float32)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        np.divide(block, norms, out=block, where=norms > 0)
        matrix[start : start + chunk] = block


def normalized_display(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    output: list[str] = []
    for char in value:
        category = unicodedata.category(char)
        output.append(char if category[0] in {"L", "N", "M"} else " ")
    return " ".join("".join(output).split())


def load_token_vectors(language: str) -> tuple[np.ndarray, dict[str, int]]:
    vector_path = TOKEN_ROOT / f"{language}_token_vectors.npy"
    index_path = TOKEN_ROOT / f"{language}_tokens.jsonl"
    if not vector_path.exists():
        return np.empty((0, 300), dtype=np.float32), {}
    vectors = np.load(vector_path, mmap_mode="r")
    index = {row["token"]: int(row["index"]) for row in jsonl(index_path)}
    return vectors, index


def build_fasttext_tag_vectors(records: list[dict]) -> tuple[np.ndarray, list[dict]]:
    path = ALIAS_ROOT / "fasttext_tag_vectors.npy"
    report_path = ALIAS_ROOT / "fasttext_tag_vector_report.json"
    if path.exists() and report_path.exists():
        return np.load(path, mmap_mode="r"), json.loads(report_path.read_text(encoding="utf-8"))[
            "rows"
        ]

    language_vectors = {}
    language_indexes = {}
    for language in ("en", "my"):
        vectors, index = load_token_vectors(language)
        language_vectors[language] = vectors
        language_indexes[language] = index

    tag_token_rows = {row["tag"]: row for row in jsonl(TOKEN_ROOT / "tag_tokens.jsonl")}
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(records), 300))
    row_reports: list[dict] = []
    covered = 0
    for record in records:
        token_row = tag_token_rows[record["tag"]]
        vector_sum = np.zeros(300, dtype=np.float32)
        weight_sum = 0.0
        atomic_found = 0
        phrase_found = 0
        for token in token_row["coveredTokens"]:
            language = token["language"]
            vector_index = int(token["vectorIndex"])
            document_frequency = max(1, int(token["documentFrequency"]))
            component_count = token["token"].count("_") + 1
            idf = math.log((len(records) + 1) / (document_frequency + 1)) + 1.0
            weight = idf / math.sqrt(component_count)
            vector_sum += (
                np.asarray(language_vectors[language][vector_index], dtype=np.float32) * weight
            )
            weight_sum += weight
            if component_count == 1:
                atomic_found += 1
            else:
                phrase_found += 1
        if weight_sum > 0:
            vector_sum /= weight_sum
            norm = float(np.linalg.norm(vector_sum))
            if norm > 0:
                vector_sum /= norm
                covered += 1
        output[int(record["index"])] = vector_sum
        row_reports.append(
            {
                "index": int(record["index"]),
                "tag": record["tag"],
                "atomicTokensFound": atomic_found,
                "phraseTokensFound": phrase_found,
                "hasVector": bool(weight_sum > 0),
            }
        )
    output.flush()
    report = {
        "records": len(records),
        "covered": covered,
        "coverage": covered / len(records),
        "pooling": (
            "IDF-weighted mean of separately embedded tag components and "
            "available multi-token vocabulary entries"
        ),
        "rows": row_reports,
    }
    write_json(report_path, report)
    return np.load(path, mmap_mode="r"), row_reports


def build_relation_vectors(
    records: list[dict],
    tag_to_index: dict[str, int],
) -> np.ndarray:
    path = ALIAS_ROOT / "incident_relation_vectors.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")

    edge_records = list(jsonl(CLUSTER_ROOT / "edge_records.jsonl"))
    edge_index = {row["tag"]: int(row["index"]) for row in edge_records}
    edge_vectors = np.load(CLUSTER_ROOT / "edge_base_vectors.npy", mmap_mode="r")
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(records), 768))
    counts = np.zeros(len(records), dtype=np.int32)
    for occurrence in jsonl(EMBEDDING_ROOT / "occurrences.jsonl"):
        predicate_vector = np.asarray(
            edge_vectors[edge_index[occurrence["predicate"]]], dtype=np.float32
        )
        for tag in (occurrence["subject"], occurrence["object"]):
            index = tag_to_index[tag]
            output[index] += predicate_vector
            counts[index] += 1
    normalize_rows(output)
    output.flush()
    np.save(ALIAS_ROOT / "incident_relation_counts.npy", counts)
    return np.load(path, mmap_mode="r")


def pair_features(
    pairs: np.ndarray,
    base: np.ndarray,
    context: np.ndarray,
    fasttext: np.ndarray,
    relation: np.ndarray,
) -> np.ndarray:
    output = np.empty((len(pairs), 5), dtype=np.float32)
    for start in range(0, len(pairs), 50000):
        batch = pairs[start : start + 50000]
        left = batch[:, 0]
        right = batch[:, 1]
        base_similarity = np.einsum("ij,ij->i", base[left], base[right])
        context_similarity = np.einsum("ij,ij->i", context[left], context[right])
        relation_similarity = np.einsum("ij,ij->i", relation[left], relation[right])
        fasttext_similarity = np.einsum("ij,ij->i", fasttext[left], fasttext[right])
        fasttext_available = (np.linalg.norm(fasttext[left], axis=1) > 0) & (
            np.linalg.norm(fasttext[right], axis=1) > 0
        )
        output[start : start + len(batch), 0] = base_similarity
        output[start : start + len(batch), 1] = context_similarity
        output[start : start + len(batch), 2] = relation_similarity
        output[start : start + len(batch), 3] = fasttext_similarity
        output[start : start + len(batch), 4] = fasttext_available.astype(np.float32)
    return output


def composite_similarity(features: np.ndarray) -> np.ndarray:
    weights = np.array([0.40, 0.15, 0.20, 0.25], dtype=np.float32)
    available = np.ones((len(features), 4), dtype=np.float32)
    available[:, 3] = features[:, 4]
    effective = available * weights
    denominator = effective.sum(axis=1)
    return (features[:, :4] * effective).sum(axis=1) / denominator


def all_pair_similarity(
    members: list[int],
    base: np.ndarray,
    context: np.ndarray,
    fasttext: np.ndarray,
    relation: np.ndarray,
) -> np.ndarray:
    indices = np.asarray(members, dtype=np.int64)
    matrices = [base[indices], context[indices], relation[indices], fasttext[indices]]
    weights = [0.40, 0.15, 0.20, 0.25]
    available_ft = np.linalg.norm(matrices[3], axis=1) > 0
    similarity = np.zeros((len(indices), len(indices)), dtype=np.float32)
    denominator = np.zeros_like(similarity)
    for position, (matrix, weight) in enumerate(zip(matrices, weights)):
        part = np.asarray(matrix, dtype=np.float32) @ np.asarray(matrix, dtype=np.float32).T
        if position == 3:
            available = np.outer(available_ft, available_ft).astype(np.float32)
        else:
            available = np.ones_like(part)
        similarity += part * weight * available
        denominator += weight * available
    np.divide(similarity, denominator, out=similarity, where=denominator > 0)
    np.fill_diagonal(similarity, 1.0)
    return np.clip(similarity, -1.0, 1.0)


def canonical_member(members: list[int], records: list[dict]) -> int:
    return min(
        members,
        key=lambda index: (
            -int(records[index]["frequency"]),
            records[index]["tag"].count("_"),
            len(records[index]["tag"]),
            records[index]["tag"].casefold(),
        ),
    )


def run_alias_resolution() -> None:
    ALIAS_ROOT.mkdir(parents=True, exist_ok=True)
    records = list(jsonl(CLUSTER_ROOT / "node_records.jsonl"))
    base = np.load(CLUSTER_ROOT / "node_base_vectors.npy", mmap_mode="r")
    context = np.load(CLUSTER_ROOT / "node_context_vectors.npy", mmap_mode="r")

    knn = np.load(CLUSTER_ROOT / "node_knn.npz")
    pair_set: set[tuple[int, int]] = set()
    for left in range(len(records)):
        for right in knn["indices"][left]:
            right = int(right)
            if left != right:
                pair_set.add((min(left, right), max(left, right)))
    lexical_groups: dict[str, list[int]] = defaultdict(list)
    for record in records:
        lexical_groups[normalized_display(record["tag"])].append(int(record["index"]))
    for members in lexical_groups.values():
        if len(members) < 2:
            continue
        for left_position, left in enumerate(members):
            for right in members[left_position + 1 :]:
                pair_set.add((min(left, right), max(left, right)))
    pairs = np.asarray(sorted(pair_set), dtype=np.int32)
    pair_cache = ALIAS_ROOT / "gemini_candidate_pairs.npy"
    feature_cache = ALIAS_ROOT / "gemini_candidate_pair_features.npy"
    if pair_cache.exists() and feature_cache.exists():
        cached_pairs = np.load(pair_cache)
        if cached_pairs.shape != pairs.shape or not np.array_equal(cached_pairs, pairs):
            raise RuntimeError("Cached Gemini candidate pairs do not match")
        features = np.load(feature_cache)
    else:
        features = np.empty((len(pairs), 2), dtype=np.float32)
        for start in range(0, len(pairs), 50000):
            batch = pairs[start : start + 50000]
            left = batch[:, 0]
            right = batch[:, 1]
            features[start : start + len(batch), 0] = np.einsum("ij,ij->i", base[left], base[right])
            features[start : start + len(batch), 1] = np.einsum(
                "ij,ij->i", context[left], context[right]
            )
        np.save(pair_cache, pairs)
        np.save(feature_cache, features)
    composite = features @ np.asarray([0.20, 0.80], dtype=np.float32)

    rng = np.random.default_rng(RANDOM_SEED)
    fit_indices = (
        rng.choice(len(pairs), size=150000, replace=False)
        if len(pairs) > 150000
        else np.arange(len(pairs))
    )
    mixture = GaussianMixture(
        n_components=5,
        covariance_type="full",
        reg_covar=1e-5,
        max_iter=300,
        random_state=RANDOM_SEED,
    )
    mixture.fit(features[fit_indices])
    means = mixture.means_
    component_scores = means @ np.asarray([0.20, 0.80])
    alias_component = int(np.argmax(component_scores))
    posterior = mixture.predict_proba(features)[:, alias_component]

    tag_sentence_ids: list[set[str]] = [set() for _ in records]
    tag_to_index = {row["tag"]: int(row["index"]) for row in records}
    for occurrence in jsonl(EMBEDDING_ROOT / "occurrences.jsonl"):
        sid = occurrence["sid"]
        tag_sentence_ids[tag_to_index[occurrence["subject"]]].add(sid)
        tag_sentence_ids[tag_to_index[occurrence["object"]]].add(sid)
    format_equivalent = np.zeros(len(pairs), dtype=bool)
    independent_context = np.zeros(len(pairs), dtype=bool)
    cooccurring_list_pair = np.zeros(len(pairs), dtype=bool)
    for pair_index, (left, right) in enumerate(pairs):
        left = int(left)
        right = int(right)
        format_equivalent[pair_index] = normalized_display(
            records[left]["tag"]
        ) == normalized_display(records[right]["tag"])
        left_sids = tag_sentence_ids[left]
        right_sids = tag_sentence_ids[right]
        shared = len(left_sids & right_sids)
        smaller = min(len(left_sids), len(right_sids))
        shared_ratio = shared / max(1, smaller)
        left_independent = len(left_sids) - shared
        right_independent = len(right_sids) - shared
        independent_context[pair_index] = (
            len(left_sids) >= 3
            and len(right_sids) >= 3
            and left_independent >= 3
            and right_independent >= 3
            and shared_ratio <= 0.10
        )
        cooccurring_list_pair[pair_index] = shared_ratio > 0.10

    alias_mean = float(component_scores[alias_component])
    auto_floor = max(0.975, alias_mean - 0.012)
    auto_mask = (
        (posterior >= 0.99)
        & (composite >= auto_floor)
        & (features[:, 1] >= 0.975)
        & (np.maximum(features[:, 0], features[:, 1]) >= 0.98)
        & (format_equivalent | independent_context)
    )
    review_mask = (posterior >= 0.70) & (composite >= auto_floor - 0.025) & ~auto_mask

    graph = nx.Graph()
    graph.add_nodes_from(range(len(records)))
    for pair, probability, score in zip(
        pairs[auto_mask], posterior[auto_mask], composite[auto_mask]
    ):
        graph.add_edge(
            int(pair[0]),
            int(pair[1]),
            probability=float(probability),
            similarity=float(score),
        )

    similarity_floor = max(
        0.972,
        float(np.quantile(composite[auto_mask], 0.05)) if np.any(auto_mask) else 0.975,
    )
    final_groups: list[list[int]] = []
    split_components = 0
    for component in nx.connected_components(graph):
        members = sorted(component)
        if len(members) <= 1:
            final_groups.append(members)
            continue
        indices = np.asarray(members, dtype=np.int64)
        base_similarities = (
            np.asarray(base[indices], dtype=np.float32)
            @ np.asarray(base[indices], dtype=np.float32).T
        )
        context_similarities = (
            np.asarray(context[indices], dtype=np.float32)
            @ np.asarray(context[indices], dtype=np.float32).T
        )
        similarities = 0.20 * base_similarities + 0.80 * context_similarities
        similarities[np.maximum(base_similarities, context_similarities) < 0.98] = 0.0
        np.fill_diagonal(similarities, 1.0)
        if float(similarities.min()) >= similarity_floor:
            final_groups.append(members)
            continue
        split_components += 1
        distances = np.clip(1.0 - similarities, 0.0, 2.0)
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="complete",
            distance_threshold=1.0 - similarity_floor,
        )
        labels = model.fit_predict(distances)
        grouped: dict[int, list[int]] = defaultdict(list)
        for member, label in zip(members, labels):
            grouped[int(label)].append(member)
        final_groups.extend(sorted(group) for group in grouped.values())

    final_groups.sort(
        key=lambda members: (
            -sum(int(records[index]["frequency"]) for index in members),
            records[canonical_member(members, records)]["tag"].casefold(),
        )
    )
    assignments: list[dict | None] = [None] * len(records)
    clusters: list[dict] = []
    for cluster_number, members in enumerate(final_groups, start=1):
        canonical_index = canonical_member(members, records)
        cluster_id = f"ALIAS_{cluster_number:05d}"
        if len(members) > 1:
            indices = np.asarray(members, dtype=np.int64)
            base_similarities = (
                np.asarray(base[indices], dtype=np.float32)
                @ np.asarray(base[indices], dtype=np.float32).T
            )
            context_similarities = (
                np.asarray(context[indices], dtype=np.float32)
                @ np.asarray(context[indices], dtype=np.float32).T
            )
            similarities = 0.20 * base_similarities + 0.80 * context_similarities
            similarities[np.maximum(base_similarities, context_similarities) < 0.98] = 0.0
            np.fill_diagonal(similarities, 1.0)
            confidence = float(similarities[np.triu_indices(len(members), k=1)].min())
        else:
            confidence = 1.0
        variants = sorted(
            (
                {
                    "tag": records[index]["tag"],
                    "frequency": int(records[index]["frequency"]),
                }
                for index in members
            ),
            key=lambda row: (-row["frequency"], row["tag"].casefold()),
        )
        cluster = {
            "aliasClusterId": cluster_id,
            "canonicalTag": records[canonical_index]["tag"],
            "totalFrequency": sum(row["frequency"] for row in variants),
            "variantCount": len(variants),
            "minimumEmbeddingSimilarity": round(confidence, 6),
            "variants": variants,
        }
        clusters.append(cluster)
        for index in members:
            assignments[index] = {
                "index": index,
                "tag": records[index]["tag"],
                "frequency": int(records[index]["frequency"]),
                "aliasClusterId": cluster_id,
                "canonicalTag": records[canonical_index]["tag"],
                "clusterConfidence": round(confidence, 6),
            }

    with (ALIAS_ROOT / "alias_assignments.jsonl").open("w", encoding="utf-8") as handle:
        for row in assignments:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_json(ALIAS_ROOT / "alias_clusters.json", clusters)

    review_pairs = []
    for pair, probability, score, feature in zip(
        pairs[review_mask],
        posterior[review_mask],
        composite[review_mask],
        features[review_mask],
    ):
        review_pairs.append(
            {
                "left": records[int(pair[0])]["tag"],
                "right": records[int(pair[1])]["tag"],
                "probability": round(float(probability), 6),
                "compositeSimilarity": round(float(score), 6),
                "baseSimilarity": round(float(feature[0]), 6),
                "contextSimilarity": round(float(feature[1]), 6),
            }
        )
    review_pairs.sort(key=lambda row: (-row["probability"], -row["compositeSimilarity"]))
    write_json(ALIAS_ROOT / "review_pairs.json", review_pairs)

    diagnostic_names = [
        "Alaungpaya",
        "Alaungmintaya",
        "King Alaungpaya",
        "Alaungmintayagyi",
        "King Mindon",
        "Mindon Min",
        "King",
        "Konbaung King",
    ]
    diagnostics = {
        name: assignments[tag_to_index[name]] for name in diagnostic_names if name in tag_to_index
    }
    report = {
        "method": (
            "Gemini tag/context embedding similarity graph followed by "
            "Gaussian-mixture identity-component inference and complete-linkage "
            "anti-chaining splits"
        ),
        "featuresUsed": {
            "GeminiTagEmbedding": 0.20,
            "GeminiSentenceContextEmbedding": 0.80,
            "fastTextUsed": False,
            "relationProfileUsed": False,
        },
        "records": len(records),
        "candidatePairs": len(pairs),
        "autoPairs": int(auto_mask.sum()),
        "reviewPairs": int(review_mask.sum()),
        "mixtureMeans": means.tolist(),
        "mixtureWeights": mixture.weights_.tolist(),
        "aliasComponent": alias_component,
        "aliasComponentCompositeMean": alias_mean,
        "autoCompositeFloor": auto_floor,
        "minimumSingleViewSimilarity": 0.98,
        "independentContextRule": {
            "minimumDistinctSentencesPerTag": 3,
            "minimumNonSharedSentencesPerTag": 3,
            "maximumSharedSentenceRatio": 0.10,
            "formatEquivalentTagsExempt": True,
        },
        "formatEquivalentCandidatePairs": int(format_equivalent.sum()),
        "independentContextQualifiedPairs": int(independent_context.sum()),
        "cooccurringListPairsRejected": int(np.sum(cooccurring_list_pair & ~format_equivalent)),
        "completeLinkSimilarityFloor": similarity_floor,
        "connectedComponentsSplit": split_components,
        "clusters": len(clusters),
        "multiVariantClusters": sum(cluster["variantCount"] > 1 for cluster in clusters),
        "tagsInMultiVariantClusters": sum(
            cluster["variantCount"] for cluster in clusters if cluster["variantCount"] > 1
        ),
        "diagnostics": diagnostics,
        "sourceFilesModified": False,
    }
    write_json(ALIAS_ROOT / "alias_resolution_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


class VectorStore:
    def __init__(self, name: str) -> None:
        vector_path = EMBEDDING_ROOT / "vectors" / f"{name}.npy"
        key_path = EMBEDDING_ROOT / "vectors" / f"{name}.keys.jsonl"
        self.vectors = np.load(vector_path, mmap_mode="r")
        self.key_to_row = {row["key"]: int(row["row"]) for row in jsonl(key_path)}

    def combine(self, keys: list[str]) -> np.ndarray:
        rows = [self.key_to_row[key] for key in keys]
        vector = np.asarray(self.vectors[rows], dtype=np.float32).mean(axis=0)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector


def build_mentions(
    tag_to_index: dict[str, int],
    alias_by_tag: dict[str, dict],
    alias_index: dict[str, int],
    base: np.ndarray,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    COREF_ROOT.mkdir(parents=True, exist_ok=True)
    record_path = COREF_ROOT / "mention_records.jsonl"
    residual_path = COREF_ROOT / "mention_context_residual_vectors.npy"
    triple_path = COREF_ROOT / "mention_triple_context_vectors.npy"
    manifest_path = COREF_ROOT / "mention_vector_manifest.json"
    alias_hash = sha256(ALIAS_ROOT / "alias_assignments.jsonl")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("aliasAssignmentsSha256") == alias_hash:
            records = list(jsonl(record_path))
            return (
                records,
                np.load(residual_path, mmap_mode="r"),
                np.load(triple_path, mmap_mode="r"),
            )

    occurrences = list(jsonl(EMBEDDING_ROOT / "occurrences.jsonl"))
    mention_count = len(occurrences) * 2
    residuals = np.lib.format.open_memmap(
        residual_path, mode="w+", dtype=np.float32, shape=(mention_count, 768)
    )
    triples = np.lib.format.open_memmap(
        triple_path, mode="w+", dtype=np.float32, shape=(mention_count, 768)
    )
    argument_context = VectorStore("argument_context")
    triple_context = VectorStore("triple_context")
    records: list[dict] = []

    with record_path.open("w", encoding="utf-8") as handle:
        for occurrence_position, occurrence in enumerate(occurrences):
            triple_vector = triple_context.combine(occurrence["embeddingKeys"]["SPOBE"])
            endpoint_specs = (
                (
                    occurrence["subject"],
                    occurrence["object"],
                    occurrence["embeddingKeys"]["SBE"],
                ),
                (
                    occurrence["object"],
                    occurrence["subject"],
                    occurrence["embeddingKeys"]["OBE"],
                ),
            )
            for endpoint_ordinal, (tag, counterpart, context_keys) in enumerate(endpoint_specs):
                mention_index = occurrence_position * 2 + endpoint_ordinal
                tag_index = tag_to_index[tag]
                context_vector = argument_context.combine(context_keys)
                base_vector = np.asarray(base[tag_index], dtype=np.float32)
                residual = context_vector - float(context_vector @ base_vector) * base_vector
                residual_norm = float(np.linalg.norm(residual))
                if residual_norm > 0:
                    residual /= residual_norm
                residuals[mention_index] = residual
                triples[mention_index] = triple_vector
                alias_assignment = alias_by_tag[tag]
                row = {
                    "mentionIndex": mention_index,
                    "mentionId": f"{occurrence['occurrenceId']}:n{endpoint_ordinal + 1}",
                    "occurrenceId": occurrence["occurrenceId"],
                    "endpointOrdinal": endpoint_ordinal,
                    "tag": tag,
                    "tagIndex": tag_index,
                    "aliasClusterId": alias_assignment["aliasClusterId"],
                    "aliasIndex": alias_index[alias_assignment["aliasClusterId"]],
                    "canonicalTag": alias_assignment["canonicalTag"],
                    "counterpartTag": counterpart,
                    "predicate": occurrence["predicate"],
                    "sid": occurrence["sid"],
                    "volumeId": occurrence["volumeId"],
                    "ownerPage": int(occurrence["ownerPage"]),
                }
                records.append(row)
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    residuals.flush()
    triples.flush()
    write_json(
        manifest_path,
        {
            "mentions": mention_count,
            "aliasAssignmentsSha256": alias_hash,
            "endpointRoleUsedAsFeature": False,
            "contextRepresentation": (
                "Gemini argument-with-sentence embedding with the projection "
                "onto the tag-only embedding removed"
            ),
            "tripleContextRepresentation": "Gemini triple-with-sentence embedding",
        },
    )
    return records, np.load(residual_path, mmap_mode="r"), np.load(triple_path, mmap_mode="r")


def build_alias_statistics(
    alias_clusters: list[dict],
    alias_by_tag: dict[str, dict],
    tag_to_index: dict[str, int],
    mention_records: list[dict],
    residuals: np.ndarray,
    triples: np.ndarray,
    base: np.ndarray,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    statistics_path = COREF_ROOT / "alias_ambiguity_statistics.json"
    base_path = COREF_ROOT / "alias_base_centroids.npy"
    residual_path = COREF_ROOT / "alias_residual_centroids.npy"
    triple_path = COREF_ROOT / "alias_triple_centroids.npy"
    manifest_path = COREF_ROOT / "alias_statistics_manifest.json"
    alias_hash = sha256(ALIAS_ROOT / "alias_assignments.jsonl")
    if (
        manifest_path.exists()
        and statistics_path.exists()
        and base_path.exists()
        and residual_path.exists()
        and triple_path.exists()
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("aliasAssignmentsSha256") == alias_hash:
            return (
                json.loads(statistics_path.read_text(encoding="utf-8")),
                np.load(base_path, mmap_mode="r"),
                np.load(residual_path, mmap_mode="r"),
                np.load(triple_path, mmap_mode="r"),
            )

    alias_index = {cluster["aliasClusterId"]: index for index, cluster in enumerate(alias_clusters)}
    count = len(alias_clusters)
    base_sums = np.zeros((count, 768), dtype=np.float32)
    base_weights = np.zeros(count, dtype=np.float32)
    for cluster in alias_clusters:
        cluster_index = alias_index[cluster["aliasClusterId"]]
        for variant in cluster["variants"]:
            frequency = float(variant["frequency"])
            base_sums[cluster_index] += (
                np.asarray(base[tag_to_index[variant["tag"]]], dtype=np.float32) * frequency
            )
            base_weights[cluster_index] += frequency
    normalize_rows(base_sums)

    mention_counts = np.zeros(count, dtype=np.int32)
    pages: list[set[tuple[str, int]]] = [set() for _ in range(count)]
    mention_aliases = np.fromiter(
        (int(row["aliasIndex"]) for row in mention_records),
        dtype=np.int32,
        count=len(mention_records),
    )
    mention_counts[:] = np.bincount(mention_aliases, minlength=count)
    membership = csr_matrix(
        (
            np.ones(len(mention_aliases), dtype=np.float32),
            (mention_aliases, np.arange(len(mention_aliases), dtype=np.int32)),
        ),
        shape=(count, len(mention_aliases)),
    )
    residual_sums = np.asarray(
        membership @ np.asarray(residuals, dtype=np.float32), dtype=np.float32
    )
    triple_sums = np.asarray(membership @ np.asarray(triples, dtype=np.float32), dtype=np.float32)
    for row in mention_records:
        cluster_index = int(row["aliasIndex"])
        pages[cluster_index].add((row["volumeId"], int(row["ownerPage"])))
    residual_cohesion = np.linalg.norm(
        residual_sums / np.maximum(mention_counts[:, None], 1), axis=1
    )
    triple_cohesion = np.linalg.norm(triple_sums / np.maximum(mention_counts[:, None], 1), axis=1)
    normalize_rows(residual_sums)
    normalize_rows(triple_sums)

    node_knn = np.load(CLUSTER_ROOT / "node_knn.npz")
    index_to_tag = [None] * len(tag_to_index)
    for tag, index in tag_to_index.items():
        index_to_tag[index] = tag
    hubness = np.zeros(count, dtype=np.float32)
    for cluster in alias_clusters:
        cluster_index = alias_index[cluster["aliasClusterId"]]
        canonical_index = tag_to_index[cluster["canonicalTag"]]
        distinct_neighbors: dict[int, float] = {}
        for neighbor, similarity in zip(
            node_knn["indices"][canonical_index],
            node_knn["similarities"][canonical_index],
        ):
            neighbor_tag = alias_by_tag[index_to_tag[int(neighbor)]]
            neighbor_alias = alias_index[neighbor_tag["aliasClusterId"]]
            if neighbor_alias == cluster_index:
                continue
            distinct_neighbors[neighbor_alias] = max(
                distinct_neighbors.get(neighbor_alias, -1.0), float(similarity)
            )
        top = sorted(distinct_neighbors.values(), reverse=True)[:10]
        hubness[cluster_index] = float(np.mean(top)) if top else 0.0

    statistics = []
    for index, cluster in enumerate(alias_clusters):
        cohesion = 0.75 * float(residual_cohesion[index]) + 0.25 * float(triple_cohesion[index])
        ambiguity_score = (
            math.log1p(int(mention_counts[index]))
            * max(0.01, 1.0 - cohesion)
            * max(0.1, float(hubness[index]))
        )
        statistics.append(
            {
                "aliasIndex": index,
                "aliasClusterId": cluster["aliasClusterId"],
                "canonicalTag": cluster["canonicalTag"],
                "mentionCount": int(mention_counts[index]),
                "variantCount": int(cluster["variantCount"]),
                "pageCount": len(pages[index]),
                "residualContextCohesion": round(float(residual_cohesion[index]), 6),
                "tripleContextCohesion": round(float(triple_cohesion[index]), 6),
                "combinedCohesion": round(cohesion, 6),
                "embeddingHubness": round(float(hubness[index]), 6),
                "ambiguityScore": round(ambiguity_score, 6),
            }
        )
    statistics.sort(key=lambda row: -row["ambiguityScore"])
    write_json(statistics_path, statistics)
    np.save(base_path, base_sums)
    np.save(residual_path, residual_sums)
    np.save(triple_path, triple_sums)
    write_json(
        manifest_path,
        {
            "aliasAssignmentsSha256": alias_hash,
            "aliasClusters": len(alias_clusters),
            "mentions": len(mention_records),
        },
    )
    return statistics, base_sums, residual_sums, triple_sums


def run_coreference() -> None:
    COREF_ROOT.mkdir(parents=True, exist_ok=True)
    node_records = list(jsonl(CLUSTER_ROOT / "node_records.jsonl"))
    tag_to_index = {row["tag"]: int(row["index"]) for row in node_records}
    index_to_tag = [row["tag"] for row in node_records]
    alias_clusters = json.loads((ALIAS_ROOT / "alias_clusters.json").read_text(encoding="utf-8"))
    alias_assignments = list(jsonl(ALIAS_ROOT / "alias_assignments.jsonl"))
    alias_by_tag = {row["tag"]: row for row in alias_assignments}
    alias_index = {cluster["aliasClusterId"]: index for index, cluster in enumerate(alias_clusters)}
    base = np.load(CLUSTER_ROOT / "node_base_vectors.npy", mmap_mode="r")
    mention_records, residuals, triples = build_mentions(
        tag_to_index, alias_by_tag, alias_index, base
    )
    statistics, alias_base, alias_residual, alias_triple = build_alias_statistics(
        alias_clusters,
        alias_by_tag,
        tag_to_index,
        mention_records,
        residuals,
        triples,
        base,
    )
    node_context = np.load(CLUSTER_ROOT / "node_context_vectors.npy", mmap_mode="r")
    seed_assignment = alias_by_tag["King"]
    seed_alias_index = alias_index[seed_assignment["aliasClusterId"]]
    seed_tag_index = tag_to_index[seed_assignment["canonicalTag"]]
    selected = []
    for row in statistics:
        if row["mentionCount"] < 20:
            continue
        candidate_tag_index = tag_to_index[row["canonicalTag"]]
        seed_similarity = float(node_context[seed_tag_index] @ node_context[candidate_tag_index])
        if row["aliasIndex"] == seed_alias_index or seed_similarity >= 0.975:
            selected.append({**row, "genericSeedSimilarity": round(seed_similarity, 6)})
    selected.sort(key=lambda row: (-row["genericSeedSimilarity"], -row["mentionCount"]))
    selected_aliases = {row["aliasIndex"] for row in selected}
    mention_indices_by_alias: dict[int, list[int]] = defaultdict(list)
    page_mentions: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in mention_records:
        mention_index = int(row["mentionIndex"])
        mention_indices_by_alias[int(row["aliasIndex"])].append(mention_index)
        page_mentions[(row["volumeId"], int(row["ownerPage"]))].append(mention_index)

    # Build a small high-precision ruler anchor set from pass-1 alias clusters.
    # The actual linking below is entirely based on Gemini embedding geometry
    # and local occurrence evidence; names are used only to declare which
    # statistically resolved clusters represent the same historical identity.
    ruler_alias_to_identity: dict[int, str] = {}
    ruler_identities: dict[str, dict] = {}
    missing_identity_seeds: dict[str, list[str]] = {}
    for identity, seeds in RULER_IDENTITY_SEEDS.items():
        identity_aliases: set[int] = set()
        missing: list[str] = []
        for seed in seeds:
            assignment = alias_by_tag.get(seed)
            if assignment is None:
                missing.append(seed)
                continue
            identity_aliases.add(alias_index[assignment["aliasClusterId"]])
        if missing:
            missing_identity_seeds[identity] = missing
        identity_mentions = sorted(
            {
                mention_index
                for alias in identity_aliases
                for mention_index in mention_indices_by_alias[alias]
            }
        )
        if not identity_mentions:
            continue
        identity_base = np.sum(
            [
                alias_base[alias] * alias_clusters[alias]["totalFrequency"]
                for alias in identity_aliases
            ],
            axis=0,
        )
        identity_base /= max(float(np.linalg.norm(identity_base)), 1e-9)
        for alias in identity_aliases:
            if alias in ruler_alias_to_identity:
                raise RuntimeError(f"Alias cluster {alias} assigned to multiple ruler identities")
            ruler_alias_to_identity[alias] = identity
        ruler_identities[identity] = {
            "aliasIndices": sorted(identity_aliases),
            "mentionIndices": identity_mentions,
            "baseVector": identity_base.astype(np.float32),
        }

    ruler_page_mentions: dict[tuple[str, int], list[int]] = defaultdict(list)
    for alias in ruler_alias_to_identity:
        for mention_index in mention_indices_by_alias[alias]:
            mention = mention_records[mention_index]
            ruler_page_mentions[(mention["volumeId"], int(mention["ownerPage"]))].append(
                mention_index
            )

    context_cluster_labels: dict[int, tuple[str, int]] = {}
    context_cluster_summaries: list[dict] = []
    for selected_row in selected:
        current_alias = int(selected_row["aliasIndex"])
        indices = np.asarray(mention_indices_by_alias[current_alias], dtype=np.int64)
        fused = np.concatenate(
            [
                math.sqrt(0.75) * np.asarray(residuals[indices]),
                math.sqrt(0.25) * np.asarray(triples[indices]),
            ],
            axis=1,
        )
        projected = GaussianRandomProjection(
            n_components=64, random_state=RANDOM_SEED
        ).fit_transform(fused)
        projected_norms = np.linalg.norm(projected, axis=1, keepdims=True)
        np.divide(
            projected,
            projected_norms,
            out=projected,
            where=projected_norms > 0,
        )
        min_cluster_size = max(5, min(40, len(indices) // 80))
        labels = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=3,
            metric="euclidean",
            cluster_selection_method="leaf",
            n_jobs=-1,
        ).fit_predict(projected)
        label_counts = Counter(int(label) for label in labels if label >= 0)
        for mention_index, label in zip(indices, labels):
            context_cluster_labels[int(mention_index)] = (
                selected_row["aliasClusterId"],
                int(label),
            )
        context_cluster_summaries.append(
            {
                "aliasClusterId": selected_row["aliasClusterId"],
                "canonicalTag": selected_row["canonicalTag"],
                "mentions": len(indices),
                "contextClusters": len(label_counts),
                "noiseMentions": int(np.sum(labels < 0)),
                "clusterSizes": dict(sorted(label_counts.items())),
            }
        )

    resolution_rows: list[dict] = []
    accepted = 0
    unresolved = 0
    for selected_row in selected:
        current_alias = int(selected_row["aliasIndex"])
        for mention_index in mention_indices_by_alias[current_alias]:
            mention = mention_records[mention_index]
            page = int(mention["ownerPage"])
            candidate_mentions: list[int] = []
            for candidate_page in range(page - 8, page + 9):
                candidate_mentions.extend(
                    ruler_page_mentions.get((mention["volumeId"], candidate_page), [])
                )
            grouped_candidates: dict[str, list[int]] = defaultdict(list)
            for candidate_mention in candidate_mentions:
                candidate_record = mention_records[candidate_mention]
                # A different named endpoint in the same sentence is often the
                # person acted on or compared with, not the referent of "King".
                # Require independent local textual evidence.
                if candidate_record["sid"] == mention["sid"]:
                    continue
                candidate_alias = int(candidate_record["aliasIndex"])
                identity = ruler_alias_to_identity[candidate_alias]
                grouped_candidates[identity].append(candidate_mention)

            scored: list[tuple[float, str, float, float, float, int]] = []
            source_residual = np.asarray(residuals[mention_index], dtype=np.float32)
            source_triple = np.asarray(triples[mention_index], dtype=np.float32)
            for identity, candidate_indices in grouped_candidates.items():
                candidate_array = np.asarray(candidate_indices, dtype=np.int64)
                contextual = 0.75 * (
                    np.asarray(residuals[candidate_array], dtype=np.float32) @ source_residual
                ) + 0.25 * (np.asarray(triples[candidate_array], dtype=np.float32) @ source_triple)
                best_position = int(np.argmax(contextual))
                best_mention_index = int(candidate_array[best_position])
                best_context = float(contextual[best_position])
                distances = np.abs(
                    np.asarray(
                        [int(mention_records[index]["ownerPage"]) for index in candidate_array],
                        dtype=np.float32,
                    )
                    - page
                )
                proximity_mass = float(np.sum(np.exp(-distances / 1.5)))
                proximity_score = 1.0 - math.exp(-proximity_mass)
                tag_similarity = float(
                    alias_base[current_alias] @ ruler_identities[identity]["baseVector"]
                )
                score = 0.65 * best_context + 0.25 * proximity_score + 0.10 * tag_similarity
                scored.append(
                    (
                        score,
                        identity,
                        best_context,
                        tag_similarity,
                        proximity_score,
                        best_mention_index,
                    )
                )
            scored.sort(reverse=True)
            best = scored[0] if scored else None
            second_score = scored[1][0] if len(scored) > 1 else 0.0
            margin = best[0] - second_score if best else 0.0
            status = "unresolved"
            if (
                best is not None
                and best[0] >= 0.73
                and best[2] >= 0.60
                and best[4] >= 0.50
                and margin >= 0.05
            ):
                status = "resolved"
                accepted += 1
            else:
                unresolved += 1
            context_cluster = context_cluster_labels.get(
                mention_index, (selected_row["aliasClusterId"], -1)
            )
            resolution_rows.append(
                {
                    "mentionId": mention["mentionId"],
                    "occurrenceId": mention["occurrenceId"],
                    "endpointOrdinal": mention["endpointOrdinal"],
                    "sid": mention["sid"],
                    "volumeId": mention["volumeId"],
                    "ownerPage": page,
                    "originalTag": mention["tag"],
                    "genericAliasClusterId": selected_row["aliasClusterId"],
                    "genericCanonicalTag": selected_row["canonicalTag"],
                    "contextCluster": int(context_cluster[1]),
                    "status": status,
                    "resolvedAliasClusterId": (None),
                    "resolvedCanonicalTag": (best[1] if status == "resolved" else None),
                    "bestCandidateIdentity": best[1] if best else None,
                    "score": round(float(best[0]), 6) if best else None,
                    "margin": round(float(margin), 6) if best else None,
                    "contextSimilarity": (round(float(best[2]), 6) if best else None),
                    "tagEmbeddingSimilarity": (round(float(best[3]), 6) if best else None),
                    "localAnchorSupport": (round(float(best[4]), 6) if best else None),
                    "supportMentionId": (mention_records[best[5]]["mentionId"] if best else None),
                    "supportTag": (mention_records[best[5]]["tag"] if best else None),
                    "supportPage": (int(mention_records[best[5]]["ownerPage"]) if best else None),
                }
            )

    with (COREF_ROOT / "coreference_resolutions.jsonl").open("w", encoding="utf-8") as handle:
        for row in resolution_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_json(COREF_ROOT / "context_clusters.json", context_cluster_summaries)
    selected_output = []
    by_generic: dict[str, list[dict]] = defaultdict(list)
    for row in resolution_rows:
        by_generic[row["genericAliasClusterId"]].append(row)
    for row in selected:
        resolutions = by_generic[row["aliasClusterId"]]
        resolved_counts = Counter(
            item["resolvedCanonicalTag"] for item in resolutions if item["status"] == "resolved"
        )
        selected_output.append(
            {
                **row,
                "resolvedMentions": sum(item["status"] == "resolved" for item in resolutions),
                "unresolvedMentions": sum(item["status"] != "resolved" for item in resolutions),
                "topResolvedEntities": [
                    {"canonicalTag": tag, "mentions": count}
                    for tag, count in resolved_counts.most_common(15)
                ],
            }
        )
    write_json(COREF_ROOT / "selected_ambiguous_aliases.json", selected_output)
    write_json(
        COREF_ROOT / "ruler_identity_anchors.json",
        {
            identity: {
                "seedTags": list(RULER_IDENTITY_SEEDS[identity]),
                "aliasClusterIds": [
                    alias_clusters[index]["aliasClusterId"] for index in row["aliasIndices"]
                ],
                "anchorMentions": len(row["mentionIndices"]),
            }
            for identity, row in ruler_identities.items()
        },
    )
    report = {
        "method": (
            "Gemini mention-context residual clustering plus conservative local "
            "linking to curated historical-identity anchors built from resolved "
            "pass-1 alias clusters"
        ),
        "featuresUsed": {
            "argumentSentenceResidualEmbedding": 0.75,
            "tripleSentenceEmbedding": 0.25,
            "candidateScoreContext": 0.65,
            "candidateScoreLocalAnchorSupport": 0.25,
            "candidateScoreTagEmbedding": 0.10,
            "endpointRoleUsed": False,
            "fastTextUsed": False,
        },
        "rulerIdentities": len(ruler_identities),
        "rulerIdentityAnchorMentions": sum(
            len(row["mentionIndices"]) for row in ruler_identities.values()
        ),
        "candidateWindowPages": 8,
        "sameSentenceAnchorsExcluded": True,
        "acceptanceThresholds": {
            "score": 0.73,
            "contextSimilarity": 0.60,
            "localAnchorSupport": 0.50,
            "candidateMargin": 0.05,
        },
        "missingIdentitySeeds": missing_identity_seeds,
        "ambiguousAliasClustersSelected": len(selected),
        "mentionsAttempted": len(resolution_rows),
        "mentionsResolved": accepted,
        "mentionsUnresolved": unresolved,
        "resolutionRate": accepted / len(resolution_rows) if resolution_rows else 0.0,
        "sourceFilesModified": False,
    }
    write_json(COREF_ROOT / "coreference_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("alias", "coref", "all"), default="all")
    args = parser.parse_args()
    if args.stage in {"alias", "all"}:
        run_alias_resolution()
    if args.stage in {"coref", "all"}:
        run_coreference()


if __name__ == "__main__":
    main()

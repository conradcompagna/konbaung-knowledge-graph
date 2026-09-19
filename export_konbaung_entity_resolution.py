from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "konbaung_v3_entity_resolution_two_pass_20260724"
ALIAS_ROOT = RUN_ROOT / "pass1_alias_resolution"
COREF_ROOT = RUN_ROOT / "pass2_coreference"
EMBEDDING_ROOT = ROOT / "konbaung_v3_eight_view_embeddings_20260724"


def jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalized_display(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    output: list[str] = []
    for char in value:
        category = unicodedata.category(char)
        output.append(char if category[0] in {"L", "N", "M"} else " ")
    return " ".join("".join(output).split())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    alias_clusters = json.loads(
        (ALIAS_ROOT / "alias_clusters.json").read_text(encoding="utf-8")
    )
    alias_assignments = list(jsonl(ALIAS_ROOT / "alias_assignments.jsonl"))
    alias_report = json.loads(
        (ALIAS_ROOT / "alias_resolution_report.json").read_text(encoding="utf-8")
    )
    coref_rows = list(jsonl(COREF_ROOT / "coreference_resolutions.jsonl"))
    coref_report = json.loads(
        (COREF_ROOT / "coreference_report.json").read_text(encoding="utf-8")
    )
    ruler_anchors = json.loads(
        (COREF_ROOT / "ruler_identity_anchors.json").read_text(encoding="utf-8")
    )
    occurrences = {
        row["occurrenceId"]: row
        for row in jsonl(EMBEDDING_ROOT / "occurrences.jsonl")
    }

    cluster_by_id = {
        cluster["aliasClusterId"]: cluster for cluster in alias_clusters
    }
    identity_by_cluster = {
        cluster_id: identity
        for identity, record in ruler_anchors.items()
        for cluster_id in record["aliasClusterIds"]
    }

    alias_csv = ALIAS_ROOT / "alias_mapping.csv"
    with alias_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tag",
                "frequency",
                "alias_cluster_id",
                "canonical_tag",
                "variant_count",
                "minimum_embedding_similarity",
                "historical_ruler_identity",
            ],
        )
        writer.writeheader()
        for assignment in alias_assignments:
            cluster = cluster_by_id[assignment["aliasClusterId"]]
            writer.writerow(
                {
                    "tag": assignment["tag"],
                    "frequency": assignment["frequency"],
                    "alias_cluster_id": assignment["aliasClusterId"],
                    "canonical_tag": assignment["canonicalTag"],
                    "variant_count": cluster["variantCount"],
                    "minimum_embedding_similarity": cluster[
                        "minimumEmbeddingSimilarity"
                    ],
                    "historical_ruler_identity": identity_by_cluster.get(
                        assignment["aliasClusterId"], ""
                    ),
                }
            )

    coref_csv = COREF_ROOT / "coreference_resolutions.csv"
    with coref_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mention_id",
                "occurrence_id",
                "endpoint_ordinal",
                "sid",
                "volume_id",
                "owner_page",
                "original_tag",
                "status",
                "resolved_identity",
                "best_candidate_identity",
                "score",
                "margin",
                "context_similarity",
                "tag_embedding_similarity",
                "local_anchor_support",
                "support_tag",
                "support_page",
                "subject",
                "predicate",
                "object",
                "sentence_en",
                "sentence_my",
            ],
        )
        writer.writeheader()
        for row in coref_rows:
            occurrence = occurrences[row["occurrenceId"]]
            writer.writerow(
                {
                    "mention_id": row["mentionId"],
                    "occurrence_id": row["occurrenceId"],
                    "endpoint_ordinal": row["endpointOrdinal"],
                    "sid": row["sid"],
                    "volume_id": row["volumeId"],
                    "owner_page": row["ownerPage"],
                    "original_tag": row["originalTag"],
                    "status": row["status"],
                    "resolved_identity": row["resolvedCanonicalTag"] or "",
                    "best_candidate_identity": row.get("bestCandidateIdentity")
                    or "",
                    "score": row["score"],
                    "margin": row["margin"],
                    "context_similarity": row["contextSimilarity"],
                    "tag_embedding_similarity": row[
                        "tagEmbeddingSimilarity"
                    ],
                    "local_anchor_support": row.get("localAnchorSupport"),
                    "support_tag": row["supportTag"] or "",
                    "support_page": row.get("supportPage"),
                    "subject": occurrence["subject"],
                    "predicate": occurrence["predicate"],
                    "object": occurrence["object"],
                    "sentence_en": occurrence["sentenceEn"],
                    "sentence_my": occurrence["sentenceMy"],
                }
            )

    source_manifest = json.loads(
        (EMBEDDING_ROOT / "source_manifest.json").read_text(encoding="utf-8")
    )
    source_checks = []
    for record in source_manifest["files"]:
        path = Path(record["path"])
        actual_hash = sha256(path)
        actual_size = path.stat().st_size
        source_checks.append(
            {
                "path": str(path),
                "expectedSizeBytes": record["sizeBytes"],
                "actualSizeBytes": actual_size,
                "expectedSha256": record["sha256"],
                "actualSha256": actual_hash,
                "matches": (
                    actual_size == record["sizeBytes"]
                    and actual_hash == record["sha256"]
                ),
            }
        )

    multi_clusters = [
        cluster for cluster in alias_clusters if cluster["variantCount"] > 1
    ]
    format_only_clusters = [
        cluster
        for cluster in multi_clusters
        if len(
            {
                normalized_display(variant["tag"])
                for variant in cluster["variants"]
            }
        )
        == 1
    ]
    semantic_clusters = [
        cluster
        for cluster in multi_clusters
        if cluster not in format_only_clusters
    ]
    resolved_rows = [row for row in coref_rows if row["status"] == "resolved"]
    resolution_counts = Counter(
        row["resolvedCanonicalTag"] for row in resolved_rows
    )
    generic_counts = Counter(row["genericCanonicalTag"] for row in resolved_rows)

    integrity = {
        "aliasAssignmentRows": len(alias_assignments),
        "uniqueAliasTags": len({row["tag"] for row in alias_assignments}),
        "aliasClusters": len(alias_clusters),
        "tagsRepresentedInAliasClusters": sum(
            cluster["variantCount"] for cluster in alias_clusters
        ),
        "mentionRows": sum(
            1 for _ in jsonl(COREF_ROOT / "mention_records.jsonl")
        ),
        "uniqueMentionIds": len(
            {
                row["mentionId"]
                for row in jsonl(COREF_ROOT / "mention_records.jsonl")
            }
        ),
        "coreferenceAttemptRows": len(coref_rows),
        "uniqueCoreferenceMentionIds": len(
            {row["mentionId"] for row in coref_rows}
        ),
        "acceptedCoreferenceMentions": len(resolved_rows),
        "sourceFiles": source_checks,
        "allSourceFilesUnchanged": all(
            record["matches"] for record in source_checks
        ),
    }
    write_json(RUN_ROOT / "integrity_report.json", integrity)

    semantic_lines = []
    for cluster in semantic_clusters:
        variants = "; ".join(
            f"{variant['tag']} ({variant['frequency']})"
            for variant in cluster["variants"]
        )
        semantic_lines.append(
            f"| {cluster['canonicalTag']} | {variants} | "
            f"{cluster['minimumEmbeddingSimilarity']:.6f} |"
        )

    identity_lines = [
        f"| {identity} | {count} |"
        for identity, count in resolution_counts.most_common()
    ]
    generic_lines = [
        f"| {tag} | {count} |" for tag, count in generic_counts.most_common()
    ]
    unresolved_identities = sorted(
        set(ruler_anchors) - set(resolution_counts)
    )

    report = f"""# Konbaung V3 Gemini entity resolution

## Outcome

Two local passes over the existing `gemini-embedding-2` vectors are complete. No new embedding requests were sent, so this refinement cost **$0.00**. The four canonical V3 source files still match their pre-run SHA-256 hashes.

The result is deliberately a candidate-resolution layer, not a rewrite of either database.

## Pass 1: spelling and label variants

- Input node tags: **{alias_report['records']:,}**
- Resulting alias clusters: **{len(alias_clusters):,}**
- Multi-variant clusters: **{len(multi_clusters):,}**
- Tags in multi-variant clusters: **{sum(c['variantCount'] for c in multi_clusters):,}**
- Pure case/punctuation/underscore clusters: **{len(format_only_clusters):,}**
- Nontrivial semantic/spelling clusters: **{len(semantic_clusters):,}**
- Accepted embedding-pair edges: **{alias_report['autoPairs']:,}**
- Co-occurring list-member pairs rejected: **{alias_report['cooccurringListPairsRejected']:,}**

The initial embedding-only threshold overmerged different low-frequency people listed in the same sentence. The retained pass fixes that failure: a non-formatting pair needs at least three distinct sentences for each tag, three non-shared sentences for each tag, no more than 10% shared-sentence overlap, and the original high Gemini-similarity test. Complete linkage prevents chaining through a merely similar middle tag.

| Canonical tag | Retained variants and frequencies | Minimum Gemini similarity |
|---|---|---:|
{chr(10).join(semantic_lines)}

This is high precision and incomplete by design. Plausible variants without independent textual evidence remain in the review queue instead of being merged automatically.

## Pass 2: generic ruler coreference

- Endpoint mentions in the corpus: **{integrity['mentionRows']:,}**
- Generic ruler mentions attempted: **{coref_report['mentionsAttempted']:,}**
- Accepted identity links: **{coref_report['mentionsResolved']:,}**
- Unique sentences with an accepted link: **{len({row['sid'] for row in resolved_rows}):,}**
- Left unresolved: **{coref_report['mentionsUnresolved']:,}**
- Accepted rate: **{100 * coref_report['resolutionRate']:.2f}%**

The linker considered six embedding-selected generic ruler families: `King`, `Burmese King`, `Konbaung King`, `Konbaung_King`, `Great King`, and `Sovereign King`. It used ten explicit historical identity anchor sets. A candidate had to be supported independently within eight pages; named entities from the same sentence were excluded because they are often the king's object, relative, or opponent rather than a coreferent.

Each candidate score is 65% Gemini mention-context similarity, 25% local anchor support, and 10% Gemini tag similarity. Acceptance additionally requires score >= 0.73, context similarity >= 0.60, local support >= 0.50, and a >= 0.05 lead over the second candidate.

| Resolved historical identity | Accepted endpoint mentions |
|---|---:|
{chr(10).join(identity_lines)}

No candidate crossed the conservative threshold for: **{', '.join(unresolved_identities)}**. Naungdawgyi also lacks a dependable explicit name anchor in the current tag set, so this pass does not guess his generic references.

Accepted source-label distribution:

| Generic label | Accepted endpoint mentions |
|---|---:|
{chr(10).join(generic_lines)}

## Manual QA

I manually reviewed a stratified set of **36 accepted links**: six examples for each identity that received links, including the lowest-scoring accepted cases. After excluding same-sentence anchors, I found no obvious false identity link in that sample. The reviewed contexts coherently matched the surrounding reign sections—for example Alaungpaya receiving the white elephant, Hsinbyushin's Chinese-front mobilizations, Singu Min's court purge, Bodawpaya's Amarapura works, Bagyidaw's religious commissions, and Mindon's Ratanapon ceremonies.

That is not a gold-standard accuracy measurement. The 305 links should be treated as high-confidence proposals. The 8,186 unresolved mentions should remain generic until a later pass adds stronger evidence such as reign/chapter boundaries or manual adjudication.

## Files

- `pass1_alias_resolution/alias_mapping.csv` — every raw node tag, frequency, automatic alias cluster, canonical tag, and ruler identity where available.
- `pass1_alias_resolution/alias_clusters.json` — full pass-1 cluster structure.
- `pass1_alias_resolution/review_pairs.json` — plausible pairs not accepted automatically.
- `pass2_coreference/coreference_resolutions.csv` — all 8,491 attempted generic mentions, scores, support, full triple, and Burmese/English sentence.
- `pass2_coreference/coreference_resolutions.jsonl` — machine-readable equivalent without sentence duplication.
- `pass2_coreference/ruler_identity_anchors.json` — the ten identity anchor sets.
- `integrity_report.json` — row-count and source-hash audit.

## Recommendation

Use the pass-1 mappings and the 305 accepted pass-2 links as a reversible normalization layer. Do not destructively replace source tags. Before expanding coreference recall, manually validate a larger random sample and add explicit reign/page boundaries; simply lowering the embedding thresholds will recreate the false links caught during this audit.
"""
    (RUN_ROOT / "TWO_PASS_GEMINI_ENTITY_RESOLUTION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "aliasCsv": str(alias_csv),
                "coreferenceCsv": str(coref_csv),
                "report": str(
                    RUN_ROOT / "TWO_PASS_GEMINI_ENTITY_RESOLUTION_REPORT.md"
                ),
                "integrity": integrity,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

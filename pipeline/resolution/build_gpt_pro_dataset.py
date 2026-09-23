#!/usr/bin/env python3
"""Build compact analysis files from the completed Konbaung entity ledger."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"C:\Users\conra\Desktop\dighumproject")
SOURCE = ROOT / "konbaung_singleton_completion_gemini_20260902"
OUTPUT = ROOT / "konbaung_entity_resolution_dataset_gpt_pro_20260902"


def load_json(path: Path):
    """Load a UTF-8 JSON artifact from the completed resolution archive."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_rows(path: Path) -> list[dict[str, str]]:
    """Load the authoritative one-row-per-entity assignment ledger."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write a deterministic UTF-8 CSV with stable field ordering."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    """Hash a generated file so an uploaded copy can be checked for integrity."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Create cluster-level, member-level, and compact JSONL representations."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ledger = load_rows(SOURCE / "resolved_entities_all.csv")
    clusters = load_json(SOURCE / "resolved_clusters_all.json")
    final_manifest = load_json(SOURCE / "FINAL_ALL_ENTITY_RESOLUTION_MANIFEST.json")
    row_by_id = {row["id"]: row for row in ledger}

    if len(ledger) != 23_890 or len(row_by_id) != 23_890:
        raise RuntimeError("The source ledger does not contain 23,890 unique entity IDs.")
    if len(clusters) != 17_658:
        raise RuntimeError("The source archive does not contain 17,658 clusters.")

    cluster_rows: list[dict] = []
    member_rows: list[dict] = []
    compact_records: list[dict] = []
    seen_members: set[str] = set()
    phase_cluster_counts: Counter[str] = Counter()
    total_corpus_mentions = 0

    for cluster in sorted(clusters, key=lambda row: (int(row["wave"]), int(row["waveOrder"]))):
        parent_id = cluster["parentId"]
        member_ids = cluster["memberIds"]
        parent_row = row_by_id[parent_id]
        phase = parent_row["phase"]
        member_tuples: list[list[object]] = []
        cluster_mentions = 0

        if parent_id not in member_ids:
            raise RuntimeError(f"Cluster {parent_id} does not contain its own parent ID.")

        for member_id in member_ids:
            if member_id in seen_members:
                raise RuntimeError(f"Entity ID {member_id} appears in more than one cluster.")
            seen_members.add(member_id)
            member = row_by_id[member_id]
            mentions = int(member["mentions"])
            cluster_mentions += mentions
            total_corpus_mentions += mentions
            member_tuples.append([member_id, member["entity"], mentions])
            member_rows.append(
                {
                    "cluster_id": parent_id,
                    "canonical_entity": parent_row["entity"],
                    "entity_id": member_id,
                    "entity": member["entity"],
                    "mentions": mentions,
                    "is_canonical": 1 if member_id == parent_id else 0,
                    "phase": phase,
                    "wave": int(cluster["wave"]),
                    "wave_order": int(cluster["waveOrder"]),
                }
            )

        member_count = len(member_ids)
        cluster_rows.append(
            {
                "cluster_id": parent_id,
                "canonical_entity": parent_row["entity"],
                "canonical_mentions": int(parent_row["mentions"]),
                "member_count": member_count,
                "alias_count": member_count - 1,
                "total_mentions": cluster_mentions,
                "phase": phase,
                "wave": int(cluster["wave"]),
                "wave_order": int(cluster["waveOrder"]),
            }
        )
        compact_records.append(
            {
                "cluster_id": parent_id,
                "canonical_entity": parent_row["entity"],
                "phase": phase,
                "wave": int(cluster["wave"]),
                "member_count": member_count,
                "total_mentions": cluster_mentions,
                "members": member_tuples,
            }
        )
        phase_cluster_counts[phase] += 1

    if seen_members != set(row_by_id):
        raise RuntimeError("The generated clusters do not cover the complete source ledger.")

    cluster_path = OUTPUT / "entity_cluster_summary.csv"
    member_path = OUTPUT / "entity_cluster_members.csv"
    jsonl_path = OUTPUT / "entity_clusters_compact.jsonl"
    write_csv(
        cluster_path,
        [
            "cluster_id",
            "canonical_entity",
            "canonical_mentions",
            "member_count",
            "alias_count",
            "total_mentions",
            "phase",
            "wave",
            "wave_order",
        ],
        cluster_rows,
    )
    write_csv(
        member_path,
        [
            "cluster_id",
            "canonical_entity",
            "entity_id",
            "entity",
            "mentions",
            "is_canonical",
            "phase",
            "wave",
            "wave_order",
        ],
        member_rows,
    )
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in compact_records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    positive_clusters = sum(1 for row in cluster_rows if row["member_count"] > 1)
    aliases = sum(row["alias_count"] for row in cluster_rows)
    phase_entity_counts = Counter(row["phase"] for row in ledger)
    manifest = {
        "dataset": "Konbaung entity-resolution clusters for GPT Pro",
        "version": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sourceArchive": str(SOURCE),
        "model": final_manifest["model"],
        "thinkingLevel": final_manifest["thinkingLevel"],
        "clusterCount": len(cluster_rows),
        "entityTagCount": len(member_rows),
        "positiveMultiMemberClusters": positive_clusters,
        "singleMemberClusters": len(cluster_rows) - positive_clusters,
        "acceptedAliases": aliases,
        "totalCorpusMentions": total_corpus_mentions,
        "phaseClusterCounts": dict(phase_cluster_counts),
        "phaseEntityTagCounts": dict(phase_entity_counts),
        "waveRange": [1, 155],
        "files": {
            cluster_path.name: {
                "rows": len(cluster_rows),
                "sha256": sha256(cluster_path),
            },
            member_path.name: {
                "rows": len(member_rows),
                "sha256": sha256(member_path),
            },
            jsonl_path.name: {
                "records": len(compact_records),
                "memberTuple": ["entity_id", "entity", "mentions"],
                "sha256": sha256(jsonl_path),
            },
        },
    }
    manifest_path = OUTPUT / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

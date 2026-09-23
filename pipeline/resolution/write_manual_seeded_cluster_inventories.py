from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent / "konbaung_v3_manual_seeded_clustering_20260724"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def markdown_text(value: Any) -> str:
    if value is None:
        return ""
    return (
        str(value).replace("\r", " ").replace("\n", " ").replace("\\", "\\\\").replace("|", "\\|")
    )


def inline_code(value: Any) -> str:
    return f"`{markdown_text(value).replace('`', '``')}`"


def write_inventory(kind: str, title: str, output_name: str) -> dict[str, int]:
    clusters = json.loads((ROOT / f"{kind}_clusters.json").read_text(encoding="utf-8"))
    assignments = load_jsonl(ROOT / f"{kind}_assignments.jsonl")
    expected_indices = list(range(len(assignments)))
    actual_indices = [int(row["index"]) for row in assignments]
    if actual_indices != expected_indices:
        raise RuntimeError(f"{kind} assignments are incomplete or unordered")

    cluster_by_id = {str(row["clusterId"]): row for row in clusters}
    if len(cluster_by_id) != len(clusters):
        raise RuntimeError(f"{kind} cluster IDs are not unique")

    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignments:
        members[str(row["clusterId"])].append(row)
    if set(members) != set(cluster_by_id):
        raise RuntimeError(
            f"{kind} cluster/assignment mismatch: "
            f"missing={sorted(set(cluster_by_id) - set(members))[:10]}, "
            f"extra={sorted(set(members) - set(cluster_by_id))[:10]}"
        )

    manual_clusters = [row for row in clusters if row["clusterType"] == "manual_seed"]
    residual_clusters = [row for row in clusters if row["clusterType"] == "residual_louvain"]
    manual_clusters.sort(
        key=lambda row: (
            -int(row["mentionCount"]),
            str(row["clusterId"]),
        )
    )
    residual_clusters.sort(
        key=lambda row: (
            -int(row["mentionCount"]),
            str(row["clusterId"]),
        )
    )

    source_mentions = sum(int(row["frequency"]) for row in assignments)
    lines = [
        f"# Konbaung V3 {title} cluster inventory",
        "",
        "This report is exhaustive. It lists every output cluster and every tag "
        "assigned to it; tag tables are not sampled or truncated.",
        "",
        "## Totals",
        "",
        f"- Clusters: **{len(clusters):,}**",
        f"- Unique tags: **{len(assignments):,}**",
        f"- Total tag mentions: **{source_mentions:,}**",
        f"- Manual-seeded clusters: **{len(manual_clusters):,}**",
        f"- Provisional residual clusters: **{len(residual_clusters):,}**",
        f"- Reviewed workbook tags: **"
        f"{sum(bool(row['manualSeedMember']) for row in assignments):,}**",
        f"- Automatically attached tags: **"
        f"{sum(row['assignmentMethod'] == 'manual_centroid' for row in assignments):,}**",
        f"- Residual Louvain tags: **"
        f"{sum(row['assignmentMethod'] == 'residual_louvain' for row in assignments):,}**",
        "",
        "Manual-seeded clusters are listed first, followed by provisional "
        "residual communities. Within each section, clusters are ordered by "
        "total mention count. Tags within a cluster are ordered by descending "
        "frequency and then alphabetically.",
        "",
    ]

    def append_cluster(cluster: dict[str, Any]) -> None:
        cluster_id = str(cluster["clusterId"])
        cluster_members = sorted(
            members[cluster_id],
            key=lambda row: (
                -int(row["frequency"]),
                str(row["tag"]).casefold(),
                str(row["tag"]),
                int(row["index"]),
            ),
        )
        mention_count = sum(int(row["frequency"]) for row in cluster_members)
        if mention_count != int(cluster["mentionCount"]):
            raise RuntimeError(f"{kind} cluster {cluster_id} mention total mismatch")
        if len(cluster_members) != int(cluster["uniqueTags"]):
            raise RuntimeError(f"{kind} cluster {cluster_id} tag total mismatch")

        lines.extend(
            [
                f"### {markdown_text(cluster_id)} — {markdown_text(cluster['canonicalLabel'])}",
                "",
                f"- Meta-tag: {inline_code(cluster['canonicalLabel'])}",
                f"- Cluster ID: {inline_code(cluster_id)}",
                f"- Cluster type: {inline_code(cluster['clusterType'])}",
                f"- Total count: **{mention_count:,}**",
                f"- Unique tags: **{len(cluster_members):,}**",
            ]
        )
        if cluster["clusterType"] == "manual_seed":
            lines.extend(
                [
                    f"- Workbook action: {inline_code(cluster.get('manualAction'))}",
                    f"- Reviewed tags: **{int(cluster['manualMemberCount']):,}**",
                    f"- Automatically attached tags: **{int(cluster['automaticMemberCount']):,}**",
                ]
            )
        basis = cluster.get("basis")
        if basis:
            lines.append(f"- Basis: {markdown_text(basis)}")
        notes = cluster.get("notes")
        if notes:
            lines.append(f"- Notes: {markdown_text(notes)}")
        lines.extend(
            [
                "",
                "| Tag | Frequency | Assignment |",
                "|---|---:|---|",
            ]
        )
        assignment_names = {
            "fixed_manual_assignment": "reviewed workbook",
            "manual_centroid": "automatic manual-seed attachment",
            "residual_louvain": "provisional residual",
        }
        for row in cluster_members:
            lines.append(
                f"| {markdown_text(row['tag'])} | "
                f"{int(row['frequency'])} | "
                f"{assignment_names[row['assignmentMethod']]} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Manual-seeded clusters",
            "",
        ]
    )
    for cluster in manual_clusters:
        append_cluster(cluster)

    lines.extend(
        [
            "## Provisional residual clusters",
            "",
            "These `*_NEW_*` communities are automatic review neighborhoods, not "
            "canonical equivalence classes.",
            "",
        ]
    )
    for cluster in residual_clusters:
        append_cluster(cluster)

    output_path = ROOT / output_name
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "clusters": len(clusters),
        "tags": len(assignments),
        "mentions": source_mentions,
        "manualClusters": len(manual_clusters),
        "residualClusters": len(residual_clusters),
        "tableRows": sum(len(group) for group in members.values()),
        "bytes": output_path.stat().st_size,
    }


def main() -> None:
    result = {
        "nodes": write_inventory(
            "node",
            "node",
            "NODE_CLUSTERING_REPORT.md",
        ),
        "relations": write_inventory(
            "edge",
            "relation",
            "RELATION_CLUSTERING_REPORT.md",
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

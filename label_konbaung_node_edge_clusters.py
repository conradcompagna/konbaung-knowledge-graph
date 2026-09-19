from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from konbaung_gemini_summary_claim_completion_annotator import api_key


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    PROJECT_ROOT / "konbaung_v3_node_edge_clustering_first_pass_20260724"
)
MODEL = "gemini-3.1-flash-lite"
BATCH_SIZE = 10

MANUAL_OVERRIDES = {
    "NODE_C001": {
        "candidateMetaTag": "ROYAL_SOVEREIGNTY_AND_MONARCHY",
        "alternativeMetaTag": "ROYAL_INSTITUTION",
    },
    "NODE_C003": {
        "candidateMetaTag": "NAMED_COURT_AND_MILITARY_OFFICIALS",
        "alternativeMetaTag": "COURT_OFFICIAL_TITLES",
    },
    "NODE_C009": {
        "candidateMetaTag": "LATE_KONBAUNG_ROYALTY",
        "alternativeMetaTag": "KONBAUNG_ROYAL_PERSONAGES",
    },
    "NODE_C017": {
        "candidateMetaTag": "LOWER_BURMA_PLACES_AND_POLITICAL_ACTORS",
        "alternativeMetaTag": "HANTHAWADDY_YANGON_POLITICAL_SPHERE",
        "description": (
            "This historical-neighborhood cluster centers on Hanthawaddy/Pegu, "
            "Yangon, Thanlyin, Prome, and the Burmese and British officials "
            "associated with those places."
        ),
        "coherenceAssessment": "medium",
        "splitNote": (
            "Separate places and jurisdictions from named Burmese and British "
            "officials in the ontology-building pass."
        ),
    },
    "NODE_C023": {
        "candidateMetaTag": "NAMED_MILITARY_AND_POLITICAL_FIGURES",
        "alternativeMetaTag": "MON_AND_BURMESE_COMMANDERS",
    },
    "NODE_C026": {
        "candidateMetaTag": "AVA_SAGAING_PLACES_AND_ASSOCIATED_ACTORS",
        "alternativeMetaTag": "AVA_SAGAING_GEOGRAPHY",
    },
    "NODE_C027": {
        "candidateMetaTag": "MON_MILITARY_AND_POLITICAL_ACTORS",
        "alternativeMetaTag": "MON_FORCES_AND_LEADERS",
    },
    "NODE_C038": {
        "candidateMetaTag": "BURMESE_PERSONAL_NAMES_AND_HONORIFICS",
        "alternativeMetaTag": "PERSONAL_NAMES_AND_TITLES",
    },
    "NODE_C041": {
        "candidateMetaTag": "OMENS_DISASTERS_AND_CALENDRICAL_GROUPS",
        "alternativeMetaTag": "COSMIC_AND_SOCIAL_INDICATORS",
    },
    "NODE_C052": {
        "candidateMetaTag": "MAHABODHI_SITES_OBJECTS_AND_RITUALS",
        "alternativeMetaTag": "MAHABODHI_CULT_PRACTICES",
    },
    "EDGE_C001": {
        "candidateMetaTag": "ROYAL_TITLE_AND_REGALIA_TRANSACTIONS",
        "alternativeMetaTag": "COURT_RANK_CONFERRAL",
    },
    "EDGE_C007": {
        "candidateMetaTag": "MILITARY_COMBAT_OPERATIONS",
        "alternativeMetaTag": "MILITARY_HOSTILITIES",
    },
    "EDGE_C014": {
        "candidateMetaTag": "CONSTRUCTION_COMMISSIONING_AND_SUPERVISION",
        "alternativeMetaTag": "INFRASTRUCTURE_CONSTRUCTION",
    },
    "EDGE_C015": {
        "candidateMetaTag": "ENCAMPMENT_AND_FORTIFICATION",
        "alternativeMetaTag": "DEFENSIVE_POSITIONING",
    },
    "EDGE_C020": {
        "candidateMetaTag": "ORDERS_COMMANDS_AND_EXECUTION",
        "alternativeMetaTag": "ROYAL_ORDER_ISSUANCE",
    },
    "EDGE_C024": {
        "candidateMetaTag": "MISSIONS_COMMUNICATION_AND_INQUIRY",
        "alternativeMetaTag": "OFFICIAL_DISPATCH_AND_COMMUNICATION",
    },
    "EDGE_C034": {
        "candidateMetaTag": "SERVICE_AND_ALLEGIANCE",
        "alternativeMetaTag": "OATHS_OF_ALLEGIANCE",
    },
    "EDGE_C051": {
        "candidateMetaTag": "COMPILATION_AND_TECHNICAL_UTILIZATION",
        "alternativeMetaTag": "INTELLECTUAL_PRODUCTION_AND_SPONSORSHIP",
    },
}


class LabelProposal(BaseModel):
    clusterId: str
    candidateMetaTag: str
    alternativeMetaTag: str | None = None
    description: str
    coherenceAssessment: Literal["high", "medium", "low"]
    splitNote: str | None = None


class LabelBatch(BaseModel):
    proposals: list[LabelProposal] = Field(default_factory=list)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_assignments(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                grouped[row["clusterId"]].append(row)
    return grouped


def compact_examples(cluster: dict, assignments: list[dict]) -> dict:
    chosen: list[dict] = []
    seen: set[str] = set()

    def add(tag: str, frequency: int, similarity: float, source: str) -> None:
        key = tag.casefold().strip()
        if not key or key in seen:
            return
        seen.add(key)
        chosen.append(
            {
                "tag": tag,
                "frequency": frequency,
                "similarityToCentroid": round(similarity, 4),
                "source": source,
            }
        )

    for item in cluster["centralTags"][:12]:
        add(
            item["tag"],
            item["frequency"],
            item["similarityToCentroid"],
            "central",
        )
    for item in cluster["frequentTags"][:12]:
        add(
            item["tag"],
            item["frequency"],
            item["similarityToCentroid"],
            "frequent",
        )

    boundary = sorted(assignments, key=lambda row: row["similarityToCentroid"])
    if boundary:
        positions = sorted(
            {
                0,
                min(1, len(boundary) - 1),
                min(2, len(boundary) - 1),
                len(boundary) // 10,
                len(boundary) // 4,
                len(boundary) // 2,
                (3 * len(boundary)) // 4,
            }
        )
        for index in positions:
            item = boundary[index]
            add(
                item["tag"],
                item["frequency"],
                item["similarityToCentroid"],
                "boundary",
            )

    return {
        "clusterId": cluster["clusterId"],
        "uniqueTags": cluster["uniqueTags"],
        "mentionCount": cluster["mentionCount"],
        "meanSimilarityToCentroid": cluster["meanSimilarityToCentroid"],
        "p10SimilarityToCentroid": cluster["p10SimilarityToCentroid"],
        "examples": chosen,
    }


def prompt_for(kind: str, payload: list[dict]) -> str:
    if kind == "node":
        kind_rules = """
These are NODE clusters. Subject and object are deliberately not different
categories: both were pooled before clustering and grammatical position played
no role. Name the semantic kind of entity or concept represented by each
cluster. Never use labels such as SUBJECT, OBJECT, ACTOR, or PATIENT.
"""
    else:
        kind_rules = """
These are EDGE clusters made from predicates. Name the broad historical
relation family represented by each cluster. A cluster may contain opposite
surface phrasings (for example bestowed a title on / received a title from)
because this is only a first-pass family, not yet a canonical directed
predicate. Explicitly note such directionality or other meaningful mixtures in
splitNote rather than hiding them.
"""

    return f"""You are reviewing unsupervised semantic clusters from a
Konbaung-dynasty historical triples corpus.

{kind_rules.strip()}

For every supplied cluster:
- propose one concise UPPER_SNAKE_CASE candidateMetaTag suitable as a broad
  database category;
- give a useful alternativeMetaTag only when there is a genuinely plausible
  alternative framing;
- describe the cluster in one precise sentence;
- assess coherence as high, medium, or low;
- use splitNote for a concrete recommended split or cleanup when the boundary
  examples reveal distinct meanings. Otherwise return null.

Judge the whole example set. Central examples reveal the core, frequent
examples reveal corpus importance, and boundary examples deliberately expose
outliers. Prefer broad, historically useful categories over labels copied from
one tag. Do not force a falsely precise ontology. Return each clusterId exactly
once, in the supplied order, and return only the requested structured JSON.

CLUSTERS:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}
"""


def call_batch(
    client: genai.Client,
    kind: str,
    payload: list[dict],
    expected_ids: list[str],
) -> tuple[list[dict], dict]:
    prompt = prompt_for(kind, payload)
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=LabelBatch.model_json_schema(),
                    temperature=0.0,
                    candidate_count=1,
                    max_output_tokens=8192,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.LOW,
                        include_thoughts=False,
                    ),
                ),
            )
            parsed = LabelBatch.model_validate_json(response.text)
            rows = [item.model_dump(mode="json") for item in parsed.proposals]
            returned_ids = [row["clusterId"] for row in rows]
            if returned_ids != expected_ids:
                raise ValueError(
                    f"Expected cluster IDs {expected_ids}, got {returned_ids}"
                )
            for row in rows:
                if not re.fullmatch(r"[A-Z][A-Z0-9_]*", row["candidateMetaTag"]):
                    raise ValueError(
                        f"Invalid meta-tag {row['candidateMetaTag']!r}"
                    )
            usage = (
                response.usage_metadata.model_dump(mode="json")
                if response.usage_metadata
                else {}
            )
            return rows, usage
        except Exception as exc:
            last_error = exc
            if attempt == 5:
                break
            time.sleep(min(20, 2**attempt))
    raise RuntimeError(f"Label batch failed after retries: {last_error}")


def label_kind(
    client: genai.Client,
    output_dir: Path,
    kind: str,
    clusters: list[dict],
    assignments: dict[str, list[dict]],
) -> tuple[list[dict], list[dict]]:
    progress_path = output_dir / f"{kind}_label_progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {}
    results: dict[str, dict] = dict(progress.get("labels", {}))
    usages: list[dict] = list(progress.get("usage", []))

    substantive = [cluster for cluster in clusters if not cluster["unresolved"]]
    payloads = [
        compact_examples(cluster, assignments[cluster["clusterId"]])
        for cluster in substantive
    ]
    for start in range(0, len(payloads), BATCH_SIZE):
        batch = payloads[start : start + BATCH_SIZE]
        missing = [item for item in batch if item["clusterId"] not in results]
        if not missing:
            continue
        expected_ids = [item["clusterId"] for item in missing]
        rows, usage = call_batch(client, kind, missing, expected_ids)
        for row in rows:
            results[row["clusterId"]] = row
        usages.append(
            {
                "clusterIds": expected_ids,
                "usage": usage,
            }
        )
        write_json(
            progress_path,
            {
                "model": MODEL,
                "kind": kind,
                "labels": results,
                "usage": usages,
            },
        )
        print(
            f"{kind}: labeled {len(results)}/{len(substantive)} substantive clusters",
            flush=True,
        )

    ordered = [results[cluster["clusterId"]] for cluster in substantive]
    return ordered, usages


def top_tag_text(cluster: dict, count: int = 5) -> str:
    return ", ".join(
        f"{item['tag']} ({item['frequency']})"
        for item in cluster["frequentTags"][:count]
    )


def make_report(
    node_clusters: list[dict],
    edge_clusters: list[dict],
    labels_by_id: dict[str, dict],
    selected: dict,
    audit: dict,
) -> str:
    lines = [
        "# Konbaung V3 Node-and-Edge Clustering: First Pass",
        "",
        "## Result",
        "",
        (
            "This pass treats every subject and object tag as the same kind of "
            "**node**. Predicates alone are **edges**. Grammatical subject/object "
            "position was not retained or weighted."
        ),
        "",
        (
            f"The selected graph resolution produced "
            f"**{selected['node']['substantiveClusters']} substantive node "
            f"communities** and **{selected['edge']['substantiveClusters']} "
            "substantive edge communities**. Small weak communities remain "
            "unresolved rather than being forced into a category."
        ),
        "",
        "The names below are candidate meta-tags for review, not a finalized ontology.",
        "",
        "## Method",
        "",
        "- Node representation: 70% exact tag embedding + 30% pooled sentence-context embedding; subject and object occurrences contribute identically.",
        "- Edge representation: 70% exact predicate embedding + 30% pooled sentence-context embedding.",
        "- Similarity: exact weighted cosine after approximate nearest-neighbor candidate retrieval.",
        "- Communities: Louvain resolution sweep; selected resolution 2.0 for both node and edge graphs.",
        (
            f"- Node stability/cohesion: ARI {audit['node']['stability']:.3f}; "
            f"weighted mean cohesion {audit['node']['cohesion']:.3f}."
        ),
        (
            f"- Edge stability/cohesion: ARI {audit['edge']['stability']:.3f}; "
            f"weighted mean cohesion {audit['edge']['cohesion']:.3f}."
        ),
        "",
    ]

    for kind, clusters in (("Node", node_clusters), ("Edge", edge_clusters)):
        substantive = [cluster for cluster in clusters if not cluster["unresolved"]]
        unresolved = [cluster for cluster in clusters if cluster["unresolved"]]
        unresolved_tags = sum(cluster["uniqueTags"] for cluster in unresolved)
        lines.extend(
            [
                f"## {kind} candidate meta-tags",
                "",
                (
                    f"{len(substantive)} substantive communities; "
                    f"{unresolved_tags} tags in {len(unresolved)} unresolved "
                    "small communities."
                ),
                "",
                "| Cluster | Candidate meta-tag | Tags | Mentions | Coherence | Frequent examples |",
                "|---|---|---:|---:|---|---|",
            ]
        )
        for cluster in substantive:
            label = labels_by_id[cluster["clusterId"]]
            examples = top_tag_text(cluster).replace("|", "\\|")
            lines.append(
                f"| {cluster['clusterId']} | `{label['candidateMetaTag']}` | "
                f"{cluster['uniqueTags']:,} | {cluster['mentionCount']:,} | "
                f"{label['coherenceAssessment']} | {examples} |"
            )
        lines.append("")
        lines.append(f"### {kind} cluster notes")
        lines.append("")
        for cluster in substantive:
            label = labels_by_id[cluster["clusterId"]]
            alternative = (
                f" Alternative: `{label['alternativeMetaTag']}`."
                if label.get("alternativeMetaTag")
                else ""
            )
            split_note = (
                f" **Split/cleanup note:** {label['splitNote']}"
                if label.get("splitNote")
                else ""
            )
            lines.extend(
                [
                    f"- **{cluster['clusterId']} — "
                    f"`{label['candidateMetaTag']}`.** "
                    f"{label['description']}{alternative}{split_note}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Interpretation",
            "",
            "These communities are a first reduction of lexical variation. A node community does not imply a subject or object role. An edge community is a broad semantic family; it must not silently collapse inverse directions or distinct canonical predicates. Clusters marked medium/low coherence or carrying a split note are the first candidates for the second pass.",
            "",
            "## Integrity",
            "",
            f"- Node tags assigned exactly once: {audit['node']['assigned']:,}.",
            f"- Edge tags assigned exactly once: {audit['edge']['assigned']:,}.",
            f"- Substantive node clusters labeled: {audit['node']['labeled']}.",
            f"- Substantive edge clusters labeled: {audit['edge']['labeled']}.",
            "- The embedding corpus and source triple databases were read only.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    output_dir = args.input_dir.resolve()

    selected_path = output_dir / "selected_resolution.json"
    node_cluster_path = output_dir / "node_clusters.json"
    edge_cluster_path = output_dir / "edge_clusters.json"
    node_assignment_path = output_dir / "node_assignments.jsonl"
    edge_assignment_path = output_dir / "edge_assignments.jsonl"
    source_hashes = {
        path.name: sha256(path)
        for path in (
            selected_path,
            node_cluster_path,
            edge_cluster_path,
            node_assignment_path,
            edge_assignment_path,
        )
    }

    selected = read_json(selected_path)
    node_clusters = read_json(node_cluster_path)
    edge_clusters = read_json(edge_cluster_path)
    node_assignments = load_assignments(node_assignment_path)
    edge_assignments = load_assignments(edge_assignment_path)

    client = genai.Client(api_key=api_key())
    node_labels, node_usage = label_kind(
        client, output_dir, "node", node_clusters, node_assignments
    )
    edge_labels, edge_usage = label_kind(
        client, output_dir, "edge", edge_clusters, edge_assignments
    )
    labels_by_id = {
        label["clusterId"]: label for label in node_labels + edge_labels
    }
    for cluster_id, override in MANUAL_OVERRIDES.items():
        labels_by_id[cluster_id].update(override)
    node_labels = [
        labels_by_id[cluster["clusterId"]]
        for cluster in node_clusters
        if not cluster["unresolved"]
    ]
    edge_labels = [
        labels_by_id[cluster["clusterId"]]
        for cluster in edge_clusters
        if not cluster["unresolved"]
    ]

    write_json(output_dir / "node_cluster_labels.json", node_labels)
    write_json(output_dir / "edge_cluster_labels.json", edge_labels)

    labeled_clusters: dict[str, list[dict]] = {}
    for kind, clusters in (("node", node_clusters), ("edge", edge_clusters)):
        output = []
        for cluster in clusters:
            row = dict(cluster)
            if not cluster["unresolved"]:
                row.update(labels_by_id[cluster["clusterId"]])
            output.append(row)
        labeled_clusters[kind] = output
        write_json(output_dir / f"{kind}_clusters_labeled.json", output)

    for kind, assignment_path in (
        ("node", node_assignment_path),
        ("edge", edge_assignment_path),
    ):
        labeled_path = output_dir / f"{kind}_assignments_labeled.jsonl"
        with assignment_path.open("r", encoding="utf-8") as source, labeled_path.open(
            "w", encoding="utf-8"
        ) as target:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                label = labels_by_id.get(row["clusterId"])
                row["candidateMetaTag"] = (
                    label["candidateMetaTag"] if label else "UNRESOLVED"
                )
                target.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )

    sweep = read_json(output_dir / "resolution_sweep.json")
    chosen_node = next(
        row
        for row in sweep["node"]
        if float(row["resolution"]) == float(selected["node"]["resolution"])
    )
    chosen_edge = next(
        row
        for row in sweep["edge"]
        if float(row["resolution"]) == float(selected["edge"]["resolution"])
    )
    audit = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "node": {
            "assigned": sum(len(rows) for rows in node_assignments.values()),
            "expected": selected["node"]["records"],
            "labeled": len(node_labels),
            "expectedLabels": selected["node"]["substantiveClusters"],
            "stability": chosen_node["stabilityAdjustedRandMean"],
            "cohesion": chosen_node["cohesion"]["weightedMean"],
        },
        "edge": {
            "assigned": sum(len(rows) for rows in edge_assignments.values()),
            "expected": selected["edge"]["records"],
            "labeled": len(edge_labels),
            "expectedLabels": selected["edge"]["substantiveClusters"],
            "stability": chosen_edge["stabilityAdjustedRandMean"],
            "cohesion": chosen_edge["cohesion"]["weightedMean"],
        },
        "sourceHashesBefore": source_hashes,
        "sourceHashesAfter": {
            path.name: sha256(path)
            for path in (
                selected_path,
                node_cluster_path,
                edge_cluster_path,
                node_assignment_path,
                edge_assignment_path,
            )
        },
        "manualOverrides": sorted(MANUAL_OVERRIDES),
        "usage": {"node": node_usage, "edge": edge_usage},
    }
    audit["passed"] = (
        audit["node"]["assigned"] == audit["node"]["expected"]
        and audit["edge"]["assigned"] == audit["edge"]["expected"]
        and audit["node"]["labeled"] == audit["node"]["expectedLabels"]
        and audit["edge"]["labeled"] == audit["edge"]["expectedLabels"]
        and audit["sourceHashesBefore"] == audit["sourceHashesAfter"]
    )
    write_json(output_dir / "cluster_labeling_audit.json", audit)
    report = make_report(
        labeled_clusters["node"],
        labeled_clusters["edge"],
        labels_by_id,
        selected,
        audit,
    )
    (output_dir / "FIRST_PASS_CLUSTER_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": audit["passed"],
                "nodeLabels": len(node_labels),
                "edgeLabels": len(edge_labels),
                "report": str(output_dir / "FIRST_PASS_CLUSTER_REPORT.md"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

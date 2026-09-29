from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(r"C:\Users\conra\Desktop\dighumproject")
OUT = ROOT / "konbaung_node_identity_statistical_refinement_20260725"
CLUSTERS_IN = OUT / "corrected_initial_clusters.jsonl"
ASSIGNMENTS_IN = OUT / "corrected_initial_tag_assignments.csv"
REVIEW_IN = OUT / "pass_01_tentative_review_pairs.json"
RUN_REPORT_IN = OUT / "pass_01_run_report.json"

FINAL_CLUSTERS_JSONL = OUT / "stopped_final_clusters.jsonl"
FINAL_CLUSTERS_CSV = OUT / "stopped_final_clusters.csv"
FINAL_CLUSTERS_MD = OUT / "STOPPED_FINAL_CLUSTERS.md"
FINAL_ASSIGNMENTS_CSV = OUT / "stopped_final_tag_assignments.csv"
FINAL_ASSIGNMENTS_JSONL = OUT / "stopped_final_tag_assignments.jsonl"
ADJUDICATION_JSON = OUT / "pass_01_adjudication.json"
ADJUDICATION_CSV = OUT / "pass_01_adjudication.csv"
ADJUDICATION_MD = OUT / "PASS_01_ADJUDICATED_REVIEW.md"
FINAL_REPORT_JSON = OUT / "stopped_refinement_report.json"
FINAL_REPORT_MD = OUT / "STOPPED_REFINEMENT_REPORT.md"

EXPECTED_INPUT_CLUSTERS = 19_002
EXPECTED_INPUT_TAGS = 23_890
EXPECTED_TOTAL_FREQUENCY = 54_258
EXPECTED_FINAL_CLUSTERS = 18_995

# The left cluster survives. These were manually checked in descending model-score
# order before the first error appeared.
ACCEPTED = [
    (
        "N-5272",
        "N-15271",
        "Exact duplicate personal name; the contextual and role-qualified forms refer to Sithu Thihakyawhtin.",
    ),
    (
        "N-3318",
        "N-10228",
        "Exact duplicate personal name/title, Maha Thiri Thihathu.",
    ),
    (
        "N-4849",
        "N-17119",
        "The bare place name and town/township-qualified forms all refer to Yenangyaung.",
    ),
    (
        "N-5372",
        "N-5376",
        "Formatting and wording variants of the same quantitative entity: 100 service personnel.",
    ),
    (
        "N-0501",
        "N-10786",
        "Spacing/underscore variants of the same official, Min Hla Minkhaung Kyaw.",
    ),
    (
        "N-5659",
        "N-5691",
        "Comma-formatting variants of the same quantitative entity: 3,000 musketeers.",
    ),
    (
        "N-2128",
        "N-5658",
        "Comma/underscore-formatting variants of the same quantitative entity: 3,000 armed men.",
    ),
]

FIRST_ERROR = (
    "N-0182",
    "N-0671",
    "False identity merge: Min Hla Nawrahta and Min Htin Nawrahta are distinct officials. "
    "In vol1_s000240, Nga Hnaung receives the title Min Hla Nawrahta; in vol1_s000252, "
    "Nga Nyo Thu receives the title Min Htin Nawrahta. Later passages also assign them "
    "separate commands and campaign roles.",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def write_csv(path: Path, records: list[dict], fields: list[str] | None = None) -> None:
    if not records:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames = fields or list(records[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def load_assignments() -> tuple[list[dict], list[str]]:
    with ASSIGNMENTS_IN.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def pair_key(pair: dict) -> tuple[str, str]:
    return pair["leftClusterId"], pair["rightClusterId"]


def main() -> None:
    run_report = json.loads(RUN_REPORT_IN.read_text(encoding="utf-8"))
    protected_before = {
        Path(path): digest for path, digest in run_report["protectedHashesBefore"].items()
    }
    for path, expected_hash in protected_before.items():
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Protected input changed before commit: {path}")

    clusters = load_jsonl(CLUSTERS_IN)
    assignments, assignment_fields = load_assignments()
    review_pairs = json.loads(REVIEW_IN.read_text(encoding="utf-8"))

    if len(clusters) != EXPECTED_INPUT_CLUSTERS:
        raise RuntimeError(
            f"Expected {EXPECTED_INPUT_CLUSTERS} input clusters, found {len(clusters)}"
        )
    if len(assignments) != EXPECTED_INPUT_TAGS:
        raise RuntimeError(f"Expected {EXPECTED_INPUT_TAGS} assignments, found {len(assignments)}")
    if len(review_pairs) != 100:
        raise RuntimeError(f"Expected 100 review proposals, found {len(review_pairs)}")

    expected_first_eight = [(left, right) for left, right, _ in ACCEPTED] + [
        (FIRST_ERROR[0], FIRST_ERROR[1])
    ]
    actual_first_eight = [pair_key(pair) for pair in review_pairs[:8]]
    if actual_first_eight != expected_first_eight:
        raise RuntimeError(
            "The first eight review pairs do not match the manually adjudicated sequence.\n"
            f"Expected: {expected_first_eight}\nActual: {actual_first_eight}"
        )

    clusters_by_id = {cluster["clusterId"]: cluster for cluster in clusters}
    redirects = {retired: survivor for survivor, retired, _ in ACCEPTED}
    if len(redirects) != len(ACCEPTED):
        raise RuntimeError("Duplicate retired cluster in accepted merge list")

    for survivor, retired, _ in ACCEPTED:
        if survivor not in clusters_by_id or retired not in clusters_by_id:
            raise RuntimeError(
                f"Accepted merge references a missing cluster: {survivor}, {retired}"
            )
        if survivor == retired:
            raise RuntimeError(f"Self-merge is invalid: {survivor}")

    accepted_by_survivor: dict[str, list[str]] = defaultdict(list)
    reasons_by_survivor: dict[str, list[str]] = defaultdict(list)
    for proposal_number, (survivor, retired, reason) in enumerate(ACCEPTED, start=1):
        accepted_by_survivor[survivor].append(retired)
        reasons_by_survivor[survivor].append(reason)

    merged_clusters: list[dict] = []
    for cluster in clusters:
        cluster_id = cluster["clusterId"]
        if cluster_id in redirects:
            continue

        component_clusters = [cluster]
        for retired in accepted_by_survivor.get(cluster_id, []):
            component_clusters.append(clusters_by_id[retired])

        member_pairs: list[tuple[int, str]] = []
        source_cluster_ids: set[str] = set()
        documented_merge = False
        for component in component_clusters:
            member_pairs.extend(zip(component["memberIndices"], component["memberTags"]))
            source_cluster_ids.update(component.get("sourceClusterIds", [component["clusterId"]]))
            documented_merge = documented_merge or bool(
                component.get("documentedFrozenMerge", False)
            )

        member_pairs.sort(key=lambda item: item[0])
        member_indices = [index for index, _ in member_pairs]
        member_tags = [tag for _, tag in member_pairs]
        if len(member_indices) != len(set(member_indices)):
            raise RuntimeError(f"Duplicate member index after merging into {cluster_id}")

        merged_clusters.append(
            {
                "clusterIndex": len(merged_clusters),
                "clusterId": cluster_id,
                "canonicalLabel": cluster["canonicalLabel"],
                "tagCount": len(member_indices),
                "totalFrequency": sum(
                    int(assignments[index]["frequency"]) for index in member_indices
                ),
                "memberIndices": member_indices,
                "memberTags": member_tags,
                "sourceClusterIds": sorted(source_cluster_ids),
                "documentedFrozenMerge": documented_merge,
                "acceptedStatisticalMerge": bool(accepted_by_survivor.get(cluster_id)),
                "retiredClusterIds": accepted_by_survivor.get(cluster_id, []),
                "acceptedProposalNumbers": [
                    proposal_number
                    for proposal_number, (survivor, _, _) in enumerate(ACCEPTED, start=1)
                    if survivor == cluster_id
                ],
                "adjudicationNotes": reasons_by_survivor.get(cluster_id, []),
            }
        )

    if len(merged_clusters) != EXPECTED_FINAL_CLUSTERS:
        raise RuntimeError(
            f"Expected {EXPECTED_FINAL_CLUSTERS} final clusters, found {len(merged_clusters)}"
        )

    final_by_member: dict[int, dict] = {}
    for cluster in merged_clusters:
        for member_index in cluster["memberIndices"]:
            if member_index in final_by_member:
                raise RuntimeError(f"Member index assigned twice: {member_index}")
            final_by_member[member_index] = cluster
    if set(final_by_member) != set(range(EXPECTED_INPUT_TAGS)):
        missing = sorted(set(range(EXPECTED_INPUT_TAGS)) - set(final_by_member))
        extra = sorted(set(final_by_member) - set(range(EXPECTED_INPUT_TAGS)))
        raise RuntimeError(f"Assignment coverage error. Missing={missing[:10]}, extra={extra[:10]}")

    accepted_lookup = {
        retired: (proposal_number, survivor)
        for proposal_number, (survivor, retired, _) in enumerate(ACCEPTED, start=1)
    }
    final_assignments: list[dict] = []
    for member_index, row in enumerate(assignments):
        cluster = final_by_member[member_index]
        previous_cluster = row["working_cluster_id"]
        accepted_info = accepted_lookup.get(previous_cluster)
        output = dict(row)
        output.update(
            {
                "refined_cluster_id": cluster["clusterId"],
                "refined_canonical_label": cluster["canonicalLabel"],
                "statistical_merge_applied": "YES" if accepted_info else "NO",
                "statistical_merge_proposal": str(accepted_info[0]) if accepted_info else "",
                "statistical_merge_survivor": accepted_info[1] if accepted_info else "",
                "refinement_status": "STOPPED_AFTER_FIRST_FALSE_MERGE_PROPOSAL",
            }
        )
        final_assignments.append(output)

    total_frequency = sum(int(row["frequency"]) for row in final_assignments)
    if total_frequency != EXPECTED_TOTAL_FREQUENCY:
        raise RuntimeError(
            f"Expected total frequency {EXPECTED_TOTAL_FREQUENCY}, found {total_frequency}"
        )
    if sum(cluster["totalFrequency"] for cluster in merged_clusters) != total_frequency:
        raise RuntimeError("Cluster frequencies do not reconcile to assignment frequencies")

    adjudication: list[dict] = []
    accepted_reasons = {(left, right): reason for left, right, reason in ACCEPTED}
    for proposal_number, pair in enumerate(review_pairs, start=1):
        key = pair_key(pair)
        if proposal_number <= 7:
            status = "ACCEPTED"
            reason = accepted_reasons[key]
            reviewed = True
        elif proposal_number == 8:
            status = "REJECTED_FIRST_ERROR"
            reason = FIRST_ERROR[2]
            reviewed = True
        else:
            status = "NOT_REVIEWED_AFTER_STOP"
            reason = "Not adjudicated because the required stop condition triggered at proposal 8."
            reviewed = False
        adjudication.append(
            {
                "proposalNumber": proposal_number,
                "status": status,
                "manuallyReviewed": reviewed,
                "leftClusterId": pair["leftClusterId"],
                "rightClusterId": pair["rightClusterId"],
                "leftCanonical": pair["leftCanonical"],
                "rightCanonical": pair["rightCanonical"],
                "pairType": pair["pairType"],
                "modelProbability": pair["modelProbability"],
                "reason": reason,
            }
        )

    write_jsonl(FINAL_CLUSTERS_JSONL, merged_clusters)
    cluster_csv_records = [
        {
            "cluster_index": cluster["clusterIndex"],
            "cluster_id": cluster["clusterId"],
            "canonical_label": cluster["canonicalLabel"],
            "tag_count": cluster["tagCount"],
            "total_frequency": cluster["totalFrequency"],
            "member_tags_json": json.dumps(cluster["memberTags"], ensure_ascii=False),
            "source_cluster_ids_json": json.dumps(cluster["sourceClusterIds"], ensure_ascii=False),
            "documented_frozen_merge": cluster["documentedFrozenMerge"],
            "accepted_statistical_merge": cluster["acceptedStatisticalMerge"],
            "retired_cluster_ids_json": json.dumps(
                cluster["retiredClusterIds"], ensure_ascii=False
            ),
            "accepted_proposal_numbers_json": json.dumps(cluster["acceptedProposalNumbers"]),
        }
        for cluster in merged_clusters
    ]
    write_csv(FINAL_CLUSTERS_CSV, cluster_csv_records)
    write_csv(
        FINAL_ASSIGNMENTS_CSV,
        final_assignments,
        assignment_fields
        + [
            "refined_cluster_id",
            "refined_canonical_label",
            "statistical_merge_applied",
            "statistical_merge_proposal",
            "statistical_merge_survivor",
            "refinement_status",
        ],
    )
    write_jsonl(FINAL_ASSIGNMENTS_JSONL, final_assignments)
    ADJUDICATION_JSON.write_text(
        json.dumps(adjudication, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(ADJUDICATION_CSV, adjudication)

    review_lines = [
        "# Pass 1 adjudicated review",
        "",
        "Review proceeded in descending model-score order. The first false identity merge "
        "appeared at proposal 8, so review and all further statistical passes stopped there.",
        "",
        "## Reviewed proposals",
        "",
    ]
    for item in adjudication[:8]:
        review_lines.extend(
            [
                f"### {item['proposalNumber']:03d} — {item['status']}",
                "",
                f"- Left: `{item['leftClusterId']}` — {item['leftCanonical']}",
                f"- Right: `{item['rightClusterId']}` — {item['rightCanonical']}",
                f"- Pair type: {item['pairType']}",
                f"- Model probability: {item['modelProbability']:.9f}",
                f"- Adjudication: {item['reason']}",
                "",
            ]
        )
    review_lines.extend(
        [
            "## Unreviewed proposals",
            "",
            "Proposals 009–100 were not adjudicated or merged because proposal 008 triggered "
            "the user-specified stop condition. Their candidate data remains in the machine-readable "
            "adjudication files with status `NOT_REVIEWED_AFTER_STOP`.",
            "",
        ]
    )
    ADJUDICATION_MD.write_text("\n".join(review_lines), encoding="utf-8")

    cluster_lines = [
        "# Stopped final entity-disambiguation clusters",
        "",
        f"- Clusters: {len(merged_clusters):,}",
        f"- Tags: {len(final_assignments):,}",
        f"- Total frequency: {total_frequency:,}",
        "- Includes the documented N-0397 → N-0501 correction and only accepted statistical "
        "proposals 001–007.",
        "",
    ]
    for cluster in merged_clusters:
        merge_note = ""
        if cluster["retiredClusterIds"]:
            merge_note = f" | merged: {', '.join(cluster['retiredClusterIds'])}"
        cluster_lines.extend(
            [
                f"## {cluster['clusterId']} — {cluster['canonicalLabel']}",
                "",
                f"Tags: {cluster['tagCount']:,} | Frequency: {cluster['totalFrequency']:,}{merge_note}",
                "",
            ]
        )
        for member_index, tag in zip(cluster["memberIndices"], cluster["memberTags"]):
            cluster_lines.append(f"- {tag} — {int(assignments[member_index]['frequency']):,}")
        cluster_lines.append("")
    FINAL_CLUSTERS_MD.write_text("\n".join(cluster_lines), encoding="utf-8")

    protected_after = {str(path): sha256(path) for path in protected_before}
    protected_unchanged = all(
        protected_after[str(path)] == expected_hash
        for path, expected_hash in protected_before.items()
    )
    if not protected_unchanged:
        raise RuntimeError("A protected input changed during finalization")

    final_report = {
        "status": "STOPPED_ON_FIRST_MANUAL_ERROR",
        "reviewPass": 1,
        "firstRejectedProposal": 8,
        "reviewedProposals": 8,
        "acceptedStatisticalMerges": 7,
        "rejectedStatisticalMerges": 1,
        "unreviewedProposals": 92,
        "inputManualClusters": 19_003,
        "documentedCorrection": {
            "retiredClusterId": "N-0397",
            "survivingClusterId": "N-0501",
            "reason": "The supplied bundle's own review queue identified these as the same official.",
        },
        "correctedInitialClusters": EXPECTED_INPUT_CLUSTERS,
        "finalClusters": len(merged_clusters),
        "tags": len(final_assignments),
        "totalFrequency": total_frequency,
        "candidatePairs": run_report["candidatePairs"],
        "candidatePairTypes": run_report["candidatePairTypes"],
        "acceptedPairs": [
            {
                "proposalNumber": proposal_number,
                "survivor": survivor,
                "retired": retired,
                "reason": reason,
            }
            for proposal_number, (survivor, retired, reason) in enumerate(ACCEPTED, start=1)
        ],
        "firstError": {
            "proposalNumber": 8,
            "leftClusterId": FIRST_ERROR[0],
            "rightClusterId": FIRST_ERROR[1],
            "reason": FIRST_ERROR[2],
        },
        "validation": {
            "everyTagAssignedExactlyOnce": len(final_by_member) == EXPECTED_INPUT_TAGS,
            "tagCount": len(final_assignments),
            "frequencyPreserved": total_frequency == EXPECTED_TOTAL_FREQUENCY,
            "totalFrequency": total_frequency,
            "clusterCount": len(merged_clusters),
            "protectedSourcesUnchanged": protected_unchanged,
            "protectedHashes": protected_after,
        },
        "newEmbeddingApiCalls": 0,
        "artifacts": [
            path.name
            for path in [
                FINAL_CLUSTERS_JSONL,
                FINAL_CLUSTERS_CSV,
                FINAL_CLUSTERS_MD,
                FINAL_ASSIGNMENTS_CSV,
                FINAL_ASSIGNMENTS_JSONL,
                ADJUDICATION_JSON,
                ADJUDICATION_CSV,
                ADJUDICATION_MD,
                FINAL_REPORT_JSON,
                FINAL_REPORT_MD,
            ]
        ],
    }
    FINAL_REPORT_JSON.write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report_lines = [
        "# Stopped statistical entity-disambiguation refinement",
        "",
        "## Outcome",
        "",
        "The requested stop condition triggered in the first manual-review pass. Proposals "
        "001–007 were valid identity merges. Proposal 008 was the first error, so it was "
        "rejected and no later proposal or later pass was applied.",
        "",
        f"- Starting manual clusters: {19_003:,}",
        f"- After documented N-0397 → N-0501 correction: {EXPECTED_INPUT_CLUSTERS:,}",
        f"- Accepted statistical merges: {len(ACCEPTED):,}",
        f"- Final stopped clusters: {len(merged_clusters):,}",
        f"- Tags assigned exactly once: {len(final_assignments):,}",
        f"- Preserved total frequency: {total_frequency:,}",
        f"- Candidate pairs considered statistically: {run_report['candidatePairs']:,}",
        "- New embedding/API calls: 0",
        "- Protected source bundle and vector files: unchanged (SHA-256 verified)",
        "",
        "## First rejected proposal",
        "",
        f"`{FIRST_ERROR[0]}` (Min Hla Nawrahta) must not merge with `{FIRST_ERROR[1]}` "
        "(Min Htin Nawrahta). They are distinct officials: `vol1_s000240` identifies Nga "
        "Hnaung as Min Hla Nawrahta, while `vol1_s000252` identifies Nga Nyo Thu as Min "
        "Htin Nawrahta. This is proposal 008 in score order.",
        "",
        "## Scope of the committed result",
        "",
        "The final working set includes the one documented correction from the supplied "
        "bundle plus accepted review proposals 001–007. Proposal 008 and proposals "
        "009–100 were not merged. No second statistical pass was run.",
        "",
        "## Validation",
        "",
        "- Every one of the 23,890 source tags occurs exactly once in the stopped assignment set.",
        "- Cluster frequencies reconcile exactly to the preserved corpus total of 54,258.",
        f"- The final cluster count is exactly {EXPECTED_FINAL_CLUSTERS:,}.",
        "- All protected input hashes match their pre-run values.",
        "",
        "See `PASS_01_ADJUDICATED_REVIEW.md` for the decision log and "
        "`stopped_refinement_report.json` for machine-readable validation and provenance.",
        "",
    ]
    FINAL_REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")

    # Re-hash protected sources after every output has been written.
    for path, expected_hash in protected_before.items():
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Protected input changed after commit: {path}")

    print(
        json.dumps(
            {
                "status": final_report["status"],
                "finalClusters": len(merged_clusters),
                "tags": len(final_assignments),
                "totalFrequency": total_frequency,
                "accepted": len(ACCEPTED),
                "firstRejectedProposal": 8,
                "protectedSourcesUnchanged": True,
                "report": str(FINAL_REPORT_MD),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build one compact review containing only Gemini calls with accepted aliases."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(
    r"C:\Users\conra\Desktop\dighumproject"
    r"\konbaung_nonsingleton_top50_frequency_wave_gemini_trial_20260902_first100"
)
CLUSTERS_PATH = ROOT / "resolved_clusters_after_nonsingleton_completion.json"
MANIFEST_PATH = ROOT / "full_nonsingleton_run_manifest.json"
OUTPUT_PATH = ROOT / "MASTER_POSITIVE_RESOLUTIONS_REVIEW.md"


# Escape characters that would otherwise split a compact Markdown value across fields.
def markdown_value(value: object) -> str:
    return (
        str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    )


# Render only clusters where Gemini accepted at least one supplied alias as the same entity.
def main() -> None:
    clusters = json.loads(CLUSTERS_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    positive = [cluster for cluster in clusters if len(cluster["memberIds"]) > 1]
    positive.sort(key=lambda cluster: (int(cluster["wave"]), int(cluster["waveOrder"])))

    accepted_aliases = sum(len(cluster["memberIds"]) - 1 for cluster in positive)
    passed_calls = int(manifest["totalCalls"]) - len(positive)
    positive_tags = len(positive) + accepted_aliases

    if accepted_aliases != int(manifest["totalAliasesAbsorbedWithoutOwnCall"]):
        raise RuntimeError(
            "Positive-cluster alias total does not match the completed-run manifest."
        )

    lines = [
        "# Master review: positive entity resolutions",
        "",
        "This review includes only calls where Gemini accepted at least one supplied candidate as the same underlying entity. Calls returning no matches are excluded.",
        "",
        f"- Completed waves: {manifest['completedWaves']}",
        f"- Total Gemini calls: {manifest['totalCalls']}",
        f"- Positive resolution pages included: {len(positive)}",
        f"- No-match pages excluded: {passed_calls}",
        f"- Accepted aliases: {accepted_aliases}",
        f"- Singleton aliases accepted: {manifest['singletonAliasesAbsorbed']}",
        f"- Non-singleton aliases accepted before their own call: {manifest['futureNonSingletonCallsAvoided']}",
        f"- Entity tags represented in positive clusters: {positive_tags}",
        "",
        "The `YES` values below are the final conformed resolutions. Five model-returned strings that were not supplied candidates were discarded and are not listed.",
        "",
    ]

    current_wave = None
    review_number = 0
    for cluster in positive:
        wave = int(cluster["wave"])
        if wave != current_wave:
            current_wave = wave
            lines.extend([f"## Wave {wave}", ""])

        review_number += 1
        parent = markdown_value(cluster["parentEntity"])
        mentions = int(cluster["parentMentions"])
        aliases = [markdown_value(value) for value in cluster["members"][1:]]
        rejected_count = len(cluster["different"])
        candidates_sent = len(aliases) + rejected_count
        lines.extend(
            [
                f"### {review_number}. {parent} ({mentions} mentions)",
                "",
                f"- YES: {' | '.join(aliases)}",
                f"- NO: {rejected_count} omitted supplied candidates",
                f"- Candidates sent: {candidates_sent}",
                "",
            ]
        )

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT_PATH),
                "positivePages": len(positive),
                "excludedPasses": passed_calls,
                "acceptedAliases": accepted_aliases,
                "representedTags": positive_tags,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

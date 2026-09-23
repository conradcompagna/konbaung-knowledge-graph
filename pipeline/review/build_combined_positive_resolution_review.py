#!/usr/bin/env python3
"""Build one positive-only review spanning both Konbaung resolution phases."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"C:\Users\conra\Desktop\dighumproject")
OUTPUT_ROOT = ROOT / "konbaung_singleton_completion_gemini_20260902"
CLUSTERS_PATH = OUTPUT_ROOT / "resolved_clusters_all.json"
MANIFEST_PATH = OUTPUT_ROOT / "FINAL_ALL_ENTITY_RESOLUTION_MANIFEST.json"
OUTPUT_PATH = OUTPUT_ROOT / "MASTER_POSITIVE_RESOLUTIONS_ALL.md"


def markdown_value(value: object) -> str:
    """Escape characters that would split a compact Markdown field."""
    return (
        str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    )


def main() -> None:
    """Render only calls that accepted at least one supplied alias."""
    clusters = json.loads(CLUSTERS_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    positive = [cluster for cluster in clusters if len(cluster["memberIds"]) > 1]
    positive.sort(key=lambda cluster: (int(cluster["wave"]), int(cluster["waveOrder"])))

    accepted_aliases = sum(len(cluster["memberIds"]) - 1 for cluster in positive)
    no_match_calls = int(manifest["totalCallsBothPhases"]) - len(positive)
    represented_tags = len(positive) + accepted_aliases
    if accepted_aliases != int(manifest["totalCallsAvoidedBothPhases"]):
        raise RuntimeError("Accepted-alias total does not match the final manifest.")

    lines = [
        "# Master review: all positive entity resolutions",
        "",
        "This combines both phases and includes only calls where Gemini accepted at least one supplied candidate as the same underlying entity. Calls returning no matches are excluded.",
        "",
        f"- Completed waves: {manifest['lastCombinedWave']}",
        f"- Total Gemini calls: {manifest['totalCallsBothPhases']}",
        f"- Positive resolution pages included: {len(positive)}",
        f"- No-match pages excluded: {no_match_calls}",
        f"- Accepted aliases: {accepted_aliases}",
        f"- Accepted aliases in waves 1-95: {accepted_aliases - int(manifest['singletonCallsAvoided'])}",
        f"- Accepted aliases in waves 96-155: {manifest['singletonCallsAvoided']}",
        f"- Entity tags represented in positive clusters: {represented_tags}",
        "",
        "The `YES` values are final conformed decisions. Twenty-two returned strings that were not supplied candidates were discarded and are not shown.",
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
                "excludedNoMatches": no_match_calls,
                "acceptedAliases": accepted_aliases,
                "representedTags": represented_tags,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bundle the five latest random-page trial responses into one JSON artifact."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRIAL_ROOT = ROOT / "konbaung_historiography_ungrounded_trials"
OUTPUT_PATH = TRIAL_ROOT / "five_random_pages_high_thinking_triples.json"
MARKDOWN_OUTPUT_PATH = TRIAL_ROOT / "five_random_pages_high_thinking_triples.md"

RESULT_PATHS = [
    TRIAL_ROOT
    / "vol2"
    / "page_0300"
    / "historiography_ungrounded_high_thinking_01"
    / "result.json",
    TRIAL_ROOT
    / "vol1"
    / "page_0078"
    / "historiography_ungrounded_high_thinking_retry_03"
    / "result.json",
    TRIAL_ROOT
    / "vol1"
    / "page_0409"
    / "historiography_ungrounded_high_thinking_01"
    / "result.json",
    TRIAL_ROOT
    / "vol2"
    / "page_0391"
    / "historiography_ungrounded_high_thinking_01"
    / "result.json",
    TRIAL_ROOT
    / "vol1"
    / "page_0310"
    / "historiography_ungrounded_high_thinking_01"
    / "result.json",
]


def main() -> None:
    pages: list[dict[str, object]] = []
    total_sentences = 0
    total_annotated = 0
    total_skipped = 0
    total_triples = 0

    for path in RESULT_PATHS:
        result = json.loads(path.read_text(encoding="utf-8"))
        input_data = json.loads((path.parent / "input.json").read_text(encoding="utf-8"))
        validation = result["validation"]
        source_sentences = {sentence["sid"]: sentence for sentence in input_data["sentences"]}
        pages.append(
            {
                "volume": result["volume"],
                "page": result["page"],
                "summary": input_data["page_summary"],
                "sourceSentences": source_sentences,
                "sentences": result["response"]["sentences"],
            }
        )
        total_sentences += int(validation["sentenceCount"])
        total_annotated += int(validation["annotatedSentenceCount"])
        total_skipped += int(validation["skippedSentenceCount"])
        total_triples += int(validation["tripleCount"])

    bundle = {
        "schema": {
            "sentence": ["sid", "decision", "justification", "triples"],
            "triple": ["subject", "predicate", "object"],
        },
        "totals": {
            "pageCount": len(pages),
            "sentenceCount": total_sentences,
            "annotatedSentenceCount": total_annotated,
            "skippedSentenceCount": total_skipped,
            "tripleCount": total_triples,
        },
        "pages": pages,
    }
    OUTPUT_PATH.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Five random pages: historiography triples",
        "",
        f"- Pages: {len(pages)}",
        f"- Sentences: {total_sentences}",
        f"- Triples: {total_triples}",
        "",
    ]
    for page in pages:
        lines.extend(
            [
                f"## Volume {page['volume']}, page {int(page['page']):04d}",
                "",
                "### Page summary",
                "",
                page["summary"],
                "",
            ]
        )
        for sentence in page["sentences"]:
            source = page["sourceSentences"][sentence["sid"]]
            lines.extend(
                [
                    f"### `{sentence['sid']}`",
                    "",
                    "**Burmese sentence**",
                    "",
                    source["my"],
                    "",
                    "**English translation**",
                    "",
                    source["en"],
                    "",
                    f"**Decision:** `{sentence['decision']}`",
                    "",
                ]
            )
            if sentence["decision"] == "skip":
                lines.extend(
                    [
                        "**Skip justification**",
                        "",
                        sentence["justification"],
                        "",
                    ]
                )
                continue
            lines.extend(["**Triples**", ""])
            for triple in sentence["triples"]:
                lines.extend(
                    [
                        f"- **S:** {triple['subject']}",
                        f"  **P:** `{triple['predicate']}`",
                        f"  **O:** {triple['object']}",
                        "",
                    ]
                )

    MARKDOWN_OUTPUT_PATH.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    print(MARKDOWN_OUTPUT_PATH)


if __name__ == "__main__":
    main()

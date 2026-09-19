#!/usr/bin/env python3
"""Run the sentence-analysis enrichment schema without modifying the corpus."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

from konbaung_gemini_summary_claim_completion_annotator import api_key
from konbaung_gemini_translated_sentence_triple_enrichment_test import (
    EnrichmentResult,
    build_payload,
    dynamic_input,
    validate as validate_additions,
    write_json,
)
from konbaung_gemini_translated_sentence_triples_test import (
    SentenceTriples,
    Triple as ExistingTriple,
)


ROOT = Path(__file__).resolve().parent
PROMPT_PATH = ROOT / "konbaung_enrichment_review_prompt.md"
OUTPUT_ROOT = ROOT / "konbaung_enrichment_review_trials"
MODEL = "gemini-3.1-flash-lite"
PREDICATE_PATTERN = r"^[A-Z0-9]+(?:_[A-Z0-9]+)*$"


class Endpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    my: str
    en: str
    tag: str
    source: Literal["sentence", "page", "inferred"]


class Triple(BaseModel):
    model_config = ConfigDict(extra="forbid")

    s: Endpoint
    p: str = Field(
        description="Concise relation label in ALL_CAPS_WITH_UNDERSCORES.",
        pattern=PREDICATE_PATTERN,
    )
    o: Endpoint


class SentenceReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sid: str = Field(description="Exact supplied sentence ID.")
    analysis: str = Field(
        max_length=1000,
        description=(
            "One freeform paragraph of no more than 100 words explaining relevance, existing "
            "coverage, and whether anything important remains missing."
        ),
    )
    T: list[Triple] = Field(
        description="Only genuinely missing triples; use an empty array if coverage is sufficient."
    )


class ReviewResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        title="KonbaungEnrichmentReview",
    )

    S: list[SentenceReview] = Field(
        description="One review record for every supplied sentence, in the same order."
    )


def render_prompt(payload: dict, prompt_path: Path) -> str:
    template = prompt_path.read_text(encoding="utf-8-sig")
    marker = "{{ENRICHMENT_INPUT}}"
    if template.count(marker) != 1:
        raise ValueError(f"Expected exactly one {marker} marker in {prompt_path}")
    return template.replace(marker, dynamic_input(payload).strip())


def response_parts(response: types.GenerateContentResponse) -> tuple[str, list[str]]:
    answer_parts: list[str] = []
    thought_parts: list[str] = []
    for candidate in response.candidates or []:
        content = candidate.content
        if content is None:
            continue
        for part in content.parts or []:
            text = part.text or ""
            if not text:
                continue
            if getattr(part, "thought", False):
                thought_parts.append(text)
            else:
                answer_parts.append(text)
    return "".join(answer_parts), thought_parts


def validate(payload: dict, result: ReviewResult) -> list[str]:
    errors: list[str] = []
    expected_ids = [sentence["id"] for sentence in payload["sentences"]]
    returned_ids = [review.sid for review in result.S]
    if returned_ids != expected_ids:
        errors.append(
            "Sentence IDs/order differ from supplied input: "
            f"expected {expected_ids}, returned {returned_ids}"
        )

    for review in result.S:
        words = re.findall(r"\S+", review.analysis)
        if len(words) > 100:
            errors.append(f"{review.sid}: analysis has {len(words)} words; maximum is 100")
        if "\n" in review.analysis.strip():
            errors.append(f"{review.sid}: analysis must be one paragraph")

    addition_groups: list[SentenceTriples] = []
    for review in result.S:
        if not review.T:
            continue
        addition_groups.append(
            SentenceTriples(
                sid=review.sid,
                T=[ExistingTriple.model_validate(triple.model_dump()) for triple in review.T],
            )
        )
    addition_result = EnrichmentResult(S=addition_groups)
    errors.extend(validate_additions(payload, addition_result))
    return errors


def triple_lines(index: int, triple: Triple | ExistingTriple) -> list[str]:
    return [
        f"{index}. Subject: `{triple.s.my}` — {triple.s.en} "
        f"[`{triple.s.tag}`; source=`{triple.s.source}`]",
        f"   Relation: `{triple.p}`",
        f"   Object: `{triple.o.my}` — {triple.o.en} "
        f"[`{triple.o.tag}`; source=`{triple.o.source}`]",
    ]


def review_text(
    payload: dict,
    result: ReviewResult,
    thoughts: list[str],
    errors: list[str],
) -> str:
    sentence_by_sid = {sentence["id"]: sentence for sentence in payload["sentences"]}
    existing_by_sid = {
        sentence["id"]: [
            ExistingTriple.model_validate(item) for item in sentence["existing_triples"]
        ]
        for sentence in payload["sentences"]
    }
    status = "VALID" if not errors else "FLAGGED"
    added_count = sum(len(review.T) for review in result.S)
    lines = [
        f"# {payload['page_id']} analysis-first enrichment review",
        "",
        status,
        "",
        f"Gemini returned {len(result.S)} sentence reviews and {added_count} proposed triple(s).",
        "",
        "## Gemini returned thinking",
        "",
    ]
    if thoughts:
        for index, thought in enumerate(thoughts, start=1):
            if len(thoughts) > 1:
                lines.extend([f"### Thought part {index}", ""])
            lines.extend([thought.strip(), ""])
    else:
        lines.extend(["No thought summary was returned by the API.", ""])

    for review in result.S:
        sentence = sentence_by_sid.get(review.sid)
        if sentence is None:
            continue
        lines.extend(
            [
                f"## {review.sid}",
                "",
                "### Original Burmese sentence",
                "",
                sentence["my"],
                "",
                "### English translation",
                "",
                sentence["en"],
                "",
                "### Existing triples",
                "",
            ]
        )
        existing = existing_by_sid.get(review.sid, [])
        if existing:
            for index, triple in enumerate(existing, start=1):
                lines.extend(triple_lines(index, triple))
                lines.append("")
        else:
            lines.extend(["None.", ""])
        lines.extend(["### Gemini analysis", "", review.analysis.strip(), ""])
        lines.extend(["### Gemini proposed additions", ""])
        if review.T:
            for index, triple in enumerate(review.T, start=1):
                lines.extend(triple_lines(index, triple))
                lines.append("")
        else:
            lines.extend(["None proposed.", ""])

    if errors:
        lines.extend(["## Validation warnings", ""])
        lines.extend(f"- {error}" for error in errors)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run(
    volume: int,
    page_number: int,
    output_name: str,
    thinking_level: types.ThinkingLevel,
    max_output_tokens: int,
    prompt_path: Path,
) -> Path:
    payload = build_payload(volume, page_number)
    output_dir = OUTPUT_ROOT / f"vol{volume}" / f"page_{page_number:04d}" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = render_prompt(payload, prompt_path)

    write_json(output_dir / "payload.json", payload)
    write_json(output_dir / "response_schema.json", ReviewResult.model_json_schema())
    (output_dir / "prompt_sent.txt").write_text(prompt, encoding="utf-8")

    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=ReviewResult.model_json_schema(),
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_level=thinking_level,
                include_thoughts=True,
            ),
        ),
    )
    raw, thoughts = response_parts(response)
    (output_dir / "raw_response.json").write_text(raw, encoding="utf-8")
    (output_dir / "returned_thoughts.txt").write_text(
        "\n\n".join(thoughts), encoding="utf-8"
    )
    result = ReviewResult.model_validate_json(raw)
    errors = validate(payload, result)
    write_json(
        output_dir / "result.json",
        {
            "accepted": not errors,
            "errors": errors,
            "request": {
                "model": MODEL,
                "thinking_level": thinking_level.value,
                "include_thoughts": True,
                "temperature": 0.0,
                "candidate_count": 1,
                "max_output_tokens": max_output_tokens,
                "prompt_path": str(prompt_path),
            },
            "usage": response.usage_metadata.model_dump(mode="json")
            if response.usage_metadata
            else {},
            "thought_parts": thoughts,
            "response": result.model_dump(mode="json"),
        },
    )
    (output_dir / "review.md").write_text(
        review_text(payload, result, thoughts, errors), encoding="utf-8"
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int, required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--output-name", default="analysis_first_trial_01")
    parser.add_argument(
        "--thinking-level",
        choices=("low", "medium", "high"),
        default="low",
        help="Gemini 3 native reasoning control; minimal is intentionally excluded.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--prompt", type=Path, default=PROMPT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = run(
        args.volume,
        args.page,
        args.output_name,
        types.ThinkingLevel(args.thinking_level.upper()),
        args.max_output_tokens,
        args.prompt,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "review": str(output_dir / "review.md"),
                "thoughts": str(output_dir / "returned_thoughts.txt"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

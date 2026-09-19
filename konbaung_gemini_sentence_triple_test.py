#!/usr/bin/env python3
"""Sentence-first translation and triple extraction experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from konbaung_gemini_cite_sources_annotator import MODEL, ORIGINAL_PROMPT_PATH
from konbaung_gemini_summary_claim_completion_annotator import api_key
from konbaung_gemini_triple_metadata_semantic_frame import load_page


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "konbaung_sentence_triple_tests"


TASK_PROMPT = """<NEW_SENTENCE_FIRST_TASK>
The original prompt above is read-only context. Use it only to understand the historical research question and the intended style of open-coded entities and relations. Do not follow its output instructions. Follow only this new task.

Segment TARGET_PAGE_TEXT into complete Burmese sentences or sentence-like grammatical units. Chronicle sentences may be long, cross OCR line breaks, and contain several propositions. Preserve each accepted sentence verbatim in `my`; OCR line breaks are not sentence boundaries.

Put running headers, page numbers, footnotes, citations, publisher text, and incomplete sentence fragments cut off at the beginning or end of the page in `X`. Copy each ignored fragment verbatim. Do not translate or annotate `X`.

For every complete substantive sentence, first provide a faithful full English translation. Then extract every coherent subject-predicate-object proposition expressed by that sentence. A sentence can, and often does, require more than one triple. Do not collapse distinct propositions into one relation and do not omit propositions merely because they share an actor.

For each subject and object, record a concise Burmese referent, English gloss, and open-coded entity tag. Set `source` to:
- `sentence` when the Burmese referent is explicitly present in that sentence;
- `page` when it is omitted from the sentence but explicitly named elsewhere in TARGET_PAGE_TEXT;
- `inferred` when it is absent from the page and must be reconstructed from grammar, genre, or context.

When `source` is `sentence` or `page`, `my` must be an exact Burmese span from that location. When `source` is `inferred`, write the concise referent you believe is intended and explicitly retain the `inferred` marker. Burmese royal-chronicle prose frequently omits royal agents and shared objects, so recover them when needed for a coherent claim rather than producing malformed triples.

Use a concise ALL_CAPS open-coded relation label in `p`. Return every accepted sentence in page order and every proposition in sentence order. Return only `X` and `S` through the structured schema. Do not return dates, places, quantities, evidence blocks, explanations, confidence scores, axial codes, or other metadata.
</NEW_SENTENCE_FIRST_TASK>"""


class Endpoint(BaseModel):
    my: str = Field(description="Burmese referent span or concise inferred Burmese referent.")
    en: str = Field(description="Compact English gloss of the referent.")
    tag: str = Field(description="Concise open-coded entity tag.")
    source: Literal["sentence", "page", "inferred"]


class Triple(BaseModel):
    s: Endpoint
    p: str = Field(description="Concise ALL_CAPS open-coded relation label.")
    o: Endpoint


class SentenceRecord(BaseModel):
    n: int
    my: str = Field(description="Complete verbatim Burmese sentence from TARGET_PAGE_TEXT.")
    en: str = Field(description="Full faithful English sentence translation.")
    T: list[Triple] = Field(min_length=1)


class SentenceTripleResult(BaseModel):
    X: list[str] = Field(description="Verbatim ignored page fragments.")
    S: list[SentenceRecord]


def full_prompt(page_text: str) -> str:
    original = ORIGINAL_PROMPT_PATH.read_text(encoding="utf-8-sig")
    return f"""<ORIGINAL_PROMPT_READ_ONLY>
{original}
</ORIGINAL_PROMPT_READ_ONLY>

{TASK_PROMPT}

<TARGET_PAGE_TEXT>
{page_text}
</TARGET_PAGE_TEXT>
"""


def review_text(result: SentenceTripleResult) -> str:
    lines = ["# Ignored", ""]
    for item in result.X:
        lines.extend([item, ""])
    lines.extend(["# Sentences", ""])
    for sentence in result.S:
        lines.extend([f"## {sentence.n}", "", sentence.my, "", sentence.en, ""])
        for triple in sentence.T:
            lines.append(
                f"- {triple.s.my} [{triple.s.tag}; {triple.s.source}] "
                f"--{triple.p}--> {triple.o.my} [{triple.o.tag}; {triple.o.source}]"
            )
            lines.append(f"  {triple.s.en} -> {triple.o.en}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run(volume: int, page_number: int) -> Path:
    page = load_page(volume, page_number)
    prompt = full_prompt(page["canonicalText"])
    output_dir = OUTPUT_ROOT / f"vol{volume}" / f"page_{page_number:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_sent.txt").write_text(prompt, encoding="utf-8")

    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SentenceTripleResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=12000,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = response.text or ""
    (output_dir / "raw_response.json").write_text(raw, encoding="utf-8")
    result = SentenceTripleResult.model_validate_json(raw)
    (output_dir / "result.json").write_text(
        json.dumps(
            {
                "usage": response.usage_metadata.model_dump(mode="json")
                if response.usage_metadata
                else {},
                "response": result.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "review.md").write_text(review_text(result), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int, required=True)
    parser.add_argument("--page", type=int, required=True)
    args = parser.parse_args()
    print(run(args.volume, args.page))


if __name__ == "__main__":
    main()

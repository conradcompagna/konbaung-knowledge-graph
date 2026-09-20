#!/usr/bin/env python3
"""Minimal semantic S-P-O extraction from existing Burmese sentence translations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel

from build_konbaung_reader_data import span_candidates
from konbaung_gemini_summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "konbaung_reader_app" / "static" / "data" / "konbaung"
OUTPUT_ROOT = ROOT / "konbaung_minimal_semantic_triple_trials"
MODEL = "gemini-3.1-flash-lite"


class SemanticTriple(BaseModel):
    s_my: str
    s_en: str
    p_my: str
    p_en: str
    o_my: str
    o_en: str


class SentenceTriples(BaseModel):
    sid: str
    decision: Literal["annotate", "skip"]
    justification: str
    T: list[SemanticTriple]


class SemanticTripleResult(BaseModel):
    S: list[SentenceTriples]


INSTRUCTION = """Use the supplied page summary as a retrieval guide and decide separately whether each sentence contains useful factual information for a triples database of Konbaung Dynasty history. Never convert the summary's interpretation into a triple unless the sentence itself supplies the evidence.

For every sentence selected for annotation, exhaustively extract all possible textual evidence that can support any factual statement in the page summary. Do not sample representative relations or stop after finding one example for a summary claim. Include every relation in the sentence that supports, exemplifies, explains, or supplies actors, actions, institutions, objects, locations, or material circumstances for the summary. The combined triples from the page should highlight all evidence connected to the summary.

Return every `sid` exactly once with a `decision`, `justification`, and `T`:
- Use `decision: "annotate"` when the sentence contains historically useful factual relations. Set `justification` to an empty string and return the relevant triples in `T`.
- Use `decision: "skip"` when the sentence is a fragment, an unusable continuation, purely formulaic or quoted material with no useful historical assertion, or otherwise contains no useful information for the historical database. Return `T: []` and give exactly one concise English sentence in `justification` explaining why it was skipped.

A quotation, letter, or reported speech is not automatically a historical fact. Distinguish the narrator's factual account that a document was sent, received, carried, or read from the propositions merely asserted inside the quoted document. Decide whether any quoted content is independently useful enough to annotate; otherwise skip it.

For every sentence selected for annotation, thoroughly tag every major historically meaningful verbal predicate and the subject and object or complement governed by it. Do not omit relevant embedded, coordinated, subordinate, quoted or reported, causative, passive, negative, modal, or shared-argument predicates. Return enough triples to capture the historically meaningful relations, but do not split incidental procedural details into separate triples.

Each triple has exactly these six fields:
- `s_my`: Burmese subject
- `s_en`: verbatim English gloss of the subject
- `p_my`: Burmese verbal predicate
- `p_en`: verbatim English gloss of the predicate
- `o_my`: Burmese object
- `o_en`: verbatim English gloss of the object or complement

When a subject, predicate, or object/complement is explicitly present, its Burmese value must be the exact contiguous span from that sentence. You may infer an omitted or grammatically implicit value when necessary for a coherent triple, for example passive constructions or clauses whose understood actor is the king. For an inferred value, write only the shortest appropriate Burmese referent and its English gloss; prefer spans that exist elsewhere on the page.

Pay close attention to Burmese grammatical order and clause structure. Before returning each triple, verify that its subject is actually the actor or grammatical subject of that predicate, that its object is governed by that predicate, and that all three values form a coherent proposition.

The Burmese triple statements must be grammatically well-formed, semantically coherent, and express a coherent argument, statement or proposition. All historically meaningful explicit arguments, statements, or propositions in each annotated sentence must be included.

The English glosses must be highly accurate, scholarly translations of their corresponding Burmese spans.

Preserve sentence order and return only the requested JSON."""


def read_page(volume: int, page_number: int) -> dict[str, Any]:
    path = DATA_ROOT / "pages" / f"vol{volume}" / f"{page_number:04d}.json"
    if not path.exists():
        raise FileNotFoundError(f"Unknown reader page: volume {volume}, page {page_number}")
    return json.loads(path.read_text(encoding="utf-8"))


def sentence_pairs(page: dict[str, Any]) -> list[dict[str, str]]:
    pairs = [
        {
            "sid": str(sentence["id"]),
            "my": str(sentence["text"]),
            "en": str(sentence["translation"]),
        }
        for sentence in page["sentences"]
    ]
    if not pairs or any(not pair["my"] or not pair["en"] for pair in pairs):
        raise ValueError("Selected page does not have complete Burmese sentence/translation pairs")
    return pairs


def build_prompt(summary: str, pairs: list[dict[str, str]]) -> str:
    return f"""{INSTRUCTION}

{json.dumps({"page_summary": summary, "sentences": pairs}, ensure_ascii=False, indent=2)}
"""


def validate(
    pairs: list[dict[str, str]],
    result: SemanticTripleResult,
) -> dict[str, Any]:
    errors: list[str] = []
    expected_ids = [pair["sid"] for pair in pairs]
    observed_ids = [group.sid for group in result.S]
    if observed_ids != expected_ids:
        errors.append(f"Sentence IDs/order mismatch: expected {expected_ids}, got {observed_ids}")
    pair_map = {pair["sid"]: pair for pair in pairs}
    triple_count = 0
    grounded_spans = 0
    inferred_spans = 0
    annotated_sentence_count = 0
    skipped_sentence_count = 0
    for group in result.S:
        pair = pair_map.get(group.sid)
        if pair is None:
            continue
        justification = group.justification.strip()
        if group.decision == "skip":
            skipped_sentence_count += 1
            if group.T:
                errors.append(f"{group.sid}: skipped sentence returned triples")
            if not justification:
                errors.append(f"{group.sid}: skipped sentence lacks a justification")
            elif "\n" in justification:
                errors.append(
                    f"{group.sid}: skip justification must be exactly one sentence"
                )
            continue
        annotated_sentence_count += 1
        if justification:
            errors.append(
                f"{group.sid}: annotated sentence must use an empty justification"
            )
        if not group.T:
            errors.append(f"{group.sid}: annotated sentence returned no triples")
        triple_count += len(group.T)
        for triple_index, triple in enumerate(group.T, start=1):
            for role in ("s", "p", "o"):
                burmese = str(getattr(triple, f"{role}_my")).strip()
                english = str(getattr(triple, f"{role}_en")).strip()
                if burmese.lower() in {"", "null", "none", "n/a", "-"}:
                    errors.append(
                        f"{group.sid} triple {triple_index} {role}: invalid Burmese value"
                    )
                elif span_candidates(pair["my"], burmese):
                    grounded_spans += 1
                else:
                    inferred_spans += 1
                if english.lower() in {"", "null", "none", "n/a", "-"}:
                    errors.append(
                        f"{group.sid} triple {triple_index} {role}: invalid English gloss"
                    )
    return {
        "accepted": not errors,
        "errors": errors,
        "sentenceCount": len(result.S),
        "annotatedSentenceCount": annotated_sentence_count,
        "skippedSentenceCount": skipped_sentence_count,
        "tripleCount": triple_count,
        "groundedSpanCount": grounded_spans,
        "inferredSpanCount": inferred_spans,
    }


def review_markdown(
    pairs: list[dict[str, str]],
    result: SemanticTripleResult,
    validation: dict[str, Any],
) -> str:
    groups = {group.sid: group for group in result.S}
    lines = [
        "# Minimal semantic triple trial",
        "",
        "VALID" if validation["accepted"] else "FLAGGED",
        "",
    ]
    for error in validation["errors"]:
        lines.append(f"- {error}")
    if validation["errors"]:
        lines.append("")
    for pair in pairs:
        lines.extend([f"## {pair['sid']}", "", pair["my"], "", pair["en"], ""])
        group = groups.get(pair["sid"])
        if group:
            lines.extend([f"**Decision:** {group.decision}", ""])
            if group.decision == "skip":
                lines.extend([f"**Justification:** {group.justification}", ""])
            else:
                for triple in group.T:
                    lines.extend(
                        [
                            f"- **S:** {triple.s_my} — {triple.s_en}",
                            f"  **P:** {triple.p_my} — {triple.p_en}",
                            f"  **O:** {triple.o_my} — {triple.o_en}",
                        ]
                    )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run(volume: int, page_number: int, output_name: str) -> Path:
    page = read_page(volume, page_number)
    pairs = sentence_pairs(page)
    summary = str(page.get("summary", "")).strip()
    prompt = build_prompt(summary, pairs)
    output_dir = OUTPUT_ROOT / f"vol{volume}" / f"page_{page_number:04d}" / output_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "input.json").write_text(
        json.dumps(
            {"page_summary": summary, "sentences": pairs},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SemanticTripleResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=16000,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW,
                include_thoughts=True,
            ),
        ),
    )
    answer_parts: list[str] = []
    thought_parts: list[str] = []
    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts or []:
            text = part.text or ""
            if not text:
                continue
            if part.thought:
                thought_parts.append(text)
            else:
                answer_parts.append(text)
    raw = "".join(answer_parts).strip() or response.text or ""
    thoughts = "\n\n".join(thought_parts).strip()
    (output_dir / "raw_response.json").write_text(raw, encoding="utf-8")
    (output_dir / "thoughts.md").write_text(
        ("# Gemini thought summary\n\n" + thoughts + "\n")
        if thoughts
        else "# Gemini thought summary\n\nNo thought summary was returned.\n",
        encoding="utf-8",
    )
    result = SemanticTripleResult.model_validate_json(raw)
    validation = validate(pairs, result)
    usage = (
        response.usage_metadata.model_dump(mode="json")
        if response.usage_metadata
        else {}
    )
    (output_dir / "result.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "volume": volume,
                "page": page_number,
                "usage": usage,
                "thoughts": thoughts,
                "validation": validation,
                "response": result.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "review.md").write_text(
        review_markdown(pairs, result, validation),
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int, required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--output-name", default="minimal_semantic_only_01")
    args = parser.parse_args()
    print(run(args.volume, args.page, args.output_name))


if __name__ == "__main__":
    main()

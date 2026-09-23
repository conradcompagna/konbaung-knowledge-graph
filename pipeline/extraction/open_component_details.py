#!/usr/bin/env python3
"""Gloss evidence blocks and open-code new details from masked chronicle pages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from konbaung_gemini_cite_sources_annotator import MODEL, ORIGINAL_PROMPT_PATH
from konbaung_gemini_summary_claim_completion_annotator import api_key
from konbaung_gemini_triple_metadata_semantic_frame import load_page, normalized


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "konbaung_open_component_details_tests"


class Detail(BaseModel):
    tag: str = Field(description="Concise free-form metadata tag.")
    my: str = Field(description="Exact Burmese span from TARGET_PAGE_TEXT.")
    en: str = Field(description="Short English gloss.")


class PredicatePart(BaseModel):
    details: list[Detail] = Field(description="New open-coded predicate details only.")


class TripleDetails(BaseModel):
    id: str = Field(description="Exact existing triple ID.")
    e: str = Field(description="Concise English gloss or summary of the existing evidence block.")
    s: list[Detail] = Field(description="New subject details only.")
    p: PredicatePart
    o: list[Detail] = Field(description="New object details only.")


class OpenDetailsResult(BaseModel):
    D: list[TripleDetails]


DETAILS_PROMPT = """<OPEN_DETAILS_TASK>
This is a new evidence-glossing and open-coding enrichment task. The complete first-pass prompt above is read-only context explaining the historical research question and the process that created the existing triples. Read it thoroughly to understand the original task, but follow only the instructions in this block.

The existing triples already identify a subject, an analytical predicate tag, and an object. Their redundant Burmese anchor fields have been omitted from the triple records; only their analytical tags and English glosses are supplied. Each triple also includes its complete unredacted evidence excerpt. For every supplied triple, write a concise English gloss of that complete evidence excerpt. When an evidence excerpt is long, condense it into a shorter faithful summary preserving the relation and historically meaningful supporting information. Do not return a Burmese predicate span or a separate predicate gloss.

The page and evidence excerpts are unredacted. Siphon only genuinely new Burmese information into the database rather than repeating information already represented by the supplied triple tags and English glosses.

For each triple, return new details in three component lists:

- `s` contains additional composition, identity, status, membership, material, or attributes of the subject.
- `p.details` contains modifiers of the relation: time/date, place, manner, reason, purpose, instrument, means, procedure, route, authority, condition, or result.
- `o` contains additional composition, identity, status, membership, material, or attributes of the object.

You must discern whether or not your additional details are modifiers on the subject, object, or predicate. Any details modifying the subject or object must go in their details column. Only details that modify the event as a whole go in the predicate column: where or when something happened, why it happened, the method through which it happened, and so on. Do not put material related to the agent or patient/theme of the action in the predicate details; put them in the subject or object column.

For every optional detail, provide a concise free-form `tag`, an exact Burmese span in `my`, and a compact English gloss in `en`. For any additional time or place data you find, reuse the existing `time` or `place` tag. For numerical modifiers, use `quantity`.

Other common tags may include `manner`, `reason`, `purpose`, `instrument`, `means`, `procedure`, `route`, `authority`, `condition`, and `result` for predicates, and `composition`, `membership`, `identity`, `status`, `material`, `contents`, `attire`, `role`, `parts`, and `attributes` for subjects or objects. These are suggestions for consistency, not a closed vocabulary: this remains an open-coding task, and you may create a concise tag when the evidence requires another category.

A detail should recover information visible in the page but absent from the current triple. Lists and long noun phrases are important: if a broad subject or object gloss compresses a historically meaningful list of people, offices, gifts, resources, ritual objects, military units, or other constituents, record the omitted constituents as one or more details on that subject or object.

Return every supplied triple exactly once and in the supplied order. For each one, return only its ID, the English evidence gloss in `e`, and genuinely new details in `s`, `p.details`, and `o`. Do not reproduce the existing subject, object, analytical predicate tag, existing time/place details, or any Burmese predicate span. Empty detail lists are correct when no additional information remains visible.

The following three examples are drawn from the chronicles.

Example 1: evidence glossing and predicate enrichment.

The existing triple says that the sacred objects `ဓာတ်တော်မွေတော် ဆင်းတုတော်များ` are installed on the second terrace `ဒုတိယ ပစ္စယာ` through `IS_INSTALLED_ON`. It already contains the place detail `စေတီတော်`. These three Burmese spans have already been siphoned into the triple. Do not annotate `ဓာတ်တော်မွေတော် ဆင်းတုတော်များ`, `ဒုတိယ ပစ္စယာ`, `စေတီတော်`, or any shorter or longer version containing them.

The surrounding text is: `စေတီတော်သို့ ရောက်လျှင် ဝေါဗလာတော်ကို တန်ဆောင်းတွင် ချ၍ ဓာတ်တော်မွေတော် ဆင်းတုတော်များကို ရိုသေစွာ ပင့်ဆောင်မှ ဒုတိယ ပစ္စယာပေါ်တွင် စင်ရှင်နှင့် တံခွန် ကုက္ကား မုလေးပွား စီစဉ် စိုက်ဆောက်လျက် ထားရသည်`.

Gloss the evidence as “The sacred objects were reverently installed on the second terrace.” Return `ရိုသေစွာ`, “reverently,” in `p.details` with the tag `manner`. Do not return the original subject, object, analytical predicate tag, place detail, or a Burmese predicate span.

Example 2: subject enrichment.

The existing triple says that the subject `ရေတပ်`, “naval fleet,” is deployed in the object `ကျုံးတော်`, “palace moat,” through `IS_DEPLOYED_IN`. It already contains `ကျုံးတော်` as a place detail. The spans `ရေတပ်` and `ကျုံးတော်` have already been siphoned into the triple. Do not annotate them again or use longer spans containing them.

The surrounding text is: `လှော်ကားတော်, ခတ်လှေတော်, လှော်လှေတော်, ရဲလှေတော်, သင်္ဘောသမ္ဗန်တော်, မင်းညီမင်းသား, မှူးတော်မတ်တော်တို့ စီးနင်းရန် လှော်ကားများကို ကျုံးတော်အတွင်း ရေတပ်ခင်းကျင်းရန်`.

Gloss the evidence as “The naval fleet, composed of royal barges and several classes of boats, was to be deployed inside the palace moat.” Return the new Burmese vessel list in `s` with the tag `composition`. Do not return the original subject, object, analytical predicate tag, or place detail.

Example 3: object enrichment.

The existing triple says that the subject `မင်းတရားကြီး`, “the great king,” bestows the object `ကွမ်းခွက်, လက်ဖက်အိုး, တကောင်းပိတ်`, “betel boxes, tea pots, and betel-leaf containers,” through `BESTOWS_CEREMONIAL_EQUIPMENT_AND_REGALIA`. The subject and object spans have already been siphoned into the triple. Do not annotate those spans again or use longer spans containing them.

The surrounding text is: `မျဉ်းရှည် ကတ္တီပါနီ ယက်ပြားသတ်၊ ကွမ်းခွက်, လက်ဖက်အိုး, တကောင်းပိတ် ရှစ်ထောင့် ပယင်းတပ်၊ ကွမ်းလိပ် သုံးမြွှာ ပယင်းတပ်`.

Gloss the evidence as a concise English statement of the bestowal and its ornamented objects. Return `ရှစ်ထောင့် ပယင်းတပ်`, “eight-sided amber inlay,” in `o` with the tag `material`. Do not return the existing object span or any Burmese predicate span.
</OPEN_DETAILS_TASK>"""


def research_prefix() -> str:
    return ORIGINAL_PROMPT_PATH.read_text(encoding="utf-8-sig").rstrip()


def input_triples(page: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for annotation in page["annotations"]:
        predicate_details = []
        metadata = annotation.get("metadata", {})
        for key, source_key in (("time", "date"), ("place", "location")):
            value = metadata.get(source_key, {})
            if value.get("originalText"):
                predicate_details.append(
                    {"tag": key, "my": value["originalText"], "en": value.get("gloss") or ""}
                )
        rows.append(
            {
                "id": annotation["id"],
                "e": annotation["evidence"]["canonicalText"],
                "s": {
                    "tag": annotation["subject"].get("type") or "",
                    "my": annotation["subject"]["text"],
                    "en": annotation["subject"].get("gloss") or "",
                    "details": [],
                },
                "p": {
                    "tag": annotation["relation"]["rawLabel"],
                    "details": predicate_details,
                },
                "o": {
                    "tag": annotation["object"].get("type") or "",
                    "my": annotation["object"]["text"],
                    "en": annotation["object"].get("gloss") or "",
                    "details": [],
                },
            }
        )
    return rows


def prompt_triples(triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": triple["id"],
            "e": triple["e"],
            "s": {"tag": triple["s"]["tag"], "en": triple["s"]["en"]},
            "p": {
                "tag": triple["p"]["tag"],
                "details": [
                    {"tag": detail["tag"], "en": detail["en"]} for detail in triple["p"]["details"]
                ],
            },
            "o": {"tag": triple["o"]["tag"], "en": triple["o"]["en"]},
        }
        for triple in triples
    ]


def build_prompt(volume: int, page_number: int) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    page = load_page(volume, page_number)
    triples = input_triples(page)
    page_text = page["canonicalText"]
    serialized_triples = prompt_triples(triples)
    prompt = f"""<ORIGINAL_FIRST_PASS_PROMPT_READ_ONLY>
The text inside this block is the complete original prompt that created the first-pass summaries and triples. It is historical and methodological context only. Do not execute its original output instructions; the current task begins after this block.

{research_prefix()}
</ORIGINAL_FIRST_PASS_PROMPT_READ_ONLY>

{DETAILS_PROMPT}

<TARGET_PAGE_TEXT>
{page_text}
</TARGET_PAGE_TEXT>

<EXISTING_SUMMARY_READ_ONLY>
{page.get("summary") or ""}
</EXISTING_SUMMARY_READ_ONLY>

<EXISTING_TRIPLES_READ_ONLY>
{json.dumps(serialized_triples, ensure_ascii=False, indent=2)}
</EXISTING_TRIPLES_READ_ONLY>
"""
    return prompt, page, triples


def validate(
    result: OpenDetailsResult, page: dict[str, Any], triples: list[dict[str, Any]]
) -> list[str]:
    flags: list[str] = []
    source = {row["id"]: row for row in triples}
    visible_target = normalized(page["canonicalText"])
    returned_ids = [row.id for row in result.D]
    for triple_id in source:
        if triple_id not in returned_ids:
            flags.append(f"missing triple ID {triple_id}")
    for triple_id in set(returned_ids):
        if returned_ids.count(triple_id) > 1:
            flags.append(f"duplicate triple ID {triple_id}")
    for index, row in enumerate(result.D, 1):
        triple = source.get(row.id)
        if triple is None:
            flags.append(f"D[{index}]: unknown triple ID {row.id}")
            continue
        if not row.e.strip():
            flags.append(f"D[{index}]: evidence gloss is empty")

        existing_spans = {
            normalized(triple["s"]["my"]),
            normalized(triple["o"]["my"]),
            *(normalized(item["my"]) for item in triple["p"]["details"]),
        }
        for part_name, details in (("s", row.s), ("p", row.p.details), ("o", row.o)):
            for detail in details:
                span = normalized(detail.my)
                if span not in visible_target:
                    flags.append(f"D[{index}]: {part_name} detail span absent from page")
                reuses_existing = any(
                    span == existing or span in existing or existing in span
                    for existing in existing_spans
                    if existing
                )
                if reuses_existing:
                    flags.append(f"D[{index}]: {part_name} detail reuses an existing span")
    return flags


def new_predicate_details(row: TripleDetails, triple: dict[str, Any]) -> list[Detail]:
    return row.p.details


def write_review(
    output_path: Path,
    page: dict[str, Any],
    triples: list[dict[str, Any]],
    result: OpenDetailsResult,
) -> None:
    source = {row["id"]: row for row in triples}
    lines = ["# Page summary", "", page.get("summary") or "", "", "# Enriched triples", ""]
    for addition in result.D:
        triple = source.get(addition.id)
        if triple is None:
            continue
        p_new = new_predicate_details(addition, triple)
        lines.extend(
            [
                f"## {addition.id}",
                "",
                f"**Evidence:** {triple['e']}",
                "",
                f"**Evidence gloss:** {addition.e}",
                "",
            ]
        )
        for title, value, details in (
            (
                "Subject",
                f"{triple['s']['my']} | {triple['s']['tag']} | {triple['s']['en']}",
                addition.s,
            ),
            (
                "Predicate",
                triple["p"]["tag"],
                p_new,
            ),
            (
                "Object",
                f"{triple['o']['my']} | {triple['o']['tag']} | {triple['o']['en']}",
                addition.o,
            ),
        ):
            lines.extend([f"**{title}:** {value}", "", "**New details:**"])
            if details:
                for detail in details:
                    lines.append(f"- **{detail.tag}:** {detail.my} | {detail.en}")
            else:
                lines.append("- None")
            lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run(volume: int, page_number: int, output_name: str) -> Path:
    prompt, page, triples = build_prompt(volume, page_number)
    output_dir = OUTPUT_ROOT / f"vol{volume}" / f"page_{page_number:04d}" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_sent.txt").write_text(prompt, encoding="utf-8")
    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=OpenDetailsResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=6000,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = response.text or ""
    (output_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
    flags: list[str] = []
    try:
        result = OpenDetailsResult.model_validate_json(raw)
        flags.extend(validate(result, page, triples))
        parsed: dict[str, Any] | str = result.model_dump()
        write_review(output_dir / "review.md", page, triples, result)
    except Exception as exc:
        flags.append(f"structured parse failed: {exc}")
        parsed = raw
    payload = {
        "accepted": True,
        "flagged": bool(flags),
        "flags": flags,
        "usage": response.usage_metadata.model_dump(mode="json") if response.usage_metadata else {},
        "result": parsed,
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int, required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--output-name", default="open_details_01")
    args = parser.parse_args()
    print(run(args.volume, args.page, args.output_name))


if __name__ == "__main__":
    main()

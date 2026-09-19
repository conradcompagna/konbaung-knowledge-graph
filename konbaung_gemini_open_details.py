#!/usr/bin/env python3
"""Open-code additional details for existing Burmese chronicle triples."""

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
OUTPUT_ROOT = ROOT / "konbaung_open_details_tests"


class Detail(BaseModel):
    tag: str = Field(description="Concise free-form metadata tag.")
    my: str = Field(description="Exact Burmese span from TARGET_PAGE_TEXT.")
    en: str = Field(description="Short English gloss.")


class TriplePart(BaseModel):
    tag: str = Field(description="Analytical tag for this triple component.")
    my: str = Field(description="Exact Burmese span for this triple component.")
    en: str = Field(description="Short English gloss for this triple component.")
    details: list[Detail] = Field(description="Open-coded details attached to this component.")


class TripleDetails(BaseModel):
    id: str = Field(description="Exact existing triple ID.")
    s: TriplePart
    p: TriplePart
    o: TriplePart


class OpenDetailsResult(BaseModel):
    D: list[TripleDetails]


DETAILS_PROMPT = """<OPEN_DETAILS_TASK>
This is a new predicate-grounding and open-coding enrichment task. The complete first-pass prompt above is read-only context explaining the historical research question and the process that created the existing triples. Follow only the instructions in this block.

The existing triples already identify a subject, an analytical predicate tag, and an object. Subjects and objects have Burmese spans, English glosses, and open-coded tags. Predicates currently have only an analytical tag. Your first task is to complete every triple by identifying the exact Burmese predicate expression that links its subject and object and by glossing that predicate in English.

Each subject, predicate, and object also has a `details` list. Use these lists for open-coded enrichment:

- `s.details` describes additional composition, identity, status, membership, material, or attributes of the subject.
- `p.details` describes modifiers of the relation: time/date, place, manner, reason, purpose, instrument, means, procedure, route, authority, condition, result, or another useful open-coded dimension.
- `o.details` describes additional composition, identity, status, membership, material, or attributes of the object.

For every optional detail, provide a concise free-form `tag`, an exact Burmese span in `my`, and a compact English gloss in `en`. A detail should recover information that is present in the page but absent from the current triple. Lists and long noun phrases are important: if a broad subject or object gloss compresses a historically meaningful list of people, offices, gifts, resources, ritual objects, military units, or other constituents, record the omitted constituents as one or more details on that subject or object.

The following before-and-after examples are drawn from the chronicles.

Example 1 - grounding a predicate while preserving its existing time and place details

Before:
`s = {tag: RoyalCommandTechnology, my: အမိန့်တော်, en: royal order, details: []}`
`p = {tag: COMMANDS_MOBILIZATION_OF, details: [{tag: time, my: သက္ကရာဇ် (၁၁၁၉) တစ်ထောင့်တစ်ရာ့တစ်ဆယ့်ကိုးခု၊ ကဆုန်လပြည့်ကျော် လေးရက် သောကြာနေ့, en: Friday, 4th waning of Kason, 1119 BE}, {tag: place, my: ပဲကူးမြို့, en: Pegu city}]}`
`o = {tag: MilitaryForceScale, my: လူပေါင်း တစ်ရာ့ကိုးကျိပ်သုံးယောက်, en: 193 men, details: []}`

Chronicle text:
`လူပေါင်း တစ်ရာ့ကိုးကျိပ်သုံးယောက် ရွေးကောက်တော်မူပြီးလျှင် သက္ကရာဇ် (၁၁၁၉) တစ်ထောင့်တစ်ရာ့တစ်ဆယ့်ကိုးခု၊ ကဆုန်လပြည့်ကျော် လေးရက် သောကြာနေ့ ညနေ လေးချက်တီးကျော် လမပေါ်မီ ပဲကူးမြို့တွင်းသို့ ဝင်၍ မီးထင်းတိုက်ရှို့ ရမည်၊ တံခါးကိုလည်း ဖွင့်ရမည်ဟု ခန့်တော်မူသည်`

After:
`s = {tag: RoyalCommandTechnology, my: အမိန့်တော်, en: royal order, details: []}`
`p = {tag: COMMANDS_MOBILIZATION_OF, my: ခန့်တော်မူသည်, en: ordered and appointed, details: [{tag: time, my: သက္ကရာဇ် (၁၁၁၉) တစ်ထောင့်တစ်ရာ့တစ်ဆယ့်ကိုးခု၊ ကဆုန်လပြည့်ကျော် လေးရက် သောကြာနေ့, en: Friday, 4th waning of Kason, 1119 BE}, {tag: place, my: ပဲကူးမြို့, en: Pegu city}]}`
`o = {tag: MilitaryForceScale, my: လူပေါင်း တစ်ရာ့ကိုးကျိပ်သုံးယောက်, en: 193 men, details: []}`

Example 2 - adding a new predicate detail after grounding the predicate

Before:
`s = {tag: SacredObjects, my: ဓာတ်တော်မွေတော် ဆင်းတုတော်များ, en: relics, bone fragments, and images, details: []}`
`p = {tag: IS_INSTALLED_ON, details: [{tag: place, my: စေတီတော်, en: the stupa}]}`
`o = {tag: StupaArchitecturalFeature, my: ဒုတိယ ပစ္စယာ, en: the second terrace, details: []}`

Chronicle text:
`ဓာတ်တော်မွေတော် ဆင်းတုတော်များကို ရိုသေစွာ ပင့်ဆောင်မှ ဒုတိယ ပစ္စယာပေါ်တွင် စင်ရှင်နှင့် တံခွန် ကုက္ကား မုလေးပွား စီစဉ် စိုက်ဆောက်လျက် ထားရသည်`

After:
`s = {tag: SacredObjects, my: ဓာတ်တော်မွေတော် ဆင်းတုတော်များ, en: relics, bone fragments, and images, details: []}`
`p = {tag: IS_INSTALLED_ON, my: ထားရသည်, en: were installed, details: [{tag: place, my: စေတီတော်, en: the stupa}, {tag: ritual_manner, my: ရိုသေစွာ, en: reverently}]}`
`o = {tag: StupaArchitecturalFeature, my: ဒုတိယ ပစ္စယာ, en: the second terrace, details: []}`

Example 3 - expanding a collective subject

Before:
`s = {tag: MilitaryForceComposition, my: ရေတပ်, en: naval fleet, details: []}`
`p = {tag: IS_DEPLOYED_IN, details: [{tag: place, my: ကျုံးတော်, en: palace moat}]}`
`o = {tag: RoyalRitualSpace, my: ကျုံးတော်, en: palace moat, details: []}`

Chronicle text:
`လှော်ကားတော်, ခတ်လှေတော်, လှော်လှေတော်, ရဲလှေတော်, သင်္ဘောသမ္ဗန်တော်, မင်းညီမင်းသား, မှူးတော်မတ်တော်တို့ စီးနင်းရန် လှော်ကားများကို ကျုံးတော်အတွင်း ရေတပ်ခင်းကျင်းရန်`

After:
`s = {tag: MilitaryForceComposition, my: ရေတပ်, en: naval fleet, details: [{tag: vessel_composition, my: လှော်ကားတော်, ခတ်လှေတော်, လှော်လှေတော်, ရဲလှေတော်, သင်္ဘောသမ္ဗန်တော်, en: royal barges, paddling boats, rowing boats, guard boats, ships, and sampans}]}`
`p = {tag: IS_DEPLOYED_IN, my: ကျုံးတော်အတွင်း ရေတပ်ခင်းကျင်းရန်, en: to deploy the naval force inside the moat, details: [{tag: place, my: ကျုံးတော်, en: palace moat}]}`
`o = {tag: RoyalRitualSpace, my: ကျုံးတော်, en: palace moat, details: []}`

Example 4 - expanding an object and adding a predicate modifier

Before:
`s = {tag: RoyalSacredVessel, my: ကရဝိက်ရွှေဖောင်တော်, en: the Karaweik royal golden barge, details: []}`
`p = {tag: CARRIES_IN_PROCESSION, details: []}`
`o = {tag: Monarch, my: အသျှင် ဘဝရှင်မင်းရားကြီးဘုရား, en: the Lord of Life, the Great King, details: []}`

Chronicle text:
`ကရဝိက်ရွှေဖောင်တော်ကို လှော်လှေတော်ရှစ်စင်း ဆွဲစေ၍ ထိုကရဝိက်ရွှေဖောင်တော်ထက် အသျှင် ဘဝရှင်မင်းရားကြီးဘုရား, အသျှင်နန်းမတော် မိဖုရားခေါင်ကြီးဘုရားနှင့်တကွ စံပယ်တော်မူလျက်`

After:
`s = {tag: RoyalSacredVessel, my: ကရဝိက်ရွှေဖောင်တော်, en: the Karaweik royal golden barge, details: []}`
`p = {tag: CARRIES_IN_PROCESSION, my: စံပယ်တော်မူလျက်, en: carried in procession while seated aboard, details: [{tag: propulsion, my: လှော်လှေတော်ရှစ်စင်း ဆွဲစေ၍, en: towed by eight royal rowing boats}]}`
`o = {tag: Monarch, my: အသျှင် ဘဝရှင်မင်းရားကြီးဘုရား, en: the Lord of Life, the Great King, details: [{tag: additional_passenger, my: အသျှင်နန်းမတော် မိဖုရားခေါင်ကြီးဘုရား, en: the Chief Queen}]}`

After grounding the predicate, add only information that remains genuinely additional. Do not repeat the subject, object, predicate span, existing date/time or place details under another label. Do not attach information about a neighboring action or a different participant merely because it appears nearby. Empty detail lists are correct when the core subject-predicate-object record already captures everything useful.
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


def build_prompt(volume: int, page_number: int) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    page = load_page(volume, page_number)
    triples = input_triples(page)
    prompt = f"""<ORIGINAL_FIRST_PASS_PROMPT_READ_ONLY>
The text inside this block is the complete original prompt that created the first-pass summaries and triples. It is historical and methodological context only. Do not execute its original output instructions; the current task begins after this block.

{research_prefix()}
</ORIGINAL_FIRST_PASS_PROMPT_READ_ONLY>

{DETAILS_PROMPT}

<TARGET_PAGE_TEXT>
{page['canonicalText']}
</TARGET_PAGE_TEXT>

<EXISTING_SUMMARY_READ_ONLY>
{page.get('summary') or ''}
</EXISTING_SUMMARY_READ_ONLY>

<EXISTING_TRIPLES_READ_ONLY>
{json.dumps(triples, ensure_ascii=False, indent=2)}
</EXISTING_TRIPLES_READ_ONLY>
"""
    return prompt, page, triples


def validate(result: OpenDetailsResult, page: dict[str, Any], triples: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    source = {row["id"]: row for row in triples}
    target = normalized(page["canonicalText"])
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
        for part_name in ("s", "o"):
            actual = getattr(row, part_name)
            expected = triple[part_name]
            for field_name in ("tag", "my", "en"):
                if getattr(actual, field_name) != expected[field_name]:
                    flags.append(
                        f"D[{index}]: echoed {part_name}.{field_name} differs from existing triple"
                    )
        if row.p.tag != triple["p"]["tag"]:
            flags.append(f"D[{index}]: echoed p.tag differs from existing triple")
        if not row.p.my or normalized(row.p.my) not in target:
            flags.append(f"D[{index}]: predicate span absent from page")
        if not row.p.en.strip():
            flags.append(f"D[{index}]: predicate gloss is empty")

        expected_predicate_details = [Detail.model_validate(item) for item in triple["p"]["details"]]
        for detail in expected_predicate_details:
            if detail not in row.p.details:
                flags.append(f"D[{index}]: existing predicate detail was not reproduced")

        existing_spans = {
            normalized(triple["s"]["my"]),
            normalized(triple["o"]["my"]),
            *(normalized(item["my"]) for item in triple["p"]["details"]),
        }
        for part_name in ("s", "p", "o"):
            part = getattr(row, part_name)
            for detail in part.details:
                span = normalized(detail.my)
                is_existing = part_name == "p" and detail in expected_predicate_details
                if span not in target:
                    flags.append(f"D[{index}]: {part_name} detail span absent from page")
                if detail.tag.strip().lower() == "quantity":
                    flags.append(f"D[{index}]: quantity detail returned despite task exclusion")
                if not is_existing and span in existing_spans:
                    flags.append(f"D[{index}]: {part_name} detail repeats an existing span")
    return flags


def new_predicate_details(row: TripleDetails, triple: dict[str, Any]) -> list[Detail]:
    existing = [Detail.model_validate(item) for item in triple["p"]["details"]]
    return [detail for detail in row.p.details if detail not in existing]


def write_review(
    output_path: Path,
    page: dict[str, Any],
    triples: list[dict[str, Any]],
    result: OpenDetailsResult,
) -> None:
    source = {row["id"]: row for row in triples}
    lines = ["# Page summary", "", page.get("summary") or "", "", "# Completed triples", ""]
    for addition in result.D:
        triple = source.get(addition.id)
        if triple is None:
            continue
        p_new = new_predicate_details(addition, triple)
        lines.extend(
            [
                f"## {addition.id}",
                "",
                f"**S:** {addition.s.my} | {addition.s.tag} | {addition.s.en}  ",
                f"**P:** {addition.p.my} | {addition.p.tag} | {addition.p.en}  ",
                f"**O:** {addition.o.my} | {addition.o.tag} | {addition.o.en}",
                "",
            ]
        )
        additions = [("S", item) for item in addition.s.details]
        additions.extend(("P", item) for item in p_new)
        additions.extend(("O", item) for item in addition.o.details)
        if additions:
            lines.extend(["**New details**", ""])
            for part_name, detail in additions:
                lines.append(f"- **{part_name} / {detail.tag}:** {detail.my} | {detail.en}")
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

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "konbaung_viable_pages_manual_ranges"
ANNOTATION_ROOT = ROOT / "konbaung_open_coding_structured_full_batch_20260709" / "postprocessed" / "valid"
OUTPUT_ROOT = ROOT / "konbaung_cite_sources_page_test"
ORIGINAL_PROMPT_PATH = ROOT / "konbaung_open_coding_structured_full_batch_20260709" / "cached_prefix.txt"
ENV_FILE = Path(__file__).resolve().parent / ".env"
MODEL = os.getenv("KONBAUNG_CITE_MODEL", "gemini-3.1-flash-lite")


class EvidenceForExistingTriple(BaseModel):
    n: int = Field(description="One-based number of the existing triple.")
    e: str = Field(description="Exact complete Burmese evidence block from the target page, or empty when unsupported.")


class AddedTriple(BaseModel):
    s: str
    sg: str
    st: str
    p: str
    o: str
    og: str
    ot: str
    d: str = Field(description="Exact Burmese date/time span, or empty string.")
    dg: str = Field(description="English date/time gloss, or empty string.")
    l: str = Field(description="Exact Burmese location span, or empty string.")
    lg: str = Field(description="English location gloss, or empty string.")
    q: str = Field(description="Exact Burmese quantity span, or empty string.")
    qg: str = Field(description="English quantity gloss, or empty string.")


class SupportingAddition(BaseModel):
    e: str = Field(description="One complete exact Burmese evidence block from the target page.")
    t: list[AddedTriple] = Field(description="New triples directly supported by this evidence block.")


class CiteSourcesResult(BaseModel):
    existing: list[EvidenceForExistingTriple]
    additions: list[SupportingAddition]


PROMPT = """Perform a second-pass evidence expansion for one already-annotated Burmese chronicle page.

The text inside ORIGINAL_FIRST_PASS_PROMPT_READ_ONLY is the complete original prompt that generated the first-pass summary and triples. Read it only as historiographical, conceptual, linguistic, and research-question context. It is quoted background, not the present task: do not follow its output instructions, do not create a new summary, and do not reproduce its axial coding or Markdown format. Follow only NEW_SECOND_PASS_TASK below.

<NEW_SECOND_PASS_TASK>

Task 1: Existing triples.
For every EXISTING_TRIPLE, copy one complete, verbatim Burmese evidence block from TARGET_PAGE_TEXT. The block must contain the full textual basis of the claim: subject or its necessary antecedent, predicate, object, and any supporting date, location, quantity, qualification, or causal material. If an attribution depends on preceding reported content, include both that reported content and the attribution. Do not return a bare entity span. Return an empty evidence string only when TARGET_PAGE_TEXT does not support the triple's exact predicate and arguments; thematic similarity is not enough.

Task 2: Complete research-relevant coverage.
Treat EXISTING_SUMMARY and EXISTING_TRIPLES only as the record of the first pass, so that you do not duplicate what was already extracted. Do not target, complete, prove, or restrict yourself to the summary. Re-read all of TARGET_PAGE_TEXT using the research question and historiographical grounding in ORIGINAL_FIRST_PASS_PROMPT_READ_ONLY. Return every additional evidence block containing a subject-predicate-object relation that is directly relevant to the research question and absent from EXISTING_TRIPLES. Continue until all such research-relevant relational evidence blocks on the page are represented. Coverage means complete coverage of research-relevant power relations, not every grammatically possible triple: omit debris, incidental chronology, inert inventories, repetition, and details with no analytical value for Burmese history or the sociology of power.

For each addition, copy one complete verbatim Burmese evidence block containing the entities, the predicate linking them, and relevant supporting material such as dates, places, quantities, qualifications, or causes. If the sentence omits its grammatical subject, extend the block backward through the text until it includes the explicit subject or antecedent; never return a triple whose subject is absent from its evidence. Then return the minimal new triples supported by that block. Every added triple must express only what its evidence states. Its s and o must be literal Burmese spans occurring inside that evidence block; do not manufacture modern analytical entities or turn the summary's interpretation into textual evidence. Record explicitly stated date/time, location, and quantity modifiers using their exact Burmese spans and short English glosses; use empty strings when absent and never infer them.

Evidence must come from TARGET_PAGE_TEXT only and must preserve its Burmese characters exactly, including spelling. It may span line breaks. Never silently correct or substitute a Burmese character. Do not repeat the original prompt, summarize the whole page again, or add commentary outside the structured result.
</NEW_SECOND_PASS_TASK>
"""


def read_env_value(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        current, value = line.split("=", 1)
        if current.strip() == key:
            return value.strip().strip("\"'")
    return ""


def api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip() or read_env_value(ENV_FILE, "GEMINI_API_KEY")


def source_path(volume: int, page: int) -> Path:
    return SOURCE_ROOT / f"konbaung_vol{volume}" / "pages" / f"page_{page:04d}.txt"


def annotation_path(volume: int, page: int) -> Path:
    filename = f"page_{page:04d}.json"
    valid = ANNOTATION_ROOT / f"konbaung_vol{volume}" / filename
    if valid.exists():
        return valid
    return ANNOTATION_ROOT.parent / "invalid" / f"konbaung_vol{volume}" / filename


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", "", text)


def evidence_is_exact(page_text: str, evidence: str) -> bool:
    if not evidence:
        return False
    return compact_whitespace(evidence) in compact_whitespace(page_text)


def build_payload(page_text: str, annotation: dict[str, Any]) -> str:
    original_prompt = ORIGINAL_PROMPT_PATH.read_text(encoding="utf-8-sig")
    triples = [
        {
            "n": index,
            "s": triple.get("s", ""),
            "sg": triple.get("sg", ""),
            "st": triple.get("st", ""),
            "p": triple.get("p", ""),
            "o": triple.get("o", ""),
            "og": triple.get("og", ""),
            "ot": triple.get("ot", ""),
            "d": triple.get("d", ""),
            "dg": triple.get("dg", ""),
            "l": triple.get("l", ""),
            "lg": triple.get("lg", ""),
            "q": triple.get("q", ""),
            "qg": triple.get("qg", ""),
        }
        for index, triple in enumerate(annotation.get("T", []), start=1)
    ]
    return f"""<ORIGINAL_FIRST_PASS_PROMPT_READ_ONLY>
{original_prompt}
</ORIGINAL_FIRST_PASS_PROMPT_READ_ONLY>

{PROMPT}

<TARGET_PAGE_TEXT>
{page_text}
</TARGET_PAGE_TEXT>

<EXISTING_SUMMARY>
{annotation.get('summary', '')}
</EXISTING_SUMMARY>

<EXISTING_TRIPLES>
{json.dumps(triples, ensure_ascii=False, indent=2)}
</EXISTING_TRIPLES>
"""


def validate_result(result: CiteSourcesResult, page_text: str, triple_count: int) -> list[str]:
    warnings: list[str] = []
    returned_numbers = [item.n for item in result.existing]
    expected_numbers = list(range(1, triple_count + 1))
    if returned_numbers != expected_numbers:
        warnings.append(f"existing triple numbers {returned_numbers} != {expected_numbers}")
    for item in result.existing:
        if item.e and not evidence_is_exact(page_text, item.e):
            warnings.append(f"existing {item.n}: evidence is not a whitespace-normalized page substring")
    for addition_index, addition in enumerate(result.additions, start=1):
        if not evidence_is_exact(page_text, addition.e):
            warnings.append(f"addition {addition_index}: evidence is not a page substring")
        normalized_evidence = compact_whitespace(addition.e)
        for triple_index, triple in enumerate(addition.t, start=1):
            if compact_whitespace(triple.s) not in normalized_evidence:
                warnings.append(f"addition {addition_index} triple {triple_index}: subject is not in cited evidence")
            if compact_whitespace(triple.o) not in normalized_evidence:
                warnings.append(f"addition {addition_index} triple {triple_index}: object is not in cited evidence")
            for field_name in ("d", "l", "q"):
                value = getattr(triple, field_name)
                if value and compact_whitespace(value) not in normalized_evidence:
                    warnings.append(
                        f"addition {addition_index} triple {triple_index}: {field_name} is not in cited evidence"
                    )
    return warnings


def markdown_review(
    page_text: str,
    annotation: dict[str, Any],
    result: CiteSourcesResult,
    warnings: list[str],
    usage: dict[str, Any],
) -> str:
    lines = ["# Cite-your-sources review", "", "## Existing summary", "", annotation.get("summary", ""), ""]
    lines.extend(["## Existing triples and expanded evidence", ""])
    triples = annotation.get("T", [])
    for item in result.existing:
        original = triples[item.n - 1] if 0 < item.n <= len(triples) else {}
        lines.extend(
            [
                f"### {item.n}. {original.get('sg', original.get('s', ''))} | {original.get('p', '')} | {original.get('og', original.get('o', ''))}",
                "",
                "```text",
                item.e,
                "```",
                "",
            ]
        )
    lines.extend(["## Additional summary-supporting evidence", ""])
    if not result.additions:
        lines.extend(["None identified.", ""])
    for index, addition in enumerate(result.additions, start=1):
        lines.extend([f"### Addition {index}", "", "```text", addition.e, "```", ""])
        for triple in addition.t:
            lines.append(f"- {triple.sg} [{triple.st}] | {triple.p} | {triple.og} [{triple.ot}]")
            modifiers = []
            if triple.d:
                modifiers.append(f"date: {triple.d} ({triple.dg})")
            if triple.l:
                modifiers.append(f"location: {triple.l} ({triple.lg})")
            if triple.q:
                modifiers.append(f"quantity: {triple.q} ({triple.qg})")
            if modifiers:
                lines.append(f"  - {'; '.join(modifiers)}")
        lines.append("")
    lines.extend(["## Validation", ""])
    lines.extend([f"- {warning}" for warning in warnings] or ["- All evidence blocks resolved to the page text."])
    lines.extend(["", "## Usage", "", "```json", json.dumps(usage, ensure_ascii=False, indent=2), "```", "", "## Original page", "", "```text", page_text, "```", ""])
    return "\n".join(lines)


def run(volume: int, page: int, output_name: str = "") -> Path:
    page_file = source_path(volume, page)
    annotation_file = annotation_path(volume, page)
    page_text = page_file.read_text(encoding="utf-8-sig")
    annotation = json.loads(annotation_file.read_text(encoding="utf-8"))
    prompt = build_payload(page_text, annotation)

    output_dir = OUTPUT_ROOT / f"konbaung_vol{volume}" / f"page_{page:04d}"
    if output_name:
        output_dir = output_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_sent.txt").write_text(prompt, encoding="utf-8")

    key = api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CiteSourcesResult,
            temperature=0.1,
            candidate_count=1,
            max_output_tokens=12000,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    (output_dir / "raw_response.json").write_text(response.text + "\n", encoding="utf-8")
    result = response.parsed or CiteSourcesResult.model_validate_json(response.text)
    warnings = validate_result(result, page_text, len(annotation.get("T", [])))
    usage = response.usage_metadata.model_dump() if response.usage_metadata is not None else {}
    record = {
        "volume": volume,
        "page": page,
        "model": MODEL,
        "valid": not warnings,
        "warnings": warnings,
        "usage": usage,
        "result": result.model_dump(),
    }
    (output_dir / "validated.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "review.md").write_text(
        markdown_review(page_text, annotation, result, warnings, usage),
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one cite-your-sources annotation audit.")
    parser.add_argument("--volume", type=int, default=2)
    parser.add_argument("--page", type=int, default=222)
    parser.add_argument("--output-name", default="")
    args = parser.parse_args()
    output_dir = run(args.volume, args.page, args.output_name)
    print(output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Add only the triples missing from an existing page summary/triple set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from konbaung_gemini_page_kg_annotator import DEFAULT_ENV_FILE, DEFAULT_SOURCE_ROOT, resolve_api_key
from konbaung_gemini_structured_open_coding_annotator import (
    OpenCodedTriple,
    usage_dict,
    validate_annotation,
    OpenCodingAnnotation,
)
from konbaung_gemini_xlmr_seed_annotator import page_job_from_path, write_json, write_text


DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_BASE_PROMPT = Path("konbaung_open_coding_user_base_prompt_v2.md")
DEFAULT_RESULTS_ROOT = Path("konbaung_open_coding_structured_full_batch_20260709/postprocessed")
DEFAULT_OUT_DIR = Path("konbaung_research_gap_enrichment_test")


class GapFillAnnotation(BaseModel):
    critique: str = Field(
        description="About 250 words identifying research-relevant evidence omitted or materially underspecified in both the existing summary and triples."
    )
    T: list[OpenCodedTriple] = Field(description="Only new triples missing from the existing triple set.")


PURPOSE = """This is a second-pass research-gap discovery task over an existing extraction from a Burmese royal chronicle. The first pass already produced an interpretive summary and span-grounded triples. Your purpose now is to find genuinely distinct evidence important to the research question that the first pass omitted or materially underspecified."""


GAP_FILL_INSTRUCTIONS = """Re-read TARGET_PAGE_TEXT independently in light of the research question. First, write a roughly 250-word critique identifying evidence important to understanding how power operated that is omitted or materially underspecified in BOTH EXISTING_SUMMARY and EXISTING_TRIPLES. The summary and triples are prior findings, not a checklist to complete.

Look for genuinely additional mechanisms of power, institutional practices, actor relationships, upward or downward flows of authority and resources, coercion, incorporation, negotiation, resistance, legitimacy, administration, or spatial organization. Evidence may qualify even when the summary omits it entirely.

Do not add a triple merely because the page supplies a quantity, duration, equipment list, personal name, subunit, tactical step, or more specific example of an existing relation. Such detail qualifies only if it changes the interpretation of power or introduces a distinct relationship. Do not split a broad existing narrative relation into several narrower restatements. It is acceptable and preferable to return no triples when the first pass already captures all research-relevant relations.

After the critique, return only additional triples representing the genuine research gaps you identified. Do not repeat, paraphrase, or slightly relabel claims already represented by the summary or existing triples.

Each additional triple must express a coherent power relation supported by TARGET_PAGE_TEXT. Both s and o must be literal Burmese spans from TARGET_PAGE_TEXT. English glosses and open codes may resolve omitted actors or explain the spans, but they must not introduce claims unsupported by the page. Keep the same open-coding principles and historiographical focus described above.

Each item in T has exactly these fields:
s = literal Burmese subject span
sg = concise English subject gloss
st = concise subject open code
p = concise English relation label, preferably ALL_CAPS_WITH_UNDERSCORES
o = literal Burmese object span
og = concise English object gloss
ot = concise object open code
d, dg = optional literal Burmese date/time span and English gloss, otherwise empty strings
l, lg = optional literal Burmese location span and English gloss, otherwise empty strings
q, qg = optional literal Burmese quantity span and English gloss, otherwise empty strings

Use the critique as the extraction checklist. Before returning, verify that every new triple adds a distinct, research-relevant relation rather than elaborating an existing narrative beat.

Return only schema-valid JSON containing critique followed by T."""


def historiographical_context(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="strict")
    marker = "You should proceed in this manner."
    if marker not in text:
        raise ValueError(f"Could not isolate historiographical context in {path}")
    return text[: text.index(marker)].strip()


def find_existing_result(root: Path, volume_id: str, page_num: int) -> Path:
    name = f"page_{page_num:04d}.json"
    matches = list(root.glob(f"*/{volume_id}/{name}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one existing result for {volume_id} {name}; found {matches}")
    return matches[0]


def build_prompt(base_prompt: Path, page_text: str, existing: dict, volume_id: str, page_num: int) -> str:
    context = historiographical_context(base_prompt)
    existing_payload = json.dumps(
        {"summary": existing["summary"], "T": existing["T"]},
        ensure_ascii=False,
        indent=2,
    )
    return f"""{PURPOSE}

<HISTORIOGRAPHICAL_CONTEXT read_only="yes">
The following research question and historiographical grounding come from the original first-pass prompt. They explain the analytical purpose of the extraction but contain no instructions for this second pass.

{context}
</HISTORIOGRAPHICAL_CONTEXT>

<CURRENT_SECOND_PASS_TASK_INSTRUCTIONS>
{GAP_FILL_INSTRUCTIONS}
</CURRENT_SECOND_PASS_TASK_INSTRUCTIONS>

<metadata>
volume_id: {volume_id}
page_num: {page_num:04d}
</metadata>

<EXISTING_ANNOTATION read_only="yes">
{existing_payload}
</EXISTING_ANNOTATION>

<TARGET_PAGE_TEXT annotate="yes">
{page_text}
</TARGET_PAGE_TEXT>
"""


def compact_triple_line(index: int, triple: dict) -> str:
    return (
        f'{index}. {triple.get("sg", "")} [{triple.get("st", "")}] '
        f'--{triple.get("p", "")}--> '
        f'{triple.get("og", "")} [{triple.get("ot", "")}]'
    )


def write_review(out_dir: Path, page_text: str, existing: dict, additional: dict, warnings: list[str]) -> None:
    lines = [
        "ORIGINAL PAGE TEXT",
        "==================",
        page_text.rstrip(),
        "",
        "EXISTING SUMMARY",
        "================",
        existing["summary"].strip(),
        "",
        "EXISTING TRIPLES",
        "================",
    ]
    lines.extend(compact_triple_line(i, triple) for i, triple in enumerate(existing["T"], start=1))
    lines.extend(["", "MODEL CRITIQUE OF EXISTING COVERAGE", "==================================="])
    lines.append(additional["critique"].strip())
    lines.extend(["", "ADDITIONAL TRIPLES", "=================="])
    lines.extend(compact_triple_line(i, triple) for i, triple in enumerate(additional["T"], start=1))
    lines.extend(["", "VALIDATION WARNINGS", "==================="])
    lines.extend(warnings or ["None"])
    write_text(out_dir / "review.txt", "\n".join(lines) + "\n")


def write_scan_review(out_dir: Path, existing: dict, additional: dict) -> None:
    lines = ["SUMMARY", "=======", existing["summary"].strip(), "", "ORIGINAL TRIPLES", "================"]
    lines.extend(compact_triple_line(i, triple) for i, triple in enumerate(existing["T"], start=1))
    lines.extend(["", "NEW TRIPLES", "==========="])
    lines.extend(compact_triple_line(i, triple) for i, triple in enumerate(additional["T"], start=1))
    write_text(out_dir / "scan_comparison.txt", "\n".join(lines) + "\n")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-id", default="konbaung_vol2")
    parser.add_argument("--page-num", type=int, default=258)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--base-prompt-file", default=str(DEFAULT_BASE_PROMPT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    page_path = Path(args.source_root) / args.volume_id / "pages" / f"page_{args.page_num:04d}.txt"
    existing_path = find_existing_result(Path(args.results_root), args.volume_id, args.page_num)
    existing = json.loads(existing_path.read_text(encoding="utf-8"))
    page_text = page_path.read_text(encoding="utf-8", errors="strict")
    prompt = build_prompt(Path(args.base_prompt_file), page_text, existing, args.volume_id, args.page_num)

    out_dir = Path(args.out_dir) / args.volume_id / f"page_{args.page_num:04d}"
    write_text(out_dir / "prompt_sent.txt", prompt)
    write_json(out_dir / "existing_annotation.json", existing)
    write_text(out_dir / "original_page.txt", page_text)

    if not args.run:
        print(json.dumps({"prompt": str(out_dir / "prompt_sent.txt"), "will_call_api": False}, indent=2))
        return

    api_key = resolve_api_key(args)
    if not api_key:
        raise RuntimeError("No Gemini API key found.")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=args.model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GapFillAnnotation,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=args.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    annotation = response.parsed or GapFillAnnotation.model_validate_json(response.text)
    write_json(out_dir / "raw_model_output.json", annotation.model_dump())
    synthetic = OpenCodingAnnotation(
        summary=existing["summary"],
        T=annotation.T,
        coverage_report="Second-pass critique and additional triples.",
    )
    job = page_job_from_path(page_path)
    normalized, warnings, resolutions = validate_annotation(job, synthetic)
    result = {
        "critique": annotation.critique,
        "T": normalized["T"],
    }
    write_json(out_dir / "additional_triples.json", result)
    write_review(out_dir, page_text, existing, result, warnings)
    write_scan_review(out_dir, existing, result)
    write_json(
        out_dir / "validation.json",
        {
            "validation_warnings": warnings,
            "span_resolutions": resolutions,
            "usage_metadata": usage_dict(response),
            "existing_triple_count": len(existing["T"]),
            "additional_triple_count": len(annotation.T),
            "combined_triple_count": len(existing["T"]) + len(annotation.T),
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "existing_triples": len(existing["T"]),
                "additional_triples": len(annotation.T),
                "combined_triples": len(existing["T"]) + len(annotation.T),
                "validation_warnings": warnings,
                "usage_metadata": usage_dict(response),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

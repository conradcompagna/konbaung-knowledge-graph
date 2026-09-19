#!/usr/bin/env python3
"""Structured open-coding annotator for Konbaung OCR pages.

This forks the successful Markdown open-coding prompt into enforced JSON while
keeping the annotation payload slim:
- T: span/gloss/code, relation, span/gloss/code

Span grounding is deliberately whitespace-insensitive. Spaces and OCR line
breaks are ignored for validation and offset recovery.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from konbaung_gemini_page_kg_annotator import DEFAULT_ENV_FILE, DEFAULT_SOURCE_ROOT, PageJob, resolve_api_key
from konbaung_gemini_xlmr_seed_annotator import (
    line_start_offsets,
    page_job_from_path,
    write_json,
    write_text,
)


DEFAULT_OUT_DIR = Path("konbaung_open_coding_structured_ready")
DEFAULT_MAX_OUTPUT_TOKENS = 10000
DEFAULT_BASE_PROMPT_FILE = Path("konbaung_open_coding_original_prompt.md")


class OpenCodedTriple(BaseModel):
    s: str = Field(description="Burmese subject span copied from TARGET_PAGE_TEXT.")
    sg: str = Field(description="Plain English gloss of the subject span.")
    st: str = Field(description="Open code tag for the subject span.")
    p: str = Field(description="Open-coded relation label, preferably ALL_CAPS_WITH_UNDERSCORES.")
    o: str = Field(description="Burmese object span copied from TARGET_PAGE_TEXT.")
    og: str = Field(description="Plain English gloss of the object span.")
    ot: str = Field(description="Open code tag for the object span.")
    d: str = Field(default="", description="Optional literal Burmese date/time span from TARGET_PAGE_TEXT.")
    dg: str = Field(default="", description="English gloss of d, or empty string if d is absent.")
    l: str = Field(default="", description="Optional literal Burmese location span from TARGET_PAGE_TEXT.")
    lg: str = Field(default="", description="English gloss of l, or empty string if l is absent.")
    q: str = Field(default="", description="Optional literal Burmese quantity span from TARGET_PAGE_TEXT.")
    qg: str = Field(default="", description="English gloss of q, or empty string if q is absent.")


class OpenCodingAnnotation(BaseModel):
    summary: str = Field(description="About 250 words summarizing the target page content in English.")
    T: list[OpenCodedTriple] = Field(description="Open-coded span-grounded triples.")
    coverage_report: str = Field(
        description="One sentence declaring that the summary contents are fully anchored in the triples."
    )


STRUCTURED_OUTPUT_INSTRUCTIONS = """The output format has three parts, returned through the JSON schema.

First return summary. It should be about 250 words in English and should summarize the content of TARGET_PAGE_TEXT before coding. Use this summary as the coverage checklist for T: the triple list should encode the same major actors, actions, objects, places, institutions, and power relations that appear in the summary. Do not write a richer summary and then a thin triple list.

Part 1: Open-coded triples. Return items in T. Each item has exactly these fields:

s = literal Burmese subject span from TARGET_PAGE_TEXT
sg = English gloss of s
st = subject open code label
p = open-coded relation label
o = literal Burmese object span from TARGET_PAGE_TEXT
og = English gloss of o
ot = object open code label
d = optional literal Burmese date/time metadata span from TARGET_PAGE_TEXT, or empty string
dg = English gloss of d, or empty string if d is absent
l = optional literal Burmese location metadata span from TARGET_PAGE_TEXT, or empty string
lg = English gloss of l, or empty string if l is absent
q = optional literal Burmese quantity metadata span from TARGET_PAGE_TEXT, or empty string
qg = English gloss of q, or empty string if q is absent

Extract all predicates that are clearly related to the research question and needed to reconstruct the summary. If a relation, event, command, movement, grant, construction, report, conflict, ritual act, or institutional action appears as a main point in the summary, represent it in T. Do not stop after a few examples if the page contains more relevant power relations. If a subject or object is a long list-like span, you may write it as `short literal start phrase ... short literal end phrase` so the boundaries are clear without copying half the page. Use that boundary notation only when the full span would be unwieldy; otherwise copy the literal Burmese span normally.

Optional triple metadata: date/time, location, and quantity.

For each triple, include metadata when the relevant information is explicitly present in TARGET_PAGE_TEXT. Do not infer dates, times, places, or quantities from outside knowledge, or the general historical setting.

For date/time metadata, copy the literal Burmese date or time span exactly as it appears in the page, and provide a plain English gloss. This may include regnal/calendar dates, month/day expressions, festival dates, time-of-day phrases, clock-beat phrases, dawn/night expressions, or duration phrases, but only when tied to the event or relation expressed by the triple.

For location metadata, copy the literal Burmese place span exactly as it appears in the page, and provide a plain English gloss. Include towns, villages, forts, monasteries, pagodas, camps, rivers, routes, gates, palace spaces, frontier zones, or named regions when they locate the triple's action.

For quantity metadata, copy the literal Burmese quantity span and gloss it in English. Include numbers of troops, captives, prisoners, monks, elephants, horses, weapons, boats, money, materials, days, stages, or other counted resources when relevant to the triple.

Metadata should clarify when, where, or how much. It should not repeat information already present in the base triple; it is intended as enrichment.

It is important to be specific and accurate because these will all be used in different modes of visualization. For example, dates will feed into chronological timelines of events, places will feed into spatial/map-based visualizations, and so on. Such graph-based visualizations will only work if you produce clean, machine-readable metadata that can be plotted on a chart.

Example triple:

s: `မြို့ ရွာသူကြီး, ခေါင်းအကြီးတို့`
sg: “town and village headmen, senior heads”
st: `LocalSecurityOfficials`
p: `MUST_COMPENSATE_PROPERTY_OWNER_FOR_THEFT`
o: `ဥစ္စာရှင်`
og: “the property owner”
ot: `VictimPropertyHolder`

Another example triple:

s: `ဗြိတိသျှတို့`
sg: “the British”
st: `ColonialPacificationAuthority`
p: `PRICES_REBEL_BODY_AS_BOUNTY`
o: `ဆုငွေ သုံးထောင် (၃၀၀၀-ကျပ်)`
og: “reward of 3,000 kyat”
ot: `LiveCaptureReward`

Part 3: summary coverage report. Return coverage_report as one sentence confirming that the summary is fully anchored in T and that no summary content has been left out of the triple list.

Do not return tables. Do not add a preface. Do not explain the research project again. Return only schema-valid JSON."""


FINAL_CONTEXT_INSTRUCTIONS = """Now process TARGET_PAGE_TEXT only. Return summary, T, and coverage_report only through the structured schema.

<metadata>
job_id: {job_id}
volume_id: {volume_id}
page_num: {page_num}
</metadata>

<TARGET_PAGE_TEXT annotate="yes">
{target_page_text}
</TARGET_PAGE_TEXT>
"""


COMPLETE_BASE_CONTEXT_INSTRUCTIONS = """Now process TARGET_PAGE_TEXT only. Return summary, T, and coverage_report only through the structured schema.

<metadata>
job_id: {job_id}
volume_id: {volume_id}
page_num: {page_num}
</metadata>

<TARGET_PAGE_TEXT>
{target_page_text}
</TARGET_PAGE_TEXT>
"""


def prompt_output_path(out_dir: Path, job: PageJob) -> Path:
    return out_dir / "prompts_sent" / job.volume_id / f"page_{job.page_num:04d}.txt"


def output_path(out_dir: Path, phase: str, bucket: str, job: PageJob) -> Path:
    return out_dir / phase / bucket / job.volume_id / f"page_{job.page_num:04d}.json"


def validation_output_path(out_dir: Path, bucket: str, job: PageJob) -> Path:
    return out_dir / "validation" / bucket / job.volume_id / f"page_{job.page_num:04d}.json"


def structuredize_base_prompt(base_prompt: str) -> str:
    output_marker = "The output format has two parts."
    look_marker = "What to look for in Burmese chronicle prose:"
    final_marker = "Now process the following Burmese passage."
    try:
        output_start = base_prompt.index(output_marker)
        look_start = base_prompt.index(look_marker)
        final_start = base_prompt.index(final_marker)
    except ValueError as exc:
        raise ValueError("Base open-coding prompt does not contain the expected section markers") from exc
    if not output_start < look_start < final_start:
        raise ValueError("Base open-coding prompt markers are out of order")

    prefix = base_prompt[:output_start].rstrip()
    look_section = base_prompt[look_start:final_start].rstrip()
    return f"{prefix}\n\n{STRUCTURED_OUTPUT_INSTRUCTIONS}\n\n{look_section}"


def read_structured_base_prompt(path: Path) -> str:
    base_prompt = path.read_text(encoding="utf-8", errors="strict")
    try:
        return structuredize_base_prompt(base_prompt)
    except ValueError:
        return base_prompt.strip()


def read_base_prompt_and_final_template(path: Path) -> tuple[str, str]:
    base_prompt = path.read_text(encoding="utf-8", errors="strict")
    try:
        return structuredize_base_prompt(base_prompt), FINAL_CONTEXT_INSTRUCTIONS
    except ValueError:
        return base_prompt.strip(), COMPLETE_BASE_CONTEXT_INSTRUCTIONS


def build_prompt(
    job: PageJob,
    base_prompt_file: Path,
) -> str:
    base_prompt, final_template = read_base_prompt_and_final_template(base_prompt_file)
    final_block = final_template.format(
        job_id=job.job_id,
        volume_id=job.volume_id,
        page_num=f"{job.page_num:04d}",
        target_page_text=job.raw_text,
    )
    return f"{base_prompt}\n\n{final_block}"


def usage_dict(response: Any) -> dict[str, Any]:
    if getattr(response, "usage_metadata", None) is None:
        return {}
    raw_usage = response.usage_metadata
    if hasattr(raw_usage, "model_dump"):
        return raw_usage.model_dump()
    return dict(raw_usage)


def call_gemini(
    client: genai.Client,
    *,
    model: str,
    prompt: str,
    max_output_tokens: int,
    max_retries: int,
    retry_sleep: float,
) -> tuple[OpenCodingAnnotation, dict[str, Any]]:
    last_err: Optional[Exception] = None
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=OpenCodingAnnotation,
        temperature=0.0,
        candidate_count=1,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            usage = usage_dict(response)
            if getattr(response, "parsed", None) is not None:
                return response.parsed, usage
            return OpenCodingAnnotation.model_validate_json(response.text), usage
        except (ValidationError, Exception) as exc:
            last_err = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep * (2**attempt) + random.random())
    raise RuntimeError(f"Gemini call failed after retries: {last_err}")


def compact_with_offsets(raw_text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    offsets: list[int] = []
    for idx, char in enumerate(raw_text):
        if char.isspace():
            continue
        chars.append(char)
        offsets.append(idx)
    return "".join(chars), offsets


def normalize_span_text(text: str) -> str:
    return " ".join(text.split())


def line_for_abs(offsets: list[int], abs_pos: int) -> tuple[int, int]:
    line_idx = 0
    for idx, start in enumerate(offsets):
        if start > abs_pos:
            break
        line_idx = idx
    return line_idx + 1, abs_pos - offsets[line_idx]


def line_spans_for_abs_range(raw_text: str, abs_s: int, abs_e: int) -> list[dict[str, Any]]:
    line_offsets = line_start_offsets(raw_text)
    spans: list[dict[str, Any]] = []
    for ln, line_start in enumerate(line_offsets, start=1):
        if ln < len(line_offsets):
            line_end = line_offsets[ln] - 1
        else:
            line_end = len(raw_text)
        start = max(abs_s, line_start)
        end = min(abs_e, line_end)
        if start >= end:
            continue
        line_text = raw_text[line_start:line_end]
        local_s = start - line_start
        local_e = end - line_start
        spans.append({"ln": ln, "s": local_s, "e": local_e, "tx": line_text[local_s:local_e]})
    return spans


def resolve_span_whitespace_insensitive(raw_text: str, tx: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    normalized_tx = normalize_span_text(tx)
    if not normalized_tx:
        return None, "blank span"

    compact_page, compact_offsets = compact_with_offsets(raw_text)

    boundary_match = re.match(r"^(.+?)\s*(?:\.\.\.|…)\s*(.+?)$", normalized_tx)
    if boundary_match:
        start_fragment = normalize_span_text(boundary_match.group(1))
        end_fragment = normalize_span_text(boundary_match.group(2))
        compact_start = "".join(start_fragment.split())
        compact_end = "".join(end_fragment.split())
        if not compact_start or not compact_end:
            return None, f"blank boundary span fragment: {tx!r}"
        start = compact_page.find(compact_start)
        if start == -1:
            return None, f"boundary start not found after ignoring spaces/newlines: {start_fragment!r}"
        end_start = compact_page.find(compact_end, start + len(compact_start))
        if end_start == -1:
            return None, f"boundary end not found after boundary start: {end_fragment!r}"
        compact_end_pos = end_start + len(compact_end)
        abs_s = compact_offsets[start]
        abs_e = compact_offsets[compact_end_pos - 1] + 1
        spans = line_spans_for_abs_range(raw_text, abs_s, abs_e)
        if not spans:
            return None, f"boundary span matched but offset recovery failed: {tx!r}"
        ln, local_s = line_for_abs(line_start_offsets(raw_text), abs_s)
        return (
            {
                "tx": normalized_tx,
                "boundary_start": start_fragment,
                "boundary_end": end_fragment,
                "abs_s": abs_s,
                "abs_e": abs_e,
                "ln": ln,
                "s": local_s,
                "e": spans[-1]["e"] if len(spans) == 1 else None,
                "spans": spans,
                "match_mode": "boundary_whitespace_insensitive",
                "match_count": 1,
            },
            None,
        )

    compact_needle = "".join(normalized_tx.split())
    if not compact_needle:
        return None, "blank span after whitespace removal"

    start = compact_page.find(compact_needle)
    if start == -1:
        return None, f"span not found after ignoring spaces/newlines: {tx!r}"

    match_count = compact_page.count(compact_needle)
    compact_end = start + len(compact_needle)
    abs_s = compact_offsets[start]
    abs_e = compact_offsets[compact_end - 1] + 1
    spans = line_spans_for_abs_range(raw_text, abs_s, abs_e)
    if not spans:
        return None, f"span matched but offset recovery failed: {tx!r}"
    ln, local_s = line_for_abs(line_start_offsets(raw_text), abs_s)
    return (
        {
            "tx": normalized_tx,
            "abs_s": abs_s,
            "abs_e": abs_e,
            "ln": ln,
            "s": local_s,
            "e": spans[-1]["e"] if len(spans) == 1 else None,
            "spans": spans,
            "match_mode": "whitespace_insensitive",
            "match_count": match_count,
        },
        None,
    )


def validate_annotation(job: PageJob, annotation: OpenCodingAnnotation) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    warnings: list[str] = []
    resolutions: list[dict[str, Any]] = []
    normalized_items: list[dict[str, Any]] = []

    seen: set[tuple[str, str, str]] = set()
    for idx, triple in enumerate(annotation.T, start=1):
        s = normalize_span_text(triple.s)
        o = normalize_span_text(triple.o)
        p = triple.p.strip()
        d = normalize_span_text(triple.d)
        l = normalize_span_text(triple.l)
        q = normalize_span_text(triple.q)
        key = (s, p.upper(), o)
        if key in seen:
            continue
        seen.add(key)

        item = {
            "s": s,
            "sg": triple.sg.strip(),
            "st": triple.st.strip(),
            "p": p,
            "o": o,
            "og": triple.og.strip(),
            "ot": triple.ot.strip(),
            "d": d,
            "dg": triple.dg.strip(),
            "l": l,
            "lg": triple.lg.strip(),
            "q": q,
            "qg": triple.qg.strip(),
        }
        normalized_items.append(item)

        s_resolved, s_warning = resolve_span_whitespace_insensitive(job.raw_text, s)
        o_resolved, o_warning = resolve_span_whitespace_insensitive(job.raw_text, o)
        d_resolved = None
        l_resolved = None
        q_resolved = None
        if s_warning:
            warnings.append(f"T[{idx}].s: {s_warning}")
        if o_warning:
            warnings.append(f"T[{idx}].o: {o_warning}")
        if d:
            d_resolved, d_warning = resolve_span_whitespace_insensitive(job.raw_text, d)
            if d_warning:
                warnings.append(f"T[{idx}].d: {d_warning}")
        if l:
            l_resolved, l_warning = resolve_span_whitespace_insensitive(job.raw_text, l)
            if l_warning:
                warnings.append(f"T[{idx}].l: {l_warning}")
        if q:
            q_resolved, q_warning = resolve_span_whitespace_insensitive(job.raw_text, q)
            if q_warning:
                warnings.append(f"T[{idx}].q: {q_warning}")
        resolutions.append(
            {
                "n": idx,
                "s": s,
                "o": o,
                "d": d,
                "l": l,
                "q": q,
                "s_resolved": s_resolved,
                "o_resolved": o_resolved,
                "d_resolved": d_resolved,
                "l_resolved": l_resolved,
                "q_resolved": q_resolved,
            }
        )

    return {
        "summary": annotation.summary.strip(),
        "T": normalized_items,
        "coverage_report": annotation.coverage_report.strip(),
    }, warnings, resolutions


def page_metadata(job: PageJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "volume_id": job.volume_id,
        "page_num": job.page_num,
        "source_path": job.source_path,
        "sha1": job.sha1,
        "char_count": job.char_count,
        "line_count": job.line_count,
    }


def run_page(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root)
    job_path = source_root / args.volume_id / "pages" / f"page_{args.page_num:04d}.txt"
    if not job_path.exists():
        raise FileNotFoundError(job_path)

    job = page_job_from_path(job_path)
    out_dir = Path(args.out_dir)
    prompt = build_prompt(job, Path(args.base_prompt_file))
    prompt_path = prompt_output_path(out_dir, job)
    write_text(prompt_path, prompt)

    summary = {
        **page_metadata(job),
        "out_dir": str(out_dir),
        "model": args.model,
        "schema": "structured open coding: summary, T=s/sg/st/p/o/og/ot/d/dg/l/lg/q/qg, coverage_report",
        "context_included": False,
        "base_prompt_file": str(args.base_prompt_file),
        "prompt_path": str(prompt_path),
        "max_output_tokens": args.max_output_tokens,
        "will_call_api": bool(args.run),
    }
    write_json(out_dir / "run_summary.json", summary)

    if not args.run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    api_key = resolve_api_key(args)
    if not api_key:
        raise RuntimeError("No Gemini API key found. Set GEMINI_API_KEY, pass --api-key, or use --env-file.")

    client = genai.Client(api_key=api_key)
    annotation, usage = call_gemini(
        client,
        model=args.model,
        prompt=prompt,
        max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )
    normalized, warnings, resolutions = validate_annotation(job, annotation)
    bucket = "invalid" if warnings else "valid"
    validation_record = {
        **page_metadata(job),
        "stage": "structured_open_coding",
        "usage_metadata": usage,
        "prompt_path": str(prompt_path),
        "validation_warnings": warnings,
        "span_resolutions": resolutions,
    }
    write_json(output_path(out_dir, "raw", bucket, job), annotation.model_dump())
    write_json(output_path(out_dir, "postprocessed", bucket, job), normalized)
    write_json(validation_output_path(out_dir, bucket, job), validation_record)
    print(
        json.dumps(
            {
                **summary,
                "bucket": bucket,
                "usage_metadata": usage,
                "triple_count": len(normalized["T"]),
                "validation_warnings": warnings,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structured open coding on one Konbaung OCR page.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--volume-id", default="konbaung_vol1")
    parser.add_argument("--page-num", type=int, default=222)
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--base-prompt-file", default=str(DEFAULT_BASE_PROMPT_FILE))
    parser.add_argument("--context-ratio", type=float, default=0.0, help="Ignored; prompts are one page only.")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    run_page(parse_args(argv))


if __name__ == "__main__":
    main()

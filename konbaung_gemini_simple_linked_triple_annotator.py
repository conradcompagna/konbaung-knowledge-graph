#!/usr/bin/env python3
"""One-pass simplified linked-triple annotator for Konbaung pages.

This variant deliberately avoids the large KG relation label set. It asks
Gemini for only linked triples where both arguments are exact spans in the
target page, using a six-label entity schema and an open English verb phrase.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Literal, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from konbaung_gemini_page_kg_annotator import DEFAULT_ENV_FILE, DEFAULT_SOURCE_ROOT, PageJob, resolve_api_key
from konbaung_gemini_xlmr_seed_annotator import (
    DEFAULT_CONTEXT_RATIO,
    build_page_index_with_neighbors,
    context_blocks,
    page_job_from_path,
    resolve_text_ref,
    write_json,
    write_text,
)


SimpleEntityType = Literal["PERSON", "GROUP", "PLACE", "OBJECT", "TIME", "EVENT"]

DEFAULT_OUT_DIR = Path("konbaung_simple_linked_triples_ready")
DEFAULT_MAX_OUTPUT_TOKENS = 7000


class LinkedSpan(BaseModel):
    tx: str = Field(description="Exact Burmese subject/object span copied from TARGET_PAGE_TEXT.")
    label: SimpleEntityType = Field(description="One simplified entity label.")
    en: str = Field(description="Compact English gloss for this span.")
    ln: int = Field(description="1-based starting line number in TARGET_PAGE_TEXT.")
    i: int = Field(default=1, description="1-based occurrence of tx starting from ln; use >1 only when duplicated.")


class EvidenceChunk(BaseModel):
    tx: str = Field(description="Exact Burmese evidence chunk copied from TARGET_PAGE_TEXT.")
    ln: int = Field(description="1-based starting line number in TARGET_PAGE_TEXT.")
    i: int = Field(default=1, description="1-based occurrence of tx starting from ln; use >1 only when duplicated.")


class LinkedTriple(BaseModel):
    n: int = Field(description="1-based triple number in reading order.")
    s: LinkedSpan = Field(description="Explicit subject/source span.")
    p: str = Field(description="Short English verb phrase for the relation, e.g. ordered, attacked, moved to.")
    o: LinkedSpan = Field(description="Explicit object/target span.")
    e: EvidenceChunk = Field(description="Exact text chunk grounding the relation.")


class LinkedTripleAnnotation(BaseModel):
    T: list[LinkedTriple] = Field(description="Only triples with two explicit linked spans in TARGET_PAGE_TEXT.")


LABEL_DEFINITIONS = """- PERSON: individual person or named human title/person reference
- GROUP: collective human actor, army, ethnic/social group, officeholders as a group
- PLACE: settlement, polity, geographic feature, palace, fort, camp, road, built site
- OBJECT: material object, weapon, vehicle, resource, text, offering, animal when treated as possession/equipment
- TIME: date, time of day, duration, calendrical phrase
- EVENT: action, process, state, command, battle, movement, construction, report, attack, retreat, oath"""


PROMPT_TEMPLATE = """Extract linked triples from one Burmese royal chronicle page.

You are creating seed training data for a span-based entity/relation model.

Only output triples where both subject and object are explicit spans in TARGET_PAGE_TEXT and are connected by a clear relation in the page. Do not output standalone NER. Do not include an entity unless it is part of a linked triple.

Use this closed entity label set only:
{label_definitions}

The relation label is open: write a short English verb phrase for p, such as ordered, attacked, moved to, built, heard, rewarded, interpreted, retreated from, was stationed at. Do not use the old closed relation set.

For each triple, copy exact Burmese spans for s.tx and o.tx, give each a compact English gloss, and choose one exact Burmese evidence chunk e.tx that grounds the relation. The evidence should contain or directly ground the linked spans and relation.

Skip page numbers, headers, footers, publisher text, footnotes, OCR debris, and isolated names or objects that are not linked by a clear relation.

Previous/next context is read-only. Use it only for speaker resolution, omitted subjects, pronouns, incomplete sentences, and continuity. Never return spans or evidence from context.

<metadata>
job_id: {job_id}
volume_id: {volume_id}
page_num: {page_num}
</metadata>

<PREVIOUS_CONTEXT annotate="no">
{previous_context}
</PREVIOUS_CONTEXT>

<TARGET_PAGE_TEXT annotate="yes">
{target_page_text}
</TARGET_PAGE_TEXT>

<NEXT_CONTEXT annotate="no">
{next_context}
</NEXT_CONTEXT>
"""


def build_prompt(job: PageJob, page_index: dict[tuple[str, int], PageJob], context_ratio: float) -> str:
    blocks = context_blocks(job, page_index, context_ratio)
    return PROMPT_TEMPLATE.format(
        label_definitions=LABEL_DEFINITIONS,
        job_id=job.job_id,
        volume_id=job.volume_id,
        page_num=f"{job.page_num:04d}",
        **blocks,
    )


def prompt_output_path(out_dir: Path, job: PageJob) -> Path:
    return out_dir / "prompts_sent" / job.volume_id / f"page_{job.page_num:04d}.txt"


def output_path(out_dir: Path, phase: str, bucket: str, job: PageJob) -> Path:
    return out_dir / phase / bucket / job.volume_id / f"page_{job.page_num:04d}.json"


def review_output_path(out_dir: Path, bucket: str, job: PageJob) -> Path:
    return out_dir / "review" / bucket / job.volume_id / f"page_{job.page_num:04d}.txt"


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
) -> tuple[LinkedTripleAnnotation, dict[str, Any]]:
    last_err: Optional[Exception] = None
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=LinkedTripleAnnotation,
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
            return LinkedTripleAnnotation.model_validate_json(response.text), usage
        except (ValidationError, Exception) as exc:
            last_err = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep * (2**attempt) + random.random())
    raise RuntimeError(f"Gemini call failed after retries: {last_err}")


def page_metadata(job: PageJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "volume_id": job.volume_id,
        "page_num": job.page_num,
        "source_path": job.source_path,
        "source_sha1": job.sha1,
        "char_count": job.char_count,
        "line_count": job.line_count,
    }


def validate_annotation(job: PageJob, annotation: LinkedTripleAnnotation) -> tuple[dict[str, Any], list[str], list[str]]:
    lines = job.raw_text.splitlines()
    errors: list[str] = []
    warnings: list[str] = []
    resolved: dict[str, Any] = {"T": []}
    seen: set[tuple[int, int, str, int, int]] = set()

    for idx, triple in enumerate(annotation.T, start=1):
        ref = f"T[{idx}]"
        if triple.n != idx:
            warnings.append(f"{ref}: n should be {idx}, got {triple.n}")
        if not triple.p.strip():
            errors.append(f"{ref}: blank p")

        s_resolved, s_error = resolve_text_ref(job.raw_text, lines, ln=triple.s.ln, tx=triple.s.tx, i=triple.s.i)
        o_resolved, o_error = resolve_text_ref(job.raw_text, lines, ln=triple.o.ln, tx=triple.o.tx, i=triple.o.i)
        e_resolved, e_error = resolve_text_ref(job.raw_text, lines, ln=triple.e.ln, tx=triple.e.tx, i=triple.e.i)
        if s_error:
            errors.append(f"{ref}.s: {s_error}")
        if o_error:
            errors.append(f"{ref}.o: {o_error}")
        if e_error:
            errors.append(f"{ref}.e: {e_error}")

        if s_resolved and o_resolved and e_resolved:
            key = (
                int(s_resolved.get("abs_s", -1)),
                int(s_resolved.get("abs_e", -1)),
                triple.p.strip().lower(),
                int(o_resolved.get("abs_s", -1)),
                int(o_resolved.get("abs_e", -1)),
            )
            if key in seen:
                warnings.append(f"{ref}: duplicate triple dropped")
                continue
            seen.add(key)
            for side, side_resolved in (("s", s_resolved), ("o", o_resolved)):
                if not (
                    e_resolved.get("abs_s", -1) <= side_resolved.get("abs_s", -2)
                    and side_resolved.get("abs_e", -2) <= e_resolved.get("abs_e", -1)
                ):
                    warnings.append(f"{ref}.{side}: span is outside evidence chunk")

        triple_data = triple.model_dump()
        triple_data["s"]["resolved"] = s_resolved or {}
        triple_data["o"]["resolved"] = o_resolved or {}
        triple_data["e"]["resolved"] = e_resolved or {}
        resolved["T"].append(triple_data)

    return resolved, errors, warnings


def collect_review_spans(annotation: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[int, int, str], str]]:
    review_spans: list[dict[str, Any]] = []
    id_by_key: dict[tuple[int, int, str], str] = {}
    next_id = 1
    for triple in annotation.get("T", []):
        for side in ("s", "o"):
            span = triple.get(side, {})
            resolved = span.get("resolved", {})
            abs_s = resolved.get("abs_s")
            abs_e = resolved.get("abs_e")
            label = span.get("label")
            if not isinstance(abs_s, int) or not isinstance(abs_e, int) or not label:
                continue
            key = (abs_s, abs_e, str(label))
            if key not in id_by_key:
                entity_id = f"E{next_id}"
                next_id += 1
                id_by_key[key] = entity_id
                for part in resolved.get("spans", []):
                    review_spans.append(
                        {
                            "id": entity_id,
                            "label": label,
                            "en": span.get("en", ""),
                            "ln": part.get("ln"),
                            "s": part.get("s"),
                            "e": part.get("e"),
                        }
                    )
            span["review_id"] = id_by_key[key]
    return review_spans, id_by_key


def annotate_line(line: str, spans: list[dict[str, Any]]) -> str:
    opens: dict[int, list[dict[str, Any]]] = {}
    closes: dict[int, list[dict[str, Any]]] = {}
    for span in spans:
        if not isinstance(span.get("s"), int) or not isinstance(span.get("e"), int):
            continue
        opens.setdefault(span["s"], []).append(span)
        closes.setdefault(span["e"], []).append(span)
    parts: list[str] = []
    for pos in range(len(line) + 1):
        for _span in sorted(closes.get(pos, []), key=lambda item: item["s"], reverse=True):
            parts.append("]")
        for span in sorted(opens.get(pos, []), key=lambda item: item["e"], reverse=True):
            parts.append(f"[{span.get('id')}|{span.get('label')} ")
        if pos < len(line):
            parts.append(line[pos])
    return "".join(parts)


def relation_line(triple: dict[str, Any]) -> str:
    s = triple.get("s", {})
    o = triple.get("o", {})
    return (
        f"R{triple.get('n')} "
        f"{s.get('review_id')}|{s.get('label')}|{s.get('en')} "
        f"--{triple.get('p')}--> "
        f"{o.get('review_id')}|{o.get('label')}|{o.get('en')}"
    )


def build_review_text(job: PageJob, annotation: dict[str, Any]) -> str:
    lines = job.raw_text.splitlines()
    review_spans, _id_by_key = collect_review_spans(annotation)
    spans_by_line: dict[int, list[dict[str, Any]]] = {}
    for span in review_spans:
        ln = span.get("ln")
        if isinstance(ln, int) and 1 <= ln <= len(lines):
            spans_by_line.setdefault(ln, []).append(span)

    relations_by_line: dict[int, list[str]] = {}
    for triple in annotation.get("T", []):
        e_spans = triple.get("e", {}).get("resolved", {}).get("spans", [])
        if e_spans:
            ln = max(part.get("ln", 0) for part in e_spans if isinstance(part.get("ln"), int))
        else:
            ln = triple.get("e", {}).get("ln")
        if isinstance(ln, int) and 1 <= ln <= len(lines):
            relations_by_line.setdefault(ln, []).append(relation_line(triple))

    output: list[str] = []
    for ln, line in enumerate(lines, start=1):
        output.append(annotate_line(line, spans_by_line.get(ln, [])))
        output.extend(relations_by_line.get(ln, []))
    return "\n".join(output) + "\n"


def run_page(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root)
    job_path = source_root / args.volume_id / "pages" / f"page_{args.page_num:04d}.txt"
    if not job_path.exists():
        raise FileNotFoundError(job_path)

    job = page_job_from_path(job_path)
    page_index = build_page_index_with_neighbors(source_root, [job])
    out_dir = Path(args.out_dir)
    prompt = build_prompt(job, page_index, args.context_ratio)
    prompt_path = prompt_output_path(out_dir, job)
    write_text(prompt_path, prompt)

    summary = {
        **page_metadata(job),
        "out_dir": str(out_dir),
        "model": args.model,
        "schema": "simple linked triples: T[].s(label/tx/en/ln/i), p open English verb phrase, o(label/tx/en/ln/i), e(tx/ln/i)",
        "prompt_path": str(prompt_path),
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
    resolved, errors, warnings = validate_annotation(job, annotation)
    bucket = "invalid" if errors else "valid"
    record = {
        **page_metadata(job),
        "stage": "simple_linked_triples",
        "usage_metadata": usage,
        "prompt_path": str(prompt_path),
        "annotation": resolved,
        "model_annotation": annotation.model_dump(),
        "validation_errors": errors,
        "validation_warnings": warnings,
    }
    write_json(output_path(out_dir, "raw", bucket, job), {**record, "annotation": annotation.model_dump()})
    write_json(output_path(out_dir, "postprocessed", bucket, job), record)
    write_text(review_output_path(out_dir, bucket, job), build_review_text(job, resolved))
    print(json.dumps({**summary, "bucket": bucket, "usage_metadata": usage, "triple_count": len(resolved["T"]), "errors": errors, "warnings": warnings}, ensure_ascii=False, indent=2))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-pass simple linked triple extraction on one Konbaung OCR page.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--volume-id", default="konbaung_vol1")
    parser.add_argument("--page-num", type=int, default=177)
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--context-ratio", type=float, default=DEFAULT_CONTEXT_RATIO)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    run_page(parse_args(argv))


if __name__ == "__main__":
    main()

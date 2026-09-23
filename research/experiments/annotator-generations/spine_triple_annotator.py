#!/usr/bin/env python3
"""Gemini narrative-spine triple annotator for Konbaung per-page OCR files.

Each output item starts with one exact Burmese evidence chunk, then records a
compact subject-predicate-object claim that helps reconstruct the page narrative.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from konbaung_gemini_kg_annotator import (
    ENTITY_DEFINITIONS,
    EntityType,
    RELATION_DEFINITIONS,
    RelationType,
)
from konbaung_gemini_page_kg_annotator import (
    DEFAULT_ENV_FILE,
    DEFAULT_SOURCE_ROOT,
    PageJob,
    build_jobs,
    line_numbered_text,
    page_lines,
    resolve_api_key,
)


DEFAULT_OUT_DIR = Path("konbaung_spine_triple_annotations_ready")
DEFAULT_MAX_OUTPUT_TOKENS = 6000
DEFAULT_CONTEXT_RATIO = 0.20


class EvidenceChunk(BaseModel):
    ln: int = Field(description="1-based line number in TARGET_PAGE_TEXT.")
    tx: str = Field(description="Exact original Burmese evidence chunk.")
    i: int = Field(
        default=1, description="1-based occurrence of tx on this line; use >1 only when duplicated."
    )


class TripleOut(BaseModel):
    n: int = Field(description="1-based triple number in narrative order.")
    s_en: str = Field(description="Short English subject gloss or canonical name.")
    s_type: EntityType = Field(description="One existing entity label for the subject.")
    p: RelationType = Field(description="Exactly one existing relationship type.")
    o_en: str = Field(description="Short English object gloss.")
    o_type: EntityType = Field(description="One existing entity label for the object.")
    en: str = Field(description="Short English gloss of the whole triple.")
    e: EvidenceChunk = Field(description="Exact Burmese evidence chunk supporting this triple.")


class TripleAnnotation(BaseModel):
    T: list[TripleOut] = Field(description="Narrative-spine subject-predicate-object triples.")


def definition_block(definitions: dict[str, str]) -> str:
    return "\n".join(f"- {label}: {definition}" for label, definition in definitions.items())


ENTITY_LABELS = definition_block(ENTITY_DEFINITIONS)
RELATION_TYPES = definition_block(RELATION_DEFINITIONS)


PROMPT_TEMPLATE = """Extract the narrative spine of one Burmese royal chronicle page as RDF-style triples.

Cover all major events in enough detail so that your triples can serve as a faithful summary of the document contents, but do not pad with overly minute details. 15-20 is enough on average.

Each triple must be a coherent subject-predicate-object claim grounded in the page. Annotate entities and relations using the schema labels, and write compact glosses for the subject/objects in English. Then choose a chunk of Burmese text to ground the evidence in and record it verbatim.

Skip page numbers, footnotes, publisher text, headers, footers, non-Burmese text, and OCR debris.

Previous/next context is read-only. Use it only for continuity; cite TARGET_PAGE_TEXT only.

Take care to remember that the subject is often omitted in chronicle prose - it may not be directly stated. Use context and reasoning in those cases to recover it.

<metadata>
job_id: {job_id}
volume_id: {volume_id}
page_num: {page_num}
</metadata>

<PREVIOUS_CONTEXT annotate="no">
{previous_context}
</PREVIOUS_CONTEXT>

<TARGET_PAGE_TEXT annotate="yes">
{raw_text}
</TARGET_PAGE_TEXT>

<NEXT_CONTEXT annotate="no">
{next_context}
</NEXT_CONTEXT>
"""


def context_excerpt(raw_text: str, max_chars: int, side: str) -> str:
    if max_chars <= 0 or not raw_text.strip():
        return ""
    lines = page_lines(raw_text)
    if side == "previous":
        selected: list[str] = []
        total = 0
        for line in reversed(lines):
            selected.append(line)
            total += len(line) + 1
            if total >= max_chars:
                break
        return line_numbered_text("\n".join(reversed(selected)))

    selected = []
    total = 0
    for line in lines:
        selected.append(line)
        total += len(line) + 1
        if total >= max_chars:
            break
    return line_numbered_text("\n".join(selected))


def build_page_index(jobs: list[PageJob]) -> dict[tuple[str, int], PageJob]:
    return {(job.volume_id, job.page_num): job for job in jobs}


def build_prompt(
    job: PageJob,
    page_index: Optional[dict[tuple[str, int], PageJob]] = None,
    context_ratio: float = DEFAULT_CONTEXT_RATIO,
) -> str:
    context_chars = int(job.char_count * context_ratio)
    previous_job = page_index.get((job.volume_id, job.page_num - 1)) if page_index else None
    next_job = page_index.get((job.volume_id, job.page_num + 1)) if page_index else None
    return PROMPT_TEMPLATE.format(
        entity_labels=ENTITY_LABELS,
        relation_types=RELATION_TYPES,
        job_id=job.job_id,
        volume_id=job.volume_id,
        page_num=f"{job.page_num:04d}",
        previous_context=context_excerpt(previous_job.raw_text, context_chars, "previous")
        if previous_job
        else "",
        raw_text=line_numbered_text(job.raw_text),
        next_context=context_excerpt(next_job.raw_text, context_chars, "next") if next_job else "",
    )


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def page_output_path(out_dir: Path, bucket: str, job: PageJob) -> Path:
    return out_dir / bucket / job.volume_id / f"page_{job.page_num:04d}.json"


def marked_page_output_path(out_dir: Path, bucket: str, job: PageJob) -> Path:
    return out_dir / bucket / job.volume_id / f"page_{job.page_num:04d}.txt"


def prompt_output_path(out_dir: Path, job: PageJob) -> Path:
    return out_dir / "prompts_sent" / job.volume_id / f"page_{job.page_num:04d}.txt"


def load_done_page_job_ids(out_dir: Path) -> set[str]:
    done: set[str] = set()
    for bucket in (
        "postprocessed/valid",
        "postprocessed/invalid",
        "raw/valid",
        "raw/invalid",
        "valid",
        "invalid",
        "errors",
    ):
        for path in (out_dir / bucket).glob("konbaung_vol*/page_*.json"):
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if obj.get("job_id"):
                done.add(obj["job_id"])
    return done


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def normalized_page_window(
    lines: list[str], start_ln: int
) -> tuple[str, list[Optional[tuple[int, int]]]]:
    chars: list[str] = []
    positions: list[Optional[tuple[int, int]]] = []
    previous_space = True

    for line_idx in range(start_ln - 1, len(lines)):
        line = lines[line_idx]
        if not previous_space:
            chars.append(" ")
            positions.append(None)
            previous_space = True
        for char_idx, char in enumerate(line):
            if char.isspace():
                if not previous_space:
                    chars.append(" ")
                    positions.append((line_idx + 1, char_idx))
                    previous_space = True
            else:
                chars.append(char)
                positions.append((line_idx + 1, char_idx))
                previous_space = False

    return "".join(chars).strip(), positions


def spans_from_match(
    lines: list[str],
    positions: list[Optional[tuple[int, int]]],
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    by_line: dict[int, list[int]] = {}
    for position in positions[start:end]:
        if position is None:
            continue
        ln, char_idx = position
        by_line.setdefault(ln, []).append(char_idx)

    spans: list[dict[str, Any]] = []
    for ln in sorted(by_line):
        indexes = by_line[ln]
        span_start = min(indexes)
        span_end = max(indexes) + 1
        spans.append(
            {
                "ln": ln,
                "s": span_start,
                "e": span_end,
                "tx": lines[ln - 1][span_start:span_end],
            }
        )
    return spans


def without_line_wrap_spaces(
    text: str, positions: list[Optional[tuple[int, int]]]
) -> tuple[str, list[Optional[tuple[int, int]]]]:
    chars: list[str] = []
    kept_positions: list[Optional[tuple[int, int]]] = []
    for char, position in zip(text, positions):
        if char == " " and position is None:
            continue
        chars.append(char)
        kept_positions.append(position)
    return "".join(chars), kept_positions


def without_all_spaces(
    text: str, positions: list[Optional[tuple[int, int]]]
) -> tuple[str, list[Optional[tuple[int, int]]]]:
    chars: list[str] = []
    kept_positions: list[Optional[tuple[int, int]]] = []
    for char, position in zip(text, positions):
        if char.isspace():
            continue
        chars.append(char)
        kept_positions.append(position)
    return "".join(chars), kept_positions


def resolve_mention(
    lines: list[str], mention: EvidenceChunk
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if mention.ln < 1 or mention.ln > len(lines):
        return None, f"line {mention.ln} is outside page line range"
    if mention.i < 1:
        return None, f"occurrence i must be >= 1 for {mention.tx!r}"
    if not mention.tx:
        return None, "blank mention text"

    line = lines[mention.ln - 1]
    start = -1
    search_from = 0
    for _ in range(mention.i):
        start = line.find(mention.tx, search_from)
        if start == -1:
            break
        search_from = start + len(mention.tx)

    if start != -1:
        return {
            "ln": mention.ln,
            "tx": mention.tx,
            "i": mention.i,
            "s": start,
            "e": start + len(mention.tx),
            "spans": [
                {"ln": mention.ln, "s": start, "e": start + len(mention.tx), "tx": mention.tx}
            ],
        }, None

    needle = normalize_ws(mention.tx)
    if not needle:
        return None, "blank mention text after whitespace normalization"

    line_hint_corrected = False
    haystack, positions = normalized_page_window(lines, mention.ln)
    start = -1
    search_from = 0
    for _ in range(mention.i):
        start = haystack.find(needle, search_from)
        if start == -1:
            break
        search_from = start + len(needle)

    match_text = haystack
    match_positions = positions
    if start == -1:
        match_text, match_positions = without_line_wrap_spaces(haystack, positions)
        search_from = 0
        for _ in range(mention.i):
            start = match_text.find(needle, search_from)
            if start == -1:
                full_haystack, full_positions = normalized_page_window(lines, 1)
                start = -1
                search_from = 0
                for _ in range(mention.i):
                    start = full_haystack.find(needle, search_from)
                    if start == -1:
                        break
                    search_from = start + len(needle)
                match_text = full_haystack
                match_positions = full_positions
                line_hint_corrected = start != -1
                if start == -1:
                    match_text, match_positions = without_line_wrap_spaces(
                        full_haystack, full_positions
                    )
                    search_from = 0
                    for _ in range(mention.i):
                        start = match_text.find(needle, search_from)
                        if start == -1:
                            compact_needle = needle.replace(" ", "")
                            compact_text, compact_positions = without_all_spaces(
                                full_haystack, full_positions
                            )
                            start = -1
                            search_from = 0
                            for _ in range(mention.i):
                                start = compact_text.find(compact_needle, search_from)
                                if start == -1:
                                    return (
                                        None,
                                        f"mention text not found in target page after whitespace normalization: {mention.tx!r}",
                                    )
                                search_from = start + len(compact_needle)
                            match_text = compact_text
                            match_positions = compact_positions
                            break
                        search_from = start + len(needle)
                    line_hint_corrected = True
                break
            search_from = start + len(needle)

    spans = spans_from_match(lines, match_positions, start, start + len(needle))
    resolved = {
        "ln": mention.ln,
        "tx": mention.tx,
        "i": mention.i,
        "normalized_tx": needle,
        "spans": spans,
    }
    if line_hint_corrected and spans:
        resolved["ln_hint"] = mention.ln
        resolved["ln"] = spans[0]["ln"]
        resolved["line_hint_corrected"] = True
    if len(spans) == 1:
        resolved["s"] = spans[0]["s"]
        resolved["e"] = spans[0]["e"]
    return resolved, None


def text_in_evidence(text: str, evidence: dict[str, Any]) -> bool:
    needle = normalize_ws(text)
    haystack = normalize_ws(str(evidence.get("normalized_tx") or evidence.get("tx") or ""))
    if not needle or not haystack:
        return False
    return needle in haystack or needle.replace(" ", "") in haystack.replace(" ", "")


def resolve_annotation(
    job: PageJob, annotation: TripleAnnotation
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    lines = page_lines(job.raw_text)
    resolved: dict[str, Any] = {"T": []}
    seen: set[tuple[str, str, str]] = set()

    for idx, triple in enumerate(annotation.T, start=1):
        triple_id = f"T[{idx}]"
        if triple.n != idx:
            errors.append(f"{triple_id}: n should be {idx}, got {triple.n}")
        if not triple.s_en.strip():
            errors.append(f"{triple_id}: blank subject")
        if not triple.o_en.strip():
            errors.append(f"{triple_id}: blank object")
        if not triple.en.strip():
            errors.append(f"{triple_id}: blank English explanation")
        key = (triple.s_en.strip(), str(triple.p), triple.o_en.strip())
        if key in seen:
            errors.append(f"{triple_id}: duplicate triple")
        seen.add(key)

        resolved_evidence, error = resolve_mention(lines, triple.e)
        if error:
            errors.append(f"{triple_id}: {error}")
            resolved_evidence = triple.e.model_dump()

        triple_data = triple.model_dump()
        triple_data["e"] = resolved_evidence
        resolved["T"].append(triple_data)

    return resolved, errors


def triple_marker_label(idx: int) -> str:
    return f"T{idx:02d}"


def triple_legend_line(idx: int, triple: dict[str, Any]) -> str:
    label = triple_marker_label(idx)
    return f"{label}: {triple.get('s_en', '')} --{triple.get('p', '')}--> {triple.get('o_en', '')} | {triple.get('en', '')}"


def triple_tag_line(triple: dict[str, Any]) -> str:
    return f"{triple.get('s_type', '')} --{triple.get('p', '')}--> {triple.get('o_type', '')} | {triple.get('en', '')}"


def build_marked_page_text(
    job: PageJob, annotation: dict[str, Any], validation_errors: list[str]
) -> str:
    lines = page_lines(job.raw_text)
    line_annotations: dict[int, list[dict[str, Any]]] = {}

    for triple_idx, triple in enumerate(annotation.get("T", []), start=1):
        evidence = triple.get("e", {})
        spans = evidence.get("spans") or []
        if not spans and {"ln", "s", "e"}.issubset(evidence):
            spans = [{"ln": evidence["ln"], "s": evidence["s"], "e": evidence["e"]}]
        valid_spans: list[dict[str, Any]] = []
        for span_idx, span in enumerate(spans, start=1):
            ln = span.get("ln")
            start = span.get("s")
            end = span.get("e")
            if not isinstance(ln, int) or not isinstance(start, int) or not isinstance(end, int):
                continue
            if ln < 1 or ln > len(lines) or start < 0 or end < start or end > len(lines[ln - 1]):
                continue
            valid_spans.append({"ln": ln, "start": start, "end": end, "span_idx": span_idx})
        for valid_idx, valid_span in enumerate(valid_spans, start=1):
            ln = valid_span["ln"]
            line_annotations.setdefault(ln, []).append(
                {
                    "triple_idx": triple_idx,
                    "mention_idx": 1,
                    "span_idx": valid_span["span_idx"],
                    "start": valid_span["start"],
                    "end": valid_span["end"],
                    "triple": triple,
                    "open_bracket": valid_idx == 1,
                    "close_bracket": valid_idx == len(valid_spans),
                    "emit_annotation": valid_idx == len(valid_spans),
                }
            )

    output: list[str] = []
    for ln, line in enumerate(lines, start=1):
        annotations = sorted(
            line_annotations.get(ln, []),
            key=lambda item: (
                item["start"],
                item["end"],
                item["triple_idx"],
                item["mention_idx"],
                item["span_idx"],
            ),
        )
        cursor = 0
        if not annotations:
            output.append(line)
            continue

        for item in annotations:
            triple = item["triple"]
            start = item["start"]
            end = item["end"]
            if start < cursor:
                continue
            if start > cursor:
                output.append(line[cursor:start])
            prefix = "[" if item["open_bracket"] else ""
            suffix = "]" if item["close_bracket"] else ""
            output.append(f"{prefix}{line[start:end]}{suffix}")
            if item["emit_annotation"]:
                output.append(triple_tag_line(triple))
                output.append("")
            cursor = end
        if cursor < len(line):
            output.append(line[cursor:])

    if validation_errors:
        output.append("")
        output.append("VALIDATION ERRORS")
        output.extend(f"- {error}" for error in validation_errors)
    return "\n".join(output) + "\n"


def call_gemini_structured(
    client: genai.Client,
    *,
    model: str,
    prompt: str,
    max_output_tokens: int,
    max_retries: int,
    retry_sleep: float,
) -> tuple[TripleAnnotation, dict[str, Any]]:
    last_err: Optional[Exception] = None
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=TripleAnnotation,
        temperature=0.0,
        candidate_count=1,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            usage = {}
            if getattr(response, "usage_metadata", None) is not None:
                raw_usage = response.usage_metadata
                if hasattr(raw_usage, "model_dump"):
                    usage = raw_usage.model_dump()
                else:
                    usage = dict(raw_usage)
            if getattr(response, "parsed", None) is not None:
                return response.parsed, usage
            return TripleAnnotation.model_validate_json(response.text), usage
        except (ValidationError, Exception) as exc:
            last_err = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep * (2**attempt) + random.random())
    raise RuntimeError(f"Gemini call failed after retries: {last_err}")


def prepare_outputs(
    args: argparse.Namespace, jobs: list[PageJob], page_index: dict[tuple[str, int], PageJob]
) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_volume: dict[str, int] = {}
    for job in jobs:
        by_volume[job.volume_id] = by_volume.get(job.volume_id, 0) + 1

    summary = {
        "source_root": str(Path(args.source_root)),
        "out_dir": str(out_dir),
        "annotation_output_layout": "raw/valid|invalid JSON, postprocessed/valid|invalid JSON, marked/valid|invalid TXT",
        "schema": "narrative-spine triples: T[].n/s_en/s_type/p/o_en/o_type/en/e",
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "context_ratio_each_side": args.context_ratio,
        "jobs_prepared": len(jobs),
        "jobs_by_volume": by_volume,
        "will_call_api": bool(args.run),
        "api_key_source": "not_resolved_in_prepare_mode"
        if not args.run
        else "resolved_at_run_time",
    }
    write_json(out_dir / "run_summary.json", summary)

    if args.save_prompts:
        prompt_jobs = jobs[: args.prompt_limit] if args.prompt_limit else jobs
        for job in prompt_jobs:
            write_text(
                prompt_output_path(out_dir, job),
                build_prompt(job, page_index=page_index, context_ratio=args.context_ratio),
            )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_annotation(
    args: argparse.Namespace, jobs: list[PageJob], page_index: dict[tuple[str, int], PageJob]
) -> None:
    api_key = resolve_api_key(args)
    if not api_key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY, pass --api-key, or use --env-file."
        )

    out_dir = Path(args.out_dir)
    done = load_done_page_job_ids(out_dir) if args.resume else set()
    client = genai.Client(api_key=api_key)

    if args.limit is not None:
        jobs = jobs[: args.limit]

    for job in tqdm(jobs, desc="Annotating triples"):
        if job.job_id in done:
            continue
        try:
            prompt = build_prompt(job, page_index=page_index, context_ratio=args.context_ratio)
            prompt_path = prompt_output_path(out_dir, job)
            write_text(prompt_path, prompt)
            annotation, usage = call_gemini_structured(
                client,
                model=args.model,
                prompt=prompt,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )
            resolved_annotation, validation_errors = resolve_annotation(job, annotation)
            base_record = {
                "job_id": job.job_id,
                "volume_id": job.volume_id,
                "page_num": job.page_num,
                "source_path": job.source_path,
                "source_sha1": job.sha1,
                "char_count": job.char_count,
                "line_count": job.line_count,
                "usage_metadata": usage,
                "prompt_path": str(prompt_path),
            }
            raw_record = {
                **base_record,
                "annotation": annotation.model_dump(),
            }
            postprocessed_record = {
                **base_record,
                "annotation": resolved_annotation,
            }
            if validation_errors:
                raw_record["validation_errors"] = validation_errors
                postprocessed_record["validation_errors"] = validation_errors
                write_json(page_output_path(out_dir, "raw/invalid", job), raw_record)
                write_json(
                    page_output_path(out_dir, "postprocessed/invalid", job), postprocessed_record
                )
                write_text(
                    marked_page_output_path(out_dir, "marked/invalid", job),
                    build_marked_page_text(job, resolved_annotation, validation_errors),
                )
            else:
                write_json(page_output_path(out_dir, "raw/valid", job), raw_record)
                write_json(
                    page_output_path(out_dir, "postprocessed/valid", job), postprocessed_record
                )
                write_text(
                    marked_page_output_path(out_dir, "marked/valid", job),
                    build_marked_page_text(job, resolved_annotation, validation_errors),
                )
        except Exception as exc:
            write_json(
                page_output_path(out_dir, "errors", job),
                {
                    "job_id": job.job_id,
                    "volume_id": job.volume_id,
                    "page_num": job.page_num,
                    "source_path": job.source_path,
                    "source_sha1": job.sha1,
                    "prompt_path": str(prompt_output_path(out_dir, job)),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            if not args.continue_on_error:
                raise


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or run Gemini narrative-spine triple annotation over Konbaung pages."
    )
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Root containing konbaung_vol*/pages/page_*.txt.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for narrative-spine triple annotations.",
    )
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Optional .env file containing GEMINI_API_KEY.",
    )
    parser.add_argument(
        "--api-key", default=None, help="Optional Gemini API key. Prefer env var or --env-file."
    )
    parser.add_argument(
        "--run", action="store_true", help="Actually call Gemini. Omit for prepare-only."
    )
    parser.add_argument(
        "--volume-id", default=None, help="Optional single volume filter, e.g. konbaung_vol1."
    )
    parser.add_argument(
        "--page-num", type=int, default=None, help="Optional single page-number filter."
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit jobs during API run.")
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Write debug prompt files without calling Gemini.",
    )
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=25,
        help="Max prompts to save when --save-prompts is used. Use 0 for all.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument(
        "--context-ratio",
        type=float,
        default=DEFAULT_CONTEXT_RATIO,
        help="Previous and next context size as a fraction of target page characters per side. Use 0 to disable.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    source_root = Path(args.source_root)
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    all_jobs = build_jobs(source_root)
    page_index = build_page_index(all_jobs)
    jobs = all_jobs
    if args.volume_id:
        jobs = [job for job in jobs if job.volume_id == args.volume_id]
    if args.page_num is not None:
        jobs = [job for job in jobs if job.page_num == args.page_num]
    if not jobs:
        raise RuntimeError(f"No page jobs found under {source_root}")
    prepare_outputs(args, jobs, page_index)
    if args.run:
        run_annotation(args, jobs, page_index)


if __name__ == "__main__":
    main()

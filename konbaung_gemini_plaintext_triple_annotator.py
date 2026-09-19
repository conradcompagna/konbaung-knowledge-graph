#!/usr/bin/env python3
"""Plain-text Gemini triple annotator for Konbaung per-page OCR files.

This variant does not use Gemini structured output. Gemini returns plain text
blocks; this script parses and validates them locally.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from google import genai
from google.genai import types
from tqdm import tqdm

from konbaung_gemini_kg_annotator import ENTITY_DEFINITIONS, RELATION_DEFINITIONS
from konbaung_gemini_page_kg_annotator import (
    DEFAULT_ENV_FILE,
    DEFAULT_SOURCE_ROOT,
    PageJob,
    build_jobs,
    line_numbered_text,
    page_lines,
    resolve_api_key,
)


DEFAULT_OUT_DIR = Path("konbaung_plaintext_triple_annotations_ready")
DEFAULT_MAX_OUTPUT_TOKENS = 12000
VALID_ENTITY_LABELS = set(ENTITY_DEFINITIONS)
VALID_RELATION_TYPES = set(RELATION_DEFINITIONS)


@dataclass
class ParsedTriple:
    chunk: str
    subject: str
    subject_tags: list[str]
    predicate: str
    object: str
    object_tags: list[str]
    gloss: str


def definition_block(definitions: dict[str, str]) -> str:
    return "\n".join(f"- {label}: {definition}" for label, definition in definitions.items())


ENTITY_LABELS = definition_block(ENTITY_DEFINITIONS)
RELATION_TYPES = definition_block(RELATION_DEFINITIONS)


PROMPT_TEMPLATE = """Extract RDF triples from one page of a Burmese royal chronicle.

Annotate the main facts of the narrative as subject-predicate-object triples with a short English gloss. Your triples should summarize the key people, places, events, movements, kinship claims, offices, and other information needed to reconstruct the core meaning of the page.

Skip page numbers, footnotes, headers, footers, non-Burmese text, OCR debris, and other noise. Do not annotate random word fragments. Do not exhaustively list every name in a genealogy unless the page's main content is that genealogy. Each triple must be supported by the Burmese CHUNK you quote.

Return plain text only. Number each annotation with a single line like 1., then use this exact 5-line block format, then --- after each block. Do not add JSON, markdown, explanations, or any metadata other than the number.

1.
exact Burmese evidence chunk from the page
subject text | ENTITY_LABEL or ENTITY_LABEL+ENTITY_LABEL
RELATION_TYPE
object text | ENTITY_LABEL or ENTITY_LABEL+ENTITY_LABEL
short English gloss
---

Examples:

1.
ကံရာဇာကြီးမူကား ရခိုင်ဓညဝတီကို စိုးအုပ်စံနေတော်မူသည်
ကံရာဇာကြီး | PERSON
RULES_CONTROLS_ADMINISTERS
ရခိုင်ဓညဝတီ | SETTLEMENT+POLITY_OR_REALM
Kanrazagyi ruled Rakhine Dhanyawaddy.
---

2.
ဓဇရာဇာမည်သော သာကီဝင်မင်းသည် ဆွေတော် မျိုးတော် အလုံးအရင်းနှင့်တကွ မဇ္ဈိမဒေသမှ ဖဲခဲ့သဖြင့်
ဓဇရာဇာ | PERSON
MOVES_TRAVELS
မဇ္ဈိမဒေသ | ADMINISTRATIVE_TERRITORY
Dhajaraja left the Middle Country with his kin and followers.
---

3.
မြောက်ဘက်နော်ရထာဟု တွင်လေသည်၊ မြောက်ဘက်နော်ရထာ၏ သားတော် မြေးတော်တို့လည်း ရိုးရာမပျက်
မြောက်ဘက်နော်ရထာ | PERSON
HAS_NAME_ALIAS_TITLE
မြောက်ဘက်နော်ရထာ | PERSON
He was called North Nawrahta.
---

ENTITY LABEL DEFINITIONS:
{entity_labels}

RELATION TYPE DEFINITIONS:
{relation_types}

<metadata>
job_id: {job_id}
volume_id: {volume_id}
page_num: {page_num}
</metadata>

<page_text>
{raw_text}
</page_text>
"""


def build_prompt(job: PageJob) -> str:
    return PROMPT_TEMPLATE.format(
        entity_labels=ENTITY_LABELS,
        relation_types=RELATION_TYPES,
        job_id=job.job_id,
        volume_id=job.volume_id,
        page_num=f"{job.page_num:04d}",
        raw_text=line_numbered_text(job.raw_text),
    )


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def output_path(out_dir: Path, bucket: str, job: PageJob, suffix: str) -> Path:
    return out_dir / bucket / job.volume_id / f"page_{job.page_num:04d}.{suffix}"


def load_done_page_job_ids(out_dir: Path) -> set[str]:
    done: set[str] = set()
    for bucket in ("postprocessed/valid", "postprocessed/invalid", "raw/valid", "raw/invalid", "errors"):
        for path in (out_dir / bucket).glob("konbaung_vol*/page_*.*"):
            if path.suffix == ".json":
                try:
                    obj = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if obj.get("job_id"):
                    done.add(obj["job_id"])
            else:
                stem = path.stem
                if stem.startswith("page_"):
                    # Raw text files do not contain job metadata; let JSON decide done state.
                    continue
    return done


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def normalized_page_window(lines: list[str]) -> tuple[str, list[Optional[tuple[int, int]]]]:
    chars: list[str] = []
    positions: list[Optional[tuple[int, int]]] = []
    previous_space = True
    for line_idx, line in enumerate(lines):
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


def without_line_wrap_spaces(text: str, positions: list[Optional[tuple[int, int]]]) -> tuple[str, list[Optional[tuple[int, int]]]]:
    chars: list[str] = []
    kept_positions: list[Optional[tuple[int, int]]] = []
    for char, position in zip(text, positions):
        if char == " " and position is None:
            continue
        chars.append(char)
        kept_positions.append(position)
    return "".join(chars), kept_positions


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
        spans.append({"ln": ln, "s": span_start, "e": span_end, "tx": lines[ln - 1][span_start:span_end]})
    return spans


def resolve_chunk(lines: list[str], chunk: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    needle = normalize_ws(chunk)
    if not needle:
        return None, "blank CHUNK"
    haystack, positions = normalized_page_window(lines)
    start = haystack.find(needle)
    match_text = haystack
    match_positions = positions
    if start == -1:
        match_text, match_positions = without_line_wrap_spaces(haystack, positions)
        start = match_text.find(needle)
    if start == -1:
        return None, f"CHUNK not found in page after whitespace normalization: {chunk!r}"
    spans = spans_from_match(lines, match_positions, start, start + len(needle))
    resolved = {"tx": chunk, "normalized_tx": needle, "spans": spans}
    if len(spans) == 1:
        resolved["ln"] = spans[0]["ln"]
        resolved["s"] = spans[0]["s"]
        resolved["e"] = spans[0]["e"]
    return resolved, None


def text_in_chunk(text: str, chunk: str) -> bool:
    needle = normalize_ws(text)
    haystack = normalize_ws(chunk)
    if not needle or not haystack:
        return False
    return needle in haystack or needle.replace(" ", "") in haystack.replace(" ", "")


def parse_tags(raw: str) -> list[str]:
    cleaned = raw.strip().strip(",").strip().strip("\"'")
    return [part.strip().strip("\"'") for part in re.split(r"[+,]", cleaned) if part.strip().strip("\"'")]


def parse_text_and_tags(line: str) -> tuple[str, list[str]]:
    cleaned = line.strip().strip(",")
    if "|" in cleaned:
        text, tags = cleaned.rsplit("|", 1)
        return text.strip().strip("\"'"), parse_tags(tags)

    match = re.match(r"^(?P<text>.*?)[\s\u00a0]+[\"']?(?P<tags>[A-Z_]+(?:[+,][A-Z_]+)*)[\"']?$", cleaned)
    if match:
        return match.group("text").strip().strip("\"'"), parse_tags(match.group("tags"))
    return cleaned.strip().strip("\"'"), []


def parse_plaintext_response(text: str) -> tuple[list[ParsedTriple], list[str]]:
    triples: list[ParsedTriple] = []
    errors: list[str] = []
    block: list[str] = []
    block_num = 0

    def finish_block(lines: list[str]) -> None:
        nonlocal block_num
        if not lines:
            return
        block_num += 1
        if len(lines) != 5:
            errors.append(f"block {block_num}: expected 5 lines, got {len(lines)}")
            return
        chunk = lines[0].strip().strip("\"'")
        subject, subject_tags = parse_text_and_tags(lines[1])
        predicate = lines[2].strip().strip(",").strip().strip("\"'")
        obj, object_tags = parse_text_and_tags(lines[3])
        gloss = lines[4].strip().strip("\"'")
        triples.append(
            ParsedTriple(
                chunk=chunk,
                subject=subject,
                subject_tags=subject_tags,
                predicate=predicate,
                object=obj,
                object_tags=object_tags,
                gloss=gloss,
            )
        )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"\d+[\.)]", line):
            continue
        if line in {"---", "END"}:
            finish_block(block)
            block = []
            continue
        if line == "TRIPLE":
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() in {"CHUNK", "SUBJECT", "SUBJECT_TAGS", "PREDICATE", "OBJECT", "OBJECT_TAGS", "GLOSS"}:
                line = value.strip()
        block.append(line)
    finish_block(block)
    return triples, errors


def validate_and_resolve(job: PageJob, triples: list[ParsedTriple], parse_errors: list[str]) -> tuple[dict[str, Any], list[str]]:
    errors = list(parse_errors)
    lines = page_lines(job.raw_text)
    resolved: dict[str, Any] = {"T": []}
    seen: set[tuple[str, str, str]] = set()

    for idx, triple in enumerate(triples, start=1):
        triple_id = f"T[{idx}]"
        chunk, chunk_error = resolve_chunk(lines, triple.chunk)
        if chunk_error:
            errors.append(f"{triple_id}: {chunk_error}")
            chunk = {"tx": triple.chunk}

        for label in triple.subject_tags:
            if label not in VALID_ENTITY_LABELS:
                errors.append(f"{triple_id}: invalid SUBJECT_TAGS label {label}")
        for label in triple.object_tags:
            if label not in VALID_ENTITY_LABELS:
                errors.append(f"{triple_id}: invalid OBJECT_TAGS label {label}")
        if triple.predicate not in VALID_RELATION_TYPES:
            errors.append(f"{triple_id}: invalid PREDICATE {triple.predicate}")
        if not text_in_chunk(triple.subject, triple.chunk):
            errors.append(f"{triple_id}: SUBJECT not found inside CHUNK")
        if not text_in_chunk(triple.object, triple.chunk):
            errors.append(f"{triple_id}: OBJECT not found inside CHUNK")
        key = (triple.subject.strip(), triple.predicate, triple.object.strip())
        if key in seen:
            errors.append(f"{triple_id}: duplicate triple")
        seen.add(key)

        resolved["T"].append(
            {
                "e": chunk,
                "s": triple.subject,
                "sl": triple.subject_tags,
                "p": triple.predicate,
                "o": triple.object,
                "ol": triple.object_tags,
                "en": triple.gloss,
            }
        )
    return resolved, errors


def triple_tag_line(triple: dict[str, Any]) -> str:
    subject_labels = "+".join(triple.get("sl", []))
    object_labels = "+".join(triple.get("ol", []))
    return f"{subject_labels} --{triple.get('p', '')}--> {object_labels} | {triple.get('en', '')}"


def build_marked_page_text(job: PageJob, annotation: dict[str, Any], validation_errors: list[str]) -> str:
    lines = page_lines(job.raw_text)
    line_annotations: dict[int, list[dict[str, Any]]] = {}
    for triple_idx, triple in enumerate(annotation.get("T", []), start=1):
        evidence = triple.get("e", {})
        spans = evidence.get("spans") or []
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
            key=lambda item: (item["start"], item["end"], item["triple_idx"], item["span_idx"]),
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


def call_gemini_plaintext(
    client: genai.Client,
    *,
    model: str,
    prompt: str,
    max_output_tokens: int,
    max_retries: int,
    retry_sleep: float,
) -> tuple[str, dict[str, Any]]:
    last_err: Optional[Exception] = None
    config = types.GenerateContentConfig(
        temperature=0.0,
        candidate_count=1,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            usage = {}
            if getattr(response, "usage_metadata", None) is not None:
                raw_usage = response.usage_metadata
                usage = raw_usage.model_dump() if hasattr(raw_usage, "model_dump") else dict(raw_usage)
            return response.text or "", usage
        except Exception as exc:
            last_err = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep * (2**attempt) + random.random())
    raise RuntimeError(f"Gemini call failed after retries: {last_err}")


def prepare_outputs(args: argparse.Namespace, jobs: list[PageJob]) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_volume: dict[str, int] = {}
    for job in jobs:
        by_volume[job.volume_id] = by_volume.get(job.volume_id, 0) + 1
    summary = {
        "source_root": str(Path(args.source_root)),
        "out_dir": str(out_dir),
        "annotation_output_layout": "raw/valid|invalid TXT, postprocessed/valid|invalid JSON, marked/valid|invalid TXT",
        "schema": "plain-text 5-line blocks: CHUNK, SUBJECT|TAGS, PREDICATE, OBJECT|TAGS, GLOSS",
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "jobs_prepared": len(jobs),
        "jobs_by_volume": by_volume,
        "will_call_api": bool(args.run),
        "api_key_source": "not_resolved_in_prepare_mode" if not args.run else "resolved_at_run_time",
    }
    write_json(out_dir / "run_summary.json", summary)
    if args.save_prompts:
        prompt_dir = out_dir / "debug_prompts"
        prompt_jobs = jobs[: args.prompt_limit] if args.prompt_limit else jobs
        for job in prompt_jobs:
            prompt_path = prompt_dir / job.volume_id / f"page_{job.page_num:04d}.txt"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(build_prompt(job), encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_annotation(args: argparse.Namespace, jobs: list[PageJob]) -> None:
    api_key = resolve_api_key(args)
    if not api_key:
        raise RuntimeError("No Gemini API key found. Set GEMINI_API_KEY, pass --api-key, or use --env-file.")
    out_dir = Path(args.out_dir)
    done = load_done_page_job_ids(out_dir) if args.resume else set()
    client = genai.Client(api_key=api_key)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    for job in tqdm(jobs, desc="Annotating plaintext triples"):
        if job.job_id in done:
            continue
        try:
            raw_text, usage = call_gemini_plaintext(
                client,
                model=args.model,
                prompt=build_prompt(job),
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )
            triples, parse_errors = parse_plaintext_response(raw_text)
            resolved_annotation, validation_errors = validate_and_resolve(job, triples, parse_errors)
            base_record = {
                "job_id": job.job_id,
                "volume_id": job.volume_id,
                "page_num": job.page_num,
                "source_path": job.source_path,
                "source_sha1": job.sha1,
                "char_count": job.char_count,
                "line_count": job.line_count,
                "usage_metadata": usage,
            }
            bucket = "invalid" if validation_errors else "valid"
            write_text(output_path(out_dir, f"raw/{bucket}", job, "txt"), raw_text)
            record = {**base_record, "annotation": resolved_annotation}
            if validation_errors:
                record["validation_errors"] = validation_errors
            write_json(output_path(out_dir, f"postprocessed/{bucket}", job, "json"), record)
            write_text(
                output_path(out_dir, f"marked/{bucket}", job, "txt"),
                build_marked_page_text(job, resolved_annotation, validation_errors),
            )
        except Exception as exc:
            write_json(
                output_path(out_dir, "errors", job, "json"),
                {
                    "job_id": job.job_id,
                    "volume_id": job.volume_id,
                    "page_num": job.page_num,
                    "source_path": job.source_path,
                    "source_sha1": job.sha1,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            if not args.continue_on_error:
                raise


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or run Gemini plain-text triple annotation over Konbaung pages.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT), help="Root containing konbaung_vol*/pages/page_*.txt.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for plain-text triple annotations.")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="Optional .env file containing GEMINI_API_KEY.")
    parser.add_argument("--api-key", default=None, help="Optional Gemini API key. Prefer env var or --env-file.")
    parser.add_argument("--run", action="store_true", help="Actually call Gemini. Omit for prepare-only.")
    parser.add_argument("--volume-id", default=None, help="Optional single volume filter, e.g. konbaung_vol1.")
    parser.add_argument("--page-num", type=int, default=None, help="Optional single page-number filter.")
    parser.add_argument("--limit", type=int, default=None, help="Limit jobs during API run.")
    parser.add_argument("--save-prompts", action="store_true", help="Write debug prompt files without calling Gemini.")
    parser.add_argument("--prompt-limit", type=int, default=25, help="Max prompts to save when --save-prompts is used. Use 0 for all.")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
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
    jobs = build_jobs(source_root)
    if args.volume_id:
        jobs = [job for job in jobs if job.volume_id == args.volume_id]
    if args.page_num is not None:
        jobs = [job for job in jobs if job.page_num == args.page_num]
    if not jobs:
        raise RuntimeError(f"No page jobs found under {source_root}")
    prepare_outputs(args, jobs)
    if args.run:
        run_annotation(args, jobs)


if __name__ == "__main__":
    main()

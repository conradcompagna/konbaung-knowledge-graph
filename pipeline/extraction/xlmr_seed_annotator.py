#!/usr/bin/env python3
"""Two-stage Gemini annotator for XLM-R seed data.

Default behavior is prepare-only. The script will not call Gemini unless
--run is supplied.

Stage 1 creates supervised seed data for an entity/event span tagger:
    raw page text -> labeled spans

Stage 2 creates supervised seed data for a relation classifier:
    raw page text + validated span IDs -> positive labeled relations
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Optional, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from pipeline.extraction.kg_annotator import (
    ENTITY_DEFINITIONS,
    EntityType,
    RELATION_DEFINITIONS,
    RelationType,
)
from pipeline.extraction.page_kg_annotator import (
    DEFAULT_ENV_FILE,
    DEFAULT_SOURCE_ROOT,
    PageJob,
    build_jobs,
    iter_page_files,
    line_numbered_text,
    page_lines,
    resolve_api_key,
    sha1_text,
)


DEFAULT_OUT_DIR = Path("konbaung_xlmr_seed_annotations_ready")
DEFAULT_CONTEXT_RATIO = 0.20
DEFAULT_MAX_ENTITY_OUTPUT_TOKENS = 12000
DEFAULT_MAX_RELATION_OUTPUT_TOKENS = 9000


class SpanOut(BaseModel):
    id: str = Field(description="Local span ID: E1, E2, E3...")
    label: EntityType = Field(description="Exactly one existing entity/event label.")
    ln: int = Field(description="1-based starting line number in TARGET_PAGE_TEXT.")
    tx: str = Field(description="Exact Burmese text span copied from TARGET_PAGE_TEXT.")
    i: int = Field(
        default=1,
        description="1-based occurrence of tx starting from ln; use >1 only when duplicated.",
    )


class SpanAnnotation(BaseModel):
    spans: list[SpanOut] = Field(description="Entity/event spans for XLM-R span tagger training.")


class RelationOut(BaseModel):
    id: str = Field(description="Local relation ID: R1, R2, R3...")
    s: str = Field(description="Source/subject span ID from Call 1.")
    p: RelationType = Field(description="Exactly one existing relationship type.")
    o: str = Field(description="Target/object span ID from Call 1.")
    ln: int = Field(description="1-based starting line number for evidence in TARGET_PAGE_TEXT.")
    tx: str = Field(description="Exact Burmese evidence text copied from TARGET_PAGE_TEXT.")
    i: int = Field(
        default=1,
        description="1-based occurrence of tx starting from ln; use >1 only when duplicated.",
    )


class RelationAnnotation(BaseModel):
    relations: list[RelationOut] = Field(
        description="Positive relation labels between Call 1 span IDs."
    )


T = TypeVar("T", bound=BaseModel)


def definition_block(definitions: dict[str, str]) -> str:
    return "\n".join(f"- {label}: {definition}" for label, definition in definitions.items())


def schema_block(definitions: dict[str, str], detail: str) -> str:
    if detail == "none":
        return "(schema labels are enforced by the structured output schema)"
    if detail == "names":
        return ", ".join(definitions.keys())
    return definition_block(definitions)


ENTITY_PROMPT_TEMPLATE = """Tag entity and event spans in one Burmese royal chronicle page.

You are creating supervised training data for an XLM-R entity/event span tagger. This is not a final knowledge graph and not a narrative summary.

Mark explicit text spans in TARGET_PAGE_TEXT that should train a model to find graph-node candidates. Use only schema entity labels. Include people, groups, polities, places, offices, titles, objects, dates, named works, omens, abstract statuses, and event/process spans when they are expressed in the target page.

Tag useful repeated mentions when they occur in different places. Do not tag page numbers, headers, footers, publisher text, punctuation-only spans, generic particles, pronouns, or OCR debris.

Prefer the longest clear span for one semantic unit. Avoid overlapping spans unless the shorter span has a different label and is genuinely needed. Event spans may include the main verb phrase and essential complements, but should not swallow an entire paragraph.

Previous/next context is read-only. Use it only for speaker resolution, omitted subjects, pronouns, incomplete sentences, and continuity. Never return spans from context. Every tx must come from TARGET_PAGE_TEXT.

ENTITY_LABELS:
{entity_labels}

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


RELATION_PROMPT_TEMPLATE = """Tag relations between provided entity/event span IDs in one Burmese royal chronicle page.

You are creating supervised training data for an XLM-R relation classifier. Use TARGET_PAGE_TEXT plus ENTITY_SPANS_FROM_CALL_1 to identify positive relations. This is not a final knowledge graph and not a narrative summary.

Use only schema relation labels. Each relation must connect two provided span IDs. Do not invent new spans. Direction matters: s is the source/subject span ID, o is the target/object span ID.

Return positive relations only. NO_RELATION examples will be generated later by code from unlinked span pairs.

Use previous/next context only to resolve omitted subjects, pronouns, speakers, and continuity. Do not create relations whose evidence is only in context. Every evidence tx must come from TARGET_PAGE_TEXT.

If a relation depends on an omitted subject, link it to the best explicit span ID from TARGET_PAGE_TEXT when context clearly resolves it. If no provided span can serve as an argument, skip the relation rather than inventing an implicit node.

Avoid relation spam. Tag graph-relevant links such as command, report, movement, attack, defense, defeat, retreat, construction, naming, reward, office/status, kinship, succession, cause/result, event date/place, and similar historical relations.

RELATION_LABELS:
{relation_labels}

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

<ENTITY_SPANS_FROM_CALL_1>
{entity_spans_json}
</ENTITY_SPANS_FROM_CALL_1>
"""


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


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


def page_job_from_path(path: Path) -> PageJob:
    volume_id = path.parent.parent.name
    page_num = int(path.stem.split("_", 1)[1])
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    digest = sha1_text(raw_text)
    return PageJob(
        job_id=f"{volume_id}_p{page_num:04d}_{digest[:8]}",
        volume_id=volume_id,
        page_num=page_num,
        source_path=str(path),
        raw_text=raw_text,
        sha1=digest,
        char_count=len(raw_text),
        line_count=len(page_lines(raw_text)),
    )


def build_jobs_with_limit(source_root: Path, limit: Optional[int]) -> list[PageJob]:
    if limit is None:
        return build_jobs(source_root)

    jobs: list[PageJob] = []
    for path in iter_page_files(source_root):
        jobs.append(page_job_from_path(path))
        if len(jobs) >= limit:
            break
    return jobs


def build_page_index_with_neighbors(
    source_root: Path, jobs: list[PageJob]
) -> dict[tuple[str, int], PageJob]:
    index = build_page_index(jobs)
    for job in list(jobs):
        for page_num in (job.page_num - 1, job.page_num + 1):
            key = (job.volume_id, page_num)
            if key in index:
                continue
            path = source_root / job.volume_id / "pages" / f"page_{page_num:04d}.txt"
            if path.exists():
                index[key] = page_job_from_path(path)
    return index


def build_selected_jobs(args: argparse.Namespace, source_root: Path) -> list[PageJob]:
    if args.volume_id or args.page_num is not None:
        if not args.volume_id or args.page_num is None:
            raise ValueError("--volume-id and --page-num must be supplied together")
        path = source_root / args.volume_id / "pages" / f"page_{args.page_num:04d}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        return [page_job_from_path(path)]
    return build_jobs_with_limit(source_root, args.limit)


def context_blocks(
    job: PageJob, page_index: dict[tuple[str, int], PageJob], context_ratio: float
) -> dict[str, str]:
    context_chars = int(job.char_count * context_ratio)
    previous_job = page_index.get((job.volume_id, job.page_num - 1))
    next_job = page_index.get((job.volume_id, job.page_num + 1))
    return {
        "previous_context": context_excerpt(previous_job.raw_text, context_chars, "previous")
        if previous_job
        else "",
        "target_page_text": line_numbered_text(job.raw_text),
        "next_context": context_excerpt(next_job.raw_text, context_chars, "next")
        if next_job
        else "",
    }


def build_entity_prompt(
    job: PageJob,
    page_index: dict[tuple[str, int], PageJob],
    context_ratio: float,
    schema_detail: str,
) -> str:
    blocks = context_blocks(job, page_index, context_ratio)
    return ENTITY_PROMPT_TEMPLATE.format(
        entity_labels=schema_block(ENTITY_DEFINITIONS, schema_detail),
        job_id=job.job_id,
        volume_id=job.volume_id,
        page_num=f"{job.page_num:04d}",
        **blocks,
    )


def relation_prompt_spans(entity_record: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_spans: list[dict[str, Any]] = []
    for span in entity_record.get("annotation", {}).get("spans", []):
        prompt_spans.append(
            {
                "id": span.get("id"),
                "label": span.get("label"),
                "ln": span.get("ln"),
                "tx": span.get("tx"),
                "i": span.get("i", 1),
                "abs_s": span.get("abs_s"),
                "abs_e": span.get("abs_e"),
            }
        )
    return prompt_spans


def build_relation_prompt(
    job: PageJob,
    page_index: dict[tuple[str, int], PageJob],
    context_ratio: float,
    schema_detail: str,
    entity_record: dict[str, Any],
) -> str:
    blocks = context_blocks(job, page_index, context_ratio)
    return RELATION_PROMPT_TEMPLATE.format(
        relation_labels=schema_block(RELATION_DEFINITIONS, schema_detail),
        job_id=job.job_id,
        volume_id=job.volume_id,
        page_num=f"{job.page_num:04d}",
        entity_spans_json=json.dumps(
            relation_prompt_spans(entity_record), ensure_ascii=False, indent=2
        ),
        **blocks,
    )


def line_start_offsets(raw_text: str) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    for part in raw_text.splitlines(keepends=True):
        offsets.append(cursor)
        cursor += len(part)
    if raw_text and not offsets:
        offsets.append(0)
    return offsets


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

    return "".join(chars), positions


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


def add_absolute_offsets(raw_text: str, resolved: dict[str, Any]) -> dict[str, Any]:
    offsets = line_start_offsets(raw_text)
    for span in resolved.get("spans", []):
        ln = span.get("ln")
        if isinstance(ln, int) and 1 <= ln <= len(offsets):
            span["abs_s"] = offsets[ln - 1] + span["s"]
            span["abs_e"] = offsets[ln - 1] + span["e"]
    if resolved.get("spans"):
        first = resolved["spans"][0]
        last = resolved["spans"][-1]
        if "abs_s" in first and "abs_e" in last:
            resolved["abs_s"] = first["abs_s"]
            resolved["abs_e"] = last["abs_e"]
    return resolved


def resolve_text_ref(
    raw_text: str,
    lines: list[str],
    *,
    ln: int,
    tx: str,
    i: int,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if ln < 1 or ln > len(lines):
        return None, f"line {ln} is outside page line range"
    if i < 1:
        return None, f"occurrence i must be >= 1 for {tx!r}"
    if not tx:
        return None, "blank text span"

    line = lines[ln - 1]
    start = -1
    search_from = 0
    for _ in range(i):
        start = line.find(tx, search_from)
        if start == -1:
            break
        search_from = start + len(tx)
    if start != -1:
        resolved = {
            "ln": ln,
            "tx": tx,
            "i": i,
            "s": start,
            "e": start + len(tx),
            "spans": [{"ln": ln, "s": start, "e": start + len(tx), "tx": tx}],
            "match_mode": "exact_line",
        }
        return add_absolute_offsets(raw_text, resolved), None

    needle = normalize_ws(tx)
    if not needle:
        return None, "blank text span after whitespace normalization"

    line_hint_corrected = False
    match_text, match_positions = normalized_page_window(lines, ln)
    match_needle = needle
    start = -1
    search_from = 0
    for _ in range(i):
        start = match_text.find(match_needle, search_from)
        if start == -1:
            break
        search_from = start + len(match_needle)

    match_mode = "normalized_from_hint"
    if start == -1:
        match_text, match_positions = without_line_wrap_spaces(match_text, match_positions)
        search_from = 0
        for _ in range(i):
            start = match_text.find(match_needle, search_from)
            if start == -1:
                break
            search_from = start + len(match_needle)
        match_mode = "normalized_no_linewrap_space_from_hint"

    if start == -1:
        match_text, match_positions = normalized_page_window(lines, 1)
        search_from = 0
        for _ in range(i):
            start = match_text.find(match_needle, search_from)
            if start == -1:
                break
            search_from = start + len(match_needle)
        line_hint_corrected = start != -1
        match_mode = "normalized_full_page"

    if start == -1:
        match_text, match_positions = without_line_wrap_spaces(match_text, match_positions)
        search_from = 0
        for _ in range(i):
            start = match_text.find(match_needle, search_from)
            if start == -1:
                break
            search_from = start + len(match_needle)
        line_hint_corrected = start != -1
        match_mode = "normalized_no_linewrap_space_full_page"

    if start == -1:
        compact_needle = needle.replace(" ", "")
        compact_text, compact_positions = without_all_spaces(*normalized_page_window(lines, 1))
        match_text = compact_text
        match_positions = compact_positions
        match_needle = compact_needle
        search_from = 0
        for _ in range(i):
            start = match_text.find(match_needle, search_from)
            if start == -1:
                return (
                    None,
                    f"text span not found in target page after whitespace normalization: {tx!r}",
                )
            search_from = start + len(match_needle)
        line_hint_corrected = True
        match_mode = "compact_full_page"

    spans = spans_from_match(lines, match_positions, start, start + len(match_needle))
    if not spans:
        return None, f"text span matched only inserted whitespace: {tx!r}"
    resolved = {
        "ln": spans[0]["ln"],
        "ln_hint": ln,
        "tx": tx,
        "i": i,
        "normalized_tx": needle,
        "spans": spans,
        "match_mode": match_mode,
    }
    if line_hint_corrected:
        resolved["line_hint_corrected"] = True
    if len(spans) == 1:
        resolved["s"] = spans[0]["s"]
        resolved["e"] = spans[0]["e"]
    return add_absolute_offsets(raw_text, resolved), None


def validate_span_annotation(
    job: PageJob, annotation: SpanAnnotation
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    lines = page_lines(job.raw_text)
    resolved: dict[str, Any] = {"spans": []}
    seen_ids: set[str] = set()
    seen_locations: set[tuple[str, int, int]] = set()

    for idx, span in enumerate(annotation.spans, start=1):
        span_ref = f"spans[{idx}]"
        if not re.fullmatch(r"E[1-9]\d*", span.id):
            errors.append(f"{span_ref}: id should look like E1, E2, E3; got {span.id!r}")
        if span.id in seen_ids:
            errors.append(f"{span_ref}: duplicate span id {span.id}")
        seen_ids.add(span.id)
        if not span.tx.strip():
            errors.append(f"{span_ref}: blank tx")

        resolved_ref, error = resolve_text_ref(
            job.raw_text, lines, ln=span.ln, tx=span.tx, i=span.i
        )
        if error:
            errors.append(f"{span_ref} {span.id}: {error}")
            resolved_ref = span.model_dump()
        else:
            key = (
                str(span.label),
                int(resolved_ref.get("abs_s", -1)),
                int(resolved_ref.get("abs_e", -1)),
            )
            if key in seen_locations:
                warnings.append(f"{span_ref} {span.id}: duplicate label/location dropped")
                continue
            seen_locations.add(key)
            if resolved_ref.get("line_hint_corrected"):
                warnings.append(
                    f"{span_ref} {span.id}: line hint corrected from {span.ln} to {resolved_ref.get('ln')}"
                )

        span_data = span.model_dump()
        if resolved_ref is not None:
            span_data.update(resolved_ref)
        resolved["spans"].append(span_data)

    return resolved, errors, warnings


def validate_relation_annotation(
    job: PageJob,
    annotation: RelationAnnotation,
    entity_record: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    lines = page_lines(job.raw_text)
    resolved: dict[str, Any] = {"relations": []}
    span_ids = {span.get("id") for span in entity_record.get("annotation", {}).get("spans", [])}
    seen_ids: set[str] = set()
    seen_relations: set[tuple[str, str, str, int, int]] = set()

    for idx, relation in enumerate(annotation.relations, start=1):
        rel_ref = f"relations[{idx}]"
        if not re.fullmatch(r"R[1-9]\d*", relation.id):
            errors.append(f"{rel_ref}: id should look like R1, R2, R3; got {relation.id!r}")
        if relation.id in seen_ids:
            errors.append(f"{rel_ref}: duplicate relation id {relation.id}")
        seen_ids.add(relation.id)
        if relation.s not in span_ids:
            errors.append(f"{rel_ref} {relation.id}: source span id not found: {relation.s}")
        if relation.o not in span_ids:
            errors.append(f"{rel_ref} {relation.id}: object span id not found: {relation.o}")
        if not relation.tx.strip():
            errors.append(f"{rel_ref} {relation.id}: blank evidence tx")

        resolved_ref, error = resolve_text_ref(
            job.raw_text, lines, ln=relation.ln, tx=relation.tx, i=relation.i
        )
        if error:
            errors.append(f"{rel_ref} {relation.id}: {error}")
            resolved_ref = {
                "ln": relation.ln,
                "tx": relation.tx,
                "i": relation.i,
            }
        else:
            key = (
                relation.s,
                str(relation.p),
                relation.o,
                int(resolved_ref.get("abs_s", -1)),
                int(resolved_ref.get("abs_e", -1)),
            )
            if key in seen_relations:
                errors.append(f"{rel_ref} {relation.id}: duplicate relation/evidence")
            seen_relations.add(key)
            if resolved_ref.get("line_hint_corrected"):
                warnings.append(
                    f"{rel_ref} {relation.id}: line hint corrected from {relation.ln} to {resolved_ref.get('ln')}"
                )

        relation_data = relation.model_dump()
        relation_data["evidence"] = resolved_ref
        resolved["relations"].append(relation_data)

    return resolved, errors, warnings


def prompt_output_path(out_dir: Path, stage: str, job: PageJob) -> Path:
    return out_dir / "prompts_sent" / stage / job.volume_id / f"page_{job.page_num:04d}.txt"


def stage_output_path(out_dir: Path, stage_dir: str, phase: str, bucket: str, job: PageJob) -> Path:
    return out_dir / stage_dir / phase / bucket / job.volume_id / f"page_{job.page_num:04d}.json"


def joined_output_path(out_dir: Path, bucket: str, job: PageJob) -> Path:
    return out_dir / "joined" / bucket / job.volume_id / f"page_{job.page_num:04d}.json"


def review_output_path(out_dir: Path, bucket: str, job: PageJob) -> Path:
    return out_dir / "review" / bucket / job.volume_id / f"page_{job.page_num:04d}.txt"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_entity_record(out_dir: Path, job: PageJob) -> Optional[dict[str, Any]]:
    path = stage_output_path(out_dir, "entity_spans", "postprocessed", "valid", job)
    if not path.exists():
        return None
    return load_json(path)


def load_done_page_job_ids(out_dir: Path, stage: str) -> set[str]:
    done: set[str] = set()
    if stage == "entities":
        roots = [out_dir / "entity_spans" / "postprocessed"]
    elif stage == "relations":
        roots = [out_dir / "relations" / "postprocessed"]
    else:
        roots = [out_dir / "joined"]
    for root in roots:
        for bucket in ("valid", "invalid", "errors"):
            for path in (root / bucket).glob("konbaung_vol*/page_*.json"):
                try:
                    obj = load_json(path)
                except (OSError, json.JSONDecodeError):
                    continue
                if obj.get("job_id"):
                    done.add(obj["job_id"])
    return done


def usage_dict(response: Any) -> dict[str, Any]:
    if getattr(response, "usage_metadata", None) is None:
        return {}
    raw_usage = response.usage_metadata
    if hasattr(raw_usage, "model_dump"):
        return raw_usage.model_dump()
    return dict(raw_usage)


def call_gemini_structured(
    client: genai.Client,
    *,
    model: str,
    prompt: str,
    response_schema: type[T],
    max_output_tokens: int,
    max_retries: int,
    retry_sleep: float,
) -> tuple[T, dict[str, Any]]:
    last_err: Optional[Exception] = None
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
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
            usage = usage_dict(response)
            if getattr(response, "parsed", None) is not None:
                return response.parsed, usage
            return response_schema.model_validate_json(response.text), usage
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


def write_entity_records(
    out_dir: Path,
    job: PageJob,
    annotation: SpanAnnotation,
    resolved_annotation: dict[str, Any],
    validation_errors: list[str],
    validation_warnings: list[str],
    usage: dict[str, Any],
    prompt_path: Path,
) -> tuple[str, dict[str, Any]]:
    bucket = "invalid" if validation_errors else "valid"
    base = {
        **page_metadata(job),
        "stage": "entity_spans",
        "usage_metadata": usage,
        "prompt_path": str(prompt_path),
    }
    raw_record = {
        **base,
        "annotation": annotation.model_dump(),
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
    }
    post_record = {
        **base,
        "annotation": resolved_annotation,
        "model_annotation": annotation.model_dump(),
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
    }
    write_json(stage_output_path(out_dir, "entity_spans", "raw", bucket, job), raw_record)
    write_json(
        stage_output_path(out_dir, "entity_spans", "postprocessed", bucket, job), post_record
    )
    return bucket, post_record


def write_relation_records(
    out_dir: Path,
    job: PageJob,
    annotation: RelationAnnotation,
    resolved_annotation: dict[str, Any],
    validation_errors: list[str],
    validation_warnings: list[str],
    usage: dict[str, Any],
    prompt_path: Path,
    entity_record: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    bucket = "invalid" if validation_errors else "valid"
    base = {
        **page_metadata(job),
        "stage": "relations",
        "usage_metadata": usage,
        "prompt_path": str(prompt_path),
        "entity_spans_path": str(
            stage_output_path(out_dir, "entity_spans", "postprocessed", "valid", job)
        ),
    }
    raw_record = {
        **base,
        "annotation": annotation.model_dump(),
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
    }
    post_record = {
        **base,
        "annotation": resolved_annotation,
        "model_annotation": annotation.model_dump(),
        "entity_spans": entity_record.get("annotation", {}).get("spans", []),
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
    }
    write_json(stage_output_path(out_dir, "relations", "raw", bucket, job), raw_record)
    write_json(stage_output_path(out_dir, "relations", "postprocessed", bucket, job), post_record)
    return bucket, post_record


def span_review_label(span: dict[str, Any]) -> str:
    return f"{span.get('id', '')}|{span.get('label', '')}"


def relation_review_line(relation: dict[str, Any], span_by_id: dict[str, dict[str, Any]]) -> str:
    source = span_by_id.get(str(relation.get("s")), {})
    target = span_by_id.get(str(relation.get("o")), {})
    source_text = str(source.get("tx", "")).replace("\n", " ")
    target_text = str(target.get("tx", "")).replace("\n", " ")
    return (
        f"{relation.get('id', '')} "
        f"{relation.get('s', '')}|{source.get('label', '')}:{source_text} "
        f"--{relation.get('p', '')}--> "
        f"{relation.get('o', '')}|{target.get('label', '')}:{target_text}"
    )


def annotate_line_with_spans(line: str, spans: list[dict[str, Any]]) -> str:
    opens: dict[int, list[dict[str, Any]]] = {}
    closes: dict[int, list[dict[str, Any]]] = {}
    for span in spans:
        start = span["s"]
        end = span["e"]
        opens.setdefault(start, []).append(span)
        closes.setdefault(end, []).append(span)

    parts: list[str] = []
    for pos in range(len(line) + 1):
        for span in sorted(closes.get(pos, []), key=lambda item: item["s"], reverse=True):
            parts.append("]")
        for span in sorted(opens.get(pos, []), key=lambda item: item["e"], reverse=True):
            parts.append(f"[{span_review_label(span)} ")
        if pos < len(line):
            parts.append(line[pos])
    return "".join(parts)


def build_review_text(job: PageJob, joined_record: dict[str, Any]) -> str:
    lines = page_lines(job.raw_text)
    line_spans: dict[int, list[dict[str, Any]]] = {}
    span_by_id = {str(span.get("id")): span for span in joined_record.get("entity_spans", [])}

    for span in joined_record.get("entity_spans", []):
        for part in span.get("spans", []):
            ln = part.get("ln")
            start = part.get("s")
            end = part.get("e")
            if not isinstance(ln, int) or not isinstance(start, int) or not isinstance(end, int):
                continue
            if ln < 1 or ln > len(lines) or start < 0 or end > len(lines[ln - 1]) or end <= start:
                continue
            line_spans.setdefault(ln, []).append(
                {
                    "id": span.get("id"),
                    "label": span.get("label"),
                    "s": start,
                    "e": end,
                }
            )

    relation_lines: dict[int, list[str]] = {}
    for relation in joined_record.get("relations", []):
        evidence = relation.get("evidence", {})
        evidence_spans = evidence.get("spans") or []
        if evidence_spans:
            ln = max(
                part.get("ln", 0) for part in evidence_spans if isinstance(part.get("ln"), int)
            )
        else:
            ln = evidence.get("ln")
        if not isinstance(ln, int) or ln < 1 or ln > len(lines):
            continue
        relation_lines.setdefault(ln, []).append(relation_review_line(relation, span_by_id))

    output: list[str] = []
    for ln, line in enumerate(lines, start=1):
        output.append(annotate_line_with_spans(line, line_spans.get(ln, [])))
        output.extend(relation_lines.get(ln, []))
    return "\n".join(output) + "\n"


def write_joined_record(
    out_dir: Path,
    job: PageJob,
    entity_record: Optional[dict[str, Any]],
    relation_record: Optional[dict[str, Any]],
    errors: list[str],
) -> None:
    bucket = "invalid" if errors else "valid"
    record = {
        **page_metadata(job),
        "schema": "xlmr_seed_two_stage",
        "entity_spans": entity_record.get("annotation", {}).get("spans", [])
        if entity_record
        else [],
        "relations": relation_record.get("annotation", {}).get("relations", [])
        if relation_record
        else [],
        "validation_errors": errors,
        "negative_relation_pairs": "not generated here; derive later from unlinked candidate span pairs",
    }
    write_json(joined_output_path(out_dir, bucket, job), record)
    write_text(review_output_path(out_dir, bucket, job), build_review_text(job, record))


def prepare_outputs(
    args: argparse.Namespace, jobs: list[PageJob], page_index: dict[tuple[str, int], PageJob]
) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    effective_jobs = jobs[: args.limit] if args.limit is not None else jobs
    write_jsonl(
        out_dir / "page_jobs.jsonl",
        [
            {
                "job_id": job.job_id,
                "volume_id": job.volume_id,
                "page_num": job.page_num,
                "source_path": job.source_path,
                "sha1": job.sha1,
                "char_count": job.char_count,
                "line_count": job.line_count,
            }
            for job in effective_jobs
        ],
    )

    entity_prompts_saved = 0
    relation_prompts_saved = 0
    if args.save_prompts:
        prompt_jobs = effective_jobs[: args.prompt_limit] if args.prompt_limit else effective_jobs
        if args.stage in ("entities", "both"):
            for job in prompt_jobs:
                write_text(
                    prompt_output_path(out_dir, "entity_spans", job),
                    build_entity_prompt(job, page_index, args.context_ratio, args.schema_detail),
                )
                entity_prompts_saved += 1
        if args.stage in ("relations", "both"):
            for job in prompt_jobs:
                entity_record = load_entity_record(out_dir, job)
                if not entity_record:
                    continue
                write_text(
                    prompt_output_path(out_dir, "relations", job),
                    build_relation_prompt(
                        job, page_index, args.context_ratio, args.schema_detail, entity_record
                    ),
                )
                relation_prompts_saved += 1

    by_volume: dict[str, int] = {}
    for job in effective_jobs:
        by_volume[job.volume_id] = by_volume.get(job.volume_id, 0) + 1

    summary = {
        "source_root": str(Path(args.source_root)),
        "out_dir": str(out_dir),
        "script": "pipeline/extraction/xlmr_seed_annotator.py",
        "schema": {
            "stage_1": "XLM-R entity/event span tagger seed data: spans[].id/label/ln/tx/i plus computed offsets",
            "stage_2": "XLM-R relation classifier seed data: relations[].id/s/p/o/ln/tx/i plus computed evidence offsets",
        },
        "output_layout": {
            "entity_spans": "entity_spans/raw|postprocessed/valid|invalid/<volume_id>/page_XXXX.json",
            "relations": "relations/raw|postprocessed/valid|invalid/<volume_id>/page_XXXX.json",
            "joined": "joined/valid|invalid/<volume_id>/page_XXXX.json",
            "review": "review/valid|invalid/<volume_id>/page_XXXX.txt",
            "prompts": "prompts_sent/entity_spans|relations/<volume_id>/page_XXXX.txt",
        },
        "model": args.model,
        "stage": args.stage,
        "schema_detail": args.schema_detail,
        "context_ratio_each_side": args.context_ratio,
        "max_entity_output_tokens": args.max_entity_output_tokens,
        "max_relation_output_tokens": args.max_relation_output_tokens,
        "jobs_prepared": len(effective_jobs),
        "jobs_by_volume": by_volume,
        "entity_prompts_saved": entity_prompts_saved,
        "relation_prompts_saved": relation_prompts_saved,
        "will_call_api": bool(args.run),
        "api_key_source": "not_resolved_in_prepare_mode"
        if not args.run
        else "resolved_at_run_time",
    }
    write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_entity_stage(
    args: argparse.Namespace,
    client: genai.Client,
    job: PageJob,
    page_index: dict[tuple[str, int], PageJob],
) -> tuple[str, Optional[dict[str, Any]]]:
    out_dir = Path(args.out_dir)
    prompt = build_entity_prompt(job, page_index, args.context_ratio, args.schema_detail)
    prompt_path = prompt_output_path(out_dir, "entity_spans", job)
    write_text(prompt_path, prompt)
    annotation, usage = call_gemini_structured(
        client,
        model=args.model,
        prompt=prompt,
        response_schema=SpanAnnotation,
        max_output_tokens=args.max_entity_output_tokens,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )
    resolved, errors, warnings = validate_span_annotation(job, annotation)
    return write_entity_records(
        out_dir, job, annotation, resolved, errors, warnings, usage, prompt_path
    )


def run_relation_stage(
    args: argparse.Namespace,
    client: genai.Client,
    job: PageJob,
    page_index: dict[tuple[str, int], PageJob],
    entity_record: dict[str, Any],
) -> tuple[str, Optional[dict[str, Any]]]:
    out_dir = Path(args.out_dir)
    prompt = build_relation_prompt(
        job, page_index, args.context_ratio, args.schema_detail, entity_record
    )
    prompt_path = prompt_output_path(out_dir, "relations", job)
    write_text(prompt_path, prompt)
    annotation, usage = call_gemini_structured(
        client,
        model=args.model,
        prompt=prompt,
        response_schema=RelationAnnotation,
        max_output_tokens=args.max_relation_output_tokens,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )
    resolved, errors, warnings = validate_relation_annotation(job, annotation, entity_record)
    return write_relation_records(
        out_dir, job, annotation, resolved, errors, warnings, usage, prompt_path, entity_record
    )


def write_error_record(out_dir: Path, job: PageJob, stage_dir: str, exc: Exception) -> None:
    write_json(
        stage_output_path(out_dir, stage_dir, "postprocessed", "errors", job),
        {
            **page_metadata(job),
            "stage": stage_dir,
            "error_type": type(exc).__name__,
            "error": str(exc),
        },
    )


def run_annotation(
    args: argparse.Namespace, jobs: list[PageJob], page_index: dict[tuple[str, int], PageJob]
) -> None:
    api_key = resolve_api_key(args)
    if not api_key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY, pass --api-key, or use --env-file."
        )

    out_dir = Path(args.out_dir)
    done = load_done_page_job_ids(out_dir, args.stage) if args.resume else set()
    client = genai.Client(api_key=api_key)
    effective_jobs = jobs[: args.limit] if args.limit is not None else jobs

    for job in tqdm(effective_jobs, desc=f"Running {args.stage}"):
        if job.job_id in done:
            continue
        entity_record: Optional[dict[str, Any]] = None
        relation_record: Optional[dict[str, Any]] = None
        joined_errors: list[str] = []
        try:
            if args.stage in ("entities", "both"):
                entity_bucket, entity_record = run_entity_stage(args, client, job, page_index)
                if entity_bucket != "valid":
                    joined_errors.append("entity span stage failed validation")
                    if args.stage == "both":
                        write_joined_record(out_dir, job, entity_record, None, joined_errors)
                    continue
            else:
                entity_record = load_entity_record(out_dir, job)
                if entity_record is None:
                    raise RuntimeError("No valid entity span record found for relation stage")

            if args.stage in ("relations", "both"):
                relation_bucket, relation_record = run_relation_stage(
                    args, client, job, page_index, entity_record
                )
                if relation_bucket != "valid":
                    joined_errors.append("relation stage failed validation")

            if args.stage == "both":
                write_joined_record(out_dir, job, entity_record, relation_record, joined_errors)
        except Exception as exc:
            stage_dir = (
                "entity_spans"
                if args.stage in ("entities", "both") and entity_record is None
                else "relations"
            )
            write_error_record(out_dir, job, stage_dir, exc)
            if args.stage == "both":
                write_joined_record(
                    out_dir, job, entity_record, relation_record, [f"{type(exc).__name__}: {exc}"]
                )
            if not args.continue_on_error:
                raise


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or run two-stage Gemini seed-data annotation for XLM-R entity and relation models."
    )
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Root containing konbaung_vol*/pages/page_*.txt.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for prompts and annotations.",
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
    parser.add_argument("--stage", choices=("entities", "relations", "both"), default="both")
    parser.add_argument(
        "--volume-id", default=None, help="Optional single volume selector, e.g. konbaung_vol1."
    )
    parser.add_argument(
        "--page-num", type=int, default=None, help="Optional single OCR page selector, e.g. 177."
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit jobs for prepare/run.")
    parser.add_argument(
        "--save-prompts", action="store_true", help="Write prompt files without calling Gemini."
    )
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=25,
        help="Max prompts to save when --save-prompts is used. Use 0 for all.",
    )
    parser.add_argument(
        "--context-ratio",
        type=float,
        default=DEFAULT_CONTEXT_RATIO,
        help="Previous/next context as a ratio of target page chars.",
    )
    parser.add_argument(
        "--schema-detail",
        choices=("full", "names", "none"),
        default="full",
        help="How much label information to print in prompts. Structured schema still enforces labels.",
    )
    parser.add_argument(
        "--max-entity-output-tokens",
        type=int,
        default=DEFAULT_MAX_ENTITY_OUTPUT_TOKENS,
        help="Output cap for the entity span call. Intended only to catch runaway outputs.",
    )
    parser.add_argument(
        "--max-relation-output-tokens",
        type=int,
        default=DEFAULT_MAX_RELATION_OUTPUT_TOKENS,
        help="Output cap for the relation call. Intended only to catch runaway outputs.",
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
    jobs = build_selected_jobs(args, source_root)
    if not jobs:
        raise RuntimeError(f"No page jobs found under {source_root}")
    page_index = build_page_index_with_neighbors(source_root, jobs)
    prepare_outputs(args, jobs, page_index)
    if args.run:
        run_annotation(args, jobs, page_index)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Gemini KG annotator for manually selected Konbaung per-page OCR files.



Default behavior is prepare-only. The script will not call Gemini unless

--run is supplied.

"""

from __future__ import annotations


import argparse

import hashlib

import json

import os

import random

import time

from dataclasses import dataclass

from pathlib import Path

from typing import Any, Iterable, Optional


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


DEFAULT_SOURCE_ROOT = Path("konbaung_viable_pages_manual_ranges")

DEFAULT_OUT_DIR = Path("konbaung_page_annotations_ready")

DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

DEFAULT_MAX_OUTPUT_TOKENS = 20000


@dataclass
class PageJob:
    job_id: str

    volume_id: str

    page_num: int

    source_path: str

    raw_text: str

    sha1: str

    char_count: int

    line_count: int


class Mention(BaseModel):
    ln: int = Field(description="1-based line number in <page_text>.")

    tx: str = Field(description="Exact original text substring from that line.")

    i: int = Field(
        default=1, description="1-based occurrence of tx on this line; use >1 only when duplicated."
    )


class Prop(BaseModel):
    k: str = Field(description="Property key.")

    v: str = Field(description="Property value as page-grounded text.")


class NodeOut(BaseModel):
    id: str = Field(description="Local node ID: n1, n2, n3...")

    n: str = Field(description="Best exact Burmese text name for this node from the page.")

    l: list[EntityType] = Field(description="One or more existing entity labels.")

    m: list[Mention] = Field(description="Exact page mentions of this node.")

    props: list[Prop] = Field(description="Open node properties; use [] when none.")


class RelOut(BaseModel):
    s: str = Field(description="Source node ID.")

    o: str = Field(description="Target node ID.")

    t: RelationType = Field(description="Exactly one existing relationship type.")

    m: list[Mention] = Field(description="Exact page evidence mentions for this relationship.")

    props: list[Prop] = Field(description="Open relationship properties; use [] when none.")


class KGAnnotation(BaseModel):
    N: list[NodeOut] = Field(description="Nodes.")

    R: list[RelOut] = Field(description="Directed relationships.")


ENTITY_LABELS = ", ".join(ENTITY_DEFINITIONS.keys())

RELATION_TYPES = ", ".join(RELATION_DEFINITIONS.keys())


PROMPT_TEMPLATE = """Annotate one raw OCR page from a Burmese royal chronicle for a Neo4j property graph.



The text is line-numbered as: LINE<TAB>TEXT.



Task:

- Extract only the main narrative spine: important entities, events, and relation-bearing facts needed to reconstruct the page's historical narrative.

- Do not tag every noun, phrase, epithet, or grammatical role word.

- Skip page numbers, running headers, publisher footers, footnote/date lines, OCR debris, broken line wrapping, and printer/image noise.

- Skip generic kinship words, status words, adjectives, ordinals, title fragments, and formulaic genealogy phrases unless they identify a specific person, group, office, place, date, quantity, object, text, or event.

- Merge repeated mentions of the same local entity into one node and list every useful mention in m.

- Put aliases, titles, ranks, lineage strings, uncertainty, OCR notes, normalized values, and minor qualifiers in props instead of making extra nodes unless they are standalone narrative entities.



Output JSON:

- Top-level keys N and R are required arrays. Use [] if none.

- N: nodes. Each node requires id, n, l, m, props.

- R: directed relationships. Each relationship requires s, o, t, m, props.

- Node n is the best exact Burmese page-text name for the local node.

- Node l is one or more labels from ENTITY_LABELS. Do not invent labels.

- Relationship t is exactly one type from RELATION_TYPES. Do not invent types.

- Relationship direction is s -> o.

- props is always required: [{{"k":"property_name","v":"page-grounded value"}}], or [] when no useful properties exist.

- m is one or more exact mentions: {{"ln": line_number, "tx": "exact original text on that line", "i": 1}}.

- Do not output character offsets; Python will compute offsets from m.tx after validation.

- If a node or relation evidence wraps across OCR lines, use multiple m entries in reading order.

- Use i > 1 only when the same tx appears more than once on the same line.



ENTITY_LABELS: {entity_labels}



RELATION_TYPES: {relation_types}



<metadata>

job_id: {job_id}

volume_id: {volume_id}

page_num: {page_num}

</metadata>



<page_text>

{raw_text}

</page_text>

"""


def read_env_file(path: Path) -> dict[str, str]:

    values: dict[str, str] = {}

    if not path.exists():
        return values

    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)

        values[key.strip()] = value.strip().strip("\"'")

    return values


def resolve_api_key(args: argparse.Namespace) -> str:

    if args.api_key:
        return args.api_key

    if os.getenv("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"].strip()

    if args.env_file:
        values = read_env_file(Path(args.env_file))

        if values.get("GEMINI_API_KEY"):
            return values["GEMINI_API_KEY"].strip()

    return ""


def sha1_text(text: str) -> str:

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def page_lines(raw_text: str) -> list[str]:

    return raw_text.splitlines()


def line_numbered_text(raw_text: str) -> str:

    return "\n".join(f"{idx:04d}\t{line}" for idx, line in enumerate(page_lines(raw_text), start=1))


def iter_page_files(source_root: Path) -> Iterable[Path]:

    for vol_dir in sorted(source_root.glob("konbaung_vol*/pages")):
        for path in sorted(vol_dir.glob("page_*.txt")):
            yield path


def build_jobs(source_root: Path) -> list[PageJob]:

    jobs: list[PageJob] = []

    for path in iter_page_files(source_root):
        volume_id = path.parent.parent.name

        page_num = int(path.stem.split("_", 1)[1])

        raw_text = path.read_text(encoding="utf-8", errors="replace")

        digest = sha1_text(raw_text)

        jobs.append(
            PageJob(
                job_id=f"{volume_id}_p{page_num:04d}_{digest[:8]}",
                volume_id=volume_id,
                page_num=page_num,
                source_path=str(path),
                raw_text=raw_text,
                sha1=digest,
                char_count=len(raw_text),
                line_count=len(page_lines(raw_text)),
            )
        )

    return jobs


def build_prompt(job: PageJob) -> str:

    return PROMPT_TEMPLATE.format(
        entity_labels=ENTITY_LABELS,
        relation_types=RELATION_TYPES,
        job_id=job.job_id,
        volume_id=job.volume_id,
        page_num=f"{job.page_num:04d}",
        raw_text=line_numbered_text(job.raw_text),
    )


def write_jsonl(path: Path, records: Iterable[dict]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, record: dict) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def page_output_path(out_dir: Path, bucket: str, job: PageJob) -> Path:

    return out_dir / bucket / job.volume_id / f"page_{job.page_num:04d}.json"


def load_done_job_ids_from_jsonl(path: Path) -> set[str]:

    if not path.exists():
        return set()

    done: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)

            except json.JSONDecodeError:
                continue

            if obj.get("job_id"):
                done.add(obj["job_id"])

            elif obj.get("chunk_id"):
                done.add(obj["chunk_id"])

    return done


def load_done_page_job_ids(out_dir: Path) -> set[str]:

    done: set[str] = set()

    for bucket in ("valid", "invalid"):
        for path in (out_dir / bucket).glob("konbaung_vol*/page_*.json"):
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))

            except (OSError, json.JSONDecodeError):
                continue

            if obj.get("job_id"):
                done.add(obj["job_id"])

    done.update(load_done_job_ids_from_jsonl(out_dir / "annotations.raw.jsonl"))

    done.update(load_done_job_ids_from_jsonl(out_dir / "annotations.invalid.jsonl"))

    return done


def resolve_mention(
    lines: list[str], mention: Mention
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
            return None, f"mention text not found on line {mention.ln}: {mention.tx!r}"

        search_from = start + len(mention.tx)

    end = start + len(mention.tx)

    return {
        "ln": mention.ln,
        "tx": mention.tx,
        "i": mention.i,
        "s": start,
        "e": end,
    }, None


def validate_properties(owner: str, props: list[Prop]) -> list[str]:

    errors: list[str] = []

    seen: set[str] = set()

    for prop in props:
        key = prop.k.strip()

        if not key:
            errors.append(f"{owner}: blank/non-string property key")

        if key in seen:
            errors.append(f"{owner}.{key}: duplicate property key")

        seen.add(key)

    return errors


def resolve_annotation(job: PageJob, annotation: KGAnnotation) -> tuple[dict[str, Any], list[str]]:

    errors: list[str] = []

    lines = page_lines(job.raw_text)

    resolved: dict[str, Any] = {"N": [], "R": []}

    node_ids = [node.id for node in annotation.N]

    node_id_set = set(node_ids)

    if len(node_ids) != len(node_id_set):
        errors.append("duplicate node id")

    for node in annotation.N:
        resolved_mentions: list[dict[str, Any]] = []

        if not node.l:
            errors.append(f"{node.id}: no labels")

        if not node.n.strip():
            errors.append(f"{node.id}: blank node text")

        if not node.m:
            errors.append(f"{node.id}: no mentions")

        for mention in node.m:
            resolved_mention, error = resolve_mention(lines, mention)

            if error:
                errors.append(f"{node.id}: {error}")

                resolved_mentions.append(mention.model_dump())

            elif resolved_mention is not None:
                resolved_mentions.append(resolved_mention)

        errors.extend(validate_properties(node.id, node.props))

        node_data = node.model_dump()

        node_data["m"] = resolved_mentions

        resolved["N"].append(node_data)

    for idx, rel in enumerate(annotation.R, start=1):
        rel_id = f"R[{idx}]"

        resolved_mentions = []

        if rel.s not in node_id_set:
            errors.append(f"{rel_id}: missing source node {rel.s}")

        if rel.o not in node_id_set:
            errors.append(f"{rel_id}: missing target node {rel.o}")

        if not rel.m:
            errors.append(f"{rel_id}: no mentions")

        for mention in rel.m:
            resolved_mention, error = resolve_mention(lines, mention)

            if error:
                errors.append(f"{rel_id}: {error}")

                resolved_mentions.append(mention.model_dump())

            elif resolved_mention is not None:
                resolved_mentions.append(resolved_mention)

        errors.extend(validate_properties(rel_id, rel.props))

        rel_data = rel.model_dump()

        rel_data["m"] = resolved_mentions

        resolved["R"].append(rel_data)

    return resolved, errors


def call_gemini_structured(
    client: genai.Client,
    *,
    model: str,
    prompt: str,
    max_output_tokens: int,
    max_retries: int,
    retry_sleep: float,
) -> tuple[KGAnnotation, dict[str, Any]]:

    last_err: Optional[Exception] = None

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=KGAnnotation,
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

            return KGAnnotation.model_validate_json(response.text), usage

        except (ValidationError, Exception) as exc:
            last_err = exc

            if attempt >= max_retries:
                break

            time.sleep(retry_sleep * (2**attempt) + random.random())

    raise RuntimeError(f"Gemini call failed after retries: {last_err}")


def prepare_outputs(args: argparse.Namespace, jobs: list[PageJob]) -> None:

    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

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
            for job in jobs
        ],
    )

    if args.save_prompts:
        prompt_dir = out_dir / "debug_prompts"

        prompt_jobs = jobs[: args.prompt_limit] if args.prompt_limit else jobs

        for job in prompt_jobs:
            prompt_path = prompt_dir / job.volume_id / f"page_{job.page_num:04d}.txt"

            prompt_path.parent.mkdir(parents=True, exist_ok=True)

            prompt_path.write_text(build_prompt(job), encoding="utf-8", newline="\n")

    by_volume: dict[str, int] = {}

    for job in jobs:
        by_volume[job.volume_id] = by_volume.get(job.volume_id, 0) + 1

    summary = {
        "source_root": str(Path(args.source_root)),
        "out_dir": str(out_dir),
        "annotation_output_layout": "valid/<volume_id>/page_XXXX.json and invalid/<volume_id>/page_XXXX.json",
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "jobs_prepared": len(jobs),
        "jobs_by_volume": by_volume,
        "will_call_api": bool(args.run),
        "api_key_source": "not_resolved_in_prepare_mode"
        if not args.run
        else "resolved_at_run_time",
    }

    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_annotation(args: argparse.Namespace, jobs: list[PageJob]) -> None:

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

    for job in tqdm(jobs, desc="Annotating pages"):
        if job.job_id in done:
            continue

        try:
            annotation, usage = call_gemini_structured(
                client,
                model=args.model,
                prompt=build_prompt(job),
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )

            resolved_annotation, validation_errors = resolve_annotation(job, annotation)

            record = {
                "job_id": job.job_id,
                "volume_id": job.volume_id,
                "page_num": job.page_num,
                "source_path": job.source_path,
                "source_sha1": job.sha1,
                "char_count": job.char_count,
                "line_count": job.line_count,
                "usage_metadata": usage,
                "annotation": resolved_annotation,
                "model_annotation": annotation.model_dump(),
            }

            if validation_errors:
                record["validation_errors"] = validation_errors

                write_json(page_output_path(out_dir, "invalid", job), record)

            else:
                write_json(page_output_path(out_dir, "valid", job), record)

        except Exception as exc:
            write_json(
                page_output_path(out_dir, "errors", job),
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

    parser = argparse.ArgumentParser(
        description="Prepare or run Gemini KG annotation over copied Konbaung per-page OCR."
    )

    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Root containing konbaung_vol*/pages/page_*.txt.",
    )

    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for jobs and annotations."
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

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Output cap. Default is a single-page guardrail, intended only to catch runaway outputs.",
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

    jobs = build_jobs(source_root)

    if not jobs:
        raise RuntimeError(f"No page jobs found under {source_root}")

    prepare_outputs(args, jobs)

    if args.run:
        run_annotation(args, jobs)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cached Gemini Batch API runner for final quantitative fourth-pass annotations."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from google import genai
from google.genai import types
from pydantic import ValidationError

from konbaung_gemini_page_kg_annotator import DEFAULT_ENV_FILE, DEFAULT_SOURCE_ROOT, PageJob, resolve_api_key
from konbaung_gemini_structured_open_coding_annotator import page_metadata, usage_dict
from konbaung_gemini_cite_sources_annotator import ORIGINAL_PROMPT_PATH
from konbaung_gemini_quantitative_fourth_pass import (
    FOURTH_PASS_PROMPT,
    FourthPassResult,
    READER_DATA,
    build_prompt as build_single_prompt,
    compact_existing,
    coverage_record,
    marked_page_text,
    public_coverage,
    uncovered_intervals,
)
from konbaung_gemini_xlmr_seed_annotator import page_job_from_path, write_json, write_text


DEFAULT_OUT_DIR = Path("konbaung_quantitative_fourth_pass_full_batch_20260712")
DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_MAX_OUTPUT_TOKENS = 10000
DEFAULT_COVERAGE_FILE = Path("konbaung_fourth_pass_below_80pct.json")
DEFAULT_MANIFEST_FILE = Path("konbaung_summary_support_run_manifest.json")
TERMINAL_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"}


def utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def static_cached_prefix() -> str:
    original = ORIGINAL_PROMPT_PATH.read_text(encoding="utf-8-sig")
    return (
        "<ORIGINAL_FIRST_PASS_PROMPT_READ_ONLY>\n"
        f"{original}\n"
        "</ORIGINAL_FIRST_PASS_PROMPT_READ_ONLY>\n\n"
        f"{FOURTH_PASS_PROMPT}\n"
    )


def json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def job_key(job: PageJob) -> str:
    return f"{job.volume_id}_p{job.page_num:04d}_{job.sha1[:10]}"


def completed_final_pages(manifest_path: Path) -> set[tuple[int, int]]:
    return {(2, 27), (3, 169)}


def target_pages(coverage_path: Path, manifest_path: Path) -> set[tuple[int, int]]:
    coverage = load_json(coverage_path)
    completed = completed_final_pages(manifest_path)
    return {
        (int(row["volume"]), int(row["page"]))
        for row in coverage.get("pages", [])
        if (int(row["volume"]), int(row["page"])) not in completed
    }


def all_page_jobs(source_root: Path, volume_id: str, args: argparse.Namespace) -> list[PageJob]:
    page_dir = source_root / volume_id / "pages"
    paths = sorted(page_dir.glob("page_*.txt"))
    jobs = [page_job_from_path(path) for path in paths]
    volume = int(volume_id.removeprefix("konbaung_vol"))
    targets = target_pages(Path(args.coverage_file), Path(args.manifest_file))
    return [job for job in jobs if (volume, job.page_num) in targets]


def first_volume_pages(source_root: Path, volume_id: str, count: int, args: argparse.Namespace) -> list[PageJob]:
    return all_page_jobs(source_root, volume_id, args)[:count]


def dynamic_page_block(job: PageJob) -> str:
    volume = int(job.volume_id.removeprefix("konbaung_vol"))
    page = load_json(READER_DATA / f"vol{volume}" / f"{job.page_num:04d}.json")
    triples = compact_existing(page)
    coverage = coverage_record(page)
    intervals = uncovered_intervals(page, coverage)
    return (
        "Now perform NEW_FOURTH_PASS_TASK using only the structured schema.\n\n"
        "<metadata>\n"
        f"job_id: {job.job_id}\n"
        f"volume_id: {job.volume_id}\n"
        f"page_num: {job.page_num:04d}\n"
        "</metadata>\n\n"
        "<EXISTING_SUMMARY_READ_ONLY>\n"
        f"{page.get('summary', '')}\n"
        "</EXISTING_SUMMARY_READ_ONLY>\n\n"
        "<EXISTING_TRIPLES_READ_ONLY>\n"
        f"{json.dumps(triples, ensure_ascii=False, indent=2)}\n"
        "</EXISTING_TRIPLES_READ_ONLY>\n\n"
        "<PAGE_COVERAGE>\n"
        f"Current evidence coverage: {coverage['coverage_pct']:.2f}%\n"
        f"Uncovered text: {coverage['unannotated_pct']:.2f}% "
        f"({coverage['unannotated_characters']} of {coverage['retained_characters']} retained characters)\n"
        "</PAGE_COVERAGE>\n\n"
        "<TARGET_PAGE_TEXT_WITH_MARKERS>\n"
        f"{marked_page_text(page, intervals)}\n"
        "</TARGET_PAGE_TEXT_WITH_MARKERS>\n"
    )


def prompt_record_text(static_prefix: str, job: PageJob) -> str:
    return (
        "<CACHED_STATIC_PREFIX>\n"
        f"{static_prefix}\n"
        "</CACHED_STATIC_PREFIX>\n\n"
        f"{dynamic_page_block(job)}"
    )


def build_request(job: PageJob, cache_name: str, max_output_tokens: int) -> types.InlinedRequest:
    config = types.GenerateContentConfig(
        cached_content=cache_name,
        response_mime_type="application/json",
        response_schema=FourthPassResult,
        temperature=0.0,
        candidate_count=1,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    return types.InlinedRequest(
        contents=[
            types.Content(role="user", parts=[types.Part(text=dynamic_page_block(job))]),
        ],
        metadata={
            "key": job_key(job),
            "volume_id": job.volume_id,
            "page_num": f"{job.page_num:04d}",
            "sha1": job.sha1,
        },
        config=config,
    )


def batch_state_name(state: Any) -> str:
    if state is None:
        return ""
    return getattr(state, "value", str(state))


def token_count_total(response: Any) -> Optional[int]:
    if response is None:
        return None
    value = getattr(response, "total_tokens", None)
    if value is not None:
        return int(value)
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        for key in ("total_tokens", "totalTokens"):
            if dumped.get(key) is not None:
                return int(dumped[key])
    return None


def create_client(args: argparse.Namespace) -> genai.Client:
    api_key = resolve_api_key(args)
    if not api_key:
        raise RuntimeError("No Gemini API key found. Set GEMINI_API_KEY, pass --api-key, or use --env-file.")
    return genai.Client(api_key=api_key)


def create_cached_prefix(
    client: genai.Client,
    *,
    model: str,
    static_prefix: str,
    out_dir: Path,
    ttl_seconds: int,
) -> tuple[str, dict[str, Any]]:
    count_response = client.models.count_tokens(model=model, contents=static_prefix)
    prefix_tokens = token_count_total(count_response)
    cached = client.caches.create(
        model=model,
        config=types.CreateCachedContentConfig(
            display_name=f"konbaung_open_coding_prefix_{utc_stamp()}",
            contents=static_prefix,
            ttl=f"{ttl_seconds}s",
        ),
    )
    record = {
        "cache_name": cached.name,
        "model": model,
        "prefix_token_count": prefix_tokens,
        "ttl_seconds": ttl_seconds,
        "cached_content": json_safe(cached),
    }
    write_text(out_dir / "cached_prefix.txt", static_prefix)
    write_json(out_dir / "cache_record.json", record)
    return cached.name, record


def submit_batch(
    client: genai.Client,
    *,
    model: str,
    requests: list[types.InlinedRequest],
    out_dir: Path,
    label: str,
) -> Any:
    display_name = f"konbaung_{label}_{utc_stamp()}"
    batch = client.batches.create(
        model=model,
        src=requests,
        config=types.CreateBatchJobConfig(display_name=display_name),
    )
    write_json(out_dir / "batch_jobs" / f"{label}_submitted.json", json_safe(batch))
    return batch


def wait_for_batch(
    client: genai.Client,
    *,
    batch_name: str,
    out_dir: Path,
    label: str,
    poll_seconds: int,
    max_wait_seconds: int,
) -> Any:
    started = time.time()
    while True:
        job = client.batches.get(name=batch_name)
        write_json(out_dir / "batch_jobs" / f"{label}_latest.json", json_safe(job))
        state = batch_state_name(job.state)
        elapsed = int(time.time() - started)
        print(json.dumps({"label": label, "batch": batch_name, "state": state, "elapsed_seconds": elapsed}))
        if state in TERMINAL_STATES:
            return job
        if elapsed >= max_wait_seconds:
            raise TimeoutError(f"Batch {batch_name} did not finish within {max_wait_seconds} seconds")
        time.sleep(poll_seconds)


def response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text
    if not getattr(response, "candidates", None):
        return ""
    parts = []
    for candidate in response.candidates or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            if getattr(part, "text", None):
                parts.append(part.text)
    return "\n".join(parts)


def parse_annotation(response: Any) -> FourthPassResult:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return parsed if isinstance(parsed, FourthPassResult) else FourthPassResult.model_validate(parsed)
    text = response_text(response)
    return FourthPassResult.model_validate_json(text)


def output_path(out_dir: Path, phase: str, bucket: str, job: PageJob) -> Path:
    return out_dir / phase / bucket / job.volume_id / f"page_{job.page_num:04d}.json"


def validation_output_path(out_dir: Path, bucket: str, job: PageJob) -> Path:
    return out_dir / "validation" / bucket / job.volume_id / f"page_{job.page_num:04d}.json"


def prompt_output_path(out_dir: Path, job: PageJob) -> Path:
    return out_dir / "prompts_sent" / job.volume_id / f"page_{job.page_num:04d}.txt"


def normalize_whitespace(text: str) -> str:
    return "".join(text.split())


def validate_completion(annotation: FourthPassResult, page: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    target = normalize_whitespace(page["canonicalText"])
    existing = {
        (
            normalize_whitespace(item["subject"]["text"]),
            item["relation"]["rawLabel"],
            normalize_whitespace(item["object"]["text"]),
        )
        for item in page["annotations"]
    }
    for index, triple in enumerate(annotation.T, start=1):
        evidence = normalize_whitespace(triple.e)
        if not evidence or evidence not in target:
            warnings.append(f"T[{index}]: evidence does not resolve to TARGET_PAGE_TEXT")
        for field in ("d", "l", "q"):
            value = getattr(triple, field)
            if value and normalize_whitespace(value) not in target:
                warnings.append(f"T[{index}].{field}: value does not occur in TARGET_PAGE_TEXT")
        signature = (
            normalize_whitespace(triple.s),
            triple.p,
            normalize_whitespace(triple.o),
        )
        if signature in existing:
            warnings.append(f"T[{index}]: duplicate existing subject-relation-object signature")
    return warnings


def process_batch_outputs(
    *,
    batch_job: Any,
    jobs_by_key: dict[str, PageJob],
    static_prefix: str,
    out_dir: Path,
    label: str,
) -> dict[str, Any]:
    if batch_state_name(batch_job.state) != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch {label} did not succeed: {batch_state_name(batch_job.state)}")
    responses = getattr(getattr(batch_job, "dest", None), "inlined_responses", None) or []
    result_rows: list[dict[str, Any]] = []
    totals = {
        "responses_seen": 0,
        "response_errors": 0,
        "schema_parse_errors": 0,
        "valid_pages": 0,
        "invalid_pages": 0,
        "prompt_token_count": 0,
        "cached_content_token_count": 0,
        "candidates_token_count": 0,
        "total_token_count": 0,
    }
    completed_keys: list[str] = []

    for inlined in responses:
        totals["responses_seen"] += 1
        metadata = getattr(inlined, "metadata", None) or {}
        key = metadata.get("key")
        job = jobs_by_key.get(key or "")
        row: dict[str, Any] = {"key": key, "metadata": metadata}
        if job is None:
            totals["response_errors"] += 1
            row["error"] = f"Unknown batch response key: {key!r}"
            result_rows.append(row)
            continue
        if getattr(inlined, "error", None) is not None:
            totals["response_errors"] += 1
            row["error"] = json_safe(inlined.error)
            write_json(output_path(out_dir, "errors", "batch_error", job), row)
            result_rows.append(row)
            continue
        response = getattr(inlined, "response", None)
        if response is None:
            totals["response_errors"] += 1
            row["error"] = "Missing response"
            result_rows.append(row)
            continue

        usage = usage_dict(response)
        for usage_key in (
            "prompt_token_count",
            "cached_content_token_count",
            "candidates_token_count",
            "total_token_count",
        ):
            value = usage.get(usage_key)
            if isinstance(value, int):
                totals[usage_key] += value

        write_text(prompt_output_path(out_dir, job), prompt_record_text(static_prefix, job))
        try:
            annotation = parse_annotation(response)
        except (ValidationError, ValueError) as exc:
            totals["schema_parse_errors"] += 1
            row["error"] = f"Schema parse failed: {exc}"
            row["raw_text"] = response_text(response)
            write_json(output_path(out_dir, "raw", "schema_error", job), row)
            result_rows.append(row)
            continue

        volume = int(job.volume_id.removeprefix("konbaung_vol"))
        page = load_json(READER_DATA / f"vol{volume}" / f"{job.page_num:04d}.json")
        warnings = validate_completion(annotation, page)
        normalized = annotation.model_dump()
        bucket = "invalid" if warnings else "valid"
        if warnings:
            totals["invalid_pages"] += 1
        else:
            totals["valid_pages"] += 1
        completed_keys.append(key or "")
        validation_record = {
            **page_metadata(job),
            "stage": "quantitative_fourth_pass_batch",
            "batch_label": label,
            "batch_name": batch_job.name,
            "usage_metadata": usage,
            "prompt_path": str(prompt_output_path(out_dir, job)),
            "validation_warnings": warnings,
        }
        write_json(output_path(out_dir, "raw", bucket, job), annotation.model_dump())
        write_json(output_path(out_dir, "postprocessed", bucket, job), normalized)
        write_json(validation_output_path(out_dir, bucket, job), validation_record)
        row.update(
            {
                "volume_id": job.volume_id,
                "page_num": job.page_num,
                "bucket": bucket,
                "added_triple_count": len(normalized["T"]),
                "usage_metadata": usage,
                "validation_warning_count": len(warnings),
            }
        )
        result_rows.append(row)

    summary = {
        "label": label,
        "batch_name": batch_job.name,
        "batch_state": batch_state_name(batch_job.state),
        "completion_stats": json_safe(getattr(batch_job, "completion_stats", None)),
        "totals": totals,
        "completed_keys": completed_keys,
        "results": result_rows,
    }
    write_json(out_dir / "batch_results" / f"{label}_summary.json", summary)
    write_text(
        out_dir / "batch_results" / f"{label}_results.jsonl",
        "\n".join(json.dumps(row, ensure_ascii=False) for row in result_rows) + "\n",
    )
    return summary


def save_batch_plan(out_dir: Path, label: str, jobs: Iterable[PageJob]) -> dict[str, Any]:
    pages = [
        {
            "key": job_key(job),
            "volume_id": job.volume_id,
            "page_num": job.page_num,
            "source_path": job.source_path,
            "sha1": job.sha1,
            "char_count": job.char_count,
            "line_count": job.line_count,
        }
        for job in jobs
    ]
    plan = {"label": label, "page_count": len(pages), "pages": pages}
    write_json(out_dir / "batch_plans" / f"{label}.json", plan)
    return plan


def load_jobs_from_plan(out_dir: Path, label: str) -> list[PageJob]:
    plan_path = out_dir / "batch_plans" / f"{label}.json"
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)
    plan = load_json(plan_path)
    return [page_job_from_path(Path(row["source_path"])) for row in plan["pages"]]


def load_submitted_batch_name(out_dir: Path, label: str) -> str:
    submitted_path = out_dir / "batch_jobs" / f"{label}_submitted.json"
    if not submitted_path.exists():
        raise FileNotFoundError(submitted_path)
    submitted = load_json(submitted_path)
    return submitted["name"]


def require_smoke_success(smoke_summary: dict[str, Any], expected_count: int) -> None:
    totals = smoke_summary["totals"]
    if totals["responses_seen"] != expected_count:
        raise RuntimeError(f"Smoke response count mismatch: {totals['responses_seen']} != {expected_count}")
    if totals["response_errors"] or totals["schema_parse_errors"]:
        raise RuntimeError(f"Smoke had response/schema errors: {totals}")
    if totals["cached_content_token_count"] <= 0:
        raise RuntimeError(f"Smoke had no cached_content_token_count; refusing full run: {totals}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_cache_name(args: argparse.Namespace, out_dir: Path) -> str:
    if args.cache_name:
        return args.cache_name
    cache_record_path = out_dir / "cache_record.json"
    if not cache_record_path.exists():
        raise FileNotFoundError(f"No cache record found at {cache_record_path}; pass --cache-name or run smoke first.")
    cache_record = load_json(cache_record_path)
    return cache_record["cache_name"]


def load_smoke_keys(out_dir: Path) -> set[str]:
    smoke_path = out_dir / "batch_results" / "smoke_vol1_first_pages_summary.json"
    if not smoke_path.exists():
        raise FileNotFoundError(f"No smoke summary found at {smoke_path}; run smoke first.")
    smoke_summary = load_json(smoke_path)
    require_smoke_success(smoke_summary, len(smoke_summary.get("completed_keys", [])))
    return set(smoke_summary["completed_keys"])


def submit_and_process_full(
    *,
    args: argparse.Namespace,
    client: genai.Client,
    static_prefix: str,
    cache_name: str,
    smoke_keys: set[str],
) -> list[dict[str, Any]]:
    source_root = Path(args.source_root)
    out_dir = Path(args.out_dir)
    full_summaries: list[dict[str, Any]] = []
    submitted_full_jobs: list[tuple[str, Any, list[PageJob]]] = []
    for volume_id in ("konbaung_vol1", "konbaung_vol2", "konbaung_vol3"):
        jobs = all_page_jobs(source_root, volume_id, args)
        if volume_id == "konbaung_vol1":
            jobs = [job for job in jobs if job_key(job) not in smoke_keys]
        label = f"full_{volume_id}"
        save_batch_plan(out_dir, label, jobs)
        requests = [build_request(job, cache_name, args.max_output_tokens) for job in jobs]
        submitted = submit_batch(client, model=args.model, requests=requests, out_dir=out_dir, label=label)
        submitted_full_jobs.append((label, submitted, jobs))

    if args.submit_only:
        submit_summary = {
            "submitted_only": True,
            "jobs": [
                {"label": label, "batch_name": submitted.name, "page_count": len(jobs)}
                for label, submitted, jobs in submitted_full_jobs
            ],
        }
        write_json(out_dir / "batch_jobs" / "full_submitted_summary.json", submit_summary)
        print(json.dumps(submit_summary, ensure_ascii=False, indent=2))
        return []

    for label, submitted, jobs in submitted_full_jobs:
        done = wait_for_batch(
            client,
            batch_name=submitted.name,
            out_dir=out_dir,
            label=label,
            poll_seconds=args.poll_seconds,
            max_wait_seconds=args.full_max_wait_seconds,
        )
        full_summaries.append(
            process_batch_outputs(
                batch_job=done,
                jobs_by_key={job_key(job): job for job in jobs},
                static_prefix=static_prefix,
                out_dir=out_dir,
                label=label,
            )
        )
    return full_summaries


def process_existing_full(
    *,
    args: argparse.Namespace,
    client: genai.Client,
    static_prefix: str,
) -> list[dict[str, Any]]:
    out_dir = Path(args.out_dir)
    full_summaries: list[dict[str, Any]] = []
    for label in ("full_konbaung_vol1", "full_konbaung_vol2", "full_konbaung_vol3"):
        jobs = load_jobs_from_plan(out_dir, label)
        batch_name = load_submitted_batch_name(out_dir, label)
        done = wait_for_batch(
            client,
            batch_name=batch_name,
            out_dir=out_dir,
            label=label,
            poll_seconds=args.poll_seconds,
            max_wait_seconds=args.full_max_wait_seconds,
        )
        full_summaries.append(
            process_batch_outputs(
                batch_job=done,
                jobs_by_key={job_key(job): job for job in jobs},
                static_prefix=static_prefix,
                out_dir=out_dir,
                label=label,
            )
        )
    return full_summaries


def run_smoke_and_full(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    static_prefix = static_cached_prefix()

    if args.prepare_only:
        write_text(out_dir / "cached_prefix_preview.txt", static_prefix)
        smoke_jobs = first_volume_pages(source_root, "konbaung_vol1", args.smoke_pages, args)
        smoke_keys = {job_key(job) for job in smoke_jobs}
        plans = [save_batch_plan(out_dir, "smoke_vol1_first_pages", smoke_jobs)]
        for volume_id in ("konbaung_vol1", "konbaung_vol2", "konbaung_vol3"):
            jobs = all_page_jobs(source_root, volume_id, args)
            if volume_id == "konbaung_vol1":
                jobs = [job for job in jobs if job_key(job) not in smoke_keys]
            plans.append(save_batch_plan(out_dir, f"full_{volume_id}", jobs))
        readiness = {
            "prepared_only": True,
            "model": args.model,
            "source_root": str(source_root),
            "first_pass_prompt": str(ORIGINAL_PROMPT_PATH),
            "cached_prefix_preview": str(out_dir / "cached_prefix_preview.txt"),
            "schema": "T=e,s,sg,st,p,o,og,ot,d,dg,l,lg,q,qg",
            "plans": [{"label": plan["label"], "page_count": plan["page_count"]} for plan in plans],
            "execution_command": (
                f'python {Path(__file__).name} --execute --stop-after-smoke --out-dir "{out_dir}"'
            ),
            "full_after_smoke_command": (
                f'python {Path(__file__).name} --execute --full-only --out-dir "{out_dir}"'
            ),
        }
        write_json(out_dir / "readiness.json", readiness)
        print(json.dumps(readiness, ensure_ascii=False, indent=2))
        return

    if not args.execute:
        raise RuntimeError("Refusing API work without --execute. Use --prepare-only to inspect plans.")

    client = create_client(args)

    if args.process_existing_full:
        full_summaries = process_existing_full(args=args, client=client, static_prefix=static_prefix)
        aggregate = {
            "model": args.model,
            "source_root": str(source_root),
            "first_pass_prompt": str(ORIGINAL_PROMPT_PATH),
            "schema": "T=e,s,sg,st,p,o,og,ot,d,dg,l,lg,q,qg",
            "full": full_summaries,
        }
        write_json(out_dir / "run_summary.json", aggregate)
        print(json.dumps(aggregate, ensure_ascii=False, indent=2))
        return

    if args.full_only:
        cache_name = load_cache_name(args, out_dir)
        smoke_keys = load_smoke_keys(out_dir)
        full_summaries = submit_and_process_full(
            args=args,
            client=client,
            static_prefix=static_prefix,
            cache_name=cache_name,
            smoke_keys=smoke_keys,
        )
        aggregate = {
            "model": args.model,
            "source_root": str(source_root),
            "first_pass_prompt": str(ORIGINAL_PROMPT_PATH),
            "schema": "T=e,s,sg,st,p,o,og,ot,d,dg,l,lg,q,qg",
            "cache_name": cache_name,
            "full": full_summaries,
        }
        write_json(out_dir / "run_summary.json", aggregate)
        print(json.dumps(aggregate, ensure_ascii=False, indent=2))
        return

    cache_name, cache_record = create_cached_prefix(
        client,
        model=args.model,
        static_prefix=static_prefix,
        out_dir=out_dir,
        ttl_seconds=args.cache_ttl_seconds,
    )

    smoke_jobs = first_volume_pages(source_root, "konbaung_vol1", args.smoke_pages, args)
    save_batch_plan(out_dir, "smoke_vol1_first_pages", smoke_jobs)
    smoke_requests = [build_request(job, cache_name, args.max_output_tokens) for job in smoke_jobs]
    smoke_submitted = submit_batch(
        client,
        model=args.model,
        requests=smoke_requests,
        out_dir=out_dir,
        label="smoke_vol1_first_pages",
    )
    smoke_done = wait_for_batch(
        client,
        batch_name=smoke_submitted.name,
        out_dir=out_dir,
        label="smoke_vol1_first_pages",
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.smoke_max_wait_seconds,
    )
    smoke_summary = process_batch_outputs(
        batch_job=smoke_done,
        jobs_by_key={job_key(job): job for job in smoke_jobs},
        static_prefix=static_prefix,
        out_dir=out_dir,
        label="smoke_vol1_first_pages",
    )
    require_smoke_success(smoke_summary, len(smoke_jobs))

    if args.stop_after_smoke:
        aggregate = {
            "model": args.model,
            "source_root": str(source_root),
            "first_pass_prompt": str(ORIGINAL_PROMPT_PATH),
            "schema": "T=e,s,sg,st,p,o,og,ot,d,dg,l,lg,q,qg",
            "cache": cache_record,
            "smoke": smoke_summary,
            "stopped_after_smoke": True,
        }
        write_json(out_dir / "run_summary.json", aggregate)
        print(json.dumps(aggregate, ensure_ascii=False, indent=2))
        return

    smoke_keys = set(smoke_summary["completed_keys"])
    full_summaries = submit_and_process_full(
        args=args,
        client=client,
        static_prefix=static_prefix,
        cache_name=cache_name,
        smoke_keys=smoke_keys,
    )

    aggregate = {
        "model": args.model,
        "source_root": str(source_root),
        "first_pass_prompt": str(ORIGINAL_PROMPT_PATH),
        "schema": "T=e,s,sg,st,p,o,og,ot,d,dg,l,lg,q,qg",
        "cache": cache_record,
        "smoke": smoke_summary,
        "full": full_summaries,
    }
    write_json(out_dir / "run_summary.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cached Gemini Batch API annotation for all Konbaung volumes.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--coverage-file", default=str(DEFAULT_COVERAGE_FILE))
    parser.add_argument("--manifest-file", default=str(DEFAULT_MANIFEST_FILE))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--cache-ttl-seconds", type=int, default=86400)
    parser.add_argument("--smoke-pages", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--smoke-max-wait-seconds", type=int, default=3600)
    parser.add_argument("--full-max-wait-seconds", type=int, default=86400)
    parser.add_argument("--stop-after-smoke", action="store_true")
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--process-existing-full", action="store_true")
    parser.add_argument("--cache-name", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    run_smoke_and_full(parse_args(argv))


if __name__ == "__main__":
    main()

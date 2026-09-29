#!/usr/bin/env python3
"""Retry the 16 unresolved axial-coding pages with Gemini minimal thinking."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

import konbaung_gemini_flashlite_axial_full_corpus_batch as full


FULL_ROOT = full.DEFAULT_OUT_DIR
RETRY_ROOT = FULL_ROOT / "retry_minimal_20260729"
PAGE_KEYS = (
    "vol1-p0061",
    "vol1-p0167",
    "vol1-p0206",
    "vol1-p0208",
    "vol1-p0209",
    "vol2-p0024",
    "vol2-p0032",
    "vol2-p0086",
    "vol2-p0165",
    "vol2-p0170",
    "vol2-p0178",
    "vol2-p0217",
    "vol2-p0426",
    "vol3-p0186",
    "vol3-p0208",
    "vol3-p0479",
)
MAX_OUTPUT_TOKENS = 15000
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
}


def retry_jobs() -> list[full.PageJob]:
    by_key = {job.key: job for job in full.load_jobs()}
    missing = [key for key in PAGE_KEYS if key not in by_key]
    if missing:
        raise ValueError(f"Retry pages are absent from the canonical corpus: {missing}")
    jobs = [by_key[key] for key in PAGE_KEYS]
    if sum(len(job.triples) for job in jobs) != 992:
        raise ValueError("Retry triple count differs from the audited 992")
    return jobs


def client_and_cache() -> tuple[genai.Client, str, dict[str, Any]]:
    key = full.api_key()
    if not key:
        raise RuntimeError("No Gemini API key found")
    client = genai.Client(api_key=key)
    cache_record = full.read_json(FULL_ROOT / "cache_record.json")
    cache = client.caches.get(name=cache_record["name"])
    return client, cache.name, cache_record


def build_request(
    job: full.PageJob,
    cache_name: str,
    template: str,
    schema: dict[str, Any],
) -> types.InlinedRequest:
    return types.InlinedRequest(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=full.dynamic_text(template, job))],
            )
        ],
        metadata={
            "key": job.key,
            "volume": str(job.volume),
            "page": f"{job.page:04d}",
            "triple_count": str(len(job.triples)),
        },
        config=types.GenerateContentConfig(
            cached_content=cache_name,
            response_mime_type="application/json",
            response_json_schema=full.response_schema(schema, len(job.triples)),
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
            ),
        ),
    )


def submit() -> dict[str, Any]:
    RETRY_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = retry_jobs()
    _, template, schema = full.load_prompt_package()
    client, cache_name, cache_record = client_and_cache()
    submitted_path = RETRY_ROOT / "batch_submitted.json"
    if submitted_path.exists():
        batch = client.batches.get(name=full.read_json(submitted_path)["name"])
    else:
        batch = client.batches.create(
            model=full.MODEL,
            src=[build_request(job, cache_name, template, schema) for job in jobs],
            config=types.CreateBatchJobConfig(
                display_name=f"konbaung_axial_retry_minimal_{full.utc_stamp()}"
            ),
        )
        full.write_json(submitted_path, full.json_safe(batch))
    manifest = {
        "status": "submitted",
        "submitted_at": full.utc_now(),
        "model": full.MODEL,
        "thinking_level": "minimal",
        "temperature": 0.0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "cached_prefix": cache_record,
        "page_count": len(jobs),
        "triple_count": sum(len(job.triples) for job in jobs),
        "pages": [
            {
                "key": job.key,
                "sentence_count": len(job.sentences),
                "triple_count": len(job.triples),
            }
            for job in jobs
        ],
        "batch": {
            "name": batch.name,
            "state": full.enum_name(batch.state),
        },
    }
    full.write_json(RETRY_ROOT / "manifest.json", manifest)
    return manifest


def status() -> dict[str, Any]:
    client, _, _ = client_and_cache()
    submitted = full.read_json(RETRY_ROOT / "batch_submitted.json")
    batch = client.batches.get(name=submitted["name"])
    record = {
        "checked_at": full.utc_now(),
        "name": batch.name,
        "state": full.enum_name(batch.state),
        "completion_stats": full.json_safe(getattr(batch, "completion_stats", None)),
    }
    full.write_json(RETRY_ROOT / "batch_latest.json", full.json_safe(batch))
    full.write_json(RETRY_ROOT / "status.json", record)
    return record


def process_batch(batch: Any, jobs: list[full.PageJob]) -> dict[str, Any]:
    if full.enum_name(batch.state) != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"{batch.name} ended in {full.enum_name(batch.state)}")
    by_key = {job.key: job for job in jobs}
    responses = getattr(getattr(batch, "dest", None), "inlined_responses", None) or []
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for inlined in responses:
        metadata = getattr(inlined, "metadata", None) or {}
        job = by_key.get(metadata.get("key", ""))
        if job is None:
            rows.append({"key": metadata.get("key"), "category": "unknown_response"})
            totals["unknown_response"] += 1
            continue
        seen.add(job.key)
        if getattr(inlined, "error", None) is not None:
            attempt = {
                "key": job.key,
                "category": "transport_error",
                "error": full.json_safe(inlined.error),
            }
        elif getattr(inlined, "response", None) is None:
            attempt = {
                "key": job.key,
                "category": "transport_error",
                "error": "missing response",
            }
        else:
            response = inlined.response
            full.write_json(
                RETRY_ROOT / "raw_responses" / f"{job.key}.json",
                full.json_safe(response),
            )
            attempt = full.classify_response(
                job,
                response,
                MAX_OUTPUT_TOKENS,
            )
        totals[attempt["category"]] += 1
        for usage_key in full.USAGE_KEYS:
            value = attempt.get("usage", {}).get(usage_key)
            if isinstance(value, int):
                totals[usage_key] += value
        full.write_json(RETRY_ROOT / "attempts" / f"{job.key}.json", attempt)
        rows.append(
            {
                "key": job.key,
                "category": attempt["category"],
                "finish_reasons": attempt.get("finish_reasons", []),
                "usage": attempt.get("usage", {}),
                "errors": attempt.get("errors", attempt.get("error")),
            }
        )
    for job in jobs:
        if job.key in seen:
            continue
        attempt = {
            "key": job.key,
            "category": "transport_error",
            "error": "missing inlined batch response",
        }
        full.write_json(RETRY_ROOT / "attempts" / f"{job.key}.json", attempt)
        totals["transport_error"] += 1
        rows.append(attempt)
    return {
        "batch_name": batch.name,
        "state": full.enum_name(batch.state),
        "thinking_level": "minimal",
        "completion_stats": full.json_safe(getattr(batch, "completion_stats", None)),
        "totals": dict(totals),
        "results": rows,
    }


def backup_file(path: Path) -> None:
    if not path.exists():
        return
    relative = path.relative_to(FULL_ROOT)
    target = RETRY_ROOT / "premerge_backup" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(path, target)


def merge(summary: dict[str, Any], jobs: list[full.PageJob]) -> dict[str, Any]:
    accepted = {row["key"] for row in summary["results"] if row.get("category") == "accepted"}
    for aggregate in (
        FULL_ROOT / "final_report.json",
        FULL_ROOT / "run_summary.json",
        FULL_ROOT / "final" / "all_pages.jsonl",
        FULL_ROOT / "final" / "all_tagged_triples.jsonl",
        FULL_ROOT / "final" / "all_provisional_categories.jsonl",
        FULL_ROOT / "final" / "truncated_pages.json",
        FULL_ROOT / "final" / "unresolved_pages.json",
    ):
        backup_file(aggregate)

    by_key = {job.key: job for job in jobs}
    for key in accepted:
        job = by_key[key]
        attempt = full.read_json(RETRY_ROOT / "attempts" / f"{key}.json")
        for original in (
            FULL_ROOT / "attempts" / f"vol{job.volume}" / f"{key}.json",
            FULL_ROOT / "raw_responses" / f"vol{job.volume}" / f"{key}.json",
            full.page_output_dir(FULL_ROOT, job) / "review.md",
        ):
            backup_file(original)
        shutil.copy2(
            RETRY_ROOT / "attempts" / f"{key}.json",
            FULL_ROOT / "attempts" / f"vol{job.volume}" / f"{key}.json",
        )
        shutil.copy2(
            RETRY_ROOT / "raw_responses" / f"{key}.json",
            FULL_ROOT / "raw_responses" / f"vol{job.volume}" / f"{key}.json",
        )
        target = full.page_output_dir(FULL_ROOT, job)
        full.write_json(target / "page_data.json", job.payload)
        full.write_json(target / "result.json", attempt["result"])
        full.write_json(
            target / "run_metadata.json",
            {
                "status": "completed_valid",
                "source": "gemini_batch_minimal_retry",
                "model": full.MODEL,
                "thinking_level": "minimal",
                "temperature": 0.0,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "batch_name": summary["batch_name"],
                "finish_reasons": attempt.get("finish_reasons", []),
                "usage": attempt.get("usage", {}),
                "validation": {"accepted": True, "errors": []},
            },
        )

    all_jobs = full.load_jobs()
    prefix, _, _ = full.load_prompt_package()
    original_summaries = [
        full.read_json(FULL_ROOT / "batch_results" / f"axial_vol{volume}.json")
        for volume in (1, 2, 3)
    ]
    report = full.finalize(
        FULL_ROOT,
        all_jobs,
        [*original_summaries, summary],
        prefix,
    )
    remaining = [row for row in summary["results"] if row.get("category") != "accepted"]
    report["retry_policy_applied"] = "one 16-page minimal-thinking batch"
    report["minimal_retry"] = {
        "batch_name": summary["batch_name"],
        "submitted_pages": len(jobs),
        "submitted_triples": sum(len(job.triples) for job in jobs),
        "accepted_pages": len(accepted),
        "accepted_triples": sum(len(by_key[key].triples) for key in accepted),
        "remaining_pages": remaining,
        "usage": {key: summary["totals"].get(key, 0) for key in full.USAGE_KEYS},
    }
    full.write_json(FULL_ROOT / "final_report.json", report)
    full.write_json(
        FULL_ROOT / "run_summary.json",
        {"batches": [*original_summaries, summary], "final": report},
    )
    combined_review = RETRY_ROOT / "combined_review.md"
    combined_parts = [
        "# Konbaung axial-coding minimal-thinking retry review",
        "",
        f"- Submitted pages: **{len(jobs)}**",
        f"- Submitted triples: **{sum(len(job.triples) for job in jobs)}**",
        f"- Accepted pages: **{len(accepted)}**",
        f"- Accepted triples: **{sum(len(by_key[key].triples) for key in accepted)}**",
        f"- Remaining unresolved pages: **{len(remaining)}**",
        "",
    ]
    for job in jobs:
        combined_parts.extend(
            [
                "---",
                "",
                (full.page_output_dir(FULL_ROOT, job) / "review.md")
                .read_text(encoding="utf-8")
                .rstrip(),
                "",
            ]
        )
    full.write_text(combined_review, "\n".join(combined_parts).rstrip() + "\n")
    merge_record = {
        "merged_at": full.utc_now(),
        "accepted_pages": sorted(accepted),
        "remaining_pages": remaining,
        "final_status": report["status"],
        "tagged_triples": report["tagged_triples"],
        "combined_review": str(combined_review),
    }
    full.write_json(RETRY_ROOT / "merge_record.json", merge_record)
    return merge_record


def collect() -> dict[str, Any]:
    client, _, _ = client_and_cache()
    submitted = full.read_json(RETRY_ROOT / "batch_submitted.json")
    batch = client.batches.get(name=submitted["name"])
    state = full.enum_name(batch.state)
    if state not in TERMINAL_STATES:
        return {"status": "not_ready", "batch": status()}
    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Retry batch ended in {state}")
    summary_path = RETRY_ROOT / "batch_result.json"
    if summary_path.exists():
        summary = full.read_json(summary_path)
    else:
        summary = process_batch(batch, retry_jobs())
        full.write_json(summary_path, summary)
    return {
        "status": "collected",
        "summary": summary,
        "merge": merge(summary, retry_jobs()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("submit", "status", "collect"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "submit":
        result = submit()
    elif args.command == "status":
        result = status()
    else:
        result = collect()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

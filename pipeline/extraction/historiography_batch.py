#!/usr/bin/env python3
"""Run the ungrounded historiography triple prompt over all three reader volumes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import ValidationError

from pipeline.extraction.historiography import (
    DATA_ROOT,
    MODEL,
    TrialResult,
    historiography_prompt,
    sentence_pairs,
    validate,
)
from pipeline.extraction.summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "konbaung_historiography_ungrounded_v3_full_batch_20260723"
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}
USAGE_KEYS = (
    "prompt_token_count",
    "cached_content_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "total_token_count",
)
PRODUCTION_ADDITION = """

For relations with three semantic participants—especially titles, offices, fiefs,
appanages, named rewards, and other grants—orient the triple so the specific
recipient and the specific title, office, fief, appanage, reward, or granted
status are both retained. Prefer forms such as:

- `recipient — RECEIVED_TITLE_FROM_KING — specific title`
- `recipient — APPOINTED_BY_KING_AS — specific office`
- `recipient — RECEIVED_FIEF_FROM_KING — specific fief`

Do not return `king — BESTOWED_TITLE — recipient` when that would discard the
actual title, office, or grant named in the sentence.
""".strip()


@dataclass(frozen=True)
class PageJob:
    volume: int
    page: int
    source_path: Path
    summary: str
    pairs: list[dict[str, str]]
    source_sha256: str

    @property
    def key(self) -> str:
        return f"vol{self.volume}-p{self.page:04d}"

    @property
    def input_payload(self) -> dict[str, Any]:
        return {
            "page_summary": self.summary,
            "sentences": self.pairs,
        }


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def static_prefix() -> str:
    return (
        historiography_prompt().rstrip()
        + "\n\n"
        + PRODUCTION_ADDITION
        + "\n\nReturn only the requested JSON through the supplied schema."
    )


def dynamic_input(job: PageJob) -> str:
    return "SUPPLIED PAGE:\n" + json.dumps(job.input_payload, ensure_ascii=False, indent=2) + "\n"


def discover_jobs() -> list[PageJob]:
    jobs: list[PageJob] = []
    for volume in (1, 2, 3):
        volume_dir = DATA_ROOT / "pages" / f"vol{volume}"
        for path in sorted(volume_dir.glob("*.json")):
            raw_bytes = path.read_bytes()
            page_data = json.loads(raw_bytes.decode("utf-8"))
            pairs = sentence_pairs(page_data)
            summary = str(page_data.get("summary", "")).strip()
            if not summary:
                raise ValueError(f"{path}: missing page summary")
            page = int(path.stem)
            jobs.append(
                PageJob(
                    volume=volume,
                    page=page,
                    source_path=path,
                    summary=summary,
                    pairs=pairs,
                    source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                )
            )
    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate page keys in reader corpus")
    return jobs


def page_dir(out_dir: Path, job: PageJob) -> Path:
    return out_dir / "pages" / f"vol{job.volume}" / f"page_{job.page:04d}"


def attempt_path(out_dir: Path, job: PageJob, label: str) -> Path:
    return page_dir(out_dir, job) / "attempts" / f"{label}.json"


def accepted_path(out_dir: Path, job: PageJob) -> Path:
    return page_dir(out_dir, job) / "accepted.json"


def state_name(state: Any) -> str:
    return getattr(state, "value", str(state or ""))


def finish_reason_name(reason: Any) -> str:
    return getattr(reason, "value", str(reason or ""))


def usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    return json_safe(usage) if usage is not None else {}


def response_parts(response: Any) -> tuple[str, str]:
    answer_parts: list[str] = []
    thought_parts: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None) or ""
            if not text:
                continue
            if getattr(part, "thought", False):
                thought_parts.append(text)
            else:
                answer_parts.append(text)
    if not answer_parts and getattr(response, "text", None):
        answer_parts.append(response.text)
    return "".join(answer_parts).strip(), "\n\n".join(thought_parts).strip()


def response_finish_reasons(response: Any) -> list[str]:
    return [
        finish_reason_name(getattr(candidate, "finish_reason", None))
        for candidate in getattr(response, "candidates", None) or []
    ]


def thinking_level(level: str) -> types.ThinkingLevel:
    return {
        "high": types.ThinkingLevel.HIGH,
        "medium": types.ThinkingLevel.MEDIUM,
        "minimal": types.ThinkingLevel.MINIMAL,
    }[level]


def build_request(
    job: PageJob,
    cache_name: str,
    level: str,
    max_output_tokens: int,
) -> types.InlinedRequest:
    return types.InlinedRequest(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=dynamic_input(job))],
            )
        ],
        metadata={
            "key": job.key,
            "volume": str(job.volume),
            "page": f"{job.page:04d}",
        },
        config=types.GenerateContentConfig(
            cached_content=cache_name,
            response_mime_type="application/json",
            response_schema=TrialResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_level=thinking_level(level),
                include_thoughts=True,
            ),
        ),
    )


def token_count_total(response: Any) -> int | None:
    value = getattr(response, "total_tokens", None)
    if value is not None:
        return int(value)
    dumped = json_safe(response)
    if isinstance(dumped, dict):
        for key in ("total_tokens", "totalTokens"):
            if dumped.get(key) is not None:
                return int(dumped[key])
    return None


def create_cache(
    client: genai.Client,
    out_dir: Path,
    prefix: str,
    ttl_seconds: int,
) -> tuple[str, dict[str, Any]]:
    record_path = out_dir / "cache_record.json"
    if record_path.exists():
        record = load_json(record_path)
        cache_name = str(record["name"])
        client.caches.get(name=cache_name)
        return cache_name, record
    count = client.models.count_tokens(model=MODEL, contents=prefix)
    cache = client.caches.create(
        model=MODEL,
        config=types.CreateCachedContentConfig(
            display_name=f"konbaung_historiography_v3_{utc_stamp()}",
            contents=prefix,
            ttl=f"{ttl_seconds}s",
        ),
    )
    record = {
        "name": cache.name,
        "model": MODEL,
        "prefix_tokens": token_count_total(count),
        "prefix_chars": len(prefix),
        "ttl_seconds": ttl_seconds,
        "created": json_safe(cache),
    }
    write_text(out_dir / "cached_prefix.txt", prefix)
    write_json(record_path, record)
    return cache.name, record


def save_manifest(out_dir: Path, jobs: list[PageJob], args: argparse.Namespace) -> None:
    by_volume = Counter(job.volume for job in jobs)
    manifest = {
        "model": MODEL,
        "strategy": ["high", "medium_for_high_truncations", "minimal_for_medium_truncations"],
        "max_output_tokens": args.max_output_tokens,
        "cache_ttl_seconds": args.cache_ttl_seconds,
        "page_count": len(jobs),
        "sentence_count": sum(len(job.pairs) for job in jobs),
        "pages_by_volume": dict(by_volume),
        "existing_databases_modified": False,
        "pages": [
            {
                "key": job.key,
                "volume": job.volume,
                "page": job.page,
                "source_path": str(job.source_path),
                "source_sha256": job.source_sha256,
                "sentence_count": len(job.pairs),
            }
            for job in jobs
        ],
    }
    write_json(out_dir / "source_manifest.json", manifest)
    write_text(out_dir / "production_prompt_prefix.txt", static_prefix())


def save_plan(
    out_dir: Path,
    label: str,
    level: str,
    jobs: list[PageJob],
) -> None:
    write_json(
        out_dir / "batch_plans" / f"{label}.json",
        {
            "label": label,
            "thinking_level": level,
            "page_count": len(jobs),
            "sentence_count": sum(len(job.pairs) for job in jobs),
            "keys": [job.key for job in jobs],
        },
    )


def submit_batch(
    client: genai.Client,
    out_dir: Path,
    label: str,
    level: str,
    jobs: list[PageJob],
    cache_name: str,
    max_output_tokens: int,
) -> Any:
    submitted_path = out_dir / "batch_jobs" / f"{label}_submitted.json"
    if submitted_path.exists():
        return client.batches.get(name=load_json(submitted_path)["name"])
    save_plan(out_dir, label, level, jobs)
    requests = [build_request(job, cache_name, level, max_output_tokens) for job in jobs]
    batch = client.batches.create(
        model=MODEL,
        src=requests,
        config=types.CreateBatchJobConfig(display_name=f"konbaung_v3_{label}_{utc_stamp()}"),
    )
    write_json(submitted_path, json_safe(batch))
    return batch


def wait_for_batch(
    client: genai.Client,
    out_dir: Path,
    label: str,
    batch_name: str,
    poll_seconds: int,
    max_wait_seconds: int,
) -> Any:
    started = time.time()
    while True:
        batch = client.batches.get(name=batch_name)
        write_json(out_dir / "batch_jobs" / f"{label}_latest.json", json_safe(batch))
        state = state_name(batch.state)
        elapsed = int(time.time() - started)
        print(
            json.dumps({"label": label, "state": state, "elapsed_seconds": elapsed}),
            flush=True,
        )
        if state in TERMINAL_STATES:
            return batch
        if elapsed >= max_wait_seconds:
            raise TimeoutError(f"{label} exceeded {max_wait_seconds} seconds")
        time.sleep(poll_seconds)


def classify_response(
    job: PageJob,
    response: Any,
    max_output_tokens: int,
) -> dict[str, Any]:
    raw, thoughts = response_parts(response)
    usage = usage_dict(response)
    finish_reasons = response_finish_reasons(response)
    max_tokens_finish = any("MAX_TOKENS" in reason for reason in finish_reasons)
    generation_tokens = int(usage.get("thoughts_token_count") or 0) + int(
        usage.get("candidates_token_count") or 0
    )
    near_cap = generation_tokens >= max_output_tokens - 64
    base: dict[str, Any] = {
        "key": job.key,
        "raw_response": raw,
        "thought_summary": thoughts,
        "finish_reasons": finish_reasons,
        "usage": usage,
        "generation_tokens": generation_tokens,
        "max_output_tokens": max_output_tokens,
    }
    try:
        result = TrialResult.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        base.update(
            {
                "category": "truncated" if max_tokens_finish or near_cap else "schema_invalid",
                "error": f"schema parse failed: {exc}",
            }
        )
        return base
    validation = validate(job.pairs, result)
    base["response"] = result.model_dump(mode="json")
    base["validation"] = validation
    if validation["accepted"]:
        base["category"] = "accepted"
    elif max_tokens_finish or near_cap:
        base["category"] = "truncated"
    else:
        base["category"] = "validation_invalid"
    return base


def save_accepted(
    out_dir: Path,
    job: PageJob,
    label: str,
    level: str,
    attempt: dict[str, Any],
) -> None:
    target = page_dir(out_dir, job)
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "input.json", job.input_payload)
    write_json(
        accepted_path(out_dir, job),
        {
            "key": job.key,
            "volume": job.volume,
            "page": job.page,
            "source_path": str(job.source_path),
            "source_sha256": job.source_sha256,
            "accepted_from": label,
            "thinking_level": level,
            "usage": attempt["usage"],
            "validation": attempt["validation"],
            "response": attempt["response"],
        },
    )


def process_batch(
    out_dir: Path,
    label: str,
    level: str,
    batch: Any,
    jobs: list[PageJob],
    max_output_tokens: int,
) -> dict[str, Any]:
    summary_path = out_dir / "batch_results" / f"{label}.json"
    if summary_path.exists():
        return load_json(summary_path)
    if state_name(batch.state) != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch {label} ended in {state_name(batch.state)}")
    jobs_by_key = {job.key: job for job in jobs}
    responses = getattr(getattr(batch, "dest", None), "inlined_responses", None) or []
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    totals = Counter()
    for inlined in responses:
        metadata = getattr(inlined, "metadata", None) or {}
        key = metadata.get("key")
        job = jobs_by_key.get(key or "")
        if job is None:
            totals["unknown_response"] += 1
            rows.append({"key": key, "category": "unknown_response"})
            continue
        seen.add(job.key)
        if getattr(inlined, "error", None) is not None:
            attempt = {
                "key": job.key,
                "category": "transport_error",
                "error": json_safe(inlined.error),
            }
        elif getattr(inlined, "response", None) is None:
            attempt = {
                "key": job.key,
                "category": "transport_error",
                "error": "missing response",
            }
        else:
            attempt = classify_response(job, inlined.response, max_output_tokens)
        attempt.update(
            {
                "batch_label": label,
                "batch_name": batch.name,
                "thinking_level": level,
            }
        )
        write_json(attempt_path(out_dir, job, label), attempt)
        totals[attempt["category"]] += 1
        usage = attempt.get("usage", {})
        for usage_key in USAGE_KEYS:
            if isinstance(usage.get(usage_key), int):
                totals[usage_key] += usage[usage_key]
        if attempt["category"] == "accepted":
            save_accepted(out_dir, job, label, level, attempt)
        rows.append(
            {
                "key": job.key,
                "category": attempt["category"],
                "usage": usage,
                "finish_reasons": attempt.get("finish_reasons", []),
                "error": attempt.get("error"),
            }
        )
    for job in jobs:
        if job.key in seen:
            continue
        attempt = {
            "key": job.key,
            "category": "transport_error",
            "error": "missing inlined batch response",
            "batch_label": label,
            "batch_name": batch.name,
            "thinking_level": level,
        }
        write_json(attempt_path(out_dir, job, label), attempt)
        totals["transport_error"] += 1
        rows.append(
            {
                "key": job.key,
                "category": "transport_error",
                "error": attempt["error"],
            }
        )
    summary = {
        "label": label,
        "batch_name": batch.name,
        "state": state_name(batch.state),
        "thinking_level": level,
        "expected_pages": len(jobs),
        "completion_stats": json_safe(getattr(batch, "completion_stats", None)),
        "totals": dict(totals),
        "results": rows,
    }
    write_json(summary_path, summary)
    return summary


def run_batch_group(
    client: genai.Client,
    out_dir: Path,
    label: str,
    level: str,
    jobs: list[PageJob],
    cache_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not jobs:
        return {
            "label": label,
            "thinking_level": level,
            "expected_pages": 0,
            "totals": {},
            "results": [],
        }
    batch = submit_batch(
        client,
        out_dir,
        label,
        level,
        jobs,
        cache_name,
        args.max_output_tokens,
    )
    done = wait_for_batch(
        client,
        out_dir,
        label,
        batch.name,
        args.poll_seconds,
        args.max_wait_seconds,
    )
    return process_batch(
        out_dir,
        label,
        level,
        done,
        jobs,
        args.max_output_tokens,
    )


def jobs_for_categories(
    jobs_by_key: dict[str, PageJob],
    summaries: list[dict[str, Any]],
    categories: set[str],
) -> list[PageJob]:
    keys = {
        row["key"]
        for summary in summaries
        for row in summary.get("results", [])
        if row.get("category") in categories and row.get("key") in jobs_by_key
    }
    return [jobs_by_key[key] for key in sorted(keys)]


def accepted_keys(out_dir: Path, jobs: list[PageJob]) -> set[str]:
    return {job.key for job in jobs if accepted_path(out_dir, job).exists()}


def retry_transport_errors(
    client: genai.Client,
    out_dir: Path,
    base_label: str,
    level: str,
    source_summary: dict[str, Any],
    jobs_by_key: dict[str, PageJob],
    cache_name: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    retry_jobs = jobs_for_categories(jobs_by_key, [source_summary], {"transport_error"})
    if not retry_jobs:
        return None
    return run_batch_group(
        client,
        out_dir,
        f"{base_label}_transport_retry",
        level,
        retry_jobs,
        cache_name,
        args,
    )


def finalize(
    out_dir: Path,
    jobs: list[PageJob],
    all_summaries: list[dict[str, Any]],
    cache_record: dict[str, Any],
) -> dict[str, Any]:
    page_rows: list[dict[str, Any]] = []
    triple_rows: list[dict[str, Any]] = []
    unresolved: list[str] = []
    accepted_by_level = Counter()
    usage_totals = Counter()
    for summary in all_summaries:
        totals = summary.get("totals", {})
        for key in USAGE_KEYS:
            if isinstance(totals.get(key), int):
                usage_totals[key] += totals[key]
    for job in jobs:
        path = accepted_path(out_dir, job)
        if not path.exists():
            unresolved.append(job.key)
            continue
        accepted = load_json(path)
        accepted_by_level[accepted["thinking_level"]] += 1
        page_rows.append(
            {
                "key": job.key,
                "volume": job.volume,
                "page": job.page,
                "summary": job.summary,
                "sentences": job.pairs,
                "thinking_level": accepted["thinking_level"],
                "usage": accepted["usage"],
                "response": accepted["response"],
            }
        )
        source_by_sid = {pair["sid"]: pair for pair in job.pairs}
        for sentence in accepted["response"]["sentences"]:
            source = source_by_sid[sentence["sid"]]
            for ordinal, triple in enumerate(sentence["triples"], start=1):
                triple_rows.append(
                    {
                        "key": job.key,
                        "volume": job.volume,
                        "page": job.page,
                        "sid": sentence["sid"],
                        "sentence_my": source["my"],
                        "sentence_en": source["en"],
                        "ordinal": ordinal,
                        **triple,
                    }
                )
    page_rows.sort(key=lambda row: (row["volume"], row["page"]))
    triple_rows.sort(key=lambda row: (row["volume"], row["page"], row["sid"], row["ordinal"]))
    write_text(
        out_dir / "final" / "all_pages.jsonl",
        "\n".join(json.dumps(row, ensure_ascii=False) for row in page_rows) + "\n",
    )
    write_text(
        out_dir / "final" / "all_triples.jsonl",
        "\n".join(json.dumps(row, ensure_ascii=False) for row in triple_rows) + "\n",
    )
    batch_input_tokens = int(usage_totals.get("prompt_token_count", 0))
    cached_tokens = int(usage_totals.get("cached_content_token_count", 0))
    output_tokens = int(usage_totals.get("candidates_token_count", 0)) + int(
        usage_totals.get("thoughts_token_count", 0)
    )
    noncached_tokens = max(0, batch_input_tokens - cached_tokens)
    estimated_batch_usd = (
        noncached_tokens * 0.125 + cached_tokens * 0.0125 + output_tokens * 0.75
    ) / 1_000_000
    category_totals = Counter()
    for summary in all_summaries:
        for row in summary.get("results", []):
            category_totals[row.get("category", "unknown")] += 1
    report = {
        "model": MODEL,
        "expected_pages": len(jobs),
        "accepted_pages": len(page_rows),
        "unresolved_pages": unresolved,
        "accepted_by_thinking_level": dict(accepted_by_level),
        "triples": len(triple_rows),
        "usage_totals_all_attempts": dict(usage_totals),
        "attempt_category_totals": dict(category_totals),
        "estimated_batch_api_cost_usd_excluding_cache_storage": round(estimated_batch_usd, 6),
        "cache": cache_record,
        "existing_databases_modified": False,
    }
    write_json(out_dir / "final_report.json", report)
    return report


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = discover_jobs()
    jobs_by_key = {job.key: job for job in jobs}
    save_manifest(out_dir, jobs, args)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "prepared": True,
                    "out_dir": str(out_dir),
                    "pages": len(jobs),
                    "sentences": sum(len(job.pairs) for job in jobs),
                },
                indent=2,
            )
        )
        return

    client = genai.Client(api_key=api_key())
    prefix = static_prefix()
    cache_name, cache_record = create_cache(client, out_dir, prefix, args.cache_ttl_seconds)
    if args.retry_validation_invalid_high:
        run_summary_path = out_dir / "run_summary.json"
        if not run_summary_path.exists():
            raise RuntimeError("Cannot retry validation failures before the full run")
        prior_run = load_json(run_summary_path)
        summaries = list(prior_run.get("batches", []))
        retry_jobs = [
            jobs_by_key[key]
            for key in prior_run.get("final", {}).get("unresolved_pages", [])
            if key in jobs_by_key
            and not accepted_path(out_dir, jobs_by_key[key]).exists()
            and load_json(
                attempt_path(
                    out_dir,
                    jobs_by_key[key],
                    f"high_vol{jobs_by_key[key].volume}",
                )
            ).get("category")
            == "validation_invalid"
        ]
        write_json(
            out_dir / "retry_sets" / "high_validation_retry_1.json",
            {"count": len(retry_jobs), "keys": [job.key for job in retry_jobs]},
        )
        retry = run_batch_group(
            client,
            out_dir,
            "high_validation_retry_1",
            "high",
            retry_jobs,
            cache_name,
            args,
        )
        summaries.append(retry)
        transport_retry = retry_transport_errors(
            client,
            out_dir,
            "high_validation_retry_1",
            "high",
            retry,
            jobs_by_key,
            cache_name,
            args,
        )
        if transport_retry:
            summaries.append(transport_retry)
        report = finalize(out_dir, jobs, summaries, cache_record)
        write_json(out_dir / "run_summary.json", {"batches": summaries, "final": report})
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
    summaries: list[dict[str, Any]] = []

    smoke_key = "vol2-p0391"
    smoke_job = jobs_by_key[smoke_key]
    smoke = run_batch_group(
        client,
        out_dir,
        "smoke_high",
        "high",
        [smoke_job],
        cache_name,
        args,
    )
    summaries.append(smoke)
    if smoke.get("totals", {}).get("accepted") != 1:
        raise RuntimeError(f"Smoke page was not accepted: {smoke}")
    if smoke.get("totals", {}).get("cached_content_token_count", 0) <= 0:
        raise RuntimeError(f"Smoke page did not use cached content: {smoke}")
    if args.stop_after_smoke:
        print(json.dumps({"cache": cache_record, "smoke": smoke}, indent=2))
        return

    high_batches: list[tuple[str, list[PageJob], Any]] = []
    already_accepted = accepted_keys(out_dir, jobs)
    for volume in (1, 2, 3):
        volume_jobs = [
            job for job in jobs if job.volume == volume and job.key not in already_accepted
        ]
        label = f"high_vol{volume}"
        batch = submit_batch(
            client,
            out_dir,
            label,
            "high",
            volume_jobs,
            cache_name,
            args.max_output_tokens,
        )
        high_batches.append((label, volume_jobs, batch))
    for label, volume_jobs, batch in high_batches:
        done = wait_for_batch(
            client,
            out_dir,
            label,
            batch.name,
            args.poll_seconds,
            args.max_wait_seconds,
        )
        summary = process_batch(
            out_dir,
            label,
            "high",
            done,
            volume_jobs,
            args.max_output_tokens,
        )
        summaries.append(summary)
        transport_retry = retry_transport_errors(
            client,
            out_dir,
            label,
            "high",
            summary,
            jobs_by_key,
            cache_name,
            args,
        )
        if transport_retry:
            summaries.append(transport_retry)

    high_truncated = jobs_for_categories(
        jobs_by_key,
        [summary for summary in summaries if summary.get("thinking_level") == "high"],
        {"truncated"},
    )
    high_truncated = [job for job in high_truncated if not accepted_path(out_dir, job).exists()]
    write_json(
        out_dir / "retry_sets" / "medium_from_high_truncations.json",
        {"count": len(high_truncated), "keys": [job.key for job in high_truncated]},
    )
    medium = run_batch_group(
        client,
        out_dir,
        "medium_truncation_retry",
        "medium",
        high_truncated,
        cache_name,
        args,
    )
    summaries.append(medium)
    medium_transport = retry_transport_errors(
        client,
        out_dir,
        "medium_truncation_retry",
        "medium",
        medium,
        jobs_by_key,
        cache_name,
        args,
    )
    if medium_transport:
        summaries.append(medium_transport)

    medium_truncated = jobs_for_categories(
        jobs_by_key,
        [summary for summary in summaries if summary.get("thinking_level") == "medium"],
        {"truncated"},
    )
    medium_truncated = [job for job in medium_truncated if not accepted_path(out_dir, job).exists()]
    write_json(
        out_dir / "retry_sets" / "minimal_from_medium_truncations.json",
        {
            "count": len(medium_truncated),
            "keys": [job.key for job in medium_truncated],
        },
    )
    minimal = run_batch_group(
        client,
        out_dir,
        "minimal_truncation_retry",
        "minimal",
        medium_truncated,
        cache_name,
        args,
    )
    summaries.append(minimal)
    minimal_transport = retry_transport_errors(
        client,
        out_dir,
        "minimal_truncation_retry",
        "minimal",
        minimal,
        jobs_by_key,
        cache_name,
        args,
    )
    if minimal_transport:
        summaries.append(minimal_transport)

    report = finalize(out_dir, jobs, summaries, cache_record)
    write_json(out_dir / "run_summary.json", {"batches": summaries, "final": report})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    parser.add_argument("--cache-ttl-seconds", type=int, default=604800)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=172800)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--stop-after-smoke", action="store_true")
    parser.add_argument("--retry-validation-invalid-high", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

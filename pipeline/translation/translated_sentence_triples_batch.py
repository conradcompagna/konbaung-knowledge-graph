#!/usr/bin/env python3
"""Run the final translated-sentence triple prompt over the full corpus."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import ValidationError

from konbaung_gemini_cite_sources_annotator import MODEL
from konbaung_gemini_summary_claim_completion_annotator import api_key
from konbaung_gemini_translated_sentence_triples_test import (
    PROMPT_PATH,
    READER_DATA_ROOT,
    TRANSLATION_ROOT,
    TripleResult,
    build_payload,
    full_prompt,
    review_text,
    validate,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    ROOT / "konbaung_translated_sentence_triples_full_batch_20260713_high_thinking"
)
TEST_OUTPUT_ROOT = ROOT / "konbaung_translated_sentence_triple_tests_unified"
REUSED_TESTS = (
    (1, 55, "high_thinking_all_relevant_01"),
    (1, 384, "high_thinking_all_relevant_01"),
    (2, 190, "high_thinking_all_relevant_01"),
    (2, 304, "high_thinking_all_relevant_01"),
    (3, 369, "high_thinking_all_relevant_01"),
)
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


@dataclass(frozen=True)
class PageJob:
    volume: int
    page: int
    payload: dict[str, Any]

    @property
    def key(self) -> str:
        return f"vol{self.volume}-p{self.page:04d}"


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


def page_dir(out_dir: Path, job: PageJob) -> Path:
    return out_dir / "pages" / f"vol{job.volume}" / f"page_{job.page:04d}"


def dynamic_input(payload: dict[str, Any]) -> str:
    return (
        "<ANNOTATION_INPUT>\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "</ANNOTATION_INPUT>\n"
    )


def discover_jobs() -> list[PageJob]:
    jobs: list[PageJob] = []
    for volume in (1, 2, 3):
        volume_dir = TRANSLATION_ROOT / f"vol{volume}"
        for source_dir in sorted(volume_dir.glob("page_*")):
            page = int(source_dir.name.removeprefix("page_"))
            jobs.append(PageJob(volume, page, build_payload(volume, page)))
    return jobs


def build_request(
    job: PageJob,
    cache_name: str,
    max_output_tokens: int,
    thinking_budget: int | None = None,
) -> types.InlinedRequest:
    thinking_config = (
        types.ThinkingConfig(thinking_budget=thinking_budget)
        if thinking_budget is not None
        else types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH)
    )
    return types.InlinedRequest(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=dynamic_input(job.payload))],
            )
        ],
        metadata={"key": job.key, "volume": str(job.volume), "page": f"{job.page:04d}"},
        config=types.GenerateContentConfig(
            cached_content=cache_name,
            response_mime_type="application/json",
            response_schema=TripleResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=max_output_tokens,
            thinking_config=thinking_config,
        ),
    )


def state_name(state: Any) -> str:
    return getattr(state, "value", str(state or ""))


def response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text
    parts: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None):
                parts.append(part.text)
    return "".join(parts)


def usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    return json_safe(usage) if usage is not None else {}


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


def create_cached_prefix(
    client: genai.Client,
    out_dir: Path,
    prefix: str,
    ttl_seconds: int,
) -> tuple[str, dict[str, Any]]:
    token_count = client.models.count_tokens(model=MODEL, contents=prefix)
    cache = client.caches.create(
        model=MODEL,
        config=types.CreateCachedContentConfig(
            display_name=f"konbaung_sentence_triples_{utc_stamp()}",
            contents=prefix,
            ttl=f"{ttl_seconds}s",
        ),
    )
    record = {
        "name": cache.name,
        "model": MODEL,
        "prompt_path": str(PROMPT_PATH),
        "prefix_tokens": token_count_total(token_count),
        "ttl_seconds": ttl_seconds,
        "created": json_safe(cache),
    }
    write_text(out_dir / "cached_prefix.txt", prefix)
    write_json(out_dir / "cache_record.json", record)
    return cache.name, record


def submit_batch(
    client: genai.Client,
    out_dir: Path,
    label: str,
    jobs: list[PageJob],
    cache_name: str,
    max_output_tokens: int,
    thinking_budget: int | None = None,
) -> Any:
    requests = [build_request(job, cache_name, max_output_tokens, thinking_budget) for job in jobs]
    batch = client.batches.create(
        model=MODEL,
        src=requests,
        config=types.CreateBatchJobConfig(display_name=f"konbaung_{label}_{utc_stamp()}"),
    )
    write_json(out_dir / "batch_jobs" / f"{label}_submitted.json", json_safe(batch))
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
        print(json.dumps({"label": label, "state": state, "elapsed_seconds": elapsed}), flush=True)
        if state in TERMINAL_STATES:
            return batch
        if elapsed >= max_wait_seconds:
            raise TimeoutError(f"{label} exceeded {max_wait_seconds} seconds")
        time.sleep(poll_seconds)


def save_page_result(
    out_dir: Path,
    job: PageJob,
    raw: str,
    result: TripleResult,
    errors: list[str],
    usage: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    target = page_dir(out_dir, job)
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "payload.json", job.payload)
    write_text(target / "prompt_sent.txt", full_prompt(job.payload))
    write_text(target / "raw_response.json", raw)
    write_json(
        target / "result.json",
        {
            "accepted": not errors,
            "errors": errors,
            "usage": usage,
            "response": result.model_dump(mode="json"),
        },
    )
    write_text(target / "review.md", review_text(job.payload, result, errors))
    write_json(target / "provenance.json", provenance)


def materialize_reused_tests(out_dir: Path, jobs_by_key: dict[str, PageJob]) -> set[str]:
    reused: set[str] = set()
    manifest: list[dict[str, Any]] = []
    for volume, page, output_name in REUSED_TESTS:
        key = f"vol{volume}-p{page:04d}"
        job = jobs_by_key[key]
        source = TEST_OUTPUT_ROOT / f"vol{volume}" / f"page_{page:04d}" / output_name
        stored_payload = load_json(source / "payload.json")
        stored_prompt = (source / "prompt_sent.txt").read_text(encoding="utf-8")
        stored_result = load_json(source / "result.json")
        result = TripleResult.model_validate(stored_result["response"])
        errors = validate(job.payload, result)
        checks = {
            "payload_exact": stored_payload == job.payload,
            "prompt_exact": stored_prompt == full_prompt(job.payload),
            "stored_accepted": stored_result.get("accepted") is True,
            "current_validation_errors": errors,
        }
        if (
            not all((checks["payload_exact"], checks["prompt_exact"], checks["stored_accepted"]))
            or errors
        ):
            raise RuntimeError(f"Reusable test failed current checks: {key}: {checks}")
        target = page_dir(out_dir, job)
        target.mkdir(parents=True, exist_ok=True)
        for name in (
            "payload.json",
            "prompt_sent.txt",
            "raw_response.json",
            "result.json",
            "review.md",
        ):
            shutil.copy2(source / name, target / name)
        provenance = {
            "source": "reused_final_high_thinking_test",
            "source_dir": str(source),
            "model": MODEL,
            "thinking_level": "HIGH",
            "checks": checks,
        }
        write_json(target / "provenance.json", provenance)
        reused.add(key)
        manifest.append({"key": key, "source_dir": str(source), "checks": checks})
    write_json(out_dir / "reused_test_pages.json", manifest)
    return reused


def process_batch(
    out_dir: Path,
    label: str,
    batch: Any,
    jobs_by_key: dict[str, PageJob],
    thinking_mode: str = "HIGH",
) -> tuple[dict[str, Any], set[str]]:
    if state_name(batch.state) != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch {label} ended in {state_name(batch.state)}")
    responses = getattr(getattr(batch, "dest", None), "inlined_responses", None) or []
    totals = Counter()
    completed: set[str] = set()
    rows: list[dict[str, Any]] = []
    for inlined in responses:
        totals["responses_seen"] += 1
        metadata = getattr(inlined, "metadata", None) or {}
        key = metadata.get("key")
        job = jobs_by_key.get(key or "")
        row: dict[str, Any] = {"key": key}
        if job is None:
            totals["unknown_responses"] += 1
            row["error"] = "unknown response key"
            rows.append(row)
            continue
        if getattr(inlined, "error", None) is not None:
            totals["response_errors"] += 1
            row["error"] = json_safe(inlined.error)
            rows.append(row)
            continue
        response = getattr(inlined, "response", None)
        if response is None:
            totals["response_errors"] += 1
            row["error"] = "missing response"
            rows.append(row)
            continue
        usage = usage_dict(response)
        for usage_key in USAGE_KEYS:
            if isinstance(usage.get(usage_key), int):
                totals[usage_key] += usage[usage_key]
        raw = response_text(response)
        try:
            result = TripleResult.model_validate_json(raw)
        except (ValidationError, ValueError) as exc:
            totals["schema_errors"] += 1
            row["error"] = f"schema parse failed: {exc}"
            row["raw"] = raw
            rows.append(row)
            write_json(out_dir / "batch_errors" / label / f"{job.key}.json", row)
            continue
        errors = validate(job.payload, result)
        save_page_result(
            out_dir,
            job,
            raw,
            result,
            errors,
            usage,
            {
                "source": "gemini_cached_batch",
                "batch_label": label,
                "batch_name": batch.name,
                "model": MODEL,
                "thinking": thinking_mode,
            },
        )
        row.update(
            {
                "accepted": not errors,
                "errors": errors,
                "sentence_count": len(result.S),
                "triple_count": sum(len(group.T) for group in result.S),
                "usage": usage,
            }
        )
        if errors:
            totals["validation_errors"] += 1
        else:
            totals["accepted_pages"] += 1
            completed.add(job.key)
        rows.append(row)
    expected = set(jobs_by_key)
    missing = sorted(expected - {row.get("key") for row in rows})
    totals["missing_responses"] = len(missing)
    summary = {
        "label": label,
        "batch_name": batch.name,
        "state": state_name(batch.state),
        "expected_pages": len(jobs_by_key),
        "totals": dict(totals),
        "completed_keys": sorted(completed),
        "missing_response_keys": missing,
        "results": rows,
    }
    write_json(out_dir / "batch_results" / f"{label}.json", summary)
    return summary, completed


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def finalize(out_dir: Path, jobs: list[PageJob]) -> dict[str, Any]:
    annotation_dir = out_dir / "annotations"
    triple_dir = out_dir / "triples"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    triple_dir.mkdir(parents=True, exist_ok=True)
    page_rows: list[dict[str, Any]] = []
    triple_rows: list[dict[str, Any]] = []
    endpoint_audit = Counter()
    provenance_counts = Counter()
    usage_totals = Counter()
    missing_pages: list[str] = []
    invalid_pages: list[str] = []

    for job in jobs:
        target = page_dir(out_dir, job)
        result_path = target / "result.json"
        if not result_path.exists():
            missing_pages.append(job.key)
            continue
        stored = load_json(result_path)
        if not stored.get("accepted"):
            invalid_pages.append(job.key)
            continue
        result = TripleResult.model_validate(stored["response"])
        provenance = load_json(target / "provenance.json")
        provenance_counts[provenance["source"]] += 1
        usage = stored.get("usage", {})
        for usage_key in USAGE_KEYS:
            if isinstance(usage.get(usage_key), int):
                usage_totals[usage_key] += usage[usage_key]
        sentence_lookup = {item["id"]: item for item in job.payload["sentence_pairs"]}
        page_text = load_json(
            READER_DATA_ROOT / "pages" / f"vol{job.volume}" / f"{job.page:04d}.json"
        )["canonicalText"]
        page_norm = normalize_space(page_text)
        page_rows.append(
            {
                "page_id": job.payload["page_id"],
                "volume": job.volume,
                "page": job.page,
                "summary": job.payload["summary"],
                "S": result.model_dump(mode="json")["S"],
                "provenance": provenance["source"],
                "usage": usage,
            }
        )
        for group in result.S:
            sentence = sentence_lookup[group.sid]
            sentence_norm = normalize_space(sentence["my"])
            for ordinal, triple in enumerate(group.T, 1):
                for role, endpoint in (("subject", triple.s), ("object", triple.o)):
                    endpoint_audit[f"{role}_{endpoint.source}_total"] += 1
                    endpoint_norm = normalize_space(endpoint.my)
                    haystack = sentence_norm if endpoint.source == "sentence" else page_norm
                    if endpoint.source == "inferred":
                        endpoint_audit[f"{role}_inferred"] += 1
                    elif endpoint_norm and endpoint_norm in haystack:
                        endpoint_audit[f"{role}_{endpoint.source}_resolved"] += 1
                    else:
                        endpoint_audit[f"{role}_{endpoint.source}_unresolved"] += 1
                triple_rows.append(
                    {
                        "page_id": job.payload["page_id"],
                        "volume": job.volume,
                        "page": job.page,
                        "sid": group.sid,
                        "sentence_my": sentence["my"],
                        "sentence_en": sentence["en"],
                        "ordinal": ordinal,
                        "s": triple.s.model_dump(mode="json"),
                        "p": triple.p,
                        "o": triple.o.model_dump(mode="json"),
                    }
                )

    page_rows.sort(key=lambda row: (row["volume"], row["page"]))
    triple_rows.sort(key=lambda row: (row["volume"], row["page"], row["sid"], row["ordinal"]))
    all_page_lines = [json.dumps(row, ensure_ascii=False) for row in page_rows]
    all_triple_lines = [json.dumps(row, ensure_ascii=False) for row in triple_rows]
    write_text(annotation_dir / "all_pages.jsonl", "\n".join(all_page_lines) + "\n")
    write_text(triple_dir / "all_triples.jsonl", "\n".join(all_triple_lines) + "\n")
    for volume in (1, 2, 3):
        write_text(
            annotation_dir / f"vol{volume}.jsonl",
            "\n".join(
                json.dumps(row, ensure_ascii=False) for row in page_rows if row["volume"] == volume
            )
            + "\n",
        )
        write_text(
            triple_dir / f"vol{volume}.jsonl",
            "\n".join(
                json.dumps(row, ensure_ascii=False)
                for row in triple_rows
                if row["volume"] == volume
            )
            + "\n",
        )
    audit = {
        "expected_owner_pages": len(jobs),
        "accepted_owner_pages": len(page_rows),
        "missing_pages": missing_pages,
        "invalid_pages": invalid_pages,
        "sentence_groups": sum(len(row["S"]) for row in page_rows),
        "triples": len(triple_rows),
        "pages_by_volume": dict(Counter(row["volume"] for row in page_rows)),
        "triples_by_volume": dict(Counter(row["volume"] for row in triple_rows)),
        "provenance": dict(provenance_counts),
        "usage": dict(usage_totals),
        "endpoint_audit_warnings_only": dict(endpoint_audit),
    }
    write_json(out_dir / "final_audit.json", audit)
    if missing_pages or invalid_pages or len(page_rows) != len(jobs):
        raise RuntimeError(f"Final corpus incomplete: {audit}")
    return audit


def save_plan(out_dir: Path, label: str, jobs: list[PageJob]) -> None:
    write_json(
        out_dir / "batch_plans" / f"{label}.json",
        {
            "label": label,
            "page_count": len(jobs),
            "sentence_count": sum(len(job.payload["sentence_pairs"]) for job in jobs),
            "pages": [
                {
                    "key": job.key,
                    "volume": job.volume,
                    "page": job.page,
                    "sentence_count": len(job.payload["sentence_pairs"]),
                }
                for job in jobs
            ],
        },
    )


def accepted_page_keys(out_dir: Path, jobs: list[PageJob]) -> set[str]:
    accepted: set[str] = set()
    for job in jobs:
        result_path = page_dir(out_dir, job) / "result.json"
        if result_path.exists() and load_json(result_path).get("accepted") is True:
            accepted.add(job.key)
    return accepted


def jobs_from_saved_plan(
    out_dir: Path, label: str, jobs_by_key: dict[str, PageJob]
) -> list[PageJob]:
    plan = load_json(out_dir / "batch_plans" / f"{label}.json")
    return [jobs_by_key[row["key"]] for row in plan["pages"]]


def submitted_batch_name(out_dir: Path, label: str) -> str:
    submitted = load_json(out_dir / "batch_jobs" / f"{label}_submitted.json")
    return submitted["name"]


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = discover_jobs()
    jobs_by_key = {job.key: job for job in jobs}
    if len(jobs_by_key) != len(jobs):
        raise RuntimeError("Duplicate owner-page jobs")
    reused = materialize_reused_tests(out_dir, jobs_by_key)
    prefix = PROMPT_PATH.read_text(encoding="utf-8-sig").rstrip()
    write_json(
        out_dir / "run_config.json",
        {
            "model": MODEL,
            "thinking_level": "HIGH",
            "max_output_tokens": args.max_output_tokens,
            "owner_pages": len(jobs),
            "sentence_groups": sum(len(job.payload["sentence_pairs"]) for job in jobs),
            "reused_test_pages": sorted(reused),
        },
    )
    client = genai.Client(api_key=api_key())
    if args.resume:
        cache_record = load_json(out_dir / "cache_record.json")
        cache_name = cache_record["name"]
        smoke_summary = load_json(out_dir / "batch_results" / "smoke.json")
        completed = accepted_page_keys(out_dir, jobs)
    else:
        cache_name, cache_record = create_cached_prefix(
            client, out_dir, prefix, args.cache_ttl_seconds
        )
        pending = [job for job in jobs if job.key not in reused]
        smoke_jobs = pending[:1]
        save_plan(out_dir, "smoke", smoke_jobs)
        smoke_batch = submit_batch(
            client, out_dir, "smoke", smoke_jobs, cache_name, args.max_output_tokens
        )
        smoke_done = wait_for_batch(
            client,
            out_dir,
            "smoke",
            smoke_batch.name,
            args.poll_seconds,
            args.max_wait_seconds,
        )
        smoke_summary, smoke_completed = process_batch(
            out_dir, "smoke", smoke_done, {job.key: job for job in smoke_jobs}
        )
        smoke_totals = smoke_summary["totals"]
        if smoke_completed != {smoke_jobs[0].key}:
            raise RuntimeError(f"Smoke validation failed: {smoke_summary}")
        if smoke_totals.get("cached_content_token_count", 0) <= 0:
            raise RuntimeError(f"Smoke did not report cached prefix use: {smoke_totals}")
        completed = set(reused) | smoke_completed

    full_summaries: list[dict[str, Any]] = []
    for volume in (1, 2, 3):
        label = f"vol{volume}"
        result_path = out_dir / "batch_results" / f"{label}.json"
        if result_path.exists():
            full_summaries.append(load_json(result_path))
            continue
        submitted_path = out_dir / "batch_jobs" / f"{label}_submitted.json"
        if submitted_path.exists():
            volume_jobs = jobs_from_saved_plan(out_dir, label, jobs_by_key)
            batch_name = submitted_batch_name(out_dir, label)
        else:
            volume_jobs = [job for job in jobs if job.volume == volume and job.key not in completed]
            if not volume_jobs:
                continue
            save_plan(out_dir, label, volume_jobs)
            batch = submit_batch(
                client, out_dir, label, volume_jobs, cache_name, args.max_output_tokens
            )
            batch_name = batch.name
        done = wait_for_batch(
            client,
            out_dir,
            label,
            batch_name,
            args.poll_seconds,
            args.max_wait_seconds,
        )
        summary, accepted = process_batch(
            out_dir, label, done, {job.key: job for job in volume_jobs}
        )
        full_summaries.append(summary)
        completed.update(accepted)

    for retry_round in range(1, args.retry_rounds + 1):
        failures = [job for job in jobs if job.key not in completed]
        if not failures:
            break
        label = f"retry_{retry_round}"
        retry_cap = args.retry_max_output_tokens + (retry_round - 1) * 4000
        submitted_path = out_dir / "batch_jobs" / f"{label}_submitted.json"
        result_path = out_dir / "batch_results" / f"{label}.json"
        if result_path.exists():
            summary = load_json(result_path)
            full_summaries.append(summary)
            completed.update(summary.get("completed_keys", []))
            continue
        if submitted_path.exists():
            retry_jobs = jobs_from_saved_plan(out_dir, label, jobs_by_key)
            batch_name = submitted_batch_name(out_dir, label)
        else:
            retry_jobs = failures
            save_plan(out_dir, label, retry_jobs)
            batch = submit_batch(
                client,
                out_dir,
                label,
                retry_jobs,
                cache_name,
                retry_cap,
                args.retry_thinking_budget,
            )
            batch_name = batch.name
        done = wait_for_batch(
            client,
            out_dir,
            label,
            batch_name,
            args.poll_seconds,
            args.max_wait_seconds,
        )
        summary, accepted = process_batch(
            out_dir,
            label,
            done,
            {job.key: job for job in retry_jobs},
            thinking_mode=f"budget={args.retry_thinking_budget}",
        )
        full_summaries.append(summary)
        completed.update(accepted)

    audit = finalize(out_dir, jobs)
    run_summary = {
        "model": MODEL,
        "cache": cache_record,
        "smoke": smoke_summary,
        "batches": [
            {
                "label": summary["label"],
                "batch_name": summary["batch_name"],
                "totals": summary["totals"],
            }
            for summary in full_summaries
        ],
        "audit": audit,
    }
    write_json(out_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-output-tokens", type=int, default=8000)
    parser.add_argument("--retry-max-output-tokens", type=int, default=16000)
    parser.add_argument("--retry-thinking-budget", type=int, default=6000)
    parser.add_argument("--cache-ttl-seconds", type=int, default=259200)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=172800)
    parser.add_argument("--retry-rounds", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
